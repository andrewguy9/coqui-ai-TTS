from collections.abc import Callable
import os
from pathlib import Path
import re
import sys
from typing import Dict, List

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager
from docopt import docopt

from audio_emotion import Emotion
from make_trainingset import conditingset_reader, get_primary_emotion
from path_utils import get_base_name, is_audio, walk_paths
from shutil import copyfile

import json
from validate_dataset import DatasetSampleDict, dataset_reader
import torch

# TODO retire
def model_run_prefix(model_name: str) -> str:
    return f"{model_name}"

def find_runs(run_name: str, run_dir: Path) -> List[Path]:
    print("FINDING RUNS FOR DATASET:", run_name, "IN DIRECTORY:", run_dir)
    glob_pattern = f"{run_name}*"
    print("GLOB PATTERN:", glob_pattern)
    runs = list(run_dir.glob(glob_pattern))
    return runs

def find_models(run_dir: Path):
    entries = run_dir.iterdir()
    dirs = filter(lambda p: p.is_dir(), entries)
    names = map(lambda p: p.name, dirs)
    no_prefixes = map(lambda s: s.removeprefix("naqqal_"), names)
    no_suffix = map(lambda s: re.sub(r"-.*$", "", s), no_prefixes)
    yield from no_suffix

def custom_formatter(root_path, meta_file, **kwargs):
    json_file = Path(root_path) / meta_file
    items = []
    with open(json_file, encoding="utf-8") as fh:
        samples = json.load(fh)
        for sample in samples:
            ds = DatasetSampleDict(
                identifier=str(sample['identifier']),
                unnormalized_text=sample['unnormalized_text'],
                normalized_text=sample['normalized_text'],
                duration=sample['duration'],
                emotions=sample['emotions'],
                speaker_name=sample['speaker_name'],
            )
            speaker_name = ds['speaker_name']
            wav_file = (Path(root_path) / "wavs" / f"{ds['identifier']}.wav")
            text = ds['normalized_text']
            emotion = get_primary_emotion(ds['emotions'])
            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker_name": speaker_name,
                "root_path": root_path,
                "emotion_name": emotion,
                })
    return items

def dataset_configuration(dataset_path: Path) -> BaseDatasetConfig:
    # Define here the dataset that you want to use for the fine-tuning on.
    dataset_name = get_base_name(dataset_path)

    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata file not found: {metadata_path}")

    config_dataset = BaseDatasetConfig(
        # TODO we need to replace with our json format.
        formatter="custom_formatter",
        dataset_name=dataset_name,
        path=str(dataset_path),
        meta_file_train=str(metadata_path.relative_to(dataset_path)),
        language="en", # TODO option
    )
    return config_dataset


def train_voice(run_name: str, datasets_config: List[BaseDatasetConfig], training_dir: Path, batch_size: int = 9):
    # setup variables
    # Logging parameters
    PROJECT_NAME = "naqqal"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Training Parameters
    # TODO setup for multi gpu as option.
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = False  # for multi-gpu training please make it False
    START_WITH_EVAL = True  # if True it will start with evaluation
    GRAD_ACUMM_STEPS = 84  # set here the grad accumulation steps
    # Note: we recommend that batch_size * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease batch_size but then set GRAD_ACUMM_STEPS accordingly.

    sample_count = sum(map(len, datasets_config))
    if sample_count == 0:
        raise ValueError(f"No samples found in the dataset. Please check the dataset path and metadata file.")
    eval_size_pct = (sample_count ** .5) / sample_count
    if eval_size_pct * sample_count < 2:
        raise ValueError(f"Dataset is too small ({sample_count}) for evaluation.")
    # print("DATASET CONFIGURATION:")
    # print(datasets_config)

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = str(os.path.join(str(training_dir), "XTTS_v2.0_original_model_files/"))
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
    MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 files!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    print("MODEL ARGS:")
    print(model_args)

    # define audio config
    # TODO do we need to reformat the input audio files?
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    print("AUDIO CONFIG:")
    print(audio_config)

    # training parameters config
    config = GPTTrainerConfig(
        output_path=str(training_dir),
        model_args=model_args,
        run_name=run_name,
        project_name=PROJECT_NAME,
        run_description=f"""
            GPT XTTS training on {run_name} voice.
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_group_size=48,
        num_loader_workers=8,
        eval_batch_size=batch_size,
        eval_split_max_size=None, # None is the default, allow the evaluation split to be as big as the training set.
        eval_split_size = eval_size_pct,
        print_eval=True,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        epochs=80,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
    )

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        datasets_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size, # This was set to None, meaning the eval split can be as big as the training set.
        eval_split_size=config.eval_split_size,
    )

    print("TRAINING SAMPLES:")
    print(f"Number of training samples: {len(train_samples)}")
    print("Example training sample:", train_samples[0])
    print(f"Number of evaluation samples: {len(eval_samples)}")
    print("Example evaluation sample:", eval_samples[0])

    # Training sentences generations
    # TODO all this stuff should be the speaker name perhaps namespaced by dataset name.
    test_sentences = []
    all_speaker_references: dict[str, dict[Emotion, Path]] = {}
    for dataset_config in datasets_config:
        speaker_references = conditingset_reader(Path(dataset_config.path) / "references")
        print(f"SPEAKER_REFERENCES: {dataset_config.dataset_name}", *speaker_references.items(), sep="\n")
        LANGUAGE = dataset_config.language
        test_sentences += [
                {
                    "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                    "speaker_wav": str(speaker_references["neutral"]),
                    "language": LANGUAGE,
                },
                {
                    "text": "This cake is great. It's so delicious and moist.",
                    "speaker_wav": str(speaker_references['happy']),
                    "language": LANGUAGE,
                },
                {
                    "text": "I am not angry, I am just disappointed.",
                    "speaker_wav": str(speaker_references['sad']),
                    "language": LANGUAGE,
                },
                {
                    "text": "I'm so angry right now, I can't even think straight.",
                    "speaker_wav": str(speaker_references['angry']),
                    "language": LANGUAGE,
                },
            ]
        all_speaker_references[dataset_config.dataset_name] = speaker_references
    config.test_sentences = test_sentences

    print("CONFIGURATION:")
    print(config)

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=str(training_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        test_samples=config.test_sentences,
    )
    trainer.fit()
    # Copy reference files to the model directory
    run_dir= trainer.output_path
    # TODO all this dataset_name stuff shoud be the speaker name perhaps namespaced by dataset name.
    for dataset_name, speaker_references in all_speaker_references.items():
        for emotion, ref_path in speaker_references.items():
            dst_path = run_dir/ f"{dataset_name}_{emotion}.wav"
            print("Created reference file:", dst_path)
            copyfile(ref_path, dst_path)
        base_reference = run_dir / f"{dataset_name}_reference.wav"
        copyfile(speaker_references['neutral'], base_reference)
        print("Created reference file:", base_reference)

def check_cuda_devices(device_name: str | None):
    """
    Set CUDA_VISIBLE_DEVICES based on device_name if provided.
    If device_name is None, use the first available GPU if present, otherwise use CPU.
    Print a warning if using CPU and a GPU was detected.
    """
    if device_name:
        # If device_name is like 'cuda:0', extract the GPU index
        if device_name.startswith("cuda:"):
            gpu_index = device_name.split(":")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        elif device_name == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            print(f"Unknown device_name format: {device_name}")
    else:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            if torch.cuda.device_count() > 0:
                print("Warning: No GPU selected, using CPU even though a GPU was detected.")
            else:
                print("No GPU found. Using CPU.")

import TTS.tts.datasets as td
def add_formatter(name: str, formatter: Callable) -> None:
    if not hasattr(td, name.lower()):
        setattr(td, name.lower(), formatter)
    else:
        raise ValueError(f"Formatter {name} already exists.")

def main(args: Dict) -> None:

    device_name: str | None = args['--device']
    check_cuda_devices(device_name)

    add_formatter("custom_formatter", custom_formatter)
    datasets_config = []
    for dataset_str_path in args['<dataset>']:
        dataset_path = Path(dataset_str_path)
        dataset_name = get_base_name(dataset_path)
        datasets_config.append(dataset_configuration(dataset_path))

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    training_dir = Path(args['--output'])

    model_name = args['<name>']
    run_name = model_run_prefix(model_name)
    RUNS = find_runs(run_name, training_dir)
    if len(RUNS) > 0:
        print(f"Found existing runs: {[str(run) for run in RUNS]}. Skipping...")
        return
    else:
        print("No existing runs found. Proceeding with training...")

    batch_size = int(args['--batch-size'])
    train_voice(run_name, datasets_config, training_dir, batch_size=batch_size)

USAGE = """
Train GPT XTTS model.
Usage:
  train_xtts.py [options] <name> <dataset>...

Options:
  --output=<path>      Directory to save the training output [default: ./run/training/].
  --device=<device>    Device to use for training (e.g., cuda:0, cpu) [default: cuda:0].
  --batch-size=<size>  Batch size for training [default: 9].
"""
if __name__ == "__main__":
    argv = sys.argv[1:]
    # When working with trainer.distribute, many distribution parameters are passed in before the -- marker.
    if '--' in argv:                          # keep only the part after `--`
        argv = argv[argv.index('--') + 1:]

    args = docopt(__doc__, argv=argv)
    main(args)
