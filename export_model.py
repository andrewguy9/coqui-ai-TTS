from coqpit import Coqpit
from docopt import docopt
import glob, os
from pathlib import Path
import shutil
import torch
import json
from train_xtts import find_models, find_runs, model_run_prefix

from TTS.tts.models.xtts import Xtts
from TTS.config import load_config

# find your latest checkpoint (adjust the glob as needed)
def find_best_ckpt(run_dir):
    "Find the config.json and best_model.pth file Paths."
    best = glob.glob(os.path.join(run_dir, "best_model.pth"))
    if not best:
        raise FileNotFoundError(f"No best model found in {run_dir}")
    best_path = run_dir / Path(best[0])
    config = glob.glob(os.path.join(run_dir, "config.json"))
    if not config:
        raise FileNotFoundError(f"No config file found in {run_dir}")
    config_path = run_dir / Path(config[0])
    return config_path, best_path

def find_tokenizer_file(config: Coqpit) -> Path:
    """
    Note: The trainer puts run/training at the start of the path,
    so we need to remove that part to get the correct relative path.
    """
    model_args = config.get("model_args")
    vocab_path = Path(model_args.get('tokenizer_file'))
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Tokenizer file {vocab_path} not found")
    if not vocab_path.is_absolute():
        raise ValueError(f"Tokenizer file {vocab_path} must be an absolute path.")
    return vocab_path

def copy_vocab_file(old_vocab_path: Path, output_dir: Path):
    output_vocab_path = output_dir / "vocab.json"
    shutil.copy(old_vocab_path, output_vocab_path)
    return output_vocab_path

def export_xtts_weights(run_dir: Path, output_dir: Path):
    config_path, ckpt_path = find_best_ckpt(run_dir)
    # load full checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # remove optimizer state (this typically cuts ~2/3 of the size)
    checkpoint.pop("optimizer", None)
    # optionally remove DVAE/discriminator if present
    for key in list(checkpoint.get("model", {})):
        if "dvae" in key or "disc" in key:
            del checkpoint["model"][key]

    ckpt_path_out = output_dir / "model.pth"
    config_path_out = output_dir / "config.json"

    # save pruned, inference-only checkpoint
    torch.save(checkpoint, ckpt_path_out)
    shutil.copy(config_path, config_path_out)

# TODO you need to copy vocab.json
# TODO .test_sentences[0].speaker_wav has the reference_wav path full path.
# TODO .model_args.tokenizer_file has relative path to the vocab.json file.
# TODO .model_args.mel_norm_file has relative path to the origional xtts mel norm file.
# TODO .model_args.dvae_checkpoint has relative path to the original xtts dvae checkpoint.
# TODO .model_args.xtts_checkpoint has relative path to the original xtts checkpoint.
def export_xtts_tokenizer(run_dir: Path, config, out_dir: Path):
    """
    Rundir is the directory containing all the training runs.
    This is needed to find the XTTS tokenizer file.
    """
    # TODO 
    old_vocab_path = find_tokenizer_file(config)
    if not old_vocab_path.is_file():
        raise FileNotFoundError(f"Tokenizer file {old_vocab_path} not found in run directory {run_dir}.")
    new_vocab_path = copy_vocab_file(old_vocab_path, out_dir)
    print(f"Copied tokenizer file from: {old_vocab_path} to {new_vocab_path}")
    return new_vocab_path

def export_xtts_weights_minified(config, weights_path: Path, out_dir: Path):
    vocab_path = find_tokenizer_file(config)
    try:
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_path=weights_path, vocab_path=vocab_path)
    except Exception:
        raise RuntimeError(f"Failed to load model from {weights_path} and config:\n{config.to_json()}")
    new_weights_path = out_dir / "model.pth"
    torch.save(model.state_dict(), new_weights_path)
    return new_weights_path

def export_xtts_model_config(config, new_vocab_path: Path, new_weights_path: Path, out_dir: Path):
    # TODO update relative paths in the config
    config['model_args']['tokenizer_file'] = new_vocab_path
    # TODO update relative paths in the config
    config['model_args']['xtts_checkpoint'] = new_weights_path
    config_out_path = out_dir / "config.json"
    json_config = config.to_json()
    print(f"New model config at {config_out_path}")
    print(json_config)
    with open(config_out_path, 'w') as f:
        json.dump(config.to_json(), f, indent=4)
    return config_out_path

def export_xtts_finetune(run_dir: Path, src_dir: Path, out_dir: Path):
    """
    run_dir is the directory containing all the training runs.
    This is needed to find config resources from the base XTTS files.
    src_dir is the directory containing the XTTS training run.
    out_dir is the directory where the exported model will be saved.
    It should be inside the dist directory by convention.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path, best_path = find_best_ckpt(src_dir)
    config = load_config(config_path)
    new_vocab_path = export_xtts_tokenizer(run_dir, config, out_dir)
    new_best_path = export_xtts_weights_minified(config, best_path, out_dir)
    export_xtts_model_config(config, new_vocab_path, new_best_path, out_dir)

def main(args):
    run_dir = Path(args['<run_dir>'])
    dist_dir = Path(args['<dist_dir>'])

    model_names = args['<model_name>']
    if len(model_names) == 0:
        print(*find_models(run_dir), sep="\n")
        return 0
    for model_name in model_names:
        model_prefix = model_run_prefix(model_name)

        runs = find_runs(model_prefix, run_dir)
        if len(runs) == 0:
            raise FileNotFoundError(f"No runs found for model {model_name} in directory {run_dir}.")
        if len(runs) > 1:
            raise ValueError(f"Multiple runs found for model {model_name} in directory {run_dir}. Please specify a unique run.")
        src_dir = runs[0]

        output_dir = dist_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output directory {output_dir} does not exist or is not a directory.")

        print(f"Exporting XTTS model from {src_dir} to {output_dir}")
        export_xtts_finetune(run_dir, src_dir, output_dir)

USAGE = """
Export a trained XTTS model from rundir to a specified output directory.

Usage:
    export_model.py <run_dir> <dist_dir> [<model_name>...]
"""

if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)