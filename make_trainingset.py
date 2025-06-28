import json
from typing import Dict, Iterator, List, Tuple, get_args
from docopt import docopt
from audio_emotion import EmotionFallbacks
from functional_tools import flat_map, identity, juxt, uniq
from label_audio import Emotion
from path_utils import is_audio, walk_paths, is_normal_file, make_extension_replacer
from itertools import count
from pathlib import Path

import csv

from shutil import copyfile
from validate_dataset import get_audio_duration, training_validators, conditioning_validators, compose_validators, get_sample_length, DatasetSample

def mk_sample(dst_path: Path, label: str, duration: float, emotions: Dict[Emotion, float]) -> DatasetSample:
    return (dst_path, label, label, duration, emotions)

def samples_generator(wav_dir: Path):
    def generate(data: List[Tuple[int, Path, str, Dict[Emotion, float]]]) -> Iterator[DatasetSample]:
        for index, path, label, emotions in data:
            print(f"Copying {path} to {wav_dir / f'{index:05d}.wav'}")
            dst_path = wav_dir / f"{index:05d}.wav" # TODO they are not always wavs!
            duration = get_audio_duration(path)
            copyfile(path, dst_path)
            row = mk_sample(dst_path, label, duration, emotions)
            yield row
    return generate

def trainingset_writer(metadata_path: Path, wav_dir: Path):
    def writer(data: List[DatasetSample]):
        with metadata_path.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter="|")
            for wav, unnorm, norm, duration, emotions in data:
                emo_cols = [emotions.get(e, 0.0) for e in get_args(Emotion)]
                writer.writerow([wav.stem, unnorm, norm, duration, *emo_cols])
    return writer

def get_primary_emotion(emotions: Dict[Emotion, float]) -> Emotion:
    """
    Get the primary emotion from the emotions dictionary.
    Returns the emotion with the highest value.
    """
    if not emotions:
        return "neutral" # Default to NEUTRAL if no emotions are present
    return max(emotions, key=emotions.get) # type: ignore

def conditioningset_writer(references_dir: Path):
    """
    """
    def writer(samples: List[DatasetSample]):
        """
        Finds one sample per emotion and copies it to a references directory.
        Keeps the longest sample for each emotion.
        """
        example_samples: Dict[Emotion, DatasetSample] = {}
        for sample in samples:
            wav_path, _, _, _, emotions = sample
            duration = get_sample_length(sample)
            primary_emotion = get_primary_emotion(emotions)
            if primary_emotion not in example_samples or duration > get_sample_length(example_samples[primary_emotion]):
                example_samples[primary_emotion] = sample
        for emotion, sample in example_samples.items():
            wav_path, _, _, _, _ = sample
            dst_path = references_dir / f"{emotion}.wav"
            print(f"Copying {wav_path} to {dst_path}")
            copyfile(wav_path, dst_path)
    return writer

def find_conditioning_sample(references_dir: Path, emotion: Emotion) -> Path:
    path = references_dir / f"{emotion}.wav"
    if path.exists():
        return path
    for fallback in EmotionFallbacks[emotion]:
        fallback_path = references_dir / f"{fallback}.wav"
        if fallback_path.exists():
            return fallback_path
    raise FileNotFoundError(f"No conditioning sample found for emotion {emotion} in {references_dir}.")

def conditingset_reader(references_dir: Path) -> Dict[Emotion, Path]:
    """
    Reads the conditioning set from the references directory.
    Returns a dictionary mapping emotions to their corresponding wav file paths.
    """
    conditioning_samples: Dict[Emotion, Path] = {}
    for emotion in get_args(Emotion):
        path = find_conditioning_sample(references_dir, emotion)
        conditioning_samples[emotion] = path
    return conditioning_samples

def load_tag(path: Path) -> str:
  return path.read_text().strip()

def load_emote(path: Path) -> Dict[Emotion, float]:
    """
    Load the emotion label from a file.
    """
    with path.open('r') as f:
        emotions: Dict[Emotion, float] = json.load(f)
        return emotions

is_valid_training_sample = compose_validators(training_validators)
is_valid_conditioning_sample = compose_validators(conditioning_validators)

def trainingset_builder(output_dir: Path, sources: List[Path]):

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    wav_dir = output_dir / "wavs"
    reference_dir = output_dir / "references"
    label_writer = trainingset_writer(metadata_path, wav_dir)
    conditioning_writer = conditioningset_writer(reference_dir)

    wav_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    mk_label_path = make_extension_replacer("txt")
    mk_emote_path = make_extension_replacer("emo")
    mk_path_pairs = juxt(identity, mk_label_path, mk_emote_path)
    mk_samples = samples_generator(wav_dir)

    paths = flat_map(walk_paths, sources)
    file_paths = filter(is_normal_file, paths)
    audio_paths = filter(is_audio, file_paths)
    uniq_audio_paths = uniq(audio_paths)
    path_groups = map(mk_path_pairs, uniq_audio_paths) # wav, label, emote
    have_label_files = filter(lambda fp: is_normal_file(fp[0]) and is_normal_file(fp[1]) and is_normal_file(fp[2]), path_groups)
    path_data: Iterator[Tuple[int, Path, str, Dict[Emotion, float]]] = map(lambda group, index: (index, group[0], load_tag(group[1]), load_emote(group[2])), have_label_files, count())
    samples = list(mk_samples(path_data))

    training_samples = filter(is_valid_training_sample, samples)
    conditioning_samples = filter(is_valid_conditioning_sample, samples)

    label_writer(training_samples)
    conditioning_writer(conditioning_samples)


def main(args):
    output_path = args['<output>']
    sources = args['<source>']

    trainingset_builder(Path(output_path), map(Path, sources))

USAGE = """
Combine dataset directories to produce a trainingset.

Usage:
  make_trainingset.py <output> <source>...
"""
if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)