import json
from typing import Dict, Iterator, List, Tuple, get_args
from docopt import docopt
from audio_emotion import EmotionFallbacks
from functional_tools import flat_map, identity, juxt, uniq
from label_audio import Emotion
from path_utils import is_audio, walk_paths, is_normal_file, make_extension_replacer
from itertools import count, groupby
from pathlib import Path

import csv

from shutil import copyfile
from validate_dataset import get_audio_duration, is_valid_conditioning_length, training_validators, conditioning_validators, compose_validators, get_sample_length, DatasetSample

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

def sample_emotion(sample: DatasetSample) -> Emotion:
    """
    Get the primary emotion from a sample.
    """
    _, _, _, _, emotions = sample
    return get_primary_emotion(emotions)

def pack_reference_samples(max_duration: float, samples: List[DatasetSample]):
    """
    Take a list of samples and find a way to maximally pack clips into a single reference sample.
    Returns a list of the DatasetSample objects which can be combined.
    Optimizes for the fewest number of samples which get as close to the max_duration as possible.
    """
    ordered_samples = sorted(samples, key=get_sample_length, reverse=True)
    references = []
    total_duration = 0.0
    for candidate in ordered_samples:
        candidate_duration = get_sample_length(candidate)
        if total_duration + candidate_duration <= max_duration:
            references.append(candidate)
            total_duration += candidate_duration
            continue
    return references

from torchaudio import load, save
from torch import cat

def make_reference_wav(dst_path: Path, samples: List[DatasetSample]):
    """
    Combine the given samples into a single wav file at the dst_path.
    """
    wavs = [load(sample[0]) for sample in samples]
    # Each wav is a tuple: (tensor, sample_rate)
    tensors = [wav[0] for wav in wavs]
    sample_rate = wavs[0][1]
    combined = cat(tensors, dim=1)
    save(dst_path, combined, sample_rate)


from itertools import groupby
def conditioningset_writer(references_dir: Path):
    def writer(samples: List[DatasetSample]):
        """
        Produces a single reference wav file for each emotion.
        """
        # First, sort samples by primary emotion so groupby works correctly
        sorted_samples = sorted(samples, key=sample_emotion)
        emotion_samples = groupby(sorted_samples, key=sample_emotion)
        for emotion, group in emotion_samples:
            group_samples = list(group)
            if not group_samples:
                continue
            # Pack samples into a single reference wav
            packed_samples = pack_reference_samples(6.0, group_samples)
            packed_duration = sum([get_sample_length(sample) for sample in packed_samples])
            if is_valid_conditioning_length(packed_duration):
                dst_path = references_dir / f"{emotion}.wav"
                print(f"Creating reference for {emotion} at {dst_path} with {len(packed_samples)} samples")
                make_reference_wav(dst_path, packed_samples)
            else:
                print(f"Skipping {emotion} reference due to invalid duration: {packed_duration:.2f} seconds")
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