from typing import Iterator, List, Tuple
from docopt import docopt
from functional_tools import flat_map, identity, juxt, uniq
from path_utils import is_audio, walk_paths, is_normal_file, make_extension_replacer
from itertools import count
from pathlib import Path

import csv

from shutil import copyfile
from validate_dataset import training_validators, conditioning_validators, compose_validators, get_audio_length, DatasetSample

def mk_sample(dst_path: Path, label: str) -> DatasetSample:
    return (dst_path, label, label)

def trainingset_generator(metadata_path: Path, wav_dir: Path):
    def generate(data: List[Tuple[int, Path, str]]):
        for index, path, label in data:
            print(f"Copying {path} to {wav_dir / f'{index:05d}.wav'}")
            dst_path = wav_dir / f"{index:05d}.wav" # TODO they are not always wavs!
            copyfile(path, dst_path)
            row = mk_sample(dst_path, label)
            yield row
    return generate

def trainingset_writer(metadata_path: Path, wav_dir: Path):
    def writer(data: List[DatasetSample]):
        with metadata_path.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter="|")
            for wav, unnorm, norm in data:
                writer.writerow([wav.stem, unnorm, norm])
    return writer

def load_tag(path: Path) -> str:
  return path.read_text().strip()

is_valid_training_sample = compose_validators(training_validators)
is_valid_conditioning_sample = compose_validators(conditioning_validators)

def trainingset_builder(output_dir: Path, sources: List[Path]):

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    wav_dir = output_dir / "wavs"
    label_writer = trainingset_writer(metadata_path, wav_dir)

    if not wav_dir.exists():
        wav_dir.mkdir(parents=True, exist_ok=True)

    mk_label_path = make_extension_replacer("txt")
    mk_path_pairs = juxt(identity, mk_label_path)
    mk_samples = trainingset_generator(metadata_path, wav_dir)

    paths = flat_map(walk_paths, sources)
    file_paths = filter(is_normal_file, paths)
    audio_paths = filter(is_audio, file_paths)
    uniq_audio_paths = uniq(audio_paths)
    path_pairs = map(mk_path_pairs, uniq_audio_paths)
    normal_pairs = filter(lambda fp: is_normal_file(fp[0]) and is_normal_file(fp[1]), path_pairs)
    path_labels: Iterator[Tuple[int, Path, str]] = map(lambda pl, index: (index, pl[0], load_tag(pl[1])), normal_pairs, count())
    samples = mk_samples(path_labels)
    valid_samples = filter(is_valid_training_sample, samples)
    label_writer(valid_samples)


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