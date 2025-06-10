from typing import List
from docopt import docopt
from functional_tools import identity, juxt, uniq
from path_utils import get_parent_path, get_relative_path, is_audio, walk_paths, is_normal_file, make_extension_replacer
from itertools import chain, count
from pathlib import Path

import csv

from shutil import copyfile

def trainingset_writer(metadata_path: Path, wav_dir: Path):
    def writer(data: list[int, Path, str]):
        with metadata_path.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter="|")
            for index, path, label in data:
                print(f"Copying {path} to {wav_dir / f'{index:05d}.wav'}")
                path: Path
                dst_path = wav_dir / f"{index:05d}.wav" # TODO they are not always wavs!
                copyfile(path, dst_path)
                writer.writerow([dst_path.stem, label, label])
    return writer

def load_tag(path: Path) -> str:
  return path.read_text().strip()

def trainingset_builder(output_dir: Path, sources: List[Path]):

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    wav_dir = output_dir / "wavs"
    label_writer = trainingset_writer(metadata_path, wav_dir)

    if not wav_dir.exists():
        wav_dir.mkdir(parents=True, exist_ok=True)

    label_path = make_extension_replacer("txt")
    path_pairs = juxt(identity, label_path)

    paths = chain(*map(walk_paths, sources))
    file_paths = filter(is_normal_file, paths)
    audio_paths = filter(is_audio, file_paths)
    uniq_audio_paths = uniq(audio_paths)
    file_pairs = map(path_pairs, uniq_audio_paths)
    valid_pairs = filter(lambda fp: is_normal_file(fp[0]) and is_normal_file(fp[1]), file_pairs)
    path_labels = map(lambda pl, index: (index, pl[0], load_tag(pl[1])), valid_pairs, count())
    label_writer(path_labels)

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