from typing import List
from docopt import docopt
from functional_tools import identity, juxt, uniq
from path_utils import get_parent_path, get_relative_path, is_audio, walk_paths, is_normal_file, make_extension_replacer
from itertools import chain
from pathlib import Path

import csv

def dataset_writer(output_path: Path):
    cwd = get_parent_path(output_path)
    def writer(data: list[Path, str]):
        with output_path.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter="|")
            for path, label in data:
                relative = get_relative_path(cwd, path)
                writer.writerow([str(relative), label, label])
    return writer

def load_tag(path: Path) -> str:
  return path.read_text().strip()

def make_dataset(output_path: Path, sources: List[Path]):
    label_writer = dataset_writer(output_path)
    label_path = make_extension_replacer("txt")
    path_pairs = juxt(identity, label_path)

    paths = chain(*map(walk_paths, sources))
    file_paths = filter(is_normal_file, paths)
    audio_paths = filter(is_audio, file_paths)
    uniq_paths = uniq(audio_paths)
    audio_files = filter(is_audio, uniq_paths)
    existing_audio_files = filter(is_normal_file, audio_files)
    file_pairs = map(path_pairs, existing_audio_files)
    valid_pairs = filter(lambda fp: is_normal_file(fp[0]) and is_normal_file(fp[1]), file_pairs)
    path_labels = map(lambda pl: (pl[0], load_tag(pl[1])), valid_pairs)
    label_writer(path_labels)

def main(args):
    output_path = args['<output>']
    sources = args['<source>']

    make_dataset(Path(output_path), map(Path, sources))

USAGE = """
Combine directories to produce a dataset.

Usage:
  make_dataset.py <output> <source>...
"""
if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)