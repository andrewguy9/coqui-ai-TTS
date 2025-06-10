from typing import Callable, Iterable, Iterator, List, Tuple
from docopt import docopt
from functional_tools import identity, juxt, uniq
from pathlib import Path

import csv

DatasetSample = Tuple[Path, str, str]  # (auto_file_stem, unnormalized text, normalized_text)
def dataset_reader(dataset_path: Path) -> Iterator[DatasetSample]:
    metadata_path = dataset_path / "metadata.csv"
    wav_dir = dataset_path / "wavs"

    if not metadata_path.exists() or not wav_dir.exists():
        raise FileNotFoundError("Dataset metadata or wav directory does not exist.")

    with metadata_path.open('r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        for row in reader:
            yield wav_dir / f"{row[0]}.wav", row[1], row[2]

def validate_sample_text_length(r: DatasetSample) -> bool:
    _, text, normalized = r
    return len(text) < 250 and len(normalized) < 250

from torchaudio import info

def get_audio_length(r: DatasetSample) -> float:
    audio_path, _, _ = r
    metadata = info(audio_path)
    return metadata.num_frames / metadata.sample_rate

def validate_sample_audio_length(r: DatasetSample) -> bool:
    max_audio_length = 11.6  # seconds
    audio_path, _, _ = r
    seconds = get_audio_length(r)
    return seconds <= max_audio_length

def validate_sample_conditioning_length(r: DatasetSample) -> bool:
    min_conditioning_length = 3  # seconds
    max_conditioning_length = 6  # seconds
    audio_path, _, _ = r
    seconds = get_audio_length(r)
    return min_conditioning_length <= seconds <= max_conditioning_length

import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Pattern: any token containing a run of the same letter 4+ times
matcher.add(
    "OUTCRY",
    [
        [{"TEXT": {"REGEX": r"([A-Za-z!?.])\1{3,}"}}]
    ]
)

def find_outcries(text):
    doc = nlp(text)
    matches = matcher(doc)
    screams = [doc[start:end].text for _, start, end in matches]
    return screams

def validate_outcries(r: DatasetSample) -> bool:
    _, text, _ = r
    screams = find_outcries(text)
    return len(screams) == 0

def validate_all(r: DatasetSample) -> bool:
    return (validate_sample_text_length(r) and
            validate_sample_audio_length(r) and
            validate_sample_conditioning_length(r) and
            validate_outcries(r))


training_validators = [
    validate_sample_text_length,
    validate_sample_audio_length,
    validate_outcries,]
conditioning_validators = [
    validate_outcries,
    validate_sample_conditioning_length,
]

def all_validators(validators: List[Callable[[DatasetSample], bool]]) -> Callable[[DatasetSample], bool]:
    def combined_validator(sample: DatasetSample) -> bool:
        return all(validator(sample) for validator in validators)
    return combined_validator

def validate_dataset_records(samples: Iterable[DatasetSample], validators: List[Callable[[DatasetSample], bool]]) -> None:
    samples = list(samples)
    combined = all_validators(validators)
    
    for validator in validators + [combined]:
        valid_samples = list(map(validator, samples))
        pct_valid = sum(valid_samples) / len(samples)
        print(f"Valid samples for {validator.__name__}: {pct_valid:.2%}")
    
def main(args):
    output_path = Path(args['<source>'])

    validate_dataset_records(dataset_reader(output_path), training_validators)

USAGE = """
Combine directories to produce a dataset.

Usage:
  validate_dataset.py <source>
"""
if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)