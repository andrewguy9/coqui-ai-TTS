from typing import Callable, Iterable, Iterator, List, Tuple, Dict, get_args
from audio_emotion import Emotion
from docopt import docopt
from pathlib import Path

import csv

DatasetSample = Tuple[Path, str, str, float, Dict[Emotion, float]]  # (auto_file_stem, unnormalized text, normalized_text)
def dataset_reader(dataset_path: Path) -> Iterator[DatasetSample]:
    metadata_path = dataset_path / "metadata.csv"
    wav_dir = dataset_path / "wavs"

    if not metadata_path.exists() or not wav_dir.exists():
        raise FileNotFoundError("Dataset metadata or wav directory does not exist.")

    with metadata_path.open('r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        for row in reader:
            emotions = {e: float(v) for e, v in zip(get_args(Emotion), row[3:])}
            sample_path = wav_dir / f"{row[0]}.wav"
            duration = get_audio_duration(sample_path)
            yield sample_path, row[1], row[2], duration, emotions

def validate_sample_text_length(r: DatasetSample) -> bool:
    _, text, normalized, _, _ = r
    return len(text) < 250 and len(normalized) < 250

from torchaudio import info as audio_info

def get_audio_duration(audio_path: Path) -> float:
    metadata = audio_info(audio_path)
    return metadata.num_frames / metadata.sample_rate

# TODO remove, once sample duration is part of the DatasetSample record.
def get_sample_length(r: DatasetSample) -> float:
    _, _, _, duration, _ = r
    return duration

def validate_sample_audio_length(r: DatasetSample) -> bool:
    max_audio_length = 11.6  # seconds
    _, _, _, duration, _ = r
    return duration <= max_audio_length

def validate_sample_conditioning_length(r: DatasetSample) -> bool:
    min_conditioning_length = 3  # seconds
    max_conditioning_length = 6  # seconds
    _, _, _, duration, _ = r
    return min_conditioning_length <= duration <= max_conditioning_length

import spacy
from spacy.matcher import Matcher

# Pattern: any token containing a run of the same letter 4+ times
"""
WOOHOOHOO
AWWWWWW
Ha!
Whoop!
Wah!
Waha!
HUAH!
HOOP
DENIED!
POING
Hehehe
BOOOOOO!
Ha ha!
Ho ho ho!
Ho ho ho! Ho ho ho! You got owned!
HAHAHA
HA HA HA HA HA HA HA HA HA
Ha!
AHHHHHHHHHH
AHHHHH
UGH!
Argh!
AH!
Oof!
PFFT
WOOO!
BWAH!

few words
lots of all caps
"""

nlp = spacy.load("en_core_web_sm")
outcry_matcher = Matcher(nlp.vocab)

outcry_matcher.add("SCREAMING", [[{"TEXT": {"REGEX": r"([A-Za-z])\1{3,}"}}, {"ORTH": "!", "OP": "?"}]])
outcry_matcher.add("LAUGHING",  [[{"TEXT": {"REGEX": r"([Hh][Aa])+"}}, {"ORTH": "!", "OP": "?"}],
                                 [{"TEXT": {"REGEX": r"([Hh][Ee])+"}}, {"ORTH": "!", "OP": "?"}],
                                 [{"TEXT": {"REGEX": r"([Hh][Oo])+"}}, {"ORTH": "!", "OP": "?"}]])
outcry_matcher.add("YELLING",   [[{"TEXT": {"REGEX": r"WO{2,}"}}, {"ORTH": "!"}],
                                 [{"TEXT": {"REGEX": r"[A-Z]{4,}"}}, {"ORTH": "!", "OP": "?"}]])
outcry_matcher.add("BOOING",    [[{"TEXT": {"REGEX": r"BO{2,}"}}, {"ORTH": "!"}]])
outcry_matcher.add("GASPING",   [[{"TEXT": "Wah"}, {"ORTH": "!", "OP": "?"}],
                                 [{"TEXT": "AH"}, {"ORTH": "!", "OP": "?"}]])
outcry_matcher.add("CHEERING",  [[{"TEXT": {"REGEX": r"Whoop"}}]])
outcry_matcher.add("GRUNTING", [[{"TEXT": "HUAH"}, {"ORTH": "!", "OP": "?"}],
                                [{"TEXT": "UGH"}, {"ORTH": "!", "OP": "?"}],
                                [{"TEXT": "Argh"}, {"ORTH": "!", "OP": "?"}],
                                [{"TEXT": "Oof"}, {"ORTH": "!", "OP": "?"}],
                                ])
outcry_matcher.add("SPUTTERING",[[{"TEXT": "PFFT"}],
                                 [{"TEXT": "BWAH"}, {"ORTH": "!", "OP": "?"}]])

def find_outcries(text):
    doc = nlp(text)
    matches = outcry_matcher(doc)
    outcries = [doc[start:end].text for _, start, end in matches]
    return outcries

def test_find_outcries():
    emotive_cases = \
    """
    WOOHOOHOO
    AWWWWWW
    Haha
    Ha!
    Whoop!
    Wah!
    Waha!
    HUAH!
    HOOP
    Haha!
    Hehehe
    BOOOOOO!
    Ha ha!
    Ho ho ho!
    Ho ho ho! Ho ho ho! You got owned!
    HAHAHA
    HA HA HA HA HA HA HA HA HA
    Ha!
    AHHHHHHHHHH
    AHHHHH
    UGH!
    Argh!
    AH!
    Oof!
    PFFT
    WOOO!
    BWAH!
    """.strip().splitlines(keepends=False)
    yelling_word_cases = \
    """
    DENIED!
    POING
    """.strip().splitlines(keepends=False)
    pos_cases = emotive_cases + yelling_word_cases
    for s in pos_cases:
        s = s.strip()
        # print("testing:", s)
        outcries = find_outcries(s)
        assert len(outcries) > 0, s
        # print("\tfound outcries:", outcries)

test_find_outcries()

def validate_outcries(r: DatasetSample) -> bool:
    _, text, _, _, _ = r
    outcries = find_outcries(text)
    return len(outcries) == 0

def validate_all(r: DatasetSample) -> bool:
    return (validate_sample_text_length(r) and
            validate_sample_audio_length(r) and
            validate_sample_conditioning_length(r) and
            validate_outcries(r))


training_validators: List[Callable[[DatasetSample], bool]] = [
    validate_sample_text_length,
    validate_sample_audio_length,
    validate_outcries,]
conditioning_validators: List[Callable[[DatasetSample], bool]] = [
    validate_outcries,
    validate_sample_conditioning_length,
]

# TODO every-pred
def compose_validators(validators: List[Callable[[DatasetSample], bool]]) -> Callable[[DatasetSample], bool]:
    def combined_validator(sample: DatasetSample) -> bool:
        return all(validator(sample) for validator in validators)
    return combined_validator

def validate_dataset_records(samples: Iterable[DatasetSample], validators: List[Callable[[DatasetSample], bool]]) -> None:
    samples = list(samples)
    combined = compose_validators(validators)
    
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