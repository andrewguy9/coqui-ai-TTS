from typing import Callable, Iterable, Iterator, List, Tuple, Dict, get_args, TypedDict
import json
from audio_emotion import Emotion
from docopt import docopt
from pathlib import Path

class DatasetSampleDict(TypedDict):
    identifier: str
    unnormalized_text: str
    normalized_text: str
    duration: float
    emotions: Dict[Emotion, float]
    speaker_name: str

def dataset_reader(dataset_path: Path) -> Iterator[DatasetSampleDict]:
    metadata_path = dataset_path / "metadata.json"
    wav_dir = dataset_path / "wavs"

    if not metadata_path.exists() or not wav_dir.exists():
        raise FileNotFoundError("Dataset metadata or wav directory does not exist.")

    with metadata_path.open('r', newline='') as json_file:
        samples = json.load(json_file)
        for sample in samples:
            sDict = DatasetSampleDict(
                identifier=str(sample['identifier']),
                unnormalized_text=sample['unnormalized_text'],
                normalized_text=sample['normalized_text'],
                duration=sample['duration'],
                emotions=sample['emotions'],
                speaker_name=sample['speaker_name']
            )
            yield sDict

def validate_sample_text_length(r: DatasetSampleDict) -> bool:
    return len(r['unnormalized_text']) < 250 and len(r['normalized_text']) < 250

def validate_sample_audio_length(r: DatasetSampleDict) -> bool:
    max_audio_length = 11.6  # seconds
    return r['duration'] <= max_audio_length

def is_valid_conditioning_length(duration: float) -> bool:
    min_conditioning_length = 3  # seconds
    max_conditioning_length = 6  # seconds
    return min_conditioning_length <= duration <= max_conditioning_length

def validate_sample_conditioning_length(r: DatasetSampleDict) -> bool:
    return is_valid_conditioning_length(r['duration'])

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

def validate_outcries(r: DatasetSampleDict) -> bool:
    outcries = find_outcries(r['unnormalized_text'])
    return len(outcries) == 0

def validate_all(r: DatasetSampleDict) -> bool:
    return (validate_sample_text_length(r) and
            validate_sample_audio_length(r) and
            validate_sample_conditioning_length(r) and
            validate_outcries(r))


training_validators: List[Callable[[DatasetSampleDict], bool]] = [
    validate_sample_text_length,
    validate_sample_audio_length,
    validate_outcries,]
conditioning_validators: List[Callable[[DatasetSampleDict], bool]] = [
    validate_outcries,
    # validate_sample_conditioning_length, # TODO trying out compound conditioning samples.
]

# TODO every-pred
def compose_validators(validators: List[Callable[[DatasetSampleDict], bool]]) -> Callable[[DatasetSampleDict], bool]:
    def combined_validator(sample: DatasetSampleDict) -> bool:
        return all(validator(sample) for validator in validators)
    return combined_validator

def validate_dataset_records(samples: Iterable[DatasetSampleDict], validators: List[Callable[[DatasetSampleDict], bool]]) -> None:
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