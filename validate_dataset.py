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

def validate_sample_word_count(r: DatasetSampleDict) -> bool:
    unnormalized_word_count = len(r['unnormalized_text'].split())
    normalized_word_count = len(r['normalized_text'].split())
    return unnormalized_word_count >= 1 and normalized_word_count >= 1

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

nlp = spacy.load("en_core_web_sm")
outcry_matcher = Matcher(nlp.vocab)

outcry_matcher.add("SCREAMING", [
    [{"LOWER": {"REGEX": r"^([a-z])\1{2,}$"}}, {"ORTH": "!", "OP": "?"}], # "mmm!" "zzz"
])
outcry_matcher.add("LAUGHING",  [
    [{"TEXT": {"REGEX": r"^([Hh][Aa])+$"}}, {"ORTH": "!", "OP": "?"}], # "Ha!", "Hahaha"
    [{"TEXT": {"REGEX": r"^([Hh][Ee])+$"}}, {"ORTH": "!", "OP": "?"}], # "He!" "Hehehe"
    [{"TEXT": {"REGEX": r"^([Hh][Oo])+$"}}, {"ORTH": "!", "OP": "?"}], # "Ho!" "Hohoho"
    [{"TEXT": {"REGEX": r"^[Ww][Aa]([Hh][Aa])+$"}}, {"ORTH": "!", "OP": "?"}], # "Waha!" "Wahaha"
])
outcry_matcher.add("YELLING",   [
    [{"TEXT": {"REGEX": r"^WO{2,}$"}}, {"ORTH": "!"}], # "Woo!"
    [{"TEXT": {"REGEX": r"^[A-Z]{3,}$"}}, {"ORTH": "!", "OP": "?"}], # "BOOM!" "YES!"
    [{"TEXT": {"REGEX": r"^[A-Za-z] $",}}], # Single character words # TODO broken because of tokenization. # TODO bad rule?
    [{"TEXT": {"REGEX": r"^[Nn][Oo]{1,}$"}}, {"ORTH": "!", "OP": "+"}], # "No!" "NOOO!" "Noooooo!!!!"
    [{"TEXT": "YES"}, {"ORTH": "!", "OP": "*"}], # "YES" "YES!" "YES!!" "YES!!!"
])
outcry_matcher.add("BOOING", [
    [{"TEXT": {"REGEX": r"^B[Oo]{2,}$"}}, {"ORTH": "!"}], # "Boo!" "Booooo"
    [{"TEXT": "FOO"}, {"ORTH": "!", "OP": "?"}], # "FOO!" "FOO"
])
outcry_matcher.add("GASPING",   [
    [{"TEXT": "Wah"}, {"ORTH": "!", "OP": "?"}], # "Wah" "Wah!"
    [{"TEXT": {"REGEX": r"^[Aa]{1,}[hH]{1,}"}}, {"ORTH": "!", "OP": "?"}], # "AH", "AH!" Ah Aaaaah! AaaaHhhhh
    [{"LOWER": "gah"}, {"ORTH": "!", "OP": "?"}], # "GAH" "GAH!" "gah" "gah!"
])
outcry_matcher.add("CHEERING",  [
    [{"TEXT": "Whoop"}], # "Whoop"
    [{"TEXT": "BAM"}], # "BAM"
    [{"TEXT": "BAH"}], # "BAH"
]) 
outcry_matcher.add("GRUNTING", [
    [{"TEXT": "HUAH"}, {"ORTH": "!", "OP": "?"}], # "HUAH" "HUAH!"
    [{"LOWER": "ugh"}, {"ORTH": "!", "OP": "?"}], # "UGH" "UGH!" "ugh"
    [{"TEXT": "Argh"}, {"ORTH": "!", "OP": "?"}], # "Argh" "Argh!"
    [{"TEXT": "AGH"}, {"ORTH": "!", "OP": "?"}], # "AGH" "AGH!"
    [{"TEXT": {"REGEX": r"^[Oo]{2,}f$"}}, {"ORTH": "!", "OP": "?"}], # "Oof" "Oof!" "OOF"
])
outcry_matcher.add("SPUTTERING",[
    [{"TEXT": "PFFT"}], # "PFFT"
    [{"TEXT": "BWAH"}, {"ORTH": "!", "OP": "?"}] # "BWAH" "BWAH!"
])
outcry_matcher.add("RECOILING", [
    [{"TEXT": "Eekk"}, {"ORTH": "!", "OP": "?"}],  # "Eekk" "Eekk!"
])
outcry_matcher.add("SHRUGGING", [
    [{"TEXT": "Eh"}, {"ORTH": "!", "OP": "?"}],  # "Eh" "Eh!"
    [{"TEXT": "Eh"}, {"ORTH": "?", "OP": "?"}],  # "Eh" "Eh?"
])
outcry_matcher.add("TAUNTING", [
    [{"TEXT": "POING"}],  # "POING"
    [{"TEXT": "BAP"}], # "BAP"
    [{"TEXT": "Boink"}, {"ORTH": "!", "OP": "?"}], # "Boink"
])
outcry_matcher.add("GAGGING", [
    [{"LOWER": "ew"}, {"ORTH": "!", "OP": "?"}], # EW EW!
])
outcry_matcher.add("GROWLING", [
    [{"LOWER": "grrr"}, {"ORTH": "!", "OP": "?"}], # "grrr" "grrr!" "Grrr" "Grrr!"
])
outcry_matcher.add("HUMMING", [
    [{"LOWER": {"REGEX": r"^hm{2,}$"}}, {"ORTH": "!", "OP": "?"}], # "hmmm"
])
outcry_matcher.add("QUESTIONING", [
    [{"LOWER": "huh"}, {"TEXT": {"REGEX": r"^[?!]{1,}$"}}], # "huh?" "huh!" "huh?!"
])
outcry_matcher.add("EATING", [
    [{"LOWER": "nom"}],
])
outcry_matcher.add("ROARING", [
    [{"LOWER": "rawr"}, {"ORTH": "!", "OP": "?"}], # "Rawr" "Rawr!" "RAwR"
])

def find_outcry_matches(text):
    doc = nlp(text)
    matches = outcry_matcher(doc)
    return [(doc[start:end].text, outcry_matcher.vocab[label].text) for label, start, end in matches]

def find_outcry_texts(text):
    doc = nlp(text)
    matches = outcry_matcher(doc)
    outcries = [doc[start:end].text for _, start, end in matches]
    return outcries

def test_find_outcry_matcher():
    emotive_cases = \
    """
    AAH!
    AGH!
    AH!
    AHHHHH
    AHHHHHHHHHH
    AWWWWWW
    Ah
    Ah!
    Argh!
    BAM
    BOOOOOO!
    Boink!
    BWAH!
    Boo!
    Eekk!
    Eh!
    Eh?
    EW!
    Ew!
    FOO
    gah
    Gah
    GAH
    GAH GAH GAH
    grrr!
    HA HA HA HA HA HA HA HA HA
    HAHAHA
    HOOP
    HUAH!
    Ha ha!
    Ha!
    Ha!
    Haha
    Haha!
    Hehehe
    huh?
    huh!
    huh?!
    hmm
    hmmm
    Ho ho ho!
    Ho ho ho! Ho ho ho! You got owned!
    Mmmm
    nom nom nom
    No!
    NOOO!
    Noooooo!!!!
    nom, nom, nom
    Oof!
    OOF
    PFFT
    UGH!
    ugh
    rawr
    RAWR!
    WOOHOOHOO
    WOOO!
    Wah!
    Waha!
    Whoop!
    """.strip().splitlines(keepends=False)
    yelling_word_cases = \
    """
    DENIED!
    POING
    HIT!
    YES!
    YES
    """.strip().splitlines(keepends=False)
    pos_cases = emotive_cases + yelling_word_cases
    for s in pos_cases:
        s = s.strip()
        # print("testing:", s)
        outcries = find_outcry_matches(s)
        # print("\tfound outcries:", outcries)
        assert len(outcries) > 0, s

# test_find_outcry_matcher()

def validate_outcries(r: DatasetSampleDict) -> bool:
    outcries = find_outcry_texts(r['unnormalized_text'])
    return len(outcries) == 0

def validate_all(r: DatasetSampleDict) -> bool:
    return (validate_sample_text_length(r) and
            validate_sample_word_count(r) and
            validate_sample_audio_length(r) and
            validate_sample_conditioning_length(r) and
            validate_outcries(r))


training_validators: List[Callable[[DatasetSampleDict], bool]] = [
    validate_sample_text_length,
    validate_sample_word_count,
    validate_sample_audio_length,
    validate_outcries,]
conditioning_validators: List[Callable[[DatasetSampleDict], bool]] = [
    validate_outcries,
    validate_sample_word_count,
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