#!/usr/bin/env python3
import signal
import sys
from typing import Literal, Dict, List
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from docopt import docopt
from pathlib import Path
import warnings
import logging

Tag = Literal["neu", "hap", "sad", "ang"]
Emotion = Literal["neutral", "happy", "sad", "angry"]

RENAME: Dict[Tag, Emotion] = {
"neu": "neutral",
"hap": "happy",
"sad": "sad",
"ang": "angry",
}
EmotionFallbacks: Dict[Emotion, List[Emotion]] = {
    "neutral": ["happy", "sad", "angry"],
    "happy": ["neutral", "sad", "angry"],
    "sad": ["neutral", "happy", "angry"],
    "angry": ["neutral", "happy", "sad"],
}


def load_emotion_classifier():

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"^Passing `gradient_checkpointing` to a config initialization is deprecated and will be removed in v5.*")
    lg = logging.getLogger(
        "speechbrain.lobes.models.huggingface_transformers.huggingface")
    lg.disabled = True

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="tmp/emotion", # TODO remove?
    )
    classifier.hparams.label_encoder.expect_len(4)
    return classifier

classifier = load_emotion_classifier()

def _load_normalized_wav(wav_path: Path, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)      # [channels, time]
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.ndim == 2:                        # mix to mono if needed
        wav = wav.mean(dim=0)
    return wav.unsqueeze(0)                  # [1, time]

def _score_emotions(waveform: torch.Tensor) -> torch.Tensor:
    """
    Return the emotion probabilities for the given wav file path.
    """
    lengths = torch.tensor([1.0])
    # Encoder
    x = classifier.mods.wav2vec2(waveform, lengths)
    if "mean_var_norm" in classifier.mods:
        x = classifier.mods.mean_var_norm(x)
    # Heads
    for name, module in classifier.mods.items():
        if name in ("wav2vec2", "mean_var_norm"):
            continue
        x = module(x)
    logits = x.squeeze()  # [classes]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs
    
def _label_emotions(probs: torch.Tensor) -> Dict[Emotion, float]:
    encoder = classifier.hparams.label_encoder
    # Build a list of labels in index order
    labels = [encoder.ind2lab[i] for i in range(len(encoder.ind2lab))]
    # Map each label to its probability
    return {
        RENAME[label]: probs[i].item()
        for i, label in enumerate(labels)
    }

def _label_emotion(probs: torch.Tensor) -> tuple[Emotion, float]:
    """
    Classify a single emotion of a given wav file path with a confidence score.
    """
    # TODO probs has all the labels, not just the top one.
    pred_idx = torch.argmax(probs, dim=-1)
    label = classifier.hparams.label_encoder.decode_ndim(pred_idx)
    confidence = probs[pred_idx].item()
    return RENAME[label], confidence
    
def emotion_from_wav(wav_path: Path) -> tuple[Emotion, float]:
    """
    Classify the emotion of a given wav file path with a confidence score.
    """
    waveform = _load_normalized_wav(wav_path)
    probs = _score_emotions(waveform)
    return _label_emotion(probs)

def emotion_scores_from_wav(wav_path: Path) -> Dict[Emotion, float]:
    """
    Classify the emotions of a given wav file path with confidence scores.
    """
    waveform = _load_normalized_wav(wav_path)
    probs = _score_emotions(waveform)
    return _label_emotions(probs)

def main(args):
    wav_paths = args['<wav>']

    # Process each file
    print(f"{'Emotion'}\t{'Confidence'}\t{'File'}")
    print("-"*55)
    if args['--all']:
        infer = emotion_scores_from_wav
        display = lambda path, label2score: print(path, label2score)
    else:
        infer = emotion_from_wav
        display = lambda path, label_conf: print(f"{label_conf[0]}\t{label_conf[1]:.3f}\t{path}")

    for path in map(Path, wav_paths):
        try:
            result = infer(path)
            display(path, result)
            
        except Exception as e:
            print(f'Error processing "{path}"  ERROR: {e}')

USAGE = """
Usage: audio_emotion.py [options] <wav>...

Options:
    --all Show all emotions with confidence scores.
"""
if __name__ == "__main__":
    try:
        args = docopt(USAGE)
        main(args)
    except KeyboardInterrupt:
        sys.exit(128 + signal.SIGINT)
    except BrokenPipeError:
        sys.exit(128 + signal.SIGPIPE)

