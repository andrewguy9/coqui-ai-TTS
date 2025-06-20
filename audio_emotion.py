#!/usr/bin/env python3
import signal
import sys
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from docopt import docopt

#TODO Lets standardize the audio loading and preprocessing.
def load_and_preprocess(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)      # [channels, time]
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.ndim == 2:                        # mix to mono if needed
        wav = wav.mean(dim=0)
    return wav.unsqueeze(0)                  # [1, time]

def get_emotion_classifier():
    # Load model once
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="tmp/emotion" # TODO remove?
    )

    # TODO classify_file should take the waveform, and we can have a helper to load it/normalize it.
    def classify_file(wav_path: str):
        waveform = load_and_preprocess(wav_path)  # [1, T]
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
        # TODO probs has all the labels, not just the top one.
        pred_idx = torch.argmax(probs, dim=-1)
        label = classifier.hparams.label_encoder.decode_ndim(pred_idx)
        confidence = probs[pred_idx].item()
        return label, confidence
    
    return classify_file

def main(args):
    wav_paths = args['<wav>']

    classify_file = get_emotion_classifier()

    # Process each file
    print(f"{'Emotion'}\t{'Confidence'}\t{'File'}")
    print("-"*55)
    for path in wav_paths:
        try:
            label, conf = classify_file(path)
            print(f"{label}\t{conf:.3f}\t{path:<30}")
        except Exception as e:
            print(f'Error processing "{path}"  ERROR: {e}')

USAGE = """
Usage: audio_emotion.py <wav>...
"""
if __name__ == "__main__":
    try:
        args = docopt(USAGE)
        main(args)
    except KeyboardInterrupt:
        sys.exit(128 + signal.SIGINT)
    except BrokenPipeError:
        sys.exit(128 + signal.SIGPIPE)

