#!/usr/bin/env python3
import sys
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

def load_and_preprocess(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)      # [channels, time]
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.ndim == 2:                        # mix to mono if needed
        wav = wav.mean(dim=0)
    return wav.unsqueeze(0)                  # [1, time]

def classify_file(classifier, wav_path: str):
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
    pred_idx = torch.argmax(probs, dim=-1)
    label = classifier.hparams.label_encoder.decode_ndim(pred_idx)
    confidence = probs[pred_idx].item()
    return label, confidence

def main(wav_paths):
    # Load model once
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir="tmp/emotion"
    )

    # Process each file
    print(f"{'File':<30}  {'Emotion':<10}  {'Confidence'}")
    print("-"*55)
    for path in wav_paths:
        try:
            label, conf = classify_file(classifier, path)
            print(f"{path:<30}  {label:<10}  {conf:.3f}")
        except Exception as e:
            print(f"{path:<30}  ERROR: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <wav1> [wav2] [wav3] ...", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1:])

