"""Bench harness for QuantumFlow's transcription params — run BEFORE touching
flow.py's decode path. Answers: does one full-audio pass fit the ~5-8s budget,
and which model/beam/hotwords/VAD config gives the best accuracy inside it?

Usage:
    py -3.12 bench.py

Looks for samples/*.wav with a sibling samples/*.txt holding the exact ground
truth. Record samples via flow.py's SAVE_AUDIO=True flag, or any 16kHz mono wav.
Never commit samples/ — it's gitignored, this repo is public, and it's Bruno's
voice.
"""
import glob
import os
import time
import wave

import numpy as np
from faster_whisper import WhisperModel

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

# The QuantumFlow dictionary, used identically to flow.py's vocab_hint().
DICTIONARY = ["Wispr Flow", "QuantumFlow", "Ollama", "Claude", "hotkey"]
HOTWORDS = ", ".join(DICTIONARY) + "."


def wer(ref, hyp):
    """Word-level Levenshtein edit distance / len(ref). Correct on edge cases,
    unlike difflib.SequenceMatcher (longest-match-first != minimal edit distance,
    which can flip close rankings). Normalize before scoring or you measure
    punctuation style, not recognition."""
    def norm(s):
        return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).split()
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def load_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def load_samples():
    samples = []
    for wav in sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.wav"))):
        txt = wav[:-4] + ".txt"
        if not os.path.exists(txt):
            print(f"  skip {os.path.basename(wav)} — no matching .txt ground truth")
            continue
        truth = open(txt, encoding="utf-8").read().strip()
        samples.append((os.path.basename(wav), load_wav(wav), truth))
    return samples


# (label, model_size, transcribe_kwargs) — condition_on_previous_text and
# hotwords are what the one-pass rewrite depends on; vad_filter on/off and
# beam_size are the open questions this bench exists to close.
# temperature matches flow.py's transcribe() exactly (adversarial review
# finding #4: bench used to omit it, so every run silently used the library
# default [0,.2,.4,.6,.8,1.0] instead of production's truncated ladder —
# the cited WER/wall numbers didn't strictly correspond to what ships).
TEMP = [0.0, 0.2, 0.4, 0.6]
CONFIGS = [
    ("base b1 novad hot",      "base",  dict(beam_size=1, vad_filter=False, hotwords=HOTWORDS, condition_on_previous_text=True, temperature=TEMP)),
    ("base b5 novad hot",      "base",  dict(beam_size=5, vad_filter=False, hotwords=HOTWORDS, condition_on_previous_text=True, temperature=TEMP)),
    ("base b5 vad hot",        "base",  dict(beam_size=5, vad_filter=True,  hotwords=HOTWORDS, condition_on_previous_text=True, temperature=TEMP)),
    ("base b5 novad noprompt", "base",  dict(beam_size=5, vad_filter=False, condition_on_previous_text=True, temperature=TEMP)),
    ("small b1 novad hot",     "small", dict(beam_size=1, vad_filter=False, hotwords=HOTWORDS, condition_on_previous_text=True, temperature=TEMP)),
]


def main():
    samples = load_samples()
    if not samples:
        print(f"No samples in {SAMPLES_DIR} — record some via flow.py's "
              f"SAVE_AUDIO=True flag, then write a matching .txt ground truth "
              f"beside each .wav.")
        return

    models = {}
    results = []  # (label, wav, seconds, wall, wer)

    for label, size, kwargs in CONFIGS:
        if size not in models:
            print(f"loading {size}...")
            models[size] = WhisperModel(size, device="cpu", compute_type="int8")
        model = models[size]
        for wav_name, audio, truth in samples:
            t0 = time.perf_counter()
            segments, info = model.transcribe(audio, language=None, **kwargs)
            text = " ".join(s.text.strip() for s in segments).strip()
            wall = time.perf_counter() - t0
            err = wer(truth, text)
            results.append((label, wav_name, len(audio) / 16000, wall, err, text))
            print(f"[{label:24}] {wav_name:20} audio={len(audio)/16000:5.1f}s "
                  f"wall={wall:5.1f}s wer={err:5.1%}  {text[:70]}")

    print("\n--- median per config ---")
    for label, _, _ in CONFIGS:
        rows = [r for r in results if r[0] == label]
        walls = sorted(r[3] for r in rows)
        wers = sorted(r[4] for r in rows)
        mid = len(rows) // 2
        print(f"{label:24} median_wall={walls[mid]:5.1f}s  median_wer={wers[mid]:5.1%}  n={len(rows)}")


if __name__ == "__main__":
    main()
