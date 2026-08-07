"""Basic eval: XTTS model-load and per-sentence synthesis latency.

Reports whatever device pipeline.xtts_backend picks (CUDA if available,
else CPU) -- run this on each machine you care about (CPU dev box vs. a
GPU desktop) and report the device alongside the numbers, since the two
are not comparable.

Usage: .venv/Scripts/python.exe scripts/eval_xtts_latency.py
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("COQUI_TOS_AGREED", "1")

from pipeline.xtts_backend import XTTSNarrationSynthesizer

SENTENCES = [
    "The little fox crept quietly through the moonlit forest.",
    "Suddenly, a twig snapped behind her!",
    "She turned around, heart pounding, to see a small rabbit.",
    "They both laughed with relief and became friends.",
]


class FakeLine:
    def __init__(self, text: str):
        self.text = text
        self.speak_text = text
        self.pitch = 3
        self.rate = 3
        self.volume = 3


def main() -> None:
    synth = XTTSNarrationSynthesizer()

    t0 = time.time()
    synth._ensure_model()
    load_time = time.time() - t0
    print(f"Device: {synth._device}")
    print(f"Model load time: {load_time:.1f}s")

    ref = str(ROOT / "assets" / "voices" / "warm.mp3")
    per_sentence = []
    t_total0 = time.time()
    for i, text in enumerate(SENTENCES, 1):
        line = FakeLine(text)
        t0 = time.time()
        synth.synthesize_sentences([line], ref, "English")
        dt = time.time() - t0
        per_sentence.append(dt)
        print(f"Sentence {i} ({len(text)} chars): {dt:.1f}s")
    total = time.time() - t_total0

    avg = total / len(SENTENCES)
    print(f"\nDevice: {synth._device}")
    print(f"Total synth time for {len(SENTENCES)} sentences: {total:.1f}s (avg {avg:.1f}s/sentence)")
    print(f"Estimated full story (8-12 sentences): {8*avg:.0f}s - {12*avg:.0f}s + {load_time:.0f}s model load")


if __name__ == "__main__":
    main()
