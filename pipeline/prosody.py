"""Deterministic prosody DSP (storyteller §6 / Havoc 06b)."""
from __future__ import annotations

import io
import re

_BRACKET_TAG_PATTERN = re.compile(r"\[[^\]]*\]")

EMOTION_DSP_DEFAULTS = {
    # Keep pitch near 3 — librosa pitch_shift on XTTS makes a watery/roomy echo,
    # especially on calm third-person narration.
    "angry": {"pitch": 3, "volume": 5, "rate": 4},
    "excited": {"pitch": 4, "volume": 4, "rate": 4},
    "sad": {"pitch": 3, "volume": 2, "rate": 2},
    "calm": {"pitch": 3, "volume": 3, "rate": 2},
    "neutral": {"pitch": 3, "volume": 3, "rate": 3},
}

# Narrator prose: no pitch shift (avoids the "echo chamber" artifact).
NARRATOR_DSP = {"pitch": 3, "volume": 3, "rate": 3}


def map_dsp_params(pitch: int, rate: int, volume: int) -> tuple[float, float, float]:
    # Milder than before: ±1.0 st and ±6% tempo (was ±1.5 st / ±8%).
    pitch_st = float(max(-2.0, min(2.0, (int(pitch) - 3) * 1.0)))
    tempo = float(1.0 + (int(rate) - 3) * 0.06)
    tempo = max(0.88, min(1.12, tempo))
    gain_db = float((int(volume) - 3) * 2.0)
    return pitch_st, tempo, gain_db


def rate_to_speed(rate: int) -> float:
    """Tempo for a rate DSP value, for XTTS's native `speed` synthesis kwarg.

    Same formula as map_dsp_params' tempo — exposed standalone so tempo can
    be set at generation time (no post-hoc time-stretch artifact) instead of
    via apply_prosody_to_wav_bytes.
    """
    _, tempo, _ = map_dsp_params(3, rate, 3)
    return tempo


def strip_delivery_tags(text: str) -> str:
    """Remove inline [whispers]/[gasps] tags so XTTS does not read them aloud."""
    cleaned = _BRACKET_TAG_PATTERN.sub(" ", text or "")
    return " ".join(cleaned.split())


def apply_prosody_to_wav_bytes(
    wav_bytes: bytes,
    pitch: int = 3,
    rate: int = 3,
    volume: int = 3,
) -> bytes:
    """Apply gain to a WAV (or other) clip; return WAV bytes.

    pitch/rate are accepted for call-site compatibility but no longer shape
    the audio here: librosa's pitch_shift/time_stretch (phase vocoder) gave
    XTTS output a watery/echoey artifact even at the milder ±1st/±6% range
    this used to use. Tempo is now set natively via XTTS's own `speed` kwarg
    at synthesis time (see rate_to_speed / xtts_backend.py) — no post-hoc
    resampling artifact. Pitch has no artifact-free equivalent available, so
    emotional pitch variation was dropped rather than kept at that cost.
    """
    import numpy as np
    import soundfile as sf

    _, _, gain_db = map_dsp_params(pitch, rate, volume)
    data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=False)
    wav = np.asarray(data, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if abs(gain_db) > 0.01:
        wav = wav * (10 ** (gain_db / 20.0))
        peak = float(np.max(np.abs(wav)) or 1.0)
        if peak > 0.99:
            wav = wav * (0.99 / peak)

    out = io.BytesIO()
    sf.write(out, wav, sr, format="WAV")
    return out.getvalue()
