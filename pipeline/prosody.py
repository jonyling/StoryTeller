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
    """Apply pitch/tempo/gain to a WAV (or other) clip; return WAV bytes."""
    import numpy as np
    import soundfile as sf

    pitch_st, tempo, gain_db = map_dsp_params(pitch, rate, volume)
    if abs(pitch_st) < 0.05 and abs(tempo - 1.0) < 0.01 and abs(gain_db) < 0.01:
        # Still normalize container to WAV for consistent st.audio playback
        try:
            data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=False)
        except Exception:
            return wav_bytes
        out = io.BytesIO()
        sf.write(out, np.asarray(data, dtype=np.float32), sr, format="WAV")
        return out.getvalue()

    import librosa

    data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=False)
    wav = np.asarray(data, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if abs(pitch_st) > 0.05:
        wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=pitch_st)
    if abs(tempo - 1.0) > 0.01:
        wav = librosa.effects.time_stretch(wav, rate=tempo)
    if abs(gain_db) > 0.01:
        wav = wav * (10 ** (gain_db / 20.0))

    peak = float(np.max(np.abs(wav)) or 1.0)
    if peak > 0.99:
        wav = wav * (0.99 / peak)

    out = io.BytesIO()
    sf.write(out, wav, sr, format="WAV")
    return out.getvalue()
