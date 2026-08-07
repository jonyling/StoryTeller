import io

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from pipeline.prosody import apply_prosody_to_wav_bytes, rate_to_speed


def _tone_wav_bytes(duration_ms: int = 1000, freq: int = 440) -> bytes:
    buffer = io.BytesIO()
    Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-20).export(buffer, format="wav")
    return buffer.getvalue()


def test_rate_to_speed_neutral_is_identity():
    assert rate_to_speed(3) == pytest.approx(1.0)


def test_rate_to_speed_matches_map_dsp_params_tempo_formula():
    # Same ±6%-per-step, clamped-to-±12% formula as map_dsp_params' tempo —
    # rate_to_speed just exposes it standalone so XTTS's native `speed` kwarg
    # can use it without going through post-hoc DSP.
    assert rate_to_speed(4) == pytest.approx(1.06)
    assert rate_to_speed(2) == pytest.approx(0.94)
    assert rate_to_speed(5) == pytest.approx(1.12)
    assert rate_to_speed(1) == pytest.approx(0.88)


def test_apply_prosody_never_changes_duration_regardless_of_pitch_or_rate():
    # librosa's pitch_shift/time_stretch (phase vocoder) gave XTTS output a
    # watery/echoey artifact even at mild settings. Tempo is now handled
    # natively via XTTS's own `speed` kwarg at synthesis time (see
    # rate_to_speed / xtts_backend.py); pitch has no artifact-free
    # equivalent, so it's no longer shaped here at all. This function should
    # now only ever touch loudness, never duration.
    original = _tone_wav_bytes(1000)
    original_duration = AudioSegment.from_file(io.BytesIO(original)).duration_seconds

    for pitch in (1, 2, 3, 4, 5):
        for rate in (1, 2, 3, 4, 5):
            shaped = apply_prosody_to_wav_bytes(original, pitch=pitch, rate=rate, volume=3)
            shaped_duration = AudioSegment.from_file(io.BytesIO(shaped)).duration_seconds
            assert shaped_duration == pytest.approx(original_duration, abs=0.05)


def test_apply_prosody_volume_three_is_near_identity_level():
    original = _tone_wav_bytes()
    original_dbfs = AudioSegment.from_file(io.BytesIO(original)).dBFS

    shaped = apply_prosody_to_wav_bytes(original, pitch=3, rate=3, volume=3)
    shaped_dbfs = AudioSegment.from_file(io.BytesIO(shaped)).dBFS

    assert shaped_dbfs == pytest.approx(original_dbfs, abs=0.5)


def test_apply_prosody_raises_volume_increases_loudness():
    original = _tone_wav_bytes()
    original_dbfs = AudioSegment.from_file(io.BytesIO(original)).dBFS

    shaped = apply_prosody_to_wav_bytes(original, pitch=3, rate=3, volume=5)
    shaped_dbfs = AudioSegment.from_file(io.BytesIO(shaped)).dBFS

    assert shaped_dbfs > original_dbfs + 2


def test_apply_prosody_lowers_volume_decreases_loudness():
    original = _tone_wav_bytes()
    original_dbfs = AudioSegment.from_file(io.BytesIO(original)).dBFS

    shaped = apply_prosody_to_wav_bytes(original, pitch=3, rate=3, volume=1)
    shaped_dbfs = AudioSegment.from_file(io.BytesIO(shaped)).dBFS

    assert shaped_dbfs < original_dbfs - 2
