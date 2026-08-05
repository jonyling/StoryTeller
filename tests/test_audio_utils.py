import io

import pytest
from pydub import AudioSegment

from pipeline.audio_utils import get_duration_seconds, validate_duration
from pipeline.errors import ValidationError


def _silence_wav_bytes(duration_ms: int) -> bytes:
    segment = AudioSegment.silent(duration=duration_ms, frame_rate=16000)
    buffer = io.BytesIO()
    segment.export(buffer, format="wav")
    return buffer.getvalue()


def test_get_duration_seconds_matches_generated_length():
    audio_bytes = _silence_wav_bytes(2500)
    assert get_duration_seconds(audio_bytes) == pytest.approx(2.5, abs=0.05)


def test_validate_duration_raises_when_too_short():
    audio_bytes = _silence_wav_bytes(1000)
    with pytest.raises(ValidationError, match="at least"):
        validate_duration(audio_bytes, min_seconds=5, max_seconds=300)


def test_validate_duration_raises_when_too_long():
    audio_bytes = _silence_wav_bytes(2000)
    with pytest.raises(ValidationError, match="at most"):
        validate_duration(audio_bytes, min_seconds=0, max_seconds=1)


def test_validate_duration_passes_within_range():
    audio_bytes = _silence_wav_bytes(2000)
    duration = validate_duration(audio_bytes, min_seconds=1, max_seconds=5)
    assert duration == pytest.approx(2.0, abs=0.05)
