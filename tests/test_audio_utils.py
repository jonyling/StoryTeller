import io

import pytest
from pydub import AudioSegment

from pipeline.audio_utils import (
    get_duration_seconds,
    slice_audio_by_sentences,
    validate_duration,
)
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


def _char_timestamps(text: str, seconds_per_char: float = 0.1):
    characters = list(text)
    starts = [i * seconds_per_char for i in range(len(characters))]
    ends = [(i + 1) * seconds_per_char for i in range(len(characters))]
    return characters, starts, ends


def _to_mp3_bytes(segment: AudioSegment) -> bytes:
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()


def test_slice_audio_by_sentences_splits_on_exact_text_match():
    text = "Hello world. Goodbye now."
    characters, starts, ends = _char_timestamps(text)
    audio_bytes = _to_mp3_bytes(AudioSegment.silent(duration=int(ends[-1] * 1000)))

    clips = slice_audio_by_sentences(
        audio_bytes, characters, starts, ends, ["Hello world.", "Goodbye now."]
    )

    assert len(clips) == 2
    assert get_duration_seconds(clips[0]) == pytest.approx(1.2, abs=0.15)
    assert get_duration_seconds(clips[1]) == pytest.approx(1.2, abs=0.2)


def test_slice_audio_by_sentences_falls_back_to_bracket_stripped_match():
    text = "Wow!"
    characters, starts, ends = _char_timestamps(text)
    audio_bytes = _to_mp3_bytes(AudioSegment.silent(duration=int(ends[-1] * 1000)))

    clips = slice_audio_by_sentences(audio_bytes, characters, starts, ends, ["[gasps] Wow!"])

    assert len(clips) == 1
    assert get_duration_seconds(clips[0]) == pytest.approx(0.4, abs=0.1)


def test_slice_audio_by_sentences_falls_back_when_text_not_found():
    text = "abc"
    characters, starts, ends = _char_timestamps(text)
    audio_bytes = _to_mp3_bytes(AudioSegment.silent(duration=int(ends[-1] * 1000)))

    clips = slice_audio_by_sentences(
        audio_bytes, characters, starts, ends, ["completely different text"]
    )

    assert len(clips) == 1
    assert isinstance(clips[0], bytes)
    assert len(clips[0]) > 0
