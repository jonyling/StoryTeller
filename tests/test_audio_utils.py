import io

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from pipeline.audio_utils import (
    get_duration_seconds,
    mix_ambience_under_narration,
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


def _tone_wav_bytes(duration_ms: int, freq: int = 440) -> bytes:
    from pydub.generators import Sine

    buffer = io.BytesIO()
    # -20dBFS leaves headroom so overlaying two tones doesn't clip and skew
    # the dBFS comparisons below — real narration/ambience recordings have
    # similar headroom; full-scale sines would be an unrealistic worst case.
    tone = Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-20)
    tone.export(buffer, format="wav")
    return buffer.getvalue()


def test_mix_ambience_under_narration_returns_narration_unchanged_when_no_ambience():
    narration = _tone_wav_bytes(1000)

    mixed = mix_ambience_under_narration(narration, None)

    assert mixed == narration


def test_mix_ambience_under_narration_preserves_narration_duration():
    narration = _tone_wav_bytes(3000, freq=440)
    ambience = _tone_wav_bytes(500, freq=220)  # shorter than narration — must loop

    mixed = mix_ambience_under_narration(narration, ambience)

    assert get_duration_seconds(mixed) == pytest.approx(3.0, abs=0.05)


def test_mix_ambience_under_narration_ducks_loud_ambience_under_narration_level():
    narration = _tone_wav_bytes(2000, freq=440)
    loud_ambience_segment = Sine(220).to_audio_segment(duration=2000).apply_gain(-5)
    buf = io.BytesIO()
    loud_ambience_segment.export(buf, format="wav")
    loud_ambience_bytes = buf.getvalue()
    narration_segment = AudioSegment.from_file(io.BytesIO(narration))

    mixed_bytes = mix_ambience_under_narration(narration, loud_ambience_bytes)
    mixed_segment = AudioSegment.from_file(io.BytesIO(mixed_bytes))

    # Even though the raw ambience clip started well louder than narration,
    # the mix must still duck it down to a background level relative to the
    # narration's own loudness, not just apply a fixed attenuation that
    # assumes a particular starting loudness.
    assert mixed_segment.dBFS > narration_segment.dBFS
    assert mixed_segment.dBFS < narration_segment.dBFS + 3


def test_mix_ambience_under_narration_normalizes_regardless_of_ambience_starting_loudness():
    # Regression: a fixed absolute attenuation (e.g. -18dB) assumes the raw
    # ambience clip already sits near narration's loudness. Real Freesound
    # ambience clips are often already quiet (~-25dBFS) relative to XTTS
    # narration (~-20 to -25dBFS) — attenuating further by a fixed amount
    # pushed them down to ~-43dBFS, effectively inaudible. Ducking must be
    # relative to how loud the narration actually is: a very quiet raw
    # ambience clip and a very loud one should land at close to the same
    # final mix level once both are ducked under the SAME narration.
    narration = _tone_wav_bytes(2000, freq=440)

    def tone_wav(gain_db):
        buf = io.BytesIO()
        Sine(220).to_audio_segment(duration=2000).apply_gain(gain_db).export(buf, format="wav")
        return buf.getvalue()

    mixed_from_quiet = AudioSegment.from_file(
        io.BytesIO(mix_ambience_under_narration(narration, tone_wav(-40)))
    )
    mixed_from_loud = AudioSegment.from_file(
        io.BytesIO(mix_ambience_under_narration(narration, tone_wav(-5)))
    )

    assert mixed_from_quiet.dBFS == pytest.approx(mixed_from_loud.dBFS, abs=0.3)


def test_mix_ambience_under_narration_trims_ambience_longer_than_narration():
    narration = _tone_wav_bytes(1000)
    ambience = _tone_wav_bytes(5000, freq=220)

    mixed = mix_ambience_under_narration(narration, ambience)

    assert get_duration_seconds(mixed) == pytest.approx(1.0, abs=0.05)
