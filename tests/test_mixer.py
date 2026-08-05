import io

from pydub import AudioSegment

from pipeline.mixer import _loop_to_length, mix_narration_with_ambience


def _to_mp3_bytes(segment: AudioSegment) -> bytes:
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()


def test_loop_to_length_loops_short_segment_to_target():
    short_segment = AudioSegment.silent(duration=300)
    result = _loop_to_length(short_segment, target_length_ms=1000)
    assert len(result) == 1000


def test_loop_to_length_trims_long_segment_to_target():
    long_segment = AudioSegment.silent(duration=2000)
    result = _loop_to_length(long_segment, target_length_ms=500)
    assert len(result) == 500


def test_mix_narration_with_ambience_matches_narration_length():
    narration_bytes = _to_mp3_bytes(AudioSegment.silent(duration=3000))
    ambience_bytes = _to_mp3_bytes(AudioSegment.silent(duration=800))

    mixed_bytes = mix_narration_with_ambience(narration_bytes, ambience_bytes)
    mixed = AudioSegment.from_file(io.BytesIO(mixed_bytes))

    assert abs(len(mixed) - 3000) < 100


def test_mix_narration_with_ambience_returns_dry_narration_when_no_ambience():
    narration_bytes = _to_mp3_bytes(AudioSegment.silent(duration=1500))

    mixed_bytes = mix_narration_with_ambience(narration_bytes, None)
    mixed = AudioSegment.from_file(io.BytesIO(mixed_bytes))

    assert abs(len(mixed) - 1500) < 100
