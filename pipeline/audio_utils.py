import io

from pydub import AudioSegment

from pipeline.errors import ValidationError


def get_duration_seconds(audio_bytes: bytes) -> float:
    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    return len(segment) / 1000.0


def validate_duration(audio_bytes: bytes, min_seconds: float, max_seconds: float) -> float:
    duration = get_duration_seconds(audio_bytes)
    if duration < min_seconds:
        raise ValidationError(
            f"Voice sample is {duration:.1f}s long; it needs to be at least "
            f"{min_seconds:.0f}s for voice cloning."
        )
    if duration > max_seconds:
        raise ValidationError(
            f"Voice sample is {duration:.1f}s long; it needs to be at most "
            f"{max_seconds:.0f}s for voice cloning."
        )
    return duration
