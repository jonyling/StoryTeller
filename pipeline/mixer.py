import io
import typing

from pydub import AudioSegment

SFX_ATTENUATION_DB = -18


def mix_narration_with_ambience(
    narration_bytes: bytes, ambience_bytes: typing.Optional[bytes]
) -> bytes:
    narration = AudioSegment.from_file(io.BytesIO(narration_bytes))
    if not ambience_bytes:
        return _export_mp3(narration)

    ambience = AudioSegment.from_file(io.BytesIO(ambience_bytes))
    ambience = _loop_to_length(ambience, len(narration))
    ambience = ambience + SFX_ATTENUATION_DB
    mixed = narration.overlay(ambience)
    return _export_mp3(mixed)


def _loop_to_length(segment: AudioSegment, target_length_ms: int) -> AudioSegment:
    if len(segment) == 0:
        return segment
    if len(segment) < target_length_ms:
        loops_required = (target_length_ms // len(segment)) + 1
        segment = segment * loops_required
    return segment[:target_length_ms]


def _export_mp3(segment: AudioSegment) -> bytes:
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()
