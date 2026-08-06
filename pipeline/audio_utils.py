import io
import re

from pydub import AudioSegment

from pipeline.errors import ValidationError

_BRACKET_TAG_PATTERN = re.compile(r"\[[^\]]*\]")

# How far below the narration's OWN measured loudness ambience should sit.
# A fixed absolute attenuation (the -18dB this replaces, carried over from
# the now-removed pipeline/mixer.py) assumes the raw ambience clip already
# sits near narration's loudness. Real Freesound ambience clips are often
# already quiet (~-25dBFS) relative to XTTS narration (~-20 to -25dBFS), so
# a further fixed -18dB attenuation pushed ambience down to ~-43dBFS —
# effectively inaudible. Ducking relative to narration's actual level keeps
# it a consistent amount below the voice regardless of how loud either
# clip started out.
_AMBIENCE_RELATIVE_DB = -12
# Fallback target when narration is digital silence (dBFS == -inf) and there
# is no useful "below narration" reference to duck relative to.
_AMBIENCE_SILENT_NARRATION_FALLBACK_DBFS = -25.0


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


def slice_audio_by_sentences(
    audio_bytes: bytes,
    characters,
    character_start_times_seconds,
    character_end_times_seconds,
    sentence_texts,
):
    """Split one narration audio into one clip per sentence using ElevenLabs'
    character-level alignment. Locates each sentence by searching for its text within
    the alignment's reconstructed string, preferring an exact match, falling back to a
    bracket-tag-stripped match (alignment may not preserve inline delivery tags like
    [gasps]), and finally to a positional heuristic slice rather than raising, so one
    unmatched sentence doesn't break playback for the rest of the story.
    """
    reconstructed = "".join(characters)
    last_index = len(character_end_times_seconds) - 1
    cursor = 0
    ranges = []
    for sentence_text in sentence_texts:
        start_idx = reconstructed.find(sentence_text, cursor)
        search_text = sentence_text
        if start_idx == -1:
            search_text = _BRACKET_TAG_PATTERN.sub("", sentence_text).strip()
            start_idx = reconstructed.find(search_text, cursor)
        if start_idx == -1:
            start_idx = cursor
            end_idx = min(cursor + len(search_text), last_index)
        else:
            end_idx = start_idx + len(search_text) - 1
        end_idx = min(max(end_idx, start_idx), last_index)
        ranges.append((character_start_times_seconds[start_idx], character_end_times_seconds[end_idx]))
        cursor = end_idx + 1

    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    clips = []
    for start_s, end_s in ranges:
        clip = segment[int(start_s * 1000):int(end_s * 1000) + 1]
        buffer = io.BytesIO()
        clip.export(buffer, format="mp3")
        clips.append(buffer.getvalue())
    return clips


def mix_ambience_under_narration(narration_bytes: bytes, ambience_bytes: bytes | None) -> bytes:
    """Loop/trim ambience to narration's length, duck it, and overlay under it.

    Returns WAV bytes (the format every per-sentence clip already uses).
    With no ambience, narration is returned unchanged.
    """
    if not ambience_bytes:
        return narration_bytes

    narration = AudioSegment.from_file(io.BytesIO(narration_bytes))
    ambience = _loop_to_length(AudioSegment.from_file(io.BytesIO(ambience_bytes)), len(narration))

    narration_dbfs = narration.dBFS
    if narration_dbfs == float("-inf"):
        narration_dbfs = _AMBIENCE_SILENT_NARRATION_FALLBACK_DBFS
    target_dbfs = narration_dbfs + _AMBIENCE_RELATIVE_DB
    if ambience.dBFS != float("-inf"):
        ambience = ambience.apply_gain(target_dbfs - ambience.dBFS)

    mixed = narration.overlay(ambience)

    buffer = io.BytesIO()
    mixed.export(buffer, format="wav")
    return buffer.getvalue()


def _loop_to_length(segment: AudioSegment, target_length_ms: int) -> AudioSegment:
    if len(segment) == 0 or target_length_ms <= 0:
        return segment
    if len(segment) < target_length_ms:
        loops_required = (target_length_ms // len(segment)) + 1
        segment = segment * loops_required
    return segment[:target_length_ms]
