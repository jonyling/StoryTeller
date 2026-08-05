import io
import re

from pydub import AudioSegment

from pipeline.errors import ValidationError

_BRACKET_TAG_PATTERN = re.compile(r"\[[^\]]*\]")


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
