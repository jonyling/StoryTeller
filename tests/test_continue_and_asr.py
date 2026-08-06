from unittest.mock import MagicMock

import pytest

from pipeline.asr import transcribe_wav_bytes
from pipeline.companion import append_to_canon, new_session
from pipeline.continue_story import (
    append_beat_page,
    parse_continue_payload,
    transcript_from_pages,
)
from pipeline.errors import PipelineError
from pipeline.story_gen import StorySentence


def _pages():
    return [
        {
            "page": 1,
            "sentences": [
                {"text": "Ember lived in a quiet valley.", "speaker": "narrator", "emotion": "calm"},
                {"text": "A storm arrived suddenly.", "speaker": "narrator", "emotion": "angry"},
            ],
        }
    ]


def test_transcribe_wav_bytes_uses_whisper():
    fake = MagicMock()
    fake.audio.transcriptions.create.return_value = MagicMock(text="Where is Ember?")
    text = transcribe_wav_bytes(fake, b"RIFF....", language="English")
    assert text == "Where is Ember?"
    kwargs = fake.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "whisper-1"
    assert kwargs["language"] == "en"


def test_transcribe_empty_raises():
    with pytest.raises(PipelineError):
        transcribe_wav_bytes(MagicMock(), b"", language="English")


def test_parse_continue_payload():
    raw = '{"sentences":[{"text":"Ember smiled.","speaker":"narrator","emotion":"calm"},' \
          '{"text":"Rain slowed.","speaker":"narrator","emotion":"neutral"}]}'
    sents = parse_continue_payload(raw)
    assert len(sents) == 2
    assert sents[0].text == "Ember smiled."


def test_parse_continue_payload_empty_raises():
    with pytest.raises(PipelineError):
        parse_continue_payload(None)


def test_append_beat_page_and_canon():
    pages = _pages()
    beat = [
        {
            "text": "Sunlight returned.",
            "speaker": "narrator",
            "emotion": "excited",
            "audio_path": b"x",
        }
    ]
    new_pages = append_beat_page(pages, beat)
    assert len(new_pages) == 2
    assert new_pages[-1]["page"] == 2
    assert new_pages[-1]["sentences"][0]["text"] == "Sunlight returned."

    session = new_session(pages)
    session.advance_to(1)
    append_to_canon(session, beat, page_no=2)
    assert session.canon[-1]["text"] == "Sunlight returned."
    assert session.story_position == 2


def test_transcript_from_pages():
    text = transcript_from_pages(_pages())
    assert "Ember lived" in text
    assert "storm" in text
