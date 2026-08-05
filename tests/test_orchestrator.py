import io
from unittest.mock import patch

import fitz
from pydub import AudioSegment

from pipeline.accent import AccentResult
from pipeline.orchestrator import run_pipeline
from pipeline.story_gen import StoryResult


def _make_pdf_bytes(num_pages=2) -> bytes:
    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def _silence_mp3_bytes(duration_ms):
    segment = AudioSegment.silent(duration=duration_ms)
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()


class FakeStoryGenerator:
    def generate(self, images, language):
        return StoryResult(
            story_text="Once upon a time",
            sfx_mood="gentle rain",
            tts_style_description="warm",
        )


class FakeAccentDetector:
    def __init__(self, detected_language="English"):
        self._detected_language = detected_language

    def detect(self, audio_bytes):
        return AccentResult(accent_label="American English", detected_language=self._detected_language)


class FakeVoiceCloner:
    def clone(self, audio_bytes, name):
        return "voice-123"


class FakeNarrationSynthesizer:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice_id, style_description, language):
        self.calls.append((text, voice_id, style_description, language))
        return _silence_mp3_bytes(1000)


@patch("pipeline.orchestrator.fetch_ambience_clip")
def test_run_pipeline_without_sfx(mock_fetch_ambience, tmp_path):
    narration_synth = FakeNarrationSynthesizer()

    result = run_pipeline(
        pdf_bytes=_make_pdf_bytes(),
        voice_bytes=b"fake-voice-sample",
        language="English",
        enable_sfx=False,
        story_generator=FakeStoryGenerator(),
        accent_detector=FakeAccentDetector(detected_language="English"),
        voice_cloner=FakeVoiceCloner(),
        narration_synthesizer=narration_synth,
        freesound_api_key="fake-key",
        sfx_cache_dir=str(tmp_path),
    )

    assert result.story_text == "Once upon a time"
    assert result.used_sfx is False
    mock_fetch_ambience.assert_not_called()
    assert narration_synth.calls[0][2] == "warm"


@patch("pipeline.orchestrator.fetch_ambience_clip")
def test_run_pipeline_with_sfx_fetches_ambience(mock_fetch_ambience, tmp_path):
    mock_fetch_ambience.return_value = _silence_mp3_bytes(500)
    narration_synth = FakeNarrationSynthesizer()

    result = run_pipeline(
        pdf_bytes=_make_pdf_bytes(),
        voice_bytes=b"fake-voice-sample",
        language="English",
        enable_sfx=True,
        story_generator=FakeStoryGenerator(),
        accent_detector=FakeAccentDetector(detected_language="English"),
        voice_cloner=FakeVoiceCloner(),
        narration_synthesizer=narration_synth,
        freesound_api_key="fake-key",
        sfx_cache_dir=str(tmp_path),
    )

    assert result.used_sfx is True
    mock_fetch_ambience.assert_called_once_with("gentle rain", "fake-key", str(tmp_path))


def test_run_pipeline_appends_accent_hint_when_cross_lingual(tmp_path):
    narration_synth = FakeNarrationSynthesizer()

    run_pipeline(
        pdf_bytes=_make_pdf_bytes(),
        voice_bytes=b"fake-voice-sample",
        language="Mandarin",
        enable_sfx=False,
        story_generator=FakeStoryGenerator(),
        accent_detector=FakeAccentDetector(detected_language="English"),
        voice_cloner=FakeVoiceCloner(),
        narration_synthesizer=narration_synth,
        freesound_api_key="fake-key",
        sfx_cache_dir=str(tmp_path),
    )

    style_description_used = narration_synth.calls[0][2]
    assert "American English accent flavor" in style_description_used
