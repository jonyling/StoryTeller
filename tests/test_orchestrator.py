import io
from unittest.mock import patch

from PIL import Image
from pydub import AudioSegment

from pipeline.orchestrator import run_pipeline
from pipeline.story_gen import StoryResult, StorySentence
from pipeline.tts import NarrationAudio


def _sample_images(count=2):
    return [Image.new("RGB", (10, 10), color="white") for _ in range(count)]


def _silence_mp3_bytes(duration_ms):
    segment = AudioSegment.silent(duration=duration_ms)
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()


def _fake_voice_bytes(duration_ms=2000):
    segment = AudioSegment.silent(duration=duration_ms, frame_rate=16000)
    buffer = io.BytesIO()
    segment.export(buffer, format="wav")
    return buffer.getvalue()


def _char_timestamps(text, seconds_per_char=0.05):
    characters = list(text)
    starts = [i * seconds_per_char for i in range(len(characters))]
    ends = [(i + 1) * seconds_per_char for i in range(len(characters))]
    return characters, starts, ends


class FakeStoryGenerator:
    def generate(self, images, language):
        return StoryResult(sentences=[
            StorySentence(text="Once upon a time.", speaker="narrator", emotion="calm"),
            StorySentence(text="Wow!", speaker="Ember", emotion="excited"),
        ])


class FakeVoiceCloner:
    def clone(self, audio_bytes, name):
        return "voice-123"


class FakeNarrationSynthesizer:
    def __init__(self):
        self.calls = []

    def synthesize_with_timestamps(self, text, voice_id, language):
        self.calls.append((text, voice_id, language))
        characters, starts, ends = _char_timestamps(text)
        total_ms = int(ends[-1] * 1000) if characters else 100
        return NarrationAudio(_silence_mp3_bytes(total_ms), characters, starts, ends)


@patch("pipeline.orchestrator.fetch_ambience_clip")
def test_run_pipeline_without_sfx(mock_fetch_ambience, tmp_path):
    narration_synth = FakeNarrationSynthesizer()

    result = run_pipeline(
        images=_sample_images(),
        voice_bytes=_fake_voice_bytes(),
        language="English",
        enable_sfx=False,
        story_generator=FakeStoryGenerator(),
        voice_cloner=FakeVoiceCloner(),
        narration_synthesizer=narration_synth,
        freesound_api_key="fake-key",
        sfx_cache_dir=str(tmp_path),
    )

    assert result.used_sfx is False
    mock_fetch_ambience.assert_not_called()
    sentences = result.pages[0]["sentences"]
    assert len(sentences) == 2
    assert sentences[0]["text"] == "Once upon a time."
    assert sentences[0]["speaker"] == "narrator"
    assert sentences[0]["emotion"] == "calm"
    assert sentences[0]["pitch"] == 2
    assert sentences[0]["volume"] == 2
    assert sentences[0]["rate"] == 2
    assert sentences[0]["audio_path"]
    assert sentences[1]["emotion"] == "excited"
    assert sentences[1]["pitch"] == 5
    assert sentences[1]["audio_path"]
    assert sentences[0]["audio_path"] != sentences[1]["audio_path"]


@patch("pipeline.orchestrator.fetch_ambience_clip")
def test_run_pipeline_with_sfx_fetches_ambience_per_distinct_emotion(mock_fetch_ambience, tmp_path):
    mock_fetch_ambience.return_value = _silence_mp3_bytes(500)
    narration_synth = FakeNarrationSynthesizer()

    result = run_pipeline(
        images=_sample_images(),
        voice_bytes=_fake_voice_bytes(),
        language="English",
        enable_sfx=True,
        story_generator=FakeStoryGenerator(),
        voice_cloner=FakeVoiceCloner(),
        narration_synthesizer=narration_synth,
        freesound_api_key="fake-key",
        sfx_cache_dir=str(tmp_path),
    )

    assert result.used_sfx is True
    assert mock_fetch_ambience.call_count == 2
    called_moods = {call.args[0] for call in mock_fetch_ambience.call_args_list}
    assert called_moods == {"flowing river", "cheerful sparkle"}
    assert set(result.ambience_by_emotion.keys()) == {"calm", "excited"}


@patch("pipeline.orchestrator.fetch_ambience_clip")
def test_run_pipeline_survives_ambience_fetch_error(mock_fetch_ambience, tmp_path):
    mock_fetch_ambience.side_effect = RuntimeError("Freesound is down")
    narration_synth = FakeNarrationSynthesizer()

    result = run_pipeline(
        images=_sample_images(),
        voice_bytes=_fake_voice_bytes(),
        language="English",
        enable_sfx=True,
        story_generator=FakeStoryGenerator(),
        voice_cloner=FakeVoiceCloner(),
        narration_synthesizer=narration_synth,
        freesound_api_key="fake-key",
        sfx_cache_dir=str(tmp_path),
    )

    assert result.used_sfx is False
    assert result.pages[0]["sentences"][0]["audio_path"]


@patch("pipeline.orchestrator.fetch_ambience_clip")
def test_run_pipeline_reports_progress(mock_fetch_ambience, tmp_path):
    mock_fetch_ambience.return_value = _silence_mp3_bytes(500)
    narration_synth = FakeNarrationSynthesizer()
    progress_messages = []

    run_pipeline(
        images=_sample_images(),
        voice_bytes=_fake_voice_bytes(),
        language="English",
        enable_sfx=True,
        story_generator=FakeStoryGenerator(),
        voice_cloner=FakeVoiceCloner(),
        narration_synthesizer=narration_synth,
        freesound_api_key="fake-key",
        sfx_cache_dir=str(tmp_path),
        on_progress=progress_messages.append,
    )

    assert len(progress_messages) > 1
