import base64
from unittest.mock import MagicMock

from pipeline.tts import ElevenLabsNarrationSynthesizer


def test_synthesize_with_timestamps_decodes_audio_and_returns_alignment():
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.audio_base_64 = base64.b64encode(b"fake-audio-bytes").decode("ascii")
    fake_response.alignment.characters = ["H", "i"]
    fake_response.alignment.character_start_times_seconds = [0.0, 0.1]
    fake_response.alignment.character_end_times_seconds = [0.1, 0.2]
    fake_client.text_to_speech.convert_with_timestamps.return_value = fake_response
    synthesizer = ElevenLabsNarrationSynthesizer(fake_client)

    result = synthesizer.synthesize_with_timestamps(text="Hi", voice_id="voice-123", language="Mandarin")

    assert result.audio_bytes == b"fake-audio-bytes"
    assert result.characters == ["H", "i"]
    assert result.character_start_times_seconds == [0.0, 0.1]
    assert result.character_end_times_seconds == [0.1, 0.2]
    args, kwargs = fake_client.text_to_speech.convert_with_timestamps.call_args
    assert args[0] == "voice-123"
    assert kwargs["text"] == "Hi"
    assert kwargs["model_id"] == "eleven_v3"
    assert kwargs["language_code"] == "zh"
