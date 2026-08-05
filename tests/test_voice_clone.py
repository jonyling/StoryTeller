from unittest.mock import MagicMock

from pipeline.voice_clone import ElevenLabsVoiceCloner


def test_clone_returns_voice_id_and_sends_audio_file():
    fake_client = MagicMock()
    fake_client.voices.ivc.create.return_value.voice_id = "voice-123"
    cloner = ElevenLabsVoiceCloner(fake_client)

    voice_id = cloner.clone(b"sample-audio-bytes", "My Story Voice")

    assert voice_id == "voice-123"
    _, kwargs = fake_client.voices.ivc.create.call_args
    assert kwargs["name"] == "My Story Voice"
    filename, content, content_type = kwargs["files"][0]
    assert filename == "voice_sample.wav"
    assert content == b"sample-audio-bytes"
    assert content_type == "audio/wav"
