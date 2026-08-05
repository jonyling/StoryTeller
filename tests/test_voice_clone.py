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
    assert filename == "voice_sample.bin"
    assert content == b"sample-audio-bytes"
    assert content_type == "application/octet-stream"


def test_clone_labels_wav_input_correctly():
    fake_client = MagicMock()
    fake_client.voices.ivc.create.return_value.voice_id = "voice-123"
    cloner = ElevenLabsVoiceCloner(fake_client)

    wav_bytes = b"RIFF...." + b"\x00" * 20

    cloner.clone(wav_bytes, "My Story Voice")

    _, kwargs = fake_client.voices.ivc.create.call_args
    filename, content, content_type = kwargs["files"][0]
    assert filename == "voice_sample.wav"
    assert content == wav_bytes
    assert content_type == "audio/wav"


def test_clone_labels_mp3_input_correctly():
    fake_client = MagicMock()
    fake_client.voices.ivc.create.return_value.voice_id = "voice-123"
    cloner = ElevenLabsVoiceCloner(fake_client)

    mp3_bytes = b"ID3...." + b"\x00" * 20

    cloner.clone(mp3_bytes, "My Story Voice")

    _, kwargs = fake_client.voices.ivc.create.call_args
    filename, content, content_type = kwargs["files"][0]
    assert filename == "voice_sample.mp3"
    assert content == mp3_bytes
    assert content_type == "audio/mpeg"


def test_clone_labels_mp3_frame_header_input_correctly():
    fake_client = MagicMock()
    fake_client.voices.ivc.create.return_value.voice_id = "voice-123"
    cloner = ElevenLabsVoiceCloner(fake_client)

    mp3_bytes = b"\xff\xfb...." + b"\x00" * 20

    cloner.clone(mp3_bytes, "My Story Voice")

    _, kwargs = fake_client.voices.ivc.create.call_args
    filename, content, content_type = kwargs["files"][0]
    assert filename == "voice_sample.mp3"
    assert content_type == "audio/mpeg"


def test_clone_falls_back_to_generic_binary_for_unrecognized_bytes():
    fake_client = MagicMock()
    fake_client.voices.ivc.create.return_value.voice_id = "voice-123"
    cloner = ElevenLabsVoiceCloner(fake_client)

    unknown_bytes = b"not-a-real-audio-format"

    cloner.clone(unknown_bytes, "My Story Voice")

    _, kwargs = fake_client.voices.ivc.create.call_args
    filename, content, content_type = kwargs["files"][0]
    assert filename == "voice_sample.bin"
    assert content_type == "application/octet-stream"
