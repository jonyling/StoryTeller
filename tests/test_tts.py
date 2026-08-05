from unittest.mock import MagicMock

from pipeline.tts import ElevenLabsNarrationSynthesizer


def test_synthesize_joins_audio_chunks_and_passes_expected_params():
    fake_client = MagicMock()
    fake_client.text_to_speech.convert.return_value = iter([b"chunk1", b"chunk2"])
    synthesizer = ElevenLabsNarrationSynthesizer(fake_client)

    audio_bytes = synthesizer.synthesize(
        text="Once upon a time...",
        voice_id="voice-123",
        style_description="warm and slow",
        language="Mandarin",
    )

    assert audio_bytes == b"chunk1chunk2"
    args, kwargs = fake_client.text_to_speech.convert.call_args
    assert args[0] == "voice-123"
    # style_description is accepted by the signature but must not be prepended
    # into the text sent to the API (see comment in tts.py for why).
    assert kwargs["text"] == "Once upon a time..."
    assert kwargs["model_id"] == "eleven_v3"
    assert kwargs["language_code"] == "zh"
