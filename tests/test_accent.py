import base64
import json
from unittest.mock import MagicMock

from pipeline.accent import OpenAIAudioAccentDetector


def test_detect_parses_response_and_encodes_audio():
    fake_client = MagicMock()
    payload = {"accent_label": "Singaporean English", "detected_language": "English"}
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps(payload)))
    ]
    detector = OpenAIAudioAccentDetector(fake_client)

    result = detector.detect(b"fake-audio-bytes", audio_format="wav")

    assert result.accent_label == "Singaporean English"
    assert result.detected_language == "English"
    _, kwargs = fake_client.chat.completions.create.call_args
    content = kwargs["messages"][0]["content"]
    audio_block = next(block for block in content if block["type"] == "input_audio")
    assert audio_block["input_audio"]["format"] == "wav"
    assert audio_block["input_audio"]["data"] == base64.b64encode(b"fake-audio-bytes").decode("utf-8")
