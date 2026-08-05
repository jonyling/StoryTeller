import json
from unittest.mock import MagicMock

import pytest
from google.genai.types import Part
from PIL import Image

from pipeline.story_gen import (
    ClaudeStoryGenerator,
    GeminiStoryGenerator,
    GrokStoryGenerator,
    OpenAIStoryGenerator,
    create_story_generator,
)


def _sample_images(count=2):
    return [Image.new("RGB", (10, 10), color="white") for _ in range(count)]


def _payload():
    return {
        "sentences": [
            {
                "text": "Once upon a time, a small fox named Ember lived in the forest.",
                "speaker": "narrator",
                "emotion": "calm",
            },
            {
                "text": "[gasps] I found something amazing!",
                "speaker": "Ember",
                "emotion": "excited",
            },
        ]
    }


def _assert_sentences_match_payload(sentences):
    assert len(sentences) == 2
    assert sentences[0].text == "Once upon a time, a small fox named Ember lived in the forest."
    assert sentences[0].speaker == "narrator"
    assert sentences[0].emotion == "calm"
    assert sentences[1].text == "[gasps] I found something amazing!"
    assert sentences[1].speaker == "Ember"
    assert sentences[1].emotion == "excited"


def test_openai_story_generator_parses_response_and_sends_all_images():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps(_payload())))
    ]
    generator = OpenAIStoryGenerator(fake_client)

    result = generator.generate(_sample_images(3), "English")

    _assert_sentences_match_payload(result.sentences)
    _, kwargs = fake_client.chat.completions.create.call_args
    content = kwargs["messages"][0]["content"]
    image_blocks = [block for block in content if block["type"] == "image_url"]
    assert len(image_blocks) == 3


def test_openai_story_generator_falls_back_to_neutral_for_unknown_emotion():
    fake_client = MagicMock()
    payload = {"sentences": [{"text": "Hello.", "speaker": "narrator", "emotion": "confused"}]}
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps(payload)))
    ]
    generator = OpenAIStoryGenerator(fake_client)

    result = generator.generate(_sample_images(1), "English")

    assert result.sentences[0].emotion == "neutral"


def test_claude_story_generator_parses_response_and_sends_all_images():
    fake_client = MagicMock()
    fake_client.messages.create.return_value.content = [
        MagicMock(type="thinking", thinking="pondering the story..."),
        MagicMock(type="text", text=json.dumps(_payload())),
    ]
    generator = ClaudeStoryGenerator(fake_client)

    result = generator.generate(_sample_images(2), "Mandarin")

    _assert_sentences_match_payload(result.sentences)
    _, kwargs = fake_client.messages.create.call_args
    content = kwargs["messages"][0]["content"]
    image_blocks = [block for block in content if block["type"] == "image"]
    assert len(image_blocks) == 2
    assert kwargs["thinking"] == {"type": "disabled"}


def test_gemini_story_generator_parses_response_and_sends_all_images():
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value.text = json.dumps(_payload())
    generator = GeminiStoryGenerator(fake_client)

    result = generator.generate(_sample_images(2), "Mandarin")

    _assert_sentences_match_payload(result.sentences)
    _, kwargs = fake_client.models.generate_content.call_args
    image_parts = [item for item in kwargs["contents"] if isinstance(item, Part)]
    assert len(image_parts) == 2
    assert kwargs["config"].response_mime_type == "application/json"


def test_grok_story_generator_parses_response_and_sends_all_images():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps(_payload())))
    ]
    generator = GrokStoryGenerator(fake_client)

    result = generator.generate(_sample_images(4), "English")

    _assert_sentences_match_payload(result.sentences)
    _, kwargs = fake_client.chat.completions.create.call_args
    content = kwargs["messages"][0]["content"]
    image_blocks = [block for block in content if block["type"] == "image_url"]
    assert len(image_blocks) == 4
    assert kwargs["model"] == "grok-2-vision-latest"


def test_create_story_generator_openai():
    generator = create_story_generator("openai", openai_client=MagicMock())
    assert isinstance(generator, OpenAIStoryGenerator)


def test_create_story_generator_claude():
    generator = create_story_generator("claude", anthropic_client=MagicMock())
    assert isinstance(generator, ClaudeStoryGenerator)


def test_create_story_generator_gemini():
    generator = create_story_generator("gemini", gemini_client=MagicMock())
    assert isinstance(generator, GeminiStoryGenerator)


def test_create_story_generator_grok():
    generator = create_story_generator("grok", grok_client=MagicMock())
    assert isinstance(generator, GrokStoryGenerator)


def test_create_story_generator_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown story provider"):
        create_story_generator("nope")
