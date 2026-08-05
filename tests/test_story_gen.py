import json
from unittest.mock import MagicMock

import pytest
from PIL import Image

from pipeline.story_gen import (
    ClaudeStoryGenerator,
    OpenAIStoryGenerator,
    create_story_generator,
)


def _sample_images(count=2):
    return [Image.new("RGB", (10, 10), color="white") for _ in range(count)]


def _payload():
    return {
        "story": "Once upon a time...",
        "sfx_mood": "gentle rain",
        "style_description": "warm, slow, soothing",
    }


def test_openai_story_generator_parses_response_and_sends_all_images():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps(_payload())))
    ]
    generator = OpenAIStoryGenerator(fake_client)

    result = generator.generate(_sample_images(3), "English")

    assert result.story_text == "Once upon a time..."
    assert result.sfx_mood == "gentle rain"
    assert result.tts_style_description == "warm, slow, soothing"
    _, kwargs = fake_client.chat.completions.create.call_args
    content = kwargs["messages"][0]["content"]
    image_blocks = [block for block in content if block["type"] == "image_url"]
    assert len(image_blocks) == 3


def test_claude_story_generator_parses_response_and_sends_all_images():
    fake_client = MagicMock()
    fake_client.messages.create.return_value.content = [
        MagicMock(text=json.dumps(_payload()))
    ]
    generator = ClaudeStoryGenerator(fake_client)

    result = generator.generate(_sample_images(2), "Mandarin")

    assert result.story_text == "Once upon a time..."
    _, kwargs = fake_client.messages.create.call_args
    content = kwargs["messages"][0]["content"]
    image_blocks = [block for block in content if block["type"] == "image"]
    assert len(image_blocks) == 2


def test_create_story_generator_openai():
    generator = create_story_generator("openai", openai_client=MagicMock())
    assert isinstance(generator, OpenAIStoryGenerator)


def test_create_story_generator_claude():
    generator = create_story_generator("claude", anthropic_client=MagicMock())
    assert isinstance(generator, ClaudeStoryGenerator)


def test_create_story_generator_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown story provider"):
        create_story_generator("nope")
