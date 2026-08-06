from unittest.mock import MagicMock

import pytest

from pipeline.errors import PipelineError
from pipeline.story_gen import StoryResult, StorySentence
from pipeline.translate import (
    needs_translation,
    story_needs_translation,
    translate_story_result,
    translate_story_sentences,
)


def test_needs_translation_english_to_mandarin():
    assert needs_translation("Charlie walked into the chocolate factory.", "Mandarin")
    assert story_needs_translation(
        [StorySentence("Hello world.", "narrator", "neutral")],
        "Mandarin",
    )


def test_needs_translation_chinese_to_english():
    assert needs_translation("查理走进了巧克力工厂。", "English")
    assert story_needs_translation(
        [StorySentence("很久以前，有一个男孩。", "narrator", "calm")],
        "English",
    )


def test_needs_translation_skips_when_already_matching():
    assert not needs_translation("查理走进了巧克力工厂。", "Mandarin")
    assert not needs_translation("Charlie walked in.", "English")
    assert not needs_translation("", "Mandarin")
    assert not needs_translation("Hello", "French")


def test_translate_story_sentences_preserves_meta():
    fake = MagicMock()
    fake.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=(
                        '{"sentences":['
                        '{"text":"查理走进了工厂。","speaker":"narrator","emotion":"calm"},'
                        '{"text":"“太神奇了！”威利说。","speaker":"Willy","emotion":"excited"}'
                        "]}"
                    )
                )
            )
        ]
    )
    originals = [
        StorySentence("Charlie walked into the factory.", "narrator", "calm"),
        StorySentence('"How marvelous!" said Willy.', "Willy", "excited"),
    ]
    out = translate_story_sentences(
        fake, originals, target_language="Mandarin", model="gpt-4o-mini"
    )
    assert len(out) == 2
    assert out[0].text == "查理走进了工厂。"
    assert out[0].speaker == "narrator"
    assert out[0].emotion == "calm"
    assert out[1].speaker == "Willy"
    assert out[1].emotion == "excited"
    assert fake.chat.completions.create.called


def test_translate_zh_to_en():
    fake = MagicMock()
    fake.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=(
                        '{"sentences":[{"text":"Charlie smiled.","speaker":"narrator","emotion":"calm"}]}'
                    )
                )
            )
        ]
    )
    out = translate_story_sentences(
        fake,
        [StorySentence("查理笑了。", "narrator", "calm")],
        target_language="English",
    )
    assert out[0].text == "Charlie smiled."
    assert out[0].speaker == "narrator"


def test_translate_skips_when_already_matching():
    fake = MagicMock()
    assert (
        translate_story_sentences(
            fake,
            [StorySentence("查理笑了。", "narrator", "calm")],
            target_language="Mandarin",
        )[0].text
        == "查理笑了。"
    )
    assert (
        translate_story_sentences(
            fake,
            [StorySentence("Charlie smiled.", "narrator", "calm")],
            target_language="English",
        )[0].text
        == "Charlie smiled."
    )
    fake.chat.completions.create.assert_not_called()


def test_translate_story_result_and_empty_raises():
    fake = MagicMock()
    fake.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=""))]
    )
    with pytest.raises(PipelineError):
        translate_story_sentences(
            fake,
            [StorySentence("Once upon a time.", "narrator", "neutral")],
            target_language="Mandarin",
        )

    fake.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"sentences":[{"text":"很久以前。","speaker":"narrator","emotion":"neutral"}]}'
                )
            )
        ]
    )
    result = translate_story_result(
        fake,
        StoryResult([StorySentence("Once upon a time.", "narrator", "neutral")]),
        target_language="Mandarin",
    )
    assert result.sentences[0].text == "很久以前。"
