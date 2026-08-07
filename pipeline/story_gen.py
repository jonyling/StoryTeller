import base64
import io
import json
import re

from google.genai import types as genai_types

from pipeline.errors import PipelineError
from pipeline.prosody import strip_delivery_tags

_LANGUAGE_NAMES = {"English": "English", "Mandarin": "Mandarin Chinese"}
VALID_EMOTIONS = {"angry", "excited", "sad", "calm", "neutral"}
_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
# For long stories, the model can run its own "speaker"/"emotion" schema
# into a sentence's "text" value instead of stopping at the real sentence —
# the JSON itself still parses fine (single quotes need no escaping inside
# a double-quoted JSON string), so this can't be caught at the JSON-parse
# level; it has to be cleaned up after parsing.
_LEAKED_JSON_TAIL = re.compile(
    r"""\s*,\s*['"]speaker['"]\s*:\s*['"][^'"]*['"]\s*"""
    r"""(?:,\s*['"]emotion['"]\s*:\s*['"][^'"]*['"]?)?\s*\}?\s*$""",
    re.IGNORECASE,
)


class StorySentence:
    def __init__(self, text: str, speaker: str, emotion: str):
        self.text = text
        self.speaker = speaker
        self.emotion = emotion


class StoryResult:
    def __init__(self, sentences):
        self.sentences = sentences


def _build_prompt(language: str) -> str:
    language_name = _LANGUAGE_NAMES[language]
    return (
        f"Look at these sequential storybook images. Write a short bedtime-appropriate "
        f"story (under 200 words total) in {language_name} that follows what happens "
        f"across the images, broken into individual sentences.\n\n"
        f"For each sentence, provide:\n"
        f'- "text": the sentence itself. Where the narration should shift tone, include '
        f"inline delivery tags like [whispers], [gasps], [laughs] directly in the text.\n"
        f'- "speaker": "narrator" for descriptive/narration lines, or a simple name you '
        f"give to a character depicted in the images for lines that represent that "
        f"character's dialogue or direct thoughts/actions. Keep names consistent across "
        f"the story.\n"
        f'- "emotion": exactly one of "angry", "excited", "sad", "calm", "neutral" — '
        f"whichever best matches this sentence's mood.\n\n"
        f"Respond as strict JSON with exactly this shape: "
        f'{{"sentences": [{{"text": "...", "speaker": "...", "emotion": "..."}}, ...]}}'
    )


def _image_to_jpeg_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def _normalize_emotion(value: str) -> str:
    return value if value in VALID_EMOTIONS else "neutral"


def _coerce_message_text(message) -> str | None:
    """Normalize OpenAI-style message content (str | list | None) to a string."""
    if message is None:
        return None
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or "")
            else:
                parts.append(getattr(part, "text", None) or "")
        content = "\n".join(p for p in parts if p)
    if isinstance(content, str):
        content = content.strip()
        return content or None
    return None


def _parse_story_payload(raw_json: str) -> StoryResult:
    if raw_json is None or not str(raw_json).strip():
        raise PipelineError(
            "Vision story model returned empty content. Check OPENAI_API_KEY / "
            "STORY_PROVIDER, model access (needs vision, e.g. gpt-4o), and that the "
            "key is not rate-limited or content-filtered."
        )
    text = str(raw_json).strip()
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # Last resort: first {...} blob
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                raise PipelineError(
                    f"Vision story model returned non-JSON text: {text[:240]!r}"
                ) from exc
        else:
            raise PipelineError(
                f"Vision story model returned non-JSON text: {text[:240]!r}"
            ) from exc

    sentences_raw = payload.get("sentences") if isinstance(payload, dict) else None
    if not sentences_raw:
        raise PipelineError(
            f"Vision story JSON missing 'sentences': {str(payload)[:240]!r}"
        )
    sentences = [
        StorySentence(
            text=_LEAKED_JSON_TAIL.sub("", item["text"]),
            speaker=item["speaker"],
            emotion=_normalize_emotion(item["emotion"]),
        )
        for item in sentences_raw
    ]
    return StoryResult(_merge_tag_only_sentences(sentences))


def _merge_tag_only_sentences(sentences: list) -> list:
    """Fold a sentence whose text is ONLY delivery tags (e.g. the model is
    told to put [whispers]/[gasps] inline within a sentence's text, but
    sometimes emits one as its own list entry instead) into the previous
    sentence. Left standalone it has no speakable content — strip_delivery_tags
    empties it, so it becomes a silent phantom TTS clip whose raw "[whispers]"
    text still shows up as if it were a real sentence in the UI.
    """
    merged: list = []
    for sentence in sentences:
        if not strip_delivery_tags(sentence.text) and merged:
            merged[-1].text = f"{merged[-1].text} {sentence.text}".strip()
            continue
        merged.append(sentence)
    return merged


def _generate_via_openai_compatible_chat(client, model: str, images, language: str) -> StoryResult:
    content = [{"type": "text", "text": _build_prompt(language)}]
    for image in images:
        encoded = base64.b64encode(_image_to_jpeg_bytes(image)).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        })

    def _call(*, use_json_object: bool):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            # 1500 was too tight for longer stories — near the ceiling the
            # model can lose track of proper JSON structure and spill its
            # own "speaker"/"emotion" schema into a sentence's text instead
            # of stopping cleanly (see _LEAKED_JSON_TAIL above).
            "max_tokens": 4000,
        }
        if use_json_object:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    try:
        response = _call(use_json_object=True)
    except Exception as exc:
        # Some gateways reject response_format with vision — retry plain.
        try:
            response = _call(use_json_object=False)
        except Exception:
            raise PipelineError(f"Vision story API call failed ({model}): {exc}") from exc

    if not getattr(response, "choices", None):
        raise PipelineError(f"Vision story API returned no choices ({model}).")

    choice = response.choices[0]
    message = choice.message
    text = _coerce_message_text(message)
    if not text:
        refusal = getattr(message, "refusal", None)
        finish = getattr(choice, "finish_reason", None)
        raise PipelineError(
            f"Vision story model returned empty content ({model}). "
            f"finish_reason={finish!r}, refusal={refusal!r}. "
            "Confirm the key can call a vision chat model (gpt-4o / gpt-4o-mini)."
        )
    return _parse_story_payload(text)



class OpenAIStoryGenerator:
    def __init__(self, client, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model

    def generate(self, images, language: str) -> StoryResult:
        return _generate_via_openai_compatible_chat(self._client, self._model, images, language)


class GrokStoryGenerator:
    """xAI's Grok API is OpenAI-compatible, so this reuses the same chat-completions shape."""

    def __init__(self, client, model: str = "grok-2-vision-latest"):
        self._client = client
        self._model = model

    def generate(self, images, language: str) -> StoryResult:
        return _generate_via_openai_compatible_chat(self._client, self._model, images, language)


class ClaudeStoryGenerator:
    def __init__(self, client, model: str = "claude-sonnet-5"):
        self._client = client
        self._model = model

    def generate(self, images, language: str) -> StoryResult:
        prompt = _build_prompt(language) + " Respond with ONLY the JSON object, no other text."
        content = [{"type": "text", "text": prompt}]
        for image in images:
            encoded = base64.b64encode(_image_to_jpeg_bytes(image)).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded},
            })
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4000,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": content}],
        )
        text_block = next(block for block in response.content if block.type == "text")
        return _parse_story_payload(text_block.text)


class GeminiStoryGenerator:
    def __init__(self, client, model: str = "gemini-3.5-flash"):
        self._client = client
        self._model = model

    def generate(self, images, language: str) -> StoryResult:
        contents = [_build_prompt(language)]
        for image in images:
            contents.append(
                genai_types.Part.from_bytes(data=_image_to_jpeg_bytes(image), mime_type="image/jpeg")
            )
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return _parse_story_payload(response.text)


def create_story_generator(
    provider: str,
    openai_client=None,
    anthropic_client=None,
    gemini_client=None,
    grok_client=None,
):
    if provider == "openai":
        if openai_client is None:
            raise ValueError("openai_client is required for the 'openai' story provider")
        return OpenAIStoryGenerator(openai_client)
    if provider == "claude":
        if anthropic_client is None:
            raise ValueError("anthropic_client is required for the 'claude' story provider")
        return ClaudeStoryGenerator(anthropic_client)
    if provider == "gemini":
        if gemini_client is None:
            raise ValueError("gemini_client is required for the 'gemini' story provider")
        return GeminiStoryGenerator(gemini_client)
    if provider == "grok":
        if grok_client is None:
            raise ValueError("grok_client is required for the 'grok' story provider")
        return GrokStoryGenerator(grok_client)
    raise ValueError(f"Unknown story provider: {provider!r}")
