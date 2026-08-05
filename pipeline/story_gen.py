import base64
import io
import json

from google.genai import types as genai_types

_LANGUAGE_NAMES = {"English": "English", "Mandarin": "Mandarin Chinese"}
VALID_EMOTIONS = {"angry", "excited", "sad", "calm", "neutral"}


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


def _parse_story_payload(raw_json: str) -> StoryResult:
    payload = json.loads(raw_json)
    sentences = [
        StorySentence(
            text=item["text"],
            speaker=item["speaker"],
            emotion=_normalize_emotion(item["emotion"]),
        )
        for item in payload["sentences"]
    ]
    return StoryResult(sentences)


def _generate_via_openai_compatible_chat(client, model: str, images, language: str) -> StoryResult:
    content = [{"type": "text", "text": _build_prompt(language)}]
    for image in images:
        encoded = base64.b64encode(_image_to_jpeg_bytes(image)).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        })
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        max_tokens=1500,
    )
    return _parse_story_payload(response.choices[0].message.content)


class OpenAIStoryGenerator:
    def __init__(self, client, model: str = "gpt-4o"):
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
            max_tokens=1500,
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
