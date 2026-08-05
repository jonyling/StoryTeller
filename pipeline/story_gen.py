import base64
import io
import json

_LANGUAGE_NAMES = {"English": "English", "Mandarin": "Mandarin Chinese"}


class StoryResult:
    def __init__(self, story_text: str, sfx_mood: str, tts_style_description: str):
        self.story_text = story_text
        self.sfx_mood = sfx_mood
        self.tts_style_description = tts_style_description


def _build_prompt(language: str) -> str:
    language_name = _LANGUAGE_NAMES[language]
    return (
        f"Look at these sequential storybook images. Write a short bedtime-appropriate "
        f"story (under 200 words) in {language_name} that follows what happens across "
        f"the images. Add inline delivery tags like [whispers], [gasps], [laughs] where "
        f"the narration should shift tone. Also suggest one background ambience mood "
        f"keyword for the whole story (e.g. 'gentle rain', 'cozy fireplace', 'forest wind') "
        f"and a short natural-language description of how the narrator should deliver the "
        f"story overall (pace, warmth, energy). "
        f"Respond as strict JSON with exactly these keys: "
        f'{{"story": "...", "sfx_mood": "...", "style_description": "..."}}'
    )


def _image_to_jpeg_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


class OpenAIStoryGenerator:
    def __init__(self, client, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    def generate(self, images, language: str) -> StoryResult:
        content = [{"type": "text", "text": _build_prompt(language)}]
        for image in images:
            encoded = base64.b64encode(_image_to_jpeg_bytes(image)).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_tokens=700,
        )
        payload = json.loads(response.choices[0].message.content)
        return StoryResult(
            story_text=payload["story"],
            sfx_mood=payload["sfx_mood"],
            tts_style_description=payload["style_description"],
        )


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
            max_tokens=700,
            messages=[{"role": "user", "content": content}],
        )
        payload = json.loads(response.content[0].text)
        return StoryResult(
            story_text=payload["story"],
            sfx_mood=payload["sfx_mood"],
            tts_style_description=payload["style_description"],
        )


def create_story_generator(provider: str, openai_client=None, anthropic_client=None):
    if provider == "openai":
        if openai_client is None:
            raise ValueError("openai_client is required for the 'openai' story provider")
        return OpenAIStoryGenerator(openai_client)
    if provider == "claude":
        if anthropic_client is None:
            raise ValueError("anthropic_client is required for the 'claude' story provider")
        return ClaudeStoryGenerator(anthropic_client)
    raise ValueError(f"Unknown story provider: {provider!r}")
