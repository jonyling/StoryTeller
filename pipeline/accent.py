import base64
import json


class AccentResult:
    def __init__(self, accent_label: str, detected_language: str):
        self.accent_label = accent_label
        self.detected_language = detected_language


_PROMPT = (
    "Listen to this voice sample. Identify: "
    "(1) the speaker's regional accent or language variety in a short phrase "
    "(e.g. 'Singaporean English', 'American English', 'Beijing Mandarin'), and "
    "(2) the primary language being spoken, as exactly one of: 'English' or 'Mandarin'. "
    "Respond as strict JSON with exactly these keys: "
    '{"accent_label": "...", "detected_language": "..."}'
)


class OpenAIAudioAccentDetector:
    def __init__(self, client, model: str = "gpt-4o-audio-preview"):
        self._client = client
        self._model = model

    def detect(self, audio_bytes: bytes, audio_format: str = "wav") -> AccentResult:
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        response = self._client.chat.completions.create(
            model=self._model,
            modalities=["text"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}},
                ],
            }],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content)
        return AccentResult(
            accent_label=payload["accent_label"],
            detected_language=payload["detected_language"],
        )
