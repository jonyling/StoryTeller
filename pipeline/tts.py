_LANGUAGE_CODES = {"English": "en", "Mandarin": "zh"}


class ElevenLabsNarrationSynthesizer:
    def __init__(self, client, model_id: str = "eleven_v3"):
        self._client = client
        self._model_id = model_id

    def synthesize(self, text: str, voice_id: str, style_description: str, language: str) -> bytes:
        full_text = f"[{style_description}]\n{text}" if style_description else text
        chunks = self._client.text_to_speech.convert(
            voice_id,
            text=full_text,
            model_id=self._model_id,
            output_format="mp3_44100_128",
            language_code=_LANGUAGE_CODES.get(language),
        )
        return b"".join(chunks)
