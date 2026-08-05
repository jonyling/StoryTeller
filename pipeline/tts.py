_LANGUAGE_CODES = {"English": "en", "Mandarin": "zh"}


class ElevenLabsNarrationSynthesizer:
    def __init__(self, client, model_id: str = "eleven_v3"):
        self._client = client
        self._model_id = model_id

    def synthesize(self, text: str, voice_id: str, style_description: str, language: str) -> bytes:
        # style_description is intentionally not sent to the API: eleven_v3 treats
        # bracketed text as short inline audio-tag cues (e.g. "[whispers]"), not a
        # natural-language delivery description, so prepending a full sentence risks
        # it being read aloud literally. Dynamic delivery instead comes from the
        # [whisper]/[gasp]-style inline tags story_gen.py already embeds in the text.
        chunks = self._client.text_to_speech.convert(
            voice_id,
            text=text,
            model_id=self._model_id,
            output_format="mp3_44100_128",
            language_code=_LANGUAGE_CODES.get(language),
        )
        return b"".join(chunks)
