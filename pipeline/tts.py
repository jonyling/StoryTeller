"""Narration synthesizers.

- ``XTTSNarrationSynthesizer`` (default) lives in ``pipeline.xtts_backend``
  and exposes ``synthesize_sentences`` (per-line, EN/ZH).
- ``ElevenLabsNarrationSynthesizer`` below is the paid backup using one timed take.
"""
import base64

_LANGUAGE_CODES = {"English": "en", "Mandarin": "zh"}


class NarrationAudio:
    def __init__(self, audio_bytes: bytes, characters, character_start_times_seconds, character_end_times_seconds):
        self.audio_bytes = audio_bytes
        self.characters = characters
        self.character_start_times_seconds = character_start_times_seconds
        self.character_end_times_seconds = character_end_times_seconds


class ElevenLabsNarrationSynthesizer:
    """Paid backup: one ``eleven_v3`` call with character timestamps."""

    def __init__(self, client, model_id: str = "eleven_v3"):
        self._client = client
        self._model_id = model_id

    def synthesize_with_timestamps(self, text: str, voice_id: str, language: str) -> NarrationAudio:
        response = self._client.text_to_speech.convert_with_timestamps(
            voice_id,
            text=text,
            model_id=self._model_id,
            output_format="mp3_44100_128",
            language_code=_LANGUAGE_CODES.get(language),
        )
        return NarrationAudio(
            audio_bytes=base64.b64decode(response.audio_base_64),
            characters=response.alignment.characters,
            character_start_times_seconds=response.alignment.character_start_times_seconds,
            character_end_times_seconds=response.alignment.character_end_times_seconds,
        )

