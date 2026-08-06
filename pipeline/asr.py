"""Speech-to-text for Companion voice questions (OpenAI Whisper)."""
from __future__ import annotations

import io

from pipeline.errors import PipelineError


def transcribe_wav_bytes(client, wav_bytes: bytes, *, language: str | None = None) -> str:
    """Transcribe a mono/stereo WAV via OpenAI Whisper.

    ``language`` may be ``English`` / ``Mandarin`` / ISO codes; mapped for Whisper.
    """
    if not wav_bytes:
        raise PipelineError("No audio to transcribe.")
    lang = None
    if language:
        lowered = language.strip().lower()
        if lowered.startswith("en"):
            lang = "en"
        elif lowered.startswith("zh") or "mandarin" in lowered or "chinese" in lowered:
            lang = "zh"
    try:
        buffer = io.BytesIO(wav_bytes)
        buffer.name = "question.wav"
        kwargs = {"model": "whisper-1", "file": buffer}
        if lang:
            kwargs["language"] = lang
        result = client.audio.transcriptions.create(**kwargs)
    except Exception as exc:
        raise PipelineError(f"Whisper transcription failed: {exc}") from exc
    text = (getattr(result, "text", None) or str(result) or "").strip()
    if not text:
        raise PipelineError("Whisper returned empty text — try speaking more clearly.")
    return text
