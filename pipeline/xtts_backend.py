"""Local free voice cloning via Coqui XTTS-v2 (Havoc path).

ElevenLabs classes remain in voice_clone.py / tts.py as paid backup.
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import threading
from pathlib import Path

from pipeline.prosody import apply_prosody_to_wav_bytes, strip_delivery_tags

_LANGUAGE_CODES = {"English": "en", "Mandarin": "zh"}

# Soft bounds for local XTTS (not ElevenLabs' 60–300s IVC window)
XTTS_MIN_REF_SECONDS = 6.0
XTTS_MAX_REF_SECONDS = 300.0
XTTS_TRIM_SECONDS = 12.0
# XTTS-v2 warns / truncates English above ~250 chars — keep chunks under that.
XTTS_MAX_CHARS = 220


def _audio_bytes_to_wav_file(audio_bytes: bytes, dest: Path, *, trim_s: float = XTTS_TRIM_SECONDS) -> Path:
    """Decode upload → mono WAV, trimmed to ~trim_s of speech for XTTS ref."""
    from pydub import AudioSegment

    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    segment = segment.set_channels(1)
    # Prefer a mid-clip window if long (often cleaner than absolute start)
    total_ms = len(segment)
    want_ms = int(trim_s * 1000)
    if total_ms > want_ms * 2:
        start = max(0, (total_ms // 2) - (want_ms // 2))
        segment = segment[start : start + want_ms]
    elif total_ms > want_ms:
        segment = segment[:want_ms]
    dest.parent.mkdir(parents=True, exist_ok=True)
    segment.export(str(dest), format="wav")
    return dest


def _chunk_text_for_xtts(text: str, max_chars: int = XTTS_MAX_CHARS) -> list[str]:
    """Split ONLY when a single sentence exceeds XTTS's ~250-char limit.

    Board cards stay one full sentence; these chunks are synthesized and
    crossfaded into one clip so the listener still hears a continuous line.
    Prefer semicolon / em-dash before commas; never mid-word if avoidable.
    """
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Already a single sentence (caller should pass one). Soft wraps only.
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        cut = -1
        for sep in ("; ", " — ", " – ", ", ", " "):
            pos = window.rfind(sep)
            if pos >= max_chars // 3:
                cut = pos + len(sep)
                break
        if cut < 0:
            cut = max_chars
        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _concat_wav_bytes(parts: list[bytes]) -> bytes:
    from pydub import AudioSegment

    clean = [p for p in parts if p]
    if not clean:
        return b""
    if len(clean) == 1:
        return clean[0]
    combined = AudioSegment.from_file(io.BytesIO(clean[0]), format="wav")
    for raw in clean[1:]:
        nxt = AudioSegment.from_file(io.BytesIO(raw), format="wav")
        # Tight crossfade — internal XTTS limit chunks should feel like one line.
        overlap = min(60, len(combined) // 5, len(nxt) // 5)
        if overlap > 8:
            combined = combined.append(nxt, crossfade=overlap)
        else:
            combined += nxt
    out = io.BytesIO()
    combined.export(out, format="wav")
    return out.getvalue()


class XTTSVoiceCloner:
    """Writes a trimmed local speaker reference; returns its path as voice_id."""

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = Path(cache_dir or tempfile.mkdtemp(prefix="storyteller_xtts_"))

    def clone(self, audio_bytes: bytes, name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (name or "voice"))[:40]
        dest = self._cache_dir / f"{safe}_ref.wav"
        _audio_bytes_to_wav_file(audio_bytes, dest)
        return str(dest)


class XTTSNarrationSynthesizer:
    """Per-sentence XTTS synthesis (EN + ZH). Lazy-loads one GPU/CPU model."""

    def __init__(self, device: str | None = None):
        self._device = device
        self._tts = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._tts is not None:
            return
        with self._lock:
            if self._tts is not None:
                return
            try:
                from TTS.api import TTS as XTTS
            except ImportError as exc:
                raise RuntimeError(
                    "coqui-tts is not installed. Run: "
                    "pip install 'coqui-tts' 'transformers>=4.46,<5' soundfile librosa"
                ) from exc
            import torch

            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tts = XTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            self._device = device

    def _synth_one(self, text: str, ref: str, lang: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        try:
            self._tts.tts_to_file(
                text=text,
                file_path=out_path,
                speaker_wav=ref,
                language=lang,
                split_sentences=True,
            )
            return Path(out_path).read_bytes()
        finally:
            try:
                Path(out_path).unlink(missing_ok=True)
            except OSError:
                pass

    def synthesize_sentences(self, lines, voice_id: str, language: str, on_progress=None) -> list[bytes]:
        """`lines` are TheatreLine-like objects with speak_text/text and pitch/rate/volume."""
        if on_progress:
            on_progress("Loading XTTS model onto GPU (one-time; can take 1–2 min)…")
        self._ensure_model()
        lang = _LANGUAGE_CODES.get(language, "en")
        ref = str(voice_id)
        if not Path(ref).exists():
            raise FileNotFoundError(f"XTTS speaker reference missing: {ref}")

        total = len(lines)
        clips: list[bytes] = []
        for idx, line in enumerate(lines, start=1):
            text = strip_delivery_tags(
                getattr(line, "speak_text", None) or getattr(line, "text", "") or ""
            )
            if on_progress:
                preview = (text[:48] + "…") if len(text) > 48 else text
                on_progress(f"XTTS {idx}/{total} on {self._device}: {preview}")
            if not text:
                clips.append(b"")
                continue

            chunks = _chunk_text_for_xtts(text)
            chunk_wavs = []
            for c_i, chunk in enumerate(chunks, start=1):
                if on_progress and len(chunks) > 1:
                    on_progress(
                        f"XTTS {idx}/{total} (part {c_i}/{len(chunks)}) on {self._device}"
                    )
                chunk_wavs.append(self._synth_one(chunk, ref, lang))
            dry = _concat_wav_bytes(chunk_wavs)
            shaped = apply_prosody_to_wav_bytes(
                dry,
                pitch=int(getattr(line, "pitch", 3)),
                rate=int(getattr(line, "rate", 3)),
                volume=int(getattr(line, "volume", 3)),
            )
            clips.append(shaped)
        if on_progress:
            on_progress(f"XTTS done — {total}/{total} lines synthesized")
        return clips
