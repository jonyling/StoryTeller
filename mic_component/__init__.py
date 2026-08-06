"""Browser mic recorder that returns audio bytes to Streamlit.

Uses the same getUserMedia device picker as the level meter, unlike
``st.audio_input`` which ignores that selection and often records silence.
"""
from __future__ import annotations

import base64
import os
from typing import Any

import streamlit.components.v1 as components

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")
_recorder = components.declare_component("storyteller_mic_recorder", path=_FRONTEND)


def record_voice(*, key: str | None = None) -> dict[str, Any] | None:
    """Render mic check + recorder. Returns ``{"mime", "data_b64", "peak"}`` or None."""
    return _recorder(key=key, default=None)


def recording_to_wav_bytes(payload: dict[str, Any]) -> tuple[bytes, float]:
    """Decode component payload → WAV bytes + peak (0..1)."""
    import io

    from pydub import AudioSegment

    raw = base64.b64decode(payload["data_b64"])
    mime = (payload.get("mime") or "audio/webm").split(";")[0].strip()
    segment = AudioSegment.from_file(io.BytesIO(raw), format=_mime_to_format(mime))
    segment = segment.set_channels(1)
    if segment.frame_rate != 16000:
        segment = segment.set_frame_rate(16000)
    if segment.sample_width != 2:
        segment = segment.set_sample_width(2)
    peak = float(segment.max) / float(segment.max_possible_amplitude or 1)
    reported = float(payload.get("peak") or 0.0)
    peak = max(peak, reported)
    if 0.02 <= peak < 0.25:
        segment = segment.apply_gain(min(18.0, 20.0 * (0.25 / max(peak, 1e-6))))
        peak = float(segment.max) / float(segment.max_possible_amplitude or 1)
    out = io.BytesIO()
    segment.export(out, format="wav")
    return out.getvalue(), peak


def _mime_to_format(mime: str) -> str:
    if "wav" in mime:
        return "wav"
    if "mpeg" in mime or "mp3" in mime:
        return "mp3"
    if "ogg" in mime:
        return "ogg"
    if "mp4" in mime or "m4a" in mime:
        return "mp4"
    return "webm"
