"""Continue-story: LLM next beat + helpers to append narrated sentences."""
from __future__ import annotations

import json

from pipeline.errors import PipelineError
from pipeline.prosody import EMOTION_DSP_DEFAULTS, NARRATOR_DSP
from pipeline.story_gen import StoryResult, StorySentence, VALID_EMOTIONS, _JSON_FENCE
from pipeline.theatre import RuleBasedTheatreAdapter, TheatreLine

_CONTINUE_SYSTEM = (
    "You are a children's bedtime storyteller. Continue an existing short story "
    "with 2–4 new sentences only. Keep the same characters and tone. "
    "Do not restart or summarize the whole plot. Bedtime-safe. "
    "Respond as strict JSON: "
    '{"sentences":[{"text":"...","speaker":"narrator|Name","emotion":"angry|excited|sad|calm|neutral"},...]}'
)


def _normalize_emotion(value: str) -> str:
    return value if value in VALID_EMOTIONS else "neutral"


def transcript_from_pages(pages) -> str:
    lines = []
    for page in pages or []:
        for sent in page.get("sentences", []):
            speaker = sent.get("speaker") or "narrator"
            text = (sent.get("text") or "").strip()
            if text:
                lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def parse_continue_payload(raw_json: str) -> list[StorySentence]:
    if raw_json is None or not str(raw_json).strip():
        raise PipelineError("Continue-story model returned empty content.")
    text = str(raw_json).strip()
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise PipelineError(f"Continue-story returned non-JSON: {text[:200]!r}")
        payload = json.loads(text[start : end + 1])
    items = payload.get("sentences") if isinstance(payload, dict) else None
    if not items:
        raise PipelineError("Continue-story JSON missing sentences.")
    sentences = []
    for item in items[:4]:
        sentences.append(
            StorySentence(
                text=(item.get("text") or "").strip(),
                speaker=(item.get("speaker") or "narrator").strip() or "narrator",
                emotion=_normalize_emotion(item.get("emotion") or "neutral"),
            )
        )
    sentences = [s for s in sentences if s.text]
    if not sentences:
        raise PipelineError("Continue-story produced no usable sentences.")
    return sentences


def generate_next_beat(client, pages, *, language: str = "English", model: str = "gpt-4o-mini") -> list[StorySentence]:
    """Ask an OpenAI-compatible chat model for the next 2–4 story sentences."""
    so_far = transcript_from_pages(pages) or "(story just beginning)"
    user = (
        f"language={language}\n\n"
        f"STORY SO FAR:\n{so_far}\n\n"
        "Write the next 2–4 sentences as JSON only."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CONTINUE_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=600,
        )
    except Exception as exc:
        # Retry without response_format for picky gateways
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _CONTINUE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=600,
            )
        except Exception:
            raise PipelineError(f"Continue-story API failed: {exc}") from exc

    if not getattr(response, "choices", None):
        raise PipelineError("Continue-story API returned no choices.")
    content = response.choices[0].message.content
    return parse_continue_payload(content)


def narrate_sentences(
    sentences: list[StorySentence],
    *,
    voice_bytes: bytes,
    language: str,
    voice_cloner,
    narration_synthesizer,
    theatre_adapter=None,
) -> list[dict]:
    """Theatre + clone + TTS → sentence dicts with audio_path bytes."""
    adapter = theatre_adapter or RuleBasedTheatreAdapter()
    story = StoryResult(sentences)
    lines = adapter.adapt(story.sentences)
    voice_id = voice_cloner.clone(voice_bytes, "StoryTeller Continue")
    if callable(getattr(narration_synthesizer, "synthesize_sentences", None)):
        clips = narration_synthesizer.synthesize_sentences(lines, voice_id, language)
    else:
        raise PipelineError("Narration backend cannot synthesize per-sentence clips.")

    out = []
    for line, clip in zip(lines, clips):
        dsp = dict(EMOTION_DSP_DEFAULTS.get(line.emotion, EMOTION_DSP_DEFAULTS["neutral"]))
        if (line.speaker or "").lower() == "narrator":
            dsp = dict(NARRATOR_DSP)
        out.append(
            {
                "text": line.text,
                "speaker": line.speaker,
                "emotion": line.emotion,
                "stage_direction": line.stage_direction,
                "pitch": line.pitch if line.pitch is not None else dsp["pitch"],
                "volume": line.volume if line.volume is not None else dsp["volume"],
                "rate": line.rate if line.rate is not None else dsp["rate"],
                "audio_path": clip,
            }
        )
    return out


def append_beat_page(pages: list, sentence_dicts: list[dict]) -> list:
    """Return a new pages list with one extra page for the continued beat."""
    pages = list(pages or [])
    next_page = 1
    if pages:
        next_page = max(int(p.get("page", 1)) for p in pages) + 1
    pages.append({"page": next_page, "sentences": list(sentence_dicts)})
    return pages


def speak_reply(
    text: str,
    *,
    voice_bytes: bytes,
    language: str,
    voice_cloner,
    narration_synthesizer,
) -> bytes:
    """Synthesize a Companion reply in the narrator voice (full text, chunked for XTTS)."""
    from pipeline.xtts_backend import XTTS_MAX_CHARS, _chunk_text_for_xtts, _concat_wav_bytes

    clean = " ".join((text or "").split())
    if not clean:
        return b""
    voice_id = voice_cloner.clone(voice_bytes, "StoryTeller Companion")
    chunks = _chunk_text_for_xtts(clean, max_chars=XTTS_MAX_CHARS)
    lines = [
        TheatreLine(
            id=i,
            speaker="narrator",
            stage_direction="warmly, to the child",
            text=chunk,
            speak_text=chunk,
            emotion="calm",
            pitch=3,
            rate=3,
            volume=3,
        )
        for i, chunk in enumerate(chunks)
    ]
    if callable(getattr(narration_synthesizer, "synthesize_sentences", None)):
        clips = narration_synthesizer.synthesize_sentences(lines, voice_id, language)
        return _concat_wav_bytes([c for c in clips if c])
    raise PipelineError("Narration backend cannot synthesize Companion replies.")
