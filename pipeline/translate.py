"""Translate extracted story sentences into the app target language (text PDFs)."""
from __future__ import annotations

import json
import re

from pipeline.errors import PipelineError
from pipeline.story_gen import VALID_EMOTIONS, StoryResult, StorySentence

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_LANGUAGE_NAMES = {"English": "English", "Mandarin": "Mandarin Chinese"}


def needs_translation(text: str, target_language: str) -> bool:
    """True when extracted text script doesn't match the UI target language.

    - Target Mandarin + mostly Latin → translate EN→ZH
    - Target English + mostly Chinese → translate ZH→EN
    """
    target = (target_language or "").strip()
    sample = (text or "").strip()
    if not sample or target not in ("Mandarin", "English"):
        return False
    cjk = len(_CJK_RE.findall(sample))
    latin = len(_LATIN_RE.findall(sample))
    if cjk == 0 and latin == 0:
        return False
    if target == "Mandarin":
        # Already mostly Chinese — skip
        return latin > cjk
    # English target: translate when mostly Chinese
    return cjk > latin


def story_needs_translation(sentences: list[StorySentence], target_language: str) -> bool:
    blob = " ".join(s.text for s in sentences if s and s.text)
    return needs_translation(blob, target_language)


def _normalize_emotion(value: str) -> str:
    return value if value in VALID_EMOTIONS else "neutral"


def _parse_translated_payload(raw_json: str, originals: list[StorySentence]) -> list[StorySentence]:
    if raw_json is None or not str(raw_json).strip():
        raise PipelineError("Translation model returned empty content.")
    text = str(raw_json).strip()
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                raise PipelineError(
                    f"Translation model returned non-JSON text: {text[:240]!r}"
                ) from exc
        else:
            raise PipelineError(
                f"Translation model returned non-JSON text: {text[:240]!r}"
            ) from exc

    rows = payload.get("sentences") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise PipelineError(
            f"Translation JSON missing 'sentences': {str(payload)[:240]!r}"
        )

    out: list[StorySentence] = []
    for i, orig in enumerate(originals):
        if i < len(rows) and isinstance(rows[i], dict):
            item = rows[i]
            translated = (item.get("text") or "").strip()
            if not translated:
                translated = orig.text
            out.append(
                StorySentence(
                    text=translated,
                    speaker=item.get("speaker") or orig.speaker,
                    emotion=_normalize_emotion(item.get("emotion") or orig.emotion),
                )
            )
        else:
            out.append(orig)
    return out


def translate_story_sentences(
    client,
    sentences: list[StorySentence],
    *,
    target_language: str = "Mandarin",
    model: str = "gpt-4o-mini",
) -> list[StorySentence]:
    """Translate sentence texts to ``target_language``; keep speaker/emotion aligned."""
    if not sentences:
        return []
    if not story_needs_translation(sentences, target_language):
        return list(sentences)

    language_name = _LANGUAGE_NAMES.get(target_language, target_language)
    payload_in = {
        "sentences": [
            {"text": s.text, "speaker": s.speaker, "emotion": s.emotion}
            for s in sentences
        ]
    }
    prompt = (
        f"Translate each story sentence into {language_name} for a children's bedtime "
        f"narration. Keep the same number of sentences, in the same order. "
        f"Preserve speaker and emotion fields exactly (do not translate those labels). "
        f"Only translate the \"text\" values. Keep names consistent. "
        f"Respond as strict JSON: "
        f'{{"sentences": [{{"text": "...", "speaker": "...", "emotion": "..."}}, ...]}}\n\n'
        f"Input:\n{json.dumps(payload_in, ensure_ascii=False)}"
    )

    def _call(*, use_json_object: bool):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
        }
        if use_json_object:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    try:
        response = _call(use_json_object=True)
    except Exception:
        response = _call(use_json_object=False)

    message = response.choices[0].message if response.choices else None
    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or "")
            else:
                parts.append(getattr(part, "text", None) or "")
        content = "\n".join(p for p in parts if p)

    return _parse_translated_payload(content, sentences)


def translate_story_result(
    client,
    story: StoryResult,
    *,
    target_language: str = "Mandarin",
    model: str = "gpt-4o-mini",
) -> StoryResult:
    translated = translate_story_sentences(
        client,
        list(story.sentences),
        target_language=target_language,
        model=model,
    )
    return StoryResult(translated)
