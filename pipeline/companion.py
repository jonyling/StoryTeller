"""Companion mode: listener Q&A with accumulating session memory (no story rewrite)."""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_WORD = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


@dataclass
class CompanionTurn:
    role: str  # user | assistant
    text: str
    ts: float
    story_position: int


@dataclass
class CompanionSession:
    session_id: str
    book_id: str
    language: str
    canon: list[dict]  # {index, text, speaker, emotion, page}
    story_position: int = 0
    turns: list[CompanionTurn] = field(default_factory=list)
    summary: str = ""

    def advance_to(self, index: int) -> None:
        self.story_position = max(self.story_position, int(index))


def canon_from_pages(pages) -> list[dict]:
    canon = []
    idx = 0
    for page in pages or []:
        page_no = int(page.get("page", 1))
        for sent in page.get("sentences", []):
            canon.append(
                {
                    "index": idx,
                    "page": page_no,
                    "text": sent.get("text", ""),
                    "speaker": sent.get("speaker", "narrator"),
                    "emotion": sent.get("emotion", "neutral"),
                }
            )
            idx += 1
    return canon


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD.finditer(text or "")}


def retrieve_passages(session: CompanionSession, question: str, *, k: int = 6) -> list[dict]:
    """Strict spoiler policy: only passages with index <= story_position."""
    q_toks = _tokens(question)
    scored = []
    for item in session.canon:
        if item["index"] > session.story_position:
            continue
        overlap = len(q_toks & _tokens(item["text"]))
        scored.append((overlap, item))
    scored.sort(key=lambda x: (-x[0], x[1]["index"]))
    # Always include the most recent heard lines so short questions still ground
    recent = [c for c in session.canon if c["index"] <= session.story_position][-3:]
    picked = []
    seen = set()
    for _, item in scored[:k]:
        if item["index"] not in seen:
            picked.append(item)
            seen.add(item["index"])
    for item in recent:
        if item["index"] not in seen:
            picked.append(item)
            seen.add(item["index"])
    return picked[:k]


_SYSTEM_PROMPT = (
    "You are a story companion for a children's book already loaded in context. "
    "Answer ONLY using the provided passages and prior Q&A. "
    "Respect story_position: do not spoil later pages. "
    "If unknown from context, say so briefly and invite the listener to continue the story. "
    "Keep answers short, warm, and age-appropriate. "
    "Respond in the same language as the listener's question when possible."
)


class CompanionReasoner:
    """Thin OpenAI-compatible chat wrapper (works with OpenAI or xAI-compatible clients)."""

    def __init__(self, client, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model

    def answer(self, session: CompanionSession, question: str, passages: list[dict]) -> str:
        recent = session.turns[-8:]
        history = "\n".join(f"{t.role}: {t.text}" for t in recent)
        passage_block = "\n".join(
            f"[{p['index']}|p{p['page']}|{p['speaker']}] {p['text']}" for p in passages
        ) or "(no heard passages yet — invite the listener to start the story)"
        user = (
            f"language={session.language}\n"
            f"story_position={session.story_position} (highest heard sentence index)\n"
            f"session_summary={session.summary or '(none)'}\n\n"
            f"PASSAGES:\n{passage_block}\n\n"
            f"RECENT_QA:\n{history or '(none)'}\n\n"
            f"QUESTION:\n{question}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=350,
        )
        return (response.choices[0].message.content or "").strip()


def ask_companion(session: CompanionSession, question: str, reasoner: CompanionReasoner) -> str:
    q = (question or "").strip()
    if not q:
        return ""
    session.turns.append(
        CompanionTurn(role="user", text=q, ts=time.time(), story_position=session.story_position)
    )
    passages = retrieve_passages(session, q)
    answer = reasoner.answer(session, q, passages)
    session.turns.append(
        CompanionTurn(
            role="assistant", text=answer, ts=time.time(), story_position=session.story_position
        )
    )
    if len(session.turns) >= 10 and len(session.turns) % 5 == 0:
        # Lightweight rolling summary from last turns (no extra LLM call)
        session.summary = " | ".join(
            f"{t.role}:{t.text[:80]}" for t in session.turns[-6:]
        )
    return answer


def append_to_canon(session, sentence_dicts: list[dict], *, page_no: int) -> None:
    """Extend CompanionSession.canon with newly narrated sentences and advance position."""
    start = len(session.canon)
    for offset, sent in enumerate(sentence_dicts):
        session.canon.append(
            {
                "index": start + offset,
                "page": page_no,
                "text": sent.get("text", ""),
                "speaker": sent.get("speaker", "narrator"),
                "emotion": sent.get("emotion", "neutral"),
            }
        )
    if session.canon:
        session.advance_to(session.canon[-1]["index"])


def new_session(pages, *, language: str = "English", book_id: str | None = None) -> CompanionSession:
    return CompanionSession(
        session_id=str(uuid.uuid4()),
        book_id=book_id or "story",
        language=language,
        canon=canon_from_pages(pages),
        story_position=0,
    )


def save_session(session: CompanionSession, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session.session_id,
        "book_id": session.book_id,
        "language": session.language,
        "story_position": session.story_position,
        "summary": session.summary,
        "canon": session.canon,
        "turns": [
            {"role": t.role, "text": t.text, "ts": t.ts, "story_position": t.story_position}
            for t in session.turns
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
