# Addendum: Companion Mode (Listener Q&A + Accumulating Memory)

**Date:** 2026-08-05  
**Parent design:** [`2026-08-03-context-aware-storyteller-design.md`](2026-08-03-context-aware-storyteller-design.md)  
**Status:** Implemented (Streamlit app, 2026-08-06) — see §9 at end of doc  
**Scope:** **Companion mode** — Q&A + **Continue story** stretch (not full co-author rewrite)

---

## 1. What Companion Mode Is

After (or while) the book is narrated with context-aware prosody, the **listener can ask or type questions**. A reasoning model answers using:

1. **Canon** — the ingested story text + page context tags already in Chroma  
2. **Progress** — how far the story has been told so far  
3. **Session memory** — prior Q&A in this listening session  

Answers **explain and deepen** the story; they do **not** rewrite the plot or append new narrative beats to the book (that would be *co-author mode*, out of scope here).

```text
PDF → [existing pipeline] → tagged chunks + narrated audio
                                    │
                         story_position + Chroma canon
                                    │
              listener Q ──▶ retrieve ──▶ reasoner (API router) ──▶ answer
                                    │
                                    └─▶ upsert turn → session memory
```

This is **RAG + chat state on top of the storyteller cache**, not a new model to train.

---

## 2. Why This Is a Good Addition

| Benefit | Why it matters for the group product |
| :--- | :--- |
| **Fits the existing architecture** | Reuses ingestion, page tags, Chroma, and the LLM fallback router. No second GPU stack for reasoning. |
| **Differentiates from flat TTS** | Emotive narration answers *how it sounds*; Companion answers *what it means* — especially useful for children / ESL / classroom demos. |
| **Accumulating context without bloat** | Memory grows as structured turns + optional rolling summary, not as “ship an LLM.” App stays an orchestrator. |
| **Safe for a POC / demo** | Grounded answers stay within the book; easier to evaluate than open-ended story generation. |
| **Natural group split** | One subgroup owns narration (TTS + DSP); another owns Companion (retrieve + reason + memory UI/notebook cells). |
| **Clear cut line** | Companion is cuttable: core deliverable remains one narrated audio file if time runs out. |

**What we deliberately avoid:** treating conversation as new canon that must be re-tagged and re-TTS’d every turn. That path explodes scope (consistency, kids’ safety, re-render cost). Keep Companion **read-only w.r.t. the book**.

---

## 3. Memory Model (Three Layers)

| Layer | Store | Contents | Update rule |
| :--- | :--- | :--- | :--- |
| **Canon** | Existing `context_tags` (+ book text chunks) | Sentences, page, speaker, emotion, pitch/rate/volume | Written once by the analysis pipeline; Companion only **reads** |
| **Progress** | Session metadata | `book_id`, `story_position` (page / chunk index / “told up to …”) | Advances when narration plays or user jumps chapters |
| **Dialogue** | New Chroma collection `session_turns` | `{turn_id, role, text, ts, retrieved_chunk_ids}` | Append on every Q and A |

**Working window for each call** (keep token cost bounded):

- Current `story_position` and a short progress blurb  
- Top-*k* retrieved canon chunks (e.g. *k* = 4–8) near that position + semantic match to the question  
- Last *N* dialogue turns (e.g. *N* = 8–12), **or** a rolling summary every ~5 turns + last 3 raw turns  

Do **not** dump the entire book or entire chat history into every prompt.

---

## 4. How to Implement It

### 4.1 Prerequisites (already in parent design)

- Book ingested and chunked  
- `context_tags` populated (page-level vision LLM tags)  
- Chroma `PersistentClient` on Drive  
- Same API key file + LLM router (Claude → GPT-4o → Gemini → Grok)  

Companion needs **no** XTTS/DSP unless you optionally speak the answer with the cloned voice later.

### 4.2 New Chroma collection

```text
session_turns
  metadata: session_id, book_id, turn_id, role ("user"|"assistant"),
            story_position, ts
  document: turn text
  embedding: sentence-transformers (same MiniLM as parent design)
```

Optional sidecar JSON on Drive for easy notebook debugging:

`/StoryTeller_cache/sessions/{session_id}.json` — full ordered transcript + summary.

### 4.3 Turn pipeline (per question)

1. **Ingest question** — notebook text box or `input()`; store as user turn.  
2. **Retrieve**  
   - Exact / metadata filter: chunks with `page_num <= story_position` (or ±1 page if “peek ahead” is disallowed for spoilers).  
   - Semantic query: embed the question → similarity search on canon text.  
   - Merge & dedupe; keep top-*k*.  
3. **Build prompt** — system instructions + retrieved passages + progress + recent turns + current question.  
4. **Call reasoner** — reuse the **same provider fallback router** as §4 of the parent design (text-only is enough; no page image required for Companion).  
5. **Validate lightly** — refuse if the model invents plot beyond retrieved text; prefer “the story hasn’t said yet” when progress blocks spoilers.  
6. **Persist** — upsert assistant turn to `session_turns`; refresh rolling summary if due.  
7. **Present** — print / markdown answer; optional: TTS the answer with XTTS + neutral tags (pitch/rate/volume = 3) for voice continuity.

### 4.4 Suggested prompt contract

```text
You are a story companion for a children's book already loaded in context.
Answer ONLY using the provided passages and prior Q&A.
Respect story_position: do not spoil later pages.
If unknown from context, say so briefly and invite the listener to continue the story.
Keep answers short, warm, and age-appropriate.
```

Return plain text for the POC (JSON optional if you later want `citations: [chunk_ids]`).

### 4.5 Spoiler / progress policy (pick one and stick to it)

| Policy | Behavior |
| :--- | :--- |
| **Strict (recommended for kids)** | Retrieve only chunks the listener has already heard (`<= story_position`). |
| **Soft** | Allow +1 page for “what does this picture mean?” when that page is on screen. |

Advance `story_position` when a page’s narration finishes (or via a simple “I’m on page N” control in the notebook).

### 4.6 Notebook / group integration

Add a clearly labeled Colab section **after** stitch/export:

1. Load Drive + Chroma + `API_KEYS`  
2. `session_id = new_uuid()`; set `book_id`, initial `story_position`  
3. Loop: ask → retrieve → reason → display → save  
4. Demo script: 5 fixed questions at mid-book to show memory (e.g. Q2 refers to Q1)

**Group split suggestion**

| Owner | Deliverable |
| :--- | :--- |
| Narration track | Parent pipeline Days 1–4 (audio out) |
| Companion track | `session_turns` + retrieve + reason loop + 5-question demo cell |
| Shared contract | `book_id`, chunk schema, `story_position`, router adapters |

### 4.7 Optional stretch (still Companion, not co-author)

- Speak answers with the same cloned voice (neutral DSP).  
- Cite page numbers in the answer (“On page 3, …”).  
- “Remember this for later” pin: promote a turn into a short `session_summary` field used every call.

---

## 5. Explicit Non-Goals (This Addendum)

- **Co-author mode** — listener replies become new story beats that re-enter tagging/TTS.  
- Training or hosting a local reasoning LLM.  
- Full chat product UI / accounts / multi-user sync.  
- Replacing the narrated full-book audio with interactive-only storytelling.

---

## 6. Success Criteria (Demo)

- Mid-book, ask a question grounded in an earlier page → answer uses that content.  
- Ask a follow-up that depends on the previous answer → model uses session memory (not only the book).  
- Ask about a later, unheard plot point under **strict** policy → model declines to spoil.  
- Restart Colab runtime → reopen same `session_id` from Drive → prior turns still load.  
- If Companion is dropped, parent narrated-audio POC still stands alone.

---

## 7. Size / Cost Reality Check

| Concern | Reality |
| :--- | :--- |
| “Is this almost building an LLM?” | No — API reasoner + retrieval + thin session store. |
| App / notebook size | Small: one section + one Chroma collection. |
| GPU | Unchanged for Q&A; XTTS only if answering aloud. |
| What actually grows | Tokens and Drive session logs — bound with *k*, *N*, and rolling summaries. |

---

## 8. One-Line Summary for the Group

**Companion mode** turns the storyteller from a one-shot emotive audiobook into a **context-aware listening buddy**: same book memory and LLM router, plus accumulating Q&A, without rewriting the story or shipping a custom LLM.

---

## 9. Implementation status (2026-08-06)

**Status:** Implemented in the Streamlit app (`app.py` + `pipeline/companion.py`, `continue_story.py`, `asr.py`).

| Addendum concept | Shipped as |
| :--- | :--- |
| Listener Q&A (text) | Typed question + Companion reasoner |
| Voice questions | Whisper ASR (`pipeline/asr.py`) + custom mic component |
| Spoken answers | XTTS `speak_reply` in narrator voice |
| Session memory | In-memory `companion_chat` + `CompanionSession` canon/progress |
| Chronological UI | `story_timeline` — chapters, Q&A, continues in order |
| Continue story | **Stretch beyond original read-only Companion:** LLM next beat + XTTS, appended to timeline and full-story audio |

Not yet shipped from this addendum: Chroma `session_turns` on Drive, Colab notebook cells, strict spoiler-decline policy tests, “remember this” pins.

See [`CHANGES.md`](../CHANGES.md) and [`README.md`](../README.md) for run instructions.
