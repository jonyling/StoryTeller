# StoryTeller — Live Demo Script (full app)

**Audience:** class / mini-project demo  
**Suggested total:** ~8–10 minutes talking + live clicks  
**Split:**

| Who | Owns |
| :--- | :--- |
| **Hongming** | Problem, pipeline, generate story, voice clone, theatre/prosody, ambience, playback |
| **Jony (you)** | Companion: chronological feed, typed Q&A, voice ask, Continue story, memory/safety pitch |

**Prep (before the room fills):**

1. App running: `http://localhost:8501` (prefer **Edge** for headphones).
2. Secrets loaded (OpenAI + Freesound if showing ambience).
3. Demo assets ready: `demo files/sample_story.pdf` + `demo files/sample photo.jpg`, or Charlie PDF; built-in voice **Warm** / **Gentle**.
4. Mic allowed in browser; pick **External Mic / Realtek** (not Steam).
5. Optional: one story already generated so Companion can start immediately if Generate runs long.

### Page layout (top → bottom)

After **Generate & read story**, the main column is:

1. **Play entire story so far** — full-story WAV player  
2. **Read along** — expander: one sentence at a time, emotion badge, ← / →, per-sentence audio  
3. **Story & Companion — in order** — chronological feed (chapters, Q&A, continues)  
4. **Add next** — **Continue story** · **Type a question** + **Ask** · **Ask with your voice** (mic)

Hard-refresh (**Ctrl+Shift+R**) if the mic player or **Add to Companion** button is missing after Record.

**Timing cheat sheet**

| Block | Who | ~Time |
| :--- | :--- | ---: |
| Hook + one-liner | Hongming | 0:45 |
| Setup → Generate → play | Hongming | 3:30 |
| Hand-off | Hongming → Jony | 0:15 |
| Companion (text + voice + continue) | Jony | 3:30 |
| Close + limitations | Both (Jony leads close) | 1:00 |

---

## 0. Opening (Hongming) — 45s

> “StoryTeller turns a short picture book or photo into a **narrated story in your own voice**, in English or Mandarin.  
> Under the hood: story text or vision → **theatre tags** (who speaks, emotion, pitch/rate) → **local XTTS voice clone** → optional **ambience** → playback.  
> After the story, we add a **Companion** so a listener can ask questions and even continue the story — Jony will show that second half.”

*(One sentence on group split is enough — don’t list every module.)*

---

## 1. Main app — generate & listen (Hongming) — ~3.5 min

### 1.1 Settings (while pointing at the header)

**Say:**

> “Theme and language drive both the UI and the story language.  
> Source can be PDF, picture, or camera.  
> Voice: built-in sample, upload, or record.  
> Ambience is optional Freesound beds matched to emotion.”

**Do:**

1. Theme: **Classic** (or whatever looks best on the projector).
2. Language: **English** first (safer for a mixed room); mention **中文** + text PDF will **translate then narrate**.
3. Source: **PDF** → upload `demo files/sample_story.pdf` (or Charlie pages).
4. Voice: **Built-in → Warm** (or Gentle). Briefly click preview if available.
5. Toggle **Ambience ON** if Freesound key works; otherwise leave OFF and say “optional; we can skip if the key is cold.”

### 1.2 Generate

**Say:**

> “Generate runs the full pipeline: extract or invent sentences, theatre adaptation, clone TTS with prosody DSP, then stitch a full-story player. On GPU this takes a bit; first XTTS load can be slow.”

**Do:** Click **Generate & read story**.  
While waiting: gesture at progress / status; name the stages once (story → theatre → XTTS → ambience).

**If it fails:** Fall back to a **pre-generated session** you prepared, or the built-in sample path — don’t debug APIs live.

### 1.3 Playback & theatre

**Say:**

> “Here’s the **full story audio**. Open **Read along** for sentence-by-sentence view with emotion badges.  
> Below that, the **Story & Companion** feed is chronological.  
> Each line carries speaker and emotion; theatre tags drive pitch, rate, and volume so it’s not flat TTS.”

**Do:**

1. Play **Full story audio** (~10–15 s, then pause).
2. Expand **Read along** → flip one sentence, point at speaker/emotion badge (~10 s).
3. Scroll **Story & Companion — in order** → show Chapter 1 text + chapter player.
4. Optionally open **Theatre script JSON** for 5 seconds (“this is the stage directions object”).

### 1.4 Hand-off line (Hongming)

> “So far: book in, cloned voice out, emotive narration.  
> Next: what if the child asks a question mid-story, or wants ‘what happens next?’ — Jony.”

*(Slide or verbal: “Companion mode”.)*

---

## 2. Companion (Jony) — ~3.5 min

### 2.1 Framing (30s)

**Say:**

> “Companion is the **listening buddy**, not a co-author by default.  
> It answers from **story canon + how far you’ve heard + this session’s Q&A**.  
> Stretch feature: **Continue story** — a short new beat, narrated in the **same cloned voice**, appended **in order** in the feed.  
> Design idea: narration answers *how it sounds*; Companion answers *what it means* — useful for kids / ESL / classroom demos.”

### 2.2 Chronological feed (20s)

**Do:** Point at **Story & Companion — in order**.

**Say:**

> “Everything appends top to bottom: original chapter, then each question, answer, and continue beat. Controls stay at the bottom under **Add next** so the timeline doesn’t jumble.”

### 2.3 Typed question (60–75s) — primary reliable demo

**Do:**

1. Scroll to **Type a question** (above the mic panel).
2. Type something grounded in the visible story, e.g.  
   - EN: `Who is the main character, and how do they feel at the start?`  
   - or a concrete detail from the sample PDF.
3. Click **Ask**. Wait for status / clock if shown.

**Say (while waiting):**

> “We retrieve heard passages, call a reasoner with session memory, then **speak the answer in the narrator voice** — same clone as the book.”

**Do:** Play the **Narrator reply** audio in the feed; point at the user bubble + assistant bubble.

**Backup question** if the first answer is vague: `What happens on the first page?`

### 2.4 Voice question (45–60s) — optional if mic is flaky

**Do:**

1. **Ask with your voice** → mic: **External Mic**.
2. **Check level** until the bar moves.
3. **Record** → short clear question → **Stop**.
4. Click **Add to Companion** in the mic box (starts Whisper + answer — not “Check level”, which is meter only).

**Say:**

> “Custom mic panel so we don’t get Streamlit’s silent default device. Whisper transcribes; then same Companion path. Prefer Edge for playback; Brave may need Download.”

**If mic fails:** Skip — typed path already proved the feature. Don’t fight Brave mid-demo.

### 2.5 Continue story (60–75s)

**Do:** Click **Continue story** (primary button under Add next).

**Say:**

> “This asks the LLM for a short next beat, narrates it with XTTS, appends a new chapter block, and updates the full-story audio and companion memory.  
> It’s a controlled stretch beyond pure Q&A — still one short beat, not rewriting the whole book.”

**Do:** When ready, scroll to the new **Chapter N — Continued** in the feed; play a few seconds.

### 2.6 One technical punchline (15s)

**Say:**

> “Implementation-wise: in-session canon and progress, Whisper for voice, reasoner API for answers, XTTS for spoken replies, and a single `story_timeline` so the UI stays chronological.  
> We’re honest in the report: this is a strong integration POC — future work is voice adaptation and listening / live benchmarks.”

---

## 3. Close (Jony leads; Hongming can nod) — ~1 min

**Together, ~3 bullets:**

1. **Pipeline:** PDF/photo → theatre-aware XTTS narration (EN/ZH), optional ambience.  
2. **Companion:** ask by text/voice; continue the story; same voice; ordered feed.  
3. **Limits:** zero-shot clone (no task-specific fine-tune yet); no large MOS study; demos need GPU + keys.

**Exit line:**

> “Questions? Happy to replay Companion or regenerate with Mandarin / ambience if time allows.”

---

## Live demo click path (one page checklist)

```text
[Hongming]
Theme → Lang EN → PDF upload → Voice Warm → Ambience ON?
→ Generate & read story
→ Play full story (short)
→ Expand Read along → one sentence + emotion badge
→ Show chapter in feed + optional theatre JSON
→ Hand off

[Jony]
Point at chronological feed
→ Type question → Ask → play narrator reply
→ (Optional) Mic → Check level → Record → Add to Companion
→ Continue story → play new chapter
→ Close: integration POC + future voice FT / eval
```

---

## Contingencies

| Problem | Move |
| :--- | :--- |
| Generate too slow | Use pre-baked story session; jump to Companion |
| Ambience / Freesound fails | Toggle off; say optional |
| Brave silent audio | Download button / switch to Edge |
| Mic silent / no Add button | Hard-refresh; typed Ask only |
| Companion API error | Show timeline + Continue if TTS still works; explain key |
| Wrong language UI | Stay EN for the room; mention ZH translation as one liner |

---

## Suggested Mandarin one-liner (if asked)

> “If language is 中文 and the PDF is English text, we **translate sentences first**, then XTTS speaks Mandarin in the cloned voice — and Companion can answer in Chinese when the question is Chinese.”

---

## Speaker cue cards (print / second screen)

### Hongming — 4 lines
1. Hook: book + your voice + EN/ZH.  
2. Settings → Generate → full player.  
3. Theatre tags + prosody (+ ambience).  
4. “Companion next — Jony.”

### Jony — 5 lines
1. Companion = meaning, not just sound; timeline in order.  
2. Typed Ask → spoken answer in cloned voice.  
3. Voice Ask = Whisper + same path (optional).  
4. Continue = next beat, same voice, appended.  
5. Close: POC today; voice FT + perceptual/live eval next.
