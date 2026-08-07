# StoryTeller — session change summary (2026-08-05 / 2026-08-06)

Summary of work done on the StoryTeller Streamlit app (fork + parent remotes: `origin` = jonyling, `upstream` = hongming111).

## High-level outcomes

- Free local **XTTS** narration (EN + ZH) as the default TTS path; ElevenLabs remains optional.
- **Companion mode**: voice/text Q&A plus **Continue story**, with a single **chronological timeline**.
- Robust **browser mic** recorder (device picker + level check) and headphone-friendly playback.
- **Bidirectional text-PDF translation** (EN ↔ 中文) driven by the language picker, then TTS in that language.

---

## Pipeline & TTS

| Change | Where |
| :--- | :--- |
| Default `TTS_BACKEND=xtts`; ElevenLabs backup | `pipeline/config.py`, `requirements.txt`, `README.md` |
| Coqui XTTS clone + per-sentence synth (EN/ZH) | `pipeline/xtts_backend.py` |
| Prosody / theatre stage directions | `pipeline/theatre.py`, `pipeline/prosody.py` |
| Text-PDF path narrates extracted text (no silent placeholders) | `pipeline/orchestrator.py`, `app.py` |
| Long lines chunked for XTTS ~250-char limit | `app.py` / XTTS helpers |
| Vision story gen hardened; default model `gpt-4o-mini` | `pipeline/story_gen.py` |
| `pypinyin` for Mandarin XTTS | `requirements.txt` |

## Translation (text PDFs)

| Change | Where |
| :--- | :--- |
| Detect Latin vs CJK; translate when picker ≠ PDF script | `pipeline/translate.py` |
| EN→中文 and 中文→EN before theatre + TTS | `app.py` (`generate_mock_story`) |
| Unit tests | `tests/test_translate.py` |
| Documented in pipeline list | `README.md` |

Vision / picture / camera stories still generate directly in the selected language (no separate translate step).

## Companion mode

| Change | Where |
| :--- | :--- |
| Whisper ASR for voice questions | `pipeline/asr.py` |
| RAG-style companion Q&A + session memory | `pipeline/companion.py` |
| Continue story: LLM beat + XTTS + append to canon | `pipeline/continue_story.py` |
| Spoken answers in narrator voice | `app.py` |
| Chronological `story_timeline` (chapters + Q&A + continues in order) | `app.py` |
| Controls (Continue / mic / type) pinned at bottom | `app.py` |
| Tests | `tests/test_companion.py`, `tests/test_continue_and_asr.py` |
| Design addendum | `docs/2026-08-05-storyteller-companion-mode-addendum.md` |

## Mic, recording, playback

| Change | Where |
| :--- | :--- |
| Custom mic component (device list, Check level, Record) | `mic_component/` |
| Prefer External Mic; hide Steam/virtual devices; dedupe labels | `mic_component/frontend/index.html` |
| Unique `take_id` so WebM header prefixes don’t drop new takes | same + `app.py` |
| In-iframe preview + **Push to app** | `mic_component/frontend/index.html` |
| Stereo/48k + MP3 + download for Brave/Chrome headphones | `app.py` (`_for_browser_playback`, `_st_play_wav`) |
| Defer Whisper until **Send question** after preview | `app.py` |

## UI / product behavior

- Full-story WAV player plus chapter / timeline feed.
- Sentence follow-along moved to an optional expander.
- Built-in voice assets / generator script: `assets/voices/`, `scripts/generate_builtin_voices.py`.
- Secrets example updated; real keys stay in ignored `.streamlit/secrets.toml`.

## Waiting-state indicators (2026-08-07)

**Continue story**, **Add to Companion** (voice question), and **Ask** (typed question) previously ran their LLM/Whisper/TTS calls with no visual feedback — on CPU-only XTTS this could look frozen for 30s-2min.

| Change | Where |
| :--- | :--- |
| Themed `st.status()` per stage (Listening → Thinking → Voicing), ticking elapsed clock | `app.py` (`_run_staged`) |
| The 4 controls (Continue / Add to Companion / Ask / question text box) disable, visibly (not just dimmed), while any one wait is running | `app.py` (`wait_kind` session-state flag) |
| Persistent error banner + collapsible technical detail on failure, instead of a rerun wiping the error | `app.py` (`wait_error`) |
| New feed entry gets a one-shot accent-colour flash after a long silent wait | `app.py` CSS (`.st-key-tl_flash`) |
| Stage labels/reassurance copy localized (EN/ZH × formal/playful) | `app.py` copy dicts |

## Tests & docs

- New/updated: translate, companion, continue/ASR, theatre, orchestrator, story_gen tests.
- `README.md`, `CHANGES.md`, design spec §11, Companion addendum §9 updated (2026-08-06).

## Not committed

- `.streamlit/secrets.toml` (gitignored; contains API keys).
- Local sample `Charlie_3-5.pdf` (copyrighted book extract; keep local only).

## How to run

```bash
cd Week8/Week8_Capstone/StoryTeller
export COQUI_TOS_AGREED=1
pip install -r requirements.txt
streamlit run app.py
```

Pick **中文** + English text PDF (or **English** + Chinese PDF) to exercise translation; use Companion at the bottom of the chronological feed after Generate.
