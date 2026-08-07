# StoryTeller — Viva / Q&A Prep FAQ

Answers are grounded in the actual code (file:line references included) — not aspirational description. If asked to elaborate, open the referenced file.

---

## 1. Architecture

**Q: What's the overall architecture?**
A layered pipeline behind a Streamlit UI, orchestrated by `pipeline/orchestrator.py::run_pipeline()`:

```
Story source (PDF text / vision / camera)
  → Story Generator (LLM)
  → Theatre Adapter (stage directions, pitch/rate/volume)
  → Voice Cloner (reference audio prep)
  → TTS Synthesizer (XTTS or ElevenLabs)
  → Ambience mixer (optional)
  → Companion (post-narration Q&A / continue-story)
```

**Q: What design pattern ties the providers together?**
Strategy pattern. Each stage is an interface with swappable concrete classes:
- Story generation: `OpenAIStoryGenerator` / `ClaudeStoryGenerator` / `GeminiStoryGenerator` / `GrokStoryGenerator` (`pipeline/story_gen.py`), all exposing `.generate(images, language)`.
- TTS: `XTTSNarrationSynthesizer` vs `ElevenLabsNarrationSynthesizer` — both implement per-sentence synthesis but with different method shapes; `orchestrator.py::_supports_per_sentence()` duck-types which interface a synthesizer offers.
- Theatre: `RuleBasedTheatreAdapter` (default, deterministic) vs `LLMTheatreAdapter` (optional polish, falls back to rule-based on any failure).

A factory function, `create_story_generator(provider, ...)` (`story_gen.py:251`), picks the concrete class from the `STORY_PROVIDER` env var — swapping LLM providers is a config change, not a code change.

**Q: Is the pipeline coupled to Streamlit?**
No. `pipeline/` has no Streamlit imports except one defensive `try/except ImportError` in `config.py` (for reading `st.secrets` when running inside the app, falling back to env vars otherwise). This is why the pipeline is unit-testable headless with `pytest` — tests fake the LLM/TTS clients rather than hitting real APIs or a real Streamlit session.

**Q: What's in `app.py` vs `pipeline/`?**
`app.py` is the UI/session layer: widgets, theme/language state, locking behavior, rendering the story feed. `pipeline/` is the actual generation logic. `app.py` composes pipeline objects and calls `run_pipeline()`; it doesn't contain generation logic itself.

---

## 2. LLM usage & "creative settings"

**Q: What temperature/top_p do you use for story generation?**
None are set — every call uses the provider's default (OpenAI defaults to 1.0). This is deliberate: structure is enforced via `response_format={"type": "json_object"}` plus an explicit JSON shape in the prompt, not by dialing down randomness. If asked "why not lower the temperature for consistency" — the honest answer is it wasn't tuned; the JSON-object constraint was judged sufficient, and lower temperature wasn't tested against output quality.

**Q: What models are used per pipeline stage?**

| Stage | Model(s) | max_tokens | File |
|---|---|---|---|
| Story generation (vision) | gpt-4o-mini (openai) / claude-sonnet-5 / gemini-3.5-flash / grok-2-vision-latest | 4000 (raised from 1500 — see below) | `story_gen.py` |
| Theatre stage-direction polish (optional) | gpt-4o-mini | 800 | `theatre.py` |
| Text-PDF translation (EN↔中文) | gpt-4o-mini | 4000 | `translate.py` |
| Companion Q&A | gpt-4o-mini | 350 | `companion.py` |
| Continue-story beat | gpt-4o-mini | 600 | `continue_story.py` |
| Voice-question transcription | `whisper-1` | — | `asr.py` |

**Q: How do you guarantee the LLM returns valid JSON?**
Three layers of defense, in order (`story_gen.py::_parse_story_payload`, mirrored in `translate.py`):
1. Ask for `response_format={"type":"json_object"}` (retry once without it if the gateway rejects it alongside vision input — some do).
2. Strip a markdown code fence if the model wrapped the JSON in ` ```json ` anyway.
3. If `json.loads()` still fails, regex out the first `{...}` blob and try again.
If all three fail, raise a `PipelineError` with the raw text (truncated to 240 chars) so the UI shows *what* came back instead of a bare traceback.

**Q: What happens if the vision model refuses or returns empty content?**
Caught explicitly — `_generate_via_openai_compatible_chat` checks `finish_reason` and `refusal` fields and raises a `PipelineError` naming the model and suggesting checks (vision access, rate limits, content filtering), rather than crashing on `None` content (`story_gen.py:170-183`).

**Q: Why was `max_tokens` raised from 1500 to 4000 for story generation?**
A real observed failure mode, not a preemptive guess: near a tight token ceiling, the model could truncate mid-JSON and leak its own `"speaker"/"emotion"` schema fragments into a sentence's `text` field — because that's still syntactically valid JSON (a string containing stray quoted words), it passed the parser cleanly and only showed up as garbled narration text later. Fixed two ways: raised the ceiling to 4000 so longer stories have room to finish cleanly, and added a regex (`_LEAKED_JSON_TAIL`) that strips any such fragment that still slips through, applied after parsing since it can't be caught at the JSON-validity level. Good example to cite if asked "what did you learn/iterate on" — this was found and fixed mid-project, not designed in from the start.

---

## 3. Voice cloning & TTS

**Q: Is the voice cloning fine-tuned per user?**
No — it's **zero-shot**, not fine-tuned. `XTTSVoiceCloner.clone()` (`xtts_backend.py:108`) does no training at all: it trims the uploaded/recorded audio to ~12 seconds and saves it as a reference WAV file; that file path *is* the `voice_id`. At synthesis time, XTTS-v2 takes that reference WAV as `speaker_wav` per sentence and generates speech in that timbre on the fly. There is no persistent per-user voice model built or stored.

**Q: How is that different from the ElevenLabs backup path?**
ElevenLabs' Instant Voice Cloning actually registers a voice server-side via their API (`pipeline/voice_clone.py::ElevenLabsVoiceCloner`) and returns a real `voice_id` you call back into for synthesis — a persistent cloud resource. XTTS's "voice_id" is just a local file path; nothing is registered anywhere.

**Q: What TTS models are used?**
- Default: `tts_models/multilingual/multi-dataset/xtts_v2` (Coqui XTTS-v2, free, local, CPU or GPU) — a multilingual multi-speaker transformer TTS model.
- Paid backup (`TTS_BACKEND=elevenlabs`): ElevenLabs `eleven_v3` (`pipeline/tts.py:23`).

**Q: How is prosody (pitch/rate/volume) controlled? Is that the LLM's doing?**
No — it's rule-based DSP, not learned or LLM-driven. Each sentence gets an `emotion` label (from the LLM for vision stories, from keyword tagging for text-PDFs — see §4), which maps to default pitch/rate/volume via `EMOTION_DSP_DEFAULTS` (`pipeline/prosody.py`). Those values are applied as an audio post-processing step (`apply_prosody_to_wav_bytes`) on the raw XTTS output.

**Q: XTTS has a character limit — how do long sentences get handled?**
XTTS truncates/warns above ~250 characters. `_chunk_text_for_xtts` (`xtts_backend.py:45`) splits only when a single sentence exceeds that, preferring to cut at semicolons/em-dashes/commas before falling back to a hard cut — never mid-word if avoidable. The resulting chunks are synthesized separately and stitched back together with a short crossfade (`_concat_wav_bytes`) so it still sounds like one continuous line.

**Q: Why free XTTS as the default instead of ElevenLabs?**
Cost — XTTS runs locally with no per-character API charge, at the expense of speed on CPU (30s–2min per generation without a GPU, per `README.md`) and needing a specific ffmpeg build (see §6).

---

## 4. Companion mode — retrieval is *not* embeddings-based

**Q: What retrieval technique powers Companion Q&A — is it RAG with a vector database?**
No — and this is worth being precise about if pressed. `retrieve_passages()` (`pipeline/companion.py:59`) scores passages by **plain lexical token overlap**: a regex word tokenizer, set intersection between question tokens and passage tokens, sorted by overlap count. No embedding model, no vector store, no cosine similarity. It's closer to a minimal BM25 than semantic search.

**Q: Why not real embeddings/vector search?**
Not stated in the code as a deliberate tradeoff write-up, but the honest framing: keyword overlap is zero-latency, zero-cost, fully deterministic, and easy to reason about for the spoiler guard below — at the cost of missing semantically-related-but-lexically-different matches (e.g. a question about "the fox" wouldn't match a passage that only says "he" via pronoun resolution).

**Q: How do you prevent Companion from spoiling later pages?**
A hard cutoff, not a soft one: `retrieve_passages` filters out any passage with `index > session.story_position` before scoring even runs (`companion.py:64`). `story_position` only advances as narration is actually heard (`append_to_canon`). So Companion is structurally incapable of retrieving unheard content — it's not relying on the LLM to "choose" not to spoil.

**Q: Is there a summarization/memory step for long sessions?**
A lightweight one: every 5 turns past the 10th, `ask_companion` (`companion.py:140`) builds a rolling summary by concatenating truncated recent turns — no extra LLM call, just string joining. It's included in the next prompt as `session_summary` context.

---

## 5. PDF ingestion & the text-vs-vision decision

**Q: How does the app decide whether to use vision or extracted text for a PDF?**
`extract_pages_from_pdf` (`app.py:215`) tries `pypdf`'s `PdfReader.extract_text()` per page first. If **any** page yields text, it's treated as a text PDF and returns sentence data directly (no vision call). If **no** page yields extractable text (e.g. a scanned/image-only PDF), it returns `None`, and the caller falls back to rasterizing pages to images (`pipeline/pdf_ingest.py::extract_page_images`, via PyMuPDF/`fitz` at 150 DPI, downscaled to max 1024px) and running the vision LLM path instead.

**Q: For text-PDFs, how is per-sentence emotion assigned, if there's no vision call?**
A local keyword-count heuristic, `tag_emotion()` (`app.py:196`) — not an LLM call. It counts keyword hits per emotion category from `EMOTION_KEYWORDS` and picks the highest-scoring category, defaulting to `"neutral"` if nothing matches. This keeps text-PDF narration fast and free of an extra API round-trip.

**Q: What are the PDF limits?**
1–10 pages (`pipeline/pdf_ingest.py:6-7`); more than 10 raises a `ValidationError` telling the user to trim it, rather than silently truncating or timing out on an oversized vision request.

---

## 6. Concurrency, performance, and infra quirks

**Q: What runs in parallel?**
Story generation and voice-reference prep run concurrently via `ThreadPoolExecutor(max_workers=2)` (`orchestrator.py:73`) since they're independent of each other. If ambience is enabled, ambience-clip fetching (Freesound) is prefetched on its own thread pool while XTTS synthesis proceeds on the main thread.

**Q: Why isn't ambience fetching just done inside the same thread pool as everything else?**
Streamlit forbids progress-callback UI updates from a non-main thread — calling `st.*` off-thread raises `NoSessionContext` and aborts generation. XTTS synthesis needs to report per-line progress, so it must run on the main thread; ambience fetching doesn't need to report progress, so it's the one safely pushed to a background thread.

**Q: What's the ffmpeg gotcha mentioned in the README?**
Newer PyTorch (2.9+) needs `torchcodec` for audio I/O, which dynamically loads `libavcodec`/`libavformat` via the OS loader — that requires a **shared**-library ffmpeg build. The default Windows winget package (`Gyan.FFmpeg`) is statically linked (no DLLs), so XTTS audio loading silently fails with `Could not load libtorchcodec` even though plain `ffmpeg.exe` decoding (used by `pydub`) works fine. `pipeline/config.py::ensure_ffmpeg_on_path()` auto-discovers a shared build (`BtbN.FFmpeg.GPL.Shared`) and patches `PATH` at import time so a stale shell doesn't need to already resolve it.

---

## 7. Testing

**Q: How is the pipeline tested without hitting real APIs?**
`tests/` (pytest) fakes provider clients — story generation, translation, companion, and TTS tests all pass fake/stub client objects rather than real OpenAI/Anthropic/ElevenLabs clients, so the suite runs fully offline and free. There's one test file per pipeline module (`test_story_gen.py`, `test_theatre.py`, `test_translate.py`, `test_companion.py`, `test_continue_and_asr.py`, `test_tts.py`, `test_voice_clone.py`, `test_sfx.py`, `test_orchestrator.py`, `test_pdf_ingest.py`, `test_prosody.py`, `test_audio_utils.py`, `test_config.py`).

**Q: What's *not* covered by automated tests?**
XTTS GPU/CPU synthesis itself is manual smoke-testing only (per `README.md`) — running the actual multi-GB model in CI isn't practical.

---

## 8. Config, secrets, and settings surface

**Q: What are the environment variables?**
- `STORY_PROVIDER` (default `openai`): `openai` / `claude` / `gemini` / `grok`.
- `TTS_BACKEND` (default `xtts`): `xtts` (free/local) or `elevenlabs` (paid).
- `COQUI_TOS_AGREED`: must be `1` to accept XTTS's license non-interactively.

**Q: Where do API keys live, and why?**
`.streamlit/secrets.toml`, which is git-ignored — keys never get committed. `get_secret()` (`pipeline/config.py:75`) checks `st.secrets` first, then falls back to an OS environment variable, so the same code works whether run inside Streamlit or as a plain script.

**Q: What in-app settings exist, and what happens once a story is generated?**
Theme (4 skins: Classic, Crayon, Sticker, Bedtime — each a full color/font/copy pack), Language (EN/中文), Ambience toggle, Story source (PDF/Picture/Camera), Voice source (4 built-in presets or upload/record your own). Once a story exists, everything **except** Theme/Language/Ambience locks (disabled, not hidden — values still resolve normally) and collapses into a summary. This is deliberate: it stops a mid-story settings change from desyncing the visible settings from the audio that's already been generated.

---

## 9. Known limitations / "why didn't you..." questions

- **Why not a vector DB for Companion?** See §4 — deterministic keyword overlap was chosen over embeddings; trades recall for simplicity, cost, and spoiler-guard auditability.
- **Why is generation slow?** CPU-only XTTS with no GPU — 30s–2min per generation is expected, documented in `README.md`, not a bug.
- **Why no temperature tuning?** Not investigated — JSON-object response formatting was treated as sufficient for structural reliability; creative variance wasn't separately tuned.
- **Why keyword-based emotion tagging for text-PDFs instead of always using the LLM?** Avoids an extra API round-trip when the text is already fully extracted and speed/cost matters more than nuance there.
- **What happens if the LLM's JSON is malformed after all three parsing attempts?** A `PipelineError` surfaces the raw (truncated) text to the user in the UI rather than crashing silently — see §2.
