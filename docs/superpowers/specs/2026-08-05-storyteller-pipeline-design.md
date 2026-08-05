# StoryTeller — PDF-to-Narrated-Story Pipeline: Design Spec

**Date:** 2026-08-05
**Status:** Approved for implementation planning
**Scope:** POC for a live demo. Local, single-user Streamlit app.

## 1. Purpose

A user uploads a short picture-book PDF (~4 images) and a voice sample, picks
whether to include background sound effects, and picks an output language
(English or Mandarin). The app generates a short story grounded in the images,
clones the user's voice, and renders a single narrated audio file with
dynamic prosody and optional ambience — in the chosen language, in the user's
own cloned voice.

## 2. Key Constraints Discovered During Design

These correct assumptions in the source guide (`Hume_AI_EVI_Guide_v2.md`) and
materially shape the architecture:

- **Hume EVI vs. Octave TTS:** "Hume EVI" (Empathic Voice Interface) is a
  real-time speech-to-speech conversational product, not a batch
  text-to-speech renderer. The actual narration engine is **Hume Octave TTS +
  Voice Cloning** (`hume_client.tts.synthesize_json`,
  `create_custom_voice`), which is what the guide's own reference code uses.
- **No native Hume SFX/ambience API.** Hume's documented audio effects are
  voice-processing filters (reverb, radio, telephone, robot voice,
  underwater, cave, etc.), not environmental ambience mixing. The guide's
  "Context-Aware SFX & Ambience Mixing" describes what the *app* does
  (confirmed in its own reference code: local files mixed via PyDub), not a
  Hume feature.
- **Hume Octave does not support Mandarin.** Verified against Hume's official
  docs (dev.hume.ai) and blog: Octave 2 supports Arabic, English, French,
  German, Hindi, Italian, Japanese, Korean, Portuguese, Russian, Spanish —
  no Chinese variant, none announced as planned. Since the pipeline requires
  an English/Mandarin toggle, **Mandarin output uses ElevenLabs** instead
  (confirmed: supports Mandarin, Instant Voice Cloning from a 1-5 min sample,
  and cross-lingual clone-once-speak-any-language).

## 3. High-Level Architecture

Single local Streamlit app (`streamlit run app.py`). One story processed per
run; no background workers, job queues, or multi-user concurrency (local
single-user POC).

```
[Streamlit Form]
 PDF upload + voice sample upload + SFX toggle + language choice (EN/中文)
        │
        ▼
0. Validation          Page count sanity check on PDF; voice sample duration
        │              checked against the active backend's clone requirements
        │              (Hume ~10s-3min / ElevenLabs ~1-5min) before any paid
        │              API call fires.
        ▼
┌─────────────────────────── Stage 1 (parallel) ───────────────────────────┐
│  1a. Vision → Story    Vision-LLM takes page images + language, returns   │
│                        { story_text (with [whisper]/[gasp]-style tags),  │
│                          sfx_mood, tts_style_description }               │
│  1b. Accent Detection  Audio-capable LLM listens to the voice sample,    │
│                        returns { accent_label, detected_language }       │
│  1c. Voice Cloning     Registers the voice sample with the selected      │
│                        backend (Hume or ElevenLabs per language) → voice_id│
└────────────────────────────────────────────────────────────────────────┘
        │  (cross-lingual hint: if detected_language != output language,
        │   accent_label is folded into tts_style_description)
        ▼
┌─────────────────────────── Stage 2 (parallel) ───────────────────────────┐
│  2a. Narration Synth   backend.synthesize(story_text, voice_id,          │
│                        style_description, language) → narration audio    │
│  2b. SFX Fetch         (only if SFX enabled) Freesound search on         │
│                        sfx_mood → cached preview clip                    │
└────────────────────────────────────────────────────────────────────────┘
        ▼
3. Mixing               Loop/trim SFX clip to narration length, overlay at
        │                -18dB beneath narration (PyDub). Skipped (dry
        │                narration only) if SFX disabled or fetch failed.
        ▼
[ Final MP3 ]  → st.audio() player + download button
```

Stage 1's three calls are independent of each other (each depends only on
the original upload, not on another stage-1 output), and Stage 2's two calls
are independent of each other — both stages run their calls concurrently via
`concurrent.futures.ThreadPoolExecutor` since these are I/O-bound network
calls. This is the primary lever for keeping demo wait time short: wall-clock
per stage becomes roughly "slowest call in that stage" instead of "sum of all
calls in that stage."

Additional speed measures:
- Rasterized PDF page images are downscaled/compressed before being sent to
  the vision LLM (smaller upload, faster response).
- `st.status()` / `st.progress()` shows live per-stage progress messages so
  the wait reads as responsive during a live demo.

## 4. Project Structure

```
app.py                  # Streamlit UI + orchestration (stage sequencing, thread pool)
pipeline/
  pdf_ingest.py         # PDF -> list[PIL.Image], validation, downscaling
  story_gen.py          # StoryGenerator interface + OpenAIStoryGenerator, ClaudeStoryGenerator
  accent.py             # AccentDetector interface + OpenAIAudioAccentDetector
  voice_clone.py        # VoiceCloner interface + HumeVoiceCloner, ElevenLabsVoiceCloner
  tts.py                # NarrationSynthesizer interface + HumeNarrationSynthesizer, ElevenLabsNarrationSynthesizer
  sfx.py                # Freesound search/download + local disk cache
  mixer.py              # PyDub: loop/trim/overlay
  config.py             # loads secrets, resolves active provider classes
assets/sfx_cache/        # gitignored, runtime-populated by sfx.py
requirements.txt
.streamlit/secrets.toml.example
.gitignore
```

## 5. Component Design

### 5.1 Story generation (`story_gen.py`)

Interface: `StoryGenerator.generate(images, language) -> StoryResult(story_text, sfx_mood, tts_style_description)`.

Two concrete implementations (`OpenAIStoryGenerator`, `ClaudeStoryGenerator`)
wrap each provider's multimodal API with equivalent prompts: produce a short
story directly in the requested output language, with embedded expressive
tags (e.g. `[whisper]`, `[gasp]`), one SFX mood keyword, and a natural-language
delivery-style description for the TTS call. Active provider is chosen by a
constant in `config.py` (`STORY_PROVIDER = "openai" | "claude"`) — a
code-level switch, not an end-user UI toggle.

### 5.2 Accent detection (`accent.py`)

Interface: `AccentDetector.detect(audio_bytes) -> AccentResult(accent_label, detected_language)`,
implemented via an audio-capable multimodal LLM (e.g. GPT-4o-audio-preview).
Always runs once per submission (cheap, single call). Its result is only
*used* — folded into `tts_style_description` as a steering hint — when
`detected_language != output_language` (the cross-lingual case).

### 5.3 Voice cloning + narration synthesis (`voice_clone.py`, `tts.py`)

Common interfaces:
- `VoiceCloner.clone(audio_bytes, name) -> voice_id`
- `NarrationSynthesizer.synthesize(text, voice_id, style_description, language) -> audio_bytes`

`HumeVoiceCloner` / `HumeNarrationSynthesizer` implement these via Hume's
`create_custom_voice` + `tts.synthesize_json` (Octave). `ElevenLabsVoiceCloner`
/ `ElevenLabsNarrationSynthesizer` implement the same interfaces via
ElevenLabs' Instant Voice Cloning + multilingual TTS.

**Backend selection by language:**
- Mandarin → ElevenLabs (only Mandarin-capable option of the two).
- English → Hume first, with automatic fallback to ElevenLabs (see 5.4).

No other pipeline code needs to know which concrete backend is active for a
given run — `app.py` resolves it once per run and passes the resolved
`VoiceCloner`/`NarrationSynthesizer` pair through.

### 5.4 English fallback: Hume → ElevenLabs on quota exhaustion

For the English path only: attempt Hume voice-clone + Octave TTS first. If
Hume raises a quota/credit-exhausted error (rate-limit or billing error
response) at either the cloning or synthesis call, the run transparently
redoes **both** cloning and synthesis via ElevenLabs — a Hume `voice_id` and
an ElevenLabs `voice_id` are not interchangeable, so the fallback can't apply
mid-way through a single voice's lifecycle. A small `st.info("Switched to
backup voice engine")` notes the switch. This is automatic-detection-only (no
manual override toggle) — the app tries Hume and only falls back on an actual
quota error. Mandarin has no further fallback since ElevenLabs is already its
only backend among the two integrated here.

### 5.5 SFX (`sfx.py`, `mixer.py`)

`sfx.py` takes the vision-LLM's `sfx_mood` keyword, queries the Freesound API
(requires a Freesound API key/account, noted in setup docs), picks a result
favoring CC0/CC-BY-licensed previews, downloads the preview mp3, and caches it
on disk keyed by mood keyword so repeated moods across runs skip the network
call. `mixer.py` loops/trims the cached clip to the narration's length and
overlays it at a fixed **-18dB** offset beneath the narration via PyDub. If
Freesound returns no usable result, the dry narration is returned with an
`st.warning` rather than failing the whole generation.

## 6. Configuration & Secrets

API keys (OpenAI and/or Anthropic, Hume, ElevenLabs, Freesound) live in
`.streamlit/secrets.toml` (gitignored), loaded via `st.secrets` in
`config.py`. `.streamlit/secrets.toml.example` ships in the repo with
placeholder values. Provider-selection constants (`STORY_PROVIDER`,
`ACCENT_PROVIDER`) also live in `config.py`.

## 7. Error Handling

Surfaced as `st.error`/`st.warning` in the UI, never raw stack traces:

- PDF page-count sanity check and voice-sample duration check run locally
  *before* any paid API call, with a clear message on the required range.
- Every external API call (vision LLM, accent LLM, voice clone, TTS,
  Freesound) is caught at its call site and reported with which specific
  step failed.
- Freesound no-results is non-fatal (5.5).
- Each pipeline stage is wrapped so a later-stage failure preserves and still
  displays whatever succeeded earlier in the same run (e.g. the generated
  story text stays visible even if TTS later fails), instead of discarding
  everything.

## 8. Testing Approach

Given this is a local single-user POC built on several paid external APIs,
there is no full mocked-free integration test of the whole pipeline. Split:

- **Unit tests (pytest, no network):** PDF rasterization count/order,
  duration-validation boundaries, mood-keyword→search-query mapping,
  language→backend selection lookup, mixing/looping length math in
  `mixer.py`. All provider classes are faked/mocked here.
- **Manual smoke-test checklist** (run once with real API keys before the
  demo): one full English run end-to-end; one full Mandarin run end-to-end
  (verifying the ElevenLabs cross-lingual path); one run each with SFX
  on/off; one deliberately-too-short voice sample to confirm the validation
  message appears; one simulated Hume-quota-exhausted run to confirm the
  English fallback path (can be forced by temporarily using an
  already-exhausted/invalid Hume key).

## 9. Out of Scope (for this POC)

- Multi-user hosting, auth, concurrency/rate-limiting infrastructure.
- End-user-facing LLM/accent-provider selection UI (code-level constants only).
- Manual override toggle for the English TTS fallback.
- Any TTS backend for Mandarin fallback (ElevenLabs is the sole Mandarin
  backend; no further fallback exists if it fails).
