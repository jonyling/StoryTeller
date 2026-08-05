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
  no Chinese variant, none announced as planned.
- **Hume's public API cannot clone a voice from an uploaded audio sample at
  all**, for either language. Verified directly against the
  `HumeAI/hume-python-sdk` source: the only voice-creation method,
  `client.tts.voices.create()`, takes a `generation_id` from a prior TTS
  synthesis call, never an audio file. `client.tts.convert_voice_file/json()`
  ("Voice Conversion") does accept an `audio` parameter, but it re-renders
  *existing* speech into an *already-existing* voice — the opposite
  direction from cloning a new voice out of a sample. The "record or upload
  → Create Voice" flow shown on Hume's web dashboard has no corresponding
  public API route. **Decision: Hume is dropped from the narration/voice-
  cloning role entirely. ElevenLabs is the sole narration backend for both
  English and Mandarin** — confirmed via its Python SDK source
  (`elevenlabs/elevenlabs-python`): `client.voices.ivc.create(name, files)`
  clones a voice from an uploaded sample and returns a real `voice_id`, and
  `client.text_to_speech.convert(voice_id, text, model_id="eleven_v3", ...)`
  synthesizes it. The `eleven_v3` model additionally supports the inline
  `[whisper]`/`[gasp]`/`[laughs]`-style delivery tags our story-generation
  step produces, across 70+ languages including Mandarin — so the
  expressive-tag design from the original guide still works, just through a
  different provider than it assumed.
- **Accent detection has no channel to actually steer ElevenLabs output, so
  it's dropped.** The original rationale (from brainstorming) was
  cross-lingual steering: detect the sample's accent so a hint like "speak
  Mandarin with a Singaporean-English-influenced accent" could be fed to the
  narration call, mirroring Hume's free-text `description` field. ElevenLabs'
  `eleven_v3` has no equivalent natural-language delivery-description input —
  its only text-embedded control is short bracketed audio-tag cues
  (`[whispers]`, `[gasps]`), and a full accent-hint sentence risks being read
  aloud verbatim if forced into that channel (confirmed as a real risk during
  final review). Since ElevenLabs' Instant Voice Cloning already carries the
  speaker's timbre across languages on its own (that's the whole basis for
  picking it as the cross-lingual backend in the first place), a separate
  accent-detection call bought nothing — it was a paid API call with no
  observable effect on the output. **Decision: drop the accent-detection
  stage and `pipeline/accent.py` entirely.**

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
        │              checked against ElevenLabs' Instant Voice Cloning
        │              requirement (~1-5 min) before any paid API call fires.
        ▼
┌─────────────────────────── Stage 1 (parallel) ───────────────────────────┐
│  1a. Vision → Story    Vision-LLM takes page images + language, returns   │
│                        { story_text (with [whisper]/[gasp]-style tags),  │
│                          sfx_mood, tts_style_description }               │
│  1b. Voice Cloning     ElevenLabs Instant Voice Cloning registers the    │
│                        voice sample → voice_id                          │
└────────────────────────────────────────────────────────────────────────┘
        ▼
┌─────────────────────────── Stage 2 (parallel) ───────────────────────────┐
│  2a. Narration Synth   ElevenLabs synthesize(story_text, voice_id,       │
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

Stage 1's two calls are independent of each other (each depends only on
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
  voice_clone.py        # VoiceCloner interface + ElevenLabsVoiceCloner
  tts.py                # NarrationSynthesizer interface + ElevenLabsNarrationSynthesizer
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
delivery-style description. Active provider is chosen by a constant in
`config.py` (`STORY_PROVIDER = "openai" | "claude"`) — a code-level switch,
not an end-user UI toggle. The delivery-style description is accepted by
`NarrationSynthesizer.synthesize` for interface stability but is not
currently forwarded to ElevenLabs (see 5.2) — dynamic delivery comes entirely
from the inline `[whisper]`/`[gasp]`-style tags embedded directly in
`story_text`, which `eleven_v3` natively interprets.

### 5.2 Voice cloning + narration synthesis (`voice_clone.py`, `tts.py`)

Single backend for both languages: **ElevenLabs**. Confirmed via its Python
SDK source:
- `client.voices.ivc.create(name=..., files=[...]) -> AddVoiceIvcResponseModel`
  with a `.voice_id` field — this is the real, working voice-cloning-from-a-
  sample call (Hume has no equivalent public endpoint; see Section 2).
- `client.text_to_speech.convert(voice_id, text=..., model_id="eleven_v3",
  output_format=..., language_code=...) -> Iterator[bytes]` — synthesizes the
  story text (with inline `[whisper]`/`[gasp]`-style tags, which `eleven_v3`
  natively interprets) in the cloned voice. Returned as a byte-chunk
  iterator; the wrapper joins it into a single `bytes` object.

Interfaces (kept as interfaces, not just concrete functions, so a second
backend could be added later without touching calling code — but only one
implementation is built now, per YAGNI):
- `VoiceCloner.clone(audio_bytes, name) -> voice_id`
- `NarrationSynthesizer.synthesize(text, voice_id, style_description, language) -> audio_bytes`

`ElevenLabsVoiceCloner` / `ElevenLabsNarrationSynthesizer` are the only
concrete implementations. `app.py` uses them directly for both English and
Mandarin — there is no per-language backend selection or fallback logic.

### 5.3 SFX (`sfx.py`, `mixer.py`)

`sfx.py` takes the vision-LLM's `sfx_mood` keyword, queries the Freesound API
(requires a Freesound API key/account, noted in setup docs), picks a result
favoring CC0/CC-BY-licensed previews, downloads the preview mp3, and caches it
on disk keyed by mood keyword so repeated moods across runs skip the network
call. `mixer.py` loops/trims the cached clip to the narration's length and
overlays it at a fixed **-18dB** offset beneath the narration via PyDub. If
Freesound returns no usable result, the dry narration is returned with an
`st.warning` rather than failing the whole generation.

## 6. Configuration & Secrets

API keys (OpenAI and/or Anthropic, ElevenLabs, Freesound) live in
`.streamlit/secrets.toml` (gitignored), loaded via `st.secrets` in
`config.py`. `.streamlit/secrets.toml.example` ships in the repo with
placeholder values. The provider-selection constant `STORY_PROVIDER` also
lives in `config.py`.

## 7. Error Handling

Surfaced as `st.error`/`st.warning` in the UI, never raw stack traces:

- PDF page-count sanity check and voice-sample duration check run locally
  *before* any paid API call, with a clear message on the required range.
- Every external API call (vision LLM, voice clone, TTS, Freesound) is
  caught at its call site and reported with which specific step failed.
- Freesound no-results is non-fatal (5.3).
- Each pipeline stage is wrapped so a later-stage failure preserves and still
  displays whatever succeeded earlier in the same run (e.g. the generated
  story text stays visible even if TTS later fails), instead of discarding
  everything.

## 8. Testing Approach

Given this is a local single-user POC built on several paid external APIs,
there is no full mocked-free integration test of the whole pipeline. Split:

- **Unit tests (pytest, no network):** PDF rasterization count/order,
  duration-validation boundaries, mood-keyword→search-query mapping,
  mixing/looping length math in `mixer.py`. All provider classes are
  faked/mocked here.
- **Manual smoke-test checklist** (run once with real API keys before the
  demo): one full English run end-to-end; one full Mandarin run end-to-end
  (verifying ElevenLabs' cross-lingual clone-once-speak-any-language path);
  one run each with SFX on/off; one deliberately-too-short voice sample to
  confirm the validation message appears.

## 9. Out of Scope (for this POC)

- Multi-user hosting, auth, concurrency/rate-limiting infrastructure.
- End-user-facing LLM-provider selection UI (code-level constant only).
- Any fallback/backup TTS backend (ElevenLabs is the sole narration backend
  for both languages; no further fallback exists if it fails).
- Hume AI integration of any kind — dropped entirely per Section 2's
  findings. Could be revisited if Hume ships a public audio-upload voice-
  cloning endpoint in the future.
- Accent detection of any kind — dropped entirely per Section 2's findings
  (no channel exists to feed the hint to ElevenLabs). Could be revisited if
  ElevenLabs ships a natural-language delivery-description input.
