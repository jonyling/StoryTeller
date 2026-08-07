# StoryTeller — PDF-to-Narrated-Story Pipeline: Design Spec

**Date:** 2026-08-05 (original) — Section 3 onward revised same day after
merging a teammate's UI design
**Status:** Implemented
**Scope:** POC for a live demo. Local, single-user Streamlit app.

## 1. Purpose

A user uploads a short picture-book PDF (~4 images), a picture, or a camera
snapshot, plus a voice sample; picks a visual theme and language (English/
Mandarin — one control drives both UI copy and story language); and picks
whether to include background ambience. The app generates a short story
grounded in the images, breaks it into individual sentences (each with an
invented or narrator `speaker`, and a fixed-taxonomy `emotion`), clones the
user's voice, and plays the story back **one sentence at a time**, each with
its own narration clip and an emotion-matched looping ambience track when
enabled.

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

## 3. UI Merge: Per-Sentence Architecture (Option 3a)

A teammate independently designed and built a themed, fully-mocked Streamlit
UI (4 visual themes × EN/ZH, three story sources, a voice picker, a
sentence-by-sentence player) with two integration seams —
`extract_pages_from_pdf()` (real `pypdf` text extraction) and
`generate_mock_story()` (the mock to replace) — and this data contract:

```
list[{
    "page": int,
    "sentences": list[{
        "text": str,           # may contain inline [whisper]/[gasp]-style tags
        "speaker": str,        # "narrator" or an invented character name
        "emotion": str,        # "angry" | "excited" | "sad" | "calm" | "neutral"
        "pitch": int,          # 1-5, cosmetic (see below)
        "volume": int,         # 1-5, cosmetic
        "rate": int,           # 1-5, cosmetic
        "audio_path": bytes,   # real narration clip for this sentence
    }]
}]
```

This is a fundamentally different granularity from what Section 1-2 originally
assumed: **one narration clip per sentence**, with per-sentence emotion
driving which ambience loop plays, rather than one story-wide narration +
one story-wide mood. Reconciling this against a real backend had three
options, worked through with the project owner before implementing:

1. **Keep whole-story generation, flatten the UI down** to one shared audio
   player — simplest, but drops the sentence-by-sentence experience the UI
   was built for.
2. **Full per-sentence architecture**: the LLM tags emotion per sentence
   *and* ElevenLabs is called once per sentence, each independently tunable
   via `voice_settings` from that sentence's `pitch`/`volume`/`rate`. Richest
   possible result, but multiplies API calls (~15-20 per story, based on a
   ~200-word budget at 8-15 words/sentence), which on Free/Starter/Creator
   ElevenLabs tiers (2/3/5 concurrent request limits) turns one ~5-8s
   narration call into several sequential batches, and compounds failure
   risk to roughly a 1-in-6 chance *some* sentence fails per run even at a
   generous 99%-per-call success rate — a real conflict with the "keep the
   demo fast and reliable" priority.
3. **Chosen: one synthesis call, sliced after the fact.** The vision LLM
   still writes one flowing story, but returns it pre-split into sentences
   with per-sentence `emotion` (LLM-authored, richer than keyword-matching)
   instead of a whole-story `sfx_mood`. All sentences are joined into one
   string and synthesized in **one** ElevenLabs call using
   `convert_with_timestamps` (confirmed real: returns character-level
   `start`/`end` times), then sliced into one clip per sentence locally via
   `pipeline/audio_utils.py: slice_audio_by_sentences`. This preserves the
   UI's per-sentence player using only one API call — the entire point of
   this option — at the cost of not being able to independently tune each
   sentence's vocal delivery (it's one continuous take, sliced, not N
   distinct performances). `pitch`/`volume`/`rate` are therefore populated
   from a fixed per-emotion lookup table (`_EMOTION_DSP_DEFAULTS`, matching
   the UI's own `EMOTION_DSP_DEFAULTS`) rather than independently computed —
   they're carried through the schema for UI compatibility but don't drive
   real per-sentence audio DSP. This is the same category of accepted
   simplification as the whole-story `tts_style_description` gap noted in
   the original design (Section 5.1) — visible in the schema, not acted on.

**Ambience** is handled the same way the UI already designed it: a separate,
simultaneously-playing looping `<audio>` element per sentence (not mixed into
the narration file), keyed by that sentence's `emotion`. This meant
`pipeline/mixer.py` (the old whole-story mix-into-one-file approach) became
dead code and was deleted — nothing merges ambience into the narration
audio anymore.

**Speaker names:** the vision LLM is asked to invent simple character names
for identifiable figures in the images (e.g. "Ember" for a dragon) and use
`"narrator"` for descriptive lines — a prompt change, not an architecture
change, since `speaker` is a display label only; there is still exactly one
cloned voice narrating the whole story.

**Page grouping:** everything is returned under a single `page: 1` — the
UI's rendering already flattens all pages into one continuous sentence list
regardless of page number, so this isn't load-bearing, and our story is one
flowing narrative rather than naturally one paragraph per source image.

**PDF text-extraction fallback:** `extract_pages_from_pdf()` (the
teammate's `pypdf`-based real text extraction + keyword emotion tagging) is
kept as-is and tried first. It only succeeds for a PDF that already
contains real embedded text — rare for an actual picture book, but a
legitimate lighter-weight path when it applies (skips vision generation
entirely, using the already-extracted text and keyword-tagged emotions
directly). When it returns `None` (the normal picture-book case, no
extractable text), the original raw PDF bytes are rasterized via
`pipeline/pdf_ingest.py` and run through the full vision pipeline below.

## 4. High-Level Architecture

Single local Streamlit app (`streamlit run app.py`). One story processed per
run; no background workers, job queues, or multi-user concurrency (local
single-user POC).

```
[Streamlit UI: theme/language header, story source (PDF/Picture/Camera),
 voice picker (own upload or built-in preset), ambience toggle, Generate]
        │
        ▼
0. Source resolution   PDF with real text → extract_pages_from_pdf() (pypdf +
        │              keyword tagging), skip straight to step 4. Otherwise:
        │              PDF rasterized to page images (pdf_ingest.py), or a
        │              single Picture/Camera image opened directly.
        ▼
0.5 Validation          Voice sample duration checked against ElevenLabs'
        │              Instant Voice Cloning requirement (~1-5 min) before
        │              any paid API call fires.
        ▼
┌─────────────────────────── Stage 1 (parallel) ───────────────────────────┐
│  1a. Vision → Story    Vision-LLM takes image(s) + language, returns a   │
│                        list of sentences: { text (with [whisper]/[gasp]  │
│                        tags), speaker, emotion }                        │
│  1b. Voice Cloning     ElevenLabs Instant Voice Cloning registers the    │
│                        voice sample → voice_id                          │
└────────────────────────────────────────────────────────────────────────┘
        ▼
┌─────────────────────────── Stage 2 (parallel) ───────────────────────────┐
│  2a. Narration Synth   ONE ElevenLabs convert_with_timestamps call on    │
│                        all sentences joined together → narration audio  │
│                        + character-level alignment                      │
│  2b. Ambience Fetch    (only if enabled) one Freesound fetch per         │
│                        *distinct* emotion actually used in the story     │
└────────────────────────────────────────────────────────────────────────┘
        ▼
3. Per-Sentence Slicing  slice_audio_by_sentences() locates each sentence's
        │                text in the alignment (exact match → bracket-tag-
        │                stripped match → heuristic fallback) and cuts one
        │                clip per sentence from the single narration take.
        ▼
[ {page: 1, sentences: [...]} ]  → per-sentence player, one at a time,
                                    with matching looping ambience if enabled
```

Stage 1's two calls are independent of each other, and Stage 2's calls are
independent of each other — both stages run concurrently via
`concurrent.futures.ThreadPoolExecutor`. This is the primary lever for
keeping demo wait time short: wall-clock per stage becomes roughly "slowest
call in that stage" instead of "sum of all calls," and per-emotion ambience
fetches (typically 1-5, one per distinct emotion, not per sentence) run
alongside the one narration call rather than serially after it.

Additional speed/UX measures:
- Rasterized PDF page images are downscaled/compressed before being sent to
  the vision LLM (smaller upload, faster response).
- `run_pipeline`'s `on_progress` callback updates the UI's themed loading
  placeholder live between stages, so the wait reads as responsive.

## 5. Project Structure

```
app.py                  # Streamlit UI (theme/i18n/CSS + real backend wiring)
pipeline/
  pdf_ingest.py         # PDF -> list[PIL.Image], validation, downscaling
  story_gen.py          # StoryGenerator interface + OpenAI/Claude/Gemini/Grok generators,
                         # per-sentence contract (StorySentence: text/speaker/emotion)
  voice_clone.py        # VoiceCloner interface + ElevenLabsVoiceCloner
  tts.py                # NarrationSynthesizer interface + ElevenLabsNarrationSynthesizer,
                         # convert_with_timestamps -> NarrationAudio (audio + alignment)
  audio_utils.py        # duration validation + slice_audio_by_sentences
  sfx.py                # Freesound search/download + local disk cache
  orchestrator.py       # ties it all together, per-emotion ambience, no mixing
  errors.py             # PipelineError / ValidationError hierarchy
  config.py             # loads secrets, STORY_PROVIDER constant
requirements.txt
.streamlit/secrets.toml.example
.gitignore
```

`pipeline/mixer.py` (whole-story ambience mixing) and `pipeline/accent.py`
(accent detection) were both built earlier in this project's history and
later deleted once their approach was superseded — see Section 3 and
Section 2's last bullet respectively.

## 6. Component Design

### 6.1 Story generation (`story_gen.py`)

Interface: `StoryGenerator.generate(images, language) -> StoryResult(sentences: list[StorySentence])`,
where `StorySentence` has `text`, `speaker`, `emotion` (normalized to one of
`{"angry", "excited", "sad", "calm", "neutral"}`, falling back to `"neutral"`
for anything else the LLM returns — mirroring the UI's own keyword-tagger's
fallback behavior).

Four concrete implementations (`OpenAIStoryGenerator`, `ClaudeStoryGenerator`,
`GeminiStoryGenerator`, `GrokStoryGenerator`) wrap each provider's multimodal
API with equivalent prompts: produce a short story (~200 words) directly in
the requested output language, broken into sentences, each with inline
`[whisper]`/`[gasp]`-style delivery tags where appropriate, an invented
character name or `"narrator"` as speaker, and one fixed-taxonomy emotion.
`GrokStoryGenerator` reuses the `openai` SDK pointed at xAI's
OpenAI-compatible endpoint (`base_url="https://api.x.ai/v1"`) rather than a
separate SDK — both it and `OpenAIStoryGenerator` share one internal
`_generate_via_openai_compatible_chat` helper. `GeminiStoryGenerator` uses
Google's `google-genai` SDK (`client.models.generate_content` with
`types.Part.from_bytes` images and `response_mime_type="application/json"`).
Active provider is chosen by `STORY_PROVIDER` in `config.py` — a code-level
switch, not an end-user UI toggle. The EN/ZH UI language picker doubles as
the story-language selector (`app.py: STORY_LANGUAGE = {"EN": "English", "ZH": "Mandarin"}`).

### 6.2 Voice cloning + narration synthesis (`voice_clone.py`, `tts.py`)

Single backend for both languages: **ElevenLabs**.
- `client.voices.ivc.create(name=..., files=[...]) -> AddVoiceIvcResponseModel`
  clones a voice from the uploaded sample. `voice_clone.py` sniffs the
  uploaded bytes' magic numbers (RIFF → wav, ID3/MPEG frame sync → mp3, else
  a generic fallback) to label the upload correctly regardless of its real
  format, since the file extension alone isn't trustworthy.
- `client.text_to_speech.convert_with_timestamps(voice_id, text=..., model_id="eleven_v3", ...) -> AudioWithTimestampsResponse`
  synthesizes **all of the story's sentences joined into one string** in a
  single call, returning base64 audio plus character-level
  `character_start_times_seconds`/`character_end_times_seconds` alignment.
  `tts.py` decodes this into a `NarrationAudio(audio_bytes, characters,
  character_start_times_seconds, character_end_times_seconds)`.

`ElevenLabsVoiceCloner` / `ElevenLabsNarrationSynthesizer` are the only
concrete implementations, used directly for both English and Mandarin —
there is no per-language backend selection or fallback logic.

### 6.3 Per-sentence audio slicing (`audio_utils.py`)

`slice_audio_by_sentences(audio_bytes, characters, character_start_times_seconds,
character_end_times_seconds, sentence_texts) -> list[bytes]` locates each
sentence's text within the alignment's reconstructed character string and
cuts a clip for it from the one synthesized take:
1. Exact substring match (search starts from where the previous sentence's
   match ended, to avoid false matches on repeated words).
2. If not found, strip `[bracket tags]` and retry — ElevenLabs' alignment
   may not preserve inline delivery-tag text verbatim.
3. If still not found, fall back to a positional heuristic slice starting
   at the current search cursor, rather than raising — one unmatched
   sentence shouldn't break playback for the rest of the story. This is a
   best-effort mechanism, not exact-alignment guaranteed; the manual
   smoke-test checklist (README) includes explicitly listening through a
   real generated story's sentences to confirm each clip matches its text,
   since this is the one part of the pipeline unverifiable from unit tests
   alone (they use synthetic, evenly-spaced timestamps).

### 6.4 SFX (`sfx.py`)

`orchestrator.py` calls `fetch_ambience_clip(mood, api_key, cache_dir)` once
per **distinct emotion** actually present in the story (via a fixed
`_EMOTION_SFX_MOOD` map: angry→"thunderstorm", excited→"carnival
atmosphere", sad→"gentle rain", calm→"flowing river", neutral→"room tone"),
not once per whole-story mood. There is no mixing step — the UI plays each
fetched clip as its own simultaneous looping element per sentence, exactly
as it was already designed to.

**Selection logic (revised after a real quality complaint):** live-testing
the original five mood queries against Freesound found the top result for
two of them was a poor match despite better candidates being one or two
positions lower in the same result set — "cheerful sparkle" (excited)
returned a calm/mellow chime track; "quiet room tone" (neutral) returned an
orchestral/choir recording. Root cause: the original implementation only
ever requested one result (`page_size=1`), sorted by `sort=rating_desc` —
but every real result checked came back with `rating: None`, so that sort
was silently doing nothing. Fixed by requesting 10 candidates sorted by
`downloads_desc` (a field Freesound actually populates) and re-ranking them
by how many ambience-indicating tags each has (`ambient`, `ambience`,
`field-recording`, `room-tone`, `soundscape`, etc. — see
`_AMBIENCE_TAG_HINTS` in `sfx.py`), falling back to the downloads-sorted
order when no candidate matches any hint (never worse than the old
behavior, only better when a clearly better-tagged option exists). The two
worst-performing mood phrases were also reworded to more literal sound
descriptors ("cheerful sparkle" → "carnival atmosphere", "quiet room tone" →
"room tone"), verified live to return clean, consistently on-vibe results.
No license filtering still applied — same known POC-level gap as before.
Once fetched, a clip is cached by mood keyword indefinitely; this fix
doesn't change that, so a newly-fixed mood only actually improves once its
old cache entry (if any) has expired or been cleared.

### 6.5 UI (`app.py`)

The teammate's themed UI (4 themes × EN/ZH via `st.segmented_control` bound
directly to `st.session_state` keys — no `default=`/compare/`st.rerun()`
pattern, since that has a stale-first-click bug; three story sources; a
voice picker; sentence-by-sentence navigation with autoplay) is unchanged in
its rendering, CSS, and session-state structure. The only real-backend
touches:
- `generate_mock_story()` — same two positional params (`pages`,
  `voice_file`) and same `{page, sentences: [...]}` return schema, extended
  with keyword-only params (`raw_source_bytes`, `language`, `enable_sfx`,
  `on_progress`) to carry what the mock never needed. Real implementation
  builds whichever provider client `STORY_PROVIDER` needs, validates voice
  duration, and calls `run_pipeline`; any `PipelineError` or unexpected
  exception is shown via `st.error` and falls back to `MOCK_STORY_PAGES`
  (preserving the UI's existing `used_fallback` detection, which compares
  by identity against that exact object).
- The Generate button handler now also keeps the raw PDF bytes or opened
  image around (previously discarded once `extract_pages_from_pdf()` had
  been tried) so the real vision path has something to rasterize/read when
  there's no extractable text.
- The per-sentence ambience lookup now checks `st.session_state.ambience_by_emotion`
  (populated by `generate_mock_story()` after a real run) before falling
  back to the UI's static `SFX` file-path dict — fetched clips are raw
  `bytes`, not file paths, and `st.audio()` accepts both. This has to be
  `st.session_state`, not a plain reassigned module global: Streamlit
  reruns the whole script top-to-bottom on every interaction, so a
  module-level `SFX = {...}` reassignment made during generation would be
  silently reset back to the static dict on the very next rerun (e.g.
  clicking "Next"), while `session_state` persists correctly across reruns.

## 7. Configuration & Secrets

API keys (OpenAI and/or Anthropic and/or Gemini and/or xAI — whichever
`STORY_PROVIDER` is active — plus ElevenLabs, Freesound) live in
`.streamlit/secrets.toml` (gitignored), loaded via `st.secrets` in
`config.py`. `.streamlit/secrets.toml.example` ships in the repo with
placeholder values. The provider-selection constant `STORY_PROVIDER` also
lives in `config.py`.

## 8. Error Handling

Surfaced as `st.error`/`st.warning` in the UI, never raw stack traces:

- Voice-sample duration check runs locally *before* any paid API call, with
  a clear message on the required range.
- A malformed/unreadable PDF passed to `extract_pages_from_pdf()` is
  treated the same as "no extractable text" (caught at the call site in
  `app.py`) rather than crashing the page.
- Every external API call in the real generation path is covered by one
  broad exception handler in `generate_mock_story()`, which reports a
  generic "Generation failed: {exc}" (or the specific message for a
  `PipelineError`) and falls back to the mock story rather than attributing
  the failure to a specific step or preserving partial results. This is an
  accepted simplification for this POC, not a gap.
- Freesound no-results, or an outright fetch error, is non-fatal per
  emotion — that emotion's sentences simply play without ambience.
- Audio-slicing's positional-heuristic fallback (6.3) means a sentence
  whose text can't be located in the alignment still gets *a* clip rather
  than breaking the run, at the cost of that one clip's precision.

## 9. Testing Approach

Given this is a local single-user POC built on several paid external APIs,
there is no full mocked-free integration test of the whole pipeline. Split:

- **Unit tests (pytest, no network):** PDF rasterization count/order,
  duration-validation boundaries, the four story generators' per-sentence
  parsing (including emotion-normalization fallback), the audio-slicing
  utility's exact/bracket-stripped/heuristic-fallback paths, the
  timestamp-based narration synthesis wrapper, and the orchestrator's
  per-distinct-emotion ambience fetching and non-fatal-failure behavior.
  All provider classes are faked/mocked here.
- **Manual smoke-test checklist** (see README): one full English run with a
  real picture-book PDF; one full Mandarin run; one run each via
  Picture/Camera source and via a real-text PDF (skips vision generation);
  SFX on/off; a deliberately-too-short voice sample; and, specifically
  because it can't be verified from unit tests (which use synthetic,
  evenly-spaced timestamps), listening through a real generated story's
  sentences to confirm each sliced clip actually matches its displayed
  text.

## 10. Out of Scope (for this POC)

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
- True per-sentence audio DSP (independently-tuned pitch/volume/rate per
  sentence via separate synthesis calls) — the explicitly rejected
  higher-fidelity alternative to the one-call-plus-slicing approach in
  Section 3, on cost/latency/reliability grounds. `pitch`/`volume`/`rate`
  remain schema-only, populated from a fixed per-emotion lookup table.
- License-aware SFX selection (favoring CC0/CC-BY Freesound results) — a
  known gap carried over from the original design, unaffected by this
  merge.

## 11. Post-implementation updates (2026-08-06)

The live app diverged from Sections 2–10 where noted below. Historical
wording above is kept for design context; treat this section as current.

| Topic | Original spec | Current implementation |
| :--- | :--- | :--- |
| TTS backend | ElevenLabs sole backend | **XTTS-v2 default** (`TTS_BACKEND=xtts`); ElevenLabs optional backup |
| Narration | One ElevenLabs call + timestamp slice | **Per-sentence XTTS** (+ chunking for long lines); theatre **DSP prosody** via `pipeline/prosody.py` |
| Voice sample length | 60–300 s (ElevenLabs IVC) | **~6–20 s** ref OK for XTTS (`pipeline/xtts_backend.py`) |
| Text PDF | Vision skip implied | Extracted text narrated; **EN↔中文 translate** when picker ≠ script (`pipeline/translate.py`) |
| Playback UI | Sentence-by-sentence primary | **Full-story player** + chronological **timeline**; sentence follow-along in expander |
| Companion | Out of scope in §10 | **Implemented** — Q&A, voice ask, Continue story (`docs/2026-08-05-storyteller-companion-mode-addendum.md` §9) |
| Mic | Streamlit `st.audio_input` | Custom **`mic_component/`** (device picker, level check, preview) |
| Mandarin deps | ElevenLabs cross-lingual | XTTS + **`pypinyin`** in `requirements.txt` |

**Changelog:** [`CHANGES.md`](../../../CHANGES.md) at repo root.
