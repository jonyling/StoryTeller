# StoryTeller

Turns a short picture-book PDF (or a picture/camera snapshot) and a voice
sample into a narrated story, read back sentence-by-sentence in your own
cloned voice, in English or Mandarin, with per-sentence emotion badges and
optional matching background ambience. Four visual themes × EN/ZH, picked in
the header. See `docs/superpowers/specs/2026-08-05-storyteller-pipeline-design.md`
for the full design.

## Prerequisites

- Python 3.10+
- ffmpeg installed and on PATH (required by `pydub` for audio processing):
  - Windows: `winget install ffmpeg` or `choco install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`
- API keys for:
  - OpenAI (vision story generation, only if `STORY_PROVIDER=openai`, the default)
  - Anthropic (only if `STORY_PROVIDER=claude`)
  - Gemini (only if `STORY_PROVIDER=gemini`)
  - xAI/Grok (only if `STORY_PROVIDER=grok`)
  - ElevenLabs (voice cloning + narration synthesis)
  - Freesound (background ambience search) — request one directly from the
    Freesound developer portal (https://freesound.org/apiv2/apply/); see
    the design spec for how it's used here.

## Setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then fill in real API keys in .streamlit/secrets.toml
```

Optional, not required to run: `assets/voices/{warm,bright,gentle}.wav` (the
three built-in "Default" voice presets) and `assets/sfx/*.mp3` (static
fallback ambience tracks). Neither ships in this repo — the app works fully
without them. Default-voice presets just won't be selectable until those
files exist; ambience always prefers a freshly Freesound-fetched clip over
the static fallback regardless.

## Running

```bash
streamlit run app.py
```

Pick a theme/language in the header, a story source (PDF / Picture / Camera),
a voice (upload your own, or a built-in preset if those asset files exist),
then Generate. The story plays back one sentence at a time, each with its
own narration clip, emotion badge, and (if the ambience toggle is on)
looping background sound matched to that sentence's mood.

## Running tests

```bash
pytest
```

All unit tests run offline against faked provider clients. There is no
automated end-to-end test for `app.py` (Streamlit UI is verified manually) —
see the manual smoke-test checklist below.

## How a run works

1. **Story source → images.** A PDF with real embedded text (rare for a
   picture book) is read directly via `pypdf` (`extract_pages_from_pdf`) and
   keyword-tagged — no vision call needed. A PDF with no extractable text
   (the normal picture-book case) is rasterized to page images; a
   Picture/Camera upload is used as a single image. Either way, real story
   generation only runs when there's no pre-existing text to read.
2. **Vision LLM → per-sentence story.** The active `STORY_PROVIDER` looks at
   the image(s) and returns a list of sentences, each with `text` (possibly
   containing inline `[whispers]`/`[gasps]`-style delivery tags), a
   `speaker` (a character name it invents from what's in the images, or
   `"narrator"`), and one `emotion` from a fixed set: `angry` / `excited` /
   `sad` / `calm` / `neutral`.
3. **Voice cloning + one narration call.** ElevenLabs clones the uploaded
   voice sample, then synthesizes the *entire* story's sentences (joined
   into one string) in a single `eleven_v3` call with character-level
   timestamps (`convert_with_timestamps`) — one API call regardless of
   sentence count, not one call per sentence.
4. **Per-sentence slicing.** The one narration audio is sliced into
   individual sentence clips using those timestamps (`pipeline/audio_utils.py:
   slice_audio_by_sentences`), so the UI's sentence-by-sentence player gets a
   real, distinct clip per sentence without the cost/latency/reliability
   hit of N separate synthesis calls.
5. **Per-emotion ambience.** If the ambience toggle is on, one Freesound
   clip is fetched per *distinct* emotion actually used in the story (not
   per sentence) and cached; the UI plays it as a separate looping track
   alongside whichever sentence is showing, keyed by that sentence's
   emotion — no audio mixing involved.

## Manual smoke-test checklist (run once with real API keys before a demo)

- [ ] **Run this first, before anything else below:** one real English
      synthesis call succeeds end-to-end. ElevenLabs' `language_code`
      parameter has historically been restricted to certain models, and this
      code always sends it to `eleven_v3`. If a real call rejects it, every
      run fails — English included, not just Mandarin — so this needs to be
      confirmed before working through the rest of the checklist.
- [ ] Full English run with a real picture-book PDF (no embedded text, SFX
      on): story renders sentence-by-sentence, each sentence has its own
      audio clip, narration sounds like the uploaded voice, ambience changes
      as the emotion changes across sentences.
- [ ] Full Mandarin run end-to-end: story text and narration are in
      Mandarin, still in the cloned voice (ElevenLabs cross-lingual
      clone-once-speak-any-language path).
- [ ] One run via the Picture or Camera source instead of PDF: confirm the
      single image is used for real vision generation (not the mock
      fallback story).
- [ ] One run with a PDF that already has real embedded text (e.g. a typed
      story, not a picture book): confirm it uses that text directly
      (keyword-tagged emotions) rather than calling the vision LLM.
- [ ] One run with SFX off: confirm no background ambience plays for any
      sentence.
- [ ] One deliberately-too-short voice sample (under 60s): confirm the
      validation error appears before any paid API call fires.
- [ ] Sentence-slicing sanity check: page through a few sentences with the
      Next/Previous buttons and confirm each one's audio actually matches
      its displayed text (not silence, not a different sentence's clip) —
      this is the one part of the pipeline that can't be verified from unit
      tests alone, since it depends on ElevenLabs' real character-alignment
      output.
- [ ] ElevenLabs voice-slot check: every run creates a new ElevenLabs custom
      voice (named "StoryTeller Voice") via Instant Voice Cloning and never
      deletes it. Most ElevenLabs plans cap the number of custom voice
      slots. Before a demo, check the ElevenLabs dashboard's voice count
      isn't near the plan's limit, and periodically delete old "StoryTeller
      Voice" entries accumulated from testing.

## Environment variables

- `STORY_PROVIDER`: `openai` (default), `claude`, `gemini`, or `grok` —
  selects the vision-LLM used for story generation.
