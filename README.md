# StoryTeller

Turns a short picture-book PDF and a voice sample into a narrated audio story
in your own cloned voice, in English or Mandarin, with optional background
ambience. See `docs/superpowers/specs/2026-08-05-storyteller-pipeline-design.md`
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
  - ElevenLabs (voice cloning + narration synthesis)
  - Freesound (background ambience search) — request one directly from the
    Freesound developer portal (https://freesound.org/apiv2/apply/); see
    `docs/superpowers/specs/2026-08-05-storyteller-pipeline-design.md` for how
    it's used here.

## Setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then fill in real API keys in .streamlit/secrets.toml
```

## Running

```bash
streamlit run app.py
```

## Running tests

```bash
pytest
```

All unit tests run offline against faked provider clients. There is no
automated end-to-end test — see the manual smoke-test checklist below.

## Manual smoke-test checklist (run once with real API keys before a demo)

- [ ] **Run this first, before anything else below:** one real English
      synthesis call succeeds end-to-end. ElevenLabs' `language_code`
      parameter has historically been restricted to certain models, and this
      code always sends it to `eleven_v3`. If a real call rejects it, every
      run fails — English included, not just Mandarin — so this needs to be
      confirmed before working through the rest of the checklist.
- [ ] Full English run end-to-end (PDF + 1-2 min voice sample, SFX on):
      story renders, audio plays, narration sounds like the uploaded voice.
- [ ] Full Mandarin run end-to-end: story text and narration are in Mandarin,
      still in the cloned voice (ElevenLabs cross-lingual clone-once-speak-
      any-language path).
- [ ] One run with SFX off: confirm no background ambience in the output.
- [ ] One run with SFX on but an unusual `sfx_mood` keyword: confirm the app
      falls back to dry narration with a warning instead of crashing, if
      Freesound returns no results.
- [ ] One deliberately-too-short voice sample (under 60s): confirm the
      validation error appears before any paid API call fires.
- [ ] ElevenLabs voice-slot check: every run creates a new ElevenLabs custom
      voice (named "StoryTeller Voice") via Instant Voice Cloning and never
      deletes it. Most ElevenLabs plans cap the number of custom voice
      slots. Before a demo, check the ElevenLabs dashboard's voice count
      isn't near the plan's limit, and periodically delete old "StoryTeller
      Voice" entries accumulated from testing.

## Environment variables

- `STORY_PROVIDER`: `openai` (default) or `claude` — selects the vision-LLM
  used for story generation.
