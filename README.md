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
  - OpenAI (vision story generation)
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

## Environment variables

- `STORY_PROVIDER`: `openai` (default) or `claude` — selects the vision-LLM
  used for story generation.
