# StoryTeller

Turns a short picture-book PDF (or a picture/camera snapshot) and a voice sample into a narrated story, read back sentence-by-sentence in your own cloned voice, in **English or Mandarin**, with theatre stage directions, per-sentence emotion badges, optional ambience, and a **Companion** Q&A panel.

**Default voice path:** free local **Coqui XTTS-v2** (same approach as the Havoc EN-dub pipeline). **ElevenLabs** remains as a paid backup (`TTS_BACKEND=elevenlabs`).

See `docs/superpowers/specs/2026-08-05-storyteller-pipeline-design.md` for the original design.

## Prerequisites

- Python 3.10+ (3.12 recommended for `coqui-tts`)
- ffmpeg on PATH
- GPU recommended for XTTS (CUDA). CPU works but is slow.
- API keys for story generation (`STORY_PROVIDER`: openai / claude / gemini / grok)
- Optional: Freesound (ambience). ElevenLabs only if using the paid backup.

## Setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill keys; ELEVENLABS_API_KEY only needed for TTS_BACKEND=elevenlabs
export COQUI_TOS_AGREED=1
```

Optional: `assets/voices/{warm,bright,gentle}.wav` and `assets/sfx/*.mp3` — not required.

## Running

```bash
# Default: free XTTS clone (EN + ZH)
streamlit run app.py

# Paid backup
TTS_BACKEND=elevenlabs streamlit run app.py
```

Pick theme/language (EN/ZH), story source, then a voice:

- **Built-in voice** — Warm / Bright / Gentle samples in `assets/voices/`
- **My voice → Upload** — WAV/MP3 file
- **My voice → Record** — browser mic (`st.audio_input`, ≥ ~6 s; XTTS auto-trims ~12 s)

Then Generate. After narration, open **Companion** to:
- **Ask by voice** (mic) or text — answers are spoken in the narrator voice
- **Continue story** — generates and narrates the next short beat into the full story

## Pipeline (current)

1. Vision (or embedded PDF text) → story sentences (EN or 中文)
   - **Text PDF:** if language picker ≠ PDF script, translate (EN↔中文) then TTS
2. **Theatre adaptation** → speakers, stage directions, pitch/rate/volume (1–5)
3. **XTTS** zero-shot clone → per-sentence WAV + DSP prosody  
   *(or ElevenLabs one-shot + timestamp slice if `TTS_BACKEND=elevenlabs`)*
4. Optional Freesound ambience by emotion
5. **Companion** — retrieve heard passages + session Q&A → reasoner API

## Running tests

```bash
pytest
```

Unit tests run offline against faked providers. XTTS GPU smoke is manual.

## Environment variables

| Var | Default | Meaning |
| :--- | :--- | :--- |
| `STORY_PROVIDER` | `openai` | `openai` / `claude` / `gemini` / `grok` |
| `TTS_BACKEND` | `xtts` | `xtts` (free) or `elevenlabs` (paid backup) |
| `COQUI_TOS_AGREED` | — | Set to `1` for XTTS |
