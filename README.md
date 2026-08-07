# StoryTeller

Turns a short picture-book PDF (or a picture/camera snapshot) and a voice sample into a narrated story, read back in your own cloned voice, in **English or Mandarin**, with theatre stage directions, per-sentence emotion badges, optional ambience, and a **Companion** panel (Q&A + continue).

**Default voice path:** free local **Coqui XTTS-v2** (same approach as the Havoc EN-dub pipeline). **ElevenLabs** remains as a paid backup (`TTS_BACKEND=elevenlabs`).

**Session changelog:** [`CHANGES.md`](CHANGES.md)  
**Design:** [`docs/superpowers/specs/2026-08-05-storyteller-pipeline-design.md`](docs/superpowers/specs/2026-08-05-storyteller-pipeline-design.md)  
**Companion addendum:** [`docs/2026-08-05-storyteller-companion-mode-addendum.md`](docs/2026-08-05-storyteller-companion-mode-addendum.md)

## Prerequisites

- Python 3.10+ (3.12 recommended for `coqui-tts`)
- ffmpeg on PATH — must be a **shared** build (ships `avcodec-*.dll` etc.), FFmpeg version 4-8.
  The default winget package (`Gyan.FFmpeg`) is statically linked with no DLLs and, once it
  reaches v9+, is also newer than `torchcodec` (used internally by `torchaudio`/XTTS) supports —
  either gap makes XTTS audio loading fail with `Could not load libtorchcodec`, even though plain
  ffmpeg.exe-based decoding (pydub) works fine. On Windows, install a shared build instead, e.g.
  `winget install --id BtbN.FFmpeg.GPL.Shared.8.1`. `pipeline/config.py`'s `ensure_ffmpeg_on_path()`
  auto-discovers both that shared package and `Gyan.FFmpeg`, patching `PATH` at import time so a
  stale shell/IDE environment doesn't need either already resolvable.
- GPU recommended for XTTS (CUDA). CPU works but is slow.
- API keys for story generation (`STORY_PROVIDER`: openai / claude / gemini / grok)
- **OpenAI API key** also used for Whisper (Companion voice questions) when using openai/grok-compatible routing
- Optional: Freesound (ambience). ElevenLabs only if using the paid backup.
- Mandarin XTTS requires `pypinyin` (in `requirements.txt`)

## Setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill keys; ELEVENLABS_API_KEY only needed for TTS_BACKEND=elevenlabs
export COQUI_TOS_AGREED=1
```

Built-in narrator refs ship in `assets/voices/{warm,bright,gentle}.wav` (regenerate via `scripts/generate_builtin_voices.py`).

## Running

```bash
# Default: free XTTS clone (EN + ZH)
streamlit run app.py

# Paid backup
TTS_BACKEND=elevenlabs streamlit run app.py
```

Pick theme / **language (EN / 中文)**, story source, then a voice:

- **Built-in voice** — Warm / Bright / Gentle samples in `assets/voices/`
- **My voice → Upload** — WAV/MP3 file
- **My voice → Record** — custom browser mic panel (`mic_component/`): **Check level** (meter), **Record/Stop**, preview, then Generate

Then **Generate**. After narration:

- Scroll the **Story & Companion — in order** feed (chapters, Q&A, continues top → bottom)
- **Continue story** or **Ask** (voice or text) at the bottom — new beats append below
- Each of those shows a themed, staged progress status (with a ticking elapsed clock) while it works — these can take 30s-2min on CPU-only XTTS

### Browser / mic tips

- Prefer **Edge** for headphone playback; Brave/Chrome may need **Download** on quiet mono clips.
- In the mic dropdown, pick **External Mic / Realtek** — not Steam Streaming Microphone.
- **Check level** is a meter only, not playback. After Record, use the player below the mic or **Send question**.

## Pipeline (current)

1. Vision (or embedded PDF text) → story sentences (EN or 中文)
   - **Text PDF:** if language picker ≠ PDF script, **translate** (EN↔中文) via LLM, then TTS
2. **Theatre adaptation** → speakers, stage directions, pitch/rate/volume (1–5)
3. **XTTS** zero-shot clone → per-sentence WAV + DSP prosody  
   *(or ElevenLabs one-shot + timestamp slice if `TTS_BACKEND=elevenlabs`)*
4. Optional Freesound ambience by emotion
5. **Companion** — retrieve heard passages + session Q&A → reasoner API; optional **Continue story** beat

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
