# StoryTeller Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit app that turns a ~4-page picture-book PDF and a voice sample into a single narrated MP3 — story generated from the images, narrated in the user's own cloned voice, with optional background ambience — in English or Mandarin.

**Architecture:** A `pipeline/` package of small, independently-testable modules (PDF ingestion, story generation, accent detection, voice cloning, narration synthesis, SFX fetch/cache, audio mixing), wired together by a pure-Python `run_pipeline()` orchestrator that parallelizes independent API calls via `ThreadPoolExecutor`. `app.py` is a thin Streamlit UI layer that constructs the real provider clients and calls the orchestrator.

**Tech Stack:** Python 3.10+, Streamlit, PyMuPDF (PDF rasterization), Pillow, pydub + ffmpeg (audio), OpenAI SDK (vision story-gen + audio accent detection), Anthropic SDK (alternate vision story-gen), ElevenLabs SDK (voice cloning + TTS), `requests` (Freesound), pytest.

## Global Constraints

- Python 3.10+ (uses `typing.Optional`, no walrus-required syntax beyond that).
- **ffmpeg must be installed and on PATH** — pydub shells out to it for anything beyond raw WAV (all our narration/SFX audio is MP3). Windows: `winget install ffmpeg` or `choco install ffmpeg`; macOS: `brew install ffmpeg`; Linux: `apt install ffmpeg`.
- Hume AI is **not used anywhere** in this codebase — confirmed to have no public API for cloning a voice from an uploaded sample (see `docs/superpowers/specs/2026-08-05-storyteller-pipeline-design.md`, Section 2). Do not add Hume SDK calls.
- ElevenLabs is the sole narration backend for both English and Mandarin. Model id: `eleven_v3`.
- SFX attenuation is fixed at **-18dB** relative to the narration.
- PDF page bounds: 1–10 pages. Voice sample duration bounds: 60–300 seconds (1–5 min, ElevenLabs Instant Voice Cloning's recommended range).
- All API keys are read via `pipeline.config.get_secret(name)` — never hardcoded, never read directly from `st.secrets` or `os.environ` outside that function.
- No multi-user/concurrency infrastructure, no auth, no job queue — local single-user POC only.

---

### Task 1: Project scaffolding & config

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `.streamlit/secrets.toml.example`
- Create: `pipeline/__init__.py`
- Create: `pipeline/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `pipeline.config.get_secret(name: str) -> str`, `pipeline.config.ConfigError`, `pipeline.config.STORY_PROVIDER: str`

- [ ] **Step 1: Create scaffolding files**

`requirements.txt`:
```
streamlit>=1.38
PyMuPDF>=1.24
Pillow>=10.0
pydub>=0.25
openai>=1.50
anthropic>=0.34
elevenlabs>=1.9
requests>=2.31
pytest>=7.4
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
venv/
.streamlit/secrets.toml
assets/sfx_cache/
.pytest_cache/
```

`pytest.ini`:
```ini
[pytest]
pythonpath = .
```

`.streamlit/secrets.toml.example`:
```toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
ELEVENLABS_API_KEY = "..."
FREESOUND_API_KEY = "..."
```

`pipeline/__init__.py`: empty file.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config.py
import pytest

from pipeline.config import ConfigError, get_secret


def test_get_secret_reads_from_environment(monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "abc123")
    assert get_secret("MY_TEST_KEY") == "abc123"


def test_get_secret_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    with pytest.raises(ConfigError, match="MISSING_TEST_KEY"):
        get_secret("MISSING_TEST_KEY")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.config'`

- [ ] **Step 4: Write the implementation**

```python
# pipeline/config.py
import os

try:
    import streamlit as st
except ImportError:
    st = None


class ConfigError(Exception):
    """Raised when a required secret or config value is missing."""


STORY_PROVIDER = os.environ.get("STORY_PROVIDER", "openai")


def get_secret(name: str) -> str:
    if st is not None:
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            pass
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"Missing required secret: {name}")
    return value
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore pytest.ini .streamlit/secrets.toml.example pipeline/__init__.py pipeline/config.py tests/test_config.py
git commit -m "Add project scaffolding and secret-loading config"
```

---

### Task 2: Shared errors + audio duration validation

**Files:**
- Create: `pipeline/errors.py`
- Create: `pipeline/audio_utils.py`
- Test: `tests/test_audio_utils.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `pipeline.errors.PipelineError`, `pipeline.errors.ValidationError`; `pipeline.audio_utils.get_duration_seconds(audio_bytes: bytes) -> float`, `pipeline.audio_utils.validate_duration(audio_bytes: bytes, min_seconds: float, max_seconds: float) -> float` (raises `ValidationError`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audio_utils.py
import io

import pytest
from pydub import AudioSegment

from pipeline.audio_utils import get_duration_seconds, validate_duration
from pipeline.errors import ValidationError


def _silence_wav_bytes(duration_ms: int) -> bytes:
    segment = AudioSegment.silent(duration=duration_ms, frame_rate=16000)
    buffer = io.BytesIO()
    segment.export(buffer, format="wav")
    return buffer.getvalue()


def test_get_duration_seconds_matches_generated_length():
    audio_bytes = _silence_wav_bytes(2500)
    assert get_duration_seconds(audio_bytes) == pytest.approx(2.5, abs=0.05)


def test_validate_duration_raises_when_too_short():
    audio_bytes = _silence_wav_bytes(1000)
    with pytest.raises(ValidationError, match="at least"):
        validate_duration(audio_bytes, min_seconds=5, max_seconds=300)


def test_validate_duration_raises_when_too_long():
    audio_bytes = _silence_wav_bytes(2000)
    with pytest.raises(ValidationError, match="at most"):
        validate_duration(audio_bytes, min_seconds=0, max_seconds=1)


def test_validate_duration_passes_within_range():
    audio_bytes = _silence_wav_bytes(2000)
    duration = validate_duration(audio_bytes, min_seconds=1, max_seconds=5)
    assert duration == pytest.approx(2.0, abs=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.audio_utils'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/errors.py
class PipelineError(Exception):
    """Base class for pipeline errors meant to be shown to the user."""


class ValidationError(PipelineError):
    """Raised when user-supplied input fails a local validation check."""
```

```python
# pipeline/audio_utils.py
import io

from pydub import AudioSegment

from pipeline.errors import ValidationError


def get_duration_seconds(audio_bytes: bytes) -> float:
    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    return len(segment) / 1000.0


def validate_duration(audio_bytes: bytes, min_seconds: float, max_seconds: float) -> float:
    duration = get_duration_seconds(audio_bytes)
    if duration < min_seconds:
        raise ValidationError(
            f"Voice sample is {duration:.1f}s long; it needs to be at least "
            f"{min_seconds:.0f}s for voice cloning."
        )
    if duration > max_seconds:
        raise ValidationError(
            f"Voice sample is {duration:.1f}s long; it needs to be at most "
            f"{max_seconds:.0f}s for voice cloning."
        )
    return duration
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_utils.py -v`
Expected: PASS (4 passed). If this fails with a pydub/ffmpeg error, confirm ffmpeg is installed and on PATH (see Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add pipeline/errors.py pipeline/audio_utils.py tests/test_audio_utils.py
git commit -m "Add shared pipeline errors and audio duration validation"
```

---

### Task 3: PDF ingestion

**Files:**
- Create: `pipeline/pdf_ingest.py`
- Test: `tests/test_pdf_ingest.py`

**Interfaces:**
- Consumes: `pipeline.errors.ValidationError` (Task 2)
- Produces: `pipeline.pdf_ingest.extract_page_images(pdf_bytes: bytes) -> list[PIL.Image.Image]`, `pipeline.pdf_ingest.downscale(image, max_dimension: int) -> PIL.Image.Image`, `pipeline.pdf_ingest.MAX_PAGES: int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pdf_ingest.py
import fitz
import pytest
from PIL import Image

from pipeline.errors import ValidationError
from pipeline.pdf_ingest import MAX_PAGES, downscale, extract_page_images


def _make_pdf_bytes(num_pages: int) -> bytes:
    doc = fitz.open()
    for _ in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), "test page")
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_page_images_returns_one_image_per_page():
    images = extract_page_images(_make_pdf_bytes(4))
    assert len(images) == 4
    assert all(isinstance(image, Image.Image) for image in images)


def test_extract_page_images_rejects_too_many_pages():
    with pytest.raises(ValidationError, match="pages"):
        extract_page_images(_make_pdf_bytes(MAX_PAGES + 1))


def test_extract_page_images_rejects_empty_pdf():
    doc = fitz.open()
    doc.new_page()
    doc.delete_page(0)
    pdf_bytes = doc.tobytes()
    doc.close()
    with pytest.raises(ValidationError, match="no pages"):
        extract_page_images(pdf_bytes)


def test_downscale_shrinks_large_image_to_max_dimension():
    large_image = Image.new("RGB", (2000, 1000), color="white")
    result = downscale(large_image, max_dimension=500)
    assert max(result.size) == 500


def test_downscale_leaves_small_image_unchanged():
    small_image = Image.new("RGB", (200, 100), color="white")
    result = downscale(small_image, max_dimension=500)
    assert result.size == (200, 100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pdf_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.pdf_ingest'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/pdf_ingest.py
import fitz
from PIL import Image

from pipeline.errors import ValidationError

MIN_PAGES = 1
MAX_PAGES = 10
RASTER_DPI = 150
MAX_DIMENSION = 1024


def extract_page_images(pdf_bytes: bytes) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = doc.page_count
        if page_count < MIN_PAGES:
            raise ValidationError("The PDF has no pages.")
        if page_count > MAX_PAGES:
            raise ValidationError(
                f"The PDF has {page_count} pages; please upload {MAX_PAGES} or fewer."
            )
        zoom = RASTER_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        images = []
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(downscale(image))
        return images
    finally:
        doc.close()


def downscale(image, max_dimension: int = MAX_DIMENSION):
    width, height = image.size
    largest = max(width, height)
    if largest <= max_dimension:
        return image
    scale = max_dimension / largest
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pdf_ingest.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/pdf_ingest.py tests/test_pdf_ingest.py
git commit -m "Add PDF page rasterization and downscaling"
```

---

### Task 4: Story generation (OpenAI + Claude, swappable)

**Files:**
- Create: `pipeline/story_gen.py`
- Test: `tests/test_story_gen.py`

**Interfaces:**
- Consumes: `list[PIL.Image.Image]` (Task 3's return type, used only as a type — no import needed)
- Produces: `pipeline.story_gen.StoryResult(story_text: str, sfx_mood: str, tts_style_description: str)`, `pipeline.story_gen.OpenAIStoryGenerator(client, model="gpt-4o")` with `.generate(images, language) -> StoryResult`, `pipeline.story_gen.ClaudeStoryGenerator(client, model="claude-sonnet-5")` with `.generate(images, language) -> StoryResult`, `pipeline.story_gen.create_story_generator(provider, openai_client=None, anthropic_client=None) -> StoryGenerator`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_story_gen.py
import json
from unittest.mock import MagicMock

import pytest
from PIL import Image

from pipeline.story_gen import (
    ClaudeStoryGenerator,
    OpenAIStoryGenerator,
    create_story_generator,
)


def _sample_images(count=2):
    return [Image.new("RGB", (10, 10), color="white") for _ in range(count)]


def _payload():
    return {
        "story": "Once upon a time...",
        "sfx_mood": "gentle rain",
        "style_description": "warm, slow, soothing",
    }


def test_openai_story_generator_parses_response_and_sends_all_images():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps(_payload())))
    ]
    generator = OpenAIStoryGenerator(fake_client)

    result = generator.generate(_sample_images(3), "English")

    assert result.story_text == "Once upon a time..."
    assert result.sfx_mood == "gentle rain"
    assert result.tts_style_description == "warm, slow, soothing"
    _, kwargs = fake_client.chat.completions.create.call_args
    content = kwargs["messages"][0]["content"]
    image_blocks = [block for block in content if block["type"] == "image_url"]
    assert len(image_blocks) == 3


def test_claude_story_generator_parses_response_and_sends_all_images():
    fake_client = MagicMock()
    fake_client.messages.create.return_value.content = [
        MagicMock(text=json.dumps(_payload()))
    ]
    generator = ClaudeStoryGenerator(fake_client)

    result = generator.generate(_sample_images(2), "Mandarin")

    assert result.story_text == "Once upon a time..."
    _, kwargs = fake_client.messages.create.call_args
    content = kwargs["messages"][0]["content"]
    image_blocks = [block for block in content if block["type"] == "image"]
    assert len(image_blocks) == 2


def test_create_story_generator_openai():
    generator = create_story_generator("openai", openai_client=MagicMock())
    assert isinstance(generator, OpenAIStoryGenerator)


def test_create_story_generator_claude():
    generator = create_story_generator("claude", anthropic_client=MagicMock())
    assert isinstance(generator, ClaudeStoryGenerator)


def test_create_story_generator_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown story provider"):
        create_story_generator("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_story_gen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.story_gen'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/story_gen.py
import base64
import io
import json

_LANGUAGE_NAMES = {"English": "English", "Mandarin": "Mandarin Chinese"}


class StoryResult:
    def __init__(self, story_text: str, sfx_mood: str, tts_style_description: str):
        self.story_text = story_text
        self.sfx_mood = sfx_mood
        self.tts_style_description = tts_style_description


def _build_prompt(language: str) -> str:
    language_name = _LANGUAGE_NAMES[language]
    return (
        f"Look at these sequential storybook images. Write a short bedtime-appropriate "
        f"story (under 200 words) in {language_name} that follows what happens across "
        f"the images. Add inline delivery tags like [whispers], [gasps], [laughs] where "
        f"the narration should shift tone. Also suggest one background ambience mood "
        f"keyword for the whole story (e.g. 'gentle rain', 'cozy fireplace', 'forest wind') "
        f"and a short natural-language description of how the narrator should deliver the "
        f"story overall (pace, warmth, energy). "
        f"Respond as strict JSON with exactly these keys: "
        f'{{"story": "...", "sfx_mood": "...", "style_description": "..."}}'
    )


def _image_to_jpeg_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


class OpenAIStoryGenerator:
    def __init__(self, client, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    def generate(self, images, language: str) -> StoryResult:
        content = [{"type": "text", "text": _build_prompt(language)}]
        for image in images:
            encoded = base64.b64encode(_image_to_jpeg_bytes(image)).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_tokens=700,
        )
        payload = json.loads(response.choices[0].message.content)
        return StoryResult(
            story_text=payload["story"],
            sfx_mood=payload["sfx_mood"],
            tts_style_description=payload["style_description"],
        )


class ClaudeStoryGenerator:
    def __init__(self, client, model: str = "claude-sonnet-5"):
        self._client = client
        self._model = model

    def generate(self, images, language: str) -> StoryResult:
        prompt = _build_prompt(language) + " Respond with ONLY the JSON object, no other text."
        content = [{"type": "text", "text": prompt}]
        for image in images:
            encoded = base64.b64encode(_image_to_jpeg_bytes(image)).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded},
            })
        response = self._client.messages.create(
            model=self._model,
            max_tokens=700,
            messages=[{"role": "user", "content": content}],
        )
        payload = json.loads(response.content[0].text)
        return StoryResult(
            story_text=payload["story"],
            sfx_mood=payload["sfx_mood"],
            tts_style_description=payload["style_description"],
        )


def create_story_generator(provider: str, openai_client=None, anthropic_client=None):
    if provider == "openai":
        if openai_client is None:
            raise ValueError("openai_client is required for the 'openai' story provider")
        return OpenAIStoryGenerator(openai_client)
    if provider == "claude":
        if anthropic_client is None:
            raise ValueError("anthropic_client is required for the 'claude' story provider")
        return ClaudeStoryGenerator(anthropic_client)
    raise ValueError(f"Unknown story provider: {provider!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_story_gen.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/story_gen.py tests/test_story_gen.py
git commit -m "Add swappable vision-LLM story generation (OpenAI + Claude)"
```

---

### Task 5: Accent detection

**Files:**
- Create: `pipeline/accent.py`
- Test: `tests/test_accent.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `pipeline.accent.AccentResult(accent_label: str, detected_language: str)`, `pipeline.accent.OpenAIAudioAccentDetector(client, model="gpt-4o-audio-preview")` with `.detect(audio_bytes, audio_format="wav") -> AccentResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_accent.py
import base64
import json
from unittest.mock import MagicMock

from pipeline.accent import OpenAIAudioAccentDetector


def test_detect_parses_response_and_encodes_audio():
    fake_client = MagicMock()
    payload = {"accent_label": "Singaporean English", "detected_language": "English"}
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps(payload)))
    ]
    detector = OpenAIAudioAccentDetector(fake_client)

    result = detector.detect(b"fake-audio-bytes", audio_format="wav")

    assert result.accent_label == "Singaporean English"
    assert result.detected_language == "English"
    _, kwargs = fake_client.chat.completions.create.call_args
    content = kwargs["messages"][0]["content"]
    audio_block = next(block for block in content if block["type"] == "input_audio")
    assert audio_block["input_audio"]["format"] == "wav"
    assert audio_block["input_audio"]["data"] == base64.b64encode(b"fake-audio-bytes").decode("utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_accent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.accent'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/accent.py
import base64
import json


class AccentResult:
    def __init__(self, accent_label: str, detected_language: str):
        self.accent_label = accent_label
        self.detected_language = detected_language


_PROMPT = (
    "Listen to this voice sample. Identify: "
    "(1) the speaker's regional accent or language variety in a short phrase "
    "(e.g. 'Singaporean English', 'American English', 'Beijing Mandarin'), and "
    "(2) the primary language being spoken, as exactly one of: 'English' or 'Mandarin'. "
    "Respond as strict JSON with exactly these keys: "
    '{"accent_label": "...", "detected_language": "..."}'
)


class OpenAIAudioAccentDetector:
    def __init__(self, client, model: str = "gpt-4o-audio-preview"):
        self._client = client
        self._model = model

    def detect(self, audio_bytes: bytes, audio_format: str = "wav") -> AccentResult:
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        response = self._client.chat.completions.create(
            model=self._model,
            modalities=["text"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}},
                ],
            }],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content)
        return AccentResult(
            accent_label=payload["accent_label"],
            detected_language=payload["detected_language"],
        )
```

**Note for Task 12's manual smoke test:** `response_format={"type": "json_object"}` combined with audio input on `gpt-4o-audio-preview` is untested against the real API in this plan (only mocked here). If the real API rejects that combination, remove `response_format` from the call above, keep the "Respond as strict JSON..." instruction in the prompt, and change the parse to extract the first `{...}` substring from the response text before `json.loads`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_accent.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/accent.py tests/test_accent.py
git commit -m "Add audio-based accent detection"
```

---

### Task 6: Voice cloning (ElevenLabs)

**Files:**
- Create: `pipeline/voice_clone.py`
- Test: `tests/test_voice_clone.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `pipeline.voice_clone.ElevenLabsVoiceCloner(client)` with `.clone(audio_bytes: bytes, name: str) -> str` (voice_id)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_voice_clone.py
from unittest.mock import MagicMock

from pipeline.voice_clone import ElevenLabsVoiceCloner


def test_clone_returns_voice_id_and_sends_audio_file():
    fake_client = MagicMock()
    fake_client.voices.ivc.create.return_value.voice_id = "voice-123"
    cloner = ElevenLabsVoiceCloner(fake_client)

    voice_id = cloner.clone(b"sample-audio-bytes", "My Story Voice")

    assert voice_id == "voice-123"
    _, kwargs = fake_client.voices.ivc.create.call_args
    assert kwargs["name"] == "My Story Voice"
    filename, content, content_type = kwargs["files"][0]
    assert filename == "voice_sample.wav"
    assert content == b"sample-audio-bytes"
    assert content_type == "audio/wav"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_voice_clone.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.voice_clone'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/voice_clone.py
class ElevenLabsVoiceCloner:
    def __init__(self, client):
        self._client = client

    def clone(self, audio_bytes: bytes, name: str) -> str:
        response = self._client.voices.ivc.create(
            name=name,
            files=[("voice_sample.wav", audio_bytes, "audio/wav")],
        )
        return response.voice_id
```

This uses ElevenLabs' Instant Voice Cloning endpoint, confirmed against the
`elevenlabs/elevenlabs-python` SDK source: `client.voices.ivc.create(name, files, ...)`
returns an `AddVoiceIvcResponseModel` with a `.voice_id` field.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_voice_clone.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/voice_clone.py tests/test_voice_clone.py
git commit -m "Add ElevenLabs instant voice cloning wrapper"
```

---

### Task 7: Narration synthesis (ElevenLabs)

**Files:**
- Create: `pipeline/tts.py`
- Test: `tests/test_tts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (takes plain strings/voice_id).
- Produces: `pipeline.tts.ElevenLabsNarrationSynthesizer(client, model_id="eleven_v3")` with `.synthesize(text: str, voice_id: str, style_description: str, language: str) -> bytes`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tts.py
from unittest.mock import MagicMock

from pipeline.tts import ElevenLabsNarrationSynthesizer


def test_synthesize_joins_audio_chunks_and_passes_expected_params():
    fake_client = MagicMock()
    fake_client.text_to_speech.convert.return_value = iter([b"chunk1", b"chunk2"])
    synthesizer = ElevenLabsNarrationSynthesizer(fake_client)

    audio_bytes = synthesizer.synthesize(
        text="Once upon a time...",
        voice_id="voice-123",
        style_description="warm and slow",
        language="Mandarin",
    )

    assert audio_bytes == b"chunk1chunk2"
    args, kwargs = fake_client.text_to_speech.convert.call_args
    assert args[0] == "voice-123"
    assert "warm and slow" in kwargs["text"]
    assert "Once upon a time..." in kwargs["text"]
    assert kwargs["model_id"] == "eleven_v3"
    assert kwargs["language_code"] == "zh"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.tts'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/tts.py
_LANGUAGE_CODES = {"English": "en", "Mandarin": "zh"}


class ElevenLabsNarrationSynthesizer:
    def __init__(self, client, model_id: str = "eleven_v3"):
        self._client = client
        self._model_id = model_id

    def synthesize(self, text: str, voice_id: str, style_description: str, language: str) -> bytes:
        full_text = f"[{style_description}]\n{text}" if style_description else text
        chunks = self._client.text_to_speech.convert(
            voice_id,
            text=full_text,
            model_id=self._model_id,
            output_format="mp3_44100_128",
            language_code=_LANGUAGE_CODES.get(language),
        )
        return b"".join(chunks)
```

This uses ElevenLabs' `client.text_to_speech.convert(voice_id, text=, model_id=,
output_format=, language_code=)`, confirmed against the SDK source, which returns an
iterator of audio byte chunks (joined here into one `bytes` object). `eleven_v3`
natively interprets the `[whisper]`/`[gasp]`-style tags embedded in `text` by
`story_gen.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/tts.py tests/test_tts.py
git commit -m "Add ElevenLabs narration synthesis wrapper"
```

---

### Task 8: SFX fetch + cache (Freesound)

**Files:**
- Create: `pipeline/sfx.py`
- Test: `tests/test_sfx.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `pipeline.sfx.fetch_ambience_clip(mood: str, api_key: str, cache_dir: str) -> typing.Optional[bytes]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sfx.py
import os
from unittest.mock import MagicMock, patch

from pipeline.sfx import fetch_ambience_clip


def _search_response(has_results=True):
    response = MagicMock()
    response.raise_for_status.return_value = None
    if has_results:
        response.json.return_value = {
            "results": [{
                "id": 1,
                "name": "rain loop",
                "previews": {"preview-hq-mp3": "https://freesound.org/preview/1.mp3"},
                "license": "CC0",
                "duration": 30.0,
            }]
        }
    else:
        response.json.return_value = {"results": []}
    return response


def _audio_response(content=b"fake-mp3-bytes"):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.content = content
    return response


@patch("pipeline.sfx.requests.get")
def test_fetch_ambience_clip_downloads_and_caches(mock_get, tmp_path):
    mock_get.side_effect = [_search_response(), _audio_response()]
    cache_dir = str(tmp_path)

    audio_bytes = fetch_ambience_clip("gentle rain", api_key="fake-key", cache_dir=cache_dir)

    assert audio_bytes == b"fake-mp3-bytes"
    assert mock_get.call_count == 2
    assert len(os.listdir(cache_dir)) == 1


@patch("pipeline.sfx.requests.get")
def test_fetch_ambience_clip_uses_cache_on_second_call(mock_get, tmp_path):
    mock_get.side_effect = [_search_response(), _audio_response()]
    cache_dir = str(tmp_path)

    first = fetch_ambience_clip("gentle rain", api_key="fake-key", cache_dir=cache_dir)
    second = fetch_ambience_clip("gentle rain", api_key="fake-key", cache_dir=cache_dir)

    assert first == second == b"fake-mp3-bytes"
    assert mock_get.call_count == 2  # second call served entirely from cache


@patch("pipeline.sfx.requests.get")
def test_fetch_ambience_clip_returns_none_when_no_results(mock_get, tmp_path):
    mock_get.side_effect = [_search_response(has_results=False)]

    result = fetch_ambience_clip("nonexistent mood", api_key="fake-key", cache_dir=str(tmp_path))

    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sfx.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.sfx'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/sfx.py
import hashlib
import os
import typing

import requests

FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/"


def fetch_ambience_clip(mood: str, api_key: str, cache_dir: str) -> typing.Optional[bytes]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _cache_path(cache_dir, mood)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    search_response = requests.get(
        FREESOUND_SEARCH_URL,
        headers={"Authorization": f"Token {api_key}"},
        params={
            "query": mood,
            "fields": "id,name,previews,license,duration",
            "filter": "duration:[5.0 TO 120.0]",
            "sort": "rating_desc",
            "page_size": 1,
        },
        timeout=15,
    )
    search_response.raise_for_status()
    results = search_response.json().get("results", [])
    if not results:
        return None

    preview_url = results[0]["previews"]["preview-hq-mp3"]
    audio_response = requests.get(
        preview_url,
        headers={"Authorization": f"Token {api_key}"},
        timeout=30,
    )
    audio_response.raise_for_status()
    audio_bytes = audio_response.content

    with open(cache_path, "wb") as f:
        f.write(audio_bytes)
    return audio_bytes


def _cache_path(cache_dir: str, mood: str) -> str:
    key = hashlib.sha1(mood.strip().lower().encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.mp3")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sfx.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/sfx.py tests/test_sfx.py
git commit -m "Add Freesound ambience search, download, and disk cache"
```

---

### Task 9: Audio mixing

**Files:**
- Create: `pipeline/mixer.py`
- Test: `tests/test_mixer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `pipeline.mixer.mix_narration_with_ambience(narration_bytes: bytes, ambience_bytes: typing.Optional[bytes]) -> bytes`, `pipeline.mixer.SFX_ATTENUATION_DB: int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mixer.py
import io

from pydub import AudioSegment

from pipeline.mixer import _loop_to_length, mix_narration_with_ambience


def _to_mp3_bytes(segment: AudioSegment) -> bytes:
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()


def test_loop_to_length_loops_short_segment_to_target():
    short_segment = AudioSegment.silent(duration=300)
    result = _loop_to_length(short_segment, target_length_ms=1000)
    assert len(result) == 1000


def test_loop_to_length_trims_long_segment_to_target():
    long_segment = AudioSegment.silent(duration=2000)
    result = _loop_to_length(long_segment, target_length_ms=500)
    assert len(result) == 500


def test_mix_narration_with_ambience_matches_narration_length():
    narration_bytes = _to_mp3_bytes(AudioSegment.silent(duration=3000))
    ambience_bytes = _to_mp3_bytes(AudioSegment.silent(duration=800))

    mixed_bytes = mix_narration_with_ambience(narration_bytes, ambience_bytes)
    mixed = AudioSegment.from_file(io.BytesIO(mixed_bytes))

    assert abs(len(mixed) - 3000) < 100


def test_mix_narration_with_ambience_returns_dry_narration_when_no_ambience():
    narration_bytes = _to_mp3_bytes(AudioSegment.silent(duration=1500))

    mixed_bytes = mix_narration_with_ambience(narration_bytes, None)
    mixed = AudioSegment.from_file(io.BytesIO(mixed_bytes))

    assert abs(len(mixed) - 1500) < 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mixer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.mixer'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/mixer.py
import io
import typing

from pydub import AudioSegment

SFX_ATTENUATION_DB = -18


def mix_narration_with_ambience(
    narration_bytes: bytes, ambience_bytes: typing.Optional[bytes]
) -> bytes:
    narration = AudioSegment.from_file(io.BytesIO(narration_bytes))
    if not ambience_bytes:
        return _export_mp3(narration)

    ambience = AudioSegment.from_file(io.BytesIO(ambience_bytes))
    ambience = _loop_to_length(ambience, len(narration))
    ambience = ambience + SFX_ATTENUATION_DB
    mixed = narration.overlay(ambience)
    return _export_mp3(mixed)


def _loop_to_length(segment: AudioSegment, target_length_ms: int) -> AudioSegment:
    if len(segment) == 0:
        return segment
    if len(segment) < target_length_ms:
        loops_required = (target_length_ms // len(segment)) + 1
        segment = segment * loops_required
    return segment[:target_length_ms]


def _export_mp3(segment: AudioSegment) -> bytes:
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mixer.py -v`
Expected: PASS (4 passed). Requires ffmpeg on PATH (Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add pipeline/mixer.py tests/test_mixer.py
git commit -m "Add narration/ambience mixing with loop-to-length and -18dB attenuation"
```

---

### Task 10: Pipeline orchestrator

**Files:**
- Create: `pipeline/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `pipeline.pdf_ingest.extract_page_images` (Task 3), `pipeline.story_gen.StoryResult` (Task 4, shape only), `pipeline.accent.AccentResult` (Task 5, shape only), `pipeline.sfx.fetch_ambience_clip` (Task 8), `pipeline.mixer.mix_narration_with_ambience` (Task 9). Takes already-constructed `story_generator`/`accent_detector`/`voice_cloner`/`narration_synthesizer` objects matching Tasks 4/5/6/7's interfaces as keyword arguments — it does not construct clients itself.
- Produces: `pipeline.orchestrator.PipelineResult(story_text: str, sfx_mood: str, final_audio_bytes: bytes, used_sfx: bool)`, `pipeline.orchestrator.run_pipeline(pdf_bytes, voice_bytes, language, enable_sfx, *, story_generator, accent_detector, voice_cloner, narration_synthesizer, freesound_api_key, sfx_cache_dir) -> PipelineResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestrator.py
import io
from unittest.mock import patch

import fitz
from pydub import AudioSegment

from pipeline.accent import AccentResult
from pipeline.orchestrator import run_pipeline
from pipeline.story_gen import StoryResult


def _make_pdf_bytes(num_pages=2) -> bytes:
    doc = fitz.open()
    for _ in range(num_pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def _silence_mp3_bytes(duration_ms):
    segment = AudioSegment.silent(duration=duration_ms)
    buffer = io.BytesIO()
    segment.export(buffer, format="mp3")
    return buffer.getvalue()


class FakeStoryGenerator:
    def generate(self, images, language):
        return StoryResult(
            story_text="Once upon a time",
            sfx_mood="gentle rain",
            tts_style_description="warm",
        )


class FakeAccentDetector:
    def __init__(self, detected_language="English"):
        self._detected_language = detected_language

    def detect(self, audio_bytes):
        return AccentResult(accent_label="American English", detected_language=self._detected_language)


class FakeVoiceCloner:
    def clone(self, audio_bytes, name):
        return "voice-123"


class FakeNarrationSynthesizer:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice_id, style_description, language):
        self.calls.append((text, voice_id, style_description, language))
        return _silence_mp3_bytes(1000)


@patch("pipeline.orchestrator.fetch_ambience_clip")
def test_run_pipeline_without_sfx(mock_fetch_ambience, tmp_path):
    narration_synth = FakeNarrationSynthesizer()

    result = run_pipeline(
        pdf_bytes=_make_pdf_bytes(),
        voice_bytes=b"fake-voice-sample",
        language="English",
        enable_sfx=False,
        story_generator=FakeStoryGenerator(),
        accent_detector=FakeAccentDetector(detected_language="English"),
        voice_cloner=FakeVoiceCloner(),
        narration_synthesizer=narration_synth,
        freesound_api_key="fake-key",
        sfx_cache_dir=str(tmp_path),
    )

    assert result.story_text == "Once upon a time"
    assert result.used_sfx is False
    mock_fetch_ambience.assert_not_called()
    assert narration_synth.calls[0][2] == "warm"


@patch("pipeline.orchestrator.fetch_ambience_clip")
def test_run_pipeline_with_sfx_fetches_ambience(mock_fetch_ambience, tmp_path):
    mock_fetch_ambience.return_value = _silence_mp3_bytes(500)
    narration_synth = FakeNarrationSynthesizer()

    result = run_pipeline(
        pdf_bytes=_make_pdf_bytes(),
        voice_bytes=b"fake-voice-sample",
        language="English",
        enable_sfx=True,
        story_generator=FakeStoryGenerator(),
        accent_detector=FakeAccentDetector(detected_language="English"),
        voice_cloner=FakeVoiceCloner(),
        narration_synthesizer=narration_synth,
        freesound_api_key="fake-key",
        sfx_cache_dir=str(tmp_path),
    )

    assert result.used_sfx is True
    mock_fetch_ambience.assert_called_once_with("gentle rain", "fake-key", str(tmp_path))


def test_run_pipeline_appends_accent_hint_when_cross_lingual(tmp_path):
    narration_synth = FakeNarrationSynthesizer()

    run_pipeline(
        pdf_bytes=_make_pdf_bytes(),
        voice_bytes=b"fake-voice-sample",
        language="Mandarin",
        enable_sfx=False,
        story_generator=FakeStoryGenerator(),
        accent_detector=FakeAccentDetector(detected_language="English"),
        voice_cloner=FakeVoiceCloner(),
        narration_synthesizer=narration_synth,
        freesound_api_key="fake-key",
        sfx_cache_dir=str(tmp_path),
    )

    style_description_used = narration_synth.calls[0][2]
    assert "American English accent flavor" in style_description_used
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.orchestrator'`

- [ ] **Step 3: Write the implementation**

```python
# pipeline/orchestrator.py
import concurrent.futures

from pipeline.mixer import mix_narration_with_ambience
from pipeline.pdf_ingest import extract_page_images
from pipeline.sfx import fetch_ambience_clip


class PipelineResult:
    def __init__(self, story_text: str, sfx_mood: str, final_audio_bytes: bytes, used_sfx: bool):
        self.story_text = story_text
        self.sfx_mood = sfx_mood
        self.final_audio_bytes = final_audio_bytes
        self.used_sfx = used_sfx


def run_pipeline(
    pdf_bytes: bytes,
    voice_bytes: bytes,
    language: str,
    enable_sfx: bool,
    *,
    story_generator,
    accent_detector,
    voice_cloner,
    narration_synthesizer,
    freesound_api_key: str,
    sfx_cache_dir: str,
) -> PipelineResult:
    images = extract_page_images(pdf_bytes)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        story_future = executor.submit(story_generator.generate, images, language)
        accent_future = executor.submit(accent_detector.detect, voice_bytes)
        clone_future = executor.submit(voice_cloner.clone, voice_bytes, "StoryTeller Voice")

        story_result = story_future.result()
        accent_result = accent_future.result()
        voice_id = clone_future.result()

    style_description = story_result.tts_style_description
    if accent_result.detected_language != language:
        style_description = (
            f"{style_description} Speak with a {accent_result.accent_label} accent flavor."
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        narration_future = executor.submit(
            narration_synthesizer.synthesize,
            story_result.story_text,
            voice_id,
            style_description,
            language,
        )
        ambience_future = None
        if enable_sfx:
            ambience_future = executor.submit(
                fetch_ambience_clip, story_result.sfx_mood, freesound_api_key, sfx_cache_dir
            )

        narration_bytes = narration_future.result()
        ambience_bytes = ambience_future.result() if ambience_future else None

    final_audio_bytes = mix_narration_with_ambience(narration_bytes, ambience_bytes)

    return PipelineResult(
        story_text=story_result.story_text,
        sfx_mood=story_result.sfx_mood,
        final_audio_bytes=final_audio_bytes,
        used_sfx=bool(ambience_bytes),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pipeline/orchestrator.py tests/test_orchestrator.py
git commit -m "Add pipeline orchestrator with parallel stage execution"
```

---

### Task 11: Streamlit UI (`app.py`)

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `pipeline.config.get_secret`, `pipeline.config.STORY_PROVIDER` (Task 1); `pipeline.audio_utils.validate_duration` (Task 2); `pipeline.story_gen.create_story_generator` (Task 4); `pipeline.accent.OpenAIAudioAccentDetector` (Task 5); `pipeline.voice_clone.ElevenLabsVoiceCloner` (Task 6); `pipeline.tts.ElevenLabsNarrationSynthesizer` (Task 7); `pipeline.orchestrator.run_pipeline` (Task 10); `pipeline.errors.ValidationError` (Task 2).
- Produces: nothing consumed by other tasks — this is the top-level entry point.

There is no automated test for this task, per the design spec's testing approach (Streamlit UI is verified manually) — Step 2 below is a manual verification checklist instead of a pytest run.

- [ ] **Step 1: Write the implementation**

```python
# app.py
import os
import tempfile

import streamlit as st
from anthropic import Anthropic
from elevenlabs import ElevenLabs
from openai import OpenAI

from pipeline.accent import OpenAIAudioAccentDetector
from pipeline.audio_utils import validate_duration
from pipeline.config import STORY_PROVIDER, get_secret
from pipeline.errors import ValidationError
from pipeline.orchestrator import run_pipeline
from pipeline.story_gen import create_story_generator
from pipeline.tts import ElevenLabsNarrationSynthesizer
from pipeline.voice_clone import ElevenLabsVoiceCloner

MIN_VOICE_SECONDS = 60
MAX_VOICE_SECONDS = 300
SFX_CACHE_DIR = os.path.join(tempfile.gettempdir(), "storyteller_sfx_cache")

st.set_page_config(page_title="StoryTeller", page_icon="📖")
st.title("StoryTeller")
st.write("Upload a short picture-book PDF and a voice sample to generate a narrated story.")

pdf_file = st.file_uploader("Picture-book PDF (about 4 pages)", type=["pdf"])
voice_file = st.file_uploader("Voice sample (1-5 minutes)", type=["wav", "mp3", "m4a"])
enable_sfx = st.checkbox("Include background sound effects", value=True)
language = st.selectbox("Output language", ["English", "Mandarin"])

if st.button("Generate story", type="primary", disabled=not (pdf_file and voice_file)):
    pdf_bytes = pdf_file.read()
    voice_bytes = voice_file.read()

    try:
        with st.status("Checking voice sample...", expanded=True) as status:
            validate_duration(voice_bytes, MIN_VOICE_SECONDS, MAX_VOICE_SECONDS)
            status.update(label="Voice sample OK. Generating story and cloning voice...")

            openai_client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
            anthropic_client = (
                Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
                if STORY_PROVIDER == "claude"
                else None
            )
            elevenlabs_client = ElevenLabs(api_key=get_secret("ELEVENLABS_API_KEY"))

            story_generator = create_story_generator(
                STORY_PROVIDER, openai_client=openai_client, anthropic_client=anthropic_client
            )
            accent_detector = OpenAIAudioAccentDetector(openai_client)
            voice_cloner = ElevenLabsVoiceCloner(elevenlabs_client)
            narration_synthesizer = ElevenLabsNarrationSynthesizer(elevenlabs_client)

            result = run_pipeline(
                pdf_bytes=pdf_bytes,
                voice_bytes=voice_bytes,
                language=language,
                enable_sfx=enable_sfx,
                story_generator=story_generator,
                accent_detector=accent_detector,
                voice_cloner=voice_cloner,
                narration_synthesizer=narration_synthesizer,
                freesound_api_key=get_secret("FREESOUND_API_KEY"),
                sfx_cache_dir=SFX_CACHE_DIR,
            )
            status.update(label="Done!", state="complete")

        st.subheader("Story")
        st.write(result.story_text)
        if enable_sfx and not result.used_sfx:
            st.warning("No matching background ambience was found; narration has no SFX.")
        st.audio(result.final_audio_bytes, format="audio/mp3")
        st.download_button(
            "Download narrated story",
            data=result.final_audio_bytes,
            file_name="story.mp3",
            mime="audio/mp3",
        )
    except ValidationError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
```

- [ ] **Step 2: Manual verification**

Run: `streamlit run app.py`

Checklist (needs real API keys in `.streamlit/secrets.toml` — see Task 12):
1. Upload a ~4-page PDF and a 1-2 minute voice sample (wav or mp3), leave SFX on, pick English. Click Generate story. Confirm: per-stage status messages appear, a story renders, an audio player appears, and the narration sounds like the uploaded voice.
2. Repeat with a voice sample under 60 seconds. Confirm a red validation error appears immediately (no API calls fire) instead of a stack trace.
3. Retry a full run with SFX unchecked. Confirm the output has no background ambience.
4. Retry a full run with Mandarin selected. Confirm the story text is in Mandarin and the narration is in Mandarin, still in the cloned voice.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "Add Streamlit UI wiring the pipeline together"
```

---

### Task 12: Setup docs

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing consumed by other tasks.

- [ ] **Step 1: Write the implementation**

```markdown
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
  - OpenAI (vision story generation + audio accent detection)
  - Anthropic (only if `STORY_PROVIDER=claude`)
  - ElevenLabs (voice cloning + narration synthesis)
  - Freesound (background ambience search) — see the Freesound API guide in
    this repo's docs for how to request one and its token-based auth model.

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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Add setup docs and manual smoke-test checklist"
```
