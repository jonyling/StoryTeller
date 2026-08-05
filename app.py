"""
Context-Aware Storyteller — Streamlit app.
Display layer only. Data schema unchanged:
    {page, sentences: [{text, speaker, emotion, pitch, volume, rate, audio_path}]}

Four themes, two languages (EN/ZH), three story sources (PDF/Picture/Camera),
picked in the header. Adding a fifth theme means adding one dict to THEMES
and nothing else — the stylesheet only ever reads T[...].
"""

import base64
import io
import os
import re
import struct
import tempfile
import time

import streamlit as st
from anthropic import Anthropic
from elevenlabs import ElevenLabs
from google import genai
from openai import OpenAI
from PIL import Image
from pypdf import PdfReader

from pipeline.audio_utils import validate_duration
from pipeline.config import STORY_PROVIDER, get_secret
from pipeline.errors import PipelineError
from pipeline.orchestrator import run_pipeline
from pipeline.pdf_ingest import extract_page_images
from pipeline.story_gen import create_story_generator
from pipeline.tts import ElevenLabsNarrationSynthesizer
from pipeline.voice_clone import ElevenLabsVoiceCloner

st.set_page_config(page_title="Context-Aware Storyteller", page_icon="📖",
                   layout="centered", initial_sidebar_state="collapsed")

for k, v in {"theme": "Classic", "lang": "EN", "sfx": False, "story": None,
             "used_fallback": False, "source": "PDF", "illustration": None,
             "voice_mode": "Default", "voice_preset": "warm",
             "idx": 0, "dir": "fwd", "autoplay": False,
             "ambience_by_emotion": {}}.items():
    st.session_state.setdefault(k, v)

# Real generation config. Story language reuses the EN/ZH picker (see the
# THEME/LANGUAGE/AMBIENCE PICKER section) rather than adding a separate control.
MIN_VOICE_SECONDS = 60
MAX_VOICE_SECONDS = 300
SFX_CACHE_DIR = os.path.join(tempfile.gettempdir(), "storyteller_sfx_cache")
STORY_LANGUAGE = {"EN": "English", "ZH": "Mandarin"}

# ---------------------------------------------------------------------------
# Backend — generate_mock_story() now calls the real pipeline. Its signature
# grew three keyword-only params (raw_source_bytes, language, enable_sfx,
# on_progress) to carry what the mock never needed; the original two
# positional params (pages, voice_file) and the {page, sentences:[...]}
# return schema are unchanged, per the "don't touch the seam without telling
# the UI team" agreement — this only extends it.
# ---------------------------------------------------------------------------

MOCK_STORY_PAGES = [
    {
        "page": 1,
        "sentences": [
            {
                "text": "Once upon a time, in a quiet village by the hills, there lived a small dragon named Ember.",
                "speaker": "narrator", "emotion": "neutral",
                "pitch": 3, "volume": 3, "rate": 3,
            },
            {
                "text": "Ember loved to nap in the warm sun, dreaming of faraway skies.",
                "speaker": "narrator", "emotion": "calm",
                "pitch": 2, "volume": 2, "rate": 2,
            },
        ],
    },
    {
        "page": 2,
        "sentences": [
            {
                "text": "One day, a fierce storm rolled over the mountains, and Ember's home shook violently!",
                "speaker": "narrator", "emotion": "angry",
                "pitch": 4, "volume": 5, "rate": 4,
            },
            {
                "text": "\"We have to get out of here, now!\" Ember roared, wings snapping open.",
                "speaker": "ember", "emotion": "angry",
                "pitch": 5, "volume": 5, "rate": 5,
            },
        ],
    },
    {
        "page": 3,
        "sentences": [
            {
                "text": "But when the storm cleared, Ember found a hidden valley glowing with golden light!",
                "speaker": "narrator", "emotion": "excited",
                "pitch": 5, "volume": 4, "rate": 4,
            },
            {
                "text": "\"It's beautiful,\" Ember whispered, eyes wide with wonder.",
                "speaker": "ember", "emotion": "excited",
                "pitch": 4, "volume": 3, "rate": 3,
            },
        ],
    },
    {
        "page": 4,
        "sentences": [
            {
                "text": "But far from home, Ember suddenly felt very small and very alone.",
                "speaker": "narrator", "emotion": "sad",
                "pitch": 2, "volume": 2, "rate": 2,
            },
        ],
    },
]

EMOTION_KEYWORDS = {
    "angry": ["roar", "roared", "furious", "shout", "shouted", "scream", "screamed",
              "rage", "storm", "thunder", "crash", "fierce", "growl", "growled"],
    "excited": ["amazing", "wonderful", "wow", "celebrate", "laughed", "cheer",
                "sparkle", "glow", "glowing", "adventure", "discover", "wonder",
                "excited", "beautiful"],
    "sad": ["cry", "cried", "tears", "alone", "lonely", "lost", "gloomy", "dark",
            "missed", "sorrow", "sad", "small"],
    "calm": ["quiet", "peaceful", "gentle", "softly", "whisper", "rest", "sleep",
             "calm", "warm", "nap"],
}

EMOTION_DSP_DEFAULTS = {
    "angry": {"pitch": 4, "volume": 5, "rate": 4},
    "excited": {"pitch": 5, "volume": 4, "rate": 4},
    "sad": {"pitch": 2, "volume": 2, "rate": 2},
    "calm": {"pitch": 2, "volume": 2, "rate": 2},
    "neutral": {"pitch": 3, "volume": 3, "rate": 3},
}


def tag_emotion(sentence_text: str) -> str:
    lowered = sentence_text.lower()
    scores = {
        emotion: sum(1 for kw in keywords if kw in lowered)
        for emotion, keywords in EMOTION_KEYWORDS.items()
    }
    best_emotion, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_emotion if best_score > 0 else "neutral"


def extract_pages_from_pdf(pdf_file):
    """Extract real text from an uploaded PDF and tag each sentence by
    keyword match. Returns None if the PDF has no extractable text (e.g.
    scanned/image-only pages), so the caller can fall back to mock data."""
    reader = PdfReader(pdf_file)
    pages = []
    for page_num, pdf_page in enumerate(reader.pages, start=1):
        text = (pdf_page.extract_text() or "").strip()
        if not text:
            continue
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = []
        for raw in raw_sentences:
            cleaned = raw.strip()
            if not cleaned:
                continue
            emotion = tag_emotion(cleaned)
            sentences.append({
                "text": cleaned,
                "speaker": "narrator",
                "emotion": emotion,
                **EMOTION_DSP_DEFAULTS[emotion],
            })
        if sentences:
            pages.append({"page": page_num, "sentences": sentences})
    return pages or None


def _read_voice_bytes(voice_file) -> bytes:
    """`voice_file` is either an uploaded file (Own) or a path string to a
    bundled preset (Default) — normalize both to raw bytes."""
    if hasattr(voice_file, "getvalue"):
        return voice_file.getvalue()
    with open(voice_file, "rb") as f:
        return f.read()


def _build_story_provider_client():
    """Construct only the client the active STORY_PROVIDER needs, mirroring
    which secret is actually required."""
    if STORY_PROVIDER == "openai":
        return "openai_client", OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    if STORY_PROVIDER == "claude":
        return "anthropic_client", Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
    if STORY_PROVIDER == "gemini":
        return "gemini_client", genai.Client(api_key=get_secret("GEMINI_API_KEY"))
    if STORY_PROVIDER == "grok":
        return "grok_client", OpenAI(
            api_key=get_secret("XAI_API_KEY"), base_url="https://api.x.ai/v1"
        )
    raise PipelineError(f"Unknown STORY_PROVIDER: {STORY_PROVIDER!r}")


def generate_mock_story(pages, voice_file, *, raw_source_bytes, language, enable_sfx,
                         on_progress=None):
    """Real pipeline call. `pages` is either:
    - a list of already-tagged page dicts, from extract_pages_from_pdf() (a
      PDF that already had real embedded text) — used directly, no vision
      call needed, exactly like the mock's fast path.
    - None (a PDF with no extractable text — the normal picture-book case)
      or a list holding one raw uploaded image (Picture/Camera) — in both
      cases `raw_source_bytes` carries what's needed for real vision
      generation: PDF bytes to rasterize, or a single already-opened image.

    Clones the voice from `voice_file`, runs vision story generation + voice
    cloning + narration synthesis + per-sentence slicing + per-emotion
    ambience fetch, and returns pages in the same {page, sentences: [...]}
    schema, each sentence now carrying real per-sentence audio bytes as
    `audio_path` (Streamlit's st.audio() accepts raw bytes, not just a path).
    """
    if pages and isinstance(pages[0], dict) and "sentences" in pages[0]:
        return pages

    try:
        if isinstance(raw_source_bytes, list):
            images = raw_source_bytes
        else:
            images = extract_page_images(raw_source_bytes)

        voice_bytes = _read_voice_bytes(voice_file)
        validate_duration(voice_bytes, MIN_VOICE_SECONDS, MAX_VOICE_SECONDS)

        client_kwarg_name, story_client = _build_story_provider_client()
        story_generator = create_story_generator(STORY_PROVIDER, **{client_kwarg_name: story_client})
        elevenlabs_client = ElevenLabs(api_key=get_secret("ELEVENLABS_API_KEY"))

        result = run_pipeline(
            images=images,
            voice_bytes=voice_bytes,
            language=language,
            enable_sfx=enable_sfx,
            story_generator=story_generator,
            voice_cloner=ElevenLabsVoiceCloner(elevenlabs_client),
            narration_synthesizer=ElevenLabsNarrationSynthesizer(elevenlabs_client),
            freesound_api_key=get_secret("FREESOUND_API_KEY") if enable_sfx else "",
            sfx_cache_dir=SFX_CACHE_DIR,
            on_progress=on_progress,
        )
    except PipelineError as exc:
        st.error(str(exc))
        return MOCK_STORY_PAGES
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
        return MOCK_STORY_PAGES

    st.session_state.ambience_by_emotion = {
        emotion: clip for emotion, clip in result.ambience_by_emotion.items() if clip is not None
    }
    return result.pages


def _silent_wav_bytes(duration_s: float = 1.2, sample_rate: int = 16000) -> bytes:
    """Silent WAV placeholder for per-sentence narration audio until the
    real TTS pipeline supplies `audio_path`."""
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + n_samples * 2))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", n_samples * 2))
    buf.write(b"\x00\x00" * n_samples)
    return buf.getvalue()


PLACEHOLDER_AUDIO = _silent_wav_bytes()

# ---------------------------------------------------------------------------
# COPY
# ---------------------------------------------------------------------------

FORMAL_EN = {
    "title": "Context-Aware <em>Storyteller</em>",
    "sub": ("Upload a storybook and a reference voice. Each sentence is read in "
            "its own emotional register — and the page changes colour to match."),
    "kicker": "Emotion-aware narration",
    "up_pdf": "Storybook (PDF)", "up_voice": "Reference voice",
    "up_img": "Picture", "src_label": "Story source",
    "v_label": "Voice", "v_own": "Upload a voice", "v_default": "Built-in voice",
    "v_pick": "Choose a narrator",
    "src_img": "Picture", "src_pdf": "PDF", "src_cam": "Camera",
    "cam_hint": "Point at the picture and take a shot.",
    "illus": "The picture",
    "cta": "Generate & read story", "prev": "← Previous", "next": "Next →",
    "play": "▶ Auto-play", "pause": "Pause auto-play",
    "empty_h": "Nothing on the page yet",
    "empty_p": ("Give the storyteller a book to read and a voice to read it in. "
                "The story appears here, one sentence at a time, in the mood it "
                "was written."),
    "steps": ["01 · Storybook PDF", "02 · Voice sample", "03 · Generate"],
    "load": "Reading pages · tagging emotion · shaping voice",
    "atmos": "Atmosphere", "of": "Sentence {a} of {b}",
    "fallback_pdf": "Couldn't extract text from that PDF (scanned/image-only?) — showing the built-in sample story instead.",
    "fallback_img": "Picture-to-story isn't wired up yet — showing the built-in sample story instead.",
    "need_source": "Add a storybook or picture to begin.",
    "need_voice": "Pick a voice to begin.",
}
PLAYFUL_EN = {
    "title": "Story Time!",
    "sub": "Pick a story book and a voice — then I'll read it to you.",
    "kicker": "📖 Stories that feel things",
    "up_pdf": "📚 Pick a story book", "up_voice": "🎤 Pick a voice",
    "up_img": "🖼 Pick a picture", "src_label": "Where's the story?",
    "v_label": "Voice", "v_own": "🎤 My own voice", "v_default": "⭐ Ready-made voice",
    "v_pick": "Who should read it?",
    "src_img": "🖼 Picture", "src_pdf": "📚 Book", "src_cam": "📷 Camera",
    "cam_hint": "Hold the picture up and snap it!",
    "illus": "Our picture",
    "cta": "Read my story!", "prev": "← Back", "next": "Next →",
    "play": "▶ Read to me", "pause": "⏸ Wait a sec",
    "empty_h": "No story yet!",
    "empty_p": ("Choose a picture book and a voice. Your story will pop up "
                "right here."),
    "steps": ["1 · Story book", "2 · A voice", "3 · Read it!"],
    "load": "Turning the pages…",
    "atmos": "Feeling", "of": "Page {a} of {b}",
    "fallback_pdf": "Couldn't read that book — here's a story to try instead!",
    "fallback_img": "I can't read pictures yet — here's a story to try instead!",
    "need_source": "Add a picture or book to begin!",
    "need_voice": "Pick a voice to begin!",
}

FORMAL_ZH = {
    "title": "情境<em>说书人</em>",
    "sub": "上传一本故事书和一段参考声音。每一句都会以它自己的情绪被朗读——页面颜色也随之改变。",
    "kicker": "情绪感知朗读",
    "up_pdf": "故事书（PDF）", "up_voice": "参考声音",
    "up_img": "图片", "src_label": "故事来源",
    "v_label": "声音", "v_own": "上传声音", "v_default": "内置声音",
    "v_pick": "选择朗读者",
    "src_img": "图片", "src_pdf": "PDF", "src_cam": "拍照",
    "cam_hint": "对准图片拍一张。",
    "illus": "这张图",
    "cta": "生成并朗读故事", "prev": "← 上一句", "next": "下一句 →",
    "play": "▶ 自动播放", "pause": "暂停自动播放",
    "empty_h": "这一页还是空的",
    "empty_p": "请给说书人一本书，和一把声音。故事会出现在这里，一句一句，带着它原本的情绪。",
    "steps": ["01 · 故事书 PDF", "02 · 声音样本", "03 · 生成"],
    "load": "翻阅页面 · 标注情绪 · 塑造声音",
    "atmos": "氛围", "of": "第 {a} 句，共 {b} 句",
    "fallback_pdf": "无法从该 PDF 中提取文字（可能是扫描件或图片）——改为显示内置的示例故事。",
    "fallback_img": "图片转故事功能尚未接入——改为显示内置的示例故事。",
    "need_source": "请先添加一本故事书或图片。",
    "need_voice": "请先选择一把声音。",
}

PLAYFUL_ZH = {
    "title": "讲故事时间！",
    "sub": "选一本故事书和一把声音——我就念给你听。",
    "kicker": "📖 会有感情的故事",
    "up_pdf": "📚 选一本故事书", "up_voice": "🎤 选一把声音",
    "up_img": "🖼 选一张图片", "src_label": "故事在哪里？",
    "v_label": "声音", "v_own": "🎤 我自己的声音", "v_default": "⭐ 现成的声音",
    "v_pick": "谁来念呢？",
    "src_img": "🖼 图片", "src_pdf": "📚 故事书", "src_cam": "📷 拍照",
    "cam_hint": "把图片举起来，拍一张！",
    "illus": "我们的图",
    "cta": "念我的故事！", "prev": "← 回上页", "next": "下一页 →",
    "play": "▶ 念给我听", "pause": "⏸ 等一下",
    "empty_h": "还没有故事哦！",
    "empty_p": "选一本图画书和一把声音，你的故事马上就会出现在这里。",
    "steps": ["1 · 故事书", "2 · 一把声音", "3 · 开始念！"],
    "load": "正在翻页……",
    "atmos": "心情", "of": "第 {a} 页，共 {b} 页",
    "fallback_pdf": "没能读懂这本书——先给你讲个别的故事！",
    "fallback_img": "我还不会读图片——先给你讲个别的故事！",
    "need_source": "先加一张图片或一本书吧！",
    "need_voice": "先选一把声音吧！",
}

FORMAL = {"EN": FORMAL_EN, "ZH": FORMAL_ZH}
PLAYFUL = {"EN": PLAYFUL_EN, "ZH": PLAYFUL_ZH}

# Emotion labels in Chinese, keyed by their English label so the theme dicts
# below stay untouched.
ZH_LABELS = {
    "Crimson Anger": "赤色怒火", "Still Water": "静水", "Golden Spark": "金色火花",
    "Gloomy Violet": "幽紫", "Quiet Page": "安静的一页",
    "Grumpy!": "生气啦！", "Grumpy": "生气", "Calm": "平静", "Zoomy!": "好兴奋！",
    "Sparkly": "闪闪发亮", "A bit sad": "有点难过", "Sleepy & soft": "困困的",
    "Just telling": "慢慢讲",
}

# Built-in narrators. Point these at reference clips shipped with the app; the
# path is handed to generate_mock_story() exactly like an uploaded file would
# be. Missing files just disable "Default" mode until the team adds them.
DEFAULT_VOICES = {
    "warm":   {"EN": "Warm grandparent", "ZH": "温暖的爷爷奶奶", "face": "🧓",
               "path": "assets/voices/warm.wav"},
    "bright": {"EN": "Bright & bouncy",  "ZH": "活泼开朗",       "face": "🐣",
               "path": "assets/voices/bright.wav"},
    "gentle": {"EN": "Gentle bedtime",   "ZH": "轻声哄睡",       "face": "🌙",
               "path": "assets/voices/gentle.wav"},
}

# Looping background atmosphere, one file per emotion. Point these at your own
# assets; missing files are skipped silently (the toggle just does nothing).
SFX = {
    "angry":   "assets/sfx/storm.mp3",
    "calm":    "assets/sfx/river.mp3",
    "excited": "assets/sfx/sparkle.mp3",
    "sad":     "assets/sfx/rain.mp3",
    "neutral": "assets/sfx/room.mp3",
}


# ---------------------------------------------------------------------------
# THEMES
# Every ink-on-wash pairing below was checked against WCAG AA (4.5:1). The
# lowest in the file is 9.4:1 (Classic muted ink on its lightest gradient stop);
# all child themes sit above 10:1. Emotion is never colour-only — a face and a
# word label always travel with it.
# ---------------------------------------------------------------------------

THEMES = {

    "Classic": {
        "copy": FORMAL,
        "f_display": "'Source Serif 4', Georgia, serif",
        "f_ui": "'IBM Plex Sans', system-ui, sans-serif",
        "f_mono": "'IBM Plex Mono', ui-monospace, monospace",
        "app_bg": "radial-gradient(1100px 600px at 50% -10%,#16181F 0%,#0D0F13 60%)",
        "ink": "#F6F1E8", "ink_muted": "#CFC4B4", "ink_faint": "#9A9186",
        "hair": "rgba(246,241,232,.10)", "surface": "rgba(246,241,232,.04)",
        "grain": "rgba(246,241,232,.022)",
        "title_w": "600", "title_max": "46px", "kick_size": "11px",
        "kick_ls": ".16em", "kick_tt": "uppercase", "kick_w": "400",
        "sub_size": "15px", "rule_h": "1px",
        "up_border": "1px dashed rgba(246,241,232,.16)", "up_radius": "14px",
        "up_hover": "rgba(246,241,232,.32)", "up_lift": "none",
        "lbl_size": "11px", "lbl_w": "400", "lbl_ls": ".14em", "lbl_tt": "uppercase",
        "btn_bg": "rgba(246,241,232,.04)", "btn_ink": "#F6F1E8",
        "btn_border": "1px solid rgba(246,241,232,.14)", "btn_radius": "12px",
        "btn_shadow": "none", "btn_press": "none", "btn_minh": "46px",
        "btn_w": "500", "btn_size": "14px", "btn_font": "'IBM Plex Sans', sans-serif",
        "btn_dy": "0",
        "cta_bg": "linear-gradient(180deg,#F3E3C4,#E4CB9B)", "cta_ink": "#1A1408",
        "cta_border": "1px solid transparent",
        "card_border": "1px solid rgba(246,241,232,.09)",
        "card_radius": "6px 20px 20px 6px",
        "card_pad_y": "52px", "card_pad_x": "56px",
        "card_shadow": "0 26px 70px -28px rgba({glow},.85)",
        "spine": "rgba(0,0,0,.45)", "curl": "rgba(246,241,232,.06)",
        "story_size": "clamp(19px,4.4vw,30px)", "story_style": "italic",
        "story_w": "400",
        "meta_size": "10.5px", "meta_ls": ".18em", "meta_tt": "uppercase",
        "meta_w": "400", "meta_font": "'IBM Plex Mono', monospace",
        "badge_font": "'IBM Plex Mono', monospace", "badge_w": "500",
        "badge_size": "clamp(11px,2.6vw,11.5px)", "badge_pad": "7px 15px",
        "badge_border": "none", "badge_ls": ".14em", "badge_tt": "uppercase",
        "badge_rot": "0deg", "badge_glyph": "13px",
        "face_size": "44px", "face_font": "20px", "face_op": ".8",
        "face_border": "1px solid rgba(246,241,232,.16)",
        "face_shadow": "0 6px 18px -8px rgba(0,0,0,.6)", "face_gap": "58px",
        "audio_pad": "0", "audio_wrap_bg": "transparent",
        "audio_wrap_border": "1px solid rgba(246,241,232,.09)",
        "audio_wrap_radius": "0 0 18px 18px",
        "audio_filter": "invert(92%) hue-rotate(180deg) contrast(.92) saturate(.6)",
        "wave_h": "26px", "wave_radius": "2px", "wave_op": ".55",
        "progress": "bars", "prog_h": "3px", "prog_radius": "2px", "prog_gap": "4px",
        "prog_track": "rgba(246,241,232,.10)",
        "count_size": "11px", "count_ls": ".14em", "count_tt": "uppercase",
        "count_w": "400",
        "empty_border": "1px solid rgba(246,241,232,.10)", "empty_radius": "18px",
        "empty_shadow": "none", "empty_h_size": "22px", "empty_h_w": "500",
        "empty_p_size": "14px",
        "step_size": "11px", "step_w": "400", "step_ls": ".08em",
        "step_pad": "7px 14px", "step_border": "1px solid rgba(246,241,232,.10)",
        "step_done_bg": "#E4CB9B", "step_done_ink": "#1A1408",
        "step_done_border": "transparent",
        "dot_size": "10px", "dot_border": "none",
        "dot_a": "#E4CB9B", "dot_b": "#CFC4B4", "dot_c": "#9A9186",
        "load_size": "13px", "load_w": "400", "load_ls": ".14em",
        "load_tt": "uppercase",
        "emotions": {
            "angry":   {"label": "Crimson Anger", "face": "😠", "wash": "linear-gradient(155deg,#2E1113,#5B1E20)", "ink": "#F6F1E8", "line": "#C0605C", "chip": "#E9A6A2", "chip_ink": "#14100C", "glow": "91,30,32",  "wig": "none"},
            "calm":    {"label": "Still Water",   "face": "😌", "wash": "linear-gradient(155deg,#0E2429,#17383D)", "ink": "#F6F1E8", "line": "#5C9EA3", "chip": "#9FD3D6", "chip_ink": "#14100C", "glow": "23,56,61",  "wig": "none"},
            "excited": {"label": "Golden Spark",  "face": "🤩", "wash": "linear-gradient(155deg,#2C1C08,#573710)", "ink": "#F6F1E8", "line": "#C89446", "chip": "#F0C67F", "chip_ink": "#14100C", "glow": "87,55,16",  "wig": "none"},
            "sad":     {"label": "Gloomy Violet", "face": "😢", "wash": "linear-gradient(155deg,#151A31,#2A2450)", "ink": "#F6F1E8", "line": "#7C77BE", "chip": "#B6B2E4", "chip_ink": "#14100C", "glow": "42,36,80",  "wig": "none"},
            "neutral": {"label": "Quiet Page",    "face": "🙂", "wash": "linear-gradient(155deg,#1B1917,#2B2724)", "ink": "#F6F1E8", "line": "#8A8076", "chip": "#D6CCBC", "chip_ink": "#14100C", "glow": "43,39,36",  "wig": "none"},
        },
    },

    "Crayon": {
        "copy": PLAYFUL,
        "f_display": "'Baloo 2', 'Comic Sans MS', cursive",
        "f_ui": "'Nunito', system-ui, sans-serif",
        "f_mono": "'Nunito', system-ui, sans-serif",
        "app_bg": "#FFF9EE",
        "ink": "#2A2520", "ink_muted": "#5C5449", "ink_faint": "#8A7A60",
        "hair": "rgba(42,37,32,.14)", "surface": "#FFFBF0",
        "grain": "rgba(42,37,32,.05)",
        "title_w": "800", "title_max": "52px", "kick_size": "13px",
        "kick_ls": ".06em", "kick_tt": "uppercase", "kick_w": "800",
        "sub_size": "17px", "rule_h": "2px",
        "up_border": "2.5px dashed #2A2520", "up_radius": "20px",
        "up_hover": "#F4845F", "up_lift": "translateY(-2px)",
        "lbl_size": "14px", "lbl_w": "800", "lbl_ls": ".02em", "lbl_tt": "none",
        "btn_bg": "#FFFBF0", "btn_ink": "#2A2520",
        "btn_border": "2.5px solid #2A2520", "btn_radius": "16px",
        "btn_shadow": "0 4px 0 #2A2520", "btn_press": "0 1px 0 #2A2520",
        "btn_minh": "54px", "btn_w": "700", "btn_size": "16px",
        "btn_font": "'Baloo 2', cursive", "btn_dy": "3px",
        "cta_bg": "#F4845F", "cta_ink": "#2A2520",
        "cta_border": "2.5px solid #2A2520",
        "card_border": "3px solid #2A2520", "card_radius": "8px 24px 24px 8px",
        "card_pad_y": "44px", "card_pad_x": "40px",
        "card_shadow": "7px 7px 0 {ink}",
        "spine": "rgba(42,37,32,.10)", "curl": "rgba(42,37,32,.10)",
        "story_size": "clamp(21px,5vw,30px)", "story_style": "normal",
        "story_w": "600",
        "meta_size": "12px", "meta_ls": ".06em", "meta_tt": "uppercase",
        "meta_w": "800", "meta_font": "'Nunito', sans-serif",
        "badge_font": "'Baloo 2', cursive", "badge_w": "800",
        "badge_size": "clamp(13px,3vw,15px)", "badge_pad": "8px 16px",
        "badge_border": "2.5px solid #2A2520", "badge_ls": "0", "badge_tt": "none",
        "badge_rot": "0deg", "badge_glyph": "19px",
        "face_size": "62px", "face_font": "32px", "face_op": "1",
        "face_border": "3px solid currentColor",
        "face_shadow": "4px 4px 0 currentColor", "face_gap": "78px",
        "audio_pad": "10px 12px", "audio_wrap_bg": "#FFFBF0",
        "audio_wrap_border": "3px solid #2A2520",
        "audio_wrap_radius": "0 0 24px 8px",
        "audio_filter": "none",
        "wave_h": "30px", "wave_radius": "3px", "wave_op": ".85",
        "progress": "bars", "prog_h": "9px", "prog_radius": "5px", "prog_gap": "5px",
        "prog_track": "rgba(42,37,32,.12)",
        "count_size": "14px", "count_ls": "0", "count_tt": "none", "count_w": "700",
        "empty_border": "3px solid #2A2520", "empty_radius": "26px",
        "empty_shadow": "7px 7px 0 #2A2520", "empty_h_size": "28px",
        "empty_h_w": "800", "empty_p_size": "16px",
        "step_size": "14px", "step_w": "800", "step_ls": "0",
        "step_pad": "9px 16px", "step_border": "2.5px solid #2A2520",
        "step_done_bg": "#A8E0A0", "step_done_ink": "#12280F",
        "step_done_border": "#2A2520",
        "dot_size": "18px", "dot_border": "2.5px solid #2A2520",
        "dot_a": "#F4845F", "dot_b": "#FFD166", "dot_c": "#A8E0A0",
        "load_size": "20px", "load_w": "700", "load_ls": "0", "load_tt": "none",
        "emotions": {
            "angry":   {"label": "Grumpy!",      "face": "😠", "wash": "#FBE0DB", "ink": "#46150F", "line": "#C4503C", "chip": "#F49A8B", "chip_ink": "#3A0F0A", "glow": "70,21,15", "wig": "cs-shake 1.1s ease-in-out infinite"},
            "calm":    {"label": "Calm",         "face": "😌", "wash": "#E4F5E1", "ink": "#16301B", "line": "#5FA85A", "chip": "#A8E0A0", "chip_ink": "#12280F", "glow": "22,48,27", "wig": "cs-bob 3.2s ease-in-out infinite"},
            "excited": {"label": "Zoomy!",       "face": "🤩", "wash": "#FFEFC7", "ink": "#2A2520", "line": "#F4845F", "chip": "#FFD166", "chip_ink": "#2A2520", "glow": "42,37,32", "wig": "cs-shake 1.6s ease-in-out infinite"},
            "sad":     {"label": "A bit sad",    "face": "😢", "wash": "#E3E7F8", "ink": "#1E2450", "line": "#5764B4", "chip": "#A8B4E8", "chip_ink": "#161B3E", "glow": "30,36,80", "wig": "cs-bob 4s ease-in-out infinite"},
            "neutral": {"label": "Just telling", "face": "🙂", "wash": "#F2EDE2", "ink": "#33302A", "line": "#8A8073", "chip": "#DED6C6", "chip_ink": "#2A2822", "glow": "51,48,42", "wig": "none"},
        },
    },

    "Sticker": {
        "copy": PLAYFUL,
        "f_display": "'Baloo 2', 'Comic Sans MS', cursive",
        "f_ui": "'Nunito', system-ui, sans-serif",
        "f_mono": "'Nunito', system-ui, sans-serif",
        "app_bg": "#EAF1FF",
        "ink": "#1B2A4A", "ink_muted": "#3C4E75", "ink_faint": "#5C77A8",
        "hair": "rgba(27,42,74,.16)", "surface": "#F2F6FF",
        "grain": "rgba(27,42,74,.04)",
        "title_w": "800", "title_max": "52px", "kick_size": "13px",
        "kick_ls": ".08em", "kick_tt": "uppercase", "kick_w": "800",
        "sub_size": "17px", "rule_h": "3px",
        "up_border": "3px solid #1B2A4A", "up_radius": "18px",
        "up_hover": "#7AC74F", "up_lift": "translateY(-3px)",
        "lbl_size": "14px", "lbl_w": "800", "lbl_ls": ".02em", "lbl_tt": "none",
        "btn_bg": "#FFFFFF", "btn_ink": "#1B2A4A",
        "btn_border": "3px solid #1B2A4A", "btn_radius": "16px",
        "btn_shadow": "0 4px 0 #1B2A4A", "btn_press": "0 1px 0 #1B2A4A",
        "btn_minh": "54px", "btn_w": "800", "btn_size": "16px",
        "btn_font": "'Baloo 2', cursive", "btn_dy": "3px",
        "cta_bg": "#FFD166", "cta_ink": "#1B2A4A",
        "cta_border": "3px solid #1B2A4A",
        "card_border": "3px solid #1B2A4A", "card_radius": "18px",
        "card_pad_y": "40px", "card_pad_x": "36px",
        "card_shadow": "0 0 0 4px #FFFFFF, 0 8px 16px -2px rgba({glow},.4)",
        "spine": "rgba(27,42,74,.10)", "curl": "rgba(27,42,74,.08)",
        "story_size": "clamp(21px,5vw,29px)", "story_style": "normal",
        "story_w": "600",
        "meta_size": "12px", "meta_ls": ".12em", "meta_tt": "uppercase",
        "meta_w": "800", "meta_font": "'Nunito', sans-serif",
        "badge_font": "'Baloo 2', cursive", "badge_w": "800",
        "badge_size": "clamp(13px,3vw,15px)", "badge_pad": "8px 16px",
        "badge_border": "3px solid #1B2A4A", "badge_ls": "0", "badge_tt": "none",
        "badge_rot": "-10deg", "badge_glyph": "20px",
        "face_size": "60px", "face_font": "30px", "face_op": "1",
        "face_border": "3px solid #1B2A4A", "face_shadow": "0 4px 0 #1B2A4A",
        "face_gap": "76px",
        "audio_pad": "10px 12px", "audio_wrap_bg": "#F2F6FF",
        "audio_wrap_border": "3px solid #1B2A4A",
        "audio_wrap_radius": "0 0 15px 15px",
        "audio_filter": "none",
        "wave_h": "26px", "wave_radius": "2px", "wave_op": ".9",
        "progress": "dots", "prog_h": "14px", "prog_radius": "999px",
        "prog_gap": "7px", "prog_track": "#FFFFFF",
        "count_size": "14px", "count_ls": "0", "count_tt": "none", "count_w": "800",
        "empty_border": "3px solid #1B2A4A", "empty_radius": "24px",
        "empty_shadow": "0 8px 0 #C9D6EE", "empty_h_size": "28px",
        "empty_h_w": "800", "empty_p_size": "16px",
        "step_size": "14px", "step_w": "800", "step_ls": "0",
        "step_pad": "9px 16px", "step_border": "3px solid #1B2A4A",
        "step_done_bg": "#7AC74F", "step_done_ink": "#10240A",
        "step_done_border": "#1B2A4A",
        "dot_size": "18px", "dot_border": "3px solid #1B2A4A",
        "dot_a": "#F4845F", "dot_b": "#FFD166", "dot_c": "#7AC74F",
        "load_size": "20px", "load_w": "800", "load_ls": "0", "load_tt": "none",
        "emotions": {
            "angry":   {"label": "Grumpy!",      "face": "😠", "wash": "#FFE4DE", "ink": "#4A1408", "line": "#E0553A", "chip": "#F4845F", "chip_ink": "#3A0F04", "glow": "224,85,58",  "wig": "cs-shake 1s ease-in-out infinite"},
            "calm":    {"label": "Calm",         "face": "😌", "wash": "#E6F7DF", "ink": "#173A10", "line": "#5CA83F", "chip": "#7AC74F", "chip_ink": "#10240A", "glow": "92,168,63",  "wig": "cs-bob 3.4s ease-in-out infinite"},
            "excited": {"label": "Zoomy!",       "face": "🤩", "wash": "#FFF3D2", "ink": "#4A3405", "line": "#E8A317", "chip": "#FFD166", "chip_ink": "#3A2803", "glow": "232,163,23", "wig": "cs-shake 1.5s ease-in-out infinite"},
            "sad":     {"label": "A bit sad",    "face": "😢", "wash": "#E2EDFF", "ink": "#14294F", "line": "#4B7FD1", "chip": "#7FB2FF", "chip_ink": "#0F2144", "glow": "75,127,209", "wig": "cs-bob 4.2s ease-in-out infinite"},
            "neutral": {"label": "Just telling", "face": "🙂", "wash": "#F2F6FF", "ink": "#1B2A4A", "line": "#5C77A8", "chip": "#D6E2F7", "chip_ink": "#14213D", "glow": "92,119,168", "wig": "none"},
        },
    },

    "Bedtime": {
        "copy": PLAYFUL,
        "f_display": "'Quicksand', system-ui, sans-serif",
        "f_ui": "'Quicksand', system-ui, sans-serif",
        "f_mono": "'Quicksand', system-ui, sans-serif",
        "app_bg": "#F4F1FA",
        "ink": "#332A4A", "ink_muted": "#5F5280", "ink_faint": "#9A8CB8",
        "hair": "rgba(122,104,160,.22)", "surface": "#FFFDF9",
        "grain": "rgba(74,62,104,.035)",
        "title_w": "700", "title_max": "48px", "kick_size": "12px",
        "kick_ls": ".14em", "kick_tt": "uppercase", "kick_w": "700",
        "sub_size": "16px", "rule_h": "1px",
        "up_border": "2px solid #C7BBE0", "up_radius": "24px",
        "up_hover": "#7A6CA0", "up_lift": "translateY(-2px)",
        "lbl_size": "14px", "lbl_w": "700", "lbl_ls": ".02em", "lbl_tt": "none",
        "btn_bg": "#FFFFFF", "btn_ink": "#4A3E68",
        "btn_border": "2px solid #C7BBE0", "btn_radius": "999px",
        "btn_shadow": "0 4px 14px -8px rgba(90,74,130,.6)", "btn_press": "none",
        "btn_minh": "50px", "btn_w": "700", "btn_size": "15px",
        "btn_font": "'Quicksand', sans-serif", "btn_dy": "0",
        "cta_bg": "#7A6CA0", "cta_ink": "#FFFDF9",
        "cta_border": "2px solid #7A6CA0",
        "card_border": "2px solid #C7BBE0", "card_radius": "10px 24px 24px 10px",
        "card_pad_y": "46px", "card_pad_x": "42px",
        "card_shadow": "0 20px 44px -22px rgba({glow},.75)",
        "spine": "rgba(122,104,160,.16)", "curl": "rgba(122,104,160,.10)",
        "story_size": "clamp(20px,4.6vw,29px)", "story_style": "normal",
        "story_w": "600",
        "meta_size": "11px", "meta_ls": ".16em", "meta_tt": "uppercase",
        "meta_w": "700", "meta_font": "'Quicksand', sans-serif",
        "badge_font": "'Quicksand', sans-serif", "badge_w": "700",
        "badge_size": "clamp(12px,2.8vw,14px)", "badge_pad": "8px 16px",
        "badge_border": "2px solid #C7BBE0", "badge_ls": "0", "badge_tt": "none",
        "badge_rot": "0deg", "badge_glyph": "18px",
        "face_size": "56px", "face_font": "28px", "face_op": "1",
        "face_border": "2px solid #C7BBE0",
        "face_shadow": "0 8px 20px -10px rgba(90,74,130,.7)", "face_gap": "72px",
        "audio_pad": "10px 12px", "audio_wrap_bg": "#FFFDF9",
        "audio_wrap_border": "2px solid #C7BBE0",
        "audio_wrap_radius": "0 0 24px 10px",
        "audio_filter": "none",
        "wave_h": "22px", "wave_radius": "3px", "wave_op": ".8",
        "progress": "track", "prog_h": "8px", "prog_radius": "5px",
        "prog_gap": "0", "prog_track": "#FFFFFF",
        "count_size": "13px", "count_ls": "0", "count_tt": "none", "count_w": "700",
        "empty_border": "2px solid #C7BBE0", "empty_radius": "28px",
        "empty_shadow": "0 18px 40px -24px rgba(90,74,130,.7)",
        "empty_h_size": "26px", "empty_h_w": "700", "empty_p_size": "16px",
        "step_size": "13px", "step_w": "700", "step_ls": "0",
        "step_pad": "9px 16px", "step_border": "2px solid #C7BBE0",
        "step_done_bg": "#7A6CA0", "step_done_ink": "#FFFDF9",
        "step_done_border": "#7A6CA0",
        "dot_size": "14px", "dot_border": "none",
        "dot_a": "#B7A8D6", "dot_b": "#C7BBE0", "dot_c": "#A8B4E8",
        "load_size": "18px", "load_w": "700", "load_ls": "0", "load_tt": "none",
        "emotions": {
            "angry":   {"label": "Grumpy",       "face": "😠", "wash": "linear-gradient(170deg,#FBEAE6,#F5DED8)", "ink": "#4A1F16", "line": "#C0705E", "chip": "#F0CFC6", "chip_ink": "#3E1810", "glow": "122,80,70",  "wig": "cs-shake 1.4s ease-in-out infinite"},
            "calm":    {"label": "Calm",         "face": "😌", "wash": "linear-gradient(170deg,#E7F3EE,#DCEBF3)", "ink": "#1E3A38", "line": "#5E9A93", "chip": "#BFDFD8", "chip_ink": "#122B29", "glow": "94,154,147", "wig": "cs-bob 3.8s ease-in-out infinite"},
            "excited": {"label": "Sparkly",      "face": "🤩", "wash": "linear-gradient(170deg,#FBF0DC,#F6E7D2)", "ink": "#4A3618", "line": "#C79A55", "chip": "#EFD9AE", "chip_ink": "#3C2B11", "glow": "199,154,85", "wig": "cs-bob 2.4s ease-in-out infinite"},
            "sad":     {"label": "Sleepy & soft", "face": "😴", "wash": "linear-gradient(170deg,#EFE7FB,#E4EDF9)", "ink": "#332A4A", "line": "#7A6CA0", "chip": "#D7CDEC", "chip_ink": "#2A2240", "glow": "122,104,160", "wig": "cs-bob 4.4s ease-in-out infinite"},
            "neutral": {"label": "Just telling", "face": "🙂", "wash": "linear-gradient(170deg,#F6F3FB,#EFEEF6)", "ink": "#3A3550", "line": "#9A8CB8", "chip": "#E2DCF0", "chip_ink": "#2F2A45", "glow": "154,140,184", "wig": "none"},
        },
    },
}

if st.session_state.theme is None:
    # segmented_control deselects (-> None) if its active option is
    # clicked again; recover here rather than let THEMES[None] crash.
    st.session_state.theme = "Classic"
if st.session_state.lang is None:
    st.session_state.lang = "EN"
T = dict(THEMES[st.session_state.theme])
LANG = st.session_state.lang
C = T["copy"][LANG]
if LANG == "ZH":
    _cjk = ", 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif"
    T["f_display"] = T["f_display"] + ", 'Noto Serif SC'" + _cjk
    T["f_ui"] = T["f_ui"] + _cjk
    T["f_mono"] = T["f_mono"] + _cjk
    T["story_style"] = "normal"        # no synthetic italics on CJK
    T["meta_ls"] = "0"                 # tracking hurts Chinese
    T["kick_ls"] = "0"
    T["badge_ls"] = "0"


def emo(name):
    return T["emotions"].get((name or "neutral").lower(), T["emotions"]["neutral"])


def lab(e):
    """Emotion label in the active language."""
    return ZH_LABELS.get(e["label"], e["label"]) if LANG == "ZH" else e["label"]


# ---------------------------------------------------------------------------
# STYLE — reads T[...] only; no per-theme branching.
# Injected via st.html() (not st.markdown), since the Markdown parser reads
# CSS comment banners like /* ── Title ── */ as emphasis/hr syntax and
# mangles the stylesheet instead of applying it.
# ---------------------------------------------------------------------------

st.html(f"""
<style>
/* Fonts are pulled in via @import, not a link element in the HTML body —
   st.html()'s sanitizer silently strips link elements (and drops the rest
   of the payload if it spots anything else that merely looks like a tag,
   even inside a style block's raw-text content, e.g. a stray angle-bracket
   pair in a comment). @import inside the style element sidesteps both. */
@import url("https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300..700;1,8..60,300..600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Baloo+2:wght@400;500;600;700;800&family=Nunito:wght@400;600;700;800&family=Quicksand:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700;900&family=Noto+Serif+SC:wght@400;600;700&display=swap");
:root{{ --ink:{T['ink']}; --muted:{T['ink_muted']}; --faint:{T['ink_faint']};
        --hair:{T['hair']}; --step:8px; }}
.stApp{{ background:{T['app_bg']}; }}
.block-container{{ max-width:800px; padding:calc(var(--step)*3) var(--step) calc(var(--step)*10); }}
header[data-testid="stHeader"]{{ background:transparent; }}
html, body, [class*="css"], .stMarkdown{{ font-family:{T['f_ui']}; color:var(--ink); }}

/* Picker row container (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"]{{
  background:{T['surface']}; border:{T['card_border']} !important;
  border-radius:{T['up_radius']}; padding:calc(var(--step)*1.5) calc(var(--step)*2);
  margin-bottom:calc(var(--step)*3); }}

/* Segmented control — Streamlit renders options as
   stBaseButton-segmented_control (unselected) / -segmented_controlActive
   (selected), grouped in a stButtonGroup. Style both states from theme
   tokens so the selected option never shows Streamlit's default red. */
[data-testid="stButtonGroup"]{{ flex-wrap:wrap; gap:6px; }}
[data-testid="stButtonGroup"] [data-testid="stWidgetLabel"] p{{
  font-family:{T['f_mono']}; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted) !important; }}
[data-testid^="stBaseButton-segmented_control"]{{
  font-family:{T['f_display']}; font-weight:{T['btn_w']}; font-size:14px;
  min-height:40px; border-radius:{T['btn_radius']};
  background:{T['btn_bg']} !important; color:{T['btn_ink']} !important;
  border:{T['btn_border']} !important; transition:background .2s ease; }}
[data-testid="stBaseButton-segmented_controlActive"]{{
  background:{T['cta_bg']} !important; color:{T['cta_ink']} !important;
  border:{T['cta_border']} !important; font-weight:800; }}
[data-testid="stRadio"] label p{{
  font-family:{T['f_display']}; font-weight:{T['btn_w']}; font-size:14px;
  color:var(--ink) !important; }}
[data-testid="stRadio"] > div{{ gap:14px; }}

/* Toggle (st.toggle renders as stCheckbox styled as a switch). The track
   is the first inner div, the knob is the div nested inside it; neither
   carries a stable testid, so they're targeted structurally under the
   stable stCheckbox container. */
[data-testid="stCheckbox"] p{{ color:var(--ink) !important; font-family:{T['f_ui']}; }}
[data-testid="stCheckbox"] label > div:first-child{{
  background:{T['btn_bg']} !important; border:{T['btn_border']}; }}
[data-testid="stCheckbox"] label:has(input:checked) > div:first-child{{
  background:{T['cta_bg']} !important; }}
[data-testid="stCheckbox"] label > div:first-child > div{{
  background:{T['btn_ink']} !important; }}
[data-testid="stCheckbox"] label:has(input:checked) > div:first-child > div{{
  background:{T['cta_ink']} !important; }}

.cs-masthead{{ display:flex; flex-direction:column; gap:calc(var(--step)*1.5);
  margin:calc(var(--step)*3) 0 calc(var(--step)*4); }}
.cs-title{{ font-family:{T['f_display']}; font-weight:{T['title_w']};
  font-size:clamp(32px,6.5vw,{T['title_max']}); line-height:1.05;
  letter-spacing:-.02em; margin:0; text-wrap:balance;
  color:var(--ink) !important; }}
.cs-title em{{ font-style:italic; font-weight:400; color:var(--muted) !important; }}
.cs-kicker{{ font-family:{T['f_mono']}; font-size:{T['kick_size']};
  letter-spacing:{T['kick_ls']}; text-transform:{T['kick_tt']};
  font-weight:{T['kick_w']}; color:var(--muted) !important; margin:0; }}
.cs-sub{{ font-size:{T['sub_size']}; line-height:1.6; color:var(--muted) !important;
  margin:0; max-width:52ch; text-wrap:pretty; }}
.cs-rule{{ height:{T['rule_h']}; border:0; border-radius:3px; background:var(--hair);
  margin:calc(var(--step)*3) 0; }}

[data-testid="stFileUploader"] section{{ background:{T['surface']};
  border:{T['up_border']}; border-radius:{T['up_radius']};
  padding:calc(var(--step)*2.5);
  transition:border-color .25s ease, background .25s ease, transform .2s ease; }}
[data-testid="stFileUploader"] section:hover{{ border-color:{T['up_hover']};
  transform:{T['up_lift']}; }}
[data-testid="stFileUploader"] label p{{ font-family:{T['f_mono']};
  font-size:{T['lbl_size']}; font-weight:{T['lbl_w']}; letter-spacing:{T['lbl_ls']};
  text-transform:{T['lbl_tt']}; color:var(--muted) !important; }}
[data-testid="stFileUploaderDropzone"] button{{
  background:{T['btn_bg']} !important; color:{T['btn_ink']} !important;
  border:{T['btn_border']} !important; border-radius:{T['btn_radius']}; }}
[data-testid="stFileUploaderDropzone"] button p{{ color:{T['btn_ink']} !important; }}
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"]{{
  color:{T['btn_ink']} !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] span{{
  color:var(--muted) !important; font-family:{T['f_ui']}; }}
[data-testid="stCameraInput"] button{{
  background:{T['btn_bg']} !important; color:{T['btn_ink']} !important;
  border:{T['btn_border']} !important; }}

.stButton > button{{ width:100%; min-height:{T['btn_minh']};
  font-family:{T['btn_font']}; font-weight:{T['btn_w']}; font-size:{T['btn_size']};
  border-radius:{T['btn_radius']}; border:{T['btn_border']}; background:{T['btn_bg']};
  color:{T['btn_ink']}; box-shadow:{T['btn_shadow']};
  transition:transform .12s ease, box-shadow .12s ease, background .2s ease; }}
.stButton > button:hover{{ transform:translateY(-1px); }}
.stButton > button:active{{ transform:translateY({T['btn_dy']});
  box-shadow:{T['btn_press']}; }}
.stButton > button:focus-visible{{ outline:3px solid {T['ink_faint']}; outline-offset:3px; }}
.stButton > button:disabled, .stButton > button:disabled *{{ opacity:1 !important; }}
.stButton > button:disabled{{ box-shadow:none; transform:none;
  background:{T['surface']} !important;
  border:2px dashed {T['ink_faint']} !important; }}
.stButton > button:disabled, .stButton > button:disabled p,
.stButton > button:disabled span, .stButton > button:disabled div{{
  color:var(--muted) !important; -webkit-text-fill-color:var(--muted) !important; }}
.stButton > button[kind="primary"]{{ background:{T['cta_bg']}; color:{T['cta_ink']};
  border:{T['cta_border']}; font-weight:800; box-shadow:{T['btn_shadow']}; }}
.cs-need{{ text-align:center; font-size:13px; color:var(--muted) !important;
  margin:calc(var(--step)*1) 0 0; }}

.cs-empty{{ position:relative; overflow:hidden; text-align:center;
  border:{T['empty_border']}; border-radius:{T['empty_radius']};
  background:{T['surface']}; box-shadow:{T['empty_shadow']};
  padding:calc(var(--step)*5) calc(var(--step)*4); margin-top:calc(var(--step)*3); }}
.cs-empty::before{{ content:""; position:absolute; inset:0; pointer-events:none;
  opacity:.6; background:repeating-linear-gradient(0deg,{T['grain']} 0 1px,transparent 1px 4px); }}
.cs-empty::after{{ content:""; position:absolute; right:0; bottom:0;
  width:74px; height:74px;
  background:linear-gradient(135deg,transparent 50%,{T['curl']} 50%); }}
.cs-empty h3{{ font-family:{T['f_display']}; font-weight:{T['empty_h_w']};
  font-size:{T['empty_h_size']}; margin:0 0 10px; position:relative;
  color:var(--ink) !important; }}
.cs-empty p{{ color:var(--muted) !important; font-size:{T['empty_p_size']}; line-height:1.65;
  margin:0 auto; max-width:42ch; text-wrap:pretty; position:relative; }}
.cs-steps{{ display:flex; gap:10px; justify-content:center; flex-wrap:wrap;
  margin-top:calc(var(--step)*3); position:relative; }}
.cs-step{{ font-family:{T['f_mono']}; font-size:{T['step_size']};
  font-weight:{T['step_w']}; letter-spacing:{T['step_ls']}; color:var(--muted) !important;
  border:{T['step_border']}; border-radius:999px; padding:{T['step_pad']};
  background:{T['surface']}; }}
.cs-step[data-done="1"]{{ background:{T['step_done_bg']}; color:{T['step_done_ink']} !important;
  border-color:{T['step_done_border']}; }}

.cs-loading{{ text-align:center; padding:calc(var(--step)*6) 0; }}
.cs-dots{{ display:flex; gap:12px; justify-content:center; margin-bottom:18px; }}
.cs-dots i{{ width:{T['dot_size']}; height:{T['dot_size']}; border-radius:999px;
  border:{T['dot_border']}; background:{T['dot_a']};
  animation:cs-bounce 1.1s ease-in-out infinite; }}
.cs-dots i:nth-child(2){{ animation-delay:.14s; background:{T['dot_b']}; }}
.cs-dots i:nth-child(3){{ animation-delay:.28s; background:{T['dot_c']}; }}
@keyframes cs-bounce{{ 0%,80%,100%{{transform:translateY(0)}} 40%{{transform:translateY(-14px)}} }}
.cs-loading p{{ font-family:{T['f_display']}; font-size:{T['load_size']};
  font-weight:{T['load_w']}; letter-spacing:{T['load_ls']};
  text-transform:{T['load_tt']}; color:var(--muted) !important; margin:0; }}

.cs-badge{{ display:inline-flex; align-items:center; gap:9px;
  font-family:{T['badge_font']}; font-weight:{T['badge_w']};
  font-size:{T['badge_size']}; letter-spacing:{T['badge_ls']};
  text-transform:{T['badge_tt']}; padding:{T['badge_pad']}; border-radius:999px;
  border:{T['badge_border']}; transform:rotate({T['badge_rot']});
  margin-bottom:calc(var(--step)*1.5);
  transition:background .8s ease, color .8s ease, border-color .8s ease; }}
.cs-badge span{{ font-size:{T['badge_glyph']}; line-height:1; display:inline-block; }}
@keyframes cs-bob{{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-4px)}} }}
@keyframes cs-shake{{ 0%,100%{{transform:rotate(-9deg)}} 50%{{transform:rotate(9deg)}} }}

.cs-stage{{ perspective:1600px; }}
.cs-page{{ position:relative; overflow:hidden; border:{T['card_border']};
  border-radius:{T['card_radius']};
  padding:clamp(24px,5vw,{T['card_pad_y']}) clamp(22px,5vw,{T['card_pad_x']});
  transform-origin:left center; backface-visibility:hidden;
  transition:background .8s ease, box-shadow .8s ease, border-color .8s ease; }}
.cs-page::before{{ content:""; position:absolute; inset:0; pointer-events:none;
  opacity:.6; background:repeating-linear-gradient(0deg,{T['grain']} 0 1px,transparent 1px 4px); }}
.cs-page::after{{ content:""; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(90deg,{T['spine']} 0,transparent 44px),
             radial-gradient(110px 110px at 100% 100%,{T['curl']},transparent 62%); }}
.cs-face{{ position:absolute; z-index:2; top:16px; right:16px; opacity:{T['face_op']};
  width:{T['face_size']}; height:{T['face_size']}; border-radius:999px;
  display:grid; place-items:center; font-size:{T['face_font']}; line-height:1;
  border:{T['face_border']}; box-shadow:{T['face_shadow']};
  transition:background .8s ease; }}
.cs-face i{{ display:block; font-style:normal; }}
/* Spacer that reserves room for the face medallion only where text would
   actually overlap it (roughly its first two lines), instead of padding
   the whole paragraph and leaving a permanent empty gutter down the card. */
.cs-face-spacer{{ float:right; width:{T['face_size']}; height:calc({T['face_size']} + 6px);
  shape-outside:margin-box; }}
.cs-illus{{ position:relative; z-index:1; margin:0 0 calc(var(--step)*3);
  border:{T['card_border']}; border-radius:{T['up_radius']}; overflow:hidden;
  box-shadow:{T['face_shadow']}; line-height:0; }}
.cs-illus img{{ width:100%; height:auto; display:block;
  max-height:46vh; object-fit:cover; }}
.cs-illus figcaption{{ font-family:{T['f_mono']}; font-size:{T['count_size']};
  font-weight:{T['count_w']}; letter-spacing:{T['count_ls']};
  text-transform:{T['count_tt']}; line-height:1.4; opacity:.7;
  padding:10px 14px; }}
.cs-meta{{ position:relative; z-index:1; display:flex; align-items:center; gap:12px;
  padding-right:{T['face_gap']}; font-family:{T['meta_font']};
  font-weight:{T['meta_w']}; font-size:{T['meta_size']};
  letter-spacing:{T['meta_ls']}; text-transform:{T['meta_tt']};
  margin-bottom:calc(var(--step)*2.5); }}
.cs-meta .cs-dot{{ flex:1; height:2px; border-radius:2px; opacity:.35; }}
.cs-text{{ position:relative; z-index:1; margin:0; text-wrap:pretty;
  font-family:{T['f_display']};
  font-style:{T['story_style']}; font-weight:{T['story_w']};
  font-size:{T['story_size']}; line-height:1.5; overflow:hidden; }}
.cs-speaker{{ position:relative; z-index:1; display:flex; align-items:center; gap:10px;
  margin-top:calc(var(--step)*3); font-size:14px; font-weight:600; }}
.cs-speaker i{{ display:block; width:22px; height:3px; border-radius:2px; }}

@keyframes cs-turn-fwd{{ 0%{{transform:rotateY(-38deg) translateX(-4%);opacity:0;filter:brightness(.7)}}
  60%{{opacity:1}} 100%{{transform:none;opacity:1;filter:none}} }}
@keyframes cs-turn-back{{ 0%{{transform:rotateY(26deg) translateX(4%);opacity:0;filter:brightness(.7)}}
  60%{{opacity:1}} 100%{{transform:none;opacity:1;filter:none}} }}
.cs-page[data-dir="fwd"]{{ animation:cs-turn-fwd .62s cubic-bezier(.22,.68,.28,1) both; }}
.cs-page[data-dir="back"]{{ animation:cs-turn-back .62s cubic-bezier(.22,.68,.28,1) both;
  transform-origin:right center; }}

.cs-audio-lip{{ height:2px; border-radius:2px; margin-top:calc(var(--step)*3);
  opacity:.4; position:relative; z-index:1; }}
.cs-wave{{ display:flex; align-items:flex-end; gap:3px; height:{T['wave_h']};
  padding-top:14px; position:relative; z-index:1; }}
.cs-wave i{{ flex:1; border-radius:{T['wave_radius']}; opacity:{T['wave_op']};
  transform-origin:bottom; animation:cs-bar 1.4s ease-in-out infinite alternate; }}
@keyframes cs-bar{{ from{{transform:scaleY(.22)}} to{{transform:scaleY(1)}} }}
/* data-testid="stAudio" sits directly on the native audio player tag —
   there is no wrapping div. That native player is browser-drawn and only
   lightly restylable without JS: filter/size/background/border are all CSS
   can reliably do. Giving it its own background, border and bottom radius
   (matching the card above it) keeps it reading as part of the card
   instead of a detached grey/charcoal bar. Chromium's own default panel is
   already light, so light themes leave the filter off; only Classic (a
   dark theme) inverts it. The ::-webkit-media-controls-panel rule is a
   Chromium-only bonus tint — Firefox/Safari fall back to their own native
   control colour. The ~16px seam above the player is Streamlit's own
   flexbox gap between widget containers (stVerticalBlock's `gap`), which
   is shared by every widget on the page — closing it here would mean
   overriding that gap globally, at the cost of spacing everywhere else,
   so it's left as the documented limit of a display-layer-only fix. */
audio[data-testid="stAudio"]{{ display:block !important; box-sizing:border-box;
  width:100% !important; height:44px !important; margin:-2px 0 0 !important;
  background:{T['audio_wrap_bg']} !important;
  border:{T['audio_wrap_border']} !important; border-top:none !important;
  border-radius:{T['audio_wrap_radius']} !important; padding:{T['audio_pad']} !important;
  filter:{T['audio_filter']} !important; }}
audio[data-testid="stAudio"]::-webkit-media-controls-panel{{
  background-color:{T['audio_wrap_bg']} !important; }}

.cs-ambience + div[data-testid="stAudio"], .cs-ambience ~ div[data-testid="stAudio"]:last-of-type{{
  height:0; overflow:hidden; opacity:0; pointer-events:none; margin:0; padding:0; }}
.cs-amb-tag{{ position:absolute; z-index:2; left:16px; bottom:14px;
  display:inline-flex; align-items:center; gap:6px; font-family:{T['f_mono']};
  font-size:11px; font-weight:{T['count_w']}; opacity:.55; }}
.cs-amb-tag i{{ width:6px; height:6px; border-radius:999px; background:currentColor;
  animation:cs-bounce 1.6s ease-in-out infinite; }}

.cs-progress{{ display:flex; gap:{T['prog_gap']}; align-items:center;
  margin:calc(var(--step)*3) 0 calc(var(--step)*1.5); }}
.cs-progress i{{ flex:1; height:{T['prog_h']}; border-radius:{T['prog_radius']};
  transition:background .4s ease; }}
.cs-progress[data-kind="dots"]{{ justify-content:center; flex-wrap:wrap; }}
.cs-progress[data-kind="dots"] i{{ flex:0 0 auto; width:{T['prog_h']};
  border:{T['dot_border']}; }}
.cs-progress[data-kind="track"]{{ background:{T['prog_track']};
  border-radius:{T['prog_radius']}; overflow:hidden; }}
.cs-counter{{ display:flex; justify-content:space-between; font-family:{T['f_mono']};
  font-weight:{T['count_w']}; font-size:{T['count_size']};
  letter-spacing:{T['count_ls']}; text-transform:{T['count_tt']}; color:var(--muted) !important; }}

@media (max-width:640px){{
  .block-container{{ padding:20px 16px 60px; }}
  .cs-face{{ width:44px; height:44px; font-size:22px; top:12px; right:12px; }}
  .cs-face-spacer{{ width:44px; height:50px; }}
  .cs-meta{{ padding-right:52px; }}
  .cs-page::after{{ background:linear-gradient(90deg,{T['spine']} 0,transparent 24px); }}
  .stButton > button{{ font-size:14px; padding-left:6px; padding-right:6px; }}
  .cs-counter{{ font-size:12px; }}
}}
@media (prefers-reduced-motion:reduce){{
  .cs-page[data-dir]{{ animation:none; }}
  .cs-wave i, .cs-dots i, .cs-badge span, .cs-face i{{ animation:none !important; }}
  *{{ transition-duration:.01ms !important; }}
}}
</style>
""")

# ---------------------------------------------------------------------------
# THEME / LANGUAGE / AMBIENCE PICKER
# ---------------------------------------------------------------------------

names = list(THEMES.keys())
langs = {"EN": "English", "ZH": "中文"}

# Masthead leads — the product title is the first thing on the page, not
# three unlabelled controls. The picker row also used to sit directly under
# Streamlit's sticky toolbar, where its own overlay can intercept clicks
# meant for the buttons beneath it; moving the row down avoids that overlap.
st.html(f"""
<div class="cs-masthead">
  <p class="cs-kicker">{C['kicker']}</p>
  <h1 class="cs-title">{C['title']}</h1>
  <p class="cs-sub">{C['sub']}</p>
</div>
""")

with st.container(border=True):
    bar1, bar2, bar3 = st.columns([2.2, 1.1, 1], gap="small")

    with bar1:
        theme_label = "主题" if LANG == "ZH" else "Theme"
        if hasattr(st, "segmented_control"):
            st.segmented_control(theme_label, names, key="theme")
        else:
            st.radio(theme_label, names, key="theme", horizontal=True)

    with bar2:
        lang_label = "语言" if LANG == "ZH" else "Language"
        if hasattr(st, "segmented_control"):
            st.segmented_control(lang_label, list(langs),
                                 format_func=lambda k: langs[k], key="lang")
        else:
            st.radio(lang_label, list(langs), format_func=lambda k: langs[k],
                     key="lang", horizontal=True)

    with bar3:
        st.toggle("🔊 " + ("背景音" if LANG == "ZH" else "Ambience"), key="sfx")

# ---------------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------------

if st.session_state.source is None:
    # segmented_control deselects (-> None) if its active option is clicked
    # again; a widget-bound key can't be reassigned after instantiation, so
    # this must run before the widget below, not after.
    st.session_state.source = "PDF"
srcs = {"PDF": C["src_pdf"], "Picture": C["src_img"], "Camera": C["src_cam"]}
if hasattr(st, "segmented_control"):
    st.segmented_control(C["src_label"], list(srcs),
                         format_func=lambda k: srcs[k], key="source")
else:
    st.radio(C["src_label"], list(srcs), format_func=lambda k: srcs[k],
             key="source", horizontal=True)
src = st.session_state.source

c1, c2 = st.columns(2, gap="medium")
with c1:
    if src == "PDF":
        story_file = st.file_uploader(C["up_pdf"], type=["pdf"])
    elif src == "Camera":
        st.caption(C["cam_hint"])
        story_file = st.camera_input(C["src_cam"], label_visibility="collapsed")
    else:
        story_file = st.file_uploader(C["up_img"],
                                      type=["png", "jpg", "jpeg", "webp"])
with c2:
    if st.session_state.voice_mode is None:
        st.session_state.voice_mode = "Default"
    vmodes = {"Default": C["v_default"], "Own": C["v_own"]}
    if hasattr(st, "segmented_control"):
        st.segmented_control(C["v_label"], list(vmodes),
                             format_func=lambda k: vmodes[k],
                             key="voice_mode", label_visibility="collapsed")
    else:
        st.radio(C["v_label"], list(vmodes), format_func=lambda k: vmodes[k],
                 key="voice_mode", horizontal=True, label_visibility="collapsed")

    if st.session_state.voice_mode == "Own":
        voice_file = st.file_uploader(C["up_voice"], type=["wav", "mp3"])
    else:
        keys = list(DEFAULT_VOICES)
        pick = st.selectbox(
            C["v_pick"], keys,
            index=keys.index(st.session_state.voice_preset),
            format_func=lambda k: f'{DEFAULT_VOICES[k]["face"]}  '
                                  f'{DEFAULT_VOICES[k][LANG]}')
        st.session_state.voice_preset = pick
        path = DEFAULT_VOICES[pick]["path"]
        voice_file = path if os.path.exists(path) else None
        if voice_file:
            st.audio(voice_file)

ready = bool(story_file and voice_file)
go = st.button(C["cta"], type="primary" if ready else "secondary",
               disabled=not ready, use_container_width=True)
if not ready:
    missing = C["need_source"] if not story_file else C["need_voice"]
    st.html(f'<p class="cs-need">{missing}</p>')

# ---------------------------------------------------------------------------
# GENERATE (backend untouched — only the loading presentation is themed)
# ---------------------------------------------------------------------------

if go:
    slot = st.empty()

    def _show_loading(message):
        slot.html(f'<div class="cs-loading"><div class="cs-dots"><i></i><i></i><i></i>'
                  f'</div><p>{message}</p></div>')

    _show_loading(C["load"])
    if src == "PDF":
        pdf_bytes = story_file.getvalue()
        try:
            pages = extract_pages_from_pdf(story_file)
        except Exception:
            # extract_pages_from_pdf() documents returning None for
            # unreadable text; treat an outright parse failure the same way
            # rather than crashing the whole page on a malformed upload.
            pages = None
        st.session_state.illustration = None
        fallback_msg = C["fallback_pdf"]
        raw_source = pdf_bytes
    else:
        # Single image → one page. generate_mock_story still receives a
        # list of "pages" and returns the unchanged {page, sentences: [...]}
        # schema; raw_source carries the opened image for real vision
        # generation.
        pages = [story_file]
        image_bytes = story_file.getvalue()
        st.session_state.illustration = image_bytes
        fallback_msg = C["fallback_img"]
        raw_source = [Image.open(io.BytesIO(image_bytes))]
    story = generate_mock_story(
        pages, voice_file,
        raw_source_bytes=raw_source,
        language=STORY_LANGUAGE[LANG],
        enable_sfx=st.session_state.sfx,
        on_progress=_show_loading,
    )
    st.session_state.story = story
    st.session_state.used_fallback = story is MOCK_STORY_PAGES
    st.session_state.idx = 0
    st.session_state.dir = "fwd"
    slot.empty()

    if st.session_state.used_fallback:
        st.info(fallback_msg)

# ---------------------------------------------------------------------------
# EMPTY STATE
# ---------------------------------------------------------------------------

if not st.session_state.story:
    done = [int(bool(story_file)), int(bool(voice_file)), 0]
    chips = "".join(f'<span class="cs-step" data-done="{d}">{s}</span>'
                    for s, d in zip(C["steps"], done))
    st.html(f'<div class="cs-empty"><h3>{C["empty_h"]}</h3><p>{C["empty_p"]}</p>'
            f'<div class="cs-steps">{chips}</div></div>')
    st.stop()

# ---------------------------------------------------------------------------
# FLATTEN (schema untouched)
# ---------------------------------------------------------------------------

flat = [(p.get("page"), s) for p in st.session_state.story
        for s in p.get("sentences", [])]
total = len(flat)
st.session_state.idx = max(0, min(st.session_state.idx, total - 1))
i = st.session_state.idx
page_no, sent = flat[i]
e = emo(sent.get("emotion"))

st.html('<hr class="cs-rule">')

st.html(f'<div class="cs-badge" style="background:{e["chip"]};color:{e["chip_ink"]}">'
        f'<span style="animation:{e["wig"]}">{e["face"]}</span>'
        f'{C["atmos"]}: {lab(e)}</div>')

bars = "".join(f'<i style="background:{e["line"]};animation-delay:{(n % 9) * .11:.2f}s;'
               f'height:{30 + (n * 41) % 70}%"></i>' for n in range(26))

illus_html = ""
if st.session_state.illustration:
    b64 = base64.b64encode(st.session_state.illustration).decode()
    illus_html = (f'<figure class="cs-illus" style="background:{e["chip"]}">'
                  f'<img src="data:image/png;base64,{b64}" alt="{C["illus"]}">'
                  f'<figcaption>{C["illus"]}</figcaption></figure>')

st.html(f"""
<div class="cs-stage">
  <div class="cs-page" data-dir="{st.session_state.dir}" data-key="{i}"
       style="background:{e['wash']};color:{e['ink']};
              box-shadow:{T['card_shadow'].format(glow=e['glow'], ink=e['ink'])};">
    <div class="cs-face" style="background:{e['chip']};color:{e['ink']}">
      <i style="animation:{e['wig']}">{e['face']}</i></div>
    {illus_html}
    <div class="cs-meta" style="opacity:.75">
      <span>Page {page_no}</span>
      <span class="cs-dot" style="background:{e['line']}"></span>
      <span>{i + 1}/{total}</span></div>
    <p class="cs-text"><span class="cs-face-spacer"></span>{sent.get("text", "")}</p>
    <div class="cs-speaker" style="opacity:.8">
      <i style="background:{e['line']}"></i>{sent.get("speaker", "narrator")}</div>
    <div class="cs-audio-lip" style="background:{e['line']}"></div>
    <div class="cs-wave">{bars}</div>
  </div>
</div>
""")

# Per-sentence audio: real audio_path once the TTS pipeline provides one,
# otherwise a silent placeholder so the player is always present to test.
if sent.get("audio_path"):
    st.audio(sent["audio_path"])
else:
    st.audio(PLACEHOLDER_AUDIO, format="audio/wav")

# Looping ambience for the current emotion. Prefers a freshly Freesound-fetched
# clip (raw bytes, kept in session_state since it's fetched mid-script-run and
# a plain module global like SFX gets reset to its static defaults on every
# rerun) and falls back to a static bundled file path if present.
if st.session_state.sfx:
    emotion_key = (sent.get("emotion") or "neutral").lower()
    track = st.session_state.ambience_by_emotion.get(emotion_key)
    if track is None:
        track = SFX.get(emotion_key)
    track_is_playable = isinstance(track, (bytes, bytearray)) or (
        isinstance(track, str) and os.path.exists(track)
    )
    if track_is_playable:
        st.html('<div class="cs-ambience">')
        try:
            st.audio(track, loop=True, autoplay=True)
        except TypeError:
            st.audio(track)
        st.html('</div>')

# Progress — three shapes, one dict key
kind = T["progress"]
if kind == "track":
    pct = (i + 1) / total * 100
    seg = f'<i style="flex:0 0 {pct}%;background:{e["line"]}"></i>'
else:
    seg = "".join(f'<i style="background:{e["line"] if n <= i else T["prog_track"]}">'
                  f'</i>' for n in range(total))
st.html(f'<div class="cs-progress" data-kind="{kind}">{seg}</div>')

n1, n2, n3 = st.columns([1, 1.3, 1], gap="small")
with n1:
    if st.button(C["prev"], disabled=i == 0, use_container_width=True):
        st.session_state.dir = "back"
        st.session_state.idx -= 1
        st.rerun()
with n2:
    if st.button(C["pause"] if st.session_state.autoplay else C["play"],
                 use_container_width=True):
        st.session_state.autoplay = not st.session_state.autoplay
        st.rerun()
with n3:
    if st.button(C["next"], disabled=i >= total - 1, use_container_width=True):
        st.session_state.dir = "fwd"
        st.session_state.idx += 1
        st.rerun()

st.html(f'<div class="cs-counter"><span>{C["of"].format(a=i + 1, b=total)}</span>'
        f'<span>{lab(e)}</span></div>')

if st.session_state.autoplay and i < total - 1:
    time.sleep(3.2)
    st.session_state.dir = "fwd"
    st.session_state.idx += 1
    st.rerun()
