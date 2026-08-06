"""Pipeline orchestrator: story → theatre → clone TTS → per-sentence clips + ambience."""
from __future__ import annotations

import concurrent.futures
from collections import OrderedDict

from pipeline.audio_utils import slice_audio_by_sentences
from pipeline.prosody import EMOTION_DSP_DEFAULTS
from pipeline.sfx import fetch_ambience_clip
from pipeline.theatre import RuleBasedTheatreAdapter, theatre_lines_to_script_doc

_EMOTION_SFX_MOOD = {
    "angry": "thunderstorm",
    # "cheerful sparkle" and "quiet room tone" were empirically confirmed (live
    # Freesound queries) to return musical/off-vibe top results ("cheerful sparkle"
    # returned a calm/mellow chime track; "quiet room tone" returned an orchestral
    # recording) despite better matches existing further down the same result set —
    # these two phrasings return consistently on-vibe, ambience-tagged results instead.
    "excited": "carnival atmosphere",
    "sad": "gentle rain",
    "calm": "flowing river",
    "neutral": "room tone",
}


class PipelineResult:
    def __init__(self, pages, ambience_by_emotion, used_sfx: bool, theatre_script=None):
        self.pages = pages
        self.ambience_by_emotion = ambience_by_emotion
        self.used_sfx = used_sfx
        self.theatre_script = theatre_script or {}


def _supports_per_sentence(synth) -> bool:
    return callable(getattr(synth, "synthesize_sentences", None))


def run_pipeline(
    images,
    voice_bytes: bytes,
    language: str,
    enable_sfx: bool,
    *,
    story_generator,
    voice_cloner,
    narration_synthesizer,
    freesound_api_key: str,
    sfx_cache_dir: str,
    on_progress=None,
    theatre_adapter=None,
    prebuilt_story=None,
    page_nums=None,
) -> PipelineResult:
    """Run theatre + voice clone + narration.

    If ``prebuilt_story`` is provided (e.g. sentences extracted from a text PDF),
    vision story generation is skipped but TTS still runs — otherwise the UI would
    only show the ~1s silent placeholder.
    """
    def _report(message: str) -> None:
        if on_progress:
            on_progress(message)

    adapter = theatre_adapter or RuleBasedTheatreAdapter()

    _report("Preparing voice clone...")
    if prebuilt_story is not None:
        voice_id = voice_cloner.clone(voice_bytes, "StoryTeller Voice")
        story_result = prebuilt_story
        _report("Using extracted story text (skipping vision)…")
    else:
        _report("Generating story and preparing voice...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            story_future = executor.submit(story_generator.generate, images, language)
            clone_future = executor.submit(voice_cloner.clone, voice_bytes, "StoryTeller Voice")
            story_result = story_future.result()
            voice_id = clone_future.result()

    _report("Theatre adaptation (speakers, stage directions, performance tags)...")
    theatre_lines = adapter.adapt(story_result.sentences)
    theatre_script = theatre_lines_to_script_doc(theatre_lines, language=language)

    if page_nums is None:
        page_nums = [1] * len(theatre_lines)
    elif len(page_nums) != len(theatre_lines):
        page_nums = (list(page_nums) + [page_nums[-1]] * len(theatre_lines))[: len(theatre_lines)]

    distinct_emotions = sorted({line.emotion for line in theatre_lines})

    _report(
        f"Synthesizing narration ({len(theatre_lines)} lines on GPU — "
        "first run loads XTTS; Charlie-length PDFs can take several minutes)..."
    )

    # Prefetch ambience in the background, but run TTS on this thread.
    # Streamlit progress callbacks must NOT run inside ThreadPoolExecutor
    # (raises NoSessionContext and aborts generation).
    ambience_by_emotion = {}
    ambience_executor = None
    ambience_futures = {}
    if enable_sfx:
        ambience_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(distinct_emotions))
        )
        for emotion in distinct_emotions:
            mood = _EMOTION_SFX_MOOD[emotion]
            ambience_futures[emotion] = ambience_executor.submit(
                fetch_ambience_clip, mood, freesound_api_key, sfx_cache_dir
            )

    try:
        if _supports_per_sentence(narration_synthesizer):
            import inspect

            synth_kwargs = {}
            try:
                if "on_progress" in inspect.signature(
                    narration_synthesizer.synthesize_sentences
                ).parameters:
                    synth_kwargs["on_progress"] = _report
            except (TypeError, ValueError):
                pass
            clips = narration_synthesizer.synthesize_sentences(
                theatre_lines, voice_id, language, **synth_kwargs
            )
        else:
            combined_text = " ".join(line.text for line in theatre_lines)
            narration_result = narration_synthesizer.synthesize_with_timestamps(
                combined_text, voice_id, language
            )
            _report("Slicing narration per sentence...")
            clips = slice_audio_by_sentences(
                narration_result.audio_bytes,
                narration_result.characters,
                narration_result.character_start_times_seconds,
                narration_result.character_end_times_seconds,
                [line.text for line in theatre_lines],
            )
    finally:
        for emotion, future in ambience_futures.items():
            try:
                ambience_by_emotion[emotion] = future.result()
            except Exception:
                ambience_by_emotion[emotion] = None
        if ambience_executor is not None:
            ambience_executor.shutdown(wait=False)

    by_page: OrderedDict[int, list] = OrderedDict()
    for line, clip_bytes, page_no in zip(theatre_lines, clips, page_nums):
        dsp = EMOTION_DSP_DEFAULTS.get(line.emotion, EMOTION_DSP_DEFAULTS["neutral"])
        by_page.setdefault(int(page_no), []).append(
            {
                "text": line.text,
                "speaker": line.speaker,
                "emotion": line.emotion,
                "stage_direction": line.stage_direction,
                "pitch": line.pitch if line.pitch is not None else dsp["pitch"],
                "volume": line.volume if line.volume is not None else dsp["volume"],
                "rate": line.rate if line.rate is not None else dsp["rate"],
                "audio_path": clip_bytes,
            }
        )

    pages = [{"page": page_no, "sentences": sents} for page_no, sents in by_page.items()]
    used_sfx = enable_sfx and any(clip is not None for clip in ambience_by_emotion.values())

    return PipelineResult(
        pages=pages,
        ambience_by_emotion=ambience_by_emotion,
        used_sfx=used_sfx,
        theatre_script=theatre_script,
    )
