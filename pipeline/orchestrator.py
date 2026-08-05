import concurrent.futures

from pipeline.audio_utils import slice_audio_by_sentences
from pipeline.sfx import fetch_ambience_clip

_EMOTION_SFX_MOOD = {
    "angry": "thunderstorm",
    "excited": "cheerful sparkle",
    "sad": "gentle rain",
    "calm": "flowing river",
    "neutral": "quiet room tone",
}

_EMOTION_DSP_DEFAULTS = {
    "angry": {"pitch": 4, "volume": 5, "rate": 4},
    "excited": {"pitch": 5, "volume": 4, "rate": 4},
    "sad": {"pitch": 2, "volume": 2, "rate": 2},
    "calm": {"pitch": 2, "volume": 2, "rate": 2},
    "neutral": {"pitch": 3, "volume": 3, "rate": 3},
}


class PipelineResult:
    def __init__(self, pages, ambience_by_emotion, used_sfx: bool):
        self.pages = pages
        self.ambience_by_emotion = ambience_by_emotion
        self.used_sfx = used_sfx


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
) -> PipelineResult:
    def _report(message: str) -> None:
        if on_progress:
            on_progress(message)

    _report("Generating story and cloning voice...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        story_future = executor.submit(story_generator.generate, images, language)
        clone_future = executor.submit(voice_cloner.clone, voice_bytes, "StoryTeller Voice")

        story_result = story_future.result()
        voice_id = clone_future.result()

    sentences = story_result.sentences
    combined_text = " ".join(sentence.text for sentence in sentences)
    distinct_emotions = sorted({sentence.emotion for sentence in sentences})

    _report("Synthesizing narration...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1 + len(distinct_emotions)) as executor:
        narration_future = executor.submit(
            narration_synthesizer.synthesize_with_timestamps, combined_text, voice_id, language
        )
        ambience_futures = {}
        if enable_sfx:
            for emotion in distinct_emotions:
                mood = _EMOTION_SFX_MOOD[emotion]
                ambience_futures[emotion] = executor.submit(
                    fetch_ambience_clip, mood, freesound_api_key, sfx_cache_dir
                )

        narration_audio = narration_future.result()
        ambience_by_emotion = {}
        for emotion, future in ambience_futures.items():
            try:
                ambience_by_emotion[emotion] = future.result()
            except Exception:
                ambience_by_emotion[emotion] = None

    _report("Slicing narration per sentence...")
    clips = slice_audio_by_sentences(
        narration_audio.audio_bytes,
        narration_audio.characters,
        narration_audio.character_start_times_seconds,
        narration_audio.character_end_times_seconds,
        [sentence.text for sentence in sentences],
    )

    page_sentences = []
    for sentence, clip_bytes in zip(sentences, clips):
        dsp = _EMOTION_DSP_DEFAULTS[sentence.emotion]
        page_sentences.append({
            "text": sentence.text,
            "speaker": sentence.speaker,
            "emotion": sentence.emotion,
            "pitch": dsp["pitch"],
            "volume": dsp["volume"],
            "rate": dsp["rate"],
            "audio_path": clip_bytes,
        })

    used_sfx = enable_sfx and any(clip is not None for clip in ambience_by_emotion.values())

    return PipelineResult(
        pages=[{"page": 1, "sentences": page_sentences}],
        ambience_by_emotion=ambience_by_emotion,
        used_sfx=used_sfx,
    )
