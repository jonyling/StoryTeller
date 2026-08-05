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
