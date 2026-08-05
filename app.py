# app.py
import os
import tempfile

import streamlit as st
from anthropic import Anthropic
from elevenlabs import ElevenLabs
from openai import OpenAI

from pipeline.audio_utils import validate_duration
from pipeline.config import STORY_PROVIDER, get_secret
from pipeline.errors import PipelineError
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
voice_file = st.file_uploader("Voice sample (1-5 minutes)", type=["wav", "mp3"])
enable_sfx = st.checkbox("Include background sound effects", value=True)
language = st.selectbox("Output language", ["English", "Mandarin"])

if st.button("Generate story", type="primary", disabled=not (pdf_file and voice_file)):
    pdf_bytes = pdf_file.getvalue()
    voice_bytes = voice_file.getvalue()

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
            voice_cloner = ElevenLabsVoiceCloner(elevenlabs_client)
            narration_synthesizer = ElevenLabsNarrationSynthesizer(elevenlabs_client)

            result = run_pipeline(
                pdf_bytes=pdf_bytes,
                voice_bytes=voice_bytes,
                language=language,
                enable_sfx=enable_sfx,
                story_generator=story_generator,
                voice_cloner=voice_cloner,
                narration_synthesizer=narration_synthesizer,
                freesound_api_key=get_secret("FREESOUND_API_KEY") if enable_sfx else "",
                sfx_cache_dir=SFX_CACHE_DIR,
                on_progress=lambda msg: status.update(label=msg),
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
    except PipelineError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Generation failed: {exc}")
