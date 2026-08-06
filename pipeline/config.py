import os

from pipeline.errors import PipelineError

try:
    import streamlit as st
except ImportError:
    st = None


class ConfigError(PipelineError):
    """Raised when a required secret or config value is missing."""


STORY_PROVIDER = os.environ.get("STORY_PROVIDER", "openai")
# Default: free local XTTS (Havoc path). Set TTS_BACKEND=elevenlabs for paid backup.
TTS_BACKEND = os.environ.get("TTS_BACKEND", "xtts").strip().lower()


def get_secret(name: str, *, required: bool = True) -> str:
    if st is not None:
        try:
            if name in st.secrets and st.secrets[name]:
                return st.secrets[name]
        except Exception:
            pass
    value = os.environ.get(name)
    if not value and required:
        raise ConfigError(f"Missing required secret: {name}")
    return value or ""
