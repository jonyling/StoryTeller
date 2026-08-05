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
