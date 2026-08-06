import glob
import os
import shutil

from pipeline.errors import PipelineError

try:
    import streamlit as st
except ImportError:
    st = None


class ConfigError(PipelineError):
    """Raised when a required secret or config value is missing."""


def ensure_ffmpeg_on_path() -> None:
    """Make ffmpeg resolvable even if the launching shell's PATH is stale.

    winget updates the registry-level PATH, but a shell/IDE terminal opened
    before that (or hosted by a parent process with its own cached
    environment) won't see it until its whole process tree restarts. Rather
    than depend on that, look in ffmpeg's known winget install location and
    patch os.environ["PATH"] directly.
    """
    if shutil.which("ffmpeg") is not None:
        return
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if not local_app_data:
        return
    candidates = glob.glob(
        os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg_*", "ffmpeg-*", "bin")
    )
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "ffmpeg.exe")):
            os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            return


ensure_ffmpeg_on_path()

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
