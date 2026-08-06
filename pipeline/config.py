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
    """Make ffmpeg — and its shared decoding libraries — resolvable even if
    the launching shell's PATH is stale or incomplete.

    winget updates the registry-level PATH, but a shell/IDE terminal opened
    before that (or hosted by a parent process with its own cached
    environment) won't see it until its whole process tree restarts. Rather
    than depend on that, look in known winget install locations and patch
    os.environ["PATH"] directly.

    This covers two independent needs:
    - pydub shells out to an `ffmpeg.exe` — any build satisfies this.
    - torchcodec (used internally by torchaudio/XTTS) dynamically loads
      libavcodec/libavformat/etc. via the OS loader, which only searches
      PATH. That requires a "shared" FFmpeg build that ships those DLLs —
      the default winget package (Gyan.FFmpeg) is statically linked and
      doesn't have them. So the shared-build search always runs, even when
      `ffmpeg` is already resolvable via some other (DLL-less) build.
    """
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        shared_dlls = glob.glob(
            os.path.join(
                local_app_data, "Microsoft", "WinGet", "Packages",
                "*FFmpeg*Shared*", "ffmpeg-*", "bin", "avcodec-*.dll",
            )
        )
        if shared_dlls:
            _prepend_to_path(os.path.dirname(shared_dlls[0]))

    if shutil.which("ffmpeg") is not None:
        return

    if not local_app_data:
        return
    candidates = glob.glob(
        os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg_*", "ffmpeg-*", "bin")
    )
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "ffmpeg.exe")):
            _prepend_to_path(candidate)
            return


def _prepend_to_path(directory: str) -> None:
    current = os.environ.get("PATH", "")
    if directory in current.split(os.pathsep):
        return
    os.environ["PATH"] = directory + os.pathsep + current if current else directory


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
