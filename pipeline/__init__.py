from pipeline import config as _config  # noqa: F401

# Submodules like audio_utils/asr import pydub/torchaudio directly, without
# importing pipeline.config themselves — those libraries probe PATH for
# ffmpeg at their own import time. Importing config here, first, guarantees
# ensure_ffmpeg_on_path() has already patched PATH before any submodule of
# this package gets a chance to trigger that probe.
