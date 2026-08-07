import os
import subprocess
import sys

import pytest

from pipeline.config import ConfigError, ensure_ffmpeg_on_path, get_secret
from pipeline.errors import PipelineError


def test_get_secret_reads_from_environment(monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "abc123")
    assert get_secret("MY_TEST_KEY") == "abc123"


def test_get_secret_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    with pytest.raises(ConfigError, match="MISSING_TEST_KEY"):
        get_secret("MISSING_TEST_KEY")


def test_config_error_is_a_pipeline_error():
    assert issubclass(ConfigError, PipelineError)


def _glob_by_marker(shared_result=None, static_result=None):
    """Route glob.glob calls by which search they belong to, based on the
    pattern's shape: the shared-DLL search ends in an avcodec-*.dll glob,
    the static-build search ends in a bare bin/ directory glob."""

    def fake_glob(pattern):
        if "avcodec" in pattern:
            return shared_result or []
        return static_result or []

    return fake_glob


def test_ensure_ffmpeg_on_path_prepends_shared_dll_dir_even_if_already_resolvable(monkeypatch, tmp_path):
    # torchcodec needs the shared build's DLLs on PATH regardless of whether
    # some other (static) ffmpeg.exe is already resolvable — the two needs
    # are independent, so this search must not be gated on shutil.which.
    dll_dir = tmp_path / "BtbN.FFmpeg.GPL.Shared.8.1" / "ffmpeg-8.1" / "bin"
    dll_dir.mkdir(parents=True)
    dll_path = dll_dir / "avcodec-62.dll"
    dll_path.write_bytes(b"")

    monkeypatch.setattr("pipeline.config.shutil.which", lambda name: "C:\\ffmpeg\\ffmpeg.exe")
    monkeypatch.setattr(
        "pipeline.config.glob.glob",
        _glob_by_marker(shared_result=[str(dll_path)]),
    )
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("PATH", "C:\\existing")

    ensure_ffmpeg_on_path()

    assert os.environ["PATH"] == str(dll_dir) + os.pathsep + "C:\\existing"


def test_ensure_ffmpeg_on_path_noop_when_nothing_found_and_already_resolvable(monkeypatch):
    monkeypatch.setattr("pipeline.config.shutil.which", lambda name: "C:\\ffmpeg\\ffmpeg.exe")
    monkeypatch.setattr("pipeline.config.glob.glob", _glob_by_marker())
    original_path = "C:\\already\\on\\path"
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\Someone\\AppData\\Local")
    monkeypatch.setenv("PATH", original_path)

    ensure_ffmpeg_on_path()

    assert os.environ["PATH"] == original_path


def test_ensure_ffmpeg_on_path_prepends_winget_static_build_dir_when_not_resolvable(monkeypatch, tmp_path):
    bin_dir = tmp_path / "Gyan.FFmpeg_1.0" / "ffmpeg-7.0" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "ffmpeg.exe").write_bytes(b"")

    monkeypatch.setattr("pipeline.config.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "pipeline.config.glob.glob",
        _glob_by_marker(static_result=[str(bin_dir)]),
    )
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("PATH", "C:\\existing")

    ensure_ffmpeg_on_path()

    assert os.environ["PATH"] == str(bin_dir) + os.pathsep + "C:\\existing"


def test_ensure_ffmpeg_on_path_leaves_path_untouched_when_no_candidate_found(monkeypatch):
    monkeypatch.setattr("pipeline.config.shutil.which", lambda name: None)
    monkeypatch.setattr("pipeline.config.glob.glob", _glob_by_marker())
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\Someone\\AppData\\Local")
    monkeypatch.setenv("PATH", "C:\\existing")

    ensure_ffmpeg_on_path()

    assert os.environ["PATH"] == "C:\\existing"


def test_ensure_ffmpeg_on_path_does_not_duplicate_entry_on_repeated_calls(monkeypatch, tmp_path):
    dll_dir = tmp_path / "BtbN.FFmpeg.GPL.Shared.8.1" / "ffmpeg-8.1" / "bin"
    dll_dir.mkdir(parents=True)
    dll_path = dll_dir / "avcodec-62.dll"
    dll_path.write_bytes(b"")

    monkeypatch.setattr("pipeline.config.shutil.which", lambda name: "C:\\ffmpeg\\ffmpeg.exe")
    monkeypatch.setattr(
        "pipeline.config.glob.glob",
        _glob_by_marker(shared_result=[str(dll_path)]),
    )
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("PATH", "C:\\existing")

    ensure_ffmpeg_on_path()
    ensure_ffmpeg_on_path()

    assert os.environ["PATH"] == str(dll_dir) + os.pathsep + "C:\\existing"


def test_importing_any_pipeline_submodule_patches_ffmpeg_path_first(tmp_path):
    """pipeline/audio_utils.py imports pydub directly and never imports
    pipeline.config — pydub probes for ffmpeg at its own import time, so on
    a fresh interpreter that probe would run before our PATH patch unless
    pipeline/__init__.py guarantees the patch happens first for every
    submodule import. Runs in a subprocess because import order is only
    observable on a fresh interpreter — modules stay cached across tests
    in-process."""
    dll_dir = (
        tmp_path / "Microsoft" / "WinGet" / "Packages"
        / "BtbN.FFmpeg.GPL.Shared.8.1" / "ffmpeg-8.1" / "bin"
    )
    dll_dir.mkdir(parents=True)
    (dll_dir / "avcodec-62.dll").write_bytes(b"")

    env = dict(os.environ)
    env["LOCALAPPDATA"] = str(tmp_path)
    env["PATH"] = "C:\\existing"

    script = "import os; import pipeline.audio_utils; print(os.environ['PATH'])"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert str(dll_dir) in result.stdout
