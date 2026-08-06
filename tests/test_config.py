import os

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


def test_ensure_ffmpeg_on_path_noop_when_already_resolvable(monkeypatch):
    monkeypatch.setattr("pipeline.config.shutil.which", lambda name: "C:\\ffmpeg\\ffmpeg.exe")
    monkeypatch.setattr("pipeline.config.glob.glob", lambda pattern: (_ for _ in ()).throw(
        AssertionError("glob should not run when ffmpeg is already resolvable")
    ))
    original_path = "C:\\already\\on\\path"
    monkeypatch.setenv("PATH", original_path)

    ensure_ffmpeg_on_path()

    assert os.environ["PATH"] == original_path


def test_ensure_ffmpeg_on_path_prepends_winget_install_dir(monkeypatch, tmp_path):
    bin_dir = tmp_path / "Gyan.FFmpeg_1.0" / "ffmpeg-7.0" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "ffmpeg.exe").write_bytes(b"")

    monkeypatch.setattr("pipeline.config.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "pipeline.config.glob.glob", lambda pattern: [str(bin_dir)]
    )
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("PATH", "C:\\existing")

    ensure_ffmpeg_on_path()

    assert os.environ["PATH"] == str(bin_dir) + os.pathsep + "C:\\existing"


def test_ensure_ffmpeg_on_path_leaves_path_untouched_when_no_candidate_found(monkeypatch):
    monkeypatch.setattr("pipeline.config.shutil.which", lambda name: None)
    monkeypatch.setattr("pipeline.config.glob.glob", lambda pattern: [])
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\Someone\\AppData\\Local")
    monkeypatch.setenv("PATH", "C:\\existing")

    ensure_ffmpeg_on_path()

    assert os.environ["PATH"] == "C:\\existing"
