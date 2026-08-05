import pytest

from pipeline.config import ConfigError, get_secret


def test_get_secret_reads_from_environment(monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "abc123")
    assert get_secret("MY_TEST_KEY") == "abc123"


def test_get_secret_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    with pytest.raises(ConfigError, match="MISSING_TEST_KEY"):
        get_secret("MISSING_TEST_KEY")
