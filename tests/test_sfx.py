import os
from unittest.mock import MagicMock, patch

from pipeline.sfx import fetch_ambience_clip


def _search_response(has_results=True):
    response = MagicMock()
    response.raise_for_status.return_value = None
    if has_results:
        response.json.return_value = {
            "results": [{
                "id": 1,
                "name": "rain loop",
                "previews": {"preview-hq-mp3": "https://freesound.org/preview/1.mp3"},
                "license": "CC0",
                "duration": 30.0,
            }]
        }
    else:
        response.json.return_value = {"results": []}
    return response


def _audio_response(content=b"fake-mp3-bytes"):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.content = content
    return response


@patch("pipeline.sfx.requests.get")
def test_fetch_ambience_clip_downloads_and_caches(mock_get, tmp_path):
    mock_get.side_effect = [_search_response(), _audio_response()]
    cache_dir = str(tmp_path)

    audio_bytes = fetch_ambience_clip("gentle rain", api_key="fake-key", cache_dir=cache_dir)

    assert audio_bytes == b"fake-mp3-bytes"
    assert mock_get.call_count == 2
    assert len(os.listdir(cache_dir)) == 1


@patch("pipeline.sfx.requests.get")
def test_fetch_ambience_clip_uses_cache_on_second_call(mock_get, tmp_path):
    mock_get.side_effect = [_search_response(), _audio_response()]
    cache_dir = str(tmp_path)

    first = fetch_ambience_clip("gentle rain", api_key="fake-key", cache_dir=cache_dir)
    second = fetch_ambience_clip("gentle rain", api_key="fake-key", cache_dir=cache_dir)

    assert first == second == b"fake-mp3-bytes"
    assert mock_get.call_count == 2  # second call served entirely from cache


@patch("pipeline.sfx.requests.get")
def test_fetch_ambience_clip_returns_none_when_no_results(mock_get, tmp_path):
    mock_get.side_effect = [_search_response(has_results=False)]

    result = fetch_ambience_clip("nonexistent mood", api_key="fake-key", cache_dir=str(tmp_path))

    assert result is None
