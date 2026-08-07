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


@patch("pipeline.sfx.requests.get")
def test_fetch_ambience_clip_requests_multiple_candidates_sorted_by_downloads(mock_get, tmp_path):
    mock_get.side_effect = [_search_response(), _audio_response()]

    fetch_ambience_clip("gentle rain", api_key="fake-key", cache_dir=str(tmp_path))

    search_params = mock_get.call_args_list[0].kwargs["params"]
    assert search_params["sort"] == "downloads_desc"
    assert search_params["page_size"] == 10
    assert "tags" in search_params["fields"]


@patch("pipeline.sfx.requests.get")
def test_fetch_ambience_clip_prefers_result_with_more_ambience_tags(mock_get, tmp_path):
    search_response = MagicMock()
    search_response.raise_for_status.return_value = None
    search_response.json.return_value = {
        "results": [
            {
                "id": 1,
                "name": "engine rev",
                "previews": {"preview-hq-mp3": "https://freesound.org/preview/1.mp3"},
                "license": "CC0",
                "duration": 30.0,
                "tags": ["engine", "car", "race"],
            },
            {
                "id": 2,
                "name": "forest ambience loop",
                "previews": {"preview-hq-mp3": "https://freesound.org/preview/2.mp3"},
                "license": "CC0",
                "duration": 60.0,
                "tags": ["ambience", "ambient", "field-recording", "forest"],
            },
        ]
    }
    mock_get.side_effect = [search_response, _audio_response(b"forest-clip-bytes")]

    audio_bytes = fetch_ambience_clip("forest", api_key="fake-key", cache_dir=str(tmp_path))

    assert audio_bytes == b"forest-clip-bytes"
    downloaded_url = mock_get.call_args_list[1].args[0]
    assert downloaded_url == "https://freesound.org/preview/2.mp3"


@patch("pipeline.sfx.requests.get")
def test_fetch_ambience_clip_falls_back_to_first_result_when_no_tags_match(mock_get, tmp_path):
    search_response = MagicMock()
    search_response.raise_for_status.return_value = None
    search_response.json.return_value = {
        "results": [
            {
                "id": 1,
                "name": "unrelated sound",
                "previews": {"preview-hq-mp3": "https://freesound.org/preview/1.mp3"},
                "license": "CC0",
                "duration": 30.0,
                "tags": ["engine", "car"],
            },
            {
                "id": 2,
                "name": "also unrelated",
                "previews": {"preview-hq-mp3": "https://freesound.org/preview/2.mp3"},
                "license": "CC0",
                "duration": 60.0,
                "tags": ["kitchen", "utensil"],
            },
        ]
    }
    mock_get.side_effect = [search_response, _audio_response(b"first-result-bytes")]

    audio_bytes = fetch_ambience_clip("mood", api_key="fake-key", cache_dir=str(tmp_path))

    assert audio_bytes == b"first-result-bytes"
    downloaded_url = mock_get.call_args_list[1].args[0]
    assert downloaded_url == "https://freesound.org/preview/1.mp3"
