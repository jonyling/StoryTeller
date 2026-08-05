import hashlib
import os
import typing

import requests

FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/"


def fetch_ambience_clip(mood: str, api_key: str, cache_dir: str) -> typing.Optional[bytes]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _cache_path(cache_dir, mood)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    search_response = requests.get(
        FREESOUND_SEARCH_URL,
        headers={"Authorization": f"Token {api_key}"},
        params={
            "query": mood,
            "fields": "id,name,previews,license,duration",
            "filter": "duration:[5.0 TO 120.0]",
            "sort": "rating_desc",
            "page_size": 1,
        },
        timeout=15,
    )
    search_response.raise_for_status()
    results = search_response.json().get("results", [])
    if not results:
        return None

    preview_url = results[0]["previews"]["preview-hq-mp3"]
    audio_response = requests.get(
        preview_url,
        headers={"Authorization": f"Token {api_key}"},
        timeout=30,
    )
    audio_response.raise_for_status()
    audio_bytes = audio_response.content

    with open(cache_path, "wb") as f:
        f.write(audio_bytes)
    return audio_bytes


def _cache_path(cache_dir: str, mood: str) -> str:
    key = hashlib.sha1(mood.strip().lower().encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.mp3")
