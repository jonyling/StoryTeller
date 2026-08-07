import hashlib
import os
import typing

import requests

FREESOUND_SEARCH_URL = "https://freesound.org/apiv2/search/"

# Freesound's own "rating" field is empty for most sounds (confirmed empirically —
# every result across five real mood queries came back with rating=None), so
# sort=rating_desc silently does nothing. downloads_desc is a real, populated signal.
# On top of that, a loose text query can still latch onto a tangentially-tagged but
# wrong-vibe result (e.g. "cheerful sparkle" -> a calm/mellow chime track). Pulling
# multiple candidates and preferring whichever one actually looks tagged like ambience
# catches that without needing an extra API call.
_AMBIENCE_TAG_HINTS = {
    "ambient", "ambience", "ambiance", "atmosphere", "atmos", "atmospheric",
    "field-recording", "loop", "background", "background-sound", "room-tone",
    "roomtone", "nature", "outdoor", "indoor", "drone", "soundscape",
}


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
            "fields": "id,name,previews,license,duration,tags",
            "filter": "duration:[5.0 TO 120.0]",
            "sort": "downloads_desc",
            "page_size": 10,
        },
        timeout=15,
    )
    search_response.raise_for_status()
    results = search_response.json().get("results", [])
    if not results:
        return None

    best = _pick_best_result(results)
    preview_url = best["previews"]["preview-hq-mp3"]
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


def _pick_best_result(results):
    """Prefer whichever candidate (already sorted by downloads_desc) has the most
    ambience-indicating tags; ties (including "no candidate has any") keep the
    original downloads_desc order, so this only ever reorders in favor of a clearly
    better-tagged match, never away from a reasonable default."""
    scored = [(_ambience_score(result), position) for position, result in enumerate(results)]
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_position = scored[0][1]
    return results[best_position]


def _ambience_score(result) -> int:
    tags = {tag.lower() for tag in result.get("tags") or []}
    return len(tags & _AMBIENCE_TAG_HINTS)


def _cache_path(cache_dir: str, mood: str) -> str:
    key = hashlib.sha1(mood.strip().lower().encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.mp3")
