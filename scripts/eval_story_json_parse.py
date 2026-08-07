"""Basic eval: how often does the vision story-gen LLM call return parseable JSON?

Runs the real OpenAI vision call N times against a fixed demo image and
classifies each response by which parsing layer in
pipeline.story_gen._parse_story_payload actually succeeded:

  clean        -- json.loads() succeeds immediately (no fence, no salvage)
  fenced       -- model wrapped JSON in a ```json fence; strip-then-parse succeeds
  regex_salvage -- direct parse failed; regex-extracted {...} blob parses
  failed       -- nothing parses; would surface as a PipelineError in the app

Costs a handful of real gpt-4o-mini vision calls (~$0.01-0.02 total for
N=10). Requires a real OPENAI_API_KEY in .streamlit/secrets.toml.

Usage: .venv/Scripts/python.exe scripts/eval_story_json_parse.py [N]
"""
import json
import sys
import time
import tomllib
from pathlib import Path

from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.story_gen import _build_prompt, _image_to_jpeg_bytes, _coerce_message_text, _JSON_FENCE
import base64


def load_api_key() -> str:
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    key = secrets.get("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-..."):
        raise SystemExit("OPENAI_API_KEY in .streamlit/secrets.toml is missing or still a placeholder.")
    return key


def classify_parse(text: str) -> tuple[str, dict | None]:
    """Mirror pipeline.story_gen._parse_story_payload's layers, but report which
    one succeeded instead of just returning the payload."""
    text = (text or "").strip()
    fence = _JSON_FENCE.search(text)
    was_fenced = bool(fence)
    if fence:
        text = fence.group(1).strip()

    try:
        payload = json.loads(text)
        return ("fenced" if was_fenced else "clean"), payload
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            return "regex_salvage", payload
        except json.JSONDecodeError:
            pass

    return "failed", None


def run_trial(client: OpenAI, image: Image.Image, language: str = "English") -> dict:
    content = [{"type": "text", "text": _build_prompt(language)}]
    encoded = base64.b64encode(_image_to_jpeg_bytes(image)).decode("utf-8")
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})

    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    elapsed = time.time() - t0

    message = response.choices[0].message if response.choices else None
    text = _coerce_message_text(message) if message else None
    outcome, payload = classify_parse(text or "")
    n_sentences = len(payload.get("sentences", [])) if isinstance(payload, dict) else 0
    return {
        "outcome": outcome,
        "elapsed_s": round(elapsed, 2),
        "n_sentences": n_sentences,
        "raw_preview": (text or "")[:120],
    }


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    image_path = ROOT / "demo files" / "sample photo.jpg"
    image = Image.open(image_path)

    results = []
    for i in range(1, n_trials + 1):
        print(f"[{i}/{n_trials}] calling gpt-4o-mini...", end=" ", flush=True)
        try:
            result = run_trial(client, image)
        except Exception as exc:
            result = {"outcome": "api_error", "elapsed_s": None, "n_sentences": 0, "raw_preview": str(exc)[:120]}
        print(result["outcome"], f"({result['elapsed_s']}s, {result['n_sentences']} sentences)")
        results.append(result)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    print("\n--- Summary ---")
    for outcome in ("clean", "fenced", "regex_salvage", "failed", "api_error"):
        if outcome in counts:
            pct = 100 * counts[outcome] / n_trials
            print(f"{outcome:14s} {counts[outcome]:3d}/{n_trials}  ({pct:.0f}%)")

    parseable = sum(counts.get(k, 0) for k in ("clean", "fenced", "regex_salvage"))
    print(f"\nOverall parse success: {parseable}/{n_trials} ({100 * parseable / n_trials:.0f}%)")

    out_path = ROOT / "docs" / "eval_story_json_parse_results.json"
    out_path.write_text(json.dumps({"n_trials": n_trials, "counts": counts, "results": results}, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
