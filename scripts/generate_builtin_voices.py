"""Regenerate built-in narrator WAVs under assets/voices/ (requires edge-tts + ffmpeg)."""
from __future__ import annotations

import asyncio
from pathlib import Path

SAMPLES = {
    "warm": (
        "en-US-GuyNeural",
        "Once upon a time, in a quiet village by the hills, there lived a small dragon who loved warm sunlight and gentle stories. "
        "Every evening, the dragon listened carefully, and the soft voice of a kind grandparent made the night feel safe.",
    ),
    "bright": (
        "en-US-JennyNeural",
        "Hello there, friends! Are you ready for an adventure? Today we will meet brave little Ember, who flaps bright wings and laughs at sparkly clouds. "
        "Come along, let's hurry to the glowing valley together!",
    ),
    "gentle": (
        "en-US-AriaNeural",
        "Close your eyes for a moment. Breathe slowly. The river whispers past the reeds, and moonlight rests on quiet leaves. "
        "I will tell you a gentle bedtime story, soft and calm, until you feel peaceful and ready for sleep.",
    ),
}


async def main() -> None:
    import edge_tts
    from pydub import AudioSegment

    out_dir = Path(__file__).resolve().parents[1] / "assets" / "voices"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (voice, text) in SAMPLES.items():
        mp3 = out_dir / f"{name}.mp3"
        wav = out_dir / f"{name}.wav"
        await edge_tts.Communicate(text, voice).save(str(mp3))
        seg = AudioSegment.from_file(mp3).set_channels(1).set_frame_rate(22050)
        seg.export(wav, format="wav")
        mp3.unlink(missing_ok=True)
        print(f"wrote {wav} ({len(seg)/1000:.1f}s)")


if __name__ == "__main__":
    asyncio.run(main())
