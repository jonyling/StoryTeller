"""Theatre adaptation: prose sentences → script lines with stage directions + tags."""
from __future__ import annotations

from dataclasses import dataclass

from pipeline.prosody import EMOTION_DSP_DEFAULTS, NARRATOR_DSP, strip_delivery_tags
from pipeline.story_gen import StorySentence

_STAGE_BY_EMOTION = {
    "angry": "sharply, with force",
    "excited": "brightly, leaning forward",
    "sad": "softly, eyes downcast",
    "calm": "gently, unhurried",
    "neutral": "evenly, to the audience",
}


@dataclass
class TheatreLine:
    id: int
    speaker: str
    stage_direction: str
    text: str
    speak_text: str
    emotion: str
    pitch: int
    rate: int
    volume: int


class RuleBasedTheatreAdapter:
    """Deterministic theatre pass (no LLM). Always available / offline-safe."""

    def adapt(self, sentences: list[StorySentence]) -> list[TheatreLine]:
        lines = []
        for i, s in enumerate(sentences):
            emotion = s.emotion or "neutral"
            dsp = dict(EMOTION_DSP_DEFAULTS.get(emotion, EMOTION_DSP_DEFAULTS["neutral"]))
            speaker = s.speaker or "narrator"
            # Third-person narrator: keep dry/clear — pitch-shifting XTTS sounds echoey.
            if speaker.lower() == "narrator":
                dsp = dict(NARRATOR_DSP)
                if emotion == "calm":
                    dsp["rate"] = 2
                elif emotion in {"excited", "angry"}:
                    dsp["rate"] = 4
                    dsp["volume"] = 4
            stage = _STAGE_BY_EMOTION.get(emotion, _STAGE_BY_EMOTION["neutral"])
            if speaker.lower() != "narrator":
                stage = f"as {speaker}; {stage}"
            speak = strip_delivery_tags(s.text)
            lines.append(
                TheatreLine(
                    id=i,
                    speaker=speaker,
                    stage_direction=stage,
                    text=s.text,
                    speak_text=speak or s.text,
                    emotion=emotion,
                    pitch=int(dsp["pitch"]),
                    rate=int(dsp["rate"]),
                    volume=int(dsp["volume"]),
                )
            )
        return lines


class LLMTheatreAdapter:
    """Optional LLM polish of stage directions; falls back to rule-based on failure."""

    def __init__(self, client, model: str = "gpt-4o-mini", *, fallback=None):
        self._client = client
        self._model = model
        self._fallback = fallback or RuleBasedTheatreAdapter()

    def adapt(self, sentences: list[StorySentence]) -> list[TheatreLine]:
        base = self._fallback.adapt(sentences)
        try:
            import json

            payload = [
                {
                    "id": line.id,
                    "speaker": line.speaker,
                    "text": line.text,
                    "emotion": line.emotion,
                }
                for line in base
            ]
            prompt = (
                "You are a children's theatre director. For each line, return a short "
                "stage_direction (physical/vocal business, under 12 words). "
                "Keep speaker/emotion unchanged. Respond as JSON: "
                '{"lines": [{"id": 0, "stage_direction": "..."}, ...]}\n\n'
                f"LINES:\n{json.dumps(payload, ensure_ascii=False)}"
            )
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=800,
            )
            raw = json.loads(response.choices[0].message.content)
            by_id = {int(item["id"]): item.get("stage_direction") for item in raw.get("lines", [])}
            for line in base:
                polished = (by_id.get(line.id) or "").strip()
                if polished:
                    line.stage_direction = polished
            return base
        except Exception:
            return base


def theatre_lines_to_script_doc(lines: list[TheatreLine], *, title: str = "Story", language: str = "English") -> dict:
    """Export theatre-v1 schema (see story-theatre-audio skill reference)."""
    return {
        "schema_version": "theatre-v1",
        "title": title,
        "source": "pdf",
        "language": language,
        "acts": [
            {
                "act": 1,
                "title": "Main",
                "scenes": [
                    {
                        "scene": 1,
                        "title": "Story",
                        "page_num": 1,
                        "setting": "picture-book stage",
                        "ambience_emotion": lines[0].emotion if lines else "neutral",
                        "lines": [
                            {
                                "id": line.id,
                                "speaker": line.speaker,
                                "stage_direction": line.stage_direction,
                                "text": line.text,
                                "speak_text": line.speak_text,
                                "emotion": line.emotion,
                                "pitch": line.pitch,
                                "rate": line.rate,
                                "volume": line.volume,
                            }
                            for line in lines
                        ],
                    }
                ],
            }
        ],
    }
