# StoryTeller — Measured Performance & Eval Results

Real numbers measured on this machine (CPU-only, no CUDA GPU — `torch 2.13.0+cpu`). Not estimates.

---

## 1. Story-generation JSON-parse eval

`scripts/eval_story_json_parse.py` — 10 live calls to `gpt-4o-mini` (vision), same demo photo each time. Classifies each response by which layer of `pipeline/story_gen.py::_parse_story_payload` succeeded: `clean` (parsed immediately) / `fenced` (needed markdown-fence stripping) / `regex_salvage` (needed regex `{...}` extraction) / `failed`.

| Metric | Value |
|---|---|
| JSON parse success rate | **10/10 (100%)** |
| Fallback layers triggered | 0 |
| Avg latency (story-gen LLM call only) | **4.9s** (range 4.1s–7.2s) |
| Sentences generated per story | 8–12 (target: "under 200 words") |

Raw per-trial data: `docs/eval_story_json_parse_results.json`.

**Caveat to state honestly in the report:** 100% at N=10 mostly validates the `response_format=json_object` constraint on OpenAI specifically — it's close to a hard guarantee there. The fence-strip/regex-salvage fallback code exists for other providers/gateways (Grok, Gemini, Claude routes) that don't honor `response_format` the same way, especially combined with vision input — that path wasn't stress-tested here.

---

## 2. XTTS synthesis latency (CPU, no GPU)

Measured directly against `pipeline/xtts_backend.py::XTTSNarrationSynthesizer` using `assets/voices/warm.mp3` as the reference and 4 representative sentences (36–57 chars each).

| Step | Time |
|---|---|
| XTTS model load (once per app session) | **102.1s** (~1.7 min) |
| Per-sentence synthesis | **17.2s – 29.6s** (avg **23.1s**) |

### What this means for a full generation

```
First generation this session  ≈ 100s (model load)
                                + 5s   (story-gen LLM call)
                                + 8-12 × 23s (per-sentence XTTS synthesis)
                                ≈ 290s – 380s
                                ≈ 4.8 – 6.4 minutes
```

Once the model is already loaded (warm session), a full 8–12 sentence story still takes **~3–4.5 minutes** of synthesis alone.

---

## 3. Demo-day implication (10-minute live slot)

The rubric requires a pre-recorded fallback video "in case of live GPU/inference hiccups" — given the numbers above, treat it as load-bearing, not optional:

- **Warm the model before going up.** Run one throwaway generation beforehand so the ~102s cold-load doesn't happen live.
- **Even warm, a full story is ~3–4.5 minutes live.** That alone can eat half the demo slot.
- **Mitigation options:**
  - Use a short custom PDF/photo (2–4 sentences) for anything shown live, to keep it under ~1 minute.
  - Show a pre-recorded full-length run for the "real" result, and narrate over it.
  - If a GPU is available on your desktop or a lab machine, XTTS auto-selects `cuda` when present (`xtts_backend.py`) — per-sentence synthesis would very likely drop to low single-digit seconds, making a live full-length demo actually feasible. Worth testing there before deciding the demo format.

---

## 4. Open gap: no formal generation-quality accuracy metric

There is currently no WER/BLEU/MOS-style score computed anywhere in the codebase — no eval compares generated story/audio against ground truth. If asked "what's your model's accuracy," the honest answer is that JSON-parse success rate (§1) is the only quantified proxy tracked so far.
