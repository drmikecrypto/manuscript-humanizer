# Calibration workflow (manual ZeroGPT verification)

The built-in **ZeroGPT-proxy scorer** simulates [ZeroGPT](https://www.zerogpt.com/) sentence
highlighting offline. It is not identical to ZeroGPT's DeepAnalyse model — always verify manually.

## Target: 0–5% on ZeroGPT (one-shot bootstrap)

```powershell
pip install -e ".[full]"
mh models download
mh examples/demo_manuscript.md -o humanized_output.md
mh score humanized_output.md --show-spans
```

Paste your output into the [ZeroGPT free checker](https://www.zerogpt.com/).

For private manuscripts, keep files under `tests/test_manuscripts/` (gitignored) and write
output to `tests/output/` (also gitignored).

## Proxy vs ZeroGPT offset

On the demo fixture, ZeroGPT can read **much higher** than the internal proxy
(e.g. proxy ~50 → ZeroGPT ~0% after bootstrap calibration). Treat **live ZeroGPT** as ground truth.

```toml
[detector]
calibration_offset = 8.0

[pipeline]
one_shot = true
target_ai_score = 5.0
max_passes = 1
```

## Quick workflow

1. **Score** the draft:
   ```powershell
   mh score examples/demo_manuscript.md --show-spans
   ```

2. **Humanize** (one-shot bootstrap):
   ```powershell
   mh examples/demo_manuscript.md -o humanized_output.md
   ```

3. **Paste** output into ZeroGPT and note highlighted sentences.

4. **Compare** with `--show-spans`. Add missing rules to `data/templates/academic/`.

5. **Re-run** until ZeroGPT reports your target band.

## What the proxy optimizes

| Signal | Weight | Purpose |
|--------|--------|---------|
| Template phrases (P25–P36) | 25% | Stock academic AI tells |
| Sentence ONNX score | 20% | ML classifier per sentence |
| Legacy stylometrics | 20% | Burstiness, patterns |
| Parallel triple verbs | 25% | "reducing X, increasing Y, decreasing Z" |
| Opener repetition | 10% | Same sentence opener used 2+ times |
| Uniform length | 10% | Sentences near doc median length |

Document score = mean of **top 5 sentence scores** (short texts: worst sentences dominate).

## Adding templates

Edit JSON files in `data/templates/academic/`. Templates use **hard gates only** (numbers,
citations, protected terms) — no overlap gate.

```json
{
  "pattern": "The results indicate that",
  "replacement": "We observed that"
}
```

Run `pytest tests/test_zerogpt_recovery.py` after adding rules.

## Manual success criteria

- ZeroGPT manual check at your target (e.g. **0–5%**)
- Numbers, citations, and section structure preserved
- Academic tone acceptable for your venue
