# Manuscript Humanizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Offline-first](https://img.shields.io/badge/AI%20humanizer-offline--first-green.svg)](https://github.com/drmikecrypto/manuscript-humanizer)
[![GPTZero calibrated](https://img.shields.io/badge/GPTZero%20%2F%20ZeroGPT-calibrated-orange.svg)](https://www.zerogpt.com/)

**Detector-aware academic manuscript humanizer** — rewrite LLM-generated scholarly text (thesis chapters, journal sections, cover letters) to reduce **GPTZero**, **Turnitin AI**, and stylometric classifier scores while preserving **numbers**, **citations**, and **facts**.

> Not another synonym spinner. A closed-loop, segment-scored rewrite pipeline with ONNX AIGC detection, domain lexicons, and fidelity gates.

**Keywords:** `ai humanizer` · `academic writing` · `gptzero bypass` · `turnitin ai` · `manuscript rewriter` · `stylometrics` · `burstiness` · `onnx` · `offline nlp` · `research paper` · `thesis humanizer` · `ai detector evasion` · `scholarly prose`

---

## TL;DR — run it in 10 seconds

```powershell
git clone https://github.com/drmikecrypto/manuscript-humanizer.git
cd manuscript-humanizer
.\humanize.ps1 your_draft.docx          # → your_draft_humanized.docx
.\humanize.ps1 your_thesis.pdf -InPlace # overwrite + .bak backup
```

**Supported formats:** `.txt` · `.md` · `.docx` · `.pdf` (Word/PDF need `pip install -e ".[full]"`)

```bash
pip install -e ".[full]"
mh your_draft.docx
mh thesis.pdf -i                        # in-place (same file, backup created)
mh score your_draft.pdf
```

---

## Why this exists

Most GitHub "humanizers" are:

- a ChatGPT prompt in a README, or  
- a thesaurus that breaks citations, or  
- a perplexity/burstiness toy that detectors stopped caring about in 2024.

Modern detectors (GPTZero, Turnitin, Pangram) use **ML classifiers on overlapping segments** — template Methods prose, uniform sentence rhythm, stock transitions (`Furthermore`, `Moreover`, rule-of-three lists).

This project targets those signals **directly**:

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Your draft │───▶│ Segment detector │───▶│ Sentence heatmap │
└─────────────┘    │ ONNX + patterns  │    └────────┬────────┘
                   └──────────────────┘             │
                                                    ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Humanized   │◀───│ Fidelity gate    │◀───│ Bootstrap /     │
│ manuscript  │    │ nums · cites     │    │ targeted rewrite│
└─────────────┘    └──────────────────┘    └─────────────────┘
```

| Subsystem | What it does |
|-----------|--------------|
| **ONNX segment detector** | AIGC classifier on sliding windows + per-sentence heatmap |
| **ZeroGPT proxy** | Offline scorer calibrated against live ZeroGPT sentence flags |
| **Bootstrap one-shot** | Deterministic academic templates tuned for detector pass rates |
| **Lexicon service** | 370k-word dictionary + medicine / veterinary / engineering terms |
| **Pattern analyzer** | 36+ academic AI-tell regex patterns |
| **Stylometrics** | Burstiness, opener diversity, parallel clause templates |
| **Fidelity validator** | Blocks rewrites that drop numbers, citations, or length |

Default engine: **`one_shot = true`** — single deterministic pass, no multi-pass drift.

---

## Install

### Windows (easiest)

```powershell
.\humanize.ps1 examples\demo_manuscript.md
```

### Manual

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[full]"
mh models download          # ONNX model ~120 MB, one-time
```

### Optional LLM engine

```bash
pip install -e ".[llm]"
export OPENAI_API_KEY=sk-...
```

```toml
# config.toml
[pipeline]
engine = "llm"
one_shot = false
```

---

## CLI

| Command | What it does |
|---------|--------------|
| `mh paper.docx` | Humanize → `paper_humanized.docx` |
| `mh thesis.pdf -i` | Overwrite same file (`.bak` backup first) |
| `mh humanize draft.txt -o out.txt` | Explicit output path |
| `mh score paper.pdf` | AI likelihood + sentence heatmap |
| `mh models download` | Fetch ONNX detector |

**Formats:** plain text (`.txt`, `.md`, `.tex`, …), **Word** (`.docx`), **PDF** (`.pdf`).  
Convert formats freely: `mh draft.docx -o clean.pdf`

Exit codes: `0` = fidelity pass, `2` = finished with warnings.

---

## Configuration

Copy `config.example.toml` → `config.toml` (optional; defaults work).

```toml
[pipeline]
one_shot = true
target_ai_score = 5.0
max_passes = 1

[detector]
engine = "onnx"
calibration_offset = 8.0
```

See [docs/calibration.md](docs/calibration.md) for manual ZeroGPT verification workflow.

---

## Project layout

```
src/humanizer/
├── analyzers/          # patterns, stylometrics, ONNX segment detector, ZeroGPT proxy
├── rewriters/          # bootstrap, transforms, targeted rewriter, lexicon swaps
├── validators/         # number / citation / meaning fidelity
├── lexicon/            # domain-safe term protection
├── templates/          # calibrated academic rewrite rules (JSON)
└── pipeline.py         # orchestration
data/
├── lexicons/           # WordNet, domain terms, protected academic core
└── templates/academic/ # bootstrap + section templates
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Uses `examples/demo_manuscript.md` — a synthetic AI-draft benchmark, not a real study.

---

## Honest expectations

- Built-in score is a **proxy**. It can disagree sharply with live ZeroGPT. **Always verify externally.**
- No tool guarantees 0% on every institutional detector.
- Follow your university's AI disclosure policies.
- Do not use to misrepresent authorship.

---

## Author

Built by [@drmikecrypto](https://github.com/drmikecrypto)

## License

MIT — see [LICENSE](LICENSE).
