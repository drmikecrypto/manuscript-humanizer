# Manuscript Humanizer

Detector-aware academic manuscript humanizer. Rewrites LLM-generated scholarly text to reduce AI-detector scores while preserving meaning, citations, and numbers.

## What makes this different

Most GitHub "humanizers" are prompt files or synonym swappers. This project is a **closed-loop pipeline**:

```
Input → Score (patterns + statistics [+ optional GPTZero])
      → Rewrite (LLM with academic constraints)
      → Validate (numbers, citations, meaning overlap)
      → Re-score → iterate until target or max passes
```

| Layer | What it does |
|-------|-------------|
| **Pattern analyzer** | 24 academic AI-tell regex patterns (stock phrases, AI vocab, structure) |
| **Statistical analyzer** | Burstiness, repetition, sentence-opener diversity |
| **Fidelity validator** | Blocks rewrites that drop numbers, citations, or drift in length |
| **Iterative rewriter** | LLM rewrite with issue-specific feedback each pass |
| **External detector** | Optional GPTZero API blend for real detector signal |

## Quick start

```bash
cd E:\GithubProject\manuscript-humanizer

# Create venv
python -m venv .venv
.venv\Scripts\activate

# Install
pip install -e .

# Configure
copy config.example.toml config.toml
# Edit config.toml — set api_key and model

# Score without rewriting (no API key needed)
mh score -f examples\sample_ai_draft.txt

# Humanize (requires API key)
mh humanize -f examples\sample_ai_draft.txt -o output.txt
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI / compatible APIs |
| `DEEPSEEK_API_KEY` | DeepSeek (set `provider = "deepseek"` in config) |
| `GPTZERO_API_KEY` | Optional real detector feedback |

### DeepSeek example config

```toml
[llm]
provider = "deepseek"
model = "deepseek-chat"
temperature = 0.9

[pipeline]
target_ai_score = 15.0
max_iterations = 5
```

## CLI

```bash
mh score -f manuscript.txt          # analyze only
mh humanize -f manuscript.txt -o out.txt
mh humanize --target 10 --iterations 7 -f draft.txt
echo "text" | mh score
```

Exit codes: `0` = target met, `2` = finished but target not met.

## Configuration

See `config.example.toml` for all options:

- `target_ai_score` — stop when composite score drops below this (0–100)
- `min_meaning_similarity` — lexical overlap floor for meaning preservation
- `chunk_size` — split long manuscripts into chunks
- `use_external_detector` — blend GPTZero score into composite

## Architecture

```
src/humanizer/
├── analyzers/
│   ├── patterns.py      # AI-tell pattern matching
│   ├── statistics.py    # Burstiness / repetition metrics
│   ├── detector.py      # Composite scoring
│   └── external.py      # GPTZero API integration
├── validators/
│   └── fidelity.py      # Number/citation/meaning checks
├── rewriters/
│   └── llm_rewriter.py  # OpenAI-compatible LLM calls
├── prompts/
│   └── academic.py      # Academic rewrite system prompt
├── pipeline.py          # Iterative orchestration
└── cli.py               # Command-line interface
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Honest expectations

Built-in scoring is a **proxy** — it correlates with detector signals but is not Turnitin. Enable `use_external_detector = true` with a GPTZero key for tighter feedback. No tool guarantees 0% on every institutional detector; this maximizes your odds through iteration + fidelity gates.

Always follow your institution's AI disclosure policies.

## License

MIT
