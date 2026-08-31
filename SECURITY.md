# Security

## Reporting

If you discover a security issue, please open a private advisory on GitHub or contact the maintainer via their GitHub profile.

## Secrets

Never commit:

- `config.toml` (use `config.example.toml` only)
- `.env` / API keys (`OPENAI_API_KEY`, `GPTZERO_API_KEY`, etc.)
- Private manuscripts under `tests/test_manuscripts/`
- Humanized output under `tests/output/`

The CLI reads API keys from environment variables when the optional LLM or external detector engines are enabled.

## Default mode

The default **one-shot bootstrap** engine runs **offline** — no network calls, no API keys required.
