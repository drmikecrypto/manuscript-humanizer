from __future__ import annotations

import httpx

from humanizer.config import DetectorConfig


async def fetch_gptzero_score(text: str, api_key: str) -> float | None:
    """Return GPTZero AI probability 0-100, or None on failure."""
    if not api_key or len(text.strip()) < 50:
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.gptzero.me/v2/predict/text",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json={"document": text[:15000]},
            )
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("documents", [])
            if docs:
                prob = docs[0].get("completely_generated_prob", docs[0].get("average_generated_prob"))
                if prob is not None:
                    return float(prob) * 100
    except Exception:
        return None
    return None


async def fetch_external_score(text: str, config: DetectorConfig) -> float | None:
    if not config.use_external_detector:
        return None
    if config.gptzero_api_key:
        return await fetch_gptzero_score(text, config.gptzero_api_key)
    return None
