#!/usr/bin/env python3
"""Download and build lexicon data files for manuscript-humanizer."""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "lexicons"


def download_words_alpha() -> int:
    (ROOT / "general").mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    data = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    words = [w.strip().lower() for w in data.splitlines() if w.strip().isalpha() and len(w.strip()) >= 2]
    out = ROOT / "general" / "words_alpha.txt.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        f.write("\n".join(words))
    return len(words)


def write_synonyms() -> None:
    """Bundled synonym map (WordNet-derived subset for academic paraphrase)."""
    synonyms = {
        "utilize": ["use", "apply", "employ"],
        "utilized": ["used", "applied", "employed"],
        "facilitate": ["enable", "allow", "support"],
        "facilitated": ["enabled", "allowed", "supported"],
        "leverage": ["apply", "use", "draw on"],
        "comprehensive": ["broad", "detailed", "full"],
        "robust": ["stable", "reliable", "solid"],
        "significantly": ["markedly", "clearly", "noticeably"],
        "substantially": ["considerably", "markedly", "clearly"],
        "demonstrate": ["show", "reveal", "indicate"],
        "demonstrated": ["showed", "revealed", "indicated"],
        "indicate": ["suggest", "show", "imply"],
        "indicated": ["suggested", "showed", "implied"],
        "conducted": ["carried out", "performed", "done"],
        "performed": ["carried out", "conducted", "done"],
        "approximately": ["about", "roughly", "around"],
        "various": ["several", "different", "multiple"],
        "commonly": ["often", "typically", "usually"],
        "effective": ["useful", "beneficial", "active"],
        "results": ["findings", "outcomes", "data"],
        "study": ["work", "investigation", "trial"],
        "however": ["yet", "still", "nevertheless"],
        "therefore": ["thus", "hence", "so"],
        "important": ["relevant", "notable", "key"],
        "characterized": ["marked", "defined", "described"],
        "primarily": ["mainly", "chiefly", "largely"],
        "continuous": ["ongoing", "regular", "sustained"],
        "serious": ["severe", "major", "critical"],
        "develop": ["shape", "form", "build"],
        "strategies": ["approaches", "plans", "methods"],
        "therapeutic": ["treatment", "clinical", "medical"],
        "management": ["care", "handling", "control"],
        "quality": ["standard", "level", "grade"],
        "patients": ["subjects", "individuals", "people"],
        "physicians": ["clinicians", "doctors", "practitioners"],
        "scientists": ["researchers", "investigators", "scholars"],
        "necessary": ["required", "needed", "essential"],
        "findings": ["results", "observations", "data"],
        "confirm": ["verify", "validate", "establish"],
        "appropriate": ["suitable", "proper", "adequate"],
        "extracts": ["preparations", "formulations", "treatments"],
        "diabetes": ["glycemic disorder", "hyperglycemia"],
        "glucose": ["blood sugar", "glycemia"],
        "insulin": ["insulin levels", "insulin concentration"],
        "furthermore": ["also", "in addition", "besides"],
        "moreover": ["also", "besides", "in addition"],
        "additionally": ["also", "besides", "further"],
    }
    path = ROOT / "wordnet" / "synonyms.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(synonyms, indent=2), encoding="utf-8")


def main() -> None:
    n = download_words_alpha()
    write_synonyms()
    print(f"Bootstrap complete: {n} words, synonyms written to {ROOT}")


if __name__ == "__main__":
    main()
