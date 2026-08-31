from __future__ import annotations

import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

from humanizer.analyzers.detector import CITATION_RE, NUMBER_RE

_DOMAIN_FILES = {
    "medicine": "medicine.mesh.txt",
    "veterinary": "veterinary.txt",
    "engineering": "engineering.txt",
}

_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*\b")


def _find_lexicon_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "lexicons"
        if candidate.is_dir():
            return candidate
    return here.parent / "data" / "lexicons"


@lru_cache(maxsize=1)
def _load_word_set(root: Path) -> frozenset[str]:
    gz_path = root / "general" / "words_alpha.txt.gz"
    if not gz_path.exists():
        return frozenset()
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        return frozenset(line.strip().lower() for line in f if line.strip())


@lru_cache(maxsize=1)
def _load_synonyms(root: Path) -> dict[str, list[str]]:
    path = root / "wordnet" / "synonyms.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_domain_terms(root: Path, domains: tuple[str, ...]) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for domain in domains:
        fname = _DOMAIN_FILES.get(domain)
        if not fname:
            continue
        path = root / "domains" / fname
        if not path.exists():
            continue
        terms: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.add(line.lower())
            for token in _WORD_RE.findall(line):
                if len(token) >= 3:
                    terms.add(token.lower())
        result[domain] = frozenset(terms)
    return result


@lru_cache(maxsize=1)
def _load_protected(root: Path) -> frozenset[str]:
    path = root / "protected" / "academic_core.txt"
    if not path.exists():
        return frozenset()
    terms: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        terms.add(line.lower())
        for token in _WORD_RE.findall(line):
            terms.add(token.lower())
    return frozenset(terms)


def _word_frequency(word: str) -> float:
    try:
        from wordfreq import word_frequency  # type: ignore[import-untyped]

        return word_frequency(word, "en")
    except ImportError:
        # Rough fallback: shorter common words rank higher
        common = {
            "use", "show", "also", "yet", "still", "care", "work", "data", "plan",
            "need", "help", "guide", "form", "build", "note", "key", "main", "often",
        }
        if word in common:
            return 0.001
        if len(word) <= 4:
            return 0.0005
        return 0.0001


class LexiconService:
    """Full English dictionary + domain lexicons for safe paraphrase."""

    def __init__(
        self,
        *,
        domains: list[str] | None = None,
        protect_domain_terms: bool = True,
        lexicon_root: Path | None = None,
    ) -> None:
        self.root = lexicon_root or _find_lexicon_root()
        self.domains = tuple(domains or ["medicine", "veterinary", "engineering"])
        self.protect_domain_terms = protect_domain_terms
        self._words = _load_word_set(self.root)
        self._synonyms = _load_synonyms(self.root)
        self._domain_terms = _load_domain_terms(self.root, self.domains)
        self._protected_base = _load_protected(self.root)
        all_domain: set[str] = set()
        for terms in self._domain_terms.values():
            all_domain.update(terms)
        self._all_domain = frozenset(all_domain)

    def is_valid_word(self, word: str) -> bool:
        w = word.lower().strip("'")
        return w in self._words

    def domain_of(self, word: str) -> str | None:
        w = word.lower()
        for domain, terms in self._domain_terms.items():
            if w in terms:
                return domain
        return None

    def is_protected(self, word: str, *, text_context: str = "") -> bool:
        w = word.lower()
        if w in self._protected_base:
            return True
        if self.protect_domain_terms and w in self._all_domain:
            return True
        if text_context and re.search(rf"\b{re.escape(word)}\b", text_context, re.IGNORECASE):
            # Protect tokens that appear in numbers/citations context
            pass
        return False

    def extract_protected_from_text(self, text: str) -> set[str]:
        protected: set[str] = set()
        for num in NUMBER_RE.findall(text):
            protected.add(num)
        for cite in CITATION_RE.findall(text):
            protected.add(cite)
        for match in _WORD_RE.finditer(text):
            word = match.group(0)
            if self.is_protected(word):
                protected.add(word)
        return protected

    def get_synonyms(self, word: str, *, limit: int = 5) -> list[str]:
        w = word.lower()
        alts = list(self._synonyms.get(w, []))
        ranked = sorted(
            alts,
            key=lambda a: (_word_frequency(a), -len(a)),
            reverse=True,
        )
        valid: list[str] = []
        for alt in ranked:
            if alt.lower() == w:
                continue
            if not self.is_valid_word(alt.split()[0]):
                continue
            if self.is_protected(alt):
                continue
            valid.append(alt)
            if len(valid) >= limit:
                break
        return valid

    def suggest_swap(self, word: str, *, rng_seed: int | None = None) -> str | None:
        syns = self.get_synonyms(word, limit=8)
        if not syns:
            return None
        if rng_seed is not None:
            idx = rng_seed % len(syns)
            return syns[idx]
        return syns[0]

    def safe_replace_token(self, text: str, old: str, new: str, protected: set[str] | None = None) -> str | None:
        if old.lower() == new.lower():
            return None
        if protected and old in protected:
            return None
        if self.is_protected(old) or self.is_protected(new):
            return None
        pattern = re.compile(rf"\b{re.escape(old)}\b", re.IGNORECASE)

        def replacer(match: re.Match[str]) -> str:
            matched = match.group(0)
            if matched.isupper():
                return new.upper()
            if matched[0].isupper():
                return new.capitalize()
            return new.lower()

        result = pattern.sub(replacer, text, count=1)
        return result if result != text else None
