from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


def _templates_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "templates" / "academic"
        if candidate.is_dir():
            return candidate
    return here.parent / "data" / "templates" / "academic"


def _outbound_templates_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "templates" / "outbound"
        if candidate.is_dir():
            return candidate
    return here.parent / "data" / "templates" / "outbound"


@lru_cache(maxsize=1)
def load_motivation_zerogpt_rules() -> list[tuple[str, str]]:
    path = _outbound_templates_root() / "motivation_zerogpt.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = [(item["pattern"], item["replacement"]) for item in data.get("rules", []) if item.get("pattern")]
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


@lru_cache(maxsize=1)
def load_recommendation_zerogpt_rules() -> list[tuple[str, str]]:
    path = _outbound_templates_root() / "recommendation_zerogpt.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = [(item["pattern"], item["replacement"]) for item in data.get("rules", []) if item.get("pattern")]
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


@lru_cache(maxsize=1)
def load_abstract_zerogpt_rules() -> list[tuple[str, str]]:
    path = _outbound_templates_root() / "abstracts_zerogpt.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = [(item["pattern"], item["replacement"]) for item in data.get("rules", []) if item.get("pattern")]
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


@lru_cache(maxsize=1)
def load_zerogpt_pass_rules() -> list[tuple[str, str]]:
    path = _templates_root() / "zerogpt_pass.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = [(item["pattern"], item["replacement"]) for item in data.get("rules", []) if item.get("pattern")]
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


@lru_cache(maxsize=1)
def load_academic_rules() -> list[tuple[str, str]]:
    root = _templates_root()
    rules: list[tuple[str, str]] = []
    if not root.exists():
        return rules
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("rules", []):
            pattern = item.get("pattern", "")
            replacement = item.get("replacement", "")
            if pattern and replacement is not None:
                rules.append((pattern, replacement))
    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


_SHARED_VOICE_FILES = frozenset({"human_voice.json"})


@lru_cache(maxsize=8)
def load_section_rules(section: str) -> list[tuple[str, str]]:
    """Load rules for one section pack plus shared human_voice rules."""
    root = _templates_root()
    rules: list[tuple[str, str]] = []
    if not root.exists():
        return rules

    section_file = root / f"{section}.json"
    if section_file.exists():
        data = json.loads(section_file.read_text(encoding="utf-8"))
        for item in data.get("rules", []):
            pattern = item.get("pattern", "")
            replacement = item.get("replacement", "")
            if pattern and replacement is not None:
                rules.append((pattern, replacement))

    for shared in _SHARED_VOICE_FILES:
        path = root / shared
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("rules", []):
            pattern = item.get("pattern", "")
            replacement = item.get("replacement", "")
            if pattern and replacement is not None:
                rules.append((pattern, replacement))

    rules.sort(key=lambda r: len(r[0]), reverse=True)
    return rules


def apply_template_rules(text: str) -> str:
    current = text
    for pattern, repl in load_academic_rules():
        if re.search(pattern, current, flags=re.IGNORECASE):
            current = re.sub(pattern, repl, current, flags=re.IGNORECASE)
    return current
