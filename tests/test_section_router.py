"""Section routing tests."""

from humanizer.rewriters.section_router import (
    build_sentence_section_map,
    normalize_section_name,
    section_for_sentence_index,
)
from humanizer.rewriters.transforms import split_manuscript_sentences
from humanizer.templates.loader import load_section_rules


def test_normalize_section_names():
    assert normalize_section_name("2. Materials and Methods") == "methods"
    assert normalize_section_name("3. Results") == "results"
    assert normalize_section_name("4. Discussion and Conclusion") == "discussion"
    assert normalize_section_name("1. Introduction and Background") == "introduction"


def test_build_sentence_section_map():
    full = (
        "1. Introduction\n"
        "Diabetes mellitus is common in many patient populations worldwide today.\n"
        "2. Methods\n"
        "Male Wistar rats were used in this experimental study design.\n"
        "3. Results\n"
        "Glucose concentrations fell significantly in treated diabetic rats.\n"
    )
    sents = split_manuscript_sentences(full)
    mapping = build_sentence_section_map(full, sents)
    assert len(mapping) == len(sents)
    assert mapping[0] == "introduction"
    assert mapping[1] == "methods"
    assert mapping[2] == "results"


def test_section_rules_do_not_load_all_packs():
    methods_rules = load_section_rules("methods")
    intro_rules = load_section_rules("introduction")
    methods_patterns = {p for p, _ in methods_rules}
    intro_patterns = {p for p, _ in intro_rules}
    assert methods_patterns != intro_patterns


def test_section_for_sentence_index():
    text = (
        "1. Introduction and Background\n"
        "Diabetes mellitus is common in many patient populations worldwide today.\n"
        "2. Materials and Methods\n"
        "Rats were allocated to experimental groups for the study protocol.\n"
    )
    assert section_for_sentence_index(text, 0) == "introduction"
    assert section_for_sentence_index(text, 1) == "methods"
