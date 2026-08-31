from humanizer.lexicon.service import LexiconService


def test_lexicon_loads_words():
    lex = LexiconService()
    assert lex.is_valid_word("diabetes")
    assert lex.is_valid_word("algorithm")
    assert not lex.is_valid_word("zzzznotaword123")


def test_domain_protection():
    lex = LexiconService(domains=["medicine"])
    assert lex.domain_of("streptozotocin") == "medicine"
    assert lex.is_protected("streptozotocin")
    assert lex.is_protected("glibenclamide")


def test_synonym_suggestions():
    lex = LexiconService()
    syns = lex.get_synonyms("demonstrate")
    assert syns
    assert "show" in syns or "revealed" in syns or "indicated" in syns


def test_safe_replace_respects_protected():
    lex = LexiconService(domains=["medicine"])
    protected = lex.extract_protected_from_text("Rats received streptozotocin at 50 mg/kg.")
    result = lex.safe_replace_token(
        "Rats received streptozotocin at 50 mg/kg.",
        "streptozotocin",
        "drug",
        protected,
    )
    assert result is None


def test_extract_protected_numbers():
    lex = LexiconService()
    protected = lex.extract_protected_from_text("Body weight was 190-210 g in 8 groups.")
    assert "190-210" in protected or any("190" in p for p in protected)
