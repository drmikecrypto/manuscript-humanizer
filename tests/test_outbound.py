
from humanizer.rewriters.bootstrap import is_bootstrap_manuscript
from humanizer.rewriters.outbound import (
    apply_outbound_humanize,
    set_outbound_aggression,
    split_outbound_regions,
    _reflow_to_line_lengths,
)
from humanizer.validators.fidelity import build_quality_report, validate_template_fidelity


MOTIVATION_SNIPPET = (
    "I am applying for the position advertised as PhD computational neuroscience at KI SciLifeLab. "
    "DVM with R/Python practice and first-author pathway/epitope analysis poster (ESHG 2025)."
)

RECOMMENDATION_SNIPPET = (
    "I am writing to provide my strongest recommendation for MohammadJavad Darabi, who completed his "
    "Doctor of Veterinary Medicine (DVM) degree under my direct supervision."
)

RECOMMENDATION_BODY = (
    'I am writing to provide my strongest recommendation for Mohammad Javad Darabi, who completed his '
    'Doctor of Veterinary Medicine (DVM) degree under my direct supervision at the Islamic Azad '
    'University of Shushtar. As his thesis supervisor, I worked closely with Mr. Darabi and can attest '
    'to his exceptional research capabilities and academic excellence. Mr. Darabi achieved outstanding '
    'academic performance with a cumulative grade of 17.1 out of 20, placing him among the top students '
    'in his cohort. I supervised his thesis research titled "Comparing the hypoglycemic effect of '
    'glibenclamide and extracts of nettle and fenugreek plants in diabetic rats." Please feel free to '
    'contact me if you require additional information.'
)

ABSTRACT_BODY = (
    "Pancreatic cancer has high mortality because diagnosis often comes late, tumors grow aggressively, "
    "and treatment resistance is common. Cancer stem cells (CSCs) contribute to progression, metastasis, "
    "and immune evasion. Immunotherapy directed at CSCs is one way to address these problems. "
    "We combined bioinformatics and pathway analysis to identify CSC markers (CD44, CD133, EPCAM) and "
    "their roles in pancreatic cancer. We designed epitope-based vaccine candidates from peptides "
    "predicted to bind HLA with high affinity. Each epitope was checked for immunogenicity, safety, "
    "allergenicity, and population coverage. We identified high-affinity epitopes restricted to "
    "HLA-A02:01 and HLA-A24:02. Validation in experimental models and clinical settings is still required."
)


def setup_function() -> None:
    set_outbound_aggression("conservative", allow_tone_down=False)


def test_bootstrap_detector():
    assert is_bootstrap_manuscript("Diabetes mellitus is one of the most common metabolic disorders.")
    assert not is_bootstrap_manuscript(MOTIVATION_SNIPPET)


def test_outbound_humanize_motivation_preserves_program_and_eshg():
    out = apply_outbound_humanize(MOTIVATION_SNIPPET)
    assert "KI SciLifeLab" in out or "computational neuroscience" in out
    assert "ESHG 2025" in out
    assert "without fuss" not in out.lower()
    assert "happy to discuss" not in out.lower()
    report = build_quality_report(MOTIVATION_SNIPPET, out)
    assert report.similarity >= 0.72
    assert report.length_ratio >= 0.85


def test_outbound_humanize_recommendation_preserves_endorsement_strength():
    out = apply_outbound_humanize(RECOMMENDATION_SNIPPET)
    # Spacing fix only — do not downgrade "strongest"
    assert "strongest recommendation" in out.lower()
    assert "direct supervision" in out.lower()
    assert "mohammad javad darabi" in out.lower()
    assert "without fuss" not in out.lower()


def test_recommendation_body_preserves_grade_title_and_formality():
    letter = (
        "Letter of Recommendation\n"
        "To Whom It May Concern:\n"
        f"{RECOMMENDATION_BODY}\n"
        "Sincerely,\n"
    )
    out = apply_outbound_humanize(letter)
    body = " ".join(
        " ".join(line.strip() for line in region.lines)
        for region in split_outbound_regions(out)
        if not region.frozen
    )
    assert "17.1" in body
    assert "glibenclamide" in body.lower()
    assert "direct supervision" in body.lower() or "thesis supervisor" in body.lower()
    assert "without fuss" not in body.lower()
    assert "happy to" not in body.lower()
    report = build_quality_report(RECOMMENDATION_BODY, body)
    assert report.similarity >= 0.72
    assert 0.85 <= report.length_ratio <= 1.15
    assert not report.invented_numbers


def test_abstract_preserves_markers_hla_and_validation():
    doc = (
        "Technical writing sample\n"
        "First-author conference abstract\n"
        "B ACK GR O UND\n"
        f"{ABSTRACT_BODY}\n"
    )
    out = apply_outbound_humanize(doc)
    body = " ".join(
        " ".join(line.strip() for line in region.lines)
        for region in split_outbound_regions(out)
        if not region.frozen
    )
    for marker in ("CD44", "CD133", "EPCAM", "HLA-A02:01", "HLA-A24:02"):
        assert marker in body
    assert "validation" in body.lower()
    report = build_quality_report(ABSTRACT_BODY, body)
    assert report.similarity >= 0.72
    assert report.claim_strength_delta <= 2.0


def test_template_fidelity_rejects_praise_downgrade():
    original = "He showed exceptional analytical thinking and outstanding academic performance."
    weak = "He showed routine techniques and usable records."
    report = validate_template_fidelity(original, weak, allow_tone_down=False)
    assert not report.passed


def test_template_fidelity_rejects_invented_numbers():
    original = "He finished with years of tutoring and technical reports."
    invented = "He finished with 11 years of tutoring and technical reports."
    report = validate_template_fidelity(original, invented)
    assert not report.passed
    assert report.invented_numbers


def test_reflow_preserves_all_tokens_or_returns_none():
    original = ["short line here", "another line here", "final line"]
    out = _reflow_to_line_lengths("one two three four five six seven eight", original)
    assert out is not None
    assert len(out) == len(original)
    assert " ".join(out).split() == "one two three four five six seven eight".split()


def test_reflow_preserves_tokens_on_overflow():
    original = ["a", "b", "c"]
    words = " ".join(f"word{i}" for i in range(40))
    out = _reflow_to_line_lengths(words, original)
    assert out is not None
    assert " ".join(out).split() == words.split()

def test_pipeline_quality_gate_on_recommendation():
    import asyncio
    from humanizer.config import AppConfig
    from humanizer.pipeline import HumanizerPipeline

    text = (
        "Letter of Recommendation\n"
        "To Whom It May Concern:\n"
        + RECOMMENDATION_BODY
        + "\nSincerely,\n"
    )
    config = AppConfig.load("config.example.toml")
    config.pipeline.max_passes = 4
    config.pipeline.aggression = "conservative"
    config.pipeline.allow_tone_down = False
    config.detector.calibration_offset = 0.0
    result = asyncio.run(HumanizerPipeline(config).run(text))
    assert result.quality_report is not None
    assert result.quality_report.similarity >= 0.72
    assert "strongest recommendation" in result.final.lower()
    assert "17.1" in result.final
    assert "without fuss" not in result.final.lower()
