"""
tests/test_entity_head.py
===========================
Unit tests for caem.verification.entity_head (entity_head_consistency
signal — pairwise head-noun agreement across the M=3 self-consistency
chains).

Coverage
--------
  * extract_head_noun: MCQ labels (A-E + lowercase), title-case multi-
    word runs, 4-digit years, alphanumeric fallback, empty input
  * score_entity_head_consistency:
      - 1.0 when all chains agree
      - 0.0 when no pair agrees
      - 1/3 when 2 of 3 chains agree (one agreeing pair / three total)
      - 0.5 (neutral) when fewer than two chains produce extractable
        heads
      - monotonically increasing in fraction-of-agreeing-pairs
  * UnifiedVerifierOutput.entity_head_consistency default = 0.5
  * COMPOSITE_SIGNALS includes entity_head_consistency
  * eval/harness.VERIFIER_FIELDS auto-derives the field
  * Output always in [0, 1]
"""
from __future__ import annotations

import sys


def test_extract_head_noun_mcq_label():
    from caem.verification.entity_head import extract_head_noun
    # Various A-E labelings
    assert extract_head_noun("B") == "b"
    assert extract_head_noun("B.") == "b"
    assert extract_head_noun("(C)") == "c"
    assert extract_head_noun("D)") == "d"
    assert extract_head_noun("E:") == "e"
    # Lowercase preserved
    assert extract_head_noun("a") == "a"


def test_extract_head_noun_title_case_run():
    from caem.verification.entity_head import extract_head_noun
    # Multi-word title-case run — pick the rightmost
    h = extract_head_noun("The president of the United States is Bill Clinton.")
    # Rightmost title-case run is "Bill Clinton" (the dot is stripped by the regex)
    assert "bill clinton" in h or h == "bill clinton"
    # Connector word "of" should keep multi-word entities together
    h2 = extract_head_noun("She studied at the University of Cambridge.")
    assert "cambridge" in h2 or "university" in h2


def test_extract_head_noun_year_numeric():
    from caem.verification.entity_head import extract_head_noun
    # Year as numeric fallback
    h = extract_head_noun("the year was 1969")
    assert h == "1969"


def test_extract_head_noun_alphanumeric_fallback():
    from caem.verification.entity_head import extract_head_noun
    # No title-case, no MCQ label, no year — first non-stopword token
    h = extract_head_noun("the answer is computer")
    assert h == "answer" or h == "computer"


def test_extract_head_noun_empty():
    from caem.verification.entity_head import extract_head_noun
    assert extract_head_noun("") == ""
    assert extract_head_noun("   ") == ""
    # Stopwords-only string returns empty
    assert extract_head_noun("the of and") == ""


def test_score_full_agreement():
    """All chains identify the same head — score = 1.0."""
    from caem.verification.entity_head import score_entity_head_consistency
    chains = [
        "The 42nd president was Bill Clinton.",
        "Bill Clinton served from 1993 to 2001.",
        "The answer is Bill Clinton.",
    ]
    s = score_entity_head_consistency(chains)
    assert s == 1.0, f"expected full agreement = 1.0; got {s}"


def test_score_total_disagreement():
    """No chain pair agrees — score = 0.0."""
    from caem.verification.entity_head import score_entity_head_consistency
    chains = [
        "The answer is Bill Clinton.",
        "The answer is George Bush.",
        "The answer is Barack Obama.",
    ]
    s = score_entity_head_consistency(chains)
    assert s == 0.0, f"expected zero agreement = 0.0; got {s}"


def test_score_partial_agreement():
    """2 of 3 chains agree — score = 1/3 (one agreeing pair out of three)."""
    from caem.verification.entity_head import score_entity_head_consistency
    chains = [
        "The answer is Bill Clinton.",
        "Bill Clinton served from 1993.",
        "The answer is George Bush.",
    ]
    s = score_entity_head_consistency(chains)
    # Pairs: (c0,c1)=agree, (c0,c2)=disagree, (c1,c2)=disagree → 1/3
    assert abs(s - (1.0 / 3.0)) < 1e-9, (
        f"expected 2-of-3 agreement = 1/3; got {s}"
    )


def test_score_neutral_fallback_few_chains():
    """Fewer than two extractable heads → 0.5 neutral fallback."""
    from caem.verification.entity_head import score_entity_head_consistency
    # Empty chain list
    assert score_entity_head_consistency([]) == 0.5
    # Single chain
    assert score_entity_head_consistency(["Bill Clinton served as president."]) == 0.5
    # Two chains but only one extractable head
    assert score_entity_head_consistency([
        "Bill Clinton served as president.",
        "the of and",  # stopwords only — no head
    ]) == 0.5


def test_score_monotone_in_agreement_fraction():
    """As more chains agree, the score should monotonically increase."""
    from caem.verification.entity_head import score_entity_head_consistency
    # 5-chain: 0/5 agree on "X"
    s_0 = score_entity_head_consistency([f"E{i} answer." for i in range(5)])
    # 5-chain: 2/5 agree on "X"
    s_2 = score_entity_head_consistency([
        "X answer.", "X answer.", "Y answer.", "Z answer.", "W answer.",
    ])
    # 5-chain: all agree on "X"
    s_5 = score_entity_head_consistency(["X answer."] * 5)

    assert s_0 < s_2 <= s_5
    assert s_5 == 1.0


def test_unified_verifier_output_carries_entity_head_consistency():
    from caem.verification.verifier import UnifiedVerifierOutput
    vout = UnifiedVerifierOutput(
        u_token=0.7, u_dropout=0.1, u_internal=0.6,
        s_avg=0.7, h_norm=0.2, p_entail=0.6,
        p_ground_max=0.5, p_ground_mean=0.5, p_ground_atomic=0.5,
        p_contra=0.05,
        u_stored=0.6, decision="STORE",
        early_exit_triggered=False, abstained=False,
    )
    # Default neutral
    assert vout.entity_head_consistency == 0.5
    # Override propagates
    vout2 = UnifiedVerifierOutput(
        u_token=0.7, u_dropout=0.1, u_internal=0.6,
        s_avg=0.7, h_norm=0.2, p_entail=0.6,
        p_ground_max=0.5, p_ground_mean=0.5, p_ground_atomic=0.5,
        p_contra=0.05,
        u_stored=0.6, decision="STORE",
        early_exit_triggered=False, abstained=False,
        entity_head_consistency=0.91,
    )
    assert vout2.entity_head_consistency == 0.91


def test_composite_signals_excludes_entity_head_consistency():
    """Phase 1c (2026-05-09): entity_head_consistency is computed and logged
    but NOT fed to the composite. Pooled AUROC 0.598 on v2.1 cycle-0 — barely
    above random."""
    from caem.verification.cal_prob_composite import COMPOSITE_SIGNALS
    assert "entity_head_consistency" not in COMPOSITE_SIGNALS, (
        f"COMPOSITE_SIGNALS unexpectedly contains entity_head_consistency; "
        f"got {COMPOSITE_SIGNALS}"
    )


def test_harness_verifier_fields_auto_picks_up_entity_head_consistency():
    from eval.harness import VERIFIER_FIELDS
    assert "entity_head_consistency" in VERIFIER_FIELDS, (
        f"eval.harness.VERIFIER_FIELDS missing entity_head_consistency; "
        f"got {VERIFIER_FIELDS}"
    )


def test_score_returns_float_in_unit_interval():
    """Sanity: scoring function output is always in [0, 1]."""
    from caem.verification.entity_head import score_entity_head_consistency
    cases = [
        [],
        ["only one chain"],
        ["A.", "B.", "C."],
        ["X X", "X X", "X X"],
        ["", "", ""],
        ["foo bar baz", "qux quux", "corge"],
    ]
    for chains in cases:
        s = score_entity_head_consistency(chains)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0, f"out-of-range output {s} for {chains}"


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:
                failures += 1
                print(f"  ERR   {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
    print(f"\nAll tests passed.")
