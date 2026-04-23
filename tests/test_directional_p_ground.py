"""
tests/test_directional_p_ground.py
==================================
Unit tests for caem.verification.directional_p_ground (Branch-C
refutation-bias fix, 2026-04-23).

Covers:
 - Task detection (FEVER / StrategyQA / None)
 - Claim + label extraction
 - Directional scoring on all three labels (supports / refutes / NEI)
 - Backward compatibility when env flag is off
 - Fallback to legacy when claim or label can't be parsed
 - NEI bidirectional certainty-of-uncertainty formula

All tests use mocked nli_judge and Negator so no models are loaded.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from caem.verification.directional_p_ground import (
    DirectionalScorer,
    detect_task,
    extract_fever_claim,
    extract_fever_label,
    extract_strategyqa_label,
    extract_strategyqa_question,
)
from caem.verification.negation import NegationResult


# ======================================================================
# Task detection
# ======================================================================

class TestDetectTask:
    def test_fever_task_detected(self):
        q = "Answer with one of: supports, refutes, not enough info. Claim: X is Y"
        assert detect_task(q) == "fever"

    def test_strategyqa_task_detected(self):
        q = "Answer yes or no. Question: Is the sky blue?"
        assert detect_task(q) == "strategyqa"

    def test_triviaqa_not_detected(self):
        q = "Who wrote Hamlet?"
        assert detect_task(q) is None

    def test_arc_not_detected(self):
        q = "Question: Which of... Choices: (A)... (B)... Answer with just the multiple choice letter."
        assert detect_task(q) is None

    def test_empty_query_not_detected(self):
        assert detect_task("") is None
        assert detect_task(None) is None


# ======================================================================
# Extraction helpers
# ======================================================================

class TestFeverExtraction:
    def test_claim_extracted(self):
        q = "Answer with one of: supports, refutes, not enough info. Claim: Reign Over Me was directed by Binder."
        assert extract_fever_claim(q) == "Reign Over Me was directed by Binder."

    def test_label_supports(self):
        a = "Reasoning: ... Answer: supports"
        assert extract_fever_label(a) == "supports"

    def test_label_refutes(self):
        a = "Reasoning: ... Answer: refutes"
        assert extract_fever_label(a) == "refutes"

    def test_label_nei_variants(self):
        assert extract_fever_label("Answer: not enough info") == "not enough info"
        assert extract_fever_label("Answer: not  enough  info") == "not enough info"

    def test_no_label_returns_none(self):
        assert extract_fever_label("Some text without an answer line") is None

    def test_empty_inputs(self):
        assert extract_fever_claim("") is None
        assert extract_fever_label("") is None


class TestStrategyQAExtraction:
    def test_question_extracted(self):
        q = "Answer yes or no. Question: Can food be cooked in the microwave?"
        assert extract_strategyqa_question(q) == "Can food be cooked in the microwave?"

    def test_yes_label(self):
        assert extract_strategyqa_label("Answer: yes") == "yes"

    def test_no_label(self):
        assert extract_strategyqa_label("Answer: no") == "no"

    def test_case_insensitive(self):
        assert extract_strategyqa_label("ANSWER: YES") == "yes"


# ======================================================================
# DirectionalScorer - core logic
# ======================================================================

@pytest.fixture
def mock_nli():
    """MiniCheck mock: returns a configurable list of entailment probs."""
    nli = MagicMock()
    nli.batch_entail_prob = MagicMock(return_value=[0.5, 0.5, 0.5])
    return nli


@pytest.fixture
def mock_negator():
    """Negator mock: returns a rule-based negation."""
    n = MagicMock()

    def negate_fn(claim):
        # Simple synthetic negation for tests
        if "not " in claim:
            return NegationResult(claim.replace("not ", ""), "rule")
        return NegationResult(
            claim.replace(" is ", " is not ").replace(" are ", " are not ")
                 .replace(" was ", " was not "),
            "rule",
        )

    n.negate.side_effect = negate_fn
    return n


@pytest.fixture(autouse=True)
def reset_env():
    """Ensure CAEM_DIRECTIONAL_P_GROUND default-on unless test overrides."""
    original = os.environ.get("CAEM_DIRECTIONAL_P_GROUND")
    os.environ["CAEM_DIRECTIONAL_P_GROUND"] = "1"
    yield
    if original is None:
        os.environ.pop("CAEM_DIRECTIONAL_P_GROUND", None)
    else:
        os.environ["CAEM_DIRECTIONAL_P_GROUND"] = original


class TestDirectionalScorer:
    def test_supports_uses_claim_unchanged(self, mock_nli, mock_negator):
        scorer = DirectionalScorer(mock_nli, mock_negator)
        mock_nli.batch_entail_prob.return_value = [0.85, 0.80, 0.75]
        q = "Answer with one of: supports, refutes, not enough info. Claim: X is Y."
        a = "Reasoning: ... Answer: supports"
        passages = ["p1", "p2", "p3"]
        result = scorer.score(q, a, passages)
        assert result is not None
        pg_max, pg_mean = result
        assert pg_max == pytest.approx(0.85)
        assert pg_mean == pytest.approx((0.85 + 0.80 + 0.75) / 3)
        # Hypothesis should be the claim itself (no negation called)
        mock_negator.negate.assert_not_called()

    def test_refutes_uses_negated_claim(self, mock_nli, mock_negator):
        scorer = DirectionalScorer(mock_nli, mock_negator)
        mock_nli.batch_entail_prob.return_value = [0.90, 0.85]
        q = "Answer with one of: supports, refutes, not enough info. Claim: X is Y."
        a = "Reasoning: ... Answer: refutes"
        passages = ["p1", "p2"]
        result = scorer.score(q, a, passages)
        assert result is not None
        pg_max, pg_mean = result
        assert pg_max == pytest.approx(0.90)
        # Negator should have been called on the claim
        mock_negator.negate.assert_called_once()

    def test_nei_uses_bidirectional_certainty_high_ambiguity(self, mock_nli, mock_negator):
        """Both directions near 0.5 → genuine NEI → score near 1.0."""
        scorer = DirectionalScorer(mock_nli, mock_negator)
        # Two calls to MiniCheck (positive then negative), both returning ~0.5
        mock_nli.batch_entail_prob.side_effect = [
            [0.5, 0.5, 0.5],   # positive direction
            [0.5, 0.5, 0.5],   # negative direction
        ]
        q = "Answer with one of: supports, refutes, not enough info. Claim: X is Y."
        a = "Reasoning: ... Answer: not enough info"
        passages = ["p1", "p2", "p3"]
        result = scorer.score(q, a, passages)
        assert result is not None
        _, pg_mean = result
        # Both directions near 0.5 → cert = 0 → score = 1 - 2*0 = 1.0
        assert pg_mean == pytest.approx(1.0, abs=0.05)

    def test_nei_low_score_on_informative_passages(self, mock_nli, mock_negator):
        """One direction strongly supported → defensive NEI → low score."""
        scorer = DirectionalScorer(mock_nli, mock_negator)
        mock_nli.batch_entail_prob.side_effect = [
            [0.95, 0.95, 0.95],   # positive direction (strong support)
            [0.05, 0.05, 0.05],   # negative direction (strong rejection)
        ]
        q = "Answer with one of: supports, refutes, not enough info. Claim: X is Y."
        a = "Reasoning: ... Answer: not enough info"
        passages = ["p1", "p2", "p3"]
        result = scorer.score(q, a, passages)
        assert result is not None
        _, pg_mean = result
        # One direction strongly confident → cert = 0.45 → score = 1 - 2*0.45 = 0.1
        assert pg_mean < 0.2

    def test_strategyqa_yes_uses_question(self, mock_nli, mock_negator):
        scorer = DirectionalScorer(mock_nli, mock_negator)
        mock_nli.batch_entail_prob.return_value = [0.80]
        q = "Answer yes or no. Question: Is X true?"
        a = "Answer: yes"
        passages = ["p1"]
        result = scorer.score(q, a, passages)
        assert result is not None
        # Yes → use question as-is, no negation
        mock_negator.negate.assert_not_called()

    def test_strategyqa_no_calls_negator(self, mock_nli, mock_negator):
        scorer = DirectionalScorer(mock_nli, mock_negator)
        mock_nli.batch_entail_prob.return_value = [0.80]
        q = "Answer yes or no. Question: Is X true?"
        a = "Answer: no"
        passages = ["p1"]
        result = scorer.score(q, a, passages)
        assert result is not None
        mock_negator.negate.assert_called_once()


# ======================================================================
# Fallback / compatibility
# ======================================================================

class TestFallback:
    def test_triviaqa_returns_none_preserves_legacy(self, mock_nli, mock_negator):
        scorer = DirectionalScorer(mock_nli, mock_negator)
        q = "Who wrote Hamlet?"
        a = "William Shakespeare"
        passages = ["p1"]
        assert scorer.score(q, a, passages) is None

    def test_no_passages_returns_none(self, mock_nli, mock_negator):
        scorer = DirectionalScorer(mock_nli, mock_negator)
        q = "Answer with one of: supports, refutes, not enough info. Claim: X."
        a = "Answer: supports"
        assert scorer.score(q, a, []) is None

    def test_no_label_in_answer_returns_none(self, mock_nli, mock_negator):
        scorer = DirectionalScorer(mock_nli, mock_negator)
        q = "Answer with one of: supports, refutes, not enough info. Claim: X."
        a = "Reasoning: ... (no answer line)"
        assert scorer.score(q, a, ["p1"]) is None

    def test_env_flag_disables_fix(self, mock_nli, mock_negator):
        os.environ["CAEM_DIRECTIONAL_P_GROUND"] = "0"
        scorer = DirectionalScorer(mock_nli, mock_negator)
        q = "Answer with one of: supports, refutes, not enough info. Claim: X."
        a = "Answer: refutes"
        assert scorer.score(q, a, ["p1"]) is None

    def test_failed_negation_returns_none(self, mock_nli):
        # Negator returns failed result → scorer returns None
        bad_negator = MagicMock()
        bad_negator.negate.return_value = NegationResult(None, "failed")
        scorer = DirectionalScorer(mock_nli, bad_negator)
        q = "Answer with one of: supports, refutes, not enough info. Claim: X."
        a = "Answer: refutes"
        assert scorer.score(q, a, ["p1"]) is None
