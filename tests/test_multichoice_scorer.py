"""
tests/test_multichoice_scorer.py
================================
Unit tests for caem.verification.multichoice_scorer (Branch-C ARC-Challenge
option-text-substitution fix, 2026-04-23).

Covers:
 - Task detection on ARC-style queries
 - Option parsing (one-line and multi-line choice blocks)
 - Letter extraction from CoT answers
 - Full score() path with mocked MiniCheck
 - Backward-compat fallback when env flag off / not a multi-choice task
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from caem.verification.multichoice_scorer import (
    MultichoiceScorer,
    detect_multichoice_task,
    extract_multichoice_letter,
    extract_options,
)


# ======================================================================
# Task detection
# ======================================================================

class TestDetectMultichoiceTask:
    def test_arc_style_detected(self):
        q = (
            "Question: Which best explains how stems transport water to other parts of the plant? "
            "Choices: (A) chlorophyll (B) photosynthesis (C) a system of tubes (D) water to food\n"
            "Answer with just the multiple choice letter."
        )
        assert detect_multichoice_task(q) is True

    def test_arc_variant_without_multiple_choice_phrase(self):
        q = (
            "Question: X? Choices: (A) foo (B) bar (C) baz (D) qux\n"
            "Answer with the letter."
        )
        # Still multi-choice — our regex accepts "with (just)? the (multiple choice)? letter"
        assert detect_multichoice_task(q) is True

    def test_fever_not_detected(self):
        q = "Answer with one of: supports, refutes, not enough info. Claim: X is Y."
        assert detect_multichoice_task(q) is False

    def test_strategyqa_not_detected(self):
        q = "Answer yes or no. Question: Is X true?"
        assert detect_multichoice_task(q) is False

    def test_triviaqa_not_detected(self):
        q = "Who wrote Hamlet?"
        assert detect_multichoice_task(q) is False

    def test_empty_query_not_detected(self):
        assert detect_multichoice_task("") is False
        assert detect_multichoice_task(None) is False

    def test_query_with_choices_but_no_letter_instruction_not_detected(self):
        # Must have BOTH "Choices:" AND "Answer with ... letter".
        q = "Question: X? Choices: (A) foo (B) bar (C) baz (D) qux"
        assert detect_multichoice_task(q) is False


# ======================================================================
# Option parsing
# ======================================================================

class TestExtractOptions:
    def test_one_line_four_choice(self):
        q = (
            "Question: X? "
            "Choices: (A) chlorophyll (B) photosynthesis (C) a system of tubes (D) water to food\n"
            "Answer with just the multiple choice letter."
        )
        opts = extract_options(q)
        assert opts == {
            "A": "chlorophyll",
            "B": "photosynthesis",
            "C": "a system of tubes",
            "D": "water to food",
        }

    def test_multi_line_four_choice(self):
        q = (
            "Question: X?\n"
            "Choices:\n"
            "(A) foo\n"
            "(B) bar\n"
            "(C) baz\n"
            "(D) qux\n"
            "Answer with just the multiple choice letter."
        )
        opts = extract_options(q)
        assert opts == {"A": "foo", "B": "bar", "C": "baz", "D": "qux"}

    def test_five_choice(self):
        q = (
            "Question: Y? "
            "Choices: (A) one (B) two (C) three (D) four (E) five\n"
            "Answer with just the letter."
        )
        opts = extract_options(q)
        assert len(opts) == 5
        assert opts["E"] == "five"

    def test_option_with_internal_punctuation(self):
        q = (
            "Question: Z? "
            "Choices: (A) heat, and light (B) sound, and heat (C) light, and electricity (D) electricity, and sound\n"
            "Answer with just the letter."
        )
        opts = extract_options(q)
        assert opts["A"] == "heat, and light"
        assert opts["D"] == "electricity, and sound"

    def test_empty_query_returns_empty_dict(self):
        assert extract_options("") == {}
        assert extract_options(None) == {}

    def test_no_choices_block_returns_empty_dict(self):
        assert extract_options("Some regular question?") == {}

    def test_whitespace_normalised_in_option_text(self):
        q = (
            "Question: X? "
            "Choices: (A) foo   bar (B) baz  qux (C) alpha (D) beta\n"
            "Answer with just the letter."
        )
        opts = extract_options(q)
        assert opts["A"] == "foo bar"
        assert opts["B"] == "baz qux"


# ======================================================================
# Letter extraction
# ======================================================================

class TestExtractMultichoiceLetter:
    def test_answer_c(self):
        assert extract_multichoice_letter("Reasoning: ... Answer: C") == "C"

    def test_answer_a_lowercase_uppercased(self):
        assert extract_multichoice_letter("Answer: a") == "A"

    def test_answer_with_punctuation(self):
        assert extract_multichoice_letter("Answer: C.") == "C"

    def test_no_answer_line_returns_none(self):
        assert extract_multichoice_letter("Some text without answer") is None

    def test_multi_letter_not_matched(self):
        # We only want A-E single letters. "CD" as a blob has no word-boundary
        # between C and D, so the regex strictly rejects it — caller falls
        # back to legacy rather than guessing which letter the model meant.
        assert extract_multichoice_letter("Answer: CD") is None

    def test_single_letter_then_text_matched(self):
        # "Answer: C with reasoning" — legitimate, letter followed by space.
        assert extract_multichoice_letter("Answer: C with more reasoning") == "C"

    def test_empty_returns_none(self):
        assert extract_multichoice_letter("") is None
        assert extract_multichoice_letter(None) is None


# ======================================================================
# Full scorer path
# ======================================================================

@pytest.fixture
def mock_nli():
    nli = MagicMock()
    nli.batch_entail_prob = MagicMock(return_value=[0.5, 0.5, 0.5])
    return nli


@pytest.fixture(autouse=True)
def reset_env():
    original = os.environ.get("CAEM_MULTICHOICE_SUBSTITUTION")
    os.environ["CAEM_MULTICHOICE_SUBSTITUTION"] = "1"
    yield
    if original is None:
        os.environ.pop("CAEM_MULTICHOICE_SUBSTITUTION", None)
    else:
        os.environ["CAEM_MULTICHOICE_SUBSTITUTION"] = original


class TestMultichoiceScorer:
    def test_substitution_happy_path(self, mock_nli):
        scorer = MultichoiceScorer(mock_nli)
        mock_nli.batch_entail_prob.return_value = [0.85, 0.80, 0.75]
        q = (
            "Question: Which best explains stems transporting water? "
            "Choices: (A) chlorophyll (B) photosynthesis (C) a system of tubes (D) water to food\n"
            "Answer with just the multiple choice letter."
        )
        a = "Reasoning: ... Answer: C"
        passages = ["p1", "p2", "p3"]
        result = scorer.score(q, a, passages)
        assert result is not None
        pg_max, pg_mean = result
        assert pg_max == pytest.approx(0.85)
        assert pg_mean == pytest.approx((0.85 + 0.80 + 0.75) / 3)
        # Verify MiniCheck was called with substituted hypothesis
        call_args = mock_nli.batch_entail_prob.call_args
        pairs = call_args[0][0]
        premise, hypothesis = pairs[0]
        assert "a system of tubes" in hypothesis
        assert hypothesis.startswith("The answer to the question is:")

    def test_last_substitution_recorded(self, mock_nli):
        scorer = MultichoiceScorer(mock_nli)
        q = (
            "Question: X? Choices: (A) foo (B) bar (C) baz (D) qux\n"
            "Answer with just the multiple choice letter."
        )
        scorer.score(q, "Answer: B", ["p1"])
        letter, hypothesis = scorer.last_substitution()
        assert letter == "B"
        assert "bar" in hypothesis

    def test_non_multichoice_returns_none(self, mock_nli):
        scorer = MultichoiceScorer(mock_nli)
        q = "Who wrote Hamlet?"
        result = scorer.score(q, "Shakespeare", ["p1"])
        assert result is None

    def test_fever_query_returns_none(self, mock_nli):
        scorer = MultichoiceScorer(mock_nli)
        q = "Answer with one of: supports, refutes, not enough info. Claim: X."
        result = scorer.score(q, "Answer: supports", ["p1"])
        assert result is None

    def test_no_letter_in_answer_returns_none(self, mock_nli):
        scorer = MultichoiceScorer(mock_nli)
        q = (
            "Question: X? Choices: (A) foo (B) bar (C) baz (D) qux\n"
            "Answer with just the multiple choice letter."
        )
        result = scorer.score(q, "Reasoning without answer line", ["p1"])
        assert result is None

    def test_letter_not_in_options_returns_none(self, mock_nli):
        scorer = MultichoiceScorer(mock_nli)
        q = (
            "Question: X? Choices: (A) foo (B) bar (C) baz (D) qux\n"
            "Answer with just the multiple choice letter."
        )
        # Model said "E" but only A-D are valid options
        result = scorer.score(q, "Answer: E", ["p1"])
        assert result is None

    def test_env_flag_disables_fix(self, mock_nli):
        os.environ["CAEM_MULTICHOICE_SUBSTITUTION"] = "0"
        scorer = MultichoiceScorer(mock_nli)
        q = (
            "Question: X? Choices: (A) foo (B) bar (C) baz (D) qux\n"
            "Answer with just the multiple choice letter."
        )
        result = scorer.score(q, "Answer: B", ["p1"])
        assert result is None

    def test_no_passages_returns_none(self, mock_nli):
        scorer = MultichoiceScorer(mock_nli)
        q = (
            "Question: X? Choices: (A) foo (B) bar (C) baz (D) qux\n"
            "Answer with just the multiple choice letter."
        )
        result = scorer.score(q, "Answer: B", [])
        assert result is None

    def test_no_nli_returns_none(self):
        scorer = MultichoiceScorer(None)
        q = (
            "Question: X? Choices: (A) foo (B) bar (C) baz (D) qux\n"
            "Answer with just the multiple choice letter."
        )
        result = scorer.score(q, "Answer: B", ["p1"])
        assert result is None

    def test_nli_exception_returns_none(self, mock_nli):
        mock_nli.batch_entail_prob.side_effect = RuntimeError("crash")
        scorer = MultichoiceScorer(mock_nli)
        q = (
            "Question: X? Choices: (A) foo (B) bar (C) baz (D) qux\n"
            "Answer with just the multiple choice letter."
        )
        result = scorer.score(q, "Answer: B", ["p1"])
        assert result is None
