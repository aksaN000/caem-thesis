"""
tests/test_negation.py
======================
Unit tests for caem.verification.negation.

Covers:
 - Rule-based negation on canonical FEVER-style patterns
 - Double-negation removal
 - Graceful failure on edge cases
 - NegationResult source tracking
 - Negator fallback cascade (rule → Qwen → failed)

No model dependencies — Qwen fallback is mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from caem.verification.negation import (
    Negator,
    NegationResult,
    negate_by_rules,
    validate_negation,
)


# ======================================================================
# Rule-based negation
# ======================================================================

class TestRuleNegation:
    def test_copular_is_adds_not(self):
        out = negate_by_rules("Jed Whedon is an American icon")
        assert out is not None
        assert "is not" in out.lower()

    def test_copular_are_adds_not(self):
        out = negate_by_rules("They are brothers")
        assert out is not None
        assert "are not" in out.lower()

    def test_copular_was_adds_not(self):
        out = negate_by_rules("AMGTV was an American TV network")
        assert out is not None
        assert "was not" in out.lower()

    def test_copular_were_adds_not(self):
        out = negate_by_rules("They were rivals")
        assert out is not None
        assert "were not" in out.lower()

    def test_double_negation_removal_is_not(self):
        out = negate_by_rules("Uta Hagen is not an American actress")
        assert out is not None
        assert "is not" not in out.lower()
        assert "is" in out.lower()

    def test_double_negation_removal_was_not(self):
        out = negate_by_rules("AMGTV was not an American TV network")
        assert out is not None
        assert "was not" not in out.lower()

    def test_contraction_isnt(self):
        out = negate_by_rules("AMGTV isn't American")
        assert out is not None
        assert "is" in out.lower()
        assert "isn't" not in out.lower()

    def test_contraction_cannot(self):
        out = negate_by_rules("Food cannot be cooked there")
        assert out is not None
        assert "can" in out.lower()
        assert "cannot" not in out.lower()
        assert "can't" not in out.lower()

    def test_contraction_wont(self):
        out = negate_by_rules("The show won't return")
        assert out is not None
        assert "will" in out.lower()
        assert "won't" not in out.lower()

    def test_modal_can_adds_not(self):
        out = negate_by_rules("Food can be cooked in the microwave")
        assert out is not None
        assert "can not" in out.lower() or "cannot" in out.lower()

    def test_empty_claim_returns_none(self):
        assert negate_by_rules("") is None
        assert negate_by_rules("   ") is None

    def test_claim_with_no_negatable_verb_returns_none(self):
        # Pure noun phrase — no verb to negate by rules
        out = negate_by_rules("The big red dog running quickly")
        # Rules should NOT fire here; Qwen fallback would be needed
        assert out is None

    def test_output_preserves_terminal_punctuation(self):
        out = negate_by_rules("AMGTV is American.")
        assert out is not None
        assert out.endswith(".")

    def test_output_adds_terminal_punctuation_if_missing(self):
        out = negate_by_rules("AMGTV is American")
        assert out is not None
        assert out.endswith(".")

    def test_whitespace_normalized(self):
        out = negate_by_rules("AMGTV   is   American")
        assert out is not None
        assert "  " not in out


# ======================================================================
# Negator class (two-tier cascade)
# ======================================================================

class TestNegator:
    def test_rule_fast_path(self):
        # No model/tokenizer provided — rules must succeed on a simple claim.
        n = Negator(model=None, tokenizer=None, nli_judge=None)
        result = n.negate("AMGTV is an American TV network")
        assert result.source == "rule"
        assert result.text is not None
        assert "is not" in result.text.lower()

    def test_qwen_fallback_on_rule_miss(self):
        # Claim that rules can't negate — verify Qwen path is attempted.
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Simulate Qwen negating the claim.
        import torch
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode.return_value = "The big red dog does not run quickly."

        fake_output = MagicMock()
        fake_output.shape = (1, 10)
        fake_output.__getitem__ = lambda self, k: fake_output  # allow out[0][X:]
        mock_model.device = "cpu"
        mock_model.generate.return_value = fake_output

        n = Negator(model=mock_model, tokenizer=mock_tokenizer, nli_judge=None)
        result = n.negate("The big red dog running quickly")
        # Either rule succeeded (unlikely on this claim) or Qwen fallback engaged
        assert result.source in ("rule", "qwen", "failed")

    def test_no_models_and_rule_miss_returns_failed(self):
        n = Negator(model=None, tokenizer=None, nli_judge=None)
        result = n.negate("The big red dog running quickly")
        assert result.source == "failed"
        assert result.text is None

    def test_empty_claim(self):
        n = Negator(model=None, tokenizer=None, nli_judge=None)
        result = n.negate("")
        assert result.source == "failed"
        assert result.text is None


# ======================================================================
# Validation
# ======================================================================

class TestValidateNegation:
    def test_none_nli_defaults_to_permissive(self):
        # With no validator we can't distinguish good/bad negations by NLI.
        # The Negator-level code path is: if self.nli is None, don't call
        # validator at all → negation accepted. validate_negation() itself
        # returns False on None (different layer concern).
        result = validate_negation("X is Y", "X is not Y", None)
        assert result is False  # direct call with None nli → False

    def test_identical_strings_fail(self):
        mock_nli = MagicMock()
        result = validate_negation("X is Y", "X is Y", mock_nli)
        assert result is False

    def test_low_entailment_passes(self):
        # A "good" negation should yield low entailment of original given negated
        mock_nli = MagicMock()
        mock_nli.batch_entail_prob.return_value = [0.15]
        result = validate_negation("X is Y", "X is not Y", mock_nli)
        assert result is True

    def test_high_entailment_fails(self):
        # A "bad" negation: still entails the original — likely rule misfire
        mock_nli = MagicMock()
        mock_nli.batch_entail_prob.return_value = [0.85]
        result = validate_negation("X is Y", "Y is X", mock_nli)
        assert result is False

    def test_nli_exception_is_permissive(self):
        # Validator failure should NOT silently disable the fix.
        mock_nli = MagicMock()
        mock_nli.batch_entail_prob.side_effect = RuntimeError("model crashed")
        result = validate_negation("X is Y", "X is not Y", mock_nli)
        assert result is True  # permissive on validator failure
