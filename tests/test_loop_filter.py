"""
tests/test_loop_filter.py
=========================
Unit tests for ``caem.training.loop_filter.is_repetitive_loop``.

Coverage:
  - distinct-4 n-gram signal detects token-level loops
  - zlib compression signal detects char-level loops that escape the
    distinct-4 heuristic
  - Short chains (< loop_min_tokens) exempt by design
  - Clean scaffolded CoT passes
  - Empty / whitespace-only inputs handled
  - Thresholds picked from CAEMConfig (not hardcoded)

Also exercises the SIL integration:
  - ``SelfImprovementLoop._collect_episodes`` falls back to ``entry.answer``
    when the chain trips the filter
  - ``_last_chain_diagnostics["n_loop_filtered"]`` is incremented
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import pytest

from caem.config import CAEMConfig
from caem.memory.store import EpisodicMemoryStore
from caem.training.loop_filter import (
    _compression_ratio,
    _distinct_ngram_ratio,
    is_repetitive_loop,
)
from caem.training.self_improvement import SelfImprovementLoop

from tests.test_self_improvement import make_entry, make_mock_model, make_mock_tokenizer


# -----------------------------------------------------------------------------
# Primitive signal tests
# -----------------------------------------------------------------------------

class TestDistinctNgramRatio:
    def test_empty_tokens_returns_one(self):
        assert _distinct_ngram_ratio([], n=4) == pytest.approx(1.0)

    def test_too_short_for_ngram_returns_one(self):
        # Fewer than n tokens -> no n-grams computable -> cannot be a loop
        assert _distinct_ngram_ratio(["a", "b"], n=4) == pytest.approx(1.0)

    def test_all_unique_yields_one(self):
        tokens = ["a", "b", "c", "d", "e", "f", "g"]
        # 4 unique 4-grams, 4 total -> 1.0
        assert _distinct_ngram_ratio(tokens, n=4) == pytest.approx(1.0)

    def test_heavy_repetition_goes_low(self):
        tokens = ["yes"] * 20
        # Only one distinct 4-gram in 17 total -> ratio ~= 1/17 < 0.1
        ratio = _distinct_ngram_ratio(tokens, n=4)
        assert ratio < 0.1


class TestCompressionRatio:
    def test_empty_returns_one(self):
        assert _compression_ratio("") == pytest.approx(1.0)

    def test_prose_ratio_reasonable(self):
        prose = (
            "The model computes token probabilities over the vocabulary "
            "and selects the highest scoring next token at each step."
        )
        # Prose compresses but not dramatically; expect > 0.5 for short English.
        assert _compression_ratio(prose) > 0.45

    def test_heavy_repetition_compresses_hard(self):
        loop = "yes " * 500
        assert _compression_ratio(loop) < 0.1


# -----------------------------------------------------------------------------
# is_repetitive_loop
# -----------------------------------------------------------------------------

class TestIsRepetitiveLoop:
    def test_short_text_exempt(self):
        cfg = CAEMConfig()
        # Only 5 tokens, below loop_min_tokens (20) -- cannot be a loop.
        assert is_repetitive_loop("the answer is yes indeed", cfg) is False

    def test_empty_not_a_loop(self):
        cfg = CAEMConfig()
        assert is_repetitive_loop("", cfg) is False
        assert is_repetitive_loop("   \n  \t  ", cfg) is False

    def test_clean_cot_passes(self):
        cfg = CAEMConfig()
        chain = (
            "Reasoning: The question asks about the capital of France. "
            "Paris is widely known as the French capital, historically "
            "established as the seat of government and cultural centre "
            "of the country since at least the medieval period.\n"
            "Answer: Paris"
        )
        assert is_repetitive_loop(chain, cfg) is False

    def test_token_loop_flagged(self):
        cfg = CAEMConfig()
        loop = "yes " * 50
        assert is_repetitive_loop(loop, cfg) is True

    def test_char_loop_flagged_by_compression(self):
        # Long enough to hit the token threshold and the zlib ratio.
        cfg = CAEMConfig()
        loop = ("aaaa " * 100).strip()
        assert is_repetitive_loop(loop, cfg) is True

    def test_thresholds_come_from_config(self):
        cfg = CAEMConfig()
        # Force the filter off by lowering thresholds to zero; previously-
        # flagged loops now pass. Protects against accidental hardcoding.
        cfg.loop_distinct4_threshold = 0.0
        cfg.loop_compression_threshold = 0.0
        assert is_repetitive_loop("yes " * 50, cfg) is False


# -----------------------------------------------------------------------------
# SIL integration
# -----------------------------------------------------------------------------

def _loop_chain() -> str:
    # ~30 tokens of pure repetition, classical distinct-4 trigger.
    return " ".join(["yes"] * 30)


class TestCollectEpisodesLoopFilter:
    def _make_loop_entry(self):
        entry = make_entry("Is water wet?", answer="yes", u_stored=0.80)
        entry.reasoning_chain = _loop_chain()
        return entry

    def test_looped_chain_falls_back_to_answer(self):
        cfg = CAEMConfig()
        cfg.min_u_stored_for_training = 0.70
        with tempfile.TemporaryDirectory() as tmpdir:
            loop = SelfImprovementLoop(
                make_mock_model(), make_mock_tokenizer(), cfg,
                device="cpu", output_dir=tmpdir,
            )
            store = EpisodicMemoryStore()
            store.add(self._make_loop_entry())
            pairs = loop._collect_episodes(store)
            # Single pair produced; its target is the short answer, not the loop.
            assert len(pairs) == 1
            assert pairs[0].answer == "yes"
            # Diagnostic counter flipped.
            diag = loop._last_chain_diagnostics or {}
            assert diag.get("n_loop_filtered") == 1

    def test_non_loop_chain_preserved(self):
        cfg = CAEMConfig()
        cfg.min_u_stored_for_training = 0.70
        with tempfile.TemporaryDirectory() as tmpdir:
            loop = SelfImprovementLoop(
                make_mock_model(), make_mock_tokenizer(), cfg,
                device="cpu", output_dir=tmpdir,
            )
            store = EpisodicMemoryStore()
            entry = make_entry("Who wrote Hamlet?", answer="Shakespeare", u_stored=0.9)
            entry.reasoning_chain = (
                "Reasoning: Shakespeare is widely attributed as the author "
                "of Hamlet; the First Folio of 1623 lists him as the "
                "sole author and the play is part of his tragedies.\n"
                "Answer: Shakespeare"
            )
            store.add(entry)
            pairs = loop._collect_episodes(store)
            # Chain retained (NOT the answer field).
            assert "Shakespeare is widely attributed" in pairs[0].answer
            diag = loop._last_chain_diagnostics or {}
            assert diag.get("n_loop_filtered", 0) == 0
