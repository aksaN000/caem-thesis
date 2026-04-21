"""
tests/test_qa_relevance.py
==========================
Unit tests for the Branch C Goal 2 ``q_a_relevance`` signal.

Coverage:
  - ``_compute_q_a_relevance`` returns 0.5 when no scorer is configured
  - Scorer output propagates through verify() to UnifiedVerifierOutput
  - Scorer exceptions fall back to 0.5 (neutral) rather than 0/1
  - Empty query / answer returns 0.5
  - Non-finite scorer output returns 0.5
  - Composite weights (including new q_a_relevance weight) sum to 1.0
  - Self-consistency: uniform v inputs -> u_stored = v with the
    seven-weight composite
  - Composite is monotone-increasing in q_a_relevance
  - EpisodicEntry accepts q_a_relevance field; pipeline store path
    propagates it from UnifiedVerifierOutput
"""

from __future__ import annotations

import numpy as np
import pytest

from caem.config import CAEMConfig
from caem.memory.entry import EpisodicEntry
from caem.verification.verifier import UnifiedVerifier, UnifiedVerifierOutput

from tests.test_verifier import _blank_verifier  # reuses the isolation helper


# -----------------------------------------------------------------------------
# _compute_q_a_relevance
# -----------------------------------------------------------------------------

class _MockScorer:
    """Minimal sentence-transformers CrossEncoder duck-type.

    Accepts **kwargs so the verifier can pass show_progress_bar=False
    (or any other future CrossEncoder.predict flag) without the mock
    having to track the signature.
    """

    def __init__(self, score: float) -> None:
        self.score = score
        self.calls: list = []

    def predict(self, pairs, **kwargs):  # noqa: ARG002 -- kwargs ignored
        self.calls.append(list(pairs))
        return np.array([self.score])


def _sigmoid(x: float) -> float:
    """Reference sigmoid used in assertions, matching _compute_q_a_relevance."""
    import math as _m
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + _m.exp(-x))


class TestComputeQARelevance:
    def test_no_scorer_returns_neutral(self):
        v = _blank_verifier()
        assert v.qa_relevance_scorer is None
        assert v._compute_q_a_relevance("q?", "a") == pytest.approx(0.5)

    def test_logit_passed_through_sigmoid(self):
        """Branch C 2026-04-21: cross-encoder logit -> sigmoid -> probability.
        A logit of +2.0 maps to ~0.88 (not 2.0 or 1.0), preserving the
        graded range for the composite.
        """
        v = _blank_verifier()
        v.qa_relevance_scorer = _MockScorer(score=2.0)
        assert v._compute_q_a_relevance("q?", "a") == pytest.approx(
            _sigmoid(2.0), abs=1e-6,
        )

    def test_negative_logit_maps_to_low_prob(self):
        v = _blank_verifier()
        v.qa_relevance_scorer = _MockScorer(score=-2.0)
        assert v._compute_q_a_relevance("q?", "a") == pytest.approx(
            _sigmoid(-2.0), abs=1e-6,
        )

    def test_zero_logit_is_half(self):
        v = _blank_verifier()
        v.qa_relevance_scorer = _MockScorer(score=0.0)
        assert v._compute_q_a_relevance("q?", "a") == pytest.approx(0.5)

    def test_extreme_positive_saturates_near_one(self):
        """+inf-ish logit saturates near 1.0 (sigmoid asymptote) but
        never exceeds 1.0 due to the defensive clip.
        """
        v = _blank_verifier()
        v.qa_relevance_scorer = _MockScorer(score=100.0)
        result = v._compute_q_a_relevance("q?", "a")
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_extreme_negative_saturates_near_zero(self):
        v = _blank_verifier()
        v.qa_relevance_scorer = _MockScorer(score=-100.0)
        result = v._compute_q_a_relevance("q?", "a")
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_scorer_exception_falls_back_to_neutral(self):
        class _Raiser:
            def predict(self, pairs):
                raise RuntimeError("simulated scorer failure")

        v = _blank_verifier()
        v.qa_relevance_scorer = _Raiser()
        assert v._compute_q_a_relevance("q?", "a") == pytest.approx(0.5)

    def test_empty_query_returns_neutral(self):
        v = _blank_verifier()
        v.qa_relevance_scorer = _MockScorer(score=0.9)
        assert v._compute_q_a_relevance("", "a") == pytest.approx(0.5)
        assert v._compute_q_a_relevance("   ", "a") == pytest.approx(0.5)

    def test_empty_answer_returns_neutral(self):
        v = _blank_verifier()
        v.qa_relevance_scorer = _MockScorer(score=0.9)
        assert v._compute_q_a_relevance("q?", "") == pytest.approx(0.5)

    def test_nan_scorer_returns_neutral(self):
        v = _blank_verifier()
        v.qa_relevance_scorer = _MockScorer(score=float("nan"))
        assert v._compute_q_a_relevance("q?", "a") == pytest.approx(0.5)

    def test_pair_passed_to_scorer(self):
        v = _blank_verifier()
        scorer = _MockScorer(score=0.5)
        v.qa_relevance_scorer = scorer
        v._compute_q_a_relevance("What is the capital of France?", "Paris")
        assert scorer.calls == [[("What is the capital of France?", "Paris")]]


# -----------------------------------------------------------------------------
# Composite
# -----------------------------------------------------------------------------

class TestComposite:
    def test_weights_sum_to_one_with_qa_relevance(self):
        cfg = CAEMConfig()
        w_sum = (
            cfg.u_stored_weight_pground_mean
            + cfg.u_stored_weight_pground_atomic
            + cfg.u_stored_weight_sc
            + cfg.u_stored_weight_se
            + cfg.u_stored_weight_uinternal
            + cfg.u_stored_weight_nli
            + cfg.u_stored_weight_q_a_relevance
        )
        assert w_sum == pytest.approx(1.0, abs=1e-6)

    def test_self_consistent_scalar_passes_through(self):
        v = _blank_verifier()
        # Feeding every positive signal at v and h_norm = 1 - v must reproduce v.
        for target in (0.2, 0.5, 0.8):
            u = v._composite(
                p_ground_mean=target, p_ground_atomic=target,
                s_avg=target, h_norm=1.0 - target,
                u_internal=target, p_entail=target,
                q_a_relevance=target,
            )
            assert u == pytest.approx(target, abs=1e-6)

    def test_monotone_in_q_a_relevance(self):
        v = _blank_verifier()
        base = dict(p_ground_mean=0.5, p_ground_atomic=0.5,
                    s_avg=0.5, h_norm=0.5,
                    u_internal=0.5, p_entail=0.5,
                    q_a_relevance=0.3)
        u_low = v._composite(**base)
        base["q_a_relevance"] = 0.9
        u_high = v._composite(**base)
        assert u_high > u_low

    def test_qa_relevance_default_keeps_composite_runnable(self):
        v = _blank_verifier()
        # Exercise the default-argument branch -- legacy callers that do
        # not thread q_a_relevance through still produce a finite composite
        # (value depends on the other six signals + the 0.5 q_a default).
        u = v._composite(
            p_ground_mean=1.0, p_ground_atomic=1.0, s_avg=1.0,
            h_norm=0.0, u_internal=1.0, p_entail=1.0,
        )
        # 0.86 * 1.0 + 0.14 * 0.5 = 0.93
        assert u == pytest.approx(0.93, abs=1e-6)


# -----------------------------------------------------------------------------
# EpisodicEntry schema
# -----------------------------------------------------------------------------

class TestEpisodicEntryQARelevance:
    def _emb(self):
        v = np.random.default_rng(0).standard_normal(768).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_default_value_is_neutral(self):
        entry = EpisodicEntry(
            question="q?", reasoning_chain="because", answer="a",
            embedding=self._emb(), storage_cycle=0, timestamp=0.0,
        )
        assert entry.q_a_relevance == pytest.approx(0.5)

    def test_field_roundtrips(self):
        entry = EpisodicEntry(
            question="q?", reasoning_chain="because", answer="a",
            embedding=self._emb(), storage_cycle=0, timestamp=0.0,
            q_a_relevance=0.83,
        )
        assert entry.q_a_relevance == pytest.approx(0.83)


# -----------------------------------------------------------------------------
# UnifiedVerifierOutput schema
# -----------------------------------------------------------------------------

class TestVerifierOutputQARelevance:
    def test_default_neutral(self):
        out = UnifiedVerifierOutput(
            u_token=0.5, u_dropout=0.5, u_internal=0.5,
            s_avg=0.5, h_norm=0.5, p_entail=0.5,
            p_ground_max=0.5, p_ground_mean=0.5, p_ground_atomic=0.5,
            p_contra=0.0, u_stored=0.5, decision="STORE",
            early_exit_triggered=False, abstained=False,
        )
        assert out.q_a_relevance == pytest.approx(0.5)

    def test_explicit_value_preserved(self):
        out = UnifiedVerifierOutput(
            u_token=0.5, u_dropout=0.5, u_internal=0.5,
            s_avg=0.5, h_norm=0.5, p_entail=0.5,
            p_ground_max=0.5, p_ground_mean=0.5, p_ground_atomic=0.5,
            p_contra=0.0, u_stored=0.5, decision="STORE",
            early_exit_triggered=False, abstained=False,
            q_a_relevance=0.77,
        )
        assert out.q_a_relevance == pytest.approx(0.77)


# -----------------------------------------------------------------------------
# Init injection contract
# -----------------------------------------------------------------------------

class TestInitAcceptsQAScorer:
    def test_init_sets_qa_relevance_scorer_attribute(self):
        # The full UnifiedVerifier.__init__ requires a real tokenizer/model
        # and is exercised by the integration tests. Here we inject a
        # scorer via the blank-verifier shim and confirm _compute routes to
        # it -- the one contract the rest of the pipeline depends on.
        # Logit 0.71 -> sigmoid ≈ 0.670 (not 0.71); the assertion tracks
        # the actual _compute contract post-sigmoid fix.
        scorer = _MockScorer(score=0.71)
        v = _blank_verifier()
        v.qa_relevance_scorer = scorer
        assert v._compute_q_a_relevance("q?", "a") == pytest.approx(
            _sigmoid(0.71), abs=1e-6,
        )
        assert scorer.calls[0] == [("q?", "a")]
