"""
tests/test_router.py
====================
Unit tests for AdaptiveRouter (Stage 3 dispatch).

Tests cover every routing path exhaustively:
  - OR-condition fires (u_pre < 0.60)         -> always Tier 3, safety_override=True
  - Tier 1: routing_score ≥ 0.90
  - Tier 2: score < 0.90, similarity > 0.75
  - Tier 3: score < 0.90, similarity ≤ 0.75
  - Empty memory                               -> always Tier 3
  - Boundary values on every threshold
  - explain() output
  - batch routing

No model, no FAISS -- pure routing logic only.

Run with:
    python -m pytest tests/test_router.py -v
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from caem.config import CAEMConfig
from caem.memory.entry import EpisodicEntry, PreRoutingConfidence, RoutingDecision
from caem.routing.router import AdaptiveRouter


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_config(
    tier1_threshold: float = 0.90,
    tier2_sim_threshold: float = 0.75,
    safety_u_pre_min: float = 0.60,
    routing_lambda: float = 0.70,
) -> CAEMConfig:
    cfg = CAEMConfig()
    cfg.tier1_combined_threshold  = tier1_threshold
    cfg.tier2_similarity_threshold = tier2_sim_threshold
    cfg.safety_u_pre_min          = safety_u_pre_min
    cfg.routing_lambda            = routing_lambda
    return cfg


def make_pc(u_pre: float, u_token: float = 0.8, c_conv: float = 0.2) -> PreRoutingConfidence:
    return PreRoutingConfidence(u_token=u_token, c_conv=c_conv, u_pre=u_pre)


def make_entry(u_stored: float = 0.85) -> EpisodicEntry:
    emb = np.zeros(768, dtype=np.float32)
    emb[0] = 1.0
    return EpisodicEntry(
        question="q", reasoning_chain="r", answer="a",
        embedding=emb, storage_cycle=0, u_stored=u_stored,
    )


def make_results(similarity: float, u_stored: float = 0.85):
    """Return a search_results list with one entry."""
    return [(make_entry(u_stored=u_stored), similarity)]


def make_results_with_id(similarity: float, u_stored: float = 0.85, entry_id: int = 7):
    """Return a search_results list including entry_id."""
    return [(make_entry(u_stored=u_stored), entry_id, similarity)]


def routing_score(sim: float, u_stored: float, lam: float = 0.70) -> float:
    return lam * sim + (1 - lam) * u_stored


# -----------------------------------------------------------------------------
# OR-condition (Mechanism 1)
# -----------------------------------------------------------------------------

class TestORCondition:
    """The OR-condition must fire BEFORE any routing score is computed.

    Even a perfect routing score (1.0) must be overridden by a dangerous u_pre.
    """

    def test_low_u_pre_forces_tier3(self):
        router = AdaptiveRouter(make_config())
        pc = make_pc(u_pre=0.45)
        # Memory match that would otherwise route to Tier 1
        results = make_results(similarity=0.98, u_stored=0.99)
        d = router.route(pc, results)
        assert d.tier == 3
        assert d.safety_override is True

    def test_u_pre_at_zero_forces_tier3(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.0), make_results(0.99, 0.99))
        assert d.tier == 3
        assert d.safety_override is True

    def test_u_pre_just_below_threshold_fires(self):
        router = AdaptiveRouter(make_config(safety_u_pre_min=0.60))
        d = router.route(make_pc(0.5999), make_results(0.99, 0.99))
        assert d.tier == 3
        assert d.safety_override is True

    def test_u_pre_at_exact_threshold_does_not_fire(self):
        """u_pre == threshold is safe (>= condition)."""
        router = AdaptiveRouter(make_config(safety_u_pre_min=0.60))
        # With u_pre=0.60, OR-condition does NOT fire.
        # routing_score = 0.70*0.99 + 0.30*0.99 = 0.99 ≥ 0.90 -> Tier 1
        d = router.route(make_pc(0.60), make_results(0.99, 0.99))
        assert d.safety_override is False
        assert d.tier == 1

    def test_u_pre_above_threshold_does_not_fire(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.75), make_results(0.80, 0.80))
        assert d.safety_override is False

    def test_safety_override_recorded_in_decision(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.30), make_results(0.99, 0.99))
        assert d.safety_override is True
        assert d.tier == 3
        assert d.u_pre == pytest.approx(0.30, abs=1e-6)

    def test_or_condition_fires_with_empty_memory_too(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.40), [])
        assert d.tier == 3
        assert d.safety_override is True


# -----------------------------------------------------------------------------
# Tier 1 routing (Mechanism 2, high routing_score)
# -----------------------------------------------------------------------------

class TestTier1Routing:
    def test_high_score_routes_tier1(self):
        # routing_score = 0.70*0.95 + 0.30*0.95 = 0.95 ≥ 0.90
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.75), make_results(0.95, 0.95))
        assert d.tier == 1
        assert d.safety_override is False

    def test_routing_score_at_exact_tier1_threshold(self):
        # routing_score = 0.70*s + 0.30*u = 0.90 exactly -> Tier 1
        # Solve: 0.70*s + 0.30*0.80 = 0.90 -> s ≈ 0.9428...
        # Add small epsilon to ensure floating-point result is strictly ≥ 0.90.
        cfg = make_config()
        u_stored = 0.80
        sim = (cfg.tier1_combined_threshold - (1 - cfg.routing_lambda) * u_stored) / cfg.routing_lambda
        sim += 1e-9   # guard against floating-point underflow to 0.8999...
        router = AdaptiveRouter(cfg)
        d = router.route(make_pc(0.75), make_results(sim, u_stored))
        assert d.tier == 1
        assert d.routing_score >= cfg.tier1_combined_threshold - 1e-9

    def test_routing_score_just_below_tier1_not_tier1(self):
        # Ensure 0.8999 < 0.90 -> not Tier 1
        cfg = make_config()
        u_stored = 0.80
        sim = (0.8999 - (1 - cfg.routing_lambda) * u_stored) / cfg.routing_lambda
        sim = max(0.0, sim)
        router = AdaptiveRouter(cfg)
        d = router.route(make_pc(0.75), make_results(sim, u_stored))
        assert d.tier != 1

    def test_tier1_routing_score_stored_correctly(self):
        router = AdaptiveRouter(make_config())
        results = make_results(0.95, 0.95)
        d = router.route(make_pc(0.75), results)
        expected = 0.70 * 0.95 + 0.30 * 0.95
        assert math.isclose(d.routing_score, expected, abs_tol=1e-6)

    def test_tier1_u_stored_retrieved_stored(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.75), make_results(0.95, u_stored=0.88))
        assert math.isclose(d.u_stored_retrieved, 0.88, abs_tol=1e-6)

    def test_tier1_similarity_stored(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.75), make_results(0.95, 0.95))
        assert math.isclose(d.similarity, 0.95, abs_tol=1e-6)

    def test_retrieved_entry_id_propagated_when_present(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.75), make_results_with_id(0.95, 0.95, entry_id=42))
        assert d.retrieved_entry_id == 42


# -----------------------------------------------------------------------------
# Tier 2 routing
# -----------------------------------------------------------------------------

class TestTier2Routing:
    def test_moderate_sim_above_tier2_threshold_routes_tier2(self):
        # routing_score < 0.90, similarity > 0.75 -> Tier 2
        # sim=0.80, u_stored=0.60: score = 0.70*0.80 + 0.30*0.60 = 0.56+0.18 = 0.74 < 0.90
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.75), make_results(0.80, u_stored=0.60))
        assert d.tier == 2
        assert d.safety_override is False

    def test_sim_at_tier2_threshold_boundary(self):
        """similarity > 0.75 -> Tier 2; similarity == 0.75 -> Tier 3."""
        cfg = make_config()
        router = AdaptiveRouter(cfg)

        # sim = 0.7501 (just above): score = 0.70*0.7501 + 0.30*0.50 = 0.5250+0.15 = 0.675 < 0.90
        d_above = router.route(make_pc(0.75), make_results(0.7501, u_stored=0.50))
        assert d_above.tier == 2

        # sim = 0.75 (exactly): NOT > threshold -> Tier 3
        d_exact = router.route(make_pc(0.75), make_results(0.75, u_stored=0.50))
        assert d_exact.tier == 3

    def test_tier2_routing_score_below_tier1(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.75), make_results(0.80, 0.60))
        assert d.routing_score < 0.90
        assert d.similarity > 0.75


# -----------------------------------------------------------------------------
# Tier 3 routing (non-safety-override paths)
# -----------------------------------------------------------------------------

class TestTier3Routing:
    def test_low_sim_low_score_routes_tier3(self):
        # sim=0.40, u_stored=0.50: score = 0.28+0.15 = 0.43 < 0.90, sim ≤ 0.75
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.75), make_results(0.40, 0.50))
        assert d.tier == 3
        assert d.safety_override is False

    def test_empty_memory_routes_tier3(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.75), [])
        assert d.tier == 3
        assert d.safety_override is False
        assert d.similarity == 0.0
        assert d.u_stored_retrieved == 0.0
        assert math.isclose(d.routing_score, 0.0, abs_tol=1e-6)

    def test_high_u_pre_but_zero_memory_still_tier3(self):
        """Even a fully confident model routes to Tier 3 with empty memory."""
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.99), [])
        assert d.tier == 3
        assert d.safety_override is False

    def test_tier3_routing_score_stored(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.75), make_results(0.40, 0.50))
        expected = 0.70 * 0.40 + 0.30 * 0.50
        assert math.isclose(d.routing_score, expected, abs_tol=1e-6)


# -----------------------------------------------------------------------------
# Routing score formula correctness
# -----------------------------------------------------------------------------

class TestRoutingScoreFormula:
    def test_formula_uses_config_lambda(self):
        """routing_score = λ·sim + (1-λ)·û_stored."""
        cfg = make_config(routing_lambda=0.60)
        router = AdaptiveRouter(cfg)
        sim, u_stored = 0.80, 0.70
        d = router.route(make_pc(0.75), make_results(sim, u_stored))
        expected = 0.60 * sim + 0.40 * u_stored
        assert math.isclose(d.routing_score, expected, abs_tol=1e-6)

    def test_formula_with_different_lambda(self):
        cfg = make_config(routing_lambda=0.50)
        router = AdaptiveRouter(cfg)
        sim, u_stored = 0.90, 0.80
        d = router.route(make_pc(0.75), make_results(sim, u_stored))
        expected = 0.50 * 0.90 + 0.50 * 0.80
        assert math.isclose(d.routing_score, expected, abs_tol=1e-6)

    def test_u_pre_not_in_routing_score(self):
        """u_pre is only used in the OR-condition, NOT in the routing score formula."""
        cfg = make_config()
        router = AdaptiveRouter(cfg)
        sim, u_stored = 0.85, 0.80
        # Same sim/u_stored, different u_pre (both above threshold)
        d1 = router.route(make_pc(0.65), make_results(sim, u_stored))
        d2 = router.route(make_pc(0.99), make_results(sim, u_stored))
        # Routing score must be identical
        assert math.isclose(d1.routing_score, d2.routing_score, abs_tol=1e-6)
        # Tier must be the same (both above OR-condition threshold)
        assert d1.tier == d2.tier


# -----------------------------------------------------------------------------
# Complete routing matrix (all four paths)
# -----------------------------------------------------------------------------

class TestFullRoutingMatrix:
    """Verify all four paths with representative values."""

    def setup_method(self):
        self.router = AdaptiveRouter(make_config())

    def test_path_or_condition(self):
        d = self.router.route(make_pc(0.40), make_results(0.99, 0.99))
        assert d.tier == 3 and d.safety_override is True

    def test_path_tier1(self):
        # score = 0.70*0.96 + 0.30*0.95 = 0.672 + 0.285 = 0.957 ≥ 0.90
        d = self.router.route(make_pc(0.80), make_results(0.96, 0.95))
        assert d.tier == 1 and d.safety_override is False

    def test_path_tier2(self):
        # score = 0.70*0.80 + 0.30*0.55 = 0.56 + 0.165 = 0.725 < 0.90
        # sim = 0.80 > 0.75 -> Tier 2
        d = self.router.route(make_pc(0.80), make_results(0.80, 0.55))
        assert d.tier == 2 and d.safety_override is False

    def test_path_tier3_no_memory(self):
        d = self.router.route(make_pc(0.80), [])
        assert d.tier == 3 and d.safety_override is False

    def test_path_tier3_low_sim(self):
        # score = 0.70*0.50 + 0.30*0.50 = 0.50 < 0.90, sim=0.50 ≤ 0.75 -> Tier 3
        d = self.router.route(make_pc(0.80), make_results(0.50, 0.50))
        assert d.tier == 3 and d.safety_override is False


# -----------------------------------------------------------------------------
# explain()
# -----------------------------------------------------------------------------

class TestExplain:
    def test_explain_or_condition_mentions_u_pre(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.35), make_results(0.99, 0.99))
        explanation = router.explain(d)
        assert "OR-condition" in explanation
        assert "0.35" in explanation or "u_pre" in explanation

    def test_explain_tier1_mentions_score(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.80), make_results(0.96, 0.95))
        explanation = router.explain(d)
        assert "Tier 1" in explanation

    def test_explain_tier2_mentions_similarity(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.80), make_results(0.80, 0.55))
        explanation = router.explain(d)
        assert "Tier 2" in explanation

    def test_explain_tier3_no_memory(self):
        router = AdaptiveRouter(make_config())
        d = router.route(make_pc(0.80), [])
        explanation = router.explain(d)
        assert "Tier 3" in explanation

    def test_explain_returns_string(self):
        router = AdaptiveRouter(make_config())
        for pc_val, sim in [(0.35, 0.99), (0.80, 0.96), (0.80, 0.80), (0.80, 0.40)]:
            d = router.route(make_pc(pc_val), make_results(sim, 0.85))
            assert isinstance(router.explain(d), str)


# -----------------------------------------------------------------------------
# Batch routing
# -----------------------------------------------------------------------------

class TestBatchRouting:
    def test_batch_length(self):
        router = AdaptiveRouter(make_config())
        items = [
            (make_pc(0.35), make_results(0.99, 0.99)),
            (make_pc(0.80), make_results(0.96, 0.95)),
            (make_pc(0.80), make_results(0.80, 0.55)),
            (make_pc(0.80), []),
        ]
        results = router.route_batch(items)
        assert len(results) == 4

    def test_batch_tiers_correct(self):
        router = AdaptiveRouter(make_config())
        items = [
            (make_pc(0.35), make_results(0.99, 0.99)),   # OR-condition -> Tier 3
            (make_pc(0.80), make_results(0.96, 0.95)),   # Tier 1
            (make_pc(0.80), make_results(0.80, 0.55)),   # Tier 2
            (make_pc(0.80), []),                          # Tier 3
        ]
        results = router.route_batch(items)
        tiers = [d.tier for d in results]
        assert tiers == [3, 1, 2, 3]
        assert results[0].safety_override is True
        assert results[1].safety_override is False


# =========================================================================== #
# Per-benchmark safety_u_pre_min_b dispatch (v2 Fix 12)                        #
# =========================================================================== #
# The companion estimator-side tests live in tests/test_pre_routing.py
# (TestPerBenchmarkUPre); the calibration-side fitting helpers live in
# tests/test_calibration_batch_equivalence.py.

class TestPerBenchmarkSafetyFloor:
    def test_router_dispatches_safety_floor_by_benchmark(self):
        """The router's OR-condition must use safety_u_pre_min_per_benchmark
        when a known source_benchmark is supplied."""
        from caem.config import CAEMConfig
        from caem.routing.router import AdaptiveRouter
        from caem.memory.entry import PreRoutingConfidence

        cfg = CAEMConfig()
        cfg.safety_u_pre_min = 0.38  # pooled global
        cfg.safety_u_pre_min_per_benchmark = {
            "fever":    0.55,   # stricter — more queries forced to Tier 3
            "triviaqa": 0.20,   # looser — fewer queries forced to Tier 3
        }
        router = AdaptiveRouter(cfg)
        pc = PreRoutingConfidence(u_token=0.5, c_conv=0.0, u_pre=0.40)

        d_pooled = router.route(pc, [], source_benchmark=None)
        d_fever = router.route(pc, [], source_benchmark="fever")
        d_tqa = router.route(pc, [], source_benchmark="triviaqa")
        d_unknown = router.route(pc, [], source_benchmark="not_a_real_bench")

        # u_pre=0.40 vs pooled 0.38 → safe; vs fever 0.55 → unsafe;
        # vs triviaqa 0.20 → safe; unknown → falls back to pooled
        assert d_pooled.safety_override is False
        assert d_fever.safety_override is True
        assert d_tqa.safety_override is False
        assert d_unknown.safety_override == d_pooled.safety_override
        assert d_fever.tier == 3  # override forces Tier 3

    def test_router_back_compat_when_per_bench_dict_empty(self):
        """Empty per_bench dict + any source_benchmark → pooled behaviour."""
        from caem.config import CAEMConfig
        from caem.routing.router import AdaptiveRouter
        from caem.memory.entry import PreRoutingConfidence

        cfg = CAEMConfig()
        cfg.safety_u_pre_min = 0.45
        # empty per-bench dict (default)
        router = AdaptiveRouter(cfg)
        pc = PreRoutingConfidence(u_token=0.5, c_conv=0.0, u_pre=0.40)

        d_no_tag = router.route(pc, [], source_benchmark=None)
        d_fever = router.route(pc, [], source_benchmark="fever")
        assert d_no_tag.safety_override == d_fever.safety_override
