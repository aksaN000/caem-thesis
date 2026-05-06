"""
tests/test_fix12_per_benchmark_u_pre.py
=========================================
Smoke test for v2 Fix 12 — per-benchmark u_pre temperature scaling
(T_b) + per-benchmark safety_u_pre_min_b dispatch.

Verifies:
  1. CAEMConfig defaults have empty per-benchmark dicts (back-compat
     path: every query falls back to pooled global)
  2. get_temperature_for() / get_safety_u_pre_min_for() resolve correctly
     for None / known / unknown source_benchmark
  3. PreRoutingConfidenceEstimator._apply_temperature_scaling() honours
     per-bench T_b when present, falls back to pooled T otherwise
  4. AdaptiveRouter.route() honours per-bench safety_u_pre_min_b when
     present, falls back to pooled safety_u_pre_min otherwise
  5. Per-benchmark T_b actually changes u_pre (not silent identity)
  6. Per-benchmark safety_u_pre_min_b actually changes safety_override
     (not silent identity)
"""
from __future__ import annotations

import math
import os
import sys


def test_config_defaults_have_empty_per_bench_dicts():
    from caem.config import CAEMConfig
    cfg = CAEMConfig()
    # Pooled fallbacks must remain
    assert hasattr(cfg, "temperature_scalar")
    assert hasattr(cfg, "safety_u_pre_min")
    # New per-bench dicts must exist and start empty (= back-compat path)
    assert hasattr(cfg, "temperature_scalar_per_benchmark"), (
        "CAEMConfig must expose temperature_scalar_per_benchmark"
    )
    assert hasattr(cfg, "safety_u_pre_min_per_benchmark"), (
        "CAEMConfig must expose safety_u_pre_min_per_benchmark"
    )
    assert isinstance(cfg.temperature_scalar_per_benchmark, dict)
    assert isinstance(cfg.safety_u_pre_min_per_benchmark, dict)
    assert len(cfg.temperature_scalar_per_benchmark) == 0
    assert len(cfg.safety_u_pre_min_per_benchmark) == 0


def test_helper_getters_dispatch_correctly():
    from caem.config import CAEMConfig
    cfg = CAEMConfig()
    # No per-bench data → fallback to pooled global
    assert cfg.get_temperature_for(None) == cfg.temperature_scalar
    assert cfg.get_temperature_for("fever") == cfg.temperature_scalar
    assert cfg.get_safety_u_pre_min_for(None) == cfg.safety_u_pre_min
    assert cfg.get_safety_u_pre_min_for("fever") == cfg.safety_u_pre_min

    # Populate per-bench dicts and confirm dispatch picks them up
    cfg.temperature_scalar_per_benchmark = {"fever": 2.5, "triviaqa": 0.7}
    cfg.safety_u_pre_min_per_benchmark = {"fever": 0.55, "triviaqa": 0.20}

    assert cfg.get_temperature_for("fever") == 2.5
    assert cfg.get_temperature_for("triviaqa") == 0.7
    # Unknown benchmark falls back to pooled global
    assert cfg.get_temperature_for("hotpotqa") == cfg.temperature_scalar
    # None falls back to pooled global
    assert cfg.get_temperature_for(None) == cfg.temperature_scalar

    assert cfg.get_safety_u_pre_min_for("fever") == 0.55
    assert cfg.get_safety_u_pre_min_for("triviaqa") == 0.20
    assert cfg.get_safety_u_pre_min_for("hotpotqa") == cfg.safety_u_pre_min
    assert cfg.get_safety_u_pre_min_for(None) == cfg.safety_u_pre_min


def test_apply_temperature_scaling_dispatches_per_benchmark():
    """The _apply_temperature_scaling helper must read T_b from
    temperature_scalar_per_benchmark when source_benchmark is supplied."""
    from caem.config import CAEMConfig
    from caem.confidence.pre_routing import PreRoutingConfidenceEstimator

    cfg = CAEMConfig()
    cfg.temperature_scalar = 1.0  # pooled = identity
    cfg.temperature_scalar_per_benchmark = {
        "fever": 2.0,    # T > 1 squashes confidence toward 0.5
        "triviaqa": 0.5,  # T < 1 sharpens confidence away from 0.5
    }

    # Build estimator without instantiating a model — only the helper is exercised
    est = PreRoutingConfidenceEstimator.__new__(PreRoutingConfidenceEstimator)
    est.config = cfg

    raw = 0.80  # "above 0.5" — T > 1 should pull DOWN, T < 1 should push UP

    p_pooled = est._apply_temperature_scaling(raw, source_benchmark=None)
    p_fever = est._apply_temperature_scaling(raw, source_benchmark="fever")
    p_tqa = est._apply_temperature_scaling(raw, source_benchmark="triviaqa")
    p_unknown = est._apply_temperature_scaling(raw, source_benchmark="not_a_real_bench")

    # Pooled T=1 → identity
    assert abs(p_pooled - raw) < 1e-9, f"pooled T=1 should be identity; got {p_pooled}"
    # Unknown benchmark falls back to pooled (= identity here)
    assert abs(p_unknown - raw) < 1e-9, (
        f"unknown bench should fall back to pooled identity; got {p_unknown}"
    )
    # T=2 should pull a >0.5 prob DOWN toward 0.5
    assert p_fever < raw, f"fever T=2 should reduce p; got {p_fever} >= raw {raw}"
    assert p_fever > 0.5, f"fever T=2 should still be >0.5; got {p_fever}"
    # T=0.5 should push a >0.5 prob UP, away from 0.5
    assert p_tqa > raw, f"triviaqa T=0.5 should increase p; got {p_tqa} <= raw {raw}"
    # All three must produce *different* values — proves dispatch is live
    assert len({round(p_pooled, 6), round(p_fever, 6), round(p_tqa, 6)}) == 3, (
        f"dispatch silent — pooled/fever/triviaqa identical: "
        f"{p_pooled:.6f}, {p_fever:.6f}, {p_tqa:.6f}"
    )


def test_router_dispatches_safety_floor_by_benchmark():
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

    # Same u_pre tested under three thresholds
    pc = PreRoutingConfidence(u_token=0.5, c_conv=0.0, u_pre=0.40)

    # u_pre=0.40 vs:
    #  - pooled 0.38 → safe (no override)
    #  - fever 0.55 → unsafe (override fires)
    #  - triviaqa 0.20 → safe (no override)
    decision_pooled = router.route(pc, [], source_benchmark=None)
    decision_fever = router.route(pc, [], source_benchmark="fever")
    decision_tqa = router.route(pc, [], source_benchmark="triviaqa")
    decision_unknown = router.route(pc, [], source_benchmark="not_a_real_bench")

    assert decision_pooled.safety_override is False, (
        f"u_pre=0.40 vs pooled 0.38 should be safe; got override={decision_pooled.safety_override}"
    )
    assert decision_fever.safety_override is True, (
        f"u_pre=0.40 vs fever 0.55 should fire override; got override={decision_fever.safety_override}"
    )
    assert decision_tqa.safety_override is False, (
        f"u_pre=0.40 vs triviaqa 0.20 should be safe; got override={decision_tqa.safety_override}"
    )
    # Unknown benchmark falls back to pooled
    assert decision_unknown.safety_override == decision_pooled.safety_override, (
        f"unknown bench should match pooled; got "
        f"unknown={decision_unknown.safety_override} pooled={decision_pooled.safety_override}"
    )

    # Tier should follow the override
    assert decision_fever.tier == 3, f"fever override should force Tier 3; got {decision_fever.tier}"


def test_back_compat_unchanged_when_dicts_empty():
    """Old code paths (no source_benchmark, no per-bench dicts) must
    behave identically to the v1 single-T / single-threshold pipeline."""
    from caem.config import CAEMConfig
    from caem.confidence.pre_routing import PreRoutingConfidenceEstimator
    from caem.routing.router import AdaptiveRouter
    from caem.memory.entry import PreRoutingConfidence

    cfg = CAEMConfig()
    cfg.temperature_scalar = 1.7
    cfg.safety_u_pre_min = 0.45
    # empty per-bench dicts (default)

    est = PreRoutingConfidenceEstimator.__new__(PreRoutingConfidenceEstimator)
    est.config = cfg
    router = AdaptiveRouter(cfg)

    raw = 0.82
    # No bench tag = pooled T (=1.7)
    p_no_tag = est._apply_temperature_scaling(raw, source_benchmark=None)
    # Any bench tag with empty dict = falls back to pooled
    p_fever = est._apply_temperature_scaling(raw, source_benchmark="fever")
    assert abs(p_no_tag - p_fever) < 1e-9, (
        f"empty per-bench dict should yield identical T regardless of bench; "
        f"got {p_no_tag} vs {p_fever}"
    )

    # Likewise for router
    pc = PreRoutingConfidence(u_token=0.5, c_conv=0.0, u_pre=0.40)
    d_no_tag = router.route(pc, [], source_benchmark=None)
    d_fever = router.route(pc, [], source_benchmark="fever")
    assert d_no_tag.safety_override == d_fever.safety_override


def test_pipeline_threading_signature_accepts_source_benchmark():
    """Smoke check: confirm pipeline.answer signature still accepts
    source_benchmark and that pre_estimator.estimate / router.route
    call sites have been updated to pass it through. We don't construct
    a real pipeline (no GPU); just inspect the call wiring statically."""
    import inspect
    from caem.pipeline import CAEMPipeline
    from caem.confidence.pre_routing import PreRoutingConfidenceEstimator
    from caem.routing.router import AdaptiveRouter

    # estimate accepts source_benchmark
    sig = inspect.signature(PreRoutingConfidenceEstimator.estimate)
    assert "source_benchmark" in sig.parameters, (
        "PreRoutingConfidenceEstimator.estimate must accept source_benchmark"
    )

    # router.route accepts source_benchmark
    sig = inspect.signature(AdaptiveRouter.route)
    assert "source_benchmark" in sig.parameters, (
        "AdaptiveRouter.route must accept source_benchmark"
    )

    # pipeline.answer already had source_benchmark; just confirm it threads it.
    src = inspect.getsource(CAEMPipeline.answer)
    assert "self.pre_estimator.estimate(" in src
    assert "source_benchmark=source_benchmark" in src, (
        "CAEMPipeline.answer must pass source_benchmark to pre_estimator and router"
    )
    assert "self.router.route(" in src


def _make_synthetic_calib_records(n_per_bench: int, em_rate: float, seed: int = 0):
    """Build calibration-fold records used by fit_per_benchmark_temperatures /
    fit_per_benchmark_safety_floors. Higher u_pre correlates with em=1."""
    import random
    rng = random.Random(seed)
    records = []
    for i in range(n_per_bench):
        em = 1 if rng.random() < em_rate else 0
        if em == 1:
            u = rng.uniform(0.55, 0.95)
        else:
            u = rng.uniform(0.05, 0.50)
        records.append({"benchmark": "x", "u_pre": u, "em": em})
    return records


def test_fit_per_benchmark_temperatures():
    from scripts.run_calibration import fit_per_benchmark_temperatures
    # Build records for two benchmarks with different em-rates so T_b should
    # plausibly differ. Mix the per-bench records into one flat list.
    recs_a = _make_synthetic_calib_records(120, em_rate=0.55, seed=1)
    recs_b = _make_synthetic_calib_records(120, em_rate=0.40, seed=2)
    for r in recs_a:
        r["benchmark"] = "fever"
    for r in recs_b:
        r["benchmark"] = "triviaqa"
    records = recs_a + recs_b

    fit = fit_per_benchmark_temperatures(records, pooled_T=1.0)
    assert set(fit.keys()) == {"fever", "triviaqa"}
    # fit_temperature_scalar bounds T to [exp(-3), exp(3)] = [0.0498, 20.09]
    for bm, T_b in fit.items():
        assert 0.04 <= T_b <= 20.1, f"T_b for {bm} out of bounds: {T_b}"


def test_fit_per_benchmark_safety_floors():
    from scripts.run_calibration import fit_per_benchmark_safety_floors
    # Construct one benchmark with a clean precision/u_pre curve (em=1 only
    # at high u_pre) — the floor should land well above 0.0.
    records = []
    import random
    rng = random.Random(7)
    for _ in range(200):
        em = rng.randint(0, 1)
        u = rng.uniform(0.6, 0.95) if em == 1 else rng.uniform(0.05, 0.4)
        records.append({"benchmark": "fever", "u_pre": u, "em": em})

    # Use a HIGH precision target so the chosen floor is forced into the
    # gap between the two bands. At target=0.95, almost all em=0 records
    # must be excluded from the kept set, which means u_pre must exceed
    # the top of the wrong band (~0.4).
    floors = fit_per_benchmark_safety_floors(
        records, pooled_floor=0.38, target_precision=0.95,
    )
    assert "fever" in floors
    assert floors["fever"] >= 0.30, (
        f"floor unexpectedly low for cleanly-separated bands at "
        f"target_precision=0.95: {floors['fever']}"
    )

    # Also verify the lower-target case: at target=0.80 the floor lands
    # somewhere inside the wrong band, but it MUST still be above the
    # all-zeros floor (0.0) — otherwise the sweep silently never fires.
    floors_loose = fit_per_benchmark_safety_floors(
        records, pooled_floor=0.38, target_precision=0.80,
    )
    assert floors_loose["fever"] > 0.0, (
        f"loose-target sweep silently failed; got {floors_loose['fever']}"
    )


def test_fit_per_benchmark_falls_back_for_thin_data():
    from scripts.run_calibration import (
        fit_per_benchmark_temperatures,
        fit_per_benchmark_safety_floors,
    )
    # Two benchmarks: one with plenty of records, one with too few.
    recs = []
    import random
    rng = random.Random(11)
    for _ in range(120):
        em = rng.randint(0, 1)
        u = rng.uniform(0.4, 0.9) if em == 1 else rng.uniform(0.0, 0.5)
        recs.append({"benchmark": "fever", "u_pre": u, "em": em})
    # Only 5 records for triviaqa — below the min_n_per_bench=20 threshold
    for _ in range(5):
        recs.append({"benchmark": "triviaqa", "u_pre": 0.5, "em": 1})

    pooled_T = 1.7
    pooled_floor = 0.42
    Ts = fit_per_benchmark_temperatures(recs, pooled_T=pooled_T)
    floors = fit_per_benchmark_safety_floors(recs, pooled_floor=pooled_floor)
    # Thin benchmark must fall back to pooled values
    assert Ts["triviaqa"] == pooled_T
    assert floors["triviaqa"] == pooled_floor
    # Healthy benchmark gets a real fit
    assert 0.05 <= Ts["fever"] <= 20.1


def test_temperature_scaling_math_invariants():
    """Sanity: T → 1 collapses to identity; T → ∞ collapses to 0.5."""
    from caem.config import CAEMConfig
    from caem.confidence.pre_routing import PreRoutingConfidenceEstimator

    cfg = CAEMConfig()
    est = PreRoutingConfidenceEstimator.__new__(PreRoutingConfidenceEstimator)
    est.config = cfg

    # T=1 → identity
    cfg.temperature_scalar = 1.0
    assert abs(est._apply_temperature_scaling(0.30, None) - 0.30) < 1e-9
    assert abs(est._apply_temperature_scaling(0.78, None) - 0.78) < 1e-9

    # T = very large → calibrated value approaches 0.5
    cfg.temperature_scalar = 1000.0
    p_big = est._apply_temperature_scaling(0.95, None)
    assert abs(p_big - 0.5) < 1e-2, f"large T should collapse to ~0.5; got {p_big}"


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
