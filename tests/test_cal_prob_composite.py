"""
tests/test_cal_prob_composite.py
==================================
Unit tests for caem.verification.cal_prob_composite.CalProbComposite,
including the per-benchmark dispatch surface and v1/v2 schema
back-compat.

Coverage
--------
  * SCHEMA_VERSION default = V2 ("branchC.2026-05-06")
  * fit_per_benchmark(): pooled-global + per-bench children
  * predict() dispatch:
      - source_benchmark="X" → per_benchmark["X"]
      - source_benchmark=None → pooled global fallback
      - source_benchmark="unknown" → pooled fallback
  * Per-bench composites yield DIFFERENT predictions for the same
    signal vector when their training data has inverted relationships
    (proves dispatch is not silently identical)
  * save() emits v2 nested schema when per_benchmark is non-empty
  * load() auto-detects v1 flat vs v2 nested (back-compat)
  * Real cycle-0 v1 artifact loads cleanly via the back-compat path
"""
from __future__ import annotations

import json
import os
import sys
import tempfile


def _make_synthetic_samples(n: int, em_pattern: str, seed: int = 0):
    """Build labelled cal-fold samples. em_pattern controls the signal-em
    relationship (helps verify per-bench composites differ)."""
    import random
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        # Make signals correlated with em differently per pattern
        em = rng.randint(0, 1)
        if em_pattern == "high_p_ground_implies_correct":
            p_ga = rng.uniform(0.6, 1.0) if em == 1 else rng.uniform(0.0, 0.4)
        else:  # "low_p_ground_implies_correct" — inverted
            p_ga = rng.uniform(0.0, 0.4) if em == 1 else rng.uniform(0.6, 1.0)
        samples.append({
            "em": em,
            "u_token": rng.uniform(0.4, 0.9),
            "u_dropout": rng.uniform(0.0, 0.4),
            "u_internal": rng.uniform(0.4, 0.9),
            "s_avg": rng.uniform(0.5, 0.95),
            "h_norm": rng.uniform(0.0, 0.5),
            "p_entail": rng.uniform(0.3, 0.9),
            "p_ground_max": rng.uniform(0.0, 1.0),
            "p_ground_mean": rng.uniform(0.0, 0.8),
            "p_ground_atomic": p_ga,
            "q_a_relevance": rng.uniform(0.3, 0.9),
        })
    return samples


def test_schema_version_default_is_v2():
    from caem.verification.cal_prob_composite import CalProbComposite
    assert CalProbComposite.SCHEMA_VERSION == CalProbComposite.SCHEMA_VERSION_V2
    assert CalProbComposite.SCHEMA_VERSION_V2 == "branchC.2026-05-06"
    assert CalProbComposite.SCHEMA_VERSION_V1 == "branchC.2026-04-25"


def test_fit_per_benchmark_populates_dict():
    from caem.verification.cal_prob_composite import CalProbComposite
    samples_by_bench = {
        "fever": _make_synthetic_samples(120, "high_p_ground_implies_correct", seed=1),
        "triviaqa": _make_synthetic_samples(120, "low_p_ground_implies_correct", seed=2),
        "hotpotqa": _make_synthetic_samples(120, "high_p_ground_implies_correct", seed=3),
    }
    c = CalProbComposite()
    c.fit_per_benchmark(samples_by_bench, fit_boost=False)
    # Pooled global populated
    assert len(c.calibrations) > 0, "pooled global calibration not populated"
    # Per-benchmark dict populated
    assert set(c.per_benchmark.keys()) == {"fever", "triviaqa", "hotpotqa"}, (
        f"per_benchmark missing benchmarks; got {sorted(c.per_benchmark.keys())}"
    )
    # Each child has its own calibrations
    for bm, child in c.per_benchmark.items():
        assert len(child.calibrations) > 0, f"per_benchmark[{bm}] has no signals"


def test_predict_dispatches_by_source_benchmark():
    from caem.verification.cal_prob_composite import CalProbComposite
    samples_by_bench = {
        "fever": _make_synthetic_samples(120, "high_p_ground_implies_correct", seed=1),
        "triviaqa": _make_synthetic_samples(120, "low_p_ground_implies_correct", seed=2),
    }
    c = CalProbComposite()
    c.fit_per_benchmark(samples_by_bench, fit_boost=False)

    # Same signal vector — different per-bench composites should give
    # different predictions (because the em-vs-p_ground_atomic relationship
    # is inverted between fever and triviaqa).
    sig = {
        "u_token": 0.7, "u_dropout": 0.2, "u_internal": 0.7,
        "s_avg": 0.7, "h_norm": 0.2, "p_entail": 0.6,
        "p_ground_max": 0.5, "p_ground_mean": 0.5, "p_ground_atomic": 0.85,
        "q_a_relevance": 0.6,
    }
    p_fever = c.predict(sig, source_benchmark="fever")
    p_tqa = c.predict(sig, source_benchmark="triviaqa")
    p_global = c.predict(sig, source_benchmark=None)
    p_unknown = c.predict(sig, source_benchmark="not_a_real_benchmark")

    # Per-bench dispatch must produce different scores for inverted relationships
    assert abs(p_fever - p_tqa) > 0.05, (
        f"per_benchmark dispatch is silent — fever and triviaqa with inverted "
        f"em-vs-p_ground_atomic gave nearly identical scores: "
        f"{p_fever:.4f} vs {p_tqa:.4f}"
    )
    # Unknown benchmark falls back to the global pooled composite
    assert abs(p_unknown - p_global) < 1e-6, (
        f"unknown source_benchmark should fall back to global; "
        f"got {p_unknown:.4f} vs global {p_global:.4f}"
    )


def test_save_emits_v2_nested_schema():
    from caem.verification.cal_prob_composite import CalProbComposite
    samples_by_bench = {
        "fever": _make_synthetic_samples(120, "high_p_ground_implies_correct"),
        "triviaqa": _make_synthetic_samples(120, "low_p_ground_implies_correct", seed=42),
    }
    c = CalProbComposite()
    c.fit_per_benchmark(samples_by_bench, fit_boost=False)

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "v2_test.json")
        c.save(out_path)
        with open(out_path) as f:
            saved = json.load(f)

        assert saved["schema_version"] == CalProbComposite.SCHEMA_VERSION_V2
        assert "global" in saved
        assert "per_benchmark" in saved
        assert set(saved["per_benchmark"].keys()) == {"fever", "triviaqa"}
        # Each per-benchmark block has the calibration triplet
        for bm in ("fever", "triviaqa"):
            block = saved["per_benchmark"][bm]
            assert "calibrations" in block
            assert "boost_weights" in block
            assert "boost_intercept" in block


def test_load_v2_round_trip():
    from caem.verification.cal_prob_composite import CalProbComposite
    samples_by_bench = {
        "fever": _make_synthetic_samples(120, "high_p_ground_implies_correct"),
        "triviaqa": _make_synthetic_samples(120, "low_p_ground_implies_correct", seed=42),
    }
    c = CalProbComposite()
    c.fit_per_benchmark(samples_by_bench, fit_boost=False)

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "v2_roundtrip.json")
        c.save(out_path)
        loaded = CalProbComposite.load(out_path)

        # Same per-benchmark set
        assert set(loaded.per_benchmark.keys()) == {"fever", "triviaqa"}
        # Same global signals
        assert set(loaded.calibrations.keys()) == set(c.calibrations.keys())
        # Predictions match
        sig = {
            "u_token": 0.7, "u_dropout": 0.2, "u_internal": 0.7,
            "s_avg": 0.7, "h_norm": 0.2, "p_entail": 0.6,
            "p_ground_max": 0.5, "p_ground_mean": 0.5, "p_ground_atomic": 0.85,
            "q_a_relevance": 0.6,
        }
        for bm in ("fever", "triviaqa"):
            assert abs(c.predict(sig, source_benchmark=bm)
                       - loaded.predict(sig, source_benchmark=bm)) < 1e-9, (
                f"v2 round-trip predict mismatch on {bm}"
            )


def test_load_v1_flat_back_compat():
    """Old v1 single-composite JSON files (no per_benchmark) must still load
    and predict correctly via the global fallback path."""
    from caem.verification.cal_prob_composite import CalProbComposite
    samples = _make_synthetic_samples(120, "high_p_ground_implies_correct")
    c = CalProbComposite()
    c.fit(samples, fit_boost=False)

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "v1_flat.json")
        c.save(out_path)
        with open(out_path) as f:
            saved = json.load(f)
        # When per_benchmark is empty, save uses v1 flat schema
        assert saved["schema_version"] == CalProbComposite.SCHEMA_VERSION_V1
        assert "global" not in saved
        assert "per_benchmark" not in saved
        assert "calibrations" in saved  # v1 flat layout

        loaded = CalProbComposite.load(out_path)
        assert len(loaded.per_benchmark) == 0  # nothing per-bench
        assert len(loaded.calibrations) > 0  # has the global

        # Predict on v1-loaded composite still works (no source_benchmark)
        sig = {
            "u_token": 0.7, "u_dropout": 0.2, "u_internal": 0.7,
            "s_avg": 0.7, "h_norm": 0.2, "p_entail": 0.6,
            "p_ground_max": 0.5, "p_ground_mean": 0.5, "p_ground_atomic": 0.85,
            "q_a_relevance": 0.6,
        }
        p = loaded.predict(sig, source_benchmark=None)
        assert 0.0 < p < 1.0, f"v1 loaded predict gave {p}"


def test_load_actual_v1_artifact_if_present():
    """If the cycle-0 v1 composite_calibration.json exists, verify it loads
    cleanly via the back-compat path. Skipped if not present."""
    import os
    from caem.verification.cal_prob_composite import CalProbComposite
    p = "/workspace/caem/outputs/cycle_0/composite_calibration.json"
    if not os.path.isfile(p):
        return  # not present; skip silently
    loaded = CalProbComposite.load(p)
    assert len(loaded.calibrations) > 0, (
        f"v1 cycle-0 composite at {p} loaded zero signals — back-compat broken"
    )
    # Predict-with-no-bench shouldn't error
    sig = {
        "u_token": 0.7, "u_dropout": 0.2, "u_internal": 0.7,
        "s_avg": 0.7, "h_norm": 0.2, "p_entail": 0.6,
        "p_ground_max": 0.5, "p_ground_mean": 0.5, "p_ground_atomic": 0.5,
        "q_a_relevance": 0.6,
    }
    p_legacy = loaded.predict(sig, source_benchmark=None)
    assert 0.0 < p_legacy < 1.0


def test_shrinkage_alpha_blends_per_bench_with_pooled():
    """Phase 1c — shrinkage prior toward pooled fit.

    The per-bench predict at α=0.6 must lie between α=1.0 (pure per-bench)
    and α=0.0 (pure pooled) on a synthetic 3-bench fold where one bench's
    signal direction differs from the pooled trend.
    """
    from caem.verification.cal_prob_composite import CalProbComposite
    samples_by_bench = {
        "benchA": _make_synthetic_samples(300, "high_p_ground_implies_correct", seed=11),
        "benchB": _make_synthetic_samples(300, "low_p_ground_implies_correct", seed=12),
        "benchC": _make_synthetic_samples(300, "high_p_ground_implies_correct", seed=13),
    }

    c_pure = CalProbComposite()
    c_pure.fit_per_benchmark(samples_by_bench, fit_boost=False, shrinkage_alpha=1.0)

    c_shrunk = CalProbComposite()
    c_shrunk.fit_per_benchmark(samples_by_bench, fit_boost=False, shrinkage_alpha=0.6)

    c_pooled = CalProbComposite()
    c_pooled.fit_per_benchmark(samples_by_bench, fit_boost=False, shrinkage_alpha=0.0)

    # Metadata records alpha for downstream tooling
    assert c_pure.metadata["shrinkage_alpha"] == 1.0
    assert c_shrunk.metadata["shrinkage_alpha"] == 0.6
    assert c_pooled.metadata["shrinkage_alpha"] == 0.0

    # Predict at a probe point on each bench; the shrunk prediction must
    # fall (within numerical slack) between the two extremes.
    probe = {
        "u_token": 0.5, "u_dropout": 0.5, "u_internal": 0.5,
        "s_avg": 0.5, "h_norm": 0.5, "p_entail": 0.5,
        "p_ground_max": 0.7, "p_ground_mean": 0.7,
        "p_ground_atomic": 0.7, "q_a_relevance": 0.5,
    }
    for bm in ("benchA", "benchB", "benchC"):
        p_pure = c_pure.predict(probe, source_benchmark=bm)
        p_shrunk = c_shrunk.predict(probe, source_benchmark=bm)
        p_pooled = c_pooled.predict(probe, source_benchmark=bm)
        lo = min(p_pure, p_pooled) - 0.02
        hi = max(p_pure, p_pooled) + 0.02
        assert lo <= p_shrunk <= hi, (
            f"shrinkage out of range on {bm}: pure={p_pure:.3f} "
            f"shrunk={p_shrunk:.3f} pooled={p_pooled:.3f}"
        )


def test_shrinkage_alpha_zero_collapses_to_pooled():
    """At α=0 the per-bench composites predict (essentially) the pooled curve."""
    from caem.verification.cal_prob_composite import CalProbComposite
    samples_by_bench = {
        "benchA": _make_synthetic_samples(200, "high_p_ground_implies_correct", seed=21),
        "benchB": _make_synthetic_samples(200, "high_p_ground_implies_correct", seed=22),
    }
    c = CalProbComposite()
    c.fit_per_benchmark(samples_by_bench, fit_boost=False, shrinkage_alpha=0.0)
    probe = {
        "u_token": 0.5, "u_dropout": 0.5, "u_internal": 0.5,
        "s_avg": 0.5, "h_norm": 0.5, "p_entail": 0.5,
        "p_ground_max": 0.7, "p_ground_mean": 0.7,
        "p_ground_atomic": 0.7, "q_a_relevance": 0.5,
    }
    p_pooled_only = c._predict_pooled(probe)
    p_via_A = c.predict(probe, source_benchmark="benchA")
    p_via_B = c.predict(probe, source_benchmark="benchB")
    # Both per-bench paths should now match the pooled prediction (within
    # interpolation noise — the per-bench knot positions differ from the
    # pooled knot positions, so the recomputed knot_y values give an
    # approximation of pooled, not an exact match).
    assert abs(p_via_A - p_pooled_only) < 0.05, (
        f"α=0 benchA predict {p_via_A:.3f} should approximate pooled {p_pooled_only:.3f}"
    )
    assert abs(p_via_B - p_pooled_only) < 0.05, (
        f"α=0 benchB predict {p_via_B:.3f} should approximate pooled {p_pooled_only:.3f}"
    )


def test_shrinkage_alpha_invalid_rejected():
    """Negative or super-unity alpha must raise ValueError."""
    from caem.verification.cal_prob_composite import CalProbComposite
    samples_by_bench = {
        "benchA": _make_synthetic_samples(120, "high_p_ground_implies_correct", seed=31),
    }
    for bad_alpha in (-0.1, 1.5):
        c = CalProbComposite()
        try:
            c.fit_per_benchmark(samples_by_bench, fit_boost=False, shrinkage_alpha=bad_alpha)
        except ValueError:
            continue
        raise AssertionError(f"shrinkage_alpha={bad_alpha} should have raised ValueError")


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
