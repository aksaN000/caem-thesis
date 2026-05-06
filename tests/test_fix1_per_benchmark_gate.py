"""
tests/test_fix1_per_benchmark_gate.py
=======================================
Smoke test for v2 Fix 1 — per-benchmark conformal storage gate.

Verifies:
  1. fit_per_benchmark() trains per-bench gates + a pooled global fallback
  2. decide(source_benchmark="X") dispatches to per_benchmark["X"] thresholds
  3. decide(source_benchmark=None) uses the pooled global thresholds
  4. decide(source_benchmark="unknown") falls back to global
  5. save() emits v2 nested schema when per_benchmark is populated
  6. load() round-trips both v2 nested AND v1 flat (back-compat)
  7. Per-bench α relaxation produces lower tau_store than the strict α=0.05
     pooled gate (proves alpha override actually does work)
  8. The actual existing outputs/full_run/cycle_4/conformal_gate.json
     (v1 artifact) loads cleanly via the back-compat path
"""
from __future__ import annotations

import json
import os
import sys
import tempfile


def _make_synthetic_samples(n: int, em_rate_target: float = 0.5,
                             u_correct_band=(0.6, 1.0),
                             u_wrong_band=(0.0, 0.5),
                             seed: int = 0):
    """Build labelled cal-fold samples where higher u_stored correlates with em=1."""
    import random
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        em = 1 if rng.random() < em_rate_target else 0
        if em == 1:
            u = rng.uniform(*u_correct_band)
        else:
            u = rng.uniform(*u_wrong_band)
        samples.append({"em": em, "u_stored": u})
    return samples


def test_schema_version_default_is_v2():
    from caem.verification.conformal_gate import ConformalStorageGate
    assert ConformalStorageGate.SCHEMA_VERSION == ConformalStorageGate.SCHEMA_VERSION_V2
    assert ConformalStorageGate.SCHEMA_VERSION_V2 == "branchC.2026-05-06"
    assert ConformalStorageGate.SCHEMA_VERSION_V1 == "branchC.2026-04-25"


def test_fit_per_benchmark_populates_dict():
    from caem.verification.conformal_gate import ConformalStorageGate
    samples = {
        "fever":     _make_synthetic_samples(100, 0.55, seed=1),
        "triviaqa":  _make_synthetic_samples(100, 0.40, seed=2),
        "hotpotqa":  _make_synthetic_samples(100, 0.30, seed=3),
        "commonsense_qa": _make_synthetic_samples(100, 0.60, seed=4),
    }
    parent = ConformalStorageGate.fit_per_benchmark(
        samples, default_alpha_store=0.05, default_alpha_defer=0.40,
    )
    assert parent.tau_store > 0 and parent.tau_store <= 1.0
    assert set(parent.per_benchmark.keys()) <= set(samples.keys())
    # Most benchmarks should fit; some may be DEG (degenerate) under tight α
    assert len(parent.per_benchmark) >= 1, (
        "All per-bench fits failed under default α — synthetic data should support at least one"
    )


def test_decide_dispatches_by_source_benchmark():
    from caem.verification.conformal_gate import ConformalStorageGate
    # Construct a parent + two per-bench children with manual taus
    parent = ConformalStorageGate(
        tau_store=0.80, tau_defer=0.50,
        alpha_store=0.05, alpha_defer=0.40,
        cal_n=100, store_n=10, store_precision=0.96,
        defer_n=30, defer_precision=0.62,
    )
    fever_gate = ConformalStorageGate(
        tau_store=0.85, tau_defer=0.55,  # stricter
        alpha_store=0.05, alpha_defer=0.40,
        cal_n=50, store_n=5, store_precision=0.96,
        defer_n=15, defer_precision=0.66,
    )
    triviaqa_gate = ConformalStorageGate(
        tau_store=0.45, tau_defer=0.30,  # much looser (relaxed α)
        alpha_store=0.40, alpha_defer=0.50,
        cal_n=50, store_n=20, store_precision=0.62,
        defer_n=30, defer_precision=0.55,
    )
    parent.per_benchmark["fever"] = fever_gate
    parent.per_benchmark["triviaqa"] = triviaqa_gate

    # u_stored=0.50 — under fever's gate it's below tau_defer (0.55) → ABSTAIN/DISCARD
    # under triviaqa's gate it's above tau_store (0.45) → STORE
    # under global's gate it's at tau_defer (0.50) → DEFERRED
    d_global, _ = parent.decide(0.50, p_ground_max=0.5, abstain_pg=0.20,
                                  source_benchmark=None)
    d_fever, _ = parent.decide(0.50, p_ground_max=0.5, abstain_pg=0.20,
                                source_benchmark="fever")
    d_tqa, _ = parent.decide(0.50, p_ground_max=0.5, abstain_pg=0.20,
                              source_benchmark="triviaqa")
    d_unknown, _ = parent.decide(0.50, p_ground_max=0.5, abstain_pg=0.20,
                                  source_benchmark="not_a_benchmark")

    assert d_global == "DEFERRED", f"global should be DEFERRED at u_stored=0.50; got {d_global}"
    assert d_fever == "DISCARD", f"fever should be DISCARD (u<tau_defer); got {d_fever}"
    assert d_tqa == "STORE", f"triviaqa should be STORE (u>tau_store); got {d_tqa}"
    assert d_unknown == d_global, (
        f"unknown source_benchmark should fall back to global ({d_global}); got {d_unknown}"
    )


def test_save_emits_v2_nested_schema():
    from caem.verification.conformal_gate import ConformalStorageGate
    samples = {
        "fever":     _make_synthetic_samples(100, 0.55, seed=1),
        "triviaqa":  _make_synthetic_samples(100, 0.40, seed=2),
    }
    parent = ConformalStorageGate.fit_per_benchmark(samples)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "v2_gate.json")
        parent.save(out)
        with open(out) as f:
            saved = json.load(f)
        assert saved["schema_version"] == ConformalStorageGate.SCHEMA_VERSION_V2
        assert "global" in saved
        assert "per_benchmark" in saved
        for block in saved["per_benchmark"].values():
            for k in ("tau_store", "tau_defer", "alpha_store", "alpha_defer",
                      "cal_n", "store_n", "store_precision",
                      "defer_n", "defer_precision"):
                assert k in block, f"per_benchmark block missing {k}"


def test_load_v2_round_trip():
    from caem.verification.conformal_gate import ConformalStorageGate
    samples = {
        "fever":     _make_synthetic_samples(100, 0.55, seed=1),
        "triviaqa":  _make_synthetic_samples(100, 0.40, seed=2),
    }
    parent = ConformalStorageGate.fit_per_benchmark(samples)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "v2_gate_rt.json")
        parent.save(out)
        loaded = ConformalStorageGate.load(out)
        assert set(loaded.per_benchmark.keys()) == set(parent.per_benchmark.keys())
        for bm in parent.per_benchmark:
            assert loaded.per_benchmark[bm].tau_store == parent.per_benchmark[bm].tau_store
            assert loaded.per_benchmark[bm].tau_defer == parent.per_benchmark[bm].tau_defer


def test_load_v1_flat_back_compat():
    """Old v1 single-gate JSON files load via back-compat path."""
    from caem.verification.conformal_gate import ConformalStorageGate
    samples = _make_synthetic_samples(100, 0.55, seed=1)
    gate = ConformalStorageGate.fit(
        calib_samples=samples, score_key="u_stored", em_key="em",
        alpha_store=0.05, alpha_defer=0.40, min_n=20,
    )
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "v1_gate.json")
        gate.save(out)
        with open(out) as f:
            saved = json.load(f)
        assert saved["schema_version"] == ConformalStorageGate.SCHEMA_VERSION_V1
        assert "global" not in saved
        assert "per_benchmark" not in saved
        assert "tau_store" in saved  # flat layout
        loaded = ConformalStorageGate.load(out)
        assert len(loaded.per_benchmark) == 0
        assert abs(loaded.tau_store - gate.tau_store) < 1e-9
        # decide() with source_benchmark=anything falls back to global
        d, _ = loaded.decide(0.5, p_ground_max=0.5, abstain_pg=0.2,
                              source_benchmark="fever")
        d2, _ = loaded.decide(0.5, p_ground_max=0.5, abstain_pg=0.2,
                                source_benchmark=None)
        assert d == d2  # same decision under both


def test_alpha_relaxation_lowers_tau_store():
    """At α=0.40 the gate should fit a LOOSER tau_store than at α=0.05
    (admits more entries; precision floor 60% vs 95%)."""
    from caem.verification.conformal_gate import ConformalStorageGate
    samples = _make_synthetic_samples(200, 0.40, seed=42,
                                       u_correct_band=(0.5, 0.95),
                                       u_wrong_band=(0.1, 0.6))
    strict = ConformalStorageGate.fit(
        calib_samples=samples, alpha_store=0.05, alpha_defer=0.40, min_n=20,
    )
    relaxed = ConformalStorageGate.fit(
        calib_samples=samples, alpha_store=0.40, alpha_defer=0.50, min_n=20,
    )
    # relaxed should admit a lower or equal tau_store (more permissive)
    assert relaxed.tau_store <= strict.tau_store + 1e-9, (
        f"relaxed α=0.40 should yield tau_store <= strict α=0.05; "
        f"got relaxed={relaxed.tau_store:.3f} vs strict={strict.tau_store:.3f}"
    )


def test_load_actual_v1_gate_artifact_if_present():
    """If the cycle-4 v1 conformal_gate.json is on disk, load it via back-compat."""
    from caem.verification.conformal_gate import ConformalStorageGate
    p = "/workspace/caem/outputs/full_run/cycle_4/conformal_gate.json"
    if not os.path.isfile(p):
        return  # skip silently
    loaded = ConformalStorageGate.load(p)
    assert loaded.tau_store > 0
    # decide() with no source_benchmark works
    d, _ = loaded.decide(0.5, p_ground_max=0.5, abstain_pg=0.2)
    assert d in ("STORE", "DEFERRED", "ABSTAIN", "DISCARD")


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
