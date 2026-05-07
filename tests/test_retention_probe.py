"""
tests/test_retention_probe.py
===============================
Unit tests for caem.training.retention_probe (v2 Fix 4 — multi-modal
retention probe orchestration).

Coverage
--------
  * register_probe / unregister_probe expose the registry to tests
  * run_retention_probes runs every registered probe and returns a
    {name: float} dict with the right shape
  * Unknown probe names are skipped with a warning (not raised)
  * Probe runner exceptions are caught and recorded as NaN
  * retention_ratios computes current/pristine, propagates NaN on
    missing or zero pristine
  * any_probe_below_tolerance halts on the first finite ratio under
    threshold; ignores NaN entries; returns False when every ratio
    is NaN (guard inactive)
  * worst_probe returns the lowest finite ratio's name, None when all
    NaN
  * CAEMConfig defaults: retention_probes is the v2 triple, n=200
  * SelfImprovementLoop.run_cycle source contains the multi-probe
    branch (static wiring check)
"""
from __future__ import annotations

import math
import sys
from typing import Any


# ---------------------------------------------------------------------- #
# Registry                                                               #
# ---------------------------------------------------------------------- #

def test_register_and_unregister_probe():
    from caem.training.retention_probe import (
        register_probe, unregister_probe, PROBE_REGISTRY,
    )

    def _stub(model, tok, n):
        return 0.5

    register_probe("_smoke_probe", _stub)
    assert "_smoke_probe" in PROBE_REGISTRY
    unregister_probe("_smoke_probe")
    assert "_smoke_probe" not in PROBE_REGISTRY


# ---------------------------------------------------------------------- #
# run_retention_probes orchestration                                     #
# ---------------------------------------------------------------------- #

def test_run_retention_probes_returns_each_registered_probe():
    from caem.training.retention_probe import (
        register_probe, unregister_probe, run_retention_probes,
    )

    def _p1(model, tok, n):
        return 0.80

    def _p2(model, tok, n):
        return 0.65

    register_probe("_p1_test", _p1)
    register_probe("_p2_test", _p2)
    try:
        result = run_retention_probes(
            None, None, probes=["_p1_test", "_p2_test"], n_per_probe=10,
        )
        assert set(result.keys()) == {"_p1_test", "_p2_test"}
        assert abs(result["_p1_test"] - 0.80) < 1e-9
        assert abs(result["_p2_test"] - 0.65) < 1e-9
    finally:
        unregister_probe("_p1_test")
        unregister_probe("_p2_test")


def test_run_retention_probes_skips_unknown_name():
    from caem.training.retention_probe import run_retention_probes
    result = run_retention_probes(
        None, None, probes=["definitely_not_a_real_probe"], n_per_probe=10,
    )
    assert result == {}  # nothing returned for unknown probes


def test_run_retention_probes_records_nan_on_runner_exception():
    from caem.training.retention_probe import (
        register_probe, unregister_probe, run_retention_probes,
    )

    def _bad(model, tok, n):
        raise RuntimeError("simulated probe failure")

    register_probe("_bad_test", _bad)
    try:
        result = run_retention_probes(
            None, None, probes=["_bad_test"], n_per_probe=10,
        )
        assert "_bad_test" in result
        assert math.isnan(result["_bad_test"])
    finally:
        unregister_probe("_bad_test")


# ---------------------------------------------------------------------- #
# retention_ratios                                                       #
# ---------------------------------------------------------------------- #

def test_retention_ratios_basic():
    from caem.training.retention_probe import retention_ratios
    pristine = {"a": 0.80, "b": 0.50, "c": 0.40}
    current = {"a": 0.76, "b": 0.55, "c": 0.20}
    r = retention_ratios(pristine, current)
    assert abs(r["a"] - 0.95) < 1e-9
    assert abs(r["b"] - 1.10) < 1e-9
    assert abs(r["c"] - 0.50) < 1e-9


def test_retention_ratios_nan_pristine_propagates():
    from caem.training.retention_probe import retention_ratios
    pristine = {"a": float("nan"), "b": 0.0, "c": 0.50}
    current = {"a": 0.5, "b": 0.5, "c": 0.5}
    r = retention_ratios(pristine, current)
    assert math.isnan(r["a"])
    assert math.isnan(r["b"])  # zero pristine → NaN ratio
    assert abs(r["c"] - 1.0) < 1e-9


def test_retention_ratios_missing_current():
    from caem.training.retention_probe import retention_ratios
    pristine = {"a": 0.80, "b": 0.50}
    current = {"a": 0.76}
    r = retention_ratios(pristine, current)
    assert abs(r["a"] - 0.95) < 1e-9
    assert math.isnan(r["b"])  # missing current → NaN


# ---------------------------------------------------------------------- #
# any_probe_below_tolerance                                              #
# ---------------------------------------------------------------------- #

def test_any_probe_below_tolerance_fires_on_first_failure():
    from caem.training.retention_probe import any_probe_below_tolerance
    ratios = {"a": 0.96, "b": 0.85, "c": 0.99}
    assert any_probe_below_tolerance(ratios, 0.93) is True  # b fails
    assert any_probe_below_tolerance(ratios, 0.50) is False  # all ok


def test_any_probe_below_tolerance_ignores_nan():
    from caem.training.retention_probe import any_probe_below_tolerance
    ratios = {"a": float("nan"), "b": 0.95, "c": 0.99}
    assert any_probe_below_tolerance(ratios, 0.93) is False
    # NaN-only → no guard active → False
    assert any_probe_below_tolerance({"a": float("nan")}, 0.93) is False


def test_worst_probe_returns_lowest_finite():
    from caem.training.retention_probe import worst_probe
    assert worst_probe({"a": 0.95, "b": 0.80, "c": 0.99}) == "b"
    # With NaN entries, only finite ratios count.
    assert worst_probe({"a": float("nan"), "b": 0.95, "c": 0.99}) == "b"
    # Empty / all-NaN → None
    assert worst_probe({}) is None
    assert worst_probe({"a": float("nan"), "b": float("nan")}) is None


# ---------------------------------------------------------------------- #
# CAEMConfig defaults                                                    #
# ---------------------------------------------------------------------- #

def test_config_retention_probe_defaults():
    from caem.config import CAEMConfig
    cfg = CAEMConfig()
    assert "mmlu" in cfg.retention_probes
    assert "triviaqa_test" in cfg.retention_probes
    assert "hotpotqa_test" in cfg.retention_probes
    assert cfg.retention_probe_n == 200
    assert cfg.forgetting_tolerance == 0.93


# ---------------------------------------------------------------------- #
# Wiring                                                                 #
# ---------------------------------------------------------------------- #

def test_sil_run_cycle_uses_multi_probe_path():
    """SelfImprovementLoop.run_cycle source contains the multi-probe
    orchestration calls."""
    import inspect
    from caem.training.self_improvement import SelfImprovementLoop
    src = inspect.getsource(SelfImprovementLoop.run_cycle)
    assert "run_retention_probes" in src
    assert "retention_ratios" in src
    assert "any_probe_below_tolerance" in src
    assert "self._pristine_probes" in src


def test_cycle_result_carries_probe_dicts():
    """CycleResult exposes probe_retentions and probe_post fields."""
    from caem.training.self_improvement import CycleResult
    cr = CycleResult(
        cycle_num=0, n_episodes_used=0, n_general_used=0,
        epochs_completed=0, final_train_loss=0.0,
        mmlu_retention_ratio=1.0, aborted=False,
        checkpoint_path="x",
    )
    assert hasattr(cr, "probe_retentions")
    assert hasattr(cr, "probe_post")
    # Default factories yield empty dicts.
    assert cr.probe_retentions == {}
    assert cr.probe_post == {}


def test_reset_pristine_mmlu_resets_multi_probe_too():
    """reset_pristine_mmlu must clear BOTH the legacy mmlu anchor and
    the multi-probe pristine dict."""
    import inspect
    from caem.training.self_improvement import SelfImprovementLoop
    src = inspect.getsource(SelfImprovementLoop.reset_pristine_mmlu)
    assert "self._pristine_mmlu = None" in src
    assert "self._pristine_probes = None" in src


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
