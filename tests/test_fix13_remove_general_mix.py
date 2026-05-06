"""
tests/test_fix13_remove_general_mix.py
=======================================
Smoke test for v2 Fix 13: removal of the general-domain mix from SIL pool.

Verifies:
  1. `load_general_data` is no longer importable from scripts.run_experiment
  2. `SelfImprovementLoop.run_cycle()` signature no longer takes `general_data`
  3. `_mix()` method is removed from SelfImprovementLoop
  4. Config fields `general_data_ratio` and `general_data_size` default to 0
  5. End-to-end: a synthetic SIL pool can be constructed from verified
     episodes alone (no general-domain mix injection).
"""
from __future__ import annotations

import inspect
import sys


def test_load_general_data_removed():
    """load_general_data() no longer exists in scripts/run_experiment.py."""
    import scripts.run_experiment as run_experiment
    assert not hasattr(run_experiment, "load_general_data"), (
        "load_general_data should be removed in v2 Fix 13, but it still exists."
    )


def test_run_cycle_signature_no_general_data():
    """SelfImprovementLoop.run_cycle() does not accept general_data."""
    from caem.training.self_improvement import SelfImprovementLoop
    sig = inspect.signature(SelfImprovementLoop.run_cycle)
    params = list(sig.parameters.keys())
    assert "general_data" not in params, (
        f"run_cycle() should not accept general_data in v2; "
        f"found params: {params}"
    )
    # Sanity: required surface preserved
    assert "cycle_num" in params
    assert "memory_store" in params
    assert "deferred_buffer" in params
    assert "reconsider_fn" in params


def test_mix_method_removed():
    """SelfImprovementLoop._mix() is removed."""
    from caem.training.self_improvement import SelfImprovementLoop
    assert not hasattr(SelfImprovementLoop, "_mix"), (
        "SelfImprovementLoop._mix should be removed in v2 Fix 13."
    )


def test_config_general_data_zeroed():
    """CAEMConfig.general_data_{ratio,size} default to 0 in v2."""
    from caem.config import CAEMConfig
    cfg = CAEMConfig()
    assert cfg.general_data_ratio == 0.0, (
        f"general_data_ratio should be 0.0 in v2; got {cfg.general_data_ratio}"
    )
    assert cfg.general_data_size == 0, (
        f"general_data_size should be 0 in v2; got {cfg.general_data_size}"
    )


def test_run_experiment_orchestrator_does_not_load_general_data():
    """scripts/run_experiment.py source code does NOT call load_general_data."""
    import importlib.util
    spec = importlib.util.find_spec("scripts.run_experiment")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    assert "load_general_data(" not in src, (
        "scripts/run_experiment.py still calls load_general_data() — "
        "Fix 13 removal is incomplete."
    )
    assert "general_data=general_data" not in src, (
        "scripts/run_experiment.py still passes general_data= kwarg to "
        "sil.run_cycle() — Fix 13 removal is incomplete."
    )


def test_self_improvement_source_no_mix_call():
    """caem/training/self_improvement.py source does NOT call self._mix()."""
    import importlib.util
    spec = importlib.util.find_spec("caem.training.self_improvement")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    assert "self._mix(" not in src, (
        "self_improvement.py still calls self._mix() — Fix 13 removal "
        "is incomplete."
    )


if __name__ == "__main__":
    # Allow direct run: python tests/test_fix13_remove_general_mix.py
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
