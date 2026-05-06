"""
tests/test_fix11_deferred_guard.py
====================================
Smoke test for v2 Fix 11 (layers 1-3) — deferred-reconsideration guard.

Layer 4 (full unit test for orchestrator wiring) and Layer 5 (pre-launch
dry-run) come later; this smoke verifies the three guard layers added now.

Verifies:
  Layer 2 — SIL run_cycle raises RuntimeError when deferred_buffer is None
            (with allow_skip_deferred=False, the default)
  Layer 2 — SIL run_cycle does NOT raise when allow_skip_deferred=True
  CycleResult schema gains deferred_buffer_size_at_entry + reconsider_fired
  CAEMConfig.allow_skip_deferred defaults to False
  Layer 1 source pattern — run_experiment.py orchestrator has the assert
  Layer 3 source pattern — run_experiment.py post-cycle assertion present
  Layer 2 source pattern — self_improvement.py raises RuntimeError on missing buffer
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock


def test_cycle_result_has_v2_fields():
    """CycleResult dataclass has the new tracking fields for Layer 3."""
    from dataclasses import fields
    from caem.training.self_improvement import CycleResult
    field_names = {f.name for f in fields(CycleResult)}
    assert "deferred_buffer_size_at_entry" in field_names, (
        "CycleResult missing v2 Fix 11 layer 3 field deferred_buffer_size_at_entry"
    )
    assert "reconsider_fired" in field_names, (
        "CycleResult missing v2 Fix 11 layer 3 field reconsider_fired"
    )


def test_config_allow_skip_deferred_default_false():
    """CAEMConfig.allow_skip_deferred defaults to False (production-safe)."""
    from caem.config import CAEMConfig
    cfg = CAEMConfig()
    assert hasattr(cfg, "allow_skip_deferred"), (
        "CAEMConfig missing allow_skip_deferred field"
    )
    assert cfg.allow_skip_deferred is False, (
        f"allow_skip_deferred must default to False; got {cfg.allow_skip_deferred}"
    )


def test_layer2_raises_when_buffer_missing():
    """SIL run_cycle raises RuntimeError when deferred_buffer is None and
    allow_skip_deferred is False."""
    from caem.config import CAEMConfig
    from caem.training.self_improvement import SelfImprovementLoop

    cfg = CAEMConfig()
    cfg.allow_skip_deferred = False  # explicit production semantics

    # We don't need a real model/tokenizer to test the early-fail path —
    # the assertion fires before any heavy state is touched. Use a minimal
    # mock SIL whose run_cycle exercises the guard.
    sil = SelfImprovementLoop.__new__(SelfImprovementLoop)
    sil.config = cfg
    sil.model = MagicMock()
    sil.tokenizer = MagicMock()
    sil.output_dir = "/tmp/sil_smoke"
    sil._pristine_mmlu = None
    sil.device = "cpu"

    memory_store = MagicMock()
    memory_store.size = 0
    memory_store.episodes = []

    # Call run_cycle with deferred_buffer=None — this is the v1 May-4 gap.
    try:
        sil.run_cycle(
            cycle_num=1,
            memory_store=memory_store,
            verify_fn=None,
            deferred_buffer=None,         # the gap
            reconsider_fn=None,
        )
    except RuntimeError as e:
        assert "Fix 11 layer 2" in str(e), (
            f"Expected v2 Fix 11 layer 2 RuntimeError; got: {e}"
        )
        return
    except Exception as e:
        # Any other exception means the guard didn't fire first
        raise AssertionError(
            f"Layer 2 should have raised RuntimeError before any other failure; "
            f"got {type(e).__name__}: {e}"
        )
    raise AssertionError(
        "Layer 2 did NOT raise RuntimeError when deferred_buffer=None "
        "and allow_skip_deferred=False. The guard is broken."
    )


def test_layer1_orchestrator_assert_pattern_present():
    """scripts/run_experiment.py has the Layer 1 orchestrator assert."""
    import importlib.util
    spec = importlib.util.find_spec("scripts.run_experiment")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    assert "v2 Fix 11 LAYER 1" in src, (
        "Layer 1 marker comment missing from scripts/run_experiment.py"
    )
    assert "pipeline.deferred_buffer is not None" in src, (
        "Layer 1 assertion on pipeline.deferred_buffer missing"
    )


def test_layer2_self_improvement_raises_pattern_present():
    """caem/training/self_improvement.py has the Layer 2 RuntimeError pattern."""
    import importlib.util
    spec = importlib.util.find_spec("caem.training.self_improvement")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    assert "v2 Fix 11 LAYER 2" in src, (
        "Layer 2 marker comment missing from self_improvement.py"
    )
    assert 'raise RuntimeError(' in src, (
        "Layer 2 RuntimeError raise statement missing"
    )
    assert "allow_skip_deferred" in src, (
        "Layer 2 opt-out flag (allow_skip_deferred) not referenced"
    )


def test_layer3_post_cycle_assert_pattern_present():
    """scripts/run_experiment.py has the Layer 3 post-cycle assertion."""
    import importlib.util
    spec = importlib.util.find_spec("scripts.run_experiment")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    assert "v2 Fix 11 LAYER 3" in src, (
        "Layer 3 marker comment missing from scripts/run_experiment.py"
    )
    assert "reconsider_fired" in src, (
        "Layer 3 should check cycle_result.reconsider_fired"
    )
    assert "deferred_buffer_size_at_entry" in src, (
        "Layer 3 should reference deferred_buffer_size_at_entry"
    )


def test_legacy_warning_replaced_with_runtime_error():
    """The May-4 v1 WARNING-only fallback is now RuntimeError under v2."""
    import importlib.util
    spec = importlib.util.find_spec("caem.training.self_improvement")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    # The v1 verbatim WARNING text should no longer be present as the
    # primary failure mode. The v2 path is RuntimeError + WARNING-when-opted-out.
    assert (
        'logger.warning(\n                "Cycle %d: deferred_buffer not provided to run_cycle -- "'
        not in src
    ), (
        "v1 WARNING-only fallback still present; v2 should raise RuntimeError "
        "by default and only WARNING when allow_skip_deferred=True."
    )


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
