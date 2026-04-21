"""
tests/test_compile_drift.py
===========================
Unit tests for the statistics + gate layer of ``scripts/compile_drift_check.py``.

The live-model measurement path is Vast-GPU-gated and runs manually via
the script's CLI. Here we cover the pure-Python classification and
aggregation layers that decide whether a measured drift is SAFE / WARN /
UNSAFE. Keeping the gate logic unit-tested means a future refactor that
silently loosens the thresholds will fail at commit time.
"""

from __future__ import annotations

import pytest

from scripts.compile_drift_check import (
    DriftResult,
    SAFE_THRESHOLD,
    WARN_THRESHOLD,
    classify_drift,
    summarise_drift,
)


# -----------------------------------------------------------------------------
# classify_drift boundary tests
# -----------------------------------------------------------------------------

class TestClassifyDrift:
    def test_well_below_safe_is_safe(self):
        assert classify_drift(1e-5) == "SAFE"
        assert classify_drift(1e-4) == "SAFE"  # documented baseline

    def test_exactly_at_safe_threshold_is_safe(self):
        # safe_threshold = 1e-3: <= is SAFE, > is WARN
        assert classify_drift(SAFE_THRESHOLD) == "SAFE"

    def test_just_above_safe_is_warn(self):
        assert classify_drift(SAFE_THRESHOLD * 1.0001) == "WARN"
        assert classify_drift(5e-3) == "WARN"

    def test_exactly_at_warn_threshold_is_warn(self):
        # warn_threshold = 1e-2: <= is WARN, > is UNSAFE
        assert classify_drift(WARN_THRESHOLD) == "WARN"

    def test_above_warn_is_unsafe(self):
        assert classify_drift(WARN_THRESHOLD * 1.0001) == "UNSAFE"
        assert classify_drift(1e-1) == "UNSAFE"
        assert classify_drift(1.0) == "UNSAFE"

    def test_zero_drift_is_safe(self):
        assert classify_drift(0.0) == "SAFE"

    def test_custom_thresholds(self):
        # A calling context that wants tighter gating can pass override
        # thresholds; the classifier must honour them.
        assert classify_drift(5e-5, safe_threshold=1e-4, warn_threshold=1e-3) == "SAFE"
        assert classify_drift(5e-4, safe_threshold=1e-4, warn_threshold=1e-3) == "WARN"
        assert classify_drift(5e-3, safe_threshold=1e-4, warn_threshold=1e-3) == "UNSAFE"

    def test_thresholds_match_docstring_guard_values(self):
        """Guard against a future edit silently loosening the gate.
        Thresholds MUST match the values documented in the module
        docstring and branch_C.md §Goal 5 -- raising them defeats the
        gate's purpose.
        """
        assert SAFE_THRESHOLD == 1e-3, (
            f"SAFE_THRESHOLD changed from 1e-3 to {SAFE_THRESHOLD}. "
            "If this is intentional, update branch_C.md §Goal 5 and "
            "this test in the same commit."
        )
        assert WARN_THRESHOLD == 1e-2, (
            f"WARN_THRESHOLD changed from 1e-2 to {WARN_THRESHOLD}. "
            "See guidance above."
        )


# -----------------------------------------------------------------------------
# summarise_drift aggregation
# -----------------------------------------------------------------------------

class TestSummariseDrift:
    def test_empty_inputs_return_safe_zero(self):
        r = summarise_drift([], [])
        assert r.n_prompts == 0
        assert r.max_abs_drift == 0.0
        assert r.mean_abs_drift == 0.0
        assert r.gate == "SAFE"

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            summarise_drift([1.0], [1.0, 2.0])

    def test_single_prompt_safe(self):
        r = summarise_drift([1e-5], [5e-6])
        assert r.n_prompts == 1
        assert r.max_abs_drift == pytest.approx(1e-5)
        assert r.mean_abs_drift == pytest.approx(5e-6)
        assert r.gate == "SAFE"

    def test_max_across_prompts(self):
        r = summarise_drift(
            per_prompt_max=[1e-5, 3e-3, 2e-4],   # max = 3e-3 -> WARN
            per_prompt_mean=[1e-6, 1e-4, 5e-5],
        )
        assert r.max_abs_drift == pytest.approx(3e-3)
        assert r.gate == "WARN"

    def test_unsafe_triggers_on_single_outlier(self):
        r = summarise_drift(
            per_prompt_max=[1e-5, 1e-5, 5e-2],   # one UNSAFE prompt fails the gate
            per_prompt_mean=[1e-6, 1e-6, 1e-4],
        )
        assert r.gate == "UNSAFE"
        assert r.max_abs_drift == pytest.approx(5e-2)
        # Mean is still small — aggregation must NOT hide the outlier
        # behind a well-behaved average.
        assert r.mean_abs_drift == pytest.approx(
            (1e-6 + 1e-6 + 1e-4) / 3, rel=1e-3,
        )

    def test_custom_thresholds_propagate(self):
        # Same data, different thresholds -> different gate.
        r_default = summarise_drift([5e-4], [1e-5])
        r_tight = summarise_drift(
            [5e-4], [1e-5],
            safe_threshold=1e-4, warn_threshold=1e-3,
        )
        assert r_default.gate == "SAFE"   # 5e-4 <= 1e-3 default
        assert r_tight.gate == "WARN"     # 5e-4 > 1e-4 tight


# -----------------------------------------------------------------------------
# DriftResult roundtrip
# -----------------------------------------------------------------------------

class TestDriftResultFields:
    def test_dataclass_fields_present(self):
        r = DriftResult(
            n_prompts=3, max_abs_drift=1e-4, mean_abs_drift=5e-5,
            per_prompt_max=[1e-5, 1e-4, 5e-5],
            per_prompt_mean=[1e-6, 5e-6, 1e-5],
            gate="SAFE",
        )
        assert r.n_prompts == 3
        assert r.gate == "SAFE"
        assert len(r.per_prompt_max) == len(r.per_prompt_mean) == 3
