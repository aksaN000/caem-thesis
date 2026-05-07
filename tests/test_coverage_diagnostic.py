"""
tests/test_coverage_diagnostic.py
===================================
Unit tests for caem.diagnostic.coverage (v2 Fix 5 — coverage feedback
diagnostic + halt triggers).

Coverage
--------
  * per_benchmark_admission_rates: cand=0 → NaN; happy-path math
  * pool_composition_entropy: empty → 0; uniform N → ln(N); single
    benchmark → 0
  * adapter_singular_values: returns {} when peft model not available;
    extracts SVs from a real toy LoRA-shaped param dict (mocked
    via a tiny torch.nn.Module)
  * adapter_sv_collapse_score: 1.0 when all mass on top SV, low when
    spread evenly, NaN on empty input
  * HaltDecision: "continue" by default; "halt" on SV collapse;
    "relax_alpha" on N-cycle zero-admission window
  * build_coverage_diagnostic returns the right keys + types
"""
from __future__ import annotations

import math
import sys
from typing import Dict


# ---------------------------------------------------------------------- #
# 1. Admission rates                                                     #
# ---------------------------------------------------------------------- #

def test_admission_rates_zero_candidates_returns_nan():
    from caem.diagnostic.coverage import per_benchmark_admission_rates
    rates = per_benchmark_admission_rates({"fever": 0}, {"fever": 0})
    assert math.isnan(rates["fever"])


def test_admission_rates_happy_path():
    from caem.diagnostic.coverage import per_benchmark_admission_rates
    rates = per_benchmark_admission_rates(
        {"fever": 200, "triviaqa": 300},
        {"fever": 80, "triviaqa": 120},
    )
    assert abs(rates["fever"] - 0.40) < 1e-9
    assert abs(rates["triviaqa"] - 0.40) < 1e-9


def test_admission_rates_admitted_only_benchmark():
    """A benchmark in admitted_counts but not candidate_counts is
    surfaced as 1.0 (every admitted sample passed)."""
    from caem.diagnostic.coverage import per_benchmark_admission_rates
    rates = per_benchmark_admission_rates(
        {"fever": 100}, {"fever": 80, "untracked": 50},
    )
    assert "untracked" in rates
    assert rates["untracked"] == 1.0


# ---------------------------------------------------------------------- #
# 2. Pool entropy                                                        #
# ---------------------------------------------------------------------- #

def test_pool_entropy_empty_zero():
    from caem.diagnostic.coverage import pool_composition_entropy
    assert pool_composition_entropy({}) == 0.0
    assert pool_composition_entropy({"fever": 0}) == 0.0


def test_pool_entropy_uniform_equals_log_n():
    from caem.diagnostic.coverage import pool_composition_entropy
    H = pool_composition_entropy({"a": 100, "b": 100, "c": 100, "d": 100})
    assert abs(H - math.log(4)) < 1e-9


def test_pool_entropy_single_benchmark_zero():
    from caem.diagnostic.coverage import pool_composition_entropy
    assert pool_composition_entropy({"fever": 1000}) == 0.0


def test_pool_entropy_increasing_with_balance():
    from caem.diagnostic.coverage import pool_composition_entropy
    skewed = pool_composition_entropy({"a": 95, "b": 5})
    balanced = pool_composition_entropy({"a": 50, "b": 50})
    assert balanced > skewed


# ---------------------------------------------------------------------- #
# 3. Adapter SVD                                                         #
# ---------------------------------------------------------------------- #

def test_adapter_singular_values_empty_when_no_peft_model():
    from caem.diagnostic.coverage import adapter_singular_values

    class _NoLora:
        def named_parameters(self):
            return iter([])
    assert adapter_singular_values(_NoLora()) == {}


def test_adapter_singular_values_extracts_from_lora_named_params():
    """If a model exposes named_parameters whose names contain
    'lora_A' / 'lora_B' and have 2D weights, the SVD function returns
    the top-k singular values."""
    import torch
    import torch.nn as nn
    from caem.diagnostic.coverage import adapter_singular_values

    class _ToyLora(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Mock named-parameter style by registering parameters
            # whose names match the LoRA naming convention.
            self.lora_A = nn.Parameter(torch.randn(8, 16))
            self.lora_B = nn.Parameter(torch.randn(16, 8))
            self.dense_other = nn.Parameter(torch.randn(16, 16))

    m = _ToyLora()
    spectra = adapter_singular_values(m, top_k=4)
    # Both lora_* params present, dense_other excluded
    assert any("lora_A" in k for k in spectra), (
        f"missing lora_A in {list(spectra.keys())}"
    )
    assert any("lora_B" in k for k in spectra), (
        f"missing lora_B in {list(spectra.keys())}"
    )
    assert all("dense_other" not in k for k in spectra)
    # Top-k cap honoured
    for name, svs in spectra.items():
        assert len(svs) <= 4


# ---------------------------------------------------------------------- #
# 4. SV collapse score                                                   #
# ---------------------------------------------------------------------- #

def test_sv_collapse_score_all_mass_on_top():
    from caem.diagnostic.coverage import adapter_sv_collapse_score
    spectra = {"a": [10.0, 0.0, 0.0, 0.0]}
    assert abs(adapter_sv_collapse_score(spectra) - 1.0) < 1e-9


def test_sv_collapse_score_uniform_spread():
    from caem.diagnostic.coverage import adapter_sv_collapse_score
    # Four equal SVs → top SV is 0.25 of total.
    spectra = {"a": [1.0, 1.0, 1.0, 1.0]}
    assert abs(adapter_sv_collapse_score(spectra) - 0.25) < 1e-9


def test_sv_collapse_score_empty_returns_nan():
    from caem.diagnostic.coverage import adapter_sv_collapse_score
    assert math.isnan(adapter_sv_collapse_score({}))


def test_sv_collapse_score_averages_across_modules():
    from caem.diagnostic.coverage import adapter_sv_collapse_score
    spectra = {"a": [10.0, 0.0, 0.0, 0.0],   # ratio 1.0
                "b": [1.0, 1.0, 1.0, 1.0]}  # ratio 0.25
    score = adapter_sv_collapse_score(spectra)
    assert abs(score - 0.625) < 1e-9


# ---------------------------------------------------------------------- #
# 5. Halt triggers                                                        #
# ---------------------------------------------------------------------- #

def test_halt_continue_when_nothing_wrong():
    from caem.diagnostic.coverage import evaluate_halt_triggers
    diags = [{
        "admission_rates": {"fever": 0.4, "triviaqa": 0.3},
        "adapter_sv_collapse": 0.5,
    }]
    assert evaluate_halt_triggers(diags).action == "continue"


def test_halt_on_sv_collapse():
    from caem.diagnostic.coverage import evaluate_halt_triggers
    diags = [{
        "admission_rates": {"fever": 0.4},
        "adapter_sv_collapse": 0.97,
    }]
    decision = evaluate_halt_triggers(diags, sv_collapse_threshold=0.95)
    assert decision.action == "halt"
    assert "sv_collapse" in decision.reason or "0.97" in decision.reason


def test_halt_relax_alpha_on_zero_admission_window():
    from caem.diagnostic.coverage import evaluate_halt_triggers
    diags = [
        {
            "admission_rates": {"fever": 0.4, "triviaqa": 0.0},
            "adapter_sv_collapse": 0.5,
        },
        {
            "admission_rates": {"fever": 0.5, "triviaqa": 0.0},
            "adapter_sv_collapse": 0.5,
        },
    ]
    decision = evaluate_halt_triggers(
        diags, zero_admission_window=2, sv_collapse_threshold=0.95,
    )
    assert decision.action == "relax_alpha"
    assert "triviaqa" in decision.benchmarks_to_relax


def test_halt_no_relax_when_window_unmet():
    from caem.diagnostic.coverage import evaluate_halt_triggers
    diags = [{
        "admission_rates": {"fever": 0.4, "triviaqa": 0.0},
        "adapter_sv_collapse": 0.5,
    }]  # only 1 cycle — window=2 unmet
    decision = evaluate_halt_triggers(diags, zero_admission_window=2)
    assert decision.action == "continue"


def test_halt_skips_nan_admission_rates():
    """NaN rates (no candidates this cycle) don't count toward the
    zero-admission window."""
    from caem.diagnostic.coverage import evaluate_halt_triggers
    diags = [
        {
            "admission_rates": {"triviaqa": float("nan")},
            "adapter_sv_collapse": 0.5,
        },
        {
            "admission_rates": {"triviaqa": 0.0},
            "adapter_sv_collapse": 0.5,
        },
    ]
    # Only one finite zero in the window → no relax.
    decision = evaluate_halt_triggers(diags, zero_admission_window=2)
    assert decision.action == "continue"


# ---------------------------------------------------------------------- #
# 6. build_coverage_diagnostic                                           #
# ---------------------------------------------------------------------- #

def test_build_coverage_diagnostic_keys_present():
    from caem.diagnostic.coverage import build_coverage_diagnostic
    d = build_coverage_diagnostic(
        cycle_num=3,
        candidate_counts={"fever": 100, "triviaqa": 200},
        admitted_counts={"fever": 30, "triviaqa": 80},
        pool_counts={"fever": 50, "triviaqa": 50},
        per_bench_em={"fever": 0.62, "triviaqa": 0.45},
    )
    expected_keys = {
        "cycle_num", "admission_rates", "candidate_counts",
        "admitted_counts", "pool_counts", "pool_entropy",
        "per_bench_em", "adapter_sv_spectra", "adapter_sv_collapse",
    }
    assert expected_keys.issubset(d.keys())
    assert d["cycle_num"] == 3
    assert abs(d["admission_rates"]["fever"] - 0.30) < 1e-9
    assert abs(d["admission_rates"]["triviaqa"] - 0.40) < 1e-9
    assert abs(d["pool_entropy"] - math.log(2)) < 1e-9
    # No model passed → empty SV spectra and NaN collapse
    assert d["adapter_sv_spectra"] == {}
    assert math.isnan(d["adapter_sv_collapse"])


def test_build_coverage_diagnostic_json_serialisable():
    """The diagnostic must round-trip through json.dumps/loads cleanly."""
    import json
    from caem.diagnostic.coverage import build_coverage_diagnostic
    d = build_coverage_diagnostic(
        cycle_num=0,
        candidate_counts={"fever": 100},
        admitted_counts={"fever": 30},
        pool_counts={"fever": 50},
    )
    # NaN is not strictly JSON-spec but Python json with default
    # encoder handles it (allow_nan=True default). The point is no
    # tensor / non-serialisable object slips through.
    s = json.dumps(d)
    back = json.loads(s)
    assert back["cycle_num"] == 0


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
