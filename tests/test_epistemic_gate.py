"""
tests/test_epistemic_gate.py
============================
Unit tests for ``scripts/epistemic_gate.py``.

Coverage focuses on the statistics + decision logic, which is what the gate
depends on for correctness. The live-judge labelling path (MiniCheck re-
scoring) is exercised manually against a real model on Vast; this file
tests the deterministic Python surface only.

Layers tested
-------------
1. ``_rankdata`` average-rank tiebreaker
2. ``_spearman_rho`` on aligned / inverted / uncorrelated / degenerate inputs
3. ``spearman_with_ci`` bootstrap CI ordering + determinism (seeded)
4. ``gate_decision`` three-way threshold mapping at the boundaries
5. ``pair_u_stored_with_labels`` record -> label joining with missing keys
6. ``run_gate`` end-to-end over a synthetic per-sample-signals stream
7. ``report_to_markdown`` renders the required sections
8. ``write_report`` round-trips through JSON + Markdown
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.epistemic_gate import (
    GateReport,
    GateResult,
    _rankdata,
    _spearman_rho,
    gate_decision,
    load_signals_jsonl,
    pair_u_stored_with_labels,
    report_to_markdown,
    run_gate,
    spearman_with_ci,
    write_report,
)


# -----------------------------------------------------------------------------
# _rankdata
# -----------------------------------------------------------------------------

class TestRankdata:
    def test_strict_monotonic(self):
        assert _rankdata([10.0, 20.0, 30.0]) == [1.0, 2.0, 3.0]

    def test_reversed_order_returns_reverse_ranks(self):
        assert _rankdata([30.0, 20.0, 10.0]) == [3.0, 2.0, 1.0]

    def test_ties_get_average_rank(self):
        # Values [1, 1, 2, 3] -> ranks: tied pair averages (1+2)/2 = 1.5
        ranks = _rankdata([1.0, 1.0, 2.0, 3.0])
        assert ranks == [1.5, 1.5, 3.0, 4.0]

    def test_all_tied_gives_same_rank(self):
        ranks = _rankdata([7.0, 7.0, 7.0, 7.0])
        # 4-way tie: everyone gets (1+2+3+4)/4 = 2.5
        assert ranks == [2.5, 2.5, 2.5, 2.5]


# -----------------------------------------------------------------------------
# _spearman_rho
# -----------------------------------------------------------------------------

class TestSpearmanRho:
    def test_perfect_alignment_returns_one(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _spearman_rho(x, y) == pytest.approx(1.0)

    def test_perfect_inversion_returns_minus_one(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _spearman_rho(x, y) == pytest.approx(-1.0)

    def test_uncorrelated_returns_near_zero(self):
        # Random-ish but deterministic small sample.
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 1.0, 4.0, 1.0, 5.0]   # some repetition; not monotonic
        rho = _spearman_rho(x, y)
        assert -1.0 <= rho <= 1.0
        assert abs(rho) < 0.9

    def test_single_element_returns_zero(self):
        assert _spearman_rho([0.5], [0.7]) == 0.0

    def test_empty_returns_zero(self):
        assert _spearman_rho([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _spearman_rho([1.0, 2.0], [1.0])

    def test_constant_axis_returns_zero(self):
        # One axis is constant -> rank variance is zero -> rho undefined.
        assert _spearman_rho([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]) == 0.0

    def test_monotonic_with_ties_positive(self):
        x = [0.1, 0.2, 0.3, 0.3, 0.4]
        y = [1.0, 2.0, 3.0, 3.0, 4.0]
        rho = _spearman_rho(x, y)
        # Identical rank structure -> still perfect correlation.
        assert rho == pytest.approx(1.0, abs=1e-9)


# -----------------------------------------------------------------------------
# spearman_with_ci
# -----------------------------------------------------------------------------

class TestSpearmanWithCI:
    def test_perfect_alignment_ci_includes_one(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = list(x)
        rho, lo, hi = spearman_with_ci(x, y, n_bootstrap=200, seed=0)
        assert rho == pytest.approx(1.0)
        assert lo <= rho <= hi
        # Perfect monotonic pairs -> every bootstrap resample is also perfect.
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_ci_brackets_point_estimate(self):
        # Synthesise a moderately-correlated pair.
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y = [0.1, 0.25, 0.2, 0.5, 0.4, 0.6, 0.9, 0.7, 0.8, 0.95]
        rho, lo, hi = spearman_with_ci(x, y, n_bootstrap=500, seed=42)
        assert lo <= rho <= hi

    def test_deterministic_under_seed(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y = [0.1, 0.25, 0.2, 0.5, 0.4, 0.6, 0.9, 0.7, 0.8, 0.95]
        out1 = spearman_with_ci(x, y, n_bootstrap=300, seed=123)
        out2 = spearman_with_ci(x, y, n_bootstrap=300, seed=123)
        assert out1 == out2

    def test_n_bootstrap_zero_collapses_ci(self):
        x = [0.1, 0.2, 0.3]
        y = [0.3, 0.2, 0.1]
        rho, lo, hi = spearman_with_ci(x, y, n_bootstrap=0)
        assert rho == lo == hi == pytest.approx(-1.0)


# -----------------------------------------------------------------------------
# gate_decision
# -----------------------------------------------------------------------------

class TestGateDecision:
    def test_proceed_above_point_five(self):
        assert gate_decision(0.75) == "PROCEED"
        assert gate_decision(0.51) == "PROCEED"

    def test_honesty_band(self):
        assert gate_decision(0.5) == "PROCEED_WITH_HONESTY"
        assert gate_decision(0.4) == "PROCEED_WITH_HONESTY"
        assert gate_decision(0.31) == "PROCEED_WITH_HONESTY"

    def test_stop_at_or_below_point_three(self):
        assert gate_decision(0.3) == "STOP"
        assert gate_decision(0.1) == "STOP"
        assert gate_decision(0.0) == "STOP"
        assert gate_decision(-0.2) == "STOP"


# -----------------------------------------------------------------------------
# pair_u_stored_with_labels
# -----------------------------------------------------------------------------

class TestPairing:
    def test_joins_on_default_key(self):
        rows = [
            {"u_stored": 0.8, "question": "Q1", "benchmark": "fever"},
            {"u_stored": 0.6, "question": "Q2", "benchmark": "fever"},
            {"u_stored": 0.4, "question": "Q3", "benchmark": "triviaqa"},
        ]
        labels = {"Q1": 1.0, "Q2": 0.0, "Q3": 1.0}
        paired = pair_u_stored_with_labels(rows, labels)
        assert set(paired.keys()) == {"fever", "triviaqa", "pooled"}
        assert paired["fever"][0] == [0.8, 0.6]
        assert paired["fever"][1] == [1.0, 0.0]
        assert paired["pooled"][0] == [0.8, 0.6, 0.4]

    def test_unlabeled_records_dropped(self):
        rows = [
            {"u_stored": 0.8, "question": "Q1", "benchmark": "fever"},
            {"u_stored": 0.5, "question": "orphan", "benchmark": "fever"},
        ]
        labels = {"Q1": 1.0}
        paired = pair_u_stored_with_labels(rows, labels)
        assert paired["fever"][0] == [0.8]
        assert paired["pooled"][0] == [0.8]

    def test_missing_u_stored_dropped(self):
        rows = [{"question": "Q1", "benchmark": "fever"}]
        labels = {"Q1": 1.0}
        paired = pair_u_stored_with_labels(rows, labels)
        # "pooled" still present (as the empty axis).
        assert paired["pooled"] == ([], [])

    def test_missing_benchmark_falls_back_to_unknown(self):
        rows = [{"u_stored": 0.8, "question": "Q1"}]
        labels = {"Q1": 1.0}
        paired = pair_u_stored_with_labels(rows, labels)
        assert "unknown" in paired


# -----------------------------------------------------------------------------
# run_gate
# -----------------------------------------------------------------------------

class TestRunGate:
    def _perfectly_correlated_rows(self):
        xs = [0.1, 0.2, 0.3, 0.4, 0.5]
        return [
            {"u_stored": u, "question": f"Q{i}", "benchmark": "fever"}
            for i, u in enumerate(xs)
        ], {f"Q{i}": u for i, u in enumerate(xs)}

    def test_perfect_correlation_proceeds(self):
        rows, labels = self._perfectly_correlated_rows()
        rep = run_gate(rows, labels, n_bootstrap=100, seed=0)
        assert rep.pooled.rho == pytest.approx(1.0)
        assert rep.pooled.decision == "PROCEED"
        assert rep.overall_decision == "PROCEED"
        # Per-benchmark mirror for the single-benchmark case.
        assert len(rep.per_benchmark) == 1
        assert rep.per_benchmark[0].axis == "fever"
        assert rep.per_benchmark[0].decision == "PROCEED"

    def test_anti_correlated_stops(self):
        rows = [
            {"u_stored": u, "question": f"Q{i}", "benchmark": "fever"}
            for i, u in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])
        ]
        labels = {"Q0": 1.0, "Q1": 0.9, "Q2": 0.5, "Q3": 0.2, "Q4": 0.0}
        rep = run_gate(rows, labels, n_bootstrap=50, seed=0)
        assert rep.pooled.decision == "STOP"

    def test_multi_benchmark_pooling(self):
        # Label values monotonically track u_stored so the pooled rank
        # correlation is exactly 1.0 (no ties across the two benchmarks).
        rows = [
            {"u_stored": 0.90, "question": "Q1", "benchmark": "fever"},
            {"u_stored": 0.10, "question": "Q2", "benchmark": "fever"},
            {"u_stored": 0.80, "question": "Q3", "benchmark": "triviaqa"},
            {"u_stored": 0.20, "question": "Q4", "benchmark": "triviaqa"},
        ]
        labels = {"Q1": 0.95, "Q2": 0.05, "Q3": 0.85, "Q4": 0.15}
        rep = run_gate(rows, labels, n_bootstrap=50, seed=0)
        assert rep.pooled.n == 4
        assert {r.axis for r in rep.per_benchmark} == {"fever", "triviaqa"}
        assert rep.pooled.rho == pytest.approx(1.0)
        assert rep.pooled.decision == "PROCEED"

    def test_multi_benchmark_tied_labels_drop_pooled_rho(self):
        # When labels repeat across benchmarks, rank ties push the pooled
        # rho below 1.0 even though each benchmark internally lines up.
        # This is the statistically-correct Spearman behaviour (see scipy
        # rankdata average-tie convention) -- not a bug. Guards against a
        # future "optimise away ties" regression that would break the Ch5
        # honesty narrative by overstating the correlation.
        rows = [
            {"u_stored": 0.9, "question": "Q1", "benchmark": "fever"},
            {"u_stored": 0.1, "question": "Q2", "benchmark": "fever"},
            {"u_stored": 0.8, "question": "Q3", "benchmark": "triviaqa"},
            {"u_stored": 0.2, "question": "Q4", "benchmark": "triviaqa"},
        ]
        labels = {"Q1": 1.0, "Q2": 0.0, "Q3": 1.0, "Q4": 0.0}
        rep = run_gate(rows, labels, n_bootstrap=50, seed=0)
        assert 0.8 < rep.pooled.rho < 1.0
        assert rep.pooled.decision == "PROCEED"


# -----------------------------------------------------------------------------
# report_to_markdown + write_report
# -----------------------------------------------------------------------------

class TestReporting:
    def _dummy_report(self):
        pooled = GateResult(
            axis="pooled", n=10, rho=0.62, ci_lo=0.48, ci_hi=0.76,
            decision="PROCEED",
        )
        bm = GateResult(
            axis="fever", n=5, rho=0.7, ci_lo=0.5, ci_hi=0.9,
            decision="PROCEED",
        )
        return GateReport(
            pooled=pooled, per_benchmark=[bm],
            n_bootstrap=1000, random_seed=42,
            overall_decision="PROCEED",
        )

    def test_markdown_contains_key_sections(self):
        md = report_to_markdown(self._dummy_report())
        assert "# Epistemic Gate Report" in md
        assert "Bootstrap resamples: 1000" in md
        assert "Pooled decision: PROCEED" in md
        assert "| fever |" in md
        assert "| pooled |" in md
        assert "PROCEED_WITH_HONESTY" in md   # listed in thresholds section

    def test_write_report_roundtrip(self, tmp_path: Path):
        report = self._dummy_report()
        json_path, md_path = write_report(report, tmp_path)
        assert json_path.is_file() and md_path.is_file()
        parsed = json.loads(json_path.read_text())
        assert parsed["overall_decision"] == "PROCEED"
        assert parsed["pooled"]["rho"] == pytest.approx(0.62)
        assert parsed["per_benchmark"][0]["axis"] == "fever"
        md = md_path.read_text()
        assert "PROCEED" in md


# -----------------------------------------------------------------------------
# load_signals_jsonl
# -----------------------------------------------------------------------------

class TestLoadSignals:
    def test_loads_valid_jsonl(self, tmp_path: Path):
        path = tmp_path / "signals.jsonl"
        path.write_text(
            json.dumps({"u_stored": 0.8, "question": "Q1"}) + "\n"
            + json.dumps({"u_stored": 0.6, "question": "Q2"}) + "\n"
        )
        rows = load_signals_jsonl(path)
        assert len(rows) == 2
        assert rows[0]["question"] == "Q1"

    def test_skips_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "signals.jsonl"
        path.write_text(
            json.dumps({"u_stored": 0.8, "question": "Q1"}) + "\n"
            + "{this is not json}\n"
            + json.dumps({"u_stored": 0.6, "question": "Q2"}) + "\n"
        )
        rows = load_signals_jsonl(path)
        assert len(rows) == 2

    def test_ignores_blank_lines(self, tmp_path: Path):
        path = tmp_path / "signals.jsonl"
        path.write_text(
            json.dumps({"u_stored": 0.8, "question": "Q1"}) + "\n\n"
            + json.dumps({"u_stored": 0.6, "question": "Q2"}) + "\n"
        )
        rows = load_signals_jsonl(path)
        assert len(rows) == 2
