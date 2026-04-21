"""
tests/test_perf_harness.py
==========================
Unit tests for ``scripts/perf_baseline.py`` (Branch C Goal 5 perf harness).

Coverage targets the statistics layer + GPU sampler abstraction. The
live-model path is Vast-GPU-gated and exercised by the Cycle-1
regression sweep, not here.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pytest

from scripts.perf_baseline import (
    CSV_FIELDS,
    GPUSampler,
    GPUSummary,
    PerfRow,
    TimingRecorder,
    TimingSummary,
    _GPUSample,
    _percentile,
    append_rows,
    run_smoke,
    summarise,
)


# -----------------------------------------------------------------------------
# _percentile
# -----------------------------------------------------------------------------

class TestPercentile:
    def test_empty_returns_nan(self):
        import math
        assert math.isnan(_percentile([], 50.0))

    def test_single_value(self):
        assert _percentile([42.0], 95.0) == pytest.approx(42.0)

    def test_median_matches(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Nearest-rank median of 5 samples -> index 2 -> 3.0
        assert _percentile(xs, 50.0) == pytest.approx(3.0)

    def test_p100_is_max(self):
        xs = [1.0, 9.0, 5.0, 3.0]
        assert _percentile(xs, 100.0) == pytest.approx(9.0)

    def test_p0_is_min(self):
        xs = [1.0, 9.0, 5.0, 3.0]
        assert _percentile(xs, 0.0) == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# summarise
# -----------------------------------------------------------------------------

class TestSummarise:
    def test_empty_observations_yield_nan(self):
        import math
        s = summarise("empty", [])
        assert s.n == 0
        for f in ("mean_ms", "median_ms", "p95_ms", "p99_ms",
                  "min_ms", "max_ms", "stdev_ms"):
            assert math.isnan(getattr(s, f))

    def test_single_observation(self):
        s = summarise("one", [42.0])
        assert s.n == 1
        assert s.mean_ms == pytest.approx(42.0)
        assert s.median_ms == pytest.approx(42.0)
        assert s.min_ms == pytest.approx(42.0)
        assert s.max_ms == pytest.approx(42.0)
        # stdev on n=1 is 0.0 (not StatisticsError).
        assert s.stdev_ms == pytest.approx(0.0)

    def test_five_observations(self):
        s = summarise("five", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert s.n == 5
        assert s.mean_ms == pytest.approx(3.0)
        assert s.median_ms == pytest.approx(3.0)
        assert s.min_ms == pytest.approx(1.0)
        assert s.max_ms == pytest.approx(5.0)
        assert s.stdev_ms > 0.0


# -----------------------------------------------------------------------------
# TimingRecorder
# -----------------------------------------------------------------------------

class TestTimingRecorder:
    def test_time_value_accumulates(self):
        rec = TimingRecorder()
        rec.time_value("tier2", 10.0)
        rec.time_value("tier2", 20.0)
        rec.time_value("tier3", 30.0)
        assert rec.observations("tier2") == [10.0, 20.0]
        assert rec.observations("tier3") == [30.0]

    def test_time_context_records_duration(self):
        rec = TimingRecorder()
        with rec.time("work"):
            time.sleep(0.002)
        obs = rec.observations("work")
        assert len(obs) == 1
        # Sleep is approximate; lower-bound check only.
        assert obs[0] > 0.0

    def test_time_context_records_on_exception(self):
        rec = TimingRecorder()
        with pytest.raises(RuntimeError, match="boom"):
            with rec.time("broken"):
                raise RuntimeError("boom")
        # Even on exception, the block's duration was recorded.
        assert len(rec.observations("broken")) == 1

    def test_summarise_matches_summarise_function(self):
        rec = TimingRecorder()
        for v in (10.0, 20.0, 30.0):
            rec.time_value("x", v)
        s = rec.summarise("x")
        ref = summarise("x", [10.0, 20.0, 30.0])
        assert s == ref

    def test_summarise_all(self):
        rec = TimingRecorder()
        rec.time_value("a", 1.0)
        rec.time_value("b", 2.0)
        out = rec.summarise_all()
        assert set(out.keys()) == {"a", "b"}


# -----------------------------------------------------------------------------
# GPUSampler
# -----------------------------------------------------------------------------

class TestGPUSampler:
    def test_no_samples_returns_nan_summary(self):
        # Poll function that always returns None -> no samples collected.
        sampler = GPUSampler(interval_s=0.05, poll_fn=lambda: None)
        with sampler:
            time.sleep(0.15)
        summary = sampler.summary()
        assert summary.n_samples == 0
        import math
        assert math.isnan(summary.mean_util_pct)
        assert math.isnan(summary.peak_vram_mb)

    def test_collects_mocked_samples(self):
        values = iter([
            _GPUSample(util_pct=40.0, vram_mb=12000.0),
            _GPUSample(util_pct=80.0, vram_mb=18000.0),
            _GPUSample(util_pct=50.0, vram_mb=15000.0),
            # Continue returning a value so the background thread always
            # has something to yield after join.
        ])
        last = [_GPUSample(util_pct=50.0, vram_mb=15000.0)]

        def _poll():
            try:
                v = next(values)
                last[0] = v
                return v
            except StopIteration:
                return last[0]

        sampler = GPUSampler(interval_s=0.05, poll_fn=_poll)
        with sampler:
            time.sleep(0.3)
        summary = sampler.summary()
        assert summary.n_samples >= 3
        assert summary.peak_util_pct == pytest.approx(80.0)
        assert summary.peak_vram_mb == pytest.approx(18000.0)
        # Mean is between min and max.
        utils = [s.util_pct for s in sampler.raw_samples()]
        assert min(utils) <= summary.mean_util_pct <= max(utils)


# -----------------------------------------------------------------------------
# CSV writer
# -----------------------------------------------------------------------------

class TestCSVWriter:
    def _row(self, label: str = "test") -> PerfRow:
        timing = TimingSummary(
            label="tier2.generate", n=3,
            mean_ms=15.0, median_ms=15.0, p95_ms=20.0, p99_ms=20.0,
            min_ms=10.0, max_ms=20.0, stdev_ms=4.0,
        )
        gpu = GPUSummary(
            n_samples=5, mean_util_pct=60.0, peak_util_pct=80.0,
            mean_vram_mb=12000.0, peak_vram_mb=15000.0,
        )
        return PerfRow(
            label=label, timestamp=1000.0, n_queries=30, batch_size=8,
            timing=timing, gpu=gpu, metadata={"k": "v"},
        )

    def test_header_written_on_empty_file(self, tmp_path: Path):
        csv_path = tmp_path / "perf_log.csv"
        append_rows([self._row()], csv_path)
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert tuple(header) == CSV_FIELDS

    def test_subsequent_appends_skip_header(self, tmp_path: Path):
        csv_path = tmp_path / "perf_log.csv"
        append_rows([self._row("first")], csv_path)
        append_rows([self._row("second")], csv_path)
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Exactly one header row + two data rows.
        assert rows[0] == list(CSV_FIELDS)
        assert len(rows) == 3

    def test_metadata_json_roundtrip(self, tmp_path: Path):
        csv_path = tmp_path / "perf_log.csv"
        append_rows([self._row("x")], csv_path)
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        meta = json.loads(row["metadata_json"])
        assert meta == {"k": "v"}


# -----------------------------------------------------------------------------
# run_smoke end-to-end
# -----------------------------------------------------------------------------

class TestRunSmoke:
    def test_smoke_writes_four_batch_rows(self, tmp_path: Path):
        csv_path = tmp_path / "perf_log.csv"
        rows = run_smoke(csv_path, n_queries=5)
        assert len(rows) == 4
        assert [r.batch_size for r in rows] == [1, 8, 16, 32]
        # CSV has four data rows + one header.
        with csv_path.open("r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 5
