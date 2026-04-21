"""
scripts/perf_baseline.py
========================
Per-tier perf baseline harness for Branch C Goal 5 (hardware utilization).

Goal
----
Measure CAEM's wall-clock and GPU-utilization behaviour in a reproducible
way so each Goal-5 optimization can be graded against the required
regression gate: **>= 10% improvement on the targeted metric AND no
regression > 2% on any other tracked metric** (per branch_C.md).

The harness is deliberately split into two layers:

* **Statistics layer** (``TimingRecorder``, ``summarise``). Pure Python,
  CPU-testable, no CUDA dependency. Covered by
  ``tests/test_perf_harness.py``.
* **Device sampler** (``GPUSampler``). Reads ``nvidia-smi`` in a polling
  thread. Mockable for unit tests. On a CPU-only CI host the sampler
  still runs and yields empty samples; every metric field reports NaN.

The perf log (``outputs/perf_log.csv``) accumulates one row per harness
run. ``scripts/aggregate_perf.py`` (future) reads this CSV to compute
the per-optimization deltas.

CLI
---
    PYTHONPATH=. python scripts/perf_baseline.py \\
        --label "baseline-pre-goal5" \\
        --n_queries 100 \\
        --batch_sizes 1,8,16,32 \\
        --output_csv outputs/perf_log.csv

In-session smoke (no GPU, no CUDA):
    PYTHONPATH=. python scripts/perf_baseline.py --smoke

CAEM_PROFILE integration
------------------------
This harness itself is always in production mode (``CAEM_PROFILE=0``)
so the measurement is not perturbed by NVTX/profiler overhead. The
per-tier pipeline IT invokes picks up whatever ``CAEM_PROFILE`` the
operator has set -- but the recommended flow for the Goal-5 regression
gate is to set ``CAEM_PROFILE=0`` on the measurement run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Statistics layer (pure Python, CPU-testable)
# =============================================================================

@dataclass
class TimingSummary:
    """Summary statistics for a set of latency observations.

    All fields in milliseconds. NaN values indicate insufficient data
    (e.g., empty observation list) rather than zero, so downstream CSV
    joins don't misinterpret missing measurements as instantaneous.
    """

    label: str
    n: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float


@dataclass
class GPUSummary:
    """GPU-sampler summary over the measurement window."""

    n_samples: int
    mean_util_pct: float
    peak_util_pct: float
    mean_vram_mb: float
    peak_vram_mb: float


@dataclass
class PerfRow:
    """One row in ``outputs/perf_log.csv``."""

    label: str
    timestamp: float
    n_queries: int
    batch_size: int
    timing: TimingSummary
    gpu: GPUSummary
    metadata: Dict[str, str] = field(default_factory=dict)


def _percentile(xs: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. Returns NaN on empty input. ``pct`` in [0, 100]."""
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pct = max(0.0, min(100.0, pct))
    sorted_xs = sorted(xs)
    rank = max(0, min(len(sorted_xs) - 1, int(math.ceil(pct / 100.0 * len(sorted_xs))) - 1))
    return sorted_xs[rank]


def summarise(label: str, observations_ms: Sequence[float]) -> TimingSummary:
    """Compute the canonical latency summary for ``observations_ms``.

    ``observations_ms`` is a sequence of individual latency measurements
    in milliseconds. The returned summary covers count, mean, median,
    p95, p99, min, max, and stdev. Single-observation input returns a
    valid row with stdev = 0.0 (rather than the stdlib's StatisticsError).
    """
    n = len(observations_ms)
    if n == 0:
        nan = float("nan")
        return TimingSummary(
            label=label, n=0,
            mean_ms=nan, median_ms=nan, p95_ms=nan, p99_ms=nan,
            min_ms=nan, max_ms=nan, stdev_ms=nan,
        )
    xs = list(observations_ms)
    return TimingSummary(
        label=label,
        n=n,
        mean_ms=statistics.fmean(xs),
        median_ms=statistics.median(xs),
        p95_ms=_percentile(xs, 95.0),
        p99_ms=_percentile(xs, 99.0),
        min_ms=min(xs),
        max_ms=max(xs),
        stdev_ms=statistics.pstdev(xs) if n > 1 else 0.0,
    )


class TimingRecorder:
    """Per-label latency accumulator.

    Usage::

        rec = TimingRecorder()
        with rec.time("tier2.generate"):
            do_work()
        rec.time_value("tier3.retrieval", 42.1)
        summary = rec.summarise("tier2.generate")
    """

    def __init__(self) -> None:
        self._obs: Dict[str, List[float]] = {}

    def time_value(self, label: str, value_ms: float) -> None:
        """Record a pre-computed latency in milliseconds."""
        self._obs.setdefault(label, []).append(float(value_ms))

    def time(self, label: str):
        """Context manager form: records the wall-clock duration."""
        return _TimedBlock(self, label)

    def observations(self, label: str) -> List[float]:
        return list(self._obs.get(label, []))

    def labels(self) -> List[str]:
        return list(self._obs.keys())

    def summarise(self, label: str) -> TimingSummary:
        return summarise(label, self._obs.get(label, []))

    def summarise_all(self) -> Dict[str, TimingSummary]:
        return {label: self.summarise(label) for label in self._obs}


class _TimedBlock:
    def __init__(self, rec: TimingRecorder, label: str) -> None:
        self._rec = rec
        self._label = label
        self._start: float = 0.0

    def __enter__(self) -> "_TimedBlock":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._rec.time_value(self._label, elapsed_ms)


# =============================================================================
# GPU sampler (nvidia-smi polling; mockable)
# =============================================================================

@dataclass
class _GPUSample:
    util_pct: float
    vram_mb: float


def _default_nvidia_smi_poll() -> Optional[_GPUSample]:
    """Run ``nvidia-smi`` once and return the first GPU's util+VRAM.

    Returns None on any failure (missing nvidia-smi, no GPU, parse error).
    The sampler treats None as "sample unavailable" and skips it.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except Exception:
        return None
    try:
        first = out.decode("utf-8", errors="replace").splitlines()[0]
        util, vram = [t.strip() for t in first.split(",")]
        return _GPUSample(util_pct=float(util), vram_mb=float(vram))
    except Exception:
        return None


class GPUSampler:
    """Background thread that polls GPU utilization + VRAM.

    The polling function is injectable for tests. Start / stop via the
    context manager:

        sampler = GPUSampler(interval_s=0.5)
        with sampler:
            run_workload()
        summary = sampler.summary()
    """

    def __init__(
        self,
        interval_s: float = 0.5,
        poll_fn: Optional[Callable[[], Optional[_GPUSample]]] = None,
    ) -> None:
        self.interval_s = max(0.05, float(interval_s))
        self._poll = poll_fn or _default_nvidia_smi_poll
        self._samples: List[_GPUSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "GPUSampler":
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._poll()
            if sample is not None:
                self._samples.append(sample)
            # Event.wait returns True if set; acts as an interruptible sleep.
            if self._stop.wait(self.interval_s):
                break

    def raw_samples(self) -> List[_GPUSample]:
        return list(self._samples)

    def summary(self) -> GPUSummary:
        if not self._samples:
            nan = float("nan")
            return GPUSummary(
                n_samples=0,
                mean_util_pct=nan, peak_util_pct=nan,
                mean_vram_mb=nan, peak_vram_mb=nan,
            )
        utils = [s.util_pct for s in self._samples]
        vrams = [s.vram_mb for s in self._samples]
        return GPUSummary(
            n_samples=len(self._samples),
            mean_util_pct=statistics.fmean(utils),
            peak_util_pct=max(utils),
            mean_vram_mb=statistics.fmean(vrams),
            peak_vram_mb=max(vrams),
        )


# =============================================================================
# CSV + JSON writers
# =============================================================================

CSV_FIELDS: Tuple[str, ...] = (
    "label", "timestamp", "n_queries", "batch_size",
    "timing_label", "timing_n", "timing_mean_ms", "timing_median_ms",
    "timing_p95_ms", "timing_p99_ms", "timing_min_ms", "timing_max_ms",
    "timing_stdev_ms",
    "gpu_n_samples", "gpu_mean_util_pct", "gpu_peak_util_pct",
    "gpu_mean_vram_mb", "gpu_peak_vram_mb",
    "metadata_json",
)


def _row_dict(label: str, timestamp: float, n_queries: int, batch_size: int,
              timing: TimingSummary, gpu: GPUSummary,
              metadata: Dict[str, str]) -> Dict[str, object]:
    return {
        "label": label,
        "timestamp": timestamp,
        "n_queries": n_queries,
        "batch_size": batch_size,
        "timing_label": timing.label,
        "timing_n": timing.n,
        "timing_mean_ms": timing.mean_ms,
        "timing_median_ms": timing.median_ms,
        "timing_p95_ms": timing.p95_ms,
        "timing_p99_ms": timing.p99_ms,
        "timing_min_ms": timing.min_ms,
        "timing_max_ms": timing.max_ms,
        "timing_stdev_ms": timing.stdev_ms,
        "gpu_n_samples": gpu.n_samples,
        "gpu_mean_util_pct": gpu.mean_util_pct,
        "gpu_peak_util_pct": gpu.peak_util_pct,
        "gpu_mean_vram_mb": gpu.mean_vram_mb,
        "gpu_peak_vram_mb": gpu.peak_vram_mb,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def append_rows(rows: List[PerfRow], csv_path: Path) -> None:
    """Append perf rows to ``csv_path``, writing the header if missing."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if need_header:
            writer.writeheader()
        for row in rows:
            for timing_label, timing in [(row.timing.label, row.timing)] \
                    if isinstance(row.timing, TimingSummary) else []:
                writer.writerow(_row_dict(
                    row.label, row.timestamp, row.n_queries, row.batch_size,
                    timing, row.gpu, row.metadata,
                ))


# =============================================================================
# Smoke runner (CPU-only synthetic workload)
# =============================================================================

def _synthetic_workload(
    n_queries: int, batch_size: int, per_query_ms: float,
) -> TimingRecorder:
    """Deterministic synthetic latencies for the CPU smoke path.

    Emulates N sequential queries each taking ``per_query_ms``. Useful
    for verifying the CSV writer and sampler lifecycle without loading a
    real model.
    """
    rec = TimingRecorder()
    for _ in range(n_queries):
        # Sleep is approximate, so use a fixed recorded value to keep the
        # smoke reproducible under variable host scheduling.
        rec.time_value(f"synthetic.batch{batch_size}", per_query_ms)
    return rec


def run_smoke(csv_path: Path, n_queries: int = 20) -> List[PerfRow]:
    """Run the synthetic CPU smoke and append one row per batch size."""
    rows: List[PerfRow] = []
    for bs in (1, 8, 16, 32):
        rec = _synthetic_workload(n_queries, bs, per_query_ms=12.3)
        gpu_summary = GPUSampler(poll_fn=lambda: None).summary()
        row = PerfRow(
            label=f"smoke-bs{bs}",
            timestamp=time.time(),
            n_queries=n_queries,
            batch_size=bs,
            timing=rec.summarise(f"synthetic.batch{bs}"),
            gpu=gpu_summary,
            metadata={"mode": "smoke", "cpu_only": "true"},
        )
        rows.append(row)
    append_rows(rows, csv_path)
    return rows


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Goal 5 perf-baseline harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--label", default="perf-baseline",
                   help="Run label recorded in each perf_log row.")
    p.add_argument("--n_queries", type=int, default=100,
                   help="Number of queries per batch-size measurement.")
    p.add_argument("--batch_sizes", default="1,8,16,32",
                   help="Comma-separated batch sizes to sweep.")
    p.add_argument("--output_csv", type=Path,
                   default=Path("outputs/perf_log.csv"),
                   help="Perf log CSV (appended).")
    p.add_argument("--smoke", action="store_true",
                   help="Run the synthetic CPU-only smoke (no model load).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ns = _build_parser().parse_args(argv)
    if ns.smoke:
        rows = run_smoke(ns.output_csv, n_queries=ns.n_queries)
        logger.info("Smoke wrote %d rows to %s", len(rows), ns.output_csv)
        return 0
    # The full model-loaded path is Vast-GPU-gated; we leave it as a
    # TODO slot here so the CLI surface is stable.
    logger.error(
        "Live-model perf run is not yet implemented -- re-run with --smoke "
        "for the CPU statistics path. Vast-GPU integration lands with the "
        "Goal 5 perf-tuning sweep.",
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
