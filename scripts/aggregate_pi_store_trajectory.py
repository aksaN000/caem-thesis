#!/usr/bin/env python
"""
scripts/aggregate_pi_store_trajectory.py
=========================================
Per-cycle storage-precision (π_store) trajectory aggregator.

Computes the cycle-by-cycle empirical precision of the storage class at
the locked cut-point τ_store = 0.60, reading both:

  - Calibration-fold π_store: P(em=1 | decision==STORE) on the fixed
    1500-sample cal fold, recomputed each cycle as the verifier composite
    is refit under the post-fine-tune model.
  - Stream-chunk π_store: P(em=1 | stored==True) on the cycle's SIL
    training pool (~4700 samples/cycle).

These are the empirical receipts for:
  - H1 verdict (calibration-fold storage class precision exceeds chance)
  - thm:purity (memory pool precision lower bound)
  - §sec:adj-locked (locked-configuration empirical anchor)
  - §sec:setup-benchmarks (SIL pool quality claim)

Inputs
------
  outputs/full_run/cycle_<N>/calibration/<bench>_cycle<N>.json
    Per-sample calibration-fold scoring. Uses `decision == "STORE"` to
    identify gate decisions, `em` for correctness.
  outputs/full_run/eval/<bench>_cycle<N>_streamchunk.json
    Per-sample stream-chunk processing records. Uses `stored == True`
    to identify admitted entries, `em` for correctness.
  outputs/cycle_0/weight_validation.json (cycle-0 baseline)
    Memory poisoning rate per benchmark (1 - poisoning_rate = π_store).

Outputs
-------
  outputs/full_run/pi_store_trajectory.csv
    cycle, source (cal_fold | stream_chunk | weight_validation),
    benchmark | "pooled", n_stored, n_correct, pi_store
  thesis_report/figures/auto/tab_pi_store_trajectory.tex
    LaTeX table for Ch5 §sec:adj-locked / §sec:check-purity reference.

Usage
-----
    python -m scripts.aggregate_pi_store_trajectory \\
        --run_dir outputs/full_run \\
        --cycle_0_dir outputs/cycle_0 \\
        --csv_out outputs/full_run/pi_store_trajectory.csv \\
        --tex_out thesis_report/figures/auto/tab_pi_store_trajectory.tex

Idempotent: re-runs against a closed trajectory produce identical
output. Safe to invoke any number of times.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TRAINING_BENCHES = ("fever", "triviaqa", "commonsense_qa")


def _load_cycle_0_pi_store(weight_validation_path: Path) -> List[Dict]:
    """Cycle-0 reading from weight_validation.json memory_poisoning_rate."""
    rows: List[Dict] = []
    if not weight_validation_path.exists():
        logger.warning("cycle-0 weight_validation.json not found at %s", weight_validation_path)
        return rows
    j = json.load(open(weight_validation_path))
    mp = j.get("checks", {}).get("memory_poisoning_rate", {}).get("values", {})
    pooled_n = 0
    pooled_c = 0
    for bench, d in mp.items():
        n_stored = int(d.get("total_stored", 0))
        n_wrong = int(d.get("wrong_stored", 0))
        n_correct = n_stored - n_wrong
        pi = n_correct / n_stored if n_stored > 0 else float("nan")
        rows.append({
            "cycle": 0, "source": "weight_validation", "benchmark": bench,
            "n_stored": n_stored, "n_correct": n_correct, "pi_store": pi,
        })
        if bench in TRAINING_BENCHES or bench == "strategyqa":  # v1 panel pooled
            pooled_n += n_stored
            pooled_c += n_correct
    if pooled_n > 0:
        rows.append({
            "cycle": 0, "source": "weight_validation", "benchmark": "pooled",
            "n_stored": pooled_n, "n_correct": pooled_c,
            "pi_store": pooled_c / pooled_n,
        })
    return rows


def _aggregate_cal_fold(run_dir: Path, cycle: int) -> List[Dict]:
    """Cycle-N cal-fold π_store from cycle_<N>/calibration/<bench>_cycle<N>.json."""
    rows: List[Dict] = []
    pooled_n = 0
    pooled_c = 0
    for bench in TRAINING_BENCHES:
        f = run_dir / f"cycle_{cycle}" / "calibration" / f"{bench}_cycle{cycle}.json"
        if not f.exists():
            continue
        j = json.load(open(f))
        samples = j.get("samples", [])
        stored = [s for s in samples if s.get("decision") == "STORE"]
        if not stored:
            # Fallback: u_stored >= 0.60 threshold (in case decision field absent)
            stored = [s for s in samples if (s.get("u_stored") or 0) >= 0.60]
        if not stored:
            continue
        n = len(stored)
        c_correct = sum(1 for s in stored if (s.get("em") or 0) == 1)
        rows.append({
            "cycle": cycle, "source": "cal_fold", "benchmark": bench,
            "n_stored": n, "n_correct": c_correct, "pi_store": c_correct / n,
        })
        pooled_n += n
        pooled_c += c_correct
    if pooled_n > 0:
        rows.append({
            "cycle": cycle, "source": "cal_fold", "benchmark": "pooled",
            "n_stored": pooled_n, "n_correct": pooled_c,
            "pi_store": pooled_c / pooled_n,
        })
    return rows


def _aggregate_stream_chunk(run_dir: Path, cycle: int) -> List[Dict]:
    """Cycle-N stream-chunk π_store from eval/<bench>_cycle<N>_streamchunk.json."""
    rows: List[Dict] = []
    pooled_n = 0
    pooled_c = 0
    for bench in TRAINING_BENCHES:
        f = run_dir / "eval" / f"{bench}_cycle{cycle}_streamchunk.json"
        if not f.exists():
            continue
        j = json.load(open(f))
        samples = j.get("samples", [])
        stored = [s for s in samples if s.get("stored", False)]
        if not stored:
            continue
        n = len(stored)
        c_correct = sum(1 for s in stored if (s.get("em") or 0) == 1)
        rows.append({
            "cycle": cycle, "source": "stream_chunk", "benchmark": bench,
            "n_stored": n, "n_correct": c_correct, "pi_store": c_correct / n,
        })
        pooled_n += n
        pooled_c += c_correct
    if pooled_n > 0:
        rows.append({
            "cycle": cycle, "source": "stream_chunk", "benchmark": "pooled",
            "n_stored": pooled_n, "n_correct": pooled_c,
            "pi_store": pooled_c / pooled_n,
        })
    return rows


def aggregate(run_dir: Path, cycle_0_dir: Path, max_cycle: int = 10) -> List[Dict]:
    """Full trajectory aggregation."""
    rows: List[Dict] = []
    rows.extend(_load_cycle_0_pi_store(cycle_0_dir / "weight_validation.json"))
    for c in range(1, max_cycle + 1):
        rows.extend(_aggregate_cal_fold(run_dir, c))
        rows.extend(_aggregate_stream_chunk(run_dir, c))
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cycle", "source", "benchmark", "n_stored", "n_correct", "pi_store"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if k == "pi_store" else r[k]) for k in fields})
    logger.info("π_store trajectory CSV -> %s (%d rows)", path, len(rows))


def write_tex(rows: List[Dict], path: Path) -> None:
    """LaTeX table: pooled cal_fold + stream_chunk per cycle side-by-side."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Index by (cycle, source, benchmark)
    idx = {(r["cycle"], r["source"], r["benchmark"]): r for r in rows}
    cycles = sorted(set(r["cycle"] for r in rows))

    lines: List[str] = []
    lines.append("% Auto-generated by scripts/aggregate_pi_store_trajectory.py")
    lines.append("% Per-cycle empirical storage precision π_store at τ_store = 0.60")
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l|cc|cc|cc|cc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Cycle} & \multicolumn{2}{c|}{\textbf{Cal-fold pooled}} & \multicolumn{2}{c|}{\textbf{Cal-fold FEVER}} & \multicolumn{2}{c|}{\textbf{Cal-fold TriviaQA}} & \multicolumn{2}{c}{\textbf{Cal-fold CSQA}} \\")
    lines.append(r"  & $n$ & $\pi_{\text{store}}$ & $n$ & $\pi$ & $n$ & $\pi$ & $n$ & $\pi$ \\")
    lines.append(r"\midrule")
    for c in cycles:
        row_parts = [f"C{c}"]
        for bench in ("pooled", "fever", "triviaqa", "commonsense_qa"):
            r = idx.get((c, "cal_fold", bench)) or idx.get((c, "weight_validation", bench))
            if r:
                row_parts.append(f"${r['n_stored']}$")
                row_parts.append(f"${r['pi_store']:.3f}$")
            else:
                row_parts.append("--")
                row_parts.append("--")
        lines.append(" & ".join(row_parts) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Per-cycle calibration-fold storage precision $\pi_{\text{store}}$ at the locked cut-point $\tau_{\text{store}} = 0.60$. Cycle 0 reads from \texttt{outputs/cycle\_0/weight\_validation.json} (pre-SIL anchor); cycles $\ge 1$ aggregate per-sample \texttt{decision == STORE} ground truth from \texttt{cycle\_N/calibration/}. Aborted cycles (rolled-back fine-tunes) skip the per-cycle composite refit and contribute no new cal-fold reading. Empirical receipt for H1 and \Cref{thm:purity}.}")
    lines.append(r"\label{tab:pi-store-trajectory}")
    lines.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("π_store trajectory TeX -> %s", path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, default=Path("outputs/full_run"))
    p.add_argument("--cycle_0_dir", type=Path, default=Path("outputs/cycle_0"))
    p.add_argument("--max_cycle", type=int, default=10)
    p.add_argument("--csv_out", type=Path, default=Path("outputs/full_run/pi_store_trajectory.csv"))
    p.add_argument("--tex_out", type=Path, default=Path("thesis_report/figures/auto/tab_pi_store_trajectory.tex"))
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(level=ns.log_level, format="%(asctime)s %(levelname)s %(message)s")

    rows = aggregate(ns.run_dir, ns.cycle_0_dir, ns.max_cycle)
    if not rows:
        logger.error("no rows aggregated — are the cycle JSONs present?")
        return 1
    write_csv(rows, ns.csv_out)
    write_tex(rows, ns.tex_out)

    # Print a brief summary to stdout
    print("\nπ_store trajectory summary (cal-fold pooled):")
    print(f"{'cycle':<7}{'n':<8}{'π_store':<10}")
    print("-" * 25)
    for r in rows:
        if r["source"] in ("cal_fold", "weight_validation") and r["benchmark"] == "pooled":
            print(f"c{r['cycle']:<6}{r['n_stored']:<8}{r['pi_store']:<10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
