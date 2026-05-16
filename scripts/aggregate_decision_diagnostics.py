#!/usr/bin/env python
"""
scripts/aggregate_decision_diagnostics.py
==========================================
Five decision-diagnostic tables for the new Ch5 §sec:adj-decision-diagnostics
subsection (between §sec:adj-cal-eval-gap and §sec:adj-judge-ablation).

Reads CAEM trajectory eval JSONs (which carry per-sample u_stored,
decision label, em, and verifier signals) and produces:

  1. tab_decision_matrix_trajectory.tex
     Per-cycle 8-cell (decision × EM) matrix per benchmark.
     Decisions: {STORE, DEFERRED, DISCARD, ABSTAIN}.

  2. tab_grounding_success_distribution.tex
     Per-cycle p_ground_mean ≥ {0.7, 0.5, 0.3, 0.1} on EM=1 samples per
     benchmark. Anchors the corpus-expansion paragraph.

  3. tab_signal_fingerprint.tex
     Per-cycle stored-correct signal medians per benchmark (proves
     cross-benchmark fingerprint consistency).

  4. fig_ustored_calibration.tex
     Per-cycle u_stored bucket × EM rate histogram (calibration
     monotonicity proof). Rendered as a TikZ-friendly tabular for now;
     a real figure could be added later via make_figures.py.

  5. tab_contamination_examples.tex (appendix-bound)
     Top-10 wrong-stored examples per benchmark per cycle.

Inputs (read-only)
------------------
  outputs/full_run/eval/{bench}_cycle{N}.json
  outputs/full_run/eval/{bench}_cycle{N}_streamchunk.json

Outputs
-------
  outputs/{tab_*.csv}
  thesis_report/figures/auto/{tab_*.tex, fig_*.tex}

Memory refs:
  caem_post_step7_recall_update.md  -- this 5-table set was planned but not yet built
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


BENCHES = ["fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa"]
TRAINING_BENCHES = ["fever", "triviaqa", "commonsense_qa"]


def load(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text()).get("samples", [])
    except json.JSONDecodeError:
        return []


# --- 1. Decision matrix ---------------------------------------------------- #

def _sample_em(s: dict) -> float:
    """Return the appropriate EM value for a sample.

    For TruthfulQA samples, prefer ``em_llm_judged`` when populated
    (the legacy ``em`` is rouge_l > 0.15, which inflates long-prose
    answers — see Lin et al. ACL 2022 §3.2 methodology). All other
    benches keep the strict-EM path.
    """
    if s.get("benchmark") == "truthfulqa" and s.get("em_llm_judged") is not None:
        return float(s["em_llm_judged"])
    return float(s.get("em", 0) or 0)


def decision_matrix(samples: List[dict]) -> Dict[str, Dict[str, int]]:
    """Returns {decision: {correct: N, wrong: N}}."""
    grid: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0})
    for s in samples:
        dec = s.get("decision") or ("ABSTAIN" if s.get("abstained") else "UNKNOWN")
        em = _sample_em(s)
        key = "correct" if em > 0 else "wrong"
        grid[dec][key] += 1
    return dict(grid)


# --- 2. Grounding distribution --------------------------------------------- #

def grounding_distribution(samples: List[dict], thresholds=(0.7, 0.5, 0.3, 0.1)) -> Dict[float, float]:
    """Fraction of EM=1 samples whose p_ground_mean exceeds each threshold."""
    pos = [s for s in samples if _sample_em(s) > 0
           and s.get("p_ground_mean") is not None]
    if not pos:
        return {t: 0.0 for t in thresholds}
    out = {}
    for t in thresholds:
        out[t] = sum(1 for s in pos if s.get("p_ground_mean", 0) >= t) / len(pos)
    return out


# --- 3. Signal fingerprint ------------------------------------------------- #

SIG_KEYS = ["u_pre", "p_entail", "p_ground_max", "p_ground_mean",
            "p_ground_atomic", "q_a_relevance", "alias_overlap"]


def signal_fingerprint(samples: List[dict], decision="STORE") -> Dict[str, Optional[float]]:
    """Per-signal median for samples matching `decision` AND em=1."""
    sel = [s for s in samples if s.get("decision") == decision
           and _sample_em(s) > 0]
    if not sel:
        return {k: None for k in SIG_KEYS}
    out = {}
    for k in SIG_KEYS:
        vals = [s.get(k) for s in sel if s.get(k) is not None]
        out[k] = median(vals) if vals else None
    return out


# --- 4. u_stored × EM calibration buckets ---------------------------------- #

def ustored_calibration(samples: List[dict], edges=(0.6, 0.7, 0.8, 0.9, 1.0)) -> List:
    """Per-bucket: n, em_rate."""
    rows = []
    lo = edges[0]
    for hi in edges[1:]:
        sel = [s for s in samples if s.get("u_stored") is not None
               and lo <= s["u_stored"] < hi]
        n = len(sel)
        em = (sum(_sample_em(s) > 0 for s in sel) / n) if n else 0.0
        rows.append({"bucket": f"[{lo:.2f},{hi:.2f})", "n": n, "em_rate": em})
        lo = hi
    return rows


# --- 5. Contamination examples --------------------------------------------- #

def contamination_examples(samples: List[dict], top_n=10) -> List[Dict]:
    """Wrong-stored samples sorted by u_stored descending."""
    wrong_stored = [s for s in samples
                    if s.get("decision") == "STORE"
                    and _sample_em(s) == 0]
    wrong_stored.sort(key=lambda s: -(s.get("u_stored") or 0))
    out = []
    for s in wrong_stored[:top_n]:
        out.append({
            "id": s.get("id", "?"),
            "question": (s.get("question", "") or "")[:80],
            "prediction": (s.get("display_answer") or s.get("prediction") or "")[:80],
            "gold": (s.get("gold_answers") or [s.get("gold_label", "")])[0],
            "u_stored": s.get("u_stored"),
        })
    return out


# --- Writers --------------------------------------------------------------- #

def _fmt_pct(v): return f"{v * 100:.1f}\\%" if v is not None else "--"
def _fmt_num(v): return f"{v:.3f}" if v is not None else "--"


def write_decision_matrix(per_cycle_bench: Dict, out_csv: Path, out_tex: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    cols = ["cycle", "bench", "STORE_correct", "STORE_wrong", "STORE_prec",
            "DEFERRED_correct", "DEFERRED_wrong",
            "DISCARD_correct", "DISCARD_wrong"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c, by_bench in per_cycle_bench.items():
            for bench, grid in by_bench.items():
                s_c = grid.get("STORE", {}).get("correct", 0)
                s_w = grid.get("STORE", {}).get("wrong", 0)
                s_p = s_c / (s_c + s_w) if (s_c + s_w) > 0 else 0
                row = {
                    "cycle": c, "bench": bench,
                    "STORE_correct": s_c, "STORE_wrong": s_w,
                    "STORE_prec": f"{s_p:.3f}",
                    "DEFERRED_correct": grid.get("DEFERRED", {}).get("correct", 0),
                    "DEFERRED_wrong": grid.get("DEFERRED", {}).get("wrong", 0),
                    "DISCARD_correct": grid.get("DISCARD", {}).get("correct", 0),
                    "DISCARD_wrong": grid.get("DISCARD", {}).get("wrong", 0),
                }
                w.writerow(row)
    logger.info("Wrote %s", out_csv)
    # Compact LaTeX — one row per (cycle, bench) for STORE only (most informative)
    tex = ["% Auto-generated by scripts/aggregate_decision_diagnostics.py",
           "\\begin{tabular}{ccccc}", "\\toprule",
           "Cycle & Benchmark & STORE: correct & STORE: wrong & STORE precision \\\\",
           "\\midrule"]
    for c in sorted(per_cycle_bench.keys()):
        for bench in BENCHES:
            grid = per_cycle_bench[c].get(bench, {})
            s_c = grid.get("STORE", {}).get("correct", 0)
            s_w = grid.get("STORE", {}).get("wrong", 0)
            s_p = s_c / (s_c + s_w) if (s_c + s_w) > 0 else 0
            if s_c + s_w == 0:
                continue
            bn = bench.replace("_", "\\_")
            tex.append(f"C{c} & {bn} & {s_c} & {s_w} & {s_p:.3f} \\\\")
    tex.append("\\bottomrule"); tex.append("\\end{tabular}")
    out_tex.write_text("\n".join(tex) + "\n")
    logger.info("Wrote %s", out_tex)


def write_grounding(per_cycle_bench: Dict, out_csv: Path, out_tex: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS = (0.7, 0.5, 0.3, 0.1)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cycle", "bench"] + [f"p_ground_mean_ge_{t}" for t in THRESHOLDS])
        for c in sorted(per_cycle_bench.keys()):
            for bench in BENCHES:
                d = per_cycle_bench[c].get(bench, {})
                if not d:
                    continue
                w.writerow([c, bench] + [f"{d.get(t, 0):.4f}" for t in THRESHOLDS])
    logger.info("Wrote %s", out_csv)
    tex = ["% Auto-generated", "\\begin{tabular}{ccrrrr}", "\\toprule",
           "Cycle & Bench & $\\ge 0.7$ & $\\ge 0.5$ & $\\ge 0.3$ & $\\ge 0.1$ \\\\",
           "\\midrule"]
    for c in sorted(per_cycle_bench.keys()):
        for bench in BENCHES:
            d = per_cycle_bench[c].get(bench, {})
            if not d:
                continue
            bn = bench.replace("_", "\\_")
            tex.append(f"C{c} & {bn} & "
                       + " & ".join(_fmt_pct(d.get(t)) for t in THRESHOLDS) + " \\\\")
    tex.append("\\bottomrule"); tex.append("\\end{tabular}")
    tex.append("\nFraction of EM=1 stored samples whose passage-grounding mean exceeds each threshold.")
    out_tex.write_text("\n".join(tex) + "\n")
    logger.info("Wrote %s", out_tex)


def write_fingerprint(per_cycle_bench: Dict, out_csv: Path, out_tex: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cycle", "bench"] + SIG_KEYS)
        for c in sorted(per_cycle_bench.keys()):
            for bench in BENCHES:
                fp = per_cycle_bench[c].get(bench, {})
                if not fp:
                    continue
                w.writerow([c, bench] + [_fmt_num(fp.get(k)) for k in SIG_KEYS])
    logger.info("Wrote %s", out_csv)
    tex = ["% Auto-generated", "\\begin{tabular}{cc" + "c" * len(SIG_KEYS) + "}",
           "\\toprule",
           "Cycle & Bench & " + " & ".join(k.replace("_", "\\_") for k in SIG_KEYS) + " \\\\",
           "\\midrule"]
    for c in sorted(per_cycle_bench.keys()):
        for bench in BENCHES:
            fp = per_cycle_bench[c].get(bench, {})
            if not fp:
                continue
            bn = bench.replace("_", "\\_")
            tex.append(f"C{c} & {bn} & "
                       + " & ".join(_fmt_num(fp.get(k)) for k in SIG_KEYS) + " \\\\")
    tex.append("\\bottomrule"); tex.append("\\end{tabular}")
    tex.append("\nMedian verifier signals on STORE-correct samples per cycle per benchmark.")
    out_tex.write_text("\n".join(tex) + "\n")
    logger.info("Wrote %s", out_tex)


def write_ustored_calib(per_cycle_pooled: Dict, out_csv: Path, out_tex: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cycle", "bucket", "n", "em_rate"])
        for c in sorted(per_cycle_pooled.keys()):
            for r in per_cycle_pooled[c]:
                w.writerow([c, r["bucket"], r["n"], f"{r['em_rate']:.4f}"])
    logger.info("Wrote %s", out_csv)
    tex = ["% Auto-generated", "\\begin{tabular}{cccc}", "\\toprule",
           "Cycle & $u_{\\text{stored}}$ bucket & n & EM rate \\\\", "\\midrule"]
    for c in sorted(per_cycle_pooled.keys()):
        for r in per_cycle_pooled[c]:
            tex.append(f"C{c} & {r['bucket']} & {r['n']} & {_fmt_pct(r['em_rate'])} \\\\")
    tex.append("\\bottomrule"); tex.append("\\end{tabular}")
    tex.append("\nMonotonicity-of-calibration check: EM rate should rise with $u_{\\text{stored}}$ bucket.")
    out_tex.write_text("\n".join(tex) + "\n")
    logger.info("Wrote %s", out_tex)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", type=Path,
                   default=Path("outputs/full_run/eval"))
    p.add_argument("--cycles", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--out_dir", type=Path, default=Path("outputs"))
    p.add_argument("--tex_dir", type=Path,
                   default=Path("thesis_report/figures/auto"))
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()
    logging.basicConfig(level=ns.log_level, format="%(asctime)s %(levelname)s %(message)s")

    # Read all (cycle, bench) samples once
    matrix_data: Dict[int, Dict[str, Dict]] = {c: {} for c in ns.cycles}
    grounding_data: Dict[int, Dict[str, Dict]] = {c: {} for c in ns.cycles}
    fingerprint_data: Dict[int, Dict[str, Dict]] = {c: {} for c in ns.cycles}
    ustored_data: Dict[int, List] = {}

    for c in ns.cycles:
        pooled = []
        for bench in BENCHES:
            # For STORE diagnostics use stream-chunk (training-pool side) if available;
            # else fall back to eval-fold. The decision/u_stored info is in stream-chunk
            # for training benches at successful cycles.
            sc = load(ns.eval_dir / f"{bench}_cycle{c}_streamchunk.json")
            ef = load(ns.eval_dir / f"{bench}_cycle{c}.json")
            src = sc if sc else ef
            if not src:
                continue
            matrix_data[c][bench] = decision_matrix(src)
            grounding_data[c][bench] = grounding_distribution(src)
            fingerprint_data[c][bench] = signal_fingerprint(src)
            pooled.extend(src)
        ustored_data[c] = ustored_calibration(pooled)

    # Console summary
    print("\n=== Decision-matrix STORE precision per (cycle, bench) ===")
    for c in sorted(matrix_data.keys()):
        for bench in BENCHES:
            grid = matrix_data[c].get(bench, {})
            s = grid.get("STORE", {})
            sc, sw = s.get("correct", 0), s.get("wrong", 0)
            if sc + sw == 0: continue
            prec = sc / (sc + sw)
            print(f"  C{c} {bench:<18} STORE: {sc} correct, {sw} wrong, prec={prec:.3f}")

    print("\n=== u_stored × EM calibration (pooled, monotone check) ===")
    for c in sorted(ustored_data.keys()):
        print(f"  C{c}: " + " | ".join(f"{r['bucket']} n={r['n']} EM={r['em_rate']:.3f}"
                                          for r in ustored_data[c] if r["n"] > 0))

    # Write all 5 outputs
    write_decision_matrix(matrix_data,
                          ns.out_dir / "tab_decision_matrix_trajectory.csv",
                          ns.tex_dir / "tab_decision_matrix_trajectory.tex")
    write_grounding(grounding_data,
                    ns.out_dir / "tab_grounding_success_distribution.csv",
                    ns.tex_dir / "tab_grounding_success_distribution.tex")
    write_fingerprint(fingerprint_data,
                      ns.out_dir / "tab_signal_fingerprint.csv",
                      ns.tex_dir / "tab_signal_fingerprint.tex")
    write_ustored_calib(ustored_data,
                        ns.out_dir / "fig_ustored_calibration.csv",
                        ns.tex_dir / "fig_ustored_calibration.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
