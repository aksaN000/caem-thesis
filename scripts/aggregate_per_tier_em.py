#!/usr/bin/env python
"""
scripts/aggregate_per_tier_em.py
=================================
Per-tier EM decomposition for CAEM cycles (Ch5 §5.3.3).

Reports each CAEM cycle's EM broken down by which tier handled the
query (T1 memory / T2 direct generation / T3 RAG), plus the per-tier
sample count and CAEM's pooled EM for reference.

This is the diagnostic table that reveals the CSQA selection effect:
T2 EM is consistently higher than T3 EM for CSQA, yet the router still
sends most CSQA queries to T3 due to the uniform safety_u_pre_min
threshold. The per-tier breakdown surfaces this in one glance.

Inputs (read-only)
------------------
  outputs/full_run/eval/<bench>_cycle{N}.json   for cycles 0..5

Outputs
-------
  outputs/tab_per_tier_em.csv                           -- raw data
  thesis_report/figures/auto/tab_per_tier_em.tex        -- LaTeX

Memory refs:
  caem_per_tier_em_selection_effect.md  -- lead with per-tier EM in Ch5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


BENCHES = ["fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa"]
TIERS = [1, 2, 3]


def load_samples(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text()).get("samples", [])
    except json.JSONDecodeError:
        logger.warning("Could not parse %s", path)
        return []


def per_tier_em(samples: List[dict]) -> Dict[int, Dict[str, Optional[float]]]:
    """Return {tier: {em, n, share}} keyed by tier."""
    if not samples:
        return {t: {"em": None, "n": 0, "share": 0.0} for t in TIERS}
    total = len(samples)
    out = {}
    for t in TIERS:
        sub = [s for s in samples if s.get("tier") == t]
        if not sub:
            out[t] = {"em": None, "n": 0, "share": 0.0}
            continue
        # For TruthfulQA prefer em_llm_judged
        if samples[0].get("benchmark") == "truthfulqa":
            judged = [s.get("em_llm_judged") for s in sub
                      if s.get("em_llm_judged") is not None]
            if judged:
                em = sum(judged) / len(judged)
            else:
                em = sum(s.get("em", 0.0) for s in sub) / len(sub)
        else:
            em = sum(s.get("em", 0.0) for s in sub) / len(sub)
        out[t] = {"em": em, "n": len(sub), "share": len(sub) / total}
    return out


def collect_cycle(eval_dir: Path, cycle: int) -> Dict[str, Dict]:
    """Return {bench: {tier_breakdown, pooled_em, total_n}} for one cycle."""
    out = {}
    for bench in BENCHES:
        path = eval_dir / f"{bench}_cycle{cycle}.json"
        samples = load_samples(path)
        per_tier = per_tier_em(samples)
        if samples:
            if bench == "truthfulqa":
                judged = [s.get("em_llm_judged") for s in samples
                          if s.get("em_llm_judged") is not None]
                pooled = sum(judged) / len(judged) if judged else \
                         sum(s.get("em", 0) for s in samples) / len(samples)
            else:
                pooled = sum(s.get("em", 0) for s in samples) / len(samples)
        else:
            pooled = None
        out[bench] = {
            "per_tier": per_tier,
            "pooled_em": pooled,
            "total_n": len(samples),
        }
    return out


def _fmt(v, kind="em"):
    if v is None:
        return "--"
    if kind == "em":
        return f"{v * 100:.1f}"
    if kind == "share":
        return f"{v * 100:.0f}\\%"
    if kind == "n":
        return str(int(v))
    return str(v)


def write_csv(cycles: Dict[int, Dict[str, Dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["cycle", "bench",
            "T1_em", "T1_n", "T1_share",
            "T2_em", "T2_n", "T2_share",
            "T3_em", "T3_n", "T3_share",
            "pooled_em", "total_n"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c, by_bench in cycles.items():
            for bench, d in by_bench.items():
                pt = d["per_tier"]
                row = {"cycle": c, "bench": bench,
                       "pooled_em": _fmt(d["pooled_em"]),
                       "total_n": d["total_n"]}
                for t in TIERS:
                    row[f"T{t}_em"] = _fmt(pt[t]["em"])
                    row[f"T{t}_n"] = pt[t]["n"]
                    row[f"T{t}_share"] = f"{pt[t]['share']:.3f}"
                w.writerow(row)
    logger.info("Wrote CSV: %s", path)


def write_latex(
    cycles: Dict[int, Dict[str, Dict]],
    focus_cycle: int,
    path: Path,
) -> None:
    """Render a single-cycle deep-dive table at focus_cycle (default 3).

    Format:
      Bench  | T1 EM (n) | T2 EM (n) | T3 EM (n) | Pooled EM
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    by_bench = cycles.get(focus_cycle)
    if not by_bench:
        logger.warning("focus_cycle=%d has no data; LaTeX table will be empty", focus_cycle)
        return
    tex = []
    tex.append(f"% Auto-generated by scripts/aggregate_per_tier_em.py")
    tex.append(f"% Per-tier EM decomposition for CAEM cycle {focus_cycle}.")
    tex.append("% Memory: caem_per_tier_em_selection_effect.md")
    tex.append("\\begin{tabular}{lccccc}")
    tex.append("\\toprule")
    tex.append("Benchmark & "
               "T1 EM (n) & T2 EM (n) & T3 EM (n) & "
               "Pooled EM & Tier share (T1/T2/T3) \\\\")
    tex.append("\\midrule")
    for bench in BENCHES:
        d = by_bench[bench]
        pt = d["per_tier"]
        t1_cell = "--" if pt[1]["n"] == 0 else f"{_fmt(pt[1]['em'])} ({pt[1]['n']})"
        t2_cell = "--" if pt[2]["n"] == 0 else f"{_fmt(pt[2]['em'])} ({pt[2]['n']})"
        t3_cell = "--" if pt[3]["n"] == 0 else f"{_fmt(pt[3]['em'])} ({pt[3]['n']})"
        pooled = _fmt(d["pooled_em"])
        share = f"{_fmt(pt[1]['share'], 'share')}/{_fmt(pt[2]['share'], 'share')}/{_fmt(pt[3]['share'], 'share')}"
        bench_display = bench.replace("_", "\\_")
        tex.append(f"{bench_display} & {t1_cell} & {t2_cell} & {t3_cell} & "
                   f"{pooled} & {share} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("")
    tex.append("EM values in percentage points; (n) is per-tier sample count.")
    tex.append(f"CommonsenseQA row shows the selection effect: T2 EM "
               f"$>$ T3 EM but the router still sends $>$50\\% of CSQA queries to T3 "
               f"(see \\Cref{{sec:disc-threats}}).")
    path.write_text("\n".join(tex) + "\n")
    logger.info("Wrote LaTeX: %s", path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", type=Path,
                   default=Path("outputs/full_run/eval"))
    p.add_argument("--cycles", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--focus_cycle", type=int, default=3,
                   help="Cycle to render as the LaTeX table (default 3).")
    p.add_argument("--out_csv", type=Path,
                   default=Path("outputs/tab_per_tier_em.csv"))
    p.add_argument("--out_tex", type=Path,
                   default=Path("thesis_report/figures/auto/tab_per_tier_em.tex"))
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(level=ns.log_level, format="%(asctime)s %(levelname)s %(message)s")

    cycles = {c: collect_cycle(ns.eval_dir, c) for c in ns.cycles}

    # Console summary across cycles for CSQA (the headline)
    print()
    print("=" * 90)
    print("CSQA per-tier EM trajectory (the selection-effect headline)")
    print("=" * 90)
    print(f"{'Cycle':<6} | {'T2 EM':>7} | {'T2 n':>5} | {'T3 EM':>7} | {'T3 n':>5} | "
          f"{'Pooled':>7} | T2-share")
    print("-" * 90)
    for c in ns.cycles:
        d = cycles[c].get("commonsense_qa", {})
        if not d or d.get("total_n", 0) == 0:
            continue
        pt = d["per_tier"]
        t2_em = _fmt(pt[2]["em"])
        t3_em = _fmt(pt[3]["em"])
        pooled = _fmt(d["pooled_em"])
        t2_share = _fmt(pt[2]["share"], "share").replace("\\%", "%")
        print(f"C{c:<5} | {t2_em:>7} | {pt[2]['n']:>5} | "
              f"{t3_em:>7} | {pt[3]['n']:>5} | "
              f"{pooled:>7} | {t2_share}")

    print()
    print(f"=== Full per-tier table for cycle {ns.focus_cycle} ===")
    by_bench = cycles[ns.focus_cycle]
    print(f"{'Bench':<18} | {'T1 EM (n)':>14} | {'T2 EM (n)':>14} | {'T3 EM (n)':>14} | {'pooled':>6}")
    print("-" * 90)
    for bench in BENCHES:
        d = by_bench[bench]
        pt = d["per_tier"]
        def cell(t):
            if pt[t]["n"] == 0:
                return "--"
            return f"{_fmt(pt[t]['em'])} ({pt[t]['n']})"
        print(f"{bench:<18} | {cell(1):>14} | {cell(2):>14} | {cell(3):>14} | "
              f"{_fmt(d['pooled_em']):>6}")

    write_csv(cycles, ns.out_csv)
    write_latex(cycles, ns.focus_cycle, ns.out_tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
