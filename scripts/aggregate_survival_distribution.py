#!/usr/bin/env python
"""
scripts/aggregate_survival_distribution.py
============================================
Deferred-buffer survival distribution + cycle-by-cycle promotion rates.

Empirical receipt for cor:self-correction (bounded survival distribution
for entries that eventually exit the deferred buffer).

For each cycle's deferred buffer, computes:
  - Buffer size
  - Distribution of storage_cycle_pushed (when entries entered)
  - Distribution of age (how many reconsider invocations each entry survived)
  - Promotions estimated from buffer-size deltas + push rates

Inputs (read-only)
------------------
  outputs/full_run/deferred_buffer_cycle_{N}.pkl

Outputs
-------
  outputs/tab_survival_distribution.csv
  thesis_report/figures/auto/tab_survival_distribution.tex

Memory refs:
  caem_deferred_buffer_abort_skip.md  -- recovery rate ~4.4%, bounded by successful cycles
  caem_post_step7_recall_update.md     -- per-cycle promotion rates needed
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_buffer(path: Path) -> List:
    if not path.is_file():
        return []
    try:
        with path.open("rb") as fh:
            d = pickle.load(fh)
    except Exception as e:
        logger.warning("Could not unpickle %s: %s", path, e)
        return []
    if isinstance(d, dict) and "entries" in d:
        return list(d["entries"])
    if isinstance(d, list):
        return list(d)
    return []


def summarize_cycle(entries: List) -> Dict:
    n = len(entries)
    push_cycle_dist = Counter()
    age_dist = Counter()
    bench_dist = Counter()
    for e in entries:
        pc = getattr(e, "storage_cycle_pushed", None)
        ag = getattr(e, "age", None)
        b = getattr(e, "source_benchmark", None)
        if pc is not None:
            push_cycle_dist[pc] += 1
        if ag is not None:
            age_dist[ag] += 1
        if b:
            bench_dist[b] += 1
    return {
        "n": n,
        "push_cycle_dist": dict(push_cycle_dist),
        "age_dist": dict(age_dist),
        "bench_dist": dict(bench_dist),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full_run_dir", type=Path,
                   default=Path("outputs/full_run"))
    p.add_argument("--cycles", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--out_csv", type=Path,
                   default=Path("outputs/tab_survival_distribution.csv"))
    p.add_argument("--out_tex", type=Path,
                   default=Path("thesis_report/figures/auto/tab_survival_distribution.tex"))
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()
    logging.basicConfig(level=ns.log_level, format="%(asctime)s %(levelname)s %(message)s")

    cycle_summaries: Dict[int, Dict] = {}
    for c in ns.cycles:
        entries = load_buffer(ns.full_run_dir / f"deferred_buffer_cycle_{c}.pkl")
        cycle_summaries[c] = summarize_cycle(entries)

    # Per-cycle promotion + push estimation
    print(f"\n{'Cycle':<6} | {'BufSize':>7} | {'NewPushed':>9} | {'NetDelta':>9} | "
          f"{'EstPromoted':>11}")
    print("-" * 70)
    prev_size = 0
    push_counts_seen = Counter()
    rows = []
    for c in ns.cycles:
        s = cycle_summaries[c]
        size = s["n"]
        # New entries pushed THIS cycle = entries with storage_cycle_pushed = c
        new_pushed = s["push_cycle_dist"].get(c, 0)
        net_delta = size - prev_size
        # Estimate promotions: prev_size + new_pushed - size = promotions + ttl_dropped
        exits = prev_size + new_pushed - size
        # If positive, that's the count that left; promotion rate is part of that
        print(f"C{c:<5} | {size:>7} | {new_pushed:>9} | {net_delta:>+9} | "
              f"~{exits:>9}")
        rows.append({
            "cycle": c, "buffer_size": size, "new_pushed_this_cycle": new_pushed,
            "net_delta": net_delta, "estimated_exits": exits,
        })
        prev_size = size

    # Final-buffer push-cycle distribution = where did the C5 survivors come from?
    final = cycle_summaries[max(ns.cycles)]
    print(f"\n=== C{max(ns.cycles)} buffer push-cycle distribution (origin of survivors) ===")
    for pc in sorted(final["push_cycle_dist"].keys()):
        cnt = final["push_cycle_dist"][pc]
        print(f"  Pushed at C{pc}: {cnt} survivors")

    print(f"\n=== C{max(ns.cycles)} buffer age distribution (reconsider invocations survived) ===")
    for ag in sorted(final["age_dist"].keys()):
        cnt = final["age_dist"][ag]
        print(f"  Age {ag}: {cnt} entries")

    print(f"\n=== C{max(ns.cycles)} buffer per-benchmark composition ===")
    total = final["n"]
    for b, cnt in sorted(final["bench_dist"].items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total if total else 0
        print(f"  {b}: {cnt} ({pct:.1f}%)")

    # CSV
    ns.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with ns.out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cycle", "buffer_size", "new_pushed_this_cycle",
                                            "net_delta", "estimated_exits"])
        w.writeheader()
        for r in rows: w.writerow(r)
    logger.info("Wrote CSV: %s", ns.out_csv)

    # LaTeX
    ns.out_tex.parent.mkdir(parents=True, exist_ok=True)
    tex = []
    tex.append("% Auto-generated by scripts/aggregate_survival_distribution.py")
    tex.append("\\begin{tabular}{cccccc}")
    tex.append("\\toprule")
    tex.append("Cycle & Buffer size & Newly pushed & Net $\\Delta$ & "
               "Estimated exits & Cumulative recovery \\\\")
    tex.append("\\midrule")
    cum_exits = 0
    for r in rows:
        if r["estimated_exits"] is not None and r["estimated_exits"] > 0:
            cum_exits += r["estimated_exits"]
        tex.append(f"C{r['cycle']} & {r['buffer_size']} & {r['new_pushed_this_cycle']} & "
                   f"$+${r['net_delta']:+d} & {max(0, r['estimated_exits'])} & {cum_exits} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("")
    tex.append("Per-cycle deferred-buffer dynamics. Exits include promotions to memory and "
               "TTL-drops. Cumulative recovery is the empirical receipt for "
               "\\Cref{cor:self-correction}.")
    ns.out_tex.write_text("\n".join(tex) + "\n")
    logger.info("Wrote LaTeX: %s", ns.out_tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
