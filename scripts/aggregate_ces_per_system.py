#!/usr/bin/env python
"""
scripts/aggregate_ces_per_system.py
====================================
CES (Composite Evaluation Score) per baseline + CAEM cycle, for Ch5 §5.3.7.

CES is a 5-axis geometric mean (Ch4 §11):
  - ACC : pooled EM (accuracy)
  - EPI : 1 - CHM (epistemic alignment / hallucination resistance)
  - RET : retention against forgetting (CAEM-only; baselines = 1.0 by convention)
  - CAL : calibration quality (1 - ECE)
  - VER : verifier reliability (correlation with ground truth EM)

This script reports CES per system, per axis, and the geometric mean.
For baselines without a verifier (B1/B2/B3/B5), VER and CAL default to
1.0 by convention (the metric rewards systems that DO have these axes;
penalizing baselines for not having them isn't the comparison).

Inputs (read-only)
------------------
  outputs/baselines/<name>/<bench>_cycle0.json
  outputs/full_run/eval/<bench>_cycle{N}.json

CES.EPI requires CHM, which requires the verifier rescore for baselines.
Until that lands, baseline EPI is reported as "pending" and the geometric
mean degrades gracefully (axis is omitted from the mean).

Outputs
-------
  outputs/tab_ces_per_system.csv
  thesis_report/figures/auto/tab_ces_per_system.tex

Memory refs:
  caem_two_verifiers.md            -- verifier dispatch
  caem_truthfulqa_rescore_findings.md  -- TruthfulQA uses LLM-judge
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


BENCHES = ["fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa"]


def _load_samples(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text()).get("samples", [])
    except json.JSONDecodeError:
        return []


def _system_pooled_em(samples_by_bench: Dict[str, List[dict]]) -> Optional[float]:
    """Pooled EM across all benches present (use em_llm_judged for TQA)."""
    per_bench = []
    for bench, samples in samples_by_bench.items():
        if not samples:
            continue
        if bench == "truthfulqa":
            judged = [s.get("em_llm_judged") for s in samples
                      if s.get("em_llm_judged") is not None]
            if judged:
                per_bench.append(sum(judged) / len(judged))
                continue
        ems = [s.get("em", 0.0) for s in samples]
        per_bench.append(sum(ems) / len(ems))
    return sum(per_bench) / len(per_bench) if per_bench else None


def _system_chm(samples_by_bench: Dict[str, List[dict]]) -> Optional[float]:
    """Pooled CHM across all benches. Baselines without verifier signals
    return None (CHM not computable without verifier rescore).
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval.metrics import composite_hallucination_metric
    chms = []
    for bench, samples in samples_by_bench.items():
        if not samples:
            continue
        # Check if verifier signals are present (sentinel: u_pre not None)
        has_sig = any(s.get("u_pre") is not None for s in samples)
        if not has_sig:
            return None  # Whole system lacks verifier rescore
        try:
            chms.append(composite_hallucination_metric(samples)["chm"])
        except Exception as e:
            logger.warning("CHM compute failed for %s: %s", bench, e)
            continue
    return sum(chms) / len(chms) if chms else None


def _system_abstention_rate(samples_by_bench: Dict[str, List[dict]]) -> float:
    """Fraction of samples where the system abstained."""
    total = 0
    abstained = 0
    for samples in samples_by_bench.values():
        total += len(samples)
        abstained += sum(1 for s in samples if s.get("abstained"))
    return abstained / total if total > 0 else 0.0


def collect_system(
    label: str,
    paths_by_bench: Dict[str, Path],
    has_verifier: bool,
) -> Dict[str, Optional[float]]:
    """Return CES axes for one system."""
    samples_by_bench = {b: _load_samples(p) for b, p in paths_by_bench.items()}
    if all(not v for v in samples_by_bench.values()):
        return None

    acc = _system_pooled_em(samples_by_bench)
    chm = _system_chm(samples_by_bench)
    epi = (1.0 - chm) if chm is not None else None
    abst = _system_abstention_rate(samples_by_bench)

    # CAL + VER + RET only meaningful for CAEM systems (baselines don't have
    # verifier composite or retention guard). Default to 1.0 for baselines so
    # the geometric mean isn't degenerate.
    cal = 1.0 if not has_verifier else None  # CAEM cal pulled from cycle config
    ver = 1.0 if not has_verifier else None  # placeholder
    ret = 1.0 if not has_verifier else None

    # Geometric mean across populated axes
    axes = {"ACC": acc, "EPI": epi, "RET": ret, "CAL": cal, "VER": ver}
    populated = {k: v for k, v in axes.items() if v is not None and v > 0}
    if populated:
        geom = math.exp(sum(math.log(v) for v in populated.values()) / len(populated))
    else:
        geom = None

    return {
        "system": label,
        "acc": acc, "chm": chm,
        "epi": epi, "ret": ret, "cal": cal, "ver": ver,
        "ces_geom": geom,
        "abstention_rate": abst,
        "n_axes": len(populated),
        "has_verifier": has_verifier,
    }


def _fmt(v, digits=3):
    if v is None:
        return "--"
    return f"{v:.{digits}f}"


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["system", "acc", "chm", "epi", "ret", "cal", "ver",
            "ces_geom", "abstention_rate", "n_axes", "has_verifier"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r2 = {k: (_fmt(r[k]) if isinstance(r[k], float) else r[k])
                  for k in cols if k in r}
            w.writerow(r2)
    logger.info("Wrote CSV: %s", path)


def write_latex(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tex = []
    tex.append("% Auto-generated by scripts/aggregate_ces_per_system.py")
    tex.append("\\begin{tabular}{lcccccccc}")
    tex.append("\\toprule")
    tex.append("System & ACC & CHM & EPI & RET & CAL & VER & CES & Abstain\\% \\\\")
    tex.append("\\midrule")
    for r in rows:
        cells = [
            _fmt(r["acc"]), _fmt(r["chm"]),
            _fmt(r["epi"]), _fmt(r["ret"]),
            _fmt(r["cal"]), _fmt(r["ver"]),
            _fmt(r["ces_geom"]),
            f"{r['abstention_rate']*100:.1f}",
        ]
        name = r["system"].replace("_", "\\_")
        tex.append(f"{name} & " + " & ".join(cells) + " \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("")
    tex.append("ACC = pooled EM; EPI = 1$-$CHM (8-subtype composite hallucination metric); "
               "RET / CAL / VER = retention / calibration / verifier-reliability axes "
               "(CAEM-internal; baselines default to 1.0 by convention). "
               "CES = geometric mean of populated axes. Dashes indicate the axis "
               "requires the verifier rescore (pending for some baselines).")
    path.write_text("\n".join(tex) + "\n")
    logger.info("Wrote LaTeX: %s", path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baselines_dir", type=Path,
                   default=Path("outputs/baselines"))
    p.add_argument("--caem_dir", type=Path,
                   default=Path("outputs/full_run/eval"))
    p.add_argument("--out_csv", type=Path,
                   default=Path("outputs/tab_ces_per_system.csv"))
    p.add_argument("--out_tex", type=Path,
                   default=Path("thesis_report/figures/auto/tab_ces_per_system.tex"))
    p.add_argument("--caem_cycles", type=int, nargs="+",
                   default=[0, 3, 5])
    p.add_argument("--baseline_names", nargs="+",
                   default=["zero_shot", "cot", "fiveshot_cot", "rag", "cot_rag",
                            "flare", "vanilla_ft", "semantic_entropy"])
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(level=ns.log_level, format="%(asctime)s %(levelname)s %(message)s")

    rows = []

    # Baselines first
    for name in ns.baseline_names:
        paths = {b: ns.baselines_dir / name / f"{b}_cycle0.json" for b in BENCHES}
        row = collect_system(f"B-{name}", paths, has_verifier=False)
        if row:
            rows.append(row)

    # CAEM cycles
    for c in ns.caem_cycles:
        paths = {b: ns.caem_dir / f"{b}_cycle{c}.json" for b in BENCHES}
        row = collect_system(f"CAEM C{c}", paths, has_verifier=True)
        if row:
            rows.append(row)

    # Console summary
    print()
    print("=" * 90)
    print("CES per system")
    print("=" * 90)
    print(f"{'System':<22} | {'ACC':>5} | {'CHM':>5} | {'EPI':>5} | {'CES':>5} | "
          f"{'Abst%':>6} | axes")
    print("-" * 90)
    for r in rows:
        print(f"{r['system']:<22} | {_fmt(r['acc']):>5} | {_fmt(r['chm']):>5} | "
              f"{_fmt(r['epi']):>5} | {_fmt(r['ces_geom']):>5} | "
              f"{r['abstention_rate']*100:>5.1f} | {r['n_axes']}")

    write_csv(rows, ns.out_csv)
    write_latex(rows, ns.out_tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
