#!/usr/bin/env python
"""
scripts/aggregate_three_headline_decomposition.py
==================================================
Three-headline decomposition table for Ch5 §5.3.1.

Reports three pooled-EM contrasts that isolate where each architectural
piece of CAEM contributes:

  Layer A — B1 → CAEM C3   (full architecture + SIL)
  Layer B — B1 → CAEM C0   (architecture only — router + RAG + verifier,
                            BEFORE any SIL fine-tuning)
  Layer C — CAEM C0 → C3   (SIL only — what the self-improvement loop
                            adds on top of the cold-start architecture)

Layer A is the headline reviewers see; layers B + C decompose it.

The decomposition runs on per-bench EM and on three pooled views:
  - 5-bench pooled  (all benches; TruthfulQA via LLM-judge when available)
  - 4-bench pooled  (excluding TruthfulQA — apples-to-apples on benches
                     where the legacy EM metric is meaningful)
  - 3-bench pooled  (training benches only: FEVER / TriviaQA / CSQA)

Inputs (read-only)
------------------
  outputs/baselines/zero_shot/<bench>_cycle0.json    -- B1
  outputs/full_run/eval/<bench>_cycle0.json          -- CAEM C0
  outputs/full_run/eval/<bench>_cycle3.json          -- CAEM C3

Outputs
-------
  outputs/tab_headline_decomposition.csv               -- raw deltas
  thesis_report/figures/auto/tab_headline_decomposition.tex  -- LaTeX

For TruthfulQA, prefers em_llm_judged when populated (proper truthfulness
metric per Lin et al. ACL 2022 §3.2); falls back to legacy em otherwise.

Memory refs:
  caem_three_headline_decomposition.md  -- pre-registered framing
  caem_b1_vs_c0_framing.md              -- headline is B1→C3, not C0→C3
  caem_truthfulqa_rescore_findings.md   -- LLM-judge preferred for TQA
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


BENCHES = ["fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa"]
TRAINING_BENCHES = ["fever", "triviaqa", "commonsense_qa"]
NON_TQA_BENCHES = [b for b in BENCHES if b != "truthfulqa"]


def load_em(path: Path, *, use_judged_for_tqa: bool = True) -> Optional[float]:
    """Compute per-bench EM from a single evaluation JSON.

    For TruthfulQA, prefer ``em_llm_judged`` over the legacy ``em`` field
    when populated (the legacy field is rouge_l>0.15, which inflates
    long-prose answers — see eval/harness.py:571-575).
    """
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    samples = d.get("samples") or []
    if not samples:
        return None

    if "truthfulqa" in path.name and use_judged_for_tqa:
        judged = [s.get("em_llm_judged") for s in samples
                  if s.get("em_llm_judged") is not None]
        if judged:
            return sum(judged) / len(judged)
        # Fall through to legacy em if judged isn't populated yet
        logger.warning("TruthfulQA %s has no em_llm_judged; using legacy em "
                       "(inflated). Run scripts/rescore_truthfulqa.py.",
                       path)

    ems = [s.get("em", 0.0) for s in samples]
    return sum(ems) / len(ems) if ems else None


def collect_row(label: str, paths: Dict[str, Path]) -> Dict[str, Optional[float]]:
    """Pull per-bench EMs into one row dict keyed by bench name."""
    row: Dict[str, Optional[float]] = {"label": label}
    for bench in BENCHES:
        row[bench] = load_em(paths[bench])
    # Pooled views
    row["pooled_5"] = _safe_mean(row[b] for b in BENCHES)
    row["pooled_4"] = _safe_mean(row[b] for b in NON_TQA_BENCHES)
    row["pooled_3"] = _safe_mean(row[b] for b in TRAINING_BENCHES)
    return row


def _safe_mean(it) -> Optional[float]:
    vals = [v for v in it if v is not None]
    return sum(vals) / len(vals) if vals else None


def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return b - a   # delta = end - start, positive = improvement


def build_decomposition(
    b1: Dict[str, Optional[float]],
    c0: Dict[str, Optional[float]],
    c3: Dict[str, Optional[float]],
) -> List[Dict[str, Optional[float]]]:
    """Build three rows: B1→C3, B1→C0, C0→C3."""
    layers = []
    for label, src, dst in [
        ("B1 → CAEM C3 (full)",          b1, c3),
        ("B1 → CAEM C0 (architecture)",  b1, c0),
        ("CAEM C0 → C3 (SIL only)",       c0, c3),
    ]:
        row: Dict[str, Optional[float]] = {"contrast": label}
        for col in BENCHES + ["pooled_5", "pooled_4", "pooled_3"]:
            row[col] = delta(src[col], dst[col])
        layers.append(row)
    return layers


def write_csv(rows: List[Dict[str, Optional[float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["contrast"] + BENCHES + ["pooled_5", "pooled_4", "pooled_3"]
    def _fmt(v):
        if v is None:
            return ""
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in cols})
    logger.info("Wrote CSV: %s", path)


def _fmt_pp(v: Optional[float]) -> str:
    if v is None:
        return "--"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}"


def write_latex(
    rows: List[Dict[str, Optional[float]]],
    b1_row: Dict[str, Optional[float]],
    c0_row: Dict[str, Optional[float]],
    c3_row: Dict[str, Optional[float]],
    path: Path,
) -> None:
    """Render a tcolorbox-friendly tabular with the three contrasts +
    the absolute B1 / C0 / C3 baselines underneath for context.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt_abs(v):
        return f"{v * 100:.1f}" if v is not None else "--"

    tex = []
    tex.append("% Auto-generated by scripts/aggregate_three_headline_decomposition.py")
    tex.append("% Three-row decomposition + three reference rows.")
    tex.append("% Memory: caem_three_headline_decomposition.md, caem_b1_vs_c0_framing.md")
    tex.append("\\begin{tabular}{lcccccccc}")
    tex.append("\\toprule")
    tex.append("Contrast & FEVER & TriviaQA & CSQA & TruthfulQA$^\\dagger$ & StrategyQA & "
               "Pooled-5 & Pooled-4 & Pooled-3 \\\\")
    tex.append("\\midrule")
    tex.append("\\multicolumn{9}{l}{\\textit{Per-cell value = end-EM minus start-EM, in percentage points.}} \\\\")
    for r in rows:
        cells = [_fmt_pp(r[c]) for c in BENCHES + ["pooled_5", "pooled_4", "pooled_3"]]
        tex.append(f"{r['contrast']} & " + " & ".join(cells) + " \\\\")
    tex.append("\\midrule")
    tex.append("\\multicolumn{9}{l}{\\textit{Reference: absolute EM (\\%) per row.}} \\\\")
    for label, row in [
        ("B1 zero\\_shot (baseline)", b1_row),
        ("CAEM C0 (cold start)",      c0_row),
        ("CAEM C3 (post-SIL)",        c3_row),
    ]:
        cells = [_fmt_abs(row[c]) for c in BENCHES + ["pooled_5", "pooled_4", "pooled_3"]]
        tex.append(f"{label} & " + " & ".join(cells) + " \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("")
    tex.append("$^\\dagger$ TruthfulQA EM via Claude Haiku 4.5 LLM judge "
               "(see \\Cref{sec:disc-threats} for the metric-correction caveat); "
               "all other cells use strict EM/label accuracy.")
    path.write_text("\n".join(tex) + "\n")
    logger.info("Wrote LaTeX: %s", path)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--baselines_dir", type=Path,
                   default=Path("outputs/baselines"))
    p.add_argument("--caem_dir", type=Path,
                   default=Path("outputs/full_run/eval"))
    p.add_argument("--b1_name", default="zero_shot",
                   help="Baseline directory name to use as the B1 anchor.")
    p.add_argument("--caem_c3_cycle", type=int, default=3,
                   help="Which CAEM cycle is the 'post-SIL headline'. "
                        "Defaults to 3 (last successful SIL cycle in v2.1).")
    p.add_argument("--out_csv", type=Path,
                   default=Path("outputs/tab_headline_decomposition.csv"))
    p.add_argument("--out_tex", type=Path,
                   default=Path("thesis_report/figures/auto/tab_headline_decomposition.tex"))
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(level=ns.log_level, format="%(asctime)s %(levelname)s %(message)s")

    # Resolve paths per source
    b1_paths = {b: ns.baselines_dir / ns.b1_name / f"{b}_cycle0.json" for b in BENCHES}
    c0_paths = {b: ns.caem_dir / f"{b}_cycle0.json" for b in BENCHES}
    c3_paths = {b: ns.caem_dir / f"{b}_cycle{ns.caem_c3_cycle}.json" for b in BENCHES}

    b1 = collect_row("B1 zero_shot", b1_paths)
    c0 = collect_row("CAEM C0", c0_paths)
    c3 = collect_row(f"CAEM C{ns.caem_c3_cycle}", c3_paths)

    # Sanity log
    for label, row in [("B1", b1), ("C0", c0), (f"C{ns.caem_c3_cycle}", c3)]:
        missing = [b for b in BENCHES if row[b] is None]
        if missing:
            logger.warning("%s missing per-bench EM for: %s", label, missing)

    rows = build_decomposition(b1, c0, c3)

    # Console summary
    print()
    print("=" * 88)
    print("Three-headline decomposition (per-cell value = Δ EM in percentage points)")
    print("=" * 88)
    print(f"{'Contrast':<30} | {'pooled-5':>8} | {'pooled-4':>8} | {'pooled-3':>8}")
    print("-" * 88)
    for r in rows:
        print(f"{r['contrast']:<30} | "
              f"{_fmt_pp(r['pooled_5']):>8} | "
              f"{_fmt_pp(r['pooled_4']):>8} | "
              f"{_fmt_pp(r['pooled_3']):>8}")
    print()
    print(f"{'Reference (absolute EM %)':<30} | {'pooled-5':>8} | {'pooled-4':>8} | {'pooled-3':>8}")
    print("-" * 88)
    for label, row in [("B1 zero_shot", b1), ("CAEM C0", c0), (f"CAEM C{ns.caem_c3_cycle}", c3)]:
        def _abs(v): return f"{v*100:.1f}" if v is not None else "--"
        print(f"{label:<30} | "
              f"{_abs(row['pooled_5']):>8} | "
              f"{_abs(row['pooled_4']):>8} | "
              f"{_abs(row['pooled_3']):>8}")

    write_csv(rows, ns.out_csv)
    write_latex(rows, b1, c0, c3, ns.out_tex)

    return 0


if __name__ == "__main__":
    sys.exit(main())
