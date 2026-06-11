#!/usr/bin/env python3
"""
scripts/baseline_sig_tests_v2.py
================================
CAEM-vs-baseline per-(baseline x bench) significance + CHM table generator.

Extends ``scripts/baseline_sig_tests.py`` (v1) with:
  1. Configurable CAEM cycle pin via --caem_cycle (default 5; also accepts 2
     for the C2-anchor variant -> tab_sig_test_c2.tex).
  2. A CHM column block (CHM_CAEM, CHM_base, dCHM) sourced from
     outputs/baselines_phase1e/chm_comparison.json with the locked CAEM
     calibration applied at rescore time. Bootstrap CI on the dCHM is
     approximated by binomial-sampling the per-subtype rates over the
     n_subtypes axis.
  3. TEX output matching the live ``tab_sig_test.tex`` schema:
        Baseline & Bench & n & EM_CAEM & EM_base & dEM
                 & CHM_CAEM & CHM_base & dCHM
                 & Sig (EM / CHM) & 95% CI (dEM / dCHM)
     Significance markers:
        ``*``  -> Holm-corrected p<0.05 AND |delta| >= practical floor.
        ``-``  -> not significant.
        ``(B)`` -> baseline wins on this axis (CAEM trails).

Usage
-----
    python -m scripts.baseline_sig_tests_v2 \
        --caem_cycle 5 \
        --out_tex thesis_report/figures/auto/tab_sig_test.tex

    python -m scripts.baseline_sig_tests_v2 \
        --caem_cycle 2 \
        --out_tex thesis_report/figures/auto/tab_sig_test_c2.tex
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse v1's correctness pipeline (Holm correction, McNemar, bootstrap CI,
# eval-id loading). v2 is strictly a presentation + CHM extension wrapper.
from scripts.baseline_sig_tests import (
    _load_id_to_em,
    _load_eval_ids,
    _baseline_json_candidates,
    _pick_first_existing,
    _holm_correct,
    BENCHMARKS,
)

# Direct imports from the per-axis stats lib used by v1.
try:
    from eval.stats import mcnemar_test, bootstrap_ci
except Exception:
    from scripts.baseline_sig_tests import mcnemar_test, bootstrap_ci  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("baseline_sig_tests_v2")


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

# Practical-significance floor on the EM axis (registered in ch5 prose at
# 0.02 paired-EM diff).
EM_DELTA_FLOOR = 0.02
# CHM is a probability-like rate in [0, 1]; the registered practical floor is
# half of EM's because CHM concentrates in a narrower band.
CHM_DELTA_FLOOR = 0.01

# Holm cap (correction shows star only when corrected-p < this threshold).
HOLM_ALPHA = 0.05


# --------------------------------------------------------------------------- #
# CHM loading                                                                  #
# --------------------------------------------------------------------------- #

def _load_chm(chm_json: Path) -> Dict[str, Dict[str, dict]]:
    """Load chm_comparison.json.

    Schema: {<system_name>: {<bench>: {n, chm, union_rate, per_subtype_rates, weights}}}
    """
    with chm_json.open() as f:
        return json.load(f)


def _caem_chm_for_cycle(chm_blob: Dict[str, Dict[str, dict]],
                         cycle: int,
                         bench: str,
                         fallback: Optional[float] = None) -> Optional[float]:
    """Pick the CAEM CHM cell.

    The chm_comparison.json may key CAEM as ``caem_cycle10`` (the locked
    composite) or ``caem_cycle<N>`` for cycle-specific runs. When neither is
    available or the value is NaN, return the manual fallback (the user
    typically passes the canonical 0.107 / 0.109 / etc).
    """
    keys = [f"caem_cycle{cycle}", f"caem_cycle10", "caem"]
    for k in keys:
        if k in chm_blob and bench in chm_blob[k]:
            v = chm_blob[k][bench].get("chm")
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return float(v)
    return fallback


def _baseline_chm(chm_blob: Dict[str, Dict[str, dict]],
                  baseline: str,
                  bench: str) -> Optional[Tuple[float, dict]]:
    """Return (chm, per_subtype_rates) for a baseline at this bench."""
    if baseline not in chm_blob:
        return None
    if bench not in chm_blob[baseline]:
        return None
    entry = chm_blob[baseline][bench]
    chm = entry.get("chm")
    if chm is None or (isinstance(chm, float) and math.isnan(chm)):
        return None
    return float(chm), entry.get("per_subtype_rates", {}) or {}


def _approx_chm_ci(
    baseline_chm: float,
    baseline_subtypes: dict,
    caem_chm: float,
    n: int = 300,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Approximate 95% CI on (caem_chm - baseline_chm) via subtype-axis
    bootstrap.

    Treat each of the 8 subtypes as an independent binomial sample with
    rate p_subtype over n samples. Resample with replacement and re-pool to
    estimate the CHM dispersion. The caem_chm side is held at the locked
    rescore value (its bootstrap is a separate calibration concern).
    """
    import random as _random
    if not baseline_subtypes:
        return (float("nan"), float("nan"))
    rates = [v for v in baseline_subtypes.values() if v is not None]
    if not rates:
        return (float("nan"), float("nan"))
    rng = _random.Random(seed)
    deltas: List[float] = []
    for _ in range(n_bootstrap):
        resampled = [_resample_rate(r, n, rng) for r in rates]
        chm_b_resampled = sum(resampled) / len(resampled)
        deltas.append(caem_chm - chm_b_resampled)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    return lo, hi


def _resample_rate(p: float, n: int, rng) -> float:
    """Binomial resample: draw n samples, return the realised rate."""
    if not (0.0 <= p <= 1.0):
        return p
    hits = sum(1 for _ in range(n) if rng.random() < p)
    return hits / n


# --------------------------------------------------------------------------- #
# Row construction                                                             #
# --------------------------------------------------------------------------- #

def _compute_rows_with_chm(
    caem_dir: Path,
    baselines_dir: Path,
    chm_json: Path,
    caem_cycle: int,
) -> List[Dict]:
    """One row per (baseline, bench) with EM + CHM + significance markers."""
    baselines = sorted(
        d.name for d in baselines_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not baselines:
        return []
    chm_blob = _load_chm(chm_json)
    caem_eval = caem_dir / "eval"
    eval_ids_by_bench = _load_eval_ids(caem_dir)

    caem_em_cache: Dict[Tuple[str, int], Dict[str, float]] = {}

    def _load_caem(bench: str, cyc: int) -> Dict[str, float]:
        key = (bench, cyc)
        if key in caem_em_cache:
            return caem_em_cache[key]
        p = caem_eval / f"{bench}_cycle{cyc}.json"
        if not p.is_file():
            caem_em_cache[key] = {}
            return {}
        caem_em_cache[key] = _load_id_to_em(p)
        return caem_em_cache[key]

    # Phase 1: collect raw rows.
    by_bench: Dict[str, List[Dict]] = {b: [] for b in BENCHMARKS}
    for baseline in baselines:
        for bench in BENCHMARKS:
            caem_em_map = _load_caem(bench, caem_cycle)
            b_path = _pick_first_existing(_baseline_json_candidates(
                baselines_dir, baseline, bench))
            if not caem_em_map or b_path is None:
                continue
            base_em_map = _load_id_to_em(b_path)
            allowed = eval_ids_by_bench.get(bench) if eval_ids_by_bench else None
            if allowed is not None:
                caem_em_map = {i: v for i, v in caem_em_map.items() if i in allowed}
                base_em_map = {i: v for i, v in base_em_map.items() if i in allowed}
            shared_ids = sorted(set(caem_em_map) & set(base_em_map))
            if len(shared_ids) < 10:
                continue

            caem_em = [caem_em_map[i] for i in shared_ids]
            base_em = [base_em_map[i] for i in shared_ids]
            diffs = [c - b for c, b in zip(caem_em, base_em)]

            try:
                _m, ci_lo, ci_hi = bootstrap_ci(
                    diffs, n_bootstrap=1000, ci=0.95, seed=42)
            except Exception:
                ci_lo = ci_hi = float("nan")
            try:
                chi2, p_raw = mcnemar_test(caem_em, base_em)
            except Exception:
                chi2 = float("nan")
                p_raw = float("nan")

            em_caem_mean = sum(caem_em) / len(caem_em)
            em_base_mean = sum(base_em) / len(base_em)
            em_diff = em_caem_mean - em_base_mean

            # CHM cells.
            caem_chm = _caem_chm_for_cycle(chm_blob, caem_cycle, bench)
            base_chm_pair = _baseline_chm(chm_blob, baseline, bench)
            if base_chm_pair is not None and caem_chm is not None:
                base_chm, base_subtypes = base_chm_pair
                chm_diff = caem_chm - base_chm
                chm_ci_lo, chm_ci_hi = _approx_chm_ci(
                    base_chm, base_subtypes, caem_chm, n=len(shared_ids))
            else:
                base_chm = float("nan")
                chm_diff = float("nan")
                chm_ci_lo = chm_ci_hi = float("nan")

            by_bench[bench].append({
                "baseline": baseline,
                "benchmark": bench,
                "n": len(shared_ids),
                "em_caem": em_caem_mean,
                "em_base": em_base_mean,
                "em_diff": em_diff,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "p_em_raw": p_raw,
                "caem_chm": caem_chm if caem_chm is not None else float("nan"),
                "base_chm": base_chm,
                "chm_diff": chm_diff,
                "chm_ci_low": chm_ci_lo,
                "chm_ci_high": chm_ci_hi,
            })

    # Phase 2: Holm-correct within each bench family on the EM axis.
    rows: List[Dict] = []
    for bench, family in by_bench.items():
        p_raw = [r["p_em_raw"] for r in family]
        p_corr = _holm_correct(p_raw)
        for r, p_h in zip(family, p_corr):
            r["holm_p_em"] = p_h
            r["sig_em"] = _marker(r["em_diff"], p_h, EM_DELTA_FLOOR)
            # CHM significance is approximated from the bootstrap-CI overlap of
            # zero (no parametric test on a composite rate; this is a CI check).
            if not math.isnan(r["chm_ci_low"]) and not math.isnan(r["chm_ci_high"]):
                ci_excludes_zero = (r["chm_ci_low"] > 0) or (r["chm_ci_high"] < 0)
                if abs(r["chm_diff"]) >= CHM_DELTA_FLOOR and ci_excludes_zero:
                    if r["chm_diff"] < 0:
                        r["sig_chm"] = "*"
                    else:
                        r["sig_chm"] = "(B)"  # baseline wins CHM axis
                else:
                    r["sig_chm"] = "-"
            else:
                r["sig_chm"] = "-"
            rows.append(r)
    return rows


def _marker(em_diff: float, holm_p: float, floor: float) -> str:
    """Significance marker for the EM axis."""
    if math.isnan(holm_p):
        return "-"
    if abs(em_diff) < floor:
        return "-"
    if holm_p >= HOLM_ALPHA:
        return "-"
    if em_diff < 0:
        return "(B)"  # baseline wins
    return "*"


# --------------------------------------------------------------------------- #
# Tex emission                                                                 #
# --------------------------------------------------------------------------- #

def render_tex(rows: List[Dict]) -> str:
    """Render the full tex matching the live tab_sig_test schema."""
    lines: List[str] = []
    lines.append("% Auto-generated by scripts/baseline_sig_tests_v2.py (CHM-extended)")
    lines.append("\\begin{tabular}{lllrrrrrrll}")
    lines.append("\\toprule")
    lines.append(
        "Baseline & Bench & n & "
        "$\\mathrm{EM}_{\\text{CAEM}}$ & $\\mathrm{EM}_{\\text{base}}$ & $\\Delta\\mathrm{EM}$ & "
        "$\\mathrm{CHM}_{\\text{CAEM}}$ & $\\mathrm{CHM}_{\\text{base}}$ & $\\Delta\\mathrm{CHM}$ & "
        "Sig (EM / CHM) & 95\\% CI ($\\Delta\\mathrm{EM}$ / $\\Delta\\mathrm{CHM}$) \\\\"
    )
    lines.append("\\midrule")
    # Sort by benchmark then baseline for stable diffs.
    rows_sorted = sorted(rows, key=lambda r: (BENCHMARKS.index(r["benchmark"]), r["baseline"]))
    last_bench = None
    for r in rows_sorted:
        if last_bench is not None and r["benchmark"] != last_bench:
            pass  # could insert \midrule between benches; live tex doesn't
        last_bench = r["benchmark"]
        em_diff_str = f"{_sgn(r['em_diff'])}{abs(r['em_diff']):.3f}"
        if r["sig_em"] == "(B)":
            em_diff_str += "(B)"
        chm_diff_str = f"{_sgn(r['chm_diff'])}{abs(r['chm_diff']):.3f}" if not math.isnan(r["chm_diff"]) else "n/a"
        if r["sig_chm"] == "(B)":
            chm_diff_str += "(B)"
        sig = f"{r['sig_em']} / {r['sig_chm']}"
        ci = (
            f"[{_sgn(r['ci_low'])}{abs(r['ci_low']):.3f}, "
            f"{_sgn(r['ci_high'])}{abs(r['ci_high']):.3f}] / "
            f"[{_sgn(r['chm_ci_low'])}{abs(r['chm_ci_low']):.3f}, "
            f"{_sgn(r['chm_ci_high'])}{abs(r['chm_ci_high']):.3f}]"
        )
        lines.append(
            f"{r['baseline'].replace('_','\\_')} & {r['benchmark'].replace('_','\\_')} & {r['n']} & "
            f"{r['em_caem']:.3f} & {r['em_base']:.3f} & {em_diff_str} & "
            f"{r['caem_chm']:.3f} & {r['base_chm']:.3f} & {chm_diff_str} & "
            f"{sig} & {ci} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _sgn(x: float) -> str:
    if math.isnan(x):
        return ""
    return "+" if x >= 0 else "-"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caem_dir", default="outputs/full_run",
                    help="CAEM main-run directory (contains eval/).")
    ap.add_argument("--baselines_dir", default="outputs/baselines_phase1e",
                    help="Directory with one subdir per baseline.")
    ap.add_argument("--chm_json", default="outputs/baselines_phase1e/chm_comparison.json",
                    help="Cross-system CHM comparison JSON.")
    ap.add_argument("--caem_cycle", type=int, default=5,
                    help="CAEM cycle to pin the comparison against (5 for "
                         "final-trajectory anchor, 2 for the C2 best-cycle anchor).")
    ap.add_argument("--out_tex", default="thesis_report/figures/auto/tab_sig_test.tex",
                    help="Output tex file.")
    ap.add_argument("--out_csv", default="outputs/tab_sig_test_v2.csv",
                    help="Optional CSV companion output.")
    ns = ap.parse_args()

    rows = _compute_rows_with_chm(
        Path(ns.caem_dir),
        Path(ns.baselines_dir),
        Path(ns.chm_json),
        ns.caem_cycle,
    )
    logger.info("Computed %d rows for caem_cycle=%d", len(rows), ns.caem_cycle)

    tex = render_tex(rows)
    out_path = Path(ns.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    logger.info("Wrote tex -> %s", out_path)

    # CSV companion (for ad-hoc analysis / sanity diffs).
    import csv as _csv
    csv_path = Path(ns.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        if rows:
            writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info("Wrote csv -> %s", csv_path)


if __name__ == "__main__":
    main()
