#!/usr/bin/env python
"""
scripts/per_bench_alpha_decision_table.py
==========================================
Per-benchmark conformal-α decision table for the v2 storage gate.

The Phase 1a runner currently fits the conformal gate at uniform
α_store=0.05 across all four training benchmarks. The locked v2 plan
specifies α_TQA=0.40 (relaxed because TQA's bare-entity verifier
signals are weak), but TQA was the only benchmark whose α has v1
empirical justification. HotpotQA and CommonsenseQA are NEW to the v2
panel — never measured under the v2 composite — so their α=0.05 prior
is unvalidated.

This diagnostic is the empirical receipt that converts priors into
data-driven per-bench α choices. For each benchmark in the v2 training
panel and each candidate α, it:

  1. Re-fits a per-bench ``ConformalStorageGate`` on the cal-fold subset
     for that benchmark.
  2. Applies the fitted ``tau_store`` to the eval_rescored subset for
     that benchmark.
  3. Reports realized eval-fold precision and coverage at that α.

The output table lets the operator choose α per bench so that:
  - precision target is empirically attainable under v2's per-bench
    composite (not just hoped for by extrapolating v1 numbers), AND
  - coverage stays above a minimum (e.g. ≥10% admission rate) so cycle
    1+ doesn't starve the SIL training pool for that bench.

Inputs
------
- ``outputs/cycle_0/calibration/calibration_fold_samples.json``  cal fold
- ``outputs/cycle_0/eval_rescored/*_cycle0.json``                eval fold

Outputs
-------
- ``outputs/cycle_0/per_bench_alpha_decision_table.json``  full table
- stdout: a human-readable summary table with recommendations

Usage
-----
    python scripts/per_bench_alpha_decision_table.py
    python scripts/per_bench_alpha_decision_table.py --min_coverage 0.15

After reviewing the table, refit the conformal gate with chosen overrides:
    python scripts/fit_conformal_gate.py \\
        --calib_jsons outputs/cycle_0/calibration/calibration_fold_samples.json \\
        --composite_calibration_json outputs/cycle_0/composite_calibration.json \\
        --output_json outputs/cycle_0/conformal_gate.json \\
        --alpha_store 0.05 --alpha_defer 0.40 \\
        --alpha_store_overrides triviaqa=0.40 hotpotqa=0.10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("per_bench_alpha_decision")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Candidate α values to evaluate. Spans tight (95% target) through loose
# (60% target). The grid is dense at the tight end where precision can
# fall off a cliff, sparser at the loose end.
_ALPHA_GRID: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)

# Precision targets we care about for the recommendation column.
_PRECISION_TARGETS: Tuple[float, ...] = (0.95, 0.90, 0.80, 0.70, 0.60)


def _load_cal_fold(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", data) if isinstance(data, dict) else data
    return list(samples)


def _load_eval_rescored(eval_dir: Path) -> List[Dict[str, Any]]:
    """Concatenate per-benchmark eval_rescored JSONs into a flat sample list."""
    pooled: List[Dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*_cycle0.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", []) if isinstance(data, dict) else data
        # Stamp source_benchmark from the filename if missing on the sample
        bench = path.stem.replace("_cycle0", "")
        for s in samples:
            if not s.get("source_benchmark"):
                s["source_benchmark"] = bench
        pooled.extend(samples)
    return pooled


def _split_by_bench(
    samples: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        bm = s.get("source_benchmark") or "_untagged"
        out[bm].append(s)
    return dict(out)


def _eval_fold_precision_at_tau(
    eval_samples: Sequence[Dict[str, Any]],
    tau_store: float,
    *,
    score_key: str = "u_stored",
    em_key: str = "em",
) -> Tuple[int, int, float, float]:
    """Apply tau_store to the eval-fold subset.

    Returns (stored_n, true_positive_n, precision, coverage).
    """
    n_total = 0
    n_stored = 0
    n_tp = 0
    for s in eval_samples:
        sc = s.get(score_key)
        em = s.get(em_key)
        if not isinstance(sc, (int, float)) or em not in (0, 1, 0.0, 1.0):
            continue
        n_total += 1
        if float(sc) >= tau_store:
            n_stored += 1
            if int(em) == 1:
                n_tp += 1
    precision = (n_tp / n_stored) if n_stored > 0 else float("nan")
    coverage = (n_stored / n_total) if n_total > 0 else 0.0
    return n_stored, n_tp, precision, coverage


def _recommend_alpha(
    per_alpha_rows: List[Dict[str, Any]],
    min_coverage: float,
) -> Dict[str, Any]:
    """Pick the tightest α that meets each precision target with min coverage."""
    rec: Dict[str, Any] = {}
    for target in _PRECISION_TARGETS:
        # Smallest α (tightest gate) where eval_precision ≥ target AND coverage ≥ floor
        chosen = None
        for row in per_alpha_rows:  # _ALPHA_GRID is ascending α
            if (
                row["eval_precision"] is not None
                and row["eval_precision"] >= target
                and row["eval_coverage"] >= min_coverage
            ):
                chosen = row["alpha"]
                break
        rec[f"alpha_for_target_{target:.2f}"] = chosen
    # Practical recommendation: aim for 80% precision unless tight (95%) is attainable.
    if rec.get("alpha_for_target_0.95") is not None:
        rec["recommended_alpha"] = rec["alpha_for_target_0.95"]
        rec["recommended_target"] = 0.95
    elif rec.get("alpha_for_target_0.90") is not None:
        rec["recommended_alpha"] = rec["alpha_for_target_0.90"]
        rec["recommended_target"] = 0.90
    elif rec.get("alpha_for_target_0.80") is not None:
        rec["recommended_alpha"] = rec["alpha_for_target_0.80"]
        rec["recommended_target"] = 0.80
    elif rec.get("alpha_for_target_0.70") is not None:
        rec["recommended_alpha"] = rec["alpha_for_target_0.70"]
        rec["recommended_target"] = 0.70
    else:
        rec["recommended_alpha"] = 0.40  # loosest in the grid
        rec["recommended_target"] = 0.60
    return rec


def build_table(
    cal_path: Path,
    eval_dir: Path,
    *,
    min_coverage: float = 0.10,
) -> Dict[str, Any]:
    from caem.verification.conformal_gate import ConformalStorageGate  # type: ignore

    cal_samples = _load_cal_fold(cal_path)
    eval_samples = _load_eval_rescored(eval_dir)
    cal_by_bench = _split_by_bench(cal_samples)
    eval_by_bench = _split_by_bench(eval_samples)

    benches = sorted(set(cal_by_bench) | set(eval_by_bench))
    logger.info("Benchmarks present: %s", benches)
    logger.info(
        "Cal-fold sizes: %s",
        {b: len(cal_by_bench.get(b, [])) for b in benches},
    )
    logger.info(
        "Eval-fold sizes: %s",
        {b: len(eval_by_bench.get(b, [])) for b in benches},
    )

    per_bench_rows: Dict[str, List[Dict[str, Any]]] = {}
    per_bench_recs: Dict[str, Dict[str, Any]] = {}

    for bm in benches:
        cal_b = cal_by_bench.get(bm, [])
        eval_b = eval_by_bench.get(bm, [])
        rows: List[Dict[str, Any]] = []

        for alpha in _ALPHA_GRID:
            row: Dict[str, Any] = {"alpha": alpha}
            try:
                gate = ConformalStorageGate.fit(
                    cal_b,
                    score_key="u_stored",
                    em_key="em",
                    alpha_store=alpha,
                    alpha_defer=0.40,
                )
                row["tau_store"] = gate.tau_store
                row["cal_precision"] = gate.store_precision
                row["cal_n_stored"] = gate.store_n
                row["cal_n_total"] = gate.cal_n

                n_st, n_tp, ev_prec, ev_cov = _eval_fold_precision_at_tau(
                    eval_b, gate.tau_store,
                )
                row["eval_n_stored"] = n_st
                row["eval_n_tp"] = n_tp
                row["eval_precision"] = (
                    None if (ev_prec != ev_prec) else round(ev_prec, 4)  # NaN check
                )
                row["eval_coverage"] = round(ev_cov, 4)
                row["eval_n_total"] = len(eval_b)
            except ValueError as exc:
                row["error"] = f"fit failed: {exc}"
                row["tau_store"] = None
                row["cal_precision"] = None
                row["eval_precision"] = None
                row["eval_coverage"] = None
            rows.append(row)

        per_bench_rows[bm] = rows
        per_bench_recs[bm] = _recommend_alpha(rows, min_coverage=min_coverage)

    table: Dict[str, Any] = {
        "_min_coverage_floor": min_coverage,
        "_alpha_grid": list(_ALPHA_GRID),
        "_precision_targets": list(_PRECISION_TARGETS),
        "per_benchmark": {
            bm: {"rows": per_bench_rows[bm], "recommendation": per_bench_recs[bm]}
            for bm in benches
        },
    }
    return table


def _print_summary(table: Dict[str, Any]) -> None:
    pb = table["per_benchmark"]
    print()
    print("=" * 110)
    print(
        f"{'benchmark':<18} {'α':>5} {'τ_store':>8} {'cal_prec':>9} "
        f"{'eval_prec':>10} {'eval_cov':>9} {'n_stored':>9}"
    )
    print("-" * 110)
    for bm in sorted(pb):
        rows = pb[bm]["rows"]
        for r in rows:
            tau = "?" if r.get("tau_store") is None else f"{r['tau_store']:.3f}"
            cp = "?" if r.get("cal_precision") is None else f"{r['cal_precision']:.3f}"
            ep = "?" if r.get("eval_precision") is None else f"{r['eval_precision']:.3f}"
            ec = "?" if r.get("eval_coverage") is None else f"{r['eval_coverage']:.3f}"
            ns = r.get("eval_n_stored", "?")
            print(
                f"{bm:<18} {r['alpha']:>5.2f} {tau:>8} {cp:>9} "
                f"{ep:>10} {ec:>9} {ns:>9}"
            )
        rec = pb[bm]["recommendation"]
        print(
            f"  ↳ recommended: α={rec.get('recommended_alpha')} "
            f"(target={rec.get('recommended_target')})"
        )
        print()
    print("=" * 110)
    print()
    print("Recommended `--alpha_store_overrides` argument:")
    overrides: List[str] = []
    for bm in sorted(pb):
        a = pb[bm]["recommendation"].get("recommended_alpha")
        if a is not None and abs(a - 0.05) > 1e-6:
            overrides.append(f"{bm}={a}")
    if overrides:
        print(f"  --alpha_store_overrides {' '.join(overrides)}")
    else:
        print("  (none — uniform α=0.05 is empirically attainable for all benchmarks)")
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--cal_path", type=Path,
        default=Path("outputs/cycle_0/calibration/calibration_fold_samples.json"),
    )
    p.add_argument(
        "--eval_dir", type=Path, default=Path("outputs/cycle_0/eval_rescored"),
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("outputs/cycle_0/per_bench_alpha_decision_table.json"),
    )
    p.add_argument(
        "--min_coverage", type=float, default=0.10,
        help="Floor on eval-fold admission rate when recommending α; below this,"
             " the gate is too tight regardless of precision.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.cal_path.exists():
        logger.error("Cal-fold not found: %s", args.cal_path)
        return 1
    if not args.eval_dir.exists():
        logger.error("Eval-rescored dir not found: %s", args.eval_dir)
        return 1

    table = build_table(args.cal_path, args.eval_dir, min_coverage=args.min_coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2))
    logger.info("Wrote per-benchmark α decision table -> %s", args.output)

    _print_summary(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
