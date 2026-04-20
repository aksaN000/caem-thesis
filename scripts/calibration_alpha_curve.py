"""
scripts/calibration_alpha_curve.py
====================================
Post-hoc validation of CAEM's calibrated τ_store against the Ch4 α > ½
theorem premise. Consumes the per-sample JSON dumps written by
``scripts/run_purity_validation.py`` under FIX-8, replays the Stage-5
decision tree at a grid of candidate τ_store values, and emits the α(τ)
curve per cycle for the Ch5 Appendix.

Why (Ch4 calibration-sensitivity argument)
-------------------------------------------
τ_store is fit at the P70 quantile of Cycle-0 calibration-fold u_stored
values. P70 is a design-time heuristic, not theorem-derived. This script
turns the heuristic into an empirical claim by measuring α at the fitted
τ_store AND at neighbouring quantile candidates — if α(τ_store) > 0.5
across a ±10-percentile band, the calibration is robust to the specific
quantile choice.

The replay uses saved per-sample scalars (u_stored, p_contra, p_ground_max,
decision) and rebuilds the ``_decide`` tree at candidate τ_store values
while holding τ_defer + contradiction-veto fixed at their calibrated
values. All scalars come from the actual verifier run, so no inference
re-execution is needed — this is a pure post-hoc analysis.

Three artefacts are produced:

  outputs/calibration/alpha_vs_tau_by_cycle.json
      16-point α(τ) curves per (cycle × {benchmarks, pooled}), machine-
      readable. ~90 KB.

  pre thesis 1 report/figures/alpha_vs_tau_cal_vs_converged.pdf
      Two-curve figure: Cycle-0 (calibration-matched) vs Cycle-N
      (converged). Red horizontal line at α=0.5; black vertical line at
      fitted τ_store.

  pre thesis 1 report/figures/alpha_trajectory.pdf
      α at fitted τ_store vs cycle index — provides visual support for
      Theorem 2 monotonicity.

Typical usage (post-Step-19, end of Phase 1a)
----------------------------------------------
    PYTHONPATH=. python scripts/calibration_alpha_curve.py
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("calibration_alpha_curve")

# Hold these fixed at their CAEMConfig values. τ_store is the one we sweep.
DEFAULT_CONTRA_VETO = 0.30


def _replay_stage5(rec: Dict[str, Any], tau_store: float,
                   tau_defer: float, contra_veto: float) -> str:
    """Replay verifier._decide at the candidate τ_store.

    Mirrors caem/verification/verifier.py:_decide line-for-line. Only
    τ_store is varied; τ_defer + veto are held fixed at their calibrated /
    config values per Ch4's "hold other gates fixed while sweeping τ_store"
    sensitivity design.
    """
    if rec["p_contra"] >= contra_veto:
        return "DISCARD"
    if rec["u_stored"] >= tau_store:
        return "STORE"
    if rec["u_stored"] >= tau_defer:
        return "DEFERRED"
    # ABSTAIN vs DISCARD depends on p_ground_max vs abstain_pground_ceiling
    # but both map to "not STORE" for α purposes. We return "ABSTAIN" as a
    # single catch-all since α only cares about STORE vs not-STORE.
    return "ABSTAIN"


def _alpha_at(records: List[Dict[str, Any]], tau_store: float,
              tau_defer: float, contra_veto: float) -> Tuple[float, int, int, int, int]:
    """Compute α (balanced accuracy) at a candidate τ_store.

    Returns (α, tp, tn, fp, fn) so callers can inspect the confusion cells.
    """
    tp = tn = fp = fn = 0
    for r in records:
        passed = _replay_stage5(r, tau_store, tau_defer, contra_veto) == "STORE"
        correct = bool(r["is_correct"])
        if   correct and     passed: tp += 1
        elif correct and not passed: fn += 1
        elif not correct and     passed: fp += 1
        else:                         tn += 1
    # Match measure_verification_balanced_accuracy's 0.0-default convention
    # (run_purity_validation.py line ~397) so that α at the fitted τ_store
    # replays to the same value as the raw purity measurement, rather than
    # drifting by up to 0.25 when a benchmark has a degenerate confusion cell.
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr), tp, tn, fp, fn


def _load_per_sample_dumps(per_sample_dir: Path) -> Dict[int, Dict[str, List[Dict]]]:
    """Group per-sample dumps by cycle, then by benchmark.

    File-name convention from FIX-8:
        {per_sample_dir}/purity_raw_{bench}_cycle{N}.json
    """
    cycles: Dict[int, Dict[str, List[Dict]]] = {}
    pattern = str(per_sample_dir / "purity_raw_*_cycle*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.error("No per-sample dumps found matching %s", pattern)
        logger.error(
            "Run Step 19 (scripts/run_purity_validation.py) first — it now "
            "writes per-sample scalars under FIX-8."
        )
        sys.exit(1)

    for fpath in files:
        name = Path(fpath).stem  # e.g., purity_raw_fever_cycle7
        # parse benchmark and cycle
        try:
            _, _, bench_plus = name.partition("purity_raw_")
            bench, _, cycle_tag = bench_plus.rpartition("_cycle")
            cycle = int(cycle_tag)
        except Exception:
            logger.warning("Skipping unparseable filename: %s", fpath)
            continue
        cycles.setdefault(cycle, {})[bench] = json.load(open(fpath))
    return cycles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per_sample_dir", type=Path,
                    default=Path("outputs/purity_validation/per_sample"))
    ap.add_argument("--calibrated_thresholds", type=Path,
                    default=Path("outputs/cycle_0/calibrated_thresholds.json"))
    ap.add_argument("--contra_veto", type=float, default=DEFAULT_CONTRA_VETO,
                    help="Contradiction veto threshold (held fixed during sweep).")
    ap.add_argument("--output_json", type=Path,
                    default=Path("outputs/calibration/alpha_vs_tau_by_cycle.json"))
    ap.add_argument("--figures_dir", type=Path,
                    default=Path("pre thesis 1 report/figures"))
    ap.add_argument("--tau_grid_halfwidth", type=float, default=0.15,
                    help="τ_store sweep = fitted ± halfwidth, clipped to [0.30, 0.85].")
    ap.add_argument("--tau_grid_points", type=int, default=16)
    ap.add_argument("--converged_cycle", type=int, default=None,
                    help="Which cycle represents the converged state for the 2-curve "
                         "figure. Defaults to the largest cycle index found in the dumps.")
    args = ap.parse_args()

    # Load the calibrated thresholds fit at Step 7.0.2
    cal = json.load(open(args.calibrated_thresholds))
    tau_store_fit = float(cal["thresholds"]["store"])
    tau_defer_fit = float(cal["thresholds"]["defer"])
    logger.info("Loaded calibrated thresholds: store=%.4f  defer=%.4f  contra_veto=%.3f",
                tau_store_fit, tau_defer_fit, args.contra_veto)

    # Load per-sample dumps
    cycles_data = _load_per_sample_dumps(args.per_sample_dir)
    available_cycles = sorted(cycles_data.keys())
    logger.info("Found per-sample dumps for cycles: %s", available_cycles)
    if args.converged_cycle is None:
        args.converged_cycle = available_cycles[-1]

    # Build τ_store sweep grid
    tau_grid = np.linspace(
        max(0.30, tau_store_fit - args.tau_grid_halfwidth),
        min(0.85, tau_store_fit + args.tau_grid_halfwidth),
        args.tau_grid_points,
    )

    # Compute α curves for each cycle
    results: Dict[str, Any] = {
        "fitted_tau_store": tau_store_fit,
        "fitted_tau_defer": tau_defer_fit,
        "contra_veto":      args.contra_veto,
        "tau_candidates":   [float(t) for t in tau_grid],
        "by_cycle":         {},
    }

    for cycle in available_cycles:
        bench_map = cycles_data[cycle]
        all_records = [r for recs in bench_map.values() for r in recs]

        pooled_curve = [
            (float(t), _alpha_at(all_records, t, tau_defer_fit, args.contra_veto)[0])
            for t in tau_grid
        ]
        per_bench_curves = {
            b: [(float(t), _alpha_at(recs, t, tau_defer_fit, args.contra_veto)[0])
                for t in tau_grid]
            for b, recs in bench_map.items()
        }
        alpha_at_fitted, tp, tn, fp, fn = _alpha_at(
            all_records, tau_store_fit, tau_defer_fit, args.contra_veto
        )

        results["by_cycle"][str(cycle)] = {
            "n_samples_pooled":    len(all_records),
            "pooled":              pooled_curve,
            "per_benchmark":       per_bench_curves,
            "alpha_at_fitted_tau": alpha_at_fitted,
            "confusion_at_fitted": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }
        logger.info("Cycle %2d: α(fitted τ=%.4f) = %.4f  [tp=%d tn=%d fp=%d fn=%d  n=%d]",
                    cycle, tau_store_fit, alpha_at_fitted, tp, tn, fp, fn, len(all_records))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", args.output_json)

    # -- Figures ------------------------------------------------------------ #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figure generation. "
                       "JSON curves are still at %s.", args.output_json)
        return

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure A: Cycle-0 vs converged-cycle pooled α(τ) curves
    fig_a_path = args.figures_dir / "alpha_vs_tau_cal_vs_converged.pdf"
    plt.figure(figsize=(7.5, 4.8))
    for cycle, color, label in [
        (0,                       "C0", "Cycle 0 (calibration-matched distribution)"),
        (args.converged_cycle,    "C3", f"Cycle {args.converged_cycle} (converged distribution)"),
    ]:
        if str(cycle) not in results["by_cycle"]:
            continue
        xs, ys = zip(*results["by_cycle"][str(cycle)]["pooled"])
        plt.plot(xs, ys, color=color, label=label, linewidth=2.0)
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1.0,
                label=r"$\alpha = 0.5$ (theorem threshold)")
    plt.axvline(tau_store_fit, color="black", linestyle=":", linewidth=1.0,
                label=fr"fitted $\tau_{{\mathrm{{store}}}} = {tau_store_fit:.3f}$")
    plt.xlabel(r"$\tau_{\mathrm{store}}$ (candidate)")
    plt.ylabel(r"$\alpha$ (balanced accuracy)")
    plt.title(r"CAEM verifier $\alpha$ vs STORE threshold" "\n"
              "calibration-matched vs converged distribution")
    plt.legend(loc="lower right", fontsize=9, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_a_path, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", fig_a_path)

    # Figure B: α at fitted τ_store across all cycles
    fig_b_path = args.figures_dir / "alpha_trajectory.pdf"
    plt.figure(figsize=(7.5, 3.8))
    xs = available_cycles
    ys = [results["by_cycle"][str(c)]["alpha_at_fitted_tau"] for c in xs]
    plt.plot(xs, ys, "-o", linewidth=2.0, markersize=5)
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1.0,
                label=r"$\alpha = 0.5$ (theorem threshold)")
    plt.xlabel("Cycle")
    plt.ylabel(fr"$\alpha$ at fitted $\tau_{{\mathrm{{store}}}} = {tau_store_fit:.3f}$")
    plt.title(r"$\alpha$ trajectory across CAEM self-improvement cycles"
              "\n(Theorem 2 monotonicity)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_b_path, bbox_inches="tight")
    plt.close()
    logger.info("Wrote %s", fig_b_path)


if __name__ == "__main__":
    main()
