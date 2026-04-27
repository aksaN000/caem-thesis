#!/usr/bin/env python
"""Per-cycle CalProbComposite + ConformalStorageGate re-fit with EMA smoothing.

Phase 2.4 / Phase 2.5 per-cycle integration (Branch C 2026-04-25).

This is the conformal-gate analog of ``recalibrate_thresholds_at_cycle.py``:
where that script re-fits the legacy quantile thresholds on the current
cycle's calibration fold and EMA-smooths against the previous cycle, this
script does the same for:

  (a) CalProbComposite — re-fits per-signal isotonic regression + the
      Cherian boost layer on the current-cycle calibration fold.
  (b) ConformalStorageGate — re-fits τ_store at α=0.20 and τ_defer at
      α=0.40 on the rescored calibration fold; EMA-smooths the τ values
      against the previous cycle's gate to keep the trajectory stable.

Runs at every cycle boundary (Cycle 1 onwards; Cycle 0 fits via the
fresh-fit ``fit_composite_calibration.py`` + ``fit_conformal_gate.py``
in ``step_7_0_calibrate``).

Usage (called by run_experiment.py at each cycle boundary):

    python scripts/recalibrate_conformal_at_cycle.py \\
        --calib_jsons outputs/full_run/cycle_3/calibration/*.json \\
        --previous_composite outputs/full_run/cycle_2/composite_calibration.json \\
        --previous_gate outputs/full_run/cycle_2/conformal_gate.json \\
        --output_composite outputs/full_run/cycle_3/composite_calibration.json \\
        --output_gate outputs/full_run/cycle_3/conformal_gate.json \\
        --ema_alpha 0.7 \\
        --alpha_store 0.20 \\
        --alpha_defer 0.40

Resume contract: on resume, the run loop loads the most recent cycle's
composite_calibration.json + conformal_gate.json. If absent, the verifier
falls back to the legacy fixed-threshold path (bootstrap behavior — same
as Cycle-0 before Step 7.0.2 fits).

Why EMA smoothing on the conformal thresholds:
  Mohri-Hashimoto 2024 + Yadkori et al. 2024 give a per-fold formal
  precision guarantee, but the threshold value can jitter across folds
  with noisy estimates of the precision-cliff. EMA-smoothing with the
  previous cycle's threshold (alpha=0.7 default) damps cycle-to-cycle
  threshold oscillation while preserving long-run convergence to the
  true τ. Same philosophy as the temperature re-fit at run_experiment.py
  ``run_per_cycle_recalibration``.

Composite calibration is NOT EMA-smoothed: the per-signal isotonic
curves are refit fresh each cycle (they're high-dimensional functions,
not scalars). EMA on isotonic knots would require careful handling of
mismatched knot counts between cycles; we punt and accept the noise of
fresh-fit isotonic each cycle (the Cherian boost layer regularizes
cross-cycle variance via its logistic-regression L2 prior).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("recalibrate_conformal_at_cycle")


def _load_samples(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for p in paths:
        with open(p) as f:
            doc = json.load(f)
        rows = doc.get("samples") or doc.get("results") or []
        for r in rows:
            if "em" in r and r["em"] in (0, 1, 0.0, 1.0):
                samples.append(r)
        logger.info("loaded %d labeled samples from %s", len(rows), p)
    return samples


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--calib_jsons",
        nargs="+",
        required=True,
        help="Calibration-fold JSON file(s) for the current cycle.",
    )
    p.add_argument(
        "--previous_composite",
        type=Path,
        default=None,
        help="Previous cycle's composite_calibration.json (optional, for "
             "logging — the composite is always re-fit fresh, no EMA on it).",
    )
    p.add_argument(
        "--previous_gate",
        type=Path,
        default=None,
        help="Previous cycle's conformal_gate.json (used for EMA smoothing).",
    )
    p.add_argument("--output_composite", type=Path, required=True)
    p.add_argument("--output_gate", type=Path, required=True)
    p.add_argument("--ema_alpha", type=float, default=0.7,
                   help="EMA smoothing factor: tau_new = alpha*tau_prev + "
                        "(1-alpha)*tau_fit. Higher = slower drift.")
    p.add_argument("--alpha_store", type=float, default=0.20)
    p.add_argument("--alpha_defer", type=float, default=0.40)
    p.add_argument("--cherian_boost", action="store_true",
                   help="Fit Cherian boost on the per-signal calibrated "
                        "log-odds. Default OFF for per-cycle (fresh isotonic "
                        "is usually enough); ON in Step 7.0.2 baseline.")
    args = p.parse_args()

    paths: List[Path] = []
    for spec in args.calib_jsons:
        paths.extend(Path(x) for x in glob.glob(spec) or [spec])
    paths = [p for p in paths if p.exists()]
    if not paths:
        logger.error("no calibration JSONs found at the supplied paths")
        return 2

    samples = _load_samples(paths)
    if len(samples) < 100:
        logger.error(
            "only %d labeled samples; need >=100 for per-cycle re-fit. "
            "Aborting (will keep previous-cycle composite + gate in place).",
            len(samples),
        )
        return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from caem.verification.cal_prob_composite import CalProbComposite
    from caem.verification.conformal_gate import ConformalStorageGate

    # ============== Step A: refit CalProbComposite (no EMA) ==============
    logger.info("Step A — refit CalProbComposite (fresh isotonic, no EMA)")
    composite = CalProbComposite().fit(samples, fit_boost=args.cherian_boost)
    composite.save(args.output_composite)
    logger.info("saved %s", args.output_composite)

    # Score the calibration fold under the new composite for gate fit
    for s in samples:
        s["u_stored"] = composite.predict(s)

    # ============== Step B: refit ConformalStorageGate ==============
    logger.info("Step B — fresh-fit ConformalStorageGate on the rescored fold")
    fresh_gate = ConformalStorageGate.fit(
        samples,
        score_key="u_stored",
        em_key="em",
        alpha_store=args.alpha_store,
        alpha_defer=args.alpha_defer,
    )
    logger.info("fresh tau_store=%.4f  tau_defer=%.4f",
                fresh_gate.tau_store, fresh_gate.tau_defer)

    # ============== Step C: EMA-smooth tau against previous cycle ==============
    smoothed_gate = fresh_gate
    prev_gate: Optional[ConformalStorageGate] = None
    if args.previous_gate and args.previous_gate.exists():
        try:
            prev_gate = ConformalStorageGate.load(args.previous_gate)
            tau_store_smooth = (
                args.ema_alpha * prev_gate.tau_store
                + (1.0 - args.ema_alpha) * fresh_gate.tau_store
            )
            tau_defer_smooth = (
                args.ema_alpha * prev_gate.tau_defer
                + (1.0 - args.ema_alpha) * fresh_gate.tau_defer
            )
            # Build a smoothed gate object (reusing all the diagnostic fields
            # from the fresh fit so the JSON is well-formed; only the τ values
            # are smoothed).
            smoothed_gate = ConformalStorageGate(
                tau_store=tau_store_smooth,
                tau_defer=tau_defer_smooth,
                alpha_store=fresh_gate.alpha_store,
                alpha_defer=fresh_gate.alpha_defer,
                cal_n=fresh_gate.cal_n,
                store_n=fresh_gate.store_n,
                store_precision=fresh_gate.store_precision,
                defer_n=fresh_gate.defer_n,
                defer_precision=fresh_gate.defer_precision,
            )
            logger.info(
                "EMA-smoothed (alpha=%.2f):  tau_store=%.4f (was %.4f, fresh %.4f)  "
                "tau_defer=%.4f (was %.4f, fresh %.4f)",
                args.ema_alpha,
                smoothed_gate.tau_store, prev_gate.tau_store, fresh_gate.tau_store,
                smoothed_gate.tau_defer, prev_gate.tau_defer, fresh_gate.tau_defer,
            )
        except Exception as exc:
            logger.warning(
                "failed to load previous gate (%s); using fresh fit (no EMA)",
                exc,
            )
    else:
        logger.info(
            "no previous gate at %s; using fresh fit (no EMA — same as Cycle-0 baseline)",
            args.previous_gate,
        )

    smoothed_gate.save(args.output_gate)
    logger.info("saved %s", args.output_gate)
    print(smoothed_gate.summary())

    # Also dump diagnostic JSON alongside for thesis evidence
    diag_path = args.output_gate.parent / (args.output_gate.stem + "_diag.json")
    with open(diag_path, "w") as f:
        json.dump({
            "fresh_fit": fresh_gate.to_dict(),
            "previous": prev_gate.to_dict() if prev_gate else None,
            "smoothed": smoothed_gate.to_dict(),
            "ema_alpha": args.ema_alpha,
            "n_samples": len(samples),
        }, f, indent=2)
    logger.info("saved diagnostic JSON %s", diag_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
