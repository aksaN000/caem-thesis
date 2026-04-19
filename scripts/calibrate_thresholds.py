"""
scripts/calibrate_thresholds.py
================================
Fit CAEM's decision-tree thresholds from Cycle-0 evaluation u_stored
values, matching the per-cycle temperature-calibration pattern
documented in Chapter 4 "Temperature scaling".

Why
---
The default CAEMConfig thresholds (tau_store=0.65, tau_defer=0.45,
tau_train=0.75) were tuned against RoBERTa-MNLI's p_entail
distribution. Under the MiniCheck-Flan-T5-Large verifier the u_stored
composite peaks around 0.55 rather than ~0.85; fixed RoBERTa-era
thresholds yield approximately zero STORE decisions in Cycle 0,
starving the self-improvement loop.

This script replaces hard-coded thresholds with quantile-targeted
thresholds fit once on the **Cycle-0 n_cal calibration fold** and
held fixed for cycles 1..N. The quantile targets (not the thresholds
themselves) are the design-time hyperparameters; the thresholds are
outputs of the calibration step.

Disjointness requirement (from Chapter 5 Implementation Details)
----------------------------------------------------------------
The calibration fold (n_cal=500 per training benchmark) must be
disjoint from the evaluation fold (n_eval=500 per benchmark), the
purity fold (n_purity=500), the MMLU retention probe (n_MMLU=200),
and the screening fold (n_screen=1500). Fitting thresholds on a fold
that later evaluates thesis metrics would constitute peeking at the
test set and inflate headline comparisons. This script therefore
consumes **calibration-fold JSONs only** and includes a safety guard
against accidentally being pointed at evaluation-fold JSONs.

Target quantiles
----------------
    tau_store  ->  target 30 percent STORE rate  ->  P70
    tau_defer  ->  target 30 percent DEFERRED band below STORE  ->  P40
    tau_train  ->  top 10 percent of u_stored  ->  P90
                   (strictly above tau_store by design)

These targets keep the relative ordering the thesis commits to
(tau_train > tau_store > tau_defer) while letting the absolute values
track the observed composite distribution of whichever verifier
backend is in use.

Input
-----
One or more per-benchmark Cycle-0 **calibration-fold** JSONs produced
by eval.harness (emitted via ``harness.run_all(calib_samples, ...)``
under the per-cycle recalibration pass), typically at
``outputs/cycle_0/calibration/*_cycle0.json``. Each sample record
carries u_stored as a scalar field.

Safety guard: paths containing the string ``eval/`` are refused
(those are the evaluation-fold JSONs and would leak test-set
information into the threshold fit).

Output
------
JSON dict at ``--output_json`` with the fitted threshold values plus
the observed quantile at each target point, a source-data provenance
block, and a pass-through echo of ``--target_quantiles`` so the fit
is fully reproducible:

    {
        "verifier_backend": "minicheck",
        "target_quantiles": {"store": 0.70, "defer": 0.40, "train": 0.90},
        "thresholds": {"store": ..., "defer": ..., "train": ...},
        "n_samples": 300,
        "u_stored_stats": {"min": ..., "p25": ..., "median": ...,
                           "p75": ..., "max": ..., "mean": ..., "std": ...},
        "source_jsons": [...]
    }

Usage
-----
python scripts/calibrate_thresholds.py \\
    --calib_jsons outputs/cycle_0/calibration/*_cycle0.json \\
    --verifier_backend minicheck \\
    --output_json outputs/cycle_0/calibrated_thresholds.json

run_experiment.py then reads this file at cycle-1 boundary and
overrides CAEMConfig.store_threshold / defer_threshold / train_threshold
for all subsequent cycles.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("calibrate_thresholds")


DEFAULT_TARGETS = {
    "store": 0.70,   # P70 -> top 30 percent of u_stored gets STORE
    "defer": 0.40,   # P40 -> middle 30 percent gets DEFERRED
    "train": 0.90,   # P90 -> top 10 percent gets training-pool admission
}


def _load_u_stored(calib_jsons: List[Path]) -> List[float]:
    values: List[float] = []
    for p in calib_jsons:
        # Safety guard: refuse evaluation-fold JSONs. Fitting thresholds on
        # the eval fold and then evaluating Cycle 1..N metrics on the same
        # fold would constitute peeking at the test set (Chapter 5
        # Implementation Details requires disjoint splits).
        if "/eval/" in str(p) or str(p).endswith("_eval.json"):
            raise RuntimeError(
                f"Refused to fit thresholds on evaluation-fold JSON: {p}. "
                f"Use calibration-fold JSONs only (typically under "
                f"outputs/cycle_0/calibration/). See Chapter 4 \u00a7 Threshold "
                f"calibration + Chapter 5 \u00a7 Implementation Details."
            )
        with open(p) as f:
            d = json.load(f)
        for s in d.get("samples", []):
            v = s.get("u_stored")
            if isinstance(v, (int, float)) and np.isfinite(v):
                values.append(float(v))
    return values


def _quantile(values: np.ndarray, q: float) -> float:
    """Linear-interpolation quantile, numpy default."""
    return float(np.quantile(values, q))


def _validate_ordering(thresholds: Dict[str, float]) -> None:
    """The design invariant tau_train > tau_store > tau_defer must hold.

    Quantile targets enforce this by construction, but small sample sizes
    can violate the ordering due to ties. Abort with a descriptive error
    so the caller fixes the input rather than silently shipping a broken
    decision tree.
    """
    if not (thresholds["train"] > thresholds["store"] > thresholds["defer"]):
        raise RuntimeError(
            "Calibrated thresholds violate design ordering: "
            f"train={thresholds['train']:.4f} > "
            f"store={thresholds['store']:.4f} > "
            f"defer={thresholds['defer']:.4f} is not satisfied. Either "
            "tighten the target quantiles or collect more Cycle-0 samples."
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Preferred flag: calibration-fold JSONs (disjoint from eval fold).
    # Legacy alias --eval_jsons kept for back-compat; triggers a warning
    # and runs the safety guard on every provided path.
    p.add_argument(
        "--calib_jsons", nargs="+", required=False, type=str,
        help="Glob(s) of Cycle-0 calibration-fold JSONs, e.g. "
             "outputs/cycle_0/calibration/*_cycle0.json.",
    )
    p.add_argument(
        "--eval_jsons", nargs="+", required=False, type=str,
        help="DEPRECATED alias for --calib_jsons. Will be rejected if the "
             "paths match the evaluation-fold pattern.",
    )
    p.add_argument("--output_json", required=True, type=Path)
    p.add_argument("--verifier_backend", type=str, default="minicheck")
    p.add_argument(
        "--target_store_quantile", type=float,
        default=DEFAULT_TARGETS["store"],
        help="Quantile of u_stored above which STORE fires (0.70 = top 30%%).",
    )
    p.add_argument(
        "--target_defer_quantile", type=float,
        default=DEFAULT_TARGETS["defer"],
        help="Quantile above which DEFERRED fires (0.40 = top 60%%).",
    )
    p.add_argument(
        "--target_train_quantile", type=float,
        default=DEFAULT_TARGETS["train"],
        help="Quantile above which training-pool admission fires (0.90 = top 10%%).",
    )
    p.add_argument(
        "--min_samples", type=int, default=100,
        help="Refuse to fit if fewer samples are available; avoids noisy quantiles.",
    )
    return p.parse_args()


def main() -> None:
    ns = _parse_args()

    # Resolve which fold source was provided; prefer --calib_jsons.
    source = ns.calib_jsons or ns.eval_jsons
    if source is None:
        logger.error("Provide --calib_jsons (preferred) or --eval_jsons.")
        sys.exit(1)
    if ns.eval_jsons and not ns.calib_jsons:
        logger.warning(
            "--eval_jsons is a deprecated alias. Use --calib_jsons to fit "
            "thresholds on the n_cal=500 calibration fold (Chapter 4 "
            "\u00a7 Threshold calibration). Safety guard will reject "
            "anything that looks like an evaluation-fold JSON.",
        )

    expanded: List[Path] = []
    for pat in source:
        matches = glob.glob(pat)
        if not matches:
            logger.warning("No files match %r", pat)
        expanded.extend(Path(m) for m in matches)
    if not expanded:
        logger.error("No calibration JSONs found. Exiting.")
        sys.exit(1)

    values = _load_u_stored(expanded)
    if len(values) < ns.min_samples:
        logger.error(
            "Only %d u_stored samples available; refusing to fit (min=%d). "
            "Increase Cycle-0 eval size or lower --min_samples if you "
            "accept the quantile-noise risk.",
            len(values), ns.min_samples,
        )
        sys.exit(1)
    arr = np.asarray(values, dtype=np.float64)

    targets = {
        "store": ns.target_store_quantile,
        "defer": ns.target_defer_quantile,
        "train": ns.target_train_quantile,
    }
    thresholds = {k: _quantile(arr, q) for k, q in targets.items()}
    _validate_ordering(thresholds)

    stats = {
        "n_samples": int(arr.size),
        "min": float(arr.min()),
        "p10": _quantile(arr, 0.10),
        "p25": _quantile(arr, 0.25),
        "median": _quantile(arr, 0.50),
        "p75": _quantile(arr, 0.75),
        "p90": _quantile(arr, 0.90),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }
    payload: Dict[str, Any] = {
        "verifier_backend": ns.verifier_backend,
        "target_quantiles": targets,
        "thresholds": thresholds,
        "n_samples": int(arr.size),
        "u_stored_stats": stats,
        "source_jsons": [str(p) for p in expanded],
    }

    ns.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(ns.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "Fitted thresholds (backend=%s): store=%.4f  defer=%.4f  train=%.4f  "
        "(n=%d, median=%.3f, max=%.3f)",
        ns.verifier_backend,
        thresholds["store"], thresholds["defer"], thresholds["train"],
        arr.size, stats["median"], stats["max"],
    )
    logger.info("Wrote %s", ns.output_json)


if __name__ == "__main__":
    main()
