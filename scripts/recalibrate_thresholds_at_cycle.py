#!/usr/bin/env python
"""
scripts/recalibrate_thresholds_at_cycle.py
==========================================
Per-cycle threshold re-fit with EMA smoothing — opt-in adaptive
threshold mechanism (CAEMConfig.adaptive_thresholds_per_cycle).

Mirrors the existing per-cycle temperature re-fit pattern in
run_experiment.py::run_per_cycle_recalibration. Where T is re-fit
because fine-tuning shifts model logits, the thresholds are re-fit
because SIL training shifts the u_stored distribution. Same calibration
fold, same disjointness guarantee, same EMA smoothing philosophy.

Quantile targets are inherited from scripts/calibrate_thresholds.py
(P70 store, P40 defer, P90 train), so this module just refits with
those targets on the cycle-N calibration fold and smooths with cycle-(N-1).

Usage (called by run_experiment.py at each cycle boundary)
----------------------------------------------------------
    python scripts/recalibrate_thresholds_at_cycle.py \\
        --calib_jsons outputs/full_run/cycle_3/calibration/*.json \\
        --previous_thresholds outputs/full_run/cycle_2/calibrated_thresholds.json \\
        --output_json outputs/full_run/cycle_3/calibrated_thresholds.json \\
        --ema_alpha 0.7

Resume contract (mirrors temperature loader at run_experiment.py:1247)
----------------------------------------------------------------------
On resume, the run loop loads the most recent cycle's
calibrated_thresholds_cycle{N}.json. If absent, falls back to
outputs/cycle_0/calibrated_thresholds.json (Step 7.0.2 baseline).

Phase 1a research mode (default OFF): this module is never invoked.
Production deployment: enable via CAEMConfig.adaptive_thresholds_per_cycle = True.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("recalibrate_thresholds_at_cycle")

DEFAULT_TARGETS = {"store": 0.70, "defer": 0.40, "train": 0.90}


def _load_u_stored(calib_jsons: List[Path]) -> List[float]:
    values: List[float] = []
    for p in calib_jsons:
        if "/eval/" in str(p) or str(p).endswith("_eval.json"):
            raise RuntimeError(
                f"Refused to fit on evaluation-fold JSON: {p}. Use calibration-fold only."
            )
        with open(p) as f:
            d = json.load(f)
        for s in d.get("samples", []):
            v = s.get("u_stored")
            if isinstance(v, (int, float)) and np.isfinite(v):
                values.append(float(v))
    return values


def _fit_quantile_thresholds(values: List[float], targets: Dict[str, float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "store": float(np.quantile(arr, targets["store"])),
        "defer": float(np.quantile(arr, targets["defer"])),
        "train": float(np.quantile(arr, targets["train"])),
    }


def _ema_smooth(
    prev: Dict[str, float], current: Dict[str, float], alpha: float
) -> Dict[str, float]:
    """tau_new = alpha * prev + (1 - alpha) * current."""
    return {
        k: alpha * prev[k] + (1.0 - alpha) * current[k]
        for k in ("store", "defer", "train")
    }


def _validate_ordering(thresholds: Dict[str, float]) -> None:
    if not (thresholds["train"] > thresholds["store"] > thresholds["defer"]):
        raise RuntimeError(
            f"Adaptive thresholds violate design ordering: "
            f"train={thresholds['train']:.4f} > "
            f"store={thresholds['store']:.4f} > "
            f"defer={thresholds['defer']:.4f}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--calib_jsons", nargs="+", required=True, type=str)
    p.add_argument("--previous_thresholds", type=Path, required=False, default=None,
                   help="Previous cycle's calibrated_thresholds.json (for EMA). "
                        "If absent, no smoothing — direct fit.")
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--ema_alpha", type=float, default=0.7,
                   help="EMA smoothing factor (default 0.7 = 70% prev, 30% current).")
    p.add_argument("--target_store_quantile", type=float, default=DEFAULT_TARGETS["store"])
    p.add_argument("--target_defer_quantile", type=float, default=DEFAULT_TARGETS["defer"])
    p.add_argument("--target_train_quantile", type=float, default=DEFAULT_TARGETS["train"])
    p.add_argument("--min_samples", type=int, default=100)
    ns = p.parse_args()

    # Expand glob patterns
    expanded: List[Path] = []
    for pat in ns.calib_jsons:
        matches = glob.glob(pat)
        if not matches:
            logger.warning("No files match %r", pat)
        expanded.extend(Path(m) for m in matches)

    if not expanded:
        logger.error("No calibration JSONs resolved.")
        return 1

    values = _load_u_stored(expanded)
    if len(values) < ns.min_samples:
        logger.error(
            "Only %d u_stored samples available (need >= %d). Skipping refit.",
            len(values), ns.min_samples,
        )
        return 1

    targets = {
        "store": ns.target_store_quantile,
        "defer": ns.target_defer_quantile,
        "train": ns.target_train_quantile,
    }
    current_fit = _fit_quantile_thresholds(values, targets)
    logger.info(
        "Cycle fit (n=%d):  store=%.4f  defer=%.4f  train=%.4f",
        len(values), current_fit["store"], current_fit["defer"], current_fit["train"],
    )

    # EMA-smooth with previous cycle if available
    smoothed = current_fit
    prev_loaded: Dict[str, float] | None = None
    if ns.previous_thresholds and ns.previous_thresholds.exists():
        try:
            with open(ns.previous_thresholds) as f:
                prev_data = json.load(f)
            prev_loaded = prev_data.get("thresholds") or prev_data
            smoothed = _ema_smooth(prev_loaded, current_fit, ns.ema_alpha)
            logger.info(
                "EMA-smoothed (alpha=%.2f):  store=%.4f  defer=%.4f  train=%.4f",
                ns.ema_alpha, smoothed["store"], smoothed["defer"], smoothed["train"],
            )
        except Exception as exc:
            logger.warning(
                "Failed to load previous thresholds (%s); using direct fit.", exc,
            )
    else:
        logger.info("No previous thresholds; using direct fit (no EMA).")

    _validate_ordering(smoothed)

    out = {
        "thresholds": smoothed,
        "current_fit": current_fit,
        "previous_thresholds": prev_loaded,
        "ema_alpha": ns.ema_alpha,
        "target_quantiles": targets,
        "n_samples": len(values),
        "u_stored_stats": {
            "min": float(np.min(values)),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        },
        "source_jsons": [str(p) for p in expanded],
    }
    ns.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(ns.output_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", ns.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
