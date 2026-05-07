#!/usr/bin/env python
"""Fit the per-signal CalProbComposite calibration on a labeled fold.

Reads one or more Cycle-0 JSON files (must have ``samples`` with ``em``
plus the 10 verifier signals), fits a per-signal isotonic regression,
and writes the calibration JSON read by ``UnifiedVerifier`` at
construction time when ``CAEMConfig.composite_mode`` is ``"auto"`` or
``"cal_prob"``.

Usage (called by run_phase1a.sh at step_7_0_calibrate):

    python scripts/fit_composite_calibration.py \\
        --calib_jsons outputs/cycle_0/calibration/calibration_fold_samples.json \\
        --output_json outputs/cycle_0/composite_calibration.json

Or fit on all eval JSONs (diagnostic, not recommended for thesis runs
because it leaks eval-fold data into the calibration):

    python scripts/fit_composite_calibration.py \\
        --eval_dir outputs/cycle_0/eval \\
        --output_json /tmp/composite_calibration_diagnostic.json

Phase 1a research mode: invoked once at Step 7.0.2 immediately after the
Cycle-0 calibration fold is written. The fitted JSON then drives the
verifier composite for the entire Step 7 main self-improvement loop;
``recalibrate_thresholds_at_cycle.py`` may re-fit at each cycle boundary
on the cycle's own labeled fold.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fit_composite_calibration")


def _load_samples(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for p in paths:
        with open(p) as f:
            doc = json.load(f)
        rows = doc.get("samples") or doc.get("results") or []
        if not rows:
            logger.warning("no samples found in %s", p)
            continue
        for r in rows:
            if "em" in r and r["em"] in (0, 1, 0.0, 1.0):
                samples.append(r)
        logger.info("loaded %d samples from %s", len(rows), p)
    return samples


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--calib_jsons",
        nargs="+",
        help="Calibration-fold JSON file(s) — preferred. "
             "Disjoint from the evaluation fold.",
    )
    p.add_argument(
        "--eval_dir",
        help="Directory containing eval-fold JSONs. ALSO accepted for "
             "diagnostic fits, but these leak eval data into the "
             "calibration; do NOT use for thesis runs.",
    )
    p.add_argument(
        "--output_json",
        required=True,
        help="Where to write the fitted CalProbComposite JSON.",
    )
    p.add_argument(
        "--cherian_boost",
        action="store_true",
        help="Additionally fit Cherian-style logistic regression on per-signal "
             "calibrated probabilities (default: identity weights / plain log-odds sum).",
    )
    p.add_argument(
        "--boost_C",
        type=float,
        default=1.0,
        help="L2 regularization strength for Cherian boost (passed to "
             "sklearn LogisticRegression as C). Lower = stronger regularization. "
             "Default 1.0 matches sklearn default; 0.01 used in production "
             "(see branch_C_log.md 2026-04-26 entry + outputs/cycle_0/sweep/).",
    )
    # v2 Fix 2B: per-benchmark composite calibration. Default ON in v2
    # so the cycle-0 fit produces the nested {global, per_benchmark}
    # JSON the runtime verifier dispatches against. Pass
    # --no-fit_per_benchmark to fall back to the v1 pooled-only fit
    # (used by the back-compat ablation sweep).
    p.add_argument(
        "--fit_per_benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="v2 default: fit a per-benchmark CalProbComposite (pooled "
             "global + per-bench children). --no-fit_per_benchmark "
             "reverts to the v1 pooled-only fit.",
    )
    args = p.parse_args()

    paths: List[Path] = []
    if args.calib_jsons:
        for spec in args.calib_jsons:
            paths.extend(Path(x) for x in glob.glob(spec) or [spec])
    elif args.eval_dir:
        eval_paths = sorted(Path(args.eval_dir).glob("*_cycle*.json"))
        if not eval_paths:
            logger.error("--eval_dir %s contains no *_cycle*.json files", args.eval_dir)
            return 2
        paths.extend(eval_paths)
        logger.warning(
            "fitting on EVAL JSONs (diagnostic mode). Calibration leakage; "
            "do not ship the resulting JSON for thesis runs."
        )
    else:
        logger.error("provide --calib_jsons or --eval_dir")
        return 2

    paths = [p for p in paths if p.exists()]
    if not paths:
        logger.error("no calibration JSONs found at the supplied paths")
        return 2

    samples = _load_samples(paths)
    if len(samples) < 200:
        logger.error(
            "only %d labeled samples available; need ≥200 for stable "
            "per-signal isotonic. Aborting.",
            len(samples),
        )
        return 2

    em_pos = sum(1 for s in samples if s["em"] in (1, 1.0))
    em_neg = sum(1 for s in samples if s["em"] in (0, 0.0))
    logger.info(
        "fitting CalProbComposite on n=%d (em=1: %d  em=0: %d  rate=%.3f)",
        len(samples), em_pos, em_neg, em_pos / max(1, em_pos + em_neg),
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from caem.verification.cal_prob_composite import CalProbComposite

    if args.fit_per_benchmark:
        # Group samples by their benchmark tag. Samples without a
        # benchmark tag (legacy / unit-test paths) feed into the
        # pooled global only.
        samples_by_bench: Dict[str, List[Dict[str, Any]]] = {}
        for s in samples:
            bm = s.get("benchmark") or "_untagged"
            samples_by_bench.setdefault(bm, []).append(s)
        # Drop the untagged bucket from per-bench fitting; it still
        # contributes to the pooled global via fit_per_benchmark's
        # internal flat-fit on the union of all samples.
        per_bench_input = {
            bm: rows for bm, rows in samples_by_bench.items()
            if bm != "_untagged"
        }
        logger.info(
            "per-benchmark composite fit: %d benchmarks (%s); "
            "untagged samples = %d.",
            len(per_bench_input),
            sorted(per_bench_input.keys()),
            len(samples_by_bench.get("_untagged", [])),
        )
        calib = CalProbComposite()
        calib.fit_per_benchmark(
            per_bench_input,
            fit_boost=args.cherian_boost,
            boost_C=args.boost_C,
        )
    else:
        calib = CalProbComposite().fit(
            samples, fit_boost=args.cherian_boost, boost_C=args.boost_C,
        )
    calib.save(args.output_json)
    logger.info("saved %s", args.output_json)
    print(calib.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
