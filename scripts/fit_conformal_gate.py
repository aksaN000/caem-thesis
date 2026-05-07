#!/usr/bin/env python
"""Fit the conformal storage gate (τ_store + τ_defer) at calibrated precisions.

Phase 2.4 fitter — companion to scripts/fit_composite_calibration.py.
Reads labeled calibration-fold JSONs (with em + u_stored already computed
under the post-2.1 cal-prob composite), fits τ_store at α_store and
τ_defer at α_defer, and writes the gate JSON read by UnifiedVerifier at
construction time.

Order in run_phase1a.sh's step_7_0_calibrate:
    1. Run Cycle-0 eval to produce calibration_fold_samples.json
       (already in the existing pipeline)
    2. python scripts/fit_composite_calibration.py
         --calib_jsons outputs/cycle_0/calibration/calibration_fold_samples.json
         --output_json outputs/cycle_0/composite_calibration.json
         --cherian_boost
    3. Re-score the calibration fold using the fitted CalProbComposite
       (this script does it in-memory; no separate run needed)
    4. python scripts/fit_conformal_gate.py
         --calib_jsons outputs/cycle_0/calibration/calibration_fold_samples.json
         --composite_calibration_json outputs/cycle_0/composite_calibration.json
         --output_json outputs/cycle_0/conformal_gate.json
         --alpha_store 0.20  --alpha_defer 0.40

Usage notes:
  - α_store=0.20 targets 80% precision in the STORE band (Tier-1 +
    training-pool eligible).
  - α_defer=0.40 targets 60% precision in the DEFERRED band (memory only).
  - If --composite_calibration_json is omitted, the script uses the
    existing u_stored values stored in the calibration JSON (i.e., the
    pre-Phase-2.1 weighted-sum composite). Useful for ablation.
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
logger = logging.getLogger("fit_conformal_gate")


def _load_samples(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for p in paths:
        with open(p) as f:
            doc = json.load(f)
        rows = doc.get("samples") or doc.get("results") or []
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
        required=True,
        help="Calibration-fold JSON file(s) — must contain em labels.",
    )
    p.add_argument("--output_json", required=True)
    p.add_argument(
        "--composite_calibration_json",
        default=None,
        help="Optional CalProbComposite JSON. If supplied, samples are "
             "rescored with it before fitting the gate. If omitted, the "
             "existing u_stored field in the calibration JSON is used "
             "(legacy weighted-sum composite, useful for ablation).",
    )
    p.add_argument("--alpha_store", type=float, default=0.20)
    p.add_argument("--alpha_defer", type=float, default=0.40)
    # v2 Fix 1: per-benchmark conformal gate. Default ON in v2 so the
    # cycle-0 fit produces the nested {global, per_benchmark} JSON the
    # runtime gate dispatches against. --no-fit_per_benchmark falls
    # back to the v1 pooled-only fit (used by the back-compat ablation
    # sweep).
    p.add_argument(
        "--fit_per_benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="v2 default: fit a per-benchmark ConformalStorageGate "
             "(pooled global + per-bench children). "
             "--no-fit_per_benchmark reverts to the v1 pooled-only fit.",
    )
    # Optional per-benchmark α overrides. Empty default → every
    # per-benchmark child uses the pooled --alpha_store / --alpha_defer
    # values; pass e.g. fever=0.05 triviaqa=0.40 to relax FEVER while
    # keeping TQA strict.
    p.add_argument(
        "--alpha_store_overrides", nargs="*", default=[],
        help="Optional per-bench α_store overrides as 'bench=value' pairs.",
    )
    p.add_argument(
        "--alpha_defer_overrides", nargs="*", default=[],
        help="Optional per-bench α_defer overrides as 'bench=value' pairs.",
    )
    args = p.parse_args()

    def _parse_kv_list(raw: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for entry in raw:
            if "=" not in entry:
                logger.warning("skipping malformed override %r (expected bench=value)", entry)
                continue
            k, v = entry.split("=", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                logger.warning("skipping malformed override %r (bad float)", entry)
        return out

    alpha_store_overrides = _parse_kv_list(args.alpha_store_overrides)
    alpha_defer_overrides = _parse_kv_list(args.alpha_defer_overrides)

    paths: List[Path] = []
    for spec in args.calib_jsons:
        paths.extend(Path(x) for x in glob.glob(spec) or [spec])
    paths = [p for p in paths if p.exists()]
    if not paths:
        logger.error("no calibration JSONs found")
        return 2

    samples = _load_samples(paths)
    if len(samples) < 200:
        logger.error("need ≥200 labeled samples; got %d", len(samples))
        return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from caem.verification.conformal_gate import ConformalStorageGate

    if args.composite_calibration_json:
        from caem.verification.cal_prob_composite import CalProbComposite
        cp = CalProbComposite.load(args.composite_calibration_json)
        for s in samples:
            s["u_stored"] = cp.predict(s)
        logger.info(
            "rescored %d samples with CalProbComposite from %s",
            len(samples), args.composite_calibration_json,
        )

    if args.fit_per_benchmark:
        # Group samples by benchmark for the per-bench fit. Untagged
        # samples are dropped from per-bench fitting; the parent class-
        # method's pooled-global path still sees the union internally.
        samples_by_bench: Dict[str, List[Dict[str, Any]]] = {}
        for s in samples:
            bm = s.get("benchmark") or "_untagged"
            samples_by_bench.setdefault(bm, []).append(s)
        per_bench_input = {
            bm: rows for bm, rows in samples_by_bench.items()
            if bm != "_untagged"
        }
        logger.info(
            "per-benchmark conformal-gate fit: %d benchmarks (%s); "
            "untagged samples = %d.",
            len(per_bench_input),
            sorted(per_bench_input.keys()),
            len(samples_by_bench.get("_untagged", [])),
        )
        gate = ConformalStorageGate.fit_per_benchmark(
            per_bench_input,
            default_alpha_store=args.alpha_store,
            default_alpha_defer=args.alpha_defer,
            alpha_store_overrides=alpha_store_overrides,
            alpha_defer_overrides=alpha_defer_overrides,
        )
    else:
        gate = ConformalStorageGate.fit(
            samples,
            score_key="u_stored",
            em_key="em",
            alpha_store=args.alpha_store,
            alpha_defer=args.alpha_defer,
        )
    gate.save(args.output_json)
    print(gate.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
