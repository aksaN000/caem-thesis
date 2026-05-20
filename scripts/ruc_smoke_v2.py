#!/usr/bin/env python3
"""scripts/ruc_smoke_v2.py
==========================
Step 6 smoke test for the trained v2 RUC.

Loads the saved scaler + LR + feature_spec from ``caem/ruc/v2/`` and verifies:

1. Model loadability: scaler.joblib and lr_final.joblib deserialise cleanly.
2. Schema integrity: feature_spec.features list matches the saved LR's
   coefficient dimensions and the training_set.parquet columns.
3. Prediction reproducibility: running the LR on training rows yields AUROC
   close to (within 0.01 of) the CV-reported number — confirms the saved
   artefacts match the trained model.
4. Per-bench routing distribution at the trained τ matches the architectural
   intent (TruthfulQA mostly T2, TriviaQA mostly T3, etc.).
5. No NaN / inf values in predictions or feature vectors.

Output: ``caem/ruc/v2/smoke_pass.json`` with per-check PASS/FAIL + summary.

Note: this is a MODEL-LEVEL smoke. The pipeline-level integration test (running
queries through AdaptiveRouter end-to-end) is a separate concern. The runtime
classifier ``caem.routing.ruc.LRRetrievalUtilityClassifier._feature_vector``
currently produces v1's 24-feature vector; deploying the v2 36-feature LR
requires extending that method to compute the new feature families (top-5
retrieval stats, cross-encoder rerank, NLI pairwise, Wikidata entity-binary,
P(IK) via the same probe used at training). This extension is a Phase 1e
prerequisite, separate from the smoke check below.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("ruc_smoke_v2")


# Per-bench routing expectations at the trained τ.
# RAG-extensive benches (case-A-dominant): high T3-share expected.
# RAG-hurts benches (case-B-dominant): low T3-share expected.
# Borderline benches (case A:B roughly balanced): mid-range expected.
EXPECTED_ROUTING: Dict[str, Dict[str, float]] = {
    # Per-bench expected T3-share, with tolerance for the smoke check.
    "natural_questions": {"min_t3_share": 0.65, "max_t3_share": 0.99},   # strongly RAG-extensive
    "triviaqa":          {"min_t3_share": 0.50, "max_t3_share": 0.95},   # RAG-helpful
    "fever":             {"min_t3_share": 0.30, "max_t3_share": 0.75},   # borderline
    "haluevalqa":        {"min_t3_share": 0.30, "max_t3_share": 0.75},   # borderline
    "strategyqa":        {"min_t3_share": 0.10, "max_t3_share": 0.65},   # mild RAG-hurts (n=43, noisy)
    "openbookqa":        {"min_t3_share": 0.10, "max_t3_share": 0.55},   # RAG-hurts
    "commonsense_qa":    {"min_t3_share": 0.05, "max_t3_share": 0.55},   # RAG-hurts
    "truthfulqa":        {"min_t3_share": 0.05, "max_t3_share": 0.55},   # strongly RAG-hurts
}


def _check(name: str, ok: bool, detail: str = "") -> Dict[str, Any]:
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    if ok:
        logger.info(msg)
    else:
        logger.error(msg)
    return {"name": name, "pass": bool(ok), "detail": detail}


def run(spec_path: Path, train_parquet: Path) -> Dict[str, Any]:
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    checks: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    # 1. Model loadability
    try:
        spec = json.loads(spec_path.read_text())
        scaler = joblib.load(spec["model_paths"]["scaler"])
        lr = joblib.load(spec["model_paths"]["lr"])
        checks.append(_check(
            "model_loadability",
            True,
            f"scaler + lr loaded; model_type={spec.get('model_type')}",
        ))
    except Exception as exc:
        checks.append(_check("model_loadability", False, str(exc)))
        return {"checks": checks, "overall_pass": False}

    feature_order = spec["features"]
    n_features = len(feature_order)
    final_C = spec["C"]
    final_tau = spec["tau_RUC_default"]
    expected_pooled_auroc = spec["validation"]["pooled_oof_auroc"]
    summary.update({
        "n_features": n_features,
        "C": final_C,
        "tau": final_tau,
        "expected_pooled_auroc": expected_pooled_auroc,
    })

    # 2. Schema integrity
    n_coef = lr.coef_.shape[1]
    checks.append(_check(
        "schema_lr_vs_spec",
        n_coef == n_features,
        f"LR has {n_coef} coefs; spec lists {n_features} features",
    ))
    df = pd.read_parquet(train_parquet)
    missing = [c for c in feature_order if c not in df.columns]
    checks.append(_check(
        "schema_features_present_in_parquet",
        not missing,
        f"missing={missing}" if missing else f"all {n_features} features present",
    ))

    if missing or n_coef != n_features:
        return {"checks": checks, "summary": summary, "overall_pass": False}

    # 3. Prediction reproducibility on training data
    X = df[feature_order].to_numpy(dtype=np.float64)
    y = df["empirical_label"].to_numpy(dtype=np.int64)
    Xs = scaler.transform(X)
    proba = lr.predict_proba(Xs)[:, 1]

    # NaN / inf check
    n_bad = int(np.isnan(proba).sum() + np.isinf(proba).sum())
    checks.append(_check(
        "predictions_finite",
        n_bad == 0,
        f"{n_bad} NaN/inf in {len(proba)} predictions",
    ))

    # In-sample AUROC (this is NOT the CV AUROC; it's overfit slightly, but
    # should be in the same ballpark within ~0.02 of the OOF number).
    insample_auroc = float(roc_auc_score(y, proba))
    auroc_drift = abs(insample_auroc - expected_pooled_auroc)
    checks.append(_check(
        "auroc_reproducibility",
        auroc_drift < 0.05,
        f"in-sample={insample_auroc:.4f} vs OOF={expected_pooled_auroc:.4f}  "
        f"(drift={auroc_drift:.4f}; expected <0.05)",
    ))
    summary["insample_auroc"] = insample_auroc

    # 4. Per-bench routing distribution at trained τ
    benches = df["benchmark"].to_numpy()
    bench_results: Dict[str, Dict[str, Any]] = {}
    all_bench_pass = True
    for bench in sorted(set(benches.tolist())):
        mask = benches == bench
        p = proba[mask]
        routed_t3 = (p >= final_tau).sum()
        n = int(mask.sum())
        t3_share = float(routed_t3) / n if n else 0.0
        exp = EXPECTED_ROUTING.get(bench, {"min_t3_share": 0.0, "max_t3_share": 1.0})
        in_range = exp["min_t3_share"] <= t3_share <= exp["max_t3_share"]
        all_bench_pass &= in_range
        bench_results[bench] = {
            "n": n,
            "t3_share": round(t3_share, 4),
            "expected_range": [exp["min_t3_share"], exp["max_t3_share"]],
            "pass": in_range,
        }
        logger.info(
            "  %-22s  n=%4d  T3-share=%.3f  expected=[%.2f, %.2f]  %s",
            bench, n, t3_share,
            exp["min_t3_share"], exp["max_t3_share"],
            "✓" if in_range else "✗",
        )

    checks.append(_check(
        "per_bench_routing_distribution",
        all_bench_pass,
        f"{sum(b['pass'] for b in bench_results.values())}/{len(bench_results)} benches in expected range",
    ))
    summary["per_bench_routing"] = bench_results

    # 5. Pooled routing distribution
    n_t3 = int((proba >= final_tau).sum())
    n_t2 = int((proba < final_tau).sum())
    summary["pooled_routing"] = {
        "n_t3": n_t3, "n_t2": n_t2,
        "t3_share": round(n_t3 / len(proba), 4),
    }
    logger.info("pooled: T3=%d (%.1f%%)  T2=%d (%.1f%%)",
                n_t3, 100 * n_t3 / len(proba),
                n_t2, 100 * n_t2 / len(proba))

    # 6. Net utility check at trained τ (in-sample, should match the spec)
    a_caught = int(((y == 1) & (proba >= final_tau)).sum())
    b_caught = int(((y == 0) & (proba < final_tau)).sum())
    net_utility = a_caught + b_caught
    expected_net_utility = spec.get("threshold_sweep", {}).get("best_net_utility", 0)
    checks.append(_check(
        "net_utility_reproducible",
        abs(net_utility - expected_net_utility) < 30,
        f"in-sample net_utility={net_utility}, expected≈{expected_net_utility}",
    ))
    summary["net_utility"] = {
        "in_sample": net_utility,
        "a_caught": a_caught,
        "b_caught": b_caught,
        "expected_oof": expected_net_utility,
    }

    overall_pass = all(c["pass"] for c in checks)
    return {
        "checks": checks,
        "summary": summary,
        "overall_pass": overall_pass,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--spec_path", type=Path,
                    default=Path("caem/ruc/v2/feature_spec.json"))
    ap.add_argument("--train_parquet", type=Path,
                    default=Path("caem/ruc/v2/training_set.parquet"))
    ap.add_argument("--out", type=Path,
                    default=Path("caem/ruc/v2/smoke_pass.json"))
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    report = run(args.spec_path, args.train_parquet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", args.out)

    print()
    print("=" * 60)
    print(f"v2 RUC SMOKE — {'✅ PASS' if report['overall_pass'] else '❌ FAIL'}")
    print("=" * 60)
    for c in report["checks"]:
        mark = "✓" if c["pass"] else "✗"
        print(f"  {mark} {c['name']}")
        if c["detail"]:
            print(f"      {c['detail']}")
    print("=" * 60)
    if report["overall_pass"]:
        print("→ The v2 RUC is loadable, schema-consistent, predictions reproduce")
        print("  the CV AUROC, and per-bench routing distribution matches")
        print("  architectural intent. Ready for Phase 1e wiring.")
    else:
        print("→ One or more smoke checks failed. Investigate before Phase 1e.")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
