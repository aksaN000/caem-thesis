#!/usr/bin/env python3
"""scripts/inspect_ruc.py
=============================
Manual-inspection helper for a trained RUC.

Run AFTER ``scripts/train_ruc.py``. Reads the LOBO-CV artefacts and the
training parquet, and produces three diagnostics to eyeball before shipping:

1. SHAP top-20 feature contributions for the ship flavour (TruthfulQA fold).
2. Five worst wrong predictions per benchmark — both directions:
   - high p_rag (RUC said RAG) where RAG was actually worse
   - low p_rag (RUC said DIRECT) where RAG was actually better
3. Per-benchmark confusion matrix at the optimal τ_RUC.

Outputs
-------
- ``caem/ruc/results/shap_summary.png``   (TruthfulQA fold SHAP)
- ``caem/ruc/results/inspect_report.md``  (human-readable summary)

Usage
-----
    python -m scripts.inspect_ruc --out_dir caem/ruc
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENCHES = ["fever", "triviaqa", "commonsense_qa", "strategyqa", "truthfulqa"]


def confusion_at_threshold(p, y, tau):
    pred = (np.asarray(p) >= tau).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(tp / max(tp + fp, 1), 3),
            "recall":    round(tp / max(tp + fn, 1), 3),
            "accuracy":  round((tp + tn) / max(len(y), 1), 3)}


def find_wrong_predictions(df, p_col: str, y_col: str, threshold: float, k: int = 5):
    """Top-k FP and top-k FN samples, sorted by confidence of error."""
    df = df.copy()
    df["p"] = df[p_col]
    df["y"] = df[y_col]
    df["pred"] = (df["p"] >= threshold).astype(int)
    fp = df[(df["pred"] == 1) & (df["y"] == 0)].sort_values("p", ascending=False).head(k)
    fn = df[(df["pred"] == 0) & (df["y"] == 1)].sort_values("p", ascending=True).head(k)
    return fp, fn


def make_shap_plot(model, X, feature_names, out_path: Path) -> None:
    """Save a SHAP summary plot for the LightGBM ship-flavour model."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning("SHAP/matplotlib not available: %s", e)
        return

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    # LightGBM binary models can return either a single array or [neg, pos]
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X, feature_names=feature_names, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info("SHAP plot -> %s", out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--out_dir", type=Path, default=Path("caem/ruc"))
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    metrics_path = args.out_dir / "results" / "lobo_cv_metrics.json"
    if not metrics_path.exists():
        logger.error("No lobo_cv_metrics.json — run scripts/train_ruc.py first.")
        return 1
    results = json.loads(metrics_path.read_text())
    ship = results["ship_flavour"]
    flav = results["flavours"][ship]
    tau = flav["threshold_sweep"]["tau_optimal"]
    logger.info(f"Ship flavour: {ship}; τ_RUC optimal: {tau}")

    import pandas as pd
    df = pd.read_parquet(args.in_parquet)
    df = df.dropna(subset=["training_label"]).reset_index(drop=True)

    # We need per-row p_rag predictions. The LOBO-CV models predict on the
    # held-out fold only, so we use those predictions directly from the
    # metrics JSON (which stored test_ids + p_test_cal per fold).
    # If those aren't present, re-load each LOBO model and predict.
    # For simplicity we use the deployment FINAL model on all data here.
    import lightgbm as lgb
    import joblib
    from scripts.train_ruc import assemble_X

    final_model_path = args.out_dir / "models" / f"ruc_lightgbm_{ship}_FINAL.txt"
    final_iso_path   = args.out_dir / "calibration" / f"isotonic_{ship}_FINAL.joblib"
    tfidf_vec = None
    if ship in {"A2", "A4"}:
        tfidf_path = args.out_dir / "models" / f"tfidf_{ship}_FINAL.joblib"
        if tfidf_path.exists():
            tfidf_vec = joblib.load(tfidf_path)

    if not final_model_path.exists():
        logger.error("Final deployment model missing at %s. Cannot run inspection.",
                     final_model_path)
        return 1
    booster = lgb.Booster(model_file=str(final_model_path))
    iso = joblib.load(final_iso_path)

    # Predict on every row
    X, _ = assemble_X(df, ship, tfidf_vec=tfidf_vec)
    p_raw = booster.predict(X)
    p_cal = iso.predict(p_raw)
    df["p_rag_cal"] = p_cal
    df["pred_rag"] = (p_cal >= tau).astype(int)

    # SHAP on the ship model — sample 200 rows for plot stability
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X), size=min(200, len(X)), replace=False)
    X_sample = X[sample_idx]
    from scripts.train_ruc import feature_names
    names = feature_names(ship, tfidf_vec=tfidf_vec)
    shap_path = args.out_dir / "results" / "shap_summary.png"
    make_shap_plot(booster, X_sample, names, shap_path)

    # Per-benchmark confusion matrix + wrong predictions
    report_lines: List[str] = []
    report_lines.append(f"# RUC inspection report — ship flavour {ship}")
    report_lines.append(f"")
    report_lines.append(f"- τ_RUC = {tau}")
    report_lines.append(f"- Pooled AUROC = {flav['pooled_auroc']}")
    report_lines.append(f"- TruthfulQA held-out AUROC = "
                        f"{flav['per_benchmark'].get('truthfulqa', {}).get('auroc_held')}")
    report_lines.append(f"- N total = {len(df)}")
    report_lines.append("")
    report_lines.append("## Per-benchmark confusion matrix (at τ_RUC)")
    report_lines.append("")
    report_lines.append("| Benchmark | N | TP | FP | TN | FN | Prec | Recall | Acc |")
    report_lines.append("|---|---|---|---|---|---|---|---|---|")
    for bench in BENCHES:
        sub = df[df["benchmark"] == bench]
        if len(sub) == 0:
            continue
        cm = confusion_at_threshold(sub["p_rag_cal"], sub["training_label"], tau)
        report_lines.append(
            f"| {bench} | {len(sub)} | {cm['tp']} | {cm['fp']} | {cm['tn']} | {cm['fn']} | "
            f"{cm['precision']} | {cm['recall']} | {cm['accuracy']} |"
        )

    # Wrong predictions per benchmark (5 of each direction)
    for bench in BENCHES:
        sub = df[df["benchmark"] == bench]
        if len(sub) == 0:
            continue
        fp, fn = find_wrong_predictions(sub, "p_rag_cal", "training_label", tau, k=3)
        report_lines.append("")
        report_lines.append(f"## {bench} — FALSE POSITIVES (RUC said RAG, RAG was worse)")
        for _, r in fp.iterrows():
            report_lines.append(f"")
            report_lines.append(f"- **p_rag={r['p_rag_cal']:.3f}** | direct_em={r['direct_em']} rag_em={r['rag_em']} | direct_chm={r['direct_chm']:.3f} rag_chm={r['rag_chm']:.3f}")
            report_lines.append(f"  - Q: {r['question'][:200]}")
            report_lines.append(f"  - direct → {(r['direct_prediction'] or '')[:160]!r}")
            report_lines.append(f"  - rag    → {(r['rag_prediction'] or '')[:160]!r}")
        report_lines.append("")
        report_lines.append(f"## {bench} — FALSE NEGATIVES (RUC said DIRECT, RAG was better)")
        for _, r in fn.iterrows():
            report_lines.append(f"")
            report_lines.append(f"- **p_rag={r['p_rag_cal']:.3f}** | direct_em={r['direct_em']} rag_em={r['rag_em']} | direct_chm={r['direct_chm']:.3f} rag_chm={r['rag_chm']:.3f}")
            report_lines.append(f"  - Q: {r['question'][:200]}")
            report_lines.append(f"  - direct → {(r['direct_prediction'] or '')[:160]!r}")
            report_lines.append(f"  - rag    → {(r['rag_prediction'] or '')[:160]!r}")

    # Save report
    report_path = args.out_dir / "results" / "inspect_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nInspection report → {report_path}")
    print(f"SHAP plot         → {shap_path}")
    print(f"\nTo eyeball: less {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
