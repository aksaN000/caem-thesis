#!/usr/bin/env python3
"""scripts/train_ruc_v2.py
=========================
Train the v2 Retrieval Utility Classifier (LR + StandardScaler) with nested
5-fold stratified CV, threshold sweep, and coefficient-sign inspection.

Pipeline
--------
1. Load ``caem/ruc/v2/training_set.parquet`` (built by
   ``scripts/build_ruc_training_set.py``). Defensive: drop EM-tie rows under
   Path B (case ∈ {A, B} only). Expected ~3 200 rows.
2. Identify feature columns: every numeric column except metadata fields.
3. Nested 5-fold stratified CV (stratified by ``case``):
   - Outer 5 folds → per-fold AUROC + per-bench AUROC partitioned from OOF
     predictions.
   - Inner CV per outer fold → sweep ``C ∈ {0.005, 0.01, 0.05, 0.1, 0.5,
     1.0, 5.0}`` and select best by inner-fold AUROC.
4. Fit final LR on all rows with mode of outer-fold best Cs.
5. Threshold τ sweep on full-data OOF predictions: τ ∈ {0.20, 0.25, …, 0.75}
   → maximise net deployment utility = ``A_caught(τ) − B_caught(τ)``.
6. Coefficient-sign inspection (architectural-intent receipt):
   - ``coef[p_ik] < 0`` (high model self-confidence → keep at T2 / C-default)
   - ``coef[top1_passage_sim] > 0`` (good retrieval → send to T3 / A-pattern)
   - ``coef[rerank_top1] > 0`` (well-matched passage → send to T3)
7. Write artefacts:
   - ``caem/ruc/v2/models/scaler.joblib``
   - ``caem/ruc/v2/models/lr_final.joblib``
   - ``caem/ruc/v2/feature_spec.json``
   - ``caem/ruc/v2/cv_results.json``

Usage
-----
    python -m scripts.train_ruc_v2 \\
        --in_parquet caem/ruc/v2/training_set.parquet \\
        --out_dir caem/ruc/v2
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("train_ruc_v2")


# --------------------------------------------------------------------------- #
# Feature schema                                                               #
# --------------------------------------------------------------------------- #

# Columns to exclude from the feature matrix (metadata, label, weight, etc.)
_NON_FEATURE_COLS = {
    "id", "benchmark", "pairing", "question", "case",
    "direct_baseline", "rag_baseline",
    "direct_em", "rag_em", "direct_chm", "rag_chm",
    "direct_prediction", "rag_prediction", "top_passage",
    "empirical_label", "empirical_weight",
    "em_delta", "chm_delta", "utility",
    # Optional embedding columns (legacy)
    "bge_embedding", "tfidf_embedding",
}


def _select_feature_columns(df) -> List[str]:
    """Pick all numeric columns that are not metadata/label/weight."""
    cols: List[str] = []
    for c in df.columns:
        if c in _NON_FEATURE_COLS:
            continue
        # Skip list-valued embedding columns
        if df[c].dtype == object:
            sample = df[c].dropna().head(1)
            if len(sample) and isinstance(sample.iloc[0], (list, np.ndarray)):
                continue
            continue  # other object dtypes (strings) skipped
        cols.append(c)
    return cols


# --------------------------------------------------------------------------- #
# Nested CV                                                                    #
# --------------------------------------------------------------------------- #

C_GRID: Tuple[float, ...] = (0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
TAU_GRID = np.arange(0.20, 0.76, 0.025)


def _fit_inner_best_c(
    X_train: np.ndarray, y_train: np.ndarray,
    inner_splits: int = 5, seed: int = 42,
) -> Tuple[float, Dict[float, float]]:
    """Sweep C on a stratified inner CV; return best C + per-C mean AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline

    auroc_by_c: Dict[float, float] = {}
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    for C in C_GRID:
        aurocs: List[float] = []
        for tr_idx, va_idx in inner.split(X_train, y_train):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    C=C, penalty="l2", solver="liblinear",
                    class_weight="balanced", max_iter=1000, random_state=seed,
                )),
            ])
            pipe.fit(X_train[tr_idx], y_train[tr_idx])
            proba = pipe.predict_proba(X_train[va_idx])[:, 1]
            try:
                aurocs.append(float(roc_auc_score(y_train[va_idx], proba)))
            except ValueError:
                continue
        auroc_by_c[C] = float(np.mean(aurocs)) if aurocs else float("nan")

    valid = {c: v for c, v in auroc_by_c.items() if not np.isnan(v)}
    best_c = max(valid, key=valid.get) if valid else 0.05
    return float(best_c), auroc_by_c


def _nested_cv(
    X: np.ndarray, y: np.ndarray, benches: np.ndarray,
    outer_splits: int = 5, seed: int = 42,
) -> Dict[str, Any]:
    """Outer 5-fold stratified CV. Returns per-fold AUROC, per-bench AUROC
    partitioned from pooled OOF predictions, and per-outer-fold best C.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline

    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    oof_proba = np.full(len(y), np.nan, dtype=np.float64)
    per_fold_aurocs: List[float] = []
    per_fold_best_c: List[float] = []
    per_fold_inner_grid: List[Dict[float, float]] = []

    for fold_i, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
        best_c, inner_grid = _fit_inner_best_c(X[tr_idx], y[tr_idx], seed=seed + fold_i)
        per_fold_best_c.append(best_c)
        per_fold_inner_grid.append(inner_grid)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=best_c, penalty="l2", solver="liblinear",
                class_weight="balanced", max_iter=1000, random_state=seed,
            )),
        ])
        pipe.fit(X[tr_idx], y[tr_idx])
        proba = pipe.predict_proba(X[te_idx])[:, 1]
        oof_proba[te_idx] = proba
        try:
            fold_auroc = float(roc_auc_score(y[te_idx], proba))
        except ValueError:
            fold_auroc = float("nan")
        per_fold_aurocs.append(fold_auroc)
        logger.info("  outer fold %d: best_C=%.4f  AUROC=%.4f",
                    fold_i + 1, best_c, fold_auroc)

    valid = ~np.isnan(oof_proba)
    pooled_auroc = float(roc_auc_score(y[valid], oof_proba[valid]))

    # Per-bench AUROC partitioned from OOF predictions
    per_bench: Dict[str, Dict[str, Any]] = {}
    for bench in sorted(set(benches.tolist())):
        mask = (benches == bench) & valid
        n = int(mask.sum())
        if n < 5:
            per_bench[bench] = {"n": n, "auroc": None, "note": "insufficient samples"}
            continue
        try:
            bench_auroc = float(roc_auc_score(y[mask], oof_proba[mask]))
        except ValueError:
            bench_auroc = None
        per_bench[bench] = {"n": n, "auroc": bench_auroc}

    return {
        "outer_splits": outer_splits,
        "pooled_oof_auroc": pooled_auroc,
        "per_fold_aurocs": per_fold_aurocs,
        "per_fold_best_c": per_fold_best_c,
        "per_fold_inner_grid": per_fold_inner_grid,
        "per_bench_auroc": per_bench,
        "oof_proba": oof_proba.tolist(),
    }


# --------------------------------------------------------------------------- #
# Threshold sweep                                                              #
# --------------------------------------------------------------------------- #

def _threshold_sweep(y: np.ndarray, proba: np.ndarray) -> Dict[str, Any]:
    """Sweep τ ∈ TAU_GRID. For each τ compute net utility = A_caught − B_caught
    where:
      A_caught = #(label=1) with proba >= τ  (correctly routed to T3)
      B_caught = #(label=0) with proba <  τ  (correctly routed to T2)
    Return the τ that maximises A_caught + B_caught (total correctly routed).
    """
    valid = ~np.isnan(proba)
    y = y[valid]
    proba = proba[valid]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    records: List[Dict[str, Any]] = []
    for tau in TAU_GRID:
        pred = (proba >= tau).astype(int)
        a_caught = int(((y == 1) & (pred == 1)).sum())
        b_caught = int(((y == 0) & (pred == 0)).sum())
        a_missed = n_pos - a_caught
        b_missed = n_neg - b_caught
        records.append({
            "tau": float(tau),
            "a_caught": a_caught, "a_missed": a_missed,
            "b_caught": b_caught, "b_missed": b_missed,
            "net_utility": a_caught + b_caught,  # total correct
            "utility_a_minus_b_missed": a_caught - b_missed,  # alternative metric
        })
    best = max(records, key=lambda r: r["net_utility"])
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "records": records,
        "best": best,
    }


# --------------------------------------------------------------------------- #
# Final fit + coefficient-sign inspection                                      #
# --------------------------------------------------------------------------- #

EXPECTED_SIGNS: Dict[str, int] = {
    # Architectural-intent receipt (replaces the C/D held-out test).
    "p_ik":             -1,  # high self-confidence → low p_rag → T2 (C-default)
    "top1_passage_sim": +1,  # strong retrieval → high p_rag → T3 (A-pattern)
    "rerank_top1":      +1,  # well-matched passage → high p_rag → T3
}


def _fit_final(X: np.ndarray, y: np.ndarray, C: float, seed: int = 42):
    """Fit final scaler + LR on all rows. Returns (scaler, lr, signed_coef_dict)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(
        C=C, penalty="l2", solver="liblinear",
        class_weight="balanced", max_iter=1000, random_state=seed,
    )
    lr.fit(Xs, y)
    return scaler, lr


def _inspect_coefficients(
    feature_names: List[str], coef: np.ndarray,
) -> Dict[str, Any]:
    coef_by_name = {name: float(c) for name, c in zip(feature_names, coef)}
    receipt = {}
    for name, expected_sign in EXPECTED_SIGNS.items():
        if name not in coef_by_name:
            receipt[name] = {
                "coef": None, "expected_sign": expected_sign,
                "actual_sign": None, "pass": False,
                "note": "feature missing from training set",
            }
            continue
        c = coef_by_name[name]
        actual_sign = 1 if c > 0 else (-1 if c < 0 else 0)
        receipt[name] = {
            "coef": c,
            "expected_sign": expected_sign,
            "actual_sign": actual_sign,
            "pass": actual_sign == expected_sign,
        }
    receipt_pass = all(r["pass"] for r in receipt.values())

    # Top-10 absolute coefficient ranking for interpretability
    ranked = sorted(coef_by_name.items(), key=lambda kv: -abs(kv[1]))[:15]

    return {
        "all_coef": coef_by_name,
        "architectural_intent_receipt": receipt,
        "architectural_intent_pass": receipt_pass,
        "top_by_abs_coef": [{"feature": n, "coef": c} for n, c in ranked],
    }


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in_parquet", type=Path,
                    default=Path("caem/ruc/v2/training_set.parquet"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("caem/ruc/v2"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outer_splits", type=int, default=5)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    import pandas as pd
    import joblib

    logger.info("loading %s", args.in_parquet)
    df = pd.read_parquet(args.in_parquet)
    logger.info("loaded %d rows × %d cols", len(df), len(df.columns))

    # Path B: drop EM ties (case ∈ {C, D}) defensively
    if "case" in df.columns:
        n_before = len(df)
        df = df[df["case"].isin(["A", "B"])].reset_index(drop=True)
        logger.info("Path B filter: %d → %d rows (dropped %d ties)",
                    n_before, len(df), n_before - len(df))

    if "empirical_label" not in df.columns:
        raise SystemExit("training set missing 'empirical_label' column")

    feature_cols = _select_feature_columns(df)
    logger.info("feature columns (%d):", len(feature_cols))
    for c in feature_cols:
        logger.info("  %s", c)

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["empirical_label"].to_numpy(dtype=np.int64)
    benches = df["benchmark"].to_numpy()

    logger.info("class balance: A (label=1) = %d, B (label=0) = %d",
                int((y == 1).sum()), int((y == 0).sum()))

    # 1. Nested 5-fold CV
    logger.info("running nested 5-fold CV ...")
    cv = _nested_cv(X, y, benches, outer_splits=args.outer_splits, seed=args.seed)
    logger.info("pooled OOF AUROC = %.4f", cv["pooled_oof_auroc"])
    logger.info("per-bench AUROC:")
    for bench, info in cv["per_bench_auroc"].items():
        auroc_str = f"{info['auroc']:.4f}" if info.get("auroc") is not None else "n/a"
        logger.info("  %-22s  n=%d  auroc=%s", bench, info["n"], auroc_str)

    # 2. Pick final C as mode of outer-fold best Cs
    best_c_counter = Counter(cv["per_fold_best_c"])
    final_c = float(best_c_counter.most_common(1)[0][0])
    logger.info("final C (mode of outer-fold best): %.4f  (distribution: %s)",
                final_c, dict(best_c_counter))

    # 3. Fit final scaler + LR on all rows
    scaler, lr = _fit_final(X, y, C=final_c, seed=args.seed)

    # 4. Threshold sweep on OOF predictions
    oof_proba = np.asarray(cv["oof_proba"], dtype=np.float64)
    sweep = _threshold_sweep(y, oof_proba)
    final_tau = sweep["best"]["tau"]
    logger.info(
        "threshold sweep: best τ=%.3f → net_utility=%d (A_caught=%d, B_caught=%d)",
        final_tau,
        sweep["best"]["net_utility"],
        sweep["best"]["a_caught"], sweep["best"]["b_caught"],
    )

    # 5. Coefficient-sign inspection
    inspection = _inspect_coefficients(feature_cols, lr.coef_[0])
    logger.info(
        "architectural-intent coefficient-sign receipt: %s",
        "PASS" if inspection["architectural_intent_pass"] else "FAIL",
    )
    for name, r in inspection["architectural_intent_receipt"].items():
        coef_str = f"{r['coef']:+.4f}" if r["coef"] is not None else "missing"
        logger.info(
            "  %-22s  coef=%s  expected=%+d  actual=%s  %s",
            name, coef_str, r["expected_sign"],
            f"{r['actual_sign']:+d}" if r.get("actual_sign") is not None else "n/a",
            "✓" if r["pass"] else "✗",
        )
    logger.info("top features by |coef|:")
    for entry in inspection["top_by_abs_coef"]:
        logger.info("  %-22s  coef=%+.4f", entry["feature"], entry["coef"])

    # 6. Save artefacts
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / "scaler.joblib"
    lr_path = models_dir / "lr_final.joblib"
    joblib.dump(scaler, scaler_path)
    joblib.dump(lr, lr_path)
    logger.info("wrote %s + %s", scaler_path, lr_path)

    feature_spec = {
        "model_type": "logistic_regression_scaled",
        "ship_flavour": "LR_v2",
        "model_paths": {
            "scaler": str(scaler_path),
            "lr": str(lr_path),
        },
        "features": feature_cols,
        "C": final_c,
        "tau_RUC_default": final_tau,
        "training": {
            "in_parquet": str(args.in_parquet),
            "n_rows": int(len(df)),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "outer_splits": args.outer_splits,
            "inner_splits": 5,
            "C_grid": list(C_GRID),
            "tau_grid": [float(t) for t in TAU_GRID],
            "seed": args.seed,
        },
        "validation": {
            "pooled_oof_auroc": cv["pooled_oof_auroc"],
            "per_fold_aurocs": cv["per_fold_aurocs"],
            "per_fold_best_c": cv["per_fold_best_c"],
            "per_bench_auroc": cv["per_bench_auroc"],
        },
        "threshold_sweep": {
            "best_tau": final_tau,
            "best_net_utility": sweep["best"]["net_utility"],
            "best_a_caught": sweep["best"]["a_caught"],
            "best_b_caught": sweep["best"]["b_caught"],
        },
        "architectural_intent_receipt": inspection["architectural_intent_receipt"],
        "architectural_intent_pass": inspection["architectural_intent_pass"],
        "top_by_abs_coef": inspection["top_by_abs_coef"],
    }
    spec_path = args.out_dir / "feature_spec.json"
    spec_path.write_text(json.dumps(feature_spec, indent=2))
    logger.info("wrote %s", spec_path)

    cv_results_path = args.out_dir / "cv_results.json"
    cv_results = {
        "nested_cv": {k: v for k, v in cv.items() if k != "oof_proba"},
        "threshold_sweep": sweep,
        "final_C": final_c,
        "final_tau": final_tau,
        "all_coef": inspection["all_coef"],
        "architectural_intent_receipt": inspection["architectural_intent_receipt"],
        "architectural_intent_pass": inspection["architectural_intent_pass"],
    }
    cv_results_path.write_text(json.dumps(cv_results, indent=2))
    logger.info("wrote %s", cv_results_path)

    # Final summary
    primary_tqa = cv["per_bench_auroc"].get("truthfulqa", {}).get("auroc")
    secondary_pooled = cv["pooled_oof_auroc"]
    print()
    print("=" * 60)
    print(f"v2 RUC training complete")
    print("=" * 60)
    print(f"  N rows trained on:            {len(df)}")
    print(f"  Features used:                {len(feature_cols)}")
    print(f"  Final C:                      {final_c}")
    print(f"  Final τ:                      {final_tau:.3f}")
    print(f"  Pooled OOF AUROC:             {secondary_pooled:.4f}")
    print(f"  TruthfulQA per-bench AUROC:   "
          f"{primary_tqa:.4f}" if primary_tqa is not None else
          f"  TruthfulQA per-bench AUROC:   insufficient samples")
    print(f"  Architectural-intent receipt: "
          f"{'✅ PASS' if inspection['architectural_intent_pass'] else '❌ FAIL'}")
    print(f"  Net deployment utility:       "
          f"{sweep['best']['net_utility']} "
          f"(A_caught={sweep['best']['a_caught']}, B_caught={sweep['best']['b_caught']})")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
