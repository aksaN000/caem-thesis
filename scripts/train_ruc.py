#!/usr/bin/env python3
"""scripts/train_ruc.py
==========================
Retrieval Utility Classifier (RUC) — LOBO-CV training and evaluation.

Reads ``caem/ruc/training_set.parquet`` (must be Sonnet-labelled first via
``scripts/label_ruc_with_sonnet.py``), runs leave-one-benchmark-out
cross-validation across the 4 ablation flavours, picks the shipping flavour
on TruthfulQA held-out AUROC, and writes:

- ``caem/ruc/models/ruc_lightgbm_{A1..A4}.txt``       (LightGBM models per flavour, per fold)
- ``caem/ruc/calibration/isotonic_{flavour}_{bench}.joblib``
- ``caem/ruc/results/lobo_cv_metrics.json``           (AUROC, AUPRC per fold per flavour)
- ``caem/ruc/results/threshold_sweep.json``           (τ_RUC sweep, picks the CHM-minimising threshold)
- ``caem/ruc/results/shap_summary.png``               (interpretability for ship flavour)
- ``caem/ruc/feature_spec.json``                       (locked feature schema for inference)

Ablation flavours
-----------------
- A1: engineered features only (~14 dims)
- A2: engineered + TF-IDF
- A3: engineered + BGE-small (ship default)
- A4: engineered + TF-IDF + BGE-small

Acceptance gates (pre-registered)
---------------------------------
- PRIMARY:   TruthfulQA held-out AUROC ≥ 0.75 (stretch 0.85)
- SECONDARY: pooled AUROC ≥ 0.78 (stretch 0.85)
- TERTIARY:  FEVER + TriviaQA held-out AUROC each ≥ 0.60
- Red flag:  AUROC ≥ 0.90 (likely leakage; audit before shipping)

Usage
-----
    python -m scripts.train_ruc \\
        --in_parquet caem/ruc/training_set.parquet \\
        --out_dir caem/ruc
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature spec — what goes into the LightGBM input matrix
# ---------------------------------------------------------------------------

ENGINEERED_FEATURES: List[str] = [
    "q_token_len",
    "negation_present",
    "temporal_cue",
    "numerical_cue",
    "myth_regex_hit",
    "interrog_who",
    "interrog_what",
    "interrog_when",
    "interrog_where",
    "interrog_why",
    "interrog_how",
    "interrog_yesno",
    "entity_count",
    "log_pageviews_max",
    "top1_passage_sim",
    "top1_passage_entity_overlap",
    # 2026-05-17 (RUC v1.1 P(IK) test): Kadavath-style confidence probe on
    # frozen Qwen-2.5-3B hidden states. Probe AUROC on direct_em alone is
    # 0.66 on the canonical 1500-row pool. Filled to 0.5 sentinel when the
    # parquet was built before scripts/ruc_train_pik_probe.py ran, so
    # backward-compatible with v1 datasets.
    "p_ik",
    # 2026-05-17 (RUC v1.2 verifier-aware test): the 9 active composite
    # signals + u_stored computed by the cycle-3 verifier on the DIRECT
    # (zero_shot) prediction. These preview whether retrieval would have
    # useful content for the question: low direct_p_ground_* means
    # passages do not support the direct answer, so RAG could fix it;
    # high direct_u_stored means the verifier already trusts direct, so
    # RAG is unlikely to add value. The RUC at routing time runs AFTER
    # the Tier 2 generation + verifier pass, so these signals are
    # available pre-Tier-3 by construction in the live pipeline.
    "direct_u_token",
    "direct_u_dropout",
    "direct_u_internal",
    "direct_s_avg",
    "direct_p_entail",
    "direct_p_ground_max",
    "direct_p_ground_mean",
    "direct_p_ground_atomic",
    "direct_q_a_relevance",
    "direct_u_stored",
]
# qtype_* features deliberately excluded for v1 — the BGE embedding
# captures semantic question type more reliably, and the placeholder
# values leaked benchmark identity at LOBO-CV.

BENCHES = ["fever", "triviaqa", "commonsense_qa", "strategyqa", "truthfulqa"]
FLAVOURS = ["A1", "A2", "A3", "A4"]
SHIP_FLAVOUR_DEFAULT = "A3"  # ship unless A4 beats by >=0.03 TruthfulQA AUROC

LGBM_PARAMS = dict(
    objective="binary",
    metric="auc",
    max_depth=6,
    num_leaves=31,
    min_data_in_leaf=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    learning_rate=0.05,
    n_estimators=500,
    verbose=-1,
    random_state=42,
)

EARLY_STOPPING_ROUNDS = 30


# ---------------------------------------------------------------------------
# Feature matrix assembly per flavour
# ---------------------------------------------------------------------------

def _bge_matrix(df) -> np.ndarray:
    """Stack the per-row BGE embedding lists into a (n, 384) matrix."""
    arr = np.stack([np.asarray(x, dtype=np.float32) for x in df["bge_embedding"].values])
    return arr


def assemble_X(df, flavour: str, tfidf_vec=None) -> Tuple[np.ndarray, Optional[Any]]:
    """Return (X, fitted_tfidf_vectoriser_or_None) for the given flavour.

    For flavours with TF-IDF, callers pass ``tfidf_vec=None`` to fit one,
    or pass an already-fitted vectoriser for the validation/test fold.
    """
    n = len(df)
    blocks: List[np.ndarray] = []
    eng = df[ENGINEERED_FEATURES].values.astype(np.float32)
    blocks.append(eng)

    if flavour in {"A2", "A4"}:
        from sklearn.feature_extraction.text import TfidfVectorizer
        questions = df["question"].fillna("").tolist()
        if tfidf_vec is None:
            tfidf_vec = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2),
                min_df=5, max_features=5000,
            )
            tf = tfidf_vec.fit_transform(questions).toarray().astype(np.float32)
        else:
            tf = tfidf_vec.transform(questions).toarray().astype(np.float32)
        blocks.append(tf)

    if flavour in {"A3", "A4"}:
        blocks.append(_bge_matrix(df))

    X = np.hstack(blocks)
    return X, tfidf_vec


def feature_names(flavour: str, tfidf_vec=None) -> List[str]:
    names = list(ENGINEERED_FEATURES)
    if flavour in {"A2", "A4"} and tfidf_vec is not None:
        names += [f"tfidf__{w}" for w in tfidf_vec.get_feature_names_out()]
    if flavour in {"A3", "A4"}:
        names += [f"bge__{i}" for i in range(384)]
    return names


# ---------------------------------------------------------------------------
# LOBO-CV training
# ---------------------------------------------------------------------------

def train_lobo_fold(
    train_df, test_df,
    *,
    flavour: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train one LightGBM model on train_df, evaluate on test_df.

    Validation slice (for early stopping + isotonic) is carved from train_df
    via a fixed 80/20 stratified split.
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Drop rows with missing training_label (Sonnet-tied with no empirical
    # fallback — should not happen in practice but guard anyway).
    train_df = train_df.dropna(subset=["training_label"]).copy()
    test_df = test_df.dropna(subset=["training_label"]).copy()

    # Stratified train/val split inside the LOBO training set.
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=seed,
        stratify=train_df["training_label"].astype(int),
    )
    inner_train = train_df.iloc[tr_idx].copy()
    inner_val   = train_df.iloc[val_idx].copy()

    # Assemble feature matrices. TF-IDF is fit on inner_train only.
    X_tr, tfidf = assemble_X(inner_train, flavour, tfidf_vec=None)
    X_val, _    = assemble_X(inner_val,   flavour, tfidf_vec=tfidf)
    X_te, _     = assemble_X(test_df,     flavour, tfidf_vec=tfidf)

    y_tr  = inner_train["training_label"].astype(int).values
    y_val = inner_val["training_label"].astype(int).values
    y_te  = test_df["training_label"].astype(int).values

    w_tr  = inner_train["training_weight"].astype(float).values
    w_val = inner_val["training_weight"].astype(float).values

    # Class imbalance: scale_pos_weight = N_neg / N_pos on training set
    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int(len(y_tr) - n_pos))
    scale_pos = n_neg / n_pos

    params = dict(LGBM_PARAMS, scale_pos_weight=scale_pos, random_state=seed)
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr, sample_weight=w_tr,
        eval_set=[(X_val, y_val)], eval_sample_weight=[w_val],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    # Isotonic calibration on the validation slice.
    p_val_raw = model.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val)

    p_te_raw = model.predict_proba(X_te)[:, 1]
    p_te_cal = iso.predict(p_te_raw)

    try:
        auroc_raw = float(roc_auc_score(y_te, p_te_raw))
        auroc_cal = float(roc_auc_score(y_te, p_te_cal))
        auprc_cal = float(average_precision_score(y_te, p_te_cal))
    except ValueError:
        # Single-class test fold — undefined AUROC; skip the fold.
        auroc_raw = float("nan")
        auroc_cal = float("nan")
        auprc_cal = float("nan")

    return {
        "model": model,
        "tfidf": tfidf,
        "isotonic": iso,
        "p_test_raw": p_te_raw.tolist(),
        "p_test_cal": p_te_cal.tolist(),
        "y_test":     y_te.tolist(),
        "test_ids":   test_df["id"].astype(str).tolist(),
        "test_benchmarks": test_df["benchmark"].tolist(),
        "test_direct_chm": test_df["direct_chm"].astype(float).tolist(),
        "test_rag_chm":    test_df["rag_chm"].astype(float).tolist(),
        "n_train_pos": n_pos, "n_train_neg": n_neg,
        "auroc_raw_held":  auroc_raw,
        "auroc_cal_held":  auroc_cal,
        "auprc_cal_held":  auprc_cal,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }


# ---------------------------------------------------------------------------
# Threshold sweep — pick τ_RUC that minimises pooled CHM
# ---------------------------------------------------------------------------

def pooled_chm_at_threshold(
    p: np.ndarray, direct_chm: np.ndarray, rag_chm: np.ndarray, tau: float
) -> float:
    """Pooled CHM when the RUC routes p>=tau to RAG and the rest to DIRECT."""
    use_rag = p >= tau
    chm = np.where(use_rag, rag_chm, direct_chm)
    return float(chm.mean())


def threshold_sweep(
    p_all: np.ndarray,
    direct_chm: np.ndarray,
    rag_chm: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Sweep τ ∈ [0.05, 0.95] in 0.05 steps, return per-τ pooled CHM."""
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    rows = []
    for tau in grid:
        chm = pooled_chm_at_threshold(p_all, direct_chm, rag_chm, float(tau))
        share_rag = float((p_all >= tau).mean())
        rows.append({"tau": float(round(tau, 3)), "pooled_chm": round(chm, 4),
                     "share_rag": round(share_rag, 4)})
    best = min(rows, key=lambda r: r["pooled_chm"])
    return {"sweep": rows, "tau_optimal": best["tau"],
            "tau_optimal_chm": best["pooled_chm"]}


# ---------------------------------------------------------------------------
# Per-benchmark roll-up
# ---------------------------------------------------------------------------

def per_benchmark_metrics(folds: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for fold in folds:
        bench = fold["test_benchmarks"][0]  # LOBO: every row in fold is from one benchmark
        out[bench] = {
            "auroc_held":    round(fold["auroc_cal_held"], 4) if not math.isnan(fold["auroc_cal_held"]) else None,
            "auprc_held":    round(fold["auprc_cal_held"], 4) if not math.isnan(fold["auprc_cal_held"]) else None,
            "n_test":        len(fold["y_test"]),
            "n_train_pos":   fold["n_train_pos"],
            "n_train_neg":   fold["n_train_neg"],
            "pos_rate_test": round(float(np.mean(fold["y_test"])), 4) if fold["y_test"] else None,
        }
    return out


def pooled_metric(folds: List[Dict[str, Any]]) -> float:
    p_all, y_all = [], []
    for f in folds:
        p_all.extend(f["p_test_cal"])
        y_all.extend(f["y_test"])
    if not p_all:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_all, p_all))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Train all 4 flavours, return master dict
# ---------------------------------------------------------------------------

def train_all_flavours(df, out_dir: Path, seed: int = 42) -> Dict[str, Any]:
    import joblib

    results: Dict[str, Any] = {"flavours": {}, "ship_flavour": None}

    for flavour in FLAVOURS:
        logger.info(f"=== Flavour {flavour} ===")
        folds_data: List[Dict[str, Any]] = []
        for held_out in BENCHES:
            train_df = df[df["benchmark"] != held_out].copy()
            test_df  = df[df["benchmark"] == held_out].copy()
            if len(test_df) == 0:
                logger.warning(f"  bench {held_out} has 0 test rows; skipping")
                continue
            fold = train_lobo_fold(train_df, test_df, flavour=flavour, seed=seed)
            folds_data.append(fold)
            logger.info(
                f"  held_out={held_out:<14} n_train={len(train_df)} "
                f"n_test={len(test_df)} auroc_cal={fold['auroc_cal_held']:.4f} "
                f"auprc={fold['auprc_cal_held']:.4f}"
            )

            # Persist model + tfidf + isotonic per fold
            model_path = out_dir / "models" / f"ruc_lightgbm_{flavour}_{held_out}.txt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            fold["model"].booster_.save_model(str(model_path))
            iso_path = out_dir / "calibration" / f"isotonic_{flavour}_{held_out}.joblib"
            iso_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(fold["isotonic"], iso_path)
            if fold["tfidf"] is not None:
                tfidf_path = out_dir / "models" / f"tfidf_{flavour}_{held_out}.joblib"
                joblib.dump(fold["tfidf"], tfidf_path)

        # Per-benchmark + pooled
        per_bench = per_benchmark_metrics(folds_data)
        pooled_auroc = pooled_metric(folds_data)
        # Threshold sweep on pooled predictions
        p_all = np.concatenate([np.asarray(f["p_test_cal"]) for f in folds_data])
        d_chm = np.concatenate([np.asarray(f["test_direct_chm"]) for f in folds_data])
        r_chm = np.concatenate([np.asarray(f["test_rag_chm"]) for f in folds_data])
        sweep = threshold_sweep(p_all, d_chm, r_chm)

        results["flavours"][flavour] = {
            "per_benchmark": per_bench,
            "pooled_auroc":  round(pooled_auroc, 4),
            "threshold_sweep": sweep,
        }
        logger.info(
            f"  POOLED auroc={pooled_auroc:.4f}  tau*={sweep['tau_optimal']:.2f}  "
            f"pooled_chm@tau*={sweep['tau_optimal_chm']:.4f}"
        )

    # Pick ship flavour: A3 unless A4 beats it on TruthfulQA AUROC by ≥0.03
    a3_tqa = results["flavours"]["A3"]["per_benchmark"].get("truthfulqa", {}).get("auroc_held") or 0
    a4_tqa = results["flavours"]["A4"]["per_benchmark"].get("truthfulqa", {}).get("auroc_held") or 0
    if a4_tqa - a3_tqa >= 0.03:
        results["ship_flavour"] = "A4"
        logger.info(f"Ship flavour: A4 (TQA AUROC {a4_tqa:.4f} beats A3 {a3_tqa:.4f} by >=0.03)")
    else:
        results["ship_flavour"] = SHIP_FLAVOUR_DEFAULT
        logger.info(f"Ship flavour: A3 (default; A4 TQA AUROC {a4_tqa:.4f} - A3 {a3_tqa:.4f} = {a4_tqa-a3_tqa:.4f} < 0.03)")

    # Fit a final all-data deployment model for the ship flavour.
    # LOBO-CV models are diagnostic only; the deployed RUC sees questions from
    # all 5 benchmarks and should be trained on all of them.
    logger.info("Fitting final all-data deployment model for ship flavour %s...",
                results["ship_flavour"])
    final_artefacts = _train_final_deployment_model(df, results["ship_flavour"],
                                                    out_dir, seed=seed)
    results["deployment_model"] = final_artefacts

    return results


def _train_final_deployment_model(
    df, flavour: str, out_dir: Path, seed: int = 42,
) -> Dict[str, Any]:
    """Train one LightGBM on ALL labelled rows (no LOBO holdout) for deployment.

    Saves under caem/ruc/models/ruc_lightgbm_{flavour}_FINAL.txt.
    A held-out 20% slice fits the isotonic calibrator for the threshold.
    Returns artefact paths for the feature_spec.json.
    """
    import lightgbm as lgb
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.isotonic import IsotonicRegression

    train_df = df.dropna(subset=["training_label"]).copy()
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=seed,
        stratify=train_df["training_label"].astype(int),
    )
    inner_train = train_df.iloc[tr_idx].copy()
    inner_val   = train_df.iloc[val_idx].copy()

    X_tr, tfidf = assemble_X(inner_train, flavour, tfidf_vec=None)
    X_val, _    = assemble_X(inner_val,   flavour, tfidf_vec=tfidf)
    y_tr  = inner_train["training_label"].astype(int).values
    y_val = inner_val["training_label"].astype(int).values
    w_tr  = inner_train["training_weight"].astype(float).values
    w_val = inner_val["training_weight"].astype(float).values

    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int(len(y_tr) - n_pos))
    scale_pos = n_neg / n_pos
    params = dict(LGBM_PARAMS, scale_pos_weight=scale_pos, random_state=seed)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr, sample_weight=w_tr,
        eval_set=[(X_val, y_val)], eval_sample_weight=[w_val],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    p_val_raw = model.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, y_val)

    final_model_path = out_dir / "models" / f"ruc_lightgbm_{flavour}_FINAL.txt"
    final_iso_path   = out_dir / "calibration" / f"isotonic_{flavour}_FINAL.joblib"
    final_tfidf_path = out_dir / "models" / f"tfidf_{flavour}_FINAL.joblib"
    model.booster_.save_model(str(final_model_path))
    joblib.dump(iso, final_iso_path)
    if tfidf is not None:
        joblib.dump(tfidf, final_tfidf_path)

    logger.info(
        "Final deployment model saved: %s (best_iter=%d, n_train=%d, n_val=%d)",
        final_model_path, model.best_iteration_ or model.n_estimators,
        len(inner_train), len(inner_val),
    )
    return {
        "flavour": flavour,
        "model_path": str(final_model_path),
        "isotonic_path": str(final_iso_path),
        "tfidf_path": str(final_tfidf_path) if tfidf is not None else None,
        "n_train": int(len(inner_train)),
        "n_val": int(len(inner_val)),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }


# ---------------------------------------------------------------------------
# Acceptance gates
# ---------------------------------------------------------------------------

def check_gates(results: Dict[str, Any]) -> Dict[str, Any]:
    ship = results["ship_flavour"]
    flav = results["flavours"][ship]
    pb = flav["per_benchmark"]
    pooled = flav["pooled_auroc"]
    out = {
        "ship_flavour": ship,
        "primary_truthfulqa":  pb.get("truthfulqa", {}).get("auroc_held"),
        "secondary_pooled":    pooled,
        "tertiary_fever":      pb.get("fever", {}).get("auroc_held"),
        "tertiary_triviaqa":   pb.get("triviaqa", {}).get("auroc_held"),
        "gates": {},
    }
    out["gates"]["primary_truthfulqa_ge_0.75"]  = bool(out["primary_truthfulqa"] is not None and out["primary_truthfulqa"] >= 0.75)
    out["gates"]["secondary_pooled_ge_0.78"]    = bool(pooled is not None and pooled >= 0.78)
    out["gates"]["tertiary_fever_ge_0.60"]      = bool(out["tertiary_fever"] is not None and out["tertiary_fever"] >= 0.60)
    out["gates"]["tertiary_triviaqa_ge_0.60"]   = bool(out["tertiary_triviaqa"] is not None and out["tertiary_triviaqa"] >= 0.60)
    out["gates"]["red_flag_max_lt_0.90"]        = all(
        (v is None or v < 0.90)
        for v in [out["primary_truthfulqa"], pooled, out["tertiary_fever"], out["tertiary_triviaqa"]]
    )
    out["all_gates_pass"] = all(out["gates"].values())
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--out_dir", type=Path, default=Path("caem/ruc"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "models").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "calibration").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results").mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.read_parquet(args.in_parquet)
    if "training_label" not in df.columns:
        logger.error("training_label not in parquet; run label_ruc_with_sonnet.py first")
        return 1
    df = df.dropna(subset=["training_label"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} labelled rows.")
    logger.info(f"Label-source distribution:\n{df['label_source'].value_counts().to_string()}")

    results = train_all_flavours(df, args.out_dir, seed=args.seed)
    gates = check_gates(results)
    results["gates"] = gates

    # Save feature spec for inference
    spec = {
        "engineered_features": ENGINEERED_FEATURES,
        "bge_model": "BAAI/bge-small-en-v1.5",
        "bge_dim": 384,
        "ship_flavour": results["ship_flavour"],
        "model_path_template": "caem/ruc/models/ruc_lightgbm_{flavour}_{held_out}.txt",
        "isotonic_path_template": "caem/ruc/calibration/isotonic_{flavour}_{held_out}.joblib",
        "tau_RUC": results["flavours"][results["ship_flavour"]]["threshold_sweep"]["tau_optimal"],
    }
    (args.out_dir / "feature_spec.json").write_text(json.dumps(spec, indent=2))

    # Save metrics
    metrics_path = args.out_dir / "results" / "lobo_cv_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Metrics -> {metrics_path}")

    # Print final summary
    print("\n" + "=" * 78)
    print(f"RUC training complete — ship flavour: {results['ship_flavour']}")
    print("=" * 78)
    for flav in FLAVOURS:
        r = results["flavours"][flav]
        pooled = r["pooled_auroc"]
        tqa = r["per_benchmark"].get("truthfulqa", {}).get("auroc_held")
        fev = r["per_benchmark"].get("fever", {}).get("auroc_held")
        tqar = r["per_benchmark"].get("triviaqa", {}).get("auroc_held")
        csq = r["per_benchmark"].get("commonsense_qa", {}).get("auroc_held")
        sqa = r["per_benchmark"].get("strategyqa", {}).get("auroc_held")
        tau = r["threshold_sweep"]["tau_optimal"]
        marker = " ← SHIP" if flav == results["ship_flavour"] else ""
        print(f"  {flav}: pooled={pooled:.4f}  TQA={tqa}  FEV={fev}  TRV={tqar}  "
              f"CSQ={csq}  STR={sqa}  τ*={tau}{marker}")
    print()
    print("Acceptance gates:")
    for k, v in gates["gates"].items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")
    print(f"  ALL GATES PASS: {gates['all_gates_pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
