#!/usr/bin/env python3
"""scripts/ruc_eval_with_top5.py
======================================
After the FAISS sweep finishes (writes caem/ruc/training_set_v1_canonical_full_v2.parquet
with top-5 similarity stats), this script merges those features into the
clean-A+B training set, retrains LR + StandardScaler with 29 features
(24 + 5 top-5 stats), and compares:

  - 5-fold CV AUROC on clean A vs B rows
  - Deployment-policy utility at tau=0.60 vs the 24-feature baseline

Adopts the 29-feature version as the v1 deployment if EITHER pooled AUROC
lifts by >=0.02 OR pooled utility at tau=0.60 lifts by >=10.

Usage
-----
    /venv/main/bin/python -m scripts.ruc_eval_with_top5

Outputs (on adoption):
    caem/ruc/v1_shortcut/feature_spec.json     (updated, ship_flavour=LR_v1_clean_AB_top5)
    caem/ruc/v1_shortcut/models/scaler.joblib  (overwritten)
    caem/ruc/v1_shortcut/models/lr_final.joblib (overwritten)
    caem/ruc/v1_shortcut_24dim_backup/         (the prior 24-feature artefacts)
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BENCHES = ["fever", "triviaqa", "commonsense_qa", "strategyqa", "truthfulqa"]
FEATURES_24 = [
    "q_token_len", "negation_present", "temporal_cue", "numerical_cue", "myth_regex_hit",
    "interrog_who", "interrog_what", "interrog_when", "interrog_where", "interrog_why",
    "interrog_how", "interrog_yesno", "entity_count", "log_pageviews_max",
    "p_ik", "top1_passage_sim", "top1_passage_entity_overlap",
    "is_adversarial_phrasing", "is_opinion_seeking", "is_open_ended",
    "year_match", "number_match", "passage_has_year", "passage_has_number",
]
TOP5_FEATURES = [
    "top5_sim_mean", "top5_sim_std", "top5_sim_max", "top5_sim_min", "top5_sim_range",
]
TAU = 0.60

ADOPT_AUROC_DELTA = 0.02
ADOPT_UTILITY_DELTA = 10


def case_of(row):
    d, r = float(row["direct_em"]), float(row["rag_em"])
    if d == 0 and r == 1: return "A"
    if d == 1 and r == 0: return "B"
    if d == 1 and r == 1: return "C"
    return "D"


def evaluate(X, y, df_full_aligned, p_all_default=None):
    """5-fold CV AUROC on clean A+B + deployment utility @ tau on all rows."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    pooled_auroc = roc_auc_score(y, oof)
    per_bench_auroc = {}
    for b in BENCHES:
        m = (df_full_aligned[df_full_aligned["case"].isin(["A","B"])]["benchmark"]==b).values
        if y[m].sum() not in (0, m.sum()):
            per_bench_auroc[b] = roc_auc_score(y[m], oof[m])
        else:
            per_bench_auroc[b] = float("nan")
    return pooled_auroc, per_bench_auroc


def main() -> int:
    candidates = [
        Path("caem/ruc/training_set_v1_canonical_full_v2.parquet"),
    ]
    p_v2 = next((p for p in candidates if p.exists()), None)
    if p_v2 is None:
        print(f"ERROR: top-5 parquet not found. Waiting on the FAISS sweep.", file=sys.stderr)
        print(f"  expected one of: {[str(p) for p in candidates]}", file=sys.stderr)
        return 1
    df_v2 = pd.read_parquet(p_v2)
    missing_top5 = [c for c in TOP5_FEATURES if c not in df_v2.columns]
    if missing_top5:
        print(f"ERROR: top-5 columns missing from {p_v2}: {missing_top5}", file=sys.stderr)
        return 1

    # Source-of-truth case-labelled training set
    df = pd.read_parquet("caem/ruc/training_set_v1_canonical_shortcut.parquet")
    if "case" not in df.columns:
        df["case"] = df.apply(case_of, axis=1)
    df["id"] = df["id"].astype(str)
    df_v2["id"] = df_v2["id"].astype(str)

    # Merge top-5 stats into the case-labelled parquet
    df = df.merge(df_v2[["id","benchmark"] + TOP5_FEATURES],
                  on=["id","benchmark"], how="left")
    missing = df[TOP5_FEATURES].isna().any(axis=1).sum()
    print(f"merged top-5 features (missing rows: {missing})")
    for c in TOP5_FEATURES:
        df[c] = df[c].fillna(0.0)

    clean = df[df["case"].isin(["A","B"])].reset_index(drop=True)
    y = (clean["case"] == "A").astype(int).values

    X24 = clean[FEATURES_24].values.astype(np.float32)
    X29 = clean[FEATURES_24 + TOP5_FEATURES].values.astype(np.float32)

    auroc_24, perb_24 = evaluate(X24, y, clean)
    auroc_29, perb_29 = evaluate(X29, y, clean)

    print()
    print(f"{'bench':<16} {'24-dim AUROC':>14} {'29-dim AUROC':>14} {'delta':>8}")
    print("-"*60)
    print(f"{'pooled':<16} {auroc_24:>14.4f} {auroc_29:>14.4f} {auroc_29-auroc_24:>+8.4f}")
    for b in BENCHES:
        d = perb_29[b] - perb_24[b]
        print(f"{b:<16} {perb_24[b]:>14.4f} {perb_29[b]:>14.4f} {d:>+8.4f}")

    # Deployment-policy utility @ tau=0.60
    def util_at_tau(X_clean, X_all_aligned):
        sc = StandardScaler().fit(X_clean)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X_clean), y)
        p_all = clf.predict_proba(sc.transform(X_all_aligned))[:, 1]
        sent = p_all >= TAU
        A_caught = int(((df["case"]=="A") & sent).sum())
        B_caught = int(((df["case"]=="B") & sent).sum())
        return A_caught - B_caught, A_caught, B_caught, sc, clf, p_all

    X24_all = df[FEATURES_24].values.astype(np.float32)
    X29_all = df[FEATURES_24 + TOP5_FEATURES].values.astype(np.float32)
    u24, a24, b24, _, _, _ = util_at_tau(X24, X24_all)
    u29, a29, b29, sc29, clf29, p29 = util_at_tau(X29, X29_all)
    print()
    print(f"Utility @ tau={TAU} (A_caught - B_caught) on all 1500:")
    print(f"  24-dim: util={u24:+d}  (A={a24}, B={b24})")
    print(f"  29-dim: util={u29:+d}  (A={a29}, B={b29})")
    print(f"  delta:  {u29-u24:+d}")

    adopt = (auroc_29 - auroc_24 >= ADOPT_AUROC_DELTA) or (u29 - u24 >= ADOPT_UTILITY_DELTA)
    print()
    print(f"Adoption gate: AUROC delta >= {ADOPT_AUROC_DELTA} OR utility delta >= {ADOPT_UTILITY_DELTA}")
    print(f"  AUROC delta  = {auroc_29-auroc_24:+.4f}  ({'PASS' if auroc_29-auroc_24 >= ADOPT_AUROC_DELTA else 'no lift'})")
    print(f"  utility delta = {u29-u24:+d}  ({'PASS' if u29-u24 >= ADOPT_UTILITY_DELTA else 'no lift'})")
    print(f"  ==> {'ADOPT 29-dim' if adopt else 'KEEP 24-dim'}")

    if adopt:
        outdir = Path("caem/ruc/v1_shortcut")
        backup = Path("caem/ruc/v1_shortcut_24dim_backup")
        if outdir.exists() and not backup.exists():
            shutil.copytree(outdir, backup)
            print(f"backed up 24-dim artefacts -> {backup}")
        (outdir / "models").mkdir(parents=True, exist_ok=True)
        joblib.dump(sc29, outdir / "models" / "scaler.joblib")
        joblib.dump(clf29, outdir / "models" / "lr_final.joblib")
        spec = json.loads((outdir / "feature_spec.json").read_text())
        spec["ship_flavour"] = "LR_v1_clean_AB_top5"
        spec["features"] = FEATURES_24 + TOP5_FEATURES
        spec["n_features"] = len(FEATURES_24) + len(TOP5_FEATURES)
        spec["pooled_auroc_cv5_AvsB"] = float(auroc_29)
        spec["per_bench_auroc_cv5_AvsB"] = {b: float(perb_29[b]) for b in BENCHES}
        spec["deployment_utility_at_default_tau"]["pooled"] = {
            "A_caught": a29, "B_caught": b29, "utility": u29,
            "always_T3_utility": spec["deployment_utility_at_default_tau"]["pooled"]["always_T3_utility"],
            "improvement_over_always_T3": u29 - spec["deployment_utility_at_default_tau"]["pooled"]["always_T3_utility"],
        }
        spec["training_set"] = "caem/ruc/training_set_v1_canonical_shortcut.parquet (with top-5 merged from caem/ruc/training_set_v1_canonical_full_v2.parquet)"
        spec["trained_at"] = "2026-05-18 (top-5 features added)"
        (outdir / "feature_spec.json").write_text(json.dumps(spec, indent=2))
        print(f"WROTE updated artefacts at {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
