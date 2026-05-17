#!/usr/bin/env python3
"""
scripts/retrieval_utility_classifier.py
========================================
Retrieval Utility Classifier (RUC) — post-hoc diagnostic for Ch6.

Motivation
----------
Per-benchmark analysis of B1 vs CAEM C3 reveals that Tier-3 RAG improves
performance on FEVER and TriviaQA but degrades it on TruthfulQA, CommonsenseQA,
and StrategyQA (see Ch5 §sec:comp-baselines). Empirically, 100% of wrong T3
TruthfulQA predictions reference retrieved context explicitly, versus 0% of T2
predictions. This indicates a retrieval-grounding compulsion failure mode: the
model adopts whatever passage is retrieved even when it is misleading.

This script asks: can we predict — using only non-grounding self-consistency
signals available after T3 generation but before any passage comparison — whether
a given T3 generation succeeded?

Method
------
Features: 6 non-grounding signals from the unified verifier
  u_token      — token-level uncertainty (MC-dropout)
  u_dropout    — output dropout uncertainty
  u_internal   — calibrated pre-routing confidence
  s_avg        — semantic self-consistency across n_consistency samples
  h_norm       — normalised output-token entropy
  q_a_relevance — question-answer relevance (NLI, no passage)

Label:  em = 1 (T3 succeeded) / em = 0 (T3 failed)
        TruthfulQA uses em_llm_judged.

Validation: Leave-One-Benchmark-Out CV (LOBO-CV).
  Train on 4 benchmarks, test on the held-out one.  Repeated for all 5.
  This tests cross-task generalisation, not just i.i.d. accuracy.

Model: Logistic Regression with L2 penalty (liblinear solver).
  Chosen for interpretability: coefficients directly quantify each signal's
  contribution, which is reportable in the thesis.

Outputs
-------
  outputs/full_run/ruc_lobo_results.json   — per-benchmark AUC + accuracy
  outputs/full_run/ruc_coefficients.json   — LR coefficients per feature
  thesis_report/figures/auto/tab_ruc_lobo.tex  — LaTeX table for Ch6

Usage
-----
    python -m scripts.retrieval_utility_classifier \\
        --eval_dir outputs/full_run/eval \\
        --cycle 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURES = ["u_token", "u_dropout", "u_internal", "s_avg", "h_norm", "q_a_relevance"]

BENCHES = ["fever", "triviaqa", "commonsense_qa", "strategyqa", "truthfulqa"]

BENCH_LABELS = {
    "fever": "FEVER",
    "triviaqa": "TriviaQA",
    "commonsense_qa": "CommonsenseQA",
    "strategyqa": "StrategyQA",
    "truthfulqa": "TruthfulQA",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _sample_em(s: Dict) -> float:
    if s.get("benchmark") == "truthfulqa" and s.get("em_llm_judged") is not None:
        return float(s["em_llm_judged"])
    return float(s.get("em") or 0.0)


def load_t3_samples(eval_dir: Path, cycle: int) -> List[Dict]:
    """Load all Tier-3 samples from a given cycle across all benchmarks."""
    samples = []
    for bench in BENCHES:
        path = eval_dir / f"{bench}_cycle{cycle}.json"
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue
        doc = json.load(open(path))
        for s in doc.get("samples", []):
            if s.get("tier") != 3:
                continue
            # Attach benchmark for LOBO split and label computation
            s = dict(s)
            s["benchmark"] = bench
            samples.append(s)
    return samples


def build_feature_matrix(
    samples: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, bench_labels) for sklearn."""
    X_rows, y_rows, bench_rows = [], [], []
    skipped = 0
    for s in samples:
        row = []
        missing = False
        for feat in FEATURES:
            val = s.get(feat)
            if val is None:
                missing = True
                break
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                missing = True
                break
        if missing:
            skipped += 1
            continue
        X_rows.append(row)
        y_rows.append(1.0 if _sample_em(s) >= 1.0 else 0.0)
        bench_rows.append(s["benchmark"])
    if skipped:
        logger.warning("Skipped %d samples with missing features", skipped)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows), bench_rows


# ---------------------------------------------------------------------------
# LOBO-CV
# ---------------------------------------------------------------------------

def lobo_cv(
    X: np.ndarray,
    y: np.ndarray,
    bench_labels: List[str],
    C: float = 1.0,
) -> Dict:
    """Leave-One-Benchmark-Out cross-validation with logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score

    benches = list(dict.fromkeys(bench_labels))  # preserve order, unique
    bench_arr = np.array(bench_labels)

    results = {}
    all_probs, all_y = [], []

    for held_out in benches:
        train_mask = bench_arr != held_out
        test_mask = bench_arr == held_out

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        if len(np.unique(y_te)) < 2:
            logger.warning("Held-out %s has single class — skipping AUC", held_out)
            results[held_out] = {"n_test": int(test_mask.sum()), "auc": float("nan"),
                                 "acc": float("nan"), "pos_rate": float(y_te.mean())}
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(C=C, solver="liblinear", max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_tr)

        probs = clf.predict_proba(X_te_s)[:, 1]
        preds = clf.predict(X_te_s)

        auc = roc_auc_score(y_te, probs)
        acc = accuracy_score(y_te, preds)

        results[held_out] = {
            "n_test": int(test_mask.sum()),
            "auc": round(float(auc), 4),
            "acc": round(float(acc), 4),
            "pos_rate": round(float(y_te.mean()), 4),
        }
        all_probs.extend(probs.tolist())
        all_y.extend(y_te.tolist())

    # Pooled AUC across all held-out predictions
    pooled_auc = float(roc_auc_score(all_y, all_probs)) if len(np.unique(all_y)) > 1 else float("nan")
    results["POOLED"] = {"auc": round(pooled_auc, 4)}
    return results


# ---------------------------------------------------------------------------
# Full-data coefficient fit (for reporting)
# ---------------------------------------------------------------------------

def fit_full_coefficients(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Dict:
    """Fit on all data; return standardised coefficients per feature."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(C=C, solver="liblinear", max_iter=1000, random_state=42)
    clf.fit(X_s, y)

    coef = {feat: round(float(c), 4) for feat, c in zip(FEATURES, clf.coef_[0])}
    intercept = round(float(clf.intercept_[0]), 4)
    return {"coefficients": coef, "intercept": intercept,
            "note": "Standardised LR coefficients (L2, C=1.0). Positive = predicts retrieval success."}


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def write_tex(lobo: Dict, coef: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/retrieval_utility_classifier.py",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textbf{Held-out benchmark} & \textbf{N} & \textbf{Pos. rate} & \textbf{AUROC} \\",
        r"\midrule",
    ]
    for bench in BENCHES:
        r = lobo.get(bench, {})
        n = r.get("n_test", "--")
        pos = f"{r['pos_rate']:.3f}" if "pos_rate" in r else "--"
        auc = f"{r['auc']:.3f}" if "auc" in r and not np.isnan(r["auc"]) else "--"
        label = BENCH_LABELS.get(bench, bench)
        lines.append(fr"{label} & {n} & {pos} & {auc} \\")
    pooled_auc = lobo.get("POOLED", {}).get("auc", float("nan"))
    pool_str = f"{pooled_auc:.3f}" if not np.isnan(pooled_auc) else "--"
    lines += [
        r"\midrule",
        fr"\textbf{{Pooled}} & -- & -- & \textbf{{{pool_str}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"\textbf{Signal} & \textbf{Coefficient} \\",
        r"\midrule",
    ]
    for feat, val in sorted(coef["coefficients"].items(), key=lambda x: -abs(x[1])):
        sign = "+" if val >= 0 else ""
        lines.append(fr"\texttt{{{feat}}} & ${sign}{val:.4f}$ \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Retrieval Utility Classifier (RUC) evaluated under leave-one-benchmark-out"
        r" cross-validation. The classifier is a logistic regression trained on six"
        r" non-grounding self-consistency signals. Positive coefficients predict retrieval"
        r" success; negative coefficients predict retrieval harm. AUROC above 0.5 on a"
        r" held-out benchmark indicates that signals generalise across task types.}",
        r"\label{tab:ruc-lobo}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("RUC LaTeX -> %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", type=Path, default=Path("outputs/full_run/eval"))
    p.add_argument("--cycle", type=int, default=3)
    p.add_argument("--C", type=float, default=1.0,
                   help="LR regularisation strength (higher = less regularised)")
    p.add_argument("--json_out", type=Path,
                   default=Path("outputs/full_run/ruc_lobo_results.json"))
    p.add_argument("--tex_out", type=Path,
                   default=Path("thesis_report/figures/auto/tab_ruc_lobo.tex"))
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(level=ns.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Load
    samples = load_t3_samples(ns.eval_dir, ns.cycle)
    logger.info("Loaded %d T3 samples from cycle %d", len(samples), ns.cycle)

    X, y, bench_labels = build_feature_matrix(samples)
    logger.info("Feature matrix: %s, pos_rate=%.3f", X.shape, y.mean())

    # Per-benchmark stats
    print(f"\n{'Benchmark':<18} {'N_T3':>6} {'em_rate':>8}")
    print("-" * 34)
    ba = np.array(bench_labels)
    for bench in BENCHES:
        mask = ba == bench
        if mask.sum() == 0:
            continue
        print(f"{BENCH_LABELS[bench]:<18} {mask.sum():>6} {y[mask].mean():>8.3f}")

    # Check for missing features
    missing_any = sum(1 for s in samples
                      if any(s.get(f) is None for f in FEATURES))
    if missing_any > 0:
        logger.warning("%d samples missing at least one feature — excluded", missing_any)
    if X.shape[0] < 50:
        logger.error("Too few complete samples (%d). Aborting.", X.shape[0])
        return 1

    # LOBO-CV
    lobo = lobo_cv(X, y, bench_labels, C=ns.C)

    print(f"\n{'Held-out':<18} {'N':>5} {'pos_rate':>9} {'AUC':>7} {'Acc':>7}")
    print("-" * 50)
    for bench in BENCHES:
        r = lobo.get(bench, {})
        auc_s = f"{r['auc']:.3f}" if "auc" in r and not np.isnan(r.get("auc", float("nan"))) else "  --"
        acc_s = f"{r['acc']:.3f}" if "acc" in r and not np.isnan(r.get("acc", float("nan"))) else "  --"
        print(f"{BENCH_LABELS.get(bench,bench):<18} {r.get('n_test','--'):>5} "
              f"{r.get('pos_rate',float('nan')):>9.3f} {auc_s:>7} {acc_s:>7}")
    print(f"\nPooled AUC: {lobo.get('POOLED',{}).get('auc', float('nan')):.3f}")

    # Full-data coefficients
    coef = fit_full_coefficients(X, y, C=ns.C)
    print("\nStandardised LR coefficients (full data):")
    for feat, val in sorted(coef["coefficients"].items(), key=lambda x: -abs(x[1])):
        bar = "+" * int(abs(val) * 10) if val >= 0 else "-" * int(abs(val) * 10)
        print(f"  {feat:<16} {val:>+7.4f}  {bar}")
    print(f"  (positive = predicts retrieval success)")

    # Save
    ns.json_out.parent.mkdir(parents=True, exist_ok=True)
    out = {"lobo_cv": lobo, "coefficients": coef,
           "cycle": ns.cycle, "n_samples": int(X.shape[0]),
           "features": FEATURES}
    json.dump(out, open(ns.json_out, "w"), indent=2)
    logger.info("Results -> %s", ns.json_out)

    write_tex(lobo, coef, ns.tex_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
