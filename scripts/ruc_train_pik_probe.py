#!/usr/bin/env python3
"""scripts/ruc_train_pik_probe.py
=====================================
Phase 1.4 of the RUC enrichment chain — train a P(IK) probe (Kadavath et al.
2022, arXiv:2207.05221) that predicts ``direct_em`` (will the model get this
question right?) from Qwen-2.5-3B-Instruct hidden states.

Method
------
1. Load Qwen-2.5-3B-Instruct on GPU.
2. For each unique question in the RUC training set, tokenize and do ONE
   short forward pass (no generation). Extract the hidden state at the
   final layer of the last input token (3072-dim vector for Qwen-3B).
3. Stack into (N, 3072) matrix.
4. Train an L2-regularised logistic regression on those vectors to predict
   ``direct_em`` from `caem/ruc/training_set.parquet`.
5. Apply CV-out-of-fold predictions back into the parquet as the
   ``p_ik`` feature column.
6. Persist the probe to ``caem/ruc/pik_probe.joblib`` so the runtime
   RUC can compute P(IK) at inference time.

Why this matters
----------------
The Kadavath paper (and follow-ups SeaKR, Probing-RAG, Self-RAG) shows
that hidden-state probes predict model correctness with AUROC 0.70-0.85.
For the RUC, P(IK) directly answers the underlying question: "if the model
already knows this, retrieval is unnecessary; if it doesn't, retrieval
might help." Adding it as a feature should reduce the cross-task LOBO-CV
gap.

Caveats
-------
- The probe is trained on the PRE-SIL base Qwen-2.5-3B-Instruct (the same
  checkpoint baselines were generated on). At deployment the CAEM Qwen
  may be LoRA-fine-tuned. LoRA only updates ~1 percent of weights, so
  hidden-state distribution shift is small — but documented.
- For LOBO-CV honesty, the probe predictions are computed via 5-fold
  cross-validation across questions (not benchmarks). The probe sees
  no test-bench leakage because hidden states aren't conditional on the
  benchmark label.

Cost: ~30-60 min GPU on 5090. No API.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

def _extract_hidden_states(
    questions: List[str],
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 512,
) -> np.ndarray:
    """Return (N, hidden_dim) float32 matrix of last-token last-layer states."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    logger.info("Loading %s on %s ...", model_name, device)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    hidden_dim = model.config.hidden_size
    logger.info("Model loaded. hidden_dim=%d, n_params=%dM",
                hidden_dim, sum(p.numel() for p in model.parameters()) // 1_000_000)

    out = np.zeros((len(questions), hidden_dim), dtype=np.float32)
    n_batches = (len(questions) + batch_size - 1) // batch_size
    with torch.no_grad():
        for bi in range(n_batches):
            start = bi * batch_size
            chunk = questions[start:start + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn,
                            output_hidden_states=False)
            # last_hidden_state shape: (B, T, H)
            last = outputs.last_hidden_state
            # Pick the hidden state at the last NON-PADDING token for each row
            seq_lens = attn.sum(dim=1) - 1   # 0-indexed
            for i, sl in enumerate(seq_lens.tolist()):
                out[start + i] = last[i, sl].float().cpu().numpy()
            if (bi + 1) % 20 == 0:
                logger.info("  encoded %d / %d", start + len(chunk), len(questions))

    # Free GPU mem
    del model
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Probe training (5-fold CV across questions)
# ---------------------------------------------------------------------------

def _train_probe(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Returns (final_calibrated_probe, oof_predictions). Uses 5-fold CV
    so the predictions back-fed into the RUC parquet are out-of-fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    oof = np.zeros(len(X), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_aurocs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                    class_weight="balanced", random_state=seed)
        model.fit(X[tr_idx], y[tr_idx])
        p = model.predict_proba(X[te_idx])[:, 1]
        oof[te_idx] = p
        try:
            auroc = roc_auc_score(y[te_idx], p)
        except ValueError:
            auroc = float("nan")
        fold_aurocs.append(auroc)
        logger.info("  fold %d  AUROC=%.4f", fold + 1, auroc)
    logger.info("  mean OOF AUROC=%.4f  (probe self-evaluation)",
                float(np.mean(fold_aurocs)))

    # Final probe trained on all data, calibrated
    base = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                              class_weight="balanced", random_state=seed)
    final = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    final.fit(X, y)
    return final, oof, float(np.mean(fold_aurocs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--out_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--cache_path", type=Path,
                    default=Path("caem/ruc/qwen_hidden_states.npy"),
                    help="If exists, reuse cached hidden states instead of "
                         "re-running the forward passes.")
    ap.add_argument("--id_index_path", type=Path,
                    default=Path("caem/ruc/qwen_hidden_states_ids.json"),
                    help="Stores the (id, benchmark) order of cached hidden states.")
    ap.add_argument("--probe_path", type=Path,
                    default=Path("caem/ruc/pik_probe.joblib"))
    ap.add_argument("--metrics_path", type=Path,
                    default=Path("caem/ruc/results/pik_metrics.json"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    import pandas as pd
    df = pd.read_parquet(args.in_parquet)

    # Unique (id, benchmark, question) — same question across pairings
    # shares hidden state, so encode once.
    unique = df.drop_duplicates(subset=["id", "benchmark"])[
        ["id", "benchmark", "question", "direct_em"]
    ].reset_index(drop=True)
    logger.info("Will compute hidden states for %d unique questions",
                len(unique))

    questions = unique["question"].astype(str).tolist()

    # Hidden-state extraction (cached if rerunning)
    if args.cache_path.exists() and args.id_index_path.exists():
        logger.info("Loading cached hidden states from %s", args.cache_path)
        X = np.load(args.cache_path)
        with open(args.id_index_path) as f:
            cached_ids = json.load(f)
        # sanity-check cache aligns with current dataframe
        current_ids = [f"{r['id']}|{r['benchmark']}" for _, r in unique.iterrows()]
        if cached_ids != current_ids:
            logger.warning("Cached hidden-state order mismatch with current "
                           "parquet; re-extracting.")
            X = _extract_hidden_states(
                questions, model_name=args.model, device=args.device,
                batch_size=args.batch_size, max_length=args.max_length,
            )
            args.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.cache_path, X)
            with open(args.id_index_path, "w") as f:
                json.dump([f"{r['id']}|{r['benchmark']}"
                           for _, r in unique.iterrows()], f)
    else:
        X = _extract_hidden_states(
            questions, model_name=args.model, device=args.device,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        args.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.cache_path, X)
        with open(args.id_index_path, "w") as f:
            json.dump([f"{r['id']}|{r['benchmark']}"
                       for _, r in unique.iterrows()], f)
        logger.info("Cached hidden states -> %s (shape=%s)",
                    args.cache_path, X.shape)

    # Labels: direct_em ∈ {0.0, 1.0}
    y = (unique["direct_em"].astype(float).values >= 1.0).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    logger.info("Probe labels: %d positive (direct correct), %d negative",
                pos, neg)

    final_probe, oof, mean_auroc = _train_probe(X, y, seed=args.seed)

    # Persist probe
    import joblib
    args.probe_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_probe, args.probe_path)
    logger.info("Persisted probe -> %s", args.probe_path)

    # Save metrics
    metrics = {
        "model": args.model,
        "n_questions": int(len(unique)),
        "n_pos": pos,
        "n_neg": neg,
        "mean_oof_auroc": mean_auroc,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics -> %s", args.metrics_path)

    # Merge OOF predictions back into the main parquet
    unique["p_ik"] = oof
    df["id"] = df["id"].astype(str)
    unique["id"] = unique["id"].astype(str)
    if "p_ik" in df.columns:
        df = df.drop(columns=["p_ik"])
    merged = df.merge(unique[["id", "benchmark", "p_ik"]],
                       on=["id", "benchmark"], how="left")
    merged["p_ik"] = merged["p_ik"].fillna(0.5)
    merged.to_parquet(args.out_parquet, index=False)

    print(f"\n{'='*66}")
    print(f"P(IK) probe summary  ->  {args.probe_path}")
    print(f"{'='*66}")
    print(f"  n unique questions:   {len(unique)}")
    print(f"  pos / neg labels:     {pos} / {neg}  ({pos/(pos+neg)*100:.1f}% pos)")
    print(f"  mean OOF AUROC:       {mean_auroc:.4f}  (probe self-evaluation)")
    print(f"  parquet rows updated: {len(merged)}")
    print(f"  p_ik distribution:    min={merged['p_ik'].min():.3f}  "
          f"mean={merged['p_ik'].mean():.3f}  max={merged['p_ik'].max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
