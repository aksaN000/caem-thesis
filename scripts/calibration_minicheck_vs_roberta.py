"""
scripts/calibration_minicheck_vs_roberta.py
============================================
Calibration diagnostic: does MiniCheck-Flan-T5-Large ground the CAEM
purity theorem better than legacy RoBERTa-large-MNLI on LM-generated
claims?

Motivation
----------
The Chapter 4 purity theorem

    purity = alpha * p / (alpha * p + (1 - alpha) * (1 - p))

is only defensible if P(entail | claim, passage) tracks truth. HaluEval
2025 / "Semantic Illusion" 2025 document 100%% FPR at 95%% recall for
DeBERTa-v3-large-MNLI on LM hallucinations -- RoBERTa-large-MNLI is
from the same training distribution and inherits the same blindspot.

MiniCheck (Tang et al. 2024 ACL) is trained on synthetically decomposed
LM claims and targets the LM-generated-claim-vs-document verification
task directly. This script quantifies the difference on a CAEM-sized
labeled set so the choice of backend is an empirical question, not an
appeal-to-authority.

Output
------
A single JSON file at ``ns.output_json`` carrying per-backend metrics:

    {
      "n_pairs": 500,
      "minicheck": {
          "auroc": ..., "ece": ..., "brier": ...,
          "selective_accuracy": {"0.5": ..., "0.8": ..., "0.95": ...},
          "avg_confident_correct": ..., "avg_confident_wrong": ...
      },
      "roberta_nli": { ... same keys ... },
      "delta_auroc": minicheck.auroc - roberta_nli.auroc,
      "calibration_pairs_path": <input path>,
      "thresholds": { "minicheck_entail": 0.7, "minicheck_contradict": 0.3 }
    }

A companion per-backend CSV (``<output_json>.details.csv``) keeps the
raw (pair_id, label, minicheck_prob, roberta_entail_prob,
roberta_contradict_prob) rows for downstream calibration-curve plots
into the Chapter 5 appendix.

Expected input JSONL
--------------------
Each line is one pair:

    {"id": "...", "document": "...", "claim": "...", "label": 0 or 1}

where ``label = 1`` means "claim is supported by document" (gold entail)
and ``label = 0`` means "not supported" (gold contradict or neutral).
The CAEM cold-start generator can emit this directly; see
``scripts/build_calibration_pairs.py`` if you need to synthesise one.

Usage
-----
python scripts/calibration_minicheck_vs_roberta.py \
    --pairs_jsonl data/calibration/minicheck_pairs_500.jsonl \
    --output_json outputs/calibration/minicheck_vs_roberta.json \
    --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from caem.config import CAEMConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("calibration")


# -------------------------------------------------------------------------- #
# Metric helpers                                                              #
# -------------------------------------------------------------------------- #

def _auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney U with average-rank tie handling.

    Kept dependency-free (no sklearn) so the diagnostic runs inside the
    lean Vast Docker image used for the autonomous runbook.
    """
    labels = labels.astype(np.float64)
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probs, kind="mergesort")
    ranks = np.empty_like(probs, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1, dtype=np.float64)
    # Average ranks across ties so the AUROC is well-defined on
    # discretised scores (e.g. P(entail) coarse-quantised by fp16).
    unique, inv = np.unique(probs, return_inverse=True)
    for u_i in range(len(unique)):
        idxs = np.where(inv == u_i)[0]
        if len(idxs) > 1:
            ranks[idxs] = ranks[idxs].mean()
    pos_ranks = ranks[labels.astype(bool)].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error on n equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)
    if total == 0:
        return float("nan")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs <= hi if hi == 1.0 else probs < hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(ece)


def _brier(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def _selective_accuracy(
    labels: np.ndarray, probs: np.ndarray, coverages: Sequence[float],
) -> Dict[str, float]:
    """Accuracy on the top-``coverage`` fraction of most-confident predictions.

    "Confidence" = max(P(entail), 1 - P(entail)). Prediction = (P >= 0.5).
    """
    pred = (probs >= 0.5).astype(int)
    correct = (pred == labels.astype(int)).astype(int)
    conf = np.maximum(probs, 1.0 - probs)
    order = np.argsort(-conf)
    out = {}
    n = len(labels)
    for c in coverages:
        k = max(1, int(round(c * n)))
        out[f"{c}"] = float(correct[order[:k]].mean())
    return out


@dataclass
class BackendMetrics:
    auroc: float
    ece: float
    brier: float
    selective_accuracy: Dict[str, float]
    n_pairs: int


# -------------------------------------------------------------------------- #
# Scoring backends                                                            #
# -------------------------------------------------------------------------- #

def score_with_minicheck(
    pairs: List[Dict[str, Any]], config: CAEMConfig, device: str,
) -> np.ndarray:
    """Return an (N,) array of P(supported) scores from MiniCheck."""
    from caem.verification.minicheck import load_minicheck_judge
    judge = load_minicheck_judge(
        model_name=config.minicheck_model,
        device=device,
        entail_threshold=config.minicheck_entail_threshold,
        contradict_threshold=config.minicheck_contradict_threshold,
    )
    pairs_tuples = [(p["document"], p["claim"]) for p in pairs]
    probs = judge.batch_entail_prob(pairs_tuples)
    return np.asarray(probs, dtype=np.float32)


def score_with_roberta(
    pairs: List[Dict[str, Any]], config: CAEMConfig, device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (P(entail), P(contradict)) arrays from roberta-large-mnli."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.nli_model
    ).to(device).eval()
    from caem.verification.verifier import _NLIEnsemble
    ens = _NLIEnsemble([(model, tokenizer)], device=device)
    pairs_tuples = [(p["document"], p["claim"]) for p in pairs]
    p_ent = np.asarray(ens.batch_entail_prob(pairs_tuples), dtype=np.float32)
    p_con = np.asarray(ens.batch_contradict_prob(pairs_tuples), dtype=np.float32)
    return p_ent, p_con


# -------------------------------------------------------------------------- #
# Main                                                                        #
# -------------------------------------------------------------------------- #

def _load_pairs(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    for i, p in enumerate(pairs):
        for k in ("document", "claim", "label"):
            if k not in p:
                raise ValueError(f"pair {i} missing required key {k!r}")
        if p["label"] not in (0, 1):
            raise ValueError(f"pair {i} has non-binary label {p['label']!r}")
    return pairs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs_jsonl", required=True, type=Path)
    p.add_argument("--output_json", required=True, type=Path)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--backends", nargs="+", default=["minicheck", "roberta_nli"],
        choices=["minicheck", "roberta_nli"],
    )
    p.add_argument("--coverages", nargs="+", type=float, default=[0.5, 0.8, 0.95])
    return p.parse_args()


def main() -> None:
    ns = _parse_args()
    config = CAEMConfig()

    if ns.device is None:
        import torch
        ns.device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading calibration pairs from %s", ns.pairs_jsonl)
    pairs = _load_pairs(ns.pairs_jsonl)
    labels = np.array([p["label"] for p in pairs], dtype=np.int32)
    n = len(pairs)
    logger.info("Loaded %d pairs (pos=%d, neg=%d)",
                n, int(labels.sum()), int(n - labels.sum()))

    if n < 100:
        logger.warning(
            "Only %d pairs provided. The 500-pair default is the bare "
            "minimum for 95%% CIs on AUROC deltas; smaller sets produce "
            "noisy appendix figures.", n,
        )

    results: Dict[str, Any] = {
        "n_pairs": n,
        "calibration_pairs_path": str(ns.pairs_jsonl),
        "thresholds": {
            "minicheck_entail": config.minicheck_entail_threshold,
            "minicheck_contradict": config.minicheck_contradict_threshold,
        },
    }

    minicheck_probs: Optional[np.ndarray] = None
    roberta_probs: Optional[np.ndarray] = None
    roberta_con: Optional[np.ndarray] = None

    if "minicheck" in ns.backends:
        logger.info("Scoring with MiniCheck-Flan-T5-Large ...")
        minicheck_probs = score_with_minicheck(pairs, config, ns.device)
        metrics = BackendMetrics(
            auroc=_auroc(labels, minicheck_probs),
            ece=_ece(labels, minicheck_probs),
            brier=_brier(labels.astype(float), minicheck_probs),
            selective_accuracy=_selective_accuracy(labels, minicheck_probs, ns.coverages),
            n_pairs=n,
        )
        results["minicheck"] = asdict(metrics)
        logger.info("MiniCheck: auroc=%.4f  ece=%.4f  brier=%.4f",
                    metrics.auroc, metrics.ece, metrics.brier)

    if "roberta_nli" in ns.backends:
        logger.info("Scoring with RoBERTa-large-MNLI ...")
        roberta_probs, roberta_con = score_with_roberta(pairs, config, ns.device)
        metrics = BackendMetrics(
            auroc=_auroc(labels, roberta_probs),
            ece=_ece(labels, roberta_probs),
            brier=_brier(labels.astype(float), roberta_probs),
            selective_accuracy=_selective_accuracy(labels, roberta_probs, ns.coverages),
            n_pairs=n,
        )
        results["roberta_nli"] = asdict(metrics)
        logger.info("RoBERTa: auroc=%.4f  ece=%.4f  brier=%.4f",
                    metrics.auroc, metrics.ece, metrics.brier)

    if minicheck_probs is not None and roberta_probs is not None:
        results["delta_auroc_minicheck_minus_roberta"] = (
            results["minicheck"]["auroc"] - results["roberta_nli"]["auroc"]
        )
        logger.info(
            "Delta AUROC (MiniCheck - RoBERTa): %+.4f",
            results["delta_auroc_minicheck_minus_roberta"],
        )

    ns.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(ns.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote metrics to %s", ns.output_json)

    # Companion per-pair CSV for downstream plotting.
    details_path = ns.output_json.with_suffix(".details.csv")
    with open(details_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["id", "label"]
        if minicheck_probs is not None:
            header.append("minicheck_p_supported")
        if roberta_probs is not None:
            header += ["roberta_p_entail", "roberta_p_contradict"]
        w.writerow(header)
        for i, pair in enumerate(pairs):
            row = [pair.get("id", i), int(labels[i])]
            if minicheck_probs is not None:
                row.append(float(minicheck_probs[i]))
            if roberta_probs is not None:
                row += [float(roberta_probs[i]), float(roberta_con[i])]
            w.writerow(row)
    logger.info("Wrote per-pair details to %s", details_path)


if __name__ == "__main__":
    main()
