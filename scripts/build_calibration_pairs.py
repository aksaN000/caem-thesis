"""
scripts/build_calibration_pairs.py
===================================
Derive a labelled (document, claim, label) calibration set from CAEM's
per-sample eval JSONs, to drive the MiniCheck-vs-RoBERTa diagnostic
(``calibration_minicheck_vs_roberta.py``).

Label convention
----------------
    label = 1  ->  the claim is SUPPORTED by the document
    label = 0  ->  the claim is NOT SUPPORTED (refuted or underivable)

Label source: the evaluation harness records exact-match (``em``) per
sample against the gold answers. We use EM as a first-order proxy for
claim-support:

    em = 1  ->  treat (top-1 passage, model_answer) as a supported pair
    em = 0  ->  treat (top-1 passage, model_answer) as an unsupported pair

Both classes are drawn from the actual LM-generation distribution
(the distribution the CAEM verifier is asked to score at run time),
which is the correct evaluation regime for MiniCheck's claim-support
training objective. Noise in the label (a correct answer that isn't
stated in the top-1 passage; a wrong answer that happens to be
paraphrased in the passage) is expected at ~5-15%% and is acceptable
for a 500-pair smoke-grade diagnostic -- the thesis appendix Section
should document this caveat.

Passages
--------
Eval JSONs intentionally do NOT store ``top_passages`` (the harness
filters list-typed verifier fields to keep the per-sample record
rectangular). We therefore re-retrieve the top-1 passage at build
time using the same SBERT + FAISS path the CAEM verifier uses. The
FAISS index must be the same as the one used at eval time or the
labels will not correspond.

Usage
-----
python scripts/build_calibration_pairs.py \\
    --eval_jsons outputs/cycle_0/eval/*_cycle0.json \\
    --passage_index data/passage_index \\
    --n_pairs 500 \\
    --balance 0.5 \\
    --output_jsonl data/calibration/minicheck_pairs_500.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_pairs")


def _load_eval_samples(paths: List[Path]) -> List[Dict[str, Any]]:
    """Return a flat list of per-sample records across all eval JSONs."""
    out: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            blob = json.load(f)
        samples = blob.get("samples", [])
        bench = blob.get("meta", {}).get("benchmark", p.stem)
        for s in samples:
            s = dict(s)
            s.setdefault("benchmark", bench)
            out.append(s)
    return out


def _split_by_em(
    samples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pos, neg = [], []
    for s in samples:
        em = s.get("em")
        if em is None:
            continue
        prediction = s.get("prediction")
        if prediction is None or not str(prediction).strip():
            # Blank predictions come from ABSTAIN / DISCARD branches that
            # emitted display_answer="I don't know." or empty; they carry
            # no claim to score, so they can't go into either class.
            continue
        if em >= 0.5:
            pos.append(s)
        else:
            neg.append(s)
    return pos, neg


def _balanced_sample(
    pos: List[Dict[str, Any]], neg: List[Dict[str, Any]],
    n_pairs: int, balance: float, rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pick ``n_pairs`` samples with ``balance`` positive fraction.

    Caps the request at whatever the smaller pool allows and warns if
    we fall short of the target.
    """
    target_pos = int(round(n_pairs * balance))
    target_neg = n_pairs - target_pos

    actual_pos = min(target_pos, len(pos))
    actual_neg = min(target_neg, len(neg))
    if actual_pos < target_pos or actual_neg < target_neg:
        logger.warning(
            "Falling short of target n_pairs=%d (balance=%.2f): available "
            "pos=%d (requested %d), neg=%d (requested %d). Rebalancing.",
            n_pairs, balance, len(pos), target_pos, len(neg), target_neg,
        )
        # Greedy rebalance: take min(request, pool) on each side. The
        # returned total may be < n_pairs; downstream callers should check.

    return rng.sample(pos, actual_pos), rng.sample(neg, actual_neg)


def _load_retriever(passage_index_path: Path, device: str, sbert_name: str):
    """Return a callable ``(query_text) -> top1_passage_text``."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from caem.memory.encoder import QueryEncoder
    from caem.retrieval.rag import PassageStore

    logger.info("Loading passage index %s ...", passage_index_path)
    store = PassageStore.load(str(passage_index_path))
    logger.info("Passage store loaded: %d passages", len(store.passages))

    logger.info("Loading SBERT encoder %s ...", sbert_name)
    encoder = QueryEncoder(model_name=sbert_name, device=device)

    def retrieve(query: str) -> Optional[str]:
        emb = encoder.encode(query).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        results = store.search(emb, k=1)
        if not results:
            return None
        return results[0][0]

    return retrieve


def _build_pairs(
    samples: List[Dict[str, Any]],
    label: int,
    retrieve: Any,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        question = s.get("question", "")
        claim = str(s.get("prediction", "")).strip()
        passage = retrieve(question)
        if not passage:
            continue
        out.append({
            "id": f"{s.get('benchmark', 'unk')}:{s.get('id', i)}",
            "document": str(passage),
            "claim": claim,
            "label": int(label),
            "benchmark": s.get("benchmark", "unk"),
            "em_score": float(s.get("em", 0.0)),
            "u_stored_ref": float(s.get("u_stored", 0.0)),
            "gold_answers_head": (s.get("gold_answers") or ["?"])[0][:120],
        })
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval_jsons", nargs="+", required=True, type=str,
        help="Paths or globs to per-benchmark eval JSON (e.g. "
             "outputs/cycle_0/eval/*_cycle0.json).",
    )
    p.add_argument("--passage_index", required=True, type=Path)
    p.add_argument(
        "--sbert_model", type=str, default="sentence-transformers/all-mpnet-base-v2",
    )
    p.add_argument("--n_pairs", type=int, default=500)
    p.add_argument("--balance", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_jsonl", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    ns = _parse_args()
    if not 0.0 <= ns.balance <= 1.0:
        raise ValueError(f"--balance must be in [0, 1], got {ns.balance}")

    # Expand globs
    expanded: List[Path] = []
    for pat in ns.eval_jsons:
        matches = glob.glob(pat)
        if not matches:
            logger.warning("No files match %r", pat)
        expanded.extend(Path(m) for m in matches)
    if not expanded:
        logger.error("No eval JSONs found. Exiting.")
        sys.exit(1)
    logger.info("Loading %d eval JSON(s)", len(expanded))

    samples = _load_eval_samples(expanded)
    logger.info("Loaded %d samples total", len(samples))

    pos_pool, neg_pool = _split_by_em(samples)
    logger.info("EM split: pos=%d neg=%d (skipped blanks implicitly)",
                len(pos_pool), len(neg_pool))
    if not pos_pool or not neg_pool:
        logger.error(
            "One class is empty (pos=%d, neg=%d); cannot build a "
            "balanced calibration set.", len(pos_pool), len(neg_pool),
        )
        sys.exit(1)

    rng = random.Random(ns.seed)
    pos_picked, neg_picked = _balanced_sample(
        pos_pool, neg_pool, ns.n_pairs, ns.balance, rng,
    )
    logger.info("Picked: pos=%d neg=%d", len(pos_picked), len(neg_picked))

    if ns.device is None:
        import torch
        ns.device = "cuda" if torch.cuda.is_available() else "cpu"
    retrieve = _load_retriever(ns.passage_index, ns.device, ns.sbert_model)

    pos_pairs = _build_pairs(pos_picked, label=1, retrieve=retrieve)
    neg_pairs = _build_pairs(neg_picked, label=0, retrieve=retrieve)
    pairs = pos_pairs + neg_pairs
    rng.shuffle(pairs)

    ns.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(ns.output_jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False))
            f.write("\n")
    logger.info(
        "Wrote %d pairs (pos=%d, neg=%d) to %s",
        len(pairs), len(pos_pairs), len(neg_pairs), ns.output_jsonl,
    )


if __name__ == "__main__":
    main()
