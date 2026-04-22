"""
scripts/build_calibration_pairs.py
===================================
Derive a labelled (document, claim, label) calibration set from CAEM's
per-sample eval JSONs, to drive the MiniCheck-vs-RoBERTa diagnostic
(``calibration_minicheck_vs_roberta.py``).

Label convention (benchmark-aware)
----------------------------------
    label = 1  ->  the claim is SUPPORTED by the document
    label = 0  ->  the claim is NOT SUPPORTED (refuted or underivable)

The claim / label extraction is benchmark-aware because the eval
harness records predictions in benchmark-specific formats:

  * **FEVER** (3-way entailment task): the "claim" is the FEVER
    statement itself, extracted from ``question`` after the
    instruction prefix ("Claim: ..."). Label comes from
    ``gold_label``: "supports" -> 1, "refutes"/"not enough info" -> 0.
    This is the correct convention because FEVER's ground truth
    *is* the support / no-support label; using EM here would conflate
    "model predicted correctly" with "claim is supported", which are
    different quantities.

  * **TriviaQA, Natural Questions, TruthfulQA** (open-ended QA):
    the "claim" is the model's full prediction string (e.g. "the
    answer is Hamlet"). Label comes from EM: em=1 means the model's
    answer matches gold and the top-1 retrieved passage should
    support it; em=0 means the answer is wrong and thus not supported.

  * **StrategyQA, ARC** (multiple-choice / yes-no): skipped for the
    5.5 diagnostic. The "claim" would need to be reconstructed from
    (question + predicted option) in a benchmark-specific way; the
    thesis appendix treats the 5.5 audit as sufficient on FEVER +
    open-ended QA coverage and documents the multi-choice exclusion
    as a scope note in §A.3.

Noise budget: open-ended labelling has ~5-15% label noise (a correct
answer that isn't paraphrased in the top-1 passage; a wrong answer
that happens to be mentioned) -- acceptable for a 500-pair smoke
diagnostic. FEVER labelling has near-zero noise because the label is
FEVER's own ground-truth annotation.

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


# Benchmarks we include in the 5.5 diagnostic. StrategyQA / ARC are
# multi-choice formats whose "claim" reconstruction is benchmark-
# specific and is deferred from the Phase 1a 5.5 audit scope.
SUPPORTED_BENCHMARKS = {"fever", "triviaqa", "natural_questions", "asqa", "truthfulqa"}


def _extract_claim(s: Dict[str, Any]) -> str:
    """Benchmark-aware extraction of the claim string to score.

    FEVER: parse the ``Claim: ...`` suffix from ``question``.
    Open-ended QA: return ``prediction`` stripped.
    Unknown benchmark: return ``prediction`` as a safe default.
    """
    bench = str(s.get("benchmark", "")).lower()
    question = str(s.get("question", ""))
    prediction = str(s.get("prediction", "")).strip()

    if bench == "fever":
        idx = question.find("Claim:")
        if idx >= 0:
            return question[idx + len("Claim:"):].strip()
        # Fallback: if the instruction prefix was stripped upstream,
        # use the whole question as the claim. This path should not
        # fire on current FEVER eval JSONs.
        return question.strip()

    return prediction


def _extract_label(s: Dict[str, Any]) -> Optional[int]:
    """Benchmark-aware extraction of the 0/1 support label.

    FEVER: ``gold_label == "supports"`` -> 1, else -> 0.
    Open-ended QA: ``em >= 0.5`` -> 1, else -> 0.
    Returns None when the sample has insufficient info to label.
    """
    bench = str(s.get("benchmark", "")).lower()

    if bench == "fever":
        gold_label = str(s.get("gold_label", "")).strip().lower()
        if not gold_label:
            return None
        return 1 if gold_label == "supports" else 0

    em = s.get("em")
    if em is None:
        return None
    return 1 if em >= 0.5 else 0


def _extract_retrieval_query(s: Dict[str, Any]) -> str:
    """Query string passed to the retriever to fetch the top-1 passage.

    For FEVER we use the extracted claim (cleaner retrieval target
    than the question prefix + claim concatenation). For open-ended
    QA we use the original question text (which is already a clean
    retrieval target).
    """
    bench = str(s.get("benchmark", "")).lower()
    if bench == "fever":
        return _extract_claim(s) or s.get("question", "")
    return s.get("question", "")


def _split_by_em(
    samples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split samples into label=1 / label=0 pools using benchmark-aware
    labelling. Skips samples from unsupported benchmarks and samples
    with empty claim or undefined label.
    """
    pos, neg = [], []
    for s in samples:
        bench = str(s.get("benchmark", "")).lower()
        if bench not in SUPPORTED_BENCHMARKS:
            continue

        claim = _extract_claim(s)
        if not claim:
            # No claim string available (blank prediction on open-ended
            # QA, missing Claim: on FEVER). Cannot score this pair.
            continue

        label = _extract_label(s)
        if label is None:
            continue

        if label == 1:
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
        claim = _extract_claim(s)
        if not claim:
            continue
        query = _extract_retrieval_query(s)
        passage = retrieve(query)
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
            "gold_label": s.get("gold_label", ""),
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
