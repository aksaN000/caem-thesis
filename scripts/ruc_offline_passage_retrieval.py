#!/usr/bin/env python3
"""scripts/ruc_offline_passage_retrieval.py
==============================================
Phase 1.1 of the RUC enrichment chain — compute the two retrieval-quality
features that were stubbed at 0.0 in the v1 training set:

- ``top1_passage_sim``           cosine similarity of the question's BGE-small
                                  embedding to the top-1 retrieved Wikipedia
                                  passage's BGE-small embedding.
- ``top1_passage_entity_overlap`` count of question NER entities that appear
                                  in the top-1 passage's text.

Uses the existing CAEM ``PassageStore`` (caem/retrieval/rag.py) to do the
FAISS lookup, but does NOT load the generator LLM (we only need retrieval).
Reuses the mpnet passage encoder from PassageStore, and a separately-loaded
BGE-small for the per-question semantic similarity.

Loads the index into RAM in parallel with anything else running on the GPU —
the FAISS index is CPU-side and uses about 64 GB.

Writes ``caem/ruc/passage_features.parquet`` with one row per (id, benchmark)
pair, schema:

    id, benchmark, top1_passage_sim, top1_passage_entity_overlap,
    top1_passage_text (first 500 chars, audit only)

The RUC builder merges this side-table on (id, benchmark) when present and
overwrites the stubbed-zero columns.

Usage
-----
    python -m scripts.ruc_offline_passage_retrieval \\
        --in_parquet caem/ruc/training_set.parquet \\
        --out_parquet caem/ruc/passage_features.parquet \\
        --top_k 1

After this runs once, re-run ``scripts.build_ruc_training_set`` or
manually merge into the main parquet via the small merger at the bottom of
this script.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_passage_store(config) -> "PassageStore":
    """Construct just the PassageStore (no LLM, no generator)."""
    from caem.retrieval.rag import PassageStore
    store = PassageStore(config)
    store.load()
    logger.info("PassageStore loaded; ntotal=%d", store.index.ntotal if hasattr(store, "index") else -1)
    return store


def _load_bge_small():
    from sentence_transformers import SentenceTransformer
    # 2026-05-17: previously forced CPU while the rescore pipeline held the GPU.
    # Reverted 2026-05-18 once rescore was killed and the GPU envelope returned
    # to ~0% util; auto-detect path now lands on CUDA (~5 ms/query vs ~50-100
    # ms/query on CPU). The deployed runtime RUC (caem/routing/ruc.py) keeps
    # CPU by design — that decision is unaffected.
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5")
    bge.eval()
    return bge


def _load_spacy():
    import spacy
    return spacy.load("en_core_web_sm", disable=["lemmatizer"])


def _passage_text(p) -> str:
    """Coerce the various passage-tuple shapes to a string."""
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        return p.get("text") or p.get("passage") or ""
    if isinstance(p, (tuple, list)) and p:
        first = p[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("text") or first.get("passage") or ""
    return ""


def _compute_for_question(
    question: str,
    bge,
    nlp,
    encoder,
    store,
    top_k: int = 1,
) -> Dict[str, float]:
    """Return the two retrieval-quality features for one question.

    Uses CAEM's mpnet QueryEncoder to embed, then PassageStore.search
    directly (no LLM). Matches the same retrieval path the production
    pipeline uses, minus the generation step.
    """
    try:
        emb = encoder.encode(question)
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        results = store.search(emb, k=top_k)
    except Exception as exc:
        logger.warning("retrieve failed for q=%r: %s", question[:80], exc)
        return {
            "top1_passage_sim": 0.0,
            "top1_passage_entity_overlap": 0.0,
            "top1_passage_text": "",
        }
    if not results:
        return {
            "top1_passage_sim": 0.0,
            "top1_passage_entity_overlap": 0.0,
            "top1_passage_text": "",
        }
    top1_text = _passage_text(results[0])
    if not top1_text:
        return {
            "top1_passage_sim": 0.0,
            "top1_passage_entity_overlap": 0.0,
            "top1_passage_text": "",
        }

    # Compute BGE-small similarity (matches the embedding used in the
    # RUC feature vector elsewhere).
    q_emb = bge.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    p_emb = bge.encode([top1_text], normalize_embeddings=True, show_progress_bar=False)[0]
    sim = float(np.dot(q_emb, p_emb))

    # Entity overlap
    doc = nlp(question or "")
    ents = [ent.text.lower() for ent in doc.ents]
    p_lower = top1_text.lower()
    overlap = sum(1 for e in ents if e in p_lower)

    return {
        "top1_passage_sim": sim,
        "top1_passage_entity_overlap": float(overlap),
        "top1_passage_text": top1_text[:500],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--out_parquet", type=Path,
                    default=Path("caem/ruc/passage_features.parquet"))
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--log_level", default="INFO")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rows (smoke test).")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    import pandas as pd
    df = pd.read_parquet(args.in_parquet)

    # Unique (id, benchmark, question) tuples — we don't need to re-query per
    # pairing because the question text is the same across pairings.
    unique = df.drop_duplicates(subset=["id", "benchmark"])[
        ["id", "benchmark", "question"]
    ].reset_index(drop=True)
    if args.limit is not None:
        unique = unique.head(args.limit)
    logger.info("Will retrieve top-%d passages for %d unique questions",
                args.top_k, len(unique))

    # Build a minimal CAEM config and load only what we need (no LLM).
    from caem.config import CAEMConfig
    from caem.retrieval.rag import PassageStore
    from caem.memory.encoder import QueryEncoder
    config = CAEMConfig()
    # 2026-05-18: previously forced CPU while the rescore pipeline held the
    # GPU envelope at ~31 GB / 32 GB. After the rescore was killed (cycle 3
    # branch + Ch6 close decision), the GPU returned to ~0% util and we
    # revert here. Auto-detect path takes CUDA and runs ~10x faster
    # (~8 ms/query vs ~80-120 ms/query on CPU).
    logger.info("Loading QueryEncoder (all-mpnet-base-v2, ~440 MB)...")
    encoder = QueryEncoder()
    logger.info("Loading PassageStore (64 GB FAISS index, mmap)...")
    store = PassageStore.load("data/passage_index")
    logger.info("PassageStore size=%d", store.size)
    bge = _load_bge_small()
    nlp = _load_spacy()

    rows: List[Dict] = []
    for i, r in unique.iterrows():
        feats = _compute_for_question(r["question"], bge, nlp, encoder, store,
                                      top_k=args.top_k)
        rows.append({
            "id": r["id"],
            "benchmark": r["benchmark"],
            **feats,
        })
        if (i + 1) % 100 == 0:
            logger.info("progress: %d / %d", i + 1, len(unique))

    out = pd.DataFrame(rows)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    logger.info("Wrote %d rows -> %s", len(out), args.out_parquet)

    # Print summary stats.
    print(f"\n{'='*60}")
    print(f"Passage-feature summary  ->  {args.out_parquet}")
    print(f"{'='*60}")
    print(f"  top1_passage_sim:           min={out['top1_passage_sim'].min():.3f}  "
          f"mean={out['top1_passage_sim'].mean():.3f}  max={out['top1_passage_sim'].max():.3f}")
    print(f"  top1_passage_entity_overlap: min={out['top1_passage_entity_overlap'].min():.0f}  "
          f"mean={out['top1_passage_entity_overlap'].mean():.2f}  max={out['top1_passage_entity_overlap'].max():.0f}")
    print(f"  rows with sim > 0.5: {(out['top1_passage_sim'] > 0.5).sum()} / {len(out)}")
    print(f"  rows with no passage found: {(out['top1_passage_text'] == '').sum()} / {len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
