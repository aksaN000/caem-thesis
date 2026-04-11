"""
scripts/build_passage_index.py
==============================
One-time offline script: build a Wikipedia passage corpus and encode it into
a PassageStore for use by TierThreeRAG.

What it does
------------
1. Streams the configured English Wikipedia dump from HuggingFace datasets
   in streaming mode -- avoids downloading the full ~20 GB dump at once.
2. Chunks each article into ~100-word passages (DPR-style split,
   Karpukhin et al. 2020). Each chunk keeps its section title as a prefix.
3. Encodes all passages with the same SBERT model used by EpisodicMemoryStore
    (sentence-transformers/all-mpnet-base-v2 -> 768-dim vectors).
4. L2-normalises every embedding (required for cosine similarity via inner
    product in FAISS retrieval).
5. Saves the result as a PassageStore: two files in --output_dir:
            passages.faiss   -- FAISS index (IVF-PQ by default)
       passages.pkl     -- matching list of passage strings

Usage
-----
Full run (500 K passages, ~2–4 h on CPU; ~20 min on GPU):
    python -m scripts.build_passage_index --output_dir data/passage_index

Quick smoke test (1 K passages):
    python -m scripts.build_passage_index --output_dir data/passage_index_smoke \
           --max_passages 1000

Larger corpus for production-quality RAG (requires ~16 GB RAM):
    python -m scripts.build_passage_index --output_dir data/passage_index_full \
           --max_passages 5000000

Notes
-----
- Progress is checkpointed every CHECKPOINT_EVERY passages so interrupted runs
  can be resumed with --resume.
- Wikipedia articles shorter than 20 words are skipped (stubs, redirects).
- Passages that are part of a "See also" or "External links" section are
  dropped because they add noise without factual content.
- The script respects --device; on multi-GPU machines only device 0 is used
  because sentence-transformers handles device assignment internally.

Requirements
------------
  pip install datasets sentence-transformers faiss-gpu torch tqdm
  (faiss-cpu also works, just slower for the final index build)
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CHUNK_WORDS: int = 100          # target passage length (DPR default)
CHUNK_OVERLAP: int = 10         # words shared between adjacent passages
CHECKPOINT_EVERY: int = 100_000 # save a partial index every N passages
MIN_ARTICLE_WORDS: int = 20     # skip stubs shorter than this
SKIP_SECTIONS = {               # section titles to drop (noisy, non-factual)
    "see also", "references", "external links", "notes", "bibliography",
    "further reading", "footnotes",
}

# -----------------------------------------------------------------------------
# Text utilities
# -----------------------------------------------------------------------------

def _chunk_text(text: str, title: str = "", chunk_words: int = CHUNK_WORDS,
                overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping ~chunk_words-word chunks.

    Each chunk is prefixed with the article/section title so the passage is
    self-contained (important for TF-IDF-style matching inside Flan-T5).

    Parameters
    ----------
    text : str
        Plain-text article body or section body.
    title : str
        Article title (prepended to every chunk).
    chunk_words : int
        Target number of words per chunk.
    overlap : int
        Number of words shared between adjacent chunks.

    Returns
    -------
    list of str  -- passage strings ready for embedding.
    """
    words = text.split()
    if len(words) < 10:
        return []

    chunks = []
    step = max(1, chunk_words - overlap)
    for start in range(0, len(words), step):
        chunk_words_list = words[start: start + chunk_words]
        if len(chunk_words_list) < 10:
            break
        body = " ".join(chunk_words_list)
        passage = f"{title}: {body}" if title else body
        chunks.append(passage)
    return chunks


def _article_to_passages(article: dict, chunk_words: int = CHUNK_WORDS) -> List[str]:
    """Extract passages from a single HuggingFace Wikipedia article dict.

    The wikipedia dataset has fields: ``id``, ``url``, ``title``, ``text``.
    The ``text`` field is the full article as plain text (no markup).
    We split by double-newline to get sections, then chunk each section.
    """
    title = article.get("title", "").strip()
    text = article.get("text", "").strip()

    if not text or len(text.split()) < MIN_ARTICLE_WORDS:
        return []

    passages: List[str] = []
    sections = text.split("\n\n")
    current_section_title = title

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Detect section headers: short lines (< 8 words) with no period.
        lines = section.split("\n")
        first_line = lines[0].strip()
        if (len(first_line.split()) <= 7 and "." not in first_line
                and len(lines) >= 2):
            # This looks like a section header.
            current_section_title = first_line
            # Check if this is a noisy section to skip.
            if current_section_title.lower().strip() in SKIP_SECTIONS:
                continue
            section_body = "\n".join(lines[1:]).strip()
        else:
            section_body = section

        if not section_body:
            continue

        section_passages = _chunk_text(
            section_body,
            title=current_section_title,
            chunk_words=chunk_words,
        )
        passages.extend(section_passages)

    return passages


# -----------------------------------------------------------------------------
# Streaming article -> passages generator
# -----------------------------------------------------------------------------

def stream_passages(
    max_passages: int,
    chunk_words: int = CHUNK_WORDS,
    dataset_name: str = "wikipedia",
    dataset_config: str = "20220301.en",
) -> Iterator[str]:
    """Stream Wikipedia and yield individual passage strings.

    Uses HuggingFace datasets in ``streaming=True`` mode so the full
    corpus is never materialised in memory.

    Parameters
    ----------
    max_passages : int
        Stop after yielding this many passages.
    chunk_words : int
    dataset_name, dataset_config : str
        Passed directly to ``datasets.load_dataset``.

    Yields
    ------
    str -- one passage string per iteration.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install HuggingFace datasets:  pip install datasets")
        sys.exit(1)

    logger.info(
        "Streaming %s / %s from HuggingFace ...", dataset_name, dataset_config
    )
    # Note (EXP-09 fix 2026-04-01): The original 'wikipedia'/'20220301.en' uses a
    # legacy Python script (wikipedia.py) no longer supported by HF datasets.
    # Default is now 'wikimedia/wikipedia'/'20231101.en' -- Parquet-based, same schema.
    ds = load_dataset(
        dataset_name, dataset_config,
        split="train",
        streaming=True,
    )

    yielded = 0
    articles_seen = 0
    for article in ds:
        articles_seen += 1
        for passage in _article_to_passages(article, chunk_words=chunk_words):
            yield passage
            yielded += 1
            if yielded >= max_passages:
                logger.info(
                    "Reached max_passages=%d after %d articles.",
                    max_passages, articles_seen,
                )
                return


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

def encode_passages(
    passages: List[str],
    model_name: str,
    batch_size: int = 512,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode passage strings into L2-normalised vectors.

    Parameters
    ----------
    passages : list of str
    model_name : str
        SBERT model name -- must match the one used in EpisodicMemoryStore
        (default: ``sentence-transformers/all-mpnet-base-v2``).
    batch_size : int
        Encoding batch size. 512 works well on 16 GB GPU; reduce for CPU.
    device : str or None
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    show_progress : bool

    Returns
    -------
    np.ndarray, shape (N, D), dtype float32, L2-normalised.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("Install sentence-transformers:  pip install sentence-transformers")
        sys.exit(1)

    logger.info("Loading SBERT model: %s ...", model_name)
    model = SentenceTransformer(model_name, device=device)

    logger.info(
        "Encoding %d passages in batches of %d ...", len(passages), batch_size
    )
    t0 = time.time()
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise in-place
    )
    elapsed = time.time() - t0
    logger.info(
        "Encoding done in %.1f s  (%.0f passages/s).",
        elapsed, len(passages) / max(elapsed, 1),
    )
    return embeddings.astype(np.float32)


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------

def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "_checkpoint.pkl"


def _save_checkpoint(
    output_dir: Path,
    passages_so_far: List[str],
    embeddings_so_far: List[np.ndarray],
) -> None:
    cp = _checkpoint_path(output_dir)
    with open(cp, "wb") as f:
        pickle.dump({"passages": passages_so_far, "embeddings": embeddings_so_far}, f)
    logger.info("Checkpoint saved: %d passages.", len(passages_so_far))


def _load_checkpoint(output_dir: Path):
    cp = _checkpoint_path(output_dir)
    if not cp.exists():
        return None, None
    with open(cp, "rb") as f:
        data = pickle.load(f)
    logger.info("Resuming from checkpoint: %d passages.", len(data["passages"]))
    return data["passages"], data["embeddings"]


# -----------------------------------------------------------------------------
# Main build routine
# -----------------------------------------------------------------------------

def build_passage_index(
    output_dir: str,
    max_passages: int = 500_000,
    chunk_words: int = CHUNK_WORDS,
    encode_batch_size: int = 512,
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2",
    device: Optional[str] = None,
    resume: bool = False,
    dataset_name: str = "wikipedia",
    dataset_config: str = "20220301.en",
    index_type: str = "ivf_pq",
    nlist: int = 65_536,
    nprobe: int = 64,
    pq_m: int = 64,
    pq_nbits: int = 8,
    train_sample_size: int = 500_000,
) -> None:
    """Build and save a PassageStore from Wikipedia.

    Parameters
    ----------
    output_dir : str
        Directory to write ``passages.faiss`` and ``passages.pkl``.
        Created if it does not exist.
    max_passages : int
        Maximum number of passages to include.
    chunk_words : int
        Target words per passage chunk.
    encode_batch_size : int
        SBERT encoding batch size.
    sbert_model : str
        SBERT model name. Must match the configured embedding dimension.
    device : str or None
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto).
    resume : bool
        If True, load existing checkpoint and continue.
    dataset_name, dataset_config : str
        HuggingFace dataset to stream.
    """
    import sys
    # Verify SBERT + FAISS are available before streaming anything.
    _check_dependencies()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- Determine auto device ----------------------------------------------
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    logger.info("Using device: %s", device)

    # -- Resume from checkpoint or start fresh -----------------------------
    all_passages: List[str] = []
    all_embedding_batches: List[np.ndarray] = []

    if resume:
        cp_passages, cp_embeddings = _load_checkpoint(out)
        if cp_passages is not None:
            all_passages = cp_passages
            all_embedding_batches = cp_embeddings

    already_have = len(all_passages)
    remaining = max_passages - already_have
    if remaining <= 0:
        logger.info("Already have %d passages -- nothing to stream.", already_have)
    else:
        # -- Stream passages ------------------------------------------------
        logger.info(
            "Streaming up to %d more passages (have %d, target %d) ...",
            remaining, already_have, max_passages,
        )
        buffer: List[str] = []
        total_streamed = already_have

        for passage in stream_passages(
            max_passages=remaining,
            chunk_words=chunk_words,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
        ):
            buffer.append(passage)
            total_streamed += 1

            # Encode and checkpoint in batches.
            if len(buffer) >= CHECKPOINT_EVERY:
                logger.info("Encoding buffer of %d passages ...", len(buffer))
                embs = encode_passages(
                    buffer, sbert_model,
                    batch_size=encode_batch_size,
                    device=device,
                )
                all_passages.extend(buffer)
                all_embedding_batches.append(embs)
                _save_checkpoint(out, all_passages, all_embedding_batches)
                buffer = []

        # Encode remaining buffer.
        if buffer:
            logger.info("Encoding final buffer of %d passages ...", len(buffer))
            embs = encode_passages(
                buffer, sbert_model,
                batch_size=encode_batch_size,
                device=device,
            )
            all_passages.extend(buffer)
            all_embedding_batches.append(embs)

    if not all_passages:
        logger.error("No passages were collected. Aborting.")
        sys.exit(1)

    # -- Concatenate all embeddings -----------------------------------------
    logger.info(
        "Concatenating embeddings from %d batches ...", len(all_embedding_batches)
    )
    all_embeddings = np.concatenate(all_embedding_batches, axis=0)
    emb_dim = all_embeddings.shape[1]  # Must match the configured SBERT model output dimension.
    assert all_embeddings.shape[0] == len(all_passages), (
        f"Count mismatch: {all_embeddings.shape[0]} embeddings vs {len(all_passages)} passages"
    )
    logger.info("Embeddings: %d × %d (dim=%d)", len(all_passages), emb_dim, emb_dim)

    # -- Build and save PassageStore ---------------------------------------
    logger.info("Building PassageStore for %d passages ...", len(all_passages))
    # Import here so the script can be run without the full caem package in PYTHONPATH
    # as long as the caem/ directory is on the path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from caem.config import CAEMConfig
    from caem.retrieval.rag import PassageStore

    cfg = CAEMConfig()
    cfg.rag_index_type = index_type
    cfg.rag_faiss_nlist = nlist
    cfg.rag_faiss_nprobe = nprobe
    cfg.rag_faiss_pq_m = pq_m
    cfg.rag_faiss_pq_nbits = pq_nbits
    cfg.rag_faiss_train_sample_size = train_sample_size

    store = PassageStore(all_passages, all_embeddings, config=cfg)
    store.save(str(out))
    logger.info("PassageStore saved to %s (%d passages).", out, len(all_passages))

    # -- Clean up checkpoint -----------------------------------------------
    cp = _checkpoint_path(out)
    if cp.exists():
        cp.unlink()
        logger.info("Checkpoint file removed.")

    # -- Quick sanity check ------------------------------------------------
    _sanity_check(out, sbert_model, device)


def _sanity_check(output_dir: Path, sbert_model: str, device: str) -> None:
    """Load the saved store and run one test query to verify correctness."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from caem.retrieval.rag import PassageStore
    from sentence_transformers import SentenceTransformer

    logger.info("Running sanity check ...")
    store = PassageStore.load(str(output_dir))
    model = SentenceTransformer(sbert_model, device=device)

    test_query = "Who wrote Romeo and Juliet?"
    emb = model.encode(test_query, normalize_embeddings=True).astype(np.float32)
    hits = store.search(emb, k=3)

    if not hits:
        logger.warning("Sanity check: no hits returned -- passage store may be empty.")
        return

    logger.info("Sanity check PASSED. Top-3 hits for '%s':", test_query)
    for i, (passage, score) in enumerate(hits, 1):
        logger.info("  [%d] score=%.4f  %s", i, score, passage[:120])


def _check_dependencies() -> None:
    """Fail fast if required packages are missing."""
    missing = []
    for pkg in ["datasets", "sentence_transformers", "faiss", "numpy", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Missing required packages: %s", ", ".join(missing))
        logger.error(
            "Install with:  pip install datasets sentence-transformers faiss-gpu torch"
        )
        sys.exit(1)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Wikipedia passage index for TierThreeRAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output_dir",
        default="data/passage_index",
        help="Directory to save passages.faiss + passages.pkl.",
    )
    p.add_argument(
        "--max_passages",
        type=int,
        default=500_000,
        help=(
            "Maximum number of passages to index. "
            "500K gives good benchmark coverage in ~20 min on GPU. "
            "Use --max_passages 1000 for a quick smoke test."
        ),
    )
    p.add_argument(
        "--chunk_words",
        type=int,
        default=CHUNK_WORDS,
        help="Target number of words per passage (DPR default: 100).",
    )
    p.add_argument(
        "--encode_batch_size",
        type=int,
        default=512,
        help="SBERT encoding batch size. Reduce if OOM on CPU.",
    )
    p.add_argument(
        "--sbert_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help=(
            "SBERT model for passage encoding. Must match the model used "
            "in EpisodicMemoryStore (config.sbert_model)."
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu", None],
        help="Compute device. Default: auto-detect.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if a previous run was interrupted.",
    )
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Quick test: index only 1 000 passages (overrides --max_passages).",
    )
    p.add_argument(
        "--dataset_name",
        default="wikimedia/wikipedia",
        help="HuggingFace dataset name. Default is wikimedia/wikipedia (Parquet, no legacy script).",
    )
    p.add_argument(
        "--dataset_config",
        default="20231101.en",
        help="HuggingFace dataset config / subset. Default is 20231101.en (Nov 2023 English dump).",
    )
    p.add_argument(
        "--index_type",
        default="ivf_pq",
        choices=["ivf_pq", "flat_ip"],
        help="Passage FAISS backend. Use ivf_pq for thesis-scale indexes.",
    )
    p.add_argument(
        "--nlist",
        type=int,
        default=65_536,
        help="IVF coarse centroid count for passage index.",
    )
    p.add_argument(
        "--nprobe",
        type=int,
        default=64,
        help="Number of IVF lists to probe at query time.",
    )
    p.add_argument(
        "--pq_m",
        type=int,
        default=64,
        help="Number of subquantizers for IVF-PQ.",
    )
    p.add_argument(
        "--pq_nbits",
        type=int,
        default=8,
        help="Bits per IVF-PQ subquantizer code.",
    )
    p.add_argument(
        "--train_sample_size",
        type=int,
        default=500_000,
        help="Maximum vectors sampled to train IVF-PQ quantizers.",
    )
    return p.parse_args()


if __name__ == "__main__":
    ns = _parse_args()

    if ns.smoke_test:
        logger.info("Smoke-test mode: capping at 1 000 passages.")
        ns.max_passages = 1_000

    build_passage_index(
        output_dir=ns.output_dir,
        max_passages=ns.max_passages,
        chunk_words=ns.chunk_words,
        encode_batch_size=ns.encode_batch_size,
        sbert_model=ns.sbert_model,
        device=ns.device,
        resume=ns.resume,
        dataset_name=ns.dataset_name,
        dataset_config=ns.dataset_config,
        index_type=ns.index_type,
        nlist=ns.nlist,
        nprobe=ns.nprobe,
        pq_m=ns.pq_m,
        pq_nbits=ns.pq_nbits,
        train_sample_size=ns.train_sample_size,
    )
