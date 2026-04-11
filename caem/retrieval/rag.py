"""
caem/retrieval/rag.py
======================
Tier 3 RAG -- Stage 6 of the CAEM pipeline.

Runs when a query is routed to Tier 3 by the AdaptiveRouter:
  - u_pre < safety_u_pre_min (OR-condition veto), OR
  - No high-quality memory match exists.

Two components
--------------
PassageStore
    Thin FAISS wrapper over a static Wikipedia passage corpus.
    Encodes queries with the same SBERT encoder used by EpisodicMemoryStore
    so both indices share the same SBERT embedding space (768-dim for
    all-mpnet-base-v2 in this repository).
    The corpus is read-only at inference -- passages are never modified.

TierThreeRAG
    Retrieves top-k passages from PassageStore, builds a context-augmented
    prompt, and generates an answer with Flan-T5.

Prompt format [DES]
-------------------
    Context:
    [1] <passage_1>
    [2] <passage_2>
    ...
    [k] <passage_k>

    Question: <query>
    Answer:

Flan-T5 was instruction-tuned with exactly this style of numbered context
block (see FLAN collection, Wei et al. 2022). Using it consistently means
the model's learned priors align with the prompt structure.

RAG vs Tier 1/2
---------------
Tier 1: direct answer from episodic memory -- no model call.
Tier 2: Flan-T5 generation conditioned on the query alone.
Tier 3: Flan-T5 generation conditioned on retrieved Wikipedia passages.

Tier 3 is the most expensive path but the most robust -- it is used when
the system is uncertain (low u_pre) or has no relevant memory.

After Tier 3 generation the answer still passes through Stage 5
(MultiLayerVerifier) and Stage 7 (storage decision) -- Tier 3 answers
that verify well are stored so future similar queries hit Tier 1 or 2.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import faiss
import numpy as np
import torch

from caem.config import CAEMConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# PassageStore
# -----------------------------------------------------------------------------

class PassageStore:
    """Read-only FAISS index over a Wikipedia passage corpus.

    Each passage is a ~100-word chunk from Wikipedia (DPR-style split,
    Karpukhin et al. 2020). Embeddings are N-dim SBERT vectors, L2-normalised,
    stored in a flat inner-product index (cosine similarity via dot product).

    Parameters
    ----------
    passages : list of str
        The raw passage strings. Index i in this list corresponds to FAISS
        vector id i.
    embeddings : np.ndarray, shape (N, dim), dtype float32
        Pre-computed L2-normalised SBERT embeddings for all passages.
        Computed once offline (see scripts/build_passage_index.py).

    Usage
    -----
    >>> store = PassageStore(passages, embeddings)
    >>> hits = store.search(query_emb, k=5)
    >>> for passage, score in hits:
    ...     print(score, passage[:80])
    """

    def __init__(self, passages: List[str], embeddings: np.ndarray, config: Optional[CAEMConfig] = None) -> None:
        self.config = config or CAEMConfig()
        self._dim = self.config.embedding_dim

        if len(passages) != embeddings.shape[0]:
            raise ValueError(
                f"passages length {len(passages)} != embeddings rows {embeddings.shape[0]}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D, got {embeddings.shape}"
            )
        if embeddings.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._dim}, got {embeddings.shape[1]}"
            )

        self.passages = passages

        # Inner-product index -- cosine similarity because embeddings are
        # L2-normalised (same design as EpisodicMemoryStore).
        self._index = faiss.IndexFlatIP(self._dim)
        # FAISS SWIG stubs expose low-level signatures; runtime supports add(x).
        cast(Any, self._index).add(embeddings.astype(np.float32))

        logger.info("PassageStore: %d passages indexed (dim=%d).", len(passages), self._dim)

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return top-k (passage, cosine_score) pairs for a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray, shape (dim,), float32, L2-normalised
        k : int
            Number of passages to retrieve.

        Returns
        -------
        list of (passage_str, score) sorted by score descending.
        Empty list if the store is empty.
        """
        if self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)
        q = query_embedding.astype(np.float32).reshape(1, self._dim)
        # FAISS SWIG stubs expose low-level signatures; runtime supports search(x, k).
        scores, ids = cast(Any, self._index).search(q, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((self.passages[idx], float(score)))
        return results

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save FAISS index + passages to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "passages.faiss"))
        with open(p / "passages.pkl", "wb") as f:
            pickle.dump(self.passages, f)
        logger.info("PassageStore saved to %s (%d passages).", path, len(self.passages))

    @classmethod
    def load(cls, path: str) -> "PassageStore":
        """Load PassageStore from disk (bypasses embedding recomputation)."""
        p = Path(path)
        index = faiss.read_index(str(p / "passages.faiss"))
        with open(p / "passages.pkl", "rb") as f:
            passages = pickle.load(f)

        # Reconstruct: wrap existing index directly
        store = cls.__new__(cls)
        store.passages = passages
        store._dim = index.d
        store._index = index
        logger.info("PassageStore loaded from %s (%d passages).", path, len(passages))
        return store

    @property
    def size(self) -> int:
        return self._index.ntotal


# -----------------------------------------------------------------------------
# TierThreeRAG
# -----------------------------------------------------------------------------

class TierThreeRAG:
    """Retrieve-then-generate for Tier 3 queries.

    Parameters
    ----------
    model : transformers.T5ForConditionalGeneration
        Flan-T5-Large in eval mode. Shared with Tier 2.
    tokenizer : transformers.AutoTokenizer
        Matching tokenizer.
    passage_encoder
        SBERT QueryEncoder (shared with EpisodicMemoryStore and verifier).
    passage_store : PassageStore
        Pre-built Wikipedia passage index.
    config : CAEMConfig
    device : str or None

    Usage
    -----
    >>> rag = TierThreeRAG(model, tokenizer, encoder, passage_store)
    >>> answer = rag.generate("Who wrote Hamlet?")
    """

    def __init__(
        self,
        model,
        tokenizer,
        passage_encoder,
        passage_store: PassageStore,
        config: Optional[CAEMConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.passage_encoder = passage_encoder
        self.passage_store = passage_store
        self.config = config or CAEMConfig()

        if device is None:
            device = str(next(model.parameters()).device)
        self.device = device

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        query: str,
        input_ids: Optional[torch.Tensor] = None,
    ) -> str:
        """Generate a RAG answer for a Tier 3 query.

        Parameters
        ----------
        query : str
            The original query text.
        input_ids : torch.Tensor or None
            Pre-tokenized plain query (without context). If None, tokenized
            internally. Note: the RAG prompt is always re-tokenized with
            context prepended -- input_ids here is only used as a fallback
            if retrieval fails completely.

        Returns
        -------
        str
            Decoded answer string. Returns empty string on total failure.
        """
        cfg = self.config

        # Step 1: encode query -> retrieve passages
        passages = self._retrieve(query, k=cfg.rag_top_k)

        # Step 2: build the RAG prompt
        if passages:
            prompt = self._build_prompt(query, passages)
        else:
            # Degenerate: no passages found -- fall back to query-only generation
            logger.warning("RAG: no passages retrieved -- falling back to query-only.")
            prompt = query

        # Step 3: tokenize prompt
        prompt_ids = self._tokenize_prompt(prompt)

        # Step 4: generate
        try:
            self.model.eval()
            with torch.no_grad():
                output_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=cfg.rag_max_new_tokens,
                    do_sample=cfg.rag_do_sample,
                )
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            logger.debug("RAG answer (%d passages): %s", len(passages), answer[:120])
            return answer

        except Exception as exc:
            logger.error("RAG generation failed: %s", exc)
            return ""

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Public retrieval endpoint -- returns (passage, score) pairs.

        Useful for inspection and ablation studies.
        """
        k = k if k is not None else self.config.rag_top_k
        return self._retrieve(query, k)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Encode query and search the passage store."""
        try:
            emb = self.passage_encoder.encode(query)
            emb = emb.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return self.passage_store.search(emb, k=k)
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return []

    def _build_prompt(
        self,
        query: str,
        passages: List[Tuple[str, float]],
    ) -> str:
        """Build the numbered-context RAG prompt for Flan-T5.

        Format matches the FLAN instruction-tuning style (Wei et al. 2022):
            Context:
            [1] <passage>
            ...
            Question: <query>
            Answer:
        """
        lines = ["Context:"]
        for i, (passage, _score) in enumerate(passages, start=1):
            lines.append(f"[{i}] {passage}")

        task = self._detect_query_task(query)
        if task == "fever":
            claim = self._extract_after_token(query, "Claim:")
            lines.append("")
            lines.append("Determine whether the claim is supports, refutes, or not enough info using the context above.")
            lines.append(f"Claim: {claim}")
            lines.append("Reasoning: <short explanation>")
            lines.append("Answer: supports|refutes|not enough info")
        elif task == "strategyqa":
            q_text = self._extract_after_token(query, "Question:")
            lines.append("")
            lines.append("Answer the question using the context above.")
            lines.append(f"Question: {q_text}")
            lines.append("Reasoning: <short explanation>")
            lines.append("Answer: yes|no")
        else:
            lines.append(f"\nQuestion: {query}")
            lines.append("Think step by step.")
            lines.append("Answer:")
        return "\n".join(lines)

    @staticmethod
    def _extract_after_token(query: str, token: str) -> str:
        """Return substring after token (case-insensitive), else full query."""
        q_lower = query.lower()
        t_lower = token.lower()
        idx = q_lower.find(t_lower)
        if idx == -1:
            return query.strip()
        return query[idx + len(token):].strip()

    @staticmethod
    def _detect_query_task(query: str) -> str:
        """Infer benchmark task style from constrained prompt prefixes."""
        q = query.lower().strip()
        if q.startswith("answer with one of: supports, refutes, not enough info."):
            return "fever"
        if q.startswith("answer yes or no."):
            return "strategyqa"
        return "open"

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize the full RAG prompt, truncating context to fit encoder limit."""
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,          # Flan-T5-Large encoder limit
        )
        return enc["input_ids"].to(self.device)
