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
(UnifiedVerifier) and Stage 7 (storage decision) -- Tier 3 answers
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

# Defensive FAISS thread cap -- belt-and-suspenders for
# OMP_NUM_THREADS/MKL_NUM_THREADS env vars. FAISS-CPU's internal
# OpenMP thread pool sometimes ignores environment variables if they
# were set after FAISS was already imported. Calling
# faiss.omp_set_num_threads() explicitly guarantees FAISS respects the
# cap regardless of when env vars were set. Value matches the caps in
# NEXT_SESSION_PLAN.md Step 3.2B. 16 threads is the sweet spot for
# IVF k-means on EPYC (memory-bandwidth-bound beyond that). Applied
# once at module import time so every FAISS call in the process is
# capped.
try:
    cast(Any, faiss).omp_set_num_threads(16)
    logger.info("faiss.omp_set_num_threads(16) applied at rag.py import.")
except Exception as _exc:
    logger.warning(
        "faiss.omp_set_num_threads(16) failed (%s); relying on OMP env vars.",
        _exc,
    )


# -----------------------------------------------------------------------------
# PassageStore
# -----------------------------------------------------------------------------

class PassageStore:
    """Read-only FAISS index over a Wikipedia passage corpus.

        Each passage is a ~100-word chunk from Wikipedia (DPR-style split,
        Karpukhin et al. 2020). Embeddings are N-dim SBERT vectors, L2-normalised.

        Index backend is configurable via CAEMConfig.rag_index_type:
            - "ivf_pq" (plan default): scalable ANN retrieval for large corpora.
            - "flat_ip": exact cosine search, useful for tiny debug corpora.

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
        emb_f32 = np.ascontiguousarray(embeddings.astype(np.float32))

        index_type = str(getattr(self.config, "rag_index_type", "ivf_pq")).lower()
        if index_type == "ivf_pq":
            requested_nlist = int(getattr(self.config, "rag_faiss_nlist", 65_536))
            nprobe = int(getattr(self.config, "rag_faiss_nprobe", 64))
            pq_m = int(getattr(self.config, "rag_faiss_pq_m", 64))
            pq_nbits = int(getattr(self.config, "rag_faiss_pq_nbits", 8))
            train_cap = int(getattr(self.config, "rag_faiss_train_sample_size", 500_000))

            # Keep nlist feasible for smaller debug indexes while preserving
            # the plan target on large corpora.
            effective_nlist = min(requested_nlist, max(1, emb_f32.shape[0] // 8))

            try:
                quantizer = faiss.IndexFlatIP(self._dim)
                try:
                    ivf = faiss.IndexIVFPQ(
                        quantizer,
                        self._dim,
                        effective_nlist,
                        pq_m,
                        pq_nbits,
                        faiss.METRIC_INNER_PRODUCT,
                    )
                except TypeError:
                    ivf = faiss.IndexIVFPQ(
                        quantizer,
                        self._dim,
                        effective_nlist,
                        pq_m,
                        pq_nbits,
                    )
                    if hasattr(ivf, "metric_type"):
                        ivf.metric_type = faiss.METRIC_INNER_PRODUCT

                if train_cap > 0 and emb_f32.shape[0] > train_cap:
                    rng = np.random.default_rng(seed=0)
                    idx = rng.choice(emb_f32.shape[0], size=train_cap, replace=False)
                    train_vecs = emb_f32[idx]
                else:
                    train_vecs = emb_f32

                # GPU acceleration when faiss-gpu is installed: IVF k-means
                # on 2M x 768 vectors is ~80 min on faiss-cpu vs ~2-4 min
                # on the 5090. Detection is runtime-safe: falls back to CPU
                # if StandardGpuResources is missing (faiss-cpu install).
                _use_gpu = hasattr(faiss, "StandardGpuResources")
                if _use_gpu:
                    try:
                        _gpu_res = cast(Any, faiss).StandardGpuResources()
                        ivf_gpu = cast(Any, faiss).index_cpu_to_gpu(
                            _gpu_res, 0, ivf
                        )
                        cast(Any, ivf_gpu).train(
                            np.ascontiguousarray(train_vecs))
                        cast(Any, ivf_gpu).add(emb_f32)
                        # Move back to CPU for serialization via
                        # faiss.write_index (Step 4 write_store).
                        ivf = cast(Any, faiss).index_gpu_to_cpu(ivf_gpu)
                        logger.info(
                            "PassageStore IVF-PQ train+add ran on GPU "
                            "(faiss-gpu detected); swapped back to CPU "
                            "for serialization.")
                    except Exception as _gpu_exc:
                        logger.warning(
                            "faiss-gpu path failed (%s); falling back to "
                            "CPU train+add.", _gpu_exc)
                        cast(Any, ivf).train(
                            np.ascontiguousarray(train_vecs))
                        cast(Any, ivf).add(emb_f32)
                else:
                    cast(Any, ivf).train(np.ascontiguousarray(train_vecs))
                    cast(Any, ivf).add(emb_f32)
                ivf.nprobe = nprobe
                self._index = ivf
                logger.info(
                    "PassageStore: %d passages indexed with IVF-PQ "
                    "(dim=%d, nlist=%d, nprobe=%d, m=%d, nbits=%d).",
                    len(passages),
                    self._dim,
                    effective_nlist,
                    nprobe,
                    pq_m,
                    pq_nbits,
                )
            except Exception as exc:
                logger.warning(
                    "PassageStore IVF-PQ build failed (%s); falling back to FlatIP.",
                    exc,
                )
                self._index = faiss.IndexFlatIP(self._dim)
                cast(Any, self._index).add(emb_f32)
                logger.info("PassageStore: %d passages indexed with FlatIP fallback.", len(passages))
        else:
            # Inner-product index -- cosine similarity because embeddings are
            # L2-normalised.
            self._index = faiss.IndexFlatIP(self._dim)
            cast(Any, self._index).add(emb_f32)
            logger.info("PassageStore: %d passages indexed with FlatIP (dim=%d).", len(passages), self._dim)

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
        store.config = CAEMConfig()
        store.passages = passages
        store._dim = index.d
        store._index = index
        try:
            ivf = faiss.extract_index_ivf(index)
            ivf.nprobe = int(getattr(store.config, "rag_faiss_nprobe", ivf.nprobe))
        except Exception:
            pass
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
            logger.error(
                "RAG generation failed (Tier 3 fallback returning empty string): %s. "
                "If this repeats, check passage index integrity and FAISS installation.",
                exc,
            )
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
        """Build the uniform scaffolded-CoT RAG prompt for Flan-T5.

        All benchmarks use the same Evidence / Reasoning / Answer template
        so the verifier's 9-signal composite receives substantive claim
        text uniformly across tasks. The scaffold compels the model to
        emit at least 40 words of reasoning before the final answer,
        which converts classification-style outputs (FEVER / StrategyQA /
        ARC) from degenerate 1-word labels into propositional statements
        the passage-grounding signals (p_ground_max, p_ground_mean,
        p_ground_atomic, p_contra) can score.

        Template (Tier 3 RAG, with passages):

            Context:
            [1] <passage>
            [2] <passage>
            [3] <passage>

            {task_instruction}

            You MUST follow the exact response format below. Your
            reasoning must be at least 40 words, step by step.

            Evidence: <quote the most relevant sentence from the
                       context above>
            Reasoning: <at least 40 words of step-by-step analysis
                       linking the evidence to the final answer>
            Answer: {answer_format}

        This matches the Flan instruction-tuning style (Wei et al. 2022)
        while forcing substantive reasoning output -- Flan-T5 with
        do_sample=False otherwise short-circuits to the highest-
        probability single-token continuation on classification tasks.
        """
        lines = ["Context:"]
        for i, (passage, _score) in enumerate(passages, start=1):
            lines.append(f"[{i}] {passage}")

        task = self._detect_query_task(query)
        task_line, answer_format = self._task_spec(task, query)

        lines.append("")
        lines.append(task_line)
        lines.append("")
        lines.append(
            "You MUST follow the exact response format below. Your "
            "reasoning must be at least 40 words, step by step.",
        )
        lines.append("")
        lines.append(
            "Evidence: <quote the most relevant sentence from the "
            "context above>",
        )
        lines.append(
            "Reasoning: <at least 40 words of step-by-step analysis "
            "linking the evidence to the final answer>",
        )
        lines.append(f"Answer: {answer_format}")
        return "\n".join(lines)

    def _task_spec(self, task: str, query: str) -> Tuple[str, str]:
        """Return (task_instruction_line, answer_format_spec) per task.

        Shared by Tier 3 (RAG) and caller code that needs the same
        task framing without the Context block.
        """
        if task == "fever":
            claim = self._extract_after_token(query, "Claim:")
            return (
                f"Claim: {claim}\n"
                "Determine whether the claim is SUPPORTS, REFUTES, or "
                "NOT ENOUGH INFO based on the context above.",
                "supports | refutes | not enough info",
            )
        if task == "strategyqa":
            q_text = self._extract_after_token(query, "Question:")
            return (
                f"Question: {q_text}\n"
                "Answer the question with yes or no based on the "
                "context above.",
                "yes | no",
            )
        if task == "arc":
            return (
                f"{query}\n"
                "Choose the correct answer from the listed choices "
                "based on the context above.",
                "A | B | C | D",
            )
        # Open-ended QA (TriviaQA, NQ, TruthfulQA): factual short-form
        # or detailed answer depending on the question. The answer
        # format slot lets the model decide based on question style.
        return (
            f"Question: {query}\n"
            "Answer the question based on the context above.",
            "<concise factual answer>",
        )

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
        """Infer benchmark task style from constrained prompt prefixes.

        Returns one of ``"fever"``, ``"strategyqa"``, ``"arc"``, or
        ``"open"``. ARC-Challenge queries are recognised by the
        "Choices: (A) ... (B) ..." suffix that ``load_arc_challenge``
        builds into the query string.
        """
        q = query.lower().strip()
        if q.startswith("answer with one of: supports, refutes, not enough info."):
            return "fever"
        if q.startswith("answer yes or no."):
            return "strategyqa"
        if ("choices:" in q) and ("multiple choice letter" in q):
            return "arc"
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

    def generate_batch(self, queries: List[str]) -> List[str]:
        """Generate N Tier 3 RAG answers in one batched T5 forward pass.

        Used by Level B static batching (BatchPipeline.batch_tier3_generate).
        Retrieval and prompt construction still run per-query (cheap FAISS
        + string ops); the expensive T5 ``generate`` call is batched.

        Returns a list of length N in input order. Empty string at position
        i signals a failed generation for that query (caller escalates).
        """
        if not queries:
            return []

        cfg = self.config

        prompts: List[str] = []
        for q in queries:
            passages = self._retrieve(q, k=cfg.rag_top_k)
            if passages:
                prompts.append(self._build_prompt(q, passages))
            else:
                logger.warning("RAG batch: no passages for query '%s...' -- falling back to query-only.", q[:60])
                prompts.append(q)

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        try:
            self.model.eval()
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=cfg.rag_max_new_tokens,
                    do_sample=cfg.rag_do_sample,
                )
        except Exception as exc:
            logger.error(
                "RAG generate_batch failed (N=%d): %s -- returning empty strings.",
                len(queries), exc,
            )
            return [""] * len(queries)

        decoded: List[str] = []
        for i in range(output_ids.shape[0]):
            try:
                s = self.tokenizer.decode(
                    output_ids[i], skip_special_tokens=True,
                ).strip()
            except Exception as exc:
                logger.warning("RAG generate_batch decode %d failed: %s", i, exc)
                s = ""
            decoded.append(s)
        return decoded
