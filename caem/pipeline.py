"""
caem/pipeline.py
=================
CAEMPipeline -- end-to-end orchestrator for the CAEM inference pipeline.

Ties all 8 stages together for a single query:

  Stage 1 -- EpisodicMemoryStore     encode + search
  Stage 2 -- QueryEncoder            produce L2-normalised embedding
  Stage 3 -- PreRoutingConfidence    u_pre (2 fast signals)
  Stage 3 -- AdaptiveRouter          Tier 1 / 2 / 3 dispatch
  Stage 4a -- PostGenerationConf.    û (4 signals, Tier 2 only)
  Stage 5 -- MultiLayerVerifier      û_stored (quality gate, all tiers)
  Stage 6 -- TierThreeRAG            retrieval-augmented generation
  Stage 7 -- Storage decision        novelty + û_stored threshold
  Stage 8 -- SelfImprovementLoop     (called externally; not triggered here)

Inference flow
--------------

    query -> encode -> u_pre -> memory search -> route
        Tier 1  -> return stored answer -> verify -> maybe store
        Tier 2  -> generate -> û -> [escalate?] -> verify -> maybe store
        Tier 3  -> RAG generate -> verify -> maybe store

The pipeline is stateful (the EpisodicMemoryStore grows across calls).
Thread safety: NOT thread-safe. Add an external lock if parallelising.

Usage
-----
>>> pipeline = CAEMPipeline(model, tokenizer, encoder,
...                         nli_model, nli_tokenizer, passage_store)
>>> result = pipeline.answer("Who wrote Hamlet?")
>>> result.answer          # "William Shakespeare"
>>> result.tier            # 1, 2, or 3
>>> result.stored          # True if written to memory this call
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from caem.config import CAEMConfig
from caem.confidence.post_generation import PostGenerationConfidenceEstimator
from caem.confidence.pre_routing import PreRoutingConfidenceEstimator
from caem.memory.entry import (
    EpisodicEntry,
    PostGenerationConfidence,
    PreRoutingConfidence,
    RoutingDecision,
    StoredConfidence,
)
from caem.memory.store import EpisodicMemoryStore
from caem.retrieval.rag import PassageStore, TierThreeRAG
from caem.routing.router import AdaptiveRouter
from caem.verification.verifier import MultiLayerVerifier

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Full record of one pipeline inference call.

    Attributes
    ----------
    query : str
        The input query string.
    answer : str
        The final answer returned to the caller.
    tier : int
        Which tier handled the query (1, 2, or 3).
    stored : bool
        True if this answer was written to episodic memory.
    latency_ms : float
        Wall-clock time for the full pipeline call in milliseconds.
    routing_decision : RoutingDecision
        Full routing metadata (tier, scores, safety_override, etc.).
    pre_confidence : PreRoutingConfidence
        u_pre and its components.
    post_confidence : PostGenerationConfidence or None
        û and its components -- None if Tier 1 or Tier 3.
    stored_confidence : StoredConfidence or None
        û_stored from Stage 5 verification -- None if verification was skipped.
    entry_id : int or None
        FAISS entry ID if the answer was stored; None otherwise.
    escalated : bool
        True if Tier 2 û was below threshold and escalated to Tier 3.
    """

    query: str
    answer: str
    tier: int
    stored: bool = False
    latency_ms: float = 0.0
    routing_decision: Optional[RoutingDecision] = None
    pre_confidence: Optional[PreRoutingConfidence] = None
    post_confidence: Optional[PostGenerationConfidence] = None
    stored_confidence: Optional[StoredConfidence] = None
    entry_id: Optional[int] = None
    escalated: bool = False


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

class CAEMPipeline:
    """End-to-end CAEM inference pipeline.

    Parameters
    ----------
    model : transformers.T5ForConditionalGeneration
        Flan-T5-Large in eval mode, shared across all stages.
    tokenizer : transformers.AutoTokenizer
        Matching tokenizer.
    encoder : QueryEncoder
        Sentence-BERT encoder (all-mpnet-base-v2, 768-dim).
        Shared by EpisodicMemoryStore, PostGenerationConf., and Verifier.
    nli_model : optional
        RoBERTa-Large-MNLI. If None, NLI-dependent signals fall back to
        neutral values (p_entail=0.5, u_entropy=0.5).
    nli_tokenizer : optional
        Tokenizer for nli_model.
    passage_store : PassageStore
        Pre-built Wikipedia FAISS passage index for Tier 3 RAG.
    config : CAEMConfig or None
        All thresholds and weights. Defaults to CAEMConfig() if None.
    memory_store : EpisodicMemoryStore or None
        Provide an existing store to continue from a checkpoint;
        if None, a fresh empty store is created.
    device : str or None
        'cuda' or 'cpu'. Auto-detected from model.parameters() if None.
    current_cycle : int
        Current self-improvement cycle number (used in EpisodicEntry metadata).
        Increment externally after each SelfImprovementLoop.run_cycle() call.

    Usage
    -----
    >>> pipeline = CAEMPipeline(model, tokenizer, encoder,
    ...                         nli_model, nli_tokenizer, passage_store)
    >>> result = pipeline.answer("Who wrote Hamlet?")
    """

    def __init__(
        self,
        model,
        tokenizer,
        encoder,
        nli_model=None,
        nli_tokenizer=None,
        passage_store: Optional[PassageStore] = None,
        config: Optional[CAEMConfig] = None,
        memory_store: Optional[EpisodicMemoryStore] = None,
        device: Optional[str] = None,
        current_cycle: int = 0,
    ) -> None:
        self.config = config or CAEMConfig()
        self.current_cycle = current_cycle

        # Shared model/tokenizer/encoder
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder

        # Device detection (same logic used by all sub-components)
        if device is None:
            device = str(next(model.parameters()).device)
        self.device = device

        # -- Stage 1: Episodic Memory ------------------------------------ #
        self.memory_store = memory_store or EpisodicMemoryStore(self.config)

        # -- Stage 3a: Pre-routing confidence --------------------------- #
        self.pre_estimator = PreRoutingConfidenceEstimator(
            model=model,
            tokenizer=tokenizer,
            config=self.config,
            device=device,
        )

        # -- Stage 3b: Adaptive Router ----------------------------------- #
        self.router = AdaptiveRouter(config=self.config)

        # -- Stage 4a: Post-generation confidence (Tier 2 only) --------- #
        self.post_estimator = PostGenerationConfidenceEstimator(
            model=model,
            tokenizer=tokenizer,
            sbert_encoder=encoder,
            nli_model=nli_model,
            nli_tokenizer=nli_tokenizer,
            config=self.config,
            device=device,
        )

        # -- Stage 5: Multi-layer verifier ------------------------------ #
        self.verifier = MultiLayerVerifier(
            model=model,
            tokenizer=tokenizer,
            sbert_encoder=encoder,
            nli_model=nli_model,
            nli_tokenizer=nli_tokenizer,
            config=self.config,
            device=device,
        )

        # -- Stage 6: Tier 3 RAG ---------------------------------------- #
        if passage_store is None:
            # Degenerate empty store -- Tier 3 will fall back to query-only gen.
            logger.warning(
                "CAEMPipeline: no passage_store provided -- Tier 3 will use "
                "query-only generation (no Wikipedia context). Provide a "
                "PassageStore for full RAG functionality."
            )
            passage_store = PassageStore([], np.empty((0, 768), dtype=np.float32))
        self.rag = TierThreeRAG(
            model=model,
            tokenizer=tokenizer,
            passage_encoder=encoder,
            passage_store=passage_store,
            config=self.config,
            device=device,
        )

        logger.info(
            "CAEMPipeline initialised | device=%s | cycle=%d | memory=%d episodes",
            self.device, self.current_cycle, self.memory_store.size,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def answer(self, query: str, store_to_memory: bool = True) -> PipelineResult:
        """Run the full CAEM pipeline for a single query.

        Parameters
        ----------
        query : str
            Natural-language question.

        Returns
        -------
        PipelineResult
            answer, tier, stored, latency, and all intermediate signals.
        """
        t_start = time.perf_counter()

        # -- Stage 2: Encode query --------------------------------------- #
        query_embedding = self._encode_query(query)

        # -- Stage 3a: Pre-routing confidence --------------------------- #
        pre_conf = self.pre_estimator.estimate(query)

        # -- Stage 1: Memory search (k=1 for routing) ------------------- #
        # Use search_with_ids so Tier-1 stats updates can call back without
        # scanning private _metadata for the entry ID.
        search_with_ids = self.memory_store.search_with_ids(query_embedding, k=1)

        # -- Stage 3b: Route --------------------------------------------- #
        routing = self.router.route(pre_conf, search_with_ids)

        logger.debug(
            "Routing: Tier %d | u_pre=%.4f | sim=%.4f | score=%.4f | safety=%s",
            routing.tier, routing.u_pre, routing.similarity,
            routing.routing_score, routing.safety_override,
        )

        # -- Tier dispatch ---------------------------------------------- #
        answer_str: str
        post_conf: Optional[PostGenerationConfidence] = None
        stored_conf: Optional[StoredConfidence] = None
        escalated: bool = False
        entry_id: Optional[int] = None
        stored_flag = False

        if routing.tier == 1:
            # -- Tier 1: fast path -- NO Stage-5 verification ----------- #
            # Rationale: Tier 1 is the <400 ms fast path. Running Stage 5
            # (MultiLayerVerifier) would require generating M=3 chains -- this
            # violates the "no generation" principle and inflates measured Tier-1
            # latency, making it indistinguishable from Tier 2 in experiments.
            # The stored entry already holds a verified u_stored from its
            # original storage cycle; we reconstruct StoredConfidence from those
            # scores and update retrieval stats.
            answer_str, stored_conf = self._tier1(search_with_ids)
            self._update_tier1_stats(search_with_ids=search_with_ids,
                                     stored_conf=stored_conf)

        elif routing.tier == 2:
            answer_str, post_conf, escalated = self._tier2(query, pre_conf)
            # -- Stage 5: Verify (Tier 2 and escalated-to-3 answers) --- #
            stored_conf = self._verify(query, answer_str)
            if store_to_memory:
                entry_id, stored_flag = self._maybe_store(
                    query=query, answer=answer_str,
                    query_embedding=query_embedding, stored_conf=stored_conf,
                )

        else:  # tier == 3
            answer_str = self._tier3(query)
            # -- Stage 5: Verify ----------------------------------------#
            stored_conf = self._verify(query, answer_str)
            if store_to_memory:
                entry_id, stored_flag = self._maybe_store(
                    query=query, answer=answer_str,
                    query_embedding=query_embedding, stored_conf=stored_conf,
                )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Pipeline: Tier %d | stored=%s | u_stored=%.3f | latency=%.1f ms",
            routing.tier, stored_flag,
            stored_conf.u_stored if stored_conf else 0.0,
            latency_ms,
        )

        return PipelineResult(
            query=query,
            answer=answer_str,
            tier=routing.tier,
            stored=stored_flag,
            latency_ms=latency_ms,
            routing_decision=routing,
            pre_confidence=pre_conf,
            post_confidence=post_conf,
            stored_confidence=stored_conf,
            entry_id=entry_id,
            escalated=escalated,
        )

    # ------------------------------------------------------------------ #
    # Tier handlers                                                         #
    # ------------------------------------------------------------------ #

    def _tier1(self, search_with_ids):
        """Return the stored answer for a Tier 1 hit.

        No model generation occurs. The reasoning chain and answer are read
        directly from the matched EpisodicEntry. Stage 5 (MultiLayerVerifier)
        is deliberately skipped -- see answer() for the full rationale.

        A StoredConfidence is reconstructed from the entry's stored scores so
        that PipelineResult.stored_confidence is always populated.

        Returns
        -------
        (answer_str, stored_conf)
        """
        entry, entry_id, similarity = search_with_ids[0]
        logger.debug(
            "Tier 1 hit: id=%d | sim=%.4f | answer='%s...'",
            entry_id, similarity, entry.answer[:80],
        )
        # Reconstruct StoredConfidence from stored quality scores.
        stored_conf = StoredConfidence(
            p_entail=entry.nli_score,
            s_avg=entry.sc_score,
            h_norm=1.0 - entry.se_score,
            u_stored=entry.u_stored,
        )
        return entry.answer, stored_conf

    def _tier2(self, query: str, pre_conf: PreRoutingConfidence):
        """Generate a Tier 2 answer with Flan-T5 and compute û.

        If û < u_hat_accept_threshold, escalate to Tier 3.

        Returns
        -------
        (answer_str, post_conf, escalated)
        """
        cfg = self.config

        # Tokenise query (benchmark-aware for classification tasks).
        prompt = self._build_tier2_prompt(query)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(self.device)

        # Generate
        try:
            self.model.eval()
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.cot_max_new_tokens,
                    do_sample=False,
                )
            answer_str = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            ).strip()
        except Exception as exc:
            logger.error("Tier 2 generation failed: %s -- escalating to Tier 3.", exc)
            return self._tier3(query), None, True

        if not answer_str:
            logger.warning("Tier 2 produced empty answer -- escalating to Tier 3.")
            return self._tier3(query), None, True

        # Stage 4a: post-generation confidence
        try:
            post_conf = self.post_estimator.estimate(
                query=query,
                generated_answer=answer_str,
                input_ids=input_ids,
            )
        except Exception as exc:
            logger.warning("PostGenerationConfidenceEstimator failed: %s", exc)
            # Treat as low confidence -> escalate
            return self._tier3(query), None, True

        # Efficiency gate: escalate if û below threshold
        if not post_conf.should_accept(cfg.u_hat_accept_threshold):
            logger.debug(
                "Tier 2 û=%.4f < %.2f -> escalating to Tier 3.",
                post_conf.u_hat, cfg.u_hat_accept_threshold,
            )
            escalated_answer = self._tier3(query)
            return escalated_answer, post_conf, True

        logger.debug("Tier 2 accepted: û=%.4f ≥ %.2f", post_conf.u_hat, cfg.u_hat_accept_threshold)
        return answer_str, post_conf, False

    def _tier3(self, query: str) -> str:
        """Generate a Tier 3 answer via RAG.

        Returns
        -------
        str -- may be empty string on complete failure.
        """
        answer = self.rag.generate(query)
        logger.debug("Tier 3 RAG answer: '%s...'", answer[:80])
        return answer

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

    def _build_tier2_prompt(self, query: str) -> str:
        """Build Tier 2 generation prompt with task-specific CoT formatting."""
        task = self._detect_query_task(query)

        if task == "fever":
            claim = self._extract_after_token(query, "Claim:")
            return (
                "Determine whether the claim is supports, refutes, or not enough info.\n"
                "Provide brief reasoning, then the final label.\n"
                "Format:\n"
                "Reasoning: <short explanation>\n"
                "Answer: supports|refutes|not enough info\n"
                f"Claim: {claim}"
            )

        if task == "strategyqa":
            q_text = self._extract_after_token(query, "Question:")
            return (
                "Answer the question with brief reasoning and a final yes/no label.\n"
                "Format:\n"
                "Reasoning: <short explanation>\n"
                "Answer: yes|no\n"
                f"Question: {q_text}"
            )

        return f"Question: {query}\nThink step by step:"

    # ------------------------------------------------------------------ #
    # Verification (Stage 5)                                               #
    # ------------------------------------------------------------------ #

    def _verify(self, query: str, answer: str) -> Optional[StoredConfidence]:
        """Run MultiLayerVerifier to compute û_stored.

        Returns None on total failure (answer is not stored in that case).
        """
        if not answer:
            return None
        try:
            return self.verifier.verify(query, answer)
        except Exception as exc:
            logger.error("Verification failed: %s -- answer will not be stored.", exc)
            return None

    # ------------------------------------------------------------------ #
    # Storage decision (Stage 7)                                           #
    # ------------------------------------------------------------------ #

    def _maybe_store(
        self,
        query: str,
        answer: str,
        query_embedding: np.ndarray,
        stored_conf: Optional[StoredConfidence],
    ):
        """Store the answer in episodic memory if it passes all gates.

        Gates
        -----
        1. stored_conf is not None (verification succeeded).
        2. û_stored ≥ retroverify_prune_threshold (answer is good enough).
        3. The query embedding is novel (no near-duplicate already stored).
        4. Memory is not full (if near-full, prune first).

        Returns
        -------
        (entry_id, stored_flag) : (int or None, bool)
        """
        cfg = self.config

        if stored_conf is None:
            return None, False

        if stored_conf.u_stored < cfg.retroverify_prune_threshold:
            logger.debug(
                "Not storing: û_stored=%.4f < threshold=%.2f",
                stored_conf.u_stored, cfg.retroverify_prune_threshold,
            )
            return None, False

        if not self.memory_store.is_novel(query_embedding):
            logger.debug("Not storing: near-duplicate already in memory.")
            return None, False

        # Auto-prune if approaching capacity
        if self.memory_store._should_prune():
            logger.info("Memory store near capacity -- pruning before storing.")
            self.memory_store.prune()

        entry = EpisodicEntry(
            question=query,
            reasoning_chain=answer,   # Flan-T5 output serves as the chain
            answer=answer,
            embedding=query_embedding,
            storage_cycle=self.current_cycle,
            u_stored=stored_conf.u_stored,
            nli_score=stored_conf.p_entail,
            sc_score=stored_conf.s_avg,
            se_score=1.0 - stored_conf.h_norm,
        )

        try:
            entry_id = self.memory_store.add(entry)
            logger.info(
                "Stored new episode: id=%d | û_stored=%.4f | q='%s...'",
                entry_id, stored_conf.u_stored, query[:60],
            )
            return entry_id, True
        except RuntimeError as exc:
            logger.error("Failed to store episode: %s", exc)
            return None, False

    def _update_tier1_stats(
        self,
        search_with_ids,
        stored_conf: Optional[StoredConfidence],
    ) -> None:
        """Update retrieval stats for a Tier 1 hit.

        The episode is not re-stored. retrieval_count and success_rate are
        updated via the public API. entry_id comes directly from
        search_with_ids -- no private _metadata scan required.

        Acceptance is determined by whether the stored u_stored meets the
        quality threshold (it always should for a Tier 1 hit, but we check
        defensively). No re-verification is run.
        """
        if not search_with_ids:
            return

        entry, entry_id, _ = search_with_ids[0]

        # Acceptance: the stored entry already has a verified u_stored.
        # A Tier 1 hit that reaches here passed both the routing score and
        # the OR-condition safety gate, so the entry is almost always accepted.
        accepted = entry.u_stored >= self.config.retroverify_prune_threshold
        self.memory_store.update_retrieval_stats(entry_id, was_accepted=accepted)
        logger.debug(
            "Tier 1 stats: entry %d | accepted=%s | u_stored=%.4f",
            entry_id, accepted, entry.u_stored,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query to L2-normalised 768-dim float32 embedding."""
        emb = self.encoder.encode(query).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def memory_summary(self) -> dict:
        """Return memory store summary statistics."""
        return self.memory_store.summary()

    def __repr__(self) -> str:
        return (
            f"CAEMPipeline("
            f"device={self.device}, "
            f"cycle={self.current_cycle}, "
            f"memory={self.memory_store.size} episodes)"
        )
