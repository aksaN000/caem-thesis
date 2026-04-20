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
  Stage 5 -- UnifiedVerifier         û_stored (quality gate, all tiers)
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
from typing import Any, List, Optional

import numpy as np
import torch

from caem.config import CAEMConfig
from caem.confidence.pre_routing import PreRoutingConfidenceEstimator
from caem.memory.entry import (
    EpisodicEntry,
    PostGenerationConfidence,
    PreRoutingConfidence,
    RoutingDecision,
)
from caem.memory.deferred import DeferredBuffer
from caem.memory.store import EpisodicMemoryStore
from caem.retrieval.rag import PassageStore, TierThreeRAG
from caem.routing.router import AdaptiveRouter
from caem.verification.verifier import UnifiedVerifier, UnifiedVerifierOutput

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
    verifier_output : UnifiedVerifierOutput or None
        Full Stage-5 verifier record: all nine signals (u_token, u_dropout,
        u_internal, s_avg, h_norm, p_entail, p_ground_max, p_ground_mean,
        p_ground_atomic), p_contra, the composite u_stored, the decision
        string (STORE/DEFERRED/ABSTAIN/DISCARD), and the early-exit flag.
        ``None`` for Tier 1 hits (the verifier is deliberately skipped) and
        for Tier 2/3 failures. Downstream code (harness, metric suite)
        reads ``u_stored`` / ``decision`` / per-signal fields directly.
    u_stored : float or None
        Convenience scalar. For Tier 1 hits this is the retrieved entry's
        stored composite (the verifier was not re-run). For Tier 2/3 it is
        ``verifier_output.u_stored`` when verification succeeded, else None.
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
    verifier_output: Optional[UnifiedVerifierOutput] = None
    u_stored: Optional[float] = None
    entry_id: Optional[int] = None
    escalated: bool = False
    # Display layer per Ch4 Table tab:decision-tree Stage 7 actions.
    # Separate from `answer` so scoring consumes raw predictions (preserving
    # EM/F1 measurement on ABSTAIN/DISCARD cases) while a deployment frontend
    # can use this field for user-facing output. STORE/DEFERRED pass through
    # the raw answer; ABSTAIN returns a principled refusal per Ch4 spec;
    # DISCARD suppresses output. The decision field on verifier_output
    # remains the source of truth for diagnostic aggregation.
    display_answer: str = ""


# -----------------------------------------------------------------------------
# Display-layer mapping
# -----------------------------------------------------------------------------

def _compute_display_answer(answer_str: str, vout) -> str:
    """Map verifier decision to user-facing output per Ch4 Stage-7 spec.

    Separate from the raw ``answer`` field so downstream scoring (EM/F1)
    consumes the model's literal prediction, while a deployment frontend
    can read ``display_answer`` for the user-facing string. This matches
    Ch4 Table ``tab:decision-tree`` action column:

    - STORE / DEFERRED: pass raw answer through (the model is confident
      enough, or confident enough to hold for reconsideration).
    - ABSTAIN: return the principled refusal string. Ch4 §4.6 specifies
      an explicit ``I do not know`` response as the correct failure mode
      in factual-QA settings where a confident wrong answer is costlier
      than a refusal.
    - DISCARD: suppress output entirely (empty string). Per Ch4 spec,
      either the contradiction veto or the grounding floor fired; the
      model's prediction is not safe to expose.
    - Tier 1 hits / pipeline-failure edge cases: fall through to raw
      answer (the verifier was deliberately skipped for Tier 1, and
      pipeline failures are signalled upstream with stored=False).

    The ``decision`` field on ``vout`` remains the source of truth for
    diagnostic aggregation (decision_breakdown, hallucination_rate, etc.);
    this function only affects the user-facing display string.
    """
    if vout is None:
        return answer_str
    decision = getattr(vout, "decision", None)
    if decision == "ABSTAIN":
        return "I do not know."
    if decision == "DISCARD":
        return ""
    # STORE, DEFERRED, and any unrecognised decision pass through.
    return answer_str


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
        deferred_buffer: Optional[DeferredBuffer] = None,
        device: Optional[str] = None,
        current_cycle: int = 0,
        judge: Optional[Any] = None,
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

        # -- Stage 7b: Deferred-entry buffer ----------------------------- #
        # Bounded FIFO for Stage-5 DEFERRED episodes. Populated at Stage 7
        # when vout.decision == "DEFERRED"; drained at cycle boundary by
        # SelfImprovementLoop via DeferredBuffer.reconsider(). See
        # thesis Section 4.9 (Deferred-Entry Reconsideration).
        self.deferred_buffer = deferred_buffer or DeferredBuffer(self.config)

        # -- Stage 3a: Pre-routing confidence --------------------------- #
        self.pre_estimator = PreRoutingConfidenceEstimator(
            model=model,
            tokenizer=tokenizer,
            config=self.config,
            device=device,
        )

        # -- Stage 3b: Adaptive Router ----------------------------------- #
        self.router = AdaptiveRouter(config=self.config)

        # -- Stage 6 prep: resolve PassageStore fallback BEFORE Stage 5 --- #
        # The UnifiedVerifier grounding path AND Tier 3 generation both
        # depend on the PassageStore, so the empty-store fallback must be
        # resolved here -- before the verifier is constructed -- so the
        # same store is threaded into both components.
        if passage_store is None:
            # Degenerate empty store -- Tier 3 will fall back to query-only
            # generation AND UnifiedVerifier grounding signals
            # (p_ground_max / p_ground_mean / p_ground_atomic / p_contra)
            # will return neutral defaults from the empty-passage branch.
            # The ABSTAIN decision class and the contradiction hard veto
            # cannot fire in this configuration -- provide a real
            # PassageStore for full nine-signal verification.
            logger.warning(
                "CAEMPipeline: no passage_store provided -- Tier 3 will use "
                "query-only generation (no Wikipedia context) AND verifier "
                "grounding signals (p_ground_max, p_ground_mean, "
                "p_ground_atomic, p_contra) will return neutral defaults. "
                "Provide a PassageStore for full RAG + grounding functionality."
            )
            passage_store = PassageStore([], np.empty((0, 768), dtype=np.float32))

        # Adapter closure bridging PassageStore.search (embedding-in,
        # (passage, score)-out) to the verifier's passage_retriever
        # contract ((query_str, k) -> List[str]). Empty PassageStore
        # returns [], which the verifier handles via its empty-passage
        # fallback at verifier.py:_retrieve_and_rerank. Mirrors the
        # encode-and-L2-normalise pattern used by
        # CAEMPipeline._encode_query (pipeline.py:_encode_query) and
        # TierThreeRAG.retrieve (rag.py) so grounding and Tier 3 RAG
        # hit the same index with the same query embedding.
        def _verifier_passage_retriever(query: str, k: int) -> List[str]:
            emb = encoder.encode(query).astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            results = passage_store.search(emb, k=k)
            return [p for p, _ in results]

        # -- Stage 5: UnifiedVerifier (post-generation quality gate) --- #
        # The verifier emits UnifiedVerifierOutput carrying all nine signals
        # (u_token, u_dropout, u_internal, s_avg, h_norm, p_entail,
        # p_ground_max, p_ground_mean, p_ground_atomic) plus the
        # STORE/DEFERRED/ABSTAIN/DISCARD decision and the scalar u_stored.
        # Passage retrieval for grounding signals is threaded through the
        # adapter closure above, which wraps the same PassageStore used by
        # Tier 3 RAG -- keeping the retriever path consistent across Tier 3
        # generation and Stage 5 grounding.
        self.verifier = UnifiedVerifier(
            model=model,
            tokenizer=tokenizer,
            sbert_encoder=encoder,
            judge=judge,
            nli_model=nli_model,
            nli_tokenizer=nli_tokenizer,
            passage_retriever=_verifier_passage_retriever,
            config=self.config,
            device=device,
        )

        # -- Stage 6: Tier 3 RAG ---------------------------------------- #
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

    def answer(
        self,
        query: str,
        store_to_memory: bool = True,
        source_benchmark: Optional[str] = None,
        _precomputed_tier2_answer: Optional[str] = None,
        _precomputed_tier3_answer: Optional[str] = None,
        _precomputed_vout: Optional[UnifiedVerifierOutput] = None,
        _precomputed_routing: Optional[tuple] = None,
    ) -> PipelineResult:
        """Run the full CAEM pipeline for a single query.

        Parameters
        ----------
        query : str
            Natural-language question.
        store_to_memory : bool, default True
            If False, skip the Stage-7 storage decision entirely.
        source_benchmark : str or None, default None
            Benchmark tag (e.g. "natural_questions", "truthfulqa") carried
            onto any stored EpisodicEntry / deferred entry so the Stage-8
            training-pool ID/OOD gate can be enforced downstream. Pass the
            current benchmark identifier at harness call-time; ``None`` is
            accepted as a legacy / single-benchmark default and will be
            treated as training-eligible by ``_collect_episodes``.

        Returns
        -------
        PipelineResult
            answer, tier, stored, latency, and all intermediate signals.
        """
        t_start = time.perf_counter()

        if _precomputed_routing is not None:
            # Level B Phase 2 Tier-1 fast path: BatchPipeline already ran
            # Stages 1-3 in _peek_routing when it classified tiers; thread
            # those results through here to avoid re-encoding + re-searching
            # + re-routing per sample. Saves ~30-50 ms per Tier-1 sample at
            # late cycles where Tier-1 fraction is high (38% at Cycle 10).
            query_embedding, pre_conf, search_with_ids, routing = \
                _precomputed_routing
        else:
            # -- Stage 2: Encode query --------------------------------------- #
            query_embedding = self._encode_query(query)

            # -- Stage 3a: Pre-routing confidence --------------------------- #
            pre_conf = self.pre_estimator.estimate(query)

            # -- Stage 1: Memory search (k=1 for routing) ------------------- #
            # Use search_with_ids so Tier-1 stats updates can call back without
            # scanning private _metadata for the entry ID.
            search_with_ids = self.memory_store.search_with_ids(
                query_embedding, k=1,
            )

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
        vout: Optional[UnifiedVerifierOutput] = None
        u_stored_scalar: Optional[float] = None
        escalated: bool = False
        entry_id: Optional[int] = None
        stored_flag = False

        if routing.tier == 1:
            # -- Tier 1: fast path -- NO Stage-5 verification ----------- #
            # Rationale: Tier 1 is the <400 ms fast path. Running Stage 5
            # (UnifiedVerifier) would require generating M=3 chains -- this
            # violates the "no generation" principle and inflates measured
            # Tier-1 latency, making it indistinguishable from Tier 2 in
            # experiments. The stored entry already holds a verified
            # u_stored from its original storage cycle; we surface that
            # scalar on PipelineResult and update retrieval stats.
            answer_str, u_stored_scalar = self._tier1(search_with_ids)
            self._update_tier1_stats(search_with_ids=search_with_ids)

        elif routing.tier == 2:
            if _precomputed_tier2_answer is not None:
                # Level B batched path: Tier 2 generate was run in a single
                # batched T5 forward pass by BatchPipeline.batch_tier2_generate;
                # inject the precomputed string here and skip the individual
                # _tier2 call. An empty string signals Tier 2 failed in the
                # batched generate and we escalate to Tier 3 -- matches the
                # serial _tier2 escalation contract.
                answer_str = _precomputed_tier2_answer.strip()
                post_conf = None
                escalated = False
                if not answer_str:
                    logger.warning(
                        "Tier 2 precomputed answer empty -- escalating to Tier 3.",
                    )
                    answer_str = self._tier3(query)
                    escalated = True
            else:
                answer_str, post_conf, escalated = self._tier2(query, pre_conf)
            # -- Stage 5: UnifiedVerifier (nine signals + decision) ---- #
            if _precomputed_vout is not None:
                # Level B batched path: verify_batch was called upstream;
                # inject the precomputed output and skip the internal call.
                vout = _precomputed_vout
            else:
                # Pass through the already-computed u_token and u_dropout so
                # the verifier does not pay for a duplicate forward pass.
                u_tok = getattr(post_conf, "u_token", None) if post_conf else None
                u_drop = getattr(post_conf, "u_dropout", None) if post_conf else None
                vout = self._verify(query, answer_str, u_token=u_tok, u_dropout=u_drop)
            u_stored_scalar = vout.u_stored if vout else None
            if store_to_memory:
                entry_id, stored_flag = self._maybe_store(
                    query=query, answer=answer_str,
                    query_embedding=query_embedding, vout=vout,
                    source_benchmark=source_benchmark,
                )

        else:  # tier == 3
            if _precomputed_tier3_answer is not None:
                # Level B batched path: Tier 3 RAG generate was run in a single
                # batched T5 forward pass by BatchPipeline.batch_tier3_generate;
                # inject the precomputed string here and skip the individual
                # _tier3 call. An empty string is preserved as-is (same as
                # serial _tier3 returning "" on total failure).
                answer_str = _precomputed_tier3_answer
            else:
                answer_str = self._tier3(query)
            # -- Stage 5: UnifiedVerifier ------------------------------ #
            if _precomputed_vout is not None:
                # Level B batched path: verify_batch was called upstream;
                # inject the precomputed output and skip the internal call.
                vout = _precomputed_vout
            else:
                # Tier 3 has no pre-computed internal signals, so the verifier
                # computes u_token and u_dropout itself.
                vout = self._verify(query, answer_str)
            u_stored_scalar = vout.u_stored if vout else None
            if store_to_memory:
                entry_id, stored_flag = self._maybe_store(
                    query=query, answer=answer_str,
                    query_embedding=query_embedding, vout=vout,
                    source_benchmark=source_benchmark,
                )

        # -- Early-exit confabulation gate ------------------------------ #
        # UnifiedVerifier sets ``early_exit_triggered=True`` when
        # u_internal >= 0.70 AND p_ground_max <= 0.20 -- i.e. confidently
        # asserted without grounding. The verifier has already mapped that
        # state to decision=DISCARD internally, so no tier re-route is
        # needed here; we only log the flag so it surfaces in harness logs
        # and, downstream, in Table 5.5's confabulation-rate columns.
        if vout is not None and vout.early_exit_triggered:
            logger.info(
                "Stage 5 early-exit confabulation gate fired "
                "(u_internal=%.3f, p_ground_max=%.3f) -- decision=%s",
                vout.u_internal, vout.p_ground_max, vout.decision,
            )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Pipeline: Tier %d | stored=%s | u_stored=%.3f | decision=%s | latency=%.1f ms",
            routing.tier, stored_flag,
            u_stored_scalar if u_stored_scalar is not None else 0.0,
            vout.decision if vout else ("TIER1_HIT" if routing.tier == 1 else "SKIPPED"),
            latency_ms,
        )

        display_answer = _compute_display_answer(answer_str, vout)

        return PipelineResult(
            query=query,
            answer=answer_str,
            tier=routing.tier,
            stored=stored_flag,
            latency_ms=latency_ms,
            routing_decision=routing,
            pre_confidence=pre_conf,
            post_confidence=post_conf,
            verifier_output=vout,
            u_stored=u_stored_scalar,
            entry_id=entry_id,
            escalated=escalated,
            display_answer=display_answer,
        )

    # ------------------------------------------------------------------ #
    # Tier handlers                                                         #
    # ------------------------------------------------------------------ #

    def _tier1(self, search_with_ids):
        """Return the stored answer for a Tier 1 hit.

        No model generation occurs. The reasoning chain and answer are read
        directly from the matched EpisodicEntry. Stage 5 (UnifiedVerifier)
        is deliberately skipped -- see ``answer()`` for the full rationale.

        Returns
        -------
        (answer_str, u_stored)
            ``u_stored`` is the entry's stored composite scalar, surfaced
            directly onto ``PipelineResult.u_stored`` so downstream code
            (eval harness, retrieval feedback) has a single field to read.
        """
        entry, entry_id, similarity = search_with_ids[0]
        logger.debug(
            "Tier 1 hit: id=%d | sim=%.4f | answer='%s...'",
            entry_id, similarity, entry.answer[:80],
        )
        return entry.answer, entry.u_stored

    def _tier2(self, query: str, pre_conf: PreRoutingConfidence):
        """Generate a Tier 2 answer with Flan-T5.

        Post-generation quality gating is handled entirely by UnifiedVerifier
        (Stage 5); this method returns (answer_str, None, escalated).

        Returns
        -------
        (answer_str, post_conf, escalated)
        """
        prompt = self._build_tier2_prompt(query)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(self.device)

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

        return answer_str, None, False

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
        """Infer benchmark task style from constrained prompt prefixes.

        Returns one of ``"fever"``, ``"strategyqa"``, ``"arc"``, or
        ``"open"``. ARC-Challenge queries are recognised by the
        "Choices: (A) ... (B) ..." suffix that
        ``load_arc_challenge`` builds into the query string.
        """
        q = query.lower().strip()
        if q.startswith("answer with one of: supports, refutes, not enough info."):
            return "fever"
        if q.startswith("answer yes or no."):
            return "strategyqa"
        if ("choices:" in q) and ("multiple choice letter" in q):
            return "arc"
        return "open"

    def _build_tier2_prompt(self, query: str) -> str:
        """Build Tier 2 generation prompt with uniform scaffolded CoT.

        Tier 2 has no retrieved passages so the template has no
        Context / Evidence block; the Reasoning and Answer slots
        match Tier 3's uniform template so both tiers produce the
        same substantive claim format for the verifier to score.
        Forces >= 40 words of reasoning to convert classification-
        style benchmarks (FEVER / StrategyQA / ARC) from degenerate
        1-word label outputs into propositional claims scoreable by
        the 9-signal composite.
        """
        task = self._detect_query_task(query)

        if task == "fever":
            claim = self._extract_after_token(query, "Claim:")
            task_line = (
                f"Claim: {claim}\n"
                "Determine whether the claim is SUPPORTS, REFUTES, or "
                "NOT ENOUGH INFO based on your knowledge."
            )
            answer_format = "supports | refutes | not enough info"
        elif task == "strategyqa":
            q_text = self._extract_after_token(query, "Question:")
            task_line = (
                f"Question: {q_text}\n"
                "Answer the question with yes or no."
            )
            answer_format = "yes | no"
        elif task == "arc":
            task_line = (
                f"{query}\n"
                "Choose the correct answer from the listed choices."
            )
            answer_format = "A | B | C | D"
        else:
            task_line = (
                f"Question: {query}\n"
                "Answer the question."
            )
            answer_format = "<concise factual answer>"

        return (
            f"{task_line}\n\n"
            "You MUST follow the exact response format below. Your "
            "reasoning must be at least 40 words, step by step.\n\n"
            "Reasoning: <at least 40 words of step-by-step analysis>\n"
            f"Answer: {answer_format}"
        )

    # ------------------------------------------------------------------ #
    # Verification (Stage 5)                                               #
    # ------------------------------------------------------------------ #

    def _verify(
        self,
        query: str,
        answer: str,
        *,
        u_token: Optional[float] = None,
        u_dropout: Optional[float] = None,
    ) -> Optional[UnifiedVerifierOutput]:
        """Run UnifiedVerifier and return the full nine-signal output.

        Parameters
        ----------
        query : str
            The user query.
        answer : str
            The candidate answer being verified. Returns None if empty --
            an empty answer cannot be stored.
        u_token, u_dropout : float or None
            Pre-computed internal signals from the generation stage. When
            supplied they let the verifier skip a duplicate forward pass;
            when None the verifier recomputes them. Passing them in is the
            efficient default for Tier 2 / Tier 3 generation paths.

        Returns
        -------
        UnifiedVerifierOutput or None
            ``None`` on verification failure (the answer then cannot be
            stored). Otherwise the full nine-signal record; callers read
            ``u_stored`` / ``decision`` / ``early_exit_triggered`` and the
            per-signal fields directly.
        """
        if not answer:
            return None
        try:
            return self.verifier.verify(
                query, answer,
                u_token=u_token, u_dropout=u_dropout,
            )
        except Exception as exc:
            logger.error("Verification failed: %s -- answer will not be stored.", exc)
            return None

    # ------------------------------------------------------------------ #
    # Retroactive re-verification helper (Phase 5d)                        #
    # ------------------------------------------------------------------ #

    def make_retroverify_fn(self):
        """Return a closure ``(entry) -> UnifiedVerifierOutput`` for retroverify.

        Binds the pipeline's current ``UnifiedVerifier`` (which already owns the
        passage store via ``_retrieve_and_rerank``) so that the self-improvement
        loop can call ``memory_store.retroverify(verify_fn)`` without needing
        any knowledge of retrieval or verifier wiring.

        The closure captures ``self.verifier`` by reference, so if the caller
        swaps models mid-cycle (unusual) the next retroverify pass sees the
        new weights automatically.

        Returns None on verification failure so ``retroverify`` skips the entry
        cleanly rather than raising.
        """
        verifier = self.verifier

        def _retroverify(entry) -> Optional[UnifiedVerifierOutput]:
            try:
                return verifier.verify(entry.question, entry.answer)
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning(
                    "Retroverify of entry (q=%r) raised %s -- entry left unchanged.",
                    entry.question[:60], exc,
                )
                return None

        return _retroverify

    def make_reconsider_deferred_fn(self):
        """Return a closure ``(deferred_entry) -> UnifiedVerifierOutput``.

        Used by :meth:`DeferredBuffer.reconsider` at cycle boundaries. The
        contract is identical to :meth:`make_retroverify_fn` -- the verifier
        receives ``(question, answer)`` and returns the full nine-signal
        output -- so a single verifier instance serves both passes. Kept as
        a separate closure for symmetry and so a future asymmetric signal
        policy (e.g. cheaper verification on deferred entries) can diverge
        without touching retroverify.
        """
        verifier = self.verifier

        def _reconsider(deferred_entry) -> Optional[UnifiedVerifierOutput]:
            try:
                return verifier.verify(
                    deferred_entry.question,
                    deferred_entry.answer,
                )
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning(
                    "Deferred reconsideration of (q=%r) raised %s -- entry kept.",
                    deferred_entry.question[:60], exc,
                )
                return None

        return _reconsider

    # ------------------------------------------------------------------ #
    # Storage decision (Stage 7)                                           #
    # ------------------------------------------------------------------ #

    def _maybe_store(
        self,
        query: str,
        answer: str,
        query_embedding: np.ndarray,
        vout: Optional[UnifiedVerifierOutput],
        source_benchmark: Optional[str] = None,
    ):
        """Store the answer in episodic memory if it passes all gates.

        Gates
        -----
        1. ``vout`` is not None (verification succeeded).
        2. ``vout.decision == "STORE"`` (the Stage-5 decision tree
           green-lit the episode; ABSTAIN/DISCARD skip storage entirely).
           ``DEFERRED`` is routed to ``self.deferred_buffer`` instead,
           where it awaits cycle-boundary reconsideration under the
           fine-tuned verifier (Section 4.9 of the thesis, Stage 7b).
        3. The query embedding is novel (no near-duplicate already stored).
        4. Memory is not full (if near-full, prune first).

        Returns
        -------
        (entry_id, stored_flag) : (int or None, bool)
            ``stored_flag`` is True only for the STORE path (main-memory
            write). DEFERRED-buffer pushes return ``(None, False)`` because
            the entry is not yet committed to memory; the PipelineResult's
            ``stored`` flag and the downstream counters already treat
            "held for reconsideration" as a non-store event.
        """
        if vout is None:
            return None, False

        # DEFERRED -> buffer it (do not write to main memory yet).
        if vout.decision == "DEFERRED":
            try:
                self.deferred_buffer.push(
                    question=query,
                    answer=answer,
                    embedding=query_embedding,
                    storage_cycle=self.current_cycle,
                    vout=vout,
                    source_benchmark=source_benchmark,
                )
            except Exception as exc:
                # Buffer push is non-critical -- log and continue as if the
                # entry had been DISCARDed rather than failing the whole
                # pipeline call.
                logger.warning(
                    "Deferred-buffer push failed (%s); entry dropped.", exc,
                )
            else:
                logger.debug(
                    "Deferred: u_stored=%.4f | p_ground_max=%.3f | buffer=%d",
                    vout.u_stored, vout.p_ground_max, self.deferred_buffer.size,
                )
            return None, False

        if vout.decision != "STORE":
            logger.debug(
                "Not storing: decision=%s | u_stored=%.4f | p_ground_max=%.3f",
                vout.decision, vout.u_stored, vout.p_ground_max,
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
            source_benchmark=source_benchmark,
            # Composite + nine signals + p_contra
            u_stored=vout.u_stored,
            u_token=vout.u_token,
            u_dropout=vout.u_dropout,
            u_internal=vout.u_internal,
            s_avg=vout.s_avg,
            h_norm=vout.h_norm,
            p_entail=vout.p_entail,
            p_ground_max=vout.p_ground_max,
            p_ground_mean=vout.p_ground_mean,
            p_ground_atomic=vout.p_ground_atomic,
            p_contra=vout.p_contra,
            decision=vout.decision,
            early_exit_triggered=vout.early_exit_triggered,
        )

        try:
            entry_id = self.memory_store.add(entry)
            logger.info(
                "Stored new episode: id=%d | u_stored=%.4f | q='%s...'",
                entry_id, vout.u_stored, query[:60],
            )
            return entry_id, True
        except RuntimeError as exc:
            logger.error("Failed to store episode: %s", exc)
            return None, False

    def _update_tier1_stats(self, search_with_ids) -> None:
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
        """Encode query to L2-normalised 768-dim float32 embedding.

        Shared by Stage-1 memory search and by the verifier's
        passage-retriever adapter closure (see ``__init__``), keeping the
        query embedding path identical across memory lookup, Stage-5
        grounding, and Tier-3 RAG.
        """
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
