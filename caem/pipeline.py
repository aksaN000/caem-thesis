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
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional

# Suppress cosmetic transformers warnings about greedy+sampling-param mismatch.
# Paired with the same filter in caem/verification/verifier.py. See that
# module's import block for full rationale. Greedy decoding is intentional
# (reproducibility); the inherited sampling params from Qwen's factory
# generation_config have zero functional effect in greedy mode.
_warnings.filterwarnings(
    "ignore",
    message=r".*`do_sample` is set to `False`.*",
    category=UserWarning,
)

import numpy as np
import torch

from caem.config import CAEMConfig
from caem.confidence.pre_routing import PreRoutingConfidenceEstimator
from caem.prompts import build_tier2_prompt
from caem.memory.entry import (
    EpisodicEntry,
    PreRoutingConfidence,
    RoutingDecision,
)
from caem.memory.deferred import DeferredBuffer
from caem.memory.store import EpisodicMemoryStore
from caem.retrieval.rag import PassageStore, TierThreeRAG
from caem.routing.router import AdaptiveRouter
from caem.verification.verifier import UnifiedVerifier, UnifiedVerifierOutput
from caem._profile import section

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
    post_confidence : None
        Legacy field kept for schema back-compat. The Session-42 redesign
        retired the 4-signal u_hat gate that once populated this; every
        pipeline call now returns ``None`` here and the Stage-5 verifier
        in ``verifier_output`` is the single source of post-generation
        truth. Downstream readers should consume ``verifier_output``.
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
    post_confidence: Optional[None] = None
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
        return _strip_scaffold(answer_str)
    decision = getattr(vout, "decision", None)
    if decision == "ABSTAIN":
        return "I do not know."
    if decision == "DISCARD":
        return ""
    # STORE, DEFERRED, and any unrecognised decision: strip the
    # forced "Reasoning: ... Answer: Y" scaffold so users see only
    # the final answer Y. Preserves the raw answer on PipelineResult
    # for scoring; display_answer is the user-facing view.
    return _strip_scaffold(answer_str)


_TEMPLATE_LEAK_PATTERNS = [
    r"<(?:concise|factual|your|scaffold|answer)[a-z_ ]*>",
    r"\[(?:your\s+answer|answer\s+here)\]",
    r"<(?:supports|refutes|yes|no)/",
]
_EVASIVE_PATTERNS = [
    r"the\s+context\s+(?:does\s*not|doesn'?t)\s+(?:mention|provide|contain|discuss)",
    r"no\s+information\s+(?:is\s+)?(?:provided|given|available|mentioned)",
    r"information\s+(?:is\s+)?not\s+provided",
    r"cannot\s+(?:determine|verify|be\s+determined)",
    r"(?:none\s+of\s+the\s+given|none\s+of\s+the\s+listed)\s+(?:options|choices|models|items)",
]


def _answer_is_unstorable(answer_str: str) -> bool:
    """E22/E23 sanitizer: reject answers containing template-leak placeholders
    or clearly-evasive patterns. Prevents cold-start memory from accumulating
    ``<concise factual answer>`` literals and ``"the context doesn't mention"``
    responses that add no informational value.

    Returns True if the answer should NOT be stored.
    """
    import re as _re
    if not answer_str:
        return True
    for pat in _TEMPLATE_LEAK_PATTERNS:
        if _re.search(pat, answer_str, _re.IGNORECASE):
            return True
    for pat in _EVASIVE_PATTERNS:
        if _re.search(pat, answer_str, _re.IGNORECASE):
            return True
    return False


def _strip_scaffold(answer_str: str) -> str:
    """Extract just the final answer from a "Reasoning: X Answer: Y"
    scaffolded output. Returns the input unchanged if no scaffold markers
    are present (for legacy / natural-CoT / direct-answer outputs).

    Delegates to eval.metrics.extract_cot_answer so user-facing display
    and EM scoring stay consistent.
    """
    if not answer_str:
        return ""
    try:
        from eval.metrics import extract_cot_answer
        return extract_cot_answer(answer_str)
    except Exception:
        # Never let a display-layer transform crash the pipeline.
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
        cross_encoder: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        cross_encoder : Optional[sentence_transformers.CrossEncoder]
            Branch C Goal 2 + Stage-5 passage reranker. A single
            ``CrossEncoder`` instance used for TWO roles inside the
            verifier: (a) re-ranking retrieved passages top-20 -> top-3
            before grounding NLI, and (b) scoring the q_a_relevance
            signal on (question, display_answer) pairs. The two tasks
            share the same contract (text-pair relevance scoring), so
            the same model instance saves VRAM and keeps the two paths
            calibrated against each other. When ``None`` (default),
            retrieval order is used for rerank and q_a_relevance falls
            back to the 0.5 neutral prior. ``UnifiedVerifier`` applies
            sigmoid to the cross-encoder's raw predict() output before
            clipping to [0, 1] so BGE-style logit-returning models work
            directly.
        """
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
        # v2 Fix 6 (2026-05-07 audit): construct the alias resolver if the
        # configured alias dictionary file is present. Without this, the
        # alias_overlap signal silently falls back to the 0.5 neutral prior
        # for every query, neutralising the signal in the cal_prob composite.
        _alias_resolver = None
        _alias_path = getattr(self.config, "alias_dict_path", None)
        if _alias_path:
            try:
                from pathlib import Path as _Path
                _alias_pp = _Path(_alias_path)
                if _alias_pp.is_file():
                    import json as _json
                    from caem.verification.alias_overlap import InMemoryAliasResolver
                    _alias_table = _json.loads(_alias_pp.read_text(encoding="utf-8"))
                    _alias_resolver = InMemoryAliasResolver(table=_alias_table)
                    logger.info(
                        "Loaded alias_dict from %s (%d canonical entries) — "
                        "alias_overlap signal active.",
                        _alias_pp, len(_alias_table),
                    )
                else:
                    logger.info(
                        "alias_dict_path=%s not present — alias_overlap signal "
                        "stays at neutral 0.5 prior (Phase 1c future-work item).",
                        _alias_pp,
                    )
            except Exception as _exc:
                logger.warning(
                    "alias_dict load failed (%s) — alias_overlap signal stays "
                    "at neutral 0.5 prior.", _exc,
                )

        self.verifier = UnifiedVerifier(
            model=model,
            tokenizer=tokenizer,
            sbert_encoder=encoder,
            judge=judge,
            nli_model=nli_model,
            nli_tokenizer=nli_tokenizer,
            passage_retriever=_verifier_passage_retriever,
            reranker=cross_encoder,            # passage rerank top-20 -> top-3
            qa_relevance_scorer=cross_encoder, # Branch C Goal 2: same instance
            alias_resolver=_alias_resolver,    # v2 Fix 6 — Wikidata-style aliases
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

        # -- RUC runtime-feature components (Phase 1e) ------------------- #
        # The v2 RUC needs 11 features computed at routing time:
        #   FAISS top-5 sims + entity overlap on top-1 + 3 top-5 stats
        #   Cross-encoder rerank (top-1, top-5 mean/std)
        #   Pairwise NLI agreement (mean/min/std over (5 choose 2) pairs)
        #   P(IK) probe on Qwen last-token hidden state
        #   Wikidata entity-binary lookup
        # All components except the probe are already in self.verifier; we
        # store explicit references so ``_compute_ruc_features`` doesn't have
        # to dig into verifier internals.
        self._ruc_passage_store = passage_store
        self._ruc_cross_encoder = cross_encoder
        self._ruc_judge = judge
        self._ruc_alias_resolver = _alias_resolver
        # spaCy NER — reused from the verifier's grounded-fact path
        try:
            import spacy
            self._ruc_nlp = spacy.load("en_core_web_sm")
        except Exception:
            self._ruc_nlp = None

        # P(IK) probe (Kadavath-style; predicts direct_em from Qwen
        # last-token hidden state). Loaded lazily when the router has a
        # RUC wired in. Default path: caem/ruc/v2/pik_probe.joblib.
        self._ruc_pik_probe = None
        from pathlib import Path as _Path
        _pik_path = _Path(__file__).resolve().parent.parent / "caem" / "ruc" / "v2" / "pik_probe.joblib"
        if _pik_path.is_file():
            try:
                import joblib as _joblib
                self._ruc_pik_probe = _joblib.load(_pik_path)
                logger.info(
                    "Loaded P(IK) probe from %s (RUC routing-time feature active).",
                    _pik_path,
                )
            except Exception as _exc:
                logger.warning(
                    "P(IK) probe load failed (%s); RUC will use p_ik=0.5 neutral prior.",
                    _exc,
                )

        # v2 RUC auto-load. Mirrors the P(IK) probe pattern above: if the
        # canonical feature spec exists, load and attach to the router so
        # _consult_ruc fires at Tier 2/3 entry points. Absence keeps the
        # router on the legacy u_pre-only fallback (Phase 1d behaviour).
        _ruc_spec_path = _Path(__file__).resolve().parent.parent / "caem" / "ruc" / "v2" / "feature_spec.json"
        if _ruc_spec_path.is_file():
            try:
                from caem.routing.ruc import LRRetrievalUtilityClassifier
                self.router.ruc = LRRetrievalUtilityClassifier.from_spec(_ruc_spec_path)
                logger.info(
                    "Loaded v2 RUC from %s (tau=%.3f, %d features) — Tier 2/3 routing active.",
                    _ruc_spec_path,
                    self.router.ruc.threshold,
                    len(self.router.ruc.feature_order),
                )
            except Exception as _exc:
                logger.warning(
                    "v2 RUC load failed (%s); router falls back to legacy "
                    "u_pre-only tier selection.", _exc,
                )

        logger.info(
            "CAEMPipeline initialised | device=%s | cycle=%d | memory=%d episodes | ruc=%s",
            self.device, self.current_cycle, self.memory_store.size,
            "v2" if self.router.ruc is not None else "off",
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
        _precomputed_latency_ms: Optional[float] = None,
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
            # v2 Fix 12: source_benchmark dispatches per-benchmark T_b.
            pre_conf = self.pre_estimator.estimate(
                query, source_benchmark=source_benchmark,
            )

            # -- Stage 1: Memory search (k=1 for routing) ------------------- #
            # Use search_with_ids so Tier-1 stats updates can call back without
            # scanning private _metadata for the entry ID.
            search_with_ids = self.memory_store.search_with_ids(
                query_embedding, k=1,
            )

            # -- Stage 3b: Route --------------------------------------------- #
            # v2 Fix 12: source_benchmark dispatches per-benchmark
            # safety_u_pre_min_b for the OR-condition.
            # Branch D (RUC): question is passed so the router can consult the
            # frozen RUC at the two Tier-3 entry points (safety-veto fall-through
            # and score-formula fall-through). The router falls back to the
            # pre-RUC default tier when self.router.ruc is None.
            #
            # Phase 1e: when the RUC is wired in, compute the 13 v2 features
            # (FAISS top-5 stats + cross-encoder rerank + pairwise NLI +
            # Wikidata + P(IK)) and thread them through so the LR has the full
            # feature vector at deployment, matching the training-time signal.
            ruc_extras = None
            if self.router.ruc is not None:
                ruc_extras = self._compute_ruc_features(query, query_embedding)
            routing = self.router.route(
                pre_conf, search_with_ids,
                source_benchmark=source_benchmark,
                question=query,
                top1_passage_text=(ruc_extras or {}).get("_top1_passage_text"),
                top1_passage_sim=(ruc_extras or {}).get("top1_passage_sim"),
                top1_passage_entity_overlap=(ruc_extras or {}).get("top1_passage_entity_overlap"),
                p_ik=(ruc_extras or {}).get("p_ik"),
                extra_ruc_features=ruc_extras,
            )

        logger.debug(
            "Routing: Tier %d | u_pre=%.4f | sim=%.4f | score=%.4f | safety=%s",
            routing.tier, routing.u_pre, routing.similarity,
            routing.routing_score, routing.safety_override,
        )

        # -- Tier dispatch ---------------------------------------------- #
        answer_str: str
        post_conf: Optional[None] = None   # legacy slot; see PipelineResult.post_confidence
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
                vout = self._verify(
                    query, answer_str,
                    u_token=u_tok, u_dropout=u_drop,
                    source_benchmark=source_benchmark,  # v2 Fix 2 keystone
                )
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
                vout = self._verify(
                    query, answer_str,
                    source_benchmark=source_benchmark,  # v2 Fix 2 keystone
                )
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

        # Branch C Goal 5 (2026-04-21): ``_precomputed_latency_ms`` lets
        # BatchPipeline.answer_batch override the per-sample latency with
        # the amortised batch wall-time. Without this, the batched path's
        # per-sample answer() call would log ~0ms because all the heavy
        # work (batched generate + batched verify) already ran. When None
        # (the normal serial path), fall back to the local perf_counter
        # measurement which is accurate for single-query calls.
        if _precomputed_latency_ms is not None:
            latency_ms = float(_precomputed_latency_ms)
        else:
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

    def _compute_ruc_features(
        self, query: str, query_embedding: np.ndarray,
    ) -> Optional[dict]:
        """Compute the 13 v2 RUC routing-time features for one query.

        Reuses CAEM's already-loaded retrieval / rerank / NLI components
        (the same set used by UnifiedVerifier) plus the frozen P(IK) probe.
        Mirrors the offline extractor at
        ``scripts/ruc_extract_v2_features.py:extract_features`` so the
        runtime feature distribution matches training. All failures degrade
        gracefully: missing values fall back to 0.0 (or 0.5 for ``p_ik``),
        which is what the LR's training-time stub produces.

        Returns a dict of 13 v2 features plus three "v1 carry-over" features
        (top1_passage_sim, top1_passage_entity_overlap, p_ik) that the router
        ``route()`` accepts as explicit kwargs. The internal-use
        ``_top1_passage_text`` key is included so the LR's
        ``_answer_type_features`` regex can match against the top-1 passage.
        Returns ``None`` if no RUC is wired or any catastrophic failure
        prevents feature computation; the router then falls back to its
        pre-RUC default tier.
        """
        if self.router.ruc is None:
            return None
        try:
            feats: dict = {}
            # --- FAISS top-5 retrieval ----------------------------------- #
            top_k = 5
            emb = query_embedding.astype(np.float32)
            n = float(np.linalg.norm(emb))
            if n > 0:
                emb = emb / n
            try:
                results = self._ruc_passage_store.search(emb, k=top_k)
            except Exception:
                results = []
            passages: List[str] = []
            sims: List[float] = []
            for r in results:
                if isinstance(r, (tuple, list)) and r:
                    t = r[0] if isinstance(r[0], str) else (
                        r[0].get("text") or r[0].get("passage") or ""
                        if isinstance(r[0], dict) else ""
                    )
                    s = float(r[1]) if len(r) > 1 else 0.0
                elif isinstance(r, str):
                    t, s = r, 0.0
                elif isinstance(r, dict):
                    t, s = r.get("text") or r.get("passage") or "", 0.0
                else:
                    continue
                if t:
                    passages.append(t)
                    sims.append(s)

            if passages:
                sims_arr = np.asarray(sims, dtype=np.float32)
                feats["_top1_passage_text"] = passages[0]
                feats["top1_passage_sim"] = float(sims_arr[0])
                feats["top5_sim_max"] = float(sims_arr.max())
                feats["top5_sim_mean"] = float(sims_arr.mean())
                feats["top5_sim_std"] = float(sims_arr.std())

                # Top-1 entity overlap via spaCy NER
                if self._ruc_nlp is not None:
                    try:
                        doc = self._ruc_nlp(query or "")
                        ents = [e.text.lower() for e in doc.ents]
                        p_lower = passages[0].lower()
                        overlap = sum(1 for e in ents if e in p_lower)
                        feats["top1_passage_entity_overlap"] = (
                            float(overlap) / float(len(ents))
                            if ents else 0.0
                        )
                    except Exception:
                        feats["top1_passage_entity_overlap"] = 0.0
                else:
                    feats["top1_passage_entity_overlap"] = 0.0

                # Cross-encoder rerank on top-5
                if self._ruc_cross_encoder is not None:
                    try:
                        pairs = [(query, p) for p in passages]
                        rerank = self._ruc_cross_encoder.predict(
                            pairs, show_progress_bar=False,
                        )
                        rerank = [float(s) for s in rerank]
                        feats["rerank_top1"] = float(rerank[0])
                        feats["rerank_top5_mean"] = float(np.mean(rerank))
                        feats["rerank_top5_std"] = float(np.std(rerank))
                    except Exception:
                        feats["rerank_top1"] = 0.0
                        feats["rerank_top5_mean"] = 0.0
                        feats["rerank_top5_std"] = 0.0

                # Pairwise NLI on (5 choose 2) = 10 pairs
                if self._ruc_judge is not None and len(passages) >= 2:
                    try:
                        premises, hyps = [], []
                        for i in range(len(passages)):
                            for j in range(i + 1, len(passages)):
                                premises.append(passages[i])
                                hyps.append(passages[j])
                        arr = self._ruc_judge.batch_entail_prob(premises, hyps)
                        arr = np.asarray([float(x) for x in arr], dtype=np.float32)
                        feats["nli_pair_mean"] = float(arr.mean())
                        feats["nli_pair_min"] = float(arr.min())
                        feats["nli_pair_std"] = float(arr.std())
                    except Exception:
                        feats["nli_pair_mean"] = 0.5
                        feats["nli_pair_min"] = 0.5
                        feats["nli_pair_std"] = 0.0
                else:
                    feats["nli_pair_mean"] = 0.5
                    feats["nli_pair_min"] = 0.5
                    feats["nli_pair_std"] = 0.0
            else:
                # No retrieval results — neutral defaults across the board.
                feats.update({
                    "_top1_passage_text": "",
                    "top1_passage_sim": 0.0,
                    "top1_passage_entity_overlap": 0.0,
                    "top5_sim_max": 0.0,
                    "top5_sim_mean": 0.0,
                    "top5_sim_std": 0.0,
                    "rerank_top1": 0.0,
                    "rerank_top5_mean": 0.0,
                    "rerank_top5_std": 0.0,
                    "nli_pair_mean": 0.5,
                    "nli_pair_min": 0.5,
                    "nli_pair_std": 0.0,
                })

            # --- Wikidata entity-binary --------------------------------- #
            wikidata = 0.0
            if self._ruc_alias_resolver is not None and self._ruc_nlp is not None:
                try:
                    doc = self._ruc_nlp(query or "")
                    for ent in doc.ents:
                        if self._ruc_alias_resolver.resolve(ent.text):
                            wikidata = 1.0
                            break
                except Exception:
                    pass
            feats["entity_in_wikidata"] = wikidata

            # --- P(IK) via Qwen last-token hidden state ----------------- #
            # Disable any active LoRA adapter so the hidden state matches the
            # base-Qwen distribution the probe was trained on. At cycle 0 the
            # adapter doesn't exist yet, so disable_adapter is a no-op.
            p_ik = 0.5
            if self._ruc_pik_probe is not None:
                try:
                    enc = self.tokenizer(
                        query, return_tensors="pt", truncation=True, max_length=512,
                    )
                    input_ids = enc["input_ids"].to(self.device)
                    attn = enc["attention_mask"].to(self.device)
                    seq_len = int(attn.sum(dim=1).item()) - 1
                    self.model.eval()
                    # peft adapter disable when present; falls back to a no-op
                    # context for plain HF models.
                    if hasattr(self.model, "disable_adapter"):
                        ctx = self.model.disable_adapter()
                    else:
                        from contextlib import nullcontext
                        ctx = nullcontext()
                    with torch.no_grad(), ctx:
                        out = self.model(
                            input_ids=input_ids,
                            attention_mask=attn,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                    # Last layer's hidden state at the last non-pad token
                    hidden = (
                        out.hidden_states[-1][0, seq_len].float().cpu().numpy()
                    ).reshape(1, -1)
                    p_ik = float(self._ruc_pik_probe.predict_proba(hidden)[0, 1])
                except Exception as _exc:
                    logger.debug(
                        "P(IK) compute failed (%s); using 0.5 neutral prior.",
                        _exc,
                    )
                    p_ik = 0.5
            feats["p_ik"] = p_ik

            return feats
        except Exception as exc:
            logger.warning(
                "_compute_ruc_features raised (%s); returning None and letting "
                "the router fall back to its default tier.", exc,
            )
            return None

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
        """Generate a Tier 2 answer on the decoder-only backbone.

        Builds a ChatML scaffolded-CoT prompt via ``caem.prompts.build_tier2_prompt``,
        appends the ``"Reasoning:"`` forced prefix as prefill, generates, and
        decodes only the newly-generated continuation. Prepends the forced
        prefix to the returned answer so the string begins with ``Reasoning:``
        — the scaffolded-CoT contract the Stage-5 verifier expects.

        Post-generation quality gating is handled entirely by UnifiedVerifier
        (Stage 5); this method returns ``(answer_str, None, escalated)``.

        Returns
        -------
        (answer_str, post_conf, escalated)
        """
        prompt_text, forced_prefix = build_tier2_prompt(query, tokenizer=self.tokenizer)
        full_input = prompt_text + forced_prefix

        enc = self.tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.cot_max_new_tokens + 2048,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        input_len = int(input_ids.shape[1])

        try:
            self.model.eval()
            with torch.no_grad(), section("tier2.generate"):
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.cot_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            continuation = self.tokenizer.decode(
                output_ids[0, input_len:],
                skip_special_tokens=True,
            ).strip()
            answer_str = (
                f"{forced_prefix}{continuation}" if continuation else forced_prefix
            )
        except Exception as exc:
            logger.error("Tier 2 generation failed: %s -- escalating to Tier 3.", exc)
            return self._tier3(query), None, True

        if not answer_str or answer_str.strip() == forced_prefix.strip():
            logger.warning("Tier 2 produced empty answer -- escalating to Tier 3.")
            return self._tier3(query), None, True

        return answer_str, None, False

    def _tier3(self, query: str) -> str:
        """Generate a Tier 3 answer via RAG.

        Returns
        -------
        str -- may be empty string on complete failure.
        """
        with section("tier3.rag_generate"):
            answer = self.rag.generate(query)
        logger.debug("Tier 3 RAG answer: '%s...'", answer[:80])
        return answer

    # Legacy prompt-building helpers (_build_tier2_prompt, _build_forced_prefix,
    # _detect_query_task, _extract_after_token) were removed 2026-04-22 during
    # the T5-removal refactor. Tier 2 prompt construction and task detection
    # now live in caem/prompts.py (build_tier2_prompt, detect_query_task),
    # shared with caem/retrieval/rag.py's Tier 3 path. The forced-prefix
    # mechanism changed from encoder-decoder ``decoder_input_ids`` to
    # decoder-only prefill text (see _tier2 implementation above).

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
        source_benchmark: Optional[str] = None,
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
            with section("verifier.verify"):
                return self.verifier.verify(
                    query, answer,
                    u_token=u_token, u_dropout=u_dropout,
                    source_benchmark=source_benchmark,
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
                # is_query_time=False disables the confabulation early-exit
                # gate during retroactive re-verification. The early-exit
                # was registered for query-time inference; at retroverify
                # the gate's u_internal conjunct is uninformative because
                # SIL fine-tune memorised the stored chains. The composite
                # + threshold prune downstream still removes confidently-
                # wrong stored entries (new_u_stored < tau_retro = 0.50).
                # See verifier.py:verify() docstring for the full rationale.
                # v2 Fix 2 — thread source_benchmark from EpisodicEntry into
                # the verifier so the per-benchmark composite + decision tree
                # use the right calibration at retroverify time.
                return verifier.verify(
                    entry.question, entry.answer, is_query_time=False,
                    source_benchmark=getattr(entry, "source_benchmark", None),
                )
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning(
                    "Retroverify of entry (q=%r) raised %s -- entry left unchanged.",
                    entry.question[:60], exc,
                )
                return None

        return _retroverify

    def make_retroverify_fn_batch(self):
        """Return a closure ``(entries) -> List[Optional[UnifiedVerifierOutput]]``.

        Patch 2026-05-11: batched retroverify entry point. Mirrors
        :meth:`make_retroverify_fn` but takes a list of entries and dispatches
        through :meth:`UnifiedVerifier.verify_batch` with ``is_query_time=False``
        (so the confabulation early-exit stays disabled, identically to the
        serial path). Returns one ``UnifiedVerifierOutput`` per input entry,
        or ``None`` for the whole batch on an unexpected verifier exception
        (callers fall back to the serial closure for that batch).

        The batched path pools the M-chain generation and dropout passes
        across the batch dimension, dropping per-episode wall-time from
        ~9s (serial) to ~3s (batched at N=8). Composite scores are
        byte-equivalent to the serial path modulo floating-point reordering;
        validated by the cycle-0 equivalence smoke (task #105) on the
        query-time path, with no additional risk on the retroverify path
        because the only change is the disabled confab early-exit gate.
        """
        verifier = self.verifier

        def _retroverify_batch(entries) -> List[Optional[UnifiedVerifierOutput]]:
            if not entries:
                return []
            inputs = [(e.question, e.answer) for e in entries]
            source_benchmarks = [
                getattr(e, "source_benchmark", None) for e in entries
            ]
            try:
                return verifier.verify_batch(
                    inputs,
                    source_benchmarks=source_benchmarks,
                    is_query_time=False,
                )
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning(
                    "Retroverify batched call raised %s (N=%d) -- batch left "
                    "for serial fallback.", exc, len(entries),
                )
                return [None] * len(entries)

        return _retroverify_batch

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
                # v2 Fix 2 — thread source_benchmark from DeferredEntry into the
                # verifier so the per-benchmark composite + decision tree use
                # the right calibration at reconsideration time. (Previously
                # this site dropped the benchmark tag, causing all reconsidered
                # entries to be scored under the global pooled composite even
                # though the entries' source_benchmark was preserved on the
                # DeferredEntry.)
                return verifier.verify(
                    deferred_entry.question,
                    deferred_entry.answer,
                    source_benchmark=getattr(deferred_entry, "source_benchmark", None),
                )
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning(
                    "Deferred reconsideration of (q=%r) raised %s -- entry kept.",
                    deferred_entry.question[:60], exc,
                )
                return None

        return _reconsider

    def make_reconsider_deferred_fn_batch(self):
        """Return a closure ``(deferred_entries) -> List[Optional[UnifiedVerifierOutput]]``.

        Patch 2026-05-11: batched deferred-reconsideration entry point.
        Same contract as :meth:`make_retroverify_fn_batch` but takes a list
        of ``DeferredEntry`` objects. Calls ``is_query_time=True`` (matches
        :meth:`make_reconsider_deferred_fn`'s serial default, where the
        confabulation early-exit IS active because the entry has not yet
        passed the storage gate).
        """
        verifier = self.verifier

        def _reconsider_batch(deferred_entries) -> List[Optional[UnifiedVerifierOutput]]:
            if not deferred_entries:
                return []
            inputs = [(d.question, d.answer) for d in deferred_entries]
            source_benchmarks = [
                getattr(d, "source_benchmark", None) for d in deferred_entries
            ]
            try:
                return verifier.verify_batch(
                    inputs,
                    source_benchmarks=source_benchmarks,
                    is_query_time=True,
                )
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning(
                    "Deferred reconsideration batched call raised %s (N=%d) "
                    "-- batch left for serial fallback.", exc, len(deferred_entries),
                )
                return [None] * len(deferred_entries)

        return _reconsider_batch

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

        # B2 transfer-storage block (2026-05-07). Memory is by design a
        # log of TRAINING-panel experiences; transfer-panel benchmarks
        # are HELD OUT for downstream generalisation evaluation. The
        # runtime architecture already prevents transfer entries from
        # entering memory (cold-start uses the training panel only;
        # cycle_stream_chunks for transfer benchmarks are empty by
        # construction in caem/benchmark_splits.py; per-cycle eval calls
        # pass store_to_memory=False). This explicit guard codifies the
        # invariant so future code paths cannot silently re-introduce
        # cross-cycle transfer-eval contamination via Tier 1 cache hits.
        # The DEFERRED branch below also short-circuits on transfer.
        if source_benchmark is not None:
            try:
                from caem.config import TRANSFER_BENCHMARKS as _TRANSFER
                if source_benchmark in _TRANSFER:
                    logger.debug(
                        "Not storing: source_benchmark=%s is in TRANSFER_BENCHMARKS "
                        "(memory is training-panel only by design).",
                        source_benchmark,
                    )
                    return None, False
            except ImportError:
                pass  # config not loadable in some test paths

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

        # E22/E23 FIX (2026-04-24): quality sanitizer. Reject storage if the
        # answer exhibits template-leak or evasive patterns. Without this,
        # the cold-start memory accumulated ~30% unusable entries (template
        # placeholders like `<concise factual answer>` literally in the
        # stored string, and "context doesn't mention X" evasive answers).
        # See branch_C_log.md 2026-04-24 audit.
        if _answer_is_unstorable(answer):
            logger.info(
                "Not storing: answer failed quality sanitizer "
                "(template leak or evasive pattern)."
            )
            return None, False

        # Auto-prune if approaching capacity
        if self.memory_store._should_prune():
            logger.info("Memory store near capacity -- pruning before storing.")
            self.memory_store.prune()

        # E24 FIX (2026-04-24): store the short extracted answer in the
        # `answer` field, not the full verbose reasoning. Tier 1 memory
        # hits in subsequent cycles retrieve `answer` directly; storing
        # the verbose reasoning chain caused the user-facing answer to be
        # ~333 chars of scaffold instead of e.g. "Mitch Murray". The full
        # reasoning is preserved in `reasoning_chain` for diagnostic use.
        display_answer = _strip_scaffold(answer)
        entry = EpisodicEntry(
            question=query,
            reasoning_chain=answer,   # Full CoT preserved for diagnostics
            answer=display_answer if display_answer else answer,
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
            q_a_relevance=vout.q_a_relevance,
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
