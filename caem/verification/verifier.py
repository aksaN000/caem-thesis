"""
caem/verification/verifier.py
==============================
UnifiedVerifier -- the single Stage-5 quality gate of the CAEM pipeline.

This module replaces the previous split between
  Stage 4a : caem/confidence/post_generation.py (PostGenerationConfidenceEstimator)
  Stage 5  : caem/verification/verifier.py     (legacy two-layer NLI verifier)

Both of the old modules computed overlapping signals (u_consistency ≡ s_avg
and u_entropy ≡ h_norm) and the old verifier only checked self-referential
NLI. The Session 42 (2026-04-17) redesign collapses them into one pass and
adds external grounding via NLI against reranked Wikipedia passages --
finally fulfilling the Chapter 1 promise that "NLI verifies factual
correctness against retrieved evidence."

See: caem-implementation-log.md Session 42 for the full design rationale
and cost/latency envelope.

Signal set (nine signals, computed once per Tier 2 / Tier 3 answer)
-------------------------------------------------------------------
Internal calibration (cheap, no external dependency):
    u_token      -- mean token log-prob from generation     [LIT]
    u_dropout    -- variance across K=5 MC-dropout passes   [LIT: Gal 2016]
    u_internal   -- 0.5·u_token + 0.5·(1 − u_dropout)       [DES]

Sample-set signals (model talking to itself, M/K chains):
    s_avg        -- unique-pair cosine sim over M=3 chains  [LIT: Wang 2022]
                                                            (Session 40: i<j)
    h_norm       -- normalised semantic entropy, K=10        [LIT: Farquhar 2024]
    p_entail     -- P(entail | chain -> answer), averaged over M chains,
                    ensemble-min across NLI models          [DES]

External grounding (model vs retrieved Wikipedia evidence):
    p_ground_max     -- top-1 passage -> answer entailment   [DES]
    p_ground_mean    -- mean entailment over top-3 reranked  [DES]
    p_ground_atomic  -- weakest-link atomic-fact entailment  [DES]
    p_contra         -- max contradiction prob, top-3        [DES]

Composite stored confidence (Session 42 baseline prior, pre-calibration)
------------------------------------------------------------------------
    û_stored = 0.30·p_ground_mean
             + 0.15·p_ground_atomic
             + 0.15·s_avg
             + 0.10·(1 − h_norm)
             + 0.15·u_internal
             + 0.15·p_entail

Weights are re-fit by scripts/run_calibration.py after the main run; the
equal-ish split is a deliberately diffuse starting prior.

Early-exit confabulation check (hard gate, before composite is formed)
----------------------------------------------------------------------
    IF  u_internal ≥ 0.70  AND  p_ground_max ≤ 0.20 :
        REJECT -> early_exit -> caller routes to Tier 3 regeneration

This is the one explicit safety rule preventing the "confident-and-wrong"
failure mode. Triggered before any composite scoring -- remaining signals
are not consulted when this fires.

Decision tree outcomes
----------------------
    STORE     : û_stored ≥ 0.65  AND  p_contra < 0.30
    DEFERRED  : 0.45 ≤ û_stored < 0.65  AND  p_contra < 0.30
                -- held in the deferred review queue; re-checked by the
                   retroverify pass each cycle
    ABSTAIN   : û_stored < 0.45  AND  top-3 p_ground_max < 0.20
                -- no evidence either way; emit "I don't know" instead of
                   silently storing or discarding
    DISCARD   : anything else (contradiction veto hit, or composite too
                low with some grounding present)

All thresholds and weights above are read from CAEMConfig with
`getattr(cfg, name, default)` so this module continues to work during the
Phase 5b config migration. The hard-coded defaults match the Session 42
freeze.

Dependency injection
--------------------
The verifier does not own any of the large models it uses -- they are
passed in by the pipeline at construction time. Every optional dependency
has a graceful fallback so the verifier remains useful in smoke/test
configurations.

    model, tokenizer      -- Flan-T5-Large (required; used for M-chain,
                             K-sample and atomic-decomposition generation)
    sbert_encoder         -- all-mpnet-base-v2 (required; s_avg similarity)
    nli_bundles           -- list of (model, tokenizer) tuples forming the
                             ensemble. Falls back to the legacy single
                             (nli_model, nli_tokenizer) pair for BC.
                             When empty -> p_entail=0.5, p_ground_*=0.5,
                             p_contra=0.0, h_norm uses surface entropy.
    reranker              -- sentence-transformers CrossEncoder for the
                             top-20 -> top-3 passage rerank. When None,
                             the top-3 are taken directly from the
                             retriever's ordering.
    passage_retriever     -- Callable(query, k) -> List[str]. Must be
                             supplied to enable p_ground_* / p_contra;
                             otherwise those signals fall back to
                             0.5 / 0.0 respectively and the early-exit
                             gate cannot fire.
    enable_atomic         -- When False, p_ground_atomic == p_ground_mean
                             (atomic decomposition step skipped).

Tier-1 behaviour
----------------
Tier 1 queries bypass this verifier for latency; quality refresh is handled
by retroactive re-verification between cycles. pipeline.py is responsible
for NOT calling verify() on Tier 1.
"""

from __future__ import annotations

import itertools
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from caem.config import CAEMConfig
from caem._profile import section

logger = logging.getLogger(__name__)


# ============================================================================ #
# Output schema                                                                 #
# ---------------------------------------------------------------------------- #
# This is the verifier's public contract. EpisodicEntry (caem.memory.entry)    #
# stores a flattened copy of these nine signals + p_contra + decision.          #
# ============================================================================ #

@dataclass
class UnifiedVerifierOutput:
    """Full verifier output for one (query, answer) pair.

    Consumed directly by the pipeline, the eval harness, and the seven-table
    metric suite. There is no back-compat projection dataclass -- Session 42
    removed StoredConfidence in favour of this single record.
    """

    # --- Internal calibration (cheap) ------------------------------------- #
    u_token: float
    u_dropout: float
    u_internal: float

    # --- Sample-set signals (M/K chains) ---------------------------------- #
    s_avg: float
    h_norm: float
    p_entail: float

    # --- External grounding (against reranked passages) ------------------- #
    p_ground_max: float
    p_ground_mean: float
    p_ground_atomic: float
    p_contra: float

    # --- Composite + decision --------------------------------------------- #
    u_stored: float
    decision: str               # "STORE" | "DEFERRED" | "ABSTAIN" | "DISCARD"
    early_exit_triggered: bool
    abstained: bool

    # --- Metadata for logging / ablation bookkeeping ---------------------- #
    top_passages: List[str] = field(default_factory=list)
    atomic_facts: List[str] = field(default_factory=list)
    per_atom_entail: List[float] = field(default_factory=list)

    # --- Question-answer relevance (Branch C Goal 2) ---------------------- #
    # Cross-encoder score on (question, display_answer). Closes the sample-②
    # off-topic-answer-with-matching-passage failure mode (hallucinated answer
    # that happened to match a retrieved passage, so entailment and grounding
    # both pass while the answer doesn't address the question). Defaults to
    # 0.5 so callers that omit a qa_relevance_scorer get a neutral composite
    # contribution rather than a silent 0 that would penalise every answer.
    q_a_relevance: float = 0.5


# ============================================================================ #
# NLI ensemble helper                                                           #
# ============================================================================ #

class _NLIEnsemble:
    """Thin wrapper around one-or-more NLI bundles.

    Each bundle is a (model, tokenizer) pair. Label convention assumed
    throughout: [CONTRADICTION=0, NEUTRAL=1, ENTAILMENT=2]. Both
    RoBERTa-Large-MNLI and DeBERTa-v3-Large-MNLI follow this order.

    Aggregation rules:
      entail_prob      -> MIN across bundles (conservative: every model must
                          agree before we trust the evidence)
      contradict_prob  -> MAX across bundles (safety: any model flags it)
      argmax_label     -> majority vote (ties break to NEUTRAL)
    """

    def __init__(
        self,
        bundles: Sequence[Tuple[Any, Any]],
        device: str,
    ) -> None:
        self.bundles = list(bundles)
        self.device = device

    def __bool__(self) -> bool:
        return len(self.bundles) > 0

    def _probs(self, model: Any, tokenizer: Any, premise: str, hypothesis: str) -> np.ndarray:
        enc = tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True, max_length=512,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        """min over bundles of P(ENTAIL). Conservative."""
        if not self.bundles:
            return 0.5
        scores = [self._probs(m, t, premise, hypothesis)[2] for (m, t) in self.bundles]
        return float(np.clip(min(scores), 0.0, 1.0))

    def contradict_prob(self, premise: str, hypothesis: str) -> float:
        """max over bundles of P(CONTRADICT). Safety."""
        if not self.bundles:
            return 0.0
        scores = [self._probs(m, t, premise, hypothesis)[0] for (m, t) in self.bundles]
        return float(np.clip(max(scores), 0.0, 1.0))

    def argmax_label(self, premise: str, hypothesis: str) -> int:
        """Majority-vote argmax label. Used only by h_norm NLI clustering."""
        if not self.bundles:
            return 1  # NEUTRAL default when the ensemble is empty
        labels = []
        for (m, t) in self.bundles:
            p = self._probs(m, t, premise, hypothesis)
            labels.append(int(np.argmax(p)))
        counts = np.bincount(labels, minlength=3)
        top = np.flatnonzero(counts == counts.max())
        if len(top) == 1:
            return int(top[0])
        return 1  # ties -> NEUTRAL

    # ---- Batched variants -------------------------------------------------- #
    # Replace the sequential _probs() call pattern with single-batched forward
    # passes. Per-row softmax is numerically identical to per-sample; the only
    # observable difference is wall-clock (~10-50x speedup on the K*K NLI
    # clustering loop in _compute_h_norm and on the passages x samples grids in
    # _score_p_ground / _score_p_contra). Added 2026-04-19 post-Step-4 to raise
    # CAEM per-sample latency ceiling from ~10s to ~2s (see VAST_SESSION_LOG.md).

    def _probs_batch(self, model: Any, tokenizer: Any,
                     premises: Sequence[str], hypotheses: Sequence[str]
                     ) -> np.ndarray:
        """Run one batched forward pass; return (N, 3) probability matrix."""
        enc = tokenizer(
            list(premises), list(hypotheses),
            return_tensors="pt",
            truncation=True, max_length=512,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).detach().cpu().numpy()

    def batch_entail_prob(
        self, pairs: Sequence[Tuple[str, str]],
    ) -> List[float]:
        """Batched version of entail_prob. Returns min-across-bundles per pair.

        Semantics identical to calling entail_prob() in a loop:
        out[i] = min over bundles of softmax(logits(premises[i], hypotheses[i]))[ENTAILMENT]
        """
        if not self.bundles or not pairs:
            return [0.5] * len(pairs)
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        # For each bundle, collect the ENTAILMENT column.
        per_bundle_entail = []
        for (m, t) in self.bundles:
            probs = self._probs_batch(m, t, premises, hypotheses)
            per_bundle_entail.append(probs[:, 2])  # ENTAILMENT = index 2
        stacked = np.stack(per_bundle_entail, axis=0)  # (bundles, N)
        min_across = stacked.min(axis=0)  # (N,) min over bundles per pair
        return [float(np.clip(x, 0.0, 1.0)) for x in min_across]

    def batch_contradict_prob(
        self, pairs: Sequence[Tuple[str, str]],
    ) -> List[float]:
        """Batched version of contradict_prob. Max-across-bundles per pair."""
        if not self.bundles or not pairs:
            return [0.0] * len(pairs)
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        per_bundle_contra = []
        for (m, t) in self.bundles:
            probs = self._probs_batch(m, t, premises, hypotheses)
            per_bundle_contra.append(probs[:, 0])  # CONTRADICTION = index 0
        stacked = np.stack(per_bundle_contra, axis=0)
        max_across = stacked.max(axis=0)
        return [float(np.clip(x, 0.0, 1.0)) for x in max_across]

    def batch_argmax_label(
        self, pairs: Sequence[Tuple[str, str]],
    ) -> List[int]:
        """Batched version of argmax_label. Majority-vote across bundles per pair."""
        N = len(pairs)
        if not self.bundles or not pairs:
            return [1] * N  # NEUTRAL default
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        # Collect per-bundle argmax for each pair.
        per_bundle_labels = np.zeros((len(self.bundles), N), dtype=np.int64)
        for b_idx, (m, t) in enumerate(self.bundles):
            probs = self._probs_batch(m, t, premises, hypotheses)
            per_bundle_labels[b_idx] = np.argmax(probs, axis=1)
        # Majority vote per pair. Ties -> NEUTRAL (1).
        out = []
        for i in range(N):
            labels_i = per_bundle_labels[:, i]
            counts = np.bincount(labels_i, minlength=3)
            top = np.flatnonzero(counts == counts.max())
            out.append(int(top[0]) if len(top) == 1 else 1)
        return out


# ============================================================================ #
# Atomic-fact decomposer                                                        #
# ---------------------------------------------------------------------------- #
# Uses the main Flan-T5-Large model with a fixed decomposition prompt.          #
# Returns a list of simple declarative sentences; each is NLI-checked          #
# independently against the reranked passages; p_ground_atomic = min entail.   #
# ============================================================================ #

_ATOMIC_DECOMP_PROMPT = (
    "Decompose the following statement into a numbered list of simple, "
    "independent factual claims. Each claim must be one complete sentence "
    "expressing a single fact, with all pronouns resolved.\n\n"
    "Statement: {answer}\n\n"
    "Numbered list of facts:"
)

_ATOMIC_FACT_LINE = re.compile(r"^\s*\d+[.):\-]\s*(.+?)\s*$")


def _parse_atomic_facts(raw: str) -> List[str]:
    """Parse a numbered list from model output into a list of claims.

    Tolerant to "1." / "1)" / "1:" / "1-" numbering and to paragraphs that
    put one claim on each line without numbering.
    """
    facts: List[str] = []
    for line in raw.splitlines():
        m = _ATOMIC_FACT_LINE.match(line)
        if m:
            facts.append(m.group(1).strip())
        else:
            stripped = line.strip()
            if stripped and not stripped.endswith(":"):
                facts.append(stripped)
    seen = set()
    unique: List[str] = []
    for f in facts:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


# ============================================================================ #
# UnifiedVerifier                                                               #
# ============================================================================ #

class UnifiedVerifier:
    """Single Stage-5 quality gate. See module docstring for full design.

    All thresholds, composite weights, and retrieval k's are read from
    ``caem.config.CAEMConfig`` -- no magic numbers in this module. The
    authoritative defaults live in the ``UnifiedVerifier (Stage 5)``
    block of ``caem/config.py``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        sbert_encoder,
        *,
        judge: Optional[Any] = None,
        nli_bundles: Optional[Sequence[Tuple[Any, Any]]] = None,
        nli_model: Optional[Any] = None,
        nli_tokenizer: Optional[Any] = None,
        reranker: Optional[Any] = None,
        passage_retriever: Optional[Callable[[str, int], List[str]]] = None,
        qa_relevance_scorer: Optional[Any] = None,
        enable_atomic: bool = True,
        config: Optional[CAEMConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sbert_encoder = sbert_encoder
        self.config = config or CAEMConfig()
        self.reranker = reranker
        self.passage_retriever = passage_retriever
        # Branch C Goal 2: BGE / MiniLM cross-encoder scoring
        # (question, display_answer) relevance. Any object with a
        # ``.predict(List[Tuple[str, str]]) -> np.ndarray`` method is
        # compatible (sentence-transformers CrossEncoder,
        # FlagEmbedding.FlagReranker, or a thin wrapper). None disables
        # the signal and the composite falls back to the 0.5 neutral prior.
        self.qa_relevance_scorer = qa_relevance_scorer
        self.enable_atomic = enable_atomic

        if device is None:
            device = str(next(model.parameters()).device)
        self.device = device

        # Judge selection precedence:
        #   1. Explicit judge= (MiniCheck or future alternatives)
        #   2. Legacy nli_bundles (multi-model RoBERTa/DeBERTa ensemble)
        #   3. Legacy single (nli_model, nli_tokenizer) pair
        # The judge exposes the same (batch_)entail_prob / contradict_prob /
        # argmax_label triplet as _NLIEnsemble, so downstream call sites are
        # backend-agnostic.
        if judge is not None:
            self.nli = judge
            # Back-compat: the RoBERTa bundle-probe attributes are None for
            # non-NLI judges. Downstream code that probes `self.nli_model`
            # should migrate to `self.nli` instead.
            self.nli_model = None
            self.nli_tokenizer = None
        else:
            bundles: List[Tuple[Any, Any]] = []
            if nli_bundles:
                bundles.extend(nli_bundles)
            elif nli_model is not None and nli_tokenizer is not None:
                bundles.append((nli_model, nli_tokenizer))
            self.nli = _NLIEnsemble(bundles, device=self.device)
            self.nli_model = bundles[0][0] if bundles else None
            self.nli_tokenizer = bundles[0][1] if bundles else None

    # ====================================================================== #
    # Public API                                                               #
    # ====================================================================== #

    def verify(
        self,
        query: str,
        answer: str,
        input_ids: Optional[torch.Tensor] = None,
        *,
        u_token: Optional[float] = None,
        u_dropout: Optional[float] = None,
        chains: Optional[List[str]] = None,
        se_samples: Optional[List[str]] = None,
    ) -> UnifiedVerifierOutput:
        """Compute all nine signals, form the composite, and emit a decision.

        Parameters
        ----------
        query : str
        answer : str
        input_ids : torch.Tensor or None
            Pre-tokenized query; tokenized internally when None.
        u_token, u_dropout : float or None
            Pre-computed internal signals from the generation stage. When
            None, they are recomputed here. pipeline.py should pass these
            in to avoid paying for a duplicate forward pass.
        """
        if input_ids is None:
            input_ids = self._tokenize(query)["input_ids"]
        assert input_ids is not None

        # TEMP profiling 2026-04-21: per-stage wall-time accounting inside
        # a single verify() call. Aggregated at the batch level to guide
        # the deep-batching refactor. Remove after verifier_stage_time.py
        # or equivalent stops being useful.
        import time as _t
        _per_stage_ms = {}

        # ---------- internal calibration (cheap) -------------------------- #
        _ts = _t.perf_counter()
        with section("verifier.u_token_dropout"):
            if u_token is None:
                u_token = self._compute_u_token(input_ids, answer)
            if u_dropout is None:
                u_dropout = self._compute_u_dropout(input_ids)
            u_internal = 0.5 * u_token + 0.5 * (1.0 - u_dropout)
        _per_stage_ms["u_tok_drop"] = (_t.perf_counter() - _ts) * 1000.0

        # ---------- sample-set signals (M chains reused) ------------------ #
        _ts = _t.perf_counter()
        with section("verifier.m_chain_and_h_norm"):
            if chains is None:
                chains = self._generate_m_chains(input_ids)
            s_avg = self._score_s_avg(chains)
            p_entail = self._score_p_entail(chains, answer)
            if se_samples is None:
                h_norm = self._compute_h_norm(input_ids)
            else:
                h_norm = self._h_norm_from_samples(se_samples)
        _per_stage_ms["m_chain_h_norm"] = (_t.perf_counter() - _ts) * 1000.0

        # ---------- external grounding ------------------------------------ #
        _ts = _t.perf_counter()
        top_passages = self._retrieve_and_rerank(query, answer)
        _per_stage_ms["retrieve_rerank"] = (_t.perf_counter() - _ts) * 1000.0

        _ts = _t.perf_counter()
        p_ground_max, p_ground_mean = self._score_p_ground(top_passages, answer)
        _per_stage_ms["p_ground_nli"] = (_t.perf_counter() - _ts) * 1000.0

        _ts = _t.perf_counter()
        p_contra = self._score_p_contra(top_passages, answer)
        _per_stage_ms["p_contra"] = (_t.perf_counter() - _ts) * 1000.0

        _ts = _t.perf_counter()
        atomic_facts, per_atom_entail, p_ground_atomic = self._score_atomic(
            top_passages, answer, fallback=p_ground_mean
        )
        _per_stage_ms["atomic"] = (_t.perf_counter() - _ts) * 1000.0

        # ---------- question-answer relevance (Branch C Goal 2) ----------- #
        _ts = _t.perf_counter()
        with section("verifier.q_a_relevance"):
            q_a_relevance = self._compute_q_a_relevance(query, answer)
        _per_stage_ms["q_a_relevance"] = (_t.perf_counter() - _ts) * 1000.0

        # Accumulate onto instance (batch-level rollup done by verify_batch).
        if not hasattr(self, "_last_verify_stage_ms"):
            self._last_verify_stage_ms = {}
        for k, v in _per_stage_ms.items():
            self._last_verify_stage_ms[k] = self._last_verify_stage_ms.get(k, 0.0) + v

        # ---------- early-exit confabulation gate ------------------------- #
        ee_u = self.config.early_exit_u_internal
        ee_g = self.config.early_exit_p_ground_max
        early_exit = (u_internal >= ee_u) and (p_ground_max <= ee_g)

        if early_exit:
            logger.info(
                "early-exit | u_internal=%.3f >= %.2f AND p_ground_max=%.3f <= %.2f "
                "-> DISCARD (route to Tier 3 regen)",
                u_internal, ee_u, p_ground_max, ee_g,
            )
            return UnifiedVerifierOutput(
                u_token=float(u_token),
                u_dropout=float(u_dropout),
                u_internal=float(u_internal),
                s_avg=float(s_avg),
                h_norm=float(h_norm),
                p_entail=float(p_entail),
                p_ground_max=float(p_ground_max),
                p_ground_mean=float(p_ground_mean),
                p_ground_atomic=float(p_ground_atomic),
                p_contra=float(p_contra),
                u_stored=0.0,
                decision="DISCARD",
                early_exit_triggered=True,
                abstained=False,
                top_passages=top_passages,
                atomic_facts=atomic_facts,
                per_atom_entail=per_atom_entail,
                q_a_relevance=float(q_a_relevance),
            )

        # ---------- composite --------------------------------------------- #
        u_stored = self._composite(
            p_ground_mean=p_ground_mean,
            p_ground_atomic=p_ground_atomic,
            s_avg=s_avg,
            h_norm=h_norm,
            u_internal=u_internal,
            p_entail=p_entail,
            q_a_relevance=q_a_relevance,
        )

        # ---------- decision tree ----------------------------------------- #
        decision, abstained = self._decide(
            u_stored=u_stored,
            p_contra=p_contra,
            p_ground_max=p_ground_max,
        )

        logger.debug(
            "verify | u_tok=%.3f u_drop=%.3f u_int=%.3f | "
            "s_avg=%.3f h_norm=%.3f p_ent=%.3f | "
            "pg_max=%.3f pg_mean=%.3f pg_atomic=%.3f p_contra=%.3f | "
            "q_a_rel=%.3f | "
            "u_stored=%.3f decision=%s",
            u_token, u_dropout, u_internal,
            s_avg, h_norm, p_entail,
            p_ground_max, p_ground_mean, p_ground_atomic, p_contra,
            q_a_relevance,
            u_stored, decision,
        )

        return UnifiedVerifierOutput(
            u_token=float(u_token),
            u_dropout=float(u_dropout),
            u_internal=float(u_internal),
            s_avg=float(s_avg),
            h_norm=float(h_norm),
            p_entail=float(p_entail),
            p_ground_max=float(p_ground_max),
            p_ground_mean=float(p_ground_mean),
            p_ground_atomic=float(p_ground_atomic),
            p_contra=float(p_contra),
            u_stored=float(u_stored),
            decision=decision,
            early_exit_triggered=False,
            abstained=abstained,
            top_passages=top_passages,
            atomic_facts=atomic_facts,
            per_atom_entail=per_atom_entail,
            q_a_relevance=float(q_a_relevance),
        )

    def should_store(self, out: UnifiedVerifierOutput) -> bool:
        """Back-compat: True iff the verifier's decision is STORE."""
        return out.decision == "STORE"

    def verify_batch(
        self,
        inputs: List[Tuple[str, str]],
        u_tokens: Optional[List[Optional[float]]] = None,
        u_dropouts: Optional[List[Optional[float]]] = None,
    ) -> List[UnifiedVerifierOutput]:
        """Run verification on N (query, answer) pairs in one batched pass.

        Pools the two dominant per-sample T5 costs across samples:

        1. **M-chain generation** (``_generate_m_chains``): each sample
           normally draws M=3 i.i.d. chains at T=0.7 via M sequential
           ``model.generate`` calls. Here we draw N*M chains in a single
           batched ``model.generate(num_return_sequences=M)`` on padded
           input_ids, eliminating N*M kernel launches.
        2. **Semantic-entropy sampling** (``_compute_h_norm``): each
           sample normally draws K samples at T=``se_temperature`` via
           one batched ``model.generate(num_return_sequences=K)``. Here
           we pool N*K samples across the batch dimension in one call.

        All other verifier work (retrieval, NLI scoring, atomic
        decomposition, composite, decide) runs per-sample; the NLI
        ensemble already batches within-sample via its
        ``batch_entail_prob`` / ``batch_contradict_prob`` API, so
        amortisation there is already maximal.

        Parameters
        ----------
        inputs : list of (query, answer) tuples
        u_tokens, u_dropouts : optional per-sample precomputed internal
            signals from the generation stage.

        Returns
        -------
        list of UnifiedVerifierOutput of length N, same order as ``inputs``.
        """
        if not inputs:
            return []

        N = len(inputs)
        if u_tokens is None:
            u_tokens = [None] * N
        if u_dropouts is None:
            u_dropouts = [None] * N
        assert len(u_tokens) == N and len(u_dropouts) == N, (
            "verify_batch: u_tokens/u_dropouts length must match inputs"
        )

        queries = [q for q, _ in inputs]
        answers = [a for _, a in inputs]

        # Per-sample tokenisation for the internal-cal forward passes.
        # These paths expect an unpadded single-row tensor (the model
        # computes loss/gradients on exactly the prompt tokens).
        per_sample_input_ids = [
            self._tokenize(q)["input_ids"] for q in queries
        ]

        # Pooled M-chain and semantic-entropy sampling across samples.
        # These two calls are data-independent -- neither reads the other's
        # output -- so we dispatch them to separate CUDA streams (Phase 2)
        # to let the GPU scheduler overlap their kernel launches when it
        # has idle SMs. On a fully-saturated 5090 the overlap is small
        # (~3-5%) but meaningful when the batch dim is modest; on partially
        # utilised GPUs (smaller cards or smaller batches) it can approach
        # 15-20%. Falls back to sequential dispatch when CUDA is not
        # available (CPU tests, non-NVIDIA devices).
        # TEMP profiling 2026-04-21: log per-stage wall-time of batched
        # verify_batch so we can pick the right stages to deep-batch next.
        # Remove after the deep-batching refactor ships.
        import time as _t
        _t0 = _t.perf_counter()
        # Reset the per-sample stage accumulator so this batch's numbers
        # don't include stale state from a previous batch.
        self._last_verify_stage_ms = {}

        chains_per_sample, se_samples_per_sample = self._pooled_sample_dual(
            queries=queries,
            num_per_a=self.config.sc_chains_m,
            temperature_a=0.7,
            label_a="m-chain",
            num_per_b=self.config.se_samples_k,
            temperature_b=self.config.se_temperature,
            label_b="se-sample",
            max_new_tokens=self.config.cot_max_new_tokens,
        )
        t_pool = _t.perf_counter()

        # Time each per-sample verify() call so we know which internal
        # stage dominates. Report sum per stage at the end of the batch.
        per_stage_ms: dict = {
            "pool_m_chain_plus_se": (t_pool - _t0) * 1000.0,
            "verify_per_sample_total": 0.0,
        }
        outputs: List[UnifiedVerifierOutput] = []
        for i, (q, a) in enumerate(inputs):
            _ts = _t.perf_counter()
            outputs.append(self.verify(
                q, a,
                input_ids=per_sample_input_ids[i],
                u_token=u_tokens[i],
                u_dropout=u_dropouts[i],
                chains=chains_per_sample[i],
                se_samples=se_samples_per_sample[i],
            ))
            per_stage_ms["verify_per_sample_total"] += (_t.perf_counter() - _ts) * 1000.0
        total_ms = (_t.perf_counter() - _t0) * 1000.0
        logger.info(
            "verify_batch N=%d: total=%.0fms | pool(m+se)=%.0fms | "
            "per-sample-verify=%.0fms (%.0fms/sample)",
            N, total_ms,
            per_stage_ms["pool_m_chain_plus_se"],
            per_stage_ms["verify_per_sample_total"],
            per_stage_ms["verify_per_sample_total"] / max(N, 1),
        )
        # Per-stage breakdown summed across the N samples. Shows which
        # internal verify() stage is the deep-batching priority.
        stages = self._last_verify_stage_ms or {}
        if stages:
            stage_lines = " | ".join(
                f"{k}={stages.get(k, 0.0):.0f}ms ({stages.get(k, 0.0)/max(N, 1):.0f}ms/sample)"
                for k in ("u_tok_drop", "m_chain_h_norm", "retrieve_rerank",
                         "p_ground_nli", "p_contra", "atomic", "q_a_relevance")
                if k in stages
            )
            logger.info("verify_batch stage breakdown [total over %d samples]: %s", N, stage_lines)
        return outputs

    def _pooled_sample(
        self,
        queries: List[str],
        *,
        num_per: int,
        temperature: float,
        max_new_tokens: int,
        label: str,
    ) -> List[List[str]]:
        """Draw ``num_per`` i.i.d. samples for each of N queries in one
        batched decoder-only ``model.generate(num_return_sequences=num_per)``
        call. Queries are ChatML-wrapped + left-padded so rows in the batch
        align for decoding.

        Returns a list of N lists, each of length ``num_per``. On failure
        returns N empty lists so downstream scorers fall back gracefully
        (matches the serial ``_generate_m_chains`` / ``_compute_h_norm``
        error semantics).
        """
        N = len(queries)
        if N == 0 or num_per <= 0:
            return [[] for _ in range(N)]
        try:
            # ChatML-wrap each query so Qwen sees assistant-turn prompts (same
            # reasoning as the single-query _tokenize path: raw text collapses
            # per-token probabilities on instruction-tuned decoder-only models).
            wrapped = [self._wrap_chatml(q) for q in queries]
            # Left-padding is required for decoder-only batched generation
            # alignment — all rows must end at the same position so
            # autoregression starts from the same index for every sample.
            original_side = getattr(self.tokenizer, "padding_side", None)
            self.tokenizer.padding_side = "left"
            enc = self.tokenizer(
                wrapped,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            if original_side is not None:
                self.tokenizer.padding_side = original_side

            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            input_len = int(input_ids.shape[1])
            self.model.eval()
            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=num_per,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # HF generate with num_return_sequences=M on batch N produces
            # (N*M, total_len) with rows grouped by input: [s0_c0..s0_cM-1,
            # s1_c0..s1_cM-1, ...]. Decode only generated tokens
            # (out[:, input_len:]) and regroup.
            per_sample: List[List[str]] = [[] for _ in range(N)]
            for row_idx in range(out.shape[0]):
                s = row_idx // num_per
                decoded = self.tokenizer.decode(
                    out[row_idx, input_len:], skip_special_tokens=True,
                ).strip()
                per_sample[s].append(decoded)
            return per_sample
        except Exception as exc:
            logger.warning(
                "_pooled_sample (%s) failed (N=%d, num_per=%d): %s -- "
                "returning empty samples so downstream scorers fall back.",
                label, N, num_per, exc,
            )
            return [[] for _ in range(N)]

    def _pooled_sample_dual(
        self,
        *,
        queries: List[str],
        num_per_a: int,
        temperature_a: float,
        label_a: str,
        num_per_b: int,
        temperature_b: float,
        label_b: str,
        max_new_tokens: int,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Run two data-independent pooled samplings concurrently.

        Level B Phase 2 CUDA-stream overlap. The two calls (M-chain at
        temperature_a, semantic-entropy at temperature_b) read the same
        tokenised input but write separate output tensors; launching
        them on separate CUDA streams lets the hardware scheduler
        overlap kernel execution when SM capacity allows.

        Stream semantics:
          * Tokenisation runs once on the main stream (both calls read
            the same input_ids / attention_mask, so we share the encode).
          * ``model.generate`` for sampler A runs on stream_a; sampler
            B on stream_b. A synchronisation barrier (stream_a.wait
            + stream_b.wait on current stream) fences both before we
            read back the outputs for decode.
          * When CUDA is unavailable (CPU, MPS, ROCm without stream
            emulation), we fall back to the serial ``_pooled_sample``
            path -- zero behavioural change.

        Returns
        -------
        (per_sample_a, per_sample_b) each a list of N lists. On any
        failure of either sampler, that sampler's output is a list of
        N empty lists and the downstream scorer falls back (matching
        the single-sampler error semantics).
        """
        N = len(queries)
        if N == 0:
            return [[] for _ in range(N)], [[] for _ in range(N)]

        # CPU or non-CUDA fallback: serial dispatch is always correct
        # and is what the tests expect when no CUDA device is present.
        use_streams = (
            torch.cuda.is_available()
            and isinstance(self.device, str)
            and self.device.startswith("cuda")
        )
        if not use_streams:
            a = self._pooled_sample(
                queries,
                num_per=num_per_a,
                temperature=temperature_a,
                max_new_tokens=max_new_tokens,
                label=label_a,
            )
            b = self._pooled_sample(
                queries,
                num_per=num_per_b,
                temperature=temperature_b,
                max_new_tokens=max_new_tokens,
                label=label_b,
            )
            return a, b

        try:
            # Decoder-only ChatML wrap + left padding (same reasoning as the
            # single-sampler path). All rows end at the same position so the
            # two concurrent streams can use the same input_len for decode.
            wrapped = [self._wrap_chatml(q) for q in queries]
            original_side = getattr(self.tokenizer, "padding_side", None)
            self.tokenizer.padding_side = "left"
            enc = self.tokenizer(
                wrapped,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )
            if original_side is not None:
                self.tokenizer.padding_side = original_side
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            input_len = int(input_ids.shape[1])
            self.model.eval()

            stream_a = torch.cuda.Stream(device=self.device)
            stream_b = torch.cuda.Stream(device=self.device)

            # Ensure streams see the current stream's pending work
            # (the tokenizer .to(device) copy above).
            current = torch.cuda.current_stream(device=self.device)
            stream_a.wait_stream(current)
            stream_b.wait_stream(current)

            out_a = None
            out_b = None
            err_a: Optional[Exception] = None
            err_b: Optional[Exception] = None

            with torch.no_grad():
                with torch.cuda.stream(stream_a):
                    try:
                        if num_per_a > 0:
                            out_a = self.model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=temperature_a,
                                num_return_sequences=num_per_a,
                                pad_token_id=self.tokenizer.pad_token_id,
                            )
                    except Exception as exc_a:
                        err_a = exc_a

                with torch.cuda.stream(stream_b):
                    try:
                        if num_per_b > 0:
                            out_b = self.model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=temperature_b,
                                num_return_sequences=num_per_b,
                                pad_token_id=self.tokenizer.pad_token_id,
                            )
                    except Exception as exc_b:
                        err_b = exc_b

            # Fence both streams before decoding on the main thread.
            current.wait_stream(stream_a)
            current.wait_stream(stream_b)
            # Guarantee the host sees the completed tensors before decode.
            torch.cuda.synchronize(device=self.device)

            per_sample_a = self._decode_grouped(
                out_a, N=N, num_per=num_per_a, label=label_a, err=err_a,
                input_len=input_len,
            )
            per_sample_b = self._decode_grouped(
                out_b, N=N, num_per=num_per_b, label=label_b, err=err_b,
                input_len=input_len,
            )
            return per_sample_a, per_sample_b
        except Exception as exc:
            logger.warning(
                "_pooled_sample_dual failed (N=%d) -- falling back to "
                "serial dispatch: %s", N, exc,
            )
            a = self._pooled_sample(
                queries, num_per=num_per_a, temperature=temperature_a,
                max_new_tokens=max_new_tokens, label=label_a,
            )
            b = self._pooled_sample(
                queries, num_per=num_per_b, temperature=temperature_b,
                max_new_tokens=max_new_tokens, label=label_b,
            )
            return a, b

    def _decode_grouped(
        self,
        out: Optional[torch.Tensor],
        *,
        N: int,
        num_per: int,
        label: str,
        err: Optional[Exception],
        input_len: int = 0,
    ) -> List[List[str]]:
        """Decode a pooled generate output into N per-sample lists.

        Shared by _pooled_sample_dual for both stream outputs. If
        the sampler raised or produced no tensor, returns N empty
        lists so downstream scorers fall back.

        ``input_len`` slices out the prompt prefix (decoder-only semantics:
        sequences = [prompt_tokens..., generated_tokens...]). Defaults to 0
        for backward compatibility with legacy callers.
        """
        if err is not None:
            logger.warning(
                "_pooled_sample_dual (%s) sampler failed (N=%d, "
                "num_per=%d): %s -- returning empty samples.",
                label, N, num_per, err,
            )
            return [[] for _ in range(N)]
        if out is None or num_per <= 0:
            return [[] for _ in range(N)]
        per_sample: List[List[str]] = [[] for _ in range(N)]
        for row_idx in range(out.shape[0]):
            s = row_idx // num_per
            try:
                decoded = self.tokenizer.decode(
                    out[row_idx, input_len:], skip_special_tokens=True,
                ).strip()
            except Exception as exc:
                logger.warning(
                    "_decode_grouped (%s) decode row %d failed: %s",
                    label, row_idx, exc,
                )
                decoded = ""
            per_sample[s].append(decoded)
        return per_sample

    # ====================================================================== #
    # Signal computation                                                      #
    # ====================================================================== #

    # ---- internal calibration --------------------------------------------- #

    def _compute_u_token(self, input_ids: torch.Tensor, answer: str) -> float:
        """Geometric mean of per-token probabilities of the answer under the model.

        Decoder-only causal-LM teacher forcing:
            1. Tokenize the answer (no special tokens added; answer continues
               the already-ChatML-wrapped prefix in ``input_ids``).
            2. Concatenate prefix + answer tokens -> one long sequence.
            3. Forward pass: ``logits[:, k, :]`` predicts the token at
               position ``k+1``.
            4. For answer token ``j`` (at full-sequence position
               ``prefix_len + j``), the predicting logit is at position
               ``prefix_len - 1 + j``. Gather log P of each answer token
               from its predicting position.
            5. Geometric mean over answer tokens = exp(mean(log_probs)).

        Padding is absent here (batch size 1, no padding), so no mask is
        needed. Returns 0.5 on any failure (neutral fallback).
        """
        try:
            self.model.eval()
            answer_ids = self.tokenizer(
                answer,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.cot_max_new_tokens,
                add_special_tokens=False,
            ).input_ids.to(self.device)
            if answer_ids.shape[1] == 0:
                return 0.5

            full_ids = torch.cat([input_ids, answer_ids], dim=1)
            prefix_len = int(input_ids.shape[1])

            with torch.no_grad():
                out = self.model(input_ids=full_ids)
                logits = out.logits  # (1, total_len, V)

            # logits at position k predict token at k+1; so for answer token
            # j (full-sequence position prefix_len + j), predicting logit is
            # at position prefix_len - 1 + j.
            answer_len = int(answer_ids.shape[1])
            start = prefix_len - 1
            end = start + answer_len  # slice end exclusive
            pred_logits = logits[0, start:end, :]  # (answer_len, V)
            log_probs = F.log_softmax(pred_logits, dim=-1)
            gathered = log_probs.gather(
                1, answer_ids[0].unsqueeze(1)
            ).squeeze(1)  # (answer_len,)
            if gathered.numel() == 0:
                return 0.5
            mean_logp = gathered.mean().item()
            return float(np.clip(math.exp(mean_logp), 0.0, 1.0))
        except Exception as exc:
            logger.warning("u_token failed: %s -- returning 0.5", exc)
            return 0.5

    def _compute_u_dropout(self, input_ids: torch.Tensor) -> float:
        """MC-dropout variance across K passes.

        Higher variance across dropout samples -> higher u_dropout. The
        composite uses (1 - u_dropout), so a model that stably produces
        the same answer under perturbation scores higher. Proxy metric:
        fraction of K samples that do NOT match the plurality answer.

        Goal-5 pull-forward: K samples drawn in ONE batched
        ``model.generate(num_return_sequences=K)`` call under train() mode
        (so dropout is active). Previously this was K sequential calls —
        ~K× kernel-launch overhead. Now one sustained GPU burst.
        """
        K = getattr(self.config, "mc_dropout_k", 5)
        prefix_len = int(input_ids.shape[1])
        try:
            self.model.train()  # activate dropout
            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.cot_max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=K,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # Decoder-only: slice [prefix_len:] to decode only the newly
            # generated tokens, not the repeated prompt.
            samples: List[str] = []
            for i in range(out.shape[0]):
                decoded = self.tokenizer.decode(
                    out[i, prefix_len:],
                    skip_special_tokens=True,
                ).strip()
                samples.append(decoded)
            self.model.eval()
            if len(samples) < 2:
                return 0.5
            counts: dict = {}
            for s in samples:
                k = re.sub(r"[^\w\s]", "", s.lower()).strip()
                counts[k] = counts.get(k, 0) + 1
            max_count = max(counts.values())
            var = 1.0 - (max_count / float(K))
            return float(np.clip(var, 0.0, 1.0))
        except Exception as exc:
            self.model.eval()
            logger.warning("u_dropout failed: %s -- returning 0.5", exc)
            return 0.5

    # ---- sample-set (M chains shared between s_avg and p_entail) ---------- #

    def _generate_m_chains(self, input_ids: torch.Tensor) -> List[str]:
        """Generate M=3 sampled chains (T=0.7) shared by s_avg and p_entail.

        Goal-5 pull-forward: M samples drawn in ONE batched
        ``model.generate(num_return_sequences=M)`` call instead of M
        sequential kernel launches. Semantically equivalent (M i.i.d.
        samples at the same temperature).
        """
        M = self.config.sc_chains_m
        prefix_len = int(input_ids.shape[1])
        chains: List[str] = []
        try:
            self.model.eval()
            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.cot_max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=M,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for i in range(out.shape[0]):
                decoded = self.tokenizer.decode(
                    out[i, prefix_len:],
                    skip_special_tokens=True,
                ).strip()
                chains.append(decoded)
        except Exception as exc:
            logger.warning("m-chain generation failed: %s", exc)
        return chains

    def _score_s_avg(self, chains: List[str]) -> float:
        """Unique-pair (i<j) cosine similarity over M chains via SBERT."""
        if len(chains) < 2:
            return 0.5
        try:
            embs = self.sbert_encoder.encode(chains)
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)
            sims = [
                float(np.dot(embs[i], embs[j]))
                for i, j in itertools.combinations(range(len(embs)), 2)
            ]
            s_avg = float(np.mean(sims)) if sims else 0.5
            return float(np.clip(s_avg, 0.0, 1.0))
        except Exception as exc:
            logger.warning("s_avg failed: %s -- returning 0.5", exc)
            return 0.5

    def _score_p_entail(self, chains: List[str], answer: str) -> float:
        """Avg over chains of ensemble-min P(ENTAIL | chain -> answer).

        Batched: one NLI forward per bundle instead of len(chains) forwards.
        """
        if not self.nli or not chains:
            return 0.5
        try:
            pairs = [(c, answer) for c in chains]
            scores = self.nli.batch_entail_prob(pairs)
            return float(np.clip(np.mean(scores), 0.0, 1.0))
        except Exception as exc:
            logger.warning("p_entail failed: %s -- returning 0.5", exc)
            return 0.5

    # ---- semantic entropy ------------------------------------------------- #

    def _h_norm_from_samples(self, samples: List[str]) -> float:
        """Compute h_norm from a pre-drawn list of K semantic-entropy samples.

        Shared between the serial ``_compute_h_norm`` (which draws the
        samples itself) and ``verify_batch`` (which pools the draws
        across samples). Returns 0.5 on any internal failure so
        downstream composite/decide always sees a well-formed float.
        """
        K = self.config.se_samples_k
        try:
            if not samples:
                return 0.5
            H = (self._semantic_entropy_nli(samples)
                 if self.nli else self._semantic_entropy_surface(samples))
            h_norm = H / math.log2(max(K, 2))
            return float(np.clip(h_norm, 0.0, 1.0))
        except Exception as exc:
            logger.warning("h_norm-from-samples failed: %s -- returning 0.5", exc)
            return 0.5

    def _compute_h_norm(self, input_ids: torch.Tensor) -> float:
        """Normalised semantic entropy via NLI clustering (ensemble argmax).

        Batched: single model.generate(num_return_sequences=K) instead of K
        sequential generations. Semantically identical (K i.i.d. samples at
        the same temperature) but exercises the GPU in one sustained burst
        instead of K sequential launches.
        """
        K = self.config.se_samples_k
        T = self.config.se_temperature
        prefix_len = int(input_ids.shape[1])
        try:
            self.model.eval()
            samples: List[str] = []
            with torch.no_grad():
                # Single batched call producing K i.i.d. samples. Equivalent to
                # K sequential calls with the same temperature/do_sample setup
                # but avoids K kernel-launch round-trips.
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.cot_max_new_tokens,
                    do_sample=True, temperature=T,
                    num_return_sequences=K,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                for i in range(out.shape[0]):
                    # Decoder-only: slice [prefix_len:] for generated tokens only
                    decoded = self.tokenizer.decode(
                        out[i, prefix_len:], skip_special_tokens=True
                    ).strip()
                    samples.append(decoded)
            return self._h_norm_from_samples(samples)
        except Exception as exc:
            logger.warning("h_norm failed: %s -- returning 0.5", exc)
            return 0.5

    def _semantic_entropy_nli(self, samples: List[str]) -> float:
        """Bidirectional NLI clustering -> Shannon entropy (bits).

        Batched: collect all (i, j) and (j, i) pairs first, do ONE batched
        argmax across both directions, then union-find on the results. Same
        math as the nested-loop version (N*(N-1) NLI calls) but executed as
        a single forward pass per bundle instead of N*(N-1) forwards.
        """
        N = len(samples)
        parent = list(range(N))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        ENTAILMENT = 2
        # Collect all bidirectional (i, j) pairs in a single list.
        # pairs_fwd[k] = (samples[i], samples[j]); pairs_bwd[k] = (samples[j], samples[i])
        ij_index = []
        pairs_fwd: List[Tuple[str, str]] = []
        pairs_bwd: List[Tuple[str, str]] = []
        for i in range(N):
            for j in range(i + 1, N):
                ij_index.append((i, j))
                pairs_fwd.append((samples[i], samples[j]))
                pairs_bwd.append((samples[j], samples[i]))

        # One batched NLI forward per bundle for each direction.
        labels_fwd = self.nli.batch_argmax_label(pairs_fwd)
        labels_bwd = self.nli.batch_argmax_label(pairs_bwd)

        # Union-find on bidirectional entailment.
        for k, (i, j) in enumerate(ij_index):
            if labels_fwd[k] == ENTAILMENT and labels_bwd[k] == ENTAILMENT:
                union(i, j)

        counts: dict = {}
        for i in range(N):
            r = find(i)
            counts[r] = counts.get(r, 0) + 1

        H = 0.0
        for c in counts.values():
            p = c / N
            if p > 0:
                H -= p * math.log2(p)
        return H

    def _semantic_entropy_surface(self, samples: List[str]) -> float:
        """Surface-level entropy fallback when no NLI ensemble is available."""
        def normalise(s: str) -> str:
            return re.sub(r"[^\w\s]", "", s.lower()).strip()
        counts: dict = {}
        for s in samples:
            key = normalise(s)
            counts[key] = counts.get(key, 0) + 1
        N = len(samples)
        H = 0.0
        for c in counts.values():
            p = c / N
            if p > 0:
                H -= p * math.log2(p)
        return H

    # ---- external grounding ---------------------------------------------- #

    def _retrieve_and_rerank(self, query: str, answer: str) -> List[str]:
        """Pull top-K passages and rerank down to top-M via the cross-encoder."""
        if self.passage_retriever is None:
            return []
        retrieve_k = self.config.verifier_retrieve_k
        rerank_k = self.config.verifier_rerank_k
        try:
            candidates = self.passage_retriever(query, retrieve_k)
        except Exception as exc:
            logger.warning("passage retrieval failed: %s -- returning []", exc)
            return []
        if not candidates:
            return []
        if self.reranker is None or len(candidates) <= rerank_k:
            return list(candidates[:rerank_k])
        try:
            pairs = [(f"{query} {answer}", p) for p in candidates]
            # show_progress_bar=False: sentence-transformers defaults to a
            # tqdm bar per predict() call; that adds hundreds of
            # "Batches: 100%|..." lines per query to the log and drowns
            # the useful decision traces. The rerank workload is already
            # a single forward pass; no progress feedback is needed.
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            order = np.argsort(scores)[::-1][:rerank_k]
            return [candidates[int(i)] for i in order]
        except Exception as exc:
            logger.warning("reranker failed: %s -- using retriever order", exc)
            return list(candidates[:rerank_k])

    def _score_p_ground(
        self,
        passages: List[str],
        answer: str,
    ) -> Tuple[float, float]:
        """Return (p_ground_max, p_ground_mean) over the reranked passages.

        Batched: one NLI forward per bundle covering all passages.
        """
        if not self.nli or not passages:
            return 0.5, 0.5
        try:
            pairs = [(p, answer) for p in passages]
            scores = self.nli.batch_entail_prob(pairs)
            return (
                float(np.clip(max(scores), 0.0, 1.0)),
                float(np.clip(np.mean(scores), 0.0, 1.0)),
            )
        except Exception as exc:
            logger.warning("p_ground failed: %s -- returning 0.5/0.5", exc)
            return 0.5, 0.5

    def _score_p_contra(self, passages: List[str], answer: str) -> float:
        """Max contradiction probability across passages (ensemble max per passage).

        Batched: one NLI forward per bundle covering all passages.
        """
        if not self.nli or not passages:
            return 0.0
        try:
            pairs = [(p, answer) for p in passages]
            scores = self.nli.batch_contradict_prob(pairs)
            return float(np.clip(max(scores), 0.0, 1.0))
        except Exception as exc:
            logger.warning("p_contra failed: %s -- returning 0.0", exc)
            return 0.0

    def _score_atomic(
        self,
        passages: List[str],
        answer: str,
        *,
        fallback: float,
    ) -> Tuple[List[str], List[float], float]:
        """Decompose answer to atomic facts; return (facts, per-fact p_entail, min).

        When decomposition is disabled or impossible, we return the caller's
        `fallback` so the composite's p_ground_atomic slot does not silently
        zero-out. Typically `fallback == p_ground_mean`.
        """
        if not self.enable_atomic or not self.nli or not passages:
            return [], [], float(fallback)
        try:
            self.model.eval()
            prompt = _ATOMIC_DECOMP_PROMPT.format(answer=answer)
            enc = self._tokenize(prompt)
            with torch.no_grad():
                out = self.model.generate(
                    enc["input_ids"],
                    max_new_tokens=self.config.cot_max_new_tokens,
                    do_sample=False,
                )
            decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
            facts = _parse_atomic_facts(decoded)
            if not facts:
                return [], [], float(fallback)
            # Batched: build (passage, fact) pair list for all facts x passages,
            # do ONE batched NLI forward per bundle, then reshape to per-fact
            # max-over-passages. Equivalent to the nested loop but N*M calls
            # collapse into a single batched forward per bundle.
            P = len(passages)
            pairs_flat: List[Tuple[str, str]] = [
                (p, f) for f in facts for p in passages
            ]
            flat_scores = self.nli.batch_entail_prob(pairs_flat)
            per_atom: List[float] = []
            for k in range(len(facts)):
                chunk = flat_scores[k * P:(k + 1) * P]
                per_atom.append(float(np.clip(max(chunk), 0.0, 1.0)))
            return facts, per_atom, float(min(per_atom))
        except Exception as exc:
            logger.warning("atomic decomposition failed: %s -- using fallback", exc)
            return [], [], float(fallback)

    # ====================================================================== #
    # Question-answer relevance (Branch C Goal 2)                             #
    # ====================================================================== #

    def _compute_q_a_relevance(self, query: str, answer: str) -> float:
        """Score the semantic relevance of ``answer`` to ``query`` in [0, 1].

        Runs the injected cross-encoder (e.g. BGE-reranker-v2-m3 or
        cross-encoder/ms-marco-MiniLM-L6-v2) on a single (question, answer)
        pair. When the scorer is not configured, returns 0.5 -- the neutral
        prior that leaves the composite numerically unchanged if its weight
        is zeroed via an ablation config.

        Failure modes are treated as neutral (0.5) rather than escalated
        to 0.0 / 1.0, for the same reason p_entail / p_ground_max return
        0.5 when their NLI bundle is unavailable: zero would add a false
        penalty, one would add a false reward.

        Output normalisation (sigmoid, 2026-04-21)
        ------------------------------------------
        Cross-encoders typically return **raw logits** (BGE-reranker-v2-m3
        and the default ``sentence_transformers.CrossEncoder.predict``
        both do), often in [-10, +10] for relevant/irrelevant extremes. A
        bare ``np.clip`` would truncate the distribution to {0.0, 1.0} with
        almost nothing in between, turning q_a_relevance into a binary
        step signal that destroys the composite's graded information. We
        apply a standard logistic sigmoid so logits map to probabilities,
        matching the calibrated-probability contract p_entail / p_ground_*
        already satisfy.

        Consequences for pre-calibrated scorers: if the caller injects a
        scorer whose ``predict()`` already returns [0, 1] probabilities,
        sigmoid compresses those to roughly [0.5, 0.73] -- the monotonic
        squish preserves ranking but shrinks the effective discriminating
        range. For such scorers wrap the ``predict`` output to multiply
        by ~10 before injection, or expose a raw-logit version of the
        scorer. The common BGE / MS-MARCO cases need no wrapping.

        Why this signal exists (the sample-② failure mode):
            Phase-1a Cycle 0 produced hallucinated off-topic answers that
            scored high on p_entail and p_ground_max because a retrieved
            passage happened to *support* the hallucinated text, even though
            the text did not *address the question*. Neither entailment nor
            grounding checks question↔answer relevance -- q_a_relevance
            closes that gap.
        """
        scorer = self.qa_relevance_scorer
        if scorer is None:
            return 0.5
        q = (query or "").strip()
        a = (answer or "").strip()
        if not q or not a:
            return 0.5
        try:
            # show_progress_bar=False: see comment in _retrieve_and_rerank
            # above. Single (q, a) pair; tqdm adds nothing but log noise.
            raw = scorer.predict([(q, a)], show_progress_bar=False)
        except Exception as exc:
            logger.warning(
                "q_a_relevance scorer failed (%s) -- falling back to 0.5 "
                "neutral prior.", exc,
            )
            return 0.5
        # Accept any array-like with a single scalar in position 0.
        try:
            logit = float(np.asarray(raw).reshape(-1)[0])
        except Exception:
            return 0.5
        if not math.isfinite(logit):
            return 0.5
        # Numerically-stable logistic sigmoid. Clamp the logit to +/-30
        # first so exp() cannot overflow fp64 on extreme-confidence
        # cross-encoder outputs (~e^30 is near fp64 max).
        logit = max(-30.0, min(30.0, logit))
        prob = 1.0 / (1.0 + math.exp(-logit))
        return float(np.clip(prob, 0.0, 1.0))

    # ====================================================================== #
    # Composite + decision                                                    #
    # ====================================================================== #

    def _composite(
        self,
        *,
        p_ground_mean: float,
        p_ground_atomic: float,
        s_avg: float,
        h_norm: float,
        u_internal: float,
        p_entail: float,
        q_a_relevance: float = 0.5,
    ) -> float:
        cfg = self.config
        w_pg_mean = cfg.u_stored_weight_pground_mean
        w_pg_atom = cfg.u_stored_weight_pground_atomic
        w_savg = cfg.u_stored_weight_sc
        w_hnorm = cfg.u_stored_weight_se    # applied as (1 - h_norm)
        w_uint = cfg.u_stored_weight_uinternal
        w_pent = cfg.u_stored_weight_nli
        # Branch C Goal 2: getattr keeps the composite runnable against a
        # pre-Goal-2 CAEMConfig (zero weight -> q_a_relevance contribution is
        # silenced, matches Session-42 baseline numerics exactly).
        w_qarel = getattr(cfg, "u_stored_weight_q_a_relevance", 0.0)

        u = (
            w_pg_mean * p_ground_mean
            + w_pg_atom * p_ground_atomic
            + w_savg * s_avg
            + w_hnorm * (1.0 - h_norm)
            + w_uint * u_internal
            + w_pent * p_entail
            + w_qarel * q_a_relevance
        )
        return float(np.clip(u, 0.0, 1.0))

    def _decide(
        self,
        *,
        u_stored: float,
        p_contra: float,
        p_ground_max: float,
    ) -> Tuple[str, bool]:
        """Apply the Branch C decision tree. Returns (decision, abstained).

        Note: the Session-42 contradiction veto branch
        (``p_contra >= contradiction_veto_threshold -> DISCARD``) was removed
        on 2026-04-22 because MiniCheck --- the default judge since Branch C
        --- is a unary P(supported) model that returns ``p_contra = 0.0`` by
        construction (see ``caem/verification/minicheck.py::batch_contradict_prob``).
        Under that backend the veto could never fire, so every stored entry
        carried ``p_contra = 0`` and the ``no_contradiction_veto`` ablation
        variant was a no-op. ``p_contra`` remains in the schema as a
        diagnostic under the ``roberta_nli_backend`` ablation variant (where
        the judge actually emits 3-class probabilities), but no decision
        logic reads it.
        """
        cfg = self.config
        store_thr = cfg.store_threshold
        defer_thr = cfg.defer_threshold
        abstain_pg = cfg.abstain_pground_ceiling

        if u_stored >= store_thr:
            return "STORE", False

        if u_stored >= defer_thr:
            return "DEFERRED", False

        if p_ground_max < abstain_pg:
            return "ABSTAIN", True

        return "DISCARD", False

    # ====================================================================== #
    # Internals                                                               #
    # ====================================================================== #

    def _tokenize(self, text: str) -> dict:
        """Tokenize a query string after wrapping it in ChatML user-turn.

        Mirrors ``PreRoutingConfidenceEstimator._tokenize`` (Goal 1 Phase C.2):
        instruction-tuned decoder-only models require chat-format inputs to
        produce well-calibrated predictions. The ChatML envelope puts Qwen
        into "assistant answering a user" mode so SC-sampling / u_dropout /
        h_norm generation produces coherent continuations instead of
        scattering probability across the 151k vocab.
        """
        wrapped = self._wrap_chatml(text)
        inputs = self.tokenizer(
            wrapped,
            return_tensors="pt",
            truncation=True, max_length=2048,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _wrap_chatml(self, query: str) -> str:
        """Wrap query in minimal ChatML user turn + generation prompt.

        Falls back to manual ChatML concatenation if the tokenizer lacks
        ``apply_chat_template`` (rare on modern instruction-tuned models).
        """
        messages = [{"role": "user", "content": query}]
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    "apply_chat_template failed (%s); falling back to manual ChatML.",
                    exc,
                )
        return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


__all__ = [
    "UnifiedVerifier",
    "UnifiedVerifierOutput",
]
