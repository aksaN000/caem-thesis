"""
caem/memory/entry.py
====================
Data schemas for the CAEM episodic memory system.

Two distinct pre-verification confidence value types live here:

  PreRoutingConfidence     -- Stage 3, BEFORE routing, 2 fast signals.
  PostGenerationConfidence -- Stage 4a, Tier 2 ONLY, 4 signals.

Stage-5 verification output is ``UnifiedVerifierOutput`` (see
``caem.verification.verifier``). EpisodicEntry stores a flattened
copy of its nine signals -- plus p_contra, the decision string, and the
early-exit flag -- directly on the entry; the Session-42 redesign removed
the prior ``StoredConfidence`` projection dataclass.

EpisodicEntry holds BOTH immutable content fields (never changed after
storage, because the fine-tuning dataset is derived from them -- a moving
target would break training) AND mutable quality metadata (updated across
cycles by retroactive re-verification and retrieval feedback).

See: pipeline-technical.md §Data Schemas and §EpisodicMemoryStore
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# -----------------------------------------------------------------------------
# Core episode
# -----------------------------------------------------------------------------

@dataclass
class EpisodicEntry:
    """A single verified (question, reasoning, answer) triple stored in memory.

    Immutable fields
    ----------------
    These fields are set at storage time and NEVER updated afterward.
    Rationale: the fine-tuning dataset (Stage 8) is derived directly from
    these fields. Mutating them mid-cycle would mean training on a moving
    target -- an auditable, stable record is mandatory.

    Mutable fields
    --------------
    Quality and usage metadata are updated across cycles by:
      - Retrieval feedback loop (u_stored, success_rate, retrieval_count)
      - Retroactive re-verification (u_stored plus all nine signals
        u_token/u_dropout/u_internal/s_avg/h_norm/p_entail/p_ground_max/
        p_ground_mean/p_ground_atomic, p_contra, decision, early_exit_triggered,
        retroverified) -- run once per improvement cycle on the full memory.
    """

    # -- Immutable content -------------------------------------------------- #
    question: str
    """Original query text."""

    reasoning_chain: str
    """Chain-of-thought steps -- the core transferable knowledge."""

    answer: str
    """Final answer string."""

    embedding: np.ndarray
    """768-dim Sentence-BERT vector (all-mpnet-base-v2), L2-normalised.
    Shape: (768,).  dtype: float32."""

    storage_cycle: int
    """Which self-improvement cycle stored this episode (0, 1, or 2)."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp at storage time (set automatically if not provided)."""

    # -- Mutable quality metadata (Session 42 nine-signal layout) ----------- #
    u_stored: float = 0.0
    """Combined stored confidence in [0, 1] (the composite prior).
    Formula (Session 42, six-weight):
        0.30*p_ground_mean + 0.15*p_ground_atomic
        + 0.15*p_entail + 0.15*s_avg + 0.15*u_internal
        + 0.10*(1 - h_norm)
    Updated by retrieval feedback and retroactive re-verification."""

    # Internal calibration (3 signals) --------------------------------------- #
    u_token: float = 0.0
    """Mean token log-prob from generation in [0, 1] (higher = more confident)."""

    u_dropout: float = 0.0
    """MC-dropout variance across K=5 passes in [0, 1] (Gal & Ghahramani 2016)."""

    u_internal: float = 0.0
    """0.5*u_token + 0.5*(1 - u_dropout)  in [0, 1]."""

    # Sample-set signals (3) ------------------------------------------------- #
    s_avg: float = 0.0
    """Mean unique-pair SBERT cosine sim across M=3 chains in [0, 1]
    (Wang et al. 2022, self-consistency; i<j pair convention)."""

    h_norm: float = 0.0
    """Normalised semantic entropy H / log2(K) over K=10 samples in [0, 1]
    (Farquhar et al. 2024). Higher = more disagreement. Stored raw; the
    u_stored formula applies (1 - h_norm)."""

    p_entail: float = 0.0
    """Mean P(chain -> answer entailment) over M chains, ensemble-min across
    NLI models in [0, 1]."""

    # External grounding (4 signals) ----------------------------------------- #
    p_ground_max: float = 0.0
    """Top-1 reranked passage -> answer NLI entailment in [0, 1]."""

    p_ground_mean: float = 0.0
    """Mean entailment over top-3 reranked passages in [0, 1]."""

    p_ground_atomic: float = 0.0
    """Weakest-link atomic-fact entailment (min over atomic facts) in [0, 1]."""

    p_contra: float = 0.0
    """Max contradiction probability over top-3 passages in [0, 1].
    Drives the DISCARD veto when >= contradiction_veto_threshold."""

    # Decision record -------------------------------------------------------- #
    decision: str = "STORE"
    """Stage-5 decision: STORE / DEFERRED / ABSTAIN / DISCARD."""

    early_exit_triggered: bool = False
    """True iff the confabulation gate fired during verification
    (u_internal >= 0.70 AND p_ground_max <= 0.20)."""

    # Usage / feedback ------------------------------------------------------- #
    retrieval_count: int = 0
    """Total number of times this episode has been retrieved."""

    success_rate: float = 0.0
    """Running fraction of retrievals that were accepted (not overridden)."""

    retroverified: bool = False
    """True if this episode passed retroactive re-verification in the latest cycle."""

    def __post_init__(self) -> None:
        # Validate embedding shape immediately to catch dimension bugs early.
        if self.embedding.ndim != 1 or self.embedding.shape[0] != 768:
            raise ValueError(
                f"EpisodicEntry.embedding must be shape (768,), "
                f"got {self.embedding.shape}. "
                "Check that all-mpnet-base-v2 is used (768-dim, NOT 384)."
            )
        if self.embedding.dtype != np.float32:
            self.embedding = self.embedding.astype(np.float32)

    def age(self) -> float:
        """Return the age of this episode in seconds."""
        return time.time() - self.timestamp


# -----------------------------------------------------------------------------
# Confidence value types  (three distinct types -- do not conflate)
# -----------------------------------------------------------------------------

@dataclass
class PreRoutingConfidence:
    """Stage 3: computed BEFORE routing, from 2 fast signals only.

    Uses encoder hidden states + token log-probabilities.
    Does NOT involve any generation beyond a quick forward pass.
    """

    u_token: float
    """Geometric mean of per-token log-probabilities from the model's output."""

    c_conv: float
    """Raw internal convergence ratio: var(early encoder layers) / var(late layers).
    Higher c_conv = more instability = lower confidence.
    Converted to confidence via 1 / (1 + c_conv) before combination."""

    u_pre: float
    """Combined pre-routing confidence.
    Formula: 0.60*u_token + 0.40*(1 / (1 + c_conv)).
    Weights are design choices from CAEMConfig."""

    def is_safe(self, safety_threshold: float = 0.60) -> bool:
        """Return False if u_pre is below the OR-condition safety threshold.

        When False, the router MUST force Tier 3 regardless of memory similarity.
        The OR-condition is a safety-first design principle: Tier 3's computational
        cost is explicitly preferred over a confident-but-wrong Tier 1/2 answer.
        """
        return self.u_pre >= safety_threshold


@dataclass
class PostGenerationConfidence:
    """Stage 4a: computed in Tier 2 ONLY, AFTER generation, from 4 signals.

    This is an EFFICIENCY GATE -- not a quality gate.
    Purpose: catch obvious low-confidence outputs before committing to the
    full NLI + SC + SE verification pipeline (which is expensive).
    Verification (Stage 5) is the actual quality gate.

    Initial weights: 0.25/0.25/0.25/0.25 (equal).
    Projected post-calibration: ~0.20/0.20/0.20/0.40 (SE upweighted).
    Actual calibrated weights are fitted after Cycle 1 and reported in Ch. 5.
    """

    u_token: float
    """Geometric mean of per-token log-probs, recomputed on generated answer."""

    u_dropout: float
    """MC Dropout uncertainty: 1 / (1 + Var[answer_probs]) across K=5 passes.
    K is fixed from Gal & Ghahramani (2016)."""

    u_consistency: float
    """Average pairwise cosine sim across M=3 chain-of-thought generations.
    M is fixed from Wang et al. (2022)."""

    u_entropy: float
    """1 - H_semantic / log2(K) across K=10 samples at T=1.0.
    K and T are fixed from Farquhar et al. (2024)."""

    u_hat: float
    """Combined post-generation confidence.
    Formula uses weights from CAEMConfig.u_hat_weight_* (initially 0.25 each).
    If u_hat >= CAEMConfig.u_hat_accept_threshold (0.60): accept -> verify.
    Else: escalate to Tier 3."""

    def should_accept(self, threshold: float = 0.60) -> bool:
        """True if u_hat meets the acceptance threshold; False -> escalate to Tier 3."""
        return self.u_hat >= threshold


# -----------------------------------------------------------------------------
# Routing decision record
# -----------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    """Records how a query was routed and why.

    Note on the two-mechanism design:
    CAEM uses two SEPARATE mechanisms that do NOT interact mathematically:
      1. OR-condition: u_pre < safety_u_pre_min -> hard veto -> Tier 3.
         This fires BEFORE any formula runs.
      2. Routing score: 0.70*s + 0.30*u_stored -> Tier 1 or Tier 2.
         Only evaluated if the OR-condition did NOT fire.

    The separation is intentional: combining them into a single formula would
    allow a high u_stored to arithmetically compensate for a dangerously low
    u_pre. Memory quality and model readiness are mandatory, not tradeable.
    """

    tier: int
    """Tier dispatched to: 1 (direct retrieval), 2 (guided generation), 3 (RAG)."""

    similarity: float
    """Cosine similarity between query embedding and best memory match (s in [0, 1])."""

    u_stored_retrieved: float
    """u_stored from the retrieved episode (0.0 if no episode retrieved)."""

    u_pre: float
    """Pre-routing confidence of the current query."""

    routing_score: float
    """Combined score: 0.70*s + 0.30*u_stored. Only meaningful when tier in {1, 2}."""

    safety_override: bool = False
    """True when the OR-condition (u_pre < threshold) forced Tier 3."""

    retrieved_entry_id: Optional[int] = None
    """FAISS entry ID of the retrieved episode (None if memory was empty)."""
