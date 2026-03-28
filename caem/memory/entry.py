"""
caem/memory/entry.py
====================
Data schemas for the CAEM episodic memory system.

Three distinct confidence value types are defined here — do NOT confuse them:

  PreRoutingConfidence  — computed in Stage 3, BEFORE routing, from 2 fast signals.
  PostGenerationConfidence — computed in Stage 4a, Tier 2 ONLY, from 4 signals.
  StoredConfidence      — derived from Stage 5 verification; stored with the episode.

The EpisodicEntry holds BOTH immutable content fields (never changed after storage,
because the fine-tuning dataset is derived from them — a moving target would break
training) AND mutable quality metadata (updated across cycles by retroactive
re-verification and retrieval feedback).

See: pipeline-technical.md §Data Schemas and §EpisodicMemoryStore
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core episode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodicEntry:
    """A single verified (question, reasoning, answer) triple stored in memory.

    Immutable fields
    ----------------
    These fields are set at storage time and NEVER updated afterward.
    Rationale: the fine-tuning dataset (Stage 8) is derived directly from
    these fields. Mutating them mid-cycle would mean training on a moving
    target — an auditable, stable record is mandatory.

    Mutable fields
    --------------
    Quality and usage metadata are updated across cycles by:
      - Retrieval feedback loop (u_stored, success_rate, retrieval_count)
      - Retroactive re-verification (u_stored, nli_score, sc_score, se_score,
        retroverified) — run once per improvement cycle on the full memory.
    """

    # ── Immutable content ────────────────────────────────────────────────── #
    question: str
    """Original query text."""

    reasoning_chain: str
    """Chain-of-thought steps — the core transferable knowledge."""

    answer: str
    """Final answer string."""

    embedding: np.ndarray
    """384-dim Sentence-BERT vector (all-mpnet-base-v2), L2-normalised.
    Shape: (384,).  dtype: float32."""

    storage_cycle: int
    """Which self-improvement cycle stored this episode (0, 1, or 2)."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp at storage time (set automatically if not provided)."""

    # ── Mutable quality metadata ─────────────────────────────────────────── #
    u_stored: float = 0.0
    """Combined stored confidence ∈ [0, 1].
    Initialised from Stage 5 verification output.
    Formula (full): 0.50·p_entail + 0.30·s_avg + 0.20·(1 − h_norm).
    Updated by retrieval feedback and retroactive re-verification."""

    nli_score: float = 0.0
    """P(ENTAILMENT) from RoBERTa-Large-MNLI ∈ [0, 1]."""

    sc_score: float = 0.0
    """s_avg from self-consistency (average pairwise cosine sim) ∈ [0, 1]."""

    se_score: float = 0.0
    """1 − Ĥ from semantic entropy ∈ [0, 1] (higher = less uncertain)."""

    retrieval_count: int = 0
    """Total number of times this episode has been retrieved."""

    success_rate: float = 0.0
    """Running fraction of retrievals that were accepted (not overridden)."""

    retroverified: bool = False
    """True if this episode passed retroactive re-verification in the latest cycle."""

    def __post_init__(self) -> None:
        # Validate embedding shape immediately to catch dimension bugs early.
        if self.embedding.ndim != 1 or self.embedding.shape[0] != 384:
            raise ValueError(
                f"EpisodicEntry.embedding must be shape (384,), "
                f"got {self.embedding.shape}. "
                "Check that all-mpnet-base-v2 is used (384-dim, NOT 768)."
            )
        if self.embedding.dtype != np.float32:
            self.embedding = self.embedding.astype(np.float32)

    def age(self) -> float:
        """Return the age of this episode in seconds."""
        return time.time() - self.timestamp


# ─────────────────────────────────────────────────────────────────────────────
# Confidence value types  (three distinct types — do not conflate)
# ─────────────────────────────────────────────────────────────────────────────

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
    Formula: 0.60·u_token + 0.40·(1 / (1 + c_conv)).
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

    This is an EFFICIENCY GATE — not a quality gate.
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
    """1 − H_semantic / log2(K) across K=10 samples at T=1.0.
    K and T are fixed from Farquhar et al. (2024)."""

    u_hat: float
    """Combined post-generation confidence.
    Formula uses weights from CAEMConfig.u_hat_weight_* (initially 0.25 each).
    If u_hat ≥ CAEMConfig.u_hat_accept_threshold (0.60): accept → verify.
    Else: escalate to Tier 3."""

    def should_accept(self, threshold: float = 0.60) -> bool:
        """True if û meets the acceptance threshold; False → escalate to Tier 3."""
        return self.u_hat >= threshold


@dataclass
class StoredConfidence:
    """Stage 7: derived from Stage 5 verification output; stored with the episode.

    IMPORTANT: û_stored is computed from VERIFICATION results, NOT from û (Stage 4a).
    Tier 3 answers never produce a û value, yet they still need a stored confidence.
    Verification (Stage 5) is the only pipeline stage shared by all three tiers.
    """

    p_entail: float
    """P(ENTAILMENT) from RoBERTa-Large-MNLI ∈ [0, 1]."""

    s_avg: float
    """Self-consistency average pairwise similarity ∈ [0, 1]."""

    h_norm: float
    """Normalised semantic entropy Ĥ = H / log2(K) ∈ [0, 1]."""

    u_stored: float
    """Combined stored confidence.
    Full formula: 0.50·p_entail + 0.30·s_avg + 0.20·(1 − h_norm).
    FEVER only (NLI primary): u_stored = p_entail.
    Weights are design choices (Category 2) from CAEMConfig."""


# ─────────────────────────────────────────────────────────────────────────────
# Routing decision record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """Records how a query was routed and why.

    Note on the two-mechanism design:
    CAEM uses two SEPARATE mechanisms that do NOT interact mathematically:
      1. OR-condition: u_pre < safety_u_pre_min → hard veto → Tier 3.
         This fires BEFORE any formula runs.
      2. Routing score: 0.70·s + 0.30·û_stored → Tier 1 or Tier 2.
         Only evaluated if the OR-condition did NOT fire.

    The separation is intentional: combining them into a single formula would
    allow a high û_stored to arithmetically compensate for a dangerously low
    u_pre. Memory quality and model readiness are mandatory, not tradeable.
    """

    tier: int
    """Tier dispatched to: 1 (direct retrieval), 2 (guided generation), 3 (RAG)."""

    similarity: float
    """Cosine similarity between query embedding and best memory match (s ∈ [0, 1])."""

    u_stored_retrieved: float
    """û_stored from the retrieved episode (0.0 if no episode retrieved)."""

    u_pre: float
    """Pre-routing confidence of the current query."""

    routing_score: float
    """Combined score: 0.70·s + 0.30·û_stored. Only meaningful when tier ∈ {1, 2}."""

    safety_override: bool = False
    """True when the OR-condition (u_pre < threshold) forced Tier 3."""

    retrieved_entry_id: Optional[int] = None
    """FAISS entry ID of the retrieved episode (None if memory was empty)."""
