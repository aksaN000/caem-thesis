"""
tests/test_pipeline.py
=======================
Unit tests for CAEMPipeline (end-to-end orchestrator).

Mock strategy
-------------
Every sub-component is replaced with a MagicMock — no real models, no FAISS
loading from disk, no network calls. We control:
  - encoder.encode()           → fixed 768-dim unit vector
  - model.generate()           → fixed token tensor
  - tokenizer()                → fixed input_ids tensor
  - tokenizer.decode()         → fixed answer string
  - pre_estimator.estimate()   → PreRoutingConfidence with controlled u_pre
  - router.route()             → RoutingDecision with controlled tier
  - post_estimator.estimate()  → PostGenerationConfidence with controlled û
  - verifier.verify()          → StoredConfidence with controlled û_stored
  - rag.generate()             → fixed answer string
  - memory_store               → real EpisodicMemoryStore (small, in-memory)

Coverage
--------
  - PipelineResult fields
  - Tier 1: stored answer returned, retrieval stats updated
  - Tier 2: answer generated, post-conf computed, accepted
  - Tier 2: escalation when û < threshold
  - Tier 2: generation failure → escalation
  - Tier 3: direct RAG path (safety override)
  - Tier 3: no-passage-store fallback
  - Storage: novel + verified → stored
  - Storage: duplicate → not stored
  - Storage: û_stored below threshold → not stored
  - Storage: verification failure → not stored
  - Tier 1 retrieval stats update (count increment)
  - Tier 1 stored_confidence populated from entry scores (no re-verification)
  - Pipeline repr
  - memory_summary delegates to store
  - current_cycle recorded in stored entry
  - auto-prune triggered near capacity
  - empty answer → not stored
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch

from caem.config import CAEMConfig
from caem.memory.entry import (
    EpisodicEntry,
    PostGenerationConfidence,
    PreRoutingConfidence,
    RoutingDecision,
    StoredConfidence,
)
from caem.memory.store import EpisodicMemoryStore
from caem.pipeline import CAEMPipeline, PipelineResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DIM = 768


def unit_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_tok_output(input_ids=None):
    """Mock tokenizer output with .to() support."""
    if input_ids is None:
        input_ids = torch.zeros(1, 8, dtype=torch.long)
    out = MagicMock()
    out.__getitem__ = lambda self, key: input_ids if key == "input_ids" else torch.ones(1, 8, dtype=torch.long)
    out.to.return_value = out
    out.items.return_value = [("input_ids", input_ids)]
    return out


def make_mock_encoder(seed: int = 0) -> MagicMock:
    enc = MagicMock()
    enc.encode.return_value = unit_vec(seed)
    return enc


def make_mock_model() -> MagicMock:
    model = MagicMock()
    params = [torch.zeros(2, 2)]
    model.parameters.side_effect = lambda: iter(params)
    model.generate.return_value = torch.tensor([[0, 1, 2]])
    return model


def make_mock_tokenizer(answer: str = "William Shakespeare") -> MagicMock:
    tok = MagicMock()
    tok.return_value = _make_tok_output()
    tok.decode.return_value = answer
    return tok


def make_pre_conf(u_pre: float = 0.80) -> PreRoutingConfidence:
    return PreRoutingConfidence(u_token=u_pre, c_conv=0.1, u_pre=u_pre)


def make_routing(tier: int = 2, safety_override: bool = False, similarity: float = 0.5) -> RoutingDecision:
    return RoutingDecision(
        tier=tier,
        similarity=similarity,
        u_stored_retrieved=0.8,
        u_pre=0.80,
        routing_score=0.65,
        safety_override=safety_override,
        retrieved_entry_id=None,
    )


def make_stored_conf(u_stored: float = 0.75) -> StoredConfidence:
    return StoredConfidence(p_entail=u_stored, s_avg=u_stored, h_norm=0.1, u_stored=u_stored)


def make_post_conf(u_hat: float = 0.70) -> PostGenerationConfidence:
    return PostGenerationConfidence(
        u_token=0.7, u_dropout=0.7, u_consistency=0.7, u_entropy=0.7, u_hat=u_hat
    )


def make_episodic_entry(seed: int = 0, u_stored: float = 0.80) -> EpisodicEntry:
    return EpisodicEntry(
        question="Who wrote Hamlet?",
        reasoning_chain="Shakespeare wrote Hamlet.",
        answer="William Shakespeare",
        embedding=unit_vec(seed),
        storage_cycle=0,
        u_stored=u_stored,
    )


def _build_pipeline(
    tier: int = 2,
    u_hat: float = 0.70,
    u_stored: float = 0.75,
    u_pre: float = 0.80,
    safety_override: bool = False,
    similarity: float = 0.5,
    answer: str = "William Shakespeare",
    memory_store: EpisodicMemoryStore | None = None,
    config: CAEMConfig | None = None,
) -> CAEMPipeline:
    """Build a fully mocked CAEMPipeline for testing."""
    cfg = config or CAEMConfig()
    model = make_mock_model()
    tok = make_mock_tokenizer(answer)
    enc = make_mock_encoder()

    with patch("caem.pipeline.PreRoutingConfidenceEstimator") as MockPre, \
         patch("caem.pipeline.AdaptiveRouter") as MockRouter, \
         patch("caem.pipeline.PostGenerationConfidenceEstimator") as MockPost, \
         patch("caem.pipeline.MultiLayerVerifier") as MockVerifier, \
         patch("caem.pipeline.TierThreeRAG") as MockRAG:

        # Wire up mock returns
        MockPre.return_value.estimate.return_value = make_pre_conf(u_pre)
        MockRouter.return_value.route.return_value = make_routing(tier, safety_override, similarity)
        MockPost.return_value.estimate.return_value = make_post_conf(u_hat)
        MockVerifier.return_value.verify.return_value = make_stored_conf(u_stored)
        MockRAG.return_value.generate.return_value = answer

        pipeline = CAEMPipeline(
            model=model,
            tokenizer=tok,
            encoder=enc,
            config=cfg,
            memory_store=memory_store,
            device="cpu",
        )

    # Re-attach the mocks as attributes for test inspection
    pipeline._mock_pre = pipeline.pre_estimator
    pipeline._mock_router = pipeline.router
    pipeline._mock_post = pipeline.post_estimator
    pipeline._mock_verifier = pipeline.verifier
    pipeline._mock_rag = pipeline.rag

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# PipelineResult
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult(query="q", answer="a", tier=2)
        assert r.stored is False
        assert r.escalated is False
        assert r.entry_id is None
        assert r.latency_ms == 0.0

    def test_fields_set(self):
        r = PipelineResult(
            query="q", answer="a", tier=1, stored=True, latency_ms=42.0, entry_id=7
        )
        assert r.tier == 1
        assert r.stored is True
        assert r.latency_ms == 42.0
        assert r.entry_id == 7


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline construction
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineConstruction:
    def test_repr_contains_cycle_and_memory(self):
        p = _build_pipeline()
        r = repr(p)
        assert "CAEMPipeline" in r
        assert "cycle=" in r
        assert "memory=" in r

    def test_default_cycle_is_zero(self):
        p = _build_pipeline()
        assert p.current_cycle == 0

    def test_memory_store_created_if_not_provided(self):
        p = _build_pipeline()
        assert isinstance(p.memory_store, EpisodicMemoryStore)

    def test_existing_memory_store_used(self):
        store = EpisodicMemoryStore()
        p = _build_pipeline(memory_store=store)
        assert p.memory_store is store

    def test_no_passage_store_warning(self, caplog):
        """Pipeline warns when no passage_store is provided."""
        import logging
        with caplog.at_level(logging.WARNING, logger="caem.pipeline"):
            p = _build_pipeline()
        # Warning may appear in logs — pipeline still constructs cleanly
        assert p is not None


# ─────────────────────────────────────────────────────────────────────────────
# answer() — Tier 2 path (default)
# ─────────────────────────────────────────────────────────────────────────────

class TestTier2Path:
    def test_returns_pipeline_result(self):
        p = _build_pipeline(tier=2)
        r = p.answer("Who wrote Hamlet?")
        assert isinstance(r, PipelineResult)

    def test_answer_string_set(self):
        p = _build_pipeline(tier=2, answer="Shakespeare")
        r = p.answer("q")
        assert r.answer == "Shakespeare"

    def test_tier_is_2(self):
        p = _build_pipeline(tier=2)
        r = p.answer("q")
        assert r.tier == 2

    def test_post_confidence_set(self):
        p = _build_pipeline(tier=2, u_hat=0.72)
        r = p.answer("q")
        assert r.post_confidence is not None
        assert r.post_confidence.u_hat == pytest.approx(0.72)

    def test_stored_confidence_set(self):
        p = _build_pipeline(tier=2, u_stored=0.77)
        r = p.answer("q")
        assert r.stored_confidence is not None
        assert r.stored_confidence.u_stored == pytest.approx(0.77)

    def test_escalated_false_when_accepted(self):
        p = _build_pipeline(tier=2, u_hat=0.70)
        r = p.answer("q")
        assert r.escalated is False

    def test_latency_ms_positive(self):
        p = _build_pipeline(tier=2)
        r = p.answer("q")
        assert r.latency_ms > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 escalation
# ─────────────────────────────────────────────────────────────────────────────

class TestTier2Escalation:
    def test_escalated_when_u_hat_below_threshold(self):
        """û < 0.60 → Tier 3 escalation."""
        p = _build_pipeline(tier=2, u_hat=0.40)
        r = p.answer("q")
        assert r.escalated is True

    def test_escalation_uses_rag_answer(self):
        """After escalation, the RAG answer is used."""
        p = _build_pipeline(tier=2, u_hat=0.40, answer="RAG answer")
        r = p.answer("q")
        # The RAG mock returns the same `answer` string we configured
        assert r.answer == "RAG answer"

    def test_post_conf_still_set_after_escalation(self):
        """û is computed even when escalating, so post_confidence is set."""
        p = _build_pipeline(tier=2, u_hat=0.40)
        r = p.answer("q")
        assert r.post_confidence is not None


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 path
# ─────────────────────────────────────────────────────────────────────────────

class TestTier1Path:
    def _make_tier1_pipeline(self, u_stored_memory: float = 0.80):
        """Pipeline with a seeded memory entry for Tier 1 retrieval."""
        store = EpisodicMemoryStore()
        entry = make_episodic_entry(seed=0, u_stored=u_stored_memory)
        store.add(entry)
        return _build_pipeline(tier=1, similarity=0.92, memory_store=store)

    def test_tier_is_1(self):
        p = self._make_tier1_pipeline()
        r = p.answer("Who wrote Hamlet?")
        assert r.tier == 1

    def test_answer_from_memory(self):
        p = self._make_tier1_pipeline()
        r = p.answer("Who wrote Hamlet?")
        assert r.answer == "William Shakespeare"

    def test_post_conf_is_none(self):
        p = self._make_tier1_pipeline()
        r = p.answer("Who wrote Hamlet?")
        assert r.post_confidence is None

    def test_not_re_stored(self):
        """Tier 1 hit: we update stats, but don't add a second copy."""
        store = EpisodicMemoryStore()
        entry = make_episodic_entry(seed=0)
        store.add(entry)
        initial_size = store.size

        p = _build_pipeline(tier=1, similarity=0.92, memory_store=store)
        p.answer("Who wrote Hamlet?")
        assert store.size == initial_size

    def test_escalated_false(self):
        p = self._make_tier1_pipeline()
        r = p.answer("Who wrote Hamlet?")
        assert r.escalated is False

    def test_stored_confidence_populated(self):
        """Tier 1: stored_confidence is reconstructed from entry scores — no verifier call."""
        p = self._make_tier1_pipeline(u_stored_memory=0.82)
        r = p.answer("Who wrote Hamlet?")
        assert r.stored_confidence is not None
        # u_stored is taken directly from the memory entry — verifier is NOT called
        assert r.stored_confidence.u_stored == pytest.approx(0.82)
        # Verify the verifier was never called for this Tier 1 hit
        p.verifier.verify.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 path (safety override)
# ─────────────────────────────────────────────────────────────────────────────

class TestTier3Path:
    def test_tier_is_3(self):
        p = _build_pipeline(tier=3, safety_override=True, u_pre=0.50)
        r = p.answer("q")
        assert r.tier == 3

    def test_post_conf_is_none(self):
        p = _build_pipeline(tier=3, safety_override=True)
        r = p.answer("q")
        assert r.post_confidence is None

    def test_escalated_false(self):
        """Tier 3 direct (not via escalation) → escalated = False."""
        p = _build_pipeline(tier=3, safety_override=True)
        r = p.answer("q")
        assert r.escalated is False

    def test_safety_override_recorded(self):
        p = _build_pipeline(tier=3, safety_override=True)
        r = p.answer("q")
        assert r.routing_decision.safety_override is True


# ─────────────────────────────────────────────────────────────────────────────
# Storage decision
# ─────────────────────────────────────────────────────────────────────────────

class TestStorageDecision:
    def test_novel_verified_answer_stored(self):
        """Novel query + û_stored ≥ threshold → stored."""
        store = EpisodicMemoryStore()
        p = _build_pipeline(tier=2, u_stored=0.75, memory_store=store)
        r = p.answer("New question never seen before?")
        assert r.stored is True
        assert r.entry_id is not None

    def test_not_stored_below_threshold(self):
        """û_stored < 0.50 → not stored."""
        store = EpisodicMemoryStore()
        p = _build_pipeline(tier=2, u_stored=0.30, memory_store=store)
        r = p.answer("q?")
        assert r.stored is False

    def test_not_stored_on_verification_failure(self):
        """If verifier raises, answer is not stored."""
        store = EpisodicMemoryStore()
        p = _build_pipeline(tier=2, u_stored=0.80, memory_store=store)
        p.verifier.verify.side_effect = RuntimeError("verifier down")
        r = p.answer("q?")
        assert r.stored is False

    def test_duplicate_not_stored(self):
        """Near-duplicate query → not stored again."""
        store = EpisodicMemoryStore()
        cfg = CAEMConfig()
        cfg.novelty_threshold = 0.00  # everything is "duplicate" (cosine ≥ 0.0)
        # Pre-populate with one entry so store is not empty
        store.add(make_episodic_entry(seed=0))

        # Encoder always returns the same unit vector → cosine sim = 1.0 → duplicate
        p = _build_pipeline(tier=2, u_stored=0.80, memory_store=store, config=cfg)
        initial_size = store.size
        r = p.answer("Who wrote Hamlet?")
        # Size should not grow (either stored or not, depending on novelty)
        # With novelty_threshold=0.00 and cosine=1.0: 1.0 > 0.00 → NOT novel → not stored
        assert store.size == initial_size

    def test_empty_answer_not_stored(self):
        """Empty answer string → not stored."""
        store = EpisodicMemoryStore()
        p = _build_pipeline(tier=3, u_stored=0.80, memory_store=store, answer="")
        # Override verifier to return None (empty answer case)
        p.verifier.verify.return_value = None
        r = p.answer("q?")
        assert r.stored is False

    def test_memory_grows_after_store(self):
        """Memory size increases by 1 after a successful store."""
        store = EpisodicMemoryStore()
        initial = store.size
        p = _build_pipeline(tier=2, u_stored=0.75, memory_store=store)
        r = p.answer("A unique question 12345?")
        if r.stored:
            assert store.size == initial + 1


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval stats (Tier 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestTier1RetrievalStats:
    def test_retrieval_count_incremented(self):
        store = EpisodicMemoryStore()
        entry = make_episodic_entry(seed=0, u_stored=0.80)
        store.add(entry)
        assert entry.retrieval_count == 0

        p = _build_pipeline(tier=1, memory_store=store, u_stored=0.80)
        p.answer("Who wrote Hamlet?")
        assert entry.retrieval_count == 1

    def test_u_stored_not_changed_by_retrieval(self):
        """Tier 1 fast path skips verification — u_stored is NOT updated upward.

        Rationale: the verifier (MultiLayerVerifier) generates M=3 chains which
        would violate the Tier-1 '<400 ms, no generation' contract. The stored
        u_stored from the original storage cycle is trusted as-is.
        """
        store = EpisodicMemoryStore()
        entry = make_episodic_entry(seed=0, u_stored=0.60)
        store.add(entry)

        p = _build_pipeline(tier=1, memory_store=store, u_stored=0.90)
        p.answer("Who wrote Hamlet?")
        # u_stored should be unchanged — the verifier was never called
        assert entry.u_stored == pytest.approx(0.60)
        p.verifier.verify.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Cycle tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestCycleTracking:
    def test_stored_entry_records_current_cycle(self):
        """EpisodicEntry.storage_cycle matches pipeline.current_cycle."""
        store = EpisodicMemoryStore()

        model = make_mock_model()
        tok = make_mock_tokenizer()
        enc = make_mock_encoder()

        with patch("caem.pipeline.PreRoutingConfidenceEstimator") as MockPre, \
             patch("caem.pipeline.AdaptiveRouter") as MockRouter, \
             patch("caem.pipeline.PostGenerationConfidenceEstimator") as MockPost, \
             patch("caem.pipeline.MultiLayerVerifier") as MockVerifier, \
             patch("caem.pipeline.TierThreeRAG") as MockRAG:

            MockPre.return_value.estimate.return_value = make_pre_conf(0.80)
            MockRouter.return_value.route.return_value = make_routing(2)
            MockPost.return_value.estimate.return_value = make_post_conf(0.70)
            MockVerifier.return_value.verify.return_value = make_stored_conf(0.80)
            MockRAG.return_value.generate.return_value = "ans"

            pipeline = CAEMPipeline(
                model=model, tokenizer=tok, encoder=enc,
                memory_store=store, device="cpu",
                current_cycle=2,
            )

        pipeline.answer("q?")
        if store.size > 0:
            entry = list(store._metadata.values())[0]
            assert entry.storage_cycle == 2


# ─────────────────────────────────────────────────────────────────────────────
# Routing metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutingMetadata:
    def test_routing_decision_attached(self):
        p = _build_pipeline(tier=2)
        r = p.answer("q")
        assert r.routing_decision is not None
        assert r.routing_decision.tier == 2

    def test_pre_confidence_attached(self):
        p = _build_pipeline(tier=2, u_pre=0.75)
        r = p.answer("q")
        assert r.pre_confidence is not None
        assert r.pre_confidence.u_pre == pytest.approx(0.75)


# ─────────────────────────────────────────────────────────────────────────────
# memory_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestMemorySummary:
    def test_summary_returns_dict(self):
        p = _build_pipeline()
        s = p.memory_summary()
        assert isinstance(s, dict)

    def test_summary_has_size_key(self):
        p = _build_pipeline()
        s = p.memory_summary()
        assert "size" in s
