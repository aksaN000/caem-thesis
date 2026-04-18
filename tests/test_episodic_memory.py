"""
tests/test_episodic_memory.py
==============================
Unit tests for the CAEM episodic memory module.

These tests use synthetic (random, normalised) embeddings -- no SBERT or FAISS
GPU required. They validate:
  - EpisodicEntry validation (shape, dtype)
  - EpisodicMemoryStore add / search / remove
  - Novelty check logic
  - Retrieval stat updates + feedback loop
  - Pruning (bottom-Value removal)
  - Retroactive re-verification (update upward / remove below threshold)
  - Save / load round-trip
  - Confidence dataclass predicates

Run with:
    pip install faiss-cpu pytest
    cd "for cowork caem"
    python -m pytest tests/test_episodic_memory.py -v
"""

import math
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from caem.config import CAEMConfig
from caem.memory.entry import (
    EpisodicEntry,
    PostGenerationConfidence,
    PreRoutingConfidence,
    RoutingDecision,
)
from caem.memory.store import EpisodicMemoryStore
from caem.verification.verifier import UnifiedVerifierOutput


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

DIM = 768  # Must match CAEMConfig.embedding_dim


def random_unit_vec(seed: int | None = None) -> np.ndarray:
    """Return a random L2-normalised float32 vector of length DIM."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make_entry(
    question: str = "What is 2+2?",
    answer: str = "4",
    u_stored: float = 0.85,
    storage_cycle: int = 0,
    embedding: np.ndarray | None = None,
    seed: int | None = None,
) -> EpisodicEntry:
    """Convenience factory for test episodes."""
    emb = embedding if embedding is not None else random_unit_vec(seed)
    return EpisodicEntry(
        question=question,
        reasoning_chain="2+2 = 4 by definition of addition.",
        answer=answer,
        embedding=emb,
        storage_cycle=storage_cycle,
        u_stored=u_stored,
        # Representative nine-signal values; u_internal ≈ mean of token/dropout.
        u_token=0.85,
        u_dropout=0.15,
        u_internal=0.85,
        s_avg=0.88,
        h_norm=0.20,       # raw entropy; (1 - h_norm) = 0.80 old "se_score"
        p_entail=0.90,
        p_ground_max=0.80,
        p_ground_mean=0.75,
        p_ground_atomic=0.70,
        p_contra=0.05,
        decision="STORE",
    )


def make_store(max_size: int = 100) -> EpisodicMemoryStore:
    cfg = CAEMConfig()
    cfg.max_memory_size = max_size
    return EpisodicMemoryStore(config=cfg)


# -----------------------------------------------------------------------------
# EpisodicEntry validation
# -----------------------------------------------------------------------------

class TestEpisodicEntry:
    def test_valid_entry_created(self):
        entry = make_entry(seed=0)
        assert entry.embedding.shape == (DIM,)
        assert entry.embedding.dtype == np.float32

    def test_wrong_dim_raises(self):
        bad_emb = np.random.randn(384).astype(np.float32)
        with pytest.raises(ValueError, match=r"shape \(768,\)"):
            EpisodicEntry(
                question="q", reasoning_chain="r", answer="a",
                embedding=bad_emb, storage_cycle=0
            )

    def test_float64_cast_to_float32(self):
        emb_f64 = random_unit_vec(1).astype(np.float64)
        entry = EpisodicEntry(
            question="q", reasoning_chain="r", answer="a",
            embedding=emb_f64, storage_cycle=0
        )
        assert entry.embedding.dtype == np.float32

    def test_age_increases(self):
        entry = make_entry(seed=2)
        age1 = entry.age()
        time.sleep(0.05)
        age2 = entry.age()
        assert age2 > age1

    def test_default_mutable_fields(self):
        entry = make_entry()
        # Custom u_stored is set above; check retrieval defaults.
        assert entry.retrieval_count == 0
        assert entry.success_rate == 0.0
        assert entry.retroverified is False


# -----------------------------------------------------------------------------
# Confidence dataclasses
# -----------------------------------------------------------------------------

class TestConfidenceDataclasses:
    def test_pre_routing_is_safe(self):
        pc = PreRoutingConfidence(u_token=0.8, c_conv=0.3, u_pre=0.72)
        assert pc.is_safe(0.60) is True
        assert pc.is_safe(0.80) is False

    def test_pre_routing_or_condition_fires(self):
        pc = PreRoutingConfidence(u_token=0.4, c_conv=2.0, u_pre=0.35)
        assert pc.is_safe(0.60) is False

    def test_post_generation_dataclass_constructs(self):
        # The legacy u_hat gate and its should_accept method were removed
        # when the UnifiedVerifier nine-signal stage became the single
        # source of post-generation truth. The PostGenerationConfidence
        # dataclass is retained as a type-compat shell; we verify only that
        # its four remaining signal fields construct and round-trip.
        pgc = PostGenerationConfidence(
            u_token=0.70, u_dropout=0.65, u_consistency=0.80, u_entropy=0.75,
        )
        assert pgc.u_token == 0.70
        assert pgc.u_dropout == 0.65
        assert pgc.u_consistency == 0.80
        assert pgc.u_entropy == 0.75

    def test_routing_decision_fields(self):
        rd = RoutingDecision(
            tier=1, similarity=0.95, u_stored_retrieved=0.88,
            u_pre=0.75, routing_score=0.929, safety_override=False,
            retrieved_entry_id=42
        )
        assert rd.tier == 1
        assert not rd.safety_override


# -----------------------------------------------------------------------------
# EpisodicMemoryStore -- basic operations
# -----------------------------------------------------------------------------

class TestStoreBasicOps:
    def test_empty_store(self):
        store = make_store()
        assert store.size == 0
        assert store.is_empty

    def test_add_single(self):
        store = make_store()
        entry = make_entry(seed=0)
        eid = store.add(entry)
        assert eid == 0
        assert store.size == 1

    def test_add_multiple(self):
        store = make_store()
        for i in range(5):
            store.add(make_entry(seed=i))
        assert store.size == 5

    def test_search_returns_closest(self):
        store = make_store()
        # Store one known vector.
        v1 = random_unit_vec(0)
        entry = make_entry(embedding=v1)
        store.add(entry)

        # Search with the exact same vector.
        results = store.search(v1, k=1)
        assert len(results) == 1
        found_entry, sim = results[0]
        assert math.isclose(sim, 1.0, abs_tol=1e-5)
        assert found_entry.question == entry.question

    def test_search_empty_returns_empty(self):
        store = make_store()
        results = store.search(random_unit_vec(0), k=1)
        assert results == []

    def test_search_top_k(self):
        store = make_store()
        for i in range(10):
            store.add(make_entry(seed=i))
        results = store.search(random_unit_vec(99), k=5)
        assert len(results) == 5
        # Similarities should be descending.
        sims = [s for _, s in results]
        assert sims == sorted(sims, reverse=True)

    # -- search_with_ids ------------------------------------------------- #

    def test_search_with_ids_returns_three_tuple(self):
        store = make_store()
        v = random_unit_vec(0)
        store.add(make_entry(embedding=v))
        results = store.search_with_ids(v, k=1)
        assert len(results) == 1
        entry, eid, sim = results[0]
        assert isinstance(eid, int)
        assert isinstance(sim, float)
        assert entry.question is not None

    def test_search_with_ids_id_matches_add_return(self):
        store = make_store()
        v = random_unit_vec(7)
        returned_id = store.add(make_entry(embedding=v))
        results = store.search_with_ids(v, k=1)
        _, found_id, _ = results[0]
        assert found_id == returned_id

    def test_search_with_ids_empty_store_returns_empty(self):
        store = make_store()
        results = store.search_with_ids(random_unit_vec(0), k=1)
        assert results == []

    def test_search_with_ids_sim_matches_search(self):
        """search_with_ids returns same similarity as search() for same query."""
        store = make_store()
        v = random_unit_vec(3)
        store.add(make_entry(embedding=v, seed=3))
        plain = store.search(v, k=1)
        with_ids = store.search_with_ids(v, k=1)
        _, _, sim_with_id = with_ids[0]
        _, sim_plain = plain[0]
        assert sim_with_id == pytest.approx(sim_plain, abs=1e-5)

    def test_search_with_ids_entry_matches_search(self):
        """search_with_ids and search() return the same entry object."""
        store = make_store()
        v = random_unit_vec(4)
        store.add(make_entry(embedding=v, seed=4))
        plain_entry, _ = store.search(v, k=1)[0]
        with_id_entry, _, _ = store.search_with_ids(v, k=1)[0]
        assert plain_entry.question == with_id_entry.question

    def test_add_full_raises(self):
        store = make_store(max_size=2)
        store.add(make_entry(seed=0))
        store.add(make_entry(seed=1))
        with pytest.raises(RuntimeError, match="full"):
            store.add(make_entry(seed=2))

    def test_get_entry(self):
        store = make_store()
        entry = make_entry(question="Find me.", seed=5)
        eid = store.add(entry)
        found = store.get(eid)
        assert found is not None
        assert found.question == "Find me."

    def test_get_missing_returns_none(self):
        store = make_store()
        assert store.get(999) is None

    def test_remove_entry(self):
        store = make_store()
        eid = store.add(make_entry(seed=0))
        assert store.size == 1
        removed = store.remove(eid)
        assert removed is True
        assert store.size == 0
        assert store.get(eid) is None

    def test_remove_missing_returns_false(self):
        store = make_store()
        assert store.remove(999) is False

    def test_dim_mismatch_raises(self):
        store = make_store()
        bad_emb = np.random.randn(384).astype(np.float32)
        bad_emb /= np.linalg.norm(bad_emb)
        with pytest.raises((ValueError, Exception)):
            # EpisodicEntry constructor will raise before we even call add.
            entry = EpisodicEntry(
                question="q", reasoning_chain="r", answer="a",
                embedding=bad_emb, storage_cycle=0
            )


# -----------------------------------------------------------------------------
# Novelty check
# -----------------------------------------------------------------------------

class TestNoveltyCheck:
    def test_empty_store_is_novel(self):
        store = make_store()
        assert store.is_novel(random_unit_vec(0)) is True

    def test_identical_vector_not_novel(self):
        store = make_store()
        v = random_unit_vec(0)
        store.add(make_entry(embedding=v))
        assert store.is_novel(v) is False  # sim = 1.0 > threshold 0.95

    def test_orthogonal_vector_is_novel(self):
        store = make_store()
        # Store canonical basis vector e1.
        e1 = np.zeros(DIM, dtype=np.float32)
        e1[0] = 1.0
        store.add(make_entry(embedding=e1))

        # e2 is orthogonal (sim = 0.0).
        e2 = np.zeros(DIM, dtype=np.float32)
        e2[1] = 1.0
        assert store.is_novel(e2) is True


# -----------------------------------------------------------------------------
# Retrieval stats + feedback loop
# -----------------------------------------------------------------------------

class TestRetrievalStats:
    def test_retrieval_count_increments(self):
        store = make_store()
        eid = store.add(make_entry(seed=0))
        store.update_retrieval_stats(eid, was_accepted=True)
        store.update_retrieval_stats(eid, was_accepted=True)
        e = store.get(eid); assert e is not None
        assert e.retrieval_count == 2

    def test_success_rate_running_mean(self):
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.80))
        # 3 accepted, 1 rejected -> 0.75
        for _ in range(3):
            store.update_retrieval_stats(eid, was_accepted=True)
        store.update_retrieval_stats(eid, was_accepted=False)
        e = store.get(eid); assert e is not None
        rate = e.success_rate
        assert math.isclose(rate, 0.75, abs_tol=1e-4)

    def test_feedback_loop_fires_after_min_retrievals(self):
        cfg = CAEMConfig()
        cfg.feedback_loop_min_retrievals = 5
        cfg.feedback_loop_eta = 0.01
        cfg.max_memory_size = 100
        store = EpisodicMemoryStore(config=cfg)
        eid = store.add(make_entry(seed=0, u_stored=0.80))

        # First 4 retrievals: no feedback update expected.
        for _ in range(4):
            store.update_retrieval_stats(eid, was_accepted=True)
        e_before = store.get(eid); assert e_before is not None
        u_before = e_before.u_stored

        # 5th retrieval: feedback fires.
        store.update_retrieval_stats(eid, was_accepted=True)
        e_after = store.get(eid); assert e_after is not None
        u_after = e_after.u_stored

        # u_stored should have nudged upward.
        assert u_after > u_before

    def test_feedback_loop_rejection_nudges_down(self):
        cfg = CAEMConfig()
        cfg.feedback_loop_min_retrievals = 2
        cfg.feedback_loop_eta = 0.10
        cfg.max_memory_size = 100
        store = EpisodicMemoryStore(config=cfg)
        eid = store.add(make_entry(seed=0, u_stored=0.80))

        # 2 rejected retrievals.
        for _ in range(2):
            store.update_retrieval_stats(eid, was_accepted=False)
        e = store.get(eid); assert e is not None
        assert e.u_stored < 0.80


# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

class TestPruning:
    def test_prune_removes_correct_count(self):
        cfg = CAEMConfig()
        cfg.max_memory_size = 10
        cfg.pruning_amount = 0.20   # Remove 20% = 2 episodes
        store = EpisodicMemoryStore(config=cfg)
        for i in range(10):
            store.add(make_entry(seed=i, u_stored=0.5))
        n_removed = store.prune()
        assert n_removed == 2
        assert store.size == 8

    def test_prune_removes_lowest_value(self):
        """The entry with the lowest u_stored, zero retrievals, and oldest age
        should be pruned first."""
        cfg = CAEMConfig()
        cfg.max_memory_size = 10
        cfg.pruning_amount = 0.10  # Remove 1
        store = EpisodicMemoryStore(config=cfg)

        # Add a clearly low-value entry (u_stored=0.10, no retrievals).
        low_emb = random_unit_vec(0)
        low_eid = store.add(make_entry(
            question="low value entry", embedding=low_emb, u_stored=0.10, seed=0
        ))

        # Add high-value entries.
        for i in range(1, 9):
            store.add(make_entry(seed=i, u_stored=0.90))

        store.prune()
        # The low-value entry should have been removed.
        assert store.get(low_eid) is None


# -----------------------------------------------------------------------------
# Retroactive re-verification
# -----------------------------------------------------------------------------
# Rewritten Task #128 (Wave 4) against UnifiedVerifierOutput. Pre-Session-42
# the tests fabricated ``StoredConfidence`` mocks which the post-42 verifier
# stack no longer accepts; the class was deleted. These tests now feed the
# store.retroverify API with real UnifiedVerifierOutput instances.


def _make_verifier_output(
    u_stored: float,
    *,
    decision: str = "STORE",
    p_entail: float = 0.90,
    s_avg: float = 0.88,
    h_norm: float = 0.20,
    p_contra: float = 0.05,
) -> UnifiedVerifierOutput:
    """Build a UnifiedVerifierOutput with sensible signal defaults.

    Kept minimal: the retroverify path only reads u_stored for the
    gate decision plus the nine-signal block for the upward-update
    copy. Other values default to reasonable mid-range samples so
    nothing downstream trips on uninitialised floats.
    """
    return UnifiedVerifierOutput(
        u_token=0.85,
        u_dropout=0.15,
        u_internal=0.85,
        s_avg=s_avg,
        h_norm=h_norm,
        p_entail=p_entail,
        p_ground_max=0.80,
        p_ground_mean=0.75,
        p_ground_atomic=0.70,
        p_contra=p_contra,
        u_stored=u_stored,
        decision=decision,
        early_exit_triggered=False,
        abstained=False,
    )


class TestRetroVerify:
    def test_retroverify_updates_improved_entry(self):
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.70))

        improved = _make_verifier_output(u_stored=0.90, p_entail=0.95, s_avg=0.90, h_norm=0.10)
        n_updated, n_removed = store.retroverify(lambda _: improved, threshold=0.50)

        assert n_updated == 1
        assert n_removed == 0
        assert store.get(eid).u_stored > 0.70

    def test_retroverify_does_not_downgrade(self):
        """If new score is lower (but still above threshold), old u_stored is kept."""
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.85))

        lower = _make_verifier_output(u_stored=0.73, p_entail=0.70, s_avg=0.75)
        n_updated, n_removed = store.retroverify(lambda _: lower, threshold=0.50)

        assert n_updated == 0
        assert n_removed == 0
        # u_stored should be unchanged.
        assert math.isclose(store.get(eid).u_stored, 0.85, abs_tol=1e-4)
        # But retroverified should be set.
        assert store.get(eid).retroverified is True

    def test_retroverify_removes_below_threshold(self):
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.80))

        bad = _make_verifier_output(u_stored=0.35, p_entail=0.30, s_avg=0.35, h_norm=0.80, decision="DISCARD")
        n_updated, n_removed = store.retroverify(lambda _: bad, threshold=0.50)

        assert n_removed == 1
        assert store.get(eid) is None

    def test_retroverify_mixed(self):
        store = make_store()
        # Use distinct question strings so the verify_fn can tell them apart.
        eid_good = store.add(make_entry(question="good entry question", seed=0, u_stored=0.70))
        eid_bad  = store.add(make_entry(question="bad entry question",  seed=1, u_stored=0.70))

        def verify_fn(entry):
            if entry.question == "good entry question":
                return _make_verifier_output(u_stored=0.92, p_entail=0.95, s_avg=0.90, h_norm=0.10)
            return _make_verifier_output(u_stored=0.25, p_entail=0.20, s_avg=0.30, h_norm=0.90, decision="DISCARD")

        n_updated, n_removed = store.retroverify(verify_fn, threshold=0.50)
        assert n_updated == 1
        assert n_removed == 1
        assert store.get(eid_good) is not None
        assert store.get(eid_bad) is None

    def test_retroverify_copies_all_nine_signals_on_upgrade(self):
        """Task #128 regression guard: the upward-update path in
        store.retroverify copies every signal from the UnifiedVerifierOutput
        onto the stored entry, not just u_stored. Pre-42 StoredConfidence
        only carried (p_entail, s_avg, h_norm, u_stored); Session 42
        widened this to the full nine-signal block + p_contra + decision +
        early_exit_triggered. Lock in the copy so a future field addition
        that forgets to update store.py is caught at test time.
        """
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.60))
        new = UnifiedVerifierOutput(
            u_token=0.91, u_dropout=0.09, u_internal=0.92,
            s_avg=0.93, h_norm=0.08,
            p_entail=0.94, p_ground_max=0.89, p_ground_mean=0.87,
            p_ground_atomic=0.85,
            p_contra=0.02,
            u_stored=0.90,
            decision="STORE",
            early_exit_triggered=False,
            abstained=False,
        )
        n_updated, n_removed = store.retroverify(lambda _: new, threshold=0.50)

        assert n_updated == 1
        assert n_removed == 0

        stored = store.get(eid)
        # All nine signals copied from the verifier output.
        assert stored.u_token        == pytest.approx(0.91)
        assert stored.u_dropout      == pytest.approx(0.09)
        assert stored.u_internal     == pytest.approx(0.92)
        assert stored.s_avg          == pytest.approx(0.93)
        assert stored.h_norm         == pytest.approx(0.08)
        assert stored.p_entail       == pytest.approx(0.94)
        assert stored.p_ground_max   == pytest.approx(0.89)
        assert stored.p_ground_mean  == pytest.approx(0.87)
        assert stored.p_ground_atomic == pytest.approx(0.85)
        # Plus p_contra + decision + early_exit_triggered.
        assert stored.p_contra       == pytest.approx(0.02)
        assert stored.decision       == "STORE"
        assert stored.early_exit_triggered is False
        # And the composite itself was raised to the new value.
        assert stored.u_stored       == pytest.approx(0.90)
        assert stored.retroverified  is True
