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

import json
import math
import pickle
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from caem.config import CAEMConfig
from caem.memory.entry import (
    EpisodicEntry,
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

    # The PostGenerationConfidence dataclass was removed as dead code on
    # 2026-04-22 (retired in Session 42, removed in the veto/dead-code
    # cleanup). Its test_post_generation_dataclass_constructs sibling
    # is gone with it.

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

    def test_retroverify_downgrades_by_default(self):
        """Branch C Goal 4 item 4: new score below old but above threshold
        now downgrades stored u_stored + overwrites the nine-signal block
        (pre-Goal-4 behaviour was raise-only; stale confident entries
        silently survived).
        """
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.85))

        lower = _make_verifier_output(u_stored=0.73, p_entail=0.70, s_avg=0.75)
        n_updated, n_removed = store.retroverify(lambda _: lower, threshold=0.50)

        assert n_updated == 1
        assert n_removed == 0
        assert math.isclose(store.get(eid).u_stored, 0.73, abs_tol=1e-4)
        assert store.get(eid).retroverified is True

    def test_retroverify_respects_allow_downgrade_false(self):
        """When cfg.retroverify_allow_downgrade is False, fall back to the
        pre-Goal-4 raise-only behaviour so the legacy contract is recoverable
        for smoke tests or reproductions of pre-2026-04-22 results.
        """
        store = make_store()
        store.config.retroverify_allow_downgrade = False
        eid = store.add(make_entry(seed=0, u_stored=0.85))

        lower = _make_verifier_output(u_stored=0.73, p_entail=0.70, s_avg=0.75)
        n_updated, n_removed = store.retroverify(lambda _: lower, threshold=0.50)

        assert n_updated == 0
        assert n_removed == 0
        assert math.isclose(store.get(eid).u_stored, 0.85, abs_tol=1e-4)
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

    def test_retroverify_loop_prunes_by_default(self):
        """Branch C Goal 4 item 2: loop-contaminated entries are removed at
        retroverify time before any verifier forward pass. The verify_fn
        must NOT be called on the loop-pruned entry.
        """
        store = make_store()
        # Force the entry's chain to trigger the loop filter: repeat "yes"
        # ~30 times so distinct-4 collapses. u_stored stays high so only
        # the loop filter can remove it (threshold-prune would not fire).
        loop_entry = make_entry(seed=0, u_stored=0.90)
        loop_entry.reasoning_chain = " ".join(["yes"] * 30)
        store.add(loop_entry)

        # Healthy entry with a clean chain -- should survive.
        clean_entry = make_entry(
            question="clean entry", seed=1, u_stored=0.80,
        )
        clean_entry.reasoning_chain = (
            "Reasoning: The capital of France is Paris, a fact established "
            "as the seat of French government for centuries. Answer: Paris"
        )
        clean_eid = store.add(clean_entry)

        calls = {"n": 0}

        def verify_fn(entry):
            calls["n"] += 1
            return _make_verifier_output(u_stored=0.90)

        n_updated, n_removed = store.retroverify(verify_fn, threshold=0.50)

        # One loop pruned without a verifier call, one clean entry re-scored.
        assert n_removed == 1
        assert calls["n"] == 1
        # Breakdown diagnostic attribute.
        assert store._last_retroverify_breakdown == {
            "loop_pruned": 1,
            "threshold_pruned": 0,
        }
        # Clean entry still in memory.
        assert store.get(clean_eid) is not None

    def test_retroverify_respects_prune_loops_false(self):
        """When cfg.retroverify_prune_loops is False, loop entries survive
        retroverify (they still get re-scored like any other entry). Lets
        the pre-Goal-4 behaviour be recovered for ablations.
        """
        store = make_store()
        store.config.retroverify_prune_loops = False
        loop_entry = make_entry(seed=0, u_stored=0.90)
        loop_entry.reasoning_chain = " ".join(["yes"] * 30)
        eid = store.add(loop_entry)

        n_updated, n_removed = store.retroverify(
            lambda _: _make_verifier_output(u_stored=0.85), threshold=0.50,
        )

        # Not loop-pruned, re-scored normally.
        assert n_removed == 0
        assert store.get(eid) is not None
        assert store._last_retroverify_breakdown == {
            "loop_pruned": 0,
            "threshold_pruned": 0,
        }

    def test_retroverify_loop_prune_counts_separately_from_threshold(self):
        """Combined loop + threshold prunes: the 2-tuple return sums both,
        but the breakdown attribute distinguishes them for downstream logs.
        """
        store = make_store()

        looped = make_entry(question="loop", seed=0, u_stored=0.90)
        looped.reasoning_chain = " ".join(["yes"] * 30)
        store.add(looped)

        clean_good = make_entry(question="clean-good", seed=1, u_stored=0.70)
        clean_good.reasoning_chain = (
            "Reasoning: Water boils at 100 C at sea level. Answer: 100."
        )
        store.add(clean_good)

        clean_bad = make_entry(question="clean-bad", seed=2, u_stored=0.70)
        clean_bad.reasoning_chain = (
            "Reasoning: Paris is the capital of France historically. Answer: Paris."
        )
        store.add(clean_bad)

        def verify_fn(entry):
            if entry.question == "clean-good":
                return _make_verifier_output(u_stored=0.85)
            return _make_verifier_output(u_stored=0.30, decision="DISCARD")

        n_updated, n_removed = store.retroverify(verify_fn, threshold=0.50)

        assert n_updated == 1              # clean-good raised
        assert n_removed == 2              # 1 loop + 1 threshold
        assert store._last_retroverify_breakdown == {
            "loop_pruned": 1,
            "threshold_pruned": 1,
        }


# -----------------------------------------------------------------------------
# Consolidation (Branch C Goal 4 item 3)
# -----------------------------------------------------------------------------

def _near_dup_pair(target_sim: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return a pair of unit vectors with exact pairwise cosine similarity
    ``target_sim`` (modulo fp32 rounding). Used to construct paraphrase
    clusters in consolidation tests without depending on a real SBERT
    encoder.

    Construction: pick a reference ``ref`` and a unit vector ``p``
    orthogonal to ``ref``. Define
        a = ref
        b = target_sim * ref + sqrt(1 - target_sim^2) * p
    Then ``<a, b> = target_sim`` by construction.
    """
    ref = random_unit_vec(0)
    p = random_unit_vec(1)
    p = p - float(np.dot(p, ref)) * ref
    norm = float(np.linalg.norm(p))
    if norm > 0:
        p = p / norm
    a = ref.astype(np.float32)
    b = (target_sim * ref + (1.0 - target_sim ** 2) ** 0.5 * p).astype(np.float32)
    b = b / float(np.linalg.norm(b))
    return a, b


def _dup_entry(
    question: str, u_stored: float, emb: np.ndarray, *,
    answer: str = "4",
    retrieval_count: int = 0,
    success_rate: float = 0.0,
    source_benchmark: Optional[str] = None,
) -> EpisodicEntry:
    """Build an EpisodicEntry with an injected embedding + customisable
    answer/benchmark fields for the consolidation safety-guard tests.
    """
    e = make_entry(
        question=question, answer=answer, u_stored=u_stored, embedding=emb,
    )
    e.retrieval_count = retrieval_count
    e.success_rate = success_rate
    if source_benchmark is not None:
        e.source_benchmark = source_benchmark
    return e


class TestConsolidate:
    def test_empty_store_noop(self):
        store = make_store()
        n_clusters, n_removed = store.consolidate()
        assert n_clusters == 0
        assert n_removed == 0

    def test_singletons_untouched(self):
        store = make_store()
        store.add(make_entry(question="Q1", seed=101, u_stored=0.80))
        store.add(make_entry(question="Q2", seed=202, u_stored=0.70))
        n_clusters, n_removed = store.consolidate()
        assert n_clusters == 0
        assert n_removed == 0
        assert store.size == 2

    def test_near_duplicate_pair_collapses_to_max_u(self):
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        # Stay within cfg.consolidation_max_u_spread (0.15) so the safety
        # guard does not fire; it is tested separately in
        # TestConsolidateSafetyGuards::test_u_spread_guardrail_skips_merge.
        id_low = store.add(_dup_entry("paraphrase-low", 0.80, a, answer="Paris"))
        id_high = store.add(_dup_entry("paraphrase-high", 0.90, b, answer="Paris"))

        n_clusters, n_removed = store.consolidate()
        assert n_clusters == 1
        assert n_removed == 1
        assert store.get(id_high) is not None
        assert store.get(id_low) is None

    def test_retrieval_metadata_merged(self):
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        # u_stored spread (0.90 - 0.82 = 0.08) stays under the 0.15 guard.
        e_rep = _dup_entry(
            "rep", 0.90, a, answer="Paris",
            retrieval_count=4, success_rate=0.75,   # 3/4
        )
        e_mem = _dup_entry(
            "member", 0.82, b, answer="Paris",
            retrieval_count=6, success_rate=0.50,   # 3/6
        )
        id_rep = store.add(e_rep)
        store.add(e_mem)

        store.consolidate()
        rep = store.get(id_rep)
        assert rep is not None
        assert rep.retrieval_count == 10
        # (3 + 3) / (4 + 6) = 0.60
        assert math.isclose(rep.success_rate, 0.60, abs_tol=1e-9)

    def test_config_gate_disables_pass(self):
        store = make_store()
        store.config.enable_consolidation = False
        a, b = _near_dup_pair(target_sim=0.95)
        store.add(_dup_entry("Q1", 0.90, a, answer="Paris"))
        store.add(_dup_entry("Q2", 0.85, b, answer="Paris"))
        n_clusters, n_removed = store.consolidate()
        assert n_clusters == 0
        assert n_removed == 0
        assert store.size == 2

    def test_breakdown_attribute_populated(self):
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        store.add(_dup_entry("Q1", 0.90, a, answer="Paris"))
        store.add(_dup_entry("Q2", 0.85, b, answer="Paris"))
        store.consolidate()
        bd = store._last_consolidation_breakdown
        assert bd["clusters_merged"] == 1
        assert bd["clusters_skipped_answer_mismatch"] == 0
        assert bd["clusters_skipped_u_spread"] == 0
        assert bd["removed"] == 1
        assert bd["final_size"] == 1

    def test_threshold_override_strict(self):
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.93)
        store.add(_dup_entry("Q1", 0.90, a, answer="Paris"))
        store.add(_dup_entry("Q2", 0.85, b, answer="Paris"))
        # Default 0.92 would cluster; tighten to 0.99.
        n_clusters, n_removed = store.consolidate(similarity_threshold=0.99)
        assert n_clusters == 0
        assert n_removed == 0
        assert store.size == 2

    def test_tiebreak_deterministic_across_runs(self):
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        id_a = store.add(_dup_entry("A", 0.75, a, answer="Paris"))
        id_b = store.add(_dup_entry("B", 0.75, b, answer="Paris"))
        store.consolidate()
        surviving = id_a if store.get(id_a) is not None else id_b
        assert surviving == min(id_a, id_b)


class TestConsolidateSafetyGuards:
    """Branch C Goal 4 item 3 design-review safety fixes (2026-04-22)."""

    def test_answer_mismatch_skips_merge(self):
        """Clusters where the normalised answer field differs across members
        are NOT merged -- this is the sample-mode fix for "Who directed X"
        vs "Who starred in X" which score high on SBERT query similarity
        but have different correct answers.
        """
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        id_a = store.add(_dup_entry("directed?", 0.90, a, answer="Coppola"))
        id_b = store.add(_dup_entry("starred?", 0.80, b, answer="Brando"))

        n_clusters, n_removed = store.consolidate()
        assert n_clusters == 0
        assert n_removed == 0
        # Both entries still present.
        assert store.get(id_a) is not None
        assert store.get(id_b) is not None
        # Breakdown flags the skip reason.
        bd = store._last_consolidation_breakdown
        assert bd["clusters_skipped_answer_mismatch"] == 1
        assert bd["clusters_merged"] == 0

    def test_answer_equality_is_normalised(self):
        """Surface-form differences (punctuation, case) do not block the
        merge -- only semantic disagreement does.
        """
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        id_a = store.add(_dup_entry("q1", 0.90, a, answer="Paris."))
        store.add(_dup_entry("q2", 0.85, b, answer="paris "))
        n_clusters, _ = store.consolidate()
        assert n_clusters == 1
        assert store.get(id_a) is not None

    def test_u_spread_guardrail_skips_merge(self):
        """Clusters with internal u_stored spread > cfg.consolidation_max_u_spread
        are flagged for audit and NOT merged. Guards against "cluster looks
        semantically tight but one member is suspiciously unconfident".
        """
        store = make_store()
        # Default spread ceiling is 0.15; 0.92 - 0.70 = 0.22 > ceiling.
        a, b = _near_dup_pair(target_sim=0.95)
        id_a = store.add(_dup_entry("q1", 0.92, a, answer="Paris"))
        id_b = store.add(_dup_entry("q2", 0.70, b, answer="Paris"))
        n_clusters, n_removed = store.consolidate()
        assert n_clusters == 0
        assert n_removed == 0
        assert store.get(id_a) is not None
        assert store.get(id_b) is not None
        bd = store._last_consolidation_breakdown
        assert bd["clusters_skipped_u_spread"] == 1

    def test_u_spread_guardrail_configurable(self):
        store = make_store()
        store.config.consolidation_max_u_spread = 0.25  # allow wider spread
        a, b = _near_dup_pair(target_sim=0.95)
        store.add(_dup_entry("q1", 0.92, a, answer="Paris"))
        store.add(_dup_entry("q2", 0.70, b, answer="Paris"))
        n_clusters, _ = store.consolidate()
        assert n_clusters == 1

    def test_merged_source_benchmarks_populated(self):
        """Consolidating across benchmarks unions the tags onto the
        representative's ``merged_source_benchmarks``.
        """
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        id_rep = store.add(_dup_entry(
            "q1", 0.90, a, answer="Paris", source_benchmark="fever",
        ))
        store.add(_dup_entry(
            "q2", 0.85, b, answer="Paris", source_benchmark="triviaqa",
        ))
        store.consolidate()
        rep = store.get(id_rep)
        assert rep is not None
        assert rep.source_benchmark == "fever"
        assert rep.merged_source_benchmarks == ("triviaqa",)

    def test_sil_gate_defense_in_depth_on_manually_mixed_entry(self):
        """Defense-in-depth: if a mixed-pool entry somehow slips past the
        pre-split (e.g., future code path, storage bug), the SIL
        _collect_episodes gate still excludes it. Manually inject the
        cross-pool condition here since the pre-split means consolidation
        itself will never produce one in production.
        """
        from caem.training.self_improvement import SelfImprovementLoop
        from caem.config import CAEMConfig
        import tempfile

        store = make_store()
        # One FEVER entry, merge-tagged with truthfulqa directly on the
        # EpisodicEntry field (bypassing consolidation).
        entry = make_entry(
            question="q1", answer="Paris", u_stored=0.90, seed=0,
        )
        entry.source_benchmark = "fever"
        entry.merged_source_benchmarks = ("truthfulqa",)
        store.add(entry)

        cfg = CAEMConfig()
        cfg.min_u_stored_for_training = 0.70
        with tempfile.TemporaryDirectory() as tmpdir:
            from tests.test_self_improvement import (
                make_mock_model, make_mock_tokenizer,
            )
            loop = SelfImprovementLoop(
                make_mock_model(), make_mock_tokenizer(), cfg,
                device="cpu", output_dir=tmpdir,
            )
            pairs = loop._collect_episodes(store)
        # OOD-contaminated entry excluded from the SIL pool.
        assert pairs == []

    def test_audit_log_uses_enum_skip_reasons(self, tmp_path):
        """The audit log's ``outcome``/``reason`` fields come from the
        module-level SKIP_* / OUTCOME_* constants so Ch1 audits can grep
        for machine-readable values rather than regex over prose. Covers
        the merged outcome + at least one skip outcome in the same pass.
        """
        from caem.memory.store import (
            SKIP_ANSWER_MISMATCH,
            OUTCOME_MERGED,
        )
        store = make_store()

        # Merge case: clean cluster.
        a, b = _near_dup_pair(target_sim=0.95)
        store.add(_dup_entry("q-merge-1", 0.90, a, answer="Paris"))
        store.add(_dup_entry("q-merge-2", 0.85, b, answer="Paris"))

        log_path = tmp_path / "consolidation.jsonl"
        store.consolidate(audit_log_path=log_path)

        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        parsed = [json.loads(line) for line in lines]
        outcomes = {r["outcome"] for r in parsed}
        reasons = {r.get("reason") for r in parsed}
        assert OUTCOME_MERGED in outcomes
        # Exact-enum string (not a free-form synonym).
        assert OUTCOME_MERGED == "merged"
        assert SKIP_ANSWER_MISMATCH == "answer_mismatch"
        # Merged record carries reason == OUTCOME_MERGED too (enum-only).
        assert OUTCOME_MERGED in reasons


class TestConsolidatePreSplitAndCycleGuard:
    """Branch C Goal 4 item 3 design-review fixes (2026-04-22 round 2):
    pre-split by training / transfer pool + Cycle-0 protection."""

    def test_cycle_0_is_protected(self):
        """consolidate(cycle_num=0) is a no-op: cold-start memory stays
        intentionally diverse. First real pass runs at Cycle 1 -> Cycle 2.
        """
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        store.add(_dup_entry("q1", 0.90, a, answer="Paris"))
        store.add(_dup_entry("q2", 0.85, b, answer="Paris"))
        n_clusters, n_removed = store.consolidate(cycle_num=0)
        assert n_clusters == 0
        assert n_removed == 0
        assert store.size == 2

    def test_cycle_1_runs(self):
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        store.add(_dup_entry("q1", 0.90, a, answer="Paris"))
        store.add(_dup_entry("q2", 0.85, b, answer="Paris"))
        n_clusters, _ = store.consolidate(cycle_num=1)
        assert n_clusters == 1

    def test_cycle_num_none_runs(self):
        """When cycle_num is not supplied, pass runs unconditionally
        (preserves back-compat for existing tests and single-call scripts).
        """
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        store.add(_dup_entry("q1", 0.90, a, answer="Paris"))
        store.add(_dup_entry("q2", 0.85, b, answer="Paris"))
        n_clusters, _ = store.consolidate()
        assert n_clusters == 1

    def test_cross_pool_pair_never_merges(self):
        """A FEVER (training) entry + a TruthfulQA (transfer) entry at
        0.95 similarity are NOT merged, even though the query-side SBERT
        embedding would cluster them. This protects the SIL training pool
        from OOD leak by construction rather than by the downstream gate.
        """
        from caem.memory.store import SKIP_CROSS_POOL

        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        id_train = store.add(_dup_entry(
            "q1", 0.90, a, answer="Paris", source_benchmark="fever",
        ))
        id_transfer = store.add(_dup_entry(
            "q2", 0.85, b, answer="Paris", source_benchmark="truthfulqa",
        ))
        n_clusters, n_removed = store.consolidate()
        assert n_clusters == 0
        assert n_removed == 0
        # Both still present.
        assert store.get(id_train) is not None
        assert store.get(id_transfer) is not None
        # Cross-pool rejection logged in the breakdown.
        bd = store._last_consolidation_breakdown
        assert bd["clusters_skipped_cross_pool"] >= 1

    def test_within_pool_pair_merges(self):
        """Two FEVER (training) entries at 0.95 similarity merge normally.
        The representative's merged_source_benchmarks stays within the
        training pool (no OOD contamination).
        """
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        id_rep = store.add(_dup_entry(
            "q1", 0.90, a, answer="Paris", source_benchmark="fever",
        ))
        store.add(_dup_entry(
            "q2", 0.85, b, answer="Paris", source_benchmark="triviaqa",
        ))
        n_clusters, _ = store.consolidate()
        assert n_clusters == 1
        rep = store.get(id_rep)
        assert rep is not None
        # Representative carries the merged training-pool benchmark.
        assert rep.merged_source_benchmarks == ("triviaqa",)
        # Transfer benchmarks never appear on a within-training-pool rep.
        from caem.config import TRANSFER_BENCHMARKS
        for bm in rep.merged_source_benchmarks:
            assert bm not in TRANSFER_BENCHMARKS


class TestConsolidateTiebreaker:
    def test_tiebreaker_by_u_then_cycle_then_eid(self):
        """When u_stored ties, older storage_cycle wins; if those tie too,
        lower entry_id wins. Guards against FAISS-ordering-dependent
        non-determinism that would break checkpoint reproducibility.
        """
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)

        # Two entries with same u_stored, different storage_cycle.
        e_newer = _dup_entry("newer", 0.85, a, answer="Paris")
        e_newer.storage_cycle = 3
        e_older = _dup_entry("older", 0.85, b, answer="Paris")
        e_older.storage_cycle = 1

        id_newer = store.add(e_newer)
        id_older = store.add(e_older)
        store.consolidate()

        # Older cycle wins the tie (most-observed-across-cycles member).
        assert store.get(id_older) is not None
        assert store.get(id_newer) is None

    def test_tiebreaker_final_fallback_to_entry_id(self):
        """All three fields tied -> lowest entry_id wins (insertion order
        in these tests gives deterministic eids)."""
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        id_a = store.add(_dup_entry("A", 0.85, a, answer="Paris"))
        id_b = store.add(_dup_entry("B", 0.85, b, answer="Paris"))
        # Both at storage_cycle=0 by default.
        store.consolidate()
        assert store.get(min(id_a, id_b)) is not None
        assert store.get(max(id_a, id_b)) is None


# -----------------------------------------------------------------------------
# Hit-counter forced re-verification (Branch C Goal 4 item 5)
# -----------------------------------------------------------------------------

class TestHitCounter:
    def test_default_value_is_zero(self):
        e = make_entry()
        assert e.hit_counter == 0

    def test_increments_on_tier1_serve(self):
        """update_retrieval_stats is the Tier-1 serve entry point; each
        call bumps hit_counter by 1 regardless of accepted/overridden.
        """
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.85))
        store.update_retrieval_stats(eid, was_accepted=True)
        store.update_retrieval_stats(eid, was_accepted=False)
        store.update_retrieval_stats(eid, was_accepted=True)
        e = store.get(eid)
        assert e.hit_counter == 3
        # retrieval_count also bumps; the two counters move in lockstep
        # until retroverify resets the hit counter.
        assert e.retrieval_count == 3

    def test_retroverify_resets_counter_on_upgrade(self):
        """When retroverify re-scores an entry (either direction), the
        hit_counter returns to zero for the next cycle's pressure
        accounting. retrieval_count is cumulative and stays intact.
        """
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.70))
        for _ in range(5):
            store.update_retrieval_stats(eid, was_accepted=True)
        assert store.get(eid).hit_counter == 5

        improved = _make_verifier_output(u_stored=0.90)
        store.retroverify(lambda _: improved, threshold=0.50)

        e = store.get(eid)
        assert e.hit_counter == 0
        # Cumulative retrieval_count untouched by retroverify.
        assert e.retrieval_count == 5

    def test_retroverify_resets_counter_on_noop_tie(self):
        """Even when the new score equals the old (no u_stored update),
        the retroverified flag flips and hit_counter resets -- the entry
        WAS re-checked, so "serves since last check" should restart.
        """
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.80))
        for _ in range(4):
            store.update_retrieval_stats(eid, was_accepted=True)
        assert store.get(eid).hit_counter == 4

        tie = _make_verifier_output(u_stored=0.80)
        # allow_downgrade=False so the identical-u_stored branch falls to
        # the no-op tie path (still marks retroverified + resets counter).
        store.config.retroverify_allow_downgrade = False
        store.retroverify(lambda _: tie, threshold=0.50)

        assert store.get(eid).hit_counter == 0

    def test_consolidation_sums_hit_counter(self):
        """Closes the goal-4-item-5 TODO in consolidate(): the merged
        representative inherits all Tier-1 serve pressure from the cluster.
        """
        store = make_store()
        a, b = _near_dup_pair(target_sim=0.95)
        e_rep = _dup_entry("rep", 0.90, a, answer="Paris")
        e_rep.hit_counter = 7
        e_mem = _dup_entry("member", 0.85, b, answer="Paris")
        e_mem.hit_counter = 4
        id_rep = store.add(e_rep)
        store.add(e_mem)

        store.consolidate()
        rep = store.get(id_rep)
        assert rep is not None
        assert rep.hit_counter == 11

    def test_force_retroverify_queue_returns_crossed_entries(self):
        """force_retroverify_queue lists entries whose hit_counter >=
        threshold, sorted by descending pressure with entry_id tie-break.
        """
        store = make_store()
        id_low = store.add(make_entry(question="Q1", seed=0, u_stored=0.80))
        id_mid = store.add(make_entry(question="Q2", seed=1, u_stored=0.80))
        id_high = store.add(make_entry(question="Q3", seed=2, u_stored=0.80))

        for _ in range(3):  # below default threshold 10
            store.update_retrieval_stats(id_low, was_accepted=True)
        for _ in range(11):  # above threshold
            store.update_retrieval_stats(id_mid, was_accepted=True)
        for _ in range(15):  # also above -> highest pressure
            store.update_retrieval_stats(id_high, was_accepted=True)

        queue = store.force_retroverify_queue()
        assert queue == [id_high, id_mid]

    def test_force_retroverify_queue_threshold_override(self):
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.80))
        for _ in range(5):
            store.update_retrieval_stats(eid, was_accepted=True)

        # Default threshold = 10, entry has 5 hits -> not in queue.
        assert store.force_retroverify_queue() == []
        # Tighter threshold -> in queue.
        assert store.force_retroverify_queue(hit_threshold=3) == [eid]
        # Zero/negative disables.
        assert store.force_retroverify_queue(hit_threshold=0) == []

    def test_force_retroverify_queue_respects_config_value(self):
        store = make_store()
        store.config.hit_counter_force_retroverify = 4
        eid = store.add(make_entry(seed=0, u_stored=0.80))
        for _ in range(4):
            store.update_retrieval_stats(eid, was_accepted=True)
        assert store.force_retroverify_queue() == [eid]

    def test_pruning_threshold_path_does_not_keep_hit_counter(self):
        """Branch check: entries below the retroverify threshold get
        removed outright; their hit_counter disappears with them. No
        special semantics needed, but guard against a future "move the
        counter onto a sibling" regression.
        """
        store = make_store()
        eid = store.add(make_entry(seed=0, u_stored=0.80))
        for _ in range(6):
            store.update_retrieval_stats(eid, was_accepted=True)

        # Below-threshold new score -> remove.
        bad = _make_verifier_output(
            u_stored=0.30, decision="DISCARD", p_contra=0.05,
        )
        store.retroverify(lambda _: bad, threshold=0.50)
        assert store.get(eid) is None
