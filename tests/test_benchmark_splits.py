"""
tests/test_benchmark_splits.py
==============================
Unit tests for caem.benchmark_splits (the Branch C 2026-04-22 evening pool
builder). Validates:

  - Content-hash ID is deterministic on question text and tolerant of
    missing fields.
  - _dedupe_by_content_id removes within-split duplicates (the bug that
    caused FEVER 9902 / TriviaQA 61861 cross-pool leaks until we added it).
  - BenchmarkPools.cycle_chunk(n) indexes 1-based and returns empty tuples
    for transfer-only benchmarks.
  - assert_no_leakage raises PoolLeakageError on manual duplicate injection.
  - assert_cross_benchmark_disjoint raises when two benchmarks share a
    content_id.
  - build_benchmark_pools with mock loader returns the 6-way split and
    respects the sizing arguments.

No HF dataset downloads — all tests use small mock samples or a
monkeypatched load_benchmark.
"""

from __future__ import annotations

import pytest

from caem.benchmark_splits import (
    BenchmarkPools,
    DEFAULT_N_CYCLES,
    DEFAULT_SEED_SIZE,
    PoolLeakageError,
    TRAINING_BENCHMARKS,
    TRANSFER_BENCHMARKS,
    ALL_BENCHMARKS,
    _dedupe_by_content_id,
    assert_cross_benchmark_disjoint,
    assert_no_leakage,
    content_id,
)


# =============================================================================
# content_id
# =============================================================================

class TestContentId:
    def test_deterministic_on_same_question(self):
        s1 = {"question": "What is the capital of France?"}
        s2 = {"question": "What is the capital of France?"}
        assert content_id(s1) == content_id(s2)

    def test_distinct_on_different_questions(self):
        s1 = {"question": "What is the capital of France?"}
        s2 = {"question": "What is the capital of Germany?"}
        assert content_id(s1) != content_id(s2)

    def test_tolerant_of_missing_question(self):
        s = {"id": "abc"}  # no question field
        # Should not raise; returns hash of empty string
        cid = content_id(s)
        assert isinstance(cid, str)
        assert len(cid) == 16

    def test_deterministic_on_whitespace(self):
        s1 = {"question": "  hello world  "}
        s2 = {"question": "hello world"}
        assert content_id(s1) == content_id(s2)  # stripped


# =============================================================================
# _dedupe_by_content_id
# =============================================================================

class TestDedupe:
    def test_removes_within_split_duplicates(self):
        samples = [
            {"question": "Q1", "id": "a"},
            {"question": "Q1", "id": "b"},  # same content, different native id
            {"question": "Q2", "id": "c"},
            {"question": "Q2", "id": "d"},
            {"question": "Q3", "id": "e"},
        ]
        unique = _dedupe_by_content_id(samples)
        assert len(unique) == 3  # Q1, Q2, Q3
        assert unique[0]["id"] == "a"  # first occurrence retained
        assert unique[1]["id"] == "c"
        assert unique[2]["id"] == "e"

    def test_no_duplicates_passes_through(self):
        samples = [
            {"question": "Q1"}, {"question": "Q2"}, {"question": "Q3"},
        ]
        assert _dedupe_by_content_id(samples) == samples

    def test_empty_list(self):
        assert _dedupe_by_content_id([]) == []


# =============================================================================
# BenchmarkPools.cycle_chunk
# =============================================================================

class TestCycleChunk:
    def test_1_indexed(self):
        chunks = (
            ({"id": 1, "question": "a"},),
            ({"id": 2, "question": "b"},),
            ({"id": 3, "question": "c"},),
        )
        pools = BenchmarkPools(
            benchmark="fever", is_training=True, sil_train_chunks=chunks,
        )
        assert pools.cycle_chunk(1) == chunks[0]
        assert pools.cycle_chunk(2) == chunks[1]
        assert pools.cycle_chunk(3) == chunks[2]

    def test_out_of_range_raises(self):
        pools = BenchmarkPools(
            benchmark="fever", is_training=True,
            sil_train_chunks=({"id": 1},),
        )
        with pytest.raises(IndexError):
            pools.cycle_chunk(0)
        with pytest.raises(IndexError):
            pools.cycle_chunk(2)  # only 1 chunk

    def test_transfer_only_empty(self):
        pools = BenchmarkPools(benchmark="asqa", is_training=False)
        assert pools.cycle_chunk(1) == ()
        assert pools.cycle_chunk(5) == ()  # no IndexError for transfer-only


# =============================================================================
# assert_no_leakage
# =============================================================================

class TestAssertNoLeakage:
    def test_clean_pools_pass(self):
        pools = BenchmarkPools(
            benchmark="fever", is_training=True,
            seed=({"question": "Q1"},),
            purity=({"question": "Q2"},),
            calibration=({"question": "Q3"},),
            sil_train_chunks=(
                ({"question": "Q4"}, {"question": "Q5"}),
                ({"question": "Q6"},),
            ),
            eval=({"question": "Q7"},),
            test=({"question": "Q8"},),
        )
        # Should not raise
        assert_no_leakage(pools)

    def test_raises_on_cross_pool_dup(self):
        """Same question appears in seed AND eval — leakage."""
        pools = BenchmarkPools(
            benchmark="fever", is_training=True,
            seed=({"question": "leaked"},),
            eval=({"question": "leaked"},),  # same content_id
        )
        with pytest.raises(PoolLeakageError) as exc_info:
            assert_no_leakage(pools)
        assert "fever" in str(exc_info.value)
        assert "duplicate" in str(exc_info.value).lower()

    def test_raises_on_cycle_chunk_dup(self):
        pools = BenchmarkPools(
            benchmark="fever", is_training=True,
            sil_train_chunks=(
                ({"question": "Q1"},),
                ({"question": "Q1"},),  # same content in cycle 2 as cycle 1
            ),
        )
        with pytest.raises(PoolLeakageError):
            assert_no_leakage(pools)


# =============================================================================
# assert_cross_benchmark_disjoint
# =============================================================================

class TestAssertCrossBenchmarkDisjoint:
    def test_disjoint_benchmarks_pass(self):
        pools = {
            "fever": BenchmarkPools(
                benchmark="fever", is_training=True,
                seed=({"question": "fever Q1"},),
            ),
            "triviaqa": BenchmarkPools(
                benchmark="triviaqa", is_training=True,
                seed=({"question": "trivia Q1"},),
            ),
        }
        # Should not raise
        assert_cross_benchmark_disjoint(pools)

    def test_raises_on_cross_benchmark_dup(self):
        shared_question = "Who directed Inception?"
        pools = {
            "fever": BenchmarkPools(
                benchmark="fever", is_training=True,
                seed=({"question": shared_question},),
            ),
            "triviaqa": BenchmarkPools(
                benchmark="triviaqa", is_training=True,
                seed=({"question": shared_question},),
            ),
        }
        with pytest.raises(PoolLeakageError) as exc_info:
            assert_cross_benchmark_disjoint(pools)
        msg = str(exc_info.value)
        assert "fever" in msg
        assert "triviaqa" in msg


# =============================================================================
# Panel definition
# =============================================================================

class TestPanelDefinition:
    def test_training_and_transfer_are_disjoint(self):
        train_set = set(TRAINING_BENCHMARKS)
        transfer_set = set(TRANSFER_BENCHMARKS)
        assert train_set.isdisjoint(transfer_set)

    def test_all_benchmarks_union(self):
        assert set(ALL_BENCHMARKS) == set(TRAINING_BENCHMARKS) | set(TRANSFER_BENCHMARKS)

    def test_natural_questions_is_transfer_only(self):
        # v2 (2026-05-06): NQ moved from training to transfer panel
        # alongside the FEVER-monoculture failure-mode response. NQ is
        # now structural-failure-only (used to measure the v1 cycle-2
        # NQ collapse mode). See branch_C_log 2026-05-06.
        assert "natural_questions" in TRANSFER_BENCHMARKS
        assert "natural_questions" not in TRAINING_BENCHMARKS

    def test_hotpotqa_and_csqa_are_training(self):
        # v2 added HotpotQA + CommonsenseQA to the training panel to
        # exercise multi-hop composition + 5-choice MCQ regimes that
        # v1's 3-bench panel didn't cover.
        assert "hotpotqa" in TRAINING_BENCHMARKS
        assert "commonsense_qa" in TRAINING_BENCHMARKS

    def test_training_size(self):
        # v2: 4 training benchmarks {fever, triviaqa, hotpotqa, commonsense_qa}
        # was 3 in v1.
        assert len(TRAINING_BENCHMARKS) == 4

    def test_transfer_size(self):
        # v2: 3 transfer benchmarks {truthfulqa, strategyqa, natural_questions}
        # was 4 in v1 (which included arc_challenge and asqa).
        assert len(TRANSFER_BENCHMARKS) == 3


# =============================================================================
# Defaults sanity
# =============================================================================

class TestDefaults:
    def test_seed_size_is_candidate_pool(self):
        # DEFAULT_SEED_SIZE was bumped 200 → 1000 to provide store-rate headroom.
        # At ~60% store rate, 1000 candidates yields ~600 stored episodes,
        # comfortably above the 200 target_episodes threshold.
        assert DEFAULT_SEED_SIZE >= 200

    def test_n_cycles_matches_plan(self):
        assert DEFAULT_N_CYCLES == 10


# =============================================================================
# Integration-ish (uses real build_benchmark_pools via monkeypatch)
# =============================================================================

class TestBuildBenchmarkPoolsWithMockLoader:
    def test_training_benchmark_pool_structure(self, monkeypatch):
        """Use a monkeypatched load_benchmark that returns 6000 unique samples
        so the 6-way split (1000 + 500 + 500 + 10×5000 = 16000) FAILS as
        expected — capacity check works."""
        from caem.benchmark_splits import (
            build_benchmark_pools, InsufficientBenchmarkDataError,
        )

        def mock_load_fever(benchmark, split="train", n=None, **kwargs):
            return [{"question": f"Q{i}", "id": str(i)} for i in range(6000)]

        import eval.benchmarks as ebm
        monkeypatch.setattr(ebm, "load_benchmark", mock_load_fever)

        with pytest.raises(InsufficientBenchmarkDataError) as exc_info:
            build_benchmark_pools(
                "fever", n_cycles=10, train_chunk_size=5000,
                seed_size=1000, purity_size=500, calibration_size=500,
                eval_size=500, test_size=500,
            )
        assert "fever" in str(exc_info.value)
        assert "train split has 6000" in str(exc_info.value) or "6000" in str(exc_info.value)
