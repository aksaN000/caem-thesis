"""
tests/test_pipeline_batch_prefetch.py
======================================
Phase 2 (async overlap) tests for PrefetchingBatchPipeline. These run
without GPU via mocked BatchPipeline stubs, mirroring the Phase 1
test strategy. Real-GPU validation lives in scripts/level_b_smoke.py.

Scaffold tests (this commit): cover the chunking math and contract
that PrefetchingBatchPipeline delegates to BatchPipeline in submitted
order. Overlap-specific tests (thread-pool dispatch, worker-thread
prep, failure recovery) are added in Phase-2-step-2 when the real
prefetch logic lands.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import pytest

from caem.pipeline_batch import BatchSample
from caem.pipeline_batch_prefetch import (
    PrefetchingBatchPipeline,
    _PreparedChunk,
)


# --------------------------------------------------------------------------- #
# Helper fixtures                                                             #
# --------------------------------------------------------------------------- #

def _fake_result(i: int) -> SimpleNamespace:
    """Deterministic PipelineResult stand-in for per-index verification."""
    return SimpleNamespace(
        query=f"q{i}", answer=f"a{i}", tier=3,
        u_stored=0.5, decision="DISCARD", stored=False,
    )


def _make_batch_pipeline_stub(max_chunk_seen: List[int]):
    """BatchPipeline stand-in whose ``answer_batch`` returns one fake
    result per sample and records the size of each chunk it was asked
    to process."""
    bp = MagicMock()
    result_counter = {"i": 0}
    chunks_received: List[List[BatchSample]] = []

    def _answer_batch(samples):
        chunks_received.append(list(samples))
        max_chunk_seen.append(len(samples))
        out = []
        for _ in samples:
            out.append(_fake_result(result_counter["i"]))
            result_counter["i"] += 1
        return out

    bp.answer_batch.side_effect = _answer_batch
    bp._chunks_received = chunks_received
    return bp


# --------------------------------------------------------------------------- #
# Scaffold tests                                                              #
# --------------------------------------------------------------------------- #

def test_empty_returns_empty():
    bp = MagicMock()
    pp = PrefetchingBatchPipeline(bp, chunk_size=4)
    try:
        assert pp.answer_batch([]) == []
        bp.answer_batch.assert_not_called()
    finally:
        pp.shutdown()


def test_single_chunk_delegates_directly_to_batch_pipeline():
    """When chunk_size >= N, the wrapper passes the full list through
    to BatchPipeline in one call."""
    max_seen: List[int] = []
    bp = _make_batch_pipeline_stub(max_seen)
    pp = PrefetchingBatchPipeline(bp, chunk_size=16)
    try:
        samples = [BatchSample(query=f"q{i}") for i in range(5)]
        results = pp.answer_batch(samples)
        assert len(results) == 5
        # Exactly one answer_batch call with all 5 samples.
        assert bp.answer_batch.call_count == 1
        assert max_seen == [5]
    finally:
        pp.shutdown()


def test_multi_chunk_splits_by_chunk_size_and_preserves_order():
    """With chunk_size=4 and 10 samples, the wrapper should issue
    three calls of sizes 4, 4, 2 in that order."""
    max_seen: List[int] = []
    bp = _make_batch_pipeline_stub(max_seen)
    pp = PrefetchingBatchPipeline(bp, chunk_size=4)
    try:
        samples = [BatchSample(query=f"q{i}") for i in range(10)]
        results = pp.answer_batch(samples)
        assert len(results) == 10
        assert max_seen == [4, 4, 2]
        # Order preservation: every chunk's queries match the slice.
        chunks = bp._chunks_received
        assert [s.query for s in chunks[0]] == [f"q{i}" for i in range(0, 4)]
        assert [s.query for s in chunks[1]] == [f"q{i}" for i in range(4, 8)]
        assert [s.query for s in chunks[2]] == [f"q{i}" for i in range(8, 10)]
    finally:
        pp.shutdown()


def test_exact_chunk_boundary_no_short_trailing_chunk():
    """When N is an exact multiple of chunk_size, no 0-length tail."""
    max_seen: List[int] = []
    bp = _make_batch_pipeline_stub(max_seen)
    pp = PrefetchingBatchPipeline(bp, chunk_size=4)
    try:
        samples = [BatchSample(query=f"q{i}") for i in range(8)]
        pp.answer_batch(samples)
        assert max_seen == [4, 4]
    finally:
        pp.shutdown()


def test_chunk_size_one_degenerate_but_correct():
    """chunk_size=1 is the pathological worst case for prefetch but
    must still produce correct results (each sample a single 1-element
    call into BatchPipeline)."""
    max_seen: List[int] = []
    bp = _make_batch_pipeline_stub(max_seen)
    pp = PrefetchingBatchPipeline(bp, chunk_size=1)
    try:
        samples = [BatchSample(query=f"q{i}") for i in range(3)]
        results = pp.answer_batch(samples)
        assert len(results) == 3
        assert max_seen == [1, 1, 1]
    finally:
        pp.shutdown()


def test_chunk_size_zero_clamped_to_one():
    """chunk_size=0 is invalid; constructor should clamp to 1."""
    max_seen: List[int] = []
    bp = _make_batch_pipeline_stub(max_seen)
    pp = PrefetchingBatchPipeline(bp, chunk_size=0)
    try:
        assert pp.chunk_size == 1
    finally:
        pp.shutdown()


def test_context_manager_shuts_down_worker():
    """Using PrefetchingBatchPipeline as a context manager must
    cleanly shut down its worker thread on exit."""
    max_seen: List[int] = []
    bp = _make_batch_pipeline_stub(max_seen)
    with PrefetchingBatchPipeline(bp, chunk_size=4) as pp:
        samples = [BatchSample(query="q0")]
        pp.answer_batch(samples)
    # After __exit__ the worker pool should be shut down; submitting
    # a new task must raise.
    with pytest.raises(RuntimeError):
        pp._executor.submit(lambda: None)


def test_prepared_chunk_dataclass_shape():
    """Sanity check on the _PreparedChunk dataclass that the worker
    thread produces."""
    pc = _PreparedChunk(samples=[BatchSample(query="q")])
    assert pc.samples[0].query == "q"
    assert pc.tiers_peek is None
    assert pc.prepared_extras is None


# --------------------------------------------------------------------------- #
# Prefetch dispatch tests (real worker-thread overlap)                        #
# --------------------------------------------------------------------------- #

def test_prefetch_prep_called_per_chunk_in_order():
    """Each chunk must receive exactly one _prep_chunk invocation,
    and the prepped samples must match the original slice."""
    max_seen: List[int] = []
    bp = _make_batch_pipeline_stub(max_seen)
    pp = PrefetchingBatchPipeline(bp, chunk_size=3)
    prep_calls: List[List[str]] = []

    original_prep = pp._prep_chunk

    def _tracking_prep(chunk):
        prep_calls.append([s.query for s in chunk])
        return original_prep(chunk)

    pp._prep_chunk = _tracking_prep  # type: ignore[method-assign]

    try:
        samples = [BatchSample(query=f"q{i}") for i in range(7)]
        pp.answer_batch(samples)
        # 3 chunks: [q0..q2], [q3..q5], [q6]
        assert prep_calls == [
            ["q0", "q1", "q2"],
            ["q3", "q4", "q5"],
            ["q6"],
        ]
    finally:
        pp.shutdown()


def test_prefetch_recovers_from_worker_exception():
    """If the worker raises during a chunk prep, the consumer must
    fall back to synchronous prep for that chunk and continue on."""
    max_seen: List[int] = []
    bp = _make_batch_pipeline_stub(max_seen)
    pp = PrefetchingBatchPipeline(bp, chunk_size=4)
    fail_on_first = {"fired": False}

    original_prep = pp._prep_chunk

    def _flaky_prep(chunk):
        # Fail the first time the worker is invoked; succeed thereafter.
        if not fail_on_first["fired"]:
            fail_on_first["fired"] = True
            raise RuntimeError("synthetic worker failure for test")
        return original_prep(chunk)

    pp._prep_chunk = _flaky_prep  # type: ignore[method-assign]

    try:
        samples = [BatchSample(query=f"q{i}") for i in range(8)]
        results = pp.answer_batch(samples)
        assert len(results) == 8
        # Both chunks still ran through bp.answer_batch despite the
        # worker exception on chunk 0's prep.
        assert max_seen == [4, 4]
    finally:
        pp.shutdown()


def test_prefetch_worker_overlap_observable():
    """If we inject a deliberate sleep into the worker prep AND a
    separate sleep into the GPU work, the total wall-clock of the
    pipelined path should be strictly less than the sum (no overlap
    would be >= max of the two)."""
    import time

    prep_wait = 0.05
    gpu_wait = 0.05

    bp = MagicMock()

    def _slow_answer(samples):
        time.sleep(gpu_wait)
        return [_fake_result(i) for i in range(len(samples))]

    bp.answer_batch.side_effect = _slow_answer
    pp = PrefetchingBatchPipeline(bp, chunk_size=2)

    original_prep = pp._prep_chunk

    def _slow_prep(chunk):
        time.sleep(prep_wait)
        return original_prep(chunk)

    pp._prep_chunk = _slow_prep  # type: ignore[method-assign]

    try:
        samples = [BatchSample(query=f"q{i}") for i in range(8)]
        t0 = time.perf_counter()
        pp.answer_batch(samples)  # 4 chunks of 2
        elapsed = time.perf_counter() - t0
        # 4 chunks; naive serial = 4 * (prep + gpu) = 0.4s.
        # With overlap: prep of chunk N+1 hides behind GPU of chunk N,
        # so total should be close to 4 * gpu + 1 * prep = 0.25s on
        # the worst case. Assert strictly less than serial-no-overlap.
        serial_no_overlap = 4 * (prep_wait + gpu_wait)
        assert elapsed < serial_no_overlap * 0.95, (
            f"Expected overlap speedup, got {elapsed:.3f}s vs "
            f"serial-no-overlap {serial_no_overlap:.3f}s"
        )
    finally:
        pp.shutdown()
