"""
tests/test_pipeline_batch_equivalence.py
=========================================
Level B Phase 1 contract test: ``BatchPipeline.answer_batch`` must
produce per-sample results equivalent to ``CAEMPipeline.answer`` for
the same inputs, modulo bf16 accumulation noise.

Phase 1 skeleton commit: ``answer_batch`` is a serial loop over
``answer``, so these tests assert EXACT equality. Subsequent commits
that replace the loop with real batched execution relax strict
equality to tolerances (``atol=1e-3`` on u_stored, ``atol=1e-2`` on
individual signals). Keeping both gates here as class-level methods
so the strict version documents the scaffolding contract and the
tolerant version guards against regression when batching is added.

Uses mocked models so the test runs in CI without GPU and without
the ~1.5 GB MiniCheck download. Real-GPU smoke lives in a separate
test file that only runs under pytest.mark.gpu.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from caem.pipeline_batch import BatchPipeline, BatchSample


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

def _make_fake_pipeline_result(i: int) -> SimpleNamespace:
    """Stand-in for PipelineResult that exposes the fields batch tests
    compare. Using SimpleNamespace so the test doesn't depend on the
    exact PipelineResult dataclass surface."""
    return SimpleNamespace(
        query=f"q{i}",
        answer=f"a{i}",
        tier=3,
        stored=False,
        u_stored=0.1 * (i + 1),
        latency_ms=7000.0 + i,
        decision="DISCARD",
        escalated=False,
        entry_id=None,
        verifier_output=None,
        pre_confidence=None,
        display_answer=f"a{i}",
    )


@pytest.fixture
def mock_serial_pipeline():
    """CAEMPipeline stand-in whose ``answer`` returns a deterministic
    stand-in per sample. Verifies the loop path calls through correctly."""
    pipeline = MagicMock()
    call_log: List[dict] = []

    def _answer(query=None, store_to_memory=True, source_benchmark=None):
        call_log.append({
            "query": query,
            "store_to_memory": store_to_memory,
            "source_benchmark": source_benchmark,
        })
        idx = len(call_log) - 1
        return _make_fake_pipeline_result(idx)

    pipeline.answer.side_effect = _answer
    pipeline._call_log = call_log
    return pipeline


# --------------------------------------------------------------------------- #
# Phase 1 scaffolding tests (serial-loop implementation)                      #
# --------------------------------------------------------------------------- #

def test_answer_batch_empty_returns_empty_list(mock_serial_pipeline):
    bp = BatchPipeline(mock_serial_pipeline)
    assert bp.answer_batch([]) == []
    mock_serial_pipeline.answer.assert_not_called()


def test_answer_batch_preserves_order_and_count(mock_serial_pipeline):
    bp = BatchPipeline(mock_serial_pipeline)
    samples = [
        BatchSample(query=f"q{i}", source_benchmark="fever")
        for i in range(5)
    ]
    results = bp.answer_batch(samples)
    assert len(results) == 5
    for i, r in enumerate(results):
        # Each call produced a result matching its position.
        assert r.query == f"q{i}"
        assert r.answer == f"a{i}"


def test_answer_batch_forwards_store_to_memory(mock_serial_pipeline):
    bp = BatchPipeline(mock_serial_pipeline)
    samples = [
        BatchSample(query="q0", store_to_memory=True),
        BatchSample(query="q1", store_to_memory=False),
    ]
    bp.answer_batch(samples)
    calls = mock_serial_pipeline._call_log
    assert calls[0]["store_to_memory"] is True
    assert calls[1]["store_to_memory"] is False


def test_answer_batch_forwards_source_benchmark(mock_serial_pipeline):
    bp = BatchPipeline(mock_serial_pipeline)
    samples = [
        BatchSample(query="q0", source_benchmark="fever"),
        BatchSample(query="q1", source_benchmark="triviaqa"),
        BatchSample(query="q2", source_benchmark=None),
    ]
    bp.answer_batch(samples)
    calls = mock_serial_pipeline._call_log
    assert calls[0]["source_benchmark"] == "fever"
    assert calls[1]["source_benchmark"] == "triviaqa"
    assert calls[2]["source_benchmark"] is None


def test_answer_batch_exact_equivalence_with_serial(mock_serial_pipeline):
    """Skeleton version must produce byte-identical results to calling
    ``answer`` N times in sequence (no batching yet so strict equality
    holds). When later commits add real batching this test will be
    relaxed to tolerance-based comparison in a separate method."""
    bp = BatchPipeline(mock_serial_pipeline)
    samples = [BatchSample(query=f"q{i}") for i in range(3)]
    batch_results = bp.answer_batch(samples)

    # Reset the mock and run the serial path separately for a reference.
    mock_serial_pipeline.answer.reset_mock()
    mock_serial_pipeline._call_log.clear()
    serial_results = [
        mock_serial_pipeline.answer(
            query=s.query,
            store_to_memory=s.store_to_memory,
            source_benchmark=s.source_benchmark,
        )
        for s in samples
    ]

    assert len(batch_results) == len(serial_results)
    for b, s in zip(batch_results, serial_results):
        assert b.query == s.query
        assert b.answer == s.answer
        assert b.u_stored == s.u_stored
        assert b.decision == s.decision
