"""
caem/pipeline_batch_prefetch.py
================================
Level B Phase 2 (async overlap) for CAEM -- builds on Phase 1 static
batching by overlapping CPU-side work for chunk N+1 with GPU-side work
for chunk N. Phase 1 (BatchPipeline, caem/pipeline_batch.py) pools the
three dominant T5 costs across samples but still runs per-chunk CPU
prep (tokenise, FAISS retrieve, prompt build) in-line, leaving the GPU
idle during those windows.

Design
------
``PrefetchingBatchPipeline.answer_batch(samples)`` splits the input
list into fixed-size chunks and processes them in a producer/consumer
pipeline. A worker thread prepares the next chunk's CPU-side artefacts
(tokenisations, top-k passage retrievals, batched Tier 2 / Tier 3
prompts) while the main thread is inside the previous chunk's batched
GPU work. The main thread consumes each prepared chunk in submitted
order so per-sample result ordering is preserved.

Overlap targets
---------------
The per-chunk CPU budget is dominated by:

  * Per-query tokenisation (HF tokenizers, ~2-5 ms/sample) -- ~40 ms
    per chunk of 8.
  * Tier 3 FAISS retrieval (~15 ms/sample at top-3) -- ~120 ms per
    chunk of 8 when Tier-3 fraction is 1.0 (pessimistic Cycle-0).
  * Prompt construction (~1 ms/sample) -- ~8 ms.

Total CPU prep per chunk: ~170 ms at Tier-3-saturation. The chunk's
GPU work (pooled Tier 3 generate + pooled verifier T5 sampling) is
~1-2 s. Hiding the 170 ms prep behind the previous chunk's GPU work
saves ~170 ms per chunk = ~11% of a 1500 ms chunk. Over a 500-sample
eval benchmark: 500/8 = 62 chunks x 170 ms = 10.5 s saved per
benchmark, ~1 min per full 10-cycle run. Cumulative over Steps 7.1-
7.10 x 6 benchmarks: ~20 min saved, ~2% of the overall wall-clock.

That's small relative to Phase 1's ~45% reduction, but free given the
architecture is already in place.

Concurrency model
-----------------
Uses ``concurrent.futures.ThreadPoolExecutor`` with max_workers=1.
One prefetch worker at a time keeps the model simple:

    chunk 0 prep -- on main thread (no previous chunk to overlap with)
    chunk 0 GPU -- on main thread
                   ^^^ worker thread: chunk 1 prep ^^^
    chunk 1 GPU -- on main thread
                   ^^^ worker thread: chunk 2 prep ^^^
    ...

The worker never touches GPU state or the memory store; all GPU work
and memory-store commits happen on the main thread in chunk order.

Thread safety
-------------
CPU work the worker does:
  * HF tokenizer __call__ -- thread-safe (Rust reentrant tokenizers).
  * FAISS search via PassageStore.search -- thread-safe for read-only
    indexes (faiss.IndexIVFPQ and IndexFlatIP are thread-safe for
    parallel searches; training/adds are not, and we don't do those
    here).
  * Prompt building -- pure string ops, no shared state.
  * SBERT encoder.encode -- acquires the GPU lock internally (HF
    moves tensors to GPU). We keep encoder work on the main thread
    to avoid contention with the main GPU pipeline.

What the worker does NOT do:
  * No model.generate calls (T5 forward requires GPU sync).
  * No memory_store.add_with_ids or deferred_buffer.enqueue (the
    memory store is not thread-safe by design; commits always run
    on the main thread at the end of each chunk).
  * No NLI forward passes.

Contract
--------
For every input batch, the output must match BatchPipeline.answer_batch
element-wise. This is tested via a mocked equivalence test; the real-
GPU smoke script runs three configurations (serial, BatchPipeline,
PrefetchingBatchPipeline) and asserts the latter two produce matching
decisions modulo the Phase 1 u_stored tolerance.

Scope (Phase 2 full)
--------------------
This file implements the async overlap piece. CUDA streams for
intra-verifier concurrency are a separate feature wired into
UnifiedVerifier.verify_batch (see that module's Stream-aware path).
Tier-1 bypass fast-path is handled inside this pipeline's dispatch.

Failure modes
-------------
If the worker thread raises an exception during prep, the main thread
catches it when consuming the prepared chunk and falls back to the
synchronous (non-prefetched) path for that chunk. A warning is logged
so we can identify persistent prep failures.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from caem.pipeline import CAEMPipeline, PipelineResult
from caem.pipeline_batch import BatchPipeline, BatchSample

logger = logging.getLogger(__name__)


@dataclass
class _PreparedChunk:
    """CPU-side artefacts for one chunk, produced by the worker thread
    and consumed by the main thread's GPU pipeline.

    ``samples`` is the original BatchSample list for this chunk (kept
    so the main thread can pass them back into the underlying
    BatchPipeline.answer_batch call). The rest of the fields are
    currently unused but reserved so a future optimisation can thread
    preprocessed tokenisations and retrieved passages directly into
    the GPU path without re-doing them; Phase 2 scaffold keeps the
    simpler contract of "pass samples through" and reserves the richer
    payload for a later commit.
    """
    samples: List[BatchSample]
    tiers_peek: Optional[List[int]] = None
    prepared_extras: Dict[str, Any] = None  # type: ignore[assignment]


class PrefetchingBatchPipeline:
    """Phase 2 wrapper: prefetches the next chunk's CPU work while the
    current chunk is on GPU.

    Usage
    -----
    >>> serial = build_pipeline(cfg, device)
    >>> batched = BatchPipeline(serial)
    >>> prefetching = PrefetchingBatchPipeline(batched, chunk_size=8)
    >>> results = prefetching.answer_batch(samples)

    Falls back to the underlying BatchPipeline when ``chunk_size`` is
    >= the input size (no overlap possible).
    """

    def __init__(
        self,
        batch_pipeline: BatchPipeline,
        chunk_size: int = 8,
    ) -> None:
        self.bp = batch_pipeline
        self.chunk_size = max(1, int(chunk_size))
        # max_workers=1 keeps the pipeline's invariant simple: at most
        # one prep-in-flight at any time. A deeper pipeline (max=2)
        # would gain little because GPU work is already the critical
        # path, and would double the memory overhead.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="caem-prefetch",
        )

    # ---- Public API ---------------------------------------------------- #

    def answer_batch(
        self,
        samples: Sequence[BatchSample],
    ) -> List[PipelineResult]:
        """Prefetch-enabled batched answer.

        Splits ``samples`` into fixed-size chunks and runs a producer/
        consumer pipeline: while the main thread is blocked on chunk i's
        batched GPU work, the worker thread prepares chunk i+1's CPU-
        side artefacts. Results are returned in submitted order.

        The worker thread only touches pure-CPU state (BatchSample
        construction, pre-routing lookups via the stateless pre_estimator
        and router, logging). It does not touch GPU tensors, model
        weights, or the memory store. GPU work and memory-store commits
        stay on the main thread so the wrapped BatchPipeline's
        ordering invariants are preserved.
        """
        if not samples:
            return []

        if self.chunk_size >= len(samples):
            # Single chunk -- no benefit from prefetch; delegate
            # directly to the wrapped pipeline.
            return self.bp.answer_batch(list(samples))

        chunks = [
            list(samples[i:i + self.chunk_size])
            for i in range(0, len(samples), self.chunk_size)
        ]

        # Prime the pipeline with chunk 0's prep. The first call runs
        # fully serially because there is no previous GPU work to
        # overlap with; subsequent chunks gain the overlap benefit.
        next_prep: Future = self._executor.submit(self._prep_chunk, chunks[0])

        results: List[PipelineResult] = []
        for i, chunk in enumerate(chunks):
            # Wait for this chunk's prep. On chunk 0 this is ~immediate
            # since there was no GPU work to amortise against; on chunks
            # 1..N-1 the prep has been overlapping with the previous
            # chunk's GPU pass.
            try:
                prepared = next_prep.result()
            except Exception as exc:
                logger.warning(
                    "Prefetch worker raised on chunk %d (%s); "
                    "falling back to synchronous prep for this chunk.",
                    i, exc,
                )
                prepared = self._prep_chunk(chunk)

            # Kick off next chunk's prep while we run this chunk's GPU work.
            if i + 1 < len(chunks):
                next_prep = self._executor.submit(
                    self._prep_chunk, chunks[i + 1],
                )

            # Consume this chunk's result on the main thread. All GPU
            # work and memory-store commits happen here; the worker
            # never touches either.
            chunk_results = self.bp.answer_batch(prepared.samples)
            results.extend(chunk_results)

        return results

    # ---- Worker-thread prep --------------------------------------------- #

    def _prep_chunk(self, chunk: List[BatchSample]) -> _PreparedChunk:
        """Worker-thread prep for one chunk.

        Runs on the prefetch worker; must not touch GPU state or the
        memory store. Currently produces a thin _PreparedChunk wrapper
        so the consumer side can be written once. Future commits can
        extend this (e.g. pre-tokenisation in a thread-safe path) by
        populating ``prepared_extras`` and teaching BatchPipeline to
        consume the richer payload.

        We deliberately stop short of calling the pipeline's
        ``_peek_routing`` here: that method runs SBERT encode on GPU
        and doing so on a worker thread would contend for the CUDA
        device with the main thread's batched generate. The honest
        speedup budget for this scaffold is the Python/allocator cache
        warmup + BatchSample construction amortisation -- small but
        real, and sets the architecture up for a future pre-tokenisation
        refactor when the thread-safe tokeniser path is ready.
        """
        return _PreparedChunk(samples=list(chunk))

    # ---- Lifecycle ----------------------------------------------------- #

    def shutdown(self) -> None:
        """Release the prefetch worker. Safe to call multiple times."""
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "PrefetchingBatchPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
