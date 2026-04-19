"""
caem/pipeline_batch.py
=======================
Level B (static batching) for CAEM — Phase 1 of the concurrent-pipeline
optimization. Processes N samples in lockstep through each of the eight
pipeline stages, batching where the GPU naturally supports it (SBERT
encode, T5 generate, MiniCheck scoring) and serializing where the
architecture requires it (per-sample routing decision, memory-store
commits at batch end).

Design
------
The per-sample ``CAEMPipeline.answer()`` entry point remains unchanged
and backward-compatible. ``answer_batch(samples)`` is a new entry point
that takes a list of N queries and returns a list of N ``PipelineResult``
records in the same order.

The batched path is numerically equivalent to calling ``answer()`` N
times sequentially (modulo the tiny order-of-operations differences
bf16 tensors exhibit when the batch dimension changes). The
equivalence test ``tests/test_pipeline_batch_equivalence.py`` asserts
per-sample fields match within ``atol=1e-3`` on the u_stored scalar
and ``atol=1e-2`` on the individual verifier signals (tolerances
chosen to survive bf16 accumulation noise).

Why batching at all
-------------------
Under the serial pipeline, per-sample wall-clock is ~8 s on the 5090
with MiniCheck:
    ~5.3 s GPU-active + ~2.5 s Python orchestration overhead + 0.2 s I/O
GPU utilisation is ~18% because between each of the ~10 forward
passes per sample, Python is preparing the next call while the GPU
sits idle. Batch dim = 8 amortises kernel launch overhead by 8x and
keeps the GPU hot; empirically we expect ~1.8x per-sample speedup
(8 s -> 4.5 s) without any change to the numerical pipeline.

Tier heterogeneity
------------------
Samples within a batch may route to different tiers. The batch
processor therefore splits each batch into three tier buckets
(Tier 1 cached retrieval, Tier 2 zero-shot generate, Tier 3 RAG
generate) and runs each bucket through its tier-appropriate stage
with its own batched forward. The verifier (Stage 5) is called
on the combined post-generation set (Tier 2 + Tier 3), which
maximises batch size on the MiniCheck/RoBERTa forward pass.

Memory-store commits
--------------------
EpisodicMemoryStore writes serialize naturally at batch end. Any
decision of STORE commits to memory in sample order (deterministic
for reproducibility). The novelty filter is applied in that same
order so commits cannot race with each other. DeferredBuffer writes
follow the same pattern.

Bit-equivalence expectations
----------------------------
Batched forward passes change the tensor shapes that reach kernel
execution. On bf16 this introduces small numerical differences vs
the serial path (typically < 1e-3 on post-softmax probabilities).
The equivalence test therefore uses tolerances rather than strict
equality; deterministic same-ordering of STORE commits is tested
separately.

Scope (Phase 1 of Level B)
--------------------------
This file implements *static* batching: N samples are processed
lockstep through each stage. True async overlapping (Level B Phase 2)
would add event-loop concurrency between stages; deferred.

Implementation strategy
-----------------------
Built incrementally, commit by commit:

1. **Skeleton (done)**: ``BatchPipeline.answer_batch`` delegates to
   serial ``answer()`` in a loop. Interface works, no speedup yet.
   Validates integration with the harness and memory store.

2. **Batched SBERT + FAISS** (next): pre-compute embeddings and
   nearest-neighbour results for the whole batch in one forward,
   pass them to a thin per-sample wrapper that reuses the serial
   tier dispatch. Small speedup (~5%) but low risk.

3. **Batched T5 generate** (for Tier 2 + Tier 3 buckets): replace
   per-sample ``self.model.generate()`` with a batched call on
   padded input IDs. Moderate speedup (~15-20%) because T5 generate
   is one of the dominant forward passes.

4. **Batched verifier** (``_verify_batch``): new method on
   UnifiedVerifier that accepts a list of (query, answer, passages)
   tuples and runs the nine-signal pipeline with all pairs pooled
   into a single MiniCheck forward. Biggest single speedup (~30-40%)
   because the MiniCheck call is the dominant GPU work.

5. **Cross-sample K-chain pooling** (stretch): the K=10
   self-consistency chain generation currently does K samples *per
   input*; a batched version does K * N samples in one call. Small
   additional gain on top of #4.

Success criteria (test gate before launching Step 7 under this path):
  1. All 104+ existing tests pass unchanged.
  2. tests/test_pipeline_batch_equivalence.py passes on 50 samples.
  3. scripts/run_cyclic_ablation.py --variant full --smoke_test
     completes under the batch path in <= 10 minutes (vs ~36 min serial).
  4. No race conditions observed in a 1000-sample stress test.
  5. GPU utilisation >= 60% during batched run (vs 18% serial).

Revert path: this file and its test + any caller wiring lives on
branch ``level-b-static-batching``. Main remains on serial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from caem.pipeline import CAEMPipeline, PipelineResult

logger = logging.getLogger(__name__)


@dataclass
class BatchSample:
    """One input to ``answer_batch``.

    Kept separate from ``eval/benchmarks.BenchmarkSample`` so the batch
    entry point is independent of the harness payload format.
    """
    query: str
    source_benchmark: Optional[str] = None
    store_to_memory: bool = True


class BatchPipeline:
    """Batched wrapper around CAEMPipeline for Level B static batching.

    Keeps a reference to the underlying serial pipeline and implements
    ``answer_batch`` by running each stage in lockstep with the batch
    dim populated. Uses the serial pipeline's individual primitives
    (``_encode_query``, ``_tier1/2/3``, ``_verify``) but arranges them
    to operate on batch-shaped inputs.

    Not a subclass of CAEMPipeline to keep ownership explicit and to
    preserve the serial path untouched on main.
    """

    def __init__(self, serial_pipeline: CAEMPipeline):
        self.p = serial_pipeline
        # Batch execution writes to the same memory_store / deferred_buffer
        # that the serial pipeline holds. Commit ordering is preserved:
        # within a batch, commits run in submitted-sample order.

    # ---- Public API ------------------------------------------------------ #

    def answer_batch(
        self,
        samples: Sequence[BatchSample],
    ) -> List[PipelineResult]:
        """Process N samples and return N results in the same order.

        Phase 1 skeleton: delegates to serial ``answer()`` for each
        sample. Subsequent commits replace this loop with batched
        execution stage by stage.

        Contract: the returned list has the same length as ``samples``
        and preserves submission order. Memory-store commits happen in
        that same order so running ``answer_batch(S)`` is observationally
        equivalent to looping ``p.answer(s) for s in S`` (same memory
        state, same stored entries).
        """
        if not samples:
            return []

        results: List[PipelineResult] = []
        for s in samples:
            r = self.p.answer(
                query=s.query,
                store_to_memory=s.store_to_memory,
                source_benchmark=s.source_benchmark,
            )
            results.append(r)
        return results
