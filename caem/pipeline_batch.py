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

Backend-agnosticism
-------------------
Level B's batched-verifier logic calls the polymorphic
``batch_entail_prob / batch_contradict_prob / batch_argmax_label``
methods that both ``_NLIEnsemble`` (RoBERTa) and ``_MiniCheckJudge``
(MiniCheck) expose. A future ``_HybridJudge`` would expose the same
interface. This means ~95 percent of Level B code is agnostic to the
Step 5.5 backend decision; only the small interface-call layer is
backend-aware, and even that uses a polymorphic contract.

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

2. **Batched Tier 2 generate (this commit)**: ``_batch_tier2_generate``
   helper builds N Tier-2 prompts, tokenises with padding, runs a
   single batched ``model.generate`` call, and returns the decoded
   answers in input order. Not yet wired into ``answer_batch`` —
   that happens in commit 3. Backend-agnostic (just T5).

3. **Wire Tier 2 batching into answer_batch (next)**: per-sample
   Stages 1-3 (encode, search, route) then tier-bucket split, then
   batched Tier 2 generate for Tier 2 samples, per-sample Tier 3,
   per-sample verify.

4. **Batched T5 generate for K-chain self-consistency and atomic
   decomposition (next)**: these are the two other dominant T5
   forwards; each currently runs once per sample, so cross-sample
   batching amortises launch overhead.

5. **Batched verifier** (``_verify_batch``): new method on
   UnifiedVerifier that accepts a list of (query, answer, passages)
   tuples and runs the nine-signal pipeline with all pairs pooled
   into a single MiniCheck forward. Biggest single speedup (~30-40%)
   because the MiniCheck call is the dominant GPU work.

6. **Cross-sample K-chain pooling** (stretch): the K=10
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

import torch

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

    # ---- Batched generation helpers ------------------------------------- #

    def batch_tier2_generate(self, queries: Sequence[str]) -> List[str]:
        """Generate N Tier-2 answers in one batched T5 forward pass.

        Equivalent to calling ``self.p._tier2(q, pre_conf)[0]`` in a
        loop but runs a single ``model.generate`` with padded input IDs
        of shape ``(N, max_prompt_len)``. The output is decoded row by
        row; positions that hit a decoder failure fall back to an
        empty string (caller escalates that to Tier 3).

        Parameters
        ----------
        queries : sequence of str
            N Tier-2 queries in the order their answers should be
            returned.

        Returns
        -------
        list of str, length N, same order as ``queries``. Empty
        string at position i signals Tier 2 failed for that query and
        the caller should escalate to Tier 3.
        """
        if not queries:
            return []

        cfg = self.p.config
        prompts = [self.p._build_tier2_prompt(q) for q in queries]

        enc = self.p.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = enc["input_ids"].to(self.p.device)
        attention_mask = enc["attention_mask"].to(self.p.device)

        try:
            self.p.model.eval()
            with torch.no_grad():
                output_ids = self.p.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=cfg.cot_max_new_tokens,
                    do_sample=False,
                )
        except Exception as exc:
            logger.error(
                "batch_tier2_generate failed (N=%d): %s -- returning "
                "empty strings so caller escalates each to Tier 3.",
                len(queries), exc,
            )
            return [""] * len(queries)

        # Decode each row. Empty decodes become empty strings so the
        # caller can route them to Tier 3 the same way the serial
        # path does.
        decoded: List[str] = []
        for i in range(output_ids.shape[0]):
            try:
                s = self.p.tokenizer.decode(
                    output_ids[i], skip_special_tokens=True,
                ).strip()
            except Exception as exc:
                logger.warning("batch_tier2_generate decode %d failed: %s", i, exc)
                s = ""
            decoded.append(s)
        return decoded

    def batch_tier3_generate(self, queries: Sequence[str]) -> List[str]:
        """Generate N Tier-3 RAG answers in one batched T5 forward pass.

        Equivalent to calling ``self.p._tier3(q)`` in a loop: each query
        is retrieved + prompt-built serially (cheap FAISS + string ops),
        then a single padded ``model.generate`` runs the expensive T5
        pass over all N prompts at once. Answers come back in input order.

        Empty string at position i signals a decode/generate failure;
        the caller preserves it (matches serial contract: ``_tier3``
        also returns "" on total failure, and the verifier handles an
        empty answer just like the serial path).
        """
        if not queries:
            return []
        return self.p.rag.generate_batch(list(queries))

    # ---- Public API ------------------------------------------------------ #

    def answer_batch(
        self,
        samples: Sequence[BatchSample],
    ) -> List[PipelineResult]:
        """Process N samples and return N results in the same order.

        Level B Phase 1 step 3: tier-bucketed execution with batched
        Tier 2 generation.

        Flow:
          1. For each sample, determine routing tier via serial
             ``_encode_query`` + ``pre_estimator`` + ``memory_store.search``
             + ``router.route``. (These are cheap encoder/FAISS ops;
             will be batched in a later commit.)
          2. Bucket sample indices by tier.
          3. Batch-generate Tier 2 answers via ``batch_tier2_generate``
             (single padded T5 forward pass).
          4. For each sample, call serial ``answer()`` to complete the
             pipeline (Tier 1 retrieval / Tier 3 RAG generate + verify
             + commit). Tier 2 samples pass the precomputed answer via
             the ``_precomputed_tier2_answer`` kwarg so the
             serial ``_tier2`` call is skipped.

        Contract: the returned list has the same length as ``samples``
        and preserves submission order. Memory-store commits happen in
        that same order so running ``answer_batch(S)`` is observationally
        equivalent to looping ``p.answer(s) for s in S`` (same memory
        state, same stored entries, same STORE ordering).
        """
        if not samples:
            return []

        # Phase 1: per-sample tier determination.
        # We need tier info to know which samples to batch-generate for.
        # This is a cheap set of ops (SBERT encode + FAISS search +
        # router dispatch); batching them is a later optimisation.
        per_sample_tier: List[int] = []
        for s in samples:
            tier = self._peek_tier(s.query)
            per_sample_tier.append(tier)

        # Phase 2: collect Tier 2 / Tier 3 samples + batch-generate their
        # answers. Tier 1 samples produce no generation work.
        tier2_indices = [i for i, t in enumerate(per_sample_tier) if t == 2]
        tier3_indices = [i for i, t in enumerate(per_sample_tier) if t == 3]

        precomputed_tier2: dict[int, str] = {}
        if tier2_indices:
            tier2_queries = [samples[i].query for i in tier2_indices]
            tier2_answers = self.batch_tier2_generate(tier2_queries)
            for idx, ans in zip(tier2_indices, tier2_answers):
                precomputed_tier2[idx] = ans

        precomputed_tier3: dict[int, str] = {}
        if tier3_indices:
            tier3_queries = [samples[i].query for i in tier3_indices]
            tier3_answers = self.batch_tier3_generate(tier3_queries)
            for idx, ans in zip(tier3_indices, tier3_answers):
                precomputed_tier3[idx] = ans

        # Phase 3: complete each sample's pipeline via serial answer(),
        # injecting the precomputed Tier 2 / Tier 3 answers where applicable.
        # NOTE: tier determination is re-run inside answer() -- the
        # _peek_tier() result above is only used to decide *which*
        # samples need batched generation. Router outputs are
        # deterministic given the same inputs, so the tier decision
        # inside answer() will match per_sample_tier[i].
        results: List[PipelineResult] = []
        for i, s in enumerate(samples):
            kwargs = dict(
                query=s.query,
                store_to_memory=s.store_to_memory,
                source_benchmark=s.source_benchmark,
            )
            if i in precomputed_tier2:
                kwargs["_precomputed_tier2_answer"] = precomputed_tier2[i]
            if i in precomputed_tier3:
                kwargs["_precomputed_tier3_answer"] = precomputed_tier3[i]
            r = self.p.answer(**kwargs)
            results.append(r)
        return results

    # ---- Internal helpers ----------------------------------------------- #

    def _peek_tier(self, query: str) -> int:
        """Return the tier this query would be routed to, without
        generating an answer. Runs the cheap upstream ops (encode,
        search, pre-route, router) and inspects the routing decision.

        This duplicates work that ``answer()`` will do again, but the
        duplicated work is inexpensive compared to the Tier 2/3
        generate calls. Future commits batch these upstream ops
        across samples to eliminate the duplication cost.
        """
        query_embedding = self.p._encode_query(query)
        pre_conf = self.p.pre_estimator.estimate(query)
        search_with_ids = self.p.memory_store.search_with_ids(
            query_embedding, k=1,
        )
        routing = self.p.router.route(pre_conf, search_with_ids)
        return int(routing.tier)
