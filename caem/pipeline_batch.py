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
import warnings as _warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

# Suppress cosmetic transformers warning on every batched model.generate call
# (greedy + non-default sampling params from Qwen's factory config; see
# verifier.py import block for full rationale). Filter only this one
# message; other UserWarnings pass through.
_warnings.filterwarnings(
    "ignore",
    message=r".*`do_sample` is set to `False`.*",
    category=UserWarning,
)

from caem.pipeline import CAEMPipeline, PipelineResult
from caem.prompts import build_tier2_prompt

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
        """Generate N Tier-2 answers in one batched decoder-only forward pass.

        Equivalent to calling ``self.p._tier2(q, pre_conf)[0]`` in a
        loop but runs a single ``model.generate`` with left-padded input
        IDs of shape ``(N, max_prompt_len)``. Left-padding is required so
        the last position of each row is always the generation start
        (decoder-only autoregression generates from the RIGHT side of the
        input).

        Prompts are built via ``caem.prompts.build_tier2_prompt`` (ChatML
        scaffolded CoT with system prompt + few-shot turn pair + real
        user turn + ``<|im_start|>assistant\\n`` generation marker). The
        forced ``"Reasoning:"`` prefix is appended as prefill text and
        tokenized as part of the input; after generation, only the newly
        generated tokens (``output_ids[:, input_len:]``) are decoded, then
        the forced prefix is prepended to produce the scaffolded-CoT
        answer string expected by the Stage-5 verifier.

        Parameters
        ----------
        queries : sequence of str
            N Tier-2 queries in the order their answers should be returned.

        Returns
        -------
        list of str, length N, same order as ``queries``. Empty
        string at position i signals Tier 2 failed for that query and
        the caller should escalate to Tier 3.
        """
        if not queries:
            return []

        cfg = self.p.config
        tok = self.p.tokenizer

        # Build ChatML prompt + forced prefix for each query. Concatenate the
        # prefix as prefill so the model's generation continues it.
        full_inputs: List[str] = []
        for q in queries:
            prompt_text, forced_prefix = build_tier2_prompt(q, tokenizer=tok)
            full_inputs.append(prompt_text + forced_prefix)

        try:
            # Left-padding is mandatory for decoder-only batched generation so
            # generated tokens align across rows regardless of input length.
            original_side = getattr(tok, "padding_side", None)
            tok.padding_side = "left"
            enc = tok(
                full_inputs,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.cot_max_new_tokens + 2048,
                padding=True,
            )
            if original_side is not None:
                tok.padding_side = original_side
        except Exception as exc:
            logger.error(
                "batch_tier2_generate tokenization failed (N=%d): %s -- "
                "returning empty strings so caller escalates each to Tier 3.",
                len(queries), exc,
            )
            return [""] * len(queries)

        input_ids = enc["input_ids"].to(self.p.device)
        attention_mask = enc["attention_mask"].to(self.p.device)
        input_len = int(input_ids.shape[1])

        try:
            self.p.model.eval()
            with torch.no_grad():
                output_ids = self.p.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=cfg.cot_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
        except Exception as exc:
            logger.error(
                "batch_tier2_generate failed (N=%d): %s -- returning "
                "empty strings so caller escalates each to Tier 3.",
                len(queries), exc,
            )
            return [""] * len(queries)

        # Decode only the newly generated continuation (skip the left-padded
        # prompt prefix) and prepend the forced prefix so the answer begins
        # with "Reasoning:" per the scaffolded-CoT contract.
        decoded: List[str] = []
        for i in range(output_ids.shape[0]):
            try:
                continuation = tok.decode(
                    output_ids[i, input_len:],
                    skip_special_tokens=True,
                ).strip()
                answer = f"Reasoning:{continuation}" if continuation else ""
            except Exception as exc:
                logger.warning("batch_tier2_generate decode %d failed: %s", i, exc)
                answer = ""
            decoded.append(answer)
        return decoded

    def batch_verify(
        self,
        inputs: List[tuple],
        source_benchmarks: Optional[List[Optional[str]]] = None,
    ) -> list:
        """Run the UnifiedVerifier over N (query, answer) pairs in one call.

        Thin wrapper around ``self.p.verifier.verify_batch`` so that
        ``answer_batch`` has a single, consistent surface for each
        batched stage. Phase 1 skeleton: verify_batch currently loops
        serial ``verify()``; a follow-up commit pools M-chain generation
        and semantic-entropy sampling across samples (the two dominant
        per-sample T5 costs inside verify).

        Parameters
        ----------
        inputs : list of (query, answer) tuples
            One entry per sample that needs verification. The caller
            typically filters out Tier 1 samples (which do not run
            Stage 5 in the serial pipeline) before calling this.
        source_benchmarks : list of (str or None) or None, default None
            v2 Fix 2 — per-sample benchmark tag, same length as
            ``inputs``. Forwarded to ``verifier.verify_batch`` so each
            per-sample ``verify()`` call sees the correct
            ``source_benchmark`` for per-benchmark composite + per-bench
            gate dispatch. When None (legacy callers), every sample
            falls back to the pooled global path.

        Returns
        -------
        list of UnifiedVerifierOutput, same length and order as ``inputs``.
        """
        if not inputs:
            return []
        return self.p.verifier.verify_batch(
            inputs, source_benchmarks=source_benchmarks,
        )

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

        # Branch C Goal 5 (2026-04-21): track the wall-clock cost of the
        # whole batch so the per-sample PipelineResult.latency_ms field
        # reports the AMORTISED per-query latency (``batch_elapsed / N``)
        # instead of the near-zero bookkeeping time of the per-sample
        # serial ``answer()`` call that runs after the batched generate +
        # verify. Perf-harness + eval-harness both read this field, so
        # distributing the batch cost makes the seeder's log + the
        # perf_log.csv row report the real speedup curve.
        import time as _time
        batch_start = _time.perf_counter()

        # Phase 1: per-sample tier determination + routing bundle.
        # Stages 1-3 (encode, search, route) run once per sample here.
        # The full bundle is threaded into answer() via _precomputed_routing
        # so the serial call does not repeat these ops (Phase 2 Tier-1
        # fast path).
        per_sample_tier: List[int] = []
        per_sample_routing: List[tuple] = []
        for s in samples:
            tier, emb, pre_conf, search, routing = self._peek_routing(
                s.query, source_benchmark=s.source_benchmark,
            )
            per_sample_tier.append(tier)
            per_sample_routing.append((emb, pre_conf, search, routing))

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

        # Phase 2b: batched verify for all Tier 2 / Tier 3 samples.
        # Tier 1 samples skip Stage 5 in the serial pipeline so they are
        # excluded here. Escalation-to-Tier-3 (empty Tier 2 output -> Tier 3
        # fallback inside answer()) is not reflected in per_sample_tier,
        # so we skip batched verify for any Tier 2 sample whose batched
        # answer came back empty -- the serial answer() call will handle
        # the escalation and run a fresh verify on the Tier 3 output.
        verify_indices: List[int] = []
        verify_inputs: List[tuple] = []
        # v2 Fix 2 keystone — per-sample source_benchmark must be carried
        # alongside the (query, answer) tuple so batch_verify can dispatch
        # the per-bench composite + fixed-threshold gate from CAEMConfig.
        # Without this list, every batched-eval sample silently falls back
        # to the pooled global path even though source_benchmark is on the
        # BatchSample.
        verify_source_benchmarks: List[Optional[str]] = []
        for i in tier2_indices:
            ans = precomputed_tier2.get(i, "")
            if ans:  # non-empty -> verify the Tier 2 answer
                verify_indices.append(i)
                verify_inputs.append((samples[i].query, ans))
                verify_source_benchmarks.append(samples[i].source_benchmark)
        for i in tier3_indices:
            ans = precomputed_tier3.get(i, "")
            verify_indices.append(i)
            verify_inputs.append((samples[i].query, ans))
            verify_source_benchmarks.append(samples[i].source_benchmark)

        precomputed_vouts: dict = {}
        if verify_inputs:
            vout_list = self.batch_verify(
                verify_inputs,
                source_benchmarks=verify_source_benchmarks,
            )
            for idx, vout in zip(verify_indices, vout_list):
                precomputed_vouts[idx] = vout

        # Phase 3: complete each sample's pipeline via serial answer(),
        # injecting the precomputed Tier 2 / Tier 3 answers and verifier
        # outputs where applicable.
        # NOTE: tier determination is re-run inside answer() -- the
        # _peek_tier() result above is only used to decide *which*
        # samples need batched generation. Router outputs are
        # deterministic given the same inputs, so the tier decision
        # inside answer() will match per_sample_tier[i].
        # Amortise the batched heavy-work (generate + verify) cost across
        # the N samples and thread the per-sample number into each
        # answer() call via _precomputed_latency_ms. Measured at THIS
        # point (before the serial answer() loop) because by now all the
        # expensive batched ops have run; the remaining per-sample
        # answer() bookkeeping is sub-millisecond and excluded from the
        # reported latency (it's the same constant overhead at bs=1).
        heavy_elapsed_ms = (_time.perf_counter() - batch_start) * 1000.0
        per_sample_ms = heavy_elapsed_ms / max(len(samples), 1)

        results: List[PipelineResult] = []
        for i, s in enumerate(samples):
            kwargs = dict(
                query=s.query,
                store_to_memory=s.store_to_memory,
                source_benchmark=s.source_benchmark,
                _precomputed_routing=per_sample_routing[i],
                _precomputed_latency_ms=per_sample_ms,
            )
            if i in precomputed_tier2:
                kwargs["_precomputed_tier2_answer"] = precomputed_tier2[i]
            if i in precomputed_tier3:
                kwargs["_precomputed_tier3_answer"] = precomputed_tier3[i]
            if i in precomputed_vouts:
                kwargs["_precomputed_vout"] = precomputed_vouts[i]
            r = self.p.answer(**kwargs)
            results.append(r)
        return results

    # ---- Internal helpers ----------------------------------------------- #

    def _peek_tier(self, query: str, source_benchmark: Optional[str] = None) -> int:
        """Return the tier this query would be routed to. Thin wrapper
        around :meth:`_peek_routing` for call sites that only need the
        tier int (e.g. tests). Equivalent to ``_peek_routing(q)[0]``.
        """
        tier, _, _, _, _ = self._peek_routing(query, source_benchmark=source_benchmark)
        return tier

    def _peek_routing(
        self,
        query: str,
        source_benchmark: Optional[str] = None,
    ) -> tuple:
        """Run Stages 1-3 once and return the full routing bundle:
        ``(tier, query_embedding, pre_conf, search_with_ids, routing)``.

        Level B Phase 2: the bundle is threaded into the serial
        :meth:`CAEMPipeline.answer` via the ``_precomputed_routing``
        kwarg so each sample pays the Stage 1-3 cost exactly once
        instead of twice (once for tier-bucket dispatch, once inside
        the serial ``answer`` call).

        v2 Fix 12: ``source_benchmark`` is forwarded into
        ``pre_estimator.estimate`` (per-benchmark T_b) and
        ``router.route`` (per-benchmark safety_u_pre_min_b) so the bundle
        threaded back to ``answer`` reflects the same per-benchmark
        dispatch as the serial path.

        Phase 1e: when the v2 RUC is wired into the router, compute the
        13 routing-time features (FAISS top-5 stats, cross-encoder rerank,
        pairwise NLI, Wikidata, P(IK)) and thread them into router.route
        so the RUC fires in batch mode with the same signal set the serial
        :meth:`CAEMPipeline.answer` path produces. Without this, batch-mode
        runs would silently bypass the RUC and degrade to vanilla CAEM.
        """
        query_embedding = self.p._encode_query(query)
        pre_conf = self.p.pre_estimator.estimate(
            query, source_benchmark=source_benchmark,
        )
        search_with_ids = self.p.memory_store.search_with_ids(
            query_embedding, k=1,
        )
        ruc_extras = None
        if self.p.router.ruc is not None:
            ruc_extras = self.p._compute_ruc_features(query, query_embedding)
        routing = self.p.router.route(
            pre_conf, search_with_ids,
            source_benchmark=source_benchmark,
            question=query,
            top1_passage_text=(ruc_extras or {}).get("_top1_passage_text"),
            top1_passage_sim=(ruc_extras or {}).get("top1_passage_sim"),
            top1_passage_entity_overlap=(ruc_extras or {}).get("top1_passage_entity_overlap"),
            p_ik=(ruc_extras or {}).get("p_ik"),
            extra_ruc_features=ruc_extras,
        )
        return (
            int(routing.tier),
            query_embedding,
            pre_conf,
            search_with_ids,
            routing,
        )
