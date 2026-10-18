---
name: Tier-1 amortisation diagnostic (post-trajectory)
description: How to measure true per-tier latency once cycle-N (final) closes; resolves the batch-amortised latency artefact and supplies the empirical receipt for H5.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Run this after the last cycle closes (cycle 5 if user stops there, cycle 10 if they push through).** It produces the Tier-1 cost-amortisation receipt the live `latency_ms` column can't supply.

**Why a separate diagnostic is needed.** `BatchPipeline.answer_batch` at `caem/pipeline_batch.py:463-464` averages the batch wall-time across all samples: `per_sample_ms = heavy_elapsed_ms / N`. So in every batched eval JSON, a Tier-1 hit logs the same latency as the Tier-3 queries in the same batch (we saw 10300 ms for a Tier-1 hit in cycle-3 stream-chunk). The per-tier intrinsic latencies the slide deck claims ($\sim$50 ms / $\sim$3-5 s / $\sim$10 s) cannot be verified from the existing JSONs.

**How.** Use the serial `Pipeline` class directly (not `BatchPipeline`) at bs=1. Serial path at `caem/pipeline.py:644-645` uses real `time.perf_counter()` per sample.

**Cycle-by-cycle reconstruction works without re-running per cycle.** Per-tier intrinsic costs are approximately constant across cycles (FAISS lookup grows only $O(\log n)$ with store size; LoRA inference is invariant; passage index is fixed). So one measurement on cycle-N state plus the per-cycle tier-share trajectory (already in eval JSONs `tier` field) lets us compute every cycle's pooled wall-time as $\sum_t \pi_t^{(N)} c_t$.

**Eval set.** 100 queries, deliberately constructed:
- 50 paraphrased FEVER/TQA/CSQA claims of cycle-stored entries (should hit Tier 1; verifies near-duplicate retrieval works)
- 30 in-distribution but novel queries (should hit Tier 2)
- 20 truly OOD queries (should hit Tier 3)

**Measurements per query.** `actual_tier`, `latency_ms` from `time.perf_counter()`, `pipeline.answer().latency_ms` (sanity check), `em` against gold.

**Cycle-0 sanity pass.** Also run against the cycle-0 pipeline state (HF snapshot `aksaN000/caem-passage-index-21m`, `pre_main_snapshot/`). If $c_1, c_2, c_3$ at cycle 0 are within ~10% of the final-cycle values, the per-cycle decomposition is licensed.

**Output artefact.** `outputs/diagnostics/tier_latency.json` with per-tier descriptives + `outputs/diagnostics/per_cycle_pooled_latency.csv` derived from tier shares × intrinsic costs.

**Time budget.** Pipeline setup ~1 h, 100-query bs=1 inference ~10 min, analysis ~30 min. Total ~2 h.

**Cite in Ch5.** This is the empirical receipt for H5 ("Tier-1 cost amortisation grows as memory fills"). Without it, H5 is reported as "supported via per-cycle Tier-share trajectory; per-tier intrinsic latency measured separately in the post-trajectory diagnostic."

**Task ID.** Existing task #154.
