---
name: T1 retrieval has 9% failure rate driven by FEVER prompt-boilerplate similarity inflation
description: Of 55 T1 hits in Phase 1e through C2, 87% true match / 4% lucky / 9% failed. All failures are FEVER polarity mismatches where the prefix dominates the embedding. Two fixes registered.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Finding logged 2026-05-23 during Phase 1e C2 close analysis.**

**Empirical breakdown across all 55 T1 hits** (eval + stream + cal-fold through Phase 1e C2):
- 48 (87.3%) TRUE T1 (genuine semantic match, em=1)
- 2 (3.6%) RISKY (em=1 by luck — matched unrelated stored entry)
- 5 (9.1%) FAILED (em=0, wrong answer)

**All 5 failures are FEVER claim verification with polarity/specificity mismatches:**
- "Game of Thrones did NOT have multiple actors" matched "GoT won outstanding cinematography"
- "Planet of the Apes is NOT a film" matched "Dawn of POTA is part of the series"
- "Baahubali LOST the National Film Award" matched "Baahubali WAS AWARDED the NFA"
- "The first movie of Divergent is NOT a..." matched "...is a..."
- "John Constantine is a Marvel comics character" (he's DC) matched "JC is a comics character"

**Similarity distribution:**
- em=1 hits: min 0.918, mean 0.982, p10 0.953, median 0.995 (most cluster at sim≈1.0)
- em=0 hits: 0.904, 0.932, 0.934, 0.975, 0.980 — two of which are inside em=1 distribution
- All 5 failures had routing_score 0.902-0.913, JUST over the tier1_combined_threshold of 0.90

**Root cause — FEVER prompt boilerplate inflates similarity:**
Every FEVER query starts with the 70-token prefix `"Answer with one of: supports, refutes, not enough info. Claim:"`. SBERT averages over prefix + claim, so two FEVER queries with completely opposite claims still share ~80% token overlap. The 0.934 Baahubali pair's similarity is dominated by the shared prefix; the actual claims ("lost" vs "was awarded") contribute little to the cosine. This is a structural problem on FEVER specifically — TQA and CSQA don't have it.

**Why this matters for the report:**

1. **T1's currently-rare failures will become common in production.** Held-out eval has T1 rate ~0%; production with recurring queries would have T1 rate much higher. The 9% failure rate becomes a real safety concern.

2. **Ch5 §threats receipt is clean:** quote the Baahubali example. "Episodic memory retrieval is polarity-sensitive on claim-verification tasks; the FEVER prompt boilerplate inflates cosine similarity, leading to 9% T1 failure rate on this benchmark."

3. **Ch6 §future-work has two concrete actionable fixes:**
   - **Quick fix:** raise `tier1_combined_threshold` from 0.90 → 0.92. Blocks all 5 observed failures (max failure routing_score = 0.913) while preserving most em=1 hits (which cluster at routing_score > 0.95).
   - **Better fix:** strip the FEVER prompt prefix before SBERT encoding for both indexing and lookup. The prefix dominates the embedding and is causally responsible for the polarity-blindness. Single-line change in `caem/memory/store.py` and `caem/pipeline.py` at the encode call sites.

4. **The 9% failure rate is benchmark-specific.** TQA and CSQA T1 hits are essentially all clean (exact matches with paraphrased duplicates). The failure mode is FEVER-only and the fix is FEVER-only.

**How to apply:**

- Land the Baahubali example as a worked-case in Ch5 §threats (storage gate fragility section).
- Pre-register the two fixes in Ch6 §future-work as bench-conditional T1 improvements.
- The "T1 hits are mostly noise on held-out eval" finding (~0% rate) belongs in Ch5 §evaluation-protocol — eval understates T1's deployment value because eval_ids are disjoint from train_ids.
- For the abstract: do NOT lead with T1 contribution; T1's actual production value is realized via SIL pool feedback, not direct memory lookup. Frame as "episodic memory provides high-precision training data for cyclic self-improvement" not "memory directly answers queries."

**Verification path:**
Computed by re-embedding all 55 T1 queries via `sentence-transformers/all-mpnet-base-v2` (the actual encoder used by `caem/memory/store.py`) and searching the C2 FAISS index for top-1 cosine. Script run 2026-05-23 in conversation transcript; can be re-run any time against `outputs/full_run/memory_store_cycle_2.faiss`.

**Watch for:**
- Does C3+ show more T1 hits (memory growing)? If yes, the 9% failure becomes more visible in absolute terms.
- Does the failure rate stay FEVER-specific or spread to other benches as memory diversifies?
- If user later applies the "strip prefix" fix, expect FEVER T1 hit rate to DROP (currently ~0.85% on streams) because spurious matches disappear — but the surviving T1 hits become higher quality.
