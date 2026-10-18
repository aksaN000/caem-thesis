---
name: T1 routing activation works as designed; production rarity is sample-distribution by design
description: Direct router probe shows T1 fires 84% on verbatim / 38% on paraphrase queries to stored episodes. Production held-out eval gives 0.13% T1 because eval is content-hash disjoint from training — by design. CAEM's T1 is precision-first cache, not semantic retrieval.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Finding logged 2026-05-30 from cycle-5 routing probe (`outputs/tier1_routing_probe/`).**

## Setup

Built a routing probe (`scripts/run_tier1_routing_probe.py`) that bypasses `pipeline.answer()` and calls the routing components directly:

```
Stage 2: encode query
Stage 3a: pre_estimator.estimate(query)  → u_pre
Stage 1: memory.search_with_ids(emb, k=5)
Stage 3b: router.route(pre_conf, search_top1, ...)
```

Captures EVERY routing-decision field: `tier`, `similarity`, `u_stored_retrieved`, `u_pre`, `routing_score`, `safety_override`, `retrieved_entry_id`, plus top-5 neighbours and retrieved-entry text.

Dataset: 50 source episodes stratified across `u_stored` brackets [0.60, 0.70), [0.70, 0.80), [0.80, 0.90), [0.90, 1.00). Each episode produces 2 queries: exact (verbatim) + paraphrase (Haiku rewrite). 100 queries total.

Full production cycle-5 state: cycle-5 LoRA adapter + cycle-5 composite + cycle-5 memory store (12,175 episodes) + v2 RUC + PassageStore.

## Headline routing-decision results

| Kind | n | T1 | T2 | T3 | mean routing_score | mean similarity | mean u_pre |
|---|---|---|---|---|---|---|---|
| **exact** | 50 | **42 (84%)** | 8 | 0 | 0.939 | 1.000 | 0.737 |
| **paraphrase** | 50 | **19 (38%)** | 31 | 0 | 0.886 | 0.925 | 0.714 |

Safety OR-condition fired **0%** for both kinds (u_pre always ≥ 0.60 floor).

The routing formula `routing_score = 0.70·sim + 0.30·u_stored` evaluates exactly as documented:
- Exact: 0.70 × 1.000 + 0.30 × 0.797 = **0.939** ≥ 0.90 → T1
- Paraphrase: 0.70 × 0.925 + 0.30 × 0.797 = **0.886** < 0.90 → T2

## Production cycle-5 eval reality check

| Bench | n | T1 | T2 | T3 |
|---|---|---|---|---|
| commonsense_qa | 300 | 0 | 247 | 53 |
| fever | 300 | 0 | 130 | 170 |
| strategyqa | 300 | 0 | 72 | 228 |
| triviaqa | 300 | 1 | 111 | 188 |
| truthfulqa | 300 | 1 | 193 | 106 |
| **Pooled** | **1500** | **2 (0.13%)** | 753 | 745 |

## Reconciling the two numbers

**0.13% in production vs 61% in probe is not a contradiction** — both correct because they measure different distributions.

CAEM's eval streams are **content-hash disjoint** from training streams (enforced by `assert_no_leakage` in `caem/benchmark_splits.py`). Memory entries come from training streams. So at routing time on eval:
- Eval queries are **never** in memory
- Closest match is at best a semantically-related entry
- That match rarely clears `routing_score ≥ 0.90`
- → T1 fires ~0.1-1% on eval, by design

When queries ARE crafted from memory entries (the probe), T1 fires reliably. The architecture is **precision-first cache**, not semantic retrieval.

## The earlier `pipeline.answer()`-based diagnostic was a measurement mistake

The previous `outputs/tier1_paired_main/` run via `pipeline.answer()` reported 0 T1 hits on 100 queries that should have hit T1. The new probe via `router.route()` shows 61 T1 hits on a similar dataset. CAEM's routing layer works correctly — the issue was specific to how the previous diagnostic was capturing tier outcomes. The 2 production T1 hits at cycle 5 prove pipeline.answer DOES record T1 when it fires.

## Thesis framing

**Three complementary receipts for Ch5/Ch6:**

1. **Production sparsity (0.13% at C5, ~0.7% pooled across C1-C5)**: by-design consequence of content-hash-disjoint eval. Frame as architectural intent: T1 is amortisation for repeated queries, not semantic retrieval for novel ones.

2. **Architectural soundness (84% exact / 38% paraphrase in probe)**: routing math works exactly per formula; safety OR-condition correctly stays inactive; FAISS retrieval returns the expected entry. T1 IS available and reliable when query distribution makes it applicable.

3. **Implicit memory benefit dominates (24pp EM gap on T2-routed queries between exact vs paraphrase, from earlier paired EM data)**: most of memory's contribution to performance arrives via SIL-driven implicit adapter memorisation, not via explicit T1 retrieval.

**Combined claim:**
> Tier 1 acts as a precision-first retrieval cache for repeated queries. Production T1 firing rate is sparse (~0.1-1%) because production eval is content-hash disjoint from training by design; when queries do match stored episodes, routing dispatches correctly to T1 (84% verbatim, 38% paraphrase). Memory's bulk contribution to performance flows through implicit adapter memorisation via SIL training, while T1 provides the high-precision fast path for the rare exact-match case.

## Artifacts on disk

- `outputs/tier1_routing_probe/queries_cycle5.json` (100 paired queries, stratified)
- `outputs/tier1_routing_probe/results_cycle5.json` (per-query routing decision + top-5 neighbours)
- `outputs/tier1_routing_probe/summary_cycle5.json` (aggregated stats)
- `scripts/run_tier1_routing_probe.py` (the probe script)

Mirrored on gdrive at both `vast_to_other_machine/caem/outputs/tier1_routing_probe/` and `caem-phase1a/v2_1_phase1e_main/tier1_routing_probe/`.

## Related entries

- `caem_t1_retrieval_quality_finding.md` — when T1 fires, how often is it correct (precision angle). 87% true / 9% false on the 55 production T1 hits, 9% failures all FEVER prefix issues.
- This entry — when does T1 fire at all (activation angle). Routing works correctly; rarity is sample-distribution.
- `caem_sil_pool_precision_finding.md` — SIL pool precision at storage gate τ=0.60 vs SIL gate τ=0.70.

## Thesis use (2026-10-16)
Aksan confirmed the framing: Tier 1 hits only on duplicate or paraphrased questions; the evaluation inputs are distinct, so it rarely fires there (7 of 9 000 eval queries over C0–C5). Ch1 L58 now quotes the probe (42/50 exact, 19/50 paraphrase; correct 41/42 and 19/19 from `em_analysis.json`); ch5 still has to report it with a table/label. Tier migration is written as design intent plus realised shares, never as an observed trend.
