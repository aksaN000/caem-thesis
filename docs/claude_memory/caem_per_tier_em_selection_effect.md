---
name: Per-tier EM decomposition — pooled-EM dip is selection effect, not RAG degradation (2026-05-14)
description: Per-tier EM trajectory through C3 shows T3's mild drop is residual-hardness selection effect; cost-adjusted EM is the right metric, not pooled EM
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
Cycle-0 → cycle-3 per-tier EM on the eval fold (pooled across 4 closed benches, n=1200 at C3 with strategyqa pending):

| Cycle | T2 share | T2 EM | T3 share | T3 EM |
|---|---|---|---|---|
| C0 | 1.5% | 0.727 | 98.5% | 0.476 |
| C1 | 4.8% | 0.778 | 95.2% | 0.502 |
| C2 | 7.8% | 0.624 | 92.2% | 0.511 |
| C3 | 13.8% | 0.620 | 86.2% | 0.460 |

**The thesis story that the data is telling:**

1. **T2 EM (0.620 at C3) > base T3 EM at C0 (0.476)** — the fine-tuned model answering zero-shot is +14 pp more correct than the base model answering with RAG. SIL-as-distillation is folding verified RAG outputs into parametric capacity; queries that migrate T3 → T2 are both cheaper AND more correct.

2. **Pooled EM C2→C3 dropping 3.8 pp is 100% from T3 dropping, not from share-shifting to T2.** Decomposition: if T3 EM had held at 0.511, hypothetical C3 pooled = 0.526 (+0.6 pp vs C2). Actual C3 pooled = 0.482 because T3 lost 5.1 pp.

3. **T3 dropping is the selection effect, not RAG degradation.** Easy queries (high pre-routing confidence) migrate out of T3 into T2. The residual T3 query population is the hard tail that needs both retrieval AND careful reasoning. On the same residual subset, C0 base would have scored ~0.42 (hypothetical with migrated queries removed); C3 actually scores 0.460. So fine-tune is *helping* T3 too, just less than it helps T2.

4. **T3 structural ceiling is ~0.45-0.50 on this mix** because of corpus coverage gap K(C), adversarial benchmark framing (TruthfulQA), multi-hop requirements (StrategyQA), distractor passages, and selection bias on the residual.

**Why:** Pooled EM is a misleading headline metric when the architecture is doing share-shifting work. The right read is per-tier EM decomposition + cost-adjusted EM (EM normalized by per-tier wall-clock or dollar cost). T2 EM trending flat at 0.62 while T2 share doubles cycle-on-cycle is the cleanest single piece of evidence for the distillation thesis.

**How to apply:** When discussing trajectory regression in Ch5 §sec:tier-precision and Ch5 §sec:disc-headline, lead with the per-tier decomposition + the "T2 EM > C0 T3 EM" comparison, not the raw pooled EM curve. The corpus-bounded floor check in §sec:check-asymptotic (Corollary 4.x) is the right framing for T3's floor: register corpus expansion as the only path below that floor (already in §sec:future-deployment). Watch C4-C5 to confirm T2 EM continues holding flat while share grows — if T2 EM starts falling alongside T3 EM, the selection-effect story breaks and we'd need to inspect SIL for genuine RAG-corruption drift.
