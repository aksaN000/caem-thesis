---
name: Ch4 theorems are pool-level guarantees, not model-level guarantees — scope limit to acknowledge
description: T1-T7 cover memory pool purity and convergence; none predict held-out model EM. Phase 1e shows pool purity slowly decaying while model EM regresses at C3 — not a contradiction, a scope limit.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Finding logged 2026-05-24.**

**The gap:** Every theorem in Ch4 §11 (T1 purity bound, T2 Bayes identity, T3-T4 monotonicity, T5 convergence envelope, T6 fixed-point convergence, T7 ε_arch) is a statement about the **stored memory pool**, not about the trained model's held-out generation EM. Phase 1e shows:
- Pool purity drifting slowly: C0 1.000 → C1 0.787 → C2 0.762 → C3 TBD (T5 may still hold if this is converging to π_∞ ≈ 0.70-0.75)
- Model held-out EM regressing C2→C3: FEVER −3pp, TQA −3pp
- **No theorem violation** — these two trajectories are independent
- But also no theorem covers "training on contaminated SIL pool degrades model EM"

**The mechanism that's outside theorem scope:**
- SIL pool C2: 2966 episodes, 16.6% em=0 (~491 wrong)
- The 491 wrong examples have HIGHER gradient magnitudes than the 80% correct (large loss → large adapter movement)
- The 491 are systematically biased (FEVER polarity flips, hallucinated entity overlap), forming a coherent wrong pattern the adapter learns
- LoRA r=32 accumulates these wrong directions across cycles
- Held-out eval feels the drift more than the SIL pool (distribution shift)

**Why this matters for the paper:**

This scope limit is **publishable as a finding**, not a weakness. Pre-register the gap honestly:

1. **Ch5 §threats addition:** "T1-T7 are pool-level guarantees. They establish that the stored memory pool maintains a precision floor and converges in distribution. They do NOT bound the held-out generation EM of the LoRA adapter trained on this pool. The observed EM regression at C3 is consistent with — though not predicted by — the theorems, because contamination acts on the *training* side via per-example gradient magnitude, while T1-T7 characterize the *storage* side via fractional purity."

2. **Ch6 §future-work pre-registration:** Propose ε_train as a complementary quantity. Define as the fraction of em=0 training examples whose per-example loss exceeds the per-batch mean. Observe that ε_train doubled when β doubled C1→C2. This characterizes the gradient-impact asymmetry that fractional purity misses.

3. **Methodological framing:** "We empirically demonstrate that pool-level purity guarantees do not transfer to model-level performance guarantees. The mechanism is asymmetric gradient impact from contaminated training data combined with LoRA accumulation of cycle-specific direction vectors. We propose this as a research opening rather than a methodology failure."

**How to apply:**

- Don't claim the theorems predict EM monotonicity — they don't.
- Don't claim the EM regression is a theorem violation — it isn't.
- Frame the gap as "the theorems characterise what they characterise; we identify what they do not, and propose the complementary quantity needed."
- Cite this scope limit in §main-results, §threats, AND §future-work — three touchpoints make the story coherent rather than a hedge.

**Watch for:**
- If C4-C5 pool purity continues dropping monotonically (no asymptote), T5 (geometric envelope) starts to fail empirically. The fit quality of G_t = A·r^t becomes the empirical test.
- If C3-C5 model EM keeps dropping in lock-step with β rising, the case for ε_train as a contamination proxy becomes stronger.
- If model EM stabilizes at some level while pool purity keeps dropping, that's evidence that LoRA L2 anchor + retention guard are bounding model drift independent of pool quality.

**Related entries:**
- `caem_sil_pool_precision_finding.md` — two operating points (storage τ=0.60 vs SIL τ=0.70)
- `caem_t1_retrieval_quality_finding.md` — 9% T1 failure rate driven by FEVER prefix
- T3/T4 empirical failures already documented in conversation transcripts
