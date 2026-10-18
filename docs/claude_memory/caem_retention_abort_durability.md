---
name: Retention abort durability — asymmetric rollback + parametric ceiling
description: When the retention guard fires (any probe < 0.93), only the LoRA-weight channel pauses; memory growth, calibration refit, retroverify, deferred-buffer, T1 hit-rate all continue. The architecture is designed to handle chronic LoRA stagnation via non-parametric channels (memory + RAG). Most retention failures are fixable engineering issues; only the true parametric ceiling requires base-model change. Save for Ch5 §threats and Ch6 §durability / §future-work.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
## The asymmetric rollback property

Per-cycle retention guard: if any probe (MMLU + TriviaQA-test + HotpotQA-test or production equivalents) drops below 0.93 of cycle-0 baseline, the LoRA reverts to θ_prev. **Memory survives the rollback; the cycle's fine-tune compute is the only thing lost.**

| Channel | On retention abort |
|---|---|
| LoRA weights | rolled back to previous cycle |
| Memory store growth | continues (STORE decisions add live) |
| Composite calibration refit | continues |
| T scalar refit | continues |
| τ_store / τ_defer EMA | continues |
| Retroactive re-verification | continues |
| Deferred-buffer promotion | continues |
| Tier 1 hit rate | keeps growing |
| RUC routing | unchanged (frozen) |

**Phase 1d empirical precedent (C4 abort, 2026-05-14):** TriviaQA-test 0.395 → 0.345 fired the guard. LoRA reverted to C3 weights, memory + calibration kept advancing, retroverify pruned stale entries under the new composite. Trajectory continued to C5 close per the runbook.

## Five failure modes that can trigger retention abort

| Cause | Fixable without base-model swap? | Mechanism |
|---|---|---|
| **LoRA SV-collapse** (rank exhaustion) | Yes — engineering | Reset LoRA, restart from research-final; bump rank r=32 → 64; multi-LoRA per domain |
| **Verifier noise** (composite false-positives at storage gate) | Yes — engineering | Tighten τ_store; add signals; better calibration; human-label boundary samples |
| **Catastrophic forgetting from SIL pool dominance** | Yes — engineering | `pool_max_share` cap (v2 Fix 3, already in code); loss reweighting; replay buffer |
| **Distribution drift in live traffic** | Yes — engineering | Multi-LoRA / mixture-of-LoRAs / hot-swappable LoRAs per domain |
| **Base model parametric ceiling** | **NO** — requires base-model change | Qwen-2.5-3B knowledge capacity is fixed by pretraining; LoRA cannot expand it |

## Where the actual ceiling lives

The base model (Qwen-2.5-3B) has finite knowledge capacity — bounded by what fit in its 3B parameters during pretraining. LoRA can re-weight existing representations but cannot create new knowledge from nothing. At the true ceiling: adding more SIL data causes interference between learned facts, regardless of pool quality.

## Why CAEM's architecture is designed to push past this ceiling

CAEM has **three learning channels**, two of which are non-parametric and unbounded:

| Channel | Bounded by | Bypasses parametric ceiling? |
|---|---|---|
| LoRA weights | base model capacity | NO — hits the ceiling |
| Episodic memory store | disk + retrieval cost | YES — unbounded growth |
| RAG corpus (Wikipedia 21M passages, etc.) | external retrieval index | YES — fixed but huge |

When the LoRA channel saturates, CAEM gracefully degrades to "frozen model + growing memory + RAG." That's not a failure — that's the architecture's designed operating point at the parametric ceiling. T1 and T3 carry knowledge growth that LoRA can no longer carry.

## Honest answer for the thesis

The question "is chronic retention abort a fundamental limit?" has a layered answer:

1. **Most retention aborts are engineering-fixable**: SV-collapse, calibration noise, pool dominance, and distribution drift all have concrete mitigations that don't require changing the base model.
2. **The true parametric ceiling exists but is masked by CAEM's non-parametric channels**: memory + RAG continue to learn even after LoRA saturates. So the system as a whole keeps improving on user-facing metrics (Tier 1 hit rate, retrieval recall) even when the LoRA-channel stops.
3. **Going beyond the system-level ceiling requires base-model change**: bigger backbone (Qwen-2.5-7B / 14B / 72B), MoE, or a fundamentally different architecture (retrieval-pretrained LM, state-space model).

## What to write in the report

**Ch5 §threats:**
> "CAEM's primary learning channel (LoRA fine-tune at cycle close) is protected by a multi-probe retention guard with a 0.93 forgetting floor. When the guard fires, the asymmetric rollback property preserves memory growth, calibration refit, and retroactive re-verification while reverting LoRA weights. Phase 1d's Cycle 4 abort demonstrates this graceful behavior: the trajectory continued to C5 with memory and calibration advancing, while LoRA capacity was bounded at C3."

**Ch6 §durability:**
> "Chronic retention aborts (5+ consecutive cycles) are diagnosable via `coverage_diagnostic.json` triggers. Of the five failure modes that can cause them, four are engineering-fixable (SV-collapse → LoRA reset, verifier noise → composite retightening, pool dominance → `pool_max_share` cap, distribution drift → multi-LoRA). Only the fifth — base-model parametric ceiling — requires a backbone upgrade. CAEM's non-parametric channels (memory store, RAG corpus) continue to absorb knowledge growth past the parametric ceiling, so the system-level learning capacity exceeds the parametric capacity by design."

**Ch6 §future-work:**
> "Phase 2 enhancements: (a) multi-LoRA per domain to reduce inter-domain interference; (b) periodic LoRA reset with restart-from-research-final when SV-collapse fires; (c) adaptive LoRA rank (r=32 → 64 on stagnation); (d) backbone upgrade to Qwen-2.5-7B for deployments where the 3B parametric ceiling becomes binding."

## Pre-registered durability claim for Phase 1e

If Phase 1e completes 10 cycles without retention abort, that's empirical evidence that:
- The v2 pool_max_share cap (Fix 3) is sufficient to prevent pool-dominance forgetting at this scale.
- The RUC's bench-conditional routing eliminates the specific failure mode that triggered Phase 1d's C4 abort (TriviaQA pool dilution by FEVER-dominated SIL).

If Phase 1e DOES abort at some cycle X, the diagnosis flow is:
1. Check `coverage_diagnostic.json` at cycle X for `relax_alpha` or `halt` triggers.
2. Check per-bench storage rates; if FEVER dominates, the pool_max_share cap isn't enough.
3. Check LoRA SVD norms; if collapsing, recommend LoRA reset.
4. If none of the above, register as base-model ceiling evidence for Ch6 §future-work backbone upgrade claim.
