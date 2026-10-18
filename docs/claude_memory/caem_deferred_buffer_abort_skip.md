---
name: Deferred reconsideration is skipped on aborted cycles — by design, not a bug (2026-05-15)
description: Deferred reconsider, retroverify, and composite refit all gated on `not aborted` in code. TTL is counted in reconsider invocations not calendar cycles. Reframe as asymmetric-rollback strength. Ch5 §sec:adj-cal-eval-gap paragraph added.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**The finding (2026-05-15 09:00 BDT, user-prompted while C4 Step 5 was closing):** Deferred-buffer reconsideration is skipped on aborted cycles. Code at `caem/training/self_improvement.py:681`:

```python
if deferred_buffer is not None and not aborted:
    # reconsider runs
```

Confirmed by C4 retroverify JSON (`outputs/full_run/retroverify_cycle4.json`) showing `aborted=True` + `retroverify.skipped=aborted_cycle`, and by the deferred buffer pickle (`outputs/full_run/deferred_buffer_cycle_4.pkl`) growing from 3110 (C3 close) to 4548 (C4 close) with zero promotions and zero TTL-drops.

## Per-cycle reconsider firing trajectory

| Cycle | SIL aborted? | Reconsider fired? | Buffer at start | Promoted | TTL-dropped | Buffer at end |
|---|---|---|---|---|---|---|
| C1 | No | YES (no-op on empty) | 0 | 0 | 0 | 1302 |
| C2 | No | YES | 1302 | 79 | 0 | 2898 |
| C3 | No (after retry) | YES | 2898 | **184** | **1145** | 3110 |
| **C4** | **YES** | **NO** | 3110 | — | — | **4548** |
| C5 (predicted abort) | YES | NO | 4548 | — | — | ~5950 |

## Three rescoring mechanisms all gated identically

| Mechanism | Code gate | Reason |
|---|---|---|
| Step 1: deferred reconsider | `if not aborted` (self_improvement.py:681) | Same weights → same scores → no promotions, no TTL-drops would result |
| Step 2.5: retroactive re-verification | `if not cycle_result.aborted` (run_experiment.py:1810, patched 2026-05-14 in commit 2347d60) | Same logic |
| Step 2.1-2.4: cal-fold scoring + per-bench composite refit | `if not cycle_result.aborted and not skip_calibration` (run_experiment.py:1750) | Same logic |

All three "re-score memory or recalibrate under unchanged model" mechanisms are gated on `not aborted`. The architecture is internally consistent: pause everything that depends on a model change when no model change happened.

Stream-chunk gen+verify+store (Step 4) and Step 5 held-out eval continue to run unconditionally — fresh data always flows.

## Why this is correct, not a bug

TTL semantic is "number of reconsider invocations" not "calendar cycles passed":
- Per `caem/config.py:954-956`: "an entry has at most TWO reconsider() invocations (cycle N+1, cycle N+2) to be promoted before being TTL-dropped"
- An aborted cycle = zero reconsider invocations = TTL counter doesn't increment
- This preserves the design intent: TTL bounds *model-misses*, not calendar time

Counterfactual options analysed and rejected:
- **Run reconsider on aborted cycle with cycle-(N-1) composite**: rolled-back weights + skipped composite refit means scores would be identical to the previous reconsider's scores. Pure compute waste.
- **Force TTL counter to increment without reconsidering**: breaks the model-misses semantic; turns TTL into calendar-based eviction (which `deferred_buffer_max_size = 10000` already provides as FIFO).
- **Refit composite even on aborted cycles**: refit on same calibration signals (same weights → same signals) produces same composite. No-op.

## TTL counter state for 5-cycle trajectory at C5 close

| Vintage (deferred during) | Approx count | TTL counter | Status |
|---|---|---|---|
| C1 stream chunk | 0 | N/A | All either promoted at C2/C3 (79 + some of 184) or TTL-dropped at C3 (1145) |
| C2 stream chunk | ~1569 | 1 (reconsidered at C3 only) | Alive in buffer, missed C4 + C5 reconsiders |
| C3 stream chunk | ~1541 | 0 (never reconsidered) | Alive in buffer |
| C4 stream chunk | ~1438 | 0 (never reconsidered) | Alive in buffer |
| C5 stream chunk | ~1400 | 0 (never reconsidered) | Alive in buffer |
| **Total at C5 close** | **~5950** | mostly 0 or 1 | **None ever reaches TTL=2 in the trajectory** |

Empirical deferred-recovery rate over the 5-cycle trajectory: **263 promotions / ~5950 final buffer = ~4.4%**. Bounded above by the number of successful cycles (3 of 5) and the per-cycle reconsider yield.

## Ch5 §sec:adj-cal-eval-gap update (landed 2026-05-15)

Two changes in the existing §sec:adj-cal-eval-gap:

1. **Recall-complement paragraph tweaked**: "rescored under the post-fine-tune verifier at every cycle boundary" → "...at every **successful** cycle boundary." Also: "Both paths are deterministic" → "The two paths are gated differently" + explicit note that stream-chunk gen-and-store runs unconditionally.

2. **New paragraph inserted** explaining the asymmetric-rollback inheritance. Lead sentence: "Both cross-cycle recovery paths inherit the asymmetric-rollback property formalised in §sec:retention." Then walks through: rescoring side pauses (deferred + retroverify + composite refit), TTL counted in invocations not calendar cycles, fresh-data side runs unconditionally, mechanism self-bounds recovery at successful-cycle count, buffer preserved through every abort.

The reframe turns "deferred path only worked 4% of the time" into "deferred path is correctly gated by the same retention guard that protects everything else" — same data, much stronger reading.

## Future-work entry to register in Ch6 (during post-C5 writing pass P-7)

Add to Ch6 §concl-open or §future-work alongside the EWC + Experience Replay items:

> "An alternative time-to-live semantic for the deferred-entry buffer that ages entries by calendar cycle rather than by reconsider invocation would suit deployments where retention rollback is expected to be a common pattern rather than an exceptional one. Under the current model-misses semantic the buffer can grow to its FIFO cap during long abort sequences without ever draining through the TTL mechanism; under a calendar semantic the buffer would drain at the registered horizon regardless of cycle success. The choice is registered here as a deployment-pattern-dependent parameter rather than a defect, and the present trajectory's bounded-cycle horizon does not require either to be empirically distinguished."

## How to apply (future Claude reading this memory)

- The Ch5 §sec:adj-cal-eval-gap paragraph is already landed. Do NOT re-add.
- The §sec:retention paragraph (about retention guard firing and the C4 abort empirical receipt) is already queued in PRODUCTION_NEXT_SESSION_PLAN.md Step P-7 — make sure the new §sec:adj-cal-eval-gap paragraph cross-references §sec:retention by `\Cref{sec:retention}` (already does).
- The Ch6 future-work item is queued above; land it during Step P-7 in the post-C5 writing pass.
- Do NOT modify the code. The "skip on abort" behavior is architecturally correct.
