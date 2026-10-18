---
name: New-prompt pipeline raises u_stored; cold-start uses τ=0.50 (updated 2026-04-22)
description: Forced decoder prefix + few-shot scaffolded CoT shifts u_stored upward; cold-start seed uses explicit --cold_start_store_threshold 0.50 override per user decision 2026-04-22 (was 0.45 on 2026-04-20).
type: project
originSessionId: 2f06df76-4467-4646-9e32-0a9234c27ac4
---
On 2026-04-20 Step 6 re-seed under new prompts (forced decoder prefix +
few-shot scaffolded CoT) produced STORE events at default
CAEMConfig.store_threshold=0.65 (u_stored 0.666–0.755, mean ~0.70).
Under OLD prompts the same threshold yielded zero STOREs — which is
why scripts/seed_cold_start.py docstring recommends 0.45 (and why the
earlier Step 6 run used --cold_start_store_threshold 0.35).

**Cold-start override decision (2026-04-22, supersedes 2026-04-20):**
user chose `--cold_start_store_threshold 0.50` for Phase 1a Step 6 —
midway between the default 0.65 and the earlier 0.45 decision.

**Why the change from 0.45 → 0.50 (2026-04-22):** at 0.45 the storage
rate in Profile v8 measurement was ~80% (51/64 on FEVER smoke at
Branch C post-atomic-revert); user wants a slightly more selective
cold-start seed so early cycles see less noisy memory content. The
0.50 threshold is expected to drop storage rate to ~65–70%, still well
above the ~50% floor at which 200 target episodes/benchmark requires
more than 300 queries.

**Applied at:** `run_phase1a.sh` step_6_reseed() line ~79 and
`scripts/seed_cold_start.py --cold_start_store_threshold 0.50` when
invoked manually.

**NOT applied to Step 7 main run** — still uses the quantile-fitted
τ_store from `outputs/cycle_0/calibrated_thresholds.json`.

How to apply:
- Phase 1a Step 6 (cold-start seed under new prompts): **use
  `--cold_start_store_threshold 0.50`**. Do NOT use 0.45 (superseded),
  0.35 (too permissive under new prompts), or default 0.65 (risks
  episodes_stored < 180 target when storage rate drops).
- Chapter 5 §5.5 calibration narrative: the quantile-based threshold
  fit at P70/P40/P90 remains the authoritative source for Step 7+.
  The 0.50 cold-start override is documented as a seeding-phase
  hyperparameter, not the Step 7 operating point.
- Sanity check future runs: if u_stored mean on STORE samples drifts
  far from ~0.70 in a new-prompt run, suspect prompt regression
  (missing forced prefix, truncated scaffold) before tweaking τ.
