---
name: Verify exact sample count and dedup discipline before launching ANY GPU run
description: Hard rule — before launching any script that processes calibration/eval/training samples on GPU, prove the count and dedup behavior matches the in-line/canonical path. Burning Vast credit on extra samples is unacceptable.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
On 2026-04-26, I launched the cycle-0 re-calibration via `scripts/run_calibration.py`
(standalone path) instead of the canonical in-line path. The standalone
loaded **1915 samples** because TriviaQA's HuggingFace dump has duplicate IDs,
and the standalone path lacks the dedup defenses that `run_experiment.py:1108-1199`
applies in the in-line path. The user expected 1500 samples (500 × 3 training
benchmarks) and the standalone produced 415 extra non-unique TriviaQA rows.

This wasted ~75 minutes of GPU on duplicate questions and made cycle-0 calibration
inconsistent with future per-cycle calibrations (which route through the in-line
dedup path). The user was rightly upset:

> "dont ever do this type of mistakes, please, it took already 5hours per run
> only for step 7.0! so many things left to run. it burns my vast ai budget.
> how can you just run more samples (and they are not even unique). be careful
> from next time, and dont forrget this instruciton."

## Rule

Before launching ANY GPU run that processes a sample fold (calibration, eval,
seed, purity, training):

1. **Compute the expected sample count** from `dataset_splits.json` /
   `caem/benchmark_splits.py` constants (e.g., DEFAULT_CALIBRATION_SIZE = 500
   per training benchmark × N training benchmarks).
2. **Read the launch script's loader code** to confirm it produces exactly
   that count after dedup. Specifically check:
   - Does it apply content-hash dedup before filtering by IDs? (TriviaQA
     and FEVER have duplicate native IDs that need dedup.)
   - Does it cross-check against eval-fold IDs to enforce disjointness?
   - Does it match the canonical path used by `run_phase1a.sh` step_7_*?
3. **If the standalone path lacks defenses**, do NOT launch the standalone.
   Either fix the standalone first OR launch via the canonical in-line path
   (`run_experiment.py` with appropriate flags / skip-if-exists guards) so
   the in-line dedup applies.
4. **Predict total wall-time** = expected_n × known per-sample latency.
   If predicted time on Vast is > 30 min, surface the cost estimate to the
   user before launching.

## Why

The user is paying real money per GPU-hour on Vast.ai. Running 415 extra
samples = ~75 wasted minutes = ~$0.50 burned, plus statistical noise
introduced into the calibration that needed post-hoc dedup to clean up.
Repeating this pattern across the remaining Phase 1a steps could waste
many hours.

## How to apply

When recommending a launch command:
- Cite the expected sample count and the source-of-truth file (dataset_splits.json
  line numbers or DEFAULT_*_SIZE constants).
- Cite the dedup defense in code (file:line) and confirm the chosen launcher
  applies it.
- If the launcher is a NEW path or one I haven't audited, audit it first.
- Default to the canonical run_phase1a.sh step_* function calls rather than
  improvising standalone scripts. The runbook's authors already vetted those.

## Specific known dedup pitfalls

- **TriviaQA HF dump**: duplicate `qb_/sfq_/qw_/qz_` IDs within a single split
  → must content-hash dedup before filtering.
- **FEVER lucadiliello dump**: integer claim IDs reused between train and dev
  splits → must cross-check fold disjointness by content-hash, not native ID.
- **Generic risk**: any HuggingFace dataset where `sample.get("id")` is not
  guaranteed unique within a split.

## Specific known sample-count pitfalls (loader-function bugs)

- **`scripts/seed_cold_start.py::load_train_samples(benchmark, n, seed)`** —
  the `n` parameter is documented but IGNORED for fever/triviaqa/natural_questions.
  Function calls `build_benchmark_pools(...)` with module defaults and returns
  `pools.seed` whose size is `DEFAULT_SEED_SIZE = 1000`, NOT n. Caught
  2026-04-26 17:30 UTC after a 53-min misconfigured `diff_verify_serial_vs_batch.py`
  run. Either patch the loader to slice to `n` post-pool-build, or read its
  source before relying on the n parameter.
- **Generic rule**: if an upstream loader function has a count parameter that
  diverges from the default pool size, READ the function source and confirm
  the count is honored before launching ANY GPU run that consumes its output.

## Standalone vs canonical entry points

| Script | Has dedup defenses? | Use when |
|---|---|---|
| `run_phase1a.sh::step_7_0_cycle0` (calls run_experiment.py) | ✅ yes (lines 1108-1199) | normal cycle-0 |
| `scripts/run_calibration.py` (standalone) | ❌ no | NEVER for production calibration; legacy/dev only |
| `scripts/run_experiment.py` direct | ✅ yes | resume/replay scenarios |

If a clean restart of step_7_0_calibrate is needed, prefer re-running the
in-line path with --skip-if-eval-exists guards rather than firing standalones.
