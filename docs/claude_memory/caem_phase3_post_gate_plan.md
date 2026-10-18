---
name: CAEM Phase 3 — post-7.0.3 plan with BatchPipeline integration window (2026-04-26)
description: Full sequential plan from Phase 3 validation gate through Ch5/Ch6 artefacts, including a NEW BatchPipeline-wiring window between 7.0.3 and step_5_5_pairs to save ~50-150 GPU-h on step_7_main.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
Phase 3 launched 2026-04-25 12:29 UTC. Cycle-0 eval complete; in-line legacy
temperature calibration in flight (1500 samples, sequential, ~5 GPU-h). When
this lands, the runbook proceeds through 7.0.1 → 7.0.2 → 7.0.3 (validation
gate). After the gate fires we INSERT a BatchPipeline-wiring window before
resuming step_5_5_pairs.

## Why insert the BatchPipeline window

Audit on 2026-04-26 found `scripts/run_calibration.py::collect_calibration_data`
(lines 343-365) is sequential and ignores BatchPipeline. Run_phase1a.sh
2026-04-23 audit (`branch_C_log.md:2803-2871`) had flagged this as missed
wiring; the eval-side wiring shipped, the calibration-side did not.

Numerical equivalence is documented at `caem/pipeline_batch.py:18-24` —
batched path matches serial at ≤1e-3 on u_stored, ≤1e-2 on signals (well
below isotonic discretization granularity). NO contamination concern.

`run_purity_validation.py` has a separate FIX-6 invariant (must NOT call
`pipeline.answer` because Stage 7 mutates memory mid-α-measurement). The
patch preserves this — batches the underlying generate + verifier calls
directly, never goes through `BatchPipeline.answer_batch`.

## Worktree with patches staged (ready to apply)

`/workspace/caem/.claude/worktrees/agent-aad2027e221164c24/` (branch
`worktree-agent-aad2027e221164c24`). 4 modified files + 1 new test:

- `scripts/run_calibration.py` (+206/-54) — `collect_calibration_data`
  takes `batch_size: int = 32`; chunks via `BatchPipeline.answer_batch`
  with `BatchSample(query=..., store_to_memory=False, source_benchmark=bm)`;
  per-sample bookkeeping factored into `_ingest_one` helper; two-tier
  fallback (whole-batch → per-sample serial; per-sample ingestion error
  → skip just that sample). `batch_size` threaded through `calibrate_pipeline`
  + `calibrate_pipeline_temperature_only`.

- `scripts/run_purity_validation.py` (+326/-51) — new
  `_generate_greedy_answer_batch` helper batches `tokenizer + model.generate`
  directly (NOT `pipeline.answer`); verifier batched via existing
  `pipeline.verifier.verify_batch`; re-tokenizes single rows unpadded for
  verifier internal-cal path. Public `compute_alpha` alias added.

- `scripts/run_ablation.py` / `scripts/run_cyclic_ablation.py` — docstring
  notes only (no `eval_batch_size` argparse currently consumed; documented
  expectation that 32 should be passed by future invocations).

- `tests/test_calibration_batch_equivalence.py` (+304) — 3 new tests
  including parametrised batch_size = 4 and 32, plus clamp-on-zero edge.

Test green: 18 existing + 3 new + 425 full suite all pass; CPU-only purity
smoke runs end-to-end.

## Full sequential plan

### Phase 3 validation gate (in flight)
1. (#99) Wait for in-flight Cycle-0 calibration (~5 GPU-h sequential)
2. (#100) Step 7.0.1 — `fit_composite_calibration.py` → `composite_calibration.json`
3. (#101) Step 7.0.2 — `fit_conformal_gate.py` → `conformal_gate.json`
   (τ_store @ α=0.20 / 80% precision; τ_defer @ α=0.40 / 60%)
4. (#102) **🚨 Step 7.0.3 — `validate_composite_weights.py` DECISION GATE**

### BatchPipeline integration window (NEW — between 7.0.3 and step_5_5_pairs)
5. (#103) Halt runner cleanly after 7.0.3, before step_5_5_pairs
6. (#104) Apply worktree patches to main repo
7. (#105) **🚨 Real-GPU equivalence smoke** — 50 cycle-0 samples re-run
   through batched path on GPU; assert atol=1e-3 on u_stored vs unbatched
   JSONL. Catches bf16 batch-dim drift before committing step_7_main.
8. (#106) Resume runbook from `step_5_5_pairs`

### Pre-Step-7 calibration + gates
9. step_5_5_pairs / step_5_5_headhead / step_19_2_slice
10. (#107) `step_platt_calibrate` — Frozen Qwen judge Platt scaling
    (activates `AdaptiveNLIJudge` for ASQA-class long hypotheses)
11. (#108) `step_hf_upload_pre_main` — HF snapshot for credit-burnout recovery
12. (#109) **🚨 `step_u_tok_drop_gate`** — \|Δ u_stored\| < 0.02

### Headline experiment
13. (#110) **`step_7_main`** — 10-cycle SIL. ~7-8 days under batched cal
    (vs ~14 days unbatched). Headline result.
14. (#111) **🚨 Halt after step_7_main** — rewire B6/B7/ablation dynamic
    cycle count per `caem_stop_after_step7_main.md`.

### Post-main
15. (#112) External baselines + sig-tests (B1-B7 + McNemar + bootstrap CI + Holm)
16. (#113) Diagnostics — step_19_purity / 19_2_eval / 19_5_corr / 20_aggregate
17. (#114) Ch5/Ch6 artefacts — step_21_tables / 22_figures / 23_calib_traj
    / 24_session5_artifacts / 25_t1_robustness
18. (#88) Phase 4 — empirical-evidence artifacts for the report
    (can fire in parallel immediately after #102 lands; doesn't depend on step_7_main)

## Decision points

- #102 (7.0.3 gate): pass/fail/halt-and-iterate
- #105 (real-GPU smoke): pass = green-light step_7_main; fail = debug bf16 drift
- #110 (step_7_main launch): final go/no-go on long burn
- #111 (post-main halt): confirm B6/B7 rewire before resuming baselines

## How to apply / Why

After any compaction, READ this file FIRST to recover the post-gate plan.
The worktree path and patch list are the load-bearing details — don't redo
the audit from scratch. The validation gate at #102 may halt or pass; the
BatchPipeline window at #103-#108 runs regardless of gate outcome (the
patches are independent of composite design decisions).
