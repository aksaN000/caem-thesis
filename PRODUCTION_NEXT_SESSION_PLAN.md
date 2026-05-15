# CAEM Production Plan — v2.1 (Architectural Redesign + Phase 1c/1d patch sequence)

**v2.1 update:** 2026-05-10 (Phase 1a cycle 1 in flight under v2.1+).
**v2 created:** 2026-05-06.
**v1 superseded:** v1 (2026-05-04) was a continuation plan for the broken-architecture trajectory. v2 replaced it with the discovered-failure-mode + architectural-response plan.
**v2.1 supersedes v2 launch-ready state:** the v2 launch-ready architecture (2026-05-07) was hit by a cycle-0 MiniCheck AUROC failure (P0). The Phase 1c sequence (P0', P1, P2, P3a, P3b) restructured the storage gate from per-bench conformal to fixed-threshold-on-calibrated-probability with shrinkage-prior per-bench composite. Phase 1d (Option 4) lowered the gate to 0.60 and rescored cycle 0. Patch 2026-05-10 stopped the canonical composite path being overwritten each cycle close. The architecture decisions block below reflects v2.1 as actually launched.
**Source of authority:** this file (v2.1) + `branch_C_log.md` 2026-05-06 to 2026-05-10 entries + `CAEM_FIX_AUDIT.md` for the Phase-0 architecture lock. v2 conformal-gate sections in this file are explicitly marked RETIRED below; do not re-introduce.

Mark each step `[x]` once complete. Commit + push after every batch.

---

## Step status legend
- `[ ]` = not started
- `[~]` = in progress
- `[x]` = done
- `[!]` = blocked / needs decision

---

## Why v2 (the discovered failure mode)

The v1 Phase 1a trajectory ran cycles 0-4 under the original CAEM architecture (single global conformal gate, single global composite, full-parameter SIL fine-tune, MMLU-only retention probe). Cycle-4 close empirics (n=3500 pooled eval) showed:

- **Pooled CHM dropped −10.5%** (-as predicted by the registered hypothesis H2)
- **Pooled EM dropped −17.7%** (NOT predicted; thesis register said nothing about EM)
- **TriviaQA EM −73.8%** (0.374 → 0.098), **NQ EM −61.6%**, **TruthfulQA EM −25.1%**
- **FEVER stored pool: 100% FEVER by cycle 4** (873 of 874 entries — TriviaQA + NQ admissions went to zero)
- **MMLU retention 1.04× pristine** — the registered retention guard was empirically blind to the open-text crashing because it probes only bounded-label MCQ skill

Root-cause diagnosis (logged in `branch_C_log.md` 2026-05-06): the verifier's task-conditioned signal-to-noise asymmetry creates a positive feedback loop. FEVER passes the gate at higher rate → SIL training pool becomes FEVER-monoculture → model becomes a FEVER classifier → next-cycle SIL pool is even more FEVER-dominant → catastrophic forgetting on TriviaQA/NQ via parametric capacity competition.

Three coupled architectural gaps were identified:
1. **Single-α global gate** can't admit benchmarks where verifier signals don't reach 95% precision
2. **Verifier signals are weak on bare-entity QA** (TriviaQA bare entity vs MiniCheck which expects sentence-shaped claims)
3. **Full-parameter SIL primitive** has no structural defense against monoculture overwrite
4. **Single-probe retention guard (MMLU)** is blind to open-text generation degradation

v2 deploys 13 coupled fixes targeting these gaps. Architecture is locked. Estimated cost: ~$165, ~12-13 days code work + ~17-20 days trajectory wall-clock.

---

## Architecture decisions (locked v2.1)

```
Model:               Qwen 2.5-3B-Instruct, no quantization (7B QLoRA registered as Phase 1c future work)
SIL primitive:       LoRA r=32, α=64, all-linear targets {q,k,v,o,gate,up,down}, LR=2e-4 with cycle decay
                     L2 anchor REMOVED on LoRA path (LoRA's parameter budget IS the implicit anchor)
Storage gate:        FIXED THRESHOLD on calibrated probability (Phase 1c P0', 2026-05-09)
                     store_threshold = 0.60  (Option 4 lock, was 0.65 pre-Phase-1d, task #142)
                     defer_threshold = 0.45
                     RETIRED: per-benchmark conformal split-CP gate (v2 launch-ready spec).
                     Refit-per-cycle was conformal-α; under fixed-threshold this becomes
                     "empirical precision target", not a coverage-guarantee parameter.
Composite:           Per-benchmark isotonic + Cherian boost (boost_C=0.01), 9 signals
                     Signals: u_token, u_dropout, u_internal, s_avg, p_entail,
                              p_ground_max, p_ground_mean, p_ground_atomic, q_a_relevance
                     RETIRED (Phase 1c P1, 2026-05-09): alias_overlap, entity_head_consistency
                     RETIRED (Phase 1d, 2026-05-09): h_norm (≈0 boost weight at cycle 0)
                     Per-bench shrinkage prior toward pooled fit, shrinkage_alpha = 0.6 (P3a)
                     Per-bench pool_max_share cap, bench-agnostic (P3b)
u_pre:               Per-benchmark T_b temperature scaling (Fix 12 retained)
                     Per-benchmark safety_u_pre_min_b (data-driven from cal fold)
SIL pool:            Loss reweighting via temperature mixing T=2 + 3× upsampling cap + DoReMi floor + cold-start gold fallback
                     General-domain mix REMOVED (Fix 13)
                     pool_max_share cap (P3b) prevents single-bench monoculture
Retention probe:     Multi-modal — MMLU 200 + TriviaQA test 200 + HotpotQA test 200
                     Halt-and-rollback if ANY probe drops > 7% from pristine (forgetting_tolerance=0.93)
Coverage diag:       Per-benchmark admission rate + pool composition entropy + per-cycle adapter SVD + per-bench EM trajectory
                     Halt triggers: zero-admission for 2 cycles → operator decision (auto-relax retired with conformal)
Prompts:             Per-benchmark templates with answer-expansion for MCQ; benchmark-conditional verifier dispatch
                     CommonsenseQA 5-choice template, HotpotQA multi-hop template, FEVER NEI directional fix (P2)
                     Verifier-input canonicalisation (`Answer: <X>.` form, idempotent), MCQ letter→option-text expansion
Deferred reconsider: Five-layer hard-fail guard (orchestrator assert + SIL hard-fail + post-cycle log assert + unit test + pre-launch dry-run)
Cycle-boundary calibration (Phase 1c P0' design):
                     Step 2.1 — score cal fold under post-SIL model
                     Step 2.2 — re-fit per-bench u_pre T_b (ECE-min)
                     Step 2.3 — re-fit per-signal isotonic + Cherian boost, shrinkage_alpha=0.6
                     Step 2.4 — reload verifier from cycle_{N}/composite_calibration.json
                     Step 2.5 — retroactive re-verification under recalibrated verifier
                     Patch 2026-05-10: canonical (cycle_0) path NOT overwritten per cycle close.

Training panel (3):       FEVER + TriviaQA + CommonsenseQA
                          (HotpotQA dropped from training panel; remains as transfer-only candidate.
                           See memory caem_hotpotqa_vs_nq_history.md for rationale.)
Transfer eval panel (2):  TruthfulQA + StrategyQA
                          NaturalQuestions held back per the v2 panel-design audit.
Stream chunk:             2000/cycle (FEVER, TriviaQA), 700/cycle (CSQA)
                          Logged at Step 4 start as: sizes={'fever':2000,'triviaqa':2000,'commonsense_qa':700}
Held-out eval per cycle:  300/bench × 5 benches (training panel + transfer panel)
Cycle count:              10 trajectory cycles + cycle 0 calibration = 11 total
```

Three small ambiguities resolved (2026-05-06):
- **Fix 13 (general-domain mix):** full removal (cleaner code, smaller maintenance surface)
- **Fix 6 (alias data):** local 150 MB Wikidata file (avoids 9 min/cycle network latency) — **but the signal itself was retired in Phase 1c P1 (task #136)**; the file is preserved for ablation reproduction
- **LoRA checkpoint retention:** keep all 11 cycle adapters (~580 MB total — trivial)

---

## Phase 0 — Implementation prep (CODE-COMPLETE 2026-05-07)

All 13 architecture fixes have landed on `feat/qwen-3b-goal2`. The
itemised file:line targets that previously occupied this section are
preserved in `CAEM_FIX_AUDIT.md` for archival reference.

### Architecture-fix status

| Fix | Subject | Commit | Smoke tests |
|-----|---------|--------|-------------|
| 9   | Training panel update + new benchmark loaders         | `fc665f9` | 10/10 |
| 13  | Remove general-domain mix from SIL pool               | `82da905` | 6/6   |
| 11  | Deferred-reconsideration five-layer guard             | `8f59d85` | 7/7   |
| 2A  | KEYSTONE — `source_benchmark` threaded end-to-end     | `f6960d6` | 10/10 |
| 2B  | Per-benchmark composite weights + nested JSON schema  | `86382c8` | 7/7   |
| 1   | Per-benchmark conformal storage gate                  | `5002c23` | 8/8   |
| 12  | Per-benchmark u_pre T_b + safety_u_pre_min_b          | `f930d61` | 10/10 |
| 6   | alias_overlap signal (Wikidata)                       | `225fec3` | 11/11 |
| 7   | entity_head_consistency signal (M-chain heads)        | `803a667` | 14/14 |
| 10  | Per-bench prompts + uniform verifier-input canonicalisation | `d470066` | 18/18 |
| 8   | LoRA SIL primitive (CRITICAL — full FT replaced)      | `65f3102` | 13/13 |
| 3   | Loss-reweighted SIL training pool builder             | `d0504b6` | 14/14 |
| 4   | Multi-modal retention probe (MMLU + TQA + HotpotQA)   | `30b46fb` | 14/14 |
| 5   | Coverage feedback diagnostic + halt triggers          | `3be0ece` | 20/20 |

Plus tests-consolidation cleanup (`337229d`) and gdrive bucket layout
(`25ea0cf`). 377 v2-fix smoke + regression tests pass cleanly. Eight
pre-existing `test_self_improvement.py` failures are mock-seed test-
fixture bugs; same count fails on baseline pre-Fix-9 → not introduced
by this work.

### Headline deliverables

* **12-signal verifier composite** — `+ alias_overlap, + entity_head_consistency`
  on top of v1's ten signals. Both auto-derive into eval-harness
  schema via `_derive_verifier_fields(UnifiedVerifierOutput)`.
* **Per-benchmark dispatch wired end-to-end** — gate, composite,
  prompts, T_b, safety floor all consume the same `source_benchmark`
  token; v1 calibration JSONs load via the back-compat path.
* **Uniform verifier-input canonicalisation** — every signal that
  reads the answer (`p_entail`, `p_ground_*`, `atomic`, `q_a_relevance`,
  `alias_overlap`) sees the same `Answer: <X>.` claim form regardless
  of benchmark; MCQ letters expand against the Choices block, bare
  entities wrap, declarative answers pass through.
* **LoRA r=32 α=64 all-linear** SIL primary path — adapter-only
  checkpoints (~120 MB vs 6.2 GB), L2 anchor removed (frozen base
  IS the implicit anchor), graceful fall-through to full FT when peft
  is missing or the model is a mock.
* **Multi-probe forgetting guard** — MMLU + TriviaQA test + HotpotQA
  validation; abort fires if ANY probe drops below
  `forgetting_tolerance` (0.93) from the pristine baseline.
* **Coverage diagnostic + halt triggers** — per-cycle JSON with
  admission rates, pool composition entropy, adapter SV spectra,
  and the SV-collapse score; auto-relax α_b on zero-admission for
  2 consecutive cycles; hard halt on adapter rank-collapse.

### Integration / verification status — COMPLETE (2026-05-07)

All architecture fixes, integration audits, and orchestrator wiring
have landed on `feat/qwen-3b-goal2`. Phase 1a is **launch-ready** as
of commit `6397650`. The Day-13 checklist resolves as:

- [x] **Integration audit** — three parallel audit passes covering
  v2-fix consistency, downstream scripts, and data leakage. Every
  CRITICAL finding fixed. ~25 false positives verified by line-by-line
  re-reading.
- [x] **Downstream-script audit (Fix 9b)** — 15 scripts updated to
  read `caem.config.TRAINING_BENCHMARKS / TRANSFER_BENCHMARKS`:
  `run_calibration.py`, `run_purity_validation.py`, `run_simple_ft.py`,
  `run_experiment.py:1016`, `run_cyclic_ablation.py`,
  `conditional_conformal_ablation.py`, `baseline_sig_tests.py`,
  `build_calibration_pairs.py`, `seed_cold_start.py`,
  `run_ablation.py`, `rescore_baselines_through_verifier.py`,
  `phase4_artifacts.py`, plus four diagnostic-fallback constants in
  `calibration_alpha_curve.py`, `validate_composite_weights.py`,
  `rescore_eval_with_fitted_gate.py`, `sweep_composite_variants.py`.
  `extract_arc_label(n_choices=5)` dispatch added in two scoring
  scripts.
- [x] **Baseline-script audit** — `run_baseline.py` argparse defaults
  read v2 panel, `rescore_baselines_through_verifier.py` `_VERIFIER_FIELDS`
  extended with `alias_overlap` + `entity_head_consistency`,
  `caem/ablation/runner.py` skip-verifier stub explicit on the new
  fields.
- [x] **Production-code audit (Fix 14)** — `docs/PRODUCTION_RUNBOOK.md`
  §11 amendments document the four operator-facing v2 deltas
  (adapter directory replaces model.pt; nested JSON schema; new
  coverage_diagnostic.json artefact; multi-probe forgetting guard)
  and extend the config-flags table. `caem_demo_server.py` updated
  with adapter-detection load path. `caem/production/confidence.py`
  audited clean.
- [x] **Orchestrator cold_start_loader wiring**
  (`scripts/run_experiment.py:1597-1638`) — closure passes
  `build_benchmark_pools(...).seed` gold pairs to
  `sil.run_cycle(..., cold_start_loader=...)`. Fix-3 cold-start now
  active.
- [x] **Orchestrator coverage_diagnostic.json write**
  (`scripts/run_experiment.py:2024-2086`) — `build_coverage_diagnostic`
  output written per cycle. Fix-5 halt triggers and Phase-4 reports
  now have the inputs they need.
- [x] **Cycle-0 fit-script per-benchmark dispatch**
  (`fit_composite_calibration.py`, `fit_conformal_gate.py`) — both
  default to `--fit_per_benchmark` so the cycle-0 JSONs land in v2
  nested schema; v1 pooled-only fit available via
  `--no-fit_per_benchmark` for ablation.
- [x] **`fit_per_benchmark_safety_floors` invocation**
  (`run_calibration.py::calibrate_pipeline`) — runs after the
  global T fit; updates both `cfg.*_per_benchmark` dicts in-memory
  AND persists them to `calibrated_config.json`. Fix-12 per-bench
  T_b + safety_u_pre_min_b active.
- [x] **`load_checkpoint` adapter detection**
  (`caem/training/self_improvement.py`) — detects
  `cycle_<N>/adapter/` first via `PeftModel.from_pretrained`; falls
  back to `cycle_<N>/model.pt` for v1. Resume-mid-trajectory now
  loads the right LoRA weights instead of silently restarting from
  pristine.
- [x] **Pre-existing v1-asserting tests** updated for v2 invariants
  (`test_benchmark_splits::TestPanelDefinition` panel sizes,
  `test_model_loader::test_config_defaults_branch_c` LoRA primary
  path).

### Phase 1a launch — READY (2026-05-07)

Total v2 commit count on `feat/qwen-3b-goal2`: 21 commits since
2026-05-06 architecture-lock entry. 773 tests pass; pre-existing 8
mock-seed `test_self_improvement.py` failures unchanged on baseline.

Wired-but-not-trivial follow-ups (deferred until Phase 1a closes):
- [ ] **Orchestrator wiring** in `scripts/run_experiment.py`
  post-cycle hook: feed
  `(candidate_counts, admitted_counts, pool_counts, per_bench_em, model)`
  to `caem.diagnostic.coverage.build_coverage_diagnostic`, write the
  result to `cycle_<N>/coverage_diagnostic.json`, append to a sliding
  window, call `evaluate_halt_triggers`, honour the `HaltDecision`.
- [ ] **Cold-start loader wiring** — orchestrator must construct a
  `cold_start_loader: Callable[[str, int], List[QAPair]]` reading
  gold pairs from `caem.config.TRAINING_BENCHMARKS` via the existing
  benchmark loaders, and pass it to `sil.run_cycle(...,
  cold_start_loader=loader)`. Without this wiring the cold-start
  fallback in Fix 3 is a no-op.

After integration sign-off:

- [ ] Pre-launch dry-run with synthetic 5-entry deferred buffer in
  `run_phase1a.sh` before step_7_main (Fix 11 layer 5)
- [ ] Final commit + push to `feat/qwen-3b-goal2`

### Day 13 — cycle 0 cold-start rebuild + calibration

- [x] Cold-start seed: 3 training benchmarks × ~1000 verified candidates (~3000 entries after dedup, ~11 GPU-hours) — completed pre-2026-05-09
- [x] Per-benchmark composite calibration (3 benchmarks × 500 cal samples = 1500 pooled)
- [x] Per-benchmark u_pre T_b calibration + safety_u_pre_min_b fit
- [~] Per-benchmark conformal gate calibration — **SUPERSEDED by Phase 1c P0' fixed-threshold gate (see Phase 0.5 below)**
- [x] Cycle 0 validation — outcome captured offline in P4 (task #140) audit report; on-disk `weight_validation.json` is empty `{}` (known gap, see "What we can claim today" in Phase 4)

---

## Phase 0.5 — Phase 1c/1d patch sequence (2026-05-09 → 2026-05-10)

Cycle-0 cal fold scored under the v2 launch-ready architecture exposed a MiniCheck pooled AUROC of 0.518 (essentially random). The patch sequence below restructured the gate and composite to recover discrimination without re-architecting the system. All patches landed before step_7_main launched 2026-05-10 07:01 UTC; the in-flight cycle-1 trajectory is running under v2.1.

### Phase 1c patches (signal + composite restructure, 2026-05-09)

| Patch | Subject | Task | Outcome |
|-----|---------|------|---------|
| P0  | Diagnose MiniCheck pooled AUROC=0.518                                  | #135 | Identified weak signals + composite over-fit to cold-start distribution |
| P0' | Replace conformal split-CP gate with fixed threshold on calibrated prob | #139 | Stable gate across cycles; no per-cycle τ refit; **store_threshold=0.60 (post-Option-4)**, defer_threshold=0.45 |
| P1  | Drop `alias_overlap` + `entity_head_consistency` from composite        | #136 | 11→9 signals; both rated essentially zero discriminative weight at cycle 0 |
| P2  | Patch directional `p_ground_max` for FEVER NEI samples                 | #137 | Fixes the "NEI is the truthful claim" direction; restores FEVER discrimination |
| P3a | Shrinkage prior (α=0.6) toward pooled fit in per-bench composite       | #138 | Reduces per-bench overfitting on n=500; KEYSTONE keeps per-bench specialisation |
| P3b | Hard `pool_max_share` cap, bench-agnostic                              | #141 | Prevents v1's FEVER-monoculture SIL-pool failure mode independently of admission rates |
| P4  | Re-run cycle 0 with patches; validate AUROC + gen→final EM gap          | #140 | Composite passes the operational test; logged offline (the empty `weight_validation.json` is a known reporting gap) |

### Phase 1d patches (gate calibration + dropped signal, 2026-05-09)

| Patch | Subject | Task | Outcome |
|-----|---------|------|---------|
| Option 4 | `store_threshold` 0.65 → 0.60 + rescore cycle 0                   | #142 | Empirical precision-cliff sat further right than the original sweep suggested |
| h_norm drop | Phase 1d removed `h_norm` from the composite (≈ −5e-4 weight at cycle 0) | (rolled into Option 4) | 10→9 signals |

### Patch 2026-05-10 — canonical composite no longer overwritten (in-flight bug fix)

`run_per_cycle_composite_refit` previously wrote `cycle_{N}/composite_calibration.json` AND atomically replaced `outputs/cycle_0/composite_calibration.json` (the canonical path) at every cycle close. Side effect: after cycle 1 closes, the file named `cycle_0/composite_calibration.json` no longer holds the cycle-0 baseline. Patch (2026-05-10) comments out the canonical-replace block with a full rationale comment and pairs it with a resume-path fix so `--resume_from_cycle N` prefers the latest `cycle_{k}/composite_calibration.json` for `k < N` over the canonical. The patch is on disk in `scripts/run_experiment.py` but the running process (PID 340588, launched 2026-05-10 07:01 UTC) has the pre-patch code cached in RAM; the patch takes effect only on next process restart.

**Cycle-2 resume protocol (patched code activates here, queued for 2026-05-11 ~06:00 UTC):**

1. Wait for cycle 1 close — both markers must fire in `outputs/full_run/run.log`:
   - `Cycle 1: checkpoint saved to outputs/full_run/cycle_1 (aborted=False).`
   - `Step 6c: gdrive offload cycle_1/ artefacts OK`
2. `tmux send-keys -t plan_a C-c`; verify PID 340588 exits cleanly.
3. **Restore the cycle-0 canonical composite** by re-running the cycle-0 fit on preserved cal-fold data:
   ```
   python scripts/fit_composite_calibration.py \
     --calib_jsons outputs/cycle_0/calibration/*.json \
     --output_json outputs/cycle_0/composite_calibration.json \
     --shrinkage_alpha 0.6 --cherian_boost --boost_C 0.01
   ```
   The pre-launch composite is not preserved on disk (overwritten by cycle 1's refit); the `pre_main_snapshot` on gdrive is two patch generations older (pre-Phase-1d, 12 signals incl. `h_norm` + `alias_overlap`). Re-fitting from the preserved cal-fold JSON reproduces the Option-4 launch state exactly.
4. Sanity-check the regenerated canonical: `em_rate ≈ 0.4153`, 9 signals, `shrinkage_alpha=0.6`.
5. Relaunch in tmux plan_a with `--resume_from_cycle 2` so a fresh Python process loads the patched `run_experiment.py`. Pipeline init reads the regenerated canonical (cycle 0 baseline); the patched resume reload loads `cycle_1/composite_calibration.json` into the verifier; Step 2.x for cycle 2 refits and writes `cycle_2/composite_calibration.json` without touching the canonical.

### Verifier reasoning-blindspot (logged 2026-05-07)

All 9 active composite signals score the canonical answer claim, not per-step claims in the chain. A reasoning chain that hits a wrong intermediate step but recovers the correct conclusion still passes the gate. Logged in memory `caem_verifier_reasoning_blindspot.md` and queued for Ch5 §sec:disc-threats + Ch6 §future-work during the report-rewrite phase. NOT a v2.1 code fix.

---

## Phase 1 — Trajectory cycles 1-10 (~17-20 days continuous, ~$160 GPU)

**Status as of 2026-05-15 05:30 BDT:** Trajectory stop locked at cycle 5. Cycles 0-3 fully closed; Cycle 4 SIL aborted on retention guard (TriviaQA-test ratio 0.8354 < 0.93 floor), Step 4 + Step 5 running on rolled-back C3 weights with C4-augmented memory; Cycle 5 will attempt next (predicted abort), then trajectory ends regardless of outcome. See "Phase 1.5 — C5-stop pivot" below for the post-trajectory plan.

**Status as of 2026-05-10 22:00 UTC:** step_7_main launched 2026-05-10 07:01 UTC in tmux `plan_a` (PID 340588). Cycle 1 is mid-Step-4 (stream-chunk training pass), currently on TriviaQA at [~1150/2000]. FEVER stream chunk closed at 18:03 UTC (EM=0.5895, 707 STORE / 2000). Cycle 1 close estimated 2026-05-11 ~06:00 UTC.

Per-cycle wall-time: ~22-26 hours on 3B + LoRA + 3-bench training panel + 2-bench transfer eval + 9-signal verifier. Cycle 1 trending toward the high end of that range.

- [x] **Cold-start + cycle 0 calibration** complete (pre-Phase-1c)
- [~] **Cycle 1** in flight under v2.1
- [ ] **Cycle 2** — patched-code restart required (see Phase 0.5 "Cycle-2 resume protocol"); cycles 2-10 run under v2.1 + Patch 2026-05-10 canonical-overwrite fix
- [ ] **Cycles 3-10**: continue under the same patched code path
- [ ] **Per-cycle artefacts** (each cycle):
  - `cycle_N/adapter/` (LoRA adapter dir; ~50 MB)
  - `cycle_N/composite_calibration.json` (per-bench dict, 9 signals, shrinkage_alpha=0.6)
  - `cycle_N/calibration/{bench}_cycle{N}.json` (cal-fold per-sample signals, n=500/bench × 3)
  - `cycle_N/calibration/calibrated_config_cycle{N}.json` (per-bench T_b + safety floors)
  - **NOT WRITTEN:** `conformal_gate.json` (retired Phase 1c P0')
  - `cycle_N/coverage_diagnostic.json` (admission rates, pool entropy, SV) — Fix 5 wiring
  - `memory_store_cycle_N.{faiss,meta}` + `deferred_buffer_cycle_N.pkl` + `retroverify_cycle{N}.json`
  - `eval/{bench}_cycle{N}.json` (held-out eval, n=300/bench × 5)
  - `eval/{bench}_cycle{N}_streamchunk.json` (Step-4 training-pass per-sample data, preserved by Step 4b snapshot)
- [ ] **Per-cycle gdrive offload**: `rclone copy outputs/full_run/cycle_N/ gdrive:caem-phase1a/v2_1_phase1d/full_run/cycle_N/` (bucket is `v2_1_phase1d`, NOT `v2-launch`; verified from the runner env at PID 340588)
- [ ] **Halt criteria**: any retention probe drops > 7% (auto-rollback fires); zero admission on training bench for 2 consecutive cycles (operator decision under fixed-threshold; auto-relax retired with conformal); manual halt if anything else looks wrong

---

## Phase 1.5 — C5-stop pivot decision (2026-05-14)

**Decision locked 2026-05-14 11:20 BDT:** Stop trajectory after C5 close regardless of outcome. Do NOT extend to C6-C10. Credit budget needs ~$30-40 for the baseline panel (the single largest remaining thesis evidence gap, H2).

### What triggered the decision
- C4 SIL aborted on retention guard: TriviaQA-test post-train 0.345 → 0.330 (ratio 0.8734 → 0.8354 across two attempts), both < 0.93 floor.
- MMLU still gaining (ratio 1.0432) and CSQA-test marginal (0.9333). Drift is concentrated on one probe.
- Cause: C4 SIL training pool was 2881 episodes vs C3's 1542 (cycle-3 deferred-buffer promotions + fuller stream chunk). 2× gradient signal in one shot exceeded the implicit LoRA envelope.
- Retention rollback is **architecturally correct behavior** (asymmetric rollback: memory preserved, adapter reverts). C4 is the first in-flight firing of the guard on the main trajectory → direct empirical receipt for H4 + load-bearing differentiator #4.

### Skip-retroverify-on-abort patch (commit 2347d60, 2026-05-14)
Applied to `scripts/run_experiment.py`. Step 2.5 retroverify now SKIPPED when `cycle_result.aborted=True`. Saves ~6h GPU per aborted cycle. Records `skipped="aborted_cycle"` in `retroverify_cycle{N}.json` so downstream readers can distinguish a skip from a genuine zero-delta retroverify. Runner restarted 2026-05-14 11:20 BDT to pick up patch; patch verified firing in C4 retry at 11:37 BDT.

### Empirical state at C3 close (the high-water mark)
| Metric | C0 | C3 | Δ |
|---|---|---|---|
| Pooled CHM | 0.156 | 0.121 | **−22% relative** |
| FEVER CHM | 0.163 | 0.117 | −29% |
| TriviaQA CHM | 0.165 | 0.123 | −25% |
| CommonsenseQA CHM | 0.132 | 0.131 | ~flat |
| TruthfulQA CHM (transfer) | 0.160 | 0.116 | −27% |
| StrategyQA CHM (transfer) | 0.161 | 0.119 | −26% |
| Pooled EM | 0.480 | 0.482 | +0.2 pp (flat) |
| T2 share (3-bench training) | 1.5% | 18.1% | +16.6 pp |
| T2 EM | 0.727 | 0.620 | held high |
| T3 EM | 0.476 | 0.460 | held (selection effect dominates) |
| False refusal rate | 0.296 | 0.069 | **−77%** (biggest single-subtype win) |
| Confident confabulation rate | 0.110 | 0.242 | +120% ✗ (CC regression, register as Ch5 §disc-threats item) |

### What C5 outcome locks
- **C5 passes** → 5-cycle parametric progression, conventional "improvement curve" headline
- **C5 also aborts** → saturation at C3 confirmed across two consecutive cycles → "architecture self-bounds at parametric ceiling, memory tier keeps amortising cost" → matches Theorem 4.x asymptote prediction empirically (arguably *more* compelling)

Either outcome publishable. The C5 result determines which story leads in Ch5 §sec:disc-headline.

### 5 pending baseline-panel decisions (lock before launching B1-B7)
From `outputs/research/topvenue_panel_2026-05-14.md`:
- [ ] Drop the agent's "MiniCheck as 10th composite signal" recommendation (CONFIRMED REDUNDANT)
- [ ] Substitute AlignScore as the lone companion evaluator (yes/no)
- [ ] Lock panel of 3 vs panel of 7 baselines (decide)
- [ ] Defer LongFact + VeriScore to rebuttal phase (decide)
- [ ] Skip GPT-4o-mini TruthfulQA judge (decide)

---

## Phase 1.6 — Post-C5 finish-line roadmap (~12-13 days, ~$50-60 GPU)

**Trigger:** C5 fully closes (estimated ~06:00 BDT May 16 if C5 aborts; ~12 hours later if it passes).

### Step P-1 — Kill trajectory runner + checkpoint state (operator, ~5 min)
- [ ] Verify C5 close: `experiment_summary.csv` has cycle-5 row, `memory_store_cycle_5.faiss` + `.meta` + `deferred_buffer_cycle_5.pkl` written.
- [ ] `tmux kill-session -t plan_a`
- [ ] `git status` (nothing should be modified by the runner itself; if it is, investigate)
- [ ] Final gdrive offload: `rclone copy outputs/full_run/cycle_5/ gdrive:caem-phase1a/v2_1_phase1d/full_run/cycle_5/`
- [ ] HF snapshot of full run (memory + adapters all cycles) for reproducibility

### Step P-2 — Baselines B1-B7 (~$30-40 GPU, ~24h with overhead)
- [ ] Lock the 5 panel decisions above
- [ ] **One-shot launcher** (post-2026-05-15 audit; bakes in all correct overrides):
  ```
  bash scripts/launch_baselines.sh all
  ```
  This runs B1-B7 sequentially (5 inference + 2 training), then post-hoc rescores all 7 through the cycle-3 verifier composite, then prints the command for stats. Defaults baked in:
  - `--n_questions 300` / `--n_eval_per_bench 300` (matches CAEM eval fold)
  - `--eval_batch_size 32` (matches CAEM step_7_main)
  - `--num_cycles 5` for B6/B7 (matches C5-stop decision)
  - `--composite_calibration outputs/full_run/cycle_3/composite_calibration.json` (matched-protocol pin; replaces the retired `outputs/production/` default)
  - `--passage_index data/passage_index` + `--seed 42` (matches CAEM)
- [ ] For individual baselines (debugging): `bash scripts/launch_baselines.sh zero_shot` (or `cot`, `rag`, etc.)
- [ ] For rescore only (after manual baseline runs): `bash scripts/launch_baselines.sh rescore`
- [ ] Verify pre-flight: launcher checks composite-pin exists + passage index exists before launching; aborts cleanly if C3 not yet closed
- [ ] Add a Ch5 §sec:setup-metrics methodology sentence: *"Every baseline's signals are computed through the same locked cycle-3 verifier composite (`outputs/full_run/cycle_3/composite_calibration.json`), so the measurement instrument is held fixed across CAEM and all seven baselines."*
- [ ] Add the cross-distribution-application caveat paragraph to Ch5 §sec:disc-threats (full text in memory file `caem_b1_vs_c0_framing.md` under "Methodology pre-registrations").
- [ ] **B7 retention-guard scope:** Ch5 §sec:comp-ewc-ft updated 2026-05-15 to say "MMLU-only retention guard" (matches `run_simple_ft.py --use_mmlu_guard`); multi-modal probe is a CAEM-specific extension not extended to B7. Document this as a deliberate scope choice in the comparison contrast, not an oversight.

### Step P-3 — Architectural ablation panel (~$15 GPU, 1-2 days)
- [ ] `python -m scripts.run_ablation --variants no_retroverify no_self_improvement no_forgetting_guard --output_dir outputs/ablation`
- [ ] All 3 lesions registered in `caem/ablation/variants.py`; the runner reuses the same `BatchPipeline` with mutated `CAEMConfig`
- [ ] Outputs land under `outputs/ablation/{variant}/cycle_{0..5}/` (only cycles that produce divergent state need re-running per `needs_cyclic_rerun` flag)

### Step P-4 — Diagnostics + theorem-receipt aggregations (~$5 GPU + scripts, ~1 day)

Complete script-run sequence (verified 2026-05-15 against actual coverage):

**Theorem-receipt aggregations** (most can run immediately after C5 close, no extra GPU):

- [ ] **Purity validation** (Theorems 1, 2, 3): `python -m scripts.run_purity_validation --cycles 0 1 2 3 4 5` — outputs `outputs/purity_validation/*.json`
- [ ] **Theorem receipts** (Thm convergence + asymptotic-elim + 3 corollaries): `python -m scripts.theorem_receipts --output_dir outputs/full_run/theorem_receipts` — outputs 5 receipt JSONs
- [ ] **Calibration trajectory** (per-cycle τ, T, store/defer rates, α purity): `python scripts/aggregate_calibration_trajectory.py --csv_out outputs/full_run/calibration_trajectory.csv --tex_out thesis_report/figures/auto/tab_calibration_trajectory.tex`
  - WARNING: script's default `--tex_out` is the legacy `"pre thesis 1 report/tables/"` path. Override with `thesis_report/figures/auto/` explicitly.
- [ ] **Per-cycle π_store trajectory** (NEW 2026-05-15): `python -m scripts.aggregate_pi_store_trajectory --max_cycle 5` — empirical receipt for H1 + thm:purity, outputs `tab_pi_store_trajectory.tex` (cal-fold + stream-chunk decomposition)
- [ ] **Per-tier CHM trajectory** (NEW 2026-05-15): `python -m scripts.aggregate_tier_chm --max_cycle 5` — empirical receipt for cor:tier1-floor, outputs `tab_tier_chm_trajectory.tex` (T1/T2/T3 EM + CHM separately)
- [ ] **Stochastic-equilibrium aggregator** (NEW 2026-05-15): `python -m scripts.aggregate_stochastic_equilibrium` — empirical receipt for cor:stochastic-equilibrium, outputs `tab_stochastic_equilibrium.tex` (per-cycle commit indicator + worst-probe + π_commit)

**Other diagnostics:**

- [ ] Per-signal correlation matrix: `python -m scripts.signal_correlation_matrix`
- [ ] Per-cycle ablation aggregation: `python -m scripts.aggregate_ablation --root outputs/ablation` (after P-3 baseline panel completes)
- [ ] Tier-1 amortisation diagnostic (Task #154): serial Pipeline at bs=1, custom small eval set, fills H5 empirical receipt
- [ ] Cohen's d trajectory per signal across cycles
- [ ] Coverage feedback diagnostic

**Known infrastructure gaps to acknowledge during P-4 run:**

- `scripts/phase4_artifacts.py` writes to `outputs/phase4/`, NOT to `thesis_report/figures/auto/` — outputs (cohen_d, composite_weights, atomic_scope, prompt_compliance, precision_cliff) need to be either (a) symlinked/copied to figures/auto/, or (b) the script's OUT_DIR patched. Pre-existing `figures/auto/cohen_d.tex` was from a v1-panel run that includes retired benchmarks (NQ, ARC, ASQA) — regenerate before final compile.
- `scripts/aggregate_calibration_trajectory.py` default `--tex_out` points to `"pre thesis 1 report/"` (legacy path). Override at the CLI call.
- `scripts/generate_session5_artifacts.py` default `--out_tables` also points to legacy path. Override at CLI call.
- Auto-tables with NO known generator: `tab_tier_baseline_cost.tex`, `tab_production_envelope_examples.tex`, `tab_prompt_design_ablation.tex` — need to be either built or removed from Ch5 if not needed.

### Step P-5 — Statistical analysis (scripts, hours)
- [ ] `python -m scripts.baseline_sig_tests` (McNemar paired + bootstrap BCa per benchmark per baseline)
- [ ] Holm correction within each benchmark family
- [ ] Architectural-claim licensing rule: Holm-adjusted p ≤ 0.05 AND bootstrap 95% CI lower bound ≥ 0.02
- [ ] Output: `outputs/baselines/sig_test_results.csv` → renders into `thesis_report/figures/auto/tab_sig_test.tex`

### Step P-6 — Auto-table generation (scripts, hours)
- [ ] `python -m scripts.make_tables` (consumes all `outputs/{full_run,baselines,ablation,purity,...}` and writes `thesis_report/figures/auto/*.tex`)
- [ ] Tables referenced from chapters: `tab_headline`, `tab_cycle_progression`, `tab_halluc_subtypes`, `tab_calibration_trajectory`, `tab_grounding`, `tab_purity`, `tab_continual`, `tab_calibration`, `tab_sig_test`, `tab_tier_baseline_cost`, `tab_ablation_results`, `cohen_d`, `composite_weights`, `cross_benchmark_summary`, `sweep_top`
- [ ] Verify each generated table renders cleanly in a test compile

### Step P-7 — Ch5/Ch6 integration writing (~2 days, no GPU)
- [ ] **THREE-HEADLINE TABLE in §sec:summary-headline** (per `caem_three_headline_decomposition.md`):
  - Row 1: B1 → CAEM C3 (architecture + SIL, the marketing number)
  - Row 2: C0 → CAEM C3 (SIL increment only, methodologically conservative)
  - Row 3: B1 → CAEM C0 (architecture only, no SIL)
  - Plus bonus row: T2 EM at C3 vs base T3 EM at C0 (+14.4 pp, distillation receipt)
  - Do NOT collapse to a single headline. The decomposition is what defuses "how much is just RAG?" in advance.
- [ ] §sec:summary-headline: replace placeholder with the three-row table populated from baseline + trajectory data
- [ ] H1-H5 verdicts in `tab:hypothesis-verdicts`: replace `main-run pending` / `ablation-run pending` with the licensed verdict (supported / partially supported / unsupported)
- [ ] §sec:disc-headline: lead with matched-protocol framing verbatim: *"Under matched-protocol cross-system evaluation, CAEM C3 reduces pooled hallucination metric by [X%] relative to the raw Qwen-2.5-3B-Instruct baseline."* Then per-row decomposition rationale.
- [ ] §sec:retention: add the C4-abort empirical receipt and the asymmetric-rollback receipt
- [ ] §sec:disc-threats: add the confident-confabulation rise (CC: 0.110 → 0.242 across C0-C3) as a probe-precision threat-to-validity item
- [ ] §sec:disc-threats: also add the single-probe-precision threat (TQA-test ratio shows ±4% relative noise between runs; per-cycle drift readings should be interpreted with this band)
- [ ] §sec:disc-threats: add the cross-distribution-application caveat paragraph (full text in `caem_b1_vs_c0_framing.md` → "Methodology pre-registrations" → "Suggested Ch5 §sec:disc-threats paragraph")
- [ ] Ch6 §concl-headline: finalize with licensing-rule outcome, same three-row decomposition structure (abstract-length version)
- [ ] Ch6 §concl-open: register the parametric-ceiling-at-3B + corpus-coverage-floor + Experience-Replay / EWC continual-learning extensions (already partially in place, expand)
- [ ] **Abstract rewrite** (`thesis_report/core/abstract.tex`) — use template from `caem_three_headline_decomposition.md`:
  - "CAEM reduces pooled hallucination metric by [X%] relative to a raw Qwen-2.5-3B-Instruct baseline under matched cross-system evaluation, with [Y%] of the reduction attributable to deployment-time mechanisms (retrieval, multi-signal verification, episodic memory, abstain class) and [Z%] to training-time self-improvement via verifier-gated distillation. The distilled model's zero-shot outputs further exceed the base-model-with-retrieval pipeline by 14.4 points exact match, demonstrating that calibrated self-distillation produces a parametric model that improves on its retrieval-augmented source."
  - Variables: X% (B1 → C3 pooled CHM reduction), Y% (B1 → C0), Z% (X − Y, SIL-attributable increment); use multiplicative-honest phrasing not subtractive percentages

### Step P-8 — Visual polish (~1-2 days)
- [ ] TikZ figures: system architecture diagram, cycle loop diagram, three-tier flow diagram
- [ ] Headline plot: pooled CHM curve C0-C5 + per-tier-share curves overlay + multi-modal probe retention curves
- [ ] Per-bench EM trajectory plot
- [ ] System architecture figure refresh in `thesis_report/figures/system_architecture_figure.tex`

### Step P-9 — Final cross-bench coherence pass (~1 day)
- [ ] Numbers match across Ch1 abstract → Ch5 tables → Ch6 conclusion (use `grep` to find every quoted number and trace to its source)
- [ ] All `\Cref{}` and `\ref{}` references resolve (LaTeX clean compile)
- [ ] Bibliography complete and sorted
- [ ] Appendix references valid

### Step P-10 — PDF build + defense prep (~1 day)
- [ ] `latexmk -pdf -interaction=nonstopmode thesis_report/main.tex` clean compile (zero warnings)
- [ ] Update `presentation/caem_supervisor.tex` with the final numbers
- [ ] Practice defense run-through with the slides

### Realistic finish date
Assuming C5 closes ~06:00 BDT May 16: thesis defensible PDF by **~May 28-30**. Parallelisable: writing (P-7) can run while GPU work (P-2 / P-3 / P-4) is in flight.

---

## Phase 2 — Report rewrite (PARALLEL to Phase 1, CPU-only)

While the GPU trajectory runs, rewrite chapters touching architecture / theory / methodology. Do not block on cycle close.

### Phase 2.1 — Theorem updates (`thesis_report/chapters/chapter_4.tex`)

- [ ] Generalize `thm:bayes-purity` to per-benchmark form: `P_c^b = p_c^b α_c^b / (p_c^b α_c^b + (1-p_c^b)(1-α_c^b))`
- [ ] Generalize `thm:monotone-purification` to per-benchmark fixed-α^b precondition
- [ ] Add precondition-starvation remark to `thm:bayes-convergence`: when admission rate falls to zero on benchmark b, the recurrence is suspended for b and trajectory is governed by parametric capacity competition; deferred-buffer reconsideration is the registered counter-mechanism
- [ ] Mark `cor:self-correction`'s improving-α precondition as empirically violated on cycle 2-4 FEVER trajectory (registered diagnostic)
- [ ] Add new theorem-or-remark: "Pool-composition divergence under verifier signal asymmetry" — formalizes the discovered failure mode with falsifiable predictions per benchmark

### Phase 2.2 — Architecture updates (`chapter_4.tex`)

- [ ] §sec:verifier — add the 11-signal description (alias_overlap + entity_head_consistency)
- [ ] §sec:storage-gate — per-benchmark conformal gate description; per-benchmark precision contracts
- [ ] §sec:sil — LoRA SIL primitive description (rank, alpha, target modules, LR schedule)
- [ ] §sec:sil — loss reweighting + cold-start fallback description
- [ ] §sec:retroverify + §sec:deferred-reconsider — five-layer guard registration
- [ ] §sec:router — per-benchmark u_pre T_b + safety_u_pre_min_b + dual-contract fit procedure

### Phase 2.3 — Methodology updates (`chapter_5.tex`)

- [ ] §sec:setup-benchmarks `tab:benchmark-pools` — replace 3 training rows (FEVER + TQA + NQ) with 4 (FEVER + TQA + HotpotQA + CSQA); move ARC + ASQA + StrategyQA to transfer panel; mark NQ as eval-only structural-failure
- [ ] §sec:setup-metrics — multi-modal retention probe description
- [ ] §sec:adj-cal-eval-gap — per-benchmark precision contracts; document α_FEVER=0.05 vs α_TQA=0.40 contract relaxation rationale
- [ ] §sec:disc-threats — three-mechanism failure-mode disclosure (FEVER monoculture; verifier task-conditioning; iterative-LoRA SV-collapse contribution gap)
- [ ] §sec:check-purity through §sec:check-corollaries — per-benchmark theorem checks against the v2 trajectory data

### Phase 2.4 — Bibliography updates (`thesis_report/main.bib`)

- [ ] Biderman 2024 (LoRA hyperparameter sweep)
- [ ] Schulman 2025 (LoRA Without Regret)
- [ ] Dettmers 2023 (QLoRA)
- [ ] Farquhar 2024 (Nature semantic entropy)
- [ ] Kuhn 2023 (semantic entropy AUROC on TriviaQA)
- [ ] Kossen 2024 (SE Probes — registered as Phase 1c future work)
- [ ] Si 2021 (Answer Equivalence — alias-aware)
- [ ] Ayoola 2022 (ReFinED entity linker — registered as Phase 1c future work)
- [ ] Lambert/Ivison 2024 (Tülu 3 — multi-task SFT precedent)
- [ ] Longpre 2023 (FLAN-v2 mixture rates)
- [ ] Quach 2024 + Mohri-Hashimoto 2024 (conformal QA precedent)
- [ ] Liang 2024 (CoDyRA continual LoRA)
- [ ] Ding 2024 (Tail Narrowing in Self-Improvement — failure-mode precedent)

### Phase 2.5 — Future work registration (`chapter_6.tex`)

- [ ] Phase 1c LoRA → 7B QLoRA scaling
- [ ] Phase 1c SE Probes (Kossen 2024) for cheap semantic entropy
- [ ] Phase 1c ReFinED entity linker (Ayoola 2022) for stronger alias overlap
- [ ] Iterative-LoRA SV-collapse measurement across self-improvement cycles (open contribution gap)
- [ ] Zero-count task fallback in iterative SIL (registered contribution given Ding 2024 documents but doesn't solve)
- [ ] Conditional conformal extension (Cherian 2024 already cited; deployment as per-benchmark gate is the v2 contribution)

---

## Phase 3 — Post-trajectory baselines + significance + Ch6 final (~$15-20 GPU)

After cycle 10 closes:

- [ ] Run B1-B7 external baselines on the matched-protocol eval fold (~$10-15)
- [ ] Run `scripts/rescore_baselines_through_verifier.py` to rescore baselines through the locked cycle-10 verifier (~$5-10)
- [ ] Statistical significance panel (paired McNemar + BCa bootstrap, Holm correction)
- [ ] Final Ch5 `tab:sig_test` + `tab:hypothesis-verdicts` populated
- [ ] Ch6 conclusion final pass

### Verifier-accuracy trajectory aggregator (NEW, queued 2026-05-10)

The reliability diagram (Fig 5.2) plots calibration shifting per cycle visually, but the codebase does not produce a scalar **verifier-accuracy trajectory** — one row per cycle with AUROC, Brier, ECE, and STORE-bucket EM. This is the headline graph that demonstrates "the verifier improves as the SIL loop improves the model." All primitives exist in `eval/metrics.py` (`auroc`, `brier_score`, `reliability_bins`) and per-sample data exists per cycle in `eval/{bench}_cycle{N}.json` + `cycle_{N}/calibration/{bench}_cycle{N}.json`; what is missing is the aggregator that iterates cycles and emits a CSV + figure.

Phase 1c P0' replaced the conformal storage gate with a fixed threshold on the calibrated probability, so the v1 plan's "verifier accuracy α increasing per cycle" claim is now expressed as **empirical storage precision rising per cycle** under the constant 0.60 gate, plus discriminative-power (AUROC) and calibration-quality (Brier, ECE) trajectories. These four metrics together replace the per-cycle α that conformal would have produced.

- [ ] Write `scripts/verifier_trajectory.py` (~80 lines):
  - For each closed cycle N and each of the 5 benches plus a pooled view, compute:
    - `auroc(u_stored, em)`, `brier_score(u_stored, em)`, ECE from `reliability_bins(u_stored, em, n_bins=10)`
    - `store_bucket_em` = mean(em where `decision == 'STORE'`), with `n_store` count and 95% Clopper-Pearson interval
    - `defer_store_em` = mean(em where `decision in {'STORE','DEFERRED'}`) as a secondary headline (optional)
  - Read held-out eval JSONs at `outputs/full_run/eval/{bench}_cycle{N}.json` for N ∈ {0, 1, …, N_closed}.
  - Also emit cal-fold version reading `outputs/full_run/cycle_{N}/calibration/{bench}_cycle{N}.json` for N ≥ 1 (plus `outputs/cycle_0/calibration/calibration_fold_samples.json` for N=0).
- [ ] Output: `outputs/phase4_artifacts/verifier_trajectory.csv` (columns: `cycle, bench, eval_or_cal, auroc, brier, ece, store_em, store_em_lo, store_em_hi, n_store, n_total`).
- [ ] Output: `outputs/phase4_artifacts/fig5_X_verifier_trajectory.{png,pdf}` with four lines (AUROC, 1−Brier, 1−ECE, STORE-bucket EM) over cycle index, one panel per bench plus a pooled panel.
- [ ] Register the new figure in `scripts/make_figures.py:FIGURE_FILES` and the new CSV in `scripts/phase4_artifacts.py:emit_summary_table`.
- [ ] Add a `tab:verifier_trajectory` reference in Ch5 §5.3 main results, with the per-cycle scalar trajectory replacing the conformal-α-per-cycle row originally drafted.
- [ ] Sanity-check option: run the aggregator against cycle 0 only (already on disk) before cycle 1 close, to validate the metric definitions and bench keys; iterate until the CSV rows match a hand-spot-check.

---

## Phase 4 — Defense day (already prepped, demo work complete)

The v1 Phase A demo work is preserved and still valid. The cycle-3 memory snapshot decision will need to be **updated to cycle-10 v2 memory snapshot** once Phase 1 closes.

- [x] **A.1.1** Mark `scripts/caem_chat.py` as legacy (header note added 2026-05-04)
- [x] **A.1.2-A.1.3** Demo server import smoke (clean import 2026-05-04)
- [x] **A.1.4** Memory snapshot decision: cycle-3 (v1) → SUPERSEDED — switch to cycle-10 v2 after Phase 1 closes
- [x] **A.2.1** `docs/PANEL_DEMO_SCRIPT.md` (7 questions, written 2026-05-04 — questions still valid; just need re-anchoring to cycle-10 v2 readings)
- [x] **A.2.2** `docs/DEMO_QUICKSTART.md` (operator cheat sheet, 2026-05-04)
- [x] **A.2.3** `scripts/expose_demo_remote.sh` (Cloudflare tunnel, 2026-05-04)
- [x] **A.3.1-A.3.3** Demo server UI polish (2026-05-04)
- [ ] **C.1.1-C.3.3** Defense-day launch + walkthrough (operator-executed on D-day)

---

## Budget envelope

```
Phase 0 (code work):           CPU only, ~$0  -- DONE
Phase 0.5 (Phase 1c/1d patches): CPU + cycle-0 rescore on GPU, ~$3-5 -- DONE
Phase 1 (trajectory):
  Cold-start rebuild           ~11 GPU-hours, ~$6  -- DONE
  Cycle 0 calibration          ~3 GPU-hours, ~$2   -- DONE
  Cycles 1-10 trajectory       ~280 GPU-hours, ~$155  -- cycle 1 in flight (Step 4)
  Subtotal Phase 1:            ~294 GPU-hours, ~$163
Phase 2 (report rewrite):      CPU only, runs IN PARALLEL with Phase 1, ~$0
Phase 3 (baselines + sig):     ~25-35 GPU-hours, ~$15-20
Phase 4 (defense day):         negligible -- single-launch demo

Total Phase 0-3:               ~$180-190
Vast credit at v2.1 launch (2026-05-10 07:00 UTC start, after top-up): ~$102 (per memory caem_phase1_credit.md)
Burn since launch (~16h wall-clock, ~1 GPU-hr/h): ~$10-12
Remaining as of 2026-05-10 22:00 UTC: ~$90-92 -- VERIFY BEFORE CYCLE-2 RELAUNCH
Recharge required for full cycles 2-10: ~$60-80 depending on actual cycle wall-time

Cycle wall-time: ~22-26 hours per cycle (cycle 1 trending high end)
Trajectory wall-clock: ~17-20 days continuous (cycles 1-10)
Total v2.1 timeline: ~30-35 days from code-start to defense-ready
```

---

## Cycle-0 launch protocol (after Phase 0 + cold-start complete)

```bash
cd /workspace/caem
git checkout feat/qwen-3b-goal2  # NEW v2 branch
source /venv/main/bin/activate
export PYTHONPATH=$PWD

# Pre-launch dry-run: verifies Fix 11 layer 5 (5-entry synthetic deferred reconsideration)
python scripts/test_deferred_reconsider_dryrun.py --synthetic_n 5

# Tmux launch
tmux new-session -d -s plan_a -x 220 -y 60 './run_phase1a.sh 2>&1 | tee -a outputs/phase1a_runner.log'

# Verify within 5 min
grep "PRE-LAUNCH DRY-RUN PASSED" outputs/phase1a_runner.log
grep "RESUMING\|STARTING EXPERIMENT" outputs/full_run/run.log

# Verify within 30 min (during cycle 0 SIL/calibration)
grep "Per-benchmark gate fitted" outputs/full_run/run.log
grep "Per-benchmark T_b fitted" outputs/full_run/run.log
grep "Cohen's d on benchmark.*passes 0.20" outputs/full_run/run.log
```

---

## Decisions log

- 2026-05-06 — v2 architecture locked. 13-fix scope, ~$165 budget, ~12-13 days code work, ~17-20 days trajectory.
- 2026-05-06 — Three ambiguities resolved: full removal of general-domain mix; local Wikidata file; keep all 11 LoRA adapters.
- 2026-05-06 — Cycle-5 partial trajectory on broken architecture halted; ~$3 burned today; ~$95 remaining.
- 2026-05-06 — `feat/qwen-3b-goal2` will be the v2 branch (parallel to v1's `feat/qwen-3b-goal1` which records the failure-mode trajectory).
- 2026-05-07 — All 13 architecture fixes merged on `feat/qwen-3b-goal2`. 377 smoke + regression tests pass. Code work compressed from the 12-13-day estimate to 1 active day after the keystone (Fix 2) landed; remaining work is integration audit + orchestrator glue rather than architectural lifts.
- 2026-05-07 — Verifier-input canonicalisation uses the short `Answer: <X>.` form (saves ~6-7 tokens/NLI call vs the longer alternative). Persisted as `feedback_uniform_verifier_input.md` memory entry.
- 2026-05-07 — Test-file consolidation: 6 fix-prefix smoke files renamed/merged into existing test homes (`test_conformal_gate.py`, `test_cal_prob_composite.py`, `test_alias_overlap.py`, `test_entity_head.py`, `test_answer_canonicalizer.py`, `test_entity_expansion_scorer.py`); cross-cutting `test_fix12` and `test_fix10` contents moved into `test_pre_routing.py` / `test_router.py` / `test_calibration_batch_equivalence.py` / `test_prompts.py` / `test_eval.py`. Old fix-prefix files for Fix 2, 9, 11, 13 still present as historical (pre-this-turn) commits.
- 2026-05-09 — Phase 1c patch sequence (P0, P0', P1, P2, P3a, P3b, P4): MiniCheck pooled AUROC=0.518 at cycle-0 cal fold triggered restructure. Conformal split-CP gate retired; replaced by fixed threshold on calibrated probability. `alias_overlap` + `entity_head_consistency` retired from composite. Shrinkage prior (α=0.6) toward pooled fit. `pool_max_share` hard cap. P4 audit logged offline (no `weight_validation.json` on disk; known reporting gap).
- 2026-05-09 — Phase 1d Option 4: `store_threshold` 0.65 → 0.60; cycle-0 rescored. `h_norm` removed (≈0 weight at cycle 0). Composite now 9 signals.
- 2026-05-10 — step_7_main launched 07:01 UTC in tmux `plan_a` (PID 340588) with the v2.1 stack on `feat/qwen-3b-goal2`. Cycle 1 SIL fine-tune closed 07:09 UTC after two backward-pass retries; recalibration + retroverify complete by 12:36 UTC; Step 4 stream-chunk pass in progress (FEVER closed 18:03 UTC, TriviaQA in flight).
- 2026-05-10 — Audited the CSQA verifier-input path. Verifier reads canonicalised option text (`canonicalize_answer` expands letters via Choices block) for all NLI-based signals; multichoice scorer wraps as `"The answer to the question is: <option_text>"`. Template inconsistency between canonicalizer (`Answer: X.`) and multichoice scorer (`The answer to the question is: X`) noted for §Threats but no functional bug.
- 2026-05-10 — Patch 2026-05-10 applied to `scripts/run_experiment.py` (canonical composite path no longer overwritten per cycle close; resume reload prefers latest per-cycle composite). Takes effect on next process restart (running process has pre-patch code cached in RAM). Cycle-2 resume protocol queued in Phase 0.5.
- 2026-05-10 — Verifier-accuracy trajectory aggregator queued (Phase 3, new task). Replaces the v1 "α increasing per cycle" conformal narrative with AUROC + Brier + ECE + STORE-bucket EM trajectories under the fixed-threshold gate.
- 2026-05-13 — Tier-1 latency measurement artefact discovered. `BatchPipeline.answer_batch` averages batch wall-time across all samples (`pipeline_batch.py:463-464`), so per-sample `latency_ms` in eval JSONs cannot resolve per-tier intrinsic cost. Slide deck's "Tier 1 ~50 ms / Tier 2 ~3-5 s / Tier 3 ~10 s" table reflects design-intent intrinsic latency, not the JSON column. Post-trajectory diagnostic (task #154) will fix the receipt path; see new "Post-trajectory diagnostics" section below.
- 2026-05-13 — Trajectory-length flexibility analysis. User considering stopping at cycle 4 or 5 instead of full 10. Empirical receipts for H1, H2 (load-bearing architectural claims), H3 (convergence deceleration) are defensible at cycle 4-5. H4 (asymptotic floor) needs cycle 7+ for strong support; reportable as "partial / consistent with prediction" at cycle 5. Recommended sweet spot: cycle 5 (frees ~$100 of credit + 5 days for writing). Cycle 10 only worth pushing for if Cohen's d on H4 looks marginal at cycle 5.

---

## Post-trajectory diagnostics (after cycle-N final close)

These run after the trajectory stops at its registered horizon (cycle 5 or cycle 10, user's choice). Each takes hours, not days, and resolves a measurement gap in the live trajectory artefacts.

### Tier-1 amortisation diagnostic (task #154)

**Why.** `BatchPipeline.answer_batch` averages batch wall-time across all samples, so the per-sample `latency_ms` column in eval JSONs reports the batch's amortised cost for every sample regardless of which tier it actually used. A Tier-1 hit logs the same latency as the Tier-3 queries in its batch. The slide deck's per-tier intrinsic latency table (~50 ms / ~3-5 s / ~10 s) is the design intent, not measurable from the live JSON column.

**What to measure.** True per-tier intrinsic cost ($c_1, c_2, c_3$) at one cycle-state. Per-cycle intrinsic costs are approximately constant across cycles (FAISS lookup grows only $O(\log n)$ with store size; LoRA inference is invariant; passage index is fixed at 21M). So one measurement plus the per-cycle tier-share trajectory (already in eval JSONs `tier` field) reconstructs every cycle's pooled wall-time.

**How.** Use the serial `caem.pipeline.Pipeline` class directly (not `BatchPipeline`), at bs=1. The serial path (`pipeline.py:644-645`) uses real `time.perf_counter()` per sample.

**Eval set.** 100 deliberately-constructed queries:
- 50 paraphrased FEVER/TQA/CSQA claims of cycle-stored entries (target Tier 1)
- 30 in-distribution but novel queries (target Tier 2)
- 20 OOD queries (target Tier 3)

**Sanity pass.** Also run against cycle-0 pipeline state (load from HF snapshot `aksaN000/caem-passage-index-21m/pre_main_snapshot/`). If $c_1, c_2, c_3$ at cycle 0 are within ~10% of the final-cycle values, the per-cycle decomposition is licensed for Ch5.

**Outputs.**
- `outputs/diagnostics/tier_latency.json` — per-tier descriptives (mean, p10, p50, p90)
- `outputs/diagnostics/per_cycle_pooled_latency.csv` — derived per-cycle pooled wall-time = $\sum_t \pi_t^{(N)} c_t$
- Cite in Ch5 §setup-reproducibility for the per-tier latency table and §results for the per-cycle amortisation receipt (H5).

**Time budget.** Pipeline setup ~1 h, 100-query bs=1 inference ~10 min per state (run twice = 20 min), analysis ~30 min. Total ~2 h.

**Script.** `scripts/measure_tier_latency.py` — to be written; sketch in `caem_tier1_latency_diagnostic.md` memory file.

### Verifier-accuracy trajectory aggregator (existing task)

Already queued from 2026-05-10. AUROC, Brier, ECE, storage-bucket exact-match per cycle per benchmark. Built from per-cycle calibration-fold sample JSONs + composite JSONs that are already on disk for cycles 0-N. CPU only, ~30 min.

### Reconsider-after-recalibration ordering audit (v2 future work, optional)

Architectural improvement: move deferred-buffer reconsideration from inside `sil.run_cycle` (Step 1) to a new Step 2.6 alongside retroverify at Step 2.5, so both passes use the same freshly refit composite. Saves ~5% retroverify compute per cycle (~15 min) and removes one-cycle promotion lag for entries the new composite would admit. **Do NOT land mid-trajectory** — breaks cycle 0-N comparability. Document as planned v2 improvement in Ch6 §future-work.

---

## What v1 left behind (preserved as evidence of failure mode)

- `gdrive:caem-phase1a/full_run/cycle_{0,1,2,3,4}/` — original-architecture trajectory artefacts
- `gdrive:caem-phase1a/full_run/cycle_5/` — cycle 5 partial (SIL + reconsideration first-fire) on broken architecture
- `outputs/full_run/aborted_cycle5_sil_2026-05-04/` — local copy of cycle-5 SIL weights from May-4 abort
- `branch_C_log.md` 2026-05-04 + 2026-05-06 entries — narrative of discovery + response
- `outputs/research/agent_research_2026-05-06.md` — literature audit informing v2

These are the "before" condition for the thesis. v2 trajectory is the "after". Ch5 §sec:disc-threats documents the failure-mode discovery + architectural response as the central methodological contribution.
