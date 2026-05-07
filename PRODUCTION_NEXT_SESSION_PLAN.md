# CAEM Production Plan — v2 (Architectural Redesign)

**v2 created:** 2026-05-06
**v1 superseded:** v1 (2026-05-04) was a continuation plan for the broken-architecture trajectory. v2 replaces it with the discovered-failure-mode + architectural-response plan.
**Source of authority:** `CAEM_FIX_AUDIT.md` (pre-implementation read), `branch_C_log.md` 2026-05-06 entry (failure mode + v2 architecture lock), `outputs/research/agent_research_2026-05-06.md` (literature audit).

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

## Architecture decisions (locked v2)

```
Model:               Qwen 2.5-3B-Instruct, no quantization (7B QLoRA registered as Phase 1c future work)
SIL primitive:       LoRA r=32, α=64, all-linear targets {q,k,v,o,gate,up,down}, LR=2e-4 with cycle decay
                     L2 anchor REMOVED on LoRA path (LoRA's parameter budget IS the implicit anchor)
Storage gate:        Per-benchmark conformal at α_b
                     α_FEVER=0.05, α_TQA=0.40 (refit per-cycle), α_HotpotQA=0.05, α_CSQA=0.05
Composite:           Per-benchmark isotonic + boost, 11 signals (h_norm STAYS RETIRED)
                     9 base signals + alias_overlap (NEW) + entity_head_consistency (NEW)
u_pre:               Per-benchmark T_b temperature scaling
                     Per-benchmark safety_u_pre_min_b (data-driven from cal fold, dual contract: precision ≥ 0.85 AND coverage ≥ 0.40)
SIL pool:            Loss reweighting via temperature mixing T=2 + 3× upsampling cap + DoReMi floor + cold-start gold fallback
                     REMOVE the general-domain mix (1000 hardcoded TriviaQA samples per cycle)
Retention probe:     Multi-modal — MMLU 200 + TriviaQA test 200 + HotpotQA test 200
                     Halt-and-rollback if ANY probe drops > 7% from pristine
Coverage diag:       Per-benchmark admission rate + pool composition entropy + per-cycle adapter SVD + per-bench EM trajectory
                     Halt triggers: zero-admission for 2 cycles → auto-relax α_b
Prompts:             Per-benchmark templates with answer-expansion for MCQ; benchmark-conditional verifier dispatch
                     New: CommonsenseQA 5-choice template, HotpotQA multi-hop template
Deferred reconsider: Five-layer hard-fail guard (orchestrator assert + SIL hard-fail + post-cycle log assert + unit test + pre-launch dry-run)

Training panel (4):       FEVER + TriviaQA + HotpotQA + CommonsenseQA
Transfer eval panel (3):  TruthfulQA + StrategyQA + NaturalQuestions
Stream chunk:             1000/cycle (FEVER, TQA, HotpotQA), 700/cycle (CSQA — small training pool)
Cycle count:              10 trajectory cycles + cycle 0 calibration = 11 total
```

Three small ambiguities resolved (2026-05-06):
- **Fix 13 (general-domain mix):** full removal (cleaner code, smaller maintenance surface)
- **Fix 6 (alias data):** local 150 MB Wikidata file (avoids 9 min/cycle network latency)
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

- [ ] Cold-start seed: 4 training benchmarks × 1000 verified candidates → ~3000-3200 cold-start memory entries (~11 GPU-hours)
- [ ] Per-benchmark composite calibration (4 benchmarks × 500 cal samples)
- [ ] Per-benchmark conformal gate calibration
- [ ] Per-benchmark u_pre T_b calibration + safety_u_pre_min_b fit
- [ ] Cycle 0 validation gate: per-benchmark Cohen's d ≥ 0.20 (verify each benchmark passes)

---

## Phase 1 — Trajectory cycles 1-10 (~17-20 days continuous, ~$160 GPU)

Per-cycle wall-time: ~22-26 hours on 3B + LoRA + 4-bench panel + new signals.

- [ ] **Cycle 1** through **Cycle 10**: launch in tmux `plan_a` via `./run_phase1a.sh` after Phase 0 lock
- [ ] **Per-cycle artefacts** (each cycle):
  - `cycle_N/adapter_model.bin` (~50 MB LoRA adapter)
  - `cycle_N/adapter_config.json` (PEFT metadata)
  - `cycle_N/composite_calibration.json` (per-benchmark dict)
  - `cycle_N/conformal_gate.json` (per-benchmark dict)
  - `cycle_N/safety_threshold.json` (per-benchmark T_b + safety_u_pre_min_b)
  - `cycle_N/coverage_diagnostic.json` (admission rates, pool entropy, SV)
  - `memory_store_cycle_N.{faiss,meta}` + `deferred_buffer_cycle_N.pkl` + `retroverify_cycleN.json`
- [ ] **Per-cycle gdrive offload**: `rclone copy outputs/full_run/cycle_N/ gdrive:caem-phase1a-v2/full_run/cycle_N/`
- [ ] **Halt criteria**: any retention probe drops > 7% (auto-rollback fires); per-benchmark admission rate zero for 2 cycles (auto-relax α_b); manual halt if anything else looks wrong

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
Phase 0 (code work):           CPU only, ~$0
Phase 1 (trajectory):
  Cold-start rebuild           ~11 GPU-hours, ~$6
  Cycle 0 calibration          ~3 GPU-hours, ~$2
  Cycles 1-10 trajectory       ~280 GPU-hours, ~$155
  Subtotal Phase 1:            ~294 GPU-hours, ~$163
Phase 2 (report rewrite):      CPU only, runs IN PARALLEL with Phase 1, ~$0
Phase 3 (baselines + sig):     ~25-35 GPU-hours, ~$15-20
Phase 4 (defense day):         negligible — single-launch demo

Total Phase 0-3:               ~$180-185
Current Vast budget:           ~$95 (after today's burn)
Recharge required:             ~$90-100

Cycle wall-time: ~22-26 hours per cycle
Trajectory wall-clock: ~17-20 days continuous (cycles 1-10)
Total v2 timeline: ~30-35 days from code-start to defense-ready
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

---

## What v1 left behind (preserved as evidence of failure mode)

- `gdrive:caem-phase1a/full_run/cycle_{0,1,2,3,4}/` — original-architecture trajectory artefacts
- `gdrive:caem-phase1a/full_run/cycle_5/` — cycle 5 partial (SIL + reconsideration first-fire) on broken architecture
- `outputs/full_run/aborted_cycle5_sil_2026-05-04/` — local copy of cycle-5 SIL weights from May-4 abort
- `branch_C_log.md` 2026-05-04 + 2026-05-06 entries — narrative of discovery + response
- `outputs/research/agent_research_2026-05-06.md` — literature audit informing v2

These are the "before" condition for the thesis. v2 trajectory is the "after". Ch5 §sec:disc-threats documents the failure-mode discovery + architectural response as the central methodological contribution.
