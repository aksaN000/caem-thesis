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

## Phase 0 — Implementation prep (~12-13 days, ALL CPU-only, no GPU spend)

The 13 fixes in dependency-respecting order. Detailed file:line targets in `CAEM_FIX_AUDIT.md`.

### Day 1-2 — independent + mechanical fixes

- [ ] **Fix 9** Training panel update + new benchmark loaders
  - Update `caem/config.py` `TRAINING_BENCHMARKS` and `TRANSFER_BENCHMARKS` constants
  - Update `caem/benchmark_splits.py` `_EVAL_SPLIT_MAP` for HotpotQA + CommonsenseQA
  - Add per-benchmark stream-chunk override (CSQA=700, others=1000)
  - Add `load_hotpotqa()` and `load_commonsense_qa()` loaders to `eval/benchmarks.py`
  - Add metric-scoring branches for new benchmarks (extract_csqa_label 5-choice; HotpotQA span extraction)
  - Update `run_phase1a.sh:78` BENCHMARKS array
  - Fix hardcoded benchmark check at `scripts/seed_cold_start.py:161` (read from TRAINING_BENCHMARKS)
  - ~200 lines

- [ ] **Fix 13** Remove general-domain mix from SIL pool
  - Delete `load_general_data()` (`scripts/run_experiment.py:305-365`)
  - Remove `general_data` kwarg from `sil.run_cycle()` invocation
  - Simplify `_mix()` (`caem/training/self_improvement.py:696-710`) or remove entirely
  - ~30 lines

- [ ] **Fix 11 (layers 1-3)** Deferred-reconsideration hard-fail guard
  - Layer 1: Orchestrator-level `assert pipeline.deferred_buffer is not None` at `run_experiment.py:1591`
  - Layer 2: Promote SIL `WARNING` to `RuntimeError` (`self_improvement.py:449-462`) with `--allow_skip_deferred` opt-out
  - Layer 3: Post-cycle log assertion (regex `Deferred reconsideration: sweeping \d+ entries`)
  - ~120 lines + commits

### Day 3-5 — keystone per-benchmark cascade

- [ ] **Fix 2** Per-benchmark composite weights (KEYSTONE — every per-bench fix flows from this)
  - `CalProbComposite` accepts per-benchmark calibrations dict
  - JSON schema bumped to `branchC.2026-05-06` with nested `{benchmark: {signal_calibrations}}`
  - Per-benchmark fit in `scripts/fit_composite_calibration.py` (groupby benchmark)
  - **Add `source_benchmark` parameter to `verify()` and `verify_batch()`** — every callsite must update
  - Pass `source_benchmark` to `_composite()` for per-bench lookup
  - Update every verifier call site: pipeline (line 1350), retroverify closure, deferred reconsider closure, run_calibration cache loader
  - ~200 lines

- [ ] **Fix 1** Per-benchmark conformal gate
  - Promote `scripts/conditional_conformal_ablation.py:81-120` `fit_per_benchmark_gate()` to deployment
  - Wrap `ConformalStorageGate` for per-benchmark dict-of-gates
  - Add per-benchmark fit loop to `scripts/fit_conformal_gate.py`
  - Add per-benchmark EMA blending at `scripts/recalibrate_conformal_at_cycle.py`
  - Pipeline routes by `source_benchmark` to gate at storage decision (`pipeline.py:907-930`)
  - ~80 lines (pattern already exists)

- [ ] **Fix 12** Per-benchmark u_pre T_b + safety_u_pre_min_b
  - `config.temperature_scalar` → `config.temperature_scalars: Dict[str, float]` (with `_global_fallback` key)
  - `config.safety_u_pre_min` → `config.safety_u_pre_mins: Dict[str, float]`
  - `_apply_temperature_scaling()` accepts `source_benchmark` and dict-lookup T_b (`pre_routing.py:344-359`)
  - `AdaptiveRouter.route()` looks up `safety_u_pre_min_b` by benchmark
  - `run_calibration.py:refit_temperature_only()` fits per-benchmark T_b on per-benchmark cal slices
  - NEW: `safety_u_pre_min_b` data-driven fit at cycle 0 (precision floor 0.85 AND coverage floor 0.40 dual contract)
  - Pipeline threads `source_benchmark` to `pre_routing.estimate()` and `router.route()`
  - ~110 lines

- [ ] **Fix 11 (layer 4)** Unit test scaffolding
  - New file `tests/test_orchestrator_wiring.py`
  - Tests: orchestrator passes deferred_buffer + reconsider_fn; SIL raises RuntimeError without; end-to-end synthetic deferred entry promotion
  - ~80 lines

### Day 6-7 — verifier signals + dispatch

- [ ] **Fix 6** alias_overlap signal (Wikidata lookup)
  - New file `caem/verification/alias_overlap.py`
  - Local 150 MB Wikidata alias file download (one-time setup)
  - Integrate as 10th signal in `verifier.verify()` and `verify_batch()`
  - Add to `UnifiedVerifierOutput` schema (`verifier.py:166-207`)
  - Update `_derive_verifier_fields()` in eval harness
  - Add to `COMPOSITE_SIGNALS` tuple in `cal_prob_composite.py`
  - ~220 lines

- [ ] **Fix 7** entity_head_consistency signal (M=3 chain head agreement)
  - Head-noun extractor (regex + spaCy fallback) in `caem/verification/`
  - Pairwise agreement scoring derived from existing M=3 chains in `verify_batch` (zero extra compute)
  - Add to `UnifiedVerifierOutput` and `COMPOSITE_SIGNALS`
  - ~80 lines

- [ ] **Fix 10** Per-benchmark prompts + verifier dispatch
  - Add `commonsense_qa` and `hotpotqa` branches to `detect_query_task()` (`prompts.py:73-85`)
  - Add `_task_spec()` branches: CSQA 5-choice instruction; HotpotQA multi-hop instruction
  - Add `_few_shot_parts()` examples for CSQA + HotpotQA
  - Verify `multichoice_scorer.py` regex handles 5-choice (already supports A-E per audit)
  - NEW file `caem/verification/entity_expansion_scorer.py` for bare-entity NLI grounding (TriviaQA + HotpotQA)
  - Insert into `_p_ground_with_direction()` dispatcher (`verifier.py:699-726`)
  - Update `extract_arc_label` (4-choice) to handle 5-choice for CSQA in `eval/metrics.py`
  - ~500 lines

### Day 8-10 — training-side + diagnostics

- [ ] **Fix 8** LoRA SIL primitive (CRITICAL — biggest single change)
  - Wrap base model with `get_peft_model(model, LoraConfig(r=32, α=64, all-linear))` post-construction (`model_loader.py:446-470`)
  - Verify `torch.compile` compatibility with PeftModel wrapper (fullgraph=False already set)
  - Gate L2 anchor: `if cfg.use_lora_training: skip _l2_penalty()` (`self_improvement.py:893-894, 971-996`)
  - Conditional checkpoint save: adapter-only via `model.save_pretrained()` (`self_improvement.py:1183`)
  - Conditional checkpoint load: `PeftModel.from_pretrained(base, adapter_path)` (`self_improvement.py:559`)
  - LoRA-specific LR (2e-4) with cycle decay `LR_c = 2e-4 / (1 + 0.15·c)`
  - Per-cycle adapter SVD logging (Biderman 2024 contribution gap)
  - Update rolling-N retention deletion (smaller adapter files; keep all 11 instead of rolling)
  - Update `--resume_from_cycle` path for adapter-only restore
  - ~280 lines

- [ ] **Fix 3** Loss-reweighted SIL pool builder
  - Insert per-task weight computation between `_collect_episodes` and `_mix` (`self_improvement.py:301-317`)
  - Implement bounded upsampling 3× cap + DoReMi floor (≥1e-3) + temperature mixing T=2
  - Cold-start gold-labelled fallback: new method `_load_gold_fallback(task, n)` for zero-count benchmarks
  - Add config fields: `loss_temperature=2.0`, `max_task_upsample_ratio=3.0`, `min_task_weight=1e-3`, `cold_start_fallback_per_task=100`
  - ~150 lines

- [ ] **Fix 4** Multi-modal retention probe
  - Replace single `_mmlu_score(n=200)` with `_measure_all_retention_probes()` returning dict (`self_improvement.py:1010-1113`)
  - New methods `_triviaqa_open_score(n=200)` (F1) and `_hotpotqa_multihop_score(n=200)` (EM)
  - Halt-and-rollback gate fires on ANY probe drop > 7%
  - Pristine baseline measurement at cycle 0 includes all 3 probes
  - Add config fields: retention probe sizes + drop threshold
  - ~100 lines

- [ ] **Fix 5** Coverage feedback diagnostic + halt triggers
  - New module `caem/diagnostic/coverage.py`
  - Per-cycle JSON: per-benchmark admission rate + pool composition entropy + per-bench EM + adapter SV
  - Halt triggers: zero-admission-for-2-cycles → auto-relax α_b; SV collapse alarm
  - Orchestrator integration in `run_experiment.py` post-cycle
  - ~150 lines

### Day 11-12 — smoke test + bug fixes

- [ ] **Smoke test 1**: PEFT wrapping smoke (10 samples through cycle 0 SIL with LoRA)
- [ ] **Smoke test 2**: Per-benchmark composite end-to-end (verify routes by source_benchmark)
- [ ] **Smoke test 3**: Multi-modal retention probe (all 3 probes return non-NaN)
- [ ] **Smoke test 4**: Deferred reconsideration five-layer guard fires when expected
- [ ] **Fix 11 (layer 5)** Pre-launch dry-run with synthetic 5-entry deferred buffer in `run_phase1a.sh` before step_7_main
- [ ] Fix any bugs surfaced; commit + push to `feat/qwen-3b-goal2` branch (NEW BRANCH for v2 architecture)

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

---

## What v1 left behind (preserved as evidence of failure mode)

- `gdrive:caem-phase1a/full_run/cycle_{0,1,2,3,4}/` — original-architecture trajectory artefacts
- `gdrive:caem-phase1a/full_run/cycle_5/` — cycle 5 partial (SIL + reconsideration first-fire) on broken architecture
- `outputs/full_run/aborted_cycle5_sil_2026-05-04/` — local copy of cycle-5 SIL weights from May-4 abort
- `branch_C_log.md` 2026-05-04 + 2026-05-06 entries — narrative of discovery + response
- `outputs/research/agent_research_2026-05-06.md` — literature audit informing v2

These are the "before" condition for the thesis. v2 trajectory is the "after". Ch5 §sec:disc-threats documents the failure-mode discovery + architectural response as the central methodological contribution.
