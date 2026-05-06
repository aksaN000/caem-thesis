# CAEM Fix Audit — pre-implementation read

**Date:** 2026-05-06
**Purpose:** Comprehensive read of every change-surface area for the 13-fix CAEM redesign. Surfaces refactor risks, hidden coupling, and concrete file:line targets BEFORE any code change.
**Method:** Five parallel Explore agents covered thematic slices; this doc synthesizes their findings.
**Status:** Read complete. Awaits review before implementation begins.

---

## 1. Locked architecture (recap, for cross-reference)

```
Model:               Qwen 2.5-3B-Instruct, no quantization
SIL primitive:       LoRA r=32, α=64, all-linear targets, LR=2e-4 with cycle decay; L2 anchor REMOVED
Storage gate:        Per-benchmark conformal at α_b
Composite:           Per-benchmark isotonic + boost, 11 signals (h_norm RETIRED)
                     Add: alias_overlap, entity_head_consistency
u_pre:               Per-benchmark T_b temperature; per-benchmark safety_u_pre_min_b
SIL pool:            Loss reweighting (T=2 mixing) + 3× upsampling cap + DoReMi floor + cold-start fallback
                     REMOVE the general-domain mix (1000 hardcoded TriviaQA samples)
Retention probe:     Multi-modal — MMLU 200 + TriviaQA 200 + HotpotQA 200; halt on any > 7% drop
Coverage diag:       Per-benchmark admission/EM/SV monitoring
Prompts:             Per-benchmark templates (CSQA, HotpotQA new); benchmark-conditional verifier dispatch
Deferred reconsider: Five-layer hard-fail guard
Training panel:      FEVER + TriviaQA + HotpotQA + CommonsenseQA
Transfer eval panel: TruthfulQA + StrategyQA + NaturalQuestions
Stream chunk:        1000/cycle (FEVER, TQA, HotpotQA), 700/cycle (CSQA)
```

---

## 2. The single biggest finding from the read

**Critical coupling gap** — the verifier currently has NO `source_benchmark` parameter on its `verify()` API. Without that parameter, per-benchmark composite weights cannot be looked up at scoring time.

- `caem/pipeline.py:answer()` (lines 422-432) **does** receive `source_benchmark` from `eval/harness.py:382`
- `caem/pipeline.py:_maybe_store()` (lines 849-856) **does** receive it
- BUT `pipeline.verifier.verify(query, answer, ...)` at `pipeline.py:1350` does **NOT** pass `source_benchmark` to the verifier

**Implication for Fix 2 (per-benchmark composite):** every callsite of `verifier.verify()` and `verifier.verify_batch()` must be updated to pass `source_benchmark`. The verifier signature must change. This is the MOST IMPORTANT refactor in the whole 13-fix set because every per-benchmark behavior depends on it.

The `make_reconsider_deferred_fn()` closure at `pipeline.py:830-843` also drops the benchmark tag — `verifier.verify(deferred_entry.question, deferred_entry.answer)` — even though `deferred_entry.source_benchmark` is available. Fix 2 + Fix 11 must both update this site.

---

## 3. Per-fix implementation plan with file:line targets

### Fix 1 — Per-benchmark conformal gate

**Promote existing prototype**: `scripts/conditional_conformal_ablation.py:81-120` already implements `fit_per_benchmark_gate()` and is production-ready. Copy that pattern into deployment paths.

| Change | File:line | Risk |
|--------|-----------|------|
| Wrap `ConformalStorageGate` for per-benchmark dict-of-gates | `caem/verification/conformal_gate.py:55-218` | MEDIUM |
| Add per-benchmark fit loop to deployment fitter | `scripts/fit_conformal_gate.py:84-117` | LOW |
| Add per-benchmark EMA blending at cycle boundary | `scripts/recalibrate_conformal_at_cycle.py:131-226` | LOW |
| Update JSON schema (nested `{per_benchmark: {bm: gate_dict}}` + `global` fallback) | `conformal_gate.py:181-218` | MEDIUM |
| Pipeline routes by `source_benchmark` to gate at storage decision | `caem/pipeline.py:907-930` | MEDIUM |

**Risk total: MEDIUM.** Pattern already exists; mostly mechanical promotion.

### Fix 2 — Per-benchmark composite weights (HIGHEST coupling impact)

| Change | File:line | Risk |
|--------|-----------|------|
| `CalProbComposite` accepts per-benchmark calibrations dict | `caem/verification/cal_prob_composite.py:191-417` | HIGH |
| JSON schema bumped to `branchC.2026-05-06` with nested `{benchmark: {signal_calibrations}}` | `cal_prob_composite.py:211, 374-417` | HIGH |
| Per-benchmark fit in deployment fitter (groupby benchmark, fit each) | `scripts/fit_composite_calibration.py:62-150` | HIGH |
| **Add `source_benchmark` parameter to `verify()` and `verify_batch()`** | `caem/verification/verifier.py:732-995, 1001-1199` | HIGH |
| Pass `source_benchmark` to `_composite()` for per-bench lookup | `verifier.py:943, 2465-2542` | HIGH |
| Update every verifier call site (pipeline, retroverify, deferred reconsider, run_calibration) | multiple — see §4 below | HIGH |

**Risk total: HIGH.** Not because the math is hard, but because every consumer of `verifier.verify()` must be updated.

### Fix 3 — Loss-reweighted SIL pool builder (no hard cap)

| Change | File:line | Risk |
|--------|-----------|------|
| Insert per-task weight computation between `_collect_episodes` and `_mix` | `caem/training/self_improvement.py:301-317` | HIGH |
| Implement bounded upsampling 3× cap + DoReMi floor (≥1e-3) + temperature mixing T=2 | new method `_compute_task_weights()` + `_upsample_by_task()` | HIGH |
| Cold-start gold-labelled fallback when verified count = 0 | new method `_load_gold_fallback()` | MEDIUM |
| Add config fields: `loss_temperature=2.0`, `max_task_upsample_ratio=3.0`, `min_task_weight=1e-3`, `cold_start_fallback_per_task=100` | `caem/config.py:614-650` | LOW |

**Risk total: HIGH.** Algorithmic correctness needs unit tests (per-task gradient share verification).

### Fix 4 — Multi-modal retention probe

| Change | File:line | Risk |
|--------|-----------|------|
| Replace single `_mmlu_score(n=200)` with `_measure_all_retention_probes()` returning dict | `caem/training/self_improvement.py:1010-1113` | HIGH |
| New methods `_triviaqa_open_score(n=200)` (F1) and `_hotpotqa_multihop_score(n=200)` (EM) | append to `self_improvement.py` | HIGH |
| Halt-and-rollback gate fires on ANY probe drop > 7% | `self_improvement.py:384-391` | HIGH |
| Add config fields: `retention_probe_triviaqa_n=200`, `retention_probe_hotpotqa_n=200`, `retention_drop_threshold=0.07` | `caem/config.py:649` | LOW |
| Pristine baseline measurement at cycle 0 must include all 3 probes | `self_improvement.py:325-345` | MEDIUM |

**Risk total: HIGH.** Multi-probe rollback logic needs careful unit tests.

### Fix 5 — Coverage feedback diagnostic + halt triggers

| Change | File:line | Risk |
|--------|-----------|------|
| New file with per-cycle diagnostic JSON writer | `caem/diagnostic/coverage.py` (new) | LOW |
| Per-benchmark admission rate + pool composition entropy + per-bench EM + SV monitoring | new module | MEDIUM |
| Halt triggers: zero-admission-for-2-cycles → relax α_b automatically; SV collapse alarm | new module | MEDIUM |
| Orchestrator integration | `scripts/run_experiment.py` (post-cycle) | LOW |

**Risk total: MEDIUM.** Pure diagnostic + halt logic; doesn't change the trained model.

### Fix 6 — alias_overlap signal (Wikidata lookup)

| Change | File:line | Risk |
|--------|-----------|------|
| New file with alias-set lookup logic | `caem/verification/alias_overlap.py` (new) | MEDIUM |
| Integrate as 10th signal in verifier | `caem/verification/verifier.py:732-995` (within `verify()`) | HIGH |
| Add to `UnifiedVerifierOutput` schema | `verifier.py:166-207` | HIGH |
| Update verifier-fields registry in eval harness | `eval/harness.py:89-121` (`_derive_verifier_fields`) | MEDIUM |
| Add to `COMPOSITE_SIGNALS` tuple | `cal_prob_composite.py:70-81` | LOW |
| Wikidata alias data file: ~150 MB download, cache locally | new — `data/wikidata_aliases/` | LOW |

**Risk total: HIGH.** Schema changes ripple to harness, metrics, memory.

### Fix 7 — entity_head_consistency signal (M-chain head agreement)

| Change | File:line | Risk |
|--------|-----------|------|
| Derive from existing M=3 chains in `verify_batch` | `verifier.py:1085-1094` | MEDIUM |
| Head-noun extractor (regex + spaCy fallback) | new helper in `verifier.py` or `caem/prompts.py` | MEDIUM |
| Pairwise agreement scoring | new method | LOW |
| Add to `UnifiedVerifierOutput` + `COMPOSITE_SIGNALS` | same as Fix 6 | LOW |

**Risk total: MEDIUM.** Reuses existing chains so zero extra compute.

### Fix 8 — LoRA SIL primitive

| Change | File:line | Risk |
|--------|-----------|------|
| Wrap base model with `get_peft_model(model, LoraConfig(r=32, α=64, all-linear, lora_dropout=0.05))` post-construction | `caem/model_loader.py:446-470` | CRITICAL |
| Verify `torch.compile` compatibility with PeftModel wrapper (fullgraph=False already set, should work) | `model_loader.py:505-510` | MEDIUM |
| Gate L2 anchor: `if cfg.use_lora_training: skip _l2_penalty()` | `caem/training/self_improvement.py:893-894, 971-996` | HIGH |
| Conditional checkpoint save: adapter-only via `model.save_pretrained()` | `self_improvement.py:1183` | HIGH |
| Conditional checkpoint load: `PeftModel.from_pretrained(base, adapter_path)` | `self_improvement.py:559` | HIGH |
| LoRA-specific LR (2e-4 vs 5e-5) with cycle decay `LR_c = 2e-4 / (1 + 0.15·c)` | `self_improvement.py:716-757` | LOW |
| Per-cycle adapter SVD logging (Biderman 2024 contribution gap) | new method, called post-train | LOW |
| Update rolling-N retention deletion (smaller adapter files, ~50 MB vs 6 GB) | `self_improvement.py:1206-1226` | LOW (already works, just smaller files) |

**Risk total: CRITICAL.** PEFT wrapping + conditional save/load is the largest behavioral change in the entire fix set. Requires careful smoke-testing before launch.

### Fix 9 — Training panel update + new benchmark loaders

| Change | File:line | Risk |
|--------|-----------|------|
| Update `TRAINING_BENCHMARKS = ("fever", "triviaqa", "hotpotqa", "commonsense_qa")` and `TRANSFER_BENCHMARKS = ("truthfulqa", "strategyqa", "natural_questions")` | `caem/config.py:39-59` (likely TRAINING_BENCHMARKS constant) | LOW |
| Add `_EVAL_SPLIT_MAP` entries for HotpotQA and CommonsenseQA | `caem/benchmark_splits.py:88-102` | LOW |
| Add per-benchmark stream-chunk override mechanism (CSQA=700, others=1000) | `benchmark_splits.py:233-380` | MEDIUM |
| Add `load_hotpotqa()` and `load_commonsense_qa()` loaders | `eval/benchmarks.py` (new functions) | MEDIUM |
| Add metric-scoring branches for new benchmarks | `eval/metrics.py` + `eval/harness.py` | MEDIUM |
| Update `BENCHMARKS=(...)` array | `run_phase1a.sh:78` | LOW |
| Verify `seed_cold_start.py` reads from `TRAINING_BENCHMARKS` (not hardcoded) | `scripts/seed_cold_start.py:161` (hardcoded check identified by audit) | MEDIUM |
| Update `TRAINING_BENCHMARKS` references across thesis chapters (Ch3, Ch5 §sec:setup-benchmarks, Ch1 hypotheses table) | thesis_report/*.tex | LOW |

**Risk total: MEDIUM.** Mostly mechanical, but the seed_cold_start hardcoding is a hidden coupling that could silently break the cold-start phase.

### Fix 10 — Per-benchmark prompts + verifier dispatch

| Change | File:line | Risk |
|--------|-----------|------|
| Add `commonsense_qa` and `hotpotqa` branches to `detect_query_task()` | `caem/prompts.py:73-85` | MEDIUM |
| Add `_task_spec()` branches for CSQA + HotpotQA (5-choice instruction; multi-hop instruction) | `prompts.py:104-177` | MEDIUM |
| Add `_few_shot_parts()` examples for CSQA + HotpotQA | `prompts.py:184-306` | MEDIUM |
| Verify `multichoice_scorer.py` regex handles 5-choice (already supports A-E per audit) | `caem/verification/multichoice_scorer.py:86-141` | LOW |
| New `entity_expansion_scorer.py` for bare-entity NLI grounding fallback (TriviaQA, HotpotQA) | new file | HIGH |
| Insert into `_p_ground_with_direction()` dispatcher between Tier-2 multichoice and fallback NLI | `verifier.py:699-726` | HIGH |
| Update `extract_arc_label` (4-choice) to handle 5-choice for CSQA | `eval/metrics.py` | LOW |

**Risk total: HIGH.** Bare-entity expansion scorer is genuinely new code; entity-name normalization edge cases need tests.

### Fix 11 — Deferred-reconsideration five-layer guard

**Audit confirms the May-4 patch is in place** (`run_experiment.py:1609`, `self_improvement.py:476-480`). What's missing is the additional 4 hardening layers:

| Change | File:line | Risk |
|--------|-----------|------|
| Layer 1: Orchestrator-level `assert pipeline.deferred_buffer is not None` at run_cycle entry | `scripts/run_experiment.py:1591-1610` (before sil.run_cycle) | LOW |
| Layer 2: Promote SIL `WARNING` to `RuntimeError` (with `--allow_skip_deferred` opt-out) | `caem/training/self_improvement.py:449-462` | LOW |
| Layer 3: Post-cycle log assertion (regex `Deferred reconsideration: sweeping \d+ entries`) | `run_experiment.py` (post-cycle) | LOW |
| Layer 4: Unit test file | `tests/test_orchestrator_wiring.py` (new) | LOW |
| Layer 5: Pre-launch dry-run with synthetic 5-entry deferred buffer | `run_phase1a.sh` pre-step_7_main check | MEDIUM |

**Risk total: LOW.** Mostly defensive plumbing.

### Fix 12 — Per-benchmark u_pre T_b + safety_u_pre_min_b

| Change | File:line | Risk |
|--------|-----------|------|
| `config.temperature_scalar` → `config.temperature_scalars: Dict[str, float]` (with `_global_fallback` key) | `caem/config.py:139` | LOW |
| `config.safety_u_pre_min` → `config.safety_u_pre_mins: Dict[str, float]` | `caem/config.py:134` | LOW |
| `_apply_temperature_scaling()` accepts `source_benchmark` and looks up T_b | `caem/confidence/pre_routing.py:344-359` | LOW |
| `PreRoutingConfidenceEstimator.estimate()` accepts and threads `source_benchmark` | `pre_routing.py:120-162` | LOW |
| `AdaptiveRouter.route()` looks up `safety_u_pre_min_b` by benchmark | `caem/routing/router.py:128-143` | MEDIUM |
| `run_calibration.py:refit_temperature_only()` fits per-benchmark T_b on per-benchmark cal slices | `scripts/run_calibration.py:692-812` | MEDIUM |
| Cycle-0 calibration step adds the `safety_u_pre_min_b` data-driven fit (precision floor 0.85, coverage floor 0.40) | `scripts/run_calibration.py` (new function) | MEDIUM |
| Pipeline threads `source_benchmark` to `pre_routing.estimate()` and `router.route()` | `caem/pipeline.py:answer()` | MEDIUM |

**Risk total: MEDIUM.** All single-scalar reads must be updated to dict lookups; pipeline must thread benchmark tag through.

### Fix 13 — Remove general-domain mix from SIL pool

| Change | File:line | Risk |
|--------|-----------|------|
| Remove `load_general_data()` function | `scripts/run_experiment.py:305-365` | LOW |
| Remove `general_data` kwarg from `sil.run_cycle()` invocation | `run_experiment.py:1284-1300, 1607` | LOW |
| Remove `general_data` parameter from `SelfImprovementLoop.run_cycle` (or accept None default) | `caem/training/self_improvement.py:288-317` | LOW |
| Remove `_mix()` method or simplify to identity | `self_improvement.py:696-710` | LOW |

**Risk total: LOW.** Pure deletion. Net code reduction.

---

## 4. Cross-cutting concerns surfaced by the read

### 4.1 `source_benchmark` threading — ALL paths must propagate it

The verifier currently does NOT receive `source_benchmark`. Every callsite must be updated:

| Site | Currently | After Fix 2 |
|------|-----------|-------------|
| `pipeline.py:1350` (Tier 2/3 verify in answer()) | `verifier.verify(query, answer, input_ids=..., u_token=..., u_dropout=...)` | add `source_benchmark=source_benchmark` |
| `pipeline.py:830-843` (`make_reconsider_deferred_fn` closure) | `verifier.verify(de.question, de.answer)` | add `source_benchmark=de.source_benchmark` |
| `pipeline.py:778-815` (`make_retroverify_fn` closure) | iterates over `EpisodicEntry` objects | thread `entry.source_benchmark` to `verifier.verify` |
| `eval/harness.py` (eval-time verifier calls via batched pipeline) | uniform | thread `source_benchmark=benchmark` already available at iteration |
| `scripts/fit_composite_calibration.py:121-128` (offline calibration) | pools all benchmarks | groupby benchmark, fit each |
| `scripts/run_calibration.py:747-749` (cycle-boundary recalibration cache) | pools u_pre across benchmarks | return `Dict[benchmark, (logits, labels)]` |

Fixes 1, 2, 4, 6, 7, 10, 11, 12 ALL depend on `source_benchmark` being available at the right call site. **This is the keystone refactor**; all per-benchmark fixes assume it.

### 4.2 Hidden coupling: `seed_cold_start.py:161` hardcoded benchmark check

The orchestrator audit flagged that `seed_cold_start.py:161` has a hardcoded benchmark check rather than reading from `TRAINING_BENCHMARKS`. With Fix 9 adding HotpotQA + CommonsenseQA, this hardcoding will silently exclude them from cold-start seeding. **Must be fixed as part of Fix 9.**

### 4.3 Model checkpoint format change ripples to gdrive offload + watchdog

LoRA adapter checkpoints are ~50 MB vs full state dict ~6 GB. The `outputs/full_run/cycle_N/` directory schema changes:

```
Old layout:
  cycle_N/model.pt                     ~6.2 GB (full state dict)
  cycle_N/composite_calibration.json
  cycle_N/conformal_gate.json
  cycle_N/calibrated_thresholds.json
  cycle_N/meta.pkl

New layout (LoRA path):
  cycle_N/adapter_model.bin            ~50 MB (LoRA adapter)
  cycle_N/adapter_config.json          (PEFT metadata)
  cycle_N/composite_calibration.json   (per-benchmark dict now)
  cycle_N/conformal_gate.json          (per-benchmark dict now)
  cycle_N/safety_threshold.json        (NEW — per-benchmark u_pre + safety)
  cycle_N/meta.pkl
  cycle_N/coverage_diagnostic.json     (NEW — per-benchmark coverage stats)
```

Net per-cycle artefact size drops from ~6.3 GB → ~52 MB. Trivial gdrive offload. 11 cycles ≈ 580 MB total checkpoint storage. Watchdog scripts (`scripts/watchdog_cycles_3plus.sh`) need cycle-detection logic verified for the new layout.

### 4.4 Resume-from-cycle compatibility

`scripts/run_experiment.py` `--resume_from_cycle N` currently expects `cycle_{N-1}/model.pt`. With LoRA, the resume must:
1. Load base Qwen 2.5-3B (frozen)
2. `PeftModel.from_pretrained(base, cycle_{N-1}/adapter)` to restore adapter
3. Load per-benchmark composite + conformal + safety dicts from cycle_{N-1}/

Resume logic update is part of Fix 8.

### 4.5 Backward-compat for existing cycle 0-4 artefacts

The 4 cycles we already ran (and cycle 5 partial, now archived) used the OLD architecture. The CAEM_FIX_AUDIT path forward:

- Cycle 0-4 artefacts stay on gdrive as historical evidence of the discovered failure mode
- New trajectory restarts from a fresh cycle 0 under the new architecture
- Thesis Ch5 §sec:disc-threats documents the discovered failure mode + the architectural response
- Old artefacts NOT used in the new trajectory; they're audit trail only

---

## 5. Ambiguities the read surfaced (need user decision)

### 5.1 General-domain mix removal — full removal or keep as ablation?

Two options:
- **Full removal** (current locked decision per Fix 13): cleanest; remove `load_general_data()` entirely; loss reweighting + LoRA's implicit regularization replace it
- **Keep behind a flag** (alternative): retain code path with `cfg.use_general_data_mix=False` default, allowing Phase 1c ablation comparison

**Read recommendation: full removal.** Cleaner code, smaller maintenance surface. The thesis can compare against the historical cycle 0-4 trajectory which DID use the general-domain mix as an implicit ablation. **Confirm.**

### 5.2 Wikidata alias data — local file vs API

For Fix 6 (`alias_overlap`), two paths:
- **Local download** (~150 MB compressed Wikidata alias subset, e.g., from `dbpedia/canonicalize` HF dataset): zero runtime cost after one-time download
- **HF API call per verify** (`refined-fork` or similar): zero local disk but adds network latency to each verify call

**Read recommendation: local file.** Latency-sensitive path. Verifier runs ~10K times per cycle; even 50ms network latency would add ~9 minutes per cycle. **Confirm.**

### 5.3 Model checkpoint size budget

LoRA adapters are ~50 MB. Local rolling-N retention currently keeps cycle_0 (baseline) + last 2 + final = 4 checkpoints. With LoRA: 4 × 50 MB = 200 MB. Trivial; can keep ALL 11 cycle checkpoints locally without rolling deletion.

**Read recommendation: keep all 11 cycle adapters** (~580 MB total). Simplifies resume logic. **Confirm.**

---

## 6. Refactor risk matrix (synthesized across all 5 audits)

| Fix | Risk | Effort (LOC) | Effort (days) | Critical dependencies |
|-----|------|-------------:|--------------:|----------------------|
| 1 — Per-bench conformal gate | MEDIUM | 80 | 0.5 | Existing prototype in `conditional_conformal_ablation.py` |
| 2 — Per-bench composite + verifier API change | **HIGH** | 200 | 1.0 | Keystone for Fix 6, 7, 10, 12; all consumers must update |
| 3 — Loss reweighting + DoReMi + cold-start fallback | HIGH | 150 | 1.0 | `_collect_episodes` → `_mix` insertion point; unit test critical |
| 4 — Multi-modal retention probe | HIGH | 100 | 0.5 | Need TriviaQA + HotpotQA test folds frozen |
| 5 — Coverage diagnostic | MEDIUM | 150 | 0.5 | New module; orchestrator integration |
| 6 — alias_overlap signal | HIGH | 220 | 1.0 | Wikidata data file + verifier schema change |
| 7 — entity_head_consistency | MEDIUM | 80 | 0.3 | Reuses M=3 chains; head-noun extractor needed |
| 8 — LoRA SIL | **CRITICAL** | 280 | 1.5 | PEFT wrapping + conditional save/load + smoke test |
| 9 — Training panel + loaders | MEDIUM | 200 | 1.0 | HotpotQA + CSQA HF loaders; thesis updates |
| 10 — Per-bench prompts + verifier dispatch | HIGH | 500 | 2.0 | Bare-entity expansion scorer is new code |
| 11 — Five-layer deferred guard | LOW | 200 | 0.5 | Defensive plumbing; pre-launch dry-run |
| 12 — Per-bench u_pre T_b + safety | MEDIUM | 110 | 0.5 | Pipeline threading + dict-of-temperatures |
| 13 — Remove general-domain mix | LOW | 30 | 0.2 | Pure deletion |
| **Total** | — | **~2,300** | **~10 days** | — |

10 working days of careful implementation + testing. Add 2-3 days for unit-test scaffolding, smoke-test cycle 0, and bug-fix iterations = **~12-13 days of focused code work** before the trajectory can launch.

---

## 7. Recommended implementation order (dependency-respecting)

```
Day 1-2:  Fix 9   (training panel + loaders) — independent, mostly mechanical
          Fix 13  (remove general-domain mix) — pure deletion
          Fix 11  (deferred guard layers 1-3) — defensive

Day 3-5:  Fix 2   (per-bench composite + verifier API change) — KEYSTONE
          Fix 1   (per-bench conformal gate, depends on Fix 2)
          Fix 12  (per-bench u_pre T_b + safety, depends on Fix 2)
          Fix 11  (layer 4 unit tests)

Day 6-7:  Fix 6   (alias_overlap signal)
          Fix 7   (entity_head_consistency signal)
          Fix 10  (per-bench prompts + verifier dispatch)

Day 8-10: Fix 8   (LoRA SIL primitive — biggest single change)
          Fix 3   (loss reweighting in SIL pool)
          Fix 4   (multi-modal retention probe)
          Fix 5   (coverage diagnostic)

Day 11-12: Smoke test on cycle 0 (small-scale)
          Pre-launch dry-run for Fix 11 layer 5
          Bug-fix iterations
          Commit + push to feature branch

Day 13+:  Cycle 0 launch + per-benchmark cal artifacts
          Cycles 1-10 trajectory + parallel report rewrite
```

---

## 8. Outstanding questions before code starts

1. **Fix 13 (general-domain mix)**: full removal or keep behind a flag? — *Read recommends full removal.*
2. **Fix 6 (alias data)**: local Wikidata file or HF API? — *Read recommends local file.*
3. **Checkpoint retention**: keep all 11 LoRA adapters locally or roll? — *Read recommends keep all (only 580 MB total).*
4. **Cycle 5 partial cycle_5/ directory** (currently at `aborted_cycle5_sil_2026-05-04/`): keep as audit or delete? — *Read recommends keep as audit, no GPU cost to retain.*

---

## 9. Confirmation needed before I start coding

The audit is complete. Key findings:

- **Architecture is implementable cleanly** — no surprises that would force re-design
- **Keystone refactor is Fix 2** (verifier API change to accept `source_benchmark`); all per-benchmark fixes flow from this
- **One hidden coupling caught**: `seed_cold_start.py:161` hardcodes benchmark names (must be fixed as part of Fix 9)
- **Total ~2,300 lines across 13 fixes**; ~12-13 days work
- **Three small ambiguities** flagged in §5; read recommendations stand unless overridden

After your review of this audit, the next step is:
- Update `PRODUCTION_NEXT_SESSION_PLAN.md` with v2 architecture + 13-fix order + parallel report-rewrite phase
- Log a 2026-05-06 entry in `branch_C_log.md` capturing the discovered failure mode + architectural response
- Begin Fix 9 + Fix 13 + Fix 11 layers 1-3 (Day 1-2 work)

**Ready to proceed?**
