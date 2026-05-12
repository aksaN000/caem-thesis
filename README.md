# CAEM — Confidence-Aware Episodic Memory with Self-Improvement

**CSE400 Final Year Thesis — BRAC University**
**Student:** Aksan Gony Alif
**Last updated:** 2026-05-12

---

## Current Status

`step_7_main` 10-cycle trajectory is in flight. Cycles 1 and 2 have closed cleanly; cycle 3 is mid-run.

| Phase | Status |
|---|---|
| Core architecture (8-stage per-query pipeline + cycle-boundary loop) | done |
| Phase 1c restructure (P0–P4): conformal gate retired, per-benchmark composite with shrinkage prior, alias_overlap + entity_head_consistency + h_norm dropped, FEVER NEI directional fix | done |
| Phase 1d (Option 4): `store_threshold` lowered from 0.65 to 0.60; cycle-0 rescored | done |
| Patch 2026-05-10: canonical composite path no longer overwritten at cycle close; resume reload prefers latest per-cycle composite | done |
| Patch 2026-05-11: per-bench T_b refit synthesises combined records on cache-hit; Thm 3 precondition tightened in Ch4 §11 | done |
| Patch 2026-05-11 (perf): batched retroverify + batched deferred-buffer reconsideration in `store.retroverify` / `deferred.reconsider` | done |
| Cycles 1 + 2 SIL fine-tunes | passed retention; adapters saved |
| Cycle 3 SIL | in flight (retention boundary-crossing observed: 2/3 attempts pass, 1/3 fail) |
| External baselines B1–B7 + significance panel | queued post step_7_main |
| Phase 4 receipts (theorem aggregators, verifier-accuracy trajectory, Tier-1 amortisation diagnostic) | queued post step_7_main |
| Ch5 / Ch6 artefact generation | queued post Phase 4 |

Empirical trajectory so far (held-out, n=300 per benchmark):

| benchmark | EM cycle 0 | EM cycle 1 | EM cycle 2 | CHM cycle 0 | CHM cycle 1 | CHM cycle 2 |
|---|---|---|---|---|---|---|
| FEVER | 0.4500 | 0.5433 | 0.5200 | 0.1633 | 0.1113 | 0.1125 |
| TriviaQA | 0.4067 | 0.4633 | 0.4433 | 0.1646 | 0.1150 | 0.1150 |
| CommonsenseQA | 0.6067 | 0.6300 | 0.6333 | 0.1317 | 0.1229 | 0.1179 |
| TruthfulQA | 0.3300 | 0.3100 | 0.3900 | 0.1596 | 0.1383 | 0.1258 |
| StrategyQA | 0.6067 | 0.6300 | 0.6133 | 0.1608 | 0.1225 | 0.1300 |
| **pooled** | **0.4800** | **0.5153** | **0.5200** | **0.1560** | **0.1220** | **0.1202** |

---

## What CAEM Is

CAEM is an architectural hallucination-reduction system for open-domain question answering, built on **Qwen-2.5-3B-Instruct** (3.1 B parameters, decoder-only, bf16). It operates per-query through an 8-stage pipeline and per-cycle through a self-improvement loop. Five components carry the value proposition:

1. **Episodic memory** — FAISS-backed store of verified question-answer-chain triples with per-entry calibrated `u_stored` confidence; serves familiar queries through a memory-direct tier at retrieval-only cost.
2. **Three-tier router with per-benchmark safety override** — combined `S = 0.70 · sim + 0.30 · u_stored_neighbour` dispatch over {memory-direct, zero-shot, RAG}. A per-benchmark safety floor `u_pre,min_b` forces RAG when the model's pre-routing confidence falls below the benchmark-specific threshold.
3. **Nine-signal verifier ensemble** — three internal-calibration signals (`u_token`, `u_dropout`, `u_internal`), two sample-set agreement signals (`s_avg`, `p_entail`), three external-grounding signals (`p_ground_max`, `p_ground_mean`, length-gated `p_ground_atomic`), and one question-answer relevance signal (`q_a_relevance`).
4. **Per-benchmark calibrated probability composite + fixed-threshold storage gate** — per-signal isotonic regression maps each raw signal to a calibrated probability of correctness, shrunk toward a pooled prior at α = 0.6 to prevent per-benchmark overfitting. A per-benchmark Cherian L2 logistic boost layer aggregates the calibrated signals into `u_stored`. The storage gate is a fixed threshold on `u_stored`: `≥ 0.60 → STORE`, `≥ 0.45 → DEFERRED`, otherwise `ABSTAIN` or `DISCARD` (split by a passage-grounding floor).
5. **Cycle-boundary self-improvement loop** — at every cycle boundary: LoRA SIL fine-tune (r=32, α=64, all-linear targets) with no explicit L2 anchor (LoRA's parameter budget is the implicit anchor); score the calibration fold under the post-SIL model; refit per-benchmark T_b and per-signal isotonic curves + Cherian boost weights; reload the verifier; retroactively re-verify every memory entry under the recalibrated verifier; reconsider buffered candidates; ingest a fresh stream chunk; held-out evaluation; save artefacts.

---

## Locked Configuration

Every value pinned by `caem/config.py` at the trajectory launch. Reproducible from a single git commit.

### Gate + composite

| parameter | value | role |
|---|---|---|
| `store_threshold` | **0.60** | STORE cut on calibrated probability |
| `defer_threshold` | **0.45** | DEFERRED cut on calibrated probability |
| `retroverify_prune_threshold` | 0.50 | drop entry from memory at cycle boundary if recomputed `u_stored` falls below this |
| `min_u_stored_for_training` | 0.75 | training-pool floor (strictly above `store_threshold` so borderline entries do not contaminate the gradient signal) |
| abstention p_ground_max floor | 0.20 | splits ABSTAIN (no evidence) from DISCARD (conflicting evidence) when `u_stored < 0.45` |
| confabulation early-exit | `u_internal ≥ 0.70 ∧ p_ground_max ≤ 0.20` | precedence over composite at query time; disabled at retroverify time |
| shrinkage_alpha (P3a) | 0.60 | per-benchmark isotonic curves are shrunk toward pooled prior |
| Cherian boost L2 strength `C` | 0.01 | 25-variant sweep selection at cycle-zero validation |
| active signals in composite | 9 | u_token, u_dropout, u_internal, s_avg, p_entail, p_ground_max, p_ground_mean, p_ground_atomic, q_a_relevance |
| retired from composite | 3 | h_norm (Phase 1d), alias_overlap + entity_head_consistency (Phase 1c P1) |

### Router + safety

| parameter | value | role |
|---|---|---|
| `routing_lambda` | 0.70 | combined-score weight on similarity vs neighbour `u_stored` |
| `tier1_combined_threshold` | 0.90 | combined score above this routes to memory-direct |
| `tier2_similarity_threshold` | 0.75 | similarity above this routes to zero-shot |
| `safety_u_pre_min` (pooled fallback) | 0.38 | force RAG when pre-routing confidence below per-benchmark floor |
| `safety_u_pre_min_per_benchmark` | data-driven from cal fold (Fix 12) | per-benchmark override; replaces pooled fallback when present |
| `temperature_scalar_per_benchmark` | data-driven from cal fold (Fix 12) | per-benchmark T_b for u_pre calibration |
| `M` chains × `T` | 3 × 0.7 | sample-set agreement signals |
| `K` dropout passes | 5 | MC-Dropout `u_dropout` |

### SIL + retention

| parameter | value | role |
|---|---|---|
| SIL primitive | LoRA r=32, α=64, all-linear targets | adapter-only fine-tune; no L2 anchor |
| `forgetting_tolerance` | 0.93 | any retention probe below this ratio of pristine triggers cycle abort and adapter rollback |
| `retention_probes` | MMLU, TriviaQA-test, CommonsenseQA-test | multi-modal probe spans bounded-label and open-text retention |
| `retention_probe_n` | 200 | samples per probe |
| `batch_size`, `grad_accum_steps` | 4, 4 (effective batch 16) | SIL fine-tune |
| `epochs_per_cycle` | 3 | SIL fine-tune |
| `deferred_buffer_ttl_cycles` | 2 | TTL for deferred candidates before they are dropped |

### Benchmark panel

| panel | benchmarks | role |
|---|---|---|
| training | FEVER, TriviaQA, CommonsenseQA | calibration fold + stream-chunk SIL pool |
| transfer | TruthfulQA, StrategyQA | held-out evaluation only; never enter calibration or SIL pool |
| stream-chunk sizes per cycle | FEVER 2000, TriviaQA 2000, CSQA 700 (= 4700 / cycle) | gold-disjoint per-cycle slice |
| held-out eval per cycle | 300 per benchmark × 5 benchmarks = 1500 | trajectory measurement |
| calibration fold | 500 per training benchmark × 3 = 1500 | refit composite + T_b each cycle |
| cycle count | 10 cycles + cycle 0 calibration = 11 total | early-stop equilibrium gate may end sooner |

---

## Per-Query Pipeline

```
Stage 1   pre-routing confidence u_pre   (encoder forward pass)
            ↓ scaled by per-benchmark T_b
Stage 2   encode query + nearest-memory lookup  (SBERT + FAISS)
Stage 3   three-tier dispatch
            ├── safety override: u_pre < safety_u_pre_min_b → force Tier 3
            ├── Tier 1: combined score S ≥ 0.90 → memory-direct (skip verifier)
            ├── Tier 2: similarity > 0.75 and not Tier 1 → zero-shot generation
            └── Tier 3: otherwise → retrieval-augmented generation
Stage 4   tier execution (Tier 2 / Tier 3 only)
            ↓ generates answer + reasoning chain
Stage 5   nine-signal verifier ensemble
            ↓ per-benchmark dispatch via source_benchmark token
Stage 6   per-benchmark composite + fixed-threshold gate
            ├── STORE       u_stored ≥ 0.60
            ├── DEFERRED    0.45 ≤ u_stored < 0.60 → buffered for reconsideration
            ├── ABSTAIN     u_stored < 0.45 AND p_ground_max < 0.20  → no evidence
            └── DISCARD     u_stored < 0.45 AND p_ground_max ≥ 0.20  → conflicting evidence
            (confabulation early-exit at query time only; retroverify uses
             is_query_time=False so the gate's u_internal conjunct does not
             over-prune memorised reasoning chains)
Stage 7   serve answer or ABSTAIN refusal to user
Stage 8   per-query housekeeping (memory write if STORE; deferred buffer if DEFERRED)
```

---

## Cycle Boundary

```
Step 1     LoRA SIL fine-tune
              ├── training pool: stored episodes with u_stored ≥ 0.75
              ├── per-bench share cap (pool_max_share) prevents monoculture
              ├── retention probe pre + post on (MMLU, TQA-test, CSQA-test)
              └── ABORT + adapter rollback if any probe drops < 0.93 of pristine
                    (memory pool survives the rollback)

Step 2.1   score calibration fold under post-SIL model
              → outputs/full_run/cycle_{N}/calibration/{bench}_cycle{N}.json
              (per-benchmark per-sample signals + EM)

Step 2.2   refit per-benchmark temperature T_b (ECE-min) + safety floor u_pre,min_b
              → outputs/full_run/cycle_{N}/calibration/calibrated_config_cycle{N}.json

Step 2.3   refit per-signal isotonic curves + per-benchmark Cherian L2 boost weights
              (shrinkage prior α = 0.6, boost_C = 0.01)
              → outputs/full_run/cycle_{N}/composite_calibration.json

Step 2.4   verifier reload from the freshly-written cycle composite

Step 2.5   retroactive re-verification under the recalibrated verifier
              → batched at N=8 (Patch 2026-05-11)
              → prune entries whose recomputed u_stored < 0.50
              → update u_stored in either direction for kept entries
              → outputs/full_run/retroverify_cycle{N}.json

Step 3     save retroverify stats

Step 4     stream chunk under cycle-N calibration
              (FEVER 2000 + TriviaQA 2000 + CSQA 700; store_to_memory=True)
              → outputs/full_run/eval/{bench}_cycle{N}_streamchunk.json (Step 4b snapshot)

Step 5     held-out evaluation (300 per benchmark × 5 benchmarks; store_to_memory=False)
              → outputs/full_run/eval/{bench}_cycle{N}.json

Step 6 / 6b  save memory + deferred buffer checkpoints
              → memory_store_cycle_{N}.{faiss,meta}
              → deferred_buffer_cycle_{N}.pkl

Step 6c    gdrive offload of the whole cycle directory
              → gdrive:caem-phase1a/v2_1_phase1d/full_run/cycle_{N}/
```

Order is load-bearing: recalibrate before retroverify, so retroverify reads memory through the freshly fitted isotonic curves. The previous order over-pruned EM-correct cold-seed entries by scoring memory through stale Cycle-0 calibration.

---

## Repository Map

```text
caem/
  config.py                              # all hyperparameters; CAEMConfig dataclass
  pipeline.py                            # per-query pipeline + cycle orchestration
                                         #   includes batched retroverify + reconsider closures
  pipeline_batch.py                      # verify-batch pooled variant
  ablation/                              # ablation registry + variant overrides
  benchmark_splits.py                    # Gap-5 disjoint per-cycle stream-chunk allocation
  confidence/pre_routing.py              # u_pre estimation
  memory/
    encoder.py                           # SBERT wrapper
    entry.py                             # EpisodicEntry dataclass (preserves source_benchmark)
    store.py                             # FAISS IndexIDMap + retroverify (batched at N=8)
    deferred.py                          # bounded deferral buffer + reconsider (batched at N=8)
  retrieval/rag.py                       # DPR + FAISS passage retrieval
  routing/router.py                      # combined-score + per-benchmark safety-override dispatch
  training/self_improvement.py           # LoRA SIL fine-tune + multi-modal retention guard
  verification/
    verifier.py                          # nine-signal ensemble + per-benchmark composite dispatch
    cal_prob_composite.py                # per-signal isotonic + Cherian L2 boost + shrinkage prior
    multichoice_scorer.py                # ARC/CSQA letter→option-text substitution for NLI
    answer_canonicalizer.py              # uniform "Answer: <X>." claim form across all signals
    directional_p_ground.py              # FEVER NEI directional grounding (Phase 1c P2)
    minicheck.py                         # MiniCheck NLI judge (default for short premises)
    qwen_judge.py                        # Frozen Qwen NLI judge (length-dispatched for long premises)
    adaptive_nli_judge.py                # length-dispatch wrapper

eval/
  baselines.py                           # B1–B7 baseline runners
  benchmarks.py                          # benchmark loaders for FEVER, TriviaQA, CSQA,
                                         #   TruthfulQA, StrategyQA + retention probes
  harness.py                             # eval driver (per-cycle JSON output, batched)
  metrics.py                             # EM, F1, CHM (9-subtype), CES, McNemar, BCa bootstrap,
                                         #   AUROC, Brier, reliability bins
  reporting.py                           # table + figure aggregation
                                         #   per_sample_signals.jsonl for figure 5.2

scripts/
  run_experiment.py                      # main 10-cycle CAEM driver (resumable; Patch 2026-05-10
                                         #   keeps canonical composite path untouched)
  run_phase1a.sh                         # canonical 30+-step Phase 1a runbook (auto-resume)
  run_baseline.py                        # B1–B7 baseline panel
  run_ablation.py / run_cyclic_ablation.py
  fit_composite_calibration.py           # cycle-zero + per-cycle composite fit
  run_calibration.py                     # per-cycle T_b refit (Patch 2026-05-11)
  validate_composite_weights.py          # cycle-zero validation gate
  sweep_composite_variants.py            # pre-launch composite sweep
  rescore_eval_with_fitted_gate.py       # cycle-0 rescore under locked gate
  build_passage_index.py                 # 21M-passage FAISS index build
  seed_cold_start.py                     # cold-start memory seeding
  baseline_sig_tests.py                  # paired McNemar + BCa bootstrap + Holm correction
  run_purity_validation.py               # theorem 1 + 2 + 4 + 6 receipts
  theorem_receipts.py                    # theorem 5 + 7 + corollaries receipts
  aggregate_calibration_trajectory.py    # per-cycle T + threshold + decision trajectory
  phase4_artifacts.py                    # Cohen's d table, precision cliff, atomic scope,
                                         #   prompt compliance, summary table
  generate_session5_artifacts.py         # cycle-zero audit + claim-evidence map
  compare_prompt_design.py
  make_figures.py                        # Ch5 figures 5.1–5.7
  make_tables.py
  render_production_samples.py
  caem_demo_server.py                    # production demo server (Cloudflare-exposed)

tests/
  test_*.py                              # unit + integration tests; 377 v2-fix tests pass

docs/
  PRODUCTION_RUNBOOK.md                  # end-to-end production deployment + per-cycle refresh
  PANEL_DEMO_SCRIPT.md                   # defense-day demo question set
  DEMO_QUICKSTART.md                     # operator cheat sheet
  metrics-audit.md                       # metric definitions + computation map

thesis_report/                           # active thesis (Ch1–6 + Appendix A–G + bibliography)
pre thesis 1 report/                     # earlier proposal (pre-implementation); preserved
                                         #   as the v1-era theorem reference
presentation/
  caem_supervisor.tex                    # supervisor presentation (current architecture)

branch_C.md                              # Branch C architecture lock document
branch_C_log.md                          # dated implementation diary (load-bearing audit trail)
PRODUCTION_NEXT_SESSION_PLAN.md          # active session plan (v2.1)
NEXT_SESSION_PLAN.md                     # SUPERSEDED — v1-era; do not act on
hyperparameter-reference.md              # three-category hyperparameter taxonomy
```

---

## Benchmarks and Metrics

### Active panel

**Training panel** (3 benchmarks; samples flow into calibration / SIL pool / stream chunk):

| benchmark | source | answer surface | metric |
|---|---|---|---|
| FEVER | `lucadiliello/fever` | 3-way label (SUPPORTS / REFUTES / NEI) | label EM |
| TriviaQA | `mandarjoshi/trivia_qa` (rc.nocontext) | bare entity | EM with alias-aware matching, token F1 |
| CommonsenseQA | `tau/commonsense_qa` | 5-choice MCQ (A–E) | letter EM with option-text substitution at NLI |

**Transfer panel** (2 benchmarks; held-out, never enter calibration or SIL):

| benchmark | source | answer surface | metric |
|---|---|---|---|
| TruthfulQA | `truthfulqa/truthful_qa` (generation) | open-text | ROUGE-L threshold + alias match |
| StrategyQA | `ChilleD/StrategyQA` (test split) | yes / no | label EM |

Dropped from the active trajectory but retained as loaders for ablation reproduction: HotpotQA, Natural Questions, ARC-Challenge, ASQA. Their loaders live in `eval/benchmarks.py`; the rationale for removal is the cycle-0 precondition audit (HotpotQA + NQ failed the Bayes-floor inequality; ARC was redundant with CSQA; ASQA's long-form output makes EM the wrong metric).

### Retention probes

Multi-modal forgetting guard runs before and after every SIL fine-tune:

| probe | source | n | role |
|---|---|---|---|
| MMLU | `cais/mmlu` (200 dev questions) | 200 | bounded-label MCQ retention |
| TriviaQA-test | TriviaQA test split | 200 | open-text factoid retention |
| CommonsenseQA-test | CSQA test split | 200 | 5-choice commonsense MCQ retention |

Cycle aborts with adapter rollback if **any** probe drops below 0.93 of its pristine value. Memory pool survives the rollback (asymmetric rollback).

### Headline metrics

| metric | definition |
|---|---|
| EM | exact match against gold answers, alias-aware where applicable |
| F1 | token-level F1 against gold |
| CHM | composite hallucination metric: equal-weighted mean over 8 measurable failure-mode subtypes (confident confabulation, factual fabrication, logical fabrication, off-topic, defensive evasion, template leakage, false refusal, length-padded over-generation). The 9th subtype (factual contradiction) is structurally zero under MiniCheck and excluded from the denominator. |
| CES | composite evaluation score: geometric mean over five axes (accuracy, epistemic quality 1−CHM, retention, calibration, verification reliability). Geometric mean penalises collapse on any single axis. |
| storage-bucket precision | `Pr(EM = 1 \| decision = STORE)` per benchmark per cycle |
| AUROC | discrimination of `u_stored` between EM=1 and EM=0 samples |
| Brier, ECE | calibration of `u_stored` against EM |

---

## How to Run

### Canonical Phase 1a pipeline

```bash
bash run_phase1a.sh
```

End-to-end: passage-index build → calibration-pair build → cold-start memory seeding → cycle-zero composite fit → validation gate → step_7_main 10-cycle run → external baselines → diagnostics → Phase 4 artefacts. Idempotent: every step checks a sentinel and skips if already done. Safe to re-run after any halt.

### Resume after halt

`run_phase1a.sh` auto-detects the last completed cycle from `outputs/full_run/memory_store_cycle_*.faiss` and passes `--resume_from_cycle N+1` to `scripts/run_experiment.py`. No manual flags needed.

### Test suite (no GPU, pure-python)

```bash
python -m pytest tests/ -v
```

Focused modules:

```bash
python -m pytest tests/test_self_improvement.py -v
python -m pytest tests/test_pipeline_batch_equivalence.py
python -m pytest tests/test_calibration_batch_equivalence.py
python -m pytest tests/test_cal_prob_composite.py
```

### Cycle 0 calibration refit (if needed)

```bash
python scripts/fit_composite_calibration.py \
  --calib_jsons outputs/cycle_0/calibration/*.json \
  --output_json outputs/cycle_0/composite_calibration.json \
  --shrinkage_alpha 0.6 --cherian_boost --boost_C 0.01

python scripts/validate_composite_weights.py \
  --output outputs/cycle_0/weight_validation.json
```

The conformal-gate fit script (`fit_conformal_gate.py`) is no longer in the canonical chain; Phase 1c replaced its output with the fixed thresholds from `CAEMConfig`.

---

## Main Outputs

`outputs/full_run/` (written by `step_7_main`):

```
cycle_0/                                  # cycle-0 calibration + baseline
  calibration/calibration_fold_samples.json
  composite_calibration.json              # canonical composite (frozen; Patch 2026-05-10)
  calibrated_thresholds.json
  weight_validation.json

cycle_{1..10}/                            # per-cycle artefacts
  adapter/                                # LoRA adapter (~50 MB; replaces v1 model.pt)
  calibration/{bench}_cycle{N}.json       # per-bench post-SIL cal-fold per-sample signals
  calibration/calibrated_config_cycle{N}.json   # per-bench T_b + safety floors
  composite_calibration.json              # per-cycle refit (per-benchmark)
  coverage_diagnostic.json                # admission rates + pool entropy + adapter SVD
  meta.pkl
  retroverify_cycle{N}.json               # total / updated / pruned + fine-tune stats

retroverify_cycle{N}.json                 # top-level mirror of cycle_{N}/retroverify
memory_store_cycle_{N}.{faiss,meta}       # per-cycle memory snapshot
deferred_buffer_cycle_{N}.pkl

eval/                                     # per-cycle per-bench evaluation
  {bench}_cycle{N}.json                   # held-out eval (n=300; store_to_memory=False)
  {bench}_cycle{N}_streamchunk.json       # Step 4b snapshot of stream-chunk pass (with stores)

retention_baseline.json                   # pristine probe scores
mmlu_baseline.json                        # pristine MMLU
run_complete.json                         # written when trajectory finishes
experiment_summary.csv                    # flat per-cycle aggregator (written once at trajectory close)
```

`outputs/phase4/` (post-trajectory):

```
verifier_trajectory.csv                   # AUROC + Brier + ECE + STORE-EM per cycle per bench
tier1_amortisation.csv                    # H5 Tier-1 hit-rate diagnostic (custom eval set)
cohen_d_table.csv
precision_cliff.png / .pdf
fig5_{1..7}_*.{png,pdf}                   # Ch5 figures
theorem_receipts/
  receipt_envelope_fit.json               # Thm 5 geometric envelope fit
  receipt_eps_arch.json                   # Thm 7 four-gate joint FNR
  receipt_gap_decay.json                  # Cor. convergence-rate exp fit
  receipt_corpus_floor.json               # Cor. corpus-floor proxy
  receipt_self_correction.json            # Cor. self-correction survival distribution
```

---

## Backups

**Per-cycle gdrive offload** (Step 6c, automatic):

```
gdrive:caem-phase1a/v2_1_phase1d/full_run/cycle_{N}/
```

Mirrors the full per-cycle directory: adapter, calibration, eval_transfer (held-out evals), eval_stream_chunk (stream-chunk snapshots), composite + retroverify + memory + deferred buffer. Triggered by `CAEM_GDRIVE_OFFLOAD=1` env var. Failures are non-fatal; local artefacts are the on-disk guarantee.

**Pre-Step-7 snapshot** (one-time, HuggingFace Hub):

```
aksaN000/caem-passage-index-21m:pre_main_snapshot/c5b8066/
```

Contains the 64 GB FAISS passage index plus the cycle-0 baseline (cold-start memory, calibration fold, composite, conformal_gate.json from the pre-Phase-1c era, eval scoring). Recovery procedure if the rented instance dies:

```bash
hf download aksaN000/caem-passage-index-21m \
  --include 'passages.*' 'pre_main_snapshot/*'
bash run_phase1a.sh
```

The runner picks up from the highest-cycle gdrive-resident artefact set.

---

## Documentation

| document | purpose |
|---|---|
| `run_phase1a.sh` | canonical 30+-step Phase 1a runbook |
| `docs/PRODUCTION_RUNBOOK.md` | end-to-end production deployment + per-cycle refresh recipe |
| `PRODUCTION_NEXT_SESSION_PLAN.md` | active session plan (v2.1) |
| `branch_C.md` | architecture-lock document |
| `branch_C_log.md` | dated implementation diary (audit trail) |
| `hyperparameter-reference.md` | three-category hyperparameter taxonomy |
| `thesis_report/main.tex` | Ch 1–6 + Appendix A–G thesis manuscript |
| `thesis_report/bibliography/references.bib` | citation database |
| `presentation/caem_supervisor.tex` | supervisor presentation slides |

---

## Reproducibility

- Python 3.12 + `/venv/main` (PyTorch 2.10 + CUDA 13.0, transformers, sentence-transformers, faiss-cpu, scikit-learn, scipy, bitsandbytes, peft)
- Single recorded seed propagates through generator sampling, FAISS, MC-dropout
- Every threshold, fit-time procedure, prompt template, and panel composition pinned to one git commit before the trajectory launched
- All decision artefacts on disk: every $\tau$, $T_b$, isotonic curve, weight-validation verdict, sweep entry, retention probe lives at a registered path under `outputs/`
- Calibration fold, purity fold, seed pool, stream chunks across cycles, held-out eval, and retention probes are all content-hash disjoint per benchmark (Gap-5 isolation)
- Hardware: rented Vast.ai RTX 5090 (32 GiB physical, ~31 GiB usable); OOM mitigation in `caem/training/self_improvement.py` and `run_phase1a.sh` keeps peak VRAM in envelope

---

## Quick-Reference Decision Map

| question | answer | source |
|---|---|---|
| Which model? | Qwen-2.5-3B-Instruct, bf16 | `caem/model_loader.py` |
| Which NLI judge? | MiniCheck (default for short premises) + Frozen Qwen NLI (length-dispatched for long premises) | `caem/verification/adaptive_nli_judge.py` |
| Signal count in composite? | 9 (3 internal + 2 sample-set + 3 grounding + 1 q_a_relevance) | `outputs/cycle_0/composite_calibration.json :: metadata.signals` |
| Composite type? | Per-benchmark isotonic with shrinkage prior + per-benchmark Cherian L2 logistic boost | `caem/verification/cal_prob_composite.py` |
| Storage gate type? | Fixed threshold on calibrated probability at 0.60 / 0.45 | `caem/config.py:store_threshold, defer_threshold` |
| Cycle-boundary recalibration order? | SIL fine-tune → cal-fold scoring → T_b refit → composite refit → verifier reload → retroverify | `scripts/run_experiment.py` Step 1 + Steps 2.1–2.5 |
| Early-exit at retroverify? | Disabled via `is_query_time=False` to avoid over-pruning memorised reasoning chains | `caem/verification/verifier.py:verify` |
| Cold-start size? | ~3000 verified-correct episodes seeded from training-bench gold pools | `outputs/cold_start_memory/memory_store.{faiss,meta}` |
| Retention floor? | ρ_min = 0.93 on any of (MMLU, TriviaQA-test, CommonsenseQA-test); cycle aborts and adapter rolls back; memory preserved | `caem/training/self_improvement.py` + `caem/config.py:forgetting_tolerance, retention_probes` |
| SIL primitive? | LoRA r=32, α=64, all-linear target modules; no explicit L2 anchor on LoRA path | `caem/config.py:lora_r, lora_alpha, lora_target_modules` |
| Stream-chunk panel composition? | 2000 FEVER + 2000 TriviaQA + 700 CommonsenseQA = 4700 per cycle | `caem/benchmark_splits.py` |
| Held-out evaluation panel? | 300 per benchmark × 5 benchmarks (3 training + 2 transfer) = 1500 per cycle | `caem/config.py:n_eval_questions`, run command |

---

## License and Citation

Research project for academic submission (CSE400, BRAC University, Fall 2026 cohort). Citation pending thesis acceptance.

---

## Acknowledgements

Literature foundation documented in Ch 2 (Literature Review). The architectural lineage owes to:

- Mohri and Hashimoto (2024) — conformal factuality at the LM claim level
- Cherian, Gibbs, and Candès (2024) — boosted logistic aggregation over per-claim scores
- Yadkori et al. (2024) — conformal abstention for LLM hallucination
- Farquhar et al. (2024) and Kuhn et al. (2023) — semantic entropy
- Tang et al. (2024) — MiniCheck claim-support verification
- Kirkpatrick et al. (2017) — elastic weight consolidation foundation for the L2 anchor (retained in v1; removed on the LoRA path in v2.1)
- Hu et al. (2021) and Biderman et al. (2024) — LoRA + BetterLoRA hyperparameter sweep
- Zelikman et al. (2022) — STaR self-improvement loop
- Hosseini et al. (2024) — V-STaR verifier-guided self-training
- Song et al. (2024), Fu et al. (2025), Das et al. (2025), Huang-Block-Foster (2025), Lang-Vijayaraghavan-Sontag (2024) — theoretical foundations for verifier-filtered self-training

Full citations and bibliography entries in `thesis_report/bibliography/references.bib`.
