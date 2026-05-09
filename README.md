# CAEM — Confidence-Aware Episodic Memory with Self-Improvement

**CSE400 Final Year Thesis — BRAC University**
**Student:** Aksan Gony Alif
**Last updated:** 2026-04-27

---

## Current Status (2026-04-27)

Implementation complete. Cycle 0 calibration locked. **`step_7_main` 10-cycle main run is in flight** — Cycle 1 SIL fine-tune just completed at 07:55 UTC with retention ρ=1.0000; Cycle 1 cycle-boundary recalibration (the wired-up Step 2.1–2.5) is currently scoring the calibration fold under the post-SIL model.

| Phase | Status |
|---|---|
| Core architecture (8-stage pipeline + cycle-boundary block) | ✅ |
| Phase 2 architectural patches (cal-prob composite, conformal split-CP gate, KLE, length-gated atomic, prompt fixes, retired CoVe) | ✅ |
| Cold-start memory seeded + purged (260 verified-correct episodes) | ✅ |
| Cycle 0 fits: composite_calibration.json, conformal_gate.json, weight_validation.json (PASS) | ✅ |
| Cycle 0 evaluation rescored under locked configuration | ✅ |
| Pre-Step-7 HF Hub snapshot uploaded (`aksaN000/caem-passage-index-21m:pre_main_snapshot/`) | ✅ |
| **`step_7_main`** 10-cycle SIL trajectory | 🔄 Cycle 1 in progress |
| External baselines (B1–B7) + significance tests | ⏳ pending step_7_main completion |
| Diagnostics (purity validation, retention, signal correlations) | ⏳ pending |
| Phase 4 — Ch5/Ch6 auto-generated artefacts (12 stub tables → real data) | ⏳ pending step_7_main + baselines |

**Five OOM mitigation patches landed today** (2026-04-27): bf16 anchor on CPU + PCIe stream, `expandable_segments:True` allocator, `torch.compiler.set_stance("force_eager")` for SIL backward, `batch_size=4` × `grad_accum_steps=4` (effective batch unchanged at 16). **Two architectural fixes today**: (a) early-exit gated by `is_query_time` so retroverify no longer over-prunes EM-correct stored entries on stale calibration; (b) cycle-boundary recalibration wired up properly — every cycle now scores its calibration fold under the post-SIL model, refits T + isotonic curves + conformal τ, reloads the verifier, then runs retroverify against the freshly-recalibrated thresholds.

---

## Phase 1c update (2026-05-09)

The conformal split-CP storage gate was replaced by a fixed-threshold gate on the calibrated composite probability. `CAEMConfig.store_threshold` (default `0.60`), `CAEMConfig.defer_threshold`, and `CAEMConfig.min_u_stored_for_training` are the active thresholds; the per-cycle composite refit (`run_per_cycle_composite_refit` in `scripts/run_experiment.py`) keeps `u_stored` calibrated under SIL-induced drift, so a fixed threshold on the calibrated probability stays meaningful across cycles. The conformal scripts (`fit_conformal_gate.py`, `recalibrate_conformal_at_cycle.py`, `calibrate_thresholds.py`) and the `conformal_gate.json` artefact are no longer in the active runner path. Other Phase 1c changes shipped together: `alias_overlap` and `entity_head_consistency` were dropped from the composite (P1); FEVER NEI directional grounding was patched (P2); a shrinkage prior toward the pooled fit was added to the per-bench composite (P3a); a bench-agnostic pool-share cap was introduced (P3b); resume detection switched to per-cycle artefact presence. Sections of this README written before Phase 1c may still describe the conformal gate as if active; treat the bullet above as the source of truth for storage-decision behaviour.

---

## What CAEM Is

CAEM is an architectural hallucination-reduction system for open-domain question answering, built on **Qwen-2.5-3B-Instruct** (3.1B params, decoder-only, bf16). The architecture operates per-query through an 8-stage pipeline and per-cycle through a cycle-boundary update loop. Five components carry the value proposition:

1. **Episodic memory** — a FAISS-backed store of verified question-answer-chain triples with per-entry calibrated `u_stored` confidence; serves familiar queries through a memory-direct path at retrieval-only cost.
2. **Three-tier router with safety override** — combined `S = λ·sim + (1−λ)·u_stored` dispatch (λ=0.70) with a hard `u_pre < φ_pg = 0.20` override that forces the retrieval-augmented tier on low-pre-routing-confidence queries.
3. **Ten-signal verifier ensemble** — three internal-calibration signals (token log-prob, MC-Dropout variance, derived `u_internal`), three sample-set agreement signals (`s_avg`, `p_entail`, `h_norm`), three external-grounding signals (`p_ground_max`, `p_ground_mean`, length-gated `p_ground_atomic`), plus Branch-C question-answer relevance `q_a_rel`.
4. **Calibrated-probability composite + conformal split-CP storage gate** — per-signal isotonic regression maps each raw signal to a calibrated probability of correctness; an L2-regularised logistic boost layer aggregates the per-signal log-odds; the conformal gate at `α_store=0.05` and `α_defer=0.40` produces τ thresholds with a precision-floor guarantee under exchangeability.
5. **Cycle-boundary self-improvement loop** — at every cycle: SIL fine-tune (8-bit AdamW + L2 anchor + retention guard at ρ_min=0.93) → score calibration fold under post-SIL model → re-fit T + isotonic + conformal τ → reload verifier → retroactive re-verify under the freshly-recalibrated verifier → ingest fresh queries → eval → save artefacts.

---

## Locked Configuration (Cycle 0 Calibration)

| Parameter | Value | Source |
|---|---|---|
| Backbone | Qwen-2.5-3B-Instruct | locked at experiment start |
| Sentence encoder | `sentence-transformers/all-mpnet-base-v2` (768-dim) | locked at experiment start |
| NLI judge | MiniCheck-Flan-T5-Large (`lytang/MiniCheck-Flan-T5-Large`) | step_5_5; long-hypothesis Qwen-judge ablated 2026-04-26 |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | locked at experiment start |
| Conformal miscoverage levels | `α_store = 0.05`, `α_defer = 0.40` | sweep variant V_α=0.050,C=0.010 selected at cycle-zero validation gate |
| Storage threshold | `τ_store = 0.6676` | conformal fit, `outputs/cycle_0/conformal_gate.json` |
| Defer threshold | `τ_defer = 0.5207` | same |
| Training threshold | `τ_train = 0.5784` | conformal fit |
| Boost-layer L2 strength | `C = 0.01` | sweep selection |
| Boost intercept | `+1.099` | composite fit |
| Routing weight | `λ = 0.70` | locked |
| Pre-routing safety floor | `φ_pg = 0.20` | locked |
| Retention floor | `ρ_min = 0.93` | locked |
| L2 anchor strength | `λ_anc = 0.01` | locked (CPU-resident anchor, PCIe-streamed at L2-step) |
| Cycle count | 10 (early-stop equilibrium gate may end sooner) | locked |
| Per-cycle ingest | 3000 q × 3 training benchmarks | locked |
| Per-cycle eval | 500 q × 7 benchmarks | locked |
| EMA factor for τ refit | `α_ema = 0.7` | locked |

---

## Per-Query Pipeline

```
Stage 1   pre-routing confidence u_pre  (encoder forward + cheap fusion)
Stage 2   encode + nearest-memory lookup (SBERT + FAISS)
Stage 3   three-tier dispatch
            ├── safety override: u_pre < φ_pg → force Tier 3
            ├── Tier 1: combined score S above threshold → memory-direct
            ├── Tier 2: similarity-based → zero-shot generation
            └── Tier 3: low confidence → retrieval-augmented generation
Stage 4   generation (Tier 2 / Tier 3 only)
Stage 5   ten-signal verifier ensemble (parallel signal computation)
Stage 6   calibrated-probability composite → fixed-threshold gate on the
            calibrated probability (Phase 1c; see Phase 1c update above)
            → STORE / DEFER / ABSTAIN / DISCARD
            (confabulation early-exit fires at query time only;
             retroverify uses is_query_time=False to skip it)
Stage 7   serve answer (or ABSTAIN refusal) to user
```

## Cycle Boundary

```
Step 1     SIL fine-tune (verify_fn=None — internal retroverify disabled)
Step 2.1   score calibration fold under post-SIL model
              → outputs/full_run/cycle_{N}/calibration/{bm}_cycle{N}.json
Step 2.2   re-fit temperature scalar T (ECE-min on cycle's labelled fold)
Step 2.3   re-fit per-signal isotonic curves + conformal τ (label-dependent;
              EMA-smoothed against previous cycle's τ at α_ema = 0.7)
              → outputs/full_run/cycle_{N}/composite_calibration.json
              → outputs/full_run/cycle_{N}/conformal_gate.json
Step 2.4   pipeline.verifier.reload_calibration(...) — verifier rebound to
              the freshly-fitted JSONs without restarting auxiliary models
Step 2.5   retroactive re-verification under the recalibrated verifier
              (prune entries whose new u_stored < τ_retro = 0.50)
Step 3     save retroverify stats → cycle_{N}/retroverify_cycle{N}.json
Step 4     ingest fresh queries from this cycle's stream chunk
              (3000 q × 3 training benchmarks, deduplicated)
Step 5     eval pass over the held-out fold (500 q × 7 benchmarks)
Step 6     save memory checkpoint + deferred buffer
              → memory_store_cycle_{N}.{faiss,meta}
              → deferred_buffer_cycle_{N}.pkl
              → gdrive offload to gdrive:caem-phase1a/full_run/cycle_{N}/model.pt
```

---

## Repository Map

```text
caem/
  config.py                              # all hyperparameters + locked-configuration paths
  pipeline.py                            # 8-stage per-query pipeline + cycle orchestration
  pipeline_batch.py                      # batched-pool variant for verify-time speedup
  ablation/                              # ablation registry + variant overrides
  confidence/pre_routing.py              # u_pre estimation
  memory/
    encoder.py                           # SBERT wrapper
    entry.py                             # EpisodicEntry dataclass
    store.py                             # FAISS IndexIDMap + retroverify
    deferred.py                          # bounded deferral buffer (TTL-gated)
  retrieval/rag.py                       # DPR + FAISS passage retrieval
  routing/router.py                      # combined-score + safety-override dispatch
  training/self_improvement.py           # SIL fine-tune + L2 anchor + retention guard
  verification/
    verifier.py                          # 10-signal ensemble + early-exit + composite
    cal_prob_composite.py                # per-signal isotonic + L2 boost layer (Phase 2.1)
    conformal_gate.py                    # split-CP storage gate (Phase 2.4)
    kle.py                               # Kernel Language Entropy (registered upgrade)
    minicheck.py / qwen_judge.py         # NLI backends (Qwen-judge ablated)
    adaptive_nli_judge.py                # length-dispatched judge wrapper
    directional_p_ground.py              # refutation-bias-fixed grounding (Branch C)

eval/
  baselines.py                           # B1–B7 baseline runners
  benchmarks.py                          # benchmark loaders (FEVER, TQA, NQ, TruthfulQA,
                                         # StrategyQA, ARC-C, ASQA, MMLU)
  harness.py                             # eval driver (per-cycle JSON output)
  metrics.py                             # EM, F1, CHM, CES, McNemar, BCa bootstrap
  reporting.py                           # table/figure aggregation

scripts/
  run_experiment.py                      # main 10-cycle CAEM driver (resumable)
  run_baseline.py                        # B1-B7 baseline panel
  run_ablation.py / run_cyclic_ablation.py  # ablation sweeps
  fit_composite_calibration.py           # Phase 2.1 cal-prob composite fit
  fit_conformal_gate.py                  # Phase 2.4 conformal split-CP fit
  recalibrate_conformal_at_cycle.py      # per-cycle isotonic + conformal refit
  recalibrate_thresholds_at_cycle.py     # per-cycle quantile threshold refit
  run_calibration.py                     # T refit (ECE-min)
  validate_composite_weights.py          # cycle-zero validation gate
  sweep_composite_variants.py            # 25-variant sweep (Phase 1.5)
  rescore_eval_with_fitted_gate.py       # rescore Cycle 0 eval under locked gate
  build_passage_index.py                 # 21M-passage FAISS index build
  seed_cold_start.py                     # cold-start memory seeding
  baseline_sig_tests.py                  # paired McNemar + BCa bootstrap + Holm
  run_purity_validation.py               # empirical pool-purity diagnostic
  cycle2_retention_diagnostic.py         # Cycle-2 retention probe
  signal_correlation_matrix.py           # 10×10 Spearman matrix per benchmark
  aggregate_calibration_trajectory.py    # per-cycle T + τ trajectory aggregator
  aggregate_ablation.py                  # ablation panel aggregator
  generate_session5_artifacts.py         # cycle-zero audit + claim-evidence map TeX
  compare_prompt_design.py               # pre/post Phase-2 prompt-fix ablation
  phase4_artifacts.py                    # cycle-zero precision-cliff figure + table
  render_production_samples.py           # production-envelope sample renderings

tests/
  test_*.py                              # unit + integration tests for every module

run_phase1a.sh                           # CANONICAL 30+-step Phase 1a runbook
                                         # (build index → seed → calibrate →
                                         #  validate gate → step_7_main → diagnostics)

docs/
  PRODUCTION_RUNBOOK.md                  # end-to-end production deployment + per-cycle
                                         #   labelled refresh recipe (NEW 2026-04-27)
  caem-package-audit-report.md
  metrics-audit.md
  scripts-audit-report.md

thesis_report/                           # active thesis (Ch1-6 + Appendix A-G + bib)
pre thesis 1 report/                     # earlier rewrite tree (parallel edits)
```

---

## Benchmarks and Metrics

**Training panel** (3 benchmarks, contribute to seed/calibration/SIL training pools):

| Benchmark | Source | Metric |
|---|---|---|
| FEVER | `lucadiliello/fever` | label EM (SUPPORTS / REFUTES / NEI) |
| TriviaQA | `mandarjoshi/trivia_qa` (rc.nocontext) | EM + token F1 |
| Natural Questions | NQ-Open | EM + token F1 |

**Transfer panel** (4 benchmarks, evaluation/test only — never enter calibration or SIL pool):

| Benchmark | Source | Metric |
|---|---|---|
| TruthfulQA | `truthfulqa/truthful_qa` (generation) | EM proxy |
| StrategyQA | `ChilleD/StrategyQA` (test split) | EM (yes/no) |
| ARC-Challenge | `allenai/ai2_arc` (test split) | label EM (multiple-choice) |
| ASQA | `din0s/asqa` (dev split) | ROUGE-L |

**Out-of-distribution retention probe:** MMLU 200-sample subset, before/after every cycle's SIL fine-tune. Retention ratio ρ = MMLU_post / MMLU_pre; cycle aborts with weight rollback if ρ < 0.93.

**Composite hallucination metric (CHM):** equal-weighted mean over 8 measurable failure-mode subtypes (confident confabulation, factual fabrication, logical fabrication, off-topic, defensive evasion, template leakage, false refusal, length-padded over-generation). The 9th subtype (factual contradiction) is structurally unreachable under the MiniCheck-only deployment and is excluded from the denominator.

**Composite evaluation score (CES):** geometric mean over five axes (accuracy, epistemic quality = 1−CHM, retention, calibration, verification reliability). The geometric mean penalises collapse on any single axis rather than averaging it away.

---

## How to Run

### One-shot full Phase 1a pipeline (canonical)

```bash
bash run_phase1a.sh
```

This executes the full 30+-step runbook end-to-end: passage-index build → calibration-pair build → cold-start memory seeding → cycle-zero composite/conformal fits → validation gate → step_7_main 10-cycle run → external baselines → diagnostics → Phase 4 artefacts. Idempotent: each step checks for its own completion sentinel and skips if already done. Safe to re-run after any halt.

### Resume after halt

`run_phase1a.sh` auto-detects the last completed cycle from `outputs/full_run/cycle_*/` and passes `--resume_from_cycle N+1` to `scripts/run_experiment.py`. No manual flags needed.

### Smoke test (no GPU, pure-python)

```bash
python -m pytest tests/ -v
```

Single-module focused tests:

```bash
python -m pytest tests/test_self_improvement.py -v          # SIL + retention guard
python -m pytest tests/test_pipeline_batch_equivalence.py   # batch=serial proof
python -m pytest tests/test_calibration_batch_equivalence.py
```

### Just Cycle 0 calibration (re-fit if drifted)

```bash
python scripts/fit_composite_calibration.py --calib_jsons outputs/cycle_0/calibration/*.json --output outputs/cycle_0/composite_calibration.json
python scripts/fit_conformal_gate.py        --calib_jsons outputs/cycle_0/eval_rescored/*.json --output outputs/cycle_0/conformal_gate.json
python scripts/validate_composite_weights.py --output outputs/cycle_0/weight_validation.json
```

---

## Main Outputs

`outputs/full_run/` (built by `step_7_main`):

```
cycle_0/                                # baseline-eval staging from Cycle 0
  eval/{benchmark}_cycle0.json
cycle_{1..10}/                          # per-cycle artefacts
  model.pt + meta.pkl                   # SIL fine-tuned weights
  calibration/{benchmark}_cycle{N}.json # post-SIL calibration fold
  composite_calibration.json            # per-cycle isotonic + boost
  conformal_gate.json                   # per-cycle τ_store, τ_defer
  calibrated_config_cycle{N}.json       # per-cycle T
  retroverify_cycle{N}.json             # per-cycle prune count + memory size
  memory_store_cycle_{N}.{faiss,meta}   # per-cycle memory snapshot
  deferred_buffer_cycle_{N}.pkl
  retention_diagnostic.json             # MMLU pre/post + retention ratio
eval/{benchmark}_cycle{N}.json          # per-cycle eval results (500 q × 7)
mmlu_baseline.json                      # pristine-model MMLU = 0.6250
experiment_summary.csv                  # flat per-cycle aggregator
```

`outputs/calibration/` and `outputs/cycle_0/` hold cycle-zero locked artefacts that the trajectory inherits.

`outputs/baselines/` holds B1–B7 panel + significance tests (post-step_7_main).

`outputs/phase4/` holds the Cycle-0 thesis-ready artefacts: precision_cliff figure, atomic_scope, prompt_compliance, etc.

---

## Backups

**Local (rolling-N retention):** every cycle's `model.pt` saved to `outputs/full_run/cycle_{N}/model.pt`.

**Cloud (per-cycle gdrive offload):** `CAEM_GDRIVE_OFFLOAD=1` triggers `rclone copy` of every cycle's `model.pt` to `gdrive:caem-phase1a/full_run/cycle_{N}/model.pt`. Configured remote: `gdrive:`. Failures are non-fatal — local rolling-N is the on-disk guarantee.

**Cloud (one-time pre-Step-7 snapshot on HuggingFace Hub):** `aksaN000/caem-passage-index-21m:pre_main_snapshot/` contains the 64GB FAISS passage index + cold-start memory + Cycle 0 calibration JSONs + sweep evidence + audit logs. If the rented Vast instance dies:

```bash
hf download aksaN000/caem-passage-index-21m \
  --include 'passages.*' 'pre_main_snapshot/*' \
  && bash run_phase1a.sh
```

The runner picks up from whichever cycle is the highest gdrive-resident `model.pt`.

---

## Documentation

| Document | Purpose |
|---|---|
| `run_phase1a.sh` | canonical 30+-step Phase 1a runbook |
| `docs/PRODUCTION_RUNBOOK.md` | production deployment + per-cycle labelled refresh recipe |
| `thesis_report/main.tex` | Ch 1–6 + Appendix A–G thesis manuscript |
| `thesis_report/appendix/appendix_g.tex` | architecture-positioning Q&A: 5 differentiators vs plain fine-tuning |
| `thesis_report/bibliography/references.bib` | ~91 entries; ~84 cited |
| `branch_C_log.md` | dated implementation diary |
| `caem-implementation-log.md` | longer-form decision rationale |
| `hyperparameter-reference.md` | three-category hyperparameter taxonomy |

---

## Reproducibility

- Python 3.12 + `/venv/main` (PyTorch 2.10.0+cu130, transformers, sentence-transformers, faiss-cpu, scikit-learn, scipy, bitsandbytes)
- Single recorded seed propagates through generator, sampler, dropout, FAISS
- All decision artefacts on disk: every τ, T, isotonic curve, weight-validation verdict, sweep entry, retention probe lives at a registered path under `outputs/`
- Hardware: rented Vast.ai RTX 5090 (32 GiB physical, ~31 GiB usable). Five OOM mitigation patches in `caem/training/self_improvement.py` and `run_phase1a.sh` keep peak VRAM under the envelope; full discussion in `docs/PRODUCTION_RUNBOOK.md` §Hardware.

---

## Phase 1a Quick-Reference Decision Map

| Question | Answer | Where |
|---|---|---|
| Which model? | Qwen-2.5-3B-Instruct, bf16 | `caem/model_loader.py` |
| Which NLI judge? | MiniCheck only (long-hypothesis Qwen judge ablated 2026-04-26) | `caem/verification/adaptive_nli_judge.py` |
| How many signals in the composite? | 10 (3 internal + 3 sample-set + 3 grounding + 1 q_a_relevance); `p_contra` excluded as structurally zero under MiniCheck | `outputs/cycle_0/composite_calibration.json:metadata.signals` |
| Composite type? | Per-signal isotonic + L2-regularised logistic boost | `caem/verification/cal_prob_composite.py` |
| Storage gate type? | Conformal split-CP at α_store=0.05, α_defer=0.40, with EMA-smoothed per-cycle refit at α_ema=0.7 | `caem/verification/conformal_gate.py` |
| How are cycle-boundary recalibrations sequenced? | SIL → score calibration fold under post-SIL → re-fit T + isotonic + τ → reload verifier → retroverify | `scripts/run_experiment.py` Step 1 + Steps 2.1–2.5 |
| What was the early-exit calibration-drift fix? | `is_query_time` flag on `verifier.verify()`; retroverify passes False so its first conjunct doesn't auto-fire on memorised chains | `caem/verification/verifier.py:verify` |
| Cold-start size? | 260 verified-correct episodes (purged from 348 against gold) | `outputs/cold_start_memory/memory_store.{faiss,meta}` |
| Locked thresholds? | τ_store=0.6676, τ_defer=0.5207, τ_train=0.5784 | `outputs/cycle_0/conformal_gate.json` |
| MMLU retention floor? | ρ_min = 0.93 (cycle aborts + weight rollback if breached; memory preserved) | `caem/training/self_improvement.py` |

---

## License & Citation

Research project for academic submission (CSE400, BRAC University, Fall 2026 cohort). Citation pending thesis acceptance.

---

## Acknowledgements

The literature foundation for this work is documented in Ch 2 (Literature Review). The architectural lineage in particular owes to:
- Mohri & Hashimoto (2024) — conformal factuality at the LM claim level
- Cherian, Gibbs, Cand\`{e}s (2024) — boosted logistic aggregation over per-claim scores
- Yadkori et al. (2024) — conformal abstention for LLM hallucination
- Farquhar et al. (2024) / Kuhn et al. (2023) — semantic entropy
- Tang et al. (2024) — MiniCheck claim-support verification
- Kirkpatrick et al. (2017) — elastic weight consolidation foundation for L₂ anchor
- Zelikman et al. (2022) — STaR self-improvement loop
- Hosseini et al. (2024) — V-STaR verifier-guided self-training
- Song et al. (2024), Fu et al. (2025), Das et al. (2025), Huang-Block-Foster (2025), Lang-Vijayaraghavan-Sontag (2024) — theoretical foundations for verifier-filtered self-training

Full citations + ~91 bibliography entries in `thesis_report/bibliography/references.bib`.
