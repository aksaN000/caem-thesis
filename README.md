# CAEM — Confidence-Aware Episodic Memory with Self-Improvement

CSE400 Final Year Thesis | BRAC University
Student: Aksan Gony Alif

## Current Status

Implementation and thesis drafting are both complete. The project is at the inflection point between design and experiment: the codebase is frozen, Chapters 1–5 are drafted and audited, and Phase 1 of the experimental evaluation is about to launch on a rented Vast.ai RTX 5090.

1. ✅ Core pipeline implementation (Stages 1–8)
2. ✅ 8 external baselines (B1–B7 + optional STaR ceiling)
3. ✅ 16-variant ablation registry + cyclic ablation driver
4. ✅ Cold-start memory seeding (447 verified episodes stored)
5. ✅ Unit-test suite for every module
6. ✅ Static AST + cross-file rename validation (caught two corruption bugs, both fixed)
7. ✅ Chapter 1–5 drafts, three-layer audited, cross-chapter consistent
8. ⏳ Phase 1 Vast run (n=5000 × 6 benchmarks × 10 cycles, ~40 h, ~$28) — pending launch
9. ❌ Chapter 5 tables populated from real data — blocked on Step 8
10. ❌ Chapter 6 (Conclusion) — blocked on Step 9

**Drafted chapters:** Ch 1 (345 lines), Ch 2 (250), Ch 3 (283), Ch 4 (1395), Ch 5 (1354). Ch 6 is a stub file awaiting results.

## What CAEM Is

CAEM is an architectural hallucination-reduction system built on Flan-T5-Large (780M params). It combines:

- **Episodic memory** of verified QA episodes, indexed in FAISS for sub-millisecond retrieval.
- **Confidence-aware adaptive routing** through three tiers of increasing cost (memory recall → guided generation → RAG).
- **Nine-signal verification**: three internal signals (token probability, MC Dropout, token entropy), three sample-set signals (self-consistency, semantic entropy, answer stability), three external-grounding signals (NLI entailment, atomic-fact grounding, passage-level grounding), plus a contradiction probability `p_contra` used as a separate veto.
- **Iterative self-improvement** across 10 cycles: each cycle fine-tunes on verified episodes and re-verifies prior episodes retroactively.
- **L2 anchor regularization** (λ=0.01) as an EWC approximation that prevents catastrophic forgetting without the prohibitive cost of a Fisher information matrix at 780M params.

The emphasis is system-level architecture, not a post-hoc detection patch.

## Core Architecture

Single-query pipeline:

1. Encode query with Sentence-BERT (`all-mpnet-base-v2`, 768-dim).
2. Estimate pre-routing confidence `u_pre`.
3. Safety override: if `u_pre < τ_safe = 0.60`, force Tier 3 regardless of memory match.
4. Otherwise compute combined routing score `S = 0.70 · sim + 0.30 · u_stored` and dispatch:
   - **Tier 1** (memory recall): direct answer from the matched episode. No per-query verifier call — quality refresh deferred to between-cycle retroverification.
   - **Tier 2** (guided generation): generate with memory context, then run the nine-signal verifier.
   - **Tier 3** (RAG generation): retrieve passages from the DPR Wikipedia index, generate, then run the nine-signal verifier.
5. Combine verifier outputs into `u_stored = 0.30·p_ground_mean + 0.15·p_ground_atomic + 0.15·p_entail + 0.15·s_avg + 0.15·u_internal + 0.10·(1 − h_norm)`.
6. Four-outcome storage decision: **Store** (u ≥ τ_store = 0.65 and p_contra < veto), **Deferred** (borderline), **Abstain** (refuse to answer), or **Discard** (low confidence, not stored).
7. Between cycles: retroactively re-verify stored episodes, fine-tune on episodes above τ_train = 0.75, abort the cycle if the MMLU retention ratio drops below ρ_min = 0.93.

## Repository Map

```text
caem/
  config.py                    # all hyperparameters + TRAINING/TRANSFER benchmark lists
  pipeline.py                  # orchestrates routing → generation → verification → storage
  ablation/                    # 16-variant registry, variant-specific overrides
  confidence/
    pre_routing.py             # u_pre estimation (pre-routing safety gate)
  memory/
    entry.py                   # EpisodicEntry dataclass (includes source_benchmark)
    encoder.py                 # Sentence-BERT wrapper
    store.py                   # FAISS IndexIDMap(IndexFlatIP), add/search/update
  routing/
    router.py                  # OR-safety + S = 0.70·sim + 0.30·u_stored dispatch
  retrieval/
    rag.py                     # DPR + FAISS passage retrieval, FLARE look-ahead
  verification/
    verifier.py                # 9-signal verifier + p_contra veto + u_stored composite
  training/
    self_improvement.py        # 10-cycle loop, L2 anchor, MMLU abort guard, resume path

eval/
  baselines.py                 # B1–B5 inference baselines (Zero-shot, CoT, RAG, CoT+RAG, FLARE)
  benchmarks.py                # loaders for the 6 factual-QA benchmarks + MMLU probe
  harness.py                   # per-cycle evaluation driver
  metrics.py                   # EM, F1, pooled_chm, chm_reduction_verdict, McNemar, bootstrap CI
  reporting.py                 # table-ready result summarisation

scripts/
  build_passage_index.py       # DPR Wikipedia FAISS index build
  check_base_model.py          # Gap 1 diagnostic (Flan-T5-Large baseline accuracy)
  check_env.py                 # interpreter/import sanity
  seed_cold_start.py           # cold-start 447 verified episodes
  run_experiment.py            # headline 10-cycle CAEM run, resumable
  run_baseline.py              # driver for B1–B5 (inference-only baselines)
  run_simple_ft.py             # driver for B6 Vanilla FT, B7 EWC-only FT, optional STaR ceiling
  run_calibration.py           # per-cycle temperature T refit on the disjoint calibration slice
  run_ablation.py              # single-point (cycle 10) ablation sweep
  run_cyclic_ablation.py       # full 16-variant × 3-cycle screening sweep
  aggregate_ablation.py        # collapse screening outputs, select top-N for confirmatory sweep
  run_purity_validation.py     # empirical check of the data-purity theorem
  make_tables.py               # generate Chapter 5 result tables from run outputs
  make_figures.py              # generate Chapter 5 result figures (CES trajectory, calibration, etc.)
  hardware.py                  # hardware probing, mixed-precision + batch-size policies

tests/
  test_ablation.py
  test_episodic_memory.py
  test_eval.py
  test_flare_smoke.py          # tier-A mock FLARE decoder-slice regression test
  test_pipeline.py
  test_pre_routing.py
  test_rag.py
  test_router.py
  test_self_improvement.py
  test_verifier.py
```

## Key Technical Facts

- Backbone: **Flan-T5-Large** (780M params), Sentence-BERT: `sentence-transformers/all-mpnet-base-v2` (768-dim).
- Episodic memory: **FAISS `IndexIDMap(IndexFlatIP)`**, per-episode `u_stored` stored alongside the vector.
- **Nine verifier signals** (3 internal + 3 sample-set + 3 external-grounding) + **p_contra veto**.
- `u_stored` composite weights: start equal at `0.25/0.25/0.25/0.25` across the four signal families; projected post-calibration target is `0.20/0.20/0.20/0.40` (grounding-weighted). Chapter 5's hparam table reports measured calibrated values.
- `τ_safe = 0.60` (OR-condition RAG override), `τ_store = 0.65`, `τ_train = 0.75`, `ρ_min = 0.93` (MMLU retention floor for cycle abort).
- **10 self-improvement cycles** per full run.
- L2 anchor regularization (`λ_reg · ||θ − θ_prev||²`), λ=0.01 — **not** EWC (Fisher matrix too expensive at 780M params).
- `CycleResult.mmlu_retention_ratio` (renamed in Session 72+ from the deprecated `forgetting_score` field name; see `docs/metrics-audit.md` rows 15 / 15b).

## Benchmarks and Metrics

| Benchmark | Metric | Role |
|---|---|---|
| FEVER | label accuracy | fact verification |
| TriviaQA | EM + token F1 | open-domain QA (training-pool) |
| Natural Questions | EM + token F1 | open-domain QA (training-pool) |
| TruthfulQA | ROUGE-L (with EM proxy) | adversarial truthfulness |
| StrategyQA | EM on yes/no | implicit multi-hop |
| ARC-Challenge | EM | grade-school science |
| MMLU (n=200 probe) | 4-choice accuracy | **retention control**, not a baseline-comparison target |

Pooled CHM (equal-weighted 8-subtype composite hallucination metric) is split into in-distribution (training-pool) and out-of-distribution (held-out transfer) via `eval/metrics.py::pooled_chm`. The ≥30% relative CHM reduction target is committed as a pass/fail criterion via `chm_reduction_verdict`, and CES.EPI is now defined as `1 − CHM` so the efficacy score and the pre-registered gate speak the same language.

## Environment Setup

Python 3.10+ with `.venv` (Pyright configured in `.vscode/settings.json`).

```bash
pip install torch transformers datasets sentence-transformers faiss-cpu pytest scipy scikit-learn tqdm
```

For CUDA systems:

```bash
pip install faiss-gpu
```

### VS Code Fresh-Session Bootstrap

Run this at the start of each fresh session:

1. Run task: `CAEM: Session Bootstrap`
2. If old squiggles remain: `Python: Restart Language Server`
3. Then `Developer: Reload Window`

Task definitions are in `.vscode/tasks.json`:

- `CAEM: Check Env` — verifies interpreter and key imports
- `CAEM: Quick Diagnostics` — quick pytest sanity check
- `CAEM: Session Bootstrap` — runs both in sequence

## Execution Workflow

**The canonical runbook is `NEXT_SESSION_PLAN.md`.** It contains the 20-step, line-by-line, command-by-command sequence for Phase 1 on Vast.ai. The summary below is the one-paragraph version for context — do not run it as-is; follow the plan.

1. Pre-flight on local PC (git clean, GitHub PAT, Vast balance).
2. Rent an RTX 5090 (~$0.57–0.63/hr total including disk), 150 GB disk.
3. Remote env setup + HF model cache.
4. **Step 3B: mandatory `pytest tests/test_self_improvement.py` gate** before any paid GPU pipeline work.
5. Build the DPR Wikipedia passage index (2.5 h).
6. Smoke test: 1 cycle, n=50 per benchmark (20 min, catches integration bugs).
7. Cold-start memory seeding.
8. Headline 10-cycle CAEM run (17 h, full n=5000 per benchmark).
9. FLARE pre-flight smoke + baselines B1–B7 (~18 h cumulative).
10. 16-variant screening sweep at 3 cycles (17 h).
11. Aggregate screening, pick top-N for confirmatory.
12. Confirmatory sweep: top-N + `full` reference, 10 cycles each (55 h).
13. Purity-theorem validation, aggregate all outputs, download, stop.

**Resume a crashed run:**

```bash
python scripts/run_experiment.py --resume_from_cycle 2 \
  --output_dir outputs/full_run --passage_index data/passage_index
```

The resume path persists memory store, deferred buffer, fine-tuned weights, and calibration `T` per cycle (Session-81 fix).

## Testing

```bash
python -m pytest tests -v                             # full suite
python -m pytest tests/test_self_improvement.py -v    # focused: self-improvement module (must pass before Vast Step 3B)
python -m pytest tests/test_pipeline.py -v            # focused: pipeline integration
```

## Main Outputs

From `run_experiment.py` (Step 7):

- `all_cycle_results.json` — per-cycle summary, all benchmarks
- `experiment_summary.csv` — flat table with `mmlu_retention_ratio_pct`, `mean_latency_ms`, etc.
- `dataset_splits.json` — train/calib/test seeds for reproducibility
- `memory_store_cycle_0.faiss/.meta` … `memory_store_cycle_10.faiss/.meta`
- `deferred_buffer_cycle_*.pkl`
- `cycle_1/` … `cycle_10/` checkpoints (Flan-T5-Large weights + optimizer state)
- `calibration/calibrated_config_cycle*.json` — per-cycle `T` refit
- `retroverify_cycle*.json` — between-cycle retroverification logs

From auxiliary scripts:

- `outputs/purity_validation/theory_validation.json` — empirical check of P > p
- `outputs/ablation/ablation_summary.json` — screening sweep aggregated scores
- `outputs/baselines/*/results.json` — per-baseline EM/F1/latency

## Documentation Pointers

- **`NEXT_SESSION_PLAN.md`** — the master 20-step Vast runbook. Canonical for all execution.
- **`VAST_AI_DEPLOYMENT_GUIDE.md`** — Vast-specific deployment notes (SSH, tmux, model caching, download patterns).
- `docs/metrics-audit.md` — current metric definitions and deprecated-field history (read rows 15 / 15b for the MMLU abort guard).
- `caem-implementation-log.md` — chronological implementation decisions and fixes, session-indexed.
- `hyperparameter-reference.md` — three-category hparam taxonomy (literature-fixed / design / calibrated).
- `writing-suggestions.md` — chapter-by-chapter writing guidance (C1–C6 corrections, GEN entries, FIG/ALG/EQN/THM/TAB inventory, PUB-01–PUB-06 publication-elevation gaps).

## Thesis Chapter Files

| Chapter | File | Lines | Status |
|---|---|---:|---|
| Ch 1 — Introduction | `pre thesis 1 report/chapters/chapter_1.tex` | 345 | ✅ Drafted + audited |
| Ch 2 — Literature Review | `pre thesis 1 report/chapters/chapter_2.tex` | 250 | ✅ Drafted + audited |
| Ch 3 — Requirements | `pre thesis 1 report/chapters/chapter_3.tex` | 283 | ✅ Drafted + audited |
| Ch 4 — Methodology | `pre thesis 1 report/chapters/chapter_4.tex` | 1395 | ✅ Drafted + audited |
| Ch 5 — Results | `pre thesis 1 report/chapters/chapter_5.tex` | 1354 | ⚠️ Prose drafted; result tables scaffolded, awaiting Phase 1 data |
| Ch 6 — Conclusion | `pre thesis 1 report/chapters/chapter_6.tex` | 0 | ❌ Stub — written after results |

File numbering matches chapter numbering: `chapter_N.tex` = Chapter N. (The earlier template quirk noted in old docs has been resolved.)
