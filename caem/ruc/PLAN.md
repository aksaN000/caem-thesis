# RUC v2 build plan + status

Tracks the leakage-free v2 RUC build, from baseline-generation through CAEM-with-RUC trajectory. Updates as phases complete.

For *what v1 does today*, see `README.md`. For *architecture / features / label*, see `DESIGN.md`.

---

## Why v2 is needed

The v1 RUC's training rows come from `benchmark_pools[bench].eval` — the same eval pool CAEM uses to score its trajectory. Deploying v1 inside CAEM and then evaluating on the same pool is label leakage. v1 stays as methodology evidence; v2 is the leakage-free deployable.

## v2 fresh pool (already on disk)

Content-hash disjoint from every CAEM pool (seed / purity / calibration / sil_train / eval / test) by construction.

| Bench | Samples | Type | RAG-extensive? |
|---|---:|---|:---:|
| FEVER | 2 500 | claim verification | borderline neutral (−0.02 utility in v1) |
| TriviaQA | 2 500 | factoid QA | ✅ yes (+0.11) |
| CommonsenseQA | 500 | commonsense MCQ | no (−0.13) |
| StrategyQA | 187 | multi-hop yes/no | no (−0.04) |
| TruthfulQA | 317 | adversarial myth | no (−0.20) |
| HaluEval-QA | 2 500 | hallucination induction | unknown — expected near-zero |
| OpenBookQA | 500 | science MCQ | unknown — expected slightly negative |
| **Natural Questions** | **3 000** | **open-domain factoid** | ✅ **yes** (+0.13 in smoke) |
| **Total** | **12 004** | 8 benches, 2 RAG-extensive | A:B balance ≈ 1.0 at pool |

NQ was added 2026-05-18 to balance the training distribution — v1 had only TriviaQA as a RAG-extensive bench, which constrained cross-bench transfer. NQ adds a second positive bench without modifying the CAEM training panel (it stays at fever / triviaqa / commonsense_qa per `caem.config.TRAINING_BENCHMARKS`).

---

## v2 build phases

### Phase 1 — v2 baselines on fresh pool (zero_shot + rag)

| Detail | Value |
|---|---|
| Status | 🔄 **in progress** (launched 2026-05-18) |
| Compute | ~24 GPU-h |
| Cost | ~$10 Vast |
| Output | `outputs/baselines_v2/{zero_shot,rag}/<bench>_cycle0.json` |
| tmux session | `baselines_v2` |
| Launcher | `scripts/launch_baselines_v2.sh all` |
| Env | `CAEM_BATCH_U_TOK_DROP=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Why these baselines | Canonical pairing for the label is `(zero_shot, rag)`. Other baselines added 0.00 AUROC in v1 (registered no-lift). |

### Phase 2 — Rescore both baselines through cycle-3 verifier

| Detail | Value |
|---|---|
| Status | ⏳ pending Phase 1 |
| Compute | ~25 GPU-h |
| Cost | ~$10 Vast |
| Output | `outputs/baselines_v2/{zero_shot,rag}/<bench>_cycle0_with_chm.json` (with 9 active signals + `u_stored`) |
| Optimisations | `CAEM_BATCH_U_TOK_DROP=1`, `--batch_size 32` (env-var-only, no verifier.py edits). Required because the cycle-3 verifier composite was fitted with the env var ON. |
| Script | `scripts/rescore_baselines_through_verifier.py` |

### Phase 3 — Feature engineering on the 12 004 fresh-pool questions

| Sub-stage | Compute | Notes |
|---|---|---|
| P(IK) probe on Qwen hidden states | ~1 GPU-h | Reuses `scripts/ruc_train_pik_probe.py`; output: `pik_probe_v2.joblib` + cached hidden states |
| FAISS top-1 + top-5 retrieval features | ~4 GPU-h (mostly CPU-bound FAISS-IVF) | Reuses production `QueryEncoder` (mpnet 768-dim) + 21M passage index. Top-1 already cached at `caem/ruc/passage_features.parquet` for v1's 1500; v2's extra 10 504 questions need a fresh pass. |
| Wikipedia pageviews (free Wikimedia REST) | CPU + WAN | Reuses `caem/ruc/wiki_pageviews_cache.jsonl` from v1 (1309 entities cached) + new fetch for novel entities |
| Linguistic + answer-type features | ~5 min CPU | Pure regex, no API |

### Phase 4 — Build v2 training set

| Detail | Value |
|---|---|
| Status | ⏳ pending Phase 3 |
| Compute | ~30 min CPU |
| Output | `caem/ruc/training_set_v2_canonical.parquet` (~12 004 rows × pairing-canonical) |
| Label | per-sample binary: case A → 1, case B → 0, **cases C and D DROPPED** (CHM-tiebreak noise; v1 evidence: dropping ties lifted AUROC by +0.08) |
| Expected clean A+B rows | ~3 000 (v1 had 519 clean rows from 1500 raw) |
| Expected pos rate | ~42–48 % (vs v1's 42 %; balance improves with NQ) |
| Script | `scripts/build_ruc_training_set.py` |

### Phase 5 — v2 RUC training

| Detail | Value |
|---|---|
| Status | ⏳ pending Phase 4 |
| Compute | ~10 min CPU |
| Output | `caem/ruc/v2_shortcut/{models/{scaler,lr_final}.joblib, feature_spec.json}` |
| Model | Logistic Regression + StandardScaler (same family as v1) |
| Validation | **Nested 5-fold CV**: inner sweeps C ∈ {0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}; outer reports pooled and per-bench AUROC |
| Threshold τ | Swept post-fit on full-data predictions to maximise net utility (A_caught − B_caught) on the v2 distribution |
| Expected hyperparameters | C ≈ 0.05-0.5 (less regularisation than v1's 0.005 because n/p jumps from 22 to ~125); τ ≈ 0.40-0.55 |
| Expected pooled CV AUROC | 0.70-0.74 |
| Expected per-bench AUROCs | TruthfulQA ~0.68, TriviaQA ~0.70, NQ ~0.65, CSQA ~0.62, StrategyQA ~0.60, FEVER ~0.55 |

### Phase 6 — Wire v2 RUC into router, smoke test

| Detail | Value |
|---|---|
| Status | ⏳ pending Phase 5 |
| Compute | CPU |
| Files touched | None — `caem/routing/router.py` already accepts the LR-shape RUC via `LRRetrievalUtilityClassifier`. Only the spec path changes from `v1_shortcut/feature_spec.json` to `v2_shortcut/feature_spec.json`. |
| Smoke | 50-query matched-protocol pass: RUC-routed vs always-T3 on cycle-0 cal-fold |

### Phase 7 — CAEM-with-RUC trajectory (the leakage-free Ch6 result)

| Detail | Value |
|---|---|
| Status | ⏳ pending Phase 6 |
| Compute | ~4-8 GPU-days |
| Cost | ~$35-75 Vast |
| Output | `outputs/full_run_ruc/{cycle_N,eval}/` |
| Run path | (a) Phase 1a (run_phase1a.sh) with `AdaptiveRouter(ruc=...)`. (b) step_7_main trajectory with the same router. Retention guard decides cycle count. |
| Eval pool | Standard CAEM eval (300/bench × 5 benches) — content-hash disjoint from v2 RUC training by construction |

---

## Total budget (v2 build, end-to-end)

| | Compute | Cost |
|---|---|---|
| Phase 1 baselines | ~24 GPU-h | $10 |
| Phase 2 rescore | ~25 GPU-h | $10 |
| Phase 3 features | ~5 GPU-h | $2 |
| Phase 5 training | CPU | $0 |
| Phase 7 trajectory | ~100-200 GPU-h | $40-80 |
| **Total** | **~150-250 GPU-h** | **$60-100 Vast, $0 Anthropic** |
| **Wall time unattended** | **~6-10 days** | |

---

## Live progress (auto-updated as phases close)

```
2026-05-18  Phase 1 launched in tmux baselines_v2 (zero_shot + rag on 12 004 questions)
            Cleanup + doc reorganisation: legacy LightGBM artefacts archived under v1_archive/intermediate/
            NQ added to fresh pool (3 000 samples) to balance training distribution
            README.md + DESIGN.md rewritten; this PLAN.md created
            v1 RUC frozen at C=0.005, tau=0.40 (CV AUROC 0.647, net utility +33)
            Launcher updated with skip-if-exists so resume-after-interrupt is safe
            Confirmed 9-baseline Ch5 panel (B1-B9 exist at outputs/baselines/);
              only B1 + B3 re-run on fresh pool for RUC training; B2/B4-B9 stay
              at outputs/baselines/ for the thesis comparison panel.
            Empirical throughput at bs=32 + prefetch: ~12 samples/sec (vs my
              conservative 2 sec/sample estimate). Phase 1 ETA revised
              ~24 h -> ~2-3 h; cost revised ~$10 -> ~$1.20.

Decisions locked 2026-05-18 (post-baseline-audit):
  - B6 vanilla_ft: 6/10 cycle files rescored; 4 resumable
      (truthfulqa c3/c5, strategyqa c3/c5). Will fire as Side-work C1 (B6 vanilla_ft rescore resume)
      alongside the v2 build, no GPU contention with the main critical path.
  - B7 ewc_only_ft: SKIPPED FROM THE PANEL (planned). The L2 anchor's
      fp32 cast on 3B params OOMs on the 32 GiB envelope; even with
      bs=4 grad_accum=8 and 8-bit AdamW the anchor sum overflows.
      Ch5 §sec:comp-ewc-ft footnote documents this as a hardware-bound
      scope decision and references the OOM evidence at
      outputs/baselines/ewc_only_ft.train.log. No baseline rerun, no
      rescore attempt. The 9-baseline panel becomes
      "8 baselines + 1 hardware-limited footnote" in Ch5.
```
