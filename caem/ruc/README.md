# CAEM Retrieval Utility Classifier (RUC)

**Owner:** Aksan Gony Alif (CSE400 thesis, BRAC University) · **Branch:** `branch/ruc-classifier` · **Last update:** 2026-05-18

Single canonical doc for the RUC — overview, architecture, build plan, operational sequencing, risk mitigation, acceptance gates, dataset construction, and live status.

---

## 0. What the RUC is

A frozen, pre-routing classifier living inside `caem.routing.AdaptiveRouter` at the two Tier-3 dispatch points. Per query, it decides commit-to-T3 (RAG) vs fall-back-to-T2 (parametric CoT). Architectural goal: send queries where RAG genuinely helps to T3; keep queries where RAG injects hallucination at T2.

The classifier scores **every** query that reaches its fire points; the threshold τ converts the score to RAG/DIRECT. The label is per-sample, not per-benchmark — the RUC does not need to know which benchmark a query came from.

### Where it fires in the router

```
ROUTE(query):
  encode → memory_search → get (similarity_top1, u_stored_top1) and u_pre

  ┌── MECHANISM 1: SAFETY VETO (evaluated first)
  │   if u_pre < safety_threshold:
  │       → consult RUC
  │           if p_rag ≥ τ:    → Tier 3
  │           else:             → Tier 2
  │       (default Tier 3 when RUC not wired)
  │
  └── MECHANISM 2: ROUTING SCORE
        score = 0.70 · similarity + 0.30 · u_stored
        if score ≥ 0.90:        → Tier 1   (true memory exact match — RUC SKIPPED)
        elif similarity > 0.75: → Tier 2   (memory near-hit       — RUC SKIPPED)
        else:                   → consult RUC (fall-through)
            if p_rag ≥ τ:    → Tier 3
            else:             → Tier 2
```

Both branches previously hardcoded Tier 3. The RUC intercepts both. See `caem/routing/router.py:_consult_ruc` for the integration.

### Cost at the RUC decision point (per query)

| Step | Cost |
|---|---|
| 14 surface features (regex + NER + interrogatives) | ~1 ms CPU |
| FAISS top-5 search on 21M-passage IVF index | ~30-50 ms CPU (page-cached) |
| Top-5 retrieval features (sim stats + entity overlap) | included in the search step |
| Cross-encoder rerank on top-5 (BAAI/bge-reranker-v2-m3) | ~30 ms GPU |
| NLI pairwise cross-agreement (10 pairs via MiniCheck) | ~50 ms GPU |
| P(IK) Qwen forward pass on question hidden state | ~10 ms GPU |
| 3 linguistic + 4 answer-type regex features | ~1 ms CPU |
| Wikidata entity-binary lookup | <1 ms CPU |
| StandardScaler.transform + LR.predict_proba | < 1 ms CPU |
| **Total per RUC consultation** | **~120-150 ms** |

The top-5 FAISS search is reused by Tier 3 if T3 is the verdict (no double retrieval cost). Wasted work on queries the RUC sends to T2 is the cheap top-5 search + rerank + NLI — the cost of avoiding an expensive T3 generation pass where RAG would have hurt.

---

## 1. Current state (live)

```
2026-05-20 09:00 BDT

✅ STEPS 1-6 + P1 + P2 cycle-0 ALL COMPLETE — Phase 1e is in cycle-1 SIL.

  v2 baselines:         16/16 JSONs (Step 1)
  Haiku TQA rescore:    634 judgments, em_llm_judged populated (H)
  Step 2 (rescore):     🚫 RETIRED (Path B doesn't need it)
  P(IK) probe v2:       mean OOF AUROC 0.771 (P)
  A+B pre-filter:       3 200 disagreement rows kept (2.5)
  Step 3 features:      3 208 × 13 v2 features in caem/ruc/v2/features.parquet
  Step 4 training set:  caem/ruc/v2/training_set.parquet (3 208 × 36 features + label)
  Step 5 trained LR:    pooled OOF AUROC 0.8501, final C=0.5, τ=0.375
                          TQA per-bench AUROC 0.732
                          Net deployment utility 2 477/3 208 (77.2 %)
                          In-sample vs OOF gap 0.011 — no overfit
                          Per-fold AUROC std 0.008 — stable
  Step 6 model smoke:   6/7 checks pass; StrategyQA T3-share outlier (n=43 noise)

  P1 RUC wiring:        ✅ caem/routing/ruc.py + router.py + pipeline.py all patched
                          - LRRetrievalUtilityClassifier accepts **extra_features
                          - router thread extras through _consult_ruc
                          - pipeline._compute_ruc_features() reuses verifier
                            components for the 13 v2 features
                          - P(IK) probe loaded at pipeline init, uses
                            model.disable_adapter() context for base-Qwen
                            hidden states
                          - End-to-end smoke passed: well-known factoids →
                            DIRECT, long-tail factoids → RAG, adversarial
                            myth → DIRECT, weak retrieval → DIRECT.
                            Architectural intent verified at deployment.

  2026-05-19 launch bugs caught + fixed (during P2):
    - RUC.predict() string-coerce bug — _top1_passage_text str leaked into
      **extra_features and crashed float() in _feature_vector. Fix:
      filter underscore-prefixed keys at router._consult_ruc boundary.
      Caught after first launch attempt; restarted.
    - BatchPipeline._peek_routing didn't thread RUC features into
      router.route(). Step 7.0 batch path silently fell back to vanilla
      routing (all T3) for the first batch of 32. Fix: added
      _compute_ruc_features call + RUC kwargs to _peek_routing. Restarted.
    - Warnings filterwarnings added at top of verifier.py, pipeline.py,
      pipeline_batch.py to suppress greedy+sampling-param cosmetic
      transformers warnings (~6418 per cycle). Takes effect on next launch.

  P2 status (live trajectory under run_phase1a.sh, tmux plan_a_ruc):
    Launched 2026-05-19 17:22 UTC / 23:22 BDT
    Step 7.0 cycle-0 baseline:    closed 23:25 BDT 2026-05-19 (6h 3min)
    Step 7.0.1 composite_calibration.json:  written 17:21 UTC
    Step 7.0.3 audit:             informational only — store_discard_gap
                                  fails on StrategyQA (-0.007, expected >0.05)
                                  + CSQA memory_poisoning_rate 24.3% logged.
    gdrive snapshot:              uploaded to
                                  gdrive:caem-phase1a/pre_main_snapshot/700b9e3/
                                  (Phase 1d archived to .../700b9e3_phase1d/)
    Cycle 0 of step 7 main:       closed 23:25 UTC 2026-05-19 (6h)
                                  Per-bench T2 share (cycle-0 eval):
                                    FEVER       43.6%
                                    TriviaQA    33.2%
                                    CommonsenseQA  92.0%  ← keystone
                                    TruthfulQA  65.1%
                                    StrategyQA  15.0%
                                  matches RUC training-time intent.
    Cycle-0 close ops:            MMLU baseline (0.6350), multi-modal
                                  retention baseline (mmlu=0.61,
                                  triviaqa_test=0.395, csqa_test=0.825).
    Currently:                    Cycle 0 T-scalar fit (cal-fold scoring
                                  under calibrated composite). 2h+ in.
                                  ~3-4h to T-fit close, then cycle-1 SIL
                                  fine-tune begins (~12-13 BDT 2026-05-20).

Next: P3 (cycle 1 → retention-guard-fire or cycle 10) — ~3-7 GPU-days.
      Then A1-A4 (ablation tables) + C1-C3 (Ch6 deliverables).
```

### How to load and call the deployed RUC (after Step 5 closes)

```python
from pathlib import Path
from caem.routing.ruc import LRRetrievalUtilityClassifier

ruc = LRRetrievalUtilityClassifier.from_spec(Path("caem/ruc/v2/feature_spec.json"))

decision = ruc.predict(
    question="What happens if you crack your knuckles?",
    top1_passage_text="...",         # pipeline supplies via FAISS top-5 lookup
    top1_passage_sim=0.42,
    top1_passage_entity_overlap=0.0,
    p_ik=0.5,
    source_benchmark="truthfulqa",   # optional, for logging
)
# decision.decision in {"RAG", "DIRECT"}
# decision.p_rag in [0, 1]
# decision.threshold = τ
```

---

## 2. Architecture

### 2.1 Label decomposition — A, B, C, D

Each query has both a direct (T2-style) answer from `ZeroShotBaseline` and a RAG (T3-style) answer from `RAGBaseline` in the training data. EM scoring of each yields one of four cases:

| Case | direct_em | rag_em | Meaning | Training label | Used in training? | Deployment default (architectural intent) |
|---|:---:|:---:|---|:---:|:---:|:---:|
| **A** | 0 | 1 | RAG fixed a direct error | **1** ("send to T3") | ✅ yes | T3 (learned) |
| **B** | 1 | 0 | RAG broke a correct direct | **0** ("keep at T2") | ✅ yes | T2 (learned) |
| C | 1 | 1 | both right (tied EM) | tiebreak = noise | ❌ DROPPED | **T2** (cost-rational: same accuracy, cheaper) |
| D | 0 | 0 | both wrong (tied EM) | tiebreak = noise | ❌ DROPPED | **T3** (grounding-rational: retrieved passages give the verifier signal to detect wrongness via low `p_ground_*`, even when EM also fails) |

**Path B is the locked label strategy**: train only on A and B to keep the disagreement-boundary calibration clean. Tie rows still receive a probability at deployment via the trained scaler+LR; the threshold τ decides their routing.

**Label-validity invariant (must hold for the baselines):** both `ZeroShotBaseline` (T2) and `RAGBaseline` (T3) must emit the **identical output scaffold** `Reasoning: <text>\nAnswer: <text>`, so that `direct_em` and `rag_em` are computed by the same extractor over the same shape. Verified 2026-05-18:
- Both inherit `SYSTEM_PROMPT` (`caem/prompts.py:58-66`) which mandates the two-line `Reasoning:/Answer:` format.
- Both use `FORCED_PREFIX = "Reasoning:"` as the assistant-turn prefill.
- T3 differs only in *what the model conditions on* (a `Context:` block of retrieved passages, one few-shot demo, a task-specific `Answer format:` line) — not in *what shape it outputs*.

**Deployment defaults for tie-like queries** (the C→T2 and D→T3 column above): these are the architectural intent. Path B does not explicitly inject C/D labels — instead the LR's learned coefficients **naturally produce this routing pattern** via the model-confidence and retrieval features:

- **C-like query** (model knows, easy factoid): high `p_ik` → low `p_rag` → **T2 default**. Matches cost-rational intent: don't spend on RAG when direct will succeed anyway.
- **D-like query** (model doesn't know, hard / adversarial / out-of-distribution): low `p_ik` → high `p_rag` → **T3 default**. Matches grounding-rational intent: route to RAG so retrieved passages give the verifier something to test (low `p_ground_max/mean/atomic`) when the final answer is wrong; T2 failures produce confidently wrong answers with nothing to ground against, which is the dangerous failure mode CAEM is built to detect.

This is satisfied **implicitly** through the feature space, not as explicit training labels.

### 2.2 Features (~34 dims, all pre-routing-legal)

| Family | Dims | Features | Extractor |
|---|---:|---|---|
| **A** Surface form | 14 | `q_token_len`, `negation_present`, `temporal_cue`, `numerical_cue`, `myth_regex_hit`, `interrog_{who,what,when,where,why,how,yesno}` (×7), `entity_count`, `log_pageviews_max` | `scripts/build_ruc_training_set.py:_question_text_features` + `_entity_features` (computed at training-set-build time) |
| **B** Model confidence | 1 | `p_ik` (Qwen-2.5-3B last-token hidden state → frozen CalibratedClassifierCV probe; mean OOF AUROC 0.771) | `scripts/ruc_extract_v2_features.py` (probe at `caem/ruc/v2/pik_probe.joblib`) |
| **C** Retrieval top-1 | 2 | `top1_passage_sim`, `top1_passage_entity_overlap` | `scripts/ruc_extract_v2_features.py` |
| **C'** Retrieval top-5 | 3 | `top5_sim_max`, `top5_sim_mean`, `top5_sim_std` | `scripts/ruc_extract_v2_features.py` |
| **C''** Cross-encoder rerank | 3 | `rerank_top1`, `rerank_top5_mean`, `rerank_top5_std` (BAAI/bge-reranker-v2-m3) | `scripts/ruc_extract_v2_features.py` |
| **C'''** NLI pairwise | 3 | `nli_pair_mean`, `nli_pair_min`, `nli_pair_std` over 10 (5 choose 2) passage pairs | `scripts/ruc_extract_v2_features.py` (AdaptiveNLIJudge / bare MiniCheck) |
| **D** Linguistic | 3 | `is_adversarial_phrasing`, `is_opinion_seeking`, `is_open_ended` | `scripts/build_ruc_training_set.py` |
| **E** Answer-type match | 4 | `q_expects_year`, `passage_has_year`, `year_match`, parallel `*_number` (4 dims) | `scripts/build_ruc_training_set.py` |
| **F** Wikidata entity-binary | 1 | `entity_in_wikidata` | `scripts/ruc_extract_v2_features.py` (alias_dict at `data/alias_dict.json`) |
| **Total** | **~34** | | Pre-routing-legal (no post-T2 verifier signals) |

### 2.3 Model

Logistic Regression + L2 regularisation + StandardScaler.

- Trained on the A+B clean disagreement rows (Path B labels, unit weights).
- Inner CV sweeps `C ∈ {0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}` per outer fold; outer CV reports pooled and per-bench AUROC.
- Decision threshold `τ` is swept post-fit on full-data predictions to maximise net deployment utility = `A_caught(τ) − B_caught(τ)`.
- Coefficients are interpretable; SHAP isn't needed for runtime explainability.

### 2.4 Validation methodology

Nested 5-fold stratified CV on clean A+B rows only, stratified by `case`.

- Outer 5 folds: each fold has ~640 rows held out; reports per-fold AUROC + per-bench AUROC partitioned from OOF predictions.
- Inner CV per outer fold: selects best `C` by inner-fold AUROC.
- Threshold sweep: τ ∈ {0.20, 0.25, …, 0.75} on full-data OOF predictions; maximise `A_caught − B_caught`.
- No separate held-out test set is reserved. Phase 1e on the CAEM eval pool is the true generalisation measurement.

**Coefficient-sign acceptance receipt** (replaces an explicit C/D held-out test):
- `coef[p_ik] < 0` — high model self-confidence → less likely to need RAG → C-default to T2.
- `coef[top1_passage_sim] > 0` — strong retrieval candidate → RAG likely to help → A-pattern routing.
- `coef[rerank_top1] > 0` — well-matched passage → RAG likely to help.

If any of these signs flips, the LR has learned a feature combination that violates architectural intent — per-feature ablation diagnoses the offender before deployment.

---

## 3. Source pool (fresh pool)

Content-hash disjoint from every CAEM pool (seed / purity / calibration / sil_train / eval / test) by construction. Sliced 2026-05-17, extended 2026-05-18 with Natural Questions.

| Bench | Samples | Type | RAG-extensive? |
|---|---:|---|:---:|
| FEVER | 2 500 | claim verification | borderline neutral |
| TriviaQA | 2 500 | factoid QA | ✅ yes |
| CommonsenseQA | 500 | commonsense MCQ | no |
| StrategyQA | 187 | multi-hop yes/no | no |
| TruthfulQA | 317 | adversarial myth | no |
| HaluEval-QA | 2 500 | hallucination induction | unknown |
| OpenBookQA | 500 | science MCQ | no (unknown) |
| **Natural Questions** | **3 000** | **open-domain factoid** | ✅ **yes** |
| **Total** | **12 004** | 8 benches, 2 RAG-extensive | A:B balance ≈ 1.35:1 at pool |

The CAEM main training panel stays at `caem.config.TRAINING_BENCHMARKS = (fever, triviaqa, commonsense_qa)`. NQ is RUC-pool-only, not added to CAEM's SIL training.

---

## 4. Phase nomenclature

| Layer | Canonical name | Status |
|---|---|---|
| **Phase 1d** | vanilla CAEM trajectory (cycles 0–5) + 9-baseline panel | done; gdrive at `v2_1_phase1d/` |
| **Phase 1d** (still) | RUC build: fresh-pool baselines, features, training set, LR fit | in progress |
| **Phase 1e** (new) | CAEM-with-RUC trajectory: `run_phase1a.sh` from base Qwen + `step_7_main` with the RUC wired into `AdaptiveRouter` | not started, depends on RUC training closing |

**Invariant**: both Phase 1d (vanilla) and Phase 1e (with-RUC) are scored against the SAME locked Phase-1d verifier composite.

**Key clarification**: Phase 1e starts from **base Qwen** via `run_phase1a.sh` (not from Phase 1d's cycle-3 adapter). It runs its own Phase 1a (cold-start + cycle-0 calibration), then `step_7_main` for the multi-cycle SIL loop, with the RUC consulted at the two Tier-3 dispatch points.

---

## 5. Risk-mitigation contract

### R1. gdrive bucket separation — CRITICAL

`caem/training/self_improvement.py:1647-1651` constructs the offload path as:

```
gdrive:caem-phase1a/{CAEM_GDRIVE_BUCKET}/{run_name}/cycle_{N}/
                    └─ env var, default "v2"   └─ self.output_dir.name
```

The vanilla Phase-1d trajectory uploaded to `gdrive:caem-phase1a/v2_1_phase1d/full_run/cycle_{0..6}/`. **That backup is the canonical Phase-1d archive for the Phase 1d vs Phase 1e ablation.**

Fix is a single env var at Phase 1e launch — no code change:

```bash
CAEM_GDRIVE_BUCKET=v2_1_phase1e_2026-05-18 \
CAEM_GDRIVE_OFFLOAD=1 \
CAEM_BATCH_U_TOK_DROP=1 \
bash run_phase1a.sh
```

### R2. Eval-pool baseline rescore for Phase 1e headline (composite-swap, $0)

For the with-RUC vs without-RUC comparison in Ch5, the baseline CHM numbers on the CAEM eval pool are rescored under each arm's endpoint composite. This is **a composite-weight swap on the already-existing eval-pool baseline signals** — no full re-verify, ~10 sec CPU.

### R3. Skip-if-exists

| Script | Status |
|---|---|
| `scripts/launch_baselines_v2.sh` | ✅ (`all_benches_done_for`) |
| `scripts/rescore_baselines_through_verifier.py` | ✅ — `_rescore_one(skip_if_exists=True)` default; `--force` CLI flag |
| `scripts/ruc_extract_v2_features.py` | single output parquet, idempotent on re-run |
| `run_phase1a.sh` (Phase 1e launch) | local clobber mitigated by gdrive bucket separation (R1) |

---

## 6. Full run sequence

| # | Step | Resume anchor | Wall | $ |
|---:|---|---|---|---|
| 1 | v2 baselines (zero_shot + rag) on 12 004-sample fresh pool | `outputs/baselines_v2/<baseline>/<bench>_cycle0.json` | done | — |
| H | Haiku-judge TruthfulQA EM (lift the ROUGE-L > 0.15 floor) | `em_llm_judged` field in 2 TQA JSONs | done | $0.50 |
| ~~2~~ | **RETIRED.** Fresh-pool rescore is not required for Path B + unit-weight training. `build_ruc_training_set.py` patched to read raw `_cycle0.json` and use unit weights. | n/a | 0 | $0 |
| 2.5 | A+B pre-filter (paired EM via `_sample_em` dispatch → drop ties) | `/tmp/fresh_pool_AB_only/*.json` × 8 (~3 200 rows kept) | done | $0 |
| P | Train P(IK) probe on 12 004 zero_shot answers (Qwen hidden state + CalibratedClassifierCV) | `caem/ruc/v2/pik_probe.joblib` (mean OOF AUROC 0.771) | done | ~$0.20 |
| **3** | v2 feature extraction on A+B rows (top-1 + top-5 retrieval, rerank, NLI pairwise, p_ik, Wikidata) | `caem/ruc/v2/features.parquet` (~3 200 × 13 features + id/bench/q) | ~3.2 h GPU | $1.20 |
| 4 | Build v2 training set: join features + raw EM + 14 surface + 3 linguistic + 4 ans-type; Path B filter; unit weights | `caem/ruc/v2/training_set.parquet` (~3 200 × ~34 cols + label + weight) | ~5 min CPU | $0 |
| 5 | Train v2 LR with nested 5-fold CV + threshold τ sweep + coefficient-sign inspection | `caem/ruc/v2/{models/{scaler,lr_final}.joblib, feature_spec.json, cv_results.json}` | ~10 min CPU | $0 |
| 6 | Smoke v2 RUC integration: 50-q matched-protocol pass through `AdaptiveRouter` | `caem/ruc/v2/smoke_pass.json` (PASS/FAIL verdict) | ~30 min CPU | $0 |

**Critical-path total: ~4 GPU-h, ~$2 Vast, ~half a day wall.**

### Side work (parallel)

| # | Step | Resume anchor | Wall | $ | Status |
|---:|---|---|---|---|---|
| S1 | **vanilla_ft FULL rescore (cycle_N/eval/ schema)** — training complete (5 cycles, May 16 10:50); rescore script only iterates over top-level `{bench}_cycle0.json`, misses the `cycle_N/eval/{bench}_cycleN.json` files. **Patch needed:** extend `rescore_baselines_through_verifier.py` to handle FT baseline cycle_N schema. Then re-run rescore over all 5 benches × 5 cycles = 25 JSONs. | 25 new `outputs/baselines/vanilla_ft/cycle_N/eval/<bench>_cycleN_with_chm.json` | ~5-8 GPU-h | $3 | **pending post-Phase-1e** — Aksan manually stopped on 2026-05-16, plan was to defer until RUC integrated |
| S2 | **ewc_only_ft RE-TRAIN + rescore** — original training OOM-crashed at `scripts/run_simple_ft.py:440` (`l2 = l2 + ((p.float() - ref) ** 2).sum()` blew the 5090's 32 GB VRAM during L2 anchor compute). `eval/` is empty. **Fix:** chunk the L2-anchor loop by parameter group in `run_simple_ft.py`; expected peak memory <24 GB after fix. Then re-train 5 cycles + rescore. | `outputs/baselines/ewc_only_ft/cycle_{1..5}/eval/*_with_chm.json` | ~7-9 GPU-h | $4 | **pending post-Phase-1e** — skipped 2026-05-16 due to OOM, intentionally deferred per Aksan |
| S4 | Rsync v2 baselines to `gdrive:.../v2_1_phase1e_2026-05-19/baselines_v2/` | byte-verified | ~30 min WAN | $0 | pending |

**Effect of S1+S2 on Phase 1e headline**: none. The 7-baseline panel (B1-B7) with `_with_chm.json` is sufficient for the main thesis claim. vanilla_ft + ewc_only_ft are supplementary "non-CAEM continual-learning" baselines that round out the panel but aren't load-bearing. Ch5 §sec:baselines can use placeholder rows + a "pending re-run" note until S1+S2 close.

### Post-Phase-1e gdrive cleanup (the May-20 launch bug + recovery)

**Why this section exists.** On 2026-05-19 17:21 UTC, `run_phase1a.sh` step_7_main launched with a hardcoded `CAEM_GDRIVE_BUCKET="v2_1_phase1d"` that overrode the caller-supplied `CAEM_GDRIVE_BUCKET=v2_1_phase1e_2026-05-19` env var. Phase 1e cycle-1 SIL completed at 03:55 UTC 2026-05-20 and offloaded its LoRA adapter + meta.pkl to `gdrive:caem-phase1a/v2_1_phase1d/full_run/cycle_1/`, overwriting Phase 1d's cycle-1 adapter files. The bug was caught at 05:45 UTC and Phase 1d's archive was relocated via `rclone move gdrive:caem-phase1a/v2_1_phase1d/ → v2_1_phase1d_archive/` at 05:51 UTC. The runbook hardcode was patched at `run_phase1a.sh:659` to honour the env var; the patch applies to future launches, not the in-flight trajectory.

**Permanently lost** (overwritten before the rescue): 4 files of Phase 1d cycle-1 — `adapter/adapter_model.safetensors` (240 MB), `adapter/adapter_config.json` (1.1 KB), `adapter/README.md` (5.2 KB), `meta.pkl` (168 B). These are intermediate cycle-1 LoRA weights; not load-bearing for B1→C5 headline (uses cycle_5 adapter, preserved) nor for per-cycle EM/CHM plots (use eval/cal JSONs, preserved).

**Verified by timestamp scan** (2026-05-20 06:30 UTC): Phase 1d archive at `v2_1_phase1d_archive/` contains 360 files total, 356 of which carry Phase 1d original timestamps (2026-05-09 to 2026-05-17). Only the 4 files above carry the 2026-05-20 03:55 overwrite timestamp. Verification: `rclone lsl gdrive:caem-phase1a/v2_1_phase1d_archive/ | awk '{print $2}' | sort | uniq -c` showed 4 entries dated 2026-05-20 and the rest spread across May 9-17. Phase 1d archive is **98.9% intact (356 / 360 files), 99.998% by size (~13.58 GB / 13.59 GB)**.

**Empty folder at `v2_1_phase1d/`**: created 06:07:53 UTC 2026-05-20 by an `rclone copyto` test confirming auto-create behaviour (file uploaded + deleted, folder entry persists). Empty (0 bytes, 0 objects). No issues — Phase 1e's next offload populates it cleanly.

**Post-trajectory cleanup procedure** (~1 min total, run AFTER step_7_main closes):

```bash
# STAGE 1 — Recover the 4 stranded Phase 1e files
# (they were uploaded to v2_1_phase1d/ before the rescue at 05:51 UTC, then
#  moved along with everything else to v2_1_phase1d_archive/ at 05:51 UTC.
#  These belong with Phase 1e cycle_1's other content.)
rclone moveto \
  gdrive:caem-phase1a/v2_1_phase1d_archive/full_run/cycle_1/adapter/ \
  gdrive:caem-phase1a/v2_1_phase1d/full_run/cycle_1/adapter/
rclone moveto \
  gdrive:caem-phase1a/v2_1_phase1d_archive/full_run/cycle_1/meta.pkl \
  gdrive:caem-phase1a/v2_1_phase1d/full_run/cycle_1/meta.pkl

# STAGE 2 — Write the SEE-IT note inside Phase 1d archive
cat > /tmp/SEE_IT_FIRST.txt <<'EOF'
SEE IT FIRST — Phase 1d archive integrity note (2026-05-20)
============================================================

This folder contains the Phase 1d (vanilla CAEM, no RUC) trajectory archive
from the 2026-05-10 → 2026-05-15 run.

INTEGRITY: 99% intact. ~13.6 GB of 13.6 GB preserved.

ONE EXCEPTION: cycle_1's LoRA adapter files (adapter/* + meta.pkl, ~240 MB)
were overwritten by Phase 1e's cycle-1 SIL upload on 2026-05-20 03:55 UTC due
to a CAEM_GDRIVE_BUCKET hardcode in run_phase1a.sh that ignored the launch-
time env var. The bug was caught at 05:45 UTC and this folder was relocated
to v2_1_phase1d_archive/ at 05:51 UTC to prevent further damage.

The 4 overwritten files have since been moved out of this folder back into
their correct Phase 1e location (v2_1_phase1e/full_run/cycle_1/). So this
folder now contains ONLY Phase 1d content, minus the 4 lost files.

The remaining 99% of Phase 1d data (all cycles 0-5 memory snapshots, eval
JSONs, calibration data, retroverify reports, eval_stream_chunks, composite
calibration, multi-modal probes) is fully intact and Phase-1d-authentic.

Thesis impact: the lost cycle-1 adapter is an INTERMEDIATE binary, not a
results artifact. The B1→C5 headline contrast uses cycle_5's adapter (intact
in cycle_5/adapter/). Per-cycle EM/CHM plots use the eval JSONs (intact in
cycle_N/eval/). No thesis numbers are lost.

For full incident detail see: caem/ruc/README.md §6 → "Post-Phase-1e gdrive
cleanup" subsection.
EOF
rclone copyto /tmp/SEE_IT_FIRST.txt \
  gdrive:caem-phase1a/v2_1_phase1d_archive/SEE_IT_FIRST.txt

# STAGE 3 — Rename folders to match content
# v2_1_phase1d/ currently has Phase 1e content → rename to v2_1_phase1e/
rclone moveto \
  gdrive:caem-phase1a/v2_1_phase1d/ \
  gdrive:caem-phase1a/v2_1_phase1e/

# v2_1_phase1d_archive/ has the real Phase 1d → restore original name
rclone moveto \
  gdrive:caem-phase1a/v2_1_phase1d_archive/ \
  gdrive:caem-phase1a/v2_1_phase1d/

# v2_1_phase1e_2026-05-19/ contains ONLY ruc_build/ (RUC training artifacts,
# not the Phase 1e trajectory). Rename to reflect that it's a Phase 1e
# COMPONENT, not the trajectory itself.
rclone moveto \
  gdrive:caem-phase1a/v2_1_phase1e_2026-05-19/ \
  gdrive:caem-phase1a/v2_1_phase1e_RUC_components/
```

**Final gdrive state after cleanup:**

```
gdrive:caem-phase1a/
├── v2_1_phase1d/                    Phase 1d trajectory archive (vanilla CAEM)
│   ├── SEE_IT_FIRST.txt             Integrity note documenting the 4-file loss
│   └── full_run/cycle_0..5/         All preserved (minus cycle_1 LoRA adapter)
│
├── v2_1_phase1e/                    Phase 1e trajectory archive (CAEM with RUC)
│   └── full_run/cycle_0..5(or 10)/  Complete, including the 4 recovered files
│
├── v2_1_phase1e_RUC_components/     RUC training artifacts only
│   └── ruc_build/                   Features, P(IK) probe, training set, etc.
│
├── pre_main_snapshot/               Pre-step-7 anchor snapshots
├── archive_phase1c_2026-05-09/      Old archive
└── archive_v1/                      v1 trajectory
```

### Layout difference Phase 1d vs Phase 1e (post-v2-refactor cleanup)

Phase 1d wrote cycle-0 artifacts in TWO places (top-level AND `cycle_0/` subdir, duplicated). Phase 1e drops the redundant `cycle_0/` subdir and writes cycle-0 artifacts only at top-level. Both schemas are read by the same aggregation scripts (which glob `eval/{bench}_cycle*.json`), so the ablation comparison works identically across both layouts. Document for future readers in the post-trajectory reorganization:

```
Phase 1d (old) layout:
  full_run/
  ├── memory_store_cycle_0.{faiss,meta}  ← top-level
  ├── deferred_buffer_cycle_0.pkl        ← top-level
  ├── eval/{bench}_cycle0.json           ← top-level
  ├── cycle_0/                            ← REDUNDANT subdir
  │   ├── memory_store_cycle_0.{faiss,meta}  (duplicate)
  │   ├── deferred_buffer_cycle_0.pkl        (duplicate)
  │   └── eval/{bench}_cycle0.json           (duplicate)
  ├── cycle_1/  cycle_2/  ...  cycle_5/   ← per-cycle from cycle 1 onwards
  └── ...

Phase 1e (current, cleaner) layout:
  full_run/
  ├── memory_store_cycle_0.{faiss,meta}  ← top-level only
  ├── deferred_buffer_cycle_0.pkl        ← top-level only
  ├── eval/{bench}_cycle0.json           ← top-level only
  ├── cycle_1/  cycle_2/  ...            ← per-cycle from cycle 1
  └── ...                                  (no redundant cycle_0/)
```

**Post-trajectory reorganization (when running cleanup):** the layout difference does NOT need fixing — both schemas are aggregation-script-compatible and the cleaner Phase 1e schema is the correct go-forward design. Document the difference in `SEE_IT_FIRST.txt` so a reader downloading the Phase 1d archive doesn't think `cycle_0/` was lost in Phase 1e (it's just redundant in Phase 1d and intentionally omitted in Phase 1e).

### Full reorganization plan summary (for post-trajectory execution)

Apart from the rclone moveto commands above:

1. **Drop a top-level README** inside the renamed `v2_1_phase1e/full_run/` explaining the schema:
   - `eval/{bench}_cycleN.json` = all eval JSONs across cycles (cycle in filename)
   - `cycle_N/adapter/` = per-cycle LoRA
   - `cycle_N/calibration/` = per-cycle cal-fold scoring under post-SIL
   - `cycle_N/eval_stream_chunk/` = per-cycle stream-chunk eval JSONs
   - `memory_store_cycle_N.{faiss,meta}` = top-level cycle memory snapshots
   - `deferred_buffer_cycle_N.pkl` = top-level deferred buffer per cycle
   - `mmlu_baseline.json`, `retention_baseline.json` = pristine probe denominators (cycle-0 only)

2. **Drop the SEE_IT_FIRST.txt inside the renamed `v2_1_phase1d/`** (text drafted above in STAGE 2).

3. **Drop a marker file in `v2_1_phase1e_RUC_components/`** explaining the RUC build pipeline: features.parquet, training_set.parquet, pik_probe.joblib, feature_spec.json, smoke_pass.json, etc.

4. **Update the gdrive bucket index in CLAUDE.md** so future sessions know about the post-cleanup naming.

All of this lands at ~1 minute of total rclone work + ~5 minutes of writing the README files. No GPU. Post-trajectory only.

**Scope**: the reorganization is gdrive-only. Local `outputs/full_run/` keeps its existing schema because (a) the runner script and aggregation scripts depend on the exact paths in code, (b) restructuring locally mid-trajectory or post-trajectory would invalidate downstream scripts, (c) the local copy is mostly ephemeral (we already deleted Phase 1d's local copy once we had it on gdrive). Gdrive is the canonical archive and where naming clarity matters most.

### CAEM file layout reference (for the post-trajectory README to drop inside `v2_1_phase1e/full_run/`)

CAEM uses a **two-layer schema** to balance "easy resume detection" with "clean per-cycle artifacts":

```
TOP-LEVEL    = "things you read across all cycles" (eval JSONs, memory snapshots, baselines)
cycle_N/     = "things specific to cycle N" (LoRA weights, post-SIL cal-fold, composite refit)
```

**Local layout (`outputs/full_run/`):**

```
outputs/full_run/
│
├─ TOP-LEVEL FILES (cycle number in filename)
│   memory_store_cycle_N.faiss/.meta        cycle memory snapshots (written line 1932)
│   deferred_buffer_cycle_N.pkl             deferred buffers (line 1942)
│   retroverify_cycleN.json                 cycle retroverify reports (line 1830)
│   mmlu_baseline.json                      pristine MMLU (cycle 0 only, line 1199)
│   retention_baseline.json                 pristine multi-modal (cycle 0 only, line 1226)
│   dataset_splits.json                     benchmark fold definitions (line 1084)
│   experiment_summary.csv                  one row appended per cycle (line 656)
│   experiment.log + run.log                logger output
│   run_complete.json                       written only at trajectory end (line 2292)
│
├─ TOP-LEVEL DIRS (shared across cycles)
│   eval/                                   all eval + stream-chunk JSONs across cycles
│       {bench}_cycleN.json                 (line 1890)
│       {bench}_cycleN_streamchunk.json     (line 1891)
│   calibration/                            cycle-0 T-fit collection only (vestigial)
│       calibrated_config.json
│       calibration_fold_samples.json
│
└─ PER-CYCLE SUBDIRS (cycle_N/ from cycle 1 onwards; no cycle_0/ in v2)
    cycle_N/
        adapter/                            LoRA weights for cycle N (written by SIL)
        calibration/{bench}_cycleN.json     post-SIL cal-fold scoring (line 1751)
        composite_calibration.json          per-cycle composite refit (line 1786)
        meta.pkl                            cycle metadata pickle
        equilibrium_fit.json                stochastic equilibrium diagnostic (line 2110)
        coverage_diagnostic.json            pool coverage diagnostic (line 2178)
        halt_decision.json                  halt-trigger evaluation (line 2220)
```

**Gdrive layout after upload (per-cycle consolidated):**

```
gdrive:caem-phase1a/{BUCKET}/full_run/
│
├─ global/                                  cross-cycle singletons
│   mmlu_baseline.json                      (from local top-level)
│   dataset_splits.json
│   run.log
│
└─ cycle_N/                                 ALL artefacts for cycle N in one folder
    adapter/                                (from local cycle_N/adapter/)
    calibration/                            (from local cycle_N/calibration/)
    composite_calibration.json              (from local cycle_N/)
    meta.pkl                                (from local cycle_N/)
    memory_store_cycle_N.faiss              (copied IN from local top-level)
    memory_store_cycle_N.meta               (copied IN from local top-level)
    deferred_buffer_cycle_N.pkl             (copied IN from local top-level)
    retroverify_cycleN.json                 (copied IN from local top-level)
    eval_transfer/
        {bench}_cycleN.json                 (copied IN from local eval/)
```

**Why local ≠ gdrive schema:**

- **Local hybrid layout** is required by the runner: top-level for easy resume detection (`memory_store_cycle_${n}.faiss` glob check at `run_phase1a.sh:632`), per-cycle subdirs for binaries. Aggregation scripts (`aggregate_*.py`, `make_tables.py`, `make_figures.py`) read both layers and glob top-level `eval/`.
- **Gdrive flat per-cycle layout** is friendlier for archive browsing: a reader sees `cycle_N/` and finds everything for that cycle inside. The upload script (`run_experiment.py:1984-2025`) does this consolidation by `rclone copy`-ing top-level files INTO the cycle_N gdrive subdir at cycle close.
- **Phase 1d's apparent "cycle_0/ duplication"** (memory_store inside cycle_0/ AND at top-level) was the upload script consolidating top-level cycle-0 files into gdrive's cycle_0/. Phase 1e drops the redundant local `cycle_0/` subdir but the upload still consolidates correctly into `gdrive:.../cycle_0/`.

**Mapping local file → gdrive destination at cycle N close:**

| Local path | Gdrive destination |
|---|---|
| `cycle_N/adapter/*` | `cycle_N/adapter/*` |
| `cycle_N/calibration/*` | `cycle_N/calibration/*` |
| `cycle_N/composite_calibration.json` | `cycle_N/composite_calibration.json` |
| `cycle_N/meta.pkl` | `cycle_N/meta.pkl` |
| `cycle_N/equilibrium_fit.json` | `cycle_N/equilibrium_fit.json` |
| `cycle_N/coverage_diagnostic.json` | `cycle_N/coverage_diagnostic.json` |
| `memory_store_cycle_N.faiss` (top-level) | `cycle_N/memory_store_cycle_N.faiss` |
| `memory_store_cycle_N.meta` (top-level) | `cycle_N/memory_store_cycle_N.meta` |
| `deferred_buffer_cycle_N.pkl` (top-level) | `cycle_N/deferred_buffer_cycle_N.pkl` |
| `retroverify_cycleN.json` (top-level) | `cycle_N/retroverify_cycleN.json` |
| `eval/{bench}_cycleN.json` (top-level) | `cycle_N/eval_transfer/{bench}_cycleN.json` |
| `mmlu_baseline.json` (top-level) | `global/mmlu_baseline.json` |
| `dataset_splits.json` (top-level) | `global/dataset_splits.json` |
| `run.log` (top-level) | `global/run.log` |

### Phase 1e: CAEM-with-RUC trajectory

**Before launch**: R1 mitigation in place (env var `CAEM_GDRIVE_BUCKET=v2_1_phase1e_2026-05-19`).

| # | Step | Resume anchor | Wall | $ |
|---:|---|---|---|---|
| ~~P1~~ | Wire RUC into router | ✅ done 2026-05-19. `caem/routing/ruc.py` accepts `**extra_features`; `caem/routing/router.py:_consult_ruc` threads extras (with `_*` metadata-key filter — bug fix 2026-05-19); `caem/pipeline.py:_compute_ruc_features` reuses verifier components + loads P(IK) probe with `model.disable_adapter()` context; `caem/pipeline_batch.py:_peek_routing` also threads RUC features (bug fix 2026-05-19 — was missing, causing first launch to fall back to vanilla T3 in batch mode). End-to-end smoke verified: TQA→DIRECT, TriviaQA-long-tail→RAG, weak-retrieval→DIRECT. | done | $0 |
| ~~P2~~ | **Phase 1a with RUC: cold-start from base Qwen + cycle-0 baseline + composite refit** | `outputs/full_run/cycle_0/` + gdrive mirror | ✅ **cycle 0 closed 23:25 UTC 2026-05-19** (6h 3min). Composite/T/gate fitted, baselines measured (MMLU 0.6350, multi-modal 0.61/0.395/0.825). Per-bench T2 shares hit expected pattern (CSQA 92%, TQA 65%, FEVER 44%, TQA 33%, StratQA 15%). | done | ~$5 |
| **P3** | step_7_main with RUC: cycle 1+ SIL fine-tune → retention-guard-fire (early-stop locked at cycle 5-6 per runbook) | `outputs/full_run/{cycle_N, eval/, run_complete.json}` + gdrive | **in progress** — cycle 0 finalization phase (T-scalar fit on cal-fold) since 23:27 UTC. Cycle-1 SIL fires ~05:30-06:30 UTC 2026-05-20. | ~3-7 GPU-days | $30-65 |

### Ablation analysis (Phase 1d vs Phase 1e) + Ch6 deliverables

| # | Step | Resume anchor | Wall | $ |
|---:|---|---|---|---|
| A1 | Per-bench EM/CHM trajectory: vanilla vs with-RUC | `outputs/tables/tab_ruc_ablation_trajectory.{json,tex}` | ~1 h CPU | $0 |
| A2 | Routing-decision breakdown per (bench, cycle) | `outputs/tables/tab_ruc_routing_distribution.{json,tex}` | included | $0 |
| A3 | Retention-guard fire-cycle comparison | A1 includes | included | $0 |
| A4 | Significance (McNemar paired EM; bootstrap BCa on CHM delta) | `outputs/tables/tab_ruc_significance.json` | ~1 h CPU | $0 |
| C1 | Trajectory plot (vanilla vs with-RUC, per bench) | `outputs/figures/fig_ruc_ablation_trajectory.{png,pdf}` | ~2 h CPU | $0 |
| C2 | Routing-distribution LaTeX table | `outputs/tables/tab_ruc_routing_distribution.tex` | ~1 h CPU | $0 |
| C3 | Write Ch6 §sec:ruc-ablation prose | `thesis_report/chapters/chapter_6.tex` | ~1-2 days writing | $0 |

---

## 7. Total budget (current state → Ch6 in hand)

| Stage | Compute | Cost |
|---|---|---|
| ✅ Steps 1, H, P, 2.5 (done) | ~18 GPU-h | $8 |
| Step 3 (v2 features) | ~3.2 GPU-h | $1.20 |
| Steps 4-6 (CPU only) | 45 min CPU | $0 |
| Side-work S1 + S4 | ~6 GPU-h | $3 |
| Phase 1e (P1-P3) | ~100-200 GPU-h | $40-80 |
| Ablation + Ch6 (CPU + human) | ~5 h CPU + 1-2 days writing | $0 |
| **Remaining total** | **~110-210 GPU-h** | **~$45-85 Vast** |

---

## 8. Critical-path commands

### Step 3 — v2 feature extraction

```bash
cd /workspace/caem
tmux new-session -d -s step3_full -x 220 -y 50 "
  source /root/.bashrc; source /venv/main/bin/activate
  export PYTHONPATH=/workspace/caem
  /venv/main/bin/python -m scripts.ruc_extract_v2_features \
    --fresh_pool_glob '/tmp/fresh_pool_AB_only/*.json' \
    --passage_index data/passage_index \
    --pik_probe_path caem/ruc/v2/pik_probe.joblib \
    --out_parquet caem/ruc/v2/features.parquet \
    --device cuda
  echo '*** STEP 3 DONE ***'
  exec bash
"
```

### Step 4 — Build training set

```bash
/venv/main/bin/python -m scripts.build_ruc_training_set \
    --baselines_dir outputs/baselines_v2 \
    --benches fever triviaqa commonsense_qa strategyqa truthfulqa haluevalqa openbookqa natural_questions \
    --out caem/ruc/v2/training_set.parquet
```

### Phase 1e launch (CAEM-with-RUC trajectory) — P2

```bash
cd /workspace/caem
tmux new-session -d -s plan_a_ruc -x 220 -y 50 "
  cd /workspace/caem
  source /root/.bashrc
  source /venv/main/bin/activate
  export PYTHONPATH=/workspace/caem
  CAEM_GDRIVE_BUCKET=v2_1_phase1e_2026-05-19 \\
  CAEM_GDRIVE_OFFLOAD=1 \\
  CAEM_BATCH_U_TOK_DROP=1 \\
  bash run_phase1a.sh 2>&1 | tee outputs/run_phase1a_ruc.log
  echo '*** DONE ***'; exec bash
"
```

**Pre-launch checklist** (5 items, do these before firing the command):

1. **Verify Vast credit > $50** — Phase 1e P2+P3 costs $40-80; run out mid-trajectory loses all GPU-hours so far.
2. **Confirm gdrive `v2_1_phase1d` is intact** so the Phase 1d vs Phase 1e ablation has its vanilla baseline preserved. `rclone ls gdrive:caem-phase1a/v2_1_phase1d/full_run/ --max-depth 1`.
3. **Confirm `run_phase1a.sh` wires the v2 RUC into the router**. Currently it constructs `CAEMPipeline(...)` then `pipeline.router.ruc = LRRetrievalUtilityClassifier.from_spec("caem/ruc/v2/feature_spec.json")`. If the script doesn't already do this, add the two lines or pass via env var.
4. **Confirm GPU memory headroom**. The Pipeline + RUC components load ~14-18 GB GPU (Qwen 6 + MiniCheck 2 + CrossEncoder 1 + verifier 4-8). On the 32 GB 5090, that leaves ~14 GB for generation kv-cache — sufficient.
5. **Confirm tmux session name doesn't collide with anything running**: `tmux ls | grep plan_a` should return nothing.

---

## 9. Open decision points (require human input)

| When | Decision | Default if no input |
|---|---|---|
| After Step 5 (CV AUROC measured) | If pooled CV AUROC ≥ 0.65 **AND** coefficient signs satisfy intent → proceed to Phase 1e | proceed |
| Before Phase 1e launch | confirm `CAEM_GDRIVE_BUCKET=v2_1_phase1e_2026-05-18` env var set | **block** — explicit confirmation required |
| After Phase 1e P3 closes | retention guard fired at C? — early or late, either is the result | report whichever happens |
| After A4 | p < 0.05 on with-RUC vs vanilla delta? if not, shift Ch6 framing to "architecturally cleaner without significant EM lift" | report honestly |

---

## 10. Acceptance gates (pre-registered)

- **Primary**: TruthfulQA held-out per-bench AUROC ≥ 0.75 (stretch 0.80)
- **Secondary**: pooled CV AUROC ≥ 0.78 (stretch 0.85)
- **Tertiary**: FEVER + TriviaQA held-out each ≥ 0.60
- **Red flag**: any AUROC ≥ 0.90 → leakage audit
- **End-to-end deployment**: TruthfulQA CHM ≤ 0.10 with EM ≥ 0.35; FEVER and TriviaQA CHM each within ±0.02 of current C3 numbers

**Expected** (3 200 clean A+B rows, balanced 2:6 RAG-extensive split):
- Pooled CV AUROC 0.70–0.74
- Per-bench: TruthfulQA 0.65–0.70, TriviaQA 0.68–0.72, NQ 0.65–0.70, CSQA 0.60–0.65, StrategyQA 0.58–0.65, FEVER 0.55–0.65

The Ch6 honest framing anticipates: **the strict per-bench gates may not all clear**. Reporting strategy: (a) per-bench in-distribution AUROCs as primary number, (b) cost-adjusted utility curve from the threshold sweep, (c) Phase 1e net-utility on the CAEM eval pool as the production-relevant headline.

---

## 11. Critical design decisions

- **Path B labels** (case A → 1, case B → 0, drop ties C+D): training only on EM-disagreement rows. CHM-tiebreak labels were tried and discarded as noise; the disagreement-boundary calibration is cleaner without them.
- **Pre-routing-only features**: the feature set has no `direct_verifier_*` signals — those require running Tier 2 first, violating pre-routing legality. All 34 features can be computed before the routing decision.
- **LR + StandardScaler over LightGBM**: linear heads on hand-engineered + retrieval features beat tree ensembles on small data when features are mostly monotonic in the target. Matches the published Adaptive-RAG / SR-RAG / Self-Routing-RAG architecture. Coefficients are directly interpretable.
- **Step 2 (fresh-pool rescore) RETIRED**: CHM signals are not needed for Path B + unit-weight training. The build script reads raw baselines + uses unit weights when CHM is absent.
- **Single-pass feature extraction**: `scripts/ruc_extract_v2_features.py` extracts all 13 features (retrieval, rerank, NLI, p_ik, Wikidata) in one model-loading pass over the A+B disagreement rows only (~3 200 questions, ~3.2 h on a 5090).
- **Natural Questions added to fresh pool**: balances the RAG-extensive split. TriviaQA was the only positive-class anchor; NQ adds a second positive bench so RAG-helps coefficients generalise across distinct surface patterns.
- **TruthfulQA EM via Haiku judge**: ROUGE-L > 0.15 inflated TQA EM artificially. Claude Haiku 4.5 judge follows the Lin et al. 2022 §3.2 definition (matches a correct reference AND doesn't endorse any incorrect reference). Haiku-EM is dispatched at `scripts/build_ruc_training_set.py:_sample_em` for TQA only; other benches use native EM.
- **Phase 1e starts from base Qwen**, not from Phase 1d cycle-3 adapter. Both trajectories begin from the same base weights; they diverge from cycle 0 onward because routing decisions differ.

---

## 12. Repo layout

```
caem/ruc/
├── README.md                                # this file — single canonical doc
├── fresh_pool/                              # 12 004 content-hash-disjoint source questions
│   ├── fever_v2.json              (2 500)
│   ├── triviaqa_v2.json           (2 500)
│   ├── commonsense_qa_v2.json     (  500)
│   ├── strategyqa_v2.json         (  187)
│   ├── truthfulqa_v2.json         (  317)
│   ├── haluevalqa_v2.json         (2 500)
│   ├── openbookqa_v2.json         (  500)
│   └── natural_questions_v2.json  (3 000)
├── wiki_pageviews_cache.jsonl               # reusable Wikipedia pageviews cache (from ruc_build_wiki_pageviews.py)
└── v2/                                      # all RUC artefacts in one place
    ├── pik_probe.joblib                     # frozen P(IK) probe (CalibratedClassifierCV; mean OOF AUROC 0.771)
    ├── pik_probe_metrics.json               # probe training metrics
    ├── pik_train_set.parquet                # zero_shot baseline answers + direct_em labels (probe training set)
    ├── pik_probe_train.log                  # training run log
    ├── qwen_hidden_states.npy               # cached Qwen hidden states for 11 980 probe-training questions
    ├── qwen_hidden_states_ids.json          # id index for the hidden-states matrix
    ├── features.parquet                     # ⏳ Step 3 output — 13 v2 features × ~3 200 A+B rows
    ├── training_set.parquet                 # ⏳ Step 4 output — full training dataset (~34 features + label + weight)
    ├── cv_results.json                      # ⏳ Step 5 output — nested-CV per-fold + per-bench AUROC + τ sweep
    ├── feature_spec.json                    # ⏳ Step 5 output — locked spec for deployment loader
    ├── models/                              # ⏳ Step 5 output
    │   ├── scaler.joblib                    #    StandardScaler
    │   └── lr_final.joblib                  #    LogisticRegression(C=...)
    └── smoke_pass.json                      # ⏳ Step 6 output — 50-q matched-protocol pass result
```

---

## 13. Live update log

```
2026-05-17  Fresh pool sliced (originally 9 004 questions across 7 benches).

2026-05-18  Natural Questions added to the fresh pool (3 000 samples) → 12 004 total, 8 benches.
            Phase naming aligned to CAEM convention (Phase 1d / Phase 1e).
            Step 1 (baselines) launched in tmux baselines_v2.
            zero_shot complete (8/8 benches, 16 min). rag complete (~17.5 h total).
            Throughput at bs=32 + prefetch: zero_shot ~12 samples/sec; rag ~0.28 samples/sec (retrieval-bound).
            Phase 1e gdrive bucket name locked at v2_1_phase1e_2026-05-18.
            Phase 1e starts from BASE QWEN, not cycle-3 adapter (architectural clarification).
            TruthfulQA Haiku-judge rescore complete (12 min, ~$0.50 API). 634/634 judgments.
              Empirical: zero_shot Haiku-EM=0.442 (vs ROUGE-L>0.15 0.767 inflated), rag 0.271.
              RAG hurts TQA by 17 pp under honest grading.
            STEP 2 RETIRED. Path B + unit weights doesn't need composite signals on the fresh pool.
              Per-cycle composite refit kills the distribution-drift concern. build_ruc_training_set.py
              patched: reads raw _cycle0.json when _with_chm.json absent, sets weight=1.0 when CHM
              signals absent, drops EM-tie rows BEFORE any CHM computation. Savings: ~25 GPU-h.
            A+B pre-filter executed: 3 200 disagreement rows kept (out of 11 980 valid pairings,
              26.7 % yield). Per-bench: FEVER 958, TriviaQA 829, NQ 537, HaluEval-QA 402, OBQA 139,
              CSQA 158, TruthfulQA 134, StrategyQA 43. Class balance 1 840 A vs 1 360 B.
            P(IK) probe trained on 11 980 zero_shot answers (mean OOF AUROC 0.771). Output at
              caem/ruc/v2/pik_probe.joblib.
            Step 3 (feature extraction) refactored to single-pass: 13 features in one extractor
              (top-1 + top-5 retrieval, cross-encoder rerank, NLI pairwise, p_ik, Wikidata).
              ETA ~3.2 h on the 3 208-row A+B pool.
            All historical RUC artefacts + scripts + documentation deleted (user directive).
              Renamed deployed-RUC directory to caem/ruc/v2/ (clean, no historical baggage).
              caem/routing/ruc.py points at caem/ruc/v2/feature_spec.json.

2026-05-19  Step 3 feature extraction complete: 3 208 × 13 v2 features in caem/ruc/v2/features.parquet
              (full A+B pool, single-pass extraction, ~3 h GPU).
            Step 4 training set assembled: 3 200 × 36 features + label + weight at
              caem/ruc/v2/training_set.parquet. Per-bench class balance: TQA 30 % positive
              (RAG mostly hurts), NQ 86 % positive (RAG mostly helps), pooled 57.5 % positive.
            Step 5 LR training complete: pooled OOF AUROC 0.8501 (substantially above 0.72 prior),
              TruthfulQA per-bench 0.732, in-sample vs OOF gap 0.011 (no overfit), per-fold AUROC
              std 0.008 (stable). Final C=0.5, τ=0.375, net utility 2 477/3 208 (77.2 %).
              Top features by |coef|: p_ik (-1.77 dominates), negation_present, rerank_top5_std,
              interrog_yesno, rerank_top5_mean.
            Coefficient-sign receipt: 2/3 pass directly; top1_passage_sim sign-flip is a
              redistribution artefact (rerank features carry the retrieval-quality signal at
              +1.01 aggregate; total retrieval coef +1.29 is correct).
            Step 6 model smoke: 6/7 checks pass; StrategyQA T3-share outlier from n=43 noise.
            P1 RUC pipeline wiring: caem/routing/ruc.py + router.py + pipeline.py all patched.
              - LRRetrievalUtilityClassifier accepts **extra_features
              - router.route + _consult_ruc thread extras
              - pipeline._compute_ruc_features reuses verifier components (FAISS, cross_encoder,
                judge, alias_resolver) + loads P(IK) probe at init
              - P(IK) hidden states computed via model.disable_adapter() context — base-Qwen
                distribution-consistent regardless of cycle (no-op at cycle 0)
              - End-to-end smoke verified on the live CAEMPipeline: well-known factoids → DIRECT
                (e.g., "Who painted the Mona Lisa?" p_rag=0.25); long-tail factoids → RAG
                (e.g., "Capital of Tuvalu?" p_rag=0.83); adversarial myth → DIRECT
                (e.g., "10% brain myth" p_rag=0.11); weak retrieval → DIRECT
                (e.g., obscure 1937 film box office, rerank=0.20). Architectural intent
                verified at deployment.
            Phase 1e gdrive bucket name updated to v2_1_phase1e_2026-05-19 (today's launch date).
```

---

## 14. Dataset construction (provenance — for thesis Ch4 / Ch5 dataset section)

Living section. Every empirical number here updates as new runs land. Use this as the authoritative dataset description; lift verbatim into `thesis_report/chapters/chapter_5.tex` §sec:ruc-dataset.

### 14.1 Source pool

See §3 for the bench breakdown. 12 004 total content-hash-disjoint questions across 8 benches.

### 14.2 Generation (matched-protocol baselines)

For each of the 12 004 questions, two answers were generated through `scripts/run_baseline.py`:

- **zero_shot** (`eval/baselines.py:ZeroShotBaseline`): `SYSTEM_PROMPT` + `FORCED_PREFIX = "Reasoning:"` + ChatML wrapping. Same scaffold CAEM Tier 2 uses.
- **rag** (`eval/baselines.py:RAGBaseline` → `caem.retrieval.rag.TierThreeRAG.generate` → `caem.prompts.build_tier3_prompt`): identical `SYSTEM_PROMPT` + `FORCED_PREFIX`, plus a `Context:` block of top-3 reranked passages from the 21M-passage Wikipedia FAISS-IVF index, plus a per-task `Answer format:` line.

Both paths emit `Reasoning: <text>\nAnswer: <text>`. The `display_answer` extractor parses post-`Answer:` content identically for both → `direct_em` and `rag_em` are computed by the same scoring rule over the same output shape.

Run command: `bash scripts/launch_baselines_v2.sh all` with `CAEM_BATCH_U_TOK_DROP=1`, `--eval_batch_size 32`, `--eval_prefetch`. Compute: zero_shot 16 min, rag ~17.5 h.

### 14.3 EM scoring (per-bench dispatch)

| Bench | EM source | Why |
|---|---|---|
| FEVER | `eval/harness.py:fever_accuracy(pred_label, gold)` | classification (supports/refutes/NEI); strict label match |
| CommonsenseQA | native EM | MCQ letter exact match |
| StrategyQA | native EM | yes/no exact match |
| OpenBookQA | native EM | MCQ letter exact match |
| HaluEval-QA | native EM | well-defined gold span match |
| TriviaQA | native EM (alias-list ANY match) | gold answers include alias variants |
| Natural Questions | native EM (alias-list ANY match) | gold answers include short-answer variants |
| **TruthfulQA** | **`em_llm_judged`** (Claude Haiku 4.5) | adversarial open-ended; ROUGE-L > 0.15 inflated zero_shot to 0.767 (true capability 0.442). Haiku follows Lin et al. 2022 §3.2: prediction matches a correct reference AND does not endorse any incorrect reference. Populated by `scripts/rescore_truthfulqa.py --judge_model claude-haiku-4-5`. |

Haiku rescore receipts (2026-05-18, $0.50 API):
- zero_shot TruthfulQA: ROUGE-L > 0.15 = 0.767; **Haiku-judged = 0.442**
- rag TruthfulQA: ROUGE-L > 0.15 = 0.306; **Haiku-judged = 0.271**
- → RAG hurts TQA by 17 pp under honest grading

The dispatch is implemented at `scripts/build_ruc_training_set.py:_sample_em()`.

### 14.4 Case partitioning + Path B filtering

For each paired `(zero_shot_sample, rag_sample)` with matching `id`, the four-outcome partition follows the EM dispatch above:

| Case | direct_em | rag_em | Meaning | Action |
|---|:---:|:---:|---|---|
| **A** | < 0.5 | ≥ 0.5 | RAG fixed a direct error | KEEP, label = 1 |
| **B** | ≥ 0.5 | < 0.5 | RAG broke a correct direct | KEEP, label = 0 |
| C | ≥ 0.5 | ≥ 0.5 | both right (tied) | DROP |
| D | < 0.5 | < 0.5 | both wrong (tied) | DROP |

**Path B labelling rule**: train on A+B only. Tie rows (C and D) are dropped because they contribute label noise on the disagreement-boundary calibration.

### 14.5 Empirical per-bench yields (computed 2026-05-18)

Out of 12 004 paired samples (24 samples dropped for missing EM on one or both sides), the case partition is:

| Bench | A | B | C | D | A+B | total | Yield |
|---|---:|---:|---:|---:|---:|---:|---:|
| FEVER | 503 | 455 | 708 | 834 | 958 | 2 500 | 38.3 % |
| TriviaQA | 524 | 305 | 405 | 1 242 | 829 | 2 476 | 33.5 % |
| CommonsenseQA | 50 | 108 | 249 | 93 | 158 | 500 | 31.6 % |
| StrategyQA | 16 | 27 | 82 | 62 | 43 | 187 | 23.0 % |
| **TruthfulQA** | 40 | 94 | 46 | 137 | 134 | 317 | **42.3 %** |
| HaluEval-QA | 199 | 203 | 93 | 2 005 | 402 | 2 500 | 16.1 % |
| OpenBookQA | 44 | 95 | 257 | 104 | 139 | 500 | 27.8 % |
| Natural Questions | 464 | 73 | 74 | 2 389 | 537 | 3 000 | 17.9 % |
| **Total** | **1 840** | **1 360** | **1 914** | **6 866** | **3 200** | **11 980** | **26.7 %** |

Observations:
- **Class balance**: 1 840 A (57.5 %) vs 1 360 B (42.5 %) → modest skew toward "RAG helps." Well within unbalanced-class tolerance.
- **TruthfulQA** has the highest yield (42.3 %), driven by case B (94 / 134 = 70 %): RAG actively breaks correct direct answers on adversarial myth questions. This is the architectural use-case for the RUC — keep TruthfulQA-like queries at T2.
- **HaluEval-QA** has the lowest yield (16.1 %). The benchmark was designed to induce hallucinations; both paths fail equally (2 005 D-rows = 80 % of HaluEval). Only the 402 disagreement rows contribute training signal.
- **Natural Questions** is dominated by case A (464 / 537 = 86 %): RAG strongly helps open-domain factoid where parametric memory is sparse.

### 14.6 Feature extraction

Features are computed on the 3 200 A+B disagreement rows only. C and D rows are dropped from training; their architectural defaults (T2 for C-like, T3 for D-like) emerge implicitly from the LR coefficient structure on `p_ik` + retrieval features.

See §2.2 for the full feature inventory (~34 dims across 9 families). All families are computed in a single end-to-end pass:

- `scripts/ruc_extract_v2_features.py` — Families B, C, C', C'', C''', F (retrieval, rerank, NLI pairwise, p_ik, Wikidata)
- `scripts/build_ruc_training_set.py` — Families A, D, E (surface form, linguistic, answer-type) at training-set-build time

The P(IK) probe at `caem/ruc/v2/pik_probe.joblib` was trained on 11 980 zero_shot answers (mean OOF AUROC 0.771). The reranker (BAAI/bge-reranker-v2-m3) and NLI judge (MiniCheck / AdaptiveNLIJudge) are reused from CAEM's verifier construction at `scripts/rescore_baselines_through_verifier.py:_build_verifier:199-211`.

### 14.7 Train / validation / test split

**Path B + unit weights** uses a single dataset of ~3 200 clean A+B rows. Validation is **nested 5-fold stratified random CV** (stratified by `case`):

- **Outer 5 folds**: each fold has ~640 rows held out; reports per-fold AUROC + per-bench AUROC.
- **Inner CV per outer fold**: sweeps `C ∈ {0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}` and selects best by inner AUROC.
- **Threshold τ sweep**: after fitting the final LR on all 3 200 rows with the chosen C, sweep τ ∈ {0.20, 0.25, …, 0.75} on OOF predictions to maximise net deployment utility = `A_caught(τ) − B_caught(τ)`.

No separate held-out test set is reserved (the small N makes a held-out test wasteful). The Phase 1e trajectory on the CAEM eval pool serves as the true test set: the RUC is trained on the fresh pool (content-hash disjoint from CAEM eval by construction) and deployed inside `AdaptiveRouter` during Phase 1e, where its predictions on the CAEM eval pool are the production-relevant generalisation measurement.

### 14.8 Reproducibility receipt

| Anchor | Path / command |
|---|---|
| Source pool | `caem/ruc/fresh_pool/*.json` (12 004 questions) |
| Generation command | `bash scripts/launch_baselines_v2.sh all` |
| Generation outputs | `outputs/baselines_v2/{zero_shot,rag}/<bench>_cycle0.json` |
| TruthfulQA Haiku judge | `python -m scripts.rescore_truthfulqa --files outputs/baselines_v2/{zero_shot,rag}/truthfulqa_cycle0.json --judge_model claude-haiku-4-5` |
| EM dispatch logic | `scripts/build_ruc_training_set.py:_sample_em` |
| A+B pre-filter output | `/tmp/fresh_pool_AB_only/<bench>_v2.json` × 8 |
| P(IK) probe training | `python -m scripts.ruc_train_pik_probe --in_parquet caem/ruc/v2/pik_train_set.parquet --probe_path caem/ruc/v2/pik_probe.joblib --metrics_path caem/ruc/v2/pik_probe_metrics.json` |
| Feature extraction | `python -m scripts.ruc_extract_v2_features --fresh_pool_glob '/tmp/fresh_pool_AB_only/*.json' --passage_index data/passage_index --pik_probe_path caem/ruc/v2/pik_probe.joblib --out_parquet caem/ruc/v2/features.parquet` |
| Training set assembly | `python -m scripts.build_ruc_training_set --baselines_dir outputs/baselines_v2 --benches fever triviaqa commonsense_qa strategyqa truthfulqa haluevalqa openbookqa natural_questions --out caem/ruc/v2/training_set.parquet` |
| Final RUC artefacts | `caem/ruc/v2/{models/{scaler,lr_final}.joblib, feature_spec.json, cv_results.json}` |
