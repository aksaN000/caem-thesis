# RUC next-session plan

Canonical "what to do next" document for the RUC v2 build through Ch6 delivery. Living doc; the "Current state" block updates as phases close; the rest stays stable as the contract.

For *architecture / labels / features*, see DESIGN.md.
For *deployed v1 state*, see README.md.
For *phase-by-phase compute budgets*, see PLAN.md.
This document covers **operational sequencing + risk-mitigation + resume protocols**.

---

## Current state (live)

```
2026-05-18 03:25 BDT
Step 1 (v2 baselines) in progress: zero_shot baseline generating on the 12 004-question fresh pool.
  tmux:        baselines_v2
  log:         outputs/baselines_v2/zero_shot.gen.log
  done so far: fever, triviaqa, commonsense_qa, strategyqa, truthfulqa (5/8)
  current:     haluevalqa 960/2500 (38 %)
  pending:     openbookqa, natural_questions
  throughput:  ~12 samples/sec at bs=32 + prefetch (much faster than smoke estimate)
  ETA to zero_shot done: ~15 min
  ETA to rag done:       ~2-3 h after zero_shot
  ETA to all v2 baselines done: ~3 h total

Step 2 (v2 rescore)-B6 + Side-work C1 (B6 vanilla_ft rescore resume) + Phase 1e + Phase 1e analysis + Phase F: queued.
```

---

## Risk-mitigation contract (READ BEFORE FIRING NEXT PHASE)

### R1. gdrive bucket separation — CRITICAL (local folder doesn't matter)

The gdrive offload path is constructed at `caem/training/self_improvement.py:1647-1651`:

```
gdrive:caem-phase1a/{CAEM_GDRIVE_BUCKET}/{run_name}/cycle_{N}/
                    └─ env var, default "v2"   └─ self.output_dir.name
```

The vanilla CAEM trajectory uploaded its Phase-1d artefacts to
`gdrive:caem-phase1a/v2_1_phase1d/full_run/cycle_{0..6}/`. **That backup is the
canonical Phase-1d archive we rely on for the RUC vs no-RUC ablation.**
(The default `CAEM_GDRIVE_BUCKET=v2` was overridden to `v2_1_phase1d` at the
time of that run.)

The fix is a single env var at launch time — no code change:

```bash
# Phase 1e launch — CAEM-with-RUC trajectory
CAEM_GDRIVE_BUCKET=v2_1_phase1e_2026-05-18 \   # ← date-tagged, no collision
CAEM_GDRIVE_OFFLOAD=1 \
CAEM_BATCH_U_TOK_DROP=1 \
bash run_phase1a.sh
# Uploads land at gdrive:caem-phase1a/v2_1_phase1e_2026-05-18/full_run/cycle_{0..N}/
# Vanilla backup at gdrive:caem-phase1a/v2_1_phase1d/full_run/ stays untouched.
```

Local `outputs/full_run/` may be clobbered — that is acceptable because
gdrive has the Phase-1d backup at the separate bucket. If you want to
preserve the local copy too, optionally do `mv outputs/full_run outputs/full_run_archive`
before launch; it is NOT required for correctness of the ablation.

**Watchdog caveat**: `scripts/watchdog_cycle*.sh` hardcode `GDRIVE=gdrive:caem-phase1a/full_run` (no bucket prefix). They already read the variable from env, so override at launch if used:

```bash
GDRIVE=gdrive:caem-phase1a/v2_ruc/full_run \
bash scripts/watchdog_cycles_3plus.sh
```

### R2. Rescore-script bench list

`scripts/rescore_baselines_through_verifier.py` defaults to v1 panel. For v2 fresh-pool baselines, must pass the full 8-bench list explicitly:

```bash
CAEM_BATCH_U_TOK_DROP=1 \
python -m scripts.rescore_baselines_through_verifier \
    --composite_calibration outputs/full_run/cycle_3/composite_calibration.json \
    --passage_index data/passage_index \
    --checkpoint outputs/full_run/cycle_3/adapter \
    --baselines zero_shot rag \
    --baselines_dir outputs/baselines_v2 \
    --output_dir outputs/baselines_v2 \
    --benchmarks fever triviaqa commonsense_qa strategyqa truthfulqa haluevalqa openbookqa natural_questions \
    --batch_size 32
```

After R1 the composite_calibration path becomes `outputs/full_run/cycle_3/composite_calibration.json` instead of `outputs/full_run/cycle_3/composite_calibration.json`. Both Step 2 (v2 rescore) and Phase 1e consume the same Phase-1d frozen composite — that is the **invariant** that lets us compare vanilla vs with-RUC apples-to-apples.

### R3. Skip-if-exists

| Script | Skip status | Action needed |
|---|---|---|
| `scripts/launch_baselines_v2.sh` | ✅ added 2026-05-18 (`all_benches_done_for`) | none |
| `scripts/rescore_baselines_through_verifier.py` | ⚠ partial (checks for existence of inputs only, not outputs) | add output-exists check at top of `_rescore_one` so re-runs skip already-rescored (baseline, bench) pairs |
| `run_phase1a.sh` (with RUC) | ⚠ existing pre-flight checks but assume `outputs/full_run` | parameterise before Phase 1e launch |

The rescore-script skip is the most urgent — without it, Step 2 (v2 rescore) + Side-work C1 (B6 vanilla_ft rescore resume) might re-do completed work if interrupted.

### R4. Baselines archival to gdrive

12 GB at `outputs/baselines/` covers all 9 v1 baselines (B1-B9). Since we are NOT re-running them on the v2 fresh pool, we need to ensure they're preserved + accessible to the with-RUC analysis pipeline. Plan:

```bash
# rsync to the gdrive baselines folder for archival
rclone copy outputs/baselines/ gdrive:caem/baselines_v1_panel/ -P
# verify byte-count match
rclone check outputs/baselines/ gdrive:caem/baselines_v1_panel/
```

Action timing: fire after Step 1 (v2 baselines) ends (≈15 min) when there's idle wall-time available. Don't block any compute on it; rclone is bandwidth-bound and runs in parallel.

---

## Full run sequence (with explicit resume points)

Each phase has an explicit **"resume from"** anchor in case of interruption. Resume points are the file/dir that must exist on disk for the next phase to be reproducible.

### BUILD STEPS 1-6 — v2 RUC under the Phase-1d instrument — v2 RUC build

| # | Phase | Resume anchor | Wall | $ |
|---:|---|---|---|---|
| B1 | v2 baselines (zero_shot + rag) on fresh pool | `outputs/baselines_v2/<baseline>/<bench>_cycle0.json` per bench, full sample count | ~3 h | ~$1.20 |
| B2 | Rescore both baselines through Phase-1d verifier (`CAEM_BATCH_U_TOK_DROP=1`, `--batch_size 32`) | `outputs/baselines_v2/<baseline>/<bench>_cycle0_with_chm.json` per bench | ~25 GPU-h | ~$10 |
| B3 | Feature engineering: P(IK), FAISS top-1+top-5, pageviews, linguistic, ans-type | merged into `caem/ruc/training_set_v2_canonical.parquet` | ~5 GPU-h | ~$2 |
| B4 | Build v2 training set: case A/B (drop ties), label, split | `caem/ruc/training_set_v2_canonical.parquet` exists with `case` and `training_label` columns | ~30 min CPU | $0 |
| B5 | Train v2 LR: nested CV for C, sweep τ on full predictions | `caem/ruc/v2_shortcut/{models/{scaler,lr_final}.joblib, feature_spec.json}` | ~10 min CPU | $0 |
| B6 | Smoke-test v2 RUC integration: 50-query matched-protocol pass | `caem/ruc/v2_artefacts/smoke_pass.json` (PASS/FAIL verdict) | ~30 min CPU | $0 |

**Step 1-6 critical-path total: ~30 GPU-h, ~$13 Vast, ~1.5 days wall (mostly B2 rescore).**

### SIDE WORK — Phase-1d cleanup + tooling — Side work (parallel where possible)

| # | Phase | Resume anchor | Wall | $ |
|---:|---|---|---|---|
| C1 | Resume B6 vanilla_ft rescore for 4 missing files (truthfulqa c3/c5, strategyqa c3/c5) | 4 new `outputs/baselines/vanilla_ft/eval/<bench>_cycleN_with_chm.json` | ~5-8 GPU-h | ~$3 |
| C2 | Rsync baselines to gdrive: `outputs/baselines/` → `gdrive:caem/baselines_v1_panel/` | 12 GB at gdrive endpoint, byte-verified | ~30 min WAN | $0 |
| C3 | Add output-exists skip to `rescore_baselines_through_verifier.py` (defensive against re-runs) | code change committed | 15 min CPU | $0 |
| C4 | Add `--output_dir` flag to `run_phase1a.sh` (defensive against clobber on RUC re-run) | code change committed | 30 min CPU | $0 |

**Side-work runs in parallel with B2 (rescore holds the GPU; C2-C4 are network/CPU only).**

### PHASE 1E — CAEM-with-RUC trajectory — CAEM-with-RUC trajectory

**Before launch**: R1 mitigation done (env var `CAEM_GDRIVE_BUCKET=v2_1_phase1e_2026-05-18` set; local `outputs/full_run/` may be clobbered which is fine since `gdrive:caem-phase1a/v2_1_phase1d/full_run/` holds the canonical Phase-1d backup).

| # | Phase | Resume anchor | Wall | $ |
|---:|---|---|---|---|
| D1 | Wire v2 RUC into router by pointing spec path at `caem/ruc/v2_shortcut/feature_spec.json` | `caem/routing/router.py` constructed with `ruc=LRRetrievalUtilityClassifier.from_spec(...)`; smoke-tested | ~5 min CPU | $0 |
| D2 | Phase 1a with RUC: cold-start + cycle-0 calibration | `outputs/full_run_ruc/cycle_0/{composite_calibration.json, adapter}` | ~12-24 GPU-h | ~$5-10 |
| D3 | step_7_main with RUC: cycles 0 → retention-guard-fire | `outputs/full_run_ruc/{cycle_N, eval, run_complete.json}` | ~3-7 GPU-days | ~$30-65 |

**Phase 1e total: ~4-8 GPU-days, ~$35-75.**

### PHASE 1E ABLATION ANALYSIS — Ablation analysis (Phase-1d RUC vs no-RUC)

| # | Phase | Resume anchor | Wall | $ |
|---:|---|---|---|---|
| E1 | Build per-bench EM/CHM trajectory table: `gdrive:caem-phase1a/v2_1_phase1d/full_run/` (vanilla, Phase 1d) vs `gdrive:caem-phase1a/v2_1_phase1e_2026-05-18/full_run/` (with-RUC, Phase 1e) | `outputs/tables/tab_ruc_ablation_trajectory.{json,tex}` | ~1 h CPU | $0 |
| E2 | Routing-decision breakdown: how often the RUC re-routed T3→T2 per (bench, cycle) | `outputs/tables/tab_ruc_routing_distribution.{json,tex}` | included in E1 | $0 |
| E3 | Retention-guard fire-cycle comparison (vanilla=C4; with-RUC=?) | E1's output table includes this | included | $0 |
| E4 | Statistical-significance tests (McNemar paired EM; bootstrap BCa on CHM delta) | `outputs/tables/tab_ruc_significance.json` | ~1 h CPU | $0 |

### PHASE 1E CH6 DELIVERABLES — Ch6 deliverables

| # | Phase | Resume anchor | Wall | $ |
|---:|---|---|---|---|
| F1 | Generate trajectory plot (vanilla vs with-RUC, per bench) | `outputs/figures/fig_ruc_ablation_trajectory.{png,pdf}` | ~2 h CPU | $0 |
| F2 | Generate routing-distribution table (LaTeX) | `outputs/tables/tab_ruc_routing_distribution.tex` | ~1 h CPU | $0 |
| F3 | Write Ch6 §sec:ruc-ablation prose + table refs + figure refs | `thesis_report/chapters/chapter_6.tex` updates | ~1-2 days writing | $0 |

---

## Total budget (current state → Ch6 in hand)

| | Compute | Cost |
|---|---|---|
| Step 1-6 (v2 RUC build) | ~30 GPU-h | $13 |
| Side-work (side work) | ~6 GPU-h | $3 |
| Phase 1e (CAEM-with-RUC trajectory) | ~100-200 GPU-h | $40-80 |
| Phase 1e analysis + F (analysis + writing) | CPU + human | $0 |
| **Total** | **~140-240 GPU-h** | **$56-96 Vast, $0 API** |
| **Unattended wall time** | **~6-10 days** | |

---

## Critical-path commands (fire in order)

### When Step 1 (v2 baselines) finishes (~3 h from launch)

```bash
# 1. Verify all 16 expected JSONs exist
ls outputs/baselines_v2/zero_shot/ outputs/baselines_v2/rag/ | grep -c cycle0.json   # expect 16

# 2. Fire Step 2 (v2 rescore) rescore (uses Phase-1d frozen composite from vanilla CAEM)
tmux new-session -d -s rescore_v2 -x 220 -y 50 "
  CAEM_BATCH_U_TOK_DROP=1 \\
  python -m scripts.rescore_baselines_through_verifier \\
    --composite_calibration outputs/full_run/cycle_3/composite_calibration.json \\
    --passage_index data/passage_index \\
    --checkpoint outputs/full_run/cycle_3/adapter \\
    --baselines zero_shot rag \\
    --baselines_dir outputs/baselines_v2 \\
    --output_dir outputs/baselines_v2 \\
    --benchmarks fever triviaqa commonsense_qa strategyqa truthfulqa haluevalqa openbookqa natural_questions \\
    --batch_size 32 \\
    2>&1 | tee outputs/baselines_v2/rescore.log
  echo '*** DONE ***'
  exec bash
"

# 3. While rescore runs (~25 GPU-h), fire C2 (gdrive copy) in another shell
rclone copy outputs/baselines/ gdrive:caem/baselines_v1_panel/ -P &
```

### After Step 2 (v2 rescore) + C1 finish (~1-2 days)

```bash
# Step 3 (v2 features) feature engineering
/venv/main/bin/python -m scripts.ruc_build_wiki_pageviews --in_jsons caem/ruc/fresh_pool/*.json --out_parquet data/ruc/pageviews_v2.parquet
/venv/main/bin/python -m scripts.ruc_train_pik_probe ...
/venv/main/bin/python -m scripts.ruc_offline_passage_retrieval ...

# Step 4 (v2 training-set build)-B5 (CPU only, ~40 min)
/venv/main/bin/python -m scripts.build_ruc_training_set ...
/venv/main/bin/python -m scripts.train_ruc ...   # with nested CV for C + tau sweep

# Step 6 (v2 RUC smoke) smoke
/venv/main/bin/python -m scripts.ruc_smoke_v2  # not yet written; pattern from existing smoke
```

### Phase 1e launch

```bash
# R1 protection is via CAEM_GDRIVE_BUCKET=v2_ruc — no code change needed.
# The vanilla Phase-1d gdrive backup stays at gdrive:caem-phase1a/v2/full_run/
# and the with-RUC run uploads to gdrive:caem-phase1a/v2_ruc/full_run/.

tmux new-session -d -s plan_a_ruc -x 220 -y 50 "
  CAEM_GDRIVE_BUCKET=v2_ruc \\
  CAEM_GDRIVE_OFFLOAD=1 \\
  CAEM_BATCH_U_TOK_DROP=1 \\
  bash run_phase1a.sh 2>&1 | tee outputs/run_phase1a_ruc.log
  echo '*** DONE ***'
  exec bash
"
```

---

## Open decision points (require human input)

| When | Decision | Default action if no input |
|---|---|---|
| After Step 5 (v2 RUC training) (v2 RUC CV AUROC measured) | If pooled CV AUROC ≥ 0.65 → proceed to Phase 1e. If < 0.65 → add cross-encoder rerank feature first (~30 min build). | proceed |
| Before Phase 1e launch | confirm R1 archive done | block — explicit confirmation required |
| After Phase 1e step 3 (step_7_main with RUC) (with-RUC trajectory finishes) | retention guard fired at C? → if early, that itself is the result; if late, extended trajectory is the result | report whichever happens |
| After Phase 1e analysis step 4 (significance tests) | does the significance test show p < 0.05 on the with-RUC vs no-RUC delta? if not, the Ch6 framing shifts to "RUC produces architecturally cleaner routing without statistically significant EM lift" | report honestly |


---

## 2026-05-18 late-day update

- rag baseline rolling on fever at 1216/2500 (49 %, ~70 min elapsed). Pace ~3.5 sec/sample. ETA ~10 h total.
- zero_shot baseline completed (8/8 benches, 16 min) — outputs at outputs/baselines_v2/zero_shot/.
- gdrive cleanup done: empty v2/ purged, baselines/ → baselines_v1_panel/ inside v2_1_phase1d/.
- gdrive v1 baselines sync verified: 0 bytes transferred (already in sync at v2_1_phase1d/baselines_v1_panel/).
- Phase naming aligned with CAEM convention: Phase 1d (current) / Phase 1e (new with-RUC).
- Phase 1e gdrive bucket name locked at `v2_1_phase1e_2026-05-18`.
- rescore script skip-if-exists code change committed (`_rescore_one(skip_if_exists=True)` default + `--force` CLI flag).
- B7 ewc_only_ft confirmed SKIPPED FROM PANEL (planned, Ch5 §footnote).
