# CAEM — Next Session Plan 2.0 (Vast.ai Runbook)

**Updated:** 2026-04-24 (Session 5 close) — BDT 2026-04-24 14:00
**Status:** Phase 1a running in `tmux plan_a` (step_6 FEVER seed, ~55% through)
**Supersedes:** `NEXT_SESSION_PLAN.md` (v1) — kept on disk as historical record.
**Single source of truth for:** the metric pipeline, the runner chain, the post-run path, and the handoff to Chapters 4/5/6 writing.

---

## TL;DR — where we are

- **Runner is live.** `run_phase1a_hardened.sh` → `run_phase1a.sh` → currently in `step_6_reseed` (FEVER seed ~55% stored toward target). GPU healthy, cgroup tight but not OOM. No crashes.
- **Metric pipeline is consolidated.** One hallucination family (CHM, 8-subtype equal-weighted mean) replaces the prior CE + hallucination_rate + confident_error_rate triplet. CES.EPI = `1 − CHM`. Single pre-registered pass/fail gate: `chm_reduction_verdict`, floor 30%.
- **Baseline panel is finalized.** B1–B7 run numerically; B8 Self-RAG is citation-only; STaR optional ceiling. B5 = 5-shot CoT (reclaimed from FLARE, which is now dormant).
- **Benchmark splits are locked.** Training = {fever, triviaqa, natural_questions}. Transfer = {truthfulqa, strategyqa, arc_challenge, asqa}. ASQA is transfer-only (insufficient train volume for 10-cycle stream).
- **Post-step_7 path is one command.** `./run_post_step7.sh` runs all of Phase 1b (baselines + sig tests + diagnostics + Ch5/6 artefacts) sequentially in one script.

---

## Session 5 — what shipped in this session

### Session 5a (early) — verifier audit + 27 of 30 bugs fixed

A 4-pass audit of 1500 cycle-0 samples uncovered 30 bugs across input routing, signal calibration, composite formula, decision tree, memory poisoning, prompts, and validation gaps. 27 fixed; 3 architecturally correct-by-design.

Key architectural changes landed:
- **Adaptive thresholds per cycle** (label-free EMA quantile re-fit). Default ON.
- **`skip_per_cycle_temperature`** flag for production mode (no labels). Default False (research mode).
- **Composite weights revised, `p_ground_max` added** (sum=1.00):
  ```
  0.22·p_ground_mean + 0.18·p_ground_max + 0.06·p_ground_atomic
  + 0.14·p_entail    + 0.20·q_a_relevance + 0.10·s_avg
  + 0.08·u_internal  + 0.02·(1 − h_norm)
  ```
- **Input field routing** — verifier scoring now receives `display_answer` (extracted), not the verbose prediction chain.
- **Storage sanitizer** — template-leak + evasive-pattern regexes reject pollution at write time.
- **u_dropout saturation fix** — extract answer before token-drop hash.
- **Bidirectional NEI threshold tightened** — 4× decay > 0.10.
- **Prompt revision G27/G28/G29** — anti-evasion + placeholder removed.
- **Memory schema** — `entry.answer = display_answer` (not verbose chain).
- **Step 5.9 prompt smoke gate** — pre-Step-6 validation.
- **Step 7.0.3 weight validation gate** — post-7.0.2 Cohen's d / poisoning / store-discard check.

### Session 5b (mid) — metric consolidation

Removed redundant hallucination metric definitions. Goal: one headline, one gate, one taxonomy.

| Removed | Replaced by |
|---|---|
| `hallucination_rate(em, u, 0.5)` | `confabulation_rate(em, u, 0.5, "ge")` (semantic equivalent, parameterized) |
| `confident_error_rate` in `aggregate()` output | `hallucination_subtypes()["confident_confabulation_rate"]` (same formula, one of 9 subtypes) |
| `pooled_ce(...)` | `pooled_chm(samples_by_bench, ...)` (same ID/OOD split, CHM-based) |
| `ce_reduction_verdict()` (40% pooled-CE floor) | `chm_reduction_verdict()` (30% CHM floor) |
| `tab_halluc.csv` legacy 3-column table | `tab_halluc_subtypes.csv` (CHM + 9 rows) |
| `figure_halluc_decomp` 3-bar figure | `figure_chm_trajectory` (CHM line + 9-bar panel) |
| CES.EPI = `1 − confident_error_rate` | CES.EPI = `1 − CHM` (efficacy + gate now speak the same language) |

New helpers added:
- `composite_hallucination_metric` — taxonomy mean with 8-subtype default denominator + 9-subtype override for roberta_nli ablation.
- `pooled_chm` — ID/OOD split on CHM.
- `chm_reduction_verdict` — pre-registered 30% floor verdict helper.
- `build_table_halluc_subtypes` — per-cycle CHM + 9 subtype rows.
- `figure_chm_trajectory` — headline CHM line + grouped-bar subtype decomposition.

### Session 5c (late) — drift fixes + audit

Comprehensive grep + inspection across **80 pipeline files** (45 scripts + 6 eval + ~30 caem). Forbidden-symbol sweep was clean except for intentional historical-note comments. Drift fixes applied:

1. **eval/__init__.py** module docstring — ASQA→transfer-only, NQ back in training (Branch C 2026-04-22 evening decision).
2. **chapter_5.tex** — B1–B8 references corrected to B1–B7 (running) + B8 (Self-RAG citation-only) + STaR (optional ceiling). Holm correction recalculated: 8×6=48 → 7×6=42; 0.95⁴⁸ → 0.95⁴². `m=8` → `m=7` in caption.
3. **eval/reporting.py** — same B1-B8 → B1-B7 in docstrings.
4. **eval/baselines.py** — module docstring rewrote class list to B1–B7 + FLARE dormant + B8 citation-only. **All 4 `PipelineResult` constructions now set `display_answer=extract_cot_answer(answer)` — previous schema drift had baselines emitting empty display_answer.**
5. **caem/ablation/runner.py** — **CRITICAL**: `ces_axes_from_cycle()` now forwards `samples=...`, so ablation EPI uses CHM (matches headline). Without this, ablation CES would have diverged from headline CES.
6. **scripts/run_cyclic_ablation.py** — same fix as runner.py (conditional `pass_samples` guarding the CHM path vs legacy fallback).
7. **caem/ablation/variants.py** header comment "8 hand-picked" → "3 hand-picked" (matches actual registry).
8. **caem/verification/verifier.py** module docstring — removed stale `p_contra < 0.30` from decision tree (the veto was deleted 2026-04-22 because MiniCheck returns structurally 0; actual `_decide()` reads only `u_stored` + `p_ground_max`).
9. **scripts/run_baseline.py** CLI help rewritten.
10. **scripts/baseline_sig_tests.py** docstring B1-B5 list corrected.
11. **scripts/compare_prompt_design.py** field rename confident_error_rate → confident_confabulation_rate.

### Session 5d (close) — runner chain gap closure

Audit of `run_phase1a.sh` step chain vs available scripts found 4 critical orphans:
- `make_tables.py` — LaTeX `.tex` bodies for every tab_*.csv
- `make_figures.py` — 7 Ch5 PNG+PDF figures
- `aggregate_calibration_trajectory.py` — per-cycle τ/T/memory trajectory
- `generate_session5_artifacts.py` — audit summary + before/after + α-trajectory

Plus one important:
- `test_t1_routing_robustness.py` — T1 routing experimental validation

**Fix:** added `step_21_tables`, `step_22_figures`, `step_23_calib_traj`, `step_24_session5_artifacts`, `step_25_t1_robustness` to `run_phase1a.sh`. Each is idempotent (skips if output present) and non-fatal (logs + continues on error). Wired into `main()` after `step_20_aggregate`.

**New single-entry post-main runner:** `run_post_step7.sh` sources `run_phase1a.sh` (main is now BASH_SOURCE-guarded) and runs the entire post-step_7 chain in one command — baselines + sig tests + diagnostics + 5 artefact steps.

---

## Current architecture — metric pipeline as of Session 5 close

### Per-sample record (rectangular JSON schema)

Every sample across 6 benchmarks × 11 cycles writes one JSON record. Single source of truth for every downstream metric.

| Group | Fields |
|---|---|
| Identity | `id, benchmark, cycle, question, gold_answers, gold_label` |
| Answer | `prediction, display_answer, production_response` |
| Scoring | `em, f1` |
| Routing | `tier (1/2/3 or -1 crash), stored, latency_ms, escalated` |
| Verifier (12) | `u_stored, u_token, u_dropout, u_internal, s_avg, h_norm, p_entail, p_ground_max, p_ground_mean, p_ground_atomic, p_contra, decision, early_exit_triggered` |

Tier-1 hits + pipeline crashes leave verifier fields = None (rectangular schema preserved).

### u_stored composite (weighted sum of 8 signals)

```
u_stored = 0.22·p_ground_mean + 0.20·q_a_relevance + 0.18·p_ground_max
         + 0.14·p_entail      + 0.10·s_avg         + 0.08·u_internal
         + 0.06·p_ground_atomic + 0.02·(1 − h_norm)
```

Lane totals: grounding 0.46 · consistency 0.24 · semantic 0.20 · intrinsic 0.10. Grounding dominates by design.

`p_contra` is written into the record as a diagnostic but **no weight and no decision reads it** under the default MiniCheck backend (MiniCheck returns structurally 0.0 for contradict; only `roberta_nli_backend` ablation populates it).

### Stage-5 decision tree

```
u_stored ≥ 0.65              → STORE      (write to memory)
0.45 ≤ u_stored < 0.65       → DEFERRED   (review queue, re-checked next cycle)
u_stored < 0.45 AND p_ground_max < 0.20  → ABSTAIN ("I don't know")
otherwise                    → DISCARD
```

Second gate for SIL training eligibility: `u_stored ≥ 0.75` required to enter the next cycle's training batch.

### 9 hallucination subtypes (measured per sample)

| # | Subtype | Boolean condition | Default backend |
|---|---|---|---|
| 1 | confident_confabulation_rate | em=0 AND u_stored ≥ 0.50 | ✅ measured |
| 2 | factual_fabrication_rate | em=0 AND p_ground_atomic < 0.30 | ✅ measured |
| 3 | factual_contradiction_rate | em=0 AND p_contra > 0.30 | ❌ always 0 under MiniCheck; roberta_nli ablation only |
| 4 | logical_fabrication_rate | em=0 AND p_entail < 0.30 | ✅ measured |
| 5 | off_topic_rate | em=0 AND q_a_relevance < 0.50 | ✅ measured |
| 6 | defensive_evasion_rate | em=0 AND evasive regex matches | ✅ measured |
| 7 | template_leak_rate | template regex matches (EM-agnostic) | ✅ measured |
| 8 | false_refusal_rate | em=1 AND decision ∈ {ABSTAIN, DISCARD} | ✅ measured |
| 9 | over_long_rate | em=0 AND len(prediction) > 600 chars | ✅ measured |

### Headline scalars

**CHM (Composite Hallucination Metric):**
```
CHM = (1/8) · Σ subtype_rate_i   over 8 measurable subtypes
```
Equal-weighted mean across 8 default subtypes. `factual_contradiction_rate` excluded from denominator (structural 0 under MiniCheck). Opt-in override to 9 for roberta_nli ablation.

Also returned: `union_rate` (fraction with any subtype firing), `per_subtype_rates` (full 9).

**CES (CAEM Efficacy Score) — 5-axis geometric mean:**
```
CES = (ACC · EPI · RET · CAL · VER)^(1/5)
```
- **ACC** = cross-benchmark mean EM
- **EPI** = `1 − CHM` (post Session 5)
- **RET** = min(MMLU@cycle / MMLU@cycle-0, 1.0)
- **CAL** = `1 − 2·min(ECE, 0.5)`
- **VER** = balanced accuracy of Stage-5 STORE/DISCARD vs gold (0.5 placeholder until labels)

### Pre-registered pass/fail gate

```
chm_reduction_verdict(chm_baseline, chm_final, floor=0.30) → PASS/FAIL
```
30% relative CHM reduction from cycle 0 → final cycle. Broader than the old 40% CE floor because CHM is an 8-axis mean; a 30% drop across the taxonomy is structurally stronger than a 40% drop on one subtype.

### Benchmark split

| Pool | Benchmarks | Role |
|---|---|---|
| TRAINING_BENCHMARKS | `fever, triviaqa, natural_questions` | Eligible for SIL; large train splits ~87k–145k |
| TRANSFER_BENCHMARKS | `truthfulqa, strategyqa, arc_challenge, asqa` | Held-out eval only; never trained on |

ASQA is transfer-only (Branch C 2026-04-22 evening decision) because its 4,353 train samples can't sustain 10-cycle stream mode at 5000 samples/cycle.

### External baseline panel (B1–B7 running + B8 citation)

| # | Method | Step | Tests |
|---|---|---|---|
| B1 | Zero-shot (Qwen2.5-3B) | step_9_b1 | EM floor |
| B2 | CoT (Kojima 2022) | step_10_b2 | Reasoning alone close the gap? |
| B3 | RAG (DPR + top-k) | step_11_b3 | Retrieval alone? |
| B4 | CoT + RAG | step_12_b4 | Combination subsumes CAEM? |
| B5 | 5-shot CoT (Wei 2022) | step_11_5_b5 | Few-shot beats zero-shot CoT? |
| B6 | Vanilla SFT (no L2, no MMLU guard) | step_14_b6_vanilla_ft | Vanilla FT matches CAEM? |
| B7 | EWC-only FT (L2 anchor + MMLU guard) | step_15_b7_ewc_only | Regularized FT matches CAEM without memory? |
| B8 | Self-RAG (Asai 2024) | **NOT RUN** — citation-only in Ch 2 lit review |
| Optional | STaR ceiling | step_20B | Run only if budget allows |

**FLARE** (Jiang 2023) — removed from running chain 2026-04-22. `FLAREBaseline` class kept in `eval/baselines.py` for ablation/smoke but not invoked. B5 slot reclaimed for 5-shot CoT. See `run_phase1a.sh` line 813-819 for removal rationale.

---

## Phase 1a — currently running (you do nothing)

### Runner chain (in order)

```
step_5_9_prompt_smoke         ✓ passed before session 5 began
step_6_reseed                 ← CURRENTLY RUNNING (FEVER ~55%)
step_7_0_cycle0               ← 500 samples × 6 benchmarks, pristine Qwen
step_7_0_calibrate            ← τ_store/defer/train from cycle-0 u_stored
step_7_0_3_validate_weights   ← GATE: composite Cohen's d ≥ 0.20, poisoning ≤ 0.30
step_5_5_pairs                ← build 500-pair calibration set
step_5_5_headhead             ← MiniCheck vs RoBERTa vs Qwen-judge AUROC/ECE
step_19_2_slice               ← freeze cycle-0 500-sample retention slice
step_platt_calibrate          ← Platt scaling Qwen-judge → MiniCheck
step_hf_upload_pre_main       ← HF snapshot (safety checkpoint)
step_u_tok_drop_gate          ← GATE: CAEM_BATCH_U_TOK_DROP pool equivalence
step_7_main                   ← 10 cycles × 5000 samples × 6 benchmarks, SIL training
                                 (early-stop allowed at cycle ≥ 5 on MMLU plateau/rollback)
```

### Key invariants Phase 1a must preserve

| Guard | Condition | Severity |
|---|---|---|
| step_6 seed assertion | `total_seeded ≥ 540` (180 per training bench) | Halts chain if fails |
| step_7.0.2 threshold ordering | `train > store > defer` | Halts chain |
| step_7.0.3 weight validation | Cohen's d ≥ 0.20, store/discard gap ≥ 0.05, poisoning ≤ 0.30 | Halts chain (good — means weights broken) |
| Platt calibration | Pearson ρ ≥ 0.70 (logit space) | Halts chain |
| u_tok_drop gate | `|mean Δ u_stored| < 0.02` | Halts chain |
| Post-step_7 MMLU | retention ≥ 93% on all cycles | Assertion fails (does not halt — post-hoc sanity) |

### Expected Phase 1a duration

- Step 6 reseed: ~2-4 hours remaining (FEVER + TriviaQA + NQ)
- Steps 7.0, 7.0.2, 7.0.3, 5.5, 19.2.1, Platt, HF upload, u_tok_drop: ~2-3 hours combined
- Step 7 main: **~14 GPU-days worst case, much faster with bs=32 + SDPA + compiled graph** (original docs said 335h; Branch-C optimizations reduce meaningfully)

### What you monitor

```bash
tmux attach -t plan_a                                 # live runner output
tail -f outputs/phase1a_runner.log                    # orchestrator log
tail -f outputs/heartbeat.log                         # cgroup/GPU/OOM ticker
ls outputs/full_run/cycle_*/                          # cycle progress
```

### What you do when step_7_main finishes

**MANUALLY STOP THE RUNNER HERE.**

Detection signals:
```bash
# either of these means step_7_main completed:
ls outputs/full_run/run_complete.json                 # preferred
wc -l outputs/full_run/experiment_summary.csv         # ≥7 rows = success
```

Then:
```bash
tmux kill-session -t plan_a                           # cleanly stops the runner
```

Per memory note `caem_stop_after_step7_main.md`: do NOT let Phase 1a continue into step_8 onwards until the baseline rewire below is done.

---

## Interlude — baseline + ablation rewire (manual, ~20 min)

### Why

B6/B7 training scripts (`scripts/run_simple_ft.py`) and ablation variants (`caem/ablation/runner.py`, `scripts/run_cyclic_ablation.py`) hardcode `num_cycles=10` (baselines) and `num_cycles=5` (ablations).

If CAEM early-stopped at cycle 7, running B6 for 10 cycles gives B6 more training budget than CAEM. The comparison then measures "extra 3 cycles of training" not "selective retention vs vanilla SFT". Unfair.

### What to change

Read the actual completed cycle count from `outputs/full_run/run_complete.json::cycles_completed` (or from `experiment_summary.csv` row count) and pass it as `--num_cycles` to B6/B7 + ablation runners.

Suggested patch pattern:
```python
def read_caem_cycles(run_dir: Path) -> int:
    rc_path = run_dir / "run_complete.json"
    if rc_path.exists():
        return int(json.load(open(rc_path))["cycles_completed"])
    csv_path = run_dir / "experiment_summary.csv"
    if csv_path.exists():
        rows = list(csv.reader(open(csv_path)))
        cycles = {int(r[0]) for r in rows[1:] if r[0].isdigit()}
        return max(cycles) if cycles else 10
    return 10  # last-resort fallback
```

Call sites to patch:
- `scripts/run_simple_ft.py` — B6 / B7 driver
- `caem/ablation/runner.py` — if it holds a cycle default
- `scripts/run_cyclic_ablation.py` — cyclic ablation driver

### Validation

Before launching Phase 1b, verify:
```bash
grep -nE "num_cycles|cycles\s*=\s*10|cycles\s*=\s*5" scripts/run_simple_ft.py \
    caem/ablation/runner.py scripts/run_cyclic_ablation.py
# Every hit should be a comment or a default-fallback, not a hardcoded training budget.
```

---

## Phase 1b — `./run_post_step7.sh` (you launch once)

### Single command

```bash
tmux new-session -d -s phase1b ./run_post_step7.sh
tmux attach -t phase1b                                # watch live
tail -f outputs/phase1b_runner.log outputs/heartbeat.log
```

### What happens (in order)

| # | Step | Purpose | Output |
|---|---|---|---|
| 1 | step_9_b1 | B1 zero-shot | `outputs/baselines/zero_shot/<bm>_cycle0.json` |
| 2 | step_10_b2 | B2 CoT | `outputs/baselines/cot/<bm>_cycle0.json` |
| 3 | step_11_5_b5 | B5 5-shot CoT (reclaimed from FLARE) | `outputs/baselines/fiveshot_cot/<bm>_cycle0.json` |
| 4 | step_11_b3 | B3 RAG | `outputs/baselines/rag/<bm>_cycle0.json` |
| 5 | step_12_b4 | B4 CoT+RAG | `outputs/baselines/cot_rag/<bm>_cycle0.json` |
| 6 | step_14_b6_vanilla_ft | B6 vanilla SFT (dynamic cycle count post-rewire) | `outputs/baselines/vanilla_ft/eval/<bm>_cycle<N>.json` |
| 7 | step_15_b7_ewc_only | B7 EWC-only FT (dynamic cycle count post-rewire) | `outputs/baselines/ewc_only_ft/eval/<bm>_cycle<N>.json` |
| 8 | step_15_5_sig | McNemar + bootstrap CI + Holm correction across B1-B7 | `outputs/tab_sig_test.csv` |
| 9 | step_19_purity | Claim-2 α > ½ purity theorem validation | `outputs/purity_validation/theory_validation.json` + `results.json` |
| 10 | step_19_2_eval | Retention diagnostic on last-cycle checkpoint | `outputs/full_run/cycle_N/retention_diagnostic.json` |
| 11 | step_19_5_corr | Nine-signal correlation matrix (Ch4 Eq 3.5 defense) | `outputs/signal_correlation/redundancy_report.json` + figure |
| 12 | step_20_aggregate | Ablation aggregate (Phase 1a has `full` row only; sweep happens separately) | `outputs/ablation/ablation_table.csv` |
| 13 | **step_21_tables** | LaTeX `.tex` bodies from tab_*.csv | `outputs/full_run/tab_*.tex` (6 files) |
| 14 | **step_22_figures** | 7 Ch5 figures (PNG + PDF) | `outputs/full_run/fig5_{1..7}_*.{png,pdf}` |
| 15 | **step_23_calib_traj** | Per-cycle τ/T/memory-size trajectory | `outputs/full_run/calibration_trajectory.csv` + `tab_calibration_trajectory.tex` + `fig_tau_trajectory.pdf` |
| 16 | **step_24_session5_artifacts** | Audit summary + before/after + α-trajectory | `pre thesis 1 report/tables/tab_audit_summary.tex`, `tab_claim_evidence_map.tex`, `fig_audit_before_after.pdf`, `fig_composite_discrim_trajectory.pdf`, `fig_memory_growth.pdf`, `fig_alpha_trajectory.pdf` |
| 17 | **step_25_t1_robustness** | T1 routing validation (5 paraphrase strategies × last-cycle memory) | `outputs/t1_routing_test/results.json` + figure |

### Safety properties

- **Prerequisite check at entry** — fails fast if `run_complete.json` or `experiment_summary.csv` (≥7 rows) is missing. No partial launches.
- **Every step idempotent** — skip-if-output-present guard on all 17 steps. Rerun is safe; picks up where it left off.
- **Non-fatal steps** — each logs + continues on error (won't abort the chain if one step fails).
- **Heartbeat sidecar** — 60s ticker logging cgroup/GPU/OOM to `outputs/heartbeat.log`.
- **Own log** — `outputs/phase1b_runner.log` is truncated at start of each run for clean tracing.

### Runtime estimate

| Block | Estimate |
|---|---|
| Inference baselines B1-B5 (batched bs=32 where supported) | ~3.5 GPU-h |
| Training baselines B6-B7 (10 cycles each or CAEM-matched) | ~12 GPU-h |
| Sig tests + purity + retention + correlation | ~1 GPU-h |
| Ch5/6 artefact steps 21-25 | ~5 minutes (CPU-bound post-processing) |
| **Total Phase 1b** | **~17 GPU-h sequential** |

---

## After Phase 1b — thesis-ready state

### Ch 4 — Methodology (descriptive)

| Needs | Source |
|---|---|
| Architecture diagrams | Already hand-drawn |
| Gate definitions, u_stored composite, Stage-5 decision tree | Code docstrings + `caem/config.py` + `caem/verification/verifier.py` |
| 9-signal correlation matrix (Eq 3.5 defense) | `outputs/signal_correlation/redundancy_report.json` |
| MiniCheck vs RoBERTa backend-choice justification | `outputs/calibration/minicheck_vs_roberta_v5.json` |
| Benchmark splits (3 training + 4 transfer) | `caem/config.py` TRAINING_BENCHMARKS / TRANSFER_BENCHMARKS |
| 9-subtype taxonomy + coverage table | `tab_claim_evidence_map.tex` (step 24) |

### Ch 5 — Results (data-intensive)

| Section | Table / Figure | Source |
|---|---|---|
| §5.1 Headline (EM, latency, tier routing, CES) | `tab_headline.tex`, `fig5_1_ces_radar.pdf` | step 21, 22 |
| §5.2 Calibration | `tab_calibration.tex`, `fig5_2_reliability.pdf` | step 21, 22 |
| §5.3 Hallucination decomposition (CHM + 9 subtypes) | `tab_halluc_subtypes.tex`, `fig5_3_chm_trajectory.pdf` | step 21, 22 |
| §5.4 Grounding | `tab_grounding.tex`, `fig5_4_grounding.pdf` | step 21, 22 |
| §5.5 Purity (STORE/DEFERRED/ABSTAIN/DISCARD + α) | `tab_purity.tex`, `fig5_5_purity.pdf` + `fig_alpha_trajectory.pdf` | step 21, 22, 24 |
| §5.6 Continual learning (MMLU, BWT, FWT) | `tab_continual.tex`, `fig5_6_continual.pdf` | step 21, 22 |
| §5.7 Cycle-over-cycle significance (within-CAEM) | `tab_cycle_progression.tex`, `fig5_7_em_progression.pdf` | step 21, 22 |
| §5.8 CAEM vs B1-B7 significance | `tab_sig_test.csv` (convert to .tex manually or extend make_tables) | step 15.5 |
| §5.9 Per-cycle calibration trajectory | `tab_calibration_trajectory.tex`, `fig_tau_trajectory.pdf` | step 23 |
| §5.10 Audit before/after | `tab_audit_summary.tex`, `fig_audit_before_after.pdf`, `fig_composite_discrim_trajectory.pdf`, `fig_memory_growth.pdf` | step 24 |
| §5.11 T1 routing robustness | `outputs/t1_routing_test/results.json` (text) | step 25 |

### Ch 6 — Discussion (interpretive)

Drawn from Ch5 artefacts; no new data collection. Specific talking points:
- CHM trajectory narrative — how the headline shrinks over cycles, which subtypes shrink most.
- Purity theorem validation (α > 0.5 across all cycles).
- Taxonomy coverage disclosure — 8/9 subtypes measurable under MiniCheck; factual_contradiction requires roberta_nli ablation.
- CAEM vs prior work — where CAEM beats B1-B7 significantly, where it doesn't (B5 5-shot CoT may match CAEM on benchmarks where strong prompting suffices; that's the designed-for-task asymmetry called out in §5.8 pre-registration).
- Self-RAG positioning — citation-only because LLaMA-specific critic tokens make cross-backbone comparison measure prompt engineering not architecture.
- Memory growth vs purity tradeoff — `fig_memory_growth.pdf` + `fig_alpha_trajectory.pdf`.
- Failure-mode analysis — slice `per_sample_signals.jsonl` for specific error patterns.
- Limitations — MiniCheck backend, ASQA transfer-only, fixed benchmark panel.

---

## Deferred — Phase 1c (ablation sweep, post-thesis-submission)

**NOT in the current runner.** Per user budget + timing decisions, ablation sweep is Phase 1 Full Steps 16-18, handled separately after baselines complete and baseline rewire validated on real data.

### Scope

3 hand-picked ablation variants (2026-04-22 pruned registry):
- `no_retroverify` (Claim 2 + 3 defense — time dimension)
- `no_self_improvement` (Claim 3 defense — SIL contribution)
- `no_forgetting_guard` (Claim 4 defense — MMLU rollback)

Plus the `full` anchor.

### Runner

`scripts/run_cyclic_ablation.py` per-variant, aggregated by `scripts/aggregate_ablation.py`. Both already CHM-consistent (CES.EPI = 1−CHM fix applied Session 5d).

### Expected runtime

~20 GPU-h for the 3 variants at 5 cycles each.

### When to run

After baselines complete AND baseline rewire validated on real output. Not blocking for thesis draft — `aggregate_ablation.py` writes empty table gracefully when variants haven't run yet, so Phase 1b tolerates absence of ablation data.

---

## Post-run — local thesis compilation

### scp outputs down

```bash
rsync -avz --progress you@vast:~/caem/outputs/ ~/caem/outputs/
rsync -avz --progress you@vast:'~/caem/pre thesis 1 report/' '~/caem/pre thesis 1 report/'
```

### Compile Ch5

The `.tex` files land in positions chapter_5.tex already `\input{}`s:
```bash
cd 'pre thesis 1 report'
xelatex main.tex  # or pdflatex, depending on your setup
```

### What to sanity-check locally

1. `tab_headline.tex` has 11 cycle rows (or fewer if early-stop).
2. `tab_halluc_subtypes.tex` has 9 subtype rows + CHM summary.
3. `fig5_3_chm_trajectory.pdf` shows monotone-decreasing CHM across cycles (if the thesis claim holds).
4. `tab_sig_test.csv` — CAEM beats B1-B4 on training benchmarks with McNemar p < 0.05 (the pre-registered prediction in chapter_5.tex §5.3).
5. MMLU retention ≥ 0.95 on all cycles per `tab_continual.tex`.
6. `chm_reduction_verdict` output (add manual call at end of phase 1b if desired) shows PASS at 30% floor.

---

## Rollback + recovery

### If step_7_main crashes mid-cycle

Hardened wrapper captures exit code. Relaunch automatically resumes from last completed cycle:
```bash
./run_phase1a_hardened.sh   # finds highest cycle_* dir, adds --resume_from_cycle
```

### If OOM at cgroup limit

MALLOC_TRIM_THRESHOLD_ + MALLOC_MMAP_THRESHOLD_ already aggressive (131072 / 128KB). `PYTHONMALLOC=malloc` routes Python allocs through glibc for tighter trim.

If OOM counter climbs, options:
1. Reduce `--eval_batch_size` (currently 32).
2. Disable `--eval_prefetch`.
3. Downgrade to `python -m scripts.run_experiment ... --no-compile` if torch.compile peak memory is the culprit.
4. In hardened wrapper, reduce heartbeat frequency (currently 60s — unlikely to be cause).

### If Phase 1b step fails

Idempotent skip guards mean:
1. Fix the cause.
2. Rerun `./run_post_step7.sh`.
3. It skips completed steps and retries the failed one.

No full-chain restart needed.

---

## Handoff state for the next engineer

### What's ready

- Metric pipeline consolidated (CHM headline, CES rebased, pre-registered gate aligned).
- Runner chain complete (step_5_9 through step_25_t1_robustness).
- Post-step_7 path is one command (`./run_post_step7.sh`).
- All drift between thesis + code + config fixed (ASQA transfer-only, B1-B7 running, FLARE dormant, B5 = 5-shot CoT).
- 80 files audited for forbidden-symbol drift; only intentional historical-note comments remain.

### What's pending

- **Task #42** — Document CHM methodology in thesis Ch5 §Methods. 1 paragraph + coverage table. Deferred until Phase 1a data lands so the paragraph can reference real cycle-0 and cycle-N CHM values.
- **Baseline rewire** — dynamic cycle count read in `run_simple_ft.py` + ablation scripts. ~20 min manual edit before Phase 1b launches.

### What's out of scope for this session

- Ablation sweep (Phase 1c, post-baseline).
- Post-hoc RoBERTa re-scoring for `factual_contradiction_rate` (user decided against; CHM stays at 8-axis denominator).
- Ship 3 (shadow-verify on baselines for CHM comparability) — rejected as scope creep; baselines answer EM parity, not hallucination.

### Critical invariants to preserve

1. **Per-sample JSON is the source of truth.** Every metric derives from it. Don't pre-aggregate anything that can be derived later.
2. **CHM denominator defaults to 8.** `factual_contradiction_rate` is only in the denominator when `measured_subtypes` is explicitly overridden (roberta_nli ablation path).
3. **CES.EPI = 1 − CHM.** If you touch one, touch the other. Diverging them breaks the headline narrative.
4. **Baseline panel is B1-B7 running + B8 citation-only.** Don't add running baselines without updating Holm correction m= count in chapter_5.tex §5.3.
5. **ASQA is transfer-only.** Don't train on it. `caem/config.py` TRAINING_BENCHMARKS is the authoritative list.
6. **FLARE is dormant.** `FLAREBaseline` exists but not in the sig-test panel. Don't re-add without updating chapter_5.tex §5.3 pre-registration paragraph.
7. **`run_phase1a.sh` main() is BASH_SOURCE-guarded.** `run_post_step7.sh` sources it for function defs. Don't remove the guard.
8. **Each step in both runners is idempotent.** If you add a new step, include a skip-if-output-present guard. Breaking idempotency breaks resume-on-crash.

---

## Credit + timing

- **Current Vast credit:** $102 as of 2026-04-21 (memory: `caem_phase1_credit.md`).
- **Estimated Phase 1a remaining:** ~14-20 GPU-h (step 7 main dominates).
- **Estimated Phase 1b:** ~17 GPU-h.
- **Estimated Phase 1c (ablations, post-thesis):** ~20 GPU-h.
- **Total remaining through thesis submission:** ~50 GPU-h.
- **At ~$1/h Vast 5090:** comfortably within current credit.

Topup needed only if Phase 1c runs before baselines, or if step_7_main takes the full worst-case 335 GPU-h (original estimate; Branch-C optimizations should significantly reduce).

---

## References

- **Previous plan:** `NEXT_SESSION_PLAN.md` (v1, historical — supersed by this file)
- **Branch C decision log:** `branch_C_log.md`
- **Chapter 5 draft:** `pre thesis 1 report/chapters/chapter_5.tex`
- **Audit tracking:** `CH_AUDIT_TRACKING.md`
- **Memory persistence:** `/root/.claude/projects/-workspace/memory/`
- **Hardware deployment:** `VAST_AI_DEPLOYMENT_GUIDE.md`
- **CAEM package audit:** `docs/caem-package-audit-report.md`
- **Metrics audit:** `docs/metrics-audit.md`
- **Scripts audit:** `docs/scripts-audit-report.md`

---

**End of plan.** One script to finish Phase 1a (running). One command for Phase 1b (`./run_post_step7.sh`). One manual interlude (baseline rewire). After that: thesis writing.
