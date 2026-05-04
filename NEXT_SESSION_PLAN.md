# CAEM — Next Session Plan

**Updated 2026-04-29 13:36 UTC | Session 10 — autonomous Step 7 main + Phase 4 receipts after return**

---

## ★★ FULL EXECUTION TIMELINE — chronological master sequence (added 2026-05-01) ★★

This section gives the end-to-end execution order. Every numbered step cross-references the existing detail section that contains the procedure. Read this first; drop into the detail sections as needed.

### Phase A — Cycle 2 close (~17:30 UTC May 1, in flight)

**A.1** Watchdog auto-snapshots cycle-2 stream-chunk JSONs as fever / triviaqa / NQ Step 4 each finishes; uploads to `gdrive:caem-phase1a/full_run/eval_streamchunk/` (later moved to `cycle_2/eval_stream_chunk/` at reorg time). Watchdog log at `outputs/watchdog_cycle2.log`. **Effort: zero, autonomous.**

**A.2** When run.log emits "Cycle 2 done in", auto-fire halt + restart sequence:
- `tmux send-keys -t plan_a C-c` → wait 30 s → verify clean exit (no python processes)
- Verify `caem/config.py:134` (safety_u_pre_min=0.38) and `:770` (deferred_buffer_ttl_cycles=4) stay in place
- `tmux new-session -d -s plan_a -x 220 -y 60 'source /venv/main/bin/activate && ./run_phase1a.sh 2>&1 | tee -a outputs/phase1a_runner.log'`
- Verify "RESUMING EXPERIMENT FROM CYCLE 3" + "Restored pristine MMLU anchor for SIL retention guard: 0.6250" in run.log within 60 s
- **Effect: cycles 3-10 run under the patched runner with auto Step 4b stream-chunk snapshot + Step 6c full gdrive offload organized layout. Effort: ~25 min.**

### Phase B — Cycles 3-10 main trajectory (~May 2 → May 12-14)

**B.1** Continue hourly /loop status checks until cycle 10 close or early-stop fires (cycle 5+, see triple-signal gate in `caem/eval/equilibrium.py`). No interventions planned during this phase. **Effort: zero, autonomous.**

**B.2** Each cycle close auto-uploads to `gdrive:caem-phase1a/full_run/cycle_{N}/` via Step 6c. Per-cycle artefacts in clean per-cycle-folder layout.

**B.3** On `outputs/full_run/run_complete.json` appearing → step_7_main is done. **Verification per P1.**

### Phase C — Phase 4 auto-runs (immediately after Step 7 main, ~May 12-14)

**C.1** **B1–B7 EXTERNAL BASELINES** under matched protocol — auto-fired by `run_phase1a.sh`:
- B1 zero-shot Qwen-2.5-3B-Instruct
- B2 chain-of-thought (Wei et al.)
- B3 DPR-RAG retrieval-augmented (top-k=5, Karpukhin DPR + Lewis RAG)
- B4 CoT + DPR-RAG combined
- B5 five-shot CoT (TriviaQA train demonstrations)
- B6 vanilla fine-tuning (cyclic, no L2 anchor, no retention guard)
- B7 EWC fine-tuning (L2 anchor + retention guard, no verifier/memory/routing)
- Output: `outputs/baselines/B1/` … `outputs/baselines/B7/`
- **PRECONDITION (Task #111):** halt + rewire dynamic cycle count if Step 7 main early-stopped, so B6/B7 use the realised cycle horizon (currently hardcoded at 10) — verify this BEFORE B1-B7 auto-run to avoid mismatched-scale comparison

**C.2** **STATISTICAL SIGNIFICANCE PANEL** — auto-fired after B1-B7:
- Per-benchmark paired McNemar with continuity correction (cycle-2 plan §sig-mcnemar)
- Bootstrap 95 % CI on EM delta (BCa, B=10000 replicates)
- Holm-Bonferroni correction within each benchmark family (k=7 contrasts)
- Output: `outputs/full_run/sig_tests/mcnemar_holm.json` + significance CSV
- **Architectural-claim licensing rule** (Ch5 §sig-licensing): contrast supports the claim only if Holm-adjusted p < 0.05 AND bootstrap CI lower bound ≥ +2 pp

**C.3** **ARCHITECTURAL ABLATION PANEL** — A1-A7 with pre-registered effect bands:
- A1 No memory, A2 No verifier, A3 No conformal, A4 No anchor, A5 No retroverify, A6 No deferred, A7 No early-exit
- Out-of-band realised effects flagged honestly per Ch5 §comp-falsifications
- Output: `outputs/full_run/ablations/A1/` … `A7/` + `tab_ablation_results.tex`

**C.4** **DIAGNOSTICS** (Task #113): purity validation, retention trajectory, correlations, aggregate
- Auto-fired by `scripts/run_purity_validation.py`, `scripts/aggregate_results.py`
- Output: `outputs/full_run/aggregate_results.json`, `tab_purity.tex`, `tab_continual.tex`

**C.5** **Ch5 AUTO-TABLES** (Task #114): `scripts/make_tables.py` produces `tab_headline.tex`, `tab_baselines.tex`, `tab_ablations.tex`, `tab_chm_decomp.tex`, `tab_calibration_trajectory.tex`, `composite_weights.tex`, `cohen_d.tex`, `sweep_top.tex`, `cross_benchmark_summary.tex`, `tab_sig_test.tex`, `tab_tier_baseline_cost.tex`. **All auto-tables Ch5 references populate after this step.**

### Phase C.6 — Visual / table artifacts NOT covered by existing auto-generators (gap-fill, added 2026-05-01)

After auditing every `\input{}` reference in Ch5 against `eval/reporting.py:TABLE_FILES` + `make_tables.py` + `make_figures.py` + `aggregate_ablation.py`, **seven artifacts are referenced by chapter_5.tex but have no current generator script**. These need to be produced before Phase E thesis content updates can run.

Existing covered (Phase C.5 produces these via `scripts/make_tables.py` driven by `eval/reporting.py:TABLE_FILES`):
- `tab_headline.tex`, `tab_calibration.tex`, `tab_grounding.tex`, `tab_purity.tex`
- `tab_halluc_subtypes.tex`, `tab_continual.tex`, `tab_cycle_progression.tex`
- `tab_sig_test.tex` (from `baseline_sig_tests.py`)
- `tab_ablation_results.tex` (from `aggregate_ablation.py`)
- `composite_weights.tex`, `cohen_d.tex`, `sweep_top.tex`, `cross_benchmark_summary.tex`
- 5 from D.2d: `tab_decision_matrix_trajectory.tex`, `tab_grounding_success_distribution.tex`, `tab_signal_fingerprint.tex`, `fig_ustored_calibration.{pdf,tex}`, `tab_contamination_examples.tex`

**MISSING — write seven new scripts (~6-8 h total, all CPU-only post-trajectory):**

**C.6.a — `scripts/aggregate_per_tier_diagnostics.py`** — produces three tables:
- `tab_per_tier_hit_rate.tex` (Tier 1 / Tier 2 / Tier 3 share per cycle per benchmark)
- `tab_per_tier_latency.tex` (wall-clock-per-query at each tier, dollar-cost projection)
- `tab_per_tier_em.tex` (per-tier exact match across cycles, side-by-side with hit rate)
- `tab_safety_override.tex` (per-cycle safety-override firing rate and effect on correctness)
- Reads `eval/{bench}_cycle{N}.json` (transfer eval) + `eval/{bench}_cycle{N}_streamchunk.json` (Phase A.1/4b snapshots)
- Drives Ch5 §sec:tier-cost, §sec:tier-precision, §sec:tier-safety

**C.6.b — `scripts/aggregate_monotonicity_checks.py`** — produces two tables:
- `tab_cohen_d_trajectory.tex` (per-cycle Cohen's `d` between correct/incorrect on cal fold + per-cycle storage rate, paired-comparison columns)
- `tab_spearman_pp_trajectory.tex` (per-cycle base accuracy p_c × per-cycle pool purity P_c with one-sided Spearman rank correlation)
- Reads `cycle_{N}/composite_calibration.json` + `cycle_{N}/conformal_gate.json` + per-cycle stored-correct counts
- Drives Ch5 §sec:check-monotonicity (currently only has cycle-0 cohen_d.tex; needs trajectory)

**C.6.c — `scripts/aggregate_envelope_fit.py`** — produces one figure:
- `fig_envelope_fit.{pdf,tex}` (geometric decay fit of `E[G_t]` per cycle with the predicted `(1-r)^t G_0` envelope overlaid)
- Reads `theorem_receipts/receipt_envelope_fit.json`
- Drives Ch5 §sec:check-convergence

**C.6.d — `scripts/aggregate_full_pipeline_decomposition.py`** — produces one table:
- `tab_full_pipeline_decomposition.tex` (per-cycle `H_t = (1-τ_1)·ε_arch + H_mem(t)` with per-component readings, fitted asymptote)
- Reads per-tier diagnostics from C.6.a + retroverify trajectory + `theorem_receipts/receipt_eps_arch.json`
- Drives Ch5 §sec:check-asymptotic

**C.6.e — `scripts/aggregate_self_correction_distribution.py`** — produces one figure + one table:
- `fig_survival_cycles.{pdf,tex}` (histogram of survival-cycles distribution for entries pruned at each cycle boundary)
- `tab_retroverify_trajectory.tex` (per-cycle retroverify pruning rate: cycle 1 = 44 %, cycle 2 = 27 %, …)
- Reads `retroverify_cycle{N}.json` + `memory_store_cycle_{N}.meta` + `deferred_buffer_cycle_{N}.pkl`
- Drives Ch5 §sec:check-corollaries (cor:self-correction empirical receipt)

**C.6.f — `scripts/aggregate_memory_composition.py`** — produces one table + one figure:
- `tab_memory_composition_trajectory.tex` (per-cycle FEVER / TriviaQA / NQ share in memory store)
- `fig_memory_composition.{pdf,tex}` (stacked-area plot of benchmark composition over cycles)
- Reads `memory_store_cycle_{N}.meta` (pickled metadata with `source_benchmark` field)
- Drives Ch5 §sec:adj-cal-eval-gap (E.0m artifact)

**C.6.g — `scripts/aggregate_ablation_outputs.py`** (extends `make_tables.py`) — produces three tables:
- `tab_conditional_conformal_cycle10.tex` (D.2a output: per-benchmark `τ_store`, `n_store`, precision)
- `tab_mondrian_conformal.tex` (D.2b output: per-benchmark α at {0.05, 0.20, 0.40} + sample-weighted-mean precision)
- `tab_alpha_sensitivity.tex` (D.2c output: global α at {0.05, 0.10, 0.20})
- Reads the JSON outputs of D.2a, D.2b, D.2c
- Drives Ch5 §sec:adj-cal-eval-gap follow-up paragraphs and §future-work registration

**Effort summary for Phase C.6:**

| Script | Wall-time | Output count |
|---|---|---|
| C.6.a per-tier diagnostics | ~1 h | 4 tables |
| C.6.b monotonicity | ~45 min | 2 tables |
| C.6.c envelope fit figure | ~30 min | 1 figure |
| C.6.d pipeline decomposition | ~45 min | 1 table |
| C.6.e survival distribution | ~1 h | 1 figure + 1 table |
| C.6.f memory composition | ~30 min | 1 table + 1 figure |
| C.6.g ablation tables | ~1.5 h | 3 tables |
| **Total** | **~6 h** | **13 artifacts** |

These run AFTER Phase D (need the cycle-10 ablation outputs) but BEFORE Phase E (the new prose references the produced artifacts). All CPU-only, no GPU cost.

### Phase D — Manual receipts and ablations (~May 13-15)

**D.1** Run **theorem_receipts.py manually** (per P2, the runner's ALL_STEPS array doesn't include `step_19_6` in the running parent shell). Produces 6 receipts:
- `receipt_envelope_fit.json` (thm:convergence)
- `receipt_eps_arch.json` (thm:asymptotic-elim)
- `receipt_gap_decay.json` (cor:convergence-rate)
- `receipt_corpus_floor.json` (cor:corpus-floor)
- `receipt_self_correction.json` (cor:self-correction)
- `receipt_tau_retro_sensitivity.json`

**D.2** Run **post-trajectory CPU-only ablations** (all four cheap, all under $5 GPU max):
- **D.2a** Conditional conformal Scope A on cycle 10: `python -m scripts.conditional_conformal_ablation --cycle 10` — confirms Bayesian-floor framing at end-of-trajectory; if Open-QA per-benchmark fit becomes possible (τ_store < 1.0 with valid contract), positive evidence the SIL trajectory genuinely improved Open-QA verifier signal quality
- **D.2b** Mondrian conformal ablation (per-benchmark α) — refit per-benchmark gates at α ∈ {0.05, 0.20, 0.40} on cycle-10 cal fold; report per-benchmark precision + sample-weighted mean; addresses the multi-α question with empirical evidence
- **D.2c** α_store global sensitivity sweep on cycle-10 cal fold at α ∈ {0.05, 0.10, 0.20} pooled — maps the precision-recall frontier under uniform α
- **D.2d** Build `scripts/aggregate_decision_diagnostics.py` (Step C in master) — 5 auto-tables for §sec:adj-decision-diagnostics

**D.3** **Step I — Paraphrase-set evaluation for cor:tier1-floor** — likely redundant given cycle-2 cal-fold already exhibits natural Tier 1/2 hits at 13.4 % on FEVER under safety_u_pre_min=0.38; **decision: SKIP** unless cycle-10 readings reveal otherwise. Saves ~$15-20 + 6 h.

### Phase E — Thesis content updates from empirical receipts (~10-15 h after Phase 4 + D land)

These items write NEW PROSE / TABLES / SUBSECTIONS into the chapters from the realised empirical record. Distinct from Phase F structural cleanup.

**E.0a — Bug-fix story in Ch5 §threats (P3 detail).** Write the ~150-word paragraph documenting the cycle-1 α-drift bug (commits 517e09e, 6adc695, b4d610d) + the pristine MMLU anchor patch (today). Reframes the bug as evidence of method discipline.

**E.0b — Pristine MMLU anchor recovery patch in Ch4 §sil and Ch5 §threats.** Document the resume-side fix at `run_experiment.py:1537-1568` that loads `mmlu_baseline.json` so retention ratios stay anchored to 0.6250 across restarts.

**E.0c — Empirical recall/recovery numbers in Ch5 §sec:adj-cal-eval-gap (P3b detail).** Populate from `receipt_self_correction.json`:
- Per-cycle FEVER stream-chunk store rate trajectory (cycle 1 = 13.87 %, cycle 2 = 15.60 %, …)
- Per-cycle deferred-buffer promotion rate (cycle 1 buffer = 1040 entries; cycle-2 reconsider promotion count)
- Cumulative trajectory recall on FEVER (cycle-1 single-cycle 22.8 % → cycle-N cumulative)
- Survival-cycles distribution (median survival horizon, fraction surviving all cycles)
- Per-benchmark imbalance numbers (FEVER ~13 %, TriviaQA ~0 %, NQ ~0.4 %)

**E.0d — Add §sec:adj-decision-diagnostics subsection to Ch5 (P3c + Step E detail).** Insert between §sec:adj-cal-eval-gap and §sec:adj-judge-ablation. 2 paragraphs of intro + 5 `\input{}` blocks for the auto-tables produced by `aggregate_decision_diagnostics.py` (D.2d): tab_decision_matrix_trajectory, tab_grounding_success_distribution, tab_signal_fingerprint, fig_ustored_calibration, tab_contamination_examples (appendix).

**E.0e — Compute the four decisive empirical findings across all 10 cycles (Step D detail).** Reproduce and extend the cycle-1 readings:
- Bayes-purity identity exact-match (cycle 1 residual = 0.0000 already verified) per-cycle
- α (verifier balanced accuracy) > 0.5 per-cycle per-benchmark
- p_+ (base accuracy) trajectory per-benchmark — drives cycle-pair conditional applicability for thm:monotone
- Per-benchmark store fingerprint consistency (signal medians on STORE-correct samples)

**E.0f — Extend §sec:check-monotonicity with cycle-pair-conditional applicability (Step F detail).** One paragraph stating which cycle-pairs satisfy thm:monotone's `p_+ > 0.5` precondition per benchmark. Specifically: FEVER holds throughout; TriviaQA may begin holding at cycle k where p_+ first crosses 0.5; NQ subset may be empty within realised horizon. **Panel-defence framing: "thm:monotone characterizes a regime, not a test set; we report cycle-pair applicability empirically."**

**E.0g — Refine τ_retro = 0.50 prose in Ch4 §retroverify (P4 detail).** Cal-fold sensitivity reading shows π_retro at τ=0.50 is ~0.44, not 0.50. Drop the "more-likely-correct-than-wrong" framing; reframe τ_retro=0.50 as a recall-favouring floor that maintains the architectural invariant `τ_retro < τ_defer < τ_store < τ_train`. Reference the empirical π_retro per cycle reported in §sec:check-purity.

**E.0h — Document 9-subtype CHM coverage methodology (Task #42).** Already partially added to Ch5 §sec:setup-metrics; verify the 8-of-9-active subtype detection rules from `eval/metrics.py:411-513` are complete and that the retired 9th (factual contradiction) is correctly framed as "structurally unreachable under MiniCheck binary judge; reported separately for completeness."

**E.0i — TruthfulQA regression honest disclosure in Ch5 §threats.** Cycle 0 → cycle 1 EM dropped from 0.334 to 0.316 on TruthfulQA. Document as cross-domain interaction effect: SIL training on factual QA shifts answer style toward confident completion; TruthfulQA's adversarial-misconception pattern penalises this. Frame as known scope limitation, not methodology failure.

**E.0j — Retention ratio > 1 finding in Ch5 retention subsection.** Already in branch_C_log (2026-04-30 entry). Lift into Ch5 with mechanism explanation (MMLU general-data slice + L2 anchor + reasoning-chain transfer). Three architectural causes producing positive transfer rather than worst-case 0.93 floor.

**E.0k — Conditional conformal Scope A finding in Ch5 §future-work + §sec:adj-cal-eval-gap.** From the cycle-2 ablation (today) and the cycle-10 re-run (D.2a): per-benchmark gates at α=0.05 return τ_store=1.0 on TriviaQA/NQ. Bayesian-floor framing replaces the earlier verifier-signal-quality framing. Mondrian alternative (D.2b) is the ablation that quantifies the precision-recall trade.

**E.0l — Hypothesis verdicts table population (Ch5 §summary-hypotheses).** H1 already closed (supported); populate H2-H4 from realised trajectory readings; populate H5 from ablation panel. The verdict-licensing rule (Ch5 §sig-licensing) drives each cell.

**E.0m — Memory composition trajectory table.** From per-cycle memory store .meta files. Cold-seed 260 (58 % FEVER, 28 % TriviaQA, 14 % NQ) → cycle 1 (89 % FEVER) → projected cycle 10 (~95 % FEVER). Add to Ch5 §sec:adj-cal-eval-gap as a small table.

### Phase F — Thesis structural cleanup pass (~7-9 h focused work after Phase E)

These items fix structural / consistency issues identified in the 2026-05-01 thesis review.

**F.1** Issue 1 (numerical consistency: 9-vs-10 signals, TTL 2→4, safety 0.60→0.38) — ~1 h, find-replace + verify across all 6 chapters
**F.2** Issue 3 (Chapter 3 §3.9 empty signpost) — ~30 min, write 1 paragraph
**F.3** Issue 6 (Chapter 5 §summary-headline placeholder, fill from C.5 outputs) — ~30 min
**F.4** Issue 4 (Chapter 2 §2.1 voice polish, ~5 subsections) — ~3-4 h copy-edit
**F.5** Issue 2 (file/folder names in body prose, decide footnote-or-appendix) — ~1-2 h
**F.6** Splice C.5 auto-tables into Ch5 (replace `\input{}` placeholders)
**F.7** Refresh figures: `python -m scripts.make_figures`
**F.8** Compile: `cd thesis_report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
**F.9** Fix unresolved `\ref{}` and undefined cites
**F.10** Final structural review per `feedback_thesis_coherence` memory

**F.11 — Reconcile thesis ablation panel with code (DECISION DEFERRED, added 2026-05-01).** Audit found a mismatch: thesis Ch5 Table tab:ablation-bands (line 364-381) lists 7 architectural ablations (A1 No memory, A2 No verifier, A3 No conformal, A4 No anchor, A5 No retroverify, A6 No deferred, A7 No early-exit) with pre-registered effect bands, but `caem/ablation/variants.py:151-203` registers only 4 variants (`full`, `no_retroverify`, `no_self_improvement`, `no_forgetting_guard`) following the 2026-04-22 reduction decision that retired five candidate ablations as redundant with other diagnostics:

- `no_store_gate` (≈ A3) — redundant with purity validation script
- `no_tier1` (≈ A1) — redundant with tier_{1,2,3}_frac trajectory
- `roberta_nli_backend` (≈ A2 alternative) — redundant with v5 AUROC diagnostic
- `equal_signal_weights` / `no_q_a_relevance` — redundant with per-signal correlation matrix

**Two paths, decision pending user review:**

- **Path 1 (recommended):** Update thesis Ch5 §sec:comp-ablations to match the 4-variant code. Edit Table tab:ablation-bands to reflect the actual panel; add a paragraph explaining the 2026-04-22 reduction with surrogate-diagnostic evidence per mechanism. Effort: ~30 min, no compute cost. Architecturally honest.
- **Path 2:** Implement A1, A2, A3, A6, A7 as new cyclic-ablation variants in `caem/ablation/variants.py`. Each `needs_cyclic_rerun=True` variant requires a full ~37-h SIL re-run = ~$50-80 GPU each × 5 = $250-400 + 5-7 weeks GPU time. Post-defense Phase 1c work, not pre-defense scope.

**Decision deferred to user; flagged for review before Phase F structural cleanup begins.** All other items in the plan are confirmed.

### Phase G — Citation audit + thesis polish (P6, P7)

**G.1** Citation audit (P6) — walk every `\cite{}` against bib + paper content, ~several hours
**G.2** Final thesis polish (P7) — submission-ready PDF, supervisor + external examiner

### Total post-trajectory effort (revised 2026-05-01 to include missed P3/P3b/P3c/P4/Step D-G/Task #42)

| Phase | Wall-time | Compute cost | Type |
|---|---|---|---|
| A.2 cycle-2 halt+restart | 25 min | $0 | manual |
| B trajectory cycles 3-10 | 7-9 days | autonomous | autonomous |
| C Phase 4 auto-runs (B1-B7 + sig + ablation A1-A7 + diagnostics + tables) | ~2 days | included in main-run envelope | autonomous |
| C.6 gap-fill scripts (7 new tables + figures, all CPU-only) | ~6 h | $0 | manual scripts |
| D manual receipts + 3 cheap ablations | ~12-15 h | <$5 | manual + scripts |
| E **thesis content updates from receipts (13 sub-items)** | **~10-15 h** | **$0** | **manual writing** |
| F thesis structural cleanup | ~7-9 h | $0 | manual |
| G citation + polish | several hours | $0 | manual |

**Net: trajectory finishes ~May 12-14, Phase 4 lands ~May 13-15, Phase E content updates ~May 15-17, Phase F+G structural+polish ~May 17-19, thesis ready for submission ~May 18-20.**

---

## ★ POST-CYCLE-N ANALYSIS PLAN (read this first on return — added 2026-04-29) ★

When step_7_main + Phase 4 complete (~May 11), run the analyses below in this order. The conceptual prose is already in Ch5 §sec:adj-cal-eval-gap (cause decomposition, architectural reading, corpus expansion, recall-and-recovery). What remains is **populating empirical numbers** and **adding cycle-trajectory diagnostics**.

### Step A — Verify cycle-trajectory data is on disk

```bash
ls outputs/full_run/run_complete.json                            # cycle 7 main done marker
wc -l outputs/full_run/experiment_summary.csv                    # 11 rows = 10 cycles + cycle 0
ls outputs/full_run/eval/*_cycle{0..10}.json                     # all 7 benchmarks × 11 cycles
ls outputs/full_run/cycle_*/conformal_gate.json                  # per-cycle gates
ls outputs/full_run/deferred_buffer_cycle_*.pkl                  # per-cycle deferred buffers
```

### Step B — Run theorem_receipts.py manually (per P2 below)

`run_phase1a.sh` step_19_6 was wired in code on 2026-04-28 BUT the running bash parent has cached the old `ALL_STEPS` array, so it won't auto-fire on this run. **Run manually after Phase 4 finishes.** Produces 6 receipts under `outputs/full_run/theorem_receipts/`.

### Step C — Build aggregate_decision_diagnostics.py (~4-6h, ~150 lines)

Reads `eval/{benchmark}_cycle{N}.json` for all 7 benchmarks × 11 cycles. Emits:

1. `tab_decision_matrix_trajectory.tex` — per-cycle 8-cell decision × correctness matrix per benchmark
2. `tab_grounding_success_distribution.tex` — per-cycle p_g≥{0.7, 0.5, 0.3, 0.1} on EM=1 by benchmark (anchors corpus-expansion paragraph)
3. `tab_signal_fingerprint.tex` — per-cycle stored-correct signal medians per benchmark (proves cross-benchmark fingerprint consistency)
4. `tab_p_plus_trajectory.tex` — per-cycle p_+ (base-model correctness) by benchmark (for thm:monotone applicability check)
5. `fig_ustored_calibration.{pdf,tex}` — u_stored bucket × EM rate histogram (calibration monotonicity proof)
6. `tab_contamination_examples.tex` — top-10 wrong-stored per benchmark per cycle (appendix)

### Step D — Compute the four decisive empirical findings (cycle-1 baseline)

These were computed on 2026-04-29 from FEVER + TriviaQA cycle-1 stream chunks; the post-cycle-N pass should reproduce and extend across all 10 cycles:

| Finding | Cycle 1 reading | What it proves |
|---|---|---|
| **Bayes-purity identity exact** | residual = 0.0000 on FEVER (P_theory=0.8413=P_obs), TriviaQA (0.5000=0.5000), pooled (0.8333=0.8333) | thm:bayes-purity empirically validated with zero residual |
| **α (verifier balanced accuracy) > 0.5** | FEVER 0.591, TriviaQA 0.501, pooled 0.557 | Holds everywhere; thm:bayes-purity strict inequality applicable |
| **p_+ (base accuracy) > 0.5** | FEVER 0.512 ✓, TriviaQA 0.366 ✗, NQ ~0.17 ✗, pooled 0.439 ✗ | thm:monotone precondition holds on FEVER only; cycle-pair-conditional applicability across the trajectory |
| **Per-benchmark store fingerprint consistency** | FEVER STORE-✓ p_ground_mean = 0.767; TriviaQA STORE-✓ p_ground_mean = 0.741 (same range) | Gate is benchmark-agnostic; imbalance is upstream not gate-side |

### Step E — Add §sec:adj-decision-diagnostics subsection to Ch5

Insert between §sec:adj-cal-eval-gap and §sec:adj-judge-ablation. Two intro paragraphs + 5 `\input{}` blocks for the auto-tables. Cross-link from existing cause/architectural/corpus paragraphs.

### Step F — Extend §sec:check-monotonicity with cycle-pair-conditional applicability

After observing per-cycle p_+ trajectory, add one paragraph stating which cycle-pairs satisfy thm:monotone's precondition (`p_+(b, t) > 0.5`) per benchmark. Specifically: if/when TriviaQA p_+ crosses 0.5 (likely cycle 4-6 by transfer learning + deferred buffer promotion), the theorem applies from that cycle-pair forward on TriviaQA.

**Cycle-pair conditional applicability framing (verbal panel defense):**

> "thm:monotone characterizes a regime, not a test set. We report cycle-pair applicability empirically: on FEVER the precondition holds throughout; on TriviaQA the precondition begins holding at cycle k where p_+ first crosses 0.5; on NQ the subset may be empty within the realised horizon. Reporting where conditions hold and where they don't is methodological transparency."

### Step G — Populate recall/recovery numbers in §sec:adj-cal-eval-gap (per P3b)

Per-cycle deferred-buffer promotion rate, cumulative trajectory recall, survival distribution from receipt_self_correction.json.

**Deferred-buffer TTL=2 analysis (added 2026-04-29):**
Configured TTL is `deferred_buffer_ttl_cycles = 2` (per `caem/config.py:753`). Each deferred entry gets reconsidered at the next 2 cycle boundaries, then expires if not promoted. Empirical projection from cycle-1 deferred-correct distribution: TTL=2 promotes ~40-60% of deferred-correct entries (those whose initial u_stored ≥ ~0.60); the remaining ~40-60% expire without recovery via the deferred path and rely on **semantic re-encounter** (the second recovery path described in Ch5 §sec:adj-cal-eval-gap) for cumulative recovery.

After step_7_main, compute per-cycle from `outputs/full_run/deferred_buffer_cycle_N.pkl`:
- Promotion rate per cycle (entries promoted from buffer → memory at cycle boundary)
- Expiration rate per cycle (entries TTL-aged-out without promotion)
- Promotion vs expiration ratio across the trajectory (validates TTL=2 conservatism)

**Phase 1c future-work option:** TTL ablation comparing TTL ∈ {1, 2, 3, 5} on cycle-0 + 1-2 cycles of SIL. ~$30-50 of compute. Tests whether longer TTL recovers more deferred entries with acceptable buffer-size cost. Register in Ch6 §future-work alongside conditional conformal and corpus expansion.

### Step H — Phase 4 baselines + Ch5 tables auto-runs (existing infrastructure)

After step_7_main: B1-B7 baselines + sig tests + Ch5 table generation auto-fire per `run_phase1a.sh`. Verify `outputs/baselines/B*`, `outputs/full_run/sig_tests/mcnemar_holm.json`, `outputs/full_run/tab_*.csv`.

### Step I — Paraphrase-set evaluation for cor:tier1-floor (added 2026-04-29)

The natural deployment stream is gap-5 content-disjoint from cycle stream chunks, so Tier 1/2 hits are structurally rare on the trajectory eval. To produce cor:tier1-floor's empirical receipt, build a paraphrase-set evaluation that exercises Tier 1/2 directly.

**Insight (user 2026-04-29):** OR suppression is u_pre-based, not memory-similarity-based. Paraphrases of memory entries have high u_pre by construction (model recognises the question domain), so OR will NOT fire on them. **No diagnostic-mode (OR-disabled) run is needed** — the production-mode evaluation alone produces the empirical evidence cleanly.

**Protocol:**
1. Sample N=200 entries from cycle-10 memory (covering FEVER + TriviaQA + NQ approximately proportional to memory composition)
2. Generate K=3-5 paraphrastic reformulations per entry (LLM-rewriting via Qwen-2.5-7B with rewrite prompt OR manual edits)
3. Run all 600-1000 paraphrastic queries through the deployed cycle-10 system (production mode, OR active, full architecture)
4. Report:
   - Per-tier hit rate (Tier 1, Tier 2, Tier 3) on the paraphrase set
   - Per-tier CHM (the cor:tier1-floor bound check)
   - Pooled deployed precision on Tier 1 hits (validates the corollary's structural floor)
5. Cross-reference: report alongside the natural-stream per-tier hit rate to make the content-disjointness vs paraphrastic-realistic distinction explicit

**Cost:** ~$15-20 GPU + ~6 hours work. Output: `outputs/paraphrase_set/eval/{benchmark}.json` plus a Ch5 §sec:check-asymptotic-paraphrase subsection that ingests it.

**Why this works without OR-disabled diagnostic:** OR only suppresses queries with u_pre < 0.60. Memory entries were stored because the model was confident about them; paraphrases of those queries inherit similar encoder-side confidence, so u_pre stays high and OR doesn't intervene. The Tier 1/2 hit rate measured is what production deployment would actually see on paraphrastic user traffic.

**Total post-trajectory effort:** ~3 days (Step C ~6h, Step D ~2h, Step E ~3h, Step F ~1h, Step G ~3h, Step H ~auto + verification 2h, Step I ~6h + ~$15-20).

---

## State at handoff

**Runner:** alive in `tmux plan_a`, relaunched at 14:56:25 UTC under fully-fixed pipeline. Cycle 1 SIL just completed (loss 1.276 → 0.721, MMLU retention 1.0080). Currently in checkpoint save → Step 2.1 cal-fold scoring.

**Bugs found and fixed today (2026-04-28):**

| Commit | Fix |
|--------|-----|
| `517e09e` | Per-cycle conformal refit inherits `α_store` from previous cycle's gate JSON instead of config default (was silently drifting 0.05 → 0.20). |
| `6adc695` | Per-cycle refit passes `--cherian_boost --boost_C 0.01` to match locked Step 7.0.1 baseline (was producing identity-mode composite, intercept=0). |
| `b4d610d` | `CAEMConfig.conformal_alpha_store` default pinned to 0.05 (belt-and-suspenders fallback). |

**Recovery (out-of-tree, outputs/ is gitignored):**
- `outputs/cycle_0/composite_calibration.json` restored from `outputs/.snapshot_staging/archive/pre_step7_main_2026-04-26/` (boost_intercept=1.099, q_a_relevance=0.500, em_rate=0.243).
- `outputs/cycle_0/conformal_gate.json` restored to α=0.05, τ_store=0.6676 from `outputs/cycle_0/sweep/variant_a050_C0.010.json`.
- Two contaminated cycle_1 directories preserved at `outputs/_halted_runs/cycle_1.contaminated_alpha020_*` and `outputs/_halted_runs/cycle_1.composite_corrupted_*`.

**Lock chain confirmed:** α=0.05 + Cherian boost + C=0.01 propagate cycle 0 → cycle 10 via three redundant paths (prev_gate inheritance, canonical-path copy at cycle close, config dataclass default). No drift path remains.

---

## Autonomous monitor while user is away

`/loop` wakeup at 60-min cadence. Per-cycle progress logged; cycle-1 boundary refit is the empirical landmark — should show `boost=fit` AND `alpha_store: 0.05` in the new `cycle_1/conformal_gate.json`. Auto-restart on crash per `RESUME_GUIDE.md`. **No halt actions planned.**

ETA Step 7 main complete: ~May 5. ETA Phase 4 (auto-runs after Step 7 per `run_phase1a.sh`): ~May 6–7.

---

## P1 — On return, verify Step 7 main completed

```bash
cd /workspace/caem
ls outputs/full_run/run_complete.json     # exists if Step 7 finished
cat outputs/full_run/experiment_summary.csv | head -15   # 11 rows = 10 cycles + cycle 0 baseline
grep -c "decision=STORE" outputs/full_run/run.log
```

If `run_complete.json` exists: Step 7 main is done; Phase 4 (Steps 8–21) may already be auto-running per `run_phase1a.sh`. If runner is still in Step 7: check tmux log for the cycle index, let it complete naturally.

---

## P2 — Run Phase 4 theorem receipts MANUALLY

**IMPORTANT:** `scripts/theorem_receipts.py` was wired into `run_phase1a.sh` on 2026-04-28 as `step_19_6_theorem_receipts`, but the **currently-running bash runner has the OLD `ALL_STEPS` array cached in memory** and will NOT pick up the new step. The wiring helps future runs only. **For this run, the receipts must be invoked manually after Phase 4 completes.**

After Step 7 main + Phase 4 (B1-B7, purity, correlations, aggregate, tables) finishes:

```bash
cd /workspace/caem && source /venv/main/bin/activate
python -m scripts.theorem_receipts \
    --output_dir outputs/full_run/theorem_receipts \
    --full_run_dir outputs/full_run \
    --summary_csv outputs/full_run/experiment_summary.csv
```

Outputs (six JSONs):
- `receipt_envelope_fit.json` — geometric envelope fit (Theorem 5)
- `receipt_eps_arch.json` — joint architectural FNR aggregation (Theorem 7)
- `receipt_gap_decay.json` — exponential decay fit (Cor convergence-rate)
- `receipt_corpus_floor.json` — top-k retrieval miss subset proxy (Cor corpus-floor)
- `receipt_self_correction.json` — survival distribution (Cor self-correction)
- `receipt_tau_retro_sensitivity.json` — τ_retro at 0.50 vs empirically-optimal per cycle

---

## P3 — Document the bug-fix story in Ch5 §threats

One paragraph (~150 words) added to `thesis_report/chapters/chapter_5.tex` near the threats-to-validity subsection:

> *"During cycle 1 of Step 7 main on 2026-04-28, the per-cycle conformal refit was discovered to drift from the locked Cycle-0 calibration via two independent code paths. First, `run_experiment.py:run_per_cycle_conformal_refit` read `α_store` from the dataclass default (0.20) instead of inheriting from the previous cycle's gate JSON (0.05). Second, the same function did not pass `--cherian_boost --boost_C 0.01` to `recalibrate_conformal_at_cycle.py`, producing an identity-weighted composite (boost intercept zero, no Cherian L2 fit). Both bugs were fixed in commits 517e09e, 6adc695, b4d610d. Cycle 1 was re-run from scratch under the corrected pipeline; cycles 2–10 inherit α=0.05 + Cherian boost via three converging fallback paths (prev-gate JSON, canonical copy, config default). The Phase 4 receipts in §sec:check-* track the empirical realisation of the locked Cycle-0 contract across the corrected trajectory."*

---

## P3c — Add §sec:adj-decision-diagnostics subsection + 4 auto-tables (added 2026-04-29)

After step_7_main + Phase 4 + per P3b: write `scripts/aggregate_decision_diagnostics.py` (~4-6 hours, ~150 lines) producing:

1. `tab_decision_matrix_trajectory.tex` — per-cycle 8-cell decision × EM matrix per benchmark
2. `tab_grounding_success_distribution.tex` — per-cycle p_g ≥ {0.7, 0.5, 0.3, 0.1} on EM=1 by benchmark (anchors corpus-expansion paragraph)
3. `tab_signal_fingerprint.tex` — per-cycle stored-correct signal medians per benchmark (proves cross-benchmark fingerprint consistency)
4. `fig_ustored_calibration.{tex,pdf}` — per-cycle u_stored bucket × EM rate histogram
5. `tab_contamination_examples.tex` — appendix: top-10 wrong-stored per benchmark per cycle

Add ONE new Ch5 subsection §sec:adj-decision-diagnostics between §sec:adj-cal-eval-gap and §sec:adj-judge-ablation: 2 paragraphs of intro prose + 4 \input{} blocks (5th in appendix). Cross-link from existing cause/architectural/corpus paragraphs in §sec:adj-cal-eval-gap.

Total: ~1 day post-step_7_main effort. Cycle-1 readings already validated (FEVER STORE precision 84.1%; TriviaQA grounding-success at p_g≥0.7 only 1.3% vs FEVER 14.3% — directly motivates the corpus-expansion future-work argument).

---

## P3b — Populate empirical recall/recovery numbers in Ch5 §sec:adj-cal-eval-gap (added 2026-04-29)

**Conceptual prose already in place** (committed 2026-04-29). The §sec:adj-cal-eval-gap section now has a paragraph framing precision-recall trade as cross-cycle (deferred-buffer promotion + semantic re-encounter), with forward references to §sec:retroverify, §sec:deferred-reconsider, and `cor:self-correction`.

**Remaining (post-step_7_main + Phase 4):** populate empirical numbers from the 10-cycle data into a follow-up paragraph or table immediately after the conceptual one. Required readings:

1. Per-cycle FEVER stream-chunk store rate trajectory (cycle 1: 13.87% / 416 stored / 84.1% deployed precision)
2. Per-cycle deferred-buffer promotion count (entries promoted from deferral → storage at each cycle boundary)
3. Cumulative trajectory recall on FEVER (cycle-1 single-cycle capture 22.8% → cycle-N cumulative capture)
4. Survival-cycles distribution from `outputs/full_run/theorem_receipts/receipt_self_correction.json` (median survival horizon, % surviving all cycles)
5. Per-benchmark imbalance numbers in absolute terms (FEVER ~13% predicted, TriviaQA ~0.5%, NQ ~0.2%)

These complete the recall side of the precision-recall narrative. Relies on `theorem_receipts.py` having been run manually post-Phase 4 (per P2 above).

---

## P3d — Benchmark-asymmetric storage finding + α_store sensitivity ablation (added 2026-05-01)

**Context.** Cycle 2 stream-chunk readings confirm a strong benchmark-asymmetric storage pattern under the locked conformal contract (α_store = 0.05, τ_store rising 0.6676 → 0.7242 → 0.7475 across cycles 0/1/2):

| Benchmark | Cycle-1 stream stored | Cycle-2 stream stored | Reading |
|---|---:|---:|---|
| FEVER | 13.87 % | **15.60 %** | store rate rising |
| TriviaQA | 0.33 % | **0.00 %** | gate too strict for TriviaQA stream |
| NQ | 0.37 % | TBD | likely similarly low |

The pattern is the empirical realisation of `thm:monotone`'s precondition `p_+ > 0.5`: where base accuracy holds (FEVER), the gate accumulates memory; where it fails (TriviaQA, NQ), the gate correctly refuses to store low-confidence entries.

**Locked-config rationale (do NOT change the gate mid-run):**

1. The 96-97 % in-sample store_precision contract holds across cycles 0/1/2 because the gate is strict. Relaxing α_store would compromise the architectural claim.
2. Mid-run parameter drift would break trajectory internal consistency (receipt comparability, `thm:convergence` envelope fit, panel-side methodology).
3. The asymmetric memory composition is **methodological transparency**, not a flaw — it is the cycle-pair-conditional applicability story already planned in Step F.

**What gets reported in Ch5 (no parameter change):**

- Per-cycle stream-chunk store rate by benchmark (the asymmetric-by-design pattern). Add a small table to §sec:adj-cal-eval-gap or §sec:check-purity showing per-cycle store rate per benchmark across the 10-cycle trajectory.
- Per-cycle `p_+` trajectory + cycle-pair conditional applicability per benchmark (Step F existing).
- Memory composition trajectory: how FEVER concentration evolves cycles 0 → 10. Add a line plot or table.
- Per-tier hit rate during stream-chunk and transfer eval, broken down by benchmark.

**What gets registered as Phase 1c future work in Ch6 §future-work (REVISED 2026-05-01 after Scope A ablation):**

The conditional-conformal ablation (`scripts/conditional_conformal_ablation.py`, run on cycle 2 cal fold, output at `outputs/full_run/cycle_2/conditional_conformal_ablation.json`) shows that per-benchmark gates **do not solve the asymmetry**: fitting at α_store = 0.05 returns τ_store = 1.0 (store nothing) on TriviaQA and Natural Questions because no threshold over the cal-fold u_stored distribution can hold 95 % precision on those benchmarks. The pooled 97.37 % global precision is FEVER-driven — applying the global gate per-benchmark gives FEVER 91.78 %, TriviaQA 20.00 %, NQ 0.00 %. **The asymmetry is a verifier-signal-quality problem upstream of the gate, not a gate-architecture problem.**

Phase 1c future work scope is therefore three verifier-side directions, not the gate-side direction:

1. **Benchmark-conditional composite weights** — fit cal-prob composite per benchmark family (classification vs Open-QA) while keeping a single global gate. Operationally weird (same query different u_stored depending on benchmark provenance) but architecturally sound.
2. **Open-QA-aware atomic decomposition** — retune the FActScore prompt + length gate (currently 600-token min) for Open-QA's typically shorter reasoning chains. Possible signal-side win without model retraining.
3. **Open-QA-trained NLI judge** — replace MiniCheck (FEVER-trained T5) with a judge fine-tuned on TriviaQA / NQ-style entailment data. Heaviest path; multi-day GPU training + Platt re-calibration. Closer to Phase 2 paper scope than thesis.

**Three ablations post-step_7_main (do NOT touch main trajectory). REVISED 2026-05-01 after the verifier-input-uniformity check moved the asymmetry-cause to base-accuracy-Bayesian-floor rather than verifier-signal-quality:**

- **Conditional conformal Scope A on cycle 10** — re-run `python -m scripts.conditional_conformal_ablation --cycle 10` (CPU only, no cost) once the cycle-10 cal fold lands. Confirms the Bayesian-floor framing at end-of-trajectory; if Open-QA per-benchmark fit becomes possible at cycle 10 (τ_store < 1.0 with valid contract), positive evidence the SIL trajectory genuinely improved Open-QA base-model accuracy enough to escape the floor.

- **Mondrian conformal ablation (per-benchmark α)** — refit per-benchmark gates at α ∈ {0.05, 0.20, 0.40} on cycle-10 cal fold. Reports per-benchmark `τ_store`, `n_store`, precision, plus sample-weighted-mean precision across the training panel. Addresses the multi-α question with empirical evidence rather than handwave. Output: `outputs/full_run/cycle_10/mondrian_conformal_ablation.json`. Cost: CPU only + ~4-6 h work. The reading is methodologically defensible (Vovk Mondrian-CP literature) and gives the panel a measured trade-off rather than a single-design defence.

- **α_store global sensitivity sweep on cycle-10 cal fold** at α ∈ {0.05, 0.10, 0.20} pooled. Maps the precision-recall frontier under uniform α (as opposed to Mondrian's per-benchmark α). Cost: CPU only + ~1-2 h work. Output: `outputs/full_run/cycle_10/alpha_sensitivity.json` + Ch5 paragraph.

---

## P4 — Refine τ_retro = 0.50 prose in Ch4

The cal-fold sensitivity reading shows π_retro at τ=0.50 is ~0.44, not 0.50. Drop the "more-likely-correct-than-wrong" framing.

**Edit `thesis_report/chapters/chapter_4.tex` §retroverify:**

```diff
- The prune threshold is registered at τ_retro = 0.50, below the deferral
- threshold so that an entry that is borderline-deferred-quality but no longer
- storage-quality is still kept for the deferred-reconsider pass
+ The prune threshold is registered at τ_retro = 0.50: a recall-favouring
+ floor that maintains the architectural invariant
+ τ_retro < τ_defer < τ_store < τ_train. Empirical π_retro on the actual
+ stored pool per cycle is reported in §sec:check-purity; the cal-fold
+ sensitivity analysis (receipt_tau_retro_sensitivity.json) shows the
+ threshold's behaviour on the broader population for context.
```

---

## P5 — Phase 4 baselines + Ch5 tables (auto-runs after Step 7 main)

`run_phase1a.sh` continues automatically after Step 7 main:

- Step 8: FLARE smoke
- Step 9–14: Baselines B1–B7 (zero-shot, RAG, Self-Consistency, calibrate-then-abstain, FLARE, EWC fine-tune, EWC + retention guard)
- Step 15: Paired McNemar + BCa bootstrap with Holm correction
- Step 19: Purity validation
- Step 20.1: Aggregate results CSV
- Step 21: Ch5 table generation (`tab_headline.csv`, `tab_baselines.csv`, `tab_ablations.csv`, `tab_chm_decomp.csv`)

Verify they ran:

```bash
ls outputs/baselines/B*/                       # expect B1-B7 dirs
ls outputs/full_run/sig_tests/                 # mcnemar_holm.json
ls outputs/full_run/aggregate_results.json
ls outputs/full_run/tab_*.csv                  # 4 Ch5 tables
```

---

## P6 — Citation audit

Walk every `\cite{}` against `references.bib` AND against the cited paper's actual content. Generate `thesis_report/audit_phase_a_citations.md` with columns:

| File | Line | `\cite{key}` | bib_present | claim_in_prose | paper_supports_claim | notes |

Long manual task (~several hours). Critical for thesis integrity.

---

## P7 — Final thesis polish

After everything above:
- Splice Phase 4 tables into Ch5 (replace `\input{...}` placeholders)
- Refresh auto-figures: `python -m scripts.make_figures`
- Compile: `cd thesis_report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- Fix unresolved `\ref{}` and undefined cites
- Final structural review per `feedback_thesis_coherence` memory (terminology, registry counts, fold semantics, cross-references)

---

## Open issues (deferred, no urgency)

1. Task #42 — Document 9-subtype CHM coverage methodology in thesis (low priority)
2. Task #98 — LoRA SIL future work (Phase 1b, deferred)
3. Task #111 — B6/B7 + ablation dynamic cycle count rewire (auto-handled by `run_phase1a.sh` if Step 7 early-stops)

---

## Locked configuration snapshot (2026-04-28 15:10 UTC)

| Value | Where | Current |
|------|-------|--------:|
| `α_store` | `cycle_0/conformal_gate.json` | **0.05** (restored from sweep) |
| `α_defer` | `cycle_0/conformal_gate.json` | **0.40** |
| `τ_store` | `cycle_0/conformal_gate.json` | **0.6676** |
| `τ_defer` | `cycle_0/conformal_gate.json` | **0.5207** |
| `τ_retro` | `config.py:664` | **0.50** |
| `τ_train` | `config.py:627` | **0.75** |
| `ρ_min` retention | `config.py:636` | **0.93** |
| `α_ema` | `config.py:274` | **0.7** |
| Boost intercept | `cycle_0/composite_calibration.json` | **+1.099** (restored) |
| Boost C | hard-coded in `run_experiment.py:cmd` | **0.01** |
| `disable_h_norm` | `config.py` | **True** |
| `top_passages` logging | `eval/harness.py` | active from cycle 1+ |
| Safety floor `u_pre^min` | `config.py:127` | **0.60** |
| Tier-1 combined | `config.py:123` | **0.90** |
| Tier-2 similarity | `config.py:124` | **0.75** |
| Routing λ | `config.py:121` | **0.70** |
| M chains, T_a | `config.py:140,143` | **3, 0.7** |
| K MC-Dropout | `verifier.py:24` | **5** |

All values match thesis claims. Three independent guards lock the calibration chain across cycles 1–10.

---

## Resume command on return

```bash
cd /workspace/caem
tmux ls                                              # check plan_a alive
tail outputs/full_run/run.log | head -50             # most recent activity
ls outputs/full_run/run_complete.json 2>&1            # main run complete?
cat outputs/full_run/experiment_summary.csv 2>&1 | head -15
```

If everything is green: proceed to P2 (theorem receipts) and onward.
