---
name: Post-C5 metric inventory + gap audit (queued for after C5 closes)
description: Systematic audit of all empirical findings the thesis claims vs what's actually calculated by scripts. Identified gaps so far (π_store trajectory, etc.). Run after C5 closes, BEFORE writing-integration step P-7.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Plan locked 2026-05-15 18:00 BDT (user-locked):** When C5 closes (~01:30 BDT May 16), do a systematic metric inventory + gap audit BEFORE doing any Ch5/Ch6 integration writing. Three phases:

1. **Phase A — Enumerate all empirical findings** the thesis claims (or should claim) across all chapters. Build the complete inventory list.
2. **Phase B — Check existing script coverage.** For each finding in the inventory, identify whether an existing script computes it + where it lands on disk + whether it's surfaced in any auto-generated table referenced by the thesis.
3. **Phase C — Build only the missing pieces.** Write aggregation scripts + auto-tables for the gaps.

## Starter inventory (built 2026-05-15 — pre-audit)

To pre-load the audit, here are the categories of findings the thesis touches or should touch. The post-C5 audit fills in coverage status for each.

### Storage / precision metrics

| Finding | Source | Currently aggregated? |
|---|---|---|
| π_store cal-fold per cycle (pooled + per-bench) | `cycle_N/calibration/{bench}_cycle{N}.json` | **GAP — not in any table** |
| π_store stream-chunk per cycle | `eval/{bench}_cycle{N}_streamchunk.json` | **GAP — not in any table** |
| π_retro (retroverify-survivor precision) per cycle | `retroverify_cycle{N}.json` | Partial — counts only, no precision |
| Memory poisoning rate per cycle | `cycle_N/weight_validation.json` (only C0 exists) | C0 only quoted in prose |
| Stored count per cycle (n_store) | derived from cal-fold + stream-chunk | partial |
| Decision distribution (STORE/DEFER/ABSTAIN/DISCARD per cycle) | sample-level in eval JSONs | tab_calibration stub exists |

### Trajectory metrics (per-cycle EM / CHM)

| Finding | Source | Currently aggregated? |
|---|---|---|
| Pooled EM per cycle | `eval/{bench}_cycle{N}.json` | tab_headline + tab_cycle_progression stubs exist |
| Per-bench EM per cycle | same | tab_cycle_progression |
| Pooled CHM per cycle | derived via `composite_hallucination_metric()` | tab_headline stub |
| Per-subtype CHM per cycle (8 subtypes) | derived | tab_halluc_subtypes stub |
| gen_em vs final_em gap (Story A/B disambiguation) | sample-level | **GAP — important per Phase 1c memory** |

### Tier metrics

| Finding | Source | Currently aggregated? |
|---|---|---|
| Per-cycle Tier 1/2/3 share (pooled + per-bench) | sample-level `tier` field | tab_grounding stub |
| Per-cycle per-tier EM (T2 EM, T3 EM separately) | sample-level | **GAP — I computed this manually for ch5 §tier-precision** |
| Memory-tier hit rate trajectory | derives from tier share | partial |
| Per-tier latency | sample-level `latency_ms` | **GAP — Task #154 Tier-1 amortisation diagnostic queued** |
| Per-tier dollar cost projection | derives from latency × hit rate | **GAP** |

### Retention metrics

| Finding | Source | Currently aggregated? |
|---|---|---|
| Per-cycle multi-modal probe ratios (MMLU, TQA-test, CSQA-test) | `retroverify_cycle{N}.json` `fine_tune.mmlu_retention_ratio` (MMLU only) + log | **GAP — multi-modal not in any JSON file** |
| Retention-guard firing pattern (commits vs aborts per cycle) | log | **GAP — for Cor stochastic-equilibrium** |
| Worst-probe ratio per cycle | log | **GAP** |

### Verifier signal metrics

| Finding | Source | Currently aggregated? |
|---|---|---|
| Per-signal Cohen's d trajectory (across cycles) | cycle-zero weight_validation has it | cohen_d.tex exists (cycle-0 only?) |
| Per-bench Cohen's d trajectory | cycle_N/composite_calibration.json | **GAP** |
| Signal correlation matrix | `outputs/signal_correlation/` | mentioned in runbook |

### Calibration metrics

| Finding | Source | Currently aggregated? |
|---|---|---|
| Per-cycle ECE (Expected Calibration Error) | log shows ECE_after at refit time | **GAP** |
| Per-cycle T_b (temperature scalar per bench) | `cycle_N/calibration/calibrated_config_cycle{N}.json` | **GAP** |
| Per-cycle composite weights | `cycle_N/composite_calibration.json` | composite_weights.tex (cycle-0 only?) |
| Calibration-vs-evaluation gap trajectory | derives | tab_calibration_trajectory stub exists |

### Deferred-buffer metrics

| Finding | Source | Currently aggregated? |
|---|---|---|
| Buffer size per cycle | `deferred_buffer_cycle_{N}.pkl` | **GAP** |
| Promoted per cycle (79, 184, ...) | log | **GAP** |
| TTL-dropped per cycle | log | **GAP** |
| Cumulative deferred-recovery rate | derives | **GAP — needed for §sec:adj-cal-eval-gap reframe** |

### Memory store metrics

| Finding | Source | Currently aggregated? |
|---|---|---|
| Memory store size per cycle | `memory_store_cycle_{N}.meta` count | **GAP** |
| Consolidation events per cycle | log | **GAP** |
| Retroverify updated / pruned per cycle | `retroverify_cycle{N}.json` | partial — values exist, not tabled |

### Statistical metrics (post-baselines)

| Finding | Source | Currently aggregated? |
|---|---|---|
| Per-baseline EM + CHM | post-rescore baseline JSONs | scripts/baseline_sig_tests.py |
| McNemar p-value per (bench, baseline) | scripts/baseline_sig_tests.py | tab_sig_test stub |
| Bootstrap BCa CI per (bench, baseline) | scripts/baseline_sig_tests.py | tab_sig_test |
| Holm-adjusted p | scripts/baseline_sig_tests.py | tab_sig_test |
| Licensing-rule verdict (per H + per bench) | post-stats | **GAP — verdict population** |

### Hypothesis verdicts (final integration)

| H | Deciding statistic | Currently populated? |
|---|---|---|
| H1 | π_store at cycle-0 + trajectory | partial (C0 in prose, trajectory missing) |
| H2 | CHM vs baselines, matched protocol | gap — needs baselines |
| H3 | Tier-1/2 share growth | partial |
| H4 | Retention floor maintained | partial — needs trajectory aggregation |
| H5 | Ablation contributions | gap — needs ablation panel |

## Audit completed 2026-05-15 — 3 new aggregators built + tested

**Updates to this memory entry 2026-05-15 16:30 BDT (after script audit):**

### Verified script-to-theorem coverage (corrects my earlier "11 of 12 missing" alarm)

- `run_purity_validation.py` covers thm:purity + thm:bayes-purity + thm:monotone + thm:bayes-convergence (4 theorems)
- `theorem_receipts.py` covers thm:convergence + thm:asymptotic-elim + cor:convergence-rate + cor:corpus-floor + cor:self-correction (5 theorems)
- Combined: **9 of 12 theorems** have dedicated computation scripts that already exist

### 3 new aggregators built + tested (commit pending)

1. **`scripts/aggregate_pi_store_trajectory.py`** — empirical receipt for H1 + cycle-on-cycle complement to thm:purity. Reads cycle_N/calibration/*.json (decision==STORE) and eval/*_streamchunk.json (stored==True). Outputs `outputs/full_run/pi_store_trajectory.csv` + `thesis_report/figures/auto/tab_pi_store_trajectory.tex`. Tested on cycles 0-5; reproduces 0.640 → 0.702 → 0.748 → 0.757 cal-fold trajectory I quoted manually.

2. **`scripts/aggregate_tier_chm.py`** — empirical receipt for cor:tier1-floor. Slices eval/*_cycleN.json samples by `tier` field (T1/T2/T3) and runs `composite_hallucination_metric()` per slice. Outputs CSV + tab_tier_chm_trajectory.tex. Tested on cycles 0-5; confirms T2 EM (0.620 at C3) > base T3 EM (0.476 at C0) by +14.4 pp (distillation receipt).

3. **`scripts/aggregate_stochastic_equilibrium.py`** — empirical receipt for cor:stochastic-equilibrium (the new corollary added today). Log-scrapes `experiment.log` for retention-guard firing pattern, computes per-cycle commit indicator + worst-probe + π_commit. Outputs CSV + tab_stochastic_equilibrium.tex. Tested; produces 5-row table: C1-C3 COMMIT, C4-C5 ABORT, π_commit = 0.6000, with worst-probe shifting from TQA-test (C4) to CSQA-test (C5).

### Known infrastructure gaps surfaced by the audit (not yet fixed)

- `scripts/phase4_artifacts.py` writes to `outputs/phase4/`, NOT `thesis_report/figures/auto/`. Path fix needed before make_tables.py pipeline produces correctly-located outputs.
- `scripts/aggregate_calibration_trajectory.py` default --tex_out points to legacy "pre thesis 1 report/" path. Override at CLI call (now documented in runbook).
- `scripts/generate_session5_artifacts.py` same legacy-path issue.
- Three auto-tables referenced by Ch5 have NO generator script: `tab_tier_baseline_cost.tex`, `tab_production_envelope_examples.tex`, `tab_prompt_design_ablation.tex`. Each needs ~40-60 line generator OR must be removed from Ch5 references.
- Pre-existing `figures/auto/cohen_d.tex` and `figures/auto/composite_weights.tex` reflect v1 panel (NQ, ARC, ASQA in benchmarks); regenerate against v2.1 panel during P-4.
- `tab_sig_test.tex` will be generated by `baseline_sig_tests.py` in P-5; no current generator runs are needed.

### Remaining minor extensions

- **Spearman correlation for thm:monotone-purification** — ~10 lines to add inside run_purity_validation.py after the Theory 2 monotonicity check completes for all cycles. Defer to post-C5 when purity validation runs.
- Per-cycle Cohen's d trajectory aggregation (across cycles, per signal) — partial overlap with `phase4_artifacts.py:emit_cohen_d_table` which is cycle-0 only.

## How to apply (post-C5 audit execution)

When C5 closes:

1. Run a complete pass on the inventory above; expand it with any findings I've missed
2. For each finding, grep the codebase for whether an existing script computes it (`scripts/aggregate_*.py`, `scripts/make_tables.py`, `scripts/baseline_sig_tests.py`, `caem/training/diagnostics.py`)
3. Tabulate each finding's coverage status in a `outputs/metric_audit_2026-05-XX.md` artefact
4. Identify the gaps and prioritise by thesis-evidence impact:
   - Critical: any finding that's quoted in Ch5/Ch6 prose but has no underlying table
   - High: any finding required to populate the H1-H5 verdict table
   - Medium: any finding the thesis claims but doesn't yet quote
   - Low: nice-to-have diagnostics
5. Build the missing aggregation scripts + auto-tables. Most are <100 lines each.
6. Re-run `make_tables.py` (or extended equivalent) so all generated `figures/auto/*.tex` files reflect the post-trajectory state.
7. THEN start Ch5/Ch6 integration writing (Step P-7).

The reason this ordering matters: if writing happens before the audit, prose paragraphs get committed citing tables that don't exist, and we get another drift-fix cycle. Better to lock the table content first, then write prose against it.

## Notes for execution

- Don't trust the existing `scripts/make_tables.py` to produce everything — verify each generated `.tex` independently. The placeholder stubs currently in `figures/auto/` suggest the script may not cover every artefact referenced in chapters.
- The Tier-1 amortisation diagnostic (Task #154) is a special case: it needs serial Pipeline at bs=1 on a custom small eval set, separate from the main trajectory data. Schedule it within the post-C5 work, but it's a dedicated GPU run, not just aggregation.
- The single-probe-precision threat (TQA-test ±4% noise) and the worst-probe trajectory (0.937 → 0.84 → 0.92) come from the log, not from any JSON file — will need a log-scraping aggregator.
- Some metrics are observable only at successful cycles (cal-fold π_store) and some at all cycles (memory store size, tier shares). The audit needs to make this distinction explicit per finding.
