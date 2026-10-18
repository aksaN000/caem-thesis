---
name: CAEM post-step_7_main — recall + recovery empirical update
description: After step_7_main completes, populate Ch5 §sec:adj-cal-eval-gap with empirical recall-trajectory numbers from the 10-cycle data; conceptual framing already added 2026-04-29.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
After step_7_main + Phase 4 complete (~May 11), update Ch5 §sec:adj-cal-eval-gap with empirical recall-and-recovery numbers from the cycle-trajectory data. The conceptual framing (precision-recall tradeoff, deferred-buffer recovery path, semantic re-encounter recovery path) was added on 2026-04-29 and committed; only the empirical numbers need filling in.

**Why:** the cycle-1 single-cycle recall reading (22.8% capture, 50.8% irrecoverable on FEVER stream chunk) is a misleading underestimate without the trajectory data; the deferred-buffer promotion rate per cycle and the survival-cycles distribution from cor:self-correction are the correct receipt for "trajectory recall" framing.

**How to apply:** once `outputs/full_run/theorem_receipts/receipt_self_correction.json` exists (manual run after Phase 4 per NEXT_SESSION_PLAN P2), extend Ch5 §sec:adj-cal-eval-gap with concrete numbers:

1. Per-cycle FEVER store rate trajectory (cycle 1 had 13.87% / 416 stored, 84.1% deployed precision)
2. Per-cycle deferred-buffer promotion rate (count of entries promoted from deferral → storage at each cycle boundary)
3. Cumulative trajectory recall on FEVER (cycle-1 capture 22.8% → cycle-N cumulative capture)
4. Survival-cycles distribution from `receipt_self_correction.json` (median survival horizon, % of stored entries that survive all 10 cycles)
5. Per-benchmark imbalance numbers (FEVER ~13% vs TriviaQA ~0.5% vs NQ ~0.2% predicted; verify with actual)

These numbers go into Ch5 §sec:adj-cal-eval-gap as a follow-up paragraph or table, right after the conceptual framing already in place. They also feed the cor:self-correction empirical receipt narrative in Ch5 §sec:check-* (subsection on equilibrium structure and self-correction).

**Source artefacts:** experiment_summary.csv, deferred_buffer_cycle_N.pkl per cycle, receipt_self_correction.json (manual receipt run), receipt_envelope_fit.json (for trajectory geometry).

---

**ALSO post-step_7_main: decision diagnostics deep-dive subsection (added to plan 2026-04-29).**

After all 10 cycles' eval/{benchmark}_cycleN.json files exist, write `scripts/aggregate_decision_diagnostics.py` (~150 lines, ~4-6h) producing five new auto-tables:

1. **tab_decision_matrix_trajectory.tex** — per-cycle 8-cell decision × EM matrix per benchmark (cycle 1 readings: FEVER STORE=416 ✓350 ✗66 prec=84.1%; TriviaQA STORE=10 ✓5 ✗5 prec=50%; per-cycle trajectory)
2. **tab_grounding_success_distribution.tex** — per-cycle p_ground_mean ≥ {0.7, 0.5, 0.3, 0.1} on EM=1 samples per benchmark (anchors corpus-expansion paragraph; cycle-1 readings: FEVER 14.3% / 26.6% / 43.4% / 53.5%; TriviaQA 1.3% / 8.5% / 38.7% / 86.1%)
3. **tab_signal_fingerprint.tex** — per-cycle stored-correct signal medians per benchmark (proves cross-benchmark fingerprint consistency: FEVER vs TriviaQA STORE-correct p_ground_mean = 0.767 vs 0.741 — same range)
4. **fig_ustored_calibration.tex** — per-cycle u_stored bucket × EM rate histogram (calibration monotonicity proof; cycle-1 union: [0.6,0.72)=78.6% EM, [0.72,0.80)=74.2%, [0.80,0.90)=85.3%, [0.90,1.0]=92.5% — sharp transition at τ_store)
5. **tab_contamination_examples.tex** (appendix) — top-10 wrong-stored examples per benchmark per cycle (cycle-1 reading: ~50% of FEVER contamination is "not enough info" gold-label mislabel issue — structural, not gate failure)

Add ONE new Ch5 subsection §sec:adj-decision-diagnostics between §sec:adj-cal-eval-gap and §sec:adj-judge-ablation; 2 paragraphs of intro prose + the 4 auto-tables (5th in appendix). Reference the new tables from the existing cause/architectural/corpus paragraphs already in place. Total post-step_7_main effort: ~1 day.
