# Classification of the local regen diff (authoritative run, user's machine)

Source: regen_diff.txt (Drive 1WxDMjKpjNAOJR3M5Z6v5IWSMrEwuWttO; markdown-escaping removed before parsing), regen_stat.txt,
regenerated .tex files in Drive folder 17mA_PNsGD-Ii0mZ3uz3rAzlRsIzjqFQu, committed tables in
/home/user/caem-thesis/thesis_report/figures/auto/. Generator order taken from
thesis_report/ref_check/local_regen_check.py (aggregate_section_5_2..5_7 first, then the per-table aggregators, then
`make_tables` last) and confirmed by the Drive modification times: 03:24:26 = aggregate_section_5_x,
03:24:44-03:24:50 = per-table aggregators, 03:24:56 = make_tables.

Legend: F = formatting/layout/caption wording only. N = a numeric value differs, sub-classified
r = rounding/units only, h = committed table has a hand-added row/column/panel the generator does not emit (not a conflict),
g = GENUINE: the thesis shows a number the current outputs/generators do not produce.
Two extra labels used where r/h/g does not fit:
s = the "regenerated" value is a stale string literal hard-coded in the generator, not read from outputs.
c = generator collision: a later script overwrote the file, so this run did not check the committed table.

## Summary table

| # | Table | Result |
|---|---|---|
| 1 | cohen_d | F only |
| 2 | composite_weights | F only |
| 3 | cross_benchmark_summary | F only |
| 4 | sweep_top | F only (caption literal, see note) |
| 5 | tab_audit_summary | F only |
| 6 | tab_calibration | F only (both versions empty) |
| 7 | tab_calibration_trajectory | N: r0 h14 rows g0 (the 5 cells that overlap match) |
| 8 | tab_claim_evidence_map | F only |
| 9 | tab_continual | N: r4 h0 g0 |
| 10 | tab_continual_retention | N: r0 h1 col (5 cells) g0 |
| 11 | tab_cycle_progression | F only (placeholder vs empty table) |
| 12 | tab_decision_distribution | N: r0 h1 panel (3 rows) g0 |
| 13 | tab_decision_matrix_trajectory | N: r0 h4 cols x10 rows g0 (30 overlapping cells match) |
| 14 | tab_grounding | F only |
| 15 | tab_halluc_subtypes | c: not checked by this run (overwritten by make_tables); CHM and B1 rows cross-check OK |
| 16 | tab_headline | F only |
| 17 | tab_memory_store_trajectory | F only |
| 18 | tab_per_bench_trajectory | N: r0 h3 (B1 column, 2 delta columns, 2 pooled rows) g0 |
| 19 | tab_per_tier_em | N: r0 h(entire C0/C2/C5 body; generator now emits only C3) g0 |
| 20 | tab_pi_store_trajectory | N: r0 h(5 eval-fold columns, 6 rows) g0 |
| 21 | tab_pooled_trajectory | N: r0 h2 rows g0 |
| 22 | tab_purity | F only |
| 23 | tab_safety_override | N: g (eval fires 1 vs 0) + h (panel b, min u_pre column) |
| 24 | tab_self_correction_check | F only |
| 25 | tab_stochastic_equilibrium | F only |
| 26 | tab_survival_distribution | N: r0 h3 cols + totals row g0 (all overlapping values reconcile) |
| 27 | tab_theorem_receipts_summary | N: s3 (T1 cell, known-stale 5_6 literal) h2 g0 |
| 28 | tab_tier_chm_trajectory | F only |
| 29 | tab_tier_latency | N: r6 h6 g0 |
| 30 | tab_tier_trajectory | N: r0 h1 panel (5 rows) g0. FLAG: panel (b) contradicts the regenerated pooled per-tier EM (below) |
| 31 | tab_validation_gate | N: r8 h0 g0 (per-bench d values match the regenerated cohen_d.tex) |

---

## Per-table detail

### 1. cohen_d.tex: F
The only difference is the caption: `\caption[short]{...}` became `\caption{...}`, and the retirement list changed from "HotpotQA, Natural Questions, and ARC-Challenge" (thesis) to "HotpotQA, Natural Questions, ARC-Challenge, and ASQA" (generator), with "five-benchmark" becoming "five-benchmark-v2.1". No cell differs. The regenerated cohen_d.tex therefore still contains the thesis values (u_stored row: +0.57 / +1.52 / +0.04 / +0.63 / +0.29 / +0.30 / +0.17).

### 2. composite_weights.tex: F
Caption short-form only.

### 3. cross_benchmark_summary.tex: F
The difference is caption-only (short form, and ASQA added to the retirement list). Note: the generator's caption says "Three benchmarks ... (HotpotQA, Natural Questions, ARC-Challenge, and ASQA)", which names four. That wording is a literal in aggregate_section_5_7.py:252.

### 4. sweep_top.tex: F
- The tabular spec changed and the third column is now broken into line-wrapped rows. Every parameter value is the same (0.60, 0.20, 0.010, 0.6, 0.93, grids).
- In the caption, the thesis has "one-hundred-variant search space ($4 \times 5 \times 5$ ...)" and the generator has "twenty-five-variant search space". The generator's figure is a hard-coded caption literal (aggregate_section_5_7.py:521). The thesis wording agrees with the grids in the table itself (4 τ values × 5 C values × 5 α values = 100), so the thesis is right and this is not g.

### 5. tab_audit_summary.tex: F
Caption short-form only.

### 6. tab_calibration.tex: F
The thesis has the "no populated rows" placeholder. The generator emits a table for cycles 1–5 with `n scored = 0` and every cell `--`. Neither version contains a numeric value.

### 7. tab_calibration_trajectory.tex: N (h only; no conflict)
The thesis table has the header "Auto-generated by scripts/aggregate_section_5_7.py", but that script no longer writes this file. In this run it was written by aggregate_calibration_trajectory.py, which produces a different layout: Cycle, τ_store, τ_defer, τ_train, T, store/defer/discard/abstain share, Memory, α(t).
- Overlapping cells (pooled temperature T), thesis vs regenerated: C0 20.086 / 20.086, C1 1.089 / 1.089, C2 0.836 / 0.836, C3 0.740 / 0.740, C5 0.869 / 0.869. All match, and C4 is absent in both.
- h (rows in the thesis that no current generator emits; none are contradicted by the outputs):
  - the |Δ C0→C5| column, including T 19.217
  - the ECE row: 0.044 / 0.049 / 0.041 / 0.043
  - the boost intercept b0 row: +0.265 / −0.613 / −0.634 / −0.565 / −0.520, Δ 0.785
  - the per-benchmark T rows:
    - FEVER 14.092 / 9.899 / 6.986 / 4.972
    - TriviaQA 14.123 / 10.007 / 7.163 / 11.040
    - CSQA 1.342 / 1.209 / 1.083 / 1.028
  - the φ_safe^CSQA row: 0.380 / 0.380 / 0.380 / 0.491 / 0.569, Δ 0.189
  - the 9 boost-weight rows of panel (d)
- Values the generator emits that the thesis does not show: the C0 shares 15.1% / 17.3% / 17.4% / 50.2%.
- Side observation, not part of the diff: in the thesis, the p_ground^mean and p_ground^atomic rows of panel (d) are identical cell for cell (+0.184, +0.262, +0.238, +0.223, +0.230, 0.046). The same duplication exists between those two rows in cohen_d.tex, so this may be real (atomic grounding falling back to mean), but it is worth a glance.

### 8. tab_claim_evidence_map.tex: F
Caption short-form only.

### 9. tab_continual.tex: N (r only)
- The cycle labels change from 1.0000 to 1 (F), and MMLU % from 63.5 to 63.500, the same value (F).
- r (mmlu_retention_ratio): C2 0.9370 → 0.937, C3 0.9528 → 0.953, C4 0.9783 → 0.978, C5 0.9593 → 0.959. C1 changes from 1.0000 to 1.000, the same value.
- bwt/fwt change from 0 to 0.0000, the same value (F).

### 10. tab_continual_retention.tex: N (h)
- h: the "B1 ref" column is hand-added: FEVER 0.453, TriviaQA 0.293, CSQA 0.703, StrategyQA 0.650, TruthfulQA 0.433.
- The C0, C5, Δ, min, max, range and Outcome cells are identical in both versions.
- The \resizebox and the caption rewording are F.

### 11. tab_cycle_progression.tex: F
The thesis has a placeholder stub. The generator emits an empty table whose only row is `--`. There are no numbers in either version.

### 12. tab_decision_distribution.tex: N (h)
- Pooled rows C1, C2, C3 and C5 are identical; they do not appear in the diff.
- h: panel (b), the per-benchmark snapshot at C3, is hand-added. It has three rows: FEVER 55.1% / 0.76 / 38.1% / 0.47 / 4.3% / 0.52 / 2.6% / 0.38, TriviaQA 40.5% / 0.79 / ..., and CSQA 85.7% / 0.72 / ....
- Layout and caption changes are F.

### 13. tab_decision_matrix_trajectory.tex: N (h)
- The thesis shows C2 and C5 with Store ✓/×/π, Defer ✓/×, Discard ✓/×. The generator (aggregate_decision_diagnostics.py) emits STORE correct/wrong/precision for C0–C5.
- The overlapping cells are all identical (thesis = regenerated):
  - C2: FEVER 803/240/0.770, TriviaQA 605/209/0.743, CSQA 465/172/0.730, TruthfulQA 33/58/0.363, StrategyQA 57/20/0.740
  - C5: FEVER 764/210/0.784, TriviaQA 532/269/0.664, CSQA 428/177/0.707, TruthfulQA 23/52/0.307, StrategyQA 45/10/0.818
- h: the Defer ✓, Defer ×, Discard ✓ and Discard × columns (10 rows × 4 cells) are not emitted by the generator.

### 14. tab_grounding.tex: F
The cycle labels change from 1.0000 to 1 and unsupported_correct_rate from 0.0 to 0.000. The values are the same and every other cell is `--` with n = 0.

### 15. tab_halluc_subtypes.tex: c (not checked by this run)
- The thesis table was produced by `aggregate_section_5_2.py` (its header says so). In this run, 5_2 wrote the file at 03:24:26, and `make_tables` overwrote it at 03:24:56 with its own CSV-driven version: retroverify rows, n = 0, all `--`. So the regenerated file is not a re-render of the thesis table, and this diff cannot classify its 8 subtype rows × 7 columns.
- Partial cross-check against other regenerated tables:
  - The thesis "CHM (mean of 8)" row (C0..C5 = 0.135 / 0.112 / 0.107 / 0.115 / 0.115 / 0.109) matches tab_pooled_trajectory CHM, which was regenerated identically.
  - B1 0.120 matches the regenerated tab_baseline_pooled (B1 CHM 0.120).
- To check the subtype cells, rerun `python -m scripts.aggregate_section_5_2` alone and diff that file.

### 16. tab_headline.tex: F
Cycle labels change from 1.0000 to 1. The CES values (0.0549 / 0.0542 / 0.0544 / 0.0547 / 0.0545) are identical.

### 17. tab_memory_store_trajectory.tex: F
- Thin-space and math formatting change: `$2\,818$` becomes 2818, and so on. Every value is identical: 431/0, 2818/1123/+2387, 5151/2399/+2333, 7477/2692/+2326, 9961/4068/+2484, 12175/4284/+2214.
- The C4$^{\ast}$ marker and caption are F.

### 18. tab_per_bench_trajectory.tex: N (h)
- Every per-benchmark C0–C5 EM and CHM cell is identical.
- h, the B1 ref column: FEVER 0.453/0.159, TriviaQA 0.293/0.175, CSQA 0.703/0.096, TruthfulQA 0.433/0.051, StrategyQA 0.650/0.120.
- h, the Δ B1→C2 and Δ B1→C5 columns: all 10 rows. Spot checks recompute correctly from the cells, for example FEVER EM +0.080 (+17.7%).
- h, the two shaded "train Pooled" rows. These match the regenerated tab_train_vs_transfer.tex (aggregate_section_5_2) exactly:
  - EM 0.571 / 0.584 / 0.581 / 0.570 / 0.571 / 0.549
  - CHM 0.124 / 0.104 / 0.101 / 0.109 / 0.108 / 0.114
  - B1 pooled 0.483 / 0.143 equals the mean of the B1 cells.

### 19. tab_per_tier_em.tex: N (h, no overlap)
- The thesis is a hand-made snapshot ("from outputs/tab_per_tier_em.csv") of C0, C2 and C5 per benchmark. The current generator, aggregate_per_tier_em.py, emits only C3: FEVER 50.3, TriviaQA 46.3, CSQA 74.3, TruthfulQA 31.0, StrategyQA 64.0.
- No cell overlaps, so none conflicts. The whole committed body (15 rows) is h.
- The thesis pooled-EM column agrees with tab_per_bench_trajectory, for example C0 FEVER 48.7 vs 0.487 and C5 TruthfulQA 30.0 vs 0.300.

### 20. tab_pi_store_trajectory.tex: N (h)
- The cal-fold pooled n/π are identical: C0 575/0.746, C1 631/0.740, C2 1001/0.736, C3 899/0.746, C5 868/0.732. `1\,001` vs `1001` is F.
- h: the thesis eval-fold columns (pooled n, pooled π, per-benchmark π FEVER/TriviaQA/CSQA, including the eval-only C4* row) are not emitted. The generator now emits cal-fold per-benchmark n/π instead.
- The thesis C0 eval cells are corroborated by the regenerated decision_matrix C0 rows: FEVER 37/19 gives 0.661, CSQA 218/62 gives 0.779, and pooled (37+218)/336 gives 0.759.
- C1–C5 eval-fold cells cannot be verified from this run.

### 21. tab_pooled_trajectory.tex: N (h)
- The CAEM C0–C5 rows (EM, CHM, the 3 probe ratios, outcome) are identical.
- h, the B1 (zero-shot) row: EM 0.507, CHM 0.120. Both match the regenerated tab_baseline_pooled.
- h, the "Headline contrast" row: ΔEM +0.037 (+7.4%) matches tab_baseline_pooled. ΔCHM −0.013 (−10.8%) does not quite match: tab_baseline_pooled prints −0.013 (−11.0%). −10.8% is −0.013/0.120 computed from the rounded figures, while the generator uses unrounded values. This is a rounding effect inside a hand-added row, not g, but the thesis prints two different relative CHM figures for the same contrast.

### 22. tab_purity.tex: F
The cycle labels change from 1.0000 to 1. All counts are 0 and all rates `--` in both versions.

### 23. tab_safety_override.tex: N (g + h)
- The thesis table's header says it was measured by the proxy `u_pre < 0.38`, because the dumped flag is not persisted. The generator (aggregate_section_5_4.py lines 52–53, 174–176) counts `s.get('escalated')` per cycle on the eval fold. No script in scripts/ computes the u_pre < 0.38 proxy.
- g, eval fold:
  - The thesis has StrategyQA fires = 1 (0.056%, min u_pre 0.376) and an eval subtotal of 9,000 queries with 1 fire (0.011%).
  - The regenerated table has 0 fires out of 1500 (0.00%) at every cycle C0–C5, so 0 out of 9,000 in total.
  - The other benchmarks (0 fires each, n = 1,800 = 6 × 300) are consistent with 0.
- h, not reproducible by any generator:
  - panel (b), SIL stream chunks: FEVER 10,000/0, TriviaQA 10,000/0, CSQA 3,500/3 (0.086%, min 0.358), subtotal 23,500/3 (0.013%)
  - the Combined row: 32,500 / 4 / 0.012% / 0.358
  - the Min u_pre column (0.491, 0.499, 0.402, 0.437, 0.376 for the eval fold)
  - The generator's "Effect on EM: not measurable" column is new.

### 24. tab_self_correction_check.tex: F
The \resizebox was removed and the caption changed to the plain form. No cells are in the diff.

### 25. tab_stochastic_equilibrium.tex: F
The \resizebox was removed and the caption changed to the plain form. No cells are in the diff.

### 26. tab_survival_distribution.tex: N (h; everything reconciles)
- The generator (aggregate_survival_distribution.py) emits Buffer size, Newly pushed, Net Δ, Estimated exits and Cumulative recovery for C0–C5. The thesis shows Buffer at reconsider, Promoted, TTL-dropped, Kept and New pushed for C1–C5, plus totals.
- Row convention: the thesis's "buffer at reconsider" at C(n) is the regenerated buffer size at C(n−1).
- Reconciliation:
  - Buffer: thesis 1123, 2399, 2692, 4068 match regenerated 1123, 2399, 2692, 4068.
  - New pushed: thesis 1123, 1407, 1437, 1376, 1684 match regenerated "Newly pushed" exactly.
  - Promoted + TTL equals the regenerated "Estimated exits": C2 131+0 = 131, C3 220+924 = 1144, C5 299+1169 = 1468, and C4 is skipped (regenerated 0).
  - Totals: 650 + 2093 = 2743, the regenerated "Cumulative recovery" at C5. Sum of new pushed = 7027.
  - Kept: 992 = 1123 − 131, 1255 = 2399 − 220 − 924, 2600 = 4068 − 299 − 1169.
- h: the Promoted/TTL/Kept split and the totals row (650, 2093, 7027) are not emitted. They are consistent with the generator's outputs.
- Wording note: the generator's caption calls promotions + TTL drops (2743) "cumulative recovery". The thesis reports 650 promotions as the recovery receipt.

### 27. tab_theorem_receipts_summary.tex: N (s + h; known-stale 5_6)
- s, row T1:
  - Thesis: "Pooled cal-fold π_t ∈ [0.732, 0.746] clears the 0.64 floor by 9 to 11 pp".
  - Regenerated: "Pooled π_t ∈ [0.7226, 0.7536]; per-bench min 0.6637; ... by 8 to 11 pp".
  - The regenerated string is a literal in scripts/aggregate_section_5_6.py:69, not computed from outputs.
  - The thesis range [0.732, 0.746] is exactly the min/max of the regenerated tab_pi_store_trajectory cal-fold pooled column (0.746, 0.740, 0.736, 0.746, 0.732), so the thesis matches the current outputs. Not g.
- h:
  - The T1 sentence "Per-bench eval-fold FEVER dips below floor at C1–C4 (band [0.614, 0.634])" matches the thesis's own pi_store eval columns: 0.634, 0.630, 0.617, 0.614.
  - The C5 row's "memory grows from 431 to 12,175" matches the regenerated memory_store.
- F: the generator emits a C2 row (cor:convergence-rate, PROJECTED) that the thesis removed. The T3, C5 and caption wording, and the \allowbreak changes, are also F.
- Unchanged in both versions: the seven Tier-1 firings, the prune trajectory 13 → 9 → 7 → 4, π_commit = 4/5 = 0.80, and band width ≤ 0.031.

### 28. tab_tier_chm_trajectory.tex: F
Caption short-form only.

### 29. tab_tier_latency.tex: N (r + h)
- Per-query seconds are identical: 14.4, 13.6, 13.8, 14.1, 14.0, 14.7. The generator adds an ms column: 14442, 13635, 13808, 14146, 14035, 14652.
- r, wall time where the thesis converts seconds to hours:
  - C0 6.02 h = 21662 s (6.017)
  - C1 5.68 h = 20452 s (5.681)
  - C2 5.75 h = 20711 s (5.753)
  - C3 5.89 h = 21219 s (5.894)
  - C4 5.85 h = 21052 s (5.848)
  - C5 6.11 h = 21978 s (6.105)
- h: panel (a) B1 zero-shot 142 ms, and panel (c) C5 per-benchmark latencies: FEVER 15.5 s, TriviaQA 16.3 s, CSQA 8.6 s, TruthfulQA 17.8 s, StrategyQA 15.1 s.

### 30. tab_tier_trajectory.tex: N (h), with a consistency FLAG
- The pooled rows C0–C5 are identical.
- h: panel (b), the C5 per-benchmark snapshot, is hand-added:
  - FEVER 0.0% / -- / 43.3% / 0.554 / 56.7% / 0.418 / 0.477
  - TriviaQA 0.3% / 1.000 / 37.0% / 0.514 / 62.7% / 0.420 / 0.453
  - CSQA 0.0% / -- / 82.3% / 0.733 / 17.7% / 0.642 / 0.717
  - TruthfulQA 0.3% / 1.000 / 64.3% / 0.591 / 35.3% / 0.311 / 0.493
  - StrategyQA 0.0% / -- / 24.0% / 0.736 / 76.0% / 0.610 / 0.640
- FLAG (not from the diff; consistency check against regenerated values): panel (b)'s per-tier EMs contradict the C5 pooled per-tier EM that the generator emits in the same table, T2 0.537 and T3 0.497.
  - Weighting panel (b) by tier counts (T2 n = 130 / 111 / 247 / 193 / 72; T3 n = 170 / 188 / 53 / 106 / 228) gives T2 ≈ 0.634 and T3 ≈ 0.478.
  - The thesis's tab_per_tier_em C5 values (T2 50.0 / 42.3 / 74.5 / 32.1 / 63.9; T3 45.9 / 46.8 / 58.5 / 25.5 / 64.0) reproduce T2 ≈ 0.536 and T3 ≈ 0.497.
  - Tier shares match in both tables, so panel (b) and tab_per_tier_em show different per-tier EMs for the same C5 cells.
  - Panel (b)'s TruthfulQA pooled EM 0.493 is the unjudged EM; tab_per_bench_trajectory and tab_per_tier_em use the Haiku-judged 0.300.
  - Panel (b) appears to come from a different tier attribution or EM definition, and the generator's pooled row does not reproduce it.

### 31. tab_validation_gate.tex: N (r only)
- r:
  - ID composite d: pooled +0.63 → +0.635
  - Transfer composite d: pooled +0.17 → +0.173
  - Store-vs-discard gap: +0.28 / +0.32 / +0.26 → +0.282 / +0.320 / +0.258
  - Poisoning rate: 0.21 / 0.18 / 0.25 → 0.206 / 0.179 / 0.247
- The per-benchmark d cells in the thesis (ID +0.57 / +1.52 / +0.04; Transfer +0.29 / +0.30) are not in the regenerated gate table. They are identical to the u_stored row of the regenerated cohen_d.tex, so they are reproduced by the current outputs (not h, not g).
- Rounding note: the regenerated cohen_d prints ID pool +0.63 while the regenerated gate table prints +0.635. The two generator outputs are consistent; the thesis +0.63 is fine.
- Layout and caption changes are F.

---

## GENUINE NUMERIC DIFFERENCES (g)

1. **tab_safety_override.tex, panel (a) evaluation fold, row "StrategyQA", columns Fires / Rate.**
   - Thesis: `$1$` fire, `$0.056\%$` (n = 1,800, min u_pre `$\mathbf{0.376}$`).
   - Regenerated: 0 fires at every cycle; the table shows `C0..C5 & 0 & 1500 & 0.00\%`, with no per-benchmark split.
2. **tab_safety_override.tex, row "Eval subtotal", columns Fires / Rate.**
   - Thesis: `$\mathbf{1}$` fire out of `$9\,000$`, `$0.011\%$`.
   - Regenerated: 0 out of 9,000 (0 / 1500 / 0.00% at each of C0–C5).

Cause: the two versions measure different things. The thesis uses the hand-computed proxy u_pre < 0.38. The generator (aggregate_section_5_4.py) counts the `escalated` flag, and no script in scripts/ computes the proxy. As a result, the dependent hand-only cells cannot be reproduced by any generator in the repo either:
- panel (b): CSQA 3 fires out of 3,500 (0.086%), subtotal 3 out of 23,500 (0.013%)
- the Combined row: 4 fires out of 32,500 (0.012%)
- the Min u_pre column

The generator's caption also states the opposite conclusion ("does not fire on any of the nine thousand evaluation queries").

No other g was found.

### Not g, but worth checking
- **tab_halluc_subtypes**: not checked by this run, because make_tables overwrote the 5_2 output (generator collision). Rerun `scripts.aggregate_section_5_2` alone to verify the 56 subtype cells. Its CHM row and B1 CHM do cross-check.
- **tab_tier_trajectory panel (b)** (hand-added): the C5 per-tier EMs are inconsistent with the regenerated pooled per-tier EM and with tab_per_tier_em C5. Details are in §30.
- **tab_pooled_trajectory headline row**: ΔCHM "−10.8%" vs "−11.0%" in the regenerated tab_baseline_pooled (rounded vs unrounded base).
- **tab_theorem_receipts_summary T1**: the generator literal is stale ([0.7226, 0.7536], 8–11 pp). The thesis ([0.732, 0.746], 9–11 pp) matches the current outputs, so keep the thesis.
- **sweep_top / cohen_d / cross_benchmark_summary captions**: the generator literals ("twenty-five-variant"; "Three benchmarks" while listing four) are wrong. The thesis wording is correct.
