# Colab regeneration diff: classification

Source: Drive file `regen_diff.txt` (fileId 1Zj6V8rKx4qhYCqHnoV23kVVkaGUOYGvB). It is the `git diff --word-diff=plain -U0` of `thesis_report/figures/auto/*.tex` after `thesis_report/ref_check/colab_regen_check.py` ran on the Drive artefacts. 30 table files differ. No repo file was edited.

Notation: committed = [-old-] (thesis), regenerated = {+new+} (Drive).
Classes: **F** means formatting, layout or caption wording only. **N** means a numeric value differs, with these causes:
- (a) known-stale B6 inputs
- (b) faiss-missing empty output
- (c) rounding or precision only
- (d) genuine difference

Each (d) item also gets a sub-tag:
- **d-data**: the Drive inputs really hold different numbers.
- **d-pipe**: a pipeline or ordering artefact. The data agrees, but the generator got no input or a different input.
- **d-text**: a stale number hard-coded in the generator script's prose. The committed table was corrected by hand.

A common pattern: many committed tables have rows, columns or panels added by hand that the generator scripts do not emit. Examples are the B1 reference columns and rows, the "panel b" per-benchmark snapshots, the delta columns, `\resizebox` and `\caption[short]{...}`. The regenerated files drop these. That only removes content, so it is classed F. Wherever a cell exists in both versions, the values were checked.

---

## Caption / layout only (F)

| Table | What changed | Verdict |
|---|---|---|
| cohen_d.tex | `\caption[short]` removed; retirement list says "v2.1 panel … ARC-Challenge, and ASQA". No cell changes | F |
| composite_weights.tex | `\caption[short]` removed | F |
| cross_benchmark_summary.tex | `\caption[short]` removed; retirement list gains ASQA (the regenerated text says "Three benchmarks" but names four) | F |
| tab_audit_summary.tex | `\caption[short]` removed | F |
| tab_claim_evidence_map.tex | `\caption[short]` removed | F |
| tab_self_correction_check.tex | `\resizebox`, `\caption[short]` | F |
| tab_stochastic_equilibrium.tex | `\resizebox`, `\caption[short]`. All rows identical, including the retention ratios | F |
| tab_tier_chm_trajectory.tex | `\caption[short]` | F |
| tab_continual_retention.tex | Hand-added "B1 ref" column removed (0.453/0.293/0.703/0.650/0.433), `\resizebox`, caption reworded. Every C0/C5/Δ/min/max/range cell is identical | F |
| tab_halluc_subtypes.tex | B1 ref column removed, `\resizebox`, caption reworded ("$n=1500$" became "one thousand five hundred", the same value). C0–C5 cells are identical | F |
| tab_per_bench_trajectory.tex | B1 ref column, shaded train-pooled EM/CHM rows and the two Δ columns removed (all hand-added), `\resizebox`, caption reworded. Per-benchmark C0–C5 cells are identical | F |
| tab_decision_distribution.tex | Panel (b), the per-benchmark C3 snapshot, removed. Header and caption changed. Pooled C1/C2/C3/C5 rows are identical | F |
| tab_decision_matrix_trajectory.tex | Different layout: the regenerated version has all cycles C0–C5 × benchmark with only STORE correct/wrong/precision; the committed version has C2 and C5 with Store/Defer/Discard columns. All overlapping cells (C2 and C5 store ✓/×/π for all 5 benchmarks) match exactly. Committed defer/discard columns are dropped | F |
| tab_per_tier_em.tex | Regenerated is a **C3-only** snapshot (older script layout); committed has C0/C2/C5. No overlapping cell. Regenerated C3 pooled EMs (50.3/46.3/74.3/31.0/64.0) match committed tab_per_bench_trajectory C3. The regenerated note ">50 % of CSQA to T3" contradicts its own 0/80/20 row; this is the stale note C3 in numbers_mismatch_origins.md and does not affect the committed table | F |
| tab_pi_store_trajectory.tex | Different column set: regenerated has cal-fold per-benchmark (n, π); committed has eval-fold pooled plus eval-fold per-benchmark. Cal-fold pooled n/π match at every cycle (`1\,001` vs `1001` is formatting only). Committed C4 eval-fold row and footnote dropped. No comparable cell differs | F |
| tab_calibration_trajectory.tex | Produced by a different (older) generator, `aggregate_calibration_trajectory.py`; the committed table comes from `aggregate_section_5_7.py`. The layout is entirely different. The only overlapping quantity, pooled T per cycle (20.086 / 1.089 / 0.836 / 0.740 / 0.869), matches exactly. ECE, per-benchmark T, φ_safe and boost weights are absent from the regenerated table | F |
| tab_tier_latency.tex | Units changed (ms/s/h became ms and s columns); B1 142 ms row and panel (c) per-benchmark snapshot dropped (hand-added). Values agree: 14442 ms = 14.4 s; wall 21662/20452/20711/21219/21052/21978 s = 6.02/5.68/5.75/5.89/5.85/6.11 h | F |
| tab_tier_trajectory.tex | Panel (b), the C5 per-benchmark snapshot, removed (hand-added; see C1 in numbers_mismatch_origins.md). Pooled C0–C5 rows identical. The regenerated caption's "≈12 pp" T2–T3 gap at C0 equals 0.602 − 0.482 | F |

## Tables with numeric differences (N)

### tab_baseline_pooled.tex: N = 6 (3 a, 3 d)
| Row | Column | Committed | Regenerated | Class |
|---|---|---|---|---|
| B6 vanilla FT | EM | 0.490 | 0.325 | (a) |
| B6 | ΔEM vs C2 | +0.054 (+11.0 %) | +0.219 (+67.5 %) | (a) |
| B6 | ΔEM vs C5 | +0.027 (+5.6 %) | +0.193 (+59.3 %) | (a) |
| B8 semantic entropy | EM | 0.505 | 0.507 | (d-data), see D1 |
| B8 | ΔEM vs C2 | +0.039 (+7.8 %) | +0.037 (+7.4 %) | (d-data), D1 |
| B8 | ΔEM vs C5 | +0.013 (+2.5 %) | +0.011 (+2.1 %) | (d-data), D1 |

### tab_sig_test.tex (CAEM C5): N = 32 (16 a, 16 d)
- vanilla_ft (B6) rows, all **(a)**. Each list is EM / ΔEM / EM CI:
  - fever: 0.543→0.560; −0.067→−0.083; [−0.123,−0.010]→[−0.140,−0.027]
  - triviaqa: 0.227→0.190; +0.227→+0.263; [+0.163,+0.283]→[+0.197,+0.323]
  - commonsense_qa: 0.780→**0.000**; −0.063→+0.717; sig "-"→"*"; [−0.120,−0.010]→[+0.663,+0.767] (the pre-fix format-collapse value)
  - truthfulqa: 0.273→0.263; +0.027→+0.037; [−0.040,+0.093]→[−0.033,+0.107]
  - strategyqa: 0.627→0.613; +0.013→+0.027; [−0.050,+0.083]→[−0.040,+0.097]
- semantic_entropy (B8) rows, all **(d-data), D1**. Each list is EM / ΔEM / EM CI:
  - fever: 0.477→0.453; +0.000→+0.023; [−0.080,+0.080]→[−0.060,+0.110]
  - triviaqa: 0.303→0.293; +0.150→+0.160; [+0.090,+0.203]→[+0.100,+0.210]
  - commonsense_qa: 0.760→0.703; −0.043→+0.013; [−0.093,+0.003]→[−0.043,+0.067]
  - truthfulqa: 0.307→0.433; −0.007→−0.133(B); EM sig "-"→"(B)"; [−0.067,+0.057]→[−0.207,−0.060]
  - strategyqa: 0.677→0.650; −0.037→−0.010; [−0.107,+0.043]→[−0.087,+0.070]
- All CHM cells are unchanged.

### tab_sig_test_c2.tex (CAEM C2): N = 32 (16 a, 16 d)
- vanilla_ft rows, **(a)**:
  - fever: 0.543→0.560; −0.010→−0.027; CI [−0.067,+0.053]→[−0.087,+0.037]
  - triviaqa: 0.227→0.190; +0.267→+0.303; CI [+0.203,+0.320]→[+0.243,+0.360]
  - csqa: 0.780→0.000; −0.063→+0.717; sig "-"→"*"; CI [−0.117,−0.007]→[+0.667,+0.767]
  - tqa: 0.273→0.263; +0.050→+0.060; CI [−0.017,+0.117]→[−0.003,+0.130]
  - strategyqa: 0.627→0.613; +0.027→+0.040; CI [−0.037,+0.090]→[−0.023,+0.110]
- semantic_entropy rows, **(d-data), D1**:
  - fever: 0.477→0.453; +0.057→+0.080; CI [−0.027,+0.143]→[−0.003,+0.163]
  - triviaqa: 0.303→0.293; +0.190→+0.200; CI [+0.137,+0.237]→[+0.140,+0.257]
  - csqa: 0.760→0.703; −0.043→+0.013; CI [−0.090,+0.007]→[−0.040,+0.067]
  - tqa: 0.307→0.433; +0.017→−0.110(B); sig "-"→"(B)"; CI [−0.040,+0.077]→[−0.177,−0.043]
  - strategyqa: 0.677→0.650; −0.023→+0.003; CI [−0.093,+0.043]→[−0.067,+0.077]

### tab_ces_per_system.tex: N = 5 (all d-data, D2)
| Row | Col | Committed | Regenerated |
|---|---|---|---|
| B-cot | ACC | 0.499 | 0.500 |
| B-rag | ACC | 0.457 | 0.458 |
| B-rag | CES | 0.822 | 0.823 (derived: ACC^¼) |
| B-cot_rag | ACC | 0.466 | 0.464 |
| B-cot_rag | CES | 0.826 | 0.825 (derived) |

B-cot and B-rag differ only in the last digit, but the cause is a different input file, not rounding (see D2). The regenerated values equal the committed tab_baseline_pooled B2/B3/B4 (0.500/0.458/0.464).

### tab_headline_decomposition_capability.tex: N = 12 (all d-data, D3; one root cause)
| Row | CSQA | Pooled-5 | Pooled-4 | Pooled-3 |
|---|---|---|---|---|
| B1 → C3 (full) | +0.7→+4.0 | +1.9→+2.5 | +5.4→+6.2 | +7.6→+8.7 |
| B1 → C0 (arch) | +2.7→+6.0 | +2.7→+3.4 | +4.7→+5.6 | +7.7→+8.8 |
| B1 zero_shot abs. EM | 73.7→70.3 | 51.3→50.7 | 53.3→52.5 | 49.4→48.3 |

### tab_pooled_trajectory.tex: N = 19 (all d-pipe, D4), plus F changes
- Retention ratios MMLU / TQA-test / CSQA-test for C1, C2, C3, C4 and C5 (15 cells): committed values (C1 1.041/1.013/1.006, C2 0.975/0.949/0.982, C3 0.992/0.987/0.976, C4 1.018/0.886/0.982, C5 0.999/1.013/0.958) are all "--" in the regenerated table.
- Outcome: C1, C2, C3 and C5 change from COMMIT to ABORT (4 cells). C4 is ABORT in both.
- F: the hand-added B1 row and the headline-contrast line are dropped, as are the `\rowcolor`/bold on C2 and the "CAEM" configuration column. EM/CHM per cycle are identical.

### tab_safety_override.tex: N, d-pipe/measure (D5)
- Committed: proxy $u_{pre} < 0.38$. On the eval fold that is 1 fire / 9000 (StrategyQA, min u_pre 0.376); stream chunks give 3 / 23,500 (CSQA); 4 / 32,500 combined. Committed also lists per-benchmark n and min $u_{pre}$.
- Regenerated: per-cycle 0 fires / 1500 for C0–C5, so 0 / 9000. The caption says the override "does not fire on any of the nine thousand".
- Differing values: eval fires 1→0 and rate 0.011 %→0.00 %. The stream-chunk panel and min-u_pre column are absent from the regenerated table.

### tab_theorem_receipts_summary.tex: N, d-text (D6)
- T1: "Pooled π_t ∈ [0.732, 0.746] … by 9 to 11 pp" became "[0.7226, 0.7536]; per-bench min 0.6637 … by 8 to 11 pp".
- T2: committed cites composite Cohen's d +0.635 and balanced accuracy ≈0.673 ("precondition audited"). Regenerated says "identity", with no number.
- C5: committed "memory grows from 431 to 12 175". Regenerated drops the numbers and the per-sub-claim verdict.
- Regenerated adds row C2 `cor:convergence-rate` (PROJECTED, ≥15 cycles). This row is absent from the committed table.
- Formatting: `\allowbreak` and column widths.

### tab_validation_gate.tex: N = 8 (all c, precision), plus F changes
| Gate | Committed | Regenerated |
|---|---|---|
| Composite d (ID pooled) | +0.63 | +0.635 |
| Composite d (Transfer pooled) | +0.17 | +0.173 |
| Store-vs-discard FEVER / TriviaQA / CSQA | +0.28 / +0.32 / +0.26 | +0.282 / +0.320 / +0.258 |
| Poisoning FEVER / TriviaQA / CSQA | 0.21 / 0.18 / 0.25 | 0.206 / 0.179 / 0.247 |

F: the committed per-benchmark composite d (+0.57/+1.52/+0.04 ID; +0.29/+0.30 transfer) is not emitted by the generator (hand-added). The layout and the caption were also rewritten.

### sweep_top.tex: N = 1 (d-text, D7), plus F changes
- Caption: "one-hundred-variant search space ($4\times5\times5$ …)" became "twenty-five-variant search space".
- F: tabular spec `p{}` became `lll`, rationale cells wrapped with extra `\\ & &` lines, and `\caption[short]`. The parameter values are unchanged.

### tab_memory_store_trajectory.tex: N, all (b)
Every committed value (431; 2 818/5 151/7 477/9 961/12 175; buffer 1 123…4 284; Δ +2 387…+2 214) is "--" or 0 in the regenerated table. Ignored.

### tab_survival_distribution.tex: N, all (b)
The regenerated table has all zeros and a different column set (promoted 131/220/299, TTL-dropped 924/1 169, totals 650 / 2 093 / 7 027 are lost). Ignored.

---

## GENUINE NUMERIC DIFFERENCES (d)

**D1. B8 semantic-entropy EM. Affects tab_baseline_pooled (3 cells), tab_sig_test (16) and tab_sig_test_c2 (16). Class: d-data.**
- Committed per-benchmark EM: FEVER 0.477, TriviaQA 0.303, CSQA 0.760, TQA 0.307, StrategyQA 0.677 (pooled 0.505).
- Regenerated: 0.453 / 0.293 / 0.703 / 0.433 / 0.650 (pooled 0.507).
- These regenerated values are exactly the B1 zero-shot per-benchmark EMs (tab_continual_retention / tab_per_bench_trajectory B1 ref: 0.453/0.293/0.703/0.433/0.650; tab_baseline_pooled B1 0.507).
- So the Drive `phase1e_baselines/semantic_entropy/<bench>_cycle0.json` files carry B1-identical answers. This looks like a pre-rerun B8 copy (or its EM field is the greedy answer). The committed tables use the post-rerun "abstain outputs" that the tab_baseline_pooled caption describes.
- CHM cells for B8 are unchanged, so only the EM/answer payload differs.
- This is most likely another Drive-staleness case like B6, but it is **not** covered by the known-B6 note. Check that the Drive B8 eval JSONs carry the post-rerun outputs.
- Side note: the *committed* tab_ces_per_system already shows B-semantic_entropy ACC 0.507 (= B1), which disagrees with committed tab_baseline_pooled B8 0.505. So the committed tables are already internally inconsistent on B8.

**D2. tab_ces_per_system baseline ACC/CES (5 cells). Class: d-data.**
- B-cot 0.499→0.500, B-rag 0.457→0.458 (CES 0.822→0.823), B-cot_rag 0.466→0.464 (CES 0.826→0.825).
- `scripts/aggregate_ces_per_system.py` reads `outputs/baselines/<name>/`, while tab_baseline_pooled reads `outputs/baselines_phase1e/`. The Colab script copies the same Drive `phase1e_baselines` folder into both, so the regenerated CES rows now agree with the committed tab_baseline_pooled B2/B3/B4 (0.500/0.458/0.464).
- The committed CES rows come from an older local `outputs/baselines` set. B-cot_rag 0.466 matches no version of tab_baseline_pooled (pre-a240921 had 0.461; now 0.464).
- The regenerated values are the internally consistent ones; the committed table is stale.

**D3. tab_headline_decomposition_capability: B1 CSQA (12 cells, one root cause). Class: d-data.**
- B1 zero-shot CSQA capability-EM: 73.7 committed vs 70.3 regenerated. This propagates to every B1 Pooled-5/4/3 cell and to the CSQA and pooled cells of both B1→C3 and B1→C0 rows (values in the N section above).
- The regenerated 70.3 equals strict EM (committed tab_headline_decomposition.tex B1 CSQA = 70.3). `collect_row(..., field="capability_em")` falls back to `em` when the field is missing (scripts/aggregate_three_headline_decomposition.py:101). So the Drive phase1e zero_shot CSQA JSON has no `capability_em` field: `scripts/add_capability_em.py` was not applied to the Drive copy.
- The committed 73.7 has been in the table since 04fe2e7.
- Neither tab_headline_decomposition*.tex is `\input` anywhere in the thesis, so no thesis text depends on it.

**D4. tab_pooled_trajectory retention ratios and outcomes (19 cells). Class: d-pipe, no data disagreement.**
- 15 retention-ratio cells are "--" and C1/C2/C3/C5 flip from COMMIT to ABORT.
- `aggregate_section_5_2.py` reads `outputs/full_run/stochastic_equilibrium_trajectory.csv` (line 52). That CSV is written only by `aggregate_stochastic_equilibrium.py`, which colab_regen_check.py runs *after* section_5_2 (step 17 vs step 1).
- With the CSV missing, `load_retention()` returns {} and every outcome defaults to ABORT (line 187).
- The regenerated tab_stochastic_equilibrium has the same ratios and commits as the committed table (no row diff), so the data agrees. Re-running section_5_2 after stochastic_equilibrium should clear this.

**D5. tab_safety_override: 1 eval-fold fire committed vs 0 regenerated. Class: d-pipe / different measurement.**
- `aggregate_section_5_4.py` counts `s.get('escalated')`, which is a FLARE/baseline flag and not the router's safety override. Its caption has the "does not fire on any of the nine thousand" text hard-coded.
- The committed table was hand-built from the proxy $u_{pre} < 0.38$: 1/9000 eval (StrategyQA, 0.376) plus 3/23,500 stream.
- This is a generator/measurement mismatch, not a data change. It also identifies where the ch5:181 "0 of 9000" prose came from: numbers_mismatch_origins.md A3 says "No 0-fire source exists", but this script is one.

**D6. tab_theorem_receipts_summary prose numbers. Class: d-text.**
- T1 band [0.732, 0.746] / "9 to 11 pp" became [0.7226, 0.7536] / "8 to 11 pp" / "per-bench min 0.6637".
- T2 "+0.635, balanced accuracy ≈0.673" is dropped.
- C5 "431 to 12 175" is dropped.
- An extra C2 `cor:convergence-rate` row appears.
- All of these are hard-coded strings in `scripts/aggregate_section_5_6.py:67ff` (introduced in 7c7800d, a pre-Phase-1e state). The committed table was corrected by hand.
- The committed T1 band agrees with tab_pi_store_trajectory cal-fold pooled π (0.732–0.746) in both committed and regenerated versions. The regenerated text is stale, not the data.

**D7. sweep_top caption variant count. Class: d-text.**
- "one-hundred-variant ($4\times5\times5$)" committed vs "twenty-five-variant" regenerated.
- This is a hard-coded caption in `scripts/aggregate_section_5_7.py`; the old 25-variant sweep is per numbers_mismatch_origins.md O29/A22.
- The listed grids (4 τ × 5 C × 5 α) give 100, so the committed caption is consistent with the table and the regenerated one is stale.

**Bottom line:** no CAEM-trajectory number (EM, CHM, π_store, tiers, latency, retention, calibration T, decision counts) disagrees between the thesis and the Drive artefacts. The only data-level disagreements involve baseline files: B6 (known), B8 (D1), legacy `outputs/baselines` for CES (D2), and a missing `capability_em` field (D3). Everything else is pipeline ordering (D4), a generator measuring a different quantity (D5), stale hard-coded generator prose (D6, D7), precision (validation gate), or faiss (memory/survival).
