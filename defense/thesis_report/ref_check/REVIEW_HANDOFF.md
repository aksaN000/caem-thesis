# Thesis review handoff (read this first)

You are continuing a section-by-section review of the CAEM thesis in this repository (`thesis_report/`), working
on the author's own machine with the full codebase, `outputs/`, `data/` and LaTeX. Earlier work happened in a
cloud session that could not see `outputs/`. This file gives you its state.

## Ground rules from the author
- **No AI attribution** in commits, PRs or files: no `Co-Authored-By`, no "Generated with" footers. Commit as the
  author (git config already set). Branch names must not contain "claude".
- **Never fabricate.** Every number in the thesis must trace to `outputs/` (data and logs), the code, or a
  committed table. If no source exists, soften or remove the claim and say so. Never invent χ², p-values or
  counts.
- **Sources of truth:**
  - data: local `outputs/` (Drive is a backup and is in sync);
  - code and text: GitHub `branch/ruc-classifier`, the default branch.
- **Work on a branch** (e.g. `review/ch1`), one PR per chapter into `branch/ruc-classifier`. The author merges.
- **Log every change** before → after, with its evidence, in `thesis_report/ref_check/fixes_applied.md`, under a new
  heading per chapter.
- **Compile before each PR** (`latexmk`): 0 errors, 0 undefined references or citations.
- **Never overwrite committed tables** with generator output. Regenerate into a temp copy
  (`thesis_report/ref_check/local_regen_check.py`) and compare numbers. The README's "do not regenerate" list
  names the hand-finished tables.

## What is already done (don't redo)
- **References:** all 95 verified against the papers' full text (`ref_check/reference_check_*.md`), plus about 100
  citation fixes.
- **Theory (ch4):** two proven results (`thm:purity`, `thm:bayes-purity`); five statements relabelled as
  heuristics.
- **Wording:** "pre-registered" → "pre-specified" throughout.
- **Numbers:** about 64 text-vs-table mismatches fixed (`ref_check/FINAL_FIX_LIST.md`, `fixes_applied.md`). All
  tables match `outputs/`.
- **Results that changed:**
  - RUC ablation: TruthfulQA 0.210 → 0.310, still below B1's 0.433, so gate 2 does not close.
  - Safety override: 63 of 9,000 firings (per-benchmark floors).
  - Run-log counts: 1,061 prunes and 650 promotions.
  - Appendix D: rebuilt from post-fix statistics.
- **Generators:** fixed so they reproduce the corrected tables; unit test in `tests/test_safety_override_count.py`.

## The review method (per section, start to end)
Go through the abstract, then chapters 1–6, then appendices A–G. For each section:
1. **Missing evidence.** A method, test, metric or result is described or announced but no
   table, figure or number shows it. Check that the `\Cref` target really contains it, and check the appendices.
2. **Orphans.** Tables or figures included but never interpreted; labels never cited; generated tables never
   included.
3. **References that don't deliver.** A pointer that doesn't show what the sentence claims.
4. **Numbers.** Check against the tables and against `outputs/` and the logs directly.
5. **Consistency.** The same claim must read the same in the abstract, ch1, ch5 and ch6; definitions (e.g. the
   licensing rule) must agree.
6. **Redundancy and misplacement.** Repeated results, methods inside results, limitations duplicated.
7. **Overclaiming.** Wording stronger than the evidence ("clean win", "every", "monotone", "parity").

Then fix, log, and summarise the section's changes for the author before moving on.

## Already found in chapters 5 and 6 (review not yet applied)
The full detail, with line numbers and quotes, is in `thesis_report/ref_check/review_ch5_part1.md` and
`review_ch5_part2_ch6.md`. Line numbers refer to commit 22f78c0. Headlines:

**A. Wording or overclaims**
- StrategyQA "clean win" vs B1 is not significant (CI [−0.017, +0.002]).
- "Parity with the strongest fine-tuned baselines" is wrong: CAEM leads B6, 0.517 vs 0.490.
- The licensing rule is defined two ways: ch5 L386 says the CI must lie above 0.02; appendix D L144–153 says the
  point estimate must be ≥ 0.02. The tables follow appendix D.
- H2 is marked supported, but its own falsification criterion triggers (purity decreases at 3 cycles).
  "Scope-limited" is not a registered verdict.
- "Headline interpretation" (ch5 ~L554), the ch6 closing remark, ch6 contribution 3 and the ch5 signpost are
  stale pre-results conditionals. They also claim untested cost and "rather than scaling" results.
- TruthfulQA abstention contradiction at L190: the text says abstention rose; the table shows 33.3% → 23.0%.
- The RUC gate is defined differently at L463 and L473, and the TruthfulQA CHM half (+2.9%) is never adjudicated.

**B. Missing evidence**
- McNemar, Holm and Wilcoxon: the significance tables have no χ²/p/Holm-p columns, yet the text says they are
  tabulated. `app:significance` is never cited. Holm sequences exist for only 2 of 10 families. Wilcoxon is never
  described.
- The monotonicity check (Spearman; storage rate vs Cohen's d) has no numbers anywhere.
- Validation gate: the 4th check has no row; CommonsenseQA d = +0.04 is unmentioned; the calibration vs
  evaluation fold is inconsistent between L289 and L341.
- Empty sections:
  - "Top variants" has no variants table.
  - "Tier-by-baseline cost" points to a placeholder.
  - "Optional reference points" names Self-RAG and STaR rows that don't exist.
  - "Per-tier latency and cost" shows neither.
- The judge ablation promises slope, intercept and MAE that were not persisted.
- Metrics:
  - F1 is defined but never reported.
  - The factual-contradiction row is missing.
  - The Haiku-judged TruthfulQA rule is never defined.
  - About ten reported quantities are undefined.
- The purity anchor π ≈ 0.64, the per-benchmark minimum 0.6637, and the n = 1500 standard error are in no table.

**C. Check in code first**
- `p_ground^mean` and `p_ground^atomic` are identical in every table. Either they are one signal ("nine signals" would really be eight) or there is an extraction bug.

**D. Structure (optional)**
- Headline numbers repeated in 5–7 places.
- Limitations split across three places.
- Nine limitations have no ch6 future-work item, and four ch6 items follow from no limitation.
- ch6's "five contributions in ch1" don't exist (ch1 has 7 objectives and 5 hypotheses).
- The statistical-methods text belongs in ch3.
- ch4 repeats the ch5 receipts.
- About 18 generated tables are never included.

## Next steps
1. Check item C in the code.
2. Start the review at the abstract and chapter 1, applying the method above.
3. When you reach chapters 5 and 6, apply the findings above, computing any missing tables from `outputs/`
   (monotonicity/Spearman, top sweep variants, per-tier latency) or cutting the claims.
