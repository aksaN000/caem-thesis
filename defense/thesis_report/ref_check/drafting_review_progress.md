# Final drafting review: progress

Working copy: `defense/thesis_report/` (started from `branch/ruc-classifier` @ 9619ef9).
Change log: `ref_check/fixes_applied.md`, section "Final drafting pass (2026-10)".
Baseline build (latexmk, 2026-10-16): 324 pages, 0 errors, 0 undefined refs/citations, 0 multiply-defined labels,
0 biber warnings, 75 overfull hboxes (28 > 10 pt).

## Status

| Part | Status | Notes |
|---|---|---|
| Step 1: p_ground^mean vs p_ground^atomic | done (report) | Identical on 14,951/14,951 run samples; see open question Q1 |
| Front matter | in progress | |
| Chapter 1 | pending | |
| Chapter 2 | pending | |
| Chapter 3 | pending | |
| Chapter 4 | pending | |
| Chapter 5 | pending | pre-found issues: `review_ch5_part1.md` |
| Chapter 6 | pending | pre-found issues: `review_ch5_part2_ch6.md` |
| Appendices A–G | pending | |
| Bibliography | pending | |
| Whole-document pass | pending | |

## Open questions for the author

**Q1 (step 1, affects abstract, ch1, ch4, ch5, ch6).** `p_ground^atomic` equalled `p_ground^mean` on every run sample
(8,993 eval-fold C0–C5, 5,958 calibration-fold C1–C5). Cause (code + `outputs/full_run/experiment.log`): the atomic
signal falls back to the passage-mean signal for answers under `atomic_min_tokens = 12` (99.0% of samples skipped the
stage) or when decomposition returns no facts (the remaining 1.0%, 82 samples). The composite therefore ran on eight
distinct signals with passage-mean grounding entered twice. Options: (a) keep "nine signals" as the architecture and
add a realized-run note + limitation + future-work item (recommended); (b) rewrite as eight signals throughout.
