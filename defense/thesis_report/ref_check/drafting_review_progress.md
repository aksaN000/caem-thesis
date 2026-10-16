# Final drafting review: progress

Working copy: `defense/thesis_report/` (started from `branch/ruc-classifier` @ 9619ef9).
Change log: `ref_check/fixes_applied.md`, section "Final drafting pass (2026-10)".
Baseline build (latexmk, 2026-10-16): 324 pages, 0 errors, 0 undefined refs/citations, 0 multiply-defined labels,
0 biber warnings, 75 overfull hboxes (28 > 10 pt).

## Status

| Part | Status | Notes |
|---|---|---|
| Step 1: p_ground^mean vs p_ground^atomic | done (report) | Identical on 14,951/14,951 run samples; see open question Q1 |
| Front matter | done; PR open; Q8 (parskip) + title month pending | branch `review/front`; 22 changes logged |
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

**Q2 (template formatting; each reflows the whole document).** Compared with `archive/brac_thesis_template_reference`:
citation style `numeric-comp, sorting=none` vs template `ieee, sorting=ynt`; text width 434.9 pt (left 1.25 in, right 1 in)
vs template 150 mm = 425.2 pt; paragraph indent on vs template `\parindent 0em`; 1.5 line spacing vs template default.
Front-matter order matches the template.

**Q3 (dates).** Title page "February 2026."; approval "of Fall, 2025 … in Spring 2026". Confirm the actual submission
month and semester.

**Q4 (sweep; abstract, ch3, ch5, sweep_top).** "one-hundred-variant pre-specified sweep": no sweep result exists for the
final panel (searched outputs/, Drive CAEM_thesis incl. archive, Vast backup). Only artefacts: v1-panel conformal-era sweep
(25 variants, `9_archive/v1_panel/full_run/cycle_0/sweep/`) and a Phase 1c alpha-sweep tarball. The generator itself notes
"the historical sweep was run on a different alpha schema and a different bench panel". The cycle-0 calibration fold
(1500 samples, all nine signals + em + decision) is on disk, so the 4×5×5 grid can be re-run offline. Options: (a) re-run
and persist the sweep; (b) soften to "a locked configuration chosen on a pre-specified grid", no sweep results claimed.

**Q5 (abstract length).** 709 words; the template sets no limit. Typical thesis abstracts are 250–400 words. Condense?

**Q6 (acknowledgement).** `core/acknowledgement.tex` is empty; the template treats this page as expected (dedication and
ethics are marked optional and may stay empty). Needs the author's own text.

**Q7 (abstract claim).** "a memory-direct tier that compounds cost savings as the pool grows": Tier 1 served 0.0–0.3% of
evaluation queries (tab_per_tier_em_capability) and no cost saving was measured. Soften to a design property, or remove?

**Logged for chapter 5 / appendix D:** in tab_sig_test (C5) and tab_sig_test_c2, the ΔCHM point estimate lies outside its
own 95% CI in 31 of 40 rows (e.g. cot·fever −0.042 vs [−0.034, −0.012]); the CHM column and the CHM bootstrap appear to be
computed on different quantities. Also: ch5 L301 "tenfold tighter" vs C = 0.01 against 1.0 (hundredfold); sweep_top caption
names "sklearn".

## Author decisions (2026-10-16)

- **Q1 → (a):** keep "nine signals" as the design; add a realized-run note (atomic = passage-mean on every run sample) and a
  limitation + matching future-work item. Apply in ch4 (signal definitions) and ch5 (tables showing identical rows); ch6
  limitation/future work.
- **Q2 → match the BRAC template:** applied in `review/front` (report class, 150 mm text, single spacing, no indent, plain
  pages, black links, IEEE year-sorted references, template TOC entries).
- **Q3 → final defense Fall 2026:** title page "Fall 2026." and approval "in Fall 2026" applied; author to confirm whether the
  title page should show an exact month ("Month Year" in the template).
- **Q4 → re-run the 100-variant sweep** offline on the cycle-0 calibration fold and persist it (affects abstract, ch3, ch5,
  sweep_top). Status: not started.
- **Q5 → condense the abstract last**, after ch5/ch6 are final.
- **Q6 → drafted** `core/acknowledgement.tex` (professional thanks only); authors to personalise.
- **Q7 → already measured:** `outputs/tier1_routing_probe/` (2026-05-30, cycle 5, 100 queries built from stored episodes):
  Tier 1 fired on 42/50 exact repeats (84%) and 19/50 paraphrases (38%); Tier-1 answers correct 41/42 and 19/19
  (`em_analysis.json`). Not reported anywhere in the thesis. Plan: report it in ch5's Tier-1 discussion and use it to
  ground the abstract's memory-tier claim (no new GPU run needed). Author's explanation: Tier 1 rarely fires on the main
  evaluation because evaluation questions are content-disjoint from training episodes; it fires on repeated or
  paraphrased questions.

**Q8 (new).** With the template's `\parindent 0em` and no `\parskip`, consecutive paragraphs run together. Add a small
paragraph gap (e.g. `\setlength{\parskip}{0.5\baselineskip}`), or keep the template exactly?
