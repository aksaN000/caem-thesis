# Final drafting review: progress

Working copy: `defense/thesis_report/` (started from `branch/ruc-classifier` @ 9619ef9).
Change log: `ref_check/fixes_applied.md`, section "Final drafting pass (2026-10)".
Baseline build (latexmk, 2026-10-16): 324 pages, 0 errors, 0 undefined refs/citations, 0 multiply-defined labels,
0 biber warnings, 75 overfull hboxes (28 > 10 pt).

## Status

| Part | Status | Notes |
|---|---|---|
| Step 1: p_ground^mean vs p_ground^atomic | done (report) | Identical on 14,951/14,951 run samples; see open question Q1 |
| Front matter | done; PR #4 open; title month pending | branch `review/front`; 22 changes logged |
| Chapter 1 | done; merged (PR #5) | branch `review/ch1`; 45 changes logged in fixes_applied.md |
| Chapter 2 | done; merged (PR #6 into review/ch1, then PR #7 into main) | branch `review/ch2`; 16 152 → 11 740 words |
| Chapter 3 | done; author read-through closed 2026-10-16; merged (PR #8) | branch `review/ch3`; 6 437 → 4 861 words |
| Chapter 4 | Claude pass done (full rewrite + 6 figures); PR open; awaiting author's read | branch `review/ch4`; ~21 000 → ~9 700 words |
| Chapter 5 | pending | pre-found issues: `archive/audits/thesis_audit_2026-10-15/review_ch5_part1.md` |
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

- **Q8 → resolved:** template says nothing about paragraph spacing; added `parskip` (0.5 baselineskip gap), 281 pages.

- **Q4 → done (re-run 2026-10-16):** `scripts/rerun_cycle0_sweep.py` → `outputs/cycle_0/sweep_v2_1/` (variants.json/.csv +
  README). Findings that change thesis numbers (author decision needed in the ch5 session, also abstract/ch3/ch6):
  (i) "289 entries at 0.796" was produced by the legacy weighted-sum bootstrap composite, not the locked configuration;
  the locked configuration scores 558 stored at 0.751 on the held-out validation fold (FEVER 93, TriviaQA 0,
  CommonsenseQA 465), pooled d 0.72. (ii) "306 of 478" and "70 of 105" are not reproduced by any variant on either fold;
  re-run calibration-fold values are 437/560 (τ 0.60) and 243/281 (τ 0.65). (iii) tab_validation_gate is computed from
  the bootstrap scores and needs recomputing under the locked composite (gate still passes: d 0.72 ≥ 0.20). (iv) the
  locked configuration passes criteria 1–2 but is not the criterion-3 maximum (α = 1.0 stores more correct items), so
  its choice should be justified by the shrinkage rationale. (v) every variant stores 0 TriviaQA items on the validation
  fold (calibration-to-validation shift), consistent with the main run's C1 calibration fold (1 TriviaQA store).

## Chapter 1 (2026-10-16): decisions for the author (RESOLVED same day, see resolutions below)

**D1 (ch1 L64; also ch2 L275/281/287/291 [frozen], ch3 L145, ch4 L555, ch5 L41, appendix E L15).** Every chapter says HotpotQA,
Natural Questions and ARC-Challenge were retired "on cycle-zero diagnostic evidence that they violated the
verifier-balanced-accuracy precondition", and that "all three are kept on disk as registered exclusion evidence". The only
record found (archive/branch_C_log_historical.md:4680–4717) gives a different reason: HotpotQA cold-start base EM 0.090,
"unstoreable" (the Bayes floor at the then-conformal target needed a TPR/FPR ratio near 192); Natural Questions halted
part-way (384 processed) and dropped with it; no record found for ARC-Challenge. The receipts the log names
(`outputs/cycle_0/eval_excluded_v21/`, `outputs/cycle_0/eval/{hotpotqa,natural_questions}_cycle0.json`) are not in local
`outputs/`. Options: (a) search Drive `9_archive` for the receipts and restate the real reason (base accuracy too low for the
gate to store at the target precision); (b) soften everywhere to "retired before the main run on cycle-zero diagnostics" and
drop "kept on disk" unless found. Touches frozen ch2.

**D2 (tab:hyp-registration; ch5 tab:hypothesis-verdicts).** H2 ("pool purity is non-decreasing") is mapped to thm:purity,
which is a floor, not a monotonicity result. The monotonicity statement in ch4 is the remark thm:bayes-convergence (and the
storage-rate heuristic thm:monotone). Hypothesis text left untouched (pre-specified); correct only the mapping column? Ties to
the H2 verdict problem already logged for ch5.

**D3 (ch1 L72, problem statement).** The requirement "user-facing answers carry a precision guarantee that holds under modest
distribution shift between calibration data and live data" conflicts with the design history (conformal rejected because the
shift broke its guarantee) and with thm:purity's exchangeability premise. Soften to e.g. "carry a calibrated reliability
estimate whose precision is checked on held-out data"? Changes the problem definition, so not applied.

**D4 (redundancy).** sec:three-shortcomings and sec:coupling-exclusions (L76–90) restate sec:stateless-gap (L25–34) almost
sentence for sentence, with the same citations. Proposal: shrink sec:three-shortcomings to three sentences that point back to
sec:stateless-gap, and keep only the exclusions paragraph of sec:coupling-exclusions. Cuts content, so not applied.

**D5 (fig:experimental-workflow body).** Caption now says "design view". The bullets "Tier 1 share rises", "Highest Tier 1
share" and "Memory: mature, near-equilibrium" still read as outcomes (realised Tier 1 ≤ 0.1%). Reword the bullets (e.g.
"Memory grows; Tier 1 serves repeated questions"), or keep them as the design picture?

**D6 (Obj 4 L116, L275).** "The storage cut-point is selected against the cycle-zero threshold sweep so that the realised
per-store empirical precision on the calibration fold is anchored to a known number": align with the Q4 sweep outcome once the
ch5 decision is made (locked configuration is not the criterion-3 maximum; justify by the shrinkage rationale).

## Carried from chapter 1 to later chapters

- **ch4 L37/49/73, appendix F L57:** "encoder-layer convergence ratio": the backbone is decoder-only; the ratio is computed
  from decoder-stack hidden states (`caem/confidence/pre_routing.py`). Ch1 now says "layer-convergence ratio".
- **ch4 L42/97/105, ch5 L507:** "upward-only" retroactive update rule: the code overwrites the stored score up or down
  (`retroverify_allow_downgrade = True`) and prunes below 0.50, not below the storage threshold. Ch4's thm:purity (ii) is
  already correct.
- **ch4 L509, thm:bayes-convergence (ch4 ~L694), ch5 L232:** "the $L_2$ adapter anchor of eq:sil-anchor": the L2 anchor is
  removed on the LoRA path (`caem/config.py:670`); check what eq:sil-anchor defines. Ch1 says "no explicit quadratic anchor".
- **ch2 L118 (frozen), ch4 L348, ch5 L317/L575, ch6 L32:** "fifteen to fifty percentage points" below the conformal target:
  only source is archive/branch_C_log_historical.md:4788; no artefact. Removed from ch1.
- **ch6:** the novelty threshold (0.95) has no sensitivity analysis; ch1's promise was removed. Consider a limitation +
  future-work pair.
- **ch3/ch6 (cost):** retroactive pass wall time grows with memory: about 38 min for 431 entries (C1) to about 16 h for
  10,214 entries (C5) (experiment.log).
- **ch5 (Q7):** once the Tier-1 probe is reported, ch1 L58 (realised tier shares) can point to it as the repeated-question
  evidence.
- **Style, all chapters:** the closing section is titled "Chapter signpost" (sentence case) in ch1–ch5 while other headings
  use title case; left unchanged because ch2 is frozen.

### Resolutions (author, 2026-10-16)

- **D1 → applied globally.** Only HotpotQA and Natural Questions are mentioned; ARC-Challenge removed from the prose
  everywhere (ch2 L212 keeps a cited study's ARC metric). The real reason, from the receipts on Drive: base EM 0.090 / 0.172;
  the gate stored 26 / 12 of 500 at purity 0.500 / 0.417; gate balanced accuracy was 0.630 / 0.521, i.e. above one half, so the
  old "balanced-accuracy precondition violated" reason was false. **Appendix session:** appendix B tables prompt_compliance,
  atomic_scope and tab_prompt_design_ablation still carry ARC-C rows (earlier-panel data with NQ and ARC but no
  CommonsenseQA); decide whether to drop those rows or the tables.
- **D2 → mapping fixed.** H2 maps to the Bayes-purification recurrence remark (thm:bayes-convergence) in ch1 and ch5; the
  pre-specified hypothesis text is unchanged. **ch5 session (verdict change APPROVED by the author 2026-10-16):** as registered, H2 is falsified
  (cal-fold purity 0.746 → 0.740 → 0.736 → 0.746 → 0.732 falls at C1, C2, C5); recommended verdict "unsupported as registered;
  the fluctuation (width 0.014) is within binomial noise and purity stays above the thm:purity floor". The ch5 caption's
  "bounded-band restatement of thm:purity" must go with it.
- **D3 → softened** to a per-cycle refitted calibrated probability whose precision is checked on held-out data; rejected
  conformal history removed from ch1 (kept in ch4).
- **D4 → trimmed:** shortcomings/coupling merged; per-query-pipeline duplicate removed; mitigation subsection removed;
  objective-section methodology paragraph moved; panel described once (sec:scope-research). 9 215 → 8 377 words.
- **D5 → figure bullets and caption reworded;** Tier-1 probe numbers added to ch1 L58. **ch5 session:** report the probe in
  ch5 with a table/label, then point ch1 L58 at it.
- **D6 → ch1 now number-free** ("chosen on the cycle-zero calibration fold before the main run"), so the Q4 outcome in ch5 cannot
  contradict it.
- **Signpost → keep** (all chapters use one; the BRAC template lists minimum sections, not a closed list).

## Carried from chapter 2

- **ch5 L446 (B6):** says vanilla fine-tuning trains "on the verified episode pool … every selected episode is treated as a clean
  label" and "without the bounded low-rank-adapter envelope". Code: full fine-tune on raw question–answer pairs from the three
  training benchmarks (`scripts/run_simple_ft.py`, `launch_baselines.sh`: 700 per benchmark, 5 cycles), no verifier filter.
- **ch5 (B5):** five-shot demonstrations come from the first benchmark of the launch list (`scripts/run_baseline.py:296`); confirm
  which benchmark the final B5 run used before naming it.
- **ch5/ch6:** STaR and Self-RAG are not run; remove any "optional reference" rows or promises.
- **Semantic entropy as a signal:** canonical reason now in ch2 sec:lit-self-consistency, ch4 L285 and appendix G (learned weight
  ≈ −0.0005, largest verifier cost). Never cite the "Pearson = 0" reading: it was measured after the signal had been disabled.
- **Unsourced claims removed from ch2** (tell the author if wanted back with a source): Google FACTS 6.4% legal, Vectara 3%/27.2%,
  retrieval-utility AUC "ceiling" 0.80–0.86 / 0.65–0.75 and SeaKR.
## Carried from the ch1/ch2 read-through (author, 2026-10-16)

- **Citation pass, ch3–ch6 + appendices (NOT yet done):** apply the citation-style rule in SESSION_START "Standing author
  decisions". Bracket-as-noun counts found on 2026-10-16: ch3 1 (L163), ch4 12 (e.g. L64, L73, L278, L285, L343, L507 ×3,
  L600 ×2), ch5 7 (e.g. L366, L426, L600), appendices 0; also name the study on bare claims that carry a number.
- **Transfer-panel framing (ch5, ch6, abstract):** keep both transfer benchmarks; say plainly that the headline is the
  training panel; split the TruthfulQA loss into the part present at C0 before any training (−5.3 points, retrieval and
  architecture) and the part during self-improvement (−8.0 points, C0 → C5; "consistent with" drift away from what
  TruthfulQA rewards, not shown). StrategyQA: EM flat, CHM falls. Source: tab_per_bench_trajectory, tab_headline_decomposition.
- **Forgetting controls (if an examiner-facing passage needs it, ch4/ch6):** LoRA rank 32 on a frozen base, retention probe
  with rollback (fired once, C4, TriviaQA-test 0.395 → 0.350, ratio 0.886), pool rebalancing (max share 0.40), training filter 0.70; no L2,
  no EWC. LoRA + EWC was never ablated: say so if the choice is defended.

## Carried from chapter 3 (2026-10-16)

- **Early-stop gate (APPROVED 2026-10-16: disclose; ch3 done, apply in ch3 table row + ch4, ch5 L104/L609, ch6, appendix C L94):** the C5 stop
  fired on two mis-wired signals (`outputs/full_run/experiment.log` L70711; `caem/eval/equilibrium.py::three_signal_gate`
  fed from `scripts/run_experiment.py` ~L2046–2100): storage_saturation read the eval harness `storage_rate` (always 0;
  real stream admissions ≈50–53%); mmlu_floor compared raw MMLU accuracy (≈0.61) with the 0.93 ratio floor (real ratios
  0.937–1.00). The gate was certain to fire at C5, the first cycle it ran. Ch3 currently says only "stopped after cycle
  five, when the automatic early-stop check fired" (true, neutral).
- **Q4 numbers** remain for ch5 (ch3 is now number-free on the sweep).
- **Appendix A:** the Drive subfolder names (`v2_1_phase1e_main/`, `pre_main_snapshot/`) predate the numbered Drive layout
  (1_thesis … 9_archive); update in the appendix session.
- **Lead-in punctuation (author 2026-10-16):** bold openings end with a colon, not a full stop. Fixed in ch1 and ch3;
  still to fix: ch4 L306 ("Fit stage."), L322 ("Predict stage."); ch5 L613–L618 (six items). Table abbreviations such as
  "MMLU acc." (appendix C) and "Stored conf." (ch2 table) are not lead-ins; leave them.
- **Early-stop code fixed (2026-10-16)** in `scripts/run_experiment.py`; replay evidence `outputs/analysis/early_stop_gate_replay.json`
  (script `scripts/replay_early_stop_gate.py`): corrected wiring fires no signal at C5. ch4/ch5/ch6/appendix C should say
  the fault was corrected after the run and cite the replay, as ch3 sec:env-cycles does.
- **Earlier runs:** three earlier trajectories (author confirmed 2026-10-16: "mark it as 3").

## Carried from chapter 4 (2026-10-16)

- **ch5 L239 and L509:** use CHM_1 and the old ε_arch inequality; align to ch4 (H_1(t) Tier-1 error rate; equality
  H_t = (1−τ1)H_gen + τ1 H_mem). The "≈ 0.36" ceiling comes from the unreproducible π_store ≈ 0.64 (Q4): recompute or drop.
- **FEVER eval-fold π_store dip** (0.614–0.634 at C1–C4, 0.649 at C5; tab:pi-store-trajectory) was discussed in ch4's old
  purity receipt and is now only referenced there; make sure ch5 discusses it.
- **Phantom "context" citations** in Tier 2 answers after SIL (3% at C0 → 93–99% at C1–C5; memory
  caem_phantom_context_finding): ask the author how to report it in ch5/ch6.
- **ch6 L32** still has "fifteen to fifty percentage points".
- **Early-stop disclosure** still to apply in ch5 (L104, L116, L609), ch6, appendix C (L94).
- **Isotonic small-sample caveat (ch6 limitation, optional):** per-benchmark isotonic curves are fitted on 500 points, inside
  the 200–1,000 range where Niculescu-Mizil and Caruana find isotonic can overfit; mitigated by shrinkage (α 0.6), clipping,
  the 50-sample floor and C = 0.01. A Platt-versus-isotonic comparison per benchmark is a possible future-work item.
- **Storage quality filter:** active in the main run, rejected 7 answers (12,218 stored); regex-based, so it can reject a
  correct answer whose reasoning says "cannot be determined" and it does not catch the phantom "The context provides"
  phrasing. Mention in ch5/ch6 only if the phantom-context finding is reported.
- **Theory kept (recommendation given to author 2026-10-16):** all 11 statements stay, labelled by status; removing them
  would break the H1–H4 mappings and the CO5 "theoretical proofs" outcome.
- **Abstract:** still "storage precision 0.796 over 289" (legacy composite, Q4).

## Chapter 5 in progress (2026-10-16, branch `review/ch5`)

- New evidence scripts (committed with the chapter): `scripts/build_ch5_evidence_tables.py` → `figures/auto/tab_gate_quality.tex`,
  `tab_sweep_variants.tex`, `tab_headline_steps.tex`, `outputs/analysis/ch5_evidence_tables.json`;
  `scripts/recompute_validation_gate.py` → `figures/auto/tab_validation_gate_locked.tex`, `outputs/analysis/validation_gate_locked.json`.
- Gate quality (cal fold, STORE decision; matches pi_store_trajectory): balanced accuracy C1 0.615, C2 0.663, C3 0.664,
  C5 0.658 (C0 0.753 in-sample); Cohen's d 0.593 → 0.842; per-benchmark CSQA 0.52–0.55, TriviaQA C1 0.500.
- H2 trace must use one consistent C0 value: the old C0 0.746 is the "weight_validation" reading (validation fold, five
  benchmarks, legacy composite). Locked C0 on the cal fold is 0.780 (in-sample).
- Validation gate under the locked composite (validation fold): pooled training d 0.717 (pass); per-benchmark d FEVER
  0.555, TriviaQA 1.396, CSQA 0.142 (< 0.20); stored 93 / 0 / 465; store-minus-discard gap computable only on FEVER
  (0.268; TriviaQA stored nothing, CSQA discarded nothing); wrong share among stored FEVER 0.28, CSQA 0.243; transfer d
  TruthfulQA 0.383, StrategyQA −0.002. AUTHOR DECISION PENDING on the verdict wording (old table said every check passed).
- CHM confident-confabulation threshold in code is u ≥ 0.50 (eval/metrics.py), as the chapter states (not 0.60).
- **B8 CHM is from the OLD, broken semantic-entropy run (found 2026-10-16):** `outputs/baselines_phase1e/chm_comparison.json`
  semantic_entropy CHM (FEVER 0.1571, TriviaQA 0.1750, CSQA 0.0967, TruthfulQA 0.0500, StrategyQA 0.1192) equals the
  per-sample mean of `outputs/baselines_phase1e_pre_se_rerun_backup/semantic_entropy/*_with_chm.json` (rescored 2026-05-29,
  `rescore.log` L1251–1351). The corrected B8 run (2026-06-06, `baselines_phase1e/semantic_entropy/*_cycle0.json`) has
  `u_stored = None` on every sample: it was never rescored through the verifier. B8 EM uses the corrected run. Rescoring needs
  a GPU (verifier resampling + MiniCheck + 64 GB passage index). AUTHOR DECISION PENDING: rescore on GPU, or report B8 CHM as
  unavailable.
- **CHM significance is approximate in the code** (`archive/scripts_superseded/baseline_sig_tests_v2.py::_approx_chm_ci`: CAEM fixed, baseline
  subtype rates binomially resampled, no Holm, floor 0.01); ch5 and appendix D claimed paired Wilcoxon + Holm. Per-sample
  CHM (`eval/metrics.py::per_sample_chm`) reproduces the committed CHM for CAEM (C2, C5) and B1–B7, so a true paired test is
  possible without GPU for every baseline except B8 (no per-sample verifier fields in the corrected run).
- **Repo clean-up phase (author request 2026-10-16):** after the chapter pass, make the repository plug-and-play (one
  environment file, one command per stage, generators that reproduce every committed table without hand edits, superseded
  scripts archived with a README).
- **ch6 must contain (added 2026-10-17):** off-topic detection as future work (question-match signal cannot detect off-topic on short answers; a post-hoc Haiku judge agreed with 200 hand labels at kappa 0.35, outputs/analysis/off_topic_judge_validation.json); CHM is the mean of SEVEN types (off-topic and factual contradiction recorded but excluded); all ch6 CHM numbers must come from outputs/analysis/chm_judged.json (B1 0.156, C0 0.153, C2 0.126, C5 0.133; FLARE 0.226; B6 0.140); signal renamed question-answer match $q_{\text{match}}$.
- **ch6 must contain (promised by ch5, 2026-10-16):** a frozen-composite ablation cycle (monotonicity); an arm-isolating
  classifier run (corrections without the classifier); the phantom-context preprocessing fix (one sentence, author request);
  multi-seed and second-backbone replication; flexible-match re-scoring (format learning); retraining on a cleaned pool
  (poison test); a labelled per-cycle TPR/FPR/p measurement; ≥ 15 kept cycles; tier-latency at batch 1; corpus-ceiling proxy;
  evaluation with repeated questions (deployment). ch6 and the abstract must also adopt the ch5 headline (three-step
  decomposition), the verdicts (H1 supported, H2 unsupported, H3 supported, H4 partially supported, H5 unsupported), and drop
  "clean StrategyQA win", "parity with fine-tuned baselines", "deployed cost envelope / without scaling".
- **B8 decided (author's group, 2026-10-16, no budget):** B8 CHM reported as unavailable in ch5 (tables n/a, figure omits B8, limitation + future work). CHM counts without B8: C5 19 wins / 2 defeats of 30; C2 18 / 3. ch6 must add 'rescore the corrected B8 run' to future work; abstract/ch6 'every baseline' → 'every baseline whose CHM was measured'.
- **Paired CHM test (next session):** replace the approximate CHM test with a paired Wilcoxon + Holm from `eval/metrics.py::per_sample_chm` (validated for CAEM C2/C5 and B1–B7; B6 at c5); then fix appendix D (L63 Wilcoxon claim, L96 Holm 'each axis', CHM floor 0.01 not 0.02).
