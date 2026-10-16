# Structural audit: chapter_5.tex lines 1-354 (Performance Evaluation; Final Design Adjustments)

Read-only. No files edited. Line numbers refer to thesis_report/chapters/chapter_5.tex.
Ranked by how likely an examiner is to notice.

## Mechanical checks (clean)
- Every \Cref target in L1-354 resolves to an existing \label.
- Every \input table in L1-354 is \Cref'd at least once, and so are the inline tables tab:benchmark-pools and tab:metrics. There are no orphan table labels.
- Section labels that are never referenced (harmless): sec:setup-benchmarks, sec:setup-reproducibility, sec:headline-pooled, sec:headline-per-bench, sec:headline-id-vs-transfer, sec:tier-safety, sec:check-corollaries, sec:adj-sweep, sec:adj-top, sec:performance.
- Auto-generated files that are relevant here but are not \input anywhere: tab_audit_summary.tex (see F12), tab_grounding_success_distribution.tex (see F10), fig_ustored_calibration.tex (it overlaps tab:u-stored-distribution).
- McNemar (the author's example) is at L361-414, outside this range. Evidence for it does exist: tab:sig-test, tab:sig-test-c2 and appendix D.

## Ranked findings

### F1. Monotonicity check has no numbers at all (MISSING EVIDENCE). Severity: very high
- L225 announces two tests: (a) "pairs the per-cycle change in storage rate with the per-cycle change in calibration-fold Cohen's d", and (b) "computes the one-sided Spearman rank correlation between ... p_c and ... P_c". It closes with "Both readings are presented as paired-comparison results across the trajectory."
- No table anywhere gives per-cycle cal-fold storage rate, per-cycle cal-fold composite Cohen's d, the dominance condition, or a Spearman rho/p. "Spearman" appears nowhere else in the thesis.
- L227 describes the transitions verbally ("sign-aligns at large amplitude", "sub-noise", "small reverse-sign") with no values. Chapter 4 L582 says these are "reported in \Cref{ch:evaluation}", but they are not.
- tab:theorem-receipts-summary row T3 gives tab:pi-store-trajectory as its anchor. That table has neither storage rate nor Cohen's d (DANGLING).
- Fix: add one small table, "Per-cycle cal-fold storage rate, composite d, dominance condition" (C0, C1, C2, C3, C5), with Delta signs and the Spearman rho and p. Re-anchor T3 to it. The same table also supplies the per-cycle d that the VER axis uses (F9).

### F2. Validation gate: a missing check, and the evidence contradicts the prose (MISSING / DANGLING). Severity: very high
- L341: "The fourth check requires that no single signal exhibit strong-direction inversion ... No strong-signal inversion is observed." tab:validation-gate has no row for this check.
- tab:cohen-d-per-bench appears to contradict that claim. composite_weights registers p_entail as "decreasing" (Pearson -0.156), but the ID-pool d is +0.20. q_a,rel has d = -0.37 on FEVER.
- tab:validation-gate shows CommonsenseQA composite d = +0.04, far below the 0.20 floor. The prose reports only the pooled +0.635 and never mentions this. An examiner will ask how a verifier with no discrimination on one training benchmark passes the gate.
- L289 (criterion 1) says "per-benchmark composite Cohen's d ... on the calibration fold must clear ... on the training-pooled slice". That mixes per-benchmark with pooled. L341 and the table caption say the evaluation fold, not the calibration fold.
- The tab:validation-gate caption promises "The per-benchmark Cohen's d heterogeneity and the Simpson's-style pooling artefact on the transfer panel are discussed in the surrounding prose". Neither is discussed (DANGLING). The same pattern shows in the table: u_internal ID-pool +0.55 is larger than every per-benchmark value (0.14, 0.34, 0.01), and the transfer pool +0.17 is below both members (0.29, 0.30).
- Fix: add a "direction inversions" row (count of signals with sign(d) different from the registered direction, plus the threshold). Add one paragraph on CSQA d = 0.04 and the pooled-vs-per-benchmark Simpson effect. Make L289 and L341 name the same fold and the same pooled-vs-per-benchmark scope.

### F3. "Top variants" shows no variants (MISSING EVIDENCE, TITLE MISMATCH). Severity: high
- L294: "The sweep comparison records each variant's calibration-fold area under the receiver operating characteristic, stored-volume, and per-store precision ... reported in \Cref{tab:sweep-top}."
- sweep_top.tex is a table of locked parameters and their rationale. It has no AUROC, no stored volume, no precision and no competing variant. Its caption mentions a 100-variant sweep (4x5x5), but none of the variants is shown.
- The comparison with V_{tau=0.65} is given only in prose, with numbers that do not match: L294 says 289 entries on the validation fold versus "roughly a factor of four fewer"; L301 says "306 against 70 on the cycle-zero calibration fold". No precision value for V0.65 is given anywhere.
- No table shows the outcome of selection criteria 1-2 (L289): which variants failed d >= 0.20 or n_min = 20.
- Fix: add a top-k variants table (tau, C, alpha, AUROC, d, n_stored, pi_store, which criteria pass), in the body or in appendix B with a pointer. Give one number pair for 0.60 vs 0.65 on one named fold. Alternatively, rename the subsection "Selected variant".

### F4. The judge-ablation appendix pointer promises content the appendix does not have (DANGLING). Severity: high
- L336: "The full Platt-fit diagnostic (slope, intercept, mean absolute error in both logit and probability space) is recorded in \Cref{app:judge-ablation}."
- appendix_b.tex (app:judge-ablation) says: "The two persisted headline statistics are the observed Pearson rho = 0.5843 ... and the pre-specified floor". It has no slope, intercept or MAE.
- L336 also says "a rank correlation of Pearson rho". Pearson is not a rank correlation.
- The "500-sample overlap fold" is not in tab:benchmark-pools.
- Fix: rewrite the sentence to say only rho and the floor were persisted, and choose either "Pearson correlation" or "rank correlation".

### F5. Metrics section defines things that are never reported, and the scoring rule for TruthfulQA does not match the tables (METRICS). Severity: high
Defined or announced but never reported:
- Token-level F1 (L66 "reported alongside exact match on ... TriviaQA"; tab:metrics row; L108 comment). No table in the thesis reports F1.
- Factual-contradiction rate. L80 says it "is therefore reported separately in the per-cycle subtype table". tab:halluc-subtypes has no such row. L127 even says the subtype is "near-zero".
- "Label accuracy" (tab:metrics row) is never reported separately and is the same as EM for FEVER, CSQA and StrategyQA. Cut the row.
- AUROC is named as a sweep diagnostic (L284, L294) and never reported (see F3).
- The TriviaQA "fallback to token-level F1 above the registered threshold" gives no threshold value.

Definition conflict:
- L66 defines TruthfulQA correctness as "best-alias match against the truthful and informative reference set".
- Every table and paragraph uses the "Haiku-judged correctness rule" (tab:pooled-trajectory, tab:per-bench-trajectory, L127, L190).
- The LLM judge is never defined in Metrics or in tab:metrics.

Reported but not defined in Metrics or tab:metrics:
- pi_store, pi_SIL, store/defer/discard/abstain share, abstention rate, tier share, per-tier EM, ECE (appears only inside the CAL axis), composite Cohen's d, balanced accuracy, memory-poisoning rate (first defined at L341, in a later section), prune rate, wrong-caught and correct-recovered rates, commit fraction, temperature scalar, latency.

Fix:
- Delete F1 everywhere, or add an F1 row to tab:per-bench-trajectory.
- Either add the factual-contradiction row or change L80 and L127.
- Define the Haiku judge rule in L66 and in tab:metrics.
- Extend tab:metrics with a "diagnostics" block for the quantities listed above.

### F6. The TruthfulQA abstention claim contradicts its own table (DANGLING). Severity: high
- L190: "The TruthfulQA decline tracks an increase in abstention rather than a degradation in retrieved knowledge".
- tab:abstention-per-bench shows TruthfulQA abstention falling from 33.3% (C0) to 23.0% (C5). L151 itself calls it "high and stable".
- Fix: remove the causal claim, or support it with a TruthfulQA wrong-answer breakdown.

### F7. Decision matrix and cross-benchmark summary: unstated fold, and impossible transfer-panel counts (DANGLING). Severity: high
- In tab:decision-matrix-trajectory, FEVER at C2 has about 1,900 decisions, but the eval fold has n = 300 and the cal fold n = 500. The counts look like stream chunks, but L147 never names the fold.
- The same table gives TruthfulQA and StrategyQA store counts. Its caption explains this as "calibration-fold rescore". But tab:benchmark-pools gives Calib = 0 for both, and the tab:validation-gate caption says "the storage gate does not fire on transfer-panel queries by design".
- tab:cross-bench-summary likewise reports TruthfulQA Store n = 67 and StrategyQA 21, and L317 relies on the TruthfulQA store precision of 0.522.
- L147 calls it a "four-outcome decomposition", but the matrix has no abstain column.
- The three tables in this subsection use three different folds: decision-distribution uses the cal fold, the matrix most likely the stream chunks, and abstention the eval fold. None of this is stated in prose.
- Fix: state the fold in the first sentence of each paragraph. Fix the "does not fire on transfer" wording (the gate is computed but not committed to memory). Either add an abstain column or say "three-outcome".

### F8. The purity "anchor" and per-benchmark minimum are not in any table (MISSING EVIDENCE). Severity: medium-high
- L208: "The pre-specified cycle-zero anchor at pi_store ~ 0.64". L239 and the ch4 receipts reuse 0.64. No table shows 0.64.
- The C0 cal-fold value in tab:pi-store-trajectory is 0.746. sec:adj-top gives 0.796 on the validation fold. That makes three different C0 precisions; say which one 0.64 is (a registered floor, or a sweep reading).
- L208: "The per-benchmark minimum on the pooled cal-fold axis ... sits at 0.6637 on the TriviaQA stream chunk at cycle five". tab:pi-store-trajectory gives per-benchmark values only for the eval fold, not for cal-fold or stream chunks.
- L232: band width compared with binomial SE "at the pooled-fold sample size of one thousand five hundred queries". The relevant n is the stored count, 575-1,001 (tab:pi-store-trajectory). Ch4 L603 says the reading is on the stream-chunk sample. Three different denominators are in use.
- Fix: add a "registered floor" footnote to tab:pi-store-trajectory, add per-benchmark cal-fold or stream-chunk columns, and correct the n.

### F9. The CES CAL and VER axes have no visible inputs (MISSING EVIDENCE). Severity: medium-high
- tab:ces-caem-trajectory gives CAL C0 = 0.898, which implies ECE of about 0.051. tab:calibration-trajectory reports ECE at C0 as "---".
- VER is computed from "per-cycle composite Cohen's d on the training-panel calibration fold" (L68). Those d values are never tabulated.
- VER C0 = 0.174 implies d of about 0.31. The only C0 d shown is +0.635, on the eval fold. A reader cannot reproduce L120's headline "verifier-reliability lift of 0.274".
- Fix: add C0 ECE and per-cycle cal-fold d rows to tab:calibration-trajectory, or to the new F1 table.

### F10. "Calibration-versus-evaluation gap" subsection: missing evidence, off-title content, wrong pointers (MISSING / MISPLACED / DANGLING). Severity: medium-high
- L317: the motivating evidence for rejecting conformal ("0.9643 over 56 stored entries", "dropped fifteen to fifty percentage points") has no table. It says "run artefacts of this rejected formulation were not retained". This is design history, and it belongs in ch4 or a design-alternatives section.
- The subsection never shows a cal-versus-eval gap for the locked configuration. tab:pi-store-trajectory has exactly that (cal 0.732-0.746 vs eval 0.693-0.759), but it is not cited here.
- L319-L327 (recall complement, asymmetric rollback, causes of the store-rate imbalance, corpus expansion) do not match the title. They are methodology (ch4), discussion (sec:disc-*) and future work (ch6).
- L323: "The per-benchmark store-rate imbalance reported above". The rates (12.6 / 29.0 / 16.2%) appear only in tab:cross-bench-summary and are never stated.
- L323: claims about "benchmark-conditional retrieval-grounding success" and the "empirical mean grounding score on stored-correct outputs landing in the same range across benchmarks" give no number. tab_grounding_success_distribution.tex exists and is never \input.
- L321: "the asymmetric-rollback property formalised in \Cref{sec:retention}". sec:retention is a results subsection; the formalisation is in ch4.
- L327: "the base-model floor reported in \Cref{cor:corpus-floor}". That corollary is explicitly deferred, with no receipt (L246, tab:theorem-receipts-summary row C3).
- Fix: keep L317 and point it to tab:pi-store-trajectory. Move L319-L321 to ch4 (retroverify / deferred buffer) and L323-L327 to the Discussion. \input the grounding-success table, or cut the grounding claims. Fix the two pointers.

### F11. A promised "survival-cycles distribution" that the thesis says does not exist (DANGLING). Severity: medium
- L319: "the empirical receipt for cumulative recovery is the survival-cycles distribution registered against \Cref{cor:self-correction}".
- L246 and ch4 L649 both say the passes-survived counter was not persisted, so this distribution was deferred.
- The label tab:survival-distribution is attached to a table of deferred-buffer reconsider dynamics, not survival counts. The name reinforces the false promise.
- Fix: rewrite L319 to point to the promotion counts. Rename the label to something like tab:deferred-reconsider.

### F12. The failed-iteration claim has no evidence, and the audit table is never \input (MISSING). Severity: medium
- L352: "The full audit summary across pre-fix and post-fix phases is auto-generated by the chapter-five artefact-generation script." tab_audit_summary.tex (label tab:audit_summary) exists but is not \input anywhere.
- No pre-fix gate value is given. tab_audit_summary row C13 has one ("Composite Cohen's d only +0.086").
- That audit ledger describes a different, 30-bug audit, not the signal-dump bug narrated in L352.
- Fix: \input the audit table in an appendix and cite it, and give the pre-fix d. Otherwise cut the sentence.

### F13. "Per-tier latency and cost" shows neither per-tier latency nor cost (TITLE MISMATCH, UNINTERPRETED RESULT). Severity: medium
- L174 explains why per-tier latency is not decomposed, and the cost projection lives at L483 (sec:comp-tier-baseline) with no pointer from here.
- tab:tier-latency shows B1 at 142 ms against CAEM at 13.6-14.7 s, about 100x slower. The prose never mentions this, and an examiner will.
- Panel (c) (per-benchmark latency; CSQA 8.6 s vs TruthfulQA 17.8 s) is never discussed.
- L174 says wall time is "in seconds"; the table reports hours.
- Fix: retitle the subsection "Pooled latency". Add one sentence interpreting the 100x overhead and pointing to sec:comp-tier-baseline. Fix the units.

### F14. The same numbers appear in many tables (REDUNDANCY). Severity: medium
- Retention probe ratios appear three times: tab:pooled-trajectory, tab:sil-training-trajectory and tab:stochastic-equilibrium all carry the same MMLU/TQA-test/CSQA-test ratios. Prose repeats them at L116, L120, L194, L232 and ch4 L675. Keep them in pooled-trajectory, drop them from the other two, and put the commit-fraction line in pooled-trajectory or a caption. tab:stochastic-equilibrium's "Attempts" column is all 1s.
- Per-benchmark EM drift C0->C5 is narrated three times (L127, L138, L190). tab:continual-retention is the EM rows of tab:per-bench-trajectory plus min/max. Cut the table and make L190 point back.
- tab:tier-chm-trajectory's EM and pooled columns equal tab:tier-trajectory panel (a). tab:tier-trajectory panel (b) "copies the C5 rows of \Cref{tab:per-tier-em}" (its own caption). Merge the CHM columns into tier-trajectory, drop tier-chm, and drop panel (b).
- Tier-1 "7 of 9,000, all correct" is stated at L163, L167, L239 and in the receipts table.
- Eval-fold n_stored and pi_store appear in both tab:pi-store-trajectory and tab:storage-pool-quality panel (a).
- Deferred-buffer totals (1,123 / 2,399 / 2,692 / 4,068 / 4,284) appear in tab:memory-store-trajectory, tab:survival-distribution and tab:deferred-buffer-ttl. Merge the TTL table into survival-distribution, or move it to an appendix.
- n_episodes_used (107 ... 5,006) appears in both tab:sil-training-trajectory and tab:storage-pool-quality.
- The C0 boost weights and intercept appear in both composite_weights and tab:calibration-trajectory column C0.
- The composite-d row appears in both tab:validation-gate and tab:cohen-d-per-bench (row u_stored).
- sec:adj-discipline (L276) restates L41-L43 almost line for line: six-fold partition, content-hash guards, panel membership. Cut it to two sentences and point to sec:setup-benchmarks.

### F15. Chapter 4 already narrates the chapter 5 results (REDUNDANCY ACROSS CHAPTERS). Severity: medium
- Ch4 "Empirical receipt" paragraphs (ch4 L532, 534, 582, 603, 625, 649, 675) restate sec:theory-checks almost verbatim: the FEVER 0.614-0.634 dip disclosure, the three monotonicity transitions, the bounded band of width 0.014, the sub-claims (i)-(iv) with 0.80, and the 13->4% prune rates.
- Fix: in ch4, keep only "what would falsify X; see \Cref{sec:check-...}" and keep the numbers in ch5.

### F16. Theory-checks section holds mechanism tables; titles don't match content (MISPLACED). Severity: medium
- sec:theory-checks carries 12 tables. Seven are pool-mechanics descriptions rather than tests of a claim: memory-store, storage-pool-quality, u-stored distribution, em-conditional, consolidation, survival, deferred-buffer-TTL.
- Consolidation (L254) is tied to no theorem.
- sec:check-corollaries is titled "equilibrium structure and self-correction". The equilibrium material is in sec:check-convergence; this subsection actually covers deployment-cost non-identifiability, the corpus floor and four pool-maintenance tables.
- tab:sil-training-trajectory (training loss) sits under "Retention and continual learning". Loss is not retention.
- Fix: add a subsection such as "Memory-pool and SIL dynamics" before Theory checks, holding memory-store, storage-pool-quality, u-stored, sil-training, em-conditional, consolidation and survival. Move the TTL table to appendix C. Leave sec:theory-checks with pi-store, stochastic-equilibrium, tier-chm, self-correction and the receipts summary.

### F17. Captions promise prose discussion that never happens (DANGLING / ORPHAN PANELS). Severity: medium-low
- tab:halluc-subtypes: "The B1-to-cycle contrasts on each subtype are discussed in the surrounding prose". L127 discusses only C0->C5. The B1 comparison is unflattering and visible: confident confabulation is B1 0.268 vs C5 0.269, and false refusal is B1 0.060 vs C5 0.064. So most of the "19% CHM reduction" is C0's false-refusal spike (0.305) being undone, and CAEM C5 CHM (0.109) is only about 0.011 below B1 (0.120). An examiner will ask about this.
- tab:tier-trajectory: "per-benchmark tier-mix heterogeneity surfaced by panel b are discussed in the surrounding prose". It is not.
- tab:decision-distribution panel (b), the per-benchmark C3 snapshot, is never mentioned in L143-L147.
- tab:safety-override panel (b) (3 stream-chunk firings, combined >= 66) is never mentioned in L181.
- Fix: either add one sentence each, or delete the panel and the caption promise.

### F18. Termination gate not linked to its evidence (MISSING POINTER). Severity: low-medium
- L104 and L116 say the run ended under the "equilibrium-saturation gate". Its definition is in appendix F and its per-cycle verdicts are in appendix C (tab:app-per-cycle-equilibrium). Neither is cited from ch5 L1-354.
- Fix: add a \Cref to tab:app-per-cycle-equilibrium at L116.

### F19. Small inconsistencies an attentive examiner may spot. Severity: low
- tab:per-tier-em caption says Tier 2 beats Tier 3 at C0 on three benchmarks; L167 says four. TruthfulQA (42.6 vs 29.1) makes the prose right. The caption also says "on TriviaQA the Tier-3 ... holds a slight EM lead", but at C2 Tier 2 leads, 51.0 vs 48.5.
- L239: "CHM_1 = 0 < CHM_2 < CHM_3 at every cycle". Tier 1 never fires at C0 or C1.
- L254 compares with 1,061 prunes / 650 promotions (run log). The tab:consolidation-trajectory caption uses 1,094 / 620 (offline reconstruction).
- L250: "the storage pool's ~78% correct rate" is in no table. Stream pi_store is 0.724-0.753 and cal-fold 0.732-0.746.
- L246 says "locked verifier ensemble"; the tab:self-correction-check caption says "an improving verifier".
- L301 says "carries forward five parameter values"; tab:sweep-top lists seven (adds tau_retro and rho_min).
- The tab:theorem-receipts-summary caption defines a "PROJECTED" verdict that no row uses. Row IDs skip C2.
- Pool sizes "at pass" in tab:self-correction-check (2,945 at C2; 10,214 at C5) do not match tab:memory-store-trajectory (2,818 at C1; 9,961 at C4). The definitions need a footnote.
- p_ground^mean and p_ground^atomic are identical in every cell of composite_weights, tab:calibration-trajectory panel (d) and tab:cohen-d-per-bench. This looks like a duplicated signal or a bug; explain it in a footnote or fix it.
- L138 gives the training-pooled EM peak at C1 (0.584), but tab:per-bench-trajectory computes its "best cycle" deltas at C2 (0.581). Say that "best" is joint EM/CHM.
- L111 apologises for the section order. That is fine, but it signals that 5.2 is pre-run methodology. Consider a three-line "locked configuration at a glance" in sec:setup so readers need not jump ahead.
