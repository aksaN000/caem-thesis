# README.md fix log (2026-10-15)

Only `README.md` was edited. Original copy: `README.orig.md`; exact line diff: `readme.diff` (both in this scratchpad folder).
Sources: `thesis_report/ref_check/fixes_applied.md` (FA), chapters 1/4/5/6, appendices A/E, `figures/auto/*.tex`, `ref_check/SYNC_PLAN.md`, `ref_check/drive_sync_check.py`, `ref_check/local_regen_check.py`.

## 1. "pre-registered" -> "pre-specified" (source: FA "Follow-up: pre-registered -> pre-specified"; chapter_3.tex definition sentence)

| Location | Before | After |
|---|---|---|
| l.55 | pre-registered five-hypothesis verdicts | pre-specified five-hypothesis verdicts (line rewritten, see 2) |
| l.63 | the pre-registered hypothesis adjudication | the pre-specified hypothesis adjudication |
| l.91 | ablated under the pre-registered Pearson correlation floor | ... pre-specified ... |
| l.280 | A pre-registered conjunction of four checks | A pre-specified conjunction ... |
| l.295 | ## Pre-registered hypothesis verdicts | ## Pre-specified hypothesis verdicts |
| Threats item 3 | failed the pre-registered Pearson correlation floor | ... pre-specified ... |

## 2. Theory (source: chapter_4.tex sec:theory; tab_theorem_receipts_summary.tex; chapter_5.tex sec:theory-checks; FA "Theory restructure")

| Location | Before | After |
|---|---|---|
| TL;DR l.55 | The full theorem-by-theorem receipt panel (three theorems plus three corollaries) and the pre-registered five-hypothesis verdicts are reported in Chapter 5 §5.5. | The receipt panel for the two proven results (one theorem and one proposition) and the five design heuristics is reported in Chapter 5 §5.1.6; the pre-specified five-hypothesis verdicts are in Chapter 5 §5.5.2. |
| Section heading | ## Theorem-by-theorem closeout (canonical reading from Ch4 §4.11 + Ch5 §5.5.1) | ## Theory closeout: proven results and design heuristics (canonical reading from Ch4 §4.3 + Ch5 §5.1.6, `tab:theorem-receipts-summary`) |
| Intro | "Three theorems and three corollaries carry the formal load ..." | Two proven results vs five heuristics; idealising assumptions (Gaussian score classes, stationarity, independent verifier errors); heuristics have argument sketches and directional receipts; four scaffolding remarks listed by label (ch4 l.506-509, l.559-562, l.677-680) |
| Table | 6 rows; `thm:purity`, `cor:tier1-floor`, `cor:self-correction` marked "Complete"; others "Sketch"; no `thm:bayes-purity` row | 7 rows. `thm:purity` Theorem, "Proven (proof sketch)" (ch4 uses `proof`[Proof sketch]); `thm:bayes-purity` Proposition, "Proven (complete proof)"; `thm:monotone`, `thm:convergence`, `cor:tier1-floor`, `cor:self-correction`, `cor:stochastic-equilibrium` all "Heuristic (argument sketch)". Receipts/verdicts copied from tab_theorem_receipts_summary: T1 PASS (pooled cal fold), band [0.732, 0.746], 9-11 pp, FEVER eval-fold [0.614, 0.634] at C1-C4; T2 precondition audited, d = +0.635, BA ~0.673; T3 three transitions (C1->C2 aligned, C2->C3 sub-noise, C3->C5 small reverse sign), was "1 of 3 ... 1 confounded by composite refit"; T5 band width 0.014 (0.746-0.732), rate form not measured; C1 capability bound, 7 firings of 9,000, ceiling ~0.36; C4 partial, now states the independence premise and that systematic errors are not removed (ch4 heuristic statement + scope note, FA C9/C10); C5 (i),(iii) PASS, (ii) inherits bounded band, (iv) heuristic, memory 431 -> 12,175. Unescaped `|π_t − π_∞|` replaced by escaped pipes so the table renders. "Phase 1f future work" dropped (not in thesis). |
| Closing line | The full proof sketches ... are in Chapter 4 §4.11 and Chapter 5 §5.5.1. | The statements, proofs, argument sketches and their idealising assumptions are in Chapter 4 §4.3; the empirical receipts are in Chapter 5 §5.1.6. |
| H1 row | (`thm:purity` precondition) | (`thm:bayes-purity` precondition) — ch5 tab:hypothesis-verdicts H1 |
| l.307 | cross-reference back to the theorem-receipt section | ... theory-receipt section |
| l.123, l.251 | the corollary's sub-claim (i) | the heuristic's sub-claim (i) |
| l.214 | the framing is `cor:tier1-floor` in Chapter 4 §4.10 and the receipt is in Chapter 5 §5.4 | the framing is the Tier-1 floor heuristic `cor:tier1-floor` in Chapter 4 §4.3.3 and the receipt is in Chapter 5 §5.1.6 |
| Threats item 2 | Sketch-only proofs on three of six load-bearing statements (`thm:monotone`, `thm:convergence`, `cor:stochastic-equilibrium`) ... registered as a Phase 1f item in Ch6 §6.2 ... | Five of the seven formal statements are heuristics (lists all five), argument sketches under idealising assumptions, directional receipts; supermartingale construction future work; fixed-composite ablation Ch6 §6.2.10; 15-cycle probe Ch6 §6.2.11; systematic verifier errors not removed by repeated re-verification (ch4 scope note) |
| Future work item 3 | Rigorous supermartingale proof ... adapted from Robbins-Monro and Polyak-Juditsky in a dedicated proofs appendix. | Rigorous supermartingale construction for the `thm:convergence` heuristic, adapted from Robbins-Monro, Robbins-Siegmund and Polyak-Juditsky. (ch4 l.600; FA notes the false "relocated to Appendix E" sentence was removed; no proofs appendix exists) |
| Future work items 1 | for `thm:monotone` | for the `thm:monotone` heuristic |

## 3. Reproducibility (sources: SYNC_PLAN.md, drive_sync_check.py MAP, local_regen_check.py docstring, appendix_e.tex l.26-40, FA fixlog_rest header)

| Location | Before | After |
|---|---|---|
| Step 1 | Clone this repository and pin to commit `c5b8066` (pre-Step-7 anchor) or `700b9e3` (retrieval-utility-classifier-era anchor), depending on the trajectory phase to reproduce. | Check out `branch/ruc-classifier`, the final thesis version; `c5b8066` / `700b9e3` kept as historical anchors matching the pre-main snapshot bundles of Appendix E, not the thesis version. |
| New `### Data map` (after step 5) | — | Table: outputs/full_run -> full_run (+ Drive-only cycle_1..cycle_5/ and global/); outputs/baselines_phase1e -> phase1e_baselines (B6 vanilla_ft/eval/*_cycle5.json and B8 semantic_entropy/*_cycle0.json from the 13 June 2026 fix are canonical; earlier copies do not reproduce the tables — SYNC_PLAN: Drive copy gives pre-fix B6 0.325 and a zero-shot copy of B8); outputs/cycle_0 -> cycle_0_cold_start; outputs/tier1_routing_probe -> tier1_routing_probe |
| New `### Checking the tables against the data` | — | local_regen_check.py regenerates every table in a temporary copy and diffs against the committed tables without touching the repo or outputs/; several tables are hand-finished (`% Hand-corrected 2026-10-15`, 12 files) and some generators differ in layout, so compare numbers not formatting; audit reports in thesis_report/ref_check/ |
| Hardware table, main-run row | dominated by per-cycle fine-tune plus cycle-boundary retroactive re-verification | dominated by verifier-side scoring (stream-chunk admission, evaluation and retroactive re-verification); the per-cycle fine-tune takes about 35–45 minutes (appendix_e "Production envelope" and "Per-cycle wall-clock budget"; also README's own paragraph below the table) |

## 4. Other contradictions with the corrected thesis

| Location | Before | After | Source |
|---|---|---|---|
| l.49 | The architectural fix is the retrieval-utility classifier (next bullet). | The retrieval-utility classifier (next bullet) narrows this regression but does not close it. | ch5 l.491 ("narrows"), ch5 sec:ruc-ablation second gate does not close |
| l.50 | ...lifts TruthfulQA exact match from 0.210 to 0.310 while preserving the training-panel pooled reading. | ...0.210 to 0.310, still below the B1 zero-shot reference of 0.433, while preserving ...; contrast bundles the classifier with pipeline corrections | ch5 l.491, l.459-473; FA "#11 (second site)" |
| l.151 | §5.7 lifts TruthfulQA EM from 0.210 to 0.310 by | §5.4.2 lifts ... 0.310 (still below the B1 reference of 0.433) by | same |
| RUC ablation intro | (no confound note) | adds: classifier-on arm also ships pipeline corrections; joint effect; arm-isolating re-run in Ch6 §6.2.7 | ch5 sec:ruc-ablation; ch6 sec:future-ablations; tab_ruc_ablation caption |
| Threats item 6 | the architectural fix is the RUC ablation in §5.7, which lifts TruthfulQA exact match by +10.0 pp | ablation in §5.4.2 narrows the gap, 0.210 -> 0.310 (+10.0 pp), still below B1's 0.433; joint effect with pipeline corrections | same |
| "The problem" l.61 | "exceed their actual accuracy by ten or more percentage points (guo, kadavath)"; "Single-signal correctness predictors achieve AUROC 0.50–0.60 (xiong)"; "Clinical-question studies report hallucination rates of 69–77% (alkaissi)"; "structural property ... (`ji2023survey`, `huang-etal-2023-large`)" | ECE above 10 pp on image benchmarks (guo); RLHF models miscalibrated until rescaled (kadavath); verbalised confidence AUROC ~0.63 for GPT-4, no method consistently best (xiong); non-existent references (alkaissi) and 39.6/28.6/91.4% hallucinated references (info:doi/10.2196/53164); persists in large, instruction-tuned and preference-tuned models (`ji2023survey`, `huang2025survey`; `huang-etal-2023-large` deleted from .bib) | FA Tier B rows 1, 4, 5, 6 |
| l.63 | overconfidence (single-signal correctness predictors are near-random) | overconfidence (the confidence a model verbalises discriminates correct from incorrect answers only weakly) | FA Tier B row 12; ch1 l.83 |
| l.95 | logistic boost layer (Cherian-style, C = 0.01) | logistic boost layer (C = 0.01, following the logistic multi-scoring of Valentin et al.) | FA Tier B rows 13, 32, 83, 92 |
| Config table, boost C | L2 regulariser on Cherian boost layer | L2 regulariser on the logistic boost layer | same |
| l.91 and config table | h_norm boost coefficient "approximately −5e-4" / "≈ −5e-4" | boost-layer weight "within 5e-4 of zero on every fit" | ch4 l.285 |
| Config table, safety_u_pre_min | re-tuned at cycle one from the registered 0.60 | + per-benchmark refit raised CommonsenseQA's floor to 0.491 (C3, C4) and 0.569 (C5); override fired on 63 of 9,000 evaluation queries (Ch5 §5.1.4) | ch5 sec:tier-safety; FA "Safety override recount" |
| Validation gate table, poisoning floor | ≤ 0.50 | < 0.50 | tab_validation_gate ("< 0.50"); ch5 sec:adj-gate ("below 0.50") |
| Sig-test paragraph | the bootstrap 95% confidence intervals on every ΔEM ... Chapter 5 §5.4 | the paired percentile-bootstrap 95% confidence intervals (1,000 replicates) on every per-benchmark ΔEM ... Chapter 5 §5.3 and §5.4.1 | ch5 sec:sig-bootstrap |
| Threats item 1 | the fixed-threshold replacement is robust to the shift | makes no marginal-coverage claim that the shift can silently void, but the shift can still lower realised precision (TruthfulQA store precision 0.522 at cycle zero) | FA Tier C rows C5–C7 |
| Future work intro | The thesis registers six future-work items | Chapter 6 §6.2 registers eleven future-work items; the six below tie most directly to theory/headline (items 3 and 4 registered in Ch4 §4.3 and Ch5 §5.1.6) | ch6 has 11 subsections (6.2.1–6.2.11); supermartingale and paraphrase items are registered in ch4 l.600 and ch5 l.239 |
| Future work item 4 | lifts the 0.13% Tier 1 firing rate | lifts the Tier 1 firing rate (τ_1(C5) = 0.0013) | ch5 l.239 (0.13% was the C5 rate, not the 7/9,000 = 0.08% overall rate) |

### Section-number pointers (all pointed at non-existent or wrong subsections; numbering checked against chapter source and the TOC of the current main.pdf)

| Claim / table | Before | After |
|---|---|---|
| tab:per-bench-trajectory, tab:pooled-trajectory, CHM −19% | Ch5 §5.5 / §5.2 | §5.1.2 |
| tab:ruc-ablation | §5.7 | §5.4.2 |
| tab:pi-store-trajectory (TL;DR) | §5.5.1 | §5.1.6 |
| C4 rollback (TL;DR, trajectory findings) | §5.2 | §5.1.5 |
| RUC spec | Chapter 4 §4.5 (x2) | §4.2.5 |
| conformal rejection evidence | Ch4 §4.7, Ch5 §5.5.1 | Ch4 §4.2.8, Ch5 §5.2.5 |
| tab:baseline-pooled | §5.4 | §5.4.1 |
| licensing rule | Chapter 5 §5.4 | §5.3.5 |
| tab:tier-trajectory, tab:per-tier-em | §5.4 | §5.1.4 |
| tab:sil-training-trajectory | §5.5.4 | §5.1.5 |
| asymmetric rollback registration | Chapter 4 §4.9 | §4.2.10 |
| tab:decision-distribution | §5.3 | §5.1.3 |
| tab:validation-gate; failed first iteration | §5.1 | §5.2.7; §5.2.8 |
| sec:adj-cal-eval-gap (threat 1) | Ch5 §5.6; Ch6 §6.2 | Ch5 §5.2.5; Ch6 §6.1.3 (Open questions) |
| judge ablation (threat 3) | Ch5 §5.6 | §5.2.6 |

## Checked, unchanged
All numbers in the pooled, per-benchmark, baseline, RUC-ablation, tier, per-tier EM, SIL, memory, stochastic-equilibrium, decision-distribution, validation-gate and hypothesis tables match `figures/auto` and ch5 cell for cell. Sig-test tallies (9/40, 21/35; 14/40, 20/35), Tier-1 7 of 9,000, cold start 431, RUC fit numbers (3,208 rows; 1,846/1,362; AUROC 0.850 ± 0.008), hardware (960 GPU-h, ~672 USD, 64 GB index, ~90 GB page cache, 2.2–2.6x, accumulation 4 -> 16), C2-T2 vs C0-T3 margins 4.6–20.4 pp. README does not state prune (1,061) or promotion (650) counts.

## Not changed / for the user
- l.19 "Thesis report (325 pages, PDF)": main.pdf predates the corrections and must be rebuilt; page count not re-checked.
- Appendix-section pointers (§A.1, §A.2, §A.3, §E.4, §E.5) were not re-verified.
- tab_validation_gate caption (thesis file, not README) still says the Cohen's d gate audits the precondition of thm:purity; ch5 text says thm:bayes-purity.
