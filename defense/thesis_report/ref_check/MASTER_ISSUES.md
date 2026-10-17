# Master list of issues and limitations (consolidated 2026-10-17)

One list of everything found so far: the clean-up tracker (C/D/T/S ids), chapter 5 limitations and threats,
and the 2026-10-17 audit. Each item says what it takes to fix. The plan agreed with the author: solve all of
them, then re-run the experiment if needed, then rewrite the affected thesis text once.

Fix types:
- **RUN** needs a new main run (cycle zero + cycles), because the fix changes what the system stores or learns.
- **BASE** needs baseline answers re-generated (inference only; local RTX 3060 is enough).
- **SCORE** needs re-scoring of saved answers only (CPU / API).
- **TEXT** thesis wording only.
- **REPO** code, script or file clean-up; no effect on results.
- **INHERENT** a property of the study's scope; reported, not fixed.

## A. System faults (would change what CAEM stores or learns)
| # | Issue | Fix | Type | Ref |
|---|---|---|---|---|
| A1 | Token-confidence signal scores the whole response (reasoning + answer) after the bare question, not the prompt that produced it; near-useless as a signal | Score only the answer tokens, under the real tier prompt | RUN | C16 |
| A2 | Confabulation early exit could not fire at C0 (0/1500) and fired at most 23/1500 later, because A1 caps internal confidence | Follows from A1; re-check thresholds after A1 | RUN | C17 |
| A3 | Question-match signal scores the bare wrapped answer and is converted twice; cannot detect off-topic answers (acts as a weak correctness cue) | Remove the second conversion; redesign (claim built from question + answer, answer-type check) or drop | RUN | C14, C15 |
| A4 | Atomic-fact signal equals the mean-grounding signal on every sample (length gate skips short answers); duplicate input to the composite | Drop it, or apply it to the reasoning chain | RUN | C9 |
| A5 | Pre-routing temperature fit hits its cap on FEVER/TriviaQA; safety override nearly inactive | Fix the temperature fit or the floor; re-check | RUN | C7 |
| A6 | Phantom context: after C1, 93–99% of zero-shot answers cite a "context" they never had (training chains written under the retrieval prompt) | Strip or rewrite context references in training chains before fine-tuning | RUN | C12 |
| A7 | About a fifth of the training pool is wrong; self-improvement erodes accuracy after C1–C2 | Stricter training pool (e.g. confidence weighting, cleaned pool), test with an ablation | RUN | D6, ch5 |
| A8 | Retention guard does not probe TruthfulQA; the CommonsenseQA and TruthfulQA losses went undetected | Add probes for the transfer panel / CommonsenseQA | RUN | ch5 |
| A9 | Composite refitted every cycle; cannot separate gate drift from model change (monotonicity check fails after C2) | Add a frozen-composite control cycle | RUN | ch5 |
| A10 | Verifier checks only the final answer, never the reasoning steps | Step-level checks on the chain (ties to A4) | RUN | memory |
| A11 | Tier-1 memory lookups: FEVER prompt boilerplate dominates the question embedding (5/55 probe failures were label flips) | Strip the benchmark prefix before embedding, or raise the Tier-1 threshold | RUN | memory |
| A12 | Early-stop check was mis-wired (fired at C5 on two wrong inputs) | Already fixed in code; a new run uses the fixed check | RUN (fixed) | C1 |
| A13 | Semantic-entropy signal disabled (0.5 sentinel) but still computed; contradiction and two diagnostics computed but unused | Remove dead paths | REPO | C8, C11 |

## B. Experiment design limits (need extra runs, not code fixes)
| # | Issue | Fix | Type | Ref |
|---|---|---|---|---|
| B1 | Single seed, single backbone | More seeds; a second backbone | RUN | ch5 |
| B2 | Only six cycles; no convergence rate or limit can be estimated (H3/H4) | At least 15 kept cycles | RUN | ch5 |
| B3 | Tier 1 never exercised on evaluation (fresh, non-repeating questions) | A repeated-question evaluation stream with a realistic repeat rate | RUN | ch5 |
| B4 | Classifier ablation compares two runs that also differ in other corrections | A run with the corrections but without the classifier | RUN | ch5 |
| B5 | Classifier not tested on a benchmark held out from its training | Held-out benchmark test | RUN | ch5 |
| B6 | Per-tier latency could not be separated (batched log); CAEM ~100x slower than B1 | Batch-size-1 timing per tier | RUN (short) | ch5 |
| B7 | Deferred-buffer survival per entry not recorded; only three reconsideration passes | Record pass counts; longer run | RUN | ch5 |
| B8 | Corpus ceiling not measured | Proxy measurement | RUN (short) | ch5 |
| B9 | Calibration fold 500 per benchmark (isotonic regression reliable from about 1,000) | Larger calibration fold | RUN | ch5 |

## C. Baseline faults (re-generate baseline answers; local GPU)
| # | Issue | Fix | Type | Ref |
|---|---|---|---|---|
| C1 | Prompt mismatch: B1, B2, B5, B8 and FLARE's first draft got the bare question, CAEM got task instructions ("SHORT answer", FEVER/CSQA/StrategyQA rules) on all five benchmarks; thesis claims matched protocol | Re-run with CAEM's exact question message | BASE | D16, new |
| C2 | B5's five demonstrations are all FEVER; it answers "supports/refutes" on 235/300 TriviaQA and 289/300 TruthfulQA | Benchmark-matched demonstrations | BASE | D15 |
| C3 | B3/B4: 32 empty TruthfulQA answers (generation failure) | Re-generate those answers | BASE | D18 |
| C4 | B8 has no valid CHM (corrected run never put through the verifier) | Verifier pass on the corrected B8 run | BASE/SCORE | D1 |
| C5 | B6 (vanilla fine-tune): retention not measured; no "Answer:" line; its cycle-3 CHM file is a copy of cycle 5; weights availability not yet checked | Check weights; measure retention; re-run with matched prompt | BASE | D4, D19 |
| C6 | STaR and Self-RAG not run | Out of scope, or add | BASE | ch5 |

## D. Scoring faults (re-score saved answers)
| # | Issue | Fix | Type | Ref |
|---|---|---|---|---|
| D1 | TriviaQA strict match marks full-sentence correct answers wrong (B1 0.293 strict vs 0.553 lenient; CAEM +0.03); threatens the B1→C0 EM claim | Mostly removed by C1; then judge or lenient check for all systems | SCORE | D16 |
| D2 | TruthfulQA judge reads the whole response and credits 30% of refusals whose references contain no refusal (CAEM C0 most affected) | Judge the final answer only; a refusal counts only if the references include one | SCORE | D17 |
| D3 | Off-topic failure type cannot be measured (detector fails; LLM judge kappa 0.35) | Excluded from CHM (done); validated detector is future work | SCORE (done) | C14, C15 |
| D4 | CHM partly measured with CAEM's own verifier | Verifier-free check added (done); external judge for two subtypes is future work | SCORE (done) | ch5 |
| D5 | CHM used ROUGE on TruthfulQA | Fixed: Haiku label | SCORE (done) | D11 |
| D6 | CHM significance test was approximate | Fixed: paired Wilcoxon + Holm | SCORE (done) | C3 |
| D7 | B4 pooled EM dropped 16 empty answers | Fixed: 0.461 | SCORE (done) | D12 |
| D8 | Cycle-zero evaluation files carry the legacy composite score; C0 storage-precision row mixed folds | Workaround in place; a new run removes it | SCORE | D2, D3 |
| D9 | Tier-1 probe field `top1_is_source_rate` always 0 (id-space bug) | Fix the comparison | SCORE | C2 |

## E. Thesis text still to fix
| # | Issue | Fix | Type | Ref |
|---|---|---|---|---|
| E1 | Token-confidence signal and early exit described as working as designed (ch1–ch4, ch5, appendices) | Describe A1/A2 as they are (or after re-run) | TEXT | C16, C17 |
| E2 | Matched-protocol claim for baselines is false on all five benchmarks | Rewrite after C1 | TEXT | new |
| E3 | Chapter 6 not yet revised: stale CHM numbers, old claims, must-contain list | Chapter 6 pass | TEXT | D14 |
| E4 | Appendices: Drive folder names (A), ARC rows (B), appendix C, citations (G) | Appendix pass | TEXT | D7 |
| E5 | Abstract 709 words; numbers to refresh | Condense after results settle | TEXT | |
| E6 | Bibliography key `fu2025collapse` (first author Yi) | Rename key | TEXT/REPO | D10 |

## F. Repository and reproducibility
| # | Issue | Fix | Type | Ref |
|---|---|---|---|---|
| F1 | Twelve committed tables hand-edited; generators do not reproduce them | Make generators emit the final layout from source JSON | REPO | T1–T11 |
| F2 | Section-5 aggregate scripts run on import and overwrite tables | Guard with a main block | REPO | S1 |
| F3 | No requirements file | Add one from appendix E versions | REPO | S2 |
| F4 | Evidence JSONs force-added from an ignored folder | Decide repo vs Drive evidence | REPO | S5 |
| F5 | Superseded scripts and tables still present | Archive with README | REPO | S3, S6, T6, T8 |
| F6 | Stale comments and dead config fields | Clean | REPO | C4, C5, C6, C10, C13 |
| F7 | Cold-start memory built by several steps, not one script | One documented stage | REPO | D9 |
| F8 | Dropped-benchmark receipts only on Drive; no per-cycle git tags | Document | REPO | D8, S4 |

## G. Inherent limits of scope (reported, not fixed)
| # | Issue |
|---|---|
| G1 | Short-answer factual and commonsense questions only; no long-form, code, maths, dialogue, other languages |
| G2 | Memory reached 12,175 entries; behaviour at millions untested |
| G3 | Training filtered by a verifier converges toward the verifier's own knowledge (Yi et al.) |
| G4 | Agreement-based signals miss misconceptions the model holds consistently; retrieval can reinforce them (TruthfulQA) |
| G5 | Binary and three-way tasks reward lucky guesses with poor reasoning |
| G6 | Low-rank adaptation learns less and forgets more with training |
| G7 | Conformal guarantees dropped because calibration and evaluation data differ |
| G8 | Long-input judge dropped; MiniCheck truncates inputs over 512 tokens |
