# Log-search resolution: D1–D10, U1–U15

Source: Drive file 1Vi94NrJdxdR11aZvvgjldVKQPAYfD6Qx (caem_log_search.md, 415 KB after unescaping), produced by
`thesis_report/ref_check/log_value_search.py` on the user's machine (repo root `/workspace/caem`, 773 files scanned).
I read it in full. The Drive tool did not cut it off: the response was too big to return inline, so it was saved to a
file, and that file ends with the script's own U15 trailer. The script itself does cap what it shows: at most 15 files
per item and 25 lines per file. The cap matters for D1, D10, U4, U5, U7, U10, U12 and U13. For those items the
evidence could be in the hidden files, so a missing match is **inconclusive**. It is not proof that the value is absent.

Evidence classes used below:
- **RUN** means a run log or machine-written artefact (`outputs/**/*.log`, `*.json`, `*.csv`).
- **DERIVED** means an LLM or analysis write-up stored under `outputs/analysis/` (`theorem_review.json`, `*_synthesis.md`).
  These quote run files but are not run files.
- **NOTES** means `archive/*.md`, `docs/`, `scripts/*.md`. These are planning notes and write-ups, not evidence.

Also note:
- `outputs/full_run/run.log`, `outputs/full_run/experiment.log` and `outputs/run_phase1a_ruc.log` hold the same main-run
  lines at different line offsets. I quote `experiment.log` unless only another file is shown.
- The search matched a lot by coincidence. Examples: `u_stored=0.759`, `336ms/sample`, episode ids like `id=2443`,
  tokenizer ids, and HH:**53**:SS timestamps. I discarded all of these.

---

## D1: C0 stored set, 289 @ 0.796 vs 336 @ 0.759
- (a) Supported by run evidence: none visible. Matches were found in 183 files and the top 15 are shown. Every visible
  hit is a coincidence: `u_stored=0.759/0.796` per-sample values, `336ms/sample` and `289ms/sample` latencies, and
  hex ids. No pooled precision or `n_stored` line appears.
- Indirect run evidence on which fold each number belongs to:
  - `outputs/run_phase1a_ruc.log:71`: `Built pools: fever (is_training=True): seed=1000 purity=500 calib=500 sil_train=0×0 eval=500 test=500 total=3000`.
    This is the stand-alone cycle_0 run, with eval = 500 per benchmark, so the fold has 1500 samples.
  - `outputs/full_run/experiment.log:47`: `Built pools: fever ... sil_train=10×2000 eval=300 test=500 total=22800`.
    This is the main run, with eval = 300 per benchmark, so the fold has 900 samples.
- What this means: the earlier audit traced 289 to 63+145+81 from `outputs/cycle_0/eval/<bench>_cycle0.json` with n=500
  per benchmark (numbers_mismatch_origins.md A19). The logs support that the 1500-sample fold exists only in the
  cycle_0 run. So 289 @ 0.796 belongs to the **cycle-0 run's 500/bench fold (1500)**, and 336 @ 0.759 belongs to the
  **main-run C0 eval fold, 300/bench (900)**, which tab_pi_store_trajectory uses.
- **Verdict: NOT FOUND** in this output (inconclusive because of the file cap). Fix the fold label rather than the numbers.

## D2: retroverify prunes, 1094 (ch5) vs 56/271/352/382 = 1061
- (a) Supported by run evidence:
  - `outputs/full_run/experiment.log:5710` `Retroactive re-verification done: 431 total | 375 updated | 56 pruned` (C1)
  - `:19133` `2945 total | 2674 updated | 271 pruned` (C2)
  - `:33683` `5358 total | 5006 updated | 352 pruned` (C3)
  - `:60457` `Retroverify complete (batched): 9832 updated, 382 removed (loop-pruned=2, threshold-pruned=380)` (C5)
  - `outputs/full_run/retroverify_cycle{1,2,3,5}.json:6` `"pruned": 56 / 271 / 352 / 382`. C4 was skipped (aborted).
  - The sum is **1061**.
- (b) Contradicted: 1094. No file contains it. It appears at ch5:254 and in the tab_consolidation_trajectory caption.
- **Verdict: CORRECT-TO 1061**. The C5 figure of 382 is 380 threshold prunes plus 2 loop prunes.

## D3: deferred-buffer promotions, 620 vs 650 vs 131/220
- (a) Supported by run evidence:
  - `experiment.log:16404` `Cycle 2: deferred reconsideration complete -- 131 promoted, 0 TTL-dropped, 992 kept`
  - `experiment.log:30357` `Cycle 3: ... 220 promoted, 924 TTL-dropped, 1255 kept`
  - `experiment.log:55957` `Cycle 5: deferred reconsideration complete -- 299 promoted, 1169 TTL-dropped, 2600 kept (buffer size 2600, store size 10258).`
    This line appears under U11 (via the "53" in the timestamp).
  - C1 and C4 have none (C4 was aborted).
  - Total: 131 + 220 + 299 = **650**, which matches the tables and appendix C (C5 = 299).
- (b) Contradicted: 620, at ch5:254 and in the tab_consolidation_trajectory caption.
- The local-regen values 1144 and 1468 are **exits** (promoted + TTL-dropped): 220+924 = 1144 and 299+1169 = 1468.
  They are not promotions. `outputs/tab_survival_distribution.csv:4` `2,2399,1407,1276,131` is consistent with this.
- **Verdict: CORRECT-TO 650** (C2 131, C3 220, C5 299).

## D4: "B1 zero-shot floor" TQA 0.27 vs 0.433
- (a) Supported by run evidence (DERIVED csv written by the pipeline). The columns are method, bench, n, CAEM EM,
  baseline EM, and so on.
  - `outputs/full_run/tab_sig_test.csv:33` `zero_shot,truthfulqa,300,0.3,0.4333,-0.1333,...`
  - `outputs/tab_sig_test_v2.csv:33` `zero_shot,truthfulqa,300,0.3233...,0.43333...`
  - `outputs/tab_sig_test_c2_post_haiku.csv:33` and `outputs/tab_sig_test_c5_post_haiku.csv:33` say the same.
  - So **B1 zero-shot TQA = 0.433**.
- Where 0.27 really comes from (it is never B1):
  - `outputs/tab_sig_test_v2.csv:32` `vanilla_ft,truthfulqa,300,0.3233,0.2733,...` (B-vanilla-FT TQA 0.273)
  - `outputs/baselines_v2/tqa_haiku_rescore.log:456` `LLM-judged EM: 0.2713` and `:916` `baselines_v2/rag/truthfulqa_cycle0.json | 0.3060 | 0.0473 | 0.2713` (RAG TQA after LLM rescoring)
  - `tab_sig_test*.csv:10` `cot,triviaqa,...,0.27` (CoT on TriviaQA). This is a coincidence.
- (b) Contradicted: "B1 floor 0.27". No B1 TruthfulQA EM of 0.27 exists anywhere in the visible output.
- Consequence: the claim at ch5:473 that 0.310 "lifts above the B1 floor" is false. 0.310 is below B1's 0.433.
- **Verdict: CORRECT-TO 0.433.** Alternatively, relabel 0.27 as the vanilla-FT (0.273) or LLM-judged RAG (0.271) number.

## D5: RUC ablation TQA EM 0.310 (on) vs 0.210 (off); off-arm aborted at C5
- (a) Supported:
  - DERIVED: `outputs/analysis/phase1d_vs_phase1e_ruc.json:48` `"em_1d": 0.21,` and `:49` `"em_1e": 0.31,`.
    Phase 1d is the RUC-off arm and 1e is RUC-on.
  - RUN, a different number: `outputs/baselines_phase1e/launch.log:539` `EvalHarness done: truthfulqa | cycle=0 | EM=0.3100 ... Tier3=100.00%`.
    This is a cycle-0 baseline that also scores 0.310, so it should not be confused with the ablation result.
- The "aborted at C5" part is not confirmed. NOTES contradict it: `docs/PRODUCTION_END_TO_END.md:839` `Phase 1d's Cycle 4 retention guard fired (TriviaQA-test 0.395 → 0.345, ratio 0.8734 < 0.93)`,
  and `archive/branch_C_log_historical.md:217` says the same. `outputs/phase1d_full_run/cycle_6/...` exists, so the
  phase-1d run went on to C6.
- **Verdict: CONFIRMED (0.310/0.210, DERIVED json). The abort cycle of the off arm is unverified, and notes say C4, not C5.**

## D6: appendix C "CES" 0.557/0.567/0.547/0.548/0.556 vs 0.678/0.704/0.703/0.690/0.710
- (a) Supported:
  - RUN: `outputs/full_run/cycle_5/equilibrium_fit.json:4-6` `0.5573400000000001,` `0.5673199999999999,` `0.54732,`.
    This is the history array the saturation fit runs on.
  - DERIVED: `outputs/analysis/theorem_review.json:368` `Per-cycle pooled eval-fold EM (p_c proxy) from outputs/full_run/all_cycle_results.json: C0=0.5480, C1=0.5573, C2=0.5673, C3=0.5473, C4=0.5480, C5=0.5560 (n=1500 each cycle)`.
- So the appendix-C column is **pooled eval-fold EM** for C1..C5 (C0 is 0.548). It is the same series the equilibrium-fit
  record stores as its "CES" history, which is why c_inf = 0.548 sits on the EM scale.
- The values 0.678/0.704/0.703/0.690/0.710 (tab_ces_caem_trajectory) were not found as CES. Every visible hit is a
  coincidental `u_stored`/`q_a_relevance` value. The keyword "ces" matches "Choices", "process" and similar words.
- **Verdict: CORRECT-TO "pooled EM" label.** The numbers are run-backed but the column is mislabelled as CES. Either
  rename the column, or use the composite CES series if that is what is meant. The source of the composite series is
  not visible here.

## D7: storage rate 36–43% vs ~52%; admissions 2443/2477/2469/2484/2339 vs 2456/2494/2490/2522/2380
- (a) Supported (RUN): `outputs/full_run/pi_store_trajectory.csv:14` `1,stream_chunk,pooled,2443,1841,0.7536`, `:22` `2,stream_chunk,pooled,2477,1860,0.7509`, `:34` `4,stream_chunk,pooled,2484,1831,0.7371`.
- C5: DERIVED `outputs/analysis/theorem_review.json:26` `pooled minimum 0.7226 (C5 stream_chunk, n=2343)`.
  - Pool arithmetic gives the same answer: the C5 store after retroverify is 9832 (`experiment.log:60457`), and
    appendix C's close is 12175, so admissions = 12175 − 9832 = **2343, not 2339**.
- C3: 2469 is not in pi_store_trajectory.csv; no C3 row was matched. The pool arithmetic (7477 − 5006 = 2471) does not
  reproduce 2469 either. Unverified.
- (b) Contradicted: the table set 2456/2494/2490/2522/2380 appears only as coincidental episode ids
  (`Stored new episode: id=2456`). No run artefact supports them.
- Rate: admissions / 4700 = 2443/4700 = 52.0% and 2484/4700 = 52.9%. That supports **~50–53%**. The range 36–43% fits
  *correct*-stored / 4700 instead (1841/4700 = 39.2%, 1831/4700 = 39.0%), or an older run.
- **Verdict: CORRECT-TO ~52% (49.9–52.9%). Admissions are CONFIRMED for C1, C2 and C4. C5 is CORRECT-TO 2343. C3 is NOT FOUND.**

## D8: temperature 1.089/0.836/0.740/0.869
- (a) Supported (RUN):
  - `experiment.log:18385` `Cycle 2 temperature re-fit: T 1.0889 -> 0.8360 | ECE 0.050896 -> 0.049408`
  - `:32333` `Cycle 3 temperature re-fit: T 0.8360 -> 0.7404 | ECE 0.045098 -> 0.040665`
  - `:57894` `Cycle 5 temperature re-fit: T 0.7404 -> 0.8685 | ECE 0.055149 -> 0.042696`
  - `outputs/full_run/calibration/calibrated_config_cycle3.json:4` `"temperature_after": 0.7403865858515201`
  - C4 has no refit (aborted), and C5 starts from the C3 value.
- The ceiling (C0 = 20.086 = e^3) is not in this output, because it was not searched.
- **Verdict: CONFIRMED** (C1 1.089, C2 0.836, C3 0.740, C5 0.869).

## D9: MMLU retention 1.041/0.975/0.992/1.018/0.999
- (a) Supported (RUN):
  - `experiment.log:3649` `Cycle 1: post-training probes = {'mmlu': 0.635, ...} | ratios = {'mmlu': 1.041, 'triviaqa_test': 1.0127, 'commonsense_qa_test': 1.0061}`
  - `:16105` `'mmlu': 0.9754` (C2)
  - `:29739` `'mmlu': 0.9918` (C3)
  - `:44266` `Cycle 4: ... 'mmlu': 0.6212 ... ratios = {'mmlu': 1.0184, 'triviaqa_test': 0.8861, ...}`
  - `:55961` `Fine-tuning done: 5006 episodes | retention_ratio=0.999`
  - `retroverify_cycle2.json:11` 0.9754 and `retroverify_cycle4.json:12` 1.01838 say the same.
- (b) The experiment_summary percentages 93.7/95.28/97.83/95.93 do not appear in any main-run line. The only 0.9367
  hits are in `outputs/phase1d_full_run/stochastic_equilibrium_trajectory.csv:3` (`triviaqa_test,0.9367`), which is
  the Phase-1d run.
- **Verdict: CONFIRMED** (the thesis ratios). Treat experiment_summary as another run or metric.

## D10: safety override (prose "none of 9000" vs table "1 on StrategyQA, u_pre 0.376")
- (a) Supported: nothing visible. All 15 files shown hold coincidental per-sample `u_stored: 0.376x` values from
  FEVER and TriviaQA stream-chunk JSONs. They matched because a `u_pre` key sits within 3 lines. None of these files
  is a StrategyQA eval file, and none contains "safety" or "override". 59 files are hidden.
- **Verdict: NOT FOUND (inconclusive).** No eval log shows an override firing. To settle it, grep the run logs for
  `override`/`safety` directly, and check the `strategyqa_cycle*.json` files for a record with `u_pre` 0.376.

## U1: RUC data 3208 / 1846 / 1362
- (a) RUN: `outputs/baselines_v2/v2_features_extract.log:9` `ruc_extract_v2_features fresh-pool total: 3208 questions across 8 files`.
- 1846/1362: only coincidences (episode ids, tokenizer ids, years).
- **Verdict: CONFIRMED for 3208. The helpful/hurt split is NOT FOUND.**

## U2: RUC fit (AUROC 0.850, recall 0.842, spec 0.678, confusion 1554/292/923/439, 77.2%)
- Every hit is a coincidence: `u_stored=0.678`, `id: "439"`, `[439/3208]`.
- **Verdict: NOT FOUND.**

## U3: RUC per-bench AUROC, probe n=11980, AUROC 0.771, coef −1.77
- Every hit is a coincidence (u_stored values, a loss of 0.7280, tier1 probe floats). theorem_review.json:93's "0.7714"
  is a Spearman critical value.
- **Verdict: NOT FOUND.**

## U4: C0 cal sweep, τ 0.65 → 70/105, τ 0.60 → 306/478, π_store ≈ 0.64
- The search was swamped (21,759 lines in 330 files, all per-sample floats 0.64x). No sweep line is visible.
- **Verdict: NOT FOUND (inconclusive).**

## U5: split-conformal 0.9643 over 56 entries
- The only hits are `"s_avg": 0.96436...` per-sample values, which are coincidences.
- **Verdict: NOT FOUND (inconclusive).**

## U6: Qwen judge ρ 0.5843 (floor 0.70)
- The value appears in every run log as `caem.verification AdaptiveNLIJudge inactive ... Platt-fit Pearson floor (rho >= 0.70) was ...`
  (`experiment.log:20`). That line is a **hard-coded string** in `caem/verification/__init__.py:136-137`
  (`"(measured rho = 0.5843, slope 0.137; see Appendix B ..."`).
- It was not measured in this run. The run only echoes text an author wrote into the code.
- **Verdict: NOTES-ONLY** (code-embedded claim; no computed output found).

## U7: ECE-before 0.2761 / 0.0508 / 0.0509 / 0.0451 / 0.0551 (C0, C1, C2, C3, C5)
- (a) RUN (from the D8 hits): C2 `ECE 0.050896 -> 0.049408` (`experiment.log:18385`), C3 `0.045098 -> 0.040665`
  (`:32333`), C5 `0.055149 -> 0.042696` (`:57894`).
  - These round to C2 0.0509/0.0494, C3 0.0451/0.0407 and C5 0.0551/0.0427, matching the before and after columns.
- C0 0.2761 and C1 0.0508 are not visible. U7's own section is swamped by `p_entail` floats. One near-miss:
  `outputs/full_run/cycle_1/composite_calibration.json:868` `"u_dropout": 0.05078...` is probably a coincidence.
- **Verdict: CONFIRMED for C2, C3 and C5. NOT FOUND for C0 and C1.**

## U8: equilibrium fit c_inf 0.548, a −0.018, k 0.350, R² 0.207
- (a) RUN: `outputs/full_run/cycle_5/equilibrium_fit.json:28` `"k": 0.3503100934072891,`.
- a and R² were not matched. The regex misses rounded forms, for example 0.2068 is not matched as 0.207. c_inf was not
  searched.
- **Verdict: CONFIRMED for k. The rest is NOT FOUND (read equilibrium_fit.json directly).**

## U9: probe anchors (pristine 0.610/0.395/0.825; TriviaQA-test 0.350)
- (a) RUN:
  - `experiment.log:1998` `Pristine retention baseline saved: {'mmlu': 0.61, 'triviaqa_test': 0.395, 'commonsense_qa_test': 0.825}`
  - `outputs/full_run/retention_baseline.json:4-5`
  - C4: `experiment.log:44266` `'triviaqa_test': 0.35` with ratio `0.8861`
- NOTES give different numbers: `archive/branch_C_log_historical.md:217` and `docs/PRODUCTION_END_TO_END.md:839` say
  0.345 and ratio 0.8734. Those come from the Phase-1d run, not the main run.
- **Verdict: CONFIRMED.**

## U10: ROUGE-rule zero-shot TQA 0.803; chi² 0.232 / 56.860 / 76.891
- The search was swamped by per-sample floats. No ROUGE or chi² line is visible.
- **Verdict: NOT FOUND (inconclusive).**

## U11: ~53% stale-calibration C1 prune; FEVER id 13957
- RUN: `outputs/full_run/dataset_splits.json:808` `"13957",`, a bare numeric id, which is FEVER's id style. This
  confirms the id exists in the split.
- "~53%": every "53" hit is a timestamp. The actual main-run C1 prune rate is 56/431 = 13.0%.
- **Verdict: CONFIRMED for id 13957. ~53% is NOT FOUND.** It probably comes from a different (stale-calibration)
  experiment, so check the notes.

## U12: eval wall time 75–95 min per cycle; tier latency 5.7–6.1 h per cycle
- No line records evaluation wall time directly. Timestamps only bracket it. The window from fine-tuning done to
  retroverify done holds cal-fold scoring (1500), the 1500-query eval and retroverify:
  - C1: 03:55:39 → 09:40:56 = **5 h 45 m** (`experiment.log:3659` → `:5710`)
  - C2: 11:05:51 → 20:49:26 = **9 h 44 m** (`:16408` → `:19133`)
  - C3: 01:01:59 → 14:56:06 = **13 h 54 m** (`:30361` → `:33683`)
  - C5: 23:53:36 → 21:43:10 next day = **21 h 50 m** (`:55961` → `:60458`). Step 2.1 cal-fold scoring starts at
    23:53:36 (`:55962`).
- These windows are far larger than 75–95 min. tab_tier_latency's 5.68–6.11 h equals mean latency × 1500
  (13.6 s × 1500 = 5.67 h), so it is arithmetic, not a measured wall time.
- **Verdict: NOT FOUND** (no direct measurement). 75–95 min is not supported. Run evidence bounds the
  cal+eval+retroverify phase at 5.75 h (C1) to 21.8 h (C5).

## U13: dataset sizes
- (a) RUN:
  - `experiment.log:41` `FEVER: 145405 samples loaded (train split).`
  - `experiment.log:66` `StrategyQA: 687 samples loaded from ChilleD/StrategyQA:test.` This is the ChilleD *test*
    split, not "dev".
- TriviaQA 87k and CSQA ~10k after dedup are not visible. The regex cannot match "87xxx" as "87".
- **Verdict: CONFIRMED for FEVER 145k and StrategyQA 687 (relabel as the test split). The rest is NOT FOUND.**

## U14: C-sensitivity, spread ≤ ~1 pp over five C values
- NOTES only:
  - `archive/branch_C_log_historical.md:284` `25-variant sweep ... boost_C ∈ {0.01, 0.025, 0.05, 0.10, 1.0} on the same 1500-sample calibration fold`
  - `:286` `V_a050_C0.010 locked ... pooled 80.2%`
- The sweep output files (`outputs/cycle_0/sweep/`) are referenced but no spread figure is visible.
- **Verdict: NOTES-ONLY.**

## U15: consolidation ~1 s and ~7 s; 4 answer-mismatch rejections
- (a) RUN:
  - `experiment.log:16405` `11:05:50 Consolidation pass: 2948 episodes` and `:16406` `11:05:51 Consolidation complete: 3 merged, skipped (answer=4, u_spread=0, cross_pool=0)`
    show C2 took about 1 s with 4 answer-mismatch rejections.
  - `:55958` `23:53:29 Consolidation pass: 10258 episodes` and `:55959` `23:53:36 Consolidation complete: 43 merged, skipped (answer=45, u_spread=12, ...), 44 episodes removed`
    show C5 took about 7 s.
  - C1 (answer=1) and C3 (11 merged, answer=17, u_spread=1) match tab_consolidation_trajectory as well.
- **Verdict: CONFIRMED.**

---

## Summary table

| item | thesis value | verdict | correct value | evidence file:line |
|---|---|---|---|---|
| D1 | 289 @ 0.796 "eval fold 1500" vs 336 @ 0.759 | NOT FOUND (inconclusive) | 289/0.796 belongs to the stand-alone cycle_0 run fold (500/bench = 1500). 336/0.759 belongs to the main-run eval fold (300/bench = 900) | run_phase1a_ruc.log:71 (eval=500) vs full_run/experiment.log:47 (eval=300) |
| D2 | 1094 (ch5) / 1061 (table) | CORRECT-TO | 1061 (56/271/352/—/382) | full_run/experiment.log:5710, 19133, 33683, 60457; retroverify_cycle{1,2,3,5}.json:6 |
| D3 | 620 (ch5) / 650 (tables) | CORRECT-TO | 650 (C2 131, C3 220, C5 299). 1144/1468 are exits (promoted + TTL), not promotions | full_run/experiment.log:16404, 30357, 55957 |
| D4 | B1 TQA floor 0.27 | CORRECT-TO | 0.433. 0.27 is vanilla-FT (0.273) or LLM-judged RAG (0.271) | full_run/tab_sig_test.csv:33; tab_sig_test_v2.csv:32; baselines_v2/tqa_haiku_rescore.log:456 |
| D5 | 0.310 vs 0.210; off-arm abort C5 | CONFIRMED (EMs, DERIVED); abort cycle unverified | the notes say the Phase-1d abort was at C4 | analysis/phase1d_vs_phase1e_ruc.json:48-49; docs/PRODUCTION_END_TO_END.md:839 (notes) |
| D6 | App C "CES" 0.557/0.567/0.547/0.548/0.556 | CORRECT-TO (label) | the values are pooled eval-fold EM (fit input), not composite CES | full_run/cycle_5/equilibrium_fit.json:4-6; analysis/theorem_review.json:368 |
| D7 | 36–43% rate; admissions 2443/2477/2469/2484/2339 | CORRECT-TO | rate ≈ 52% (49.9–52.9%); C1/C2/C4 confirmed; C5 = 2343; C3 unverified | full_run/pi_store_trajectory.csv:14, 22, 34; theorem_review.json:26 |
| D8 | T 1.089/0.836/0.740/0.869 | CONFIRMED | — | full_run/experiment.log:18385, 32333, 57894 |
| D9 | MMLU 1.041/0.975/0.992/1.018/0.999 | CONFIRMED | — | full_run/experiment.log:3649, 16105, 29739, 44266, 55961 |
| D10 | 1 override on StrategyQA, u_pre 0.376 | NOT FOUND (inconclusive) | — | (only coincidental u_stored 0.376x values) |
| U1 | 3208 / 1846 / 1362 | CONFIRMED (3208); split NOT FOUND | — | baselines_v2/v2_features_extract.log:9 |
| U2 | AUROC 0.850 etc. | NOT FOUND | — | — |
| U3 | per-bench AUROC, n=11980, 0.771, −1.77 | NOT FOUND | — | — |
| U4 | 70/105, 306/478, 0.64 | NOT FOUND (swamped) | — | — |
| U5 | 0.9643 over 56 | NOT FOUND | — | — |
| U6 | ρ 0.5843 | NOTES-ONLY (hard-coded log string) | — | caem/verification/__init__.py:137, echoed at full_run/experiment.log:20 |
| U7 | ECE-before 0.2761/0.0508/0.0509/0.0451/0.0551 | CONFIRMED for C2/C3/C5; C0/C1 NOT FOUND | — | full_run/experiment.log:18385, 32333, 57894 |
| U8 | c_inf 0.548, a −0.018, k 0.350, R² 0.207 | CONFIRMED (k); rest NOT FOUND | — | full_run/cycle_5/equilibrium_fit.json:28 |
| U9 | 0.610/0.395/0.825; TQA-test 0.350 | CONFIRMED | — | full_run/experiment.log:1998, 44266; full_run/retention_baseline.json:4-5 |
| U10 | 0.803; chi² 0.232/56.860/76.891 | NOT FOUND (swamped) | — | — |
| U11 | ~53% C1 prune; id 13957 | id CONFIRMED; 53% NOT FOUND | main-run C1 prune is 13.0% (56/431) | full_run/dataset_splits.json:808; experiment.log:5710 |
| U12 | eval 75–95 min; 5.7–6.1 h | NOT FOUND (75–95 min not supported) | the FT→retroverify window is 5.75 h (C1) to 21.8 h (C5); 5.7–6.1 h = latency × 1500 | full_run/experiment.log:3659→5710, 16408→19133, 30361→33683, 55961→60458 |
| U13 | FEVER 145k, TQA 87k, CSQA ~10k, SQA dev 687 | CONFIRMED (FEVER, SQA); rest NOT FOUND | SQA 687 is the ChilleD *test* split | full_run/experiment.log:41, 66 |
| U14 | spread ≤ 1 pp over 5 C values | NOTES-ONLY | — | archive/branch_C_log_historical.md:284 |
| U15 | ~1 s, ~7 s; 4 answer-mismatch | CONFIRMED | — | full_run/experiment.log:16405-16406, 55958-55959 |

## Surprises
1. **D4 changes a conclusion.** ch5:473 says RUC-on TQA 0.310 "lifts above the B1 floor 0.27". B1 is actually 0.433,
   so the second ablation gate does not close as written.
2. **Appendix C "MMLU ρ" column is shifted and mislabelled.** It reads 0.635/0.635/0.595/0.605/0.609. The log has MMLU
   probe *accuracies* of C1 0.635, C2 0.595, C3 0.605, C4 0.6212 (`experiment.log:3649/16105/29739/44266`). So the
   column holds accuracies, not ratios, and is offset by one cycle from C2 onward. The thesis ratios are
   1.041/0.975/0.992/1.018/0.999.
3. **The C4 abort log line is misleading.** `experiment.log:44272` reads `Cycle 4 ABORTED (retention ratio 1.018 < 0.930)`.
   It prints the MMLU ratio, but the abort was triggered by triviaqa_test 0.8861. theorem_review.json:553 quotes it
   without comment.
4. **U6 (ρ = 0.5843) comes from code.** It exists only as text written into a log message in the code. No run computes it.
5. **The deferred-promotion store sizes differ from simple sums by 1–2.** C2 shows store 2948 where 2818 + 131 = 2949.
   C5 shows 10258 where 9961 + 299 = 10260. Promotion may dedup entries. This could also explain the C3 admissions
   mismatch (2469 vs 2471 from arithmetic).
6. **The script's keyword filter caused much of the noise.** The substring "store" matches every `u_stored`, "ces"
   matches "Choices", and "eval" matches the `eval.harness` logger name. The 15-file cap then hid the relevant files
   for D1, D10, U4, U5, U7, U10 and U12. A rerun with exact phrases would settle these. Suggested phrases:
   `override`, `n_stored`, `conformal`, `chi2`, `AUROC`, `Temperature scalar fitted` for C0/C1, and `EvalHarness done`
   plus timestamps for U12.
