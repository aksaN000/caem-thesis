# Numeric consistency audit: chapters/chapter_4.tex and chapters/chapter_5.tex

Repo root: `/home/user/caem-thesis/thesis_report`. Auto tables are under `figures/auto/`. Line numbers are 1-based. I re-read both locations for every mismatch listed below. Rounding rule: a value counts as consistent when it agrees after rounding to the precision written in the prose.

---

## (1) MISMATCHES

### A. Prose vs table/source: clear errors

| # | Prose location | Prose value (exact text) | Source location and value | Explanation |
|---|---|---|---|---|
| 1 | ch5:138 | "a clean exact-match win on TriviaQA ($+17.7$ percentage points at the best cycle, $+16.0$ at the trajectory-end cycle)" | tab_per_bench_trajectory.tex:19, TriviaQA ΔB1→C2 = `+0.200 (+68.3%)` | TriviaQA at the best cycle is +20.0 pp. The value +17.7 is FEVER's *relative* gain (tab_per_bench_trajectory.tex:17, `+0.080 (+17.7%)`). The +16.0 at C5 is correct. |
| 2 | ch5:473 | "recovery of TruthfulQA exact match toward the B1 zero-shot floor of $0.27$; the realised reading is $0.310$ ... lifting above the B1 floor" | B1 TruthfulQA EM = 0.433 (tab_per_bench_trajectory.tex:28; tab_sig_test.tex:37 `zero_shot truthfulqa ... 0.433`; tab_continual_retention.tex:19) | The B1 floor is 0.433, so 0.310 is still below it. The 0.27 is close to B6 vanilla_ft (0.273, tab_sig_test.tex:36). The same claim is repeated at ch6:75 ("from $0.210$ to $0.310$ above the B1 zero-shot floor"). |
| 3 | ch5:181 | "the override does not fire on any of the nine thousand evaluation queries" | tab_safety_override.tex:20–21: StrategyQA `1 & 0.056%`; "Eval subtotal & 9 000 & **1**" | The table records 1 eval-fold firing (u_pre = 0.376 < 0.38), plus 3 on the stream chunks. |
| 4 | ch5:163 | "fires on only seven of the nine thousand evaluation queries across the six cycles, a rate of $0.13$ percent" | Arithmetic: 7/9000 = 0.078 %. (The per-cycle rate is 2/1500 = 0.13 %, which matches the "0.0013" at ch5:239 and ch5:509.) | 0.13 % is the per-cycle C5 rate, not the 7-of-9000 rate. The correct figure is ≈0.08 %. |
| 5 | ch5:127 | "FEVER and TriviaQA exact match readings sit within a narrow band of two percentage points across the six cycles" | tab_per_bench_trajectory.tex:17,19: FEVER 0.477–0.533; TriviaQA 0.453–0.493. tab_continual_retention.tex:14–15 range column: `5.67pp`, `4.00pp` | The ranges are 5.7 pp and 4.0 pp, not two. Only the C0→C5 *net* change is 1.0 pp. |
| 6 | ch5:127 | "StrategyQA stabilises at $0.640$ from cycle one onward" | tab_per_bench_trajectory.tex:30: C1 0.647, C2 0.653, C3–C5 0.640 | The value is 0.640 only from cycle three onward. |
| 7 | ch5:127 | "Confident confabulation rises monotonically from $0.087$ at cycle zero to $0.269$ at cycle five" | tab_halluc_subtypes.tex:13: …C3 0.268, C4 0.270, C5 0.269 | C4→C5 is a decrease (0.270→0.269), so the series is not monotone. The endpoints are correct. |
| 8 | ch5:143 | "from a $14$ percentage-point gap at cycle one to a $24$-to-$28$ percentage-point gap from cycle two onward" and "widens monotonically across cycles" | tab_decision_distribution.tex:16–19: store-minus-defer EM is C1 0.74−0.60=14; **C2 0.74−0.44=30**; C3 0.75−0.47=28; C5 0.73−0.49=24 | C2's gap is 30 pp, outside "24-to-28". The sequence 14→30→28→24 does not widen monotonically. |
| 9 | ch5:147 | "The within-bench ordering $\pi_{\text{store}} > \mathrm{prec}(\text{defer}) > \mathrm{prec}(\text{discard})$ holds at every cell of the matrix" | tab_decision_matrix_trajectory.tex:13–24. Defer vs discard precision: C2 FEVER 344/763=0.451 < 58/103=0.563; C2 CSQA 34/53=0.642 < 1/1=1.0; C2 StrategyQA 125/198=0.631 < 1/1; C5 FEVER 474/906=0.523 < 10/12=0.833; C5 CSQA 32/64=0.500 < 11/17=0.647 | The defer > discard ordering fails in 5 of the 10 rows. The table caption (line 28) repeats the claim. |
| 10 | ch5:167 | "the Tier-2-versus-Tier-3 ordering ... and the ordering is preserved across the trajectory rather than reversed" | tab_per_tier_em.tex:18 C2 FEVER T2 51.6 < T3 54.5; :29 C5 StrategyQA T2 63.9 < T3 64.0 | The ordering reverses on FEVER at C2 and (barely) on StrategyQA at C5. The "four of five benchmarks at C0" part is correct. |
| 11 | ch5:216 | "with the $\tau_{\text{train}}$ filter removing between $35\%$ and $49\%$ of the wrong-stored entries" | tab_storage_pool_quality.tex:23–27. Wrong entries passing τ_train = n_pass·(1−π_SIL): C1 238/606, C2 275/621, C3 270/688, C4 278/660, C5 176/656 | The share removed is 56 %–73 %; the share retained is 27 %–44 %. Neither matches 35–49 %. The same figure appears in the table caption (line 32). |
| 12 | ch5:216 | "in every cycle except cycle two the em-rate weakly increases with the bucket index" | tab_u_stored_distribution.tex:13–25. Eval fold: C0 0.764→0.731 and 0.804→0.700 (falls); C3 0.738→0.706 (falls); C5 0.929→0.667 (falls). Stream chunks: monotone in every cycle, *including* C2 | On the eval fold (the subject of this paragraph) the claim fails at C0, C3 and C5. On the stream chunks it holds in all cycles, so "except cycle two" is wrong either way. The table caption gives a third version: "except the cycle-two and cycle-three rows". |
| 13 | ch5:301 | "$C=0.01$ ... with a tenfold tighter L2 penalty than the library default of $C = 1.0$" | Arithmetic: 1.0/0.01 = 100 | The penalty is a hundredfold tighter, not tenfold. The same error is in sweep_top.tex:12 ("tenfold tighter than the sklearn default"). |
| 14 | ch5:308 | "$p_{\text{entail}}$ lifts from $+0.067$ ... to $+0.285$ ... (an absolute movement of $0.218$, more than fivefold the cycle-zero magnitude)" | tab_calibration_trajectory.tex:38: 0.067 → 0.285, Δ 0.218 | 0.218/0.067 = 3.3× and 0.285/0.067 = 4.3×. Neither ratio exceeds five. |
| 15 | ch5:308 | "the FEVER and TriviaQA per-benchmark scalars sit an order of magnitude above the CommonsenseQA scalar at every cycle" | tab_calibration_trajectory.tex:26,28: C5 T_FEVER 4.972 vs T_CSQA 1.028 | At C5 FEVER is only 4.8× CSQA (C3: 6.5×), so not an order of magnitude at every cycle. |
| 16 | ch5:609 | "the per-cycle re-fit of the temperature scalar on the pre-routing confidence reaches the registered upper bound of the optimisation envelope from cycle one onward, so single-parameter temperature scaling on this surface is functionally inert" | tab_calibration_trajectory.tex:21: T = 20.086 (C0), **1.089, 0.836, 0.740, 0.869** (C1–C5); ch5:308 says it "drops by an order of magnitude under the cycle-one boundary refit, then stabilises in the band $[0.74, 0.87]$" | The ceiling (≈e³) is hit only at C0. The limitation paragraph contradicts both the table and ch5:308. |
| 17 | ch5:609 | "the cycle-boundary reconsideration pass had not been engaging the deferral buffer through cycles one through four ... The cycle-five boundary therefore exhibits a one-time backlog reconciliation" | tab_survival_distribution.tex:13–14: C2 promoted 131; C3 promoted 220, TTL-dropped 924. ch5:258 (prose) cites the same C2/C3 promotions. appendix_c.tex:81–82 lists C2/C3 promotions 131/220 | The tables and ch5:258 show the reconsider pass engaging at C2 and C3. |
| 18 | ch5:491 | "$+2.0\%$ exact match and $-9.3\%$ composite hallucination metric at the trajectory-end cycle" | tab_baseline_pooled.tex:24: B1 vs C5 `+0.011 (+2.1%)`, `-0.011 (-9.3%)` | The table gives +2.1 %, not +2.0 %. |
| 19 | ch5:317 | "a pooled ID precision of $0.796$ over the $289$ stored entries on the $1500$-sample ID evaluation fold" (and "On the held-out evaluation fold the storage class admits $63$, $145$, and $81$ entries") | (a) ch5:59 tab:benchmark-pools: Eval = 300 per benchmark, so the ID eval fold is 900; (b) cross_benchmark_summary.tex:9–11,16: n = 500 per bench, and the caption says "Each row reports the **calibration-fold** sample count"; (c) tab_storage_pool_quality.tex:15 / tab_pi_store_trajectory.tex:15: C0 *eval-fold* stored = **336** at π = **0.759** | The C0 "eval-fold" stored set is given as 289 @ 0.796 in prose (also ch5:294, 534, 575, 585; ch3:58,73,233; ch6:32; abstract:22) and as 336 @ 0.759 in two auto tables. The fold size is given as 1500 (prose), 300 per bench (pool table) and 500 per bench "calibration fold" (cross-bench caption). These cannot all describe the same fold. |
| 20 | ch5:250 (also ch5:254) | "the pooled $1\,094$ pruned entries" and "The cohort of $620$ entries promoted from the deferred buffer" | tab_em_conditional_breakdown.tex:15–21 gives 1094 / 620 (prunes 56/261/355/422; promotions 117/210/293). **But** tab_self_correction_check.tex:10–13 gives prunes 56/271/352/382 (sum 1061), tab_survival_distribution.tex:13–18 gives promotions 131/220/299 = **650**, and appendix_c.tex:80–84 gives prunes 56/271/352/382 and promotions 131/220/299 | The thesis gives two incompatible sets of per-cycle prune and promotion counts. ch5:258 states "The total of $650$ promotions" four paragraphs after ch5:250's 620, and ch5:254 uses 620 again. |
| 21 | ch4:163 | "The thirty-six features partition into three groups" | Enumerated in the same paragraph (and in alg:ruc lines 203–207): group 1 = token length 1 + interrogative one-hot 7 + negation 1 + temporal 1 + numerical 1 + entity count 1 + myth-regex 1 + three linguistic markers 3 = 16; group 2 = top-1 sim 1 + entity overlap 1 + three answer-type matches 3 + Wikidata 1 = 6; group 3 = FAISS stats 3 + rerank stats 3 + NLI stats 3 + p_IK 1 = 10 | The listed features add up to 32, not 36 (ch4:161 also says "thirty-six-dimensional"). Either four features are not described, or the count is wrong. |
| 22 | ch5:284 vs ch4:343 | ch5:284: "spans the boost layer's L2 regularisation strength $C$ at five values $\{0.010,\, 0.025,\, 0.050,\, 0.100,\, 1.000\}$" (also ch5:595, sweep_top.tex:12) | ch4:343: "registered in the cycle-zero calibration sweep against the alternatives $C\in\{0.01,\,0.10,\,1.00\}$" | The two chapters give different C grids (3 values vs 5) for the same registered sweep. |

### B. Prose vs table/source: rounding or wording slips (lower severity)

| # | Prose location | Prose value | Source / arithmetic | Explanation |
|---|---|---|---|---|
| 23 | ch4:355 (repeated ch5:301, ch5:595) | "the lower cut admits roughly four-and-a-half times as many correct memories" | ch4:355 itself: 306 correct (τ=0.60) vs 70 correct (τ=0.65) gives 306/70 = **4.37** | Correct memories are ≈4.4×. 4.5× (478/105 = 4.55) is the ratio of *total admissions*, not correct ones. |
| 24 | ch5:246 | "The pool itself grows from $431$ entries at the first retroactive pass to $10\,214$ entries at the fourth, a twenty-fold increase" | tab_self_correction_check.tex:10,13: 431 → 10,214, i.e. 23.7× | "Twenty-fold" understates the 23.7× growth (≈24-fold). |
| 25 | ch4:532 | "the pooled cal-fold pool-purity reading remains in the range $0.72$ to $0.75$, clearing the registered floor by eight to eleven percentage points" | tab_pi_store_trajectory.tex:15–20: [0.732, 0.746], so 9.2–10.6 pp. ch5:208, ch5:507 and tab_pi_store caption all say "nine to eleven" | Ch4 says "eight"; ch5 and the table say "nine". 0.732 rounds to 0.73, not 0.72. |
| 26 | ch4:625 | "the realised stream-chunk pool complement, which lies in the range $[0.246, 0.277]$" | tab_storage_pool_quality.tex:23–27: π_store 0.724–0.753 gives 1−π ∈ [0.247, 0.276]. From counts: 606/2456 = 0.2467, 688/2490 = 0.2763 | Both endpoints are off by 0.001 at the stated 3-dp precision. |
| 27 | ch5:469 | "StrategyQA shows the largest increase at $+8.1\%$ relative on the smallest test fold (the three hundred-sample StrategyQA panel ...)" | ch5:56 tab:benchmark-pools: StrategyQA test = 387, eval = 300 (all eval folds are 300) | The StrategyQA test fold has 387 samples, and on the 300-sample eval fold every benchmark is the same size. Either the fold or the count is wrong. |
| 28 | ch5:491 (vs tab_sig_test) | "on nine of forty ... on the exact-match axis and on twenty-one of forty contrasts on the composite-hallucination axis" | tab_sig_test.tex / _c2: the 5 vanilla_ft rows have CHM `nan`/`n/a` | Only 35 CHM contrasts are computable, so the CHM denominator should be 35, not 40. The win counts themselves are correct (see section 3). |
| 29 | ch5:120 | "cycle two carries the within-trajectory exact-match and composite-hallucination optima reported in \Cref{tab:per-bench-trajectory} and \Cref{tab:baseline-pooled}" | tab_per_bench_trajectory.tex:24: training-pooled EM peaks at **C1 0.584** (C2 0.581). ch5:138 itself says "peak of $0.584$ at cycle one" | This holds for the 5-bench pooled EM (tab_pooled_trajectory: C2 0.544 is the maximum) but not for the per-bench-trajectory pooled row it cites. |
| 30 | ch4:603 vs tab_theorem_receipts_summary.tex:15 | ch4:603: "a non-monotone but tightly bounded band of width below $0.03$" (ch5:232: width $0.014$) | tab_theorem_receipts_summary.tex:15: "band of width $\le 0.031$" | The receipts table's bound (0.031) is above ch4's "below 0.03", and both differ from the realised 0.014 (0.746−0.732). |
| 31 | ch4:582 / ch5:227 vs tab_theorem_receipts_summary.tex:13 | Prose: four observation points, so "three available transitions"; one sign-aligns, one is sub-noise, one is reverse-sign | Receipts table T3: "Sign alignment holds on three of four cal-fold transitions" | The prose has 3 transitions with 1 aligned. The table says 3 of 4 aligned. |

### C. Table-vs-table contradictions found while sourcing prose (not prose errors themselves)

- **tab_tier_trajectory.tex:21–25 (C5 panel b) vs tab_per_tier_em.tex:25–29 (C5).** The per-tier EMs differ:
  - FEVER T2 0.554 vs 50.0; T3 0.418 vs 45.9.
  - TriviaQA T2 0.514 vs 42.3; T3 0.420 vs 46.8.
  - CSQA T2 0.733 vs 74.5; T3 0.642 vs 58.5.
  - **TruthfulQA pooled EM 0.493 vs 30.0** (tab_per_bench_trajectory gives 0.300).
- **Relative ΔCHM, C2 vs B1.** tab_pooled_trajectory.tex:27 gives "$-10.8\%$ relative"; tab_baseline_pooled.tex:24 gives "(-11.0%)". Prose (ch5:491, abstract, ch6) uses −11.0 %.
- **tab_per_tier_em_capability.tex:17 note.** It says "the router still sends $>$50\% of CSQA queries to T3", but its own row (line 10) shows a CSQA tier share of 0%/80%/20% (T3 = 20 %).
- **tab_decision_matrix_trajectory.tex:2–3 comment.** The comment says "at C3 (joint best in-distribution)", but the table body and caption report C2 and C5.
- **CHM confidence intervals that exclude their own point estimate.** tab_sig_test.tex: fever/cot ΔCHM −0.042 with CI [−0.034, −0.012]; fever/flare −0.085 with [−0.074, −0.049]; csqa/fiveshot +0.007 with [+0.008, +0.025]; and others. tab_sig_test_c2 has the same pattern. The point estimates and CIs on the CHM axis come from different estimators or are misaligned.
- **Stream-chunk storage counts.** tab_storage_pool_quality.tex:23–27 n_stored (2456/2494/2490/2522/2380) vs appendix_c.tex:81–84 stream-chunk admissions (2443/2477/2469/2484/2339).
- **Outside scope but the same quantity (ch3:233).** "stream-chunk storage rate sits between $36\%$ and $43\%$ over cycles one to three". From the tables: 2456/4700 = 52 %, 2494/4700 = 53 %, 2490/4700 = 53 %; appendix_c.tex:89 says "approximately half".

---

## (2) UNSOURCED

These prose numbers appear in no auto table, inline table or other thesis location; they probably come straight from logs or artefacts.

**Chapter 4**
- ch4:49 worked example: top-passage entailment ">0.94", consistency ">0.85", QA relevance "≈0.69", storage score "≈0.51".
- ch4:51 worked example: u_token ≈0.02, top-passage ≈0.11, atomic ≈0.09, QA-rel ≈0.50, storage score ≈0.07.
- ch4:163: p_IK coefficient "$-1.77$".
- ch4:165: 3208 rows, 1846 helpful, 1362 hurt. The arithmetic is internally consistent, but no table holds these numbers.
- ch4:169: final-fit "C = 0.5", the C grid {0.005…5.0}, and the τ grid {0.20…0.75}.
- ch4:171: pooled OOF AUROC 0.850; per-fold sd 0.008; drift 0.011; 77.2 %; confusion counts 1554/292/923/439; recall 0.842; specificity 0.678; per-bench AUROC (OpenBookQA 0.686, TriviaQA 0.936, FEVER 0.718, CSQA 0.728, TruthfulQA 0.732, StrategyQA 0.852); probe n = 11,980 with AUROC 0.771. The per-bench values are repeated at ch5:469 and ch5:590, consistently. Note that figures/auto/tab_ruc_lobo.tex, which is not \input anywhere, reports a *different* classifier with pooled AUROC 0.594.
- ch4:338: per-bench AUROC regression of "ten to fifteen points" at α=1, and "approximately five hundred samples per benchmark" (the latter is consistent with the 500 cal fold).
- ch4:348 and ch5:317: split-conformal eval precision "fifteen to fifty percentage points" below target. Consistent between the two sites; no table.
- ch4:355: cal-fold sweep counts at τ=0.65 (70/105) and τ=0.60 (306/478), giving π ≈ 0.64. This conflicts in kind with tab_pi_store_trajectory.tex:15 (C0 cal-fold n=575, π=0.746). The "π_store ≈ 0.64" anchor (ch4:532, ch5:208) is labelled "pre-specified", so it may be an earlier fit, but the thesis does not reconcile the two.
- ch5:294: "$V_{\tau=0.65}$ ... admits roughly a factor of four fewer entries on the same [eval] fold at a marginally higher per-store precision". There is no eval-fold τ=0.65 count anywhere. ch5:595 says "The two top-tier cut-point variants **in the table** differ on $\tau_{\text{store}} \in \{0.60, 0.65\}$", but sweep_top.tex lists only the locked configuration, with no τ=0.65 row.
- ch5:317: split-conformal "0.9643 over 56 stored entries" (54/56 = 0.9643 is arithmetically consistent).
- ch5:336: Pearson ρ = 0.5843. Consistent with appendix_b.tex:31 and ch3:209.
- ch5:595: "spread in calibration-fold per-store precision across these five [C values] is bounded above by approximately one percentage point".
- ch5:254: "approximately one second" and "seven seconds" wall-time; "four sibling candidate clusters ... rejected for answer-mismatch" (consistent with the table's C2 answer-mismatch of 4).
- ch5:473: "the classifier-off trajectory was aborted at cycle five under the retention guard".
- ch5:43: the CSQA train pool of "approximately ten thousand samples after dedup" (also in the ch5:59 caption, with FEVER 145k and TriviaQA 87k; table caption only).

---

## (3) Counts

**Numbers checked and found consistent: 238.** Main groups:

- **ch4 RUC arithmetic.** 1846+1362=3208; 1846/1362=1.36; 1554+292=1846; 923+439=1362; recall 0.842; specificity 0.678; (1554+923)/3208 = 77.2 %. (7 checks)
- **ch4:355.** 70/105 ≈ 67 %; 306/478 ≈ 64 %. (2)
- **ch4:532/534.** π≈0.64; per-bench min 0.66 (TriviaQA C5); prune 13 %→4 %; FEVER eval band [0.614, 0.634]; 0.649; pooled band [0.732, 0.746]. (6)
- **ch4:625.** 7 dispositions; 1500 queries per cycle; 0.36 = 1−0.64; 0.037 ≈ 382/10214. (4)
- **ch4:647/649/675.** TruthfulQA store precision 0.522; prune rates 13/9/7/4 %; worst-probe ratios 1.006/0.949/0.976/0.886/0.958; X sequence; 4/5 = 0.80; "within 0.07 of floor". (14)
- **ch5:43/59 pool budget.** Matches tab:benchmark-pools. (8)
- **ch5:116.** 0.541; 0.517; −2.4 pp; 0.135; 0.109; 19 %; min 0.107 at C2; MMLU [0.975, 1.041]; 0.886 at C4; commit fraction 0.80. (10)
- **ch5:120.** 0.593; 0.710; Δ0.117; VER lift 0.274; 0.886; C4 dip 0.013; 0.958; 0.710 peak. (8)
- **ch5:127 (other values).** CSQA 0.763→0.717; TruthfulQA 0.380→0.300; false refusal 0.305→0.064; fact. fabric. 0.364→0.303; logic. fabric. 0.283→0.231; off-topic/template zero; 19 %. (11)
- **ch5:138 (other values).** 0.483; 0.584 at C1; 0.549; 0.143; 0.101 at C2; 29.4 %; 20.3 %; TruthfulQA below B1 at every cycle; StrategyQA CHM below B1; +16.0 pp; FEVER CHM −35.8 %. (11)
- **ch5:143 (other values).** Store 42→59 %; defer 12→29 %; discard 25→3 %; abstain 21→10 %; store EM 0.73–0.75; abstain EM 0.38→0.17; strict ordering at every cycle; 14 pp at C1. (8)
- **ch5:151.** FEVER 64.3 %→5.3 %; CSQA ≤1 %; TruthfulQA [23 %, 33 %]. (3)
- **ch5:163 (other values).** 7 of 9000; T2 share 48.7/50.2/55.2 %; T2 EM 0.602; T3 0.482; 12 pp; 0.537; T3 ≈0.50; EM = 1.0 on hits. (8)
- **ch5:167.** C0 ordering on 4 of 5 benchmarks. (1)
- **ch5:174.** 1500 queries; 13–15 s band; batch size 32. (3)
- **ch5:190.** FEVER/TriviaQA −1.0 pp; CSQA 4.7 pp; StrategyQA +3.0; TruthfulQA −8.0; TruthfulQA CHM 0.150→0.104; one rollback at C4. (7)
- **ch5:194.** 107; 5006; losses 0.258/0.171/0.109/0.070; 0.886; 0.091. (8)
- **ch5:208.** [0.732, 0.746]; 9–11 pp; eval [0.693, 0.759]; 5–12 pp; within 5 pp; 0.6637 (≈0.664 in tab_decision_matrix; 2.4 pp above floor); FEVER 0.634/0.630/0.617/0.614/0.649; TriviaQA/CSQA band [0.704, 0.797]. (14)
- **ch5:212.** 431; 12,175; 28×; [2214, 2484]; buffer 0→4284. (5)
- **ch5:216 (other values).** τ_train 0.70; 336–566; 69.3–75.9 %; 17.3 %; 45–50 %; 35.9 %; π_SIL [0.798, 0.818]; +6 to +8 pp; [0.724, 0.753]; eval lift +2 to +6; 107; 5006. (12)
- **ch5:227/232.** Four observation points; trajectory 0.746→0.740→0.736→0.746→0.732; width 0.014; 1500; (1,1,1,0,1); 4/5; 431→12,175; within 0.07; π≈0.73. (9)
- **ch5:239.** 0.36; 7 of 9000; CHM₁ = 0; T2 CHM 0.101–0.111; T3 CHM 0.113–0.158; ordering CHM₁ < CHM₂ < CHM₃ at every cycle; τ₁ = 0.0013. (7)
- **ch5:246 (other values).** 13/9/7/4 % vs 13.0/9.2/6.6/3.7. (4)
- **ch5:250 vs tab_em_conditional.** 1094; 38.4 %; ~22 %; 1.75×; 620; 55.6 %; ~78 %; 345, plus the internal sums and per-cycle percentages. (8)
- **ch5:254.** 0.92; top 20; 0.15; 57 merges / 58 removed; C2 3 merged / 4 mismatches; first u-spread at C3; 42+1 clusters, 43 merges / 44 removed; twelve-fold (1→12); ~10k episodes; cross-pool 0. (10)
- **ch5:258/262.** 131; 220; 924; 299; 1169; 650; 1255; buffer arithmetic. (8)
- **ch5:294/317/341/585.** 289 = 63+145+81; pooled π 0.796 (weighted from per-bench); per-bench precision 0.794/0.821/0.753; 0.522; d +0.635; BA ≈ 0.673 = Φ(0.635/√2); transfer d +0.173; gaps 0.282/0.320/0.258; poisoning 0.206/0.179/0.247 (= 1 − store precision); ">3×" (3.18). (14)
- **ch5:308 (other values).** T = 20.086 ≈ e³; band [0.74, 0.87]; ECE [0.041, 0.049]; intercept [−0.63, −0.52]; CSQA floor 0.380→0.491→0.569; Δp_entail 0.218; q_a,rel 0.491→≈0.26, Δ0.226; other signals |Δ| ≤ 0.07. (9)
- **ch5:394/414/491 sig-test tallies.** C5: EM 9 wins / 3 defeats, CHM 21 / 3; C2: EM 14 / 3, CHM 20 / 4; "five additional"; "one fewer"; defeats confined to TruthfulQA vs zero-shot/CoT/FLARE/semantic entropy. All recounted from tab_sig_test(_c2). (8)
- **ch5:426.** B2 0.500 vs B1 0.507. (2)
- **ch5:467–473 RUC ablation.** 0.474→0.532; +5.8 pp; +12.2 %; 0.121→0.115; −5.2 %; T3 88.9→55.1 % (−33.8); CSQA +14.0 / −34.4 %; TruthfulQA +10.0; TriviaQA +2.7 / −2.0 %; StrategyQA +2.3 / +8.1 %; FEVER flat 0.503, +2.5 %; TruthfulQA +2.9 %; AUROCs; T3 collapse 21.7–59.0; TruthfulQA 99→40 %; CSQA 57.7→20.3; FEVER 92.3→70.7 (smallest); zero per-bench regression; 0.210→0.310. The "on" arm also matches the main-trajectory C3 EM/CHM and the tier shares. (20)
- **ch5:491 (other values).** 0.483→0.581, +20.3 %; 0.143→0.101, −29.4 %; +13.7 %; −20.3 %; +7.4 %; −11.0 %; −9.3 %; StrategyQA 0.099 vs 0.120, −17.5 %; 0.135→0.109, −19 %; EM band [0.517, 0.544]; FLARE edge 0.008; 37 % (36.8); CAEM beats all baselines on CHM at both cycles and on EM at C2. (15)
- **Hypothesis table (ch5:506–510).** +0.635; 0.673; band and sequence; n ∈ [575, 1001]; 4/5; 0.109; 7/7; 0.36; 0.0013; 7 of 9000; latency [13.6, 14.7]. (11)
- **ch5:575/590/615/616.** 0.796 / 289; 15–50 pp; AUROCs; 0.522 vs 0.75–0.82; 500-sample fold; fifty-sample floor. (7)

Totals: **31 prose mismatches** (22 clear in A, 9 minor in B), **7 table-vs-table contradictions** in C, and about 22 unsourced items or item groups in section 2.
