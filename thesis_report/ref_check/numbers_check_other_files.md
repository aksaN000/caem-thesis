# Numeric consistency audit: abstract, ch1, ch2, ch3, ch6, appendices A–G, README

Paths are relative to `/home/user/caem-thesis/thesis_report/` unless prefixed `../` (repo root).
I re-read every MISMATCH below at both locations. Rounding rule: values count as consistent when they agree at the precision written in the prose.

---

## (1) MISMATCHES

### A. Headline numbers (abstract / ch6 / README vs ch5 tables)

1. **The two source tables give different full-panel CHM deltas (affects the abstract and ch6).**
   - Prose: `core/abstract.tex:22` "+7.4 percent exact match and $-11.0$ percent composite hallucination"; `chapters/chapter_6.tex:27` (rendered line 54) "$+7.4\%$ exact match and $-11.0\%$ composite hallucination metric"; `../README.md:49` "ΔCHM = −11.0% relative".
   - Sources: `figures/auto/tab_baseline_pooled.tex:24` gives "$-0.013\ (-11.0\%)$". `figures/auto/tab_pooled_trajectory.tex:27` gives "$\Delta$CHM $=-0.013$ ($-10.8\%$ relative)". `../README.md:121` also says "−10.8% relative".
   - Two auto tables report the same C2-vs-B1 quantity differently. From the rounded values, 0.107/0.120 gives −10.8%. The README contradicts itself (line 49 vs line 121).

2. **The 0.796 / 289 "evaluation-fold" storage-precision anchor does not match the per-cycle π_store table.**
   - Prose: `core/abstract.tex:22` "evaluation-fold storage precision $0.796$ over $289$ stored entries"; `chapters/chapter_3.tex:58` "$\pi_{\text{store}} = 0.796$ over $289$ stored entries on the held-out training-panel evaluation fold"; `chapters/chapter_3.tex:73` (NFR2 row); `chapters/chapter_6.tex:32` "evaluation-fold precision anchor sits at $0.796$ over $289$ stored entries".
   - Source: `figures/auto/tab_pi_store_trajectory.tex:15` gives "C0 & $575$ & $0.746$ & $336$ & $0.759$". At C0 the eval fold has n=336 at π=0.759, and the cal fold has n=575 at π=0.746. `figures/auto/tab_storage_pool_quality.tex:15` agrees (C0 eval 336 / 0.759).
   - The 289/0.796 figure matches only `figures/auto/cross_benchmark_summary.tex:9-11` (63+145+81=289, weighted precision 0.796). That table's caption calls these columns the "calibration-fold sample count" with n=500 per benchmark, not the eval fold. ch5 prose (`chapters/chapter_5.tex:294,317,534`) repeats 0.796/289 as eval-fold.

3. **The "1500-sample training-panel evaluation fold" contradicts the eval-fold size elsewhere.**
   - Prose: `chapters/chapter_3.tex:233` "$289$ stored entries over the $1500$-sample training-panel evaluation fold, a storage rate near $19\%$".
   - Sources: `chapters/chapter_5.tex:52-54` (tab:benchmark-pools) gives Eval = $300$ per training benchmark, so the training-panel eval fold is 900. `appendix/appendix_e.tex:21` says "three hundred samples per benchmark". The 1500 figure is the calibration fold (`figures/auto/composite_weights.tex:24` "$n = 1500$ training-panel calibration fold").
   - The arithmetic 289/1500 = 19.3% holds, but the fold is mislabelled. On a 900-sample fold the rate would be 32%.

4. **Chapter 3's stream-chunk storage rate disagrees with the appendix and the storage-pool table.**
   - Prose: `chapters/chapter_3.tex:233` "the realised per-cycle stream-chunk storage rate sits between $36\%$ and $43\%$ over cycles one to three".
   - Sources: `appendix/appendix_c.tex:158-160,165` lists stream-chunk admissions of 2443/2477/2469 against a 4,700-sample budget, so "the storage gate admits approximately half" (52–53%). `figures/auto/tab_storage_pool_quality.tex:23-25` gives n_stored 2456/2494/2490 of 4700, which is 52–53%. No table contains 36–43%.

5. **Chapter 6 says the RUC ablation lifted TruthfulQA "above the B1 zero-shot floor", but 0.310 is below B1's 0.433.**
   - Prose: `chapters/chapter_6.tex:102` "a TruthfulQA exact-match recovery from $0.210$ to $0.310$ above the B1 zero-shot floor closes the second".
   - Sources: `figures/auto/tab_per_bench_trajectory.tex:28` gives the B1 TruthfulQA EM as 0.433. `figures/auto/tab_sig_test.tex:37` gives zero_shot truthfulqa EM_base 0.433. Separately, ch5 states a different floor: `chapters/chapter_5.tex:473` "toward the B1 zero-shot floor of $0.27$".

6. **The README's eight-baseline table disagrees with `tab:baseline-pooled`.**
   - `../README.md:161` B1 vs C5: "+0.010 (+2.0%)". Table `tab_baseline_pooled.tex:24`: "$+0.011\ (+2.1\%)$". ch5:491 also says +2.0%, which disagrees with its own table.
   - `../README.md:162` B2 C5 "+3.4%" vs `:25` "+3.5%".
   - `../README.md:163` B3 C5 "+12.9%" vs `:26` "+13.0%".
   - `../README.md:164` B4: EM 0.461, +0.083 (+17.9%), +0.056 (+12.1%). Table `:27`: 0.464, +0.080 (+17.2%), +0.053 (+11.5%).
   - `../README.md:165` B5 C5 "+27.5%" vs `:28` "+27.6%".
   - `../README.md:166` B6 vanilla FT: 0.325 / 0.144, +0.219 (+67.2%) … Table `:29`: 0.490 / 0.127, +0.054 (+11.0%) … Every cell differs.
   - `../README.md:167` B7 FLARE: 0.292 / 0.153, +0.252 (+86.3%) … Table `:30`: 0.525 / 0.173, +0.019 (+3.5%), C5 −0.008. Every cell differs.
   - `../README.md:168` B8: +7.7%, +0.012 (+2.4%). Table `:31`: +7.8%, +0.013 (+2.5%).
   - `../README.md:170` "CAEM C5 beats 8/8 baselines". Table `:33`: "CAEM C5 beats $7/8$".
   - `../README.md:53` "CAEM beats every baseline on both the exact-match and the composite-hallucination axes … at both reference cycles". At C5 FLARE leads on EM (Δ −0.008, `tab_baseline_pooled.tex:30`). The abstract (`core/abstract.tex:22`) and ch6 correctly say parity at C5.

7. **The significance win/loss tallies in the appendices and README do not match the sig-test tables.** I counted the tables myself; the abstract, ch5 and ch6 counts agree with them.
   - Counted from `figures/auto/tab_sig_test.tex` (C5): EM 9 wins / 3 defeats; CHM 21 wins / 3 defeats.
   - Counted from `figures/auto/tab_sig_test_c2.tex` (C2): EM 14 / 3; CHM 20 / 4.
   - `appendix/appendix_d.tex:155` "thirteen exact-match wins and three defeats and twenty-three composite-hallucination wins and four defeats; at the within-trajectory best cycle … seventeen exact-match wins and two defeats and twenty-six composite-hallucination wins and four defeats". It also says "The four composite-hallucination defeats at both anchors … (B1, B2, B7 FLARE, B8)". The C5 CHM defeats in the table are cot, semantic_entropy and zero_shot only (3); FLARE truthfulqa CHM is "+0.009 … (B) / -".
   - `appendix/appendix_e.tex:235` gives the same 13/3, 23/4, 17/2, 26/4.
   - `../README.md:172` "13 of 40 … 23 of 40 … 17 of 40 and 26 of 40 … four significant composite-hallucination defeats".
   - `appendix/appendix_d.tex:98` says TruthfulQA "carries the four composite-hallucination defeats" at the cycle-five anchor. The table shows 3.

8. **The README's per-tier C5 table conflicts with `tab:per-tier-em` and with TruthfulQA EM everywhere else.**
   - `../README.md:204-212` is labelled "canonical reading from `tab:per-tier-em`". Its TruthfulQA row reads "64.3% / 0.591 | 35.3% / 0.311 | 0.493". The FEVER, TriviaQA and CommonsenseQA per-tier EMs are 0.554/0.418, 0.514/0.420 and 0.733/0.642.
   - Source: `figures/auto/tab_per_tier_em.tex:25-28` gives C5 TruthfulQA T2 32.1, T3 25.5, pooled 30.0; FEVER 50.0/45.9; TriviaQA 42.3/46.8; CommonsenseQA 74.5/58.5. TruthfulQA C5 pooled EM is 0.300 in `tab_per_bench_trajectory.tex:28` and `tab_sig_test.tex`.
   - The README values come from `tab_tier_trajectory.tex:21-25` (panel b), which already disagrees with `tab_per_tier_em`. Tier shares agree; the EMs do not.

### B. Hyperparameters and run settings

9. **Appendix A says the LoRA run used 8-bit AdamW, but the code and the body chapters say 32-bit AdamW.**
   - Prose: `appendix/appendix_a.tex:325` "at learning rate two times ten to the minus four under an eight-bit AdamW optimiser". Also `:150` gives the rationale "Fits the bounded GPU envelope at batch $4$".
   - Sources: `chapters/chapter_5.tex:104` "The shipped optimiser is a thirty-two-bit … AdamW …; the eight-bit-moment optimiser variant is registered but inactive in the low-rank-adapter configuration". `chapters/chapter_4.tex:475` says "thirty-two-bit decoupled-weight-decay AdamW". `../caem/training/self_improvement.py:1037-1046` skips 8-bit AdamW when LoRA is active (`want_8bit = False`).
   - The config field `use_8bit_adamw = True` (`../caem/config.py:636`) is quoted correctly, but it has no effect on the LoRA path.

10. **The bootstrap method and replicate count disagree between the abstract/ch5 and the appendices.**
    - `core/abstract.tex:22` "bias-corrected-and-accelerated bootstrap intervals". `chapters/chapter_5.tex:376` "bias-corrected and accelerated (BCa) … $B = 10\,000$ replicates".
    - `appendix/appendix_d.tex:68,70` "standard percentile method … Bootstrap replicates: one thousand", and says percentile is preferred over BCa. The caption at `appendix/appendix_d.tex:87` says "percentile method with one thousand replicates". `appendix/appendix_f.tex:99` says "The percentile-method confidence interval".

11. **Appendix E's downgraded envelope does not preserve the production effective batch.**
    - `appendix/appendix_e.tex:168` "batch size drops from four to two, the gradient accumulation step count rises from four to sixteen (preserving the effective batch size at thirty-two)".
    - The production effective batch is 4×4 = 16 (`appendix/appendix_a.tex:152` "Holds the effective batch at $16$", `:244`; `../caem/config.py:736,741`; `../README.md:99`). Two times sixteen is 32, which doubles the effective batch rather than preserving it.

### C. Appendix C (per-cycle dictionary)

12. **The temperature band is wrong.**
    - `appendix/appendix_c.tex:94` "the scalar stabilises in the band $[0.74, 1.09]$ from cycle two onward".
    - The appendix's own table (`:106-109`: C2 0.836, C3 0.740, C5 0.869) and `chapters/chapter_5.tex:308` ("stabilises in the band $[0.74, 0.87]$ from cycle two onward") give [0.74, 0.87]. The value 1.089 is C1.

13. **The CONTINUE count is wrong.**
    - `appendix/appendix_c.tex:172` "four CONTINUE verdicts at cycles one through three".
    - The table at `:182-184` lists 3 CONTINUE rows (C1–C3).

14. **The equilibrium table's CES column disagrees with `tab:ces-caem-trajectory`.**
    - `appendix/appendix_c.tex:182-186` gives CES 0.557, 0.567, 0.547, 0.548, 0.556 for C1–C5.
    - `figures/auto/tab_ces_caem_trajectory.tex:27-31` gives 0.678, 0.704, 0.703, 0.690, 0.710. ch5:120 also says CES 0.593 to 0.710.

15. **The "MMLU ρ" column is shifted by one cycle for C2–C4.**
    - `appendix/appendix_c.tex:183-185` gives C2 0.635, C3 0.595, C4 0.605. Against the pristine MMLU of 0.610 (`:215`), these imply ratios of 1.041, 0.975 and 0.992.
    - `figures/auto/tab_pooled_trajectory.tex:22-24` and `tab_stochastic_equilibrium` give C2 0.975, C3 0.992, C4 1.018. The expected accuracies would be 0.595, 0.605 and 0.621.
    - C1 (0.635 → 1.041) and C5 (0.609 → 0.999) agree. The column is also labelled "ρ" but holds accuracies.

16. **The memory table's columns do not sum to the net change.**
    - `appendix/appendix_c.tex:146` says the table shows "the four mechanisms that move the pool size".
    - Admissions − prunes + promotions gives:
      - C2: 2477 − 271 + 131 = 2337, but the table shows +2333.
      - C3: 2469 − 352 + 220 = 2337, but the table shows +2326.
      - C5: 2339 − 382 + 299 = 2256, but the table shows +2214.
    - Subtracting the consolidation removals (3, 11 and 44 in `tab_consolidation_trajectory`) closes C3 exactly, leaves C2 off by 1 and C5 off by 2. That column is not in the table.

17. **Stream-chunk admissions disagree with the storage-pool table.**
    - `appendix/appendix_c.tex:158-162` gives 2443, 2477, 2469, 2484, 2339.
    - `figures/auto/tab_storage_pool_quality.tex:23-27` (panel b, stream-chunk n_stored) gives 2456, 2494, 2490, 2522, 2380.
    - The gap is 13–41 per cycle. A novelty filter between storage-gate pass and admission might explain it, but neither location says so.

### D. Appendix D (significance details)

18. **The B6 TriviaQA contrast does not match `tab:sig-test`.**
    - `appendix/appendix_d.tex:52` gives "$0.453$ & $0.190$ & $+0.263$".
    - `figures/auto/tab_sig_test.tex:20` (vanilla_ft triviaqa) gives EM_base 0.227, Δ +0.227.

19. **The B7 FLARE StrategyQA contrast does not match `tab:sig-test`.**
    - `appendix/appendix_d.tex:53` gives "$0.640$ & $0.250$ & $+0.390$ & $76.891$ & $< 10^{-12}$". The prose at `:39` calls it "a similarly large gap".
    - `figures/auto/tab_sig_test.tex:41` (flare strategyqa) gives EM_base 0.600, Δ +0.040, "- / *". The EM difference is not significant.

20. **The B8 TruthfulQA claim contradicts its own Δ.**
    - `appendix/appendix_d.tex:39` "the two systems agree on every TruthfulQA correctness pattern at the per-sample level".
    - The same row (`:55`) and `tab_sig_test.tex:35` give Δ = −0.007 (about 2 samples of 300). If they agreed on every sample, b = c = 0 and Δ would be 0.

21. **The bootstrap table's B6 CommonsenseQA row does not match `tab:sig-test`.**
    - `appendix/appendix_d.tex:84` gives "B6 vanilla-FT $\cdot$ CommonsenseQA & $+0.717$ & $[+0.663,\, +0.767]$ & supported (largest panel lift)".
    - `figures/auto/tab_sig_test.tex:28` gives vanilla_ft commonsense_qa Δ −0.063, CI [−0.120, −0.010], "- / -". The value 0.717 is CAEM's own EM. The largest C5 lift in the panel is fiveshot_cot TriviaQA, +0.317.

22. **The TruthfulQA Holm table disagrees with `tab:sig-test`.**
    - `appendix/appendix_d.tex:115` gives B7 FLARE raw p 0.923, not significant. The caption at `:118` says "the two CAEM defeats on TruthfulQA exact match (B1 zero-shot and B2 chain-of-thought)".
    - `figures/auto/tab_sig_test.tex:30,33,37` mark cot, flare and zero_shot as significant TruthfulQA EM defeats "(B)" (3 defeats; flare Δ −0.150, CI [−0.220, −0.080]).

23. **The FEVER Holm table disagrees with `tab:sig-test`, and creates a non-TruthfulQA defeat.**
    - `appendix/appendix_d.tex:130` gives B7 FLARE as significant ("CAEM win by a large exact-match margin"). `:131,140` give B6 as significant ("CAEM loss by approximately eight percentage points; … the registered defeat").
    - `figures/auto/tab_sig_test.tex:9` gives flare fever Δ +0.040, EM "-". `:12` gives vanilla_ft fever Δ −0.067 (about 7 pp, not about 8) with "- / -", which is not marked (B).
    - A FEVER defeat would also contradict `core/abstract.tex:22` "every defeat confined to the misconception-bait TruthfulQA surface".

### E. Appendix E (reproducibility)

24. **The expected memory pool size differs from the trajectory table.**
    - `appendix/appendix_e.tex:211` "Expected: $12\,170$ entries".
    - `figures/auto/tab_memory_store_trajectory.tex:16` and `appendix/appendix_c.tex:162` give 12,175.

25. **The checklist expects the wrong win/loss tallies.**
    - `appendix/appendix_e.tex:235` gives 13/3, 23/4, 17/2, 26/4.
    - The tables give 9/3, 21/3, 14/3, 20/4 (see item 7).

26. **The held-out evaluation time is far shorter than the latency table implies.**
    - `appendix/appendix_e.tex:166` "The held-out evaluation pass consumes approximately seventy-five to ninety-five minutes across the five-benchmark panel".
    - `figures/auto/tab_tier_latency.tex:22-27` gives a per-cycle wall time of 5.68–6.11 h over the 1,500 evaluation queries (13.6–14.7 s/query × 1500 ≈ 5.7–6.1 h).
    - Also, the stage times listed in `:166` sum to about 5.9–7.7 h, not the stated "five to seven hours".

27. **The downgraded-GPU estimate does not follow from the numbers given.**
    - `appendix/appendix_e.tex:168` "approximately two-point-two to two-point-six times the production envelope, so the full trajectory on a downgraded GPU consumes approximately twelve to sixteen GPU-days".
    - Production is 5–7 h/cycle × 6 cycles ≈ 1.25–1.75 days, which scales to about 2.8–4.6 days. The full programme of 960 GPU-h (`:164`) scales to about 88–104 days. Neither gives 12–16.

### F. Other appendices and chapter 1

28. **The cold-start memory size disagrees with the trajectory table.**
    - `appendix/appendix_a.tex:275` "Seeds the episodic memory with approximately three thousand verified entries … the EM-prune step removes fewer than ten percent". `:318` says "the cold-start memory store (approximately three thousand seeded episodes)".
    - `figures/auto/tab_memory_store_trajectory.tex:11` and `appendix/appendix_c.tex:157` give C0 = 431. `chapters/chapter_5.tex:609` says "cold-start memory after the pre-step-7 hygiene pass contains $431$". A prune of under 10% cannot take about 3000 down to 431 unless a separate, unquantified hygiene pass did it.

29. **The "ten-variant table" does not exist.**
    - `appendix/appendix_b.tex:16` "The headline ten-variant table appears in \Cref{tab:sweep-top}".
    - `figures/auto/sweep_top.tex:9-15` is a 7-row locked-parameter table with no sweep variants.

30. **Appendix G gives two different labelled-batch sizes in consecutive paragraphs.**
    - `appendix/appendix_g.tex:58` "Approximately fifteen hundred labelled samples per cycle (five hundred per training benchmark)".
    - `appendix/appendix_g.tex:60` "the cycle boundary's labelled batch is a stratified sample of approximately three thousand entries". The research fold is 1500 everywhere else (`appendix/appendix_a.tex:276`, `appendix/appendix_g.tex:111`).

31. **The Holm formula contradicts the Holm table.**
    - `appendix/appendix_f.tex:113` "multiplies the $k$-th smallest by $m - k$", with one-indexed k implied by "k-th smallest". `appendix/appendix_d.tex:96` gives the same formula, though it adds the correct equivalent "$m+1-i$".
    - The Holm table (`appendix/appendix_d.tex:108`) multiplies rank 1 by 8 (2.4e-5 → 1.9e-4), so the factor is m − k + 1.

32. **Chapter 1's tier-share claim contradicts the tier table.**
    - `chapters/chapter_1.tex:58` "At cycle zero almost every query falls to the retrieval-augmented tier".
    - `figures/auto/tab_tier_trajectory.tex:13` gives C0 Tier 3 share 51.3% (Tier 2 48.7%). `../README.md:202` agrees: "routes 51.3% of queries to the retrieval-augmented tier".

### Out of scope but found while checking sources (ch5 and auto tables)

- `chapters/chapter_5.tex:250` and `tab_em_conditional_breakdown` give 620 deferred-buffer promotions and 1,094 prunes. `chapters/chapter_5.tex:258`, `tab_survival_distribution` and `appendix/appendix_c.tex` give 650 promotions and 56+271+352+382 = 1,061 prunes. Per-cycle prunes also differ (261/355/422 vs 271/352/382).
- `chapters/chapter_5.tex:163` "seven of the nine thousand … a rate of $0.13$ percent". 7/9000 is 0.078%. 0.13% is the C5-only rate (2/1500).
- `chapters/chapter_5.tex:469,590` give RUC per-benchmark AUROC 0.718–0.936 (these match `../caem/ruc/v2/feature_spec.json`). `figures/auto/tab_ruc_lobo.tex` gives a different, legacy 6-signal classifier with AUROC 0.53–0.61, and is still `\input`-able.
- The TruthfulQA row of `tab_tier_trajectory` panel b (0.493) and its other per-tier EMs conflict with `tab_per_tier_em`.

---

## (2) UNSOURCED (found nowhere else in the thesis or tables)

- `chapters/chapter_1.tex:387`: "twelve-month compute envelope".
- `appendix/appendix_a.tex:323` and `appendix/appendix_e.tex:164`: about 40 days, about 960 GPU-h, about $672, about $0.70/GPU-h, 150 GB disk, about 90 GB host RAM. The arithmetic is internally consistent (40×24 = 960; 960×0.70 = 672). Repeated in `../README.md:141`.
- `appendix/appendix_a.tex:14,331`: 64 GB index, 21M passages, about 90 GB page cache. Writeup host RTX 3060 12 GB / 16 GB RAM.
- `appendix/appendix_a.tex:338,340,342`: runtimes (1 h, 3.5 h, 2 min, 2 min).
- `appendix/appendix_c.tex:105-109`: ECE-before values (0.2761, 0.0508, 0.0509, 0.0451, 0.0551).
- `appendix/appendix_c.tex:193`: c∞ = 0.548, a = −0.018, k = 0.350, R² = 0.207.
- `appendix/appendix_c.tex:208-212`: epoch-1 and epoch-2 losses.
- `appendix/appendix_c.tex:198,215`: TriviaQA-test 0.350/0.395 and pristine anchors 0.610 / 0.395 / 0.825. These are internally consistent with n = 200 per probe and with the 0.886 ratio.
- `appendix/appendix_d.tex:26`: ROUGE-rule zero-shot TruthfulQA 0.803.
- `appendix/appendix_d.tex:50-55,108-137`: χ² values (0.232, 56.860, 76.891, 0.010) and all raw and Holm p-values.
- `appendix/appendix_e.tex:17,21`: shuffle seeds 42/43; StrategyQA dev split 687 (300 + 387 = 687 is consistent).
- `appendix/appendix_e.tex:49-97,166`: all step wall-times; baselines at 8–10 GPU-hours; the 2.2–2.6× penalty.
- `appendix/appendix_g.tex:147`: "approximately fifty-three percent" stale-calibration cycle-one prune.
- `appendix/appendix_g.tex:181`: FEVER id 13957 example.
- `appendix/appendix_g.tex:183`: "nineteen-hundred-or-more" inflated fold count.
- `appendix/appendix_a.tex:109,151,155`: the "re-tuned from 0.60", "reduced from 16" and "reduced from 0.75" histories. The 0.75→0.70 history is echoed in `appendix/appendix_g.tex:167`.
- `../README.md:19,112`: "325 pages".
- `../README.md:91`: h_norm boost coefficient about −5e-4.

---

## (3) CHECKED CONSISTENT

About 520 values were checked and found consistent after rounding to the precision written in the prose:

- Abstract: about 17.
- ch6: about 28.
- ch3: about 14.
- ch1: about 12.
- ch2: about 7.
- Appendix A: about 75. All 60+ config-registry and hyperparameter values match `../caem/config.py` exactly. τ_RUC 0.375, C 0.5 and the 36 features match `feature_spec.json`.
- Appendix B: 5.
- Appendix C: about 50.
- Appendix D: about 15, including the Holm arithmetic.
- Appendix E: about 22.
- Appendix F: about 16.
- Appendix G: about 10.
- README: about 250 table cells. The pooled, per-benchmark, RUC, tier-share, SIL, memory, gate and hypothesis tables match their sources.

Headline items confirmed:
- Training-panel +20.3% / −29.4% at C2 and +13.7% / −20.3% at C5.
- StrategyQA 0.099 vs 0.120 (−17.5%).
- Sig-test tallies in the abstract, ch5 and ch6 (14/3 at C2, 9/3 at C5; CHM 20/4 and 21/3), recounted from the tables.
- Cohen's d +0.635 / +0.173, and "more than a factor of three" over 0.20.
- Tier shares 48.7 → 50.2 with a peak of 55.2; Tier 3 share 44.7–55.1.
- RUC +5.8 pp, 0.210 → 0.310, +10 pp TruthfulQA.
- τ 0.60 / 0.45, α_shrink 0.6, LoRA r 32 / α 64, ρ_min 0.93, 100 sweep variants (4×5×5), and the 32-token pre-route decode.
- About 2% trainable parameters (about 60M / 3.1B).
- Validation-gate values (0.282 / 0.320 / 0.258 and 0.206 / 0.179 / 0.247).
- π_store band [0.732, 0.746], and 9–11 pp over 0.64.
- Memory 431 → 12,175 (28×), commit fraction 4/5, prune rates 13 / 9 / 7 / 4%.
