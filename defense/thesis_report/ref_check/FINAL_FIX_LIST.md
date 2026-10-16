# Final fix list: thesis numbers vs data (for review before any edit)

Status as of 2026-10-15. No thesis file has been changed for any item below.

## What the evidence covers

- **Local repo:** `branch/ruc-classifier` at `da05919`; clean working tree; identical to GitHub.
- **Tables:** all 60 auto tables were regenerated from the **local `outputs/`**, which is authoritative. Every number
  matches apart from `tab_safety_override`. Captions, layout and hand-added rows or columns (B1 ref, Δ columns,
  panel b) differ, but every shared cell agrees. Reports: `tables_vs_local_outputs.md`; subtype table rechecked
  separately.
- **Logs:** the local log search covered `experiment.log`, `run.log` and the artefacts (`log_resolution` below).
- **Line numbers** refer to the files at `da05919` plus this branch.
- **Evidence tags:**
  - `T` = generated table
  - `L` = run log or artefact line
  - `C` = code
  - `A` = arithmetic
  - `G` = git history (stale value from an older table or run)

Each fix will be logged before → after in `fixes_applied.md`. Overstated claims are re-worded; no data value is
changed except where the evidence column gives the correct one.

---

## Part 1: settled; ready to apply on approval

### 1a. Stale baselines (June B6/B7 fix not carried into these files)  [G: `git show a240921^:…/tab_sig_test.tex`]

| # | Where | Now | Change to | Evidence |
|---|---|---|---|---|
| 1 | appendix_d:155, :98; appendix_e:235; README:172 | EM 13/3, CHM 23/4 (C5); 17/2, 26/4 (C2); "four CHM defeats (B1, B2, B7, B8)" | 9/3, 21/3 (C5); 14/3, 20/4 (C2); three C5 CHM defeats (B1, B2, B8) | T tab_sig_test(_c2) |
| 2 | appendix_d:52 (B6 TriviaQA row) | 0.190, +0.263, χ² 56.86 | regenerate the row from the current tab_sig_test (0.227, +0.227) | T |
| 3 | appendix_d:53, :39 (B7 FLARE StrategyQA) | 0.250, +0.390, χ² 76.89 | 0.600, +0.040, not significant | T |
| 4 | appendix_d:84 (B6 CSQA "+0.717 … largest lift") | +0.717 [+0.663, +0.767] | −0.063 [−0.120, −0.010]; drop "largest lift" | T |
| 5 | appendix_d:108–118 (TruthfulQA Holm table) | FLARE n.s.; two defeats | FLARE a significant defeat; three defeats | T |
| 6 | appendix_d:130–140 (FEVER Holm table) | FLARE significant win; B6 significant ~8 pp defeat | neither is significant (+0.040; −0.067) | T |
| 7 | ch5:491 | "+2.0%" vs B1 at C5; "of forty" CHM contrasts | +2.1%; "of thirty-five" (B6 CHM cells are n/a) | T tab_baseline_pooled, tab_sig_test |
| 8 | README:49–53, :121, :161–170 | pre-fix baseline table (B6 0.325/0.144, B7 0.292/0.153, B4 0.461), "8/8", "beats every baseline at both cycles", −10.8% | copy the current tab_baseline_pooled; 7/8 at C5 (FLARE leads EM by 0.008); −11.0% | T |

### 1b. TruthfulQA raw vs judged EM  [L: `experiment_summary.csv` raw 0.4933 vs judged 0.300]

| # | Where | Now | Change to | Evidence |
|---|---|---|---|---|
| 9 | tab_tier_trajectory panel (b) (hand-built) | C5 per-tier EMs, TruthfulQA pooled 0.493 | rebuild panel (b) from tab_per_tier_em (TQA 32.1/25.5/30.0, FEVER 50.0/45.9, …), or drop panel (b) | T tab_per_tier_em |
| 10 | README:204–212 | copy of panel (b) | copy of tab_per_tier_em C5 | T |

### 1c. Results that changed meaning

| # | Where | Now | Change to | Evidence |
|---|---|---|---|---|
| 11 | ch5:473, ch6:75, ch6:102 | "B1 zero-shot floor of 0.27"; 0.310 "lifts above the B1 floor"; the second ablation gate "closes" | B1 TruthfulQA is 0.433. 0.310 recovers toward it (from 0.210) but stays below it, so the second gate as stated does **not** close. The gate verdict text needs rewording. | T tab_per_bench_trajectory:28; L tab_sig_test.csv:33 (`zero_shot,truthfulqa,…,0.4333`). No B1 0.27 exists anywhere. |
| 12 | ch5:181 | override "does not fire on any of the nine thousand" | fires once in 9,000 (StrategyQA, u_pre 0.376), as tab_safety_override shows. Wording waits on #U1 | T (hand-built; see U1) |
| 13 | ch5:250, :254 | 1,094 prunes, 620 promotions | 1,061 prunes (56/271/352/–/382; C4 rolled back) and 650 promotions (131/220/–/–/299) | L experiment.log:5710, 19133, 33683, 60457, 16404, 30357, 55957; retroverify_cycle{1,2,3,5}.json |
| 14 | tab_em_conditional_breakdown (source of 1,094/620) | offline reconstruction | relabel as a reconstruction, or regenerate its totals to the log counts (1,061/650) | L as above |

### 1d. Wrong fold or measure named

| # | Where | Now | Change to | Evidence |
|---|---|---|---|---|
| 15 | abstract:22; ch3:58, :73, :233; ch5:294, :317, :534, :575, :585; ch6:32 | "0.796 over 289 entries" on the "evaluation fold" | keep the number, but name the fold correctly: the **cycle-zero validation fold** (500 per benchmark, 1,500 on the training panel). The main-run trajectory evaluation fold at C0 is 336 at 0.759 (300 per benchmark). | L run_phase1a_ruc.log:71 (`eval=500`) vs full_run/experiment.log:47 (`eval=300`); C aggregate_section_5_7.py:387 |
| 16 | cross_benchmark_summary caption | calls n=500 the "calibration-fold" count | "cycle-zero validation fold" | same |
| 17 | ch5:163 | 7 of 9,000 = "0.13 percent" | ≈0.08 percent (0.13% is the C5-only rate, 2/1,500) | A |
| 18 | ch5:216 + caption | τ_train removes "35% to 49%" of wrong entries | removes 56–73% of wrong stream entries (35.9–49.8% is the eval-fold pass-rate column) | T tab_storage_pool_quality |
| 19 | ch4:355, ch5:301, ch5:595 | "four-and-a-half times as many correct memories"; π ≈ 0.64 anchor | 4.4× (306/70; 4.5 is total admissions 478/105). Label these counts as the pre-Phase-1e calibration sweep (478 at 0.640); the Phase 1e C0 cal fold is 575 at 0.746. | G 04fe2e7 tab_pi_store_trajectory; T |
| 20 | ch5:469 | StrategyQA "smallest test fold", "three hundred-sample" | test fold 387 (smallest); evaluation fold 300 (all benchmarks) | L experiment.log:66 |
| 21 | ch5:120 | C2 carries the optimum in tab:per-bench-trajectory | C2 is the optimum of the 5-benchmark pooled EM (tab:pooled-trajectory); the training-pooled EM peaks at C1 (0.584) | T |
| 22 | appendix_c:104–108 "CES" column | header "CES" | relabel as "pooled eval-fold EM". The values 0.557/0.567/0.547/0.548/0.556 are the series the equilibrium fit uses. | L full_run/cycle_5/equilibrium_fit.json:4–6 |
| 23 | appendix_c:104–107 "MMLU ρ" column | 0.635/0.635/0.595/0.605/0.609, shifted by one cycle and not ratios | MMLU probe accuracy C1–C5 = 0.635/0.595/0.605/0.621/0.609, or the ratios 1.041/0.975/0.992/1.018/0.999 (labelled as such) | L experiment.log:3649, 16105, 29739, 44266, 55961 |
| 24 | appendix_e:166 | evaluation pass "75–95 minutes" | no log supports it. tab_tier_latency's 5.7–6.1 h is mean latency × 1,500, not a measured time. The measured fine-tune-to-retroverify window is 5 h 45 m (C1), 9 h 44 m (C2), 13 h 54 m (C3) and 21 h 50 m (C5). Use the measured windows and say what they include. | L experiment.log:3659→5710, 16408→19133, 30361→33683, 55961→60458 |
| 25 | appendix_e:17/21 | StrategyQA "dev split 687" | 687 is the ChilleD **test** split (300 eval + 387 test) | L experiment.log:66 |
| 26 | appendix_g:58 vs :60 | 1,500 vs ≈3,000 labelled batch | :60 is the production deployment batch (docs/PRODUCTION_RUNBOOK.md:233); say so | notes |

### 1e. Text from earlier runs or plans (Phase 1a/1d) that was never updated  [G]

| # | Where | Now | Change to | Evidence |
|---|---|---|---|---|
| 27 | abstract:22; ch5:376 | BCa bootstrap, 10,000 replicates | percentile bootstrap, 1,000 replicates (as in appendix D) | C eval/metrics.py:850; scripts/baseline_sig_tests_v2.py:142, :237 |
| 28 | appendix_a:325 (+:150) | LoRA trained with 8-bit AdamW | 32-bit AdamW on the LoRA path (the 8-bit flag applies only to full fine-tuning) | C caem/training/self_improvement.py:1035–1046 |
| 29 | ch5:609 | temperature at the envelope ceiling "from cycle one onward" | only at C0 (20.086); C1–C5 = 1.089/0.836/0.740/0.869 | L experiment.log:18385, 32333, 57894; T |
| 30 | ch5:609 | reconsider pass idle C1–C4, C5 backlog | promotions at C2 (131), C3 (220) and C5 (299) | L as #13 |
| 31 | ch3:233 | stream-chunk storage rate 36–43% | ≈50–53% of 4,700 (49.9–52.9%) | L pi_store_trajectory.csv:14, 22, 34 |
| 32 | appendix_a:275, :318 | cold start ≈3,000 entries | 431 entries (≈3,000 was the seeding plan) | T tab_memory_store_trajectory; L |
| 33 | appendix_e:168 | downgraded setup "preserving effective batch 32" | production effective batch is 16 (4×4); fix the downgraded figures to match | C config; appendix_a:152 |
| 34 | appendix_b:16; ch5:294, :595 | "ten-variant table" / "two top-tier variants in the table" | tab:sweep-top is now the locked-parameter table; remove the references to the variant table | G 449dacb |
| 35 | ch1:58 | "almost every query falls to the retrieval tier at cycle zero" | about half (51.3% Tier 3 at C0) | T tab_tier_trajectory |
| 36 | tab_per_tier_em_capability note | ">50% of CSQA to T3" | 20% | T own row |
| 37 | tab_decision_matrix_trajectory comment | "at C3 (joint best)" | C2 and C5 | T |
| 38 | figures/auto/tab_ruc_lobo.tex | legacy pre-v2 RUC table (AUROC 0.53–0.61); not `\input` anywhere | delete the file, or mark it as legacy | G 478251a |

### 1f. Writing, arithmetic and overstated claims  [T or A]

| # | Where | Now | Change to |
|---|---|---|---|
| 39 | ch5:138 | TriviaQA "+17.7 pp" at best cycle | +20.0 pp |
| 40 | ch5:127 | FEVER/TriviaQA "within two pp"; StrategyQA 0.640 "from cycle one"; confabulation "rises monotonically" | within 6 pp (net C0→C5 change about 1 pp); from cycle three; rises from 0.087 to ≈0.27 and plateaus |
| 41 | ch5:143 | gap "24 to 28 pp from cycle two", "widens monotonically" | 14 pp at C1, then 24–30 pp; not monotone |
| 42 | ch5:147 + caption | store > defer > discard "holds at every cell" | holds in 5 of 10 rows; name the exceptions |
| 43 | ch5:167 | T2 > T3 "preserved across the trajectory" | holds except C2 FEVER and C5 StrategyQA |
| 44 | ch5:216 | EM rises with u_stored bucket "except cycle two" | eval fold: not monotone at C0, C3 and C5; stream chunks: monotone at every cycle |
| 45 | ch5:301; sweep_top.tex:12 | "tenfold tighter" | hundredfold (1.0/0.01) |
| 46 | ch5:308 | "more than fivefold"; temperature ratio "order of magnitude at every cycle" | more than threefold; 4.8–10.5× |
| 47 | ch5:246 | "twenty-fold" | about 24-fold (431 → 10,214) |
| 48 | ch4:163 | lists 32 features but says "thirty-six" | keep 36 (feature_spec.json) and complete the list: add log_pageviews_max and the six answer-type features |
| 49 | ch4:343 | C grid {0.01, 0.10, 1.00} | {0.010, 0.025, 0.050, 0.100, 1.000} |
| 50 | ch4:532 | "eight to eleven" pp | nine to eleven |
| 51 | ch4:625 | [0.246, 0.277] | [0.247, 0.276] |
| 52 | tab_theorem_receipts_summary | band "≤ 0.031"; "three of four transitions" | 0.014; recompute the sign alignment from tab_pi_store_trajectory |
| 53 | ch4:603 | band "below 0.03" | 0.014 |
| 54 | tab_pooled_trajectory:27 | −10.8% | −11.0% (computed from unrounded values; matches tab_baseline_pooled) |
| 55 | appendix_c:94 | temperature band [0.74, 1.09] "from cycle two"; "four CONTINUE verdicts" | [0.74, 0.87]; three |
| 56 | appendix_c memory table | columns don't reconcile | add the consolidation-removal column (3/11/44). A 1–2 entry residual at C2/C5 is promotion dedup (store 2,948 vs 2,818 + 131); footnote it. |
| 57 | appendix_c admissions vs tab_storage_pool_quality | 2,443… vs 2,456… | footnote: the gate-pass count is higher because the novelty filter (`store.py:301`) rejects near-duplicates. C5 admissions = 2,343 (12,175 − 9,832), not 2,339. |
| 58 | appendix_d:39 | B8 and CAEM "agree on every TruthfulQA sample" | delete (Δ = −0.007) |
| 59 | appendix_d:96; appendix_f:113 | Holm multiplier "m − k" | m − k + 1 (the table itself uses ×8 at rank 1) |
| 60 | appendix_e:168 | 12–16 GPU-days | recompute from the stated factor, or remove |
| 61 | appendix_e:211 | expected pool 12,170 | 12,175 |
| 62 | sig-test captions / ch5:376 | — | add one sentence: the CHM interval resamples only the baseline's subtype rates (CAEM's CHM is held fixed), so it can exclude the point difference; it is approximate (C `_approx_chm_ci`) |

---

## Part 2: one exact lookup needed from the local agent before deciding

Run each from the repo root and paste the output. All read-only.

| # | Item | Command |
|---|---|---|
| U1 | Did the safety override fire on the eval fold (StrategyQA, u_pre 0.376)? The table was built by hand from u_pre < 0.38. | `grep -rn -i "override\|safety_u_pre\|u_pre_min" outputs/full_run/experiment.log \| head -40` and `python - <<'E'`<br>`import json,glob`<br>`for f in sorted(glob.glob('outputs/full_run/eval/*_cycle[0-5].json')):`<br>`  d=json.load(open(f)); s=d.get('samples',d.get('per_sample',[]))`<br>`  low=[x.get('u_pre') for x in s if isinstance(x,dict) and x.get('u_pre') is not None and x['u_pre']<0.38]`<br>`  print(f,len(s),len(low),low[:5])`<br>`E` |
| U2 | RUC v2 fit statistics (3,208 rows / 1,846 helpful / 1,362 hurt; OOF AUROC 0.850; confusion 1554/292/923/439; per-benchmark AUROC; probe n=11,980, AUROC 0.771; coef −1.77; C = 0.5) | `ls caem/ruc/v2/; grep -rn -i "auroc\|confusion\|helpful\|n_rows\|11980\|coef" caem/ruc/v2/*.json caem/ruc/v2/*.log outputs/baselines_v2/*.log \| head -60` |
| U3 | conformal "0.9643 over 56 stored entries" | `grep -rn "0\.964\|conformal" outputs/cycle_0 outputs/calibration outputs/analysis \| head -20` |
| U4 | Qwen-judge Pearson ρ = 0.5843 (the run only repeats a code string, `caem/verification/__init__.py:137`) | `grep -rn -i "pearson\|rho" outputs/calibration/ \| head -20; cat outputs/calibration/qwen_judge_platt.json \| head -40` |
| U5 | appendix D χ² values (0.232, 56.86, 76.89, 0.010) and ROUGE-rule 0.803 | `grep -rln "56\.86\|76\.89\|0\.803" outputs archive docs \| head` |
| U6 | cal-fold τ sweep counts 70/105 and 306/478 | `grep -rn "478\b" outputs/cycle_0 outputs/calibration outputs/analysis \| head -20` |
| U7 | ECE-before at C0 and C1 (0.2761, 0.0508) | `grep -n "Temperature scalar fitted\|ECE" outputs/full_run/experiment.log \| head -20` |
| U8 | RUC ablation off arm: aborted at C4 or C5? | `grep -rn -i "abort" outputs/phase1d_full_run/*.log outputs/analysis/phase1d_vs_phase1e_ruc.json \| head -20` |
| U9 | appendix G "≈53% stale-calibration cycle-one prune" (the main run's C1 prune is 13.0%, 56/431) | `grep -rn "53\b\|0\.53" outputs/phase1d_full_run/*.log \| grep -i prune \| head` |

## Part 3: no source found anywhere; your decision

Infrastructure and cost figures can't come from a log: 40 days, 960 GPU-h, $672, 150 GB disk, 90 GB RAM, the
64 GB / 21M-passage index, step runtimes, 8–10 GPU-h baselines, the 2.2–2.6× downgraded penalty, the
"twelve-month compute envelope", and the README's "325 pages". The arithmetic is consistent (40×24 = 960;
960×0.70 = 672). **Keep them if they come from your Vast.ai billing and notes; otherwise mark them "approximately".**
The equilibrium-fit constants (c∞ 0.548, a −0.018, R² 0.207) have only k confirmed. Check
`outputs/full_run/cycle_5/equilibrium_fit.json` before keeping the others.

## Part 4: code and data consistency (no thesis numbers change)

- **Generator scripts that would re-introduce errors if run.** Fix them, so the tables can be regenerated safely:
  - `scripts/aggregate_section_5_6.py`: stale hard-coded T1 band, "≤0.031" and "three of four".
  - `scripts/aggregate_section_5_7.py`: caption "twenty-five-variant" and "tenfold".
  - `scripts/aggregate_section_5_4.py`: counts the `escalated` flag instead of u_pre < 0.38, and its caption
    hard-codes "never fires".
- **Logging quirk (code only).** `experiment.log:44272` prints the MMLU ratio (1.018) in the C4 abort message,
  although the abort was triggered by triviaqa_test at 0.886. Optional fix in the runner's log line.
- **Drive sync (step 7 of SYNC_PLAN.md).**
  - Local → Drive: the 10 B6/B8 files that differ and the 25 local-only files. Old versions move to
    `caem-phase1a/_superseded_2026-10/`.
  - Drive → local: copy the 73 Drive-only files into a new local folder, `outputs/_from_drive_phase1e/`, never
    over `outputs/`. These are the per-cycle snapshots, `run.log`, `dataset_splits.json` and
    `mmlu_baseline.json`, the B8 `_with_chm` files and the June rerun logs.
  - Update both Drive READMEs.
