# Repository clean-up tracker (plug-and-play phase)

Every code, script, data or table problem found during the final drafting pass (chapters 1 to 6, appendices, from the first session onward) is listed here, so the
clean-up phase after the chapter pass can work through it as a checklist. Add a row whenever a new one is
found. Status: OPEN, FIXED (in code), DOCUMENTED (left as is, explained), or WORKAROUND (thesis uses a
corrected copy; generator still stale).

## Goal of the clean-up phase
1. One environment file (`requirements.txt` or `environment.yml`) with the versions of appendix E.
2. One documented command per stage (data → cold start → main run → baselines → rescore → tables → thesis).
3. Every committed table regenerated exactly by its generator (no hand edits); a check script that rebuilds
   all tables into a temp copy and diffs them against the committed ones.
4. Superseded scripts and outputs moved to `archive/` with a README saying what replaced them.
5. A top-level README "How to reproduce the thesis" that a stranger can follow.

## Code bugs and wiring faults
| # | Where | Problem | Status |
|---|---|---|---|
| C1 | `scripts/run_experiment.py` (early-stop gate) | Storage signal read the evaluation-pass storage rate (always 0); MMLU signal compared raw accuracy with the 0.93 ratio floor. Gate certain to fire at C5. | FIXED 2026-10-16 (review/ch3); replay `scripts/replay_early_stop_gate.py` |
| C2 | `scripts/run_tier1_routing_probe.py` L239–275 | `top1_is_source_rate` compares the sample-list index with the memory entry id (different id spaces), so it is always 0. Retrieved question text matches. | OPEN (field not used in thesis) |
| C3 | `scripts/baseline_sig_tests_v2.py` `_approx_chm_ci` | CHM "significance" is an unpaired approximation (CAEM fixed, baseline subtype rates resampled), no Holm, floor 0.01; thesis/appendix D described paired Wilcoxon + Holm. | FIXED 2026-10-17: `scripts/paired_baseline_tests.py` (Wilcoxon + Holm per benchmark, paired bootstrap; reproduces EM 9/3, 14/3); v2 script superseded, archive in clean-up |
| C4 | `caem/verification/verifier.py` early-exit log line | Log says "route to Tier 3 regen"; code only marks DISCARD. | OPEN (cosmetic) |
| C5 | `caem/training/self_improvement.py` docstring L48 | Says "8-bit AdamW + L2 anchor on adapter"; LoRA path uses 32-bit AdamW and λ_anc = 0. | OPEN (comment) |
| C6 | `caem/verification/__init__.py` / config comments | Many comments describe the Qwen long-hypothesis judge as active; it was inactive in the main run (log L20). | OPEN (comments) |
| C7 | `caem/config.py` per-benchmark u_pre temperature | Fit hits the e^3 cap on FEVER/TriviaQA; safety override nearly inert. | DOCUMENTED (thesis limitation) |
| C8 | `caem/verification/verifier.py` semantic entropy | Signal runtime-disabled (`disable_h_norm`) and replaced by a 0.5 sentinel; the "Pearson 0" reading was measured on the sentinel. Verifier still computes/records it. (ch2 session) | DOCUMENTED; remove dead path in clean-up |
| C9 | `caem/verification/verifier.py::_score_atomic` + `atomic_min_tokens = 12` | Atomic-fact signal equalled p_ground_mean on all 14,951 samples (99.0% length-gated, 1.0% NO_FACTS); composite carries a duplicate signal. (ch4 session) | DOCUMENTED; decide drop-or-apply-to-chains |
| C10 | `caem/memory/deferred.py` docstring | Says `promote_threshold` defaults to `store_threshold (0.65)`; the store threshold is 0.60. | OPEN (comment) |
| C11 | Verifier diagnostics `alias_overlap`, `entity_head_consistency`, `p_contra` | Computed and stored on every sample but not used by the composite (p_contra structurally 0). | OPEN (remove or flag as diagnostics) |
| C12 | SIL training targets (`self_improvement.py`, stored reasoning chains) | Chains written under the retrieval prompt teach Tier 2 to cite a non-existent "context" (3% → 93–99% after C1). Needs a preprocessing step that strips/rewrites context references. | OPEN (future work) |
| C13 | `caem/config.py` deprecated v1 fields | `general_data_ratio`, `general_data_size`, `l2_lambda` (inactive on LoRA path), `use_8bit_adamw` (ignored on LoRA path), `faiss`/`ivf_pq` defaults never reached. | OPEN (prune or document) |

## Data and provenance problems
| # | Where | Problem | Status |
|---|---|---|---|
| D1 | `outputs/baselines_phase1e/chm_comparison.json` (semantic_entropy) | B8 CHM comes from the OLD broken SE run (`baselines_phase1e_pre_se_rerun_backup`, rescored 2026-05-29); corrected run (2026-06-06) never rescored (u_stored None). | DOCUMENTED 2026-10-16 (no budget): B8 CHM reported as unavailable; sig/pooled tables set to n/a by hand; rescore = future work |
| D2 | `outputs/cycle_0/eval/*_cycle0.json` | `u_stored` from the legacy weighted-sum bootstrap composite, not the locked composite (sweep README finding 1). | WORKAROUND (`scripts/recompute_validation_gate.py`) |
| D3 | `outputs/full_run/pi_store_trajectory.csv` row C0 | "weight_validation" reading (validation fold, five benchmarks, legacy composite) mixed into a calibration-fold series. | WORKAROUND (table hand-corrected) |
| D4 | `outputs/baselines_phase1e/vanilla_ft/eval/*_cycle3_with_chm.json` | Same CHM as cycle 5 (e.g. FEVER 0.1308 both): c3 file appears to be a copy of c5. | OPEN (check) |
| D5 | Cycle-zero sweep numbers "306/478", "70/105" | Not produced by any variant; no source. | DOCUMENTED (removed from thesis) |
| D6 | `SIL_POISON_DRIFT_FINDINGS.md` | Static diagnosis; the causal ablation it proposes was never run. | DOCUMENTED |
| D8 | Dropped benchmarks (HotpotQA, NQ) | Receipts only on Drive `9_archive/` (`hotpotqa_cycle0.json`, `natural_questions_cycle0.json`); not in the repo outputs. (ch1 session) | DOCUMENTED |
| D9 | Cold-start memory | 603 verified seed answers, 172 wrong ones removed with gold labels (`em_pruning_report.json`); HotpotQA entries stripped (`memory_store_pre_v21_strip.meta`). Pipeline step not a single script. (ch4 session) | OPEN (document as one stage) |
| D7 | Drive layout | Appendix A names old subfolders (`v2_1_phase1e_main/`, `pre_main_snapshot/`) that predate the numbered layout. | OPEN (appendix session) |

## Stale or hand-edited generated tables (generator does not reproduce the committed file)
| # | Table | Generator | Hand edit | Status |
|---|---|---|---|---|
| T1 | `tab_theorem_receipts_summary.tex` | `scripts/aggregate_section_5_6.py` | rewritten 2026-10 | WORKAROUND |
| T2 | `tab_pi_store_trajectory.tex` | aggregate_section_5_* | C0 cal cell + caption | WORKAROUND |
| T3 | `cohen_d.tex` | aggregate_section_5_7 | u_stored row + caption | WORKAROUND |
| T4 | `sweep_top.tex` | aggregate_section_5_* | rationale cells + caption | WORKAROUND |
| T5 | `tab_validation_gate.tex`, `cross_benchmark_summary.tex` | aggregate_section_5_7 | superseded by `*_locked.tex` from `scripts/recompute_validation_gate.py` | WORKAROUND (old files no longer input); 2026-10-17 cross-bench merged into tab_validation_gate_locked, file removed |
| T6 | `tab_tier_baseline_cost.tex` | placeholder stub | never input | OPEN (delete) |
| T7 | Ch5 table captions (about 20 files) | various | shortened by hand in the ch5 caption pass | WORKAROUND |
| T8 | Uninput auto tables (tab_headline, tab_purity, tab_calibration, tab_continual, ... see review_ch5_part2_ch6.md §17) | various | -- | OPEN (archive or delete) |

## Scripts that are dangerous or misleading to run
| # | Script | Problem | Status |
|---|---|---|---|
| S1 | `scripts/aggregate_section_5_*` | Run on import and overwrite committed tables. | OPEN (guard with `if __name__ == "__main__"`) |
| S2 | No `requirements.txt` / `pyproject.toml` | Versions only reconstructed in appendix E. | OPEN |
| S4 | Repository tags/branches | Only two Git tags; no tag per cycle as chapter 3 once claimed. (ch3 session) | DOCUMENTED |
| S5 | `outputs/` is git-ignored | Evidence JSONs needed by the thesis are force-added one by one (`git add -f`). | OPEN (decide what evidence lives in the repo vs Drive) |
| S3 | `scripts/baseline_sig_tests.py` docstring | Talks about "CAEM cycle 10" and B7 `ewc_only_ft`; real run pins C5/C2 and B7 is FLARE. | OPEN |

## New scripts added during the drafting pass (keep)
- `scripts/replay_early_stop_gate.py` → `outputs/analysis/early_stop_gate_replay.json`
- `scripts/build_ch5_evidence_tables.py` → `tab_gate_quality`, `tab_sweep_variants`, `tab_headline_steps`
- `scripts/recompute_validation_gate.py` → `tab_validation_gate_locked`, `tab_cross_bench_locked`
- `scripts/rerun_cycle0_sweep.py` → `outputs/cycle_0/sweep_v2_1/`

## Found in the chapter 5 session (2026-10-16, continued)
| # | Where | Problem | Status |
|---|---|---|---|
| D10 | `fu2025collapse` bib key | First author is Yi (bib author field correct); key name misleads (text had "Fu et al."). | FIXED in text; rename key in clean-up |
| T9 | `scripts/build_ch5_evidence_tables.py` constants | Headline, subtype and baseline values are copied from committed tables (POOLED_EM, CONFAB, BASELINES); B8 CHM is the old-run value (D1). | OPEN (read from source JSON in clean-up) |
| C14 | `caem/verification/verifier.py::_compute_q_a_relevance` | bge-reranker via sentence-transformers already returns a sigmoid probability; the verifier applies a second sigmoid, so `q_a_relevance` ∈ [0.50, 0.731]. Off-topic CHM subtype (threshold 0.50) can never fire (0 for every system). Composite unaffected (isotonic per signal). | DOCUMENTED (ch5 text, 2026-10-17). Inverting the sigmoid does NOT give a usable detector: reranker scores short correct answers as low as wrong ones (TriviaQA C5 93% vs 90% below 0.5), so the subtype would track answer length. Clean-up: drop the second sigmoid in code; redesign or drop the off-topic subtype in a future metric version |

## Found in the chapter 5 read-through (2026-10-17)
| # | Where | Problem | Status |
|---|---|---|---|
| D11 | CHM correctness label on TruthfulQA (`eval/metrics.py` reads `em`; baseline `*_with_chm.json` files) | CHM uses the harness `em` field, which for TruthfulQA is ROUGE-L > 0.15, while EM uses the Haiku judge. Judged label changes pooled CHM: B1 0.120→0.137, C0 0.135→0.134, C2 0.107→0.110, C5 0.109→0.117; TQA CHM defeats vs B1/B2 vanish. FLARE `_with_chm` file lacks the judged field (it is in the raw file). | FIXED in thesis 2026-10-17 (author: "truthfulqa always works with haiku, any other thing used is stale method"): `scripts/recompute_chm_judged.py` → `outputs/analysis/chm_judged.json`; `scripts/patch_chm_judged_tables.py` patches the hand-laid tables; builder + paired tests read it. Clean-up: make `eval/metrics.py` take the judged label for TruthfulQA so every generator agrees |
| T10 | `tab_memory_dynamics.tex` | Hand-merged from tab_memory_store_trajectory + tab_survival_distribution + tab_consolidation_trajectory (values unchanged) | WORKAROUND |
| T11 | ch5 body tables (per_bench, halluc_subtypes, decision_distribution, tier_trajectory, sil_training, storage_pool_quality, gate_quality, pi_store, theorem_receipts, cohen_d, baseline_pooled, ruc_ablation) | Layout hand-edited: \small, no \resizebox, notes moved below tabular, shorter headers, relative-% cells dropped | WORKAROUND (generators still emit old layout) |
| S6 | `scripts/baseline_sig_tests_v2.py`, `tab_sig_test*.tex` | Superseded by `paired_baseline_tests.py` (`tab_sig_summary`, `tab_sig_full_c5/c2`) | OPEN (archive) |
| D12 | `tab_baseline_pooled` B4 pooled EM | Committed 0.464 dropped 16 empty TruthfulQA answers (judge could not parse) from the denominator; counting them wrong (as the per-benchmark EM 0.240 does) gives 0.461 | FIXED 2026-10-17 |
| D13 | ch5 H4 verdict reading | Cited R² 0.207 of the saturating fit, which is fitted on the EM history (appendix C), not CHM | FIXED 2026-10-17 (clause replaced; verdict unchanged) |
| D14 | ch6 L27 and other ch6 numbers | Still carry ROUGE-era CHM values (e.g. B1 0.120) | OPEN (ch6 pass) |
| D15 | B5 five-shot CoT baseline (`scripts/run_baseline.py`, FEVER demonstrations) | B5 answers with a FEVER label ("supports"/"refutes"/"not enough info") on 235/300 TriviaQA and 289/300 TruthfulQA questions: the FEVER-only demonstrations taught the label format. Explains B5's lowest EM (0.405); CAEM wins against B5 partly reflect this baseline flaw. | OPEN (disclose in ch5 B5 bullet; future: benchmark-matched demonstrations) |
| C15 | Off-topic detection (follow-up to C14) | Post-hoc reranker on the full response (`scripts/score_response_relevance.py` → `outputs/analysis/response_relevance.json`) fails validation against 200 labels (`outputs/analysis/off_topic_validation_sample.json`, 15 off-topic): precision 0.12, recall 0.40. Bare-answer run signal separates correct/wrong as well or better (AUROC C0 TriviaQA 0.798 vs 0.745), so the verifier was not harmed. LLM judge script ready (`scripts/judge_off_topic.py`, Haiku) but blocked: API key on this machine rejected (rotated). | DOCUMENTED 2026-10-17: Haiku judge (new key) vs 200 labels: agreement 0.78, kappa 0.35 held-out → not reliable; off-topic excluded from CHM (mean of 7), detection = future work |
