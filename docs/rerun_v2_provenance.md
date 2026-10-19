# Final-version baseline re-run: provenance (Vast RTX 5090, 2026-10-18/19)

What produced each result of the matched-prompt baseline re-run, so it can be reproduced and audited.
Run 1 (CAEM C0–C5, run-1 baselines) is unchanged at tag `run1-final-2026-10-17`.

## Machine and environment
- RTX 5090 32 GB, driver 610.43.02 (CUDA 13.3), 257 GB RAM share, 61 CPU cores, 102 GB disk; passage index on the
  RAM disk (`data/passage_index -> /dev/shm/passage_index`).
- Python 3.12, exact package versions: `requirements-lock-rental-2026-10-18.txt` (torch 2.10.0+cu130,
  transformers 4.57.6, sentence-transformers 4.1.0, peft 0.21.2, datasets 5.0.1, anthropic 0.125.0).
- Runbook: `scripts/rental/final_version.sh` (stages preflight, env, data, search, baselines, flare, b6, gate, verify,
  pack). Logs: `outputs/rerun_v2/logs/` (runbook.log has every stage start and exit).

## Code commit per result (times BDT)
| Result | Started | Commit |
|---|---|---|
| Exact passage search, 1,500 evaluation questions (top 100) | 18 Oct 21:24 | f1af6d3 |
| Search check (live = cache 50/50; run-1 passage scores reproduced on 98 %) | 18 Oct 22:01 | 934949e |
| B2 chain of thought, B8 semantic entropy | 21:51 / 21:58 | 3fe34ce |
| B1 zero-shot, B5 five-shot (re-run after the GPU auto-reap kill) | 22:19 / 22:21 | 1f98a3f |
| B3 RAG and B4 CoT + RAG on TruthfulQA, B7 FLARE, B6, verifier pass | from 22:25 | 514dc13 |

Between 3fe34ce and 514dc13 only these changed: the search check, lazy loading of the passage store (index read on
the first live search), a log line, and the runbook's GPU auto-reap switch. No change touches prompts, generation,
retrieval results or scoring, so all re-run outputs share one protocol.

## Checks that make the outputs comparable with run 1
- CAEM's Tier 2 and Tier 3 prompts are byte-identical between run-1 code and the re-run code (all 1,500 evaluation
  questions checked, 2026-10-19).
- Every re-run file holds exactly CAEM's evaluation questions in the same order, with no empty answers
  (`scripts/rental/check_eval_questions.py`).
- The verifier pass uses run 1's baseline protocol (C3 adapter + C3 composite) with the pooled token-score path that
  CAEM's main run used; passages come from the exact-search cache (identical to live search).
- One scoring rule and one TruthfulQA judge protocol for every system (`eval.metrics.score_uniform`,
  `scripts/rescore_uniform.py`, `scripts/rescore_truthfulqa.py`).

## Known run-to-run variation
Signals computed from sampled text (entailment over sampled chains, dropout agreement, chain similarity) are random
draws by design and differ between runs; greedy generation can also differ slightly across GPUs and library builds.

## From outputs to thesis numbers (local, 2026-10-19)
`bash scripts/make_final_results.sh` (add `--judge` to re-run the TruthfulQA judge, `--check` to verify): uniform scorer
(`em_final`), CHM in the model and delivered views with status-tag accuracy (`outputs/analysis/chm_judged.json`), paired
tests (`outputs/analysis/paired_baseline_tests_final.json`), chapter 5 tables and figures (`build_ch5_evidence_tables.py`,
`build_final_tables.py`). Verified 2026-10-19: the command reproduces every committed output exactly.
- TruthfulQA judge on the re-run baselines: 2026-10-19 08:30 BDT, Claude Haiku 4.5, anthropic 0.125, protocol
  final_answer_v2 (`outputs/rerun_v2/tqa_judge_v2_baselines.log`; inputs backed up in
  `outputs/rerun_v2/backup_before_tqa_judge_baselines/`). CAEM C0–C5 and Phase 1d C3 were judged on 2026-10-17.
- A baseline's verifier decision is a measurement only and is never charged to it (`final_sources.load`).
- Delivered view follows the production envelope (`caem/production/confidence.py`): only abstentions are withheld.
- Replaced run-1 baseline files: `outputs/superseded_2026-10/run1_baselines/` (README there).
