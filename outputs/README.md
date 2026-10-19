# outputs/ (run outputs; ignored by git except this file and `analysis/`)

| Folder | Contents | Used by the current thesis? |
|---|---|---|
| `full_run/` | Main run, cycles 0–5: per-cycle evaluation files (`eval/`), stream-chunk snapshots, memory stores, run logs (`run.log`, `experiment.log`) | yes (chapter 5) |
| `cycle_0/` | Cycle-zero calibration, composite fit, validation fold, 100-variant sweep (`sweep_v2_1/`) | yes |
| `cold_start_memory/` | Gold-filtered cold-start memory and its pruning report | yes (chapter 4) |
| `rerun_v2/` | Matched-protocol baseline re-run (2026-10-18/19): `baselines/` (B1, B2, B5, B7, B8 on all five benchmarks; B3/B4 TruthfulQA; B6 cycle-5 evaluation under `vanilla_ft/eval/`) with verifier passes (`*_with_chm.json`), `retrieval/` (exact-search cache and checks), `logs/`, `verifier_v2/` (post-defense exploration). See `docs/rerun_v2_provenance.md` | yes (final baselines) |
| `baselines_phase1e/` | Run-1 baseline panel (May 2026). Still used: `rag/` and `cot_rag/` except TruthfulQA (B3/B4), `vanilla_ft/cycle_*` (B6 weights) and the logs; the replaced files moved to `superseded_2026-10/run1_baselines/` | partly |
| `superseded_2026-10/` | Outputs replaced by a later run, with a README saying why | no (provenance) |
| `baselines_phase1e_rerun/` | Phase-2 re-score of B6/B7 (June 2026) | provenance |
| `baselines_phase1e_pre_se_rerun_backup/` | First, faulty semantic-entropy (B8) run | no (provenance for tracker D1) |
| `baselines/`, `baselines_v2/` | Earlier baseline panels (include the dropped EWC-only fine-tune) | no |
| `phase1d_full_run/`, `phase1d_eval/`, `phase1d_eval_trajectories/` | Earlier run without the retrieval-utility classifier (classifier ablation arm, cycle 3) | yes (classifier ablation) |
| `tier1_routing_probe/`, `tier1_recall/`, `tier1_diagnostic/` | Router probe and Tier-1 diagnostics | probe: yes |
| `analysis/` | Evidence files written during the October 2026 drafting pass (tracked in git) | yes |
| `research/`, `calibration/`, `prompt_smoke/`, `ablation/`, loose `*.csv` / `*.log` | Earlier diagnostics and one-off outputs | no |

Final numbers: `bash scripts/make_final_results.sh` (see the root README); `scripts/final_sources.py` names the file used for each system and benchmark.
