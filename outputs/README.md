# outputs/ (run outputs; ignored by git except this file and `analysis/`)

| Folder | Contents | Used by the current thesis? |
|---|---|---|
| `full_run/` | Main run, cycles 0–5: per-cycle evaluation files (`eval/`), stream-chunk snapshots, memory stores, run logs (`run.log`, `experiment.log`) | yes (chapter 5) |
| `cycle_0/` | Cycle-zero calibration, composite fit, validation fold, 100-variant sweep (`sweep_v2_1/`) | yes |
| `cold_start_memory/` | Gold-filtered cold-start memory and its pruning report | yes (chapter 4) |
| `baselines_phase1e/` | Current baseline panel B1–B8 (B6 under `vanilla_ft/`), verifier re-scores (`*_with_chm.json`) | yes |
| `baselines_phase1e_rerun/` | Phase-2 re-score of B6/B7 (June 2026) | provenance |
| `baselines_phase1e_pre_se_rerun_backup/` | First, faulty semantic-entropy (B8) run | no (provenance for tracker D1) |
| `baselines/`, `baselines_v2/` | Earlier baseline panels (include the dropped EWC-only fine-tune) | no |
| `phase1d_full_run/`, `phase1d_eval/`, `phase1d_eval_trajectories/` | Earlier run without the retrieval-utility classifier (classifier ablation arm, cycle 3) | yes (classifier ablation) |
| `tier1_routing_probe/`, `tier1_recall/`, `tier1_diagnostic/` | Router probe and Tier-1 diagnostics | probe: yes |
| `analysis/` | Evidence files written during the October 2026 drafting pass (tracked in git) | yes |
| `research/`, `calibration/`, `prompt_smoke/`, `ablation/`, loose `*.csv` / `*.log` | Earlier diagnostics and one-off outputs | no |

The re-run (MASTER_ISSUES.md) will write to new folders; nothing here is overwritten.
