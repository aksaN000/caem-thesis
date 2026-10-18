---
name: Stop runner after step_7_main — baselines/ablations need cycle-count rewire first
description: Phase 1a runner must be HALTED after step_7_main completes/early-stops; B6/B7 and ablation sweep are hardcoded to 10/5 cycles and will break matched-scale comparison if CAEM early-stopped at cycle <10.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Rule:** when `step_7_main` finishes (including any early-stop at cycle 5..9), STOP the runner before it auto-continues into `step_14_b6_vanilla_ft` / `step_15_b7_ewc_only_ft` / the ablation sweep. Do NOT let the bash chain proceed on the current (hardcoded) config.

**Why:** three places hardcode cycle counts that must instead track CAEM's actual `cycles_completed` from `outputs/full_run/run_complete.json` for matched-scale comparison:

1. `run_phase1a.sh:573,576` — `step_14_b6_vanilla_ft` uses `--num_cycles 10 --n_train_per_bench 50000`.
2. `run_phase1a.sh:598,601` — `step_15_b7_ewc_only_ft` same.
3. `run_phase1_full_ablations.sh:66` — `PHASE1_FULL_CYCLES=5` constant.

If CAEM early-stops at cycle 8, B6/B7 overtrain by 10k samples (50k vs 40k) → Chapter-5 matched-scale comparison broken. The chunk_size=5000 byte-identical-per-chunk contract (line 567-570 comment) stays intact, but the *total* training budget diverges.

**How to apply:** 
- User directive (2026-04-23 18:10 BDT): after `step_7_main` ends, stop the runner, wire the three fixes (read `cycles_completed`, propagate to `--num_cycles` and `--n_train_per_bench` for B6/B7; replace the `PHASE1_FULL_CYCLES=5` constant with a dynamic read), then resume.
- During autonomous ticks: if you see `step_7_main` complete in `run_complete.json`, issue `pkill -KILL -f run_phase1a.sh` / send `C-c` to the tmux session **before** the next step kicks off, log the stop in `branch_C_log.md`, and flag for user to authorize the rewire.
- Early-stop detection: the python script writes `outputs/full_run/run_complete.json` with `{"cycles_completed": N, "early_stopped": bool}` — use that as the stop signal, not the absence of `experiment_summary.csv` (which is also present during early-stop).
