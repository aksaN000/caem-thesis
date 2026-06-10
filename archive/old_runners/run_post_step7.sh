#!/usr/bin/env bash
# run_post_step7.sh — THE single runner for everything after step_7_main.
#
# Purpose: after Phase 1a's step_7_main completes (10-cycle CAEM training),
# run the rest of the pipeline in one sequential pass:
#   1. External baselines  B1-B7   (Steps 9-15)
#   2. Significance tests  CAEM vs each baseline  (Step 15.5)
#   3. Purity validation   memory-store α proof   (Step 19)
#   4. Retention eval      last-cycle checkpoint  (Step 19.2.2)
#   5. Signal correlation  nine-signal matrix     (Step 19.5)
#   6. Ablation aggregate  cross-seed CSV         (Step 20)
#   7. Ch5/Ch6 artefacts   LaTeX + figures + Session-5 audit
#                                                  (Steps 21-25)
#
# Prerequisite: step_7_main must have succeeded. Checked at entry.
#
# Idempotent: every step has its own "skip if output already exists"
# guard, so you can kill + rerun this script at any time and it picks up
# where it left off.
#
# Safety: mirrors run_phase1a_hardened.sh's memory-guard discipline
# (MALLOC_TRIM_THRESHOLD, heartbeat sidecar, tmux-friendly).
#
# Usage:
#   tmux new-session -d -s phase1b ./run_post_step7.sh
#   tmux attach -t phase1b        # watch live
#   tail -f outputs/phase1b_runner.log outputs/heartbeat.log
#
# All shell step function definitions come from run_phase1a.sh via source
# — this script is a pure dispatcher, no duplication.

set -u
cd "$(dirname "$(readlink -f "$0")")"

# ----------------------------------------------------------------------
# Prerequisite: Phase 1a main run must be complete.
# ----------------------------------------------------------------------
if [[ ! -f outputs/full_run/run_complete.json ]]; then
    # Fall back: check experiment_summary.csv for >=7 cycle rows (header +
    # cycles 0..5). Matches the early-stop tolerance in step_7_main's own
    # post-assertion.
    if [[ ! -f outputs/full_run/experiment_summary.csv ]]; then
        echo "[post_step7] FATAL: step_7_main has not completed."
        echo "  missing BOTH outputs/full_run/run_complete.json AND"
        echo "  outputs/full_run/experiment_summary.csv"
        echo "  Complete Phase 1a main run first (./run_phase1a_hardened.sh)."
        exit 2
    fi
    n_rows=$(wc -l < outputs/full_run/experiment_summary.csv)
    if (( n_rows < 7 )); then
        echo "[post_step7] FATAL: experiment_summary.csv has $n_rows rows;"
        echo "  need >=7 (header + cycles 0..5). step_7_main incomplete."
        exit 2
    fi
fi

# ----------------------------------------------------------------------
# Source function definitions from run_phase1a.sh.
# run_phase1a.sh's main() is guarded by BASH_SOURCE, so sourcing only
# loads env + function defs; main() does NOT fire.
# ----------------------------------------------------------------------
# shellcheck disable=SC1091
source ./run_phase1a.sh

# Override RUNNER_LOG so phase1b output lands in its own log.
RUNNER_LOG="outputs/phase1b_runner.log"
mkdir -p outputs
: > "$RUNNER_LOG"  # truncate at start so each post-step7 run is self-contained

band "CAEM post-step_7 runner starting at $(ts)"
log "Repo: $PWD"
log "Target outputs: baselines/, full_run/tab_*.tex, full_run/fig5_*.pdf"

# ----------------------------------------------------------------------
# Memory-guard heartbeat (mirrors run_phase1a_hardened.sh).
# ----------------------------------------------------------------------
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-131072}"
export PYTHONMALLOC="${PYTHONMALLOC:-malloc}"

HEARTBEAT_LOG="outputs/heartbeat.log"
heartbeat_sidecar() {
    while true; do
        local ts_now cgroup_use gpu_mem gpu_util oom_now
        ts_now=$(ts)
        cgroup_use=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
        oom_now=$(awk '/^oom_kill / {print $2; exit}' /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null || echo "?")
        gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
        printf "%s  [phase1b] cgroup=%sMB  gpu=%sMB/%s%%  oom=%s\n" \
            "$ts_now" \
            "$((cgroup_use/1024/1024))" \
            "$gpu_mem" "$gpu_util" "$oom_now" \
            >> "$HEARTBEAT_LOG"
        sleep 60
    done
}
heartbeat_sidecar &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

# ----------------------------------------------------------------------
# Step sequence. Each step is idempotent — reruns skip completed work.
# ----------------------------------------------------------------------

# 1) External baselines (B1-B7)
band "Phase 1b — external baselines (B1-B7)"
step_9_b1
step_10_b2
step_11_5_b5            # 5-shot CoT (reclaimed B5 slot)
step_11_b3
step_12_b4
step_14_b6_vanilla_ft
step_15_b7_ewc_only

# 2) Significance tests across all baselines
step_15_5_sig

# 3) Purity theorem validation
step_19_purity

# 4) Last-cycle retention diagnostic
step_19_2_eval

# 5) Nine-signal correlation matrix
step_19_5_corr

# 6) Ablation aggregate (Phase 1a only writes the `full` row; OK.
#    Ablation sweep (AB variants) lives in its own runner post-rewire.)
step_20_aggregate

# 7) Ch5/Ch6 artefacts
band "Phase 1b — Ch5/Ch6 artefact generation"
step_21_tables
step_22_figures
step_23_calib_traj
step_24_session5_artifacts
step_25_t1_robustness

band "Phase 1b COMPLETE at $(ts)"
log "All Ch5/Ch6 tables + figures + audit artefacts written."
log "Next: scp outputs/full_run/ + 'pre thesis 1 report/' down locally; thesis write-up."
