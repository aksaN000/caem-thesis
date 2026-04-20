#!/usr/bin/env bash
# Auto-restart watcher: waits for ALL pre-Step-7 artefacts (Step 6 seed,
# Step 7.0 Cycle-0 eval + calibrated thresholds, Step 5.5 v5, Level B smoke
# pass, Step 19.2.1 retention slice) to be present, then kills + relaunches
# the phase1a tmux session so the new step_hf_upload_pre_main code in
# run_phase1a.sh (added 2026-04-20 after the original launch) actually runs.
#
# Trigger condition: retention slice `data/retention/cycle0_slice_500.jsonl`
# is the LAST artefact produced before step_7_main. Once it exists with
# non-zero size, Step 19.2.1 just completed and we can safely restart.
#
# Runs under its own tmux session `watcher` so it survives SSH disconnect.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

if [[ -f /venv/main/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi

LOG="outputs/phase1a_watcher.log"
TRIGGER="data/retention/cycle0_slice_500.jsonl"
MARKER="outputs/.watcher_restarted_runner"

ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts) BDT=$(TZ=Asia/Dhaka date '+%H:%M')] $*" | tee -a "$LOG"; }

log "watcher v2 started — polling for $TRIGGER every 60s"

if [[ -f "$MARKER" ]]; then
    log "MARKER already present ($MARKER) — watcher fired previously; exiting"
    exit 0
fi

while true; do
    if [[ -s "$TRIGGER" ]]; then
        log "retention slice present ($TRIGGER, $(stat -c%s "$TRIGGER") bytes) — Step 19.2.1 done. Proceeding."
        break
    fi
    # Extra safety: if outputs/full_run/cycle_1 appears, we've missed the window
    # (step_7_main started under old code). Log and still restart so idempotency
    # recovers cleanly via --resume_from_cycle 1.
    if [[ -d outputs/full_run/cycle_0 ]] && [[ -d outputs/full_run/cycle_1 ]]; then
        log "WARN: cycle_1 already running under OLD code. Restarting anyway — HF upload will run before resume."
        break
    fi
    sleep 60
done

log "killing tmux session 'phase1a' ..."
tmux kill-session -t phase1a 2>/dev/null || log "  (no phase1a session found; may have already exited)"
sleep 3

log "archiving old logs ..."
TS=$(date -u '+%Y%m%dT%H%M%SZ')
for f in phase1a_runner.log phase1a_tmux.log; do
    [[ -f "outputs/$f" ]] && mv "outputs/$f" "outputs/${f%.log}.${TS}.log" && log "  archived outputs/$f -> outputs/${f%.log}.${TS}.log"
done

log "relaunching runner under tmux session 'phase1a' with updated code ..."
tmux new-session -d -s phase1a \
    "bash -lc './run_phase1a.sh 2>&1 | tee outputs/phase1a_tmux.log; echo EXIT=\$? >> outputs/phase1a_tmux.log; exec bash'"
sleep 3
if tmux has-session -t phase1a 2>/dev/null; then
    log "runner relaunched. Idempotency will skip Steps 6 / 7.0 / 5.5 / prompt-ablation / Level-B / 19.2.1."
    log "New step_hf_upload_pre_main will run next, uploading to aksaN000/caem-passage-index-21m:pre_main_snapshot/."
    log "Then step_7_main enters (with --resume_from_cycle if any full_run cycles already exist)."
    touch "$MARKER"
    log "marker $MARKER written. watcher exiting."
else
    log "ERROR: tmux session 'phase1a' failed to start. Manual intervention needed."
    exit 1
fi
