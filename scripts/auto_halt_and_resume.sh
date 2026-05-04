#!/usr/bin/env bash
# scripts/auto_halt_and_resume.sh
# ==============================
# AUTONOMOUS halt + restart at cycle-4 close. Polls run.log every 60s
# for "Cycle 4 done in"; when matched, halts plan_a, verifies artefacts,
# restarts with --resume_from_cycle 5, confirms RESUMING line.
#
# Designed to run unattended in the background. Logs everything to
# outputs/auto_halt_restart_audit.log for post-hoc verification.
#
# Usage (from /workspace/caem):
#   bash scripts/auto_halt_and_resume.sh > /dev/null 2>&1 &
# Or via Bash tool's run_in_background.

set -u  # NB: NOT -e because individual failures should be logged + retried

ROOT=/workspace/caem
LOG=${ROOT}/outputs/full_run/run.log
AUDIT=${ROOT}/outputs/auto_halt_restart_audit.log
CYCLE_4_DONE_PATTERN="Cycle 4 done in"
RESUME_PATTERN="RESUMING EXPERIMENT FROM CYCLE 5"

mkdir -p "$(dirname "${AUDIT}")"
log() { echo "$(date -u +%FT%TZ) [auto-halt-restart] $*" | tee -a "${AUDIT}"; }

cd "${ROOT}"

log "=== auto-halt-restart started; will poll for ${CYCLE_4_DONE_PATTERN} ==="

# ----------------------------------------------------------------- #
# Step 1: poll run.log until cycle-4-done marker appears            #
# ----------------------------------------------------------------- #
TIMEOUT_HOURS=18  # safety: bail out if we wait this long
START=$(date +%s)
while true; do
    if grep -q "${CYCLE_4_DONE_PATTERN}" "${LOG}" 2>/dev/null; then
        log "DETECTED: cycle 4 done marker in run.log."
        grep "${CYCLE_4_DONE_PATTERN}" "${LOG}" | tail -1 | tee -a "${AUDIT}"
        break
    fi
    NOW=$(date +%s)
    ELAPSED_HOURS=$(( (NOW - START) / 3600 ))
    if [[ ${ELAPSED_HOURS} -ge ${TIMEOUT_HOURS} ]]; then
        log "TIMEOUT: ${TIMEOUT_HOURS} hours elapsed without cycle-4 done marker. Aborting."
        exit 2
    fi
    sleep 60
done

# ----------------------------------------------------------------- #
# Step 2: halt plan_a tmux session                                   #
# ----------------------------------------------------------------- #
log "Step 2: halting plan_a tmux session..."
if ! tmux has-session -t plan_a 2>/dev/null; then
    log "WARN: plan_a session not running; skipping halt."
else
    tmux send-keys -t plan_a C-c
    log "Sent Ctrl+C to plan_a; waiting 30s for clean exit..."
    sleep 30
    if tmux has-session -t plan_a 2>/dev/null; then
        log "WARN: plan_a still alive after 30s; sending second Ctrl+C..."
        tmux send-keys -t plan_a C-c
        sleep 15
    fi
    if tmux has-session -t plan_a 2>/dev/null; then
        log "WARN: plan_a STILL alive after 45s; killing the session."
        tmux kill-session -t plan_a 2>/dev/null || true
        sleep 5
    fi
fi

# ----------------------------------------------------------------- #
# Step 3: verify cycle-4 artefacts on disk                           #
# ----------------------------------------------------------------- #
log "Step 3: verifying cycle-4 artefacts..."
required=(
    "outputs/full_run/cycle_4/conformal_gate.json"
    "outputs/full_run/cycle_4/composite_calibration.json"
    "outputs/full_run/memory_store_cycle_4.faiss"
    "outputs/full_run/memory_store_cycle_4.meta"
    "outputs/full_run/deferred_buffer_cycle_4.pkl"
    "outputs/full_run/retroverify_cycle4.json"
)
missing=()
for f in "${required[@]}"; do
    if [[ ! -f "${ROOT}/${f}" ]]; then
        missing+=("${f}")
    fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
    log "ERROR: missing cycle-4 artefacts:"
    for f in "${missing[@]}"; do log "  - ${f}"; done
    log "ABORTING. Investigate before manual restart."
    exit 1
fi
log "OK: all cycle-4 artefacts present."

# ----------------------------------------------------------------- #
# Step 4: confirm orchestrator patch is in place                     #
# ----------------------------------------------------------------- #
log "Step 4: confirming orchestrator deferred-buffer fix is staged..."
if ! grep -q "deferred_buffer=pipeline.deferred_buffer" "${ROOT}/scripts/run_experiment.py"; then
    log "ERROR: deferred_buffer kwarg NOT in scripts/run_experiment.py. Aborting restart."
    exit 1
fi
if ! grep -q "reconsider_fn=pipeline.make_reconsider_deferred_fn" "${ROOT}/scripts/run_experiment.py"; then
    log "ERROR: reconsider_fn kwarg NOT in scripts/run_experiment.py. Aborting restart."
    exit 1
fi
log "OK: orchestrator patch verified."

# ----------------------------------------------------------------- #
# Step 5: launch resumed runner in fresh tmux                        #
# ----------------------------------------------------------------- #
log "Step 5: launching --resume_from_cycle 5 in fresh tmux plan_a..."

# Note: --resume_from_cycle is the only required argument. Other flags
# (--num_cycles, --benchmarks, etc.) are read from config defaults or
# from the existing run state. Using the same launcher pattern that the
# original Phase 1a runbook used at run_phase1a.sh.
tmux new-session -d -s plan_a -x 220 -y 60 \
    "source /venv/main/bin/activate && cd ${ROOT} && \
     python scripts/run_experiment.py \
         --resume_from_cycle 5 \
         --output_dir outputs/full_run \
         2>&1 | tee -a outputs/phase1a_runner.log"

log "tmux launched. Waiting up to 120s for RESUMING line..."

# ----------------------------------------------------------------- #
# Step 6: confirm RESUMING line appears                              #
# ----------------------------------------------------------------- #
LAUNCHED_AT=$(date +%s)
WAIT_SECS=120
while true; do
    if grep -q "${RESUME_PATTERN}" "${LOG}" 2>/dev/null; then
        # Check it's the NEW resume line, not an old one (line number must
        # be near end of file).
        lineno=$(grep -n "${RESUME_PATTERN}" "${LOG}" | tail -1 | cut -d: -f1)
        total=$(wc -l < "${LOG}")
        if [[ ${total} -ge ${lineno} ]] && [[ $((total - lineno)) -lt 1000 ]]; then
            log "OK: '${RESUME_PATTERN}' confirmed in run.log (line ${lineno} of ${total})."
            break
        fi
    fi
    NOW=$(date +%s)
    ELAPSED=$(( NOW - LAUNCHED_AT ))
    if [[ ${ELAPSED} -ge ${WAIT_SECS} ]]; then
        log "WARN: '${RESUME_PATTERN}' not seen in ${WAIT_SECS}s. Inspecting tmux..."
        tmux capture-pane -t plan_a -p 2>&1 | tail -20 | tee -a "${AUDIT}"
        log "WARN: restart may have failed. Manual investigation required."
        exit 3
    fi
    sleep 5
done

# ----------------------------------------------------------------- #
# Step 7: report final status                                        #
# ----------------------------------------------------------------- #
log "=== auto-halt-restart COMPLETE ==="
log "Cycle 5 will run with deferred-buffer reconsideration ACTIVE."
log "Watch for 'Deferred reconsideration: sweeping N entries' in run.log"
log "during cycle 5's Step 1 SIL fine-tune (~30 min after restart)."
log ""
log "Audit log: ${AUDIT}"
log "Run log:   ${LOG}"
exit 0
