#!/usr/bin/env bash
# scripts/halt_and_resume_with_deferred_fix.sh
# =============================================
# One-shot halt-restart procedure for the cycle-4-close orchestrator
# fix (Path Y per branch_C_log 2026-05-04 entry). Halts the live
# `plan_a` tmux session cleanly after the "Cycle 4 done in" marker,
# verifies the cycle-4 artefacts on disk, then restarts a new tmux
# session with --resume_from_cycle 5 so cycles 5-10 run with the
# deferred-buffer reconsideration pass active for the first time.
#
# Run this AFTER you see "Cycle 4 done in X min." in the log. Do NOT
# run it earlier — interrupting mid-cycle would corrupt cycle-4 state.
#
# Usage:
#   bash scripts/halt_and_resume_with_deferred_fix.sh

set -eu

ROOT=/workspace/caem
LOG=${ROOT}/outputs/full_run/run.log
CYCLE_4_DONE_PATTERN="Cycle 4 done in"

cd "${ROOT}"

# ----------------------------------------------------------------- #
# Step 1: verify the cycle-4-done marker is in the log              #
# ----------------------------------------------------------------- #
echo "[halt-restart] Step 1: checking for 'Cycle 4 done in' marker in run.log..."
if ! grep -q "${CYCLE_4_DONE_PATTERN}" "${LOG}"; then
    echo "ERROR: 'Cycle 4 done in' not yet in run.log. The cycle is still"
    echo "       running. Wait until the marker appears, then re-run this script."
    echo ""
    echo "Most recent log lines:"
    tail -3 "${LOG}"
    exit 1
fi
echo "[halt-restart] OK: cycle 4 close marker found."
grep "${CYCLE_4_DONE_PATTERN}" "${LOG}" | tail -1

# ----------------------------------------------------------------- #
# Step 2: halt the plan_a tmux session                              #
# ----------------------------------------------------------------- #
echo ""
echo "[halt-restart] Step 2: halting plan_a tmux session..."
if ! tmux has-session -t plan_a 2>/dev/null; then
    echo "WARN: plan_a session not running; skipping halt."
else
    tmux send-keys -t plan_a C-c
    echo "[halt-restart] Sent Ctrl+C to plan_a; waiting 30s for clean exit..."
    sleep 30
    if tmux has-session -t plan_a 2>/dev/null; then
        echo "WARN: plan_a still alive after 30s; sending second Ctrl+C..."
        tmux send-keys -t plan_a C-c
        sleep 15
    fi
fi

# ----------------------------------------------------------------- #
# Step 3: verify cycle-4 artefacts exist                            #
# ----------------------------------------------------------------- #
echo ""
echo "[halt-restart] Step 3: verifying cycle-4 artefacts on disk..."
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
    echo "ERROR: Missing cycle-4 artefacts:"
    for f in "${missing[@]}"; do echo "  - ${f}"; done
    echo "Aborting restart. Investigate the missing files before re-running."
    exit 1
fi
echo "[halt-restart] OK: all required cycle-4 artefacts present."

# ----------------------------------------------------------------- #
# Step 4: verify the orchestrator patch is in place                 #
# ----------------------------------------------------------------- #
echo ""
echo "[halt-restart] Step 4: confirming orchestrator deferred-buffer fix..."
if ! grep -q "deferred_buffer=pipeline.deferred_buffer" "${ROOT}/scripts/run_experiment.py"; then
    echo "ERROR: orchestrator patch not in scripts/run_experiment.py"
    echo "       Apply the patch from commit on feat/qwen-3b-goal1 first."
    exit 1
fi
if ! grep -q "reconsider_fn=pipeline.make_reconsider_deferred_fn" "${ROOT}/scripts/run_experiment.py"; then
    echo "ERROR: reconsider_fn kwarg not in scripts/run_experiment.py"
    exit 1
fi
echo "[halt-restart] OK: orchestrator patch verified."

# ----------------------------------------------------------------- #
# Step 5: launch the resumed runner in a fresh tmux session         #
# ----------------------------------------------------------------- #
echo ""
echo "[halt-restart] Step 5: launching --resume_from_cycle 5 in fresh tmux..."
LAUNCH_CMD=(
    tmux new-session -d -s plan_a -x 220 -y 60
    "source /venv/main/bin/activate && cd ${ROOT} && python scripts/run_experiment.py \
        --resume_from_cycle 5 \
        --output_dir outputs/full_run \
        2>&1 | tee -a outputs/phase1a_runner.log"
)
echo "Command:"
printf '  %s\n' "${LAUNCH_CMD[@]:0:3}"
echo "  '${LAUNCH_CMD[3]}'"
echo ""
echo "About to execute. Press Enter to proceed, Ctrl+C to abort."
read -r

"${LAUNCH_CMD[@]}"

echo ""
echo "[halt-restart] launched. Verifying RESUMING line appears in log within 60s..."
sleep 60
if grep -q "RESUMING EXPERIMENT FROM CYCLE 5" "${LOG}"; then
    echo "[halt-restart] OK: 'RESUMING EXPERIMENT FROM CYCLE 5' confirmed."
else
    echo "WARN: 'RESUMING EXPERIMENT FROM CYCLE 5' not seen in 60s. Check tmux:"
    echo "  tmux attach -t plan_a"
    exit 1
fi

# ----------------------------------------------------------------- #
# Step 6: also verify deferred-buffer kwarg was actually used at    #
# the first SIL call (cycle 5). The defensive log line should      #
# NOT fire in self_improvement.py:run_cycle.                       #
# ----------------------------------------------------------------- #
echo ""
echo "[halt-restart] Step 6: monitoring for deferred-reconsideration log lines..."
echo "[halt-restart] Watch for 'Deferred reconsideration: sweeping N entries' in"
echo "[halt-restart] outputs/full_run/run.log within ~30 minutes (during cycle 5"
echo "[halt-restart] Step 1 SIL fine-tune)."
echo ""
echo "[halt-restart] If you see 'deferred_buffer not provided to run_cycle' --"
echo "[halt-restart] something went wrong; the orchestrator did not pick up the patch."
echo ""
echo "[halt-restart] DONE. Cycle 5 will start with reconsideration ACTIVE."
