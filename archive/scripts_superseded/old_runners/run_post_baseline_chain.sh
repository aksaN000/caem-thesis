#!/usr/bin/env bash
# Post-baseline autonomous pipeline driver (2026-05-16).
#
# Sources run_phase1a.sh to import the step_* function library, then
# invokes only the post-baseline chain. Avoids re-running the trajectory
# generation (step_5_9 ... step_7_main) which is already complete.
#
# Invoked from a fresh tmux session after B6 vanilla_ft closes:
#   tmux new-session -d -s pipeline 'bash run_post_baseline_chain.sh'

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$PWD"

# Pick up persisted env vars (ANTHROPIC_API_KEY etc.) — needed because
# Claude Code Bash-tool subshells don't auto-source /etc/environment or
# .bashrc on non-interactive launch.
set -a
[[ -f /etc/environment ]] && . /etc/environment
set +a

if [[ -f /venv/main/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi

# Sentinel for status messages.
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] CHAIN: $*"; }
band() { log "==================== $* ===================="; }

# Set RUNNER_LOG (used by step_* functions) before sourcing.
RUNNER_LOG="outputs/runner_post_baseline_$(date -u '+%Y-%m-%dT%H-%M-%SZ').log"
mkdir -p outputs
export RUNNER_LOG

# Source the step library (the BASH_SOURCE guard prevents main() firing).
# shellcheck disable=SC1091
source run_phase1a.sh

band "POST-BASELINE CHAIN started"
log "RUNNER_LOG=$RUNNER_LOG"
log "ANTHROPIC_API_KEY set: ${ANTHROPIC_API_KEY:+yes (len=${#ANTHROPIC_API_KEY})}"

# Step 15.1 — OOM recovery (~5 min CPU)
step_recover_oom_baselines || log "WARN: recovery returned non-zero"

# Step 15.1.5 — capability_em (~5 min CPU)
step_capability_em || log "WARN: capability_em returned non-zero"

# Step 15.2 — TruthfulQA LLM judge (~20 min, needs ANTHROPIC_API_KEY)
step_rescore_truthfulqa || log "WARN: truthfulqa rescore returned non-zero"

# Step 15.3 — Verifier rescore (~7 GPU-h) -- the BIG one
step_rescore_baselines_through_verifier || log "WARN: verifier rescore returned non-zero"

# Step 15.4 — gdrive sync (~10 min)
step_gdrive_sync_post_trajectory || log "WARN: gdrive sync returned non-zero"

# Step 15.5 — significance tests (~10 min CPU)
step_15_5_sig || log "WARN: sig tests returned non-zero"

# Step 15.6 — decomposition tables (~5 min CPU)
step_15_6_decomposition_tables || log "WARN: decomposition tables returned non-zero"

# Step 15.7 — T1-recall diagnostic (~45 min GPU + API)
step_15_7_tier1_recall_diagnostic || log "WARN: T1-recall returned non-zero"

# Diagnostics (~10-15 min CPU total)
step_19_purity || log "WARN: purity returned non-zero"
step_19_2_eval || log "WARN: eval rescore returned non-zero"
step_19_5_corr || log "WARN: correlation returned non-zero"
step_19_6_theorem_receipts || log "WARN: theorem receipts returned non-zero"

# Final aggregator + figures (~20 min CPU)
step_20_aggregate || log "WARN: step_20 returned non-zero"
step_21_tables || log "WARN: step_21 returned non-zero"
step_22_figures || log "WARN: step_22 returned non-zero"
step_23_calib_traj || log "WARN: step_23 returned non-zero"
step_24_session5_artifacts || log "WARN: step_24 returned non-zero"
step_25_t1_robustness || log "WARN: step_25 returned non-zero"

# Final sync
step_gdrive_sync_post_trajectory || log "WARN: final gdrive sync returned non-zero"

band "POST-BASELINE CHAIN COMPLETE"
log "Next: Ch5/Ch6 writeup using thesis_report/figures/auto/*.tex"
