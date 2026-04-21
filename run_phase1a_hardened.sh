#!/usr/bin/env bash
# Hardened launch wrapper for run_phase1a.sh.
#
# Why: 2026-04-21 15:26 UTC silent kill. Cause was kernel OOM at the 90 GB
# cgroup limit (peak 231 GB). Python left no traceback because SIGKILL
# bypasses handlers. This wrapper:
#   1. Captures exit code + signal in phase1a_runner.log
#   2. Records pre/post cgroup OOM counter so next kill is identified instantly
#   3. Spawns a 60s heartbeat logging RSS, GPU mem, and cgroup usage
#   4. Sets MALLOC_TRIM_THRESHOLD_ to aggressively return freed memory to OS
#
# Usage:
#   tmux new -d -s plan_a './run_phase1a_hardened.sh'
#   tmux attach -t plan_a       # to watch
#   tail -f outputs/heartbeat.log outputs/phase1a_runner.log

set -u
cd "$(dirname "$(readlink -f "$0")")"

RUNNER_LOG="outputs/phase1a_runner.log"
HEARTBEAT_LOG="outputs/heartbeat.log"
PID_FILE="outputs/phase1a_runner.pid"
mkdir -p outputs

# --- Aggressive glibc malloc tuning so freed memory returns to the OS ---
export MALLOC_TRIM_THRESHOLD_=131072       # 128 KB (default is 128 MB)
export MALLOC_MMAP_THRESHOLD_=131072
export PYTHONMALLOC=malloc                  # route python allocator through glibc

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

# --- Capture pre-launch cgroup OOM counter + RSS baseline ---
OOM_BEFORE=$(awk '{print $2}' /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null | head -1 || echo "?")
MEM_BEFORE=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 0)

{
    echo ""
    echo "==================== HARDENED WRAPPER LAUNCH $(ts) ===================="
    echo "cgroup memory.limit_in_bytes = $LIMIT ($(numfmt --to=iec $LIMIT 2>/dev/null || echo ?))"
    echo "cgroup memory.usage_in_bytes = $MEM_BEFORE ($(numfmt --to=iec $MEM_BEFORE 2>/dev/null || echo ?))"
    echo "cgroup memory OOM counter (pre) = $OOM_BEFORE"
    echo "MALLOC_TRIM_THRESHOLD_ = $MALLOC_TRIM_THRESHOLD_"
} >> "$RUNNER_LOG"

# --- Heartbeat sidecar (every 60s) ---
heartbeat() {
    while true; do
        local ts_now rss_kb cgroup_use oom_now gpu_mem gpu_util runner_alive
        ts_now=$(ts)
        cgroup_use=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
        oom_now=$(awk '{print $2}' /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null | head -1 || echo "?")
        gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            rss_kb=$(awk '/VmRSS/ {print $2}' "/proc/$(cat "$PID_FILE")/status" 2>/dev/null || echo 0)
            runner_alive=YES
        else
            rss_kb=0
            runner_alive=NO
        fi
        printf "%s  cgroup=%sMB (%d%%)  runner_rss=%sMB  gpu=%sMB/%s%%  oom=%s  alive=%s\n" \
            "$ts_now" \
            "$((cgroup_use/1024/1024))" \
            "$((cgroup_use*100/(LIMIT>0?LIMIT:1)))" \
            "$((rss_kb/1024))" \
            "$gpu_mem" "$gpu_util" \
            "$oom_now" "$runner_alive" \
            >> "$HEARTBEAT_LOG"
        sleep 60
    done
}
heartbeat &
HEARTBEAT_PID=$!

cleanup() {
    local rc=$?
    local oom_after
    oom_after=$(awk '{print $2}' /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null | head -1 || echo "?")
    {
        echo "==================== HARDENED WRAPPER EXIT $(ts) ===================="
        echo "exit_code = $rc"
        echo "cgroup memory OOM counter (post) = $oom_after (was $OOM_BEFORE)"
        if [[ "$oom_after" != "$OOM_BEFORE" ]]; then
            echo "!!! KERNEL OOM KILL DETECTED during this run !!!"
        fi
        echo "peak cgroup usage = $(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null || echo ?)"
        echo "=================================================================="
    } >> "$RUNNER_LOG"
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    exit "$rc"
}
trap cleanup EXIT INT TERM

# --- Launch the real runner with its PID pinned ---
./run_phase1a.sh &
RUNNER_PID=$!
echo "$RUNNER_PID" > "$PID_FILE"
echo "[$(ts)] hardened wrapper: runner pid=$RUNNER_PID" >> "$RUNNER_LOG"
wait "$RUNNER_PID"
