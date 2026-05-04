#!/bin/bash
# watchdog_cycle2.sh
#
# Captures cycle-2 stream-chunk per-sample JSONs (n=3000) before Step 5
# transfer eval (n=500) overwrites them, and uploads every cycle-2 artefact
# to gdrive on cycle close. Exits once cycle-2 close upload is complete.
#
# Designed to run in a fresh tmux session alongside the main runner without
# touching the running Python process. Handles cycle-2 only; cycles 3+ use
# the patched run_experiment.py auto-snapshot / auto-offload paths.
#
# Triggers:
#   1. Stream-chunk snapshot — when eval/{bench}_cycle2.json exists with
#      size > 3 MB (n=3000 version; n=500 version is ~1 MB).
#   2. Cycle-close upload — when run.log emits "Cycle 2 done in".
set -u

ROOT=/workspace/caem
LOG=$ROOT/outputs/full_run/run.log
EVAL=$ROOT/outputs/full_run/eval
ROOT_OUT=$ROOT/outputs/full_run
WATCH=$ROOT/outputs/watchdog_cycle2.log
GDRIVE=gdrive:caem-phase1a/full_run
CYCLE=2

mkdir -p "$(dirname $WATCH)"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $WATCH; }

log "=== watchdog cycle ${CYCLE} started ==="

# ---------- Tier 1: stream-chunk snapshot ---------------------------------
snapshot_streamchunk() {
    local bench=$1
    local snap=$EVAL/${bench}_cycle${CYCLE}_streamchunk.json
    [[ -f $snap ]] && return 0  # already snapshotted
    local src=$EVAL/${bench}_cycle${CYCLE}.json
    [[ -f $src ]] || return 1
    local size; size=$(stat -c %s "$src")
    # n=3000 version is ~5-6 MB; n=500 version is ~1 MB. 3 MB threshold
    # cleanly separates them. If size already < 3MB, Step 5 has overwritten
    # already and we missed the window (or Step 3 produced a tiny output).
    if (( size < 3000000 )); then
        # Mark that we attempted but file was too small; only log once
        local lost=$EVAL/.${bench}_cycle${CYCLE}_streamchunk.lost
        if [[ ! -f $lost ]] && (( size > 0 )); then
            # Only flag as lost if Step 3 has actually run for this bench
            if grep -q "EvalHarness: starting ${bench} | cycle=${CYCLE} | n=3000" "$LOG" 2>/dev/null \
                && grep -q "EvalHarness: starting ${bench} | cycle=${CYCLE} | n=500" "$LOG" 2>/dev/null; then
                touch "$lost"
                log "MISSED: ${bench}_cycle${CYCLE}.json size=${size}B < 3MB (Step 5 overwrote before snapshot)"
            fi
        fi
        return 1
    fi
    # Confirm Step 3 (n=3000) has completed for this bench by checking for
    # the corresponding "EvalHarness done" line after the n=3000 start.
    if ! awk -v b="$bench" -v c="$CYCLE" '
        $0 ~ "EvalHarness: starting "b" | cycle="c" | n=3000" {seen=1; next}
        seen && $0 ~ "EvalHarness done: "b" | cycle="c {found=1; exit}
        END {exit !found}
    ' "$LOG"; then
        return 1  # Step 3 not yet complete; size>3MB is mid-write
    fi
    cp "$src" "$snap"
    log "SNAPSHOT: ${bench}_cycle${CYCLE}.json -> ${snap##*/} (size=${size}B)"
    if rclone copy "$snap" "$GDRIVE/cycle_${CYCLE}/eval_stream_chunk/" --no-traverse 2>>$WATCH; then
        log "GDRIVE: uploaded ${bench}_cycle${CYCLE}_streamchunk.json"
    else
        log "ERROR: gdrive upload FAILED for ${bench}_cycle${CYCLE}_streamchunk.json"
    fi
}

# ---------- Tier 2: full cycle-close upload --------------------------------
upload_cycle_close() {
    local marker=$ROOT_OUT/.watchdog_cycle${CYCLE}_close_uploaded
    [[ -f $marker ]] && return 0
    grep -q "Cycle ${CYCLE} done in" "$LOG" 2>/dev/null || return 0
    log "DETECTED: Cycle ${CYCLE} close marker; uploading all cycle artefacts to gdrive"

    # All artefacts go under gdrive:.../cycle_${CYCLE}/ (organized layout).
    local CYC=$GDRIVE/cycle_${CYCLE}

    # 1. cycle_${CYCLE}/ local directory contents (calibration files etc.;
    # model.pt already offloaded by SIL into cycle_${CYCLE}/model.pt)
    rclone copy "$ROOT_OUT/cycle_${CYCLE}/" "$CYC/" \
        --exclude "**/embeddings/*" 2>>$WATCH \
        && log "GDRIVE: cycle_${CYCLE}/ artefacts uploaded -> $CYC/" \
        || log "ERROR: gdrive upload of cycle_${CYCLE}/ failed"

    # 2. cycle-close root artefacts -> cycle_${CYCLE}/
    for f in retroverify_cycle${CYCLE}.json \
             memory_store_cycle_${CYCLE}.faiss \
             memory_store_cycle_${CYCLE}.meta \
             deferred_buffer_cycle_${CYCLE}.pkl; do
        if [[ -f $ROOT_OUT/$f ]]; then
            rclone copy "$ROOT_OUT/$f" "$CYC/" --no-traverse 2>>$WATCH \
                && log "GDRIVE: $f -> cycle_${CYCLE}/" \
                || log "ERROR: gdrive upload of $f failed"
        fi
    done

    # 3. transfer-eval JSONs (n=500 final) -> cycle_${CYCLE}/eval_transfer/
    for bench in fever triviaqa natural_questions truthfulqa strategyqa arc_challenge asqa; do
        local f=$EVAL/${bench}_cycle${CYCLE}.json
        if [[ -f $f ]]; then
            rclone copy "$f" "$CYC/eval_transfer/" --no-traverse 2>>$WATCH \
                && log "GDRIVE: ${bench}_cycle${CYCLE}.json -> cycle_${CYCLE}/eval_transfer/" \
                || log "ERROR: gdrive transfer-eval upload failed for ${bench}"
        fi
    done

    # 4. global/: mmlu_baseline + dataset_splits + run.log snapshot
    [[ -f $ROOT_OUT/mmlu_baseline.json ]] && rclone copy "$ROOT_OUT/mmlu_baseline.json" "$GDRIVE/global/" --no-traverse 2>>$WATCH \
        && log "GDRIVE: mmlu_baseline.json -> global/"
    [[ -f $ROOT_OUT/dataset_splits.json ]] && rclone copy "$ROOT_OUT/dataset_splits.json" "$GDRIVE/global/" --no-traverse 2>>$WATCH \
        && log "GDRIVE: dataset_splits.json -> global/"
    [[ -f $ROOT_OUT/run.log ]] && rclone copy "$ROOT_OUT/run.log" "$GDRIVE/global/" --no-traverse 2>>$WATCH \
        && log "GDRIVE: run.log snapshot -> global/"

    touch "$marker"
    log "=== cycle ${CYCLE} close upload COMPLETE ==="
    return 0
}

# ---------- Main loop ------------------------------------------------------
while true; do
    for bench in fever triviaqa natural_questions; do
        snapshot_streamchunk "$bench"
    done
    upload_cycle_close
    if [[ -f $ROOT_OUT/.watchdog_cycle${CYCLE}_close_uploaded ]]; then
        log "=== watchdog cycle ${CYCLE} done; exiting ==="
        exit 0
    fi
    sleep 30
done
