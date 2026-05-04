#!/bin/bash
# watchdog_cycles_3plus.sh
#
# Generic stream-chunk + cycle-close watchdog for cycles 3-10.
# The running CAEM process lacks the Step 4b auto-snapshot and Step 6c
# full-gdrive-offload patches because those edits to run_experiment.py
# happened after the process started. This watchdog substitutes the
# missing behaviour from outside the process: it monitors run.log for
# stream-chunk completions, snapshots the per-sample JSON before Step 5
# overwrites it, and uploads each cycle's full artefact set to gdrive
# at cycle close.
#
# Exits when "Cycle 10 done in" appears OR when run_complete.json
# exists (whichever fires first).
#
# Layout matches the organized gdrive structure:
#   gdrive:caem-phase1a/full_run/cycle_{N}/
#       model.pt                 (already offloaded by SIL)
#       eval_stream_chunk/{bench}_cycle{N}_streamchunk.json
#       eval_transfer/{bench}_cycle{N}.json
#       composite_calibration.json, conformal_gate.json, ...
#       memory_store_cycle_{N}.faiss + .meta
#       deferred_buffer_cycle_{N}.pkl
#       retroverify_cycle{N}.json
#   gdrive:caem-phase1a/full_run/global/
#       mmlu_baseline.json, dataset_splits.json, run.log
set -u

ROOT=/workspace/caem
LOG=$ROOT/outputs/full_run/run.log
EVAL=$ROOT/outputs/full_run/eval
ROOT_OUT=$ROOT/outputs/full_run
WATCH=$ROOT/outputs/watchdog_cycles_3plus.log
GDRIVE=gdrive:caem-phase1a/full_run

mkdir -p "$(dirname $WATCH)"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $WATCH; }

log "=== watchdog cycles 3-10 started ==="
log "monitoring stream-chunk snapshots + cycle-close uploads for cycles 3-10"

snapshot_streamchunk() {
    local cycle=$1 bench=$2
    local snap=$EVAL/${bench}_cycle${cycle}_streamchunk.json
    [[ -f $snap ]] && return 0  # already snapshotted
    local src=$EVAL/${bench}_cycle${cycle}.json
    [[ -f $src ]] || return 1
    local size; size=$(stat -c %s "$src" 2>/dev/null || echo 0)
    # n=3000 stream-chunk version is ~5-6 MB; n=500 transfer-eval version ~1 MB.
    # 3 MB threshold cleanly separates them.
    if (( size < 3000000 )); then
        local lost=$EVAL/.${bench}_cycle${cycle}_streamchunk.lost
        if [[ ! -f $lost ]] && (( size > 0 )); then
            if grep -q "EvalHarness: starting ${bench} | cycle=${cycle} | n=3000" "$LOG" 2>/dev/null \
                && grep -q "EvalHarness: starting ${bench} | cycle=${cycle} | n=500" "$LOG" 2>/dev/null; then
                touch "$lost"
                log "MISSED: ${bench}_cycle${cycle}.json size=${size}B < 3MB (Step 5 overwrote)"
            fi
        fi
        return 1
    fi
    if ! awk -v b="$bench" -v c="$cycle" '
        $0 ~ "EvalHarness: starting "b" | cycle="c" | n=3000" {seen=1; next}
        seen && $0 ~ "EvalHarness done: "b" | cycle="c {found=1; exit}
        END {exit !found}
    ' "$LOG"; then
        return 1
    fi
    cp "$src" "$snap"
    log "SNAPSHOT cycle ${cycle}: ${bench}_cycle${cycle}.json -> ${snap##*/} (${size}B)"
    if rclone copy "$snap" "$GDRIVE/cycle_${cycle}/eval_stream_chunk/" --no-traverse 2>>$WATCH; then
        log "GDRIVE upload OK: ${bench}_cycle${cycle}_streamchunk.json -> cycle_${cycle}/eval_stream_chunk/"
    else
        log "ERROR: gdrive upload FAILED for ${bench}_cycle${cycle}_streamchunk.json"
    fi
}

upload_cycle_close() {
    local cycle=$1
    local marker=$ROOT_OUT/.watchdog_cycle${cycle}_close_uploaded
    [[ -f $marker ]] && return 0
    grep -q "Cycle ${cycle} done in" "$LOG" 2>/dev/null || return 0
    log "DETECTED: Cycle ${cycle} close marker; uploading all cycle artefacts to gdrive"
    local CYC=$GDRIVE/cycle_${cycle}

    # 1. cycle_{N}/ local artefacts (calibration, composite, gate)
    rclone copy "$ROOT_OUT/cycle_${cycle}/" "$CYC/" \
        --exclude "**/embeddings/*" 2>>$WATCH \
        && log "GDRIVE: cycle_${cycle}/ artefacts uploaded -> $CYC/" \
        || log "ERROR: cycle_${cycle}/ upload failed"

    # 2. cycle-close root artefacts
    for f in retroverify_cycle${cycle}.json \
             memory_store_cycle_${cycle}.faiss \
             memory_store_cycle_${cycle}.meta \
             deferred_buffer_cycle_${cycle}.pkl; do
        if [[ -f $ROOT_OUT/$f ]]; then
            rclone copy "$ROOT_OUT/$f" "$CYC/" --no-traverse 2>>$WATCH \
                && log "GDRIVE: $f -> cycle_${cycle}/" \
                || log "ERROR: gdrive upload of $f failed"
        fi
    done

    # 3. transfer-eval JSONs (n=500 final per benchmark) -> cycle_{N}/eval_transfer/
    for bench in fever triviaqa natural_questions truthfulqa strategyqa arc_challenge asqa; do
        local f=$EVAL/${bench}_cycle${cycle}.json
        if [[ -f $f ]]; then
            rclone copy "$f" "$CYC/eval_transfer/" --no-traverse 2>>$WATCH \
                && log "GDRIVE: ${bench}_cycle${cycle}.json -> cycle_${cycle}/eval_transfer/" \
                || log "ERROR: ${bench}_cycle${cycle}.json upload failed"
        fi
    done

    # 4. global/: mmlu_baseline + dataset_splits + run.log snapshot
    [[ -f $ROOT_OUT/mmlu_baseline.json ]] && rclone copy "$ROOT_OUT/mmlu_baseline.json" "$GDRIVE/global/" --no-traverse 2>>$WATCH
    [[ -f $ROOT_OUT/dataset_splits.json ]] && rclone copy "$ROOT_OUT/dataset_splits.json" "$GDRIVE/global/" --no-traverse 2>>$WATCH
    [[ -f $ROOT_OUT/run.log ]] && rclone copy "$ROOT_OUT/run.log" "$GDRIVE/global/" --no-traverse 2>>$WATCH \
        && log "GDRIVE: global/ updated (mmlu_baseline, dataset_splits, run.log)"

    touch "$marker"
    log "=== cycle ${cycle} close upload COMPLETE ==="
    return 0
}

while true; do
    # Stream-chunk snapshots — check cycles 3 through 10 for any uncaptured stream chunks
    for cycle in 3 4 5 6 7 8 9 10; do
        for bench in fever triviaqa natural_questions; do
            snapshot_streamchunk "$cycle" "$bench"
        done
    done
    # Cycle-close uploads — check cycles 3 through 10
    for cycle in 3 4 5 6 7 8 9 10; do
        upload_cycle_close "$cycle"
    done
    # Exit when cycle 10 close upload done OR run_complete.json present
    if [[ -f $ROOT_OUT/.watchdog_cycle10_close_uploaded ]] || [[ -f $ROOT_OUT/run_complete.json ]]; then
        log "=== watchdog cycles 3-10 done; exiting ==="
        exit 0
    fi
    sleep 30
done
