#!/usr/bin/env bash
# scripts/launch_baselines_v2.sh
# ===========================================
# Launch zero_shot + rag baselines on the v2 fresh pool (9 004 questions
# across 7 benchmarks, content-hash disjoint from every CAEM pool).
#
# Output: outputs/baselines_v2/<baseline>/<bench>_cycle0.json
#
# Why only zero_shot + rag: those are the only two baselines whose outputs
# are used by the v2 RUC canonical pairing. Other baselines (cot, cot_rag,
# fiveshot_cot, flare) added no AUROC lift in v1 (aug pairings empirically
# moved pooled AUROC by 0.002) and would triple the compute cost.
#
# CAEM_BATCH_U_TOK_DROP=1 — matches the cycle-3 trajectory's verifier
# config so the u_token / u_dropout signals are computed via the same
# pooled path used by step_7_main. Critical for the downstream rescore
# to be on the same instrument as the trajectory.
#
# Estimated wall-time on a single 5090 with eval_batch_size=32:
#   zero_shot baseline: 9 004 samples × ~5-8 sec ≈ 12-20 h
#   rag baseline:       9 004 samples × ~15-20 sec ≈ 38-50 h
#   Total: ~50-70 GPU-h (~2-3 days unattended)
# ============================================================================

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH="$PWD"

if [[ -f /venv/main/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi

# Match step_7_main env exactly so the rescore pipeline downstream uses
# verifier signals computed via the same code path the trajectory used.
export CAEM_BATCH_U_TOK_DROP="${CAEM_BATCH_U_TOK_DROP:-1}"
export CAEM_FORCE_GPU_CLEANUP="${CAEM_FORCE_GPU_CLEANUP:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="outputs/baselines_v2"
PASSAGE_INDEX="data/passage_index"
FRESH_POOL_JSONS=(
    caem/ruc/fresh_pool/fever_v2.json
    caem/ruc/fresh_pool/triviaqa_v2.json
    caem/ruc/fresh_pool/commonsense_qa_v2.json
    caem/ruc/fresh_pool/strategyqa_v2.json
    caem/ruc/fresh_pool/truthfulqa_v2.json
    caem/ruc/fresh_pool/haluevalqa_v2.json
    caem/ruc/fresh_pool/openbookqa_v2.json
    caem/ruc/fresh_pool/natural_questions_v2.json
)
# 2026-05-18: added natural_questions to balance the training pool. NQ is
# a known RAG-extensive benchmark (Mallen 2023 reports ~50% accuracy lift on
# long-tail entities). RUC-only: NQ is NOT added to caem.config's
# TRAINING_BENCHMARKS — the CAEM trajectory panel stays at v2.1 (fever,
# triviaqa, commonsense_qa training; truthfulqa, strategyqa transfer).
BENCHES=(fever triviaqa commonsense_qa strategyqa truthfulqa haluevalqa openbookqa natural_questions)

# Smoke-test mode: --smoke uses 30 samples per bench (210 total) via temp
# subsetted JSONs so we can validate the pipeline before committing to the
# full 9 004-question run. The temp files live in /tmp/ruc_smoke/.
SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE=1
    shift
fi

if [[ "$SMOKE" == "1" ]]; then
    OUTPUT_DIR="outputs/baselines_v2_smoke"
    SMOKE_DIR="/tmp/ruc_smoke"
    mkdir -p "$SMOKE_DIR"
    python -c "
import json, sys
from pathlib import Path
src_files = '''$(echo "${FRESH_POOL_JSONS[@]}")'''.split()
for src in src_files:
    rows = json.load(open(src))[:30]   # first 30 per file
    dst = Path('$SMOKE_DIR') / Path(src).name
    json.dump(rows, open(dst, 'w'))
    print(f'  smoke: {Path(src).name} -> {dst} ({len(rows)} rows)')
"
    FRESH_POOL_JSONS=(
        "$SMOKE_DIR/fever_v2.json"
        "$SMOKE_DIR/triviaqa_v2.json"
        "$SMOKE_DIR/commonsense_qa_v2.json"
        "$SMOKE_DIR/strategyqa_v2.json"
        "$SMOKE_DIR/truthfulqa_v2.json"
        "$SMOKE_DIR/haluevalqa_v2.json"
        "$SMOKE_DIR/openbookqa_v2.json"
    )
fi

mkdir -p "$OUTPUT_DIR"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }
band() { log "==================== $* ===================="; }

# --- Pre-flight ---
for jp in "${FRESH_POOL_JSONS[@]}"; do
    [[ -f "$jp" ]] || { log "FATAL: missing $jp"; exit 1; }
done
if [[ ! -d "$PASSAGE_INDEX" && ! -f "${PASSAGE_INDEX}/passages.faiss" ]]; then
    log "FATAL: passage index not found at $PASSAGE_INDEX"
    exit 1
fi

# --- Skip-if-exists check ---
# A baseline+bench combo is "done" iff its output JSON exists AND has the
# expected number of samples (matches the fresh-pool slice size for the
# bench). This lets us resume after an interruption without re-running
# completed benches.
all_benches_done_for() {
    local name="$1"
    for bench in "${BENCHES[@]}"; do
        local out="$OUTPUT_DIR/$name/${bench}_cycle0.json"
        [[ -f "$out" ]] || return 1
    done
    return 0
}

# --- Inference baseline runner ---
run_inference_baseline() {
    local name="$1"
    if all_benches_done_for "$name"; then
        band "v2 baseline: $name -- SKIP (all bench outputs already exist)"
        log "  Found outputs for every bench in: $OUTPUT_DIR/$name/"
        log "  Delete the JSONs you want to re-run, or run with --force to clobber."
        return 0
    fi
    band "v2 baseline: $name (mode=$([[ "$SMOKE" == "1" ]] && echo SMOKE || echo FULL))"
    python -m scripts.run_baseline \
        --baseline "$name" \
        --output_dir "$OUTPUT_DIR" \
        --benchmarks "${BENCHES[@]}" \
        --questions_jsons "${FRESH_POOL_JSONS[@]}" \
        --eval_batch_size 32 \
        --eval_prefetch \
        --passage_index "$PASSAGE_INDEX" \
        --seed 42 \
        2>&1 | tee -a "$OUTPUT_DIR/$name.gen.log"
}

# --- Dispatch ---
TARGET="${1:-all}"
case "$TARGET" in
    zero_shot)         run_inference_baseline zero_shot ;;
    rag)               run_inference_baseline rag ;;
    all)
        run_inference_baseline zero_shot
        run_inference_baseline rag
        ;;
    *)
        log "Unknown target: $TARGET (expected zero_shot|rag|all)"
        exit 1
        ;;
esac

band "v2 baselines complete (mode=$([[ "$SMOKE" == "1" ]] && echo SMOKE || echo FULL))"
log "Outputs in $OUTPUT_DIR"
