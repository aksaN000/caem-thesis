#!/usr/bin/env bash
# scripts/launch_baselines.sh
# ============================
# One-shot launcher for the post-C5 baseline panel B1-B7.
#
# Created 2026-05-15 after the baseline-implementation audit. Bakes in
# every correct override so the panel runs at matched-protocol against
# CAEM cycle 3 / cycle 5 without any per-baseline flag-fiddling.
#
# Usage:
#   bash scripts/launch_baselines.sh [zero_shot|cot|fiveshot_cot|rag|cot_rag|vanilla_ft|ewc_only_ft|all]
#
# Defaults (all baked in to match v2.1 architecture + post-C5 trajectory):
#   n_questions / n_eval_per_bench  = 300   (matches CAEM eval fold)
#   eval_batch_size                  = 32    (matches CAEM step_7_main)
#   num_cycles (B6/B7 only)          = 5     (matches C5-stop decision)
#   --composite_calibration (rescore) = outputs/full_run/cycle_3/composite_calibration.json
#                                       (matched-protocol pin; do NOT use the
#                                        retired outputs/production/ default)
#   --passage_index                   = data/passage_index (same as CAEM)
#   --seed                            = 42 (matches run_experiment.py)
#
# Pipeline phases:
#   A. Generate baseline predictions (one or all of B1-B7)
#   B. Post-hoc rescore through cycle-3 verifier (computes 9 signals + CHM)
#   C. Statistical analysis (McNemar + bootstrap BCa + Holm; via
#      scripts/baseline_sig_tests.py — invoked separately by the user)
#
# Estimated wall-time on a single 5090 with eval_batch_size=32:
#   B1 zero-shot:        ~10 min  (gen) + ~62 min (rescore)
#   B2 cot:              ~15 min  (gen) + ~62 min (rescore)
#   B3 rag:              ~65 min  (gen) + ~62 min (rescore)
#   B4 cot_rag:          ~75 min  (gen) + ~62 min (rescore)
#   B5 fiveshot_cot:     ~20 min  (gen) + ~62 min (rescore)
#   B6 vanilla_ft:       ~2 h     (train + per-cycle eval) + ~62 min (rescore)
#   B7 ewc_only_ft:      ~3 h     (train + per-cycle eval, L2 anchor slower) + ~62 min (rescore)
#   Total wall-clock:    ~15 h GPU
#   With overhead + crash recovery margin: ~24 h
# ============================================================================

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH="$PWD"

if [[ -f /venv/main/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi

OUTPUT_DIR="outputs/baselines"
COMPOSITE_PIN="outputs/full_run/cycle_3/composite_calibration.json"
PASSAGE_INDEX="data/passage_index"

mkdir -p "$OUTPUT_DIR"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }
band() { log "==================== $* ===================="; }

# --- Pre-flight checks ---
if [[ ! -f "$COMPOSITE_PIN" ]]; then
    log "FATAL: composite-pin file not found: $COMPOSITE_PIN"
    log "       Cycle 3 may not be closed yet. Wait for C5 to fully close."
    exit 1
fi
if [[ ! -d "$PASSAGE_INDEX" && ! -f "${PASSAGE_INDEX}.faiss" ]]; then
    log "FATAL: passage index not found at $PASSAGE_INDEX"
    exit 1
fi

# --- Inference baselines (B1-B5) ---
run_inference_baseline() {
    local name="$1"
    band "B-INF $name — generate predictions"
    python -m scripts.run_baseline \
        --baseline "$name" \
        --output_dir "$OUTPUT_DIR" \
        --n_questions 300 \
        --eval_batch_size 32 \
        --eval_prefetch \
        --passage_index "$PASSAGE_INDEX" \
        --seed 42 \
        2>&1 | tee -a "$OUTPUT_DIR/$name.gen.log"
}

# --- Training baselines (B6, B7) ---
run_training_baseline() {
    local name="$1"
    local extra_flags="${2:-}"
    band "B-FT $name — 5 cycles training + per-cycle eval"
    # shellcheck disable=SC2086
    python -m scripts.run_simple_ft \
        --baseline_name "$name" \
        --output_dir "$OUTPUT_DIR" \
        --num_cycles 5 \
        --n_eval_per_bench 300 \
        --eval_batch_size 32 \
        --seed 42 \
        $extra_flags \
        2>&1 | tee -a "$OUTPUT_DIR/$name.train.log"
}

# --- Rescoring (all 7 baselines at once, after all gen passes) ---
rescore_all() {
    band "PHASE B — post-hoc rescore through cycle-3 verifier (composite pin: $COMPOSITE_PIN)"
    python -m scripts.rescore_baselines_through_verifier \
        --composite_calibration "$COMPOSITE_PIN" \
        --passage_index "$PASSAGE_INDEX" \
        --baselines zero_shot cot rag cot_rag fiveshot_cot vanilla_ft ewc_only_ft \
        --baseline_dir "$OUTPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$OUTPUT_DIR/rescore.log"
}

# --- Dispatch ---
TARGET="${1:-all}"

case "$TARGET" in
    zero_shot|cot|fiveshot_cot|rag|cot_rag)
        run_inference_baseline "$TARGET"
        ;;
    vanilla_ft)
        run_training_baseline "vanilla_ft"
        ;;
    ewc_only_ft)
        run_training_baseline "ewc_only_ft" "--use_l2_anchor --use_mmlu_guard"
        ;;
    rescore)
        rescore_all
        ;;
    all)
        # Run all 5 inference baselines, then both training baselines,
        # then rescore everything in one pass. Sequential on a single GPU.
        for b in zero_shot cot fiveshot_cot rag cot_rag; do
            run_inference_baseline "$b"
        done
        run_training_baseline "vanilla_ft"
        run_training_baseline "ewc_only_ft" "--use_l2_anchor --use_mmlu_guard"
        rescore_all
        band "ALL BASELINES + RESCORE COMPLETE"
        log "Next: python -m scripts.baseline_sig_tests --baseline_dir $OUTPUT_DIR"
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: bash scripts/launch_baselines.sh [zero_shot|cot|fiveshot_cot|rag|cot_rag|vanilla_ft|ewc_only_ft|rescore|all]"
        exit 2
        ;;
esac
