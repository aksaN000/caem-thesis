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

# CUDA env hardening (2026-05-16 added after B1 OOM caused by stale context
# from prior killed CAEM trajectory runner).
#
# CAEM_FORCE_GPU_CLEANUP=1: caem.model_loader reaps stale CUDA processes
# before each baseline's pipeline loads, preventing stale-context leaks
# from contaminating new VRAM allocation. Critical when baselines launch
# immediately after a killed trajectory runner.
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: lets the allocator
# recycle reserved-but-unallocated blocks across compiled-backward and
# anchor allocations. Without this, the 32 GiB envelope fragments under
# torch.compile and small allocations can OOM with ~600 MiB reserved-but-
# unused (matches the C4 abort window OOM pattern from run_phase1a.sh).
export CAEM_FORCE_GPU_CLEANUP="${CAEM_FORCE_GPU_CLEANUP:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
    # 2026-05-16 VRAM fix: vanilla_ft OOMed at bs=16 during layernorm forward
    # (30.66 GiB allocated, 4.25 MiB free at fail point). Reduced to bs=4 with
    # grad_accum_steps=8 to keep effective batch 32. eval_batch_size also
    # reduced from 32 to 8 in case per-cycle eval OOMs on the same envelope.
    python -m scripts.run_simple_ft \
        --baseline_name "$name" \
        --output_dir "$OUTPUT_DIR" \
        --num_cycles 5 \
        --n_eval_per_bench 300 \
        --batch_size 4 \
        --grad_accum_steps 8 \
        --eval_batch_size 8 \
        --seed 42 \
        $extra_flags \
        2>&1 | tee -a "$OUTPUT_DIR/$name.train.log"
}

# --- Rescoring (all baselines, after all gen passes) ---
rescore_all() {
    band "PHASE B — post-hoc rescore through cycle-3 verifier (composite pin: $COMPOSITE_PIN)"
    python -m scripts.rescore_baselines_through_verifier \
        --composite_calibration "$COMPOSITE_PIN" \
        --passage_index "$PASSAGE_INDEX" \
        --baselines zero_shot cot rag cot_rag fiveshot_cot flare semantic_entropy self_rag vanilla_ft ewc_only_ft \
        --baseline_dir "$OUTPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$OUTPUT_DIR/rescore.log"
}

# --- Runtime-sorted execution order (2026-05-16) ---
# Sorted by expected wall-time, fastest to slowest. Sequential on single GPU.
# Estimates assume eval_batch_size=32 and the 300-sample held-out fold.
#
#   1. B1 zero_shot         ~10 min  inference, no retrieval, no CoT
#   2. B2 cot               ~15 min  inference, +CoT trigger
#   3. B5 fiveshot_cot      ~20 min  inference, +5 demos
#   4. B3 rag               ~65 min  inference, retrieval + generate
#   5. B4 cot_rag           ~75 min  inference, retrieval + CoT
#   6. B9 flare             ~90 min  inference, multi-pass retrieval
#   7. B6 vanilla_ft        ~2 h    train 5 cycles + eval
#   8. B8 semantic_entropy  ~2-3 h  inference, 10× sampling
#   9. B7 ewc_only_ft       ~3 h    train 5 cycles + L2 anchor + eval
#   10. B10 self_rag (FT)   blocked  awaiting synthetic data + FT pipeline
#
# Total B1-B9 sequential: ~10 h GPU + ~7 h rescore = ~17 h GPU (~24 h wall-clock).
# B10 staged separately; build pipeline ~7-11 engineering days + ~30 GPU-h.

# --- Dispatch ---
TARGET="${1:-all}"

case "$TARGET" in
    zero_shot|cot|fiveshot_cot|rag|cot_rag|flare|semantic_entropy|self_rag)
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
    b1_to_b9|all)
        # Run B1-B9 in runtime-sorted order (fastest to slowest).
        # B10 (Self-RAG full FT) excluded — pipeline still being built.
        # Sequential on single GPU. Each baseline finishes before next starts.
        band "Launching B1-B9 panel in runtime-sorted order (fastest -> slowest)"
        log "Order: zero_shot -> cot -> fiveshot_cot -> rag -> cot_rag -> flare -> vanilla_ft -> semantic_entropy -> ewc_only_ft"

        run_inference_baseline "zero_shot"        # ~10 min
        run_inference_baseline "cot"              # ~15 min
        run_inference_baseline "fiveshot_cot"     # ~20 min
        run_inference_baseline "rag"              # ~65 min
        run_inference_baseline "cot_rag"          # ~75 min
        run_inference_baseline "flare"            # ~90 min
        run_training_baseline  "vanilla_ft"       # ~2 h
        run_inference_baseline "semantic_entropy" # ~2-3 h
        run_training_baseline  "ewc_only_ft" "--use_l2_anchor --use_mmlu_guard"  # ~3 h

        rescore_all
        band "B1-B9 PANEL + RESCORE COMPLETE"
        log "Next: python -m scripts.baseline_sig_tests --baseline_dir $OUTPUT_DIR"
        log "Note: B10 (Self-RAG full FT) is on a separate build track; see"
        log "  scripts/generate_self_rag_synthetic_data.py + scripts/train_self_rag.py"
        ;;
    self_rag_ft_build_status)
        band "Self-RAG full-FT pipeline status check"
        log "Stage 1 (synthetic data gen): scripts/generate_self_rag_synthetic_data.py — SCAFFOLD"
        log "Stage 2 (full FT):            scripts/train_self_rag.py — SCAFFOLD"
        log "Stage 3 (baseline class):     eval/baselines.py:SelfRAGPromptAdaptedBaseline (placeholder; will be replaced)"
        log "Engineering remaining: ~7-11 days. See PRODUCTION_NEXT_SESSION_PLAN.md Phase 1.6 Step P-2."
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: bash scripts/launch_baselines.sh [TARGET]"
        echo "  TARGETs:"
        echo "    zero_shot | cot | fiveshot_cot | rag | cot_rag | flare    (individual inference)"
        echo "    semantic_entropy | self_rag                                (individual inference, new)"
        echo "    vanilla_ft | ewc_only_ft                                   (individual training)"
        echo "    rescore                                                    (post-hoc rescore all)"
        echo "    b1_to_b9 | all                                             (full panel B1-B9 sequential)"
        echo "    self_rag_ft_build_status                                   (check B10 FT pipeline progress)"
        exit 2
        ;;
esac
