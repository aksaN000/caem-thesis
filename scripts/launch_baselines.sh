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
#   B7 ewc_only_ft:      SKIPPED 2026-05-16 (L2 anchor fp32-cast on 3B params is
#                        intractable on 32 GiB envelope — even after bs=8 +
#                        8-bit AdamW + grad_ckpt the anchor sum OOMs. Defensible
#                        as ablation-style baseline; Ch5 §sec:comp-ewc-ft footnote
#                        registers this scope decision.)
#   Total wall-clock:    ~12 h GPU (B7 skip frees ~3 h)
#   With overhead + crash recovery margin: ~20 h
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
    # 2026-05-16 VRAM fix (4th iteration — final config):
    # Even with 8-bit AdamW + grad_ckpt, bs=32 OOMs at the logits tensor:
    # bs × seq × vocab = 32 × 512 × 151936 × 2 bytes = 4.97 GiB for logits
    # alone, plus shift_logits + loss intermediates push past the 32 GiB
    # envelope (observed: 26.4 GiB in use when forward asks for +6.11 GiB).
    # B7 EWC additionally OOMs in the L2 anchor (fp32 cast on 3B params).
    # Fix: drop physical batch 32→8, raise grad_accum 1→4 (effective batch
    # 32 preserved). At bs=8 logits is 1.24 GiB and total fits ~22-24 GiB.
    # Final config:
    #   - batch_size=8, grad_accum=4 (effective batch 32, identical SGD update)
    #   - 8-bit AdamW (run_simple_ft.py patch)
    #   - gradient_checkpointing (run_simple_ft.py patch)
    #   - eval_batch_size=8 (conservative; reduces eval-side OOM risk)
    #   - n_train_per_bench=700 (matches CAEM SIL pool size for sample-budget
    #     parity; total ~10.5K vs CAEM's ~9.3K)
    python -m scripts.run_simple_ft \
        --baseline_name "$name" \
        --output_dir "$OUTPUT_DIR" \
        --num_cycles 5 \
        --n_train_per_bench 700 \
        --n_eval_per_bench 300 \
        --batch_size 8 \
        --grad_accum_steps 4 \
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
        --baselines zero_shot cot rag cot_rag fiveshot_cot flare semantic_entropy self_rag vanilla_ft \
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
#   7. B6 vanilla_ft        ~1.5 h  train 5 cycles + eval (bs=8 + 8-bit AdamW)
#   8. B8 semantic_entropy  ~2-3 h  inference, 10× sampling
#   9. B7 ewc_only_ft       SKIPPED 2026-05-16 (L2 anchor OOM, see header)
#   10. B10 self_rag (FT)   blocked  awaiting synthetic data + FT pipeline
#
# Total B1-B8 sequential: ~7 h GPU + ~7 h rescore = ~14 h GPU (~20 h wall-clock).
# B7 staged for retry on Qwen-7B QLoRA in Phase 1.7 (memory-friendlier anchor).
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
        # 2026-05-16: B7 dropped from the auto-chain because the EWC L2 anchor
        # sum (p.float() over 3B params) cannot fit alongside the bs=8 forward
        # state on the 32 GiB envelope. This individual dispatch is retained
        # for future re-attempt on a memory-friendlier backbone (Qwen-7B QLoRA
        # in Phase 1.7, or larger-VRAM GPU). It is no longer reached by `all`.
        log "WARNING: B7 ewc_only_ft is the OOM-prone training baseline."
        log "  L2 anchor needs ~12 GiB fp32-cast scratch on top of forward state."
        log "  Iteration history (2026-05-16): bs=16, bs=4 grad_accum=8,"
        log "  bs=32 + 8-bit AdamW + grad_ckpt all OOMed. If retrying, drop to"
        log "  bs=4 grad_accum=8 AND wrap the L2 loop to accumulate in fp16,"
        log "  OR use a 80 GiB GPU."
        run_training_baseline "ewc_only_ft" "--use_l2_anchor --use_mmlu_guard"
        ;;
    rescore)
        rescore_all
        ;;
    b1_to_b9|all)
        # Run B1-B8 in runtime-sorted order (fastest to slowest).
        # B7 (ewc_only_ft) DROPPED 2026-05-16 — L2 anchor fp32-cast on 3B params
        # exceeds the 32 GiB envelope (see header + ewc_only_ft dispatch case).
        # B10 (Self-RAG full FT) excluded — pipeline still being built.
        # Sequential on single GPU. Each baseline finishes before next starts.
        band "Launching B1-B8 panel in runtime-sorted order (B7 skipped, B10 deferred)"
        log "Order: zero_shot -> cot -> fiveshot_cot -> rag -> cot_rag -> flare -> vanilla_ft -> semantic_entropy"

        run_inference_baseline "zero_shot"        # ~10 min
        run_inference_baseline "cot"              # ~15 min
        run_inference_baseline "fiveshot_cot"     # ~20 min
        run_inference_baseline "rag"              # ~65 min
        run_inference_baseline "cot_rag"          # ~75 min
        run_inference_baseline "flare"            # ~90 min
        run_training_baseline  "vanilla_ft"       # ~1.5 h
        run_inference_baseline "semantic_entropy" # ~2-3 h

        rescore_all
        band "B1-B8 PANEL + RESCORE COMPLETE (B7 SKIPPED)"
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
