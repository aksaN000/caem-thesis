#!/usr/bin/env bash
# Phase 1 Full — 8-variant ablation sweep (direct confirmatory, no screening).
#
# 2026-04-22 decision: Phase 1 Full scope is hand-picked 8 ablations, each at
# 5 cycles × n_sil=5000. No screening sweep (Step 16) and no aggregate-of-
# screening (Step 17) — variants are pre-committed as load-bearing, so we go
# direct to confirmatory scale.
#
# Prerequisites:
#   - run_phase1a.sh must have completed outputs/full_run/ (Step 7 main).
#   - outputs/cold_start_memory/ present (Step 6 seed).
#   - data/passage_index/ FAISS mmapped (21M Wikipedia passages).
#   - rclone configured with 'gdrive:' remote if CAEM_GDRIVE_OFFLOAD=1.
#
# Usage:
#   tmux new -d -s ablations './run_phase1_full_ablations.sh'
#   tmux attach -t ablations
#
# Cost: ~$55 @ $0.638/hr (8 variants × ~10h each = 80 GPU-h).

set -u
cd "$(dirname "$(readlink -f "$0")")"

RUNNER_LOG="outputs/ablation/phase1_full_runner.log"
mkdir -p outputs/ablation

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log()  { printf "[%s] %s\n" "$(ts)" "$*" | tee -a "$RUNNER_LOG"; }
band() { log "==================== $* ===================="; }

# 2026-04-22 credit-tuned glibc allocator (same knobs as run_phase1a_hardened).
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_MMAP_THRESHOLD_=131072
export PYTHONMALLOC=malloc

# Benchmarks: ASQA replaces NQ (Branch C 2026-04-22 swap).
BENCHMARKS=(fever triviaqa asqa truthfulqa strategyqa arc_challenge)

# Hand-picked 8 variants — each tied to a specific claim defense or design
# choice. DO NOT reorder without updating branch_C_log.md.
VARIANTS=(
    # 3 ablations after claim-vs-metric audit (2026-04-22). Each defends a
    # numbered thesis Claim that CANNOT be answered by a directly-measured
    # metric:
    no_retroverify            # Claim 2 (purity time-dim) + Claim 3 (across-cycle): memory hygiene
    no_self_improvement       # Claim 3 (SIL upper bound): Cycle-0 only
    no_forgetting_guard       # Claim 4 (no catastrophic forgetting): MMLU rollback disarmed
    # --- Claims defended by direct measurement, no ablation needed ---
    # Reference row `full` -> outputs/full_run/ from Phase 1a Step 7 main
    #   is the anchor at the same cycle count x*.
    # Claim 1 (memory routing) -> tier_{1,2,3}_frac per cycle in summary CSV
    # Claim 2 (purity α>½)     -> Step 19 purity_validation/theory_validation.json
    # Claim 5 (modularity)     -> NLIJudgeInterface protocol +
    #                             Step 5.5.2 v5 minicheck_vs_roberta_v5.json (3-way)
    # Ch3 Eq 3.5 weighting     -> Step 19.5 9-signal correlation matrix +
    #                             methods-section per-signal literature priors
    # Branch C Goal 2 contrib  -> same symmetry: Step 19.5 correlation matrix
    #                             shows q_a_relevance non-redundant; methods
    #                             section narrates the sample-② motivation
)

# Phase 1 Full uses 5 cycles per variant (enough for divergence, half cost of 10).
# retroverify ablation can be extended to 10 cycles if not divergent by 5 — the
# adaptive policy lives outside this wrapper; manually re-run with --max_cycles 10
# if needed.
PHASE1_FULL_CYCLES=5
PHASE1_FULL_N_SIL=5000
PHASE1_FULL_N_EVAL=500

_run_variant() {
    local v="$1"
    local outdir="outputs/ablation/${v}/seed_42"

    # Idempotency: skip if this variant's seed_42 already has experiment_summary.csv
    if [[ -f "${outdir}/experiment_summary.csv" ]]; then
        local rows
        rows=$(wc -l < "${outdir}/experiment_summary.csv" 2>/dev/null || echo 0)
        # header + PHASE1_FULL_CYCLES rows (cycles 0..4) = 6 rows
        if (( rows >= PHASE1_FULL_CYCLES + 1 )); then
            log "Variant ${v}: already complete (${rows} rows) — skipping"
            return 0
        fi
    fi

    band "Variant ${v} — ${PHASE1_FULL_CYCLES} cycles × n=${PHASE1_FULL_N_SIL}"
    CAEM_BATCH_U_TOK_DROP=1 CAEM_GDRIVE_OFFLOAD=1 python -m scripts.run_cyclic_ablation \
        --variant "$v" \
        --seed 42 \
        --max_cycles "$PHASE1_FULL_CYCLES" \
        --n_sil_per_cycle "$PHASE1_FULL_N_SIL" \
        --n_eval_per_benchmark "$PHASE1_FULL_N_EVAL" \
        --benchmarks "${BENCHMARKS[@]}" \
        --passage_index data/passage_index \
        --cold_start_memory outputs/cold_start_memory/memory_store \
        --eval_batch_size 32 \
        --eval_prefetch \
        --output_dir outputs/ablation \
        2>&1 | tee "outputs/ablation/${v}_phase1_full.log"
}

main() {
    band "Phase 1 Full ablation sweep starting at $(ts)"
    log "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
    log "Variants (${#VARIANTS[@]}): ${VARIANTS[*]}"
    log "Scope: ${PHASE1_FULL_CYCLES} cycles × n=${PHASE1_FULL_N_SIL} per variant"

    # Guard: Phase 1a must have produced the thresholds and cold-start memory.
    if [[ ! -f outputs/cold_start_memory/memory_store.faiss ]]; then
        log "FATAL: outputs/cold_start_memory/memory_store.faiss missing — run Phase 1a first."
        exit 2
    fi
    if [[ ! -f outputs/full_run/experiment_summary.csv ]]; then
        log "WARNING: outputs/full_run/experiment_summary.csv missing — Phase 1a Step 7 main not complete."
        log "         Ablations CAN still run (they re-seed from cold-start memory), but Chapter 5"
        log "         reference row 'full' comes from outputs/full_run/. Proceeding anyway."
    fi

    for v in "${VARIANTS[@]}"; do
        _run_variant "$v" || log "Variant ${v} returned non-zero — continuing to next"
    done

    band "Phase 1 Full ablations loop complete at $(ts)"

    # Aggregate into the Chapter-5 ablation table.
    band "Aggregating ablation outputs"
    python scripts/aggregate_ablation.py --output_dir outputs/ablation \
        2>&1 | tee -a "$RUNNER_LOG" || \
        log "aggregate_ablation.py returned non-zero; inspect outputs/ablation/ manually."

    band "Phase 1 Full COMPLETE at $(ts) — see outputs/ablation/ablation_table.csv"
}

main "$@"
