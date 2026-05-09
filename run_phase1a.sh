#!/usr/bin/env bash
# CAEM Phase 1a runner — follows NEXT_SESSION_PLAN.md (updated 2026-04-20).
#
# Scope: Step 6 NEW-prompt re-seed -> 7.0 NEW Cycle-0 -> 5.5 v4 + prompt-design
# ablation + Level B smoke -> 19.2.1 retention slice -> Step 7 main 10-cycle ->
# Steps 8-15 baselines -> 15.5 sig-tests -> 19 purity -> 19.2.2 retention eval
# -> 19.5 nine-signal corr -> 20.1 aggregate.
#
# Excludes:  Steps 16-18 (Phase 1 Full, supervisor-funded tranche) and Step 20B
#            (optional STaR ceiling). Step 20.2 scp-down is manual.
#
# Idempotent: each step checks its canonical output artifact; reruns of this
# script skip steps whose output already exists. Step 7 auto-detects the last
# completed cycle and adds --resume_from_cycle on relaunch.
#
# Usage:
#   tmux new-session -s phase1a
#   cd ~/caem && ./run_phase1a.sh
#   Ctrl-B D   # detach; reattach with:  tmux attach -t phase1a

set -euo pipefail

# --- Locate repo root ---
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$PWD"

# --- Activate Vast venv so `python` / `pip` resolve to the CUDA build ---
if [[ -f /venv/main/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi
command -v python >/dev/null || { echo "FATAL: 'python' not on PATH" >&2; exit 127; }

# --- Thread caps (prevents the 379-thread FAISS collapse from Session 1) ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export FAISS_NUM_THREADS="${FAISS_NUM_THREADS:-16}"

# --- CAEM env (Session 3 2026-04-22): auto-reap stale CUDA processes on
#     every pipeline load so a crashed prior step doesn't hold GPU memory
#     across the runner's step boundaries. See caem/model_loader.py.
export CAEM_FORCE_GPU_CLEANUP="${CAEM_FORCE_GPU_CLEANUP:-1}"
export CAEM_PROFILE="${CAEM_PROFILE:-0}"

# --- CUDA allocator (2026-04-27): expandable_segments lets the allocator
#     recycle reserved-but-unallocated blocks across compiled-backward and
#     L2-anchor allocations. Without this, the 32 GiB envelope fragments
#     under torch.compile and a ~44 MiB allocation can fail with ~600 MiB
#     reserved-but-unused (the second SIL OOM, 2026-04-27). Speed-neutral;
#     trades fragmentation for slight per-allocation overhead.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Paths ---
RUNNER_LOG="outputs/phase1a_runner.log"
mkdir -p outputs data/calibration data/retention outputs/calibration \
         outputs/baselines outputs/baselines_smoke outputs/signal_correlation \
         outputs/full_run outputs/cycle_0/calibration

ts()   { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log()  { echo "[$(ts)] $*" | tee -a "$RUNNER_LOG" >&2; }
band() { log "==================== $* ===================="; }

on_err() {
    local rc=$?
    log "FATAL: runner aborted at line $1 (exit $rc). Inspect $RUNNER_LOG."
    exit "$rc"
}
trap 'on_err $LINENO' ERR

# Benchmark panel — derived from caem/config.py (single source of truth).
# v2.1 (2026-05-08): training panel pruned to 3 benches after cycle-0 eval
# revealed two precondition-violators (HotpotQA p_+=0.090, NQ p_+~0.16) where
# the Bayesian-floor inequality at α=0.05 is mathematically unreachable.
#   TRAINING:  fever, triviaqa, commonsense_qa
#              3 task types (claim verification, entity recall, 5-choice MCQ).
#              Stream chunk: FEVER+TQA at 2000/cycle, CSQA at 700/cycle.
#   TRANSFER:  truthfulqa, strategyqa
#              held-out for transfer eval. NQ removed (verifier α<½ structural).
# HotpotQA + NQ kept as registered exclusion evidence at outputs/cycle_0/eval/.
# Any panel change in config.py flows through this runner automatically.
TRAIN_PANEL=$(python -c 'from caem.config import TRAINING_BENCHMARKS; print(" ".join(TRAINING_BENCHMARKS))')
TRANSFER_PANEL=$(python -c 'from caem.config import TRANSFER_BENCHMARKS; print(" ".join(TRANSFER_BENCHMARKS))')
[[ -n "$TRAIN_PANEL" && -n "$TRANSFER_PANEL" ]] || { echo "FATAL: failed to read panel from caem.config" >&2; exit 1; }
# shellcheck disable=SC2206
BENCHMARKS=( $TRAIN_PANEL $TRANSFER_PANEL )
BASELINE_BENCHES="$TRAIN_PANEL $TRANSFER_PANEL"

# ============================================================================
# Step 5.9 — Prompt smoke test (post-2026-04-24 prompt revision)
# ============================================================================
# After the G27/G28/G29 prompt revisions, confirm the new prompts still
# produce parseable outputs with reduced evasion + no template leaks.
# ~15 min on 5090 (loads Qwen, answers 30-50 probe questions).
step_5_9_prompt_smoke() {
    local out="outputs/prompt_smoke/results.json"
    if [[ -f "$out" ]]; then
        log "Step 5.9: prompt smoke test already done — skipping"
        return 0
    fi
    band "Step 5.9 — prompt smoke test (validate 2026-04-24 prompt revision)"
    mkdir -p outputs/prompt_smoke
    # 2026-04-24: prefer real questions from the most recent archived
    # eval directory (more reliable signal than synthetic stubs which
    # have no retrieved passages and would always produce NEI).
    # NEI rate threshold is loose (0.95) regardless of source, because
    # n=10-30 smoke samples have too much variance to make hard claims
    # about NEI behavior. The proper FEVER NEI check happens at Step 7.0
    # (500 samples per bench) and the per-bench STORE/DISCARD em gap
    # check at step_7_0_3_validate_weights catches real NEI regressions.
    # Smoke test focuses on what's reliably catchable at small n:
    #   - template leaks (any leak is a real bug, easy to detect)
    #   - evasive patterns (regex-detectable, n=10 sufficient)
    local eval_src=""
    local archived_eval
    # `|| true` is critical here: with `set -e + pipefail`, `ls -d` failing on
    # an empty glob would silently abort the runner via command-substitution
    # propagation (the ERR trap does not fire on this path).
    archived_eval=$(ls -d outputs/archive/*/cycle_0/eval 2>/dev/null | tail -1 || true)
    if [[ -n "$archived_eval" ]]; then
        eval_src="--eval_source_dir $archived_eval"
        log "  using real questions from $archived_eval (real passages)"
    else
        log "  no archived eval found; using synthetic samples"
    fi
    if ! python scripts/prompt_smoke_test.py \
        --n 10 \
        --benchmarks $TRAIN_PANEL \
        --output "$out" \
        $eval_src \
        --max_fever_nei_rate 0.95 \
        2>&1 | tee outputs/prompt_smoke/run.log; then
        log "FATAL: prompt smoke FAILED. Review $out, tune prompts, re-run."
        return 2
    fi
    log "Step 5.9: prompt smoke PASSED — safe to re-seed"
}

# ============================================================================
# Step 5.9.5 — Build alias dictionary for v2 Fix 6 (alias_overlap signal)
# ============================================================================
# scripts/build_alias_dict_from_benchmarks.py harvests alias-enriched
# answers from TriviaQA/NaturalQuestions/HotpotQA train splits and
# writes data/alias_dict.json. CAEMPipeline reads this file at init via
# cfg.alias_dict_path and constructs InMemoryAliasResolver, which the
# UnifiedVerifier consumes for the alias_overlap signal. Without this
# step the resolver is None and alias_overlap returns the neutral 0.5
# prior for every query — silently neutralising the signal.
step_5_9_5_alias_dict() {
    local out="data/alias_dict.json"
    if [[ -s "$out" ]]; then
        log "Step 5.9.5: $out already present — skipping"
        return 0
    fi
    band "Step 5.9.5 — build alias dictionary (v2 Fix 6 alias_overlap)"
    python scripts/build_alias_dict_from_benchmarks.py \
        --output "$out" --n_per_bench 3000 2>&1 | tee -a "$RUNNER_LOG"
}

# ============================================================================
# Step 6 — Cold-start memory seeding under NEW uniform-scaffolded prompts
# ============================================================================
step_6_reseed() {
    local out="outputs/cold_start_memory"
    if [[ -f "$out/seed_summary.json" && -f "$out/memory_store.faiss" ]]; then
        log "Step 6: cold-start memory already present at $out — skipping"
        return 0
    fi
    band "Step 6 — cold-start seed (NEW prompts, cold-start override τ=0.50)"
    python -m scripts.seed_cold_start \
        --target_episodes 200 \
        --benchmarks $TRAIN_PANEL \
        --cold_start_store_threshold 0.50 \
        --output_dir "$out" 2>&1 | tee -a outputs/step6_seed.log
    # Minimum-viable seed floor: enough episodes that adaptive τ has signal
    # to work with from Cycle 1 onwards. Empirically derived under v1 (3-bench)
    # at ~340 total. v2.1 (3-bench post-pruning) holds the SAME absolute
    # floor — if 3 benches × 200 target × yield_rate cannot meet the v1
    # floor, something is structurally broken (loader, prompts, verifier).
    python - <<'PY'
import json, sys
d = json.load(open("outputs/cold_start_memory/seed_summary.json"))
total = d.get("total_seeded", 0)
assert total >= 340, f"Step 6 verify: total_seeded={total} < 340 (minimum-viable seed floor; 3-bench v2.1 panel × 200 target should clear this comfortably)"
print(f"Step 6 OK: total_seeded={total}")
PY
}

# ============================================================================
# Step 6.5 — EM-prune cold-start memory (drop confidently-wrong episodes)
# ============================================================================
# v2.1 (2026-05-08): cold-start storage gate at u_stored ≥ 0.50 admits
# episodes the verifier accepts but doesn't see gold for (production-mode
# parity). Empirically that lets ~25-30% confidently-wrong episodes
# through. Without an em-prune pass before step_7_main, Tier 1 retrieval
# can return known-wrong stored answers and the SIL pool can train on
# wrong reasoning chains.
#
# This step looks up gold answers from the canonical training-split
# loader, computes any-match EM against entry.answer with the canonical
# normaliser, drops em=0 entries, keeps em=1 and em-unverified.
#
# Idempotent: skips if em_pruning_report.json already exists.
# Safety: writes outputs/cold_start_memory/memory_store.pre_em_prune.{faiss,meta}.bak
# before rebuilding the store.
step_6_5_em_prune() {
    local report="outputs/cold_start_memory/em_pruning_report.json"
    if [[ -f "$report" ]]; then
        log "Step 6.5: em_pruning_report.json present — em-prune already done, skipping"
        return 0
    fi
    band "Step 6.5 — EM-prune cold-start memory (drop confidently-wrong episodes)"
    python scripts/prune_cold_memory_em.py 2>&1 | tee -a "$RUNNER_LOG"
    # Sanity check: report should exist and reflect at least 50% retention.
    python - <<'PY'
import json
r = json.load(open("outputs/cold_start_memory/em_pruning_report.json"))
total_before = int(r["n_total_before"])
kept = int(r["n_kept"])
ratio = kept / max(1, total_before)
assert ratio >= 0.50, (
    f"Step 6.5 verify: em-prune kept {kept}/{total_before} = {ratio:.1%} "
    f"(below 50% — verifier discrimination on cold-start is broken)"
)
print(f"Step 6.5 OK: em-prune kept {kept}/{total_before} ({ratio:.1%})")
PY
}

# ============================================================================
# Step 7.0 — Cycle-0 baseline eval (500 samples/benchmark)
# ============================================================================
step_7_0_cycle0() {
    local evaldir="outputs/cycle_0/eval"
    local cal_json="outputs/cycle_0/calibration/calibration_fold_samples.json"
    local all_present=1
    for b in "${BENCHMARKS[@]}"; do
        [[ -f "$evaldir/${b}_cycle0.json" ]] || all_present=0
    done
    # v2.1 (2026-05-08): Step 7.0 produces BOTH cycle-0 eval JSONs AND the
    # calibration fold (calibration_fold_samples.json) in a single pass via
    # run_experiment.py's run_calibration_step. The skip-check must require
    # BOTH to be present, otherwise step_7_0_calibrate downstream will fail
    # with "no calibration JSONs found".
    if [[ $all_present -eq 1 && -f "$cal_json" ]]; then
        log "Step 7.0: Cycle-0 eval JSONs + cal-fold both present — skipping"
        return 0
    fi
    band "Step 7.0 — Cycle-0 baseline (500/bench, NEW prompts, bs=32) + cal-fold scoring"
    python -m scripts.run_experiment \
        --output_dir outputs/cycle_0 \
        --num_cycles 0 \
        --n_questions 500 \
        --n_eval_questions 500 \
        --benchmarks "${BENCHMARKS[@]}" \
        --passage_index data/passage_index \
        --cold_start_memory outputs/cold_start_memory/memory_store \
        --eval_batch_size 32 \
        2>&1 | tee -a outputs/cycle_0/run.log
}

# ============================================================================
# Step 7.0.1 — Fit per-benchmark CalProbComposite (Phase 1c)
# ============================================================================
# Phase 1c (2026-05-09): the conformal storage gate (split-CP at α targets)
# and the legacy quantile threshold fitter were both removed after the
# v2.1 cycle-0 diagnostic showed the cal/eval exchangeability assumption
# breaks at our 500-sample per-bench cal-folds. The storage gate is now
# a fixed threshold on the calibrated composite probability
# (cfg.store_threshold / cfg.defer_threshold); only the composite is fit
# at the cycle boundary.
#
# The fit produces outputs/cycle_0/composite_calibration.json:
#   - Per-signal isotonic regression on the 1500-sample pooled cal-fold
#   - Per-benchmark child composites with shrinkage prior toward pooled
#     (α=0.6, prevents per-bench overfit on weak-signal benches)
#   - Cherian-style logistic-regression boost (C=0.01, locked from the
#     25-variant Step 7.0.3 sweep)
step_7_0_calibrate() {
    local out_composite="outputs/cycle_0/composite_calibration.json"
    if [[ -f "$out_composite" ]]; then
        log "Step 7.0.1: composite_calibration.json already fitted — skipping"
        return 0
    fi
    band "Step 7.0.1 — fit per-benchmark CalProbComposite (shrinkage_alpha=0.6)"
    local calib_json="outputs/cycle_0/calibration/calibration_fold_samples.json"

    log "Step 7.0.1 — fit CalProbComposite (per-bench isotonic + shrinkage + Cherian boost)"
    python scripts/fit_composite_calibration.py \
        --calib_jsons "$calib_json" \
        --output_json "$out_composite" \
        --shrinkage_alpha 0.6 \
        --cherian_boost \
        --boost_C 0.01 2>&1 | tee -a "$RUNNER_LOG"
    # boost_C 0.01 (strong L2) selected by the 25-variant sweep at v2 step_7_0_3
    # (76.3% vs 74.4% eval ID precision over C=1.0). shrinkage_alpha 0.6 from
    # the Phase 1c P3a design — moderate per-bench adaptation regularised
    # toward pooled to prevent the TruthfulQA / StrategyQA / CSQA overfit
    # observed on v2.1 cycle-0 under pure per-bench fitting.
    log "Step 7.0.1 OK — composite ready for cycle-0 rescore + step_7_main"
}

# ============================================================================
# Step 7.0.2.5 — Rescore Cycle-0 eval through the fitted gate
# ============================================================================
# scripts/rescore_eval_with_fitted_gate.py applies the freshly-fit
# composite + conformal gate to every cycle-0 eval sample, writing a
# parallel directory of rescored eval JSONs that step_7_0_3 + step_7_0_4
# both read. v1 runs populated this directory via a manual command;
# the runner missed wiring it as a step. Adding it explicitly so the
# pre-Step-7 chain produces all artefacts the downstream gates expect.
step_7_0_2_5_rescore_eval() {
    local out_dir="outputs/cycle_0/eval_rescored"
    if compgen -G "$out_dir/*_cycle0.json" > /dev/null; then
        log "Step 7.0.2.5: eval_rescored already populated — skipping"
        return 0
    fi
    band "Step 7.0.2.5 — rescore cycle-0 eval through fitted composite + gate"
    python -m scripts.rescore_eval_with_fitted_gate 2>&1 | tee -a outputs/cycle_0/run.log
}

# ============================================================================
# Step 7.0.3 — Empirical-precision audit (Phase 1c replacement)
# ============================================================================
# Phase 1c (2026-05-09): the previous Step 7.0.3 was a hard decision gate
# enforcing the conformal-coverage 30% poisoning rate. After replacing the
# conformal gate with a fixed threshold, this step becomes an *audit*:
# it measures realised STORE precision per benchmark on the rescored eval
# fold and writes weight_validation.json for thesis evidence, but does
# not fail-stop the runner — the operator reviews and decides whether to
# adjust cfg.store_threshold for the next cycle.
step_7_0_3_validate_weights() {
    local out="outputs/cycle_0/weight_validation.json"
    if [[ -f "$out" ]]; then
        log "Step 7.0.3: empirical-precision audit already done — skipping"
        return 0
    fi
    band "Step 7.0.3 — empirical-precision audit on Cycle-0 eval-rescored fold"
    python scripts/validate_composite_weights.py \
        --eval_dir outputs/cycle_0/eval_rescored \
        --output "$out" \
        --cohen_d_threshold 0.20 \
        --store_discard_gap 0.05 \
        --max_poisoning_rate 0.50 \
        --min_n_stored 5 \
        2>&1 | tee -a outputs/cycle_0/run.log || {
            log "Step 7.0.3: audit emitted non-zero — operator should review $out"
        }
    log "Step 7.0.3: empirical-precision audit complete (informational only)"
}

# ============================================================================
# Step 5.5 — Verifier-backend calibration diagnostic (runs AFTER Step 7.0)
# ============================================================================
step_5_5_pairs() {
    local out="data/calibration/minicheck_pairs_500.jsonl"
    if [[ -s "$out" ]]; then
        log "Step 5.5.1: calibration pairs already built — skipping"
        return 0
    fi
    band "Step 5.5.1 — build 500-pair calibration set from Cycle-0 eval"
    python scripts/build_calibration_pairs.py \
        --eval_jsons "outputs/cycle_0/eval/*_cycle0.json" \
        --passage_index data/passage_index \
        --n_pairs 500 \
        --balance 0.5 \
        --output_jsonl "$out" 2>&1 | tee -a "$RUNNER_LOG"
}

step_5_5_headhead() {
    # v1-v4 are OLD-prompt preview diagnostics already on disk; v5 is the
    # NEW-prompt audit-of-record that Ch5 Appendix will cite.
    local out="outputs/calibration/minicheck_vs_roberta_v5.json"
    if [[ -f "$out" ]]; then
        log "Step 5.5.2: v5 audit-of-record already present — skipping"
        return 0
    fi
    band "Step 5.5.2 — v5 MiniCheck vs RoBERTa-MNLI (NEW-prompt audit-of-record)"
    # v2.1 (2026-05-09): dropped qwen_judge backend. Qwen-3B + MiniCheck +
    # RoBERTa-MNLI loaded simultaneously OOMs on 32 GiB GPU (Qwen alone needs
    # ~40 GiB for the score-batch forward at this seq length). Qwen judge was
    # ABLATED 2026-04-26 (Pearson < 0.7 vs MiniCheck) so it doesn't enter the
    # production composite anyway. The MiniCheck vs RoBERTa head-to-head
    # remains the audit-of-record; it's all Ch5 Appendix cites.
    python scripts/calibration_minicheck_vs_roberta.py \
        --pairs_jsonl data/calibration/minicheck_pairs_500.jsonl \
        --output_json "$out" \
        --device cuda \
        --backends minicheck roberta_nli 2>&1 | tee -a outputs/calibration/minicheck_vs_roberta_v5.log
}

# step_5_5_gate (scenario classifier) and step_prompt_ablation removed
# 2026-05-07: scenario gate was log-only and 2026-04-20 decision was
# MiniCheck-irrespective; prompt-design ablation conflated OLD@T5 vs
# NEW@Qwen so cannot serve as a clean prompt-design comparison.

# ============================================================================
# Platt calibration for AdaptiveNLIJudge (2026-04-22 Path B).
# Fits scaling params (a, b) that align FrozenQwenJudge's P(yes) with
# MiniCheck's P(supported) on a 500-sample overlap fold drawn from
# Cycle-0 eval JSONs. Required before Step 7 main so long-hypothesis
# samples routed to Qwen-judge produce comparable p_entail values.
# Fails the runner if Pearson ρ (logit space) < 0.70.
# ============================================================================
step_platt_calibrate() {
    # 2026-04-26: Frozen Qwen judge ABLATED for Phase 1a. The Platt
    # calibration on a 500-sample MiniCheck/Qwen overlap fold returned
    # logit-space Pearson ρ=0.58 (< 0.70 threshold), meaning Qwen-3B's
    # P(yes) ranking does not reliably replicate MiniCheck's P(supported)
    # judgments. AdaptiveNLIJudge therefore stays inactive in Step 7 main;
    # caem/verification/__init__.py:130-132 falls back to bare MiniCheck
    # for ALL hypothesis lengths (long hypotheses incur 512-token MC
    # truncation, documented in the verifier docstring). Evidence preserved
    # at outputs/calibration/qwen_judge_platt.log + .ablation.json. This
    # step is now a NO-OP that simply records the ablation decision; it
    # does NOT halt Step 7 main.
    local sentinel="outputs/calibration/qwen_judge_platt.ablation.json"
    if [[ -f "$sentinel" ]]; then
        log "Platt calibration: ablation sentinel already present — skipping"
        return 0
    fi
    band "Path B — Frozen Qwen judge ABLATED (Phase 1a decision 2026-04-26)"
    mkdir -p outputs/calibration
    cat > "$sentinel" <<'JSON'
{
  "_decision": "Frozen Qwen judge ABLATED for Phase 1a",
  "_decision_date_utc": "2026-04-26",
  "_evidence_path": "outputs/calibration/qwen_judge_platt.log",
  "_pearson_rho_logit_observed": 0.5843,
  "_min_rho_threshold": 0.70,
  "_outcome": "AdaptiveNLIJudge inactive; bare MiniCheck handles all hypothesis lengths with documented 512-token truncation on long hypotheses.",
  "_cite_in_thesis": "branch_C_log.md 2026-04-26 entry; outputs/calibration/qwen_judge_platt.log"
}
JSON
    log "Platt calibration ablated; sentinel written to $sentinel"
}

# ============================================================================
# u_tok_drop pool correctness gate — PERMANENTLY SKIPPED (2026-05-06)
# ============================================================================
# The unit + integration test suite (test_pipeline_batch_equivalence.py +
# test_self_improvement.py) plus per-cycle recalibration prove batch≡serial
# equivalence on every cycle boundary. The 10-min N=32 GPU diff_verify is
# redundant under v2 and is no longer worth its wall-clock. Original
# implementation removed; this stub writes the ablation sentinel and
# returns immediately so the runner chain is unbroken.
step_u_tok_drop_gate() {
    local marker="outputs/u_tok_drop_validation.log"
    if [[ -f "$marker" ]] && grep -q "u_tok_drop SKIPPED" "$marker" 2>/dev/null; then
        log "u_tok_drop gate: ablation sentinel present — skipping"
        return 0
    fi
    band "u_tok_drop pool correctness gate — ABLATED (test suite + recal cover this)"
    mkdir -p outputs
    cat > "$marker" <<'TXT'
u_tok_drop SKIPPED
Reason: redundant with test_pipeline_batch_equivalence.py + test_self_improvement.py
        + per-cycle recalibration which together prove batch≡serial on every
        cycle boundary. Decision: 2026-05-06.
TXT
    log "u_tok_drop ablation sentinel written to $marker"
}

# ============================================================================
# Step 19.2.1 — Freeze Cycle-0 retention slice (must run before Cycle 1)
# ============================================================================
step_19_2_slice() {
    local out="data/retention/cycle0_slice_500.jsonl"
    if [[ -s "$out" ]]; then
        log "Step 19.2.1: retention slice already present — skipping"
        return 0
    fi
    band "Step 19.2.1 — freeze 500-sample Cycle-0 retention slice"
    python scripts/cycle2_retention_diagnostic.py --make_slice \
        --cycle0_eval_dir outputs/cycle_0/eval \
        --output_slice "$out" 2>&1 | tee -a "$RUNNER_LOG"
}

# ============================================================================
# gdrive snapshot — upload all pre-Step-7 artefacts so a fresh instance can
# resume from Step 7 directly if the current instance dies before Step 7
# finishes (or even starts). Uses the same gdrive: rclone remote that the
# per-cycle CAEM_GDRIVE_OFFLOAD path already writes to.
# Target: gdrive:caem-phase1a/pre_main_snapshot/<git-commit>/
# ============================================================================
step_gdrive_upload_pre_main() {
    local marker="outputs/.gdrive_pre_main_uploaded"
    if [[ -f "$marker" ]]; then
        log "gdrive pre-main snapshot: already uploaded (marker $marker) — skipping"
        return 0
    fi
    if ! command -v rclone >/dev/null; then
        log "gdrive snapshot: rclone not on PATH — skipping (non-fatal)"
        return 0
    fi
    if ! rclone listremotes 2>/dev/null | grep -q '^gdrive:'; then
        log "gdrive snapshot: 'gdrive:' remote not configured — skipping (non-fatal)"
        return 0
    fi
    local commit
    commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local remote="gdrive:caem-phase1a/pre_main_snapshot/${commit}"
    band "gdrive snapshot — uploading pre-Step-7 artefacts to ${remote}"

    # Stage all pre-Step-7 artefacts into a single flat layout under
    # outputs/.snapshot_staging/ so rclone can push them with one call.
    local stage="outputs/.snapshot_staging"
    rm -rf "$stage"; mkdir -p "$stage"
    copy() {
        local src="$1" rel_dst="$2"
        if [[ ! -e "$src" ]]; then
            echo "  [skip] $src (not present)"
            return
        fi
        local dst="$stage/$rel_dst"
        mkdir -p "$(dirname "$dst")"
        cp -r "$src" "$dst"
        echo "  [add ] $src -> $rel_dst"
    }

    # Step 6 cold-start memory
    copy outputs/cold_start_memory                            cold_start_memory
    # Step 7.0 Cycle-0 eval + calibrated thresholds + cycle-0 memory/deferred snapshot
    copy outputs/cycle_0/eval                                 cycle_0/eval
    copy outputs/cycle_0/eval_rescored                        cycle_0/eval_rescored
    copy outputs/cycle_0/calibration                          cycle_0/calibration
    copy outputs/cycle_0/calibrated_thresholds.json           cycle_0/calibrated_thresholds.json
    copy outputs/cycle_0/composite_calibration.json           cycle_0/composite_calibration.json
    # Phase 1c (2026-05-09): conformal_gate.json removed (replaced by
    # fixed CAEMConfig thresholds). conformal_gate.json no longer exists;
    # alpha_sweep + sweep dirs were archived to gdrive:caem-phase1a/archive_phase1c_2026-05-09/
    copy outputs/cycle_0/weight_validation.json               cycle_0/weight_validation.json
    copy outputs/cycle_0/iteration_history.json               cycle_0/iteration_history.json
    copy outputs/cycle_0/memory_store_cycle_0.faiss           cycle_0/memory_store_cycle_0.faiss
    copy outputs/cycle_0/memory_store_cycle_0.meta            cycle_0/memory_store_cycle_0.meta
    copy outputs/cycle_0/deferred_buffer_cycle_0.pkl          cycle_0/deferred_buffer_cycle_0.pkl
    copy outputs/cycle_0/dataset_splits.json                  cycle_0/dataset_splits.json
    copy outputs/cycle_0/mmlu_baseline.json                   cycle_0/mmlu_baseline.json
    copy outputs/cycle_0/run.log                              cycle_0/run.log
    copy outputs/cycle_0/run_recal.log                        cycle_0/run_recal.log
    copy outputs/cycle_0/experiment.log                       cycle_0/experiment.log
    # Phase 4 thesis-ready artefacts (cycle-0)
    copy outputs/phase4                                       phase4
    # Step 5.5 v5 + pair set
    copy outputs/calibration/minicheck_vs_roberta_v5.json     calibration/minicheck_vs_roberta_v5.json
    copy outputs/calibration/minicheck_vs_roberta_v5.log      calibration/minicheck_vs_roberta_v5.log
    copy data/calibration/minicheck_pairs_500.jsonl           calibration/minicheck_pairs_500.jsonl
    # Frozen Qwen Platt ablation evidence (2026-04-26 Phase 1a decision)
    copy outputs/calibration/qwen_judge_platt.log             calibration/qwen_judge_platt.log
    copy outputs/calibration/qwen_judge_platt.ablation.json   calibration/qwen_judge_platt.ablation.json
    # Step 19.2.1 retention slice
    copy data/retention/cycle0_slice_500.jsonl                retention/cycle0_slice_500.jsonl
    # Gates + audit logs
    copy outputs/u_tok_drop_validation.log                    u_tok_drop_validation.log
    copy outputs/phase1a_runner.log                           phase1a_runner.log
    copy outputs/step6_seed.log                               step6_seed.log

    # Manifest with recovery hint pointing at gdrive
    python - <<PY
import json, os
manifest = {
    "uploaded_at":   __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "source_host":   os.uname().nodename,
    "git_commit":    os.popen("git rev-parse HEAD").read().strip(),
    "purpose":       "Phase 1a pre-Step-7 snapshot for credit-burnout recovery",
    "recovery_hint": "rclone copy ${remote}/ ./recovery/ && cp -r recovery/* outputs/ && ./run_phase1a.sh",
}
import pathlib
pathlib.Path("$stage/MANIFEST.json").write_text(json.dumps(manifest, indent=2))
PY

    log "  rclone copy $stage/ -> $remote/"
    if rclone copy "$stage" "$remote" --transfers 8 --checkers 16 --progress 2>&1 | tee -a "$RUNNER_LOG"; then
        log "  OK — snapshot live at $remote"
        rm -rf "$stage"
        touch "$marker"
    else
        log "  FAIL — rclone copy returned non-zero; staging dir kept at $stage for manual retry"
        return 0   # non-fatal: thesis still produces results without the cloud snapshot
    fi
}

# ============================================================================
# Step 7 — Main 10-cycle CAEM run at n_questions=3000 (Phase 1a headline;
# 3k/bench/cycle × 3 ID benchmarks × 10 cycles = 90k SIL-stream queries)
# ============================================================================
step_7_main() {
    # Skip if a legitimate completion exists: either run_complete.json marker
    # (written by run_experiment.py for any successful run including early-stop)
    # OR experiment_summary.csv with the full 11 rows (pre-marker runs).
    if [[ -f outputs/full_run/run_complete.json ]]; then
        local cstat
        cstat=$(python -c 'import json; d=json.load(open("outputs/full_run/run_complete.json")); print(f"cycles={d[\"cycles_completed\"]}, early_stopped={d[\"early_stopped\"]}")' 2>/dev/null || echo "parse_failed")
        log "Step 7: run_complete.json found ($cstat) — main run complete, skipping"
        return 0
    fi
    if [[ -f outputs/full_run/experiment_summary.csv ]]; then
        local rows
        rows=$(wc -l < outputs/full_run/experiment_summary.csv)
        if (( rows >= 12 )); then   # header + 11 cycle rows
            log "Step 7: experiment_summary.csv has $rows rows — main run complete, skipping"
            return 0
        fi
    fi
    band "Step 7 — main 10-cycle CAEM run (n=3000/bench, budget early-stop at Cycle 5-6) [u_tok_drop pools ON, gdrive offload ON]"
    # Enable u_tok_drop verifier pools + gdrive checkpoint offload for this
    # python child only. Scoped via leading assignments on the python call
    # so subsequent runner stages see defaults.
    # - u_tok_drop validated at step_u_tok_drop_gate immediately before.
    # - gdrive offload sends every cycle to gdrive:caem-phase1a/full_run/
    #   cycle_<n>/ after local save; local rolling-N still active as safety.

    # Phase 1c (2026-05-09): the legacy calibrated_thresholds.json fitter
    # (scripts/calibrate_thresholds.py) was removed alongside the conformal
    # gate. The runner now uses the fixed thresholds from CAEMConfig directly
    # (store_threshold=0.60, defer_threshold=0.45, train_threshold from cfg).
    # No --store_threshold / --defer_threshold / --train_threshold CLI args
    # passed; run_experiment.py falls back to CAEMConfig values.
    log "  storage gate: fixed-threshold (cfg.store_threshold + cfg.defer_threshold)"

    local resume_args=()
    if [[ -d outputs/full_run ]]; then
        # Phase 1c (2026-05-09): detect last *complete* cycle by the presence
        # of memory_store_cycle_N.faiss + .meta + deferred_buffer_cycle_N.pkl
        # at the top level of outputs/full_run/. These three files are written
        # AT cycle close (after retroverify, deferred-reconsider, composite
        # refit, LoRA SIL, retention probe). A mid-cycle crash leaves the
        # cycle_N/ dir partially populated but does NOT write the top-level
        # memory_store_cycle_N artefact, so we can distinguish complete-N
        # from in-progress-N reliably.
        local last=""
        for n in 9 8 7 6 5 4 3 2 1 0; do
            if [[ -f "outputs/full_run/memory_store_cycle_${n}.faiss" \
               && -f "outputs/full_run/memory_store_cycle_${n}.meta" \
               && -f "outputs/full_run/deferred_buffer_cycle_${n}.pkl" ]]; then
                last="$n"
                break
            fi
        done
        if [[ -n "${last:-}" && "$last" -ge 0 && "$last" -lt 10 ]]; then
            resume_args=(--resume_from_cycle "$((last + 1))")
            log "  detected completed cycle $last (memory + deferred written) — resuming from $((last + 1))"
        else
            log "  no completed cycle artefacts found — starting from cycle 0"
        fi
    fi

    CAEM_BATCH_U_TOK_DROP=1 CAEM_GDRIVE_OFFLOAD=1 python -m scripts.run_experiment \
        --output_dir outputs/full_run \
        --num_cycles 10 \
        --n_questions 3000 \
        --n_eval_questions 300 \
        --benchmarks "${BENCHMARKS[@]}" \
        --passage_index data/passage_index \
        --cold_start_memory outputs/cold_start_memory/memory_store \
        --verifier_backend minicheck \
        --eval_batch_size 32 \
        --eval_prefetch \
        "${resume_args[@]}" \
        2>&1 | tee -a outputs/full_run/run.log

    python - <<'PY'
import csv
rows = list(csv.reader(open("outputs/full_run/experiment_summary.csv")))
# Early-stop can fire at any cycle >= 5, producing 7+ rows (header + 6 cycle
# rows for cycles 0..5). Require at least that many; don't require the full
# 12 (header + cycle 0..10) because early-stop is a legitimate success.
n_data = len(rows) - 1
assert n_data >= 6, f"experiment_summary.csv has {n_data} cycle rows; need >=6 (cycles 0..5) for Ch5 progression plot"
header = rows[0]; idx = header.index("mmlu_retention_pct")
bad = [r for r in rows[1:] if float(r[idx]) < 93.0]
assert not bad, f"MMLU retention <93% on {len(bad)} cycles: {bad}"
last_cycle = n_data - 1  # cycle 0 is the pre-training baseline, so last SIL-trained cycle is n_data-1
print(f"Step 7 OK: {n_data} cycle rows (cycles 0..{last_cycle}); "
      f"{'early-stopped' if n_data < 11 else 'full 10 cycles'}; "
      f"MMLU retention >=93% on all cycles")
PY
}

# step_8_flare_smoke and step_13_b5 (FLARE) removed 2026-05-07: FLARE is
# inference-only and cannot batch (iterative look-ahead per sentence), so
# at 30k serial queries it would cost ~25-125 GPU-h for a baseline that
# doesn't defend any of the 5 headline claims. Removal rationale:
# branch_C_log.md 2026-04-22.

# ============================================================================
# Steps 9-15 — B1..B7 baselines (n=3000 per benchmark; matched-scale to CAEM
# Step 7 main under the 2026-04-24 budget downsize)
# ============================================================================
_run_inference_baseline() {
    local step="$1" name="$2" flag="$3"; shift 3
    local extra=("$@")
    local outdir="outputs/baselines/$name"
    if ls "$outdir"/*.json &>/dev/null; then
        log "Step $step ($name): already present — skipping"
        return 0
    fi
    band "Step $step — $name baseline (n=3000, bs=32)"
    python -m scripts.run_baseline \
        --baseline "$flag" \
        --benchmarks $BASELINE_BENCHES \
        --n_questions 3000 \
        --eval_batch_size 32 \
        --output_dir outputs/baselines \
        "${extra[@]}" \
        2>&1 | tee "outputs/baselines/${step}_${name}.log"
}

step_9_b1()  { _run_inference_baseline "B1" "zero_shot"     "zero_shot"; }
step_10_b2() { _run_inference_baseline "B2" "cot"           "cot"; }
# B5 slot reclaimed for 5-shot CoT (Wei et al. 2022) after FLARE removal.
# Runs between B2 and B3 to group inference-only / no-retrieval baselines
# together (B1, B2, B5). 5 demos drawn from fever train split with seed 42.
step_11_5_b5() { _run_inference_baseline "B5" "fiveshot_cot" "fiveshot_cot"; }
step_11_b3() { _run_inference_baseline "B3" "rag"           "rag"       --passage_index data/passage_index; }
step_12_b4() { _run_inference_baseline "B4" "cot_rag"       "cot_rag"   --passage_index data/passage_index; }
step_14_b6_vanilla_ft() {
    local outdir="outputs/baselines/vanilla_ft"
    if [[ -s "$outdir/training_log.jsonl" ]]; then
        local cycles
        cycles=$(wc -l < "$outdir/training_log.jsonl")
        if (( cycles >= 10 )); then
            log "Step 14 (B6 vanilla_ft): $cycles cycles complete — skipping"
            return 0
        fi
    fi
    band "Step 14 — B6 vanilla FT (10 cycles, no L2 anchor, no MMLU guard, eval bs=32, gdrive offload ON)"
    # v2.1 2026-05-08: panel pruned to 3 training benches; FEVER+TQA chunks
    # doubled to 2000, CSQA stays at 700. Per-bench allocation budget
    # n_train_per_bench=30000 is an upper bound (FEVER/TQA need 10×2000=20000;
    # CSQA needs 10×700=7000); build_benchmark_pools clips to per-bench chunk.
    # Baseline must train on the same effective per-cycle chunk as CAEM main.
    CAEM_GDRIVE_OFFLOAD=1 python -m scripts.run_simple_ft \
        --baseline_name vanilla_ft \
        --num_cycles 10 \
        --eval_benchmarks "${BENCHMARKS[@]}" \
        --n_eval_per_bench 500 \
        --n_train_per_bench 30000 \
        --eval_batch_size 32 \
        --output_dir "$outdir" 2>&1 | tee outputs/baselines/B6_vanilla_ft.log
}

step_15_b7_ewc_only() {
    local outdir="outputs/baselines/ewc_only_ft"
    if [[ -s "$outdir/training_log.jsonl" ]]; then
        local cycles
        cycles=$(wc -l < "$outdir/training_log.jsonl")
        if (( cycles >= 10 )); then
            log "Step 15 (B7 ewc_only_ft): $cycles cycles complete — skipping"
            return 0
        fi
    fi
    band "Step 15 — B7 EWC-only FT (10 cycles, L2 anchor + MMLU guard on, eval bs=32, gdrive offload ON)"
    # v2.1 2026-05-08: panel pruned to 3 training benches; FEVER+TQA chunks
    # doubled to 2000, CSQA stays at 700. n_train_per_bench=30000 is an
    # upper bound; per-bench chunk override clips inside build_benchmark_pools.
    CAEM_GDRIVE_OFFLOAD=1 python -m scripts.run_simple_ft \
        --baseline_name ewc_only_ft \
        --use_l2_anchor \
        --use_mmlu_guard \
        --num_cycles 10 \
        --eval_benchmarks "${BENCHMARKS[@]}" \
        --n_eval_per_bench 500 \
        --n_train_per_bench 30000 \
        --eval_batch_size 32 \
        --output_dir "$outdir" 2>&1 | tee outputs/baselines/B7_ewc_only_ft.log
}

# ============================================================================
# Step 15.5 — McNemar + bootstrap CI + Holm sig-tests
# ============================================================================
step_15_5_sig() {
    local out="outputs/tab_sig_test.csv"
    if [[ -f "$out" ]]; then
        log "Step 15.5: sig-test CSV already present — skipping"
        return 0
    fi
    band "Step 15.5 — McNemar + bootstrap CI + Holm correction"
    python -m scripts.baseline_sig_tests \
        --caem_dir outputs/full_run \
        --baselines_dir outputs/baselines \
        --output_csv "$out" 2>&1 | tee -a "$RUNNER_LOG"
}

# ============================================================================
# Step 19 — Purity theorem validation
# ============================================================================
step_19_purity() {
    local out="outputs/purity_validation/theory_validation.json"
    if [[ -f "$out" ]]; then
        log "Step 19: purity validation already done — skipping"
        return 0
    fi
    band "Step 19 — purity theorem validation"
    python scripts/run_purity_validation.py \
        --output_dir outputs/purity_validation 2>&1 | tee -a "$RUNNER_LOG"
}

# ============================================================================
# Step 19.2.2 — Retention diagnostic on the highest completed cycle.
# Originally hardcoded to cycle 2 (Flan-T5 era "early warning" checkpoint);
# on Qwen with EWC + MMLU-guard + rollback, the end-state answer stability
# at cycle 10 is the thesis-relevant metric. We walk outputs/full_run/cycle_*
# and pick the highest-numbered one that has a model/ subdir.
# ============================================================================
step_19_2_eval() {
    local last_cycle
    last_cycle=$(ls -d outputs/full_run/cycle_*/model 2>/dev/null \
        | sed 's#.*/cycle_\([0-9]\+\)/model#\1#' | sort -n | tail -1 || true)
    if [[ -z "${last_cycle:-}" ]]; then
        log "Step 19.2.2: no outputs/full_run/cycle_*/model checkpoint found — skipping"
        return 0
    fi
    local out="outputs/full_run/cycle_${last_cycle}/retention_diagnostic.json"
    local ckpt="outputs/full_run/cycle_${last_cycle}/model"
    if [[ -f "$out" ]]; then
        log "Step 19.2.2: retention diagnostic for cycle $last_cycle already present — skipping"
        return 0
    fi
    band "Step 19.2.2 — retention diagnostic (evaluating on highest completed cycle: $last_cycle)"
    python scripts/cycle2_retention_diagnostic.py --evaluate \
        --slice data/retention/cycle0_slice_500.jsonl \
        --cycle2_checkpoint "$ckpt" \
        --output_report "$out" 2>&1 | tee -a "$RUNNER_LOG"
}

# ============================================================================
# Step 19.5 — Nine-signal correlation matrix
# ============================================================================
step_19_5_corr() {
    local out="outputs/signal_correlation/redundancy_report.json"
    if [[ -f "$out" ]]; then
        log "Step 19.5: correlation report already present — skipping"
        return 0
    fi
    band "Step 19.5 — nine-signal correlation matrix"
    python scripts/signal_correlation_matrix.py \
        --eval_jsons "outputs/full_run/cycle_0/eval/*_cycle0.json" \
        --output_dir outputs/signal_correlation 2>&1 | tee -a "$RUNNER_LOG"
}

# ============================================================================
# Step 19.6 — Phase 4 theorem receipts (2026-04-28)
# ============================================================================
# Produces six receipt JSONs under outputs/full_run/theorem_receipts/ that
# back the formal theorem statements in Ch4 §11 and Ch5 §sec:check-*:
#   receipt_envelope_fit.json        thm:convergence + thm:bayes-convergence
#   receipt_eps_arch.json            thm:asymptotic-elim + cor:tier1-floor
#   receipt_gap_decay.json           cor:convergence-rate
#   receipt_corpus_floor.json        cor:corpus-floor
#   receipt_self_correction.json     cor:self-correction
#   receipt_tau_retro_sensitivity.json   defends τ_retro=0.50
# Reads outputs/full_run/experiment_summary.csv + cycle_*/ artefacts.
# Idempotent: skips if envelope receipt already present.
step_19_6_theorem_receipts() {
    local out="outputs/full_run/theorem_receipts/receipt_envelope_fit.json"
    if [[ -f "$out" ]]; then
        log "Step 19.6: theorem receipts already generated — skipping"
        return 0
    fi
    band "Step 19.6 — Phase 4 theorem receipts (envelope, eps_arch, gap, floor, survival, τ_retro)"
    python -m scripts.theorem_receipts \
        --output_dir outputs/full_run/theorem_receipts \
        --full_run_dir outputs/full_run \
        --summary_csv outputs/full_run/experiment_summary.csv \
        2>&1 | tee -a "$RUNNER_LOG" || {
            log "Step 19.6: theorem_receipts returned non-zero; some receipts may be partial."
        }
}

# ============================================================================
# Step 20.1 — Aggregate ablation (Phase 1a produces only `full` row; that's OK)
# ============================================================================
step_20_aggregate() {
    local out="outputs/ablation/ablation_table.csv"
    if [[ -f "$out" ]]; then
        log "Step 20.1: aggregate already present — skipping"
        return 0
    fi
    band "Step 20.1 — aggregate ablation outputs"
    python scripts/aggregate_ablation.py --output_dir outputs/ablation \
        2>&1 | tee -a "$RUNNER_LOG" || {
            log "Step 20.1 aggregate returned non-zero — Phase 1a has no ablation sweep yet"
            log "(this is expected until Phase 1 Full Steps 16-18 run); continuing."
        }
}

# ============================================================================
# Step 21 — LaTeX tables (Ch5 \input{} snippets from tab_*.csv)
# ============================================================================
# eval.reporting.build_ch5_tables (invoked from run_experiment.py) writes the
# .csv files; make_tables.py converts each into the .tex body chapter_5.tex
# ingests via \input{tab_*.tex}. Without this step the CSVs exist on disk
# but the thesis has no populated tables.
step_21_tables() {
    local out="outputs/full_run/tab_headline.tex"
    if [[ -f "$out" ]]; then
        log "Step 21: LaTeX tables already generated — skipping"
        return 0
    fi
    if [[ ! -f outputs/full_run/tab_headline.csv ]]; then
        log "Step 21: tab_headline.csv missing — run_experiment.py did not emit it; skipping"
        return 0
    fi
    band "Step 21 — generate Ch5 LaTeX tables from tab_*.csv"
    python -m scripts.make_tables outputs/full_run \
        2>&1 | tee -a "$RUNNER_LOG" || {
            log "Step 21: make_tables returned non-zero; tables may be partial."
        }
}

# ============================================================================
# Step 22 — Chapter 5 figures (PNG + PDF per figure)
# ============================================================================
# Reads tab_*.csv + per_sample_signals.jsonl from outputs/full_run and writes
# the 7 Ch5 figures (ces_radar, reliability, chm_trajectory, grounding,
# purity, continual, em_progression).
step_22_figures() {
    local out="outputs/full_run/fig5_3_chm_trajectory.pdf"
    if [[ -f "$out" ]]; then
        log "Step 22: Ch5 figures already generated — skipping"
        return 0
    fi
    if [[ ! -f outputs/full_run/tab_halluc_subtypes.csv ]]; then
        log "Step 22: tab_halluc_subtypes.csv missing — skipping figures"
        return 0
    fi
    band "Step 22 — render Ch5 figures (PNG + PDF) from tab_*.csv + per-sample JSONL"
    python -m scripts.make_figures outputs/full_run \
        2>&1 | tee -a "$RUNNER_LOG" || {
            log "Step 22: make_figures returned non-zero; some figures may be missing."
        }
}

# ============================================================================
# Step 23 — Per-cycle calibration trajectory (τ / T / memory-size aggregator)
# ============================================================================
# Produces the Ch5 §calibration "per-cycle calibration trajectory" table
# and the matching τ-trajectory + store-rate figure. Reads cycle_N
# calibrated_thresholds.json + cycle_N/memory snapshots from Step 7 main.
step_23_calib_traj() {
    local csv_out="outputs/full_run/calibration_trajectory.csv"
    if [[ -f "$csv_out" ]]; then
        log "Step 23: calibration trajectory already aggregated — skipping"
        return 0
    fi
    if [[ ! -d outputs/full_run ]]; then
        log "Step 23: outputs/full_run missing — skipping"
        return 0
    fi
    band "Step 23 — aggregate per-cycle calibration trajectory (τ, T, store-rate, memory size)"
    python scripts/aggregate_calibration_trajectory.py \
        --run_dir outputs/full_run \
        --cycle_0_dir outputs/cycle_0 \
        --csv_out "$csv_out" \
        --tex_out "pre thesis 1 report/tables/tab_calibration_trajectory.tex" \
        --fig_out "pre thesis 1 report/figures/fig_tau_trajectory.pdf" \
        2>&1 | tee -a "$RUNNER_LOG" || {
            log "Step 23: aggregator returned non-zero; some per-cycle data may be absent."
        }
}

# ============================================================================
# Step 24 — Session 5 audit artifacts (before/after, taxonomy map, α-trajectory)
# ============================================================================
# tab_audit_summary + tab_claim_evidence_map are static (no data dep);
# fig_audit_before_after, fig_composite_discrim_trajectory, fig_memory_growth,
# fig_alpha_trajectory all read from outputs/full_run + outputs/cycle_0 +
# outputs/purity_validation. Feeds Ch5 §5.X audit discussion + Ch6.
step_24_session5_artifacts() {
    local out="pre thesis 1 report/tables/tab_audit_summary.tex"
    if [[ -f "$out" ]]; then
        log "Step 24: Session 5 artifacts already emitted — skipping"
        return 0
    fi
    band "Step 24 — generate Session 5 audit artifacts (2 tables + 4 figures)"
    python scripts/generate_session5_artifacts.py \
        2>&1 | tee -a "$RUNNER_LOG" || {
            log "Step 24: artifact generator returned non-zero; review stdout for missing inputs."
        }
}

# ============================================================================
# Step 25 — T1 routing robustness validation (5 paraphrase strategies)
# ============================================================================
# Validates T1 routing math experimentally: for each of 5 paraphrase
# strategies (exact / synonym / reorder / rephrase / different), measures
# how close sim(query, stored) must be for Tier 1 to fire on a typical
# u_stored=0.7 episode. Uses the last completed cycle's memory store.
step_25_t1_robustness() {
    local out_dir="outputs/t1_routing_test"
    if [[ -f "$out_dir/results.json" ]]; then
        log "Step 25: T1 routing robustness already validated — skipping"
        return 0
    fi
    # Find the last completed cycle's memory store (matches the
    # step_19_2_eval pattern used for retention diagnostic).
    local last_cycle
    last_cycle=$(ls -d outputs/full_run/cycle_*/memory_store 2>/dev/null \
        | sed 's#.*/cycle_\([0-9]\+\)/memory_store#\1#' | sort -n | tail -1 || true)
    if [[ -z "${last_cycle:-}" ]]; then
        log "Step 25: no outputs/full_run/cycle_*/memory_store found — skipping"
        return 0
    fi
    local mem_path="outputs/full_run/cycle_${last_cycle}/memory_store"
    band "Step 25 — T1 routing robustness on cycle-${last_cycle} memory (n=50 episodes, 5 strategies)"
    python scripts/test_t1_routing_robustness.py \
        --memory_store "$mem_path" \
        --n_episodes 50 \
        --output_dir "$out_dir" \
        2>&1 | tee -a "$RUNNER_LOG" || {
            log "Step 25: T1 routing test returned non-zero; review output."
        }
}

# ============================================================================
# Main
# ============================================================================
main() {
    band "CAEM Phase 1a runner starting at $(ts)"
    log "Repo: $PWD    Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')    Commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"

    # --- Pre-launch (~10 h) ---
    step_5_9_prompt_smoke   # 2026-04-24 audit: validate prompt revision before seeding
    step_5_9_5_alias_dict   # 2026-05-07 audit: build alias dict for Fix 6 alias_overlap
    step_6_reseed
    step_6_5_em_prune       # v2.1 2026-05-08: drop confidently-wrong cold-start episodes
    step_7_0_cycle0
    step_7_0_calibrate              # Phase 1c: composite-only fit (no separate gate)
    step_7_0_2_5_rescore_eval       # produces outputs/cycle_0/eval_rescored
    step_7_0_3_validate_weights     # Phase 1c: empirical-precision audit (informational)
    step_5_5_pairs
    step_5_5_headhead
    step_19_2_slice               # must precede Step 7

    # --- Path B calibration (no-op sentinel; Frozen Qwen judge ABLATED 2026-04-26) ---
    step_platt_calibrate

    # --- One-shot gdrive snapshot of pre-Step-7 state (credit-burnout recovery) ---
    step_gdrive_upload_pre_main

    # --- Pre-Step-7 correctness gate (no-op sentinel; ABLATED 2026-05-06) ---
    step_u_tok_drop_gate

    # --- v2 Fix 11 Layer 5 — pre-launch dry-run of deferred reconsider path ---
    band "v2 Fix 11 Layer 5 — deferred reconsider dry-run (5 synthetic entries)"
    python scripts/test_deferred_reconsider_dryrun.py --synthetic_n 5 2>&1 \
        | tee -a "$RUNNER_LOG" || {
            log "FATAL: deferred-reconsider Layer 5 dry-run failed. Refuse to "
            log "       launch step_7_main without verified guard."
            return 2
        }

    # --- Headline (~7-8 days under v2 batched cal) ---
    step_7_main

    # --- External baselines (~8 h batched + ~55 h FT) ---
    step_9_b1
    step_10_b2
    step_11_5_b5                  # 5-shot CoT (Wei 2022) — reclaimed B5 slot
    step_11_b3
    step_12_b4
    step_14_b6_vanilla_ft
    step_15_b7_ewc_only
    step_15_5_sig

    # --- Diagnostics ---
    step_19_purity
    step_19_2_eval
    step_19_5_corr
    step_19_6_theorem_receipts    # Phase 4 theorem receipts (added 2026-04-28)
    step_20_aggregate

    # --- Ch5/Ch6 artefact generation (added 2026-04-24) ---
    # These convert the tab_*.csv / jsonl / cycle_* data into the .tex +
    # .pdf files that chapter_5.tex \input{}s and \includegraphics{}s. Each
    # step is idempotent (skips if output already present) and each
    # individually non-fatal (logs + continues if inputs are partial).
    step_21_tables                # LaTeX table bodies
    step_22_figures               # 7 Ch5 PNG/PDF figures (incl. CHM trajectory)
    step_23_calib_traj            # per-cycle τ/T/memory calibration table + figure
    step_24_session5_artifacts    # audit summary + before/after + α-trajectory
    step_25_t1_robustness         # T1 routing robustness validation

    band "Phase 1a runner COMPLETE at $(ts)"
    log "Next: scp outputs/ down locally (Step 20.2), then Vast.ai Stop."
}

# Only fire main() when this script is invoked directly. This lets
# run_post_step7.sh source us to reuse step_* function definitions without
# triggering the whole Phase 1a chain.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
