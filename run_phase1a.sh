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

BENCHMARKS=(fever triviaqa natural_questions truthfulqa strategyqa arc_challenge)
BASELINE_BENCHES="fever triviaqa natural_questions truthfulqa strategyqa arc_challenge"

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
        --benchmarks fever triviaqa natural_questions \
        --cold_start_store_threshold 0.50 \
        --output_dir "$out" 2>&1 | tee -a outputs/step6_seed.log
    python - <<'PY'
import json, sys
d = json.load(open("outputs/cold_start_memory/seed_summary.json"))
total = d.get("total_seeded", 0)
assert total >= 540, f"Step 6 verify: total_seeded={total} < 540 (expected 3x180+)"
print(f"Step 6 OK: total_seeded={total}")
PY
}

# ============================================================================
# Step 7.0 — Cycle-0 baseline eval (500 samples/benchmark)
# ============================================================================
step_7_0_cycle0() {
    local evaldir="outputs/cycle_0/eval"
    local all_present=1
    for b in "${BENCHMARKS[@]}"; do
        [[ -f "$evaldir/${b}_cycle0.json" ]] || all_present=0
    done
    if [[ $all_present -eq 1 ]]; then
        log "Step 7.0: Cycle-0 eval JSONs already present — skipping"
        return 0
    fi
    band "Step 7.0 — Cycle-0 baseline (500/bench, NEW prompts, bs=32)"
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
# Step 7.0.2 — Fit decision-tree thresholds from n_cal=500 calibration fold
# ============================================================================
step_7_0_calibrate() {
    local out="outputs/cycle_0/calibrated_thresholds.json"
    if [[ -f "$out" ]]; then
        log "Step 7.0.2: thresholds already fitted — skipping"
        python - "$out" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))["thresholds"]
print(f"   store={t['store']:.3f}  defer={t['defer']:.3f}  train={t['train']:.3f}")
PY
        return 0
    fi
    band "Step 7.0.2 — fit thresholds (quantile 0.70/0.40/0.90)"
    # BUGFIX 2026-04-21: run_experiment.py writes calibration_fold_samples.json
    # under cycle_0/calibration/ (subfolder), NOT at cycle_0/ top level. Path
    # must match the actual output location or calibrate_thresholds exits with
    # "No calibration JSONs found." See phase1a_flan_t5_halted snapshot for the
    # previous run that halted at this step.
    python scripts/calibrate_thresholds.py \
        --calib_jsons outputs/cycle_0/calibration/calibration_fold_samples.json \
        --verifier_backend minicheck \
        --output_json "$out" 2>&1 | tee -a "$RUNNER_LOG"
    python - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
t = d["thresholds"]
assert t["train"] > t["store"] > t["defer"], (
    f"Threshold ordering violated: train={t['train']} store={t['store']} defer={t['defer']}"
)
print(f"Step 7.0.2 OK: backend={d.get('verifier_backend')}  "
      f"store={t['store']:.3f}  defer={t['defer']:.3f}  train={t['train']:.3f}")
PY
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
    python scripts/calibration_minicheck_vs_roberta.py \
        --pairs_jsonl data/calibration/minicheck_pairs_500.jsonl \
        --output_json "$out" \
        --device cuda \
        --backends minicheck roberta_nli 2>&1 | tee -a outputs/calibration/minicheck_vs_roberta_v5.log
}

step_5_5_gate() {
    # Log-only diagnostic: user is committed to MiniCheck regardless of the
    # v5 outcome (2026-04-20 decision — Ch4 theorem alignment trumps pooled
    # AUROC on EM-label-biased pairs). Never halts the runner.
    band "Step 5.5.4 — Scenario classifier (log-only, non-blocking)"
    python - <<'PY' || true
import json
d = json.load(open("outputs/calibration/minicheck_vs_roberta_v5.json"))
mc, rb = d["minicheck"], d["roberta_nli"]
auc_delta = mc["auroc"] - rb["auroc"]
ece_delta = mc["ece"]   - rb["ece"]
print(f"  MiniCheck: AUROC={mc['auroc']:.4f} ECE={mc['ece']:.4f}")
print(f"  RoBERTa : AUROC={rb['auroc']:.4f} ECE={rb['ece']:.4f}")
print(f"  delta  : AUROC={auc_delta:+.4f}  ECE={ece_delta:+.4f}")
if auc_delta >= 0.05 and ece_delta <= -0.10:
    print("  -> Scenario A: clean MiniCheck win. (Table is Ch5-usable.)")
elif auc_delta >= 0.02:
    print("  -> Scenario B: marginal win. (Table usable with softened wording.)")
else:
    print("  -> Scenario C: null/loss on pooled pairs. (Cite EM-label caveat; backend stays MiniCheck per Ch4 theorem alignment.)")
PY
}

# ============================================================================
# Prompt-design ablation (external Cycle-0 comparison — Session 2 delta)
# ============================================================================
step_prompt_ablation() {
    local out="pre thesis 1 report/tables/tab_prompt_design_ablation.tex"
    if [[ -f "$out" ]]; then
        log "Prompt-design ablation: $out already present — skipping"
        return 0
    fi
    if [[ ! -d outputs/cycle_0_pre_cot_prompt/eval ]]; then
        log "Prompt-design ablation: OLD-prompt eval dir missing — skipping (non-fatal)"
        return 0
    fi
    band "Prompt-design ablation — OLD vs NEW Cycle-0"
    python scripts/compare_prompt_design.py \
        --old_dir outputs/cycle_0_pre_cot_prompt/eval \
        --new_dir outputs/cycle_0/eval \
        --output_tex "$out" 2>&1 | tee -a "$RUNNER_LOG"
}

# ============================================================================
# u_tok_drop pool correctness gate (replaces Level B smoke for Branch C).
# Runs diff_verify_serial_vs_batch.py at N=32 under CAEM_BATCH_U_TOK_DROP=1 —
# the exact Step 7 main config — so we catch any pool regression before
# burning 11 days on the headline. Gate threshold matches your 2026-04-22
# 05:30 BDT [PERF] report: |mean Δ u_stored| < 0.02, balanced sign pattern,
# p_ground_atomic drift exactly zero (atomic pool still guarded).
# ============================================================================
step_u_tok_drop_gate() {
    local marker="outputs/u_tok_drop_validation.log"
    if [[ -f "$marker" ]] && grep -q "u_tok_drop PASSED" "$marker" 2>/dev/null; then
        log "u_tok_drop gate: previously PASSED — skipping"
        return 0
    fi
    band "u_tok_drop pool correctness gate (N=32 diff_verify, CAEM_BATCH_U_TOK_DROP=1)"
    CAEM_BATCH_U_TOK_DROP=1 python scripts/diff_verify_serial_vs_batch.py \
        --n 32 --benchmark fever 2>&1 | tee "$marker"
    # Parse |mean_delta| for u_stored; fail if >= 0.02 (your 2026-04-22 cutoff).
    python - <<'PY'
import re, sys
log = open("outputs/u_tok_drop_validation.log").read()
m = re.search(r"u_stored\s+([+-]?\d+\.\d+)\s+(\d+\.\d+)", log)
if not m:
    print("u_tok_drop gate: could not parse u_stored drift; FAILING closed"); sys.exit(1)
mean_delta = float(m.group(1))
max_abs    = float(m.group(2))
print(f"  u_stored mean_delta={mean_delta:+.4f}  max|delta|={max_abs:.4f}")
# Thresholds from the 2026-04-22 v4 shipped N=32 gate:
#   |mean delta| < 0.02  (your report's accepted max was 0.016)
#   p_ground_atomic guard still holds (checked separately below)
if abs(mean_delta) >= 0.02:
    print(f"  u_tok_drop gate FAILED: |mean_delta|={abs(mean_delta):.4f} >= 0.02"); sys.exit(1)
# Confirm atomic-pool guard didn't fire (p_ground_atomic should be bit-zero).
if not re.search(r"p_ground_atomic\s+\+0\.0000\s+0\.0000", log):
    print("  u_tok_drop gate FAILED: atomic pool shows drift — guard broke"); sys.exit(1)
print("  u_tok_drop PASSED")
PY
    echo "u_tok_drop PASSED" >> "$marker"
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
# HF snapshot — upload all pre-Step-7 artefacts so a fresh instance can
# resume from Step 7 directly if the current instance dies before Step 7
# finishes (or even starts).
# ============================================================================
step_hf_upload_pre_main() {
    local marker="outputs/.hf_pre_main_uploaded"
    if [[ -f "$marker" ]]; then
        log "HF pre-main snapshot: already uploaded (marker $marker) — skipping"
        return 0
    fi
    band "HF snapshot — uploading pre-Step-7 artefacts to aksaN000/caem-passage-index-21m:pre_main_snapshot/"
    python - <<'PY'
import os, sys, json
from pathlib import Path
from huggingface_hub import HfApi, whoami

api  = HfApi()
repo = "aksaN000/caem-passage-index-21m"
user = whoami()["name"]
print(f"  HF user: {user}    target repo: {repo}    subfolder: pre_main_snapshot/")

# Stage all pre-Step-7 artefacts into a single flat layout under outputs/.snapshot_staging/
# so upload_folder() can push them in one call. The staging dir gets mirrored to
# pre_main_snapshot/ on HF.
stage = Path("outputs/.snapshot_staging")
if stage.exists():
    import shutil; shutil.rmtree(stage)
stage.mkdir(parents=True)

def copy(src, rel_dst):
    src = Path(src)
    if not src.exists():
        print(f"  [skip] {src} (not present)")
        return
    dst = stage / rel_dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        import shutil; shutil.copytree(src, dst)
    else:
        import shutil; shutil.copy2(src, dst)
    print(f"  [add ] {src} -> {rel_dst}")

# Step 6 cold-start memory
copy("outputs/cold_start_memory", "cold_start_memory")
# Step 7.0 Cycle-0 eval + calibrated thresholds + cycle-0 memory/deferred snapshot
copy("outputs/cycle_0/eval",                              "cycle_0/eval")
copy("outputs/cycle_0/calibration",                       "cycle_0/calibration")
copy("outputs/cycle_0/calibrated_thresholds.json",        "cycle_0/calibrated_thresholds.json")
copy("outputs/cycle_0/calibration_fold_samples.json",     "cycle_0/calibration_fold_samples.json")
copy("outputs/cycle_0/memory_store_cycle_0.faiss",        "cycle_0/memory_store_cycle_0.faiss")
copy("outputs/cycle_0/memory_store_cycle_0.meta",         "cycle_0/memory_store_cycle_0.meta")
copy("outputs/cycle_0/deferred_buffer_cycle_0.pkl",       "cycle_0/deferred_buffer_cycle_0.pkl")
copy("outputs/cycle_0/dataset_splits.json",               "cycle_0/dataset_splits.json")
copy("outputs/cycle_0/mmlu_baseline.json",                "cycle_0/mmlu_baseline.json")
copy("outputs/cycle_0/run.log",                           "cycle_0/run.log")
# Step 5.5 v5 + pair set
copy("outputs/calibration/minicheck_vs_roberta_v5.json",  "calibration/minicheck_vs_roberta_v5.json")
copy("outputs/calibration/minicheck_vs_roberta_v5.log",   "calibration/minicheck_vs_roberta_v5.log")
copy("data/calibration/minicheck_pairs_500.jsonl",        "calibration/minicheck_pairs_500.jsonl")
# Step 19.2.1 retention slice
copy("data/retention/cycle0_slice_500.jsonl",             "retention/cycle0_slice_500.jsonl")
# Gates + audit logs
copy("outputs/level_b_smoke.log",                         "level_b_smoke.log")
copy("outputs/phase1a_runner.log",                        "phase1a_runner.log")
copy("outputs/step6_seed.log",                            "step6_seed.log")

manifest = {
    "uploaded_at":     __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "source_host":     os.uname().nodename,
    "git_commit":      os.popen("git rev-parse HEAD").read().strip(),
    "purpose":         "Phase 1a pre-Step-7 snapshot for credit-burnout recovery",
    "recovery_hint":   "hf download aksaN000/caem-passage-index-21m --include 'passages.*' 'pre_main_snapshot/*' && ./run_phase1a.sh",
}
(stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

print("  uploading ...")
api.upload_folder(
    folder_path = str(stage),
    path_in_repo= "pre_main_snapshot",
    repo_id     = repo,
    repo_type   = "dataset",
    commit_message = f"Phase 1a pre-Step-7 snapshot ({manifest['git_commit'][:8]})",
)
print("  OK — snapshot live at "
      f"https://huggingface.co/datasets/{repo}/tree/main/pre_main_snapshot")
PY
    touch "$marker"
}

# ============================================================================
# Step 7 — Main 10-cycle CAEM run at n_questions=5000 (Phase 1a headline)
# ============================================================================
step_7_main() {
    if [[ -f outputs/full_run/experiment_summary.csv ]]; then
        local rows
        rows=$(wc -l < outputs/full_run/experiment_summary.csv)
        if (( rows >= 12 )); then   # header + 11 cycle rows
            log "Step 7: experiment_summary.csv has $rows rows — main run complete, skipping"
            return 0
        fi
    fi
    band "Step 7 — main 10-cycle CAEM run (n=5000/bench, ~335 GPU-h) [u_tok_drop pools ON, gdrive offload ON]"
    # Enable u_tok_drop verifier pools + gdrive checkpoint offload for this
    # python child only. Scoped via leading assignments on the python call
    # so subsequent runner stages see defaults.
    # - u_tok_drop validated at step_u_tok_drop_gate immediately before.
    # - gdrive offload sends every cycle to gdrive:caem-phase1a/full_run/
    #   cycle_<n>/ after local save; local rolling-N still active as safety.

    local tau_store tau_defer tau_train backend
    tau_store=$(python -c 'import json; print(json.load(open("outputs/cycle_0/calibrated_thresholds.json"))["thresholds"]["store"])')
    tau_defer=$(python -c 'import json; print(json.load(open("outputs/cycle_0/calibrated_thresholds.json"))["thresholds"]["defer"])')
    tau_train=$(python -c 'import json; print(json.load(open("outputs/cycle_0/calibrated_thresholds.json"))["thresholds"]["train"])')
    backend=$(python -c 'import json,sys; d=json.load(open("outputs/cycle_0/calibrated_thresholds.json")); print(d.get("verifier_backend","minicheck"))')
    log "  fitted backend=$backend store=$tau_store defer=$tau_defer train=$tau_train"

    local resume_args=()
    if [[ -d outputs/full_run ]]; then
        local last
        last=$(ls -d outputs/full_run/cycle_* 2>/dev/null | sed 's#.*cycle_##' | sort -n | tail -1 || true)
        if [[ -n "${last:-}" && "$last" -ge 0 && "$last" -lt 10 ]]; then
            resume_args=(--resume_from_cycle "$((last + 1))")
            log "  detected completed cycle $last — resuming from $((last + 1))"
        fi
    fi

    CAEM_BATCH_U_TOK_DROP=1 CAEM_GDRIVE_OFFLOAD=1 python -m scripts.run_experiment \
        --output_dir outputs/full_run \
        --num_cycles 10 \
        --n_questions 5000 \
        --n_eval_questions 500 \
        --benchmarks "${BENCHMARKS[@]}" \
        --passage_index data/passage_index \
        --cold_start_memory outputs/cold_start_memory/memory_store \
        --verifier_backend "$backend" \
        --store_threshold "$tau_store" \
        --defer_threshold "$tau_defer" \
        --train_threshold "$tau_train" \
        --eval_batch_size 32 \
        --eval_prefetch \
        "${resume_args[@]}" \
        2>&1 | tee -a outputs/full_run/run.log

    python - <<'PY'
import csv
rows = list(csv.reader(open("outputs/full_run/experiment_summary.csv")))
assert len(rows) >= 12, f"experiment_summary.csv has {len(rows)} rows, expected >=12 (header+11)"
header = rows[0]; idx = header.index("mmlu_retention_pct")
bad = [r for r in rows[1:] if float(r[idx]) < 93.0]
assert not bad, f"MMLU retention <93% on {len(bad)} cycles: {bad}"
print("Step 7 OK: 11 cycle rows, MMLU retention >=93% on all cycles")
PY
}

# ============================================================================
# Step 8 — FLARE smoke (mandatory before B5)
# ============================================================================
step_8_flare_smoke() {
    if ls outputs/baselines_smoke/flare/*.json &>/dev/null; then
        log "Step 8: FLARE smoke output already present — skipping"
        return 0
    fi
    band "Step 8 — FLARE decoder-slice smoke (5 FEVER samples)"
    python -m scripts.run_baseline \
        --baseline flare \
        --benchmarks fever \
        --n_questions 5 \
        --passage_index data/passage_index \
        --flare_theta 0.4 \
        --flare_look_ahead 64 \
        --output_dir outputs/baselines_smoke 2>&1 | tee outputs/baselines_smoke/B5_flare_smoke.log
    python - <<'PY'
import json, glob, sys
ok = False
for f in sorted(glob.glob("outputs/baselines_smoke/flare/*.json")):
    d = json.load(open(f)); s = d.get("samples", [])
    empty = sum(1 for x in s if not x.get("answer", "").strip())
    esc   = sum(1 for x in s if x.get("escalated"))
    print(f"  {f} n={len(s)} empty={empty} escalated={esc}")
    if empty == 0 and esc >= 1: ok = True
if not ok:
    print("FLARE smoke FAILED: need empty_answer=0 AND escalated>=1"); sys.exit(1)
PY
}

# ============================================================================
# Steps 9-15 — B1..B7 baselines (n=5000 per benchmark)
# ============================================================================
_run_inference_baseline() {
    local step="$1" name="$2" flag="$3"; shift 3
    local extra=("$@")
    local outdir="outputs/baselines/$name"
    if ls "$outdir"/*.json &>/dev/null; then
        log "Step $step ($name): already present — skipping"
        return 0
    fi
    band "Step $step — $name baseline (n=5000, bs=32)"
    python -m scripts.run_baseline \
        --baseline "$flag" \
        --benchmarks $BASELINE_BENCHES \
        --n_questions 5000 \
        --eval_batch_size 32 \
        --output_dir outputs/baselines \
        "${extra[@]}" \
        2>&1 | tee "outputs/baselines/${step}_${name}.log"
}

step_9_b1()  { _run_inference_baseline "B1" "zero_shot" "zero_shot"; }
step_10_b2() { _run_inference_baseline "B2" "cot"       "cot"; }
step_11_b3() { _run_inference_baseline "B3" "rag"       "rag"       --passage_index data/passage_index; }
step_12_b4() { _run_inference_baseline "B4" "cot_rag"   "cot_rag"   --passage_index data/passage_index; }
step_13_b5() { _run_inference_baseline "B5" "flare"     "flare"     --passage_index data/passage_index --flare_theta 0.4 --flare_look_ahead 64; }

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
    CAEM_GDRIVE_OFFLOAD=1 python -m scripts.run_simple_ft \
        --baseline_name vanilla_ft \
        --num_cycles 10 \
        --eval_benchmarks "${BENCHMARKS[@]}" \
        --n_eval_per_bench 5000 \
        --n_train_per_bench 4000 \
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
    CAEM_GDRIVE_OFFLOAD=1 python -m scripts.run_simple_ft \
        --baseline_name ewc_only_ft \
        --use_l2_anchor \
        --use_mmlu_guard \
        --num_cycles 10 \
        --eval_benchmarks "${BENCHMARKS[@]}" \
        --n_eval_per_bench 5000 \
        --n_train_per_bench 4000 \
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
        | sed 's#.*/cycle_\([0-9]\+\)/model#\1#' | sort -n | tail -1)
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
# Main
# ============================================================================
main() {
    band "CAEM Phase 1a runner starting at $(ts)"
    log "Repo: $PWD    Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')    Commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"

    # --- Pre-launch (~10 h) ---
    step_6_reseed
    step_7_0_cycle0
    step_7_0_calibrate
    step_5_5_pairs
    step_5_5_headhead
    # step_5_5_gate: REMOVED (scenario classifier, 2026-04-20 decision is
    #   MiniCheck irrespective; wording guidance only).
    # step_prompt_ablation: REMOVED (OLD-prompt eval dir is Flan-T5 era;
    #   comparing OLD@T5 vs NEW@Qwen conflates two variables and isn't
    #   defensible as a prompt-design ablation).
    step_19_2_slice               # must precede Step 7

    # --- One-shot HF snapshot of pre-Step-7 state (credit-burnout recovery) ---
    step_hf_upload_pre_main

    # --- Pre-Step-7 correctness gate (replaces Level B smoke) ---
    # Validates CAEM_BATCH_U_TOK_DROP=1 pool config right before Step 7 main
    # burns 11 days of GPU. Fails the runner if |mean Δ u_stored| >= 0.02
    # or if the atomic-pool guard broke.
    step_u_tok_drop_gate

    # --- Headline (~335 h / ~14 days) ---
    step_7_main

    # --- External baselines (~8 h batched + ~55 h FT) ---
    # B5 FLARE and its pre-gate removed from the chain: FLARE is inference-
    # only and cannot be batched (iterative look-ahead per sentence), so at
    # 30k serial queries it would cost ~25-125 GPU-h for a baseline that
    # doesn't defend any of the 5 headline claims. CAEM's training + memory
    # design makes any FLARE comparison structurally asymmetric (see
    # branch_C_log.md 2026-04-22 removal rationale). step_8_flare_smoke and
    # step_13_b5 function bodies remain in-file for potential Phase 2 reuse.
    step_9_b1
    step_10_b2
    step_11_b3
    step_12_b4
    step_14_b6_vanilla_ft
    step_15_b7_ewc_only
    step_15_5_sig

    # --- Diagnostics ---
    step_19_purity
    step_19_2_eval
    step_19_5_corr
    step_20_aggregate

    band "Phase 1a runner COMPLETE at $(ts)"
    log "Next: scp outputs/ down locally (Step 20.2), then Vast.ai Stop."
}

main "$@"
