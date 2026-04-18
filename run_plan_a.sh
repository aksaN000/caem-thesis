#!/usr/bin/env bash
# Plan A autonomous runner — Steps 5-20 of NEXT_SESSION_PLAN.md
# Depends on Step 4's output: data/passage_index/passages.faiss + passages.pkl
# Scope: Steps 5, 6, 7, 8, 9-15, 19, 20. Skips 16-18 (screening+confirmatory).

set -u
cd /workspace/caem
export HF_HOME=/workspace/caem/hf_cache

RUNNER_LOG=/workspace/caem/plan_a_runner.log
mkdir -p outputs/smoke outputs/baselines outputs/baselines_smoke outputs/full_run outputs/purity_validation outputs/ablation

say() {
  echo
  echo "===== $(date -u +'%Y-%m-%d %H:%M:%S UTC') | $* ====="
  echo
}

# Redirect everything to the runner log (append-mode, line-buffered)
exec > >(stdbuf -oL tee -a "$RUNNER_LOG") 2>&1

say "Plan A runner started. PID=$$"

############################################
# Wait for Step 4 (passage index) to finish
############################################
say "Waiting for Step 4 outputs (data/passage_index/passages.faiss + passages.pkl)..."
while [ ! -f data/passage_index/passages.faiss ] || [ ! -f data/passage_index/passages.pkl ]; do
  sleep 120
done
# Give it 60s to finish flushing if it just wrote the files
sleep 60
say "Step 4 outputs detected. ls -lh data/passage_index/:"
ls -lh data/passage_index/ || true

BENCH="fever triviaqa natural_questions truthfulqa strategyqa arc_challenge"

############################################
# Step 5 — smoke test (HARD GATE)
############################################
say "Step 5 — smoke test (1 cycle, n=50)"
python scripts/run_cyclic_ablation.py \
    --variant full \
    --seed 42 \
    --smoke_test \
    --passage_index data/passage_index \
    --output_dir outputs/smoke \
    2>&1 | tee outputs/smoke/run.log

say "Step 5 — VERIFY ces > 0"
python - <<'PY'
import json, glob, sys, os
cands = sorted(glob.glob("outputs/smoke/**/ces_axes_per_cycle.json", recursive=True))
if not cands:
    print("FATAL: no ces_axes_per_cycle.json under outputs/smoke/", file=sys.stderr)
    sys.exit(1)
path = cands[0]
d = json.load(open(path))
print("smoke output file:", path)
print("type:", type(d).__name__)
ces = None
if isinstance(d, list) and d:
    first = d[0]
    ces = first.get("ces") if isinstance(first, dict) else None
elif isinstance(d, dict):
    if "cycles" in d and d["cycles"]:
        ces = d["cycles"][0].get("ces")
    else:
        ces = d.get("ces")
print("ces =", ces)
if ces is None or ces <= 0:
    print("FATAL: ces is not positive", file=sys.stderr)
    sys.exit(1)
print("Step 5 gate: PASS")
PY
if [ $? -ne 0 ]; then
    say "STEP 5 HARD GATE FAILED. Halting runner."
    exit 1
fi

############################################
# Step 6 — cold-start memory seeding
############################################
say "Step 6 — cold-start memory seeding (target 200 episodes)"
python -m scripts.seed_cold_start \
    --target_episodes 200 \
    --benchmarks fever triviaqa natural_questions \
    --output_dir outputs/cold_start_memory \
    2>&1 | tee outputs/cold_start_memory_run.log || say "Step 6 returned non-zero; continuing"

say "Step 6 — VERIFY artifacts"
ls -la outputs/cold_start_memory/ || true
[ -f outputs/cold_start_memory/seed_summary.json ] && cat outputs/cold_start_memory/seed_summary.json || echo "seed_summary.json missing"

############################################
# Step 7 — main 10-cycle CAEM run (with single retry via --resume_from_cycle)
############################################
say "Step 7 — main 10-cycle run"
run_step7() {
    local last_cycle
    last_cycle=$(ls outputs/full_run/ 2>/dev/null | grep -oE '^cycle_[0-9]+$' | grep -oE '[0-9]+$' | sort -n | tail -1)
    local args=(--output_dir outputs/full_run --num_cycles 10 --n_questions 5000
                --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge
                --passage_index data/passage_index
                --cold_start_memory outputs/cold_start_memory/memory_store)
    if [ -n "${last_cycle:-}" ] && [ "$last_cycle" -ge 1 ]; then
        local resume_from=$((last_cycle + 1))
        if [ "$resume_from" -le 10 ]; then
            say "Step 7 resuming from cycle $resume_from (found cycle_$last_cycle)"
            args+=(--resume_from_cycle "$resume_from")
        else
            say "Step 7 already completed through cycle $last_cycle"
            return 0
        fi
    else
        say "Step 7 fresh start"
    fi
    python -m scripts.run_experiment "${args[@]}" 2>&1 | tee -a outputs/full_run/run.log
    return ${PIPESTATUS[0]}
}
step7_ok=0
for attempt in 1 2 3; do
    say "Step 7 attempt $attempt"
    if run_step7; then
        step7_ok=1
        break
    fi
    say "Step 7 attempt $attempt failed; sleeping 60s before retry"
    sleep 60
done
if [ "$step7_ok" = "1" ]; then
    say "Step 7 — VERIFY experiment_summary.csv"
    ls outputs/full_run/ || true
    [ -f outputs/full_run/experiment_summary.csv ] && head outputs/full_run/experiment_summary.csv || echo "experiment_summary.csv missing"
else
    say "Step 7 failed after 3 attempts — continuing to baselines (headline number may be incomplete)"
fi

############################################
# Step 8 — FLARE pre-flight smoke (HARD GATE for Step 13)
############################################
say "Step 8 — FLARE pre-flight smoke test (5 FEVER samples)"
mkdir -p outputs/baselines_smoke
python -m scripts.run_baseline \
    --baseline flare \
    --benchmarks fever \
    --n_questions 5 \
    --passage_index data/passage_index \
    --flare_theta 0.4 \
    --flare_look_ahead 64 \
    --output_dir outputs/baselines_smoke \
    2>&1 | tee outputs/baselines_smoke/B5_flare_smoke.log || say "Step 8 run returned non-zero"

FLARE_SMOKE_OK=0
python - <<'PY'
import json, glob, sys
bad = 0
files = sorted(glob.glob("outputs/baselines_smoke/flare/*.json"))
if not files:
    print("FATAL: no flare smoke output files")
    sys.exit(1)
for f in files:
    d = json.load(open(f))
    samples = d.get("samples", [])
    empty = sum(1 for s in samples if not str(s.get("answer", "")).strip())
    escalated = sum(1 for s in samples if s.get("escalated"))
    print(f, "n=", len(samples), "empty=", empty, "escalated=", escalated)
    if empty > 0 or escalated < 1:
        bad += 1
if bad:
    print("FLARE smoke: FAIL")
    sys.exit(1)
print("FLARE smoke: PASS")
PY
if [ $? -eq 0 ]; then
    FLARE_SMOKE_OK=1
    say "Step 8 gate: PASS"
else
    say "Step 8 gate: FAIL — will SKIP Step 13 B5 but continue with others"
fi

############################################
# Steps 9-13 — B1-B5 inference baselines (soft fail)
############################################
say "Step 9 — B1 zero-shot"
python -m scripts.run_baseline \
    --baseline zero_shot \
    --benchmarks $BENCH \
    --n_questions 500 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B1_zero_shot.log || say "B1 failed; continuing"

say "Step 10 — B2 chain-of-thought"
python -m scripts.run_baseline \
    --baseline cot \
    --benchmarks $BENCH \
    --n_questions 500 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B2_cot.log || say "B2 failed; continuing"

say "Step 11 — B3 DPR-RAG"
python -m scripts.run_baseline \
    --baseline rag \
    --benchmarks $BENCH \
    --n_questions 500 \
    --passage_index data/passage_index \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B3_rag.log || say "B3 failed; continuing"

say "Step 12 — B4 CoT + RAG"
python -m scripts.run_baseline \
    --baseline cot_rag \
    --benchmarks $BENCH \
    --n_questions 500 \
    --passage_index data/passage_index \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B4_cot_rag.log || say "B4 failed; continuing"

if [ "$FLARE_SMOKE_OK" = "1" ]; then
    say "Step 13 — B5 FLARE"
    python -m scripts.run_baseline \
        --baseline flare \
        --benchmarks $BENCH \
        --n_questions 500 \
        --passage_index data/passage_index \
        --flare_theta 0.4 \
        --flare_look_ahead 64 \
        --output_dir outputs/baselines \
        2>&1 | tee outputs/baselines/B5_flare.log || say "B5 failed; continuing"
else
    say "Step 13 — B5 FLARE SKIPPED (Step 8 smoke failed)"
fi

############################################
# Step 14 — B6 Vanilla FT (10 cycles, soft fail)
############################################
say "Step 14 — B6 Vanilla FT (10 cycles)"
mkdir -p outputs/baselines/vanilla_ft
python -m scripts.run_simple_ft \
    --baseline_name vanilla_ft \
    --num_cycles 10 \
    --passage_index data/passage_index \
    --benchmarks $BENCH \
    --n_questions 500 \
    --output_dir outputs/baselines/vanilla_ft \
    2>&1 | tee outputs/baselines/B6_vanilla_ft.log || say "B6 failed; continuing"

############################################
# Step 15 — B7 EWC-only FT (10 cycles, soft fail)
############################################
say "Step 15 — B7 EWC-only FT (10 cycles)"
mkdir -p outputs/baselines/ewc_only_ft
python -m scripts.run_simple_ft \
    --baseline_name ewc_only_ft \
    --use_l2_anchor \
    --use_mmlu_guard \
    --num_cycles 10 \
    --passage_index data/passage_index \
    --benchmarks $BENCH \
    --n_questions 500 \
    --output_dir outputs/baselines/ewc_only_ft \
    2>&1 | tee outputs/baselines/B7_ewc_only_ft.log || say "B7 failed; continuing"

############################################
# Step 19 — purity theorem validation
############################################
say "Step 19 — purity theorem validation"
python scripts/run_purity_validation.py \
    --output_dir outputs/purity_validation \
    2>&1 | tee outputs/purity_validation.log || say "Step 19 failed; continuing"
[ -f outputs/purity_validation/theory_validation.json ] && head -c 4000 outputs/purity_validation/theory_validation.json && echo

############################################
# Step 20 — final aggregate (Plan A: ablation dir is empty, so this is near-empty)
############################################
say "Step 20 — final aggregate (outputs/ablation is empty in Plan A; expect empty/near-empty output)"
python scripts/aggregate_ablation.py \
    --output_dir outputs/ablation \
    2>&1 | tee outputs/final_aggregate.log || say "Step 20 aggregator errored (expected if no ablation variants)"

say "Step 20 — package outputs for download"
tar czf /workspace/caem/plan_a_outputs.tar.gz \
    outputs/full_run \
    outputs/baselines \
    outputs/baselines_smoke \
    outputs/smoke \
    outputs/cold_start_memory \
    outputs/purity_validation \
    outputs/ablation 2>/dev/null || say "tar had some missing dirs"
ls -lh /workspace/caem/plan_a_outputs.tar.gz || true

say "PLAN A COMPLETE. Next: user must run scp from local PC to pull /workspace/caem/plan_a_outputs.tar.gz, then Stop the Vast instance."
