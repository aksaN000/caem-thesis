#!/usr/bin/env bash
# Rebuild every final-version number, table and figure of chapter 5 from the run outputs (2026-10-19).
#
#   bash scripts/make_final_results.sh            # deterministic steps only (CPU, about 2 minutes)
#   bash scripts/make_final_results.sh --judge    # first re-judge TruthfulQA with Claude Haiku (needs ANTHROPIC_API_KEY,
#                                                 # anthropic<1; about $2 for all systems)
#   bash scripts/make_final_results.sh --check    # rebuild, then fail if any committed output changed
#
# Inputs: outputs/full_run/eval (CAEM C0-C5), outputs/phase1d_eval (classifier ablation), outputs/rerun_v2/baselines
# (baseline re-run, scripts/rental/final_version.sh) and outputs/baselines_phase1e/{rag,cot_rag} (B3/B4 except
# TruthfulQA); scripts/final_sources.py maps each system and benchmark to its file.
# Python needs numpy, scipy and the caem package's imports (torch CPU build is enough): set PYTHON if not `python`.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
JUDGE=0; CHECK=0
for a in "$@"; do case "$a" in --judge) JUDGE=1 ;; --check) CHECK=1 ;; *) echo "unknown option $a"; exit 2 ;; esac; done

if [ "$JUDGE" = 1 ]; then
  "$PY" -m scripts.rescore_truthfulqa --files 'outputs/full_run/eval/truthfulqa_cycle[0-9].json' \
      outputs/phase1d_eval/truthfulqa_cycle3.json 'outputs/rerun_v2/baselines/*/truthfulqa_cycle0.json' \
      outputs/rerun_v2/baselines/vanilla_ft/eval/truthfulqa_cycle5.json
fi
"$PY" scripts/rescore_uniform.py            # em_final into every answer file
"$PY" scripts/recompute_chm_judged.py       # outputs/analysis/chm_judged.json (model + delivered view, status tags)
"$PY" scripts/paired_baseline_tests.py --write_tex   # outputs/analysis/paired_baseline_tests_final.json + sig tables
"$PY" scripts/build_ch5_evidence_tables.py  # gate, sweep, headline, probe tables; chapter 5 figures
"$PY" scripts/build_final_tables.py         # per-benchmark, subtypes, baselines, status tags, ablation, tiers, CES

if [ "$CHECK" = 1 ]; then
  if git diff --quiet -- outputs/analysis defense/thesis_report/figures; then
    echo "check: every committed output reproduced exactly"
  else
    git diff --stat -- outputs/analysis defense/thesis_report/figures; echo "check: outputs differ"; exit 1
  fi
fi
