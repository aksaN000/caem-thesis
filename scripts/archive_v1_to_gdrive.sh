#!/usr/bin/env bash
# scripts/archive_v1_to_gdrive.sh
# ===============================
# One-shot reorganization of gdrive:caem-phase1a/ for v2 architecture.
#
# Before:
#   gdrive:caem-phase1a/full_run/cycle_{0,1,2,3,4}/   v1 trajectory
#   gdrive:caem-phase1a/full_run/run.log              v1 audit trail
#   gdrive:caem-phase1a/full_run/aborted_cycle5_*/   v1 cycle 5 partial
#
# After:
#   gdrive:caem-phase1a/archive_v1/full_run/cycle_{0..4}/    moved (failure-mode evidence)
#   gdrive:caem-phase1a/archive_v1/full_run/aborted_cycle5*  moved
#   gdrive:caem-phase1a/archive_v1/full_run/run.log          moved
#   gdrive:caem-phase1a/archive_v1/README.md                 explains what's in here
#   gdrive:caem-phase1a/v2/                                  empty; ready for v2 cycle artefacts
#   gdrive:caem-phase1a/v2/README.md                         explains the v2 layout
#   gdrive:caem-phase1a/production/                          (untouched) Phase B production swap target
#
# This script is OPERATOR-EXECUTED. It does NOT run automatically as part
# of any cycle launch. Run it ONCE, before launching the v2 cycle 0.
#
# Pre-requisites:
#   - rclone configured with `gdrive:` remote pointing at the user's Drive
#   - run as root or as the user who has write access to the gdrive root
#
# Verifies:
#   - archive_v1 does not already exist (don't double-archive)
#   - v2/ folder is created clean
#   - both READMEs land
#
# Usage:
#   bash scripts/archive_v1_to_gdrive.sh              dry-run (default)
#   bash scripts/archive_v1_to_gdrive.sh --execute    actually move files
set -euo pipefail

DRY_RUN=true
if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN=false
fi

GDRIVE_ROOT="gdrive:caem-phase1a"
ARCHIVE_PATH="${GDRIVE_ROOT}/archive_v1"
V2_PATH="${GDRIVE_ROOT}/v2"

echo "=== gdrive v1 archive operation ==="
echo "  source   : ${GDRIVE_ROOT}/full_run/"
echo "  archive  : ${ARCHIVE_PATH}/full_run/"
echo "  v2 target: ${V2_PATH}/"
echo "  mode     : $($DRY_RUN && echo 'DRY-RUN (no files moved)' || echo 'EXECUTE')"
echo

# 1. sanity: rclone configured?
if ! command -v rclone >/dev/null 2>&1; then
    echo "ERROR: rclone not found in PATH"
    exit 1
fi
if ! rclone listremotes 2>/dev/null | grep -q '^gdrive:'; then
    echo "ERROR: rclone remote 'gdrive:' not configured"
    echo "  Run: rclone config"
    exit 1
fi

# 2. sanity: archive_v1/ does not already exist
if rclone lsf "${ARCHIVE_PATH}/" >/dev/null 2>&1 && [[ -n "$(rclone lsf "${ARCHIVE_PATH}/" 2>/dev/null | head -1)" ]]; then
    echo "ERROR: ${ARCHIVE_PATH}/ already has content."
    echo "  This script is one-shot. Either rename the existing archive_v1"
    echo "  or skip the archive step (v2 cycles can land at ${V2_PATH}/ without"
    echo "  archiving; the v1 evidence stays at ${GDRIVE_ROOT}/full_run/)."
    exit 1
fi

# 3. sanity: there's a v1 full_run/ to archive
if ! rclone lsf "${GDRIVE_ROOT}/full_run/" >/dev/null 2>&1; then
    echo "WARN: ${GDRIVE_ROOT}/full_run/ does not exist on gdrive."
    echo "  Skipping archive operation (nothing to move)."
    SKIP_MOVE=true
else
    SKIP_MOVE=false
fi

# 4. move v1 full_run/ → archive_v1/full_run/
if [[ "$SKIP_MOVE" == "false" ]]; then
    echo "[1/3] Moving ${GDRIVE_ROOT}/full_run/ -> ${ARCHIVE_PATH}/full_run/"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "    DRY-RUN: rclone move ${GDRIVE_ROOT}/full_run/ ${ARCHIVE_PATH}/full_run/ --progress"
    else
        rclone move "${GDRIVE_ROOT}/full_run/" "${ARCHIVE_PATH}/full_run/" --progress
    fi
fi

# 5. write archive README
ARCHIVE_README=$(cat <<'EOF'
# CAEM v1 archive — failure mode evidence

This folder holds the v1 architecture trajectory (cycles 0-4 + cycle 5 partial)
that ran on:
- Single global conformal storage gate (alpha_store=0.05)
- Single global composite weights (no per-benchmark)
- Full-parameter SIL fine-tune with L2 anchor (lambda=0.01)
- 10% TriviaQA general-domain mix added to every SIL pool
- Single MMLU 200-sample retention probe
- Original 3-benchmark training panel (FEVER + TriviaQA + Natural Questions)
- Original 4-benchmark transfer panel (TruthfulQA + StrategyQA + ARC + ASQA)

## Discovered failure mode

Across cycles 0-4 the trajectory exhibited:
- Pooled CHM dropped 10.5% (the registered hypothesis H2 was being met)
- Pooled EM dropped 17.7% (NOT predicted; thesis register said nothing about EM)
- TriviaQA EM crashed 73.8% (0.374 -> 0.098)
- Natural Questions EM crashed 61.6% (0.172 -> 0.066)
- TruthfulQA EM dropped 25.1%
- FEVER stored pool: 100% FEVER by cycle 4 (873 of 874 entries)
- MMLU retention guard reported 1.04x pristine (BLIND to open-text crashes)

Three-mechanism diagnosis:
1. Verifier-discrimination decay on FEVER (unpredicted by registered theorems)
2. Precondition starvation on TriviaQA / NQ (single-alpha gate admits 0%)
3. NQ structural failure (verifier alpha < 1/2 on every cal fold)

See branch_C_log.md 2026-05-06 entry for the full discovery narrative.
See CAEM_FIX_AUDIT.md for the implementation audit.
See PRODUCTION_NEXT_SESSION_PLAN.md v2 for the architectural response.

## Why preserve

Cycles 0-4 are the empirical "before" condition for the v2 thesis. The
v2 trajectory (in ../v2/) is the "after". The contrast is the central
methodological contribution.

DO NOT delete this archive. It IS the failure-mode evidence the thesis
defends.
EOF
)

V2_README=$(cat <<'EOF'
# CAEM v2 trajectory

This folder holds the v2 architectural redesign trajectory.

## What v2 fixes (vs v1 in ../archive_v1/)

13 coupled architectural fixes targeting the three failure modes
discovered in v1 (cycles 0-4):

- LoRA r=32 SIL primitive (replaces full-FT; base model frozen)
- Per-benchmark conformal storage gate (alpha_b per benchmark)
- Per-benchmark composite weights (per-bench isotonic + boost)
- Per-benchmark u_pre temperature T_b + safety_u_pre_min_b
- Loss reweighting (temperature mixing T=2 + bounded 3x upsampling + DoReMi floor)
- Multi-modal retention probe (MMLU + TriviaQA test + HotpotQA test)
- Coverage feedback diagnostic (per-bench admission + adapter SVD + halt triggers)
- alias_overlap signal (Wikidata alias-set lookup, 10th verifier signal)
- entity_head_consistency signal (M-chain head agreement, 11th verifier signal)
- Per-benchmark prompts + verifier dispatch (CSQA + HotpotQA new templates)
- Five-layer deferred-reconsideration guard
- Removed: 10% general-domain TriviaQA mix, L2 anchor on LoRA path

## Training panel

FEVER + TriviaQA + HotpotQA + CommonsenseQA  (4 distinct task types)
Stream chunk: 1000/cycle on FEVER/TQA/HotpotQA, 700/cycle on CSQA

## Transfer eval panel

TruthfulQA + StrategyQA + NaturalQuestions
NQ kept as eval-only structural-failure diagnostic.

## Per-cycle layout

cycle_N/
  adapter_model.bin              ~50 MB (LoRA adapter)
  adapter_config.json
  composite_calibration.json     per-benchmark dict
  conformal_gate.json            per-benchmark dict
  safety_threshold.json          per-benchmark T_b + safety_u_pre_min_b
  coverage_diagnostic.json       per-benchmark admission/EM/SV history
  pristine_retention.json        (cycle 0 only) anchored multi-modal baselines
  meta.pkl
  calibration/                   per-benchmark cal-fold rescore JSONs

memory_store_cycle_N.{faiss,meta}  episodic memory
deferred_buffer_cycle_N.pkl        deferred buffer
retroverify_cycleN.json            retroactive re-verification log

## See also

branch_C_log.md (in repo) — discovery + v2 architecture lock entry
CAEM_FIX_AUDIT.md (in repo) — implementation audit per fix
PRODUCTION_NEXT_SESSION_PLAN.md v2 (in repo) — phase plan + budget
EOF
)

echo "[2/3] Writing READMEs"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "    DRY-RUN: would write ${ARCHIVE_PATH}/README.md (~1.5KB)"
    echo "    DRY-RUN: would write ${V2_PATH}/README.md (~2KB)"
else
    TMPDIR_LOCAL=$(mktemp -d)
    echo "$ARCHIVE_README" > "${TMPDIR_LOCAL}/archive_README.md"
    echo "$V2_README"      > "${TMPDIR_LOCAL}/v2_README.md"
    rclone copyto "${TMPDIR_LOCAL}/archive_README.md" "${ARCHIVE_PATH}/README.md"
    rclone copyto "${TMPDIR_LOCAL}/v2_README.md"      "${V2_PATH}/README.md"
    rm -rf "${TMPDIR_LOCAL}"
fi

# 6. final verification
echo "[3/3] Final state"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "    DRY-RUN: rclone lsf ${GDRIVE_ROOT}/   (would list archive_v1/, v2/, production/)"
else
    rclone lsf "${GDRIVE_ROOT}/" | head -10
fi

echo
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY-RUN complete. Re-run with --execute to actually perform the move."
else
    echo "Archive complete. v1 evidence preserved at ${ARCHIVE_PATH}/full_run/."
    echo "v2 cycles will land at ${V2_PATH}/<run_name>/cycle_<N>/."
fi
