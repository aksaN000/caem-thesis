#!/usr/bin/env bash
# Build the upload bundle for the rental (local machine, final-version 2026-10-17).
#   bash scripts/rental/make_bundle.sh <out.tgz>
# Contents: the committed repository at HEAD (git archive, plus a COMMIT file) and the run-1 files the runbook needs:
# outputs/full_run/eval (CAEM evaluation fold and run-1 passages for the search check), the C3 adapter and composite
# (verifier pass, as in run 1) and the dataset splits. About 300 MB. The B6 weights (5.8 GB) are fetched from Drive
# on the rental or uploaded separately. Upload: scp <out.tgz> <host>:/workspace/ && ssh <host> 'mkdir -p /workspace/caem
# && tar xzf /workspace/<out.tgz> -C /workspace/caem'
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
OUTFILE="$(readlink -f "${1:?usage: make_bundle.sh <out.tgz>}")"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "commit first: tracked files have changes"; exit 1; }
TMP="$(mktemp -d)"
git archive HEAD | tar x -C "$TMP"
git log --oneline -1 > "$TMP/COMMIT"
tar c outputs/full_run/eval/*_cycle[0-9].json outputs/full_run/cycle_3/adapter \
    outputs/full_run/cycle_3/composite_calibration.json outputs/full_run/dataset_splits.json | tar x -C "$TMP"
tar czf "$OUTFILE" -C "$TMP" .
rm -rf "$TMP"
ls -lh "$OUTFILE"
