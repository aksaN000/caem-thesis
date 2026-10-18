---
name: HF dataset repo — CAEM passage index + Phase 1a pre-main snapshot
description: aksaN000/caem-passage-index-21m (private HF dataset) holds both the 64GB passage FAISS index (Step 4.5 archive) and the Phase 1a pre-Step-7 recovery snapshot under pre_main_snapshot/ subfolder.
type: reference
originSessionId: 2f06df76-4467-4646-9e32-0a9234c27ac4
---
Private HF dataset repo: **aksaN000/caem-passage-index-21m**

Contents:
- `passages.faiss` (64.5 GB) + `passages.pkl` (7.4 GB) — 21M-passage FAISS
  index archived 2026-04-19 (Step 4.5 of runbook).
- `pre_main_snapshot/` subfolder (auto-uploaded by `run_phase1a.sh` step
  `step_hf_upload_pre_main` before Step 7 main run starts) — holds
  Step 6 cold-start memory, Step 7.0 Cycle-0 eval + calibrated thresholds
  + cycle-0 memory/deferred snapshots, Step 5.5 v5 head-to-head,
  Step 19.2.1 retention slice, audit logs, and a MANIFEST.json.

Recovery workflow if Vast instance is destroyed before Step 7 completes:
1. Rent fresh Vast 5090.
2. `hf download aksaN000/caem-passage-index-21m --include 'passages.*' 'pre_main_snapshot/*' --local-dir .` then unpack into `data/passage_index/` and `outputs/`.
3. `./run_phase1a.sh` — idempotency skips Step 6, 7.0, 5.5, prompt-ablation,
   Level B smoke, and retention slice; goes straight to `step_7_main`.

Note: Step 7's own per-cycle checkpoints (`outputs/full_run/cycle_N/`) are
NOT auto-uploaded — user accepts the risk of losing mid-Step-7 work because
Vast auto-stops (not destroys) on balance=$0, and top-up restores the disk
within the 24-48h grace window.
