---
name: reference_caem_drive_layout
description: "Google Drive layout for CAEM after the 2026-10-15 reorganisation — folder names, what is public vs private, rclone remote"
metadata:
  node_type: memory
  type: reference
  originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
  modified: 2026-10-15T14:01:46.751Z
---

Reorganised 2026-10-15 (server-side renames, folder IDs preserved). rclone remote `gdrive:` (full scope) works from WSL.

- `CAEM_thesis/` (ID 1xErhsRDYe_qO0ZgUBn0WXK81eHc5jZCa, **public: anyone with link = viewer**; GitHub README links it by ID). Was `caem-phase1a/`.
  - `README.md` start-here + old-name → new-name map
  - `1_thesis/main.pdf` final PDF
  - `2_results_phase1e/` = every thesis number (was `v2_1_phase1e_main/`): full_run, phase1e_baselines (B1–B8), cycle_0_cold_start, tier1_routing_probe, logs. Mirrors local `outputs/` (verified DIFFER 0 / LOCAL ONLY 0).
  - `3_ruc_ablation_phase1d/` (was v2_1_phase1d_archive), `4_ruc_classifier_build/` (was v2_1_phase1e_2026-05-19)
  - `5_audit_2026-10/` consistency_checks, regen_check, regen_check_run1
  - `9_archive/` superseded_2026-10 (replaced files, never deleted), v1_panel, phase1c_failed_2026-05-09, pre_main_snapshot
- `CAEM_references/` (owner-only): ~100 cited-paper PDFs + fetch_ref_pdfs.py. Was `CAEM refs/` (nested folder flattened).
- `CAEM_vast_backup_2026-05-29/` (owner-only): raw Vast home-dir mirror, was `vast_to_other_machine/`. Was PUBLIC until 2026-10-15 and exposed `.claude/.credentials.json` + a GitHub token in `caem/.git/config` → never re-share it.

Rules: data flows local → Drive only; replaced files go to `9_archive/superseded_2026-10/`. Bulk `rclone move` into an existing folder fails ("fatal error"); use per-file moves or Drive API parent-change with backoff (403 = rate limit). Related: [[caem_thesis_finalization_pass]].

**Reorganised 2026-10-19 (author: "1 full folder for caem and 1 full for baselines"; Drive only, local paths unchanged):** `2_results_phase1e/` = `caem/` (full_run, cycle_0_cold_start, tier1_routing_probe, logs, scoring_logs, explorations/verifier_v2) + `baselines/` (ALL final B1–B8: rerun_v2 files + run-1 rag/cot_rag non-TQA + vanilla_ft/eval + vanilla_ft/weights/cycle_1..5, retrieval_cache, logs/rerun_v2, logs/run1; README per folder) + `analysis/`. Old run-1 baselines + logs: `9_archive/superseded_2026-10/run1_baselines/`; judge backups: `9_archive/superseded_2026-10/rerun_v2_backups/`; pre-scoring versions: `9_archive/superseded_2026-10/2026-10-19_final_scoring/`. Drive is no longer a path-for-path mirror: `2_results_phase1e/README.md` has the Drive→local mapping table. Verified file-by-file (MD5) after the move. `1_thesis/main.pdf` still the 2026-10-15 PDF (replace when thesis final).

**2026-10-19 (later):** `1_thesis/main.pdf` replaced with the final draft (182 pages, md5 4c9606db…, git 39875c6; the 210-page version is in 9_archive/superseded_2026-10/thesis_pdf_2026-10-19_210p/); old 2026-10-15 PDF in `9_archive/superseded_2026-10/thesis_pdf_2026-10-15/`; root README row updated (old README in `.../readme_root_before_pdf_2026-10-19/`).
