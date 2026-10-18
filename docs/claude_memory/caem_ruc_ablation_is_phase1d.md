---
name: caem_ruc_ablation_is_phase1d
description: "The RUC on/off 'ablation' is not a separate run: it compares the earlier Phase 1d trajectory (classifier off) with the Phase 1e main run at C3, the last cycle Phase 1d committed."
metadata:
  type: project
---

Verified 2026-10-16 (ch3 session): `figures/auto/tab_ruc_ablation.tex` is generated from `outputs/analysis/phase1d_vs_phase1e_ruc.json`. Phase 1d (Drive `CAEM_thesis/3_ruc_ablation_phase1d/`) committed cycles 1–3; C4 and C5 aborted under the retention guard; stop locked 2026-05-14 (`run_complete.json`). Hence the comparison point is C3. Phase 1d wall clock ≈134 h (C0–C5: 312.5, 1506.3, 1874.1, 2053.9, 1149.7, 1150.3 min). Main run (Phase 1e) ≈193 h, completed 2026-05-27.

Earlier rejected or superseded runs (author: "mark it as 3"; recalled 3–4, "almost the same cost" as the main run): v1 panel (`9_archive/v1_panel/`), Phase 1c failed (`9_archive/phase1c_failed_2026-05-09/`), Phase 1d. Logs of v1 and Phase 1c not archived. During one run the Vast node's disk filled; data were copied node to node with Vast support and the run resumed from its last checkpoint (author's account).

Related: [[caem_early_stop_gate_artefact]], [[caem_ruc_phase1d_baseline_audit]].
