---
name: caem_repo_cleanup_tracker
description: "Every code/data/table problem found in the drafting pass is logged in defense/thesis_report/ref_check/REPO_CLEANUP_TRACKER.md for the plug-and-play repo phase after the chapter pass; add rows as new issues appear."
metadata:
  type: project
---

Aksan (2026-10-16): "how will i make the repo plug and play with so many changes and errors. please keep track of everything." The tracker lists code bugs (C*), data/provenance problems (D*), stale or hand-edited tables (T*), dangerous scripts (S*) and the new scripts to keep. **How to apply:** whenever a session finds a script bug, a hand-edited table, a stale output or a provenance issue, add a row there in the same commit. The clean-up phase runs after all chapters (before the final defence): environment file, one command per stage, generators reproducing every committed table, archive superseded scripts, README.

Related: [[caem_early_stop_gate_artefact]], [[caem_ch5_decisions]], [[caem_drafting_session_start]].
