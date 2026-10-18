---
name: caem_repo_layout_2026-10-17
description: "Repo reorganised 2026-10-17 on branch cleanup/repo-layout: old thesis_report/ -> archive/thesis_report_2026-10-15/, presentation/ -> archive/presentation_2026-06/, categorised archive/ with README index, new root README, outputs/README.md; local git objects repaired with fetch --refetch."
metadata:
  type: project
---
Live thesis is ONLY `defense/thesis_report/`. Old paths in older memories (`thesis_report/...`, `branch_C_log.md`, `PRODUCTION_NEXT_SESSION_PLAN.md`, `docs/DEMO_QUICKSTART.md`, ref_check audit files like FINAL_FIX_LIST/local_regen_check.py) now live under `archive/` (see `archive/README.md`). SIL poison-drift diagnosis is `docs/findings/SIL_POISON_DRIFT_FINDINGS.md`. Branch order: merge review/ch5 first, then cleanup/repo-layout (branched from review/ch5).
**Why:** author asked to categorise stale files with a broad README (2026-10-17). Local repo had 9 missing git objects (prefix 30); fixed by `git fetch --refetch origin` — check `git fsck --connectivity-only` if git errors "unable to read tree".
**How to apply:** follow README/archive README for paths; never edit archive/; superseded scripts in archive/scripts_superseded/. Related: [[caem_master_issues]].

**Merged + branched (2026-10-17):** PR #10 (review/ch5) and PR #11 (cleanup/repo-layout) merged into branch/ruc-classifier; tag `run1-final-2026-10-17` = frozen run-1 state; ALL re-run work happens on branch `final-version` (author named it), with per-phase sub-branches/PRs into it. First commit there: import guards on aggregate_section_5_* + replay_early_stop_gate main().
