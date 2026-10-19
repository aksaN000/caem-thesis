# Archive

Material that is no longer part of the working thesis or the experiment pipeline, kept for provenance.
Nothing in this folder is read by the code, the tests or the thesis build. Do not edit files here; if
something is needed again, copy it out and say so in the commit message.

Reorganised on 2026-10-17 (branch `cleanup/repo-layout`); file history is preserved (`git log --follow`).

| Folder | What it holds | Why archived | What replaced it |
|---|---|---|---|
| `thesis_report_2026-10-15/` | The thesis as it stood before the final drafting pass (chapters, figures, auto tables, `ref_check/` from the October audit) | Frozen snapshot; the drafting pass works on a copy | `defense/thesis_report/` |
| `presentation_2026-06/` | Supervisor slide deck (`caem_supervisor.tex`), poster and poster template, poster guide | Built for the June 2026 milestone; content predates the re-run plan | New slides and poster after the re-run |
| `early_reports/` | Pre-thesis report, first literature review drafts (v1, v2), BRAC thesis template reference | Superseded by the full thesis | `defense/thesis_report/` |
| `design_logs/` | Design diaries and planning notes from the build period: `branch_C.md`, `branch_C_log_historical.md`, `caem-implementation-log.md`, the historical next-session plan, hyperparameter reference, research notes, writing suggestions | Dated decision history; may disagree with the code | The code (source of truth) and `defense/thesis_report/ref_check/MASTER_ISSUES.md` |
| `run_logs/` | Vast.ai session log, deployment guide, Phase 2 execution log, the agent runbook used on the rented GPU | Records of past runs | A new runbook (Phase 1a, R6, in MASTER_ISSUES.md) |
| `audits/` | Earlier code, script, metric and chapter audits; `thesis_audit_2026-10-15/` holds the October number/table/reference audit working files and helper scripts | Findings carried into the tracker and the thesis | `defense/thesis_report/ref_check/` (REPO_CLEANUP_TRACKER.md, MASTER_ISSUES.md, reference checks) |
| `docs_superseded/` | Demo quickstart and panel demo script (the live demo was removed from the thesis); the previous root README | Out of date | The root `README.md` |
| `scripts_superseded/` | (2026-10-19 additions: `patch_chm_judged_tables.py` and `aggregate_baseline_pooled.py`, replaced by `scripts/build_final_tables.py`, which writes whole tables from one source; `compare_b6_old_vs_new.py` and `postprocess_b6_c5_eval.py`, run-1 B6 evaluation helpers, replaced by the re-run.) Old runners; one-off watchdog/restart shell scripts from the Vast sessions; one-off diagnostics and smoke tests; the approximate significance script `baseline_sig_tests_v2.py` | Not imported by any code or test; superseded or single-use | `scripts/` (see the root README) and `scripts/paired_baseline_tests.py` |

Files that were checked and deliberately **kept** in place:
- `scripts/launch_baselines_v2.sh`: the command that generated the current baseline runs (provenance, per `caem/ruc/README.md`).
- `scripts/eval_b6_c5_only.py`: evaluates B6 from its saved cycle-5 weights (used by the 2026-10-19 baseline re-run).
- `scripts/archive_v1_to_gdrive.sh`, smoke and diagnostic scripts used by `tests/`.
