---
name: caem_drafting_session_start
description: START HERE for any CAEM thesis drafting/chapter-review session — the full handoff lives in the repo file SESSION_START.md
metadata:
  node_type: memory
  type: project
  originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
  modified: 2026-10-16T01:11:44.208Z
---

The author runs **one new Claude conversation per thesis chapter** (to avoid compaction degrading quality). Every such
session must begin by reading, in order:
1. `/workspace/caem/defense/thesis_report/ref_check/SESSION_START.md`: complete handoff (author's verbatim brief and
   per-section checklist, git/branch/PR workflow, environment commands, decisions Q1–Q8, carried items per chapter).
2. `.../ref_check/drafting_review_progress.md`: live status, open questions, author decisions.
3. `.../ref_check/REVIEW_HANDOFF.md` (earlier audit) and any chapter-specific pre-found report.

Key rules (details in SESSION_START.md): edit only `defense/thesis_report/`; branch `review/<part>` from the latest
`branch/ruc-classifier` after the previous PR merges; log every change in `fixes_applied.md`; never fabricate; ask
before changing claims/results/verdicts/sections; commit as aksaN000, no AI attribution anywhere.

State at 2026-10-16 (ch5 in progress): ch1–ch4 merged (PRs #5, #7, #8, #9). ch5 rewritten on `review/ch5` (pushed, no PR yet): three-step headline, new evidence tables + 4 figures from scripts/build_ch5_evidence_tables.py and scripts/recompute_validation_gate.py, captions shortened, citations re-checked against PDFs (Fu→Yi). Waiting on: Aksan's group decision on B8 CHM (old broken SE run; GPU rescore vs unavailable), then the paired CHM test, appendix D fix, Aksan's read-through, PR + merge. Repo problems go into ref_check/REPO_CLEANUP_TRACKER.md. Related: [[caem_thesis_finalization_pass]],
[[reference_caem_drive_layout]].
