---
name: caem_thesis_finalization_pass
description: "CAEM thesis defense-prep finalization — working-copy folder layout, code-first rule, and chapter order"
metadata: 
  node_type: memory
  type: project
  originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
  modified: 2026-10-16T00:49:23.079Z
---

Started 2026-10-14 (BDT). Goal: finalize the thesis chapter-by-chapter for defense, and build the defense presentation. User forgot much of the report after ~4 months, so each chapter doubles as a re-understanding pass.

**Folder layout (new):** `/workspace/caem/defense/` holds two subfolders — `presentation/` (defense talk) and `thesis_report/` (a working COPY of the real `thesis_report/`). Edit ONLY `defense/thesis_report/`; the original `/workspace/caem/thesis_report/` stays safe/unchanged as the baseline.

**Hard rule (user-emphasized, "super important"):** trust the actual `caem/` CODE over the report prose for any factual claim — anything in the report can be doubted; resolve doubts by reading raw code. Extends [[feedback_thesis_code_first]] and [[feedback_read_code_after_compaction]].

**Chapter map:** Ch1 Introduction (417ln) · Ch2 Literature Review (313ln, frozen) · Ch3 Requirements/Impact/Project-Mgmt (256ln) · Ch4 Methodology and Design (696ln, the architecture = maps to code) · Ch5 Performance Evaluation and Analysis (614ln, the results = maps to outputs/) · Ch6 Conclusion and Future Work (96ln). Appendices A–G.

**Finalization order (SUPERSEDED 2026-10-16 by user):** now front matter → Ch1 → Ch6 in sequence → appendices → bib. (Old core-first plan Ch4→5→6→3→1 dropped.) Rationale: Ch4/Ch5 are the code/data ground truth that every other chapter makes promises about; lock them first so intro/conclusion can be finalized to match reality without rework.

**Workflow (user, 2026-10-16):** the final drafting pass edits ONLY `defense/thesis_report/`. It is now TRACKED in git (baseline commit f058ec6 on branch/ruc-classifier); one branch per part (`review/front`, `review/ch1` … `review/appendices`, never "claude" in names), one PR each into branch/ruc-classifier, user merges; commit as aksaN000 <aksangoni.alif@gmail.com>, no AI attribution/footers. Change log = `defense/thesis_report/ref_check/fixes_applied.md` (heading per chapter, before → after, evidence); status + open questions = `defense/thesis_report/ref_check/drafting_review_progress.md`. Order: step 1 p_ground check (done: identical on 14,951/14,951 samples, length gate + NO_FACTS fallback), front matter, Ch1→Ch6, appendices A–G, bib, whole-doc pass. Per section: read → ranked findings → apply allowed fixes → short summary; ask before changing claims/results/verdicts/sections. Full rules: `thesis_report/ref_check/REVIEW_HANDOFF.md` + the user's pasted brief.

**Update 2026-10-15:** after the October audit (PRs #2/#3, panel-b restore), `defense/thesis_report/` was REPLACED with the audited thesis at commit `22f78c0` (324 pp). Old copy had no user edits. Finalization resumes on this copy. Known polish item: 34 pre-existing overfull hboxes >10pt. Drive layout: [[reference_caem_drive_layout]].

Thesis is already submission-ready (group A done: baseline tables regenerated Jun 13, Ch5 uses Haiku-judged TruthfulQA, PDF rebuilt). This pass is polish + defense readiness, not a rewrite. Style rules still apply: [[feedback_thesis_writing_style]], [[feedback_thesis_coherence]], [[feedback_truthfulqa_llm_judge_em]].
