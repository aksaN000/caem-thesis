---
name: Thesis rewrite workflow v2 — direct per-subsection full rewrite
description: Updated thesis-rewrite methodology (2026-04-24, folder corrected 2026-05-09) — skip claim-inventory; full rewrite per subsection in chapter order; live folder is thesis_report/ NOT pre thesis 1 report/.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
Supersedes the earlier "3-phase per-section (Phase 1 prep → user-approved artifact → Phase 2 draft → Phase 3 coherence)" workflow for chapter rewrites.

**The rule:** pick a subsection, read the relevant code + `branch_C_log.md`, then do a full rewrite in one pass. Do not produce intermediate claim-inventory tables for user review before writing.

**Why:** User feedback 2026-04-24 after §3 rewrite — the Phase-1 artifacts are accurate but slow down the work and obscure the actual writing output. The user wants to see finished prose, not drift catalogues.

**Live thesis folder (corrected 2026-05-09):** `thesis_report/`. The earlier `pre thesis 1 report/` folder is STALE — do not edit it. All rewrites land in `thesis_report/chapters/chapter_*.tex`. Verified by user: "our report is thesis report folder. not pre thesis 1. thats the stale one".

**How to apply:**
1. Enumerate subsections in chapter order.
2. For each subsection, in order:
   a. Read the existing subsection in `thesis_report/`.
   b. Read the relevant code files line-by-line (no skip — this rule from earlier memory still applies).
   c. Cross-reference `branch_C_log.md` ONLY as supplementary rationale. Code is the source of truth — see `feedback_thesis_code_first.md`. The log can drift; never transcribe a claim from the log without verifying the code.
   d. Write the full rewrite, preserving all `\label{}`, `\ref{}`, `\cite{}`, equation numbers, algorithm numbers.
   e. Apply via Edit tool in one pass.
3. Move to the next subsection.

**What to stop doing:** the "Phase 1 claim inventory → user approval → Phase 2 draft" dance. User has already authorised the rewrite scope; the job is to write, not to ask.

**Autonomy (2026-05-09):** user explicitly told me to "start automatically the methodology update, don't ask for permission" once they've set the goal. Default to full autonomous execution of the rewrite scope; only pause if a drift would require restructuring or breaking cross-chapter references.

**What to keep doing:** line-by-line code reads, label/ref/cite preservation, cross-reference verification. Those are correctness requirements, not workflow overhead.
