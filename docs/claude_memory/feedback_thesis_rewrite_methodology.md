---
name: Thesis rewrite methodology — 3-phase per-section process (prep → draft → coherence)
description: For the CAEM CSE400 thesis Ch1-6 rewrite (optimality reframe), every section goes through 3 phases: (1) Preparation — read old section AND relevant branch_C_log/branch_C.md/Master Plan entries, inventory content as KEEP/MODIFY/REPLACE/ADD, decide primary-vs-secondary placement; (2) Drafting — only after Phase 1 user-approved; (3) Coherence review — cross-refs, registry counts, terminology. Never skip Phase 1.
type: feedback
originSessionId: 2f06df76-4467-4646-9e32-0a9234c27ac4
---
**Rule:** when working on the CSE400 CAEM thesis rewrite (files under
`pre thesis 1 report/`), every section goes through a locked 3-phase
process before any .tex edit.

**Why:** on 2026-04-23 01:00 BDT, user corrected me for jumping straight
to drafting without preparation. I marked §1.3 for full REPLACE when
it only needed one new bullet + expanded wrap-up. The fix: preparation
work is required BEFORE any proposal or draft.

**Phase 1 — Preparation (required, no exceptions):**

1a. Read current section fully in the .tex file.
1b. Pull concepts from:
    - `branch_C_log.md` (grep for topic keywords; read related
      `[DECISION]` / `[IMPL]` / `[NOTE]` / `[PERF]` entries)
    - `branch_C.md` (architectural decisions)
    - Master Plan entry (2026-04-23 00:50 BDT in branch_C_log.md)
    - Config state (`caem/config.py` for current defaults)
    - Phase 1a empirical data (outputs/cycle_0/, outputs/full_run/)
1c. Inventory existing content into 5 categories:
    - KEEP (primary framing-worthy under optimality)
    - KEEP (secondary framing — system description)
    - MODIFY (valid but needs reframing)
    - REPLACE (contradicted by new framing)
    - ADD (new content required)
1d. Decide primary-vs-secondary placement. Primary layer =
    ceiling-attainment / optimality claim. Secondary = system
    description / mechanisms / implementation.
1e. Match content to Master Plan elements: theorems referenced,
    audit tables (5.A-G) previewed, baselines/ablations positioned,
    Ch1-6 rewrite TODO items satisfied.

**Output:** a single-page "Phase 1 artifact" — inventory, placement
decisions, cross-refs, contribution mapping. USER REVIEWS this
artifact BEFORE drafting begins.

**Phase 2 — Drafting (only after Phase 1 approval):**
- Edit the .tex file using surgical Edit calls
- Single commit per section
- Clear git message describing primary vs secondary changes

**Phase 3 — Coherence review (after drafting, before moving on):**
- Forward-reference check (labels in future chapters)
- Backward-reference check (terms/concepts in earlier chapters)
- Registry count check (ablations 19→? signals 9→10, benchmarks 6, etc.)
- Terminology consistency (Flan-T5 → Qwen-2.5-3B; RoBERTa → MiniCheck;
  9-signal → 10-signal; 780M → 3B; etc.)

**Default behaviors:**
- Default to KEEP, not REPLACE, when in doubt
- Surgical edits preferred over wholesale rewrites
- Preserve existing motivation/personal-story sections (user's 2024
  Claude hallucinations) — they're load-bearing under any framing
- The three existing Problem Statement bullets (stateless, calibration,
  SIL absence) are STILL RELEVANT under the new optimality framing;
  they describe WHY existing literature operates below the ceiling

**Folder decision (user question 2026-04-23 01:30 BDT):** modify the
existing `pre thesis 1 report/` folder. Do NOT create a new folder.
Git history is the versioning mechanism; folder duplication breaks
git continuity, duplicates unchanged LaTeX scaffolding, and makes
coherence review harder.

**Time budget per section:** ~1 hour (20 min Phase 1 + 30 min Phase 2
+ 10 min Phase 3). Estimated ~15 sections across Ch1-6 → ~15 focused
writing hours across 2-3 weeks.

**Critical path ordering** (from Master Plan 00:30 BDT):
  1. Ch4 §4.1 ceiling derivation
  2. Ch4 §4.9 theorems
  3. Ch1 §1.1 (ties to Ch4 §4.1)
  4. Ch1 §1.3, §1.4
  5. Ch2 positioning
  6. Rest in parallel
  7. Coherence pass at end

**When this memory applies:** every session that touches
`pre thesis 1 report/chapters/*.tex`. Skip Phase 1 only if explicitly
told to, and log the override in branch_C_log.md.
