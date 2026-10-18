---
name: Thesis edits must increase coherence, never break it
description: Every thesis edit must preserve terminology, cross-references, registry counts, fold semantics, and forward/backward references across all chapters, the bib, the runbook, and the code.
type: feedback
originSessionId: 661400c9-9e7f-4fa5-b313-84d9aee2012e
---
When editing any thesis chapter, runbook, bib, or code that the thesis
references, every change must either preserve or strengthen
coherence. Drift of any kind is a regression even if the local edit
looks fine.

**Why:** the user explicitly flagged after several edits in a long
session that modifications should "never break coherence instead
increase it." Thesis-level coherence is the defense-grade property
that reviewers check first — inconsistencies between chapters,
between thesis and code, or between runbook and chapters read as
sloppy work regardless of how correct any single chapter is in
isolation.

**How to apply:**

1. **Before editing**, scan for every co-dependent mention:
   - Numbers declared once (variant count, n_cal=500, rho_min=0.93,
     tau_train=0.75) must match in every reference. grep before and
     after.
   - Terminology picked once (e.g. "n_cal calibration fold" not
     mixing "calibration set" / "calib samples" / "calibration
     split") must stay consistent.
   - Forward references (Ch3 "see Ch5 Table tab:X") must match
     Ch5's actual label and content.

2. **Place content in the right chapter**:
   - Theorem-scope statements: Ch4 (methodology).
   - Reporting conventions / which number to quote: Ch5 (results).
   - Risk + mitigation: Ch3 (requirements).
   - Literature grounding + precedent: Ch2 (related work).
   - Implementation-level code details: stay in code; thesis
     references them via paragraph prose, not by duplicating.

3. **When adding citations**, they must appear in Ch2 prose AND bib
   AND be referenced from Ch3/4/5 where relevant. Citations in bib
   alone but not cited in prose are dead weight.

4. **When adding a new mechanism or parameter**:
   - Add to CAEMConfig if applicable.
   - Cite in Ch4 methodology paragraph.
   - Declare in Ch3 requirements / constraints paragraph if it is a
     design commitment.
   - Pre-register expected behavior in Ch5.
   - Add to ablation registry if it is a knob worth ablating.
   - Add runbook command if it is an operational step.
   - Add to the coherence check list here for future reminders.

5. **When changing a number or parameter** that was already in
   production:
   - Update every chapter that cites it (Ch1 /Ch2/Ch3/Ch4/Ch5).
   - Update CAEMConfig defaults if applicable.
   - Update runbook invocations.
   - Update CH_AUDIT_TRACKING.md audit entries.
   - Update VAST_SESSION_LOG.md with the change rationale.

6. **Phase / Step naming**: runbook's Phase 1a / Phase 1 Full are
   budget/funding partitions, orthogonal to Ch5's Phase 1 / Phase 2
   which are seed-count partitions. Always state the mapping
   explicitly where both terminologies appear in the same document.

7. **Validation after edits**:
   - `grep -n "<old term>" *.tex` returns 0 hits
   - LaTeX balance check passes
   - Cross-referenced labels exist (grep `\label{X}` for each `\ref{X}`)
   - Tests still pass if code was touched
   - Commit message explicitly names the coherence relationship
     ("Gap 1 moved from Ch4 to Ch5 because reporting convention is
     a results-chapter concern").

Never commit a thesis edit without walking this list.
