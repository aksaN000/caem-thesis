---
name: Thesis writing style — top-tier paper, no code pollution, no EM dashes
description: Style rules for the CSE400 thesis report (chapters/, core/, appendix/, bibliography/). Voice and prose conventions plus a hard ban on code/script/variable/folder names appearing in body prose.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
The user wants the thesis to read like a top-tier paper, not technical
documentation. Authored 2026-04-26 during the Phase A audit setup.

## Rule 1 — No EM dashes anywhere

The user wrote: "No EM dashes anywhere". Replace EM dashes (—) with commas,
colons, parentheses, or sentence breaks as the prose flow suggests. EN dashes
are acceptable in numeric ranges only (e.g., "0.05–0.15"). Hyphens are fine in
compounds.

## Rule 2 — No section symbol (§); use clickable references

The user wrote: "no § sign, use direct clickable reference". In LaTeX use
`\Cref{label}` / `\cref{label}` / `\ref{label}` / `\autoref{label}` instead of
"§". Render as "Section 5.1" or "Chapter 4" via the cleveref package, not "§5.1"
or "§ Sec. 5.1". Pre-existing § symbols in current chapters must be removed
during the rewrite.

## Rule 3 — No code/variable/script/folder names in body prose

The user wrote: "no code variable type names (like 'composite hallucination
metric in eval/metrics.py. it mentions folders, variables like roberta_nli, or
script name chm_table.py, verifier backend and so on.) these do not belong to a
thesis paper right? (this is not a technical documentation)".

Specifically BAN from body prose:
- Module paths (e.g., `caem/verification/verifier.py`)
- Class/function/variable names (e.g., `CalProbComposite`, `_fit_boost`,
  `tau_store`, `roberta_nli_backend`)
- Script names (e.g., `scripts/run_calibration.py`, `chm_table.py`)
- Folder names (e.g., `outputs/cycle_0/`, `tables/`)
- CLI flags (e.g., `--alpha_store 0.05`)
- File names (e.g., `composite_calibration.json`, `weight_validation.json`)
- Backend strings (e.g., `verifier_backend=minicheck`)

These belong in:
- Code comments and docstrings (live source)
- Appendix A (implementation reference, if needed for reproducibility)
- Footnotes (sparingly, only when the artifact name is a non-obvious reference)
- Replication subsections that explicitly target re-execution

In body prose, write the CONCEPT instead:
- "the calibrated probability composite" (not `CalProbComposite`)
- "the storage threshold" (not `tau_store`) or "$\tau_{\text{store}}$"
- "the verifier backend" (not `verifier_backend`) — wait, the user listed
  "verifier backend" itself as too low-level. Prefer "the natural language
  inference judge" or "the entailment scorer".
- "the conformal calibration script" (not `scripts/calibrate_thresholds.py`)
- "the empirical-evidence artifacts" (not `outputs/phase4/*.tex`)

## Rule 4 — Voice and density: top-tier thesis paper

Voice: third-person formal academic. "We" is acceptable when describing the
authoring decision; "the architecture", "the system", "the verifier" preferred
when describing the artifact. Passive voice acceptable for results-reporting
("X was measured"). No first-person singular.

Density: each subsection introduces one concept, defends it, and forward-refs
the next. Paragraphs three to six sentences. Equations introduced in body via
prose ("Storage requires the composite to exceed the calibrated threshold..."),
labeled, then referenced. Tables and figures have full captions readable
without body context.

## Rule 5 — Things flow naturally

The user wrote: "things should flow naturally". Avoid abrupt section
transitions, heavy jargon front-loading, or paragraph-to-paragraph topic jumps.
Each subsection ends with a sentence that motivates the next. Each chapter ends
with a brief signpost paragraph for the chapter that follows.

## Rule 6 — Literature review handled separately

The user wrote: "for the literature review, we need to sit down for it
seperately". Do NOT auto-rewrite chapter 2 (or any related-work subsections)
during the Phase C rewrite. Flag any literature-review prose for separate
review.

## Rule 7 — Numerical claims must match on-disk evidence

Every number in body prose (precision, threshold value, dataset size, etc.)
must be traceable to a JSON file under `outputs/`. Citations to the JSON are
unnecessary in body prose (per Rule 3) but the audit must confirm body numbers
match the JSON, and the appendix/replication note can list source paths.

## How to apply during rewrite (Phase C)

1. Open the subsection.
2. Read the live code that owns the concept.
3. Read the on-disk JSON evidence that owns the numbers.
4. Write prose using concept words, not source-tree words.
5. Numbers go inline as values; equations get labeled and clickable-referenced.
6. End paragraphs with motivation for the next. End subsections with motivation
   for the next subsection.
7. After writing, sweep for: EM dashes, § symbols, file paths, function names,
   variable names, script names, backend strings.
