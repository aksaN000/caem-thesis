---
name: All decisions, experiments, and iterations must be presentable as evidence in the thesis report
description: Every parameter choice, gate result, iteration, and design decision must leave an auditable on-disk artifact (JSON / TeX / log / archive folder) that can be cited in the thesis chapter so a reviewer cannot question the choice.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
The user is writing a CSE400 thesis where Ch5 / Ch6 / discussion sections must
defend every architectural and parameter decision with empirical evidence.
"No one can question us" requires a complete audit trail.

## Rule

For every decision made during runs (composite design, threshold values,
α values, dedup choices, validation gate iterations, restart decisions,
etc.):

1. **Generate an on-disk artifact** that captures the decision input + output.
   - JSON for numerics
   - TeX for thesis-ready tables
   - PNG/PDF for figures
   - Plain-text/markdown for narrative log entries
2. **Place it under `outputs/`** in a stable path (cycle_N, calibration/,
   archive/, phase4/) so the thesis pipeline can ingest it.
3. **Append a dated entry to `branch_C_log.md`** when a decision is made,
   citing the artifact paths and the reasoning.
4. **Archive failed-state artifacts BEFORE overwriting** them. Use
   `outputs/archive/<situation_label>_YYYY-MM-DD_TN/` paths.
5. **Cite the evidence** when reporting a decision back to the user, so
   the user can verify and so the citation is ready for the report.

## Specifically for Phase 3 / Step 7.0 iterations

Each iteration of fit_composite + fit_gate + validate must produce:
- `composite_calibration.json` (per-signal isotonic + Cherian boost weights)
- `conformal_gate.json` (τ_store, τ_defer, store_n, store_precision)
- `weight_validation.json` (gate verdict + per-benchmark numbers)
- `calibrated_thresholds.json` (legacy backward-compat)
- `calibrated_config.json` (T scalar + ECE before/after)
- An archive copy of any prior iteration's failed artifacts under
  `outputs/archive/post_failed_gate_YYYY-MM-DD_TN/`
- A dated entry in `branch_C_log.md` documenting the iteration

If the validation script is checking the wrong artifact (e.g., bootstrap
decisions instead of fitted decisions), write a one-shot rescorer + a
companion validation report so both views are on disk.

## Why

Without on-disk evidence, every decision is "trust me." With on-disk evidence,
every decision is verifiable. The user wrote a 5-chapter thesis; reviewers
will ask "why this α?" "why drop this signal?" "why not bigger N?" — every
answer needs to point to a specific JSON/TeX/log file.

## How to apply

When a decision is being made or a numerical result lands:
- Ask: is there a file on disk that captures this?
- If not, write one (or modify a script to write one) BEFORE proceeding.
- When reporting the decision to the user, cite the artifact path.
- When iterating, archive the prior state.
- Before destructive operations (delete / overwrite), archive first.
