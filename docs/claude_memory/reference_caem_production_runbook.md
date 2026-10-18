---
name: CAEM has a full deployment-design runbook in docs/PRODUCTION_RUNBOOK.md
description: The deployment design (post-Phase-1a, label-still-required, cyclic-in-production, quarterly-or-accumulation-trigger) is fully documented as a separate engineering deliverable. Always check this file before answering deployment-mode questions.
type: reference
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
The CAEM deployment design — what the system does AFTER research training finishes — is documented end-to-end in:

`/workspace/caem/docs/PRODUCTION_RUNBOOK.md` (last updated 2026-04-27, ~600 lines, operator-grade)

**Audience:** the operator deploying CAEM after Phase 1a + Phase 1b complete.

**Key points the runbook establishes:**

1. **Production is the same architecture as research, NOT a frozen-at-deployment system.** Section 1: "CAEM is cyclic in both research and production. Every cycle ends with the same three operations: SIL fine-tune, recalibration of T + isotonic + conformal τ on a fresh labeled calibration fold, and retroactive re-verification."

2. **Frozen-calibration-between-cycles is not a thing in either mode.** Section 1 line 25 explicit.

3. **Cycle trigger is configurable.** Section 3.4 lists four policies:
   - Accumulation-based (N STORE admissions, N≈1000-3000)
   - Calendar-based (Sunday/monthly/**quarterly**)
   - Hybrid (earliest of N admissions or K days)
   - Drift-triggered (safety override)
   - Sample policy `outputs/production/cycle_policy.json` shows hybrid with N=2000, max_days=60, drift_override=true.

4. **Labels still required at every cycle boundary.** Section 5.2 lists three sourcing options in preference order:
   - Automated post-hoc oracle (stronger LLM judge)
   - User-feedback signals (validated as EM proxy)
   - Human expert review
   - Section 5.2 line 252 explicitly references "quarterly cadence, weeks 1–2" for the labeling pass.

5. **Cycle-boundary sequence in production matches research:** SIL fine-tune first → score cal fold under post-SIL model → refit T → refit isotonic + conformal τ → reload verifier → retroverify → consolidation. Same script paths (`scripts/run_calibration.py`, `scripts/recalibrate_conformal_at_cycle.py`, `SelfImprovementLoop.run_cycle`).

6. **Memory store carries forward.** Section 3.3: production memory store seeded from research-final cycle's FAISS+meta. Subsequent production cycles add to it; periodic retroverify removes stale entries.

7. **Three calibration surfaces in production (Section 2):**
   - T scalar (re-fits each cycle, needs labels)
   - Per-signal isotonic + conformal τ (re-fits each cycle, needs labels)
   - Label-free quantile-only τ EMA (sanity-check mirror, label-free)

**How to apply:**

- For any deployment-mode question, read this file END-TO-END before answering. Do not infer the design from the thesis chapters — the thesis Ch6 §sec:future-deployment frames an empirical deployment STUDY as future work, while the deployment DESIGN is already registered in this runbook.
- For thesis writing: cite `docs/PRODUCTION_RUNBOOK.md` from Ch6 rather than re-deriving the design in the thesis body. The runbook is a separate engineering deliverable — co-deliverable with the thesis.
- The correct one-liner for "is CAEM label-free at deployment": **NO. Label-efficient at training, label-still-required at deployment but sourced periodically (e.g., quarterly) from automated oracle / user-feedback / human-expert review rather than a frozen pre-training cal fold.**
- Worktree copy at `/workspace/caem/.claude/worktrees/agent-aad2027e221164c24/docs/` may exist; main repo path above is canonical.
