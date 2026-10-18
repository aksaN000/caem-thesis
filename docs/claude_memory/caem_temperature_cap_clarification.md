---
name: CAEM Step 2.2 temperature cap is principled, not a Flan-T5 artifact
description: Why T-refit is bound-clipped at e^3 ≈ 20.09 from cycle 1 onward; what it does and does not affect; thesis hedging guidance.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
The CAEM Phase 1a per-cycle temperature re-fit (Step 2.2 in run_experiment.py:1656) hits the registered upper bound of `_LOG_T_HIGH = 3.0` (i.e., T = e^3 ≈ 20.086) from cycle 1 onward and stays pinned through every subsequent cycle. The optimizer wants T outside the bound but is bound-clipped.

**Why:** scripts/run_calibration.py:177-186 sets `[exp(-3), exp(3)] = [0.05, 20.1]` as a principled regularizer. T near 20 means sigmoid(logit/20) is in its near-linear region around 0.5, so the score's discriminative structure collapses. The cap prevents the optimizer from minimizing ECE by destroying signal — a known failure mode of single-parameter temperature scaling against severely overconfident generators (Guo et al. 2017; Kumar et al. 2019). Same logic applies to Qwen-3B as to Flan-T5; the cap is not backbone-specific despite the docstring referencing Flan-T5.

**Why: Apply To Empirical Trajectory:**

| Cycle | T_before | T_after | ECE_before | ECE_after |
|---|---|---|---|---|
| 1 | 1.000 | 20.086 | 0.332 | 0.150 |
| 2 | 20.086 | 20.086 | 0.177 | 0.167 |
| 3 | 20.086 | 20.086 | 0.219 | 0.209 |
| 4 | 20.086 | 20.086 | 0.276 | 0.265 |

Source: outputs/full_run/calibration/calibrated_config_cycle{1-4}.json (locally) and gdrive:caem-phase1a/full_run/calibration_global/ (cycles 1-2 only on gdrive currently).

**How to apply:**

1. **Don't uncap mid-run.** Three independent reasons: (a) methodology drift mid-experiment breaks thm:convergence stationarity, (b) re-running cycles 1-4 with uncapped T costs ~$60-70 + 7 days, (c) uncapping makes the calibration worse — it lets T → ∞ and crushes u_pre to 0.5 universally.

2. **Pinning at the cap is correct behaviour.** The cap is doing its job — preserving u_pre's discriminative power at the cost of imperfect probabilistic calibration.

3. **u_stored is NOT affected.** T scales u_pre (the routing-time confidence used by the AdaptiveRouter), not the verifier composite u_stored. u_stored is calibrated by per-signal isotonic regression at Step 2.3, which continues to refit each cycle on EM-labelled cal fold. The conformal storage contract (π_t ≥ 0.95) is preserved at every cycle by construction (depends on quantile structure of u_stored, not on probabilistic interpretation).

4. **Thesis hedge required, not theorem failure.** The thesis "calibrated probability composite" framing (Ch4 §sec:composite-design, Ch6 §concl-contributions first contribution) refers to u_stored — which IS calibrated (isotonic + conformal). The u_pre routing-confidence calibration claim weakens monotonically with cycles but does not break the theorems. Add a Threats-to-Validity paragraph to Ch5 §sec:threats:

   > "The per-cycle Step 2.2 temperature re-fit reaches the registered ceiling of T = e^3 ≈ 20.09 from cycle 1 onward. The ceiling is a principled regularizer that prevents temperature scaling from degenerate solutions in which the entire score distribution collapses toward 0.5 to minimize expected calibration error at the cost of discriminative power. T saturating at the ceiling indicates that u_pre alone — independent of T — carries strong calibration drift across cycles; the system absorbs this drift through the Step 2.3 per-signal isotonic refit and conformal threshold refit on the composite u_stored, both of which continue to track EM-labelled cal-fold drift each cycle. The conformal storage contract holds at every cycle (π_t ≥ 0.95 throughout) by construction. The u_pre routing-confidence claim weakens but does not break: u_pre still preserves rank order over the routing-relevant range even at saturated T."

5. **Phase 1b future-work item (registered, not actioned in Phase 1a):** "T-cap relaxation diagnostic" — single-cycle 36h experiment that re-fits cycle 0 with `_LOG_T_HIGH = 5.0` to characterize where unrestricted NLL optimum lives and whether unrestricted T meaningfully improves u_pre ECE. Cycle-0-only diagnostic, NOT a full trajectory rerun. Do this only after Phase 1a closes.

6. **What CAEM is, label-wise — corrected framing:**
   - Label-EFFICIENT during training: 1500-sample EM-labelled cal fold per cycle drives Step 2.3 isotonic + conformal refit.
   - NOT label-free at deployment. The deployment design is fully documented in `/workspace/caem/docs/PRODUCTION_RUNBOOK.md` (dated 2026-04-27): same cyclic architecture, calendar-or-accumulation trigger (with quarterly cadence as a listed option), labeled calibration fold built from accumulated serving log per cycle, labels sourced from automated oracle / user-feedback / human-expert-review depending on deployment stakes. Operator section 5.2 line 252 explicitly references "quarterly cadence, weeks 1–2" for the labeling pass. So deployment cycles continue to consume EM-class labels — they just come from periodic human/oracle review rather than a frozen pre-training cal fold.
   - Correct one-liner: "label-efficient at training, label-still-required at deployment but sourced periodically from human review / oracle / user-feedback rather than a frozen gold fold."
   - Thesis status: Ch6 §sec:future-deployment frames an empirical deployment STUDY as future work. The deployment DESIGN is already registered in PRODUCTION_RUNBOOK.md as a separate engineering deliverable. Cite the runbook from Ch6 rather than re-deriving the design in the thesis body.
