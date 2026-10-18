---
name: CAEM has TWO NLI judges — MiniCheck + Frozen Qwen, length-dispatched
description: Architectural fact — MiniCheck and Frozen Qwen are TWO NLI judges behind AdaptiveNLIJudge, dispatched by hypothesis length (≤408 MC-tokens → MiniCheck, >408 → Frozen Qwen). Both produce the same NLI signals.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
CAEM's NLI grounding is served by **two interchangeable NLI judges** behind one
interface (`caem/verification/adaptive_nli_judge.py::AdaptiveNLIJudge`),
dispatched by hypothesis length:

  - hypothesis ≤ 408 MC-tokens → **MiniCheck** (primary, fast claim-support NLI)
  - hypothesis > 408 MC-tokens → **Frozen Qwen judge** (fallback,
    Qwen-2.5-3B with Platt-calibrated entail head, 131k context)

408 = 512 − OVERHEAD(4) − MIN_PREMISE(100) = max hypothesis that fits
MiniCheck's encoder without truncation. Frozen Qwen's calibration is fitted on
a 500-sample overlap fold against MiniCheck (`scripts/calibrate_qwen_judge.py`)
so outputs land on a comparable scale for `u_stored` composition.

If `qwen_judge=None`, all samples route to MiniCheck (with downstream
truncation on long hypotheses, logged as a warning).

## What both judges produce (same signals)

Both judges expose `batch_entail_prob` / `batch_contradict_prob` /
`batch_argmax_label`. They feed the same downstream NLI-grounding signals:
`p_entail`, `p_ground_max`, `p_ground_mean`, `p_ground_atomic`. They are
interchangeable judges, NOT different signal sources.

## Where the OTHER signals come from (NOT the Qwen judge)

`u_token`, `u_dropout`, `u_internal`, `s_avg`, `h_norm`, `q_a_relevance` are
produced by the verifier's own token-distribution / self-consistency analysis
of the generator — they do not come from the Frozen Qwen NLI judge. Don't
conflate "Frozen Qwen judge" (the long-hypothesis NLI fallback) with
"Frozen Qwen self-consistency" (a misattribution).

## Length gate (atomic_min_tokens=12) is upstream of both judges

The FActScore-style atomic-decomposition length gate decides whether atomic
decomp runs at all. When it does, the per-fact hypotheses are routed to
MiniCheck or Frozen Qwen by `AdaptiveNLIJudge` based on length. The gate is
not a road switch and not a judge switch.

## How to apply / Why

When discussing where a fix lands or what a signal measures, name the right
component: NLI grounding signals come from the dispatched judge (MiniCheck or
Frozen Qwen judge); self-consistency / log-prob signals come from the
verifier's generator-introspection path. Saying "Frozen Qwen produces u_token"
is wrong; saying "Frozen Qwen judge handles long-hypothesis NLI when MiniCheck
would truncate" is correct.

**Update 2026-10-16:** in the realised main run the Qwen judge was INACTIVE. `outputs/full_run/experiment.log` L20: "AdaptiveNLIJudge inactive ... using bare MiniCheck judge ... ablated at Phase 2 after the Platt-fit Pearson floor (rho >= 0.70) was not cleared (measured rho = 0.5843)". Thesis ch3/ch5/appendix B say "ablated", which is correct.
