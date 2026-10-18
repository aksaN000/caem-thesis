---
name: CAEM headline contrast is B1 → C3, NOT C0 → C3 — three-comparison decomposition (2026-05-15)
description: Cycle 0 is full-architecture-minus-SIL, not raw Qwen. Headline thesis claim is raw-Qwen-to-CAEM gap; intra-trajectory C0→C3 is only the incremental-SIL slice. Reframe Ch5 §summary-headline once B1 lands.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**The framing point (raised by user 2026-05-15 06:00 BDT):** Current trajectory reporting compares CAEM C0 → C3 (pooled EM +0.2 pp, CHM −22%). But C0 is already the full CAEM architecture (verifier + retrieval + memory + gate + abstain) just *without SIL fine-tuning*. The headline thesis claim should be **raw Qwen → CAEM C3**, which the current intra-trajectory comparison drastically understates.

## The three-comparison decomposition

Per Ch5 §sec:comp-zero-shot, the baseline panel structures into three independent contrasts:

| Comparison | Isolates | Magnitude (estimated until B1 lands) |
|---|---|---|
| **B1 → CAEM C3** | Full architecture + SIL (headline) | ~+7 pp EM pooled, ~−60-70% CHM relative |
| **B1 → CAEM C0** | Architecture-only contribution (no SIL) | ~+7 pp EM pooled, ~−50% CHM relative |
| **CAEM C0 → C3** | SIL incremental on top of architecture | +0.2 pp EM, −22% CHM (measured) |
| **B6 → CAEM C3** | SIL with full architecture vs vanilla FT | TBD post-baselines |

## Estimated per-bench B1 → C3 EM gaps (will verify after B1 runs)

| Bench | B1 raw Qwen (est) | CAEM C0 (actual) | CAEM C3 (actual) | Δ B1→C3 (est) |
|---|---|---|---|---|
| FEVER | ~0.38 | 0.450 | 0.503 | +12 pp |
| TriviaQA | ~0.25 | 0.407 | 0.437 | +19 pp (RAG dominates here) |
| CommonsenseQA | ~0.57 | 0.607 | 0.603 | +3 pp (MC; parametric already strong) |
| TruthfulQA | ~0.28 | 0.330 | 0.387 | +11 pp (transfer) |
| StrategyQA | ~0.55 | 0.607 | 0.617 | +7 pp (multi-hop helped by RAG) |
| Pooled | ~0.41 | 0.480 | 0.482 | ~+7 pp |

CHM gap will be MUCH larger than EM gap: raw Qwen has no abstain class, no gate, no grounding → unmitigated confident wrong answers. Estimated B1 CHM ~0.30-0.40; CAEM C3 CHM 0.121 → ~60-70% relative reduction.

## Why this reframe matters

The C0 → C3 reading makes SIL look like a marginal incremental lever (it is, intra-trajectory). The B1 → C3 reading shows the full value proposition. The thesis story isn't "SIL fine-tuned the model a little" — it's "the architecture + SIL together turn a 3B base model into a system that hallucinates ~65% less and answers ~7 pp more correctly than raw zero-shot, while progressively shifting load off retrieval (the cost-amortisation arm)."

**SIL's job in the thesis story is to shift T2 share up (1.5% → 18.1%), not to be the dominant accuracy lever.** It's a *cost-reduction* mechanism on top of an *accuracy-and-hallucination* architecture. The C4 abort doesn't undermine the headline — most of the headline gain comes from architecture-not-SIL.

**Why:** Reviewers will read the C0 → C3 number and ask "is +0.2 pp EM and −22% CHM enough to justify a 4000-line architecture?" The answer is yes, but only when framed against B1, not C0.

**How to apply:** When B1 lands (post-C5 baseline panel, Step P-2 in PRODUCTION_NEXT_SESSION_PLAN.md Phase 1.6):

1. **Ch5 §sec:summary-headline** — currently a placeholder. Lead with B1 → C3 gap, not C0 → C3 gap. Reframe the headline sentence around the raw-Qwen contrast.
2. **Ch5 §sec:disc-headline** — emphasize the three-way decomposition. State clearly that the C0 → C3 trajectory measures SIL alone, and the architecture's broader value sits in the B1 → C0 gap.
3. **Ch5 §sec:comp-zero-shot** — already explains B1 as the empirical floor; can be tightened to make the floor explicit in numbers.
4. **Ch6 §sec:concl-headline** — finalize with the licensing-rule outcome on the B1 → C3 contrast (not C0 → C3).
5. **Abstract** — quote the B1 → C3 numbers, not C0 → C3.

## Caveats

- B1 EM estimates are educated (literature-derived for Qwen-2.5-3B-Instruct zero-shot on each benchmark family); actual numbers depend on our exact eval fold + normalization + scoring. ±5 pp uncertainty.
- CHM gap estimate is the rougher of the two — depends heavily on what threshold "confident_confabulation" uses for raw Qwen. Likely 0.30-0.50 absolute CHM for B1, but verify.
- The "B1 → C0" gap (architecture without SIL) is partially evidenced by C0 already being on disk; we just need B1 to subtract. This is the cleanest way to license the "architecture is responsible for most of the gain" claim.
- SIL's contribution is then specifically the **cost-amortisation arm** (T2 share growth) plus the modest CHM-reduction increment (−22% on top of architecture's larger reduction). Both should be reported, but neither is the headline.

## Methodology pre-registrations for the B1 → C3 comparison (user-raised 2026-05-15)

When `scripts/rescore_baselines_through_verifier.py` is run on B1 outputs post-C5, two methodological choices must be explicit in the paper, not improvised in rebuttal:

### Composite-version pre-registration

The composite is refit at every cycle boundary (per-bench shrinkage prior at α=0.6, plus per-bench T_b). Cycle 0's composite differs from cycle 3's. **The B1 rescore must use the SAME composite that CAEM C3 was scored against** — that's the cycle-3 composite at `outputs/full_run/cycle_3/composite_calibration.json`. Pass it explicitly:

```
python -m scripts.rescore_baselines_through_verifier \
    --composite_calibration outputs/full_run/cycle_3/composite_calibration.json \
    --baselines B1 B2 B3 B4 B5 B6 B7 \
    ...
```

The rescorer takes `--composite_calibration` as a CLI argument with default `outputs/production/composite_calibration.json`. **Do not rely on the default** — pin the path to the cycle-3 composite explicitly so reviewers can verify the methodology section's claim. State in Ch5 §sec:setup-metrics or §sec:sig-pairing: *"Every baseline's signals are computed through the same locked cycle-3 verifier composite (`outputs/full_run/cycle_3/composite_calibration.json`), so the measurement instrument is held fixed across CAEM and all seven baselines."*

### Cross-distribution-application caveat

The composite was fit on CAEM's signal distributions over the training-panel calibration fold. Applying it to B1 outputs is a **cross-distribution application** — B1's per-signal distributions (especially u_token, p_ground_*) will differ from CAEM's because B1 has no retrieval and no abstain class. Strictly, this is fine for matched-protocol purposes: the verifier is being used as a **fixed measurement instrument**, not as a fitted predictor for B1's distribution.

But the paper must state this explicitly. Reviewer playbook: "the composite under-discriminates on B1 because it wasn't fit on B1's distribution → your CHM comparison is unfair." The defense is the standard cross-system evaluation logic: *we want a fixed instrument that scores both systems on the same scale, even if neither system is the instrument's optimal operating point.*

Suggested Ch5 §sec:disc-threats paragraph:

> The composite probability used to score the confident-confabulation, factual-fabrication, and logical-fabrication subtypes was fit on the architecture's own calibration-fold signal distribution. When the same composite is applied to the external baseline panel to compute CHM under matched protocol, it is being used as a fixed measurement instrument rather than as a fitted predictor for the baseline's signal distribution. This is the standard cross-system evaluation pattern in the natural-language-inference literature: the same instrument scores every system on the same scale, even when the baseline's signal distribution differs from the calibration-fold distribution on which the instrument was originally fit. A composite refit per baseline would defeat the matched-protocol comparison, since the headline contrast would then conflate generator differences with measurement-instrument differences. The choice is therefore architecturally honest at the cost of placing the baselines at a measurement operating point that was not optimised for their distribution.

This paragraph belongs near §sec:disc-threat-gap (the calibration-evaluation-gap threat) in Ch5.
