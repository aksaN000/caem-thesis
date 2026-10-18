---
name: Three-headline decomposition for the abstract and Ch5 (2026-05-15)
description: Strong-paper version reports B1→C3, C0→C3, and B1→C0 in a single table with explicit isolation labels. Defuses the "how much is just RAG" reviewer objection in advance. Abstract template and table structure registered for post-B1 writing.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Decision (locked 2026-05-15 07:45 BDT):** The paper reports THREE headlines side-by-side, not a single number. Each isolates a different mechanism. A weak paper picks the largest. A strong paper reports all three with clear labels so reviewers can't surface a hidden number to undercut the headline.

## The three-headline table (belongs in Ch5 §sec:summary-headline)

| Comparison | Isolates | Magnitude (template — fill from data) |
|---|---|---|
| **B1 → CAEM C3** | Deployment-time architecture + training-time SIL (the full system) | CHM −X% rel., EM +A pp |
| **C0 → CAEM C3** | Training-time SIL increment alone (intra-trajectory) | CHM −22% rel. (measured), EM +0.2 pp (measured) |
| **B1 → CAEM C0** | Deployment-time architecture alone (retrieval + verifier + memory + gate + abstain) | CHM −Y% rel., EM +B pp |

Decomposition identity: X% ≈ Y% (architecture) + 22% (SIL increment, intra-trajectory).

Bonus headline (already measured): **T2 (zero-shot fine-tuned) EM = 0.620 vs base T3 (raw Qwen + RAG) EM = 0.476 → +14.4 pp** — the distilled model's zero-shot outputs exceed the base-model-with-retrieval pipeline. Direct receipt for SIL-as-distillation, independent of the three-headline structure.

## Why this preempts reviewer objections

| Objection a reviewer would raise | Defense baked into the three-headline structure |
|---|---|
| "You scored your system with your own verifier" | Matched-protocol cross-system evaluation: same locked cycle-3 composite scored both CAEM and every baseline (caveat documented in §disc-threats per `caem_b1_vs_c0_framing.md`) |
| "How much of the gain is just RAG, not your contribution?" | B1 → C0 row credits RAG explicitly; the SIL contribution is the C0 → C3 row, reported separately |
| "Most of your gain is just deployment-time tricks (abstain class), not real model improvement" | C0 → C3 row isolates training-time SIL; the bonus T2-vs-base-T3 finding shows the SIL actually distils into the parametric model |
| "Your C0 → C3 trajectory is small, why fine-tune at all?" | SIL's job is the **cost-amortisation arm** (T2 share growth 1.5% → 18.1%) plus the modest CHM-reduction increment; the architecture carries the accuracy lever, SIL carries the cost lever |

## Abstract template (post-B1, ~80-100 words)

User-drafted 2026-05-15:

> CAEM reduces pooled hallucination metric by [X%] relative to a raw Qwen-2.5-3B-Instruct baseline under matched cross-system evaluation, with [Y%] of the reduction attributable to deployment-time mechanisms (retrieval, multi-signal verification, episodic memory, abstain class) and [Z%] to training-time self-improvement via verifier-gated distillation. The distilled model's zero-shot outputs further exceed the base-model-with-retrieval pipeline by 14.4 points exact match, demonstrating that calibrated self-distillation produces a parametric model that improves on its retrieval-augmented source.

Variables to fill post-B1:
- **X%** = pooled CHM reduction, B1 → CAEM C3 (projected ~60-70%)
- **Y%** = pooled CHM reduction, B1 → CAEM C0 (projected ~50%)
- **Z%** = X% − Y%, the SIL-attributable increment (projected ~10-20% relative; ~22% intra-trajectory but the share differs because Y measures architecture's larger absolute reduction)

Decomposition phrasing must use **relative-of-total** semantics, not subtractive — Y% + Z% > X% if expressed as raw percentages because CHM doesn't decompose linearly. Better phrasing:

> "...with the deployment-time mechanisms contributing the larger share (CHM dropping from [B1 CHM] to [C0 CHM]) and training-time self-improvement contributing an additional 22% relative reduction on top (CHM dropping from [C0 CHM] to [C3 CHM = 0.121])."

This phrasing is multiplicatively honest and matches the measurement structure.

## Rating context (user-noted 2026-05-15)

User's project rating: **9/10**, conditional on B1 landing in the projected range. The methodological rigor of the post-hoc scoring plan (matched-protocol + composite-pin + cross-distribution caveat) reduces projection uncertainty but doesn't change the rating because the rating was already conditional on B1 landing roughly in this range.

- B1 lands at upper end (CHM 0.35+ → ~−65% relative): rating could move to 9.5
- B1 lands at lower end (CHM ~0.25 → ~−50% relative): rating stays at 9
- Both publishable at strong venues

## How to apply (post-B1, Step P-7 in PRODUCTION_NEXT_SESSION_PLAN.md)

1. **Ch5 §sec:summary-headline**: replace placeholder with the three-row table above, populated with real B1/C0/C3 numbers.
2. **Ch5 §sec:disc-headline**: lead paragraph quotes the matched-protocol framing verbatim ("Under matched-protocol cross-system evaluation, CAEM C3 reduces pooled hallucination metric by [X%] relative to the raw Qwen-2.5-3B-Instruct baseline.") followed by the per-row decomposition rationale.
3. **Ch6 §sec:concl-headline**: same three-row structure, abstract-length version.
4. **Abstract** (`thesis_report/core/abstract.tex`): rewrite to the template above, with all three magnitudes + the T2-vs-base-T3 distillation finding.
5. **Optional Ch5 paragraph**: explicitly state which reviewer objection each row preempts (this is unusual for a thesis but adds defensive structure; consider for the published version even if not the thesis-bound version).
