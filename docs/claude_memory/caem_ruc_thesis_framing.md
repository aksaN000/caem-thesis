---
name: RUC thesis framing — tier-conditional SIL for panel-wide hallucination reduction (2026-05-19)
description: Aksan's articulated thesis hypothesis for the RUC chapter. Vanilla CAEM reduces hallucination via SIL+RAG but is constrained on RAG-hurt benches because RAG-polluted outputs enter the SIL pool. The RUC identifies RAG-helpful (case A) vs RAG-hurt (case B) queries via Path B labels and routes to T3 vs T2 respectively, shifting the SIL pool composition so the adapter learns BENCH-APPROPRIATE patterns. Pooled hallucination reduction across the whole panel — not just the RAG-helpful subset — becomes achievable. This is the core Ch6 §sec:ruc-ablation story and the framing for Ch5 results presentation.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Aksan articulated the RUC's thesis hypothesis on 2026-05-19** during a back-and-forth about RAG-pollution risks. The framing is durable and load-bearing for the Ch5/Ch6 narrative.

## The hypothesis (Aksan's words, formalised)

> CAEM reduces hallucination via SIL + RAG. But on some queries RAG produces worse answers than the parametric base model — those bad answers *increase* hallucination instead of decreasing it. So we need to identify which queries benefit from RAG and which don't. For RAG-helpful queries: reduce hallucination via SIL + RAG. For RAG-hurt queries: reduce hallucination via SIL alone (no RAG contamination).

## How the architecture implements it

| Hypothesis step | Component | Mechanism |
|---|---|---|
| Identify RAG-helpful vs RAG-hurt queries | RUC (Path B classifier) | Trained on `direct_em` vs `rag_em` disagreements; `p_rag ≥ τ → T3` (RAG-helpful), else T2 (RAG-hurt) |
| RAG-helpful → SIL + RAG | Tier 3 dispatch + storage gate → SIL pool | T3 outputs that pass the storage gate become SIL training examples; next cycle's adapter learns from RAG-grounded reasoning |
| RAG-hurt → SIL alone | Tier 2 dispatch + storage gate → SIL pool | T2 outputs that pass the storage gate become SIL training examples; adapter learns from parametric reasoning without RAG contamination |
| Pooled hallucination reduction across panel | CHM trajectory measurement | Per-bench and pooled CHM measured across both RAG-helpful (TriviaQA, NQ) and RAG-hurt (TruthfulQA, CSQA) benches |

## What vanilla CAEM (Phase 1d) couldn't achieve

Vanilla CAEM ran SIL + RAG on all routable queries → SIL pool was dominated by T3 outputs. On RAG-hurt benches (TruthfulQA, CSQA), this meant the adapter learned partially from RAG-polluted outputs that themselves contained adversarial-myth amplification or commonsense-confused content. Pooled CHM reduction was therefore a weighted average of "where SIL+RAG worked" (TriviaQA, NQ — strong reduction) and "where SIL+RAG actively hurt" (TruthfulQA, CSQA — modest or negative reduction). The RAG-hurt benches dragged the pooled number down.

The empirical receipts from Phase 1d:
- Pooled CHM dropped 22% (0.156 → 0.121) C0 → C3
- Per-bench CHM dropped 25-29% on 4 of 5 benches
- TruthfulQA showed weaker improvement; SIL drift visible C0→C5
- HaluEval, OpenBookQA, NQ not in the 5-bench panel — additional v2 evidence pending

## What the RUC unlocks (the Ch6 contribution)

**Tier-conditional SIL**. Phase 1e's SIL pool composition shifts so:

| Bench | Vanilla SIL pool (Phase 1d) | RUC-mediated SIL pool (Phase 1e, predicted) | What SIL learns |
|---|---|---|---|
| TruthfulQA | ~95% T3 outputs (myth-polluted) | ~50% T3 + ~50% T2 outputs | Less myth amplification; cleaner abstention pattern |
| CSQA | ~85% T3 outputs | ~55% T3 + ~45% T2 outputs | Less commonsense confusion |
| TriviaQA | ~95% T3 outputs | ~90% T3 (RUC keeps these on T3) | Same as vanilla — RAG-grounded factoid recall |
| Natural Questions | ~95% T3 | ~95% T3 | Same as vanilla — open-domain factoid recall |

The thesis claim: **CAEM with the RUC achieves hallucination reduction across BOTH RAG-helpful and RAG-hurt benches via tier-appropriate SIL learning**, where vanilla CAEM achieved it only on the RAG-helpful subset.

## Evidence needed (Phase 1e measurements that validate the hypothesis)

| Evidence | Predicted result | Diagnostic that produces it |
|---|---|---|
| (a) Per-bench EM steady/improving on RAG-helpful benches | TriviaQA, NQ EM holds or slightly improves vs Phase 1d | A1 per-bench EM trajectory |
| (b) Per-bench CHM drops MORE on RAG-hurt benches | TruthfulQA CHM -35%, CSQA CHM -17% vs Phase 1d | A1 per-bench CHM trajectory |
| (c) Routing distribution confirms the split | TruthfulQA T3-share ~95% → ~50%; TriviaQA stays ~90% | A2 routing distribution per (bench, cycle) |
| (d) SIL learns differently on the two arms | T2 EM growth on TQA-like benches > Phase 1d's T2 EM growth | Per-tier EM diagnostic (caem_per_tier_em_selection_effect already established the methodology) |

If all four land in the predicted direction, the Ch6 headline lands:
> "CAEM with the RUC reduces hallucination across the panel — by SIL+RAG on RAG-helpful queries and by SIL alone on RAG-hurt queries. Without the RUC, vanilla CAEM's pooled hallucination reduction is constrained by RAG contamination on adversarial-myth and commonsense benches; with the RUC, each query reaches the SIL training pipeline best suited to its retrieval-utility profile."

## Honest failure-mode framing

If Phase 1e doesn't confirm one or more of (a)-(d), the thesis pivots to:

| Failure | Honest reframe |
|---|---|
| RUC routes well, SIL on T2-only outputs doesn't reduce CHM | "RUC alters routing but T2-output SIL signal is too weak to drive learning at this storage-threshold setting — investigate sample-quality threshold" |
| RUC routes well, pooled EM regresses on RAG-helpful benches | "τ tuning too aggressive — RUC's τ optimal on the training pool may shift on the eval pool" |
| RUC routes well, CHM drops on RAG-hurt benches but p > 0.05 | "Architecturally cleaner routing, trend in expected direction, not statistically significant at panel scale" |
| Retention guard fires earlier in Phase 1e than Phase 1d | "RUC alters trajectory dynamics — the early stop is itself a finding about RUC-mediated SIL pool composition" |

All four are publishable. The thesis story doesn't depend on a specific magnitude of CHM reduction; it depends on the architectural contribution being honestly demonstrated.

## How to apply this when writing

1. **Lead Ch5 §sec:ruc-architecture** with Aksan's framing: "CAEM's vanilla SIL+RAG pipeline is constrained on RAG-hurt benches. The RUC's role is to enable tier-conditional SIL."
2. **In Ch5 §sec:results**, present per-bench CHM/EM with the tier-conditional lens: "On RAG-helpful benches X/Y, CAEM-with-RUC matches vanilla CAEM. On RAG-hurt benches A/B, CAEM-with-RUC reduces CHM by Z% beyond what vanilla CAEM achieved."
3. **In Ch6 §sec:ruc-ablation**, lead with the architectural contribution: "The RUC unlocks bench-conditional SIL learning by routing each query to the tier whose training signal is most informative."
4. **In Ch6 §sec:future-work**, name the corpus-coverage-version-skew issue as adjacent future work (also captured in conversation 2026-05-19).

## Cross-references

- `caem_v2_panel_motivation.md` — v2 panel design that enables this framing
- `caem_per_tier_em_selection_effect.md` — per-tier EM methodology for evidence (d)
- `caem_b1_vs_c0_framing.md` — B1→C3 vs C0→C3 isolation framing
- `caem_three_headline_decomposition.md` — three-headline decomposition for abstract
