# CAEM — Confidence-Aware Episodic Memory with Self-Improvement

**CSE400 Final Year Thesis** | BRAC University
**Student:** Aksan Gony Alif

---

## Overview

CAEM is a hallucination-reduction system built on **Flan-T5-Large** (780M parameters). It combines:

- **Episodic memory** — a FAISS-backed store of verified (question, reasoning, answer) triples
- **Confidence-aware routing** — three-tier dispatch based on pre-routing confidence and memory similarity
- **Multi-signal verification** — NLI + self-consistency + semantic entropy pipeline
- **Self-improvement loop** — 3 cycles of verified rejection sampling with L2-regularised fine-tuning
- **Retroactive re-verification** — memory quality updated after each cycle with the improved model

CAEM is an **architectural intervention**, not post-hoc detection. It prevents unreliable knowledge from accumulating in memory and progressively improves generation quality through a closed self-improvement loop.

---

## Repository structure

```
caem/
  config.py                      ← CAEMConfig: all hyperparameters, annotated [LIT]/[DES]/[CAL]
  pipeline.py                    ← CAEMPipeline — full 8-stage orchestrator
  memory/
    entry.py                     ← EpisodicEntry + confidence dataclasses + RoutingDecision
    encoder.py                   ← QueryEncoder (Sentence-BERT all-mpnet-base-v2, 384-dim)
    store.py                     ← EpisodicMemoryStore (FAISS IndexIDMap + IndexFlatIP)
  confidence/
    pre_routing.py               ← Stage 3a — PreRoutingConfidenceEstimator (u_pre)
    post_generation.py           ← Stage 4a — PostGenerationConfidenceEstimator (û, Tier 2)
  routing/
    router.py                    ← Stage 3b — AdaptiveRouter (Tier 1/2/3 dispatch)
  verification/
    verifier.py                  ← Stage 5 — MultiLayerVerifier (NLI + SC + SE → û_stored)
  retrieval/
    rag.py                       ← Stage 6 — TierThreeRAG (DPR + FAISS passage store)
  training/
    self_improvement.py          ← Stage 8 — SelfImprovementLoop (3-cycle fine-tuning)

eval/
  metrics.py                     ← EM, F1, FEVER acc, hallucination rate, bootstrap CI, McNemar
  benchmarks.py                  ← HotpotQA / TruthfulQA / FEVER / StrategyQA loaders
  harness.py                     ← EvalHarness — run, score, save JSON

tests/
  test_episodic_memory.py        ← Memory store + search_with_ids
  test_confidence.py             ← Pre- and post-generation confidence estimators
  test_routing.py                ← AdaptiveRouter + RoutingDecision
  test_verifier.py               ← MultiLayerVerifier
  test_rag.py                    ← TierThreeRAG
  test_pipeline.py               ← Full CAEMPipeline (Tier 1/2/3 paths)
  test_self_improvement.py       ← SelfImprovementLoop
  test_eval.py                   ← eval/ metrics, benchmarks, harness

caem-implementation-log.md      ← Authoritative record of all design decisions + alternatives
caem-unified-plan-v3.tex        ← Full thesis plan and system spec
hyperparameter-reference.md     ← Three-category hyperparameter breakdown [LIT]/[DES]/[CAL]
writing-suggestions.md          ← Chapter-by-chapter corrections and thesis writing guidance
```

---

## Benchmarks

| Dataset | Failure mode tested | Primary verifier |
|---|---|---|
| HotpotQA | Multi-hop reasoning errors | Self-consistency (SC) |
| StrategyQA | Implicit chain-of-thought failures | Self-consistency (SC) |
| FEVER | Factual claim verification | NLI (entailment) |
| TruthfulQA | Systematic misconceptions / inverse scaling | Semantic entropy (SE) |

---

## Quick start

```bash
pip install faiss-cpu sentence-transformers torch transformers datasets pytest scipy

# Run all unit tests (no GPU required)
python -m pytest tests/ -v
# Expected: 361 passed
```

---

## Implementation status

**Implementation phase: COMPLETE.** All 8 pipeline stages and the full evaluation harness are implemented. Next phase: experiments.

| Module | Stage | Status |
|---|---|---|
| `caem/memory/` | Episodic store (FAISS) | ✅ Sessions 11, 20 |
| `caem/confidence/pre_routing.py` | Stage 3a — u_pre | ✅ Session 12 |
| `caem/routing/router.py` | Stage 3b — Tier dispatch | ✅ Session 13 |
| `caem/confidence/post_generation.py` | Stage 4a — û (Tier 2) | ✅ Session 14 |
| `caem/verification/verifier.py` | Stage 5 — MultiLayerVerifier | ✅ Session 15 |
| `caem/retrieval/rag.py` | Stage 6 — Tier 3 RAG | ✅ Session 16 |
| `caem/training/self_improvement.py` | Stage 8 — Fine-tune loop | ✅ Sessions 17, 20 |
| `caem/pipeline.py` | Full orchestrator | ✅ Sessions 18, 20 |
| `eval/` | Metrics + benchmarks + harness | ✅ Sessions 19, 20 |
| Calibration harness | Post-Cycle 0 temperature scaling | 🔬 Experiment phase |

---

## Key design decisions

See [`caem-implementation-log.md`](caem-implementation-log.md) for the full record. The most important:

- **FAISS backend:** `IndexIDMap(IndexFlatIP)` instead of IVF-PQ — exact cosine search, no training required, supports `remove_ids()`. At 20k × 384, the index is ~30 MB (negligible vs 15–19 GB VRAM budget).
- **u_hat weights start equal (0.25/0.25/0.25/0.25)** — post-calibration values (~0.20/0.20/0.20/0.40) are projected estimates, not implementation starting values. Actual values fitted after Cycle 1.
- **Retroverify only updates upward** — if the updated model gives a lower (but safe) u_stored, the stored value is kept. Only below-threshold episodes are removed.
