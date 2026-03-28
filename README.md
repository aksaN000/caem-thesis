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
  config.py                  ← CAEMConfig: all hyperparameters, annotated [LIT]/[DES]/[CAL]
  memory/
    entry.py                 ← EpisodicEntry + 3 confidence dataclasses + RoutingDecision
    encoder.py               ← QueryEncoder (Sentence-BERT all-mpnet-base-v2, 384-dim)
    store.py                 ← EpisodicMemoryStore (FAISS IndexIDMap + IndexFlatIP)
  confidence/                ← (coming) PreRoutingConfidenceEstimator, PostGenerationConfidenceEstimator
  routing/                   ← (coming) AdaptiveRouter
  verification/              ← (coming) MultiLayerVerifier
  training/                  ← (coming) SelfImprovementLoop

tests/
  test_episodic_memory.py    ← 37 unit tests (no GPU required)

caem-implementation-log.md  ← Authoritative record of all design decisions + alternatives
caem-unified-plan-v3.tex    ← Full thesis plan and system spec
hyperparameter-reference.md ← Three-category hyperparameter breakdown [LIT]/[DES]/[CAL]
writing-suggestions.md      ← Chapter-by-chapter corrections and thesis writing guidance
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
pip install faiss-cpu sentence-transformers torch transformers pytest

# Run unit tests (no GPU needed)
python -m pytest tests/ -v
```

---

## Implementation status

| Module | Stage | Status |
|---|---|---|
| `caem/memory/` | Episodic store | ✅ Complete (Session 11) |
| `caem/confidence/pre_routing.py` | Stage 3 pre-routing | 🔨 Next |
| `caem/routing/router.py` | Adaptive dispatch | ⬜ Pending |
| `caem/confidence/post_generation.py` | Stage 4a Tier-2 gate | ⬜ Pending |
| `caem/verification/verifier.py` | Multi-layer verifier | ⬜ Pending |
| `caem/training/self_improvement.py` | 3-cycle training loop | ⬜ Pending |

---

## Key design decisions

See [`caem-implementation-log.md`](caem-implementation-log.md) for the full record. The most important:

- **FAISS backend:** `IndexIDMap(IndexFlatIP)` instead of IVF-PQ — exact cosine search, no training required, supports `remove_ids()`. At 20k × 384, the index is ~30 MB (negligible vs 15–19 GB VRAM budget).
- **u_hat weights start equal (0.25/0.25/0.25/0.25)** — post-calibration values (~0.20/0.20/0.20/0.40) are projected estimates, not implementation starting values. Actual values fitted after Cycle 1.
- **Retroverify only updates upward** — if the updated model gives a lower (but safe) u_stored, the stored value is kept. Only below-threshold episodes are removed.
