"""
CAEM — Confidence-Aware Episodic Memory with Self-Improvement
=============================================================
Episodic memory module for hallucination reduction in Flan-T5-Large.

Package structure
-----------------
caem/
  config.py          — CAEMConfig: all hyperparameters in one place
  pipeline.py        — CAEMPipeline: end-to-end orchestrator (all 8 stages)
  memory/
    entry.py         — EpisodicEntry, PreRoutingConfidence,
                       PostGenerationConfidence, StoredConfidence, RoutingDecision
    encoder.py       — QueryEncoder (Sentence-BERT, 384-dim)
    store.py         — EpisodicMemoryStore (FAISS-backed)
  confidence/
    pre_routing.py   — PreRoutingConfidenceEstimator (Stage 3a)
    post_generation.py — PostGenerationConfidenceEstimator (Stage 4a, Tier 2 only)
  routing/
    router.py        — AdaptiveRouter (Stage 3b)
  verification/
    verifier.py      — MultiLayerVerifier (Stage 5)
  retrieval/
    rag.py           — PassageStore, TierThreeRAG (Stage 6)
  training/
    self_improvement.py — SelfImprovementLoop (Stage 8)
"""

from caem.config import CAEMConfig
from caem.memory.entry import (
    EpisodicEntry,
    PostGenerationConfidence,
    PreRoutingConfidence,
    RoutingDecision,
    StoredConfidence,
)
from caem.memory.encoder import QueryEncoder
from caem.memory.store import EpisodicMemoryStore
from caem.pipeline import CAEMPipeline, PipelineResult

__all__ = [
    "CAEMConfig",
    "CAEMPipeline",
    "PipelineResult",
    "EpisodicEntry",
    "PreRoutingConfidence",
    "PostGenerationConfidence",
    "StoredConfidence",
    "RoutingDecision",
    "QueryEncoder",
    "EpisodicMemoryStore",
]
