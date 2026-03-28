"""
CAEM — Confidence-Aware Episodic Memory with Self-Improvement
=============================================================
Episodic memory module for hallucination reduction in Flan-T5-Large.

Package structure
-----------------
caem/
  config.py          — CAEMConfig: all hyperparameters in one place
  memory/
    entry.py         — EpisodicEntry, PreRoutingConfidence,
                       PostGenerationConfidence, StoredConfidence, RoutingDecision
    encoder.py       — QueryEncoder (Sentence-BERT, 384-dim)
    store.py         — EpisodicMemoryStore (FAISS-backed)
  (coming next)
  confidence/        — PreRoutingConfidenceEstimator, PostGenerationConfidenceEstimator
  routing/           — AdaptiveRouter
  verification/      — MultiLayerVerifier
  training/          — SelfImprovementLoop
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

__all__ = [
    "CAEMConfig",
    "EpisodicEntry",
    "PreRoutingConfidence",
    "PostGenerationConfidence",
    "StoredConfidence",
    "RoutingDecision",
    "QueryEncoder",
    "EpisodicMemoryStore",
]
