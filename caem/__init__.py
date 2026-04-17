"""
CAEM -- Confidence-Aware Episodic Memory with Self-Improvement
=============================================================
Episodic memory module for hallucination reduction in Flan-T5-Large.

Package structure
-----------------
caem/
  config.py          -- CAEMConfig: all hyperparameters in one place
  pipeline.py        -- CAEMPipeline: end-to-end orchestrator (all 8 stages)
  memory/
    entry.py         -- EpisodicEntry (nine-signal schema),
                       PreRoutingConfidence, PostGenerationConfidence,
                       RoutingDecision
    encoder.py       -- QueryEncoder (Sentence-BERT, 768-dim)
    store.py         -- EpisodicMemoryStore (FAISS-backed)
  confidence/
    pre_routing.py   -- PreRoutingConfidenceEstimator (Stage 3a)
    (post-generation signals are produced inside UnifiedVerifier;
     the legacy PostGenerationConfidenceEstimator module was removed
     in Session 42 -- see caem/confidence/__init__.py for the audit.)
  routing/
    router.py        -- AdaptiveRouter (Stage 3b)
  verification/
    verifier.py      -- UnifiedVerifier (Stage 5, nine-signal gate)
  retrieval/
    rag.py           -- PassageStore, TierThreeRAG (Stage 6)
  training/
    self_improvement.py -- SelfImprovementLoop (Stage 8)
"""

from caem.config import CAEMConfig
from caem.memory.entry import (
    EpisodicEntry,
    PostGenerationConfidence,
    PreRoutingConfidence,
    RoutingDecision,
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
    "RoutingDecision",
    "QueryEncoder",
    "EpisodicMemoryStore",
]
