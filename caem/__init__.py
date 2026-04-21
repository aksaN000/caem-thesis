"""
CAEM -- Confidence-Aware Episodic Memory with Self-Improvement
=============================================================
Decoder-only Qwen-2.5-3B-Instruct stack (Branch C); episodic memory with
a seven-family verifier composite and cycle-boundary self-improvement.

Package structure
-----------------
caem/
  config.py          -- CAEMConfig: all hyperparameters in one place
  pipeline.py        -- CAEMPipeline: end-to-end orchestrator
  model_loader.py    -- load_base_generator (Qwen-3B + ChatML + pad_token)
  prompts.py         -- ChatML Tier 2 / Tier 3 prompt builders
  memory/
    entry.py         -- EpisodicEntry (seven-family composite schema),
                       PreRoutingConfidence, RoutingDecision
    encoder.py       -- QueryEncoder (Sentence-BERT, 768-dim)
    store.py         -- EpisodicMemoryStore (FAISS-backed) +
                       retroverify + consolidate + force_retroverify_queue
    deferred.py      -- DeferredBuffer (cycle-boundary reconsideration)
  confidence/
    pre_routing.py   -- PreRoutingConfidenceEstimator (Stage 3a)
  routing/
    router.py        -- AdaptiveRouter (Stage 3b)
  verification/
    verifier.py      -- UnifiedVerifier (Stage 5, seven-family composite
                       including q_a_relevance)
    minicheck.py     -- MiniCheck-Flan-T5-Large judge adapter
  retrieval/
    rag.py           -- PassageStore, TierThreeRAG (Stage 6; adaptive nprobe)
  training/
    self_improvement.py -- SelfImprovementLoop (Stage 8) + loop_filter
"""

from caem.config import CAEMConfig
from caem.memory.entry import (
    EpisodicEntry,
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
    "RoutingDecision",
    "QueryEncoder",
    "EpisodicMemoryStore",
]
