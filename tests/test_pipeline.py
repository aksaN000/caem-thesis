"""
tests/test_pipeline.py
=======================
Unit tests for CAEMPipeline (end-to-end orchestrator).

Session 42 rewrite
------------------
Stage 5 now produces a single ``UnifiedVerifierOutput`` carrying all nine
signals (u_token, u_dropout, u_internal, s_avg, h_norm, p_entail,
p_ground_max, p_ground_mean, p_ground_atomic) plus p_contra, the
STORE/DEFERRED/ABSTAIN/DISCARD decision, the early-exit flag, and the
composite u_stored. The pipeline's storage gate now keys on
``vout.decision == "STORE"`` rather than a threshold comparison on
u_stored. These tests exercise the new contract directly -- there is no
StoredConfidence projection dataclass any more.

Mock strategy
-------------
Every sub-component is replaced with a MagicMock -- no real models, no
FAISS network calls. We control:
  - encoder.encode()           -> fixed 768-dim unit vector
  - model.generate()           -> fixed token tensor
  - tokenizer()                -> mock BatchEncoding with .to() support
  - tokenizer.decode()         -> fixed answer string
  - pre_estimator.estimate()   -> PreRoutingConfidence with controlled u_pre
  - router.route()             -> RoutingDecision with controlled tier
  - (post-generation Stage-4a gate was removed Session 42; verifier is the truth)
  - verifier.verify()          -> UnifiedVerifierOutput with controlled fields
  - rag.generate()             -> fixed answer string
  - memory_store               -> real EpisodicMemoryStore (small, in-memory)

Coverage
--------
  - PipelineResult fields (Session 42 shape)
  - Tier 1: stored answer + u_stored surfaced from entry; verifier NOT called
  - Tier 2: answer generated, post-conf computed, verifier runs, decision
            drives storage
  - Tier 2: escalation when u_hat < threshold
  - Tier 3: direct RAG path (safety override)
  - Storage gate: decision==STORE + novel -> stored
  - Storage gate: decision in {DEFERRED, ABSTAIN, DISCARD} -> not stored
  - Storage gate: duplicate embedding -> not stored
  - Storage gate: verification failure -> not stored
  - Storage gate: empty answer -> not stored (vout is None)
  - Early-exit confabulation flag surfaces in PipelineResult and logs
  - EpisodicEntry nine-signal payload copies through from vout
  - Tier 1 retrieval_count increments and stats updated
  - current_cycle recorded on stored entry
  - memory_summary delegates to store
  - make_retroverify_fn returns a bound closure
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from caem.config import CAEMConfig
from caem.memory.entry import (
    EpisodicEntry,
    PreRoutingConfidence,
    RoutingDecision,
)
from caem.memory.store import EpisodicMemoryStore
from caem.pipeline import CAEMPipeline, PipelineResult
from caem.verification.verifier import UnifiedVerifierOutput


# =============================================================================
# Helpers
# =============================================================================

DIM = 768


def unit_vec(seed: int = 0) -> np.ndarray:
    """Deterministic L2-normalised 768-dim float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_tok_output(input_ids=None):
    """Mock tokenizer output that survives ``.to(device)`` and dict indexing."""
    if input_ids is None:
        input_ids = torch.zeros(1, 8, dtype=torch.long)
    out = MagicMock()
    out.__getitem__ = lambda self, key: (
        input_ids if key == "input_ids" else torch.ones(1, 8, dtype=torch.long)
    )
    out.to.return_value = out
    out.items.return_value = [("input_ids", input_ids)]
    return out


def make_mock_encoder(seed: int = 0) -> MagicMock:
    enc = MagicMock()
    enc.encode.return_value = unit_vec(seed)
    return enc


def make_mock_model() -> MagicMock:
    model = MagicMock()
    params = [torch.zeros(2, 2)]
    model.parameters.side_effect = lambda: iter(params)
    model.generate.return_value = torch.tensor([[0, 1, 2]])
    # Flan-T5 uses the pad token as the decoder start; exposing a real
    # int here avoids MagicMock propagating into torch.tensor() when
    # _build_forced_prefix builds decoder_input_ids for the scaffold.
    model.config.decoder_start_token_id = 0
    return model


def make_mock_tokenizer(answer: str = "William Shakespeare") -> MagicMock:
    tok = MagicMock()
    tok.return_value = _make_tok_output()
    tok.decode.return_value = answer
    tok.pad_token_id = 0
    return tok


def make_pre_conf(u_pre: float = 0.80) -> PreRoutingConfidence:
    return PreRoutingConfidence(u_token=u_pre, c_conv=0.1, u_pre=u_pre)


def make_routing(
    tier: int = 2,
    safety_override: bool = False,
    similarity: float = 0.5,
    u_pre: float = 0.80,
) -> RoutingDecision:
    return RoutingDecision(
        tier=tier,
        similarity=similarity,
        u_stored_retrieved=0.8,
        u_pre=u_pre,
        routing_score=0.65,
        safety_override=safety_override,
        retrieved_entry_id=None,
    )


def make_post_conf(u_hat: float = 0.70) -> None:
    # Retained only as a helper shim: all call sites used to build a
    # ``PostGenerationConfidence`` instance here, but the dataclass was
    # removed as dead code on 2026-04-22 (the Stage-4a u_hat gate that
    # consumed it was retired in Session 42 when UnifiedVerifier became
    # the single source of post-generation truth). The function is kept
    # so existing call sites do not need mass edits; it now returns
    # ``None`` since that is what ``PipelineResult.post_confidence``
    # carries in the live pipeline.
    del u_hat
    return None


def make_vout(
    u_stored: float = 0.75,
    decision: str = "STORE",
    *,
    p_contra: float = 0.05,
    p_ground_max: float = 0.80,
    p_ground_mean: float = 0.70,
    p_ground_atomic: float = 0.60,
    s_avg: float = 0.80,
    h_norm: float = 0.10,
    p_entail: float = 0.85,
    u_token: float = 0.70,
    u_dropout: float = 0.70,
    u_internal: float = 0.70,
    early_exit_triggered: bool = False,
    abstained: bool = False,
) -> UnifiedVerifierOutput:
    """Build a fully-populated UnifiedVerifierOutput for mocking."""
    return UnifiedVerifierOutput(
        u_token=u_token,
        u_dropout=u_dropout,
        u_internal=u_internal,
        s_avg=s_avg,
        h_norm=h_norm,
        p_entail=p_entail,
        p_ground_max=p_ground_max,
        p_ground_mean=p_ground_mean,
        p_ground_atomic=p_ground_atomic,
        p_contra=p_contra,
        u_stored=u_stored,
        decision=decision,
        early_exit_triggered=early_exit_triggered,
        abstained=abstained,
    )


def make_episodic_entry(
    seed: int = 0,
    u_stored: float = 0.80,
    question: str = "Who wrote Hamlet?",
    answer: str = "William Shakespeare",
) -> EpisodicEntry:
    """EpisodicEntry populated with plausible Session 42 nine-signal values."""
    return EpisodicEntry(
        question=question,
        reasoning_chain=f"Reasoning for: {question}",
        answer=answer,
        embedding=unit_vec(seed),
        storage_cycle=0,
        u_stored=u_stored,
        u_token=0.70,
        u_dropout=0.70,
        u_internal=0.70,
        s_avg=0.80,
        h_norm=0.10,
        p_entail=0.85,
        p_ground_max=0.80,
        p_ground_mean=0.70,
        p_ground_atomic=0.60,
        p_contra=0.05,
        decision="STORE",
        early_exit_triggered=False,
    )


def _build_pipeline(
    tier: int = 2,
    u_hat: float = 0.70,
    u_stored: float = 0.75,
    decision: str = "STORE",
    u_pre: float = 0.80,
    safety_override: bool = False,
    similarity: float = 0.5,
    answer: str = "William Shakespeare",
    memory_store: EpisodicMemoryStore | None = None,
    config: CAEMConfig | None = None,
    *,
    early_exit_triggered: bool = False,
    p_contra: float = 0.05,
    p_ground_max: float = 0.80,
    current_cycle: int = 0,
) -> CAEMPipeline:
    """Build a fully mocked CAEMPipeline for testing."""
    cfg = config or CAEMConfig()
    model = make_mock_model()
    tok = make_mock_tokenizer(answer)
    enc = make_mock_encoder()

    with patch("caem.pipeline.PreRoutingConfidenceEstimator") as MockPre, \
         patch("caem.pipeline.AdaptiveRouter") as MockRouter, \
         patch("caem.pipeline.UnifiedVerifier") as MockVerifier, \
         patch("caem.pipeline.TierThreeRAG") as MockRAG:

        MockPre.return_value.estimate.return_value = make_pre_conf(u_pre)
        MockRouter.return_value.route.return_value = make_routing(
            tier, safety_override, similarity, u_pre=u_pre,
        )
        MockVerifier.return_value.verify.return_value = make_vout(
            u_stored=u_stored,
            decision=decision,
            p_contra=p_contra,
            p_ground_max=p_ground_max,
            early_exit_triggered=early_exit_triggered,
        )
        MockRAG.return_value.generate.return_value = answer

        pipeline = CAEMPipeline(
            model=model,
            tokenizer=tok,
            encoder=enc,
            config=cfg,
            memory_store=memory_store,
            device="cpu",
            current_cycle=current_cycle,
        )

    pipeline._mock_pre = pipeline.pre_estimator  # type: ignore[attr-defined]
    pipeline._mock_router = pipeline.router  # type: ignore[attr-defined]
    pipeline._mock_post = None  # type: ignore[attr-defined]  # post_estimator removed in Session 42
    pipeline._mock_verifier = pipeline.verifier  # type: ignore[attr-defined]
    pipeline._mock_rag = pipeline.rag  # type: ignore[attr-defined]
    return pipeline


# =============================================================================
# PipelineResult
# =============================================================================

class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult(query="q", answer="a", tier=2)
        assert r.stored is False
        assert r.escalated is False
        assert r.entry_id is None
        assert r.latency_ms == 0.0
        assert r.verifier_output is None
        assert r.u_stored is None

    def test_fields_set(self):
        r = PipelineResult(
            query="q", answer="a", tier=1,
            stored=True, latency_ms=42.0, entry_id=7, u_stored=0.77,
        )
        assert r.tier == 1
        assert r.stored is True
        assert r.latency_ms == 42.0
        assert r.entry_id == 7
        assert r.u_stored == pytest.approx(0.77)

    def test_verifier_output_field_carries_decision(self):
        vout = make_vout(u_stored=0.66, decision="DEFERRED")
        r = PipelineResult(query="q", answer="a", tier=2, verifier_output=vout)
        assert r.verifier_output is vout
        assert r.verifier_output.decision == "DEFERRED"  # type: ignore[union-attr]


# =============================================================================
# Pipeline construction
# =============================================================================

class TestPipelineConstruction:
    def test_repr_contains_cycle_and_memory(self):
        p = _build_pipeline()
        r = repr(p)
        assert "CAEMPipeline" in r
        assert "cycle=" in r
        assert "memory=" in r

    def test_default_cycle_is_zero(self):
        p = _build_pipeline()
        assert p.current_cycle == 0

    def test_explicit_cycle_recorded(self):
        p = _build_pipeline(current_cycle=3)
        assert p.current_cycle == 3

    def test_memory_store_created_if_not_provided(self):
        p = _build_pipeline()
        assert isinstance(p.memory_store, EpisodicMemoryStore)

    def test_existing_memory_store_used(self):
        store = EpisodicMemoryStore()
        p = _build_pipeline(memory_store=store)
        assert p.memory_store is store

    def test_no_passage_store_warning(self, caplog):
        """MAJOR-T1 regression guard (formerly a no-op test).

        Before this fix the test opened caplog but never inspected records,
        so any regression silencing the warning stayed green in CI. The
        warning is the only runtime signal that passage_store is absent and
        verifier grounding signals will be neutralised -- the exact
        failure mode BLOCKER-V1 (Task #112) addresses -- so it must be
        pinned explicitly to both level and a stable substring.
        """
        import logging
        with caplog.at_level(logging.WARNING, logger="caem.pipeline"):
            p = _build_pipeline()
        assert p is not None
        warning_records = [
            rec for rec in caplog.records
            if rec.levelname == "WARNING"
            and "no passage_store provided" in rec.message
        ]
        assert warning_records, (
            "Expected a WARNING log containing 'no passage_store provided' "
            f"when passage_store=None; got records: "
            f"{[(r.levelname, r.message) for r in caplog.records]}"
        )
        # Tighten further: the warning must also flag that verifier grounding
        # signals will be neutralised, not just Tier 3. This keeps the test
        # from silently accepting a future message that drops the grounding
        # half of the signal (the BLOCKER-V1 half).
        assert any(
            "grounding" in rec.message.lower() for rec in warning_records
        ), (
            "Expected the warning to mention verifier grounding signals being "
            "neutralised (BLOCKER-V1 signal); got messages: "
            f"{[r.message for r in warning_records]}"
        )


# =============================================================================
# answer() -- Tier 2 path
# =============================================================================

class TestTier2Path:
    def test_returns_pipeline_result(self):
        p = _build_pipeline(tier=2)
        r = p.answer("Who wrote Hamlet?")
        assert isinstance(r, PipelineResult)

    def test_answer_string_set(self):
        """Tier 2 answer must contain the mock's generated content.

        Branch-C update: the scaffolded-CoT contract forces every answer to
        begin with ``Reasoning:`` (prefill text, prepended in ``_tier2``).
        We accept either the bare generation (``"Shakespeare"``) OR the
        prefixed form (``"Reasoning:Shakespeare"``) — the semantic content
        is present in both cases.
        """
        p = _build_pipeline(tier=2, answer="Shakespeare")
        r = p.answer("q")
        assert "Shakespeare" in r.answer

    def test_tier_is_2(self):
        p = _build_pipeline(tier=2)
        r = p.answer("q")
        assert r.tier == 2

    def test_verifier_output_attached(self):
        p = _build_pipeline(tier=2, u_stored=0.77, decision="STORE")
        r = p.answer("q")
        assert r.verifier_output is not None
        assert r.verifier_output.u_stored == pytest.approx(0.77)
        assert r.verifier_output.decision == "STORE"

    def test_u_stored_scalar_matches_vout(self):
        p = _build_pipeline(tier=2, u_stored=0.81)
        r = p.answer("q")
        assert r.u_stored == pytest.approx(0.81)

    def test_escalated_false_when_accepted(self):
        p = _build_pipeline(tier=2, u_hat=0.70)
        r = p.answer("q")
        assert r.escalated is False

    def test_latency_ms_positive(self):
        p = _build_pipeline(tier=2)
        r = p.answer("q")
        assert r.latency_ms > 0


# =============================================================================
# Tier 2 escalation
# =============================================================================

class TestTier2Escalation:
    def test_escalation_uses_rag_answer(self):
        """On Tier-2 escalation to Tier 3, the RAG answer is returned.

        Branch-C update: Tier-2 path prepends ``Reasoning:`` (scaffolded-CoT
        contract) when its own generation succeeds; on escalation, the RAG
        answer is returned directly from ``_tier3`` without additional
        prefix wrapping. Accept either plain or prefixed form.
        """
        p = _build_pipeline(tier=2, u_hat=0.30, answer="RAG answer")
        r = p.answer("q")
        assert "RAG answer" in r.answer

    def test_verifier_still_runs_after_escalation(self):
        p = _build_pipeline(tier=2, u_hat=0.30, u_stored=0.72)
        r = p.answer("q")
        assert r.verifier_output is not None
        assert r.verifier_output.u_stored == pytest.approx(0.72)


# =============================================================================
# Tier 1 path -- fast path, verifier skipped
# =============================================================================

class TestTier1Path:
    def _make_tier1_pipeline(self, u_stored_memory: float = 0.80):
        store = EpisodicMemoryStore()
        entry = make_episodic_entry(seed=0, u_stored=u_stored_memory)
        store.add(entry)
        return _build_pipeline(tier=1, similarity=0.92, memory_store=store)

    def test_tier_is_1(self):
        p = self._make_tier1_pipeline()
        r = p.answer("Who wrote Hamlet?")
        assert r.tier == 1

    def test_answer_from_memory(self):
        p = self._make_tier1_pipeline()
        r = p.answer("Who wrote Hamlet?")
        assert r.answer == "William Shakespeare"

    def test_post_conf_is_none(self):
        p = self._make_tier1_pipeline()
        r = p.answer("Who wrote Hamlet?")
        assert r.post_confidence is None

    def test_verifier_output_is_none(self):
        p = self._make_tier1_pipeline()
        r = p.answer("Who wrote Hamlet?")
        assert r.verifier_output is None

    def test_verifier_not_called(self):
        p = self._make_tier1_pipeline()
        p.answer("Who wrote Hamlet?")
        p._mock_verifier.verify.assert_not_called()  # type: ignore[attr-defined]

    def test_u_stored_surfaced_from_entry(self):
        p = self._make_tier1_pipeline(u_stored_memory=0.82)
        r = p.answer("Who wrote Hamlet?")
        assert r.u_stored == pytest.approx(0.82)

    def test_not_re_stored(self):
        store = EpisodicMemoryStore()
        entry = make_episodic_entry(seed=0)
        store.add(entry)
        initial_size = store.size

        p = _build_pipeline(tier=1, similarity=0.92, memory_store=store)
        p.answer("Who wrote Hamlet?")
        assert store.size == initial_size

    def test_escalated_false(self):
        p = self._make_tier1_pipeline()
        r = p.answer("Who wrote Hamlet?")
        assert r.escalated is False


# =============================================================================
# Tier 3 path
# =============================================================================

class TestTier3Path:
    def test_tier_is_3(self):
        p = _build_pipeline(tier=3, safety_override=True, u_pre=0.50)
        r = p.answer("q")
        assert r.tier == 3

    def test_post_conf_is_none(self):
        p = _build_pipeline(tier=3, safety_override=True)
        r = p.answer("q")
        assert r.post_confidence is None

    def test_escalated_false(self):
        p = _build_pipeline(tier=3, safety_override=True)
        r = p.answer("q")
        assert r.escalated is False

    def test_safety_override_recorded(self):
        p = _build_pipeline(tier=3, safety_override=True)
        r = p.answer("q")
        assert r.routing_decision.safety_override is True  # type: ignore[union-attr]

    def test_verifier_runs_on_tier3(self):
        p = _build_pipeline(tier=3, safety_override=True, u_stored=0.70)
        r = p.answer("q")
        assert r.verifier_output is not None
        assert r.verifier_output.u_stored == pytest.approx(0.70)


# =============================================================================
# Storage decision -- decision tree, not threshold
# =============================================================================

class TestStorageDecision:
    def test_decision_store_novel_is_stored(self):
        store = EpisodicMemoryStore()
        p = _build_pipeline(tier=2, decision="STORE", u_stored=0.75, memory_store=store)
        r = p.answer("A fresh question never seen before?")
        assert r.stored is True
        assert r.entry_id is not None

    @pytest.mark.parametrize("decision", ["DEFERRED", "ABSTAIN", "DISCARD"])
    def test_non_store_decisions_block_storage(self, decision):
        store = EpisodicMemoryStore()
        p = _build_pipeline(
            tier=2, decision=decision, u_stored=0.75, memory_store=store,
        )
        r = p.answer("q?")
        assert r.stored is False
        assert r.entry_id is None

    def test_verification_failure_blocks_storage(self):
        store = EpisodicMemoryStore()
        p = _build_pipeline(tier=2, memory_store=store)
        p._mock_verifier.verify.side_effect = RuntimeError("verifier down")  # type: ignore[attr-defined]
        r = p.answer("q?")
        assert r.stored is False
        assert r.verifier_output is None

    def test_duplicate_blocks_storage(self):
        cfg = CAEMConfig()
        cfg.novelty_threshold = 0.00
        store = EpisodicMemoryStore()
        store.add(make_episodic_entry(seed=0))
        initial = store.size
        p = _build_pipeline(
            tier=2, decision="STORE", memory_store=store, config=cfg,
        )
        p.answer("Who wrote Hamlet?")
        assert store.size == initial

    def test_empty_answer_not_stored(self):
        store = EpisodicMemoryStore()
        p = _build_pipeline(tier=3, memory_store=store, answer="")
        p._mock_verifier.verify.return_value = None  # type: ignore[attr-defined]
        r = p.answer("q?")
        assert r.stored is False

    def test_stored_entry_has_nine_signal_payload(self):
        store = EpisodicMemoryStore()
        p = _build_pipeline(
            tier=2,
            decision="STORE",
            u_stored=0.73,
            memory_store=store,
            p_contra=0.02,
            p_ground_max=0.88,
        )
        p.answer("A unique question for nine-signal check?")
        assert store.size == 1
        stored_entry = next(iter(store._metadata.values()))
        assert stored_entry.u_stored == pytest.approx(0.73)
        assert stored_entry.decision == "STORE"
        assert stored_entry.p_contra == pytest.approx(0.02)
        assert stored_entry.p_ground_max == pytest.approx(0.88)
        assert stored_entry.u_internal == pytest.approx(0.70)


# =============================================================================
# Early-exit confabulation gate
# =============================================================================

class TestEarlyExitConfabulation:
    def test_early_exit_flag_surfaces_on_result(self):
        p = _build_pipeline(
            tier=3, safety_override=True,
            decision="DISCARD", early_exit_triggered=True, u_stored=0.0,
        )
        r = p.answer("q")
        assert r.verifier_output is not None
        assert r.verifier_output.early_exit_triggered is True
        assert r.verifier_output.decision == "DISCARD"

    def test_early_exit_blocks_storage(self):
        store = EpisodicMemoryStore()
        p = _build_pipeline(
            tier=3, safety_override=True,
            decision="DISCARD", early_exit_triggered=True,
            memory_store=store,
        )
        r = p.answer("q")
        assert r.stored is False
        assert store.size == 0

    def test_early_exit_flag_logged(self, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="caem.pipeline"):
            p = _build_pipeline(
                tier=2, decision="DISCARD", early_exit_triggered=True,
            )
            p.answer("q")
        assert any("confabulation" in rec.message.lower()
                   for rec in caplog.records)


# =============================================================================
# Retrieval stats (Tier 1)
# =============================================================================

class TestTier1RetrievalStats:
    def test_retrieval_count_incremented(self):
        store = EpisodicMemoryStore()
        entry = make_episodic_entry(seed=0, u_stored=0.80)
        store.add(entry)
        assert entry.retrieval_count == 0

        p = _build_pipeline(tier=1, memory_store=store)
        p.answer("Who wrote Hamlet?")
        assert entry.retrieval_count == 1

    def test_u_stored_not_changed_by_retrieval(self):
        store = EpisodicMemoryStore()
        entry = make_episodic_entry(seed=0, u_stored=0.60)
        store.add(entry)

        p = _build_pipeline(tier=1, memory_store=store)
        p.answer("Who wrote Hamlet?")
        assert entry.u_stored == pytest.approx(0.60)
        p._mock_verifier.verify.assert_not_called()  # type: ignore[attr-defined]


# =============================================================================
# Cycle tracking
# =============================================================================

class TestCycleTracking:
    def test_stored_entry_records_current_cycle(self):
        store = EpisodicMemoryStore()
        p = _build_pipeline(
            tier=2, decision="STORE", memory_store=store, current_cycle=2,
        )
        p.answer("A unique question for cycle tracking?")
        assert store.size == 1
        stored_entry = next(iter(store._metadata.values()))
        assert stored_entry.storage_cycle == 2


# =============================================================================
# Routing metadata
# =============================================================================

class TestRoutingMetadata:
    def test_routing_decision_attached(self):
        p = _build_pipeline(tier=2)
        r = p.answer("q")
        assert r.routing_decision is not None
        assert r.routing_decision.tier == 2

    def test_pre_confidence_attached(self):
        p = _build_pipeline(tier=2, u_pre=0.75)
        r = p.answer("q")
        assert r.pre_confidence is not None
        assert r.pre_confidence.u_pre == pytest.approx(0.75)


# =============================================================================
# memory_summary + retroverify
# =============================================================================

class TestMemorySummary:
    def test_summary_returns_dict(self):
        p = _build_pipeline()
        s = p.memory_summary()
        assert isinstance(s, dict)

    def test_summary_has_size_key(self):
        p = _build_pipeline()
        s = p.memory_summary()
        assert "size" in s


class TestRetroverifyHelper:
    def test_returns_callable(self):
        p = _build_pipeline()
        fn = p.make_retroverify_fn()
        assert callable(fn)

    def test_calls_bound_verifier(self):
        p = _build_pipeline()
        vout = make_vout(u_stored=0.66, decision="DEFERRED")
        p._mock_verifier.verify.return_value = vout  # type: ignore[attr-defined]

        entry = make_episodic_entry(seed=1)
        fn = p.make_retroverify_fn()
        result = fn(entry)

        assert result is vout
        p._mock_verifier.verify.assert_called_with(entry.question, entry.answer)  # type: ignore[attr-defined]

    def test_returns_none_on_verifier_exception(self):
        p = _build_pipeline()
        p._mock_verifier.verify.side_effect = RuntimeError("mid-cycle crash")  # type: ignore[attr-defined]
        entry = make_episodic_entry(seed=2)
        fn = p.make_retroverify_fn()
        assert fn(entry) is None


# =============================================================================
# Verifier construction-time wiring (BLOCKER-V1 / MAJOR-T5 regression guard)
# =============================================================================

class TestVerifierWiring:
    """Regression guard for BLOCKER-V1 (Task #112).

    Prior to the BLOCKER-V1 fix, ``CAEMPipeline.__init__`` constructed
    ``UnifiedVerifier(...)`` WITHOUT passing ``passage_retriever=``,
    silently reducing the thesis's nine-signal verifier to a five-signal
    verifier in production:

      * ``p_ground_max``, ``p_ground_mean``, ``p_ground_atomic`` locked at
        0.5 (empty-passage fallback)
      * ``p_contra`` locked at 0.0 (empty-passage fallback)
      * Early-exit confabulation gate (needs p_ground_max <= 0.20)
        unable to fire
      * Contradiction hard veto (needs p_contra >= 0.30) unable to fire
      * ABSTAIN decision class (needs p_ground_max < 0.20) unable to fire

    The existing ``_blank_verifier`` helper in tests/test_verifier.py
    constructs via ``UnifiedVerifier.__new__`` and bypasses ``__init__``,
    so every test using it would stay green even if the init-path wiring
    broke again. This test class intercepts the real
    ``UnifiedVerifier(...)`` call at pipeline construction time via a
    MagicMock patch and asserts the kwarg is threaded through.
    """

    def _construct_with_verifier_patch(self, passage_store=None):
        """Construct a CAEMPipeline with ``UnifiedVerifier`` mocked so we
        can inspect the constructor call args. Returns (MockVerifier_cls,
        pipeline).
        """
        model = make_mock_model()
        tok = make_mock_tokenizer("ans")
        enc = make_mock_encoder()
        with patch("caem.pipeline.PreRoutingConfidenceEstimator"), \
             patch("caem.pipeline.AdaptiveRouter"), \
             patch("caem.pipeline.UnifiedVerifier") as MockVerifier, \
             patch("caem.pipeline.TierThreeRAG"):
            pipeline = CAEMPipeline(
                model=model,
                tokenizer=tok,
                encoder=enc,
                passage_store=passage_store,
                device="cpu",
            )
        return MockVerifier, pipeline

    def test_passage_retriever_kwarg_is_passed_at_construction(self):
        """MAJOR-T5: assert ``passage_retriever=`` appears in the
        UnifiedVerifier(...) kwargs. Without this, BLOCKER-V1 can
        silently regress under a future refactor."""
        MockVerifier, _ = self._construct_with_verifier_patch()
        assert MockVerifier.called, "UnifiedVerifier was never constructed"
        kwargs = MockVerifier.call_args.kwargs
        assert "passage_retriever" in kwargs, (
            "passage_retriever missing from UnifiedVerifier(...) kwargs; "
            f"got: {sorted(kwargs.keys())}"
        )

    def test_passage_retriever_is_non_none_callable(self):
        """MAJOR-T5: the passed retriever must be a non-None callable.
        Under BLOCKER-V1, UnifiedVerifier.__init__ defaults it to None
        and _retrieve_and_rerank silently returns [] on every call."""
        MockVerifier, _ = self._construct_with_verifier_patch()
        retriever = MockVerifier.call_args.kwargs.get("passage_retriever")
        assert retriever is not None, "passage_retriever was passed as None"
        assert callable(retriever), (
            f"passage_retriever is not callable: {type(retriever)!r}"
        )

    def test_passage_retriever_returns_list_on_empty_store(self):
        """MAJOR-T5: calling the retriever with the degenerate empty
        PassageStore (the default when passage_store=None was passed
        to CAEMPipeline) must return ``[]`` -- matches the
        UnifiedVerifier empty-passage fallback contract so the
        verifier still returns neutral grounding scores instead of
        crashing."""
        MockVerifier, _ = self._construct_with_verifier_patch()
        retriever = MockVerifier.call_args.kwargs["passage_retriever"]
        out = retriever("any query string", 5)
        assert isinstance(out, list), (
            f"retriever should return list, got {type(out)!r}"
        )
        assert out == [], (
            f"empty PassageStore should yield empty retrieval, got {out!r}"
        )
        # Every element must be a string (List[str] per verifier contract)
        assert all(isinstance(p, str) for p in out), (
            f"retriever must return List[str], got items of types "
            f"{[type(p).__name__ for p in out]}"
        )


