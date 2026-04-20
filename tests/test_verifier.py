"""
tests/test_verifier.py
======================
Phase 6a -- Unit tests for the Session 42 UnifiedVerifier nine-signal gate.

These tests replace the legacy two-layer verifier coverage (pre-Session 42).
They focus on the pure-logic surface that can be exercised without loading
Flan-T5-Large, all-mpnet-base-v2, or a real NLI ensemble:

  * UnifiedVerifierOutput field contract.
  * _NLIEnsemble aggregation (min entail, max contra, majority argmax).
  * _parse_atomic_facts numbering tolerance + dedup.
  * _composite weight-sum invariant and monotonicity.
  * _decide decision-tree corners (STORE / DEFERRED / ABSTAIN / DISCARD).
  * _semantic_entropy_surface known-probabilities entropy.
  * _score_s_avg / _score_p_ground / _score_p_contra fallback paths.
  * verify() end-to-end with a fully mocked dependency graph, including
    the early-exit confabulation gate.

Heavy-model behaviour (actual forward passes, sampling, atomic decomposition
output quality) is out-of-scope -- those are GPU-only smoke / integration
tests and live in tests/test_pipeline_integration.py (Phase 7).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from caem.config import CAEMConfig
from caem.verification.verifier import (
    UnifiedVerifier,
    UnifiedVerifierOutput,
    _NLIEnsemble,
    _parse_atomic_facts,
)


# =============================================================================
# Fake bundles / helpers
# =============================================================================

class _FakeNLIModel(torch.nn.Module):
    """Emits a fixed softmax distribution on every forward pass."""

    def __init__(self, probs: Sequence[float]) -> None:
        super().__init__()
        assert len(probs) == 3, "probs must be (CONTRA, NEUTRAL, ENTAIL)"
        # softmax(log p) == p when probs sum to 1, so no scaling is needed.
        logits = torch.log(torch.tensor(list(probs), dtype=torch.float32))
        self.logits_param = torch.nn.Parameter(logits.unsqueeze(0), requires_grad=False)

    def forward(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(logits=self.logits_param.clone())


class _FakeTokenizer:
    """Stand-in for a HuggingFace tokenizer (__call__ / decode)."""

    pad_token_id = 0

    def __call__(self, *args, **kwargs) -> Any:
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        enc = {"input_ids": ids}
        enc_with_to = dict(enc)
        enc_with_to["to"] = lambda *a, **k: enc  # type: ignore[assignment]

        class _Enc(dict):
            def to(self, *a, **k):
                return self

        return _Enc(enc)

    def decode(self, *args, **kwargs) -> str:
        return "decoded answer"


def _make_nli_bundle(p_contra: float, p_neutral: float, p_entail: float) -> Tuple[Any, Any]:
    return _FakeNLIModel([p_contra, p_neutral, p_entail]), _FakeTokenizer()


# =============================================================================
# UnifiedVerifierOutput dataclass
# =============================================================================

class TestUnifiedVerifierOutput:
    def test_required_fields_present(self):
        out = UnifiedVerifierOutput(
            u_token=0.80, u_dropout=0.10, u_internal=0.85,
            s_avg=0.70, h_norm=0.20, p_entail=0.80,
            p_ground_max=0.60, p_ground_mean=0.55, p_ground_atomic=0.50,
            p_contra=0.05,
            u_stored=0.65, decision="STORE",
            early_exit_triggered=False, abstained=False,
        )
        assert out.decision == "STORE"
        assert out.top_passages == [] and out.atomic_facts == []
        assert out.per_atom_entail == []
        assert not out.early_exit_triggered and not out.abstained

    def test_metadata_fields_override_defaults(self):
        out = UnifiedVerifierOutput(
            u_token=0.5, u_dropout=0.5, u_internal=0.5,
            s_avg=0.5, h_norm=0.5, p_entail=0.5,
            p_ground_max=0.5, p_ground_mean=0.5, p_ground_atomic=0.5,
            p_contra=0.0, u_stored=0.5, decision="DEFERRED",
            early_exit_triggered=False, abstained=False,
            top_passages=["p1", "p2"], atomic_facts=["f1"], per_atom_entail=[0.7],
        )
        assert out.top_passages == ["p1", "p2"]
        assert out.atomic_facts == ["f1"]
        assert out.per_atom_entail == [0.7]


# =============================================================================
# _NLIEnsemble aggregation
# =============================================================================

class TestNLIEnsemble:
    def test_empty_bundles_returns_fallbacks(self):
        e = _NLIEnsemble([], device="cpu")
        assert not bool(e)
        assert e.entail_prob("a", "b") == 0.5
        assert e.contradict_prob("a", "b") == 0.0
        assert e.argmax_label("a", "b") == 1   # NEUTRAL

    def test_entail_prob_is_ensemble_min(self):
        b1 = _make_nli_bundle(0.05, 0.05, 0.90)
        b2 = _make_nli_bundle(0.05, 0.35, 0.60)
        e = _NLIEnsemble([b1, b2], device="cpu")
        assert e.entail_prob("p", "h") == pytest.approx(0.60, abs=0.02)

    def test_contradict_prob_is_ensemble_max(self):
        b1 = _make_nli_bundle(0.10, 0.30, 0.60)
        b2 = _make_nli_bundle(0.80, 0.10, 0.10)
        e = _NLIEnsemble([b1, b2], device="cpu")
        assert e.contradict_prob("p", "h") == pytest.approx(0.80, abs=0.02)

    def test_argmax_majority_vote(self):
        b_ent_1 = _make_nli_bundle(0.05, 0.20, 0.75)
        b_ent_2 = _make_nli_bundle(0.05, 0.15, 0.80)
        b_neu   = _make_nli_bundle(0.20, 0.60, 0.20)
        e = _NLIEnsemble([b_ent_1, b_ent_2, b_neu], device="cpu")
        assert e.argmax_label("p", "h") == 2

    def test_argmax_tie_defaults_to_neutral(self):
        b_ent = _make_nli_bundle(0.10, 0.10, 0.80)
        b_con = _make_nli_bundle(0.80, 0.10, 0.10)
        e = _NLIEnsemble([b_ent, b_con], device="cpu")
        assert e.argmax_label("p", "h") == 1


# =============================================================================
# _parse_atomic_facts
# =============================================================================

class TestParseAtomicFacts:
    def test_standard_numbering_period(self):
        out = _parse_atomic_facts("1. Fact A.\n2. Fact B.\n3. Fact C.")
        assert out == ["Fact A.", "Fact B.", "Fact C."]

    def test_paren_and_colon_and_dash_numbering(self):
        out = _parse_atomic_facts("1) Alpha.\n2: Beta.\n3- Gamma.")
        assert out == ["Alpha.", "Beta.", "Gamma."]

    def test_unnumbered_lines_are_kept(self):
        out = _parse_atomic_facts("Paris is a capital.\nFrance is a country.")
        assert out == ["Paris is a capital.", "France is a country."]

    def test_deduplicates_case_insensitively(self):
        out = _parse_atomic_facts("1. Paris is a capital.\n2. paris is a CAPITAL.")
        assert out == ["Paris is a capital."]

    def test_trailing_colon_headers_dropped(self):
        out = _parse_atomic_facts("Numbered list of facts:\n1. Fact A.")
        assert out == ["Fact A."]

    def test_empty_input_returns_empty(self):
        assert _parse_atomic_facts("") == []
        assert _parse_atomic_facts("\n\n\n") == []


# =============================================================================
# _composite (weight-sum / monotonicity) and _decide (decision tree)
# =============================================================================

def _blank_verifier(cfg: CAEMConfig | None = None) -> UnifiedVerifier:
    """Construct a verifier skipping the __init__ forward-device probe."""
    cfg = cfg or CAEMConfig()
    v = UnifiedVerifier.__new__(UnifiedVerifier)
    v.model = MagicMock()
    v.tokenizer = _FakeTokenizer()
    v.sbert_encoder = MagicMock()
    v.config = cfg
    v.reranker = None
    v.passage_retriever = None
    v.enable_atomic = True
    v.device = "cpu"
    v.nli = _NLIEnsemble([], device="cpu")
    v.nli_model = None
    v.nli_tokenizer = None
    return v


class TestComposite:
    def test_weights_sum_to_one_by_default(self):
        cfg = CAEMConfig()
        w_sum = (
            cfg.u_stored_weight_pground_mean
            + cfg.u_stored_weight_pground_atomic
            + cfg.u_stored_weight_sc
            + cfg.u_stored_weight_se
            + cfg.u_stored_weight_uinternal
            + cfg.u_stored_weight_nli
        )
        assert w_sum == pytest.approx(1.0, abs=1e-6)

    def test_uniform_inputs_pass_through(self):
        v = _blank_verifier()
        u = v._composite(p_ground_mean=1.0, p_ground_atomic=1.0,
                         s_avg=1.0, h_norm=0.0,
                         u_internal=1.0, p_entail=1.0)
        assert u == pytest.approx(1.0)

    def test_zero_inputs_land_at_zero(self):
        v = _blank_verifier()
        u = v._composite(p_ground_mean=0.0, p_ground_atomic=0.0,
                         s_avg=0.0, h_norm=1.0,
                         u_internal=0.0, p_entail=0.0)
        assert u == pytest.approx(0.0, abs=1e-6)

    def test_monotonic_in_each_signal(self):
        v = _blank_verifier()
        base = dict(p_ground_mean=0.5, p_ground_atomic=0.5,
                    s_avg=0.5, h_norm=0.5,
                    u_internal=0.5, p_entail=0.5)
        u_base = v._composite(**base)
        for k in ("p_ground_mean", "p_ground_atomic", "s_avg",
                  "u_internal", "p_entail"):
            higher = dict(base); higher[k] = 0.9
            assert v._composite(**higher) > u_base, (
                f"increasing {k} must raise u_stored"
            )
        higher_h = dict(base); higher_h["h_norm"] = 0.9
        assert v._composite(**higher_h) < u_base


class TestDecide:
    @pytest.fixture
    def cfg(self) -> CAEMConfig:
        c = CAEMConfig()
        c.store_threshold = 0.65
        c.defer_threshold = 0.45
        c.abstain_pground_ceiling = 0.20
        c.contradiction_veto_threshold = 0.30
        return c

    def test_contradiction_veto_overrides_everything(self, cfg):
        v = _blank_verifier(cfg)
        d, _ = v._decide(u_stored=0.95, p_contra=0.80, p_ground_max=0.90)
        assert d == "DISCARD"

    def test_store_band(self, cfg):
        v = _blank_verifier(cfg)
        d, abstained = v._decide(u_stored=0.80, p_contra=0.05, p_ground_max=0.50)
        assert d == "STORE" and abstained is False

    def test_deferred_band(self, cfg):
        v = _blank_verifier(cfg)
        d, abstained = v._decide(u_stored=0.50, p_contra=0.05, p_ground_max=0.50)
        assert d == "DEFERRED" and abstained is False

    def test_abstain_corner(self, cfg):
        v = _blank_verifier(cfg)
        d, abstained = v._decide(u_stored=0.10, p_contra=0.05, p_ground_max=0.10)
        assert d == "ABSTAIN" and abstained is True

    def test_discard_fallthrough(self, cfg):
        v = _blank_verifier(cfg)
        d, abstained = v._decide(u_stored=0.30, p_contra=0.05, p_ground_max=0.50)
        assert d == "DISCARD" and abstained is False

    def test_store_boundary_inclusive(self, cfg):
        v = _blank_verifier(cfg)
        d, _ = v._decide(u_stored=cfg.store_threshold,
                         p_contra=0.05, p_ground_max=0.60)
        assert d == "STORE"

    def test_defer_boundary_inclusive(self, cfg):
        v = _blank_verifier(cfg)
        d, _ = v._decide(u_stored=cfg.defer_threshold,
                         p_contra=0.05, p_ground_max=0.60)
        assert d == "DEFERRED"


# =============================================================================
# _semantic_entropy_surface
# =============================================================================

class TestSemanticEntropySurface:
    def test_identical_samples_zero_entropy(self):
        v = _blank_verifier()
        H = v._semantic_entropy_surface(["Paris.", "Paris.", "Paris."])
        assert H == pytest.approx(0.0)

    def test_all_distinct_samples_uniform_entropy(self):
        v = _blank_verifier()
        H = v._semantic_entropy_surface(["Paris.", "London.", "Berlin.", "Rome."])
        assert H == pytest.approx(2.0, abs=1e-6)

    def test_normalisation_strips_punctuation_and_case(self):
        v = _blank_verifier()
        H = v._semantic_entropy_surface(["Paris.", "paris", "PARIS!"])
        assert H == pytest.approx(0.0)


# =============================================================================
# _score_s_avg / _score_p_ground / _score_p_contra fallback paths
# =============================================================================

class TestSignalFallbacks:
    def test_s_avg_too_few_chains(self):
        v = _blank_verifier()
        assert v._score_s_avg([]) == 0.5
        assert v._score_s_avg(["only one"]) == 0.5

    def test_s_avg_two_identical_chains(self):
        v = _blank_verifier()
        fake_vec = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        v.sbert_encoder = MagicMock()
        v.sbert_encoder.encode.return_value = fake_vec
        s = v._score_s_avg(["a", "b"])
        assert s == pytest.approx(1.0, abs=1e-5)

    def test_p_ground_neutral_with_no_passages(self):
        v = _blank_verifier()
        assert v._score_p_ground([], "answer") == (0.5, 0.5)

    def test_p_contra_zero_with_no_passages(self):
        v = _blank_verifier()
        assert v._score_p_contra([], "answer") == 0.0

    def test_p_ground_neutral_with_empty_nli(self):
        v = _blank_verifier()
        assert v._score_p_ground(["passage"], "answer") == (0.5, 0.5)

    def test_atomic_disabled_returns_fallback(self):
        v = _blank_verifier()
        v.enable_atomic = False
        facts, per_atom, p_atomic = v._score_atomic(
            ["passage"], "answer", fallback=0.42,
        )
        assert facts == [] and per_atom == [] and p_atomic == 0.42


# =============================================================================
# verify() end-to-end with fully mocked dependencies
# =============================================================================

def _stub_verifier_signals(v, *, u_token, u_dropout,
                           s_avg, h_norm, p_entail,
                           p_ground_max, p_ground_mean,
                           p_ground_atomic, p_contra,
                           passages=None, facts=None, per_atom=None):
    """Monkey-patch each signal routine to return a fixed value."""
    v._compute_u_token   = MagicMock(return_value=u_token)
    v._compute_u_dropout = MagicMock(return_value=u_dropout)
    v._generate_m_chains = MagicMock(return_value=["chain a", "chain b", "chain c"])
    v._score_s_avg       = MagicMock(return_value=s_avg)
    v._score_p_entail    = MagicMock(return_value=p_entail)
    v._compute_h_norm    = MagicMock(return_value=h_norm)
    v._retrieve_and_rerank = MagicMock(return_value=passages or ["p1", "p2", "p3"])
    v._score_p_ground    = MagicMock(return_value=(p_ground_max, p_ground_mean))
    v._score_p_contra    = MagicMock(return_value=p_contra)
    v._score_atomic      = MagicMock(return_value=(
        facts or ["Fact A."], per_atom or [p_ground_atomic], p_ground_atomic,
    ))
    v._tokenize = MagicMock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})


class TestVerifyEndToEnd:
    def test_happy_path_store(self):
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.80, u_dropout=0.10,
            s_avg=0.85, h_norm=0.10, p_entail=0.80,
            p_ground_max=0.85, p_ground_mean=0.80, p_ground_atomic=0.75,
            p_contra=0.05,
        )
        out = v.verify("q", "a", input_ids=torch.tensor([[1, 2]]))
        assert isinstance(out, UnifiedVerifierOutput)
        assert out.decision == "STORE"
        assert not out.early_exit_triggered
        assert out.u_stored > v.config.store_threshold
        assert out.u_token == pytest.approx(0.80)
        assert out.p_ground_max == pytest.approx(0.85)
        assert out.p_ground_atomic == pytest.approx(0.75)

    def test_early_exit_confabulation_gate(self):
        """u_internal >= 0.70 AND p_ground_max <= 0.20 must force DISCARD."""
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.90, u_dropout=0.05,
            s_avg=0.70, h_norm=0.20, p_entail=0.80,
            p_ground_max=0.10, p_ground_mean=0.05,
            p_ground_atomic=0.05, p_contra=0.05,
        )
        out = v.verify("q", "a", input_ids=torch.tensor([[1, 2]]))
        assert out.decision == "DISCARD"
        assert out.early_exit_triggered is True
        assert out.u_stored == pytest.approx(0.0)

    def test_contradiction_veto_discards(self):
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.80, u_dropout=0.10,
            s_avg=0.80, h_norm=0.15, p_entail=0.80,
            p_ground_max=0.85, p_ground_mean=0.80, p_ground_atomic=0.75,
            p_contra=0.90,
        )
        out = v.verify("q", "a", input_ids=torch.tensor([[1, 2]]))
        assert out.decision == "DISCARD"
        assert out.early_exit_triggered is False

    def test_deferred_or_discard_mid_band(self):
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.60, u_dropout=0.30,
            s_avg=0.55, h_norm=0.40, p_entail=0.55,
            p_ground_max=0.50, p_ground_mean=0.45, p_ground_atomic=0.40,
            p_contra=0.10,
        )
        out = v.verify("q", "a", input_ids=torch.tensor([[1, 2]]))
        assert out.decision in {"DEFERRED", "DISCARD"}
        assert out.early_exit_triggered is False

    def test_abstain_corner(self):
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.20, u_dropout=0.80,
            s_avg=0.20, h_norm=0.80, p_entail=0.10,
            p_ground_max=0.10, p_ground_mean=0.10, p_ground_atomic=0.10,
            p_contra=0.05,
        )
        out = v.verify("q", "a", input_ids=torch.tensor([[1, 2]]))
        assert out.decision == "ABSTAIN"
        assert out.abstained is True

    def test_should_store_mirrors_decision(self):
        v = _blank_verifier()
        store_out = UnifiedVerifierOutput(
            u_token=0.8, u_dropout=0.1, u_internal=0.85,
            s_avg=0.8, h_norm=0.1, p_entail=0.8,
            p_ground_max=0.8, p_ground_mean=0.8, p_ground_atomic=0.75,
            p_contra=0.05, u_stored=0.8, decision="STORE",
            early_exit_triggered=False, abstained=False,
        )
        discard_out = UnifiedVerifierOutput(
            u_token=0.8, u_dropout=0.1, u_internal=0.85,
            s_avg=0.8, h_norm=0.1, p_entail=0.8,
            p_ground_max=0.8, p_ground_mean=0.8, p_ground_atomic=0.75,
            p_contra=0.9, u_stored=0.0, decision="DISCARD",
            early_exit_triggered=False, abstained=False,
        )
        assert v.should_store(store_out) is True
        assert v.should_store(discard_out) is False

    def test_precomputed_internal_signals_are_respected(self):
        """Passing u_token / u_dropout to verify() must skip recomputation."""
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.99, u_dropout=0.01,
            s_avg=0.80, h_norm=0.10, p_entail=0.80,
            p_ground_max=0.85, p_ground_mean=0.80,
            p_ground_atomic=0.75, p_contra=0.05,
        )
        out = v.verify(
            "q", "a",
            input_ids=torch.tensor([[1, 2]]),
            u_token=0.30,
            u_dropout=0.70,
        )
        assert out.u_token == pytest.approx(0.30)
        assert out.u_dropout == pytest.approx(0.70)
        v._compute_u_token.assert_not_called()  # type: ignore[attr-defined]
        v._compute_u_dropout.assert_not_called()  # type: ignore[attr-defined]


class TestVerifyBatch:
    """Level B: ``verify_batch`` pools M-chain + semantic-entropy T5
    sampling across samples. These tests stub out the pooled sampler
    and per-sample scorers so the batch path can be validated without
    a real T5."""

    def test_verify_batch_empty_returns_empty(self):
        v = _blank_verifier()
        assert v.verify_batch([]) == []

    def test_verify_batch_single_sample_matches_verify(self):
        """One sample through verify_batch must produce the same decision
        and composite as a direct verify() call, assuming the same stubbed
        signals."""
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.80, u_dropout=0.10,
            s_avg=0.85, h_norm=0.10, p_entail=0.80,
            p_ground_max=0.85, p_ground_mean=0.80, p_ground_atomic=0.75,
            p_contra=0.05,
        )
        v._pooled_sample_t5 = MagicMock(
            return_value=[["chain a", "chain b", "chain c"]],
        )

        outs = v.verify_batch([("q0", "a0")])
        assert len(outs) == 1
        assert outs[0].decision == "STORE"
        assert outs[0].u_token == pytest.approx(0.80)

    def test_verify_batch_preserves_order_and_count(self):
        """Multiple samples: each gets its own per-sample scorers called
        and the outputs come back in input order."""
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.80, u_dropout=0.10,
            s_avg=0.85, h_norm=0.10, p_entail=0.80,
            p_ground_max=0.85, p_ground_mean=0.80, p_ground_atomic=0.75,
            p_contra=0.05,
        )
        v._pooled_sample_t5 = MagicMock(
            return_value=[["c0", "c1", "c2"], ["c0", "c1", "c2"], ["c0", "c1", "c2"]],
        )

        inputs = [("q0", "a0"), ("q1", "a1"), ("q2", "a2")]
        outs = v.verify_batch(inputs)

        assert len(outs) == 3
        # All three produce STORE decisions because stubbed signals are STORE-y.
        assert all(o.decision == "STORE" for o in outs)

    def test_verify_batch_pooled_sample_called_twice(self):
        """Pooled sampler must fire exactly twice per batch: once for
        M-chain (T=0.7) and once for semantic-entropy (T=se_temperature)."""
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.80, u_dropout=0.10,
            s_avg=0.85, h_norm=0.10, p_entail=0.80,
            p_ground_max=0.85, p_ground_mean=0.80, p_ground_atomic=0.75,
            p_contra=0.05,
        )
        v._pooled_sample_t5 = MagicMock(
            side_effect=[
                [["mc0", "mc1", "mc2"], ["mc0", "mc1", "mc2"]],
                [["se0", "se1", "se2"], ["se0", "se1", "se2"]],
            ],
        )
        v.verify_batch([("q0", "a0"), ("q1", "a1")])
        assert v._pooled_sample_t5.call_count == 2
        temps = [kw["temperature"] for _, kw in v._pooled_sample_t5.call_args_list]
        assert 0.7 in temps
        assert v.config.se_temperature in temps

    def test_verify_batch_forwards_u_token_u_dropout(self):
        """Precomputed internal signals should propagate per-sample."""
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.99, u_dropout=0.01,
            s_avg=0.85, h_norm=0.10, p_entail=0.80,
            p_ground_max=0.85, p_ground_mean=0.80, p_ground_atomic=0.75,
            p_contra=0.05,
        )
        v._pooled_sample_t5 = MagicMock(
            return_value=[["c"], ["c"]],
        )
        outs = v.verify_batch(
            [("q0", "a0"), ("q1", "a1")],
            u_tokens=[0.10, 0.20],
            u_dropouts=[0.30, 0.40],
        )
        assert outs[0].u_token == pytest.approx(0.10)
        assert outs[1].u_token == pytest.approx(0.20)
        assert outs[0].u_dropout == pytest.approx(0.30)
        assert outs[1].u_dropout == pytest.approx(0.40)


class TestPooledSampleT5Dual:
    """Level B Phase 2: CUDA-stream dual dispatch for the two
    data-independent T5 pooled samplings. On CPU (the test env)
    _pooled_sample_t5_dual falls back to serial ``_pooled_sample_t5``
    calls; on CUDA the implementation runs them on separate streams."""

    def test_cpu_fallback_to_serial_dispatch(self):
        v = _blank_verifier()
        calls: list = []

        def _record(queries, *, num_per, temperature, max_new_tokens, label):
            calls.append((label, num_per, temperature))
            return [[f"{label}_{i}_{k}" for k in range(num_per)]
                    for i in range(len(queries))]

        v._pooled_sample_t5 = MagicMock(side_effect=_record)
        a, b = v._pooled_sample_t5_dual(
            queries=["q0", "q1"],
            num_per_a=3, temperature_a=0.7, label_a="m-chain",
            num_per_b=2, temperature_b=0.5, label_b="se-sample",
            max_new_tokens=16,
        )
        labels = [c[0] for c in calls]
        assert labels == ["m-chain", "se-sample"]
        assert len(a) == 2 and all(len(x) == 3 for x in a)
        assert len(b) == 2 and all(len(x) == 2 for x in b)

    def test_empty_queries_returns_two_empty_lists(self):
        v = _blank_verifier()
        v._pooled_sample_t5 = MagicMock()
        a, b = v._pooled_sample_t5_dual(
            queries=[],
            num_per_a=3, temperature_a=0.7, label_a="m-chain",
            num_per_b=2, temperature_b=0.5, label_b="se-sample",
            max_new_tokens=16,
        )
        assert a == [] and b == []
        v._pooled_sample_t5.assert_not_called()

    def test_decode_grouped_error_returns_empty_lists(self):
        v = _blank_verifier()
        out = v._decode_grouped(
            None, N=3, num_per=2, label="m-chain",
            err=RuntimeError("synthetic CUDA error"),
        )
        assert out == [[], [], []]

    def test_decode_grouped_none_output_returns_empty_lists(self):
        v = _blank_verifier()
        out = v._decode_grouped(
            None, N=2, num_per=0, label="noop",
            err=None,
        )
        assert out == [[], []]

    def test_decode_grouped_regroups_by_input(self):
        """Given a (N*num_per, T) tensor, decode must put rows 0..M-1
        into sample 0, rows M..2M-1 into sample 1, etc."""
        v = _blank_verifier()

        class _Tok:
            def decode(self, row, **kwargs):
                return f"r{int(row[0].item())}"
        v.tokenizer = _Tok()

        N, num_per = 3, 2
        out_tensor = torch.arange(N * num_per).unsqueeze(1)
        result = v._decode_grouped(
            out_tensor, N=N, num_per=num_per, label="test", err=None,
        )
        assert result == [
            ["r0", "r1"],
            ["r2", "r3"],
            ["r4", "r5"],
        ]

    def test_verify_batch_uses_dual_pooled_sampler(self):
        """verify_batch routes through _pooled_sample_t5_dual once
        (not two separate _pooled_sample_t5 calls) so the CUDA-stream
        overlap path is actually taken when CUDA is available."""
        v = _blank_verifier()
        _stub_verifier_signals(
            v,
            u_token=0.80, u_dropout=0.10,
            s_avg=0.85, h_norm=0.10, p_entail=0.80,
            p_ground_max=0.85, p_ground_mean=0.80, p_ground_atomic=0.75,
            p_contra=0.05,
        )
        dual_called = {"n": 0}

        def _fake_dual(**kwargs):
            dual_called["n"] += 1
            N = len(kwargs["queries"])
            a = [[f"mc_{i}"] for i in range(N)]
            b = [[f"se_{i}"] for i in range(N)]
            return a, b

        v._pooled_sample_t5_dual = _fake_dual
        v._pooled_sample_t5 = MagicMock(
            side_effect=AssertionError(
                "single-sampler must not be called from verify_batch",
            ),
        )
        outs = v.verify_batch([("q0", "a0"), ("q1", "a1")])
        assert len(outs) == 2
        assert dual_called["n"] == 1
        v._pooled_sample_t5.assert_not_called()
