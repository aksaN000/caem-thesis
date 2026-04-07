"""
tests/test_verifier.py
======================
Unit tests for MultiLayerVerifier (Stage 5).

Mock strategy: identical to test_post_generation.py -- no real models,
controlled logits and embeddings, each signal isolated independently.

Coverage:
  - StoredConfidence dataclass
  - p_entail: NLI probability averaging over M chains
  - p_entail: fallback (0.5) when NLI model is None
  - s_avg: pairwise SBERT cosine similarity of M chains
  - h_norm: normalised semantic entropy (NLI clustering + surface fallback)
  - Combined û_stored formula and config weights
  - should_store() gate
  - Error handling (neutral fallback per signal)
  - u_stored weight sum = 1.0
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from caem.config import CAEMConfig
from caem.memory.entry import StoredConfidence
from caem.verification.verifier import MultiLayerVerifier


# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------

VOCAB   = 100
DIM     = 768
SEQ_LEN = 8
BATCH   = 1
EOS_ID  = 1
PAD_ID  = 0
CHOSEN  = 2


# -----------------------------------------------------------------------------
# Mock builders (reuse the patterns established in test_post_generation.py)
# -----------------------------------------------------------------------------

def make_plain_seq(n_tokens: int = 3) -> torch.Tensor:
    """Plain (1, n+2) tensor for generate() without return_dict_in_generate."""
    return torch.tensor([[PAD_ID] + [CHOSEN] * n_tokens + [EOS_ID]])


def make_mock_model(chosen_logit: float = 8.0) -> MagicMock:
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    # Stage 5 generate() calls never use return_dict_in_generate -> plain tensor
    model.generate.return_value = make_plain_seq()
    model.training = False
    return model


class _DictWithTo(dict):
    """dict subclass with .to(device) for HuggingFace-style tokenizer output."""
    def to(self, device):
        return self


def make_mock_tokenizer(seq_len: int = 4) -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = PAD_ID
    tok.eos_token_id = EOS_ID
    input_ids = torch.tensor([[CHOSEN] * seq_len])
    attention_mask = torch.ones(BATCH, seq_len, dtype=torch.long)
    tok_output = MagicMock()
    tok_output.input_ids = input_ids
    tok_output.attention_mask = attention_mask
    tok_output.items.return_value = [
        ("input_ids", input_ids),
        ("attention_mask", attention_mask),
    ]
    tok_output.to.return_value = tok_output
    tok.return_value = tok_output
    tok.decode.return_value = "Paris is the capital of France."
    return tok


def make_mock_sbert(similarity: float = 0.85) -> MagicMock:
    encoder = MagicMock()

    def encode_fn(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        base    = np.zeros(DIM, dtype=np.float32); base[0] = 1.0
        perturb = np.zeros(DIM, dtype=np.float32); perturb[1] = 1.0
        vecs = []
        for _ in range(n):
            v = (math.sqrt(similarity) * base
                 + math.sqrt(max(0.0, 1.0 - similarity)) * perturb)
            v = v / np.linalg.norm(v)
            vecs.append(v.astype(np.float32))
        return np.stack(vecs) if n > 1 else vecs[0]

    encoder.encode.side_effect = encode_fn
    return encoder


def make_mock_nli(label: int = 2) -> tuple:
    """NLI model always returning the same label (2=ENTAILMENT default)."""
    nli_model = MagicMock()
    logits = torch.zeros(1, 3)
    logits[0, label] = 10.0
    nli_model.return_value = SimpleNamespace(logits=logits)
    nli_tok = MagicMock()
    nli_tok.return_value = _DictWithTo({
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    })
    return nli_model, nli_tok


def make_verifier(
    sbert_sim: float = 0.85,
    nli_label: int = 2,
    include_nli: bool = True,
    config: CAEMConfig | None = None,
) -> MultiLayerVerifier:
    model     = make_mock_model()
    tokenizer = make_mock_tokenizer()
    encoder   = make_mock_sbert(sbert_sim)
    cfg       = config or CAEMConfig()
    nli_model, nli_tok = make_mock_nli(nli_label) if include_nli else (None, None)
    return MultiLayerVerifier(model, tokenizer, encoder, nli_model, nli_tok,
                              cfg, device="cpu")


# -----------------------------------------------------------------------------
# StoredConfidence dataclass
# -----------------------------------------------------------------------------

class TestStoredConfidenceDataclass:
    def test_fields_accessible(self):
        sc = StoredConfidence(p_entail=0.8, s_avg=0.7, h_norm=0.2, u_stored=0.74)
        assert sc.p_entail == pytest.approx(0.8)
        assert sc.s_avg    == pytest.approx(0.7)
        assert sc.h_norm   == pytest.approx(0.2)
        assert sc.u_stored == pytest.approx(0.74)

    def test_u_stored_in_unit_interval(self):
        for val in [0.0, 0.5, 1.0]:
            sc = StoredConfidence(p_entail=val, s_avg=val, h_norm=val, u_stored=val)
            assert 0.0 <= sc.u_stored <= 1.0


# -----------------------------------------------------------------------------
# p_entail signal
# -----------------------------------------------------------------------------

class TestPEntail:
    def test_all_entailment_gives_high_p_entail(self):
        """NLI model always returns ENTAILMENT -> p_entail near 1.0."""
        v = make_verifier(nli_label=2)
        input_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        p = v._compute_p_entail("q", "answer", input_ids)
        assert p > 0.95, f"Expected near-1.0, got {p:.4f}"

    def test_all_contradiction_gives_low_p_entail(self):
        """NLI model always returns CONTRADICTION -> p_entail near 0.0."""
        v = make_verifier(nli_label=0)
        input_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        p = v._compute_p_entail("q", "answer", input_ids)
        assert p < 0.10, f"Expected near-0.0, got {p:.4f}"

    def test_no_nli_model_returns_neutral(self):
        """When nli_model=None, fallback is 0.5."""
        v = make_verifier(include_nli=False)
        input_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        p = v._compute_p_entail("q", "answer", input_ids)
        assert p == pytest.approx(0.5)

    def test_p_entail_in_unit_interval(self):
        for label in [0, 1, 2]:
            v = make_verifier(nli_label=label)
            p = v._compute_p_entail("q", "ans",
                                    torch.zeros(1, SEQ_LEN, dtype=torch.long))
            assert 0.0 <= p <= 1.0

    def test_m_chains_generated_for_p_entail(self):
        """generate() called exactly M=3 times for p_entail."""
        cfg = CAEMConfig()
        cfg.sc_chains_m = 3
        v = make_verifier(config=cfg)
        count = [0]

        def count_gen(*args, **kwargs):
            count[0] += 1
            return make_plain_seq()

        v.model.generate.side_effect = count_gen
        v._compute_p_entail("q", "ans", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert count[0] == 3

    def test_error_returns_neutral(self):
        v = make_verifier()
        v.model.generate.side_effect = RuntimeError("OOM")
        p = v._compute_p_entail("q", "ans", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert p == pytest.approx(0.5)

    def test_nli_entail_prob_uses_softmax(self):
        """_nli_entail_prob returns a probability in [0,1], not an integer label."""
        v = make_verifier(nli_label=2)
        p = v._nli_entail_prob("premise", "hypothesis")
        assert 0.0 <= p <= 1.0
        # With logit=10 at ENTAILMENT and 0 elsewhere: softmax ≈ 1.0
        assert p > 0.95

    def test_nli_entail_prob_contradiction_near_zero(self):
        v = make_verifier(nli_label=0)
        p = v._nli_entail_prob("premise", "hypothesis")
        # ENTAILMENT logit=0 vs CONTRADICTION logit=10: softmax(entail) ≈ 0
        assert p < 0.05


# -----------------------------------------------------------------------------
# s_avg signal
# -----------------------------------------------------------------------------

class TestSAvg:
    def test_identical_chains_high_s_avg(self):
        """All chains identical -> pairwise sim = 1.0 -> s_avg ≈ 1.0."""
        v = make_verifier(sbert_sim=1.0)
        s = v._compute_s_avg("q", "ans", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert s > 0.95, f"Expected near-1.0, got {s:.4f}"

    def test_s_avg_in_unit_interval(self):
        for sim in [0.2, 0.6, 0.9]:
            v = make_verifier(sbert_sim=sim)
            s = v._compute_s_avg("q", "ans", torch.zeros(1, SEQ_LEN, dtype=torch.long))
            assert 0.0 <= s <= 1.0

    def test_m_chains_generated_for_s_avg(self):
        cfg = CAEMConfig()
        cfg.sc_chains_m = 3
        v = make_verifier(config=cfg)
        count = [0]

        def count_gen(*args, **kwargs):
            count[0] += 1
            return make_plain_seq()

        v.model.generate.side_effect = count_gen
        v._compute_s_avg("q", "ans", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert count[0] == 3

    def test_error_returns_neutral(self):
        v = make_verifier()
        v.model.generate.side_effect = RuntimeError("fail")
        s = v._compute_s_avg("q", "ans", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert s == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# h_norm signal
# -----------------------------------------------------------------------------

class TestHNorm:
    def test_all_entailment_low_h_norm(self):
        """All samples cluster into one -> H=0 -> h_norm=0."""
        v = make_verifier(nli_label=2)
        h = v._compute_h_norm("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert h < 0.05, f"Expected near-0, got {h:.4f}"

    def test_h_norm_in_unit_interval(self):
        for label in [0, 1, 2]:
            v = make_verifier(nli_label=label)
            h = v._compute_h_norm("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
            assert 0.0 <= h <= 1.0

    def test_surface_fallback_identical_samples_zero_h_norm(self):
        """Identical samples -> 1 surface cluster -> H=0 -> h_norm=0."""
        v = make_verifier(include_nli=False)
        v.tokenizer.decode.return_value = "identical answer"
        h = v._compute_h_norm("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert h == pytest.approx(0.0, abs=1e-4)

    def test_k_samples_generated(self):
        cfg = CAEMConfig()
        cfg.se_samples_k = 10
        v = make_verifier(config=cfg)
        count = [0]

        def count_gen(*args, **kwargs):
            count[0] += 1
            return make_plain_seq()

        v.model.generate.side_effect = count_gen
        v._compute_h_norm("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert count[0] == 10

    def test_error_returns_neutral(self):
        v = make_verifier()
        v.model.generate.side_effect = RuntimeError("OOM")
        h = v._compute_h_norm("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert h == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# Combined verify()
# -----------------------------------------------------------------------------

class TestVerify:
    def test_returns_stored_confidence(self):
        v = make_verifier()
        sc = v.verify("Who wrote Hamlet?", "William Shakespeare")
        assert isinstance(sc, StoredConfidence)

    def test_u_stored_in_unit_interval(self):
        v = make_verifier()
        sc = v.verify("q", "answer")
        assert 0.0 <= sc.u_stored <= 1.0

    def test_all_signals_in_unit_interval(self):
        v = make_verifier()
        sc = v.verify("q", "answer")
        for name, val in [("p_entail", sc.p_entail),
                          ("s_avg", sc.s_avg), ("h_norm", sc.h_norm)]:
            assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"

    def test_u_stored_formula(self):
        """û_stored = 0.50·p_entail + 0.30·s_avg + 0.20·(1 − h_norm)."""
        cfg = CAEMConfig()
        assert (cfg.u_stored_weight_nli + cfg.u_stored_weight_sc
                + cfg.u_stored_weight_se) == pytest.approx(1.0, abs=1e-6)

        v = make_verifier(config=cfg)
        sc = v.verify("q", "answer")
        expected = (
            cfg.u_stored_weight_nli * sc.p_entail
            + cfg.u_stored_weight_sc  * sc.s_avg
            + cfg.u_stored_weight_se  * (1.0 - sc.h_norm)
        )
        assert math.isclose(sc.u_stored, expected, abs_tol=1e-5)

    def test_high_confidence_all_signals(self):
        """High NLI entailment, high SBERT sim, all samples same cluster -> high û_stored."""
        v = make_verifier(nli_label=2, sbert_sim=0.99)
        sc = v.verify("q", "answer")
        assert sc.u_stored > 0.70, f"Expected high û_stored, got {sc.u_stored:.4f}"

    def test_pre_tokenized_input_ids_accepted(self):
        v = make_verifier()
        ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        sc = v.verify("q", "answer", input_ids=ids)
        assert isinstance(sc, StoredConfidence)

    def test_u_stored_weight_nli_is_dominant(self):
        """Default weights: NLI=0.50 > SC=0.30 > SE=0.20."""
        cfg = CAEMConfig()
        assert cfg.u_stored_weight_nli > cfg.u_stored_weight_sc
        assert cfg.u_stored_weight_sc  > cfg.u_stored_weight_se

    def test_weight_sum_is_one(self):
        cfg = CAEMConfig()
        total = (cfg.u_stored_weight_nli + cfg.u_stored_weight_sc
                 + cfg.u_stored_weight_se)
        assert math.isclose(total, 1.0, abs_tol=1e-6)


# -----------------------------------------------------------------------------
# should_store() gate
# -----------------------------------------------------------------------------

class TestShouldStore:
    def test_above_threshold_stores(self):
        v = make_verifier()
        sc = StoredConfidence(p_entail=0.9, s_avg=0.9, h_norm=0.1, u_stored=0.80)
        assert v.should_store(sc) is True

    def test_below_threshold_discards(self):
        v = make_verifier()
        sc = StoredConfidence(p_entail=0.3, s_avg=0.2, h_norm=0.9, u_stored=0.20)
        assert v.should_store(sc) is False

    def test_at_exact_threshold_stores(self):
        """u_stored == retroverify_prune_threshold -> should store (≥)."""
        cfg = CAEMConfig()
        v = make_verifier(config=cfg)
        sc = StoredConfidence(p_entail=0.5, s_avg=0.5, h_norm=0.5,
                              u_stored=cfg.retroverify_prune_threshold)
        assert v.should_store(sc) is True

    def test_default_threshold_is_half(self):
        """Default retroverify_prune_threshold = 0.50."""
        assert CAEMConfig().retroverify_prune_threshold == pytest.approx(0.50)
