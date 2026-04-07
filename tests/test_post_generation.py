"""
tests/test_post_generation.py
==============================
Unit tests for PostGenerationConfidenceEstimator (Stage 4a).

All tests use lightweight mocks -- no Flan-T5, no RoBERTa download.
Mock design mirrors test_pre_routing.py: controlled logits and hidden
states to isolate each signal independently.

Test coverage:
  - PostGenerationConfidence.should_accept()
  - u_token signal (answer token log-probs)
  - u_dropout signal (MC Dropout variance)
  - u_consistency signal (pairwise cosine similarity of chains)
  - u_entropy signal: NLI-based clustering + surface fallback
  - Combined û formula (weights from config)
  - accept/escalate gate
  - Error handling (pessimistic/neutral fail behaviour)
  - eval() always restored after MC Dropout

Run with:
    python -m pytest tests/test_post_generation.py -v
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from caem.config import CAEMConfig
from caem.confidence.post_generation import PostGenerationConfidenceEstimator
from caem.memory.entry import PostGenerationConfidence


# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------

VOCAB      = 100
HIDDEN     = 64
SEQ_LEN    = 8
BATCH      = 1
DIM        = 768
EOS_ID     = 1
PAD_ID     = 0
CHOSEN_TOK = 2


# -----------------------------------------------------------------------------
# Mock builders
# -----------------------------------------------------------------------------

def make_score(chosen_logit: float = 8.0) -> torch.Tensor:
    """Return (1, VOCAB) logits with high value at CHOSEN_TOK."""
    s = torch.full((BATCH, VOCAB), -10.0)
    s[0, CHOSEN_TOK] = chosen_logit
    return s


def make_generate_output(n_tokens: int = 3, chosen_logit: float = 8.0):
    """Greedy generate output: bos + n_tokens + eos."""
    scores = tuple(make_score(chosen_logit) for _ in range(n_tokens))
    seq = [0] + [CHOSEN_TOK] * n_tokens + [EOS_ID]
    return SimpleNamespace(scores=scores, sequences=torch.tensor([seq]))


def make_model_forward_output(chosen_logit: float = 8.0, seq_len: int = 4):
    """Model forward output for u_token: logits (1, seq_len, vocab)."""
    logits = torch.full((BATCH, seq_len, VOCAB), -10.0)
    logits[0, :, CHOSEN_TOK] = chosen_logit
    return SimpleNamespace(logits=logits)


def make_mock_model(
    chosen_logit: float = 8.0,
    n_generate_tokens: int = 3,
) -> MagicMock:
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    # generate() is called in two different modes:
    #   - u_dropout:     return_dict_in_generate=True  -> returns dict-like with .scores/.sequences
    #   - u_consistency / u_entropy: no return_dict_in_generate -> returns plain tensor
    # Use side_effect to differentiate so each calling site gets the right shape.
    _plain_seq = torch.tensor([[0] + [CHOSEN_TOK] * n_generate_tokens + [EOS_ID]])
    _dict_out  = make_generate_output(n_generate_tokens, chosen_logit)

    def _generate_side_effect(*args, **kwargs):
        if kwargs.get("return_dict_in_generate"):
            return _dict_out
        return _plain_seq

    model.generate.side_effect = _generate_side_effect
    model.return_value = make_model_forward_output(chosen_logit)
    model.training = False
    return model


def make_mock_tokenizer(seq_len: int = 4) -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = PAD_ID
    tok.eos_token_id = EOS_ID
    # as_target_tokenizer() returns self (context manager)
    tok.as_target_tokenizer.return_value.__enter__ = lambda s: tok
    tok.as_target_tokenizer.return_value.__exit__ = MagicMock(return_value=False)
    # Tokenizer call: must support BOTH attribute access (.input_ids) AND dict
    # iteration (.items()) since _tokenize() calls inputs.items() while
    # _compute_u_token() calls .input_ids directly.
    input_ids = torch.tensor([[CHOSEN_TOK] * seq_len])
    attention_mask = torch.ones(BATCH, seq_len, dtype=torch.long)
    tok_output = MagicMock()
    tok_output.input_ids = input_ids
    tok_output.attention_mask = attention_mask
    tok_output.items.return_value = [
        ("input_ids", input_ids),
        ("attention_mask", attention_mask),
    ]
    # Support .to(device) chaining used in _tokenize()
    tok_output.to.return_value = tok_output
    tok.return_value = tok_output
    tok.decode.return_value = "Paris is the capital of France."
    return tok


def make_mock_sbert(similarity: float = 0.85) -> MagicMock:
    """SBERT encoder that returns unit vectors with controlled pairwise cosine sim."""
    encoder = MagicMock()

    def encode_fn(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        # Build vectors such that pairwise cosine sim ≈ similarity
        base = np.zeros(DIM, dtype=np.float32)
        base[0] = 1.0
        perturb = np.zeros(DIM, dtype=np.float32)
        perturb[1] = 1.0
        # Each vector = sqrt(sim)*base + sqrt(1-sim)*perturb, then normalise
        vecs = []
        for _ in range(n):
            v = math.sqrt(similarity) * base + math.sqrt(max(0.0, 1.0 - similarity)) * perturb
            v = v / np.linalg.norm(v)
            vecs.append(v.astype(np.float32))
        return np.stack(vecs) if n > 1 else vecs[0]

    encoder.encode.side_effect = encode_fn
    return encoder


class _DictWithTo(dict):
    """dict subclass that supports .to(device) for HuggingFace-style tokenizer output."""
    def to(self, device):
        return self


def make_mock_nli(label: int = 2) -> tuple:
    """NLI model that always returns the same label (2=ENTAILMENT by default)."""
    nli_model = MagicMock()
    logits = torch.zeros(1, 3)
    logits[0, label] = 10.0
    nli_model.return_value = SimpleNamespace(logits=logits)
    nli_tok = MagicMock()
    # Must be dict-like (supports **unpacking) AND have .to(device) for
    # _semantic_entropy_nli which calls enc = nli_tok(...).to(device) then
    # nli_model(**enc).
    nli_tok.return_value = _DictWithTo({
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    })
    return nli_model, nli_tok


def make_estimator(
    chosen_logit: float = 8.0,
    sbert_sim: float = 0.85,
    nli_label: int = 2,
    include_nli: bool = True,
    config: CAEMConfig | None = None,
) -> PostGenerationConfidenceEstimator:
    model = make_mock_model(chosen_logit)
    tokenizer = make_mock_tokenizer()
    encoder = make_mock_sbert(sbert_sim)
    cfg = config or CAEMConfig()
    if include_nli:
        nli_model, nli_tok = make_mock_nli(nli_label)
    else:
        nli_model, nli_tok = None, None
    return PostGenerationConfidenceEstimator(
        model, tokenizer, encoder, nli_model, nli_tok, cfg, device="cpu"
    )


# -----------------------------------------------------------------------------
# PostGenerationConfidence dataclass
# -----------------------------------------------------------------------------

class TestPostGenerationConfidenceDataclass:
    def test_should_accept_above_threshold(self):
        pgc = PostGenerationConfidence(
            u_token=0.8, u_dropout=0.7, u_consistency=0.9, u_entropy=0.8, u_hat=0.75
        )
        assert pgc.should_accept(0.60) is True

    def test_should_accept_below_threshold_escalates(self):
        pgc = PostGenerationConfidence(
            u_token=0.3, u_dropout=0.4, u_consistency=0.5, u_entropy=0.3, u_hat=0.38
        )
        assert pgc.should_accept(0.60) is False

    def test_should_accept_at_exact_threshold(self):
        pgc = PostGenerationConfidence(
            u_token=0.6, u_dropout=0.6, u_consistency=0.6, u_entropy=0.6, u_hat=0.60
        )
        assert pgc.should_accept(0.60) is True

    def test_u_hat_in_unit_interval(self):
        for val in [0.0, 0.5, 1.0]:
            pgc = PostGenerationConfidence(
                u_token=val, u_dropout=val, u_consistency=val, u_entropy=val, u_hat=val
            )
            assert 0.0 <= pgc.u_hat <= 1.0


# -----------------------------------------------------------------------------
# u_token signal
# -----------------------------------------------------------------------------

class TestUTokenPostGen:
    def test_high_logit_produces_high_u_token(self):
        est = make_estimator(chosen_logit=20.0)
        input_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        u = est._compute_u_token(input_ids, "Paris")
        assert u > 0.90, f"Expected > 0.90, got {u:.4f}"

    def test_u_token_in_unit_interval(self):
        for logit in [0.5, 2.0, 8.0, 15.0]:
            est = make_estimator(chosen_logit=logit)
            u = est._compute_u_token(torch.zeros(1, SEQ_LEN, dtype=torch.long), "answer")
            assert 0.0 <= u <= 1.0

    def test_model_error_returns_zero(self):
        est = make_estimator()
        est.model.side_effect = RuntimeError("OOM")
        u = est._compute_u_token(torch.zeros(1, SEQ_LEN, dtype=torch.long), "ans")
        assert u == 0.0

    def test_increases_with_logit(self):
        u_low  = make_estimator(chosen_logit=1.0)._compute_u_token(
            torch.zeros(1, SEQ_LEN, dtype=torch.long), "a")
        u_high = make_estimator(chosen_logit=18.0)._compute_u_token(
            torch.zeros(1, SEQ_LEN, dtype=torch.long), "a")
        assert u_high > u_low


# -----------------------------------------------------------------------------
# u_dropout signal
# -----------------------------------------------------------------------------

class TestUDropout:
    def test_low_variance_produces_high_u_dropout(self):
        """All K passes return identical probs -> variance=0 -> u_dropout=1.0."""
        est = make_estimator(chosen_logit=20.0)
        # All K passes return same output -> var ≈ 0
        u = est._compute_u_dropout(torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert u > 0.90, f"Expected high u_dropout, got {u:.4f}"

    def test_u_dropout_in_unit_interval(self):
        est = make_estimator()
        u = est._compute_u_dropout(torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert 0.0 <= u <= 1.0

    def test_eval_mode_always_restored(self):
        """model.eval() must be called after MC Dropout even if exception occurs."""
        est = make_estimator()
        eval_called = []
        original_eval = est.model.eval
        est.model.eval = lambda: eval_called.append(True) or original_eval()

        est._compute_u_dropout(torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert len(eval_called) >= 1, "model.eval() was not called after MC Dropout"

    def test_eval_mode_restored_on_exception(self):
        """eval() must be called even when generate() raises."""
        est = make_estimator()
        eval_called = []
        est.model.train = MagicMock()
        est.model.eval = lambda: eval_called.append(True)
        est.model.generate.side_effect = RuntimeError("GPU error")

        u = est._compute_u_dropout(torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert len(eval_called) >= 1
        assert u == 0.5  # Neutral fallback on error

    def test_k_passes_called(self):
        """Model.generate() is called exactly K times."""
        cfg = CAEMConfig()
        cfg.mc_dropout_k = 5
        est = make_estimator(config=cfg)
        call_count = [0]
        orig = est.model.generate.side_effect

        def counting_generate(*args, **kwargs):
            call_count[0] += 1
            return make_generate_output()

        est.model.generate.side_effect = counting_generate
        est._compute_u_dropout(torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert call_count[0] == 5


# -----------------------------------------------------------------------------
# u_consistency signal
# -----------------------------------------------------------------------------

class TestUConsistency:
    def test_identical_chains_produce_high_consistency(self):
        """All chains decode to same string -> embeddings identical -> sim=1.0."""
        est = make_estimator(sbert_sim=1.0)
        u = est._compute_u_consistency("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert u > 0.95, f"Expected near-1.0, got {u:.4f}"

    def test_u_consistency_in_unit_interval(self):
        for sim in [0.3, 0.6, 0.9]:
            est = make_estimator(sbert_sim=sim)
            u = est._compute_u_consistency("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
            assert 0.0 <= u <= 1.0

    def test_m_chains_generated(self):
        """model.generate() called exactly M=3 times for consistency."""
        cfg = CAEMConfig()
        cfg.sc_chains_m = 3
        est = make_estimator(config=cfg)
        count = [0]

        def count_generate(*args, **kwargs):
            count[0] += 1
            return torch.zeros(1, 5, dtype=torch.long)

        est.model.generate.side_effect = count_generate
        est._compute_u_consistency("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert count[0] == 3

    def test_error_returns_neutral(self):
        est = make_estimator()
        est.model.generate.side_effect = RuntimeError("fail")
        u = est._compute_u_consistency("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert u == 0.5


# -----------------------------------------------------------------------------
# u_entropy signal
# -----------------------------------------------------------------------------

class TestUEntropy:
    def test_all_entailment_single_cluster_low_entropy(self):
        """All K samples equivalent (all entail each other) -> 1 cluster -> H=0 -> u_entropy=1.0."""
        est = make_estimator(nli_label=2, include_nli=True)  # 2=ENTAILMENT
        u = est._compute_u_entropy("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert u > 0.95, f"Expected near-1.0 with all-entailment, got {u:.4f}"

    def test_no_entailment_all_unique_high_entropy(self):
        """No sample entails any other -> K clusters -> max entropy -> u_entropy near 0."""
        est = make_estimator(nli_label=0, include_nli=True)  # 0=CONTRADICTION

        # Make each sample unique so clustering doesn't merge them
        responses = [f"unique answer {i}" for i in range(10)]
        resp_iter = iter(responses)
        est.tokenizer.decode.side_effect = lambda *a, **kw: next(resp_iter)

        u = est._compute_u_entropy("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        # With K=10 unique clusters: H = log2(10) ≈ 3.32, h_norm = 1.0 -> u_entropy ≈ 0.0
        assert u < 0.15, f"Expected near-0 with all-unique, got {u:.4f}"

    def test_surface_fallback_no_nli_model(self):
        """Without NLI model, surface entropy fallback is used and returns [0,1]."""
        est = make_estimator(include_nli=False)
        u = est._compute_u_entropy("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert 0.0 <= u <= 1.0

    def test_surface_fallback_identical_samples_high_u(self):
        """All samples identical -> 1 surface cluster -> H=0 -> u_entropy=1.0."""
        est = make_estimator(include_nli=False)
        est.tokenizer.decode.return_value = "identical answer"
        u = est._compute_u_entropy("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert u == pytest.approx(1.0, abs=1e-4)

    def test_k_samples_generated(self):
        """K=10 samples generated for SE."""
        cfg = CAEMConfig()
        cfg.se_samples_k = 10
        est = make_estimator(config=cfg)
        count = [0]

        def count_generate(*args, **kwargs):
            count[0] += 1
            return torch.zeros(1, 5, dtype=torch.long)

        est.model.generate.side_effect = count_generate
        est._compute_u_entropy("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert count[0] == 10

    def test_u_entropy_in_unit_interval(self):
        est = make_estimator(nli_label=2)
        u = est._compute_u_entropy("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert 0.0 <= u <= 1.0

    def test_error_returns_neutral(self):
        est = make_estimator()
        est.model.generate.side_effect = RuntimeError("OOM")
        u = est._compute_u_entropy("q", torch.zeros(1, SEQ_LEN, dtype=torch.long))
        assert u == 0.5


# -----------------------------------------------------------------------------
# NLI clustering (_semantic_entropy_nli)
# -----------------------------------------------------------------------------

class TestNLIClustering:
    def test_all_entailment_one_cluster(self):
        """All pairs entail each other -> single cluster -> H = 0."""
        est = make_estimator(nli_label=2)
        samples = ["Paris", "Paris is in France", "The capital is Paris"]
        H = est._semantic_entropy_nli(samples, "capital of France")
        assert H == pytest.approx(0.0, abs=1e-6)

    def test_no_entailment_all_unique_clusters(self):
        """No pairs entail -> each sample its own cluster -> H = log2(N)."""
        est = make_estimator(nli_label=0)   # CONTRADICTION
        samples = [f"answer_{i}" for i in range(4)]
        H = est._semantic_entropy_nli(samples, "q")
        expected_H = math.log2(4)
        assert H == pytest.approx(expected_H, abs=0.01)

    def test_two_equal_clusters(self):
        """Two equal-sized clusters -> H = 1.0 bit."""
        # NLI: within-cluster = ENTAILMENT, cross-cluster = CONTRADICTION
        # We need 4 samples: [A, A, B, B]
        est = make_estimator()

        call_responses = []
        # Samples: [0,1,2,3] -> 0,1 entail each other; 2,3 entail each other
        # Pairs checked: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) × 2 directions = 12 calls
        nli_model = MagicMock()

        def nli_fn(**kwargs):
            # Track call count.
            # NOTE: the algorithm short-circuits on AND -- if the forward
            # direction returns CONTRADICTION the reverse is never called.
            # With 4 samples and CONT for cross-cluster pairs, call pattern is:
            #   (0,1) fwd=call1 ENT -> (0,1) bwd=call2 ENT -> union
            #   (0,2) fwd=call3 CONT -> SHORT CIRCUIT
            #   (0,3) fwd=call4 CONT -> SHORT CIRCUIT
            #   (1,2) fwd=call5 CONT -> SHORT CIRCUIT
            #   (1,3) fwd=call6 CONT -> SHORT CIRCUIT
            #   (2,3) fwd=call7 ENT -> (2,3) bwd=call8 ENT -> union
            # Total: 8 calls (not 12).
            call_responses.append(1)
            cnt = len(call_responses)
            if cnt in (1, 2, 7, 8):
                logits = torch.tensor([[0.0, 0.0, 10.0]])  # ENTAILMENT
            else:
                logits = torch.tensor([[10.0, 0.0, 0.0]])  # CONTRADICTION
            return SimpleNamespace(logits=logits)

        nli_model.side_effect = nli_fn
        est.nli_model = nli_model

        samples = ["A1", "A2", "B1", "B2"]
        H = est._semantic_entropy_nli(samples, "q")
        # Two clusters of size 2: p=0.5 each -> H = 1.0 bit
        assert H == pytest.approx(1.0, abs=0.01)


# -----------------------------------------------------------------------------
# Combined estimate()
# -----------------------------------------------------------------------------

class TestEstimate:
    def test_returns_post_generation_confidence(self):
        est = make_estimator()
        pgc = est.estimate("What is 2+2?", "4")
        assert isinstance(pgc, PostGenerationConfidence)

    def test_u_hat_in_unit_interval(self):
        est = make_estimator()
        pgc = est.estimate("q", "answer")
        assert 0.0 <= pgc.u_hat <= 1.0

    def test_u_hat_formula_uses_config_weights(self):
        """û = w_tok·u_token + w_drop·u_dropout + w_sc·u_sc + w_ent·u_entropy."""
        cfg = CAEMConfig()
        cfg.u_hat_weight_token   = 0.25
        cfg.u_hat_weight_dropout = 0.25
        cfg.u_hat_weight_sc      = 0.25
        cfg.u_hat_weight_entropy = 0.25
        est = make_estimator(config=cfg)
        pgc = est.estimate("q", "answer")
        expected = (
            0.25 * pgc.u_token
            + 0.25 * pgc.u_dropout
            + 0.25 * pgc.u_consistency
            + 0.25 * pgc.u_entropy
        )
        assert math.isclose(pgc.u_hat, expected, abs_tol=1e-5)

    def test_all_signals_in_unit_interval(self):
        est = make_estimator()
        pgc = est.estimate("q", "answer")
        for sig, val in [
            ("u_token", pgc.u_token),
            ("u_dropout", pgc.u_dropout),
            ("u_consistency", pgc.u_consistency),
            ("u_entropy", pgc.u_entropy),
        ]:
            assert 0.0 <= val <= 1.0, f"{sig}={val} out of [0,1]"

    def test_high_confidence_accepts(self):
        """All signals high -> û > 0.60 -> should_accept()=True."""
        est = make_estimator(chosen_logit=20.0, sbert_sim=0.99, nli_label=2)
        pgc = est.estimate("q", "answer")
        assert pgc.should_accept(0.60) is True

    def test_pre_tokenized_input_ids_accepted(self):
        """Passing input_ids skips internal tokenization."""
        est = make_estimator()
        input_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        pgc = est.estimate("q", "answer", input_ids=input_ids)
        assert isinstance(pgc, PostGenerationConfidence)

    def test_initial_weights_equal(self):
        """Default CAEMConfig weights must start at 0.25 each, NOT 0.20/0.40."""
        cfg = CAEMConfig()
        assert cfg.u_hat_weight_token   == pytest.approx(0.25, abs=1e-6)
        assert cfg.u_hat_weight_dropout == pytest.approx(0.25, abs=1e-6)
        assert cfg.u_hat_weight_sc      == pytest.approx(0.25, abs=1e-6)
        assert cfg.u_hat_weight_entropy == pytest.approx(0.25, abs=1e-6)

    def test_weights_sum_to_one(self):
        cfg = CAEMConfig()
        total = (cfg.u_hat_weight_token + cfg.u_hat_weight_dropout
                 + cfg.u_hat_weight_sc + cfg.u_hat_weight_entropy)
        assert math.isclose(total, 1.0, abs_tol=1e-6)
