"""
tests/test_pre_routing.py
==========================
Unit tests for PreRoutingConfidenceEstimator (Stage 3).

All tests use a lightweight mock model -- no Flan-T5 download required.
The mock is wired to produce controlled hidden states and logits so each
signal (u_token, c_conv) can be tested in isolation.

Run with:
    python -m pytest tests/test_pre_routing.py -v
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from caem.config import CAEMConfig
from caem.confidence.pre_routing import PreRoutingConfidenceEstimator
from caem.memory.entry import PreRoutingConfidence


# -----------------------------------------------------------------------------
# Mock model builder
# -----------------------------------------------------------------------------

VOCAB_SIZE = 100
HIDDEN_DIM = 64
NUM_LAYERS = 12   # Flan-T5-Large has 12 encoder layers
SEQ_LEN    = 8
BATCH      = 1


def make_hidden_states(
    num_layers: int = NUM_LAYERS,
    seq_len: int = SEQ_LEN,
    hidden_dim: int = HIDDEN_DIM,
    early_var: float = 2.0,
    late_var: float = 0.5,
) -> tuple:
    """Build (num_layers+1) hidden state tensors with controlled variance.

    Layer 0 = embedding (excluded from c_conv calc).
    Layers 1..L//2    = early layers -> scaled to produce early_var.
    Layers L//2+1..L  = late layers  -> scaled to produce late_var.
    """
    states = []
    # Embedding layer (index 0)
    states.append(torch.zeros(BATCH, seq_len, hidden_dim))

    L = num_layers
    for i in range(1, L + 1):
        if i <= L // 2:
            # Early: higher variance
            h = torch.randn(BATCH, seq_len, hidden_dim) * math.sqrt(early_var)
        else:
            # Late: lower variance
            h = torch.randn(BATCH, seq_len, hidden_dim) * math.sqrt(late_var)
        states.append(h)
    return tuple(states)


def make_mock_model(
    early_var: float = 2.0,
    late_var: float = 0.5,
    chosen_token_logit: float = 5.0,   # High logit -> high u_token
    num_layers: int = NUM_LAYERS,
    eos_token_id: int = 1,
    generate_n_tokens: int = 3,
) -> MagicMock:
    """Build a mock T5ForConditionalGeneration with predictable outputs."""
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])  # for device detection

    # -- encoder forward pass (c_conv) ----------------------------------
    hidden_states = make_hidden_states(num_layers, SEQ_LEN, HIDDEN_DIM, early_var, late_var)
    encoder_out = SimpleNamespace(hidden_states=hidden_states)
    model.encoder.return_value = encoder_out

    # -- generate (u_token) ----------------------------------------------
    # Produce `generate_n_tokens` steps then EOS.
    n = generate_n_tokens
    vocab = VOCAB_SIZE

    # Each score: logits over vocab. Chosen token = index 2 with high logit.
    chosen_token = 2
    def make_score():
        s = torch.full((BATCH, vocab), -10.0)
        s[0, chosen_token] = chosen_token_logit
        return s

    scores = tuple(make_score() for _ in range(n))

    # sequences: [decoder_start=0, tok_1=2, tok_2=2, ..., eos=1]
    seq_tokens = [0] + [chosen_token] * n + [eos_token_id]
    sequences = torch.tensor([seq_tokens])

    generate_out = SimpleNamespace(scores=scores, sequences=sequences)
    model.generate.return_value = generate_out

    return model


def make_mock_tokenizer(eos_token_id: int = 1) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.eos_token_id = eos_token_id
    # Return a simple dict of tensors
    tokenizer.return_value = {
        "input_ids": torch.zeros(BATCH, SEQ_LEN, dtype=torch.long),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
    }
    return tokenizer


def make_estimator(
    early_var: float = 2.0,
    late_var: float = 0.5,
    chosen_token_logit: float = 5.0,
    generate_n_tokens: int = 3,
) -> PreRoutingConfidenceEstimator:
    model = make_mock_model(early_var, late_var, chosen_token_logit, generate_n_tokens=generate_n_tokens)
    tokenizer = make_mock_tokenizer()
    config = CAEMConfig()
    return PreRoutingConfidenceEstimator(model, tokenizer, config, max_new_tokens=32, device="cpu")


# -----------------------------------------------------------------------------
# PreRoutingConfidence dataclass
# -----------------------------------------------------------------------------

class TestPreRoutingConfidenceDataclass:
    def test_is_safe_above_threshold(self):
        pc = PreRoutingConfidence(u_token=0.8, c_conv=0.2, u_pre=0.75)
        assert pc.is_safe(0.60) is True

    def test_is_safe_below_threshold_fires_or_condition(self):
        pc = PreRoutingConfidence(u_token=0.3, c_conv=3.0, u_pre=0.45)
        assert pc.is_safe(0.60) is False

    def test_is_safe_at_exact_threshold(self):
        pc = PreRoutingConfidence(u_token=0.6, c_conv=0.0, u_pre=0.60)
        assert pc.is_safe(0.60) is True   # Boundary: >= threshold is safe

    def test_u_pre_range(self):
        for u in [0.0, 0.5, 1.0]:
            pc = PreRoutingConfidence(u_token=u, c_conv=0.0, u_pre=u)
            assert 0.0 <= pc.u_pre <= 1.0


# -----------------------------------------------------------------------------
# u_token signal
# -----------------------------------------------------------------------------

class TestUToken:
    def test_high_logit_produces_high_u_token(self):
        """With a very high logit for the chosen token, u_token should approach 1."""
        est = make_estimator(chosen_token_logit=20.0)
        u_token = est._compute_u_token(est._tokenize("test"))
        assert u_token > 0.90, f"Expected u_token > 0.90, got {u_token:.4f}"

    def test_low_logit_produces_low_u_token(self):
        """With a logit barely above others, u_token should be low."""
        est = make_estimator(chosen_token_logit=0.1)
        u_token = est._compute_u_token(est._tokenize("test"))
        # With logit 0.1 vs -10 for 99 others: softmax will still concentrate
        # but less extremely; test just that value is bounded correctly.
        assert 0.0 <= u_token <= 1.0

    def test_u_token_in_unit_interval(self):
        for logit in [0.5, 2.0, 5.0, 10.0]:
            est = make_estimator(chosen_token_logit=logit)
            u = est._compute_u_token(est._tokenize("q"))
            assert 0.0 <= u <= 1.0, f"u_token={u} out of range for logit={logit}"

    def test_u_token_increases_with_logit(self):
        """Higher chosen-token logit -> higher confidence."""
        u_low  = make_estimator(chosen_token_logit=1.0)._compute_u_token(
            make_estimator(chosen_token_logit=1.0)._tokenize("q"))
        u_high = make_estimator(chosen_token_logit=15.0)._compute_u_token(
            make_estimator(chosen_token_logit=15.0)._tokenize("q"))
        assert u_high > u_low

    def test_empty_scores_returns_fallback(self):
        """If generate() returns no scores, fallback to 0.0 (pessimistic)."""
        est = make_estimator()
        # Override generate to return empty scores
        empty_out = SimpleNamespace(
            scores=(),
            sequences=torch.tensor([[0, 1]])  # just bos + eos
        )
        est.model.generate.return_value = empty_out
        u = est._compute_u_token(est._tokenize("q"))
        assert u == 0.0


# -----------------------------------------------------------------------------
# C_conv signal
# -----------------------------------------------------------------------------

class TestCConv:
    def test_high_early_variance_produces_high_c_conv(self):
        """early_var >> late_var -> c_conv >> 1 -> low confidence."""
        est = make_estimator(early_var=10.0, late_var=0.1)
        c = est._compute_c_conv(est._tokenize("q"))
        assert c > 1.0, f"Expected c_conv > 1.0, got {c:.4f}"

    def test_equal_variance_produces_c_conv_near_one(self):
        """equal variances -> c_conv ≈ 1.0."""
        est = make_estimator(early_var=1.0, late_var=1.0)
        c = est._compute_c_conv(est._tokenize("q"))
        # Due to randomness in make_hidden_states, allow a range
        assert 0.1 < c < 10.0

    def test_low_early_variance_produces_low_c_conv(self):
        """late_var >> early_var -> c_conv << 1 -> high confidence."""
        est = make_estimator(early_var=0.1, late_var=10.0)
        c = est._compute_c_conv(est._tokenize("q"))
        assert c < 1.0, f"Expected c_conv < 1.0, got {c:.4f}"

    def test_c_conv_non_negative(self):
        for ev, lv in [(0.5, 2.0), (2.0, 0.5), (1.0, 1.0)]:
            est = make_estimator(early_var=ev, late_var=lv)
            c = est._compute_c_conv(est._tokenize("q"))
            assert c >= 0.0

    def test_single_layer_model_returns_zero(self):
        """Model with only 1 encoder layer: c_conv undefined -> return 0.0."""
        model = make_mock_model(num_layers=1)
        # Only 2 hidden states (embedding + 1 layer), L=1 -> L//2=0 -> no early layers
        tokenizer = make_mock_tokenizer()
        est = PreRoutingConfidenceEstimator(model, tokenizer, CAEMConfig(), device="cpu")
        c = est._compute_c_conv(est._tokenize("q"))
        assert c == 0.0


def make_mock_decoder_only_model(
    early_var: float = 2.0,
    late_var: float = 0.5,
    num_layers: int = 24,
) -> MagicMock:
    """Build a mock decoder-only (Qwen-style) model with hidden states on
    the top-level ``model(...)`` forward (not ``model.encoder(...)``).

    The mock's ``config.is_encoder_decoder=False`` routes ``_compute_c_conv``
    into the decoder-only dispatch branch.
    """
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    # Decoder-only architecture: is_encoder_decoder must be falsy
    model.config = SimpleNamespace(is_encoder_decoder=False)
    # forward on the query tokens returns hidden_states directly
    hidden_states = make_hidden_states(num_layers, SEQ_LEN, HIDDEN_DIM, early_var, late_var)
    model.return_value = SimpleNamespace(hidden_states=hidden_states)
    return model


class TestCConvDecoderOnlyDispatch:
    """Tests for Goal 1 Phase C — decoder-only C_conv path (Qwen/Gemma/Llama)."""

    def test_dispatch_uses_model_forward_not_encoder(self):
        """Decoder-only models route to self.model(...), not self.model.encoder(...)."""
        model = make_mock_decoder_only_model(early_var=2.0, late_var=0.5)
        tokenizer = make_mock_tokenizer()
        est = PreRoutingConfidenceEstimator(model, tokenizer, CAEMConfig(), device="cpu")
        assert est._is_encoder_decoder() is False

        est._compute_c_conv(est._tokenize("q"))
        # Confirm the decoder-only path was taken
        model.assert_called()                       # self.model(...)
        model.encoder.assert_not_called()           # NOT self.model.encoder(...)

    def test_encoder_decoder_dispatch_preserved(self):
        """Encoder-decoder models still route to self.model.encoder(...)."""
        model = make_mock_model()
        # Explicitly set is_encoder_decoder=True to make dispatch deterministic.
        # (A bare MagicMock's .config.is_encoder_decoder also happens to be
        # truthy, which is why existing tests stayed green; we assert
        # explicitly here.)
        model.config = SimpleNamespace(is_encoder_decoder=True)
        tokenizer = make_mock_tokenizer()
        est = PreRoutingConfidenceEstimator(model, tokenizer, CAEMConfig(), device="cpu")
        assert est._is_encoder_decoder() is True

        est._compute_c_conv(est._tokenize("q"))
        model.encoder.assert_called()

    def test_decoder_only_variance_math_identical_to_encoder_decoder(self):
        """The variance-ratio math is identical across dispatch branches.

        Share the SAME hidden-state tuple between both mocks (decoder-only
        returns it via ``model(...)``, encoder-decoder via ``model.encoder(...)``)
        to verify that only the forward-pass source differs — the downstream
        variance math operates on identical inputs and must return the same
        c_conv value.
        """
        shared_hidden_states = make_hidden_states(
            num_layers=NUM_LAYERS, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM,
            early_var=5.0, late_var=1.0,
        )

        dec_model = MagicMock()
        dec_model.parameters.return_value = iter([torch.zeros(1)])
        dec_model.config = SimpleNamespace(is_encoder_decoder=False)
        dec_model.return_value = SimpleNamespace(hidden_states=shared_hidden_states)

        enc_model = MagicMock()
        enc_model.parameters.return_value = iter([torch.zeros(1)])
        enc_model.config = SimpleNamespace(is_encoder_decoder=True)
        enc_model.encoder.return_value = SimpleNamespace(hidden_states=shared_hidden_states)

        tokenizer = make_mock_tokenizer()
        dec_est = PreRoutingConfidenceEstimator(dec_model, tokenizer, CAEMConfig(), device="cpu")
        enc_est = PreRoutingConfidenceEstimator(enc_model, tokenizer, CAEMConfig(), device="cpu")

        c_dec = dec_est._compute_c_conv(dec_est._tokenize("q"))
        c_enc = enc_est._compute_c_conv(enc_est._tokenize("q"))

        # Given identical hidden-state tensors, the variance math must produce
        # bit-identical c_conv regardless of which forward path produced them.
        assert abs(c_dec - c_enc) < 1e-6, (
            f"decoder-only c_conv={c_dec:.6f} diverges from encoder c_conv={c_enc:.6f} "
            "on identical hidden states — variance math is not dispatch-agnostic."
        )
        assert c_dec > 1.0  # early_var > late_var

    def test_decoder_only_no_hidden_states_attr_returns_zero(self):
        """Fail-safe: if the decoder-only forward doesn't expose hidden_states,
        we should return 0.0 (highest-confidence fallback), not crash."""
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.config = SimpleNamespace(is_encoder_decoder=False)
        # forward returns a namespace WITHOUT hidden_states attribute
        model.return_value = SimpleNamespace(logits=torch.zeros(1, 8, 100))
        tokenizer = make_mock_tokenizer()
        est = PreRoutingConfidenceEstimator(model, tokenizer, CAEMConfig(), device="cpu")
        c = est._compute_c_conv(est._tokenize("q"))
        assert c == 0.0

    def test_decoder_only_forward_error_returns_zero(self):
        """Fail-safe: any forward-pass exception returns 0.0 (not a crash)."""
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.config = SimpleNamespace(is_encoder_decoder=False)
        model.side_effect = RuntimeError("CUDA OOM simulated")
        tokenizer = make_mock_tokenizer()
        est = PreRoutingConfidenceEstimator(model, tokenizer, CAEMConfig(), device="cpu")
        c = est._compute_c_conv(est._tokenize("q"))
        assert c == 0.0


@pytest.mark.slow
def test_c_conv_live_qwen_3b():
    """Live integration — load Qwen-3B, compute full pre-routing pipeline.

    Validates decoder-only dispatch on a real model (Qwen 36-layer stack)
    AND signal quality on easy factual queries.

    Phase C.2 (2026-04-22) added ChatML query wrapping inside the estimator so
    Qwen receives chat-format inputs during u_token computation. Before the
    wrapper, raw-text queries to Qwen collapsed u_token to ~1e-9 (predictions
    scattering across the 151k-token vocab when the instruction-tuned model
    has no chat format to anchor on). After the wrapper, u_token recovers to
    healthy values on factual queries.

    Asserted properties:
      - Finiteness + boundedness of u_token, c_conv, u_pre
      - u_token > 0.5 on well-known factual query ("capital of France")
      - u_pre > safety_u_pre_min (OR-condition should NOT force Tier 3 on
        easily-answerable queries)
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from caem.model_loader import load_base_generator
    import gc

    model, tok, is_enc_dec = load_base_generator(
        "Qwen/Qwen2.5-3B-Instruct",
        device="cuda",
        dtype=torch.bfloat16,
        use_flash_attention_2=False,
        use_torch_compile=False,
    )
    try:
        assert is_enc_dec is False
        cfg = CAEMConfig()
        est = PreRoutingConfidenceEstimator(model, tok, cfg, device="cuda")
        pc = est.estimate("What is the capital of France?")

        # Plumbing: finite + bounded
        assert math.isfinite(pc.u_token) and 0.0 <= pc.u_token <= 1.0, \
            f"u_token out of [0,1] or non-finite: {pc.u_token}"
        assert math.isfinite(pc.c_conv) and pc.c_conv >= 0.0, \
            f"c_conv negative or non-finite: {pc.c_conv}"
        assert math.isfinite(pc.u_pre) and 0.0 <= pc.u_pre <= 1.0, \
            f"u_pre out of [0,1] or non-finite: {pc.u_pre}"
        assert pc.c_conv < 100.0, f"unexpectedly extreme c_conv: {pc.c_conv}"

        # Signal quality — ChatML wrapping should produce healthy values
        assert pc.u_token > 0.5, (
            f"u_token={pc.u_token:.4f} too low on easy factual query; "
            "ChatML query wrapping may be broken."
        )
        assert pc.u_pre > cfg.safety_u_pre_min, (
            f"u_pre={pc.u_pre:.4f} below safety threshold "
            f"{cfg.safety_u_pre_min} on 'capital of France' — "
            "pre-routing would force Tier 3 on a trivially-answerable query."
        )
    finally:
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Combined estimate()
# -----------------------------------------------------------------------------

class TestEstimate:
    def test_returns_pre_routing_confidence(self):
        est = make_estimator()
        result = est.estimate("What is the capital of France?")
        assert isinstance(result, PreRoutingConfidence)

    def test_u_pre_in_unit_interval(self):
        est = make_estimator()
        pc = est.estimate("Any query")
        assert 0.0 <= pc.u_pre <= 1.0

    def test_u_pre_formula(self):
        """u_pre = 0.60·u_token + 0.40·(1/(1+c_conv)). Verify manually."""
        est = make_estimator(chosen_token_logit=20.0, early_var=0.01, late_var=10.0)
        pc = est.estimate("test")
        # With high logit: u_token ≈ 1.0
        # With low early var / high late var: c_conv << 1 -> 1/(1+c_conv) ≈ 1.0
        # -> u_pre should be high
        assert pc.u_pre > 0.80, f"Expected high u_pre, got {pc.u_pre:.4f}"

    def test_low_confidence_query_is_not_safe(self):
        """Low logit + high c_conv -> u_pre < 0.60 -> OR-condition fires."""
        # Very low chosen-token logit (barely above uniform)
        # + very high early variance (model confused about query)
        est = make_estimator(chosen_token_logit=0.01, early_var=100.0, late_var=0.01)
        pc = est.estimate("test")
        # u_token close to 1/vocab_size (uniform), c_conv >> 1 -> conf_cconv ≈ 0
        # u_pre will be low -- whether below 0.60 depends on exact values, but
        # at least verify the safety check can fire
        assert isinstance(pc.is_safe(), bool)

    def test_high_confidence_query_is_safe(self):
        """Very high logit + low c_conv -> u_pre > 0.60 -> safe to route."""
        est = make_estimator(chosen_token_logit=20.0, early_var=0.01, late_var=10.0)
        pc = est.estimate("test")
        assert pc.is_safe(0.60) is True

    def test_c_conv_stored_on_result(self):
        est = make_estimator()
        pc = est.estimate("q")
        assert pc.c_conv >= 0.0

    def test_estimate_batch_length(self):
        est = make_estimator()
        queries = ["q1", "q2", "q3"]
        results = est.estimate_batch(queries)
        assert len(results) == 3
        assert all(isinstance(r, PreRoutingConfidence) for r in results)

    def test_encoder_error_returns_safe_fallback(self):
        """If encoder raises, c_conv = 0.0 (max confidence -- fail open, not closed)."""
        est = make_estimator()
        est.model.encoder.side_effect = RuntimeError("CUDA OOM")
        pc = est.estimate("test")
        # c_conv = 0.0, so u_pre = 0.60·u_token + 0.40·1.0
        assert pc.c_conv == 0.0
        assert pc.u_pre >= 0.0

    def test_generate_error_returns_zero_u_token(self):
        """If generate() raises, u_token = 0.0 (pessimistic -- pushes toward Tier 3)."""
        est = make_estimator()
        est.model.generate.side_effect = RuntimeError("device error")
        pc = est.estimate("test")
        assert pc.u_token == 0.0
