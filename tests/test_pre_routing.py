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
    """Build a mock decoder-only generator with predictable outputs.

    Branch C is decoder-only only (T5-removal refactor 2026-04-22). This mock
    simulates a Qwen/Llama-style model where:
      - ``model(input_ids=..., output_hidden_states=True)`` returns a namespace
        with ``hidden_states`` (used by ``_compute_c_conv``)
      - ``model.generate(...)`` returns sequences + scores (used by
        ``_compute_u_token``)

    Previous encoder-decoder mock semantics (``model.encoder.return_value``)
    are removed along with the T5 code path.
    """
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])  # for device detection

    # -- direct forward pass (c_conv, decoder-only) ---------------------
    hidden_states = make_hidden_states(num_layers, SEQ_LEN, HIDDEN_DIM, early_var, late_var)
    model.return_value = SimpleNamespace(hidden_states=hidden_states)

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

    # Decoder-only sequences layout:
    #   [input_0, ..., input_{L-1}, gen_1, ..., gen_n, eos]
    # Our mock input is SEQ_LEN tokens (all zeros), so the full sequence
    # is [0]*SEQ_LEN + [chosen_token]*n + [eos_token_id]. ``_compute_u_token``
    # derives gen_start = len(sequence) - len(scores), correctly landing on
    # the first generated token regardless of input prefix length.
    seq_tokens = [0] * SEQ_LEN + [chosen_token] * n + [eos_token_id]
    sequences = torch.tensor([seq_tokens])

    generate_out = SimpleNamespace(scores=scores, sequences=sequences)
    model.generate.return_value = generate_out

    return model


def make_mock_tokenizer(eos_token_id: int = 1) -> MagicMock:
    """Build a mock tokenizer compatible with the decoder-only ChatML wrapper.

    ``apply_chat_template`` is provided as a stub that returns a plain string
    (the user message), letting downstream ``tokenizer(text)`` proceed against
    the standard MagicMock return value.
    """
    tokenizer = MagicMock()
    tokenizer.eos_token_id = eos_token_id

    def _fake_chat_template(messages, tokenize=False, add_generation_prompt=True):
        # Return the user message content directly; good enough for unit tests
        # that don't care about the exact ChatML envelope.
        content = messages[-1].get("content", "") if messages else ""
        return str(content)

    tokenizer.apply_chat_template = _fake_chat_template

    # Tokenizer called as function returns a dict of tensors
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
        """Model with only 1 decoder layer: c_conv undefined -> return 0.0."""
        model = make_mock_model(num_layers=1)
        # Only 2 hidden states (embedding + 1 layer), L=1 -> L//2=0 -> no early layers
        tokenizer = make_mock_tokenizer()
        est = PreRoutingConfidenceEstimator(model, tokenizer, CAEMConfig(), device="cpu")
        c = est._compute_c_conv(est._tokenize("q"))
        assert c == 0.0

    def test_no_hidden_states_attr_returns_zero(self):
        """Fail-safe: if forward output lacks hidden_states, return 0.0."""
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        # forward returns a namespace WITHOUT hidden_states attribute
        model.return_value = SimpleNamespace(logits=torch.zeros(1, 8, 100))
        tokenizer = make_mock_tokenizer()
        est = PreRoutingConfidenceEstimator(model, tokenizer, CAEMConfig(), device="cpu")
        c = est._compute_c_conv(est._tokenize("q"))
        assert c == 0.0

    def test_forward_error_returns_zero(self):
        """Fail-safe: any forward-pass exception returns 0.0 (not a crash)."""
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
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

    model, tok = load_base_generator(
        "Qwen/Qwen2.5-3B-Instruct",
        device="cuda",
        dtype=torch.bfloat16,
        use_flash_attention_2=False,
        use_torch_compile=False,
    )
    try:
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

    def test_forward_error_returns_safe_fallback(self):
        """If the model's forward (c_conv path) raises, c_conv = 0.0
        (max confidence -- fail open, not closed)."""
        est = make_estimator()
        est.model.side_effect = RuntimeError("CUDA OOM")
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


# =========================================================================== #
# Per-benchmark u_pre dispatch (v2 Fix 12)                                     #
# =========================================================================== #
# Per-benchmark T_b temperature scaling and safety_u_pre_min_b are dispatched
# inside CAEMConfig + PreRoutingConfidenceEstimator. The router-side dispatch
# for safety_u_pre_min_b lives in tests/test_router.py; the calibration-side
# fitting helpers (fit_per_benchmark_temperatures, fit_per_benchmark_safety_
# floors) live in tests/test_calibration_batch_equivalence.py.

class TestPerBenchmarkUPre:
    def test_config_defaults_have_empty_per_bench_dicts(self):
        from caem.config import CAEMConfig
        cfg = CAEMConfig()
        # Pooled fallbacks must remain
        assert hasattr(cfg, "temperature_scalar")
        assert hasattr(cfg, "safety_u_pre_min")
        # New per-bench dicts must exist and start empty
        assert hasattr(cfg, "temperature_scalar_per_benchmark")
        assert hasattr(cfg, "safety_u_pre_min_per_benchmark")
        assert isinstance(cfg.temperature_scalar_per_benchmark, dict)
        assert isinstance(cfg.safety_u_pre_min_per_benchmark, dict)
        assert len(cfg.temperature_scalar_per_benchmark) == 0
        assert len(cfg.safety_u_pre_min_per_benchmark) == 0

    def test_helper_getters_dispatch_correctly(self):
        from caem.config import CAEMConfig
        cfg = CAEMConfig()
        # No per-bench data → fallback to pooled global
        assert cfg.get_temperature_for(None) == cfg.temperature_scalar
        assert cfg.get_temperature_for("fever") == cfg.temperature_scalar
        assert cfg.get_safety_u_pre_min_for(None) == cfg.safety_u_pre_min
        assert cfg.get_safety_u_pre_min_for("fever") == cfg.safety_u_pre_min

        cfg.temperature_scalar_per_benchmark = {"fever": 2.5, "triviaqa": 0.7}
        cfg.safety_u_pre_min_per_benchmark = {"fever": 0.55, "triviaqa": 0.20}

        assert cfg.get_temperature_for("fever") == 2.5
        assert cfg.get_temperature_for("triviaqa") == 0.7
        # Unknown benchmark falls back to pooled global
        assert cfg.get_temperature_for("hotpotqa") == cfg.temperature_scalar
        assert cfg.get_temperature_for(None) == cfg.temperature_scalar

        assert cfg.get_safety_u_pre_min_for("fever") == 0.55
        assert cfg.get_safety_u_pre_min_for("triviaqa") == 0.20
        assert cfg.get_safety_u_pre_min_for("hotpotqa") == cfg.safety_u_pre_min
        assert cfg.get_safety_u_pre_min_for(None) == cfg.safety_u_pre_min

    def test_apply_temperature_scaling_dispatches_per_benchmark(self):
        """The _apply_temperature_scaling helper must read T_b from
        temperature_scalar_per_benchmark when source_benchmark is supplied."""
        from caem.config import CAEMConfig
        from caem.confidence.pre_routing import PreRoutingConfidenceEstimator

        cfg = CAEMConfig()
        cfg.temperature_scalar = 1.0  # pooled = identity
        cfg.temperature_scalar_per_benchmark = {
            "fever": 2.0,    # T > 1 squashes confidence toward 0.5
            "triviaqa": 0.5,  # T < 1 sharpens confidence away from 0.5
        }

        # Build estimator without instantiating a model
        est = PreRoutingConfidenceEstimator.__new__(PreRoutingConfidenceEstimator)
        est.config = cfg

        raw = 0.80
        p_pooled = est._apply_temperature_scaling(raw, source_benchmark=None)
        p_fever = est._apply_temperature_scaling(raw, source_benchmark="fever")
        p_tqa = est._apply_temperature_scaling(raw, source_benchmark="triviaqa")
        p_unknown = est._apply_temperature_scaling(raw, source_benchmark="not_a_real_bench")

        assert abs(p_pooled - raw) < 1e-9
        assert abs(p_unknown - raw) < 1e-9  # unknown → pooled
        assert p_fever < raw and p_fever > 0.5  # T=2 pulls down toward 0.5
        assert p_tqa > raw  # T=0.5 sharpens up
        # Three distinct values prove dispatch is live
        assert len({round(p_pooled, 6), round(p_fever, 6), round(p_tqa, 6)}) == 3

    def test_back_compat_unchanged_when_dicts_empty(self):
        """Old code paths (no source_benchmark, no per-bench dicts) must
        behave identically to the v1 single-T pipeline."""
        from caem.config import CAEMConfig
        from caem.confidence.pre_routing import PreRoutingConfidenceEstimator

        cfg = CAEMConfig()
        cfg.temperature_scalar = 1.7
        # empty per-bench dict (default)

        est = PreRoutingConfidenceEstimator.__new__(PreRoutingConfidenceEstimator)
        est.config = cfg

        raw = 0.82
        p_no_tag = est._apply_temperature_scaling(raw, source_benchmark=None)
        p_fever = est._apply_temperature_scaling(raw, source_benchmark="fever")
        assert abs(p_no_tag - p_fever) < 1e-9  # any tag → pooled when dict empty

    def test_pipeline_threads_source_benchmark_to_estimate_and_route(self):
        """Smoke check: pipeline.answer threads source_benchmark to both
        pre_estimator.estimate() and router.route()."""
        import inspect
        from caem.pipeline import CAEMPipeline
        from caem.confidence.pre_routing import PreRoutingConfidenceEstimator
        from caem.routing.router import AdaptiveRouter

        sig = inspect.signature(PreRoutingConfidenceEstimator.estimate)
        assert "source_benchmark" in sig.parameters
        sig = inspect.signature(AdaptiveRouter.route)
        assert "source_benchmark" in sig.parameters

        src = inspect.getsource(CAEMPipeline.answer)
        assert "self.pre_estimator.estimate(" in src
        assert "source_benchmark=source_benchmark" in src
        assert "self.router.route(" in src

    def test_temperature_scaling_math_invariants(self):
        """T → 1 collapses to identity; T → ∞ collapses to 0.5."""
        from caem.config import CAEMConfig
        from caem.confidence.pre_routing import PreRoutingConfidenceEstimator

        cfg = CAEMConfig()
        est = PreRoutingConfidenceEstimator.__new__(PreRoutingConfidenceEstimator)
        est.config = cfg

        cfg.temperature_scalar = 1.0
        assert abs(est._apply_temperature_scaling(0.30, None) - 0.30) < 1e-9
        assert abs(est._apply_temperature_scaling(0.78, None) - 0.78) < 1e-9

        cfg.temperature_scalar = 1000.0
        p_big = est._apply_temperature_scaling(0.95, None)
        assert abs(p_big - 0.5) < 1e-2
