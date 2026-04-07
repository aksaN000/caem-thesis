"""
caem/confidence/pre_routing.py
===============================
PreRoutingConfidenceEstimator -- Stage 3 of the CAEM pipeline.

Computes u_pre BEFORE routing, using two fast signals that require only a
single forward pass through the model. No generation beyond a greedy decode
of a short prefix is needed -- this keeps Tier 1 latency well under 400 ms.

Two signals
-----------
Signal 1 -- u_token (geometric mean of per-token log-probabilities)
    Measures how confidently the model assigns probability to each generated
    token. Low u_token = the model is guessing across many alternatives.

Signal 2 -- C_conv (Internal Convergence, adapted from Nandakishor 2025)
    Original: applied to decoder hidden states in decoder-only models.
    CAEM adaptation: applied to Flan-T5's *encoder* hidden states to measure
    how stably the model represents the query *before* generation begins.

    Intuition: if early encoder layers vary much more than late layers, the
    model has not settled on a stable representation of the query -- it is
    internally "confused" about what is being asked. High variance ratio =
    low confidence.

    Formula:  c_conv  = var(early_layers) / (var(late_layers) + ε)
              confidence from c_conv = 1 / (1 + c_conv)

Combined
--------
    u_pre = 0.60 · u_token + 0.40 · (1 / (1 + c_conv))    [DES]

The 0.60/0.40 split is a design choice (Category 2): token probability is the
primary reliability signal; c_conv is a secondary prior on query stability.

OR-condition
------------
    if u_pre < CAEMConfig.safety_u_pre_min (0.60):
        -> force Tier 3, regardless of memory similarity

This is checked via PreRoutingConfidence.is_safe(), not baked into any formula.
See: writing-suggestions.md C4-04, C4-10b for thesis framing.

Latency target: < 100 ms on GPU (one encoder forward pass + short greedy decode).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch

from caem.config import CAEMConfig
from caem.memory.entry import PreRoutingConfidence

logger = logging.getLogger(__name__)


class PreRoutingConfidenceEstimator:
    """Estimate u_pre from a query string before routing to any tier.

    Parameters
    ----------
    model : transformers.T5ForConditionalGeneration
        Flan-T5-Large (or compatible encoder-decoder). Must already be on
        the correct device and in eval mode.
    tokenizer : transformers.T5Tokenizer / AutoTokenizer
        Matching tokenizer for the model.
    config : CAEMConfig
        Pipeline configuration. Weights and thresholds come from here.
    max_new_tokens : int
        How many tokens to greedily decode for the u_token signal.
        Kept short (32) to stay within latency budget -- we only need a
        confidence proxy, not a full answer.
    device : str or None
        'cuda' / 'cpu'. Auto-detected from model if None.

    Usage
    -----
    >>> estimator = PreRoutingConfidenceEstimator(model, tokenizer, config)
    >>> pc = estimator.estimate("What is the capital of France?")
    >>> pc.u_pre          # e.g. 0.74
    >>> pc.is_safe()      # True -> safe to route via memory; False -> force Tier 3
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[CAEMConfig] = None,
        max_new_tokens: int = 32,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CAEMConfig()
        self.max_new_tokens = max_new_tokens

        if device is None:
            device = next(model.parameters()).device
        self.device = device

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def estimate(self, query: str) -> PreRoutingConfidence:
        """Compute pre-routing confidence for a single query.

        Parameters
        ----------
        query : str
            The raw question text.

        Returns
        -------
        PreRoutingConfidence
            Contains u_token, c_conv (raw), and u_pre (combined).
            Call .is_safe() to check the OR-condition.
        """
        inputs = self._tokenize(query)

        with torch.no_grad():
            u_token = self._compute_u_token(inputs)
            c_conv  = self._compute_c_conv(inputs)

        cfg = self.config
        confidence_from_cconv = 1.0 / (1.0 + c_conv)
        u_pre = (
            cfg.u_pre_token_weight * u_token
            + cfg.u_pre_cconv_weight * confidence_from_cconv
        )
        u_pre = float(np.clip(u_pre, 0.0, 1.0))

        logger.debug(
            "u_pre estimate | query='%s...' | u_token=%.4f | c_conv=%.4f | "
            "conf_cconv=%.4f | u_pre=%.4f | safe=%s",
            query[:50], u_token, c_conv, confidence_from_cconv, u_pre,
            u_pre >= cfg.safety_u_pre_min,
        )

        return PreRoutingConfidence(
            u_token=float(u_token),
            c_conv=float(c_conv),
            u_pre=u_pre,
        )

    # ------------------------------------------------------------------ #
    # Signal 1 -- u_token                                                  #
    # ------------------------------------------------------------------ #

    def _compute_u_token(self, inputs: dict) -> float:
        """Geometric mean of per-token log-probabilities.

        We run a short greedy decode (max_new_tokens=32) with output_scores=True,
        then collect the log P(chosen token) at each step.

        Geometric mean = exp(mean(log_probs)), which equals the per-token
        average probability under the model -- a natural confidence measure.

        Returns float in [0, 1]. Returns 0.0 on error (fail-safe to Tier 3).
        """
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # Greedy -- deterministic, fast
                output_scores=True,
                return_dict_in_generate=True,
            )
        except Exception as exc:
            logger.warning("u_token: generate() failed with %s -- returning 0.0.", exc)
            return 0.0

        scores = outputs.scores       # Tuple of (vocab_size,) tensors, one per step
        sequences = outputs.sequences # (1, seq_len) including prompt tokens

        # The generated token IDs start after the decoder prompt (usually <pad>).
        # outputs.sequences for encoder-decoder: [decoder_start, tok_1, ..., tok_n]
        # scores[t] = logits at step t before softmax, shape (batch, vocab)
        if not scores:
            logger.warning("u_token: no scores returned -- returning 0.5.")
            return 0.5

        log_probs = []
        for t, step_scores in enumerate(scores):
            # step_scores: (1, vocab_size) logits
            log_softmax = torch.nn.functional.log_softmax(step_scores[0], dim=-1)
            # Token chosen at step t is at position t+1 in sequences (after bos/pad)
            token_id = sequences[0, t + 1].item()
            if token_id == self.tokenizer.eos_token_id:
                break   # Stop at EOS -- don't include it in the mean
            log_prob = log_softmax[token_id].item()
            log_probs.append(log_prob)

        if not log_probs:
            return 0.5

        # Geometric mean: exp(mean(log_probs))
        u_token = math.exp(sum(log_probs) / len(log_probs))
        return float(np.clip(u_token, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Signal 2 -- C_conv (Internal Convergence)                            #
    # ------------------------------------------------------------------ #

    def _compute_c_conv(self, inputs: dict) -> float:
        """Encoder layer variance ratio -- query representation stability.

        Adaptation of Nandakishor (2025) C_conv to encoder-decoder architecture:
        - Original: decoder hidden states in decoder-only models (GPT-style)
        - CAEM: encoder hidden states in Flan-T5 (encoder-decoder)

        The encoder processes the query and produces contextual representations.
        If early layers disagree more than late layers, the model hasn't converged
        on a stable representation -- a signal of low query comprehension confidence.

        Formula:
            early_layers = hidden_states[1 : L//2 + 1]   (layers 1 to L/2)
            late_layers  = hidden_states[L//2+1 : L+1]   (layers L/2+1 to L)
            c_conv       = Var(early) / (Var(late) + ε)

        High c_conv -> early layers more variable than late -> model unsettled.
        Converted to confidence: 1 / (1 + c_conv).

        Returns raw c_conv ratio (not converted) -- conversion happens in estimate().
        Returns 0.0 (highest confidence) on error (fail-safe).
        """
        try:
            encoder_outputs = self.model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
                return_dict=True,
            )
        except Exception as exc:
            logger.warning("c_conv: encoder forward pass failed with %s -- returning 0.0.", exc)
            return 0.0

        hidden_states = encoder_outputs.hidden_states
        # hidden_states: tuple of (L+1) tensors, each shape (batch, seq_len, hidden_dim)
        # Index 0 = embedding layer output; indices 1..L = transformer layer outputs
        L = len(hidden_states) - 1  # Number of encoder layers (12 for Flan-T5-Large)

        if L < 2:
            # Edge case: model has fewer than 2 layers -- c_conv is undefined.
            logger.warning("c_conv: only %d encoder layer(s) -- returning 0.0.", L)
            return 0.0

        # Split into early and late halves (excluding embedding layer at index 0).
        early_layers = hidden_states[1 : L // 2 + 1]   # Layers 1 to L/2
        late_layers  = hidden_states[L // 2 + 1 : L + 1]  # Layers L/2+1 to L

        if not early_layers or not late_layers:
            return 0.0

        # Stack and compute scalar variance across all elements.
        var_early = torch.var(torch.stack(early_layers)).item()
        var_late  = torch.var(torch.stack(late_layers)).item()

        eps = 1e-8
        c_conv = var_early / (var_late + eps)

        return float(c_conv)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _tokenize(self, query: str) -> dict:
        """Tokenize a query string and move tensors to device."""
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ------------------------------------------------------------------ #
    # Batch estimation (for calibration / evaluation harness)             #
    # ------------------------------------------------------------------ #

    def estimate_batch(self, queries: list[str]) -> list[PreRoutingConfidence]:
        """Estimate u_pre for a list of queries sequentially.

        Note: batched generation with output_scores=True requires careful
        token alignment (padding complicates log-prob indexing). Sequential
        is safer and the latency difference is negligible per-query at inference.

        Parameters
        ----------
        queries : list of str

        Returns
        -------
        list of PreRoutingConfidence, one per query.
        """
        return [self.estimate(q) for q in queries]
