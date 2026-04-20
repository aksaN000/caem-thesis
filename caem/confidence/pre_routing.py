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

Signal 2 -- C_conv (Internal Convergence, Nandakishor 2025)
    Measures query-representation stability via the variance ratio of early
    vs. late transformer-layer hidden states on the query tokens.

    Backbone dispatch (Goal 1, Branch C):
      - Decoder-only (Qwen/Gemma/Llama): run ``self.model(input_ids=...)``
        on the query tokens directly (no generation) and extract the full
        hidden-state stack. This matches the original Nandakishor (2025)
        formulation — the decoder-only path is canonical.
      - Encoder-decoder (Flan-T5): run ``self.model.encoder(...)`` and extract
        encoder hidden states. The CAEM adaptation because Flan-T5 has no
        unified decoder stack to run without generation; the encoder produces
        the stable query representation we want to measure.

    Intuition: if early layers disagree more than late layers, the model has
    not settled on a stable representation of the query -- it is internally
    "confused" about what is being asked. High variance ratio = low
    confidence.

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
        raw_u_pre = (
            cfg.u_pre_token_weight * u_token
            + cfg.u_pre_cconv_weight * confidence_from_cconv
        )
        raw_u_pre = float(np.clip(raw_u_pre, 0.0, 1.0))
        u_pre = self._apply_temperature_scaling(raw_u_pre)

        logger.debug(
            "u_pre estimate | query='%s...' | u_token=%.4f | c_conv=%.4f | "
            "conf_cconv=%.4f | raw_u_pre=%.4f | T=%.4f | u_pre=%.4f | safe=%s",
            query[:50], u_token, c_conv, confidence_from_cconv,
            raw_u_pre, cfg.temperature_scalar, u_pre,
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

        Returns float in [0, 1]. Returns 0.0 on failure (fail-safe to Tier 3).
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

        scores = outputs.scores       # Tuple of logits tensors, one per generated step
        sequences = outputs.sequences # (1, seq_len) including prompt tokens

        if not scores:
            logger.warning("u_token: no scores returned -- returning 0.0.")
            return 0.0

        # Compute generated-token offset architecture-agnostically.
        #
        # Encoder-decoder (Flan-T5):
        #   sequences = [decoder_start, tok_1, ..., tok_n]
        #   sequences.shape[1] = 1 + n_gen;  gen_start = 1
        #
        # Decoder-only (Qwen/Gemma/Llama):
        #   sequences = [input_0, ..., input_{L_input-1}, tok_1, ..., tok_n]
        #   sequences.shape[1] = L_input + n_gen;  gen_start = L_input
        #
        # In both cases, the last ``len(scores)`` tokens of ``sequences`` are
        # the generated tokens. Taking ``gen_start = sequences.shape[1] -
        # len(scores)`` works for both architectures.
        n_gen = len(scores)
        gen_start = int(sequences.shape[1]) - n_gen

        log_probs = []
        for t, step_scores in enumerate(scores):
            # step_scores: (1, vocab_size) logits
            log_softmax = torch.nn.functional.log_softmax(step_scores[0], dim=-1)
            token_id = sequences[0, gen_start + t].item()
            if token_id == self.tokenizer.eos_token_id:
                break   # Stop at EOS -- don't include it in the mean
            log_prob = log_softmax[token_id].item()
            log_probs.append(log_prob)

        if not log_probs:
            logger.warning("u_token: no non-EOS tokens scored -- returning 0.0.")
            return 0.0

        # Geometric mean: exp(mean(log_probs))
        u_token = math.exp(sum(log_probs) / len(log_probs))
        return float(np.clip(u_token, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Signal 2 -- C_conv (Internal Convergence)                            #
    # ------------------------------------------------------------------ #

    def _is_encoder_decoder(self) -> bool:
        """Detect architecture once, from model config.

        Returns True for Flan-T5 family; False for Qwen/Gemma/Llama/Phi/Mistral
        and other decoder-only families. Used to dispatch the c_conv hidden-
        state source (encoder-only vs full decoder stack).

        Note on torch.compile wrapping: ``OptimizedModule`` forwards all
        attribute access (including ``.config``) to the wrapped module, so
        we do NOT need to reach for ``._orig_mod`` manually — accessing
        ``self.model.config.is_encoder_decoder`` works through the wrapper.
        """
        cfg = getattr(self.model, "config", None)
        return bool(getattr(cfg, "is_encoder_decoder", False))

    def _compute_c_conv(self, inputs: dict) -> float:
        """Layer-variance-ratio query-stability signal; dispatches on architecture.

        Encoder-decoder (Flan-T5) path: runs ``self.model.encoder(...)`` over
        the query tokens; uses encoder hidden states. CAEM-specific adaptation
        because T5 has no unified stack to run without generation.

        Decoder-only (Qwen/Gemma/Llama) path: runs ``self.model(...)`` over
        the query tokens (forward pass only, no generation) and uses the full
        transformer-stack hidden states. This is the original Nandakishor
        (2025) formulation; cleaner than the T5 adaptation.

        Both paths apply the identical variance-ratio computation afterward;
        only the forward-pass source differs. Returns the raw c_conv ratio;
        conversion to confidence happens in ``estimate()``.

        Returns 0.0 (highest confidence, fail-safe) on any forward-pass error.
        """
        try:
            if self._is_encoder_decoder():
                outputs = self.model.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                # Decoder-only: direct forward pass on the query tokens. No
                # generation; we only need the hidden-state stack.
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=True,
                    return_dict=True,
                )
        except Exception as exc:
            logger.warning(
                "c_conv: forward pass failed with %s -- returning 0.0.",
                exc,
            )
            return 0.0

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            logger.warning(
                "c_conv: model output has no hidden_states attr -- returning 0.0. "
                "The model may not support output_hidden_states=True on this path."
            )
            return 0.0

        # hidden_states: tuple of (L+1) tensors, shape (batch, seq_len, hidden_dim)
        # Index 0 = embedding layer output; indices 1..L = transformer layer outputs.
        # Flan-T5-Large: L=24 encoder layers. Qwen2.5-3B: L=36 decoder layers.
        L = len(hidden_states) - 1

        if L < 2:
            logger.warning("c_conv: only %d layer(s) in hidden_states -- returning 0.0.", L)
            return 0.0

        # Split into early and late halves (excluding embedding layer at index 0).
        early_layers = hidden_states[1 : L // 2 + 1]
        late_layers  = hidden_states[L // 2 + 1 : L + 1]

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
        """Tokenize a query string and move tensors to device.

        Architecture-aware wrapping (Goal 1, Branch C):
        - Encoder-decoder (Flan-T5): pass query directly to tokenizer. T5 was
          pretrained on raw text + FLAN instruction tuning; the encoder accepts
          bare queries without template wrapping.
        - Decoder-only (Qwen/Gemma/Llama/Phi): wrap query in a minimal ChatML
          user turn with generation prompt via ``apply_chat_template``. Without
          this, instruction-tuned decoder-only models flail on raw text —
          predictions scatter across the ~151k vocab, making per-token
          probabilities tiny and u_token collapse to ~1e-9 (empirically
          confirmed 2026-04-22 on Qwen-2.5-3B). Wrapping restores the signal's
          intended semantic: model confidence when beginning an assistant turn
          answering the user's question.

        The ChatML envelope adds ~20 tokens of fixed structure around the
        query; c_conv's variance ratio is not materially distorted because
        the envelope is the same for every query and washes out in the
        early/late layer comparison.
        """
        text = self._wrap_query_for_backbone(query)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _wrap_query_for_backbone(self, query: str) -> str:
        """Wrap the raw query for the backbone's expected input convention.

        Returns ``query`` unchanged for encoder-decoder models. Returns a
        ChatML-formatted user turn + generation prompt for decoder-only
        models via ``tokenizer.apply_chat_template(add_generation_prompt=True)``
        when available. Falls back to a manual ChatML concatenation if the
        tokenizer lacks ``apply_chat_template`` (rare; most modern
        instruction-tuned tokenizers ship with a chat template).
        """
        if self._is_encoder_decoder():
            return query

        # Decoder-only: wrap in minimal ChatML
        messages = [{"role": "user", "content": query}]
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                logger.debug(
                    "apply_chat_template failed (%s); falling back to manual "
                    "ChatML wrap.", exc,
                )
        # Manual ChatML fallback (Qwen/Llama/Gemma all share this syntax)
        return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

    def _apply_temperature_scaling(self, prob: float) -> float:
        """Apply sigmoid(logit(prob)/T) for calibrated confidence.

        Temperature scaling is calibrated offline and stored in config.
        T=1.0 leaves values unchanged.
        """
        T = float(getattr(self.config, "temperature_scalar", 1.0) or 1.0)
        p = float(np.clip(prob, 0.0, 1.0))
        if T <= 0.0 or abs(T - 1.0) < 1e-8:
            return p

        p = float(np.clip(p, 1e-7, 1.0 - 1e-7))
        logit = math.log(p / (1.0 - p))
        scaled = float(np.clip(logit / T, -60.0, 60.0))
        calibrated = 1.0 / (1.0 + math.exp(-scaled))
        return float(np.clip(calibrated, 0.0, 1.0))

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
