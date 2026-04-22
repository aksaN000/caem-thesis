"""
caem/verification/qwen_judge.py
================================
Frozen Qwen-3B as an NLI judge for long-hypothesis samples.

Purpose
-------
MiniCheck-Flan-T5-Large caps at 512-token encoder input. When the
(premise, hypothesis) input exceeds this budget, MiniCheck must
truncate. Path B routes such samples to a frozen Qwen-3B judge that
uses constrained-decoding yes/no NLI, extracting P(entail) from the
first decoder token's softmax over the {"yes", "no"} tokens.

Frozen-ness
-----------
This judge MUST use the BASE Qwen-3B weights (before any SIL training)
and MUST remain frozen across all cycles. Using the evolving SIL-trained
Qwen as self-judge would introduce confirmation-loop bias:

  Cycle N generator trained on verifier-approved data from N-1
  -> Cycle N Qwen recognises its own style
  -> Cycle N Qwen-as-judge rates its own output highly
  -> Cycle N+1 training data biased toward "what Qwen-as-judge likes"
  -> compounding over cycles, drift into idiosyncratic distribution

CAEM's purity theorem (Claim 2) requires the verifier be a STABLE
independent oracle. This class enforces frozen-ness by loading Qwen
once at construction and refusing to accept state_dict updates.

Calibration
-----------
Qwen-as-judge's P(yes) distribution differs from MiniCheck's
P(supported) — Qwen is a general chat model fine-tuned for
helpfulness, tends to say "yes" more often. Path B's calibration
step (scripts/calibrate_qwen_judge.py) fits Platt scaling (a, b) on
a 500-sample overlap fold against MiniCheck so P_entail values from
both paths are directly comparable in the u_stored composite.

References
----------
- LLM-as-judge: Zheng et al. 2023 "Judging LLM-as-a-Judge"
- Confirmation bias in self-training: Chen et al. 2024
- Platt scaling: Platt 1999, for probability calibration
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FrozenQwenJudge:
    """NLI judge backed by frozen Qwen-3B with constrained yes/no decoding.

    Duck-typed to :class:`NLIJudgeInterface`. Reuses a Qwen model and
    tokenizer that the caller has already loaded and frozen.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Qwen-3B (or compatible causal LM) in eval() mode, frozen weights.
    tokenizer : transformers.PreTrainedTokenizer
        Matching Qwen tokenizer.
    device : str
        "cuda" or "cpu".
    platt_a : float
        Platt scaling slope (from the MC-vs-Qwen calibration fit). 1.0 =
        no calibration. Set by the calibration script and passed in here.
    platt_b : float
        Platt scaling bias. 0.0 = no calibration.
    entail_threshold : float
        P(entail) threshold for ENTAIL label (default 0.5 after calibration).
    contradict_threshold : float
        P(entail) threshold below which label=CONTRADICT (default 0.3).
    max_context_tokens : int
        Maximum input length for Qwen. Default 8192 — well under Qwen's
        131k limit but covers any realistic (premise, hypothesis) pair
        in Phase 1a. Avoids pathological very-long pairs that would
        bloat KV cache.

    Prompt format
    -------------
    Single-turn, instruction style:

        "Given the context, determine if the claim is supported.
         Context: {premise}
         Claim: {hypothesis}
         Answer with only 'yes' or 'no'.
         Answer:"

    At generation step 0, we take logits over the full vocabulary and
    extract the positions corresponding to the token ids for " yes"
    and " no" (with leading space, since that's how the tokenizer
    encodes a fresh word). Two-way softmax gives P(yes) / (P(yes) + P(no)).
    """

    # Canonical prompt template (do not change across cycles — results
    # depend on exact wording; any change invalidates Platt calibration).
    _PROMPT_TEMPLATE = (
        "Given the context, determine if the claim is supported.\n"
        "Context: {premise}\n"
        "Claim: {hypothesis}\n"
        "Answer with only 'yes' or 'no'.\n"
        "Answer:"
    )

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        *,
        platt_a: float = 1.0,
        platt_b: float = 0.0,
        entail_threshold: float = 0.5,
        contradict_threshold: float = 0.3,
        max_context_tokens: int = 8192,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.platt_a = float(platt_a)
        self.platt_b = float(platt_b)
        self.entail_threshold = float(entail_threshold)
        self.contradict_threshold = float(contradict_threshold)
        self.max_context_tokens = int(max_context_tokens)

        # Freeze: explicit eval mode + no_grad at call time.
        # The model itself is shared with the generator; we rely on
        # the caller to NOT mutate it during SIL cycles. The verifier
        # is always called under torch.no_grad() so no grad flow anyway.
        if hasattr(self.model, "eval"):
            self.model.eval()

        # Cache the yes / no token ids once.
        self._yes_id = self._token_id_for(" yes")
        self._no_id = self._token_id_for(" no")
        if self._yes_id == self._no_id:
            raise RuntimeError(
                f"Qwen tokenizer collision: ' yes' and ' no' map to same id "
                f"{self._yes_id}; two-way softmax would be degenerate."
            )
        logger.info(
            "FrozenQwenJudge: yes_id=%d no_id=%d platt=(a=%.4f, b=%.4f) max_ctx=%d",
            self._yes_id, self._no_id, self.platt_a, self.platt_b,
            self.max_context_tokens,
        )

    # ---- Helpers ------------------------------------------------------------

    def _token_id_for(self, text: str) -> int:
        """Get the single-token id for the given text (raises if multi-token)."""
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        # Some tokenizers add leading space tokens differently; take first id
        # which is the one decoder sees as the "yes/no" answer.
        if not ids:
            raise RuntimeError(f"Qwen tokenizer returned empty for {text!r}")
        return int(ids[0])

    def _apply_platt(self, p: np.ndarray) -> np.ndarray:
        """Apply Platt scaling: logit(p_cal) = a * logit(p) + b."""
        eps = 1e-6
        p = np.clip(p, eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p))
        logit_cal = self.platt_a * logit + self.platt_b
        p_cal = 1.0 / (1.0 + np.exp(-logit_cal))
        return p_cal.astype(np.float32)

    def _build_prompts(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> List[str]:
        prompts: List[str] = []
        for p, h in zip(premises, hypotheses):
            prompts.append(self._PROMPT_TEMPLATE.format(premise=p, hypothesis=h))
        return prompts

    @torch.no_grad()
    def _score_batch(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> np.ndarray:
        """Return np.ndarray (N,) of P(entail)_calibrated per pair."""
        if not premises:
            return np.zeros(0, dtype=np.float32)

        prompts = self._build_prompts(premises, hypotheses)

        # Left-padding required for decoder-only causal LMs so the last
        # prompt position is always the generation start (causal attention
        # makes right-pad tail inert).
        original_side = getattr(self.tokenizer, "padding_side", None)
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_tokens,
                padding=True,
            )
        finally:
            if original_side is not None:
                self.tokenizer.padding_side = original_side

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Single forward pass; read logits at the last (non-pad) position.
        # With left-padding, the last position is always the prompt's final
        # token, and the model's next-token prediction is at the last index.
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # out.logits shape = (N, seq_len, vocab). Next-token logit is at
        # position -1 of each row (since left-padded, position -1 is the
        # last real prompt token).
        last_logits = out.logits[:, -1, :]  # (N, vocab)

        # Two-way softmax on yes / no token ids.
        two_way = torch.stack(
            [last_logits[:, self._no_id], last_logits[:, self._yes_id]],
            dim=-1,
        )  # (N, 2) — [logit_no, logit_yes]
        probs = F.softmax(two_way, dim=-1)[:, 1]  # P(yes)
        p_raw = probs.detach().float().cpu().numpy()

        # Apply Platt calibration to shift Qwen's P(yes) scale onto
        # MiniCheck's P(supported) scale, per the fitted (a, b) parameters.
        return self._apply_platt(p_raw)

    # ---- NLIJudgeInterface public API --------------------------------------

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        arr = self._score_batch([premise], [hypothesis])
        return float(np.clip(arr[0], 0.0, 1.0))

    def contradict_prob(self, premise: str, hypothesis: str) -> float:
        """P(contradiction).

        Symmetric to MiniCheckJudge: Qwen yes/no distinguishes supported
        vs not-supported, not supported vs refuted. Returns 0.0 here to
        preserve the same hard-veto semantics across both judges.
        Consumers needing a dedicated refutation signal should add a
        separate refutation judge on top of either backend.
        """
        return 0.0

    def argmax_label(self, premise: str, hypothesis: str):
        p = self.entail_prob(premise, hypothesis)
        if p >= self.entail_threshold:
            return "ENTAIL"
        if p <= self.contradict_threshold:
            return "CONTRADICT"
        return "NEUTRAL"

    def batch_entail_prob(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> np.ndarray:
        return self._score_batch(premises, hypotheses)

    def batch_contradict_prob(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> np.ndarray:
        # See rationale in contradict_prob(); this judge cannot distinguish
        # strict contradiction from mere non-support.
        return np.zeros(len(premises), dtype=np.float32)

    def batch_argmax_label(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> List[str]:
        probs = self._score_batch(premises, hypotheses)
        labels: List[str] = []
        for p in probs:
            if p >= self.entail_threshold:
                labels.append("ENTAIL")
            elif p <= self.contradict_threshold:
                labels.append("CONTRADICT")
            else:
                labels.append("NEUTRAL")
        return labels

    # ---- Introspection -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FrozenQwenJudge(platt_a={self.platt_a:.4f}, "
            f"platt_b={self.platt_b:.4f}, yes={self._yes_id}, no={self._no_id})"
        )
