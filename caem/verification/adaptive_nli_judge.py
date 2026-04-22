"""
caem/verification/adaptive_nli_judge.py
========================================
AdaptiveNLIJudge — length-aware dispatcher between MiniCheck and frozen
Qwen-judge.

Purpose
-------
Path B of the verifier extension (thesis §5.x). Routes each (premise,
hypothesis) pair to one of two judges based on hypothesis length:

  - hypothesis ≤ 408 MC-tokens  -> MiniCheckJudge (fast, specialised NLI)
  - hypothesis > 408 MC-tokens  -> FrozenQwenJudge (long-context fallback)

Why 408
-------
MiniCheck's encoder caps at 512 tokens. Reserved budget:
  OVERHEAD (prefix + separator)        = 4 tokens
  MIN_PREMISE (minimum context)        = 100 tokens
  512 - 4 - 100 = 408                  = max hypothesis that fits

Above this, MiniCheck's adaptive truncation would still cut the
hypothesis tail. Qwen-judge has 131k context and handles any
realistic hypothesis.

Calibration
-----------
Qwen-judge output is Platt-scaled against MiniCheck on a 500-sample
overlap fold (scripts/calibrate_qwen_judge.py). The Platt (a, b)
parameters are baked into FrozenQwenJudge at construction, so all
outputs from both judges are on a comparable scale for u_stored
composition.

Fallback behaviour
------------------
If qwen_judge is None (e.g., during dev or if calibration failed to
converge), all samples go to MiniCheck — including long ones that
will trigger HF's 512-token truncation. A warning is logged per
such case so the methodology can report truncation incidence.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveNLIJudge:
    """Length-based dispatcher conforming to NLIJudgeInterface.

    Parameters
    ----------
    minicheck_judge : NLIJudgeInterface
        Primary judge — MiniCheckJudge, handles hypothesis ≤ 408 tokens.
    qwen_judge : NLIJudgeInterface | None
        Fallback judge — FrozenQwenJudge with Platt calibration, handles
        long hypotheses. If None, all samples route to MiniCheck (may
        incur truncation on long hypothesis — a warning is logged).
    hypothesis_cap_mc_tokens : int
        Threshold in MiniCheck's tokenization above which samples route
        to Qwen-judge. Default 408 = 512 - OVERHEAD(4) - MIN_PREMISE(100).
    mc_tokenizer : tokenizer
        MiniCheck tokenizer used for length measurement. Must match
        minicheck_judge.tokenizer.
    """

    # Same constants as MiniCheckJudge to keep the threshold consistent.
    _MC_MAX_INPUT_TOKENS: int = 512
    _MC_OVERHEAD_TOKENS: int = 4
    _MC_MIN_PREMISE_TOKENS: int = 100
    # Default routing threshold = 408.
    _DEFAULT_HYP_CAP: int = (
        _MC_MAX_INPUT_TOKENS - _MC_OVERHEAD_TOKENS - _MC_MIN_PREMISE_TOKENS
    )

    def __init__(
        self,
        minicheck_judge: Any,
        qwen_judge: Optional[Any],
        mc_tokenizer: Any,
        *,
        hypothesis_cap_mc_tokens: Optional[int] = None,
    ) -> None:
        self.mc = minicheck_judge
        self.qwen = qwen_judge
        self.mc_tokenizer = mc_tokenizer
        self.hyp_cap = int(hypothesis_cap_mc_tokens or self._DEFAULT_HYP_CAP)

        if self.qwen is None:
            logger.warning(
                "AdaptiveNLIJudge: qwen_judge=None; long-hypothesis samples "
                "will route to MiniCheck (may incur 512-token truncation). "
                "Run scripts/calibrate_qwen_judge.py before Step 7 main."
            )

        # Routing stats (per-instance; reset by reset_stats())
        self._stats = {"mc_calls": 0, "qwen_calls": 0, "qwen_fallback_to_mc": 0}

    # ---- Stats -------------------------------------------------------------

    def reset_stats(self) -> None:
        self._stats = {"mc_calls": 0, "qwen_calls": 0, "qwen_fallback_to_mc": 0}

    def stats(self) -> dict:
        total = self._stats["mc_calls"] + self._stats["qwen_calls"]
        mc_frac = self._stats["mc_calls"] / max(total, 1)
        qwen_frac = self._stats["qwen_calls"] / max(total, 1)
        return {
            "total_calls": total,
            "mc_calls": self._stats["mc_calls"],
            "qwen_calls": self._stats["qwen_calls"],
            "qwen_fallback_to_mc": self._stats["qwen_fallback_to_mc"],
            "mc_fraction": round(mc_frac, 4),
            "qwen_fraction": round(qwen_frac, 4),
        }

    # ---- Routing -----------------------------------------------------------

    def _partition(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> "tuple[List[int], List[int]]":
        """Return (mc_indices, qwen_indices) based on hypothesis length.

        If qwen_judge is None, returns (all_indices, []) so every sample
        goes to MiniCheck (with downstream truncation if hypothesis is long).
        """
        if self.qwen is None:
            return list(range(len(premises))), []

        mc_idx: List[int] = []
        qwen_idx: List[int] = []
        for i, h in enumerate(hypotheses):
            hyp_len = len(self.mc_tokenizer(h, add_special_tokens=False).input_ids)
            if hyp_len <= self.hyp_cap:
                mc_idx.append(i)
            else:
                qwen_idx.append(i)
        return mc_idx, qwen_idx

    # ---- NLIJudgeInterface batched API -------------------------------------

    def batch_entail_prob(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> np.ndarray:
        n = len(premises)
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        mc_idx, qwen_idx = self._partition(premises, hypotheses)
        results = np.zeros(n, dtype=np.float32)

        if mc_idx:
            mc_results = self.mc.batch_entail_prob(
                [premises[i] for i in mc_idx],
                [hypotheses[i] for i in mc_idx],
            )
            for idx, p in zip(mc_idx, mc_results):
                results[idx] = p
            self._stats["mc_calls"] += len(mc_idx)

        if qwen_idx:
            try:
                qwen_results = self.qwen.batch_entail_prob(
                    [premises[i] for i in qwen_idx],
                    [hypotheses[i] for i in qwen_idx],
                )
                for idx, p in zip(qwen_idx, qwen_results):
                    results[idx] = p
                self._stats["qwen_calls"] += len(qwen_idx)
            except Exception as exc:
                logger.warning(
                    "AdaptiveNLIJudge: Qwen-judge failed on %d samples (%s); "
                    "falling back to MiniCheck (will truncate).", len(qwen_idx), exc,
                )
                mc_fallback_results = self.mc.batch_entail_prob(
                    [premises[i] for i in qwen_idx],
                    [hypotheses[i] for i in qwen_idx],
                )
                for idx, p in zip(qwen_idx, mc_fallback_results):
                    results[idx] = p
                self._stats["qwen_fallback_to_mc"] += len(qwen_idx)

        return results

    def batch_contradict_prob(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> np.ndarray:
        # Both MC and Qwen judges return 0.0 for contradiction (neither
        # distinguishes strict refutation from non-support). See their
        # docstrings for the rationale — CAEM uses p_ground_* as the
        # low-support signal instead of a hard contradiction veto.
        return np.zeros(len(premises), dtype=np.float32)

    def batch_argmax_label(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> List:
        # Same routing as entail; return per-judge labels preserving order.
        n = len(premises)
        if n == 0:
            return []

        mc_idx, qwen_idx = self._partition(premises, hypotheses)
        results: List = [None] * n

        if mc_idx:
            mc_labels = self.mc.batch_argmax_label(
                [premises[i] for i in mc_idx],
                [hypotheses[i] for i in mc_idx],
            )
            for idx, lbl in zip(mc_idx, mc_labels):
                results[idx] = lbl

        if qwen_idx and self.qwen is not None:
            qwen_labels = self.qwen.batch_argmax_label(
                [premises[i] for i in qwen_idx],
                [hypotheses[i] for i in qwen_idx],
            )
            for idx, lbl in zip(qwen_idx, qwen_labels):
                results[idx] = lbl

        return results

    # ---- Single-pair convenience API ---------------------------------------

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        arr = self.batch_entail_prob([premise], [hypothesis])
        return float(arr[0])

    def contradict_prob(self, premise: str, hypothesis: str) -> float:
        return 0.0

    def argmax_label(self, premise: str, hypothesis: str):
        labels = self.batch_argmax_label([premise], [hypothesis])
        return labels[0]

    # ---- Introspection -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AdaptiveNLIJudge(hyp_cap={self.hyp_cap}, "
            f"mc={type(self.mc).__name__}, "
            f"qwen={type(self.qwen).__name__ if self.qwen else None})"
        )
