"""
caem/verification/judge_interface.py
=====================================
NLIJudgeInterface — formal protocol for any CAEM verifier backend.

Purpose
-------
CAEM's pipeline is architecturally agnostic to the NLI backend; this
module defines the contract that any judge must implement so the
UnifiedVerifier can call it uniformly. It anchors the thesis claim:

    "CAEM admits modular extension via three swap points:
     (1) a domain-specific retrieval index,
     (2) a domain-appropriate retrieval encoder, and
     (3) a domain-appropriate NLI verifier.
     Empirical validation of these extensions is future work;
     the present thesis validates CAEM on the retrieval-augmented
     factoid QA panel where all three components are off-the-shelf
     (DPR + MPNet + MiniCheck)."

The thesis-shipped reference implementation uses MiniCheckJudge for
hypothesis ≤ 408 tokens and the frozen-Qwen FrozenQwenJudge (§qwen_judge)
for longer hypotheses, with AdaptiveNLIJudge (§adaptive_nli_judge) as
the dispatcher.

Any future judge (Llama3-NLI, Mistral-NLI, GPT-based, domain-tuned
NLI heads, etc.) can be swapped in by implementing this protocol.

Protocol contract
-----------------
Required methods:

  entail_prob(premise, hypothesis)
      Single-pair scalar entailment probability ∈ [0, 1].

  contradict_prob(premise, hypothesis)
      Single-pair scalar contradiction probability ∈ [0, 1].
      Semantics: P(premise refutes hypothesis) as distinct from
      P(premise does not support hypothesis). Backends that cannot
      distinguish these may return 0.0 (see MiniCheckJudge rationale).

  argmax_label(premise, hypothesis)
      Discrete NLI label ∈ {"ENTAIL", "NEUTRAL", "CONTRADICT"} or
      {0, 1, 2} depending on backend convention.

  batch_entail_prob(premises, hypotheses)
      Vectorised version — returns np.ndarray[float32] of shape (N,).

  batch_contradict_prob(premises, hypotheses)
      Vectorised contradiction probabilities, shape (N,).

  batch_argmax_label(premises, hypotheses)
      Vectorised labels, length N.

The batched forms are the hot path for CAEM's verifier; single-pair
forms are provided for convenience and debugging.

Existing implementations
------------------------
- caem.verification.minicheck.MiniCheckJudge  (short-hypothesis default)
- caem.verification.qwen_judge.FrozenQwenJudge  (long-hypothesis fallback)
- caem.verification.adaptive_nli_judge.AdaptiveNLIJudge  (dispatcher)
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class NLIJudgeInterface(Protocol):
    """Protocol any CAEM verifier backend must implement.

    Runtime-checkable: ``isinstance(obj, NLIJudgeInterface)`` verifies
    the required attributes exist (Python 3.8+ structural typing).
    """

    # ---- Single-pair scalar API ---------------------------------------------

    def entail_prob(self, premise: str, hypothesis: str) -> float: ...
    def contradict_prob(self, premise: str, hypothesis: str) -> float: ...
    def argmax_label(self, premise: str, hypothesis: str): ...

    # ---- Batched vectorised API ---------------------------------------------

    def batch_entail_prob(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> np.ndarray: ...
    def batch_contradict_prob(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> np.ndarray: ...
    def batch_argmax_label(
        self, premises: Sequence[str], hypotheses: Sequence[str],
    ) -> List: ...
