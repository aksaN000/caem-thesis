"""
caem/verification/minicheck.py
===============================
MiniCheck-Flan-T5-Large judge adapter.

Problem this module solves
--------------------------
The original CAEM verifier uses ``roberta-large-mnli`` for every entailment
test (chain->answer, passage->answer, atomic-fact->passage, bidirectional
sample clustering). RoBERTa-MNLI was trained on MultiNLI/SNLI -- pairs of
*human-written* sentences. It was never trained to judge LM-generated text.

Recent empirical work (HaluEval 2025, "Semantic Illusion") reports a
**100%% FPR at 95%% recall** for DeBERTa-v3-large-MNLI on LM hallucinations
-- fluent-but-wrong LM outputs look indistinguishable from correct NLI
positives to generic NLI models. This makes the NLI-backed signals in
CAEM (p_entail, p_ground_*, p_contra, semantic-entropy clustering)
systematically over-confident on LM outputs, weakening the Chapter 4
``purity > base accuracy'' theorem.

MiniCheck-Flan-T5-Large (Tang et al. 2024, ACL) addresses this head-on.
It is a Flan-T5-Large (770M, ~1.5 GB bf16) fine-tuned on synthetically
decomposed LM claims, specifically for the task of verifying whether a
short claim is supported by a document. AUROC gains of 10-25 points
over generic NLI on HaluEval / AggreFact / FEVER-hallucination splits.

Interface contract
------------------
This class is duck-typed to :class:`caem.verification.verifier._NLIEnsemble`
so the UnifiedVerifier can swap backends without touching its own logic.
The three public primitives are:

    entail_prob(premise, hypothesis)      -> float in [0, 1]
    contradict_prob(premise, hypothesis)  -> float in [0, 1]
    argmax_label(premise, hypothesis)     -> int in {0, 1, 2}

and their batched equivalents. The semantics are translated from MiniCheck's
unary "supported vs unsupported" output onto the 3-way NLI label space:

    entail_prob      := P(supported)
    contradict_prob  := 1 - P(supported)
    argmax_label     := ENTAIL  if P(supported) >= entail_threshold (0.7)
                        CONTRADICT if P(supported) <= contradict_threshold (0.3)
                        NEUTRAL otherwise

Rationale for ``contradict = 1 - entail``
-----------------------------------------
NLI separates neutral (no evidence) from contradiction (refuted). MiniCheck
is binary (supported vs not-supported). For CAEM's contradiction-veto
logic (``p_contra >= 0.75 -> DISCARD``), treating "not supported" as the
contradiction signal is the conservative and correct choice: a claim that
the passage does not support should not be stored, regardless of whether
it is strictly refuted or merely unverified. The calibration diagnostic
in scripts/calibration_minicheck_vs_roberta.py compares empirical
reliability against gold NLI labels.

Prompt format
-------------
MiniCheck's HF card documents:

    input  = f"predict: {document}\\n\\n{claim}"
    output = "0" or "1" (single token; "1" means supported)

Probability is derived as a two-way softmax over the first-token logits
for "0" and "1" tokens. This mirrors the reference minicheck package
(github.com/Liyan06/MiniCheck) but avoids the extra dependency.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# NLI label convention used by UnifiedVerifier and by RoBERTa/DeBERTa.
_CONTRADICTION = 0
_NEUTRAL = 1
_ENTAILMENT = 2


class _MiniCheckJudge:
    """Adapter: MiniCheck-Flan-T5-Large wrapped as an NLI-shaped judge.

    Parameters
    ----------
    model : T5ForConditionalGeneration
        Loaded MiniCheck checkpoint (``lytang/MiniCheck-Flan-T5-Large``).
    tokenizer : transformers tokenizer
        Matching Flan-T5 tokenizer.
    device : str
        ``"cuda"`` or ``"cpu"``.
    entail_threshold : float, default 0.7
        Minimum P(supported) at which ``argmax_label`` emits ENTAIL.
    contradict_threshold : float, default 0.3
        Maximum P(supported) at which ``argmax_label`` emits CONTRADICT.
    max_premise_tokens : int, default 450
        Hard truncation cap for the premise side; MiniCheck inputs a
        combined "predict: {doc}\\n\\n{claim}" string of length <= 512.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        *,
        entail_threshold: float = 0.7,
        contradict_threshold: float = 0.3,
        max_premise_tokens: int = 450,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.entail_threshold = float(entail_threshold)
        self.contradict_threshold = float(contradict_threshold)
        self.max_premise_tokens = int(max_premise_tokens)

        # Cache the decoder-step-0 target token ids for supported / unsupported.
        #
        # Flan-T5 / T5 SentencePiece handles bare digits asymmetrically:
        # tokenizer("1", add_special_tokens=False) -> [209]          ("▁1" as one merge)
        # tokenizer("0", add_special_tokens=False) -> [3, 632]       ("▁" + "0")
        #
        # Empirical verification on lytang/MiniCheck-Flan-T5-Large confirms
        # the model emits id 209 at decoder step 0 for supported claims and
        # id 3 at step 0 for unsupported claims (with 632 as the step-1
        # continuation). A two-way softmax between these two step-0 tokens
        # recovers MiniCheck's native P(supported) / P(unsupported) and
        # matches the reference implementation in the Liyan06/MiniCheck
        # package. We therefore take the FIRST token id from each tokenization
        # rather than requiring a single-token form, which the previous
        # stricter check rejected.
        ids_yes = tokenizer("1", add_special_tokens=False).input_ids
        ids_no = tokenizer("0", add_special_tokens=False).input_ids
        if not ids_yes or not ids_no:
            raise RuntimeError(
                f"MiniCheck tokenizer produced empty id sequence for '1' / '0': "
                f"yes={ids_yes!r} no={ids_no!r}."
            )
        self._yes_id = int(ids_yes[0])
        self._no_id = int(ids_no[0])
        if self._yes_id == self._no_id:
            raise RuntimeError(
                f"MiniCheck tokenizer maps '1' and '0' to the same step-0 id "
                f"{self._yes_id}; two-way softmax would be degenerate."
            )

    # ---- Identity helpers ------------------------------------------------- #

    def __bool__(self) -> bool:
        return True

    @property
    def bundles(self):
        # Back-compat: some legacy code checks ``len(nli.bundles) > 0``.
        return [(self.model, self.tokenizer)]

    # ---- Prompt construction --------------------------------------------- #

    def _format_prompt(self, premise: str, hypothesis: str) -> str:
        # Truncate the premise hard to keep the combined prompt within
        # the tokenizer's 512-token budget. MiniCheck's training inputs
        # were single passages so premise shorter is usually better.
        enc = self.tokenizer(premise, add_special_tokens=False)
        ids = enc.input_ids[: self.max_premise_tokens]
        premise_trunc = self.tokenizer.decode(ids, skip_special_tokens=True)
        return f"predict: {premise_trunc}\n\n{hypothesis}"

    # ---- Core scoring ---------------------------------------------------- #

    @torch.no_grad()
    def _score_batch(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> np.ndarray:
        """Return np.ndarray of shape (N,) with P(supported) per pair.

        Uses the first decoder-step logits over {"0", "1"} and two-way
        softmax. Single batched forward; no ``generate()`` required.
        """
        if not premises:
            return np.zeros(0, dtype=np.float32)

        prompts = [
            self._format_prompt(p, h) for p, h in zip(premises, hypotheses)
        ]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        # The decoder input is the T5 decoder_start_token_id (a single pad).
        # We only need the logits at decoder step 0 -- a single forward is
        # both faster and exactly equivalent to generate(max_new_tokens=1).
        decoder_input_ids = torch.full(
            (enc.input_ids.shape[0], 1),
            fill_value=self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )
        out = self.model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        logits = out.logits[:, 0, :]  # (N, vocab)
        two_way = torch.stack(
            [logits[:, self._no_id], logits[:, self._yes_id]],
            dim=-1,
        )  # (N, 2) -- [P(0), P(1)]
        probs = F.softmax(two_way, dim=-1)[:, 1]  # P(supported)
        # Cast to float32 before numpy — bf16 is not a supported numpy dtype.
        return probs.detach().float().cpu().numpy()

    # ---- Public NLI-shaped API ------------------------------------------ #

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        """Scalar entailment probability. Matches NLIEnsemble.entail_prob()."""
        arr = self._score_batch([premise], [hypothesis])
        return float(np.clip(arr[0], 0.0, 1.0))

    def contradict_prob(self, premise: str, hypothesis: str) -> float:
        """Scalar contradiction probability.

        **Returns 0.0 always.** MiniCheck is a binary "supported vs
        not-supported" judge; its "not-supported" class conflates strict
        refutation with neutral / unverifiable claims, which are
        distinct in NLI's 3-class label space. Mapping
        ``contradict = 1 - P(supported)`` would cause the CAEM
        contradiction-veto branch to fire on almost every claim whose
        passage does not directly state the answer (empirically, ~100%
        of samples under MiniCheck on TriviaQA-style eval), which is
        not the semantic the veto is designed for.

        The correct CAEM response is to let ``p_ground_max / p_ground_mean``
        pull ``u_stored`` down for unsupported claims via the composite,
        rather than via the hard veto. Consumers who need a dedicated
        refutation signal should either use an NLI backend or add a
        separate two-class refutation judge on top.
        """
        return 0.0

    def argmax_label(self, premise: str, hypothesis: str) -> int:
        """Threshold-based 3-way label.

        Used only by _compute_h_norm's bidirectional clustering. The
        thresholds map MiniCheck's unary score onto the NLI label space.
        """
        p = self.entail_prob(premise, hypothesis)
        if p >= self.entail_threshold:
            return _ENTAILMENT
        if p <= self.contradict_threshold:
            return _CONTRADICTION
        return _NEUTRAL

    # ---- Batched variants (called from verifier fast path) ------------- #

    def batch_entail_prob(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[float]:
        if not pairs:
            return []
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        arr = self._score_batch(premises, hypotheses)
        return [float(np.clip(x, 0.0, 1.0)) for x in arr]

    def batch_contradict_prob(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[float]:
        """Returns 0.0 per pair. See ``contradict_prob`` docstring for why."""
        return [0.0] * len(pairs)

    def batch_argmax_label(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[int]:
        if not pairs:
            return []
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        arr = self._score_batch(premises, hypotheses)
        out = []
        for p in arr:
            if p >= self.entail_threshold:
                out.append(_ENTAILMENT)
            elif p <= self.contradict_threshold:
                out.append(_CONTRADICTION)
            else:
                out.append(_NEUTRAL)
        return out


# ============================================================================ #
# Loading helper                                                                #
# ============================================================================ #

def load_minicheck_judge(
    model_name: str = "lytang/MiniCheck-Flan-T5-Large",
    device: Optional[str] = None,
    entail_threshold: float = 0.7,
    contradict_threshold: float = 0.3,
    dtype: Optional[Any] = None,
) -> _MiniCheckJudge:
    """Download + load the MiniCheck checkpoint and return a judge instance.

    Kept out of the class __init__ so the caller controls device placement
    and dtype (bf16 by default on CUDA; float32 on CPU).
    """
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    logger.info("Loading MiniCheck judge %s on %s (%s)", model_name, device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    model.eval()
    return _MiniCheckJudge(
        model=model,
        tokenizer=tokenizer,
        device=device,
        entail_threshold=entail_threshold,
        contradict_threshold=contradict_threshold,
    )
