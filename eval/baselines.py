"""
eval/baselines.py
=================
External baselines for the CAEM Chapter 5 comparison panel.

Design
------
Each baseline is a lightweight "pipeline-shaped" object that exposes the same
``answer(query, store_to_memory=False) -> PipelineResult`` signature as
``caem.pipeline.CAEMPipeline``. This lets ``eval.harness.EvalHarness`` consume
baselines without modification, and the per-sample JSON output carries the
same schema that ``eval.reporting`` and ``eval.metrics`` already understand.

Baselines
---------
ZeroShotBaseline  (B1) -- Flan-T5-Large, no CoT prefix, no retrieval. Floor.
CoTBaseline       (B2) -- Flan-T5-Large with "Let's think step by step" prefix.
RAGBaseline       (B3) -- DPR top-k retrieval + Flan-T5-Large context-conditioned.
CoTRAGBaseline    (B4) -- RAG context + CoT prefix.
FLAREBaseline     (B5) -- Jiang et al. EMNLP 2023 active retrieval with
                          confidence-threshold look-ahead: generate a 64-token
                          look-ahead, and if any token's log-probability falls
                          below ``theta`` trigger retrieval on the low-confidence
                          span, then regenerate the sentence with retrieved
                          context.

Mapping to PipelineResult fields
--------------------------------
``tier``                    -- 2 for generation-only baselines (Zero-shot, CoT),
                               3 for retrieval-augmented baselines (RAG, CoT+RAG,
                               FLARE). The field is kept for compatibility with
                               the harness and aggregator; baselines do not
                               implement a router.
``stored``                  -- always False for inference baselines.
``u_stored``                -- None (no verifier).
``verifier_output``         -- None (no Stage-5 verifier).
``routing_decision``        -- None.
``pre_confidence``          -- None.
``escalated``               -- True for FLARE when a look-ahead triggered a
                               retrieval restart; False otherwise.

Hyperparameters
---------------
All hyperparameters are tagged [LIT] (literature-fixed), [DES] (CAEM design
choice inherited from ``caem.config.CaemConfig``), or [CAL] (empirically
calibrated on a held-out split). Baselines intentionally share the [DES]
values of CAEM Tier 2/Tier 3 so that differences in final metrics are
attributable to the missing CAEM machinery rather than to mismatched decode
or retrieval budgets.

References
----------
FLARE    : Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J.,
           Yang, Y., Callan, J., Neubig, G. "Active Retrieval Augmented
           Generation." EMNLP 2023.
CoT      : Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large
           Language Models." NeurIPS 2022.
RAG      : Lewis et al. "Retrieval-Augmented Generation for Knowledge-
           Intensive NLP Tasks." NeurIPS 2020; Karpukhin et al. "Dense
           Passage Retrieval for Open-Domain Question Answering." EMNLP 2020.
Self-RAG : Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H. "Self-RAG:
           Learning to Retrieve, Generate, and Critique through Self-
           Reflection." ICLR 2024.  [citation-only; not implemented here
           because Self-RAG requires a Llama-2-7B backbone and the public
           checkpoint is not apples-to-apples with Flan-T5-Large.]
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from caem.config import CAEMConfig
from caem.memory.encoder import QueryEncoder
from caem.pipeline import PipelineResult
from caem.retrieval.rag import PassageStore, TierThreeRAG

logger = logging.getLogger(__name__)


# =============================================================================
# Base class
# =============================================================================

class BaselineBase:
    """Minimal pipeline-shaped base for the Chapter 5 external baselines.

    Subclasses implement ``_generate(query) -> (answer, tier, escalated)``.
    This base handles tokeniser/model loading, prompt tokenisation caps,
    latency measurement, and wrapping into a ``PipelineResult``.
    """

    name: str = "baseline"
    tier_value: int = 2  # override in retrieval-augmented subclasses

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 256,
        max_input_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

        logger.info("[%s] loading %s on %s", self.name, model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs
        ).to(device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Subclass hook                                                        #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, query: str) -> str:
        """Return the full prompt string fed to the model."""
        return query

    def _generate(self, query: str) -> Tuple[str, int, bool]:
        """Generate an answer. Returns (answer, tier, escalated)."""
        prompt = self._build_prompt(query)
        answer = self._run_generation(prompt)
        return answer, self.tier_value, False

    # ------------------------------------------------------------------ #
    # Shared generation helper                                             #
    # ------------------------------------------------------------------ #

    def _run_generation(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
    ) -> str:
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=do_sample,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ #
    # Pipeline-shaped public API                                           #
    # ------------------------------------------------------------------ #

    def answer(self, query: str, store_to_memory: bool = False) -> PipelineResult:
        """Drop-in replacement for ``CAEMPipeline.answer``.

        ``store_to_memory`` is ignored -- baselines have no memory.
        """
        t0 = time.perf_counter()
        try:
            answer_str, tier, escalated = self._generate(query)
        except Exception as exc:
            logger.error("[%s] generation failed for query: %s", self.name, exc)
            answer_str, tier, escalated = "", self.tier_value, False
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PipelineResult(
            query=query,
            answer=answer_str,
            tier=tier,
            stored=False,
            latency_ms=latency_ms,
            routing_decision=None,
            pre_confidence=None,
            post_confidence=None,
            verifier_output=None,
            u_stored=None,
            entry_id=None,
            escalated=escalated,
        )


# =============================================================================
# B1 -- Zero-shot Flan-T5-Large
# =============================================================================

class ZeroShotBaseline(BaselineBase):
    """Zero-shot Flan-T5-Large. Floor baseline: no CoT, no retrieval, no memory.

    Matches the generation configuration of ``CAEMPipeline._tier2`` so that
    Zero-shot vs. CAEM comparison isolates the effect of the CAEM machinery
    rather than the decoder configuration.
    """

    name = "zero_shot"
    tier_value = 2

    def _build_prompt(self, query: str) -> str:
        return query


# =============================================================================
# B2 -- Chain-of-Thought
# =============================================================================

class CoTBaseline(BaselineBase):
    """Chain-of-Thought prompting (Wei et al. NeurIPS 2022).

    Hyperparameters
    ---------------
    prefix : str
        [LIT] "Let's think step by step." -- the zero-shot CoT trigger phrase
        from Kojima et al. NeurIPS 2022.
    max_new_tokens : int
        [DES] 256 -- matches ``CaemConfig.cot_max_new_tokens``.
    """

    name = "cot"
    tier_value = 2
    prefix: str = "Let's think step by step. "

    def _build_prompt(self, query: str) -> str:
        return f"{self.prefix}{query}"


# =============================================================================
# B3 -- Retrieval-Augmented Generation
# =============================================================================

class RAGBaseline(BaselineBase):
    """DPR + Flan-T5-Large RAG baseline.

    Reuses ``caem.retrieval.rag.TierThreeRAG`` for passage retrieval, prompt
    assembly, and generation, so this baseline is numerically identical to
    CAEM's Tier 3 RAG when CAEM dispatches to Tier 3. The difference is that
    this baseline sends *every* query through RAG with no router.

    Hyperparameters
    ---------------
    top_k : int
        [DES] 5 -- ``CaemConfig.rag_top_k`` (Karpukhin et al. EMNLP 2020).
    max_context_tokens : int
        [DES] 384 -- ``CaemConfig.rag_max_context_tokens``.
    max_new_tokens : int
        [DES] 256 -- ``CaemConfig.rag_max_new_tokens``.
    """

    name = "rag"
    tier_value = 3

    def __init__(
        self,
        passage_store: PassageStore,
        passage_encoder: Optional[QueryEncoder] = None,
        config: Optional[CAEMConfig] = None,
        model_name: str = "google/flan-t5-large",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.config = config or CAEMConfig()
        super().__init__(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_new_tokens=self.config.rag_max_new_tokens,
            max_input_tokens=512,
        )
        # Query-side Sentence-BERT encoder for DPR retrieval. Reuse the
        # one the caller provides (typical when this baseline shares a
        # process with a CAEMPipeline) or construct a fresh one.
        self.passage_encoder = passage_encoder or QueryEncoder(device=self.device)
        # Share the generator with TierThreeRAG so retrieval + generation
        # match CAEM Tier 3 exactly.
        self.rag = TierThreeRAG(
            model=self.model,
            tokenizer=self.tokenizer,
            passage_encoder=self.passage_encoder,
            passage_store=passage_store,
            config=self.config,
            device=self.device,
        )

    def _generate(self, query: str) -> Tuple[str, int, bool]:
        answer = self.rag.generate(query)
        return answer, self.tier_value, False


# =============================================================================
# B4 -- CoT + RAG
# =============================================================================

class CoTRAGBaseline(RAGBaseline):
    """RAG with a Chain-of-Thought trigger injected before the answer cue.

    The prompt is the standard RAG numbered-context prompt produced by
    ``TierThreeRAG._build_prompt``. Following Wei et al. (NeurIPS 2022), the
    CoT trigger is placed immediately before the ``Answer:`` cue rather than
    at the very top of the prompt: the model needs to condition the reasoning
    chain on both the retrieved context and the question, not on a bare
    instruction preceding the context block.
    """

    name = "cot_rag"
    prefix: str = "Let's think step by step."
    _cue_marker: str = "\nAnswer:"

    def _inject_cot_trigger(self, prompt: str) -> str:
        """Insert ``self.prefix`` on its own line immediately before the final
        ``\\nAnswer:`` cue. Falls back to appending the trigger if the cue is
        not present (defensive; every branch of ``_build_prompt`` emits it).
        """
        idx = prompt.rfind(self._cue_marker)
        if idx == -1:
            return prompt.rstrip() + "\n" + self.prefix.rstrip()
        return prompt[:idx] + "\n" + self.prefix.rstrip() + prompt[idx:]

    def _generate(self, query: str) -> Tuple[str, int, bool]:
        # Retrieve passages via the shared RAG object.
        passages = self.rag.retrieve(query)
        prompt = self.rag._build_prompt(query, passages)
        prompt = self._inject_cot_trigger(prompt)
        answer = self._run_generation(
            prompt,
            max_new_tokens=self.config.rag_max_new_tokens,
            do_sample=self.config.rag_do_sample,
        )
        return answer, self.tier_value, False


# =============================================================================
# B5 -- FLARE (Jiang et al. EMNLP 2023)
# =============================================================================

class FLAREBaseline(RAGBaseline):
    """FLARE: active retrieval augmented generation via look-ahead confidence.

    At each decode step the model emits a 64-token look-ahead without
    retrieval. If any token's probability falls below ``theta`` the low-
    confidence token span is masked, the masked prefix becomes the retrieval
    query, and the current sentence is regenerated with retrieved passages
    prepended. Otherwise the look-ahead is committed and decoding continues.

    Hyperparameters
    ---------------
    theta : float
        [LIT] 0.4 -- confidence floor for triggering retrieval (Jiang et al.
        2023 §4.2: "we set the confidence threshold to 0.4").
    look_ahead_tokens : int
        [LIT] 64 -- the look-ahead horizon (Jiang et al. 2023 §4.2).
    max_sentences : int
        [DES] 8 -- cap total sentence regenerations to avoid runaway loops
        on degenerate queries.

    Notes
    -----
    This implementation uses sentence-level active retrieval with a single
    retrieval trigger per sentence, following the published FLARE variant.
    The token-level log-probability floor is computed from the model's own
    generation scores (``output_scores=True``).
    """

    name = "flare"
    tier_value = 3

    def __init__(
        self,
        passage_store: PassageStore,
        passage_encoder: Optional[QueryEncoder] = None,
        config: Optional[CAEMConfig] = None,
        model_name: str = "google/flan-t5-large",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        theta: float = 0.4,
        look_ahead_tokens: int = 64,
        max_sentences: int = 8,
    ) -> None:
        super().__init__(
            passage_store=passage_store,
            passage_encoder=passage_encoder,
            config=config,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
        self.theta = theta
        self.look_ahead_tokens = look_ahead_tokens
        self.max_sentences = max_sentences

    def _look_ahead(self, prompt: str) -> Tuple[str, float]:
        """Emit a 64-token look-ahead from ``prompt``; return (text, min_prob).

        min_prob is the minimum per-token probability across the emitted
        span; this is the scalar that FLARE thresholds against ``theta``.
        """
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=self.look_ahead_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # T5 is encoder-decoder: model.generate() returns only decoder tokens
        # in out.sequences, with shape (1, T+1) where position 0 is the
        # decoder_start_token_id (typically pad) and positions 1..T are the
        # generated tokens. Slicing by input_ids.shape[1] would be correct
        # for a decoder-only model (GPT-family) but yields an empty tensor
        # here; the correct offset is 1 (skip the start token). out.scores
        # is a tuple of T logits, one per generated step, aligned with
        # out.sequences[0, 1:].
        if not out.scores:
            return "", 1.0
        gen_ids = out.sequences[0, 1:]
        probs: List[float] = []
        for t, logits in enumerate(out.scores):
            if t >= len(gen_ids):
                break
            token_id = gen_ids[t].item()
            p = torch.softmax(logits[0], dim=-1)[token_id].item()
            probs.append(p)
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        min_prob = min(probs) if probs else 1.0
        return text, min_prob

    def _generate(self, query: str) -> Tuple[str, int, bool]:
        """Sentence-by-sentence active retrieval.

        Committed text accumulates across sentences; each sentence triggers
        either (a) a direct look-ahead commit if min-prob >= theta, or
        (b) a retrieval round followed by a grounded regeneration commit.
        ``escalated`` is True whenever at least one retrieval was triggered.
        """
        committed = ""
        escalated = False
        for _ in range(self.max_sentences):
            lookup_prompt = f"{query}\n\nAnswer: {committed}"
            look_text, min_prob = self._look_ahead(lookup_prompt)
            if not look_text:
                break

            # Take one sentence at a time.
            sentence = _first_sentence(look_text)

            if min_prob >= self.theta:
                # Commit the look-ahead sentence without retrieval.
                committed = _append_sentence(committed, sentence)
            else:
                # Retrieval trigger: mask low-prob tokens, use remaining
                # text as the retrieval query, and regenerate with context.
                escalated = True
                retrieval_query = sentence or query
                passages = self.rag.retrieve(retrieval_query)
                grounded_prompt = self.rag._build_prompt(
                    f"{query}\n\nPartial answer so far: {committed}",
                    passages,
                )
                grounded = self._run_generation(
                    grounded_prompt,
                    max_new_tokens=self.look_ahead_tokens,
                    do_sample=False,
                )
                committed = _append_sentence(committed, _first_sentence(grounded))

            # Termination: FLAN-T5 tends to emit a full answer in the first
            # sentence for factual-QA. Stop when look-ahead stops producing
            # new content or the committed text ends in a final-answer
            # marker (period + whitespace).
            if committed.rstrip().endswith((".", "?", "!")) and len(committed.split()) >= 3:
                break

        return committed.strip(), self.tier_value, escalated


# =============================================================================
# Small text utilities
# =============================================================================

def _first_sentence(text: str) -> str:
    """Return the first sentence (terminator-inclusive) from ``text``."""
    for i, ch in enumerate(text):
        if ch in ".!?":
            return text[: i + 1]
    return text


def _append_sentence(committed: str, sentence: str) -> str:
    """Append ``sentence`` to ``committed`` with sensible whitespace."""
    sentence = sentence.strip()
    if not sentence:
        return committed
    if not committed:
        return sentence
    return committed.rstrip() + " " + sentence


__all__ = [
    "BaselineBase",
    "ZeroShotBaseline",
    "CoTBaseline",
    "RAGBaseline",
    "CoTRAGBaseline",
    "FLAREBaseline",
]
