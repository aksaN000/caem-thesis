"""
eval/baselines.py
=================
External baselines for the CAEM Chapter 5 comparison panel (Branch C, Qwen-3B).

Design
------
Each baseline is a lightweight "pipeline-shaped" object that exposes the same
``answer(query, store_to_memory=False) -> PipelineResult`` signature as
``caem.pipeline.CAEMPipeline``. This lets ``eval.harness.EvalHarness`` consume
baselines without modification.

**Branch C decision (2026-04-22)**: all baselines now run on the same
decoder-only backbone (Qwen-2.5-3B-Instruct by default -- see
``CAEMConfig.base_model_name``). The prior Flan-T5-Large implementation is
preserved on the ``main`` branch for historical reproducibility. This
keeps the main-panel comparison apples-to-apples: CAEM vs. baselines are
measured on the same generator and the delta attributes cleanly to the
CAEM machinery rather than to backbone differences.

Baselines
---------
ZeroShotBaseline  (B1) -- Qwen-3B, single-turn ChatML, no CoT, no retrieval.
                          Floor.
CoTBaseline       (B2) -- Qwen-3B ChatML with a ``"Let's think step by step."``
                          trigger prepended to the user message.
RAGBaseline       (B3) -- DPR top-k retrieval + Qwen-3B context-conditioned
                          generation via ``caem.retrieval.rag.TierThreeRAG``.
CoTRAGBaseline    (B4) -- RAG context + explicit CoT trigger injected into
                          the prefill.
FLAREBaseline     (B5) -- Jiang et al. EMNLP 2023 active retrieval with
                          confidence-threshold look-ahead. Decoder-only
                          slicing: the generated suffix is ``out.sequences
                          [0, input_len:]`` (prompt echoed in the generate
                          output for causal LMs).

Mapping to PipelineResult fields
--------------------------------
``tier``                    -- 2 for generation-only baselines (Zero-shot, CoT),
                               3 for retrieval-augmented baselines.
``stored``                  -- always False.
``u_stored``                -- None (no verifier).
``verifier_output``         -- None.
``routing_decision``        -- None.
``pre_confidence``          -- None.
``escalated``               -- True for FLARE when a look-ahead triggered a
                               retrieval restart; False otherwise.

References
----------
FLARE    : Jiang, Z., Xu, F. F., et al. "Active Retrieval Augmented
           Generation." EMNLP 2023.
CoT      : Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large
           Language Models." NeurIPS 2022; Kojima et al. "Large Language
           Models are Zero-Shot Reasoners." NeurIPS 2022.
RAG      : Lewis et al. NeurIPS 2020; Karpukhin et al. EMNLP 2020.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Tuple

import torch

from caem.config import CAEMConfig
from caem.memory.encoder import QueryEncoder
from caem.model_loader import load_base_generator
from caem.pipeline import PipelineResult
from caem.prompts import FORCED_PREFIX, build_tier3_prompt
from caem.retrieval.rag import PassageStore, TierThreeRAG

logger = logging.getLogger(__name__)


# =============================================================================
# Base class
# =============================================================================

class BaselineBase:
    """Minimal pipeline-shaped base for the Chapter 5 external baselines.

    Subclasses implement ``_generate(query) -> (answer, tier, escalated)``.
    This base loads the shared Qwen-3B decoder-only generator (or any
    instruction-tuned decoder-only backbone listed in
    ``CAEMConfig.base_model_name``), handles prompt tokenisation caps,
    measures latency, and wraps results into a ``PipelineResult``.

    Decoder-only slicing
    --------------------
    ``model.generate()`` on a causal LM returns ``input_ids + generated_ids``
    concatenated; the correct generated-only slice is
    ``output_ids[0, input_len:]``. (The pre-Branch-C T5 code used
    ``output_ids[0]`` because encoder-decoder generate returns only the
    decoder tokens starting after the start-of-sequence.) ``_run_generation``
    below is the single source of truth for the slice.
    """

    name: str = "baseline"
    tier_value: int = 2  # override in retrieval-augmented subclasses

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 256,
        max_input_tokens: int = 2048,
        config: Optional[CAEMConfig] = None,
        use_flash_attention_2: Optional[bool] = None,
        use_torch_compile: Optional[bool] = None,
    ) -> None:
        cfg = config or CAEMConfig()
        self.config = cfg
        self.model_name = model_name or cfg.base_model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

        load_dtype = dtype if dtype is not None else torch.bfloat16
        flash_attn = cfg.use_flash_attention_2 if use_flash_attention_2 is None else use_flash_attention_2
        compile_ = cfg.use_torch_compile if use_torch_compile is None else use_torch_compile

        logger.info("[%s] loading %s on %s", self.name, self.model_name, device)
        self.model, self.tokenizer = load_base_generator(
            self.model_name,
            device=device,
            dtype=load_dtype,
            use_flash_attention_2=flash_attn,
            use_torch_compile=compile_,
        )
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Subclass hook                                                        #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, query: str) -> str:
        """Return the full prompt string fed to the model. Subclasses override."""
        return self._wrap_chatml_user(query)

    def _generate(self, query: str) -> Tuple[str, int, bool]:
        """Generate an answer. Returns (answer, tier, escalated)."""
        prompt = self._build_prompt(query)
        answer = self._run_generation(prompt)
        return answer, self.tier_value, False

    # ------------------------------------------------------------------ #
    # ChatML wrapping                                                      #
    # ------------------------------------------------------------------ #

    def _wrap_chatml_user(self, user_content: str) -> str:
        """Wrap ``user_content`` in a single-turn ChatML prompt with the
        assistant-generation marker appended.

        Falls back to the raw user content if the tokenizer does not expose
        ``apply_chat_template`` or the template call returns a non-string
        (mock tokenizers / non-instruction-tuned models).
        """
        tok = self.tokenizer
        messages = [{"role": "user", "content": user_content}]
        if hasattr(tok, "apply_chat_template"):
            try:
                rendered = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                if isinstance(rendered, str):
                    return rendered
            except Exception:
                pass
        return user_content

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
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        input_len = int(input_ids.shape[1])
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Decoder-only slice: prompt is echoed in output_ids.
        gen_ids = output_ids[0, input_len:] if output_ids.shape[1] > input_len else output_ids[0]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def _run_generation_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
    ) -> List[str]:
        """Goal 5 Level B batched generate. Left-pads the N prompts, runs one
        model.generate(), slices per-row. For decoder-only Qwen, left-padding
        is required so the last position of each row is always the generation
        start (causal attention makes the right-pad tail inert).
        """
        if not prompts:
            return []
        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
                add_special_tokens=False,
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = original_padding_side

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        max_input_len = int(input_ids.shape[1])
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # With left-padding the prompt ends at column (max_input_len - 1), so
        # generated tokens always start at column max_input_len. Per-row slice.
        results: List[str] = []
        for i in range(output_ids.shape[0]):
            if output_ids.shape[1] > max_input_len:
                gen_ids = output_ids[i, max_input_len:]
            else:
                gen_ids = output_ids[i]
            results.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return results

    # ------------------------------------------------------------------ #
    # Pipeline-shaped public API                                           #
    # ------------------------------------------------------------------ #

    def answer_batch(self, queries: List[str], store_to_memory: bool = False) -> List[PipelineResult]:
        """Goal 5 Level B batched API. Default implementation: build N prompts via
        ``self._build_prompt`` and run one batched ``_run_generation_batch``.
        RAG / CoT-RAG override to do batched retrieve + per-query prompt assembly
        before the batched generate.
        """
        if not queries:
            return []
        t0 = time.perf_counter()
        try:
            prompts = [self._build_prompt(q) for q in queries]
            answers = self._run_generation_batch(prompts)
        except Exception as exc:
            logger.error("[%s] batched generation failed (%d queries): %s",
                         self.name, len(queries), exc)
            answers = [""] * len(queries)
        batch_ms = (time.perf_counter() - t0) * 1000.0
        per_sample_ms = batch_ms / max(len(queries), 1)
        return [
            PipelineResult(
                query=q,
                answer=ans,
                tier=self.tier_value,
                stored=False,
                latency_ms=per_sample_ms,
                routing_decision=None,
                pre_confidence=None,
                post_confidence=None,
                verifier_output=None,
                u_stored=None,
                entry_id=None,
                escalated=False,
            )
            for q, ans in zip(queries, answers)
        ]

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
# B1 -- Zero-shot
# =============================================================================

class ZeroShotBaseline(BaselineBase):
    """Zero-shot Qwen-3B. Floor baseline: no CoT, no retrieval, no memory.

    The query is ChatML-wrapped as a single user turn with no system prompt
    and no few-shot examples -- the cleanest ablation against CAEM's
    scaffolded-CoT system prompt. Differences on Chapter-5 metrics then
    attribute to CAEM's machinery rather than to ChatML-vs-flat prompting.
    """

    name = "zero_shot"
    tier_value = 2

    def _build_prompt(self, query: str) -> str:
        return self._wrap_chatml_user(query)


# =============================================================================
# B2 -- Chain-of-Thought
# =============================================================================

class CoTBaseline(BaselineBase):
    """Zero-shot CoT (Kojima et al. NeurIPS 2022 / Wei et al. NeurIPS 2022).

    Prepends ``"Let's think step by step."`` to the user turn. Still
    ChatML-wrapped, single-turn, no retrieval.

    Hyperparameters
    ---------------
    prefix : str
        [LIT] "Let's think step by step." -- Kojima et al. NeurIPS 2022 zero-
        shot CoT trigger.
    max_new_tokens : int
        [DES] Inherits ``BaselineBase`` default (256); CoT outputs rarely
        exceed that on the factual-QA panel.
    """

    name = "cot"
    tier_value = 2
    prefix: str = "Let's think step by step."

    def _build_prompt(self, query: str) -> str:
        return self._wrap_chatml_user(f"{self.prefix}\n\n{query}")


# =============================================================================
# B5 -- Few-shot CoT (Wei et al. NeurIPS 2022)
# =============================================================================

class FiveShotCoTBaseline(BaselineBase):
    """5-shot chain-of-thought baseline (Wei et al. NeurIPS 2022).

    Prepends 5 worked-example demonstrations (question -> Reasoning ->
    Answer) before the user query, then appends "Let's think step by
    step." before the live query. Demos are drawn from the benchmark's
    training split at baseline-build time via a fixed seed (42) so the
    same demo set is reused across all eval queries in a run. This is
    the canonical in-context learning reference that closes the
    "did you try few-shot before SFT?" critique.

    Hyperparameters
    ---------------
    n_shots : int
        [LIT] 5 -- Wei et al. 2022 standard few-shot CoT.
    demo_seed : int
        [DES] 42 -- fixed seed for reproducibility.
    max_input_tokens : int
        [DES] Inherits BaselineBase (2048); 5 factoid demos + query
        comfortably fit under this cap.
    """

    name = "fiveshot_cot"
    tier_value = 2
    prefix: str = "Let's think step by step."
    n_shots: int = 5

    def __init__(
        self,
        *args,
        demo_samples: Optional[List[dict]] = None,
        demo_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.demo_seed = demo_seed
        self._demo_block = self._build_demo_block(demo_samples or [])

    def _build_demo_block(self, samples: List[dict]) -> str:
        """Select n_shots demos and format as Q -> Reasoning -> A triples.

        The Reasoning field is derived heuristically from the gold answer —
        we cannot fabricate true step-by-step reasoning without an oracle,
        so Reasoning becomes "The answer is <gold>." This follows Wei et
        al.'s pattern where demos use minimal CoT templates for short-form
        QA. For rich-reasoning datasets (GSM8K-style) a future ablation
        could use annotated CoT demos; the factoid QA panel here doesn't
        need that level of demo sophistication.
        """
        if not samples:
            return ""
        import random
        rng = random.Random(self.demo_seed)
        picked = rng.sample(samples, min(self.n_shots, len(samples)))
        parts: List[str] = []
        for s in picked:
            q = str(s.get("question", "")).strip()
            golds = s.get("answers") or []
            gold = str(golds[0]).strip() if golds else ""
            if not q or not gold:
                continue
            parts.append(
                f"Question: {q}\n"
                f"Reasoning: The answer is {gold}.\n"
                f"Answer: {gold}"
            )
        if not parts:
            return ""
        return "\n\n".join(parts)

    def _build_prompt(self, query: str) -> str:
        if self._demo_block:
            content = (
                f"Here are some examples of answering questions with brief reasoning:\n\n"
                f"{self._demo_block}\n\n"
                f"{self.prefix}\n\n"
                f"Question: {query}\nReasoning:"
            )
        else:
            # Fall back to zero-shot CoT if no demos were provided (e.g. at
            # smoke-test time). Produces B2-equivalent behaviour for that run.
            content = f"{self.prefix}\n\n{query}"
        return self._wrap_chatml_user(content)


# =============================================================================
# B3 -- Retrieval-Augmented Generation
# =============================================================================

class RAGBaseline(BaselineBase):
    """DPR + Qwen-3B RAG baseline.

    Delegates to ``caem.retrieval.rag.TierThreeRAG`` for passage retrieval,
    ChatML prompt assembly, and generation, so this baseline is numerically
    identical to CAEM's Tier 3 RAG when CAEM dispatches to Tier 3. The
    difference is that this baseline routes *every* query through RAG with
    no router.

    Hyperparameters
    ---------------
    top_k : int
        [DES] ``CaemConfig.rag_top_k`` (Karpukhin et al. EMNLP 2020).
    max_context_tokens : int
        [DES] ``CaemConfig.rag_max_context_tokens``.
    max_new_tokens : int
        [DES] ``CaemConfig.rag_max_new_tokens``.
    """

    name = "rag"
    tier_value = 3

    def __init__(
        self,
        passage_store: PassageStore,
        passage_encoder: Optional[QueryEncoder] = None,
        config: Optional[CAEMConfig] = None,
        model_name: Optional[str] = None,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        cfg = config or CAEMConfig()
        super().__init__(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_new_tokens=cfg.rag_max_new_tokens,
            max_input_tokens=cfg.rag_max_context_tokens * 8 + cfg.rag_max_new_tokens,
            config=cfg,
        )
        self.passage_encoder = passage_encoder or QueryEncoder(device=self.device)
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

    def answer_batch(self, queries: List[str], store_to_memory: bool = False) -> List[PipelineResult]:
        """Goal 5 Level B batched RAG: delegates to ``TierThreeRAG.generate_batch``
        which does per-query retrieval + prompt assembly then one batched
        ``model.generate`` over left-padded prompts.
        """
        if not queries:
            return []
        t0 = time.perf_counter()
        try:
            answers = self.rag.generate_batch(queries)
        except Exception as exc:
            logger.error("[%s] batched RAG generation failed (%d queries): %s",
                         self.name, len(queries), exc)
            answers = [""] * len(queries)
        batch_ms = (time.perf_counter() - t0) * 1000.0
        per_sample_ms = batch_ms / max(len(queries), 1)
        return [
            PipelineResult(
                query=q, answer=ans, tier=self.tier_value,
                stored=False, latency_ms=per_sample_ms,
                routing_decision=None, pre_confidence=None, post_confidence=None,
                verifier_output=None, u_stored=None, entry_id=None, escalated=False,
            )
            for q, ans in zip(queries, answers)
        ]


# =============================================================================
# B4 -- CoT + RAG
# =============================================================================

class CoTRAGBaseline(RAGBaseline):
    """RAG with an explicit CoT trigger injected into the assistant prefill.

    The prompt is the scaffolded-CoT ChatML prompt from
    ``caem.prompts.build_tier3_prompt``. The CoT trigger is prepended to
    the ``"Reasoning:"`` forced prefix so the assistant turn begins with

        "Let's think step by step.\\nReasoning: ..."

    Ch5 reports B3 vs B4 together to isolate the effect of an explicit CoT
    trigger over the scaffolded Reasoning/Answer format alone.
    """

    name = "cot_rag"
    prefix: str = "Let's think step by step."

    def _generate(self, query: str) -> Tuple[str, int, bool]:
        passages = self.rag.retrieve(query)
        prompt_text, forced_prefix = build_tier3_prompt(
            query, passages, tokenizer=self.tokenizer,
        )
        # Prepend the CoT trigger to the prefill so the assistant turn reads
        # "Let's think step by step.\nReasoning: ..." -- matches the inject-
        # before-Answer-cue semantics of the pre-Branch-C T5 implementation,
        # adapted to ChatML's prefill-based forcing.
        full_prompt = f"{prompt_text}{self.prefix}\n{forced_prefix}"
        continuation = self._run_generation(
            full_prompt,
            max_new_tokens=self.config.rag_max_new_tokens,
            do_sample=self.config.rag_do_sample,
        )
        # Surface the forced prefix in the returned answer so the downstream
        # scorer sees the same "Reasoning: ...\nAnswer: ..." shape as CAEM Tier 3.
        answer = f"{forced_prefix}{continuation}" if continuation else forced_prefix
        return answer, self.tier_value, False

    def answer_batch(self, queries: List[str], store_to_memory: bool = False) -> List[PipelineResult]:
        """Goal 5 Level B batched CoT-RAG. Builds per-query CoT-RAG prompts
        (retrieve -> build_tier3_prompt -> prepend CoT trigger -> append forced
        "Reasoning:" prefix) and runs one batched left-padded ``model.generate``.
        Surfaces the forced prefix in the returned answer so Ch5 scorers see
        the same "Reasoning: .../Answer: ..." shape as CAEM Tier 3.
        """
        if not queries:
            return []
        t0 = time.perf_counter()
        forced_prefixes: List[str] = []
        full_prompts: List[str] = []
        for q in queries:
            passages = self.rag.retrieve(q)
            prompt_text, forced_prefix = build_tier3_prompt(
                q, passages, tokenizer=self.tokenizer,
            )
            forced_prefixes.append(forced_prefix)
            full_prompts.append(f"{prompt_text}{self.prefix}\n{forced_prefix}")

        try:
            continuations = self._run_generation_batch(
                full_prompts,
                max_new_tokens=self.config.rag_max_new_tokens,
                do_sample=self.config.rag_do_sample,
            )
        except Exception as exc:
            logger.error("[%s] batched CoT-RAG generation failed (%d queries): %s",
                         self.name, len(queries), exc)
            continuations = [""] * len(queries)

        batch_ms = (time.perf_counter() - t0) * 1000.0
        per_sample_ms = batch_ms / max(len(queries), 1)
        results: List[PipelineResult] = []
        for q, cont, fp in zip(queries, continuations, forced_prefixes):
            answer_str = f"{fp}{cont}" if cont else fp
            results.append(PipelineResult(
                query=q, answer=answer_str, tier=self.tier_value,
                stored=False, latency_ms=per_sample_ms,
                routing_decision=None, pre_confidence=None, post_confidence=None,
                verifier_output=None, u_stored=None, entry_id=None, escalated=False,
            ))
        return results


# =============================================================================
# B5 -- FLARE (Jiang et al. EMNLP 2023)
# =============================================================================

class FLAREBaseline(RAGBaseline):
    """FLARE: active retrieval augmented generation via look-ahead confidence.

    At each sentence boundary the model emits a 64-token look-ahead without
    retrieval. If any token's probability falls below ``theta`` the low-
    confidence span triggers a retrieval and the sentence is regenerated
    with retrieved passages prepended. Otherwise the look-ahead is committed
    and decoding continues.

    Hyperparameters
    ---------------
    theta : float
        [LIT] 0.4 -- confidence floor for triggering retrieval (Jiang et al.
        2023 §4.2).
    look_ahead_tokens : int
        [LIT] 64 -- look-ahead horizon (Jiang et al. 2023 §4.2).
    max_sentences : int
        [DES] 8 -- cap total sentence regenerations.

    Decoder-only slicing
    --------------------
    ``model.generate()`` returns ``input_ids + generated_ids`` concatenated
    for a causal LM, so ``out.sequences[0, input_len:]`` is the generated
    suffix. The pre-Branch-C T5 implementation sliced ``out.sequences[0, 1:]``
    (skip decoder-start token) -- NOT valid here. The current code paths this
    correctly; ``tests/test_flare_smoke.py`` is the regression guard.
    """

    name = "flare"
    tier_value = 3

    def __init__(
        self,
        passage_store: PassageStore,
        passage_encoder: Optional[QueryEncoder] = None,
        config: Optional[CAEMConfig] = None,
        model_name: Optional[str] = None,
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

    def _look_ahead(self, query: str, committed: str) -> Tuple[str, float]:
        """Emit a ``look_ahead_tokens`` span continuing ``committed``; return
        (text, min_prob).

        min_prob is the minimum per-token probability across the emitted span
        -- the scalar FLARE thresholds against ``theta``.

        The look-ahead prompt wraps ``query`` as a ChatML user turn and uses
        ``committed`` as the assistant prefill (so the model continues from
        wherever the caller has accumulated text). For an empty ``committed``
        the assistant starts from a bare generation marker.
        """
        prompt_text = self._wrap_chatml_user(query)
        full_prompt = f"{prompt_text}{committed}" if committed else prompt_text
        enc = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        input_len = int(input_ids.shape[1])
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.look_ahead_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if not out.scores:
            return "", 1.0

        # Decoder-only: out.sequences = [input_ids, generated_ids] concatenated.
        # The slice [0, input_len:] is the generated tail, aligned with out.scores.
        gen_ids = out.sequences[0, input_len:]
        probs: List[float] = []
        for t, logits in enumerate(out.scores):
            if t >= len(gen_ids):
                break
            token_id = int(gen_ids[t].item())
            p = torch.softmax(logits[0], dim=-1)[token_id].item()
            probs.append(p)
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        min_prob = min(probs) if probs else 1.0
        return text, min_prob

    def _grounded_generate(
        self,
        query: str,
        committed: str,
        passages: List[Tuple[str, float]],
    ) -> str:
        """Regenerate the current sentence with retrieved passages prepended.

        Uses ``build_tier3_prompt`` so the grounded regeneration matches
        CAEM's Tier 3 ChatML format. The accumulated ``committed`` text is
        threaded as a ``Partial answer so far:`` prefix in the query text
        (keeps the prompt shape simple while preserving FLARE's per-sentence
        grounding behaviour).
        """
        query_with_context = (
            f"{query}\n\nPartial answer so far: {committed}"
            if committed
            else query
        )
        prompt_text, forced_prefix = build_tier3_prompt(
            query_with_context, passages, tokenizer=self.tokenizer,
        )
        full_prompt = prompt_text + forced_prefix
        return self._run_generation(
            full_prompt,
            max_new_tokens=self.look_ahead_tokens,
            do_sample=False,
        )

    def _generate(self, query: str) -> Tuple[str, int, bool]:
        """Sentence-by-sentence active retrieval.

        Committed text accumulates across sentences; each iteration either
        (a) commits a look-ahead sentence if its min-prob >= theta, or
        (b) triggers retrieval and commits a grounded regeneration.
        ``escalated`` is True whenever at least one retrieval fires.
        """
        committed = ""
        escalated = False
        for _ in range(self.max_sentences):
            look_text, min_prob = self._look_ahead(query, committed)
            if not look_text:
                break

            sentence = _first_sentence(look_text)

            if min_prob >= self.theta:
                committed = _append_sentence(committed, sentence)
            else:
                escalated = True
                retrieval_query = sentence or query
                passages = self.rag.retrieve(retrieval_query)
                grounded = self._grounded_generate(query, committed, passages)
                # Strip the "Reasoning:" prefix that the scaffolded prompt
                # forces; FLARE's committed buffer tracks natural-language
                # sentences, not scaffolded CoT tokens.
                if grounded.lower().lstrip().startswith(FORCED_PREFIX.lower()):
                    grounded = grounded.lstrip()[len(FORCED_PREFIX):].lstrip()
                committed = _append_sentence(committed, _first_sentence(grounded))

            if committed.rstrip().endswith((".", "?", "!")) and len(committed.split()) >= 3:
                break

        return committed.strip(), self.tier_value, escalated

    def answer_batch(self, queries: List[str], store_to_memory: bool = False) -> List[PipelineResult]:
        """FLARE cannot be cleanly batched: sentence-by-sentence decode with
        per-sentence confidence gating and conditional retrieval produces
        different control flow per query and different sentence counts.
        Batching would require reducing all queries to a fixed number of
        decode steps, which changes the algorithm. Falls back to per-query
        serial ``answer()``. FLARE's ~3-hour serial cost on Step 13 is
        accepted in Ch5 §5.6 "Performance envelope" per branch_C.md.
        """
        return [self.answer(q, store_to_memory=store_to_memory) for q in queries]


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
    "FiveShotCoTBaseline",
    "RAGBaseline",
    "CoTRAGBaseline",
    "FLAREBaseline",
]
