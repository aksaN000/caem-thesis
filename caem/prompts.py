"""
caem/prompts.py
=================
Benchmark-aware prompt builders for Tier 2 (no-RAG) and Tier 3 (RAG)
generation, dispatching on ``prompt_style`` (Goal 1, Branch C):

  - ``"chatml_scaffold"`` — ChatML envelope for decoder-only backbones (Qwen,
    Gemma, Llama). Uses separate ``user``/``assistant`` turn pairs for the
    few-shot example; the scaffold semantic (Reasoning → Answer) is preserved
    exactly. Forced "Reasoning:" prefix is applied via prefill text appended
    after the generation prompt.

  - ``"flan_t5_scaffold"`` — flat-text single-turn prompt with an inline
    worked example (legacy Flan-T5 path). Forced "Reasoning:" prefix is
    applied via ``decoder_input_ids`` by the caller. Preserved bit-identically
    to the previous inline implementations in ``caem/pipeline.py`` and
    ``caem/retrieval/rag.py`` so the ``flan_t5_large_backbone`` Variant-18
    ablation row reproduces pre-Branch-C behavior exactly.

Task detection and few-shot content are shared across styles; only the
outer framing differs. This is the single source of truth for benchmark-
specific prompt content; ``pipeline.py`` and ``rag.py`` will dispatch here
in a follow-up commit.

Public API
----------
    build_tier2_prompt(query, prompt_style, tokenizer=None) -> (prompt, prefix)
    build_tier3_prompt(query, passages, prompt_style, tokenizer=None) -> (prompt, prefix)

Returns a ``(full_prompt_text, forced_prefix)`` tuple:
  - ``full_prompt_text``: the prompt as a string. For chatml_scaffold this
    ends with the ``<|im_start|>assistant`` generation-prompt marker.
  - ``forced_prefix``: the "Reasoning:" text that must start the generated
    output. For decoder-only (ChatML): caller appends this to the full prompt
    and tokenizes the combined string (prefill). For encoder-decoder
    (Flan-T5): caller tokenizes this separately and passes as
    ``decoder_input_ids``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


FORCED_PREFIX = "Reasoning:"

SYSTEM_PROMPT = (
    "You are a careful assistant. For each question, think step-by-step about "
    "what is being asked, then give a final answer in the specified format. "
    "Always respond exactly as:\n"
    "Reasoning: <your step-by-step thought>\n"
    "Answer: <final answer>"
)


# -------------------------------------------------------------------------- #
# Task detection + task spec (shared across styles)                            #
# -------------------------------------------------------------------------- #

def detect_query_task(query: str) -> str:
    """Infer benchmark task style from the query's constrained prefix.

    Returns one of ``"fever"``, ``"strategyqa"``, ``"arc"``, or ``"open"``.
    Matches the detection logic in ``caem/pipeline.py`` and
    ``caem/retrieval/rag.py`` — DO NOT diverge; all three paths must agree.
    """
    q = query.lower().strip()
    if q.startswith("answer with one of: supports, refutes, not enough info."):
        return "fever"
    if q.startswith("answer yes or no."):
        return "strategyqa"
    if ("choices:" in q) and ("multiple choice letter" in q):
        return "arc"
    return "open"


def _extract_after_token(query: str, token: str) -> str:
    """Return substring after ``token`` (case-insensitive), else full query."""
    q_lower = query.lower()
    t_lower = token.lower()
    idx = q_lower.find(t_lower)
    if idx == -1:
        return query.strip()
    return query[idx + len(token):].strip()


def _task_spec(task: str, query: str, with_passages: bool) -> Tuple[str, str]:
    """Return ``(task_instruction_line, answer_format_spec)`` per task.

    ``with_passages=True`` uses the Tier 3 instruction wording that references
    "the context above"; ``False`` uses the Tier 2 wording for no-RAG paths.
    Both match the legacy rag.py / pipeline.py wordings bit-identically.
    """
    if task == "fever":
        claim = _extract_after_token(query, "Claim:")
        if with_passages:
            return (
                f"Claim: {claim}\n"
                "Determine whether the claim is SUPPORTS, REFUTES, or "
                "NOT ENOUGH INFO based on the context above.",
                "supports | refutes | not enough info",
            )
        return (
            f"Claim: {claim}\n"
            "Determine whether the claim is SUPPORTS, REFUTES, or "
            "NOT ENOUGH INFO based on your knowledge.",
            "supports | refutes | not enough info",
        )
    if task == "strategyqa":
        q_text = _extract_after_token(query, "Question:")
        if with_passages:
            return (
                f"Question: {q_text}\n"
                "Answer the question with yes or no based on the "
                "context above.",
                "yes | no",
            )
        return (
            f"Question: {q_text}\n"
            "Answer the question with yes or no.",
            "yes | no",
        )
    if task == "arc":
        if with_passages:
            return (
                f"{query}\n"
                "Choose the correct answer from the listed choices "
                "based on the context above.",
                "A | B | C | D",
            )
        return (
            f"{query}\n"
            "Choose the correct answer from the listed choices.",
            "A | B | C | D",
        )
    # Open-ended QA (TriviaQA, NQ, TruthfulQA)
    if with_passages:
        return (
            f"Question: {query}\n"
            "Answer the question based on the context above.",
            "<concise factual answer>",
        )
    return (
        f"Question: {query}\n"
        "Answer the question.",
        "<concise factual answer>",
    )


# -------------------------------------------------------------------------- #
# Few-shot example content (shared across styles)                              #
# -------------------------------------------------------------------------- #

def _few_shot_parts(task: str, with_passages: bool) -> Tuple[str, str, str]:
    """Return the (context_block, user_instance, assistant_response) parts of
    the worked example for this task.

    - ``context_block`` is the example's passage context (empty string when
      ``with_passages=False`` or the task has no passage block).
    - ``user_instance`` is the claim/question the example is answering.
    - ``assistant_response`` is the ideal "Reasoning: ...\nAnswer: ..." output.

    These parts are composed differently by the two style builders:
    - ChatML: context prepended to user_instance in a user turn; assistant
      turn is the response.
    - Flan-T5: all three concatenated inline with the task wrapper.

    Content matches the legacy implementations in
    ``caem/pipeline.py::_build_tier2_prompt`` (with_passages=False) and
    ``caem/retrieval/rag.py::_few_shot_example`` (with_passages=True) so
    bit-identical reproduction of the Flan-T5 prompt is preserved for Variant
    18 ablation.
    """
    if task == "fever":
        if with_passages:
            ctx = (
                "Context:\n"
                "[1] Barack Obama served as the 44th President of the "
                "United States from 2009 to 2017."
            )
            user_inst = "Claim: Barack Obama was the 44th US President."
            asst = (
                "Reasoning: The context directly states that Obama "
                "served as the 44th President of the United States. "
                "The claim matches this fact exactly.\n"
                "Answer: supports"
            )
        else:
            ctx = ""
            user_inst = "Claim: Barack Obama was the 44th US President."
            asst = (
                "Reasoning: Barack Obama served as the 44th "
                "President of the United States from 2009 to 2017. "
                "The claim matches this fact.\n"
                "Answer: supports"
            )
        return ctx, user_inst, asst

    if task == "strategyqa":
        if with_passages:
            ctx = (
                "Context:\n"
                "[1] Water boils at 100 degrees Celsius at sea level. "
                "A pressure cooker can reach 120 degrees Celsius."
            )
            user_inst = (
                "Question: Can a pressure cooker cook food faster than "
                "boiling water?"
            )
            asst = (
                "Reasoning: Boiling water is capped at 100 C. A "
                "pressure cooker exceeds this, reaching 120 C, and "
                "higher temperatures speed up cooking.\n"
                "Answer: yes"
            )
        else:
            ctx = ""
            user_inst = (
                "Question: Can a pressure cooker cook food faster "
                "than boiling water?"
            )
            asst = (
                "Reasoning: Boiling water caps at 100 C. A pressure "
                "cooker reaches 120 C, and higher temperatures speed "
                "up cooking.\n"
                "Answer: yes"
            )
        return ctx, user_inst, asst

    if task == "arc":
        if with_passages:
            ctx = (
                "Context:\n"
                "[1] Plants convert sunlight into chemical energy "
                "through photosynthesis, producing glucose and oxygen."
            )
            user_inst = (
                "Question: What process allows plants to make food? "
                "Choices: (A) digestion (B) photosynthesis (C) "
                "respiration (D) fermentation\n"
                "Answer with just the multiple choice letter."
            )
            asst = (
                "Reasoning: Photosynthesis converts sunlight to "
                "chemical energy, producing glucose. This is how "
                "plants make their own food.\n"
                "Answer: B"
            )
        else:
            ctx = ""
            user_inst = (
                "Question: What process allows plants to make food? "
                "Choices: (A) digestion (B) photosynthesis (C) "
                "respiration (D) fermentation"
            )
            asst = (
                "Reasoning: Photosynthesis converts sunlight to "
                "chemical energy, producing glucose. This is how "
                "plants make their own food.\n"
                "Answer: B"
            )
        return ctx, user_inst, asst

    # Open-ended
    if with_passages:
        ctx = (
            "Context:\n"
            "[1] The Great Wall of China was built over several "
            "centuries, with most of the current structure built "
            "during the Ming dynasty (1368-1644)."
        )
        user_inst = (
            "Question: When was most of the Great Wall of China built?"
        )
        asst = (
            "Reasoning: According to the context, most of the current "
            "structure of the Great Wall was built during the Ming "
            "dynasty, which lasted from 1368 to 1644.\n"
            "Answer: during the Ming dynasty (1368-1644)"
        )
    else:
        ctx = ""
        user_inst = (
            "Question: When was most of the Great Wall of China built?"
        )
        asst = (
            "Reasoning: Most of the current structure was built "
            "during the Ming dynasty, which lasted from 1368 to "
            "1644.\n"
            "Answer: during the Ming dynasty (1368-1644)"
        )
    return ctx, user_inst, asst


# -------------------------------------------------------------------------- #
# Flan-T5 builders (legacy, bit-identical to previous inline code)             #
# -------------------------------------------------------------------------- #

def _flan_t5_few_shot_inline(task: str, with_passages: bool) -> str:
    """Compose the few-shot example as a single block matching the legacy
    inline format used by pipeline._build_tier2_prompt and
    rag._few_shot_example. Used by flan_t5_scaffold paths only.
    """
    ctx, user_inst, asst = _few_shot_parts(task, with_passages)
    if with_passages:
        # rag.py format:  Context:\n[1] ...\n\n<user_inst>\n<asst>
        return f"{ctx}\n\n{user_inst}\n{asst}"
    # pipeline.py format: <user_inst>\n<asst>
    return f"{user_inst}\n{asst}"


def _build_flan_t5_tier2(query: str) -> Tuple[str, str]:
    task = detect_query_task(query)
    task_line, answer_format = _task_spec(task, query, with_passages=False)
    example = _flan_t5_few_shot_inline(task, with_passages=False)

    prompt = (
        "Answer the question using step-by-step reasoning. "
        "Always write out your reasoning before the answer.\n\n"
        "Example:\n"
        f"{example}\n\n"
        "Now answer the following.\n\n"
        f"{task_line}\n\n"
        "Response format (fill in each field):\n"
        "Reasoning:\n"
        f"Answer: {answer_format}"
    )
    return prompt, FORCED_PREFIX


def _build_flan_t5_tier3(
    query: str,
    passages: List[Tuple[str, float]],
) -> Tuple[str, str]:
    task = detect_query_task(query)
    task_line, answer_format = _task_spec(task, query, with_passages=True)
    example = _flan_t5_few_shot_inline(task, with_passages=True)

    lines: List[str] = [
        "Answer the question using step-by-step reasoning. "
        "Always write out your reasoning before the answer.",
        "",
        "Example:",
        example,
        "",
        "Now answer the following.",
        "",
        "Context:",
    ]
    for i, (passage, _score) in enumerate(passages, start=1):
        lines.append(f"[{i}] {passage}")
    lines.append("")
    lines.append(task_line)
    lines.append("")
    lines.append("Response format (fill in each field):")
    lines.append("Reasoning:")
    lines.append(f"Answer: {answer_format}")
    return "\n".join(lines), FORCED_PREFIX


# -------------------------------------------------------------------------- #
# ChatML builders (Qwen / Gemma / Llama decoder-only backbones)                #
# -------------------------------------------------------------------------- #

def _build_chatml(
    messages: List[dict],
    tokenizer: Any,
) -> str:
    """Render a messages list to ChatML text with ``add_generation_prompt=True``.

    Uses the tokenizer's ``apply_chat_template`` when available (standard for
    Qwen/Gemma/Llama-3). Falls back to a manual ChatML concatenation if the
    tokenizer doesn't implement the method — rare; most instruction-tuned
    models ship with a chat_template.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    # Manual fallback — Qwen/ChatML syntax
    parts: List[str] = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _build_chatml_tier2(
    query: str,
    tokenizer: Any,
) -> Tuple[str, str]:
    task = detect_query_task(query)
    task_line, answer_format = _task_spec(task, query, with_passages=False)
    _, ex_user, ex_asst = _few_shot_parts(task, with_passages=False)

    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex_user},
        {"role": "assistant", "content": ex_asst},
        {
            "role": "user",
            "content": (
                f"{task_line}\n\n"
                f"Answer format: {answer_format}"
            ),
        },
    ]
    prompt_text = _build_chatml(messages, tokenizer)
    return prompt_text, FORCED_PREFIX


def _build_chatml_tier3(
    query: str,
    passages: List[Tuple[str, float]],
    tokenizer: Any,
) -> Tuple[str, str]:
    task = detect_query_task(query)
    task_line, answer_format = _task_spec(task, query, with_passages=True)
    ex_ctx, ex_user, ex_asst = _few_shot_parts(task, with_passages=True)

    # Build the actual Context block from retrieved passages
    ctx_lines = ["Context:"]
    for i, (passage, _score) in enumerate(passages, start=1):
        ctx_lines.append(f"[{i}] {passage}")
    actual_context = "\n".join(ctx_lines)

    # The example's user turn carries the example's context+claim; the real
    # user turn carries the actual context+task line.
    example_user_content = f"{ex_ctx}\n\n{ex_user}" if ex_ctx else ex_user

    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example_user_content},
        {"role": "assistant", "content": ex_asst},
        {
            "role": "user",
            "content": (
                f"{actual_context}\n\n"
                f"{task_line}\n\n"
                f"Answer format: {answer_format}"
            ),
        },
    ]
    prompt_text = _build_chatml(messages, tokenizer)
    return prompt_text, FORCED_PREFIX


# -------------------------------------------------------------------------- #
# Public API                                                                   #
# -------------------------------------------------------------------------- #

def build_tier2_prompt(
    query: str,
    prompt_style: str = "chatml_scaffold",
    tokenizer: Optional[Any] = None,
) -> Tuple[str, str]:
    """Build a Tier 2 (no-RAG) generation prompt.

    Parameters
    ----------
    query
        The benchmark-style query (e.g. starting with "Answer with one of: ..."
        for FEVER; see ``detect_query_task``).
    prompt_style
        ``"chatml_scaffold"`` for decoder-only backbones (requires ``tokenizer``),
        ``"flan_t5_scaffold"`` for the legacy Flan-T5 path.
    tokenizer
        Required for ``chatml_scaffold``; ignored for ``flan_t5_scaffold``.

    Returns
    -------
    prompt_text : str
        The assembled prompt. For ``chatml_scaffold`` this ends with the
        ``<|im_start|>assistant`` generation-prompt marker produced by
        ``apply_chat_template(add_generation_prompt=True)``. For
        ``flan_t5_scaffold`` this ends with ``"Answer: <format-spec>"``.
    forced_prefix : str
        The ``"Reasoning:"`` text the caller applies either by prefill (ChatML)
        or ``decoder_input_ids`` (Flan-T5).
    """
    if prompt_style == "chatml_scaffold":
        if tokenizer is None:
            raise ValueError(
                "prompt_style='chatml_scaffold' requires a tokenizer (needed "
                "for apply_chat_template)."
            )
        return _build_chatml_tier2(query, tokenizer)
    if prompt_style == "flan_t5_scaffold":
        return _build_flan_t5_tier2(query)
    raise ValueError(
        f"Unknown prompt_style={prompt_style!r}; expected 'chatml_scaffold' "
        "or 'flan_t5_scaffold'."
    )


def build_tier3_prompt(
    query: str,
    passages: List[Tuple[str, float]],
    prompt_style: str = "chatml_scaffold",
    tokenizer: Optional[Any] = None,
) -> Tuple[str, str]:
    """Build a Tier 3 (RAG) generation prompt conditioned on retrieved passages.

    See ``build_tier2_prompt`` for parameter semantics. ``passages`` is a list
    of ``(passage_text, score)`` tuples as returned by the PassageStore; scores
    are preserved in the signature for API compatibility even though they are
    not used in the prompt text.
    """
    if prompt_style == "chatml_scaffold":
        if tokenizer is None:
            raise ValueError(
                "prompt_style='chatml_scaffold' requires a tokenizer."
            )
        return _build_chatml_tier3(query, passages, tokenizer)
    if prompt_style == "flan_t5_scaffold":
        return _build_flan_t5_tier3(query, passages)
    raise ValueError(
        f"Unknown prompt_style={prompt_style!r}; expected 'chatml_scaffold' "
        "or 'flan_t5_scaffold'."
    )


__all__ = [
    "FORCED_PREFIX",
    "SYSTEM_PROMPT",
    "detect_query_task",
    "build_tier2_prompt",
    "build_tier3_prompt",
]
