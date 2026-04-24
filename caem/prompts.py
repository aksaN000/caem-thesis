"""
caem/prompts.py
=================
Benchmark-aware prompt builders for Tier 2 (no-RAG) and Tier 3 (RAG)
generation on decoder-only ChatML-compatible backbones (Qwen, Gemma,
Llama, Phi, Mistral instruction-tuned variants).

**Branch C decision (2026-04-22)**: the ``flan_t5_scaffold`` prompt style
has been removed. Branch C is decoder-only; ChatML is the sole prompt
format. For legacy T5-format prompts, check out the ``main`` branch. See
branch_C.md §"T5 removal (2026-04-22)" for rationale.

Public API
----------
    build_tier2_prompt(query, tokenizer) -> (prompt_text, forced_prefix)
    build_tier3_prompt(query, passages, tokenizer) -> (prompt_text, forced_prefix)

Returns a ``(full_prompt_text, forced_prefix)`` tuple:

- ``full_prompt_text``: ChatML-assembled prompt ending with the
  ``<|im_start|>assistant`` generation-prompt marker produced by
  ``apply_chat_template(add_generation_prompt=True)``.
- ``forced_prefix``: the ``"Reasoning:"`` text the caller appends as prefill
  to force the output to begin with scaffolded Chain-of-Thought reasoning.

Scaffolded CoT preserves the pattern proven in the Phase-1a pilot (uniform
Reasoning → Answer format + single worked example per task). Only the
envelope changes — ChatML turns + system prompt instead of flat-text.
"""

from __future__ import annotations

from typing import Any, List, Tuple


FORCED_PREFIX = "Reasoning:"

SYSTEM_PROMPT = (
    "You are a careful assistant. For each question, think step-by-step about "
    "what is being asked, then give a final answer in the specified format. "
    "Always respond exactly as:\n"
    "Reasoning: <your step-by-step thought>\n"
    "Answer: <final answer>"
)


# -------------------------------------------------------------------------- #
# Task detection + task spec (shared across Tier 2 / Tier 3)                   #
# -------------------------------------------------------------------------- #

def detect_query_task(query: str) -> str:
    """Infer benchmark task style from the query's constrained prefix.

    Returns one of ``"fever"``, ``"strategyqa"``, ``"arc"``, or ``"open"``.
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

    ``with_passages=True`` uses Tier 3 wording ("based on the context above");
    ``False`` uses Tier 2 wording ("based on your knowledge").
    """
    if task == "fever":
        claim = _extract_after_token(query, "Claim:")
        basis = "the context above" if with_passages else "your knowledge"
        return (
            f"Claim: {claim}\n"
            f"Determine whether the claim is SUPPORTS, REFUTES, or "
            f"NOT ENOUGH INFO based on {basis}. "
            f"Choose 'not enough info' ONLY when {basis} contains no "
            f"relevant information about the claim. If {basis} contains "
            f"evidence, commit to 'supports' or 'refutes'.",
            "supports | refutes | not enough info",
        )
    if task == "strategyqa":
        q_text = _extract_after_token(query, "Question:")
        basis_suffix = " based on the context above" if with_passages else ""
        return (
            f"Question: {q_text}\n"
            f"Answer the question with yes or no{basis_suffix}.",
            "yes | no",
        )
    if task == "arc":
        basis = " based on the context above" if with_passages else ""
        return (
            f"{query}\n"
            f"Choose the correct answer from the listed choices{basis}.",
            "A | B | C | D",
        )
    # Open-ended QA (TriviaQA, NQ, TruthfulQA)
    basis = " based on the context above" if with_passages else ""
    return (
        f"Question: {query}\n"
        f"Answer the question{basis}. Give a SHORT answer (1-5 words preferred). "
        f"If you cannot find the answer{basis}, reply exactly: I do not know. "
        f"Do not say 'the context does not mention' or similar evasive phrases.",
        "short factual answer (or 'I do not know')",
    )


# -------------------------------------------------------------------------- #
# Few-shot example content                                                      #
# -------------------------------------------------------------------------- #

def _few_shot_parts(task: str, with_passages: bool) -> Tuple[str, str, str]:
    """Return ``(context_block, user_instance, assistant_response)`` for the
    worked example. ``context_block`` is empty when ``with_passages=False``.
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
# ChatML rendering                                                             #
# -------------------------------------------------------------------------- #

def _build_chatml(
    messages: List[dict],
    tokenizer: Any,
) -> str:
    """Render a messages list to ChatML text with ``add_generation_prompt=True``.

    Uses ``tokenizer.apply_chat_template`` when available (standard for
    Qwen/Gemma/Llama-3). Falls back to a manual ChatML concatenation if the
    tokenizer lacks the method — rare; most modern instruction-tuned models
    ship with a chat template.
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
    # Manual ChatML fallback (Qwen/Llama/Gemma share this syntax)
    parts: List[str] = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# -------------------------------------------------------------------------- #
# Public API                                                                   #
# -------------------------------------------------------------------------- #

def build_tier2_prompt(
    query: str,
    tokenizer: Any,
) -> Tuple[str, str]:
    """Build a Tier 2 (no-RAG) generation prompt in ChatML format.

    Parameters
    ----------
    query
        The benchmark-style query (e.g. "Answer with one of: ..." for FEVER;
        see ``detect_query_task``).
    tokenizer
        The backbone's tokenizer; used for ``apply_chat_template``.

    Returns
    -------
    prompt_text : str
        ChatML-assembled prompt ending with the ``<|im_start|>assistant``
        generation-prompt marker.
    forced_prefix : str
        ``"Reasoning:"`` — caller appends as prefill.
    """
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
    return _build_chatml(messages, tokenizer), FORCED_PREFIX


def build_tier3_prompt(
    query: str,
    passages: List[Tuple[str, float]],
    tokenizer: Any,
) -> Tuple[str, str]:
    """Build a Tier 3 (RAG) generation prompt conditioned on retrieved passages.

    ``passages`` is a list of ``(passage_text, score)`` tuples as returned by
    the PassageStore; scores are kept in the signature for API compatibility
    but not used in the prompt text.
    """
    task = detect_query_task(query)
    task_line, answer_format = _task_spec(task, query, with_passages=True)
    ex_ctx, ex_user, ex_asst = _few_shot_parts(task, with_passages=True)

    # Actual context block from retrieved passages
    ctx_lines = ["Context:"]
    for i, (passage, _score) in enumerate(passages, start=1):
        ctx_lines.append(f"[{i}] {passage}")
    actual_context = "\n".join(ctx_lines)

    # Example's user turn carries example context+claim; real user turn carries
    # actual context + task line.
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
    return _build_chatml(messages, tokenizer), FORCED_PREFIX


__all__ = [
    "FORCED_PREFIX",
    "SYSTEM_PROMPT",
    "detect_query_task",
    "build_tier2_prompt",
    "build_tier3_prompt",
]
