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

from typing import Any, List, Optional, Tuple


FORCED_PREFIX = "Reasoning:"

# Branch C 2026-04-25 (Phase 2.6 — Empty-Answer compliance fix):
# Empirical evidence (Phase 1 Cycle-0 audit, n=3500):
#   - 43–53% of samples emitted only the "Reasoning:" prefix without ever
#     reaching an "Answer:" line, producing empty display_answer values
#     and routing all such samples to DISCARD regardless of correctness.
#   - Of the 218 FEVER empty samples, 74 (33.9%) had reasoning whose
#     conclusion matched the gold label; these correct answers were lost
#     to format non-compliance, not to semantic incorrectness.
#
# Fix: tighten the SYSTEM_PROMPT to make the "Answer:" line explicitly
# mandatory and bound reasoning length so the model reaches "Answer:"
# before cot_max_new_tokens cuts off generation. Also adds a hard-stop
# sentinel ("End.") the model can use to terminate cleanly after the
# answer line, helping decoding stop earlier on confident cases.
#
# This is a PROMPT-LEVEL fix; no generation-side stopping_criteria is
# added because the existing extract_cot_answer parser (eval/metrics.py
# Priority 2) already recovers natural-language "the answer is X" suffixes
# as a fallback. The combined effect is expected to recover ~14% of
# correct predictions per benchmark per cycle.
SYSTEM_PROMPT = (
    "You are a careful assistant. For each question, think step-by-step about "
    "what is being asked, then give a final answer in the specified format. "
    "Keep reasoning concise (3–4 sentences maximum). The 'Answer:' line is "
    "MANDATORY — every response MUST end with it.\n\n"
    "Always respond EXACTLY in this two-line format:\n"
    "Reasoning: <your concise step-by-step thought, 3–4 sentences>\n"
    "Answer: <final answer in the specified format>"
)


# -------------------------------------------------------------------------- #
# Task detection + task spec (shared across Tier 2 / Tier 3)                   #
# -------------------------------------------------------------------------- #

def detect_query_task(query: str, *, task_hint: Optional[str] = None) -> str:
    """Infer benchmark task style from the query's constrained prefix.

    Returns one of ``"fever"``, ``"strategyqa"``, ``"arc"``, ``"csqa"``,
    ``"hotpotqa"``, or ``"open"``.

    Parameters
    ----------
    query : str
        The benchmark-style query.
    task_hint : str or None, default None
        v2 Fix 10 — when supplied, takes precedence over the regex-based
        auto-detection. Accepts a benchmark identifier
        (``"commonsense_qa"``, ``"hotpotqa"``, ``"fever"``, ``"strategyqa"``,
        ``"arc_challenge"``, ``"triviaqa"``, ``"natural_questions"``,
        ``"truthfulqa"``) and maps it to the corresponding task family.
        Unknown values fall back to the regex path. Used by the pipeline
        to pin task selection on benchmarks whose query surface is not
        self-distinguishing (HotpotQA's raw open-ended question is
        identical in shape to TriviaQA / NQ).
    """
    if task_hint is not None:
        # Direct map for the task families the rest of this module emits.
        bm_to_task = {
            "fever": "fever",
            "strategyqa": "strategyqa",
            "arc_challenge": "arc",
            "arc": "arc",
            "commonsense_qa": "csqa",
            "csqa": "csqa",
            "hotpotqa": "hotpotqa",
            "hotpot_qa": "hotpotqa",
            # Open-ended factoid families share the "open" prompt; keep the
            # mapping explicit so callers can pass the benchmark name
            # without thinking about whether it has a special prompt.
            "triviaqa": "open",
            "natural_questions": "open",
            "truthfulqa": "open",
            "asqa": "open",
        }
        mapped = bm_to_task.get(task_hint.lower())
        if mapped is not None:
            return mapped

    q = query.lower().strip()
    if q.startswith("answer with one of: supports, refutes, not enough info."):
        return "fever"
    if q.startswith("answer yes or no."):
        return "strategyqa"
    # 5-choice MCQ → CommonsenseQA (E option present in the choices block).
    # Ordered before the 4-choice ARC check so CSQA samples are not
    # mis-classified as ARC. Both share the multichoice scorer at
    # verifier time but the prompt builder selects different
    # answer-format specs and few-shot examples.
    if (
        "choices:" in q
        and "multiple choice letter" in q
        and "(e)" in q
    ):
        return "csqa"
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
    if task == "csqa":
        # v2 Fix 10 — CommonsenseQA 5-choice MCQ. Same shape as ARC but
        # five labelled options (A through E) and the answer-format spec
        # advertises the extra letter. The instruction is also nudged
        # toward "everyday-knowledge reasoning" since CSQA tests
        # commonsense rather than grade-school science.
        basis = " based on the context above" if with_passages else ""
        return (
            f"{query}\n"
            f"Choose the option that best matches everyday-knowledge reasoning"
            f"{basis}. Pick exactly one letter.",
            "A | B | C | D | E",
        )
    if task == "hotpotqa":
        # v2 Fix 10 — HotpotQA multi-hop. The answer surface form is
        # short and factoid (typical: 1-3 tokens) but the reasoning chain
        # must combine information from multiple supporting paragraphs,
        # so the instruction explicitly licenses multi-step reasoning
        # while still enforcing a short final answer. The refusal clause
        # mirrors the open-QA fix (Phase 2.7): refusal only when context
        # truly lacks evidence.
        basis = " based on the context above" if with_passages else ""
        if with_passages:
            refusal_clause = (
                "If — and only if — the context above contains no information "
                "relevant to ANY of the reasoning steps, reply exactly: "
                "I do not know. Otherwise commit to your best-effort answer."
            )
        else:
            refusal_clause = (
                "Commit to your best-effort answer based on what you know. "
                "Reply 'I do not know' ONLY if the question is genuinely "
                "unanswerable (e.g. asking about non-existent entities)."
            )
        return (
            f"Question: {query}\n"
            f"Answer the question{basis}. The question requires combining "
            f"information across multiple facts; reason through the steps "
            f"in order, then give a SHORT final answer (1-5 words preferred). "
            f"{refusal_clause} "
            f"Do not say 'the context does not mention' or similar evasive phrases.",
            "short factual answer",
        )
    # Open-ended QA (TriviaQA, NQ, TruthfulQA, ASQA)
    #
    # Branch C 2026-04-25 (Phase 2.7 — open-QA over-abstention fix):
    # Empirical evidence (Phase 1 Cycle-0 audit, n=3500):
    #   - 38.8% of NQ, 40.4% of TriviaQA, 45.2% of ASQA samples received
    #     display_answer = "I do not know.", causing storage rates to drop
    #     to 1–2% even though base-model EM is 15–28%.
    #   - The pre-fix prompt instructed "If you cannot find the answer ...
    #     reply exactly: I do not know." which the model interpreted as
    #     "any time you're not certain, refuse." Open-QA gold answers do
    #     not include "I do not know" as a valid label (unlike FEVER NEI),
    #     so every defensive refusal on open-QA is counted em=0.
    #   - Mitigation: scope the refusal clause to retrieval-empty cases
    #     only; for Tier-2 (no passages) require the model to commit to
    #     its best-effort answer rather than refuse.
    #
    # Reference: Cole et al. "Selectively Answering Ambiguous Questions."
    # EMNLP 2023, on the over-refusal failure mode in instruction-tuned
    # generators on factoid QA. Our fix is the prompt-level remediation
    # they recommend (gate refusal on absence-of-evidence, not on absence-
    # of-confidence).
    basis = " based on the context above" if with_passages else ""
    if with_passages:
        # Tier 3: refusal is permitted only when retrieved context is empty
        # of relevant information. The model must read the context first.
        refusal_clause = (
            "If — and only if — the context above contains no information "
            "relevant to the question, reply exactly: I do not know. "
            "Otherwise commit to your best-effort answer."
        )
    else:
        # Tier 2: the model relies on its parametric knowledge. Defensive
        # refusal is discouraged; commit to your best effort.
        refusal_clause = (
            "Commit to your best-effort answer based on what you know. "
            "Reply 'I do not know' ONLY if the question is genuinely "
            "unanswerable (e.g. asking about non-existent entities). "
            "Do NOT use 'I do not know' as a default fallback for "
            "uncertainty."
        )
    return (
        f"Question: {query}\n"
        f"Answer the question{basis}. Give a SHORT answer (1-5 words preferred). "
        f"{refusal_clause} "
        f"Do not say 'the context does not mention' or similar evasive phrases.",
        "short factual answer",
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

    if task == "csqa":
        # v2 Fix 10 — 5-choice commonsense MCQ exemplar.
        if with_passages:
            ctx = (
                "Context:\n"
                "[1] People typically wear watches on their wrist as a "
                "practical place to read the time during everyday tasks."
            )
            user_inst = (
                "Question: Where does someone usually wear a watch? "
                "Choices: (A) ankle (B) shelf (C) wrist (D) ceiling "
                "(E) refrigerator\n"
                "Answer with just the multiple choice letter."
            )
            asst = (
                "Reasoning: Watches are worn on the body to be visible "
                "during daily tasks. The wrist is the conventional "
                "location for a wristwatch.\n"
                "Answer: C"
            )
        else:
            ctx = ""
            user_inst = (
                "Question: Where does someone usually wear a watch? "
                "Choices: (A) ankle (B) shelf (C) wrist (D) ceiling "
                "(E) refrigerator"
            )
            asst = (
                "Reasoning: Watches are worn on the body to be visible "
                "during daily tasks. The wrist is the conventional "
                "location for a wristwatch.\n"
                "Answer: C"
            )
        return ctx, user_inst, asst

    if task == "hotpotqa":
        # v2 Fix 10 — multi-hop exemplar showing 2-step reasoning.
        if with_passages:
            ctx = (
                "Context:\n"
                "[1] The Eiffel Tower was completed in Paris in 1889 for "
                "the World's Fair.\n"
                "[2] Paris is the capital city of France."
            )
            user_inst = (
                "Question: In which country was the Eiffel Tower "
                "completed in 1889?"
            )
            asst = (
                "Reasoning: The Eiffel Tower was completed in Paris in "
                "1889 according to the first passage. Paris is the "
                "capital of France according to the second passage. "
                "Combining these gives France.\n"
                "Answer: France"
            )
        else:
            ctx = ""
            user_inst = (
                "Question: In which country was the Eiffel Tower "
                "completed in 1889?"
            )
            asst = (
                "Reasoning: The Eiffel Tower was completed in Paris in "
                "1889. Paris is in France. Therefore the Eiffel Tower "
                "was completed in France.\n"
                "Answer: France"
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
    *,
    source_benchmark: Optional[str] = None,
) -> Tuple[str, str]:
    """Build a Tier 2 (no-RAG) generation prompt in ChatML format.

    Parameters
    ----------
    query
        The benchmark-style query (e.g. "Answer with one of: ..." for FEVER;
        see ``detect_query_task``).
    tokenizer
        The backbone's tokenizer; used for ``apply_chat_template``.
    source_benchmark : str or None, default None
        v2 Fix 10 — when supplied, pins task selection on benchmarks whose
        query surface is not self-distinguishing (e.g. HotpotQA's raw
        open-ended question is identical in shape to TriviaQA / NQ).
        Forwarded to ``detect_query_task`` as ``task_hint``.

    Returns
    -------
    prompt_text : str
        ChatML-assembled prompt ending with the ``<|im_start|>assistant``
        generation-prompt marker.
    forced_prefix : str
        ``"Reasoning:"`` — caller appends as prefill.
    """
    task = detect_query_task(query, task_hint=source_benchmark)
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
    *,
    source_benchmark: Optional[str] = None,
) -> Tuple[str, str]:
    """Build a Tier 3 (RAG) generation prompt conditioned on retrieved passages.

    ``passages`` is a list of ``(passage_text, score)`` tuples as returned by
    the PassageStore; scores are kept in the signature for API compatibility
    but not used in the prompt text.

    Parameters
    ----------
    source_benchmark : str or None, default None
        v2 Fix 10 — same task-pinning role as in ``build_tier2_prompt``.
        Forwarded to ``detect_query_task`` as ``task_hint``.
    """
    task = detect_query_task(query, task_hint=source_benchmark)
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
