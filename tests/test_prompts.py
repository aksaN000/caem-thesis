"""Tests for caem.prompts (Goal 1, Branch C — ChatML only).

Branch C removed the ``flan_t5_scaffold`` style. These tests cover ChatML
structure, task detection, dispatch across the 4 benchmark families
(FEVER / StrategyQA / ARC / open-ended), and Tier 2 vs Tier 3 (no-RAG vs
RAG) prompts.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("RAYON_NUM_THREADS", "16")


# -------------------------------------------------------------------------- #
# Task detection                                                               #
# -------------------------------------------------------------------------- #

def test_detect_task_fever():
    from caem.prompts import detect_query_task
    q = "Answer with one of: supports, refutes, not enough info. Claim: The Eiffel Tower is in Paris."
    assert detect_query_task(q) == "fever"


def test_detect_task_strategyqa():
    from caem.prompts import detect_query_task
    q = "Answer yes or no. Question: Can cats fly?"
    assert detect_query_task(q) == "strategyqa"


def test_detect_task_arc():
    from caem.prompts import detect_query_task
    q = ("Question: What is 2+2? Choices: (A) 3 (B) 4 (C) 5 (D) 6 "
         "Answer with just the multiple choice letter.")
    assert detect_query_task(q) == "arc"


def test_detect_task_open_default():
    from caem.prompts import detect_query_task
    assert detect_query_task("Who invented the telephone?") == "open"


# -------------------------------------------------------------------------- #
# ChatML structure + dispatch                                                  #
# -------------------------------------------------------------------------- #

def _get_qwen_tokenizer():
    """Load Qwen tokenizer for ChatML tests. Skip if unavailable."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    except Exception as exc:
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")


def test_chatml_tier2_structure_fever():
    tok = _get_qwen_tokenizer()
    from caem.prompts import build_tier2_prompt, FORCED_PREFIX
    q = "Answer with one of: supports, refutes, not enough info. Claim: The sky is blue."
    prompt, prefix = build_tier2_prompt(q, tokenizer=tok)

    # ChatML role markers present
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt
    assert "<|im_end|>" in prompt
    # System prompt body
    assert "step-by-step" in prompt
    # Few-shot example turn pair present (FEVER example about Obama)
    assert "Barack Obama was the 44th US President" in prompt
    assert "Answer: supports" in prompt  # few-shot assistant
    # Real user turn with the claim
    assert "The sky is blue" in prompt
    # Answer format spec
    assert "supports | refutes | not enough info" in prompt
    # Ends with generation prompt marker so caller can append prefix
    stripped = prompt.rstrip()
    assert (
        stripped.endswith("<|im_start|>assistant") or
        stripped.endswith("<|im_start|>assistant\n".rstrip())
    )
    assert prefix == FORCED_PREFIX


def test_chatml_tier2_structure_open():
    tok = _get_qwen_tokenizer()
    from caem.prompts import build_tier2_prompt
    q = "Who painted the Mona Lisa?"
    prompt, _ = build_tier2_prompt(q, tokenizer=tok)
    # Open-ended few-shot (Great Wall) should appear
    assert "Great Wall of China" in prompt
    assert "Who painted the Mona Lisa?" in prompt
    # 2026-04-24: placeholder revised to short factual answer spec (G29 fix)
    assert "short factual answer" in prompt


def test_chatml_tier3_includes_real_passages():
    tok = _get_qwen_tokenizer()
    from caem.prompts import build_tier3_prompt
    q = "Answer yes or no. Question: Does water conduct electricity when pure?"
    passages = [
        ("Pure water is a poor conductor of electricity.", 0.94),
        ("Dissolved ions make water conduct.", 0.81),
    ]
    prompt, _ = build_tier3_prompt(q, passages, tokenizer=tok)
    # Contains BOTH the few-shot context AND the actual passages
    assert "pressure cooker" in prompt.lower()   # StrategyQA few-shot
    assert "Pure water is a poor conductor" in prompt  # real passage 1
    assert "Dissolved ions make water conduct" in prompt  # real passage 2
    assert "[1] Pure water" in prompt
    assert "[2] Dissolved ions" in prompt
    assert "based on the context above" in prompt


def test_chatml_dispatch_matches_task_detection():
    """Each task type gets the correct few-shot example in the ChatML prompt."""
    tok = _get_qwen_tokenizer()
    from caem.prompts import build_tier2_prompt

    cases = [
        ("Answer with one of: supports, refutes, not enough info. Claim: X",
         "Barack Obama"),     # FEVER few-shot
        ("Answer yes or no. Question: Y",
         "pressure cooker"),  # StrategyQA few-shot
        ("Question: Z? Choices: (A) a (B) b (C) c (D) d "
         "Answer with just the multiple choice letter.",
         "photosynthesis"),   # ARC few-shot
        ("Who was the first person on the moon?",
         "Great Wall"),       # Open few-shot
    ]
    for q, expected_marker in cases:
        prompt, _ = build_tier2_prompt(q, tokenizer=tok)
        assert expected_marker.lower() in prompt.lower(), (
            f"Query {q!r} did not produce the expected few-shot "
            f"(looking for {expected_marker!r})"
        )


def test_chatml_tier2_all_four_benchmarks_answer_format():
    """Each benchmark family produces its correct answer-format spec."""
    tok = _get_qwen_tokenizer()
    from caem.prompts import build_tier2_prompt

    cases = [
        ("Answer with one of: supports, refutes, not enough info. Claim: X",
         "supports | refutes | not enough info"),
        ("Answer yes or no. Question: Y",
         "yes | no"),
        ("Question: Z? Choices: (A) a (B) b (C) c (D) d "
         "Answer with just the multiple choice letter.",
         "A | B | C | D"),
        ("Who was the first person on the moon?",
         "short factual answer"),
    ]
    for q, expected_format in cases:
        prompt, _ = build_tier2_prompt(q, tokenizer=tok)
        assert expected_format in prompt, (
            f"Expected answer format {expected_format!r} not found for query {q!r}"
        )


# =========================================================================== #
# v2 Fix 10 — per-benchmark prompts (CommonsenseQA + HotpotQA)                  #
# =========================================================================== #
# CSQA = 5-choice MCQ (extends ARC at the regex level). HotpotQA = open
# factoid that requires the multi-step prompt template; queries are
# surface-identical to TriviaQA / NQ so the harness pins the task via
# detect_query_task(query, task_hint=source_benchmark).

def test_detect_query_task_auto_csqa_vs_arc():
    from caem.prompts import detect_query_task
    csqa_query = (
        "Question: Where does someone usually wear a watch? "
        "Choices: (A) ankle (B) shelf (C) wrist (D) ceiling (E) refrigerator\n"
        "Answer with just the multiple choice letter."
    )
    arc_query = (
        "Question: What process allows plants to make food? "
        "Choices: (A) digestion (B) photosynthesis (C) respiration (D) fermentation\n"
        "Answer with just the multiple choice letter."
    )
    assert detect_query_task(csqa_query) == "csqa"
    assert detect_query_task(arc_query) == "arc"


def test_detect_query_task_hint_pins_hotpotqa():
    from caem.prompts import detect_query_task
    raw = "Which actor played the lead role in the 2010 film Inception?"
    assert detect_query_task(raw) == "open"
    assert detect_query_task(raw, task_hint="hotpotqa") == "hotpotqa"
    assert detect_query_task(raw, task_hint="triviaqa") == "open"
    assert detect_query_task(raw, task_hint="commonsense_qa") == "csqa"
    # Unknown hint falls back to auto-detect
    assert detect_query_task(raw, task_hint="not_a_real_benchmark") == "open"


def test_task_spec_csqa_advertises_5_choice():
    from caem.prompts import _task_spec
    q = (
        "Question: ...? Choices: (A) ... (B) ... (C) ... (D) ... (E) ...\n"
        "Answer with just the multiple choice letter."
    )
    line, fmt = _task_spec("csqa", q, with_passages=False)
    assert fmt == "A | B | C | D | E"


def test_task_spec_hotpotqa_includes_multi_step_instruction():
    from caem.prompts import _task_spec
    line, fmt = _task_spec("hotpotqa", "Some multi-hop question", with_passages=True)
    assert (
        "combining information" in line.lower()
        or "reason through the steps" in line.lower()
    )
    assert "short factual answer" in fmt.lower()


def test_few_shot_parts_csqa_5_choice_example():
    from caem.prompts import _few_shot_parts
    _, ex_user, ex_asst = _few_shot_parts("csqa", with_passages=False)
    for label in ("(A)", "(B)", "(C)", "(D)", "(E)"):
        assert label in ex_user
    assert any(f"Answer: {L}" in ex_asst for L in "ABCDE")


def test_few_shot_parts_hotpotqa_multi_step_example():
    from caem.prompts import _few_shot_parts
    _, ex_user, ex_asst = _few_shot_parts("hotpotqa", with_passages=False)
    assert "Reasoning:" in ex_asst and "Answer:" in ex_asst
    reasoning_part = ex_asst.split("Answer:")[0]
    sentence_count = sum(1 for c in reasoning_part if c == ".")
    assert sentence_count >= 2  # multi-step chain


def test_build_tier2_prompt_threads_source_benchmark():
    from caem.prompts import build_tier2_prompt

    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            parts = []
            for m in messages:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

    raw = "Which river runs through Cairo?"
    tok = _Tok()
    plain, _ = build_tier2_prompt(raw, tok)
    hotpot, _ = build_tier2_prompt(raw, tok, source_benchmark="hotpotqa")
    # Different prompts because hotpotqa branch emits a multi-hop instruction.
    assert plain != hotpot


def test_build_tier3_prompt_threads_source_benchmark():
    from caem.prompts import build_tier3_prompt

    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            parts = []
            for m in messages:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

    raw = (
        "Question: ...? Choices: (A) a (B) b (C) c (D) d (E) e\n"
        "Answer with just the multiple choice letter."
    )
    tok = _Tok()
    plain, _ = build_tier3_prompt(raw, [], tok)
    csqa, _ = build_tier3_prompt(raw, [], tok, source_benchmark="commonsense_qa")
    # Auto-detect already returns csqa for this query; task_hint should
    # not change the result.
    assert plain == csqa
