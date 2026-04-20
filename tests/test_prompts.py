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
    assert "<concise factual answer>" in prompt


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
         "<concise factual answer>"),
    ]
    for q, expected_format in cases:
        prompt, _ = build_tier2_prompt(q, tokenizer=tok)
        assert expected_format in prompt, (
            f"Expected answer format {expected_format!r} not found for query {q!r}"
        )
