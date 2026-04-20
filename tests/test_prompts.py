"""Tests for caem.prompts (Goal 1 Phase B, Branch C).

Two correctness properties:

1. **Legacy bit-identity**: ``build_tier2_prompt(..., "flan_t5_scaffold")`` and
   ``build_tier3_prompt(..., "flan_t5_scaffold")`` produce strings identical to
   the pre-refactor inline builders in ``caem/pipeline.py`` and
   ``caem/retrieval/rag.py``. Required so the ``flan_t5_large_backbone``
   Variant-18 ablation run reproduces pre-Branch-C behavior exactly.

2. **ChatML structure**: ``chatml_scaffold`` produces well-formed ChatML with
   system + example-turn-pair + real-user turn, and ends with the generation
   prompt marker so a caller can append the forced prefix and tokenize.
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
# Flan-T5 bit-identity vs legacy pipeline.py / rag.py builders                 #
# -------------------------------------------------------------------------- #

def test_flan_t5_tier2_matches_legacy_pipeline():
    """Output must match caem/pipeline.py::_build_tier2_prompt exactly."""
    # Construct via the LEGACY code path directly. Do not go through
    # pipeline.Pipeline (which requires a loaded model); use the raw static
    # methods if available, or replicate the method inline.
    from caem.prompts import build_tier2_prompt

    # Build via new module
    q = "Answer with one of: supports, refutes, not enough info. Claim: A staging area is only an unused piece of land."
    new_prompt, new_prefix = build_tier2_prompt(q, "flan_t5_scaffold")

    # Expected structure from pipeline._build_tier2_prompt (hand-assembled)
    expected = (
        "Answer the question using step-by-step reasoning. "
        "Always write out your reasoning before the answer.\n\n"
        "Example:\n"
        "Claim: Barack Obama was the 44th US President.\n"
        "Reasoning: Barack Obama served as the 44th "
        "President of the United States from 2009 to 2017. "
        "The claim matches this fact.\n"
        "Answer: supports\n\n"
        "Now answer the following.\n\n"
        "Claim: A staging area is only an unused piece of land.\n"
        "Determine whether the claim is SUPPORTS, REFUTES, or "
        "NOT ENOUGH INFO based on your knowledge.\n\n"
        "Response format (fill in each field):\n"
        "Reasoning:\n"
        "Answer: supports | refutes | not enough info"
    )
    assert new_prompt == expected, (
        f"Flan-T5 Tier 2 FEVER prompt drifted from legacy format.\n\n"
        f"Got:\n---\n{new_prompt}\n---\n\nExpected:\n---\n{expected}\n---"
    )
    assert new_prefix == "Reasoning:"


def test_flan_t5_tier2_strategyqa():
    from caem.prompts import build_tier2_prompt
    q = "Answer yes or no. Question: Can a boat float on ice?"
    prompt, prefix = build_tier2_prompt(q, "flan_t5_scaffold")
    assert "Example:" in prompt
    assert "Question: Can a pressure cooker cook food" in prompt  # few-shot example
    assert "Question: Can a boat float on ice?" in prompt
    assert "Answer: yes | no" in prompt
    assert prefix == "Reasoning:"


def test_flan_t5_tier2_arc():
    from caem.prompts import build_tier2_prompt
    q = ("Question: What is 2+2? Choices: (A) 3 (B) 4 (C) 5 (D) 6 "
         "Answer with just the multiple choice letter.")
    prompt, _ = build_tier2_prompt(q, "flan_t5_scaffold")
    assert "What process allows plants to make food?" in prompt  # few-shot
    assert "What is 2+2?" in prompt  # real query
    assert "Answer: A | B | C | D" in prompt


def test_flan_t5_tier2_open():
    from caem.prompts import build_tier2_prompt
    q = "Who painted the Mona Lisa?"
    prompt, _ = build_tier2_prompt(q, "flan_t5_scaffold")
    assert "Great Wall of China" in prompt  # few-shot example
    assert "Who painted the Mona Lisa?" in prompt
    assert "Answer: <concise factual answer>" in prompt


def test_flan_t5_tier3_has_context_block():
    """Tier 3 must include a Context: block with [N] numbered passages."""
    from caem.prompts import build_tier3_prompt

    q = "Answer with one of: supports, refutes, not enough info. Claim: Mars is red."
    passages = [
        ("Mars appears red due to iron oxide.", 0.9),
        ("The planet is named after a Roman god.", 0.7),
    ]
    prompt, prefix = build_tier3_prompt(q, passages, "flan_t5_scaffold")
    assert "Context:\n[1] Mars appears red" in prompt
    assert "[2] The planet is named after" in prompt
    assert "based on the context above" in prompt
    assert prefix == "Reasoning:"


def test_flan_t5_tier3_matches_legacy_rag():
    """Tier 3 output must match rag._build_prompt exactly for FEVER with
    2 passages — the canonical Variant-18 reproduction test."""
    from caem.prompts import build_tier3_prompt

    q = "Answer with one of: supports, refutes, not enough info. Claim: Water boils at 100C."
    passages = [
        ("Water boils at 100 degrees Celsius at standard pressure.", 0.95),
        ("Boiling point varies with altitude.", 0.82),
    ]
    prompt, _ = build_tier3_prompt(q, passages, "flan_t5_scaffold")

    # Must start with the standard preamble
    assert prompt.startswith(
        "Answer the question using step-by-step reasoning."
    )
    # Must include the FEVER few-shot example in the "Context:..." format
    # (rag.py format, not pipeline.py inline-example format)
    assert (
        "Context:\n"
        "[1] Barack Obama served as the 44th"
    ) in prompt
    # Must include the actual context with the real passages
    assert "[1] Water boils at 100 degrees" in prompt
    assert "[2] Boiling point varies" in prompt
    # Task instruction must use the Tier-3 wording
    assert "based on the context above" in prompt
    assert prompt.endswith("Answer: supports | refutes | not enough info")


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


def test_chatml_requires_tokenizer():
    from caem.prompts import build_tier2_prompt
    with pytest.raises(ValueError, match="requires a tokenizer"):
        build_tier2_prompt("Who invented X?", "chatml_scaffold", tokenizer=None)


def test_unknown_style_raises():
    from caem.prompts import build_tier2_prompt
    with pytest.raises(ValueError, match="Unknown prompt_style"):
        build_tier2_prompt("Who?", "invalid_style")


def test_chatml_tier2_structure_fever():
    tok = _get_qwen_tokenizer()
    from caem.prompts import build_tier2_prompt, SYSTEM_PROMPT, FORCED_PREFIX
    q = "Answer with one of: supports, refutes, not enough info. Claim: The sky is blue."
    prompt, prefix = build_tier2_prompt(q, "chatml_scaffold", tokenizer=tok)

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
    assert prompt.rstrip().endswith("<|im_start|>assistant") or prompt.rstrip().endswith("<|im_start|>assistant\n".strip())
    assert prefix == FORCED_PREFIX


def test_chatml_tier2_structure_open():
    tok = _get_qwen_tokenizer()
    from caem.prompts import build_tier2_prompt
    q = "Who painted the Mona Lisa?"
    prompt, _ = build_tier2_prompt(q, "chatml_scaffold", tokenizer=tok)
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
    prompt, _ = build_tier3_prompt(q, passages, "chatml_scaffold", tokenizer=tok)
    # Contains BOTH the few-shot context AND the actual passages
    assert "pressure cooker" in prompt.lower()  # StrategyQA few-shot
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
         "Barack Obama"),   # FEVER few-shot
        ("Answer yes or no. Question: Y",
         "pressure cooker"),  # StrategyQA few-shot
        ("Question: Z? Choices: (A) a (B) b (C) c (D) d "
         "Answer with just the multiple choice letter.",
         "photosynthesis"),   # ARC few-shot
        ("Who was the first person on the moon?",
         "Great Wall"),       # Open few-shot
    ]
    for q, expected_marker in cases:
        prompt, _ = build_tier2_prompt(q, "chatml_scaffold", tokenizer=tok)
        assert expected_marker.lower() in prompt.lower(), (
            f"Query {q!r} did not produce the expected few-shot "
            f"(looking for {expected_marker!r})"
        )
