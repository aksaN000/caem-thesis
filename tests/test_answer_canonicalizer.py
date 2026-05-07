"""
tests/test_answer_canonicalizer.py
====================================
Unit tests for caem.verification.answer_canonicalizer.canonicalize_answer
and supporting helpers.

Coverage
--------
  * MCQ-letter expansion: detects "C" + a parseable Choices block in the
    query and emits "Answer: <option_text>."
  * Bare-entity wrapping: short non-declarative answer wraps as
    "Answer: <entity>."
  * Already-declarative pass-through: FEVER labels (supports/refutes),
    StrategyQA yes/no, refusal sentinels, long prose with verbs.
  * Idempotency: canonicalize_answer applied twice returns the same
    string (guard against double-wrapping "Answer: Answer: ...").
  * MCQ no-Choices-block fallback: bare letter "C" without a Choices
    block does not get wrapped as "Answer: C." (would be misleading).
"""
from __future__ import annotations

import sys


def test_canonicalize_mcq_letter_expands_to_option():
    from caem.verification.answer_canonicalizer import canonicalize_answer
    query = (
        "Question: Where does someone usually wear a watch? "
        "Choices: (A) ankle (B) shelf (C) wrist (D) ceiling (E) refrigerator\n"
        "Answer with just the multiple choice letter."
    )
    out = canonicalize_answer(query, "C")
    # Short "Answer: <option>." form mirrors the model's own output and
    # saves NLI judge tokens on the per-cycle scoring path.
    assert out.lower().startswith("answer:")
    assert "wrist" in out.lower()


def test_canonicalize_bare_entity_wraps():
    from caem.verification.answer_canonicalizer import canonicalize_answer
    out = canonicalize_answer("Question: Capital of France?", "Paris")
    assert out.lower().startswith("answer:")
    assert "paris" in out.lower()


def test_canonicalize_already_declarative_passthrough():
    from caem.verification.answer_canonicalizer import canonicalize_answer
    # FEVER label
    assert canonicalize_answer("Claim: ...", "supports") == "supports"
    # StrategyQA yes/no
    assert canonicalize_answer("Question: ...", "yes") == "yes"
    # Refusal
    out = canonicalize_answer("Question: ...", "I do not know")
    assert "I do not know" in out
    # Long prose answer with a verb (already a clause)
    long = "The Eiffel Tower was completed in 1889 in Paris, France."
    assert canonicalize_answer("Question: ...", long) == long


def test_canonicalize_idempotent():
    from caem.verification.answer_canonicalizer import canonicalize_answer
    query = "Question: Capital of France?"
    once = canonicalize_answer(query, "Paris")
    twice = canonicalize_answer(query, once)
    assert once == twice, (
        f"canonicalize must be idempotent; got {once!r} vs {twice!r}"
    )


def test_canonicalize_mcq_no_choices_block_falls_back():
    """If the answer is 'C' but the query has no parseable Choices block,
    the canonicalizer must NOT emit 'Answer: C.' (which would be misleading)."""
    from caem.verification.answer_canonicalizer import canonicalize_answer
    out = canonicalize_answer("What is C++ used for?", "C")
    assert out.lower() != "answer: c."
    assert out.lower() != "answer: c"


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:
                failures += 1
                print(f"  ERR   {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
    print(f"\nAll tests passed.")
