"""
tests/test_entity_expansion_scorer.py
=======================================
Unit tests for caem.verification.entity_expansion_scorer.

The scorer is kept as an ablation-research module — it wraps bare
factoid entities in declarative "Answer: <X>." form before feeding them
to the verifier's NLI judge. The live verifier path achieves the same
effect via answer_canonicalizer; tests here pin the scorer's standalone
contract so ablation studies that re-enable it stay reproducible.

Coverage
--------
  * Declines non-open task families (FEVER, StrategyQA, ARC, CSQA)
  * Accepts open + bare-entity answers (TriviaQA, NQ, HotpotQA shape)
  * Wraps the bare entity in "Answer: <X>." before NLI scoring
  * Declines long declarative answers
  * Declines refusal sentinels ("I do not know", etc.)
"""
from __future__ import annotations

import sys


def test_entity_expansion_declines_non_open_tasks():
    from caem.verification.entity_expansion_scorer import EntityExpansionScorer
    nli_calls = []

    def nli_fn(p, h):
        nli_calls.append((p, h))
        return 0.7

    def detect_fn(q):
        return "fever"

    scorer = EntityExpansionScorer(nli_fn, detect_fn)
    out = scorer.score("Some FEVER claim", "supports", ["passage"])
    assert out is None
    assert nli_calls == []  # never called NLI


def test_entity_expansion_accepts_open_bare_entity():
    from caem.verification.entity_expansion_scorer import EntityExpansionScorer
    nli_calls = []

    def nli_fn(p, h):
        nli_calls.append((p, h))
        return 0.65

    def detect_fn(q):
        return "open"

    scorer = EntityExpansionScorer(nli_fn, detect_fn)
    out = scorer.score(
        "Question: capital of France?",
        "Paris",
        ["Paris is the capital city of France.", "France is in Europe."],
    )
    assert out is not None
    p_max, p_mean = out
    assert 0.0 <= p_mean <= p_max <= 1.0
    # Scorer must wrap the bare entity in "Answer: <X>." form.
    assert all(h.lower().startswith("answer:") for _, h in nli_calls), (
        f"hypothesis should start with 'Answer:'; got {[h for _, h in nli_calls]}"
    )


def test_entity_expansion_declines_long_answer():
    from caem.verification.entity_expansion_scorer import EntityExpansionScorer
    scorer = EntityExpansionScorer(lambda p, h: 0.5, lambda q: "open")
    out = scorer.score(
        "...",
        "The Eiffel Tower was completed in 1889 by Gustave Eiffel for the World's Fair.",
        ["passage"],
    )
    assert out is None  # long declarative is not bare-entity


def test_entity_expansion_declines_refusal():
    from caem.verification.entity_expansion_scorer import EntityExpansionScorer
    scorer = EntityExpansionScorer(lambda p, h: 0.5, lambda q: "open")
    out = scorer.score("...", "I do not know", ["passage"])
    assert out is None


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
