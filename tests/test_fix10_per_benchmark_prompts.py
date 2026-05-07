"""
tests/test_fix10_per_benchmark_prompts.py
============================================
Smoke test for v2 Fix 10 — per-benchmark prompts + uniform verifier
input canonicalisation + 5-choice multi-choice scorer support.

Verifies:
  1. detect_query_task auto-detects CSQA (5-choice) vs ARC (4-choice)
     by query content.
  2. detect_query_task respects an explicit task_hint and pins
     ambiguous benchmarks (HotpotQA vs TriviaQA both produce
     "open"-shaped queries by default).
  3. _task_spec emits "A | B | C | D | E" for csqa; emits the
     multi-hop instruction with refusal scope for hotpotqa.
  4. _few_shot_parts emits a 5-choice example for csqa and a
     2-step reasoning chain example for hotpotqa.
  5. build_tier2_prompt / build_tier3_prompt accept source_benchmark
     and forward it correctly.
  6. extract_arc_label(text, n_choices=5) accepts E / digit 5 for
     CSQA; default n_choices=4 still rejects E.
  7. canonicalize_answer expands MCQ letters using the query's
     Choices block.
  8. canonicalize_answer wraps bare entities; leaves declarative
     answers and refusals unchanged; idempotent.
  9. EntityExpansionScorer correctly declines non-open tasks and
     produces a (max, mean) tuple on bare-entity open answers.
"""
from __future__ import annotations

import sys


# ---------------------------------------------------------------- #
# 1. detect_query_task                                              #
# ---------------------------------------------------------------- #

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
    assert detect_query_task(csqa_query) == "csqa", (
        "5-choice query should auto-detect as csqa"
    )
    assert detect_query_task(arc_query) == "arc", (
        "4-choice query should auto-detect as arc"
    )


def test_detect_query_task_hint_pins_hotpotqa():
    from caem.prompts import detect_query_task
    # HotpotQA queries look identical to TriviaQA at the surface — both
    # are open factoid questions. Without a hint, both fall through to
    # "open"; with a hint, the dispatcher returns "hotpotqa".
    raw = "Which actor played the lead role in the 2010 film Inception?"
    assert detect_query_task(raw) == "open"
    assert detect_query_task(raw, task_hint="hotpotqa") == "hotpotqa"
    assert detect_query_task(raw, task_hint="triviaqa") == "open"
    assert detect_query_task(raw, task_hint="commonsense_qa") == "csqa"
    # Unknown hint falls back to auto-detect
    assert detect_query_task(raw, task_hint="not_a_real_benchmark") == "open"


# ---------------------------------------------------------------- #
# 2. _task_spec                                                     #
# ---------------------------------------------------------------- #

def test_task_spec_csqa_advertises_5_choice():
    from caem.prompts import _task_spec
    q = "Question: ...? Choices: (A) ... (B) ... (C) ... (D) ... (E) ...\nAnswer with just the multiple choice letter."
    line, fmt = _task_spec("csqa", q, with_passages=False)
    assert "A | B | C | D | E" == fmt, (
        f"csqa answer-format should expose all 5 letters; got {fmt}"
    )


def test_task_spec_hotpotqa_includes_multi_step_instruction():
    from caem.prompts import _task_spec
    line, fmt = _task_spec("hotpotqa", "Some multi-hop question", with_passages=True)
    assert "combining information" in line.lower() or "reason through the steps" in line.lower(), (
        "hotpotqa task_line should explicitly license multi-step reasoning"
    )
    assert "short factual answer" in fmt.lower()


# ---------------------------------------------------------------- #
# 3. _few_shot_parts                                                #
# ---------------------------------------------------------------- #

def test_few_shot_parts_csqa_5_choice_example():
    from caem.prompts import _few_shot_parts
    _, ex_user, ex_asst = _few_shot_parts("csqa", with_passages=False)
    # Five labelled options (A) through (E)
    for label in ("(A)", "(B)", "(C)", "(D)", "(E)"):
        assert label in ex_user, f"csqa few-shot user turn missing {label}"
    # Assistant must answer with one of the letters
    assert any(f"Answer: {L}" in ex_asst for L in "ABCDE")


def test_few_shot_parts_hotpotqa_multi_step_example():
    from caem.prompts import _few_shot_parts
    _, ex_user, ex_asst = _few_shot_parts("hotpotqa", with_passages=False)
    # Multi-step reasoning chain in the assistant turn
    assert "Reasoning:" in ex_asst and "Answer:" in ex_asst
    # 2-step chain has at least 2 sentences in Reasoning
    reasoning_part = ex_asst.split("Answer:")[0]
    sentence_count = sum(1 for c in reasoning_part if c == ".")
    assert sentence_count >= 2, (
        f"hotpotqa few-shot Reasoning should be multi-step; got {sentence_count} sentences"
    )


# ---------------------------------------------------------------- #
# 4. build_tier2/3_prompt — source_benchmark threading              #
# ---------------------------------------------------------------- #

def _stub_tokenizer():
    """Tokenizer stub that returns ChatML directly (no apply_chat_template)."""
    class _T:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            parts = []
            for m in messages:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)
    return _T()


def test_build_tier2_prompt_threads_source_benchmark():
    from caem.prompts import build_tier2_prompt
    raw = "Which river runs through Cairo?"
    tok = _stub_tokenizer()
    plain, _ = build_tier2_prompt(raw, tok)
    hotpot, _ = build_tier2_prompt(raw, tok, source_benchmark="hotpotqa")
    # Different prompts because hotpotqa branch emits a multi-hop instruction
    assert plain != hotpot, (
        "hotpotqa source_benchmark should produce a different prompt than open default"
    )


def test_build_tier3_prompt_threads_source_benchmark():
    from caem.prompts import build_tier3_prompt
    raw = (
        "Question: ...? Choices: (A) a (B) b (C) c (D) d (E) e\n"
        "Answer with just the multiple choice letter."
    )
    tok = _stub_tokenizer()
    plain, _ = build_tier3_prompt(raw, [], tok)
    csqa, _ = build_tier3_prompt(raw, [], tok, source_benchmark="commonsense_qa")
    # Both should detect csqa anyway (5-letter pattern), so they should
    # be equal (the task_hint just confirms what auto-detect produces).
    # If they're equal, we know source_benchmark is at least not breaking
    # anything.
    assert plain == csqa, (
        "csqa task auto-detected and via task_hint should produce identical prompts"
    )


# ---------------------------------------------------------------- #
# 5. extract_arc_label — 5-choice support                           #
# ---------------------------------------------------------------- #

def test_extract_arc_label_5_choice():
    from eval.metrics import extract_arc_label
    # n_choices=4 default rejects E
    assert extract_arc_label("Reasoning: ... Answer: E", n_choices=4) != "E"
    # n_choices=5 accepts E
    assert extract_arc_label("Reasoning: ... Answer: E", n_choices=5) == "E"
    # Lowercase normalised to upper
    assert extract_arc_label("Answer: e", n_choices=5) == "E"
    # Digit fallback for 5
    assert extract_arc_label("the answer is 5", n_choices=5) == "E"
    # n_choices=4 default still works for A-D
    assert extract_arc_label("Answer: B", n_choices=4) == "B"
    assert extract_arc_label("Answer: B") == "B"  # default n_choices=4


# ---------------------------------------------------------------- #
# 6. canonicalize_answer                                            #
# ---------------------------------------------------------------- #

def test_canonicalize_mcq_letter_expands_to_option():
    from caem.verification.answer_canonicalizer import canonicalize_answer
    query = (
        "Question: Where does someone usually wear a watch? "
        "Choices: (A) ankle (B) shelf (C) wrist (D) ceiling (E) refrigerator\n"
        "Answer with just the multiple choice letter."
    )
    out = canonicalize_answer(query, "C")
    assert "wrist" in out.lower()
    assert "the answer to the question is" in out.lower()


def test_canonicalize_bare_entity_wraps():
    from caem.verification.answer_canonicalizer import canonicalize_answer
    out = canonicalize_answer("Question: Capital of France?", "Paris")
    assert "the answer to the question is" in out.lower()
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
    # Long prose answer with verb
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
    the canonicalizer should NOT emit 'The answer to the question is: C.'
    (which would be misleading). It returns the raw answer."""
    from caem.verification.answer_canonicalizer import canonicalize_answer
    out = canonicalize_answer("What is C++ used for?", "C")
    # Either passthrough or refuses to wrap. NOT "The answer to the question is: C."
    assert "the answer to the question is: c." not in out.lower()


# ---------------------------------------------------------------- #
# 7. EntityExpansionScorer                                          #
# ---------------------------------------------------------------- #

def test_entity_expansion_declines_non_open_tasks():
    from caem.verification.entity_expansion_scorer import EntityExpansionScorer
    # Stubs
    nli_calls = []
    def nli_fn(p, h):
        nli_calls.append((p, h))
        return 0.7
    def detect_fn(q):
        return "fever"
    scorer = EntityExpansionScorer(nli_fn, detect_fn)
    out = scorer.score("Some FEVER claim", "supports", ["passage"])
    assert out is None
    assert nli_calls == []  # Should never call NLI


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
    # Scorer must have wrapped the bare entity
    assert all("the answer to the question is" in h.lower() for _, h in nli_calls)


def test_entity_expansion_declines_long_answer():
    from caem.verification.entity_expansion_scorer import EntityExpansionScorer
    scorer = EntityExpansionScorer(lambda p, h: 0.5, lambda q: "open")
    out = scorer.score(
        "...",
        "The Eiffel Tower was completed in 1889 by Gustave Eiffel for the World's Fair.",
        ["passage"],
    )
    # Long declarative answer is not bare-entity → decline
    assert out is None


def test_entity_expansion_declines_refusal():
    from caem.verification.entity_expansion_scorer import EntityExpansionScorer
    scorer = EntityExpansionScorer(lambda p, h: 0.5, lambda q: "open")
    out = scorer.score("...", "I do not know", ["passage"])
    assert out is None


# ---------------------------------------------------------------- #
# Run as script                                                     #
# ---------------------------------------------------------------- #

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
