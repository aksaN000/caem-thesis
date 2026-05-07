"""
caem/verification/answer_canonicalizer.py
============================================
v2 Fix 10 — uniform verifier-input canonicalisation.

The CAEM verifier composes 12 signals (Fix 6 + Fix 7 inclusive). Some
signals operate on the raw display answer string ("Answer: ..."
extracted from the model's output). When that display answer is a
benchmark-specific shorthand — a single MCQ letter ("C") or a bare
entity ("Paris") — many signals collapse to noise floor:

* p_entail: NLI of chain entailing "C" or "Paris" returns near-zero
* p_ground_atomic: atomic decomposition of a single token yields no
  facts
* q_a_relevance: cross-encoder on (question, "C") returns junk
* alias_overlap: entity extractor on "C" returns []

This module provides ONE canonicalisation function used by the
verifier so every signal sees the same claim shape regardless of
benchmark. Three transformation classes:

1. **MCQ letter** (CommonsenseQA, ARC-Challenge, OpenBookQA, MMLU):
   Detect "Answer: X" with X ∈ {A..E} AND a Choices: block in the
   query. Look up the option text matching X. Emit
   "The answer to the question is: <option_text>."

2. **Bare entity** (TriviaQA, NaturalQuestions, HotpotQA):
   Detect short non-declarative answer (≤ 5 tokens, no verb token,
   not a refusal sentinel). Emit
   "The answer to the question is: <entity>."

3. **Already declarative** (FEVER labels "supports/refutes/not enough
   info", StrategyQA "yes/no", longer prose answers):
   Pass through unchanged — these already carry propositional content
   that NLI can score.

The canonical form is computed ONCE at the top of UnifiedVerifier.
verify() and threaded into every signal that reads the answer string.
The raw display answer is preserved separately for benchmark EM
scoring at the eval harness — the canonicaliser does not interfere
with downstream label-extraction regexes.

Public surface
--------------
* :func:`canonicalize_answer(query, answer)` — returns the canonical
  claim string. Idempotent: calling it on an already-declarative
  answer returns the answer unchanged.
* :data:`REFUSAL_SENTINELS` — exposed so callers wanting to detect
  refusal answers without canonicalisation can reuse the same set.
"""
from __future__ import annotations

import re
from typing import Optional


# Refusal / sentinel responses left unchanged. These should never be
# wrapped — wrapping "I do not know" as "The answer is: I do not know"
# teaches NLI to entail refusals.
REFUSAL_SENTINELS = frozenset({
    "i do not know",
    "i don't know",
    "no answer",
    "unknown",
    "not enough info",
    "not enough information",
})

# Already-declarative single-word answers that should NOT be wrapped:
# yes/no for StrategyQA and the three FEVER labels carry their own
# truth value when scored by NLI.
_ALREADY_DECLARATIVE = frozenset({
    "yes", "no",
    "supports", "refutes",
    # "not enough info" is in REFUSAL_SENTINELS; treat the same.
})

# Verb tokens that signal the answer is already a clause, not a bare
# entity. Conservative whitelist; missing verbs is OK (we just skip
# wrapping for safety).
_VERB_TOKENS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had",
    "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might",
    "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't",
    "don't", "doesn't", "didn't",
    "won't", "wouldn't", "can't", "couldn't",
})

_PUNCT_STRIP_RE = re.compile(r"[\s\.\?!,:;'\"]+$")
_LEAD_PUNCT_STRIP_RE = re.compile(r"^[\s\.\?!,:;'\"]+")

_MCQ_LABEL_RE = re.compile(r"^\s*([A-Ea-e])(?:[.\)\]:]|\s*$)")
_SINGLE_LETTER_RE = re.compile(r"^\s*([A-Ea-e])\s*$")
_CHOICES_BLOCK_RE = re.compile(
    r"choices:\s*(.+?)(?:\n\s*answer\s+with|$)",
    re.IGNORECASE | re.DOTALL,
)
_OPTION_TEXT_RE = re.compile(
    r"\(\s*([A-Ea-e])\s*\)\s*([^()]+?)(?=\s*\([A-Ea-e]\)|$)",
    re.IGNORECASE | re.DOTALL,
)


def _normalised(s: str) -> str:
    return _LEAD_PUNCT_STRIP_RE.sub("", _PUNCT_STRIP_RE.sub("", s.strip())).strip()


def _looks_like_bare_entity(answer: str) -> bool:
    """True iff the answer is short, has no verb token, and is not a
    refusal / yes-no / FEVER-label sentinel.

    Also excludes single A-E letters: when MCQ-letter expansion fails
    (no Choices block in the query), we should NOT wrap "C" as
    "The answer to the question is: C." — that would be a misleading
    declarative since the letter has no propositional content on its
    own. Pass-through to Branch 3 instead.
    """
    a = _normalised(answer).lower()
    if not a:
        return False
    if a in REFUSAL_SENTINELS or a in _ALREADY_DECLARATIVE:
        return False
    if len(a) == 1 and a in {"a", "b", "c", "d", "e"}:
        return False
    tokens = a.split()
    if len(tokens) > 5:
        return False
    for tok in tokens:
        if tok in _VERB_TOKENS:
            return False
    return True


def _try_extract_mcq_letter(answer: str) -> Optional[str]:
    a = answer.strip()
    if not a:
        return None
    m = _SINGLE_LETTER_RE.match(a)
    if m:
        return m.group(1).upper()
    m = _MCQ_LABEL_RE.match(a)
    if m:
        return m.group(1).upper()
    return None


def _extract_choices_block(query: str) -> Optional[str]:
    m = _CHOICES_BLOCK_RE.search(query)
    if not m:
        return None
    return m.group(1).strip()


def _option_text_for(query: str, letter: str) -> Optional[str]:
    """Look up the option text for ``letter`` from the query's Choices block."""
    block = _extract_choices_block(query)
    if not block:
        return None
    target = letter.upper()
    for m in _OPTION_TEXT_RE.finditer(block):
        if m.group(1).upper() == target:
            text = m.group(2).strip()
            return text or None
    return None


def canonicalize_answer(query: str, answer: str) -> str:
    """Return the canonical claim form of ``answer`` for verifier scoring.

    Three branches:

    1. MCQ letter ("A" .. "E") + a parsable Choices block in the query:
       expand to "The answer to the question is: <option_text>."
    2. Bare entity (short, no verb token, not a sentinel): wrap as
       "The answer to the question is: <entity>."
    3. Otherwise (already-declarative / refusal / long prose):
       return ``answer`` unchanged after light whitespace cleanup.

    The function is idempotent on its own output — calling it twice
    yields the same string.
    """
    raw = (answer or "").strip()
    if not raw:
        return raw

    # Branch 1: MCQ letter substitution
    letter = _try_extract_mcq_letter(raw)
    if letter is not None:
        opt = _option_text_for(query or "", letter)
        if opt is not None:
            return f"The answer to the question is: {opt.strip().rstrip('.')}."
        # Letter detected but no choices block parseable — fall through
        # rather than emitting a bare letter wrapped in declarative
        # syntax (would be misleading).

    # Branch 2: bare-entity wrapping
    if _looks_like_bare_entity(raw):
        ent = _normalised(raw)
        if ent:
            return f"The answer to the question is: {ent}."

    # Branch 3: pass-through for already-declarative content
    return raw
