"""
caem/verification/entity_head.py
==================================
v2 Fix 7 — entity_head_consistency signal.

Pairwise agreement on the answer's head noun across the M=3
self-consistency chains the verifier already generates for s_avg and
p_entail. Zero extra compute: the chains are reused.

Why a separate signal
---------------------
``s_avg`` measures cosine similarity over chain embeddings — it tells
us whether the chains say similar things, not whether they identify
the same entity. ``p_entail`` measures NLI entailment of each chain
against the display answer — it can be high when one chain affirms a
statement that happens to overlap with the display answer's wording
even if the chains disagree on the named subject. Neither signal
catches the failure mode where the model emits three confident chains
naming three different entities (e.g. "Bill Clinton", "George Bush",
"Barack Obama" for a question about a 1990s president) and the verifier
selects one of them as the display answer.

``entity_head_consistency`` directly measures: of all unordered chain
pairs (i, j), how many extract the same head noun? Score is in
``[0, 1]``; 1.0 means every chain pair agrees on the answer entity,
0.0 means every pair disagrees.

Public surface
--------------
* :func:`extract_head_noun` — heuristic head-noun extraction. Picks the
  rightmost capitalised noun phrase by default (the typical English
  head); falls back to the longest title-case run, then to the first
  alphanumeric token. Returns the empty string when no candidate is
  found, which the scorer treats as "not extracted" (excluded from
  pair counts).
* :func:`score_entity_head_consistency` — pairwise agreement over a
  list of chains. Returns 0.5 (neutral) when fewer than 2 chains
  contain extractable head nouns, matching the missing-signal
  convention used by ``q_a_relevance`` and ``alias_overlap``.

These functions deliberately avoid spaCy / stanza so the verifier hot
path keeps no new heavy dependencies. The regex-based extractor catches
the common factoid-QA cases that the v2 panel covers (FEVER, TriviaQA,
HotpotQA, NaturalQuestions); CommonsenseQA's 5-choice scoring already
produces a clean A-E label that the extractor preserves verbatim.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Iterable, List, Optional


# Reasonable English stop-words for the head-noun heuristic. Kept small
# on purpose — the goal is to suppress trailing "the/a/an" after a
# capital span, not full POS-tagging.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "as", "by",
})

# Title-Case run, allowing internal stop-connectors ("of", "the", "de",
# "la", "le", "von", "van") so multi-word entity heads survive the
# extraction. Matches "Bill Clinton", "United States of America",
# "Frédéric de la Tour", "King of England", etc.
_TITLE_RUN_RE = re.compile(
    r"\b[A-Z][a-zA-Z'\-]+"
    r"(?:\s+(?:of|the|de|von|van|du|la|le|der|den)\s+)?"
    r"(?:\s+[A-Z][a-zA-Z'\-]+)*\b"
)

# A-E single-letter MCQ label, optionally trailed by punctuation.
# Case-insensitive so users / datasets that emit lowercase labels are
# handled identically to uppercase.
_MCQ_LABEL_RE = re.compile(r"^\s*([A-Ea-e])(?:[.\)\]:]|\b)")

# Single-letter strict equality test (CSQA dispatch shortcut).
_SINGLE_LETTER_RE = re.compile(r"^\s*([A-Ea-e])\s*$")


def extract_head_noun(text: str) -> str:
    """Heuristic head-noun extraction.

    Resolution order:
      1. ``MCQ-label``: "A.", "(B)", "C)", or just "D" → return the
         letter alone, lower-cased so capitalisation doesn't affect
         pair-equality.
      2. ``Title-Case run``: take the rightmost (longest tied)
         capitalised span — typical English head-noun position.
      3. ``Numeric / year``: any 4-digit run is a candidate (years,
         dates) — useful for factoid QA on dates.
      4. ``Fallback``: the first alphanumeric token, lower-cased.
      5. Empty string when nothing matches (caller treats this as
         "not extracted").

    Returns a lower-cased string so callers can compare for
    equality without re-normalising.
    """
    if not text:
        return ""
    text = text.strip()
    if not text:
        return ""

    # 1. Multi-choice label
    if (m := _SINGLE_LETTER_RE.match(text)):
        return m.group(1).lower()
    if (m := _MCQ_LABEL_RE.match(text)):
        return m.group(1).lower()

    # 2. Title-Case run — pick the LAST one (rightmost) since English
    # head-nouns are usually at the right edge of the noun phrase.
    runs = list(_TITLE_RUN_RE.finditer(text))
    if runs:
        candidate = runs[-1].group(0).strip()
        # Strip trailing stop-words that leak into title-case runs.
        toks = candidate.split()
        while toks and toks[-1].lower() in _STOPWORDS:
            toks.pop()
        if toks:
            return " ".join(toks).casefold()

    # 3. 4-digit numeric (year-like)
    if (m := re.search(r"\b(\d{4})\b", text)):
        return m.group(1)

    # 4. First non-stopword alphanumeric token
    for tok in re.findall(r"[A-Za-z0-9'\-]+", text):
        if tok.lower() not in _STOPWORDS:
            return tok.casefold()

    return ""


def score_entity_head_consistency(chains: Iterable[str]) -> float:
    """Compute pairwise head-noun agreement across the M chains.

    Definition::

        Let H = [head(c) for c in chains if head(c) != ""]
        Let n = |H|
        If n < 2: return 0.5  (neutral / missing signal)

        agree = sum(1 for (i, j) in combinations(H, 2) if H_i == H_j)
        total = n * (n - 1) / 2
        return agree / total

    Returns a float in ``[0, 1]``. The neutral 0.5 fallback is used
    when fewer than two chains produce extractable head nouns; this
    matches the ``q_a_relevance`` / ``alias_overlap`` convention so
    the composite isotonic treats missing signals as non-informative
    rather than catastrophic.
    """
    chain_list = list(chains or [])
    if not chain_list:
        return 0.5
    heads: List[str] = []
    for c in chain_list:
        h = extract_head_noun(c or "")
        if h:
            heads.append(h)
    if len(heads) < 2:
        return 0.5
    total = 0
    agree = 0
    for a, b in combinations(heads, 2):
        total += 1
        if a == b:
            agree += 1
    if total == 0:
        return 0.5
    return float(agree) / float(total)
