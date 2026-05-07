"""
caem/verification/entity_expansion_scorer.py
==============================================
v2 Fix 10 — bare-entity NLI grounding for short factoid answers
(TriviaQA, NaturalQuestions, HotpotQA).

Problem
-------
Open-domain factoid benchmarks routinely return one- or two-token
display answers ("Paris", "1969", "Albert Einstein"). When the
verifier asks the NLI model "do the passages support 'Paris'?", the
hypothesis carries no claim — there is no proposition to evaluate, so
``p_ground_max`` collapses to ~0.05-0.10 regardless of whether the
answer is correct. The composite then routes to DEFERRED / DISCARD on
EM-correct samples, mirroring the multichoice failure mode that
``multichoice_scorer.py`` patched for ARC.

Solution
--------
Wrap the bare entity in a declarative shell so the hypothesis carries
testable propositional content:

    "Paris"  →  "The answer to the question is: Paris."

and then score the wrapped hypothesis against the retrieved passages
via the verifier's NLI judge. The wrapping is symmetric to what the
existing :class:`MultichoiceScorer` does for letter-only MCQ answers,
and the ``p_ground_*`` math is unchanged.

Scope
-----
Applies only when ALL of:

* The query is open-ended (not FEVER 3-way, not StrategyQA yes/no, not
  multi-choice). I.e., ``detect_query_task`` would return ``"open"`` or
  ``"hotpotqa"``.
* The display answer is short (≤ 5 whitespace tokens after stripping
  trailing punctuation).
* The display answer has propositional emptiness (no subject-verb
  structure detectable). We use a lightweight heuristic: contains no
  verb-like token from a small whitelist (``is``, ``was``, ``has``,
  ``had``, ``were``, ``are``, ``did``, ``do``, ``does``).
* The display answer is not an evasive refusal (``I do not know``,
  ``not enough info``, ``no``, ``yes``).

When any condition fails, :meth:`score` returns ``None`` and the caller
falls back to legacy ``_score_p_ground``.

Backward compatibility
----------------------
Env-gated by ``CAEM_ENTITY_EXPANSION_SUBSTITUTION`` (default 1). Set to
``0`` to revert to legacy behaviour on bare-entity factoid answers.

Claim 1 theorem
---------------
The expanded hypothesis is still a positive declarative claim about
the answer entity given the passages, so the entailment probability
inherits the same monotonicity-with-correctness property required by
the calibration contract. The product-of-gates bound algebra is
unchanged; only the per-sample hypothesis text is rewritten before
NLI.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Verb tokens that indicate the answer already carries propositional
# content. Conservative on purpose: anything containing one of these
# tokens is left alone and the caller falls back to legacy NLI.
_VERB_TOKENS = frozenset({
    "is", "isn't", "isnot", "are", "aren't",
    "was", "wasn't", "were", "weren't",
    "be", "been", "being", "am",
    "has", "have", "had", "haven't", "hasn't", "hadn't",
    "do", "does", "did", "don't", "doesn't", "didn't",
    "will", "would", "won't", "wouldn't",
    "can", "could", "can't", "couldn't",
    "should", "shouldn't", "may", "might",
})

# Refusal / sentinel responses that should NOT be wrapped — wrapping
# "The answer to the question is: I do not know." would teach NLI to
# entail refusals, which is the opposite of what we want.
_REFUSAL_PHRASES = frozenset({
    "i do not know",
    "i don't know",
    "no answer",
    "unknown",
    "not enough info",
    "not enough information",
    "not enough info.",
    "no",  # short yes/no answers — strategyqa territory
    "yes",
    "supports",
    "refutes",
})

# Strip trailing punctuation before length / verb detection so
# "Paris." reads as "Paris".
_PUNCT_STRIP_RE = re.compile(r"[\s\.\?!,:;'\"]+$")


def _normalised(answer: str) -> str:
    return _PUNCT_STRIP_RE.sub("", answer.strip()).strip()


def _looks_like_bare_entity(answer: str) -> bool:
    """Cheap heuristic: short, no verb tokens, not a refusal."""
    a = _normalised(answer).lower()
    if not a:
        return False
    if a in _REFUSAL_PHRASES:
        return False
    tokens = a.split()
    if len(tokens) > 5:
        return False
    for tok in tokens:
        if tok in _VERB_TOKENS:
            return False
    return True


def _is_open_task(task: str) -> bool:
    """Return True iff this task family carries bare-entity factoid
    answers that the legacy NLI hypothesis cannot evaluate."""
    return task in {"open", "hotpotqa"}


class EntityExpansionScorer:
    """Tier-3 p_ground scorer for bare-entity factoid answers.

    The verifier dispatcher (``UnifiedVerifier._p_ground_with_direction``)
    consults this scorer AFTER the directional and multichoice scorers
    have declined. The scorer's :meth:`score` returns ``None`` on tasks
    or answers it cannot meaningfully wrap, and a
    ``(p_ground_max, p_ground_mean)`` tuple otherwise.

    Parameters
    ----------
    nli_score_fn : Callable[[str, str], float]
        A function returning ``P(entailment)`` for a given
        ``(passage, hypothesis)`` pair. The verifier's NLI ensemble /
        MiniCheck judge satisfies this signature via
        ``self.nli.entail_prob(premise, hypothesis)``.
    detect_task_fn : Callable[[str], str]
        Task detector — usually ``caem.prompts.detect_query_task``.
        Injected so tests can stub task detection without importing the
        full prompts module.
    """

    def __init__(
        self,
        nli_score_fn: Callable[[str, str], float],
        detect_task_fn: Callable[[str], str],
    ) -> None:
        self.nli_score_fn = nli_score_fn
        self.detect_task_fn = detect_task_fn

    def score(
        self,
        query: str,
        answer: str,
        passages: List[str],
    ) -> Optional[Tuple[float, float]]:
        """Score a (query, answer) pair if the bare-entity heuristic applies.

        Returns
        -------
        (p_ground_max, p_ground_mean) or None
            ``None`` indicates this scorer declines the sample
            (caller falls back to legacy ``_score_p_ground``).
        """
        if os.environ.get("CAEM_ENTITY_EXPANSION_SUBSTITUTION", "1") == "0":
            return None
        if not passages:
            return None
        try:
            task = self.detect_task_fn(query)
        except Exception:
            task = "open"
        if not _is_open_task(task):
            return None
        if not _looks_like_bare_entity(answer):
            return None

        norm_answer = _normalised(answer)
        if not norm_answer:
            return None

        hypothesis = f"The answer to the question is: {norm_answer}."

        scores: List[float] = []
        for p in passages:
            try:
                s = float(self.nli_score_fn(p, hypothesis))
            except Exception as exc:
                logger.warning(
                    "entity_expansion_scorer NLI call failed (%s) — "
                    "falling back to legacy p_ground for this passage.",
                    exc,
                )
                continue
            # Defensive clamp; NLI judges occasionally emit tiny
            # negative-zero floats outside [0, 1].
            scores.append(max(0.0, min(1.0, s)))

        if not scores:
            return None

        p_max = max(scores)
        p_mean = sum(scores) / float(len(scores))
        return float(p_max), float(p_mean)
