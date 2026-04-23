"""
caem/verification/multichoice_scorer.py
=======================================

Multichoice option-text substitution — Branch-C ARC-Challenge
failure-mode fix (2026-04-23).

Problem
-------
Multi-choice benchmarks (ARC-Challenge, MMLU, OpenBookQA) produce
answers of the form "Answer: C" — a single letter with no semantic
content on its own. When the verifier asks MiniCheck "do the passages
support 'C'?", the letter carries no truth value to evaluate, so
``p_ground_mean`` collapses to ~0.08 regardless of whether the model
answered correctly. On ARC-Challenge Cycle-0 (n=500, pre-fix composite)
this caused **18.4% of correct answers to be discarded** and only
23 of 500 samples to enter the STORE bucket.

The early-exit confabulation gate (``u_internal ≥ 0.70 AND
p_ground_max ≤ 0.20``) fires spuriously on multi-choice because the
letter-only hypothesis has no passage support — leading to
``u_stored = 0.0`` on correctly-answered samples.

Solution
--------
Before handing the answer to MiniCheck, substitute the single letter
for the **full text of the option** it refers to. The hypothesis
becomes "The answer to the question is: <option_text>", which MiniCheck
can evaluate semantically against the retrieved passages.

Example:

- Query: "Question: Which best explains ... Choices: (A) chlorophyll
  (B) photosynthesis (C) a system of tubes (D) water→food.
  Answer with just the multiple choice letter."
- Answer: "Reasoning: ... Answer: C"
- Legacy hypothesis: "C" → MiniCheck sees nothing evaluable
- Multichoice hypothesis: "The answer to the question is: a system of
  tubes" → MiniCheck evaluates passage-level entailment normally

Scope
-----
Applies only when:
- Query matches the ARC-style format (``Choices:`` block present AND
  ``Answer with (just) the (multiple choice)? letter`` instruction).
- Extracted answer is a single letter A-D (or A-E for 5-choice variants).
- Option text can be parsed from the choices block.

All other task types (FEVER 3-way, StrategyQA yes/no, factoid, long-form)
return None → caller falls back to legacy ``_score_p_ground`` or to
``DirectionalScorer`` for 3-way cases. Multichoice substitution runs
AFTER directional routing — if both match (shouldn't happen), directional
wins.

Backward compatibility
----------------------
Env-gated by ``CAEM_MULTICHOICE_SUBSTITUTION`` (default 1). Set to 0 to
revert to legacy behavior on multi-choice tasks (preserves
reproducibility with pre-fix runs).

Claim 1 theorem
---------------
The multichoice-substituted hypothesis is still an entailment probability
of a truthful statement given the passages — same [0,1] codomain, same
monotonicity with correctness. The product-of-gates bound needs a
one-line definition update to the hypothesis formulation; the bound
algebra is unchanged.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Task detection + extraction                                            #
# --------------------------------------------------------------------- #

# Detect queries with a choices block AND an answer-letter instruction.
# Matches ARC-Challenge and MMLU-style prompts.
_MC_QUERY_PATTERN = re.compile(
    r"Choices:\s*.*?\([A-E]\).*?Answer\s+with\s+(?:just\s+)?the\s+(?:multiple\s+choice\s+)?letter",
    re.IGNORECASE | re.DOTALL,
)

# Extract the choices block (everything between "Choices:" and the
# terminating newline / "Answer with" instruction).
_CHOICES_BLOCK_PATTERN = re.compile(
    r"Choices:\s*(.+?)(?:\n\s*Answer\s+with|$)",
    re.IGNORECASE | re.DOTALL,
)

# Single-letter answer extraction — matches both "Answer: C" and the
# end-of-CoT format. Capture A-E to accommodate 4- and 5-choice variants.
_MC_LABEL_PATTERN = re.compile(r"Answer\s*:\s*([A-E])\b", re.IGNORECASE)


def detect_multichoice_task(query: str) -> bool:
    """Return True if the query is a multi-choice task with a choices block."""
    if not query:
        return False
    return bool(_MC_QUERY_PATTERN.search(query))


def extract_options(query: str) -> Dict[str, str]:
    """Parse the choices block into {letter: option_text} dict.

    Handles one-line format:   "(A) foo (B) bar (C) baz (D) qux"
    Handles multi-line format:
        "(A) foo
         (B) bar
         (C) baz
         (D) qux"
    Returns an empty dict on parse failure (caller should fall back).
    """
    if not query:
        return {}
    m = _CHOICES_BLOCK_PATTERN.search(query)
    if not m:
        return {}
    block = m.group(1)
    # Find all "(X) " markers and slice the text between consecutive markers.
    markers = list(re.finditer(r"\(([A-E])\)\s*", block))
    options: Dict[str, str] = {}
    for i, mk in enumerate(markers):
        letter = mk.group(1).upper()
        start = mk.end()
        end = markers[i + 1].start() if (i + 1) < len(markers) else len(block)
        text = block[start:end]
        # Normalise whitespace; strip trailing period / punctuation the template
        # sometimes includes between options.
        text = re.sub(r"\s+", " ", text).strip()
        text = text.rstrip(".,; ").strip()
        if text:
            options[letter] = text
    return options


def extract_multichoice_letter(answer: str) -> Optional[str]:
    """Extract A/B/C/D/E from a CoT answer. Case-insensitive. Returns None if absent."""
    if not answer:
        return None
    m = _MC_LABEL_PATTERN.search(answer)
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------- #
# Scorer                                                                #
# --------------------------------------------------------------------- #

def _is_enabled() -> bool:
    """Env gate. Default ON; set CAEM_MULTICHOICE_SUBSTITUTION=0 to disable."""
    return os.environ.get("CAEM_MULTICHOICE_SUBSTITUTION", "1") == "1"


class MultichoiceScorer:
    """Option-text-substituted p_ground_mean for multi-choice tasks.

    Parameters
    ----------
    nli_judge : MiniCheckJudge or compatible
        Must support ``batch_entail_prob(pairs_or_premises, hypotheses=None)``.
    """

    # Hypothesis template. The leading "The answer to the question is: "
    # primes MiniCheck to evaluate propositional content, not loose topic
    # relevance. Empirically (ARC spot-check 2026-04-23) this template
    # lifts mean p_ground_mean on correctly-answered samples from 0.08
    # (letter-only) to ~0.6 (option-text substituted).
    HYPOTHESIS_TEMPLATE = "The answer to the question is: {option_text}"

    def __init__(self, nli_judge):
        self.nli = nli_judge
        self._last_substitution: Optional[Tuple[str, str]] = None  # diagnostic

    def score(
        self, query: str, answer: str, passages: List[str],
    ) -> Optional[Tuple[float, float]]:
        """Return (p_ground_max, p_ground_mean) or None to use legacy path.

        None means: not a multi-choice task, or substitution couldn't
        apply (couldn't parse choices, couldn't extract letter, env
        flag off). Caller should fall back to ``_score_p_ground`` or
        the upstream dispatcher.
        """
        if not _is_enabled():
            return None
        if not passages or self.nli is None:
            return None
        if not detect_multichoice_task(query):
            return None

        letter = extract_multichoice_letter(answer)
        if letter is None:
            return None

        options = extract_options(query)
        if not options or letter not in options:
            logger.debug(
                "Multichoice substitution: letter=%r not in parsed options=%r; "
                "falling back to legacy.",
                letter, list(options.keys()),
            )
            return None

        option_text = options[letter]
        hypothesis = self.HYPOTHESIS_TEMPLATE.format(option_text=option_text)
        self._last_substitution = (letter, hypothesis)

        pairs = [(p, hypothesis) for p in passages]
        try:
            scores = self.nli.batch_entail_prob(pairs)
            scores = [float(np.clip(s, 0.0, 1.0)) for s in scores]
            if not scores:
                return 0.5, 0.5
            return float(max(scores)), float(np.mean(scores))
        except Exception as exc:
            logger.warning(
                "Multichoice MiniCheck call failed (%s); falling back to legacy.",
                exc,
            )
            return None

    def last_substitution(self) -> Optional[Tuple[str, str]]:
        """For telemetry: (selected_letter, rewritten_hypothesis) from the
        most recent score() call that performed substitution."""
        return self._last_substitution
