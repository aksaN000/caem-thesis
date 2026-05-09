"""
caem/verification/directional_p_ground.py
=========================================

Directional ``p_ground_mean`` scorer — Branch-C refutation-bias fix
(2026-04-23).

Problem solved
--------------
The u_stored composite's primary signal ``p_ground_mean`` (weight 0.28)
asks MiniCheck: "do the passages support the hypothesis?". For a
correctly-*refuted* claim, the passages should NOT support the claim —
so p_ground_mean is architecturally forced to be low even when the
model's "refutes" answer is perfect. Empirical finding on FEVER
Cycle-0 (n=500): **33% of correct refutations discarded, zero stored.**

Solution
--------
When the model's answer carries a directional label (supports/refutes/NEI
for FEVER; yes/no for StrategyQA), rewrite the hypothesis fed to
MiniCheck so it always reads as *"the truthful statement"*:

- **supports / yes**  → hypothesis = claim (unchanged)
- **refutes / no**    → hypothesis = negate(claim)
- **not enough info** → bidirectional certainty-of-uncertainty:
  p_ground_mean = 1 - 2·max(|mean(p_pos) - 0.5|, |mean(p_neg) - 0.5|)
  Genuine NEI (both directions ambiguous) → high score. Defensive NEI
  on actually-informative passages → low score (correct DISCARD).

Scope
-----
- **FEVER** (query starts with "Answer with one of: supports, refutes,
  not enough info. Claim: ..."): fully supported.
- **StrategyQA** (query starts with "Answer yes or no. Question: ..."):
  yes/no directional — uses the question text as the hypothesis directly
  (e.g. MiniCheck on "Can food be cooked in the cosmic microwave
  background?" vs its negation). Works even without question→declarative
  conversion because MiniCheck handles interrogative-form hypotheses.
- **Other benchmarks** (TriviaQA, NQ, TruthfulQA, ARC-Challenge, ASQA):
  ``score()`` returns None → caller falls back to legacy
  ``_score_p_ground``. Bit-identical behavior on factoid/multi-choice
  samples.

Backward compatibility
----------------------
Env-gated by ``CAEM_DIRECTIONAL_P_GROUND`` (default 1). Set to 0 to
revert to legacy behavior — preserves reproducibility with pre-fix runs.

Claim 1 theorem
---------------
The directional p_ground_mean is still an entailment probability of a
*truthful* claim given the passages — same [0,1] codomain, same
monotonicity with correctness. The product-of-gates bound in Ch4
needs a one-line update to the definition but not to the algebra.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np

from caem.verification.negation import Negator, NegationResult

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Task detection                                                         #
# --------------------------------------------------------------------- #

_FEVER_QUERY_PATTERN    = re.compile(
    r"supports\s*,\s*refutes\s*,\s*not enough info", re.IGNORECASE,
)
_STRATEGYQA_QUERY_PATTERN = re.compile(r"^\s*answer\s+yes\s+or\s+no\b", re.IGNORECASE)

_FEVER_CLAIM_PATTERN      = re.compile(r"Claim:\s*(.+?)(?:\s*$|\n)", re.DOTALL)
_STRATEGYQA_Q_PATTERN     = re.compile(r"Question:\s*(.+?)(?:\s*$|\n)", re.DOTALL)

_FEVER_LABEL_PATTERN      = re.compile(
    r"Answer\s*:\s*(supports|refutes|not\s+enough\s+info)\b", re.IGNORECASE,
)
_STRATEGYQA_LABEL_PATTERN = re.compile(r"Answer\s*:\s*(yes|no)\b", re.IGNORECASE)


def detect_task(query: str) -> Optional[str]:
    """Return 'fever' | 'strategyqa' | None."""
    if not query:
        return None
    if _FEVER_QUERY_PATTERN.search(query):
        return "fever"
    if _STRATEGYQA_QUERY_PATTERN.match(query):
        return "strategyqa"
    return None


def extract_fever_claim(query: str) -> Optional[str]:
    m = _FEVER_CLAIM_PATTERN.search(query)
    return m.group(1).strip() if m else None


def extract_strategyqa_question(query: str) -> Optional[str]:
    m = _STRATEGYQA_Q_PATTERN.search(query)
    return m.group(1).strip() if m else None


def extract_fever_label(answer: str) -> Optional[str]:
    """Return one of {'supports', 'refutes', 'not enough info'} or None."""
    if not answer:
        return None
    m = _FEVER_LABEL_PATTERN.search(answer)
    if not m:
        return None
    raw = m.group(1).lower().strip()
    # Normalize "not  enough info" / "not enough info" / "not_enough_info"
    if "not" in raw and "enough" in raw:
        return "not enough info"
    return raw


def extract_strategyqa_label(answer: str) -> Optional[str]:
    """Return 'yes' | 'no' | None."""
    if not answer:
        return None
    m = _STRATEGYQA_LABEL_PATTERN.search(answer)
    return m.group(1).lower().strip() if m else None


# --------------------------------------------------------------------- #
# Directional scorer                                                    #
# --------------------------------------------------------------------- #

def _is_enabled() -> bool:
    """Master env-gate. Default ON; set CAEM_DIRECTIONAL_P_GROUND=0 to disable."""
    return os.environ.get("CAEM_DIRECTIONAL_P_GROUND", "1") == "1"


class DirectionalScorer:
    """Encapsulates the directional p_ground computation.

    Parameters
    ----------
    nli_judge : MiniCheckJudge or compatible
        Must support batch_entail_prob(pairs_or_premises, hypotheses=None).
    negator : Negator
        Handles claim negation (rules + Qwen fallback).
    """

    def __init__(self, nli_judge, negator: Negator):
        self.nli = nli_judge
        self.negator = negator
        self._last_negation_source = "n/a"  # diagnostic

    def score(
        self, query: str, answer: str, passages: List[str],
    ) -> Optional[Tuple[float, float]]:
        """Return (p_ground_max, p_ground_mean) or None to use legacy path.

        None means: sample isn't a directional task, or directional path
        couldn't apply cleanly (couldn't parse claim/label, negation failed,
        env flag off). Caller should fall back to the standard
        ``_score_p_ground``.
        """
        if not _is_enabled():
            return None
        if not passages or self.nli is None:
            return None

        task = detect_task(query)
        if task is None:
            return None

        if task == "fever":
            claim = extract_fever_claim(query)
            label = extract_fever_label(answer)
            if claim is None or label is None:
                return None
            return self._score_three_way(passages, claim, label)

        if task == "strategyqa":
            q_text = extract_strategyqa_question(query)
            label = extract_strategyqa_label(answer)
            if q_text is None or label is None:
                return None
            # Map yes/no to supports/refutes-style logic; use the question
            # as the hypothesis directly.
            mapped = "supports" if label == "yes" else "refutes"
            return self._score_three_way(passages, q_text, mapped)

        return None

    def _score_three_way(
        self, passages: List[str], claim: str, label: str,
    ) -> Optional[Tuple[float, float]]:
        """Core 3-way directional logic. `label` in {supports, refutes, NEI}."""
        if label == "supports":
            return self._minicheck_over_passages(passages, claim)

        if label == "refutes":
            neg = self.negator.negate(claim)
            self._last_negation_source = neg.source
            if neg.text is None:
                logger.debug("Negation failed for claim=%r — falling back to legacy", claim[:60])
                return None
            return self._minicheck_over_passages(passages, neg.text)

        if label == "not enough info":
            # Bidirectional certainty-of-uncertainty. Genuine NEI (passages
            # ambiguous about both pos AND neg forms) → high score. Defensive
            # NEI on actually-informative passages → low score (correct DISCARD).
            neg = self.negator.negate(claim)
            self._last_negation_source = neg.source
            pos_max, pos_mean = self._minicheck_over_passages(passages, claim)
            if neg.text is None:
                # Without a negation, fall back to legacy
                return None
            neg_max, neg_mean = self._minicheck_over_passages(passages, neg.text)
            # Distance from 0.5 = "certainty in one direction"
            cert_pos = abs(pos_mean - 0.5)
            cert_neg = abs(neg_mean - 0.5)
            strongest_direction = max(cert_pos, cert_neg)
            # G30 FIX 2026-04-24: tighten bidirectional NEI threshold.
            # The original formula (1 - 2·strongest_direction) was too
            # permissive: defensive NEI with pos_mean ~0.45 and neg_mean
            # ~0.45 would score 0.9+ and pass storage. In the Cycle-0
            # audit, 3/3 stored NEI samples were defensive (wrong).
            # Require BOTH directions within 0.10 of 0.5 for high score
            # (otherwise stronger decay: 1 - 4·max).
            decay = 4.0 if strongest_direction > 0.10 else 2.0
            p_ground_mean = float(np.clip(1.0 - decay * strongest_direction, 0.0, 1.0))
            # P2 FIX 2026-05-09: apply the same certainty-of-uncertainty
            # treatment to p_ground_max as p_ground_mean. The previous
            # max(pos_max, neg_max) collapsed to "high regardless of
            # correctness" — em=1 mean 0.578, em=0 mean 0.555 on v1 cycle-0
            # FEVER, AUROC 0.529 (random). Use the strongest peak's distance
            # from 0.5 as the certainty signal: genuine NEI (both peaks near
            # 0.5) → high p_ground_max; defensive NEI (one peak far from 0.5)
            # → low p_ground_max. Same decay schedule as p_ground_mean.
            cert_pos_peak = abs(pos_max - 0.5)
            cert_neg_peak = abs(neg_max - 0.5)
            strongest_peak = max(cert_pos_peak, cert_neg_peak)
            decay_peak = 4.0 if strongest_peak > 0.10 else 2.0
            p_ground_max = float(np.clip(1.0 - decay_peak * strongest_peak, 0.0, 1.0))
            return (p_ground_max, p_ground_mean)

        return None

    def _minicheck_over_passages(
        self, passages: List[str], hypothesis: str,
    ) -> Tuple[float, float]:
        """Run MiniCheck on (passage, hypothesis) for each passage, return (max, mean)."""
        pairs = [(p, hypothesis) for p in passages]
        try:
            scores = self.nli.batch_entail_prob(pairs)
            scores = [float(np.clip(s, 0.0, 1.0)) for s in scores]
            if not scores:
                return 0.5, 0.5
            return float(max(scores)), float(np.mean(scores))
        except Exception as exc:
            logger.warning("Directional MiniCheck call failed (%s); falling back.", exc)
            return 0.5, 0.5

    def last_negation_source(self) -> str:
        """For telemetry: 'rule' | 'qwen' | 'failed' | 'n/a'."""
        return self._last_negation_source
