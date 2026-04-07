"""
eval/metrics.py
===============
Evaluation metrics for the CAEM benchmark harness.

Metrics used per benchmark
--------------------------
HotpotQA   -- Exact Match (EM) + Token F1
             Standard formulation from Yang et al. 2018.
             Both are case-insensitive, after normalising punctuation and
             stripping articles ("a", "an", "the").

TruthfulQA -- ROUGE-L
             The dataset provides a list of acceptable answer strings; a
             response is scored via ROUGE-L (LCS F1) to give partial credit
             for overlapping n-grams, serving as an offline proxy for a judge.

FEVER      -- Label accuracy
             Claims are labelled SUPPORTS / REFUTES / NOT ENOUGH INFO.
             Predicted label is compared directly to gold label (case-insensitive).
             This follows the standard FEVER evaluation protocol.

Hallucination rate
------------------
Defined operationally as: fraction of answers where û_stored >= 0.50 AND
the answer does not match the gold (EM = 0). This proxy measures confident
confabulations (where the model arrogantly asserts a falsehood).

Routing distribution
--------------------
Fraction of queries routed to each tier (1/2/3) per evaluation run.
Reported alongside accuracy to show the memory utilisation curve across cycles.

Notes
-----
All public functions are pure (no side effects) for easy unit testing.
No imports from `caem` package -- this module only uses stdlib + numpy.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Text normalisation (shared by EM and F1)
# -----------------------------------------------------------------------------

_ARTICLES = frozenset({"a", "an", "the"})
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """Normalise an answer string for evaluation.

    Steps (order matters):
    1. Lowercase.
    2. Strip punctuation.
    3. Remove articles (a, an, the).
    4. Collapse whitespace.

    This exactly matches the SQuAD / HotpotQA official normalisation.
    """
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def extract_cot_answer(text: str) -> str:
    """Extract the final answer from a Chain-of-Thought string.
    
    Splits by 'Answer:' and takes the right-most side.
    If 'Answer:' is not present, returns the full string to avoid breaking
    fallback or non-CoT outputs.
    """
    marker = "Answer:"
    if marker in text:
        return text.split(marker)[-1].strip()
    return text.strip()


# -----------------------------------------------------------------------------
# Exact Match
# -----------------------------------------------------------------------------

def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 if normalised prediction == normalised gold, else 0.0.

    Parameters
    ----------
    prediction : str  -- model's answer
    gold : str        -- single gold answer string

    Returns
    -------
    float -- 1.0 or 0.0
    """
    return float(normalise(prediction) == normalise(gold))


def any_match_em(prediction: str, golds: Sequence[str]) -> float:
    """Return 1.0 if prediction matches ANY gold string (TruthfulQA).

    Parameters
    ----------
    prediction : str
    golds : sequence of str -- acceptable answers

    Returns
    -------
    float -- 1.0 or 0.0
    """
    norm_pred = normalise(prediction)
    return float(any(norm_pred == normalise(g) for g in golds))


# -----------------------------------------------------------------------------
# Token F1
# -----------------------------------------------------------------------------

def token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold.

    Matches the SQuAD / HotpotQA official F1 computation:
    - Both strings are normalised.
    - F1 is over the multiset of tokens.

    Parameters
    ----------
    prediction : str
    gold : str

    Returns
    -------
    float -- F1 score ∈ [0, 1]
    """
    pred_tokens = normalise(prediction).split()
    gold_tokens = normalise(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, golds: Sequence[str]) -> float:
    """ROUGE-L (LCS F1) across all gold strings, returning the max.
    
    Used for TruthfulQA as an offline proxy for a judge model, giving
    partial credit for overlapping free-form sequences.
    """
    def _lcs_length(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    pred_toks = normalise(prediction).split()
    best = 0.0
    for gold in golds:
        gold_toks = normalise(gold).split()
        if not pred_toks and not gold_toks:
            best = max(best, 1.0)
            continue
        if not pred_toks or not gold_toks:
            continue
            
        lcs = _lcs_length(pred_toks, gold_toks)
        p = lcs / len(pred_toks)
        r = lcs / len(gold_toks)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        best = max(best, f)
    return best


def best_token_f1(prediction: str, golds: Sequence[str]) -> float:
    """Return the highest token F1 across all gold strings.

    Used for TruthfulQA where multiple acceptable answers exist.
    """
    if not golds:
        return 0.0
    return max(token_f1(prediction, g) for g in golds)


# -----------------------------------------------------------------------------
# FEVER label accuracy
# -----------------------------------------------------------------------------

_FEVER_LABELS = frozenset({"supports", "refutes", "not enough info"})


def fever_accuracy(prediction: str, gold_label: str) -> float:
    """Return 1.0 if lowercased prediction matches lowercased gold FEVER label.

    Gold labels are: SUPPORTS, REFUTES, NOT ENOUGH INFO (case-insensitive).

    Parameters
    ----------
    prediction : str -- model's predicted label (any case)
    gold_label : str -- ground-truth FEVER label

    Returns
    -------
    float -- 1.0 or 0.0
    """
    return float(prediction.strip().lower() == gold_label.strip().lower())


def extract_fever_label(text: str) -> str:
    """Extract a FEVER label from free-form model output.

    Looks for "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO" (case-insensitive)
    anywhere in the text. Returns the first match or "not enough info" if none.

    This handles Flan-T5 outputs like:
      "The claim is supported by the evidence."  -> "supports"
      "REFUTES"                                   -> "refutes"
      "I don't have enough information."          -> "not enough info"
    """
    text_lower = text.lower()
    if "not enough info" in text_lower or "not enough information" in text_lower:
        return "not enough info"
    if "refutes" in text_lower or "refuted" in text_lower:
        return "refutes"
    if "supports" in text_lower or "supported" in text_lower:
        return "supports"
    return "not enough info"   # conservative fallback


# -----------------------------------------------------------------------------
# Hallucination rate (operational proxy)
# -----------------------------------------------------------------------------

def hallucination_rate(
    em_scores: Sequence[float],
    u_stored_values: Sequence[Optional[float]],
    u_threshold: float = 0.50,
) -> float:
    """Fraction of answers that are both wrong AND confidently generated.

    Operational definition: EM = 0 AND û_stored >= u_threshold.
    This is a proxy for confident confabulations (hallucinations), not a
    ground-truth oracle. Used for ablation analysis in Chapter 5.

    Parameters
    ----------
    em_scores : sequence of float
        Per-sample EM scores (0.0 or 1.0).
    u_stored_values : sequence of float or None
        Per-sample û_stored values. None means verification was skipped
        (treated as u_stored = 0.0 for this computation).
    u_threshold : float
        û_stored threshold below which we consider the model uncertain.

    Returns
    -------
    float -- fraction ∈ [0, 1]
    """
    if not em_scores:
        return 0.0

    n = len(em_scores)
    assert len(u_stored_values) == n, "em_scores and u_stored_values must have the same length"

    count = 0
    for em, u in zip(em_scores, u_stored_values):
        u_val = u if u is not None else 0.0
        if em == 0.0 and u_val >= u_threshold:  # Fixed: measures Confident Confabulation
            count += 1

    return count / n


# -----------------------------------------------------------------------------
# Routing distribution
# -----------------------------------------------------------------------------

def routing_distribution(tiers: Sequence[int]) -> Dict[str, float]:
    """Compute fraction of queries routed to each tier.

    Parameters
    ----------
    tiers : sequence of int -- one value per query (1, 2, or 3)

    Returns
    -------
    dict with keys "tier1_frac", "tier2_frac", "tier3_frac"
    """
    if not tiers:
        return {"tier1_frac": 0.0, "tier2_frac": 0.0, "tier3_frac": 0.0}

    n = len(tiers)
    counts = Counter(tiers)
    return {
        "tier1_frac": counts.get(1, 0) / n,
        "tier2_frac": counts.get(2, 0) / n,
        "tier3_frac": counts.get(3, 0) / n,
    }


# -----------------------------------------------------------------------------
# Aggregate summary
# -----------------------------------------------------------------------------

def aggregate(
    benchmark: str,
    em_scores: Sequence[float],
    f1_scores: Sequence[float],
    tiers: Sequence[int],
    u_stored_values: Sequence[Optional[float]],
    stored_flags: Sequence[bool],
    latencies_ms: Sequence[float],
) -> Dict:
    """Produce the Chapter 5 Table 1 row for one (benchmark, cycle) run.

    Parameters
    ----------
    benchmark : str -- "hotpotqa", "truthfulqa", or "fever"
    em_scores : per-sample EM (0.0/1.0)
    f1_scores : per-sample F1 (for HotpotQA); set to em_scores for others
    tiers : per-sample tier (1/2/3)
    u_stored_values : per-sample û_stored or None
    stored_flags : per-sample bool -- was the answer stored in memory?
    latencies_ms : per-sample wall clock time in ms

    Returns
    -------
    dict -- all metrics for this run
    """
    n = len(em_scores)
    if n == 0:
        return {"benchmark": benchmark, "n": 0}

    routing = routing_distribution(tiers)
    hall_rate = hallucination_rate(em_scores, u_stored_values)
    valid_u = [u for u in u_stored_values if u is not None]

    return {
        "benchmark": benchmark,
        "n": n,
        "em": round(sum(em_scores) / n, 4),
        "f1": round(sum(f1_scores) / n, 4),
        "hallucination_rate": round(hall_rate, 4),
        "storage_rate": round(sum(stored_flags) / n, 4),
        "mean_u_stored": round(sum(valid_u) / len(valid_u), 4) if valid_u else None,
        "mean_latency_ms": round(sum(latencies_ms) / n, 1),
        **routing,
    }


# -----------------------------------------------------------------------------
# Statistical significance utilities
# -----------------------------------------------------------------------------

def bootstrap_ci(
    scores: Sequence[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval for a mean score.

    Used in Chapter 5 to report 95% CI around EM and F1 values.

    Parameters
    ----------
    scores : sequence of float -- per-sample metric values (e.g. EM scores)
    n_bootstrap : int -- number of bootstrap resamples (1000 is standard)
    ci : float -- confidence level (0.95 for 95% CI)
    seed : int

    Returns
    -------
    (mean, lower, upper) : tuple of float
        mean -- point estimate
        lower -- lower CI bound
        upper -- upper CI bound

    Example
    -------
    >>> mean, lo, hi = bootstrap_ci(em_scores)
    >>> print(f"EM = {mean:.3f} [{lo:.3f}, {hi:.3f}]")
    """
    import random as _random
    rng = _random.Random(seed)
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0

    scores_list = list(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(scores_list) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1.0 - ci) / 2.0
    lower = means[int(alpha * n_bootstrap)]
    upper = means[int((1.0 - alpha) * n_bootstrap) - 1]
    point_mean = sum(scores_list) / n
    return point_mean, lower, upper


def mcnemar_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> Tuple[float, float]:
    """McNemar's test for statistical significance between two systems.

    Appropriate for paired binary outcomes (correct/incorrect) on the same
    test set -- the standard significance test for NLP evaluation (Dror et
    al. 2018 "Deep Dominance").

    Parameters
    ----------
    scores_a : sequence of float -- EM scores for system A (0.0 or 1.0)
    scores_b : sequence of float -- EM scores for system B (0.0 or 1.0)

    Returns
    -------
    (statistic, p_value) : tuple of float
        statistic -- McNemar chi-squared statistic
        p_value   -- two-tailed p-value (p < 0.05 = statistically significant)

    Notes
    -----
    Requires scipy. If scipy is not installed, raises ImportError with
    install instructions. scipy is an experiment-phase dependency --
    it is not required during the implementation phase.
    """
    try:
        from scipy.stats import chi2
    except ImportError as e:
        raise ImportError(
            "scipy is required for mcnemar_test. "
            "Install with: pip install scipy"
        ) from e

    assert len(scores_a) == len(scores_b), "Both score sequences must have equal length."

    # Contingency counts
    n01 = sum(1 for a, b in zip(scores_a, scores_b) if a == 0.0 and b == 1.0)
    n10 = sum(1 for a, b in zip(scores_a, scores_b) if a == 1.0 and b == 0.0)

    if n01 + n10 == 0:
        return 0.0, 1.0   # No disagreements -- systems are identical

    # McNemar statistic with continuity correction (Edwards 1948)
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1.0 - chi2.cdf(statistic, df=1)
    return statistic, p_value
