"""
eval/metrics.py
===============
Evaluation metrics for the CAEM benchmark harness.

Metrics used per benchmark
--------------------------
FEVER          -- Label accuracy (3-class: supports / refutes / not enough info).
                  extract_fever_label() parses the constrained-prompt output.

TriviaQA       -- any-match EM + best token F1 across all answer aliases.
                  (TriviaQA provides 10-40 valid aliases per question.)

Natural Questions -- same as TriviaQA (multiple valid answer strings).

TruthfulQA     -- ROUGE-L (LCS F1) as offline judge proxy; EM = ROUGE-L > 0.15.

StrategyQA     -- Boolean EM: extract_strategyqa_label() + exact match on yes/no.

ARC-Challenge  -- Multiple-choice letter EM: extract_arc_label() + exact match.

Generic EM/F1  -- exact_match + token_f1 for single-answer open-ended QA.
                  (Both are case-insensitive after normalising punctuation and
                  stripping articles "a", "an", "the" -- matches SQuAD
                  convention.)

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
from typing import Any, Dict, List, Optional, Sequence, Tuple


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

    This exactly matches the SQuAD / TriviaQA official normalisation.
    """
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def extract_cot_answer(text: str) -> str:
    """Extract the final answer from a Chain-of-Thought string.

    Flan-T5 CoT outputs follow several conventions depending on how the
    prompt was structured. We try them in priority order:

    1. Explicit ``"Answer:"`` marker (seen when the prompt ends with
       ``"Answer:"``). Split on the last occurrence and return the right
       side. This was the only pattern the original implementation
       handled.
    2. ``"So, the answer is X"`` / ``"the answer is X"`` / ``"Therefore,
       the answer is X"`` -- Flan-T5's native CoT suffix, seen on ~98%
       of TriviaQA / NQ predictions when the prompt says ``"Think step
       by step."``. Before this change, those predictions returned the
       full CoT unchanged and failed EM against short-form gold
       aliases.
    3. Fallback: return the full string stripped.

    The extractor is case-insensitive for markers but preserves the
    case of the extracted answer (downstream :func:`normalise` handles
    casing + punctuation + article stripping).
    """
    # Priority 1: explicit "Answer:" marker.
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()

    # Priority 2: natural-language CoT suffix. Capture everything from
    # the "answer is" marker up to the first terminal punctuation or
    # end-of-string. Uses a non-greedy match so "The answer is X. Also
    # Y." captures only "X".
    m = re.search(
        r"(?:^|\W)(?:so|therefore)?,?\s*the\s+answer\s+is[:\s]+(.+?)(?:[.!?](?:\s|$)|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # Priority 3: fallback for non-CoT outputs (e.g. direct short-form
    # answers, FEVER labels, StrategyQA yes/no).
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

    Matches the SQuAD official F1 computation:
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


def best_token_f1(prediction: str, golds: Sequence[str]) -> float:
    """Return max token-F1 across a list of acceptable gold strings.

    Thin wrapper over :func:`token_f1` for benchmarks that provide multiple
    valid answer phrasings (TriviaQA aliases, NQ short-answer variants).
    The single-gold ``token_f1`` remains the canonical primitive; this helper
    exists so ``scripts/check_base_model.py`` (and any other list-input
    caller) shares the SAME normalisation as the in-pipeline numbers, which
    is what the data-purity theorem's p-anchor comparison requires.

    Parameters
    ----------
    prediction : str
    golds : sequence of str -- acceptable answer phrasings

    Returns
    -------
    float -- max token-F1 ∈ [0, 1]; 0.0 for empty ``golds``.
    """
    best = 0.0
    for g in golds:
        best = max(best, token_f1(prediction, g))
    return best


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


# -----------------------------------------------------------------------------
# FEVER label accuracy
# -----------------------------------------------------------------------------

# Sentinel returned by extract_*_label() helpers when no valid label can be
# parsed from a model's free-form output. Propagating this through the EM
# pipeline yields EM=0.0 (no string match with any gold label), so
# unparseable outputs are correctly counted as errors rather than being
# silently coerced to a "default" answer -- which previously biased FEVER
# toward "not enough info" and StrategyQA toward "no". See audit report
# MAJOR-M3 / Tasks #121, #125, #126.
UNPARSEABLE = ""

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


_FEVER_PATTERNS = (
    (re.compile(r"\bnot enough info(?:rmation)?\b", re.IGNORECASE), "not enough info"),
    (re.compile(r"\brefut(?:e|es|ed)\b",            re.IGNORECASE), "refutes"),
    (re.compile(r"\bsupport(?:s|ed)?\b",            re.IGNORECASE), "supports"),
)


def extract_fever_label(text: str) -> str:
    """Extract a FEVER label from free-form model output.

    Scans ``text`` for all case-insensitive, word-boundary matches of the
    three FEVER label families (SUPPORTS, REFUTES, NOT ENOUGH INFO) and
    returns the label whose *last* occurrence appears furthest right.
    Last-occurrence semantics reflect the model's final commitment, which
    matters for rationale-style outputs like

        "At first it looks like it supports the claim, but actually
         there is not enough info."         -> "not enough info"
        "There is not enough info to support this."
                                             -> "supports"  (mechanical)

    If no label family matches, returns :data:`UNPARSEABLE` (empty string)
    so downstream EM comparison yields 0.0 rather than silently defaulting
    to "not enough info" (the prior behaviour, which inflated NEI accuracy
    on unparseable outputs -- audit report MAJOR-M2 / MAJOR-M3).

    This handles Flan-T5 outputs like:
      "The claim is supported by the evidence."  -> "supports"
      "REFUTES"                                   -> "refutes"
      "I don't have enough information."          -> "not enough info"
      "I don't know anything."                    -> UNPARSEABLE ("")
    """
    best_label = UNPARSEABLE
    best_pos = -1
    for pattern, label in _FEVER_PATTERNS:
        for m in pattern.finditer(text):
            if m.start() > best_pos:
                best_pos = m.start()
                best_label = label
    return best_label


def extract_strategyqa_label(text: str) -> str:
    """Extract a StrategyQA label (yes/no) from free-form model output.

    Uses last-occurrence precedence across ``\\byes\\b`` / ``\\bno\\b``
    matches (case-insensitive, word-boundary-anchored) so rationale-style
    outputs commit to the label the model ended on:

      "Reasoning: ... Answer: yes"           -> "yes"
      "Yes, because ..."                      -> "yes"
      "At first no, but actually yes."        -> "yes"
      "uncertain output"                      -> UNPARSEABLE ("")

    Returns :data:`UNPARSEABLE` (empty string) when no label word appears,
    so downstream EM comparison yields 0.0 rather than silently defaulting
    to "no" (the prior behaviour, which inflated StrategyQA accuracy on
    unparseable outputs -- audit report MAJOR-M3).
    """
    best_label = UNPARSEABLE
    best_pos = -1
    for m in re.finditer(r"\byes\b", text, flags=re.IGNORECASE):
        if m.start() > best_pos:
            best_pos = m.start()
            best_label = "yes"
    for m in re.finditer(r"\bno\b", text, flags=re.IGNORECASE):
        if m.start() > best_pos:
            best_pos = m.start()
            best_label = "no"
    return best_label

def extract_arc_label(text: str, *, n_choices: int = 4) -> str:
    """Extract a multi-choice answer letter (A-D or A-E) from free-form output.

    Letter first, digit fallback: a combined ``[A-Da-d1-4]`` regex misfires on
    numbered CoT rationales like "1. Photosynthesis ... 2. The answer is B.",
    where the regex locks onto the list bullet ``1`` before reaching ``B`` and
    silently scores the sample wrong. We therefore search for an isolated
    letter first, then only fall back to a digit match when no letter is
    present.

    Parameters
    ----------
    text : str
        The model's free-form output (with or without "Reasoning: ..." /
        "Answer: ..." structure; ``extract_cot_answer`` strips the prefix).
    n_choices : int, default 4
        v2 Fix 10 — number of choices in the multi-choice schema. The
        default 4 preserves ARC-Challenge / OpenBookQA / MMLU behaviour;
        pass ``n_choices=5`` for CommonsenseQA so the regex accepts E /
        digit 5. Anything outside [2, 5] is clamped to [4, 5] because the
        verifier-side multichoice scorer regex caps at A-E.
    """
    cot_final = extract_cot_answer(text).strip()

    n_choices = max(4, min(int(n_choices), 5))
    if n_choices == 5:
        letter_pat = r"\b([A-Ea-e])\b"
        digit_pat = r"\b([1-5])\b"
        digit_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    else:
        letter_pat = r"\b([A-Da-d])\b"
        digit_pat = r"\b([1-4])\b"
        digit_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}

    letter_match = re.search(letter_pat, cot_final)
    if letter_match:
        return letter_match.group(1).upper()

    digit_match = re.search(digit_pat, cot_final)
    if digit_match:
        return digit_to_letter[digit_match.group(1)]

    return ""



# -----------------------------------------------------------------------------
# Hallucination rate (operational proxy)
# -----------------------------------------------------------------------------

_EVASIVE_RX = re.compile(
    r"the\s+context\s+(?:does\s*not|doesn'?t)\s+(?:mention|provide|contain|discuss)|"
    r"no\s+information\s+(?:is\s+)?(?:provided|given|available|mentioned)|"
    r"information\s+(?:is\s+)?not\s+provided|"
    r"cannot\s+(?:determine|verify|be\s+determined)",
    re.IGNORECASE,
)
_TEMPLATE_LEAK_RX = re.compile(
    r"<(?:concise|factual|your|scaffold|answer)[a-z_ ]*>|"
    r"\[(?:your\s+answer|answer\s+here)\]",
    re.IGNORECASE,
)


def hallucination_subtypes(
    samples: Sequence[Dict[str, Any]],
    *,
    u_threshold: float = 0.50,
    p_ground_atomic_threshold: float = 0.30,
    p_entail_threshold: float = 0.30,
    p_contra_threshold: float = 0.30,
    q_a_relevance_threshold: float = 0.50,
    long_prediction_chars: int = 600,
) -> Dict[str, float]:
    """Decompose hallucination rate into 7 measurable subtypes.

    Each rate = #samples matching the subtype condition / #samples total.
    Some subtypes are not mutually exclusive — a single sample may count
    in multiple subtypes (e.g., a confident wrong off-topic answer counts
    in `confident_confabulation_rate`, `off_topic_rate`, and possibly
    `factual_fabrication_rate`). The aggregate `hallucination_rate` is
    NOT the sum; subtypes overlap deliberately to surface failure modes.

    Subtypes
    --------
    confident_confabulation_rate
        EM=0 AND u_stored ≥ u_threshold. Same as hallucination_rate(),
        kept here for symmetry. Costliest — model wrong + apparently
        confident enough to pass the gate.
    factual_fabrication_rate
        EM=0 AND p_ground_atomic < p_ground_atomic_threshold. Atomic
        claims in the answer fail per-claim grounding — model invented
        sub-facts.
    factual_contradiction_rate
        EM=0 AND p_contra > p_contra_threshold. Retrieved passages
        actively contradict the answer. Note: under MiniCheck (the
        default), p_contra ≈ 0 by construction so this rate is 0;
        meaningful only under the roberta_nli_backend ablation.
    logical_fabrication_rate
        EM=0 AND p_entail < p_entail_threshold. Reasoning chain doesn't
        derive the answer — answer doesn't logically follow.
    off_topic_rate
        EM=0 AND q_a_relevance < q_a_relevance_threshold. Answer is
        topically unrelated to the question.
    defensive_evasion_rate
        EM=0 AND prediction matches evasive pattern regex. "Context
        doesn't mention X" instead of attempting an answer when evidence
        was actually present.
    template_leak_rate
        Prediction matches template-leak regex regardless of EM.
        Catches placeholder strings (<concise factual answer>) that
        leaked from the prompt template.
    false_refusal_rate
        Decision in {ABSTAIN, DISCARD} AND EM=1.0. Model refused to
        answer when it actually had the right answer in its prediction.
    over_long_rate
        EM=0 AND len(prediction) > long_prediction_chars. Verbose
        padding correlated with wrongness.

    Returns
    -------
    Dict[str, float] -- per-subtype rate ∈ [0, 1]
    """
    if not samples:
        return {}
    n = len(samples)
    counts = {
        "confident_confabulation_rate": 0,
        "factual_fabrication_rate":     0,
        "factual_contradiction_rate":   0,
        "logical_fabrication_rate":     0,
        "off_topic_rate":               0,
        "defensive_evasion_rate":       0,
        "template_leak_rate":           0,
        "false_refusal_rate":           0,
        "over_long_rate":               0,
    }
    for s in samples:
        em = s.get("em") or 0.0
        is_wrong = em == 0.0
        is_correct = em == 1.0
        u_stored = s.get("u_stored") or 0.0
        p_ga = s.get("p_ground_atomic")
        p_pe = s.get("p_entail")
        p_co = s.get("p_contra") or 0.0
        q_ar = s.get("q_a_relevance")
        decision = s.get("decision")
        pred = s.get("prediction") or ""

        if is_wrong and u_stored >= u_threshold:
            counts["confident_confabulation_rate"] += 1
        if is_wrong and p_ga is not None and p_ga < p_ground_atomic_threshold:
            counts["factual_fabrication_rate"] += 1
        if is_wrong and p_co > p_contra_threshold:
            counts["factual_contradiction_rate"] += 1
        if is_wrong and p_pe is not None and p_pe < p_entail_threshold:
            counts["logical_fabrication_rate"] += 1
        if is_wrong and q_ar is not None and q_ar < q_a_relevance_threshold:
            counts["off_topic_rate"] += 1
        if is_wrong and _EVASIVE_RX.search(pred):
            counts["defensive_evasion_rate"] += 1
        if _TEMPLATE_LEAK_RX.search(pred):
            counts["template_leak_rate"] += 1
        if is_correct and decision in ("ABSTAIN", "DISCARD"):
            counts["false_refusal_rate"] += 1
        if is_wrong and len(pred) > long_prediction_chars:
            counts["over_long_rate"] += 1
    return {k: v / n for k, v in counts.items()}


CHM_DEFAULT_MEASURED_SUBTYPES: Tuple[str, ...] = (
    "confident_confabulation_rate",
    "factual_fabrication_rate",
    # "factual_contradiction_rate" -- EXCLUDED from default CHM denominator.
    # Under the shipping MiniCheck backend, `p_contra` is structurally 0.0
    # (binary supported/unsupported judge, no contradiction class), so this
    # subtype would dilute CHM with a flatlined zero. It remains in
    # `per_subtype_rates` for transparency and is activated only under the
    # `roberta_nli_backend` ablation. CHM denominator = 8 by default.
    "logical_fabrication_rate",
    "off_topic_rate",
    "defensive_evasion_rate",
    "template_leak_rate",
    "false_refusal_rate",
    "over_long_rate",
)


def composite_hallucination_metric(
    samples: Sequence[Dict[str, Any]],
    *,
    weight_overrides: Optional[Dict[str, float]] = None,
    measured_subtypes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Single composite hallucination metric — equal-weighted mean prevalence
    across the 8 taxonomy subtypes measurable under the CAEM-default
    MiniCheck backend.

    Definition
    ----------
        CHM = (1/N) × Σ subtype_rate_i        (default: equal weights, N=8)

    Subtypes overlap: a single bad sample can count in multiple. CHM
    therefore measures *mean failure-mode prevalence* across the taxonomy,
    NOT the fraction of samples with any hallucination (that would be the
    union rate, computed separately as `union_rate` in the returned dict).

    Why equal weights
    -----------------
    Each taxonomy subtype represents a structurally distinct failure
    mode. Weighting one subtype higher than another would privilege a
    specific definition (e.g., Farquhar's confident confabulation) over
    others (e.g., off-topic answers). Equal weighting tracks reduction
    uniformly across the taxonomy.

    Coverage of the 9-subtype measurable taxonomy
    ----------------------------------------------
        Measured under default backend (8, in CHM denominator):
          confident_confabulation, factual_fabrication,
          logical_fabrication, off_topic, defensive_evasion,
          template_leak, false_refusal, over_long
        Requires roberta_nli ablation (1, excluded from default CHM):
          factual_contradiction   -- MiniCheck's binary
                                     supported/unsupported output has no
                                     contradiction class; p_contra == 0.0
                                     under the default backend.
        Documented taxonomy but not instrumented (future work):
          source_fabrication, sycophancy

    Callers needing the 9-axis mean (e.g., roberta_nli ablation runs)
    pass ``measured_subtypes=list(hallucination_subtypes(samples).keys())``
    to include factual_contradiction_rate in the denominator.

    Returns
    -------
    Dict with keys:
        chm                       — equal-weighted mean of subtype rates
        union_rate                — fraction with ANY subtype firing
        n_measured_subtypes       — 8 by default
        per_subtype_rates         — full 9-subtype dict (including the
                                    excluded factual_contradiction_rate,
                                    reported transparently even when not
                                    in the CHM denominator)
        weights                   — equal 1/N or override
    """
    if not samples:
        return {
            "chm": 0.0, "union_rate": 0.0,
            "n_measured_subtypes": 0,
            "per_subtype_rates": {}, "weights": {},
        }

    rates = hallucination_subtypes(samples)
    if measured_subtypes is not None:
        measured = list(measured_subtypes)
    else:
        measured = list(CHM_DEFAULT_MEASURED_SUBTYPES)
    measured = [m for m in measured if m in rates]

    # Equal weights by default; override if specified
    if weight_overrides:
        weights = {m: float(weight_overrides.get(m, 1.0)) for m in measured}
    else:
        weights = {m: 1.0 for m in measured}
    weight_sum = sum(weights.values()) or 1.0
    norm_weights = {m: weights[m] / weight_sum for m in measured}

    chm = sum(norm_weights[m] * rates[m] for m in measured)

    # Union rate: fraction of samples where ANY subtype condition fires.
    # Re-walk the samples to count "at least one issue" per sample.
    if not samples:
        union = 0.0
    else:
        any_count = 0
        for s in samples:
            em = s.get("em") or 0.0
            is_wrong = em == 0.0
            is_correct = em == 1.0
            u_stored = s.get("u_stored") or 0.0
            p_ga = s.get("p_ground_atomic")
            p_pe = s.get("p_entail")
            p_co = s.get("p_contra") or 0.0
            q_ar = s.get("q_a_relevance")
            decision = s.get("decision")
            pred = s.get("prediction") or ""
            if (
                (is_wrong and u_stored >= 0.50) or
                (is_wrong and p_ga is not None and p_ga < 0.30) or
                (is_wrong and p_co > 0.30) or
                (is_wrong and p_pe is not None and p_pe < 0.30) or
                (is_wrong and q_ar is not None and q_ar < 0.50) or
                (is_wrong and _EVASIVE_RX.search(pred)) or
                _TEMPLATE_LEAK_RX.search(pred) or
                (is_correct and decision in ("ABSTAIN", "DISCARD")) or
                (is_wrong and len(pred) > 600)
            ):
                any_count += 1
        union = any_count / len(samples)

    return {
        "chm": chm,
        "union_rate": union,
        "n_measured_subtypes": len(measured),
        "per_subtype_rates": rates,
        "weights": norm_weights,
    }


# Note: the standalone `hallucination_rate(em, u, 0.5)` helper was removed
# 2026-04-24. It is mathematically identical to
# `confabulation_rate(em, u, threshold=0.5, direction="ge")` (kept below) and
# to `hallucination_subtypes(...)["confident_confabulation_rate"]`, so three
# public names referred to the same scalar. The taxonomy decomposition in
# `composite_hallucination_metric()` is now the headline hallucination metric
# (CHM, equal-weighted mean of 9 measurable subtypes); the per-axis
# confident-error rate that drives CES.EPI uses `confabulation_rate` directly.


# -----------------------------------------------------------------------------
# Routing distribution
# -----------------------------------------------------------------------------

def routing_distribution(tiers: Sequence[int]) -> Dict[str, float]:
    """Compute fraction of queries routed to each tier.

    Any value not in ``{1, 2, 3}`` is treated as a pipeline-crash sample
    (sentinel ``tier = -1`` written by ``EvalHarness._run_one`` when
    ``pipeline.answer()`` raises with ``fail_on_error=False``) and is
    EXCLUDED from the per-tier denominator. The crash fraction is
    reported separately as ``crash_frac = n_crashes / n_total`` so that
    ``tier1_frac + tier2_frac + tier3_frac == 1.0`` over successful
    samples and the Ch5 methodology section can report crash rate as
    a distinct column rather than silently inflating Tier 3 routing.

    Audit MAJOR-H2 / Task #117 — prior behaviour conflated crashes with
    Tier 3 routes because the except block in the harness wrote
    ``tier = 3`` on any exception.

    Parameters
    ----------
    tiers : sequence of int -- one value per query; 1/2/3 are real routes,
        anything else (typically -1) is a pipeline-crash sentinel.

    Returns
    -------
    dict with keys ``tier1_frac``, ``tier2_frac``, ``tier3_frac``,
    ``crash_frac``. All fractions are in [0, 1].
    """
    if not tiers:
        return {
            "tier1_frac": 0.0, "tier2_frac": 0.0, "tier3_frac": 0.0,
            "crash_frac": 0.0,
        }

    n_total = len(tiers)
    counts = Counter(tiers)
    n_crashes = sum(c for t, c in counts.items() if t not in (1, 2, 3))
    n_valid = n_total - n_crashes

    if n_valid == 0:
        return {
            "tier1_frac": 0.0, "tier2_frac": 0.0, "tier3_frac": 0.0,
            "crash_frac": n_crashes / n_total,
        }

    return {
        "tier1_frac": counts.get(1, 0) / n_valid,
        "tier2_frac": counts.get(2, 0) / n_valid,
        "tier3_frac": counts.get(3, 0) / n_valid,
        "crash_frac": n_crashes / n_total,
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
    benchmark : str -- one of "fever", "triviaqa", "natural_questions",
                       "truthfulqa", "strategyqa", or "arc_challenge"
    em_scores : per-sample EM (0.0/1.0)
    f1_scores : per-sample token F1; set to em_scores when F1 is unused
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
    # Hallucination metrics are NOT carried here — they decompose into the
    # 9 taxonomy subtypes computed at the reporting layer by
    # build_table_halluc_subtypes(), which has access to the per-sample
    # signal dicts that the subtype conditions need. CHM (headline) +
    # per-subtype rates live in tab_halluc_subtypes.csv. `aggregate()`
    # stays as a lightweight per-run summary: accuracy, routing, storage,
    # latency.
    valid_u = [u for u in u_stored_values if u is not None]

    return {
        "benchmark": benchmark,
        "n": n,
        "em": round(sum(em_scores) / n, 4),
        "f1": round(sum(f1_scores) / n, 4),
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
    p_value = float(1.0 - chi2.cdf(statistic, df=1))
    return float(statistic), p_value


# =====================================================================
# Phase 4m.1 -- Extended metric suite for top-venue evaluation
# ---------------------------------------------------------------------
# The helpers below complement the core `aggregate()` block above. They
# expose the architectural signals that CAEM generates but that a single
# exact-match + hallucination_rate pair does not surface. Each helper is
# annotated with (a) which evaluation task it serves and (b) where it
# lands in the five-table reviewer layout designed in Phase 4l.
#
# Task/table mapping (see docs/metrics-audit.md and Ch.5 methodology):
#
#   Table 5.1 Headline         -- em_by_tier, mean_latency_by_tier
#   Table 5.2 Calibration      -- brier_score, auroc, reliability_bins
#   Table 5.3 Data Purity      -- decision_breakdown
#   Table 5.4 Grounding        -- unsupported_correct_rate,
#                                  ungrounded_assertion_rate
#   Table 5.5 Hallucination    -- confabulation_rate (per-signal variants)
#   Continual-learning panel   -- backward_transfer, forward_transfer
#   Unified efficacy scalar    -- ces_score
#
# All helpers are pure functions, take plain lists, and never depend on
# torch / transformers at import time so they can be reused from
# scripts/make_tables.py and analysis notebooks without loading the
# model weights.
# =====================================================================


def em_by_tier(em_scores, tiers):
    """Exact-match accuracy stratified by routing tier.

    CAEM's whole premise is that Tier 1 (memory hit) should dominate
    Tier 2 (deferred) and Tier 3 (abstain) on EM, because stored entries
    carry verified high-confidence answers. Reporting a single pooled EM
    hides that selectivity. This helper is used by Table 5.1 to show the
    EM-by-tier triple (Tier1, Tier2, Tier3) per benchmark.

    Parameters
    ----------
    em_scores : list of float
        Per-sample EM in {0.0, 1.0}.
    tiers : list of int
        Per-sample routing decision in {1, 2, 3}.

    Returns
    -------
    dict
        {'tier1': (em, n), 'tier2': (em, n), 'tier3': (em, n)}.
        Returns (0.0, 0) for empty tiers so downstream formatting
        does not crash on the first cycle when Tier 1 is empty.
    """
    assert len(em_scores) == len(tiers), "em_scores and tiers must align."
    out = {}
    for label, t in (("tier1", 1), ("tier2", 2), ("tier3", 3)):
        bucket = [s for s, tt in zip(em_scores, tiers) if tt == t]
        if bucket:
            out[label] = (sum(bucket) / len(bucket), len(bucket))
        else:
            out[label] = (0.0, 0)
    return out


def mean_latency_by_tier(latencies_ms, tiers):
    """Per-tier mean wall-clock latency (ms).

    Tier 1 should be ~10x faster than Tier 2 because memory retrieval
    skips full decoding. This helper feeds the latency column of Table
    5.1 and is required for the efficiency claim in the abstract.
    """
    assert len(latencies_ms) == len(tiers), "latencies and tiers must align."
    out = {}
    for label, t in (("tier1", 1), ("tier2", 2), ("tier3", 3)):
        bucket = [lat for lat, tt in zip(latencies_ms, tiers) if tt == t]
        out[label] = (sum(bucket) / len(bucket)) if bucket else 0.0
    return out


def confabulation_rate(em_scores, signal_values, threshold, direction="ge"):
    """Generic confident-error / confabulation rate.

    Follows Farquhar et al. 2024: a confabulation is an answer that is
    (a) wrong and (b) produced with high internal confidence / low
    uncertainty. The concrete "high confidence" criterion depends on
    which signal drives it -- this helper abstracts the shape so the
    same function is used for u_stored-based, u_internal-based, and
    semantic-entropy-based confabulation rates reported in Table 5.5.

    Parameters
    ----------
    em_scores : list of float
        Per-sample EM in {0.0, 1.0}.
    signal_values : list of float
        Per-sample confidence-like signal (e.g. u_stored, 1-H_sem).
    threshold : float
        Cutoff defining "high confidence".
    direction : {"ge", "le"}
        "ge" -- confident when signal >= threshold (e.g. u_stored >= 0.5)
        "le" -- confident when signal <= threshold (e.g. H_sem <= 0.3)

    Returns
    -------
    float
        Fraction of samples that are wrong AND confident.
    """
    assert direction in ("ge", "le"), "direction must be 'ge' or 'le'."
    assert len(em_scores) == len(signal_values), "lengths must align."
    if not em_scores:
        return 0.0
    if direction == "ge":
        hits = sum(1 for em, s in zip(em_scores, signal_values)
                   if em == 0.0 and s >= threshold)
    else:
        hits = sum(1 for em, s in zip(em_scores, signal_values)
                   if em == 0.0 and s <= threshold)
    return hits / len(em_scores)


def brier_score(confidences, labels):
    """Brier score = mean squared error between confidence and correctness.

    Proper scoring rule (Brier 1950). Sensitive to both calibration and
    resolution; complements ECE, which can mask counter-balanced errors
    inside bins. Reported in Table 5.2 alongside ECE and AUROC to form
    the calibration panel.

    Parameters
    ----------
    confidences : list of float in [0, 1]
    labels : list of float in {0.0, 1.0}

    Returns
    -------
    float in [0, 1]; lower is better.
    """
    assert len(confidences) == len(labels), "lengths must align."
    if not confidences:
        return 0.0
    return sum((c - y) ** 2 for c, y in zip(confidences, labels)) / len(confidences)


def auroc(confidences, labels):
    """Area under the ROC curve for confidence-as-correctness predictor.

    Measures whether a higher confidence score reliably corresponds to
    a correct answer -- a discrimination metric independent of
    calibration scale. Useful because a model can be mis-calibrated
    (wrong scale) but still rank-correct (good AUROC), which matters
    for CAEM's STORE/DEFER thresholding.

    Returns 0.5 if there is only one class present (undefined AUROC).

    Requires scikit-learn; if unavailable, falls back to a pure-python
    Mann-Whitney U implementation equivalent to AUROC.
    """
    assert len(confidences) == len(labels), "lengths must align."
    if not confidences:
        return 0.5
    pos = [c for c, y in zip(confidences, labels) if y == 1.0]
    neg = [c for c, y in zip(confidences, labels) if y == 0.0]
    if not pos or not neg:
        return 0.5

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, confidences))
    except ImportError:
        # Mann-Whitney U fallback: fraction of (pos, neg) pairs where
        # pos confidence strictly beats neg confidence, with 0.5 credit
        # for ties.
        wins = 0.0
        for p in pos:
            for n in neg:
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
        return wins / (len(pos) * len(neg))


def reliability_bins(confidences, labels, n_bins=10):
    """Equal-width reliability-diagram bins (Guo et al. 2017, ICML).

    Returns one row per bin containing (mean_conf, mean_acc, count)
    so Figure 3 in the paper can plot the classic reliability diagram
    (mean confidence on x-axis, empirical accuracy on y-axis, diagonal
    = perfect calibration). This is the standard visualization that
    grounds the ECE number reported in Table 5.2.

    Parameters
    ----------
    confidences : list of float in [0, 1]
    labels : list of float in {0.0, 1.0}
    n_bins : int
        Default 10 (i.e. [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]).

    Returns
    -------
    list of (mean_conf, mean_acc, count) tuples of length n_bins.
    Empty bins produce (bin_midpoint, 0.0, 0).
    """
    assert len(confidences) == len(labels), "lengths must align."
    assert n_bins >= 1, "n_bins must be positive."
    width = 1.0 / n_bins
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, labels):
        # Clamp c to [0, 1] defensively; last bin is closed on the right.
        cc = min(max(c, 0.0), 1.0)
        idx = min(int(cc / width), n_bins - 1)
        bins[idx].append((cc, y))
    out = []
    for i, rows in enumerate(bins):
        if rows:
            mc = sum(r[0] for r in rows) / len(rows)
            ma = sum(r[1] for r in rows) / len(rows)
            out.append((mc, ma, len(rows)))
        else:
            out.append(((i + 0.5) * width, 0.0, 0))
    return out


def decision_breakdown(decisions):
    """Counts of UnifiedVerifier decisions: STORE / DEFERRED / ABSTAIN / DISCARD.

    Feeds Table 5.3 (Data Purity Theorem). Purity analysis needs the
    raw STORE count and the correctness rate among stored items; this
    helper provides the denominator and the rejection-mass tallies
    (abstain, discard, deferred) that justify the "selective retention"
    framing in Ch 4.

    Labels match :class:`caem.unified_verifier.UnifiedVerifierOutput`
    exactly (Session-42 canonical schema). The prior "DEFER" key was a
    pre-Session-42 leftover that silently dropped every real deferred
    item, zeroing the deferred-rate column in Table 5.3 and forcing
    ``eval/reporting.py`` to maintain a parallel tally. Audit MAJOR-M4.

    Parameters
    ----------
    decisions : list of str
        Each element in {"STORE", "DEFERRED", "ABSTAIN", "DISCARD"}.

    Returns
    -------
    dict with counts and normalized fractions.
    """
    total = len(decisions)
    out = {k: 0 for k in ("STORE", "DEFERRED", "ABSTAIN", "DISCARD")}
    for d in decisions:
        if d in out:
            out[d] += 1
    result = {"counts": dict(out), "total": total}
    if total > 0:
        result["fractions"] = {k: v / total for k, v in out.items()}
    else:
        result["fractions"] = {k: 0.0 for k in out}
    return result


def backward_transfer(per_cycle_em_matrix):
    """BWT score (Lopez-Paz & Ranzato 2017, NeurIPS).

    Backward transfer measures how learning new cycles affects earlier
    benchmarks. Positive BWT = later training *improves* earlier
    benchmarks (ideal for CAEM's self-improvement claim). Negative BWT
    = catastrophic forgetting.

    Formula (for T cycles, measured on T benchmarks):
        BWT = 1/(T-1) * sum_{i=1..T-1} (R[T,i] - R[i,i])
    where R[k,i] is EM on benchmark i after cycle k.

    This is the gold-standard continual-learning metric; complements
    our MMLU retention ratio (which is cycle-0-anchored and domain-
    neutral) by tracking per-benchmark-per-cycle drift.

    Parameters
    ----------
    per_cycle_em_matrix : list of list of float
        R[k][i] = EM on benchmark i after cycle k. Square matrix of
        shape (T, T).

    Returns
    -------
    float
        BWT score; 0.0 if T < 2 or matrix empty.
    """
    T = len(per_cycle_em_matrix)
    if T < 2:
        return 0.0
    # Expect square, but be lenient: use min(T, cols).
    diffs = []
    for i in range(T - 1):
        row_final = per_cycle_em_matrix[T - 1]
        row_diag = per_cycle_em_matrix[i]
        if i < len(row_final) and i < len(row_diag):
            diffs.append(row_final[i] - row_diag[i])
    if not diffs:
        return 0.0
    return sum(diffs) / len(diffs)


def forward_transfer(per_cycle_em_matrix, baseline_em=None):
    """FWT score (Lopez-Paz & Ranzato 2017, NeurIPS).

    Forward transfer measures whether pretraining on earlier cycles
    helps performance on benchmarks not yet seen. Formally:
        FWT = 1/(T-1) * sum_{i=2..T} (R[i-1,i] - bbar[i])
    where bbar[i] is the random/zero-shot baseline for benchmark i.

    If no baseline is supplied we use the cycle-0 diagonal (i.e. the
    pristine-model EM for that benchmark), which is how the paper
    will compute it given CAEM always evaluates all benchmarks on
    every cycle.

    Parameters
    ----------
    per_cycle_em_matrix : list of list of float
        R[k][i] = EM on benchmark i after cycle k.
    baseline_em : list of float or None
        bbar[i] per benchmark. If None, uses R[0][i].

    Returns
    -------
    float
        FWT score; 0.0 if T < 2.
    """
    T = len(per_cycle_em_matrix)
    if T < 2:
        return 0.0
    bbar = baseline_em if baseline_em is not None else per_cycle_em_matrix[0]
    diffs = []
    for i in range(1, T):
        row_prev = per_cycle_em_matrix[i - 1]
        if i < len(row_prev) and i < len(bbar):
            diffs.append(row_prev[i] - bbar[i])
    if not diffs:
        return 0.0
    return sum(diffs) / len(diffs)


def pooled_chm(
    samples_by_benchmark: Dict[str, Sequence[Dict[str, Any]]],
    *,
    training_benchmarks: Optional[Sequence[str]] = None,
    transfer_benchmarks: Optional[Sequence[str]] = None,
    measured_subtypes: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Pooled CHM with ID/OOD transfer-learning split.

    Supersedes the prior `pooled_ce` helper (CE was one subtype of the 8
    CAEM measures; CHM is the equal-weighted mean across the taxonomy).
    "Pooled" = samples from every benchmark thrown into one bucket before
    CHM is computed, so small benchmarks don't dominate and distributional
    shift surfaces honestly.

    Parameters
    ----------
    samples_by_benchmark : dict[str, list of sample dicts]
        Per-sample records keyed by benchmark name. Each dict must carry
        the verifier fields that ``hallucination_subtypes`` reads
        (em, u_stored, p_ground_atomic, p_entail, p_contra, q_a_relevance,
        decision, prediction).
    training_benchmarks, transfer_benchmarks : sequence of str or None
        Override the split; defaults to caem.config constants.
    measured_subtypes : sequence of str or None
        Passed through to composite_hallucination_metric. Default keeps
        the 8-subtype denominator (factual_contradiction excluded under
        MiniCheck). Pass the full 9-key list for the roberta_nli ablation.

    Returns
    -------
    dict with keys ``"all"``, ``"id"``, ``"ood"`` → float CHM in [0, 1],
    plus ``n_all`` / ``n_id`` / ``n_ood`` sample counts. A split is 0.0
    if no samples fall into it.
    """
    if training_benchmarks is None or transfer_benchmarks is None:
        from caem.config import (
            TRAINING_BENCHMARKS as _TRAIN,
            TRANSFER_BENCHMARKS as _OOD,
        )
        if training_benchmarks is None:
            training_benchmarks = _TRAIN
        if transfer_benchmarks is None:
            transfer_benchmarks = _OOD

    train_set = set(training_benchmarks)
    ood_set = set(transfer_benchmarks)

    all_samples: List[Dict[str, Any]] = []
    id_samples:  List[Dict[str, Any]] = []
    ood_samples: List[Dict[str, Any]] = []
    for bm, samples in samples_by_benchmark.items():
        all_samples.extend(samples)
        if bm in train_set:
            id_samples.extend(samples)
        if bm in ood_set:
            ood_samples.extend(samples)

    def _chm(samples: Sequence[Dict[str, Any]]) -> float:
        if not samples:
            return 0.0
        out = composite_hallucination_metric(
            samples, measured_subtypes=measured_subtypes,
        )
        return float(out.get("chm", 0.0))

    return {
        "all": _chm(all_samples),
        "id":  _chm(id_samples),
        "ood": _chm(ood_samples),
        "n_all": len(all_samples),
        "n_id":  len(id_samples),
        "n_ood": len(ood_samples),
    }


def chm_reduction_verdict(
    chm_baseline: float,
    chm_final: float,
    floor: float = 0.30,
) -> Dict[str, object]:
    """Broader hallucination pass/fail verdict: relative reduction in CHM.

    Single hallucination pass/fail gate for the thesis (superseding the
    prior `ce_reduction_verdict`, which measured the same thing on a
    single subtype). CHM is the 8-axis equal-weighted taxonomy mean
    computed by :func:`composite_hallucination_metric`; the old
    `confident_error_rate` that drove the 40% Ch3 gate is now just
    `per_subtype_rates["confident_confabulation_rate"]` inside CHM.

    Default floor is 0.30 (30% relative reduction), more permissive than
    the CE gate (0.40) because CHM is a mean over 8 structurally distinct
    failure modes — a uniform 30% drop across the taxonomy is already a
    stronger architectural claim than a 40% drop on a single axis.

    Parameters
    ----------
    chm_baseline : float in [0, 1]
        CHM at cycle 0 (or baseline system). Pulled from
        ``composite_hallucination_metric(...)["chm"]``.
    chm_final : float in [0, 1]
        CHM at the post-cycle CAEM system.
    floor : float in (0, 1], default 0.30
        Required relative reduction.

    Returns
    -------
    dict with:
        verdict, relative_reduction, absolute_reduction, achieved, floor,
        chm_baseline, chm_final.
    """
    if chm_baseline <= 0.0:
        return {
            "verdict": "FAIL",
            "relative_reduction": 0.0,
            "achieved": 0.0,
            "absolute_reduction": 0.0 - chm_final,
            "floor": floor,
            "chm_baseline": chm_baseline,
            "chm_final": chm_final,
        }

    rel = (chm_baseline - chm_final) / chm_baseline
    rel_clamped = max(rel, 0.0)
    verdict = "PASS" if rel >= floor else "FAIL"
    return {
        "verdict": verdict,
        "relative_reduction": rel_clamped,
        "achieved": rel_clamped,
        "absolute_reduction": chm_baseline - chm_final,
        "floor": floor,
        "chm_baseline": chm_baseline,
        "chm_final": chm_final,
    }


def ces_score(acc, epi, ret, cal, ver, eps=0.01):
    """CAEM Efficacy Score: geometric mean of five orthogonal quality axes.

    Motivation: reviewers of a five-table result panel need a single
    scalar to anchor the headline claim "CAEM improves across cycles"
    without cherry-picking a favorable metric. CES is a geometric
    mean so that any near-zero axis (e.g. catastrophic forgetting
    collapsing RET) drags the whole score down -- multiplicative
    rather than additive composition punishes failure modes rather
    than masking them.

    Axes:
        ACC -- accuracy  : mean EM across eval benchmarks, in [0, 1]
        EPI -- epistemic : 1 - CHM (8-subtype composite hallucination
                           metric), clamped to [0, 1]
        RET -- retention : min(mmlu_retention_ratio, 1.0), in [0, 1]
        CAL -- calibration: 1 - 2 * min(ECE, 0.5), in [0, 1]
                            (so ECE=0 -> 1.0, ECE=0.5+ -> 0.0)
        VER -- verifier  : max(2*balanced_accuracy - 1, 0), in [0, 1]
                           (Youden's J rescaled; chance -> 0)

    Components are clamped to [eps, 1] with eps = 0.01 to prevent a
    single degenerate axis from zeroing the score (useful in
    smoke-test regimes where one axis may be uncomputed).

    Returns
    -------
    float in [eps, 1]; higher is better.

    Notes
    -----
    CES is a custom aggregate metric introduced by this thesis. It is
    NOT a substitute for the per-table results -- it is a visualization
    and headline tool only. The thesis reports CES alongside all five
    component values so readers can reconstruct the geometric mean.
    """
    comps = [acc, epi, ret, cal, ver]
    clamped = [min(max(c, eps), 1.0) for c in comps]
    prod = 1.0
    for c in clamped:
        prod *= c
    return prod ** (1.0 / len(clamped))


def unsupported_correct_rate(em_scores, p_ground_max_values, ground_threshold=0.30):
    """Fraction of correct answers that lack retrieval grounding.

    "Right for the wrong reason": the model produced a correct answer
    but retrieval evidence did not support it. This is a diagnostic
    of over-reliance on parametric memory -- not necessarily a failure,
    but an honesty signal for the grounding discussion in Table 5.4.

    Parameters
    ----------
    em_scores : list of float in {0.0, 1.0}
    p_ground_max_values : list of float in [0, 1]
        Maximum entailment probability across retrieved passages.
    ground_threshold : float
        Below this value the answer is considered ungrounded.

    Returns
    -------
    float in [0, 1] or 0.0 if no correct answers.
    """
    assert len(em_scores) == len(p_ground_max_values), "lengths must align."
    correct = [(em, g) for em, g in zip(em_scores, p_ground_max_values) if em == 1.0]
    if not correct:
        return 0.0
    return sum(1 for _, g in correct if g < ground_threshold) / len(correct)


def ungrounded_assertion_rate(p_ground_max_values, u_values,
                              g_thresh=0.30, u_thresh=0.50):
    """Fraction of answers that are confidently asserted without grounding.

    Companion to unsupported_correct_rate. This one does NOT condition
    on EM -- it counts high-confidence, low-grounding outputs regardless
    of correctness. This is the exact early-exit-confabulation gate
    signal from Stage 5 of CAEM (u_internal high AND p_ground_max low),
    but generalized to any (u, g) pair so the same helper serves both
    the Stage 5 diagnostic and the evaluation-side Table 5.4 row.

    Parameters
    ----------
    p_ground_max_values : list of float in [0, 1]
    u_values : list of float in [0, 1]
        Any uncertainty/confidence signal; typically u_internal or
        u_stored depending on whether analysis is pre- or post-STORE.
    g_thresh : float
        Upper bound for "ungrounded".
    u_thresh : float
        Lower bound for "confidently asserted".

    Returns
    -------
    float in [0, 1].
    """
    assert len(p_ground_max_values) == len(u_values), "lengths must align."
    n = len(p_ground_max_values)
    if n == 0:
        return 0.0
    hits = sum(1 for g, u in zip(p_ground_max_values, u_values)
               if g <= g_thresh and u >= u_thresh)
    return hits / n
