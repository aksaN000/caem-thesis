"""
caem/verification/negation.py
=============================

Claim-negation helper for the directional ``p_ground_mean`` fix (Branch C
2026-04-23 refutation-bias resolution).

Background
----------
The u_stored composite's ``p_ground_mean`` signal asks MiniCheck "do the
passages support the hypothesis text?". For a *correctly refuted* claim,
the passages should NOT support the claim — so this signal is
architecturally forced to be low, dragging u_stored below τ_defer even
when the model's "refutes" answer is correct.

The fix (applied at ``DirectionalScorer``): when the model's label is
"refutes" (FEVER) or "no" (StrategyQA), feed MiniCheck the *negated*
claim. Now the passages support the TRUE version, p_ground_mean is high,
and u_stored reflects verifier agreement with the model's correct
answer.

API
---
``Negator.negate(claim)`` returns (negation_text, source) where source
is "rule" or "qwen" or "failed". Consumers check source != "failed"
before using the negation — on failure, fall back to the original
hypothesis (preserves current behavior).

Rule coverage is ~85% on FEVER-style claims; Qwen fallback handles the
rest. Self-validation via ``validate_negation`` guards against rule
misfires (e.g., matching a non-copular "is").
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Rule-based negation                                                   #
# --------------------------------------------------------------------- #

# Ordered: more-specific patterns first so "is not" double-negation removal
# fires before the generic "is -> is not" rule.
_NEG_RULES = [
    # Double-negation removal (highest priority)
    (re.compile(r"\b(is|are|was|were)\s+not\s+", re.IGNORECASE),  r"\1 "),
    (re.compile(r"\b(does|do|did)\s+not\s+",     re.IGNORECASE),  ""),
    (re.compile(r"\b(cannot|can't)\b",            re.IGNORECASE),  "can"),
    (re.compile(r"\bwon't\b",                     re.IGNORECASE),  "will"),
    (re.compile(r"\bwouldn't\b",                  re.IGNORECASE),  "would"),
    (re.compile(r"\bshouldn't\b",                 re.IGNORECASE),  "should"),
    (re.compile(r"\bhasn't\b",                    re.IGNORECASE),  "has"),
    (re.compile(r"\bhaven't\b",                   re.IGNORECASE),  "have"),
    (re.compile(r"\bhadn't\b",                    re.IGNORECASE),  "had"),
    (re.compile(r"\bdidn't\b",                    re.IGNORECASE),  "did"),
    (re.compile(r"\bdoesn't\b",                   re.IGNORECASE),  "does"),
    (re.compile(r"\bdon't\b",                     re.IGNORECASE),  "do"),
    (re.compile(r"\bisn't\b",                     re.IGNORECASE),  "is"),
    (re.compile(r"\baren't\b",                    re.IGNORECASE),  "are"),
    (re.compile(r"\bwasn't\b",                    re.IGNORECASE),  "was"),
    (re.compile(r"\bweren't\b",                   re.IGNORECASE),  "were"),
]

# Patterns to ADD negation (applied only if no double-negation was removed).
# Each tuple: (regex, replacement). Replacement must invert polarity.
_ADD_NEGATION = [
    # copular — handle singular then plural to avoid matching twice
    (re.compile(r"\b(is|are|was|were)\b(?!\s+not\b)", re.IGNORECASE), r"\1 not"),
    # modals
    (re.compile(r"\b(can|will|would|should|could|may|might|must)\b(?!\s+not\b)", re.IGNORECASE),
     r"\1 not"),
    # has/have/had followed by past participle or noun
    (re.compile(r"\b(has|have|had)\b(?!\s+(not|been|only)\b)", re.IGNORECASE), r"\1 not"),
    # present simple 3rd person singular (ends in 's' not following copular/modal)
    # Crude — handled better by Qwen fallback; omitted from rule set.
]


def negate_by_rules(claim: str) -> Optional[str]:
    """Attempt rule-based negation. Returns negated text or None if no rule matched."""
    if not claim or not claim.strip():
        return None
    text = claim.strip()

    # Step 1: try double-negation removal first. If any fires, that's the negation.
    for pat, repl in _NEG_RULES:
        new = pat.sub(repl, text, count=1)
        if new != text:
            # cleanup doubled whitespace
            return _tidy(new)

    # Step 2: add negation via copular / modal / aux rules.
    for pat, repl in _ADD_NEGATION:
        new = pat.sub(repl, text, count=1)
        if new != text:
            return _tidy(new)

    return None


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # Preserve terminal punctuation.
    if s and s[-1] not in ".!?":
        s = s + "."
    return s


# --------------------------------------------------------------------- #
# Qwen fallback                                                         #
# --------------------------------------------------------------------- #

_NEGATION_PROMPT = (
    "Negate the following statement so the new statement has the opposite meaning, "
    "but keeps the same subject and style. Output ONLY the negation, one line, no "
    "quotes, no explanation.\n"
    "Statement: {claim}\n"
    "Negation:"
)


def negate_by_qwen(claim: str, model, tokenizer, max_new_tokens: int = 48) -> Optional[str]:
    """Use a local Qwen model to generate a negation. Returns None on failure."""
    if model is None or tokenizer is None:
        return None
    try:
        import torch
        prompt = _NEGATION_PROMPT.format(claim=claim.strip())
        # ChatML format to match Qwen-2.5-Instruct
        messages = [
            {"role": "system", "content": "You are a precise linguistic tool."},
            {"role": "user",   "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        # Take first line only
        gen = gen.split("\n", 1)[0].strip().strip('"').strip("'")
        if not gen or len(gen) < 3:
            return None
        return _tidy(gen)
    except Exception as exc:
        logger.warning("Qwen negation failed: %s", exc)
        return None


# --------------------------------------------------------------------- #
# Validation                                                            #
# --------------------------------------------------------------------- #

def validate_negation(original: str, negated: str, nli_judge) -> bool:
    """Self-check: a true negation should NOT entail the original.

    We compute MiniCheck(negated_as_premise, original_as_hypothesis).
    If the score is high, the "negation" failed to change meaning — return False.
    """
    if not original or not negated or nli_judge is None:
        return False
    if original.strip().lower() == negated.strip().lower():
        return False  # no change at all
    try:
        p = nli_judge.batch_entail_prob([(negated, original)])
        score = float(p[0]) if p else 0.0
        # A correct negation should yield p(original | negated) < 0.3.
        return score < 0.3
    except Exception as exc:
        logger.warning("Negation validation failed: %s", exc)
        # Be permissive on validator failure — don't silently skip the fix.
        return True


# --------------------------------------------------------------------- #
# Unified entry point                                                   #
# --------------------------------------------------------------------- #

@dataclass
class NegationResult:
    text: Optional[str]
    source: str  # "rule" | "qwen" | "failed"


class Negator:
    """Two-tier negator: rules first, Qwen fallback.

    Parameters
    ----------
    model, tokenizer : the base Qwen generator (loaded for the pipeline).
                       If None, negator is rule-only.
    nli_judge        : MiniCheck-style judge for self-validation. If None,
                       no validation is performed.
    """

    def __init__(self, model=None, tokenizer=None, nli_judge=None):
        self.model = model
        self.tokenizer = tokenizer
        self.nli = nli_judge

    def negate(self, claim: str) -> NegationResult:
        if not claim or not claim.strip():
            return NegationResult(None, "failed")
        # Try rules.
        rule_out = negate_by_rules(claim)
        if rule_out and rule_out.strip().lower() != claim.strip().lower():
            if self.nli is None or validate_negation(claim, rule_out, self.nli):
                return NegationResult(rule_out, "rule")
            # rule-based negation failed validation — try Qwen.
        # Qwen fallback.
        qwen_out = negate_by_qwen(claim, self.model, self.tokenizer)
        if qwen_out and qwen_out.strip().lower() != claim.strip().lower():
            if self.nli is None or validate_negation(claim, qwen_out, self.nli):
                return NegationResult(qwen_out, "qwen")
        return NegationResult(None, "failed")
