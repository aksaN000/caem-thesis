"""
caem/production/confidence.py
==============================
Production response rendering — translates the Stage-5 decision tree
output into a user-facing confidence envelope.

Design
------
CAEM's Stage-5 decision tree (verifier.py::_decide) emits one of four
decisions per sample:

    STORE     u_stored >= tau_store                     → high-confidence, stored in memory
    DEFERRED  tau_defer <= u_stored < tau_store         → held for cycle-boundary re-check
    DISCARD   u_stored < tau_defer AND p_ground_max OK  → evidence exists but signals disagree
    ABSTAIN   u_stored < tau_defer AND p_ground_max < ε → no supporting evidence

Note: DISCARD and ABSTAIN are split by p_ground_max, NOT by u_stored. A
DISCARD can have higher u_stored than an ABSTAIN — they describe
*different failure modes*, not a single ordinal on confidence.

The production envelope exposes this to users via:

    continuous score        u_stored ∈ [0, 1]                  "72%"
    semantic tag            verified / provisional /
                            conflicting / insufficient         (failure mode)
    human label             high / moderate / low / very_low   (ordinal hint)
    icon                    check / tilde / warning / cross    (at-a-glance)
    show_answer             bool                               (hide on insufficient)
    caveat                  deployment-aware hedge text        (or None)

Cycle-based caveat policy
-------------------------
CAEM's cycle boundary is query-volume-triggered (N queries per cycle),
not wall-clock. For DEFERRED responses, the caveat defaults to a
generic "at my next learning cycle" phrasing; production deployments
can override with an explicit ETA or a named cadence (nightly / hourly
/ continuous / shortly) to produce deployment-specific user copy.

See thesis Chapter 6 §User-Facing Deployment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

DecisionTag = Literal["verified", "provisional", "conflicting", "insufficient"]
DecisionLabel = Literal["high", "moderate", "low", "very_low"]
DecisionIcon = Literal["check", "tilde", "warning", "cross"]

# Stage-5 decision -> (tag, label, icon)
_DECISION_MAP: dict[str, tuple[DecisionTag, DecisionLabel, DecisionIcon]] = {
    "STORE":    ("verified",     "high",     "check"),
    "DEFERRED": ("provisional",  "moderate", "tilde"),
    "DISCARD":  ("conflicting",  "low",      "warning"),
    "ABSTAIN":  ("insufficient", "very_low", "cross"),
}

VALID_DECISIONS = frozenset(_DECISION_MAP.keys())

_CAVEAT_DEFERRED_GENERIC = (
    "I'll reconsider this at my next learning cycle. "
    "Please ask again later for an updated answer."
)
_CAVEAT_DISCARD = (
    "Evidence exists but some signals disagree — "
    "independent verification is recommended."
)
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have enough evidence to answer this reliably."
)

# Named cadences for deployment-specific deferred caveats.
_CADENCE_PHRASING: dict[str, str] = {
    "nightly":    "in my nightly learning cycle. Please ask again tomorrow",
    "hourly":     "in my next hourly update. Please ask again in about an hour",
    "continuous": "shortly as my learning queue clears. Please ask again in a few minutes",
    "shortly":    "shortly. Please check back in a few minutes",
}


@dataclass(frozen=True)
class ProductionResponse:
    """User-facing response envelope.

    Attributes
    ----------
    answer
        The answer text to render. None when the decision hid the answer
        (insufficient-evidence / ABSTAIN).
    confidence
        Continuous u_stored, clamped to [0, 1]. Render as percentage.
    tag
        Semantic failure-mode tag: verified / provisional / conflicting /
        insufficient.
    label
        Ordinal confidence hint: high / moderate / low / very_low. A hint
        only — prefer ``tag`` + ``confidence`` for decision-grade UX.
    icon
        Short name for the accompanying icon: check / tilde / warning / cross.
    show_answer
        True when the caller should render ``answer``; False when the
        response represents a refusal to commit.
    caveat
        Optional human-readable hedge appended to the answer (None on
        verified, None on insufficient-where-answer-is-hidden).
    decision
        Raw Stage-5 decision preserved for downstream analysis. Kept
        alongside the rendered fields so tooling can correlate.
    """

    answer: Optional[str]
    confidence: float
    tag: DecisionTag
    label: DecisionLabel
    icon: DecisionIcon
    show_answer: bool
    caveat: Optional[str]
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_deferred_caveat(
    *,
    deferred_eta: Optional[str],
    deferred_cadence: Optional[str],
) -> str:
    """Choose the most-specific deferred caveat available.

    Priority: explicit ETA > named cadence > generic fallback.
    """
    if deferred_eta:
        return f"I'll reconsider this around {deferred_eta}. Please ask again then."
    if deferred_cadence:
        phrase = _CADENCE_PHRASING.get(
            deferred_cadence,
            f"in my {deferred_cadence} learning cycle. Please check back soon",
        )
        return f"I'll reconsider this {phrase}."
    return _CAVEAT_DEFERRED_GENERIC


def render(
    answer: str,
    u_stored: float,
    decision: str,
    *,
    deferred_eta: Optional[str] = None,
    deferred_cadence: Optional[str] = None,
) -> Optional[ProductionResponse]:
    """Render a CAEM pipeline decision into a production envelope.

    Parameters
    ----------
    answer
        The model's generated answer string. For ABSTAIN the answer is
        replaced with None (``show_answer=False``); the caller should
        substitute ``INSUFFICIENT_EVIDENCE_MESSAGE`` when showing to users.
    u_stored
        Composite confidence in [0, 1]. Values outside this range are
        clamped rather than raising — the underlying composite is already
        clamp-bound but upstream callers (ablations, mocks) may pass
        out-of-range values.
    decision
        One of STORE / DEFERRED / DISCARD / ABSTAIN. Any other value
        returns None (baselines without a CAEM decision, e.g. B1-B7, are
        expected to produce None here and skip the field in their output).
    deferred_eta
        Optional explicit time hint for DEFERRED (e.g., "6am tomorrow").
        Takes priority over ``deferred_cadence``.
    deferred_cadence
        Optional named cadence for DEFERRED. Recognized values:
        nightly / hourly / continuous / shortly. Other strings are
        accepted and rendered verbatim ("in my <value> learning cycle").

    Returns
    -------
    ProductionResponse or None
        None iff ``decision`` is not one of the four valid Stage-5 decisions.
    """
    if decision not in VALID_DECISIONS:
        return None

    tag, label, icon = _DECISION_MAP[decision]
    confidence = max(0.0, min(1.0, float(u_stored)))

    if decision == "STORE":
        return ProductionResponse(
            answer=answer,
            confidence=confidence,
            tag=tag,
            label=label,
            icon=icon,
            show_answer=True,
            caveat=None,
            decision=decision,
        )
    if decision == "DEFERRED":
        caveat = _resolve_deferred_caveat(
            deferred_eta=deferred_eta,
            deferred_cadence=deferred_cadence,
        )
        return ProductionResponse(
            answer=answer,
            confidence=confidence,
            tag=tag,
            label=label,
            icon=icon,
            show_answer=True,
            caveat=caveat,
            decision=decision,
        )
    if decision == "DISCARD":
        return ProductionResponse(
            answer=answer,
            confidence=confidence,
            tag=tag,
            label=label,
            icon=icon,
            show_answer=True,
            caveat=_CAVEAT_DISCARD,
            decision=decision,
        )
    # ABSTAIN: hide the answer; caller substitutes the insufficient-evidence
    # message via user_facing_answer() or its own rendering.
    return ProductionResponse(
        answer=None,
        confidence=confidence,
        tag=tag,
        label=label,
        icon=icon,
        show_answer=False,
        caveat=None,
        decision=decision,
    )


def render_dict(
    answer: str,
    u_stored: float,
    decision: str,
    *,
    deferred_eta: Optional[str] = None,
    deferred_cadence: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Convenience wrapper returning dict (for JSON serialization).

    Returns None iff ``decision`` is invalid.
    """
    resp = render(
        answer, u_stored, decision,
        deferred_eta=deferred_eta, deferred_cadence=deferred_cadence,
    )
    return resp.to_dict() if resp is not None else None


def user_facing_answer(response: ProductionResponse) -> str:
    """Return the text to actually show the user.

    For ABSTAIN (``show_answer=False``), substitutes the insufficient-
    evidence message. Otherwise returns ``response.answer`` (or empty
    string if somehow None on a show-answer path).
    """
    if response.show_answer and response.answer is not None:
        return response.answer
    return INSUFFICIENT_EVIDENCE_MESSAGE
