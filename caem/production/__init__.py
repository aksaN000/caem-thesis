"""caem.production — user-facing confidence envelope + serving wrappers.

This package translates the research pipeline's raw decisions (STORE /
DEFERRED / DISCARD / ABSTAIN) into deployment-grade renderings with
continuous confidence (u_stored), semantic failure-mode tags, and
deployment-aware caveats.

The module is additive — importing it does not modify any pipeline
behavior. The thesis eval writers optionally call render() on each
sample to produce a `production_response` field alongside the existing
research fields, and Ship 2 (scripts/caem_chat.py, caem_demo_server.py)
wraps the live pipeline with the same rendering for interactive use.
"""

from caem.production.confidence import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    VALID_DECISIONS,
    ProductionResponse,
    render,
    render_dict,
    user_facing_answer,
)

__all__ = [
    "INSUFFICIENT_EVIDENCE_MESSAGE",
    "VALID_DECISIONS",
    "ProductionResponse",
    "render",
    "render_dict",
    "user_facing_answer",
]
