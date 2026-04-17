"""caem.confidence -- Confidence estimation modules.

Note (Session 42, 2026-04-17): PostGenerationConfidenceEstimator was removed.
Its four signals (u_token, u_dropout, u_consistency, u_entropy) have been
absorbed into the UnifiedVerifier (caem/verification/verifier.py). u_token and
u_dropout remain cheap pre-generation/post-generation internal signals used as
the early-exit confabulation check (u_internal >= 0.70 AND p_ground <= 0.20).
u_consistency and u_entropy are algorithmically identical to the verifier's
s_avg and h_norm and were already computed there, so they are no longer
duplicated. Pre-routing confidence (Stage 2) is unchanged.
"""

from caem.confidence.pre_routing import PreRoutingConfidenceEstimator

__all__ = ["PreRoutingConfidenceEstimator"]
