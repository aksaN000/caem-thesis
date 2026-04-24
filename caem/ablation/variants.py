"""caem.ablation.variants -- canonical CAEM ablation variants.

Each variant is a named mutation applied to a fresh CAEMConfig plus
optional pipeline flags. The full-CAEM reference ("full") makes no
mutation and serves as the anchor against which every lesion is scored.

Phase 1 Full scope (2026-04-22 decision, revised claim-vs-metric audit)
-----------------------------------------------------------------------
Registry is hand-picked at 4 variants (full + 3 ablations); the wrapper
runs the 3 ablations only (Phase 1a Step 7 main is the `full` anchor).
Each ablation is tied to a specific numbered thesis Claim that cannot
be answered by a directly-measured metric:

  Claim 1 (memory routing)        -> [metric] tier_{1,2,3}_frac per cycle
  Claim 2 (purity, α > ½)         -> [metric] Step 19 purity_validation
                                     + no_retroverify (time dimension)
  Claim 3 (SIL improvement)       -> no_self_improvement  +  no_retroverify
  Claim 4 (no catastrophic CF)    -> no_forgetting_guard
  Claim 5 (modularity)            -> [code] NLIJudgeInterface protocol
                                     + [metric] Step 5.5.2 v5 diagnostic
  Ch3 Eq 3.5 signal weighting     -> [metric] Step 19.5 correlation matrix
                                     + methods-section literature priors
  Branch C Goal 2 contribution    -> [metric] Step 19.5 correlation matrix
                                     + methods-section sample-② narrative

Earlier wider sweep candidates (17 screening variants + 5 initial
load-bearing picks) were pruned against two tests: (a) is the claim
defended by a direct metric? (b) is the evidence redundant with another
kept ablation? Removed on 2026-04-22 per user audit:

  no_store_gate           -> Step 19 purity_validation measures α directly
  no_tier1                -> tier1_frac per-cycle trajectory is measured
  roberta_nli_backend     -> Step 5.5.2 v5 gives AUROC/ECE on 500 pairs
  equal_signal_weights    -> Step 19.5 correlation matrix shows non-redundancy;
                             specific weights defended by methods-section priors
  no_q_a_relevance        -> same symmetry as other 6 signals: Step 19.5
                             correlation matrix shows q_a_relevance carries
                             unique info; Goal 2 is a design choice (like
                             "we use BGE reranker"), not a numbered Claim
  no_verifier, no_grounding, no_internal_calibration,
  no_semantic_entropy, no_early_exit, no_tier3_rag,
  no_novelty_filter, no_recency_decay, no_memory_consolidation,
  aggressive_store, no_u_dropout, no_m_chain,
  no_cross_encoder_rerank, tau_store_0_50
                          -> low load-bearing or covered by the
                             9-signal correlation matrix (Step 19.5)
                             + methods section wording.

See branch_C_log.md for the pruning rationale.

Design decisions
----------------
* Mutations are **pure functions** over CAEMConfig, not constructor
  arguments. This keeps the CAEMConfig dataclass unchanged and makes
  variants composable (e.g. a future "no_store_gate + no_retroverify"
  combined lesion is just a tuple of the two mutation functions).
* Pipeline-level flags (e.g. ``skip_self_improvement``) are carried on
  the AblationVariant itself. The runner consumes them when driving
  the experiment.
* ``requires_baseline_only`` flags variants that do not change model
  weights; the runner skips fine-tuning for these and uses the
  pristine-model MMLU baseline as RET.

Adding a new variant
--------------------
1. Define a ``_mut_<name>(cfg) -> None`` function that patches cfg in place.
2. Append an ``AblationVariant`` entry to ``VARIANT_REGISTRY``.
3. Add a parse test to ``tests/test_ablation.py`` asserting the mutation
   is applied (idempotent on double-application) and that the variant
   is retrievable by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from caem.config import CAEMConfig

# -----------------------------------------------------------------------------
# Mutation type
# -----------------------------------------------------------------------------
ConfigMutation = Callable[[CAEMConfig], None]


@dataclass(frozen=True)
class AblationVariant:
    """One ablation variant: name, description, config mutation, pipeline flags."""

    name: str
    description: str
    mutation: ConfigMutation
    skip_self_improvement: bool = False
    skip_retroverify: bool = False
    skip_verifier: bool = False
    skip_tier1: bool = False
    skip_tier3_rag: bool = False
    requires_baseline_only: bool = False
    mechanism_tag: str = "reference"
    # Runbook guidance: does this variant need the SIL cycle loop re-run
    # under its own config (True), or can it reuse full-CAEM's cycle-N
    # weights + memory and only flip behaviour at inference (False)?
    needs_cyclic_rerun: bool = False

    def apply(self, cfg: Optional[CAEMConfig] = None) -> CAEMConfig:
        """Return a fresh CAEMConfig with this variant's mutation applied."""
        target = CAEMConfig() if cfg is None else cfg
        self.mutation(target)
        return target


# =============================================================================
# Mutation functions -- each patches CAEMConfig in place
# =============================================================================

def _mut_noop(cfg: CAEMConfig) -> None:
    """Full-CAEM reference. Also used by pipeline-flag-only variants
    (no_retroverify, no_self_improvement) whose effect is entirely driven
    by the AblationVariant's skip_* flag, not a config mutation."""
    return None


def _mut_no_forgetting_guard(cfg: CAEMConfig) -> None:
    """Disable the MMLU-retention rollback that aborts a cycle if
    post/pre drops below forgetting_tolerance.

    Claim 4 defense: setting tolerance to 0.0 makes every finite retention
    ratio pass the guard check. The guard snapshot (theta_prev) still
    fires so compute cost is identical; only the rollback branch is
    disarmed. If MMLU craters under this variant, the rollback is what's
    preventing catastrophic forgetting.
    """
    cfg.forgetting_tolerance = 0.0


# =============================================================================
# Variant registry — Phase 1 Full: full + 3 hand-picked ablations
# (2026-04-22 decision; each defending a specific numbered thesis Claim)
# =============================================================================

# Each entry is an AblationVariant. "full" MUST come first so it's the
# reference row in the Chapter-5 ablation table.
#
# needs_cyclic_rerun semantics:
#   True   -- the variant changes what gets stored / what is trained on,
#             so SIL must be re-run under the variant's config (separate
#             memory store + separate fine-tuned weights per variant).
#   False  -- the variant changes only inference-time behaviour; it can
#             reuse full-CAEM's cycle-N memory + weights and just flip
#             the relevant knob at eval time.
_ALL_VARIANTS: Tuple[AblationVariant, ...] = (
    AblationVariant(
        name="full",
        description="Full CAEM (reference anchor for all ablation deltas)",
        mutation=_mut_noop,
        mechanism_tag="reference",
        needs_cyclic_rerun=False,
    ),
    # -- Claim 2 (purity) + Claim 3 (across-cycle) memory hygiene -----------
    # Direct metric for Claim 2 α>½: outputs/purity_validation/theory_validation.json
    # (Step 19). no_store_gate removed 2026-04-22 — was redundant with that metric.
    AblationVariant(
        name="no_retroverify",
        description="Skip retroactive re-verification after each cycle (Claim 2 + 3 defense)",
        mutation=_mut_noop,
        skip_retroverify=True,
        mechanism_tag="self_improvement",
        needs_cyclic_rerun=True,    # memory quality trajectory diverges over cycles
    ),
    # -- Claim 3 (SIL improvement) upper-bound ------------------------------
    AblationVariant(
        name="no_self_improvement",
        description="Full pipeline, no fine-tuning: Cycle-0 eval only (Claim 3 defense)",
        mutation=_mut_noop,
        skip_self_improvement=True,
        requires_baseline_only=True,
        mechanism_tag="self_improvement",
        needs_cyclic_rerun=False,   # never fine-tunes -- cycle 0 suffices
    ),
    # -- Claim 4 (no catastrophic forgetting) defense -----------------------
    AblationVariant(
        name="no_forgetting_guard",
        description="Disable MMLU rollback; forgetting_tolerance=0.0 (Claim 4 defense)",
        mutation=_mut_no_forgetting_guard,
        mechanism_tag="safety_gate",
        needs_cyclic_rerun=True,    # removing the guard changes which cycles commit -> divergent weights
    ),
    # Claim 1 (memory routing) is defended by the tier_{1,2,3}_frac metric
    # written to experiment_summary.csv and per-cycle JSONs each cycle;
    # no_tier1 ablation removed 2026-04-22 as redundant with that trajectory.
    # Claim 5 (modularity) is defended by the NLIJudgeInterface protocol +
    # Step 5.5.2 v5 MiniCheck-vs-RoBERTa-vs-Qwen AUROC diagnostic;
    # roberta_nli_backend removed 2026-04-22 as redundant with that diagnostic.

    # 9-signal weighting (Ch3 Eq 3.5) is defended by the Step 19.5 correlation
    # matrix (shows all 7 signals are non-redundant) + methods-section text
    # citing per-signal literature priors. equal_signal_weights and
    # no_q_a_relevance ablations both removed 2026-04-22: the correlation
    # matrix provides the same evidence symmetrically for every signal
    # (including q_a_relevance). Branch C Goal 2 is a design-level
    # contribution defended in methods, not a numbered thesis Claim that
    # requires per-signal ablation.
)


VARIANT_REGISTRY: Dict[str, AblationVariant] = {v.name: v for v in _ALL_VARIANTS}


def list_variants(
    mechanism: Optional[str] = None,
    cyclic_only: bool = False,
    inference_only: bool = False,
) -> List[AblationVariant]:
    """Return variants in registry order.

    Filters:
      mechanism        -- restrict to one mechanism_tag
      cyclic_only      -- only variants with needs_cyclic_rerun=True
      inference_only   -- only variants with needs_cyclic_rerun=False

    cyclic_only and inference_only are mutually exclusive.
    """
    if cyclic_only and inference_only:
        raise ValueError("cyclic_only and inference_only are mutually exclusive")
    out = list(_ALL_VARIANTS)
    if mechanism is not None:
        out = [v for v in out if v.mechanism_tag == mechanism]
    if cyclic_only:
        out = [v for v in out if v.needs_cyclic_rerun]
    if inference_only:
        out = [v for v in out if not v.needs_cyclic_rerun]
    return out


def get_variant(name: str) -> AblationVariant:
    """Look up a variant by name. Raises KeyError with a helpful message."""
    if name not in VARIANT_REGISTRY:
        known = ", ".join(sorted(VARIANT_REGISTRY))
        raise KeyError(
            f"Unknown ablation variant '{name}'. Registered variants: {known}"
        )
    return VARIANT_REGISTRY[name]


__all__ = [
    "AblationVariant",
    "ConfigMutation",
    "VARIANT_REGISTRY",
    "get_variant",
    "list_variants",
]
