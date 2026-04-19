"""caem.ablation.variants -- canonical CAEM ablation variants.

Each variant is a named mutation applied to a fresh CAEMConfig plus
optional pipeline flags. The full-CAEM reference ("full") makes no
mutation and serves as the anchor against which every lesion is scored.

Design decisions
----------------
* Mutations are **pure functions** over CAEMConfig, not constructor
  arguments. This keeps the CAEMConfig dataclass unchanged and makes
  variants composable (e.g. a future "no_verifier + no_rag" combined
  lesion is just a tuple of the two mutation functions).
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
    """Full-CAEM reference. No mutation."""
    return None


def _mut_no_grounding(cfg: CAEMConfig) -> None:
    """Zero EXTERNAL-grounding weights in u_stored; renormalize the rest.

    Grounding in u_stored is carried by ``pground_mean`` and
    ``pground_atomic`` only: both measure whether retrieved Wikipedia
    passages entail the generated answer (passage -> answer entailment).

    IMPORTANT: ``u_stored_weight_nli`` is NOT a grounding signal. Per
    ``caem/config.py:192`` it weights ``p_entail`` (chain -> answer
    entailment), which is a self-consistency / internal-coherence
    signal — orthogonal to whether external passages support the claim.
    The prior implementation zeroed ``u_stored_weight_nli`` alongside
    the pground fields, which conflated two mechanism removals in one
    variant and made the ``no_grounding`` ablation measure grounding + NLI
    self-consistency jointly (not grounding alone). Audit MAJOR-VR1 /
    Task #114.

    Post-fix: this mutation zeroes ONLY the two pground weights and
    rescales the other four (nli, sc, uinternal, se) to sum to 1.0.
    The ``no_grounding`` variant now ablates external grounding cleanly.
    A companion ``no_chain_answer_entailment`` variant (zeroing nli
    only) is deferred to future work — adding it pushes the registry
    from 16 to 17 and would require thesis table updates in Ch5 (see
    "Sixteen named ablation variants" caption / §5.2).

    Early-returns when grounding weights are already zero so repeated
    application is bitwise-idempotent (avoiding FP drift in the rescale).
    """
    if (
        cfg.u_stored_weight_pground_mean == 0.0
        and cfg.u_stored_weight_pground_atomic == 0.0
    ):
        return
    nli = cfg.u_stored_weight_nli
    sc = cfg.u_stored_weight_sc
    ui = cfg.u_stored_weight_uinternal
    se = cfg.u_stored_weight_se
    total_keep = nli + sc + ui + se
    if total_keep <= 0:
        # Pathological config (all kept weights zero); fall back to
        # equal split across the four remaining signals.
        cfg.u_stored_weight_nli = 1 / 4
        cfg.u_stored_weight_sc = 1 / 4
        cfg.u_stored_weight_uinternal = 1 / 4
        cfg.u_stored_weight_se = 1 / 4
    else:
        cfg.u_stored_weight_nli = nli / total_keep
        cfg.u_stored_weight_sc = sc / total_keep
        cfg.u_stored_weight_uinternal = ui / total_keep
        cfg.u_stored_weight_se = se / total_keep
    cfg.u_stored_weight_pground_mean = 0.0
    cfg.u_stored_weight_pground_atomic = 0.0


def _mut_no_internal_calibration(cfg: CAEMConfig) -> None:
    """Zero u_internal weight; rescale the remaining u_stored signals."""
    ui = cfg.u_stored_weight_uinternal
    if ui <= 0:
        return
    cfg.u_stored_weight_uinternal = 0.0
    remaining = (
        cfg.u_stored_weight_pground_mean
        + cfg.u_stored_weight_pground_atomic
        + cfg.u_stored_weight_nli
        + cfg.u_stored_weight_sc
        + cfg.u_stored_weight_se
    )
    if remaining <= 0:
        return
    scale = 1.0 / remaining
    cfg.u_stored_weight_pground_mean *= scale
    cfg.u_stored_weight_pground_atomic *= scale
    cfg.u_stored_weight_nli *= scale
    cfg.u_stored_weight_sc *= scale
    cfg.u_stored_weight_se *= scale


def _mut_no_semantic_entropy(cfg: CAEMConfig) -> None:
    """Disable semantic entropy (Farquhar 2024) in u_stored; rescale."""
    se = cfg.u_stored_weight_se
    if se <= 0:
        return
    cfg.u_stored_weight_se = 0.0
    remaining = (
        cfg.u_stored_weight_pground_mean
        + cfg.u_stored_weight_pground_atomic
        + cfg.u_stored_weight_nli
        + cfg.u_stored_weight_sc
        + cfg.u_stored_weight_uinternal
    )
    if remaining <= 0:
        return
    scale = 1.0 / remaining
    cfg.u_stored_weight_pground_mean *= scale
    cfg.u_stored_weight_pground_atomic *= scale
    cfg.u_stored_weight_nli *= scale
    cfg.u_stored_weight_sc *= scale
    cfg.u_stored_weight_uinternal *= scale


def _mut_no_early_exit(cfg: CAEMConfig) -> None:
    """Defuse the confabulation gate by setting thresholds that never fire."""
    cfg.early_exit_u_internal = 1.01
    cfg.early_exit_p_ground_max = -0.01


def _mut_no_contradiction_veto(cfg: CAEMConfig) -> None:
    """Neutralise the contradiction veto by setting its threshold > 1.0."""
    cfg.contradiction_veto_threshold = 1.01


def _mut_no_tier1(cfg: CAEMConfig) -> None:
    """Route every query through Stage 4+ by making Tier-1 unreachable."""
    cfg.tier1_combined_threshold = 1.01


def _mut_no_store_gate(cfg: CAEMConfig) -> None:
    """Disable the storage gate: store every Tier-2/3 generation."""
    cfg.store_threshold = 0.0
    cfg.defer_threshold = 0.0
    cfg.abstain_pground_ceiling = -0.01


def _mut_aggressive_store(cfg: CAEMConfig) -> None:
    """Counterpart to no_store_gate: raise the bar much higher."""
    cfg.store_threshold = 0.85
    cfg.defer_threshold = 0.75


def _mut_no_novelty_filter(cfg: CAEMConfig) -> None:
    """Permit duplicates in episodic memory by disabling novelty threshold."""
    cfg.novelty_threshold = 1.01


def _mut_no_recency_decay(cfg: CAEMConfig) -> None:
    """Flatten recency decay so pruning becomes pure importance ranking."""
    cfg.recency_lambda = 0.0
    cfg.value_recency_weight = 0.0
    cfg.value_importance_weight = 1.0


def _mut_roberta_nli_backend(cfg: CAEMConfig) -> None:
    """Swap the verifier judge from MiniCheck back to legacy roberta-large-MNLI.

    The main CAEM run uses MiniCheck-Flan-T5-Large (Tang 2024 ACL), trained
    on LM-generated claim-support data. This ablation tests the counterfactual:
    does CAEM still meet the Ch4 purity premise (alpha > 1/2) and the Ch5
    acceptance criteria under a generic NLI model trained on human-written
    MultiNLI / SNLI sentence pairs?

    Literature prior: HaluEval 2025 / Semantic Illusion 2025 report ~100 percent
    false positive rate at 95 percent recall for DeBERTa-v3-large-MNLI on LM
    hallucinations. RoBERTa-large-MNLI is from the same training distribution
    and is expected to exhibit the same miscalibration. This variant produces
    per-cycle EM / EPI / CES trajectories under the weaker verifier so the
    swap can be defended with trajectory evidence in addition to the one-shot
    calibration diagnostic (scripts/calibration_minicheck_vs_roberta.py).
    """
    cfg.verifier_backend = "roberta_nli"


def _mut_equal_signal_weights(cfg: CAEMConfig) -> None:
    """Flatten the six composite u_stored signal weights (Ch3 Eq 3.5) to 1/6 each.

    The full configuration uses deliberately unequal weights --
        pground_mean=0.30, pground_atomic=0.15, nli=0.15,
        sc=0.15, uinternal=0.15, se=0.10
    -- reflecting our prior that grounding-based signals dominate reliability
    on open-domain QA. This ablation tests the counterfactual: does the
    designed weighting actually buy anything, or would a flat 1/6 composite
    achieve comparable calibration / VER / hallucination reduction? The six
    fields below mirror the composite sum in CAEMConfig exactly; every
    u_stored weight is clobbered, so no normalisation step is needed.
    """
    equal = 1.0 / 6.0
    cfg.u_stored_weight_pground_mean = equal
    cfg.u_stored_weight_pground_atomic = equal
    cfg.u_stored_weight_nli = equal
    cfg.u_stored_weight_sc = equal
    cfg.u_stored_weight_uinternal = equal
    cfg.u_stored_weight_se = equal


# =============================================================================
# Variant registry
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
    # -- Verification / gating mechanisms ------------------------------------
    AblationVariant(
        name="no_verifier",
        description="Bypass UnifiedVerifier; store every Tier-2/3 generation",
        mutation=_mut_noop,
        skip_verifier=True,
        mechanism_tag="verification",
        needs_cyclic_rerun=True,    # storage behaviour changes -> SIL pool diverges
    ),
    AblationVariant(
        name="no_grounding",
        description="Disable external Wikipedia grounding in u_stored",
        mutation=_mut_no_grounding,
        mechanism_tag="verification",
        needs_cyclic_rerun=True,    # u_stored distribution changes -> training picks differ
    ),
    AblationVariant(
        name="no_contradiction_veto",
        description="Neutralise p_contra veto in the STORE/DISCARD decision",
        mutation=_mut_no_contradiction_veto,
        mechanism_tag="verification",
        needs_cyclic_rerun=True,    # contradicting episodes now enter the pool
    ),
    AblationVariant(
        name="no_store_gate",
        description="Disable u_stored threshold; store every generation",
        mutation=_mut_no_store_gate,
        mechanism_tag="verification",
        needs_cyclic_rerun=True,    # massive storage-rate change; re-run required
    ),
    # -- Calibration mechanisms ----------------------------------------------
    AblationVariant(
        name="no_internal_calibration",
        description="Zero u_internal (token + dropout) weight in u_stored",
        mutation=_mut_no_internal_calibration,
        mechanism_tag="calibration",
        needs_cyclic_rerun=True,    # u_stored mix -> different stored set
    ),
    AblationVariant(
        name="no_semantic_entropy",
        description="Drop Farquhar-2024 semantic entropy from u_stored",
        mutation=_mut_no_semantic_entropy,
        mechanism_tag="calibration",
        needs_cyclic_rerun=True,
    ),
    AblationVariant(
        name="equal_signal_weights",
        description="Revert to pre-calibration equal weights (0.25 each)",
        mutation=_mut_equal_signal_weights,
        mechanism_tag="calibration",
        needs_cyclic_rerun=True,
    ),
    # -- Safety-gate mechanisms ----------------------------------------------
    AblationVariant(
        name="no_early_exit",
        description="Defuse the early-exit confabulation gate",
        mutation=_mut_no_early_exit,
        mechanism_tag="safety_gate",
        needs_cyclic_rerun=True,    # confabulations now reach the decision tree
    ),
    # -- Self-improvement mechanisms -----------------------------------------
    AblationVariant(
        name="no_self_improvement",
        description="Full pipeline, no fine-tuning: Cycle-0 eval only",
        mutation=_mut_noop,
        skip_self_improvement=True,
        requires_baseline_only=True,
        mechanism_tag="self_improvement",
        needs_cyclic_rerun=False,   # never fine-tunes -- cycle 0 suffices
    ),
    AblationVariant(
        name="no_retroverify",
        description="Skip retroactive re-verification after each cycle",
        mutation=_mut_noop,
        skip_retroverify=True,
        mechanism_tag="self_improvement",
        needs_cyclic_rerun=True,    # memory quality trajectory diverges over cycles
    ),
    # -- Retrieval / routing mechanisms --------------------------------------
    AblationVariant(
        name="no_tier1",
        description="Force every query through Stage-4+ (no Tier-1 shortcut)",
        mutation=_mut_no_tier1,
        skip_tier1=True,
        mechanism_tag="routing",
        needs_cyclic_rerun=True,    # Tier-2/3 answers replace Tier-1 hits in
                                    # the storage pool -> SIL training set
                                    # diverges. Reclassified Session 43 after
                                    # the counterfactual-confound audit.
    ),
    AblationVariant(
        name="no_tier3_rag",
        description="Disable Tier-3 Wikipedia RAG; Tier-3 falls back to zero-shot",
        mutation=_mut_noop,
        skip_tier3_rag=True,
        mechanism_tag="retrieval",
        needs_cyclic_rerun=True,    # Tier-3 answers feed the storage pool
    ),
    AblationVariant(
        name="no_novelty_filter",
        description="Allow duplicate episodes (no cosine-novelty threshold)",
        mutation=_mut_no_novelty_filter,
        mechanism_tag="retrieval",
        needs_cyclic_rerun=True,    # memory composition diverges
    ),
    AblationVariant(
        name="no_recency_decay",
        description="Pure importance-based pruning; zero recency weight",
        mutation=_mut_no_recency_decay,
        mechanism_tag="retrieval",
        needs_cyclic_rerun=True,    # pruning diverges per cycle
    ),
    # -- Sensitivity sweep companion -----------------------------------------
    AblationVariant(
        name="aggressive_store",
        description="Raise STORE threshold to 0.85 (CES-sensitivity sweep)",
        mutation=_mut_aggressive_store,
        mechanism_tag="verification",
        needs_cyclic_rerun=True,
    ),
    # -- Verifier backend ablation ------------------------------------------
    AblationVariant(
        name="roberta_nli_backend",
        description="Swap verifier backend MiniCheck -> legacy roberta-large-MNLI",
        mutation=_mut_roberta_nli_backend,
        mechanism_tag="verification",
        needs_cyclic_rerun=True,    # different alpha/p_ground distribution -> different stored set
    ),
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
