"""caem.ablation -- CAEM ablation framework (Phase 9, Session 43).

The ablation framework answers: "Which CAEM components actually matter?"
It sweeps a registry of canonical variants, each variant a targeted lesion
of one mechanism, and ranks them by CES (CAEM Efficacy Score) -- the
geometric mean of five axes {ACC, EPI, RET, CAL, VER} defined in
``eval.metrics.ces_score`` and aligned with ``eval.reporting.build_table_cycle``.

Module layout
-------------
  variants.py  -- AblationVariant dataclass + canonical VARIANT_REGISTRY
  scoring.py   -- CES-axis composition helpers (ACC / EPI / RET / CAL / VER)
  runner.py    -- single-variant execution: build mutated pipeline, run
                  harness, compute CES, return AblationResult

Entry point
-----------
  scripts/run_ablation.py  -- sweep variants, write per-variant JSON +
                              a cross-variant CES ranking CSV

Design invariants
-----------------
1. The full-CAEM reference variant MUST reproduce Table 5.1 cycle-level
   CES to within floating-point noise (same axis definitions, same
   denominators, same clamp).
2. Every variant is a pure mutation of CAEMConfig and/or pipeline flags
   -- no new hyperparameters are introduced in the ablation code path.
3. MMLU retention for RET uses the same fixed 200-sample split as
   ``SelfImprovementLoop._mmlu_score(n=200)``; see that method's
   docstring for the rationale.
4. Variants that disable fine-tuning still measure RET against the
   pristine-model MMLU baseline (ratio == 1.0 by construction, since
   the model is unchanged). This is the correct CES semantic: a
   variant that does no training cannot regress or improve retention.
"""

from caem.ablation.scoring import (
    CESAxes,
    ces_axes_from_cycle,
    ces_from_axes,
)
from caem.ablation.variants import (
    AblationVariant,
    VARIANT_REGISTRY,
    get_variant,
    list_variants,
)

__all__ = [
    "AblationVariant",
    "VARIANT_REGISTRY",
    "get_variant",
    "list_variants",
    "CESAxes",
    "ces_axes_from_cycle",
    "ces_from_axes",
]
