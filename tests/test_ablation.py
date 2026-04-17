"""
tests/test_ablation.py
=======================
Unit tests for the Phase-9 ablation framework (``caem.ablation.*``).

Covers:
  - variants registry: enumeration, lookup, filter helpers
  - mutation idempotence (bitwise-equal after double application)
  - u_stored weight invariants (sums to 1.0 after rescaling variants)
  - full-CAEM variant is the identity mutation
  - CESAxes scoring: axis ranges, NaN clamp, requires_baseline_only RET==1.0
  - aggregate_axes: weighted means, preserved sample counts

These tests avoid importing torch / faiss / transformers, so they can
run in a CPU-only environment identical to the sandbox.
"""

from __future__ import annotations

import math

import pytest

from caem.ablation.variants import (
    VARIANT_REGISTRY,
    AblationVariant,
    get_variant,
    list_variants,
)
from caem.ablation.scoring import (
    CES_EPS,
    CESAxes,
    aggregate_axes,
    ces_axes_from_cycle,
    ces_from_axes,
)
from caem.config import CAEMConfig


# =============================================================================
# VARIANT REGISTRY
# =============================================================================

class TestVariantRegistry:
    def test_registry_is_non_empty(self):
        assert len(VARIANT_REGISTRY) > 0

    def test_full_is_first(self):
        """The anchor row must be the first registered variant."""
        assert next(iter(VARIANT_REGISTRY)) == "full"

    def test_get_variant_known(self):
        v = get_variant("no_verifier")
        assert isinstance(v, AblationVariant)
        assert v.name == "no_verifier"

    def test_get_variant_unknown_raises_with_suggestions(self):
        with pytest.raises(KeyError) as exc_info:
            get_variant("nope_not_real")
        msg = str(exc_info.value)
        assert "Registered variants" in msg
        # The error should list every registered name so the user can recover
        for name in VARIANT_REGISTRY:
            assert name in msg

    def test_all_names_are_unique(self):
        names = [v.name for v in VARIANT_REGISTRY.values()]
        assert len(names) == len(set(names))

    def test_all_variants_have_description(self):
        for v in VARIANT_REGISTRY.values():
            assert v.description
            assert isinstance(v.description, str)

    def test_mechanism_tags_are_bounded(self):
        """Only the approved mechanism tags may appear."""
        approved = {
            "reference", "verification", "calibration",
            "self_improvement", "retrieval", "routing", "safety_gate",
        }
        for v in VARIANT_REGISTRY.values():
            assert v.mechanism_tag in approved, (
                f"{v.name} has unknown mechanism_tag={v.mechanism_tag}"
            )


# =============================================================================
# MUTATION BEHAVIOUR
# =============================================================================

class TestMutationBehaviour:
    def test_full_is_identity(self):
        """The 'full' variant must not mutate CAEMConfig at all."""
        base = CAEMConfig()
        mutated = get_variant("full").apply()
        assert vars(base) == vars(mutated)

    @pytest.mark.parametrize("name", list(VARIANT_REGISTRY))
    def test_all_variants_idempotent(self, name):
        """Applying a mutation twice must produce bitwise-equal state."""
        variant = get_variant(name)
        cfg = variant.apply()
        snap = dict(vars(cfg))
        variant.mutation(cfg)
        after = dict(vars(cfg))
        assert snap == after, f"{name} is not bitwise-idempotent"

    def test_no_verifier_does_not_change_config(self):
        """skip_verifier is a pipeline-level flag; config should be unchanged."""
        base = CAEMConfig()
        mutated = get_variant("no_verifier").apply()
        assert vars(base) == vars(mutated)

    def test_no_self_improvement_does_not_change_config(self):
        """skip_self_improvement is also a pipeline-level flag."""
        base = CAEMConfig()
        mutated = get_variant("no_self_improvement").apply()
        assert vars(base) == vars(mutated)


class TestUStoredWeightRescaling:
    """Variants that zero u_stored signals must leave the composite on [0,1]."""

    @pytest.mark.parametrize("name", [
        "no_grounding", "no_internal_calibration", "no_semantic_entropy",
    ])
    def test_weights_sum_to_one(self, name):
        cfg = get_variant(name).apply()
        total = (
            cfg.u_stored_weight_pground_mean
            + cfg.u_stored_weight_pground_atomic
            + cfg.u_stored_weight_nli
            + cfg.u_stored_weight_sc
            + cfg.u_stored_weight_uinternal
            + cfg.u_stored_weight_se
        )
        assert math.isclose(total, 1.0, abs_tol=1e-9), (
            f"{name}: weights sum to {total}, expected 1.0"
        )

    def test_no_grounding_zeros_pground_and_nli(self):
        cfg = get_variant("no_grounding").apply()
        assert cfg.u_stored_weight_pground_mean == 0.0
        assert cfg.u_stored_weight_pground_atomic == 0.0
        assert cfg.u_stored_weight_nli == 0.0

    def test_no_internal_calibration_zeros_uinternal(self):
        cfg = get_variant("no_internal_calibration").apply()
        assert cfg.u_stored_weight_uinternal == 0.0

    def test_no_semantic_entropy_zeros_se(self):
        cfg = get_variant("no_semantic_entropy").apply()
        assert cfg.u_stored_weight_se == 0.0


class TestThresholdMutations:
    def test_no_early_exit_disables_gate(self):
        cfg = get_variant("no_early_exit").apply()
        # Gate: u_internal >= floor AND p_ground_max <= ceiling.
        # With floor > 1 and ceiling < 0, the AND is never satisfiable.
        assert cfg.early_exit_u_internal > 1.0
        assert cfg.early_exit_p_ground_max < 0.0

    def test_no_contradiction_veto_raises_threshold(self):
        cfg = get_variant("no_contradiction_veto").apply()
        assert cfg.contradiction_veto_threshold > 1.0

    def test_no_tier1_raises_combined_threshold(self):
        cfg = get_variant("no_tier1").apply()
        assert cfg.tier1_combined_threshold > 1.0

    def test_no_store_gate_zeroes_thresholds(self):
        cfg = get_variant("no_store_gate").apply()
        assert cfg.store_threshold == 0.0
        assert cfg.defer_threshold == 0.0

    def test_aggressive_store_raises_thresholds(self):
        cfg = get_variant("aggressive_store").apply()
        base = CAEMConfig()
        assert cfg.store_threshold > base.store_threshold
        assert cfg.defer_threshold > base.defer_threshold


class TestVariantListFilters:
    def test_list_all_returns_full_registry(self):
        assert len(list_variants()) == len(VARIANT_REGISTRY)

    def test_filter_by_mechanism(self):
        verification = list_variants(mechanism="verification")
        assert len(verification) > 0
        for v in verification:
            assert v.mechanism_tag == "verification"

    def test_filter_by_mechanism_unknown_returns_empty(self):
        assert list_variants(mechanism="nonexistent") == []

    def test_cyclic_only_filter(self):
        cyclic = list_variants(cyclic_only=True)
        assert all(v.needs_cyclic_rerun for v in cyclic)

    def test_inference_only_filter(self):
        inference = list_variants(inference_only=True)
        assert all(not v.needs_cyclic_rerun for v in inference)

    def test_cyclic_and_inference_are_disjoint_and_complete(self):
        cyclic = set(v.name for v in list_variants(cyclic_only=True))
        inference = set(v.name for v in list_variants(inference_only=True))
        assert cyclic.isdisjoint(inference)
        assert cyclic | inference == set(VARIANT_REGISTRY)

    def test_mutually_exclusive_flags_raise(self):
        with pytest.raises(ValueError):
            list_variants(cyclic_only=True, inference_only=True)

    def test_full_is_inference_only(self):
        """full-CAEM reuses cycle-N weights; no rerun needed by construction."""
        assert not get_variant("full").needs_cyclic_rerun

    def test_no_self_improvement_is_inference_only(self):
        """By construction: never trains, so no cyclic loop to re-run."""
        assert not get_variant("no_self_improvement").needs_cyclic_rerun
        assert get_variant("no_self_improvement").requires_baseline_only


# =============================================================================
# CES AXIS SCORING
# =============================================================================

class TestCesFromAxes:
    def test_identity_all_ones(self):
        """All axes at 1.0 -> CES == 1.0."""
        assert math.isclose(
            ces_from_axes(1.0, 1.0, 1.0, 1.0, 1.0), 1.0, abs_tol=1e-9
        )

    def test_all_eps_gives_eps(self):
        """All axes clamped to eps -> CES == eps."""
        assert math.isclose(
            ces_from_axes(CES_EPS, CES_EPS, CES_EPS, CES_EPS, CES_EPS),
            CES_EPS, abs_tol=1e-9,
        )

    def test_nan_is_clamped_to_eps(self):
        """NaN axes become eps in the product."""
        nan = float("nan")
        a = ces_from_axes(nan, 1.0, 1.0, 1.0, 1.0)
        # CES = (eps * 1 * 1 * 1 * 1) ^ (1/5) = eps^(1/5)
        assert math.isclose(a, CES_EPS ** (1/5), rel_tol=1e-9)

    def test_geometric_mean_property(self):
        """A single weak axis drags CES down proportionally."""
        strong = ces_from_axes(0.9, 0.9, 0.9, 0.9, 0.9)
        # Drop one axis to 0.5; geometric mean should drop too
        weaker = ces_from_axes(0.9, 0.9, 0.9, 0.9, 0.5)
        assert weaker < strong


class TestCesAxesFromCycle:
    def test_basic_pipeline(self):
        em = [1.0, 1.0, 0.0, 1.0, 0.0]
        u = [0.9, 0.8, 0.2, 0.7, 0.3]
        axes = ces_axes_from_cycle(em, u, cycle_mmlu=0.45, baseline_mmlu=0.50)
        assert isinstance(axes, CESAxes)
        assert 0.0 <= axes.acc <= 1.0
        assert 0.0 <= axes.epi <= 1.0
        assert 0.0 <= axes.ret <= 1.0
        assert 0.0 <= axes.cal <= 1.0
        assert axes.ver == 0.5  # placeholder default
        assert axes.n_samples == 5
        assert axes.n_scored_confidence == 5

    def test_acc_is_mean_em(self):
        em = [1.0, 1.0, 0.0, 1.0]
        u = [0.9, 0.8, 0.5, 0.7]
        axes = ces_axes_from_cycle(em, u)
        assert math.isclose(axes.acc, 0.75, abs_tol=1e-9)

    def test_ret_none_gives_nan(self):
        axes = ces_axes_from_cycle([1.0], [0.9])
        assert math.isnan(axes.ret)

    def test_requires_baseline_only_forces_ret_one(self):
        """Variants that don't fine-tune must have RET==1.0 by construction."""
        axes = ces_axes_from_cycle(
            [1.0, 0.0], [0.9, 0.4],
            cycle_mmlu=None, baseline_mmlu=None,
            requires_baseline_only=True,
        )
        assert axes.ret == 1.0

    def test_ret_clipped_to_one(self):
        """If cycle_mmlu improves beyond baseline, ratio must clip to 1.0."""
        axes = ces_axes_from_cycle(
            [1.0], [0.9],
            cycle_mmlu=0.60, baseline_mmlu=0.50,
        )
        assert axes.ret == 1.0

    def test_ver_override(self):
        axes = ces_axes_from_cycle(
            [1.0, 0.0], [0.8, 0.3],
            verifier_balanced_accuracy=0.88,
        )
        assert axes.ver == 0.88

    def test_empty_inputs_yield_nan_axes(self):
        axes = ces_axes_from_cycle([], [])
        assert math.isnan(axes.acc)
        assert axes.n_samples == 0

    def test_null_u_stored_filtered_from_cal(self):
        """Tier-1 passthroughs with u_stored=None should not enter CAL bins."""
        em = [1.0, 0.0, 1.0, 0.0]
        u = [0.9, None, 0.85, 0.20]
        axes = ces_axes_from_cycle(em, u)
        # 3 samples have valid confidence; 4 total for ACC
        assert axes.n_samples == 4
        assert axes.n_scored_confidence == 3


class TestAggregateAxes:
    def test_empty_aggregate(self):
        agg = aggregate_axes([])
        assert agg.n_samples == 0
        # All-eps CES for an empty aggregate (not zero, so we can spot it)
        assert math.isclose(agg.ces, CES_EPS, abs_tol=1e-9)

    def test_sample_weighting(self):
        """Two benchmarks with different sample counts weight by n_samples."""
        bm1 = ces_axes_from_cycle(
            [1.0] * 10 + [0.0] * 10,              # ACC = 0.5
            [0.8] * 20,
            cycle_mmlu=0.5, baseline_mmlu=0.5,
        )
        bm2 = ces_axes_from_cycle(
            [1.0] * 80 + [0.0] * 20,              # ACC = 0.8
            [0.9] * 100,
            cycle_mmlu=0.5, baseline_mmlu=0.5,
        )
        agg = aggregate_axes([bm1, bm2])
        expected_acc = (0.5 * 20 + 0.8 * 100) / 120
        assert math.isclose(agg.acc, expected_acc, abs_tol=1e-9)
        assert agg.n_samples == 120

    def test_ces_recomputed_not_averaged(self):
        """Aggregate CES must be geometric mean of aggregated axes, not
        arithmetic mean of per-BM CES values."""
        bm1 = ces_axes_from_cycle(
            [1.0, 0.0, 1.0], [0.9, 0.1, 0.8],
            cycle_mmlu=0.5, baseline_mmlu=0.5,
        )
        bm2 = ces_axes_from_cycle(
            [0.0, 0.0, 1.0], [0.2, 0.3, 0.85],
            cycle_mmlu=0.5, baseline_mmlu=0.5,
        )
        agg = aggregate_axes([bm1, bm2])
        # Aggregate CES should be feedable back into ces_from_axes and match
        reconstructed = ces_from_axes(
            agg.acc, agg.epi, agg.ret, agg.cal, agg.ver,
        )
        assert math.isclose(agg.ces, reconstructed, abs_tol=1e-9)

    def test_baseline_only_ret_preserved_in_aggregate(self):
        bm1 = ces_axes_from_cycle(
            [1.0, 0.0], [0.9, 0.2], requires_baseline_only=True,
        )
        bm2 = ces_axes_from_cycle(
            [1.0], [0.8], requires_baseline_only=True,
        )
        agg = aggregate_axes([bm1, bm2])
        assert math.isclose(agg.ret, 1.0, abs_tol=1e-9)
