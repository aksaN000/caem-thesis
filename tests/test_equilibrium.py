"""
tests/test_equilibrium.py
=========================
Unit tests for ``caem.eval.equilibrium``: exponential-saturation fit,
c* prediction, and the triple-signal early-stop gate.

Tests cover:
- Happy-path fits on synthetic data that matches the model
- Correct handling of noise (R^2 < 1 but close)
- Error paths (too few points, failed fit, degenerate parameters)
- c* prediction math against analytic reference values
- Triple-gate firing logic at the cycle-number boundaries
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from caem.eval.equilibrium import (
    EarlyStopDecision,
    EquilibriumFit,
    fit_ces_saturation,
    predict_ceq_star,
    three_signal_gate,
)


# ============================================================================
# Exponential-saturation fit
# ============================================================================


class TestFitCESSaturation:
    def _synthetic_ces(self, c_inf=0.70, a=0.30, k=0.40, n=10, noise=0.0, seed=42):
        rng = np.random.default_rng(seed)
        x = np.arange(1, n + 1)
        y = c_inf - a * np.exp(-k * x)
        if noise > 0:
            y = y + rng.normal(0, noise, size=x.shape)
        return y.tolist()

    def test_clean_synthetic_recovers_params(self):
        """No-noise synthetic data must recover (C_inf, A, k) to <1e-4."""
        y = self._synthetic_ces(c_inf=0.70, a=0.30, k=0.40, n=10)
        fit = fit_ces_saturation(y)
        assert fit.c_inf == pytest.approx(0.70, abs=1e-4)
        assert fit.a == pytest.approx(0.30, abs=1e-4)
        assert fit.k == pytest.approx(0.40, abs=1e-4)
        assert fit.r2 > 0.999

    def test_mild_noise_still_fits_r2_high(self):
        """Small Gaussian noise keeps R^2 > 0.95 (Sun et al. report >0.9)."""
        y = self._synthetic_ces(c_inf=0.70, a=0.30, k=0.40, n=10, noise=0.01)
        fit = fit_ces_saturation(y)
        assert fit.r2 > 0.95

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3 cycles"):
            fit_ces_saturation([0.5, 0.6])

    def test_explicit_cycle_indices_are_honoured(self):
        """Caller can pass non-default cycle indices (e.g., skipping
        the c=0 baseline)."""
        y = self._synthetic_ces(c_inf=0.70, a=0.30, k=0.40, n=10)
        fit = fit_ces_saturation(y, cycle_indices=list(range(1, 11)))
        assert fit.c_inf == pytest.approx(0.70, abs=1e-4)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="cycle_indices length"):
            fit_ces_saturation([0.5, 0.6, 0.7], cycle_indices=[1, 2])

    def test_as_dict_is_json_serialisable(self):
        import json
        y = self._synthetic_ces(c_inf=0.70, a=0.30, k=0.40, n=6)
        fit = fit_ces_saturation(y)
        d = fit.as_dict()
        # Should round-trip through JSON without errors.
        s = json.dumps(d)
        back = json.loads(s)
        assert set(back.keys()) == {"c_inf", "a", "k", "r2", "cov"}
        assert isinstance(back["cov"], list)


# ============================================================================
# c* prediction
# ============================================================================


class TestPredictCeqStar:
    def test_analytic_value_matches_reference(self):
        """c* = (1/k) * ln(A / (eps * C_inf))."""
        fit = EquilibriumFit(
            c_inf=0.70, a=0.30, k=0.40, r2=1.0,
            cov=np.eye(3),
        )
        # Reference: ln(0.30 / (0.01 * 0.70)) / 0.40 = ln(42.857) / 0.40
        expected = math.log(0.30 / (0.01 * 0.70)) / 0.40
        c_star = predict_ceq_star(fit, eps=0.01)
        assert c_star == pytest.approx(expected, rel=1e-6)

    def test_non_positive_k_returns_none(self):
        """k <= 0 means the fit doesn't imply convergence."""
        fit = EquilibriumFit(c_inf=0.7, a=0.3, k=0.0, r2=1.0, cov=np.eye(3))
        assert predict_ceq_star(fit) is None
        fit.k = -0.1
        assert predict_ceq_star(fit) is None

    def test_non_positive_c_inf_returns_none(self):
        fit = EquilibriumFit(c_inf=0.0, a=0.3, k=0.4, r2=1.0, cov=np.eye(3))
        assert predict_ceq_star(fit) is None

    def test_non_positive_a_returns_none(self):
        """Noisy early-cycle fits sometimes produce A<=0; no meaningful c*."""
        fit = EquilibriumFit(c_inf=0.7, a=-0.01, k=0.4, r2=1.0, cov=np.eye(3))
        assert predict_ceq_star(fit) is None

    def test_already_converged_returns_cycle_one(self):
        """If residual gap at c=1 is already below eps*C_inf, c*=1."""
        # A=0.001, eps=0.01, C_inf=0.70 -> eps*C_inf = 0.007 > A -> ratio<1
        fit = EquilibriumFit(c_inf=0.70, a=0.001, k=0.4, r2=1.0, cov=np.eye(3))
        c_star = predict_ceq_star(fit, eps=0.01)
        assert c_star == pytest.approx(1.0)

    def test_tighter_eps_produces_larger_c_star(self):
        fit = EquilibriumFit(c_inf=0.70, a=0.30, k=0.40, r2=1.0, cov=np.eye(3))
        c_1pct = predict_ceq_star(fit, eps=0.01)
        c_tenth = predict_ceq_star(fit, eps=0.001)
        assert c_tenth > c_1pct


# ============================================================================
# Triple-signal early-stop gate
# ============================================================================


class TestThreeSignalGate:
    def test_all_signals_clean_no_stop(self):
        """Rising CES, healthy storage, safe MMLU -> no signals fire."""
        decision = three_signal_gate(
            ces_history=[0.50, 0.55, 0.60, 0.63, 0.65],
            storage_rates=[0.30, 0.25, 0.22, 0.20, 0.18],
            mmlu_retentions=[0.97, 0.96, 0.96, 0.95, 0.95],
        )
        assert decision.signals_fired == []
        assert decision.should_stop is False

    def test_only_ces_gradient_fires_no_stop(self):
        """One signal alone is below the 2-of-3 requirement."""
        decision = three_signal_gate(
            ces_history=[0.60, 0.630, 0.631, 0.6315, 0.6317],
            storage_rates=[0.30, 0.25, 0.22, 0.20, 0.18],
            mmlu_retentions=[0.97, 0.96, 0.96, 0.95, 0.95],
        )
        assert "ces_gradient" in decision.signals_fired
        assert len(decision.signals_fired) == 1
        assert decision.should_stop is False

    def test_ces_gradient_plus_storage_fires_stop(self):
        """2-of-3 fires the gate."""
        decision = three_signal_gate(
            ces_history=[0.60, 0.630, 0.631, 0.6315, 0.6317],
            storage_rates=[0.30, 0.25, 0.22, 0.10, 0.03],
            mmlu_retentions=[0.97, 0.96, 0.96, 0.95, 0.95],
        )
        assert "ces_gradient" in decision.signals_fired
        assert "storage_saturation" in decision.signals_fired
        assert decision.should_stop is True

    def test_all_three_signals_fire_stop(self):
        decision = three_signal_gate(
            ces_history=[0.60, 0.630, 0.631, 0.6315, 0.6317],
            storage_rates=[0.30, 0.25, 0.22, 0.10, 0.03],
            mmlu_retentions=[0.97, 0.96, 0.96, 0.92, 0.91],
        )
        assert len(decision.signals_fired) == 3
        assert decision.should_stop is True

    def test_storage_alone_does_not_stop(self):
        """Storage signal alone (below 2-of-3) shouldn't trigger stop."""
        decision = three_signal_gate(
            ces_history=[0.50, 0.55, 0.60, 0.63, 0.68],
            storage_rates=[0.30, 0.25, 0.22, 0.10, 0.03],
            mmlu_retentions=[0.97, 0.96, 0.96, 0.95, 0.95],
        )
        assert "storage_saturation" in decision.signals_fired
        assert decision.should_stop is False

    def test_mmlu_floor_needs_two_consecutive(self):
        """Single-cycle dip below floor doesn't fire signal 3."""
        decision = three_signal_gate(
            ces_history=[0.50, 0.55, 0.60, 0.63, 0.68],
            storage_rates=[0.30, 0.25, 0.22, 0.20, 0.18],
            mmlu_retentions=[0.97, 0.96, 0.96, 0.95, 0.91],
        )
        assert "mmlu_floor" not in decision.signals_fired
        # Two consecutive at floor -> fires.
        decision2 = three_signal_gate(
            ces_history=[0.50, 0.55, 0.60, 0.63, 0.68],
            storage_rates=[0.30, 0.25, 0.22, 0.20, 0.18],
            mmlu_retentions=[0.97, 0.96, 0.96, 0.92, 0.91],
        )
        assert "mmlu_floor" in decision2.signals_fired

    def test_ces_gradient_needs_two_consecutive(self):
        """Single flat cycle doesn't fire signal 1."""
        decision = three_signal_gate(
            ces_history=[0.50, 0.60, 0.63, 0.6315],
            storage_rates=[0.30, 0.25, 0.22, 0.20],
            mmlu_retentions=[0.97, 0.96, 0.96, 0.95],
        )
        assert "ces_gradient" not in decision.signals_fired

    def test_burn_in_too_short_no_signals_fire(self):
        """With only 2 CES values, signal 1 can't evaluate (needs 3)."""
        decision = three_signal_gate(
            ces_history=[0.50, 0.55],
            storage_rates=[0.30],
            mmlu_retentions=[0.95],
        )
        assert decision.signals_fired == []
        assert decision.should_stop is False

    def test_custom_signals_required_threshold(self):
        """1-of-3 threshold fires on any single signal."""
        decision = three_signal_gate(
            ces_history=[0.50, 0.55, 0.60, 0.63, 0.68],
            storage_rates=[0.30, 0.25, 0.22, 0.10, 0.03],
            mmlu_retentions=[0.97, 0.96, 0.96, 0.95, 0.95],
            signals_required=1,
        )
        assert decision.should_stop is True

    def test_cycle_field_recorded(self):
        decision = three_signal_gate(
            ces_history=[0.50, 0.55, 0.60],
            storage_rates=[0.30, 0.25, 0.22],
            mmlu_retentions=[0.97, 0.96, 0.96],
            cycle=7,
        )
        assert decision.cycle == 7

    def test_as_dict_is_json_serialisable(self):
        import json
        decision = three_signal_gate(
            ces_history=[0.50, 0.55, 0.60, 0.63, 0.68],
            storage_rates=[0.30, 0.25, 0.22, 0.10, 0.03],
            mmlu_retentions=[0.97, 0.96, 0.96, 0.92, 0.91],
            cycle=5,
        )
        d = decision.as_dict()
        s = json.dumps(d)
        back = json.loads(s)
        assert back["cycle"] == 5
        assert back["should_stop"] is True
        assert set(back["signals_fired"]) <= {
            "ces_gradient", "storage_saturation", "mmlu_floor",
        }
