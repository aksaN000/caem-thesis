"""
caem/eval/equilibrium.py
========================
Equilibrium prediction and triple-signal early-stop gate for CAEM's
self-improvement loop.

Three public helpers called by ``scripts/run_experiment.py`` at each
cycle boundary:

1. :func:`fit_ces_saturation` — fits
   ``CES(c) = C_inf - A * exp(-k * c)`` to the observed trajectory
   using ``scipy.optimize.curve_fit`` (non-linear least squares).
2. :func:`predict_ceq_star` — analytic ``c*`` where the residual gap
   ``A * exp(-k * c)`` drops below ``eps * C_inf``, so future cycles
   are predicted to gain at most ``eps`` fraction of the asymptote.
3. :func:`three_signal_gate` — 2-of-3 early-stop decision combining
   the parametric CES gradient with two CAEM-native non-parametric
   signals (memory storage saturation, MMLU retention ceiling).

References
----------
Sun, Liang, Zhang, Liu, Teng (ICLR 2026). "Theoretical Modeling of
LLM Self-Improvement Training Dynamics Through Solver-Verifier Gap."
Empirically validated the exponential-saturation functional form at
R^2 > 0.9 on Phi-3 / Phi-3.5 / Phi-4-mini / Llama-3.2-3B across
GSM8k, MATH, ProntoQA, MBPP. This module adopts that functional form
as a conditional assumption (CAEM's Ch4 Corollary C4) and fits the
three parameters on CAEM's CES trajectory.

The parametric prediction is cross-checked against two non-parametric
signals to avoid over-reliance on the functional-form assumption; the
gate fires only when at least two of the three signals agree.

Thesis placement: Ch4 §Theoretical Analysis Corollary C4 cites this
module; Ch5 §Implementation Details "Performance envelope" references
the triple-gate protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit


__all__ = [
    "EquilibriumFit",
    "EarlyStopDecision",
    "fit_ces_saturation",
    "predict_ceq_star",
    "three_signal_gate",
]


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class EquilibriumFit:
    """Result of fitting CES(c) = C_inf - A * exp(-k * c)."""

    c_inf: float
    a: float
    k: float
    r2: float
    cov: np.ndarray  # 3x3 parameter covariance from curve_fit

    def as_dict(self) -> dict:
        """JSON-serialisable form for per-cycle audit artefacts."""
        return {
            "c_inf": float(self.c_inf),
            "a": float(self.a),
            "k": float(self.k),
            "r2": float(self.r2),
            "cov": self.cov.tolist(),
        }


@dataclass
class EarlyStopDecision:
    """2-of-3 triple-signal early-stop decision at a cycle boundary."""

    signals_fired: List[str] = field(default_factory=list)
    should_stop: bool = False
    predicted_c_star: Optional[float] = None
    cycle: Optional[int] = None

    def as_dict(self) -> dict:
        """JSON-serialisable form for per-cycle audit artefacts."""
        return {
            "signals_fired": list(self.signals_fired),
            "should_stop": bool(self.should_stop),
            "predicted_c_star": (
                float(self.predicted_c_star)
                if self.predicted_c_star is not None else None
            ),
            "cycle": int(self.cycle) if self.cycle is not None else None,
        }


# ============================================================================
# Exponential-saturation fit
# ============================================================================


def _saturation_model(c, c_inf, a, k):
    """CES(c) = C_inf - A * exp(-k * c). Separate from the public API so
    ``curve_fit`` can reference it directly."""
    return c_inf - a * np.exp(-k * c)


def fit_ces_saturation(
    ces_history: Sequence[float],
    cycle_indices: Optional[Sequence[int]] = None,
) -> EquilibriumFit:
    """Fit CES(c) = C_inf - A * exp(-k * c) via non-linear least squares.

    Parameters
    ----------
    ces_history : sequence of float
        Observed CES at each cycle, length N.
    cycle_indices : sequence of int, optional
        Cycle numbers corresponding to each CES value. Defaults to
        ``range(1, N+1)`` so the first observation sits at c=1.

    Returns
    -------
    :class:`EquilibriumFit`
        Fitted ``(C_inf, A, k)`` with parameter covariance and R^2.

    Raises
    ------
    ValueError
        If fewer than 3 observations are supplied. The three-parameter
        model needs at least three data points to be well-identified.
    RuntimeError
        If ``curve_fit`` fails to converge (reraised from scipy with
        context).
    """
    if len(ces_history) < 3:
        raise ValueError(
            f"fit_ces_saturation needs at least 3 cycles; "
            f"got {len(ces_history)}."
        )

    if cycle_indices is None:
        cycle_indices = list(range(1, len(ces_history) + 1))
    if len(cycle_indices) != len(ces_history):
        raise ValueError(
            f"cycle_indices length ({len(cycle_indices)}) must match "
            f"ces_history length ({len(ces_history)})."
        )

    x = np.asarray(cycle_indices, dtype=float)
    y = np.asarray(ces_history, dtype=float)

    # Initial guess: C_inf slightly above the max observed (anticipating
    # further growth), A as the observed spread, k = 0.5 (middle of the
    # range Sun et al. report on Phi-3/Llama-3.2 SIL runs).
    y_max = float(y.max())
    y_min = float(y.min())
    spread = max(y_max - y_min, 1e-3)
    p0 = [y_max + 0.1 * abs(y_max) + 0.05, spread, 0.5]

    try:
        popt, pcov = curve_fit(
            _saturation_model, x, y,
            p0=p0,
            maxfev=5000,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Exponential-saturation fit did not converge "
            f"(N={len(ces_history)} cycles): {exc}"
        ) from exc

    c_inf, a, k = (float(v) for v in popt)

    y_pred = _saturation_model(x, c_inf, a, k)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    return EquilibriumFit(
        c_inf=c_inf,
        a=a,
        k=k,
        r2=float(r2),
        cov=np.asarray(pcov, dtype=float),
    )


# ============================================================================
# c* prediction
# ============================================================================


def predict_ceq_star(
    fit: EquilibriumFit,
    eps: float = 0.01,
) -> Optional[float]:
    """Predict the cycle ``c*`` at which the residual gap falls below
    ``eps * C_inf`` of the asymptote.

    Solves ``A * exp(-k * c) = eps * C_inf`` analytically:

    .. math::
        c^* = \\frac{1}{k} \\ln\\!\\left(\\frac{A}{\\varepsilon\\, C_\\infty}\\right).

    Parameters
    ----------
    fit : :class:`EquilibriumFit`
        Output of :func:`fit_ces_saturation`.
    eps : float, default 0.01
        Tolerance as a fraction of ``C_inf``. Default 1% of asymptote.

    Returns
    -------
    float or None
        Predicted equilibrium cycle. Returns ``None`` when the
        prediction is not well-defined: ``k <= 0`` (no convergence
        implied by the fit), or ``C_inf <= 0`` (degenerate fit), or
        the residual gap is already below ``eps * C_inf`` at c=1.
    """
    if fit.k <= 0.0:
        return None
    if fit.c_inf <= 0.0:
        return None
    if fit.a <= 0.0:
        # Fit thinks CES starts above asymptote (possible with noisy early
        # data). No meaningful c*.
        return None

    denom = eps * fit.c_inf
    if denom <= 0.0:
        return None

    ratio = fit.a / denom
    if ratio <= 1.0:
        # Residual gap already within tolerance at c=1.
        return 1.0

    return float(math.log(ratio) / fit.k)


# ============================================================================
# Triple-signal early-stop gate
# ============================================================================


def three_signal_gate(
    ces_history: Sequence[float],
    storage_rates: Sequence[float],
    mmlu_retentions: Sequence[float],
    *,
    ces_eps: float = 0.002,
    storage_frac: float = 0.05,
    mmlu_floor: float = 0.93,
    signals_required: int = 2,
    cycle: Optional[int] = None,
) -> EarlyStopDecision:
    """Check which of the three early-stop signals fire at the current cycle.

    Signals (each binary, evaluated at the end of the latest cycle):

    1. **CES gradient (parametric)** — ``|CES_c - CES_{c-1}| < ces_eps``
       AND ``|CES_{c-1} - CES_{c-2}| < ces_eps``. Two-cycle confirmation
       of flatlined improvement; avoids spurious stop from single-cycle
       noise.
    2. **Storage saturation (memory-native)** — latest cycle's
       ``storage_rate < storage_frac``. When the novelty filter has
       absorbed most similar queries, further cycles add little new
       training signal.
    3. **MMLU retention ceiling (FT-native)** — latest two cycles'
       MMLU retention both ``<= mmlu_floor``. The L2 anchor is pinning
       the model; additional fine-tuning trades general capability for
       little SIL gain.

    Gate fires (``should_stop = True``) when ``signals_required`` of 3
    signals agree. Default is 2-of-3, trading off one-cycle premature
    stops (low-probability) against one-cycle unnecessary continuations
    (low cost).

    Parameters
    ----------
    ces_history : sequence of float
        Per-cycle CES values, chronological.
    storage_rates : sequence of float
        Per-cycle fraction of queries whose verifier decision was STORE.
    mmlu_retentions : sequence of float
        Per-cycle MMLU retention ratio vs the pristine baseline.
    ces_eps : float, default 0.002
        CES gradient threshold. Two consecutive cycles both below this
        fires signal 1.
    storage_frac : float, default 0.05
        Storage-rate threshold for signal 2.
    mmlu_floor : float, default 0.93
        MMLU retention floor (matches ``CAEMConfig.forgetting_tolerance``).
    signals_required : int, default 2
        Number of signals that must fire for ``should_stop = True``.
    cycle : int, optional
        Current cycle number, recorded on the returned decision for
        audit-trail JSON.

    Returns
    -------
    :class:`EarlyStopDecision`
        Fields: which signals fired, overall stop decision, the cycle
        number (if supplied).
    """
    fired: List[str] = []

    # Signal 1: CES gradient (needs >= 3 cycles for two consecutive gaps).
    if len(ces_history) >= 3:
        g_last = abs(float(ces_history[-1]) - float(ces_history[-2]))
        g_prev = abs(float(ces_history[-2]) - float(ces_history[-3]))
        if g_last < ces_eps and g_prev < ces_eps:
            fired.append("ces_gradient")

    # Signal 2: storage-rate saturation.
    if len(storage_rates) >= 1:
        if float(storage_rates[-1]) < storage_frac:
            fired.append("storage_saturation")

    # Signal 3: MMLU retention at floor in two consecutive cycles.
    if len(mmlu_retentions) >= 2:
        r_last = float(mmlu_retentions[-1])
        r_prev = float(mmlu_retentions[-2])
        if r_last <= mmlu_floor and r_prev <= mmlu_floor:
            fired.append("mmlu_floor")

    should_stop = len(fired) >= signals_required
    return EarlyStopDecision(
        signals_fired=fired,
        should_stop=should_stop,
        cycle=cycle,
    )
