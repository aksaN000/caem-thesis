"""Calibrated-probability composite for the CAEM verifier.

Replaces the fixed weighted sum over the 9 verifier signals with a
per-signal isotonic regression that maps each raw signal value to a
calibrated P(em=1 | signal). Signal contributions are then combined as a
log-odds sum (sigmoid-back-projected to [0, 1]).

Why this design (Branch C / 2026-04-25):
  - Each signal has a benchmark-dependent relationship with em-correctness
    (e.g. q_a_relevance shows Cohen's d -0.577 on FEVER but +0.498 on
    TriviaQA). A fixed weight cannot serve both signs simultaneously.
  - Per-signal isotonic regression auto-detects the direction (via Pearson
    correlation on the pooled fold) and produces a monotone calibration
    curve mapping raw values to calibrated P(em=1).
  - Dead signals (Cohen's d ≈ 0) get a near-flat calibration curve and
    contribute ~0 log-odds to the composite. No manual zero-weighting
    needed.
  - At deployment, no benchmark label is required — the calibration
    function is a constant per-signal map fitted once on the Cycle-0
    calibration fold.

Implementation notes:
  - Uses sklearn IsotonicRegression with auto-detected `increasing` based
    on Pearson correlation on the pooled fold.
  - Calibrated probabilities are clipped to [eps, 1-eps] to avoid log(0).
  - Composite log-odds = Σ_i log(P_i / (1 - P_i)); back-projected via
    sigmoid to [0, 1].
  - Signals with insufficient calibration data (n < 50) are skipped (no
    contribution).
  - Backward-compatible JSON serialization: each signal stores its
    isotonic knot points + direction flag.

References:
  - Cherian, Gibbs, Candès. "LLM Validity via Enhanced Conformal
    Prediction Methods." NeurIPS 2024. (motivates per-signal calibration)
  - Niculescu-Mizil & Caruana 2005. "Predicting Good Probabilities With
    Supervised Learning." ICML. (isotonic vs Platt comparison)

Usage:
    # Fit (one-time, offline, on labeled calibration fold):
    >>> calib = CalProbComposite()
    >>> calib.fit(labeled_samples)         # samples: list of dicts with em + signal values
    >>> calib.save("composite_calibration.json")

    # Apply (at runtime, no labels needed):
    >>> calib = CalProbComposite.load("composite_calibration.json")
    >>> u_stored = calib.predict(signal_values_dict)
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================ #
# Configuration: which signals enter the composite                             #
# ============================================================================ #

# 9 verifier signals plus q_a_relevance, all candidates for calibrated
# composite. Dead signals (u_token, u_dropout) auto-handled via flat
# isotonic curves; we include them so the composite is uniform across
# signal sources and the fitter has full information.
COMPOSITE_SIGNALS: Tuple[str, ...] = (
    "u_token",
    "u_dropout",
    "u_internal",
    "s_avg",
    "h_norm",
    "p_entail",
    "p_ground_max",
    "p_ground_mean",
    "p_ground_atomic",
    "q_a_relevance",
)

# Minimum samples per signal to fit calibration; below this, signal is skipped.
MIN_SAMPLES_PER_SIGNAL: int = 50

# Probability clipping to avoid log(0) / log(inf) in log-odds.
PROB_CLIP_EPS: float = 0.02


# ============================================================================ #
# Per-signal calibration (one isotonic per signal)                             #
# ============================================================================ #


class _SignalCalibration:
    """Per-signal isotonic-regression calibration, JSON-serialisable."""

    def __init__(
        self,
        signal: str,
        knot_x: List[float],
        knot_y: List[float],
        increasing: bool,
        n_train: int,
        pearson: float,
    ) -> None:
        self.signal = signal
        self.knot_x = list(knot_x)
        self.knot_y = list(knot_y)
        self.increasing = bool(increasing)
        self.n_train = int(n_train)
        self.pearson = float(pearson)

    @classmethod
    def fit(cls, signal: str, values: Sequence[float], labels: Sequence[int]) -> Optional["_SignalCalibration"]:
        from sklearn.isotonic import IsotonicRegression

        xs = np.asarray(values, dtype=np.float64)
        ys = np.asarray(labels, dtype=np.float64)
        if xs.size < MIN_SAMPLES_PER_SIGNAL:
            logger.warning(
                "cal_prob_composite: signal %s has n=%d < %d, skipping",
                signal, xs.size, MIN_SAMPLES_PER_SIGNAL,
            )
            return None
        pearson = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 1e-9 else 0.0
        increasing = pearson >= 0.0
        iso = IsotonicRegression(
            y_min=PROB_CLIP_EPS,
            y_max=1 - PROB_CLIP_EPS,
            increasing=increasing,
            out_of_bounds="clip",
        )
        iso.fit(xs, ys)
        # Extract knot points for serialization (sklearn stores X_thresholds_,
        # y_thresholds_ after fit).
        kx = list(map(float, iso.X_thresholds_))
        ky = list(map(float, iso.y_thresholds_))
        return cls(
            signal=signal,
            knot_x=kx,
            knot_y=ky,
            increasing=increasing,
            n_train=int(xs.size),
            pearson=pearson,
        )

    def predict(self, value: float) -> float:
        """Linear-interpolate the isotonic knots; clip to [eps, 1-eps]."""
        if not self.knot_x:
            return 0.5
        # numpy.interp does linear interpolation with edge clipping when xp is sorted.
        kx = np.asarray(self.knot_x)
        ky = np.asarray(self.knot_y)
        # When direction was reversed during fit, sklearn already encoded it
        # in the knot_y (decreasing y for increasing x). We just interp.
        if not np.all(np.diff(kx) >= 0):
            order = np.argsort(kx)
            kx = kx[order]
            ky = ky[order]
        p = float(np.interp(value, kx, ky))
        return float(min(1 - PROB_CLIP_EPS, max(PROB_CLIP_EPS, p)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "knot_x": self.knot_x,
            "knot_y": self.knot_y,
            "increasing": self.increasing,
            "n_train": self.n_train,
            "pearson": self.pearson,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_SignalCalibration":
        return cls(
            signal=str(d["signal"]),
            knot_x=list(d["knot_x"]),
            knot_y=list(d["knot_y"]),
            increasing=bool(d["increasing"]),
            n_train=int(d.get("n_train", 0)),
            pearson=float(d.get("pearson", 0.0)),
        )


# ============================================================================ #
# Calibrated-probability composite                                             #
# ============================================================================ #


class CalProbComposite:
    """Per-signal isotonic + log-odds sum, optionally boosted (Cherian et al. 2024).

    Composite score for a sample with signal values ``signals`` is::

        log_odds = Σ_i w_i · log(P_i / (1 - P_i)) + b   where:
          P_i = isotonic_i(signals[i])     (per-signal calibrated probability)
          w_i = 1                           (default — log-odds sum, identity weights)
          w_i = learned                     (Cherian-style boosting, when fit_boost=True)
        score = 1 / (1 + exp(-log_odds))    ∈ [PROB_CLIP_EPS, 1 - PROB_CLIP_EPS]

    Cherian boosting (Cherian, Gibbs & Candès NeurIPS 2024 — Enhanced Conformal
    Prediction for LLM Validity) learns per-signal weights via logistic
    regression on the calibration fold's em labels. Strictly subsumes the
    identity-weight case and can correct for residual cross-signal
    correlations that pooled isotonic doesn't capture.

    Signals not present in the calibration are skipped (no contribution).
    """

    SCHEMA_VERSION: str = "branchC.2026-04-25"

    def __init__(self) -> None:
        self.calibrations: Dict[str, _SignalCalibration] = {}
        self.metadata: Dict[str, Any] = {}
        # Cherian boost: logistic regression on per-signal calibrated probs.
        # When None, log-odds use identity weights (default behaviour).
        self.boost_weights: Optional[Dict[str, float]] = None
        self.boost_intercept: float = 0.0

    # ------------------------------------------------------------------ #
    # Fitting                                                              #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        samples: Sequence[Dict[str, Any]],
        signals: Sequence[str] = COMPOSITE_SIGNALS,
        fit_boost: bool = False,
        boost_C: float = 1.0,
    ) -> "CalProbComposite":
        """Fit per-signal isotonic on labeled samples.

        Each sample must have:
          - ``em``: int 0/1 (gold em-correctness on the calibration fold)
          - one float key per signal in ``signals``

        Parameters
        ----------
        fit_boost : bool, default False
            If True, additionally fit a Cherian-style logistic-regression layer
            over the per-signal calibrated probabilities. Stored as
            ``boost_weights`` + ``boost_intercept``. Identity weights (``False``)
            are equivalent to log-odds sum, which is what the original
            CalProbComposite paper formulation implies.
        """
        n = len(samples)
        em_rate = sum(1 for s in samples if s.get("em") == 1) / max(1, n)
        for sig in signals:
            vals = []
            labs = []
            for s in samples:
                v = s.get(sig)
                em = s.get("em")
                if isinstance(v, (int, float)) and em in (0, 1, 0.0, 1.0):
                    vals.append(float(v))
                    labs.append(int(em))
            cal = _SignalCalibration.fit(sig, vals, labs)
            if cal is not None:
                self.calibrations[sig] = cal
                logger.info(
                    "cal_prob_composite: signal=%s  n=%d  pearson=%+.3f  direction=%s",
                    sig, cal.n_train, cal.pearson, "increasing" if cal.increasing else "decreasing",
                )

        # Optional Cherian boosting: logistic regression on per-signal
        # calibrated probabilities (in log-odds space) → learned weights.
        if fit_boost and self.calibrations:
            self._fit_boost(samples, boost_C=boost_C)

        self.metadata = {
            "fitted_n": n,
            "em_rate": em_rate,
            "signals": list(signals),
            "schema_version": self.SCHEMA_VERSION,
            "boost_fit": bool(self.boost_weights is not None),
        }
        return self

    def _fit_boost(self, samples: Sequence[Dict[str, Any]], boost_C: float = 1.0) -> None:
        """Fit Cherian-style logistic regression on per-signal log-odds.

        Builds a feature matrix where each column is the calibrated
        per-signal log-odds ``log(P_i / (1 - P_i))``, then fits
        ``LogisticRegression`` on em labels. The learned coefficients become
        ``boost_weights`` (one weight per signal); the intercept becomes
        ``boost_intercept``. At ``predict()`` time these replace the identity
        weights in the log-odds aggregation.

        Skips samples with any missing signal value (must have all calibrated
        signals present to enter the regression).
        """
        from sklearn.linear_model import LogisticRegression
        sig_list = list(self.calibrations.keys())
        X_rows: List[List[float]] = []
        y_rows: List[int] = []
        for s in samples:
            em = s.get("em")
            if em not in (0, 1, 0.0, 1.0):
                continue
            row: List[float] = []
            ok = True
            for sig in sig_list:
                v = s.get(sig)
                if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    ok = False
                    break
                p = self.calibrations[sig].predict(float(v))
                p = max(PROB_CLIP_EPS, min(1 - PROB_CLIP_EPS, p))
                row.append(math.log(p / (1 - p)))
            if not ok:
                continue
            X_rows.append(row)
            y_rows.append(int(em))
        if len(X_rows) < 50:
            logger.warning(
                "cal_prob_composite._fit_boost: only n=%d complete rows; "
                "skipping boost (need ≥50)",
                len(X_rows),
            )
            return
        import numpy as _np
        X = _np.asarray(X_rows, dtype=_np.float64)
        y = _np.asarray(y_rows, dtype=_np.int64)
        lr = LogisticRegression(max_iter=2000, C=float(boost_C)).fit(X, y)
        self.boost_weights = {
            sig: float(c) for sig, c in zip(sig_list, lr.coef_[0])
        }
        self.boost_intercept = float(lr.intercept_[0])
        logger.info(
            "cal_prob_composite._fit_boost: fit on n=%d (em=1: %d). intercept=%+.3f",
            len(X_rows), int(y.sum()), self.boost_intercept,
        )
        for sig, w in self.boost_weights.items():
            logger.info("  boost weight  %-20s  %+.3f", sig, w)

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, signal_values: Dict[str, float]) -> float:
        """Compute composite score from a dict of signal values.

        Missing signals are skipped (no contribution). All-empty input
        returns 0.5 (neutral).

        When ``boost_weights`` is set (Cherian boost fitted), per-signal
        log-odds are weighted by the fitted coefficients and the boost
        intercept is added before the sigmoid; otherwise identity weights
        are used (plain log-odds sum).
        """
        log_odds = self.boost_intercept if self.boost_weights is not None else 0.0
        contrib = 0
        for sig, cal in self.calibrations.items():
            v = signal_values.get(sig)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            p = cal.predict(float(v))
            p = min(1 - PROB_CLIP_EPS, max(PROB_CLIP_EPS, p))
            lo = math.log(p / (1 - p))
            if self.boost_weights is not None:
                lo *= self.boost_weights.get(sig, 1.0)
            log_odds += lo
            contrib += 1
        if contrib == 0:
            return 0.5
        score = 1.0 / (1.0 + math.exp(-log_odds))
        return float(min(1 - PROB_CLIP_EPS, max(PROB_CLIP_EPS, score)))

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        out = {
            "schema_version": self.SCHEMA_VERSION,
            "metadata": self.metadata,
            "calibrations": [c.to_dict() for c in self.calibrations.values()],
            "boost_weights": self.boost_weights,
            "boost_intercept": self.boost_intercept,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(
            "cal_prob_composite: saved %d signals (boost=%s) -> %s",
            len(self.calibrations),
            "fit" if self.boost_weights is not None else "identity",
            path,
        )

    @classmethod
    def load(cls, path: str) -> "CalProbComposite":
        with open(path) as f:
            d = json.load(f)
        ver = d.get("schema_version")
        if ver != cls.SCHEMA_VERSION:
            logger.warning(
                "cal_prob_composite: schema mismatch (%s vs %s); attempting forward-compat load",
                ver, cls.SCHEMA_VERSION,
            )
        obj = cls()
        obj.metadata = dict(d.get("metadata", {}))
        for cd in d.get("calibrations", []):
            cal = _SignalCalibration.from_dict(cd)
            obj.calibrations[cal.signal] = cal
        bw = d.get("boost_weights")
        if isinstance(bw, dict) and bw:
            obj.boost_weights = {str(k): float(v) for k, v in bw.items()}
        obj.boost_intercept = float(d.get("boost_intercept", 0.0))
        logger.info(
            "cal_prob_composite: loaded %d signal calibrations (boost=%s) from %s",
            len(obj.calibrations),
            "fit" if obj.boost_weights is not None else "identity",
            path,
        )
        return obj

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        lines = [
            f"CalProbComposite ({self.SCHEMA_VERSION})",
            f"  fitted_n: {self.metadata.get('fitted_n', 'n/a')}",
            f"  em_rate:  {self.metadata.get('em_rate', float('nan')):.3f}",
            f"  signals:  {len(self.calibrations)}",
        ]
        for sig, cal in self.calibrations.items():
            lines.append(
                f"    {sig:<20}  n={cal.n_train:>5}  pearson={cal.pearson:+.3f}  "
                f"{'inc' if cal.increasing else 'dec'}"
            )
        return "\n".join(lines)
