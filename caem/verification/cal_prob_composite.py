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
    "alias_overlap",  # v2 Fix 6 — Wikidata alias coverage of answer entities
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

    # v2 (2026-05-06): nested schema supports per-benchmark calibrations.
    # v1 (2026-04-25) holds a single global calibration. Loader auto-detects
    # which format the JSON uses and populates either `self.calibrations`
    # only (v1 → still works as fallback / legacy) or both
    # `self.calibrations` (pooled fallback) PLUS `self.per_benchmark`
    # (the dict the verifier dispatches by source_benchmark).
    SCHEMA_VERSION_V1: str = "branchC.2026-04-25"
    SCHEMA_VERSION_V2: str = "branchC.2026-05-06"
    SCHEMA_VERSION: str = SCHEMA_VERSION_V2  # default for new fits

    def __init__(self) -> None:
        self.calibrations: Dict[str, _SignalCalibration] = {}
        self.metadata: Dict[str, Any] = {}
        # Cherian boost: logistic regression on per-signal calibrated probs.
        # When None, log-odds use identity weights (default behaviour).
        self.boost_weights: Optional[Dict[str, float]] = None
        self.boost_intercept: float = 0.0
        # v2 Fix 2 — per-benchmark calibration registry. When non-empty,
        # predict() routes by source_benchmark to the right per-bench
        # CalProbComposite. When empty (v1 fallback), predict() uses
        # the pooled `self.calibrations` directly. Both are populated by
        # fit_per_benchmark(); v1 fit() leaves this empty.
        self.per_benchmark: Dict[str, "CalProbComposite"] = {}

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
    # Per-benchmark fitting (v2)                                           #
    # ------------------------------------------------------------------ #

    def fit_per_benchmark(
        self,
        samples_by_benchmark: Dict[str, Sequence[Dict[str, Any]]],
        signals: Sequence[str] = COMPOSITE_SIGNALS,
        fit_boost: bool = True,
        boost_C: float = 0.01,
    ) -> "CalProbComposite":
        """v2 Fix 2 — fit one composite per benchmark + a pooled global fallback.

        Each entry in ``samples_by_benchmark`` is a labelled per-benchmark
        cal-fold (same row schema as ``fit()`` expects: ``em`` plus per-
        signal floats). The pooled union is fit into ``self.calibrations``
        as the global fallback (used when the verifier sees a query with
        unknown ``source_benchmark`` at deployment); each per-benchmark
        slice is fit into a child CalProbComposite stored in
        ``self.per_benchmark[benchmark]``.

        At predict time the verifier passes ``source_benchmark`` through;
        ``predict()`` dispatches to the per-bench composite when known and
        falls back to the global pooled composite otherwise.

        Parameters
        ----------
        samples_by_benchmark
            Mapping benchmark name -> labelled cal-fold rows.
        signals, fit_boost, boost_C
            Same semantics as :meth:`fit`. boost_C default 0.01 matches the
            cycle-0 sweep selection of the v1 trajectory.
        """
        # 1. Pooled global fallback fit
        pooled: List[Dict[str, Any]] = []
        for bm_samples in samples_by_benchmark.values():
            pooled.extend(bm_samples)
        if pooled:
            self.fit(
                pooled, signals=signals,
                fit_boost=fit_boost, boost_C=boost_C,
            )
            self.metadata["pooled_n"] = len(pooled)

        # 2. Per-benchmark fits
        for bm, bm_samples in samples_by_benchmark.items():
            child = CalProbComposite()
            child.fit(
                bm_samples, signals=signals,
                fit_boost=fit_boost, boost_C=boost_C,
            )
            child.metadata["benchmark"] = bm
            self.per_benchmark[bm] = child
            logger.info(
                "cal_prob_composite.fit_per_benchmark: %s n=%d signals=%d boost=%s",
                bm, len(bm_samples), len(child.calibrations),
                "fit" if child.boost_weights is not None else "identity",
            )

        self.metadata["per_benchmark_count"] = len(self.per_benchmark)
        self.metadata["per_benchmark_names"] = sorted(self.per_benchmark.keys())
        self.metadata["schema_version"] = self.SCHEMA_VERSION_V2
        return self

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        signal_values: Dict[str, float],
        source_benchmark: Optional[str] = None,
    ) -> float:
        """Compute composite score from a dict of signal values.

        v2 Fix 2 — when ``source_benchmark`` is supplied AND a matching
        per-benchmark calibration exists in ``self.per_benchmark``, route
        through that per-benchmark child composite. Otherwise fall back to
        the pooled global composite in ``self.calibrations``.

        Missing signals are skipped (no contribution). All-empty input
        returns 0.5 (neutral).

        When ``boost_weights`` is set (Cherian boost fitted), per-signal
        log-odds are weighted by the fitted coefficients and the boost
        intercept is added before the sigmoid; otherwise identity weights
        are used (plain log-odds sum).
        """
        # v2 dispatch: route by source_benchmark if a per-bench child exists
        if (
            source_benchmark is not None
            and source_benchmark in self.per_benchmark
        ):
            return self.per_benchmark[source_benchmark]._predict_pooled(signal_values)

        # Fallback path (v1-equivalent): pooled global composite
        return self._predict_pooled(signal_values)

    def _predict_pooled(self, signal_values: Dict[str, float]) -> float:
        """Internal — predict using this composite's own pooled calibrations
        + boost. Same as the v1 predict() body."""
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

    def _calibration_block(self) -> Dict[str, Any]:
        """Serialize this composite's pooled calibrations + boost as a block.
        Used for both top-level v1 writes and per-benchmark child writes."""
        return {
            "calibrations": [c.to_dict() for c in self.calibrations.values()],
            "boost_weights": self.boost_weights,
            "boost_intercept": self.boost_intercept,
        }

    def _load_calibration_block(self, block: Dict[str, Any]) -> None:
        """Populate this composite's pooled calibrations + boost from a block.
        Used for both top-level v1 loads and per-benchmark child loads."""
        self.calibrations = {}
        for cd in block.get("calibrations", []):
            cal = _SignalCalibration.from_dict(cd)
            self.calibrations[cal.signal] = cal
        bw = block.get("boost_weights")
        if isinstance(bw, dict) and bw:
            self.boost_weights = {str(k): float(v) for k, v in bw.items()}
        else:
            self.boost_weights = None
        self.boost_intercept = float(block.get("boost_intercept", 0.0))

    def save(self, path: str) -> None:
        # v2 Fix 2 — emit nested schema when per_benchmark is populated;
        # otherwise emit the v1-compatible flat schema for back-compat with
        # tooling that expects pre-v2 composite_calibration.json shape.
        if self.per_benchmark:
            out = {
                "schema_version": self.SCHEMA_VERSION_V2,
                "metadata": self.metadata,
                "global": self._calibration_block(),
                "per_benchmark": {
                    bm: child._calibration_block()
                    for bm, child in self.per_benchmark.items()
                },
            }
        else:
            out = {
                "schema_version": self.SCHEMA_VERSION_V1,
                "metadata": self.metadata,
                **self._calibration_block(),
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(
            "cal_prob_composite: saved %s (signals=%d, per_benchmark=%d, boost=%s) -> %s",
            "v2 nested" if self.per_benchmark else "v1 flat",
            len(self.calibrations),
            len(self.per_benchmark),
            "fit" if self.boost_weights is not None else "identity",
            path,
        )

    @classmethod
    def load(cls, path: str) -> "CalProbComposite":
        with open(path) as f:
            d = json.load(f)
        ver = d.get("schema_version")
        obj = cls()
        obj.metadata = dict(d.get("metadata", {}))

        # v2 nested schema: {global: {...}, per_benchmark: {bm: {...}, ...}}
        if ver == cls.SCHEMA_VERSION_V2 or "global" in d or "per_benchmark" in d:
            obj._load_calibration_block(d.get("global", {}))
            for bm, block in (d.get("per_benchmark") or {}).items():
                child = cls()
                child._load_calibration_block(block)
                child.metadata = {"benchmark": bm}
                obj.per_benchmark[bm] = child
            logger.info(
                "cal_prob_composite: loaded v2 nested (global signals=%d, per_benchmark=%d) from %s",
                len(obj.calibrations), len(obj.per_benchmark), path,
            )
        else:
            # v1 flat schema (legacy single-composite JSON)
            if ver and ver != cls.SCHEMA_VERSION_V1:
                logger.warning(
                    "cal_prob_composite: unknown schema_version %s; treating as v1 flat",
                    ver,
                )
            obj._load_calibration_block(d)
            logger.info(
                "cal_prob_composite: loaded v1 flat (%d signal calibrations, boost=%s) from %s",
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
