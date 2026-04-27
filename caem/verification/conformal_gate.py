"""Conformal storage gate for CAEM.

Phase 2.4 (Branch C 2026-04-25): replaces the four-outcome decision tree's
hard-coded threshold compares with a conformal-calibrated split-CP gate.

Two-threshold design (the user-validated version replacing my earlier
"dual-threshold" framing — see Phase 2.4 design discussion):

  τ_store  -- fitted on labeled calibration fold at quantile that achieves
              empirical em-precision ≥ (1 − α_store) on that fold.
              At deployment: u_stored ≥ τ_store ⇒ STORE (Tier-1 + train eligible).
  τ_defer  -- fitted at empirical precision ≥ (1 − α_defer) where
              α_defer > α_store. At deployment: τ_defer ≤ u_stored < τ_store
              ⇒ DEFERRED (memory only, retroverify queue).

Both thresholds come from the SAME labeled calibration fold; what differs is
the precision target (α). This is the standard split-CP construction (Mohri &
Hashimoto 2024 §3.2; Yadkori et al. 2024 conformal risk control).

References:
  - Mohri & Hashimoto. "Language Models with Conformal Factuality Guarantees."
    ICML 2024. arXiv:2402.10978.
  - Yadkori, Vovk, Verma, Steck, Faisal, Diethe, Aceron, Stanforth, Kohli,
    Cemgil, Tomasev. "Mitigating LLM Hallucinations via Conformal Abstention."
    DeepMind 2024. arXiv:2405.01563.
  - Cherian, Gibbs, Candès. "LLM Validity via Enhanced Conformal Prediction
    Methods." NeurIPS 2024. arXiv:2406.09714.

Bootstrap: at Cycle 0 the JSON does not exist; the verifier reverts to the
legacy fixed-threshold ``_decide()`` (config.store_threshold +
config.defer_threshold), which is the existing four-outcome tree. Once
Step 7.0.2 fits and writes the JSON, subsequent cycles read it and use the
conformal-calibrated thresholds.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class ConformalStorageGate:
    """Calibrated two-threshold storage gate.

    Fitted on a labeled calibration fold (em-correctness available for each
    sample). At deployment, only the fitted thresholds are read; no labels
    needed.
    """

    SCHEMA_VERSION: str = "branchC.2026-04-25"

    def __init__(
        self,
        tau_store: float,
        tau_defer: float,
        alpha_store: float,
        alpha_defer: float,
        cal_n: int,
        store_n: int,
        store_precision: float,
        defer_n: int,
        defer_precision: float,
    ) -> None:
        self.tau_store = float(tau_store)
        self.tau_defer = float(tau_defer)
        self.alpha_store = float(alpha_store)
        self.alpha_defer = float(alpha_defer)
        self.cal_n = int(cal_n)
        self.store_n = int(store_n)
        self.store_precision = float(store_precision)
        self.defer_n = int(defer_n)
        self.defer_precision = float(defer_precision)

    # ----------------------------------------------------------- decide #
    def decide(self, u_stored: float, p_ground_max: float, abstain_pg: float) -> Tuple[str, bool]:
        """Apply the conformal-calibrated four-outcome tree.

        Same outcome semantics as the existing ``UnifiedVerifier._decide``;
        only the threshold values change. The early-exit confabulation gate
        (``u_internal ≥ 0.70 AND p_ground_max ≤ 0.20`` ⇒ DISCARD) is applied
        BEFORE this gate is consulted in ``verify()``.
        """
        if u_stored >= self.tau_store:
            return "STORE", False
        if u_stored >= self.tau_defer:
            return "DEFERRED", False
        if p_ground_max < abstain_pg:
            return "ABSTAIN", True
        return "DISCARD", False

    # ----------------------------------------------------------- fit/load #
    @classmethod
    def fit(
        cls,
        calib_samples: Sequence[Dict[str, Any]],
        score_key: str = "u_stored",
        em_key: str = "em",
        alpha_store: float = 0.20,
        alpha_defer: float = 0.40,
        min_n: int = 20,
    ) -> "ConformalStorageGate":
        """Fit τ_store and τ_defer at the precision targets (1−α).

        For each precision target, walks the calibration fold sorted descending
        by score, accumulates correctness, and picks the largest score (smallest
        τ) for which empirical precision over the top-k stays ≥ target.

        ``alpha_store`` defaults to 0.20 (target: 80% precision); ``alpha_defer``
        defaults to 0.40 (target: 60% precision). Both are fitted on the SAME
        fold; the only operational difference is the precision target.
        """
        target_store = 1.0 - alpha_store
        target_defer = 1.0 - alpha_defer

        # Build (score, em) pairs from labeled fold
        pairs: List[Tuple[float, int]] = []
        for s in calib_samples:
            sc = s.get(score_key)
            em = s.get(em_key)
            if isinstance(sc, (int, float)) and em in (0, 1, 0.0, 1.0):
                pairs.append((float(sc), int(em)))
        pairs.sort(key=lambda p: -p[0])  # descending by score
        n = len(pairs)
        if n < min_n:
            raise ValueError(
                f"ConformalStorageGate.fit: only n={n} labeled samples; "
                f"need ≥{min_n} for stable threshold."
            )

        # Walk and find the largest k where empirical precision still meets target
        def _find_threshold(target: float) -> Tuple[float, int, float]:
            correct = 0
            last = (1.0, 0, 1.0)  # tau=1 means store nothing if target unreachable
            for k, (sc, em) in enumerate(pairs, start=1):
                correct += em
                if k >= min_n:
                    if (correct / k) >= target:
                        last = (float(sc), int(k), float(correct / k))
                    else:
                        break
            return last

        tau_store, store_n, store_prec = _find_threshold(target_store)
        tau_defer, defer_n, defer_prec = _find_threshold(target_defer)

        # Sanity: defer must not be stricter than store (else user error)
        if tau_defer > tau_store:
            logger.warning(
                "ConformalStorageGate.fit: tau_defer (%.3f) > tau_store (%.3f); "
                "this implies α_defer < α_store; flipping to enforce ordering.",
                tau_defer, tau_store,
            )
            tau_defer = tau_store
            defer_n = store_n
            defer_prec = store_prec

        gate = cls(
            tau_store=tau_store,
            tau_defer=tau_defer,
            alpha_store=alpha_store,
            alpha_defer=alpha_defer,
            cal_n=n,
            store_n=store_n,
            store_precision=store_prec,
            defer_n=defer_n,
            defer_precision=defer_prec,
        )
        logger.info(
            "ConformalStorageGate fitted: τ_store=%.3f (k=%d, prec=%.3f, α=%.2f)  "
            "τ_defer=%.3f (k=%d, prec=%.3f, α=%.2f)  cal_n=%d",
            gate.tau_store, gate.store_n, gate.store_precision, gate.alpha_store,
            gate.tau_defer, gate.defer_n, gate.defer_precision, gate.alpha_defer,
            gate.cal_n,
        )
        return gate

    # ----------------------------------------------------------- I/O #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "tau_store": self.tau_store,
            "tau_defer": self.tau_defer,
            "alpha_store": self.alpha_store,
            "alpha_defer": self.alpha_defer,
            "cal_n": self.cal_n,
            "store_n": self.store_n,
            "store_precision": self.store_precision,
            "defer_n": self.defer_n,
            "defer_precision": self.defer_precision,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(
            "ConformalStorageGate saved -> %s  (τ_store=%.3f τ_defer=%.3f)",
            path, self.tau_store, self.tau_defer,
        )

    @classmethod
    def load(cls, path: str) -> "ConformalStorageGate":
        with open(path) as f:
            d = json.load(f)
        return cls(
            tau_store=d["tau_store"],
            tau_defer=d["tau_defer"],
            alpha_store=d.get("alpha_store", 0.20),
            alpha_defer=d.get("alpha_defer", 0.40),
            cal_n=int(d.get("cal_n", 0)),
            store_n=int(d.get("store_n", 0)),
            store_precision=float(d.get("store_precision", 0.0)),
            defer_n=int(d.get("defer_n", 0)),
            defer_precision=float(d.get("defer_precision", 0.0)),
        )

    def summary(self) -> str:
        return (
            f"ConformalStorageGate ({self.SCHEMA_VERSION})\n"
            f"  τ_store: {self.tau_store:.3f}  (α={self.alpha_store:.2f}, "
            f"k={self.store_n}, cal precision={self.store_precision:.3f})\n"
            f"  τ_defer: {self.tau_defer:.3f}  (α={self.alpha_defer:.2f}, "
            f"k={self.defer_n}, cal precision={self.defer_precision:.3f})\n"
            f"  cal_n:   {self.cal_n}"
        )
