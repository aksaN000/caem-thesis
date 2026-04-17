"""caem.ablation.scoring -- CES axis composition for ablation runs.

This module reproduces the CES (CAEM Efficacy Score) axis definitions
that ``eval.reporting.build_table_cycle`` computes for Chapter 5, so a
full-CAEM ablation entry recovers the same CES a main-run cycle does
to within floating-point noise.

Axis definitions
----------------
  ACC : mean EM across benchmarks in the cycle.
  EPI : 1 - hallucination_rate at u_stored >= 0.50.
  RET : MMLU retention ratio (cycle_mmlu / baseline_mmlu), clipped to 1.
        For variants with ``requires_baseline_only=True`` the ratio is
        1.0 by fiat (the model is unchanged).
  CAL : 1 - 2*ECE, with ECE in [0, 0.5] computed from reliability bins
        over (u_stored, em) pairs.
  VER : 0.5 placeholder until labelled STORE/DISCARD ground truth lands.

CES combines the axes by ``eval.metrics.ces_score`` -- a geometric mean
with eps=0.01 clamp that reads off directly as "how close is each axis
to its ideal 1.0?".

The scoring helpers here take per-sample lists (em, u_stored) together
with optional retention + verifier scalars, rather than a harness-eval
JSON blob, so they stay easy to unit-test.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Sequence

from eval.metrics import (
    ces_score,
    hallucination_rate,
    reliability_bins,
)


@dataclass(frozen=True)
class CESAxes:
    """Five CES axes + the composite score + count metadata.

    All axes live on [0, 1]; NaN encodes "not measured this run" and
    is clamped to eps=0.01 before being passed to ``ces_score``.
    """

    acc: float
    epi: float
    ret: float
    cal: float
    ver: float
    ces: float
    n_samples: int
    n_scored_confidence: int  # samples with non-null u_stored

    def as_dict(self) -> dict:
        """JSON-serialisable plain dict."""
        return asdict(self)


# Thresholds and constants -- keep in lockstep with eval/reporting.py --------
CES_EPS = 0.01                      # axis clamp (matches ces_score eps)
HALLUCINATION_U_THRESHOLD = 0.50    # EPI = 1 - hall_rate(u >= 0.50)
DEFAULT_N_RELIABILITY_BINS = 10     # ECE bin count (matches reporting.py)
VER_PLACEHOLDER = 0.5               # until STORE/DISCARD gold labels land
CAL_MAX_ECE_CAP = 0.5               # CAL = 1 - 2*min(ECE, 0.5) stays in [0,1]


def _is_nan(x: Optional[float]) -> bool:
    """True if x is None or a NaN float."""
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _safe_mean(xs: Sequence[float]) -> float:
    """Mean ignoring NaN/None entries. Returns NaN when everything is NaN."""
    vals = [float(x) for x in xs if not _is_nan(x)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


# ----------------------------------------------------------------------------
# Axis builders
# ----------------------------------------------------------------------------

def _epi_axis(
    em_flags: Sequence[float],
    u_stored: Sequence[Optional[float]],
    u_threshold: float = HALLUCINATION_U_THRESHOLD,
) -> float:
    """EPI = 1 - hallucination_rate(em, u, threshold).

    Returns NaN if no confident samples are available, so the caller
    applies the eps clamp before feeding ces_score.
    """
    # eval.metrics.hallucination_rate already handles short-circuiting
    # when the input lists are empty; we mirror its behaviour here.
    em_clean: List[float] = []
    u_clean: List[float] = []
    for e, u in zip(em_flags, u_stored):
        if _is_nan(e) or u is None or _is_nan(u):
            continue
        em_clean.append(float(e))
        u_clean.append(float(u))
    if not em_clean:
        return float("nan")
    try:
        return 1.0 - hallucination_rate(em_clean, u_clean, u_threshold=u_threshold)
    except Exception:
        return float("nan")


def _ret_axis(
    cycle_mmlu: Optional[float],
    baseline_mmlu: Optional[float],
    requires_baseline_only: bool = False,
) -> float:
    """RET = min(cycle_mmlu / baseline_mmlu, 1.0).

    When the variant does no fine-tuning the retention ratio is 1.0 by
    construction (weights unchanged). If either measurement is missing
    we return NaN so the caller can clamp.
    """
    if requires_baseline_only:
        return 1.0
    if cycle_mmlu is None or baseline_mmlu is None or _is_nan(cycle_mmlu) or _is_nan(baseline_mmlu):
        return float("nan")
    base = float(baseline_mmlu)
    if base <= 0:
        return float("nan")
    return min(float(cycle_mmlu) / base, 1.0)


def _cal_axis(
    em_flags: Sequence[float],
    u_stored: Sequence[Optional[float]],
    n_bins: int = DEFAULT_N_RELIABILITY_BINS,
) -> float:
    """CAL = 1 - 2*min(ECE, 0.5).

    ECE is computed from ``reliability_bins`` over (u_stored, em).
    Samples with null u_stored are filtered (Tier-1 in a main run may
    not have a surfaced composite). Returns NaN when no samples remain.
    """
    conf: List[float] = []
    labels: List[float] = []
    for u, e in zip(u_stored, em_flags):
        if u is None or _is_nan(u) or _is_nan(e):
            continue
        conf.append(float(u))
        labels.append(float(e))
    if not conf:
        return float("nan")
    try:
        bins = reliability_bins(conf, labels, n_bins=n_bins)
        total = sum(b[2] for b in bins) or 1
        ece = sum((b[2] / total) * abs(b[0] - b[1]) for b in bins)
        return 1.0 - 2.0 * min(ece, CAL_MAX_ECE_CAP)
    except Exception:
        return float("nan")


def _ver_axis(balanced_accuracy: Optional[float]) -> float:
    """VER axis. Placeholder 0.5 unless gold-labelled STORE/DISCARD is given."""
    if balanced_accuracy is None or _is_nan(balanced_accuracy):
        return VER_PLACEHOLDER
    return float(balanced_accuracy)


# ----------------------------------------------------------------------------
# Top-level helpers
# ----------------------------------------------------------------------------

def ces_from_axes(
    acc: float,
    epi: float,
    ret: float,
    cal: float,
    ver: float,
    eps: float = CES_EPS,
) -> float:
    """Clamp NaN/<eps axes to eps and feed to ``ces_score``.

    Mirrors ``eval.reporting.build_table_cycle``'s clamp-then-score
    pattern so full-CAEM ablation CES matches the Table 5.1 CES.
    """
    inputs = [acc, epi, ret, cal, ver]
    clean = [eps if _is_nan(x) else max(float(x), eps) for x in inputs]
    return ces_score(*clean)


def ces_axes_from_cycle(
    em_flags: Sequence[float],
    u_stored: Sequence[Optional[float]],
    *,
    cycle_mmlu: Optional[float] = None,
    baseline_mmlu: Optional[float] = None,
    verifier_balanced_accuracy: Optional[float] = None,
    requires_baseline_only: bool = False,
    n_reliability_bins: int = DEFAULT_N_RELIABILITY_BINS,
) -> CESAxes:
    """Compute the five CES axes + composite from a single cycle's samples.

    Parameters
    ----------
    em_flags : sequence of {0.0, 1.0}
        EM indicator per eval sample (0 = wrong, 1 = correct). NaN
        entries are ignored when computing ACC.
    u_stored : sequence of Optional[float]
        u_stored per sample, in [0,1] or None/NaN (Tier-1 passthroughs
        in some runs carry no composite).
    cycle_mmlu, baseline_mmlu : float
        MMLU retention scalars for the RET axis. Optional when the
        variant does not touch model weights.
    verifier_balanced_accuracy : float
        Balanced accuracy of the verifier's STORE/DISCARD decision
        against gold labels. Optional; defaults to 0.5 placeholder.
    requires_baseline_only : bool
        Variant hint from ``AblationVariant.requires_baseline_only``.
        When True RET is fixed at 1.0.
    n_reliability_bins : int
        Passed through to ``reliability_bins`` for ECE/CAL.
    """
    # ACC: mean EM. NaN/None entries filtered by _safe_mean.
    acc = _safe_mean(em_flags)

    epi = _epi_axis(em_flags, u_stored)

    ret = _ret_axis(
        cycle_mmlu=cycle_mmlu,
        baseline_mmlu=baseline_mmlu,
        requires_baseline_only=requires_baseline_only,
    )

    cal = _cal_axis(em_flags, u_stored, n_bins=n_reliability_bins)
    ver = _ver_axis(verifier_balanced_accuracy)

    ces = ces_from_axes(acc, epi, ret, cal, ver)

    # Count bookkeeping
    n_samples = sum(1 for e in em_flags if not _is_nan(e))
    n_conf = sum(
        1 for u in u_stored if not _is_nan(u)
    )

    return CESAxes(
        acc=float(acc) if not _is_nan(acc) else float("nan"),
        epi=float(epi) if not _is_nan(epi) else float("nan"),
        ret=float(ret) if not _is_nan(ret) else float("nan"),
        cal=float(cal) if not _is_nan(cal) else float("nan"),
        ver=float(ver) if not _is_nan(ver) else float("nan"),
        ces=float(ces),
        n_samples=int(n_samples),
        n_scored_confidence=int(n_conf),
    )


def aggregate_axes(
    per_benchmark: Iterable[CESAxes],
) -> CESAxes:
    """Aggregate per-benchmark CESAxes into one cross-benchmark CESAxes.

    Each axis is the sample-weighted mean across benchmarks. CES is
    recomputed from the aggregated axes so it stays a geometric mean
    of the same scalars (NOT the mean of per-BM CES values).
    """
    rows = list(per_benchmark)
    if not rows:
        return CESAxes(
            acc=float("nan"), epi=float("nan"), ret=float("nan"),
            cal=float("nan"), ver=float("nan"),
            ces=ces_score(CES_EPS, CES_EPS, CES_EPS, CES_EPS, CES_EPS),
            n_samples=0, n_scored_confidence=0,
        )

    total_n = sum(r.n_samples for r in rows)
    total_conf = sum(r.n_scored_confidence for r in rows)

    def _wmean(attr: str, weights_attr: str = "n_samples") -> float:
        num = 0.0
        den = 0
        for r in rows:
            val = getattr(r, attr)
            w = getattr(r, weights_attr)
            if _is_nan(val) or w <= 0:
                continue
            num += float(val) * w
            den += w
        if den == 0:
            return float("nan")
        return num / den

    acc = _wmean("acc")
    epi = _wmean("epi")
    ret = _wmean("ret")
    # CAL is weighted by confidence-scored samples (it ignores Tier-1
    # passthroughs) so we use n_scored_confidence as the weight.
    cal = _wmean("cal", weights_attr="n_scored_confidence")
    ver = _wmean("ver")

    ces = ces_from_axes(acc, epi, ret, cal, ver)

    return CESAxes(
        acc=acc, epi=epi, ret=ret, cal=cal, ver=ver, ces=ces,
        n_samples=int(total_n),
        n_scored_confidence=int(total_conf),
    )
