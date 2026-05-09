"""
scripts/run_calibration.py
===========================
Temperature Scaling (Category 3 hyperparameter)

Runs AFTER Cycle 0 evaluation is complete. Uses the disjoint calibration
slice (non-overlapping with the SIL training pool and the held-out eval
set).

One calibration step:
  1. Temperature scaling -- fits scalar T to minimise ECE (Expected
     Calibration Error) on the calibration set using L-BFGS
     (Guo et al. 2017). Applied to u_pre only.

Note
----
The legacy post-generation u_hat gate fit four signal weights
(u_token, u_dropout, u_consistency, u_entropy) via logistic regression.
That gate was removed when the UnifiedVerifier nine-signal Stage 5
became the single source of post-generation truth. The u_stored
composite weights are fixed by design (grounding dominates by
construction) and are NOT re-fit per cycle. ``log_signal_auroc`` below
is retained as a diagnostic only.

Results are:
  - Printed to stdout with before/after ECE comparison
  - Written to outputs/calibration/calibrated_config.json
  - Applied to the live config object if called inline from run_experiment.py

Thesis reference
----------------
  §4.5 Confidence calibration (temperature scaling)
  hyperparameter-reference.md Category 3 -- Empirically calibrated

Usage
-----
  # Standalone (after run_experiment.py Cycle 0):
  python -m scripts.run_calibration \\
      --cycle0_results outputs/eval \\
      --output_dir outputs/calibration

  # Or called inline from run_experiment.py via calibrate_pipeline()
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# v2 Fix 9b: read the benchmark roster from caem.config so the calibration
# fold tracks the live training panel. Hardcoded list before this fix was
# ["fever", "triviaqa", "natural_questions"] (v1 panel); v2 trains on
# ("fever", "triviaqa", "hotpotqa", "commonsense_qa") and uses
# ("truthfulqa", "strategyqa", "natural_questions") for transfer eval.
from caem.config import TRAINING_BENCHMARKS as _CFG_TRAINING_BENCHMARKS
from caem.config import TRANSFER_BENCHMARKS as _CFG_TRANSFER_BENCHMARKS

CALIB_BENCHMARK_DEFAULTS = list(_CFG_TRAINING_BENCHMARKS)
# Calibration is carved from the SIL training pool. Training benchmarks
# always use the "train" split; transfer benchmarks use their canonical
# eval split. Extra entries are kept for back-compat with legacy ablation
# scripts that still pass v1-only benchmark names.
CALIB_BENCHMARK_SPLITS = {
    **{bm: "train" for bm in _CFG_TRAINING_BENCHMARKS},
    "natural_questions": "validation",
    "truthfulqa": "validation",
    "strategyqa": "test",
    "arc_challenge": "test",
    "hotpotqa": "train",
    "commonsense_qa": "train",
}


# -----------------------------------------------------------------------------
# ECE computation
# -----------------------------------------------------------------------------

def expected_calibration_error(
    confidences: List[float],
    accuracies: List[float],
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins.

    ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|

    Parameters
    ----------
    confidences : list of float -- model confidence scores in [0, 1]
    accuracies  : list of float -- 1.0 if correct, 0.0 if wrong
    n_bins      : int -- number of bins (15 is standard)

    Returns
    -------
    float -- ECE in [0, 1]; lower is better
    """
    if not confidences:
        return 0.0

    n = len(confidences)
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Adjust last bin to be inclusive on the right
        in_bin = [
            (c, a) for c, a in zip(confidences, accuracies)
            if (lo <= c < hi) or (hi == 1.0 and c == 1.0)
        ]
        if not in_bin:
            continue
        bin_confs, bin_accs = zip(*in_bin)
        avg_conf = sum(bin_confs) / len(bin_confs)
        avg_acc = sum(bin_accs) / len(bin_accs)
        ece += (len(in_bin) / n) * abs(avg_acc - avg_conf)

    return ece


# -----------------------------------------------------------------------------
# Temperature scaling
# -----------------------------------------------------------------------------

def fit_temperature_scalar(
    logits: List[float],
    labels: List[int],
) -> float:
    """Fit a scalar temperature T that minimises negative log-likelihood.

    Platt/temperature scaling: p_calibrated = softmax(logits / T)

    Uses L-BFGS via scipy.optimize.minimize on the NLL loss.
    This is the standard Guo et al. (2017) calibration approach.

    Parameters
    ----------
    logits : list of float -- raw pre-softmax model outputs (u_pre values
             before sigmoid, or token log-probs)
    labels : list of int  -- 1 = correct answer, 0 = wrong answer

    Returns
    -------
    float -- fitted temperature T (T > 1 means model was over-confident)
    """
    try:
        from scipy.optimize import minimize
        import numpy as np
    except ImportError:
        logger.warning("scipy not available -- temperature scaling skipped. "
                       "Install: pip install scipy --break-system-packages")
        return 1.0

    logits_arr = np.array(logits, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.float64)

    finite_mask = np.isfinite(logits_arr) & np.isfinite(labels_arr)
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        logger.warning(
            "Temperature scaling: dropping %d non-finite samples.",
            n_bad,
        )
        logits_arr = logits_arr[finite_mask]
        labels_arr = labels_arr[finite_mask]

    if logits_arr.size < 20:
        logger.warning(
            "Temperature scaling: too few valid samples (%d) after filtering; using T=1.0.",
            int(logits_arr.size),
        )
        return 1.0

    def nll_loss(log_T) -> float:
        T = math.exp(log_T[0])
        scaled = np.clip(logits_arr / T, -60.0, 60.0)
        # Binary cross-entropy
        p = 1 / (1 + np.exp(-scaled))  # sigmoid
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(labels_arr * np.log(p) + (1 - labels_arr) * np.log(1 - p)))

    # L-BFGS-B bounds on log_T. T lives in [exp(-3), exp(3)] = [0.05, 20.1],
    # wide enough to cover any realistic over/under-confidence on Flan-T5.
    # Boundary hits are almost always a sign of pathological calibration
    # data (e.g. a Cycle-0 split where every sample is wrong, forcing T to
    # infinity) rather than a genuine NLL minimum.
    _LOG_T_LOW, _LOG_T_HIGH = -3.0, 3.0
    _BOUND_TOL = 0.05  # warn if log_T parks within this of a bound

    result = minimize(nll_loss, x0=[0.0], method="L-BFGS-B",
                      bounds=[(_LOG_T_LOW, _LOG_T_HIGH)], options={"maxiter": 500})
    T_fitted = math.exp(result.x[0])
    logger.info("Temperature scalar fitted: T = %.4f (log_T = %.4f)", T_fitted, result.x[0])
    logger.info("  NLL before: %.6f | NLL after: %.6f", nll_loss([0.0]), result.fun)

    # Diagnostic guards. These are logged, not raised -- the caller uses
    # T_fitted either way, because the downstream sigmoid/logit path is
    # robust to any positive T. The warning exists so a pathological fit
    # is visible in the Vast.ai log rather than silently propagating a
    # bound-clipped T into the next cycle.
    if not bool(getattr(result, "success", True)):
        logger.warning(
            "Temperature scaling: L-BFGS-B did not converge "
            "(message=%r, nit=%s). Returning the last-iterate T=%.4f; "
            "consider re-running with a larger calibration split.",
            getattr(result, "message", "?"), getattr(result, "nit", "?"),
            T_fitted,
        )
    log_T = float(result.x[0])
    if log_T <= _LOG_T_LOW + _BOUND_TOL or log_T >= _LOG_T_HIGH - _BOUND_TOL:
        logger.warning(
            "Temperature scaling: fitted log_T=%.4f is within %.3f of the "
            "bound [%.1f, %.1f] (T=%.4f). Likely pathological calibration "
            "data (extreme under- or over-confidence on the split); the "
            "returned T is bound-clipped rather than an interior optimum.",
            log_T, _BOUND_TOL, _LOG_T_LOW, _LOG_T_HIGH, T_fitted,
        )
    return T_fitted


# -----------------------------------------------------------------------------
# v2 Fix 12 — per-benchmark u_pre calibration
# -----------------------------------------------------------------------------

def fit_per_benchmark_temperatures(
    records: List[Dict[str, Any]],
    *,
    pooled_T: float = 1.0,
    min_n_per_bench: int = 20,
) -> Dict[str, float]:
    """Fit a per-benchmark temperature T_b on the calibration fold.

    Each record must carry ``benchmark`` (str), ``u_pre`` (float in [0,1]),
    and ``em`` (0/1 or float). Benchmarks with fewer than
    ``min_n_per_bench`` valid records fall back to ``pooled_T``.

    Returns ``{benchmark: T_b}`` for every benchmark seen in records.
    """
    by_bench: Dict[str, List[Tuple[float, int]]] = {}
    for r in records:
        bm = r.get("benchmark")
        u = r.get("u_pre")
        em = r.get("em")
        if bm is None or u is None or em is None:
            continue
        if not math.isfinite(float(u)):
            continue
        by_bench.setdefault(str(bm), []).append((float(u), int(bool(em))))

    out: Dict[str, float] = {}
    for bm, pairs in by_bench.items():
        if len(pairs) < min_n_per_bench:
            logger.info(
                "fit_per_benchmark_temperatures: bench=%s has %d samples (<%d) "
                "— falling back to pooled T=%.4f",
                bm, len(pairs), min_n_per_bench, pooled_T,
            )
            out[bm] = float(pooled_T)
            continue
        u_arr = [u for u, _ in pairs]
        em_arr = [e for _, e in pairs]
        # If a benchmark is fully correct or fully wrong, T-fitting is
        # ill-posed (NLL is unbounded in the bound-direction); fall back.
        # Logged at WARN because degenerate labels indicate a calibration
        # data issue (e.g., the cal-fold size / sampling for this bench
        # produced an all-correct or all-wrong slice) — the pooled-T
        # fallback masks the issue, so the operator should know.
        if sum(em_arr) == 0 or sum(em_arr) == len(em_arr):
            logger.warning(
                "fit_per_benchmark_temperatures: bench=%s has degenerate labels "
                "(sum=%d/%d) — falling back to pooled T=%.4f",
                bm, sum(em_arr), len(em_arr), pooled_T,
            )
            out[bm] = float(pooled_T)
            continue
        # Convert u_pre [0,1] → logits in the same way calibrate_pipeline does
        logits = []
        for u in u_arr:
            u_clamped = min(max(u, 1e-7), 1.0 - 1e-7)
            logits.append(math.log(u_clamped / (1.0 - u_clamped)))
        T_b = fit_temperature_scalar(logits, em_arr)
        out[bm] = float(T_b)
    return out


def fit_per_benchmark_safety_floors(
    records: List[Dict[str, Any]],
    *,
    pooled_floor: float = 0.38,
    target_precision: float = 0.50,
    min_n_per_bench: int = 20,
) -> Dict[str, float]:
    """Fit a per-benchmark safety_u_pre_min_b on the calibration fold.

    The OR-condition ``u_pre < safety_u_pre_min_b → Tier 3`` is a
    safety-first router gate, not an EM-style classifier threshold. We
    pick the SMALLEST u_pre threshold whose conditional precision
    ``P(EM=1 | u_pre >= threshold)`` exceeds ``target_precision`` on the
    calibration fold; below that threshold, the model is empirically
    less than ``target_precision``-likely to be correct, so the router
    forces Tier 3 to be safe.

    Falls back to ``pooled_floor`` for benchmarks with fewer than
    ``min_n_per_bench`` samples or no threshold satisfying the
    precision target.
    """
    by_bench: Dict[str, List[Tuple[float, int]]] = {}
    for r in records:
        bm = r.get("benchmark")
        u = r.get("u_pre")
        em = r.get("em")
        if bm is None or u is None or em is None:
            continue
        if not math.isfinite(float(u)):
            continue
        by_bench.setdefault(str(bm), []).append((float(u), int(bool(em))))

    out: Dict[str, float] = {}
    for bm, pairs in by_bench.items():
        if len(pairs) < min_n_per_bench:
            logger.info(
                "fit_per_benchmark_safety_floors: bench=%s has %d samples (<%d) "
                "— falling back to pooled floor=%.3f",
                bm, len(pairs), min_n_per_bench, pooled_floor,
            )
            out[bm] = float(pooled_floor)
            continue
        # Sweep candidate thresholds ascending; first one whose conditional
        # precision exceeds the target wins. Step size = 0.01 of the [0,1]
        # u_pre range — coarse enough to be stable on n=100 samples, fine
        # enough that any reasonable floor lands within 1 percentage point.
        pairs.sort(key=lambda t: t[0])
        chosen: Optional[float] = None
        for cand in [i * 0.01 for i in range(0, 101)]:
            keep = [(u, e) for u, e in pairs if u >= cand]
            if len(keep) < max(10, min_n_per_bench // 2):
                # too thin to score — stop walking up
                break
            prec = sum(e for _, e in keep) / float(len(keep))
            if prec >= target_precision:
                chosen = cand
                break
        if chosen is None:
            logger.info(
                "fit_per_benchmark_safety_floors: bench=%s no threshold met "
                "precision target %.2f — falling back to pooled floor=%.3f",
                bm, target_precision, pooled_floor,
            )
            out[bm] = float(pooled_floor)
        else:
            out[bm] = float(chosen)
    return out


# -----------------------------------------------------------------------------
# Diagnostic: signal AUROC (logged only, not used to fit weights)
# -----------------------------------------------------------------------------
#
# The legacy post-generation u_hat gate (which fit 4 signal weights via
# logistic regression on an AUROC-maximising objective) was removed when
# the UnifiedVerifier nine-signal stage became the single source of
# post-generation truth. The u_stored composite weights are fixed by
# design (grounding dominates by construction) and are NOT re-fit per
# cycle. Temperature scaling on u_pre is the only runtime-active
# calibration surface. The function below is kept solely as a
# diagnostic log of per-signal AUROC on the calibration slice.


def log_signal_auroc(
    signal_matrix: List[List[float]],
    labels: List[int],
) -> None:
    """Log AUROC of each post-generation signal. No state mutation."""
    if not signal_matrix:
        return
    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np
        X = np.array(signal_matrix, dtype=np.float64)
        y = np.array(labels, dtype=np.int32)
        if len(X) < 20 or y.sum() < 5 or y.sum() == len(y):
            return
        # Thesis-canonical label first, internal UnifiedVerifierOutput field
        # in parentheses so anyone cross-referencing the log to the code can
        # find the underlying attribute. u_consistency is stored as s_avg
        # (mean pairwise semantic similarity across M chains) and u_entropy
        # is stored as 1 - h_norm (complement of normalised semantic entropy).
        names = [
            ("u_token", "vout.u_token"),
            ("u_dropout", "vout.u_dropout"),
            ("u_consistency", "vout.s_avg"),
            ("u_entropy", "1 - vout.h_norm"),
        ]
        for i, (thesis_name, field_name) in enumerate(names):
            try:
                auc = roc_auc_score(y, X[:, i])
                logger.info(
                    "  AUROC (diag) -- %s (%s): %.4f",
                    thesis_name, field_name, auc,
                )
            except Exception:
                pass
    except ImportError:
        pass


# -----------------------------------------------------------------------------
# Cycle-N calibration cache reader (Step 2.1 -> Step 2.2 fast path)
# -----------------------------------------------------------------------------

def _load_u_pre_from_cycle_cache(
    cycle_calib_dir: Path,
    benchmarks: list,
    cycle: int,
) -> Optional[Tuple[List[float], List[int]]]:
    """Read u_pre + em from cycle-N cal-fold JSONs written by the harness.

    Returns ``(u_pre_logits, u_pre_labels)`` if every benchmark JSON exists
    and every sample has a non-null u_pre field; otherwise ``None`` (caller
    falls back to live pipeline re-scoring).
    """
    cycle_calib_dir = Path(cycle_calib_dir)
    if not cycle_calib_dir.exists():
        return None
    u_pre_logits: List[float] = []
    u_pre_labels: List[int] = []
    for bm in benchmarks:
        p = cycle_calib_dir / f"{bm}_cycle{cycle}.json"
        if not p.exists():
            return None
        try:
            data = json.load(open(p, "r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        for s in data.get("samples", []):
            up = s.get("u_pre", None)
            em = s.get("em", None)
            if up is None or em is None:
                return None
            u_pre_logits.append(float(up))
            u_pre_labels.append(int(bool(em)))
    if not u_pre_logits:
        return None
    return u_pre_logits, u_pre_labels


# -----------------------------------------------------------------------------
# Collect calibration signals from pipeline results
# -----------------------------------------------------------------------------

def collect_calibration_data(
    pipeline,
    calib_samples: Dict[str, list],
    *,
    record_jsonl_path: Optional[Path] = None,
    batch_size: int = 32,
) -> Tuple[List[float], List[int], List[List[float]], List[int]]:
    """Run calibration samples through the pipeline and collect signals.

    Parameters
    ----------
    pipeline     : CAEMPipeline
    calib_samples: dict[bm -> list of BenchmarkSample]
    batch_size   : int, default 32 -- number of samples processed in a
        single ``BatchPipeline.answer_batch`` call. Larger batches amortise
        kernel-launch overhead at the cost of GPU memory; the default
        matches the harness wiring example. Set to 1 to fall back to a
        per-sample serial path equivalent to the legacy implementation
        (useful for debugging numerical drift). Numerical equivalence
        bounds: u_stored ≤ 1e-3, signals ≤ 1e-2 (see caem/pipeline_batch.py
        lines 18-24 -- well below calibration bin discretisation).

    Returns
    -------
    u_pre_logits    : list of float -- pre-generation u_pre probabilities in
                                       [0, 1] (NOT pre-sigmoid logits despite
                                       the name). The identifier is retained
                                       for backwards compatibility with
                                       downstream callers; temperature
                                       scaling converts these to logits
                                       internally via logit(p) = log(p/(1-p)).
    u_pre_labels    : list of int   -- 1 = EM correct
    signal_matrix   : list of [u_token, u_dropout, u_sc, u_entropy]
    signal_labels   : list of int   -- 1 = EM correct (same as u_pre_labels)
    """
    from caem.pipeline_batch import BatchPipeline, BatchSample
    from eval.metrics import (
        exact_match,
        extract_arc_label,
        extract_cot_answer,
        extract_fever_label,
        extract_strategyqa_label,
        fever_accuracy,
        rouge_l,
    )

    def _score_em(prediction: str, gold: List[str], gold_label: Optional[str], bm: str) -> float:
        pred = extract_cot_answer(prediction)
        if bm == "fever":
            pred_label = extract_fever_label(pred)
            ref = gold_label or (gold[0] if gold else "not enough info")
            return fever_accuracy(pred_label, ref)
        if bm == "truthfulqa":
            return float(rouge_l(pred, gold) > 0.15)
        if bm == "strategyqa":
            pred_label = extract_strategyqa_label(pred)
            ref_label = extract_strategyqa_label(gold[0] if gold else "no")
            return float(pred_label == ref_label) if pred_label and ref_label else 0.0
        if bm == "arc_challenge":
            # ARC-Challenge: 4-choice MCQ (A-D). n_choices=4 is the
            # extract_arc_label default but pass it explicitly for clarity.
            pred_label = extract_arc_label(pred, n_choices=4)
            ref = gold[0] if gold else ""
            return exact_match(pred_label, ref)
        if bm == "commonsense_qa":
            # v2 Fix 9b — CommonsenseQA: 5-choice MCQ (A-E). Without
            # n_choices=5 the extractor silently rejects "E" answers
            # and digit-fallback misses 5.
            pred_label = extract_arc_label(pred, n_choices=5)
            ref = (gold_label or (gold[0] if gold else "")).strip().upper()
            return exact_match(pred_label, ref)
        return exact_match(pred, gold[0] if gold else "")

    u_pre_logits: List[float] = []
    u_pre_labels: List[int] = []
    signal_matrix: List[List[float]] = []
    signal_labels: List[int] = []  # labels for verifier-scored samples (Tier-2/3); mirrors signal_matrix rows

    logger.info("Collecting calibration signals ...")

    # Per-sample record accumulator for threshold-fitter consumption.
    # calibrate_pipeline() passes this list back to the caller, which writes
    # it out as JSONL so scripts/calibrate_thresholds.py can read u_stored
    # values from the calibration fold (disjoint from the eval fold).
    per_sample_records: List[Dict[str, Any]] = []

    # Wrap the serial pipeline once for the whole collection. BatchPipeline
    # is a thin orchestration wrapper around CAEMPipeline; constructing it is
    # cheap (no GPU/model state -- just keeps a reference). Reusing one
    # instance across batches avoids any per-batch setup overhead.
    batch_pipeline = BatchPipeline(pipeline)

    def _ingest_one(bm: str, sample: dict, result) -> None:
        """Per-sample bookkeeping shared by the batched and serial paths.

        Lifted out of the inner loop so the batched call site can reuse it
        unchanged, keeping the JSONL schema, signal-matrix population, and
        u_pre_labels list co-indexed with signal_labels exactly as the
        serial implementation produced them.
        """
        # u_pre as a logit proxy (already in [0,1]; convert for NLL)
        u_pre = result.pre_confidence.u_pre if result.pre_confidence else 0.5

        # EM label (benchmark-aware, aligned with eval/harness.py)
        gold = sample.get("answers", [])
        gold_label = sample.get("gold_label")
        em = _score_em(result.answer, gold, gold_label, bm)

        u_pre_logits.append(u_pre)
        u_pre_labels.append(int(em))

        # Per-sample dump for threshold calibration (Chapter 4
        # Eq:threshold-calibration). Must be done here, not downstream,
        # because only calibration-fold samples are seen here and the
        # disjointness guarantee requires they never mix with eval.
        vout_for_dump = getattr(result, "verifier_output", None)
        # Dump ALL 10 verifier primary signals required by
        # scripts/fit_composite_calibration.py + fit_conformal_gate.py.
        # cal_prob_composite.COMPOSITE_SIGNALS lists exactly:
        #   u_token, u_dropout, u_internal, s_avg, h_norm, p_entail,
        #   p_ground_max, p_ground_mean, p_ground_atomic, q_a_relevance.
        # If only a subset is dumped, the per-signal isotonic fitter
        # produces a degenerate composite (saw 2-feature regression on
        # 2026-04-25 06:14 UTC -> tau_store=1.0, store_n=0). The getattr
        # default 0.0 keeps this forward-compatible: q_a_relevance is a
        # post-Goal-2 attribute that may not exist on every snapshot of
        # UnifiedVerifierOutput; missing attributes degrade gracefully
        # to a flat isotonic curve at the fitter level.
        per_sample_records.append({
            "benchmark": bm,
            "id": sample.get("id"),
            "u_pre": float(u_pre),
            "em": float(em),
            "decision": getattr(vout_for_dump, "decision", None)
                if vout_for_dump is not None else None,
            # Composite output (kept for backward compat with older
            # downstream consumers that read u_stored directly).
            "u_stored": float(getattr(vout_for_dump, "u_stored", 0.0))
                if vout_for_dump is not None else None,
            # All 12 verifier primary signals (v2.1 — was 10 pre-Fix 6+7).
            # alias_overlap + entity_head_consistency added by Fix 6 + Fix 7
            # in 2026-05-06; the cal-fold writer was missed in that pass and
            # silently produced 10-signal cal-folds, leaving the per-bench
            # composite fitting on 10 signals instead of the architectural
            # 12. Restored here on 2026-05-09 after the cycle-0 calibration
            # gate failed twice with identical numbers (proof the data was
            # identical → the FEVER signal-mask was moot because the masked
            # signals weren't being read anyway).
            "u_token":         float(getattr(vout_for_dump, "u_token", 0.0))         if vout_for_dump is not None else None,
            "u_dropout":       float(getattr(vout_for_dump, "u_dropout", 0.0))       if vout_for_dump is not None else None,
            "u_internal":      float(getattr(vout_for_dump, "u_internal", 0.0))      if vout_for_dump is not None else None,
            "s_avg":           float(getattr(vout_for_dump, "s_avg", 0.0))           if vout_for_dump is not None else None,
            "h_norm":          float(getattr(vout_for_dump, "h_norm", 0.0))          if vout_for_dump is not None else None,
            "p_entail":        float(getattr(vout_for_dump, "p_entail", 0.0))        if vout_for_dump is not None else None,
            "p_ground_max":    float(getattr(vout_for_dump, "p_ground_max", 0.0))    if vout_for_dump is not None else None,
            "p_ground_mean":   float(getattr(vout_for_dump, "p_ground_mean", 0.0))   if vout_for_dump is not None else None,
            "p_ground_atomic": float(getattr(vout_for_dump, "p_ground_atomic", 0.0)) if vout_for_dump is not None else None,
            "q_a_relevance":   float(getattr(vout_for_dump, "q_a_relevance", 0.0))   if vout_for_dump is not None else None,
            "alias_overlap":   float(getattr(vout_for_dump, "alias_overlap", 0.0))   if vout_for_dump is not None else None,
            "entity_head_consistency": float(getattr(vout_for_dump, "entity_head_consistency", 0.0)) if vout_for_dump is not None else None,
        })

        # Signal matrix -- sourced from UnifiedVerifierOutput (Stage 5).
        # Session-42 merge: PostGenerationConfidenceEstimator was
        # removed; _tier2 now always emits post_confidence=None.
        # The four calibration signals now live on vout:
        #   u_token       = vout.u_token
        #   u_dropout     = vout.u_dropout
        #   u_consistency = vout.s_avg          (mean pairwise sim across M chains)
        #   u_entropy     = 1 - vout.h_norm     (complement of normalised semantic entropy)
        # Tier-1 hits skip Stage 5 and therefore have vout=None; they
        # are excluded from the signal matrix (correct: they carry no
        # post-generation signals). Both Tier-2 and Tier-3 verified
        # samples are included, which is a superset of the pre-fix
        # Tier-2-only collection path.
        # IMPORTANT: signal_labels must be co-indexed with signal_matrix.
        # Appending here (not above) ensures length parity.
        vout = getattr(result, "verifier_output", None)
        if vout is not None:
            signals = [
                float(getattr(vout, "u_token", 0.5)),
                float(getattr(vout, "u_dropout", 0.5)),
                float(getattr(vout, "s_avg", 0.5)),
                float(1.0 - getattr(vout, "h_norm", 0.5)),
            ]
            signal_matrix.append(signals)
            signal_labels.append(int(em))

    # Sanitise batch_size: <=0 would silently return zero-length chunks and
    # collect no calibration data; clamp to >=1. The BatchPipeline call with
    # a single-element list is well-defined and exercises the same code path
    # as the multi-sample case (only the batch dim shrinks), preserving the
    # numerical-equivalence contract for the degenerate case.
    chunk = max(1, int(batch_size))

    for bm, samples in calib_samples.items():
        for start in range(0, len(samples), chunk):
            chunk_samples = samples[start:start + chunk]
            # Build BatchSample list. store_to_memory=False matches the
            # serial pipeline.answer(..., store_to_memory=False) call so
            # calibration never mutates the cycle's memory store.
            batch_inputs = [
                BatchSample(
                    query=s["question"],
                    store_to_memory=False,
                    source_benchmark=bm,
                )
                for s in chunk_samples
            ]
            # Submit one batch. If the whole batch raises (e.g. CUDA OOM),
            # fall back to per-sample serial answer() so a single problem
            # sample doesn't lose the entire chunk's calibration signal.
            try:
                results = batch_pipeline.answer_batch(batch_inputs)
            except Exception as batch_exc:
                logger.warning(
                    "Calibration batch (bm=%s, start=%d, n=%d) failed: %s -- "
                    "falling back to per-sample serial path.",
                    bm, start, len(chunk_samples), batch_exc,
                )
                results = []
                for s in chunk_samples:
                    try:
                        results.append(
                            pipeline.answer(s["question"], store_to_memory=False)
                        )
                    except Exception as serial_exc:
                        logger.debug("Calibration sample skipped: %s", serial_exc)
                        results.append(None)

            # Per-sample bookkeeping. Wrapping each sample individually so a
            # single bad result (e.g. EM scorer raising on malformed gold)
            # doesn't kill bookkeeping for the rest of the chunk.
            for s, result in zip(chunk_samples, results):
                if result is None:
                    continue
                try:
                    _ingest_one(bm, s, result)
                except Exception as exc:
                    logger.debug("Calibration sample skipped: %s", exc)

    logger.info(
        "Calibration data: %d u_pre samples, %d signal-matrix samples "
        "(%d correct / %d total)",
        len(u_pre_logits),
        len(signal_matrix),
        sum(u_pre_labels),
        len(u_pre_labels),
    )
    assert len(signal_matrix) == len(signal_labels), (
        f"signal_matrix/signal_labels length mismatch: "
        f"{len(signal_matrix)} vs {len(signal_labels)}"
    )

    # Emit per-sample records to JSONL for the threshold-fitting script to
    # consume. Structured as a dict compatible with the eval-harness format
    # (meta + samples) so scripts/calibrate_thresholds.py reads one shape.
    if record_jsonl_path is not None:
        payload = {
            "meta": {
                "fold": "calibration",
                "n_samples": len(per_sample_records),
                "n_correct": int(sum(r["em"] for r in per_sample_records)),
            },
            "samples": per_sample_records,
        }
        record_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(record_jsonl_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "Wrote %d calibration-fold per-sample records to %s",
            len(per_sample_records), record_jsonl_path,
        )

    return u_pre_logits, u_pre_labels, signal_matrix, signal_labels


# -----------------------------------------------------------------------------
# Main calibration function (called from run_experiment.py or standalone)
# -----------------------------------------------------------------------------

def calibrate_pipeline(
    pipeline,
    calib_samples: Dict[str, list],
    config,
    output_dir: Path,
    *,
    batch_size: int = 32,
) -> Dict:
    """Fit temperature scalar T on u_pre; update config in-place; save to disk.

    Called from run_experiment.py after Cycle 0. Also usable standalone.

    Parameters
    ----------
    pipeline      : CAEMPipeline
    calib_samples : dict[bm -> list of BenchmarkSample]
    config        : CAEMConfig -- updated in-place with fitted T
    output_dir    : Path -- save calibrated_config.json here
    batch_size    : int, default 32 -- forwarded to
        :func:`collect_calibration_data` to control GPU batch size for
        the calibration sweep. Default matches the eval-harness wiring.

    Returns
    -------
    dict with keys: temperature_scalar, ece_before, ece_after, n_samples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Collect raw signals ------------------------------------------------ #
    # signal_matrix/signal_labels are unused at runtime (u_hat gate removed);
    # they are passed to log_signal_auroc below as a diagnostic only.
    # Per-sample u_stored records are emitted alongside so the threshold-
    # calibration script (Chapter 4 Eq:threshold-calibration) can fit on
    # the calibration fold, disjoint from the evaluation fold.
    u_pre_logits, u_pre_labels, signal_matrix, signal_labels = \
        collect_calibration_data(
            pipeline, calib_samples,
            record_jsonl_path=output_dir / "calibration_fold_samples.json",
            batch_size=batch_size,
        )

    if not u_pre_logits:
        logger.error("No calibration data collected -- aborting calibration.")
        return {}

    # Guard against non-finite u_pre values from upstream estimators.
    cleaned = [
        (float(u), int(l))
        for u, l in zip(u_pre_logits, u_pre_labels)
        if math.isfinite(float(u))
    ]
    if len(cleaned) != len(u_pre_logits):
        logger.warning(
            "Calibration: dropped %d non-finite u_pre samples before fitting.",
            len(u_pre_logits) - len(cleaned),
        )
    if not cleaned:
        logger.error("No finite u_pre samples remain after filtering -- aborting calibration.")
        return {}
    u_pre_logits = [u for u, _ in cleaned]
    u_pre_labels = [l for _, l in cleaned]

    # -- ECE before calibration ---------------------------------------------- #
    ece_before = expected_calibration_error(u_pre_logits, [float(l) for l in u_pre_labels])
    logger.info("ECE before temperature scaling: %.6f", ece_before)

    # -- Temperature scaling ------------------------------------------------- #
    # Convert u_pre [0,1] to logit space for temperature fitting
    logits = []
    for u in u_pre_logits:
        u_clamped = min(max(float(u), 1e-7), 1.0 - 1e-7)
        logits.append(math.log(u_clamped / (1.0 - u_clamped)))
    T = fit_temperature_scalar(logits, u_pre_labels)

    # Apply T and re-compute ECE
    def apply_temp(u: float, T: float) -> float:
        logit = math.log(max(u, 1e-7) / max(1 - u, 1e-7)) / T
        return 1 / (1 + math.exp(-logit))

    calibrated_confs = [apply_temp(u, T) for u in u_pre_logits]
    ece_after = expected_calibration_error(calibrated_confs, [float(l) for l in u_pre_labels])
    logger.info("ECE after  temperature scaling: %.6f (T=%.4f)", ece_after, T)

    # -- Diagnostic only: per-signal AUROC log (no state mutation) ---------- #
    log_signal_auroc(signal_matrix, signal_labels)

    # -- Update config in-place --------------------------------------------- #
    config.temperature_scalar = T       # the only calibrated runtime surface

    # -- v2 Fix 12 — per-benchmark T_b + safety_u_pre_min_b ---------------- #
    # Read the per-sample records we just wrote out via
    # collect_calibration_data → record_jsonl_path; group by benchmark
    # and call the fitting helpers. Both helpers fall back to the
    # pooled value for benchmarks with thin or label-degenerate data.
    per_bench_T: Dict[str, float] = {}
    per_bench_safety: Dict[str, float] = {}
    try:
        records_path = output_dir / "calibration_fold_samples.json"
        if records_path.exists():
            with open(records_path, "r", encoding="utf-8") as f:
                records_payload = json.load(f)
            records_list = records_payload.get("samples", [])
            per_bench_T = fit_per_benchmark_temperatures(
                records_list, pooled_T=T,
            )
            per_bench_safety = fit_per_benchmark_safety_floors(
                records_list,
                pooled_floor=float(getattr(config, "safety_u_pre_min", 0.38)),
                target_precision=0.85,
            )
            # Update CAEMConfig in-place so the live runtime picks them
            # up immediately.
            config.temperature_scalar_per_benchmark = dict(per_bench_T)
            config.safety_u_pre_min_per_benchmark = dict(per_bench_safety)
            logger.info(
                "Per-benchmark T_b fitted: %s",
                {k: round(v, 4) for k, v in per_bench_T.items()},
            )
            logger.info(
                "Per-benchmark safety_u_pre_min_b fitted: %s",
                {k: round(v, 4) for k, v in per_bench_safety.items()},
            )
        else:
            logger.warning(
                "Per-benchmark calibration helpers skipped: "
                "calibration_fold_samples.json missing at %s.",
                records_path,
            )
    except Exception as exc:
        logger.warning(
            "Per-benchmark calibration helpers failed (%s); "
            "falling back to pooled T + pooled safety_u_pre_min.", exc,
        )

    # -- Save results ------------------------------------------------------- #
    calib_result = {
        "temperature_scalar": T,
        "ece_before": ece_before,
        "ece_after":  ece_after,
        "n_samples":  len(u_pre_logits),
        "protocol":   "temperature_scaling_only",
        # v2 Fix 12: persist per-bench dicts so the cycle-N runtime
        # can rehydrate them when the orchestrator reloads
        # calibrated_config.json.
        "temperature_scalar_per_benchmark": dict(per_bench_T),
        "safety_u_pre_min_per_benchmark":   dict(per_bench_safety),
    }

    out_path = output_dir / "calibrated_config.json"
    with open(out_path, "w") as f:
        json.dump(calib_result, f, indent=2)
    logger.info("Calibration results saved -> %s", out_path)

    # Print summary
    print("\n" + "-" * 55)
    print("  CALIBRATION RESULTS")
    print("-" * 55)
    print(f"  Temperature scalar T: {T:.4f}")
    print(f"  ECE before: {ece_before:.6f}")
    print(f"  ECE after:  {ece_after:.6f}  (improvement: {ece_before - ece_after:.6f})")
    print("-" * 55)
    print("  -> u_stored composite weights are design-fixed (not calibrated).")
    print("-" * 55 + "\n")

    return calib_result


def calibrate_pipeline_temperature_only(
    pipeline,
    calib_samples: Dict[str, list],
    config,
    output_dir: Path,
    cycle: int,
    *,
    batch_size: int = 32,
) -> Dict:
    """Re-fit only the temperature scalar T on the disjoint calibration slice.

    Conservative per-cycle recalibration protocol (Ovadia et al. NeurIPS
    2019, Thulasidasan et al. 2019). Invoked unconditionally at each cycle
    boundary: T is the only runtime-active calibration surface. The
    u_stored composite weights are fixed by design and are NOT re-fit.

    Rationale for temperature-only:
      * Fine-tuning shifts the generator's logit scale every cycle, so T
        must be re-fit to keep u_pre calibrated.
      * The u_stored composite weights are stabler across cycles and are
        vulnerable to overfitting when re-fit on a 500-sample calibration
        split, so we keep them locked at their design-time values.

    Parameters
    ----------
    pipeline      : CAEMPipeline
    calib_samples : dict[bm -> list of BenchmarkSample]
    config        : CAEMConfig -- updated in-place with new T
    output_dir    : Path -- calibrated_config_cycle{N}.json written here
    cycle         : int -- for log/output tagging
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2026-04-27: per-cycle T-refit cache. Step 2.1 (cal-fold scoring under
    # the post-SIL model) writes cycle_{N}/calibration/{bench}_cycle{N}.json
    # via eval/harness.py. Those JSONs include u_pre per-sample as of the
    # same date. Step 2.2 (this function) ran on the same post-SIL model
    # immediately after, so cached u_pre values are byte-identical to a
    # fresh pipeline re-run. Reading from the cache avoids re-generating
    # 1500 cal-fold samples (~3.5 h) per cycle. Falls back to live re-run
    # if any cached sample lacks u_pre (older runs, schema mismatch).
    # 2026-05-07 audit fix: per-cycle T_b refit must also rebuild
    # the calibration_fold_samples.json that fit_per_benchmark_temperatures
    # and fit_per_benchmark_safety_floors read from. Pass record_jsonl_path
    # so the records JSON is written for this cycle.
    cycle_calib_dir = output_dir.parent / f"cycle_{cycle}" / "calibration"
    cycle_calib_dir.mkdir(parents=True, exist_ok=True)
    records_path = cycle_calib_dir / "calibration_fold_samples.json"

    cached = _load_u_pre_from_cycle_cache(
        cycle_calib_dir,
        list(calib_samples.keys()),
        cycle,
    )
    if cached is not None:
        u_pre_logits, u_pre_labels = cached
        logger.info(
            "Cycle %d T-refit: using cached u_pre from %d cal-fold samples "
            "(skip live re-scoring).",
            cycle, len(u_pre_logits),
        )
    else:
        u_pre_logits, u_pre_labels, _, _ = collect_calibration_data(
            pipeline, calib_samples,
            record_jsonl_path=records_path,
            batch_size=batch_size,
        )
    if not u_pre_logits:
        logger.warning(
            "Cycle %d temperature re-fit: no calibration data collected; "
            "keeping T=%.4f from the previous cycle.",
            cycle, config.temperature_scalar,
        )
        return {}

    # Same finite-value guarding as calibrate_pipeline().
    cleaned = [
        (float(u), int(l))
        for u, l in zip(u_pre_logits, u_pre_labels)
        if math.isfinite(float(u))
    ]
    if not cleaned:
        logger.warning(
            "Cycle %d temperature re-fit: no finite u_pre samples after filtering; "
            "keeping T=%.4f from the previous cycle.",
            cycle, config.temperature_scalar,
        )
        return {}
    u_pre_logits = [u for u, _ in cleaned]
    u_pre_labels = [l for _, l in cleaned]

    ece_before = expected_calibration_error(u_pre_logits, [float(l) for l in u_pre_labels])

    # Convert u_pre [0,1] to logit space for temperature fitting.
    logits = []
    for u in u_pre_logits:
        u_clamped = min(max(float(u), 1e-7), 1.0 - 1e-7)
        logits.append(math.log(u_clamped / (1.0 - u_clamped)))
    T_new = fit_temperature_scalar(logits, u_pre_labels)

    def _apply_temp(u: float, T: float) -> float:
        logit = math.log(max(u, 1e-7) / max(1 - u, 1e-7)) / T
        return 1 / (1 + math.exp(-logit))

    calibrated = [_apply_temp(u, T_new) for u in u_pre_logits]
    ece_after = expected_calibration_error(calibrated, [float(l) for l in u_pre_labels])

    # Use getattr with a 1.0 default so a cold-start config (no prior T) still
    # produces a valid "before" reading — rather than AttributeError-ing here.
    T_old = getattr(config, "temperature_scalar", 1.0)
    config.temperature_scalar = T_new

    # 2026-05-07 audit fix — per-cycle per-benchmark T_b + safety_u_pre_min
    # refit with EMA smoothing against the previous cycle's per-bench dicts.
    # Without this, per-bench T_b stayed frozen at the cycle-0 fit while
    # the pooled T was the only thing refit per-cycle — defeating the v2
    # Fix 12 design intent that each benchmark gets its own per-cycle
    # calibration drift correction.
    per_bench_T_after: Dict[str, float] = {}
    per_bench_safety_after: Dict[str, float] = {}
    per_bench_T_before: Dict[str, float] = dict(
        getattr(config, "temperature_scalar_per_benchmark", {}) or {}
    )
    per_bench_safety_before: Dict[str, float] = dict(
        getattr(config, "safety_u_pre_min_per_benchmark", {}) or {}
    )
    try:
        if records_path.exists():
            with open(records_path, "r", encoding="utf-8") as f:
                _records_payload = json.load(f)
            _records_list = _records_payload.get("samples", [])
            _per_bench_T_fresh = fit_per_benchmark_temperatures(
                _records_list, pooled_T=T_new,
            )
            _per_bench_safety_fresh = fit_per_benchmark_safety_floors(
                _records_list,
                pooled_floor=float(getattr(config, "safety_u_pre_min", 0.38)),
                target_precision=0.85,
            )
            # EMA-smooth per-bench against previous cycle's values. The
            # smoothing keeps per-cycle drift bounded the same way the
            # pooled T_b is smoothed elsewhere; alpha matches
            # config.adaptive_thresholds_ema_alpha (default 0.7).
            ema_alpha = float(getattr(config, "adaptive_thresholds_ema_alpha", 0.7))
            for _bm, _T_fresh in _per_bench_T_fresh.items():
                _T_prev = per_bench_T_before.get(_bm)
                if _T_prev is not None and math.isfinite(float(_T_prev)):
                    per_bench_T_after[_bm] = (
                        ema_alpha * float(_T_prev)
                        + (1.0 - ema_alpha) * float(_T_fresh)
                    )
                else:
                    per_bench_T_after[_bm] = float(_T_fresh)
            for _bm, _f_fresh in _per_bench_safety_fresh.items():
                _f_prev = per_bench_safety_before.get(_bm)
                if _f_prev is not None and math.isfinite(float(_f_prev)):
                    per_bench_safety_after[_bm] = (
                        ema_alpha * float(_f_prev)
                        + (1.0 - ema_alpha) * float(_f_fresh)
                    )
                else:
                    per_bench_safety_after[_bm] = float(_f_fresh)
            # Update config in-place so the live runtime picks up new values.
            config.temperature_scalar_per_benchmark = dict(per_bench_T_after)
            config.safety_u_pre_min_per_benchmark = dict(per_bench_safety_after)
            logger.info(
                "Cycle %d per-bench T_b refit (EMA α=%.2f): %s -> %s",
                cycle, ema_alpha,
                {k: round(v, 4) for k, v in per_bench_T_before.items()},
                {k: round(v, 4) for k, v in per_bench_T_after.items()},
            )
            logger.info(
                "Cycle %d per-bench safety_u_pre_min_b refit: %s -> %s",
                cycle,
                {k: round(v, 4) for k, v in per_bench_safety_before.items()},
                {k: round(v, 4) for k, v in per_bench_safety_after.items()},
            )
        else:
            logger.warning(
                "Cycle %d per-bench T_b/safety refit skipped: records JSON "
                "%s missing. Per-bench dicts stay at previous-cycle values.",
                cycle, records_path,
            )
    except Exception as exc:
        logger.warning(
            "Cycle %d per-bench T_b/safety refit failed (%s); "
            "per-bench dicts stay at previous-cycle values.",
            cycle, exc,
        )

    result = {
        "cycle": cycle,
        "temperature_before": T_old,
        "temperature_after": T_new,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "n_samples": len(u_pre_logits),
        "protocol": "conservative_temperature_only",
        "temperature_scalar_per_benchmark": dict(per_bench_T_after),
        "safety_u_pre_min_per_benchmark": dict(per_bench_safety_after),
    }

    out_path = output_dir / f"calibrated_config_cycle{cycle}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "Cycle %d temperature re-fit: T %.4f -> %.4f | ECE %.6f -> %.6f | "
        "per-bench T_b: %d benchmarks updated",
        cycle, T_old, T_new, ece_before, ece_after, len(per_bench_T_after),
    )
    return result


# -----------------------------------------------------------------------------
# Standalone entry point
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAEM temperature scaling + signal weight calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cycle0_results", default="outputs/eval",
                   help="Deprecated compatibility flag (unused when dataset_splits.json is available).")
    p.add_argument(
        "--checkpoints_dir",
        default="outputs",
        help="Root directory containing dataset_splits.json and cycle/memory checkpoints.",
    )
    p.add_argument("--output_dir", default="outputs/calibration",
                   help="Where to save calibration results.")
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(CALIB_BENCHMARK_DEFAULTS),
        help=(
            "Benchmarks to calibrate. Defaults to SIL benchmarks "
            "(fever, triviaqa, natural_questions)."
        ),
    )
    p.add_argument("--n_calib", type=int, default=1000,
                   help=(
                       "Fallback sample bound used only when calib_ids are unavailable. "
                       "Windowing uses indices [500:n_calib] per benchmark."
                   ))
    p.add_argument(
        "--passage_index",
        default="data/passage_index",
        help=(
            "Path to the Wikipedia PassageStore. Must match the path used by "
            "build_passage_index.py / run_experiment.py so Tier-3 RAG is "
            "consistent across calibration and evaluation."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()

    # Standalone mode: rebuild pipeline from scratch
    from scripts.hardware import print_hardware_summary
    profile = print_hardware_summary()

    from caem.config import CAEMConfig
    config = CAEMConfig()

    # Build pipeline (reuse build_pipeline from run_experiment)
    from scripts.run_experiment import _load_imports, build_pipeline
    m = _load_imports()

    import types
    # Pass the CLI-provided passage_index through to build_pipeline instead
    # of duplicating the "data/passage_index" default in a second place;
    # the single source of truth is the --passage_index argparse default.
    ns = types.SimpleNamespace(
        passage_index=args.passage_index,
    )
    pipeline = build_pipeline(config, ns, m)
    # The build_pipeline call above creates an encoder without device explicitly passed to it, 
    # but the pipeline object is built correctly. For standalone scripts, we override:
    pipeline.encoder = m["QueryEncoder"](model_name=config.sbert_model, device=profile.device)

    from eval.benchmarks import (
        load_arc_challenge,
        load_commonsense_qa,
        load_fever,
        load_hotpotqa,
        load_natural_questions,
        load_strategyqa,
        load_triviaqa,
        load_truthfulqa,
    )

    requested_benchmarks = [bm.strip().lower() for bm in args.benchmarks if bm.strip()]
    if not requested_benchmarks:
        requested_benchmarks = list(CALIB_BENCHMARK_DEFAULTS)

    loader_by_benchmark = {
        "fever": lambda: load_fever(split=CALIB_BENCHMARK_SPLITS["fever"]),
        "triviaqa": lambda: load_triviaqa(split=CALIB_BENCHMARK_SPLITS["triviaqa"]),
        "natural_questions": lambda: load_natural_questions(split=CALIB_BENCHMARK_SPLITS["natural_questions"]),
        "truthfulqa": lambda: load_truthfulqa(),
        "strategyqa": lambda: load_strategyqa(split=CALIB_BENCHMARK_SPLITS["strategyqa"]),
        "arc_challenge": lambda: load_arc_challenge(split=CALIB_BENCHMARK_SPLITS["arc_challenge"]),
        # v2 (2026-05-06): added for new training panel — CSQA + HotpotQA loaders.
        "commonsense_qa": lambda: load_commonsense_qa(split=CALIB_BENCHMARK_SPLITS.get("commonsense_qa", "train")),
        "hotpotqa": lambda: load_hotpotqa(split=CALIB_BENCHMARK_SPLITS.get("hotpotqa", "train")),
    }

    unknown_benchmarks = [bm for bm in requested_benchmarks if bm not in loader_by_benchmark]
    if unknown_benchmarks:
        raise ValueError(
            f"Unknown benchmarks for calibration: {unknown_benchmarks}. "
            f"Supported: {sorted(loader_by_benchmark.keys())}"
        )

    splits_path = Path(args.checkpoints_dir) / "dataset_splits.json"
    calib_ids_by_bm = {}
    if splits_path.exists():
        try:
            with open(splits_path, "r", encoding="utf-8") as f:
                splits = json.load(f)
            for bm, splits_dict in splits.items():
                calib_ids_by_bm[bm] = set(splits_dict.get("calib_ids", []))
            logger.info("Loaded exact calib_ids from %s", splits_path)
        except Exception as exc:
            logger.warning("Failed to load dataset_splits.json: %s", exc)
    else:
        logger.warning("dataset_splits.json not found at %s. Using fallback windowing.", splits_path)

    def load_and_filter(bm_name: str):
        all_samples = loader_by_benchmark[bm_name]()
        if bm_name in calib_ids_by_bm and calib_ids_by_bm[bm_name]:
            target_ids = calib_ids_by_bm[bm_name]
            filtered = [
                s for i, s in enumerate(all_samples)
                if str(s.get("id", i)) in target_ids or s.get("id", i) in target_ids
            ]
            if filtered:
                logger.info("Filtered %s to %d precise calib_ids", bm_name, len(filtered))
                return filtered

        window = all_samples[500:args.n_calib]
        if not window:
            logger.warning(
                "%s: fallback calibration window is empty (loaded %d samples, need n_calib > 500).",
                bm_name,
                len(all_samples),
            )
        else:
            logger.info("Using fallback calibration window for %s: %d samples", bm_name, len(window))
        return window

    benchmarks_with_calib_ids = [
        bm for bm in requested_benchmarks
        if bm in calib_ids_by_bm and calib_ids_by_bm[bm]
    ]
    if benchmarks_with_calib_ids:
        selected_benchmarks = benchmarks_with_calib_ids
        skipped_no_ids = [bm for bm in requested_benchmarks if bm not in set(selected_benchmarks)]
        if skipped_no_ids:
            logger.info(
                "Skipping benchmarks without calib_ids in dataset_splits.json: %s",
                skipped_no_ids,
            )
    else:
        selected_benchmarks = requested_benchmarks
        logger.warning(
            "No calib_ids found for requested benchmarks; using fallback windows for: %s",
            selected_benchmarks,
        )

    calib = {bm: load_and_filter(bm) for bm in selected_benchmarks}

    calibrate_pipeline(pipeline, calib, config, Path(args.output_dir))
