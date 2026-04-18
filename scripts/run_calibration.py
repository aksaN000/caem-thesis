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
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


CALIB_BENCHMARK_DEFAULTS = ["fever", "triviaqa", "natural_questions"]
CALIB_BENCHMARK_SPLITS = {
    # Calibration is carved from the SIL training pool.
    "fever": "train",
    "triviaqa": "train",
    "natural_questions": "train",
    # Optional transfer-only benchmarks.
    "truthfulqa": "validation",
    "strategyqa": "test",
    "arc_challenge": "test",
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

    result = minimize(nll_loss, x0=[0.0], method="L-BFGS-B",
                      bounds=[(-3.0, 3.0)], options={"maxiter": 500})
    T_fitted = math.exp(result.x[0])
    logger.info("Temperature scalar fitted: T = %.4f (log_T = %.4f)", T_fitted, result.x[0])
    logger.info("  NLL before: %.6f | NLL after: %.6f", nll_loss([0.0]), result.fun)
    return T_fitted


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
        names = ["u_token", "u_dropout", "u_consistency", "u_entropy"]
        for i, name in enumerate(names):
            try:
                auc = roc_auc_score(y, X[:, i])
                logger.info("  AUROC (diag) -- %s: %.4f", name, auc)
            except Exception:
                pass
    except ImportError:
        pass


# -----------------------------------------------------------------------------
# Collect calibration signals from pipeline results
# -----------------------------------------------------------------------------

def collect_calibration_data(
    pipeline,
    calib_samples: Dict[str, list],
) -> Tuple[List[float], List[int], List[List[float]], List[int]]:
    """Run calibration samples through the pipeline and collect signals.

    Parameters
    ----------
    pipeline     : CAEMPipeline
    calib_samples: dict[bm -> list of BenchmarkSample]

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
            pred_label = extract_arc_label(pred)
            ref = gold[0] if gold else ""
            return exact_match(pred_label, ref)
        return exact_match(pred, gold[0] if gold else "")

    u_pre_logits: List[float] = []
    u_pre_labels: List[int] = []
    signal_matrix: List[List[float]] = []
    signal_labels: List[int] = []  # labels for verifier-scored samples (Tier-2/3); mirrors signal_matrix rows

    logger.info("Collecting calibration signals ...")

    for bm, samples in calib_samples.items():
        for sample in samples:
            try:
                result = pipeline.answer(sample["question"], store_to_memory=False)
                # u_pre as a logit proxy (already in [0,1]; convert for NLL)
                u_pre = result.pre_confidence.u_pre if result.pre_confidence else 0.5

                # EM label (benchmark-aware, aligned with eval/harness.py)
                gold = sample.get("answers", [])
                gold_label = sample.get("gold_label")
                em = _score_em(result.answer, gold, gold_label, bm)

                u_pre_logits.append(u_pre)
                u_pre_labels.append(int(em))

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
    return u_pre_logits, u_pre_labels, signal_matrix, signal_labels


# -----------------------------------------------------------------------------
# Main calibration function (called from run_experiment.py or standalone)
# -----------------------------------------------------------------------------

def calibrate_pipeline(
    pipeline,
    calib_samples: Dict[str, list],
    config,
    output_dir: Path,
) -> Dict:
    """Fit temperature scalar T on u_pre; update config in-place; save to disk.

    Called from run_experiment.py after Cycle 0. Also usable standalone.

    Parameters
    ----------
    pipeline      : CAEMPipeline
    calib_samples : dict[bm -> list of BenchmarkSample]
    config        : CAEMConfig -- updated in-place with fitted T
    output_dir    : Path -- save calibrated_config.json here

    Returns
    -------
    dict with keys: temperature_scalar, ece_before, ece_after, n_samples
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Collect raw signals ------------------------------------------------ #
    # signal_matrix/signal_labels are unused at runtime (u_hat gate removed);
    # they are passed to log_signal_auroc below as a diagnostic only.
    u_pre_logits, u_pre_labels, signal_matrix, signal_labels = \
        collect_calibration_data(pipeline, calib_samples)

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

    # -- Save results ------------------------------------------------------- #
    calib_result = {
        "temperature_scalar": T,
        "ece_before": ece_before,
        "ece_after":  ece_after,
        "n_samples":  len(u_pre_logits),
        "protocol":   "temperature_scaling_only",
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

    u_pre_logits, u_pre_labels, _, _ = collect_calibration_data(pipeline, calib_samples)
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

    result = {
        "cycle": cycle,
        "temperature_before": T_old,
        "temperature_after": T_new,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "n_samples": len(u_pre_logits),
        "protocol": "conservative_temperature_only",
    }

    out_path = output_dir / f"calibrated_config_cycle{cycle}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "Cycle %d temperature re-fit: T %.4f -> %.4f | ECE %.6f -> %.6f",
        cycle, T_old, T_new, ece_before, ece_after,
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
        load_fever,
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
