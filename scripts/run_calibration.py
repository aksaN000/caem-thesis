"""
scripts/run_calibration.py
===========================
Temperature Scaling + Signal Weight Calibration (Category 3 hyperparameters)

Runs AFTER Cycle 0 evaluation is complete. Uses the 500-sample calibration
set (non-overlapping with the purity validation set and eval set).

Two calibration steps:
  1. Temperature scaling -- fits scalar T to minimise ECE (Expected Calibration
     Error) on the calibration set using L-BFGS. Adjusts token probability
     confidence to be better calibrated.
  2. Signal weight calibration -- fits û signal weights (u_token, u_dropout,
     u_consistency, u_entropy) to maximise AUROC on the calibration set using
     logistic regression (scikit-learn).

Results are:
  - Printed to stdout with before/after ECE comparison
  - Written to outputs/calibration/calibrated_config.json
  - Applied to the live config object if called inline from run_experiment.py

Thesis reference
----------------
  §4.5 Confidence calibration (temperature scaling)
  hyperparameter-reference.md Category 3 -- Empirically calibrated
  Chapter 5: report actual calibrated weights here (not the projected 0.20/0.20/0.20/0.40)

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
        return -np.mean(labels_arr * np.log(p) + (1 - labels_arr) * np.log(1 - p))

    result = minimize(nll_loss, x0=[0.0], method="L-BFGS-B",
                      bounds=[(-3.0, 3.0)], options={"maxiter": 500})
    T_fitted = math.exp(result.x[0])
    logger.info("Temperature scalar fitted: T = %.4f (log_T = %.4f)", T_fitted, result.x[0])
    logger.info("  NLL before: %.6f | NLL after: %.6f", nll_loss([0.0]), result.fun)
    return T_fitted


# -----------------------------------------------------------------------------
# Signal weight calibration (AUROC-based)
# -----------------------------------------------------------------------------

def fit_signal_weights(
    signal_matrix: List[List[float]],
    labels: List[int],
) -> List[float]:
    """Fit û signal weights using logistic regression to maximise AUROC.

    Each row of signal_matrix is [u_token, u_dropout, u_consistency, u_entropy]
    for one sample. Labels: 1 = answer correct, 0 = wrong.

    The fitted coefficients (after softmax normalisation) become the
    new signal weights. This directly answers the thesis question of
    whether SE's AUROC advantage (Farquhar et al. 2024 ≈ 0.79) causes
    the calibration to up-weight u_entropy toward ~0.40.

    Thesis note (hyperparameter-reference.md Category 3):
      "Initial weights are equal at 0.25 each. Post-calibration weights
       are projected to be ~0.20/0.20/0.20/0.40. Actual values reported
       in Chapter 5."

    Parameters
    ----------
    signal_matrix : list of [u_token, u_dropout, u_sc, u_entropy] per sample
    labels        : list of int (1 = correct)

    Returns
    -------
    list of 4 floats -- normalised weights summing to 1.0
    """
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
    except ImportError:
        logger.warning("scikit-learn not available -- signal weight calibration skipped. "
                       "Install: pip install scikit-learn --break-system-packages")
        return [0.25, 0.25, 0.25, 0.25]

    X = np.array(signal_matrix, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)

    if len(X) < 20 or y.sum() < 5:
        logger.warning("Too few samples for reliable signal weight calibration "
                       "(%d total, %d positive). Using equal weights.", len(X), int(y.sum()))
        return [0.25, 0.25, 0.25, 0.25]

    clf = LogisticRegression(fit_intercept=False, max_iter=1000, C=1.0)
    clf.fit(X, y)
    coefs = clf.coef_[0]

    # Normalise to sum to 1.0 (softmax)
    coefs = coefs - coefs.min()   # shift to non-negative
    total = coefs.sum()
    weights = (coefs / total).tolist() if total > 1e-9 else [0.25, 0.25, 0.25, 0.25]

    labels_str = ["u_token", "u_dropout", "u_consistency", "u_entropy"]
    for name, w in zip(labels_str, weights):
        logger.info("  Calibrated weight -- %s: %.4f", name, w)

    # AUROC for each signal
    try:
        from sklearn.metrics import roc_auc_score
        for i, name in enumerate(labels_str):
            auc = roc_auc_score(y, X[:, i])
            logger.info("  AUROC -- %s: %.4f", name, auc)
    except Exception:
        pass

    return weights


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
    u_pre_logits    : list of float -- raw u_pre (for temperature scaling)
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
    signal_labels: List[int] = []  # labels *only* for Tier-2 samples (mirrors signal_matrix rows)

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

                # Signal matrix -- only available for Tier 2 (post-generation conf)
                # PostGenerationConfidence fields: u_token, u_dropout,
                # u_consistency (NOT u_sc), u_entropy (NOT h_entropy_norm)
                # IMPORTANT: signal_labels must be co-indexed with signal_matrix.
                # Appending here (not above) ensures len(signal_matrix) == len(signal_labels)
                # even when Tier-1 / Tier-3 samples are present in the calibration set.
                if result.post_confidence is not None:
                    pc = result.post_confidence
                    signals = [
                        getattr(pc, "u_token", 0.5),
                        getattr(pc, "u_dropout", 0.5),
                        getattr(pc, "u_consistency", 0.5),   # correct field name
                        getattr(pc, "u_entropy", 0.5),       # correct field name (already 1-H)
                    ]
                    signal_matrix.append(signals)
                    signal_labels.append(int(em))  # label for this Tier-2 sample only

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
    """Fit T and signal weights; update config in-place; save to disk.

    Called from run_experiment.py after Cycle 0. Also usable standalone.

    Parameters
    ----------
    pipeline      : CAEMPipeline
    calib_samples : dict[bm -> list of BenchmarkSample]
    config        : CAEMConfig -- updated in-place with fitted values
    output_dir    : Path -- save calibrated_config.json here

    Returns
    -------
    dict with keys: temperature, signal_weights, ece_before, ece_after
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Collect raw signals ------------------------------------------------ #
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
    logger.info("ECE after  temperature scaling: %.6f (ΔT=%.4f)", ece_after, T)

    # -- Signal weight calibration ------------------------------------------- #
    if signal_matrix:
        new_weights = fit_signal_weights(signal_matrix, signal_labels)
    else:
        logger.warning("No Tier 2 samples found in calibration set -- "
                       "signal weights not calibrated. Using equal weights (0.25 each).")
        new_weights = [0.25, 0.25, 0.25, 0.25]

    # -- Update config in-place ---------------------------------------------- #
    config.temperature_scalar = T       # new field written to config
    config.u_hat_weight_token       = new_weights[0]
    config.u_hat_weight_dropout     = new_weights[1]
    config.u_hat_weight_sc          = new_weights[2]
    config.u_hat_weight_entropy     = new_weights[3]

    # -- Save results -------------------------------------------------------- #
    calib_result = {
        "temperature_scalar": T,
        "ece_before": ece_before,
        "ece_after":  ece_after,
        "n_samples":  len(u_pre_logits),
        "n_tier2_samples": len(signal_matrix),
        "signal_weights": {
            "u_token":       new_weights[0],
            "u_dropout":     new_weights[1],
            "u_consistency": new_weights[2],   # correct field name
            "u_entropy":     new_weights[3],
        },
        "projected_weights_from_plan": {
            "u_token":       0.20,
            "u_dropout":     0.20,
            "u_consistency": 0.20,
            "u_entropy":     0.40,
        },
        "note": (
            "Actual calibrated values -- these replace the projected 0.20/0.20/0.20/0.40 "
            "estimates in the thesis plan. Report these in Chapter 5."
        ),
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
    print(f"\n  û Signal weights (calibrated vs projected):")
    names = ["u_token", "u_dropout", "u_consistency", "u_entropy"]
    proj  = [0.20, 0.20, 0.20, 0.40]
    for name, w, p in zip(names, new_weights, proj):
        delta = w - p
        print(f"    {name:<12} : {w:.4f}  (projected {p:.2f}, Δ={delta:+.4f})")
    print("-" * 55)
    print("  -> Report actual values in Chapter 5 (Table: Category 3 calibration)")
    print("-" * 55 + "\n")

    return calib_result


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
    ns = types.SimpleNamespace(
        no_nli=False, no_rag=False,
        passage_index="data/passage_index",
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
