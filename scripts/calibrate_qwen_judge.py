"""
scripts/calibrate_qwen_judge.py
================================
Fit Platt scaling to align FrozenQwenJudge P(yes) with MiniCheck
P(supported) on an overlap fold.

Purpose
-------
When the AdaptiveNLIJudge dispatcher routes a long-hypothesis sample
to Qwen-judge, the raw P(yes) from Qwen is not on the same scale as
MiniCheck's P(supported) — Qwen is a general chat model fine-tuned for
helpfulness, tends to say "yes" more often than a specialised NLI head.
Without calibration, the u_stored composite would see a systematically
higher p_entail for long-hypothesis samples than for short-hypothesis
samples, biasing storage decisions on long-answer benchmarks.

This script:
  1. Loads 500 (premise, hypothesis) pairs where hypothesis fits BOTH
     judges (≤ 408 MC-tokens) — typically from Cycle-0 eval JSONs.
  2. Runs both MiniCheck and Qwen-judge on every pair.
  3. Fits Platt scaling parameters (a, b) such that the Qwen-calibrated
     probability matches MiniCheck's probability distribution:
         logit(P_cal) = a · logit(P_qwen_raw) + b
  4. Reports correlation (Pearson ρ, MAE) and saves (a, b) to JSON.

Gate
----
If Pearson ρ < 0.70 on the 500-sample fold, the fit is unreliable and
the script exits non-zero (runner halts). This gate is stricter than
usual because Platt scaling assumes monotonic relationship between
raw and calibrated logits; low correlation means Qwen-judge's ranking
of (premise, hypothesis) pairs disagrees with MiniCheck's in a way
that linear calibration cannot repair.

Usage
-----
  python scripts/calibrate_qwen_judge.py \\
    --cycle0_eval_dir outputs/cycle_0/eval \\
    --output_json outputs/calibration/qwen_judge_platt.json \\
    --n_samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("calibrate_qwen_judge")


def collect_overlap_pairs(
    eval_dir: Path, n_samples: int, mc_tokenizer, hyp_cap: int = 408,
) -> List[Tuple[str, str]]:
    """Collect (premise, hypothesis) pairs from Cycle-0 eval JSONs whose
    hypothesis fits within MC's hyp_cap so both judges can score them.

    Hypothesis = <query> + <model's answer> (same format verify_batch uses).
    Premise = first retrieved passage from Tier 3 RAG (if present), else
    just the query — Cycle-0 eval JSONs record only the final per-sample
    record so we reconstruct this from the samples' `question` + `prediction`
    and assume a short synthetic premise. For calibration purposes the
    EXACT premise doesn't matter as long as both judges see the same one.
    """
    import random
    pairs: List[Tuple[str, str]] = []
    seen = 0
    rng = random.Random(42)

    for json_path in sorted(eval_dir.glob("*_cycle0.json")):
        with open(json_path) as f:
            d = json.load(f)
        for sample in d.get("samples", []):
            if seen >= n_samples * 3:  # collect pool, sub-sample later
                break
            q = sample.get("question", "").strip()
            a = sample.get("prediction", "").strip()
            if not q or not a:
                continue
            # Synthetic premise: use the question itself as a placeholder
            # premise. For calibration we just need both judges to see
            # an identical (premise, hypothesis) pair; the absolute
            # premise content doesn't determine the calibration fit.
            premise = q
            hypothesis = f"Question: {q}\nAnswer: {a}"
            hyp_len = len(mc_tokenizer(hypothesis, add_special_tokens=False).input_ids)
            if hyp_len > hyp_cap:
                continue  # must fit MC path for calibration
            pairs.append((premise, hypothesis))
            seen += 1

    rng.shuffle(pairs)
    return pairs[:n_samples]


def fit_platt(p_mc: np.ndarray, p_qwen: np.ndarray) -> Tuple[float, float, dict]:
    """Fit Platt scaling: logit(P_cal) = a · logit(P_raw) + b.

    Uses ordinary least squares on the logit-space pairs, which gives
    a closed-form solution and is well-conditioned as long as p_qwen
    shows some variation.
    """
    eps = 1e-6
    p_mc_c = np.clip(p_mc, eps, 1.0 - eps)
    p_qwen_c = np.clip(p_qwen, eps, 1.0 - eps)
    lm = np.log(p_mc_c / (1.0 - p_mc_c))     # target (MC logits)
    lq = np.log(p_qwen_c / (1.0 - p_qwen_c))  # input (Qwen logits)
    # OLS: lm ≈ a * lq + b
    A = np.vstack([lq, np.ones_like(lq)]).T
    sol, residuals, rank, sv = np.linalg.lstsq(A, lm, rcond=None)
    a, b = float(sol[0]), float(sol[1])

    # Compute diagnostics
    lm_pred = a * lq + b
    mae_logit = float(np.mean(np.abs(lm_pred - lm)))
    # Back to probability space for MAE-prob
    p_cal = 1.0 / (1.0 + np.exp(-lm_pred))
    mae_prob = float(np.mean(np.abs(p_cal - p_mc)))
    # Pearson correlation in logit space (monotonicity check)
    rho = float(np.corrcoef(lq, lm)[0, 1])

    diagnostics = {
        "n": int(len(p_mc)),
        "pearson_rho_logit": round(rho, 4),
        "mae_logit": round(mae_logit, 4),
        "mae_probability": round(mae_prob, 4),
        "mean_p_mc": round(float(np.mean(p_mc)), 4),
        "mean_p_qwen_raw": round(float(np.mean(p_qwen)), 4),
        "mean_p_qwen_calibrated": round(float(np.mean(p_cal)), 4),
        "std_p_mc": round(float(np.std(p_mc)), 4),
        "std_p_qwen_raw": round(float(np.std(p_qwen)), 4),
    }
    return a, b, diagnostics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle0_eval_dir", type=str, default="outputs/cycle_0/eval")
    ap.add_argument("--output_json", type=str,
                    default="outputs/calibration/qwen_judge_platt.json")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--min_rho", type=float, default=0.70,
                    help="Minimum Pearson correlation in logit space; below this "
                         "the calibration is deemed unreliable and script fails.")
    ns = ap.parse_args()

    eval_dir = Path(ns.cycle0_eval_dir)
    if not eval_dir.exists():
        logger.error("Cycle-0 eval dir not found: %s", eval_dir)
        return 2

    # Lazy imports — only load heavy models if we actually need to run.
    from caem.config import CAEMConfig
    from caem.model_loader import load_base_generator
    from caem.verification.minicheck import _MiniCheckJudge as MiniCheckJudge
    from caem.verification.qwen_judge import FrozenQwenJudge
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cfg = CAEMConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MiniCheck
    logger.info("Loading MiniCheck (%s)...", "lytang/MiniCheck-Flan-T5-Large")
    mc_model = AutoModelForSeq2SeqLM.from_pretrained(
        "lytang/MiniCheck-Flan-T5-Large", dtype=torch.bfloat16,
    ).to(device).eval()
    mc_tok = AutoTokenizer.from_pretrained("lytang/MiniCheck-Flan-T5-Large")
    mc_judge = MiniCheckJudge(mc_model, mc_tok, device=device)

    # Load Qwen (BASE weights, frozen)
    logger.info("Loading base Qwen-3B (frozen)...")
    qwen_model, qwen_tok = load_base_generator(
        cfg.base_model_name, device=device, dtype=torch.bfloat16,
    )
    qwen_model.eval()
    qwen_judge = FrozenQwenJudge(qwen_model, qwen_tok, device=device,
                                 platt_a=1.0, platt_b=0.0)

    # Collect overlap pairs
    logger.info("Collecting %d overlap pairs from %s...", ns.n_samples, eval_dir)
    pairs = collect_overlap_pairs(eval_dir, ns.n_samples, mc_tok)
    if len(pairs) < 50:
        logger.error("Only %d overlap pairs collected (need ≥50); aborting.",
                     len(pairs))
        return 3
    logger.info("Collected %d overlap pairs.", len(pairs))

    premises = [p for p, _ in pairs]
    hypotheses = [h for _, h in pairs]

    # Score with both judges. Chunk to avoid CUDA OOM: with max_context=8192
    # and Qwen-3B, batch sizes >16 OOM on a 32GB 5090. MiniCheck has tighter
    # per-pair budget (512 tokens) so it tolerates larger chunks.
    import numpy as _np

    def _chunked_score(judge, prems, hyps, chunk):
        out = []
        for i in range(0, len(prems), chunk):
            out.append(judge.batch_entail_prob(prems[i:i+chunk], hyps[i:i+chunk]))
        return _np.concatenate(out)

    logger.info("Running MiniCheck on %d pairs (chunk=64)...", len(pairs))
    p_mc = _chunked_score(mc_judge, premises, hypotheses, chunk=64)

    logger.info("Running Qwen-judge (raw, no calibration) on %d pairs (chunk=8)...", len(pairs))
    p_qwen_raw = _chunked_score(qwen_judge, premises, hypotheses, chunk=8)

    # Fit Platt
    logger.info("Fitting Platt scaling...")
    a, b, diag = fit_platt(p_mc, p_qwen_raw)
    logger.info("Platt fit: a=%.4f  b=%.4f", a, b)
    logger.info("Diagnostics: %s", json.dumps(diag, indent=2))

    # Gate
    if diag["pearson_rho_logit"] < ns.min_rho:
        logger.error(
            "FAIL: logit-space Pearson ρ=%.4f < %.4f threshold; "
            "Platt calibration unreliable. Investigate Qwen-judge prompt "
            "or use a larger calibration set.",
            diag["pearson_rho_logit"], ns.min_rho,
        )
        return 4

    # Save
    out_path = Path(ns.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "platt_a": a,
        "platt_b": b,
        "diagnostics": diag,
        "min_rho_threshold": ns.min_rho,
        "fit_source": "ordinary_least_squares_on_logits",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote Platt parameters to %s.", out_path)
    logger.info("Qwen judge calibration PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
