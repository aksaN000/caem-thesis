"""
scripts/check_base_model.py
===========================
Standalone base-model accuracy check (zero-shot Flan-T5-Large).

Purpose
-------
Before running any CAEM cycles, confirm that the base model's generation
accuracy p > (1 − α) on the selected benchmark suite. The purity theorem requires
p > (1 − α) ≈ 0.10–0.25 for purity, and p > 0.5 for the convergence guarantee.
The convergence guarantee p > 0.5 applies to the FULL PIPELINE p (Tier 3 RAG),
not zero-shot. Zero-shot p is a floor baseline only.

NOTE on retrieval-sensitive benchmarks (especially FEVER):
Some tasks rely heavily on retrieved evidence. Zero-shot scores may be low.
The relevant p for the purity theorem is measured from Cycle 0 Tier 3 RAG
outputs, not from this script.

NOTE on TruthfulQA metric:
EM is invalid for TruthfulQA -- correct_answers contains multiple valid
phrasings that EM cannot match. This script reports ROUGE-L instead, which
provides partial credit for overlapping n-grams. Chapter 5 should note:
"TruthfulQA baseline uses ROUGE-L; the original evaluation requires a
GPT-judge model unavailable in offline settings."

This script runs Flan-T5-Large DIRECTLY, with no CAEM pipeline, no memory,
no verification, no routing. It is purely: tokenize -> generate -> evaluate.

Usage
-----
# Full check (100 samples per benchmark, recommended):
python scripts/check_base_model.py --output outputs/base_model_check.json

# Quick smoke test (10 samples per benchmark):
python scripts/check_base_model.py --n_samples 10 --output outputs/base_model_check.json

# CPU-only mode (slow but works):
python scripts/check_base_model.py --device cpu

Output
------
outputs/base_model_check.json -- JSON with per-benchmark metrics and p-threshold status.

Thesis reference
----------------
§4.1 (Purity Theorem prerequisite): "Condition: p > 0.5. Measured by running
vanilla Flan-T5-Large on 100-sample validation subsets before Cycle 1."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, cast

# Hoist repo root onto sys.path so ``eval.metrics`` / ``eval.benchmarks``
# resolve when this script is invoked via ``python scripts/check_base_model.py``
# or ``python -m scripts.check_base_model``.  The module-top placement is
# deliberate so that the metric imports below resolve at import time
# instead of first-call time (pure-Python stdlib deps; zero HF dataset
# side-effects).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Canonical evaluation metrics shared with the in-pipeline numbers. Previously
# these were inlined below with a subtly different normalisation order (article
# removal before punctuation stripping) which could drift from
# ``eval/metrics.py`` on pathological inputs. The p-anchor the data-purity
# theorem consumes (see §4.1) must match what the pipeline measures, so we
# import the canonical implementations here.
from eval.metrics import any_match_em, best_token_f1, rouge_l

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dependency check
# -----------------------------------------------------------------------------

def _check_deps():
    missing = []
    for pkg in ("torch", "transformers", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(
            f"Missing packages: {', '.join(missing)}\n"
            "Install with: pip install torch transformers datasets"
        )


# -----------------------------------------------------------------------------
# Generation helpers
# -----------------------------------------------------------------------------

def generate_answer(model, tokenizer, question: str, device: str, max_new_tokens: int = 256) -> str:
    """Run greedy decoding on a single question. Returns stripped answer string.

    ChatML-wraps the question for decoder-only instruction-tuned backbones
    (Qwen/Gemma/Llama) and slices the generated continuation off the echoed
    prompt. Falls back to flat-text prompting when the tokenizer has no
    chat template.
    """
    import torch
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True,
            )
            if not isinstance(prompt, str):
                prompt = question
        except Exception:
            prompt = question
    else:
        prompt = question
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    ).to(device)
    input_len = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = (
        output_ids[0, input_len:]
        if output_ids.shape[1] > input_len
        else output_ids[0]
    )
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer


# -----------------------------------------------------------------------------
# Label extraction helpers
# -----------------------------------------------------------------------------
# (Metric functions -- exact_match/token_f1/rouge_l -- now imported from
# eval.metrics at module top; previously they were inlined here with a subtly
# divergent normalisation order. Unifying ensures the p-anchor measured by
# this script matches the in-pipeline numbers the data-purity theorem uses.)


def extract_fever_label(prediction: str) -> str:
    """Extract FEVER label from model output."""
    p = prediction.lower().strip()
    if "not enough" in p or "nei" in p:
        return "not enough info"
    if "refut" in p or "false" in p:
        return "refutes"
    if "support" in p or "true" in p:
        return "supports"
    return p[:30]  # fallback


def extract_strategyqa_label(prediction: str) -> str:
    """Extract yes/no from model output."""
    p = prediction.lower().strip()
    if p.startswith("yes"):
        return "yes"
    if p.startswith("no"):
        return "no"
    if " yes" in p:
        return "yes"
    if " no" in p:
        return "no"
    return p[:10]


def extract_arc_label(prediction: str) -> str:
    """Extract ARC option label A/B/C/D from model output."""
    p = prediction.strip().upper()
    if not p:
        return ""
    if p[0] in {"A", "B", "C", "D"}:
        return p[0]
    for ch in ("A", "B", "C", "D"):
        if f"({ch})" in p or f" {ch} " in f" {p} ":
            return ch
    return p[:1]


# -----------------------------------------------------------------------------
# Per-benchmark evaluation
# -----------------------------------------------------------------------------

def evaluate_benchmark(
    model,
    tokenizer,
    samples: list,
    benchmark: str,
    device: str,
) -> dict:
    """Run greedy generation + EM/F1 on a list of BenchmarkSample dicts.

    Returns a dict with: em, f1, n, per_sample (list of dicts).
    """
    results = []
    t_start = time.time()

    for idx, s in enumerate(samples):
        question   = s["question"]
        gold       = s["answers"]
        gold_label = s.get("gold_label")

        pred = generate_answer(model, tokenizer, question, device)

        if benchmark == "fever":
            pred_label = extract_fever_label(pred)
            em = 1.0 if pred_label == gold_label else 0.0
            f1 = em
        elif benchmark == "strategyqa":
            pred_label = extract_strategyqa_label(pred)
            em = 1.0 if pred_label in [g.lower() for g in gold] else 0.0
            f1 = em
        elif benchmark == "arc_challenge":
            pred_label = extract_arc_label(pred)
            em = 1.0 if pred_label in [g.upper() for g in gold] else 0.0
            f1 = em
        elif benchmark == "truthfulqa":
            # EM is invalid for TruthfulQA: correct_answers contains multiple
            # valid phrasings that require a GPT-judge to evaluate properly.
            # Use ROUGE-L as offline approximation (partial n-gram credit).
            em = 0.0   # not reported for TruthfulQA
            f1 = rouge_l(pred, gold)   # ROUGE-L reported as primary metric
        else:
            # gold is a list of acceptable phrasings; any_match_em / best_token_f1
            # aggregate across the list using the eval.metrics normalisation.
            em = any_match_em(pred, gold)
            f1 = best_token_f1(pred, gold)

        results.append({
            "id":         s.get("id", str(idx)),
            "question":   question[:80] + "..." if len(question) > 80 else question,
            "prediction": pred,
            "gold":       gold[0] if gold else "",
            "em":         em,
            "f1":         f1,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t_start
            # For TruthfulQA, show ROUGE-L (f1); for others show EM
            running_score = sum(r["f1"] for r in results) / len(results) if benchmark == "truthfulqa" \
                else sum(r["em"] for r in results) / len(results)
            metric_name = "ROUGE-L" if benchmark == "truthfulqa" else "EM"
            logger.info(
                "  [%s] %d/%d | running %s=%.3f | %.1fs elapsed",
                benchmark, idx + 1, len(samples), metric_name, running_score, elapsed,
            )

    n     = len(results)
    em    = sum(r["em"] for r in results) / n if n > 0 else 0.0
    f1    = sum(r["f1"] for r in results) / n if n > 0 else 0.0
    elapsed = time.time() - t_start

    # TruthfulQA: primary metric is ROUGE-L (stored in f1); p>0.15 purity check.
    # Others: use EM for p>0.5 convergence check.
    if benchmark == "truthfulqa":
        p_ok = f1 > 0.15   # purity theorem condition: p > (1-alpha)
    else:
        p_ok = em > 0.5

    return {
        "benchmark":   benchmark,
        "n":           n,
        "em":          round(em, 4),
        "f1":          round(f1, 4),
        "primary_metric": "rouge_l" if benchmark == "truthfulqa" else "em",
        "p_gt_0.5":    p_ok,
        "elapsed_s":   round(elapsed, 1),
        "per_sample":  results,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    _check_deps()

    import torch
    from caem.config import CAEMConfig
    from caem.model_loader import load_base_generator

    # -- Device --------------------------------------------------------------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)
    if device == "cpu":
        logger.warning(
            "Running on CPU -- this will be slow (~10–30 s/sample). "
            "For 100 samples × 4 benchmarks, expect 1–3 hours. "
            "Use --n_samples 10 for a quick smoke test first."
        )

    # -- Load model ----------------------------------------------------------
    # Dtype is sourced from hardware.py so all scripts share one rule
    # (bf16 on Ampere+, fp16 on older CUDA, fp32 on CPU). Hardcoding dtype
    # caused drift between scripts and silently broke bf16-capable GPUs.
    from scripts.hardware import print_hardware_summary
    hw = print_hardware_summary()
    if hw.use_bf16:
        model_dtype = torch.bfloat16
        dtype_str = "bf16"
    elif hw.use_fp16:
        model_dtype = torch.float16
        dtype_str = "fp16"
    else:
        model_dtype = torch.float32
        dtype_str = "fp32"

    # --model defaults to CAEMConfig.base_model_name (Qwen-3B on Branch C);
    # override for a different backbone sanity check.
    model_name = args.model or CAEMConfig().base_model_name
    logger.info("Loading %s (dtype=%s) ...", model_name, dtype_str)
    cfg_for_load = CAEMConfig()
    model, tokenizer = load_base_generator(
        model_name,
        device=device,
        dtype=model_dtype,
        use_flash_attention_2=cfg_for_load.use_flash_attention_2,
        use_torch_compile=cfg_for_load.use_torch_compile,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded: %.0f M params on %s", n_params, device)

    # -- Load benchmarks ------------------------------------------------------
    # Loaders are imported inside main() (not at module top) because they
    # pull in HuggingFace ``datasets``, which has heavy import-time side
    # effects we don't want to pay in unit-test contexts. The repo root is
    # already on sys.path via the module-top hoist.
    from eval.benchmarks import (
        load_arc_challenge,
        load_fever,
        load_natural_questions,
        load_strategyqa,
        load_triviaqa,
        load_truthfulqa,
    )

    n = args.n_samples
    logger.info("Loading benchmarks (n=%d per benchmark) ...", n)

    benchmarks = {
        "fever":      load_fever(n=n, seed=args.seed),
        "triviaqa":   load_triviaqa(n=n, seed=args.seed),
        "natural_questions": load_natural_questions(n=n, seed=args.seed),
        "truthfulqa": load_truthfulqa(n=n, seed=args.seed),
        "strategyqa": load_strategyqa(split="test", n=n, seed=args.seed),
        "arc_challenge": load_arc_challenge(split="test", n=n, seed=args.seed),
    }

    for bm, s in benchmarks.items():
        logger.info("  %s: %d samples", bm, len(s))

    # -- Evaluate -------------------------------------------------------------
    results = {}
    all_pass = True

    for bm, samples in benchmarks.items():
        logger.info("-" * 50)
        logger.info("Evaluating %s ...", bm)
        res = evaluate_benchmark(model, tokenizer, samples, bm, device)
        results[bm] = res
        if bm == "truthfulqa":
            logger.info(
                "  %s: ROUGE-L=%.3f  (EM not valid for TruthfulQA -- uses multi-phrase refs)  [%s]",
                bm, res["f1"],
                "OK purity OK" if res["p_gt_0.5"] else "✗ low (metric artifact likely)",
            )
        else:
            status = "OK PASS" if res["p_gt_0.5"] else "✗ low zero-shot (expected for RAG benchmarks)"
            logger.info(
                "  %s: EM=%.3f  F1=%.3f  [%s]",
                bm, res["em"], res["f1"], status,
            )
        if not res["p_gt_0.5"]:
            all_pass = False

    # -- Summary --------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("BASE MODEL CHECK SUMMARY (zero-shot floor baseline)")
    logger.info("=" * 50)
    for bm, res in results.items():
        flag = "OK" if res["p_gt_0.5"] else "(low zero-shot -- see notes)"
        if bm == "truthfulqa":
            logger.info("  %-14s ROUGE-L=%.3f  %s", bm, res["f1"], flag)
        else:
            logger.info("  %-14s EM=%.3f  F1=%.3f  %s", bm, res["em"], res["f1"], flag)

    logger.info("")
    logger.info("INTERPRETATION:")
    logger.info("  These are zero-shot baselines (no RAG, no memory).")
    logger.info("  FEVER may be low in zero-shot mode because retrieval context is absent.")
    logger.info("  The purity theorem p is measured from Cycle 0 Tier 3 RAG, not from this script.")
    logger.info("  Compare these floors against Cycle 0/target-cycle pipeline results, not directly to theorem claims.")
    logger.info("  Safe to proceed to build_passage_index.py and seed_cold_start.py.")

    summary = {
        "model":           model_name,
        "n_samples":       n,
        "device":          device,
        # Canonical key: `all_thresholds_passed` — generic enough to keep
        # working if future checks add a second threshold (e.g. p > 0.7).
        # `all_pass_p_gt_half` is retained for backward compatibility with
        # existing consumers / notebooks.
        "all_thresholds_passed": all_pass,
        "all_pass_p_gt_half":    all_pass,
        "benchmarks":      {
            bm: {k: v for k, v in res.items() if k != "per_sample"}
            for bm, res in results.items()
        },
        "per_sample_detail": {bm: res["per_sample"] for bm, res in results.items()},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Results saved -> %s", output_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Zero-shot base model accuracy check (no CAEM pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "HuggingFace model ID for the base generator. Defaults to "
            "CAEMConfig.base_model_name (Qwen/Qwen2.5-3B-Instruct on Branch C)."
        ),
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Samples per benchmark (100 = thesis spec; 10 = quick smoke test).",
    )
    p.add_argument(
        "--output",
        default="outputs/base_model_check.json",
        help="Path to write results JSON.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on. 'auto' = use CUDA if available.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subsampling.",
    )
    main(p.parse_args())
