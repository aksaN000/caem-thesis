#!/usr/bin/env python
"""
scripts/bench_sdpa_vs_eager.py
===============================
Live-GPU benchmark + equivalence test for the cuDNN-SDPA vs eager
attention comparison on Qwen-2.5-3B-Instruct. Run this between Step 6
completion and Step 7.0 launch to confirm the landed speedup before
committing Step 7 main's budget.

Usage
-----
    source /venv/main/bin/activate && export PYTHONPATH=/workspace/caem
    python scripts/bench_sdpa_vs_eager.py

What it does
------------
1. Loads Qwen-2.5-3B-Instruct twice — once with attn_implementation=
   "eager", once with "sdpa" (which dispatches to cuDNN-attention on
   Blackwell per our _configure_sdpa_backends() setup).
2. Runs N=10 identical prompts (fixed seed, greedy decoding) through
   each model.
3. Compares:
     - Generated text: should be bit-identical or differ only by fp
       rounding noise (bf16 matmul-order differences)
     - Per-prompt latency: measures tokens/sec throughput
4. Prints a summary and exits 0 on PASS, 1 on FAIL.

Pass criteria
-------------
- Text equivalence: >=90% of (eager, sdpa) output pairs agree after
  lowercase + whitespace-normalise. (Higher would be better; bf16 can
  flip a few token-level choices at boundaries.)
- Throughput: sdpa should be >=2x eager on Qwen-3B bf16 (expect 4-6x
  per gau-nernst 5090 benchmarks).

This is NOT a pytest test — it requires a live GPU with ~10 GB VRAM
free. The unit test for the configuration logic lives at
``tests/test_sdpa_config.py``.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from caem.model_loader import _configure_sdpa_backends, caem_sdpa_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


PROMPTS = [
    "Reasoning: Consider the following claim. The Eiffel Tower was completed in 1889. Based on general knowledge, the claim is",
    "Reasoning: Photosynthesis is the process by which plants convert sunlight into energy. This process primarily occurs in the",
    "Reasoning: Water boils at 100 degrees Celsius at standard atmospheric pressure. At higher altitudes where pressure is lower, water boils at",
    "Reasoning: The author of Hamlet and Romeo and Juliet is widely recognized as one of the greatest playwrights in English literature. This author's name is",
    "Reasoning: Mount Everest is the tallest mountain above sea level in the world. Its height is approximately",
    "Reasoning: The chemical symbol for gold is derived from the Latin word 'aurum'. The modern chemical symbol for gold is",
    "Reasoning: The speed of light in a vacuum is a fundamental physical constant. Its approximate value in meters per second is",
    "Reasoning: The human body contains approximately 206 bones in an adult. Infants are born with more bones because many fuse over time. The approximate number of bones a newborn has is",
    "Reasoning: The Great Wall of China was built over several centuries. Most of the current wall structure was built during which dynasty?",
    "Reasoning: DNA is a double helix structure that stores genetic information. The two scientists who first described this structure are",
]


def load_and_bench(
    model_name: str,
    attn_impl: str,
    prompts: List[str],
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> Tuple[List[str], List[float]]:
    """Load with given attn_implementation; run each prompt; return (texts, latencies_s)."""
    logger.info("=" * 70)
    logger.info("Loading Qwen-3B with attn_implementation=%s", attn_impl)
    logger.info("=" * 70)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(device).eval()

    results = []
    latencies = []

    torch.manual_seed(42)
    for i, p in enumerate(prompts):
        enc = tok(p, return_tensors="pt").to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            if attn_impl == "sdpa":
                with caem_sdpa_context():
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
            else:
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append(text)
        latencies.append(elapsed)
        logger.info("  [%d/%d] prompt %2d: %.2fs, tokens=%d (%.1f tok/s)",
                    i + 1, len(prompts), i, elapsed,
                    out.shape[1] - enc["input_ids"].shape[1],
                    (out.shape[1] - enc["input_ids"].shape[1]) / elapsed)

    # Free VRAM before next load
    del model
    torch.cuda.empty_cache()
    return results, latencies


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def compare(eager_texts: List[str], sdpa_texts: List[str]) -> Tuple[int, int, List[int]]:
    """Return (match_count, total, mismatched_indices)."""
    mismatched = []
    for i, (a, b) in enumerate(zip(eager_texts, sdpa_texts)):
        if normalize(a) != normalize(b):
            mismatched.append(i)
    return len(eager_texts) - len(mismatched), len(eager_texts), mismatched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--match_threshold", type=float, default=0.90,
                        help="Min fraction of prompts where eager and sdpa "
                             "outputs match after normalisation.")
    parser.add_argument("--speedup_threshold", type=float, default=2.0,
                        help="Min required speedup ratio sdpa/eager.")
    ns = parser.parse_args()

    # Pre-configure backends so the sdpa load dispatches to cuDNN.
    _configure_sdpa_backends()

    if not torch.cuda.is_available():
        logger.error("CUDA not available — cannot run live benchmark.")
        return 1

    logger.info("GPU: %s", torch.cuda.get_device_name(0))
    logger.info("VRAM (initial): %.1f GiB used",
                torch.cuda.memory_allocated() / 1024**3)

    logger.info("Running eager baseline ...")
    eager_texts, eager_latencies = load_and_bench(
        ns.model, "eager", PROMPTS, ns.max_new_tokens,
    )

    logger.info("Running SDPA (cuDNN-attention) ...")
    sdpa_texts, sdpa_latencies = load_and_bench(
        ns.model, "sdpa", PROMPTS, ns.max_new_tokens,
    )

    # ---- Throughput ----
    eager_mean = sum(eager_latencies) / len(eager_latencies)
    sdpa_mean = sum(sdpa_latencies) / len(sdpa_latencies)
    speedup = eager_mean / sdpa_mean

    # ---- Equivalence ----
    match, total, mismatched = compare(eager_texts, sdpa_texts)
    match_frac = match / total

    logger.info("=" * 70)
    logger.info("RESULTS (n=%d prompts, max_new_tokens=%d)", total, ns.max_new_tokens)
    logger.info("=" * 70)
    logger.info("Eager  mean latency: %.3f s/prompt  (%.1f tok/s)",
                eager_mean, ns.max_new_tokens / eager_mean)
    logger.info("SDPA   mean latency: %.3f s/prompt  (%.1f tok/s)",
                sdpa_mean, ns.max_new_tokens / sdpa_mean)
    logger.info("Speedup (eager / sdpa): %.2fx   (threshold: %.1fx)",
                speedup, ns.speedup_threshold)
    logger.info("Text equivalence: %d/%d (%.1f%%)   (threshold: %.1f%%)",
                match, total, match_frac * 100, ns.match_threshold * 100)
    if mismatched:
        logger.info("Mismatched prompt indices: %s", mismatched)
        for i in mismatched[:3]:  # show first 3
            logger.info("  [prompt %d] eager:  %s", i, eager_texts[i][:100])
            logger.info("             sdpa:   %s", sdpa_texts[i][:100])

    # ---- Gate ----
    speedup_ok = speedup >= ns.speedup_threshold
    match_ok = match_frac >= ns.match_threshold

    if speedup_ok and match_ok:
        logger.info("PASS — adopt SDPA for Step 7.0 and main run")
        return 0
    logger.error("FAIL — speedup_ok=%s  match_ok=%s", speedup_ok, match_ok)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
