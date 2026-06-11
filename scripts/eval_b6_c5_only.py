#!/usr/bin/env python
"""
scripts/eval_b6_c5_only.py
==========================
Re-evaluate the B6 vanilla fine-tune at cycle 5 with the format-anchor fix
(max_new_tokens raised 256 -> 512, max_input_tokens raised 512 -> 2048).

The 2026-05-28 B6 training run wrote a 5.8 GB Qwen-2.5-3B-Instruct checkpoint
to outputs/baselines_phase1e/vanilla_ft/cycle_5/. That checkpoint is the
cumulative result of SFT(SFT(SFT(SFT(SFT(base, c1), c2), c3), c4), c5) and
is the canonical B6@C5 weight state. We do NOT need to retrain; we only need
to re-evaluate against the same panel with the larger generation cap so the
"Answer:" line is not truncated before it lands.

The 2026-05-16 matched-protocol fix raised max_new_tokens to 512 for
B1/B2/B5 but was missed on the B6 inline eval shim at
``scripts/run_simple_ft.py:988-989``. The 256-token cap there clips B6's
"Answer: <letter>" line on CommonsenseQA (gold = single letter, prediction
ends up as "Reasoning:C" which fails strict EM against "C") and similar
clips on TruthfulQA's long-form gold spans.

Outputs land under outputs/baselines_phase1e/vanilla_ft/eval_fixed/ so the
old (broken-cap) eval JSONs at .../eval/{bench}_cycle5.json are preserved
for the archive_pre_2026-06-11/ comparison diff.

Usage
-----
$ cd /workspace/caem
$ python scripts/eval_b6_c5_only.py 2>&1 | \
      tee outputs/baselines_phase1e/vanilla_ft/eval_fixed/b6_c5_eval.log

Constraints
-----------
RTX 3060 12 GB VRAM, 7.4 GB system RAM (WSL2). Loads Qwen-2.5-3B in bf16
(~6 GB VRAM peak) via low_cpu_mem_usage=True. No FAISS, no retrieval, no
sample-set chain generation. ~25-40 min wall-time for 1500 generations.

Determinism
-----------
Eval pool is rebuilt with rng_seed=42 (the canonical training seed) so the
n=300 samples per benchmark match exactly the original 2026-05-28 eval
fold. No fold drift.
"""

from __future__ import annotations

# HF_HOME must be set BEFORE any HuggingFace import touches the disk cache.
import os
os.environ.setdefault("HF_HOME", str(os.path.expanduser("~/.cache/huggingface")))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add workspace root to sys.path so `caem` / `eval` imports resolve when
# the script is launched as `python scripts/eval_b6_c5_only.py` from /workspace/caem.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from caem.benchmark_splits import build_all_benchmark_pools, ALL_BENCHMARKS
from caem.pipeline import PipelineResult
from eval.baselines import ZeroShotBaseline
from eval.harness import EvalHarness


# --------------------------------------------------------------------------- #
# Thin wrapper that accepts the ``source_benchmark`` kwarg the harness passes  #
# --------------------------------------------------------------------------- #
# ``EvalHarness._run_one`` calls ``pipeline.answer(query, store_to_memory=...,  #
# source_benchmark=benchmark)`` (eval/harness.py:379-383). The base baseline    #
# answer signature is ``answer(self, query, store_to_memory=False)`` and has no #
# **kwargs catch, so the harness call fails with                                 #
#   TypeError: BaselineBase.answer() got an unexpected keyword argument         #
#               'source_benchmark'                                              #
# The fix is to thin-wrap ZeroShotBaseline so source_benchmark is accepted and  #
# ignored, mirroring the comment at eval/harness.py:138 ("source_benchmark is   #
# ignored by baselines (no memory / routing)").                                  #
# --------------------------------------------------------------------------- #


class B6EvalBaseline(ZeroShotBaseline):
    """ZeroShotBaseline + harness-compatible answer signature."""

    def answer(self, query, store_to_memory=False, source_benchmark=None, **kwargs):
        return super().answer(query, store_to_memory=store_to_memory)

    def answer_batch(self, queries, store_to_memory=False, source_benchmark=None, **kwargs):
        return super().answer_batch(queries, store_to_memory=store_to_memory)


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

CKPT_DIR = _ROOT / "outputs" / "baselines_phase1e" / "vanilla_ft" / "cycle_5"
OUTPUT_DIR = _ROOT / "outputs" / "baselines_phase1e" / "vanilla_ft" / "eval_fixed"

# Canonical training-run parameters (must match the 2026-05-28 run for
# fold determinism).
SEED = 42
N_CYCLES = 5
EVAL_SIZE = 300
CYCLE = 5

# Matched-protocol generation budget. The bumped caps are the fix; do NOT
# lower them or the format anchor (Reasoning:/Answer:) gets truncated again.
MAX_NEW_TOKENS = 512
MAX_INPUT_TOKENS = 2048


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_b6_c5_only")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    if not CKPT_DIR.exists():
        raise FileNotFoundError(
            f"C5 checkpoint not found at {CKPT_DIR}. "
            f"Expected ~5.8 GB across model-00001-of-00002.safetensors and "
            f"model-00002-of-00002.safetensors plus tokenizer.json + config.json."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("CUDA not detected; falling back to CPU (will be slow).")

    free_vram_gb = (
        torch.cuda.mem_get_info()[0] / 1024**3 if device == "cuda" else 0.0
    )
    logger.info("Device=%s | Free VRAM=%.2f GiB", device, free_vram_gb)

    # ----------------------------------------------------------------- #
    # Load the C5 checkpoint                                             #
    # ----------------------------------------------------------------- #
    logger.info("Loading C5 checkpoint from %s ...", CKPT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded (vocab_size=%d, pad=%s).",
                len(tokenizer), tokenizer.pad_token)

    model = AutoModelForCausalLM.from_pretrained(
        str(CKPT_DIR),
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    logger.info("Model loaded onto %s in bf16 (sdpa attention).", device)

    if device == "cuda":
        post_load_free = torch.cuda.mem_get_info()[0] / 1024**3
        logger.info("Post-load free VRAM=%.2f GiB", post_load_free)

    # ----------------------------------------------------------------- #
    # Wrap in ZeroShotBaseline (inherits SYSTEM_PROMPT + FORCED_PREFIX  #
    # class attrs that the matched-protocol fix relies on)              #
    # ----------------------------------------------------------------- #
    # __new__ bypasses BaselineBase.__init__ so we don't re-load the base
    # model. We then thread in the already-loaded model + tokenizer + the
    # bumped generation caps.
    baseline = B6EvalBaseline.__new__(B6EvalBaseline)
    baseline.model_name = "Qwen/Qwen2.5-3B-Instruct"  # cosmetic label
    baseline.device = device
    baseline.max_new_tokens = MAX_NEW_TOKENS
    baseline.max_input_tokens = MAX_INPUT_TOKENS
    baseline.tokenizer = tokenizer
    baseline.model = model
    # Initialise the minimum BaselineBase state the answer() path touches.
    baseline.config = None  # never read on the answer() path; safe placeholder
    logger.info(
        "Baseline wrapped: system_prompt=%s, forced_prefix=%r, "
        "max_new_tokens=%d, max_input_tokens=%d",
        "set" if B6EvalBaseline.system_prompt else "None",
        B6EvalBaseline.forced_prefix,
        baseline.max_new_tokens,
        baseline.max_input_tokens,
    )

    # ----------------------------------------------------------------- #
    # Build benchmark pools with the canonical seed                      #
    # ----------------------------------------------------------------- #
    panel = list(ALL_BENCHMARKS)
    logger.info("Building pools for panel=%s (seed=%d, n_cycles=%d, eval_size=%d)",
                panel, SEED, N_CYCLES, EVAL_SIZE)
    pools = build_all_benchmark_pools(
        benchmarks=panel,
        n_cycles=N_CYCLES,
        eval_size=EVAL_SIZE,
        rng_seed=SEED,
    )

    # ----------------------------------------------------------------- #
    # Run EvalHarness over the 5 benches, cycle=5                        #
    # ----------------------------------------------------------------- #
    harness = EvalHarness(
        pipeline=baseline,
        output_dir=str(OUTPUT_DIR),
        log_every=50,
        fail_on_error=False,
        batch_size=1,
        use_prefetch=False,
    )

    results = {}
    for bench in panel:
        samples = list(pools[bench].eval)
        logger.info("=== %s | cycle=%d | n=%d ===", bench, CYCLE, len(samples))
        res = harness.run(benchmark=bench, samples=samples, cycle=CYCLE)
        results[bench] = res
        logger.info(
            "  %s | EM=%.4f | F1=%.4f | n=%d",
            bench, res["em"], res["f1"], res["n"],
        )

    # ----------------------------------------------------------------- #
    # Summary                                                            #
    # ----------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("B6@C5 re-eval summary (matched-protocol fix applied):")
    logger.info("  Checkpoint: %s", CKPT_DIR)
    logger.info("  Output:     %s", OUTPUT_DIR)
    logger.info("  max_new_tokens=%d, max_input_tokens=%d", MAX_NEW_TOKENS, MAX_INPUT_TOKENS)
    logger.info("=" * 60)
    pooled_em = sum(r["meta"]["em"] for r in results.values()) / len(results)
    pooled_f1 = sum(r["meta"]["f1"] for r in results.values()) / len(results)
    for bench, res in results.items():
        logger.info("  %-20s EM=%.4f  F1=%.4f", bench, res["em"], res["f1"])
    logger.info("  %-20s EM=%.4f  F1=%.4f", "POOLED (5-bench)", pooled_em, pooled_f1)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
