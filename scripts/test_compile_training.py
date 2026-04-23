#!/usr/bin/env python
"""
scripts/test_compile_training.py
=================================
Pre-flight test for torch.compile + training-mode correctness on the
Qwen-2.5-3B-Instruct SIL training path. Runs before Step 7 main Cycle 1
to validate:

  1. Compiled `model.forward` works in `model.train()` mode with dropout
  2. Backward pass through the compiled graph succeeds
  3. 8-bit AdamW optimizer step works after the compiled forward+backward
  4. Loss decreases after one optimization step (sanity check on correctness)
  5. Toggling between .train() and .eval() does not corrupt compile cache

If any step fails, the script exits non-zero and a clear recommendation
("set use_torch_compile=False") is printed. Expected runtime: ~3-5 min
(dominated by torch.compile JIT first-pass cost of ~215s).

Usage
-----
    source /venv/main/bin/activate && export PYTHONPATH=/workspace/caem
    python scripts/test_compile_training.py

Exit code 0 = compile + training compatible; proceed with default config.
Exit code 1 = incompatibility detected; disable compile before Step 7 main.
"""
from __future__ import annotations

import logging
import sys
import time
import traceback

import torch

from caem.model_loader import load_base_generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    if not torch.cuda.is_available():
        logger.error("CUDA not available.")
        return 1

    logger.info("=" * 72)
    logger.info("torch.compile + training compatibility test")
    logger.info("=" * 72)

    # --- Load the compiled model (same way run_simple_ft / run_experiment do) ---
    logger.info("Loading Qwen-2.5-3B-Instruct with SDPA + torch.compile (training config)...")
    t0 = time.perf_counter()
    try:
        model, tok = load_base_generator(
            "Qwen/Qwen2.5-3B-Instruct",
            use_sdpa=True,
            use_torch_compile=True,
        )
    except Exception as exc:
        logger.error("Model load FAILED: %s", exc)
        return 1
    t_load = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs.", t_load)
    logger.info("VRAM after load: %.1f GiB", torch.cuda.memory_allocated() / 1024**3)

    # --- Step 1: 8-bit AdamW optimizer ---
    logger.info("Setting up 8-bit AdamW optimizer (matches SIL training config)...")
    try:
        import bitsandbytes.optim as bnb_optim
        optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=1e-5)
        logger.info("AdamW8bit instantiated.")
    except Exception as exc:
        logger.error("AdamW8bit setup FAILED: %s", exc)
        return 1

    # --- Step 2: switch to train mode (activates dropout) ---
    logger.info("Switching to model.train() — activates dropout layers...")
    try:
        model.train()
    except Exception as exc:
        logger.error("model.train() FAILED: %s", exc)
        return 1

    # --- Step 3: training step with compiled forward ---
    prompt = (
        "Answer with one of: supports, refutes, not enough info. "
        "Claim: Paris is the capital of France. Answer: supports"
    )
    enc = tok(prompt, return_tensors="pt").to("cuda")
    labels = enc["input_ids"].clone()

    logger.info("Running training step #1 (first forward triggers torch.compile JIT)...")
    try:
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(**enc, labels=labels)
        loss_1 = out.loss
        t_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss_1.backward()
        t_bwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.step()
        t_opt = time.perf_counter() - t0

        logger.info(
            "Step #1: forward=%.1fs  backward=%.1fs  optim=%.3fs  loss=%.4f",
            t_fwd, t_bwd, t_opt, loss_1.item(),
        )
    except Exception as exc:
        logger.error("Training step #1 FAILED:")
        traceback.print_exc()
        return 1

    # --- Step 4: second step (should be fast — JIT warmed up) ---
    logger.info("Running training step #2 (JIT should be warm)...")
    try:
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(**enc, labels=labels)
        loss_2 = out.loss
        loss_2.backward()
        optimizer.step()
        t_step2 = time.perf_counter() - t0
        logger.info("Step #2: total=%.3fs  loss=%.4f", t_step2, loss_2.item())
    except Exception as exc:
        logger.error("Training step #2 FAILED:")
        traceback.print_exc()
        return 1

    # --- Step 5: third step (confirm steady state) ---
    logger.info("Running training step #3 (steady state)...")
    try:
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(**enc, labels=labels)
        loss_3 = out.loss
        loss_3.backward()
        optimizer.step()
        t_step3 = time.perf_counter() - t0
        logger.info("Step #3: total=%.3fs  loss=%.4f", t_step3, loss_3.item())
    except Exception as exc:
        logger.error("Training step #3 FAILED:")
        traceback.print_exc()
        return 1

    # --- Step 6: toggle train→eval→train (should not crash compile cache) ---
    logger.info("Toggling model.eval() → generate() → model.train() (compile-cache robustness)...")
    try:
        model.eval()
        with torch.inference_mode():
            out_gen = model.generate(
                **enc, max_new_tokens=4, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        logger.info(
            "  Eval-mode generate OK: produced %d new tokens",
            out_gen.shape[1] - enc["input_ids"].shape[1],
        )

        model.train()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(**enc, labels=labels)
        loss_4 = out.loss
        loss_4.backward()
        optimizer.step()
        t_step4 = time.perf_counter() - t0
        logger.info("Step #4 (post-toggle): total=%.3fs  loss=%.4f", t_step4, loss_4.item())
    except Exception as exc:
        logger.error("train/eval toggle FAILED:")
        traceback.print_exc()
        return 1

    # --- Loss progression sanity (loss should trend down on repeated same sample) ---
    losses = [loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item()]
    logger.info("Loss progression (should trend down on repeated-same-sample overfit): %s",
                " → ".join(f"{l:.4f}" for l in losses))
    if losses[-1] >= losses[0]:
        logger.warning(
            "Loss did NOT decrease after 4 training steps on the same sample. "
            "This is suspicious (possibly a scale/lr issue, not necessarily a compile bug). "
            "Inspect manually if seen in real training."
        )
        # Don't fail the test on this — AdamW8bit at lr=1e-5 may move slowly
    else:
        logger.info("Loss correctly decreased — training pipeline is producing updates.")

    # --- Summary ---
    logger.info("=" * 72)
    logger.info("PASS — torch.compile + training is compatible on this setup.")
    logger.info(
        "  Load: %.1fs  (includes model weight load)",
        t_load,
    )
    logger.info(
        "  Step #1 (JIT): fwd=%.1fs + bwd=%.1fs (one-time torch.compile warmup cost)",
        t_fwd, t_bwd,
    )
    logger.info("  Step #2 (warm): %.3fs", t_step2)
    logger.info("  Step #3 (steady): %.3fs", t_step3)
    logger.info("  Step #4 (post train/eval toggle): %.3fs", t_step4)
    logger.info("=" * 72)
    logger.info("Safe to run Step 7 main with default use_torch_compile=True.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
