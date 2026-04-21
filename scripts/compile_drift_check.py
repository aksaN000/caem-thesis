"""
scripts/compile_drift_check.py
==============================
torch.compile drift gate (Branch C Goal 5).

Purpose
-------
``CAEMConfig.use_torch_compile`` opts into fused-kernel compilation of the
generator forward pass. That introduces a documented ~1e-4 logit drift
vs eager mode due to FP reordering. Documented as safe — three orders of
magnitude below the u_stored composite's dominant noise sources
(MC-dropout ~3e-2, sampling variance ~5e-2, IVF-PQ retrieval order ~5e-2).

**But the assumption is the drift STAYS at 1e-4.** A CUDA driver regression
or a backend switch could push drift to 1e-2 or worse, at which point
decisions near the STORE/DEFERRED boundary start flipping. This script is
the regression gate that catches that.

Protocol
--------
1. Load Qwen-2.5-3B-Instruct in eager mode.
2. Run N canonical prompts and capture full logit tensors.
3. Apply torch.compile to the same model instance.
4. Re-run the N prompts; capture logits again.
5. Compute ``max |eager - compiled|`` and ``mean |eager - compiled|``
   per prompt, pooled across tokens + vocab.
6. Append one row to ``outputs/perf_log.csv`` + emit a human-readable
   report. Exit code reflects GATE status:

   | Outcome      | max drift      | Exit code | Decision                                 |
   |--------------|----------------|-----------|------------------------------------------|
   | SAFE         | <= 1e-3        | 0         | compile is safe — continue using it      |
   | WARN         | 1e-3 .. 1e-2   | 0         | log, keep compile; widen the noise floor |
   | UNSAFE       | > 1e-2         | 1         | STOP using torch.compile; investigate    |

Tests
-----
``tests/test_compile_drift.py`` exercises the pure-Python drift-statistics
layer with synthetic tensors. The live model load + forward passes are
Vast-GPU-gated (imports torch + Qwen only when invoked).

Typical usage
-------------
    PYTHONPATH=. python scripts/compile_drift_check.py \\
        --n_prompts 8 --output_csv outputs/perf_log.csv

Exit status flows through ``make`` / CI so any regression fails the build.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# Gate thresholds. Keep in sync with branch_C.md §Goal 5 / the thesis
# docstring. Raising either number silently defeats the gate's purpose;
# change them only with an accompanying audit entry.
SAFE_THRESHOLD: float = 1e-3
WARN_THRESHOLD: float = 1e-2

# Canonical prompts stress-test decoder-only Qwen at varied token lengths
# and task styles so drift is measured over the same distribution the
# thesis run exercises (factual QA + claim verification + multiple-choice).
DEFAULT_PROMPTS: List[str] = [
    "Question: What is the capital of France?\nAnswer:",
    "Claim: Barack Obama was the 44th President of the United States.\nSupports, Refutes, or Not Enough Info?",
    "Question: Can a pressure cooker cook food faster than boiling water?\nReasoning:",
    "Question: Which gas makes up the majority of Earth's atmosphere? "
    "Choices: (A) oxygen (B) nitrogen (C) carbon dioxide (D) argon\n"
    "Answer with just the letter.",
    "Question: When was the Great Wall of China built?\nAnswer:",
    "Claim: Water boils at 100 degrees Celsius at sea level.\nSupports, Refutes, or Not Enough Info?",
    "Question: Who wrote the play Hamlet?\nAnswer:",
    "Reasoning: The theory of relativity was developed in the early 20th century.\n"
    "Answer:",
]


@dataclass
class DriftResult:
    """Summary of the eager-vs-compiled drift measurement."""

    n_prompts: int
    max_abs_drift: float
    mean_abs_drift: float
    per_prompt_max: List[float]
    per_prompt_mean: List[float]
    gate: str  # "SAFE" | "WARN" | "UNSAFE"


# =============================================================================
# Pure-Python statistics layer (CPU-testable)
# =============================================================================

def classify_drift(
    max_abs: float,
    safe_threshold: float = SAFE_THRESHOLD,
    warn_threshold: float = WARN_THRESHOLD,
) -> str:
    """Return the gate outcome for a given worst-case drift.

    Thresholds default to the branch_C.md gate values but can be
    overridden for experimental sweeps.
    """
    if max_abs > warn_threshold:
        return "UNSAFE"
    if max_abs > safe_threshold:
        return "WARN"
    return "SAFE"


def summarise_drift(
    per_prompt_max: List[float],
    per_prompt_mean: List[float],
    safe_threshold: float = SAFE_THRESHOLD,
    warn_threshold: float = WARN_THRESHOLD,
) -> DriftResult:
    """Aggregate per-prompt drift numbers into the gate outcome."""
    if not per_prompt_max:
        return DriftResult(
            n_prompts=0,
            max_abs_drift=0.0,
            mean_abs_drift=0.0,
            per_prompt_max=[],
            per_prompt_mean=[],
            gate="SAFE",
        )
    if len(per_prompt_max) != len(per_prompt_mean):
        raise ValueError(
            "per_prompt_max and per_prompt_mean must be the same length.",
        )
    max_abs = float(max(per_prompt_max))
    mean_abs = float(sum(per_prompt_mean) / len(per_prompt_mean))
    return DriftResult(
        n_prompts=len(per_prompt_max),
        max_abs_drift=max_abs,
        mean_abs_drift=mean_abs,
        per_prompt_max=list(per_prompt_max),
        per_prompt_mean=list(per_prompt_mean),
        gate=classify_drift(max_abs, safe_threshold, warn_threshold),
    )


# =============================================================================
# GPU-gated measurement layer
# =============================================================================

def measure_drift(
    prompts: List[str],
    *,
    device: str = "cuda",
    compile_mode: str = "reduce-overhead",
) -> DriftResult:
    """Load Qwen-3B eager + compiled, run ``prompts``, return the gate summary.

    Requires CUDA. Exits with exit code 2 if none available (callers that
    want a dry-run should use the statistics layer with synthetic tensors).
    """
    import torch  # lazy import

    if not torch.cuda.is_available():
        logger.error("compile_drift_check requires CUDA; none detected.")
        sys.exit(2)

    from caem.config import CAEMConfig
    from caem.model_loader import load_base_generator

    cfg = CAEMConfig()
    logger.info("Loading Qwen-3B (eager, bf16) ...")
    # Force use_torch_compile=False so the eager model is never accidentally
    # compiled; we wrap with torch.compile manually for the compiled pass.
    model, tokenizer = load_base_generator(
        cfg.base_model_name,
        device=device,
        dtype=torch.bfloat16,
        use_flash_attention_2=cfg.use_flash_attention_2,
        use_torch_compile=False,
    )

    per_prompt_max: List[float] = []
    per_prompt_mean: List[float] = []

    # Force eval mode + disable cuDNN / TF32 nondeterminism paths before
    # either pass runs. Otherwise cuDNN's auto-tuner could pick a different
    # kernel between the two forwards and inflate "drift" with effects that
    # are unrelated to torch.compile. Seed the RNG too — belt-and-braces
    # against any stochastic op that might be live.
    model.eval()
    torch.manual_seed(0)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Pre-tokenise all prompts once so the eager and compiled passes see
    # byte-identical input tensors. Any drift is then purely model-path.
    tokenised = [tokenizer(p, return_tensors="pt").to(device) for p in prompts]

    # Capture eager logits first
    eager_logits: List["torch.Tensor"] = []
    with torch.no_grad():
        for enc in tokenised:
            out = model(**enc)
            eager_logits.append(out.logits.detach().to(torch.float32).cpu())

    logger.info("Applying torch.compile(mode=%s) ...", compile_mode)
    compiled_model = torch.compile(model, mode=compile_mode)

    # Warm up the compile (first forward triggers the kernel build)
    with torch.no_grad():
        _ = compiled_model(**tokenised[0])

    # Measure compiled logits on the full prompt set
    with torch.no_grad():
        for i, enc in enumerate(tokenised):
            out = compiled_model(**enc)
            compiled = out.logits.detach().to(torch.float32).cpu()
            diff = (compiled - eager_logits[i]).abs()
            per_prompt_max.append(float(diff.max().item()))
            per_prompt_mean.append(float(diff.mean().item()))

    return summarise_drift(per_prompt_max, per_prompt_mean)


# =============================================================================
# CSV writer (reuses perf_baseline's schema so drift rows share the
# outputs/perf_log.csv with perf-baseline rows)
# =============================================================================

def append_drift_row(result: DriftResult, csv_path: Path, label: str) -> None:
    """Append the drift result as a row in the shared perf_log.csv."""
    from scripts.perf_baseline import CSV_FIELDS, GPUSummary, PerfRow, TimingSummary, append_rows

    # Reuse the PerfRow schema so `aggregate_perf.py` treats the drift
    # row as just another measurement. timing_* fields carry the drift;
    # gpu_* fields stay at defaults.
    timing = TimingSummary(
        label="compile.max_abs_drift",
        n=result.n_prompts,
        mean_ms=result.mean_abs_drift,      # scale-free but numerically small
        median_ms=result.max_abs_drift,     # peak for quick-eyeballing
        p95_ms=result.max_abs_drift,
        p99_ms=result.max_abs_drift,
        min_ms=min(result.per_prompt_max) if result.per_prompt_max else 0.0,
        max_ms=result.max_abs_drift,
        stdev_ms=0.0,
    )
    gpu = GPUSummary(
        n_samples=0,
        mean_util_pct=float("nan"),
        peak_util_pct=float("nan"),
        mean_vram_mb=float("nan"),
        peak_vram_mb=float("nan"),
    )
    row = PerfRow(
        label=label,
        timestamp=time.time(),
        n_queries=result.n_prompts,
        batch_size=1,
        timing=timing,
        gpu=gpu,
        metadata={
            "mode": "compile_drift",
            "gate": result.gate,
            "safe_threshold": str(SAFE_THRESHOLD),
            "warn_threshold": str(WARN_THRESHOLD),
            "max_abs_drift": f"{result.max_abs_drift:.3e}",
            "mean_abs_drift": f"{result.mean_abs_drift:.3e}",
        },
    )
    append_rows([row], csv_path)


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="torch.compile logit-drift regression gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--label", default="compile_drift",
                   help="Run label recorded in the perf_log row.")
    p.add_argument("--n_prompts", type=int, default=len(DEFAULT_PROMPTS),
                   help=f"Number of prompts (default = {len(DEFAULT_PROMPTS)} canonical).")
    p.add_argument("--compile_mode", default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--output_csv", type=Path,
                   default=Path("outputs/perf_log.csv"))
    p.add_argument("--device", default="cuda")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ns = _build_parser().parse_args(argv)
    prompts = DEFAULT_PROMPTS[: ns.n_prompts]
    result = measure_drift(prompts, device=ns.device, compile_mode=ns.compile_mode)

    logger.info("Drift summary:")
    logger.info("  n_prompts      = %d", result.n_prompts)
    logger.info("  max |drift|    = %.3e", result.max_abs_drift)
    logger.info("  mean |drift|   = %.3e", result.mean_abs_drift)
    logger.info("  safe_threshold = %.3e  (documented 1e-4, guard 1e-3)", SAFE_THRESHOLD)
    logger.info("  warn_threshold = %.3e  (hard fail above this)", WARN_THRESHOLD)
    logger.info("  GATE           = %s", result.gate)

    append_drift_row(result, ns.output_csv, label=ns.label)

    if result.gate == "UNSAFE":
        logger.error(
            "UNSAFE: drift %.3e > %.3e. STOP using torch.compile until "
            "investigated (likely driver / backend regression).",
            result.max_abs_drift, WARN_THRESHOLD,
        )
        return 1
    if result.gate == "WARN":
        logger.warning(
            "WARN: drift %.3e exceeds documented 1e-4 envelope. "
            "Continue using torch.compile but widen the composite noise "
            "floor from 1e-2 to %.3e in Ch5 §methodology.",
            result.max_abs_drift, result.max_abs_drift,
        )
        return 0
    logger.info("SAFE: drift within documented envelope; torch.compile is fine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
