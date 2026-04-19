"""
scripts/level_b_smoke.py
=========================
Real-GPU smoke validation for the Level B BatchPipeline.

Purpose
-------
Before enabling the batched Level B path for cyclic Steps 7.1-7.10, verify
on real GPU that BatchPipeline.answer_batch produces per-sample outputs
equivalent to CAEMPipeline.answer (within bf16 tolerance) and measures
a meaningful wall-clock speedup vs the serial loop.

The test runs a small hand-crafted smoke set of 10 FEVER-shaped claims
through both paths and reports:

    * Per-sample decision agreement rate (STORE/DEFERRED/ABSTAIN/DISCARD).
    * Per-sample u_stored delta (atol=2e-2 acceptance tolerance to survive
      bf16 accumulation noise, plus sampling noise on the pooled M-chain
      and semantic-entropy draws which are T=0.7/se_temperature and
      therefore stochastic even with a fixed seed).
    * Wall-clock speedup (serial_total / batched_total).

Gating policy
-------------
Gate failure causes the script to exit with a non-zero status code so a
shell wrapper can refuse to enable the batched path. The gates:

    G1: |u_stored_batched - u_stored_serial| <= 2e-2 on every sample
        that both paths completed (empty answers excluded since Tier 2
        escalation is inherently non-deterministic across runs when the
        model sees padded vs unpadded inputs; the serial path is the
        reference and we only fail if the batched path diverges *beyond*
        stochastic noise).
    G2: decision agreement >= 80% on non-empty samples (some edge
        samples may flip decision because of the 2e-2 u_stored jitter
        near a 0.65/0.45/0.25 threshold; this is expected and noted in
        the thesis tolerance budget in §5.6.).
    G3: batched wall-clock <= 0.85 * serial wall-clock (at least 15%
        speedup -- below this the batching overhead is not worth the
        code complexity; the Level B docstring targets 1.8x which is
        ~44% speedup, so 15% is a conservative lower gate).

Invocation
----------
Runs only when RUN_LEVEL_B_SMOKE=1 is set (gated by env var to avoid
accidental GPU occupancy during other tests):

    RUN_LEVEL_B_SMOKE=1 python scripts/level_b_smoke.py

Prints a JSON summary to stdout; exits 0 on pass, 1 on any gate fail.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

# Ensure the repo root is importable when this is run directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caem.config import CAEMConfig  # noqa: E402
from caem.pipeline_batch import BatchPipeline, BatchSample  # noqa: E402
from scripts.seed_cold_start import build_pipeline  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("level_b_smoke")


# Ten hand-crafted FEVER-shaped claims covering a range of expected
# confidence bands (high-confidence true, high-confidence false, domain
# facts, date-sensitive, numerically-adversarial). Same family as the
# FLARE smoke set so the smoke setup is familiar.
SMOKE_QUERIES: List[str] = [
    "Claim: Mount Everest is the tallest mountain on Earth.",
    "Claim: The capital of Australia is Sydney.",
    "Claim: Water boils at 100 degrees Celsius at sea level.",
    "Claim: The Great Wall of China is visible from the Moon with the naked eye.",
    "Claim: Albert Einstein developed the special theory of relativity in 1905.",
    "Claim: The human body has 206 bones in adulthood.",
    "Claim: The French Revolution began in 1789.",
    "Claim: The Pacific Ocean is the largest ocean on Earth.",
    "Claim: William Shakespeare wrote Hamlet.",
    "Claim: The speed of light in a vacuum is approximately 300,000 km/s.",
]

# Acceptance tolerances. The pooled T5 samplers use do_sample=True so
# even with a fixed torch seed the per-sample chains/se-samples differ
# between serial and batched paths (different batch shapes consume the
# RNG state differently). We tolerate this by setting atol=2e-2 on
# u_stored; this matches the thesis §5.6 claimed numerical budget.
ATOL_U_STORED = 2e-2
MIN_DECISION_AGREEMENT = 0.80
MAX_BATCHED_WALLCLOCK_FRACTION = 0.85


def _run(serial_pipeline, queries: List[str]):
    """Run the serial loop; return (results, wall_clock_seconds)."""
    t0 = time.perf_counter()
    results = [
        serial_pipeline.answer(
            query=q, store_to_memory=False, source_benchmark="smoke",
        )
        for q in queries
    ]
    return results, time.perf_counter() - t0


def _run_batched(batch_pipeline: BatchPipeline, queries: List[str]):
    """Run the batched path once; return (results, wall_clock_seconds)."""
    samples = [
        BatchSample(query=q, store_to_memory=False, source_benchmark="smoke")
        for q in queries
    ]
    t0 = time.perf_counter()
    results = batch_pipeline.answer_batch(samples)
    return results, time.perf_counter() - t0


def _compare(serial_rs, batched_rs) -> dict:
    """Per-sample comparison summary."""
    n = len(serial_rs)
    assert len(batched_rs) == n, "smoke: serial/batched length mismatch"

    max_u_diff = 0.0
    decision_matches = 0
    per_sample = []
    for i, (s, b) in enumerate(zip(serial_rs, batched_rs)):
        s_u = float(s.u_stored) if s.u_stored is not None else None
        b_u = float(b.u_stored) if b.u_stored is not None else None
        u_diff = None
        if s_u is not None and b_u is not None:
            u_diff = abs(s_u - b_u)
            if u_diff > max_u_diff:
                max_u_diff = u_diff
        decision_ok = (s.decision == b.decision)
        if decision_ok:
            decision_matches += 1
        per_sample.append({
            "i": i,
            "serial_u_stored": s_u,
            "batched_u_stored": b_u,
            "u_diff": u_diff,
            "serial_decision": s.decision,
            "batched_decision": b.decision,
            "serial_tier": s.tier,
            "batched_tier": b.tier,
            "serial_answer": (s.answer or "")[:60],
            "batched_answer": (b.answer or "")[:60],
        })

    decision_agreement = decision_matches / max(n, 1)
    return {
        "n": n,
        "max_u_stored_diff": max_u_diff,
        "decision_agreement": decision_agreement,
        "per_sample": per_sample,
    }


def main() -> int:
    if os.environ.get("RUN_LEVEL_B_SMOKE", "0") != "1":
        logger.warning(
            "RUN_LEVEL_B_SMOKE != 1 -- set the env var to actually run "
            "this smoke test. Exiting without running.",
        )
        return 0

    logger.info("Building serial pipeline for Level B smoke ...")
    config = CAEMConfig()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    serial_pipeline = build_pipeline(config, device)

    batch_pipeline = BatchPipeline(serial_pipeline)

    logger.info("Running %d-query serial baseline ...", len(SMOKE_QUERIES))
    serial_rs, serial_wall = _run(serial_pipeline, SMOKE_QUERIES)
    logger.info("Serial wall-clock: %.2fs (%.2fs per sample)",
                serial_wall, serial_wall / len(SMOKE_QUERIES))

    logger.info("Running %d-query batched Level B path ...", len(SMOKE_QUERIES))
    batched_rs, batched_wall = _run_batched(batch_pipeline, SMOKE_QUERIES)
    logger.info("Batched wall-clock: %.2fs (%.2fs per sample)",
                batched_wall, batched_wall / len(SMOKE_QUERIES))

    speedup = serial_wall / batched_wall if batched_wall > 0 else float("inf")
    comparison = _compare(serial_rs, batched_rs)

    summary = {
        "n": len(SMOKE_QUERIES),
        "serial_wall_s": round(serial_wall, 3),
        "batched_wall_s": round(batched_wall, 3),
        "speedup": round(speedup, 3),
        "batched_fraction_of_serial": round(batched_wall / serial_wall, 3)
            if serial_wall > 0 else None,
        "max_u_stored_diff": round(comparison["max_u_stored_diff"], 6),
        "decision_agreement": round(comparison["decision_agreement"], 3),
        "gates": {
            "G1_u_stored_tol": comparison["max_u_stored_diff"] <= ATOL_U_STORED,
            "G2_decision_agreement": comparison["decision_agreement"]
                >= MIN_DECISION_AGREEMENT,
            "G3_speedup": (batched_wall / serial_wall)
                <= MAX_BATCHED_WALLCLOCK_FRACTION,
        },
        "per_sample": comparison["per_sample"],
    }
    print(json.dumps(summary, indent=2, default=str))

    all_pass = all(summary["gates"].values())
    if all_pass:
        logger.info("Level B smoke PASSED all gates. Batched path is safe to use.")
        return 0
    logger.error(
        "Level B smoke FAILED: %s",
        {k: v for k, v in summary["gates"].items() if not v},
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
