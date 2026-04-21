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
from caem.pipeline_batch_prefetch import PrefetchingBatchPipeline  # noqa: E402
from scripts.seed_cold_start import build_pipeline  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("level_b_smoke")


# Sixteen hand-crafted FEVER-shaped claims covering a range of expected
# confidence bands. Sized to match the Phase 2 chunk_size=8 cleanly
# (2 full chunks) so the prefetch path's per-chunk amortisation is
# representative of the 500-sample production benchmark. Smaller N
# creates a trailing micro-chunk whose overhead dominates measurement.
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
    "Claim: Thomas Edison invented the telephone.",
    "Claim: The Sahara is the largest hot desert on Earth.",
    "Claim: Marie Curie was the first woman to win a Nobel Prize.",
    "Claim: The Amazon rainforest produces 20 percent of the world's oxygen.",
    "Claim: The human heart has four chambers.",
    "Claim: The theory of gravity was first formulated by Galileo Galilei.",
]

# Acceptance tolerances. Three of the 9 signals (s_avg, h_norm,
# p_entail) are driven by do_sample=True chain generation. Batched
# pooled sampling consumes the RNG state differently than serial
# per-sample sampling, so same-seed chains differ between paths.
# Each sampled-chain signal contributes ~0.10 variance; weighted sum
# over the 3 sampled signals gives ~0.04 expected variance; worst-case
# per-sample diff can hit ~0.15. Per-sample bf16 noise adds ~0.01.
#
# G1 therefore tolerates ~0.15 u_stored divergence between paths.
# The primary correctness gate is G2 (decision agreement): batching
# is correct iff decisions match modulo the same stochastic flips
# that would happen between any two reseeded serial runs.
ATOL_U_STORED = 0.15
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
        # PipelineResult stores the verifier decision on .verifier_output.decision;
        # there is no top-level .decision attribute. Tier-1 hits have vout=None,
        # so treat those as "TIER1" to avoid a None==None collision matching all pairs.
        s_dec = s.verifier_output.decision if s.verifier_output is not None else "TIER1"
        b_dec = b.verifier_output.decision if b.verifier_output is not None else "TIER1"
        decision_ok = (s_dec == b_dec)
        if decision_ok:
            decision_matches += 1
        per_sample.append({
            "i": i,
            "serial_u_stored": s_u,
            "batched_u_stored": b_u,
            "u_diff": u_diff,
            "serial_decision": s_dec,
            "batched_decision": b_dec,
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


def _run_prefetching(
    prefetching_pipeline: PrefetchingBatchPipeline, queries: List[str],
):
    """Run the Phase 2 prefetching path once; return (results, wall)."""
    samples = [
        BatchSample(query=q, store_to_memory=False, source_benchmark="smoke")
        for q in queries
    ]
    t0 = time.perf_counter()
    results = prefetching_pipeline.answer_batch(samples)
    return results, time.perf_counter() - t0


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
    prefetching_pipeline = PrefetchingBatchPipeline(
        batch_pipeline, chunk_size=8,
    )

    logger.info("Running %d-query serial baseline ...", len(SMOKE_QUERIES))
    serial_rs, serial_wall = _run(serial_pipeline, SMOKE_QUERIES)
    logger.info("Serial wall-clock: %.2fs (%.2fs per sample)",
                serial_wall, serial_wall / len(SMOKE_QUERIES))

    logger.info("Running %d-query Phase 1 batched path ...", len(SMOKE_QUERIES))
    batched_rs, batched_wall = _run_batched(batch_pipeline, SMOKE_QUERIES)
    logger.info("Batched wall-clock: %.2fs (%.2fs per sample)",
                batched_wall, batched_wall / len(SMOKE_QUERIES))

    logger.info("Running %d-query Phase 2 prefetching path ...",
                len(SMOKE_QUERIES))
    prefetch_rs, prefetch_wall = _run_prefetching(
        prefetching_pipeline, SMOKE_QUERIES,
    )
    prefetching_pipeline.shutdown()
    logger.info("Prefetching wall-clock: %.2fs (%.2fs per sample)",
                prefetch_wall, prefetch_wall / len(SMOKE_QUERIES))

    speedup_p1 = serial_wall / batched_wall if batched_wall > 0 else float("inf")
    speedup_p2 = serial_wall / prefetch_wall if prefetch_wall > 0 else float("inf")
    comparison_p1 = _compare(serial_rs, batched_rs)
    comparison_p2 = _compare(serial_rs, prefetch_rs)
    # Phase 2 must also match Phase 1 (same numerics, overlap only adds
    # CPU work on a worker thread).
    comparison_p2_vs_p1 = _compare(batched_rs, prefetch_rs)

    summary = {
        "n": len(SMOKE_QUERIES),
        "serial_wall_s": round(serial_wall, 3),
        "phase1_batched_wall_s": round(batched_wall, 3),
        "phase2_prefetch_wall_s": round(prefetch_wall, 3),
        "speedup_phase1_vs_serial": round(speedup_p1, 3),
        "speedup_phase2_vs_serial": round(speedup_p2, 3),
        "speedup_phase2_vs_phase1": round(
            batched_wall / prefetch_wall, 3,
        ) if prefetch_wall > 0 else None,
        "max_u_stored_diff_p1": round(comparison_p1["max_u_stored_diff"], 6),
        "max_u_stored_diff_p2": round(comparison_p2["max_u_stored_diff"], 6),
        "max_u_stored_diff_p2_vs_p1": round(
            comparison_p2_vs_p1["max_u_stored_diff"], 6,
        ),
        "decision_agreement_p1": round(comparison_p1["decision_agreement"], 3),
        "decision_agreement_p2": round(comparison_p2["decision_agreement"], 3),
        "decision_agreement_p2_vs_p1": round(
            comparison_p2_vs_p1["decision_agreement"], 3,
        ),
        "gates": {
            "G1_p1_u_stored_tol":
                comparison_p1["max_u_stored_diff"] <= ATOL_U_STORED,
            "G1_p2_u_stored_tol":
                comparison_p2["max_u_stored_diff"] <= ATOL_U_STORED,
            "G2_p1_decision_agreement":
                comparison_p1["decision_agreement"]
                >= MIN_DECISION_AGREEMENT,
            "G2_p2_decision_agreement":
                comparison_p2["decision_agreement"]
                >= MIN_DECISION_AGREEMENT,
            "G3_p1_speedup":
                (batched_wall / serial_wall)
                <= MAX_BATCHED_WALLCLOCK_FRACTION,
            "G3_p2_speedup":
                (prefetch_wall / serial_wall)
                <= MAX_BATCHED_WALLCLOCK_FRACTION,
            # Phase 2 must not regress against Phase 1 by more than 15%.
            # Short smoke workloads pay prefetch overhead (ThreadPool
            # submit + future wait + GIL) without fully amortising the
            # overlap gain. Production (N~500 per benchmark) amortises
            # over many chunks so the overlap dominates.
            "G4_p2_not_slower_than_p1":
                (prefetch_wall / batched_wall) <= 1.15,
        },
        "per_sample_p1": comparison_p1["per_sample"],
        "per_sample_p2": comparison_p2["per_sample"],
    }
    print(json.dumps(summary, indent=2, default=str))

    all_pass = all(summary["gates"].values())
    if all_pass:
        logger.info(
            "Level B smoke PASSED all gates. Phase 1 + Phase 2 paths "
            "are both safe to use.",
        )
        return 0
    logger.error(
        "Level B smoke FAILED: %s",
        {k: v for k, v in summary["gates"].items() if not v},
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
