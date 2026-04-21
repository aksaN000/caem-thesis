"""
scripts/diff_verify_serial_vs_batch.py
=======================================
Correctness diagnostic: run the UnifiedVerifier's serial verify() path
AND its batched verify_batch() path on the same (query, answer) pairs
and diff every signal per-sample. If the two paths agree within ~1e-2
on each signal and ~1e-2 on u_stored, the deep-batched verifier is
distribution-identical to the serial path. A systematic uni-directional
delta on one signal points to the responsible pool.
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from caem.config import CAEMConfig
from scripts.seed_cold_start import build_pipeline, load_train_samples


logger = logging.getLogger(__name__)


DEFAULT_QUERIES = [
    "Answer with one of: supports, refutes, not enough info. Claim: The Eiffel Tower is in Paris, France.",
    "Answer with one of: supports, refutes, not enough info. Claim: Barack Obama was born in 1961.",
    "Answer with one of: supports, refutes, not enough info. Claim: Mount Everest is shorter than K2.",
    "Answer with one of: supports, refutes, not enough info. Claim: Python is a programming language.",
]


def fmt(x):
    if x is None:
        return "None"
    return f"{x:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4, help="Sample count.")
    parser.add_argument("--benchmark", type=str, default="fever",
                        help="Benchmark to pull samples from when n>4.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = CAEMConfig()
    cfg.store_threshold = 0.45

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Building pipeline (device=%s) ...", device)
    pipeline = build_pipeline(cfg, device)
    verifier = pipeline.verifier

    if args.n <= 4:
        queries = DEFAULT_QUERIES[:args.n]
    else:
        logger.info("Loading %d %s samples (seed=%d) ...", args.n, args.benchmark, args.seed)
        samples = load_train_samples(args.benchmark, args.n, seed=args.seed)
        queries = [s.get("query") or s.get("question") or "" for s in samples]
        queries = [q for q in queries if q]
        if len(queries) < args.n:
            logger.warning("Only %d non-empty queries loaded.", len(queries))

    # Generate deterministic Tier-3 RAG answers once per query; use them
    # as frozen inputs to both verify paths so (q, a) pairs are identical.
    logger.info("Generating deterministic answers for %d queries ...", len(queries))
    answers = []
    for q in queries:
        r = pipeline.answer(q, store_to_memory=False)
        answers.append(r.answer or "[empty]")

    pairs = list(zip(queries, answers))

    # ---- Serial path: call verify() once per sample ------------------- #
    logger.info("Running SERIAL verify() on %d samples ...", len(pairs))
    torch.manual_seed(42)
    serial_outs = []
    for q, a in pairs:
        torch.manual_seed(42)
        out = verifier.verify(q, a)
        serial_outs.append(out)

    # ---- Batched path: verify_batch() on all samples at once ---------- #
    logger.info("Running BATCHED verify_batch() on %d samples ...", len(pairs))
    torch.manual_seed(42)
    batch_outs = verifier.verify_batch(pairs)

    # ---- Per-signal diff ---------------------------------------------- #
    fields = [
        "u_token", "u_dropout", "u_internal",
        "s_avg", "h_norm", "p_entail",
        "p_ground_max", "p_ground_mean", "p_ground_atomic", "p_contra",
        "q_a_relevance", "u_stored",
    ]
    verbose = len(pairs) <= 8
    if verbose:
        print()
        print("=" * 100)
        print(f"{'sample':>6}  {'signal':18}  {'serial':>10}  {'batch':>10}  {'delta':>10}  flag")
        print("=" * 100)
    deltas_by_sig = {f: [] for f in fields}
    for i, (s, b) in enumerate(zip(serial_outs, batch_outs)):
        for f in fields:
            vs = getattr(s, f)
            vb = getattr(b, f)
            d = vb - vs
            if verbose:
                flag = "*" if abs(d) > 0.02 else ""
                print(f"  {i:>4}  {f:18}  {fmt(vs):>10}  {fmt(vb):>10}  {d:>+10.4f}  {flag}")
            deltas_by_sig[f].append(d)
        if verbose:
            sfc = len(s.atomic_facts or [])
            bfc = len(b.atomic_facts or [])
            if sfc != bfc:
                print(f"  {i:>4}  {'atomic_n_facts':18}  {sfc:>10}  {bfc:>10}  {bfc-sfc:>+10}  ** count differs")
            print()

    print("=" * 100)
    print("Per-signal summary (batch - serial across all samples):")
    print(f"  {'signal':18}  {'mean_delta':>12}  {'max_|delta|':>12}  {'sign_pattern':>24}")
    for f in fields:
        deltas = deltas_by_sig[f]
        mean_d = sum(deltas) / len(deltas) if deltas else 0.0
        max_abs = max((abs(d) for d in deltas), default=0.0)
        pos = sum(1 for d in deltas if d > 0.005)
        neg = sum(1 for d in deltas if d < -0.005)
        zero = len(deltas) - pos - neg
        patt = f"+{pos} -{neg} ~{zero}"
        flag = "  <-- SYSTEMATIC BIAS" if abs(mean_d) > 0.01 and (pos == 0 or neg == 0) else ""
        print(f"  {f:18}  {mean_d:>+12.4f}  {max_abs:>12.4f}  {patt:>24}{flag}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
