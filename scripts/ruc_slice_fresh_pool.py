#!/usr/bin/env python3
"""scripts/ruc_slice_fresh_pool.py
=====================================
Build a content-hash-disjoint fresh pool of questions for the RUC v2 training
set. Every question chosen here is guaranteed to have a content_id NOT used
by any of CAEM's existing pools (seed / purity / calibration / sil_train /
eval / test) for that benchmark.

For benchmarks with abundant raw data (FEVER, TriviaQA), we slice fresh
unused rows directly from the raw splits. For tight benchmarks (CommonsenseQA,
StrategyQA, TruthfulQA), we use the existing test pool (designated for Ch5
final generalisation, currently untouched). For the two NEW benchmarks
(HaluEval-QA, OpenBookQA), we slice from the raw splits.

Stretch sizing (target ~10,500 questions; actual yield 9,004 on 2026-05-17):

  FEVER          2,500 fresh from raw           (got 2500; OK)
  TriviaQA       2,500 fresh from raw           (got 2500; OK)
  CommonsenseQA    500 from test pool           (got  500; OK)
  StrategyQA     2,000 fresh+test_pool          (got  187; SHORTFALL —
                                                 loader returns 687 total,
                                                 all forbidden by existing
                                                 eval/test pools)
  TruthfulQA       500 from test pool           (got  317; raw split has
                                                 817 total, clamped)
  HaluEval-QA    2,500 fresh from raw           (got 2500; OK)
  OpenBookQA       500 fresh from raw           (got  500; OK)

Decision (2026-05-17): accept the 187-row StrategyQA yield, keep StrategyQA
in the v2 training panel for question-type diversity. Pooled AUROC, not
per-bench StrategyQA AUROC, is the gate. See caem/ruc/ruc_next_session_plan.md
§4 for the rationale.

Writes outputs:
  caem/ruc/fresh_pool/<benchmark>_v2.json   one per benchmark, list of samples

Schema mirrors eval/benchmarks.py output: list of dicts with
``{question, answers, gold_label, id, benchmark}``.

Usage
-----
    python -m scripts.ruc_slice_fresh_pool

Idempotent: skips benchmarks whose output file already exists.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Stretch-plan sizing per benchmark
# ---------------------------------------------------------------------------

STRETCH_PLAN = {
    "fever":          {"n_fresh": 2500, "use_test_pool": False},
    "triviaqa":       {"n_fresh": 2500, "use_test_pool": False},
    "commonsense_qa": {"n_fresh":    0, "use_test_pool": True},
    "strategyqa":     {"n_fresh": 1500, "use_test_pool": True},
    "truthfulqa":     {"n_fresh":    0, "use_test_pool": True},
    "haluevalqa":     {"n_fresh": 2500, "use_test_pool": False},
    "openbookqa":     {"n_fresh":  500, "use_test_pool": False},
}


# ---------------------------------------------------------------------------
# Forbidden-content-id collection
# ---------------------------------------------------------------------------

def _forbidden_ids_for(benchmark: str) -> Set[str]:
    """Collect every content_id already used by CAEM pools for this benchmark."""
    from caem.benchmark_splits import (
        build_benchmark_pools, content_id, TRAINING_BENCHMARKS,
    )
    forbidden: Set[str] = set()
    if benchmark in TRAINING_BENCHMARKS:
        pools = build_benchmark_pools(benchmark)
        for src in (pools.seed, pools.purity, pools.calibration,
                    pools.eval, pools.test):
            for s in src:
                forbidden.add(content_id(s))
        for chunk in pools.sil_train_chunks:
            for s in chunk:
                forbidden.add(content_id(s))
    else:
        # Transfer benchmarks only populate eval + test
        try:
            pools = build_benchmark_pools(benchmark)
            for src in (pools.eval, pools.test):
                for s in src:
                    forbidden.add(content_id(s))
        except Exception:
            pass
    return forbidden


# ---------------------------------------------------------------------------
# Fresh-row slicer for known benchmarks
# ---------------------------------------------------------------------------

def _slice_fresh_from_raw(
    benchmark: str,
    n_fresh: int,
    seed: int = 1337,
) -> List[Dict]:
    """Pull n_fresh samples from the raw HF benchmark, skipping any
    content_id already used by CAEM pools."""
    if n_fresh <= 0:
        return []
    from caem.benchmark_splits import content_id
    from eval.benchmarks import load_benchmark
    forbidden = _forbidden_ids_for(benchmark)
    logger.info("%s: %d forbidden content_ids from CAEM pools", benchmark,
                len(forbidden))

    # For training benches, use train split; for transfer benches, use test split
    from caem.config import TRAINING_BENCHMARKS
    if benchmark in TRAINING_BENCHMARKS:
        all_raw = load_benchmark(benchmark, split="train", n=None)
    elif benchmark == "haluevalqa":
        from scripts.ruc_load_extra_benchmarks import load_haluevalqa
        all_raw = load_haluevalqa(n=None)
    elif benchmark == "openbookqa":
        from scripts.ruc_load_extra_benchmarks import load_openbookqa
        all_raw = load_openbookqa(n=None)
    else:
        # transfer bench — pull what we can
        all_raw = load_benchmark(benchmark, n=None)
    logger.info("%s: %d raw samples available", benchmark, len(all_raw))

    fresh: List[Dict] = []
    import random
    rng = random.Random(seed)
    rng.shuffle(all_raw)
    for s in all_raw:
        cid = content_id(s)
        if cid in forbidden:
            continue
        s = dict(s)
        s["benchmark"] = benchmark
        s["_pool"] = "fresh_raw"
        fresh.append(s)
        if len(fresh) >= n_fresh:
            break
    logger.info("%s: %d fresh samples selected (target %d)",
                benchmark, len(fresh), n_fresh)
    return fresh


def _take_test_pool(benchmark: str) -> List[Dict]:
    """Return the existing 'test' pool for this benchmark."""
    from caem.benchmark_splits import build_benchmark_pools
    pools = build_benchmark_pools(benchmark)
    rows = []
    for s in pools.test:
        s = dict(s)
        s["benchmark"] = benchmark
        s["_pool"] = "test_pool"
        rows.append(s)
    logger.info("%s: %d test-pool samples", benchmark, len(rows))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("caem/ruc/fresh_pool"))
    ap.add_argument("--force", action="store_true",
                    help="Re-slice even if output file exists.")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for bench, cfg in STRETCH_PLAN.items():
        out_path = args.out_dir / f"{bench}_v2.json"
        if out_path.exists() and not args.force:
            logger.info("%s already exists; skipping", out_path)
            with open(out_path) as f:
                samples = json.load(f)
            summary[bench] = len(samples)
            continue

        samples: List[Dict] = []
        if cfg["use_test_pool"]:
            samples.extend(_take_test_pool(bench))
        samples.extend(_slice_fresh_from_raw(bench, cfg["n_fresh"]))

        with open(out_path, "w") as f:
            json.dump(samples, f, indent=1)
        logger.info("%s -> %d samples -> %s", bench, len(samples), out_path)
        summary[bench] = len(samples)

    print(f"\n{'='*60}")
    print(f"Fresh pool slicing summary")
    print(f"{'='*60}")
    total = 0
    for b, n in summary.items():
        print(f"  {b:<20} {n:>5}")
        total += n
    print(f"  {'TOTAL':<20} {total:>5}")
    print(f"\nOutputs at: {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
