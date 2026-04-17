"""
scripts/run_baseline.py
=======================
Run an external baseline from eval/baselines.py over the CAEM benchmark panel.

Supported baselines (--baseline):
  zero_shot   B1   Flan-T5-Large, no CoT, no retrieval.
  cot         B2   Chain-of-Thought prompting.
  rag         B3   DPR top-k retrieval + Flan-T5-Large.
  cot_rag     B4   RAG with CoT prefix.
  flare       B5   FLARE active retrieval (Jiang et al. EMNLP 2023).

Training baselines (Vanilla FT, EWC-only FT) are handled by their own
scripts; this runner is inference-only.

Output
------
outputs/baselines/<baseline>/<benchmark>_cycle0.json
  Per-sample + aggregate record, schema-compatible with eval/harness.py output
  so eval/reporting.py can consume it without changes.

Usage
-----
Full baseline across the seven-benchmark panel (vast.ai / A100):

    python -m scripts.run_baseline \\
        --baseline rag \\
        --output_dir outputs/baselines \\
        --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \\
        --n_questions 500 \\
        --passage_index outputs/wiki_passage_index

Smoke test (CPU-only, tiny N):

    python -m scripts.run_baseline \\
        --baseline zero_shot \\
        --benchmarks fever triviaqa \\
        --n_questions 10 \\
        --device cpu

Notes
-----
- The RAG-family baselines (rag, cot_rag, flare) require a pre-built Wikipedia
  passage index (scripts/build_passage_index.py).
- The same model is loaded for every benchmark in a single invocation, so
  running multiple benchmarks in one call is faster than running one at a
  time. Loop over --baseline externally if you want multiple baselines in
  one pass.

Thesis reference
----------------
  §5.1 Baselines -- external-competitor panel.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_baseline")


# -----------------------------------------------------------------------------
# CLI                                                                           #
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--baseline",
        required=True,
        choices=["zero_shot", "cot", "rag", "cot_rag", "flare"],
        help="Which external baseline to run.",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/baselines",
        help="Root directory for baseline eval JSONs. "
             "Results are written to <output_dir>/<baseline>/<bench>_cycle0.json.",
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "fever", "triviaqa", "natural_questions",
            "truthfulqa", "strategyqa", "arc_challenge",
        ],
        help="Benchmarks to evaluate on.",
    )
    p.add_argument(
        "--n_questions",
        type=int,
        default=500,
        help="Samples per benchmark (use 10-20 for smoke tests).",
    )
    p.add_argument(
        "--split",
        default="paper_dev",
        help="Benchmark split (paper_dev is data-leakage-safe for FEVER).",
    )
    p.add_argument(
        "--model_name",
        default="google/flan-t5-large",
        help="HF model ID. Keep at flan-t5-large for the Chapter 5 panel.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model weight dtype (bfloat16 matches CAEM full run on 4090/A100).",
    )
    p.add_argument(
        "--passage_index",
        default="outputs/wiki_passage_index",
        help="Path to PassageStore.save() output; required for rag/cot_rag/flare.",
    )
    p.add_argument(
        "--flare_theta",
        type=float,
        default=0.4,
        help="FLARE confidence floor (Jiang et al. 2023 §4.2).",
    )
    p.add_argument(
        "--flare_look_ahead",
        type=int,
        default=64,
        help="FLARE look-ahead horizon in tokens.",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Log progress every N samples.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Factory                                                                       #
# -----------------------------------------------------------------------------

def _build_baseline(ns: argparse.Namespace):
    """Construct the selected baseline and return it."""
    import torch

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[ns.dtype]

    from eval.baselines import (
        ZeroShotBaseline, CoTBaseline,
        RAGBaseline, CoTRAGBaseline, FLAREBaseline,
    )

    name = ns.baseline

    if name == "zero_shot":
        return ZeroShotBaseline(model_name=ns.model_name, device=ns.device, dtype=dtype)
    if name == "cot":
        return CoTBaseline(model_name=ns.model_name, device=ns.device, dtype=dtype)

    # Retrieval-augmented baselines: need a loaded passage store.
    from caem.retrieval.rag import PassageStore
    index_path = Path(ns.passage_index)
    if not index_path.exists():
        logger.error(
            "Passage index not found at %s. Build it first with "
            "scripts/build_passage_index.py before running --baseline %s.",
            index_path, name,
        )
        sys.exit(2)
    logger.info("Loading passage index from %s ...", index_path)
    passage_store = PassageStore.load(str(index_path))
    logger.info("Passage store loaded: %d passages.", len(passage_store.passages))

    if name == "rag":
        return RAGBaseline(
            passage_store=passage_store,
            model_name=ns.model_name, device=ns.device, dtype=dtype,
        )
    if name == "cot_rag":
        return CoTRAGBaseline(
            passage_store=passage_store,
            model_name=ns.model_name, device=ns.device, dtype=dtype,
        )
    # argparse choices=BASELINE_NAMES enforces that name is one of the
    # five handled cases above, so no terminal branch is needed here.
    assert name == "flare", name
    return FLAREBaseline(
        passage_store=passage_store,
        model_name=ns.model_name, device=ns.device, dtype=dtype,
        theta=ns.flare_theta,
        look_ahead_tokens=ns.flare_look_ahead,
    )


# -----------------------------------------------------------------------------
# Main                                                                          #
# -----------------------------------------------------------------------------

def main() -> None:
    ns = _parse_args()

    from eval.benchmarks import load_benchmark
    from eval.harness import EvalHarness

    output_dir = Path(ns.output_dir) / ns.baseline
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing baseline results to %s", output_dir)

    baseline = _build_baseline(ns)

    harness = EvalHarness(
        pipeline=baseline,
        output_dir=str(output_dir),
        log_every=ns.log_every,
        fail_on_error=False,
    )

    summaries: Dict[str, Dict] = {}
    for bench in ns.benchmarks:
        logger.info("=" * 72)
        logger.info("BASELINE=%s | BENCHMARK=%s | N=%d",
                    ns.baseline, bench, ns.n_questions)
        logger.info("=" * 72)
        # FEVER is the only benchmark with a paper_dev split; others fall
        # back to the default split that load_benchmark picks for that task.
        if bench == "fever":
            samples = load_benchmark(bench, n=ns.n_questions, split=ns.split)
        else:
            samples = load_benchmark(bench, n=ns.n_questions)
        agg = harness.run(benchmark=bench, samples=samples, cycle=0,
                          store_to_memory=False)
        summaries[bench] = {
            "em": agg.get("em"),
            "f1": agg.get("f1"),
            "n":  agg.get("n"),
        }

    logger.info("=" * 72)
    logger.info("BASELINE=%s summary:", ns.baseline)
    for bench, s in summaries.items():
        logger.info("  %-20s  EM=%.4f  F1=%.4f  N=%d",
                    bench, s["em"] or 0.0, s["f1"] or 0.0, s["n"] or 0)
    logger.info("All baseline runs complete. Outputs under %s", output_dir)


if __name__ == "__main__":
    main()
