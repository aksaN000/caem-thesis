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
Full baseline across the six-benchmark panel (3 ID + 3 OOD; vast.ai / A100):

    python -m scripts.run_baseline \\
        --baseline rag \\
        --output_dir outputs/baselines \\
        --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \\
        --n_questions 500 \\
        --passage_index data/passage_index

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
        choices=["zero_shot", "cot", "fiveshot_cot", "rag", "cot_rag", "flare"],
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
        "--eval_batch_size",
        type=int,
        default=1,
        help=(
            "Goal 5 Level B eval batch size. bs=1 preserves serial per-query "
            "path. bs=8-32 uses the baseline's answer_batch (zero_shot / cot / "
            "rag / cot_rag gain ~1.5-2x; FLARE falls back to serial due to "
            "iterative look-ahead decoding)."
        ),
    )
    p.add_argument(
        "--eval_prefetch",
        action="store_true",
        help="Forwarded to EvalHarness (no-op for baseline paths — baselines "
             "have native answer_batch so prefetch wrapper isn't used).",
    )
    p.add_argument(
        "--split",
        default="dev",
        help="Benchmark split (dev is data-leakage-safe for FEVER; "
             "renamed from paper_dev on HuggingFace).",
    )
    p.add_argument(
        "--model_name",
        default=None,
        help=(
            "HF model ID for the decoder-only base generator. Defaults to "
            "CAEMConfig.base_model_name (Qwen/Qwen2.5-3B-Instruct on Branch C). "
            "Override to run the Chapter 5 panel against a different backbone."
        ),
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help=(
            "Model weight dtype. 'auto' (default) routes through "
            "scripts/hardware.py so the baseline matches the CAEM full run "
            "(bf16 on Ampere+, fp16 on older CUDA, fp32 on CPU)."
        ),
    )
    p.add_argument(
        "--passage_index",
        default="data/passage_index",
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
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Global seed for Python's random, NumPy, and PyTorch. Applied "
            "at main() entry so baseline runs are reproducible under the "
            "same --n_questions / --split / --model_name. Default 42 matches "
            "run_experiment.py."
        ),
    )
    p.add_argument(
        "--caem_splits_path",
        default="",
        help=(
            "Optional path to the CAEM main-run's dataset_splits.json "
            "(e.g. outputs/full_run/dataset_splits.json). When supplied, "
            "the baseline's per-benchmark sample list is filtered to the "
            "exact eval_ids CAEM used, guaranteeing the sig-test paired "
            "overlap is the full eval slice regardless of load_benchmark "
            "determinism. Mirrors the pattern in run_purity_validation.py. "
            "Empty string disables filtering (smoke runs)."
        ),
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
    if ns.dtype == "auto":
        # Consult hardware.py so baselines and the CAEM main run agree on
        # precision. Hardcoding bf16 silently broke fp16-only GPUs.
        from scripts.hardware import print_hardware_summary
        hw = print_hardware_summary()
        if hw.use_bf16:
            dtype = torch.bfloat16
        elif hw.use_fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = dtype_map[ns.dtype]

    from eval.baselines import (
        ZeroShotBaseline, CoTBaseline, FiveShotCoTBaseline,
        RAGBaseline, CoTRAGBaseline, FLAREBaseline,
    )

    name = ns.baseline

    if name == "zero_shot":
        return ZeroShotBaseline(model_name=ns.model_name, device=ns.device, dtype=dtype)
    if name == "cot":
        return CoTBaseline(model_name=ns.model_name, device=ns.device, dtype=dtype)
    if name == "fiveshot_cot":
        # Demos drawn from the FIRST benchmark's training split (ns.benchmarks[0]).
        # The same 5 demos are reused across all eval benchmarks in this run so the
        # in-context examples are constant — matching Wei et al. 2022's convention
        # of a fixed demo block for the whole eval panel.
        from eval.benchmarks import load_benchmark
        demo_bench = ns.benchmarks[0] if ns.benchmarks else "fever"
        try:
            demo_pool = load_benchmark(
                demo_bench,
                n=50,  # small pool; demos sampled from first 50 with seed 42
                split="train" if demo_bench == "fever" else None,
            )
        except Exception as exc:
            logger.warning(
                "FiveShotCoTBaseline: failed to load demos from %s (%s); "
                "falling back to zero-shot-CoT behaviour.", demo_bench, exc,
            )
            demo_pool = []
        return FiveShotCoTBaseline(
            model_name=ns.model_name, device=ns.device, dtype=dtype,
            demo_samples=demo_pool, demo_seed=42,
        )

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

def _seed_everything(seed: int) -> None:
    """Seed Python / NumPy / PyTorch RNGs. Mirrors run_experiment.py's
    seeding so FEVER/TriviaQA slicing and sampled decoding are reproducible.
    """
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main() -> None:
    ns = _parse_args()

    _seed_everything(ns.seed)
    logger.info("Seeded all RNGs with seed=%d", ns.seed)

    import json as _json
    from eval.benchmarks import load_benchmark
    from eval.harness import EvalHarness

    output_dir = Path(ns.output_dir) / ns.baseline
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing baseline results to %s", output_dir)

    # Optional CAEM-split filter (belt-and-suspenders for sig-test pairing).
    # When --caem_splits_path is set AND the file exists, we restrict each
    # loaded benchmark's samples to the eval_ids CAEM actually evaluated on.
    # Same pattern as run_purity_validation.py::load_and_filter.
    eval_ids_by_bm: Dict[str, set] = {}
    if ns.caem_splits_path:
        splits_path = Path(ns.caem_splits_path)
        if splits_path.is_file():
            try:
                with splits_path.open("r", encoding="utf-8") as f:
                    splits = _json.load(f)
                for bm, meta in splits.items():
                    if isinstance(meta, dict):
                        eval_ids_by_bm[bm] = {str(i) for i
                                              in meta.get("eval_ids", [])}
                logger.info("Loaded eval_ids from %s for %d benchmarks.",
                            splits_path, len(eval_ids_by_bm))
            except Exception as exc:
                logger.warning("Failed to parse %s (%s). Proceeding without "
                               "eval_id filter.", splits_path, exc)
        else:
            logger.warning("--caem_splits_path %s not found. Proceeding "
                           "without eval_id filter.", splits_path)

    baseline = _build_baseline(ns)

    harness = EvalHarness(
        pipeline=baseline,
        output_dir=str(output_dir),
        log_every=ns.log_every,
        fail_on_error=False,
        batch_size=getattr(ns, "eval_batch_size", 1),
        use_prefetch=getattr(ns, "eval_prefetch", False),
    )

    summaries: Dict[str, Dict] = {}
    for bench in ns.benchmarks:
        logger.info("=" * 72)
        logger.info("BASELINE=%s | BENCHMARK=%s | N=%d",
                    ns.baseline, bench, ns.n_questions)
        logger.info("=" * 72)
        # FEVER is the only benchmark with a dedicated dev split; others fall
        # back to the default split that load_benchmark picks for that task.
        if bench == "fever":
            samples = load_benchmark(bench, n=ns.n_questions, split=ns.split)
        else:
            samples = load_benchmark(bench, n=ns.n_questions)
        # Filter to CAEM's exact eval slice when splits file was loaded.
        allowed = eval_ids_by_bm.get(bench)
        if allowed:
            before = len(samples)
            samples = [s for i, s in enumerate(samples)
                       if str(s.get("id", i)) in allowed]
            logger.info("eval_id filter: %s kept %d / %d samples.",
                        bench, len(samples), before)
            if not samples:
                logger.warning("eval_id filter left 0 samples for %s; falling "
                               "back to the unfiltered load_benchmark slice.",
                               bench)
                if bench == "fever":
                    samples = load_benchmark(bench, n=ns.n_questions,
                                             split=ns.split)
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
