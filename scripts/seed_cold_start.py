"""
scripts/seed_cold_start.py
==========================
Cold-start memory seeder.

Purpose
-------
Before Cycle 1 begins, seed the episodic memory store with 200–500 verified
episodes per benchmark drawn from each benchmark's TRAINING split (never from
the dev/test split used for evaluation). Without seeding:
  - Cycle 1 starts with empty memory -> ~100% of queries route to Tier 3
  - Very few episodes get stored in Cycle 1 (model is still weak)
  - Fine-tuning after Cycle 1 gets insufficient verified data
  - The improvement trajectory from Cycle 1->2->3 is artificially flat

With seeding:
    - Memory has a reliable nucleus of 600–1,500 correct episodes at Cycle 1 start
  - A fraction (~10–20%) of Cycle 1 queries hit Tier 1 (non-trivial routing)
  - Fine-tuning after Cycle 1 has richer verified data -> stronger Cycle 2 baseline

Thesis reference: §4.5 (Cold-Start Seeding):
"Initial memory is seeded with 200–500 manually verified episodes per
benchmark to model realistic deployment conditions, then self-improvement is
measured from that starting point."

How it works
------------
1. Load training-split questions for each benchmark (never the eval split).
2. Run the CAEM Tier 3 RAG path (model generation + verification).
   Verified answers are stored with u_stored derived from the verification
   pipeline (same nine-signal / six-weight composite used in production:
     u_stored = 0.30·p_ground_mean
              + 0.15·p_ground_atomic
              + 0.15·s_avg
              + 0.10·(1 − h_norm)
              + 0.15·u_internal
              + 0.15·p_entail
   where u_internal = 0.5·u_token + 0.5·(1 − u_dropout).  p_contra feeds the
   decision tree as a separate veto rather than into the composite.  See
   caem/verification/verifier.py and §4.4 of the thesis for the canonical
   definition).
3. Stop each benchmark when target_episodes verified episodes are collected
   OR when max_questions questions have been processed (whichever first).
4. Save the populated memory store to outputs/cold_start_memory/ for later
   loading by the main experiment.

Usage
-----
# Standard seeding (200 episodes per benchmark, ~2–4 hours on RTX 3060):
python scripts/seed_cold_start.py

# More episodes (500 per benchmark, better Cycle 1 kick-start):
python scripts/seed_cold_start.py --target_episodes 500

# Quick smoke test (10 per benchmark, synthetic data):
python scripts/seed_cold_start.py --smoke_test

# Save to a specific path:
python scripts/seed_cold_start.py --output_dir outputs/cold_start_memory

Note on "human annotator" text in the thesis plan
--------------------------------------------------
The plan §4.5 says "human annotator verifies correctness against official
ground truth" for cold-start seeding. This script replaces the manual
annotation with the automated nine-signal UnifiedVerifier (internal
calibration + sample-set agreement + external grounding + NLI contradiction).
The substitution is justified on the grounds that the automated verifier
attains a sufficiently high balanced accuracy at the storage threshold to
approximate human labelling for seeding purposes; the exact balanced accuracy
alpha is MEASURED in-repo by ``scripts/run_purity_validation.py`` and
reported in Chapter 5 Section 5.5 (data-purity theorem). The thesis must
not cite a numeric value here until that measurement is available --
earlier drafts contained an unsourced "~85-90%" estimate which has been
removed pending the empirical number. Recommended Chapter 5 phrasing once
alpha is measured:

  "Cold-start seeding used the automated verification pipeline
  (Section 4.4) rather than human annotation, for scalability. The
  verifier's measured balanced accuracy on labelled validation data is
  alpha = <value> (Section 5.5), which together with ground-truth labels
  drawn from the training split approximates the human-annotator
  protocol the purity theorem assumes."
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dependency check
# -----------------------------------------------------------------------------

def _check_deps():
    missing = []
    for pkg in ("torch", "transformers", "datasets", "faiss", "sentence_transformers"):
        try:
            if pkg == "faiss":
                try:
                    import faiss
                except ImportError:
                    import faiss_cpu  # type: ignore
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(
            f"Missing packages: {', '.join(missing)}\n"
            "Install with: pip install torch transformers datasets faiss-cpu sentence-transformers"
        )


# -----------------------------------------------------------------------------
# Training-split loaders
# -----------------------------------------------------------------------------

def load_train_samples(benchmark: str, n: int, seed: int = 0) -> List[dict]:
    """Load training-split questions for cold-start seeding.

    Uses separate splits from the evaluation splits to prevent contamination:
      - FEVER train (145K samples) -> sample n
      - TriviaQA train (rc.nocontext train split) -> sample n
      - Natural Questions train (nq_open train split) -> sample n
      - TruthfulQA / StrategyQA: no usable training split -> memory starts empty.

    Parameters
    ----------
    benchmark : str
    n : int -- max training samples to load
    seed : int

    Returns
    -------
    list of BenchmarkSample dicts (same schema as eval/benchmarks.py)
    """
    # Benchmark-specific imports (random, datasets.load_dataset) are no longer
    # needed at this scope: every branch now delegates to eval/benchmarks.py
    # loaders, which import them internally.

    if benchmark == "truthfulqa":
        # TruthfulQA has no training split. Skip seeding.
        logger.info(
            "TruthfulQA: no training split available -- skipping cold-start seeding. "
            "Memory starts empty for TruthfulQA. State in Chapter 5."
        )
        return []

    elif benchmark in ("fever", "triviaqa", "natural_questions"):
        # Branch C 2026-04-22 evening: the cold-start seed pool MUST be drawn
        # from the SAME benchmark_splits allocation Step 7.0+ will use, so
        # that seed/purity/calib/train-chunks remain content-hash disjoint.
        # Previously we used load_benchmark(split="train", n=n) with its own
        # shuffle seed, which produced a DIFFERENT 1000-sample subset than
        # benchmark_splits.py's first-200 seed pool → expected ~1-2 sample
        # overlap leakage between cold-start memory and later purity/calib
        # pools. Aligning here eliminates that channel entirely.
        from caem.benchmark_splits import (
            build_benchmark_pools, DEFAULT_SEED_SIZE,
        )
        logger.info(
            "Loading %s seed pool from canonical benchmark_splits (DEFAULT_SEED_SIZE=%d, "
            "matches Step 7.0 allocation) ...",
            benchmark, DEFAULT_SEED_SIZE,
        )
        pools = build_benchmark_pools(
            benchmark,
            # Pool sizes use module defaults so this script and Step 7.0 /
            # Step 7 main produce BYTE-IDENTICAL seed/purity/calib/train/eval
            # pools. Do not override; changes must be made in
            # caem/benchmark_splits.py defaults.
            rng_seed=seed,
        )
        samples = list(pools.seed)
        # Namespace-separate cold-start IDs from eval-time IDs so that
        # sample dedup by `id` (where still used) won't confuse them.
        for s in samples:
            s["id"] = f"{s.get('id', 'unk')}_seed"
        return samples

    elif benchmark == "asqa":
        # Branch C 2026-04-22: ASQA is transfer-only. Its 4353 train samples
        # are structurally too small for stream-mode cold-start seeding, and
        # seeding would contaminate transfer-eval with training distribution
        # signal. This branch is kept for back-compat (explicit
        # --benchmarks asqa) but emits a loud warning and returns an empty list.
        logger.warning(
            "ASQA is Branch C transfer-only and should NOT be seeded into "
            "cold-start memory. Returning [] so downstream Step 6 assertions "
            "proceed on the other seeded benchmarks only."
        )
        return []

    elif benchmark == "strategyqa":
        # EXP-08 fix: wics/strategy-qa uses a legacy script, load from GitHub instead.
        # StrategyQA uses a single unified dataset with 2061 samples. Since we already
        # split these for eval/purity/calib, seeding from it would cause a massive
        # data leak (Purity Validation would evaluate on seeded data).
        logger.info(
            "StrategyQA: no separate training split available -- skipping cold-start seeding "
            "to prevent data leakage into Purity Validation. Memory starts empty."
        )
        return []

    else:
        logger.warning("Unknown benchmark '%s' -- skipping.", benchmark)
        return []


# -----------------------------------------------------------------------------
# Pipeline builder (minimal -- Tier 3 only, no routing needed for seeding)
# -----------------------------------------------------------------------------

def build_pipeline(config, device: str):
    """Build a CAEMPipeline for seeding.

    Mirrors run_experiment.py::build_pipeline() exactly. The pipeline
    handles all sub-component construction internally -- we only need to
    load the base model, tokenizer, SBERT encoder, and optionally NLI.
    """
    import torch
    from caem.memory.encoder import QueryEncoder
    from caem.model_loader import load_base_generator
    from caem.retrieval.rag import PassageStore
    from caem.pipeline import CAEMPipeline

    # Hardware-driven dtype so seeding matches the main experiment's precision
    # policy (bf16 on Ampere+/5090, fp16 on older GPUs, fp32 on CPU).  Picking
    # dtype locally would diverge from run_experiment.py and shift u_stored.
    from scripts.hardware import print_hardware_summary
    hw = print_hardware_summary()
    if hw.use_bf16:
        model_dtype = torch.bfloat16
        dtype_str = "bf16"
    elif hw.use_fp16:
        model_dtype = torch.float16
        dtype_str = "fp16"
    else:
        model_dtype = torch.float32
        dtype_str = "fp32"

    # -- Base generator (Qwen-2.5-3B-Instruct by default) -------------- #
    logger.info(
        "Loading %s for seeding (dtype=%s) ...",
        config.base_model_name, dtype_str,
    )
    model, tokenizer = load_base_generator(
        config.base_model_name,
        device=device,
        dtype=model_dtype,
        use_flash_attention_2=config.use_flash_attention_2,
        use_torch_compile=config.use_torch_compile,
    )
    logger.info(
        "Base generator loaded (%.0f M params, %s).",
        sum(p.numel() for p in model.parameters()) / 1e6,
        dtype_str,
    )

    # -- SBERT encoder --------------------------------------------------- #
    encoder = QueryEncoder(model_name=config.sbert_model)

    # -- Verifier judge (optional -- improves verification quality) ----- #
    # Single source of truth via caem.verification.load_verifier_judge();
    # driven by CAEMConfig.verifier_backend (default "minicheck").
    from caem.verification import load_verifier_judge
    judge, nli_model, nli_tokenizer = None, None, None
    try:
        logger.info(
            "Loading verifier judge (backend=%s) for seeding verification ...",
            config.verifier_backend,
        )
        judge, nli_model, nli_tokenizer = load_verifier_judge(
            config, device, allow_fallback=True,
        )
    except Exception as exc:
        logger.warning(
            "Verifier judge load failed (%s) -- seeding uses SC+SE only.  "
            "Downstream u_stored will not include p_entail, so stored "
            "episodes will have lower verifier confidence than production.",
            exc,
        )

    # -- Wikipedia passage index (optional) ----------------------------- #
    passage_store = None
    idx_path = Path("data/passage_index")
    if idx_path.exists():
        try:
            passage_store = PassageStore.load(str(idx_path))
            logger.info("Passage index loaded (%d passages).", passage_store.size)
        except Exception as exc:
            logger.warning("Could not load passage index (%s). Tier 3 runs query-only.", exc)
    else:
        logger.warning(
            "Passage index not found at data/passage_index/. "
            "Run scripts/build_passage_index.py first for best results. "
            "Continuing with query-only RAG (no retrieved context)."
        )

    # -- Cross-encoder (passage reranker + Goal-2 q_a_relevance) --------- #
    # Same CrossEncoder used for BOTH passage rerank AND q_a_relevance
    # inside UnifiedVerifier. See run_experiment.py::build_pipeline for
    # the equivalent load -- kept in sync across both entry points so
    # seeded memory carries the same q_a_relevance distribution the main
    # run expects.
    cross_encoder = None
    if getattr(config, "cross_encoder_model", None):
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder %s ...", config.cross_encoder_model)
            cross_encoder = CrossEncoder(config.cross_encoder_model, device=device)
            logger.info(
                "Cross-encoder loaded -- passage rerank + "
                "Goal-2 q_a_relevance scoring.",
            )
        except Exception as exc:
            logger.warning(
                "Cross-encoder load failed (%s); q_a_relevance defaults to "
                "the 0.5 neutral prior (degrades Goal-2 sample-2 closure).",
                exc,
            )

    # -- CAEMPipeline ---------------------------------------------------- #
    pipeline = CAEMPipeline(
        model=model,
        tokenizer=tokenizer,
        encoder=encoder,
        judge=judge,
        nli_model=nli_model,
        nli_tokenizer=nli_tokenizer,
        passage_store=passage_store,
        cross_encoder=cross_encoder,
        config=config,
        current_cycle=0,
    )
    return pipeline


# -----------------------------------------------------------------------------
# Seeding loop
# -----------------------------------------------------------------------------

def seed_benchmark(
    pipeline,
    samples: List[dict],
    benchmark: str,
    target_episodes: int,
    max_questions: int,
    *,
    batch_pipeline=None,
    batch_size: int = 8,
) -> Dict:
    """Run the pipeline on training samples and collect verified episodes.

    Branch C Goal 5 integration (2026-04-21)
    ----------------------------------------
    When ``batch_pipeline`` is supplied (a ``caem.pipeline_batch.BatchPipeline``
    wrapper over the serial pipeline), queries are processed in chunks of
    ``batch_size`` via ``batch_pipeline.answer_batch`` instead of the
    single-query loop. This lights up Tier-2 / Tier-3 generation +
    verifier M-chain + K-sample pooling across the batch dim -- typical
    ~2-3x speedup on a 5090 at bs=8.

    Commit ordering is preserved by the BatchPipeline contract (writes to
    ``memory_store`` happen in submitted-sample order within each batch),
    so the seeded store is observationally equivalent to the serial path.

    Stops when ``target_episodes`` is reached or all samples are exhausted.
    Returns stats: ``{verified, stored, processed, elapsed_s}``.
    """
    if not samples:
        logger.info("  %s: no seed samples -- skipping.", benchmark)
        return {"verified": 0, "stored": 0, "processed": 0, "elapsed_s": 0}

    t0 = time.time()
    n_verified = 0
    n_processed = 0
    to_process = samples[:max_questions]

    # Batched path: feed BatchPipeline chunks; checkpoint target check
    # between chunks so we never process more than ~batch_size extra
    # queries past the target.
    if batch_pipeline is not None:
        from caem.pipeline_batch import BatchSample
        i = 0
        while i < len(to_process):
            if n_verified >= target_episodes:
                break
            chunk = to_process[i : i + batch_size]
            i += len(chunk)

            batch_inputs = [
                BatchSample(
                    query=s["question"],
                    source_benchmark=benchmark,
                    store_to_memory=True,
                )
                for s in chunk
            ]
            results = batch_pipeline.answer_batch(batch_inputs)
            for res in results:
                n_processed += 1
                if res.stored:
                    n_verified += 1

            elapsed = time.time() - t0
            logger.info(
                "  [%s seed] processed=%d | verified+stored=%d / %d target | %.0fs "
                "(batched, bs=%d)",
                benchmark, n_processed, n_verified, target_episodes,
                elapsed, len(chunk),
            )
    else:
        # Legacy serial path (kept for debug + CPU smokes without batch deps).
        for s in to_process:
            if n_verified >= target_episodes:
                break
            result = pipeline.answer(s["question"], source_benchmark=benchmark)
            n_processed += 1
            if result.stored:
                n_verified += 1
            if n_processed % 20 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  [%s seed] processed=%d | verified+stored=%d / %d target | %.0fs",
                    benchmark, n_processed, n_verified, target_episodes, elapsed,
                )

    elapsed = time.time() - t0
    logger.info(
        "  %s seeding done: %d/%d questions -> %d verified episodes stored (%.0fs)",
        benchmark, n_processed, len(to_process), n_verified, elapsed,
    )
    return {
        "verified": n_verified,
        "stored":   n_verified,
        "processed": n_processed,
        "elapsed_s": round(elapsed, 1),
    }


# -----------------------------------------------------------------------------
# Smoke test (no real model, synthetic data)
# -----------------------------------------------------------------------------

def run_smoke_test(args: argparse.Namespace) -> None:
    """Validate seeder logic with synthetic data -- no model or GPU required."""
    # sys and pathlib.Path are already imported at module top — the former
    # duplicate `import sys` / `from pathlib import Path` inside this function
    # was dead code.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from eval.benchmarks import make_synthetic_samples

    logger.info("SMOKE TEST MODE -- synthetic data, no model loaded.")
    benchmarks = args.benchmarks or ["fever", "triviaqa", "natural_questions"]
    for bm in benchmarks:
        samples = make_synthetic_samples(bm, n=10)
        logger.info("  %s: %d synthetic samples OK", bm, len(samples))

    import json
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "smoke_test",
        "note": "No model run -- synthetic data only. "
                "Re-run without --smoke_test for actual seeding.",
        "benchmarks": {bm: {"verified": 0, "status": "skipped_smoke_test"}
                       for bm in benchmarks},
    }
    out_path = output_dir / "seed_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Smoke test complete. Summary -> %s", out_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    if args.smoke_test:
        run_smoke_test(args)
        return

    _check_deps()

    import torch, json
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # -- Resolve repo root ----------------------------------------------------
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from caem.config import CAEMConfig
    config = CAEMConfig()
    if getattr(args, "cold_start_store_threshold", None) is not None:
        logger.info(
            "Overriding store_threshold for cold-start: %.3f -> %.3f",
            config.store_threshold, args.cold_start_store_threshold,
        )
        config.store_threshold = float(args.cold_start_store_threshold)

    # -- Build pipeline -------------------------------------------------------
    pipeline = build_pipeline(config, device)

    # -- Goal-5 batched wrapper (2-3x speedup on the seeder loop) ------------
    batch_pipeline = None
    batch_size = int(getattr(args, "batch_size", 8) or 8)
    if batch_size > 1:
        try:
            from caem.pipeline_batch import BatchPipeline
            batch_pipeline = BatchPipeline(pipeline)
            logger.info(
                "Goal 5: wrapping CAEMPipeline with BatchPipeline (bs=%d) "
                "for the seeder loop.",
                batch_size,
            )
        except Exception as exc:
            logger.warning(
                "BatchPipeline wrap failed (%s); falling back to serial "
                "pipeline.answer().", exc,
            )

    # -- Seed each benchmark --------------------------------------------------
    benchmarks = args.benchmarks
    target = args.target_episodes
    max_q  = args.max_questions

    logger.info("=" * 60)
    logger.info(
        "COLD-START SEEDING  (target=%d eps/benchmark, bs=%d%s)",
        target, batch_size,
        " [serial fallback]" if batch_pipeline is None else "",
    )
    logger.info("=" * 60)

    summary_stats: Dict[str, dict] = {}

    for bm in benchmarks:
        logger.info("-" * 50)
        logger.info("Seeding %s ...", bm)
        train_samples = load_train_samples(bm, n=max_q, seed=args.seed)
        stats = seed_benchmark(
            pipeline, train_samples, bm, target, max_q,
            batch_pipeline=batch_pipeline, batch_size=batch_size,
        )
        summary_stats[bm] = stats

    # -- Save memory store ----------------------------------------------------
    # store_path is a FILE base path (not a directory) — .save() appends
    # .faiss and .meta automatically. Do NOT mkdir() on it.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "memory_store"  # -> cold_start_memory/memory_store.faiss

    # Guard against saving an empty store: an empty FAISS index on disk
    # would quietly make every downstream Tier-3 lookup miss, and the
    # thesis's cold-start protocol would still technically "succeed".
    # Fail loudly so the operator notices and re-runs with working loaders.
    if pipeline.memory_store.size == 0:
        logger.error(
            "Memory store is empty after seeding all %d benchmark(s). "
            "Refusing to save an empty .faiss / .meta pair. Check the "
            "per-benchmark loader logs above for a silent dataset failure.",
            len(benchmarks),
        )
        sys.exit(1)

    pipeline.memory_store.save(str(store_path))
    logger.info("Memory store saved -> %s (.faiss + .meta)", store_path)

    # size is a @property on EpisodicMemoryStore -- not callable
    total_seeded = pipeline.memory_store.size
    logger.info("Total episodes in store after seeding: %d", total_seeded)

    summary = {
        "target_per_benchmark": target,
        "total_seeded":         total_seeded,
        "memory_store_path":    str(store_path) + ".faiss  (+ .meta)",
        "benchmarks":           summary_stats,
    }
    out_path = output_dir / "seed_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Seed summary -> %s", out_path)
    logger.info(
        "\nNext step: load this memory store at the start of the main experiment:\n"
        "  pipeline.memory_store.load('%s')\n"
        "Then run: python scripts/run_experiment.py --n_questions 5000",
        store_path,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Cold-start memory seeder for CAEM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--target_episodes",
        type=int,
        default=200,
        help="Target verified episodes per benchmark (200 = thesis spec minimum, 500 = preferred).",
    )
    p.add_argument(
        "--max_questions",
        type=int,
        default=1000,
        help="Max training questions to process per benchmark before stopping.",
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=["fever", "triviaqa", "natural_questions"],
        help="Benchmarks to seed.",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/cold_start_memory",
        help="Directory to save the seeded memory store and summary.",
    )
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run with synthetic data only (no model, no GPU). Validates logic.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for training-split sampling.",
    )
    p.add_argument(
        "--cold_start_store_threshold",
        type=float,
        default=None,
        help=(
            "Override CAEMConfig.store_threshold for cold-start seeding only. "
            "The default 0.65 is tuned against RoBERTa-MNLI's P(entail) "
            "distribution; MiniCheck's stricter composite rarely clears 0.65 "
            "pre-calibration, so cold-start on the untouched threshold "
            "yields zero STORE. Recommended default for MiniCheck cold-start: "
            "0.45 (the DEFERRED-band bar). Cold-start entries feed retrieval "
            "only, not training (which still uses tau_train=0.75), so the "
            "looser admission bar does not compromise the thesis guarantees."
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "Goal 5 batch size for BatchPipeline.answer_batch. Set to 1 to "
            "fall back to the serial pipeline.answer() loop (useful for "
            "debugging; ~2-3x slower on a 5090). Commit ordering within "
            "each batch matches the serial path."
        ),
    )
    main(p.parse_args())
