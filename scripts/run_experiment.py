"""
scripts/run_experiment.py
==========================
CAEM Experiment Orchestrator -- Cycle 0 -> N (default N=10)

Runs the full self-improvement experiment loop:

  Cycle 0  -- baseline evaluation (zero episodic memory, no fine-tuning)
  Calibration -- temperature scaling + signal weight fitting on 500-sample set
    Cycle 1..N -- SelfImprovementLoop -> evaluate selected benchmarks

All results are saved as JSON to outputs/eval/. Cycle checkpoints are saved
to outputs/cycle_{n}/. A summary CSV is written to outputs/experiment_summary.csv
for Chapter 5 table generation.

Usage
-----
From the repo root (requires A100 GPU, ~15-19 GB VRAM):

    python -m scripts.run_experiment \\
        --output_dir outputs \\
        --num_cycles 10 \\
        --n_questions 5000 \\
        --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge

For a smoke-test (CPU, tiny N):
    python -m scripts.run_experiment --smoke_test

Thesis reference
----------------
  §4.2  Self-improvement loop
  §5.2  Experiment setup and cycle results
  §5.3  Mechanism evidence table (five-column diagnostic)

Notes
-----
- Set HUGGINGFACE_HUB_CACHE env var to your cache dir on Colab/cluster.
- Wikipedia passage index must be pre-built via scripts/build_passage_index.py
  before the experiment can run; Tier 3 RAG depends on it.
- All projected targets are from the unified plan; actual results may differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Mapping, cast

if TYPE_CHECKING:
    from caem.pipeline import CAEMPipeline

# -- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


# -- Guard: check torch/transformers before importing heavy modules -----------
def _check_deps() -> None:
    missing = []
    for pkg in ["torch", "transformers", "sentence_transformers", "faiss"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Missing dependencies: %s", missing)
        logger.error("Install via: pip install torch transformers sentence-transformers faiss-gpu")
        sys.exit(1)


# -- Imports (after dep check) ------------------------------------------------
def _load_imports() -> Dict[str, Any]:
    """Deferred import so --help works without GPU deps installed."""
    import torch

    from caem.config import CAEMConfig
    from caem.memory.encoder import QueryEncoder
    from caem.memory.store import EpisodicMemoryStore
    from caem.model_loader import load_base_generator
    from caem.pipeline import CAEMPipeline
    from caem.retrieval.rag import PassageStore
    from caem.training.self_improvement import SelfImprovementLoop
    from eval.benchmarks import (
        load_fever,
        load_strategyqa,
        load_truthfulqa,
        load_triviaqa,
        load_natural_questions,
        load_arc_challenge,
        load_benchmark,
    )
    from eval.harness import EvalHarness

    return dict(
        torch=torch,
        load_base_generator=load_base_generator,
        CAEMConfig=CAEMConfig,
        QueryEncoder=QueryEncoder,
        EpisodicMemoryStore=EpisodicMemoryStore,
        CAEMPipeline=CAEMPipeline,
        PassageStore=PassageStore,
        SelfImprovementLoop=SelfImprovementLoop,
        load_benchmark=load_benchmark,
        EvalHarness=EvalHarness,
    )


# -----------------------------------------------------------------------------
# Model initialisation
# -----------------------------------------------------------------------------

def build_pipeline(config: Any, ns: Any, m: Dict[str, Any]) -> "CAEMPipeline":
    """Load Flan-T5-Large, SBERT encoder, NLI model, and passage store.

    Parameters
    ----------
    config : CAEMConfig
    ns     : argparse.Namespace -- parsed CLI args
    m      : dict -- module namespace from _load_imports()

    Returns
    -------
    CAEMPipeline -- ready for inference
    """
    from scripts.hardware import print_hardware_summary, apply_memory_flags
    hw = print_hardware_summary()
    apply_memory_flags(hw)
    device = hw.device
    logger.info("Device: %s | GPU: %s | VRAM: %.1f GB",
                device, hw.gpu_name or "n/a", hw.vram_gb)

    # Keep fine-tuning batch size hardware-safe for the current GPU profile.
    if getattr(config, "batch_size", 0) > hw.recommended_batch_size:
        logger.info(
            "Adjusting batch_size %d -> %d for current hardware.",
            config.batch_size,
            hw.recommended_batch_size,
        )
        config.batch_size = hw.recommended_batch_size

    # Pin the effective batch size to the thesis target (=32) via gradient
    # accumulation on whatever hardware we're running on. On the thesis 5090
    # path grad_accum_steps==1 (numerical no-op); on a 4090 fallback it
    # becomes 2, on a 3060 it becomes 8, etc. See NOTE 14 resolution in
    # docs/scripts-audit-report.md. Only overwrite when the config is using
    # the default of 1, so explicit CLI/config overrides are preserved.
    hw_grad_accum = int(getattr(hw, "grad_accum_steps", 1))
    if getattr(config, "grad_accum_steps", 1) == 1 and hw_grad_accum != 1:
        logger.info(
            "Adjusting grad_accum_steps %d -> %d for current hardware "
            "(effective batch=%d).",
            config.grad_accum_steps,
            hw_grad_accum,
            config.batch_size * hw_grad_accum,
        )
        config.grad_accum_steps = hw_grad_accum

    # -- Base generator (Qwen-2.5-3B-Instruct, decoder-only) -------------- #
    # Branch C uses a decoder-only backbone via load_base_generator. The
    # model name is sourced from CAEMConfig.base_model_name so that the CLI
    # can override via --model-name without touching this function.
    logger.info("Loading base generator: %s ...", config.base_model_name)
    torch = m["torch"]
    load_dtype = (
        torch.bfloat16 if hw.use_bf16
        else torch.float16 if hw.use_fp16
        else torch.float32
    )
    model, tokenizer = m["load_base_generator"](
        config.base_model_name,
        device=device,
        dtype=load_dtype,
        use_flash_attention_2=config.use_flash_attention_2,
        use_torch_compile=config.use_torch_compile,
    )
    logger.info("Base generator loaded (%.0f M params, precision=%s)",
                sum(p.numel() for p in model.parameters()) / 1e6,
                "bf16" if hw.use_bf16 else "fp16" if hw.use_fp16 else "fp32")

    # -- SBERT encoder ------------------------------------------------------ #
    logger.info("Loading SBERT encoder (all-mpnet-base-v2) ...")
    encoder = m["QueryEncoder"](model_name=config.sbert_model, device=device)

    # -- Verifier judge (MiniCheck or legacy NLI) -------------------------- #
    # Dispatch on CAEMConfig.verifier_backend:
    #   "minicheck"   -> MiniCheck-Flan-T5-Large (default; thesis main path)
    #   "roberta_nli" -> legacy roberta-large-mnli (kept for ablation)
    # All verifier call-sites share one source of truth via
    # caem.verification.load_verifier_judge().
    from caem.verification import load_verifier_judge
    judge, nli_model, nli_tokenizer = None, None, None
    try:
        logger.info("Loading verifier judge (backend=%s) ...",
                    config.verifier_backend)
        judge, nli_model, nli_tokenizer = load_verifier_judge(
            config, device,
            allow_fallback=getattr(ns, "smoke_test", False),
        )
        logger.info("Verifier judge loaded.")
    except Exception as exc:
        msg = (
            "VERIFIER JUDGE LOAD FAILED: %s (backend=%s)\n"
            "  >> All verifier NLI-backed signals (p_entail, p_ground_*, "
            "p_contra, semantic entropy) would degenerate to neutral defaults.\n"
            "  >> Memory quality and hallucination reduction results would be\n"
            "     SEVERELY DEGRADED and scientifically invalid.\n"
            "  >> Fix: ensure the backend model is downloadable and cache\n"
            "     has sufficient disk space (~1.5 GB for MiniCheck)."
        )
        if getattr(ns, "smoke_test", False):
            logger.warning(msg, exc, config.verifier_backend)
            logger.warning("  >> Smoke test mode: continuing without judge.")
        else:
            logger.error(msg, exc, config.verifier_backend)
            sys.exit(1)

    # -- Passage store (Wikipedia FAISS index) ------------------------------ #
    passage_store = None
    passage_index_path = Path(ns.passage_index)
    if passage_index_path.exists():
        logger.info("Loading passage index from %s ...", passage_index_path)
        passage_store = m["PassageStore"].load(str(passage_index_path))
        logger.info("Passage store loaded (%d passages).", len(passage_store.passages))
    else:
        logger.warning(
            "Passage index not found at %s -- Tier 3 RAG disabled. "
            "Build it with scripts/build_passage_index.py first.",
            passage_index_path,
        )

    # -- Cross-encoder (passage reranker + Goal-2 q_a_relevance scorer) ---- #
    # Single CrossEncoder instance used for TWO roles inside the verifier:
    #  (a) passage rerank top-20 -> top-3 before grounding NLI
    #  (b) q_a_relevance signal on (question, display_answer) pairs
    # Both tasks share the "text-pair relevance scoring" contract.
    # Co-resident with Qwen + MiniCheck + SBERT on a 32 GB 5090 fits well
    # within budget (~1 GB additional VRAM for BGE-reranker-v2-m3).
    cross_encoder = None
    if getattr(config, "cross_encoder_model", None):
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder %s ...", config.cross_encoder_model)
            cross_encoder = CrossEncoder(config.cross_encoder_model, device=device)
            logger.info(
                "Cross-encoder loaded -- used for passage rerank + "
                "Goal-2 q_a_relevance scoring.",
            )
        except Exception as exc:
            logger.warning(
                "Cross-encoder load failed (%s); passage rerank falls back to "
                "retriever order and q_a_relevance defaults to 0.5 neutral prior. "
                "This degrades Goal 2's sample-(2) closure -- install "
                "sentence-transformers + ensure HF cache has the model.",
                exc,
            )

    pipeline = m["CAEMPipeline"](
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
# Dataset loading
# -----------------------------------------------------------------------------

def load_sil_training_pool(ns: argparse.Namespace, m: Dict[str, Any]) -> Dict[str, list]:
    """Load Train split data for episodic memory generation."""
    n = ns.n_questions
    samples = {}
    requested = [bm.strip().lower() for bm in ns.benchmarks]
    train_capable = {"fever", "triviaqa", "natural_questions"}
    benchmarks = [bm for bm in requested if bm in train_capable]
    if not benchmarks:
        benchmarks = ["fever", "triviaqa", "natural_questions"]
        logger.warning(
            "No train-split SIL benchmarks requested; falling back to %s.",
            benchmarks,
        )

    skipped = [bm for bm in requested if bm not in train_capable]
    if skipped:
        logger.info(
            "Skipping SIL train-pool load for benchmarks without configured train-split SIL usage: %s",
            skipped,
        )

    for bm in benchmarks:
        logger.info("Loading SIL pool %s (n=%d) ...", bm, n)
        split = "train"
        samples[bm] = m["load_benchmark"](bm, split=split, n=n)
    return samples

def load_eval_transfer_pool(ns: argparse.Namespace, m: Dict[str, Any]) -> Dict[str, list]:
    """Load Evaluation data (zero data leakage).

    Uses ``--n_eval_questions`` when provided; otherwise falls back to
    ``--n_questions``. Lets callers size the SIL pool and eval pool independently.
    """
    n = getattr(ns, "n_eval_questions", None) or ns.n_questions
    samples = {}
    benchmarks = [bm.strip().lower() for bm in ns.benchmarks]
    split_map = {
        "fever": "dev",              # lucadiliello/fever eval split (data-leakage safe; renamed from paper_dev)
        "triviaqa": "validation",
        "natural_questions": "validation",
        "strategyqa": "test",
        "arc_challenge": "test",
    }

    for bm in benchmarks:
        logger.info("Loading Eval pool %s (n=%d) ...", bm, n)
        if bm == "truthfulqa":
            samples[bm] = m["load_benchmark"](bm, n=n)
        else:
            split = split_map.get(bm, "validation")
            samples[bm] = m["load_benchmark"](bm, split=split, n=n)
    return samples


def split_calibration_sets(
    samples: Dict[str, list],
    calib_size: int = 500,
    purity_size: int = 500,
    seed: int = 42,
) -> tuple:
    """Split each benchmark's samples into purity / calibration / train sets.

    Returns
    -------
    purity_samples  : dict[bm -> shuffled purity_size (or scaled down)]
    calib_samples   : dict[bm -> shuffled calib_size (or scaled down)]
    train_samples   : dict[bm -> remainder (used for SIL memory population)]

    These three slices are strictly disjoint by index -- critical for
    calibration-isolation (the calibration slice must NEVER be seen by the
    generator during SIL) and for the memorisation-ceiling purity check.

    Notes
    -----
    TruthfulQA has only 817 questions total. Using fixed 500+500 would leave
    zero samples for train. For benchmarks where total < purity_size + calib_size,
    we scale splits proportionally (plan §5.3 specifies 250/250/317 for TruthfulQA).

    The per-benchmark list is shuffled with ``seed`` before slicing so that
    purity / calibration / train draws are not biased by the benchmark's
    native ordering (e.g. FEVER's heavily-correlated adjacent claims). The
    shuffle is deterministic and keyed off the benchmark name so reruns with
    the same ``seed`` produce the same partition.
    """
    import random
    purity, calib, train = {}, {}, {}
    for bm, slist in samples.items():
        # Copy + shuffle deterministically so the original caller list is
        # not mutated. Benchmark name is folded into the seed so different
        # benchmarks get different (but still deterministic) orderings.
        # Python 3.11+ rejects tuple seeds, and hash((seed, bm)) is unstable
        # across runs because str hashing is PEP-456 randomised; the string
        # form uses random.seed's SHA-512 path which is stable.
        rng = random.Random(f"{seed}:{bm}")
        slist = list(slist)
        rng.shuffle(slist)
        total = len(slist)
        effective_purity = purity_size
        effective_calib = calib_size

        # Scale down proportionally when the dataset is too small to fit both
        # purity + calibration windows (e.g. TruthfulQA: 817 < 500+500).
        if total < purity_size + calib_size:
            # Use ≈30% for purity, ≈30% for calibration, ≈40% for train --
            # roughly matching the 250/250/317 ratio the plan specifies.
            effective_purity = total // 3
            effective_calib = total // 3
            logger.warning(
                "  %s: only %d samples -- scaling splits to "
                "%d purity / %d calibration / %d train (plan §5.3)",
                bm, total, effective_purity, effective_calib,
                total - effective_purity - effective_calib,
            )

        purity[bm] = slist[:effective_purity]
        calib[bm] = slist[effective_purity: effective_purity + effective_calib]
        train[bm] = slist[effective_purity + effective_calib:]
        logger.info(
            "  %s: %d purity | %d calibration | %d train",
            bm, len(purity[bm]), len(calib[bm]), len(train[bm]),
        )
    return purity, calib, train


# -----------------------------------------------------------------------------
# General-domain data (anti-forgetting mix for fine-tuning)
# -----------------------------------------------------------------------------

def load_general_data(n: int = 1000, sil_pool_size: int = 0) -> list:
    """Load general QA pairs for the 10% anti-forgetting data mix.

    Uses a small subset of TriviaQA or a synthetic fallback.
    The SelfImprovementLoop mixes 10% of these into the training set
    to satisfy the forgetting_tolerance ≥ 0.93 constraint (§4.6).

    Parameters
    ----------
    n : int
        Number of general-domain pairs to load.
    sil_pool_size : int
        Size of the TriviaQA SIL training pool for the current run. We skip
        past the first ``sil_pool_size`` TriviaQA train rows so the general-
        data mix does NOT overlap with the SIL training pool when TriviaQA
        is also in ``TRAINING_BENCHMARKS``. Overlap would mean the "general"
        anti-forgetting probe was contaminated by training signal.
    """
    from caem.training.self_improvement import QAPair

    try:
        from datasets import load_dataset
        logger.info(
            "Loading TriviaQA for general-domain mix (skip=%d, n=%d) ...",
            sil_pool_size, n,
        )
        # MUST use "train" split -- "validation" overlaps with the evaluation
        # set used in run_cycle(). Using validation here would contaminate the
        # forgetting guard with evaluation data.
        ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
        # Offset past the SIL pool so the "general" anti-forgetting mix is
        # strictly disjoint from the TriviaQA training episodes.
        start = min(sil_pool_size, max(0, len(ds) - n))
        end = min(start + n, len(ds))
        pairs = []
        for item in ds.select(range(start, end)):
            row = cast(Mapping[str, Any], item)
            q = str(row.get("question", ""))
            answer_obj = cast(Mapping[str, Any], row.get("answer", {}))
            ans = str(answer_obj.get("value", "") or "")
            if q and ans:
                pairs.append(QAPair(question=q, answer=ans))
        logger.info("  General-domain mix: %d QA pairs loaded.", len(pairs))
        return pairs
    except Exception as exc:
        # Fail LOUDLY: the general-mix anchor data is used for the
        # forgetting_tolerance constraint in §4.6; silently substituting 200
        # toy arithmetic pairs would produce thesis-invalid retention numbers.
        # Callers (run_smoke_experiment) that legitimately want a synthetic
        # mix should pass the `--smoke_test` flag and go through the
        # `make_synthetic_samples` path instead.
        logger.error(
            "TriviaQA load failed (%s). Refusing to silently substitute a "
            "synthetic general-domain mix — that would fabricate the anti-"
            "forgetting retention measurements. Fix the dataset cache / "
            "network and re-run, or use --smoke_test for a flagged synthetic "
            "run.",
            exc,
        )
        raise


# -----------------------------------------------------------------------------
# Calibration (temperature scaling)
# -----------------------------------------------------------------------------

def run_calibration_step(
    pipeline,
    calib_samples: Dict[str, list],
    config,
    output_dir: Path,
    m,
) -> None:
    """Fit the temperature scalar T on the disjoint calibration slice.

    Calls run_calibration.py logic inline so the main loop stays clean.
    Results are written to outputs/calibration/calibrated_config.json
    and the config object is updated in place.

    Called once after Cycle 0 evaluation, before Cycle 1 fine-tuning.
    u_stored composite weights are fixed by design and are NOT calibrated.
    """
    logger.info("-" * 60)
    logger.info("CALIBRATION -- fitting temperature scalar T (u_stored weights fixed)")
    logger.info("-" * 60)

    calib_dir = output_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    try:
        from scripts.run_calibration import calibrate_pipeline
        calibrate_pipeline(pipeline, calib_samples, config, calib_dir)
    except Exception as exc:
        logger.warning(
            "Calibration step failed (%s). "
            "Continuing with the uncalibrated default T=1.0. "
            "Re-run scripts/run_calibration.py manually after Cycle 0.",
            exc,
        )


def run_per_cycle_recalibration(
    pipeline,
    calib_samples: Dict[str, list],
    config,
    output_dir: Path,
    cycle: int,
) -> None:
    """Unconditional per-cycle temperature re-fit at the end of Cycle `cycle`.

    Conservative recalibration protocol (Ovadia et al. NeurIPS 2019,
    Thulasidasan et al. 2019): T is the only runtime-active calibration
    surface, and fine-tuning shifts the generator's logit scale every
    cycle, so T must be re-fit. The u_stored composite weights are fixed
    by design and are NOT re-fit.

    The calibration slice must be DISJOINT from the current cycle's SIL
    training pool and from the held-out eval set. Caller is responsible
    for ensuring `calib_samples` obeys this partition (see
    :func:`assert_disjoint_calibration` below).

    Called at the end of each cycle, after retroactive re-verification
    and memory-store save, before the next cycle's fine-tuning.
    """
    calib_dir = output_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    try:
        from scripts.run_calibration import calibrate_pipeline_temperature_only
        logger.info(
            "Cycle %d: per-cycle temperature re-fit (T only; weights design-fixed).",
            cycle,
        )
        calibrate_pipeline_temperature_only(
            pipeline, calib_samples, config, calib_dir, cycle=cycle
        )
    except Exception as exc:
        logger.warning(
            "Cycle %d per-cycle temperature re-fit failed (%s); "
            "continuing with the previous cycle's T value.",
            cycle, exc,
        )


def assert_disjoint_calibration(
    calib_ids_by_bm: Dict[str, set],
    train_ids_by_bm: Dict[str, set],
    eval_ids_by_bm: Optional[Dict[str, set]] = None,
) -> None:
    """Raise AssertionError if the calibration slice overlaps SIL train or eval.

    Gap 5 isolation guarantee (ch.4 §calibration): T must be fit on a slice
    that is NEVER shown to the generator during SIL, and that is distinct
    from the held-out benchmark eval set. Violating this biases ECE.

    Parameters
    ----------
    calib_ids_by_bm : dict[benchmark_name -> set of sample IDs]
    train_ids_by_bm : dict[benchmark_name -> set of sample IDs]
    eval_ids_by_bm  : optional dict[benchmark_name -> set of sample IDs]

    Notes
    -----
    The IDs should be read from ``dataset_splits.json`` which is the single
    source of truth for the three-way partition (train / calib / eval).
    """
    # NOTE: use ``raise`` (not ``assert``) so the guard survives ``python -O``
    # and the check is NEVER silently stripped in a production run.
    for bm, calib_ids in calib_ids_by_bm.items():
        train_ids = train_ids_by_bm.get(bm, set())
        overlap_train = calib_ids & train_ids
        if overlap_train:
            raise RuntimeError(
                f"Calibration slice for {bm!r} overlaps SIL training pool "
                f"({len(overlap_train)} shared IDs). "
                f"First 5: {list(overlap_train)[:5]}"
            )
        if eval_ids_by_bm is not None:
            eval_ids = eval_ids_by_bm.get(bm, set())
            overlap_eval = calib_ids & eval_ids
            if overlap_eval:
                raise RuntimeError(
                    f"Calibration slice for {bm!r} overlaps held-out eval set "
                    f"({len(overlap_eval)} shared IDs). "
                    f"First 5: {list(overlap_eval)[:5]}"
                )
    logger.info(
        "Disjoint calibration slice verified across %d benchmarks.",
        len(calib_ids_by_bm),
    )


# -----------------------------------------------------------------------------
# Retroactive re-verification
# -----------------------------------------------------------------------------

def retroactive_reverification(pipeline, cycle: int, config) -> Dict:
    """Re-verify all stored episodes after fine-tuning.

    After each cycle the improved model may score stored episodes
    differently. Episodes that fall below retroverify_prune_threshold (0.50)
    are pruned; episodes that score higher get their u_stored upgraded.

    This is §4.7 (retroactive re-verification) of the thesis plan.

    Uses EpisodicMemoryStore.retroverify(verify_fn, threshold) -- the store's
    own method that iterates entries, calls verify_fn on each, and handles
    pruning + u_stored upgrades atomically.

    Returns
    -------
    dict with keys: total, updated, pruned
    """
    store = pipeline.memory_store
    total_before = len(store.all_entries())

    logger.info("Retroactive re-verification: %d episodes ...", total_before)

    # verify_fn: (EpisodicEntry) -> UnifiedVerifierOutput
    # Uses the pipeline's verifier so re-verification benefits from the
    # updated model weights after this cycle's fine-tuning.
    verify_fn = pipeline.make_retroverify_fn()

    n_updated, n_removed = store.retroverify(
        verify_fn=verify_fn,
        threshold=config.retroverify_prune_threshold,
    )

    logger.info(
        "  Retroactive re-verification done: %d total | %d updated | %d pruned",
        total_before, n_updated, n_removed,
    )
    return {"total": total_before, "updated": n_updated, "pruned": n_removed}


# -----------------------------------------------------------------------------
# Summary table (Chapter 5 mechanism evidence table)
# -----------------------------------------------------------------------------

def save_summary_csv(
    all_cycle_results: List[Dict],
    output_dir: Path,
    mmlu_per_cycle: Optional[List[float]] = None,
) -> None:
    """Save the five-mechanism evidence table as a CSV (Table 1 in Chapter 5).

    Columns: cycle, benchmark, em, f1, hallucination_rate,
             tier{1,2,3}_frac_pct, storage_rate_pct, mean_u_stored,
             mean_latency_ms, mmlu_retention_pct, mmlu_retention_ratio_pct.

    Parameters
    ----------
    all_cycle_results : list[dict]
        One entry per cycle; each entry maps benchmark -> eval metrics dict.
    output_dir : Path
        Where to write experiment_summary.csv.
    mmlu_per_cycle : list[float] or None
        Per-cycle MMLU retention values in [0, 1] or NaN.  Index 0 is the
        pristine-model baseline measured before any fine-tuning, and is the
        denominator used to compute ``mmlu_retention_ratio_pct`` for cycles
        >= 1.  When None or all-NaN the ratio column is left blank.

    Notes
    -----
    MMLU retention is a single value per cycle (not per benchmark).  To keep
    the CSV tidy it is written only on the first benchmark row of each cycle;
    all other rows for that cycle leave the field blank.  This matches how
    MMLU is reported in Chapter 5 Table 5.2 (one row per cycle, aggregate).

    Two MMLU columns are emitted:
      * ``mmlu_retention_pct``       -- absolute MMLU accuracy (%) in this cycle.
      * ``mmlu_retention_ratio_pct`` -- accuracy relative to the pristine
        Cycle-0 baseline (%). 100 = no forgetting; <100 = forgetting; >100
        = slight positive transfer (possible with EWC at low cycle counts).
    """
    csv_path = output_dir / "experiment_summary.csv"
    fieldnames = [
        "cycle",
        "benchmark",
        "em",
        "f1",
        "hallucination_rate",
        "tier1_frac_pct",
        "tier2_frac_pct",
        "tier3_frac_pct",
        "storage_rate_pct",
        "mean_u_stored",
        "mean_latency_ms",
        "mmlu_retention_pct",        # absolute MMLU accuracy this cycle
        "mmlu_retention_ratio_pct",  # ratio against Cycle-0 baseline
    ]

    # Baseline denominator for the retention ratio. The Cycle-0 entry is the
    # pristine-model MMLU measurement; if unavailable (NaN or <= 0) the ratio
    # column is left blank to avoid fabricating retention numbers.
    baseline_mmlu = None
    if mmlu_per_cycle and len(mmlu_per_cycle) > 0:
        b = mmlu_per_cycle[0]
        if not math.isnan(b) and b > 0:
            baseline_mmlu = b

    rows = []
    for cycle_num, cycle_results in enumerate(all_cycle_results):
        # MMLU retention is scalar per cycle; write it on the first BM row only.
        mmlu_val = (mmlu_per_cycle[cycle_num]
                    if mmlu_per_cycle and cycle_num < len(mmlu_per_cycle)
                    else float("nan"))
        first_bm = True
        for bm, res in cycle_results.items():
            if first_bm and not math.isnan(mmlu_val):
                mmlu_str = str(round(mmlu_val * 100, 2))
                if baseline_mmlu is not None:
                    ratio_str = str(round((mmlu_val / baseline_mmlu) * 100, 2))
                else:
                    ratio_str = ""
            else:
                mmlu_str = ""   # leave blank for subsequent BM rows (or if NaN)
                ratio_str = ""
            first_bm = False
            rows.append({
                "cycle":              cycle_num,
                "benchmark":          bm,
                "em":                 round(res.get("em", 0.0), 4),
                "f1":                 round(res.get("f1", 0.0), 4),
                "hallucination_rate": round(res.get("hallucination_rate", 0.0), 4),
                "tier1_frac_pct":     round(res.get("tier1_frac", 0.0) * 100, 1),
                "tier2_frac_pct":     round(res.get("tier2_frac", 0.0) * 100, 1),
                "tier3_frac_pct":     round(res.get("tier3_frac", 0.0) * 100, 1),
                "storage_rate_pct":   round(res.get("storage_rate", 0.0) * 100, 1),
                "mean_u_stored":      round(res.get("mean_u_stored", 0.0), 4),
                "mean_latency_ms":    round(res.get("mean_latency_ms", 0.0), 1),
                "mmlu_retention_pct":       mmlu_str,
                "mmlu_retention_ratio_pct": ratio_str,
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Summary CSV saved -> %s", csv_path)


def print_mechanism_table(all_cycle_results: List[Dict]) -> None:
    """Print the five-mechanism evidence table to stdout (Chapter 5, Table 1).

    Expected patterns (10-cycle design; directional hypotheses, not
    hardcoded pass/fail targets -- exact equilibrium values are
    measured empirically and reported in Chapter 5 Table 5.X):
      Tier 1 fraction: rising per cycle; late-cycle equilibrium
        determined by the empirical distribution of u_stored and is
        one of the mechanism signals we are trying to measure, not
        a pre-set target.
      Hallucination reduction: growing per cycle; practical
        equilibrium expected in late cycles (C7-C9 in the 10-cycle
        design).
      MMLU Retention: >= ``CAEMConfig.forgetting_tolerance`` across
        all cycles (config-backed -- do not hardcode a literal
        percentage here; see caem/config.py).
      Mean u_stored: rising monotonically across cycles
        (retroactive re-verification working).
      Data Purity: rising as low-quality episodes are pruned and
        high-quality ones dominate.

    This table is descriptive, not validating -- it prints whatever
    the run produced and lets the reader compare against Chapter 5.
    If you want an assertion-style check against specific numbers,
    add it to the post-run analysis script, not here.
    """
    print("\n" + "=" * 90)
    print("MECHANISM EVIDENCE TABLE  (Chapter 5, Table 1)")
    print("=" * 90)
    # Header/data column widths MUST match — otherwise reviewers skim a
    # mis-aligned table and distrust the numbers. Each header field here
    # is sized to the exact width of its data field (including the trailing
    # '%' character where present).
    print(
        f"{'Cycle':<6} {'BM':<12} {'EM':>6} {'F1':>6} "
        f"{'HallRed%':>9} {'T1%':>6} {'T3%':>6} "
        f"{'Stored%':>8} {'uMean':>7}"
    )
    print("-" * 90)

    cycle0_em: Dict[str, float] = {}

    for cycle_num, cycle_results in enumerate(all_cycle_results):
        for bm, res in sorted(cycle_results.items()):
            em = float(res.get("em", 0.0) or 0.0)
            if cycle_num == 0:
                cycle0_em[bm] = em

            # Hallucination reduction relative to cycle 0
            base = float(cycle0_em.get(bm, em) or 0.0)
            hall_red = ((base - em) / base * -100) if base > 0 else 0.0
            # NOTE: positive hall_red = reduction in hallucinations (EM improved)

            print(
                f"{cycle_num:<6} {bm:<12} "
                f"{em:>6.3f} {res.get('f1', 0.0):>6.3f} "
                f"{hall_red:>+8.1f}% "
                f"{res.get('tier1_frac', 0.0)*100:>5.1f}% "
                f"{res.get('tier3_frac', 0.0)*100:>5.1f}% "
                f"{res.get('storage_rate', 0.0)*100:>7.1f}% "
                f"{res.get('mean_u_stored', 0.0):>7.4f}"
            )
    print("=" * 90)
    print("Note: HallRed% = EM improvement vs Cycle 0 (positive = better).")
    print("MMLU Retention% is recorded per-cycle in the summary CSV (EXP-MMLU-FIX).\n")


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def run_experiment(ns: argparse.Namespace) -> None:
    _check_deps()
    m = _load_imports()
    torch = m["torch"]

    output_dir = Path(ns.output_dir)
    eval_dir = output_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # -- Persistent file log ------------------------------------------------- #
    # Main experiment runs on Vast.ai take ~12 hours; if the SSH terminal
    # drops or the instance reboots, an stdout-only log is gone. Tee every
    # log line to a file under output_dir so cycle-by-cycle routing
    # distributions, α measurements, SIL training loss, and MMLU
    # retention ratios are preserved for post-run analysis. Non-fatal
    # on failure so the experiment still launches if output_dir is
    # read-only or disk is exhausted. Same pattern applied in
    # aggregate_ablation.py (NOTE 9 resolution).
    try:
        _file_handler = logging.FileHandler(
            str(output_dir / "experiment.log"),
            mode="a",
            encoding="utf-8",
        )
        _file_handler.setLevel(logging.INFO)
        _file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        ))
        logging.getLogger().addHandler(_file_handler)
        logger.info(
            "File logging enabled: %s", str(output_dir / "experiment.log"),
        )
    except Exception as _exc:
        logger.warning(
            "Could not attach file-log handler at %s (%s); "
            "continuing with stdout logging only.",
            str(output_dir / "experiment.log"), _exc,
        )

    # -- Config ------------------------------------------------------------- #
    config = m["CAEMConfig"]()
    # Override n_questions if specified
    config.questions_per_cycle = ns.n_questions
    config.num_cycles = ns.num_cycles
    if getattr(ns, "verifier_backend", None):
        config.verifier_backend = ns.verifier_backend
    # Threshold overrides from scripts/calibrate_thresholds.py (fit once at
    # the Cycle-0 boundary, held fixed for cycles 1..N). Required ordering:
    # train > store > defer. Missing override falls back to the CAEMConfig
    # default (which is RoBERTa-era and likely wrong under MiniCheck).
    if getattr(ns, "store_threshold", None) is not None:
        config.store_threshold = float(ns.store_threshold)
    if getattr(ns, "defer_threshold", None) is not None:
        config.defer_threshold = float(ns.defer_threshold)
    if getattr(ns, "train_threshold", None) is not None:
        config.min_u_stored_for_training = float(ns.train_threshold)
    # Validate ordering if any override was set.
    if any(getattr(ns, attr, None) is not None
           for attr in ("store_threshold", "defer_threshold", "train_threshold")):
        if not (config.min_u_stored_for_training
                > config.store_threshold
                > config.defer_threshold):
            raise SystemExit(
                f"Threshold ordering violated after CLI overrides: "
                f"train={config.min_u_stored_for_training} > "
                f"store={config.store_threshold} > "
                f"defer={config.defer_threshold} "
                f"must hold. Refusing to launch."
            )
        logger.info(
            "Decision-tree thresholds overridden via CLI: "
            "store=%.4f defer=%.4f train=%.4f",
            config.store_threshold, config.defer_threshold,
            config.min_u_stored_for_training,
        )

    if ns.resume_from_cycle < 0 or ns.resume_from_cycle > config.num_cycles:
        raise ValueError(
            f"--resume_from_cycle must be between 0 and {config.num_cycles}, "
            f"got {ns.resume_from_cycle}."
        )

    # -- Build pipeline ------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info(
        "CAEM EXPERIMENT -- Full Scale Run (%d cycles, %d questions/cycle, full DPR Wikipedia)",
        config.num_cycles, config.questions_per_cycle,
    )
    logger.info("=" * 60)
    pipeline = build_pipeline(config, ns, m)

    # -- Seed cold-start memory ---------------------------------------------- #
    if ns.cold_start_memory:
        mem_path = Path(ns.cold_start_memory)
        # save() appends .faiss / .meta; check for the .faiss file to confirm existence
        faiss_file = Path(str(mem_path) + ".faiss")
        if faiss_file.exists():
            logger.info("Loading cold-start memory from %s ...", mem_path)
            from caem.memory.store import EpisodicMemoryStore
            pipeline.memory_store = EpisodicMemoryStore.load(str(mem_path))
            logger.info("Cold-start memory loaded (%d episodes).", pipeline.memory_store.size)
        else:
            raise FileNotFoundError(f"Cold-start memory path not found: {faiss_file}")

    # -- Load datasets ------------------------------------------------------- #
    if ns.smoke_test:
        logger.info("SMOKE TEST MODE -- using synthetic samples (n=10)")
        from eval.benchmarks import make_synthetic_samples
        requested = [bm.strip().lower() for bm in ns.benchmarks]
        sil_benchmarks = [bm for bm in requested if bm in {"fever", "triviaqa", "natural_questions"}]
        if not sil_benchmarks:
            sil_benchmarks = ["fever", "triviaqa", "natural_questions"]
        sil_pool = {bm: make_synthetic_samples(bm, n=10) for bm in sil_benchmarks}
        eval_samples = {bm: make_synthetic_samples(bm, n=10) for bm in requested}

        # Minimal disjoint partition for smoke mode (IDs are trivially unique).
        purity_samples = {bm: s[:3] for bm, s in sil_pool.items()}
        calib_samples  = {bm: s[3:6] for bm, s in sil_pool.items()}
        train_samples  = {bm: s[6:]  for bm, s in sil_pool.items()}
    else:
        sil_pool = load_sil_training_pool(ns, m)
        eval_samples = load_eval_transfer_pool(ns, m)

        # Purity, Calibration, and Train are carved into three DISJOINT slices
        # of the SIL Train Pool. The calibration slice is NEVER shown to the
        # generator during SIL (Gap-5 isolation; see §4.Verifier-Calibration
        # discussion in Chapter 4).
        purity_samples, calib_samples, train_samples = split_calibration_sets(
            sil_pool,
            calib_size=config.calibration_set_size,
            purity_size=config.purity_validation_set_size,
            seed=getattr(ns, "seed", 42),
        )

    # -- Content-hash stable IDs for disjointness checks ------------------- #
    # Problem: some HuggingFace dataset variants have non-unique native
    # sample IDs, either across splits (FEVER lucadiliello reuses integer
    # claim IDs between train and dev) or within a single split
    # (TriviaQA HF dump has duplicate sfq_/qw_/qz_ IDs). Relying on native
    # .id for disjointness checks therefore produces spurious overlap
    # warnings (same native ID on content-distinct samples) AND misses
    # real overlap (different native IDs on content-identical duplicates).
    # Fix: attach a content-hash identifier derived from question text to
    # every sample BEFORE slicing; use that hash as the disjointness key.
    import hashlib as _hashlib

    def _content_id(sample: dict) -> str:
        q = str(sample.get("question", "")).strip()[:500]
        return _hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]

    for bm in sil_pool:
        for s in sil_pool[bm]:
            s["_content_id"] = _content_id(s)
    for bm in eval_samples:
        for s in eval_samples[bm]:
            s["_content_id"] = _content_id(s)

    # Re-stamp post-split slices (they share object references with sil_pool
    # so the _content_id is already set; this is just a defensive pass).
    for bm in list(calib_samples.keys()):
        for s in calib_samples[bm]:
            if "_content_id" not in s:
                s["_content_id"] = _content_id(s)
        for s in purity_samples[bm]:
            if "_content_id" not in s:
                s["_content_id"] = _content_id(s)
        for s in train_samples[bm]:
            if "_content_id" not in s:
                s["_content_id"] = _content_id(s)

    # Deduplicate within each slice by content hash. Datasets like TriviaQA
    # contain literal duplicate questions within a single split; leaving
    # them in causes calib/train/purity slices to look "overlapping" at the
    # content level even when they're index-disjoint. Strategy: keep only
    # the first occurrence per content-id within each slice, then remove
    # train samples that also appear in calib or purity by content.
    for bm in list(calib_samples.keys()):
        def _dedup_inplace(slist):
            seen = set()
            out = []
            for s in slist:
                cid = s["_content_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                out.append(s)
            return out, seen

        calib_samples[bm], calib_ids = _dedup_inplace(calib_samples[bm])
        purity_samples[bm], purity_ids = _dedup_inplace(purity_samples[bm])
        # Remove train samples whose content-id is in calib or purity slice.
        reserved = calib_ids | purity_ids
        kept_train = [s for s in train_samples[bm] if s["_content_id"] not in reserved]
        train_dropped = len(train_samples[bm]) - len(kept_train)
        if train_dropped:
            logger.warning(
                "  %s: dropped %d train samples that duplicate calib/purity "
                "content within the SIL pool.", bm, train_dropped,
            )
        train_samples[bm] = kept_train

    # Drop calib / purity samples whose content hash appears in the eval
    # slice for that benchmark. Content-hash collision between train-drawn
    # calib and dev-drawn eval indicates a genuine content duplicate across
    # splits and must be filtered to preserve the disjointness guarantee.
    for bm in list(calib_samples.keys()):
        if bm not in eval_samples:
            continue
        eval_content_ids = {s["_content_id"] for s in eval_samples[bm]}

        def _not_colliding(slist):
            kept = []
            dropped = 0
            for s in slist:
                if s["_content_id"] in eval_content_ids:
                    dropped += 1
                    continue
                kept.append(s)
            return kept, dropped

        calib_filtered, calib_dropped = _not_colliding(calib_samples[bm])
        purity_filtered, purity_dropped = _not_colliding(purity_samples[bm])
        if calib_dropped or purity_dropped:
            logger.warning(
                "  %s: dropped %d calib + %d purity samples whose content "
                "hash collides with an eval-split sample. Remaining: %d "
                "calib, %d purity.",
                bm, calib_dropped, purity_dropped,
                len(calib_filtered), len(purity_filtered),
            )
        calib_samples[bm] = calib_filtered
        purity_samples[bm] = purity_filtered

    # -- Save purity/calibration/train sample IDs (for reproducibility) ---- #
    meta_path = output_dir / "dataset_splits.json"
    with open(meta_path, "w") as f:
        meta_out = {}
        for bm in eval_samples:
            meta_out[bm] = {"eval_ids": [s.get("id", i) for i, s in enumerate(eval_samples[bm])]}
            if bm in purity_samples:
                meta_out[bm]["purity_ids"] = [s.get("id", i) for i, s in enumerate(purity_samples[bm])]
                meta_out[bm]["calib_ids"]  = [s.get("id", i) for i, s in enumerate(calib_samples[bm])]
                meta_out[bm]["train_ids"]  = [s.get("id", i) for i, s in enumerate(train_samples[bm])]
        json.dump(meta_out, f, indent=2)
    logger.info("Dataset split metadata saved -> %s", meta_path)

    # -- Assert the three-way partition is disjoint (Gap-5 isolation) ------- #
    # calib vs train: never seen by the generator during SIL.
    # calib vs eval:  never seen by the harness during eval.
    try:
        # Use content-hash stable IDs (attached above) so the assertion is
        # not confused by datasets that reuse native .id values within or
        # across splits. Content-hash collisions here indicate GENUINE
        # content-level overlap and must be treated as an error.
        assert_disjoint_calibration(
            calib_ids_by_bm={
                bm: {s["_content_id"] for s in calib_samples[bm]}
                for bm in calib_samples
            },
            train_ids_by_bm={
                bm: {s["_content_id"] for s in train_samples[bm]}
                for bm in train_samples
            },
            eval_ids_by_bm={
                bm: {s["_content_id"] for s in eval_samples[bm]}
                for bm in eval_samples if bm in calib_samples
            },
        )
    except AssertionError as exc:
        logger.error("Gap-5 isolation violated: %s", exc)
        raise

    # -- General-domain data (for anti-forgetting mix) ---------------------- #
    # Skip past the TriviaQA-train rows that may be in the SIL training pool
    # so the anti-forgetting probe is strictly disjoint from training signal.
    trivia_sil_pool_size = (
        len(train_samples.get("triviaqa", []))
        if "triviaqa" in train_samples else 0
    )
    general_data = load_general_data(
        n=getattr(config, "general_data_size", 1000),
        sil_pool_size=trivia_sil_pool_size,
    )

    # -- Eval harness -------------------------------------------------------- #
    harness = m["EvalHarness"](pipeline, output_dir=str(eval_dir), log_every=100)

    # -- Self-improvement loop ----------------------------------------------- #
    sil = m["SelfImprovementLoop"](
        model=pipeline.model,
        tokenizer=pipeline.tokenizer,
        config=config,
        output_dir=str(output_dir),
    )

    # Per-cycle MMLU retention list. Index 0 is the pristine-model baseline
    # measured before any fine-tuning; subsequent indices are measured after
    # each SIL fine-tuning step. The Cycle-0 baseline is what Chapter 5's
    # retention ratio denominates against (mmlu_retention_ratio_pct).
    mmlu_per_cycle: List[float] = []

    # Per-cycle equilibrium-gate signal lists. Populated post-eval so the
    # gate has real per-cycle evidence (CES proxy = mean EM across
    # eval benchmarks; storage rate = mean of per-benchmark storage_rate
    # fields written by the harness). The MMLU retention signal is read
    # directly from mmlu_per_cycle to avoid duplicate state. See
    # caem/eval/equilibrium.py for the gate logic.
    ces_history: List[float] = []
    storage_rates: List[float] = []

    # -- CYCLE 0: Baseline evaluation & Resume Logic ------------------------ #
    if ns.resume_from_cycle == 0:
        logger.info("-" * 60)
        logger.info("CYCLE 0 -- Baseline evaluation (zero episodic memory)")
        logger.info("-" * 60)
        t0 = time.time()

        # Zero Data Leakage Eval: Never store during eval
        cycle0_results = harness.run_all(eval_samples, cycle=0, store_to_memory=False)
        logger.info("Cycle 0 done in %.1f min.", (time.time() - t0) / 60)

        # Save baseline memory store checkpoint
        pipeline.memory_store.save(str(output_dir / "memory_store_cycle_0"))

        # Save deferred buffer checkpoint (empty at cycle 0, but the file's
        # presence lets --resume_from_cycle N discover the buffer path).
        # Use getattr so the save is a no-op if the pipeline exposes no
        # deferred_buffer attribute (e.g. ablation pipelines with deferred
        # reconsideration disabled).
        _def_buf = getattr(pipeline, "deferred_buffer", None)
        if _def_buf is not None:
            try:
                _def_buf.save(str(output_dir / "deferred_buffer_cycle_0.pkl"))
            except Exception as exc:
                logger.warning("Failed to save deferred buffer at cycle 0 (%s).", exc)

        # Measure pristine-model MMLU *before* any fine-tuning. This is the
        # denominator for the retention ratio in later cycles. Persisted to
        # disk so resume paths can recover it without remeasuring.
        logger.info("Measuring pristine-model MMLU baseline (200 samples)...")
        mmlu_baseline = sil.measure_mmlu(n=200)
        logger.info("Pristine MMLU baseline: %.4f", mmlu_baseline)
        try:
            with open(output_dir / "mmlu_baseline.json", "w", encoding="utf-8") as f:
                json.dump({"mmlu_baseline": float(mmlu_baseline)}, f)
        except Exception as exc:
            logger.warning("Failed to persist mmlu_baseline.json (%s)", exc)

        all_cycle_results = [cycle0_results]
        mmlu_per_cycle.append(float(mmlu_baseline))  # Cycle 0 = pristine baseline

        # -- Calibration (after Cycle 0, before Cycle 1 fine-tuning) ------------ #
        if not ns.skip_calibration:
            run_calibration_step(pipeline, calib_samples, config, output_dir, m)
        else:
            logger.info("Calibration skipped (--skip_calibration). Using equal initial weights.")
    else:
        logger.info("-" * 60)
        logger.info("RESUMING EXPERIMENT FROM CYCLE %d", ns.resume_from_cycle)
        logger.info("Reconstructing previous metrics and memory states...")
        logger.info("-" * 60)
        
        all_cycle_results = []
        # Reconstruct all_cycle_results up to the resume point
        for c in range(ns.resume_from_cycle):
            cycle_results = {}
            for bm in eval_samples.keys():
                json_path = eval_dir / f"{bm}_cycle{c}.json"
                if not json_path.exists():
                    raise FileNotFoundError(f"Cannot resume: {json_path} is missing.")
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cycle_results[bm] = data["meta"]
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise RuntimeError(
                        f"Cannot resume: {json_path} is corrupted ({exc}). "
                        f"The file was likely truncated by a crash mid-write. "
                        f"Delete it and re-run from cycle {c} or earlier."
                    ) from exc
            all_cycle_results.append(cycle_results)

            # Reconstruct mmlu_per_cycle from saved retroverify JSONs.
            # Cycle 0 stores the pristine-model baseline in mmlu_baseline.json;
            # fall back to remeasuring if the file is missing (older runs) so
            # downstream retention ratios still have a valid denominator.
            if c == 0:
                baseline_path = output_dir / "mmlu_baseline.json"
                if baseline_path.exists():
                    try:
                        with open(baseline_path, "r", encoding="utf-8") as f:
                            mmlu_per_cycle.append(float(json.load(f)["mmlu_baseline"]))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        logger.warning(
                            "mmlu_baseline.json corrupted; remeasuring pristine MMLU."
                        )
                        mmlu_per_cycle.append(float(sil.measure_mmlu(n=200)))
                else:
                    logger.info(
                        "No mmlu_baseline.json found (older run?); "
                        "remeasuring pristine MMLU for the retention denominator."
                    )
                    mmlu_per_cycle.append(float(sil.measure_mmlu(n=200)))
            else:
                rv_path_c = output_dir / f"retroverify_cycle{c}.json"
                try:
                    with open(rv_path_c, "r", encoding="utf-8") as f:
                        rv_data = json.load(f)
                    mmlu_val = rv_data.get("fine_tune", {}).get("mmlu_retention", float("nan"))
                    mmlu_per_cycle.append(float(mmlu_val))
                except (FileNotFoundError, KeyError, ValueError):
                    mmlu_per_cycle.append(float("nan"))
            
        prev_cycle = ns.resume_from_cycle - 1
        
        # Reload memory store state
        # save() appends .faiss / .meta extensions; check for the .faiss file
        mem_path = output_dir / f"memory_store_cycle_{prev_cycle}"
        faiss_file = Path(str(mem_path) + ".faiss")
        if not faiss_file.exists():
            raise FileNotFoundError(f"Missing memory store checkpoint: {faiss_file}")
        from caem.memory.store import EpisodicMemoryStore
        pipeline.memory_store = EpisodicMemoryStore.load(str(mem_path))
        restored_size = pipeline.memory_store.size
        logger.info("Restored memory store from Cycle %d: %d episodes", prev_cycle, restored_size)
        # Sanity check: at 5000 questions/cycle across 3 training benchmarks,
        # expect at least ~50 stored episodes per completed training cycle.
        # A suspiciously small store likely means the wrong checkpoint was loaded.
        if prev_cycle >= 1 and restored_size < prev_cycle * 20:
            logger.warning(
                "Memory store has only %d episodes after %d cycles — expected ~%d+. "
                "Check that memory_store_cycle_%d is the correct checkpoint.",
                restored_size, prev_cycle, prev_cycle * 50, prev_cycle,
            )
        
        # Reload fine-tuned model weights if past cycle 0
        if prev_cycle > 0:
            sil.load_checkpoint(prev_cycle)
            logger.info("Restored fine-tuned model weights from Cycle %d.", prev_cycle)

        # Reload deferred buffer if its snapshot exists. Older runs (pre the
        # deferred-buffer persistence fix) won't have this file; in that case
        # we start the next cycle with an empty buffer and log the gap so the
        # operator knows any pre-existing deferred signal is not recovered.
        deferred_path = output_dir / f"deferred_buffer_cycle_{prev_cycle}.pkl"
        _def_buf = getattr(pipeline, "deferred_buffer", None)
        if _def_buf is None:
            logger.info(
                "Pipeline has no deferred_buffer attribute; skipping deferred "
                "buffer restore. This is expected for ablation pipelines that "
                "disable deferred reconsideration."
            )
        elif deferred_path.exists():
            try:
                _def_buf.load(str(deferred_path))
                logger.info(
                    "Restored deferred buffer from Cycle %d: %d entries.",
                    prev_cycle, _def_buf.size,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load deferred buffer snapshot %s (%s). "
                    "Continuing with an empty deferred buffer.",
                    deferred_path, exc,
                )
        else:
            logger.warning(
                "No deferred buffer snapshot at %s (older run?). "
                "Starting Cycle %d with an empty deferred buffer; any "
                "cycle-%d deferred entries will not be reconsidered.",
                deferred_path, ns.resume_from_cycle, prev_cycle,
            )

        # Reload calibration temperature T so the resumed run does not revert
        # to the default T=1.0 for one cycle before the next per-cycle
        # recalibration overwrites it. Preference order:
        #   1. Previous cycle's per-cycle re-fit (calibrated_config_cycle{N}.json)
        #   2. Cycle-0 initial fit (calibrated_config.json)
        #   3. Default (keep config.temperature_scalar as-is, log a warning)
        calib_dir = output_dir / "calibration"
        per_cycle_calib = calib_dir / f"calibrated_config_cycle{prev_cycle}.json"
        initial_calib = calib_dir / "calibrated_config.json"
        calib_src = None
        if per_cycle_calib.exists():
            calib_src = per_cycle_calib
        elif initial_calib.exists():
            calib_src = initial_calib
        if calib_src is not None:
            try:
                with open(calib_src, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
                # Two shapes are possible here:
                #   calibrated_config.json (cycle-0 initial fit) stores the
                #     value under "temperature_scalar".
                #   calibrated_config_cycle{N}.json (per-cycle re-fit) stores
                #     it under "temperature_after".
                # Try the per-cycle key first; fall back to the initial key.
                if "temperature_after" in calib_data:
                    T_reloaded = float(calib_data["temperature_after"])
                elif "temperature_scalar" in calib_data:
                    T_reloaded = float(calib_data["temperature_scalar"])
                else:
                    raise KeyError(
                        "Neither 'temperature_after' nor 'temperature_scalar' "
                        "found in calibration snapshot."
                    )
                config.temperature_scalar = T_reloaded
                logger.info(
                    "Restored calibration T = %.4f from %s",
                    T_reloaded, calib_src.name,
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning(
                    "Failed to parse %s (%s); keeping T = %.4f.",
                    calib_src, exc, config.temperature_scalar,
                )
        else:
            logger.warning(
                "No calibration snapshot found in %s; keeping default "
                "T = %.4f. The next cycle-boundary recalibration will "
                "overwrite this.",
                calib_dir, config.temperature_scalar,
            )

        pipeline.current_cycle = prev_cycle

    # -- CYCLES 1..N --------------------------------------------------------- #
    start_cycle = max(1, ns.resume_from_cycle)
    for cycle_num in range(start_cycle, config.num_cycles + 1):
        logger.info("-" * 60)
        logger.info("CYCLE %d -- Fine-tuning + evaluation", cycle_num)
        logger.info("-" * 60)
        t0 = time.time()

        # --- Per-cycle step numbering ---
        # Comment labels (Step 1..7) and logger labels MUST stay in lockstep;
        # reviewers trace `grep "  Step "` against the thesis pipeline
        # diagram to confirm the flow. Previously the comments said "Step
        # 4.5" / "Step 5" / "Step 6" while the logger emitted "Step 2.5" /
        # "Step 3" — this section is renumbered below so both agree.

        # Step 1: Fine-tune on verified episodes from current memory.
        # Passing verify_fn enables retroactive re-verification (Phase 5d):
        # after fine-tuning, every stored episode is re-scored by the updated
        # model via UnifiedVerifier. Raised u_stored propagates all nine
        # signals back onto the entry; entries below the prune threshold
        # are removed.
        logger.info("  Step 1: SelfImprovementLoop.run_cycle(%d) ...", cycle_num)
        cycle_result = sil.run_cycle(
            cycle_num=cycle_num,
            memory_store=pipeline.memory_store,
            general_data=general_data,
            verify_fn=pipeline.make_retroverify_fn(),
        )

        if cycle_result.aborted:
            logger.warning(
                "  Cycle %d ABORTED (retention ratio %.3f < %.3f). "
                "Weights restored. Eval still runs on the restored model.",
                cycle_num, cycle_result.mmlu_retention_ratio, config.forgetting_tolerance,
            )
        else:
            logger.info(
                "  Fine-tuning done: %d episodes | retention_ratio=%.3f | loss=%.4f "
                "| retroverify: %d updated / %d pruned",
                cycle_result.n_episodes_used,
                cycle_result.mmlu_retention_ratio,
                cycle_result.final_train_loss,
                cycle_result.n_retroverified,
                cycle_result.n_retropruned,
            )

        # (pipeline cycle counter update — housekeeping, not a numbered step)
        pipeline.current_cycle = cycle_num

        # Step 2: Retroactive re-verification of memory
        logger.info("  Step 2: Retroactive re-verification ...")
        retroverify_stats = retroactive_reverification(pipeline, cycle_num, config)

        # Step 3: Save retroverify stats alongside cycle results
        # mmlu_retention is included here so the resume path can reconstruct
        # mmlu_per_cycle without re-running the model.
        mmlu_val = cycle_result.mmlu_retention
        mmlu_serialisable = None if math.isnan(mmlu_val) else round(mmlu_val, 6)
        rv_path = output_dir / f"retroverify_cycle{cycle_num}.json"
        rv_payload = {
            "cycle": cycle_num,
            "retroverify": retroverify_stats,
            "fine_tune": {
                "n_episodes_used": cycle_result.n_episodes_used,
                "n_general_used": cycle_result.n_general_used,
                "mmlu_retention_ratio": cycle_result.mmlu_retention_ratio,
                "aborted": cycle_result.aborted,
                "final_train_loss": cycle_result.final_train_loss,
                "mmlu_retention": mmlu_serialisable,   # EXP-MMLU-FIX: neutral reporting metric
            },
        }
        # Atomic write: write to temp file then rename so a crash mid-write
        # never produces a truncated JSON that would break --resume_from_cycle.
        # tempfile and os are now at module top; no per-iteration aliased import.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output_dir, suffix=".tmp", delete=False, encoding="utf-8"
        ) as _tmp:
            json.dump(rv_payload, _tmp, indent=2)
            _tmp_path = _tmp.name
        os.replace(_tmp_path, rv_path)  # atomic on same filesystem (POSIX + Windows)

        # Track per-cycle MMLU for summary CSV
        mmlu_per_cycle.append(mmlu_val)

        # Step 4: Populate Memory by answering the SIL Train split (DISJOINT
        # from the calibration slice) with store_to_memory=True. This occurs
        # so that we populate the EpisodicMemoryStore with the *upgraded*
        # fine-tuned model weights. Because this is for *generation*, we do
        # not care about the benchmark scores.
        #
        # Gap-5 isolation: we pass `train_samples`, NOT `sil_pool`, so
        # the calibration slice stays unseen by the generator during SIL.
        logger.info("  Step 4: Generating Episodic Memory from SIL Train split (cycle=%d) ...", cycle_num)
        harness.run_all(train_samples, cycle=cycle_num, store_to_memory=True)

        # Step 5: Evaluate all benchmarks (Dev/Transfer split), NO Memory Leakage
        logger.info("  Step 5: Evaluating all benchmarks (cycle=%d) ...", cycle_num)
        cycle_results = harness.run_all(eval_samples, cycle=cycle_num, store_to_memory=False)
        all_cycle_results.append(cycle_results)
        
        # Step 6: Save memory checkpoint for resuming
        pipeline.memory_store.save(str(output_dir / f"memory_store_cycle_{cycle_num}"))

        # Step 6b: Save deferred buffer checkpoint so --resume_from_cycle
        # picks up any held-but-not-yet-promoted entries rather than starting
        # the next cycle with an empty buffer (which would silently discard
        # the current cycle's deferred signal).
        _def_buf = getattr(pipeline, "deferred_buffer", None)
        if _def_buf is not None:
            try:
                _def_buf.save(
                    str(output_dir / f"deferred_buffer_cycle_{cycle_num}.pkl")
                )
            except Exception as exc:
                logger.warning(
                    "Failed to save deferred buffer at cycle %d (%s).",
                    cycle_num, exc,
                )

        # Step 7: Per-cycle recalibration (Conservative default: temperature
        # only; no-op if both calibration flags are False). Runs after
        # fine-tuning + retroverify so the calibration set is scored with
        # the updated weights — matches the Cycle-0 protocol position
        # (calibration immediately follows eval). See Ovadia et al.
        # NeurIPS 2019 / Thulasidasan et al. 2019 for the motivation.
        if not getattr(ns, "skip_calibration", False):
            run_per_cycle_recalibration(
                pipeline, calib_samples, config, output_dir, cycle=cycle_num,
            )

        logger.info("Cycle %d done in %.1f min.", cycle_num, (time.time() - t0) / 60)

        # -------------------------------------------------------------- #
        # Equilibrium-prediction + triple-signal early-stop gate
        # (Sun et al. ICLR 2026 exponential-saturation form). See
        # caem/eval/equilibrium.py and Ch4 Corollary C4.
        # -------------------------------------------------------------- #
        # CES proxy: mean EM across eval benchmarks this cycle.
        # storage-rate proxy: mean of the per-benchmark storage_rate
        # fields written by the harness. mmlu_retention already lives
        # in mmlu_per_cycle (appended above).
        try:
            _em_vals = [float(r.get("em", 0.0)) for r in cycle_results.values()]
            _sr_vals = [float(r.get("storage_rate", 0.0)) for r in cycle_results.values()]
            mean_em = sum(_em_vals) / len(_em_vals) if _em_vals else 0.0
            mean_sr = sum(_sr_vals) / len(_sr_vals) if _sr_vals else 0.0
        except Exception as exc:
            logger.warning(
                "equilibrium hook: failed to extract per-cycle signals (%s); "
                "gate will skip this cycle.", exc,
            )
            mean_em = float("nan")
            mean_sr = float("nan")

        if not math.isnan(mean_em):
            ces_history.append(mean_em)
        if not math.isnan(mean_sr):
            storage_rates.append(mean_sr)

        # Only run the gate after the burn-in and when caller requested it.
        if (
            getattr(ns, "early_stop_enable", True)
            and cycle_num >= ns.early_stop_min_cycles
            and len(ces_history) >= 3
        ):
            from caem.eval.equilibrium import (
                fit_ces_saturation,
                predict_ceq_star,
                three_signal_gate,
            )
            _fit = None
            _c_star = None
            try:
                _fit = fit_ces_saturation(ces_history)
                _c_star = predict_ceq_star(_fit, eps=ns.early_stop_eps)
            except Exception as exc:
                logger.info(
                    "equilibrium fit unavailable at cycle %d (%s); relying "
                    "on non-parametric signals only.", cycle_num, exc,
                )

            _decision = three_signal_gate(
                ces_history=ces_history,
                storage_rates=storage_rates,
                mmlu_retentions=mmlu_per_cycle,
                ces_eps=ns.early_stop_ces_eps,
                storage_frac=ns.early_stop_storage_frac,
                mmlu_floor=config.forgetting_tolerance,
                signals_required=ns.early_stop_signals_required,
                cycle=cycle_num,
            )

            # Persist an audit artefact so the thesis can cite the exact
            # fit + gate state at each cycle boundary.
            try:
                (output_dir / f"cycle_{cycle_num}").mkdir(parents=True, exist_ok=True)
                _artefact = {
                    "cycle": int(cycle_num),
                    "ces_history": [float(x) for x in ces_history],
                    "storage_rates": [float(x) for x in storage_rates],
                    "mmlu_retentions": [
                        (None if math.isnan(float(v)) else float(v))
                        for v in mmlu_per_cycle
                    ],
                    "fit": (_fit.as_dict() if _fit is not None else None),
                    "c_star": (None if _c_star is None else float(_c_star)),
                    "decision": _decision.as_dict(),
                }
                with open(
                    output_dir / f"cycle_{cycle_num}" / "equilibrium_fit.json",
                    "w", encoding="utf-8",
                ) as _eq_f:
                    json.dump(_artefact, _eq_f, indent=2)
            except Exception as exc:
                logger.warning(
                    "equilibrium hook: failed to persist artefact (%s)", exc,
                )

            logger.info(
                "equilibrium gate | cycle=%d | signals=%s | c_star=%s | stop=%s",
                cycle_num,
                ",".join(_decision.signals_fired) or "none",
                f"{_c_star:.2f}" if _c_star is not None else "n/a",
                _decision.should_stop,
            )

            if _decision.should_stop:
                logger.info(
                    "EARLY_STOP cycle=%d signals=%s (2-of-3 gate fired); "
                    "skipping remaining %d cycle(s).",
                    cycle_num, _decision.signals_fired,
                    config.num_cycles - cycle_num,
                )
                break

    # -- Summary ------------------------------------------------------------- #
    save_summary_csv(all_cycle_results, output_dir, mmlu_per_cycle=mmlu_per_cycle)
    print_mechanism_table(all_cycle_results)

    # -- Save full results JSON ----------------------------------------------- #
    full_results_path = output_dir / "all_cycle_results.json"
    with open(full_results_path, "w") as f:
        json.dump(all_cycle_results, f, indent=2)
    logger.info("Full results saved -> %s", full_results_path)

    # -- Chapter 5 table split + per_sample_signals.jsonl (Phase 4m.4) -- #
    # Post-processes every per-benchmark-per-cycle JSON the harness wrote
    # during this run into the CSVs Chapter 5 references plus a flat
    # per-sample JSONL. Safe to re-run in isolation via ``build_ch5_tables``
    # if the CSVs need regenerating without re-running the cycles.
    try:
        from eval.reporting import build_ch5_tables
        manifest = build_ch5_tables(output_dir, mmlu_per_cycle=mmlu_per_cycle)
        if manifest:
            logger.info(
                "Chapter-5 tables written: %s",
                ", ".join(sorted(manifest.keys())),
            )
    except Exception as exc:
        logger.warning(
            "build_ch5_tables failed (%s). Per-cycle JSONs are still on disk "
            "-- rerun eval.reporting.build_ch5_tables(output_dir) post-hoc.",
            exc,
        )

    logger.info("Experiment complete. Outputs in: %s", output_dir)
    logger.info(
        "Next steps:\n"
        "  1. Run scripts/run_purity_validation.py for Theory 1/2/3 validation.\n"
        "  2. Run baseline + ablation variants via scripts/run_baseline.py\n"
        "     and scripts/run_ablation.py (unified-verifier migration complete).\n"
        "  3. Update caem-implementation-log.md with experiment results.\n"
        "  4. Feed outputs/ into Chapter 5 writing (use paper-writing skill)."
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the CAEM experiment orchestrator."""
    p = argparse.ArgumentParser(
        description="CAEM Experiment Orchestrator -- runs Cycle 0..N.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to write checkpoints, eval JSONs, and the summary CSV.",
    )
    p.add_argument(
        "--num_cycles",
        type=int,
        default=10,
        help="Number of SIL cycles to run after Cycle 0 baseline.",
    )
    p.add_argument(
        "--n_questions",
        type=int,
        default=5000,
        help=(
            "Per-benchmark SIL training-pool size. When --n_eval_questions is "
            "omitted, the same value is used for the evaluation pool."
        ),
    )
    p.add_argument(
        "--n_eval_questions",
        type=int,
        default=None,
        help=(
            "Optional per-benchmark size for the evaluation pool "
            "(purity + calibration + eval). Defaults to --n_questions when "
            "omitted. Splitting these lets you run a small smoke eval on top "
            "of a large SIL pool, or vice versa."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Global RNG seed for deterministic shuffles (split_calibration_sets, "
            "SIL pool sampling, etc.). The MMLU retention probe uses its own "
            "fixed seed for cross-run comparability."
        ),
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=[
            "fever",
            "triviaqa",
            "natural_questions",
            "truthfulqa",
            "strategyqa",
            "arc_challenge",
        ],
        help="Benchmarks to evaluate each cycle (Dev + Transfer split).",
    )
    p.add_argument(
        "--passage_index",
        type=str,
        default="data/passage_index",
        help="Path to pre-built Wikipedia FAISS index (for Tier 3 RAG).",
    )
    p.add_argument(
        "--cold_start_memory",
        type=str,
        default=None,
        help="Optional path to a pre-seeded EpisodicMemoryStore (e.g. from seed_cold_start.py).",
    )
    p.add_argument(
        "--resume_from_cycle",
        type=int,
        default=0,
        help="Cycle number to resume from (0 = start fresh). Requires prior checkpoints.",
    )
    p.add_argument(
        "--skip_calibration",
        action="store_true",
        help="Skip temperature scaling + signal-weight fitting (use equal initial weights).",
    )
    # ------------------------------------------------------------------
    # Early-stop gate on the self-improvement cycle loop.
    #
    # Fits the Sun et al. (ICLR 2026) exponential-saturation form
    # CES(c) = C_inf - A * exp(-k * c) to the per-cycle eval trajectory
    # and combines the parametric CES gradient with two CAEM-native
    # non-parametric signals (storage-rate saturation + MMLU-retention
    # ceiling) to decide whether further cycles are worth the compute.
    # See caem/eval/equilibrium.py for the module. Gate is ON by default
    # to honour the Plan-B1 budget; pass --no_early_stop to force a
    # full num_cycles run (e.g., for the cycle-progression plot when
    # equilibrium is not expected to fire).
    # ------------------------------------------------------------------
    p.add_argument(
        "--no_early_stop",
        dest="early_stop_enable",
        action="store_false",
        help="Disable the triple-signal early-stop gate (default: gate active).",
    )
    p.set_defaults(early_stop_enable=True)
    p.add_argument(
        "--early_stop_min_cycles",
        type=int,
        default=5,
        help="Minimum cycles to run before the gate can fire. Below this, "
             "the exponential fit is too noisy to trust.",
    )
    p.add_argument(
        "--early_stop_eps",
        type=float,
        default=0.01,
        help="predict_ceq_star tolerance: fraction of C_inf below which we "
             "consider the residual gap negligible.",
    )
    p.add_argument(
        "--early_stop_ces_eps",
        type=float,
        default=0.002,
        help="CES gradient threshold for signal 1. Two consecutive cycles "
             "both below this trigger the gradient signal.",
    )
    p.add_argument(
        "--early_stop_storage_frac",
        type=float,
        default=0.05,
        help="Storage-rate threshold for signal 2 (memory saturation).",
    )
    p.add_argument(
        "--early_stop_signals_required",
        type=int,
        default=2,
        help="Signals that must fire (of 3) to break the cycle loop.",
    )
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use synthetic n=10 samples per benchmark for fast CI/local sanity checks.",
    )
    p.add_argument(
        "--verifier_backend",
        type=str,
        default=None,
        choices=["minicheck", "roberta_nli"],
        help=(
            "Verifier judge backend. 'minicheck' (default, "
            "lytang/MiniCheck-Flan-T5-Large) is the thesis main path; "
            "'roberta_nli' is the legacy ablation path. When omitted, "
            "CAEMConfig.verifier_backend is used."
        ),
    )
    p.add_argument(
        "--store_threshold",
        type=float,
        default=None,
        help=(
            "Override CAEMConfig.store_threshold (tau_store). Set this to "
            "the value fitted by scripts/calibrate_thresholds.py at the "
            "Cycle 0 boundary (typically ~0.40-0.50 under MiniCheck, vs "
            "the RoBERTa-era default 0.65). See Chapter 4 Sec Threshold "
            "calibration. When omitted, CAEMConfig default is used."
        ),
    )
    p.add_argument(
        "--defer_threshold",
        type=float,
        default=None,
        help=(
            "Override CAEMConfig.defer_threshold (tau_defer). Set this "
            "from scripts/calibrate_thresholds.py output (typically "
            "~0.25-0.40 under MiniCheck). Must satisfy "
            "defer < store < train."
        ),
    )
    p.add_argument(
        "--train_threshold",
        type=float,
        default=None,
        help=(
            "Override CAEMConfig.min_u_stored_for_training (tau_train). "
            "Set this from scripts/calibrate_thresholds.py output "
            "(typically ~0.50-0.65 under MiniCheck). Must satisfy "
            "train > store > defer."
        ),
    )
    return p.parse_args()


def main() -> None:
    """Entry point: parse CLI args and kick off the experiment."""
    ns = _parse_args()
    run_experiment(ns)


if __name__ == "__main__":
    main()
