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
  OR the --no_rag flag will disable Tier 3 RAG (Tier 2 escalation falls back
  to a Tier 3 without RAG context -- accuracy lower but experiment still runs).
- All projected targets are from the unified plan; actual results may differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
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
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from caem.config import CAEMConfig
    from caem.memory.encoder import QueryEncoder
    from caem.memory.store import EpisodicMemoryStore
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
        AutoTokenizer=AutoTokenizer,
        T5ForConditionalGeneration=T5ForConditionalGeneration,
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

    # -- Flan-T5-Large ----------------------------------------------------- #
    logger.info("Loading Flan-T5-Large ...")
    torch = m["torch"]
    tokenizer = m["AutoTokenizer"].from_pretrained("google/flan-t5-large")
    model = m["T5ForConditionalGeneration"].from_pretrained("google/flan-t5-large")
    if hw.use_bf16:
        model = model.to(torch.bfloat16)
    elif hw.use_fp16:
        model = model.to(torch.float16)
    model = model.to(device).eval()
    logger.info("Flan-T5-Large loaded (%.0f M params, precision=%s)",
                sum(p.numel() for p in model.parameters()) / 1e6,
                "bf16" if hw.use_bf16 else "fp16" if hw.use_fp16 else "fp32")

    # -- SBERT encoder ------------------------------------------------------ #
    logger.info("Loading SBERT encoder (all-mpnet-base-v2) ...")
    encoder = m["QueryEncoder"](model_name=config.sbert_model, device=device)

    # -- NLI model ---------------------------------------------------------- #
    nli_model, nli_tokenizer = None, None
    if not ns.no_nli:
        try:
            from transformers import AutoModelForSequenceClassification
            logger.info("Loading RoBERTa-Large-MNLI ...")
            nli_tokenizer = m["AutoTokenizer"].from_pretrained(
                "roberta-large-mnli"
            )
            nli_model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-large-mnli"
            ).to(device)
            nli_model.eval()
            logger.info("NLI model loaded.")
        except Exception as exc:
            logger.warning("NLI model load failed (%s); continuing without NLI.", exc)

    # -- Passage store (Wikipedia FAISS index) ------------------------------ #
    passage_store = None
    if not ns.no_rag:
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

    pipeline = m["CAEMPipeline"](
        model=model,
        tokenizer=tokenizer,
        encoder=encoder,
        nli_model=nli_model,
        nli_tokenizer=nli_tokenizer,
        passage_store=passage_store,
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
    """Load Evaluation data (zero data leakage)."""
    n = ns.n_questions
    samples = {}
    benchmarks = [bm.strip().lower() for bm in ns.benchmarks]
    split_map = {
        "fever": "paper_dev",        # lucadiliello/fever eval split (data-leakage safe)
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
) -> tuple:
    """Split each benchmark's samples into purity / calibration / eval sets.

    Returns
    -------
    purity_samples  : dict[bm -> first purity_size (or half of available)]
    calib_samples   : dict[bm -> next calib_size (or quarter of available)]
    eval_samples    : dict[bm -> remainder]

    These sets are non-overlapping -- critical for theory validation (§5.3).

    Notes
    -----
    TruthfulQA has only 817 questions total. Using fixed 500+500 would leave
    zero samples for eval. For benchmarks where total < purity_size + calib_size,
    we scale splits proportionally (plan §5.3 specifies 250/250/317 for TruthfulQA).
    """
    purity, calib, evl = {}, {}, {}
    for bm, slist in samples.items():
        total = len(slist)
        effective_purity = purity_size
        effective_calib = calib_size

        # Scale down proportionally when the dataset is too small to fit both
        # purity + calibration windows (e.g. TruthfulQA: 817 < 500+500).
        if total < purity_size + calib_size:
            # Use ≈30% for purity, ≈30% for calibration, ≈40% for eval --
            # roughly matching the 250/250/317 ratio the plan specifies.
            effective_purity = total // 3
            effective_calib = total // 3
            logger.warning(
                "  %s: only %d samples -- scaling splits to "
                "%d purity / %d calibration / %d eval (plan §5.3)",
                bm, total, effective_purity, effective_calib,
                total - effective_purity - effective_calib,
            )

        purity[bm] = slist[:effective_purity]
        calib[bm] = slist[effective_purity: effective_purity + effective_calib]
        evl[bm] = slist[effective_purity + effective_calib:]
        logger.info(
            "  %s: %d purity | %d calibration | %d eval",
            bm, len(purity[bm]), len(calib[bm]), len(evl[bm]),
        )
    return purity, calib, evl


# -----------------------------------------------------------------------------
# General-domain data (anti-forgetting mix for fine-tuning)
# -----------------------------------------------------------------------------

def load_general_data(n: int = 1000) -> list:
    """Load general QA pairs for the 10% anti-forgetting data mix.

    Uses a small subset of TriviaQA or a synthetic fallback.
    The SelfImprovementLoop mixes 10% of these into the training set
    to satisfy the forgetting_tolerance ≥ 0.93 constraint (§4.6).
    """
    from caem.training.self_improvement import QAPair

    try:
        from datasets import load_dataset
        logger.info("Loading TriviaQA for general-domain mix ...")
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        pairs = []
        for item in ds.select(range(min(n, len(ds)))):
            row = cast(Mapping[str, Any], item)
            q = str(row.get("question", ""))
            answer_obj = cast(Mapping[str, Any], row.get("answer", {}))
            ans = str(answer_obj.get("value", "") or "")
            if q and ans:
                pairs.append(QAPair(question=q, answer=ans))
        logger.info("  General-domain mix: %d QA pairs loaded.", len(pairs))
        return pairs
    except Exception as exc:
        logger.warning(
            "TriviaQA load failed (%s); using 200 synthetic pairs.", exc
        )
        # Minimal synthetic fallback so the loop still runs
        return [
            QAPair(question=f"What is {i} + {i}?", answer=str(i * 2))
            for i in range(200)
        ]


# -----------------------------------------------------------------------------
# Calibration (temperature scaling + signal weights)
# -----------------------------------------------------------------------------

def run_calibration_step(
    pipeline,
    calib_samples: Dict[str, list],
    config,
    output_dir: Path,
    m,
) -> None:
    """Fit temperature scalar T and signal weights on calibration set.

    Calls run_calibration.py logic inline so the main loop stays clean.
    Results are written to outputs/calibration/calibrated_config.json
    and the config object is updated in place.

    Called once after Cycle 0 evaluation, before Cycle 1 fine-tuning.
    """
    logger.info("-" * 60)
    logger.info("CALIBRATION -- fitting temperature scalar and signal weights")
    logger.info("-" * 60)

    calib_dir = output_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    try:
        from scripts.run_calibration import calibrate_pipeline
        calibrate_pipeline(pipeline, calib_samples, config, calib_dir)
    except Exception as exc:
        logger.warning(
            "Calibration step failed (%s). "
            "Continuing with initial equal signal weights (0.25 each). "
            "Calibrated values will not be reported in Chapter 5 -- "
            "re-run scripts/run_calibration.py manually after Cycle 0.",
            exc,
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

    # verify_fn: (EpisodicEntry) -> StoredConfidence
    # Uses the pipeline's verifier so re-verification benefits from the
    # updated model weights after this cycle's fine-tuning.
    verify_fn = lambda e: pipeline.verifier.verify(e.question, e.answer)

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

    Columns: Cycle, HallucReduction%, Tier1Frac%, Tier3Frac%,
             MeanUStored, MMULRetention%

    Parameters
    ----------
    all_cycle_results : list[dict]
        One entry per cycle; each entry maps benchmark -> eval metrics dict.
    output_dir : Path
        Where to write experiment_summary.csv.
    mmlu_per_cycle : list[float] or None
        Per-cycle MMLU retention values (0-1 or NaN).  Index 0 corresponds to
        Cycle 0 (baseline, always NaN -- no fine-tuning occurred).  When None,
        the column is omitted with a backwards-compatible empty string.

    Notes
    -----
    MMLU retention is a single value per cycle (not per benchmark).  To keep
    the CSV tidy it is written only on the first benchmark row of each cycle;
    all other rows for that cycle leave the field blank.  This matches how
    MMLU is reported in Chapter 5 Table 5.2 (one row per cycle, aggregate).
    """
    import math
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
        "mmlu_retention_pct",   # populated on first-benchmark row only; blank for subsequent rows
    ]

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
            else:
                mmlu_str = ""   # leave blank for subsequent BM rows (or if NaN)
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
                "mmlu_retention_pct": mmlu_str,
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Summary CSV saved -> %s", csv_path)


def print_mechanism_table(all_cycle_results: List[Dict]) -> None:
    """Print the five-mechanism evidence table to stdout (Chapter 5, Table 1).

    Targets (from 10-cycle final plan):
      Tier 1 fraction: ~5% (C0) -> growing per cycle -> ~50% at equilibrium (C7-C9)
      Hallucination reduction: growing per cycle; practical equilibrium expected at C7-C9
      MMLU Retention: >=93% across all cycles (requires separate MMLU eval, not run here)
      Mean u_stored: rising monotonically across cycles (retroactive re-verification working)
      Data Purity: rising as low-quality episodes are pruned and high-quality ones dominate
    """
    print("\n" + "=" * 90)
    print("MECHANISM EVIDENCE TABLE  (Chapter 5, Table 1)")
    print("=" * 90)
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
                f"  {cycle_num:<4} {bm:<12} "
                f"{em:>6.3f} {res.get('f1', 0.0):>6.3f} "
                f"{hall_red:>+8.1f}% "
                f"{res.get('tier1_frac', 0.0)*100:>5.1f}% "
                f"{res.get('tier3_frac', 0.0)*100:>5.1f}% "
                f"{res.get('storage_rate', 0.0)*100:>7.1f}% "
                f"{res.get('mean_u_stored', 0.0):>7.4f}"
            )
    print("=" * 90)
    print("Note: HallRed% = EM improvement vs Cycle 0 (positive = better).")
    print("MMLU Retention% must be measured separately via scripts/run_ablation.py.\n")


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

    # -- Config ------------------------------------------------------------- #
    config = m["CAEMConfig"]()
    # Override n_questions if specified
    config.questions_per_cycle = ns.n_questions
    config.num_cycles = ns.num_cycles

    if ns.resume_from_cycle < 0 or ns.resume_from_cycle > config.num_cycles:
        raise ValueError(
            f"--resume_from_cycle must be between 0 and {config.num_cycles}, "
            f"got {ns.resume_from_cycle}."
        )

    # -- Build pipeline ------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("CAEM EXPERIMENT -- Full Scale Run (10 cycles, 1M episodes, full DPR Wikipedia)")
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
        sil_samples = {bm: make_synthetic_samples(bm, n=10) for bm in sil_benchmarks}
        eval_samples = {bm: make_synthetic_samples(bm, n=10) for bm in requested}
        
        purity_samples = {bm: s[:5] for bm, s in sil_samples.items()}
        calib_samples = {bm: s[5:] for bm, s in sil_samples.items()}
    else:
        sil_samples = load_sil_training_pool(ns, m)
        eval_samples = load_eval_transfer_pool(ns, m)
        
        # Purity and Calibration are carved securely out of the SIL Train Pool
        purity_samples, calib_samples, _ = split_calibration_sets(
            sil_samples,
            calib_size=config.calibration_set_size,
            purity_size=config.purity_validation_set_size,
        )

    # -- Save purity/calibration sample IDs (for reproducibility) ---------- #
    meta_path = output_dir / "dataset_splits.json"
    with open(meta_path, "w") as f:
        meta_out = {}
        for bm in eval_samples:
            meta_out[bm] = {"eval_ids": [s.get("id", i) for i, s in enumerate(eval_samples[bm])]}
            if bm in purity_samples:
                meta_out[bm]["purity_ids"] = [s.get("id", i) for i, s in enumerate(purity_samples[bm])]
                meta_out[bm]["calib_ids"] = [s.get("id", i) for i, s in enumerate(calib_samples[bm])]
        json.dump(meta_out, f, indent=2)
    logger.info("Dataset split metadata saved -> %s", meta_path)

    # -- General-domain data (for anti-forgetting mix) ---------------------- #
    general_data = load_general_data(n=1000)

    # -- Eval harness -------------------------------------------------------- #
    harness = m["EvalHarness"](pipeline, output_dir=str(eval_dir), log_every=100)

    # -- Self-improvement loop ----------------------------------------------- #
    sil = m["SelfImprovementLoop"](
        model=pipeline.model,
        tokenizer=pipeline.tokenizer,
        config=config,
        output_dir=str(output_dir),
    )

    # Per-cycle MMLU retention list (index 0 = Cycle 0 = NaN, no fine-tuning).
    # Populated after each SIL fine-tuning step; passed to save_summary_csv().
    mmlu_per_cycle: List[float] = []

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

        all_cycle_results = [cycle0_results]
        mmlu_per_cycle.append(float("nan"))  # Cycle 0: no fine-tuning, no MMLU measurement

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
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cycle_results[bm] = data["meta"]
            all_cycle_results.append(cycle_results)

            # Reconstruct mmlu_per_cycle from saved retroverify JSONs.
            # Cycle 0 never has a retroverify file (no fine-tuning), so use NaN.
            if c == 0:
                mmlu_per_cycle.append(float("nan"))
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
        logger.info("Restored memory store from Cycle %d: %d episodes", prev_cycle, pipeline.memory_store.size)
        
        # Reload fine-tuned model weights if past cycle 0
        if prev_cycle > 0:
            sil.load_checkpoint(prev_cycle)
            logger.info("Restored fine-tuned model weights from Cycle %d.", prev_cycle)
            
        pipeline.current_cycle = prev_cycle

    # -- CYCLES 1..N --------------------------------------------------------- #
    start_cycle = max(1, ns.resume_from_cycle)
    for cycle_num in range(start_cycle, config.num_cycles + 1):
        logger.info("-" * 60)
        logger.info("CYCLE %d -- Fine-tuning + evaluation", cycle_num)
        logger.info("-" * 60)
        t0 = time.time()

        # Step 1: Fine-tune on verified episodes from current memory
        logger.info("  Step 1: SelfImprovementLoop.run_cycle(%d) ...", cycle_num)
        cycle_result = sil.run_cycle(
            cycle_num=cycle_num,
            memory_store=pipeline.memory_store,
            general_data=general_data,
        )

        if cycle_result.aborted:
            logger.warning(
                "  Cycle %d ABORTED (retention ratio %.3f < %.3f). "
                "Weights restored. Eval still runs on the restored model.",
                cycle_num, cycle_result.forgetting_score, config.forgetting_tolerance,
            )
        else:
            logger.info(
                "  Fine-tuning done: %d episodes | retention_ratio=%.3f | loss=%.4f",
                cycle_result.n_episodes_used,
                cycle_result.forgetting_score,
                cycle_result.final_train_loss,
            )

        # Step 2: Update pipeline's cycle counter
        pipeline.current_cycle = cycle_num

        # Step 3: Retroactive re-verification of memory
        if getattr(ns, "disable_reverification", False):
            logger.info("  Step 2: Skipped retroactive re-verification (--disable_reverification set)")
            retroverify_stats = {}
        else:
            logger.info("  Step 2: Retroactive re-verification ...")
            retroverify_stats = retroactive_reverification(pipeline, cycle_num, config)

        # Step 4: Save retroverify stats alongside cycle results
        # mmlu_retention is included here so the resume path can reconstruct
        # mmlu_per_cycle without re-running the model.
        import math as _math
        mmlu_val = cycle_result.mmlu_retention
        mmlu_serialisable = None if _math.isnan(mmlu_val) else round(mmlu_val, 6)
        rv_path = output_dir / f"retroverify_cycle{cycle_num}.json"
        with open(rv_path, "w") as f:
            json.dump({
                "cycle": cycle_num,
                "retroverify": retroverify_stats,
                "fine_tune": {
                    "n_episodes_used": cycle_result.n_episodes_used,
                    "n_general_used": cycle_result.n_general_used,
                    "forgetting_score": cycle_result.forgetting_score,
                    "aborted": cycle_result.aborted,
                    "final_train_loss": cycle_result.final_train_loss,
                    "mmlu_retention": mmlu_serialisable,   # EXP-MMLU-FIX: neutral reporting metric
                },
            }, f, indent=2)

        # Track per-cycle MMLU for summary CSV
        mmlu_per_cycle.append(mmlu_val)

        # Step 4.5: Populate Memory by answering SIL Pool (Train split) with store_to_memory=True
        # This occurs so that we populate the EpisodicMemoryStore with the *upgraded* fine-tuned
        # model weights. Because this is for *generation*, we do not care about the benchmark scores.
        logger.info("  Step 2.5: Generating Episodic Memory from SIL Pool (cycle=%d) ...", cycle_num)
        harness.run_all(sil_samples, cycle=cycle_num, store_to_memory=True)

        # Step 5: Evaluate all benchmarks (Dev/Transfer split), NO Memory Leakage
        logger.info("  Step 3: Evaluating all benchmarks (cycle=%d) ...", cycle_num)
        cycle_results = harness.run_all(eval_samples, cycle=cycle_num, store_to_memory=False)
        all_cycle_results.append(cycle_results)
        
        # Step 6: Save memory checkpoint for resuming
        pipeline.memory_store.save(str(output_dir / f"memory_store_cycle_{cycle_num}"))

        logger.info("Cycle %d done in %.1f min.", cycle_num, (time.time() - t0) / 60)

    # -- Summary ------------------------------------------------------------- #
    save_summary_csv(all_cycle_results, output_dir, mmlu_per_cycle=mmlu_per_cycle)
    print_mechanism_table(all_cycle_results)

    # -- Save full results JSON ----------------------------------------------- #
    full_results_path = output_dir / "all_cycle_results.json"
    with open(full_results_path, "w") as f:
        json.dump(all_cycle_results, f, indent=2)
    logger.info("Full results saved -> %s", full_results_path)

    logger.info("Experiment complete. Outputs in: %s", output_dir)
    logger.info(
        "Next steps:\n"
        "  1. Run scripts/run_purity_validation.py for Theory 1/2/3 validation.\n"
        "  2. Run scripts/run_ablation.py for baselines and ablation variants.\n"
        "  3. Update caem-implementation-log.md with experiment results.\n"
        "  4. Feed outputs/ into Chapter 5 writing (use paper-writing skill)."
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAEM Experiment Orchestrator (Cycle 0->N, default N=10)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output_dir", default="outputs",
        help="Root output directory for checkpoints and eval results.",
    )
    p.add_argument(
        "--n_questions", type=int, default=5000,
        help="Questions per benchmark (full run = 5000; smoke test sets this to 10).",
    )
    p.add_argument(
        "--num_cycles", type=int, default=10,
        help="Number of self-improvement cycles to run after Cycle 0 baseline.",
    )
    p.add_argument(
        "--benchmarks", nargs="+",
        default=["truthfulqa", "strategyqa", "fever", "triviaqa", "natural_questions", "arc_challenge"],
        help="Which benchmarks to evaluate (for the evaluation metrics).",
    )
    p.add_argument(
        "--passage_index", default="data/passage_index",
        help="Path to pre-built Wikipedia FAISS passage index (for Tier 3 RAG).",
    )
    p.add_argument(
        "--cold_start_memory", type=str, default=None,
        help="Path to pre-seeded episodic memory index (from seed_cold_start.py).",
    )
    p.add_argument(
        "--no_rag", action="store_true",
        help="Disable Tier 3 RAG (run without passage index).",
    )
    p.add_argument(
        "--no_nli", action="store_true",
        help="Disable NLI model (faster but lower verification quality).",
    )
    p.add_argument(
        "--skip_calibration", action="store_true",
        help="Skip temperature scaling + signal weight calibration after Cycle 0.",
    )
    p.add_argument(
        "--smoke_test", action="store_true",
        help="Tiny synthetic run to verify the pipeline is wired correctly (no GPU needed).",
    )
    p.add_argument(
        "--disable_reverification", action="store_true",
        help="Disable retroactive re-verification between cycles (for AB4 ablation).",
    )
    p.add_argument(
        "--resume_from_cycle", type=int, default=0,
        help="Resume experiment from a specific cycle (1 to --num_cycles). Bypasses earlier cycles and reloads memory/weights.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(args)
