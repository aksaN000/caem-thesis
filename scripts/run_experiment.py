"""
scripts/run_experiment.py
==========================
CAEM Experiment Orchestrator — Cycle 0 → Cycle 3

Runs the full self-improvement experiment loop:

  Cycle 0  — baseline evaluation (zero episodic memory, no fine-tuning)
  Calibration — temperature scaling + signal weight fitting on 500-sample set
  Cycle 1  — SelfImprovementLoop → evaluate all 4 benchmarks
  Cycle 2  — SelfImprovementLoop → evaluate all 4 benchmarks
  Cycle 3  — SelfImprovementLoop → evaluate all 4 benchmarks

All results are saved as JSON to outputs/eval/. Cycle checkpoints are saved
to outputs/cycle_{n}/. A summary CSV is written to outputs/experiment_summary.csv
for Chapter 5 table generation.

Usage
-----
From the repo root (requires A100 GPU, ~15-19 GB VRAM):

    python -m scripts.run_experiment \\
        --output_dir outputs \\
        --n_questions 500 \\
        --benchmarks hotpotqa truthfulqa fever strategyqa

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
  to a Tier 3 without RAG context — accuracy lower but experiment still runs).
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
from typing import Dict, List, Optional

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


# ── Guard: check torch/transformers before importing heavy modules ───────────
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


# ── Imports (after dep check) ────────────────────────────────────────────────
def _load_imports():
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
        load_hotpotqa,
        load_strategyqa,
        load_truthfulqa,
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
        load_hotpotqa=load_hotpotqa,
        load_truthfulqa=load_truthfulqa,
        load_fever=load_fever,
        load_strategyqa=load_strategyqa,
        EvalHarness=EvalHarness,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model initialisation
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(config, ns, m) -> "CAEMPipeline":
    """Load Flan-T5-Large, SBERT encoder, NLI model, and passage store.

    Parameters
    ----------
    config : CAEMConfig
    ns     : argparse.Namespace — parsed CLI args
    m      : dict — module namespace from _load_imports()

    Returns
    -------
    CAEMPipeline — ready for inference
    """
    from scripts.hardware import print_hardware_summary, apply_memory_flags
    hw = print_hardware_summary()
    apply_memory_flags(hw)
    device = hw.device
    logger.info("Device: %s | GPU: %s | VRAM: %.1f GB",
                device, hw.gpu_name or "n/a", hw.vram_gb)

    # ── Flan-T5-Large ───────────────────────────────────────────────────── #
    logger.info("Loading Flan-T5-Large …")
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

    # ── SBERT encoder ────────────────────────────────────────────────────── #
    logger.info("Loading SBERT encoder (all-mpnet-base-v2) …")
    encoder = m["QueryEncoder"](config=config)

    # ── NLI model ────────────────────────────────────────────────────────── #
    nli_model, nli_tokenizer = None, None
    if not ns.no_nli:
        try:
            from transformers import AutoModelForSequenceClassification
            logger.info("Loading RoBERTa-Large-MNLI …")
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

    # ── Passage store (Wikipedia FAISS index) ────────────────────────────── #
    passage_store = None
    if not ns.no_rag:
        passage_index_path = Path(ns.passage_index)
        if passage_index_path.exists():
            logger.info("Loading passage index from %s …", passage_index_path)
            passage_store = m["PassageStore"].load(str(passage_index_path))
            logger.info("Passage store loaded (%d passages).", len(passage_store))
        else:
            logger.warning(
                "Passage index not found at %s — Tier 3 RAG disabled. "
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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets(ns, m) -> Dict[str, list]:
    """Load HotpotQA, TruthfulQA, FEVER, StrategyQA from HuggingFace.

    Returns a dict keyed by benchmark name.
    Samples are drawn from validation splits (not test — labels available).

    Calibration and purity validation sets are drawn from non-overlapping
    portions of each benchmark's validation split (§5.3 of the thesis plan).
    """
    n = ns.n_questions

    # Allocation (§5.3): first 500 = purity validation, 500–1000 = calibration,
    # 1000+ = experiment evaluation. If n < 1000, calibration overlaps eval —
    # only use for smoke testing.
    benchmarks = ns.benchmarks
    samples: Dict[str, list] = {}

    for bm in benchmarks:
        logger.info("Loading %s (n=%d) …", bm, n)
        if bm == "hotpotqa":
            samples[bm] = m["load_hotpotqa"](n=n)
        elif bm == "truthfulqa":
            samples[bm] = m["load_truthfulqa"](n=n)
        elif bm == "fever":
            samples[bm] = m["load_fever"](n=n)
        elif bm == "strategyqa":
            samples[bm] = m["load_strategyqa"](n=n)
        else:
            logger.warning("Unknown benchmark %s — skipping.", bm)
            continue
        logger.info("  %s: %d samples loaded.", bm, len(samples[bm]))

    return samples


def split_calibration_sets(
    samples: Dict[str, list],
    calib_size: int = 500,
    purity_size: int = 500,
) -> tuple:
    """Split each benchmark's samples into purity / calibration / eval sets.

    Returns
    -------
    purity_samples  : dict[bm → first 500]
    calib_samples   : dict[bm → next 500]
    eval_samples    : dict[bm → remainder]

    These sets are non-overlapping — critical for theory validation (§5.3).
    """
    purity, calib, evl = {}, {}, {}
    for bm, slist in samples.items():
        purity[bm] = slist[:purity_size]
        calib[bm] = slist[purity_size: purity_size + calib_size]
        evl[bm] = slist[purity_size + calib_size:]
        logger.info(
            "  %s: %d purity | %d calibration | %d eval",
            bm, len(purity[bm]), len(calib[bm]), len(evl[bm]),
        )
    return purity, calib, evl


# ─────────────────────────────────────────────────────────────────────────────
# General-domain data (anti-forgetting mix for fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

def load_general_data(n: int = 1000) -> list:
    """Load general QA pairs for the 10% anti-forgetting data mix.

    Uses a small subset of TriviaQA or a synthetic fallback.
    The SelfImprovementLoop mixes 10% of these into the training set
    to satisfy the forgetting_tolerance ≥ 0.93 constraint (§4.6).
    """
    from caem.training.self_improvement import QAPair

    try:
        from datasets import load_dataset
        logger.info("Loading TriviaQA for general-domain mix …")
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        pairs = []
        for item in ds.select(range(min(n, len(ds)))):
            q = item["question"]
            ans = item["answer"]["value"] if item["answer"]["value"] else ""
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


# ─────────────────────────────────────────────────────────────────────────────
# Calibration (temperature scaling + signal weights)
# ─────────────────────────────────────────────────────────────────────────────

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
    logger.info("─" * 60)
    logger.info("CALIBRATION — fitting temperature scalar and signal weights")
    logger.info("─" * 60)

    calib_dir = output_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)

    try:
        from scripts.run_calibration import calibrate_pipeline
        calibrate_pipeline(pipeline, calib_samples, config, calib_dir)
    except Exception as exc:
        logger.warning(
            "Calibration step failed (%s). "
            "Continuing with initial equal signal weights (0.25 each). "
            "Calibrated values will not be reported in Chapter 5 — "
            "re-run scripts/run_calibration.py manually after Cycle 0.",
            exc,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Retroactive re-verification
# ─────────────────────────────────────────────────────────────────────────────

def retroactive_reverification(pipeline, cycle: int, config) -> Dict:
    """Re-verify all stored episodes after fine-tuning.

    After each cycle the improved model may score stored episodes
    differently. Episodes that fall below retroverify_prune_threshold (0.50)
    are pruned; episodes that score higher get their u_stored upgraded.

    This is §4.7 (retroactive re-verification) of the thesis plan.

    Uses EpisodicMemoryStore.retroverify(verify_fn, threshold) — the store's
    own method that iterates entries, calls verify_fn on each, and handles
    pruning + u_stored upgrades atomically.

    Returns
    -------
    dict with keys: total, updated, pruned
    """
    store = pipeline.memory_store
    total_before = len(store.all_entries())

    logger.info("Retroactive re-verification: %d episodes …", total_before)

    # verify_fn: (EpisodicEntry) → StoredConfidence
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


# ─────────────────────────────────────────────────────────────────────────────
# Summary table (Chapter 5 mechanism evidence table)
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_csv(all_cycle_results: List[Dict], output_dir: Path) -> None:
    """Save the five-mechanism evidence table as a CSV (Table 1 in Chapter 5).

    Columns: Cycle, HallucReduction%, Tier1Frac%, Tier3Frac%,
             MeanUStored, (MMULRetention% — filled manually after MMLU eval)
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
    ]

    rows = []
    for cycle_num, cycle_results in enumerate(all_cycle_results):
        for bm, res in cycle_results.items():
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
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Summary CSV saved → %s", csv_path)


def print_mechanism_table(all_cycle_results: List[Dict]) -> None:
    """Print the five-mechanism evidence table to stdout (Chapter 5, Table 1).

    Targets (from thesis plan):
      Tier 1 fraction: ~5% (C0) → ~18% (C1) → ~30% (C2) → ~38% (C3)
      Hallucination reduction: +10% / +20% / +28% vs cycle 0
      MMLU Retention: ≥93% (requires separate MMLU eval, not run here)
      Mean û_stored: rising across cycles (retroactive re-verification working)
    """
    print("\n" + "═" * 90)
    print("MECHANISM EVIDENCE TABLE  (Chapter 5, Table 1)")
    print("═" * 90)
    print(
        f"{'Cycle':<6} {'BM':<12} {'EM':>6} {'F1':>6} "
        f"{'HallRed%':>9} {'T1%':>6} {'T3%':>6} "
        f"{'Stored%':>8} {'ûMean':>7}"
    )
    print("─" * 90)

    cycle0_em: Dict[str, float] = {}

    for cycle_num, cycle_results in enumerate(all_cycle_results):
        for bm, res in sorted(cycle_results.items()):
            em = res.get("em", 0.0)
            if cycle_num == 0:
                cycle0_em[bm] = em

            # Hallucination reduction relative to cycle 0
            base = cycle0_em.get(bm, em)
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
    print("═" * 90)
    print("Note: HallRed% = EM improvement vs Cycle 0 (positive = better).")
    print("MMLU Retention% must be measured separately via scripts/run_ablation.py.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(ns: argparse.Namespace) -> None:
    _check_deps()
    m = _load_imports()
    torch = m["torch"]

    output_dir = Path(ns.output_dir)
    eval_dir = output_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ───────────────────────────────────────────────────────────── #
    config = m["CAEMConfig"]()
    # Override n_questions if specified
    config.questions_per_cycle = ns.n_questions

    # ── Build pipeline ────────────────────────────────────────────────────── #
    logger.info("═" * 60)
    logger.info("CAEM EXPERIMENT — Session 21")
    logger.info("═" * 60)
    pipeline = build_pipeline(config, ns, m)

    # ── Load datasets ─────────────────────────────────────────────────────── #
    if ns.smoke_test:
        logger.info("SMOKE TEST MODE — using synthetic samples (n=10 per benchmark)")
        from eval.benchmarks import make_synthetic_samples
        all_samples = {bm: make_synthetic_samples(bm, n=10) for bm in ns.benchmarks}
        purity_samples = {bm: s[:5] for bm, s in all_samples.items()}
        calib_samples = {bm: s[5:] for bm, s in all_samples.items()}
        eval_samples = {bm: s for bm, s in all_samples.items()}
    else:
        all_samples = load_datasets(ns, m)
        purity_samples, calib_samples, eval_samples = split_calibration_sets(
            all_samples,
            calib_size=config.calibration_set_size,
            purity_size=config.purity_validation_set_size,
        )

    # ── Save purity/calibration sample IDs (for reproducibility) ────────── #
    meta_path = output_dir / "dataset_splits.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                bm: {
                    "purity_ids":  [s.get("id", i) for i, s in enumerate(purity_samples[bm])],
                    "calib_ids":   [s.get("id", i) for i, s in enumerate(calib_samples[bm])],
                    "eval_ids":    [s.get("id", i) for i, s in enumerate(eval_samples[bm])],
                }
                for bm in eval_samples
            },
            f, indent=2,
        )
    logger.info("Dataset split metadata saved → %s", meta_path)

    # ── General-domain data (for anti-forgetting mix) ────────────────────── #
    general_data = load_general_data(n=1000)

    # ── Eval harness ──────────────────────────────────────────────────────── #
    harness = m["EvalHarness"](pipeline, output_dir=str(eval_dir), log_every=100)

    # ── Self-improvement loop ─────────────────────────────────────────────── #
    sil = m["SelfImprovementLoop"](
        model=pipeline.model,
        tokenizer=pipeline.tokenizer,
        config=config,
        output_dir=str(output_dir),
    )

    # ── CYCLE 0: Baseline evaluation ─────────────────────────────────────── #
    logger.info("─" * 60)
    logger.info("CYCLE 0 — Baseline evaluation (zero episodic memory)")
    logger.info("─" * 60)
    t0 = time.time()
    cycle0_results = harness.run_all(eval_samples, cycle=0)
    logger.info("Cycle 0 done in %.1f min.", (time.time() - t0) / 60)

    all_cycle_results = [cycle0_results]

    # ── Calibration (after Cycle 0, before Cycle 1 fine-tuning) ──────────── #
    if not ns.skip_calibration:
        run_calibration_step(pipeline, calib_samples, config, output_dir, m)
    else:
        logger.info("Calibration skipped (--skip_calibration). Using equal initial weights.")

    # ── CYCLES 1–3 ─────────────────────────────────────────────────────────── #
    for cycle_num in range(1, config.num_cycles + 1):
        logger.info("─" * 60)
        logger.info("CYCLE %d — Fine-tuning + evaluation", cycle_num)
        logger.info("─" * 60)
        t0 = time.time()

        # Step 1: Fine-tune on verified episodes from current memory
        logger.info("  Step 1: SelfImprovementLoop.run_cycle(%d) …", cycle_num)
        cycle_result = sil.run_cycle(
            cycle_num=cycle_num,
            memory_store=pipeline.memory_store,
            general_data=general_data,
        )

        if cycle_result.aborted:
            logger.warning(
                "  Cycle %d ABORTED (forgetting score %.3f < %.3f). "
                "Weights restored. Eval still runs on the restored model.",
                cycle_num, cycle_result.forgetting_score, config.forgetting_tolerance,
            )
        else:
            logger.info(
                "  Fine-tuning done: %d episodes | forgetting=%.3f | loss=%.4f",
                cycle_result.n_episodes_used,
                cycle_result.forgetting_score,
                cycle_result.final_train_loss,
            )

        # Step 2: Update pipeline's cycle counter
        pipeline.current_cycle = cycle_num

        # Step 3: Retroactive re-verification of memory
        logger.info("  Step 2: Retroactive re-verification …")
        retroverify_stats = retroactive_reverification(pipeline, cycle_num, config)

        # Step 4: Save retroverify stats alongside cycle results
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
                },
            }, f, indent=2)

        # Step 5: Evaluate all benchmarks
        logger.info("  Step 3: Evaluating all benchmarks (cycle=%d) …", cycle_num)
        cycle_results = harness.run_all(eval_samples, cycle=cycle_num)
        all_cycle_results.append(cycle_results)

        logger.info("Cycle %d done in %.1f min.", cycle_num, (time.time() - t0) / 60)

    # ── Summary ───────────────────────────────────────────────────────────── #
    save_summary_csv(all_cycle_results, output_dir)
    print_mechanism_table(all_cycle_results)

    # ── Save full results JSON ─────────────────────────────────────────────── #
    full_results_path = output_dir / "all_cycle_results.json"
    with open(full_results_path, "w") as f:
        json.dump(all_cycle_results, f, indent=2)
    logger.info("Full results saved → %s", full_results_path)

    logger.info("Experiment complete. Outputs in: %s", output_dir)
    logger.info(
        "Next steps:\n"
        "  1. Run scripts/run_purity_validation.py for Theory 1/2/3 validation.\n"
        "  2. Run scripts/run_ablation.py for baselines and ablation variants.\n"
        "  3. Update caem-implementation-log.md with experiment results.\n"
        "  4. Feed outputs/ into Chapter 5 writing (use paper-writing skill)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAEM Experiment Orchestrator (Cycle 0→3)",
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
        "--benchmarks", nargs="+",
        default=["hotpotqa", "truthfulqa", "fever", "strategyqa"],
        help="Which benchmarks to evaluate.",
    )
    p.add_argument(
        "--passage_index", default="outputs/passage_index",
        help="Path to pre-built Wikipedia FAISS passage index (for Tier 3 RAG).",
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
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(args)
