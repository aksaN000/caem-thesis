"""
scripts/run_ablation.py
========================
CAEM Ablation Sweep Orchestrator (Phase 9, Session 43).

Sweeps every registered ablation variant against a fixed evaluation
set and produces:

  1. per-variant JSON   -> outputs/ablation/<variant>_cycle<n>.json
  2. ranking CSV        -> outputs/ablation/ablation_ranking_cycle<n>.csv
  3. summary table      -> stdout (sorted by CES descending)

Typical workflow
----------------
Run the main experiment first (``scripts.run_experiment``) to produce
the cycle-N memory store and fine-tuned weights. Then::

    python -m scripts.run_ablation \
        --cycle 7 \
        --memory_store outputs/memory_store_cycle_7 \
        --model_checkpoint outputs/cycle_7/model \
        --benchmarks fever triviaqa commonsense_qa truthfulqa strategyqa \
        --n_questions 500 \
        --output_dir outputs/ablation

All variants evaluate against the same (cycle-N model, cycle-N memory)
so deltas reflect the variant's config mutation alone.

Batched evaluation
------------------
This script does not currently expose a ``--eval_batch_size`` argparse
flag, so its evaluation path runs whatever batch size the underlying
evaluator defaults to. If a future Phase 1a integration reroutes the
runner through this orchestrator, callers should pass / configure
``eval_batch_size=32`` (matching the rest of the codebase: see
``scripts/run_calibration.py``'s default and the
``BatchPipeline.answer_batch`` example wiring in ``eval/harness.py``).
A batch dim of 1 is the legacy serial path and would erase the
~50-150 GPU-h savings the BatchPipeline was designed to produce
during the 14-day step_7_main run.

Thesis reference
----------------
  §5.4  Ablation Study (post-Session-42 rebuild with CES primary scalar)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ablation")


# -----------------------------------------------------------------------------
# Deferred imports (mirrors scripts/run_experiment.py pattern so --help works
# without GPU deps installed)
# -----------------------------------------------------------------------------
def _check_deps() -> None:
    missing = []
    for pkg in ["torch", "transformers", "sentence_transformers", "faiss"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Missing dependencies: %s", missing)
        sys.exit(1)


def _load_imports() -> Dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from caem.config import CAEMConfig
    from caem.memory.encoder import QueryEncoder
    from caem.memory.store import EpisodicMemoryStore
    from caem.pipeline import CAEMPipeline
    from caem.retrieval.rag import PassageStore
    from eval.benchmarks import load_benchmark, make_synthetic_samples
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
        load_benchmark=load_benchmark,
        make_synthetic_samples=make_synthetic_samples,
        EvalHarness=EvalHarness,
    )


# -----------------------------------------------------------------------------
# Pipeline construction per variant
# -----------------------------------------------------------------------------

def _build_variant_pipeline(
    variant,
    m: Dict[str, Any],
    ns: argparse.Namespace,
    model_obj,
    tokenizer,
    encoder,
    nli_model,
    nli_tokenizer,
    passage_store,
    memory_store,
    judge=None,
):
    """Construct a CAEMPipeline with the variant's mutated config applied."""
    # Fresh mutated config: never share a CAEMConfig across variants.
    config = variant.apply()
    pipeline = m["CAEMPipeline"](
        model=model_obj,
        tokenizer=tokenizer,
        encoder=encoder,
        judge=judge,
        nli_model=nli_model,
        nli_tokenizer=nli_tokenizer,
        passage_store=passage_store,
        config=config,
        current_cycle=ns.cycle,
    )
    # Attach the shared memory store (same for every variant).
    pipeline.memory_store = memory_store
    return pipeline


# -----------------------------------------------------------------------------
# Eval sample loading
# -----------------------------------------------------------------------------

def _load_eval_samples(ns: argparse.Namespace, m: Dict[str, Any]) -> Dict[str, list]:
    if ns.smoke_test:
        # Smoke mode honours --n_questions (capped at 10 for CPU-friendliness)
        # instead of a hardcoded 10. This lets callers do a 2-sample smoke
        # for the fastest-possible CI check via --n_questions 2, or ramp
        # up to the cap when debugging end-to-end integration.
        n_smoke = min(10, max(1, int(getattr(ns, "n_questions", 10) or 10)))
        logger.info(
            "SMOKE TEST MODE -- synthetic samples (n=%d per benchmark)", n_smoke,
        )
        return {
            bm: m["make_synthetic_samples"](bm, n=n_smoke)
            for bm in (bm.strip().lower() for bm in ns.benchmarks)
        }

    split_map = {
        "fever": "dev",
        "triviaqa": "validation",
        "natural_questions": "validation",
        "strategyqa": "test",
        "arc_challenge": "test",
    }
    samples: Dict[str, list] = {}
    for bm in (bm.strip().lower() for bm in ns.benchmarks):
        if bm == "truthfulqa":
            samples[bm] = m["load_benchmark"](bm, n=ns.n_questions)
        else:
            split = split_map.get(bm, "validation")
            samples[bm] = m["load_benchmark"](bm, split=split, n=ns.n_questions)
        logger.info("Loaded %s: %d samples", bm, len(samples[bm]))
    return samples


# -----------------------------------------------------------------------------
# Ranking CSV
# -----------------------------------------------------------------------------

def _sort_by_ces(results: List[Any]) -> List[Any]:
    """Sort ablation results by CES descending, with missing CES at the bottom.

    Uses -inf as the missing-CES sentinel (not -1) so a legitimately-negative
    CES cannot accidentally float above an un-scored variant. Centralised so
    the CSV writer and the stdout pretty-printer stay in lockstep.
    """
    return sorted(
        results,
        key=lambda r: (
            r.aggregate_axes.ces
            if r.aggregate_axes.ces is not None
            else float("-inf")
        ),
        reverse=True,
    )


def write_ranking_csv(
    results: List[Any],          # list[AblationResult]
    output_dir: Path,
    cycle: int,
    full_ces: Optional[float] = None,
) -> Path:
    """Write a one-row-per-variant ranking CSV sorted by CES descending.

    Each row carries the five CES axes, CES itself, the absolute CES
    delta against the full-CAEM reference, and the relative delta as a
    percent (so the thesis can quote "no_verifier loses 12.3% CES").
    """
    fieldnames = [
        "rank", "variant", "mechanism_tag", "description",
        "acc", "epi", "ret", "cal", "ver", "ces",
        "delta_abs_vs_full", "delta_pct_vs_full",
        "n_samples", "n_scored_confidence", "runtime_seconds",
    ]

    # Identify reference CES
    if full_ces is None:
        for r in results:
            if r.variant == "full":
                full_ces = r.aggregate_axes.ces
                break

    # -inf (not -1) as the missing-CES sentinel: CES is *usually* in [0, 1]
    # but nothing in the pipeline forbids a negative value, so -1 could float
    # a legitimately-poor variant above a missing one. -inf is unambiguous.
    sorted_results = _sort_by_ces(results)

    rows: List[Dict[str, Any]] = []
    for rank, r in enumerate(sorted_results, start=1):
        ax = r.aggregate_axes
        if full_ces is not None and full_ces > 0:
            delta_abs = ax.ces - full_ces
            delta_pct = (delta_abs / full_ces) * 100.0
        else:
            delta_abs = float("nan")
            delta_pct = float("nan")

        rows.append({
            "rank": rank,
            "variant": r.variant,
            "mechanism_tag": r.mechanism_tag,
            "description": r.description,
            "acc": round(ax.acc, 4),
            "epi": round(ax.epi, 4),
            "ret": round(ax.ret, 4),
            "cal": round(ax.cal, 4),
            "ver": round(ax.ver, 4),
            "ces": round(ax.ces, 4),
            "delta_abs_vs_full": round(delta_abs, 4),
            "delta_pct_vs_full": round(delta_pct, 2),
            "n_samples": ax.n_samples,
            "n_scored_confidence": ax.n_scored_confidence,
            "runtime_seconds": round(r.runtime_seconds, 1),
        })

    csv_path = output_dir / f"ablation_ranking_cycle{cycle}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Ranking CSV saved -> %s", csv_path)
    return csv_path


def print_ranking(results: List[Any], full_ces: Optional[float]) -> None:
    """Print the ranking table to stdout in the same order as the CSV."""
    print("\n" + "=" * 110)
    print("ABLATION RANKING  (sorted by CES descending; 'full' = reference)")
    print("=" * 110)
    print(
        f"{'Rank':<4} {'Variant':<24} {'Mech':<16} "
        f"{'CES':>6} {'ΔCES':>8} {'Δ%':>7} "
        f"{'ACC':>6} {'EPI':>6} {'RET':>6} {'CAL':>6} {'VER':>6}"
    )
    print("-" * 110)
    sorted_results = _sort_by_ces(results)
    for rank, r in enumerate(sorted_results, start=1):
        ax = r.aggregate_axes
        if full_ces and full_ces > 0:
            delta_abs = ax.ces - full_ces
            delta_pct = (delta_abs / full_ces) * 100.0
        else:
            delta_abs = 0.0
            delta_pct = 0.0
        print(
            f"{rank:<4} {r.variant:<24} {r.mechanism_tag:<16} "
            f"{ax.ces:>6.4f} {delta_abs:>+8.4f} {delta_pct:>+6.1f}% "
            f"{ax.acc:>6.3f} {ax.epi:>6.3f} {ax.ret:>6.3f} {ax.cal:>6.3f} {ax.ver:>6.3f}"
        )
    print("=" * 110 + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(ns: argparse.Namespace) -> None:
    from caem.ablation import list_variants, VARIANT_REGISTRY
    from caem.ablation.runner import run_variant

    _check_deps()
    m = _load_imports()
    torch = m["torch"]

    output_dir = Path(ns.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Hardware + model loading (shared across variants for cycle-N eval) --
    from scripts.hardware import print_hardware_summary, apply_memory_flags
    hw = print_hardware_summary()
    apply_memory_flags(hw)
    device = hw.device

    logger.info("Loading Flan-T5-Large ...")
    tokenizer = m["AutoTokenizer"].from_pretrained("google/flan-t5-large")
    model_obj = m["T5ForConditionalGeneration"].from_pretrained("google/flan-t5-large")
    if ns.model_checkpoint:
        logger.info("Loading fine-tuned weights from %s", ns.model_checkpoint)
        state = torch.load(
            ns.model_checkpoint, map_location=device, weights_only=True
        )
        model_obj.load_state_dict(state)
    if hw.use_bf16:
        model_obj = model_obj.to(torch.bfloat16)
    elif hw.use_fp16:
        model_obj = model_obj.to(torch.float16)
    model_obj = model_obj.to(device).eval()

    # Encoders and verifier deps (shared).  Use a baseline CAEMConfig so the
    # NLI model name matches what the verifier, run_experiment, and
    # run_purity_validation load.  Variants mutate a config copy downstream
    # (variant.apply()), but NLI weights are shared and are loaded once here.
    base_config = m["CAEMConfig"]()
    encoder = m["QueryEncoder"](
        model_name=base_config.sbert_model, device=device,
    )
    # Load verifier judge via the shared loader (MiniCheck or RoBERTa).
    # Ablation variants share the same judge instance; a per-variant judge
    # swap would be an orthogonal study not in scope here.
    from caem.verification import load_verifier_judge
    logger.info("Loading verifier judge (backend=%s) ...",
                base_config.verifier_backend)
    judge, nli_model, nli_tokenizer = load_verifier_judge(
        base_config, device, allow_fallback=False,
    )

    passage_store = None
    pi_path = Path(ns.passage_index) if ns.passage_index else None
    if pi_path and pi_path.exists():
        passage_store = m["PassageStore"].load(str(pi_path))
        logger.info("Passage store loaded: %d passages", len(passage_store.passages))

    memory_store = m["EpisodicMemoryStore"].load(ns.memory_store)
    logger.info("Memory store loaded: %d episodes", memory_store.size)

    # -- MMLU baselines (read from cycle-0 artefact if available) -----------
    baseline_mmlu: Optional[float] = None
    cycle_mmlu: Optional[float] = None
    baseline_path = Path(ns.baseline_mmlu_json) if ns.baseline_mmlu_json else None
    if baseline_path and baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            _baseline_payload = json.load(f)
        # Be defensive: older runs sometimes wrote the value under "mmlu"
        # instead of "mmlu_baseline". A hard KeyError here would abort the
        # whole ablation sweep on an old artefact layout — cheap to support.
        _raw = _baseline_payload.get("mmlu_baseline", _baseline_payload.get("mmlu"))
        if _raw is None:
            logger.warning(
                "Baseline MMLU JSON %s lacks 'mmlu_baseline' (and 'mmlu') key; "
                "continuing without a pristine baseline.",
                baseline_path,
            )
        else:
            baseline_mmlu = float(_raw)
            logger.info("Pristine MMLU baseline: %.4f", baseline_mmlu)

    rv_path = Path(ns.retroverify_json) if ns.retroverify_json else None
    if rv_path and rv_path.exists():
        with open(rv_path, "r", encoding="utf-8") as f:
            rv = json.load(f)
        cycle_mmlu = rv.get("fine_tune", {}).get("mmlu_retention")
        if cycle_mmlu is not None:
            logger.info("Cycle-%d MMLU: %.4f", ns.cycle, cycle_mmlu)

    # -- Eval sample loading -------------------------------------------------
    eval_samples = _load_eval_samples(ns, m)

    # -- Variant selection ---------------------------------------------------
    if ns.only:
        chosen = [VARIANT_REGISTRY[name] for name in ns.only]
    else:
        chosen = list_variants(mechanism=ns.mechanism)

    logger.info("Running %d variants: %s", len(chosen), [v.name for v in chosen])

    # -- Execute sweep -------------------------------------------------------
    results: List[Any] = []
    overall_t0 = time.time()
    for variant in chosen:
        pipeline = _build_variant_pipeline(
            variant, m, ns,
            model_obj, tokenizer, encoder,
            nli_model, nli_tokenizer, passage_store,
            memory_store, judge=judge,
        )
        harness = m["EvalHarness"](
            pipeline, output_dir=str(output_dir / variant.name), log_every=100,
        )
        try:
            result = run_variant(
                variant=variant,
                pipeline=pipeline,
                eval_samples=eval_samples,
                harness=harness,
                cycle=ns.cycle,
                baseline_mmlu=baseline_mmlu,
                cycle_mmlu=cycle_mmlu,
                output_dir=output_dir,
            )
        except Exception as exc:     # noqa: BLE001
            logger.error("Variant '%s' raised %s; continuing.", variant.name, exc)
            continue
        results.append(result)

    logger.info("Sweep finished in %.1f min.", (time.time() - overall_t0) / 60)

    # -- Ranking outputs -----------------------------------------------------
    full_ces = None
    for r in results:
        if r.variant == "full":
            full_ces = r.aggregate_axes.ces
            break
    write_ranking_csv(results, output_dir, cycle=ns.cycle, full_ces=full_ces)
    print_ranking(results, full_ces=full_ces)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--cycle", type=int, required=True,
                   help="Which cycle's memory + weights to evaluate against.")
    p.add_argument("--memory_store", type=str, required=True,
                   help="Path prefix for the EpisodicMemoryStore checkpoint.")
    p.add_argument("--model_checkpoint", type=str, default=None,
                   help="Optional fine-tuned model state_dict to load.")
    p.add_argument("--passage_index", type=str, default="data/passage_index",
                   help="Path to the Wikipedia passage FAISS index.")
    p.add_argument("--baseline_mmlu_json", type=str, default="outputs/mmlu_baseline.json",
                   help="Cycle-0 pristine MMLU JSON (for RET denominator).")
    p.add_argument("--retroverify_json", type=str, default=None,
                   help="Cycle-N retroverify JSON containing cycle MMLU (for RET).")
    p.add_argument("--n_questions", type=int, default=500,
                   help="Per-benchmark eval size for every variant.")
    # v2 Fix 9b: read benchmark roster from caem.config so the ablation
    # eval panel tracks v2 splits (was hardcoded to v1 6-bench list
    # including arc_challenge which v2 dropped).
    from caem.config import (
        TRAINING_BENCHMARKS as _CFG_TRAINING_BENCHMARKS,
        TRANSFER_BENCHMARKS as _CFG_TRANSFER_BENCHMARKS,
    )
    p.add_argument("--benchmarks", nargs="+",
                   default=list(_CFG_TRAINING_BENCHMARKS) + list(_CFG_TRANSFER_BENCHMARKS))
    p.add_argument("--output_dir", type=str, default="outputs/ablation")
    p.add_argument("--only", nargs="+", default=None,
                   help="Subset of variant names to run (default: all).")
    p.add_argument("--mechanism", type=str, default=None,
                   help="Filter variants by mechanism_tag "
                        "(e.g. 'verification', 'calibration').")
    p.add_argument("--smoke_test", action="store_true",
                   help="Use 10 synthetic samples per benchmark (CPU-friendly).")
    return p


if __name__ == "__main__":
    main(_build_argparser().parse_args())
