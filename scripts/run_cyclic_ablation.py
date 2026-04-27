"""
scripts/run_cyclic_ablation.py
==============================
Cyclic (training-time) ablation driver — one variant, one seed per run.

Context
-------
A cyclic ablation variant changes what gets stored or how the SIL pool is
shaped, so we MUST re-run the SIL cycle loop under the variant's config.
Evaluating a variant on full-CAEM's cycle-N weights would be a
counterfactual-confound — cycle-N was shaped by the mechanism that this
variant is supposed to disable. See the Session-43 ablation audit notes
in writing-suggestions.md.

What this script does
---------------------
1. Applies ``variant.apply(CAEMConfig())`` — a fresh config with the
   variant's mutation.
2. Applies pipeline-level skip flags (skip_verifier, skip_tier3_rag, ...)
   via ``caem.ablation.runner._apply_pipeline_flags``. These remain
   active for the lifetime of the run.
3. Runs the same Cycle 0 → N loop as ``scripts/run_experiment.py`` —
   reusing its helpers for pipeline construction, dataset loading,
   calibration, retroactive re-verification, and summary CSV emission.
4. After each cycle, computes CES axes from the harness's per-sample
   JSONs via ``caem.ablation.scoring.ces_axes_from_cycle`` and appends
   to ``ces_axes_per_cycle.json`` so the aggregator can read them back.
5. If ``variant.requires_baseline_only`` (no_self_improvement), runs
   Cycle 0 evaluation only and skips the SIL loop entirely.

Two-phase protocol support
--------------------------
- ``--screening_mode``: overrides to ``max_cycles=3, n_sil_per_cycle=1500``
  (cheap variant selector; results NOT used in Chapter 5's ablation table).
- Default (no flag): ``max_cycles=10, n_sil_per_cycle=5000`` (confirmatory;
  results are what go into the paper).

Multi-seed upgrade path
-----------------------
This driver is single-seed by design. Phase-1 self-funded runs use
``--seed 42`` once per variant. Phase-2 funded runs add ``--seed 123``
and ``--seed 456`` invocations. The output directory layout keeps
seeds in sibling folders so later-seed invocations never touch earlier
results.

Usage
-----
From the repo root:

    # Screening (Phase 1 — cheap selector)
    python -m scripts.run_cyclic_ablation \\
        --variant no_grounding \\
        --seed 42 \\
        --screening_mode \\
        --output_dir outputs/ablation

    # Confirmatory (Phase 1 — single seed, top-3 + full + no_SIL)
    python -m scripts.run_cyclic_ablation \\
        --variant no_grounding \\
        --seed 42 \\
        --output_dir outputs/ablation

    # Confirmatory upgrade (Phase 2 — add a new seed, same command)
    python -m scripts.run_cyclic_ablation \\
        --variant no_grounding \\
        --seed 123 \\
        --output_dir outputs/ablation

Output layout
-------------
    outputs/ablation/<variant>/seed_<N>/
        eval/                           # per-BM per-cycle JSONs (from harness)
        calibration/                    # fitted temperature + weights
        cycle_*/                        # fine-tuning checkpoints
        memory_store_cycle_*.faiss
        memory_store_cycle_*.meta
        mmlu_baseline.json              # pristine-model MMLU (cycle 0)
        retroverify_cycle*.json         # per-cycle retroverify stats
        experiment_summary.csv          # five-column evidence table
        ces_axes_per_cycle.json         # [CESAxes per cycle, 0..N]
        run_manifest.json               # variant + seed + timings

Batched evaluation
------------------
This driver does not currently expose a ``--eval_batch_size`` argparse
flag, so the harness it spins up uses whatever default the evaluator
ships with. If a future Phase 1a integration routes
``run_phase1a.sh``'s ablation pass through this driver (currently it
is not invoked), callers should pass / configure ``eval_batch_size=32``
to match the BatchPipeline wiring used elsewhere (see
``scripts/run_calibration.py`` and ``eval/harness.py``). The default
of 1 is the legacy serial path and would forfeit the BatchPipeline
speedup that the 14-day step_7_main run depends on.

Thesis reference
----------------
  §4.8 Ablation methodology (training-time vs. inference-time)
  §5.4 Mechanism evidence (per-variant CES delta table)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# -- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_cyclic_ablation")


# =============================================================================
# Preset profiles
# =============================================================================

SCREENING_PROFILE = {
    "max_cycles": 3,
    "n_sil_per_cycle": 1500,
    "n_eval_per_benchmark": 500,
}

CONFIRMATORY_PROFILE = {
    "max_cycles": 10,
    "n_sil_per_cycle": 5000,
    "n_eval_per_benchmark": 500,
}

SMOKE_PROFILE = {
    "max_cycles": 1,
    "n_sil_per_cycle": 500,
    "n_eval_per_benchmark": 50,
}


# =============================================================================
# Seed plumbing
# =============================================================================

def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs. Idempotent."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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


# =============================================================================
# Output-directory helpers
# =============================================================================

def _variant_seed_dir(output_root: Path, variant_name: str, seed: int) -> Path:
    """Return ``<output_root>/<variant>/seed_<N>/`` and create it."""
    out = output_root / variant_name / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    return out


def _write_manifest(
    out_dir: Path,
    *,
    variant_name: str,
    seed: int,
    profile_name: str,
    max_cycles: int,
    n_sil: int,
    n_eval: int,
    cycles_completed: int,
    runtime_seconds: float,
    aborted: bool,
    benchmarks: List[str],
) -> None:
    """Persist the run metadata as ``run_manifest.json``."""
    manifest = {
        "variant": variant_name,
        "seed": seed,
        "profile": profile_name,
        "max_cycles": max_cycles,
        "n_sil_per_cycle": n_sil,
        "n_eval_per_benchmark": n_eval,
        "cycles_completed": cycles_completed,
        "runtime_seconds": round(runtime_seconds, 2),
        "aborted": aborted,
        "benchmarks": benchmarks,
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written -> %s", out_dir / "run_manifest.json")


# =============================================================================
# CES-axes extraction from harness output
# =============================================================================

def _axes_for_cycle(
    *,
    eval_dir: Path,
    benchmarks: List[str],
    cycle: int,
    baseline_mmlu: Optional[float],
    cycle_mmlu: Optional[float],
    requires_baseline_only: bool,
) -> Dict[str, Any]:
    """Compute CESAxes per benchmark + aggregate for one cycle.

    Reads ``<eval_dir>/<bm>_cycle{cycle}.json`` for each benchmark and
    delegates to ``caem.ablation.scoring.ces_axes_from_cycle``.
    """
    from caem.ablation.runner import _extract_em_and_u_from_samples
    from caem.ablation.scoring import (
        CESAxes,
        aggregate_axes,
        ces_axes_from_cycle,
    )

    per_bm: Dict[str, CESAxes] = {}
    per_bm_meta: Dict[str, Dict[str, Any]] = {}

    for bm in benchmarks:
        per_sample_path = eval_dir / f"{bm}_cycle{cycle}.json"
        if not per_sample_path.exists():
            logger.warning(
                "Per-sample JSON missing for %s cycle=%d (%s); skipping axes.",
                bm, cycle, per_sample_path,
            )
            continue
        try:
            with open(per_sample_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not parse %s (%s); skipping axes for this BM.",
                per_sample_path, exc,
            )
            continue

        samples_field = data.get("samples", [])
        meta = data.get("meta", {})
        extracted = _extract_em_and_u_from_samples(samples_field)

        if not extracted["em"]:
            # Use meta scalars as last-resort fallback
            n = int(meta.get("n", 0))
            em_list = [float(meta.get("em", 0.0))] * max(n, 1)
            u_list = [float(meta.get("mean_u_stored", 0.0))] * max(n, 1)
        else:
            em_list = extracted["em"]
            u_list = extracted["u_stored"]

        axes = ces_axes_from_cycle(
            em_flags=em_list,
            u_stored=u_list,
            cycle_mmlu=cycle_mmlu,
            baseline_mmlu=baseline_mmlu,
            verifier_balanced_accuracy=None,   # VER placeholder 0.5
            requires_baseline_only=requires_baseline_only,
        )
        per_bm[bm] = axes
        per_bm_meta[bm] = dict(meta)

    agg = aggregate_axes(per_bm.values()) if per_bm else None

    return {
        "cycle": cycle,
        "per_benchmark_axes": {bm: ax.as_dict() for bm, ax in per_bm.items()},
        "aggregate_axes": agg.as_dict() if agg is not None else None,
        "per_benchmark_meta": per_bm_meta,
    }


def _append_ces_record(ces_path: Path, record: Dict[str, Any]) -> None:
    """Append one per-cycle CES record to the aggregator-consumed JSON."""
    history: List[Dict[str, Any]] = []
    if ces_path.exists():
        try:
            with open(ces_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []
    # Replace any existing entry for the same cycle to keep resume-safe.
    history = [h for h in history if h.get("cycle") != record.get("cycle")]
    history.append(record)
    history.sort(key=lambda h: h.get("cycle", 0))
    with open(ces_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# =============================================================================
# Main loop
# =============================================================================

def run_cyclic_ablation(ns: argparse.Namespace) -> None:
    """Run one (variant, seed) cyclic ablation end-to-end."""
    # --- Resolve preset profile --------------------------------------------- #
    if ns.smoke_test:
        profile = dict(SMOKE_PROFILE)
        profile_name = "smoke"
    elif ns.screening_mode:
        profile = dict(SCREENING_PROFILE)
        profile_name = "screening"
    else:
        profile = dict(CONFIRMATORY_PROFILE)
        profile_name = "confirmatory"

    # CLI overrides beat the profile (advanced users)
    if ns.max_cycles is not None:
        profile["max_cycles"] = ns.max_cycles
    if ns.n_sil_per_cycle is not None:
        profile["n_sil_per_cycle"] = ns.n_sil_per_cycle
    if ns.n_eval_per_benchmark is not None:
        profile["n_eval_per_benchmark"] = ns.n_eval_per_benchmark

    # --- Seed ---------------------------------------------------------------- #
    _seed_everything(ns.seed)
    logger.info(
        "Cyclic ablation: variant=%s seed=%d profile=%s (%d cycles, %d SIL, %d eval)",
        ns.variant, ns.seed, profile_name,
        profile["max_cycles"], profile["n_sil_per_cycle"],
        profile["n_eval_per_benchmark"],
    )

    # --- Imports (deferred so --help works without torch) ------------------- #
    # Use the run_experiment module so its private helpers remain the single
    # source of truth for pipeline construction + dataset loading.
    from scripts.run_experiment import (
        _check_deps,
        _load_imports,
        build_pipeline,
        load_sil_training_pool,
        load_eval_transfer_pool,
        split_calibration_sets,
        load_general_data,
        run_calibration_step,
        retroactive_reverification,
        save_summary_csv,
    )

    _check_deps()
    m = _load_imports()
    torch = m["torch"]

    from caem.ablation.variants import get_variant
    from caem.ablation.runner import (
        _apply_pipeline_flags,
        _restore_pipeline_flags,
    )

    variant = get_variant(ns.variant)
    logger.info(
        "Variant: %s (mechanism=%s, cyclic=%s, baseline_only=%s)",
        variant.name, variant.mechanism_tag,
        variant.needs_cyclic_rerun, variant.requires_baseline_only,
    )

    # --- Output directory (per variant, per seed) --------------------------- #
    output_root = Path(ns.output_dir)
    out_dir = _variant_seed_dir(output_root, variant.name, ns.seed)
    eval_dir = out_dir / "eval"
    ces_path = out_dir / "ces_axes_per_cycle.json"

    # --- Variant-mutated config --------------------------------------------- #
    config = variant.apply(m["CAEMConfig"]())
    # Persist the config snapshot for post-hoc reproducibility.
    # We first try to let `json` handle each value directly so that lists,
    # dicts and tuples of primitives survive as real JSON structures (rather
    # than being coerced into opaque `repr()` strings that analysis notebooks
    # then have to parse back).
    def _json_safe(v):
        try:
            json.dumps(v)
            return v
        except (TypeError, ValueError):
            return repr(v)

    try:
        cfg_snapshot = {k: _json_safe(v) for k, v in vars(config).items()}
        with open(out_dir / "variant_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg_snapshot, f, indent=2)
    except Exception as exc:
        logger.warning("Could not serialise variant config (%s); continuing.", exc)

    # Wire cycle count and SIL pool size into the config.
    # Ablation profiles (smoke / medium / full) intentionally override any
    # num_cycles / questions_per_cycle value a variant may have set, because
    # the profile governs the sweep-wide compute budget. Log when we are
    # clobbering a non-default variant setting so reviewers can trace the
    # decision instead of being surprised by a silent override.
    profile_cycles = int(profile["max_cycles"])
    profile_qpc = int(profile["n_sil_per_cycle"])
    if config.num_cycles != profile_cycles:
        logger.info(
            "Ablation profile overrides variant num_cycles: %d -> %d (variant=%s).",
            config.num_cycles, profile_cycles, variant.name,
        )
    if config.questions_per_cycle != profile_qpc:
        logger.info(
            "Ablation profile overrides variant questions_per_cycle: %d -> %d "
            "(variant=%s).",
            config.questions_per_cycle, profile_qpc, variant.name,
        )
    config.num_cycles = profile_cycles
    config.questions_per_cycle = profile_qpc

    # --- Fabricate the namespace that run_experiment helpers expect --------- #
    # Most helpers take an argparse.Namespace with specific attributes.
    # Build a shim so we can reuse them without forking the code.
    ns_shim = argparse.Namespace(
        output_dir=str(out_dir),
        num_cycles=config.num_cycles,
        n_questions=int(profile["n_sil_per_cycle"]),
        benchmarks=ns.benchmarks,
        passage_index=ns.passage_index,
        cold_start_memory=ns.cold_start_memory,
        resume_from_cycle=ns.resume_from_cycle,
        skip_calibration=ns.skip_calibration,
        smoke_test=ns.smoke_test,
    )

    # --- Build pipeline and apply variant pipeline flags -------------------- #
    pipeline = build_pipeline(config, ns_shim, m)
    flag_backups = _apply_pipeline_flags(pipeline, variant)
    logger.info(
        "Variant '%s': applied %d pipeline-level flag patches.",
        variant.name, len(flag_backups),
    )

    # --- Load datasets ------------------------------------------------------- #
    if ns.smoke_test:
        from eval.benchmarks import make_synthetic_samples
        requested = [bm.strip().lower() for bm in ns.benchmarks]
        sil_benchmarks = [
            bm for bm in requested
            if bm in {"fever", "triviaqa", "natural_questions"}
        ]
        if not sil_benchmarks:
            sil_benchmarks = ["fever", "triviaqa", "natural_questions"]
        # Honour the ablation profile's n_sil_per_cycle / n_eval_per_benchmark
        # instead of a hardcoded 10. This lets the smoke profile stay tiny
        # (n=4) while medium/full profiles still exercise realistic batch
        # sizes in smoke-test mode. Calibration split: half purity / half
        # calib -- min 2 each so the split does not degenerate at tiny n.
        n_sil_smoke = max(4, int(profile["n_sil_per_cycle"]))
        n_eval_smoke = max(4, int(profile["n_eval_per_benchmark"]))
        sil_samples = {bm: make_synthetic_samples(bm, n=n_sil_smoke) for bm in sil_benchmarks}
        eval_samples = {bm: make_synthetic_samples(bm, n=n_eval_smoke) for bm in requested}
        split_point = max(2, n_sil_smoke // 2)
        purity_samples = {bm: s[:split_point] for bm, s in sil_samples.items()}
        calib_samples = {bm: s[split_point:] for bm, s in sil_samples.items()}
    else:
        sil_samples = load_sil_training_pool(ns_shim, m)
        eval_samples = load_eval_transfer_pool(ns_shim, m)
        # Trim eval samples to the profile cap (lets screening use fewer)
        target_eval = int(profile["n_eval_per_benchmark"])
        eval_samples = {bm: s[:target_eval] for bm, s in eval_samples.items()}
        purity_samples, calib_samples, _ = split_calibration_sets(
            sil_samples,
            calib_size=config.calibration_set_size,
            purity_size=config.purity_validation_set_size,
        )

    # --- Harness + SIL loop ------------------------------------------------- #
    harness = m["EvalHarness"](pipeline, output_dir=str(eval_dir), log_every=100)

    # Respect skip_self_improvement: we still construct the loop object so
    # _mmlu_score is available, but we never call run_cycle on it.
    sil = m["SelfImprovementLoop"](
        model=pipeline.model,
        tokenizer=pipeline.tokenizer,
        config=config,
        output_dir=str(out_dir),
    )

    # Pull the general-mix size from CAEMConfig (general_data_size=1000 by
    # default) rather than hard-coding 1000 here — keeps the ablation sweep
    # in lockstep with the main experiment when the anti-forgetting budget
    # is ever re-tuned.
    general_data = load_general_data(n=int(getattr(config, "general_data_size", 1000)))

    # Per-cycle trackers mirror the main experiment
    all_cycle_results: List[Dict[str, Any]] = []
    mmlu_per_cycle: List[float] = []
    baseline_mmlu: Optional[float] = None

    start_t = time.time()
    # cycles_completed counts the number of completed cycles (cycle 0 is the
    # baseline evaluation, cycles 1..N are SIL loops). Post-loop invariant:
    # cycles_completed == last_completed_cycle_index + 1. A value of 1 means
    # only the baseline ran; 3 means cycles 0, 1, and 2 all finished.
    cycles_completed = 0
    aborted = False

    # =========================================================================
    # Cycle 0 — baseline
    # =========================================================================
    try:
        logger.info("-" * 60)
        logger.info("CYCLE 0 (variant=%s, seed=%d) -- baseline eval",
                    variant.name, ns.seed)
        logger.info("-" * 60)
        t0 = time.time()

        cycle0_results = harness.run_all(
            eval_samples, cycle=0, store_to_memory=False,
        )
        logger.info("Cycle 0 done in %.1f min.", (time.time() - t0) / 60)

        # Pristine MMLU baseline
        logger.info("Measuring pristine MMLU baseline (200 samples)...")
        mmlu_baseline = float(sil.measure_mmlu(n=200))
        logger.info("Pristine MMLU baseline: %.4f", mmlu_baseline)
        try:
            with open(out_dir / "mmlu_baseline.json", "w", encoding="utf-8") as f:
                json.dump({"mmlu_baseline": mmlu_baseline}, f)
        except OSError as exc:
            logger.warning("Failed to persist mmlu_baseline.json (%s)", exc)

        baseline_mmlu = mmlu_baseline
        mmlu_per_cycle.append(mmlu_baseline)
        all_cycle_results.append(cycle0_results)

        # CES axes for cycle 0
        ces0 = _axes_for_cycle(
            eval_dir=eval_dir,
            benchmarks=list(eval_samples.keys()),
            cycle=0,
            baseline_mmlu=baseline_mmlu,
            cycle_mmlu=mmlu_baseline,
            requires_baseline_only=variant.requires_baseline_only,
        )
        _append_ces_record(ces_path, ces0)

        # Save cycle-0 memory store snapshot (empty unless cold-start used)
        pipeline.memory_store.save(str(out_dir / "memory_store_cycle_0"))

        # Calibration (skipped for baseline-only variants since there is no
        # fine-tuning downstream that would consume the calibrated weights)
        if not ns.skip_calibration and not variant.requires_baseline_only:
            run_calibration_step(pipeline, calib_samples, config, out_dir, m)

        cycles_completed = 1

        # =====================================================================
        # Cycles 1..N  (skipped entirely for requires_baseline_only variants)
        # =====================================================================
        if variant.requires_baseline_only:
            logger.info(
                "Variant '%s' is baseline-only (requires_baseline_only=True) "
                "-- skipping SIL loop.", variant.name,
            )
        else:
            for cycle_num in range(1, config.num_cycles + 1):
                logger.info("-" * 60)
                logger.info(
                    "CYCLE %d (variant=%s, seed=%d) -- SIL + eval",
                    cycle_num, variant.name, ns.seed,
                )
                logger.info("-" * 60)
                t_cyc = time.time()

                # -- Step 1: fine-tune + (optional) retroverify --------------- #
                # no_retroverify: pass verify_fn=None so sil.run_cycle's internal
                # retroverify loop is a no-op. We ALSO skip the outer
                # retroactive_reverification(...) call below.
                verify_fn = None
                if not variant.skip_retroverify:
                    verify_fn = pipeline.make_retroverify_fn()

                cycle_result = sil.run_cycle(
                    cycle_num=cycle_num,
                    memory_store=pipeline.memory_store,
                    general_data=general_data,
                    seed=ns.seed,
                    verify_fn=verify_fn,
                )

                if cycle_result.aborted:
                    logger.warning(
                        "Cycle %d ABORTED (retention %.3f < %.3f). "
                        "Weights restored; eval still runs on the restored model.",
                        cycle_num,
                        cycle_result.mmlu_retention_ratio,
                        config.forgetting_tolerance,
                    )
                    aborted = True
                else:
                    logger.info(
                        "Fine-tune: %d ep | retention=%.3f | loss=%.4f "
                        "| retro: %d updated / %d pruned",
                        cycle_result.n_episodes_used,
                        cycle_result.mmlu_retention_ratio,
                        cycle_result.final_train_loss,
                        cycle_result.n_retroverified,
                        cycle_result.n_retropruned,
                    )

                pipeline.current_cycle = cycle_num

                # -- Step 2: outer retroactive re-verification (if enabled) --- #
                if not variant.skip_retroverify:
                    retro_stats = retroactive_reverification(pipeline, cycle_num, config)
                else:
                    retro_stats = {"total": 0, "updated": 0, "pruned": 0}
                    logger.info("Skipping outer retroverify (variant=%s).", variant.name)

                # Persist retroverify + fine-tune metrics per cycle.
                # Use getattr with a NaN default so a CycleResult that lacks
                # mmlu_retention (e.g. an aborted cycle that stopped before
                # the MMLU eval) degrades to a null JSON cell rather than
                # AttributeError-ing the whole run.
                mmlu_val = getattr(cycle_result, "mmlu_retention", float("nan"))
                mmlu_ser = (
                    None
                    if mmlu_val is None or math.isnan(float(mmlu_val))
                    else round(float(mmlu_val), 6)
                )
                rv_payload = {
                    "cycle": cycle_num,
                    "retroverify": retro_stats,
                    "fine_tune": {
                        "n_episodes_used": cycle_result.n_episodes_used,
                        "n_general_used": cycle_result.n_general_used,
                        "mmlu_retention_ratio": cycle_result.mmlu_retention_ratio,
                        "aborted": cycle_result.aborted,
                        "final_train_loss": cycle_result.final_train_loss,
                        "mmlu_retention": mmlu_ser,
                    },
                }
                rv_path = out_dir / f"retroverify_cycle{cycle_num}.json"
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=out_dir, suffix=".tmp",
                    delete=False, encoding="utf-8",
                ) as _tmp:
                    json.dump(rv_payload, _tmp, indent=2)
                    _tmp_path = _tmp.name
                os.replace(_tmp_path, rv_path)

                mmlu_per_cycle.append(
                    float(mmlu_val) if mmlu_val is not None else float("nan")
                )

                # -- Step 3: memory population (upgraded weights) ------------- #
                logger.info("Populating memory under variant config...")
                harness.run_all(
                    sil_samples, cycle=cycle_num, store_to_memory=True,
                )

                # -- Step 4: evaluation (no memory writes) -------------------- #
                cycle_results = harness.run_all(
                    eval_samples, cycle=cycle_num, store_to_memory=False,
                )
                all_cycle_results.append(cycle_results)

                # -- Step 5: memory checkpoint -------------------------------- #
                pipeline.memory_store.save(
                    str(out_dir / f"memory_store_cycle_{cycle_num}")
                )

                # -- Step 6: CES axes for this cycle -------------------------- #
                ces_rec = _axes_for_cycle(
                    eval_dir=eval_dir,
                    benchmarks=list(eval_samples.keys()),
                    cycle=cycle_num,
                    baseline_mmlu=baseline_mmlu,
                    cycle_mmlu=(
                        float(mmlu_val)
                        if mmlu_val is not None and not math.isnan(float(mmlu_val))
                        else None
                    ),
                    requires_baseline_only=variant.requires_baseline_only,
                )
                _append_ces_record(ces_path, ces_rec)

                cycles_completed = cycle_num + 1
                logger.info(
                    "Cycle %d done in %.1f min.",
                    cycle_num, (time.time() - t_cyc) / 60,
                )

                # Early stop on SIL abort if the user asked for it
                if aborted and ns.stop_on_abort:
                    logger.warning(
                        "Cycle %d was aborted and --stop_on_abort is set; "
                        "halting sweep for variant '%s'.",
                        cycle_num, variant.name,
                    )
                    break

    finally:
        # Always restore the pipeline's pre-variant state, even on crash.
        _restore_pipeline_flags(pipeline, flag_backups)
        logger.info("Pipeline flags restored.")
        # Release GPU memory -- critical when this function is invoked
        # in-process across variants (the CLI entrypoint runs one variant
        # per process today, but callers can loop). Without this, sweeps
        # accumulate stale CUDA allocations and eventually OOM.
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.ipc_collect()
        except Exception:
            pass

    # --- Summary CSV + manifest --------------------------------------------- #
    save_summary_csv(all_cycle_results, out_dir, mmlu_per_cycle=mmlu_per_cycle)

    runtime = time.time() - start_t
    _write_manifest(
        out_dir,
        variant_name=variant.name,
        seed=ns.seed,
        profile_name=profile_name,
        max_cycles=config.num_cycles,
        n_sil=int(profile["n_sil_per_cycle"]),
        n_eval=int(profile["n_eval_per_benchmark"]),
        cycles_completed=cycles_completed,
        runtime_seconds=runtime,
        aborted=aborted,
        benchmarks=list(eval_samples.keys()),
    )

    logger.info(
        "DONE variant=%s seed=%d profile=%s cycles=%d runtime=%.1fmin",
        variant.name, ns.seed, profile_name,
        cycles_completed, runtime / 60.0,
    )


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run one cyclic (training-time) ablation variant at one seed. "
            "The SIL loop runs under the variant's config — no counterfactual "
            "confound from reusing full-CAEM weights."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -- Variant selection
    p.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Variant name (see caem.ablation.variants.VARIANT_REGISTRY).",
    )
    # -- Seed plumbing
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed. Phase 1 uses 42; Phase 2 adds 123 and 456.",
    )

    # -- Profile flags (mutually informative with --max_cycles / --n_sil_*)
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--screening_mode",
        action="store_true",
        help="Cheap variant selector: 3 cycles x 1500 SIL (Phase-1 screening).",
    )
    g.add_argument(
        "--smoke_test",
        action="store_true",
        help="1 cycle x 500 SIL + synthetic samples; crash detection only.",
    )

    # -- Advanced overrides (beat the profile when supplied)
    p.add_argument("--max_cycles", type=int, default=None,
                   help="Override cycles. Defaults: confirmatory=10, screening=3, smoke=1.")
    p.add_argument("--n_sil_per_cycle", type=int, default=None,
                   help="Override SIL samples per cycle. Defaults: confirmatory=5000, screening=1500, smoke=500.")
    p.add_argument("--n_eval_per_benchmark", type=int, default=None,
                   help="Override eval set size per benchmark. Defaults: 500 (50 in smoke).")

    # -- Run-experiment compatibility
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
        help="Optional path to a pre-seeded EpisodicMemoryStore.",
    )
    p.add_argument(
        "--resume_from_cycle",
        type=int,
        default=0,
        help=(
            "[NOT YET IMPLEMENTED] Cycle to resume from (0 = fresh). "
            "Passing a non-zero value currently raises a CLI error; "
            "kept as a reserved flag so a future resume implementation "
            "lands on the same surface. Delete the variant/seed output "
            "dir and re-run from cycle 0 if a previous run crashed."
        ),
    )
    p.add_argument(
        "--skip_calibration",
        action="store_true",
        help="Skip temperature scaling + signal-weight fitting.",
    )
    p.add_argument(
        "--stop_on_abort",
        action="store_true",
        help="Halt the sweep after the first aborted cycle (useful in Phase 1).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ablation",
        help="Root directory for ablation outputs. Per-variant/per-seed subdirs are auto-created.",
    )
    return p.parse_args()


def main() -> None:
    ns = _parse_args()
    if ns.resume_from_cycle != 0:
        # Resume is a Phase-2 upgrade. Historically this was a soft
        # logger.warning that then silently restarted from cycle 0,
        # which could stay unnoticed for hours on Vast.ai when a user
        # ctrl-C'd a stuck cycle and re-invoked with
        # --resume_from_cycle N expecting resume semantics. Fail loud
        # via argparse instead: the flag is reserved for a future resume
        # implementation but any non-zero value today is a user error.
        raise SystemExit(
            "ERROR: --resume_from_cycle is not yet implemented by "
            "run_cyclic_ablation.py. Passing a non-zero value would "
            "silently re-run from cycle 0, which is almost never what "
            "you want. Delete the variant/seed output directory and "
            "re-run from cycle 0, or use the main-experiment resume "
            "flow in scripts/run_experiment.py."
        )
    run_cyclic_ablation(ns)


if __name__ == "__main__":
    main()
