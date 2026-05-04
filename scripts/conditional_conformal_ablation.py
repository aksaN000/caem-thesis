"""
scripts/conditional_conformal_ablation.py
==========================================
Per-benchmark conformal storage gate ablation (Scope A — calibration-only,
no GPU, no SIL re-run).

Hypothesis tested
-----------------
The global conformal storage gate fits one (τ_store, τ_defer) pair on the
union of training-benchmark calibration samples. If the per-benchmark
signal distributions differ, this single global threshold could be the
upstream cause of the cycle-2 stream-chunk asymmetry we observed (FEVER
stored 15.60 %, TriviaQA stored 0.00 %).

Splitting the cal fold by benchmark and re-fitting a separate gate per
benchmark — each at the same α_store = 0.05 precision contract —
produces a per-benchmark (τ_store, τ_defer) report that lets us answer:

* Do the per-benchmark thresholds differ meaningfully from the global gate?
* Would TriviaQA / NQ admit non-zero storage under their own benchmark-
  conditional gate at the same 95 % precision contract?

The answer informs whether the global-gate architecture itself is the
bottleneck (conditional conformal future-work) or whether the asymmetry
is structurally driven by base-model accuracy (no architectural fix
available without a different SIL primitive).

Usage
-----
``python -m scripts.conditional_conformal_ablation --cycle 2``

Reads ``outputs/full_run/cycle_{N}/calibration/{bench}_cycle{N}.json``
(per-sample u_stored + em from the post-SIL cal-fold scoring at Step 2.1)
plus the global ``outputs/full_run/cycle_{N}/conformal_gate.json``.

Writes ``outputs/full_run/cycle_{N}/conditional_conformal_ablation.json``
with per-benchmark and global gates side by side. CPU-only; runs in
seconds. Safe to run in parallel with the main step_7 trajectory.

Thesis placement
----------------
This artefact feeds a one-paragraph subsection in Ch5 §sec:adj-cal-eval-gap
(per NEXT_SESSION_PLAN P3d) cross-linked from Ch6 §future-work
(conditional conformal as Phase 1c).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

# The conformal gate module is in caem/verification/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from caem.verification.conformal_gate import ConformalStorageGate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("conditional_conformal_ablation")

TRAINING_BENCHMARKS = ("fever", "triviaqa", "natural_questions")


def load_calfold_samples(cal_dir: Path, bench: str, cycle: int) -> List[Dict[str, Any]]:
    p = cal_dir / f"{bench}_cycle{cycle}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing cal-fold JSON: {p}")
    with open(p) as f:
        d = json.load(f)
    for k in ("samples", "results", "per_sample", "data"):
        if k in d and isinstance(d[k], list):
            return d[k]
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    raise ValueError(f"cannot locate sample list in {p}")


def fit_per_benchmark_gate(
    samples: List[Dict[str, Any]],
    alpha_store: float,
    alpha_defer: float,
) -> Dict[str, Any]:
    """Fit a split-CP gate on a single benchmark's cal-fold samples."""
    n = len(samples)
    n_em1 = sum(1 for s in samples if s.get("em") == 1)
    em_rate = n_em1 / n if n else float("nan")

    try:
        gate = ConformalStorageGate.fit(
            calib_samples=samples,
            score_key="u_stored",
            em_key="em",
            alpha_store=alpha_store,
            alpha_defer=alpha_defer,
            min_n=20,
        )
        return {
            "n": n,
            "em_rate": em_rate,
            "tau_store": gate.tau_store,
            "tau_defer": gate.tau_defer,
            "alpha_store": gate.alpha_store,
            "alpha_defer": gate.alpha_defer,
            "store_n": gate.store_n,
            "store_precision": gate.store_precision,
            "store_rate": gate.store_n / n,
            "defer_n": gate.defer_n,
            "defer_precision": gate.defer_precision,
            "defer_rate": gate.defer_n / n,
            "fit_status": "ok",
        }
    except ValueError as exc:
        return {
            "n": n,
            "em_rate": em_rate,
            "fit_status": f"insufficient_samples: {exc}",
        }


def what_if_apply_global_gate(
    samples: List[Dict[str, Any]],
    global_tau_store: float,
    global_tau_defer: float,
) -> Dict[str, Any]:
    """Apply the GLOBAL gate to a benchmark's samples and report the
    resulting decision counts + precision. This shows what the global
    gate actually does on each benchmark slice — i.e. how the asymmetry
    presents at the gate's downstream interface."""
    store_correct = sum(
        1 for s in samples
        if s.get("u_stored", 0) >= global_tau_store and s.get("em") == 1
    )
    store_wrong = sum(
        1 for s in samples
        if s.get("u_stored", 0) >= global_tau_store and s.get("em") == 0
    )
    store_total = store_correct + store_wrong
    defer_correct = sum(
        1 for s in samples
        if global_tau_defer <= s.get("u_stored", 0) < global_tau_store
        and s.get("em") == 1
    )
    defer_wrong = sum(
        1 for s in samples
        if global_tau_defer <= s.get("u_stored", 0) < global_tau_store
        and s.get("em") == 0
    )
    defer_total = defer_correct + defer_wrong
    return {
        "store_n": store_total,
        "store_correct": store_correct,
        "store_precision": (store_correct / store_total) if store_total else float("nan"),
        "store_rate": store_total / len(samples) if samples else float("nan"),
        "defer_n": defer_total,
        "defer_correct": defer_correct,
        "defer_precision": (defer_correct / defer_total) if defer_total else float("nan"),
        "defer_rate": defer_total / len(samples) if samples else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cycle", type=int, required=True,
                   help="Cycle number (1, 2, ..., 10)")
    p.add_argument("--root", type=Path, default=Path("outputs/full_run"),
                   help="Root output dir (default: outputs/full_run)")
    p.add_argument("--alpha_store", type=float, default=0.05,
                   help="Conformal precision target for STORE (default 0.05)")
    p.add_argument("--alpha_defer", type=float, default=0.40,
                   help="Conformal precision target for DEFER (default 0.40)")
    args = p.parse_args()

    cal_dir = args.root / f"cycle_{args.cycle}" / "calibration"
    if not cal_dir.exists():
        logger.error("cal-fold dir not found: %s", cal_dir)
        return 2

    # Load per-benchmark cal-fold samples.
    per_bm: Dict[str, List[Dict[str, Any]]] = {}
    for bench in TRAINING_BENCHMARKS:
        per_bm[bench] = load_calfold_samples(cal_dir, bench, args.cycle)
        logger.info("loaded %d cal-fold samples for %s cycle=%d",
                    len(per_bm[bench]), bench, args.cycle)

    # Load the global gate for comparison.
    global_gate_path = args.root / f"cycle_{args.cycle}" / "conformal_gate.json"
    with open(global_gate_path) as f:
        global_gate = json.load(f)
    logger.info("loaded global gate: tau_store=%.4f tau_defer=%.4f store_prec=%.4f",
                global_gate["tau_store"], global_gate["tau_defer"],
                global_gate["store_precision"])

    # Per-benchmark fit.
    per_bm_gates: Dict[str, Any] = {}
    for bench, samples in per_bm.items():
        per_bm_gates[bench] = fit_per_benchmark_gate(
            samples, alpha_store=args.alpha_store, alpha_defer=args.alpha_defer,
        )
        logger.info("fitted gate for %s: %s", bench,
                    {k: v for k, v in per_bm_gates[bench].items()
                     if k in ("tau_store", "tau_defer", "store_n",
                              "store_precision", "store_rate", "fit_status")})

    # What-if: apply global gate to each benchmark slice (shows the
    # asymmetry as the global gate currently presents it).
    per_bm_under_global: Dict[str, Any] = {
        bench: what_if_apply_global_gate(
            samples,
            global_gate["tau_store"],
            global_gate["tau_defer"],
        )
        for bench, samples in per_bm.items()
    }

    # Compose final report.
    report = {
        "schema_version": "branchC.2026-05-01",
        "cycle": args.cycle,
        "alpha_store": args.alpha_store,
        "alpha_defer": args.alpha_defer,
        "global_gate": {
            "tau_store": global_gate["tau_store"],
            "tau_defer": global_gate["tau_defer"],
            "alpha_store": global_gate["alpha_store"],
            "alpha_defer": global_gate["alpha_defer"],
            "cal_n_pooled": global_gate["cal_n"],
            "store_n_pooled": global_gate["store_n"],
            "store_precision_pooled": global_gate["store_precision"],
            "defer_n_pooled": global_gate["defer_n"],
            "defer_precision_pooled": global_gate["defer_precision"],
            "applied_per_benchmark": per_bm_under_global,
        },
        "per_benchmark_gate": per_bm_gates,
        "interpretation": {
            "summary": (
                "Compares the global conformal gate (cycle_{N}/conformal_gate.json) "
                "against benchmark-conditional gates fit at the same precision "
                "targets (α_store=0.05, α_defer=0.40). Per-benchmark gate's "
                "tau_store < global tau_store on a benchmark indicates the global "
                "gate is over-strict for that benchmark — the upstream cause of "
                "the cycle-2 stream-chunk asymmetry (FEVER 15.60 %, TriviaQA 0 %)."
            ),
        },
    }

    out_path = args.root / f"cycle_{args.cycle}" / "conditional_conformal_ablation.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("wrote %s", out_path)

    # Print a small comparison table to stdout for immediate inspection.
    print()
    print("=" * 100)
    print(f"CONDITIONAL CONFORMAL ABLATION — cycle {args.cycle}, α_store={args.alpha_store}, α_defer={args.alpha_defer}")
    print("=" * 100)
    print(f"{'benchmark':<22s} {'n':>4s} {'em_rate':>8s} | "
          f"{'τ_store':>8s} {'store_n':>8s} {'store_prec':>11s} {'store_rate':>11s} | "
          f"{'τ_defer':>8s} {'defer_n':>8s} {'defer_prec':>11s}")
    print("-" * 105)

    g = report["global_gate"]
    print(f"{'GLOBAL (pooled)':<22s} {g['cal_n_pooled']:>4d} {'':>8s} | "
          f"{g['tau_store']:>8.4f} {g['store_n_pooled']:>8d} "
          f"{100*g['store_precision_pooled']:>10.2f}% {'-':>11s} | "
          f"{g['tau_defer']:>8.4f} {g['defer_n_pooled']:>8d} "
          f"{100*g['defer_precision_pooled']:>10.2f}%")
    print()
    print("--- per-benchmark gates (at same α_store=0.05) ---")
    for bench, gd in report["per_benchmark_gate"].items():
        if gd.get("fit_status") != "ok":
            print(f"{bench:<22s} {gd['n']:>4d} "
                  f"{100*gd['em_rate']:>7.2f}% | (fit failed: {gd['fit_status']})")
            continue
        print(f"{bench:<22s} {gd['n']:>4d} {100*gd['em_rate']:>7.2f}% | "
              f"{gd['tau_store']:>8.4f} {gd['store_n']:>8d} "
              f"{100*gd['store_precision']:>10.2f}% {100*gd['store_rate']:>10.2f}% | "
              f"{gd['tau_defer']:>8.4f} {gd['defer_n']:>8d} "
              f"{100*gd['defer_precision']:>10.2f}%")
    print()
    print("--- global gate applied to each benchmark (asymmetry-presenting view) ---")
    for bench, gd in report["global_gate"]["applied_per_benchmark"].items():
        sp = gd['store_precision']
        sp_str = f"{100*sp:>10.2f}%" if sp == sp else f"{'n/a':>11s}"
        dp = gd['defer_precision']
        dp_str = f"{100*dp:>10.2f}%" if dp == dp else f"{'n/a':>11s}"
        print(f"{bench:<22s} {'':>4s} {'':>8s} | "
              f"{g['tau_store']:>8.4f} {gd['store_n']:>8d} "
              f"{sp_str} {100*gd['store_rate']:>10.2f}% | "
              f"{g['tau_defer']:>8.4f} {gd['defer_n']:>8d} "
              f"{dp_str}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
