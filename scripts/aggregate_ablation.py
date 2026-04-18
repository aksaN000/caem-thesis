"""
scripts/aggregate_ablation.py
=============================
Aggregate cyclic-ablation per-seed runs into the Chapter 5 ablation table.

Input layout (produced by ``run_cyclic_ablation.py``):

    outputs/ablation/
        <variant>/
            seed_42/
                ces_axes_per_cycle.json
                run_manifest.json
                experiment_summary.csv
            seed_123/  (optional, Phase 2)
            seed_456/  (optional, Phase 2)

Output
------
1. ``outputs/ablation/ablation_table.csv`` — one row per variant,
   columns: cycles, n_seeds, acc_mean[_std], epi_mean[_std], ret_mean[_std],
   cal_mean[_std], ver_mean[_std], ces_mean[_std], delta_ces_vs_full[_std].
   Single-seed columns emit the mean only with std=0 for downstream safety.
2. ``outputs/ablation/ablation_per_cycle.csv`` — one row per (variant, cycle)
   for per-cycle CES trajectory plots.
3. Stdout: sorted CES ranking, largest ΔCES vs. full first (= most
   load-bearing mechanism). Phase-1 and Phase-2 runs use the same output
   schema; std columns populate automatically once ≥2 seeds exist.

Usage
-----
    # After any number of variant/seed runs (even just one):
    python -m scripts.aggregate_ablation \\
        --output_dir outputs/ablation \\
        --cycle_for_table 10

    # For Phase-1 screening (3 cycles max):
    python -m scripts.aggregate_ablation \\
        --output_dir outputs/ablation \\
        --cycle_for_table 3 \\
        --screening_mode

By default, rows are scored at the **last** cycle present in
``ces_axes_per_cycle.json`` for each variant. ``--cycle_for_table N``
fixes the scoring cycle to N. If a variant has fewer cycles than N, it
falls back to its last available cycle and flags it in the row.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aggregate_ablation")


# =============================================================================
# Data structures
# =============================================================================

AXIS_KEYS = ("acc", "epi", "ret", "cal", "ver", "ces")


def _empty_axis_dict() -> Dict[str, List[float]]:
    return {k: [] for k in AXIS_KEYS}


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if not _is_nan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: List[float], ddof: int = 1) -> float:
    """Sample std; returns 0 for single-element input (downstream-safe)."""
    xs = [x for x in xs if not _is_nan(x)]
    n = len(xs)
    if n <= 1:
        return 0.0
    mu = sum(xs) / n
    sq = sum((x - mu) ** 2 for x in xs)
    return math.sqrt(sq / (n - ddof))


def _is_nan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


# =============================================================================
# Loading
# =============================================================================

def _load_seed_dir(seed_dir: Path) -> Optional[Dict[str, Any]]:
    """Load a per-seed run's CES history + manifest. Returns None on failure."""
    ces_path = seed_dir / "ces_axes_per_cycle.json"
    manifest_path = seed_dir / "run_manifest.json"

    if not ces_path.exists():
        logger.warning("Missing %s; skipping.", ces_path)
        return None

    try:
        with open(ces_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse %s (%s); skipping.", ces_path, exc)
        return None

    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "seed_dir": str(seed_dir),
        "seed": manifest.get("seed", _seed_from_dirname(seed_dir)),
        "variant": manifest.get("variant", seed_dir.parent.name),
        "history": history,
        "manifest": manifest,
    }


def _seed_from_dirname(seed_dir: Path) -> Optional[int]:
    """Parse ``seed_42`` -> 42. Returns None if the dirname doesn't match."""
    name = seed_dir.name
    if name.startswith("seed_"):
        try:
            return int(name[len("seed_"):])
        except ValueError:
            return None
    return None


def _discover_runs(output_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Walk the output root and return {variant: [seed_run, ...]}."""
    runs: Dict[str, List[Dict[str, Any]]] = {}
    if not output_root.exists():
        logger.error("Output root %s does not exist.", output_root)
        return runs

    for variant_dir in sorted(output_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        for seed_dir in sorted(variant_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            if _seed_from_dirname(seed_dir) is None:
                continue
            rec = _load_seed_dir(seed_dir)
            if rec is not None:
                runs.setdefault(variant, []).append(rec)

    return runs


# =============================================================================
# Per-cycle axis extraction
# =============================================================================

def _axes_at_cycle(
    history: List[Dict[str, Any]],
    cycle: Optional[int],
) -> Optional[Tuple[Dict[str, float], int]]:
    """Return aggregate axes dict + actual cycle used.

    If cycle is None, picks the last cycle present.
    If cycle is specified but missing, falls back to the last available.
    """
    if not history:
        return None
    # Filter out entries with a missing/None cycle index — otherwise a later
    # max(by_cycle) would TypeError on a mix of int and None keys.
    by_cycle = {
        h.get("cycle"): h
        for h in history
        if h.get("aggregate_axes") and h.get("cycle") is not None
    }
    if not by_cycle:
        return None

    if cycle is None:
        actual = max(by_cycle)
    elif cycle in by_cycle:
        actual = cycle
    else:
        actual = max(by_cycle)
        logger.info(
            "  cycle=%d not present; falling back to cycle=%d.", cycle, actual,
        )
    agg = by_cycle[actual].get("aggregate_axes", {})
    return agg, actual


# =============================================================================
# Per-variant aggregation across seeds
# =============================================================================

def _aggregate_variant(
    seed_runs: List[Dict[str, Any]],
    *,
    cycle_for_table: Optional[int],
) -> Dict[str, Any]:
    """Collapse multiple seed runs for one variant into a row dict."""
    per_seed_axes: Dict[str, List[float]] = _empty_axis_dict()
    seeds_used: List[int] = []
    cycles_used: List[int] = []

    for run in seed_runs:
        picked = _axes_at_cycle(run["history"], cycle_for_table)
        if picked is None:
            continue
        agg, actual_cycle = picked
        for k in AXIS_KEYS:
            try:
                per_seed_axes[k].append(float(agg.get(k, float("nan"))))
            except (TypeError, ValueError):
                per_seed_axes[k].append(float("nan"))
        if run.get("seed") is not None:
            seeds_used.append(int(run["seed"]))
        cycles_used.append(actual_cycle)

    row: Dict[str, Any] = {
        "n_seeds": len(seeds_used),
        "seeds": sorted(set(seeds_used)),
        "cycles": sorted(set(cycles_used)),
    }
    for k in AXIS_KEYS:
        row[f"{k}_mean"] = _mean(per_seed_axes[k])
        row[f"{k}_std"] = _std(per_seed_axes[k])
    return row


# =============================================================================
# Table builders
# =============================================================================

def _fmt(x: float, digits: int = 4) -> str:
    if _is_nan(x):
        return "NaN"
    return f"{x:.{digits}f}"


def _build_summary_rows(
    variant_rows: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach ΔCES vs. full and sort. 'full' is the reference anchor."""
    full_row = variant_rows.get("full")
    full_ces_mean = full_row["ces_mean"] if full_row else float("nan")
    full_ces_std = full_row["ces_std"] if full_row else 0.0

    out = []
    for name, row in variant_rows.items():
        ces = row["ces_mean"]
        if not _is_nan(ces) and not _is_nan(full_ces_mean):
            delta = ces - full_ces_mean
            # Propagate std conservatively (independent seeds approximation):
            # std(delta) = sqrt(std_v^2 + std_full^2)
            delta_std = math.sqrt(row["ces_std"] ** 2 + full_ces_std ** 2)
        else:
            delta = float("nan")
            delta_std = 0.0
        enriched = dict(row)
        enriched["variant"] = name
        enriched["delta_ces_mean"] = delta
        enriched["delta_ces_std"] = delta_std
        out.append(enriched)

    # Sort: full first, then ascending CES (most-damaged mechanism at top
    # of "what the paper cares about"). Reviewers read top-down.
    def _sort_key(r: Dict[str, Any]) -> Tuple[int, float]:
        if r["variant"] == "full":
            return (0, 0.0)
        ces = r["ces_mean"]
        return (1, float("inf") if _is_nan(ces) else ces)

    out.sort(key=_sort_key)
    return out


def _write_summary_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "variant", "n_seeds", "seeds", "cycles",
        "acc_mean", "acc_std",
        "epi_mean", "epi_std",
        "ret_mean", "ret_std",
        "cal_mean", "cal_std",
        "ver_mean", "ver_std",
        "ces_mean", "ces_std",
        "delta_ces_mean", "delta_ces_std",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fieldnames}
            # Stringify list fields for CSV
            row["seeds"] = "|".join(str(s) for s in r.get("seeds", []))
            row["cycles"] = "|".join(str(c) for c in r.get("cycles", []))
            # Round float fields for readability
            for k in fieldnames:
                if k.endswith("_mean") or k.endswith("_std"):
                    v = row.get(k, "")
                    if isinstance(v, float):
                        row[k] = round(v, 4) if not _is_nan(v) else ""
            w.writerow(row)
    logger.info("Summary CSV -> %s", csv_path)


def _write_per_cycle_csv(
    runs: Dict[str, List[Dict[str, Any]]],
    csv_path: Path,
) -> None:
    """Emit (variant, cycle, seed, acc, epi, ret, cal, ver, ces) rows."""
    fieldnames = ["variant", "cycle", "seed"] + [k for k in AXIS_KEYS]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for variant, seed_runs in sorted(runs.items()):
            for run in seed_runs:
                seed = run.get("seed", "")
                for rec in run["history"]:
                    agg = rec.get("aggregate_axes") or {}
                    row = {
                        "variant": variant,
                        "cycle": rec.get("cycle", ""),
                        "seed": seed,
                    }
                    for k in AXIS_KEYS:
                        v = agg.get(k, "")
                        if isinstance(v, float) and _is_nan(v):
                            v = ""
                        elif isinstance(v, float):
                            v = round(v, 4)
                        row[k] = v
                    w.writerow(row)
    logger.info("Per-cycle CSV -> %s", csv_path)


def _print_ranking(rows: List[Dict[str, Any]]) -> None:
    """Pretty-print the ranking table to stdout."""
    n_seeds_max = max((r["n_seeds"] for r in rows), default=1)
    show_std = n_seeds_max >= 2

    print()
    print("=" * 88)
    print(
        "ABLATION RANKING (lower CES = more load-bearing mechanism)"
        + (f"  [n_seeds ≤ {n_seeds_max}]" if n_seeds_max else "")
    )
    print("=" * 88)

    hdr = f"{'Variant':<28} {'seeds':>5} {'cyc':>4} {'CES':>9} {'ΔCES':>9}"
    if show_std:
        hdr += f" {'±std':>7}"
    print(hdr)
    print("-" * 88)

    for r in rows:
        name = r["variant"]
        seeds_n = r["n_seeds"]
        cycles = r.get("cycles", [])
        cyc_str = str(max(cycles)) if cycles else "-"
        ces = _fmt(r["ces_mean"])
        delta = _fmt(r["delta_ces_mean"])
        line = f"{name:<28} {seeds_n:>5} {cyc_str:>4} {ces:>9} {delta:>9}"
        if show_std:
            line += f" {_fmt(r['ces_std']):>7}"
        print(line)
    print("=" * 88)


# =============================================================================
# Entry point
# =============================================================================

def aggregate(ns: argparse.Namespace) -> None:
    output_root = Path(ns.output_dir)
    runs = _discover_runs(output_root)

    if not runs:
        logger.error(
            "No variant runs found under %s. "
            "Did you run scripts/run_cyclic_ablation.py yet?",
            output_root,
        )
        sys.exit(1)

    logger.info(
        "Discovered %d variant(s) with %d total seed run(s).",
        len(runs), sum(len(rs) for rs in runs.values()),
    )

    # Cycle selection
    cycle_for_table: Optional[int]
    if ns.screening_mode:
        cycle_for_table = 3
    elif ns.cycle_for_table is not None:
        cycle_for_table = ns.cycle_for_table
    else:
        cycle_for_table = None

    # Per-variant aggregation
    variant_rows: Dict[str, Dict[str, Any]] = {}
    for variant, seed_runs in runs.items():
        row = _aggregate_variant(seed_runs, cycle_for_table=cycle_for_table)
        variant_rows[variant] = row

    summary_rows = _build_summary_rows(variant_rows)

    # Write outputs
    output_root.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(summary_rows, output_root / "ablation_table.csv")
    _write_per_cycle_csv(runs, output_root / "ablation_per_cycle.csv")

    # Stdout ranking
    _print_ranking(summary_rows)

    # Manifest
    meta = {
        "output_root": str(output_root),
        "cycle_for_table": cycle_for_table,
        "n_variants": len(runs),
        "n_seed_runs_total": sum(len(rs) for rs in runs.values()),
        "variants": {v: [r["seed"] for r in rs] for v, rs in runs.items()},
    }
    with open(output_root / "ablation_aggregate_manifest.json", "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Manifest -> %s", output_root / "ablation_aggregate_manifest.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-seed cyclic-ablation runs into the Chapter 5 table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ablation",
        help="Root directory containing <variant>/seed_<N>/ subdirs.",
    )
    p.add_argument(
        "--cycle_for_table",
        type=int,
        default=None,
        help="Cycle to score at. Default: last cycle present per variant.",
    )
    p.add_argument(
        "--screening_mode",
        action="store_true",
        help="Score at cycle 3 (overrides --cycle_for_table).",
    )
    return p.parse_args()


def main() -> None:
    aggregate(_parse_args())


if __name__ == "__main__":
    main()
