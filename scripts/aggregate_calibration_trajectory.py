#!/usr/bin/env python
"""
scripts/aggregate_calibration_trajectory.py
============================================
Per-cycle calibration trajectory aggregator — collects every cycle's
calibrated_thresholds, calibrated_config (T), store rate, memory size,
and (where available) purity α(t) into a single CSV/LaTeX table.

Generates the new thesis Table 5.X "Per-cycle calibration trajectory"
and the matching τ-trajectory + store-rate combined figure.

Inputs
------
    --run_dir outputs/full_run                # cycle_0/, cycle_1/, ...
    --cycle_0_dir outputs/cycle_0             # Step 7.0 baseline calibration

Outputs
-------
    --csv_out  outputs/full_run/calibration_trajectory.csv
    --tex_out  "pre thesis 1 report/tables/tab_calibration_trajectory.tex"
    --fig_out  "pre thesis 1 report/figures/fig_tau_trajectory.pdf"  (optional)

The CSV columns:
    cycle, tau_store, tau_defer, tau_train, T,
    store_rate, defer_rate, discard_rate, abstain_rate,
    n_stored_this_cycle, memory_size_total, alpha_purity (if computed)

Usage
-----
    python scripts/aggregate_calibration_trajectory.py \\
        --run_dir outputs/full_run \\
        --cycle_0_dir outputs/cycle_0 \\
        --csv_out outputs/full_run/calibration_trajectory.csv \\
        --tex_out "pre thesis 1 report/tables/tab_calibration_trajectory.tex"

Run AFTER step_7_main completes (or after each cycle for incremental
checkpointing).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_thresholds(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
        # Two possible shapes: top-level or nested under 'thresholds'
        t = d.get("thresholds") or d
        return {
            "store": float(t.get("store", 0)),
            "defer": float(t.get("defer", 0)),
            "train": float(t.get("train", 0)),
        }
    except Exception:
        return None


def _read_temperature(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
        # calibrated_config_cycle{N}.json uses temperature_after,
        # calibrated_config.json uses temperature_scalar
        if "temperature_after" in d:
            return float(d["temperature_after"])
        if "temperature_scalar" in d:
            return float(d["temperature_scalar"])
    except Exception:
        pass
    return None


def _read_decision_distribution(eval_dir: Path) -> Dict[str, float]:
    """Read all eval JSONs in eval_dir, count decisions, return rates."""
    if not eval_dir.exists():
        return {}
    counts = {"STORE": 0, "DEFERRED": 0, "DISCARD": 0, "ABSTAIN": 0}
    total = 0
    for p in eval_dir.glob("*.json"):
        try:
            d = json.load(open(p))
            for s in d.get("samples") or d.get("results") or []:
                dec = s.get("decision")
                if dec in counts:
                    counts[dec] += 1
                    total += 1
        except Exception:
            continue
    if total == 0:
        return {}
    return {f"{k.lower()}_rate": counts[k] / total for k in counts}


def _read_memory_size(meta_path: Path) -> Optional[int]:
    if not meta_path.exists():
        return None
    try:
        import pickle
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            md = data.get("metadata", {})
            return len(md) if isinstance(md, dict) else None
    except Exception:
        return None


def _read_alpha_purity(purity_path: Path) -> Optional[float]:
    """Optional: per-cycle purity α(t) if Step 19 produced one."""
    if not purity_path.exists():
        return None
    try:
        d = json.load(open(purity_path))
        return float(d.get("alpha", d.get("purity", 0)))
    except Exception:
        return None


def collect_trajectory(run_dir: Path, cycle_0_dir: Path) -> List[Dict[str, Any]]:
    """Walk cycle_0/, cycle_1/, ... and assemble per-cycle records."""
    rows: List[Dict[str, Any]] = []

    # Cycle 0 baseline (Step 7.0 calibration)
    c0_thresh = _read_thresholds(cycle_0_dir / "calibrated_thresholds.json")
    c0_t      = _read_temperature(cycle_0_dir / "calibration" / "calibrated_config.json")
    c0_decs   = _read_decision_distribution(cycle_0_dir / "eval")
    rows.append({
        "cycle": 0,
        "tau_store": c0_thresh["store"] if c0_thresh else None,
        "tau_defer": c0_thresh["defer"] if c0_thresh else None,
        "tau_train": c0_thresh["train"] if c0_thresh else None,
        "T":         c0_t,
        **c0_decs,
        "n_stored_this_cycle": None,  # cycle 0 is eval-only
        "memory_size_total": None,
        "alpha_purity":      None,
    })

    # Cycles 1..N from the full_run output (skip cycle_0 if it duplicates cycle_0_dir)
    if run_dir.exists():
        cycle_dirs = sorted(run_dir.glob("cycle_*"), key=lambda p: int(p.name.split("_")[1]))
        for cdir in cycle_dirs:
            n = int(cdir.name.split("_")[1])
            # Skip cycle_0 in run_dir if cycle_0_dir already covered it
            if n == 0:
                continue
            row = {
                "cycle": n,
                "tau_store": None, "tau_defer": None, "tau_train": None,
                "T": None,
                "store_rate": None, "defer_rate": None,
                "discard_rate": None, "abstain_rate": None,
                "n_stored_this_cycle": None,
                "memory_size_total": None,
                "alpha_purity": None,
            }
            # Thresholds (adaptive mode writes per-cycle file)
            t = _read_thresholds(cdir / "calibrated_thresholds.json")
            if t:
                row["tau_store"] = t["store"]
                row["tau_defer"] = t["defer"]
                row["tau_train"] = t["train"]
            # Temperature (per-cycle T re-fit writes calibrated_config_cycle{N}.json)
            calib_dir = run_dir / "calibration"
            T = _read_temperature(calib_dir / f"calibrated_config_cycle{n}.json")
            if T is not None:
                row["T"] = T
            # Decision distribution (from eval JSONs in this cycle)
            decs = _read_decision_distribution(cdir / "eval")
            row.update(decs)
            # Memory size (from store snapshot if present)
            ms = _read_memory_size(cdir / "memory_store.meta")
            if ms is None:
                ms = _read_memory_size(cdir / f"memory_store_cycle_{n}.meta")
            row["memory_size_total"] = ms
            # Purity (if Step 19 ran for this cycle)
            ap = _read_alpha_purity(cdir / "purity_validation.json")
            row["alpha_purity"] = ap
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Union of all keys across rows; cycle-0 may be missing some that cycle-N has
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] wrote {out}")


def write_latex(rows: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    def fmt(v, prec=3):
        if v is None:
            return "--"
        if isinstance(v, float):
            return f"{v:.{prec}f}"
        return str(v)
    def fmt_pct(v):
        if v is None:
            return "--"
        return f"{100*v:.1f}\\%"
    lines = [
        r"% Auto-generated by scripts/aggregate_calibration_trajectory.py",
        r"% Ch 5 §Per-cycle calibration trajectory.",
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{Per-cycle calibration trajectory under adaptive thresholds + adaptive T."
        r" Demonstrates self-tuning behavior across SIL cycles.}",
        r"\label{tab:calibration_trajectory}",
        r"\begin{tabular}{rrrrr|rrrr|rr}",
        r"\toprule",
        r"\textbf{Cycle} & $\tau_{\mathrm{store}}$ & $\tau_{\mathrm{defer}}$ & $\tau_{\mathrm{train}}$ & $T$ & "
        r"\textbf{store} & \textbf{defer} & \textbf{discard} & \textbf{abstain} & "
        r"\textbf{Memory} & $\alpha(t)$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['cycle']} & "
            f"{fmt(r['tau_store'])} & {fmt(r['tau_defer'])} & {fmt(r['tau_train'])} & "
            f"{fmt(r['T'])} & "
            f"{fmt_pct(r.get('store_rate'))} & {fmt_pct(r.get('deferred_rate'))} & "
            f"{fmt_pct(r.get('discard_rate'))} & {fmt_pct(r.get('abstain_rate'))} & "
            f"{r.get('memory_size_total') or '--'} & "
            f"{fmt(r.get('alpha_purity'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"[tex] wrote {out}")


def write_figure(rows: List[Dict[str, Any]], out: Path) -> None:
    """τ trajectory + store-rate combined plot."""
    if not out:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[fig] matplotlib unavailable ({exc}); skipping figure generation")
        return

    cycles = [r["cycle"] for r in rows]
    tau_store = [r["tau_store"] for r in rows]
    tau_defer = [r["tau_defer"] for r in rows]
    tau_train = [r["tau_train"] for r in rows]
    store_rate = [r.get("store_rate") for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    # Filter None values for plotting
    valid_idx = [i for i, t in enumerate(tau_store) if t is not None]
    cs = [cycles[i] for i in valid_idx]
    if cs:
        ax1.plot(cs, [tau_store[i] for i in valid_idx], "o-", label=r"$\tau_{\mathrm{store}}$", color="C0")
        ax1.plot(cs, [tau_defer[i] for i in valid_idx], "s--", label=r"$\tau_{\mathrm{defer}}$", color="C1")
        ax1.plot(cs, [tau_train[i] for i in valid_idx], "^:", label=r"$\tau_{\mathrm{train}}$", color="C2")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel(r"Threshold value $\tau$")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Co-plot store rate on right axis
    valid_sr = [(c, sr) for c, sr in zip(cycles, store_rate) if sr is not None]
    if valid_sr:
        ax2 = ax1.twinx()
        ax2.plot([c for c, _ in valid_sr], [sr for _, sr in valid_sr],
                 "x-.", color="C3", label="Store rate")
        ax2.set_ylabel("Store rate", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        ax2.legend(loc="upper right")

    plt.title("Adaptive threshold trajectory + store rate (target $\\sim30\\%$)")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[fig] wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Directory with cycle_N/ subdirs (e.g. outputs/full_run).")
    p.add_argument("--cycle_0_dir", type=Path, default=Path("outputs/cycle_0"),
                   help="Step 7.0 cycle-0 baseline directory.")
    p.add_argument("--csv_out", type=Path, required=True)
    p.add_argument("--tex_out", type=Path, default=None)
    p.add_argument("--fig_out", type=Path, default=None,
                   help="Optional: write tau trajectory PDF here.")
    ns = p.parse_args()

    rows = collect_trajectory(ns.run_dir, ns.cycle_0_dir)
    if not rows:
        print(f"[error] no trajectory data found in {ns.run_dir}", file=sys.stderr)
        return 1

    write_csv(rows, ns.csv_out)
    if ns.tex_out:
        write_latex(rows, ns.tex_out)
    if ns.fig_out:
        write_figure(rows, ns.fig_out)

    print(f"[info] aggregated {len(rows)} cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
