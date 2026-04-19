"""
scripts/signal_correlation_matrix.py
=====================================
Compute the 9x9 Spearman cross-signal correlation matrix from CAEM's
per-sample evaluation records -- the empirical object referenced in
Chapter~5 Section "Nine-signal correlation structure and effective-count
analysis".

Input
-----
Per-benchmark per-cycle eval JSONs written by eval.harness, e.g.
``outputs/cycle_0/eval/fever_cycle0.json``. Each ``samples[]`` record
carries the nine signals plus decision + EM:

    u_token, u_dropout, u_internal, s_avg, h_norm, p_entail,
    p_ground_max, p_ground_mean, p_ground_atomic  (9 signals)
    p_contra           (safety veto, optional 10th column)
    em                 (ground truth proxy)

Output
------
Primary:
  <output_dir>/spearman_matrix_aggregated.csv  -- pooled across all benches
  <output_dir>/spearman_matrix_by_benchmark.json -- dict[benchmark] -> 9x9

Secondary (counts for the pre-registered falsification conditions):
  <output_dir>/redundancy_report.json
    {
      "pairs_over_09_aggregated": int,
      "pairs_over_09_any_benchmark": int,
      "pairs_over_09_every_benchmark": int,
      "effective_signal_count_after_prune": int,
      "pruned_away": [ ... ]
    }

Usage
-----
PYTHONPATH=. python scripts/signal_correlation_matrix.py \\
    --eval_jsons outputs/cycle_0/eval/*_cycle0.json \\
    --output_dir outputs/signal_correlation/

The script is dependency-free beyond numpy; Spearman rank correlation is
implemented inline so the diagnostic runs inside the lean Vast Docker
image. The prune-to-effective-count routine uses a union-find over
|rho| > threshold pairs to report the number of independent signals.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("signal_corr")


SIGNAL_NAMES = [
    "u_token", "u_dropout", "u_internal",
    "s_avg", "h_norm", "p_entail",
    "p_ground_max", "p_ground_mean", "p_ground_atomic",
]


def _load_samples(paths: List[Path]) -> Dict[str, List[Dict]]:
    """Group samples by benchmark name."""
    by_bench: Dict[str, List[Dict]] = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        bench = d.get("meta", {}).get("benchmark", p.stem)
        for s in d.get("samples", []):
            if all(k in s for k in SIGNAL_NAMES):
                by_bench.setdefault(bench, []).append(s)
    return by_bench


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average-rank tie handling.

    NaN-safe: drops positions where either input is non-finite.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x = x[ok]; y = y[ok]

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty_like(v, dtype=np.float64)
        r[order] = np.arange(1, len(v) + 1, dtype=np.float64)
        # average ranks across ties
        uniq, inv = np.unique(v, return_inverse=True)
        for u_i in range(len(uniq)):
            idxs = np.where(inv == u_i)[0]
            if len(idxs) > 1:
                r[idxs] = r[idxs].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    # Pearson correlation on ranks = Spearman rho.
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = float(np.sqrt((vx * vx).sum() * (vy * vy).sum()))
    if denom == 0.0:
        return float("nan")
    return float((vx * vy).sum() / denom)


def _build_matrix(samples: List[Dict]) -> np.ndarray:
    N = len(samples)
    if N == 0:
        return np.full((len(SIGNAL_NAMES), len(SIGNAL_NAMES)), np.nan)
    mat = np.zeros((N, len(SIGNAL_NAMES)), dtype=np.float64)
    for i, s in enumerate(samples):
        for j, name in enumerate(SIGNAL_NAMES):
            mat[i, j] = float(s[name])
    K = len(SIGNAL_NAMES)
    out = np.full((K, K), np.nan, dtype=np.float64)
    for i in range(K):
        for j in range(K):
            out[i, j] = 1.0 if i == j else _spearman_rho(mat[:, i], mat[:, j])
    return out


def _pairs_over_threshold(M: np.ndarray, tau: float) -> List[Tuple[int, int, float]]:
    """Return the upper-triangle (i, j, rho) with |rho| >= tau."""
    K = M.shape[0]
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            if np.isfinite(M[i, j]) and abs(M[i, j]) >= tau:
                pairs.append((i, j, float(M[i, j])))
    return pairs


def _effective_count_via_union_find(
    M: np.ndarray, tau: float,
) -> Tuple[int, List[str]]:
    """Collapse high-correlation signals into clusters; count clusters."""
    K = M.shape[0]
    parent = list(range(K))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (i, j, _rho) in _pairs_over_threshold(M, tau):
        union(i, j)

    roots = {find(i) for i in range(K)}
    n_clusters = len(roots)

    # Pruned-away names: everything not the representative of its cluster,
    # chosen deterministically as the earliest index in SIGNAL_NAMES.
    kept = sorted(roots)
    all_idx = set(range(K))
    pruned_idx = sorted(all_idx - set(kept))
    pruned_names = [SIGNAL_NAMES[i] for i in pruned_idx]
    return n_clusters, pruned_names


def _save_csv(M: np.ndarray, path: Path, names: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, nm in enumerate(names):
            row = [nm]
            for j in range(len(names)):
                v = M[i, j]
                row.append("" if not np.isfinite(v) else f"{v:.4f}")
            w.writerow(row)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_jsons", nargs="+", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--rho_threshold", type=float, default=0.9)
    return p.parse_args()


def main() -> None:
    ns = _parse_args()
    expanded: List[Path] = []
    for pat in ns.eval_jsons:
        matches = glob.glob(pat)
        if not matches:
            logger.warning("No matches for %r", pat)
        expanded.extend(Path(m) for m in matches)
    if not expanded:
        logger.error("No eval JSONs. Exiting.")
        sys.exit(1)

    ns.output_dir.mkdir(parents=True, exist_ok=True)
    by_bench = _load_samples(expanded)
    logger.info("Loaded %d benchmarks: %s",
                len(by_bench), list(by_bench.keys()))

    # Per-benchmark matrices
    per_bench_raw: Dict[str, List[List[float]]] = {}
    pairs_any: set = set()
    pairs_all: dict = {frozenset({i, j}): 0 for i in range(len(SIGNAL_NAMES))
                       for j in range(i + 1, len(SIGNAL_NAMES))}
    for bench, samples in by_bench.items():
        M = _build_matrix(samples)
        _save_csv(M, ns.output_dir / f"spearman_{bench}.csv", SIGNAL_NAMES)
        per_bench_raw[bench] = M.tolist()
        for (i, j, _r) in _pairs_over_threshold(M, ns.rho_threshold):
            key = frozenset({i, j})
            pairs_any.add(key)
            pairs_all[key] += 1

    # Aggregated matrix: pool all samples
    all_samples: List[Dict] = [s for ss in by_bench.values() for s in ss]
    agg = _build_matrix(all_samples)
    _save_csv(agg, ns.output_dir / "spearman_matrix_aggregated.csv",
              SIGNAL_NAMES)
    with open(ns.output_dir / "spearman_matrix_by_benchmark.json", "w") as f:
        json.dump(per_bench_raw, f, indent=2)

    # Pre-registered-condition counts
    agg_pairs = _pairs_over_threshold(agg, ns.rho_threshold)
    pairs_every = [k for k, c in pairs_all.items() if c == len(by_bench)]
    effective_count, pruned = _effective_count_via_union_find(
        agg, ns.rho_threshold
    )
    report = {
        "rho_threshold": ns.rho_threshold,
        "n_benchmarks": len(by_bench),
        "pairs_over_threshold_aggregated": [
            {"i": i, "j": j, "a": SIGNAL_NAMES[i], "b": SIGNAL_NAMES[j], "rho": r}
            for (i, j, r) in agg_pairs
        ],
        "n_pairs_over_threshold_aggregated": len(agg_pairs),
        "n_pairs_over_threshold_any_benchmark": len(pairs_any),
        "n_pairs_over_threshold_every_benchmark": len(pairs_every),
        "effective_signal_count_after_prune": effective_count,
        "pruned_names": pruned,
    }
    with open(ns.output_dir / "redundancy_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Wrote matrices + redundancy report to %s", ns.output_dir)
    logger.info("Effective count after pruning: %d (dropped: %s)",
                effective_count, pruned)


if __name__ == "__main__":
    main()
