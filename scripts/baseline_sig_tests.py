#!/usr/bin/env python3
"""
scripts/baseline_sig_tests.py
=============================
CAEM-vs-baseline Holm-corrected paired significance tests.

Produces ``outputs/tab_sig_test.csv`` -- the table chapter_5.tex references
as ``\\ref{tab:sig-test}`` (Table 5.7). Not to be confused with
``outputs/full_run/eval/tab_cycle_progression.csv``, which is the within-CAEM
cycle-over-cycle progression produced by
``eval/reporting.py::build_table_cycle_progression``.

Pairing rules (chapter_5.tex §5.3)
----------------------------------
All baselines are paired against **CAEM cycle 10** — the post-training
state. For inference-only baselines (B1-B5: ``zero_shot``, ``cot``,
``rag``, ``cot_rag``, ``flare``) there is no training budget to match,
so the comparison is "CAEM's final state vs. the off-the-shelf recipe"
— the strongest CAEM-positive claim. For training baselines (B6
``vanilla_ft``, B7 ``ewc_only_ft``) cycle 10 additionally gives matched
training budget (both sides have completed the same number of fine-tuning
cycles on matched per-benchmark pair counts). Within-CAEM cycle-over-cycle
trajectory is reported separately in ``tab_cycle_progression.csv``.

Pairing is by per-sample ``id``, restricted to the authoritative eval
split read from ``<caem_dir>/dataset_splits.json`` (written by
``run_experiment.py`` as the single source of truth for which sample IDs
belong to the eval slice). This is belt-and-suspenders: even if
``load_benchmark(bench, n, split)`` were to drift across re-invocations,
the sig-test only considers ids that CAEM's own split metadata marked
as eval. When the splits file is missing (e.g. pre-CAEM-run), the script
falls back to raw id-intersection between CAEM and baseline JSONs with
a warning.

Correction
----------
Holm-Bonferroni step-down is applied **independently within each benchmark
family** with ``m = number of baselines present on disk`` (NOT hard-coded 8 --
the thesis pre-registration was rewritten to reference the code's dynamic
family size). A result earns the star marker only under the symmetric
``|ΔEM| >= 0.02`` practical-significance floor AND Holm-corrected ``p < 0.05``.

Output schema
-------------
Columns: baseline, benchmark, n_paired, em_caem, em_baseline, em_diff,
         ci_low, ci_high, mcnemar_chi2, mcnemar_p_raw, holm_p,
         dagger_marker, star_marker, baseline_wins.

Usage
-----
    python -m scripts.baseline_sig_tests \\
        --caem_dir outputs/full_run \\
        --baselines_dir outputs/baselines \\
        --output_csv outputs/tab_sig_test.csv

    # Synthetic self-check, no filesystem inputs:
    python -m scripts.baseline_sig_tests --smoke_test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make repo root importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import bootstrap_ci, mcnemar_test  # noqa: E402

logger = logging.getLogger("baseline_sig_tests")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

BENCHMARKS = [
    "fever", "triviaqa", "natural_questions",
    "truthfulqa", "strategyqa", "arc_challenge", "asqa",
]

# All baselines pair against CAEM cycle 10 (post-training, strongest-claim
# state). The two constants below are kept separate so a future variant
# (e.g., trajectory-style supplementary table that also reports cycle 3)
# can flip the inference cycle without touching the training cycle, but
# Phase 1A uses cycle 10 for both.
TRAINING_BASELINES = {"vanilla_ft", "ewc_only_ft"}
CAEM_CYCLE_INFERENCE = 10
CAEM_CYCLE_TRAINING = 10

OUTPUT_FIELDS = [
    "baseline", "benchmark", "n_paired",
    "em_caem", "em_baseline", "em_diff",
    "ci_low", "ci_high",
    "mcnemar_chi2", "mcnemar_p_raw",
    "holm_p", "dagger_marker", "star_marker", "baseline_wins",
]


def _load_id_to_em(json_path: Path) -> Dict[str, float]:
    """Return ``{sample_id: em}`` from a harness-format cycle JSON."""
    with json_path.open("r", encoding="utf-8") as f:
        blob = json.load(f)
    out: Dict[str, float] = {}
    for s in blob.get("samples", []):
        sid = s.get("id")
        if sid is None or sid == "":
            continue
        out[str(sid)] = float(s.get("em", 0.0))
    return out


def _load_eval_ids(caem_dir: Path) -> Optional[Dict[str, set]]:
    """Read authoritative eval ids from ``<caem_dir>/dataset_splits.json``.

    Returns ``{benchmark: set(eval_ids)}`` when the file is present,
    ``None`` when it is missing (callers fall back to raw id-intersection).
    """
    splits_path = caem_dir / "dataset_splits.json"
    if not splits_path.is_file():
        logger.warning(
            "dataset_splits.json not found at %s -- falling back to raw "
            "id-intersection (no authoritative eval-split filter).",
            splits_path,
        )
        return None
    try:
        with splits_path.open("r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load %s (%s) -- falling back to raw "
                       "id-intersection.", splits_path, exc)
        return None
    out: Dict[str, set] = {}
    for bm, meta in blob.items():
        ids = meta.get("eval_ids", []) if isinstance(meta, dict) else []
        out[bm] = {str(i) for i in ids}
    logger.info("Loaded eval-id splits for %d benchmarks from %s",
                len(out), splits_path)
    return out


def _baseline_json_candidates(baselines_dir: Path, baseline: str,
                              benchmark: str) -> List[Path]:
    """Per-sample JSON path candidates for a (baseline, benchmark) pair.

    Inference baselines (``run_baseline.py``) write
    ``<baselines_dir>/<baseline>/<benchmark>_cycle0.json``.
    Training baselines (``run_simple_ft.py``) write
    ``<baselines_dir>/<baseline>/<baseline>/eval/<benchmark>_cycle<N>.json``.
    We try the training-cycle snapshot first (cycle 10) and fall back to
    cycle 0 so a partial run still produces a row.
    """
    c: List[Path] = []
    c.append(baselines_dir / baseline / f"{benchmark}_cycle0.json")
    for cyc in (CAEM_CYCLE_TRAINING, 0):
        c.append(baselines_dir / baseline / baseline / "eval"
                 / f"{benchmark}_cycle{cyc}.json")
    return c


def _pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def _holm_correct(p_raw: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction.

    Input: list of raw p-values from a single family (within one benchmark).
    Output: list of corrected p-values aligned 1:1 with the input.
    m = len(p_raw) = family size (number of baselines in this benchmark).

    Standard Holm: sort ascending, adjust k-th smallest by factor (m - k),
    enforce monotonicity via running max. Aligned back to input order.
    """
    m = len(p_raw)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_raw[i])
    adj: List[float] = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        scaled = min(1.0, p_raw[idx] * (m - rank))
        running_max = max(running_max, scaled)
        adj[idx] = running_max
    return adj


def _compute_rows(caem_dir: Path, baselines_dir: Path) -> List[Dict]:
    """Build the (baseline x benchmark) rows with Holm-within-family correction."""
    if not baselines_dir.is_dir():
        logger.warning("baselines_dir does not exist: %s", baselines_dir)
        return []
    baselines = sorted(
        d.name for d in baselines_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not baselines:
        logger.warning("No baseline subdirectories under %s", baselines_dir)
        return []
    logger.info("Discovered %d baselines: %s", len(baselines), baselines)

    caem_eval = caem_dir / "eval"
    eval_ids_by_bench = _load_eval_ids(caem_dir)

    caem_cache: Dict[Tuple[str, int], Dict[str, float]] = {}

    def _load_caem(bench: str, cyc: int) -> Dict[str, float]:
        key = (bench, cyc)
        if key in caem_cache:
            return caem_cache[key]
        p = caem_eval / f"{bench}_cycle{cyc}.json"
        if not p.is_file():
            logger.warning("CAEM eval JSON missing: %s", p)
            caem_cache[key] = {}
            return {}
        caem_cache[key] = _load_id_to_em(p)
        return caem_cache[key]

    def _allowed_ids(bench: str) -> Optional[set]:
        """Return the authoritative eval_ids set for a benchmark, or None
        if dataset_splits.json was unavailable (→ fall back to id-intersection
        over both JSONs without split-filter)."""
        if eval_ids_by_bench is None:
            return None
        return eval_ids_by_bench.get(bench)

    # First pass: compute raw stats, group by benchmark family.
    pending: Dict[str, List[Dict]] = {b: [] for b in BENCHMARKS}
    for baseline in baselines:
        cycle = (CAEM_CYCLE_TRAINING if baseline in TRAINING_BASELINES
                 else CAEM_CYCLE_INFERENCE)
        for bench in BENCHMARKS:
            caem_em_map = _load_caem(bench, cycle)
            b_path = _pick_first_existing(_baseline_json_candidates(
                baselines_dir, baseline, bench))
            if not caem_em_map or b_path is None:
                logger.info("Skip %s / %s: missing eval JSON.", baseline, bench)
                continue
            base_em_map = _load_id_to_em(b_path)
            # Restrict to authoritative eval ids when available.
            allowed = _allowed_ids(bench)
            if allowed is not None:
                caem_em_map = {i: v for i, v in caem_em_map.items()
                               if i in allowed}
                base_em_map = {i: v for i, v in base_em_map.items()
                               if i in allowed}
                if not caem_em_map:
                    logger.info("Skip %s / %s: CAEM JSON has no ids in "
                                "dataset_splits.json eval_ids (%d allowed).",
                                baseline, bench, len(allowed))
                    continue
            shared_ids = sorted(set(caem_em_map) & set(base_em_map))
            if len(shared_ids) < 10:
                logger.info("Skip %s / %s: only %d paired ids (need >= 10).",
                            baseline, bench, len(shared_ids))
                continue

            caem_em = [caem_em_map[i] for i in shared_ids]
            base_em = [base_em_map[i] for i in shared_ids]
            diffs = [c - b for c, b in zip(caem_em, base_em)]

            try:
                _mean, ci_lo, ci_hi = bootstrap_ci(
                    diffs, n_bootstrap=1000, ci=0.95, seed=42)
            except Exception as exc:
                logger.warning("bootstrap_ci failed (%s/%s): %s",
                               baseline, bench, exc)
                ci_lo = ci_hi = float("nan")
            try:
                chi2, p_raw = mcnemar_test(caem_em, base_em)
            except Exception as exc:
                logger.warning("mcnemar_test failed (%s/%s): %s",
                               baseline, bench, exc)
                chi2 = float("nan")
                p_raw = float("nan")

            em_caem_mean = sum(caem_em) / len(caem_em)
            em_base_mean = sum(base_em) / len(base_em)
            em_diff = em_caem_mean - em_base_mean

            def _r(v: float, d: int = 4) -> Optional[float]:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return None
                return round(float(v), d)

            pending[bench].append({
                "baseline": baseline,
                "benchmark": bench,
                "n_paired": len(shared_ids),
                "em_caem": _r(em_caem_mean),
                "em_baseline": _r(em_base_mean),
                "em_diff": _r(em_diff),
                "ci_low": _r(ci_lo),
                "ci_high": _r(ci_hi),
                "mcnemar_chi2": _r(chi2),
                "mcnemar_p_raw": _r(p_raw, 6),
            })

    # Second pass: Holm within each benchmark family.
    rows: List[Dict] = []
    for bench in BENCHMARKS:
        family = pending[bench]
        # Raw p for Holm; replace missing with 1.0 so they don't trigger.
        safe_ps = [1.0 if r["mcnemar_p_raw"] is None else r["mcnemar_p_raw"]
                   for r in family]
        adj = _holm_correct(safe_ps)
        for r, adj_p in zip(family, adj):
            em_diff = r["em_diff"]
            raw_missing = r["mcnemar_p_raw"] is None
            holm_p = None if raw_missing else round(adj_p, 6)
            dagger = bool((holm_p is not None) and holm_p < 0.05)
            star = bool(dagger and em_diff is not None
                        and abs(em_diff) >= 0.02)
            baseline_wins = bool(
                holm_p is not None and holm_p < 0.05
                and em_diff is not None and em_diff <= -0.02)
            r["holm_p"] = holm_p
            r["dagger_marker"] = dagger
            r["star_marker"] = star
            r["baseline_wins"] = baseline_wins
            rows.append(r)

    return rows


def _write_csv(output_csv: Path, rows: List[Dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in OUTPUT_FIELDS})
    logger.info("Wrote %d rows -> %s", len(rows), output_csv)


# -----------------------------------------------------------------------------
# Synthetic self-check
# -----------------------------------------------------------------------------

def _smoke_test() -> None:
    """Built-in checks with no filesystem inputs.

    Scenario A: CAEM right on all 200, baseline right on 150 of the same 200.
                Δ = +0.25, expect dagger AND star.
    Scenario B: CAEM and baseline fully agree. No discordant pairs → not dagger.
    Scenario C: Holm on raw p=[0.01, 0.03, 0.04] must give [0.03, 0.06, 0.06].
    Scenario D: baseline_wins: CAEM worse by 0.25. Dagger AND baseline_wins.
    """
    import random
    random.seed(0)
    n = 200

    # A — CAEM wins.
    caem_A = [1.0] * n
    base_A = [0.0] * 50 + [1.0] * (n - 50)
    random.shuffle(base_A)
    _, p_A = mcnemar_test(caem_A, base_A)
    holm_A = _holm_correct([p_A])[0]
    em_diff_A = sum(caem_A)/n - sum(base_A)/n
    dagger_A = holm_A < 0.05
    star_A = dagger_A and abs(em_diff_A) >= 0.02
    assert dagger_A, f"A: should be dagger, holm_p={holm_A}"
    assert star_A, f"A: should earn star, |Δ|={em_diff_A}"

    # B — equal vectors, no discordant pairs.
    caem_B = [1.0 if random.random() < 0.6 else 0.0 for _ in range(n)]
    base_B = list(caem_B)
    _, p_B = mcnemar_test(caem_B, base_B)
    holm_B = _holm_correct([p_B])[0]
    dagger_B = holm_B < 0.05
    assert not dagger_B, f"B: should NOT be dagger, holm_p={holm_B}"

    # C — Holm arithmetic on three-test family.
    adj = _holm_correct([0.01, 0.03, 0.04])
    expect = [0.03, 0.06, 0.06]
    for a, e in zip(adj, expect):
        assert math.isclose(a, e, abs_tol=1e-9), \
            f"C: Holm wrong. Got {adj}, expected {expect}"

    # D — baseline wins. CAEM right on 50/200, baseline right on all 200.
    caem_D = [1.0] * 50 + [0.0] * 150
    random.shuffle(caem_D)
    base_D = [1.0] * n
    _, p_D = mcnemar_test(caem_D, base_D)
    holm_D = _holm_correct([p_D])[0]
    em_diff_D = sum(caem_D)/n - sum(base_D)/n
    baseline_wins_D = (holm_D < 0.05) and (em_diff_D <= -0.02)
    assert baseline_wins_D, \
        f"D: baseline_wins should fire, holm_p={holm_D}, Δ={em_diff_D}"

    logger.info("Smoke test PASSED.")
    logger.info("  A (Δ=+%.3f): raw=%.4g holm=%.4g dagger=%s star=%s",
                em_diff_A, p_A, holm_A, dagger_A, star_A)
    logger.info("  B (Δ=0):      raw=%.4g holm=%.4g dagger=%s",
                p_B, holm_B, dagger_B)
    logger.info("  C (Holm 3-family input [0.01,0.03,0.04]): %s", adj)
    logger.info("  D (Δ=%.3f):   raw=%.4g holm=%.4g baseline_wins=%s",
                em_diff_D, p_D, holm_D, baseline_wins_D)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--caem_dir", default="outputs/full_run",
                    help="CAEM main-run directory (must contain eval/).")
    ap.add_argument("--baselines_dir", default="outputs/baselines",
                    help="Directory with one subdir per baseline.")
    ap.add_argument("--output_csv", default="outputs/tab_sig_test.csv",
                    help="Where to write the CAEM-vs-baseline table.")
    ap.add_argument("--smoke_test", action="store_true",
                    help="Run the built-in self-check and exit.")
    ns = ap.parse_args()

    if ns.smoke_test:
        _smoke_test()
        return

    rows = _compute_rows(Path(ns.caem_dir), Path(ns.baselines_dir))
    _write_csv(Path(ns.output_csv), rows)


if __name__ == "__main__":
    main()
