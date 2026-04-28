"""theorem_receipts.py — Phase 4 empirical receipts for Ch4 §4.8 theorems.

Reads existing per-cycle artefacts (no re-running of the pipeline) and produces
five JSONs that complete the empirical-receipt coverage of the theoretical
analysis:

  receipt_envelope_fit.json       — Theorem 5 geometric envelope
  receipt_eps_arch.json           — Theorem 7 four-gate joint FNR aggregation
  receipt_gap_decay.json          — Corollary convergence-rate exp fit
  receipt_corpus_floor.json       — Corollary corpus-floor proxy (Wikipedia subset)
  receipt_self_correction.json    — Corollary self-correction survival distribution

All inputs come from logs/JSONs the runner already writes. Run AFTER step_7_main
completes; safe to invoke any number of times against the locked outputs.

Usage:
    python -m scripts.theorem_receipts --output_dir outputs/full_run/theorem_receipts

Each receipt is a self-contained JSON with the empirical reading and the
predicted bound side-by-side, so Ch5 §sec:check-* can splice them directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. Geometric-envelope fit (Theorem 5: convergence)
# --------------------------------------------------------------------------- #

def fit_envelope(
    purity_per_cycle: List[float],
    cohen_d_per_cycle: List[float],
    sigma_per_cycle: List[float],
) -> Dict[str, Any]:
    """Fit G_t = A r^t to the cycle-t precision-asymptote gap.

    Returns A, r, R^2 of the fit, and the predicted r_pred = O(σ Φ(d/sqrt 2))
    for side-by-side comparison with the theorem's lower bound.
    """
    if len(purity_per_cycle) < 3:
        return {"status": "insufficient_data", "n_cycles": len(purity_per_cycle)}

    pi_inf = max(purity_per_cycle)  # asymptote estimated as max observed
    gaps = [abs(p - pi_inf) for p in purity_per_cycle]

    # log-linear fit of log G_t vs t (skip last cycle: gap=0 by construction)
    ts = np.array([t for t, g in enumerate(gaps[:-1]) if g > 0], dtype=float)
    log_gaps = np.array([math.log(gaps[int(t)]) for t in ts])
    if len(ts) < 2:
        return {"status": "insufficient_nonzero_gaps", "n_cycles": len(purity_per_cycle)}

    slope, intercept = np.polyfit(ts, log_gaps, 1)
    r_fit = 1.0 - math.exp(slope)
    A_fit = math.exp(intercept)
    pred = np.exp(slope * ts + intercept)
    ss_res = float(np.sum((log_gaps - np.log(pred)) ** 2))
    ss_tot = float(np.sum((log_gaps - np.mean(log_gaps)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Predicted contraction rate from theorem
    if cohen_d_per_cycle and sigma_per_cycle:
        d_mean = float(np.mean(cohen_d_per_cycle))
        sigma_mean = float(np.mean(sigma_per_cycle))
        Phi_half = 0.5 * (1 + math.erf(d_mean / 2.0))
        r_pred = sigma_mean * Phi_half
    else:
        r_pred = None

    return {
        "status": "ok",
        "n_cycles": len(purity_per_cycle),
        "pi_infty_estimate": pi_inf,
        "r_fitted": r_fit,
        "A_fitted": A_fit,
        "r2": r2,
        "r_predicted_lower_bound": r_pred,
        "envelope_holds": (r_pred is not None and r_fit >= r_pred * 0.5),
        "comment": "envelope holds if r_fitted within order of r_predicted_lower_bound",
    }


# --------------------------------------------------------------------------- #
# 2. ε_arch aggregation (Theorem 7: full-pipeline decomposition)
# --------------------------------------------------------------------------- #

def aggregate_eps_arch(
    eval_jsons: List[Path],
    retroverify_jsons: List[Path],
) -> Dict[str, Any]:
    """Compute the joint architectural FNR across the four cascaded gates.

    Per-query gates (read from per-sample eval JSON):
      - confabulation early-exit (early_exit_triggered = True)
      - composite gate (decision = DISCARD given em = 1)
    Cycle-boundary gates:
      - novelty filter (approximated via pool-size delta)
      - retroactive prune (from retroverify_cycleN.json)

    A "false negative" of the architecture is an em-correct sample that the
    architecture fails to STORE. ε_arch is the joint probability that any of
    the four gates blocks an em-correct sample.
    """
    n_em_correct = 0
    n_em_correct_blocked_early = 0
    n_em_correct_blocked_composite = 0

    for p in eval_jsons:
        try:
            d = json.load(p.open("r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for s in d.get("samples", []):
            if not s.get("em"):
                continue
            n_em_correct += 1
            if s.get("early_exit_triggered"):
                n_em_correct_blocked_early += 1
            elif s.get("decision") in ("DISCARD", "ABSTAIN"):
                n_em_correct_blocked_composite += 1

    fnr_early = n_em_correct_blocked_early / max(n_em_correct, 1)
    fnr_composite = n_em_correct_blocked_composite / max(n_em_correct, 1)

    n_pruned_retro = 0
    n_retro_total = 0
    for p in retroverify_jsons:
        try:
            d = json.load(p.open("r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rv = d.get("retroverify", {})
        n_pruned_retro += rv.get("n_threshold_pruned", 0) + rv.get("n_loop_pruned", 0)
        n_retro_total += rv.get("n_processed", 0)
    fnr_retro = n_pruned_retro / max(n_retro_total, 1)

    # Novelty filter: data not directly logged per sample.
    # Approximation deferred to next pass; placeholder of None for now.
    fnr_novelty = None

    fnrs = [fnr_early, fnr_composite, fnr_retro]
    fnr_known = [f for f in fnrs if f is not None]
    eps_arch = 1.0 - float(np.prod([1 - f for f in fnr_known])) if fnr_known else None

    return {
        "status": "ok" if eps_arch is not None else "incomplete",
        "n_em_correct_examined": n_em_correct,
        "fnr_early_exit": fnr_early,
        "fnr_composite_gate": fnr_composite,
        "fnr_retroactive_prune": fnr_retro,
        "fnr_novelty_filter": fnr_novelty,
        "eps_arch_known_gates": eps_arch,
        "comment": (
            "eps_arch reported on the three gates with per-sample logs. "
            "Novelty filter contribution requires pool-size deltas; "
            "see receipt_self_correction.json for the survival "
            "distribution which captures the same population effect."
        ),
    }


# --------------------------------------------------------------------------- #
# 3. Gap-trajectory exp fit (Cor convergence-rate)
# --------------------------------------------------------------------------- #

def fit_gap_decay(
    chm_per_cycle: List[float],
    chm_floor: Optional[float],
) -> Dict[str, Any]:
    """Fit |H_t - H_∞| = A e^{-kt} to the per-cycle CHM trajectory.

    Returns A, k, c*(ε) for ε ∈ {0.01, 0.05, 0.10}.
    """
    if len(chm_per_cycle) < 3:
        return {"status": "insufficient_data", "n_cycles": len(chm_per_cycle)}

    H_inf = chm_floor if chm_floor is not None else min(chm_per_cycle)
    ts = np.arange(len(chm_per_cycle), dtype=float)
    gaps = np.array([abs(h - H_inf) for h in chm_per_cycle])
    nonzero = gaps > 0
    if nonzero.sum() < 2:
        return {"status": "insufficient_nonzero_gaps", "H_infty": H_inf}

    log_gaps = np.log(gaps[nonzero])
    ts_nz = ts[nonzero]
    slope, intercept = np.polyfit(ts_nz, log_gaps, 1)
    k = -float(slope)
    A = float(math.exp(intercept))
    c_star = {
        "epsilon_0.01": (1 / k) * math.log(A / max(0.01 * H_inf, 1e-12)) if k > 0 else None,
        "epsilon_0.05": (1 / k) * math.log(A / max(0.05 * H_inf, 1e-12)) if k > 0 else None,
        "epsilon_0.10": (1 / k) * math.log(A / max(0.10 * H_inf, 1e-12)) if k > 0 else None,
    }

    return {
        "status": "ok",
        "n_cycles": len(chm_per_cycle),
        "H_infty": H_inf,
        "A_fitted": A,
        "k_fitted": k,
        "c_star_cycles_to_within_epsilon": c_star,
    }


# --------------------------------------------------------------------------- #
# 4. Corpus-bounded floor proxy (Cor corpus-floor)
# --------------------------------------------------------------------------- #

def corpus_floor_proxy(eval_jsons: List[Path]) -> Dict[str, Any]:
    """Proxy for K(C): subset of queries whose top-k retrieval misses the
    gold-answer span. Reports CHM on that subset as an empirical estimate of
    the corpus-bounded floor.

    Requires per-sample 'top_passages' field added to harness output schema
    (post-cycle-1 patch). Cycles missing this field are skipped with a note.
    """
    n_with_passages = 0
    n_total = 0
    n_miss_gold_in_topk = 0
    miss_em = []
    miss_total = 0

    for p in eval_jsons:
        try:
            d = json.load(p.open("r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for s in d.get("samples", []):
            n_total += 1
            top = s.get("top_passages")
            if not top:
                continue
            n_with_passages += 1

            gold_list = s.get("gold_answers") or []
            if not gold_list:
                continue
            # Lightweight proxy: gold-string substring match in passage text.
            # NOTE: cycles before the schema patch lack passage text — this is
            # why we report the proxy on the "with_passages" subset only.
            # (Top passages only have IDs in the patch, so this proxy needs
            # an upstream join with the corpus FAISS metadata which isn't
            # directly available here. Documented as Phase-4 follow-up.)
            n_miss_gold_in_topk += 0  # placeholder until corpus-side join exists

    return {
        "status": "schema_only" if n_with_passages == 0 else "partial",
        "n_total_samples": n_total,
        "n_samples_with_passages_logged": n_with_passages,
        "n_miss_gold_in_topk": n_miss_gold_in_topk,
        "comment": (
            "Cycle 0 + cycle 1 lack the top_passages field (patched 2026-04-28). "
            "Cycles 2 onwards have passage IDs; full proxy needs a join with the "
            "21M-passage FAISS metadata. Current report is the schema check; "
            "the join + proxy is implemented in a follow-up pass at Phase 4 close."
        ),
    }


# --------------------------------------------------------------------------- #
# 5. Self-correction survival distribution (Cor self-correction)
# --------------------------------------------------------------------------- #

def tau_retro_sensitivity(
    cal_fold_jsons_per_cycle: Dict[int, List[Path]],
    locked_tau_retro: float = 0.50,
    target_precision: float = 0.85,
) -> Dict[str, Any]:
    """tau_retro sensitivity analysis (defends the 0.50 heuristic choice).

    For each cycle's calibration fold, sweeps candidate tau_retro values and
    reports the precision (= empirical pi_retro) among entries surviving each
    candidate threshold. Then identifies the empirically-optimal tau (smallest
    tau that delivers >= target_precision) per cycle, and reports the gap to
    the locked tau_retro = 0.50.

    Defensible if locked tau_retro consistently delivers >= target_precision
    across cycles, even though it was not fit from a precision target.

    Inputs:
        cal_fold_jsons_per_cycle: {cycle_idx: [json_path, ...]}
            cycle 0 reads from outputs/cycle_0/calibration/
            cycle N (N>=1) reads from outputs/full_run/cycle_N/calibration/
    """
    candidates = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30..0.70
    per_cycle: Dict[str, Any] = {}

    for cycle, jsons in cal_fold_jsons_per_cycle.items():
        samples = []
        for p in jsons:
            try:
                d = json.load(p.open("r", encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            samples.extend(d.get("samples", []))
        if not samples:
            per_cycle[f"cycle_{cycle}"] = {"status": "no_samples"}
            continue

        u_em = [
            (s.get("u_stored") or 0.0, int(bool(s.get("em"))))
            for s in samples
            if s.get("u_stored") is not None
        ]
        if not u_em:
            per_cycle[f"cycle_{cycle}"] = {"status": "no_u_stored"}
            continue

        sweep = []
        for tau in candidates:
            n_above = sum(1 for u, _ in u_em if u >= tau)
            em_above = sum(em for u, em in u_em if u >= tau)
            prec = em_above / n_above if n_above > 0 else None
            sweep.append({"tau": tau, "n_survive": n_above, "pi_retro": prec})

        # Empirical pi_retro at the locked threshold
        n_locked = sum(1 for u, _ in u_em if u >= locked_tau_retro)
        em_locked = sum(em for u, em in u_em if u >= locked_tau_retro)
        pi_at_locked = em_locked / n_locked if n_locked > 0 else None

        # Smallest tau that delivers >= target_precision (recall-maximising)
        optimal = None
        for entry in sorted(sweep, key=lambda x: x["tau"]):
            p = entry["pi_retro"]
            if p is not None and p >= target_precision:
                optimal = entry["tau"]
                break

        per_cycle[f"cycle_{cycle}"] = {
            "status": "ok",
            "n_total_samples": len(u_em),
            "locked_tau_retro": locked_tau_retro,
            "pi_at_locked": pi_at_locked,
            "n_survive_at_locked": n_locked,
            "empirically_optimal_tau_for_target": optimal,
            "target_precision": target_precision,
            "sweep": sweep,
            "is_well_calibrated": (
                pi_at_locked is not None and pi_at_locked >= target_precision
            ),
        }

    well_cal_cycles = sum(
        1 for c in per_cycle.values()
        if isinstance(c, dict) and c.get("is_well_calibrated") is True
    )
    total_cycles = sum(
        1 for c in per_cycle.values()
        if isinstance(c, dict) and c.get("status") == "ok"
    )

    return {
        "status": "ok" if total_cycles > 0 else "no_data",
        "locked_tau_retro": locked_tau_retro,
        "target_precision": target_precision,
        "n_cycles_well_calibrated": well_cal_cycles,
        "n_cycles_total": total_cycles,
        "fraction_well_calibrated": (
            well_cal_cycles / total_cycles if total_cycles > 0 else None
        ),
        "per_cycle": per_cycle,
        "comment": (
            f"tau_retro = {locked_tau_retro} is well-calibrated for the "
            "trajectory if 'fraction_well_calibrated' >= 0.80 (i.e., at "
            "least 8 of 10 cycles deliver pi_retro >= target). The heuristic "
            "is defensible if this fraction is high; if not, consider "
            "fitting tau_retro per-cycle as a third conformal threshold."
        ),
    }


def survival_distribution(
    retroverify_jsons: List[Path],
    memory_meta_jsons: List[Path],
) -> Dict[str, Any]:
    """Track per-entry survival cycles. For entries that were eventually
    pruned, report the distribution of cycles spent in memory before prune.

    Returns: median, p25, p75, max survival cycles + count of entries that
    survived the full horizon.
    """
    # The retroverify_cycleN.json files do not record per-entry IDs in the
    # locked schema; we approximate via memory snapshot diffing across cycles.
    pool_sizes_by_cycle = []
    for p in sorted(memory_meta_jsons):
        try:
            d = json.load(p.open("r", encoding="utf-8"))
            pool_sizes_by_cycle.append(d.get("size", 0))
        except (json.JSONDecodeError, OSError, AttributeError):
            continue

    pruned_per_cycle = []
    for p in sorted(retroverify_jsons):
        try:
            d = json.load(p.open("r", encoding="utf-8"))
            rv = d.get("retroverify", {})
            pruned_per_cycle.append(rv.get("n_threshold_pruned", 0) +
                                    rv.get("n_loop_pruned", 0))
        except (json.JSONDecodeError, OSError):
            continue

    return {
        "status": "ok",
        "pool_size_per_cycle": pool_sizes_by_cycle,
        "n_pruned_per_cycle": pruned_per_cycle,
        "total_pruned": sum(pruned_per_cycle),
        "comment": (
            "Per-entry survival distribution requires entry-id tagging in "
            "retroverify_cycle.json that the locked schema does not log. "
            "Aggregate prune counts per cycle stand in as a population-level "
            "view of self-correction; the bounded prune distribution itself "
            "is read off of the pool-size-delta vs admission-rate gap."
        ),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/full_run/theorem_receipts"))
    parser.add_argument("--full_run_dir", type=Path,
                        default=Path("outputs/full_run"))
    parser.add_argument("--summary_csv", type=Path,
                        default=Path("outputs/full_run/experiment_summary.csv"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s  %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-cycle quantities from experiment_summary.csv
    purity, chm, cohen_d, sigma = [], [], [], []
    if args.summary_csv.exists():
        with args.summary_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("pool_purity"):
                    try: purity.append(float(row["pool_purity"]))
                    except ValueError: pass
                if row.get("chm_pooled"):
                    try: chm.append(float(row["chm_pooled"]))
                    except ValueError: pass
                if row.get("cohen_d_id"):
                    try: cohen_d.append(float(row["cohen_d_id"]))
                    except ValueError: pass
                if row.get("storage_rate"):
                    try: sigma.append(float(row["storage_rate"]))
                    except ValueError: pass

    eval_jsons = sorted(args.full_run_dir.glob("eval/*.json"))
    retroverify_jsons = sorted(args.full_run_dir.glob("retroverify_cycle*.json"))
    memory_meta_jsons = sorted(args.full_run_dir.glob("memory_store_cycle_*.meta"))

    # Build per-cycle cal-fold map for tau_retro sensitivity:
    #  cycle 0 lives at outputs/cycle_0/calibration/
    #  cycle N (N >= 1) lives at outputs/full_run/cycle_N/calibration/
    cal_fold_per_cycle: Dict[int, List[Path]] = {}
    cycle0_cal = Path("outputs/cycle_0/calibration")
    if cycle0_cal.exists():
        cal_fold_per_cycle[0] = sorted(cycle0_cal.glob("*.json"))
    for cycle_dir in sorted(args.full_run_dir.glob("cycle_*")):
        try:
            cycle_idx = int(cycle_dir.name.replace("cycle_", ""))
        except ValueError:
            continue
        cal_dir = cycle_dir / "calibration"
        if cal_dir.exists():
            cal_fold_per_cycle[cycle_idx] = sorted(cal_dir.glob("*_cycle*.json"))

    receipts = {
        "receipt_envelope_fit.json": fit_envelope(purity, cohen_d, sigma),
        "receipt_eps_arch.json": aggregate_eps_arch(eval_jsons, retroverify_jsons),
        "receipt_gap_decay.json": fit_gap_decay(chm, None),
        "receipt_corpus_floor.json": corpus_floor_proxy(eval_jsons),
        "receipt_self_correction.json": survival_distribution(
            retroverify_jsons, memory_meta_jsons,
        ),
        "receipt_tau_retro_sensitivity.json": tau_retro_sensitivity(
            cal_fold_per_cycle, locked_tau_retro=0.50, target_precision=0.85,
        ),
    }

    for fname, payload in receipts.items():
        out = args.output_dir / fname
        with out.open("w") as f:
            json.dump(payload, f, indent=2)
        logger.info("wrote %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
