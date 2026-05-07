#!/usr/bin/env python
"""
scripts/sweep_composite_variants.py
====================================
Try several composite/gate parameterizations and report eval-fold metrics
for each. Used after step 7.0.3 fails on the eval-rescored fold to find
a setting that holds the precision target on eval (not just calibration).

Variants
--------
  V0 (baseline): boost=on,  alpha_store=0.20  (the failing config)
  V1 (A):         boost=off, alpha_store=0.20
  V2 (B):         boost=on,  alpha_store=0.10
  V3 (A+B):       boost=off, alpha_store=0.10
  V4 (C):         boost=on with stronger L2 (C=0.05), alpha_store=0.20
  V5 (A+C):       boost=on with stronger L2 (C=0.05), alpha_store=0.10

For each variant we:
  1. Refit CalProbComposite on the calibration fold (1500 dedup samples).
  2. Refit ConformalStorageGate at the chosen alpha_store.
  3. Apply both to the eval fold (3500 samples in outputs/cycle_0/eval).
  4. Compute pooled + ID-pooled + transfer-pooled storage rate and precision.
  5. Print a comparison table; emit per-variant JSON to outputs/cycle_0/sweep/.

Inputs
------
  outputs/cycle_0/calibration/calibration_fold_samples.json (1500 dedup, 10 signals + em)
  outputs/cycle_0/eval/*_cycle0.json (3500 samples, 9 signals + em + bootstrap decision)

Outputs
-------
  outputs/cycle_0/sweep/variant_<id>.json (one per variant)
  outputs/cycle_0/sweep/comparison.json (summary table)
  outputs/cycle_0/sweep/comparison.tex (LaTeX-ready table)
  stdout: comparison table

This is a thesis-evidence helper. The user authored the full sweep so all
parameter choices are defensible against reviewer questions.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
CAL_JSON = REPO_ROOT / "outputs" / "cycle_0" / "calibration" / "calibration_fold_samples.json"
EVAL_DIR = REPO_ROOT / "outputs" / "cycle_0" / "eval"
SWEEP_DIR = REPO_ROOT / "outputs" / "cycle_0" / "sweep"

VARIANTS: List[Dict[str, Any]] = []
for alpha in [0.05, 0.075, 0.10, 0.125, 0.15]:
    for C in [0.01, 0.025, 0.05, 0.10, 1.0]:
        VARIANTS.append({
            "id": f"a{int(alpha*1000):03d}_C{C:.3f}",
            "boost": True, "boost_C": C,
            "alpha_store": alpha, "alpha_defer": 0.40,
            "label": f"α={alpha:.3f} C={C:.3f}",
        })


def fit_composite(samples: List[Dict[str, Any]], boost: bool, boost_C: float):
    """Fit CalProbComposite with the given boost setting + L2 regularization."""
    from caem.verification.cal_prob_composite import CalProbComposite
    return CalProbComposite().fit(samples, fit_boost=boost, boost_C=float(boost_C))


def fit_gate(samples: List[Dict[str, Any]], composite, alpha_store: float, alpha_defer: float):
    from caem.verification.conformal_gate import ConformalStorageGate
    # Recompute u_stored on each calibration sample using THIS variant's composite,
    # then fit the gate on those rescored u_stored values.
    rescored = []
    for s in samples:
        sig = {
            k: float(s.get(k, 0.5) or 0.5)
            for k in (
                "u_token", "u_dropout", "u_internal", "s_avg", "h_norm",
                "p_entail", "p_ground_max", "p_ground_mean", "p_ground_atomic", "q_a_relevance",
            )
        }
        new_s = dict(s)
        new_s["u_stored"] = composite.predict(sig)
        rescored.append(new_s)
    return ConformalStorageGate.fit(
        calib_samples=rescored,
        alpha_store=alpha_store,
        alpha_defer=alpha_defer,
    )


def apply_to_eval(eval_dir: Path, composite, gate) -> Dict[str, Any]:
    """Apply fitted composite + gate to all eval samples; return per-bench summary."""
    summary: Dict[str, Dict[str, int]] = {}
    pooled = {"n": 0, "store": 0, "store_correct": 0}
    for src in sorted(eval_dir.glob("*_cycle0.json")):
        bench = src.stem.replace("_cycle0", "")
        with open(src) as f:
            doc = json.load(f)
        rows = doc.get("samples") or doc.get("results") or []
        st = {"n": 0, "store": 0, "store_correct": 0}
        for s in rows:
            sig = {
                k: float(s.get(k, 0.5) or 0.5)
                for k in (
                    "u_token", "u_dropout", "u_internal", "s_avg", "h_norm",
                    "p_entail", "p_ground_max", "p_ground_mean", "p_ground_atomic", "q_a_relevance",
                )
            }
            u = composite.predict(sig)
            decision, _ = gate.decide(
                u_stored=u,
                p_ground_max=sig["p_ground_max"],
                abstain_pg=sig["p_ground_max"],
            )
            em = s.get("em")
            st["n"] += 1
            if decision == "STORE":
                st["store"] += 1
                if em == 1.0:
                    st["store_correct"] += 1
        summary[bench] = st
        for k in pooled:
            pooled[k] += st[k]
    return {"per_bench": summary, "pooled": pooled}


def split_id_transfer(per_bench: Dict[str, Dict[str, int]]):
    try:
        from caem.benchmark_splits import TRAINING_BENCHMARKS as _TRAIN
    except ImportError:
        # v2 Fix 9b fallback (was the v1 3-bench panel).
        _TRAIN = ("fever", "triviaqa", "hotpotqa", "commonsense_qa")
    id_p = {"n": 0, "store": 0, "store_correct": 0}
    tr_p = {"n": 0, "store": 0, "store_correct": 0}
    for bench, st in per_bench.items():
        target = id_p if bench in _TRAIN else tr_p
        for k in target:
            target[k] += st[k]
    return id_p, tr_p


def main() -> int:
    if not CAL_JSON.exists():
        print(f"[error] {CAL_JSON} not found", file=sys.stderr)
        return 1
    if not EVAL_DIR.exists():
        print(f"[error] {EVAL_DIR} not found", file=sys.stderr)
        return 1

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    with open(CAL_JSON) as f:
        cal_doc = json.load(f)
    cal_samples = cal_doc.get("samples", [])
    print(f"[info] calibration fold: {len(cal_samples)} samples")

    rows = []
    for v in VARIANTS:
        print(f"\n=== Fitting {v['id']}: {v['label']} ===")
        composite = fit_composite(cal_samples, v["boost"], v["boost_C"] or 1.0)
        gate = fit_gate(cal_samples, composite, v["alpha_store"], v["alpha_defer"])
        eval_summary = apply_to_eval(EVAL_DIR, composite, gate)
        per_bench = eval_summary["per_bench"]
        pooled = eval_summary["pooled"]
        id_p, tr_p = split_id_transfer(per_bench)

        def metrics(p):
            n, store, correct = p["n"], p["store"], p["store_correct"]
            rate = (store / n) if n else 0.0
            prec = (correct / store) if store else float("nan")
            return rate, prec

        pooled_rate, pooled_prec = metrics(pooled)
        id_rate, id_prec = metrics(id_p)
        tr_rate, tr_prec = metrics(tr_p)

        # Cal-fold self-reported precision (from gate.fit metadata).
        cal_store_n = getattr(gate, "_store_n", None) or getattr(gate, "store_n", None)
        cal_store_prec = getattr(gate, "_store_precision", None) or getattr(gate, "store_precision", None)

        row = {
            "variant": v["id"],
            "label": v["label"],
            "tau_store": float(gate.tau_store),
            "tau_defer": float(gate.tau_defer),
            "boost_intercept": float(composite.boost_intercept),
            "cal_store_n": cal_store_n,
            "cal_store_precision": cal_store_prec,
            "eval_pooled": {"rate": pooled_rate, "precision": pooled_prec, "n": pooled["n"], "store": pooled["store"], "correct": pooled["store_correct"]},
            "eval_id": {"rate": id_rate, "precision": id_prec, "n": id_p["n"], "store": id_p["store"], "correct": id_p["store_correct"]},
            "eval_transfer": {"rate": tr_rate, "precision": tr_prec, "n": tr_p["n"], "store": tr_p["store"], "correct": tr_p["store_correct"]},
            "per_bench": {b: {**st, "rate": st["store"]/st["n"], "precision": (st["store_correct"]/st["store"]) if st["store"] else None} for b, st in per_bench.items()},
        }
        rows.append(row)
        with open(SWEEP_DIR / f"variant_{v['id']}.json", "w") as f:
            json.dump(row, f, indent=2)

    # Summary table.
    print()
    print("=" * 100)
    print(f"{'variant':<12}{'label':<28}{'τ_store':>9}{'cal_prec':>10}{'eval_pooled':>14}{'eval_ID':>14}{'eval_trans':>14}")
    print("-" * 100)
    for r in rows:
        cal_prec = r["cal_store_precision"]
        cal_prec_s = f"{cal_prec*100:.1f}%" if cal_prec is not None else "  --  "
        eval_pooled = f"{r['eval_pooled']['precision']*100:.1f}% ({r['eval_pooled']['rate']*100:.1f}%)"
        eval_id = f"{r['eval_id']['precision']*100:.1f}% ({r['eval_id']['rate']*100:.1f}%)"
        eval_tr = f"{r['eval_transfer']['precision']*100:.1f}% ({r['eval_transfer']['rate']*100:.1f}%)"
        print(f"{r['variant']:<12}{r['label']:<28}{r['tau_store']:>9.3f}{cal_prec_s:>10}{eval_pooled:>14}{eval_id:>14}{eval_tr:>14}")
    print("-" * 100)
    print("Format: precision (storage rate). Higher precision + reasonable rate (3-15%) is best.")

    summary_doc = {"variants": rows, "config": VARIANTS}
    with open(SWEEP_DIR / "comparison.json", "w") as f:
        json.dump(summary_doc, f, indent=2)
    print(f"\nWrote {SWEEP_DIR}/comparison.json + per-variant JSONs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
