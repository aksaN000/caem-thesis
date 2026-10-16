"""Re-run the cycle-zero composite-parameter sweep for the final (v2.1) panel and persist it.

The thesis (ch5, sec:adj-grid / sec:adj-criteria) registers a 4 x 5 x 5 = 100-variant grid:
    tau_store     in {0.55, 0.60, 0.65, 0.70}   (tau_defer fixed at 0.45)
    boost C       in {0.010, 0.025, 0.050, 0.100, 1.000}
    alpha_shrink  in {0.0, 0.4, 0.6, 0.8, 1.0}
No result file for this grid on the final panel survived, so this script recomputes it from artefacts
that are on disk. It is read-only on its inputs and writes only to outputs/cycle_0/sweep_v2_1/.

Fit:   CalProbComposite.fit_per_benchmark on the cycle-0 calibration fold (1500 rows, 500 per training
       benchmark), boost on, exactly as scripts/fit_composite_calibration.py does. The locked variant
       (C=0.01, alpha=0.6) reproduces outputs/cycle_0/composite_calibration.json exactly.
Score: every variant on (a) the calibration fold (in-sample; where the registered criteria are defined)
       and (b) the held-out cycle-0 validation fold (outputs/cycle_0/eval/<bench>_cycle0.json, 500 per
       benchmark). Decision rule = the verifier's: STORE iff u_stored >= tau_store.

Note: the u_stored values recorded in the cycle-0 validation files were produced by the legacy
weighted-sum composite (the pre-fit bootstrap), not by any cal-prob variant; this script therefore
recomputes u_stored from the recorded per-signal values for every variant.

Usage: python scripts/rerun_cycle0_sweep.py
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from caem.verification.cal_prob_composite import COMPOSITE_SIGNALS, CalProbComposite  # noqa: E402

BENCHES = ["fever", "triviaqa", "commonsense_qa"]
TAUS = [0.55, 0.60, 0.65, 0.70]
CS = [0.010, 0.025, 0.050, 0.100, 1.000]
ALPHAS = [0.0, 0.4, 0.6, 0.8, 1.0]
D_FLOOR = 0.20   # criterion 1: pooled Cohen's d floor
N_MIN = 20       # criterion 2: minimum stored count
OUT = ROOT / "outputs/cycle_0/sweep_v2_1"


def load_rows(path: Path) -> list[dict]:
    d = json.load(path.open())
    return d.get("samples", d) if isinstance(d, dict) else d


def cohens_d(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    sp = math.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / sp) if sp > 0 else float("nan")


def fold_metrics(model: CalProbComposite, rows_by_bench: dict[str, list[dict]]) -> dict:
    """u_stored per row (per-benchmark dispatch), pooled + per-bench d and AUROC."""
    u_all, y_all, out = [], [], {"per_bench": {}}
    for b in BENCHES:
        rows = rows_by_bench[b]
        u = np.array([model.predict({k: r.get(k) for k in COMPOSITE_SIGNALS}, b) for r in rows])
        y = np.array([1 if (r.get("em") or 0) == 1 else 0 for r in rows])
        out["per_bench"][b] = {"u": u, "y": y, "d": cohens_d(u, y),
                               "auroc": float(roc_auc_score(y, u)) if 0 < y.sum() < len(y) else float("nan")}
        u_all.append(u)
        y_all.append(y)
    out["u"], out["y"] = np.concatenate(u_all), np.concatenate(y_all)
    out["d"] = cohens_d(out["u"], out["y"])
    out["auroc"] = float(roc_auc_score(out["y"], out["u"]))
    return out


def store_stats(fm: dict, tau: float) -> dict:
    s = fm["u"] >= tau
    n, k = int(s.sum()), int(fm["y"][s].sum())
    per = {b: int((v["u"] >= tau).sum()) for b, v in fm["per_bench"].items()}
    return {"stored": n, "correct": k, "precision": (k / n) if n else float("nan"),
            "storage_rate": n / len(fm["u"]), "stored_per_bench": per}


def main() -> int:
    cal = load_rows(ROOT / "outputs/cycle_0/calibration/calibration_fold_samples.json")
    cal_by = {b: [r for r in cal if r.get("benchmark") == b] for b in BENCHES}
    val_by = {b: load_rows(ROOT / f"outputs/cycle_0/eval/{b}_cycle0.json") for b in BENCHES}
    assert all(len(cal_by[b]) == 500 and len(val_by[b]) == 500 for b in BENCHES)

    records = []
    for C, a in itertools.product(CS, ALPHAS):
        model = CalProbComposite().fit_per_benchmark(cal_by, fit_boost=True, boost_C=C, shrinkage_alpha=a)
        fc, fv = fold_metrics(model, cal_by), fold_metrics(model, val_by)
        for tau in TAUS:
            sc, sv = store_stats(fc, tau), store_stats(fv, tau)
            records.append({
                "variant": f"tau={tau:.2f},C={C:.3f},alpha={a:.1f}", "tau_store": tau, "boost_C": C, "alpha_shrink": a,
                "cal_d_pooled": fc["d"], "cal_auroc_pooled": fc["auroc"],
                "cal_d_per_bench": {b: v["d"] for b, v in fc["per_bench"].items()},
                "cal_stored": sc["stored"], "cal_correct": sc["correct"], "cal_precision": sc["precision"],
                "cal_storage_rate": sc["storage_rate"], "cal_stored_per_bench": sc["stored_per_bench"],
                "val_d_pooled": fv["d"], "val_auroc_pooled": fv["auroc"],
                "val_d_per_bench": {b: v["d"] for b, v in fv["per_bench"].items()},
                "val_stored": sv["stored"], "val_correct": sv["correct"], "val_precision": sv["precision"],
                "val_storage_rate": sv["storage_rate"], "val_stored_per_bench": sv["stored_per_bench"],
                "pass_c1_d": bool(fc["d"] >= D_FLOOR), "pass_c2_nmin": bool(sc["stored"] >= N_MIN),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "variants.json").write_text(json.dumps({"n_variants": len(records), "grid": {"tau_store": TAUS, "boost_C": CS,
        "alpha_shrink": ALPHAS, "tau_defer": 0.45}, "criteria": {"c1_cal_d_pooled_min": D_FLOOR, "c2_cal_stored_min": N_MIN},
        "variants": records}, indent=1))
    flat = ["variant", "tau_store", "boost_C", "alpha_shrink", "cal_d_pooled", "cal_auroc_pooled", "cal_stored", "cal_correct",
            "cal_precision", "val_d_pooled", "val_auroc_pooled", "val_stored", "val_correct", "val_precision",
            "pass_c1_d", "pass_c2_nmin"]
    with (OUT / "variants.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flat)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in flat})
    print(f"wrote {len(records)} variants to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
