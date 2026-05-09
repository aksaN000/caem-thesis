#!/usr/bin/env python
"""
scripts/rescore_eval_with_fitted_gate.py
=========================================
Re-score Cycle-0 eval JSONs through the FITTED CalProbComposite + the
fixed-threshold storage gate to produce a defensible post-fit validation view.

Why this exists
---------------
``scripts/validate_composite_weights.py`` reads ``decision`` and
``u_stored`` directly from cycle_0/eval/*.json. Those values were written
during the eval pass BEFORE step 7.0.1 fitted the new composite, so the
validate script ends up checking the BOOTSTRAP composite's STORE decisions
rather than the fitted ones. After step 7.0.1 lands ``composite_calibration.json``,
this script re-applies it to every eval sample (with the fixed-threshold
gate from CAEMConfig.store_threshold / defer_threshold) and writes a
parallel directory of rescored eval JSONs.

Phase 1c (2026-05-09): the conformal storage gate has been replaced by a
fixed threshold on the calibrated composite probability. Gate parameters
come from CAEMConfig (store_threshold, defer_threshold, abstain_pground_ceiling)
instead of a per-cycle conformal_gate.json.

Inputs
------
- outputs/cycle_0/eval/*.json (original eval, with all 10 signals + bootstrap u_stored)
- outputs/cycle_0/composite_calibration.json (fitted CalProbComposite)
- CAEMConfig defaults for tau_store / tau_defer / abstain_pground_ceiling

Outputs
-------
- outputs/cycle_0/eval_rescored/*.json — per-sample u_stored + decision
  recomputed through the fitted composite + fixed gate. Other fields
  preserved verbatim.
- Stdout summary: per-benchmark STORE counts and precision under fitted gate.

Usage
-----
    python -m scripts.rescore_eval_with_fitted_gate

This is a thesis-evidence helper: the rescored eval JSONs feed back into
``validate_composite_weights.py --eval_dir outputs/cycle_0/eval_rescored``
to produce a post-fit weight_validation.json that legitimately validates
the deployed gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SRC = REPO_ROOT / "outputs" / "cycle_0" / "eval"
EVAL_DST = REPO_ROOT / "outputs" / "cycle_0" / "eval_rescored"
COMPOSITE_JSON = REPO_ROOT / "outputs" / "cycle_0" / "composite_calibration.json"


def main() -> int:
    if not COMPOSITE_JSON.exists():
        print(f"[error] {COMPOSITE_JSON} not found", file=sys.stderr)
        return 1
    if not EVAL_SRC.exists():
        print(f"[error] {EVAL_SRC} not found", file=sys.stderr)
        return 1

    from caem.config import CAEMConfig
    from caem.verification.cal_prob_composite import CalProbComposite

    composite = CalProbComposite.load(COMPOSITE_JSON)
    cfg = CAEMConfig()
    tau_store = cfg.store_threshold
    tau_defer = cfg.defer_threshold
    abstain_pg = cfg.abstain_pground_ceiling
    print(
        f"[info] loaded composite ({len(composite.calibrations)} signals, "
        f"boost={composite.boost_weights is not None}); "
        f"fixed gate τ_store={tau_store:.4f}, τ_defer={tau_defer:.4f}, "
        f"abstain_pg={abstain_pg:.3f}"
    )

    EVAL_DST.mkdir(parents=True, exist_ok=True)

    # Per-benchmark accumulator for the summary.
    summary: Dict[str, Dict[str, int]] = {}

    for src_path in sorted(EVAL_SRC.glob("*_cycle0.json")):
        with open(src_path) as f:
            doc = json.load(f)
        bench = src_path.stem.replace("_cycle0", "")
        rows = doc.get("samples") or doc.get("results") or []
        if not rows:
            print(f"[warn] {src_path.name}: no samples")
            continue

        bm_stats = {"n": 0, "store": 0, "store_correct": 0, "defer": 0, "abstain": 0, "discard": 0}
        rescored: List[Dict[str, Any]] = []

        for s in rows:
            # v2.1 (2026-05-09): all 12 signals + per-bench dispatch.
            # Pre-fix this dict was 10 signals, omitting alias_overlap
            # and entity_head_consistency (Fix 6 + Fix 7); same v2-propagation
            # miss the cal-fold writer had. Without source_benchmark, predict()
            # also fell back to the pooled global composite, bypassing the
            # per-benchmark KEYSTONE dispatch.
            # Phase 1c (2026-05-09): 10 signals fed to the composite.
            # alias_overlap and entity_head_consistency dropped after the
            # AUROC diagnostic showed they were essentially random (P1).
            sig = {
                "u_token":         float(s.get("u_token", 0.5) or 0.5),
                "u_dropout":       float(s.get("u_dropout", 0.5) or 0.5),
                "u_internal":      float(s.get("u_internal", 0.5) or 0.5),
                "s_avg":           float(s.get("s_avg", 0.5) or 0.5),
                "h_norm":          float(s.get("h_norm", 0.5) or 0.5),
                "p_entail":        float(s.get("p_entail", 0.5) or 0.5),
                "p_ground_max":    float(s.get("p_ground_max", 0.5) or 0.5),
                "p_ground_mean":   float(s.get("p_ground_mean", 0.5) or 0.5),
                "p_ground_atomic": float(s.get("p_ground_atomic", 0.5) or 0.5),
                "q_a_relevance":   float(s.get("q_a_relevance", 0.5) or 0.5),
            }
            new_u_stored = composite.predict(sig, source_benchmark=bench)
            # Fixed-threshold gate: same four-outcome decision tree as in
            # UnifiedVerifier._decide(). Inlined here so this rescore script
            # doesn't depend on a verifier instance (no GPU needed).
            pgm = sig["p_ground_max"]
            if new_u_stored >= tau_store:
                decision = "STORE"
            elif new_u_stored >= tau_defer:
                decision = "DEFERRED"
            elif pgm < abstain_pg:
                decision = "ABSTAIN"
            else:
                decision = "DISCARD"

            # Preserve original sample, override u_stored + decision.
            new_s = dict(s)
            new_s["u_stored_bootstrap"] = s.get("u_stored")
            new_s["decision_bootstrap"] = s.get("decision")
            new_s["u_stored"] = new_u_stored
            new_s["decision"] = decision
            rescored.append(new_s)

            em = s.get("em")
            bm_stats["n"] += 1
            if decision == "STORE":
                bm_stats["store"] += 1
                if em == 1.0:
                    bm_stats["store_correct"] += 1
            elif decision == "DEFERRED":
                bm_stats["defer"] += 1
            elif decision == "ABSTAIN":
                bm_stats["abstain"] += 1
            elif decision == "DISCARD":
                bm_stats["discard"] += 1

        out_doc = dict(doc)
        out_doc["samples"] = rescored
        out_doc["_rescored"] = {
            "composite_calibration_json": str(COMPOSITE_JSON.relative_to(REPO_ROOT)),
            "gate_type": "fixed_threshold_on_calibrated_probability",
            "tau_store": tau_store,
            "tau_defer": tau_defer,
            "abstain_pground_ceiling": abstain_pg,
        }
        dst_path = EVAL_DST / src_path.name
        with open(dst_path, "w") as f:
            json.dump(out_doc, f)
        summary[bench] = bm_stats

    # Print summary.
    print()
    print("=" * 78)
    print(f"Rescored eval through fitted CalProbComposite + fixed-threshold gate")
    print(f"  τ_store={tau_store:.4f}  τ_defer={tau_defer:.4f}")
    print("=" * 78)
    print(f"{'benchmark':<22}{'n':>6}{'STORE':>8}{'correct':>10}{'precision':>12}")
    print("-" * 78)
    pooled_n = pooled_store = pooled_correct = 0
    for bench, st in sorted(summary.items()):
        prec = (st["store_correct"] / st["store"]) if st["store"] else float("nan")
        prec_str = f"{prec*100:6.1f}%" if st["store"] else "  --  "
        print(f"{bench:<22}{st['n']:>6}{st['store']:>8}{st['store_correct']:>10}{prec_str:>12}")
        pooled_n += st["n"]
        pooled_store += st["store"]
        pooled_correct += st["store_correct"]
    print("-" * 78)
    pooled_prec = (pooled_correct / pooled_store) if pooled_store else float("nan")
    pooled_str = f"{pooled_prec*100:6.1f}%" if pooled_store else "  --  "
    print(f"{'POOLED':<22}{pooled_n:>6}{pooled_store:>8}{pooled_correct:>10}{pooled_str:>12}")
    rate = (pooled_store / pooled_n) if pooled_n else 0.0
    print(f"\nStorage rate (POOLED): {pooled_store}/{pooled_n} = {rate*100:.2f}%")

    # Also break down ID vs Transfer.
    try:
        from caem.benchmark_splits import TRAINING_BENCHMARKS as _TRAIN
    except ImportError:
        # v2.1 fallback (HotpotQA + NQ dropped 2026-05-08).
        _TRAIN = ("fever", "triviaqa", "commonsense_qa")
    id_n = id_store = id_correct = 0
    tr_n = tr_store = tr_correct = 0
    for bench, st in summary.items():
        if bench in _TRAIN:
            id_n += st["n"]; id_store += st["store"]; id_correct += st["store_correct"]
        else:
            tr_n += st["n"]; tr_store += st["store"]; tr_correct += st["store_correct"]
    if id_store:
        print(f"ID precision      : {id_correct}/{id_store} = {id_correct/id_store*100:.2f}%  (rate {id_store/id_n*100:.2f}%)")
    if tr_store:
        print(f"Transfer precision: {tr_correct}/{tr_store} = {tr_correct/tr_store*100:.2f}%  (rate {tr_store/tr_n*100:.2f}%)")

    print(f"\nRescored JSONs written to {EVAL_DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
