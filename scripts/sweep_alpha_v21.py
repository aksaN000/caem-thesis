"""
sweep_alpha_v21.py
==================
v2.1 (2026-05-09) — focused α-store sweep over the 12-signal cal-fold
with FEVER signal-mask preserved.

Tests α_store ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40} at fixed:
  - boost_C = 0.01 (v1 sweep selected)
  - cal-fold = 1500 (500/bench × 3, 12 signals)
  - per-bench composite + per-bench conformal (KEYSTONE preserved)
  - FEVER signal-mask (alias_overlap, entity_head_consistency)

For each variant, fits composite + conformal + rescores eval-fold + runs
validate_composite_weights. Picks the smallest α that PASSES all gate
criteria; reports the full grid for thesis methodology.

Output:
  outputs/cycle_0/alpha_sweep/variant_alpha{α}.{composite,gate,validation}.json
  outputs/cycle_0/alpha_sweep/comparison.json
  outputs/cycle_0/alpha_sweep/selected.json — locked variant
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CYCLE_0 = REPO / "outputs" / "cycle_0"
CAL_JSON = CYCLE_0 / "calibration" / "calibration_fold_samples.json"
EVAL_DIR = CYCLE_0 / "eval"
SWEEP_DIR = CYCLE_0 / "alpha_sweep"

ALPHA_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
ALPHA_DEFER = 0.40
BOOST_C = 0.01


def run(cmd: List[str]) -> int:
    logger.info("CMD: %s", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(REPO))
    return r.returncode


def fit_variant(alpha: float) -> Dict[str, Any]:
    tag = f"a{alpha:.2f}"
    composite_json = SWEEP_DIR / f"composite_{tag}.json"
    gate_json = SWEEP_DIR / f"gate_{tag}.json"
    rescore_dir = SWEEP_DIR / f"rescored_{tag}"
    rescore_dir.mkdir(parents=True, exist_ok=True)

    # Fit composite (per-bench, with mask applied via BENCHMARK_SIGNAL_MASKS).
    rc = run([
        sys.executable, "-m", "scripts.fit_composite_calibration",
        "--calib_jsons", str(CAL_JSON),
        "--output_json", str(composite_json),
        "--cherian_boost", "--boost_C", str(BOOST_C),
    ])
    if rc != 0: return {"alpha": alpha, "error": f"composite fit rc={rc}"}

    # Fit conformal gate at this α.
    rc = run([
        sys.executable, "-m", "scripts.fit_conformal_gate",
        "--calib_jsons", str(CAL_JSON),
        "--composite_calibration_json", str(composite_json),
        "--output_json", str(gate_json),
        "--alpha_store", str(alpha),
        "--alpha_defer", str(ALPHA_DEFER),
    ])
    if rc != 0: return {"alpha": alpha, "error": f"gate fit rc={rc}"}

    # Rescore eval through this gate into a per-variant directory.
    # rescore_eval_with_fitted_gate.py reads from outputs/cycle_0/eval/ and
    # writes to outputs/cycle_0/eval_rescored/. We swap symlinks/dirs to
    # keep variants isolated.
    canonical_composite = CYCLE_0 / "composite_calibration.json"
    canonical_gate = CYCLE_0 / "conformal_gate.json"
    canonical_rescore = CYCLE_0 / "eval_rescored"

    # Backup current canonical state.
    backup = {}
    for src in (canonical_composite, canonical_gate):
        if src.exists():
            backup[src] = src.read_bytes()
    if canonical_rescore.exists():
        # Move existing rescored out of the way.
        tmp_swap = CYCLE_0 / f"eval_rescored.swap_{tag}"
        if tmp_swap.exists(): shutil.rmtree(tmp_swap)
        shutil.move(str(canonical_rescore), str(tmp_swap))

    try:
        # Install variant's composite + gate as canonical.
        canonical_composite.write_bytes(composite_json.read_bytes())
        canonical_gate.write_bytes(gate_json.read_bytes())

        # Rescore.
        rc = run([sys.executable, "-m", "scripts.rescore_eval_with_fitted_gate"])
        if rc != 0: return {"alpha": alpha, "error": f"rescore rc={rc}"}

        # Capture the rescored output for this variant.
        if canonical_rescore.exists():
            for f in canonical_rescore.iterdir():
                if f.is_file():
                    shutil.copy2(str(f), str(rescore_dir / f.name))

        # Validate.
        validation_json = SWEEP_DIR / f"validation_{tag}.json"
        rc = run([
            sys.executable, "scripts/validate_composite_weights.py",
            "--eval_dir", str(canonical_rescore),
            "--output", str(validation_json),
            "--cohen_d_threshold", "0.20",
            "--store_discard_gap", "0.05",
            "--max_poisoning_rate", "0.30",
            "--min_n_stored", "5",
        ])

        if validation_json.exists():
            with open(validation_json) as f:
                vd = json.load(f)
        else:
            vd = {"overall_pass": False, "checks": {}, "_note": "validation report not found"}
        return {"alpha": alpha, "validation": vd, "composite_json": str(composite_json),
                "gate_json": str(gate_json), "rescore_dir": str(rescore_dir)}
    finally:
        # Restore canonical state (caller decides which variant to install).
        for src, content in backup.items():
            src.write_bytes(content)
        if canonical_rescore.exists():
            shutil.rmtree(canonical_rescore)
        tmp_swap = CYCLE_0 / f"eval_rescored.swap_{tag}"
        if tmp_swap.exists():
            shutil.move(str(tmp_swap), str(canonical_rescore))


def main() -> int:
    if not CAL_JSON.exists():
        logger.error("cal-fold JSON missing: %s", CAL_JSON)
        return 2
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for alpha in ALPHA_GRID:
        logger.info("=" * 60)
        logger.info("VARIANT α=%.2f", alpha)
        logger.info("=" * 60)
        r = fit_variant(alpha)
        results.append(r)
        # Quick summary print.
        if "error" in r:
            print(f"  α={alpha:.2f}: ERROR — {r['error']}")
            continue
        v = r["validation"]
        ok = v.get("overall_pass", False)
        checks = v.get("checks", {})
        gap_per_bench = checks.get("store_discard_gap_per_bench", {}).get("values", {}) or {}
        pois_per_bench = checks.get("memory_poisoning_rate", {}).get("values", {}) or {}
        gap_strs = [f"{b[:4]}={g:+.3f}" for b, g in gap_per_bench.items()]
        pois_strs = [f"{b[:4]}={p.get('poisoning_rate',0):.0%}" for b, p in pois_per_bench.items()]
        print(f"  α={alpha:.2f}: pass={ok}  gap[{', '.join(gap_strs)}]  poison[{', '.join(pois_strs)}]")

    # Summary file.
    with open(SWEEP_DIR / "comparison.json", "w") as f:
        json.dump({"variants": results}, f, indent=2)

    # Pick smallest α that passes.
    passing = [r for r in results if "error" not in r and r.get("validation", {}).get("overall_pass")]
    if passing:
        selected = min(passing, key=lambda r: r["alpha"])
        print(f"\nSELECTED: α={selected['alpha']:.2f} (smallest α with overall_pass=True)")
        with open(SWEEP_DIR / "selected.json", "w") as f:
            json.dump({"alpha": selected["alpha"], "composite_json": selected["composite_json"],
                       "gate_json": selected["gate_json"]}, f, indent=2)
        # Install selected as canonical.
        shutil.copy2(selected["composite_json"], str(CYCLE_0 / "composite_calibration.json"))
        shutil.copy2(selected["gate_json"], str(CYCLE_0 / "conformal_gate.json"))
        # Re-rescore + re-validate against the selected variant so canonical
        # eval_rescored + weight_validation reflect the selected.
        canonical_rescore = CYCLE_0 / "eval_rescored"
        if canonical_rescore.exists(): shutil.rmtree(canonical_rescore)
        run([sys.executable, "-m", "scripts.rescore_eval_with_fitted_gate"])
        run([sys.executable, "scripts/validate_composite_weights.py",
             "--eval_dir", str(canonical_rescore),
             "--output", str(CYCLE_0 / "weight_validation.json"),
             "--cohen_d_threshold", "0.20",
             "--store_discard_gap", "0.05",
             "--max_poisoning_rate", "0.30",
             "--min_n_stored", "5"])
        print(f"Selected variant installed as canonical at outputs/cycle_0/.")
        return 0
    else:
        print("\nNO VARIANT PASSED. Reporting full grid for diagnosis.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
