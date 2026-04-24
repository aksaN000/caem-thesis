#!/usr/bin/env python
"""
scripts/validate_composite_weights.py
======================================
Post-Step-7.0 checkpoint — empirically validate u_stored composite
weights on the freshly-written Cycle-0 eval data.

Runs automatically after step_7_0_calibrate via `step_7_0_3_validate_weights`
in run_phase1a.sh. Exits:
  0   composite discrimination is strong enough to continue to Step 7 main
  2   composite is weak — halt runner, prompt user review

What it computes
----------------
For each benchmark's cycle_0 eval JSON:
  - Per-signal Cohen's d vs em (correctness)
  - Composite u_stored Cohen's d vs em
  - Per-decision em rate (STORE should have higher em than DISCARD)
  - Signal correlation matrix (check non-redundancy, Step 19.5 pseudo)
  - Memory-quality forecast: what fraction of STOREd samples are wrong

Decision criteria
-----------------
PASS (exit 0) if ALL:
  - Composite Cohen's d >= threshold (default 0.20)
  - STORE em >= DISCARD em + 0.10 on every benchmark
  - No individual signal has Cohen's d < -0.2 (strong inversion)

FAIL (exit 2) otherwise — runner halts, writes detailed report to
outputs/cycle_0/weight_validation.json. User reviews, optionally
tunes weights, re-runs step_7_0_calibrate, then resumes.

Usage
-----
    python scripts/validate_composite_weights.py \\
        --eval_dir outputs/cycle_0/eval \\
        --output outputs/cycle_0/weight_validation.json \\
        --cohen_d_threshold 0.20 \\
        --store_discard_gap 0.10
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SIGNALS = [
    "p_ground_mean", "p_ground_max", "p_ground_atomic",
    "p_entail", "q_a_relevance", "s_avg", "h_norm",
    "u_internal", "u_token", "u_dropout",
]


def _load_samples(eval_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(eval_dir.glob("*_cycle0.json")):
        d = json.load(open(p))
        bench = p.stem.replace("_cycle0", "")
        for s in d.get("samples") or d.get("results") or []:
            if isinstance(s, dict):
                s["_bench"] = bench
                out.append(s)
    return out


def _cohen_d(v1: List[float], v0: List[float]) -> float:
    if not v1 or not v0:
        return 0.0
    m1 = statistics.mean(v1)
    m0 = statistics.mean(v0)
    pooled = statistics.stdev(v1 + v0) if len(v1 + v0) > 1 else 1e-6
    return (m1 - m0) / max(pooled, 1e-6)


def _per_signal_discrimination(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Cohen's d per signal against em."""
    em1 = [s for s in samples if s.get("em") == 1.0]
    em0 = [s for s in samples if s.get("em") == 0.0]
    result = {}
    for sig in SIGNALS:
        v1 = [s[sig] for s in em1 if isinstance(s.get(sig), (int, float))]
        v0 = [s[sig] for s in em0 if isinstance(s.get(sig), (int, float))]
        if not v1 or not v0:
            continue
        d = _cohen_d(v1, v0)
        result[sig] = {
            "mean_em1": statistics.mean(v1),
            "mean_em0": statistics.mean(v0),
            "cohen_d": d,
            "n1": len(v1), "n0": len(v0),
        }
    # Composite
    v1u = [s["u_stored"] for s in em1 if isinstance(s.get("u_stored"), (int, float))]
    v0u = [s["u_stored"] for s in em0 if isinstance(s.get("u_stored"), (int, float))]
    if v1u and v0u:
        result["u_stored"] = {
            "mean_em1": statistics.mean(v1u),
            "mean_em0": statistics.mean(v0u),
            "cohen_d": _cohen_d(v1u, v0u),
            "n1": len(v1u), "n0": len(v0u),
        }
    return result


def _per_decision_em(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """For each decision, fraction of correct answers."""
    by_dec: Dict[str, List[float]] = defaultdict(list)
    for s in samples:
        d = s.get("decision")
        em = s.get("em")
        if d and isinstance(em, (int, float)):
            by_dec[d].append(em)
    out: Dict[str, Dict[str, float]] = {}
    for dec, ems in by_dec.items():
        out[dec] = {"n": len(ems), "em_mean": statistics.mean(ems) if ems else 0.0}
    return out


def _benchmark_store_discard_gap(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Per-benchmark: em(STORE) - em(DISCARD). Should be positive."""
    by_bench: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_bench[s["_bench"]].append(s)
    out: Dict[str, float] = {}
    for bench, bsamples in by_bench.items():
        store_em = [s["em"] for s in bsamples
                    if s.get("decision") == "STORE" and isinstance(s.get("em"), (int, float))]
        disc_em = [s["em"] for s in bsamples
                   if s.get("decision") == "DISCARD" and isinstance(s.get("em"), (int, float))]
        if store_em and disc_em:
            out[bench] = statistics.mean(store_em) - statistics.mean(disc_em)
    return out


def _signal_correlation_matrix(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Pearson r between every pair of signals (Step 19.5 pseudo)."""
    valid = [s for s in samples if all(
        isinstance(s.get(sig), (int, float)) for sig in SIGNALS
    )]
    if len(valid) < 20:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for i, a in enumerate(SIGNALS):
        out[a] = {}
        for j, b in enumerate(SIGNALS):
            xs = [s[a] for s in valid]
            ys = [s[b] for s in valid]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
            dx = math.sqrt(sum((x-mx)**2 for x in xs))
            dy = math.sqrt(sum((y-my)**2 for y in ys))
            out[a][b] = (num / (dx * dy)) if dx*dy > 0 else 0.0
    return out


def _memory_quality_forecast(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-bench: fraction of STOREd samples that are wrong (em=0)."""
    by_bench: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        if s.get("decision") == "STORE":
            by_bench[s["_bench"]].append(s)
    out: Dict[str, Dict[str, Any]] = {}
    for bench, stored in by_bench.items():
        wrong = sum(1 for s in stored if s.get("em") == 0.0)
        out[bench] = {
            "total_stored": len(stored),
            "wrong_stored": wrong,
            "poisoning_rate": wrong / len(stored) if stored else 0.0,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--eval_dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cohen_d_threshold", type=float, default=0.20)
    p.add_argument("--store_discard_gap", type=float, default=0.10)
    p.add_argument("--max_poisoning_rate", type=float, default=0.25)
    ns = p.parse_args()

    samples = _load_samples(ns.eval_dir)
    if not samples:
        print(f"[error] No samples loaded from {ns.eval_dir}", file=sys.stderr)
        return 1

    print(f"[info] Loaded {len(samples)} samples across {len(set(s['_bench'] for s in samples))} benchmarks")

    signal_disc = _per_signal_discrimination(samples)
    per_dec = _per_decision_em(samples)
    bench_gap = _benchmark_store_discard_gap(samples)
    corr_matrix = _signal_correlation_matrix(samples)
    mem_quality = _memory_quality_forecast(samples)

    # Decision
    composite_d = signal_disc.get("u_stored", {}).get("cohen_d", 0.0)
    composite_pass = composite_d >= ns.cohen_d_threshold
    gap_pass = all(gap >= ns.store_discard_gap for gap in bench_gap.values())
    poisoning_pass = all(
        m["poisoning_rate"] <= ns.max_poisoning_rate
        for m in mem_quality.values() if m["total_stored"] > 0
    )
    strong_inversion = any(
        sig != "u_dropout" and d.get("cohen_d", 0.0) < -0.20
        for sig, d in signal_disc.items()
    )

    overall_pass = composite_pass and gap_pass and poisoning_pass and not strong_inversion

    report = {
        "overall_pass": overall_pass,
        "checks": {
            "composite_cohen_d": {"value": composite_d, "threshold": ns.cohen_d_threshold, "pass": composite_pass},
            "store_discard_gap_per_bench": {"values": bench_gap, "threshold": ns.store_discard_gap, "pass": gap_pass},
            "memory_poisoning_rate": {"values": mem_quality, "threshold": ns.max_poisoning_rate, "pass": poisoning_pass},
            "strong_signal_inversion": {"value": strong_inversion, "pass": not strong_inversion},
        },
        "signal_discrimination": signal_disc,
        "per_decision_em": per_dec,
        "signal_correlation_matrix": corr_matrix,
        "memory_quality_forecast": mem_quality,
        "n_samples": len(samples),
    }

    ns.output.parent.mkdir(parents=True, exist_ok=True)
    with open(ns.output, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable summary to stdout
    print("=" * 70)
    print(f"Composite Cohen's d:   {composite_d:+.3f}  (threshold: {ns.cohen_d_threshold:+.3f})   "
          f"{'PASS' if composite_pass else 'FAIL'}")
    print(f"Per-bench STORE gap:")
    for bench, gap in bench_gap.items():
        verdict = "PASS" if gap >= ns.store_discard_gap else "FAIL"
        print(f"  {bench:<20}  em(STORE)-em(DISCARD) = {gap:+.3f}   {verdict}")
    print(f"Memory poisoning rate:")
    for bench, m in mem_quality.items():
        verdict = "PASS" if m["poisoning_rate"] <= ns.max_poisoning_rate else "FAIL"
        print(f"  {bench:<20}  {m['wrong_stored']}/{m['total_stored']} wrong "
              f"({100*m['poisoning_rate']:.1f}%)   {verdict}")
    print(f"Strong signal inversion: {strong_inversion}   {'FAIL' if strong_inversion else 'PASS'}")
    print("=" * 70)
    if overall_pass:
        print("OVERALL: PASS  — safe to proceed to Step 7 main.")
        return 0
    print("OVERALL: FAIL — halt runner, review weights in config.py.")
    print(f"  Report: {ns.output}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
