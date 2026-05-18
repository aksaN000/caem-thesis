#!/usr/bin/env python3
"""scripts/ruc_verifier_equivalence_check.py
=================================================
Signal-contamination gate for the RUC v2 rescore-throughput optimisation.

Background
----------
Before launching the v2 rescore (54 024 ops across 6 baselines, ~6 GPU-days at
the current 9.6 sec/sample throughput), we want to enable two performance
toggles in the rescore harness:

  1. ``--batch_size 32`` (was 16). Amortises the rerank, MiniCheck, Qwen-NLI,
     and atomic-decomp batched kernels over more rows. Per-row outputs are
     independent so the only possible drift is bf16 reduction-order noise.

  2. ``CAEM_BATCH_U_TOK_DROP=1``. Replaces the per-sample ``_compute_u_token``
     and ``_compute_u_dropout`` calls with the pooled
     ``_pool_u_token_batch`` + ``_pool_u_dropout_batch`` paths that already
     exist in ``caem/verification/verifier.py`` (gated on this env var).

Both changes are reversible (CLI flag and env var only) and neither touches
``caem/verification/*.py``. The composite weights, atomic decomp, and the
chain pool are unchanged.

The active cycle-3 composite consumes 9 signals (per
``caem/verification/cal_prob_composite.py:77-87``):

    u_token, u_dropout, u_internal, s_avg, p_entail,
    p_ground_max, p_ground_mean, p_ground_atomic, q_a_relevance

The retired-but-still-recorded fields (``h_norm``, ``p_contra``,
``alias_overlap``, ``entity_head_consistency``) are excluded from the
contamination gate because the composite does not read them.

Acceptance gate (signal contamination)
---------------------------------------
For each of the 9 active signals AND for ``u_stored``:

    Pearson r(reference, optimized) >= 0.999
    max |reference - optimized|     <  0.005

For ``decision`` (STORE / DEFERRED / RETRY / DISCARD):

    flip rate = 0 / 200 (exactly zero, not "small")

Failure on ANY signal -> the corresponding optimisation must be reverted
before any v2 rescore can launch.

Inputs
------
200 samples drawn from the cycle-3 eval JSONs (``outputs/full_run/eval/
<bench>_cycle3.json``) so the reference distribution matches the data the
locked cycle-3 composite was fitted on. Default split: 40 samples each from
fever, triviaqa, commonsense_qa, truthfulqa, strategyqa.

Outputs
-------
``caem/ruc/v2_artefacts/equivalence_<run_tag>.json``  per-signal table +
                                                     pass/fail verdict
``caem/ruc/v2_artefacts/equivalence_<run_tag>.log``   stdout copy

Usage
-----
After the in-flight rescore (PID 769668) frees the GPU::

    # Reference run (current production: bs=16, env var unset)
    python -m scripts.ruc_verifier_equivalence_check \\
        --mode reference \\
        --out caem/ruc/v2_artefacts/eq_ref.json

    # Optimised run (bs=32, env var set)
    CAEM_BATCH_U_TOK_DROP=1 python -m scripts.ruc_verifier_equivalence_check \\
        --mode optimised \\
        --out caem/ruc/v2_artefacts/eq_opt.json \\
        --batch_size 32

    # Compare the two
    python -m scripts.ruc_verifier_equivalence_check \\
        --mode compare \\
        --reference caem/ruc/v2_artefacts/eq_ref.json \\
        --optimised caem/ruc/v2_artefacts/eq_opt.json \\
        --out caem/ruc/v2_artefacts/eq_verdict.json

A non-zero exit code on compare means the gate failed; do NOT launch v2
rescore. The verdict JSON has per-signal r and max|delta| plus the
decision-flip count.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Active composite signals (mirror caem/verification/cal_prob_composite.py)
# ---------------------------------------------------------------------------

ACTIVE_SIGNALS: Tuple[str, ...] = (
    "u_token", "u_dropout", "u_internal", "s_avg", "p_entail",
    "p_ground_max", "p_ground_mean", "p_ground_atomic", "q_a_relevance",
)

# u_stored is the composite output; we also gate on it explicitly because it
# is the downstream value every storage / deferred / retroverify decision
# reads.
GATED_FIELDS: Tuple[str, ...] = ACTIVE_SIGNALS + ("u_stored",)

# Per-signal acceptance thresholds.
PEARSON_MIN: float = 0.999
MAX_ABS_DELTA: float = 0.005
DECISION_FLIPS_MAX: int = 0


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _load_calfold_samples(
    eval_dir: Path,
    per_bench: int,
    benches: Tuple[str, ...],
    seed: int = 1337,
) -> List[Dict[str, Any]]:
    """Draw ``per_bench`` samples from each benchmark's cycle-3 eval JSON.

    Each row carries ``id``, ``question``, ``prediction``, and the
    pre-computed verifier signals from the trajectory run. We will re-run
    the verifier on (id, question, prediction) and compare the freshly
    computed signals against the trajectory's.
    """
    import random
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for bench in benches:
        path = eval_dir / f"{bench}_cycle3.json"
        if not path.exists():
            logger.warning("missing cal-fold eval %s; bench skipped", path)
            continue
        doc = json.loads(path.read_text())
        samples = doc.get("samples") or doc.get("results") or []
        useful = [
            s for s in samples
            if s.get("question") and s.get("prediction")
            and s.get("u_stored") is not None
            and s.get("decision") is not None
        ]
        rng.shuffle(useful)
        chosen = useful[:per_bench]
        for s in chosen:
            r = dict(s)
            r["benchmark"] = bench
            r["_source"] = path.name
            rows.append(r)
        logger.info("bench=%s drew %d / %d eligible", bench, len(chosen), len(useful))
    return rows


# ---------------------------------------------------------------------------
# Verifier construction (mirrors scripts/rescore_baselines_through_verifier.py)
# ---------------------------------------------------------------------------

def _build_verifier(
    composite_calibration: Path,
    checkpoint: Optional[Path],
    passage_index: Path,
    device: str,
) -> Tuple[Any, Any]:
    """Identical construction to the rescore harness so the equivalence test
    instruments the rescore pipeline, not a different one.
    """
    from scripts.rescore_baselines_through_verifier import _build_verifier as _build
    return _build(
        composite_calibration=composite_calibration,
        checkpoint=checkpoint,
        passage_index=passage_index,
        device=device,
    )


# ---------------------------------------------------------------------------
# One mode: run the verifier on the sample list, capture all 9 signals +
# u_stored + decision per row.
# ---------------------------------------------------------------------------

def _run_verifier(
    verifier: Any,
    rows: List[Dict[str, Any]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Run the cycle-3-locked verifier on (question, prediction) pairs and
    return a new list with the freshly computed signals + decision."""
    import time as _t
    triples: List[Tuple[int, str, str]] = []
    for i, r in enumerate(rows):
        q = r.get("question") or ""
        p = r.get("prediction") or ""
        if q and p:
            triples.append((i, q, p))
    logger.info("running verifier on %d rows at bs=%d (env_CAEM_BATCH_U_TOK_DROP=%s)",
                len(triples), batch_size,
                os.environ.get("CAEM_BATCH_U_TOK_DROP", "0"))

    results: Dict[int, Any] = {}
    t0 = _t.perf_counter()
    for start in range(0, len(triples), batch_size):
        chunk = triples[start:start + batch_size]
        pairs = [(q, p) for (_, q, p) in chunk]
        tb = _t.perf_counter()
        vouts = verifier.verify_batch(pairs)
        dt = _t.perf_counter() - tb
        logger.info("  chunk [%d-%d / %d] in %.1fs (%.0f ms/sample)",
                    start, start + len(chunk), len(triples), dt,
                    dt / max(len(chunk), 1) * 1000.0)
        for (idx, _, _), vout in zip(chunk, vouts):
            results[idx] = vout
    total = _t.perf_counter() - t0
    logger.info("verifier wall = %.1fs  (%.2f sec/sample)",
                total, total / max(len(triples), 1))

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        rec: Dict[str, Any] = {
            "id":        r.get("id"),
            "benchmark": r.get("benchmark"),
        }
        vout = results.get(i)
        if vout is None:
            for f in GATED_FIELDS:
                rec[f] = None
            rec["decision"] = None
        else:
            for f in GATED_FIELDS:
                rec[f] = float(getattr(vout, f)) if getattr(vout, f) is not None else None
            rec["decision"] = getattr(vout, "decision", None)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Compare two runs
# ---------------------------------------------------------------------------

def _pearson(xs: List[float], ys: List[float]) -> float:
    import math
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _compare(ref: List[Dict[str, Any]],
             opt: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-signal Pearson r + max |delta|. Decision flip count. Verdict."""
    # Align by id (and benchmark, defensively).
    key_to_opt: Dict[Tuple[Any, Any], Dict[str, Any]] = {
        (r["id"], r["benchmark"]): r for r in opt
    }
    aligned: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for r in ref:
        key = (r["id"], r["benchmark"])
        o = key_to_opt.get(key)
        if o is None:
            continue
        aligned.append((r, o))
    if not aligned:
        return {"n": 0, "verdict": "FAIL", "reason": "no id overlap"}

    per_signal: Dict[str, Dict[str, float]] = {}
    for f in GATED_FIELDS:
        ref_vals: List[float] = []
        opt_vals: List[float] = []
        for a, b in aligned:
            if a.get(f) is None or b.get(f) is None:
                continue
            ref_vals.append(float(a[f]))
            opt_vals.append(float(b[f]))
        if len(ref_vals) < 2:
            per_signal[f] = {"n": len(ref_vals), "pearson": float("nan"),
                             "max_abs_delta": float("nan"), "pass": False}
            continue
        r = _pearson(ref_vals, opt_vals)
        mad = max(abs(x - y) for x, y in zip(ref_vals, opt_vals))
        ok = (r >= PEARSON_MIN) and (mad < MAX_ABS_DELTA)
        per_signal[f] = {
            "n": len(ref_vals),
            "pearson": r,
            "max_abs_delta": mad,
            "pass": bool(ok),
        }

    decision_flips = sum(
        1 for a, b in aligned
        if a.get("decision") is not None
        and b.get("decision") is not None
        and a["decision"] != b["decision"]
    )

    overall_ok = (
        all(per_signal[f]["pass"] for f in GATED_FIELDS)
        and decision_flips <= DECISION_FLIPS_MAX
    )

    return {
        "n":              len(aligned),
        "pearson_min":    PEARSON_MIN,
        "max_abs_delta":  MAX_ABS_DELTA,
        "per_signal":     per_signal,
        "decision_flips": decision_flips,
        "decision_flips_max_allowed": DECISION_FLIPS_MAX,
        "verdict":        "PASS" if overall_ok else "FAIL",
    }


def _print_verdict(v: Dict[str, Any]) -> None:
    n = v.get("n", 0)
    print(f"\n=== Equivalence verdict on n={n} samples ===")
    print(f"  acceptance: Pearson >= {v.get('pearson_min')} AND "
          f"max |delta| < {v.get('max_abs_delta')} AND decision flips == "
          f"{v.get('decision_flips_max_allowed')}")
    print()
    print(f"{'signal':<22} {'n':>5} {'pearson':>10} {'max|d|':>9} {'pass':>6}")
    print("-" * 60)
    for f in GATED_FIELDS:
        s = v["per_signal"].get(f, {})
        print(f"  {f:<20} {s.get('n', 0):>5} "
              f"{s.get('pearson', float('nan')):>10.5f} "
              f"{s.get('max_abs_delta', float('nan')):>9.5f} "
              f"{'YES' if s.get('pass') else 'NO':>6}")
    print()
    print(f"  decision flips: {v['decision_flips']} "
          f"(max allowed = {v['decision_flips_max_allowed']})")
    print()
    print(f"  >>> VERDICT: {v['verdict']} <<<")
    print()
    if v["verdict"] != "PASS":
        print("  FAIL: do NOT launch v2 rescore with these optimisation flags.")
        print("  Identify the failing signal(s) and revert the corresponding ")
        print("  optimisation before retrying.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", required=True,
                    choices=("reference", "optimised", "compare"))
    ap.add_argument("--composite_calibration", type=Path,
                    default=Path("outputs/full_run/cycle_3/composite_calibration.json"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("outputs/full_run/cycle_3/adapter"))
    ap.add_argument("--passage_index", type=Path, default=Path("data/passage_index"))
    ap.add_argument("--eval_dir", type=Path, default=Path("outputs/full_run/eval"))
    ap.add_argument("--benches", nargs="+",
                    default=["fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa"])
    ap.add_argument("--per_bench", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16,
                    help="In reference mode use 16; in optimised mode use 32.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path,
                    help="Compare mode: path to reference JSON.")
    ap.add_argument("--optimised", type=Path,
                    help="Compare mode: path to optimised JSON.")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        if not args.reference or not args.optimised:
            logger.error("--reference and --optimised both required in compare mode")
            return 1
        ref = json.loads(args.reference.read_text())["rows"]
        opt = json.loads(args.optimised.read_text())["rows"]
        verdict = _compare(ref, opt)
        verdict["reference_path"] = str(args.reference)
        verdict["optimised_path"] = str(args.optimised)
        verdict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.out.write_text(json.dumps(verdict, indent=2))
        _print_verdict(verdict)
        return 0 if verdict["verdict"] == "PASS" else 2

    # reference / optimised mode -> run the verifier and dump per-row signals
    rows = _load_calfold_samples(
        args.eval_dir, args.per_bench, tuple(args.benches), seed=args.seed,
    )
    logger.info("drew %d cal-fold samples total", len(rows))
    if not rows:
        logger.error("no cal-fold rows available; check --eval_dir")
        return 1

    verifier, _ = _build_verifier(
        composite_calibration=args.composite_calibration,
        checkpoint=args.checkpoint,
        passage_index=args.passage_index,
        device=args.device,
    )

    out_rows = _run_verifier(verifier, rows, batch_size=args.batch_size)

    doc = {
        "mode":      args.mode,
        "n":         len(out_rows),
        "batch_size": args.batch_size,
        "env_CAEM_BATCH_U_TOK_DROP": os.environ.get("CAEM_BATCH_U_TOK_DROP", "0"),
        "composite_calibration": str(args.composite_calibration),
        "checkpoint":             str(args.checkpoint),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows":      out_rows,
    }
    args.out.write_text(json.dumps(doc, indent=1))
    logger.info("wrote %d rows -> %s", len(out_rows), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
