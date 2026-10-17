#!/usr/bin/env python3
"""Live watcher for the post-A2v2-fix p_ground_atomic signal.

Polls outputs/cycle_0/{eval,calibration} every POLL_SECS and, when new
JSONs appear, prints a one-line summary of:
  - n samples
  - p_ground_atomic distribution (mean / median / fraction at fallback)
  - p_ground_mean distribution (for fallback comparison)
  - Cohen's d for atomic vs em correctness (when em available)
  - decision counts

Designed to run in parallel with the in-progress Step 7.0 run without
touching the GPU.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

POLL_SECS = 30
WATCH_DIRS = [
    Path("outputs/cycle_0/eval"),
    Path("outputs/cycle_0/calibration"),
]


def cohen_d(positives, negatives):
    if len(positives) < 2 or len(negatives) < 2:
        return float("nan")
    m1, m0 = statistics.mean(positives), statistics.mean(negatives)
    s1, s0 = statistics.stdev(positives), statistics.stdev(negatives)
    n1, n0 = len(positives), len(negatives)
    pooled = math.sqrt(((n1 - 1) * s1 * s1 + (n0 - 1) * s0 * s0) / (n1 + n0 - 2))
    return (m1 - m0) / pooled if pooled > 0 else float("nan")


def analyze(json_path: Path) -> str:
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as exc:
        return f"  ERROR reading {json_path.name}: {exc}"

    samples = data.get("samples") or data.get("results") or data
    if not isinstance(samples, list) or not samples:
        return f"  {json_path.name}: no samples found"

    atomic = []
    pmean = []
    em_pos_atomic = []
    em_neg_atomic = []
    decisions: dict[str, int] = {}
    fallback_count = 0
    for s in samples:
        a = s.get("p_ground_atomic")
        m = s.get("p_ground_mean")
        em = s.get("em") if "em" in s else s.get("correct")
        d = s.get("decision")
        if d is not None:
            decisions[d] = decisions.get(d, 0) + 1
        if isinstance(a, (int, float)) and isinstance(m, (int, float)):
            atomic.append(float(a))
            pmean.append(float(m))
            if abs(float(a) - float(m)) < 1e-6:
                fallback_count += 1
            if em in (0, 1, 0.0, 1.0):
                if int(em) == 1:
                    em_pos_atomic.append(float(a))
                else:
                    em_neg_atomic.append(float(a))

    if not atomic:
        return f"  {json_path.name}: no p_ground_atomic field"

    n = len(atomic)
    m_atomic = statistics.mean(atomic)
    med_atomic = statistics.median(atomic)
    m_pmean = statistics.mean(pmean)
    fb_pct = 100.0 * fallback_count / n
    cd = cohen_d(em_pos_atomic, em_neg_atomic)
    dec_str = " ".join(f"{k}={v}" for k, v in sorted(decisions.items()))

    return (
        f"  {json_path.name}: n={n}  "
        f"atomic mean={m_atomic:.3f} med={med_atomic:.3f}  "
        f"p_ground_mean={m_pmean:.3f}  "
        f"fallback={fb_pct:.0f}%  "
        f"Cohen's d (atomic vs em)={cd:+.3f}"
        + (f"\n     decisions: {dec_str}" if dec_str else "")
    )


def main():
    seen: dict[Path, float] = {}
    print(f"[watch] polling {[str(p) for p in WATCH_DIRS]} every {POLL_SECS}s")
    print(f"[watch] started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    while True:
        new = []
        for d in WATCH_DIRS:
            if not d.exists():
                continue
            for p in d.glob("*.json"):
                mtime = p.stat().st_mtime
                if seen.get(p) != mtime:
                    seen[p] = mtime
                    new.append(p)
        if new:
            ts = time.strftime("%H:%M:%S")
            print(f"\n[{ts}] new/updated:")
            for p in sorted(new):
                print(analyze(p))
            sys.stdout.flush()
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
