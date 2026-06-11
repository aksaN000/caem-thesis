#!/usr/bin/env python3
"""
scripts/compare_b6_old_vs_new.py
================================
Side-by-side comparison of B6@C5 old (broken-cap, pre-2026-06-11) eval JSONs
versus the new (matched-protocol fix + post-processed) eval JSONs.

Reads:
  OLD:  outputs/baselines_phase1e/vanilla_ft/eval/{bench}_cycle5.json
  NEW:  outputs/baselines_phase1e/vanilla_ft/eval_fixed/{bench}_cycle5.json

Reports:
  - Per-bench EM old vs new + delta + percentage points change
  - Pooled EM old vs new
  - Per-bench fraction of predictions whose final extracted answer changed
  - 5 sample (question, old_pred, new_pred, gold) tuples per bench where the
    EM verdict flipped (old EM 0 -> new EM 1)

Usage
-----
    python -m scripts.compare_b6_old_vs_new

    # Or save to file:
    python -m scripts.compare_b6_old_vs_new > /tmp/b6_diff.txt

Notes
-----
TruthfulQA EM uses meta.em for both old and new (strict-EM); the canonical
Haiku-judged em_llm_judged comes from the rescore_truthfulqa.py pass in
Phase 3.2. The strict-EM TQA "regression" old->new is expected: the OLD
JSONs used the ROUGE-L > 0.15 default rule (inflated long-prose match);
the NEW JSONs use strict EM. Haiku-judged numbers are the canonical
comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

OLD_DIR = ROOT / "outputs" / "baselines_phase1e" / "vanilla_ft" / "eval"
NEW_DIR = ROOT / "outputs" / "baselines_phase1e" / "vanilla_ft" / "eval_fixed"

PANEL = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _index_by_id(samples: list) -> Dict[str, dict]:
    return {s["id"]: s for s in samples if "id" in s}


def compare_bench(bench: str) -> Optional[Dict]:
    old_path = OLD_DIR / f"{bench}_cycle5.json"
    new_path = NEW_DIR / f"{bench}_cycle5.json"
    if not old_path.exists() or not new_path.exists():
        return None

    old = _load(old_path)
    new = _load(new_path)
    old_em = float(old["meta"]["em"])
    new_em = float(new["meta"]["em"])
    delta_em = new_em - old_em

    old_idx = _index_by_id(old.get("samples", []))
    new_idx = _index_by_id(new.get("samples", []))

    shared_ids = sorted(set(old_idx) & set(new_idx))
    n_pred_changed = 0
    flipped_0_to_1: List[Tuple[str, str, str, list]] = []
    flipped_1_to_0: List[Tuple[str, str, str, list]] = []

    for sid in shared_ids:
        old_s = old_idx[sid]
        new_s = new_idx[sid]
        if old_s.get("display_answer") != new_s.get("display_answer"):
            n_pred_changed += 1
        old_em_s = old_s.get("em", 0.0)
        new_em_s = new_s.get("em", 0.0)
        if old_em_s == 0.0 and new_em_s == 1.0:
            flipped_0_to_1.append((
                old_s.get("question", "")[:160],
                old_s.get("display_answer", ""),
                new_s.get("display_answer", ""),
                new_s.get("gold_answers", [new_s.get("gold_label", "")]),
            ))
        elif old_em_s == 1.0 and new_em_s == 0.0:
            flipped_1_to_0.append((
                old_s.get("question", "")[:160],
                old_s.get("display_answer", ""),
                new_s.get("display_answer", ""),
                new_s.get("gold_answers", [new_s.get("gold_label", "")]),
            ))

    return {
        "bench": bench,
        "n": len(shared_ids),
        "old_em": old_em,
        "new_em": new_em,
        "delta_em": delta_em,
        "n_pred_changed": n_pred_changed,
        "pct_pred_changed": (n_pred_changed / max(len(shared_ids), 1)) * 100.0,
        "n_flipped_0_to_1": len(flipped_0_to_1),
        "n_flipped_1_to_0": len(flipped_1_to_0),
        "sample_0_to_1": flipped_0_to_1[:5],
        "sample_1_to_0": flipped_1_to_0[:5],
    }


def main() -> None:
    print("=" * 90)
    print("B6@C5 OLD (broken-cap) vs NEW (matched-protocol + Reasoning:<X> extraction fix)")
    print("=" * 90)
    print()

    summaries = []
    for bench in PANEL:
        summary = compare_bench(bench)
        if summary is None:
            print(f"  SKIP: {bench} (missing old or new JSON)")
            continue
        summaries.append(summary)

    # Headline table.
    print(f"{'Benchmark':<18}{'OLD EM':>10}{'NEW EM':>10}{'Δ EM':>10}{'Δ pp':>10}{'%-pred changed':>18}{'flips 0→1':>12}{'flips 1→0':>12}")
    print("-" * 100)
    for s in summaries:
        delta_pp = s["delta_em"] * 100
        print(
            f"{s['bench']:<18}{s['old_em']:>10.4f}{s['new_em']:>10.4f}"
            f"{s['delta_em']:>+10.4f}{delta_pp:>+10.1f}"
            f"{s['pct_pred_changed']:>17.1f}%{s['n_flipped_0_to_1']:>12}{s['n_flipped_1_to_0']:>12}"
        )

    pooled_old = sum(s["old_em"] for s in summaries) / max(len(summaries), 1)
    pooled_new = sum(s["new_em"] for s in summaries) / max(len(summaries), 1)
    print("-" * 100)
    print(
        f"{'POOLED (unweighted)':<18}{pooled_old:>10.4f}{pooled_new:>10.4f}"
        f"{pooled_new - pooled_old:>+10.4f}{(pooled_new - pooled_old) * 100:>+10.1f}"
    )
    print()

    # Sample diffs per bench (where EM verdict flipped).
    for s in summaries:
        if not s["sample_0_to_1"] and not s["sample_1_to_0"]:
            continue
        print()
        print(f"=== {s['bench'].upper()} — sample EM flips ===")
        if s["sample_0_to_1"]:
            print(f"  Flips OLD=0 → NEW=1 (fix recovered correct answer): showing up to 5 of {s['n_flipped_0_to_1']}")
            for q, old_p, new_p, gold in s["sample_0_to_1"]:
                print(f"    Q: {q!s}")
                print(f"      OLD pred: {old_p!r}")
                print(f"      NEW pred: {new_p!r}")
                print(f"      Gold:     {gold!r}")
                print()
        if s["sample_1_to_0"]:
            print(f"  Flips OLD=1 → NEW=0 (NEW lost an answer the OLD got): showing up to 5 of {s['n_flipped_1_to_0']}")
            for q, old_p, new_p, gold in s["sample_1_to_0"]:
                print(f"    Q: {q!s}")
                print(f"      OLD pred: {old_p!r}")
                print(f"      NEW pred: {new_p!r}")
                print(f"      Gold:     {gold!r}")
                print()


if __name__ == "__main__":
    main()
