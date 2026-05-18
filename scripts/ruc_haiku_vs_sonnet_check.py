#!/usr/bin/env python3
"""scripts/ruc_haiku_vs_sonnet_check.py
==========================================
200-row empirical comparison of Claude Haiku 4.5 vs Claude Sonnet 4.6 on the
RUC binary-judge labelling task. Tells us — on OUR data distribution, not
literature averages — whether Haiku is good enough to replace Sonnet for
the Phase 2 v2 labelling pass (~$22 vs $81 for 15k rows, ~$52 vs $194 for
36k rows).

What it does
------------
1. Sample 200 random rows from caem/ruc/training_set.parquet that already
   have Sonnet labels in caem/ruc/sonnet_labels.jsonl.
2. Re-label each row with Claude Haiku 4.5 using the exact same v3 prompt
   (TIE option, position-swap protocol).
3. Compute three numbers:
   - Raw agreement between Haiku and Sonnet on non-tie verdicts
   - Cohen's kappa between Haiku and Sonnet labels
   - Per-benchmark breakdown (especially TruthfulQA — the adversarial slice)
4. Print a recommendation: pure-Haiku / hybrid / pure-Sonnet.

Cost: ~$0.30 (200 rows × 2 swaps × ~$0.0007/Haiku-call).

Usage
-----
    python -m scripts.ruc_haiku_vs_sonnet_check --n 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the Sonnet labeller's parser + prompt + judge call infrastructure
from scripts.label_ruc_with_sonnet import (
    JUDGE_PROMPT_TEMPLATE,
    _extract_verdict,
    _verdict_to_rag_better,
    _format_gold,
)

HAIKU_MODEL_ID = "claude-haiku-4-5"
_LAST = {"t": 0.0}
_LOCK = Lock()
_MIN_INTERVAL_SEC = 0.6  # ~100 req/min


def _throttle() -> None:
    with _LOCK:
        elapsed = time.monotonic() - _LAST["t"]
        if elapsed < _MIN_INTERVAL_SEC:
            time.sleep(_MIN_INTERVAL_SEC - elapsed)
        _LAST["t"] = time.monotonic()


def _build_client(api_key: Optional[str]) -> Any:
    import anthropic
    return anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        timeout=30.0, max_retries=0,
    )


def _haiku_judge(client, question, gold, ans_a, ans_b, max_retries=3) -> Tuple[Optional[str], str]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question.strip() or "(empty)",
        gold=gold or "(no gold reference available)",
        answer_a=(ans_a or "(empty)").strip()[:2000],
        answer_b=(ans_b or "(empty)").strip()[:2000],
    )
    for attempt in range(1, max_retries + 1):
        _throttle()
        try:
            resp = client.messages.create(
                model=HAIKU_MODEL_ID, max_tokens=128, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "")
                          for b in resp.content
                          if getattr(b, "type", "text") == "text").strip()
            verdict, reason = _extract_verdict(text)
            if verdict is not None:
                return verdict, reason
            time.sleep(min(2 ** attempt, 4))
        except Exception as exc:
            if "rate" in str(exc).lower() or "429" in str(exc):
                time.sleep(60)
            else:
                time.sleep(min(2 ** attempt, 8))
    return None, ""


def _label_one_row(client, row) -> Dict[str, Any]:
    gold = _format_gold(row)
    direct = row.get("direct_prediction") or ""
    rag = row.get("rag_prediction") or ""
    v1, _ = _haiku_judge(client, row["question"], gold, direct, rag)
    v2, _ = _haiku_judge(client, row["question"], gold, rag, direct)
    rb1 = _verdict_to_rag_better(v1, pass_num=1)
    rb2 = _verdict_to_rag_better(v2, pass_num=2)
    if rb1 is None and rb2 is None:
        label, conf = None, None
    elif rb1 is None:
        label, conf = int(bool(rb2)), 0.5
    elif rb2 is None:
        label, conf = int(bool(rb1)), 0.5
    elif rb1 == rb2:
        label, conf = int(bool(rb1)), 1.0
    else:
        label, conf = int(bool(rb1)), 0.5
    return {
        "id": str(row["id"]), "benchmark": row["benchmark"],
        "haiku_label": label, "haiku_confidence": conf,
        "haiku_v1": v1, "haiku_v2": v2,
    }


def _cohens_kappa(a, b):
    if not a:
        return float("nan")
    n = len(a)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    p_e = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - p_e) < 1e-9:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("caem/ruc/haiku_vs_sonnet_check.json"))
    args = ap.parse_args()

    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)

    import pandas as pd
    df = pd.read_parquet(args.in_parquet)
    # Sample only rows with a Sonnet label (skip ties / failed)
    sampled = df[df["sonnet_label"].notna()].sample(n=min(args.n, len(df)),
                                                     random_state=args.seed)
    logger.info("Will compare Haiku vs Sonnet on %d rows", len(sampled))

    client = _build_client(api_key=None)
    rows = sampled.to_dict(orient="records")
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_label_one_row, client, r): r for r in rows}
        completed = 0
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                logger.warning("worker failed: %s", exc)
                continue
            res["sonnet_label"] = int(r["sonnet_label"])
            res["empirical_label"] = int(r["empirical_label"])
            results.append(res)
            completed += 1
            if completed % 20 == 0:
                logger.info("progress: %d / %d", completed, len(sampled))

    # Compute agreement metrics
    haiku_labelled = [r for r in results if r["haiku_label"] is not None]
    n_total = len(results)
    n_h_labelled = len(haiku_labelled)
    n_haiku_strong = sum(1 for r in haiku_labelled if r["haiku_confidence"] == 1.0)
    agree = [int(r["haiku_label"] == r["sonnet_label"]) for r in haiku_labelled]
    raw_agree = sum(agree) / len(agree) if agree else float("nan")
    kappa = _cohens_kappa(
        [r["haiku_label"] for r in haiku_labelled],
        [r["sonnet_label"] for r in haiku_labelled],
    )
    # Strong-agreement subset
    strong = [r for r in haiku_labelled if r["haiku_confidence"] == 1.0]
    strong_agree = sum(1 for r in strong if r["haiku_label"] == r["sonnet_label"]) / max(len(strong), 1)

    # Per-benchmark
    by_bench: Dict[str, Dict[str, float]] = {}
    for bench in ["fever", "triviaqa", "commonsense_qa", "strategyqa", "truthfulqa"]:
        sub = [r for r in haiku_labelled if r["benchmark"] == bench]
        if not sub:
            continue
        ag = sum(int(r["haiku_label"] == r["sonnet_label"]) for r in sub) / len(sub)
        by_bench[bench] = {"n": len(sub), "agreement": round(ag, 3)}

    summary = {
        "n_total": n_total,
        "n_haiku_labelled": n_h_labelled,
        "n_haiku_strong": n_haiku_strong,
        "haiku_strong_pct": round(n_haiku_strong / max(n_total, 1), 3),
        "raw_agreement_with_sonnet": round(raw_agree, 3),
        "cohen_kappa": round(kappa, 3),
        "strong_only_agreement": round(strong_agree, 3),
        "per_benchmark": by_bench,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*66}")
    print(f"Haiku-vs-Sonnet 200-row sanity check")
    print(f"{'='*66}")
    print(f"  n total                        {n_total}")
    print(f"  n Haiku labelled (non-tie)     {n_h_labelled}")
    print(f"  n Haiku strong (both swaps OK) {n_haiku_strong} ({n_haiku_strong/max(n_total,1)*100:.1f}%)")
    print(f"  raw agreement with Sonnet      {raw_agree:.3f}")
    print(f"  Cohen's kappa                  {kappa:.3f}")
    print(f"  strong-only agreement          {strong_agree:.3f}")
    print(f"\nPer-benchmark agreement:")
    for b, v in by_bench.items():
        print(f"  {b:<18} n={v['n']:>3}  agree={v['agreement']:.3f}")
    print(f"\nResults saved to {args.out}")

    print(f"\n{'='*66}")
    print("Recommendation:")
    print(f"{'='*66}")
    if raw_agree >= 0.90:
        print(f"  PURE HAIKU is safe — agreement {raw_agree:.0%} is high enough")
        print(f"  Phase 2 stretch labelling cost: ~$52 instead of ~$194")
    elif raw_agree >= 0.80:
        print(f"  HYBRID recommended (Haiku primary + Sonnet escalation on disagreement)")
        print(f"  Phase 2 stretch labelling cost: ~$120 instead of ~$194")
    else:
        print(f"  PURE SONNET recommended — Haiku agreement {raw_agree:.0%} too low")
        print(f"  Phase 2 stretch labelling cost: ~$194 (no savings)")
    if "truthfulqa" in by_bench:
        tqa = by_bench["truthfulqa"]["agreement"]
        if tqa < raw_agree - 0.10:
            print(f"\n  WARNING: TruthfulQA agreement ({tqa:.0%}) much lower than pooled "
                  f"({raw_agree:.0%}).")
            print(f"  This is the benchmark we care about most. Hybrid or pure-Sonnet "
                  f"strongly preferred.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
