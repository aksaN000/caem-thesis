#!/usr/bin/env python
"""
scripts/render_production_samples.py
=====================================
Thesis Chapter 6 post-processor — reads per-sample eval JSONs and
renders representative samples under the production confidence envelope.

Inputs
------
  --eval_dir outputs/cycle_0/eval                 (or cycle_N for Step 7 main)
  --out_dir  "pre thesis 1 report/tables"         (optional; default prints)
  --per_tag  N                                    (samples per tag, default 2)

Outputs
-------
Stdout (always) — terminal-formatted rendering of 2 examples per tag.
LaTeX tables   — when --out_dir given:
  tab_production_envelope_examples.tex   (2 samples per tag with caveats)
  tab_production_distribution.tex        (aggregate tag counts + mean confidence)
Cadence demo   — rendering of one deferred sample under all three cadence modes
  (generic / nightly / explicit ETA) for the Ch6 deployment figure.

Usage
-----
  python scripts/render_production_samples.py \\
      --eval_dir outputs/cycle_0/eval

  python scripts/render_production_samples.py \\
      --eval_dir outputs/full_run/cycle_10/eval \\
      --out_dir "pre thesis 1 report/tables"

The script is idempotent and read-only on eval JSONs. Safe to run any
time, even while the main runner is active.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from caem.production.confidence import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    VALID_DECISIONS,
    render,
)

_ICON_GLYPH = {
    "check":   "✓",
    "tilde":   "~",
    "warning": "⚠",
    "cross":   "✗",
}
_TAG_ORDER = ["verified", "provisional", "conflicting", "insufficient"]


def _load_samples(eval_dir: Path) -> List[Dict[str, Any]]:
    """Load all per-sample records from all benchmark JSONs in eval_dir."""
    samples: List[Dict[str, Any]] = []
    for p in sorted(eval_dir.glob("*.json")):
        try:
            data = json.load(open(p))
        except Exception as exc:
            print(f"[warn] failed to load {p}: {exc}")
            continue
        records = data.get("samples") or data.get("results") or data
        if not isinstance(records, list):
            continue
        for r in records:
            if isinstance(r, dict):
                r.setdefault("_source_file", p.name)
                samples.append(r)
    return samples


def _render_terminal(
    record: Dict[str, Any],
    *,
    deferred_eta: Optional[str] = None,
    deferred_cadence: Optional[str] = None,
) -> str:
    """Render one sample in terminal-friendly block format.

    If ``deferred_eta`` or ``deferred_cadence`` is given, always re-renders
    from the raw decision+u_stored so the caveat reflects the deployment
    override (bypassing any cached ``production_response`` field).
    """
    question = record.get("question") or "<no question>"
    use_override = bool(deferred_eta or deferred_cadence)
    pr = None if use_override else record.get("production_response")
    if pr is None:
        decision = record.get("decision")
        u_stored = record.get("u_stored")
        answer = record.get("display_answer") or record.get("prediction") or ""
        if decision not in VALID_DECISIONS or u_stored is None:
            return (
                f"Q: {question}\n"
                f"   [no CAEM decision — Tier 1 hit or baseline sample]\n"
            )
        resp = render(
            str(answer), float(u_stored), str(decision),
            deferred_eta=deferred_eta, deferred_cadence=deferred_cadence,
        )
        pr = resp.to_dict() if resp else None
        if pr is None:
            return f"Q: {question}\n   [invalid decision {decision}]\n"

    icon = _ICON_GLYPH.get(pr["icon"], "·")
    pct = int(round(100 * pr["confidence"]))
    tag = pr["tag"]
    label = pr["label"].replace("_", " ")

    if pr["show_answer"]:
        answer_line = pr["answer"] or ""
    else:
        answer_line = INSUFFICIENT_EVIDENCE_MESSAGE

    block = [
        f"Q: {question}",
        f"   [{icon} {tag} · {pct}% · {label}]",
        f"   A: {answer_line}",
    ]
    if pr.get("caveat"):
        block.append(f"      └─ {pr['caveat']}")
    return "\n".join(block) + "\n"


def _partition_by_tag(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group samples by their production response tag."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in samples:
        pr = r.get("production_response")
        if pr is None:
            # Try reconstruct from raw fields
            decision = r.get("decision")
            u_stored = r.get("u_stored")
            if decision not in VALID_DECISIONS or u_stored is None:
                continue
            answer = r.get("display_answer") or r.get("prediction") or ""
            resp = render(str(answer), float(u_stored), str(decision))
            if resp is None:
                continue
            pr = resp.to_dict()
            r["production_response"] = pr
        buckets[pr["tag"]].append(r)
    return buckets


def _aggregate_stats(buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """Per-tag: count, mean confidence, share-of-total, show_answer_fraction."""
    total = sum(len(v) for v in buckets.values())
    stats: Dict[str, Dict[str, float]] = {}
    for tag in _TAG_ORDER:
        items = buckets.get(tag, [])
        if not items:
            stats[tag] = {
                "count": 0, "share": 0.0, "mean_confidence": 0.0, "shown_pct": 0.0,
            }
            continue
        confs = [it["production_response"]["confidence"] for it in items]
        shown = sum(1 for it in items if it["production_response"]["show_answer"])
        stats[tag] = {
            "count": len(items),
            "share": len(items) / total if total else 0.0,
            "mean_confidence": statistics.mean(confs),
            "shown_pct": 100 * shown / len(items),
        }
    stats["_total"] = {
        "count": total, "share": 1.0,
        "mean_confidence": 0.0, "shown_pct": 0.0,
    }
    return stats


def _emit_terminal(
    buckets: Dict[str, List[Dict[str, Any]]],
    stats: Dict[str, Dict[str, float]],
    per_tag: int,
    seed: int,
) -> None:
    """Print examples + aggregate stats to stdout."""
    rng = random.Random(seed)
    print("=" * 72)
    print("Production confidence envelope — sample renderings")
    print("=" * 72)
    for tag in _TAG_ORDER:
        items = buckets.get(tag, [])
        s = stats[tag]
        print()
        print(f"--- {tag.upper()}  "
              f"(n={s['count']}, {s['share']:.1%} of samples, "
              f"mean confidence {s['mean_confidence']:.3f}) ---")
        if not items:
            print("  (no samples)")
            continue
        picks = rng.sample(items, k=min(per_tag, len(items)))
        for rec in picks:
            print()
            print(_render_terminal(rec))
    print()
    print("=" * 72)
    print("Aggregate distribution")
    print("=" * 72)
    print(f"  {'tag':<14}  {'count':>6}  {'share':>7}  "
          f"{'mean conf':>10}  {'answer shown':>13}")
    for tag in _TAG_ORDER:
        s = stats[tag]
        print(f"  {tag:<14}  {s['count']:>6}  {s['share']:>7.1%}  "
              f"{s['mean_confidence']:>10.3f}  {s['shown_pct']:>11.1f}%")
    print(f"  {'TOTAL':<14}  {stats['_total']['count']:>6}")


def _emit_cadence_demo(buckets: Dict[str, List[Dict[str, Any]]], seed: int) -> None:
    """Render one DEFERRED sample under generic / named-cadence / explicit-ETA."""
    deferred = buckets.get("provisional", [])
    if not deferred:
        print("\n[cadence demo] no provisional samples available to demo.")
        return
    rng = random.Random(seed)
    rec = rng.choice(deferred)
    print()
    print("=" * 72)
    print("Cadence demo — same DEFERRED sample, three rendering modes")
    print("=" * 72)
    print()
    print("[1/3] Generic (no deployment config):")
    print(_render_terminal(rec))
    print("[2/3] Named cadence (deferred_cadence='nightly'):")
    print(_render_terminal(rec, deferred_cadence="nightly"))
    print("[3/3] Explicit ETA (deferred_eta='tomorrow 6am'):")
    print(_render_terminal(rec, deferred_eta="tomorrow 6am"))


def _emit_latex_examples(
    buckets: Dict[str, List[Dict[str, Any]]],
    out: Path,
    per_tag: int,
    seed: int,
) -> None:
    """Write a LaTeX table of example renderings per tag."""
    rng = random.Random(seed)
    lines = [
        r"% Auto-generated by scripts/render_production_samples.py",
        r"% Ch6 §User-Facing Deployment — sample renderings by tag.",
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{Production confidence envelope — example renderings from "
        r"Phase 1a evaluation. Tags are determined by the Stage-5 decision "
        r"tree; confidence is the composite $u_{\mathrm{stored}}$. "
        r"\emph{Insufficient} samples hide the generated answer.}",
        r"\label{tab:production_envelope_examples}",
        r"\begin{tabularx}{\linewidth}{l r X}",
        r"\toprule",
        r"\textbf{Tag} & \textbf{Conf.} & \textbf{Sample} \\",
        r"\midrule",
    ]
    for tag in _TAG_ORDER:
        items = buckets.get(tag, [])
        if not items:
            continue
        picks = rng.sample(items, k=min(per_tag, len(items)))
        for i, rec in enumerate(picks):
            pr = rec["production_response"]
            question = (rec.get("question") or "")[:120].replace("&", r"\&").replace("_", r"\_")
            answer = pr["answer"] if pr["show_answer"] else INSUFFICIENT_EVIDENCE_MESSAGE
            answer = (answer or "")[:160].replace("&", r"\&").replace("_", r"\_")
            pct = int(round(100 * pr["confidence"]))
            lines.append(
                rf"{tag} & {pct}\% & "
                rf"\textbf{{Q:}} {question} \\[2pt] "
                rf"& & \textbf{{A:}} {answer} \\"
            )
            if pr.get("caveat"):
                caveat = pr["caveat"].replace("&", r"\&").replace("_", r"\_")
                lines.append(rf"& & {{\itshape {caveat}}} \\")
            if i < len(picks) - 1 or tag != _TAG_ORDER[-1]:
                lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[latex] wrote {out}")


def _emit_latex_distribution(
    stats: Dict[str, Dict[str, float]], out: Path,
) -> None:
    """Write aggregate distribution table."""
    lines = [
        r"% Auto-generated by scripts/render_production_samples.py",
        r"% Ch6 §User-Facing Deployment — aggregate tag distribution.",
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{Aggregate distribution of production tags on the "
        r"Phase 1a evaluation set ($n = " + str(stats['_total']['count']) +
        r"$). \emph{Answer shown} denotes the fraction of samples where the "
        r"envelope surfaces a visible answer (false only for insufficient).}",
        r"\label{tab:production_distribution}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"\textbf{Tag} & \textbf{Count} & \textbf{Share} & "
        r"\textbf{Mean conf.} & \textbf{Answer shown} \\",
        r"\midrule",
    ]
    for tag in _TAG_ORDER:
        s = stats[tag]
        lines.append(
            rf"{tag} & {s['count']} & {s['share']:.1%} & "
            rf"{s['mean_confidence']:.3f} & {s['shown_pct']:.1f}\% \\"
        )
    lines += [
        r"\midrule",
        rf"Total & {stats['_total']['count']} & 100\% & -- & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"[latex] wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--eval_dir", type=Path, required=True,
                        help="Directory containing per-benchmark eval JSONs.")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="If given, write LaTeX tables here.")
    parser.add_argument("--per_tag", type=int, default=2,
                        help="Samples to render per tag (default 2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection.")
    args = parser.parse_args()

    eval_dir: Path = args.eval_dir
    if not eval_dir.is_dir():
        print(f"[error] eval_dir not found: {eval_dir}")
        return 1

    samples = _load_samples(eval_dir)
    if not samples:
        print(f"[error] no samples loaded from {eval_dir}")
        return 1
    print(f"[info] loaded {len(samples)} samples from {eval_dir}")

    buckets = _partition_by_tag(samples)
    stats = _aggregate_stats(buckets)
    _emit_terminal(buckets, stats, per_tag=args.per_tag, seed=args.seed)
    _emit_cadence_demo(buckets, seed=args.seed)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _emit_latex_examples(
            buckets, args.out_dir / "tab_production_envelope_examples.tex",
            per_tag=args.per_tag, seed=args.seed,
        )
        _emit_latex_distribution(
            stats, args.out_dir / "tab_production_distribution.tex",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
