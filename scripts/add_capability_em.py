#!/usr/bin/env python3
"""Add a ``capability_em`` field next to existing ``em`` for MCQ/structured benches.

Motivation (2026-05-16)
-----------------------
B6 vanilla_ft cycle 1 produced CSQA predictions of the form ``Reasoning:C``
(model truncated the chain to a single letter), which the strict ``em``
scorer cannot parse because it expects an ``Answer: X`` line. The strict
``em`` therefore reads 0.000 across all 300 samples even though the model
emitted the correct letter for many of them.

``capability_em`` answers the question: "if we had a perfect parser, would
the model be right?" — a lenient extraction that finds the canonical
answer token anywhere in the prediction. It is applied uniformly across
all systems (B1, B2, ..., B6, CAEM all cycles) so the matched-protocol
invariant is preserved.

This field is for diagnostic / Ch5-§sec:comp-vanilla-ft framing only.
Strict ``em`` remains the headline deployment-grade number.

Per-benchmark extractor
-----------------------
- ``commonsense_qa``: first word-boundary capital letter in [A-E]
- ``strategyqa``:    first standalone yes/no/true/false token
- ``fever``:         first standalone supports/refutes/nei (case-insens.)
- ``triviaqa``:      no fallback (open-ended) → capability_em = em
- ``truthfulqa``:    no fallback (open-ended) → capability_em = em

Usage
-----
    # Annotate all baseline + CAEM eval JSONs in place (idempotent)
    python -m scripts.add_capability_em \\
        --baselines_dir outputs/baselines \\
        --caem_dir outputs/full_run/eval

The script skips samples that already have ``capability_em`` set so it
can be re-run safely after new baselines or new cycles land.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Per-benchmark capability extractors
# ---------------------------------------------------------------------

# Strict-form patterns: prefer the canonical "Answer: <X>" line. These
# match the same scaffolding the strict scorer expects, so when the
# prediction is well-formed, capability_em == strict em.
_ANSWER_LINE_RE = re.compile(r"answer\s*[:\-]\s*([^\n\r.]+)", re.IGNORECASE)

# Lenient patterns: only used when the Answer: line is missing.
#
# CSQA: prefer the LAST standalone capital A-E in the prediction (the
# answer is conventionally emitted near the end of a chain). Searching
# from the end avoids picking up "A rug" / "A few" / "I" tokens in the
# reasoning prefix.
_CSQA_RE = re.compile(r"\b([A-E])\b")
_STRATEGY_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
_FEVER_RE = re.compile(r"\b(supports|refutes|nei|not[\s\-]?enough[\s\-]?info)\b",
                       re.IGNORECASE)


def _answer_line(pred: str) -> Optional[str]:
    """Return the substring after the last 'Answer:' line, or None."""
    if not pred:
        return None
    matches = list(_ANSWER_LINE_RE.finditer(pred))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _extract_csqa(pred: str) -> Optional[str]:
    # Prefer Answer: line. If present, extract the first A-E from it.
    ans_line = _answer_line(pred)
    if ans_line:
        m = _CSQA_RE.search(ans_line)
        if m:
            return m.group(1)
    # Fallback: take the LAST standalone A-E in the full prediction.
    matches = list(_CSQA_RE.finditer(pred or ""))
    return matches[-1].group(1) if matches else None


def _extract_strategy(pred: str) -> Optional[str]:
    ans_line = _answer_line(pred)
    candidate = None
    if ans_line:
        m = _STRATEGY_RE.search(ans_line)
        if m:
            candidate = m.group(1).lower()
    if candidate is None:
        # Fallback: last yes/no/true/false in the prediction.
        matches = list(_STRATEGY_RE.finditer(pred or ""))
        if matches:
            candidate = matches[-1].group(1).lower()
    if candidate is None:
        return None
    if candidate in {"yes", "true"}:
        return "yes"
    if candidate in {"no", "false"}:
        return "no"
    return None


def _extract_fever(pred: str) -> Optional[str]:
    ans_line = _answer_line(pred)
    m = None
    if ans_line:
        m = _FEVER_RE.search(ans_line)
    if m is None:
        # Fallback: last canonical token in the full prediction.
        matches = list(_FEVER_RE.finditer(pred or ""))
        if matches:
            m = matches[-1]
    if not m:
        return None
    tok = m.group(1).lower().replace("-", " ").replace("_", " ").strip()
    if tok == "supports":
        return "SUPPORTS"
    if tok == "refutes":
        return "REFUTES"
    if tok in {"nei", "not enough info"}:
        return "NOT ENOUGH INFO"
    return None


def _normalize_strategy_gold(gold: Any) -> Optional[str]:
    if gold is None:
        return None
    if isinstance(gold, bool):
        return "yes" if gold else "no"
    s = str(gold).strip().lower()
    if s in {"yes", "true", "1"}:
        return "yes"
    if s in {"no", "false", "0"}:
        return "no"
    return s or None


def _normalize_fever_gold(gold: Any) -> Optional[str]:
    if gold is None:
        return None
    s = str(gold).strip().upper().replace("-", " ").replace("_", " ")
    if s in {"SUPPORTS", "REFUTES"}:
        return s
    if "NOT ENOUGH INFO" in s or s == "NEI":
        return "NOT ENOUGH INFO"
    return s or None


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------

def _bench_of(path: Path) -> str:
    name = path.stem.lower()
    for b in ("commonsense_qa", "commonsenseqa", "csqa",
              "strategyqa", "strategy_qa",
              "fever",
              "triviaqa", "trivia_qa",
              "truthfulqa", "truthful_qa"):
        if b in name:
            return b
    return ""


def _gold_of(sample: Dict[str, Any]) -> Optional[str]:
    # CSQA / MCQ-style benchmarks store the letter in ``gold_label``.
    g = sample.get("gold_label")
    if g:
        return str(g).strip()
    gold_answers = sample.get("gold_answers")
    if isinstance(gold_answers, list) and gold_answers:
        return str(gold_answers[0]).strip()
    if "answer" in sample and sample["answer"]:
        return str(sample["answer"]).strip()
    return None


def _capability_em(sample: Dict[str, Any], bench: str) -> Optional[float]:
    """Return capability_em, or None if benchmark has no fallback."""
    pred = (sample.get("prediction")
            or sample.get("display_answer")
            or "")
    gold = _gold_of(sample)
    if not gold:
        return None

    bench = bench.lower()
    if "commonsense" in bench or bench == "csqa":
        extracted = _extract_csqa(pred)
        if extracted is None:
            return 0.0
        return 1.0 if extracted == gold.strip().upper() else 0.0

    if "strategy" in bench:
        extracted = _extract_strategy(pred)
        gold_n = _normalize_strategy_gold(gold)
        if extracted is None or gold_n is None:
            return 0.0
        return 1.0 if extracted == gold_n else 0.0

    if bench == "fever":
        extracted = _extract_fever(pred)
        gold_n = _normalize_fever_gold(gold)
        if extracted is None or gold_n is None:
            return 0.0
        return 1.0 if extracted == gold_n else 0.0

    # Open-ended benches: capability_em == strict em (no lenient fallback).
    return None


# ---------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------

def annotate_file(path: Path, *, force: bool = False) -> Tuple[int, int, int]:
    """Add ``capability_em`` to each sample in ``path`` (in place).

    Returns ``(n_total, n_added, n_matches)``.
    """
    bench = _bench_of(path)
    if not bench:
        return (0, 0, 0)

    with open(path) as f:
        doc = json.load(f)
    samples = doc.get("samples") or doc.get("results") or []
    if not samples:
        return (0, 0, 0)

    n_added = 0
    n_matches = 0
    for s in samples:
        if not force and "capability_em" in s and s["capability_em"] is not None:
            if isinstance(s["capability_em"], (int, float)) and s["capability_em"] >= 1.0:
                n_matches += 1
            continue
        cap = _capability_em(s, bench)
        if cap is None:
            # Open-ended bench: fall back to strict em so downstream
            # aggregators see a populated field for every sample.
            # 2026-05-16: for TruthfulQA, prefer em_llm_judged when populated
            # (legacy em on TQA is rouge_l > 0.15, which inflates long-prose
            # answers — Lin et al. ACL 2022 §3.2 methodology).
            if "truthfulqa" in bench and s.get("em_llm_judged") is not None:
                cap = float(s["em_llm_judged"])
            else:
                cap = float(s.get("em") or 0.0)
        s["capability_em"] = float(cap)
        n_added += 1
        if cap >= 1.0:
            n_matches += 1

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return (len(samples), n_added, n_matches)


def annotate_dir(root: Path, *, force: bool, label: str) -> None:
    if not root.is_dir():
        logger.warning("Skip missing dir: %s", root)
        return
    files = sorted(root.rglob("*.json"))
    files = [f for f in files
             if _bench_of(f)
             and "with_chm" not in f.stem
             and not f.parent.name.startswith(".")]
    if not files:
        logger.info("%s: no eval JSONs found", label)
        return

    print(f"\n=== {label} ({root}) ===")
    print(f"{'file':<70} {'n':>5} {'strict_em':>10} {'cap_em':>8} {'Δ':>6}")
    print("-" * 105)
    for p in files:
        with open(p) as f:
            doc = json.load(f)
        samples = doc.get("samples") or doc.get("results") or []
        strict = sum(float(s.get("em") or 0.0) for s in samples) / max(len(samples), 1)
        n_total, n_added, n_matches = annotate_file(p, force=force)
        if n_total == 0:
            continue
        cap = n_matches / n_total
        delta = cap - strict
        rel = str(p.relative_to(root))
        marker = " *" if delta > 0.05 else ""
        print(f"{rel:<70} {n_total:>5} {strict:>10.3f} {cap:>8.3f} "
              f"{delta:>+6.3f}{marker}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baselines_dir", type=Path,
                   default=Path("outputs/baselines"))
    p.add_argument("--caem_dir", type=Path,
                   default=Path("outputs/full_run/eval"))
    p.add_argument("--force", action="store_true",
                   help="Recompute capability_em even when already present.")
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(level=ns.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    annotate_dir(ns.baselines_dir, force=ns.force, label="BASELINES")
    annotate_dir(ns.caem_dir, force=ns.force, label="CAEM")
    print("\nLegend: ' *' = capability_em > strict_em by >5 pp (format-collapse signal)")


if __name__ == "__main__":
    main()
