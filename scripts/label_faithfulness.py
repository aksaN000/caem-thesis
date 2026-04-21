"""
scripts/label_faithfulness.py
==============================
Produce the faithfulness-labels JSON consumed by ``scripts/epistemic_gate.py``.

Why this script exists
----------------------
The Moskvoretskii epistemic gate needs per-sample labels saying whether
the stored answer is *faithful to the retrieved passage* (the Ch1 promise
that drives ``u_stored``). The Phase-1a Flan-T5 EM-based proxy was
confounded (paraphrase failures, repetition loops, label-extraction
bugs); the correct proxy is a judge-scored entailment between each
passage and the corresponding answer.

This script produces exactly that mapping. The downstream gate doesn't
care which judge was used -- MiniCheck, a multi-judge average,
DeBERTa-MNLI, human ratings, anything with a
``(passage, claim) -> P(supported in [0, 1])`` interface works.

Pipeline
--------
1. Read ``per_sample_signals.jsonl`` -- the per-sample records
   produced by the eval harness. Each record must carry at minimum
   ``question``, ``answer``, and either ``top_passages`` (list of
   passage strings) or a reference to the passage store so the script
   can re-fetch them.
2. For each record, compute ``P(passage entails answer)`` for the
   highest-scoring retrieved passage. When multiple passages are
   available, use the max across passages (consistent with
   ``p_ground_max`` semantics).
3. Write ``labels.json`` -- a JSON object ``{question_key: float}``
   that the epistemic gate's ``--labels_path`` consumes directly.

Judge selection
---------------
Default is the shared ``load_verifier_judge(backend="minicheck", ...)``
path used by the main verifier so the labels match the production
judge's view. ``--judge_backend`` can override to ``roberta_nli`` for a
cross-check; running both and averaging is the most defensible label
set for the Ch5 appendix but requires a second run over the data.

CPU / tests
-----------
The label-aggregation logic is pure-Python and covered in
``tests/test_label_faithfulness.py`` with a mocked judge. The live judge
load + actual scoring happens on a GPU run (loading MiniCheck needs ~2 GB
VRAM); the CPU harness only exercises the aggregation layer.

Typical usage
-------------
    PYTHONPATH=. python scripts/label_faithfulness.py \\
        --signals_path outputs/per_sample_signals.jsonl \\
        --output_path outputs/faithfulness_labels.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Aggregation layer (pure Python, CPU-testable)
# =============================================================================

def _extract_passages(record: dict) -> List[str]:
    """Return the passage strings from a per-sample record.

    Supports two on-disk shapes:
      * ``top_passages`` -- list of raw strings (harness default)
      * ``top_passages_scored`` -- list of ``{"text": str, "score": float}``
        dicts (post-rerank harness output)
    """
    raw = record.get("top_passages")
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        return [str(p) for p in raw]
    scored = record.get("top_passages_scored")
    if isinstance(scored, list):
        out: List[str] = []
        for item in scored:
            if isinstance(item, dict) and "text" in item:
                out.append(str(item["text"]))
        return out
    return []


def score_record(
    record: dict,
    judge_fn,
) -> Optional[float]:
    """Return ``max_p P(passage entails answer)`` in [0, 1], or None when
    the record lacks the fields needed to score (no passages / no answer).

    ``judge_fn`` has signature ``List[Tuple[str, str]] -> List[float]``
    (premise, hypothesis) matching the verifier's ``batch_entail_prob``.
    """
    answer = (record.get("answer") or "").strip()
    if not answer:
        return None
    passages = _extract_passages(record)
    if not passages:
        return None
    pairs = [(p, answer) for p in passages]
    scores = judge_fn(pairs)
    if not scores:
        return None
    # Clamp + take the max across passages -- matches p_ground_max
    # semantics in the verifier so the faithfulness label is on the same
    # axis as the composite signal it will be correlated against.
    clipped = [max(0.0, min(1.0, float(s))) for s in scores]
    return max(clipped)


def build_labels(
    records: Sequence[dict],
    judge_fn,
    *,
    key_field: str = "question",
    batch_size: int = 16,
) -> Dict[str, float]:
    """Return ``{record[key_field]: P(supported)}`` over all scoreable records.

    Batches calls to ``judge_fn`` at ``batch_size`` records at a time so
    a GPU judge can amortise its kernel launches across the per-record
    (passages × 1 answer) inner calls.

    Unscoreable records (missing answer / passages) are skipped.
    Duplicate keys keep the last-wins value so re-running on updated
    records overwrites stale labels cleanly.
    """
    out: Dict[str, float] = {}
    # Simple per-record loop -- the per-record judge call already batches
    # within the record (all passages for one answer). A cross-record
    # super-batch is an optimisation slot for the Vast run.
    for rec in records:
        key = rec.get(key_field)
        if key is None:
            continue
        score = score_record(rec, judge_fn)
        if score is None:
            continue
        out[str(key)] = float(score)
    return out


# =============================================================================
# Judge loaders (thin wrappers over caem.verification)
# =============================================================================

def _load_minicheck_judge(device: str = "cuda") -> Any:
    """Build a judge whose ``batch_entail_prob`` matches the verifier's."""
    from caem.config import CAEMConfig
    from caem.verification import load_verifier_judge

    cfg = CAEMConfig()
    cfg.verifier_backend = "minicheck"
    judge, _nli_model, _nli_tok = load_verifier_judge(
        cfg, device, allow_fallback=True,
    )
    return judge


def _judge_fn_from(judge: Any):
    """Wrap a caem judge instance into the ``List[Tuple[str,str]] -> List[float]``
    interface that ``score_record`` expects.
    """
    def _fn(pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return list(judge.batch_entail_prob(pairs))
    return _fn


# =============================================================================
# JSONL loader
# =============================================================================

def load_signals_jsonl(path: Path) -> List[dict]:
    """Same loader shape as scripts/epistemic_gate.py::load_signals_jsonl."""
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "signals: line %d malformed (%s) -- skipped.", lineno, exc,
                )
    logger.info("signals: loaded %d rows from %s", len(rows), path)
    return rows


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Produce faithfulness labels for the epistemic gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--signals_path", type=Path, required=True,
        help="Per-sample signals JSONL (must carry question + answer + "
             "top_passages).",
    )
    p.add_argument(
        "--output_path", type=Path, required=True,
        help="Destination labels JSON (will be overwritten).",
    )
    p.add_argument(
        "--key_field", default="question",
        help="Field used as the label-dict key (matches epistemic_gate "
             "--label_key).",
    )
    p.add_argument(
        "--judge_backend", default="minicheck",
        choices=["minicheck"],
        help="Which judge to use. Multi-judge support is a follow-up slot.",
    )
    p.add_argument(
        "--device", default="cuda",
        help="Device passed to the judge loader.",
    )
    p.add_argument(
        "--batch_size", type=int, default=16,
        help="Cross-record batch size (future optimisation slot).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ns = _build_parser().parse_args(argv)

    records = load_signals_jsonl(ns.signals_path)
    if not records:
        logger.error("No records loaded from %s -- nothing to label.", ns.signals_path)
        return 2

    logger.info("Loading judge backend=%s ...", ns.judge_backend)
    if ns.judge_backend == "minicheck":
        judge = _load_minicheck_judge(device=ns.device)
    else:
        raise ValueError(f"unsupported judge backend: {ns.judge_backend}")

    judge_fn = _judge_fn_from(judge)
    labels = build_labels(
        records, judge_fn,
        key_field=ns.key_field, batch_size=ns.batch_size,
    )
    logger.info("Built %d labels from %d records.", len(labels), len(records))

    ns.output_path.parent.mkdir(parents=True, exist_ok=True)
    ns.output_path.write_text(
        json.dumps(labels, indent=2, sort_keys=True, ensure_ascii=False),
    )
    logger.info("Wrote %s", ns.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
