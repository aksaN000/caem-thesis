#!/usr/bin/env python3
"""scripts/ruc_load_extra_benchmarks.py
==========================================
Loaders for the two new benchmarks added in RUC v2 to compensate for
sample shortage on the existing 5-benchmark panel:

- HaluEval-QA  — myth/hallucination benchmark, ~10k QA pairs, fills the
                 TruthfulQA shortage (TruthfulQA has only 17 raw spare).
- OpenBookQA   — science commonsense MCQ, ~5,957 samples, fills the
                 CommonsenseQA tightness (CSQA has only ~300 raw spare).

Both loaders mirror eval/benchmarks.py output schema:

    {
        "question": str,
        "answers":  List[str],     # acceptable gold answers
        "gold_label": str,         # canonical gold (first answer or letter)
        "id": str,
        "benchmark": str,
    }

So they can flow through the same baseline harness.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# HaluEval-QA
# ---------------------------------------------------------------------------

def load_haluevalqa(
    n: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load HaluEval-QA samples from HuggingFace.

    HaluEval (Yejin Choi et al. 2023) tests retrieval-induced hallucination.
    The QA subset has 10k questions with both a 'right_answer' and a
    'hallucinated_answer'. We map them to the BenchmarkSample schema:

        question      = the question text
        answers       = [right_answer]
        gold_label    = right_answer
        id            = HaluEval's auto id
        benchmark     = "haluevalqa"

    The 'hallucinated_answer' field is NOT used as a baseline output — it's
    a HaluEval-specific dataset feature we ignore. The baselines generate
    their own answers from Qwen+RAG, and the right_answer serves as gold.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace `datasets` required") from e

    # HaluEval QA subset (10k QA pairs). Try a few host/config combos
    # in decreasing order of expected reliability.
    candidates = [
        ("pminervini/HaluEval", "qa", "data"),     # canonical post-2024 hosting
        ("pminervini/HaluEval", "qa", "train"),
        ("notrichardren/HaluEval", "qa", "train"),
        ("pminervini/HaluEval", None, "train"),
    ]
    ds = None
    for repo, config, split in candidates:
        try:
            logger.info("Trying HaluEval at %s / config=%s / split=%s ...",
                        repo, config, split)
            if config is None:
                ds = load_dataset(repo, split=split)
            else:
                ds = load_dataset(repo, name=config, split=split)
            logger.info("  loaded %d rows from %s", len(ds), repo)
            break
        except Exception as e:
            logger.warning("  failed: %s", str(e)[:120])
            ds = None
    if ds is None:
        raise SystemExit(
            "Could not load HaluEval from any known HuggingFace host."
        )

    samples: List[Dict[str, Any]] = []
    for i, row in enumerate(cast(Any, ds)):
        row = cast(Dict[str, Any], row)
        question = (row.get("question") or row.get("knowledge_question")
                    or row.get("query") or "")
        right = (row.get("right_answer") or row.get("answer") or "")
        if not question or not right:
            continue
        sample_id = str(row.get("id") or row.get("knowledge_id") or f"hal_{i}")
        samples.append({
            "question": question,
            "answers": [right] if isinstance(right, str) else list(right),
            "gold_label": right if isinstance(right, str) else (right[0] if right else ""),
            "id": sample_id,
            "benchmark": "haluevalqa",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    logger.info("HaluEval-QA: loaded %d samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# OpenBookQA
# ---------------------------------------------------------------------------

def load_openbookqa(
    split: str = "train",
    n: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load OpenBookQA samples from HuggingFace.

    OpenBookQA (Mihaylov et al. 2018) is 5,957 multiple-choice science
    commonsense questions, 4 candidates per question, 1 correct.

    Mirrors ARC-Challenge's loader shape so the same baseline harness
    (which expects ``"Question: ... Choices: (A) ... (B) ... → Answer with
    the letter."`` style prompts) handles it without modification.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace `datasets` required") from e

    logger.info("Loading OpenBookQA [%s] from HuggingFace ...", split)
    ds = load_dataset("allenai/openbookqa", name="main", split=split)

    samples: List[Dict[str, Any]] = []
    for i, row in enumerate(cast(Any, ds)):
        row = cast(Dict[str, Any], row)
        stem = row["question_stem"]
        choices = row.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        correct_label = row.get("answerKey", "")
        choices_text = " ".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
        question = (
            f"Question: {stem} Choices: {choices_text}\n"
            f"Answer with just the multiple choice letter."
        )
        samples.append({
            "question": question,
            "answers": [correct_label],
            "gold_label": correct_label,
            "id": row.get("id", str(i)),
            "benchmark": "openbookqa",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    logger.info("OpenBookQA: loaded %d samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print("\n=== HaluEval-QA self-test (first 3) ===")
    try:
        hal = load_haluevalqa(n=3)
        for s in hal:
            print(f"  {s['id']}: {s['question'][:80]!r}")
            print(f"    gold: {s['gold_label'][:80]!r}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n=== OpenBookQA self-test (first 3) ===")
    try:
        obq = load_openbookqa(n=3)
        for s in obq:
            print(f"  {s['id']}: {s['question'][:120]!r}")
            print(f"    gold: {s['gold_label']!r}")
    except Exception as e:
        print(f"  FAILED: {e}")
