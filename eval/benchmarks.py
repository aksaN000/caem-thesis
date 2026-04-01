"""
eval/benchmarks.py
==================
Dataset loaders for the four CAEM evaluation benchmarks.

Benchmarks
----------
HotpotQA   (Yang et al. 2018)    — multi-hop QA; EM + F1
TruthfulQA (Lin et al. 2022)     — factual QA; any-match EM
FEVER      (Thorne et al. 2018)  — fact verification; label accuracy
StrategyQA (Geva et al. 2021)    — implicit multi-hop boolean QA; EM

Loading strategy
----------------
Each loader returns a list of BenchmarkSample dicts — a simple common
schema that the EvalHarness consumes without knowing which benchmark
it came from.

Data source: HuggingFace `datasets` library (lazy-loaded — not imported
at module level so tests can mock it without importing it).  When the
`datasets` package is unavailable, loaders raise ImportError.

On coverage gaps (Q3 in thesis framing)
----------------------------------------
The Wikipedia Dec-2018 corpus (~21M passages) does NOT cover every
question. When retrieval finds irrelevant passages:
  - The model may produce a low-confidence or hallucinated answer.
  - The verifier scores it low → not stored.
  - EM is 0 for that sample.
This is expected and is precisely what CAEM's self-improvement loop is
designed to mitigate over successive cycles. See: thesis §5.3 "Retrieval
Failure Analysis".

Sample schema
-------------
Each sample is a dict with:
  question   : str
  answers    : list[str]   — acceptable answer strings (may have 1 entry)
  gold_label : str | None  — FEVER label ("supports"/"refutes"/"not enough info")
  id         : str         — original dataset ID for traceability
  benchmark  : str         — "hotpotqa" | "truthfulqa" | "fever" | "strategyqa"

FEVER prompt design note
------------------------
FEVER uses a CONSTRAINED prompt that explicitly enumerates the three valid
labels. This prevents Flan-T5 from producing free-form answers that are hard
to parse and biases the output toward the three canonical label strings,
making extract_fever_label() reliable without complex heuristics.

  Prompt: "Answer with one of: supports, refutes, not enough info.
           Claim: {claim}"

This is consistent with how instruction-tuned models are evaluated on
classification tasks in the FLAN literature (Wei et al. 2022).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Type alias
BenchmarkSample = Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# HotpotQA
# ─────────────────────────────────────────────────────────────────────────────

def load_hotpotqa(
    split: str = "validation",
    n: Optional[int] = None,
    seed: int = 42,
    difficulty: str = "all",
) -> List[BenchmarkSample]:
    """Load HotpotQA samples from HuggingFace datasets.

    Parameters
    ----------
    split : str
        "validation" (7405 samples) or "train" (90k samples).
        We evaluate on validation to keep numbers comparable to baselines.
    n : int or None
        Number of samples to return. If None, return all. If n > available,
        return all available samples (no error).
    seed : int
        Random seed for reproducible subsampling.
    difficulty : str
        "easy", "medium", "hard", or "all" (default).
        HotpotQA provides a "level" field; filtering on it is useful for
        ablation. "all" disables the filter.

    Returns
    -------
    list of BenchmarkSample
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace `datasets` is required for HotpotQA loading. "
            "Install with: pip install datasets"
        ) from e

    logger.info("Loading HotpotQA [%s] from HuggingFace…", split)
    ds = load_dataset("hotpot_qa", "distractor", split=split)

    samples: List[BenchmarkSample] = []
    for row in ds:
        if difficulty != "all" and row.get("level", "").lower() != difficulty.lower():
            continue
        samples.append({
            "question": row["question"],
            "answers": [row["answer"]],   # HotpotQA has exactly one gold answer
            "gold_label": None,
            "id": row["id"],
            "benchmark": "hotpotqa",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    logger.info("HotpotQA: %d samples loaded (%s split).", len(samples), split)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# TruthfulQA
# ─────────────────────────────────────────────────────────────────────────────

def load_truthfulqa(
    n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkSample]:
    """Load TruthfulQA generation-format questions.

    TruthfulQA provides multiple acceptable correct answers per question AND
    a list of incorrect answers. We use the 'generation' configuration which
    provides free-form answers (not the MC format).

    Metric: any-match EM — correct if the prediction matches ANY answer
    in the `correct_answers` list.

    Parameters
    ----------
    n : int or None — number of questions to sample. Dataset has 817 total.
    seed : int

    Returns
    -------
    list of BenchmarkSample
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace `datasets` is required for TruthfulQA loading."
        ) from e

    logger.info("Loading TruthfulQA [generation] from HuggingFace…")
    ds = load_dataset("truthful_qa", "generation", split="validation")

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(ds):
        correct = row.get("correct_answers", [])
        if not correct:
            continue
        samples.append({
            "question": row["question"],
            "answers": correct,
            "gold_label": None,
            "id": str(i),
            "benchmark": "truthfulqa",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    logger.info("TruthfulQA: %d samples loaded.", len(samples))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# FEVER
# ─────────────────────────────────────────────────────────────────────────────

# Map HuggingFace label integers to canonical string labels.
_FEVER_LABEL_MAP = {
    0: "supports",
    1: "refutes",
    2: "not enough info",
    # Some versions of the dataset use string keys directly.
    "SUPPORTS": "supports",
    "REFUTES": "refutes",
    "NOT ENOUGH INFO": "not enough info",
}


def load_fever(
    split: str = "dev",
    n: Optional[int] = None,
    seed: int = 42,
    exclude_nei: bool = False,
) -> List[BenchmarkSample]:
    """Load FEVER claim-verification samples.

    The FEVER task: given a claim, predict whether it is SUPPORTS, REFUTES,
    or NOT ENOUGH INFO.

    Dataset note (EXP-07 fix 2026-04-01):
    The original HuggingFace 'fever'/'v1.0' dataset uses a legacy Python
    dataset script (fever.py) which is no longer supported by the current
    HuggingFace datasets library (raises RuntimeError at load time).
    We use 'lucadiliello/fever' which is a Parquet-based mirror of the same
    FEVER dataset with identical content. Splits: 'train', 'test', 'dev'.
    We use 'dev' (~19,998 samples) = the original paper_dev split.
    Label schema: integer {0: supports, 1: refutes, 2: not enough info}.

    Constrained prompt design: the prompt explicitly enumerates the three
    valid labels so Flan-T5 produces clean, parseable outputs:

      Prompt: "Answer with one of: supports, refutes, not enough info.
               Claim: {claim}"

    Parameters
    ----------
    split : str
        "dev" (~19,998 samples, default) or "train" or "test".
    n : int or None
    seed : int
    exclude_nei : bool
        If True, drop NOT ENOUGH INFO samples (binary classification only).
        False by default (full 3-class evaluation).

    Returns
    -------
    list of BenchmarkSample
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace `datasets` is required for FEVER loading."
        ) from e

    logger.info("Loading FEVER [lucadiliello/fever, %s] from HuggingFace...", split)
    ds = load_dataset("lucadiliello/fever", split=split)

    samples: List[BenchmarkSample] = []
    for row in ds:
        raw_label = row.get("label", 2)
        gold_label = _FEVER_LABEL_MAP.get(raw_label, "not enough info")

        if exclude_nei and gold_label == "not enough info":
            continue

        claim = row.get("claim", "")
        question = (
            f"Answer with one of: supports, refutes, not enough info. "
            f"Claim: {claim}"
        )

        samples.append({
            "question": question,
            "answers": [gold_label],
            "gold_label": gold_label,
            "id": str(row.get("key", row.get("id", ""))),
            "benchmark": "fever",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    logger.info("FEVER: %d samples loaded (%s split).", len(samples), split)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# StrategyQA
# ─────────────────────────────────────────────────────────────────────────────

def load_strategyqa(
    n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkSample]:
    """Load StrategyQA samples (Geva et al. 2021).

    StrategyQA tests implicit multi-hop reasoning: questions require
    decomposing a query into sub-questions without any explicit reasoning
    chain in the question text. Answers are boolean (yes/no).

    Dataset note (IQ-03 resolution)
    --------------------------------
    The original StrategyQA repository is no longer maintained. We use the
    HuggingFace `wics/strategy-qa` dataset. The task is identical: implicit
    strategy-based boolean QA.

    Split choice (IQ-03b fix):
    The `test` split of wics/strategy-qa has only ~490 questions, which is
    too small to survive the 500+500 purity+calibration allocation and still
    leave any samples for accuracy evaluation. We therefore use the `train`
    split (~2,290 questions, matching the thesis plan §5.3 table), which
    contains labelled boolean answers. This is standard practice when the
    official test split has no public labels — we treat this split as the
    held-out evaluation set (it is never used for model fine-tuning; CAEM's
    SIL loop trains only on its own verified generations, not on dataset
    labels). State explicitly in §5.3: "StrategyQA evaluation uses the
    wics/strategy-qa train split (~2,290 questions) because the test split
    provides no public ground-truth labels."

    Prompt format: "Answer yes or no. Question: {question}"
    This is consistent with instruction-tuning evaluation of boolean tasks
    (Wei et al. 2022) and makes extract_strategyqa_label() reliable.

    Metric: EM (exact match on "yes" / "no" after normalisation).

    Parameters
    ----------
    n : int or None — number of questions to sample. Train split has ~2290.
    seed : int

    Returns
    -------
    list of BenchmarkSample
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace `datasets` is required for StrategyQA loading."
        ) from e

    # EXP-08 fix (2026-04-01): 'wics/strategy-qa' uses a legacy dataset script
    # no longer supported by HF datasets. Load the official StrategyQA JSON
    # directly from GitHub (same data, same schema: qid, question, answer bool).
    _SQA_URL = "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json"
    try:
        logger.info("Loading StrategyQA from official GitHub JSON...")
        import json as _json, urllib.request as _ur
        with _ur.urlopen(_SQA_URL, timeout=30) as resp:
            sqa_data = _json.loads(resp.read().decode())
        # sqa_data is a list of dicts with keys: qid, question, answer (bool), facts, decomposition
        ds = sqa_data
    except Exception as exc:
        logger.warning("StrategyQA GitHub load failed (%s) -- trying HF fallback.", exc)
        try:
            ds_hf = load_dataset("json",
                data_files={"train": _SQA_URL},
            )
            ds = list(ds_hf["train"])
        except Exception as exc2:
            raise RuntimeError(f"Could not load StrategyQA from any source: {exc2}") from exc2

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(ds):
        answer_bool = row.get("answer", None)
        if answer_bool is None:
            continue
        gold = "yes" if answer_bool else "no"
        question_text = row.get("question", "")
        # Constrained boolean prompt.
        question = f"Answer yes or no. Question: {question_text}"
        samples.append({
            "question": question,
            "answers": [gold],
            "gold_label": gold,
            "id": str(row.get("id", i)),
            "benchmark": "strategyqa",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    logger.info("StrategyQA: %d samples loaded.", len(samples))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Unified loader
# ─────────────────────────────────────────────────────────────────────────────

def load_benchmark(
    name: str,
    n: Optional[int] = None,
    seed: int = 42,
    **kwargs,
) -> List[BenchmarkSample]:
    """Load any supported benchmark by name.

    Parameters
    ----------
    name : str — "hotpotqa", "truthfulqa", "fever", or "strategyqa"
    n : int or None
    seed : int
    **kwargs — passed to the specific loader

    Returns
    -------
    list of BenchmarkSample
    """
    name = name.lower().strip()
    if name == "hotpotqa":
        return load_hotpotqa(n=n, seed=seed, **kwargs)
    elif name == "truthfulqa":
        return load_truthfulqa(n=n, seed=seed, **kwargs)
    elif name == "fever":
        return load_fever(n=n, seed=seed, **kwargs)
    elif name == "strategyqa":
        return load_strategyqa(n=n, seed=seed, **kwargs)
    else:
        raise ValueError(
            f"Unknown benchmark '{name}'. "
            "Supported: 'hotpotqa', 'truthfulqa', 'fever', 'strategyqa'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data factory (for tests and dry-runs without downloading datasets)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_samples(
    benchmark: str,
    n: int = 10,
    seed: int = 0,
) -> List[BenchmarkSample]:
    """Generate reproducible synthetic BenchmarkSample dicts.

    Used in unit tests and quick smoke-tests. No network access required.

    Parameters
    ----------
    benchmark : str — "hotpotqa", "truthfulqa", "fever", or "strategyqa"
    n : int — number of samples to generate
    seed : int

    Returns
    -------
    list of BenchmarkSample
    """
    rng = random.Random(seed)
    samples = []

    if benchmark == "hotpotqa":
        for i in range(n):
            samples.append({
                "question": f"What connects entity_{i} and entity_{i+1}?",
                "answers": [f"answer_{i}"],
                "gold_label": None,
                "id": f"hotpot_synth_{i}",
                "benchmark": "hotpotqa",
            })

    elif benchmark == "truthfulqa":
        for i in range(n):
            samples.append({
                "question": f"Is claim_{i} factually correct?",
                "answers": [f"yes_{i}", f"correct_{i}"],
                "gold_label": None,
                "id": f"tqa_synth_{i}",
                "benchmark": "truthfulqa",
            })

    elif benchmark == "fever":
        labels = ["supports", "refutes", "not enough info"]
        for i in range(n):
            label = labels[i % 3]
            samples.append({
                # Use the constrained prompt format (matches load_fever).
                "question": f"Answer with one of: supports, refutes, not enough info. Claim: claim_{i}",
                "answers": [label],
                "gold_label": label,
                "id": f"fever_synth_{i}",
                "benchmark": "fever",
            })

    elif benchmark == "strategyqa":
        labels = ["yes", "no"]
        for i in range(n):
            label = labels[i % 2]
            samples.append({
                # Use the constrained boolean prompt (matches load_strategyqa).
                "question": f"Answer yes or no. Question: Is entity_{i} related to entity_{i+1}?",
                "answers": [label],
                "gold_label": label,
                "id": f"sqa_synth_{i}",
                "benchmark": "strategyqa",
            })

    else:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            "Supported: 'hotpotqa', 'truthfulqa', 'fever', 'strategyqa'."
        )

    return samples
