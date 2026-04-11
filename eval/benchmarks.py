"""
eval/benchmarks.py
==================
Dataset loaders for CAEM training/evaluation benchmarks.

Benchmarks
----------
FEVER              (Thorne et al. 2018)  -- fact verification; label accuracy
TriviaQA           (Joshi et al. 2017)   -- open-domain QA; EM/F1
Natural Questions  (Kwiatkowski et al. 2019) -- open-domain QA; EM/F1
TruthfulQA         (Lin et al. 2022)     -- factual QA; ROUGE-L thresholded EM proxy
StrategyQA         (Geva et al. 2021)    -- implicit multi-hop boolean QA; EM
ARC-Challenge      (Clark et al. 2018)   -- multiple-choice science QA; label EM
HotpotQA           (Yang et al. 2018)    -- optional legacy multi-hop QA loader

Loading strategy
----------------
Each loader returns a list of BenchmarkSample dicts -- a simple common
schema that the EvalHarness consumes without knowing which benchmark
it came from.

Data source: HuggingFace `datasets` library (lazy-loaded -- not imported
at module level so tests can mock it without importing it).  When the
`datasets` package is unavailable, loaders raise ImportError.

On coverage gaps (Q3 in thesis framing)
----------------------------------------
The Wikipedia Dec-2018 corpus (~21M passages) does NOT cover every
question. When retrieval finds irrelevant passages:
  - The model may produce a low-confidence or hallucinated answer.
  - The verifier scores it low -> not stored.
  - EM is 0 for that sample.
This is expected and is precisely what CAEM's self-improvement loop is
designed to mitigate over successive cycles. See: thesis §5.3 "Retrieval
Failure Analysis".

Sample schema
-------------
Each sample is a dict with:
  question   : str
  answers    : list[str]   -- acceptable answer strings (may have 1 entry)
  gold_label : str | None  -- FEVER label ("supports"/"refutes"/"not enough info")
  id         : str         -- original dataset ID for traceability
    benchmark  : str         -- benchmark key such as
                                                            "fever", "triviaqa", "natural_questions",
                                                            "truthfulqa", "strategyqa", "arc_challenge",
                                                            or "hotpotqa"

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
from typing import Any, Dict, Iterator, List, Optional, cast

logger = logging.getLogger(__name__)

# Type alias
BenchmarkSample = Dict[str, Any]


# -----------------------------------------------------------------------------
# HotpotQA
# -----------------------------------------------------------------------------

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

    logger.info("Loading HotpotQA [%s] from HuggingFace...", split)
    ds = load_dataset("hotpot_qa", "distractor", split=split)

    samples: List[BenchmarkSample] = []
    for row in cast(Any, ds):
        row = cast(Dict[str, Any], row)
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


# -----------------------------------------------------------------------------
# TruthfulQA
# -----------------------------------------------------------------------------

def load_truthfulqa(
    n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkSample]:
    """Load TruthfulQA generation-format questions.

    TruthfulQA provides multiple acceptable correct answers per question AND
    a list of incorrect answers. We use the 'generation' configuration which
    provides free-form answers (not the MC format).

    Metric: any-match EM -- correct if the prediction matches ANY answer
    in the `correct_answers` list.

    Parameters
    ----------
    n : int or None -- number of questions to sample. Dataset has 817 total.
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

    logger.info("Loading TruthfulQA [generation] from HuggingFace...")
    ds = load_dataset("truthful_qa", "generation", split="validation")

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(cast(Any, ds)):
        row = cast(Dict[str, Any], row)
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


# -----------------------------------------------------------------------------
# FEVER
# -----------------------------------------------------------------------------

# Map HuggingFace label integers to canonical string labels.
# lucadiliello/fever label schema (verified against dataset card):
#   0 -> SUPPORTS
#   1 -> NOT ENOUGH INFO
#   2 -> REFUTES
# NOTE: this differs from naive alphabetical ordering. Using the wrong mapping
# swaps refutes and NEI gold labels, corrupting both training and evaluation.
_FEVER_LABEL_MAP = {
    0: "supports",
    1: "not enough info",
    2: "refutes",
    # Some versions of the dataset use string keys directly.
    "SUPPORTS": "supports",
    "REFUTES": "refutes",
    "NOT ENOUGH INFO": "not enough info",
}


def load_fever(
    split: str = "paper_dev",
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
    FEVER dataset with identical content. Splits: 'train', 'paper_dev', 'paper_test'.
    We use 'paper_dev' (~19,998 samples) = the original FEVER paper evaluation split.
    Label schema per dataset card: integer {0: supports, 1: not enough info, 2: refutes}.

    Constrained prompt design: the prompt explicitly enumerates the three
    valid labels so Flan-T5 produces clean, parseable outputs:

      Prompt: "Answer with one of: supports, refutes, not enough info.
               Claim: {claim}"

    Parameters
    ----------
    split : str
        "paper_dev" (~19,998 samples, default -- standard eval split) or
        "train" (145K samples, for SIL pool) or "paper_test" (unlabeled).
        Use "paper_dev" for all evaluation to avoid data leakage from train.
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
    for row in cast(Any, ds):
        row = cast(Dict[str, Any], row)
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


# -----------------------------------------------------------------------------
# StrategyQA
# -----------------------------------------------------------------------------

def load_strategyqa(
    split: str = "test",
    n: Optional[int] = None,
    seed: int = 42,
    allow_train_fallback: bool = False,
) -> List[BenchmarkSample]:
    """Load StrategyQA samples (Geva et al. 2021).

    StrategyQA tests implicit multi-hop reasoning: questions require
    decomposing a query into sub-questions without any explicit reasoning
    chain in the question text. Answers are boolean (yes/no).

    Split behavior
    --------------
    Plan-default is `test` (transfer eval) and strict by default.
    We first try HuggingFace `wics/strategy-qa` for the requested split.
    If unavailable or unlabeled, we raise an error unless
    `allow_train_fallback=True` is explicitly requested.

    Prompt format: "Answer yes or no. Question: {question}"
    This is consistent with instruction-tuning evaluation of boolean tasks
    (Wei et al. 2022) and makes extract_strategyqa_label() reliable.

    Metric: EM (exact match on "yes" / "no" after normalisation).

    Parameters
    ----------
    split : str
        Requested split, usually "test" for transfer evaluation.
    n : int or None -- number of questions to sample.
    seed : int
    allow_train_fallback : bool
        If True, allows fallback to official train.json when the requested
        split is unavailable or unlabeled. Keep False for strict transfer
        evaluation compliance.

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

    split = split.strip().lower()

    def _normalise_rows(rows: List[Dict[str, Any]], source_name: str) -> List[BenchmarkSample]:
        samples_local: List[BenchmarkSample] = []
        for i, row in enumerate(rows):
            answer_bool = row.get("answer", None)
            if answer_bool is None:
                continue
            gold = "yes" if bool(answer_bool) else "no"
            question_text = row.get("question", "")
            question = f"Answer yes or no. Question: {question_text}"
            samples_local.append({
                "question": question,
                "answers": [gold],
                "gold_label": gold,
                "id": str(row.get("id", row.get("qid", i))),
                "benchmark": "strategyqa",
            })
        if samples_local:
            logger.info("StrategyQA: %d samples loaded from %s.", len(samples_local), source_name)
        return samples_local

    samples: List[BenchmarkSample] = []

    # Attempt 1: requested HF split.
    try:
        logger.info("Loading StrategyQA [wics/strategy-qa, %s] from HuggingFace...", split)
        ds_hf = load_dataset("wics/strategy-qa", split=split)
        hf_rows = [cast(Dict[str, Any], r) for r in cast(Any, ds_hf)]
        samples = _normalise_rows(hf_rows, f"wics/strategy-qa:{split}")
    except Exception as exc:
        logger.warning("StrategyQA HF split '%s' unavailable (%s).", split, exc)

    # Attempt 2 (optional): train JSON fallback only when explicitly enabled.
    if not samples:
        if allow_train_fallback:
            _SQA_URL = "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json"
            try:
                logger.warning(
                    "Falling back to official StrategyQA train.json (requested split=%s).",
                    split,
                )
                import json as _json
                import urllib.request as _ur
                with _ur.urlopen(_SQA_URL, timeout=30) as resp:
                    sqa_data = _json.loads(resp.read().decode())
                samples = _normalise_rows(cast(List[Dict[str, Any]], sqa_data), "official train.json")
            except Exception as exc:
                raise RuntimeError(f"Could not load StrategyQA from any source: {exc}") from exc
        else:
            raise RuntimeError(
                "StrategyQA split '%s' unavailable or unlabeled, and "
                "allow_train_fallback=False. "
                "Use a valid labeled split or call with allow_train_fallback=True "
                "for non-thesis exploratory runs." % split
            )

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    return samples

# -----------------------------------------------------------------------------
# TriviaQA
# -----------------------------------------------------------------------------

def load_triviaqa(
    split: str = "validation",
    n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkSample]:
    """Load TriviaQA rc.nocontext samples."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace `datasets` is required for TriviaQA loading.") from e

    logger.info("Loading TriviaQA (rc.nocontext) [%s] from HuggingFace...", split)
    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(cast(Any, ds)):
        row = cast(Dict[str, Any], row)
        ans_dict = row.get("answer", {})
        aliases = ans_dict.get("aliases", [])
        norm_val = ans_dict.get("normalized_value", "")
        answers = aliases if aliases else [norm_val]
        
        samples.append({
            "question": row["question"],
            "answers": answers,
            "gold_label": None,
            "id": row.get("question_id", str(i)),
            "benchmark": "triviaqa",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    return samples

# -----------------------------------------------------------------------------
# Natural Questions
# -----------------------------------------------------------------------------

def load_natural_questions(
    split: str = "validation",
    n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkSample]:
    """Load NQ-Open samples."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace `datasets` is required for NQ loading.") from e

    logger.info("Loading Natural Questions (NQ-Open) [%s] from HuggingFace...", split)
    ds = load_dataset("nq_open", split=split)

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(cast(Any, ds)):
        row = cast(Dict[str, Any], row)
        answers = row.get("answer", [])
        if not answers:
            continue
            
        samples.append({
            "question": row["question"],
            "answers": answers,
            "gold_label": None,
            "id": str(i),
            "benchmark": "natural_questions",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    return samples

# -----------------------------------------------------------------------------
# ARC-Challenge
# -----------------------------------------------------------------------------

def load_arc_challenge(
    split: str = "test",
    n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkSample]:
    """Load ARC-Challenge samples."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace `datasets` is required for ARC loading.") from e

    logger.info("Loading ARC-Challenge [%s] from HuggingFace...", split)
    ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(cast(Any, ds)):
        row = cast(Dict[str, Any], row)
        choices = row.get("choices", {})
        correct_label = row.get("answerKey", "")
        
        # Build constrained prompt
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        choices_text = " ".join([f"({lbl}) {txt}" for lbl, txt in zip(labels, texts)])
        question = f"Question: {row['question']} Choices: {choices_text}\nAnswer with just the multiple choice letter."

        samples.append({
            "question": question,
            "answers": [correct_label],
            "gold_label": correct_label,
            "id": row.get("id", str(i)),
            "benchmark": "arc_challenge",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    return samples


# -----------------------------------------------------------------------------
# Unified loader
# -----------------------------------------------------------------------------

def load_benchmark(
    name: str,
    n: Optional[int] = None,
    seed: int = 42,
    **kwargs,
) -> List[BenchmarkSample]:
    """Load any supported benchmark by name.

    Parameters
    ----------
    name : str
        One of: "fever", "triviaqa", "natural_questions" (training benchmarks)
        or "truthfulqa", "strategyqa", "arc_challenge" (transfer eval benchmarks)
        or "hotpotqa" (legacy, removed from main training loop).
    n : int or None
    seed : int
    **kwargs -- passed to the specific loader

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
    elif name == "triviaqa":
        return load_triviaqa(n=n, seed=seed, **kwargs)
    elif name == "natural_questions":
        return load_natural_questions(n=n, seed=seed, **kwargs)
    elif name == "arc_challenge":
        return load_arc_challenge(n=n, seed=seed, **kwargs)
    else:
        raise ValueError(
            f"Unknown benchmark '{name}'."
        )


# -----------------------------------------------------------------------------
# Synthetic data factory (for tests and dry-runs without downloading datasets)
# -----------------------------------------------------------------------------

def make_synthetic_samples(
    benchmark: str,
    n: int = 10,
    seed: int = 0,
) -> List[BenchmarkSample]:
    """Generate reproducible synthetic BenchmarkSample dicts.

    Used in unit tests and quick smoke-tests. No network access required.

    Parameters
    ----------
    benchmark : str
        One of: "fever", "triviaqa", "natural_questions", "truthfulqa",
        "strategyqa", "arc_challenge", or "hotpotqa" (legacy).
    n : int -- number of samples to generate
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

    elif benchmark == "triviaqa":
        for i in range(n):
            samples.append({
                "question": f"Who is person_{i}?",
                "answers": [f"answer_{i}", f"alias_{i}"],
                "gold_label": None,
                "id": f"trivia_synth_{i}",
                "benchmark": "triviaqa",
            })

    elif benchmark == "natural_questions":
        for i in range(n):
            samples.append({
                "question": f"When did event_{i} occur?",
                "answers": [f"year_{i}"],
                "gold_label": None,
                "id": f"nq_synth_{i}",
                "benchmark": "natural_questions",
            })

    elif benchmark == "arc_challenge":
        labels = ["A", "B", "C", "D"]
        for i in range(n):
            label = labels[i % 4]
            samples.append({
                "question": f"Question: Science concept {i}? Choices: (A) x (B) y (C) z (D) w\nAnswer with just the multiple choice letter.",
                "answers": [label],
                "gold_label": label,
                "id": f"arc_synth_{i}",
                "benchmark": "arc_challenge",
            })

    else:
        raise ValueError(
            f"Unknown benchmark '{benchmark}' for synthetic generation."
        )

    return samples
