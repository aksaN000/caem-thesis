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
                              "truthfulqa", "strategyqa", "arc_challenge"

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
    FEVER dataset with identical content. Splits (as of 2026): 'train', 'dev', 'test'.
    We use 'dev' (~19,998 samples) = the original FEVER paper evaluation split.
    (Historical note: this split was named 'paper_dev' before HuggingFace's
    lucadiliello/fever dataset was renamed; the content is unchanged.)
    Label schema per dataset card: integer {0: supports, 1: not enough info, 2: refutes}.

    Constrained prompt design: the prompt explicitly enumerates the three
    valid labels so Flan-T5 produces clean, parseable outputs:

      Prompt: "Answer with one of: supports, refutes, not enough info.
               Claim: {claim}"

    Parameters
    ----------
    split : str
        "dev" (~19,998 samples, default -- standard eval split) or
        "train" (145K samples, for SIL pool) or "paper_test" (unlabeled).
        Use "dev" for all evaluation to avoid data leakage from train.
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
        # Default sentinel is None (not 2 = "refutes"): silently treating a
        # missing FEVER label as "refutes" would fabricate a false refutation
        # signal. In practice the `lucadiliello/fever` train/dev splits
        # always provide a `label`, but `paper_test` is unlabeled -- anyone
        # calling load_fever(split="test") would otherwise receive a
        # dataset where every sample is marked REFUTES. Rows without a label
        # are skipped instead.
        raw_label = row.get("label", None)
        if raw_label is None:
            continue
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

    # Attempt 1: ChilleD/StrategyQA -- currently maintained HF mirror with
    # both labeled 'train' (1,603) and 'test' (687) splits. Replaces the
    # deprecated `wics/strategy-qa` dataset script (HF Datasets dropped
    # support for repository scripts in 2024-2025). Schema is a superset
    # of what CAEM needs; _normalise_rows handles the (qid, question,
    # answer:bool) → (id, question, answers:[yes/no]) conversion.
    try:
        logger.info("Loading StrategyQA [ChilleD/StrategyQA, %s] from HuggingFace...", split)
        ds_hf = load_dataset("ChilleD/StrategyQA", split=split)
        hf_rows = [cast(Dict[str, Any], r) for r in cast(Any, ds_hf)]
        samples = _normalise_rows(hf_rows, f"ChilleD/StrategyQA:{split}")
    except Exception as exc:
        logger.warning("StrategyQA ChilleD split '%s' unavailable (%s). "
                       "Trying legacy wics/strategy-qa...", split, exc)
        # Attempt 1b: legacy wics/strategy-qa (deprecated but might work on
        # older HF datasets versions). Expected to fail on newer datasets
        # installations ("Dataset scripts are no longer supported").
        try:
            ds_hf = load_dataset("wics/strategy-qa", split=split)
            hf_rows = [cast(Dict[str, Any], r) for r in cast(Any, ds_hf)]
            samples = _normalise_rows(hf_rows, f"wics/strategy-qa:{split}")
        except Exception as exc2:
            logger.warning("Legacy wics/strategy-qa also failed (%s).", exc2)

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
# ASQA — long-form synthesis of ambiguous NQ-derived questions
#
# Stelmakh et al. 2022 (NAACL). Built on AmbigQA, itself derived from
# Natural Questions. Questions come from NQ with multiple valid
# interpretations; answers are long-form syntheses covering all
# interpretations. Ideal for stressing CAEM's Path B long-hypothesis
# verifier fallback while staying inside retrieval-augmented QA scope.
#
# - Question distribution: subset of NQ (ambiguous ones)
# - Retrieval corpus: Wikipedia (same as other CAEM benchmarks)
# - Gold format: long-form string answer (~100-300 tokens typical)
# - Metric: ROUGE-L against gold (see eval/metrics.py rouge_l)
# -----------------------------------------------------------------------------

def load_asqa(
    split: str = "dev",
    n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkSample]:
    """Load ASQA samples (long-form synthesis from NQ-derived ambiguous QA).

    Parameters
    ----------
    split : str
        ASQA provides "train" and "dev" splits (no public test).
        Use "train" for cold-start seeding (step_6_reseed) and "dev"
        for Cycle-0 / Step 7 main evaluation (default).
    n : int | None
        If set, randomly sample ``n`` examples using ``seed``.
    seed : int
        RNG seed for sub-sampling.

    Returns
    -------
    List[BenchmarkSample] with:
      - question: ambiguous question text
      - answers: list of 1+ long-form gold answer strings (each typically
        100-300 tokens). Scorer uses best-ROUGE-L across the list.
      - gold_label: None
      - id: ASQA sample_id
      - benchmark: "asqa"
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace `datasets` is required for ASQA loading.") from e

    logger.info("Loading ASQA [split=%s] from HuggingFace (din0s/asqa)...", split)
    # Canonical HF location: din0s/asqa (ASQA authors' upload).
    ds = load_dataset("din0s/asqa", split=split)

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(cast(Any, ds)):
        row = cast(Dict[str, Any], row)
        question = row.get("ambiguous_question", "") or row.get("question", "")
        if not question:
            continue

        # ASQA gold answers come in the "qa_pairs" field for short answers
        # and "annotations" field for long-form annotations. For long-form
        # evaluation we want the long_answer strings.
        long_answers: List[str] = []
        annotations = row.get("annotations", [])
        if isinstance(annotations, list):
            for ann in annotations:
                if isinstance(ann, dict):
                    long = ann.get("long_answer", "")
                    if long:
                        long_answers.append(long)

        # Fallback: some ASQA formats put the single long answer under "answer".
        if not long_answers:
            single = row.get("answer", "")
            if single:
                long_answers.append(single)

        if not long_answers:
            continue

        samples.append({
            "question": question,
            "answers": long_answers,
            "gold_label": None,
            "id": str(row.get("sample_id", i)),
            "benchmark": "asqa",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    logger.info("ASQA: %d samples loaded (%s split).", len(samples), split)
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
# HotpotQA (v2 — multi-hop entity recall)
# -----------------------------------------------------------------------------

def load_hotpotqa(
    split: str = "validation",
    n: Optional[int] = None,
    seed: int = 42,
    config_name: str = "distractor",
) -> List[BenchmarkSample]:
    """Load HotpotQA samples for v2 multi-hop training-panel benchmark.

    HotpotQA contains multi-hop questions requiring reasoning across
    multiple Wikipedia paragraphs. The "distractor" config (default)
    provides 10 candidate paragraphs per question (2 supporting + 8
    distractor). For CAEM the prompt only uses the question text;
    supporting facts are accessed via the Tier-3 RAG retrieval path
    (the verifier does NOT receive distractor passages directly).

    Sample schema:
      question   : str    -- raw question text
      answers    : list   -- single gold answer string (HotpotQA is single-answer)
      gold_label : None
      id         : str
      benchmark  : "hotpotqa"
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace `datasets` is required for HotpotQA loading.") from e

    logger.info("Loading HotpotQA [%s, %s] from HuggingFace...", config_name, split)
    ds = load_dataset("hotpot_qa", config_name, split=split)

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(cast(Any, ds)):
        row = cast(Dict[str, Any], row)
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        if not question or not answer:
            continue
        samples.append({
            "question": question,
            "answers": [answer],
            "gold_label": None,
            "id": str(row.get("id", f"hotpot_{i}")),
            "benchmark": "hotpotqa",
        })

    if n is not None and n < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    return samples


# -----------------------------------------------------------------------------
# CommonsenseQA (v2 — 5-choice MCQ commonsense)
# -----------------------------------------------------------------------------

def load_commonsense_qa(
    split: str = "validation",
    n: Optional[int] = None,
    seed: int = 42,
) -> List[BenchmarkSample]:
    """Load CommonsenseQA samples for v2 5-choice MCQ training-panel benchmark.

    CommonsenseQA presents a question with 5 labelled options A-E. The
    constrained prompt explicitly enumerates the option text alongside
    each label so the verifier's multichoice scorer can substitute the
    answer text into the hypothesis at NLI time. (The 4-choice
    multichoice_scorer regex already accepts A-E; CSQA exercises the
    5-option path.)

    Sample schema:
      question   : str    -- "Question: ... Choices: (A) ... (B) ... (C) ... (D) ... (E) ...\nAnswer with..."
      answers    : list   -- single letter A/B/C/D/E
      gold_label : str    -- the correct letter
      id         : str
      benchmark  : "commonsense_qa"

    Note: CommonsenseQA's test split labels are not public; eval/test
    folds draw from the dev split (1221 samples). The v2 split allocator
    (caem/benchmark_splits.py) handles this.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace `datasets` is required for CommonsenseQA loading.") from e

    logger.info("Loading CommonsenseQA [%s] from HuggingFace...", split)
    ds = load_dataset("commonsense_qa", split=split)

    samples: List[BenchmarkSample] = []
    for i, row in enumerate(cast(Any, ds)):
        row = cast(Dict[str, Any], row)
        question_text = str(row.get("question", ""))
        choices = row.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        correct_label = str(row.get("answerKey", "") or "")
        if not question_text or not labels or not correct_label:
            continue
        choices_text = " ".join([f"({lbl}) {txt}" for lbl, txt in zip(labels, texts)])
        question = (
            f"Question: {question_text} Choices: {choices_text}\n"
            f"Answer with just the multiple choice letter."
        )
        samples.append({
            "question": question,
            "answers": [correct_label],
            "gold_label": correct_label,
            "id": str(row.get("id", f"csqa_{i}")),
            "benchmark": "commonsense_qa",
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
        or "truthfulqa", "strategyqa", "arc_challenge" (transfer eval benchmarks).
    n : int or None
    seed : int
    **kwargs -- passed to the specific loader

    Returns
    -------
    list of BenchmarkSample
    """
    name = name.lower().strip()
    if name == "truthfulqa":
        return load_truthfulqa(n=n, seed=seed, **kwargs)
    elif name == "fever":
        return load_fever(n=n, seed=seed, **kwargs)
    elif name == "strategyqa":
        return load_strategyqa(n=n, seed=seed, **kwargs)
    elif name == "triviaqa":
        return load_triviaqa(n=n, seed=seed, **kwargs)
    elif name == "natural_questions":
        return load_natural_questions(n=n, seed=seed, **kwargs)
    elif name == "asqa":
        return load_asqa(n=n, seed=seed, **kwargs)
    elif name == "arc_challenge":
        return load_arc_challenge(n=n, seed=seed, **kwargs)
    elif name == "hotpotqa":
        return load_hotpotqa(n=n, seed=seed, **kwargs)
    elif name == "commonsense_qa":
        return load_commonsense_qa(n=n, seed=seed, **kwargs)
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
        "strategyqa", or "arc_challenge".
    n : int -- number of samples to generate
    seed : int

    Returns
    -------
    list of BenchmarkSample
    """
    rng = random.Random(seed)
    samples = []

    if benchmark == "truthfulqa":
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

    elif benchmark == "asqa":
        for i in range(n):
            samples.append({
                "question": f"When did multiple event_{i} variants occur?",
                "answers": [
                    f"Long-form synthesis discussing event_{i} variants A and B at time year_{i}."
                ],
                "gold_label": None,
                "id": f"asqa_synth_{i}",
                "benchmark": "asqa",
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

    elif benchmark == "hotpotqa":
        # Multi-hop QA synthetic — bridge entity between two facts.
        for i in range(n):
            samples.append({
                "question": (
                    f"Who is the spouse of the person who founded company_{i}?"
                ),
                "answers": [f"spouse_of_founder_{i}"],
                "gold_label": None,
                "id": f"hotpot_synth_{i}",
                "benchmark": "hotpotqa",
            })

    elif benchmark == "commonsense_qa":
        # 5-choice MCQ synthetic — exercises CSQA's 5-option path.
        labels = ["A", "B", "C", "D", "E"]
        for i in range(n):
            label = labels[i % 5]
            samples.append({
                "question": (
                    f"Question: Commonsense scenario {i}? "
                    f"Choices: (A) opt_a_{i} (B) opt_b_{i} (C) opt_c_{i} "
                    f"(D) opt_d_{i} (E) opt_e_{i}\n"
                    f"Answer with just the multiple choice letter."
                ),
                "answers": [label],
                "gold_label": label,
                "id": f"csqa_synth_{i}",
                "benchmark": "commonsense_qa",
            })

    else:
        raise ValueError(
            f"Unknown benchmark '{benchmark}' for synthetic generation."
        )

    return samples
