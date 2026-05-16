#!/usr/bin/env python
"""
scripts/generate_self_rag_synthetic_data.py
============================================
Synthetic-data generation for Self-RAG fine-tuning (B10 full FT path).

Generates ~10-30K reflection-token-annotated training examples by:
  1. Loading (question, gold_answer) pairs from existing benchmark training
     splits (FEVER, TriviaQA, CommonsenseQA — same data used by CAEM's SIL).
  2. For each pair, retrieving top-k passages from the same FAISS index
     as CAEM Tier 3 (data/passage_index/).
  3. Calling the Anthropic API (using the user's Claude Max API key from
     ANTHROPIC_API_KEY env var) to annotate each (question, passages,
     answer) tuple with the four reflection-token types:
       - [Retrieve] / [No-Retrieve]
       - [Relevant] / [Irrelevant] per passage
       - [Supported] / [Partial] / [NoSupport]
       - [Useful: 1-5]
  4. Inserting the reflection tokens at appropriate positions in the
     answer text per the Self-RAG protocol (Asai et al. ICLR 2024 §3.2).
  5. Writing the annotated examples to a training file in the format
     expected by scripts/train_self_rag.py.

Status: SCAFFOLD (2026-05-16). The Anthropic API call, retrieval, and
token-insertion functions need to be implemented before this script can
run end-to-end. The structure here is the skeleton to fill in.

Expected wall-clock: ~2-3 hours for 10K examples with 10 concurrent
Anthropic API connections at ~1 sec/call. Cost: $0 (covered by Claude
Max subscription's included API credits, ASSUMING the Max plan covers
programmatic API usage at this volume — verify before running).

Engineering effort to complete: ~1-2 days (mostly prompt-template
iteration and quality validation).

Usage (planned)
---------------
    export ANTHROPIC_API_KEY=...
    python -m scripts.generate_self_rag_synthetic_data \\
        --benchmarks fever triviaqa commonsense_qa \\
        --n_per_bench 5000 \\
        --output outputs/self_rag/synthetic_train.jsonl \\
        --max_concurrent 10 \\
        --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Reflection token vocabulary (Asai et al. 2024 §3.2)
REFLECTION_TOKENS = [
    "[Retrieve]", "[No-Retrieve]",
    "[Relevant]", "[Irrelevant]",
    "[Supported]", "[Partial]", "[NoSupport]",
    "[Useful:1]", "[Useful:2]", "[Useful:3]", "[Useful:4]", "[Useful:5]",
]


# Annotation prompt template — fills in (question, retrieved_passages, gold_answer)
# and asks Claude to emit reflection-token-annotated training text.
ANNOTATION_PROMPT_TEMPLATE = """You are annotating training data for Self-RAG, a retrieval-augmented language model.

Given a question, retrieved passages, and a gold answer, produce a training example with reflection tokens inserted per the Self-RAG protocol.

The four reflection-token types:
- [Retrieve] / [No-Retrieve]: should the model retrieve for this question?
- [Relevant] / [Irrelevant]: is each retrieved passage relevant?
- [Supported] / [Partial] / [NoSupport]: is the answer supported by the passages?
- [Useful:1] to [Useful:5]: how useful is the answer (1=worst, 5=best)?

Annotation rules (Asai et al. 2024 §3.2):
1. Open-domain factual questions almost always need retrieval → use [Retrieve].
2. If a passage clearly contains evidence for the answer → [Relevant], else [Irrelevant].
3. If the answer's claims are all backed by relevant passages → [Supported]; if some claims are not → [Partial]; if none → [NoSupport].
4. Useful score: 5 if answer is correct and fluent, 1 if confused or incorrect.

Output format: produce a JSON object with this exact structure:
{{
  "retrieve_decision": "[Retrieve]" or "[No-Retrieve]",
  "passage_decisions": ["[Relevant]" or "[Irrelevant]"] (one per retrieved passage),
  "support_decision": "[Supported]" or "[Partial]" or "[NoSupport]",
  "useful_score": "[Useful:1]" to "[Useful:5]",
  "annotated_answer": "Reasoning: ... <token>...<token> Answer: ... <token>"
}}

Input:
Question: {question}
Retrieved passages:
{passages}
Gold answer: {gold_answer}

Output JSON only, no commentary."""


def annotate_one_example(
    question: str,
    passages: List[str],
    gold_answer: str,
    *,
    model: str = "claude-haiku-4-5",
    api_key: Optional[str] = None,
) -> Optional[Dict]:
    """Call Anthropic API to annotate a single example with reflection tokens.

    STATUS: NOT YET IMPLEMENTED. Stub that returns None until the
    anthropic SDK + retry + parsing logic is filled in.

    Engineering TODO:
      1. Use `anthropic.Anthropic(api_key=api_key).messages.create()` with
         the ANNOTATION_PROMPT_TEMPLATE filled in.
      2. Parse the returned JSON; validate schema.
      3. Retry on rate-limit or parse failure (3 attempts max).
      4. Return the parsed dict on success, None on permanent failure.
    """
    logger.debug("annotate_one_example: not yet implemented (stub)")
    return None


def load_qa_pairs(benchmark: str, n: int) -> List[Dict]:
    """Load n (question, answer) pairs from the benchmark's training split.

    STATUS: NOT YET IMPLEMENTED. Should reuse caem.benchmark_splits or
    eval.benchmarks loaders to pull pairs from the SIL training-pool side
    of the data (NOT the calibration / eval / test folds, to avoid
    contamination).
    """
    logger.warning("load_qa_pairs: not yet implemented (stub)")
    return []


def retrieve_passages(question: str, k: int = 5) -> List[str]:
    """Retrieve top-k passages from the same FAISS index as CAEM Tier 3.

    STATUS: NOT YET IMPLEMENTED. Should use caem.retrieval.rag.PassageStore
    + the registered QueryEncoder to mirror Tier 3 retrieval exactly. The
    annotation step needs the SAME passages CAEM would see at inference
    time so the training distribution matches.
    """
    logger.warning("retrieve_passages: not yet implemented (stub)")
    return []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", nargs="+",
                   default=["fever", "triviaqa", "commonsense_qa"])
    p.add_argument("--n_per_bench", type=int, default=5000)
    p.add_argument("--output", type=Path,
                   default=Path("outputs/self_rag/synthetic_train.jsonl"))
    p.add_argument("--max_concurrent", type=int, default=10,
                   help="Parallel API calls. Tune to Anthropic rate limits.")
    p.add_argument("--model", default="claude-haiku-4-5",
                   help="Anthropic model for annotation. Haiku is fast + cheap.")
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(level=ns.log_level, format="%(asctime)s %(levelname)s %(message)s")

    logger.error("scripts/generate_self_rag_synthetic_data.py is a SCAFFOLD; "
                 "the annotate_one_example / load_qa_pairs / retrieve_passages "
                 "helpers are stubs. Implement them before running end-to-end.")
    logger.info("Planned pipeline:")
    logger.info("  1. load_qa_pairs(bench) for each benchmark in %s", ns.benchmarks)
    logger.info("  2. retrieve_passages(q) for each q (uses CAEM's FAISS index)")
    logger.info("  3. annotate_one_example(q, passages, a) via Anthropic API")
    logger.info("  4. Write %d annotated examples per bench to %s",
                ns.n_per_bench, ns.output)
    logger.info("Reflection-token vocabulary: %s", REFLECTION_TOKENS)
    return 1  # non-zero to flag this is a stub run


if __name__ == "__main__":
    sys.exit(main())
