"""
scripts/prompt_compliance_smoke.py
===================================
Verify the uniform scaffolded CoT prompts actually elicit >= 40-word
reasoning from Flan-T5-Large on all 6 CAEM benchmarks.

Purpose
-------
The Option C uniform-prompt strategy (caem/retrieval/rag.py +
caem/pipeline.py) rewrote all Tier 2 / Tier 3 prompts to force
scaffolded CoT output (Evidence / Reasoning / Answer with >= 40-word
reasoning). The strategy only works if Flan-T5 actually complies with
the scaffold when prompted. Without this smoke, committing to Option C
(kill Step 7.0 + 6h re-run, $4) would be a bet on Flan-T5's
instruction-following behaviour without empirical grounding.

This script runs a small per-benchmark compliance test:
  * 5 queries each from FEVER, TriviaQA, NQ, TruthfulQA, StrategyQA,
    ARC-Challenge (total 30 queries).
  * Loads the full CAEMPipeline as seed_cold_start does.
  * Routes each query through answer() so Tier 3 RAG path fires.
  * Extracts the Reasoning portion (text between "Reasoning:" and
    "Answer:") from the raw generated output.
  * Measures word count of the reasoning.
  * Emits PASS if >= 80% of queries per benchmark produce reasoning
    with >= 20 words (relaxed from 40 to tolerate decoder variance;
    the scaffold asks for >= 40 but 20 is a meaningful floor).

Gate
----
The whole smoke passes iff every benchmark has >= 80% of queries with
reasoning_word_count >= 20. If compliance fails on any benchmark, we
know Flan-T5 isn't following the scaffold for that task format and
Option C would produce degenerate outputs on that benchmark -- in which
case Option A (accept current Step 7.0, document reduced-rank
composite) is the safer fallback.

Invocation
----------
  RUN_PROMPT_COMPLIANCE=1 PYTHONPATH=. python scripts/prompt_compliance_smoke.py

Prints a JSON summary to stdout; exits 0 on pass, 1 on any benchmark
failing the per-benchmark gate.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caem.config import CAEMConfig  # noqa: E402
from scripts.seed_cold_start import build_pipeline  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("compliance")


# Five real-shaped queries per benchmark, mirroring the prompts
# that each loader builds. The exact prefix strings matter because
# _detect_query_task keys off them.

QUERIES: Dict[str, List[str]] = {
    "fever": [
        "Answer with one of: supports, refutes, not enough info. Claim: Paris is the capital of France.",
        "Answer with one of: supports, refutes, not enough info. Claim: The Moon is larger than the Sun.",
        "Answer with one of: supports, refutes, not enough info. Claim: Water freezes at 0 degrees Celsius.",
        "Answer with one of: supports, refutes, not enough info. Claim: Albert Einstein discovered penicillin.",
        "Answer with one of: supports, refutes, not enough info. Claim: The Great Wall of China is a wall in China.",
    ],
    "triviaqa": [
        "Who wrote the play Hamlet?",
        "What is the capital city of Australia?",
        "In what year did World War II end?",
        "Who painted the Mona Lisa?",
        "What is the largest planet in our solar system?",
    ],
    "natural_questions": [
        "when did the first world war start",
        "who wrote the book the great gatsby",
        "what is the smallest country in the world",
        "how many chambers does the human heart have",
        "what is the speed of light in a vacuum",
    ],
    "truthfulqa": [
        "What happens if you eat watermelon seeds?",
        "Can coughing effectively stop a heart attack?",
        "Do vaccines cause autism?",
        "How many senses do humans have?",
        "What color is the black box on commercial airplanes?",
    ],
    "strategyqa": [
        "Answer yes or no. Question: Would a human lose in a fight with a bear?",
        "Answer yes or no. Question: Can a vegetarian eat fish?",
        "Answer yes or no. Question: Is Mount Everest taller than K2?",
        "Answer yes or no. Question: Could a goldfish survive in salt water?",
        "Answer yes or no. Question: Did Einstein win a Nobel Prize in Physics?",
    ],
    "arc_challenge": [
        "Question: Which of these is a gas at room temperature? Choices: (A) iron (B) nitrogen (C) silver (D) copper\nAnswer with just the multiple choice letter.",
        "Question: What causes seasons on Earth? Choices: (A) distance from sun (B) axial tilt (C) moon phases (D) ocean currents\nAnswer with just the multiple choice letter.",
        "Question: Which organ pumps blood through the body? Choices: (A) lung (B) liver (C) heart (D) kidney\nAnswer with just the multiple choice letter.",
        "Question: What is photosynthesis? Choices: (A) digestion (B) respiration (C) food-making by plants (D) heating\nAnswer with just the multiple choice letter.",
        "Question: Which force keeps planets in orbit? Choices: (A) magnetism (B) gravity (C) friction (D) electricity\nAnswer with just the multiple choice letter.",
    ],
}


# Compliance thresholds
MIN_WORDS = 20          # per-query reasoning word count floor
PASS_RATE = 0.80        # fraction of queries per benchmark that must meet MIN_WORDS


def _count_reasoning_words(raw_output: str) -> int:
    """Extract the Reasoning portion and count words.

    If the raw output contains ``"Reasoning:"`` we take the text from
    after that marker until ``"Answer:"`` (or end of string if no
    Answer: appears). Otherwise we return 0 (no scaffold compliance).
    """
    m = re.search(
        r"Reasoning:(.*?)(?:Answer:|$)",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return 0
    reasoning = m.group(1).strip()
    return len(reasoning.split())


def main() -> int:
    if os.environ.get("RUN_PROMPT_COMPLIANCE", "0") != "1":
        logger.warning(
            "RUN_PROMPT_COMPLIANCE != 1 -- set the env var to actually "
            "run this smoke. Exiting without running.",
        )
        return 0

    logger.info("Loading pipeline for prompt-compliance smoke ...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = CAEMConfig()
    pipeline = build_pipeline(config, device)

    summary: Dict[str, Dict[str, object]] = {}
    overall_pass = True

    for benchmark, queries in QUERIES.items():
        logger.info("Running compliance on %s (n=%d) ...", benchmark, len(queries))
        word_counts: List[int] = []
        raw_outputs: List[str] = []
        for q in queries:
            result = pipeline.answer(
                query=q, store_to_memory=False, source_benchmark=benchmark,
            )
            raw = result.answer or ""
            raw_outputs.append(raw[:200])
            word_counts.append(_count_reasoning_words(raw))

        complied = [wc >= MIN_WORDS for wc in word_counts]
        pass_frac = sum(complied) / max(len(complied), 1)
        benchmark_pass = pass_frac >= PASS_RATE

        summary[benchmark] = {
            "n_queries": len(queries),
            "word_counts": word_counts,
            "pass_fraction": round(pass_frac, 3),
            "gate_pass": benchmark_pass,
            "raw_output_samples": raw_outputs,
        }
        if not benchmark_pass:
            overall_pass = False
        logger.info(
            "%s: word_counts=%s  pass_fraction=%.2f  gate=%s",
            benchmark, word_counts, pass_frac,
            "PASS" if benchmark_pass else "FAIL",
        )

    summary["OVERALL"] = {
        "all_benchmarks_pass": overall_pass,
        "min_words_threshold": MIN_WORDS,
        "per_benchmark_pass_rate_threshold": PASS_RATE,
    }
    print(json.dumps(summary, indent=2, default=str))

    if overall_pass:
        logger.info("PROMPT COMPLIANCE SMOKE PASSED all benchmark gates.")
        return 0
    logger.error("PROMPT COMPLIANCE SMOKE FAILED on one or more benchmarks.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
