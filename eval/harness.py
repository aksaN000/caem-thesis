"""
eval/harness.py
===============
EvalHarness -- runs a benchmark against CAEMPipeline and collects results.

Responsibilities
----------------
1. Iterate over BenchmarkSample dicts from eval/benchmarks.py.
2. Call pipeline.answer(question) for each sample.
3. Compute per-sample metrics (EM, F1, FEVER accuracy) via eval/metrics.py.
4. For FEVER: extract the label from the model's free-form answer.
5. Collect routing stats (tier distribution), storage rate, latency.
6. Produce a per-sample result list and an aggregate summary dict.
7. Save results to JSON (per-run) for later analysis.
8. Optionally log progress to stdout for long-running evaluations.

Output schema
-------------
Per-sample (SampleResult dict):
  id, benchmark, question, prediction, gold_answers, gold_label,
  em, f1, tier, stored, u_stored, latency_ms, escalated

Aggregate (EvalResult dict -- one per run):
  benchmark, cycle, n, em, f1, hallucination_rate,
  storage_rate, mean_u_stored, mean_latency_ms,
  tier1_frac, tier2_frac, tier3_frac

Usage
-----
>>> harness = EvalHarness(pipeline, output_dir="outputs/eval")
>>> result = harness.run(benchmark="hotpotqa", samples=samples, cycle=0)
>>> result["em"]          # e.g. 0.312
>>> result["tier3_frac"]  # e.g. 0.884 (mostly Tier 3 before memory warms up)

The output JSON at outputs/eval/hotpotqa_cycle0.json can be fed directly
into the analysis scripts for Chapter 5 figures.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.benchmarks import BenchmarkSample, make_synthetic_samples
from eval.metrics import (
    aggregate,
    any_match_em,
    best_token_f1,
    rouge_l,
    exact_match,
    extract_fever_label,
    extract_strategyqa_label,
    extract_arc_label,
    extract_cot_answer,
    fever_accuracy,
    token_f1,
)

logger = logging.getLogger(__name__)

# Type aliases
SampleResult = Dict[str, Any]
EvalResult = Dict[str, Any]


class EvalHarness:
    """Run one benchmark against a CAEMPipeline and collect results.

    Parameters
    ----------
    pipeline : CAEMPipeline
        Fully initialised pipeline (model, memory, RAG all set up).
    output_dir : str or Path or None
        If provided, save JSON results here after each run.
        If None, results are returned but not saved to disk.
    log_every : int
        Log progress every N samples (0 = no progress logging).
    fail_on_error : bool
        If True, re-raise exceptions from pipeline.answer().
        If False (default), record the sample as EM=0 and continue.

    Usage
    -----
    >>> harness = EvalHarness(pipeline, output_dir="outputs/eval")
    >>> samples = load_hotpotqa(n=500)
    >>> result = harness.run("hotpotqa", samples, cycle=1)
    """

    def __init__(
        self,
        pipeline,
        output_dir: Optional[str | Path] = None,
        log_every: int = 100,
        fail_on_error: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.output_dir = Path(output_dir) if output_dir else None
        self.log_every = log_every
        self.fail_on_error = fail_on_error

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        benchmark: str,
        samples: List[BenchmarkSample],
        cycle: int = 0,
        store_to_memory: bool = False,
    ) -> EvalResult:
        """Evaluate all samples from one benchmark at one cycle.

        Parameters
        ----------
        benchmark : str -- "hotpotqa", "truthfulqa", or "fever"
        samples : list of BenchmarkSample
        cycle : int -- current self-improvement cycle (for filename and metadata)

        Returns
        -------
        EvalResult dict -- aggregate metrics for this run
        """
        benchmark = benchmark.lower()
        logger.info(
            "EvalHarness: starting %s | cycle=%d | n=%d samples",
            benchmark, cycle, len(samples),
        )

        sample_results: List[SampleResult] = []

        for i, sample in enumerate(samples):
            if self.log_every > 0 and i % self.log_every == 0:
                logger.info("  [%s/%s] %s cycle=%d ...", i, len(samples), benchmark, cycle)

            sr = self._run_one(sample, benchmark, store_to_memory)
            sample_results.append(sr)

        # -- Aggregate --------------------------------------------------- #
        em_scores = [sr["em"] for sr in sample_results]
        f1_scores = [sr["f1"] for sr in sample_results]
        tiers = [sr["tier"] for sr in sample_results]
        u_stored_vals = [sr["u_stored"] for sr in sample_results]
        stored_flags = [sr["stored"] for sr in sample_results]
        latencies = [sr["latency_ms"] for sr in sample_results]

        agg = aggregate(
            benchmark=benchmark,
            em_scores=em_scores,
            f1_scores=f1_scores,
            tiers=tiers,
            u_stored_values=u_stored_vals,
            stored_flags=stored_flags,
            latencies_ms=latencies,
        )
        agg["cycle"] = cycle

        logger.info(
            "EvalHarness done: %s | cycle=%d | EM=%.4f | F1=%.4f | "
            "Tier1=%.2f%% Tier2=%.2f%% Tier3=%.2f%% | stored=%.2f%%",
            benchmark, cycle,
            agg["em"], agg["f1"],
            agg["tier1_frac"] * 100,
            agg["tier2_frac"] * 100,
            agg["tier3_frac"] * 100,
            agg["storage_rate"] * 100,
        )

        # -- Save -------------------------------------------------------- #
        full_output = {
            "meta": agg,
            "samples": sample_results,
        }

        if self.output_dir:
            fname = f"{benchmark}_cycle{cycle}.json"
            fpath = self.output_dir / fname
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(full_output, f, indent=2, ensure_ascii=False)
            logger.info("Results saved -> %s", fpath)

        return agg

    # ------------------------------------------------------------------ #
    # Per-sample evaluation                                                #
    # ------------------------------------------------------------------ #

    def _run_one(self, sample: BenchmarkSample, benchmark: str, store_to_memory: bool = False) -> SampleResult:
        """Run one sample through the pipeline and score it.

        Returns a SampleResult dict. On error, records em=0, f1=0, tier=3.
        """
        question = sample["question"]
        gold_answers = sample["answers"]
        gold_label = sample.get("gold_label")

        # -- Pipeline call ---------------------------------------------- #
        try:
            result = self.pipeline.answer(question, store_to_memory=store_to_memory)
            prediction = result.answer
            tier = result.tier
            stored = result.stored
            u_stored = (
                result.stored_confidence.u_stored
                if result.stored_confidence is not None
                else None
            )
            latency_ms = result.latency_ms
            escalated = result.escalated
        except Exception as exc:
            if self.fail_on_error:
                raise
            logger.warning("pipeline.answer() raised for sample %s: %s", sample.get("id"), exc)
            prediction = ""
            tier = 3
            stored = False
            u_stored = None
            latency_ms = 0.0
            escalated = False

        # -- Scoring ---------------------------------------------------- #
        em, f1 = self._score(prediction, gold_answers, gold_label, benchmark)

        return {
            "id": sample.get("id", ""),
            "benchmark": benchmark,
            "question": question,
            "prediction": prediction,
            "gold_answers": gold_answers,
            "gold_label": gold_label,
            "em": em,
            "f1": f1,
            "tier": tier,
            "stored": stored,
            "u_stored": u_stored,
            "latency_ms": latency_ms,
            "escalated": escalated,
        }

    def _score(
        self,
        prediction: str,
        gold_answers: List[str],
        gold_label: Optional[str],
        benchmark: str,
    ):
        """Return (em, f1) for a prediction given the benchmark type.

        HotpotQA  -- EM + F1 against single gold answer
        TruthfulQA -- any-match EM + best F1 across accepted answers
        FEVER      -- label extraction + accuracy (F1 = EM for labels)
        """
        prediction = extract_cot_answer(prediction)

        if benchmark == "fever":
            pred_label = extract_fever_label(prediction)
            gold = gold_label or (gold_answers[0] if gold_answers else "not enough info")
            acc = fever_accuracy(pred_label, gold)
            return acc, acc   # f1 == em for label classification

        elif benchmark == "truthfulqa":
            # EM equivalent uses ROUGE-L threshold > 0.15 since TruthfulQA has no exact string matches.
            f1 = rouge_l(prediction, gold_answers)
            em = float(f1 > 0.15)
            return em, f1

        elif benchmark == "strategyqa":
            # Boolean QA: robust yes/no label extraction from free-form output.
            gold = gold_answers[0] if gold_answers else "no"
            pred_label = extract_strategyqa_label(prediction)
            em = exact_match(pred_label, gold)
            f1 = em   # F1 == EM for binary labels
            return em, f1

        elif benchmark == "arc_challenge":
            gold = gold_answers[0] if gold_answers else ""
            pred_label = extract_arc_label(prediction)
            em = exact_match(pred_label, gold)
            return em, em

        else:  # hotpotqa and any future QA benchmarks
            gold = gold_answers[0] if gold_answers else ""
            em = exact_match(prediction, gold)
            f1 = token_f1(prediction, gold)
            return em, f1

    # ------------------------------------------------------------------ #
    # Multi-benchmark convenience runner                                   #
    # ------------------------------------------------------------------ #

    def run_all(
        self,
        samples_by_benchmark: Dict[str, List[BenchmarkSample]],
        cycle: int = 0,
        store_to_memory: bool = False,
    ) -> Dict[str, EvalResult]:
        """Run all benchmarks and return a dict of results.

        Parameters
        ----------
        samples_by_benchmark : dict -- {"hotpotqa": [...], "truthfulqa": [...], ...}
        cycle : int

        Returns
        -------
        dict -- {"hotpotqa": EvalResult, ...}
        """
        results = {}
        for benchmark, samples in samples_by_benchmark.items():
            results[benchmark] = self.run(benchmark, samples, cycle=cycle, store_to_memory=store_to_memory)
        return results

    # ------------------------------------------------------------------ #
    # Result loading (for post-hoc analysis)                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_result(cls, path: str | Path) -> Dict:
        """Load a previously saved JSON result file.

        Parameters
        ----------
        path : str or Path -- path to the .json file saved by run()

        Returns
        -------
        dict with keys "meta" (EvalResult) and "samples" (list of SampleResult)
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_all_cycles(
        cls, output_dir: str | Path, benchmark: str
    ) -> List[Dict]:
        """Load all cycle results for one benchmark from an output directory.

        Returns them sorted by cycle number.
        """
        output_dir = Path(output_dir)
        pattern = f"{benchmark}_cycle*.json"
        files = sorted(output_dir.glob(pattern))
        return [cls.load_result(f) for f in files]

    # ------------------------------------------------------------------ #
    # Smoke test (quick sanity check without downloading datasets)        #
    # ------------------------------------------------------------------ #

    def smoke_test(
        self,
        benchmark: str = "hotpotqa",
        n: int = 5,
        cycle: int = 0,
    ) -> EvalResult:
        """Run a tiny synthetic eval to verify the harness is wired correctly.

        Parameters
        ----------
        benchmark : str
        n : int -- number of synthetic samples
        cycle : int

        Returns
        -------
        EvalResult
        """
        samples = make_synthetic_samples(benchmark=benchmark, n=n)
        return self.run(benchmark=benchmark, samples=samples, cycle=cycle)
