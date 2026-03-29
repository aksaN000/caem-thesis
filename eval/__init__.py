"""
eval/
=====
Evaluation harness for the CAEM benchmark experiments.

Components
----------
metrics.py    — EM, F1, FEVER accuracy, hallucination rate, routing distribution
benchmarks.py — HotpotQA / TruthfulQA / FEVER loaders + synthetic data factory
harness.py    — EvalHarness: run pipeline over samples, aggregate, save JSON

Quick start
-----------
>>> from eval.benchmarks import load_hotpotqa, make_synthetic_samples
>>> from eval.harness import EvalHarness
>>> from caem.pipeline import CAEMPipeline
>>>
>>> pipeline = CAEMPipeline(...)
>>> harness = EvalHarness(pipeline, output_dir="outputs/eval")
>>>
>>> # Smoke test (no dataset download needed)
>>> result = harness.smoke_test("hotpotqa", n=5)
>>>
>>> # Real evaluation
>>> samples = load_hotpotqa(n=500)
>>> result = harness.run("hotpotqa", samples, cycle=0)
>>> result["em"], result["f1"]
"""

from eval.benchmarks import (
    BenchmarkSample,
    load_benchmark,
    load_fever,
    load_hotpotqa,
    load_strategyqa,
    load_truthfulqa,
    make_synthetic_samples,
)
from eval.harness import EvalHarness, EvalResult, SampleResult
from eval.metrics import (
    aggregate,
    any_match_em,
    best_token_f1,
    bootstrap_ci,
    exact_match,
    extract_fever_label,
    fever_accuracy,
    hallucination_rate,
    mcnemar_test,
    normalise,
    routing_distribution,
    token_f1,
)

__all__ = [
    # benchmarks
    "BenchmarkSample",
    "load_benchmark",
    "load_hotpotqa",
    "load_truthfulqa",
    "load_fever",
    "load_strategyqa",
    "make_synthetic_samples",
    # harness
    "EvalHarness",
    "EvalResult",
    "SampleResult",
    # metrics
    "normalise",
    "exact_match",
    "any_match_em",
    "token_f1",
    "best_token_f1",
    "fever_accuracy",
    "extract_fever_label",
    "hallucination_rate",
    "routing_distribution",
    "aggregate",
    "bootstrap_ci",
    "mcnemar_test",
]
