"""
eval/
=====
Evaluation harness for the CAEM benchmark experiments.

Components
----------
metrics.py    -- EM, F1, FEVER accuracy, hallucination rate, routing distribution
benchmarks.py -- benchmark loaders + synthetic data factory
harness.py    -- EvalHarness: run pipeline over samples, aggregate, save JSON

Benchmark roles
---------------
Training benchmarks (SIL pool -- train split):
  fever, triviaqa, asqa  (Branch C: ASQA replaces Natural Questions)

Transfer eval benchmarks (held-out -- never used for SIL):
  truthfulqa, strategyqa, arc_challenge

Quick start
-----------
>>> from eval.benchmarks import load_benchmark, make_synthetic_samples
>>> from eval.harness import EvalHarness
>>> from caem.pipeline import CAEMPipeline
>>>
>>> pipeline = CAEMPipeline(...)
>>> harness = EvalHarness(pipeline, output_dir="outputs/eval")
>>>
>>> # Smoke test (no dataset download needed)
>>> result = harness.smoke_test("fever", n=5)
>>>
>>> # Real evaluation (dev = data-leakage-safe FEVER eval split, renamed from paper_dev)
>>> samples = load_benchmark("fever", n=500, split="dev")
>>> result = harness.run("fever", samples, cycle=0)
>>> result["em"], result["f1"]
"""

from eval.baselines import (
    BaselineBase,
    CoTBaseline,
    CoTRAGBaseline,
    FiveShotCoTBaseline,
    FLAREBaseline,
    RAGBaseline,
    ZeroShotBaseline,
)
from eval.benchmarks import (
    BenchmarkSample,
    load_benchmark,
    load_arc_challenge,
    load_asqa,
    load_fever,
    load_natural_questions,
    load_strategyqa,
    load_triviaqa,
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
    extract_arc_label,
    extract_fever_label,
    extract_strategyqa_label,
    fever_accuracy,
    hallucination_rate,
    mcnemar_test,
    normalise,
    rouge_l,
    routing_distribution,
    token_f1,
)

__all__ = [
    # baselines
    "BaselineBase",
    "ZeroShotBaseline",
    "CoTBaseline",
    "RAGBaseline",
    "CoTRAGBaseline",
    "FiveShotCoTBaseline",
    "FLAREBaseline",
    # benchmarks
    "BenchmarkSample",
    "load_benchmark",
    "load_arc_challenge",
    "load_asqa",
    "load_truthfulqa",
    "load_fever",
    "load_strategyqa",
    "load_triviaqa",
    "load_natural_questions",
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
    "extract_strategyqa_label",
    "extract_arc_label",
    "rouge_l",
    "hallucination_rate",
    "routing_distribution",
    "aggregate",
    "bootstrap_ci",
    "mcnemar_test",
]
