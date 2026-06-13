"""
eval/
=====
Evaluation harness for the CAEM benchmark experiments.

Components
----------
metrics.py    -- EM, F1, FEVER accuracy, CHM (composite hallucination
                 metric over 8 taxonomy subtypes), routing distribution,
                 pooled_chm + chm_reduction_verdict pass/fail gate
benchmarks.py -- benchmark loaders + synthetic data factory
harness.py    -- EvalHarness: run pipeline over samples, aggregate, save JSON
reporting.py  -- 6 Chapter-5 tables + per_sample_signals.jsonl
baselines.py  -- B1..B7 external baselines (B5 = 5-shot CoT, not FLARE)

Benchmark roles (Branch C 2026-04-22 evening decision, authoritative in
caem/config.py::TRAINING_BENCHMARKS / TRANSFER_BENCHMARKS)
-----------------------------------------------------------------------
Training benchmarks (SIL pool -- train split):
  fever, triviaqa, natural_questions
  (Training panel capped at 3 high-volume benchmarks -- FEVER ~145k,
  TriviaQA ~87k, NQ ~87k -- to sustain 10-cycle stream mode at 5000
  samples/cycle without reuse. ASQA's 4,353 train samples are
  structurally too small for stream mode, so ASQA was demoted to
  transfer-only; NQ returned to the training pool.)

Transfer eval benchmarks (held-out -- never used for SIL):
  truthfulqa, strategyqa, arc_challenge, asqa
  (ASQA stays in the eval panel as Path B / Qwen-judge long-form
  evidence; eval trajectory only.)

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

# The baselines/benchmarks/harness submodules depend on torch (and the wider
# model stack). Stats-only consumers (e.g. the paired-significance table
# generators) only need ``eval.metrics``, which is torch-free. Guard the heavy
# eager imports so the package still exposes its metrics on a torch-less host;
# on the GPU host where torch is installed this branch is a no-op.
try:
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
except ImportError:  # torch (or another heavy dep) unavailable on this host
    BaselineBase = CoTBaseline = CoTRAGBaseline = FiveShotCoTBaseline = None
    FLAREBaseline = RAGBaseline = ZeroShotBaseline = None
    BenchmarkSample = load_benchmark = load_arc_challenge = load_asqa = None
    load_fever = load_natural_questions = load_strategyqa = None
    load_triviaqa = load_truthfulqa = make_synthetic_samples = None
    EvalHarness = EvalResult = SampleResult = None
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
    chm_reduction_verdict,
    composite_hallucination_metric,
    hallucination_subtypes,
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
    "chm_reduction_verdict",
    "composite_hallucination_metric",
    "hallucination_subtypes",
    "routing_distribution",
    "aggregate",
    "bootstrap_ci",
    "mcnemar_test",
]
