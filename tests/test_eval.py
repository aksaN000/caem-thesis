"""
tests/test_eval.py
==================
Unit tests for the eval/ package (metrics, benchmarks, harness).

Mock strategy
-------------
- No real HuggingFace datasets are downloaded. All tests use
  make_synthetic_samples() or hand-crafted dicts.
- The CAEMPipeline is replaced with a MagicMock that returns a
  PipelineResult with controlled values.
- EvalHarness tests verify scoring logic, aggregation, and JSON I/O.

Coverage
--------
metrics.py
  - normalise: articles, punctuation, whitespace
  - exact_match: match / no-match / empty
  - any_match_em: matches any / matches none
  - token_f1: perfect / partial / zero
  - best_token_f1: picks the best F1
  - fever_accuracy: matching / non-matching labels
  - extract_fever_label: supports / refutes / not enough info / fallback
  - hallucination_rate: all wrong+uncertain / mixed / empty
  - routing_distribution: all tiers present / empty
  - aggregate: shape and keys

benchmarks.py
  - make_synthetic_samples: hotpotqa / truthfulqa / fever / strategyqa / unknown
  - load_benchmark: unknown name raises ValueError

harness.py
  - EvalHarness.run: returns EvalResult with correct keys
  - EvalHarness._score: hotpotqa / truthfulqa / fever paths
  - EvalHarness._run_one: error path (fail_on_error=False)
  - EvalHarness.run saves JSON to output_dir
  - EvalHarness.load_result reloads saved file
  - EvalHarness.smoke_test completes without downloading data
  - EvalHarness.run_all runs multiple benchmarks
  - EvalHarness StrategyQA scoring path (yes/no EM)

metrics.py (statistical)
  - bootstrap_ci: returns (mean, lower, upper), lower ≤ mean ≤ upper
  - mcnemar_test: identical systems → p=1.0; different systems → p<0.05
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.benchmarks import make_synthetic_samples, load_benchmark
from eval.harness import EvalHarness
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
from caem.pipeline import PipelineResult
from caem.memory.entry import StoredConfidence


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pipeline_result(
    answer: str = "William Shakespeare",
    tier: int = 2,
    stored: bool = False,
    u_stored: float = 0.75,
    latency_ms: float = 120.0,
    escalated: bool = False,
) -> PipelineResult:
    sc = StoredConfidence(p_entail=u_stored, s_avg=u_stored, h_norm=0.1, u_stored=u_stored)
    return PipelineResult(
        query="q",
        answer=answer,
        tier=tier,
        stored=stored,
        latency_ms=latency_ms,
        stored_confidence=sc,
        escalated=escalated,
    )


def _make_pipeline(answer: str = "answer_0", tier: int = 2, u_stored: float = 0.75) -> MagicMock:
    pipeline = MagicMock()
    pipeline.answer.return_value = _make_pipeline_result(answer=answer, tier=tier, u_stored=u_stored)
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — normalise
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalise:
    def test_lowercase(self):
        assert normalise("HELLO WORLD") == "hello world"

    def test_removes_articles(self):
        assert normalise("a the quick an brown fox") == "quick brown fox"

    def test_removes_punctuation(self):
        assert normalise("Hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalise("hello   world") == "hello world"

    def test_empty_string(self):
        assert normalise("") == ""

    def test_only_articles(self):
        assert normalise("a an the") == ""


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — exact_match
# ─────────────────────────────────────────────────────────────────────────────

class TestExactMatch:
    def test_exact_match_true(self):
        assert exact_match("William Shakespeare", "William Shakespeare") == 1.0

    def test_case_insensitive(self):
        assert exact_match("shakespeare", "Shakespeare") == 1.0

    def test_article_stripped(self):
        assert exact_match("the Beatles", "Beatles") == 1.0

    def test_no_match(self):
        assert exact_match("Keats", "Shakespeare") == 0.0

    def test_empty_both(self):
        assert exact_match("", "") == 1.0

    def test_empty_pred(self):
        assert exact_match("", "Shakespeare") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — any_match_em
# ─────────────────────────────────────────────────────────────────────────────

class TestAnyMatchEM:
    def test_matches_first(self):
        assert any_match_em("yes", ["yes", "correct", "true"]) == 1.0

    def test_matches_last(self):
        assert any_match_em("true", ["yes", "correct", "true"]) == 1.0

    def test_matches_none(self):
        assert any_match_em("maybe", ["yes", "correct"]) == 0.0

    def test_empty_golds(self):
        assert any_match_em("yes", []) == 0.0

    def test_case_insensitive(self):
        assert any_match_em("YES", ["yes"]) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — token_f1
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenF1:
    def test_perfect_match(self):
        assert token_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("foo bar", "baz qux") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # pred: hello world foo; gold: hello world bar
        # common: hello, world → 2
        # precision = 2/3; recall = 2/3; f1 = 2/3
        f1 = token_f1("hello world foo", "hello world bar")
        assert f1 == pytest.approx(2 / 3)

    def test_empty_both(self):
        assert token_f1("", "") == pytest.approx(1.0)

    def test_empty_pred(self):
        assert token_f1("", "hello") == pytest.approx(0.0)

    def test_empty_gold(self):
        assert token_f1("hello", "") == pytest.approx(0.0)

    def test_subset(self):
        # pred is subset of gold: precision=1, recall=0.5 → f1=2/3
        f1 = token_f1("hello", "hello world")
        assert f1 == pytest.approx(2 / 3)


class TestBestTokenF1:
    def test_picks_best(self):
        # gold1 = "foo"; gold2 = "hello world" (pred = "hello world")
        f1 = best_token_f1("hello world", ["foo", "hello world"])
        assert f1 == pytest.approx(1.0)

    def test_empty_golds(self):
        assert best_token_f1("hello", []) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — fever
# ─────────────────────────────────────────────────────────────────────────────

class TestFeverAccuracy:
    def test_supports_match(self):
        assert fever_accuracy("SUPPORTS", "supports") == 1.0

    def test_refutes_match(self):
        assert fever_accuracy("refutes", "REFUTES") == 1.0

    def test_nei_match(self):
        assert fever_accuracy("not enough info", "not enough info") == 1.0

    def test_no_match(self):
        assert fever_accuracy("supports", "refutes") == 0.0

    def test_whitespace_stripped(self):
        assert fever_accuracy("  supports  ", "supports") == 1.0


class TestExtractFeverLabel:
    def test_supports(self):
        assert extract_fever_label("The claim is supported.") == "supports"

    def test_refutes(self):
        assert extract_fever_label("This claim is refuted by evidence.") == "refutes"

    def test_nei_phrase(self):
        assert extract_fever_label("There is not enough information.") == "not enough info"

    def test_uppercase_refutes(self):
        assert extract_fever_label("REFUTES") == "refutes"

    def test_fallback(self):
        assert extract_fever_label("I don't know anything.") == "not enough info"

    def test_nei_takes_priority_over_supports(self):
        # "not enough info" should take priority
        assert extract_fever_label("There is not enough info to support this.") == "not enough info"


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — hallucination_rate
# ─────────────────────────────────────────────────────────────────────────────

class TestHallucinationRate:
    def test_all_wrong_uncertain(self):
        em = [0.0, 0.0, 0.0]
        u = [0.3, 0.2, 0.1]
        # Uncertain errors are Safe Failures, not confident confabulations.
        assert hallucination_rate(em, u) == pytest.approx(0.0)

    def test_all_correct(self):
        em = [1.0, 1.0]
        u = [0.3, 0.2]
        assert hallucination_rate(em, u) == pytest.approx(0.0)

    def test_mixed(self):
        em = [0.0, 1.0, 0.0, 1.0]
        u = [0.3, 0.3, 0.8, 0.3]   # sample 0: safe failure; sample 2: confident confabulation
        rate = hallucination_rate(em, u)
        assert rate == pytest.approx(0.25)   # 1 out of 4 (sample 2)

    def test_none_u_stored_treated_as_zero(self):
        em = [0.0]
        u = [None]
        # u=None becomes u=0.0 (uncertain), so it's a safe failure
        assert hallucination_rate(em, u) == pytest.approx(0.0)

    def test_empty(self):
        assert hallucination_rate([], []) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — routing_distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestRoutingDistribution:
    def test_all_tiers(self):
        dist = routing_distribution([1, 2, 3, 1, 2, 3, 3])
        assert dist["tier1_frac"] == pytest.approx(2 / 7)
        assert dist["tier2_frac"] == pytest.approx(2 / 7)
        assert dist["tier3_frac"] == pytest.approx(3 / 7)

    def test_all_tier3(self):
        dist = routing_distribution([3, 3, 3])
        assert dist["tier3_frac"] == pytest.approx(1.0)
        assert dist["tier1_frac"] == pytest.approx(0.0)

    def test_empty(self):
        dist = routing_distribution([])
        assert dist == {"tier1_frac": 0.0, "tier2_frac": 0.0, "tier3_frac": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — aggregate
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregate:
    def test_keys_present(self):
        agg = aggregate(
            benchmark="hotpotqa",
            em_scores=[1.0, 0.0],
            f1_scores=[1.0, 0.5],
            tiers=[1, 3],
            u_stored_values=[0.8, 0.3],
            stored_flags=[True, False],
            latencies_ms=[100.0, 200.0],
        )
        for key in ["benchmark", "n", "em", "f1", "hallucination_rate",
                    "storage_rate", "mean_u_stored", "mean_latency_ms",
                    "tier1_frac", "tier2_frac", "tier3_frac"]:
            assert key in agg, f"Missing key: {key}"

    def test_em_computed(self):
        agg = aggregate("hotpotqa", [1.0, 0.0], [1.0, 0.0], [2, 2], [0.8, 0.3], [True, False], [100.0, 200.0])
        assert agg["em"] == pytest.approx(0.5)

    def test_empty_input(self):
        agg = aggregate("hotpotqa", [], [], [], [], [], [])
        assert agg["n"] == 0

    def test_storage_rate(self):
        agg = aggregate("hotpotqa", [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2, 2, 2],
                        [0.8, 0.8, 0.3], [True, True, False], [100.0, 100.0, 100.0])
        # aggregate() rounds to 4 decimal places: round(2/3, 4) = 0.6667
        assert agg["storage_rate"] == pytest.approx(2 / 3, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# benchmarks.py — make_synthetic_samples
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeSyntheticSamples:
    def test_hotpotqa_count(self):
        s = make_synthetic_samples("hotpotqa", n=7)
        assert len(s) == 7

    def test_hotpotqa_schema(self):
        s = make_synthetic_samples("hotpotqa", n=1)[0]
        assert "question" in s and "answers" in s and "benchmark" in s
        assert s["benchmark"] == "hotpotqa"

    def test_truthfulqa_multiple_answers(self):
        s = make_synthetic_samples("truthfulqa", n=1)[0]
        assert len(s["answers"]) >= 2

    def test_fever_labels(self):
        samples = make_synthetic_samples("fever", n=6)
        labels = {s["gold_label"] for s in samples}
        assert "supports" in labels
        assert "refutes" in labels

    def test_strategyqa_count(self):
        s = make_synthetic_samples("strategyqa", n=4)
        assert len(s) == 4

    def test_strategyqa_schema(self):
        s = make_synthetic_samples("strategyqa", n=1)[0]
        assert "question" in s and "answers" in s and "benchmark" in s
        assert s["benchmark"] == "strategyqa"

    def test_strategyqa_boolean_labels(self):
        samples = make_synthetic_samples("strategyqa", n=4)
        labels = {s["gold_label"] for s in samples}
        # Alternating yes/no — both must appear in 4 samples
        assert "yes" in labels
        assert "no" in labels

    def test_strategyqa_question_format(self):
        s = make_synthetic_samples("strategyqa", n=1)[0]
        # Uses the constrained boolean prompt
        assert s["question"].startswith("Answer yes or no.")

    def test_fever_question_format(self):
        s = make_synthetic_samples("fever", n=1)[0]
        # Uses the constrained label prompt — no free-form prefix
        assert "supports, refutes, not enough info" in s["question"]

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError):
            make_synthetic_samples("unknown_bench", n=3)

    def test_load_benchmark_unknown_raises(self):
        with pytest.raises(ValueError):
            load_benchmark("nonexistent", n=5)


# ─────────────────────────────────────────────────────────────────────────────
# harness.py — EvalHarness
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalHarness:
    def test_run_returns_eval_result(self):
        pipeline = _make_pipeline(answer="answer_0")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("hotpotqa", n=3)
        result = harness.run("hotpotqa", samples, cycle=0)
        assert isinstance(result, dict)
        assert "em" in result

    def test_result_has_all_keys(self):
        pipeline = _make_pipeline()
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("hotpotqa", n=2)
        r = harness.run("hotpotqa", samples, cycle=0)
        for key in ["em", "f1", "n", "cycle", "tier1_frac", "tier2_frac", "tier3_frac",
                    "storage_rate", "hallucination_rate", "mean_latency_ms"]:
            assert key in r, f"Missing key: {key}"

    def test_n_correct(self):
        pipeline = _make_pipeline()
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("hotpotqa", n=5)
        r = harness.run("hotpotqa", samples, cycle=1)
        assert r["n"] == 5

    def test_cycle_recorded(self):
        pipeline = _make_pipeline()
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("hotpotqa", n=2)
        r = harness.run("hotpotqa", samples, cycle=2)
        assert r["cycle"] == 2

    def test_perfect_em_when_answer_matches(self):
        """Synthetic HotpotQA gold answer is 'answer_0'. Pipeline returns 'answer_0'."""
        pipeline = _make_pipeline(answer="answer_0")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("hotpotqa", n=1)  # gold = "answer_0"
        r = harness.run("hotpotqa", samples, cycle=0)
        assert r["em"] == pytest.approx(1.0)

    def test_zero_em_when_answer_wrong(self):
        pipeline = _make_pipeline(answer="wrong answer")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("hotpotqa", n=3)
        r = harness.run("hotpotqa", samples, cycle=0)
        assert r["em"] == pytest.approx(0.0)

    def test_fever_scoring_path(self):
        """Fever samples: pipeline returns 'supports', gold may match."""
        pipeline = _make_pipeline(answer="supports")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("fever", n=3)
        r = harness.run("fever", samples, cycle=0)
        assert 0.0 <= r["em"] <= 1.0

    def test_truthfulqa_scoring_path(self):
        pipeline = _make_pipeline(answer="yes_0")  # matches answers[0]
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("truthfulqa", n=1)
        r = harness.run("truthfulqa", samples, cycle=0)
        assert r["em"] == pytest.approx(1.0)

    def test_strategyqa_scoring_path_correct(self):
        """StrategyQA: pipeline returns 'yes', gold=yes → EM=1.0."""
        # Synthetic samples alternate yes/no; seed=0, first sample → gold="yes"
        pipeline = _make_pipeline(answer="yes")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("strategyqa", n=1)  # sample 0 → gold="yes"
        r = harness.run("strategyqa", samples, cycle=0)
        assert r["em"] == pytest.approx(1.0)

    def test_strategyqa_scoring_path_wrong(self):
        """StrategyQA: pipeline returns 'no', gold=yes → EM=0.0."""
        pipeline = _make_pipeline(answer="no")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("strategyqa", n=1)  # gold="yes"
        r = harness.run("strategyqa", samples, cycle=0)
        assert r["em"] == pytest.approx(0.0)

    def test_saves_json_to_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = _make_pipeline()
            harness = EvalHarness(pipeline, output_dir=tmpdir)
            samples = make_synthetic_samples("hotpotqa", n=2)
            harness.run("hotpotqa", samples, cycle=0)
            fpath = Path(tmpdir) / "hotpotqa_cycle0.json"
            assert fpath.exists()

    def test_saved_json_loadable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = _make_pipeline()
            harness = EvalHarness(pipeline, output_dir=tmpdir)
            samples = make_synthetic_samples("hotpotqa", n=2)
            harness.run("hotpotqa", samples, cycle=1)
            fpath = Path(tmpdir) / "hotpotqa_cycle1.json"
            data = EvalHarness.load_result(fpath)
            assert "meta" in data and "samples" in data
            assert data["meta"]["n"] == 2

    def test_error_in_pipeline_does_not_crash(self):
        """fail_on_error=False: bad pipeline → em=0, harness continues."""
        pipeline = MagicMock()
        pipeline.answer.side_effect = RuntimeError("model exploded")
        harness = EvalHarness(pipeline, fail_on_error=False)
        samples = make_synthetic_samples("hotpotqa", n=3)
        r = harness.run("hotpotqa", samples, cycle=0)
        assert r["em"] == pytest.approx(0.0)

    def test_error_reraises_when_fail_on_error(self):
        pipeline = MagicMock()
        pipeline.answer.side_effect = RuntimeError("boom")
        harness = EvalHarness(pipeline, fail_on_error=True)
        samples = make_synthetic_samples("hotpotqa", n=1)
        with pytest.raises(RuntimeError):
            harness.run("hotpotqa", samples, cycle=0)

    def test_smoke_test_hotpotqa(self):
        pipeline = _make_pipeline()
        harness = EvalHarness(pipeline)
        r = harness.smoke_test("hotpotqa", n=3)
        assert "em" in r

    def test_smoke_test_fever(self):
        pipeline = _make_pipeline(answer="supports")
        harness = EvalHarness(pipeline)
        r = harness.smoke_test("fever", n=3)
        assert "em" in r

    def test_run_all_multiple_benchmarks(self):
        pipeline = _make_pipeline()
        harness = EvalHarness(pipeline)
        samples = {
            "hotpotqa": make_synthetic_samples("hotpotqa", n=2),
            "fever": make_synthetic_samples("fever", n=2),
        }
        results = harness.run_all(samples, cycle=0)
        assert "hotpotqa" in results
        assert "fever" in results
        assert results["hotpotqa"]["benchmark"] == "hotpotqa"

    def test_none_stored_confidence_handled(self):
        """Pipeline returns PipelineResult with stored_confidence=None."""
        pipeline = MagicMock()
        pipeline.answer.return_value = PipelineResult(
            query="q", answer="answer_0", tier=3,
            stored_confidence=None, latency_ms=50.0,
        )
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("hotpotqa", n=1)
        r = harness.run("hotpotqa", samples, cycle=0)
        # mean_u_stored should be None (no valid u_stored values)
        assert r["mean_u_stored"] is None or isinstance(r["mean_u_stored"], float)


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — bootstrap_ci
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_returns_three_values(self):
        mean, lo, hi = bootstrap_ci([1.0, 0.0, 1.0, 0.0], n_bootstrap=100)
        assert isinstance(mean, float)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_mean_within_ci(self):
        scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        mean, lo, hi = bootstrap_ci(scores, n_bootstrap=500)
        assert lo <= mean <= hi

    def test_perfect_scores_narrow_ci(self):
        """All 1.0 → mean=1.0, lower=1.0, upper=1.0."""
        mean, lo, hi = bootstrap_ci([1.0] * 20, n_bootstrap=200)
        assert mean == pytest.approx(1.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_zero_scores(self):
        mean, lo, hi = bootstrap_ci([0.0] * 10, n_bootstrap=100)
        assert mean == pytest.approx(0.0)
        assert lo == pytest.approx(0.0)

    def test_empty_scores(self):
        mean, lo, hi = bootstrap_ci([])
        assert mean == 0.0 and lo == 0.0 and hi == 0.0

    def test_reproducible_with_seed(self):
        scores = [1.0, 0.0, 1.0, 0.0, 0.5]
        r1 = bootstrap_ci(scores, seed=42)
        r2 = bootstrap_ci(scores, seed=42)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        """Different seeds should usually produce slightly different CIs for mixed data."""
        scores = [float(i % 2) for i in range(40)]
        r1 = bootstrap_ci(scores, seed=1)
        r2 = bootstrap_ci(scores, seed=2)
        # Not guaranteed to differ, but mean should always be the same
        assert r1[0] == pytest.approx(r2[0])


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py — mcnemar_test
# ─────────────────────────────────────────────────────────────────────────────

class TestMcNemarTest:
    def test_identical_systems_p_is_one(self):
        """No disagreements → statistic=0, p=1.0."""
        a = [1.0, 0.0, 1.0, 1.0]
        b = [1.0, 0.0, 1.0, 1.0]
        stat, p = mcnemar_test(a, b)
        assert stat == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_completely_different_systems(self):
        """A always wrong, B always right → large statistic, p≪1."""
        a = [0.0] * 20
        b = [1.0] * 20
        stat, p = mcnemar_test(a, b)
        assert stat > 0
        assert p < 0.05

    def test_returns_two_floats(self):
        a = [1.0, 0.0, 1.0]
        b = [0.0, 1.0, 1.0]
        result = mcnemar_test(a, b)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(AssertionError):
            mcnemar_test([1.0, 0.0], [1.0])

    def test_scipy_required(self):
        """mcnemar_test raises ImportError if scipy is absent (mocked)."""
        import sys
        import unittest.mock as mock
        # Temporarily hide scipy
        with mock.patch.dict(sys.modules, {"scipy.stats": None, "scipy": None}):
            with pytest.raises((ImportError, Exception)):
                mcnemar_test([1.0, 0.0], [0.0, 1.0])
