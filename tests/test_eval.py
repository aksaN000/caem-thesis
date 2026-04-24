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
Legacy helpers (Phase 4 harness):
  - normalise, exact_match, any_match_em, token_f1, best_token_f1
  - fever_accuracy, extract_fever_label, extract_strategyqa_label
  - composite_hallucination_metric (CHM), routing_distribution, aggregate
  - bootstrap_ci, mcnemar_test
Session 42 helpers (Ch. 5 seven-table suite):
  - extract_cot_answer, rouge_l, extract_arc_label
  - em_by_tier, mean_latency_by_tier
  - confabulation_rate (ge / le directions)
  - brier_score, auroc, reliability_bins
  - decision_breakdown (legacy DEFER label)
  - backward_transfer, forward_transfer
  - ces_score (geometric mean with eps clamp)
  - unsupported_correct_rate, ungrounded_assertion_rate
Harness / benchmarks:
  - make_synthetic_samples (all benchmarks)
  - load_benchmark (unknown raises)
  - EvalHarness.run / _score / _run_one / smoke_test / run_all
  - JSON save + reload round-trip
  - fail_on_error on/off paths
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
    auroc,
    backward_transfer,
    best_token_f1,
    bootstrap_ci,
    brier_score,
    ces_score,
    composite_hallucination_metric,
    confabulation_rate,
    decision_breakdown,
    em_by_tier,
    exact_match,
    extract_arc_label,
    extract_cot_answer,
    extract_fever_label,
    extract_strategyqa_label,
    fever_accuracy,
    forward_transfer,
    hallucination_subtypes,
    mcnemar_test,
    mean_latency_by_tier,
    normalise,
    reliability_bins,
    rouge_l,
    routing_distribution,
    token_f1,
    ungrounded_assertion_rate,
    unsupported_correct_rate,
)
from caem.pipeline import PipelineResult


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_pipeline_result(
    answer: str = "William Shakespeare",
    tier: int = 2,
    stored: bool = False,
    u_stored: float = 0.75,
    latency_ms: float = 120.0,
    escalated: bool = False,
) -> PipelineResult:
    """Build a PipelineResult for harness-path tests.

    Session 42 removed the StoredConfidence projection; u_stored is now a
    scalar directly on PipelineResult. verifier_output stays None here --
    these harness tests only read result.u_stored.
    """
    return PipelineResult(
        query="q",
        answer=answer,
        tier=tier,
        stored=stored,
        latency_ms=latency_ms,
        u_stored=u_stored,
        escalated=escalated,
    )


def _make_pipeline(answer: str = "answer_0", tier: int = 2, u_stored: float = 0.75) -> MagicMock:
    pipeline = MagicMock()
    pipeline.answer.return_value = _make_pipeline_result(answer=answer, tier=tier, u_stored=u_stored)
    return pipeline


# -----------------------------------------------------------------------------
# metrics.py -- normalise
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# metrics.py -- exact_match
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# metrics.py -- any_match_em
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# metrics.py -- token_f1
# -----------------------------------------------------------------------------

class TestTokenF1:
    def test_perfect_match(self):
        assert token_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("foo bar", "baz qux") == pytest.approx(0.0)

    def test_partial_overlap(self):
        f1 = token_f1("hello world foo", "hello world bar")
        assert f1 == pytest.approx(2 / 3)

    def test_empty_both(self):
        assert token_f1("", "") == pytest.approx(1.0)

    def test_empty_pred(self):
        assert token_f1("", "hello") == pytest.approx(0.0)

    def test_empty_gold(self):
        assert token_f1("hello", "") == pytest.approx(0.0)

    def test_subset(self):
        f1 = token_f1("hello", "hello world")
        assert f1 == pytest.approx(2 / 3)


class TestBestTokenF1:
    def test_picks_best(self):
        f1 = best_token_f1("hello world", ["foo", "hello world"])
        assert f1 == pytest.approx(1.0)

    def test_empty_golds(self):
        assert best_token_f1("hello", []) == pytest.approx(0.0)


# -----------------------------------------------------------------------------
# metrics.py -- fever
# -----------------------------------------------------------------------------

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
        # No label family matches -> UNPARSEABLE sentinel (empty string),
        # so downstream EM correctly yields 0.0 instead of silently
        # defaulting to "not enough info" (audit MAJOR-M3).
        assert extract_fever_label("I don't know anything.") == ""

    def test_last_occurrence_wins(self):
        # Mechanical semantics: the rightmost label family match wins
        # (model's final commitment). "support" appears after
        # "not enough info", so the extractor returns "supports".
        # This replaces the prior first-match-priority test, which
        # was locking in the biased if-elif ordering (audit MAJOR-M2).
        assert extract_fever_label(
            "There is not enough info to support this."
        ) == "supports"

    def test_nei_beats_earlier_supports(self):
        # And the symmetric case: when NEI appears after a supports
        # match, NEI wins -- confirming the last-occurrence rule,
        # not a static label precedence.
        assert extract_fever_label(
            "At first it supports the claim, but actually there is "
            "not enough info."
        ) == "not enough info"


class TestExtractStrategyQALabel:
    def test_yes_plain(self):
        assert extract_strategyqa_label("yes") == "yes"

    def test_no_plain(self):
        assert extract_strategyqa_label("no") == "no"

    def test_yes_in_rationale(self):
        text = "Reasoning: whales are mammals. Answer: yes"
        assert extract_strategyqa_label(text) == "yes"

    def test_no_in_rationale(self):
        text = "Reasoning: this is false. Therefore, no."
        assert extract_strategyqa_label(text) == "no"

    def test_fallback_returns_unparseable(self):
        # No yes/no word-boundary match -> UNPARSEABLE sentinel (empty
        # string), so downstream EM correctly yields 0.0 instead of
        # silently defaulting to "no" (audit MAJOR-M3).
        assert extract_strategyqa_label("uncertain output") == ""


# -----------------------------------------------------------------------------
# metrics.py -- composite_hallucination_metric (CHM) + hallucination_subtypes
# -----------------------------------------------------------------------------

class TestCompositeHallucinationMetric:
    """CHM (Composite Hallucination Metric) replaces the removed
    `hallucination_rate` helper; it is the equal-weighted mean of the 8
    taxonomy subtypes measurable under the CAEM-default MiniCheck backend.
    """

    def _sample(self, *, em=0.0, u=0.5, p_ga=0.5, p_pe=0.5, q_ar=0.8,
                decision="STORE", pred="an answer"):
        return {
            "em": em, "u_stored": u,
            "p_ground_atomic": p_ga, "p_entail": p_pe, "p_contra": 0.0,
            "q_a_relevance": q_ar, "decision": decision, "prediction": pred,
        }

    def test_empty_returns_zero_chm(self):
        out = composite_hallucination_metric([])
        assert out["chm"] == 0.0
        assert out["n_measured_subtypes"] == 0

    def test_all_clean_samples_chm_zero(self):
        samples = [self._sample(em=1.0, u=0.2) for _ in range(5)]
        out = composite_hallucination_metric(samples)
        assert out["chm"] == pytest.approx(0.0)

    def test_confident_wrong_fires_confabulation(self):
        s = self._sample(em=0.0, u=0.80, p_ga=0.8, p_pe=0.8, q_ar=0.8)
        out = composite_hallucination_metric([s])
        rates = out["per_subtype_rates"]
        assert rates["confident_confabulation_rate"] == pytest.approx(1.0)
        # union counts any-subtype firing for the sample.
        assert out["union_rate"] == pytest.approx(1.0)

    def test_denominator_excludes_contradiction_by_default(self):
        # CHM denominator defaults to 8 (factual_contradiction excluded
        # because p_contra is structurally 0 under MiniCheck).
        samples = [self._sample(em=1.0) for _ in range(3)]
        out = composite_hallucination_metric(samples)
        assert out["n_measured_subtypes"] == 8

    def test_override_measured_includes_all_nine(self):
        # Callers running the roberta_nli ablation can opt into all 9
        # subtypes by passing the full key list.
        samples = [self._sample(em=1.0) for _ in range(3)]
        all_keys = list(hallucination_subtypes(samples).keys())
        out = composite_hallucination_metric(samples, measured_subtypes=all_keys)
        assert out["n_measured_subtypes"] == 9

    def test_template_leak_fires_regardless_of_em(self):
        # template_leak condition is EM-agnostic: regex-match fires even
        # on correct answers (e.g., prompt-echo leak on a right answer).
        s = self._sample(em=1.0, pred="<concise factual answer>")
        out = composite_hallucination_metric([s])
        rates = out["per_subtype_rates"]
        assert rates["template_leak_rate"] == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# metrics.py -- routing_distribution
# -----------------------------------------------------------------------------

class TestRoutingDistribution:
    def test_all_tiers(self):
        dist = routing_distribution([1, 2, 3, 1, 2, 3, 3])
        assert dist["tier1_frac"] == pytest.approx(2 / 7)
        assert dist["tier2_frac"] == pytest.approx(2 / 7)
        assert dist["tier3_frac"] == pytest.approx(3 / 7)
        assert dist["crash_frac"] == pytest.approx(0.0)

    def test_all_tier3(self):
        dist = routing_distribution([3, 3, 3])
        assert dist["tier3_frac"] == pytest.approx(1.0)
        assert dist["tier1_frac"] == pytest.approx(0.0)
        assert dist["crash_frac"] == pytest.approx(0.0)

    def test_empty(self):
        dist = routing_distribution([])
        assert dist == {
            "tier1_frac": 0.0, "tier2_frac": 0.0, "tier3_frac": 0.0,
            "crash_frac": 0.0,
        }

    def test_crash_sentinel_excluded_from_tier_denominator(self):
        # tier=-1 is the harness crash sentinel (MAJOR-H2 / Task #117).
        # It must NOT count toward tier1/2/3 denominators, and it must
        # surface in crash_frac (over the TOTAL, including crashes).
        dist = routing_distribution([1, 2, 3, -1, -1])
        # 3 valid samples -> each of tier1/2/3 is 1/3.
        assert dist["tier1_frac"] == pytest.approx(1 / 3)
        assert dist["tier2_frac"] == pytest.approx(1 / 3)
        assert dist["tier3_frac"] == pytest.approx(1 / 3)
        # 2 of 5 total are crashes.
        assert dist["crash_frac"] == pytest.approx(2 / 5)
        # Invariant: tier*_frac sums to 1.0 over valid samples.
        assert (
            dist["tier1_frac"] + dist["tier2_frac"] + dist["tier3_frac"]
        ) == pytest.approx(1.0)

    def test_all_crashes(self):
        # All samples crashed -> no valid routes; every tier is 0.0
        # but crash_frac is 1.0.
        dist = routing_distribution([-1, -1, -1])
        assert dist["tier1_frac"] == 0.0
        assert dist["tier2_frac"] == 0.0
        assert dist["tier3_frac"] == 0.0
        assert dist["crash_frac"] == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# metrics.py -- aggregate
# -----------------------------------------------------------------------------

class TestAggregate:
    def test_keys_present(self):
        agg = aggregate(
            benchmark="natural_questions",
            em_scores=[1.0, 0.0],
            f1_scores=[1.0, 0.5],
            tiers=[1, 3],
            u_stored_values=[0.8, 0.3],
            stored_flags=[True, False],
            latencies_ms=[100.0, 200.0],
        )
        for key in ["benchmark", "n", "em", "f1",
                    "storage_rate", "mean_u_stored", "mean_latency_ms",
                    "tier1_frac", "tier2_frac", "tier3_frac"]:
            assert key in agg, f"Missing key: {key}"

    def test_em_computed(self):
        agg = aggregate("natural_questions", [1.0, 0.0], [1.0, 0.0], [2, 2], [0.8, 0.3], [True, False], [100.0, 200.0])
        assert agg["em"] == pytest.approx(0.5)

    def test_empty_input(self):
        agg = aggregate("natural_questions", [], [], [], [], [], [])
        assert agg["n"] == 0

    def test_storage_rate(self):
        agg = aggregate("natural_questions", [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2, 2, 2],
                        [0.8, 0.8, 0.3], [True, True, False], [100.0, 100.0, 100.0])
        assert agg["storage_rate"] == pytest.approx(2 / 3, abs=1e-3)


# -----------------------------------------------------------------------------
# benchmarks.py -- make_synthetic_samples
# -----------------------------------------------------------------------------

class TestMakeSyntheticSamples:
    def test_natural_questions_count(self):
        s = make_synthetic_samples("natural_questions", n=7)
        assert len(s) == 7

    def test_natural_questions_schema(self):
        s = make_synthetic_samples("natural_questions", n=1)[0]
        assert "question" in s and "answers" in s and "benchmark" in s
        assert s["benchmark"] == "natural_questions"

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
        assert "yes" in labels
        assert "no" in labels

    def test_strategyqa_question_format(self):
        s = make_synthetic_samples("strategyqa", n=1)[0]
        assert s["question"].startswith("Answer yes or no.")

    def test_fever_question_format(self):
        s = make_synthetic_samples("fever", n=1)[0]
        assert "supports, refutes, not enough info" in s["question"]

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError):
            make_synthetic_samples("unknown_bench", n=3)

    def test_load_benchmark_unknown_raises(self):
        with pytest.raises(ValueError):
            load_benchmark("nonexistent", n=5)


# -----------------------------------------------------------------------------
# harness.py -- EvalHarness
# -----------------------------------------------------------------------------

class TestEvalHarness:
    def test_run_returns_eval_result(self):
        pipeline = _make_pipeline(answer="answer_0")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("natural_questions", n=3)
        result = harness.run("natural_questions", samples, cycle=0)
        assert isinstance(result, dict)
        assert "em" in result

    def test_result_has_all_keys(self):
        pipeline = _make_pipeline()
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("natural_questions", n=2)
        r = harness.run("natural_questions", samples, cycle=0)
        for key in ["em", "f1", "n", "cycle", "tier1_frac", "tier2_frac", "tier3_frac",
                    "storage_rate", "mean_latency_ms"]:
            assert key in r, f"Missing key: {key}"

    def test_n_correct(self):
        pipeline = _make_pipeline()
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("natural_questions", n=5)
        r = harness.run("natural_questions", samples, cycle=1)
        assert r["n"] == 5

    def test_cycle_recorded(self):
        pipeline = _make_pipeline()
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("natural_questions", n=2)
        r = harness.run("natural_questions", samples, cycle=2)
        assert r["cycle"] == 2

    def test_perfect_em_when_answer_matches(self):
        pipeline = _make_pipeline(answer="year_0")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("natural_questions", n=1)
        r = harness.run("natural_questions", samples, cycle=0)
        assert r["em"] == pytest.approx(1.0)

    def test_zero_em_when_answer_wrong(self):
        pipeline = _make_pipeline(answer="wrong answer")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("natural_questions", n=3)
        r = harness.run("natural_questions", samples, cycle=0)
        assert r["em"] == pytest.approx(0.0)

    def test_fever_scoring_path(self):
        pipeline = _make_pipeline(answer="supports")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("fever", n=3)
        r = harness.run("fever", samples, cycle=0)
        assert 0.0 <= r["em"] <= 1.0

    def test_truthfulqa_scoring_path(self):
        pipeline = _make_pipeline(answer="yes_0")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("truthfulqa", n=1)
        r = harness.run("truthfulqa", samples, cycle=0)
        assert r["em"] == pytest.approx(1.0)

    def test_strategyqa_scoring_path_correct(self):
        pipeline = _make_pipeline(answer="yes")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("strategyqa", n=1)
        r = harness.run("strategyqa", samples, cycle=0)
        assert r["em"] == pytest.approx(1.0)

    def test_strategyqa_scoring_path_wrong(self):
        pipeline = _make_pipeline(answer="no")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("strategyqa", n=1)
        r = harness.run("strategyqa", samples, cycle=0)
        assert r["em"] == pytest.approx(0.0)

    def test_strategyqa_rationale_with_answer_marker_scores_correct(self):
        pipeline = _make_pipeline(answer="Reasoning: X. Answer: yes")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("strategyqa", n=1)
        r = harness.run("strategyqa", samples, cycle=0)
        assert r["em"] == pytest.approx(1.0)

    def test_triviaqa_scoring_path_matches_any_alias(self):
        pipeline = _make_pipeline(answer="alias_0")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("triviaqa", n=1)
        r = harness.run("triviaqa", samples, cycle=0)
        assert r["em"] == pytest.approx(1.0)

    def test_triviaqa_scoring_path_wrong_answer(self):
        pipeline = _make_pipeline(answer="completely wrong")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("triviaqa", n=1)
        r = harness.run("triviaqa", samples, cycle=0)
        assert r["em"] == pytest.approx(0.0)

    def test_natural_questions_scoring_path_correct(self):
        pipeline = _make_pipeline(answer="year_0")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("natural_questions", n=1)
        r = harness.run("natural_questions", samples, cycle=0)
        assert r["em"] == pytest.approx(1.0)

    def test_natural_questions_scoring_path_wrong(self):
        pipeline = _make_pipeline(answer="wrong answer")
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("natural_questions", n=1)
        r = harness.run("natural_questions", samples, cycle=0)
        assert r["em"] == pytest.approx(0.0)

    def test_saves_json_to_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = _make_pipeline()
            harness = EvalHarness(pipeline, output_dir=tmpdir)
            samples = make_synthetic_samples("natural_questions", n=2)
            harness.run("natural_questions", samples, cycle=0)
            fpath = Path(tmpdir) / "natural_questions_cycle0.json"
            assert fpath.exists()

    def test_saved_json_loadable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = _make_pipeline()
            harness = EvalHarness(pipeline, output_dir=tmpdir)
            samples = make_synthetic_samples("natural_questions", n=2)
            harness.run("natural_questions", samples, cycle=1)
            fpath = Path(tmpdir) / "natural_questions_cycle1.json"
            data = EvalHarness.load_result(fpath)
            assert "meta" in data and "samples" in data
            assert data["meta"]["n"] == 2

    def test_error_in_pipeline_does_not_crash(self):
        pipeline = MagicMock()
        pipeline.answer.side_effect = RuntimeError("model exploded")
        harness = EvalHarness(pipeline, fail_on_error=False)
        samples = make_synthetic_samples("natural_questions", n=3)
        r = harness.run("natural_questions", samples, cycle=0)
        assert r["em"] == pytest.approx(0.0)

    def test_error_reraises_when_fail_on_error(self):
        pipeline = MagicMock()
        pipeline.answer.side_effect = RuntimeError("boom")
        harness = EvalHarness(pipeline, fail_on_error=True)
        samples = make_synthetic_samples("natural_questions", n=1)
        with pytest.raises(RuntimeError):
            harness.run("natural_questions", samples, cycle=0)

    def test_smoke_test_natural_questions(self):
        pipeline = _make_pipeline()
        harness = EvalHarness(pipeline)
        r = harness.smoke_test("natural_questions", n=3)
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
            "natural_questions": make_synthetic_samples("natural_questions", n=2),
            "fever": make_synthetic_samples("fever", n=2),
        }
        results = harness.run_all(samples, cycle=0)
        assert "natural_questions" in results
        assert "fever" in results
        assert results["natural_questions"]["benchmark"] == "natural_questions"

    def test_none_u_stored_handled(self):
        pipeline = MagicMock()
        pipeline.answer.return_value = PipelineResult(
            query="q", answer="answer_0", tier=3,
            u_stored=None, latency_ms=50.0,
        )
        harness = EvalHarness(pipeline)
        samples = make_synthetic_samples("natural_questions", n=1)
        r = harness.run("natural_questions", samples, cycle=0)
        assert r["mean_u_stored"] is None or isinstance(r["mean_u_stored"], float)


# -----------------------------------------------------------------------------
# metrics.py -- bootstrap_ci
# -----------------------------------------------------------------------------

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

    def test_different_seeds_preserve_point_estimate(self):
        """Bootstrap resampling can only shift the CI bounds, not the point
        estimate (which is the sample mean of ``scores``). Confirm that
        seed variation leaves the mean unchanged and produces CI bounds
        that bracket it on both sides.

        Task #129 rewrite: the prior test was named `may_differ` but
        asserted strict equality on the point estimate — a contradiction
        that made the test a rubber stamp. The mean is a deterministic
        function of the input, not the seed; bootstrap only affects the
        CI. That is the actual property worth regressing.
        """
        scores = [float(i % 2) for i in range(40)]
        mean_ref = sum(scores) / len(scores)

        m1, lo1, hi1 = bootstrap_ci(scores, seed=1)
        m2, lo2, hi2 = bootstrap_ci(scores, seed=2)

        # Point estimate is the sample mean -- independent of seed.
        assert m1 == pytest.approx(mean_ref)
        assert m2 == pytest.approx(mean_ref)
        assert m1 == pytest.approx(m2)

        # CI bounds bracket the mean (not a strict property of bootstrap
        # in pathological cases, but holds for this well-behaved input).
        assert lo1 <= m1 <= hi1
        assert lo2 <= m2 <= hi2


# -----------------------------------------------------------------------------
# metrics.py -- mcnemar_test
# -----------------------------------------------------------------------------

class TestMcNemarTest:
    def test_identical_systems_p_is_one(self):
        a = [1.0, 0.0, 1.0, 1.0]
        b = [1.0, 0.0, 1.0, 1.0]
        stat, p = mcnemar_test(a, b)
        assert stat == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_completely_different_systems(self):
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
        import sys
        import unittest.mock as mock
        with mock.patch.dict(sys.modules, {"scipy.stats": None, "scipy": None}):
            with pytest.raises((ImportError, Exception)):
                mcnemar_test([1.0, 0.0], [0.0, 1.0])


# =============================================================================
# Session 42 metric helpers (Ch. 5 seven-table / figure suite)
# =============================================================================
#
# The fourteen helpers below back Tables 5.1-5.7 and Figures 5.1-5.7.
# They are pure-Python, numerically deterministic, and must be covered by
# unit tests so that the thesis's headline CES number, calibration panel,
# purity counts, grounding rates, and BWT/FWT scores cannot silently drift.
# =============================================================================


class TestExtractCotAnswer:
    def test_answer_marker_is_taken(self):
        assert extract_cot_answer("Reasoning: whales are mammals. Answer: yes") == "yes"

    def test_final_answer_marker_wins_over_earlier(self):
        text = "Answer: maybe. Wait, reconsider. Answer: no"
        assert extract_cot_answer(text) == "no"

    def test_no_marker_returns_full_text_trimmed(self):
        assert extract_cot_answer("  a plain answer  ") == "a plain answer"

    def test_empty_input(self):
        assert extract_cot_answer("") == ""

    # --- Flan-T5 "the answer is X" natural-language suffix ----------------- #
    # These patterns dominate TriviaQA / NQ under CAEM's RAG prompt ("Think
    # step by step. Answer:" -> model completes with "the answer is X"
    # rather than another explicit "Answer:" marker).

    def test_so_the_answer_is(self):
        text = "The relevant info is X. So, the answer is Paris."
        assert extract_cot_answer(text) == "Paris"

    def test_therefore_the_answer_is(self):
        text = "Following the reasoning. Therefore, the answer is Paris."
        assert extract_cot_answer(text) == "Paris"

    def test_plain_the_answer_is(self):
        assert extract_cot_answer("The answer is Victoria.") == "Victoria"

    def test_multi_word_answer_captured(self):
        text = "The answer is a jacket with shoulder straps."
        assert extract_cot_answer(text) == "a jacket with shoulder straps"

    def test_explicit_answer_marker_wins_over_natural_suffix(self):
        """Priority order: 'Answer:' takes precedence over 'the answer is X'
        when both are present, because 'Answer:' is the canonical marker."""
        text = "The answer is foo. Answer: bar"
        assert extract_cot_answer(text) == "bar"

    def test_natural_suffix_stops_at_sentence_boundary(self):
        """Non-greedy match: 'The answer is X. Also Y.' captures only X."""
        text = "The answer is Paris. Also France is a country."
        assert extract_cot_answer(text) == "Paris"

    def test_case_insensitive_marker(self):
        assert extract_cot_answer("so, THE ANSWER IS London.") == "London"


class TestRougeL:
    def test_perfect_match_is_one(self):
        assert rouge_l("hello world foo", ["hello world foo"]) == pytest.approx(1.0)

    def test_empty_pred(self):
        assert rouge_l("", ["anything"]) == pytest.approx(0.0)

    def test_no_overlap_is_zero(self):
        assert rouge_l("abc", ["xyz"]) == pytest.approx(0.0)

    def test_picks_best_gold(self):
        r = rouge_l("hello world", ["foo bar", "hello world"])
        assert r == pytest.approx(1.0)

    def test_bounded_in_unit_interval(self):
        r = rouge_l("hello there world", ["hello world foo"])
        assert 0.0 <= r <= 1.0


class TestExtractArcLabel:
    def test_letter_answer_marker(self):
        assert extract_arc_label("Answer: B") == "B"

    def test_bare_letter(self):
        assert extract_arc_label("A") == "A"

    def test_lowercase_is_uppercased(self):
        assert extract_arc_label("c") == "C"

    def test_fallback_to_empty_when_no_letter(self):
        assert extract_arc_label("no idea") == ""


# ---------------------------------------------------------------------------
# Per-tier slicers (Table 5.1)
# ---------------------------------------------------------------------------

class TestEmByTier:
    def test_all_three_tiers_present(self):
        em = [1.0, 0.0, 1.0, 1.0, 0.0]
        tiers = [1, 2, 1, 3, 2]
        out = em_by_tier(em, tiers)
        assert out["tier1"] == (pytest.approx(1.0), 2)
        assert out["tier2"] == (pytest.approx(0.0), 2)
        assert out["tier3"] == (pytest.approx(1.0), 1)

    def test_missing_tier_returns_zero_zero(self):
        out = em_by_tier([1.0, 0.0], [2, 2])
        assert out["tier1"] == (0.0, 0)
        assert out["tier3"] == (0.0, 0)
        assert out["tier2"] == (pytest.approx(0.5), 2)

    def test_empty_input(self):
        out = em_by_tier([], [])
        for k in ("tier1", "tier2", "tier3"):
            assert out[k] == (0.0, 0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(AssertionError):
            em_by_tier([1.0], [1, 2])


class TestMeanLatencyByTier:
    def test_per_tier_means(self):
        lat = [100.0, 200.0, 500.0, 50.0]
        tiers = [1, 2, 3, 1]
        out = mean_latency_by_tier(lat, tiers)
        assert out["tier1"] == pytest.approx(75.0)
        assert out["tier2"] == pytest.approx(200.0)
        assert out["tier3"] == pytest.approx(500.0)

    def test_missing_tier_is_zero(self):
        out = mean_latency_by_tier([100.0, 100.0], [2, 2])
        assert out["tier1"] == 0.0
        assert out["tier3"] == 0.0

    def test_empty_input(self):
        out = mean_latency_by_tier([], [])
        assert out == {"tier1": 0.0, "tier2": 0.0, "tier3": 0.0}


# ---------------------------------------------------------------------------
# Confident-error / confabulation rates (Table 5.5)
# ---------------------------------------------------------------------------

class TestConfabulationRate:
    def test_wrong_and_confident_counted_ge(self):
        em = [0.0, 0.0, 1.0, 1.0]
        sig = [0.9, 0.1, 0.9, 0.9]
        assert confabulation_rate(em, sig, threshold=0.5) == pytest.approx(0.25)

    def test_direction_le_inverts_criterion(self):
        em = [0.0, 0.0, 1.0]
        sig = [0.1, 0.9, 0.1]
        assert confabulation_rate(em, sig, threshold=0.2, direction="le") == pytest.approx(1 / 3)

    def test_all_correct_returns_zero(self):
        assert confabulation_rate([1.0, 1.0], [0.9, 0.9], threshold=0.5) == 0.0

    def test_empty_input(self):
        assert confabulation_rate([], [], threshold=0.5) == 0.0

    def test_invalid_direction_asserts(self):
        with pytest.raises(AssertionError):
            confabulation_rate([1.0], [0.5], threshold=0.5, direction="lt")


# ---------------------------------------------------------------------------
# Calibration panel (Table 5.2 / Figure 5.2)
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_calibration_is_zero(self):
        assert brier_score([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == pytest.approx(0.0)

    def test_worst_calibration_is_one(self):
        assert brier_score([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_uniform_half_on_mixed_labels(self):
        assert brier_score([0.5, 0.5], [0.0, 1.0]) == pytest.approx(0.25)

    def test_empty_input(self):
        assert brier_score([], []) == 0.0


class TestAuroc:
    def test_perfect_ranking(self):
        conf = [0.1, 0.2, 0.8, 0.9]
        lab = [0.0, 0.0, 1.0, 1.0]
        assert auroc(conf, lab) == pytest.approx(1.0)

    def test_worst_ranking(self):
        conf = [0.9, 0.8, 0.2, 0.1]
        lab = [0.0, 0.0, 1.0, 1.0]
        assert auroc(conf, lab) == pytest.approx(0.0)

    def test_single_class_returns_half(self):
        assert auroc([0.1, 0.9], [1.0, 1.0]) == pytest.approx(0.5)

    def test_empty_returns_half(self):
        assert auroc([], []) == pytest.approx(0.5)

    def test_ties_counted_at_half_weight(self):
        conf = [0.5, 0.5]
        lab = [0.0, 1.0]
        assert auroc(conf, lab) == pytest.approx(0.5)


class TestReliabilityBins:
    def test_returns_n_bins_rows(self):
        out = reliability_bins([0.1, 0.5, 0.9], [0.0, 1.0, 1.0], n_bins=5)
        assert len(out) == 5

    def test_empty_bins_have_midpoint_and_zero_acc(self):
        out = reliability_bins([0.05], [1.0], n_bins=2)
        mc0, ma0, n0 = out[0]
        mc1, ma1, n1 = out[1]
        assert n0 == 1
        assert n1 == 0
        assert mc1 == pytest.approx(0.75)
        assert ma1 == 0.0

    def test_perfect_calibration_means(self):
        conf = [0.9, 0.9, 0.9]
        lab = [1.0, 1.0, 1.0]
        out = reliability_bins(conf, lab, n_bins=10)
        mc, ma, n = out[9]
        assert n == 3
        assert mc == pytest.approx(0.9)
        assert ma == pytest.approx(1.0)

    def test_empty_input_all_zero_buckets(self):
        out = reliability_bins([], [], n_bins=4)
        assert len(out) == 4
        for i, (mc, ma, n) in enumerate(out):
            assert n == 0
            assert ma == 0.0
            assert mc == pytest.approx((i + 0.5) * 0.25)


# ---------------------------------------------------------------------------
# Decision breakdown (Table 5.5 purity)
# ---------------------------------------------------------------------------

class TestDecisionBreakdown:
    def test_counts_and_fractions(self):
        # Session-42 canonical schema: STORE / DEFERRED / ABSTAIN / DISCARD
        # (matches caem.unified_verifier.UnifiedVerifierOutput). Prior
        # "DEFER" key was a pre-Session-42 leftover -- audit MAJOR-M4.
        decisions = ["STORE", "STORE", "DEFERRED", "DISCARD", "ABSTAIN"]
        out = decision_breakdown(decisions)
        assert out["total"] == 5
        assert out["counts"]["STORE"] == 2
        assert out["counts"]["DEFERRED"] == 1
        assert out["counts"]["DISCARD"] == 1
        assert out["counts"]["ABSTAIN"] == 1
        assert out["fractions"]["STORE"] == pytest.approx(0.4)

    def test_unknown_decisions_are_ignored(self):
        # "GARBAGE" is not a canonical label and is silently dropped;
        # "DEFERRED" is now canonical (Session-42) so it counts.
        out = decision_breakdown(["STORE", "DEFERRED", "GARBAGE"])
        assert out["counts"]["STORE"] == 1
        assert out["counts"]["DEFERRED"] == 1
        assert out["total"] == 3
        # 2 recognised + 1 dropped -> sum of counts is 2, not 3.
        assert sum(out["counts"].values()) == 2

    def test_empty_input_has_zero_fractions(self):
        out = decision_breakdown([])
        assert out["total"] == 0
        for k in ("STORE", "DEFERRED", "ABSTAIN", "DISCARD"):
            assert out["counts"][k] == 0
            assert out["fractions"][k] == 0.0


# ---------------------------------------------------------------------------
# Continual-learning transfer (Table 5.6)
# ---------------------------------------------------------------------------

class TestBackwardTransfer:
    def test_positive_bwt_when_later_improves_earlier(self):
        R = [
            [0.5, 0.4, 0.3],
            [0.6, 0.5, 0.4],
            [0.7, 0.6, 0.5],
        ]
        # BWT = mean(R[2][0]-R[0][0], R[2][1]-R[1][1]) = mean(0.2, 0.1) = 0.15
        assert backward_transfer(R) == pytest.approx(0.15)

    def test_negative_bwt_is_forgetting(self):
        R = [
            [0.8, 0.1],
            [0.5, 0.4],
        ]
        # BWT = R[1][0] - R[0][0] = -0.3
        assert backward_transfer(R) == pytest.approx(-0.3)

    def test_single_cycle_returns_zero(self):
        assert backward_transfer([[0.5]]) == 0.0

    def test_empty_matrix_returns_zero(self):
        assert backward_transfer([]) == 0.0


class TestForwardTransfer:
    def test_positive_fwt_when_prev_cycle_helps_new(self):
        R = [
            [0.3, 0.2, 0.1],
            [0.3, 0.5, 0.4],
            [0.3, 0.5, 0.6],
        ]
        # FWT = mean(R[0][1]-bbar[1], R[1][2]-bbar[2])
        #     = mean(0.2-0.2, 0.4-0.1) = mean(0.0, 0.3) = 0.15
        assert forward_transfer(R) == pytest.approx(0.15)

    def test_explicit_baseline_overrides_cycle_zero(self):
        R = [
            [0.5, 0.5, 0.5],
            [0.5, 0.6, 0.5],
            [0.5, 0.5, 0.7],
        ]
        bbar = [0.1, 0.1, 0.1]
        # FWT = mean(R[0][1]-0.1, R[1][2]-0.1) = mean(0.4, 0.4) = 0.4
        assert forward_transfer(R, baseline_em=bbar) == pytest.approx(0.4)

    def test_single_cycle_returns_zero(self):
        assert forward_transfer([[0.5, 0.6]]) == 0.0


# ---------------------------------------------------------------------------
# CES score (headline -- Table 5.1 / Figure 5.1)
# ---------------------------------------------------------------------------

class TestCesScore:
    def test_all_uniform_returns_same_value(self):
        assert ces_score(0.5, 0.5, 0.5, 0.5, 0.5) == pytest.approx(0.5)

    def test_all_perfect_returns_one(self):
        assert ces_score(1.0, 1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_one_zero_axis_is_clamped_not_fatal(self):
        ces = ces_score(1.0, 1.0, 0.0, 1.0, 1.0)
        # Geometric mean of [1,1,0.01,1,1]: (0.01)^(1/5)
        assert ces > 0.0
        assert ces == pytest.approx(0.01 ** (1 / 5))

    def test_monotone_in_each_axis(self):
        base = ces_score(0.5, 0.5, 0.5, 0.5, 0.5)
        higher = ces_score(0.9, 0.5, 0.5, 0.5, 0.5)
        assert higher > base

    def test_bounded_between_eps_and_one(self):
        low = ces_score(-5.0, -5.0, -5.0, -5.0, -5.0)
        hi = ces_score(5.0, 5.0, 5.0, 5.0, 5.0)
        assert 0.0 < low <= 1.0
        assert 0.0 < hi <= 1.0

    def test_custom_eps_respected(self):
        ces = ces_score(1.0, 1.0, 0.0, 1.0, 1.0, eps=0.2)
        assert ces == pytest.approx(0.2 ** (1 / 5))


# ---------------------------------------------------------------------------
# Grounding diagnostics (Table 5.4)
# ---------------------------------------------------------------------------

class TestUnsupportedCorrectRate:
    def test_right_for_wrong_reason(self):
        em = [1.0, 1.0, 1.0, 0.0]
        grounds = [0.1, 0.9, 0.05, 0.9]
        assert unsupported_correct_rate(em, grounds) == pytest.approx(2 / 3)

    def test_no_correct_returns_zero(self):
        assert unsupported_correct_rate([0.0, 0.0], [0.1, 0.1]) == 0.0

    def test_all_grounded_correct_is_zero(self):
        em = [1.0, 1.0]
        grounds = [0.9, 0.9]
        assert unsupported_correct_rate(em, grounds) == 0.0


class TestUngroundedAssertionRate:
    def test_confident_and_ungrounded(self):
        # (g, u) pairs under defaults g_thresh=0.30, u_thresh=0.50.
        # Hits are (g <= 0.30 AND u >= 0.50):
        #   (0.10, 0.90) -> hit
        #   (0.90, 0.90) -> grounded, miss
        #   (0.10, 0.10) -> not confident, miss
        #   (0.20, 0.80) -> hit
        # -> 2/4 = 0.5
        grounds = [0.10, 0.90, 0.10, 0.20]
        us     = [0.90, 0.90, 0.10, 0.80]
        assert ungrounded_assertion_rate(grounds, us) == pytest.approx(0.5)

    def test_all_grounded_returns_zero(self):
        # Every g > g_thresh (0.30) -> nothing is ungrounded, rate must be 0.
        grounds = [0.90, 0.80, 0.70, 0.60]
        us = [0.90, 0.90, 0.90, 0.90]
        assert ungrounded_assertion_rate(grounds, us) == 0.0

    def test_empty_input_returns_zero(self):
        assert ungrounded_assertion_rate([], []) == 0.0

    def test_custom_thresholds(self):
        # g_thresh=0.35, u_thresh=0.70 -> hit iff g<=0.35 AND u>=0.70.
        #   (0.30, 0.80) -> hit
        #   (0.40, 0.80) -> g too high, miss
        #   (0.20, 0.60) -> u too low, miss
        #   (0.50, 0.90) -> g too high, miss
        # -> 1/4 = 0.25
        grounds = [0.30, 0.40, 0.20, 0.50]
        us     = [0.80, 0.80, 0.60, 0.90]
        assert ungrounded_assertion_rate(
            grounds, us, g_thresh=0.35, u_thresh=0.70
        ) == pytest.approx(0.25)
