"""
tests/test_label_faithfulness.py
================================
Unit tests for ``scripts/label_faithfulness.py`` aggregation layer.

Only the pure-Python aggregation is covered here -- the live MiniCheck
judge load is GPU-gated and runs manually on Vast. A mocked
``judge_fn`` stands in for the (premise, hypothesis) -> scores contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.label_faithfulness import (
    _extract_passages,
    build_labels,
    load_signals_jsonl,
    score_record,
)


# -----------------------------------------------------------------------------
# _extract_passages
# -----------------------------------------------------------------------------

class TestExtractPassages:
    def test_top_passages_as_strings(self):
        rec = {"top_passages": ["p1", "p2"]}
        assert _extract_passages(rec) == ["p1", "p2"]

    def test_top_passages_scored_shape(self):
        rec = {
            "top_passages_scored": [
                {"text": "p1", "score": 0.9},
                {"text": "p2", "score": 0.7},
            ],
        }
        assert _extract_passages(rec) == ["p1", "p2"]

    def test_missing_returns_empty(self):
        assert _extract_passages({}) == []
        assert _extract_passages({"top_passages": None}) == []

    def test_empty_list_returns_empty(self):
        assert _extract_passages({"top_passages": []}) == []


# -----------------------------------------------------------------------------
# score_record
# -----------------------------------------------------------------------------

class TestScoreRecord:
    def test_returns_max_across_passages(self):
        rec = {
            "question": "Q1",
            "answer": "Paris",
            "top_passages": ["passage-1", "passage-2", "passage-3"],
        }
        # Scores per pair in input order: 0.4, 0.9, 0.5 -> max 0.9.
        def judge_fn(pairs):
            assert pairs == [(p, "Paris") for p in rec["top_passages"]]
            return [0.4, 0.9, 0.5]

        assert score_record(rec, judge_fn) == pytest.approx(0.9)

    def test_missing_answer_returns_none(self):
        rec = {"question": "Q1", "top_passages": ["p"]}
        def judge_fn(pairs):
            raise AssertionError("judge must not be called")
        assert score_record(rec, judge_fn) is None

    def test_missing_passages_returns_none(self):
        rec = {"question": "Q1", "answer": "Paris"}
        def judge_fn(pairs):
            raise AssertionError("judge must not be called")
        assert score_record(rec, judge_fn) is None

    def test_clamps_to_unit_interval(self):
        rec = {"question": "Q1", "answer": "Paris", "top_passages": ["p"]}
        def judge_fn(pairs):
            return [1.4]  # out-of-range, must clip to 1.0
        assert score_record(rec, judge_fn) == pytest.approx(1.0)

        def judge_fn2(pairs):
            return [-0.2]
        assert score_record(rec, judge_fn2) == pytest.approx(0.0)

    def test_empty_answer_string_returns_none(self):
        rec = {"question": "Q1", "answer": "   ", "top_passages": ["p"]}
        def judge_fn(pairs):
            raise AssertionError("judge must not be called")
        assert score_record(rec, judge_fn) is None


# -----------------------------------------------------------------------------
# build_labels
# -----------------------------------------------------------------------------

class TestBuildLabels:
    def test_labels_keyed_by_question(self):
        records = [
            {"question": "Q1", "answer": "A1", "top_passages": ["p"]},
            {"question": "Q2", "answer": "A2", "top_passages": ["p"]},
        ]
        scores = {"Q1": 0.8, "Q2": 0.3}
        def judge_fn(pairs):
            assert len(pairs) == 1
            _, answer = pairs[0]
            if answer == "A1":
                return [0.8]
            return [0.3]

        labels = build_labels(records, judge_fn)
        assert labels == {"Q1": pytest.approx(0.8), "Q2": pytest.approx(0.3)}

    def test_unscoreable_records_skipped(self):
        records = [
            {"question": "Q1", "answer": "A1", "top_passages": ["p"]},
            {"question": "Q2"},  # no answer, no passages -> skipped
            {"question": "Q3", "answer": "A3"},  # no passages -> skipped
        ]
        def judge_fn(pairs):
            return [0.7]
        labels = build_labels(records, judge_fn)
        assert set(labels.keys()) == {"Q1"}

    def test_duplicate_keys_last_wins(self):
        records = [
            {"question": "Q1", "answer": "A1", "top_passages": ["p"]},
            {"question": "Q1", "answer": "A1", "top_passages": ["p"]},
        ]
        calls = {"n": 0}
        def judge_fn(pairs):
            calls["n"] += 1
            return [0.5 if calls["n"] == 1 else 0.9]
        labels = build_labels(records, judge_fn)
        assert labels == {"Q1": pytest.approx(0.9)}

    def test_missing_key_field_skipped(self):
        records = [{"answer": "A", "top_passages": ["p"]}]
        def judge_fn(pairs):
            return [0.5]
        labels = build_labels(records, judge_fn)
        assert labels == {}

    def test_custom_key_field(self):
        records = [{"qid": "abc", "answer": "A", "top_passages": ["p"]}]
        def judge_fn(pairs):
            return [0.42]
        labels = build_labels(records, judge_fn, key_field="qid")
        assert labels == {"abc": pytest.approx(0.42)}


# -----------------------------------------------------------------------------
# load_signals_jsonl
# -----------------------------------------------------------------------------

class TestLoadSignalsJSONL:
    def test_loads_valid_lines(self, tmp_path: Path):
        path = tmp_path / "signals.jsonl"
        path.write_text(
            json.dumps({"question": "Q1", "answer": "A1"}) + "\n"
            + json.dumps({"question": "Q2", "answer": "A2"}) + "\n"
        )
        rows = load_signals_jsonl(path)
        assert len(rows) == 2

    def test_skips_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "signals.jsonl"
        path.write_text(
            json.dumps({"question": "Q1"}) + "\n"
            + "{malformed\n"
            + json.dumps({"question": "Q2"}) + "\n"
        )
        rows = load_signals_jsonl(path)
        assert len(rows) == 2
