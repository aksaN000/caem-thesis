"""Tests for caem.production.confidence."""
from __future__ import annotations

import pytest

from caem.production.confidence import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    VALID_DECISIONS,
    ProductionResponse,
    render,
    render_dict,
    user_facing_answer,
)


class TestRenderBasicDecisions:
    def test_store_renders_verified(self):
        r = render("Paris is the capital of France.", 0.85, "STORE")
        assert r is not None
        assert r.tag == "verified"
        assert r.label == "high"
        assert r.icon == "check"
        assert r.show_answer is True
        assert r.answer == "Paris is the capital of France."
        assert r.confidence == 0.85
        assert r.caveat is None
        assert r.decision == "STORE"

    def test_deferred_renders_provisional(self):
        r = render("Napoleon was born in 1769.", 0.54, "DEFERRED")
        assert r is not None
        assert r.tag == "provisional"
        assert r.label == "moderate"
        assert r.icon == "tilde"
        assert r.show_answer is True
        assert r.answer == "Napoleon was born in 1769."
        assert r.caveat is not None
        assert "reconsider" in r.caveat.lower()

    def test_discard_renders_conflicting(self):
        r = render("Treaty signed 1919.", 0.41, "DISCARD")
        assert r is not None
        assert r.tag == "conflicting"
        assert r.label == "low"
        assert r.icon == "warning"
        assert r.show_answer is True
        assert r.answer == "Treaty signed 1919."
        assert r.caveat is not None
        assert "disagree" in r.caveat.lower()

    def test_abstain_renders_insufficient_and_hides_answer(self):
        r = render("raw generated text", 0.22, "ABSTAIN")
        assert r is not None
        assert r.tag == "insufficient"
        assert r.label == "very_low"
        assert r.icon == "cross"
        assert r.show_answer is False
        assert r.answer is None  # hidden on ABSTAIN
        assert r.caveat is None  # caveat is None; user_facing_answer substitutes the message


class TestDeferredCaveatCadence:
    def test_generic_fallback_when_no_kwargs(self):
        r = render("x", 0.5, "DEFERRED")
        assert "next learning cycle" in r.caveat

    def test_named_cadence_nightly(self):
        r = render("x", 0.5, "DEFERRED", deferred_cadence="nightly")
        assert "nightly" in r.caveat
        assert "tomorrow" in r.caveat

    def test_named_cadence_hourly(self):
        r = render("x", 0.5, "DEFERRED", deferred_cadence="hourly")
        assert "hourly" in r.caveat
        assert "hour" in r.caveat

    def test_named_cadence_continuous(self):
        r = render("x", 0.5, "DEFERRED", deferred_cadence="continuous")
        assert "few minutes" in r.caveat

    def test_unknown_cadence_renders_verbatim(self):
        r = render("x", 0.5, "DEFERRED", deferred_cadence="fortnightly")
        assert "fortnightly" in r.caveat

    def test_explicit_eta_used(self):
        r = render("x", 0.5, "DEFERRED", deferred_eta="6am tomorrow")
        assert "6am tomorrow" in r.caveat

    def test_eta_beats_cadence(self):
        r = render(
            "x", 0.5, "DEFERRED",
            deferred_eta="3pm today", deferred_cadence="nightly",
        )
        assert "3pm today" in r.caveat
        assert "nightly" not in r.caveat


class TestConfidenceClamping:
    def test_clamp_above_one(self):
        r = render("x", 1.5, "STORE")
        assert r.confidence == 1.0

    def test_clamp_below_zero(self):
        r = render("x", -0.1, "ABSTAIN")
        assert r.confidence == 0.0

    def test_exact_zero(self):
        r = render("x", 0.0, "ABSTAIN")
        assert r.confidence == 0.0

    def test_exact_one(self):
        r = render("x", 1.0, "STORE")
        assert r.confidence == 1.0

    def test_midrange_passthrough(self):
        r = render("x", 0.713, "STORE")
        assert r.confidence == pytest.approx(0.713)

    def test_int_confidence_accepted(self):
        r = render("x", 1, "STORE")
        assert r.confidence == 1.0


class TestInvalidDecision:
    def test_unknown_string_returns_none(self):
        assert render("x", 0.5, "WEIRD") is None

    def test_empty_string_returns_none(self):
        assert render("x", 0.5, "") is None

    def test_lowercase_rejected(self):
        # decisions are uppercase per Stage-5 convention; no auto-normalization
        assert render("x", 0.5, "store") is None

    def test_render_dict_returns_none_for_invalid(self):
        assert render_dict("x", 0.5, "WEIRD") is None

    def test_valid_decisions_frozenset(self):
        assert VALID_DECISIONS == frozenset({"STORE", "DEFERRED", "DISCARD", "ABSTAIN"})


class TestUserFacingAnswer:
    def test_verified_returns_answer(self):
        r = render("Paris is the capital.", 0.85, "STORE")
        assert user_facing_answer(r) == "Paris is the capital."

    def test_provisional_returns_answer(self):
        r = render("Napoleon 1769.", 0.54, "DEFERRED")
        assert user_facing_answer(r) == "Napoleon 1769."

    def test_conflicting_returns_answer(self):
        r = render("Disputed fact.", 0.41, "DISCARD")
        assert user_facing_answer(r) == "Disputed fact."

    def test_insufficient_returns_substitution(self):
        r = render("leaked internal answer", 0.22, "ABSTAIN")
        assert user_facing_answer(r) == INSUFFICIENT_EVIDENCE_MESSAGE

    def test_substitution_is_not_the_raw_answer(self):
        r = render("a secret", 0.22, "ABSTAIN")
        assert "secret" not in user_facing_answer(r)


class TestRenderDict:
    def test_dict_has_all_keys(self):
        d = render_dict("x", 0.5, "STORE")
        expected = {
            "answer", "confidence", "tag", "label", "icon",
            "show_answer", "caveat", "decision",
        }
        assert set(d.keys()) == expected

    def test_dict_store_shape(self):
        d = render_dict("x", 0.9, "STORE")
        assert d["answer"] == "x"
        assert d["show_answer"] is True
        assert d["tag"] == "verified"
        assert d["caveat"] is None
        assert d["decision"] == "STORE"

    def test_dict_abstain_hides_answer(self):
        d = render_dict("x", 0.2, "ABSTAIN")
        assert d["answer"] is None
        assert d["show_answer"] is False

    def test_dict_deferred_with_cadence(self):
        d = render_dict("x", 0.5, "DEFERRED", deferred_cadence="nightly")
        assert "nightly" in d["caveat"]


class TestBoundaryAndEdgeCases:
    def test_empty_answer_preserved(self):
        r = render("", 0.5, "STORE")
        assert r.answer == ""
        assert r.tag == "verified"

    def test_long_answer_preserved(self):
        long = "x" * 10_000
        r = render(long, 0.5, "STORE")
        assert r.answer == long

    def test_unicode_answer_preserved(self):
        r = render("café résumé naïve 한국어 日本語", 0.6, "STORE")
        assert r.answer == "café résumé naïve 한국어 日本語"

    def test_newlines_preserved(self):
        r = render("line1\nline2", 0.6, "STORE")
        assert "\n" in r.answer

    def test_dataclass_is_frozen(self):
        r = render("x", 0.5, "STORE")
        with pytest.raises((AttributeError, Exception)):
            r.confidence = 0.99  # type: ignore[misc]
