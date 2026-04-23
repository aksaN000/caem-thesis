"""Tests for the FastAPI demo server. Mocks the pipeline — does not
load models, so runs in under a second in CI."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest


def _fake_pipeline_result(
    *, decision: str, u_stored: float, answer: str = "Paris is the capital.",
    tier: int = 3,
):
    """Synthesize a minimal PipelineResult-like object for tests."""
    vout = SimpleNamespace(
        decision=decision,
        u_stored=u_stored,
        p_ground_max=0.8,
        p_ground_mean=0.7,
        p_ground_atomic=0.6,
        p_entail=0.75,
        u_internal=0.3,
        u_dropout=0.25,
        s_avg=0.5,
    )
    result = SimpleNamespace(
        answer=answer,
        display_answer=answer,
        tier=tier,
        stored=(decision == "STORE"),
        latency_ms=123.0,
        escalated=False,
        verifier_output=vout,
    )
    return result


class FakePipeline:
    def __init__(self, next_result):
        self._next = next_result

    def answer(self, question: str, **_kwargs):
        return self._next

    @property
    def current_cycle(self) -> int:
        return 3


class FakeMemoryStore:
    def __init__(self, size: int = 42):
        self._size = size

    def __len__(self) -> int:
        return self._size


@pytest.fixture
def client_store(monkeypatch):
    """Install a fake pipeline and return a TestClient + the store."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import scripts.caem_demo_server as srv

    fake_pipe = FakePipeline(
        _fake_pipeline_result(decision="STORE", u_stored=0.85),
    )
    fake_store = FakeMemoryStore(size=42)
    srv._set_pipeline(fake_pipe, fake_store, default_cadence="nightly")

    app = srv.make_app()
    client = TestClient(app)
    return client, srv, fake_pipe, fake_store


class TestHealthAndStats:
    def test_health_ok_when_loaded(self, client_store):
        client, *_ = client_store
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["memory_size"] == 42
        assert body["default_cadence"] == "nightly"

    def test_stats_returns_memory_info(self, client_store):
        client, *_ = client_store
        r = client.get("/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["memory_size"] == 42
        assert body["current_cycle"] == 3

    def test_root_returns_html(self, client_store):
        client, *_ = client_store
        r = client.get("/")
        assert r.status_code == 200
        assert "CAEM" in r.text
        assert "/query" in r.text


class TestQueryEndpoint:
    def test_store_returns_verified(self, client_store):
        client, *_ = client_store
        r = client.post("/query", json={"question": "Who wrote Hamlet?"})
        assert r.status_code == 200
        body = r.json()
        pr = body["production_response"]
        assert pr["tag"] == "verified"
        assert pr["show_answer"] is True
        assert pr["confidence"] == pytest.approx(0.85)
        assert body["tier"] == 3
        assert body["stored"] is True
        assert body["explanation"]["p_ground_max"] == pytest.approx(0.8)

    def test_deferred_default_cadence_used(self, client_store):
        client, srv, *_ = client_store
        srv._set_pipeline(
            FakePipeline(_fake_pipeline_result(decision="DEFERRED", u_stored=0.52)),
            FakeMemoryStore(42), default_cadence="nightly",
        )
        r = client.post("/query", json={"question": "x"})
        pr = r.json()["production_response"]
        assert pr["tag"] == "provisional"
        assert "nightly" in pr["caveat"]

    def test_deferred_request_cadence_overrides_default(self, client_store):
        client, srv, *_ = client_store
        srv._set_pipeline(
            FakePipeline(_fake_pipeline_result(decision="DEFERRED", u_stored=0.52)),
            FakeMemoryStore(42), default_cadence="nightly",
        )
        r = client.post("/query", json={
            "question": "x", "deferred_cadence": "hourly",
        })
        pr = r.json()["production_response"]
        assert "hourly" in pr["caveat"]
        assert "nightly" not in pr["caveat"]

    def test_abstain_hides_answer(self, client_store):
        client, srv, *_ = client_store
        srv._set_pipeline(
            FakePipeline(_fake_pipeline_result(
                decision="ABSTAIN", u_stored=0.22,
                answer="leaked internal",
            )),
            FakeMemoryStore(42), default_cadence=None,
        )
        r = client.post("/query", json={"question": "x"})
        pr = r.json()["production_response"]
        assert pr["tag"] == "insufficient"
        assert pr["show_answer"] is False
        assert pr["answer"] is None

    def test_tier1_hit_synthesizes_verified(self, client_store):
        client, srv, *_ = client_store
        # Tier 1 hit: verifier_output is None
        tier1_result = SimpleNamespace(
            answer="From memory", display_answer="From memory",
            tier=1, stored=False, latency_ms=180.0, escalated=False,
            verifier_output=None,
        )
        srv._set_pipeline(
            FakePipeline(tier1_result), FakeMemoryStore(42),
            default_cadence=None,
        )
        r = client.post("/query", json={"question": "x"})
        body = r.json()
        pr = body["production_response"]
        assert pr["tag"] == "verified"
        assert body["tier"] == 1
        assert body["explanation"]["source"] == "memory_hit"


class TestNotLoaded:
    def test_query_503_when_pipeline_absent(self, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        import scripts.caem_demo_server as srv
        srv._set_pipeline(None, None, default_cadence=None)
        app = srv.make_app()
        client = TestClient(app)
        r = client.post("/query", json={"question": "x"})
        assert r.status_code == 503
