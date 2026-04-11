"""
tests/test_rag.py
=================
Unit tests for PassageStore and TierThreeRAG (Stage 6).

Mock strategy: same pattern as prior modules -- no real models or FAISS
corpora. Controlled SBERT embeddings, tiny in-memory passage arrays.

Coverage:
  - PassageStore: construction, search, size, save/load
  - PassageStore: empty store, k-clipping, score ordering
  - TierThreeRAG: _retrieve, _build_prompt, generate()
  - TierThreeRAG: fallback to query-only when no passages found
  - TierThreeRAG: public retrieve() endpoint
  - Config: rag_top_k, rag_max_new_tokens, rag_do_sample defaults
"""

from __future__ import annotations

import math
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from caem.config import CAEMConfig
from caem.retrieval.rag import PassageStore, TierThreeRAG


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

DIM = 768


def unit_vec(seed: int) -> np.ndarray:
    """Reproducible random unit vector in R^768."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make_passage_store(n: int = 5) -> tuple[PassageStore, list, np.ndarray]:
    passages   = [f"Passage number {i} about some topic." for i in range(n)]
    embeddings = np.stack([unit_vec(i) for i in range(n)])
    store      = PassageStore(passages, embeddings)
    return store, passages, embeddings


def make_mock_encoder(return_emb: np.ndarray | None = None) -> MagicMock:
    enc = MagicMock()
    if return_emb is None:
        return_emb = unit_vec(99)
    enc.encode.return_value = return_emb
    return enc


def make_mock_model() -> MagicMock:
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    # generate() returns a plain (1, seq_len) tensor
    model.generate.return_value = torch.tensor([[0, 2, 3, 1]])   # dummy token ids
    return model


def make_mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.zeros(1, 8, dtype=torch.long)}
    tok.decode.return_value = "William Shakespeare"
    return tok


def make_rag(
    n_passages: int = 5,
    query_emb_seed: int = 99,
    config: CAEMConfig | None = None,
) -> tuple[TierThreeRAG, PassageStore]:
    store, _, _ = make_passage_store(n_passages)
    encoder     = make_mock_encoder(unit_vec(query_emb_seed))
    model       = make_mock_model()
    tokenizer   = make_mock_tokenizer()
    cfg         = config or CAEMConfig()
    rag = TierThreeRAG(model, tokenizer, encoder, store, cfg, device="cpu")
    return rag, store


# -----------------------------------------------------------------------------
# PassageStore construction
# -----------------------------------------------------------------------------

class TestPassageStoreConstruction:
    def test_size_matches_input(self):
        store, passages, _ = make_passage_store(7)
        assert store.size == 7

    def test_mismatched_lengths_raise(self):
        passages   = ["a", "b", "c"]
        embeddings = np.stack([unit_vec(i) for i in range(5)])   # wrong count
        with pytest.raises(ValueError, match="passages length"):
            PassageStore(passages, embeddings)

    def test_wrong_dim_raises(self):
        passages   = ["a"]
        embeddings = np.ones((1, 384), dtype=np.float32)          # wrong dim
        with pytest.raises(ValueError, match="768"):
            PassageStore(passages, embeddings)

    def test_empty_store_allowed(self):
        store = PassageStore([], np.empty((0, DIM), dtype=np.float32))
        assert store.size == 0


# -----------------------------------------------------------------------------
# PassageStore search
# -----------------------------------------------------------------------------

class TestPassageStoreSearch:
    def test_returns_k_results(self):
        store, _, _ = make_passage_store(10)
        results = store.search(unit_vec(50), k=3)
        assert len(results) == 3

    def test_returns_tuples_of_str_and_float(self):
        store, _, _ = make_passage_store(5)
        results = store.search(unit_vec(0), k=2)
        for passage, score in results:
            assert isinstance(passage, str)
            assert isinstance(score, float)

    def test_scores_in_descending_order(self):
        store, _, _ = make_passage_store(10)
        results = store.search(unit_vec(7), k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_match_scores_near_one(self):
        """Searching with the same embedding as a stored passage -> score ≈ 1.0."""
        emb = unit_vec(0)
        passages   = ["exact match passage"]
        embeddings = emb.reshape(1, DIM)
        store = PassageStore(passages, embeddings)
        results = store.search(emb, k=1)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)
        assert results[0][0] == "exact match passage"

    def test_k_clipped_to_store_size(self):
        """Asking for k > store size returns store.size results."""
        store, _, _ = make_passage_store(3)
        results = store.search(unit_vec(5), k=100)
        assert len(results) == 3

    def test_empty_store_returns_empty(self):
        store = PassageStore([], np.empty((0, DIM), dtype=np.float32))
        results = store.search(unit_vec(0), k=5)
        assert results == []

    def test_returned_passages_are_subset_of_corpus(self):
        store, passages, _ = make_passage_store(8)
        results = store.search(unit_vec(3), k=4)
        for p, _ in results:
            assert p in passages


# -----------------------------------------------------------------------------
# PassageStore save / load
# -----------------------------------------------------------------------------

class TestPassageStorePersistence:
    def test_save_and_load_roundtrip(self):
        store, passages, embeddings = make_passage_store(6)
        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            loaded = PassageStore.load(tmpdir)

        assert loaded.size == store.size
        assert loaded.passages == passages

    def test_search_after_load_consistent(self):
        store, _, _ = make_passage_store(5)
        q = unit_vec(42)
        original_results = store.search(q, k=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)
            loaded = PassageStore.load(tmpdir)

        loaded_results = loaded.search(q, k=3)
        assert len(loaded_results) == len(original_results)
        for (p1, s1), (p2, s2) in zip(original_results, loaded_results):
            assert p1 == p2
            assert math.isclose(s1, s2, abs_tol=1e-4)


# -----------------------------------------------------------------------------
# TierThreeRAG internal helpers
# -----------------------------------------------------------------------------

class TestRAGInternals:
    def test_retrieve_returns_list(self):
        rag, _ = make_rag()
        results = rag._retrieve("Who wrote Hamlet?", k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_retrieve_calls_encoder(self):
        rag, _ = make_rag()
        rag._retrieve("test query", k=2)
        rag.passage_encoder.encode.assert_called_once_with("test query")

    def test_retrieve_error_returns_empty(self):
        rag, _ = make_rag()
        rag.passage_encoder.encode.side_effect = RuntimeError("encoder fail")
        results = rag._retrieve("q", k=5)
        assert results == []

    def test_build_prompt_contains_query(self):
        rag, _ = make_rag()
        passages = [("Paris is the capital of France.", 0.9),
                    ("France is in Western Europe.", 0.8)]
        prompt = rag._build_prompt("What is the capital of France?", passages)
        assert "What is the capital of France?" in prompt
        assert "Answer:" in prompt

    def test_build_prompt_numbered_passages(self):
        rag, _ = make_rag()
        passages = [("First passage.", 0.9), ("Second passage.", 0.7)]
        prompt = rag._build_prompt("q?", passages)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "First passage." in prompt
        assert "Second passage." in prompt

    def test_build_prompt_context_header(self):
        rag, _ = make_rag()
        prompt = rag._build_prompt("q?", [("p.", 0.5)])
        assert prompt.startswith("Context:")

    def test_tokenize_prompt_returns_tensor(self):
        rag, _ = make_rag()
        ids = rag._tokenize_prompt("Some prompt text.")
        assert isinstance(ids, torch.Tensor)


# -----------------------------------------------------------------------------
# TierThreeRAG.generate()
# -----------------------------------------------------------------------------

class TestRAGGenerate:
    def test_generate_returns_string(self):
        rag, _ = make_rag()
        answer = rag.generate("Who wrote Hamlet?")
        assert isinstance(answer, str)

    def test_generate_calls_model(self):
        rag, _ = make_rag()
        rag.generate("Who wrote Hamlet?")
        assert rag.model.generate.called

    def test_generate_uses_config_max_new_tokens(self):
        cfg = CAEMConfig()
        cfg.rag_max_new_tokens = 64
        rag, _ = make_rag(config=cfg)
        rag.generate("q?")
        call_kwargs = rag.model.generate.call_args.kwargs
        assert call_kwargs.get("max_new_tokens") == 64

    def test_generate_greedy_by_default(self):
        rag, _ = make_rag()
        rag.generate("q?")
        call_kwargs = rag.model.generate.call_args.kwargs
        assert call_kwargs.get("do_sample") is False

    def test_generate_fallback_on_model_error(self):
        """If model.generate raises, return empty string (don't crash)."""
        rag, _ = make_rag()
        rag.model.generate.side_effect = RuntimeError("OOM")
        answer = rag.generate("q?")
        assert answer == ""

    def test_generate_no_passages_uses_query_only(self):
        """Empty passage store -> falls back to query-only generation gracefully."""
        store  = PassageStore([], np.empty((0, DIM), dtype=np.float32))
        enc    = make_mock_encoder(unit_vec(0))
        model  = make_mock_model()
        tok    = make_mock_tokenizer()
        rag    = TierThreeRAG(model, tok, enc, store, CAEMConfig(), device="cpu")
        answer = rag.generate("q?")
        assert isinstance(answer, str)   # graceful, not crash

    def test_generate_uses_top_k_passages(self):
        """model.generate is called once -- retrieval happened beforehand."""
        cfg = CAEMConfig()
        cfg.rag_top_k = 3
        rag, _ = make_rag(n_passages=10, config=cfg)
        rag.generate("q?")
        # generate called exactly once
        assert rag.model.generate.call_count == 1


# -----------------------------------------------------------------------------
# TierThreeRAG.retrieve() public endpoint
# -----------------------------------------------------------------------------

class TestRAGPublicRetrieve:
    def test_retrieve_returns_passage_score_pairs(self):
        rag, _ = make_rag()
        results = rag.retrieve("Who wrote Hamlet?")
        assert isinstance(results, list)
        for p, s in results:
            assert isinstance(p, str)
            assert isinstance(s, float)

    def test_retrieve_respects_k_override(self):
        rag, _ = make_rag(n_passages=10)
        results = rag.retrieve("q?", k=2)
        assert len(results) == 2

    def test_retrieve_default_k_from_config(self):
        cfg = CAEMConfig()
        cfg.rag_top_k = 4
        rag, _ = make_rag(n_passages=10, config=cfg)
        results = rag.retrieve("q?")
        assert len(results) == 4


# -----------------------------------------------------------------------------
# Config defaults
# -----------------------------------------------------------------------------

class TestRAGConfig:
    def test_rag_top_k_default(self):
        assert CAEMConfig().rag_top_k == 5

    def test_rag_max_new_tokens_default(self):
        assert CAEMConfig().rag_max_new_tokens == 256

    def test_rag_do_sample_false_by_default(self):
        assert CAEMConfig().rag_do_sample is False

    def test_rag_max_context_tokens_default(self):
        assert CAEMConfig().rag_max_context_tokens == 384
