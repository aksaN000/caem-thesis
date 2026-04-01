"""
tests/test_self_improvement.py
================================
Unit tests for SelfImprovementLoop (Stage 8) and QADataset.

Mock strategy: lightweight mocks for the model and tokenizer so no
actual Flan-T5 weights are downloaded. The training loop runs with
a real optimizer but tiny batches and zero epochs where possible.

Coverage:
  - QAPair and CycleResult dataclasses
  - QADataset: tokenisation, padding, label masking
  - _collect_episodes: quality threshold filter
  - _mix: ratio maths, shuffle, clipping to available general data
  - _l2_penalty: zero when θ unchanged, positive when changed
  - _snapshot_weights / _restore_weights: round-trip fidelity
  - _forgetting_score: exact-match counting
  - run_cycle: no-episode early exit, normal flow, abort-on-forgetting
  - Checkpoint save / load round-trip
  - all_entries() on EpisodicMemoryStore
"""

from __future__ import annotations

import math
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from caem.config import CAEMConfig
from caem.memory.entry import EpisodicEntry
from caem.memory.store import EpisodicMemoryStore
from caem.training.self_improvement import (
    CycleResult,
    QADataset,
    QAPair,
    SelfImprovementLoop,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DIM = 768


def unit_emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make_entry(question: str = "Q?", answer: str = "A", u_stored: float = 0.80) -> EpisodicEntry:
    return EpisodicEntry(
        question=question,
        reasoning_chain="Because.",
        answer=answer,
        embedding=unit_emb(),
        storage_cycle=1,
        timestamp=0.0,
        u_stored=u_stored,
        nli_score=0.9,
        sc_score=0.8,
        se_score=0.7,
        retrieval_count=0,
        success_rate=0.0,
        retroverified=False,
    )


def make_store_with_entries(entries) -> EpisodicMemoryStore:
    store = EpisodicMemoryStore()
    for e in entries:
        store.add(e)
    return store


class _DictWithTo(dict):
    """dict subclass with .to(device) so tokenizer output works in _forgetting_score."""
    def to(self, device):
        return self


def make_mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    # Return a _DictWithTo so .to(device) works in _forgetting_score
    tok.return_value = _DictWithTo({
        "input_ids":      torch.zeros(1, 8, dtype=torch.long),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
    })
    tok.decode.return_value = "Paris"
    # as_target_tokenizer context manager
    tok.as_target_tokenizer.return_value.__enter__ = lambda s: tok
    tok.as_target_tokenizer.return_value.__exit__ = MagicMock(return_value=False)
    return tok


def make_mock_model(n_params: int = 3) -> MagicMock:
    """Model with real (tiny) parameter tensors so L2 penalty can be computed."""
    model = MagicMock()
    params = [torch.ones(4, requires_grad=True) for _ in range(n_params)]
    # Use side_effect so each call returns a fresh iterator (not exhausted after first use)
    model.parameters.side_effect = lambda: iter(params)
    model.named_parameters.side_effect = lambda: iter(
        [(f"p{i}", p) for i, p in enumerate(params)]
    )
    # state_dict / load_state_dict for checkpoint save/load
    model.state_dict.return_value = {f"p{i}": p.data for i, p in enumerate(params)}
    # model() forward returns an object with .loss
    model.return_value = SimpleNamespace(loss=torch.tensor(0.5, requires_grad=True))
    # generate(): return a (1,3) tensor
    model.generate.return_value = torch.tensor([[0, 2, 1]])
    return model


def make_loop(
    n_params: int = 3,
    config: CAEMConfig | None = None,
    output_dir: str | None = None,
) -> tuple[SelfImprovementLoop, str]:
    cfg = config or CAEMConfig()
    cfg.epochs_per_cycle = 1    # fast in tests
    cfg.batch_size = 2
    tmpdir = output_dir or tempfile.mkdtemp()
    model  = make_mock_model(n_params)
    tok    = make_mock_tokenizer()
    loop   = SelfImprovementLoop(model, tok, cfg, device="cpu", output_dir=tmpdir)
    return loop, tmpdir


def make_general_data(n: int = 10) -> list:
    return [QAPair(question=f"Q{i}?", answer=f"A{i}") for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# QAPair and CycleResult
# ─────────────────────────────────────────────────────────────────────────────

class TestDataclasses:
    def test_qa_pair_fields(self):
        p = QAPair(question="Who?", answer="Me.")
        assert p.question == "Who?"
        assert p.answer   == "Me."

    def test_cycle_result_fields(self):
        r = CycleResult(
            cycle_num=1, n_episodes_used=50, n_general_used=5,
            epochs_completed=3, final_train_loss=0.42,
            forgetting_score=0.96, aborted=False, checkpoint_path="/tmp/c1",
        )
        assert r.cycle_num == 1
        assert r.aborted is False
        assert r.forgetting_score == pytest.approx(0.96)


# ─────────────────────────────────────────────────────────────────────────────
# QADataset
# ─────────────────────────────────────────────────────────────────────────────

class TestQADataset:
    def test_len(self):
        pairs = [QAPair("Q?", "A"), QAPair("Q2?", "A2")]
        ds = QADataset(pairs, make_mock_tokenizer())
        assert len(ds) == 2

    def test_getitem_keys(self):
        pairs = [QAPair("Who wrote Hamlet?", "Shakespeare")]
        ds = QADataset(pairs, make_mock_tokenizer())
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_labels_pad_replaced_with_minus100(self):
        """Padding tokens in labels must be -100 for cross-entropy to ignore them."""
        tok = make_mock_tokenizer()
        tok.pad_token_id = 0
        # Labels will be all zeros (pad) after tokenisation → all should be -100
        pairs = [QAPair("Q?", "A")]
        ds = QADataset(pairs, tok)
        item = ds[0]
        # At least some labels should be -100 (padding positions)
        assert (item["labels"] == -100).any()

    def test_tensors_are_1d(self):
        ds = QADataset([QAPair("Q?", "A")], make_mock_tokenizer())
        item = ds[0]
        assert item["input_ids"].dim() == 1
        assert item["labels"].dim() == 1


# ─────────────────────────────────────────────────────────────────────────────
# _collect_episodes
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectEpisodes:
    def test_filters_by_threshold(self):
        cfg = CAEMConfig()
        cfg.min_u_stored_for_training = 0.75
        loop, _ = make_loop(config=cfg)
        store = make_store_with_entries([
            make_entry("Q1", u_stored=0.80),   # above → include
            make_entry("Q2", u_stored=0.60),   # below → exclude
            make_entry("Q3", u_stored=0.75),   # exact → include
        ])
        pairs = loop._collect_episodes(store)
        assert len(pairs) == 2

    def test_empty_store_returns_empty(self):
        loop, _ = make_loop()
        store = EpisodicMemoryStore()
        assert loop._collect_episodes(store) == []

    def test_returns_qa_pairs(self):
        """_collect_episodes uses reasoning_chain as the QAPair answer target.

        Thesis §4.3: training target is the verified reasoning chain, not the
        short answer string. This teaches the model *how* to reason, not just
        what the final answer is.
        """
        loop, _ = make_loop()
        store = make_store_with_entries([make_entry("Who?", "Me", u_stored=0.9)])
        pairs = loop._collect_episodes(store)
        assert isinstance(pairs[0], QAPair)
        assert pairs[0].question == "Who?"
        # The answer field carries reasoning_chain ("Because."), not entry.answer ("Me")
        assert pairs[0].answer == "Because."


# ─────────────────────────────────────────────────────────────────────────────
# _mix
# ─────────────────────────────────────────────────────────────────────────────

class TestMix:
    def test_general_ratio_approx(self):
        """With ratio=0.10 and 90 episodes, expect ~10 general pairs."""
        cfg = CAEMConfig()
        cfg.general_data_ratio = 0.10
        loop, _ = make_loop(config=cfg)
        episodes = [QAPair(f"Q{i}", f"A{i}") for i in range(90)]
        general  = [QAPair(f"G{i}", f"GA{i}") for i in range(100)]
        combined, n_general = loop._mix(episodes, general)
        assert math.isclose(n_general, 10, abs_tol=2)
        assert len(combined) == len(episodes) + n_general

    def test_clips_to_available_general(self):
        """If fewer general pairs exist than needed, use all available."""
        cfg = CAEMConfig()
        cfg.general_data_ratio = 0.50   # high ratio → needs many general
        loop, _ = make_loop(config=cfg)
        episodes = [QAPair(f"Q{i}", f"A{i}") for i in range(100)]
        general  = [QAPair("G0", "GA0")]   # only 1 available
        combined, n_general = loop._mix(episodes, general)
        assert n_general == 1

    def test_combined_contains_both(self):
        # Need enough episodes that ratio=0.10 rounds to ≥1 general pair.
        # n_general = round(n_ep * 0.10 / 0.90): need n_ep >= 9 for ≥1.
        loop, _ = make_loop()
        episodes = [QAPair(f"E{i}", f"ea{i}") for i in range(18)]
        general  = [QAPair("G", "ga")] * 10
        combined, n_general = loop._mix(episodes, general)
        assert n_general >= 1
        questions = {p.question for p in combined}
        assert any(q.startswith("E") for q in questions)
        assert "G" in questions


# ─────────────────────────────────────────────────────────────────────────────
# _snapshot_weights / _restore_weights
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightManagement:
    def _real_model_and_loop(self):
        """Use a real nn.Linear so parameter tensors are proper."""
        linear = torch.nn.Linear(4, 2, bias=False)
        cfg = CAEMConfig()
        tok = make_mock_tokenizer()
        with tempfile.TemporaryDirectory() as d:
            loop = SelfImprovementLoop(linear, tok, cfg, device="cpu", output_dir=d)
            return loop, linear

    def test_snapshot_clones_all_params(self):
        loop, model = self._real_model_and_loop()
        snap = loop._snapshot_weights()
        assert len(snap) == sum(1 for _ in model.parameters())

    def test_restore_reverts_mutation(self):
        loop, model = self._real_model_and_loop()
        snap = loop._snapshot_weights()
        # Mutate model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        # Restore
        loop._restore_weights(snap)
        # Should match original values
        for p, p0 in zip(model.parameters(), snap):
            assert torch.allclose(p.data, p0)

    def test_snapshot_is_independent_copy(self):
        """Mutating the model after snapshot should not affect the snapshot."""
        loop, model = self._real_model_and_loop()
        snap = loop._snapshot_weights()
        original_val = snap[0].clone()
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        assert torch.allclose(snap[0], original_val)


# ─────────────────────────────────────────────────────────────────────────────
# _l2_penalty
# ─────────────────────────────────────────────────────────────────────────────

class TestL2Penalty:
    def test_zero_when_weights_unchanged(self):
        linear = torch.nn.Linear(4, 2, bias=False)
        cfg = CAEMConfig()
        tok = make_mock_tokenizer()
        with tempfile.TemporaryDirectory() as d:
            loop = SelfImprovementLoop(linear, tok, cfg, device="cpu", output_dir=d)
            snap = loop._snapshot_weights()
            penalty = loop._l2_penalty(snap)
            assert penalty.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_after_weight_change(self):
        linear = torch.nn.Linear(4, 2, bias=False)
        cfg = CAEMConfig()
        tok = make_mock_tokenizer()
        with tempfile.TemporaryDirectory() as d:
            loop = SelfImprovementLoop(linear, tok, cfg, device="cpu", output_dir=d)
            snap = loop._snapshot_weights()
            with torch.no_grad():
                for p in linear.parameters():
                    p.fill_(5.0)
            penalty = loop._l2_penalty(snap)
            assert penalty.item() > 0.0

    def test_penalty_increases_with_deviation(self):
        linear = torch.nn.Linear(4, 2, bias=False)
        cfg = CAEMConfig()
        tok = make_mock_tokenizer()
        with tempfile.TemporaryDirectory() as d:
            loop = SelfImprovementLoop(linear, tok, cfg, device="cpu", output_dir=d)
            snap = loop._snapshot_weights()
            with torch.no_grad():
                for p in linear.parameters():
                    p.fill_(1.0)
            p_small = loop._l2_penalty(snap).item()
            with torch.no_grad():
                for p in linear.parameters():
                    p.fill_(10.0)
            p_large = loop._l2_penalty(snap).item()
            assert p_large > p_small


# ─────────────────────────────────────────────────────────────────────────────
# _forgetting_score
# ─────────────────────────────────────────────────────────────────────────────

class TestForgettingScore:
    def test_empty_eval_returns_one(self):
        loop, _ = make_loop()
        score = loop._forgetting_score([])
        assert score == pytest.approx(1.0)

    def test_all_correct_returns_one(self):
        loop, _ = make_loop()
        # tokenizer.decode returns "Paris" by default
        # Make answer match exactly
        pairs = [QAPair("Q?", "Paris")]
        score = loop._forgetting_score(pairs)
        assert score == pytest.approx(1.0)

    def test_none_correct_returns_zero(self):
        loop, _ = make_loop()
        # tokenizer.decode returns "Paris" but answer is different
        pairs = [QAPair("Q?", "London"), QAPair("Q2?", "Berlin")]
        score = loop._forgetting_score(pairs)
        assert score == pytest.approx(0.0)

    def test_partial_correct(self):
        loop, _ = make_loop()
        tok = loop.tokenizer
        decode_responses = ["Paris", "London", "Paris"]
        tok.decode.side_effect = decode_responses
        pairs = [QAPair("Q1?", "Paris"), QAPair("Q2?", "Paris"), QAPair("Q3?", "Berlin")]
        score = loop._forgetting_score(pairs)
        # "Paris"=="Paris" → correct, "London"=="Paris" → wrong, "Paris"=="Berlin" → wrong
        assert score == pytest.approx(1/3, abs=0.01)

    def test_score_in_unit_interval(self):
        loop, _ = make_loop()
        pairs = [QAPair(f"Q{i}?", "Paris") for i in range(5)]
        score = loop._forgetting_score(pairs)
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# run_cycle
# ─────────────────────────────────────────────────────────────────────────────

class TestRunCycle:
    def test_no_episodes_exits_early(self):
        """If no episodes meet the threshold, returns a skipped CycleResult."""
        cfg = CAEMConfig()
        cfg.min_u_stored_for_training = 0.99  # nothing meets this
        loop, _ = make_loop(config=cfg)
        store = make_store_with_entries([make_entry(u_stored=0.50)])
        result = loop.run_cycle(1, store, make_general_data())
        assert result.n_episodes_used == 0
        assert result.aborted is False
        assert result.epochs_completed == 0

    def test_returns_cycle_result(self):
        cfg = CAEMConfig()
        cfg.epochs_per_cycle = 0   # skip actual training
        cfg.min_u_stored_for_training = 0.70
        loop, _ = make_loop(config=cfg)
        store = make_store_with_entries([make_entry(u_stored=0.80)])
        result = loop.run_cycle(1, store, make_general_data(4))
        assert isinstance(result, CycleResult)
        assert result.cycle_num == 1

    def test_aborted_when_forgetting_fails(self):
        """If forgetting_score < tolerance, run_cycle sets aborted=True."""
        cfg = CAEMConfig()
        cfg.epochs_per_cycle = 0
        cfg.forgetting_tolerance = 0.99   # very strict — will fail
        cfg.min_u_stored_for_training = 0.70
        loop, _ = make_loop(config=cfg)
        # forgetting_score: tokenizer returns "Paris", answers are "London" → 0.0
        loop.tokenizer.decode.return_value = "Paris"
        general = [QAPair("Q?", "London")]
        store = make_store_with_entries([make_entry(u_stored=0.80)])
        result = loop.run_cycle(1, store, general * 10)
        assert result.aborted is True

    def test_not_aborted_when_forgetting_passes(self):
        cfg = CAEMConfig()
        cfg.epochs_per_cycle = 0
        cfg.forgetting_tolerance = 0.0   # always passes
        cfg.min_u_stored_for_training = 0.70
        loop, _ = make_loop(config=cfg)
        store = make_store_with_entries([make_entry(u_stored=0.80)])
        result = loop.run_cycle(1, store, make_general_data(4))
        assert result.aborted is False

    def test_checkpoint_created(self):
        """run_cycle must create a checkpoint directory."""
        import os
        cfg = CAEMConfig()
        cfg.epochs_per_cycle = 0
        cfg.forgetting_tolerance = 0.0
        cfg.min_u_stored_for_training = 0.70
        with tempfile.TemporaryDirectory() as tmpdir:
            loop = SelfImprovementLoop(
                make_mock_model(), make_mock_tokenizer(), cfg,
                device="cpu", output_dir=tmpdir,
            )
            store = make_store_with_entries([make_entry(u_stored=0.80)])
            result = loop.run_cycle(1, store, make_general_data(4))
            assert os.path.exists(result.checkpoint_path)
            assert os.path.exists(os.path.join(result.checkpoint_path, "meta.pkl"))


# ─────────────────────────────────────────────────────────────────────────────
# EpisodicMemoryStore.all_entries()
# ─────────────────────────────────────────────────────────────────────────────

class TestAllEntries:
    def test_empty_store(self):
        store = EpisodicMemoryStore()
        assert store.all_entries() == []

    def test_returns_all_added_entries(self):
        entries = [make_entry(f"Q{i}") for i in range(5)]
        store = make_store_with_entries(entries)
        all_e = store.all_entries()
        assert len(all_e) == 5

    def test_returns_copies_not_live_references(self):
        """Mutations to the list returned by all_entries() should not affect the store."""
        store = make_store_with_entries([make_entry("Q1")])
        snapshot = store.all_entries()
        snapshot.clear()   # clear the returned list
        assert store.size == 1   # store unaffected
