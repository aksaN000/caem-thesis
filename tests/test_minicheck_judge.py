"""
tests/test_minicheck_judge.py
==============================
Unit coverage for the MiniCheck adapter (caem.verification.minicheck).

Mocks the T5 + tokenizer so we exercise prompt construction, batching,
logit->probability conversion, and NLI-shaped label mapping without
pulling weights from HuggingFace. The real forward-pass quality lives
in scripts/calibration_minicheck_vs_roberta.py (GPU-only).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Sequence
from unittest.mock import MagicMock

import numpy as np
import torch

from caem.verification.minicheck import _MiniCheckJudge


def _mock_tokenizer(texts_to_ids=None):
    """Fake HF tokenizer that returns short token id lists.

    - Calls of the form ``tokenizer("1", add_special_tokens=False)`` must
      return a 1-token id list for the "supported" side; similarly for "0".
    - Batch calls return a ``SimpleNamespace`` with ``input_ids`` and
      ``attention_mask`` tensors of shape (N, L).
    - Single-call ``tokenizer(premise, add_special_tokens=False)`` returns a
      namespace with ``input_ids`` list -- used by _format_prompt.
    - ``.decode(ids, skip_special_tokens=True)`` returns a trivial string
      join so the test harness can observe truncation.
    """
    YES_ID, NO_ID = 101, 102  # arbitrary distinct ids

    def tok(*args, **kwargs):
        # Single-string premise truncation path.
        if len(args) == 1 and isinstance(args[0], str) and kwargs.get("add_special_tokens") is False:
            s = args[0]
            if s == "1":
                return SimpleNamespace(input_ids=[YES_ID])
            if s == "0":
                return SimpleNamespace(input_ids=[NO_ID])
            # Simulate 1 token per character -- caller will truncate.
            return SimpleNamespace(input_ids=list(range(200, 200 + len(s))))
        # Batched prompt path.
        if len(args) == 1 and isinstance(args[0], list):
            prompts = args[0]
            n = len(prompts)
            L = 4  # dummy length
            enc = SimpleNamespace(
                input_ids=torch.zeros((n, L), dtype=torch.long),
                attention_mask=torch.ones((n, L), dtype=torch.long),
            )
            # .to(device) should be a no-op.
            enc.to = lambda device: enc
            return enc
        raise AssertionError(f"unexpected tokenizer call: args={args} kwargs={kwargs}")

    tokenizer = MagicMock()
    tokenizer.side_effect = tok
    tokenizer.decode = lambda ids, skip_special_tokens=True: "".join(
        chr(97 + (int(i) % 26)) for i in ids
    )
    tokenizer.YES_ID = YES_ID
    tokenizer.NO_ID = NO_ID
    return tokenizer


def _mock_model(yes_logit_values: Sequence[float], no_logit_values: Sequence[float]):
    """Fake T5 model whose forward() returns logits such that softmax over
    {NO_ID, YES_ID} yields the requested two-way distribution per row.

    ``yes_logit_values[i]`` and ``no_logit_values[i]`` are the logits assigned
    to the yes / no tokens for row i. All other vocab entries are -inf so they
    contribute ~0 probability mass.
    """
    class _Out:
        def __init__(self, logits): self.logits = logits

    V = 200  # small fake vocab
    YES_ID, NO_ID = 101, 102

    def forward(input_ids, attention_mask, decoder_input_ids):
        n = input_ids.shape[0]
        # Set everything to a very low value; overwrite YES/NO per row.
        logits = torch.full((n, 1, V), -1e4)
        for i in range(n):
            logits[i, 0, YES_ID] = float(yes_logit_values[i])
            logits[i, 0, NO_ID] = float(no_logit_values[i])
        return _Out(logits)

    model = MagicMock()
    model.side_effect = forward
    model.config = SimpleNamespace(decoder_start_token_id=0)
    return model


def _make_judge(yes_logits, no_logits):
    tokenizer = _mock_tokenizer()
    model = _mock_model(yes_logits, no_logits)
    return _MiniCheckJudge(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        entail_threshold=0.7,
        contradict_threshold=0.3,
    )


def test_entail_prob_high_when_yes_logit_dominates():
    judge = _make_judge(yes_logits=[3.0], no_logits=[-3.0])
    p = judge.entail_prob("Some passage.", "A claim.")
    assert p > 0.95, f"expected high entail prob, got {p}"


def test_contradict_prob_is_zero_by_design():
    # MiniCheck conflates "refuted" with "neutral/unverifiable" under its
    # binary supported/unsupported output. Mapping 1-entail as contradict
    # fires the CAEM veto on almost every sample (see commit on 2026-04-19).
    # The correct semantic is contradict=0.0; rely on the composite's
    # p_ground_* signals to pull u_stored down for unsupported claims.
    judge = _make_judge(yes_logits=[-2.0], no_logits=[2.0])
    # Regardless of the entailment score, contradict is always 0.0.
    assert judge.contradict_prob("ctx", "claim") == 0.0
    judge2 = _make_judge(yes_logits=[3.0], no_logits=[-3.0])
    assert judge2.contradict_prob("ctx", "claim") == 0.0


def test_argmax_label_thresholds():
    # P(yes) ~ 0.88 -> ENTAIL (2)
    j = _make_judge(yes_logits=[2.0], no_logits=[0.0])
    assert j.argmax_label("p", "h") == 2
    # P(yes) ~ 0.12 -> CONTRADICT (0)
    j = _make_judge(yes_logits=[0.0], no_logits=[2.0])
    assert j.argmax_label("p", "h") == 0
    # P(yes) ~ 0.5 -> NEUTRAL (1)
    j = _make_judge(yes_logits=[0.0], no_logits=[0.0])
    assert j.argmax_label("p", "h") == 1


def test_batch_entail_prob_matches_per_row_softmax():
    yes = [3.0, -3.0, 0.0]
    no = [-3.0, 3.0, 0.0]
    judge = _make_judge(yes, no)
    pairs = [("ctx1", "c1"), ("ctx2", "c2"), ("ctx3", "c3")]
    out = judge.batch_entail_prob(pairs)
    # Reference probability is sigmoid(yes - no), i.e. softmax over [no, yes].
    expected = [1.0 / (1.0 + float(np.exp(n - y))) for y, n in zip(yes, no)]
    assert len(out) == 3
    for o, e in zip(out, expected):
        assert abs(o - e) < 1e-5


def test_batch_contradict_prob_is_all_zero():
    # Batched version of test_contradict_prob_is_zero_by_design.
    judge = _make_judge([1.0, -1.0], [-1.0, 1.0])
    pairs = [("a", "b"), ("c", "d")]
    out = judge.batch_contradict_prob(pairs)
    assert out == [0.0, 0.0]


def test_batch_argmax_label_mirrors_argmax_label():
    # Three rows spanning ENTAIL / CONTRADICT / NEUTRAL.
    yes = [3.0, -3.0, 0.0]
    no = [-3.0, 3.0, 0.0]
    judge = _make_judge(yes, no)
    pairs = [("p1", "h1"), ("p2", "h2"), ("p3", "h3")]
    out = judge.batch_argmax_label(pairs)
    assert out == [2, 0, 1]


def test_empty_pairs_returns_empty_lists():
    judge = _make_judge([], [])
    assert judge.batch_entail_prob([]) == []
    assert judge.batch_contradict_prob([]) == []
    assert judge.batch_argmax_label([]) == []


def test_bool_and_bundles_compat():
    judge = _make_judge([0.0], [0.0])
    assert bool(judge) is True
    # back-compat: bundles exposed for legacy `len(nli.bundles) > 0` probes.
    assert len(judge.bundles) == 1


def test_premise_truncation_respects_max_tokens():
    # long premise so _format_prompt triggers the truncation branch.
    judge = _make_judge([0.0], [0.0])
    # max_premise_tokens=450 by default; build a 600-char premise.
    premise = "abc" * 200
    prompt = judge._format_prompt(premise, "A short claim.")
    assert prompt.startswith("predict: ")
    # The claim must still appear after the truncation.
    assert "A short claim." in prompt
