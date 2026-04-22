"""
tests/test_adaptive_truncation_and_routing.py
==============================================
Tests for Path A (MiniCheck adaptive truncation) and Path B routing
(AdaptiveNLIJudge dispatcher). No GPU required — tests use mocks and
a lightweight T5 tokenizer instance.

Covers:
  - Path A: _format_prompt allocates premise space based on hypothesis length
  - Path A: combined prompt never exceeds 512 tokens when hypothesis ≤ 408
  - Path A: fallback behaviour when hypothesis > 408 tokens
  - Path B: AdaptiveNLIJudge routes short samples to MC, long to Qwen
  - Path B: AdaptiveNLIJudge preserves input order in output
  - Path B: stats accumulator correctly counts routing
  - NLIJudgeInterface protocol: MiniCheck/Qwen/Adaptive all conform
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def t5_tokenizer():
    """Real T5 tokenizer — cheap to load, matches what MiniCheckJudge uses."""
    try:
        from transformers import T5TokenizerFast
        return T5TokenizerFast.from_pretrained("t5-small")
    except Exception as e:
        pytest.skip(f"Cannot load T5 tokenizer: {e}")


class MockMiniCheckJudge:
    """Mock MiniCheckJudge that records calls and returns deterministic probs.

    Uses the real T5 tokenizer for length checks but fakes the model.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_premise_tokens = 450
        self.calls = []  # list of (premises, hypotheses) tuples

    def batch_entail_prob(self, premises, hypotheses):
        self.calls.append((list(premises), list(hypotheses)))
        # Return deterministic probabilities — monotonic in hypothesis length
        # so we can verify order preservation
        out = np.array(
            [0.5 + 0.01 * len(h) / 100.0 for h in hypotheses],
            dtype=np.float32,
        )
        return np.clip(out, 0, 1)

    def batch_contradict_prob(self, premises, hypotheses):
        return np.zeros(len(premises), dtype=np.float32)

    def batch_argmax_label(self, premises, hypotheses):
        return ["ENTAIL"] * len(premises)

    def entail_prob(self, p, h):
        return float(self.batch_entail_prob([p], [h])[0])
    def contradict_prob(self, p, h): return 0.0
    def argmax_label(self, p, h): return "ENTAIL"


class MockQwenJudge:
    def __init__(self):
        self.calls = []

    def batch_entail_prob(self, premises, hypotheses):
        self.calls.append((list(premises), list(hypotheses)))
        # Different range from mock MiniCheck so we can tell routes apart
        out = np.array(
            [0.8 + 0.01 * i for i in range(len(premises))],
            dtype=np.float32,
        )
        return np.clip(out, 0, 1)

    def batch_contradict_prob(self, premises, hypotheses):
        return np.zeros(len(premises), dtype=np.float32)

    def batch_argmax_label(self, premises, hypotheses):
        return ["ENTAIL"] * len(premises)

    def entail_prob(self, p, h): return float(self.batch_entail_prob([p], [h])[0])
    def contradict_prob(self, p, h): return 0.0
    def argmax_label(self, p, h): return "ENTAIL"


# ================ Path A: adaptive truncation tests =======================

def test_adaptive_truncation_short_hypothesis_preserves_premise(t5_tokenizer):
    """With short hypothesis, premise gets its full 450-token budget."""
    from caem.verification.minicheck import _MiniCheckJudge as MiniCheckJudge

    # Build a dummy judge that only uses the tokenizer (bypass model init)
    judge = MiniCheckJudge.__new__(MiniCheckJudge)
    judge.tokenizer = t5_tokenizer
    judge.max_premise_tokens = 450

    short_hyp = "The answer is Paris."  # ~6 T5 tokens
    long_premise = " ".join(["word"] * 1000)  # ~1000 tokens
    prompt = judge._format_prompt(long_premise, short_hyp)

    total = len(t5_tokenizer(prompt, add_special_tokens=False).input_ids)
    assert total <= 512, f"Prompt exceeds 512 tokens ({total})"
    # Should include both "predict:" prefix and the hypothesis
    assert "predict:" in prompt
    assert short_hyp in prompt


def test_adaptive_truncation_medium_hypothesis_trims_premise(t5_tokenizer):
    """With medium-length hypothesis, premise budget shrinks adaptively."""
    from caem.verification.minicheck import _MiniCheckJudge as MiniCheckJudge

    judge = MiniCheckJudge.__new__(MiniCheckJudge)
    judge.tokenizer = t5_tokenizer
    judge.max_premise_tokens = 450

    medium_hyp = " ".join(["tokens"] * 100)  # ~100 tokens
    long_premise = " ".join(["word"] * 1000)
    prompt = judge._format_prompt(long_premise, medium_hyp)

    total = len(t5_tokenizer(prompt, add_special_tokens=False).input_ids)
    assert total <= 512, f"Prompt exceeds 512 tokens ({total})"
    # The full hypothesis should be in the prompt (not truncated)
    assert medium_hyp in prompt


def test_adaptive_truncation_long_hypothesis_fallback(t5_tokenizer):
    """Hypothesis > 408 tokens falls back to fixed premise; may exceed 512."""
    from caem.verification.minicheck import _MiniCheckJudge as MiniCheckJudge

    judge = MiniCheckJudge.__new__(MiniCheckJudge)
    judge.tokenizer = t5_tokenizer
    judge.max_premise_tokens = 450

    # Build a hypothesis with ~500 T5 tokens
    very_long_hyp = " ".join(["tokens"] * 500)
    long_premise = " ".join(["word"] * 1000)

    # Should NOT raise; returns a prompt that's still bounded by max_premise_tokens
    prompt = judge._format_prompt(long_premise, very_long_hyp)
    assert "predict:" in prompt
    # Prompt will exceed 512 tokens here; caller (AdaptiveNLIJudge) should
    # have routed this to Qwen-judge before we got here.


# ================ Path B: AdaptiveNLIJudge routing tests ==================

def test_adaptive_routing_all_short_goes_to_mc(t5_tokenizer):
    from caem.verification.adaptive_nli_judge import AdaptiveNLIJudge

    mc = MockMiniCheckJudge(t5_tokenizer)
    qw = MockQwenJudge()
    dispatcher = AdaptiveNLIJudge(mc, qw, mc_tokenizer=t5_tokenizer)
    dispatcher.reset_stats()

    premises = ["p1", "p2", "p3"]
    hypotheses = ["short hyp 1", "short hyp 2", "short hyp 3"]
    probs = dispatcher.batch_entail_prob(premises, hypotheses)

    assert len(probs) == 3
    assert len(mc.calls) == 1
    assert len(qw.calls) == 0
    stats = dispatcher.stats()
    assert stats["mc_calls"] == 3
    assert stats["qwen_calls"] == 0


def test_adaptive_routing_long_goes_to_qwen(t5_tokenizer):
    from caem.verification.adaptive_nli_judge import AdaptiveNLIJudge

    mc = MockMiniCheckJudge(t5_tokenizer)
    qw = MockQwenJudge()
    dispatcher = AdaptiveNLIJudge(mc, qw, mc_tokenizer=t5_tokenizer)
    dispatcher.reset_stats()

    premises = ["p1"]
    # Build a hypothesis with >408 T5 tokens
    long_hyp = " ".join(["tokens"] * 500)
    hypotheses = [long_hyp]
    probs = dispatcher.batch_entail_prob(premises, hypotheses)

    assert len(probs) == 1
    assert len(mc.calls) == 0
    assert len(qw.calls) == 1
    stats = dispatcher.stats()
    assert stats["qwen_calls"] == 1


def test_adaptive_routing_mixed_preserves_order(t5_tokenizer):
    """Mixed batch: some short, some long. Outputs must preserve input order."""
    from caem.verification.adaptive_nli_judge import AdaptiveNLIJudge

    mc = MockMiniCheckJudge(t5_tokenizer)
    qw = MockQwenJudge()
    dispatcher = AdaptiveNLIJudge(mc, qw, mc_tokenizer=t5_tokenizer)

    short_hyp = "short answer"
    long_hyp = " ".join(["tokens"] * 500)
    premises = ["p0", "p1", "p2", "p3"]
    hypotheses = [short_hyp, long_hyp, short_hyp, long_hyp]
    probs = dispatcher.batch_entail_prob(premises, hypotheses)

    assert len(probs) == 4
    # Both judges should have been called
    assert len(mc.calls) == 1
    assert len(qw.calls) == 1
    # MC handled indices 0 and 2
    assert mc.calls[0][0] == ["p0", "p2"]
    assert mc.calls[0][1] == [short_hyp, short_hyp]
    # Qwen handled indices 1 and 3
    assert qw.calls[0][0] == ["p1", "p3"]
    assert qw.calls[0][1] == [long_hyp, long_hyp]


def test_adaptive_routing_no_qwen_all_to_mc(t5_tokenizer):
    """If qwen_judge=None, all samples route to MC regardless of length."""
    from caem.verification.adaptive_nli_judge import AdaptiveNLIJudge

    mc = MockMiniCheckJudge(t5_tokenizer)
    dispatcher = AdaptiveNLIJudge(mc, None, mc_tokenizer=t5_tokenizer)

    short_hyp = "short"
    long_hyp = " ".join(["tokens"] * 500)
    probs = dispatcher.batch_entail_prob(["p0", "p1"], [short_hyp, long_hyp])

    assert len(probs) == 2
    assert len(mc.calls) == 1
    # MC got both samples
    assert len(mc.calls[0][0]) == 2


def test_adaptive_empty_input(t5_tokenizer):
    from caem.verification.adaptive_nli_judge import AdaptiveNLIJudge
    mc = MockMiniCheckJudge(t5_tokenizer)
    dispatcher = AdaptiveNLIJudge(mc, MockQwenJudge(), mc_tokenizer=t5_tokenizer)
    out = dispatcher.batch_entail_prob([], [])
    assert len(out) == 0


# ================ NLIJudgeInterface protocol conformance ================

def test_mock_judges_satisfy_protocol(t5_tokenizer):
    from caem.verification.judge_interface import NLIJudgeInterface
    mc = MockMiniCheckJudge(t5_tokenizer)
    qw = MockQwenJudge()
    assert isinstance(mc, NLIJudgeInterface)
    assert isinstance(qw, NLIJudgeInterface)


def test_adaptive_judge_satisfies_protocol(t5_tokenizer):
    from caem.verification.judge_interface import NLIJudgeInterface
    from caem.verification.adaptive_nli_judge import AdaptiveNLIJudge
    dispatcher = AdaptiveNLIJudge(
        MockMiniCheckJudge(t5_tokenizer), MockQwenJudge(),
        mc_tokenizer=t5_tokenizer,
    )
    assert isinstance(dispatcher, NLIJudgeInterface)
