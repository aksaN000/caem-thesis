"""
tests/test_flare_smoke.py
=========================
Tier-A smoke test for FLAREBaseline — verifies the decoder-slicing fix
without requiring a DPR passage index.

Motivation
----------
The B5 FLARE look-ahead previously sliced ``out.sequences[0, input_ids.shape[1]:]``
as though the model were decoder-only (GPT-family). T5 is encoder-decoder:
``model.generate()`` returns only decoder tokens, with position 0 being the
decoder-start token and positions 1..T being the generated tokens. The
decoder-only slicing yields an empty tensor and causes every FLARE answer
to come back as ``""``. The fix on line 432 of ``eval/baselines.py`` replaced
the slice with ``out.sequences[0, 1:]``.

This smoke test is a regression guard against that specific bug. It:

1. Loads Flan-T5-Large on whatever device is available (CUDA preferred,
   CPU fallback).
2. Bypasses the PassageStore/FAISS/DPR-index dependency by constructing
   ``FLAREBaseline`` via ``object.__new__`` and injecting a mock ``self.rag``
   that returns a canned passage list. This lets the test run on a laptop
   or a 12 GB 3060 without the 30–60 GB Wikipedia passage index that the
   full Phase-1 pipeline requires.
3. Runs a 5-query FEVER-shaped smoke set and asserts ``empty_answer == 0``.
4. Logs the ``escalated`` count as diagnostic output — not asserted,
   because with mocked passages the retrieval branch's quality is
   meaningless; the real threshold-calibration test belongs on the Vast
   GPU run against the full DPR index.

Runtime
-------
On an RTX 3060 12 GB in FP16: ~30–60 s wall-clock after one-time model
download (~3 GB from HuggingFace cache). On CPU: ~3–5 min.

Skip policy
-----------
The test is gated behind the ``RUN_FLARE_SMOKE=1`` environment variable
so it does not run on every ``pytest`` invocation. Invoke explicitly:

    RUN_FLARE_SMOKE=1 pytest tests/test_flare_smoke.py -v -s
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from eval.baselines import FLAREBaseline


# Gate the test behind an explicit env flag so it does not run in the
# default pytest pass (which downloads the Flan-T5-Large checkpoint and
# occupies GPU memory for tens of seconds).
_RUN = os.environ.get("RUN_FLARE_SMOKE", "0") == "1"
pytestmark = pytest.mark.skipif(
    not _RUN,
    reason="Set RUN_FLARE_SMOKE=1 to run the FLARE decoder-slice smoke test.",
)


# Five hand-crafted FEVER-shaped claims covering the expected confidence
# spectrum: one obviously true, one obviously false, one specialised, one
# numerically adversarial, and one mis-named public figure. The smoke test
# does not depend on getting the answer *correct*; it depends on getting
# a non-empty answer back and on at least one query triggering the FLARE
# retrieval branch (diagnostic, not asserted).
FEVER_SMOKE_QUERIES = [
    "Claim: Mount Everest is the tallest mountain on Earth.",
    "Claim: The capital of Australia is Sydney.",
    "Claim: Water boils at 100 degrees Celsius at sea level.",
    "Claim: The Great Wall of China is visible from the Moon with the naked eye.",
    "Claim: Albert Einstein developed the special theory of relativity in 1905.",
]


class MockTierThreeRAG:
    """Minimal stand-in for ``caem.retrieval.rag.TierThreeRAG`` exposing only
    the two methods FLAREBaseline._generate calls. Real retrieval requires
    a DPR passage store and a FAISS index; this mock returns a fixed
    placeholder passage so the slicing fix can be tested in isolation."""

    def retrieve(self, query: str):  # noqa: D401
        return [
            {
                "title": "mock_passage",
                "text": (
                    "[This is a mocked passage. The FLARE smoke test does "
                    "not exercise real retrieval; it only exercises the "
                    "decoder-slicing fix.]"
                ),
                "score": 0.0,
            }
        ]

    def _build_prompt(self, query: str, passages) -> str:
        ctx = passages[0]["text"] if passages else ""
        return f"Context: {ctx}\n\n{query}\n\nAnswer:"


@pytest.fixture(scope="module")
def flare_mocked() -> FLAREBaseline:
    """Build a FLAREBaseline whose ``.rag`` is a mock, without running
    ``RAGBaseline.__init__`` (which would require a real PassageStore).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_name = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    model.eval()

    # Bypass RAGBaseline.__init__ — we don't have a PassageStore on the
    # laptop GPU, and we don't want to load 60 GB of DPR index just to
    # test the decoder slice.
    flare = object.__new__(FLAREBaseline)
    flare.name = "flare"
    flare.tier_value = 3
    flare.model_name = model_name
    flare.device = device
    flare.tokenizer = tokenizer
    flare.model = model
    flare.max_input_tokens = 512
    flare.max_new_tokens = 256
    flare.theta = 0.4
    flare.look_ahead_tokens = 64
    flare.max_sentences = 8
    flare.rag = MockTierThreeRAG()
    return flare


def test_flare_look_ahead_produces_tokens(flare_mocked: FLAREBaseline) -> None:
    """Direct test of ``_look_ahead``: the decoder-slice fix should yield a
    non-empty ``gen_ids`` tensor and therefore non-empty decoded text for
    at least one query in the smoke set.
    """
    for query in FEVER_SMOKE_QUERIES:
        prompt = f"{query}\n\nAnswer: "
        text, min_prob = flare_mocked._look_ahead(prompt)
        # The core regression: if the slicing bug returns, gen_ids is empty
        # and text == "" with min_prob == 1.0 from the fallback branch.
        # Fail loudly on the first empty-text case.
        if not text.strip():
            pytest.fail(
                f"FLARE _look_ahead returned empty text for query "
                f"{query!r}: decoder-slice regression suspected. "
                f"min_prob={min_prob}."
            )


def test_flare_smoke_non_empty_answers(flare_mocked: FLAREBaseline) -> None:
    """Five-query smoke test mirroring the NEXT_SESSION_PLAN pre-flight
    block. Asserts every query produces a non-empty answer; logs (but does
    not assert) how many triggered the retrieval branch.
    """
    empty_count = 0
    escalated_count = 0

    for i, query in enumerate(FEVER_SMOKE_QUERIES):
        answer, tier, escalated = flare_mocked._generate(query)
        if not answer.strip():
            empty_count += 1
        if escalated:
            escalated_count += 1
        print(
            f"[FLARE smoke {i + 1}/{len(FEVER_SMOKE_QUERIES)}] "
            f"tier={tier} escalated={escalated} "
            f"answer={answer[:80]!r}"
        )

    # The hard regression guard: zero empty answers.
    assert empty_count == 0, (
        f"FLAREBaseline produced {empty_count} empty answers across "
        f"{len(FEVER_SMOKE_QUERIES)} smoke queries; decoder-slice fix "
        f"may have regressed."
    )

    # Diagnostic only. The real escalation-rate calibration happens on
    # the Vast run with the real DPR index; a mocked passage carries no
    # meaningful grounding information so the retrieval branch's selection
    # criterion is not a fair check here.
    print(
        f"\n[FLARE smoke] escalated on {escalated_count}/"
        f"{len(FEVER_SMOKE_QUERIES)} queries (diagnostic only; mocked "
        f"retrieval -- real escalation calibration belongs on the Vast run)."
    )
