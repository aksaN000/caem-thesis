"""
tests/test_flare_smoke.py
=========================
Tier-A smoke test for ``FLAREBaseline`` on Qwen-3B (Branch C, decoder-only).

Motivation
----------
``FLAREBaseline._look_ahead`` slices ``out.sequences[0, input_len:]`` to recover
the generated suffix from a causal LM. Getting the slice wrong is an easy way
for every FLARE answer to come back empty (the prior Flan-T5 implementation
had exactly this bug in reverse: it sliced ``out.sequences[0, 1:]`` after the
port to decoder-only, which on Qwen returns the whole echoed prompt as text
and wastes token budget).

This smoke test is a regression guard. It:

1. Loads Qwen-2.5-3B-Instruct via ``load_base_generator`` on whatever device
   is available (CUDA preferred, CPU fallback; CPU-only is slow but works).
2. Bypasses the PassageStore/FAISS/DPR dependency by constructing
   ``FLAREBaseline`` via ``object.__new__`` and injecting a mock ``self.rag``
   that returns a canned passage list. This lets the test run on a laptop
   GPU or CPU without the 60 GB Wikipedia passage index.
3. Runs a 5-query FEVER-shaped smoke set and asserts ``empty_answer == 0``.
4. Logs the ``escalated`` count as diagnostic output (not asserted -- the
   real threshold-calibration test belongs on the Vast run with the full
   DPR index).

Skip policy
-----------
Gated behind ``RUN_FLARE_SMOKE=1`` so it does not run on every ``pytest``
invocation (one-time Qwen download ~6 GB + GPU memory + tens of seconds).
Invoke explicitly:

    RUN_FLARE_SMOKE=1 pytest tests/test_flare_smoke.py -v -s
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pytest
import torch

from caem.model_loader import load_base_generator
from eval.baselines import FLAREBaseline


_RUN = os.environ.get("RUN_FLARE_SMOKE", "0") == "1"
pytestmark = pytest.mark.skipif(
    not _RUN,
    reason="Set RUN_FLARE_SMOKE=1 to run the FLARE decoder-slice smoke test.",
)


FEVER_SMOKE_QUERIES = [
    "Claim: Mount Everest is the tallest mountain on Earth.",
    "Claim: The capital of Australia is Sydney.",
    "Claim: Water boils at 100 degrees Celsius at sea level.",
    "Claim: The Great Wall of China is visible from the Moon with the naked eye.",
    "Claim: Albert Einstein developed the special theory of relativity in 1905.",
]


class MockTierThreeRAG:
    """Minimal stand-in for ``caem.retrieval.rag.TierThreeRAG`` exposing only
    the single method FLAREBaseline._generate calls. Real retrieval requires
    a DPR passage store + FAISS index; this mock returns a canned
    (passage_text, score) pair so the slicing fix can be tested in isolation.
    """

    def retrieve(self, query: str) -> List[Tuple[str, float]]:  # noqa: D401
        return [
            (
                "[This is a mocked passage. The FLARE smoke test does not "
                "exercise real retrieval; it only exercises the decoder-only "
                "look-ahead slice.]",
                0.0,
            )
        ]


@pytest.fixture(scope="module")
def flare_mocked() -> FLAREBaseline:
    """Build a FLAREBaseline whose ``.rag`` is a mock, without running
    ``RAGBaseline.__init__`` (which would require a real PassageStore).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    model, tokenizer = load_base_generator(
        model_name,
        device=device,
        dtype=dtype,
        use_flash_attention_2=False,  # smoke host may lack flash_attn
        use_torch_compile=False,
    )

    flare = object.__new__(FLAREBaseline)
    flare.name = "flare"
    flare.tier_value = 3
    flare.model_name = model_name
    flare.device = device
    flare.tokenizer = tokenizer
    flare.model = model
    flare.max_input_tokens = 2048
    flare.max_new_tokens = 256
    flare.theta = 0.4
    flare.look_ahead_tokens = 64
    flare.max_sentences = 8
    flare.rag = MockTierThreeRAG()
    # FLAREBaseline._grounded_generate relies on self.config for
    # rag_max_new_tokens / rag_do_sample; wire a default config.
    from caem.config import CAEMConfig
    flare.config = CAEMConfig()
    return flare


def test_flare_look_ahead_produces_tokens(flare_mocked: FLAREBaseline) -> None:
    """Direct test of ``_look_ahead``: the decoder-only slice should yield a
    non-empty ``gen_ids`` tensor and therefore non-empty decoded text for
    every query in the smoke set. An empty string here signals that the
    slicing has regressed (prompt echoed, nothing generated, or the input-
    length offset is wrong).
    """
    for query in FEVER_SMOKE_QUERIES:
        text, min_prob = flare_mocked._look_ahead(query, committed="")
        if not text.strip():
            pytest.fail(
                f"FLARE _look_ahead returned empty text for query {query!r}: "
                f"decoder-slice regression suspected. min_prob={min_prob}."
            )


def test_flare_smoke_non_empty_answers(flare_mocked: FLAREBaseline) -> None:
    """Five-query smoke test: every query produces a non-empty answer.

    Logs (but does not assert) how many triggered the retrieval branch --
    mocked passages carry no grounding signal, so the real escalation-rate
    calibration belongs on the Vast run with the full DPR index.
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

    assert empty_count == 0, (
        f"FLAREBaseline produced {empty_count} empty answers across "
        f"{len(FEVER_SMOKE_QUERIES)} smoke queries; decoder-slice fix "
        f"may have regressed."
    )

    print(
        f"\n[FLARE smoke] escalated on {escalated_count}/"
        f"{len(FEVER_SMOKE_QUERIES)} queries (diagnostic only; mocked "
        f"retrieval -- real escalation calibration belongs on the Vast run)."
    )
