"""
tests/test_calibration_batch_equivalence.py
============================================
Equivalence contract for the BatchPipeline-wired
``scripts.run_calibration.collect_calibration_data``.

The function used to call ``pipeline.answer(...)`` once per sample. The
2026-04-25 wiring routes the same per-sample work through
``BatchPipeline.answer_batch`` in chunks of ``batch_size``. This test
asserts that the chunked path is observationally equivalent to the
serial path within the documented tolerances:

    u_pre   : atol = 1e-3   (composite calibration probability)
    u_stored: atol = 1e-3   (post-Stage-5 storage scalar)
    signals : atol = 1e-2   (u_token, u_dropout, s_avg, h_norm columns)

The tolerance bounds are documented at ``caem/pipeline_batch.py:18-24``
and are well below calibration-bin discretisation (n_bins=15 -> bin
width ≈ 0.067), so the calibrated temperature scalar fitted from each
path will match to many more digits than ECE itself can resolve.

Mock-only: this test does NOT spin up the real T5/MiniCheck stack so
it runs in CI in well under a second, no GPU required.

Data source
-----------
When ``outputs/cycle_0/eval/fever_cycle0.json`` (the in-flight Phase 1a
artefact) is available we sample 10 records from it as the input shape
the calibration sweep would see in production. When the file is absent
(CI checkout, fresh clone) we fall back to a deterministic synthetic
distribution that exercises the same per-sample fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

# Repo root: /workspace/caem/.claude/worktrees/agent-aad2027e221164c24
REPO_ROOT = Path(__file__).resolve().parent.parent
FEVER_PATH_CANDIDATES = [
    # Worktree-local outputs (rare; the in-flight run writes to the parent repo).
    REPO_ROOT / "outputs" / "cycle_0" / "eval" / "fever_cycle0.json",
    # Parent repo (where Phase 1a actually writes).
    REPO_ROOT.parent.parent.parent / "outputs" / "cycle_0" / "eval" / "fever_cycle0.json",
]


# --------------------------------------------------------------------------- #
# Sample loader                                                               #
# --------------------------------------------------------------------------- #

def _load_fever_subset(n: int = 10) -> List[Dict[str, Any]]:
    """Load up to ``n`` fever_cycle0 records and shape them as
    BenchmarkSample dicts (the format ``collect_calibration_data`` expects
    via ``sample["question"]``).

    Returns synthetic deterministic samples when the JSON is not found so
    the test is self-contained on a fresh checkout.
    """
    for path in FEVER_PATH_CANDIDATES:
        if path.exists():
            with path.open() as f:
                data = json.load(f)
            samples = data.get("samples", [])[:n]
            return [
                {
                    "id": s.get("id"),
                    "question": s.get("question", ""),
                    "answers": s.get("gold_answers", []),
                    "gold_label": s.get("gold_label"),
                }
                for s in samples
            ]
    # Synthetic fallback. Crafted so signals span the [0, 1] interval so
    # numerical drift under bf16 batching would be visible if it were real.
    return [
        {
            "id": f"synthetic-{i}",
            "question": f"Claim: synthetic test claim number {i} is testing batching.",
            "answers": ["supports" if i % 2 == 0 else "refutes"],
            "gold_label": "supports" if i % 2 == 0 else "refutes",
        }
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Deterministic pipeline mock                                                 #
# --------------------------------------------------------------------------- #

def _make_pipeline_mock() -> MagicMock:
    """Build a CAEMPipeline stand-in whose ``answer(query=...)`` returns a
    deterministic ``PipelineResult``-shaped object derived from the query
    string. Each query maps to one fixed output regardless of which path
    (serial loop vs ``BatchPipeline.answer_batch``) calls it -- which is
    precisely what equivalence asserts: the two paths thread the same
    samples through the same per-sample primitive.
    """
    pipeline = MagicMock()

    # Stable hash so repeated runs produce the same vout signals.
    def _q_hash(query: str) -> float:
        return (sum(ord(c) for c in query) % 1000) / 1000.0

    def _answer(
        query=None, store_to_memory=True, source_benchmark=None,
        _precomputed_routing=None,
        _precomputed_tier2_answer=None,
        _precomputed_tier3_answer=None,
        _precomputed_vout=None,
    ):
        h = _q_hash(query or "")
        # Build a deterministic verifier_output per query. Spread the
        # signals across [0, 1] so the per-signal tolerance check is real.
        # Set every attribute consumed by scripts/fit_composite_calibration.py
        # (the 10 COMPOSITE_SIGNALS) so the per-sample JSONL schema check
        # below can verify that all 10 keys are populated and consistent
        # between the serial and batched paths.
        vout = SimpleNamespace(
            u_stored=0.30 + 0.40 * h,        # in [0.30, 0.70]
            u_token=0.10 + 0.50 * h,          # in [0.10, 0.60]
            u_dropout=0.20 + 0.30 * h,
            u_internal=0.15 + 0.45 * h,
            s_avg=0.25 + 0.40 * h,
            h_norm=0.10 + 0.60 * h,
            p_entail=0.50 + 0.30 * h,
            p_ground_max=0.40 + 0.20 * h,
            p_ground_mean=0.30 + 0.20 * h,
            p_ground_atomic=0.35 + 0.25 * h,
            q_a_relevance=0.20 + 0.50 * h,
            p_contra=0.10 * h,
            decision="STORE" if h > 0.5 else "DISCARD",
        )
        pre_conf = SimpleNamespace(u_pre=0.40 + 0.30 * h)  # in [0.40, 0.70]

        # The calibration sweep treats answer prefix "supports" as label=1
        # for FEVER (via fever_accuracy). Alternate em across samples so
        # u_pre_labels has variance.
        if "test claim number" in (query or ""):
            answer_text = "Answer: supports" if int(h * 1000) % 2 == 0 else "Answer: refutes"
        else:
            # Real fever_cycle0 prediction style.
            answer_text = "Reasoning: stub.\nAnswer: supports"

        return SimpleNamespace(
            query=query,
            answer=answer_text,
            display_answer=answer_text,
            tier=3,
            stored=False,
            u_stored=vout.u_stored,
            latency_ms=8000.0,
            decision=vout.decision,
            escalated=False,
            entry_id=None,
            verifier_output=vout,
            pre_confidence=pre_conf,
        )

    pipeline.answer.side_effect = _answer

    # BatchPipeline.answer_batch internally calls _peek_routing which reads
    # pipeline._encode_query / pre_estimator / memory_store / router.
    # MagicMock's auto-attribute returns more MagicMocks which int() can't
    # cast, so we install lightweight stubs that produce plausible values.
    # The router always returns Tier 3 so every sample takes the same path
    # (matches the real fever_cycle0 distribution where 98% of samples
    # land on Tier 3).
    pipeline._encode_query = lambda q: [0.0]
    pipeline.pre_estimator.estimate = lambda q: SimpleNamespace(u_pre=0.5)
    pipeline.memory_store.search_with_ids = lambda emb, k=1: SimpleNamespace(
        scores=[], entries=[],
    )
    pipeline.router.route = lambda pre_conf, search_with_ids: SimpleNamespace(
        tier=3,
    )
    # rag.generate_batch / verifier.verify_batch are not invoked because
    # answer_batch's batched verify only fires when verify_inputs is non-
    # empty AND tier-bucketing routes a sample to Tier 2/3 here -- the
    # batched generate path *does* fire for Tier 3, so wire generate_batch
    # to return one stub answer per query, which answer() will accept via
    # _precomputed_tier3_answer kwarg and short-circuit the real RAG.
    pipeline.rag.generate_batch = lambda queries: ["stub"] * len(queries)
    # Verify is also bypassed because the serial answer() will use the
    # _precomputed_vout kwarg. The verifier.verify_batch needs to return
    # one stub per (q, a) pair so the BatchPipeline answer_batch loop
    # threads precomputed vouts.
    pipeline.verifier.verify_batch = lambda inputs: [
        SimpleNamespace(decision="DISCARD", u_stored=0.5)
        for _ in inputs
    ]
    return pipeline


# --------------------------------------------------------------------------- #
# Equivalence test                                                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("batched_size", [4, 32])
def test_collect_calibration_data_batch_equivalence(batched_size, tmp_path):
    """Per-sample u_pre / u_stored / signals must match between batch_size=1
    (serial-path) and batch_size=N (batched-path) within tolerance.

    The mock pipeline is deterministic in the query string, so any drift
    between the two paths can only come from the BatchPipeline wiring
    itself (not from numerical noise in T5 / MiniCheck, which we're
    bypassing). Real-GPU bf16 noise is bounded separately by
    ``test_pipeline_batch_equivalence.py`` and the level-b smoke test.
    """
    from scripts.run_calibration import collect_calibration_data

    samples = _load_fever_subset(n=10)
    assert len(samples) == 10

    calib_samples = {"fever": samples}

    # ---- Serial reference ----------------------------------------------- #
    serial_pipeline = _make_pipeline_mock()
    serial_jsonl = tmp_path / "serial.json"
    serial_u_pre, serial_labels, serial_signals, serial_signal_labels = (
        collect_calibration_data(
            serial_pipeline, calib_samples,
            record_jsonl_path=serial_jsonl,
            batch_size=1,
        )
    )

    # ---- Batched candidate ---------------------------------------------- #
    batch_pipeline = _make_pipeline_mock()
    batch_jsonl = tmp_path / "batch.json"
    batch_u_pre, batch_labels, batch_signals, batch_signal_labels = (
        collect_calibration_data(
            batch_pipeline, calib_samples,
            record_jsonl_path=batch_jsonl,
            batch_size=batched_size,
        )
    )

    # Lengths must match exactly: every sample must produce a record on
    # both paths or the test is uninformative.
    assert len(serial_u_pre) == len(batch_u_pre) == 10, (
        f"length mismatch serial={len(serial_u_pre)} batch={len(batch_u_pre)}"
    )
    assert len(serial_signals) == len(batch_signals)
    assert serial_labels == batch_labels
    assert serial_signal_labels == batch_signal_labels

    # ---- Per-sample u_pre tolerance ------------------------------------- #
    for i, (s, b) in enumerate(zip(serial_u_pre, batch_u_pre)):
        assert abs(s - b) <= 1e-3, (
            f"u_pre[{i}] drift exceeds 1e-3: serial={s} batch={b}"
        )

    # ---- Per-sample signal-row tolerance -------------------------------- #
    for i, (sr, br) in enumerate(zip(serial_signals, batch_signals)):
        assert len(sr) == len(br) == 4
        for j, (sv, bv) in enumerate(zip(sr, br)):
            assert abs(sv - bv) <= 1e-2, (
                f"signal[{i}][{j}] drift exceeds 1e-2: "
                f"serial={sv} batch={bv}"
            )

    # ---- Per-sample u_stored tolerance ---------------------------------- #
    # The JSONL dump carries u_stored per record. Order is preserved on
    # both paths (collect_calibration_data iterates calib_samples in
    # insertion order; chunked iteration preserves that).
    with serial_jsonl.open() as f:
        serial_records = json.load(f)["samples"]
    with batch_jsonl.open() as f:
        batch_records = json.load(f)["samples"]
    assert len(serial_records) == len(batch_records) == 10

    for i, (sr, br) in enumerate(zip(serial_records, batch_records)):
        assert sr["id"] == br["id"], (
            f"record[{i}] id mismatch: serial={sr['id']} batch={br['id']}"
        )
        s_us = sr["u_stored"]
        b_us = br["u_stored"]
        if s_us is None or b_us is None:
            assert s_us == b_us, (
                f"u_stored[{i}] None-mismatch: serial={s_us} batch={b_us}"
            )
        else:
            assert abs(s_us - b_us) <= 1e-3, (
                f"u_stored[{i}] drift exceeds 1e-3: "
                f"serial={s_us} batch={b_us}"
            )

    # ---- Schema check: all 10 COMPOSITE_SIGNALS must be present -------- #
    # Regression guard: scripts/fit_composite_calibration.py reads exactly
    # these keys from the JSONL to fit one isotonic regression per signal.
    # Dropping any key (as happened pre-fix on 2026-04-25 06:14 UTC) makes
    # the composite degenerate to a 2-feature logistic regression and the
    # downstream conformal gate produces tau_store=1.0 / store_n=0. Keep
    # this list in sync with caem.verification.cal_prob_composite.COMPOSITE_SIGNALS.
    REQUIRED_SIGNAL_KEYS = (
        "u_token",
        "u_dropout",
        "u_internal",
        "s_avg",
        "h_norm",
        "p_entail",
        "p_ground_max",
        "p_ground_mean",
        "p_ground_atomic",
        "q_a_relevance",
    )
    for i, (sr, br) in enumerate(zip(serial_records, batch_records)):
        for key in REQUIRED_SIGNAL_KEYS:
            assert key in sr, (
                f"serial record[{i}] missing required signal key {key!r}; "
                f"keys present: {sorted(sr.keys())}"
            )
            assert key in br, (
                f"batch record[{i}] missing required signal key {key!r}; "
                f"keys present: {sorted(br.keys())}"
            )
            sv, bv = sr[key], br[key]
            # Both paths run the same deterministic mock; values must
            # match within the same 1e-3 tolerance as u_stored. None on
            # one side and not the other is a hard mismatch.
            if sv is None or bv is None:
                assert sv == bv, (
                    f"signal {key!r}[{i}] None-mismatch: "
                    f"serial={sv} batch={bv}"
                )
            else:
                assert abs(sv - bv) <= 1e-3, (
                    f"signal {key!r}[{i}] drift exceeds 1e-3: "
                    f"serial={sv} batch={bv}"
                )


def test_collect_calibration_data_zero_or_negative_batch_size_clamps_to_one(tmp_path):
    """``batch_size`` <= 0 must clamp to 1 instead of producing an empty
    range and silently collecting nothing."""
    from scripts.run_calibration import collect_calibration_data

    samples = _load_fever_subset(n=4)
    pipeline = _make_pipeline_mock()
    u_pre, labels, signals, _ = collect_calibration_data(
        pipeline, {"fever": samples},
        record_jsonl_path=tmp_path / "out.json",
        batch_size=0,  # pathological -- must not zero out the loop
    )
    # Length 4 (not 0): clamp worked.
    assert len(u_pre) == 4
    assert len(labels) == 4
    assert len(signals) == 4


# =========================================================================== #
# Per-benchmark calibration fitting helpers (v2 Fix 12)                        #
# =========================================================================== #
# Tests for fit_per_benchmark_temperatures + fit_per_benchmark_safety_floors,
# the post-collect_calibration_data helpers that turn per-sample records
# into the per-bench T_b and safety_u_pre_min_b dicts CAEMConfig consumes.
# Companion dispatch tests live in tests/test_pre_routing.py and
# tests/test_router.py.


def _make_synthetic_calib_records(n_per_bench: int, em_rate: float, seed: int = 0):
    """Build calibration-fold records used by the fit helpers. Higher u_pre
    correlates with em=1."""
    import random
    rng = random.Random(seed)
    records = []
    for i in range(n_per_bench):
        em = 1 if rng.random() < em_rate else 0
        if em == 1:
            u = rng.uniform(0.55, 0.95)
        else:
            u = rng.uniform(0.05, 0.50)
        records.append({"benchmark": "x", "u_pre": u, "em": em})
    return records


def test_fit_per_benchmark_temperatures():
    from scripts.run_calibration import fit_per_benchmark_temperatures
    recs_a = _make_synthetic_calib_records(120, em_rate=0.55, seed=1)
    recs_b = _make_synthetic_calib_records(120, em_rate=0.40, seed=2)
    for r in recs_a:
        r["benchmark"] = "fever"
    for r in recs_b:
        r["benchmark"] = "triviaqa"
    records = recs_a + recs_b

    fit = fit_per_benchmark_temperatures(records, pooled_T=1.0)
    assert set(fit.keys()) == {"fever", "triviaqa"}
    # fit_temperature_scalar bounds T to [exp(-3), exp(3)] = [0.0498, 20.09]
    for bm, T_b in fit.items():
        assert 0.04 <= T_b <= 20.1, f"T_b for {bm} out of bounds: {T_b}"


def test_fit_per_benchmark_safety_floors():
    from scripts.run_calibration import fit_per_benchmark_safety_floors
    import random
    rng = random.Random(7)
    records = []
    for _ in range(200):
        em = rng.randint(0, 1)
        u = rng.uniform(0.6, 0.95) if em == 1 else rng.uniform(0.05, 0.4)
        records.append({"benchmark": "fever", "u_pre": u, "em": em})

    floors = fit_per_benchmark_safety_floors(
        records, pooled_floor=0.38, target_precision=0.95,
    )
    assert "fever" in floors
    assert floors["fever"] >= 0.30, (
        f"floor unexpectedly low for cleanly-separated bands at "
        f"target_precision=0.95: {floors['fever']}"
    )

    floors_loose = fit_per_benchmark_safety_floors(
        records, pooled_floor=0.38, target_precision=0.80,
    )
    assert floors_loose["fever"] > 0.0


def test_fit_per_benchmark_falls_back_for_thin_data():
    from scripts.run_calibration import (
        fit_per_benchmark_temperatures,
        fit_per_benchmark_safety_floors,
    )
    import random
    rng = random.Random(11)
    recs = []
    for _ in range(120):
        em = rng.randint(0, 1)
        u = rng.uniform(0.4, 0.9) if em == 1 else rng.uniform(0.0, 0.5)
        recs.append({"benchmark": "fever", "u_pre": u, "em": em})
    for _ in range(5):
        recs.append({"benchmark": "triviaqa", "u_pre": 0.5, "em": 1})

    pooled_T = 1.7
    pooled_floor = 0.42
    Ts = fit_per_benchmark_temperatures(recs, pooled_T=pooled_T)
    floors = fit_per_benchmark_safety_floors(recs, pooled_floor=pooled_floor)
    # Thin benchmark → pooled fallback
    assert Ts["triviaqa"] == pooled_T
    assert floors["triviaqa"] == pooled_floor
    # Healthy benchmark → real fit
    assert 0.05 <= Ts["fever"] <= 20.1
