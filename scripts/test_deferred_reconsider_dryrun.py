#!/usr/bin/env python
"""
scripts/test_deferred_reconsider_dryrun.py
==========================================
v2 Fix 11 Layer 5 — pre-launch synthetic dry-run of the deferred
reconsideration path.

Layers 1-3 of Fix 11 are runtime assertions that hard-fail if the
orchestrator forgets to thread the deferred-buffer + reconsider closure
into ``SelfImprovementLoop.run_cycle``. Layer 4 is a unit test on the
buffer + closure surface. This script is Layer 5: a real-process
end-to-end smoke that constructs a 5-entry synthetic deferred buffer,
runs ``deferred_buffer.reconsider`` against a stub verifier, and asserts
that:

  1. The reconsider closure is callable and accepts a deferred entry.
  2. ``source_benchmark`` survives the closure (Fix 11 + Fix 2 keystone).
  3. Promotion / TTL / kept counts add up to the buffer's entry count.
  4. Promoted entries are written to the memory store (or whatever the
     reconsider contract specifies).

The script runs on CPU and finishes in <1 second so it can be wired
into ``run_phase1a.sh`` as a pre-launch sanity check before any GPU
burns.

Exit code 0 → all assertions pass; non-zero → guard not wired.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("test_deferred_reconsider_dryrun")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--synthetic_n", type=int, default=5,
                   help="Number of synthetic deferred entries to construct.")
    args = p.parse_args()

    from caem.config import CAEMConfig, TRAINING_BENCHMARKS
    from caem.memory.deferred import DeferredBuffer
    from caem.memory.entry import EpisodicEntry
    from caem.memory.store import EpisodicMemoryStore
    from caem.verification.verifier import UnifiedVerifierOutput

    cfg = CAEMConfig()

    # Construct an empty memory store + deferred buffer.
    store = EpisodicMemoryStore(config=cfg)
    buf = DeferredBuffer()

    # Push N synthetic entries, distributed across training benchmarks
    # so the dispatch path is exercised on real benchmark identifiers.
    bench_cycle = list(TRAINING_BENCHMARKS) * (args.synthetic_n // len(TRAINING_BENCHMARKS) + 1)
    for i in range(args.synthetic_n):
        bm = bench_cycle[i]
        emb = np.random.RandomState(seed=42 + i).randn(768).astype(np.float32)
        emb /= np.linalg.norm(emb)
        # Build a synthetic UnifiedVerifierOutput in the DEFERRED band so
        # buf.push will actually buffer it (push branches on decision).
        vout = UnifiedVerifierOutput(
            u_token=0.6, u_dropout=0.3, u_internal=0.65,
            s_avg=0.7, h_norm=0.5,
            p_entail=0.6, p_ground_max=0.5, p_ground_mean=0.45,
            p_ground_atomic=0.4, p_contra=0.0,
            q_a_relevance=0.55, alias_overlap=0.5, entity_head_consistency=0.5,
            u_stored=0.55, decision="DEFERRED",
            abstained=False,
            early_exit_triggered=False,
            atomic_facts=[], per_atom_entail=[],
        )
        buf.push(
            question=f"Synthetic question {i} for {bm}?",
            answer=f"Synthetic answer {i}",
            embedding=emb,
            storage_cycle=0,
            vout=vout,
            source_benchmark=bm,
        )

    n_pushed = buf.size
    assert n_pushed == args.synthetic_n, (
        f"Layer 5: expected {args.synthetic_n} pushed entries; got {n_pushed}"
    )
    logger.info("Layer 5: pushed %d synthetic deferred entries (benches: %s).",
                n_pushed, [bench_cycle[i] for i in range(args.synthetic_n)])

    # Stub reconsider_fn that promotes about half on revisit.
    # Returns UnifiedVerifierOutput with decision=STORE for even-indexed
    # entries (synthetic deterministic split).
    promoted_seen: list = []
    def _stub_reconsider_fn(entry):
        promoted_seen.append(getattr(entry, "source_benchmark", None))
        # Promote roughly half (alternate STORE / DISCARD).
        decision = "STORE" if (len(promoted_seen) % 2 == 0) else "DISCARD"
        return UnifiedVerifierOutput(
            u_token=0.7, u_dropout=0.2, u_internal=0.75,
            s_avg=0.8, h_norm=0.4,
            p_entail=0.8, p_ground_max=0.7, p_ground_mean=0.65,
            p_ground_atomic=0.6, p_contra=0.0,
            q_a_relevance=0.7, alias_overlap=0.6, entity_head_consistency=0.7,
            u_stored=0.78 if decision == "STORE" else 0.20,
            decision=decision,
            abstained=(decision == "ABSTAIN"),
            early_exit_triggered=False,
            atomic_facts=[], per_atom_entail=[],
        )

    n_promoted, n_ttl, n_kept = buf.reconsider(
        verify_fn=_stub_reconsider_fn,
        memory_store=store,
    )

    # Assert source_benchmark survived the closure (Fix 2 keystone)
    assert all(b in TRAINING_BENCHMARKS for b in promoted_seen), (
        f"Layer 5: source_benchmark dropped in reconsider closure; "
        f"saw {set(promoted_seen)}, expected subset of {TRAINING_BENCHMARKS}"
    )
    logger.info(
        "Layer 5: reconsider OK — promoted=%d, ttl_dropped=%d, kept=%d "
        "(buf=%d, store=%d). source_benchmark survived for all entries.",
        n_promoted, n_ttl, n_kept, buf.size, store.size,
    )

    # Promoted+kept+ttl_dropped must equal the original push count
    # (entries are exhausted by the reconsider pass).
    assert n_promoted + n_ttl + n_kept == n_pushed, (
        f"Layer 5: count mismatch — promoted({n_promoted}) + ttl({n_ttl}) "
        f"+ kept({n_kept}) != pushed({n_pushed})"
    )

    # Promoted entries must be in the memory store.
    assert store.size == n_promoted, (
        f"Layer 5: memory store size {store.size} != n_promoted {n_promoted}"
    )

    print("✅ v2 Fix 11 Layer 5 dry-run PASSED — deferred reconsider path is wired.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
