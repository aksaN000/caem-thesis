"""
caem/benchmark_splits.py
=========================
Deterministic per-benchmark 6-way pool split with content-hash leakage guards.

Purpose (Branch C 2026-04-22 stream-mode refactor)
---------------------------------------------------
Every sample of every benchmark must belong to **exactly one** of six disjoint
pools:

    seed           cold-start memory episodes (Step 6)          — 200 / benchmark
    purity         α-measurement slice (Step 19)                — 500 / benchmark
    calibration    τ + temperature fitting slice (Step 7.0.2)   — 500 / benchmark
    sil_train_*    N cycles × M samples, disjoint across cycles — N × M / benchmark
    eval           per-cycle trajectory (Step 5 of each cycle)  — 500 / benchmark
    test           held-out final generalisation table (Ch5)    — 500 / benchmark

**Stream mode**: each SIL cycle consumes its own 5000-sample chunk from
sil_train_chunks[cycle_num - 1]; chunks are content-hash disjoint.

**Leakage guards**: ``build_benchmark_pools`` runs ``assert_no_leakage`` on its
output; ``build_all_benchmark_pools`` additionally runs
``assert_cross_benchmark_disjoint``. Any duplicate ``content_id`` across pools
raises ``PoolLeakageError``, halting the runner immediately.

Training vs transfer split (Branch C panel)
-------------------------------------------
* **Training benchmarks** (FEVER, TriviaQA, Natural Questions): populate every
  pool. Provide SIL training signal via sil_train_chunks.
* **Transfer-only benchmarks** (TruthfulQA, StrategyQA, ARC-Challenge, ASQA):
  populate only eval and test pools. sil_train_chunks is a tuple of empty
  chunks (one per cycle). ASQA's role is specifically to provide long-form
  hypothesis stress for Path B (Qwen-judge) evidence via eval trajectory.

Capacity
--------
Total per training benchmark: 200 + 500 + 500 + 10 × 5000 + 500 + 500 = 52,200.
Available:
  FEVER  train ≈ 145,449 → 2.8× capacity for Phase 2 re-runs
  TriviaQA train ≈ 87,622 → 1.7× capacity
  NQ-Open train ≈ 87,925 → 1.7× capacity
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Panel definition (v2 — 2026-05-06)
# =============================================================================
# v2 single source of truth re-exported from caem.config to prevent drift.
# (v1 had duplicate constants here that drifted from caem/config.py during
# the cycle 0-4 trajectory; v2 imports them so config.py is canonical.)

from caem.config import (  # noqa: E402  (deliberate late import to avoid cycle)
    TRAINING_BENCHMARKS,
    TRANSFER_BENCHMARKS,
)

ALL_BENCHMARKS: Tuple[str, ...] = TRAINING_BENCHMARKS + TRANSFER_BENCHMARKS


# =============================================================================
# Default pool sizes (override via kwargs when calling build_benchmark_pools)
# =============================================================================

DEFAULT_SEED_SIZE: int = 1000  # CANDIDATE pool, not store target.
# At ~60-70% store rate, ~1000 candidates yields ~600-700 stored episodes
# (target_episodes=200 per benchmark from run_phase1a.sh seed step means
# the seeder stops well before the 1000-candidate pool exhausts). Sizing
# at 1000 gives ~3× headroom vs the 200 target for unusually low store-rate
# benchmarks (e.g., ASQA at 53% would need ~375 candidates).
DEFAULT_PURITY_SIZE: int = 500
DEFAULT_CALIBRATION_SIZE: int = 500  # v2.1 reverted to 500 on 2026-05-09 after the
                                     # 300/bench × per-bench fit at α=0.05 then α=0.20
                                     # both produced near-degenerate gates (FEVER STORE
                                     # gap inverted; poisoning 35-54%). The cal-fold
                                     # reduction tradeoff (label-efficient story) was
                                     # under-powered for stable per-bench composite +
                                     # conformal fits. v1 used 500/bench × 3 = 1500
                                     # pooled and held 95%+ cal precision; v2.1 keeps
                                     # 500/bench × 3 = 1500 total (same labeling cost
                                     # as v1) but fits per-bench. See branch_C_log.md
                                     # 2026-05-09 entry.
DEFAULT_TRAIN_CHUNK_SIZE: int = 1000  # v2 default; see PER_BENCHMARK override below.
DEFAULT_N_CYCLES: int = 10
DEFAULT_EVAL_SIZE: int = 500
DEFAULT_TEST_SIZE: int = 500

# v2.1 (2026-05-08): per-benchmark stream-chunk override. Doubled FEVER + TriviaQA
# from 1000 -> 2000 after panel pruning to 3 training benches saved sufficient
# per-cycle wall-time budget. CSQA stays at 700 because its 9.7k train pool can't
# support a larger chunk (10 × 2000 + seed/cal/eval/test = 27k > 9.7k).
# Pool reweighting (Fix 3, T=2.0 softmax) handles the asymmetric chunk sizes
# via temperature-mixed equalization.
PER_BENCHMARK_TRAIN_CHUNK_SIZE: Dict[str, int] = {
    "fever":          2000,  # v2.1: 145k train, doubled from 1000
    "triviaqa":       2000,  # v2.1: 87k train, doubled from 1000
    "commonsense_qa":  700,  # 9.7k train: tight but no repetition; stays at 700
}
# HotpotQA + Natural Questions REMOVED at v2.1 (2026-05-08) after cycle-0
# eval revealed precondition violation:
#   HotpotQA: p_+ = 0.090, required TPR/FPR ≥ 192 at α=0.05 (verifier
#             achievable: 5-15) — mathematically unstoreable.
#   Natural Q: p_+ = 0.156-0.172, verifier α<½ on every v1 cal fold —
#              structural-failure benchmark; non-discriminative even pooled.
# Both kept as registered exclusion evidence at outputs/cycle_0/eval/.

# Per-benchmark dev/test split names for eval pool.
_EVAL_SPLIT_MAP: Dict[str, Optional[str]] = {
    "fever": "dev",
    "triviaqa": "validation",
    "natural_questions": "validation",
    "asqa": "dev",
    "truthfulqa": None,         # single-split; load_benchmark returns the pool
    "strategyqa": "test",
    "arc_challenge": "test",
    "hotpotqa": "validation",   # v2 NEW: HotpotQA distractor config dev split
    "commonsense_qa": "validation",  # v2 NEW: CSQA test labels not public, use dev
}


# =============================================================================
# Errors
# =============================================================================

class InsufficientBenchmarkDataError(RuntimeError):
    """Raised when a benchmark's split is too small for the requested allocation."""


class PoolLeakageError(AssertionError):
    """Raised when two or more pools share at least one sample (by content-hash)."""


# =============================================================================
# Sample identity
# =============================================================================

def content_id(sample: dict, max_chars: int = 500) -> str:
    """Stable SHA-256-based identifier for a benchmark sample.

    Derived from question text (first max_chars) so the ID survives:
      - HF dataset version bumps (native .id drifts, question text doesn't)
      - Reordering / sub-sampling
      - Cross-process loads

    16 hex chars (64 bits) is ample for the ≤ 10⁶-sample panel; collision
    probability is negligible at this scale.
    """
    q = str(sample.get("question", "")).strip()[:max_chars]
    return hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# Pool container
# =============================================================================

@dataclass(frozen=True)
class BenchmarkPools:
    """Frozen six-way disjoint split for one benchmark.

    Training benchmarks populate every field; transfer-only benchmarks leave
    seed/purity/calibration as empty tuples and sil_train_chunks as a tuple
    of n_cycles empty tuples (so cycle-indexed lookup never raises).
    """
    benchmark: str
    is_training: bool
    seed: Tuple[dict, ...] = ()
    purity: Tuple[dict, ...] = ()
    calibration: Tuple[dict, ...] = ()
    # sil_train_chunks[cycle_num - 1] is the 1-indexed per-cycle SIL training chunk.
    # Length is always n_cycles; transfer benchmarks have tuples of empty tuples.
    sil_train_chunks: Tuple[Tuple[dict, ...], ...] = ()
    eval: Tuple[dict, ...] = ()
    test: Tuple[dict, ...] = ()

    def all_samples(self) -> List[dict]:
        """Flatten every pool into one list (for leakage checks and logging)."""
        out: List[dict] = []
        out.extend(self.seed)
        out.extend(self.purity)
        out.extend(self.calibration)
        for chunk in self.sil_train_chunks:
            out.extend(chunk)
        out.extend(self.eval)
        out.extend(self.test)
        return out

    def total_size(self) -> int:
        return len(self.all_samples())

    def cycle_chunk(self, cycle_num: int) -> Tuple[dict, ...]:
        """Return the SIL train chunk for cycle_num (1-indexed).

        Returns empty tuple for transfer-only benchmarks. Raises IndexError
        if cycle_num is out of range.
        """
        if not self.is_training:
            return ()
        if cycle_num < 1 or cycle_num > len(self.sil_train_chunks):
            raise IndexError(
                f"cycle_num={cycle_num} out of range 1..{len(self.sil_train_chunks)}"
            )
        return self.sil_train_chunks[cycle_num - 1]

    def summary(self) -> str:
        chunks_info = (
            f"{len(self.sil_train_chunks)}×{len(self.sil_train_chunks[0]) if self.sil_train_chunks and len(self.sil_train_chunks[0]) > 0 else 0}"
        )
        return (
            f"{self.benchmark} (is_training={self.is_training}): "
            f"seed={len(self.seed)} purity={len(self.purity)} "
            f"calib={len(self.calibration)} sil_train={chunks_info} "
            f"eval={len(self.eval)} test={len(self.test)} total={self.total_size()}"
        )


# =============================================================================
# Builders
# =============================================================================

def _dedupe_by_content_id(samples: List[dict]) -> List[dict]:
    """Remove within-split duplicates by content-hash.

    Source benchmarks (FEVER lucadiliello, NQ-Open, TriviaQA) contain
    legitimate within-split duplicates — same question text appearing
    multiple times (often with different native IDs because HF release
    integer IDs drift). Without dedup, two copies of the same question
    end up in different pools (e.g., one in train chunk 0, one in calib)
    and ``assert_no_leakage`` fires.

    Dedup is stable (keeps first occurrence) so shuffles remain
    reproducible given the same rng_seed.
    """
    seen: set = set()
    out: List[dict] = []
    for s in samples:
        cid = content_id(s)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(s)
    if len(out) < len(samples):
        logger.info(
            "Deduped %d → %d samples by content-hash (dropped %d within-split duplicates)",
            len(samples), len(out), len(samples) - len(out),
        )
    return out


def build_benchmark_pools(
    benchmark: str,
    *,
    seed_size: int = DEFAULT_SEED_SIZE,
    purity_size: int = DEFAULT_PURITY_SIZE,
    calibration_size: int = DEFAULT_CALIBRATION_SIZE,
    train_chunk_size: Optional[int] = None,
    n_cycles: int = DEFAULT_N_CYCLES,
    eval_size: int = DEFAULT_EVAL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    rng_seed: int = 42,
    train_split: str = "train",
    eval_split: Optional[str] = None,
) -> BenchmarkPools:
    """Build the disjoint 6-way pool split for one benchmark.

    Training benchmarks (fever, triviaqa, natural_questions) draw
    seed/purity/calibration/sil_train_chunks from the train split, and
    eval/test from the dev/validation split.

    Transfer-only benchmarks (truthfulqa, strategyqa, arc_challenge, asqa)
    only draw eval/test; the other pools are empty.

    Raises:
        InsufficientBenchmarkDataError: the benchmark split lacks enough samples
            for the requested allocation.
        PoolLeakageError: content-hash duplication detected across pools (should
            not happen given slicing is disjoint, but guards against silent
            upstream dedup failures in load_benchmark).
    """
    from eval.benchmarks import load_benchmark  # lazy import: avoids circular

    is_training = benchmark in TRAINING_BENCHMARKS

    resolved_eval_split = (
        eval_split if eval_split is not None else _EVAL_SPLIT_MAP.get(benchmark, "dev")
    )

    # 2026-05-07 audit fix: when train_chunk_size is not explicitly passed
    # (default None), consult PER_BENCHMARK_TRAIN_CHUNK_SIZE so benchmarks
    # with small train splits (CSQA at 9.7k) get the right chunk size
    # automatically. Without this, callers using module defaults — including
    # scripts/seed_cold_start.py and the cold_start_loader closure in
    # run_experiment.py — silently raised InsufficientBenchmarkDataError on
    # CSQA because the default train_chunk_size=1000 implied a 12k allocation
    # vs 9741 available samples.
    if train_chunk_size is None:
        train_chunk_size = PER_BENCHMARK_TRAIN_CHUNK_SIZE.get(
            benchmark, DEFAULT_TRAIN_CHUNK_SIZE,
        )

    # -------- Training benchmarks: train-split pools -------- #
    seed_pool: Tuple[dict, ...] = ()
    purity_pool: Tuple[dict, ...] = ()
    calib_pool: Tuple[dict, ...] = ()
    sil_chunks: Tuple[Tuple[dict, ...], ...]

    if is_training:
        needed_train = (
            seed_size + purity_size + calibration_size + n_cycles * train_chunk_size
        )
        logger.info(
            "Loading TRAIN split of %s (need %d samples for 6-way split)...",
            benchmark, needed_train,
        )
        all_train_raw = load_benchmark(benchmark, split=train_split, n=None)
        # Dedup first — source datasets contain within-split duplicates that
        # would leak across pools (same question in train + calib + purity).
        all_train = _dedupe_by_content_id(list(all_train_raw))
        if len(all_train) < needed_train:
            raise InsufficientBenchmarkDataError(
                f"Benchmark '{benchmark}' train split has {len(all_train)} unique "
                f"samples (from {len(all_train_raw)} raw) but 6-way split requires "
                f"{needed_train} (seed={seed_size} + purity={purity_size} + "
                f"calib={calibration_size} + {n_cycles}×{train_chunk_size} train). "
                f"Either reduce one of these sizes or replace the benchmark."
            )
        rng = random.Random(rng_seed)
        train_pool = list(all_train)
        rng.shuffle(train_pool)

        cursor = 0
        seed_pool = tuple(train_pool[cursor:cursor + seed_size])
        cursor += seed_size
        purity_pool = tuple(train_pool[cursor:cursor + purity_size])
        cursor += purity_size
        calib_pool = tuple(train_pool[cursor:cursor + calibration_size])
        cursor += calibration_size
        chunks_list: List[Tuple[dict, ...]] = []
        for _ in range(n_cycles):
            chunks_list.append(tuple(train_pool[cursor:cursor + train_chunk_size]))
            cursor += train_chunk_size
        sil_chunks = tuple(chunks_list)
    else:
        # Transfer-only: empty train-side pools, empty chunk per cycle
        sil_chunks = tuple(() for _ in range(n_cycles))

    # -------- Eval + test from dev/validation/test split -------- #
    logger.info(
        "Loading eval-side split (%s) of %s for eval+test pools...",
        resolved_eval_split, benchmark,
    )
    if resolved_eval_split is None:
        # Single-split benchmarks (e.g., TruthfulQA)
        all_eval_raw = load_benchmark(benchmark, n=None)
    else:
        all_eval_raw = load_benchmark(benchmark, split=resolved_eval_split, n=None)
    # Dedup eval side too (NQ-Open dev has known near-duplicates).
    all_eval = _dedupe_by_content_id(list(all_eval_raw))

    needed_eval = eval_size + test_size
    if len(all_eval) < needed_eval:
        if is_training:
            # Training benchmarks must have full headroom for eval + test.
            raise InsufficientBenchmarkDataError(
                f"Benchmark '{benchmark}' eval-side split ({resolved_eval_split}) has "
                f"{len(all_eval)} unique samples but eval+test requires {needed_eval} "
                f"(eval={eval_size} + test={test_size})."
            )
        # Transfer-only benchmarks with small dev splits: auto-clamp.
        # TruthfulQA (817), StrategyQA (229 dev), ARC (299 dev) are known small.
        # We guarantee eval reaches its full requested size if possible; any
        # remaining samples go to test. If even eval_size > available, we clamp
        # eval to available and leave test empty.
        if len(all_eval) <= eval_size:
            effective_eval = len(all_eval)
            effective_test = 0
            logger.warning(
                "Transfer benchmark %s has %d samples < eval_size=%d; "
                "clamping eval to %d, test empty (held-out test not possible).",
                benchmark, len(all_eval), eval_size, effective_eval,
            )
        else:
            effective_eval = eval_size
            effective_test = len(all_eval) - eval_size
            logger.warning(
                "Transfer benchmark %s has %d samples; eval=%d, test clamped to %d "
                "(requested %d; transfer-only benchmark has limited dev split).",
                benchmark, len(all_eval), effective_eval, effective_test, test_size,
            )
    else:
        effective_eval = eval_size
        effective_test = test_size

    # Distinct seed so train-side and eval-side shuffles don't align.
    rng2 = random.Random(rng_seed + 1)
    eval_pool_all = list(all_eval)
    rng2.shuffle(eval_pool_all)
    eval_pool = tuple(eval_pool_all[:effective_eval])
    test_pool = tuple(eval_pool_all[effective_eval:effective_eval + effective_test])

    pools = BenchmarkPools(
        benchmark=benchmark,
        is_training=is_training,
        seed=seed_pool,
        purity=purity_pool,
        calibration=calib_pool,
        sil_train_chunks=sil_chunks,
        eval=eval_pool,
        test=test_pool,
    )
    assert_no_leakage(pools)
    logger.info("Built pools: %s", pools.summary())
    return pools


def assert_no_leakage(pools: BenchmarkPools) -> None:
    """Raise PoolLeakageError if any sample appears in more than one pool."""
    all_samples = pools.all_samples()
    all_ids = [content_id(s) for s in all_samples]
    n_total = len(all_ids)
    n_unique = len(set(all_ids))
    if n_total != n_unique:
        dup_counts = Counter(all_ids)
        duplicates = {h: c for h, c in dup_counts.items() if c > 1}
        raise PoolLeakageError(
            f"Pool leakage in {pools.benchmark}: {len(duplicates)} content-hash "
            f"duplicates across pools (total={n_total}, unique={n_unique}). "
            f"First duplicates: {list(duplicates.items())[:5]}"
        )


def assert_cross_benchmark_disjoint(
    pools_by_benchmark: Dict[str, BenchmarkPools],
) -> None:
    """Raise if two different benchmarks share a sample by content-hash.

    Not strictly required (different benchmarks have different questions by
    construction), but a safety net against dataset-release collisions (e.g.,
    FEVER claims accidentally overlapping with TriviaQA questions on the same
    HF release).
    """
    seen_ids: Dict[str, str] = {}
    for bm, pools in pools_by_benchmark.items():
        for sample in pools.all_samples():
            cid = content_id(sample)
            prev = seen_ids.get(cid)
            if prev is not None and prev != bm:
                raise PoolLeakageError(
                    f"Cross-benchmark leakage: sample with content-id {cid} "
                    f"appears in both {prev} and {bm}"
                )
            seen_ids[cid] = bm


def build_all_benchmark_pools(
    benchmarks: Sequence[str] = ALL_BENCHMARKS,
    **kwargs,
) -> Dict[str, BenchmarkPools]:
    """Build disjoint pool splits for every benchmark in the panel.

    Raises on first failure. Runs both per-benchmark and cross-benchmark
    leakage checks.
    """
    out: Dict[str, BenchmarkPools] = {}
    for bm in benchmarks:
        out[bm] = build_benchmark_pools(bm, **kwargs)
    assert_cross_benchmark_disjoint(out)
    logger.info(
        "Built pools for all %d benchmarks; cross-benchmark disjoint verified.",
        len(benchmarks),
    )
    return out


__all__ = [
    "TRAINING_BENCHMARKS",
    "TRANSFER_BENCHMARKS",
    "ALL_BENCHMARKS",
    "DEFAULT_SEED_SIZE",
    "DEFAULT_PURITY_SIZE",
    "DEFAULT_CALIBRATION_SIZE",
    "DEFAULT_TRAIN_CHUNK_SIZE",
    "DEFAULT_N_CYCLES",
    "DEFAULT_EVAL_SIZE",
    "DEFAULT_TEST_SIZE",
    "BenchmarkPools",
    "InsufficientBenchmarkDataError",
    "PoolLeakageError",
    "content_id",
    "build_benchmark_pools",
    "build_all_benchmark_pools",
    "assert_no_leakage",
    "assert_cross_benchmark_disjoint",
]
