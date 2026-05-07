"""
tests/test_pool_reweighting.py
================================
Unit tests for caem.training.pool_reweighting (v2 Fix 3 — loss-reweighted
SIL training pool builder).

Coverage
--------
  * per_benchmark_counts: counts pairs grouped by source_benchmark
  * Temperature-mixed softmax: T=1 ≈ empirical; T → ∞ → uniform
  * 3× upsample cap: each unique sample is replicated at most cap×
  * DoReMi floor: every populated benchmark gets ≥ floor samples
    in the output
  * Cold-start fallback: zero-count benchmark in the roster gets
    ``cold_start_n`` gold-labelled pairs from the loader callback
  * Empty input + no loader → empty output
  * Disabled config flag → flat shuffle (parity with v1 behaviour)
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class _Pair:
    """Minimal QAPair-shaped record used by these tests."""
    question: str
    answer: str
    source_benchmark: Optional[str] = None


def _make_pairs(bm: str, n: int) -> List[_Pair]:
    return [_Pair(question=f"q_{bm}_{i}", answer=f"a_{bm}_{i}",
                  source_benchmark=bm) for i in range(n)]


# ---------------------------------------------------------------------- #
# Diagnostic helper                                                      #
# ---------------------------------------------------------------------- #

def test_per_benchmark_counts_basic():
    from caem.training.pool_reweighting import per_benchmark_counts
    pairs = _make_pairs("fever", 5) + _make_pairs("triviaqa", 3) + [_Pair("q", "a")]
    counts = per_benchmark_counts(pairs)
    assert counts["fever"] == 5
    assert counts["triviaqa"] == 3
    assert counts["_untagged"] == 1


# ---------------------------------------------------------------------- #
# Temperature-mixed softmax                                              #
# ---------------------------------------------------------------------- #

def test_temperature_high_collapses_toward_uniform():
    """Very high T should make all benchmarks roughly equal in the output."""
    from caem.training.pool_reweighting import reweight_pool, per_benchmark_counts
    # Heavy imbalance: 100 fever, 10 triviaqa, 5 hotpotqa
    pairs = (
        _make_pairs("fever", 100)
        + _make_pairs("triviaqa", 10)
        + _make_pairs("hotpotqa", 5)
    )
    out = reweight_pool(
        pairs, benchmarks=["fever", "triviaqa", "hotpotqa"],
        temperature=100.0, upsample_cap=10.0, doremi_floor=0,
        cold_start_n=0, seed=1,
    )
    counts = per_benchmark_counts(out)
    # At T=100 the softmax is essentially uniform; per-bench shares
    # should be within ~30% of each other.
    n_total = sum(counts.values())
    fractions = [counts[b] / n_total for b in ("fever", "triviaqa", "hotpotqa")]
    spread = max(fractions) - min(fractions)
    assert spread < 0.30, (
        f"high-T should approach uniform; got fractions {fractions} "
        f"(spread {spread:.3f})"
    )


def test_temperature_one_preserves_empirical_proportions():
    """T=1 (no smoothing) should preserve empirical proportions modulo
    rounding, when no other constraints (cap, floor, cold-start) bind."""
    from caem.training.pool_reweighting import reweight_pool, per_benchmark_counts
    pairs = _make_pairs("fever", 80) + _make_pairs("triviaqa", 20)
    out = reweight_pool(
        pairs, benchmarks=["fever", "triviaqa"],
        temperature=1.0, upsample_cap=10.0, doremi_floor=0,
        cold_start_n=0, seed=1, target_total=100,
    )
    counts = per_benchmark_counts(out)
    # Within ±5 of the original 80/20 split
    assert abs(counts["fever"] - 80) <= 5
    assert abs(counts["triviaqa"] - 20) <= 5


# ---------------------------------------------------------------------- #
# Upsample cap                                                           #
# ---------------------------------------------------------------------- #

def test_upsample_cap_bounds_individual_replication():
    """No single sample should appear more than ceil(upsample_cap) times."""
    from caem.training.pool_reweighting import reweight_pool
    # Tiny benchmark + huge target → forces upsampling
    pairs = _make_pairs("rare", 5)
    out = reweight_pool(
        pairs, benchmarks=["rare"],
        temperature=1.0, upsample_cap=3.0, doremi_floor=0,
        cold_start_n=0, seed=1, target_total=100,
    )
    # With cap=3.0, each unique sample (q_rare_0..4) appears ≤ 3 times.
    # Total output capped at 5 unique × 3 = 15.
    assert len(out) <= 15
    counts = Counter(p.question for p in out)
    assert max(counts.values()) <= 3, (
        f"upsample cap 3.0 violated; got max replication {max(counts.values())}"
    )


def test_upsample_cap_one_means_no_upsampling():
    """upsample_cap=1.0 should refuse to replicate any sample beyond its
    original copy."""
    from caem.training.pool_reweighting import reweight_pool
    pairs = _make_pairs("rare", 4)
    out = reweight_pool(
        pairs, benchmarks=["rare"],
        temperature=1.0, upsample_cap=1.0, doremi_floor=0,
        cold_start_n=0, seed=1, target_total=100,
    )
    counts = Counter(p.question for p in out)
    assert max(counts.values()) == 1
    assert len(out) == 4


# ---------------------------------------------------------------------- #
# DoReMi floor                                                           #
# ---------------------------------------------------------------------- #

def test_doremi_floor_guarantees_minimum_per_benchmark():
    """A benchmark with 5 episodes that softmax would round to 0 must
    still get at least ``doremi_floor`` samples in the output."""
    from caem.training.pool_reweighting import reweight_pool, per_benchmark_counts
    # Massive imbalance: 1000 fever, 5 hotpotqa
    pairs = _make_pairs("fever", 1000) + _make_pairs("hotpotqa", 5)
    out = reweight_pool(
        pairs, benchmarks=["fever", "hotpotqa"],
        temperature=2.0, upsample_cap=10.0, doremi_floor=20,
        cold_start_n=0, seed=1, target_total=1000,
    )
    counts = per_benchmark_counts(out)
    assert counts["hotpotqa"] >= 20, (
        f"doremi_floor=20 violated; hotpotqa got {counts['hotpotqa']}"
    )


def test_doremi_floor_does_not_promote_zero_count_benchmarks():
    """A benchmark with zero pairs should NOT be auto-floored — it must
    go through the cold-start path explicitly."""
    from caem.training.pool_reweighting import reweight_pool, per_benchmark_counts
    pairs = _make_pairs("fever", 100)
    out = reweight_pool(
        pairs, benchmarks=["fever", "hotpotqa"],
        temperature=2.0, upsample_cap=3.0, doremi_floor=20,
        cold_start_n=0, seed=1, target_total=100,
    )
    counts = per_benchmark_counts(out)
    assert counts.get("hotpotqa", 0) == 0


# ---------------------------------------------------------------------- #
# Cold-start fallback                                                    #
# ---------------------------------------------------------------------- #

def test_cold_start_seeds_zero_count_benchmark():
    """A benchmark in the roster with no verified episodes should
    receive ``cold_start_n`` gold-labelled pairs via the loader."""
    from caem.training.pool_reweighting import reweight_pool, per_benchmark_counts
    pairs = _make_pairs("fever", 80)

    def loader(bm: str, n: int) -> List[_Pair]:
        return [
            _Pair(question=f"gold_{bm}_{i}", answer=f"gold_a_{i}",
                  source_benchmark=bm)
            for i in range(n)
        ]

    out = reweight_pool(
        pairs, benchmarks=["fever", "hotpotqa"],
        temperature=2.0, upsample_cap=3.0, doremi_floor=0,
        cold_start_n=30, cold_start_loader=loader, seed=1,
        target_total=200,
    )
    counts = per_benchmark_counts(out)
    assert counts["hotpotqa"] > 0, (
        "cold-start should have seeded hotpotqa; got 0"
    )
    # And the output's hotpotqa pairs come from the loader (gold prefix)
    hotpot_qs = {p.question for p in out if p.source_benchmark == "hotpotqa"}
    assert all(q.startswith("gold_hotpotqa_") for q in hotpot_qs), (
        f"hotpotqa pairs should be gold-loaded; got {sorted(hotpot_qs)[:3]}"
    )


def test_cold_start_loader_failure_falls_back_gracefully():
    """Loader exceptions don't crash reweight_pool; the bench stays at
    zero with a logged warning."""
    from caem.training.pool_reweighting import reweight_pool, per_benchmark_counts
    pairs = _make_pairs("fever", 50)

    def bad_loader(bm: str, n: int):
        raise RuntimeError("simulated cold-start failure")

    out = reweight_pool(
        pairs, benchmarks=["fever", "hotpotqa"],
        temperature=1.0, upsample_cap=3.0, doremi_floor=0,
        cold_start_n=30, cold_start_loader=bad_loader, seed=1,
    )
    counts = per_benchmark_counts(out)
    assert counts.get("hotpotqa", 0) == 0
    # fever still passed through
    assert counts["fever"] > 0


# ---------------------------------------------------------------------- #
# Edge cases                                                             #
# ---------------------------------------------------------------------- #

def test_empty_input_returns_empty():
    from caem.training.pool_reweighting import reweight_pool
    assert reweight_pool([]) == []
    # With cold_start_loader but no benchmarks → still empty
    def loader(bm, n):
        return [_Pair("g", "a", bm)]
    assert reweight_pool([], cold_start_loader=loader) == []


def test_default_target_total_preserves_size():
    """When target_total is None and no upsampling/cold-start kicks in,
    the output size matches the input size."""
    from caem.training.pool_reweighting import reweight_pool
    pairs = _make_pairs("fever", 60) + _make_pairs("triviaqa", 60)
    out = reweight_pool(
        pairs, benchmarks=["fever", "triviaqa"],
        temperature=1.0, upsample_cap=3.0, doremi_floor=0,
        cold_start_n=0, seed=1,
    )
    # 120 in, 120 out (within ±2 rounding)
    assert abs(len(out) - 120) <= 2


# ---------------------------------------------------------------------- #
# Wiring: SIL run_cycle invokes reweight_pool when flag is True          #
# ---------------------------------------------------------------------- #

def test_sil_run_cycle_calls_reweight_pool_when_enabled():
    """Verify the wiring is statically present: the run_cycle method
    body imports and calls reweight_pool under the cfg flag."""
    import inspect
    from caem.training.self_improvement import SelfImprovementLoop
    src = inspect.getsource(SelfImprovementLoop.run_cycle)
    assert "pool_reweighting_enabled" in src
    assert "reweight_pool(" in src
    assert "cold_start_loader=cold_start_loader" in src


def test_sil_disabled_flag_falls_back_to_flat_shuffle():
    """When pool_reweighting_enabled=False, the flat-shuffle path must
    still be present in the run_cycle source as the fallback."""
    import inspect
    from caem.training.self_improvement import SelfImprovementLoop
    src = inspect.getsource(SelfImprovementLoop.run_cycle)
    assert "train_pairs = list(episode_pairs)" in src
    assert "random.shuffle(train_pairs)" in src


def test_config_pool_reweighting_defaults():
    from caem.config import CAEMConfig
    cfg = CAEMConfig()
    assert cfg.pool_reweighting_enabled is True  # v2 default
    assert cfg.pool_reweighting_temperature == 2.0
    assert cfg.pool_reweighting_upsample_cap == 3.0
    assert cfg.pool_reweighting_doremi_floor == 50
    assert cfg.pool_reweighting_cold_start_n == 100


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:
                failures += 1
                print(f"  ERR   {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
    print(f"\nAll tests passed.")
