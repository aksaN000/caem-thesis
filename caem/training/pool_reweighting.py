"""
caem/training/pool_reweighting.py
====================================
v2 Fix 3 — loss reweighting for the SIL training pool.

Problem
-------
v1 used a flat training pool: every verified episode is added once and
the SIL loop iterates over the shuffled list. When one benchmark stores
materially more entries than others (FEVER monoculture in v1 cycles 0-4
— TriviaQA storage rate hit 0% by cycle 2), the gradient signal becomes
benchmark-imbalanced and the model drifts toward the dominant
benchmark's distribution. The Phase-1a audit traced ~17.7% of the
pooled-EM regression on v1 cycles 0-4 to this imbalance.

Solution
--------
Replace the flat pool with a four-mechanism reweighting that interpolates
between proportional sampling (every benchmark gets a fair share) and
empirical sampling (verified-episode counts dominate):

  1. **Temperature-mixed softmax** (T=2 default, Du et al. 2022 / DoReMi
     2023 §3.2): smooth the empirical per-benchmark proportions so a
     ~10× count gap collapses to ~3× weight gap. Bounds dominance.
        p_b      = empirical proportion of bench b
        p_b'     = softmax(log(p_b) / T)
     T → 1  reduces to empirical sampling (no smoothing).
     T → ∞  reduces to uniform sampling (every bench equal).

  2. **3× upsampling cap**: a low-count benchmark may be upsampled to
     match the target proportion, but each individual sample is
     replicated AT MOST ``upsample_cap`` × times. Prevents the
     pathological "replicate the same five FEVER NEI samples 200×"
     failure mode that destroyed cycle-1 storage diversity in v1.

  3. **DoReMi minimum floor** (Xie et al. 2023): ``floor_n`` samples
     per benchmark guaranteed in the final pool, even if temperature
     smoothing would have rounded the share down to zero. Provides a
     hard guarantee that every training benchmark sees gradient signal.

  4. **Cold-start gold fallback**: when a benchmark has zero verified
     episodes (e.g. cycle-0 with empty memory store, or a benchmark
     whose fixed-threshold gate never admitted a sample yet), fall back to
     ``cold_start_n`` gold-labelled samples drawn via an injected
     ``cold_start_loader(benchmark, n)`` function. Without this guard,
     a never-stored benchmark gets zero gradient signal forever.

The four mechanisms compose: cold-start runs FIRST (so empty benchmarks
get a nonzero count), then temperature-mixed targets are computed,
clipped to the upsample cap, and finally floored at DoReMi.

Public surface
--------------
* :func:`reweight_pool` — the actual algorithm. Takes a flat list of
  QAPairs (each carrying source_benchmark) and returns a reweighted
  flat list ready to feed the SIL DataLoader.
* :func:`per_benchmark_counts` — diagnostic helper. Counts pairs per
  benchmark, used by the coverage diagnostic (Fix 5) and the cycle
  log.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Diagnostic helper                                                              #
# ---------------------------------------------------------------------------- #

def per_benchmark_counts(pairs: List[Any]) -> Dict[str, int]:
    """Count QAPairs by source_benchmark. None / missing tag → '_untagged'."""
    counts: Dict[str, int] = {}
    for p in pairs:
        bm = getattr(p, "source_benchmark", None) or "_untagged"
        counts[bm] = counts.get(bm, 0) + 1
    return counts


# ---------------------------------------------------------------------------- #
# Temperature-mixed softmax over per-benchmark proportions                       #
# ---------------------------------------------------------------------------- #

def _temperature_mixed_targets(
    counts: Dict[str, int],
    target_total: int,
    *,
    temperature: float,
) -> Dict[str, int]:
    """Compute per-benchmark target counts via temperature-mixed softmax.

    For empirical proportions ``p_b = counts[b] / sum(counts)``::

        p_b'   = softmax(log(p_b) / T)
        target = round(p_b' * target_total)

    Benchmarks whose counts[b] == 0 receive target = 0 from this step
    (cold-start handles them before this function is called).
    """
    if not counts:
        return {}
    total = sum(counts.values())
    if total <= 0:
        return {b: 0 for b in counts}

    if temperature <= 0:
        # Degenerate: fall back to uniform.
        share = target_total // max(len(counts), 1)
        return {b: share for b in counts}

    log_ps = {}
    for b, n in counts.items():
        if n <= 0:
            log_ps[b] = -math.inf
        else:
            log_ps[b] = math.log(n / total) / temperature

    # Stable softmax over finite entries
    finite = {b: lp for b, lp in log_ps.items() if math.isfinite(lp)}
    if not finite:
        return {b: 0 for b in counts}
    max_lp = max(finite.values())
    exps = {b: math.exp(lp - max_lp) for b, lp in finite.items()}
    Z = sum(exps.values())
    if Z <= 0:
        return {b: 0 for b in counts}
    targets = {b: 0 for b in counts}
    for b, e in exps.items():
        targets[b] = int(round(target_total * e / Z))
    return targets


# ---------------------------------------------------------------------------- #
# Upsample / downsample one benchmark's pairs to a target count                  #
# ---------------------------------------------------------------------------- #

def _resize_benchmark(
    pairs: List[Any],
    target: int,
    *,
    upsample_cap: float,
    rng: random.Random,
) -> List[Any]:
    """Resize ``pairs`` to length ``target`` via subsample / capped upsample.

    Upsampling replicates samples uniformly at random; each individual
    pair appears at most ``ceil(upsample_cap)`` times in the result,
    bounding the "duplicate the same sample 100×" failure mode.
    Downsampling is uniform random subsampling without replacement.
    """
    if target <= 0 or not pairs:
        return []
    n = len(pairs)
    if target == n:
        return list(pairs)
    if target < n:
        return rng.sample(pairs, target)

    # Upsample with cap. Each pair starts at copy-count 1 (the original).
    # Repeatedly draw uniform indices and increment copy counts; reject
    # any draw that would push a single pair above the cap.
    cap_int = max(1, int(math.ceil(upsample_cap)))
    counts = [1] * n
    deficit = target - n
    # Available draws = sum(cap - count) across pairs. If less than
    # deficit, we can't reach target; emit what we can.
    max_extra = sum(cap_int - c for c in counts)
    actual_deficit = min(deficit, max_extra)
    while actual_deficit > 0:
        idx = rng.randrange(n)
        if counts[idx] < cap_int:
            counts[idx] += 1
            actual_deficit -= 1
    out: List[Any] = []
    for pair, c in zip(pairs, counts):
        for _ in range(c):
            out.append(pair)
    if len(out) < target:
        logger.info(
            "_resize_benchmark: upsample_cap=%.1f limited target %d → %d "
            "(n=%d unique pairs).", upsample_cap, target, len(out), n,
        )
    return out


# ---------------------------------------------------------------------------- #
# Public entry point                                                            #
# ---------------------------------------------------------------------------- #

def reweight_pool(
    pairs: List[Any],
    *,
    benchmarks: Optional[List[str]] = None,
    target_total: Optional[int] = None,
    temperature: float = 2.0,
    upsample_cap: float = 3.0,
    doremi_floor: int = 50,
    cold_start_n: int = 100,
    cold_start_loader: Optional[Callable[[str, int], List[Any]]] = None,
    pool_max_share: float = 0.40,
    seed: int = 42,
) -> List[Any]:
    """Build a benchmark-balanced training pool from raw verified-episode pairs.

    Parameters
    ----------
    pairs : list of QAPair-like
        Each must expose a ``source_benchmark`` attribute (None allowed
        for legacy / unit-test paths; treated as ``_untagged``).
    benchmarks : list of str or None
        The full benchmark roster to honour. Benchmarks listed here but
        absent from ``pairs`` trigger the cold-start fallback. When
        None, the roster is inferred from ``pairs`` alone (cold-start
        is then a no-op since every benchmark in scope already has at
        least one pair).
    target_total : int or None
        Desired total pool size. When None, defaults to
        ``len(pairs)`` (i.e. don't grow the pool, just rebalance).
    temperature : float, default 2.0
        Smoothing temperature for the per-benchmark softmax. T=1 is
        empirical sampling; T → ∞ is uniform.
    upsample_cap : float, default 3.0
        Maximum multiplier for individual sample replication.
    doremi_floor : int, default 50
        Minimum guaranteed pair count per benchmark in the output (DoReMi-
        style hard floor; ensures every benchmark sees gradient signal).
    cold_start_n : int, default 100
        Seed count for benchmarks with zero verified episodes.
    cold_start_loader : callable or None
        ``(benchmark: str, n: int) -> List[QAPair]`` returning gold-
        labelled pairs for the cold-start fallback. None disables the
        fallback (zero-count benchmarks stay at zero).
    pool_max_share : float, default 0.40
        Phase 1c (P3b) — hard ceiling on any single benchmark's share of
        the final pool. Bench-agnostic: applies to whichever benchmark
        ends up dominant after temperature smoothing + DoReMi floor.
        Excess from over-cap benches is water-filled into under-cap
        benches (proportional to their current targets, iteratively
        until convergence). Set to 1.0 to disable. Default 0.40 means
        no benchmark may take more than 40% of the pool — a thesis-
        defensible upper bound that survives panel growth (e.g. adding
        a 4th training bench that ships much more than FEVER).
    seed : int, default 42

    Returns
    -------
    list of QAPair-like
        The reweighted pool, shuffled once at the end.
    """
    rng = random.Random(seed)
    if not pairs and cold_start_loader is None:
        return []

    # Group by benchmark
    grouped: Dict[str, List[Any]] = {}
    for p in pairs:
        bm = getattr(p, "source_benchmark", None) or "_untagged"
        grouped.setdefault(bm, []).append(p)

    # Cold-start: extend roster with explicit benchmarks, draw gold for
    # the zero-count members.
    if benchmarks is not None and cold_start_loader is not None:
        for bm in benchmarks:
            if grouped.get(bm):
                continue
            try:
                seeds = list(cold_start_loader(bm, cold_start_n) or [])
            except Exception as exc:
                logger.warning(
                    "reweight_pool: cold_start_loader(%s, %d) raised %s; "
                    "skipping cold-start for this benchmark.",
                    bm, cold_start_n, exc,
                )
                seeds = []
            if seeds:
                grouped[bm] = seeds
                logger.info(
                    "reweight_pool: cold-start seeded %d gold pairs for "
                    "benchmark '%s' (no verified episodes available).",
                    len(seeds), bm,
                )

    # Determine target_total. By default: keep the pool size.
    counts = {bm: len(g) for bm, g in grouped.items()}
    n_input = sum(counts.values())
    if n_input == 0:
        return []
    if target_total is None:
        target_total = n_input

    # Step 1: temperature-mixed proportions
    targets = _temperature_mixed_targets(
        counts, target_total, temperature=temperature,
    )

    # Step 2: enforce DoReMi floor (hard minimum per benchmark)
    floor = max(0, int(doremi_floor))
    if floor > 0:
        for bm in targets:
            if counts.get(bm, 0) > 0 and targets[bm] < floor:
                targets[bm] = floor

    # Step 2b: enforce hard pool_max_share cap (P3b — bench-agnostic).
    # Iteratively cap any benchmark exceeding pool_max_share·target_total and
    # redistribute the excess proportionally to under-cap benches. Iterates
    # until convergence (excess < 1 sample) or 10 rounds (defensive cap).
    if 0.0 < pool_max_share < 1.0 and target_total > 0:
        share_cap = max(1, int(target_total * pool_max_share))
        for _round in range(10):
            over = {b: t for b, t in targets.items() if t > share_cap}
            if not over:
                break
            excess = sum(t - share_cap for t in over.values())
            for b in over:
                targets[b] = share_cap
            # Redistribute proportionally to benches under the cap with
            # nonzero target (DoReMi-floored benches qualify).
            under = {
                b: t for b, t in targets.items()
                if b not in over and t > 0 and t < share_cap
            }
            under_total = sum(under.values())
            if under_total <= 0 or not under:
                logger.info(
                    "reweight_pool.share_cap: no under-cap benches to absorb "
                    "excess=%d (round=%d); pool will fall short of target_total.",
                    excess, _round,
                )
                break
            for b, t in under.items():
                add = int(round(excess * t / under_total))
                # Don't push the recipient above the cap.
                targets[b] = min(share_cap, t + add)
        else:
            logger.warning(
                "reweight_pool.share_cap: did not converge in 10 rounds; "
                "final targets=%s (pool_max_share=%.2f, target_total=%d).",
                dict(sorted(targets.items())), pool_max_share, target_total,
            )

    # Step 3: resize each benchmark to its target via subsample / capped
    # upsample. The target may be unreachable when upsample_cap × n < target;
    # that's logged inside _resize_benchmark and we simply use whatever
    # _resize_benchmark returns.
    out: List[Any] = []
    for bm, g in grouped.items():
        t = targets.get(bm, 0)
        resized = _resize_benchmark(g, t, upsample_cap=upsample_cap, rng=rng)
        out.extend(resized)

    rng.shuffle(out)

    # Diagnostic: log the input → output transform per benchmark.
    out_counts = per_benchmark_counts(out)
    logger.info(
        "reweight_pool: T=%.1f cap=%.1f floor=%d max_share=%.2f cold_start_n=%d | "
        "input(%d): %s → output(%d): %s",
        temperature, upsample_cap, doremi_floor, pool_max_share, cold_start_n,
        n_input, dict(sorted(counts.items())),
        len(out), dict(sorted(out_counts.items())),
    )
    return out
