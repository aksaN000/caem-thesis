"""
caem/diagnostic/coverage.py
============================
v2 Fix 5 — per-cycle coverage feedback diagnostic + halt triggers.

Writes a small JSON summary at the end of every SIL cycle so the
operator (and downstream report generators) can detect:

  * **Zero-admission collapse**: a training benchmark whose per-bench
    conformal gate has been admitting zero verified episodes for 2+
    consecutive cycles. Triggers an auto-relax of α_b (or escalates
    to a manual halt if relaxation has already maxed out).
  * **Pool composition drift**: Shannon entropy over the per-bench
    proportions in the SIL training pool. Flat pool (all FEVER) →
    low entropy; uniform pool (every benchmark equal) → log(N_b).
  * **Adapter SVD collapse**: spectral signature of the LoRA
    adapter weights. Healthy adapters spread mass across many
    singular values; rank-collapse to a single dominant SV is a
    Biderman-2024 contribution-gap symptom (the adapter has
    memorised one direction and stopped learning).

This module is intentionally model-agnostic and mostly pure-Python so
the SIL trajectory keeps writing diagnostics even when peft / torch
are unavailable in the runtime image.

Public surface
--------------
* :func:`per_benchmark_admission_rates` — admission_count[b] /
  candidate_count[b] for each benchmark; NaN on zero candidates.
* :func:`pool_composition_entropy` — Shannon entropy in nats over a
  benchmark count dict.
* :func:`adapter_singular_values` — top-k singular values of every
  LoRA adapter matrix in a peft-wrapped model. Returns a dict mapping
  ``module_name -> List[float]``. Returns ``{}`` on any failure
  (peft missing, torch missing, model not wrapped).
* :func:`build_coverage_diagnostic` — one-cycle bundle: admission
  rates + entropy + SV spectra + per-bench EM (passed in) packaged
  as a JSON-serialisable dict.
* :func:`evaluate_halt_triggers` — given a sequence of recent
  diagnostics, return a HaltDecision describing whether to auto-relax
  α_b, halt, or continue.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Per-benchmark admission rates                                                  #
# ---------------------------------------------------------------------------- #

def per_benchmark_admission_rates(
    candidate_counts: Dict[str, int],
    admitted_counts: Dict[str, int],
) -> Dict[str, float]:
    """Compute admission_count / candidate_count for each benchmark.

    A benchmark with zero candidates returns NaN (no signal — the
    guard treats NaN as "inactive" for that benchmark, mirroring the
    retention-probe convention).
    """
    out: Dict[str, float] = {}
    for bm, cand in candidate_counts.items():
        if cand <= 0:
            out[bm] = float("nan")
            continue
        adm = int(admitted_counts.get(bm, 0))
        out[bm] = float(adm) / float(cand)
    # Surface admitted-only benchmarks (unusual but possible) as 1.0
    # since every admitted sample passed the gate.
    for bm in admitted_counts:
        if bm not in out:
            out[bm] = 1.0
    return out


# ---------------------------------------------------------------------------- #
# Shannon entropy over a benchmark count dict                                    #
# ---------------------------------------------------------------------------- #

def pool_composition_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy (in nats) over the empirical benchmark distribution.

    Empty / single-benchmark pool → 0.0. Uniform N-bench pool → ln(N).
    """
    total = sum(int(c) for c in counts.values() if c is not None)
    if total <= 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        c = int(c)
        if c <= 0:
            continue
        p = c / total
        H -= p * math.log(p)
    return float(H)


# ---------------------------------------------------------------------------- #
# Adapter singular values (LoRA contribution spectrum)                          #
# ---------------------------------------------------------------------------- #

def adapter_singular_values(
    model: Any,
    *,
    top_k: int = 8,
) -> Dict[str, List[float]]:
    """Return top-``top_k`` singular values of every LoRA-adapter matrix.

    For each named parameter whose name contains ``lora_A`` or
    ``lora_B``, computes ``torch.linalg.svdvals`` on its 2D weight and
    keeps the largest ``top_k`` values.

    Returns ``{}`` on any failure (peft missing, model not wrapped,
    torch unavailable, weights not 2D). Failure is logged once but
    NEVER raised — the diagnostic must not abort a training cycle.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        logger.info("adapter_singular_values: torch unavailable — skipping.")
        return {}

    out: Dict[str, List[float]] = {}
    try:
        named = list(model.named_parameters())
    except Exception as exc:
        logger.warning("adapter_singular_values: enumerate failed (%s).", exc)
        return {}

    for name, p in named:
        if not ("lora_A" in name or "lora_B" in name):
            continue
        try:
            w = p.detach()
            if w.ndim != 2:
                continue
            svs = torch.linalg.svdvals(w.float()).cpu().tolist()
        except Exception as exc:
            logger.debug(
                "adapter_singular_values: SVD failed on %s (%s).", name, exc,
            )
            continue
        # Sort descending and keep top-k
        svs_sorted = sorted(svs, reverse=True)[: max(1, top_k)]
        out[name] = [float(s) for s in svs_sorted]
    return out


def adapter_sv_collapse_score(sv_spectra: Dict[str, List[float]]) -> float:
    """Heuristic in [0, 1] reporting how concentrated the LoRA adapter
    contribution is on its top singular value.

    For each adapter matrix, computes ``s_1 / sum(s_i)`` (the
    participation ratio of the top SV). Averages across all matrices.

    * 0.0 → mass spread evenly across all kept SVs (healthy)
    * 1.0 → all mass on the top SV (rank-collapse)

    Returns NaN when no SV spectra were captured.
    """
    if not sv_spectra:
        return float("nan")
    ratios: List[float] = []
    for name, svs in sv_spectra.items():
        total = sum(s for s in svs if s > 0)
        if total <= 0:
            continue
        ratios.append(svs[0] / total)
    if not ratios:
        return float("nan")
    return float(sum(ratios) / len(ratios))


# ---------------------------------------------------------------------------- #
# Halt triggers                                                                  #
# ---------------------------------------------------------------------------- #

@dataclass
class HaltDecision:
    """Result of :func:`evaluate_halt_triggers`."""
    action: str  # "continue" | "relax_alpha" | "halt"
    reason: str = ""
    benchmarks_to_relax: List[str] = field(default_factory=list)
    sv_collapse_score: float = float("nan")


def evaluate_halt_triggers(
    recent_diagnostics: Sequence[Dict[str, Any]],
    *,
    zero_admission_window: int = 2,
    sv_collapse_threshold: float = 0.95,
) -> HaltDecision:
    """Decide whether the trajectory should auto-relax α_b, halt, or continue.

    Inputs
    ------
    recent_diagnostics : sequence of cycle diagnostic dicts (most
        recent last). Each dict must include at minimum the keys
        produced by :func:`build_coverage_diagnostic`:
        ``admission_rates`` (Dict[str, float]),
        ``adapter_sv_collapse`` (float).
    zero_admission_window : int, default 2
        A benchmark whose admission rate has been exactly 0.0 for
        this many consecutive recent cycles triggers the auto-relax
        signal. The rule fires only when at least
        ``zero_admission_window`` diagnostics are available.
    sv_collapse_threshold : float, default 0.95
        adapter_sv_collapse score above this on the most recent
        cycle triggers a halt with reason "sv_collapse".

    Returns
    -------
    HaltDecision
    """
    if not recent_diagnostics:
        return HaltDecision(action="continue")

    latest = recent_diagnostics[-1]

    # Trigger 1: SV collapse (single-cycle, hard halt).
    sv_score = float(latest.get("adapter_sv_collapse", float("nan")))
    if not math.isnan(sv_score) and sv_score >= sv_collapse_threshold:
        return HaltDecision(
            action="halt",
            reason=(
                f"adapter_sv_collapse {sv_score:.3f} "
                f">= threshold {sv_collapse_threshold:.3f}"
            ),
            sv_collapse_score=sv_score,
        )

    # Trigger 2: zero-admission for ``window`` consecutive cycles.
    if len(recent_diagnostics) >= zero_admission_window:
        window = list(recent_diagnostics)[-zero_admission_window:]
        # For each benchmark seen in any window cycle, check if it was
        # 0.0 in EVERY window cycle (NaN = "inactive", excluded).
        all_benchmarks = set()
        for d in window:
            ar = d.get("admission_rates", {}) or {}
            all_benchmarks.update(ar.keys())
        zeroes: List[str] = []
        for bm in all_benchmarks:
            rates = []
            for d in window:
                r = (d.get("admission_rates", {}) or {}).get(bm, float("nan"))
                rates.append(r)
            # All finite and exactly zero?
            finite = [r for r in rates if r is not None and not math.isnan(r)]
            if (
                len(finite) == zero_admission_window
                and all(r == 0.0 for r in finite)
            ):
                zeroes.append(bm)
        if zeroes:
            return HaltDecision(
                action="relax_alpha",
                reason=(
                    f"zero admission for {zero_admission_window} consecutive "
                    f"cycles: {sorted(zeroes)}"
                ),
                benchmarks_to_relax=sorted(zeroes),
                sv_collapse_score=sv_score,
            )

    return HaltDecision(action="continue", sv_collapse_score=sv_score)


# ---------------------------------------------------------------------------- #
# Bundle                                                                         #
# ---------------------------------------------------------------------------- #

def build_coverage_diagnostic(
    cycle_num: int,
    *,
    candidate_counts: Dict[str, int],
    admitted_counts: Dict[str, int],
    pool_counts: Dict[str, int],
    per_bench_em: Optional[Dict[str, float]] = None,
    model: Optional[Any] = None,
    sv_top_k: int = 8,
) -> Dict[str, Any]:
    """Build a single-cycle coverage diagnostic dict ready for JSON dump.

    The dict includes:
      * cycle_num
      * admission_rates: per-benchmark admitted/candidate
      * pool_counts:      per-benchmark count in the SIL training pool
        (after Fix 3 reweighting)
      * pool_entropy:     Shannon entropy over pool_counts (in nats)
      * per_bench_em:     per-benchmark eval EM (caller-supplied)
      * adapter_sv_spectra: top-k SVs per LoRA adapter weight
      * adapter_sv_collapse: scalar 0..1 reporting top-SV mass share
    """
    admission_rates = per_benchmark_admission_rates(
        candidate_counts, admitted_counts,
    )
    pool_entropy = pool_composition_entropy(pool_counts)
    spectra = adapter_singular_values(model, top_k=sv_top_k) if model is not None else {}
    collapse = adapter_sv_collapse_score(spectra)

    return {
        "cycle_num": int(cycle_num),
        "admission_rates": admission_rates,
        "candidate_counts": dict(candidate_counts),
        "admitted_counts": dict(admitted_counts),
        "pool_counts": dict(pool_counts),
        "pool_entropy": float(pool_entropy),
        "per_bench_em": dict(per_bench_em or {}),
        "adapter_sv_spectra": spectra,
        "adapter_sv_collapse": collapse,
    }
