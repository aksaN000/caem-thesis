"""
scripts/epistemic_gate.py
==========================
Pre-integration Moskvoretskii epistemic gate (Branch C, mandatory before
``feat/phase-2-all`` merge).

Why this script exists
----------------------
Moskvoretskii et al. 2025 reported that downstream success in DRAGIN,
SeaKR, and Adaptive-RAG correlates poorly with the underlying
self-knowledge-identification signal that those pipelines' routers consume
(Spearman ρ near zero on several datasets). If CAEM's ``u_stored`` has the
same defect, the tier-routing contribution claim is cosmetic, not load-
bearing. Building the main-run compute on an unvalidated foundation is
reckless -- branch_C.md §"Evaluation protocol" makes this gate a hard
precondition for the integration merge.

The Flan-T5 Phase-1a interim finding (ρ ≈ 0 on EM-labels) does NOT
invalidate ``u_stored``. EM on open-ended QA is confounded by paraphrase
failures, repetition loops, and label-extraction bugs (documented in
branch_C_log.md §"Phase 1a interim interpretation"). The correct proxy is
a **faithfulness label** from a high-quality external judge (MiniCheck on
a held-out subset, or cross-check via multiple judges), NOT EM.

Protocol
--------
1. Inputs
     - ``per_sample_signals.jsonl`` produced by the run's eval harness.
       Each record carries ``u_stored`` and either (a) the signals needed
       to re-score via MiniCheck (``question``, ``answer``, ``top_passages``)
       or (b) a pre-computed ``is_faithful`` label.
     - Optional labels JSON (``--labels_path``): an array of
       ``{question_hash: float_in_[0,1]}`` pairs. Label 1.0 = the judge
       marks the answer as supported; 0.0 = not supported. Graded labels
       (e.g. MiniCheck's unary P(supported)) are also accepted.
2. Computation
     - Per-benchmark Spearman ρ(u_stored, is_faithful) with 95% bootstrap
       CI (n=1000 resamples, seed=42).
     - Pooled Spearman across all benchmarks.
3. Decision thresholds (per branch_C.md §Evaluation protocol)
     | Pooled ρ  | Action                                              |
     |-----------|-----------------------------------------------------|
     | > 0.5     | PROCEED. u_stored is load-bearing.                  |
     | 0.3 < ρ ≤ 0.5 | PROCEED_WITH_HONESTY. Ch5 §honesty paragraph;  |
     |           | position tier routing as "moderately correlated"    |
     |           | not "strongly correlated".                          |
     | ≤ 0.3     | STOP. Do NOT commit Branch-C compute. Diagnose the  |
     |           | correlation problem before proceeding.              |

Artefacts
---------
  outputs/epistemic_gate/epistemic_gate_report.json
      Machine-readable: per-benchmark + pooled ρ + CI + n + decision.
  outputs/epistemic_gate/epistemic_gate_report.md
      Human-readable summary for inclusion in the Ch5 appendix.

Typical usage
-------------
    PYTHONPATH=. python scripts/epistemic_gate.py \\
        --signals_path outputs/per_sample_signals.jsonl \\
        --labels_path outputs/faithfulness_labels.json \\
        --output_dir outputs/epistemic_gate

Testing
-------
The statistics + decision logic are pure Python and covered by
``tests/test_epistemic_gate.py``. The live-judge labelling path is tested
manually against the MiniCheck judge on Vast.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Decision thresholds (per branch_C.md)
# =============================================================================

PROCEED_THRESHOLD: float = 0.5
HONESTY_THRESHOLD: float = 0.3


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class GateResult:
    """Outcome of the gate on one axis (per-benchmark or pooled)."""

    axis: str                       # "pooled" or benchmark name
    n: int
    rho: float
    ci_lo: float
    ci_hi: float
    decision: str                   # "PROCEED" | "PROCEED_WITH_HONESTY" | "STOP"


@dataclass
class GateReport:
    """Full gate output: per-benchmark axes + pooled + overall decision."""

    pooled: GateResult
    per_benchmark: List[GateResult] = field(default_factory=list)
    n_bootstrap: int = 1000
    random_seed: int = 42
    # Carries the pooled decision so downstream CI can gate on a single field.
    overall_decision: str = ""


# =============================================================================
# Spearman + bootstrap CI (pure NumPy)
# =============================================================================

def _rankdata(x: List[float]) -> List[float]:
    """Average-rank tiebreaker -- matches scipy.stats.rankdata(method='average').

    Implemented inline so the script runs in environments without SciPy
    (the CAEM eval stack only mandates NumPy).
    """
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Walk forward over the tie group.
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0   # 1-indexed average
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Return Spearman rho. Falls back to 0.0 when either axis is constant."""
    if len(x) != len(y):
        raise ValueError(
            f"spearman: length mismatch ({len(x)} vs {len(y)})"
        )
    n = len(x)
    if n < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    sx = sum((r - mx) ** 2 for r in rx)
    sy = sum((r - my) ** 2 for r in ry)
    if sx <= 0 or sy <= 0:
        # At least one axis is constant -> Pearson-on-ranks is undefined.
        return 0.0
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    rho = cov / (sx ** 0.5 * sy ** 0.5)
    # Clamp to handle FP drift.
    return max(-1.0, min(1.0, rho))


def spearman_with_ci(
    x: List[float],
    y: List[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Return (rho, ci_lo, ci_hi).

    Bootstrap over paired (x, y) resamples; percentile CI at ``ci_level``.
    When ``n_bootstrap <= 0`` or the sample size is too small to resample,
    returns (rho, rho, rho) so the CI collapses cleanly.
    """
    rho = _spearman_rho(x, y)
    n = len(x)
    if n < 2 or n_bootstrap <= 0:
        return rho, rho, rho

    rng = random.Random(seed)
    boot_rhos: List[float] = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        rx = [x[i] for i in idx]
        ry = [y[i] for i in idx]
        boot_rhos.append(_spearman_rho(rx, ry))
    boot_rhos.sort()
    alpha = (1.0 - ci_level) / 2.0
    lo_idx = max(0, int(alpha * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1.0 - alpha) * n_bootstrap) - 1)
    return rho, boot_rhos[lo_idx], boot_rhos[hi_idx]


# =============================================================================
# Decision tree
# =============================================================================

def gate_decision(rho: float) -> str:
    """Map a pooled ρ to the three-way decision per branch_C.md."""
    if rho > PROCEED_THRESHOLD:
        return "PROCEED"
    if rho > HONESTY_THRESHOLD:
        return "PROCEED_WITH_HONESTY"
    return "STOP"


# =============================================================================
# Data loading
# =============================================================================

def load_signals_jsonl(path: Path) -> List[dict]:
    """Load per-sample records from a JSONL file; skip malformed lines."""
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "signals: line %d malformed (%s) -- skipped.", lineno, exc,
                )
    logger.info("signals: loaded %d rows from %s", len(rows), path)
    return rows


def pair_u_stored_with_labels(
    rows: List[dict],
    labels: Dict[str, float],
    *,
    key: str = "question",
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Pair ``u_stored`` with ``labels`` on ``key`` (default ``question``).

    Records missing ``u_stored`` or whose ``key`` field does not appear in
    ``labels`` are dropped. Returns a dict keyed by benchmark with per-axis
    (u_stored_list, label_list). A "pooled" entry collects all benchmarks.
    """
    per_bm: Dict[str, Tuple[List[float], List[float]]] = {}
    pooled_u: List[float] = []
    pooled_f: List[float] = []
    for r in rows:
        u = r.get("u_stored")
        k = r.get(key)
        if u is None or k is None:
            continue
        if k not in labels:
            continue
        try:
            uf = float(u)
            lf = float(labels[k])
        except (TypeError, ValueError):
            continue
        bm = str(r.get("benchmark") or "unknown")
        lists = per_bm.setdefault(bm, ([], []))
        lists[0].append(uf)
        lists[1].append(lf)
        pooled_u.append(uf)
        pooled_f.append(lf)
    per_bm["pooled"] = (pooled_u, pooled_f)
    return per_bm


# =============================================================================
# Gate driver
# =============================================================================

def run_gate(
    rows: List[dict],
    labels: Dict[str, float],
    *,
    key: str = "question",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> GateReport:
    """Run the full gate and return a GateReport."""
    paired = pair_u_stored_with_labels(rows, labels, key=key)

    per_bm_results: List[GateResult] = []
    for bm, (xs, ys) in paired.items():
        if bm == "pooled":
            continue
        rho, lo, hi = spearman_with_ci(xs, ys, n_bootstrap=n_bootstrap, seed=seed)
        per_bm_results.append(GateResult(
            axis=bm, n=len(xs), rho=rho, ci_lo=lo, ci_hi=hi,
            decision=gate_decision(rho),
        ))

    pooled_u, pooled_f = paired.get("pooled", ([], []))
    pooled_rho, lo, hi = spearman_with_ci(
        pooled_u, pooled_f, n_bootstrap=n_bootstrap, seed=seed,
    )
    pooled_result = GateResult(
        axis="pooled", n=len(pooled_u), rho=pooled_rho, ci_lo=lo, ci_hi=hi,
        decision=gate_decision(pooled_rho),
    )

    per_bm_results.sort(key=lambda r: r.axis)
    return GateReport(
        pooled=pooled_result,
        per_benchmark=per_bm_results,
        n_bootstrap=n_bootstrap,
        random_seed=seed,
        overall_decision=pooled_result.decision,
    )


# =============================================================================
# Reporting
# =============================================================================

def report_to_markdown(report: GateReport) -> str:
    """Render a human-readable summary for the Ch5 appendix."""
    lines: List[str] = []
    lines.append("# Epistemic Gate Report\n")
    lines.append(f"Bootstrap resamples: {report.n_bootstrap} (seed={report.random_seed})\n")
    lines.append(f"**Pooled decision: {report.overall_decision}**\n")
    lines.append("")
    lines.append("## Per-axis results\n")
    lines.append("| Axis | n | Spearman ρ | 95% CI | Decision |")
    lines.append("|---|---|---|---|---|")
    for r in report.per_benchmark + [report.pooled]:
        lines.append(
            f"| {r.axis} | {r.n} | {r.rho:+.4f} | "
            f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.decision} |"
        )
    lines.append("")
    lines.append("## Decision thresholds\n")
    lines.append("- ρ > 0.5 → PROCEED")
    lines.append("- 0.3 < ρ ≤ 0.5 → PROCEED_WITH_HONESTY")
    lines.append("- ρ ≤ 0.3 → STOP")
    lines.append("")
    return "\n".join(lines)


def write_report(report: GateReport, output_dir: Path) -> Tuple[Path, Path]:
    """Write JSON + Markdown reports. Returns (json_path, md_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "epistemic_gate_report.json"
    md_path = output_dir / "epistemic_gate_report.md"
    json_path.write_text(json.dumps(asdict(report), indent=2))
    md_path.write_text(report_to_markdown(report))
    return json_path, md_path


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-integration Moskvoretskii epistemic gate for CAEM Branch C.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--signals_path", type=Path, required=True,
        help="Path to per_sample_signals.jsonl (u_stored + metadata).",
    )
    p.add_argument(
        "--labels_path", type=Path, required=True,
        help="Path to a JSON file mapping question (or --label_key) to a "
             "faithfulness label in [0, 1].",
    )
    p.add_argument(
        "--label_key", default="question",
        help="Record field used to index into the labels JSON.",
    )
    p.add_argument(
        "--output_dir", type=Path, default=Path("outputs/epistemic_gate"),
        help="Directory to write epistemic_gate_report.{json,md}.",
    )
    p.add_argument(
        "--n_bootstrap", type=int, default=1000,
        help="Bootstrap resamples for the 95% CI (0 to disable).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap random seed.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ns = _build_parser().parse_args(argv)
    rows = load_signals_jsonl(ns.signals_path)
    with ns.labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, dict):
        logger.error("labels_path must contain a JSON object mapping key -> float.")
        return 2
    report = run_gate(
        rows, labels,
        key=ns.label_key, n_bootstrap=ns.n_bootstrap, seed=ns.seed,
    )
    json_path, md_path = write_report(report, ns.output_dir)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    logger.info("Pooled decision: %s", report.overall_decision)
    # Exit code tracks the decision so CI can gate directly on this script.
    if report.overall_decision == "STOP":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
