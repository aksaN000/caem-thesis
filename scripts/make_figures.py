"""
scripts/make_figures.py
=======================
Phase 4m.6 -- Render the seven Chapter 5 figures from the same artefacts
scripts/make_tables.py consumes.

Figure manifest (indexing matches Ch.5):
  Fig 5.1  CES radar         -- 5-axis (ACC, EPI, RET, CAL, VER) per cycle overlay
  Fig 5.2  Reliability diag. -- calibration bins from per_sample_signals.jsonl
  Fig 5.3  Halluc. decomp.   -- grouped bars: hallucination / confab. / early-exit
  Fig 5.4  Grounding evol.   -- p_ground_max / _mean / _atomic / p_contra across cycles
  Fig 5.5  Purity breakdown  -- stacked bar STORE / DEFERRED / ABSTAIN / DISCARD
  Fig 5.6  Continual learn.  -- MMLU retention + per-benchmark EM trajectories
  Fig 5.7  EM progression    -- lines: mean EM per benchmark per cycle

Inputs
------
Directory containing (at minimum):
  * per_sample_signals.jsonl       -- from eval.reporting.write_per_sample_signals_jsonl
  * tab_headline.csv               -- Table 5.1 data (required for Figures 5.1/5.7)
  * tab_calibration.csv            -- Table 5.2
  * tab_halluc.csv                 -- Table 5.3
  * tab_grounding.csv              -- Table 5.4
  * tab_purity.csv                 -- Table 5.5
  * tab_continual.csv              -- Table 5.6

Each figure is best-effort -- if the required CSV is missing or has no rows,
the figure is skipped (logged to stderr) and the script moves on.

Output
------
PNG + PDF pair per figure:
  fig5_1_ces_radar.{png,pdf}
  fig5_2_reliability.{png,pdf}
  fig5_3_halluc_decomp.{png,pdf}
  fig5_4_grounding.{png,pdf}
  fig5_5_purity.{png,pdf}
  fig5_6_continual.{png,pdf}
  fig5_7_em_progression.{png,pdf}

PDFs are the canonical thesis-ingest format; PNGs are for slide decks /
quick previews. Each is saved at 300 dpi with tight bounding boxes.

CLI
---
::

    python scripts/make_figures.py <output_dir>
    python scripts/make_figures.py runs/cycle_3/

See also: scripts/make_tables.py (companion LaTeX generator).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Ensure repo root is on sys.path so `from eval...` imports work when this
# script is invoked from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Non-interactive matplotlib backend -- required for headless / CI runs.
import matplotlib  # noqa: E402  # type: ignore[import-untyped]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # type: ignore[import-untyped]
import numpy as np  # noqa: E402


logger = logging.getLogger("caem.make_figures")


# -----------------------------------------------------------------------------
# Figure file-name manifest
# -----------------------------------------------------------------------------

FIGURE_FILES: Dict[str, str] = {
    "ces_radar":      "fig5_1_ces_radar",
    "reliability":    "fig5_2_reliability",
    "halluc_decomp":  "fig5_3_halluc_decomp",
    "grounding":      "fig5_4_grounding",
    "purity":         "fig5_5_purity",
    "continual":      "fig5_6_continual",
    "em_progression": "fig5_7_em_progression",
}

# Default dpi/format settings used by every figure.
_DEFAULT_DPI = 300
_SAVE_KW = dict(dpi=_DEFAULT_DPI, bbox_inches="tight")


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV to a list of dicts. Empty list if path is missing."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read per_sample_signals.jsonl. Empty list if missing."""
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("make_figures: skipping malformed JSONL line (%s).", exc)
    return out


def _parse_float(cell: Any) -> Optional[float]:
    """Coerce CSV cell -> float; NaN / empty / 'NaN' map to None."""
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except ValueError:
        return None


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    """Save ``fig`` as both PNG and PDF, return (png_path, pdf_path)."""
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, **_SAVE_KW)
    fig.savefig(pdf_path, **_SAVE_KW)
    plt.close(fig)
    return png_path, pdf_path


# -----------------------------------------------------------------------------
# Figure 5.1 -- CES radar
# -----------------------------------------------------------------------------

# The five CES axes as stored by eval.metrics.ces_score(). Order is fixed;
# the axes are rendered clockwise on the radar in this same order.
_CES_AXES = ("ACC", "EPI", "RET", "CAL", "VER")


def _ces_axes_from_headline(headline_rows: Sequence[Mapping[str, Any]],
                            calib_rows: Sequence[Mapping[str, Any]],
                            continual_rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, float]]:
    """Recover best-available per-cycle (ACC, EPI, RET, CAL, VER) tuples.

    The raw CES scalar is already in tab_headline.csv on the
    ``benchmark == __cycle__`` rows, but the five underlying axes are not
    persisted individually. We reconstruct them approximately from the three
    CSVs where they live:

      ACC = mean EM on the __cycle__ summary row  (tab_headline)
      EPI = 1 - min(hallucination_rate_mean, 0.5)  (tab_halluc via caller)
      RET = mmlu_retention_ratio                   (tab_continual)
      CAL = 1 - 2 * min(ECE, 0.5)                  (tab_calibration)
      VER = 0.5 fallback                           (verifier-BA unavailable)

    We take the hallucination rate from the caller (passed separately) since
    this helper stays pure.
    """
    axes: Dict[int, Dict[str, float]] = {}

    # ACC from the __cycle__ summary rows of tab_headline.
    for row in headline_rows:
        if row.get("benchmark") != "__cycle__":
            continue
        cycle = int(_parse_float(row.get("cycle")) or 0)
        acc = _parse_float(row.get("em")) or 0.0
        axes.setdefault(cycle, {})["ACC"] = acc

    # CAL from tab_calibration (1 - 2*ECE, clipped).
    for row in calib_rows:
        cycle = int(_parse_float(row.get("cycle")) or 0)
        ece = _parse_float(row.get("ece"))
        if ece is None:
            cal = float("nan")
        else:
            cal = 1.0 - 2.0 * min(ece, 0.5)
        axes.setdefault(cycle, {})["CAL"] = cal

    # RET from tab_continual.
    for row in continual_rows:
        cycle = int(_parse_float(row.get("cycle")) or 0)
        ret = _parse_float(row.get("mmlu_retention_ratio"))
        if ret is not None:
            axes.setdefault(cycle, {})["RET"] = ret

    return axes


def figure_ces_radar(
    out_dir: Path,
    headline_rows: Sequence[Mapping[str, Any]],
    halluc_rows: Sequence[Mapping[str, Any]],
    calib_rows: Sequence[Mapping[str, Any]],
    continual_rows: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[Path, Path]]:
    """Render Figure 5.1 (CES radar) to ``out_dir``. Returns paths or None."""
    if not headline_rows:
        logger.warning("figure_ces_radar: no headline data; skipping.")
        return None

    axes_map = _ces_axes_from_headline(headline_rows, calib_rows, continual_rows)
    if not axes_map:
        logger.warning("figure_ces_radar: could not recover CES axes; skipping.")
        return None

    # EPI from the cross-benchmark mean hallucination rate per cycle.
    epi_by_cycle: Dict[int, List[float]] = {}
    for row in halluc_rows:
        cycle = int(_parse_float(row.get("cycle")) or 0)
        hr = _parse_float(row.get("hallucination_rate"))
        if hr is not None:
            epi_by_cycle.setdefault(cycle, []).append(hr)
    for cycle, vals in epi_by_cycle.items():
        mean_hr = sum(vals) / len(vals) if vals else 0.0
        axes_map.setdefault(cycle, {})["EPI"] = 1.0 - min(mean_hr, 0.5)

    # VER is the 0.5 placeholder (balanced-accuracy of the verifier is not
    # directly observable without gold STORE/DISCARD labels).
    for cycle in axes_map:
        axes_map[cycle].setdefault("VER", 0.5)

    # Build the radar figure.
    cycles_sorted = sorted(axes_map.keys())
    n_ax = len(_CES_AXES)
    theta = np.linspace(0, 2 * math.pi, n_ax, endpoint=False).tolist()
    theta += theta[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(_CES_AXES, fontsize=10)

    cmap = plt.get_cmap("viridis")
    for i, cycle in enumerate(cycles_sorted):
        values = [axes_map[cycle].get(a, float("nan")) for a in _CES_AXES]
        # NaN -> 0 for plotting; the axis label alone conveys the missingness.
        plot_values = [v if (v is not None and not math.isnan(v)) else 0.0 for v in values]
        plot_values += plot_values[:1]
        color = cmap(i / max(len(cycles_sorted) - 1, 1))
        ax.plot(theta, plot_values, color=color, linewidth=2, label=f"Cycle {cycle}")
        ax.fill(theta, plot_values, color=color, alpha=0.15)

    ax.set_title("CES axes across cycles", pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=9, frameon=False)
    return _save(fig, out_dir, FIGURE_FILES["ces_radar"])


# -----------------------------------------------------------------------------
# Figure 5.2 -- Reliability diagram (per cycle)
# -----------------------------------------------------------------------------

def figure_reliability(
    out_dir: Path,
    jsonl_rows: Sequence[Mapping[str, Any]],
    n_bins: int = 10,
) -> Optional[Tuple[Path, Path]]:
    """Render Figure 5.2 (reliability diagram) from per-sample signals.

    Groups (u_stored, em) pairs into equal-width bins and plots observed
    accuracy vs. mean confidence per cycle. The diagonal y=x line is the
    perfect-calibration reference.
    """
    if not jsonl_rows:
        logger.warning("figure_reliability: no per_sample_signals.jsonl rows; skipping.")
        return None

    by_cycle: Dict[int, List[Tuple[float, float]]] = {}
    for rec in jsonl_rows:
        u = rec.get("u_stored")
        em = rec.get("em")
        if u is None or em is None:
            continue
        try:
            u = float(u); em = float(em)
        except (TypeError, ValueError):
            continue
        if math.isnan(u) or math.isnan(em):
            continue
        by_cycle.setdefault(int(rec.get("cycle", 0)), []).append((u, em))

    if not by_cycle:
        logger.warning("figure_reliability: no valid (u_stored, em) pairs; skipping.")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1, label="perfect calibration")

    cycles_sorted = sorted(by_cycle.keys())
    cmap = plt.get_cmap("viridis")
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i, cycle in enumerate(cycles_sorted):
        pairs = by_cycle[cycle]
        confs = np.array([p[0] for p in pairs])
        labels = np.array([p[1] for p in pairs])
        mean_conf, mean_acc = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            # Half-open bins except for the final bin which includes 1.0.
            mask = (confs >= lo) & (confs < hi)
            if hi == 1.0:
                mask |= confs == 1.0
            if mask.sum() == 0:
                continue
            mean_conf.append(confs[mask].mean())
            mean_acc.append(labels[mask].mean())
        if mean_conf:
            color = cmap(i / max(len(cycles_sorted) - 1, 1))
            ax.plot(mean_conf, mean_acc, "-o", color=color,
                    linewidth=1.8, markersize=5, label=f"Cycle {cycle}")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean $\\hat{u}_{\\text{stored}}$ in bin", fontsize=10)
    ax.set_ylabel("Observed EM accuracy", fontsize=10)
    ax.set_title("Reliability of composite stored confidence", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    return _save(fig, out_dir, FIGURE_FILES["reliability"])


# -----------------------------------------------------------------------------
# Figure 5.3 -- Hallucination decomposition
# -----------------------------------------------------------------------------

def figure_halluc_decomp(
    out_dir: Path,
    halluc_rows: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[Path, Path]]:
    """Render Figure 5.3 (hallucination / confabulation / early-exit bars)."""
    if not halluc_rows:
        logger.warning("figure_halluc_decomp: no hallucination data; skipping.")
        return None

    # Average across benchmarks per cycle -- the chapter's thesis statement
    # is about the aggregated trend, not per-benchmark comparisons.
    per_cycle: Dict[int, Dict[str, List[float]]] = {}
    for row in halluc_rows:
        cycle = int(_parse_float(row.get("cycle")) or 0)
        slot = per_cycle.setdefault(cycle, {"hr": [], "cr": [], "ee": []})
        for key, col in (("hr", "hallucination_rate"),
                         ("cr", "confabulation_rate"),
                         ("ee", "early_exit_rate")):
            v = _parse_float(row.get(col))
            if v is not None:
                slot[key].append(v)

    cycles = sorted(per_cycle.keys())
    if not cycles:
        logger.warning("figure_halluc_decomp: no rows after parsing; skipping.")
        return None

    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    hr = [mean(per_cycle[c]["hr"]) for c in cycles]
    cr = [mean(per_cycle[c]["cr"]) for c in cycles]
    ee = [mean(per_cycle[c]["ee"]) for c in cycles]

    x = np.arange(len(cycles))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, 1 + 1.2 * len(cycles)), 4.5))
    ax.bar(x - width, hr, width, label="Hallucination rate", color="#d7263d")
    ax.bar(x,         cr, width, label="Confabulation rate", color="#f46036")
    ax.bar(x + width, ee, width, label="Early-exit rate",    color="#2e86ab")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cycle {c}" for c in cycles])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(1.0, max(hr + cr + ee) * 1.2) if (hr + cr + ee) else 1.0)
    ax.set_title("Hallucination decomposition across cycles")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    return _save(fig, out_dir, FIGURE_FILES["halluc_decomp"])


# -----------------------------------------------------------------------------
# Figure 5.4 -- Grounding evolution
# -----------------------------------------------------------------------------

def figure_grounding(
    out_dir: Path,
    grounding_rows: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[Path, Path]]:
    """Render Figure 5.4 (grounding signals across cycles)."""
    if not grounding_rows:
        logger.warning("figure_grounding: no grounding data; skipping.")
        return None

    cycles, gmax, gmean, gatom, gcontra, usc = [], [], [], [], [], []
    for row in sorted(grounding_rows, key=lambda r: int(_parse_float(r.get("cycle")) or 0)):
        cycles.append(int(_parse_float(row.get("cycle")) or 0))
        gmax.append(_parse_float(row.get("mean_p_ground_max")) or 0.0)
        gmean.append(_parse_float(row.get("mean_p_ground_mean")) or 0.0)
        gatom.append(_parse_float(row.get("mean_p_ground_atomic")) or 0.0)
        gcontra.append(_parse_float(row.get("mean_p_contra")) or 0.0)
        usc.append(_parse_float(row.get("unsupported_correct_rate")) or 0.0)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    ax0.plot(cycles, gmax,  "-o", label=r"$p_{ground,max}$",    color="#1b9e77")
    ax0.plot(cycles, gmean, "-s", label=r"$p_{ground,mean}$",   color="#d95f02")
    ax0.plot(cycles, gatom, "-^", label=r"$p_{ground,atomic}$", color="#7570b3")
    ax0.plot(cycles, gcontra, "--v", label=r"$p_{contra}$",     color="#e7298a")
    ax0.set_xlabel("Cycle"); ax0.set_ylabel("Mean probability")
    ax0.set_ylim(0, 1); ax0.set_xticks(cycles)
    ax0.set_title("Grounding / contradiction trajectory")
    ax0.grid(True, alpha=0.3); ax0.legend(loc="best", fontsize=9, frameon=False)

    ax1.bar(cycles, usc, color="#8e44ad")
    ax1.set_xlabel("Cycle"); ax1.set_ylabel("Unsupported-correct rate")
    ax1.set_xticks(cycles)
    ax1.set_ylim(0, max(0.5, max(usc) * 1.2) if usc else 0.5)
    ax1.set_title("\"Right for the wrong reason\"")
    ax1.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return _save(fig, out_dir, FIGURE_FILES["grounding"])


# -----------------------------------------------------------------------------
# Figure 5.5 -- Purity stacked bars
# -----------------------------------------------------------------------------

def figure_purity(
    out_dir: Path,
    purity_rows: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[Path, Path]]:
    """Render Figure 5.5 (Stage-5 decision stacked bars + mean u_stored line)."""
    if not purity_rows:
        logger.warning("figure_purity: no purity data; skipping.")
        return None

    rows = sorted(purity_rows, key=lambda r: int(_parse_float(r.get("cycle")) or 0))
    cycles = [int(_parse_float(r.get("cycle")) or 0) for r in rows]
    n_s = [_parse_float(r.get("n_store"))    or 0 for r in rows]
    n_d = [_parse_float(r.get("n_deferred")) or 0 for r in rows]
    n_a = [_parse_float(r.get("n_abstain"))  or 0 for r in rows]
    n_x = [_parse_float(r.get("n_discard"))  or 0 for r in rows]
    mu  = [_parse_float(r.get("mean_u_stored_store")) for r in rows]

    totals = [s + d + a + x for s, d, a, x in zip(n_s, n_d, n_a, n_x)]
    # Convert to fractions (safe-divide so totals == 0 renders as zero bar).
    def frac(values, totals):
        return [v / t if t > 0 else 0.0 for v, t in zip(values, totals)]
    f_s = frac(n_s, totals); f_d = frac(n_d, totals)
    f_a = frac(n_a, totals); f_x = frac(n_x, totals)

    x = np.arange(len(cycles))
    fig, ax1 = plt.subplots(figsize=(max(7, 1 + 1.5 * len(cycles)), 5))
    ax1.bar(x, f_s, label="STORE",    color="#2ecc71")
    ax1.bar(x, f_d, bottom=f_s, label="DEFERRED", color="#f1c40f")
    ax1.bar(x, f_a, bottom=[a + b for a, b in zip(f_s, f_d)],
            label="ABSTAIN", color="#95a5a6")
    ax1.bar(x, f_x, bottom=[a + b + c for a, b, c in zip(f_s, f_d, f_a)],
            label="DISCARD", color="#e74c3c")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Cycle {c}" for c in cycles])
    ax1.set_ylabel("Fraction of verified episodes")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Stage-5 decision breakdown + stored composite confidence")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22),
               ncol=4, fontsize=9, frameon=False)
    ax1.grid(True, axis="y", alpha=0.3)

    # Secondary axis: mean u_stored on stored episodes.
    ax2 = ax1.twinx()
    valid = [(xi, v) for xi, v in zip(x, mu) if v is not None and not math.isnan(v)]
    if valid:
        xs, ys = zip(*valid)
        ax2.plot(xs, ys, "k-o", linewidth=1.8, markersize=6,
                 label=r"mean $\hat{u}_{\text{stored}}$ | STORE")
        ax2.set_ylabel(r"mean $\hat{u}_{\text{stored}}$ (STORE only)")
        ax2.set_ylim(0, 1)

    fig.tight_layout()
    return _save(fig, out_dir, FIGURE_FILES["purity"])


# -----------------------------------------------------------------------------
# Figure 5.6 -- Continual learning
# -----------------------------------------------------------------------------

def figure_continual(
    out_dir: Path,
    continual_rows: Sequence[Mapping[str, Any]],
    headline_rows: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[Path, Path]]:
    """Render Figure 5.6 (MMLU retention + per-benchmark EM trajectories)."""
    if not continual_rows and not headline_rows:
        logger.warning("figure_continual: nothing to plot; skipping.")
        return None

    # Left subplot: MMLU retention ratio and absolute MMLU%.
    cycles = []
    mmlu_pct = []
    mmlu_ret = []
    for row in sorted(continual_rows, key=lambda r: int(_parse_float(r.get("cycle")) or 0)):
        cycles.append(int(_parse_float(row.get("cycle")) or 0))
        mmlu_pct.append(_parse_float(row.get("mmlu_pct")))
        mmlu_ret.append(_parse_float(row.get("mmlu_retention_ratio")))

    # Right subplot: per-benchmark EM trajectories from tab_headline.
    per_bm: Dict[str, Dict[int, float]] = {}
    for row in headline_rows:
        bm = row.get("benchmark")
        if bm == "__cycle__" or not bm:
            continue
        cyc = int(_parse_float(row.get("cycle")) or 0)
        em = _parse_float(row.get("em"))
        if em is not None:
            per_bm.setdefault(bm or "", {})[cyc] = em

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))

    if cycles:
        # Convert mmlu_pct -> fraction for plot if values look like percentages.
        mmlu_frac = [(v / 100.0) if (v is not None and v > 1.5) else v for v in mmlu_pct]
        ax0.plot(cycles, mmlu_frac, "-s", color="#2c3e50", label="MMLU accuracy")
        ax0.plot(cycles, mmlu_ret,  "-o", color="#16a085", label="MMLU retention ratio")
        ax0.set_xticks(cycles); ax0.set_xlabel("Cycle"); ax0.set_ylabel("Value")
        ax0.set_ylim(0, 1.05); ax0.grid(True, alpha=0.3)
        ax0.set_title("MMLU retention across cycles")
        ax0.legend(loc="best", fontsize=9, frameon=False)
    else:
        ax0.set_visible(False)

    if per_bm:
        cmap = plt.get_cmap("tab10")
        for i, (bm, curve) in enumerate(sorted(per_bm.items())):
            xs = sorted(curve.keys())
            ys = [curve[c] for c in xs]
            ax1.plot(xs, ys, "-o", color=cmap(i % 10), label=bm, linewidth=1.8)
        ax1.set_xlabel("Cycle"); ax1.set_ylabel("Exact match")
        ax1.set_ylim(0, 1); ax1.grid(True, alpha=0.3)
        ax1.set_title("Per-benchmark EM trajectory")
        ax1.legend(loc="best", fontsize=8, frameon=False)
    else:
        ax1.set_visible(False)

    fig.tight_layout()
    return _save(fig, out_dir, FIGURE_FILES["continual"])


# -----------------------------------------------------------------------------
# Figure 5.7 -- EM progression (cross-benchmark mean)
# -----------------------------------------------------------------------------

def figure_em_progression(
    out_dir: Path,
    headline_rows: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[Path, Path]]:
    """Render Figure 5.7 -- cross-benchmark mean EM + per-benchmark overlays."""
    if not headline_rows:
        logger.warning("figure_em_progression: no headline data; skipping.")
        return None

    mean_em: Dict[int, float] = {}
    per_bm: Dict[str, Dict[int, float]] = {}
    for row in headline_rows:
        cyc = int(_parse_float(row.get("cycle")) or 0)
        bm = row.get("benchmark")
        em = _parse_float(row.get("em"))
        if em is None:
            continue
        if bm == "__cycle__":
            mean_em[cyc] = em
        else:
            per_bm.setdefault(bm or "", {})[cyc] = em

    if not mean_em and not per_bm:
        logger.warning("figure_em_progression: no EM values; skipping.")
        return None

    cycles = sorted(set(list(mean_em.keys()) + [c for curve in per_bm.values() for c in curve]))

    fig, ax = plt.subplots(figsize=(max(7, 1 + 1.2 * len(cycles)), 4.5))

    if per_bm:
        cmap = plt.get_cmap("tab10")
        for i, (bm, curve) in enumerate(sorted(per_bm.items())):
            xs = sorted(curve.keys())
            ys = [curve[c] for c in xs]
            ax.plot(xs, ys, "-o", color=cmap(i % 10), alpha=0.75,
                    linewidth=1.4, markersize=4, label=bm)

    if mean_em:
        xs = sorted(mean_em.keys())
        ys = [mean_em[c] for c in xs]
        ax.plot(xs, ys, "k-", linewidth=2.6, label="cross-benchmark mean")
        ax.plot(xs, ys, "ko", markersize=6)

    ax.set_xticks(cycles); ax.set_xlabel("Cycle"); ax.set_ylabel("Exact match")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    ax.set_title("EM progression across self-improvement cycles")
    ax.legend(loc="best", fontsize=8, frameon=False)
    return _save(fig, out_dir, FIGURE_FILES["em_progression"])


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------

def build_all_figures(output_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Generate every Chapter 5 figure for which input data is present."""
    output_dir = Path(output_dir)

    headline_rows  = _read_csv_rows(output_dir / "tab_headline.csv")
    calib_rows     = _read_csv_rows(output_dir / "tab_calibration.csv")
    halluc_rows    = _read_csv_rows(output_dir / "tab_halluc.csv")
    grounding_rows = _read_csv_rows(output_dir / "tab_grounding.csv")
    purity_rows    = _read_csv_rows(output_dir / "tab_purity.csv")
    continual_rows = _read_csv_rows(output_dir / "tab_continual.csv")
    jsonl_rows     = _read_jsonl(output_dir / "per_sample_signals.jsonl")

    manifest: Dict[str, Tuple[Path, Path]] = {}

    builders = [
        ("ces_radar", lambda: figure_ces_radar(
            output_dir, headline_rows, halluc_rows, calib_rows, continual_rows)),
        ("reliability", lambda: figure_reliability(output_dir, jsonl_rows)),
        ("halluc_decomp", lambda: figure_halluc_decomp(output_dir, halluc_rows)),
        ("grounding", lambda: figure_grounding(output_dir, grounding_rows)),
        ("purity", lambda: figure_purity(output_dir, purity_rows)),
        ("continual", lambda: figure_continual(output_dir, continual_rows, headline_rows)),
        ("em_progression", lambda: figure_em_progression(output_dir, headline_rows)),
    ]
    for key, fn in builders:
        try:
            paths = fn()
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning("make_figures: builder %s raised %s -- skipping.", key, exc)
            paths = None
        if paths is not None:
            manifest[key] = paths
            logger.info("make_figures: wrote %s (%s + %s)",
                        key, paths[0].name, paths[1].name)

    return manifest


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render the seven Chapter 5 figures from eval.reporting artefacts."
        ),
    )
    p.add_argument("output_dir", type=Path, help="Directory containing tab_*.csv + jsonl.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logging.")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.output_dir.exists():
        logger.error("output_dir does not exist: %s", args.output_dir)
        return 2

    manifest = build_all_figures(args.output_dir)
    if not manifest:
        logger.error("no figures generated -- are the tab_*.csv files present?")
        return 1

    for k, (png, pdf) in manifest.items():
        print(f"{k}\t{png}\t{pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
