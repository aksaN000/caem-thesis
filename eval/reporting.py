"""
eval/reporting.py
=================
Post-run data pipeline that reshapes the per-benchmark-per-cycle JSONs
emitted by ``EvalHarness.run(...)`` into the seven tables Chapter 5
reports plus a single flat ``per_sample_signals.jsonl`` record stream.

Relationship to sibling modules
-------------------------------
* ``eval/harness.py``  produces ``{benchmark}_cycle{cycle}.json`` --
    the atomic source of truth. Each file carries a ``meta`` aggregate
    dict and a ``samples`` list of rectangular per-sample records (the
    twelve-field Session-42 verifier capture plus EM/F1/tier/latency).
* ``eval/metrics.py``  provides the twelve primitive helpers
    (hallucination_rate, confabulation_rate, reliability_bins,
    decision_breakdown, backward_transfer, ces_score, ...). Reporting
    composes them; it does not invent new metrics.
* ``scripts/run_experiment.py``  calls ``build_ch5_tables(output_dir)``
    at the end of the orchestration so the seven CSVs and the JSONL
    land next to ``experiment_summary.csv``.
* ``scripts/make_tables.py``  (Phase 4m.5) will read the seven CSVs
    written here and emit the ``.tex`` table bodies that ``\\input{}``
    into Chapter 5. Keeping the intermediate CSV layer makes table
    regeneration cheap (no re-run of the pipeline).

Seven-table layout (Chapter 5 §5.1 preface)
-------------------------------------------
* ``tab_headline.csv``    -- Table 5.1  accuracy, latency, routing, CES
* ``tab_calibration.csv`` -- Table 5.2  ECE, Brier, reliability bins
* ``tab_halluc.csv``      -- Table 5.3  hallucination decomposition
* ``tab_grounding.csv``   -- Table 5.4  p_ground*, unsupported-correct
* ``tab_purity.csv``      -- Table 5.5  decision breakdown, mean u_stored
* ``tab_continual.csv``   -- Table 5.6  MMLU retention, forgetting,
                                         BWT / FWT when >=2 cycles
* ``tab_cycle_progression.csv`` -- within-CAEM cycle-over-cycle
                                         bootstrap CIs & McNemar p-values
                                         for the (cycle N vs cycle 0)
                                         EM comparison. NOT to be confused
                                         with ``tab_sig_test.csv`` below.

The CAEM-vs-baseline significance table that chapter_5.tex references as
``\\ref{tab:sig-test}`` (CAEM vs each B1-B8 baseline, Holm-corrected
per-benchmark family with dagger/star markers) is produced by a separate
script ``scripts/baseline_sig_tests.py`` and lands as ``tab_sig_test.csv``
alongside the files above. That script runs post-baselines in the Plan A
runner; this module only covers the within-CAEM tables.

Rectangular flat file
---------------------
* ``per_sample_signals.jsonl`` -- one JSON record per line containing
    cycle, benchmark, and every field from the harness SampleResult
    (including the twelve verifier fields). This is the dataset that
    the figure generator (Phase 4m.6, CES radar + reliability diagrams)
    reads. It is also what future analyses (e.g. unsupported-correct
    breakdowns by question type) should consume.

Design notes
------------
* All table builders return ``(fieldnames, rows)`` pairs so the CSV
  writer is a single helper. Empty tables are allowed (returned as
  ``(fieldnames, [])``) so downstream consumers see a stable schema
  even when a pipeline stage was skipped.
* Values that would require a missing input are emitted as ``None``
  or ``float('nan')`` rather than raising. Readers (make_tables.py,
  analysis notebooks) treat NaN as "not computed this run" and render
  it as ``--``.
* Bootstrap CIs and McNemar p-values are computed only when ``len(cycles)
  >= 2``. Table 5.7 is otherwise written with the header only, which
  is the honest representation of a single-cycle smoke test.

Session 42 context
------------------
This module post-dates the StoredConfidence -> UnifiedVerifierOutput
migration (Phase 5c). It assumes every sample record already carries
the twelve verifier fields; older JSON files from pre-Session-42 runs
will produce NaNs for the verifier-derived tables, which is the
correct silent behaviour.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.metrics import (
    backward_transfer,
    bootstrap_ci,
    brier_score,
    ces_score,
    confabulation_rate,
    decision_breakdown,
    forward_transfer,
    hallucination_rate,
    mcnemar_test,
    reliability_bins,
    unsupported_correct_rate,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Verifier fields captured per sample (Phase 4m.3).  Imported from
# ``eval.harness`` so a single dataclass-derived tuple feeds both the
# per-sample JSONL writer (eval/harness.py) and the table builders below.
# Previously a duplicate hand-written tuple — silent drift from the
# verifier dataclass was the failure mode fixed by MAJOR-H3 / Task #118.
from eval.harness import VERIFIER_FIELDS  # noqa: E402  (re-export)

# Threshold used by the hallucination_rate helper's companion,
# unsupported_correct_rate, when the caller does not override it.
DEFAULT_GROUND_THRESHOLD = 0.30

# Threshold used by the "confident" side of the confabulation rate --
# matches the early-exit gate in the UnifiedVerifier decision tree.
DEFAULT_U_INTERNAL_CONFIDENT = 0.70


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def _discover_cycle_json_files(output_dir: Path) -> List[Path]:
    """Return every ``{benchmark}_cycle{cycle}.json`` file under ``output_dir``.

    The harness writes one such file per (benchmark, cycle) pair. We glob
    them up front so later passes do not have to re-enumerate the directory.
    """
    return sorted(output_dir.glob("*_cycle*.json"))


def _parse_cycle_from_filename(path: Path) -> Tuple[Optional[int], Optional[str]]:
    """Extract (cycle, benchmark) from a ``{bm}_cycle{N}.json`` filename.

    Returns ``(None, None)`` if the filename does not match the pattern,
    which lets the caller skip stray files without crashing the report.
    """
    stem = path.stem  # e.g. "fever_cycle2"
    if "_cycle" not in stem:
        return None, None
    bm, _, cycle_part = stem.rpartition("_cycle")
    try:
        cycle = int(cycle_part)
    except ValueError:
        return None, None
    return cycle, bm


def load_cycle_data(output_dir: Path) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """Load every per-benchmark-per-cycle JSON under ``output_dir``.

    Returns
    -------
    dict: ``{cycle_num: {benchmark: {"meta": agg_dict, "samples": [...]}}}``.
    Cycles and benchmarks are ordered by natural sort order (cycle ascending,
    benchmark alphabetical) for stable downstream iteration.
    """
    output_dir = Path(output_dir)
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for p in _discover_cycle_json_files(output_dir):
        cycle, bm = _parse_cycle_from_filename(p)
        if cycle is None or bm is None:
            logger.debug("reporting: skipping unrecognised file %s", p.name)
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("reporting: failed to read %s (%s) -- skipping.", p, exc)
            continue
        out.setdefault(cycle, {})[bm] = payload
    return {c: dict(sorted(v.items())) for c, v in sorted(out.items())}


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    """Atomic-ish CSV write (open-write-close) with UTF-8 encoding.

    Kept in one place so every table flushes identically (same line
    endings, same quoting, same encoding). Empty row lists are allowed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("reporting: wrote %s (%d rows).", path.name, len(rows))


def _safe_mean(xs: Sequence[float]) -> float:
    """Mean of a sequence, returning NaN when empty.

    Using NaN rather than 0 keeps downstream ``make_tables.py`` from
    mistaking "no data this run" for "observed zero".
    """
    xs = [x for x in xs if x is not None and not _isnan(x)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _isnan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _round_or_nan(x: float, ndigits: int = 4) -> Any:
    """Round to ndigits or return the string ``NaN`` so CSV stays grep-able.

    A numeric NaN round-trips through CSV as an empty cell (no ``nan``
    token), which make_tables.py cannot distinguish from a real zero.
    We write the literal string ``NaN`` instead.
    """
    if x is None or _isnan(x):
        return "NaN"
    return round(float(x), ndigits)


# -----------------------------------------------------------------------------
# per_sample_signals.jsonl
# -----------------------------------------------------------------------------

def write_per_sample_signals_jsonl(
    cycles_data: Dict[int, Dict[str, Dict[str, Any]]],
    output_path: Path,
) -> int:
    """Flatten every sample from every (cycle, benchmark) into one JSONL.

    Each record is a sample dict augmented with ``cycle`` and ``benchmark``
    keys (these are already present in the sample record from Phase 4m.3,
    but we set them explicitly here so stray/legacy files are corrected).

    Returns
    -------
    int
        Number of records written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for cycle, bm_map in cycles_data.items():
            for bm, payload in bm_map.items():
                for sample in payload.get("samples", []):
                    record = dict(sample)
                    record["cycle"] = cycle
                    record["benchmark"] = bm
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n += 1
    logger.info("reporting: wrote %s (%d records).", output_path.name, n)
    return n


# -----------------------------------------------------------------------------
# Per-cycle aggregates (precomputed so every table can reuse them cheaply)
# -----------------------------------------------------------------------------

def _collect_per_cycle(cycles_data: Dict[int, Dict[str, Dict[str, Any]]]):
    """Precompute flattened per-cycle signal lists.

    Returns a dict keyed by cycle with the following lists (concatenated
    across benchmarks):
        em, f1, tier, u_stored, latency_ms, stored, escalated,
        u_internal, p_ground_max, p_ground_mean, p_ground_atomic,
        p_contra, decision, early_exit_triggered

    Computing these once up-front means the six signal-driven tables
    (halluc, grounding, purity, ...) can iterate flat Python lists
    without re-opening any JSON.
    """
    out: Dict[int, Dict[str, List[Any]]] = {}
    keys = (
        "em", "f1", "tier", "u_stored", "latency_ms", "stored", "escalated",
        "u_internal", "p_ground_max", "p_ground_mean", "p_ground_atomic",
        "p_contra", "decision", "early_exit_triggered",
    )
    for cycle, bm_map in cycles_data.items():
        agg: Dict[str, List[Any]] = {k: [] for k in keys}
        for bm, payload in bm_map.items():
            for s in payload.get("samples", []):
                for k in keys:
                    agg[k].append(s.get(k))
        out[cycle] = agg
    return out


# -----------------------------------------------------------------------------
# Table 5.1 -- Headline
# -----------------------------------------------------------------------------

def build_table_headline(cycles_data, mmlu_per_cycle=None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Chapter 5 Table 5.1: EM / F1 / latency / routing / storage / CES.

    One row per (cycle, benchmark) plus one summary row per cycle where
    ``benchmark == '__cycle__'`` that carries the cross-benchmark mean EM
    and the cycle-level CES. Downstream ``make_tables.py`` filters on
    benchmark to split the two parts into headline and summary blocks.
    """
    fieldnames = [
        "cycle", "benchmark", "n",
        "em", "f1", "mean_latency_ms",
        "tier1_frac", "tier2_frac", "tier3_frac",
        "storage_rate", "mean_u_stored",
        "ces",
    ]
    rows: List[Dict[str, Any]] = []

    for cycle, bm_map in cycles_data.items():
        # -- per-benchmark rows -------------------------------------------- #
        for bm, payload in bm_map.items():
            meta = payload.get("meta", {})
            rows.append({
                "cycle": cycle,
                "benchmark": bm,
                "n": meta.get("n", len(payload.get("samples", []))),
                "em": _round_or_nan(meta.get("em")),
                "f1": _round_or_nan(meta.get("f1")),
                "mean_latency_ms": _round_or_nan(meta.get("mean_latency_ms"), 1),
                "tier1_frac": _round_or_nan(meta.get("tier1_frac")),
                "tier2_frac": _round_or_nan(meta.get("tier2_frac")),
                "tier3_frac": _round_or_nan(meta.get("tier3_frac")),
                "storage_rate": _round_or_nan(meta.get("storage_rate")),
                "mean_u_stored": _round_or_nan(meta.get("mean_u_stored")),
                "ces": "",   # CES is a cycle-level metric; leave blank on per-BM rows.
            })

        # -- cycle summary row: mean EM + CES ------------------------------ #
        ems = [p.get("meta", {}).get("em", float("nan")) for p in bm_map.values()]
        mean_em = _safe_mean(ems)

        # CES inputs: ACC is mean-EM; EPI is 1 - confident-error-rate
        # (hallucination_rate proxy at the u_stored=0.5 gate); RET uses
        # MMLU retention ratio if supplied; CAL is 1 - 2*ECE clipped;
        # VER is left at 0.5 in this lightweight compute (the actual
        # verifier balanced-accuracy requires gold labels for STORE vs
        # DISCARD, which are not available from eval alone).
        per_cycle = _collect_per_cycle({cycle: bm_map})[cycle]
        acc = mean_em if not _isnan(mean_em) else 0.0

        try:
            epi = 1.0 - hallucination_rate(
                per_cycle["em"],
                per_cycle["u_stored"],
                u_threshold=0.50,
            )
        except Exception:
            epi = float("nan")

        if mmlu_per_cycle and cycle < len(mmlu_per_cycle):
            base = mmlu_per_cycle[0] if len(mmlu_per_cycle) > 0 else None
            m = mmlu_per_cycle[cycle]
            if base and base > 0 and not _isnan(m):
                ret = min(m / base, 1.0)
            else:
                ret = float("nan")
        else:
            ret = float("nan")

        # ECE from reliability bins over (u_stored, em) pairs.
        conf = [c for c in per_cycle["u_stored"] if c is not None]
        labels = [e for c, e in zip(per_cycle["u_stored"], per_cycle["em"]) if c is not None]
        if conf:
            try:
                bins = reliability_bins(conf, labels, n_bins=10)
                total = sum(b[2] for b in bins) or 1
                ece = sum((b[2] / total) * abs(b[0] - b[1]) for b in bins)
                cal = 1.0 - 2.0 * min(ece, 0.5)
            except Exception:
                cal = float("nan")
        else:
            cal = float("nan")

        ver = 0.5  # Placeholder; Table 5.2's verifier-balanced-accuracy is
                   # computed when labelled STORE/DISCARD ground truth lands.

        # ces_score clamps each axis to [eps, 1]; NaN inputs fall back to eps.
        inputs = [acc, epi, ret, cal, ver]
        clean = [(0.01 if _isnan(x) else x) for x in inputs]
        ces = ces_score(*clean)

        rows.append({
            "cycle": cycle,
            "benchmark": "__cycle__",
            "n": sum(p.get("meta", {}).get("n", 0) for p in bm_map.values()),
            "em": _round_or_nan(mean_em),
            "f1": "",
            "mean_latency_ms": "",
            "tier1_frac": "",
            "tier2_frac": "",
            "tier3_frac": "",
            "storage_rate": "",
            "mean_u_stored": "",
            "ces": _round_or_nan(ces, 4),
        })

    return fieldnames, rows


# -----------------------------------------------------------------------------
# Table 5.2 -- Calibration
# -----------------------------------------------------------------------------

def build_table_calibration(cycles_data) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Chapter 5 Table 5.2: ECE, Brier, top-bin accuracy.

    Confidence signal: ``u_stored``. Labels: ``em`` treated as a 0/1
    correctness indicator. This is the "stored-composite vs. got-it-right"
    reliability measurement that the thesis reports as the headline
    calibration number. Per-signal calibration (e.g. p_entail alone) is
    left for Table 5.4 / ablation work.
    """
    fieldnames = ["cycle", "n_scored", "ece", "brier", "top_bin_conf", "top_bin_acc"]
    rows: List[Dict[str, Any]] = []
    per_cycle = _collect_per_cycle(cycles_data)

    for cycle, data in per_cycle.items():
        conf_pairs = [(c, e) for c, e in zip(data["u_stored"], data["em"]) if c is not None]
        if not conf_pairs:
            rows.append({
                "cycle": cycle, "n_scored": 0,
                "ece": "NaN", "brier": "NaN",
                "top_bin_conf": "NaN", "top_bin_acc": "NaN",
            })
            continue

        conf = [c for c, _ in conf_pairs]
        labels = [float(e) for _, e in conf_pairs]

        # ECE from reliability bins
        try:
            bins = reliability_bins(conf, labels, n_bins=10)
            total = sum(b[2] for b in bins) or 1
            ece = sum((b[2] / total) * abs(b[0] - b[1]) for b in bins)
            # Top bin: the non-empty bin with the highest mean confidence.
            nonempty = [b for b in bins if b[2] > 0]
            top = max(nonempty, key=lambda b: b[0]) if nonempty else (float("nan"),) * 3
            top_conf, top_acc = top[0], top[1]
        except Exception as exc:
            logger.warning("reporting: reliability_bins failed on cycle %d: %s", cycle, exc)
            ece = float("nan"); top_conf = float("nan"); top_acc = float("nan")

        try:
            brier = brier_score(conf, labels)
        except Exception:
            brier = float("nan")

        rows.append({
            "cycle": cycle,
            "n_scored": len(conf),
            "ece": _round_or_nan(ece),
            "brier": _round_or_nan(brier),
            "top_bin_conf": _round_or_nan(top_conf),
            "top_bin_acc": _round_or_nan(top_acc),
        })

    return fieldnames, rows


# -----------------------------------------------------------------------------
# Table 5.3 -- Hallucination decomposition
# -----------------------------------------------------------------------------

def build_table_halluc(cycles_data) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Chapter 5 Table 5.3: hallucination rate by sub-type.

    Three rates per cycle per benchmark:
      * ``hallucination_rate`` -- wrong & high-u_stored (u >= 0.50).
      * ``confabulation_rate`` -- wrong & high-u_internal (>= 0.70)
                                   (the Farquhar/2024 definition).
      * ``early_exit_rate``    -- fraction of samples where the
                                   Stage-5 confabulation gate fired.
    """
    fieldnames = [
        "cycle", "benchmark", "n",
        "hallucination_rate", "confabulation_rate", "early_exit_rate",
    ]
    rows: List[Dict[str, Any]] = []

    for cycle, bm_map in cycles_data.items():
        for bm, payload in bm_map.items():
            samples = payload.get("samples", [])
            n = len(samples)
            em = [s.get("em", 0.0) for s in samples]
            u_stored = [s.get("u_stored") for s in samples]
            u_internal = [s.get("u_internal") for s in samples]
            early_exit = [bool(s.get("early_exit_triggered", False)) for s in samples]

            try:
                hr = hallucination_rate(em, u_stored, u_threshold=0.50)
            except Exception:
                hr = float("nan")

            try:
                cr = confabulation_rate(
                    em, u_internal,
                    threshold=DEFAULT_U_INTERNAL_CONFIDENT,
                    direction="ge",
                )
            except Exception:
                cr = float("nan")

            early_rate = (sum(early_exit) / n) if n else float("nan")

            rows.append({
                "cycle": cycle,
                "benchmark": bm,
                "n": n,
                "hallucination_rate": _round_or_nan(hr),
                "confabulation_rate": _round_or_nan(cr),
                "early_exit_rate": _round_or_nan(early_rate),
            })

    return fieldnames, rows


# -----------------------------------------------------------------------------
# Table 5.4 -- Grounding
# -----------------------------------------------------------------------------

def build_table_grounding(cycles_data) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Chapter 5 Table 5.4: retrieval-grounding coverage per cycle.

    Means over non-None samples: ``p_ground_max``, ``p_ground_mean``,
    ``p_ground_atomic``, and ``p_contra``. Plus ``unsupported_correct``,
    the fraction of correct answers lacking a grounding passage (the
    "right for the wrong reason" diagnostic).

    Per-cycle (not per-benchmark) because grounding behaviour is a
    property of the verifier + retriever, not the benchmark.
    """
    fieldnames = [
        "cycle", "n_scored",
        "mean_p_ground_max", "mean_p_ground_mean",
        "mean_p_ground_atomic", "mean_p_contra",
        "unsupported_correct_rate",
    ]
    rows: List[Dict[str, Any]] = []
    per_cycle = _collect_per_cycle(cycles_data)

    for cycle, data in per_cycle.items():
        n_scored = sum(1 for x in data["p_ground_max"] if x is not None)

        try:
            usc = unsupported_correct_rate(
                data["em"],
                data["p_ground_max"],
                ground_threshold=DEFAULT_GROUND_THRESHOLD,
            )
        except Exception:
            usc = float("nan")

        rows.append({
            "cycle": cycle,
            "n_scored": n_scored,
            "mean_p_ground_max":    _round_or_nan(_safe_mean(data["p_ground_max"])),
            "mean_p_ground_mean":   _round_or_nan(_safe_mean(data["p_ground_mean"])),
            "mean_p_ground_atomic": _round_or_nan(_safe_mean(data["p_ground_atomic"])),
            "mean_p_contra":        _round_or_nan(_safe_mean(data["p_contra"])),
            "unsupported_correct_rate": _round_or_nan(usc),
        })

    return fieldnames, rows


# -----------------------------------------------------------------------------
# Table 5.5 -- Data Purity (Stage-5 decision breakdown)
# -----------------------------------------------------------------------------

def build_table_purity(cycles_data) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Chapter 5 Table 5.5: Stage-5 decision breakdown + storage quality.

    Counts of the four decisions (STORE / DEFERRED / ABSTAIN / DISCARD)
    and the mean u_stored among items that actually reached memory
    (decision == STORE). This is the empirical test of the "purity
    theorem" -- an episode is only retained when the six-weight
    composite clears the decision tree's gates.
    """
    fieldnames = [
        "cycle", "n_verified",
        "n_store", "n_deferred", "n_abstain", "n_discard",
        "store_rate", "mean_u_stored_store",
    ]
    rows: List[Dict[str, Any]] = []
    per_cycle = _collect_per_cycle(cycles_data)

    # Count against the UnifiedVerifier labels via the shared helper.
    # decision_breakdown() was fixed in Task #122 to use the Session-42
    # canonical schema (STORE / DEFERRED / ABSTAIN / DISCARD); previously
    # it used "DEFER" so reporting.py tallied manually to avoid silent
    # drop-through. Now both agree on the schema.
    for cycle, data in per_cycle.items():
        decisions = [d for d in data["decision"] if d is not None]
        n_v = len(decisions)
        breakdown = decision_breakdown(decisions)
        counts = breakdown["counts"]
        n_s = counts["STORE"]

        # Mean u_stored restricted to STORE-decision items (quality-in-memory).
        u_store = [
            u for u, d in zip(data["u_stored"], data["decision"])
            if d == "STORE" and u is not None
        ]
        mean_u_store = _safe_mean(u_store) if u_store else float("nan")

        rows.append({
            "cycle": cycle,
            "n_verified": n_v,
            "n_store":    n_s,
            "n_deferred": counts["DEFERRED"],
            "n_abstain":  counts["ABSTAIN"],
            "n_discard":  counts["DISCARD"],
            "store_rate": _round_or_nan((n_s / n_v) if n_v else float("nan")),
            "mean_u_stored_store": _round_or_nan(mean_u_store),
        })

    return fieldnames, rows


# -----------------------------------------------------------------------------
# Table 5.6 -- Continual Learning (MMLU retention + BWT / FWT)
# -----------------------------------------------------------------------------

def build_table_continual(cycles_data, mmlu_per_cycle=None) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Chapter 5 Table 5.6: continual-learning metrics across cycles.

    Columns:
      * ``mmlu_pct``           : absolute MMLU accuracy (if supplied).
      * ``mmlu_retention_ratio``: MMLU@cycle / MMLU@cycle_0.
      * ``bwt``                : backward-transfer score (needs >=2 cycles).
      * ``fwt``                : forward-transfer score (needs >=2 cycles).

    BWT and FWT are computed from a per_cycle_em_matrix constructed from
    the ordered benchmarks seen at each cycle. Degenerate runs with a
    single cycle produce BWT = FWT = 0.0 (the helpers' documented default).
    """
    fieldnames = [
        "cycle", "mmlu_pct", "mmlu_retention_ratio", "bwt", "fwt",
    ]
    rows: List[Dict[str, Any]] = []

    # Build per_cycle_em_matrix: each row is [em_bm0, em_bm1, ...] for that cycle.
    cycles_sorted = sorted(cycles_data.keys())
    bm_order = sorted({bm for c in cycles_sorted for bm in cycles_data[c].keys()})
    em_matrix = []
    for c in cycles_sorted:
        row = []
        for bm in bm_order:
            payload = cycles_data[c].get(bm)
            if payload is None:
                row.append(0.0)
            else:
                row.append(float(payload.get("meta", {}).get("em", 0.0)))
        em_matrix.append(row)

    try:
        bwt_val = backward_transfer(em_matrix) if len(em_matrix) >= 2 else 0.0
    except Exception:
        bwt_val = float("nan")
    try:
        fwt_val = forward_transfer(em_matrix) if len(em_matrix) >= 2 else 0.0
    except Exception:
        fwt_val = float("nan")

    base_mmlu = None
    if mmlu_per_cycle and len(mmlu_per_cycle) > 0:
        b = mmlu_per_cycle[0]
        if not _isnan(b) and b > 0:
            base_mmlu = b

    for i, cycle in enumerate(cycles_sorted):
        mmlu = (mmlu_per_cycle[cycle]
                if mmlu_per_cycle and cycle < len(mmlu_per_cycle)
                else float("nan"))
        if base_mmlu and not _isnan(mmlu):
            ratio = mmlu / base_mmlu
        else:
            ratio = float("nan")

        rows.append({
            "cycle": cycle,
            "mmlu_pct": _round_or_nan(mmlu * 100 if not _isnan(mmlu) else mmlu, 2),
            "mmlu_retention_ratio": _round_or_nan(ratio, 4),
            # Report BWT/FWT only on the final row -- they are whole-run
            # scalars, not per-cycle. Middle rows leave them blank so
            # make_tables.py does not repeat the value.
            "bwt": _round_or_nan(bwt_val) if i == len(cycles_sorted) - 1 else "",
            "fwt": _round_or_nan(fwt_val) if i == len(cycles_sorted) - 1 else "",
        })

    return fieldnames, rows


# -----------------------------------------------------------------------------
# Within-CAEM cycle-over-cycle progression (cycle N vs. cycle 0)
# -----------------------------------------------------------------------------

def build_table_cycle_progression(cycles_data) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Within-CAEM cycle-over-cycle progression: bootstrap CI + McNemar per benchmark.

    Not to be confused with chapter_5.tex \\ref{tab:sig-test}, which compares
    CAEM against each external baseline (B1-B8) and is produced by
    scripts/baseline_sig_tests.py. This function answers a different question:
    does CAEM's cycle-N improve over its own cycle-0 on each benchmark?

    For each benchmark and each cycle >= 1, compare to cycle 0:
      * Bootstrap 95% CI on the EM difference (cycle_n - cycle_0).
      * McNemar's test on the paired correctness vectors.

    Both pairings require the same sample set at both cycles. The harness
    draws from a fixed dev split, so the pairing is index-aligned.
    Rows are skipped when the sample lengths disagree (defensive).
    """
    fieldnames = [
        "benchmark", "cycle", "n",
        "em_cycle_0", "em_cycle_n", "em_diff",
        "ci_low", "ci_high", "mcnemar_chi2", "mcnemar_p",
    ]
    rows: List[Dict[str, Any]] = []

    cycles_sorted = sorted(cycles_data.keys())
    if 0 not in cycles_sorted or len(cycles_sorted) < 2:
        return fieldnames, rows  # nothing to test against.

    # Index cycle-0 per-benchmark em vectors once.
    base = cycles_data[0]

    for cycle in cycles_sorted[1:]:
        cur = cycles_data.get(cycle, {})
        for bm, payload in cur.items():
            base_payload = base.get(bm)
            if base_payload is None:
                continue
            base_em = [s.get("em", 0.0) for s in base_payload.get("samples", [])]
            cur_em = [s.get("em", 0.0) for s in payload.get("samples", [])]
            if len(base_em) != len(cur_em) or len(base_em) == 0:
                logger.debug(
                    "reporting: length mismatch (%d vs %d) for %s cycle=%d -- skipping.",
                    len(base_em), len(cur_em), bm, cycle,
                )
                continue

            diffs = [c - b for c, b in zip(cur_em, base_em)]
            try:
                # bootstrap_ci returns (mean, lower, upper); we only need the bounds.
                _mean, ci_low, ci_high = bootstrap_ci(diffs, n_bootstrap=1000, ci=0.95)
            except Exception:
                ci_low = float("nan"); ci_high = float("nan")
            try:
                chi2, p = mcnemar_test(base_em, cur_em)
            except Exception:
                chi2 = float("nan"); p = float("nan")

            rows.append({
                "benchmark": bm,
                "cycle": cycle,
                "n": len(base_em),
                "em_cycle_0": _round_or_nan(sum(base_em) / len(base_em)),
                "em_cycle_n": _round_or_nan(sum(cur_em) / len(cur_em)),
                "em_diff":    _round_or_nan(sum(diffs) / len(diffs)),
                "ci_low":     _round_or_nan(ci_low),
                "ci_high":    _round_or_nan(ci_high),
                "mcnemar_chi2": _round_or_nan(chi2),
                "mcnemar_p":  _round_or_nan(p, 6),
            })

    return fieldnames, rows


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------

# Stable filename map: table_id -> filename. Downstream make_tables.py
# imports this constant so the LaTeX generator stays aligned with the
# data writer even if the table IDs are renamed.
TABLE_FILES = {
    "headline":          "tab_headline.csv",
    "calibration":       "tab_calibration.csv",
    "halluc":            "tab_halluc.csv",
    "grounding":         "tab_grounding.csv",
    "purity":            "tab_purity.csv",
    "continual":         "tab_continual.csv",
    # Within-CAEM cycle-over-cycle progression (cycle N vs. cycle 0). The
    # CAEM-vs-baseline significance table (chapter_5.tex \ref{tab:sig-test})
    # is produced separately by scripts/baseline_sig_tests.py and lands as
    # tab_sig_test.csv alongside these files.
    "cycle_progression": "tab_cycle_progression.csv",
}


def build_ch5_tables(
    output_dir,
    mmlu_per_cycle: Optional[Sequence[float]] = None,
) -> Dict[str, Path]:
    """Run the full reporting pipeline.

    Reads every ``{bm}_cycle{c}.json`` under ``output_dir``, emits:
      * ``per_sample_signals.jsonl``  -- one record per sample, flat.
      * ``tab_{headline,calibration,halluc,grounding,purity,continual,cycle_progression}.csv``

    Parameters
    ----------
    output_dir : str | Path
        The same directory the harness wrote per-sample JSONs into.
    mmlu_per_cycle : sequence of float or None
        Optional MMLU retention per cycle (index 0 = pristine baseline).
        When supplied, Tables 5.1 (CES / RET axis) and 5.6 (retention
        ratio) include the real values. When None, those cells are NaN.

    Returns
    -------
    dict: ``{"<table_id>": Path, "jsonl": Path}`` mapping each artefact
    to its absolute path on disk, so the caller can log the manifest.
    """
    output_dir = Path(output_dir)
    cycles_data = load_cycle_data(output_dir)
    if not cycles_data:
        logger.warning(
            "reporting: no per-cycle JSON files found under %s -- nothing to write.",
            output_dir,
        )
        return {}

    # Flat JSONL of per-sample records.
    jsonl_path = output_dir / "per_sample_signals.jsonl"
    write_per_sample_signals_jsonl(cycles_data, jsonl_path)

    # Seven tables.
    builders = {
        "headline":    lambda: build_table_headline(cycles_data, mmlu_per_cycle),
        "calibration": lambda: build_table_calibration(cycles_data),
        "halluc":      lambda: build_table_halluc(cycles_data),
        "grounding":   lambda: build_table_grounding(cycles_data),
        "purity":      lambda: build_table_purity(cycles_data),
        "continual":   lambda: build_table_continual(cycles_data, mmlu_per_cycle),
        "cycle_progression": lambda: build_table_cycle_progression(cycles_data),
    }

    manifest: Dict[str, Path] = {"jsonl": jsonl_path}
    for table_id, fn in builders.items():
        try:
            fields, rows = fn()
        except Exception as exc:
            logger.warning(
                "reporting: table '%s' builder raised %s -- writing empty table.",
                table_id, exc,
            )
            fields, rows = ["cycle"], []
        path = output_dir / TABLE_FILES[table_id]
        _write_csv(path, fields, rows)
        manifest[table_id] = path

    return manifest


__all__ = [
    "VERIFIER_FIELDS",
    "TABLE_FILES",
    "load_cycle_data",
    "write_per_sample_signals_jsonl",
    "build_table_headline",
    "build_table_calibration",
    "build_table_halluc",
    "build_table_grounding",
    "build_table_purity",
    "build_table_continual",
    "build_table_cycle_progression",
    "build_ch5_tables",
]
