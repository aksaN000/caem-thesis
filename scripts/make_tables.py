"""
scripts/make_tables.py
======================
Phase 4m.5 -- Turn the seven CSV artefacts emitted by eval.reporting into
publication-ready LaTeX tables for Chapter 5.

Input
-----
An output directory (typically the same one run_experiment.py writes into)
that contains::

    tab_headline.csv     -- Table 5.1 + CES summary
    tab_calibration.csv  -- Table 5.2
    tab_halluc.csv       -- Table 5.3
    tab_grounding.csv    -- Table 5.4
    tab_purity.csv       -- Table 5.5
    tab_continual.csv    -- Table 5.6
    tab_sig_test.csv     -- Table 5.7

Output
------
Alongside each input CSV we write the corresponding .tex snippet::

    tab_headline.tex, tab_calibration.tex, ..., tab_sig_test.tex

and a combined ``ch5_tables.tex`` file that ``\\input{}``s all seven
snippets in order. Use ``\\input{tab_headline}`` in the thesis chapter
or ``\\input{ch5_tables}`` to pull them all in at once.

Design notes
------------
* Missing cells (``""``, ``"NaN"``, or literal ``"nan"``) render as ``--``.
* Floats are formatted to four decimals unless the column name hints at a
  percent or ms value (``*_ms``, ``*_pct``) -- those get one decimal.
* The CSV column order drives the LaTeX column order -- eval.reporting
  owns the schema, make_tables.py is purely presentational.
* Requires no external deps beyond stdlib ``csv`` and ``argparse``.

CLI
---
::

    python scripts/make_tables.py <output_dir>
    python scripts/make_tables.py runs/cycle_3/      # concrete example

See also: eval/reporting.py (source of the seven CSVs).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Ensure the repo root is on sys.path so ``from eval.reporting import ...``
# works when this script is invoked from any directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Imported for the canonical CSV filename map. Keeping this import means
# a rename in eval.reporting propagates automatically.
from eval.reporting import TABLE_FILES  # noqa: E402  (module-level import OK)


logger = logging.getLogger("caem.make_tables")


# -----------------------------------------------------------------------------
# Per-table metadata
# -----------------------------------------------------------------------------

# One entry per table id matching keys in ``TABLE_FILES``. Each entry carries
# the LaTeX caption and label used in the generated .tex snippet. Keep this
# in sync with the Ch5 methodology section.
TABLE_META: Dict[str, Dict[str, str]] = {
    "headline": {
        "caption": (
            "Headline performance per (cycle, benchmark) with the cycle-level "
            "CAEM Efficacy Score (CES). Rows where \\texttt{benchmark} is "
            "\\texttt{\\_\\_cycle\\_\\_} are the cross-benchmark summaries."
        ),
        "label": "tab:ch5-headline",
    },
    "calibration": {
        "caption": (
            "Calibration of the composite stored confidence $\\hat{u}_{\\text{stored}}$ "
            "against exact-match correctness. ECE and Brier use ten equal-width "
            "bins; the top-bin columns report the highest-confidence bin."
        ),
        "label": "tab:ch5-calibration",
    },
    "halluc": {
        "caption": (
            "Hallucination decomposition per (cycle, benchmark). "
            "\\emph{Hallucination} = wrong under the composite $\\hat{u}_{\\text{stored}}$ "
            "gate. \\emph{Confabulation} = wrong under the internal-only gate "
            "$u_{\\text{internal}} \\geq 0.70$ (Farquhar et al.\\ 2024). "
            "\\emph{Early-exit rate} counts the Stage-5 confabulation gate firings."
        ),
        "label": "tab:ch5-halluc",
    },
    "grounding": {
        "caption": (
            "Retrieval-grounding per cycle. \\emph{Unsupported-correct rate} is "
            "the fraction of correct answers whose top-ranked passage entailment "
            "falls below $p_{\\text{ground,max}} < 0.30$ -- the \"right for the "
            "wrong reason\" diagnostic."
        ),
        "label": "tab:ch5-grounding",
    },
    "purity": {
        "caption": (
            "Stage-5 decision breakdown (Chapter 4 taxonomy) and mean "
            "$\\hat{u}_{\\text{stored}}$ restricted to accepted episodes. "
            "Empirically tests the purity theorem: memory only grows on items "
            "that clear the decision tree's composite gate."
        ),
        "label": "tab:ch5-purity",
    },
    "continual": {
        "caption": (
            "Continual-learning metrics across cycles. MMLU retention ratio "
            "is MMLU@cycle / MMLU@cycle~0. Backward (BWT) and forward (FWT) "
            "transfer are whole-run scalars; they are only printed in the last row."
        ),
        "label": "tab:ch5-continual",
    },
    "sig_test": {
        "caption": (
            "Paired statistical tests for each (benchmark, cycle) versus cycle~0. "
            "CI is a 1000-bootstrap 95\\% interval on the EM delta; McNemar's test "
            "is evaluated with continuity correction on the paired correctness "
            "vectors."
        ),
        "label": "tab:ch5-sig-test",
    },
}


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------

# Column name suffixes that hint at a specific numeric formatting.
_PCT_SUFFIXES = ("_pct", "_rate", "_frac", "_ratio")
_MS_SUFFIXES = ("_ms",)
_INT_SUFFIXES = ("_n", "cycle", "n", "n_scored", "n_verified", "n_store",
                 "n_deferred", "n_abstain", "n_discard")

# Format for the LaTeX NA cell. ``--`` keeps column widths tidy and is the
# convention adopted in Ch5.
_NA_CELL = "--"


def _is_blank(cell: str) -> bool:
    """True if this CSV cell should render as a LaTeX NA."""
    if cell is None:
        return True
    s = str(cell).strip()
    if not s:
        return True
    if s.lower() in {"nan", "none", "null"}:
        return True
    try:
        return math.isnan(float(s))
    except ValueError:
        return False


def _fmt_cell(col: str, value: str) -> str:
    """Return a LaTeX cell string for ``value`` in column ``col``.

    Numbers are rendered in a stable four-decimal form unless the column
    suggests a percent (three decimals) or a latency in ms (one decimal).
    Integer-looking columns get no decimals.
    """
    if _is_blank(value):
        return _NA_CELL

    s = str(value).strip()

    # Non-numeric cells (e.g. 'fever', '__cycle__') are escaped and returned.
    try:
        x = float(s)
    except ValueError:
        return _latex_escape(s)

    col_l = col.lower()
    if any(col_l.endswith(suf) for suf in _MS_SUFFIXES):
        return f"{x:.1f}"
    if col_l in _INT_SUFFIXES or col_l.endswith("_count"):
        # Some int columns come through as "5.0" after a CSV round-trip.
        return f"{int(round(x))}"
    if any(col_l.endswith(suf) for suf in _PCT_SUFFIXES):
        return f"{x:.3f}"
    # Default scalar format.
    return f"{x:.4f}"


def _latex_escape(text: str) -> str:
    """Escape LaTeX-special characters in a raw CSV cell.

    We only escape the small set that commonly appears in benchmark names
    and decision labels (\\_ underscore chiefly). Captions/labels in
    ``TABLE_META`` are authored by hand and pass through untouched.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\~{}",
        "^": r"\^{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def _column_spec(n: int) -> str:
    """Default column alignment: first column ``l``, remainder right-aligned."""
    if n <= 0:
        return "l"
    return "l" + "r" * (n - 1)


# -----------------------------------------------------------------------------
# CSV -> LaTeX rendering
# -----------------------------------------------------------------------------

def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a tab_*.csv emitted by eval.reporting.

    Returns
    -------
    (fieldnames, rows) where rows is a list of str->str dicts.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def render_table(
    table_id: str,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> str:
    """Return the LaTeX ``table`` environment string for one CSV."""
    meta = TABLE_META.get(table_id, {})
    caption = meta.get("caption", f"Table {table_id}")
    label = meta.get("label", f"tab:{table_id}")

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(rf"  \caption{{{caption}}}")
    lines.append(rf"  \label{{{label}}}")
    lines.append(rf"  \begin{{tabular}}{{{_column_spec(len(fieldnames))}}}")
    lines.append(r"    \toprule")

    # Header row: prettify column names by replacing underscores with spaces.
    header = " & ".join(_latex_escape(c.replace("_", " ")) for c in fieldnames)
    lines.append(f"    {header} \\\\")
    lines.append(r"    \midrule")

    # Body rows.
    if rows:
        for row in rows:
            cells = [_fmt_cell(col, row.get(col, "")) for col in fieldnames]
            lines.append("    " + " & ".join(cells) + r" \\")
    else:
        # Empty-table placeholder so the .tex still compiles.
        placeholder = " & ".join([_NA_CELL] * len(fieldnames))
        lines.append("    " + placeholder + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def write_tex_snippet(tex_path: Path, body: str) -> None:
    """Write ``body`` to ``tex_path``, overwriting silently."""
    tex_path.write_text(body, encoding="utf-8")


def build_combined(manifest: Mapping[str, Path]) -> str:
    """Return a ch5_tables.tex body that \\input{}s each snippet in order."""
    order = list(TABLE_FILES.keys())  # stable order matching reporting.py.
    lines = [
        r"% Auto-generated by scripts/make_tables.py.",
        r"% Include this file in the Ch5 source with \input{ch5_tables}",
        r"% or pull snippets individually via \input{tab_headline} etc.",
        "",
    ]
    for tid in order:
        if tid not in manifest:
            continue
        snippet_name = manifest[tid].stem  # drop .tex
        lines.append(rf"\input{{{snippet_name}}}")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------

def build_all(output_dir: Path) -> Dict[str, Path]:
    """Generate every LaTeX snippet for the seven Ch5 tables.

    Parameters
    ----------
    output_dir : Path
        Directory containing the ``tab_*.csv`` files emitted by
        eval.reporting.build_ch5_tables().

    Returns
    -------
    dict: ``{table_id: Path}`` mapping each emitted .tex to its path.
    The combined ``ch5_tables.tex`` appears under the key ``"combined"``.
    """
    output_dir = Path(output_dir)
    manifest: Dict[str, Path] = {}

    for table_id, csv_filename in TABLE_FILES.items():
        csv_path = output_dir / csv_filename
        tex_path = output_dir / (Path(csv_filename).stem + ".tex")
        if not csv_path.exists():
            logger.warning(
                "make_tables: %s is missing; skipping %s.tex generation.",
                csv_path, tex_path.stem,
            )
            continue
        try:
            fieldnames, rows = read_csv(csv_path)
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning(
                "make_tables: failed to read %s (%s); emitting empty snippet.",
                csv_path, exc,
            )
            fieldnames, rows = ["cycle"], []

        body = render_table(table_id, fieldnames, rows)
        write_tex_snippet(tex_path, body)
        manifest[table_id] = tex_path
        logger.info("make_tables: wrote %s (%d rows)", tex_path, len(rows))

    combined_path = output_dir / "ch5_tables.tex"
    write_tex_snippet(combined_path, build_combined(manifest))
    manifest["combined"] = combined_path
    logger.info("make_tables: wrote combined %s", combined_path)

    return manifest


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate Chapter 5 LaTeX tables from the seven CSV artefacts "
            "written by eval.reporting.build_ch5_tables()."
        ),
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing tab_*.csv inputs (tex snippets land here too).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
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

    manifest = build_all(args.output_dir)
    if not manifest:
        logger.error("no tables generated -- are the tab_*.csv files present?")
        return 1

    # Print manifest so CI / run-scripts can grep for it.
    for k, v in manifest.items():
        print(f"{k}\t{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
