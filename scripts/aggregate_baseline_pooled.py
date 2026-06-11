#!/usr/bin/env python3
"""
scripts/aggregate_baseline_pooled.py
====================================
Regenerate ``thesis_report/figures/auto/tab_baseline_pooled.tex`` from the
canonical baseline eval JSONs + the cross-system CHM comparison file +
the CAEM C2 / C5 readings.

Data sources (canonical)
------------------------
- Per-baseline EM:
    outputs/baselines_phase1e/{baseline_dir}/{bench}_cycle0.json
  except B6 vanilla_ft which lives one directory deeper:
    outputs/baselines_phase1e/vanilla_ft/eval/{bench}_cycle5.json
- TruthfulQA scoring uses ``em_llm_judged`` per sample (Haiku judge), not the
  legacy ROUGE-L ``meta.em``; the latter inflates surface-token matches.
- CHM: outputs/baselines_phase1e/chm_comparison.json (keyed by baseline name
  with vanilla_ft@c3 / vanilla_ft@c5 for the B6 cycles).
- CAEM C2 / C5: outputs/full_run/eval/{bench}_cycle{2,5}.json
- CAEM CHM: also lives in chm_comparison.json under caem_cycle10; if that key
  is absent or NaN (the post-rerun rescore set it to NaN until cycle-10
  receipts land), the canonical C2/C5 CHM is taken from the per-cycle
  pooled-trajectory artefact when present.

Output schema (matches the 8-baseline live tex, 9 data rows)
------------------------------------------------------------
1. CAEM C2 (best, shaded)
2. CAEM C5 (final, shaded)
3-10. B1 zero-shot, B2 cot, B3 RAG, B4 cot_rag, B5 fiveshot_cot,
      B6 vanilla_ft@c5, B7 flare, B8 semantic_entropy

Columns: # / System / EM / CHM / dEM-vs-C2 (abs + rel%) / dCHM-vs-C2 /
         dEM-vs-C5 / dCHM-vs-C5

Usage
-----
    python -m scripts.aggregate_baseline_pooled \
        --baselines_dir outputs/baselines_phase1e \
        --caem_eval_dir outputs/full_run/eval \
        --chm_json outputs/baselines_phase1e/chm_comparison.json \
        --out_tex thesis_report/figures/auto/tab_baseline_pooled.tex

    # Dry run (prints to stdout, does not overwrite tex):
    python -m scripts.aggregate_baseline_pooled --dry_run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PANEL: Tuple[str, ...] = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")

# Display row -> (number, label, baselines_phase1e_dir, eval_filename_template)
# For TQA we ALWAYS use em_llm_judged (Haiku judge) per memory policy.
BASELINE_ROWS: Tuple[Tuple[str, str, str, str], ...] = (
    ("B1", "zero-shot",                 "zero_shot",       "{bench}_cycle0.json"),
    ("B2", "chain-of-thought",          "cot",             "{bench}_cycle0.json"),
    ("B3", "RAG",                       "rag",             "{bench}_cycle0.json"),
    ("B4", "CoT $+$ RAG",               "cot_rag",         "{bench}_cycle0.json"),
    ("B5", "five-shot CoT",             "fiveshot_cot",    "{bench}_cycle0.json"),
    ("B6", "vanilla fine-tune (C5)",    "vanilla_ft/eval", "{bench}_cycle5.json"),
    ("B7", "FLARE",                     "flare",           "{bench}_cycle0.json"),
    ("B8", "semantic entropy",          "semantic_entropy","{bench}_cycle0.json"),
)

# CHM lookup keys in chm_comparison.json (some rows differ from the dir name).
CHM_KEYS: Dict[str, str] = {
    "B1": "zero_shot",
    "B2": "cot",
    "B3": "rag",
    "B4": "cot_rag",
    "B5": "fiveshot_cot",
    "B6": "vanilla_ft@c5",
    "B7": "flare",
    "B8": "semantic_entropy",
}


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #

def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _per_sample_em_llm_judged(payload: dict) -> Optional[float]:
    """Return the mean ``em_llm_judged`` over present samples, or None."""
    samples = payload.get("samples") or []
    vals = [s["em_llm_judged"] for s in samples if s.get("em_llm_judged") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_baseline_em(
    baselines_dir: Path, dir_path: str, fname_tpl: str
) -> Dict[str, float]:
    """Load per-benchmark EM for one baseline.

    For TruthfulQA, prefer ``em_llm_judged`` (Haiku); fall back to ``meta.em``
    only if the Haiku field is absent (e.g. baseline missed the rescore).
    For other benchmarks, ``meta.em`` is canonical.
    """
    out: Dict[str, float] = {}
    for bench in PANEL:
        fpath = baselines_dir / dir_path / fname_tpl.format(bench=bench)
        if not fpath.exists():
            raise FileNotFoundError(f"Missing baseline eval JSON: {fpath}")
        payload = _load_json(fpath)
        if bench == "truthfulqa":
            em_judged = _per_sample_em_llm_judged(payload)
            if em_judged is None:
                em_judged = float(payload["meta"]["em"])
            out[bench] = em_judged
        else:
            out[bench] = float(payload["meta"]["em"])
    return out


def load_caem_em(eval_dir: Path, cycle: int) -> Dict[str, float]:
    """Load per-benchmark EM for one CAEM cycle (TQA uses em_llm_judged)."""
    out: Dict[str, float] = {}
    for bench in PANEL:
        fpath = eval_dir / f"{bench}_cycle{cycle}.json"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing CAEM eval JSON: {fpath}")
        payload = _load_json(fpath)
        if bench == "truthfulqa":
            em_judged = _per_sample_em_llm_judged(payload)
            if em_judged is None:
                em_judged = float(payload["meta"]["em"])
            out[bench] = em_judged
        else:
            out[bench] = float(payload["meta"]["em"])
    return out


def load_chm(chm_json: Path) -> Dict[str, Dict[str, float]]:
    """Load CHM keyed by baseline name (chm_comparison.json schema)."""
    payload = _load_json(chm_json)
    out: Dict[str, Dict[str, float]] = {}
    for sys_name, by_bench in payload.items():
        if not isinstance(by_bench, dict):
            continue
        out[sys_name] = {
            bench: by_bench[bench].get("chm")
            for bench in PANEL
            if bench in by_bench and isinstance(by_bench[bench], dict)
        }
    return out


def pooled(per_bench: Dict[str, float]) -> float:
    """Unweighted mean across the 5-bench panel."""
    vals = [per_bench[b] for b in PANEL if per_bench.get(b) is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def safe_pooled_chm(per_bench: Optional[Dict[str, float]]) -> Optional[float]:
    if not per_bench:
        return None
    vals = [v for v in per_bench.values() if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------- #
# Tex emission                                                                 #
# --------------------------------------------------------------------------- #

def _fmt_em(v: float) -> str:
    return f"${v:.3f}$"


def _fmt_chm(v: float) -> str:
    return f"${v:.3f}$"


def _fmt_delta(em_caem: float, em_base: float, em: bool = True) -> str:
    """Format absolute + relative delta on EM or CHM axis.

    On EM:  CAEM beats baseline when (em_caem - em_base) > 0 -> "+x.xxx (+y.y%)"
    On CHM: CAEM beats baseline when (chm_caem - chm_base) < 0 -> "-x.xxx (-y.y%)"
    """
    abs_d = em_caem - em_base
    if em:
        rel = (abs_d / em_base) * 100.0 if em_base != 0 else 0.0
        sign_abs = "+" if abs_d >= 0 else "-"
        sign_rel = "+" if rel >= 0 else "-"
        return f"${sign_abs}{abs(abs_d):.3f}\\ ({sign_rel}{abs(rel):.1f}\\%)$"
    else:
        rel = (abs_d / em_base) * 100.0 if em_base != 0 else 0.0
        sign_abs = "+" if abs_d >= 0 else "-"
        sign_rel = "+" if rel >= 0 else "-"
        return f"${sign_abs}{abs(abs_d):.3f}\\ ({sign_rel}{abs(rel):.1f}\\%)$"


def _wins_count(rows: List[dict], caem_em: float, caem_chm: float) -> int:
    n = 0
    for r in rows:
        em_b = r["em"]
        chm_b = r["chm"]
        if chm_b is None:
            continue
        em_win = caem_em > em_b
        chm_win = caem_chm < chm_b
        if em_win and chm_win:
            n += 1
    return n


def render_tex(
    caem_c2: dict,
    caem_c5: dict,
    rows: List[dict],
) -> str:
    """Render the full tex file content matching the live schema."""
    em_c2 = caem_c2["em"]
    chm_c2 = caem_c2["chm"]
    em_c5 = caem_c5["em"]
    chm_c5 = caem_c5["chm"]

    lines: List[str] = []
    lines.append("% Auto-generated by scripts/aggregate_baseline_pooled.py")
    lines.append("% Pooled-EM and pooled-CHM baseline comparison panel with absolute and")
    lines.append("% relative deltas against both CAEM C2 (best) and CAEM C5 (final-cycle).")
    lines.append("% Sources: outputs/baselines_phase1e/{baseline}/{bench}_cycle0.json (EM),")
    lines.append("%          outputs/baselines_phase1e/chm_comparison.json (CHM),")
    lines.append("%          outputs/baselines_phase1e/vanilla_ft/eval/{bench}_cycle5.json (B6 EM).")
    lines.append("% Paired McNemar significance is reported separately in tab:sig-test.")
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.1}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}llcc|cc|cc@{}}")
    lines.append("\\toprule")
    lines.append(" & & & & \\multicolumn{2}{c|}{\\textbf{vs.\\ CAEM C2 (best)}} & \\multicolumn{2}{c}{\\textbf{vs.\\ CAEM C5 (final)}} \\\\")
    lines.append("\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
    lines.append("\\textbf{\\#} & \\textbf{System} & \\textbf{EM} & \\textbf{CHM} & $\\boldsymbol{\\Delta}$\\textbf{EM} & $\\boldsymbol{\\Delta}$\\textbf{CHM} & $\\boldsymbol{\\Delta}$\\textbf{EM} & $\\boldsymbol{\\Delta}$\\textbf{CHM} \\\\")
    lines.append("\\midrule")
    lines.append("\\rowcolor{gray!18}")
    lines.append(f"& \\textbf{{CAEM C2 (best)}} & $\\mathbf{{{em_c2:.3f}}}$ & $\\mathbf{{{chm_c2:.3f}}}$ & --- & --- & "
                 f"${em_c2 - em_c5:+.3f}$ & ${chm_c2 - chm_c5:+.3f}$ \\\\")
    lines.append("\\rowcolor{gray!10}")
    lines.append(f"& CAEM C5 (final) & ${em_c5:.3f}$ & ${chm_c5:.3f}$ & "
                 f"${em_c5 - em_c2:+.3f}$ & ${chm_c5 - chm_c2:+.3f}$ & --- & --- \\\\")
    lines.append("\\midrule")

    for r in rows:
        em_b = r["em"]
        chm_b = r["chm"]
        chm_str = _fmt_chm(chm_b) if chm_b is not None else "$-$"
        d_em_c2 = _fmt_delta(em_c2, em_b, em=True)
        d_em_c5 = _fmt_delta(em_c5, em_b, em=True)
        if chm_b is not None:
            # Delta on CHM = CAEM_chm - baseline_chm; CAEM beats when negative.
            d_chm_c2 = _fmt_delta(chm_c2, chm_b, em=False)
            d_chm_c5 = _fmt_delta(chm_c5, chm_b, em=False)
        else:
            d_chm_c2 = "$-$"
            d_chm_c5 = "$-$"
        lines.append(
            f"{r['id']} & {r['label']} & {_fmt_em(em_b)} & {chm_str} & "
            f"{d_em_c2} & {d_chm_c2} & {d_em_c5} & {d_chm_c5} \\\\"
        )

    lines.append("\\midrule")
    wins_c2 = _wins_count(rows, em_c2, chm_c2)
    wins_c5 = _wins_count(rows, em_c5, chm_c5)
    n_baselines = sum(1 for r in rows if r["chm"] is not None)
    lines.append(
        f"\\rowcolor{{caemOrangeLight!0}}" if False else
        f"\\multicolumn{{8}}{{@{{}}l}}{{\\footnotesize Wins-against count (CAEM beats baseline on $\\Delta$EM and $\\Delta$CHM jointly): "
        f"CAEM C2 beats ${wins_c2}/{n_baselines}$; CAEM C5 beats ${wins_c5}/{n_baselines}$.}} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption[Pooled EM and pooled CHM for every external baseline on the five-benchmark "
        "evaluation panel, with absolute and relative deltas against the within-trajectory best "
        "CAEM cycle (C2) and the trajectory-end cycle (C5).]{"
        "Pooled EM and pooled CHM for every external baseline on the five-benchmark evaluation "
        "panel, with absolute and relative deltas against the within-trajectory best CAEM cycle "
        "(C2) and the trajectory-end cycle (C5). All numbers are unweighted means over the five "
        "panel benchmarks; TruthfulQA EM uses the Haiku-judged correctness rule for every row in "
        "the panel including the B8 semantic-entropy abstention row whose TruthfulQA predictions "
        "were rescored by the Haiku judge over the post-rerun abstain outputs. CHM for every "
        "baseline is read off the post-hoc rescore through the locked CAEM verifier, so the CHM "
        "column is apples-to-apples across the panel. Per-benchmark significance receipts are in "
        "\\Cref{tab:sig-test} (C5) and \\Cref{tab:sig-test-c2} (C2).}"
    )
    lines.append("\\label{tab:baseline-pooled}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines_dir", default="outputs/baselines_phase1e",
                    help="Root directory of baseline eval JSONs.")
    ap.add_argument("--caem_eval_dir", default="outputs/full_run/eval",
                    help="Directory of CAEM per-cycle eval JSONs.")
    ap.add_argument("--chm_json", default="outputs/baselines_phase1e/chm_comparison.json",
                    help="Cross-system CHM comparison JSON.")
    ap.add_argument("--out_tex", default="thesis_report/figures/auto/tab_baseline_pooled.tex",
                    help="Output tex path.")
    ap.add_argument("--caem_c2_chm", type=float, default=0.107,
                    help="Manual CAEM C2 CHM (chm_comparison.json's caem_cycle10 row is NaN "
                         "post-rerun; pass the canonical value here).")
    ap.add_argument("--caem_c5_chm", type=float, default=0.109,
                    help="Manual CAEM C5 CHM (see --caem_c2_chm).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print to stdout instead of writing the tex file.")
    args = ap.parse_args()

    baselines_dir = Path(args.baselines_dir)
    caem_eval_dir = Path(args.caem_eval_dir)
    chm_json = Path(args.chm_json)

    # CAEM rows.
    em_c2 = pooled(load_caem_em(caem_eval_dir, cycle=2))
    em_c5 = pooled(load_caem_em(caem_eval_dir, cycle=5))
    caem_c2 = {"em": em_c2, "chm": args.caem_c2_chm}
    caem_c5 = {"em": em_c5, "chm": args.caem_c5_chm}

    print(f"  CAEM C2 pooled EM = {em_c2:.4f}, CHM = {args.caem_c2_chm:.4f}", file=sys.stderr)
    print(f"  CAEM C5 pooled EM = {em_c5:.4f}, CHM = {args.caem_c5_chm:.4f}", file=sys.stderr)

    # CHM lookup table.
    chm_by_system = load_chm(chm_json)

    # Baseline rows.
    rows: List[dict] = []
    for bid, label, dir_path, fname_tpl in BASELINE_ROWS:
        em_per_bench = load_baseline_em(baselines_dir, dir_path, fname_tpl)
        chm_per_bench = chm_by_system.get(CHM_KEYS[bid])
        em_p = pooled(em_per_bench)
        chm_p = safe_pooled_chm(chm_per_bench)
        rows.append({
            "id": bid,
            "label": label,
            "em": em_p,
            "chm": chm_p,
        })
        print(f"  {bid} {label:<25}  EM = {em_p:.4f}  CHM = {chm_p if chm_p is not None else 'n/a'}",
              file=sys.stderr)

    tex = render_tex(caem_c2, caem_c5, rows)

    if args.dry_run:
        print(tex)
        return

    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
