"""
scripts/compare_prompt_design.py
================================
Emit the Cycle-0 prompt-design ablation table for Chapter~5.

Reads per-benchmark eval JSONs from two directories
(``cycle_0_pre_cot_prompt/eval`` and ``cycle_0/eval``) produced by
two independent Step 7.0 runs. The two runs are identical in every
component except the generator prompt: the old directory used the
legacy per-task minimal-CoT prompts (pre-Option-C), and the new
directory uses the uniform few-shot scaffolded CoT prompt with a
forced decoder prefix (Chapter~4 §Implementation, \textit{Generator
prompt design}). Both runs use the same cold-start memory, the same
calibration-fold quantile thresholds, and the same random seed.

Per benchmark, the script reports:

    * EM, F1 (new minus old delta)
    * Mean u_stored, plus STORE / DEFERRED / ABSTAIN / DISCARD rates
      (new minus old delta)
    * Confident-error rate (fraction of samples with u_stored >= 0.50
      that scored EM == 0)
    * Mean prediction length in words (new minus old)

Output forms
------------
The script emits both a human-readable console summary and a LaTeX
table fragment ready to `\input{}` from Chapter~5.

Invocation
----------
    PYTHONPATH=. python scripts/compare_prompt_design.py \
        --old_dir outputs/cycle_0_pre_cot_prompt/eval \
        --new_dir outputs/cycle_0/eval \
        --output_tex "pre thesis 1 report/tables/tab_prompt_design_ablation.tex"

Benchmarks
----------
Compares the six benchmarks ordered as: FEVER, TriviaQA, Natural
Questions, TruthfulQA, StrategyQA, ARC-Challenge. Benchmarks missing
from either directory are noted in the console summary and excluded
from the table (rather than crashing) -- useful when one run
finished only a subset before we decided to kill and compare.

Design notes
------------
This is an external comparison script rather than an automated
ablation variant in caem/ablation/variants.py because the prompt
shapes the generator's training-data distribution and therefore
cannot be flipped at inference time. The automated registry is for
inference-time mechanism ablations within a single training
trajectory; prompt-design requires two full trajectories. See
CH_AUDIT_TRACKING.md M10 for the architectural trade-off.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare")


BENCHMARKS: List[str] = [
    "fever",
    "triviaqa",
    "natural_questions",
    "truthfulqa",
    "strategyqa",
    "arc_challenge",
]

BENCH_DISPLAY: Dict[str, str] = {
    "fever": "FEVER",
    "triviaqa": "TriviaQA",
    "natural_questions": "Natural Questions",
    "truthfulqa": "TruthfulQA",
    "strategyqa": "StrategyQA",
    "arc_challenge": "ARC-Challenge",
}


def _load_samples(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return blob.get("samples", [])
    except Exception as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None


def _metrics(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    n = max(len(samples), 1)
    em_sum = sum(float(s.get("em", 0.0)) for s in samples)
    f1_sum = sum(float(s.get("f1", 0.0)) for s in samples)
    u_sum = sum(float(s.get("u_stored") or 0.0) for s in samples)
    decision_count: Dict[str, int] = {}
    for s in samples:
        d = s.get("decision") or "NONE"
        decision_count[d] = decision_count.get(d, 0) + 1
    # Confident-error: u_stored >= 0.50 AND em == 0
    confident_errors = sum(
        1
        for s in samples
        if (s.get("u_stored") or 0.0) >= 0.50 and float(s.get("em", 0.0)) == 0.0
    )
    # Mean prediction length in words
    pred_words = sum(
        len((s.get("prediction") or "").split()) for s in samples
    )
    return {
        "em": em_sum / n,
        "f1": f1_sum / n,
        "u_stored_mean": u_sum / n,
        "store_rate": decision_count.get("STORE", 0) / n,
        "deferred_rate": decision_count.get("DEFERRED", 0) / n,
        "abstain_rate": decision_count.get("ABSTAIN", 0) / n,
        "discard_rate": decision_count.get("DISCARD", 0) / n,
        "confident_error_rate": confident_errors / n,
        "mean_pred_words": pred_words / n,
        "n": n,
    }


def _delta_row(
    bench: str,
    old: Dict[str, float],
    new: Dict[str, float],
) -> str:
    """Produce one LaTeX table row with old / new / delta columns."""
    def fmt(x: float, digits: int = 3) -> str:
        return f"{x:.{digits}f}"

    def dfmt(old_v: float, new_v: float, digits: int = 3) -> str:
        d = new_v - old_v
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{digits}f}"

    display = BENCH_DISPLAY.get(bench, bench)
    return (
        f"{display} & "
        f"{fmt(old['em'])} & {fmt(new['em'])} & "
        f"{dfmt(old['em'], new['em'])} & "
        f"{fmt(old['u_stored_mean'])} & {fmt(new['u_stored_mean'])} & "
        f"{dfmt(old['u_stored_mean'], new['u_stored_mean'])} & "
        f"{fmt(old['mean_pred_words'], 1)} & {fmt(new['mean_pred_words'], 1)} "
        f"\\\\\n"
    )


def _latex_table(
    rows: List[str],
) -> str:
    header = (
        "% Auto-generated by scripts/compare_prompt_design.py.\n"
        "% Manual edits will be overwritten on the next regeneration.\n"
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption[Prompt-design ablation at Cycle 0]"
        "{Prompt-design ablation: Cycle-0 metrics under the legacy "
        "per-task minimal-CoT prompts (\\textsc{Old}) versus the "
        "uniform few-shot scaffolded CoT prompt with forced decoder "
        "prefix (\\textsc{New}, the design adopted in the rest of this "
        "thesis; see Chapter~\\ref{ch:methodology} \\S"
        "\\ref{sec:implementation}, \\emph{Generator prompt design}). "
        "Both runs use identical cold-start memory, identical "
        "calibration-fold quantile thresholds, and identical random "
        "seeds. $\\Delta$ columns are \\textsc{New} $-$ \\textsc{Old}. "
        "\\emph{Mean pred.\\ words} is the average number of "
        "whitespace-separated tokens in the model's generated answer "
        "(before scaffold stripping).}\n"
        "\\label{tab:prompt-design-ablation}\n"
        "\\begin{tabular}{l"
        "rrr"  # EM old, new, delta
        "rrr"  # u_stored old, new, delta
        "rr"   # pred words old, new
        "}\n"
        "\\toprule\n"
        " & \\multicolumn{3}{c}{\\textbf{EM}} "
        "& \\multicolumn{3}{c}{\\textbf{Mean $u_{\\text{stored}}$}} "
        "& \\multicolumn{2}{c}{\\textbf{Mean pred.\\ words}} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n"
        "Benchmark & Old & New & $\\Delta$ & Old & New & $\\Delta$ "
        "& Old & New \\\\\n"
        "\\midrule\n"
    )
    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return header + "".join(rows) + footer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old_dir", type=Path, required=True)
    p.add_argument("--new_dir", type=Path, required=True)
    p.add_argument(
        "--output_tex", type=Path, required=True,
        help="Where to write the LaTeX table fragment.",
    )
    ns = p.parse_args()

    rows: List[str] = []
    print(
        f"{'Benchmark':<22s}"
        f"{'Old EM':>10s}{'New EM':>10s}{'DEM':>10s}"
        f"{'Old Ust':>10s}{'New Ust':>10s}{'DUst':>10s}"
        f"{'Old Wds':>10s}{'New Wds':>10s}"
    )
    print("-" * 92)
    for bench in BENCHMARKS:
        old_path = ns.old_dir / f"{bench}_cycle0.json"
        new_path = ns.new_dir / f"{bench}_cycle0.json"
        old = _load_samples(old_path)
        new = _load_samples(new_path)
        if old is None or new is None:
            missing = []
            if old is None:
                missing.append(str(old_path))
            if new is None:
                missing.append(str(new_path))
            logger.warning(
                "Skipping %s: missing %s",
                bench, ", ".join(missing),
            )
            continue
        m_old = _metrics(old)
        m_new = _metrics(new)
        rows.append(_delta_row(bench, m_old, m_new))
        print(
            f"{BENCH_DISPLAY.get(bench, bench):<22s}"
            f"{m_old['em']:>10.3f}{m_new['em']:>10.3f}"
            f"{m_new['em'] - m_old['em']:>+10.3f}"
            f"{m_old['u_stored_mean']:>10.3f}{m_new['u_stored_mean']:>10.3f}"
            f"{m_new['u_stored_mean'] - m_old['u_stored_mean']:>+10.3f}"
            f"{m_old['mean_pred_words']:>10.1f}{m_new['mean_pred_words']:>10.1f}"
        )

    if not rows:
        logger.error("No benchmarks had both old and new JSONs; nothing to emit.")
        return 1

    ns.output_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(ns.output_tex, "w", encoding="utf-8") as f:
        f.write(_latex_table(rows))
    logger.info("Wrote %d-row prompt-design-ablation table -> %s", len(rows), ns.output_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
