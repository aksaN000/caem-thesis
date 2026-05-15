#!/usr/bin/env python
"""
scripts/aggregate_stochastic_equilibrium.py
============================================
Log scraper for the stochastic-equilibrium empirical receipt
(cor:stochastic-equilibrium in Ch4 §sec:theorem-corollaries).

Reads the experiment.log file to extract, for each attempted cycle:
  - Commit indicator X_t (0 if SIL aborted, 1 if committed)
  - Worst-probe ratio at the retention-guard check
  - Which probe was the worst (MMLU / TQA-test / CSQA-test)
  - Per-probe ratios at the retention check

Then computes:
  - π_commit = empirical commit fraction across the cycle horizon
  - Worst-probe trajectory across cycles
  - Per-probe distribution-vs-floor visualisation data

Inputs
------
  outputs/full_run/experiment.log (the runner's INFO+WARNING log)

Outputs
-------
  outputs/full_run/stochastic_equilibrium_trajectory.csv
    cycle, committed, worst_probe, worst_ratio,
    mmlu_ratio, tqa_test_ratio, csqa_test_ratio
  thesis_report/figures/auto/tab_stochastic_equilibrium.tex
    LaTeX table for Ch4/Ch5 cor:stochastic-equilibrium evidence.

Usage
-----
    python -m scripts.aggregate_stochastic_equilibrium \\
        --log outputs/full_run/experiment.log \\
        --csv_out outputs/full_run/stochastic_equilibrium_trajectory.csv \\
        --tex_out thesis_report/figures/auto/tab_stochastic_equilibrium.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Regex patterns for the runner log lines we care about
PROBE_LINE = re.compile(
    r"Cycle\s+(\d+):\s+post-training probes\s*=\s*\{([^}]+)\}\s*\|\s*ratios\s*=\s*\{([^}]+)\}"
)
ABORT_LINE = re.compile(r"Cycle\s+(\d+):\s+forgetting check FAILED")
ABORT_CONFIRM = re.compile(r"Cycle\s+(\d+)\s+ABORTED.*Weights restored")
COMMIT_CONFIRM = re.compile(r"Fine-tuning done.*retention_ratio")


def _parse_dict(s: str) -> Dict[str, float]:
    """Parse the truncated dict from log, e.g. 'mmlu': 0.9885, 'triviaqa_test': 0.93..."""
    out: Dict[str, float] = {}
    for match in re.finditer(r"'([a-z_]+)'\s*:\s*([0-9.]+)", s):
        out[match.group(1)] = float(match.group(2))
    return out


def parse_log(log_path: Path) -> List[Dict]:
    """Walk the log, return one record per (cycle, attempt) with probe ratios + commit."""
    if not log_path.exists():
        logger.error("log file not found: %s", log_path)
        return []

    # Each cycle may have multiple SIL attempts (retries) before final commit/abort.
    # We track the LAST probe-ratio reading per cycle that was followed by
    # either commit (no abort line) or an abort.
    by_cycle: Dict[int, List[Dict]] = defaultdict(list)
    last_probe_for_cycle: Dict[int, Dict] = {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_probe = PROBE_LINE.search(line)
            if m_probe:
                cycle = int(m_probe.group(1))
                probes = _parse_dict(m_probe.group(2))
                ratios = _parse_dict(m_probe.group(3))
                last_probe_for_cycle[cycle] = {
                    "probes": probes, "ratios": ratios,
                }
                continue
            m_abort_warning = ABORT_LINE.search(line)
            if m_abort_warning:
                cycle = int(m_abort_warning.group(1))
                rec = last_probe_for_cycle.get(cycle, {})
                ratios = rec.get("ratios", {})
                if ratios:
                    worst_probe = min(ratios.items(), key=lambda kv: kv[1])
                    by_cycle[cycle].append({
                        "outcome": "ABORT",
                        "worst_probe": worst_probe[0],
                        "worst_ratio": worst_probe[1],
                        "ratios": dict(ratios),
                    })
                continue
            m_commit = COMMIT_CONFIRM.search(line)
            if m_commit:
                # commit fires once per successful cycle's fine-tune; find which cycle
                # by looking at most-recent probe reading without a subsequent abort
                for cycle in list(last_probe_for_cycle.keys()):
                    if cycle not in by_cycle or by_cycle[cycle][-1]["outcome"] != "COMMIT":
                        rec = last_probe_for_cycle[cycle]
                        ratios = rec.get("ratios", {})
                        if ratios:
                            worst_probe = min(ratios.items(), key=lambda kv: kv[1])
                            by_cycle[cycle].append({
                                "outcome": "COMMIT",
                                "worst_probe": worst_probe[0],
                                "worst_ratio": worst_probe[1],
                                "ratios": dict(ratios),
                            })
                            # mark this cycle as committed so we don't double-count
                            del last_probe_for_cycle[cycle]
                            break

    # Build the per-cycle final record (last attempt per cycle determines outcome)
    rows: List[Dict] = []
    for cycle in sorted(by_cycle.keys()):
        attempts = by_cycle[cycle]
        final = attempts[-1]
        committed = 1 if final["outcome"] == "COMMIT" else 0
        rows.append({
            "cycle": cycle,
            "committed": committed,
            "n_attempts": len(attempts),
            "worst_probe": final["worst_probe"],
            "worst_ratio": final["worst_ratio"],
            "mmlu_ratio": final["ratios"].get("mmlu", float("nan")),
            "tqa_test_ratio": final["ratios"].get("triviaqa_test", float("nan")),
            "csqa_test_ratio": final["ratios"].get("commonsense_qa_test", float("nan")),
        })
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cycle", "committed", "n_attempts", "worst_probe", "worst_ratio",
              "mmlu_ratio", "tqa_test_ratio", "csqa_test_ratio"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields}
            w.writerow(row)
    logger.info("stochastic-equilibrium CSV -> %s (%d rows)", path, len(rows))


def write_tex(rows: List[Dict], path: Path, pi_commit: float, floor: float = 0.93) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("% Auto-generated by scripts/aggregate_stochastic_equilibrium.py")
    lines.append("% Per-cycle commit indicator + worst-probe trajectory")
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}cccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Cycle} & \textbf{Attempts} & \textbf{Outcome} & \textbf{MMLU} & \textbf{TQA-test} & \textbf{CSQA-test} & \textbf{Worst probe} & \textbf{Worst ratio} \\")
    lines.append(r"\midrule")
    for r in rows:
        outcome = r"\textsc{commit}" if r["committed"] == 1 else r"\textsc{abort}"
        worst_probe_name = r["worst_probe"].replace("_", r"\_")
        marker = "" if r["worst_ratio"] >= floor else r"\textbf{"
        marker_close = "" if r["worst_ratio"] >= floor else r"}"
        lines.append(
            f"C{r['cycle']} & {r['n_attempts']} & {outcome} & "
            f"${r['mmlu_ratio']:.3f}$ & ${r['tqa_test_ratio']:.3f}$ & "
            f"${r['csqa_test_ratio']:.3f}$ & {worst_probe_name} & "
            f"{marker}${r['worst_ratio']:.3f}${marker_close} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{8}}{{l}}{{\textbf{{Realised commit fraction $\pi_{{\mathrm{{commit}}}} = "
        rf"{pi_commit:.3f}$ ({sum(r['committed'] for r in rows)} of {len(rows)} attempted cycles); "
        rf"retention floor $\rho_{{\min}} = {floor:.2f}$}}}} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Per-cycle retention-guard firing pattern. Each row reports the worst-probe ratio at the final SIL attempt of that cycle; ratios below the registered floor $\rho_{\min} = 0.93$ trigger the asymmetric rollback. The realised commit fraction $\pi_{\mathrm{commit}}$ is the empirical receipt for \Cref{cor:stochastic-equilibrium}. A non-degenerate $\pi_{\mathrm{commit}} \in (0, 1)$ with worst-probe ratios bouncing near the floor confirms the stochastic-equilibrium framing; different probes failing on different cycles confirms the multi-modal panel's value over single-probe guards.}")
    lines.append(r"\label{tab:stochastic-equilibrium}")
    lines.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("stochastic-equilibrium TeX -> %s", path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, default=Path("outputs/full_run/experiment.log"))
    p.add_argument("--csv_out", type=Path, default=Path("outputs/full_run/stochastic_equilibrium_trajectory.csv"))
    p.add_argument("--tex_out", type=Path, default=Path("thesis_report/figures/auto/tab_stochastic_equilibrium.tex"))
    p.add_argument("--floor", type=float, default=0.93)
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(level=ns.log_level, format="%(asctime)s %(levelname)s %(message)s")

    rows = parse_log(ns.log)
    if not rows:
        logger.error("no cycle records parsed from %s", ns.log)
        return 1
    pi_commit = sum(r["committed"] for r in rows) / len(rows)
    write_csv(rows, ns.csv_out)
    write_tex(rows, ns.tex_out, pi_commit, ns.floor)

    # stdout summary
    print(f"\nStochastic-equilibrium trajectory (floor = {ns.floor:.2f}):")
    print(f"{'cycle':<7}{'outcome':<10}{'worst_probe':<22}{'worst_ratio':<14}")
    print("-" * 53)
    for r in rows:
        outcome = "COMMIT" if r["committed"] else "ABORT"
        print(f"c{r['cycle']:<6}{outcome:<10}{r['worst_probe']:<22}{r['worst_ratio']:<14.4f}")
    print(f"\nπ_commit = {pi_commit:.4f} ({sum(r['committed'] for r in rows)} of {len(rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
