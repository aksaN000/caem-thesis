"""
scripts/aggregate_section_5_2.py
================================
Builds the four Phase 1e tables for chapter 5 section 5.2 (Trajectory of headline metrics):
  - tab_pooled_trajectory.tex      (per-cycle pooled EM + CHM + retention)
  - tab_per_bench_trajectory.tex   (per-bench EM + CHM matrix across cycles)
  - tab_halluc_subtypes.tex        (per-cycle pooled per-subtype CHM rates)
  - tab_train_vs_transfer.tex      (training-panel vs transfer-panel pooled trajectory)

Reads from outputs/full_run/all_cycle_results.json + per-sample eval JSONs.
TruthfulQA EM uses the Haiku-judged em_llm_judged field (memory: feedback_truthfulqa_llm_judge_em).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import composite_hallucination_metric

BENCHES = ['fever', 'triviaqa', 'commonsense_qa', 'strategyqa', 'truthfulqa']
TRAINING_PANEL = ['fever', 'triviaqa', 'commonsense_qa']
TRANSFER_PANEL = ['truthfulqa', 'strategyqa']

BENCH_DISPLAY = {
    'fever': 'FEVER',
    'triviaqa': 'TriviaQA',
    'commonsense_qa': 'CommonsenseQA',
    'strategyqa': 'StrategyQA',
    'truthfulqa': 'TruthfulQA',
}

SUBTYPE_DISPLAY = {
    'confident_confabulation_rate': 'Conf.\\ confab.',
    'factual_fabrication_rate':     'Fact.\\ fabric.',
    'logical_fabrication_rate':     'Logic.\\ fabric.',
    'off_topic_rate':               'Off-topic',
    'defensive_evasion_rate':       'Def.\\ evasion',
    'template_leak_rate':           'Templ.\\ leak',
    'false_refusal_rate':           'False refusal',
    'over_long_rate':               'Over-long',
}

SUBTYPE_ORDER = list(SUBTYPE_DISPLAY.keys())

EVAL_DIR = ROOT / 'outputs/full_run/eval'
RETENTION_CSV = ROOT / 'outputs/full_run/stochastic_equilibrium_trajectory.csv'
ALL_CYCLES_JSON = ROOT / 'outputs/full_run/all_cycle_results.json'
OUT_DIR = ROOT / 'thesis_report/figures/auto'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_eval(bench: str, cycle: int) -> List[Dict]:
    p = EVAL_DIR / f'{bench}_cycle{cycle}.json'
    if not p.is_file():
        return []
    d = json.load(p.open())
    return d['samples'] if isinstance(d, dict) else d


def bench_em(samples: List[Dict], bench: str) -> float:
    if not samples:
        return float('nan')
    if bench == 'truthfulqa':
        # Haiku-judged EM is mandatory per memory entry feedback_truthfulqa_llm_judge_em
        llm = [s['em_llm_judged'] for s in samples if s.get('em_llm_judged') is not None]
        if llm:
            return sum(llm) / len(llm)
    return sum(s.get('em', 0) for s in samples) / len(samples)


def bench_chm(samples: List[Dict]) -> float:
    if not samples:
        return float('nan')
    return composite_hallucination_metric(samples).get('chm', float('nan'))


def bench_subtypes(samples: List[Dict]) -> Dict[str, float]:
    if not samples:
        return {k: float('nan') for k in SUBTYPE_ORDER}
    r = composite_hallucination_metric(samples).get('per_subtype_rates', {})
    return {k: r.get(k, float('nan')) for k in SUBTYPE_ORDER}


def load_retention() -> Dict[int, Dict]:
    """Returns {cycle: dict(committed, worst_probe, worst_ratio, mmlu, tqa_test, csqa_test)}"""
    if not RETENTION_CSV.is_file():
        return {}
    out = {}
    with RETENTION_CSV.open() as f:
        for row in csv.DictReader(f):
            out[int(row['cycle'])] = {
                'committed': int(row['committed']),
                'worst_probe': row['worst_probe'],
                'worst_ratio': float(row['worst_ratio']),
                'mmlu': float(row['mmlu_ratio']),
                'tqa_test': float(row['tqa_test_ratio']),
                'csqa_test': float(row['csqa_test_ratio']),
            }
    return out


PROBE_DISPLAY = {
    'mmlu': 'MMLU',
    'triviaqa_test': 'TQA-test',
    'commonsense_qa_test': 'CSQA-test',
    'tqa_test': 'TQA-test',
    'csqa_test': 'CSQA-test',
}


def fmt(x, dp=3):
    if x is None or x != x:  # NaN check
        return '--'
    return f'{x:.{dp}f}'


# ============================================================================
# Build per-cycle aggregates
# ============================================================================
print('Loading data for cycles 0-5...')
all_cells = {}  # all_cells[bench][cycle] = (em, chm, subtypes)
for bench in BENCHES:
    all_cells[bench] = {}
    for c in range(6):
        samples = load_eval(bench, c)
        em = bench_em(samples, bench)
        chm = bench_chm(samples)
        sub = bench_subtypes(samples)
        all_cells[bench][c] = (em, chm, sub, samples)

retention = load_retention()


def pooled_em_chm(cycle: int, panel: List[str]) -> Tuple[float, float]:
    """Pooled average across panel benches at given cycle."""
    ems = [all_cells[b][cycle][0] for b in panel if not (all_cells[b][cycle][0] != all_cells[b][cycle][0])]
    chms = [all_cells[b][cycle][1] for b in panel if not (all_cells[b][cycle][1] != all_cells[b][cycle][1])]
    em = sum(ems) / len(ems) if ems else float('nan')
    chm = sum(chms) / len(chms) if chms else float('nan')
    return em, chm


def pooled_subtypes(cycle: int) -> Dict[str, float]:
    """Pool all samples across benches at cycle, compute CHM per subtype on the pooled sample set."""
    all_samples = []
    for b in BENCHES:
        all_samples.extend(all_cells[b][cycle][3])
    if not all_samples:
        return {k: float('nan') for k in SUBTYPE_ORDER}
    r = composite_hallucination_metric(all_samples).get('per_subtype_rates', {})
    return {k: r.get(k, float('nan')) for k in SUBTYPE_ORDER}


# ============================================================================
# 1. tab_pooled_trajectory.tex
# ============================================================================
print('Building tab_pooled_trajectory.tex...')
lines = ['% Auto-generated by scripts/aggregate_section_5_2.py']
lines.append('% Per-cycle pooled EM + CHM + full 3-probe retention panel + cycle outcome.')
lines.append('% TruthfulQA EM uses Haiku-judged em_llm_judged. Pooled = unweighted bench-mean.')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\small')
lines.append('\\begin{tabular}{cccccccc}')
lines.append('\\toprule')
lines.append(' & & \\multicolumn{2}{c}{\\textbf{Pooled headline}} & \\multicolumn{3}{c}{\\textbf{Retention probes (ratio vs.\\ pristine)}} & \\\\')
lines.append('\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}')
lines.append('\\textbf{Cycle} & $n$/bench & \\textbf{EM} & \\textbf{CHM} & \\textbf{MMLU} & \\textbf{TQA-test} & \\textbf{CSQA-test} & \\textbf{Outcome} \\\\')
lines.append('\\midrule')

def _wrap_worst(value: str, is_worst: bool) -> str:
    return f'\\underline{{{value}}}' if is_worst else value

for c in range(6):
    em, chm = pooled_em_chm(c, BENCHES)
    if c == 0:
        outcome = 'baseline'
        mmlu_s = tqa_s = csqa_s = '--'
    else:
        ret = retention.get(c, {})
        committed = ret.get('committed')
        outcome = '\\textsc{commit}' if committed == 1 else '\\textsc{abort}'
        worst = ret.get('worst_probe', '')
        mmlu_s = _wrap_worst(fmt(ret.get('mmlu'), 3), worst == 'mmlu')
        tqa_s = _wrap_worst(fmt(ret.get('tqa_test'), 3), worst == 'triviaqa_test')
        csqa_s = _wrap_worst(fmt(ret.get('csqa_test'), 3), worst == 'commonsense_qa_test')
        # Bold-tag the abort probe at cycle 4
        if c == 4 and worst == 'triviaqa_test':
            tqa_s = f'\\textbf{{{tqa_s}}}'
    lines.append(f'C{c} & 300 & {fmt(em, 3)} & {fmt(chm, 3)} & {mmlu_s} & {tqa_s} & {csqa_s} & {outcome} \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Per-cycle pooled headline metrics and the full multi-modal retention panel. '
             'Pooled exact match and composite hallucination metric are unweighted means over the five panel benchmarks; '
             'TruthfulQA exact match uses the Haiku-judged correctness rule. '
             'The retention panel reports the post-fine-tune ratio against the pristine cycle-zero capability on each of the three registered probes: '
             'MMLU (general-knowledge multiple choice), TriviaQA-test (held-out open-text factoid), and CommonsenseQA-test (held-out five-option multiple choice). '
             'The worst probe per cycle is underlined; '
             'the registered floor $\\rho_{\\min} = 0.93$ triggers the asymmetric rollback of \\Cref{sec:retention} whenever any probe drops below it. '
             'Cycle four\'s TriviaQA-test ratio of $0.886$ is the first in-flight rollback in the realised trajectory. '
             'MMLU stays above the floor at every cycle in $[0.975, 1.041]$, so the retention contract is governed in practice by the two open-text and commonsense probes rather than by general-knowledge retention.}')
lines.append('\\label{tab:pooled-trajectory}')
lines.append('\\end{table}')
(OUT_DIR / 'tab_pooled_trajectory.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/tab_pooled_trajectory.tex')


# ============================================================================
# 2. tab_per_bench_trajectory.tex
# ============================================================================
print('Building tab_per_bench_trajectory.tex...')
lines = ['% Auto-generated by scripts/aggregate_section_5_2.py']
lines.append('% Per-cycle, per-benchmark EM (top sub-row) and CHM (bottom sub-row).')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\small')
lines.append('\\begin{tabular}{llcccccc}')
lines.append('\\toprule')
lines.append('\\textbf{Panel} & \\textbf{Benchmark} & \\textbf{C0} & \\textbf{C1} & \\textbf{C2} & \\textbf{C3} & \\textbf{C4} & \\textbf{C5} \\\\')
lines.append('\\midrule')

def render_metric_row(panel_label: str, bench: str, metric_idx: int, fdp: int = 3, metric_name: str = ''):
    cells = [fmt(all_cells[bench][c][metric_idx], fdp) for c in range(6)]
    return f'{panel_label} & {BENCH_DISPLAY[bench]} ({metric_name}) & ' + ' & '.join(cells) + ' \\\\'

# Training panel first, EM then CHM rows for each bench
for bench in TRAINING_PANEL:
    lines.append(render_metric_row('train', bench, 0, 3, 'EM'))
    lines.append(render_metric_row('     ', bench, 1, 3, 'CHM'))
lines.append('\\midrule')
for bench in TRANSFER_PANEL:
    lines.append(render_metric_row('transfer', bench, 0, 3, 'EM'))
    lines.append(render_metric_row('        ', bench, 1, 3, 'CHM'))
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Per-cycle exact match and composite hallucination metric, broken out by benchmark. '
             'Each benchmark contributes two rows: exact match on top, composite hallucination metric below. '
             'The training panel (FEVER, TriviaQA, CommonsenseQA) supplies the calibration fold and the self-improvement training pool; '
             'the transfer panel (TruthfulQA, StrategyQA) is held out as a generalisation-under-distribution-shift measurement. '
             'TruthfulQA exact match uses the Haiku-judged correctness rule. '
             'All cells evaluated on the $n=300$ evaluation fold.}')
lines.append('\\label{tab:per-bench-trajectory}')
lines.append('\\end{table}')
(OUT_DIR / 'tab_per_bench_trajectory.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/tab_per_bench_trajectory.tex')


# ============================================================================
# 3. tab_halluc_subtypes.tex
# ============================================================================
print('Building tab_halluc_subtypes.tex...')
lines = ['% Auto-generated by scripts/aggregate_section_5_2.py']
lines.append('% Per-cycle pooled per-subtype CHM rates. Pool = all 1500 samples (5 benches x 300) per cycle.')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\footnotesize')
lines.append('\\begin{tabular}{lcccccc}')
lines.append('\\toprule')
lines.append('\\textbf{Subtype} & \\textbf{C0} & \\textbf{C1} & \\textbf{C2} & \\textbf{C3} & \\textbf{C4} & \\textbf{C5} \\\\')
lines.append('\\midrule')

# Precompute pooled subtype rates per cycle
pooled_subs = {c: pooled_subtypes(c) for c in range(6)}

for sub in SUBTYPE_ORDER:
    cells = [fmt(pooled_subs[c][sub], 3) for c in range(6)]
    lines.append(f'{SUBTYPE_DISPLAY[sub]} & ' + ' & '.join(cells) + ' \\\\')
lines.append('\\midrule')
chms = []
for c in range(6):
    em, chm = pooled_em_chm(c, BENCHES)
    chms.append(chm)
chm_cells = [fmt(x, 3) for x in chms]
lines.append('\\textbf{CHM (mean of 8)} & ' + ' & '.join(chm_cells) + ' \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Per-cycle decomposition of the composite hallucination metric into its eight measurable failure-mode subtypes, '
             'pooled across the five-benchmark evaluation panel (one thousand five hundred samples per cycle). '
             'The composite hallucination metric on the bottom row is the equal-weighted mean of the eight per-subtype rates; '
             'the ninth axis (factual contradiction) is structurally pinned at zero under the deployed binary natural-language-inference judge '
             'and is excluded from the default denominator. '
             'Per-subtype thresholds are defined in \\Cref{sec:setup-metrics} and registered before the calibration phase.}')
lines.append('\\label{tab:halluc-subtypes}')
lines.append('\\end{table}')
(OUT_DIR / 'tab_halluc_subtypes.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/tab_halluc_subtypes.tex')


# ============================================================================
# 4. tab_train_vs_transfer.tex
# ============================================================================
print('Building tab_train_vs_transfer.tex...')
lines = ['% Auto-generated by scripts/aggregate_section_5_2.py']
lines.append('% Training-panel (FEVER + TriviaQA + CSQA) vs transfer-panel (TruthfulQA + StrategyQA) pooled trajectory.')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\small')
lines.append('\\begin{tabular}{lcccccc}')
lines.append('\\toprule')
lines.append('\\textbf{Panel} & \\textbf{C0} & \\textbf{C1} & \\textbf{C2} & \\textbf{C3} & \\textbf{C4} & \\textbf{C5} \\\\')
lines.append('\\midrule')

# Compute per-cycle pooled per-panel
train_em = []
train_chm = []
transfer_em = []
transfer_chm = []
for c in range(6):
    em_t, chm_t = pooled_em_chm(c, TRAINING_PANEL)
    em_x, chm_x = pooled_em_chm(c, TRANSFER_PANEL)
    train_em.append(em_t)
    train_chm.append(chm_t)
    transfer_em.append(em_x)
    transfer_chm.append(chm_x)

lines.append('\\textbf{Training panel (FEVER + TriviaQA + CSQA)} & & & & & & \\\\')
lines.append('\\quad Pooled EM & ' + ' & '.join(fmt(x, 3) for x in train_em) + ' \\\\')
lines.append('\\quad Pooled CHM & ' + ' & '.join(fmt(x, 3) for x in train_chm) + ' \\\\')
lines.append('\\midrule')
lines.append('\\textbf{Transfer panel (TruthfulQA + StrategyQA)} & & & & & & \\\\')
lines.append('\\quad Pooled EM & ' + ' & '.join(fmt(x, 3) for x in transfer_em) + ' \\\\')
lines.append('\\quad Pooled CHM & ' + ' & '.join(fmt(x, 3) for x in transfer_chm) + ' \\\\')
lines.append('\\midrule')

# Add deltas C0 -> C5 for both panels
d_train_em = train_em[5] - train_em[0]
d_train_chm = (train_chm[5] - train_chm[0]) / train_chm[0] * 100 if train_chm[0] else float('nan')
d_transfer_em = transfer_em[5] - transfer_em[0]
d_transfer_chm = (transfer_chm[5] - transfer_chm[0]) / transfer_chm[0] * 100 if transfer_chm[0] else float('nan')
lines.append(f'\\textbf{{$\\Delta$ EM C0$\\rightarrow$C5}} & training: {d_train_em:+.3f} & \\multicolumn{{4}}{{c}}{{}} & transfer: {d_transfer_em:+.3f} \\\\')
lines.append(f'\\textbf{{$\\Delta$ CHM C0$\\rightarrow$C5}} & training: {d_train_chm:+.1f}\\% & \\multicolumn{{4}}{{c}}{{}} & transfer: {d_transfer_chm:+.1f}\\% \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Training-panel versus transfer-panel pooled trajectory. '
             'The training panel (FEVER, TriviaQA, CommonsenseQA) supplies both the calibration fold and the self-improvement training pool, so any '
             'gain on the training panel could in principle reflect specialisation; the transfer panel (TruthfulQA, StrategyQA) tests generalisation '
             'under distribution shift because samples from the transfer panel never enter the calibration fold, the seed pool, or the per-cycle stream chunks. '
             'A non-collapsing transfer-panel reading is the empirical receipt that the calibrated-probability composite generalises '
             'beyond its fit-time distribution.}')
lines.append('\\label{tab:train-vs-transfer}')
lines.append('\\end{table}')
(OUT_DIR / 'tab_train_vs_transfer.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/tab_train_vs_transfer.tex')

print('\nAll 4 tables built.')
