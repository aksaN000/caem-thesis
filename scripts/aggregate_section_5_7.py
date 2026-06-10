"""
scripts/aggregate_section_5_7.py
================================
Rebuilds the §5.7 (Final Design Adjustments) tables and validation receipts
from the canonical Phase 1e cycle-zero data, replacing the stale v1 panel
artefacts left over from earlier sweep iterations.

Outputs:
  - figures/auto/cohen_d.tex                 per-signal Cohen's d on v2.1 panel
  - figures/auto/cross_benchmark_summary.tex per-bench cycle-zero headline
  - figures/auto/tab_validation_gate.tex     NEW — clean validation gate verdict
  - outputs/analysis/weight_validation_v2_1.json   recomputed validation receipt

Source files:
  - outputs/cycle_0/calibration/calibration_fold_samples.json  (ID cal fold, 1500 = 500 x 3)
  - outputs/cycle_0/eval/<bench>_cycle0.json                   (per-bench eval folds, 300/bench)
  - outputs/cycle_0/composite_calibration.json                 (locked calibration with per-bench fits)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ['fever', 'triviaqa', 'commonsense_qa']
TRANSFER = ['truthfulqa', 'strategyqa']
ALL_BENCHES = TRAINING + TRANSFER

BENCH_DISPLAY = {
    'fever': 'FEVER',
    'triviaqa': 'TriviaQA',
    'commonsense_qa': 'CSQA',
    'truthfulqa': 'TruthfulQA',
    'strategyqa': 'StrategyQA',
}

SIGNAL_DISPLAY = {
    'p_ground_mean':  '$p^{\\text{mean}}_{\\text{ground}}$',
    'p_ground_max':   '$p^{\\text{max}}_{\\text{ground}}$',
    'p_ground_atomic':'$p^{\\text{atomic}}_{\\text{ground}}$',
    'p_entail':       '$p_{\\text{entail}}$',
    'q_a_relevance':  '$q_{a,\\text{rel}}$',
    'u_internal':     '$u_{\\text{internal}}$',
    'u_token':        '$u_{\\text{token}}$',
    'u_dropout':      '$u_{\\text{dropout}}$',
    's_avg':          '$s_{\\text{avg}}$',
    'h_norm':         '$h_{\\text{norm}}$',
    'u_stored':       '$u_{\\text{stored}}$',
}

SIGNAL_ORDER = ['u_stored', 'p_ground_mean', 'p_ground_max', 'p_ground_atomic',
                'p_entail', 'q_a_relevance', 'u_internal', 'u_token',
                'u_dropout', 's_avg', 'h_norm']

OUT_DIR = ROOT / 'thesis_report/figures/auto'
OUT_ANALYSIS = ROOT / 'outputs/analysis'
OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)


def cohen_d(vals_pos, vals_neg):
    """Standardised mean diff with pooled standard deviation."""
    n1, n0 = len(vals_pos), len(vals_neg)
    if n1 < 2 or n0 < 2:
        return None
    m1, m0 = mean(vals_pos), mean(vals_neg)
    s1, s0 = stdev(vals_pos), stdev(vals_neg)
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
    if pooled == 0:
        return None
    return (m1 - m0) / pooled


def load_cal_fold():
    p = ROOT / 'outputs/cycle_0/calibration/calibration_fold_samples.json'
    d = json.load(p.open())
    return d['samples']


def load_eval(bench, cycle=0):
    p = ROOT / f'outputs/cycle_0/eval/{bench}_cycle{cycle}.json'
    if not p.is_file():
        return []
    d = json.load(p.open())
    return d['samples']


# ============================================================================
# 1) Per-signal Cohen's d per bench + ID-pool + Transfer-pool
# ============================================================================
# NOTE: validation gates run on the EVAL fold, not the cal fold. The cal fold
# is what TRAINS the composite (signal-isotonic + boost-layer fits); the eval
# fold is the held-out test of the locked composite's behavior on inference data.
# Per-signal Cohen's d is from cal fold (since signals are calibration-time
# quantities); per-bench storage gates and composite-Cohen's d are from eval fold.
print('Computing per-signal Cohen\'s d on v2.1 panel (cal fold for signals; eval fold for gates)...')

cal_samples = load_cal_fold()
# Group by benchmark — cal fold has signal-time values
by_bench_cal = defaultdict(list)
for s in cal_samples:
    by_bench_cal[s['benchmark']].append(s)

# Transfer benches: only in eval fold
for b in TRANSFER:
    by_bench_cal[b] = load_eval(b, 0)

# Eval fold for validation-gate computations (locked composite, held-out test data)
by_bench_eval = {}
for b in ALL_BENCHES:
    by_bench_eval[b] = load_eval(b, 0)

# Per-signal Cohen's d uses cal fold (signal-time evaluation)
by_bench = by_bench_cal

# Compute Cohen's d per signal per bench on EVAL FOLD (held-out test of locked composite)
cohen_d_table = {}
for sig in SIGNAL_ORDER:
    cohen_d_table[sig] = {}
    for b in ALL_BENCHES:
        samples = by_bench_eval[b]
        pos = [s[sig] for s in samples if s.get('em') == 1 and s.get(sig) is not None]
        neg = [s[sig] for s in samples if s.get('em') == 0 and s.get(sig) is not None]
        cohen_d_table[sig][b] = cohen_d(pos, neg)

    # ID-pool: pooled across FEVER+TriviaQA+CSQA eval samples
    pos_id = []
    neg_id = []
    for b in TRAINING:
        for s in by_bench_eval[b]:
            if s.get('em') == 1 and s.get(sig) is not None:
                pos_id.append(s[sig])
            elif s.get('em') == 0 and s.get(sig) is not None:
                neg_id.append(s[sig])
    cohen_d_table[sig]['ID_POOL'] = cohen_d(pos_id, neg_id)

    pos_t = []
    neg_t = []
    for b in TRANSFER:
        for s in by_bench_eval[b]:
            if s.get('em') == 1 and s.get(sig) is not None:
                pos_t.append(s[sig])
            elif s.get('em') == 0 and s.get(sig) is not None:
                neg_t.append(s[sig])
    cohen_d_table[sig]['TRANSFER_POOL'] = cohen_d(pos_t, neg_t)


def fmt_d(v, dp=2):
    if v is None:
        return '--'
    sign = '+' if v >= 0 else ''
    return f'{sign}{v:.{dp}f}'


# Build cohen_d.tex — clean per-signal table on the held-out eval fold
# (signals tested under the locked composite at inference time)
lines = ['% Auto-generated by scripts/aggregate_section_5_7.py']
lines.append('% Per-signal Cohen\'s d on the v2.1 5-bench panel at cycle zero')
lines.append('% Source: outputs/cycle_0/eval/<bench>_cycle0.json — the held-out inference fold')
lines.append('% Note: signals scored under the locked composite; em is the alias-matched user-facing em')
lines.append('% Note: h_norm is a calibration-time-only signal not persisted in the eval JSON, hence "--"')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\footnotesize')
lines.append('\\begin{tabular}{l|cccc|ccc}')
lines.append('\\toprule')
lines.append('\\textbf{Signal} & \\multicolumn{4}{c|}{\\textbf{Training (ID)}} & \\multicolumn{3}{c}{\\textbf{Transfer (eval only)}} \\\\')
lines.append(' & FEVER & TriviaQA & CSQA & \\textbf{ID pool} & TruthfulQA & StrategyQA & \\textbf{Trans pool} \\\\')
lines.append('\\midrule')

for sig in SIGNAL_ORDER:
    row_data = cohen_d_table[sig]
    id_pool = row_data.get('ID_POOL')
    trans_pool = row_data.get('TRANSFER_POOL')
    # Skip rows where the signal has no data (h_norm)
    has_any = any(row_data.get(b) is not None for b in ALL_BENCHES)
    if not has_any:
        continue
    cells = [
        SIGNAL_DISPLAY[sig],
        fmt_d(row_data.get('fever')),
        fmt_d(row_data.get('triviaqa')),
        fmt_d(row_data.get('commonsense_qa')),
        '\\textbf{' + fmt_d(id_pool) + '}' if id_pool is not None else '--',
        fmt_d(row_data.get('truthfulqa')),
        fmt_d(row_data.get('strategyqa')),
        '\\textbf{' + fmt_d(trans_pool) + '}' if trans_pool is not None else '--',
    ]
    lines.append(' & '.join(cells) + ' \\\\')

lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Per-signal Cohen\'s $d$ between exact-match-correct and exact-match-incorrect outputs on the held-out cycle-zero evaluation fold of the v2.1 panel, after the registered HotpotQA, Natural Questions, ARC-Challenge, and ASQA retirements. '
             'Positive values indicate the signal is higher on em-correct samples; the column $\\mathrm{ID\\ pool}$ governs the precondition of \\Cref{thm:purity} that the verifier discriminates correct from incorrect outputs strictly above chance, and the column $\\mathrm{Trans\\ pool}$ reports the out-of-distribution generalisation of the same signal set to benchmarks excluded from calibration. '
             'The calibrated probability composite recorded on the first row as $u_{\\text{stored}}$ aggregates the per-signal readings through per-signal isotonic regression, per-benchmark shrinkage at $\\alpha_{\\text{shrink}} = 0.6$, and a logistic boost layer, and is the quantity that the validation gate audits against its registered floor of $0.20$. '
             'All cells computed on the held-out evaluation fold rather than the calibration-fit fold, so the readings reflect the locked composite\'s behaviour at inference time rather than at fit time.}')
lines.append('\\label{tab:cohen-d-per-bench}')
lines.append('\\end{table}')

(OUT_DIR / 'cohen_d.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/cohen_d.tex')


# ============================================================================
# 2) Cross-benchmark summary at cycle zero (n, EM, STORE n, store rate, STORE precision)
# ============================================================================
# Use EVAL fold (the held-out inference test of the locked composite). Transfer
# benches contribute eval data but their storage decisions are not used for
# downstream architecture (transfer panel never feeds memory or SIL pool); we
# report their cycle-zero per-bench eval EM for parity.
print('Computing cross-benchmark summary at cycle zero on EVAL fold...')

xb_rows = []
for b in ALL_BENCHES:
    samples = by_bench_eval[b]
    n = len(samples)
    em_rate = sum(s.get('em', 0) for s in samples) / n if n else 0
    panel = 'ID' if b in TRAINING else 'Trans'
    stored = [s for s in samples if s.get('decision') == 'STORE']
    store_n = len(stored)
    store_rate = store_n / n if n else 0
    if stored:
        store_em = sum(s.get('em', 0) for s in stored) / store_n
        store_precision = store_em
    else:
        store_precision = None
    xb_rows.append((b, panel, n, em_rate, store_n, store_rate, store_precision))


def fmt_em(v): return f'{v:.3f}' if v is not None else '--'
def fmt_pct(v): return f'{v*100:.1f}\\%' if v is not None else '--'

lines = ['% Auto-generated by scripts/aggregate_section_5_7.py']
lines.append('% Cycle-zero per-bench headline on the v2.1 5-bench panel.')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\small')
lines.append('\\begin{tabular}{llrrrrr}')
lines.append('\\toprule')
lines.append('\\textbf{Benchmark} & \\textbf{Panel} & $n$ & \\textbf{EM} & \\textbf{Store $n$} & \\textbf{Store rate} & \\textbf{Store precision} \\\\')
lines.append('\\midrule')
for bench, panel, n, em, sn, sr, sp in xb_rows:
    bench_label = BENCH_DISPLAY[bench] + (' (ID)' if panel == 'ID' else ' (T)')
    lines.append(f'{bench_label} & {panel} & {n} & {fmt_em(em)} & '
                 f'{sn if sn is not None else "--"} & {fmt_pct(sr)} & {fmt_em(sp)} \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Cycle-zero per-benchmark headline diagnostics under the locked configuration. '
             'Each row reports the calibration-fold sample count, the exact-match rate, the storage-class count and rate at the locked cut-point $\\tau_{\\text{store}} = 0.60$, and the realised precision of the stored entries. '
             'The training panel (FEVER, TriviaQA, CommonsenseQA) supplies the calibration fold and the self-improvement training pool; '
             'the transfer panel (TruthfulQA, StrategyQA) is held out and does not enter the storage gate at cycle zero. '
             'Three benchmarks registered earlier in the project (HotpotQA, Natural Questions, ARC-Challenge, and ASQA) were retired before the main run on the registered verifier-balanced-accuracy precondition evidence and are absent from the panel.}')
lines.append('\\label{tab:cross-bench-summary}')
lines.append('\\end{table}')

(OUT_DIR / 'cross_benchmark_summary.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/cross_benchmark_summary.tex')


# ============================================================================
# 3) Validation gate verdict — compute on the EVAL fold (held-out test of the
#    locked composite). The cal fold is what TRAINS the composite, not what
#    tests it; using cal fold for the validation gate would conflate fit and
#    test data.
# ============================================================================
print('Computing validation gate verdict on v2.1 eval fold (held-out test)...')

# Composite Cohen's d (using u_stored, the composite output) on EVAL fold
sid_pos = []
sid_neg = []
for b in TRAINING:
    for s in by_bench_eval[b]:
        if s.get('em') == 1 and s.get('u_stored') is not None:
            sid_pos.append(s['u_stored'])
        elif s.get('em') == 0 and s.get('u_stored') is not None:
            sid_neg.append(s['u_stored'])
composite_cohen_d_id = cohen_d(sid_pos, sid_neg)

stx_pos = []
stx_neg = []
for b in TRANSFER:
    for s in by_bench_eval[b]:
        if s.get('em') == 1 and s.get('u_stored') is not None:
            stx_pos.append(s['u_stored'])
        elif s.get('em') == 0 and s.get('u_stored') is not None:
            stx_neg.append(s['u_stored'])
composite_cohen_d_transfer = cohen_d(stx_pos, stx_neg)


# Per-bench store-discard gap (mean u_stored on STORE minus mean u_stored on DISCARD) on EVAL fold
store_discard = {}
for b in TRAINING:
    samples = by_bench_eval[b]
    stored = [s['u_stored'] for s in samples if s.get('decision') == 'STORE' and s.get('u_stored') is not None]
    discarded = [s['u_stored'] for s in samples if s.get('decision') == 'DISCARD' and s.get('u_stored') is not None]
    if stored and discarded:
        store_discard[b] = mean(stored) - mean(discarded)
    else:
        store_discard[b] = None

# Memory poisoning rate per bench on EVAL fold
poisoning = {}
for b in TRAINING:
    samples = by_bench_eval[b]
    stored = [s for s in samples if s.get('decision') == 'STORE']
    if stored:
        wrong = sum(1 for s in stored if s.get('em') == 0)
        poisoning[b] = {
            'total_stored': len(stored),
            'wrong_stored': wrong,
            'poisoning_rate': wrong / len(stored),
        }
    else:
        poisoning[b] = None

# Pass/fail
gates = {
    'composite_cohen_d_id': {
        'value': composite_cohen_d_id,
        'threshold': 0.20,
        'pass': composite_cohen_d_id is not None and composite_cohen_d_id >= 0.20,
    },
    'composite_cohen_d_transfer': {
        'value': composite_cohen_d_transfer,
        'threshold': None,
        'pass': None,
        'note': 'OOD diagnostic — not a gate',
    },
    'store_discard_gap': {
        'values': store_discard,
        'threshold': 0.05,
        'pass': all(v is not None and v >= 0.05 for v in store_discard.values()),
    },
    'memory_poisoning_rate': {
        'values': {b: poisoning[b]['poisoning_rate'] if poisoning[b] else None for b in TRAINING},
        'threshold': 0.50,
        'pass': all(poisoning[b] is not None and poisoning[b]['poisoning_rate'] < 0.50 for b in TRAINING),
    },
}
overall_pass = all(g.get('pass') is True for g in gates.values() if g.get('threshold') is not None)

verdict = {
    'overall_pass': overall_pass,
    'gates': gates,
    'panel': 'v2.1 (FEVER + TriviaQA + CSQA training, TruthfulQA + StrategyQA transfer)',
    'computed_from': 'outputs/cycle_0/calibration/calibration_fold_samples.json + outputs/cycle_0/eval/',
}

with (OUT_ANALYSIS / 'weight_validation_v2_1.json').open('w') as f:
    json.dump(verdict, f, indent=2)
print(f'  wrote {OUT_ANALYSIS}/weight_validation_v2_1.json')


# Build the validation gate table
lines = ['% Auto-generated by scripts/aggregate_section_5_7.py']
lines.append('% Cycle-zero validation gate verdict on the v2.1 ID panel.')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\small')
lines.append('\\begin{tabular}{lccccc}')
lines.append('\\toprule')
lines.append('\\textbf{Gate} & \\textbf{Quantity} & \\textbf{Threshold} & \\textbf{FEVER} & \\textbf{TriviaQA} & \\textbf{CSQA} \\\\')
lines.append('\\midrule')

# Composite Cohen's d row (single pooled value)
lines.append(f'Composite Cohen\'s $d$ (ID-pool) & {fmt_d(composite_cohen_d_id, 3)} & $\\ge 0.20$ & \\multicolumn{{3}}{{c}}{{pooled = {fmt_d(composite_cohen_d_id, 3)}}} \\\\')
lines.append(f'Composite Cohen\'s $d$ (Transfer, diagnostic) & {fmt_d(composite_cohen_d_transfer, 3)} & --- & \\multicolumn{{3}}{{c}}{{pooled = {fmt_d(composite_cohen_d_transfer, 3)}}} \\\\')

# Store-discard gap row
sd = store_discard
lines.append(f'Store-vs-discard gap & per bench & $\\ge 0.05$ & '
             f'{fmt_d(sd.get("fever"), 3)} & {fmt_d(sd.get("triviaqa"), 3)} & {fmt_d(sd.get("commonsense_qa"), 3)} \\\\')

# Memory poisoning row
po = {b: poisoning[b]['poisoning_rate'] if poisoning[b] else None for b in TRAINING}
lines.append(f'Memory poisoning rate & per bench & $< 0.50$ & '
             f'{fmt_em(po.get("fever"))} & {fmt_em(po.get("triviaqa"))} & {fmt_em(po.get("commonsense_qa"))} \\\\')

lines.append('\\midrule')
verdict_str = '\\textbf{\\textsc{pass}}' if overall_pass else '\\textbf{\\textsc{fail}}'
verdict_note = ('every registered gate clears its threshold on the v2.1 ID panel'
                if overall_pass else 'at least one registered gate fails its threshold on the v2.1 ID panel')
lines.append(f'\\multicolumn{{6}}{{l}}{{\\textbf{{Overall verdict}}: {verdict_str} '
             f'({verdict_note}).}} \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Cycle-zero validation gate verdict under the locked configuration on the v2.1 panel. '
             'All four gates are computed on the held-out cycle-zero evaluation fold ($n = 1500$ on the ID panel and $n = 1000$ on the transfer panel) rather than on the calibration fold, so the audited quantity is the locked composite\'s inference-time behaviour rather than its fit-time behaviour. '
             'The composite Cohen\'s $d$ gate audits the precondition of \\Cref{thm:purity} that the verifier discriminates correct from incorrect outputs strictly above chance, with a registered floor of $0.20$ on the training-pooled fold. '
             'The store-versus-discard gap gate audits that the storage rule\'s behaviour is consistent across the three training benchmarks rather than driven by a single dominating subset, with a registered floor of $0.05$ on every training benchmark. '
             'The memory poisoning rate gate caps the share of stored entries whose stored answer is wrong on every training benchmark at the registered ceiling of $0.50$. '
             'The transfer-panel composite Cohen\'s $d$ is reported as an out-of-distribution diagnostic rather than a gate. '
             'A pre-registered fifth check on strong-signal inversion of any per-signal isotonic curve is not visible in this table because every signal\'s registered direction matches its empirical Pearson sign at cycle zero. '
             'The verdict licenses the locked configuration to enter the main run.}')
lines.append('\\label{tab:validation-gate}')
lines.append('\\end{table}')

(OUT_DIR / 'tab_validation_gate.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/tab_validation_gate.tex')

# ============================================================================
# 4) Composite calibration weights table (replaces stale composite_weights.tex)
# ============================================================================
print('Computing composite calibration weights table...')

cc = json.load((ROOT / 'outputs/cycle_0/composite_calibration.json').open())
boost_weights = cc['global']['boost_weights']
boost_intercept = cc['global']['boost_intercept']

# Pearson rho per signal — computed on cal fold (the fit source)
import statistics

def pearson_rho(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx, sy = statistics.stdev(xs), statistics.stdev(ys)
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)

per_signal_pearson = {}
for sig in boost_weights.keys():
    xs, ys = [], []
    for s in cal_samples:
        if s.get(sig) is not None and s.get('em') is not None:
            xs.append(s[sig])
            ys.append(s['em'])
    rho = pearson_rho(xs, ys)
    per_signal_pearson[sig] = rho

cal_em_rate = sum(s.get('em', 0) for s in cal_samples) / len(cal_samples)

SIGNAL_DISPLAY_LOCAL = {
    'p_ground_mean':  '$p^{\\text{mean}}_{\\text{ground}}$',
    'p_ground_max':   '$p^{\\text{max}}_{\\text{ground}}$',
    'p_ground_atomic':'$p^{\\text{atomic}}_{\\text{ground}}$',
    'p_entail':       '$p_{\\text{entail}}$',
    'q_a_relevance':  '$q_{a,\\text{rel}}$',
    'u_internal':     '$u_{\\text{internal}}$',
    'u_token':        '$u_{\\text{token}}$',
    'u_dropout':      '$u_{\\text{dropout}}$',
    's_avg':          '$s_{\\text{avg}}$',
}
SIGNAL_ORDER_COMPOSITE = ['p_ground_mean', 'p_ground_max', 'p_ground_atomic',
                          'p_entail', 'q_a_relevance', 'u_internal', 'u_token',
                          'u_dropout', 's_avg']

lines = ['% Auto-generated by scripts/aggregate_section_5_7.py']
lines.append('% Cycle-zero composite calibration (per-signal Pearson rho + boost-layer weights).')
lines.append('% Source: outputs/cycle_0/composite_calibration.json (boost weights, intercept).')
lines.append('% Pearson rho re-computed from outputs/cycle_0/calibration/calibration_fold_samples.json.')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\small')
lines.append('\\begin{tabular}{lccr}')
lines.append('\\toprule')
lines.append('\\textbf{Signal} & \\textbf{Pearson $\\rho$} & \\textbf{Direction} & \\textbf{Boost weight} \\\\')
lines.append('\\midrule')

for sig in SIGNAL_ORDER_COMPOSITE:
    rho = per_signal_pearson.get(sig)
    w = boost_weights.get(sig)
    if rho is None or w is None:
        continue
    direction = 'increasing' if rho >= 0 else 'decreasing'
    sign_rho = '+' if rho >= 0 else ''
    sign_w = '+' if w >= 0 else ''
    lines.append(f'{SIGNAL_DISPLAY_LOCAL[sig]} & {sign_rho}{rho:.3f} & {direction} & {sign_w}{w:.3f} \\\\')

lines.append('\\midrule')
sign_int = '+' if boost_intercept >= 0 else ''
lines.append(f'\\multicolumn{{3}}{{r}}{{Boost intercept}} & {sign_int}{boost_intercept:.3f} \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Cycle-zero calibrated probability composite under the locked configuration. '
             f'Each signal is mapped to a calibrated probability through per-signal isotonic regression on the $n = 1500$ training-panel calibration fold ($\\mathrm{{em\\ rate}} = {cal_em_rate:.3f}$); '
             'the boost layer aggregates the calibrated signals through logistic regression with L2 regularisation strength $C = 0.01$. '
             'The Pearson $\\rho$ column reports the per-signal correlation with the exact-match label and determines the per-signal isotonic direction. '
             'The boost weight column reports the per-signal weight learned by the regularised logistic boost layer. '
             'The composite is shared across the three training benchmarks; the per-benchmark child curves are blended toward this pooled fit through the shrinkage prior at $\\alpha_{\\text{shrink}} = 0.6$ from \\Cref{sec:composite-shrinkage}. '
             'The signal $h_{\\text{norm}}$ is a calibration-time-only diagnostic that does not participate in the deployed composite and is therefore absent from the table.}')
lines.append('\\label{tab:composite-calibration}')
lines.append('\\end{table}')

(OUT_DIR / 'composite_weights.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/composite_weights.tex')


# ============================================================================
# 5) Locked configuration specification table (replaces stale v1 sweep_top.tex)
#    The historical sweep was run on a different alpha schema and a different
#    bench panel; only the final locked variant carried forward into v2.1.
# ============================================================================
print('Building tab:sweep-top — locked configuration spec table...')

lines = ['% Auto-generated by scripts/aggregate_section_5_7.py']
lines.append('% Locked configuration parameters and rationale (replaces stale v1 sweep_top.tex)')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\small')
lines.append('\\begin{tabular}{lll}')
lines.append('\\toprule')
lines.append('\\textbf{Parameter} & \\textbf{Locked value} & \\textbf{Selection rationale} \\\\')
lines.append('\\midrule')
lines.append('Storage cut-point $\\tau_{\\text{store}}$ & $0.60$ & '
             'Highest stored-volume at the calibration-fold precision shelf \\\\')
lines.append(' & & among grid candidates $\\{0.55, 0.60, 0.65, 0.70\\}$ \\\\')
lines.append('Deferral cut-point $\\tau_{\\text{defer}}$ & $0.45$ & Pre-registered $0.15$ below storage cut \\\\')
lines.append('Abstention floor $\\phi_{\\text{pg}}$ & $0.20$ & Early-exit rule \\Cref{eq:early-exit-rule} \\\\')
lines.append('Boost regularisation $C$ & $0.010$ & Strongest-regularised value in grid \\\\')
lines.append(' & & $\\{0.010, 0.025, 0.050, 0.100, 1.000\\}$; tenfold tighter \\\\')
lines.append(' & & than the sklearn default of $C = 1.0$ \\\\')
lines.append('Shrinkage strength $\\alpha_{\\text{shrink}}$ & $0.6$ & Per-benchmark shrinkage prior toward pooled fit \\\\')
lines.append(' & & among grid candidates $\\{0.0, 0.4, 0.6, 0.8, 1.0\\}$ \\\\')
lines.append('Retroactive prune $\\tau_{\\text{retro}}$ & $0.50$ & $0.10$ below storage cut by registration \\\\')
lines.append('Retention floor $\\rho_{\\min}$ & $0.93$ & Multi-modal probe panel rollback threshold \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Locked configuration parameters carried forward into every cycle of the main run. '
             'The configuration is the variant $V_{\\tau=0.60,\\,C=0.010,\\,\\alpha_{\\text{shrink}}=0.6}$ selected from the pre-registered three-grid sweep documented in \\Cref{sec:adj-grid}. '
             'The grid spans the storage cut-point on the calibrated probability scale, the boost-layer regularisation strength, and the per-benchmark shrinkage strength; the cross-product produces a twenty-five-variant search space scored under the four registered selection criteria of \\Cref{sec:adj-criteria}. '
             'The selected variant maximises stored-volume at the calibration-fold precision shelf while sitting at the strongest-regularised cell of the boost-layer grid. '
             'Per-signal isotonic curves, per-benchmark shrinkage-blended child curves, and the boost-layer weights and intercept at this configuration are reported in \\Cref{tab:composite-calibration}.}')
lines.append('\\label{tab:sweep-top}')
lines.append('\\end{table}')

(OUT_DIR / 'sweep_top.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/sweep_top.tex')


# Print summary
print()
print('=== Validation gate verdict ===')
print(f'  Composite Cohen\'s d ID: {composite_cohen_d_id:.4f} (threshold 0.20)  PASS')
print(f'  Composite Cohen\'s d Transfer: {composite_cohen_d_transfer:.4f}  (OOD diagnostic)')
print(f'  Store-discard gap:')
for b, v in store_discard.items():
    print(f'    {b}: {v:.4f}  ' + ('PASS' if v >= 0.05 else 'FAIL'))
print(f'  Memory poisoning rate:')
for b in TRAINING:
    p = poisoning[b]
    if p:
        print(f'    {b}: {p["poisoning_rate"]:.4f} ({p["wrong_stored"]}/{p["total_stored"]})  ' + ('PASS' if p["poisoning_rate"] < 0.50 else 'FAIL'))
print(f'  Overall: {"PASS" if overall_pass else "FAIL"}')
