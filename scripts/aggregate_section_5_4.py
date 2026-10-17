"""
scripts/aggregate_section_5_4.py
================================
Builds Ch5 §5.4 (Three-tier router efficiency) tables:
  - tab_tier_trajectory.tex   per-cycle pooled tier share + per-tier EM
  - tab_tier_latency.tex      per-cycle per-tier mean latency + pooled
  - tab_safety_override.tex   safety-override firings, u_pre < per-benchmark floor

Data sources:
  outputs/full_run/all_cycle_results.json   per-bench tier shares
  outputs/full_run/eval/<bench>_cycle<N>.json  per-sample tier + latency + u_pre
  outputs/full_run/eval/<bench>_cycle<N>_streamchunk.json  SIL stream chunks (u_pre)
  outputs/full_run/calibration/calibrated_config[_cycle<N>].json
      per-benchmark safety floors (safety_u_pre_min_per_benchmark) in effect
      at cycle N; an aborted cycle (no file) keeps the previous cycle's floors.

The safety override is counted as u_pre < floor_b(cycle), the inequality the
router branches on (caem/routing/router.py, cfg.get_safety_u_pre_min_for).
The per-sample `escalated` field is a baseline/FLARE flag, not the override,
and is NOT used. The safety_override boolean is not persisted in the eval
JSON, so the inequality is the measurement.
"""
from __future__ import annotations

# Guard (2026-10-17): this script builds tables at module level, so importing it would overwrite files.
# Run it directly instead; it is replaced by the single results command (MASTER_ISSUES.md, Phase 1a).
if __name__ != "__main__":
    raise ImportError(f"{__file__} does its work when loaded; run it directly, do not import it.")

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHES = ['fever', 'triviaqa', 'commonsense_qa', 'strategyqa', 'truthfulqa']
EVAL_DIR = ROOT / 'outputs/full_run/eval'
CALIB_DIR = ROOT / 'outputs/full_run/calibration'
OUT_DIR = ROOT / 'thesis_report/figures/auto'
AUTO_NOTE = ('% NOTE: the committed thesis copy may be hand-finished; compare numbers with '
             'thesis_report/ref_check/local_regen_check.py before overwriting')
STREAM_BENCHES = ['fever', 'triviaqa', 'commonsense_qa']
BENCH_NAME = {'fever': 'FEVER', 'triviaqa': 'TriviaQA', 'commonsense_qa': 'CommonsenseQA',
              'strategyqa': 'StrategyQA', 'truthfulqa': 'TruthfulQA'}
NUM_WORD = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
            6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}


def _pooled_safety_floor() -> float:
    """Pooled CAEMConfig.safety_u_pre_min (0.38); constant if caem is not importable."""
    try:
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from caem.config import CAEMConfig
        return float(CAEMConfig().safety_u_pre_min)
    except Exception:  # caem/__init__ pulls torch; fall back to the registered value
        return 0.38


def safety_floors_by_cycle(calib_dir, cycles, benches, pooled):
    """{cycle: {bench: floor}} -- per-benchmark safety floor in effect at each cycle.

    Cycle 0 reads calibrated_config.json; cycle N >= 1 reads
    calibrated_config_cycle{N}.json (written at the cycle-N recalibration,
    before the cycle-N eval). A missing file (aborted cycle) carries the
    previous cycle's dict forward, as the runner does. Benchmarks absent
    from the dict use the pooled floor (config.get_safety_u_pre_min_for).
    """
    floors = {}
    current = {}
    for c in cycles:
        f = Path(calib_dir) / ('calibrated_config.json' if c == 0 else f'calibrated_config_cycle{c}.json')
        if f.is_file():
            d = json.load(f.open()).get('safety_u_pre_min_per_benchmark') or {}
            current = {str(k): float(v) for k, v in d.items()}
        floors[c] = {b: current.get(b, pooled) for b in benches}
    return floors


def count_safety_fires(samples, floor):
    """(fires, n, min_u_pre) with fires = #samples with u_pre < floor.

    Samples without a numeric u_pre count towards n but cannot fire.
    """
    fires = 0
    min_u = None
    for s in samples:
        u = s.get('u_pre')
        if u is None:
            continue
        u = float(u)
        if u < floor:
            fires += 1
        min_u = u if min_u is None else min(min_u, u)
    return fires, len(samples), min_u


def load_eval(bench: str, cycle: int):
    p = EVAL_DIR / f'{bench}_cycle{cycle}.json'
    if not p.is_file():
        return []
    d = json.load(p.open())
    return d['samples'] if isinstance(d, dict) else d


def per_cycle_aggregate():
    """Returns {cycle: {tier: {n, em_sum, lat_sum, chm_n}}}."""
    out = {}
    for c in range(6):
        bucket = {1: {'n': 0, 'em_sum': 0.0, 'lat_sum': 0.0},
                  2: {'n': 0, 'em_sum': 0.0, 'lat_sum': 0.0},
                  3: {'n': 0, 'em_sum': 0.0, 'lat_sum': 0.0}}
        for b in BENCHES:
            for s in load_eval(b, c):
                tier = s.get('tier')
                if tier in bucket:
                    is_tqa = (b == 'truthfulqa')
                    em = s['em_llm_judged'] if (is_tqa and s.get('em_llm_judged') is not None) else s.get('em', 0)
                    bucket[tier]['n'] += 1
                    bucket[tier]['em_sum'] += em
                    bucket[tier]['lat_sum'] += s.get('latency_ms', 0)
        out[c] = bucket
    return out


def fmt_pct(x, dp=1):
    if x is None or x != x:
        return '--'
    return f'{x*100:.{dp}f}\\%'


def fmt(x, dp=3):
    if x is None or x != x:
        return '--'
    return f'{x:.{dp}f}'


# ============================================================================
# Build tables
# ============================================================================
agg = per_cycle_aggregate()

# Table 1: tier_trajectory — pooled share + EM per tier per cycle
print('Building tab_tier_trajectory.tex...')
lines = ['% Auto-generated by scripts/aggregate_section_5_4.py']
lines.append(AUTO_NOTE)
lines.append('% Per-cycle pooled tier share + per-tier exact match.')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\small')
lines.append('\\begin{tabular}{c|cc|cc|cc|c}')
lines.append('\\toprule')
lines.append('  & \\multicolumn{2}{c|}{\\textbf{Tier 1 (memory)}} & \\multicolumn{2}{c|}{\\textbf{Tier 2 (zero-shot)}} & \\multicolumn{2}{c|}{\\textbf{Tier 3 (RAG)}} & \\\\')
lines.append('\\textbf{Cycle} & share & EM & share & EM & share & EM & \\textbf{Pooled EM} \\\\')
lines.append('\\midrule')

for c in range(6):
    counts = agg[c]
    total = sum(v['n'] for v in counts.values())
    row = [f'C{c}']
    pooled_em_num = 0.0
    pooled_em_den = 0
    for tier in [1, 2, 3]:
        v = counts[tier]
        share = v['n'] / total if total else 0
        em = v['em_sum'] / v['n'] if v['n'] else float('nan')
        row.append(fmt_pct(share, 1))
        row.append(fmt(em, 3))
        pooled_em_num += v['em_sum']
        pooled_em_den += v['n']
    pooled_em = pooled_em_num / pooled_em_den if pooled_em_den else float('nan')
    row.append(fmt(pooled_em, 3))
    lines.append(' & '.join(row) + ' \\\\')

lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Per-cycle pooled tier share and per-tier exact match across the five-benchmark evaluation panel. '
             'For each cycle the table reports the share of evaluation queries served by each tier and the within-tier exact-match rate. '
             'The pooled exact match in the rightmost column is the sample-weighted mean over the three within-tier rates. '
             'The memory-direct tier (Tier 1) fires on a small share of evaluation queries because the evaluation fold is held content-hash disjoint from the training stream by construction; '
             'when Tier 1 does fire, every realised firing returns the correct stored answer (an exact-match rate of $1.0$), the empirical receipt for the Tier-1 floor corollary of \\Cref{cor:tier1-floor}. '
             'Tier 2 exact match at cycle zero exceeds Tier 3 exact match at cycle zero by approximately twelve percentage points, the self-improvement-as-distillation receipt: the parametric tier already outperforms the retrieval tier at cold-start because retrieval pulls evidence that is not always topical, and the self-improvement loop subsequently transfers retrieved knowledge into the parametric tier without re-routing.}')
lines.append('\\label{tab:tier-trajectory}')
lines.append('\\end{table}')
(OUT_DIR / 'tab_tier_trajectory.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/tab_tier_trajectory.tex')


# Table 2: tier_latency — per-cycle per-tier mean latency
print('Building tab_tier_latency.tex...')
lines = ['% Auto-generated by scripts/aggregate_section_5_4.py']
lines.append(AUTO_NOTE)
lines.append('% Per-cycle pooled latency from the production batched eval log.')
lines.append('% NOTE: per-tier latency cannot be honestly extracted from batched-eval logs')
lines.append('%       because latency_ms = batch_elapsed/N at bs=32, so per-tier amortised')
lines.append('%       readings reflect the per-batch mix not the intrinsic per-tier cost.')
lines.append('%       Only the pooled-per-cycle reading is deployment-relevant from this log.')
lines.append('\\begin{table}[!htbp]')
lines.append('\\centering\\small')
lines.append('\\begin{tabular}{cccc}')
lines.append('\\toprule')
lines.append('\\textbf{Cycle} & \\textbf{Pooled mean per-query (ms)} & \\textbf{Pooled per-query (s)} & \\textbf{Per-cycle wall time (s)} \\\\')
lines.append('\\midrule')

for c in range(6):
    counts = agg[c]
    total_lat_sum = sum(counts[t]['lat_sum'] for t in [1, 2, 3])
    total_lat_n = sum(counts[t]['n'] for t in [1, 2, 3])
    pooled_lat_ms = total_lat_sum / total_lat_n if total_lat_n else None
    per_query_s = pooled_lat_ms / 1000 if pooled_lat_ms else None
    cycle_wall_s = total_lat_sum / 1000 if pooled_lat_ms else None

    def lat_fmt(x, dp=0):
        return f'{x:.{dp}f}' if x is not None else '--'

    lines.append(f'C{c} & {lat_fmt(pooled_lat_ms, 0)} & {lat_fmt(per_query_s, 1)} & {lat_fmt(cycle_wall_s, 0)} \\\\')

lines.append('\\bottomrule')
lines.append('\\end{tabular}')
lines.append('\\caption{Per-cycle pooled wall-clock latency on the production batched evaluation log. '
             'Each cell reports the per-query mean over the full one thousand five hundred evaluation queries at the indicated cycle. '
             'Reported figures are the production-batched amortised values at the deployment batch size of thirty-two. '
             'Per-tier intrinsic latency is deliberately not decomposed from the production log because the per-sample latency record reflects the per-batch wall time divided by the batch size, '
             'so any per-tier amortisation reflects the per-batch tier mix rather than the per-tier compute cost. '
             'The intrinsic per-tier latency (memory lookup at unit batch, zero-shot generation at unit batch, retrieval-augmented generation at unit batch) is a separate measurement that the writeup-hardware envelope cannot host (the dense passage index does not fit in the available random-access memory), and the measurement is registered as a future-work item in \\Cref{ch:conclusion}.}')
lines.append('\\label{tab:tier-latency}')
lines.append('\\end{table}')
(OUT_DIR / 'tab_tier_latency.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/tab_tier_latency.tex')


# Table 3: safety override -- u_pre < per-benchmark floor in effect at the cycle
print('Building tab_safety_override.tex...')
CYCLES = list(range(6))
SAFETY_ORDER = ['fever', 'triviaqa', 'commonsense_qa', 'truthfulqa', 'strategyqa']  # row order of the thesis table
POOLED_FLOOR = _pooled_safety_floor()
floors = safety_floors_by_cycle(CALIB_DIR, CYCLES, BENCHES, POOLED_FLOOR)
if not any((CALIB_DIR / n).is_file() for n in
           ['calibrated_config.json'] + [f'calibrated_config_cycle{c}.json' for c in CYCLES[1:]]):
    print(f'  WARNING: no calibrated_config*.json under {CALIB_DIR}; every floor = pooled {POOLED_FLOOR}. '
          'The main run raised the CommonsenseQA floor (0.491 at C3-C4, 0.5687 at C5), '
          'so the CommonsenseQA count will be too low.')


def fnum(n):
    return f'{n:,}'.replace(',', '\\,')


def ffloor(x):
    return f'{x:.3f}'.rstrip('0').rstrip('.')


def cycles_words(cs):
    w = [NUM_WORD.get(c, str(c)) for c in cs]
    return ('cycle ' + w[0]) if len(w) == 1 else ('cycles ' + ', '.join(w[:-1]) + ' and ' + w[-1])


def and_join(parts):
    return parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1]


def panel_rows(per_bench):
    """per_bench: {bench: (fires, n, min_u)} -> table rows + subtotal tuple."""
    rows = []
    mins = [v[2] for v in per_bench.values() if v[2] is not None]
    pmin = min(mins) if mins else None
    for b, (f, n, mu) in per_bench.items():
        rate = f / n if n else 0.0
        if mu is None:
            mu_s = '---'
        elif mu == pmin:
            mu_s = f'$\\mathbf{{{mu:.3f}}}$'
        else:
            mu_s = f'${mu:.3f}$'
        rows.append(f'{BENCH_NAME[b]} & ${fnum(n)}$ & ${f}$ & ${rate*100:.3f}\\%$ & {mu_s} \\\\')
    tf = sum(v[0] for v in per_bench.values())
    tn = sum(v[1] for v in per_bench.values())
    return rows, tf, tn, pmin


# (a) eval fold, per-benchmark floor at each cycle
eval_bench = {}
eval_by_cycle = {}          # {bench: {cycle: fires}}
eval_sizes = set()
eval_cycles = set()
for b in SAFETY_ORDER:
    tf = tn = 0
    mu = None
    for c in CYCLES:
        smp = load_eval(b, c)
        if not smp:
            continue
        eval_sizes.add(len(smp))
        eval_cycles.add(c)
        f, n, m = count_safety_fires(smp, floors[c][b])
        eval_by_cycle.setdefault(b, {})[c] = f
        tf += f
        tn += n
        if m is not None:
            mu = m if mu is None else min(mu, m)
    eval_bench[b] = (tf, tn, mu)
a_rows, a_f, a_n, a_min = panel_rows(eval_bench)

# (b) SIL stream chunks, pooled floor only (as in the committed table)
stream_bench = {}
stream_cycles = set()
for b in STREAM_BENCHES:
    tf = tn = 0
    mu = None
    for c in CYCLES:
        p = EVAL_DIR / f'{b}_cycle{c}_streamchunk.json'
        if not p.is_file():
            continue
        d = json.load(p.open())
        smp = d['samples'] if isinstance(d, dict) else d
        stream_cycles.add(c)
        f, n, m = count_safety_fires(smp, POOLED_FLOOR)
        tf += f
        tn += n
        if m is not None:
            mu = m if mu is None else min(mu, m)
    if tn:
        stream_bench[b] = (tf, tn, mu)
have_b = bool(stream_bench)

# Benchmarks / cycles whose floor differs from the pooled floor
raised = {}   # {bench: [(floor, [cycles])]}
for b in BENCHES:
    groups = []
    for c in CYCLES:
        fl = floors[c][b]
        if fl == POOLED_FLOOR:
            continue
        if groups and groups[-1][0] == fl and groups[-1][1][-1] == c - 1:
            groups[-1][1].append(c)
        else:
            groups.append((fl, [c]))
    if groups:
        raised[b] = groups

lines = ['% Auto-generated by scripts/aggregate_section_5_4.py']
lines.append(AUTO_NOTE)
lines.append('% Safety override firing rate measured by u_pre < safety floor (the inequality the')
lines.append('% router branches on, caem/routing/router.py via cfg.get_safety_u_pre_min_for).')
lines.append('% The dumped safety_override boolean is not persisted in the eval / streamchunk JSON,')
lines.append('% so the inequality is the measurement. The `escalated` field is NOT the override.')
lines.append('% Panel (a) floors per cycle (from outputs/full_run/calibration/calibrated_config*.json,')
lines.append(f'% pooled fallback {ffloor(POOLED_FLOOR)}):')
for c in CYCLES:
    lines.append(f'%   C{c}: ' + ', '.join(f'{b}={floors[c][b]:g}' for b in BENCHES))
lines.append('\\begin{table}[H]')
lines.append('\\centering\\footnotesize')
lines.append('\\setlength{\\tabcolsep}{5pt}')
lines.append('\\begin{tabular}{@{}lrrrr@{}}')
lines.append('\\toprule')
lines.append('\\textbf{Source / benchmark} & \\textbf{$n$} & \\textbf{Fires} & \\textbf{Rate} & \\textbf{Min $u_{\\text{pre}}$} \\\\')
lines.append('\\midrule')
size_s = f' $\\times$ ${fnum(next(iter(eval_sizes)))}$ samples' if len(eval_sizes) == 1 else ''
lines.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{(a) Evaluation fold ({NUM_WORD.get(len(eval_cycles), len(eval_cycles))} cycles $\\times$ '
             f'{NUM_WORD.get(len(BENCHES), len(BENCHES))} benchmarks{size_s})}}}} \\\\')
lines.extend(a_rows)
a_rate = a_f / a_n if a_n else 0.0
lines.append(f'\\quad Eval subtotal & ${fnum(a_n)}$ & $\\mathbf{{{a_f}}}$ & ${a_rate*100:.3f}\\%$ & --- \\\\')
if have_b:
    b_rows, b_f, b_n, b_min = panel_rows(stream_bench)
    b_rate = b_f / b_n if b_n else 0.0
    lines.append('\\midrule')
    lines.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{(b) SIL stream chunks (training-panel candidate pool, '
                 f'{NUM_WORD.get(len(stream_cycles), len(stream_cycles))} cycles; pooled floor ${ffloor(POOLED_FLOOR)}$ only)}}}} \\\\')
    lines.extend(b_rows)
    lines.append(f'\\quad Stream-chunk subtotal & ${fnum(b_n)}$ & $\\mathbf{{{b_f}}}$ & ${b_rate*100:.3f}\\%$ & --- \\\\')
    lines.append('\\midrule')
    c_f, c_n = a_f + b_f, a_n + b_n
    c_min = min(x for x in (a_min, b_min) if x is not None) if (a_min is not None or b_min is not None) else None
    ge = '\\geq ' if any(b in stream_bench for b in raised) else ''   # (b) pooled floor only -> lower bound
    c_min_s = f'$\\mathbf{{{c_min:.3f}}}$' if c_min is not None else '---'
    lines.append(f'\\textbf{{Combined (a) + (b)}} & $\\mathbf{{{fnum(c_n)}}}$ & $\\mathbf{{{ge}{c_f}}}$ & '
                 f'$\\mathbf{{{ge}{(c_f / c_n if c_n else 0)*100:.3f}\\%}}$ & {c_min_s} \\\\')
lines.append('\\bottomrule')
lines.append('\\end{tabular}')

# Caption (data-driven; wording follows the committed hand-finished table)
if raised:
    clauses = []
    for b, groups in raised.items():
        g = and_join([f'to ${ffloor(fl)}$ at {cycles_words(cs)}' for fl, cs in groups])
        clauses.append(f'{BENCH_NAME[b]}, whose floor is refit {g}')
    floor_s = (f'The floor is per benchmark: ${ffloor(POOLED_FLOOR)}$ throughout, except '
               + '; '.join(clauses) + '. ')
else:
    floor_s = f'The floor is ${ffloor(POOLED_FLOOR)}$ for every benchmark at every cycle. '
fire_parts = []
for b in SAFETY_ORDER:
    by_c = {c: k for c, k in eval_by_cycle.get(b, {}).items() if k}
    k = sum(by_c.values())
    if not k:
        continue
    if k == 1:
        fire_parts.append(f'one {BENCH_NAME[b]} query at {cycles_words(list(by_c))}')
    else:
        detail = ', '.join(f'${v}$ at cycle {NUM_WORD.get(c, c)}' for c, v in sorted(by_c.items()))
        rc = {c for _, cs in raised.get(b, []) for c in cs}
        after = ' after its floor rises' if rc and set(by_c) <= rc else ''
        fire_parts.append(f'${k}$ of them {BENCH_NAME[b]} queries{after} ({detail})')
a_s = (f'Panel (a) audits the served evaluation fold with these floors: the override fires on ${a_f}$ of '
       f'${fnum(a_n)}$ queries (${a_rate*100:.3f}\\%$)' + (', ' + and_join(fire_parts) if fire_parts else '') + '. ')
b_s = ''
if have_b:
    b_s = f'Panel (b) audits the SIL stream chunks with the pooled ${ffloor(POOLED_FLOOR)}$ floor only'
    rb = [b for b in raised if b in stream_bench]
    if rb:
        first = min(cs[0] for b in rb for _, cs in raised[b])
        b_s += (f', so its {and_join([BENCH_NAME[b] for b in rb])} count from cycle '
                f'{NUM_WORD.get(first, first)} onward is a lower bound')
    b_s += '. '
lines.append('\\caption[Safety override firing rate on the pre-routing confidence (\\Cref{sec:router}).]'
             '{Safety override firing rate measured by the inequality the router branches on, '
             '$u_{\\text{pre}} < \\phi_{\\text{safe},b}$, on the pre-routing confidence (\\Cref{sec:router}). '
             + floor_s + a_s + b_s +
             'The dumped \\texttt{safety\\_override} boolean is not persisted in the eval JSON serializer, '
             'so the $u_{\\text{pre}}$ inequality is reported directly; the two measurements are identical by construction.}')
lines.append('\\label{tab:safety-override}')
lines.append('\\end{table}')
(OUT_DIR / 'tab_safety_override.tex').write_text('\n'.join(lines) + '\n')
print(f'  wrote {OUT_DIR}/tab_safety_override.tex  (eval fold: {a_f} of {a_n} fire)')

print('\nAll 3 §5.4 tables built.')
