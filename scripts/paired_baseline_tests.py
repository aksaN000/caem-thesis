"""Paired CAEM-versus-baseline significance tests on EM and CHM (final drafting pass, 2026-10).

Replaces the approximate CHM check of archive/scripts_superseded/baseline_sig_tests_v2.py (CAEM held fixed,
baseline subtype rates resampled, no Holm) with a true paired test on the per-question CHM of
eval/metrics.py::per_sample_chm. The EM axis is unchanged (McNemar, Holm within each benchmark,
the helpers of scripts/baseline_sig_tests.py), so its counts reproduce the committed ones.

Per (CAEM cycle, benchmark, baseline), on the shared evaluation-fold ids:
  EM   McNemar test; paired bootstrap 95% CI on the mean difference.
  CHM  Wilcoxon signed-rank test on the per-question CHM differences (zero differences dropped);
       paired bootstrap 95% CI on the mean difference (B8 included since the final-version re-run has
       verifier scores).
  Holm-Bonferroni within each benchmark, separately per axis. Practical floors |dEM| >= 0.02,
  |dCHM| >= 0.01. Marker: '*' CAEM significantly better, '(B)' baseline significantly better.
  A verifier-free CHM (the five active subtypes that use neither CAEM's composite score nor its decision,
  divided by seven) is tested the same way as a robustness check.

Sources and correctness label: scripts/final_sources.py (`em_final` for every system and benchmark: uniform scorer,
TruthfulQA Haiku judge under the final-answer protocol), the same label on both axes.

Writes outputs/analysis/paired_baseline_tests_final.json and, with --write_tex, figures/auto/tab_sig_summary.tex
and tab_sig_full_c5.tex / tab_sig_full_c2.tex.
Read-only on its inputs.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import per_sample_chm, _EVASIVE_RX, _TEMPLATE_LEAK_RX  # noqa: E402
from scripts.baseline_sig_tests import _load_eval_ids, _holm_correct, mcnemar_test  # noqa: E402
from scripts import final_sources as FS  # noqa: E402

BENCHES, BASELINES, LABEL = FS.BENCHES, FS.BASELINES, FS.LABEL
BNAME = {"fever": "FEVER", "triviaqa": "TriviaQA", "commonsense_qa": "CommonsenseQA",
         "truthfulqa": "TruthfulQA", "strategyqa": "StrategyQA"}
CYCLES = (0, 2, 5)
NO_CHM: set = set()
EM_FLOOR, CHM_FLOOR, ALPHA, NBOOT = 0.02, 0.01, 0.05, 2000
OUT = ROOT / "defense/thesis_report/figures/auto"


def free_chm(s):
    """Per-question CHM restricted to the five active subtypes that do not use CAEM's composite or decision
    (still divided by seven, the number of active subtypes)."""
    wrong = (s.get("em") or 0.0) == 0.0
    pred = s.get("prediction") or ""
    pga, pe = s.get("p_ground_atomic"), s.get("p_entail")
    flags = [wrong and pga is not None and pga < 0.30, wrong and pe is not None and pe < 0.30,
             wrong and bool(_EVASIVE_RX.search(pred)), bool(_TEMPLATE_LEAK_RX.search(pred)),
             wrong and len(pred) > 600]
    return sum(flags) / 7.0


def chm_maps(samples):
    """{id: (chm, verifier-free chm)} for rows that already carry the final label in `em`.

    CHM is the mean of the seven flags that can fire: per_sample_chm averages eight, one of which
    (off-topic) can never fire, so the seven-type value is per_sample_chm * 8 / 7.
    """
    out = {}
    for s in samples:
        sid = str(s.get("id"))
        out[sid] = (per_sample_chm(s) * 8 / 7, free_chm(s))
    return out


def boot_ci(d, rng):
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilcox_p(d):
    d = np.asarray(d, float)
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox").pvalue)


def marker(diff, p, floor, lower_is_better):
    if p is None or math.isnan(p) or abs(diff) < floor or p >= ALPHA:
        return "-"
    good = diff < 0 if lower_is_better else diff > 0
    return "*" if good else "(B)"


def run(cycle, rng):
    eval_ids = _load_eval_ids(ROOT / "outputs/full_run") or {}
    rows = []
    for bench in BENCHES:
        c_em = FS.id_to_em(f"caem_c{cycle}", bench)
        c_chm = chm_maps(FS.load(f"caem_c{cycle}", bench))
        fam = []
        for b in BASELINES:
            b_em = FS.id_to_em(b, bench)
            allowed = eval_ids.get(bench)
            ids = sorted(set(c_em) & set(b_em) & (allowed if allowed else set(c_em)))
            ce = np.array([c_em[i] for i in ids]); be = np.array([b_em[i] for i in ids])
            _, p_em = mcnemar_test(list(ce), list(be))
            r = {"cycle": cycle, "benchmark": bench, "baseline": b, "n": len(ids),
                 "em_caem": float(ce.mean()), "em_base": float(be.mean()), "d_em": float((ce - be).mean()),
                 "ci_em": boot_ci(ce - be, rng), "p_em": float(p_em)}
            if b not in NO_CHM:
                b_chm = chm_maps(FS.load(b, bench))
                cids = [i for i in ids if i in c_chm and i in b_chm]
                cc = np.array([c_chm[i][0] for i in cids]); bc = np.array([b_chm[i][0] for i in cids])
                cf = np.array([c_chm[i][1] for i in cids]); bf = np.array([b_chm[i][1] for i in cids])
                r.update({"n_chm": len(cids), "chm_caem": float(cc.mean()), "chm_base": float(bc.mean()),
                          "d_chm": float((cc - bc).mean()), "ci_chm": boot_ci(cc - bc, rng),
                          "p_chm": wilcox_p(cc - bc),
                          "free_caem": float(cf.mean()), "free_base": float(bf.mean()),
                          "d_free": float((cf - bf).mean()), "p_free": wilcox_p(cf - bf)})
            fam.append(r)
        for key, hk in (("p_em", "holm_em"), ("p_chm", "holm_chm"), ("p_free", "holm_free")):
            sub = [r for r in fam if key in r]
            for r, h in zip(sub, _holm_correct([r[key] for r in sub])):
                r[hk] = h
        for r in fam:
            r["sig_em"] = marker(r["d_em"], r["holm_em"], EM_FLOOR, False)
            if "d_chm" in r:
                r["sig_chm"] = marker(r["d_chm"], r["holm_chm"], CHM_FLOOR, True)
                r["sig_free"] = marker(r["d_free"], r["holm_free"], CHM_FLOOR, True)
        rows += fam
    return rows


def counts(rows, key):
    sub = [r for r in rows if key in r]
    return sum(r[key] == "*" for r in sub), sum(r[key] == "(B)" for r in sub), len(sub)


def write_tex(res):
    # Compact summary: per baseline, significant wins / defeats over the five benchmarks.
    L = ["% Auto-generated by scripts/paired_baseline_tests.py (final drafting pass 2026-10).",
         "\\begin{table}[!htbp]", "\\centering\\small",
         "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{CAEM C{c}}}}}" for c in CYCLES) + " \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         "\\textbf{Baseline}" + " & \\textbf{EM} & \\textbf{CHM}" * len(CYCLES) + " \\\\", "\\midrule"]

    def cell(rows, b, key):
        sub = [r for r in rows if r["baseline"] == b and key in r]
        if not sub:
            return "n/a"
        w = sum(r[key] == "*" for r in sub); d = sum(r[key] == "(B)" for r in sub)
        return f"{w} / {d}"
    for b in BASELINES:
        L.append(LABEL[b] + "".join(f" & {cell(res[c], b, 'sig_em')} & {cell(res[c], b, 'sig_chm')}" for c in CYCLES) + " \\\\")
    tot = lambda rows, k: "{} / {}".format(*counts(rows, k)[:2])
    L += ["\\midrule",
          "Total" + "".join(f" & {tot(res[c], 'sig_em')} & {tot(res[c], 'sig_chm')}" for c in CYCLES) + " \\\\",
          "\\bottomrule", "\\end{tabular}",
          "\\caption[Significant wins and defeats against each baseline.]{Significant wins / defeats of CAEM against each baseline over the five benchmarks, so each row has $5$ contrasts per cell and the total row $40$ (paired tests, Holm-corrected within each benchmark; \\Cref{app:significance}). C0 is CAEM before any self-training.}",
          "\\label{tab:sig-summary}", "\\end{table}"]
    (OUT / "tab_sig_summary.tex").write_text("\n".join(L) + "\n")

    for cyc in (5, 2):
        rows = res[cyc]
        T = ["% Auto-generated by scripts/paired_baseline_tests.py (final drafting pass 2026-10).",
             "\\begin{table}[!htbp]", "\\centering\\small", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{@{}llrcrc@{}}", "\\toprule",
             "\\textbf{Benchmark} & \\textbf{Baseline} & $\\Delta$\\textbf{EM} [95\\% CI] & & $\\Delta$\\textbf{CHM} [95\\% CI] & \\\\",
             "\\midrule"]
        for bi, bench in enumerate(BENCHES):
            fam = [r for r in rows if r["benchmark"] == bench]
            if bi:
                T.append("\\midrule")
            for j, r in enumerate(fam):
                em = f"${r['d_em']:+.3f}$ [${r['ci_em'][0]:+.2f}$, ${r['ci_em'][1]:+.2f}$]"
                if "d_chm" in r:
                    ch = f"${r['d_chm']:+.3f}$ [${r['ci_chm'][0]:+.2f}$, ${r['ci_chm'][1]:+.2f}$]"
                    sc = r["sig_chm"]
                else:
                    ch, sc = "n/a", ""
                name = BNAME[bench] if j == 0 else ""
                T.append(f"{name} & {LABEL[r['baseline']].split()[0]} & {em} & {r['sig_em']} & {ch} & {sc} \\\\")
        T += ["\\bottomrule", "\\end{tabular}",
              f"\\caption[Per-benchmark paired tests at C{cyc}.]{{Per-benchmark paired tests of CAEM at C{cyc} against each baseline ($300$ questions per benchmark). Baselines are numbered as in \\Cref{{sec:comp-baselines}}. A positive $\\Delta$EM and a negative $\\Delta$CHM favour CAEM; $\\ast$ marks a significant CAEM win and (B) a significant baseline win.}}",
              f"\\label{{tab:sig-full-c{cyc}}}", "\\end{table}"]
        (OUT / f"tab_sig_full_c{cyc}.tex").write_text("\n".join(T) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write_tex", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    res = {c: run(c, rng) for c in (5, 2, 0)}
    (ROOT / "outputs/analysis").mkdir(parents=True, exist_ok=True)
    json.dump({"label": "em_final", "rows": res[5] + res[2] + res[0]},
              open(ROOT / "outputs/analysis/paired_baseline_tests_final.json", "w"), indent=1)
    for c in (5, 2, 0):
        print(f"C{c}", {k: counts(res[c], k) for k in ("sig_em", "sig_chm", "sig_free")})
    if a.write_tex:
        write_tex(res)


if __name__ == "__main__":
    main()
