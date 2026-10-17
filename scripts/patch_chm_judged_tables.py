"""Write the Haiku-judged CHM values into the committed chapter 5 / appendix C tables (final drafting pass, 2026-10-17).

The tables below were laid out by hand in the drafting pass, so their generators no longer reproduce
them; this script rewrites only the CHM-dependent cells from outputs/analysis/chm_judged.json
(scripts/recompute_chm_judged.py) and leaves the layout alone. Idempotent.

  tab_halluc_subtypes.tex       all subtype rows and the CHM row (B1, C0..C5)
  tab_per_bench_trajectory.tex  TruthfulQA (CHM) row (training-panel rows do not involve TruthfulQA)
  tab_baseline_pooled.tex       baseline EM / CHM and the deltas against CAEM C2 and C5
  tab_tier_chm_trajectory.tex   per-tier and pooled CHM
  tab_ruc_ablation.tex          TruthfulQA CHM and pooled CHM, classifier off (Phase 1d C3) and on (main run C3)
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "defense/thesis_report/figures/auto"
J = json.load(open(ROOT / "outputs/analysis/chm_judged.json"))
CYC = [f"caem_c{c}" for c in range(6)]
TAG = "% CHM cells re-scored with the Haiku-judged TruthfulQA label by scripts/patch_chm_judged_tables.py (2026-10-17).\n"


def f3(x):
    return f"{x:.3f}"


def sd(x):
    return f"${x:+.3f}$"


def tag(s):
    return s if TAG in s else TAG + s


def subtypes():
    p = AUTO / "tab_halluc_subtypes.tex"
    s = p.read_text()
    rows = {"Conf.\\ confab.": "confident_confabulation", "Fact.\\ fabric.": "factual_fabrication",
            "Logic.\\ fabric.": "logical_fabrication", "Off-topic": "off_topic", "Def.\\ evasion": "defensive_evasion",
            "Templ.\\ leak": "template_leak", "False refusal": "false_refusal", "Over-long": "over_long"}
    for label, key in rows.items():
        vals = [J["zero_shot"]["pooled"]["subtypes"][key]] + [J[c]["pooled"]["subtypes"][key] for c in CYC]
        new = label + " & " + " & ".join(f3(v) for v in vals) + " \\\\"
        s, n = re.subn(re.escape(label) + r" &[^\n]*\\\\", lambda m: new, s)
        assert n == 1, label
    chm = [J[c]["pooled"]["chm"] for c in CYC]
    new = "\\textbf{CHM (mean of 8)} & \\textbf{" + f3(J["zero_shot"]["pooled"]["chm"]) + "} & " + " & ".join(f3(v) for v in chm) + " \\\\"
    s, n = re.subn(r"\\textbf\{CHM \(mean of 8\)\} &[^\n]*\\\\", lambda m: new, s)
    assert n == 1
    p.write_text(tag(s))


def per_bench():
    p = AUTO / "tab_per_bench_trajectory.tex"
    s = p.read_text()
    b1 = J["zero_shot"]["truthfulqa"]["chm"]
    v = [J[c]["truthfulqa"]["chm"] for c in CYC]
    new = ("TruthfulQA (CHM) & " + f3(b1) + " & " + " & ".join(f3(x) for x in v)
           + f" & {sd(v[2] - b1)} & {sd(v[5] - b1)} \\\\")
    s, n = re.subn(r"TruthfulQA \(CHM\) &[^\n]*\\\\", lambda m: new, s)
    assert n == 1
    p.write_text(tag(s))


def baselines():
    p = AUTO / "tab_baseline_pooled.tex"
    s = p.read_text()
    c2, c5 = J["caem_c2"]["pooled"], J["caem_c5"]["pooled"]
    names = {"B1": ("zero-shot", "zero_shot"), "B2": ("chain-of-thought", "cot"), "B3": ("RAG", "rag"),
             "B4": ("CoT $+$ RAG", "cot_rag"), "B5": ("five-shot CoT", "fiveshot_cot"),
             "B6": ("vanilla fine-tune", "vanilla_ft"), "B7": ("FLARE", "flare")}
    for b, (label, key) in names.items():
        em, chm = J[key]["pooled"]["em"], J[key]["pooled"]["chm"]
        new = (f"{b} & {label} & ${f3(em)}$ & ${f3(chm)}$ & {sd(c2['em'] - em)} & {sd(c2['chm'] - chm)} & "
               f"{sd(c5['em'] - em)} & {sd(c5['chm'] - chm)} \\\\")
        s, n = re.subn(r"^" + b + r" &[^\n]*\\\\", lambda m: new, s, flags=re.M)
        assert n == 1, b
    s, n = re.subn(r"& \\textbf\{CAEM C2 \(best\)\} &[^\n]*\\\\",
                   lambda m: f"& \\textbf{{CAEM C2 (best)}} & $\\mathbf{{{f3(c2['em'])}}}$ & $\\mathbf{{{f3(c2['chm'])}}}$ & --- & --- & {sd(c2['em'] - c5['em'])} & {sd(c2['chm'] - c5['chm'])} \\\\", s)
    assert n == 1
    s, n = re.subn(r"& CAEM C5 \(final\) &[^\n]*\\\\",
                   lambda m: f"& CAEM C5 (final) & ${f3(c5['em'])}$ & ${f3(c5['chm'])}$ & {sd(c5['em'] - c2['em'])} & {sd(c5['chm'] - c2['chm'])} & --- & --- \\\\", s)
    assert n == 1
    # B8: EM only (no valid CHM); keep its EM deltas.
    p.write_text(tag(s))


def tiers():
    p = AUTO / "tab_tier_chm_trajectory.tex"
    s = p.read_text()
    T = J["_tiers"]
    for c in range(6):
        t = T[f"c{c}"]
        cells = []
        for k in ("1", "2", "3"):
            if k in t:
                cells.append(("$" + f3(t[k]["em"]) + "$", "$" + f3(t[k]["chm"]) + "$"))
            else:
                cells.append(("--", "--"))
        pooled = J[f"caem_c{c}"]["pooled"]
        new = (f"C{c} & " + " & ".join(f"{a} & {b}" for a, b in cells)
               + f" & ${f3(pooled['em'])}$ & ${f3(pooled['chm'])}$ \\\\")
        s, n = re.subn(r"^C" + str(c) + r" &[^\n]*\\\\", lambda m: new, s, flags=re.M)
        assert n == 1, c
    p.write_text(tag(s))


def ruc():
    p = AUTO / "tab_ruc_ablation.tex"
    s = p.read_text()
    off, on = J["phase1d_c3"], J["caem_c3"]

    def rel(a, b):
        return f"${100 * (b - a) / a:+.1f}\\%$"
    t_off, t_on = off["truthfulqa"]["chm"], on["truthfulqa"]["chm"]
    s, n = re.subn(r"(TruthfulQA\s*& \$0\.210\$ & \$\\mathbf\{0\.310\}\$ & \$\\mathbf\{\+10\.0\}\$ & )\$[0-9.]+\$ & \$[0-9.]+\$ & \$[+\-][0-9.]+\\%\$",
                   lambda m: m.group(1) + f"${f3(t_off)}$ & ${f3(t_on)}$ & {rel(t_off, t_on)}", s)
    assert n == 1
    p_off, p_on = off["pooled"]["chm"], on["pooled"]["chm"]
    s, n = re.subn(r"(\\textbf\{Pooled \(all five\)\} & \$\\mathbf\{0\.474\}\$ & \$\\mathbf\{0\.532\}\$ & \$\\mathbf\{\+5\.8\}\$ & )\$\\mathbf\{[0-9.]+\}\$ & \$\\mathbf\{[0-9.]+\}\$ & \$\\mathbf\{[+\-][0-9.]+\\%\}\$",
                   lambda m: m.group(1) + f"$\\mathbf{{{f3(p_off)}}}$ & $\\mathbf{{{f3(p_on)}}}$ & $\\mathbf{{{100 * (p_on - p_off) / p_off:+.1f}\\%}}$", s)
    assert n == 1
    p.write_text(tag(s))


if __name__ == "__main__":
    subtypes(); per_bench(); baselines(); tiers(); ruc()
    print("patched")
