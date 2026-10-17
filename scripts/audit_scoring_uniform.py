"""Uniform re-scoring of every system on every benchmark (final-version, 2026-10-17). Read-only, CPU, seconds.

Compares the stored EM with one fair rule applied identically to all systems:
  FEVER        final answer starts with a valid label (supports / refutes / not enough info); no "Answer:" line ->
               the response itself (B6 writes "Reasoning:supports").
  CSQA         exactly one letter A-E, optionally bracketed and followed by option text; two different letters = wrong.
  StrategyQA   final answer starts with yes / no.
  TriviaQA     strict (stored) and lenient (a gold alias occurs in the answer line; an upper bound).
  TruthfulQA   Haiku judgement as stored (em_llm_judged; empty = wrong).
Writes outputs/analysis/audit_scoring_uniform.json.
"""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")
SYSTEMS = {"CAEM C0": "outputs/full_run/eval/{b}_cycle0.json", "CAEM C2": "outputs/full_run/eval/{b}_cycle2.json",
           "CAEM C5": "outputs/full_run/eval/{b}_cycle5.json",
           "B1": "outputs/baselines_phase1e/zero_shot/{b}_cycle0.json", "B2": "outputs/baselines_phase1e/cot/{b}_cycle0.json",
           "B3": "outputs/baselines_phase1e/rag/{b}_cycle0.json", "B4": "outputs/baselines_phase1e/cot_rag/{b}_cycle0.json",
           "B5": "outputs/baselines_phase1e/fiveshot_cot/{b}_cycle0.json", "B6": "outputs/baselines_phase1e/vanilla_ft/eval/{b}_cycle5.json",
           "B7": "outputs/baselines_phase1e/flare/{b}_cycle0.json", "B8": "outputs/baselines_phase1e/semantic_entropy/{b}_cycle0.json"}


def norm(x):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(x).lower()).split())


def answer_line(pred):
    m = re.findall(r"answer\s*:\s*(.*)", pred or "", re.I)
    if m:
        a = m[-1].strip().split("\n")[0]
        return re.sub(r"^(?:answer\s*:\s*)+", "", a, flags=re.I)  # "Answer:Answer:Yes" (B6)
    t = re.sub(r"^\s*reasoning\s*:\s*", "", pred or "", flags=re.I).strip()
    return t.split("\n")[-1] if t else ""


def golds(s):
    g = s.get("gold_answers")
    if isinstance(g, str) and g.startswith("["):
        g = ast.literal_eval(g)
    if not isinstance(g, list):
        g = [g]
    lab = s.get("gold_label")
    return [str(x) for x in g if x is not None], (str(lab).lower() if lab is not None else None)


def fair(b, s):
    a = answer_line(s.get("prediction"))
    na = norm(a)
    g, lab = golds(s)
    if b == "fever":
        for l in ("not enough info", "supports", "refutes"):
            if na.startswith(l):
                return 1.0 if l == (lab or "").lower() else 0.0
        return 0.0
    if b == "strategyqa":
        w = na.split(" ")[0] if na else ""
        return 1.0 if w in ("yes", "no") and w == (lab or "").lower() else 0.0
    if b == "commonsense_qa":
        letters = set(re.findall(r"(?:^|[\s(\[|,/])([A-E])(?=$|[\s)\].:,|/])", a.strip()))
        m = re.match(r"^\(?\[?([A-Ea-e])\)?\]?(?:[\s.:)\-].*)?$", a.strip())
        if m and len({x.upper() for x in letters} | {m.group(1).upper()}) == 1:
            return 1.0 if m.group(1).upper() == (lab or "").upper() else 0.0
        return 0.0
    if b == "triviaqa":
        return 1.0 if (s.get("em") or 0) >= 1 or any(norm(x) and f" {norm(x)} " in f" {na} " for x in g) else 0.0
    return float(s.get("em_llm_judged") or 0)


def stored(b, s):
    return float(s.get("em_llm_judged") or 0) if b == "truthfulqa" else float(s.get("em") or 0)


def main():
    out = {}
    for name, pat in SYSTEMS.items():
        out[name] = {}
        for b in BENCHES:
            p = ROOT / pat.format(b=b)
            rows = json.load(open(p))
            rows = rows["samples"] if isinstance(rows, dict) else rows
            st = [stored(b, s) for s in rows]
            fr = [fair(b, s) for s in rows]
            out[name][b] = {"n": len(rows), "stored": sum(st) / len(rows), "fair": sum(fr) / len(rows),
                            "fair_gains": sum(1 for x, y in zip(st, fr) if y > x),
                            "fair_losses": sum(1 for x, y in zip(st, fr) if y < x)}
        out[name]["pooled"] = {k: sum(out[name][b][k] for b in BENCHES) / 5 for k in ("stored", "fair")}
    json.dump(out, open(ROOT / "outputs/analysis/audit_scoring_uniform.json", "w"), indent=1)
    hdr = f"{'system':8s} " + " ".join(f"{b[:9]:>17s}" for b in BENCHES) + f" {'POOLED':>13s}"
    print(hdr)
    for name, r in out.items():
        cells = " ".join(f"{r[b]['stored']:.3f}->{r[b]['fair']:.3f}(+{r[b]['fair_gains']}/-{r[b]['fair_losses']})".rjust(17) for b in BENCHES)
        print(f"{name:8s} {cells} {r['pooled']['stored']:.3f}->{r['pooled']['fair']:.3f}")


if __name__ == "__main__":
    main()
