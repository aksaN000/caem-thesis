"""Audit of how CAEM's own answers are scored (final-version, 2026-10-17). Read-only, CPU, seconds.

Needs em_final in the CAEM evaluation files (scripts/rescore_truthfulqa.py + scripts/rescore_uniform.py).
  1. Why CAEM answers are wrong on the closed benchmarks (format problem vs a different answer), C0/C2/C5.
  2. TruthfulQA judge v1 -> v2 per cycle, and how many lost rows are refusals.
  3. Withheld answers. EM and CHM read CAEM's raw answer even when the verifier withheld it (ABSTAIN shows
     "I do not know.", DISCARD shows nothing). Reports, per cycle: correct answers withheld; the selective view
     with ABSTAIN-only vs ABSTAIN+DISCARD counted as not answered; CHM as scored vs with withheld answers counted
     only as refusals (the user never sees them).
  4. Verifier quality per benchmark: AUROC of the stored composite score for correctness, and right/wrong per
     decision (STORE >= store threshold, DEFERRED >= defer threshold, else ABSTAIN if grounding is low, else DISCARD).
Writes outputs/analysis/audit_caem_scoring.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import _norm_words, csqa_letter, final_answer_line, score_uniform  # noqa: E402
from scripts.recompute_chm_judged import SUBTYPES, flags  # noqa: E402
from scripts.rescore_uniform import gold_list  # noqa: E402

BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")
WITHHELD = ("ABSTAIN", "DISCARD")
ACTIVE = [i for i, k in enumerate(SUBTYPES) if k != "off_topic"]


def rows(b, c):
    return json.load(open(ROOT / f"outputs/full_run/eval/{b}_cycle{c}.json"))["samples"]


def wrong_causes(c):
    out = {}
    for b in ("fever", "triviaqa", "commonsense_qa", "strategyqa"):
        cnt = Counter()
        for s in rows(b, c):
            p = s.get("prediction") or ""
            em, lenient = score_uniform(b, p, gold_list(s), s.get("gold_label"))
            na = _norm_words(final_answer_line(p))
            if em:
                cnt["right"] += 1
            elif not p.strip() or "answer:" not in p.lower():
                cnt["no_answer_line"] += 1
            elif b == "fever" and not any(na.startswith(x) for x in ("supports", "refutes", "not enough info")):
                cnt["invalid_label"] += 1
            elif b == "strategyqa" and na.split(" ")[:1] not in (["yes"], ["no"]):
                cnt["invalid_label"] += 1
            elif b == "commonsense_qa" and csqa_letter(final_answer_line(p)) is None:
                cnt["invalid_label"] += 1
            elif b == "triviaqa" and lenient:
                cnt["lenient_only"] += 1
            else:
                cnt["different_answer"] += 1
        out[b] = dict(cnt)
    return out


def judge_change(c):
    S = rows("truthfulqa", c)
    lost = [s for s in S if (s.get("em_llm_judged_v1") or 0) > s["em_llm_judged"]]
    return {"v1": sum(float(s.get("em_llm_judged_v1") or 0) for s in S) / len(S),
            "v2": sum(float(s["em_llm_judged"]) for s in S) / len(S),
            "lost": len(lost), "gained": sum((s.get("em_llm_judged_v1") or 0) < s["em_llm_judged"] for s in S),
            "lost_on_withheld_rows": sum(s.get("decision") in WITHHELD for s in lost),
            "lost_where_v1_graded_the_refusal_string": sum(s.get("decision") == "ABSTAIN" for s in lost)}


def withheld(c):
    S = [dict(s, em=s["em_final"]) for b in BENCHES for s in rows(b, c)]
    n = len(S)
    res = {"n": n, "em": sum(s["em"] for s in S) / n,
           "decision_x_correct": {f"{d}:{int(e)}": v for (d, e), v in
                                  sorted(Counter((s.get("decision"), s["em"]) for s in S).items(), key=str)}}
    for key, hidden in (("abstain_only", ("ABSTAIN",)), ("abstain_and_discard", WITHHELD)):
        a = [s for s in S if s.get("decision") not in hidden]
        res[f"selective_{key}"] = {"answered": len(a) / n,
                                   "wrong_among_answered": sum(1 - s["em"] for s in a) / len(a),
                                   "correct_and_shown": sum(s["em"] for s in a) / n}
    F = [flags(s) for s in S]
    U = []
    for s, f in zip(S, F):
        if s.get("decision") in WITHHELD:
            f = [False] * len(SUBTYPES)
            f[SUBTYPES.index("false_refusal")] = s["em"] == 1.0
        U.append(f)
    chm = lambda X: sum(sum(f[i] for i in ACTIVE) for f in X) / (len(ACTIVE) * n)
    res["chm_as_scored"] = chm(F)
    res["chm_withheld_as_refusal"] = chm(U)
    res["withheld_wrong_flagged"] = sum(s.get("decision") in WITHHELD and s["em"] == 0 and
                                        any(f[i] for i in ACTIVE if SUBTYPES[i] != "false_refusal")
                                        for s, f in zip(S, F))
    return res


def verifier_quality(c):
    out = {}
    for b in BENCHES:
        S = rows(b, c)
        pos = [s["u_stored"] for s in S if s["em_final"] == 1 and s.get("u_stored") is not None]
        neg = [s["u_stored"] for s in S if s["em_final"] == 0 and s.get("u_stored") is not None]
        auroc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg)) if pos and neg else None
        d = Counter((s.get("decision"), int(s["em_final"])) for s in S)
        out[b] = {"auroc": auroc, **{f"{k}_right_wrong": [d[(k, 1)], d[(k, 0)]]
                                     for k in ("STORE", "DEFERRED", "ABSTAIN", "DISCARD")}}
    return out


def main():
    out = {f"C{c}": {"wrong_causes": wrong_causes(c), "truthfulqa_judge": judge_change(c), "withheld": withheld(c),
                  "verifier": verifier_quality(c)}
           for c in (0, 2, 5)}
    (ROOT / "outputs/analysis/audit_caem_scoring.json").write_text(json.dumps(out, indent=1))
    for c, r in out.items():
        w = r["withheld"]
        print(c, f"EM {w['em']:.3f} | CHM {w['chm_as_scored']:.3f} -> {w['chm_withheld_as_refusal']:.3f} withheld-as-refusal"
                 f" | answered {w['selective_abstain_only']['answered']:.1%} vs {w['selective_abstain_and_discard']['answered']:.1%}"
                 f" | TQA judge {r['truthfulqa_judge']['v1']:.3f} -> {r['truthfulqa_judge']['v2']:.3f}")


if __name__ == "__main__":
    main()
