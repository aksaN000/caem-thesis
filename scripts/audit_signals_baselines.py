"""Audit of verifier signals and system outputs for hidden faults (final drafting pass, 2026-10-17).

Prompted by two faults found late (the double-sigmoided question-match signal, tracker C14; B5's FEVER-label
answers, tracker D15). Read-only on the saved evaluation files; writes outputs/analysis/audit_signals_baselines.json.

Signals (CAEM C0 and C5, all baselines): value range, number of distinct values, share of missing values and of
common sentinel values (0, 0.5, 1), and the area under the ROC curve for correct versus wrong answers.
Outputs (every system and benchmark): sample count, empty responses, responses with no answer line, refusals,
FEVER labels on non-FEVER questions, multiple-choice letters on non-CommonsenseQA questions, likely truncation,
identical responses, share of answers that contain a gold alias but score EM 0 (TriviaQA), and judge coverage
(TruthfulQA).
"""
import ast
import bisect
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")
SIGNALS = ("u_pre", "u_token", "u_dropout", "u_internal", "s_avg", "p_entail", "p_ground_max", "p_ground_mean",
           "p_ground_atomic", "q_a_relevance", "u_stored", "h_norm", "alias_overlap", "entity_head_consistency", "p_contra")
BL = ROOT / "outputs/baselines_phase1e"


def systems():
    for c in (0, 5):
        yield f"caem_c{c}", lambda b, c=c: ROOT / f"outputs/full_run/eval/{b}_cycle{c}.json"
    for n in ("zero_shot", "cot", "rag", "cot_rag", "fiveshot_cot", "flare"):
        yield n, lambda b, n=n: BL / n / f"{b}_cycle0_with_chm.json"
    yield "vanilla_ft", lambda b: BL / "vanilla_ft/eval" / f"{b}_cycle5_with_chm.json"
    yield "semantic_entropy", lambda b: BL / "semantic_entropy" / f"{b}_cycle0.json"


def raw(name, b):
    if name.startswith("caem"):
        return None
    if name == "vanilla_ft":
        return BL / "vanilla_ft/eval" / f"{b}_cycle5.json"
    return BL / name / f"{b}_cycle0.json"


def load(p):
    blob = json.load(open(p))
    return blob["samples"] if isinstance(blob, dict) else blob


def correct(s, b, judged):
    if b == "truthfulqa":
        return 1 if (judged.get(str(s["id"])) or 0) >= 1 else 0
    return 1 if (s.get("em") or 0) >= 1 else 0


def auroc(x, y):
    pos = [a for a, l in zip(x, y) if l]
    neg = sorted(a for a, l in zip(x, y) if not l)
    if not pos or not neg:
        return None
    tot = sum(bisect.bisect_left(neg, p) + 0.5 * (bisect.bisect_right(neg, p) - bisect.bisect_left(neg, p)) for p in pos)
    return tot / (len(pos) * len(neg))


ANSWER = re.compile(r"answer\s*:\s*(.*)", re.I | re.S)
REFUSE = re.compile(r"i do not know|i don.t know|cannot (be )?determine|not enough info", re.I)
FEVER_LABEL = re.compile(r"answer\s*:\s*(supports|refutes|not enough info)\b", re.I)
LETTER = re.compile(r"answer\s*:\s*\(?[A-E]\)?\s*\.?\s*$", re.I)


def norm(x):
    return " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", str(x).lower()).split()) + " "


def main():
    out = {"signals": {}, "outputs": {}}
    for name, path in systems():
        out["outputs"][name] = {}
        sig_vals = {k: [] for k in SIGNALS}
        labels = []
        for b in BENCHES:
            p = path(b)
            if not p.exists():
                out["outputs"][name][b] = {"missing_file": str(p.relative_to(ROOT))}
                continue
            S = load(p)
            rp = raw(name, b)
            src = load(rp) if rp and rp.exists() else S
            judged = {str(s["id"]): s.get("em_llm_judged") for s in src}
            preds = [s.get("prediction") or "" for s in S]
            ans = [ANSWER.search(x) for x in preds]
            rec = {
                "n": len(S),
                "empty": sum(1 for x in preds if not x.strip() or x.strip().lower() in ("reasoning:",)),
                "no_answer_line": sum(1 for a in ans if a is None),
                "empty_answer_line": sum(1 for a in ans if a is not None and not a.group(1).strip()),
                "refusal": sum(1 for x in preds if REFUSE.search(x[-120:])),
                "fever_label_on_other": sum(1 for x in preds if FEVER_LABEL.search(x)) if b != "fever" else None,
                "letter_on_other": sum(1 for x in preds if LETTER.search(x.strip())) if b != "commonsense_qa" else None,
                "likely_truncated": sum(1 for x in preds if len(x) > 300 and ANSWER.search(x) is None),
                "identical_responses_top": Counter(preds).most_common(1)[0][1] if preds else 0,
                "em_mean": sum((s.get("em") or 0) for s in S) / len(S),
            }
            if b == "truthfulqa":
                rec["judged_missing"] = sum(1 for s in S if judged.get(str(s["id"])) is None)
            if b == "triviaqa":
                miss = 0
                for s, a in zip(S, ans):
                    if (s.get("em") or 0) >= 1 or a is None:
                        continue
                    g = s.get("gold_answers") or []
                    g = ast.literal_eval(g) if isinstance(g, str) else g
                    if any(norm(x).strip() and norm(x) in norm(a.group(1)[:200]) for x in g):
                        miss += 1
                rec["gold_in_answer_but_em0"] = miss
            out["outputs"][name][b] = rec
            for s in S:
                labels.append(correct(s, b, judged))
                for k in SIGNALS:
                    sig_vals[k].append(s.get(k))
        sig = {}
        for k, vals in sig_vals.items():
            present = [v for v in vals if isinstance(v, (int, float))]
            if not present:
                sig[k] = {"present": 0}
                continue
            pairs = [(v, l) for v, l in zip(vals, labels) if isinstance(v, (int, float))]
            c = Counter(round(v, 6) for v in present)
            sig[k] = {"present": len(present), "missing": len(vals) - len(present),
                      "min": min(present), "max": max(present), "distinct": len(c),
                      "share_0": c.get(0.0, 0) / len(present), "share_0.5": c.get(0.5, 0) / len(present),
                      "share_1": c.get(1.0, 0) / len(present), "top_value_share": c.most_common(1)[0][1] / len(present),
                      "auroc_correct": auroc([p[0] for p in pairs], [p[1] for p in pairs])}
        out["signals"][name] = sig
    json.dump(out, open(ROOT / "outputs/analysis/audit_signals_baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
