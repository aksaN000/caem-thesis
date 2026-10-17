"""Benchmark and scoring audit (final-version, 2026-10-17). Read-only; CPU only, seconds.

Checks, per benchmark:
  1. Leakage: evaluation-fold ids against calibration/purity/train ids (dataset_splits.json), and evaluation
     question texts against every stream-chunk question text; duplicate questions inside the evaluation fold.
  2. Gold labels: missing gold, label distribution on the closed benchmarks.
  3. Closed-benchmark answers (FEVER, CommonsenseQA, StrategyQA), per system: share of final answers that are
     not a valid label, and agreement between the stored EM and a direct recomputation (valid label == gold).
Writes outputs/analysis/audit_benchmarks.json.
"""
import ast
import glob
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")
BL = ROOT / "outputs/baselines_phase1e"
SYSTEMS = {"caem_c0": "outputs/full_run/eval/{b}_cycle0.json", "caem_c5": "outputs/full_run/eval/{b}_cycle5.json",
           "B1": "outputs/baselines_phase1e/zero_shot/{b}_cycle0.json", "B2": "outputs/baselines_phase1e/cot/{b}_cycle0.json",
           "B3": "outputs/baselines_phase1e/rag/{b}_cycle0.json", "B4": "outputs/baselines_phase1e/cot_rag/{b}_cycle0.json",
           "B5": "outputs/baselines_phase1e/fiveshot_cot/{b}_cycle0.json", "B6": "outputs/baselines_phase1e/vanilla_ft/eval/{b}_cycle5.json",
           "B7": "outputs/baselines_phase1e/flare/{b}_cycle0.json", "B8": "outputs/baselines_phase1e/semantic_entropy/{b}_cycle0.json"}
VALID = {"fever": {"supports", "refutes", "not enough info"}, "strategyqa": {"yes", "no"},
         "commonsense_qa": {"a", "b", "c", "d", "e"}}


def load(p):
    blob = json.load(open(ROOT / p))
    return blob["samples"] if isinstance(blob, dict) else blob


def norm_q(q):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (q or "").lower()).split())


def final_answer(pred):
    m = re.findall(r"answer\s*:\s*(.*)", pred or "", re.I)
    a = (m[-1] if m else (pred or "")).strip().split("\n")[0]
    a = a.strip().strip(".").strip().lower()
    a = re.sub(r"^\(?([a-e])\)?(\s.*)?$", r"\1", a) if len(a) <= 40 else a
    return a


def gold(s):
    g = s.get("gold_label") or s.get("gold_answers")
    if isinstance(g, str) and g.startswith("["):
        g = ast.literal_eval(g)
    if isinstance(g, list):
        g = g[0] if g else None
    return str(g).strip().lower() if g is not None else None


def main():
    splits = json.load(open(ROOT / "outputs/full_run/dataset_splits.json"))
    out = {"leakage": {}, "gold": {}, "closed_answers": {}}
    for b in BENCHES:
        sp = splits[b]
        ev = set(map(str, sp["eval_ids"]))
        id_overlap = {k: len(ev & set(map(str, sp[k]))) for k in ("calib_ids", "purity_ids", "train_ids") if k in sp}
        eval_rows = load(f"outputs/full_run/eval/{b}_cycle0.json")
        eq = [norm_q(s["question"]) for s in eval_rows]
        stream_q = set()
        for f in glob.glob(str(ROOT / f"outputs/full_run/eval/{b}_cycle*_streamchunk.json")):
            stream_q |= {norm_q(s["question"]) for s in load(Path(f).relative_to(ROOT))}
        out["leakage"][b] = {"eval_n": len(eval_rows), "id_overlap": id_overlap,
                             "eval_questions_in_stream_chunks": sum(q in stream_q for q in eq),
                             "duplicate_questions_in_eval": len(eq) - len(set(eq))}
        golds = [gold(s) for s in eval_rows]
        out["gold"][b] = {"missing": sum(g in (None, "", "none") for g in golds),
                          "distribution": dict(Counter(golds).most_common(6)) if b in VALID else None}
        if b in VALID:
            out["closed_answers"][b] = {}
            for name, pat in SYSTEMS.items():
                p = ROOT / pat.format(b=b)
                if not p.exists():
                    continue
                rows = load(pat.format(b=b))
                invalid = em_mismatch = 0
                for s in rows:
                    a = final_answer(s.get("prediction"))
                    ok = a in VALID[b]
                    invalid += not ok
                    recomputed = 1.0 if ok and a == gold(s) else 0.0
                    em_mismatch += recomputed != float(s.get("em") or 0)
                out["closed_answers"][b][name] = {"n": len(rows), "invalid_answer": invalid, "em_mismatch": em_mismatch}
    json.dump(out, open(ROOT / "outputs/analysis/audit_benchmarks.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
