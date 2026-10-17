"""Write the final-version EM (`em_final`) into saved evaluation files (2026-10-17, MASTER_ISSUES M-B). CPU, seconds.

One rule for every system (eval.metrics.score_uniform): FEVER / StrategyQA label prefix, CommonsenseQA single
letter (option text allowed, two letters wrong), TriviaQA strict alias match (lenient upper bound stored as
`em_lenient`), TruthfulQA = Haiku judgement under the final-answer protocol (`em_llm_judged`, protocol
final_answer_v2; a row without a v2 judgement is counted wrong and reported in `tqa_unjudged`).
Adds fields only; the stored `em` is left unchanged. Summary: outputs/analysis/em_final_summary.json.

Usage:
  python scripts/rescore_uniform.py                       # default final-version file set
  python scripts/rescore_uniform.py --files 'outputs/rerun_v2/baselines/*/*_cycle0.json'
"""
import argparse
import ast
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import score_uniform  # noqa: E402

BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")
DEFAULT_GLOBS = [
    "outputs/full_run/eval/{b}_cycle[0-9].json",
    "outputs/phase1d_eval/{b}_cycle[0-9].json",
    "outputs/baselines_phase1e/*/{b}_cycle0.json",
    "outputs/baselines_phase1e/vanilla_ft/eval_fixed/{b}_cycle*.json",
    "outputs/rerun_v2/baselines/*/{b}_cycle0.json",
    "outputs/rerun_v2/baselines/vanilla_ft/eval/{b}_cycle*.json",
]
TQA_PROTOCOL = "final_answer_v2"


def gold_list(s):
    g = s.get("gold_answers")
    if isinstance(g, str) and g.startswith("["):
        g = ast.literal_eval(g)
    return [str(x) for x in (g if isinstance(g, list) else [g]) if x is not None]


def rescore(path: Path, bench: str) -> dict:
    blob = json.loads(path.read_text())
    rows = blob["samples"] if isinstance(blob, dict) else blob
    unjudged = 0
    for s in rows:
        if bench == "truthfulqa":
            ok = s.get("em_llm_judged_protocol") == TQA_PROTOCOL and s.get("em_llm_judged") is not None
            unjudged += not ok
            s["em_final"] = float(s["em_llm_judged"]) if ok else 0.0
        else:
            em, lenient = score_uniform(bench, s.get("prediction") or "", gold_list(s), s.get("gold_label"))
            s["em_final"] = em
            if lenient is not None:
                s["em_lenient"] = lenient
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, ensure_ascii=False, indent=2))
    tmp.replace(path)
    n = len(rows)
    out = {"n": n, "em_stored": sum(float(s.get("em") or 0) for s in rows) / n,
           "em_final": sum(s["em_final"] for s in rows) / n}
    if bench == "triviaqa":
        out["em_lenient"] = sum(s.get("em_lenient", 0) for s in rows) / n
    if bench == "truthfulqa":
        out["tqa_unjudged"] = unjudged
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=None, help="explicit globs (benchmark read from file name)")
    a = ap.parse_args()
    files = set()
    for pat in (a.files or [g.format(b=b) for g in DEFAULT_GLOBS for b in BENCHES]):
        files |= {Path(f) for f in glob.glob(str(ROOT / pat) if not Path(pat).is_absolute() else pat)}
    files = sorted(f for f in files if not any(x in f.name for x in ("_with_chm", "streamchunk", ".tmp")))
    summary_path = ROOT / "outputs/analysis/em_final_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for f in files:
        bench = next((b for b in BENCHES if f.name.startswith(b + "_")), None)
        if bench is None:
            continue
        key = str(f.relative_to(ROOT))
        summary[key] = rescore(f, bench)
        print(f"{key:75s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in summary[key].items()))
    summary_path.write_text(json.dumps(dict(sorted(summary.items())), indent=1))


if __name__ == "__main__":
    main()
