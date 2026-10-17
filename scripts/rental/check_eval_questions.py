"""Gate: every re-run baseline file must hold exactly CAEM's evaluation-fold questions, in order (final-version).

Usage: python scripts/rental/check_eval_questions.py outputs/rerun_v2/baselines
Exits 1 on any mismatch, empty answer, or wrong count (memory rule: verify sample counts before trusting a run).
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")


def rows(p):
    d = json.load(open(p))
    return d["samples"] if isinstance(d, dict) else d


def main(root):
    caem = {b: [s["question"] for s in rows(ROOT / f"outputs/full_run/eval/{b}_cycle0.json")] for b in BENCHES}
    bad = 0
    files = sorted(glob.glob(f"{root}/*/*_cycle0.json") + glob.glob(f"{root}/vanilla_ft/eval/*_cycle5.json"))
    for f in files:
        if "_with_chm" in f:
            continue
        b = next((x for x in BENCHES if Path(f).name.startswith(x + "_")), None)
        if b is None:
            continue
        r = rows(f)
        q = [s["question"] for s in r]
        empty = sum(not (s.get("prediction") or "").strip() or (s.get("prediction") or "").strip() == "Reasoning:"
                    for s in r)
        errors = sum(bool(s.get("pipeline_error")) for s in r)
        ok = q == caem[b] and empty == 0 and errors == 0
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {f}: n={len(q)} same_questions_in_order={q == caem[b]} "
              f"empty={empty} errors={errors}")
    print(f"{len(files)} files checked, {bad} failed")
    return 1 if bad or not files else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "outputs/rerun_v2/baselines"))
