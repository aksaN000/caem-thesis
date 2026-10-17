"""Recompute every CHM number with the Haiku-judged correctness label on TruthfulQA (final drafting pass, 2026-10-17).

The committed CHM tables read the harness `em` field, which for TruthfulQA is the stale ROUGE-L > 0.15
match; EM everywhere uses the Haiku-judged field `em_llm_judged`. This script re-scores the eight CHM
flags of eval/metrics.py::per_sample_chm per question and averages the seven that can fire (the off-topic flag reads
the question-answer match score, which cannot detect off-topic answers; it is recorded but left out of the mean, tracker C14/C15), with `em` replaced by `em_llm_judged` on
TruthfulQA (other benchmarks unchanged), for:
  CAEM C0..C5 (outputs/full_run/eval), Phase 1d C3 (outputs/phase1d_eval, classifier-off arm),
  baselines B1..B7 (outputs/baselines_phase1e; B6 at its cycle five). B8 has no valid verifier scores.
TruthfulQA judged labels for baselines come from the raw output file (the *_with_chm copy of FLARE lacks
them); verifier fields come from the *_with_chm file.

Writes outputs/analysis/chm_judged.json: per system, per benchmark and pooled (all five; training
panel), the CHM, its eight subtype rates and EM; per CAEM cycle, CHM per tier. Read-only on inputs.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import _EVASIVE_RX, _TEMPLATE_LEAK_RX  # noqa: E402

BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")
TRAIN = ("fever", "triviaqa", "commonsense_qa")
SUBTYPES = ("confident_confabulation", "factual_fabrication", "logical_fabrication", "off_topic",
            "defensive_evasion", "template_leak", "false_refusal", "over_long")
BL = ROOT / "outputs/baselines_phase1e"
BASELINES = ("zero_shot", "cot", "rag", "cot_rag", "fiveshot_cot", "vanilla_ft", "flare")


def flags(s):
    """The eight CHM flags of per_sample_chm, in SUBTYPES order."""
    em = s.get("em") or 0.0
    wrong, right = em == 0.0, em == 1.0
    pred = s.get("prediction") or ""
    pga, pe, qar = s.get("p_ground_atomic"), s.get("p_entail"), s.get("q_a_relevance")
    return [wrong and (s.get("u_stored") or 0.0) >= 0.50,
            wrong and pga is not None and pga < 0.30,
            wrong and pe is not None and pe < 0.30,
            wrong and qar is not None and qar < 0.50,
            wrong and bool(_EVASIVE_RX.search(pred)),
            bool(_TEMPLATE_LEAK_RX.search(pred)),
            right and s.get("decision") in ("ABSTAIN", "DISCARD"),
            wrong and len(pred) > 600]


def load(path):
    blob = json.load(open(path))
    return blob["samples"] if isinstance(blob, dict) else blob


def judged(samples, bench, label_path=None):
    if bench != "truthfulqa":
        return samples
    src = load(label_path) if label_path else samples
    lab = {str(s["id"]): float(s["em_llm_judged"]) for s in src if s.get("em_llm_judged") is not None}
    # A missing judged label is an empty answer the judge could not parse (16 B4 answers); it counts as
    # wrong, as in the reported EM (em_llm_judged or 0).
    return [dict(s, em=lab.get(str(s["id"]), 0.0)) for s in samples]


def summarise(samples):
    n = len(samples)
    F = [flags(s) for s in samples]
    rates = {k: sum(f[i] for f in F) / n for i, k in enumerate(SUBTYPES)}
    active = [i for i, k in enumerate(SUBTYPES) if k != "off_topic"]
    return {"n": n, "chm": sum(sum(f[i] for i in active) for f in F) / (len(active) * n), "em": sum((s.get("em") or 0) for s in samples) / n,
            "subtypes": rates}


def system(get):
    per = {b: get(b) for b in BENCHES}
    out = {b: summarise(per[b]) for b in BENCHES}
    out["pooled"] = summarise([s for b in BENCHES for s in per[b]])
    out["train"] = summarise([s for b in TRAIN for s in per[b]])
    return out, per


def main():
    res, tiers = {}, {}
    for c in range(6):
        out, per = system(lambda b: judged(load(ROOT / f"outputs/full_run/eval/{b}_cycle{c}.json"), b))
        res[f"caem_c{c}"] = out
        allS = [s for b in BENCHES for s in per[b]]
        tiers[f"c{c}"] = {t: summarise([s for s in allS if s.get("tier") == t]) for t in (1, 2, 3)
                          if any(s.get("tier") == t for s in allS)}
    res["phase1d_c3"], _ = system(lambda b: judged(load(ROOT / f"outputs/phase1d_eval/{b}_cycle3.json"), b))
    for name in BASELINES:
        if name == "vanilla_ft":
            chm_p = lambda b: BL / "vanilla_ft/eval" / f"{b}_cycle5_with_chm.json"
            raw_p = lambda b: BL / "vanilla_ft/eval" / f"{b}_cycle5.json"
        else:
            chm_p = lambda b, n=name: BL / n / f"{b}_cycle0_with_chm.json"
            raw_p = lambda b, n=name: BL / n / f"{b}_cycle0.json"
        res[name], _ = system(lambda b: judged(load(chm_p(b)), b, raw_p(b)))
    res["_tiers"] = tiers
    out = ROOT / "outputs/analysis/chm_judged.json"
    json.dump(res, open(out, "w"), indent=1)
    for k, v in res.items():
        if k.startswith("_"):
            continue
        print(f"{k:12s} pooled CHM {v['pooled']['chm']:.3f} EM {v['pooled']['em']:.3f} | TQA CHM {v['truthfulqa']['chm']:.3f}"
              f" | conf {v['pooled']['subtypes']['confident_confabulation']:.3f} fr {v['pooled']['subtypes']['false_refusal']:.3f}")


if __name__ == "__main__":
    main()
