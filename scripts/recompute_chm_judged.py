"""Recompute every CHM number from the final-version sources and labels (2026-10-17).

Per question, the eight CHM flags of eval/metrics.py::per_sample_chm are re-scored with `em` = the final label
(`em_final`: uniform scorer, TruthfulQA Haiku judge under the final-answer protocol) and the seven that can fire
are averaged (the off-topic flag reads the question-answer match score, which cannot detect off-topic answers;
it is recorded but left out of the mean, tracker C14/C15). Systems and files: scripts/final_sources.py
(CAEM C0-C5, Phase 1d C3, baselines B1-B8 with matched prompts; B8 now has verifier scores).

Writes outputs/analysis/chm_judged.json: per system, per benchmark and pooled (all five; training panel), the
CHM, its eight subtype rates and EM; per CAEM cycle, CHM per tier. Read-only on inputs.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import _EVASIVE_RX, _TEMPLATE_LEAK_RX  # noqa: E402
from scripts import final_sources as FS  # noqa: E402

BENCHES, TRAIN = FS.BENCHES, FS.TRAIN
SUBTYPES = ("confident_confabulation", "factual_fabrication", "logical_fabrication", "off_topic",
            "defensive_evasion", "template_leak", "false_refusal", "over_long")


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


def withheld(s):
    """The user is shown no answer: CAEM ABSTAIN (the production envelope hides the answer behind an
    insufficient-evidence message) or B8 "<abstain>". A DISCARD answer is shown with a 'conflicting' warning
    (author decision 2026-10-19: delivered view follows the production envelope, caem/production/confidence.py)."""
    return s.get("decision") == "ABSTAIN" or "<abstain>" in (s.get("prediction") or "")


TAGS = {"STORE": "verified", "DEFERRED": "provisional", "DISCARD": "conflicting", "ABSTAIN": "insufficient"}


def tag_accuracy(samples):
    """Share of questions and accuracy per status tag the user sees (CAEM only)."""
    n = len(samples)
    out = {}
    for d, tag in TAGS.items():
        rows = [s for s in samples if s.get("decision") == d]
        out[tag] = {"share": len(rows) / n, "accuracy": sum((s.get("em") or 0) for s in rows) / len(rows) if rows else None}
    return out


def summarise(samples):
    """Model view (EM, CHM on every raw answer) and delivered view (what the user sees; author decision 2026-10-17).

    Delivered: answered share, wrong among answered, CHM among answered (the seven active flags over answered
    questions), refusal rate, and correct answers withheld (known only where the raw answer exists, i.e. CAEM).
    """
    n = len(samples)
    F = [flags(s) for s in samples]
    rates = {k: sum(f[i] for f in F) / n for i, k in enumerate(SUBTYPES)}
    active = [i for i, k in enumerate(SUBTYPES) if k != "off_topic"]
    shown = [(s, f) for s, f in zip(samples, F) if not withheld(s)]
    na = len(shown)
    delivered = {"answered": na / n, "refusal_rate": 1 - na / n,
                 "wrong_among_answered": sum(1 - (s.get("em") or 0) for s, _ in shown) / na if na else None,
                 "chm_among_answered": sum(sum(f[i] for i in active) for _, f in shown) / (len(active) * na) if na else None,
                 "correct_withheld": sum((s.get("em") or 0) == 1.0 for s in samples if withheld(s)) / n}
    return {"n": n, "chm": sum(sum(f[i] for i in active) for f in F) / (len(active) * n), "em": sum((s.get("em") or 0) for s in samples) / n,
            "subtypes": rates, "delivered": delivered}


def system(get):
    per = {b: get(b) for b in BENCHES}
    out = {b: summarise(per[b]) for b in BENCHES}
    out["pooled"] = summarise([s for b in BENCHES for s in per[b]])
    out["train"] = summarise([s for b in TRAIN for s in per[b]])
    return out, per


def main():
    res, tiers = {}, {}
    for name in FS.CAEM_CYCLES + ("phase1d_c3",) + FS.BASELINES:
        res[name], per = system(lambda b, n=name: FS.load(n, b))
        if name in FS.CAEM_CYCLES:
            allS = [s for b in BENCHES for s in per[b]]
            tiers[name.replace("caem_", "")] = {t: summarise([s for s in allS if s.get("tier") == t])
                                                for t in (1, 2, 3) if any(s.get("tier") == t for s in allS)}
            res[name]["tags"] = tag_accuracy(allS)
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
