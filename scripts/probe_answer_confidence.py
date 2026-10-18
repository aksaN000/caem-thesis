"""Probe: does the generator's confidence in the final answer alone separate right from wrong? (2026-10-18)

CAEM's token signal averages the probability over the whole response (reasoning + answer). This probe re-scores the
saved CAEM answers with the generator that wrote them (C0: base model; C3: the cycle-3 adapter) and measures, per
answer, the probability of the final-answer tokens only, given the exact prompt CAEM used (Tier 2 prompt, or Tier 3
prompt with the cached top 5 passages) plus the model's own reasoning. Read-only on the evaluation files; GPU, minutes.

Reports AUROC against the final correctness label (em_final), per benchmark, next to the stored full-response token
signal. Writes outputs/analysis/probe_answer_confidence.json.
Usage: python scripts/probe_answer_confidence.py --cycles 0 3
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caem.prompts import build_tier2_prompt, build_tier3_prompt  # noqa: E402

BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")


def auroc(pos, neg):
    if not pos or not neg:
        return None
    import bisect
    neg = sorted(neg)
    w = 0.0
    for p in pos:
        lo, hi = bisect.bisect_left(neg, p), bisect.bisect_right(neg, p)
        w += lo + 0.5 * (hi - lo)
    return w / (len(pos) * len(neg))


def answer_logprobs(model, tok, prompt, prediction):
    """Mean and first-token log-probability of the text after the last 'Answer:' in the prediction."""
    m = list(re.finditer(r"Answer:\s*", prediction))
    if not m:
        return None
    head, ans = prediction[:m[-1].end()], prediction[m[-1].end():].split("\n")[0].rstrip()
    if not ans:
        return None
    ctx = tok(prompt + head, return_tensors="pt", add_special_tokens=False).input_ids
    full = tok(prompt + head + ans, return_tensors="pt", add_special_tokens=False).input_ids
    n_ans = full.shape[1] - ctx.shape[1]
    if n_ans <= 0:
        return None
    full = full[:, -4096:].to(model.device)
    with torch.no_grad():
        logits = model(full).logits[0, -n_ans - 1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, full[0, -n_ans:].unsqueeze(1)).squeeze(1)
    return float(lp.mean()), float(lp[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, nargs="+", default=[0, 3])
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.bfloat16, device_map="cuda")
    base.eval()
    texts = json.load(open(ROOT / "outputs/rerun_v2/retrieval/eval_fold_passages.json"))
    scores = json.load(open(ROOT / "outputs/rerun_v2/retrieval/eval_fold_questions.json"))
    import numpy as np
    sc = np.load(ROOT / "outputs/rerun_v2/retrieval/eval_fold_scores.npy")
    qscore = {q: sc[i] for i, q in enumerate(scores)}
    out = {}
    for c in a.cycles:
        model = base
        if c > 0:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, str(ROOT / f"outputs/full_run/cycle_{c}/adapter"))
            model.eval()
        out[f"C{c}"] = {}
        for b in BENCHES:
            rows = json.load(open(ROOT / f"outputs/full_run/eval/{b}_cycle{c}.json"))["samples"]
            res = []
            for s in rows:
                if s.get("tier") not in (2, 3) or s.get("em_final") is None:
                    continue
                q, pred = s["question"], s.get("prediction") or ""
                if s["tier"] == 2:
                    prompt, _ = build_tier2_prompt(q, tok)
                else:
                    passages = list(zip(texts[q][:5], [float(x) for x in qscore[q][:5]]))
                    prompt, _ = build_tier3_prompt(q, passages, tok)
                lp = answer_logprobs(model, tok, prompt, pred)
                if lp is None:
                    continue
                res.append((s["em_final"], lp[0], lp[1], s.get("u_token"), s["tier"]))
            pos = lambda k: [r[k] for r in res if r[0] == 1 and r[k] is not None]
            neg = lambda k: [r[k] for r in res if r[0] == 0 and r[k] is not None]
            out[f"C{c}"][b] = {"n": len(res), "auroc_answer_mean_logprob": auroc(pos(1), neg(1)),
                               "auroc_answer_first_token": auroc(pos(2), neg(2)),
                               "auroc_stored_full_response_token": auroc(pos(3), neg(3))}
            print(f"C{c} {b:15s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[f'C{c}'][b].items()},
                  flush=True)
        if c > 0:
            base = model.unload()
    (ROOT / "outputs/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "outputs/analysis/probe_answer_confidence.json", "w"), indent=1)


if __name__ == "__main__":
    main()
