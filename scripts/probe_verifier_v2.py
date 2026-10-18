"""Exploration for the post-defense verifier (2026-10-19). Not part of the thesis run; read-only on run-1 files.

Question: with the same labels CAEM already uses (the calibration fold), how well can a better verifier separate right
from wrong answers? Trained on the C3 calibration fold (500 labelled answers per training benchmark), tested on the C3
evaluation fold (300 per benchmark; TruthfulQA and StrategyQA have no calibration fold and use the pooled model).

Steps (run in order; each writes to outputs/rerun_v2/verifier_v2/):
  questions  write cal3_questions.json (calibration questions) for scripts/precompute_passages.py search/texts --name cal3
  features   per answer (calibration + evaluation, C3 generator = base + cycle-3 adapter): answer-only log-probability
             (mean, first token) and the last-layer hidden state averaged over the answer tokens, given the exact
             prompt CAEM used (Tier 2, or Tier 3 with the cached top 5 passages) plus the model's own reasoning
  se         semantic entropy on extracted final answers (TriviaQA, TruthfulQA evaluation answers): K samples at T=1,
             answers normalised and grouped
  fit        AUROC per benchmark on the evaluation fold for: (a) the stored composite; (b) logistic regression on
             the stored signals minus dead ones, plus answer confidence, per benchmark; (c) a logistic probe on the
             hidden state plus (b)'s features; (d) semantic entropy alone. Writes probe_verifier_v2.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/rerun_v2/verifier_v2"
RET = ROOT / "outputs/rerun_v2/retrieval"
TRAIN = ("fever", "triviaqa", "commonsense_qa")
ALL = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")
# Stored signals kept (h_norm is switched off and constant; alias/entity checks only fire on TriviaQA-style names).
SIGNALS = ("u_token", "u_dropout", "s_avg", "p_entail", "p_ground_max", "p_ground_mean", "q_a_relevance")


def rows(split, b):
    if split == "cal":
        p = ROOT / f"outputs/full_run/cycle_3/calibration/{b}_cycle3.json"
    else:
        p = ROOT / f"outputs/full_run/eval/{b}_cycle3.json"
    if not p.exists():
        return []
    S = json.load(open(p))
    S = S["samples"] if isinstance(S, dict) else S
    lab = "em" if split == "cal" else "em_final"
    return [dict(s, label=float(s[lab])) for s in S if s.get("tier") in (2, 3) and s.get(lab) is not None]


def cmd_questions():
    OUT.mkdir(parents=True, exist_ok=True)
    qs = list(dict.fromkeys(s["question"] for b in TRAIN for s in rows("cal", b)))
    json.dump(qs, open(RET / "cal3_questions.json", "w"))
    print(len(qs), "calibration questions ->", RET / "cal3_questions.json")


def passages_for(split):
    name = "cal3" if split == "cal" else "eval_fold"
    texts = json.load(open(RET / f"{name}_passages.json"))
    qs = json.load(open(RET / f"{name}_questions.json"))
    sc = np.load(RET / f"{name}_scores.npy")
    return {q: list(zip(texts[q][:5], [float(x) for x in sc[i][:5]])) for i, q in enumerate(qs)}


def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, str(ROOT / "outputs/full_run/cycle_3/adapter")).eval()
    return model, tok


def prompt_for(s, tok, pas):
    from caem.prompts import build_tier2_prompt, build_tier3_prompt
    if s["tier"] == 2:
        return build_tier2_prompt(s["question"], tok)[0]
    return build_tier3_prompt(s["question"], pas.get(s["question"], []), tok)[0]


def cmd_features():
    import torch
    model, tok = load_model()
    for split in ("cal", "eval"):
        pas = passages_for(split)
        for b in ALL:
            R = rows(split, b)
            if not R:
                continue
            lp_mean, lp_first, hid, keep = [], [], [], []
            for i, s in enumerate(R):
                pred = s.get("prediction") or ""
                m = list(re.finditer(r"Answer:", pred))
                if not m:
                    continue
                head = pred[:m[-1].end()]
                ans = " " + pred[m[-1].end():].split("\n")[0].strip()
                if not ans.strip():
                    continue
                prompt = prompt_for(s, tok, pas)
                ctx = tok(prompt + head, return_tensors="pt", add_special_tokens=False).input_ids
                full = tok(prompt + head + ans, return_tensors="pt", add_special_tokens=False).input_ids
                n = full.shape[1] - ctx.shape[1]
                if n <= 0:
                    continue
                full = full[:, -4096:].cuda()
                with torch.no_grad():
                    o = model(full, output_hidden_states=True)
                lg = torch.log_softmax(o.logits[0, -n - 1:-1].float(), -1)
                lp = lg.gather(1, full[0, -n:].unsqueeze(1)).squeeze(1)
                lp_mean.append(float(lp.mean())); lp_first.append(float(lp[0]))
                hid.append(o.hidden_states[-1][0, -n:].float().mean(0).cpu().numpy().astype(np.float16))
                keep.append(i)
            np.savez(OUT / f"feat_{split}_{b}.npz", idx=np.array(keep), lp_mean=np.array(lp_mean),
                     lp_first=np.array(lp_first), hid=np.stack(hid) if hid else np.zeros((0, 1)))
            print(f"features {split} {b}: {len(keep)}/{len(R)}", flush=True)


def norm_answer(t):
    t = t.lower().strip().strip(".").strip()
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t).split())


def cmd_se(k=10):
    import torch
    model, tok = load_model()
    pas = passages_for("eval")
    tok.padding_side = "left"
    res = {}
    for b in ("triviaqa", "truthfulqa"):
        R = rows("eval", b)
        ent = []
        for s in R:
            prompt = prompt_for(s, tok, pas) + "Reasoning:"
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=256,
                                     num_return_sequences=k, pad_token_id=tok.eos_token_id)
            answers = []
            for seq in out:
                txt = tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True)
                m = re.findall(r"Answer:\s*(.*)", txt)
                answers.append(norm_answer(m[-1].split("\n")[0]) if m else "")
            groups = []
            for a in answers:  # group answers that are equal or contain one another
                for g in groups:
                    if a and g[0] and (a == g[0] or a in g[0] or g[0] in a):
                        g.append(a); break
                else:
                    groups.append([a])
            p = np.array([len(g) for g in groups], float) / k
            ent.append(float(-(p * np.log2(p)).sum() / np.log2(k)))
        res[b] = ent
        print(f"se {b}: {len(ent)}", flush=True)
    json.dump(res, open(OUT / "se_eval.json", "w"))


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s, float)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def table(split, b):
    R = rows(split, b)
    f = np.load(OUT / f"feat_{split}_{b}.npz")
    R = [R[i] for i in f["idx"]]
    X = np.array([[float(r.get(k) if r.get(k) is not None else 0.5) for k in SIGNALS] for r in R])
    X = np.column_stack([X, f["lp_mean"], f["lp_first"]])
    y = np.array([r["label"] for r in R])
    comp = np.array([float(r.get("u_stored") or 0) for r in R])
    return X, f["hid"].astype(np.float32), y, comp


def cmd_fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cal = {b: table("cal", b) for b in TRAIN}
    Xp = np.vstack([cal[b][0] for b in TRAIN]); Hp = np.vstack([cal[b][1] for b in TRAIN])
    yp = np.concatenate([cal[b][2] for b in TRAIN])
    lr = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    pooled_b = lr(1.0).fit(Xp, yp)
    pooled_c = lr(0.05).fit(np.column_stack([Xp, Hp]), yp)
    se = json.load(open(OUT / "se_eval.json")) if (OUT / "se_eval.json").exists() else {}
    out = {}
    for b in ALL:
        X, H, y, comp = table("eval", b)
        if b in TRAIN:
            mb = lr(1.0).fit(cal[b][0], cal[b][2])
            mc = lr(0.05).fit(np.column_stack([cal[b][0], cal[b][1]]), cal[b][2])
        else:
            mb, mc = pooled_b, pooled_c
        r = {"n_eval": int(len(y)), "trained_on": "own calibration fold" if b in TRAIN else "pooled calibration folds",
             "a_stored_composite": auroc(y, comp),
             "answer_confidence_alone": auroc(y, X[:, len(SIGNALS)]),
             "b_signals_plus_answer_conf": auroc(y, mb.predict_proba(X)[:, 1]),
             "c_plus_hidden_state_probe": auroc(y, mc.predict_proba(np.column_stack([X, H]))[:, 1])}
        if b in se:
            R = rows("eval", b)
            yl = [r["label"] for r in R]
            r["d_semantic_entropy_answers"] = auroc(yl, [-e for e in se[b]])
        out[b] = r
        print(b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    json.dump(out, open(ROOT / "outputs/analysis/probe_verifier_v2.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("questions", "features", "se", "fit"))
    a = ap.parse_args()
    {"questions": cmd_questions, "features": cmd_features, "se": cmd_se, "fit": cmd_fit}[a.step]()
