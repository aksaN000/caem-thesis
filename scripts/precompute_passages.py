"""Exact passage search for a fixed set of questions, done once (final-version, 2026-10-17).

The verifier and Tier 3 search the Wikipedia index with the question text alone
(`caem/pipeline.py::_verifier_passage_retriever`: all-mpnet-base-v2 embedding, L2-normalised, exact inner-product
search over the flat index). The index never changes, so every question always gets the same passages; this
script runs that same exact search for many questions in one pass and saves the result. It changes when the
search runs, not what it returns (MASTER_ISSUES.md, P1). Deployment searches live.

Steps (each can be run separately; outputs go to outputs/rerun_v2/retrieval/):
  search  questions -> top-k passage ids and scores   (<name>_ids.npy, <name>_scores.npy, <name>_questions.json)
  texts   ids -> passage texts                         (<name>_passages.json: {question: [text, ...]})
  check   re-rank the cached top 20 with saved (question, answer) pairs and compare with the 3 passages the
          first run stored for each answer (must be identical)

Usage:
  python -u scripts/precompute_passages.py search --name eval_fold --k 100
  python -u scripts/precompute_passages.py texts  --name eval_fold
  python -u scripts/precompute_passages.py check  --name eval_fold
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/passage_index"
OUT = ROOT / "outputs/rerun_v2/retrieval"
BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")


def log(t0, msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def eval_fold_questions():
    """The 1,500 evaluation-fold questions (identical across cycles and baselines)."""
    qs = []
    for b in BENCHES:
        for s in json.load(open(ROOT / f"outputs/full_run/eval/{b}_cycle0.json"))["samples"]:
            qs.append(s["question"])
    return list(dict.fromkeys(qs))  # unique, order kept


def cmd_search(name, k):
    import faiss
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    qs = eval_fold_questions() if name == "eval_fold" else json.load(open(OUT / f"{name}_questions.json"))
    log(t0, f"{len(qs)} questions")
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    # Same as the pipeline: encode, float32, L2-normalise (one question at a time, as deployment would).
    emb = np.stack([enc.encode(q).astype(np.float32) for q in qs])
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    log(t0, f"encoded, shape {emb.shape}")
    index = faiss.read_index(str(INDEX / "passages.faiss"), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    log(t0, f"index opened: {type(index).__name__}, ntotal {index.ntotal}, d {index.d}")
    scores, ids = index.search(emb, k)
    log(t0, "search done")
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}_ids.npy", ids)
    np.save(OUT / f"{name}_scores.npy", scores)
    json.dump(qs, open(OUT / f"{name}_questions.json", "w"))
    log(t0, f"saved to {OUT}")


def cmd_texts(name):
    t0 = time.time()
    ids = np.load(OUT / f"{name}_ids.npy")
    qs = json.load(open(OUT / f"{name}_questions.json"))
    with open(INDEX / "passages.pkl", "rb") as f:
        passages = pickle.load(f)
    log(t0, f"passage store loaded ({len(passages)} passages)")
    out = {q: [passages[int(i)] for i in row if i >= 0] for q, row in zip(qs, ids)}
    del passages
    json.dump(out, open(OUT / f"{name}_passages.json", "w"))
    log(t0, f"saved {name}_passages.json")


def cmd_check(name):
    """Re-rank the cached top 20 exactly as the verifier does and compare with the first run's stored top 3."""
    from sentence_transformers import CrossEncoder
    t0 = time.time()
    cached = json.load(open(OUT / f"{name}_passages.json"))
    rr = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
    n = same = 0
    mismatches = []
    for b in BENCHES:
        for s in json.load(open(ROOT / f"outputs/full_run/eval/{b}_cycle0.json"))["samples"]:
            stored = s.get("top_passages") or []
            ans = s.get("prediction") or ""
            if not stored or not ans or s.get("tier") == 1:
                continue
            cands = cached[s["question"]][:20]
            scores = rr.predict([(f"{s['question']} {ans}", p) for p in cands], show_progress_bar=False)
            top3 = [cands[i] for i in np.argsort(-np.asarray(scores), kind="stable")[:3]]
            n += 1
            if top3 == stored[:3]:
                same += 1
            elif len(mismatches) < 20:
                mismatches.append({"benchmark": b, "id": s["id"], "same_set": set(top3) == set(stored[:3])})
    res = {"checked": n, "identical_top3": same, "share": same / max(n, 1), "first_mismatches": mismatches}
    json.dump(res, open(OUT / f"{name}_check_vs_run1.json", "w"), indent=1)
    log(t0, json.dumps({k: v for k, v in res.items() if k != "first_mismatches"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("search", "texts", "check"))
    ap.add_argument("--name", default="eval_fold")
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    {"search": lambda: cmd_search(a.name, a.k), "texts": lambda: cmd_texts(a.name),
     "check": lambda: cmd_check(a.name)}[a.step]()
