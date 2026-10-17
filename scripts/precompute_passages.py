"""Exact passage search for a fixed set of questions, done once (final-version, 2026-10-17).

The verifier and Tier 3 search the Wikipedia index with the question text alone
(`caem/pipeline.py::_verifier_passage_retriever`: all-mpnet-base-v2 embedding in bf16, L2-normalised, exact inner-product
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


def read_flat_header(path):
    """FAISS IndexFlatIP file: 'IxFI', d (int32), ntotal (int64), 2 x int64, is_trained (1 byte), metric (int32),
    then the code vector: its length (uint64, number of floats) followed by ntotal x d float32 values."""
    import struct
    with open(path, "rb") as f:
        h = f.read(45)
    assert h[:4] == b"IxFI", h[:4]
    d = struct.unpack("<i", h[4:8])[0]
    ntotal = struct.unpack("<q", h[8:16])[0]
    metric = struct.unpack("<i", h[33:37])[0]
    nfloat = struct.unpack("<Q", h[37:45])[0]
    assert metric == 0 and nfloat == ntotal * d, (metric, nfloat)
    return ntotal, d, 45


def eval_fold_questions():
    """The 1,500 evaluation-fold questions (identical across cycles and baselines)."""
    qs = []
    for b in BENCHES:
        for s in json.load(open(ROOT / f"outputs/full_run/eval/{b}_cycle0.json"))["samples"]:
            qs.append(s["question"])
    return list(dict.fromkeys(qs))  # unique, order kept


def cmd_search(name, k, engine="faiss"):
    import sys
    sys.path.insert(0, str(ROOT))
    from caem.memory.encoder import QueryEncoder
    t0 = time.time()
    qs = eval_fold_questions() if name == "eval_fold" else json.load(open(OUT / f"{name}_questions.json"))
    log(t0, f"{len(qs)} questions")
    # The pipeline's own encoder (all-mpnet-base-v2 in bf16 on GPU, L2-normalised), one question at a time as in
    # the run, then float32 + normalise exactly as caem/pipeline.py::_verifier_passage_retriever does.
    enc = QueryEncoder(device="cuda")
    emb = np.stack([enc.encode(q).astype(np.float32) for q in qs])
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    log(t0, f"encoded, shape {emb.shape}")
    OUT.mkdir(parents=True, exist_ok=True)
    if engine == "faiss":
        # Needs RAM for the whole flat index (61 GB): FAISS's own exact inner-product search, as in the run.
        import faiss
        index = faiss.read_index(str(INDEX / "passages.faiss"))
        log(t0, f"FAISS index in RAM: {index.ntotal} vectors ({type(index).__name__})")
        scores = np.zeros((len(qs), k), dtype=np.float32)
        ids = np.zeros((len(qs), k), dtype=np.int64)
        for i in range(0, len(qs), 64):
            scores[i:i + 64], ids[i:i + 64] = index.search(emb[i:i + 64], k)
        log(t0, "search done")
        np.save(OUT / f"{name}_ids.npy", ids)
        np.save(OUT / f"{name}_scores.npy", scores)
        json.dump(qs, open(OUT / f"{name}_questions.json", "w"))
        log(t0, f"saved to {OUT}")
        return
    # The installed FAISS cannot memory-map a flat index (it loads all 61 GB into RAM), so the same exact
    # inner-product search is done here by streaming the stored vectors from disk in chunks and scoring them
    # on the GPU in full fp32 (TF32 off), keeping a running exact top-k per question.
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ntotal, d, offset = read_flat_header(INDEX / "passages.faiss")
    vecs = np.memmap(INDEX / "passages.faiss", dtype=np.float32, mode="r", offset=offset, shape=(ntotal, d))
    log(t0, f"flat index: {ntotal} vectors, d {d}")
    q = torch.from_numpy(emb).cuda()
    best_s = torch.full((len(qs), k), -float("inf"), device="cuda")
    best_i = torch.full((len(qs), k), -1, dtype=torch.long, device="cuda")
    chunk = 250_000
    for start in range(0, ntotal, chunk):
        block = torch.from_numpy(np.ascontiguousarray(vecs[start:start + chunk])).cuda()
        sc = q @ block.T
        s_top, i_top = torch.topk(sc, min(k, sc.shape[1]), dim=1)
        cat_s = torch.cat([best_s, s_top], 1)
        cat_i = torch.cat([best_i, i_top + start], 1)
        keep = torch.topk(cat_s, k, dim=1)
        best_s, best_i = keep.values, torch.gather(cat_i, 1, keep.indices)
        if (start // chunk) % 10 == 0:
            log(t0, f"  {start + block.shape[0]:,} / {ntotal:,}")
        del block, sc
    scores, ids = best_s.cpu().numpy(), best_i.cpu().numpy()
    log(t0, "search done")
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
    import torch
    from sentence_transformers import CrossEncoder
    t0 = time.time()
    cached = json.load(open(OUT / f"{name}_passages.json"))
    # Loaded as the run loaded it (scripts/run_experiment.py: bf16 on GPU).
    rr = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", automodel_args={"torch_dtype": torch.bfloat16})
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
            top3 = [cands[int(i)] for i in np.argsort(scores)[::-1][:3]]  # same ordering call as the verifier
            n += 1
            if top3 == stored[:3]:
                same += 1
            elif len(mismatches) < 20:
                mismatches.append({"benchmark": b, "id": s["id"], "same_set": set(top3) == set(stored[:3])})
    same_set = sum(1 for m in mismatches if m["same_set"])
    res = {"checked": n, "identical_top3": same, "share": same / max(n, 1),
           "mismatch_same_set_in_first_20": same_set, "first_mismatches": mismatches}
    json.dump(res, open(OUT / f"{name}_check_vs_run1.json", "w"), indent=1)
    log(t0, json.dumps({k: v for k, v in res.items() if k != "first_mismatches"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("search", "texts", "check"))
    ap.add_argument("--name", default="eval_fold")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--engine", choices=("faiss", "stream"), default="faiss",
                    help="faiss: index in RAM (rental); stream: chunked disk read + GPU fp32 (low-RAM machines)")
    a = ap.parse_args()
    {"search": lambda: cmd_search(a.name, a.k, a.engine), "texts": lambda: cmd_texts(a.name),
     "check": lambda: cmd_check(a.name)}[a.step]()
