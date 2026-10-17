"""Post-hoc relevance of each saved response to its question (final drafting pass, 2026-10-17).

The run's relevance signal was computed on the bare wrapped answer ("Answer: Rudolph.") and passed
through a second sigmoid (tracker C14), so the off-topic CHM subtype never fired. This script re-scores
the full saved response (reasoning and answer) against the question with the same cross-encoder
(BAAI/bge-reranker-v2-m3, sentence-transformers CrossEncoder, which already returns a probability; no
second sigmoid). A reranker is trained on question-passage pairs, so a response that discusses the
question scores high whether it is right or wrong, and a response that discusses something else scores low.

No experiment is re-run: the inputs are the saved evaluation files. Systems: CAEM C0..C5, Phase 1d C3,
baselines B1..B7 (B6 at cycle five). Output: outputs/analysis/response_relevance.json,
{system: {benchmark: {sample_id: relevance}}}.

Needs a GPU environment with sentence-transformers (run with ~/caem_reranker_env/bin/python).
"""
import json
from pathlib import Path

from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
BENCHES = ("fever", "triviaqa", "commonsense_qa", "truthfulqa", "strategyqa")
BL = ROOT / "outputs/baselines_phase1e"


def sources():
    for c in range(6):
        yield f"caem_c{c}", lambda b, c=c: ROOT / f"outputs/full_run/eval/{b}_cycle{c}.json"
    yield "phase1d_c3", lambda b: ROOT / f"outputs/phase1d_eval/{b}_cycle3.json"
    for n in ("zero_shot", "cot", "rag", "cot_rag", "fiveshot_cot", "flare"):
        yield n, lambda b, n=n: BL / n / f"{b}_cycle0.json"
    yield "vanilla_ft", lambda b: BL / "vanilla_ft/eval" / f"{b}_cycle5.json"


def main():
    model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512)
    out = {}
    for name, path in sources():
        out[name] = {}
        for b in BENCHES:
            rows = json.load(open(path(b)))
            rows = rows["samples"] if isinstance(rows, dict) else rows
            pairs = [(s["question"], s.get("prediction") or "") for s in rows]
            scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
            out[name][b] = {str(s["id"]): float(x) for s, x in zip(rows, scores)}
        print(name, "done", flush=True)
    json.dump(out, open(ROOT / "outputs/analysis/response_relevance.json", "w"))


if __name__ == "__main__":
    main()
