"""Post-hoc off-topic judgement of wrong open-answer responses with Claude Haiku (final drafting pass, 2026-10-17).

The off-topic CHM subtype never fired in the run (tracker C14: double sigmoid on a bare-answer relevance
score), and a reranker cannot judge short or reasoning-led responses reliably (outputs/analysis/
response_relevance.json). This script asks the same Haiku model that judges TruthfulQA whether each
wrong response on the two open-answer benchmarks (TriviaQA, TruthfulQA) addresses the question asked,
regardless of correctness. Closed-answer benchmarks (FEVER labels, CommonsenseQA letters, StrategyQA
yes/no) choose from a fixed set, so their answers cannot be off-topic and are not judged.

Systems: CAEM C0..C5, Phase 1d C3, baselines B1..B7. Wrong = em == 0 (TriviaQA) or em_llm_judged < 1
(TruthfulQA). Identical (question, response) pairs are judged once. Results are cached after every call
in outputs/analysis/off_topic_judged.json ({sha1: {"label": "ON"|"OFF", "reason": str}}), so the
script can be stopped and resumed. No experiment is re-run.

Run with ~/caem_reranker_env/bin/python (needs the anthropic package and ANTHROPIC_API_KEY).
"""
import hashlib
import json
import os
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/analysis/off_topic_judged.json"
MODEL = "claude-haiku-4-5"  # same judge as the TruthfulQA re-score
MIN_INTERVAL = 1.35  # ~45 requests per minute, under the 50/min tier limit
BL = ROOT / "outputs/baselines_phase1e"
OPEN = ("triviaqa", "truthfulqa")

PROMPT = """You are checking whether a model's response is ON_TOPIC or OFF_TOPIC for the question it was asked. You are NOT checking whether it is correct. Do not use your own knowledge of the right answer: a wrong answer is still ON_TOPIC if it is the right kind of answer about the right subject.

Question: {question}

Response: {response}

Rules, applied to the response's final answer:
- OFF_TOPIC if the answer is a different kind of thing than the question asks for: a place or year when a name is asked, a person when a website is asked, or a label such as "supports", "refutes" or "not enough info" when the question is not a true/false claim.
- OFF_TOPIC if the response is about a different subject than the question names (a different person, event, object or concept) and never answers about the subject actually asked.
- ON_TOPIC if the answer is the right kind of thing about the right subject, even if it is wrong, unlikely, partly wrong or contradicts facts you know.
- ON_TOPIC if the response says it does not know, or gives no answer.

Examples: "Gerald Ford's middle name?" -> "Lincoln" is ON_TOPIC (a wrong name). "Gerald Ford's middle name?" -> "He was born in Omaha. Answer: Ford" is OFF_TOPIC (birthplace reasoning, surname not middle name). "Which country's anthem is La Brabanconne?" -> "France" is ON_TOPIC (a wrong country). "What does ADIDAS stand for?" -> "Answer: refutes" is OFF_TOPIC (a label, not an expansion).

First line: ON_TOPIC or OFF_TOPIC. Second line: a reason in under 15 words."""


def files():
    for c in range(6):
        yield f"caem_c{c}", lambda b, c=c: ROOT / f"outputs/full_run/eval/{b}_cycle{c}.json"
    yield "phase1d_c3", lambda b: ROOT / f"outputs/phase1d_eval/{b}_cycle3.json"
    for n in ("zero_shot", "cot", "rag", "cot_rag", "fiveshot_cot", "flare"):
        yield n, lambda b, n=n: BL / n / f"{b}_cycle0.json"
    yield "vanilla_ft", lambda b: BL / "vanilla_ft/eval" / f"{b}_cycle5.json"


def wrong(s, b):
    if b == "truthfulqa":
        return (s.get("em_llm_judged") or 0) < 1
    return (s.get("em") or 0) < 1


def key(q, r):
    return hashlib.sha1((q + "\x00" + r).encode()).hexdigest()


def todo():
    seen = {}
    for name, path in files():
        for b in OPEN:
            rows = json.load(open(path(b)))
            rows = rows["samples"] if isinstance(rows, dict) else rows
            for s in rows:
                if wrong(s, b):
                    q, r = s["question"], (s.get("prediction") or "")
                    seen[key(q, r)] = (q, r)
    return seen


def main():
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    work = {k: v for k, v in todo().items() if k not in cache}
    print(f"{len(cache)} cached, {len(work)} to judge", flush=True)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=30.0, max_retries=0)
    last = 0.0
    for i, (k, (q, r)) in enumerate(work.items(), 1):
        for attempt in range(5):
            wait = MIN_INTERVAL - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            try:
                msg = client.messages.create(model=MODEL, max_tokens=60, messages=[
                    {"role": "user", "content": PROMPT.format(question=q, response=(r or "(empty)")[:1500])}])
                text = "".join(getattr(c, "text", "") for c in msg.content).strip()
                first = text.splitlines()[0].strip().upper() if text else ""
                label = "OFF" if first.startswith("OFF") else "ON" if first.startswith("ON") else None
                if label is None:
                    raise ValueError(f"unparseable: {text[:60]!r}")
                cache[k] = {"label": label, "reason": " ".join(text.splitlines()[1:])[:160]}
                break
            except anthropic.RateLimitError:
                time.sleep(60)
            except Exception as exc:  # network or parse error: back off and retry
                time.sleep(min(2 ** attempt, 16))
                if attempt == 4:
                    print("failed:", exc, flush=True)
        if i % 100 == 0 or i == len(work):
            json.dump(cache, open(CACHE, "w"))
            print(f"{i}/{len(work)}", flush=True)
    json.dump(cache, open(CACHE, "w"))


if __name__ == "__main__":
    main()
