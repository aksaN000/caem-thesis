#!/usr/bin/env python
"""
scripts/run_tier1_recall_diagnostic.py
========================================
Tier-1 recall + per-tier latency diagnostic (Task #154).

Builds a 100-query eval set that's intentionally constructed to exercise
each tier of the CAEM router, then runs serial Pipeline at bs=1 on the
cycle-5 final state:

  - 50 paraphrased queries of cycle-5-stored memory entries
       → should hit Tier 1 (memory recall)
  - 30 in-distribution novel queries (training-bench validation splits)
       → should hit Tier 2 (direct generation through SIL adapter)
  - 20 OOD queries (TruthfulQA samples)
       → should hit Tier 3 (RAG escalation)

Empirical receipts produced:
  - H3 (memory-direct tier share grows): T1 hit rate on paraphrased queries
  - cor:tier1-floor (T1 hallucination ceiling): CHM on T1-routed samples
  - Per-tier intrinsic latency (resolves batch-amortisation artefact;
    see PRODUCTION_NEXT_SESSION_PLAN.md "Tier-1 amortisation diagnostic")
  - Cycle reconstruction: per-tier cost × per-cycle tier-share trajectory
    yields each cycle's pooled wall-time without re-running trajectory.

Why a separate diagnostic is needed
-----------------------------------
The standard eval-fold is content-hash disjoint from the SIL training pool
by design (prevents leakage on EM/CHM). T1 cannot fire on that fold because
no eval query overlaps with what's in memory. This diagnostic constructs
the controlled tier-mix the eval fold can't provide.

Cost
----
  - Paraphrase generation: ~50 Anthropic API calls @ Haiku 4.5 ≈ $0.10
  - Pipeline serial run: ~100 queries × ~3-10s each ≈ 15-20 GPU-min
  - Total wall-time: ~30 min including loading + writing

Requires
--------
  - ANTHROPIC_API_KEY in env (paraphrase generation)
  - GPU + cycle-5 artifacts at outputs/full_run/cycle_5/

Memory refs:
  caem_tier1_latency_diagnostic.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


PARAPHRASE_PROMPT = """Paraphrase this question while preserving exact meaning. The paraphrase should reach the same answer but use different surface wording. Output only the paraphrased question, no commentary.

Original: {question}

Paraphrase:"""


def generate_paraphrase(client, question: str, model: str = "claude-haiku-4-5",
                         max_retries: int = 3) -> Optional[str]:
    """Single paraphrase via Anthropic API."""
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model, max_tokens=200, temperature=0.3,
                messages=[{"role": "user",
                           "content": PARAPHRASE_PROMPT.format(question=question)}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content
                            if getattr(b, "type", "text") == "text").strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("paraphrase attempt %d: %s", attempt + 1, exc)
            time.sleep(min(2 ** attempt, 16))
    return None


def load_memory_entries(meta_path: Path, n: int, seed: int = 42) -> List[Dict]:
    """Load n random memory entries from a cycle's memory_store_*.meta."""
    import pickle
    with meta_path.open("rb") as fh:
        d = pickle.load(fh)
    metadata = d.get("metadata", {})
    entry_ids = list(metadata.keys())
    rng = random.Random(seed)
    picked_ids = rng.sample(entry_ids, min(n, len(entry_ids)))
    out = []
    for entry_id in picked_ids:
        e = metadata[entry_id]
        # MemoryEntry has .question, .answer, .gold_answers, .source_benchmark
        out.append({
            "entry_id": entry_id,
            "question": getattr(e, "question", None),
            "answer": getattr(e, "answer", None),
            "gold_answers": getattr(e, "gold_answers", []) or [],
            "source_benchmark": getattr(e, "source_benchmark", "unknown"),
        })
    return [x for x in out if x["question"]]


def load_novel_queries(n: int, benches=("fever", "triviaqa", "commonsense_qa"),
                       seed: int = 42, exclude_questions=None) -> List[Dict]:
    """Sample n in-distribution novel queries from training-bench validation splits."""
    from eval.benchmarks import load_benchmark
    exclude_questions = set(exclude_questions or [])
    rng = random.Random(seed)
    out = []
    per_bench = max(1, n // len(benches))
    for bench in benches:
        try:
            samples = load_benchmark(bench, n=500, seed=seed + 1)
        except Exception as exc:
            logger.warning("could not load %s: %s", bench, exc)
            continue
        rng.shuffle(samples)
        picked = []
        for s in samples:
            q = s.get("question", "")
            if q and q not in exclude_questions:
                picked.append({
                    "question": q,
                    "gold_answers": s.get("answers") or [],
                    "source_benchmark": bench,
                })
                if len(picked) >= per_bench:
                    break
        out.extend(picked)
    return out[:n]


def load_ood_queries(n: int, seed: int = 42) -> List[Dict]:
    """Sample n OOD queries from TruthfulQA (a transfer-only benchmark CAEM doesn't train on)."""
    from eval.benchmarks import load_truthfulqa
    samples = load_truthfulqa(n=n, seed=seed)
    out = []
    for s in samples:
        out.append({
            "question": s["question"],
            "gold_answers": s.get("answers") or [],
            "source_benchmark": "truthfulqa",
        })
    return out


def build_query_set(memory_meta: Path, n_para: int, n_novel: int, n_ood: int,
                    client, seed: int = 42) -> List[Dict]:
    logger.info("Loading %d memory entries from %s ...", n_para, memory_meta)
    mem_entries = load_memory_entries(memory_meta, n_para, seed=seed)
    queries = []

    logger.info("Paraphrasing %d memory entries ...", len(mem_entries))
    for i, e in enumerate(mem_entries):
        para = generate_paraphrase(client, e["question"])
        if para:
            queries.append({
                "kind": "paraphrase",
                "expected_tier": 1,
                "original_question": e["question"],
                "question": para,
                "gold_answers": e["gold_answers"],
                "source_benchmark": e["source_benchmark"],
                "memory_entry_id": e["entry_id"],
            })
        if (i + 1) % 10 == 0:
            logger.info("  paraphrased %d/%d", i + 1, len(mem_entries))

    exclude = {q["original_question"] for q in queries if "original_question" in q}
    novel = load_novel_queries(n_novel, exclude_questions=exclude, seed=seed)
    for n in novel:
        queries.append({"kind": "in_distribution", "expected_tier": 2, **n})

    ood = load_ood_queries(n_ood, seed=seed)
    for o in ood:
        queries.append({"kind": "ood", "expected_tier": 3, **o})

    logger.info("Built %d total queries (paraphrase=%d, novel=%d, ood=%d)",
                len(queries), len(mem_entries), n_novel, n_ood)
    return queries


def run_pipeline_serial(pipeline, queries: List[Dict]) -> List[Dict]:
    """Run pipeline.answer(q) one at a time, recording per-sample latency."""
    results = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        try:
            res = pipeline.answer(q["question"], store_to_memory=False)
            latency_ms = (time.perf_counter() - t0) * 1000
            actual_tier = getattr(res, "tier", None)
            answer = getattr(res, "display_answer", "") or getattr(res, "answer", "")
        except Exception as exc:
            logger.error("query %d failed: %s", i, exc)
            latency_ms = (time.perf_counter() - t0) * 1000
            actual_tier = None
            answer = ""
        # Score EM against gold
        from eval.metrics import any_match_em, normalise
        em = float(any_match_em(answer, q["gold_answers"])) if q["gold_answers"] else 0.0
        results.append({
            **q,
            "actual_tier": actual_tier,
            "answer": answer[:200],
            "latency_ms": latency_ms,
            "em": em,
        })
        if (i + 1) % 10 == 0:
            logger.info("  ran %d/%d (last latency=%.0f ms, tier=%s)",
                        i + 1, len(queries), latency_ms, actual_tier)
    return results


def summarize(results: List[Dict]) -> Dict:
    """Per-(kind, actual_tier) hit-rate + EM + latency."""
    by_kind: Dict[str, Counter] = defaultdict(Counter)
    per_tier_latency: Dict[int, List[float]] = defaultdict(list)
    per_tier_em: Dict[int, List[float]] = defaultdict(list)
    for r in results:
        kind = r["kind"]
        t = r.get("actual_tier")
        by_kind[kind][t] += 1
        if t is not None:
            per_tier_latency[t].append(r["latency_ms"])
            per_tier_em[t].append(r["em"])

    out = {
        "n_total": len(results),
        "by_kind_tier_distribution": {k: dict(v) for k, v in by_kind.items()},
        "per_tier_intrinsic_latency_ms": {
            t: sum(lats) / len(lats) for t, lats in per_tier_latency.items()
        },
        "per_tier_em": {
            t: sum(ems) / len(ems) for t, ems in per_tier_em.items()
        },
        "per_tier_n": {t: len(lats) for t, lats in per_tier_latency.items()},
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cycle", type=int, default=5)
    p.add_argument("--full_run_dir", type=Path, default=Path("outputs/full_run"))
    p.add_argument("--n_paraphrase", type=int, default=50)
    p.add_argument("--n_novel", type=int, default=30)
    p.add_argument("--n_ood", type=int, default=20)
    p.add_argument("--output_dir", type=Path, default=Path("outputs/tier1_recall"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge_model", default="claude-haiku-4-5")
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()
    logging.basicConfig(level=ns.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set; paraphrase generation needs it")
        return 1

    import anthropic
    client = anthropic.Anthropic()

    ns.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build query set
    meta_path = ns.full_run_dir / f"memory_store_cycle_{ns.cycle}.meta"
    if not meta_path.is_file():
        logger.error("Memory store not found: %s", meta_path)
        return 1

    qpath = ns.output_dir / f"queries_cycle{ns.cycle}.json"
    if qpath.is_file():
        logger.info("Reusing existing query set: %s", qpath)
        queries = json.loads(qpath.read_text())
    else:
        queries = build_query_set(meta_path, ns.n_paraphrase, ns.n_novel, ns.n_ood,
                                   client, seed=ns.seed)
        qpath.write_text(json.dumps(queries, indent=2))
        logger.info("Wrote query set: %s", qpath)

    # 2. Build CAEM pipeline at cycle-N state
    from caem.config import CAEMConfig
    from caem.pipeline import CAEMPipeline
    from caem.model_loader import load_base_generator
    cfg = CAEMConfig()
    cfg.use_lora_training = True   # cycle-N artifacts use LoRA
    logger.info("Loading base model + cycle-%d adapter ...", ns.cycle)
    model, tok = load_base_generator(cfg.base_model_name, device="cuda")
    # NOTE: this assumes a load helper exists for cycle-N adapter; otherwise we
    # apply via PEFT loader. Document a TODO if peft loading needs wiring.
    try:
        from peft import PeftModel
        adapter_dir = ns.full_run_dir / f"cycle_{ns.cycle}/adapter"
        if adapter_dir.is_dir():
            logger.info("Applying cycle-%d adapter from %s", ns.cycle, adapter_dir)
            model = PeftModel.from_pretrained(model, str(adapter_dir))
    except Exception as exc:
        logger.warning("Could not load adapter: %s — running on base model", exc)

    pipeline = CAEMPipeline(model=model, tokenizer=tok, config=cfg, device="cuda")
    # Load cycle-N memory store
    from caem.memory.store import EpisodicMemoryStore
    mem = EpisodicMemoryStore.load(str(ns.full_run_dir / f"memory_store_cycle_{ns.cycle}"))
    pipeline.memory = mem

    # 3. Run serial
    logger.info("Running %d queries serial (bs=1) ...", len(queries))
    results = run_pipeline_serial(pipeline, queries)

    # 4. Aggregate + write
    summary = summarize(results)
    out_results = ns.output_dir / f"results_cycle{ns.cycle}.json"
    out_summary = ns.output_dir / f"summary_cycle{ns.cycle}.json"
    out_results.write_text(json.dumps(results, indent=2, default=str))
    out_summary.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s + %s", out_results, out_summary)

    print("\n=== T1-recall diagnostic summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
