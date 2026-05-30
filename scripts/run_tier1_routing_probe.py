"""
scripts/run_tier1_routing_probe.py
====================================
Tier-1 routing probe — captures the FULL routing decision for each query
WITHOUT running Tier 2 or Tier 3 generation. Sub-second per query.

Why this exists (vs run_tier1_recall_diagnostic.py):
- The legacy script calls pipeline.answer() which runs T2/T3 generation
  and discards routing internals. Slow (15-30s/query) and uninspectable.
- This script calls the routing components directly (encode, pre_conf,
  memory_search, router.route) and captures EVERY routing-decision field
  + top-5 retrieved entries + ground-truth comparison.

Usage:
    python -m scripts.run_tier1_routing_probe \
        --full_run_dir outputs/full_run \
        --cycle 5 \
        --queries_path outputs/tier1_routing_probe/queries_cycle5.json \
        --output_dir outputs/tier1_routing_probe
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tier1_routing_probe")


def _build_pipeline(full_run_dir: Path, cycle: int):
    """Build the CAEM pipeline matching the production state at given cycle.

    Loads base model, applies cycle-N LoRA adapter, loads cycle-N composite,
    cycle-N memory store, RUC, encoder, passage store.
    """
    from caem.config import CAEMConfig
    from caem.pipeline import CAEMPipeline
    from caem.model_loader import load_base_generator
    from caem.memory.encoder import QueryEncoder
    from caem.memory.store import EpisodicMemoryStore
    from caem.retrieval.rag import PassageStore
    from peft import PeftModel

    cfg = CAEMConfig()
    cfg.use_lora_training = True

    # Cycle-N composite (matches the model state we test against)
    cycle_composite = full_run_dir / f"cycle_{cycle}/composite_calibration.json"
    if cycle_composite.is_file():
        cfg.composite_calibration_path = str(cycle_composite)
        logger.info("Using cycle-%d composite: %s", cycle, cycle_composite)

    logger.info("Loading base model + cycle-%d adapter...", cycle)
    model, tok = load_base_generator(cfg.base_model_name, device="cuda")
    adapter_dir = full_run_dir / f"cycle_{cycle}/adapter"
    if adapter_dir.is_dir():
        logger.info("Applying cycle-%d adapter from %s", cycle, adapter_dir)
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    encoder = QueryEncoder()

    passage_store = None
    pi_dir = Path("/workspace/caem/data/passage_index")
    if pi_dir.is_dir():
        logger.info("Loading PassageStore from %s...", pi_dir)
        passage_store = PassageStore.load(str(pi_dir))

    pipeline = CAEMPipeline(
        model=model, tokenizer=tok, encoder=encoder,
        passage_store=passage_store, config=cfg, device="cuda",
    )

    # Load cycle-N memory store into the pipeline
    mem = EpisodicMemoryStore.load(str(full_run_dir / f"memory_store_cycle_{cycle}"))
    pipeline.memory_store = mem
    pipeline.memory = mem
    n_episodes = mem.size if isinstance(mem.size, int) else mem.size()
    logger.info("Loaded memory store: %d episodes", n_episodes)

    return pipeline


def probe_one(pipeline, q: Dict, source_question_text: Optional[str] = None) -> Dict:
    """Probe one query, capturing full routing decision fields."""
    question = q["question"]
    source_benchmark = q.get("source_benchmark")

    t0 = time.perf_counter()

    # Stage 2: encode
    query_embedding = pipeline._encode_query(question)
    t_encode = (time.perf_counter() - t0) * 1000

    # Stage 3a: pre-routing confidence (requires LLM forward pass)
    t1 = time.perf_counter()
    pre_conf = pipeline.pre_estimator.estimate(question, source_benchmark=source_benchmark)
    t_pre = (time.perf_counter() - t1) * 1000

    # Stage 1: top-5 memory search (vs the production top-1 — we want diagnostics)
    t2 = time.perf_counter()
    search_top5 = pipeline.memory_store.search_with_ids(query_embedding, k=5)
    t_search = (time.perf_counter() - t2) * 1000

    # Stage 3b: compute RUC features + run the actual router with TOP-1
    t3 = time.perf_counter()
    search_top1 = search_top5[:1] if search_top5 else []
    ruc_extras = None
    if pipeline.router.ruc is not None:
        try:
            ruc_extras = pipeline._compute_ruc_features(question, query_embedding)
        except Exception as e:
            logger.warning("RUC feature compute failed: %s", e)

    routing = pipeline.router.route(
        pre_conf, search_top1,
        source_benchmark=source_benchmark,
        question=question,
        top1_passage_text=(ruc_extras or {}).get("_top1_passage_text"),
        top1_passage_sim=(ruc_extras or {}).get("top1_passage_sim"),
        top1_passage_entity_overlap=(ruc_extras or {}).get("top1_passage_entity_overlap"),
        p_ik=(ruc_extras or {}).get("p_ik"),
        extra_ruc_features=ruc_extras,
    )
    t_route = (time.perf_counter() - t3) * 1000

    # Inspect the top-1 retrieved entry
    if search_top5:
        top1_entry, top1_id, top1_sim = search_top5[0]
        retrieved_question = top1_entry.question
        retrieved_u_stored = top1_entry.u_stored
        retrieved_bench = top1_entry.source_benchmark
        retrieved_answer = top1_entry.answer
        exact_text_match = (retrieved_question == question) or (
            source_question_text is not None and retrieved_question == source_question_text
        )
    else:
        top1_id, top1_sim = None, None
        retrieved_question, retrieved_u_stored, retrieved_bench = None, None, None
        retrieved_answer = None
        exact_text_match = False

    top5_similarities = [float(s[2]) for s in search_top5]
    top5_entry_ids = [s[1] for s in search_top5]

    return {
        # Query metadata
        "kind": q.get("kind"),
        "source_episode_id": q.get("source_episode_id"),
        "question": question,
        "source_question_text": source_question_text,
        "source_benchmark": source_benchmark,
        "source_u_stored_at_sample": q.get("u_stored_at_sample"),
        # Routing decision (the headline)
        "tier": routing.tier,
        "similarity": float(routing.similarity),
        "u_stored_retrieved": float(routing.u_stored_retrieved),
        "u_pre": float(routing.u_pre),
        "u_token": float(pre_conf.u_token),
        "c_conv": float(pre_conf.c_conv),
        "routing_score": float(routing.routing_score),
        "safety_override": bool(routing.safety_override),
        "retrieved_entry_id": routing.retrieved_entry_id,
        # RUC output (if consulted)
        "ruc_decision": (routing.ruc_output or {}).get("decision") if routing.ruc_output else None,
        "ruc_p_rag": (routing.ruc_output or {}).get("p_rag") if routing.ruc_output else None,
        # FAISS retrieval inspection
        "top5_similarities": top5_similarities,
        "top5_entry_ids": top5_entry_ids,
        "retrieved_question": retrieved_question,
        "retrieved_u_stored_now": float(retrieved_u_stored) if retrieved_u_stored is not None else None,
        "retrieved_bench": retrieved_bench,
        "retrieved_answer": retrieved_answer,
        "exact_text_match": exact_text_match,
        # Latency breakdown
        "latency_ms": {
            "encode": round(t_encode, 1),
            "pre_conf": round(t_pre, 1),
            "search_top5": round(t_search, 1),
            "route": round(t_route, 1),
            "total": round(t_encode + t_pre + t_search + t_route, 1),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full_run_dir", type=Path, default=Path("outputs/full_run"))
    p.add_argument("--cycle", type=int, default=5)
    p.add_argument("--queries_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    ns = p.parse_args()

    ns.output_dir.mkdir(parents=True, exist_ok=True)
    queries = json.loads(ns.queries_path.read_text())
    logger.info("Loaded %d queries from %s", len(queries), ns.queries_path)

    pipeline = _build_pipeline(ns.full_run_dir, ns.cycle)

    results: List[Dict] = []
    logger.info("Probing %d queries serial (bs=1) ...", len(queries))
    for i, q in enumerate(queries):
        # If the query has 'original_question' (paraphrase case), pass it as source-text
        # for the exact_text_match check against the retrieved entry.
        source_text = q.get("original_question") or q.get("source_question_text")
        try:
            res = probe_one(pipeline, q, source_question_text=source_text)
        except Exception as exc:
            logger.error("query %d failed: %s", i, exc)
            res = {**q, "error": str(exc)}
        results.append(res)
        if (i + 1) % 10 == 0:
            logger.info("  ran %d/%d (last total latency=%.0f ms, tier=%s)",
                        i + 1, len(queries),
                        res.get("latency_ms", {}).get("total", -1),
                        res.get("tier"))

    out_results = ns.output_dir / f"results_cycle{ns.cycle}.json"
    out_results.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_results)

    # Summary aggregation
    from collections import Counter
    by_kind_tier = {}
    routing_score_by_kind = {}
    similarity_by_kind = {}
    u_pre_by_kind = {}
    safety_by_kind = {}
    top1_is_source = {}
    for r in results:
        if "error" in r:
            continue
        k = r["kind"]
        by_kind_tier.setdefault(k, Counter())
        by_kind_tier[k][r["tier"]] += 1
        routing_score_by_kind.setdefault(k, []).append(r["routing_score"])
        similarity_by_kind.setdefault(k, []).append(r["similarity"])
        u_pre_by_kind.setdefault(k, []).append(r["u_pre"])
        safety_by_kind.setdefault(k, []).append(1 if r["safety_override"] else 0)
        if r.get("source_episode_id") is not None and r.get("retrieved_entry_id") is not None:
            top1_is_source.setdefault(k, []).append(
                1 if r["retrieved_entry_id"] == r["source_episode_id"] else 0
            )

    def _stats(xs):
        if not xs:
            return None
        n = len(xs)
        return {
            "n": n,
            "mean": round(sum(xs) / n, 4),
            "min": round(min(xs), 4),
            "max": round(max(xs), 4),
        }

    summary = {
        "n_total": len(results),
        "by_kind_tier": {k: dict(v) for k, v in by_kind_tier.items()},
        "routing_score": {k: _stats(v) for k, v in routing_score_by_kind.items()},
        "similarity": {k: _stats(v) for k, v in similarity_by_kind.items()},
        "u_pre": {k: _stats(v) for k, v in u_pre_by_kind.items()},
        "safety_override_rate": {k: round(sum(v) / len(v), 3) if v else None
                                  for k, v in safety_by_kind.items()},
        "top1_is_source_rate": {k: round(sum(v) / len(v), 3) if v else None
                                 for k, v in top1_is_source.items()},
    }

    out_summary = ns.output_dir / f"summary_cycle{ns.cycle}.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", out_summary)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
