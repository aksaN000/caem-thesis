#!/usr/bin/env python
"""
scripts/test_t1_routing_robustness.py
======================================
T1 routing robustness test (post-Phase-1a diagnostic).

Tests the user's intuition: T1 should fire for near-duplicate queries
but not for paraphrases. Validates that the routing logic behaves as
expected when stored memory is fed back with controlled tweaks.

Why this matters
----------------
In benchmark eval, T1 hit rate is naturally low because eval samples
are drawn from disjoint splits — they're rarely near-duplicates of
stored episodes. In production with repeat user queries, T1 would be
much higher. This test demonstrates T1 is FUNCTIONAL even when its
trigger conditions are rare in benchmark mode.

Methodology
-----------
1. Sample N stored episodes from outputs/full_run/cycle_10/memory_store
2. For each: generate K paraphrase variants of varying similarity
3. Feed each variant to CAEMPipeline.answer()
4. Record (variant_similarity, tier_fired, latency, decision)
5. Plot: similarity-vs-tier histogram + per-tier hit rate

Paraphrase strategies
---------------------
  - "exact":      original question (similarity ≈ 1.0)
  - "synonym":    replace 1-2 words with synonyms (~0.92-0.97)
  - "reorder":    rearrange clauses (~0.85-0.92)
  - "rephrase":   restate as a different question style (~0.75-0.85)
  - "different":  unrelated question on same topic (~0.50-0.75)

Expected results under default routing config
---------------------------------------------
  - exact:      T1 (similarity ≈ 1.0, routing_score > 0.9)
  - synonym:    T1 if u_stored > 0.85, else T2
  - reorder:    T2 (similarity 0.85-0.92)
  - rephrase:   T2 if sim > 0.75, else T3
  - different:  T3 (similarity < 0.75)

Output
------
  outputs/t1_routing_test/results.csv
  outputs/t1_routing_test/figure.pdf
  outputs/t1_routing_test/report.txt

Usage
-----
    python scripts/test_t1_routing_robustness.py \\
        --memory_store outputs/full_run/cycle_10/memory_store_cycle_10 \\
        --n_episodes 50 \\
        --output_dir outputs/t1_routing_test
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Simple paraphrase strategies (no LLM needed; uses templates + synonym swap)
_SYNONYM_MAP = {
    "what": "which", "who": "what person", "when": "at what time",
    "where": "in what place", "is": "happens to be", "are": "happen to be",
    "began": "started", "wrote": "authored", "made": "produced",
    "the": "a particular", "first": "initial", "released": "published",
    "answer": "respond", "claim": "assertion", "reasoning": "analysis",
}


def _exact(q: str) -> str:
    return q


def _synonym_swap(q: str, n_swaps: int = 2) -> str:
    words = q.split()
    swap_indices = [i for i, w in enumerate(words) if w.lower().strip(".,?!") in _SYNONYM_MAP]
    random.shuffle(swap_indices)
    for idx in swap_indices[:n_swaps]:
        original = words[idx]
        key = original.lower().strip(".,?!")
        if key in _SYNONYM_MAP:
            new_word = _SYNONYM_MAP[key]
            if original[0].isupper():
                new_word = new_word.capitalize()
            # Preserve trailing punctuation
            trailing = ""
            for c in reversed(original):
                if not c.isalnum():
                    trailing = c + trailing
                else:
                    break
            words[idx] = new_word + trailing
    return " ".join(words)


def _reorder(q: str) -> str:
    """Swap two adjacent clauses (separated by commas) if any; else swap two words."""
    if "," in q:
        clauses = [c.strip() for c in q.split(",", 1)]
        if len(clauses) == 2:
            return f"{clauses[1]}, {clauses[0]}"
    words = q.split()
    if len(words) >= 4:
        # Swap a middle pair
        mid = len(words) // 2
        words[mid], words[mid + 1] = words[mid + 1], words[mid]
    return " ".join(words)


def _rephrase(q: str) -> str:
    """Restate as a different question style (template-based)."""
    if q.lower().startswith("who"):
        return q.replace("Who", "Tell me the person who").replace("who", "tell me who")
    if q.lower().startswith("what"):
        return q.replace("What", "Can you tell me what").replace("what", "can you tell me what")
    if q.lower().startswith("when"):
        return q.replace("When", "At what time").replace("when", "at what time")
    return f"Could you tell me: {q}"


def _different_topic(q: str) -> str:
    """Generate an unrelated question (controls for topic drift baseline)."""
    return "What is the capital of an unrelated country?"


PARAPHRASE_STRATEGIES = [
    ("exact",     _exact,           "Original question (similarity ≈ 1.0)"),
    ("synonym",   _synonym_swap,    "1-2 synonym swaps (~0.92-0.97)"),
    ("reorder",   _reorder,         "Clause reordering (~0.85-0.92)"),
    ("rephrase",  _rephrase,        "Restate as different style (~0.75-0.85)"),
    ("different", _different_topic, "Unrelated question (~0.30-0.50)"),
]


def _load_stored_episodes(memory_path: Path, n: int, seed: int = 42) -> List[Dict[str, Any]]:
    import pickle
    meta_path = memory_path.parent / (memory_path.name + ".meta") \
        if not memory_path.suffix else memory_path
    if not meta_path.exists():
        # Try standard layout: <path>.meta
        meta_path = Path(str(memory_path) + ".meta")
    if not meta_path.exists():
        raise FileNotFoundError(f"Memory meta not found at {meta_path}")
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    md = data.get("metadata", {}) if isinstance(data, dict) else data
    entries = list(md.values()) if isinstance(md, dict) else list(md)
    rng = random.Random(seed)
    sampled = rng.sample(entries, min(n, len(entries)))
    out: List[Dict[str, Any]] = []
    for e in sampled:
        d = vars(e) if hasattr(e, "__dict__") else e
        if d.get("question"):
            out.append({
                "question": d["question"],
                "answer": d.get("answer", ""),
                "u_stored": float(d.get("u_stored", 0)),
                "source_benchmark": d.get("source_benchmark", "unknown"),
            })
    return out


def _build_pipeline(memory_path: Path):
    import torch
    from caem.config import CAEMConfig
    from caem.memory.encoder import QueryEncoder
    from caem.memory.store import EpisodicMemoryStore
    from caem.model_loader import load_base_generator
    from caem.pipeline import CAEMPipeline
    from caem.retrieval.rag import PassageStore
    from caem.verification import load_verifier_judge

    config = CAEMConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_base_generator(
        "Qwen/Qwen2.5-3B-Instruct", use_sdpa=True, use_torch_compile=False,
    )
    encoder = QueryEncoder(device=device)
    judge, nli_model, nli_tokenizer = load_verifier_judge(config, device, allow_fallback=True)
    cross_encoder = None
    if config.cross_encoder_model:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(
            config.cross_encoder_model, device=device,
            automodel_args={"torch_dtype": torch.bfloat16} if device != "cpu" else {},
        )
    memory_store = EpisodicMemoryStore(config=config)
    memory_store.load(str(memory_path))
    passage_store = None
    if Path("data/passage_index").exists():
        passage_store = PassageStore.load("data/passage_index")
    pipeline = CAEMPipeline(
        model=model, tokenizer=tokenizer, encoder=encoder,
        judge=judge, nli_model=nli_model, nli_tokenizer=nli_tokenizer,
        passage_store=passage_store, memory_store=memory_store,
        cross_encoder=cross_encoder, config=config,
    )
    return pipeline


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--memory_store", type=Path, required=True,
                   help="Path to memory store (without .meta suffix).")
    p.add_argument("--n_episodes", type=int, default=50,
                   help="Number of stored episodes to test (default 50).")
    p.add_argument("--output_dir", type=Path, default=Path("outputs/t1_routing_test"))
    p.add_argument("--seed", type=int, default=42)
    ns = p.parse_args()

    ns.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading stored episodes from %s ...", ns.memory_store)
    episodes = _load_stored_episodes(ns.memory_store, ns.n_episodes, ns.seed)
    logger.info("Loaded %d stored episodes for tweak-and-test", len(episodes))

    logger.info("Building pipeline (loads Qwen + verifier + rerankers)...")
    pipeline = _build_pipeline(ns.memory_store)

    results: List[Dict[str, Any]] = []
    for i, ep in enumerate(episodes):
        for strat_name, fn, desc in PARAPHRASE_STRATEGIES:
            variant_q = fn(ep["question"])
            try:
                result = pipeline.answer(variant_q, store_to_memory=False)
            except Exception as exc:
                logger.warning("pipeline.answer raised on %s/%s: %s", i, strat_name, exc)
                continue
            row = {
                "episode_idx": i,
                "source_benchmark": ep["source_benchmark"],
                "stored_u_stored": ep["u_stored"],
                "strategy": strat_name,
                "original_question": ep["question"][:100],
                "variant_question": variant_q[:100],
                "tier_fired": result.tier,
                "latency_ms": result.latency_ms,
                "decision": (result.verifier_output.decision
                             if getattr(result, "verifier_output", None) else None),
                "u_stored": (result.verifier_output.u_stored
                             if getattr(result, "verifier_output", None) else None),
            }
            results.append(row)
        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d episodes", i + 1, len(episodes))

    # Write CSV
    csv_path = ns.output_dir / "results.csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow(r)
        logger.info("Wrote %s", csv_path)

    # Aggregate report
    from collections import Counter, defaultdict
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    report_lines = ["T1 routing robustness test — summary", "=" * 60]
    report_lines.append(f"{'Strategy':<12} {'n':<5} {'%T1':<6} {'%T2':<6} {'%T3':<6}")
    for strat_name, _, desc in PARAPHRASE_STRATEGIES:
        rs = by_strategy.get(strat_name, [])
        if not rs:
            continue
        tiers = Counter(r["tier_fired"] for r in rs)
        n = len(rs)
        t1_pct = 100 * tiers.get(1, 0) / n
        t2_pct = 100 * tiers.get(2, 0) / n
        t3_pct = 100 * tiers.get(3, 0) / n
        report_lines.append(f"{strat_name:<12} {n:<5} {t1_pct:<6.1f} {t2_pct:<6.1f} {t3_pct:<6.1f}")
    report_lines.append("")
    report_lines.append("Interpretation:")
    report_lines.append("  - 'exact' should fire T1 ~100% (validates near-duplicate routing)")
    report_lines.append("  - 'synonym' should mostly fire T1 (u_stored permitting)")
    report_lines.append("  - 'reorder' should mostly fire T2 (similarity in 0.85-0.92 range)")
    report_lines.append("  - 'rephrase' should mostly fire T2 or T3 (similarity ~0.75-0.85)")
    report_lines.append("  - 'different' should fire T3 (similarity below T2 threshold)")
    report_lines.append("")
    report_lines.append("If 'exact' fires T2/T3 → routing logic broken")
    report_lines.append("If 'different' fires T1 → similarity threshold too lax")
    report_lines.append("Otherwise → routing is functional as designed.")

    report_path = ns.output_dir / "report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    logger.info("Wrote %s", report_path)

    # Optional figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        strategies = [s for s, _, _ in PARAPHRASE_STRATEGIES if by_strategy.get(s)]
        x = np.arange(len(strategies))
        width = 0.25
        t1 = [100 * sum(1 for r in by_strategy[s] if r["tier_fired"] == 1) / len(by_strategy[s])
              for s in strategies]
        t2 = [100 * sum(1 for r in by_strategy[s] if r["tier_fired"] == 2) / len(by_strategy[s])
              for s in strategies]
        t3 = [100 * sum(1 for r in by_strategy[s] if r["tier_fired"] == 3) / len(by_strategy[s])
              for s in strategies]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width, t1, width, label="Tier 1", color="#1b9e77")
        ax.bar(x,         t2, width, label="Tier 2", color="#d95f02")
        ax.bar(x + width, t3, width, label="Tier 3", color="#7570b3")
        ax.set_xticks(x); ax.set_xticklabels(strategies)
        ax.set_ylabel("% of variants routed to tier")
        ax.set_title("T1 routing sensitivity to paraphrase variation")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig_path = ns.output_dir / "figure.pdf"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Wrote %s", fig_path)
    except ImportError:
        logger.info("matplotlib unavailable; skipping figure")

    return 0


if __name__ == "__main__":
    sys.exit(main())
