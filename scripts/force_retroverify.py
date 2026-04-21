"""
scripts/force_retroverify.py
============================
Between-cycle fast path for Goal 4 item 5 — re-score only the entries whose
``hit_counter`` has crossed ``CAEMConfig.hit_counter_force_retroverify``.

Motivation
----------
``SelfImprovementLoop.run_cycle`` calls ``memory_store.retroverify(verify_fn)``
which re-scores **every** stored entry at every cycle boundary. That is
expensive (O(N_stored × verifier latency)) and catches popular-but-wrong
entries only at the next boundary. The hit-counter queue enables a
cheaper between-cycle pass: a small N (typically ~1-5 % of the store) that
has accumulated Tier-1 serve pressure above the threshold, re-scored
out-of-band.

This script is the standalone caller. Invoke it between cycles (e.g. via a
cron job or mid-experiment manual trigger) with a saved memory-store
snapshot. It loads the store, constructs a verifier, fetches the queue,
re-scores those entries, and saves back.

Typical usage
-------------
    PYTHONPATH=. python scripts/force_retroverify.py \\
        --memory_store outputs/memory_store_cycle_3 \\
        --passage_index data/passage_index \\
        --output_dir outputs/memory_store_cycle_3_fq \\
        --hit_threshold 10

The saved output is a drop-in replacement for the input snapshot; the main
run picks it up at the next cycle if the paths are wired through
``--cold_start_memory`` or similar.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hit-counter between-cycle forced re-verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--memory_store", type=Path, required=True,
        help="Path to a saved EpisodicMemoryStore snapshot (base path, "
             "no .faiss suffix -- matches store.save() output).",
    )
    p.add_argument(
        "--passage_index", type=Path, required=True,
        help="Path to the 21M Wikipedia passage index (for grounding signals).",
    )
    p.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory to write the re-scored memory-store snapshot.",
    )
    p.add_argument(
        "--hit_threshold", type=int, default=None,
        help="Override for cfg.hit_counter_force_retroverify. "
             "Default: read from CAEMConfig (10).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run", action="store_true",
                   help="Compute the queue + print its size; do NOT re-score.")
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ns = _build_parser().parse_args(argv)

    # Lazy imports so --help works without the model deps.
    import torch
    from caem.config import CAEMConfig
    from caem.memory.store import EpisodicMemoryStore
    from caem.model_loader import load_base_generator
    from caem.pipeline import CAEMPipeline
    from caem.retrieval.rag import PassageStore

    cfg = CAEMConfig()
    store = EpisodicMemoryStore(config=cfg)
    store.load(str(ns.memory_store))
    logger.info("Loaded memory store (%d episodes) from %s",
                store.size, ns.memory_store)

    queue = store.force_retroverify_queue(hit_threshold=ns.hit_threshold)
    logger.info("Forced-queue size: %d / %d total episodes "
                "(threshold=%s)", len(queue), store.size,
                ns.hit_threshold if ns.hit_threshold is not None
                else cfg.hit_counter_force_retroverify)

    if not queue:
        logger.info("No entries crossed the threshold -- nothing to do.")
        return 0

    if ns.dry_run:
        logger.info("--dry_run: skipping re-score. Top-5 queued entry IDs: %s",
                    queue[:5])
        return 0

    # Load the full CAEM stack + attach the pre-loaded store to the pipeline.
    model, tokenizer = load_base_generator(
        cfg.base_model_name, device=ns.device, dtype=torch.bfloat16,
        use_flash_attention_2=cfg.use_flash_attention_2,
        use_torch_compile=cfg.use_torch_compile,
    )
    from caem.memory.encoder import QueryEncoder
    encoder = QueryEncoder(model_name=cfg.sbert_model, device=ns.device)

    from caem.verification import load_verifier_judge
    judge, nli_model, nli_tokenizer = load_verifier_judge(
        cfg, ns.device, allow_fallback=False,
    )

    passage_store = PassageStore.load(str(ns.passage_index))

    cross_encoder = None
    if cfg.cross_encoder_model:
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder(cfg.cross_encoder_model, device=ns.device)
        except Exception as exc:
            logger.warning("cross-encoder load failed (%s); rerank + "
                           "q_a_relevance degrade to defaults.", exc)

    pipeline = CAEMPipeline(
        model=model, tokenizer=tokenizer, encoder=encoder,
        judge=judge, nli_model=nli_model, nli_tokenizer=nli_tokenizer,
        passage_store=passage_store, cross_encoder=cross_encoder,
        config=cfg, memory_store=store, device=ns.device, current_cycle=0,
    )
    verify_fn = pipeline.make_retroverify_fn()

    # Re-score exactly the queued entries (not the whole store). For each,
    # if the new u_stored is below the prune threshold the entry gets
    # removed; otherwise the nine-signal block is refreshed + hit_counter
    # resets to 0. Same semantics as the full retroverify path.
    n_updated = 0
    n_removed = 0
    for eid in queue:
        entry = store.get(eid)
        if entry is None:
            continue
        try:
            new_scores = verify_fn(entry)
        except Exception as exc:
            logger.warning("verify_fn raised for entry %d (%s) -- skipping.",
                           eid, exc)
            continue
        new_u = new_scores.u_stored
        if new_u < cfg.retroverify_prune_threshold:
            store.remove(eid)
            n_removed += 1
            continue
        store.update_u_stored(eid, new_u)
        entry.u_token = new_scores.u_token
        entry.u_dropout = new_scores.u_dropout
        entry.u_internal = new_scores.u_internal
        entry.s_avg = new_scores.s_avg
        entry.h_norm = new_scores.h_norm
        entry.p_entail = new_scores.p_entail
        entry.p_ground_max = new_scores.p_ground_max
        entry.p_ground_mean = new_scores.p_ground_mean
        entry.p_ground_atomic = new_scores.p_ground_atomic
        entry.p_contra = new_scores.p_contra
        entry.q_a_relevance = new_scores.q_a_relevance
        entry.decision = new_scores.decision
        entry.early_exit_triggered = new_scores.early_exit_triggered
        entry.retroverified = True
        entry.hit_counter = 0
        n_updated += 1

    logger.info("Force-retroverify complete: %d updated, %d removed. "
                "Store size: %d", n_updated, n_removed, store.size)

    ns.output_dir.mkdir(parents=True, exist_ok=True)
    out_base = ns.output_dir / ns.memory_store.name
    store.save(str(out_base))
    (ns.output_dir / "force_retroverify_summary.json").write_text(
        json.dumps({
            "queue_size": len(queue),
            "updated": n_updated,
            "removed": n_removed,
            "final_store_size": store.size,
            "hit_threshold": ns.hit_threshold or cfg.hit_counter_force_retroverify,
        }, indent=2),
    )
    logger.info("Saved to %s", out_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
