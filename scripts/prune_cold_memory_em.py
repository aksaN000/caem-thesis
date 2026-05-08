"""
scripts/prune_cold_memory_em.py
================================
Filter cold-start episodic memory by exact-match (EM) against gold answers.

The cold-start seeder (``scripts/seed_cold_start.py``) admits an episode
into memory when the verifier composite ``u_stored`` clears the cold-start
threshold (``--cold_start_store_threshold``, currently 0.50). The
verifier scores answers without seeing the gold label — by design, since
the production-mode pipeline has no oracle. As a result the cold-start
pool can contain confidently-wrong episodes: the model produces an
answer the verifier accepts, but the answer itself is exact-match wrong
against the training-split gold.

This script removes those confidently-wrong entries before the SIL
trajectory begins, so:
  - Tier 1 memory hits cannot return a known-wrong stored answer.
  - The SIL training pool never trains the model to reproduce a wrong
    answer that originated in cold memory.
  - The cycle-0 calibration fold is fitted against a memory whose EM
    distribution is post-pruning, matching what cycles 1+ will see.

Filtering rule
--------------
For each entry e:
  1. Look up the gold answer list for ``e.source_benchmark`` and
     ``e.question`` from the canonical training-split loader
     (``eval/benchmarks.py``). The cold-start seeder feeds bare loader
     samples, so ``e.question`` is byte-identical to the loader's
     ``question`` field — a direct dict lookup suffices.
  2. Compute ``em = any_match_em(e.answer, gold_list)`` using the
     canonical normaliser at ``eval/metrics.py``.
  3. Keep e if em == 1.0; drop e if em == 0.0; keep with
     em_unverified flag if no gold lookup was found.

Outputs
-------
  outputs/cold_start_memory/memory_store.{faiss,meta}    -- pruned store
  outputs/cold_start_memory/memory_store.pre_em_prune.{faiss,meta}.bak
                                                          -- safety backup
  outputs/cold_start_memory/em_pruning_report.json       -- audit trail
  outputs/cold_start_memory/seed_summary.json            -- updated total_seeded

Usage
-----
  python scripts/prune_cold_memory_em.py
  python scripts/prune_cold_memory_em.py --memory_dir outputs/cold_start_memory
  python scripts/prune_cold_memory_em.py --dry_run   # report only, no writes
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Repo root on PYTHONPATH so caem.* + eval.* import cleanly.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("prune_cold_memory_em")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _build_gold_lookup(benchmark: str) -> Dict[str, List[str]]:
    """Return a ``{question_text: gold_answers_list}`` map for a benchmark.

    Uses the same loader the cold-start seeder uses so that
    ``entry.question`` (a verbatim copy of the loader's ``question``)
    keys directly into the lookup.
    """
    from eval.benchmarks import (  # type: ignore
        load_fever, load_triviaqa, load_natural_questions,
        load_hotpotqa, load_commonsense_qa,
    )

    loaders = {
        "fever":             lambda: load_fever(split="train"),
        "triviaqa":          lambda: load_triviaqa(split="train"),
        "natural_questions": lambda: load_natural_questions(split="train"),
        "hotpotqa":          lambda: load_hotpotqa(split="train"),
        "commonsense_qa":    lambda: load_commonsense_qa(split="train"),
    }
    if benchmark not in loaders:
        logger.warning("No loader registered for benchmark=%r; skipping gold lookup", benchmark)
        return {}

    logger.info("Loading gold answers for benchmark=%s ...", benchmark)
    t0 = time.time()
    samples = loaders[benchmark]()
    lookup: Dict[str, List[str]] = {}
    for s in samples:
        q = s.get("question")
        a = s.get("answers")
        if q and a:
            lookup[q] = list(a)
    logger.info("  %s: %d gold-pairs loaded (%.1fs)", benchmark, len(lookup), time.time() - t0)
    return lookup


def _compute_em(prediction: str, gold_list: List[str]) -> float:
    from eval.metrics import any_match_em  # type: ignore
    return any_match_em(prediction, gold_list)


def prune(
    memory_dir: Path,
    dry_run: bool = False,
) -> Dict:
    """Run the EM-prune pass and return the audit report dict."""
    from caem.memory.store import EpisodicMemoryStore  # type: ignore

    base_path = memory_dir / "memory_store"
    faiss_path = Path(str(base_path) + ".faiss")
    meta_path  = Path(str(base_path) + ".meta")
    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Cold-start memory not found at {base_path}.{{faiss,meta}}. "
            "Run scripts/seed_cold_start.py first."
        )

    logger.info("Loading memory store from %s.{faiss,meta}", base_path)
    store = EpisodicMemoryStore.load(base_path)
    metadata = store._metadata  # {entry_id: EpisodicEntry}
    n_total = len(metadata)
    logger.info("  loaded %d entries", n_total)

    # Group by benchmark for efficient gold-lookup loading
    by_bench: Dict[str, List[int]] = defaultdict(list)
    for eid, e in metadata.items():
        by_bench[e.source_benchmark or "_untagged"].append(int(eid))

    benches = sorted(by_bench.keys())
    logger.info("Entries grouped by source_benchmark: %s",
                {b: len(ids) for b, ids in by_bench.items()})

    # Build gold lookups once per benchmark
    gold_by_bench: Dict[str, Dict[str, List[str]]] = {}
    for b in benches:
        if b == "_untagged":
            gold_by_bench[b] = {}
            continue
        gold_by_bench[b] = _build_gold_lookup(b)

    # Walk entries, classify
    keep_ids:    List[int] = []
    drop_ids:    List[int] = []
    unverified_ids: List[int] = []
    per_bench_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "em_1": 0, "em_0": 0, "em_unverified": 0}
    )

    for eid, e in metadata.items():
        eid = int(eid)
        bench = e.source_benchmark or "_untagged"
        per_bench_stats[bench]["total"] += 1

        gold_list = gold_by_bench.get(bench, {}).get(e.question)
        if gold_list is None:
            # No gold lookup available — keep with unverified flag
            unverified_ids.append(eid)
            keep_ids.append(eid)
            per_bench_stats[bench]["em_unverified"] += 1
            continue

        em = _compute_em(e.answer or "", gold_list)
        if em >= 1.0:
            keep_ids.append(eid)
            per_bench_stats[bench]["em_1"] += 1
        else:
            drop_ids.append(eid)
            per_bench_stats[bench]["em_0"] += 1

    n_keep = len(keep_ids)
    n_drop = len(drop_ids)
    n_unverified = len(unverified_ids)
    logger.info("EM classification complete:")
    for bench in benches:
        s = per_bench_stats[bench]
        rate = (s["em_1"] / max(s["total"] - s["em_unverified"], 1)) if s["total"] else 0.0
        logger.info(
            "  %-18s | total=%4d  em=1: %4d  em=0: %4d  unverified: %4d  em_rate=%.3f",
            bench, s["total"], s["em_1"], s["em_0"], s["em_unverified"], rate,
        )
    logger.info("OVERALL: keep=%d (em=1 or unverified), drop=%d (em=0), unverified=%d",
                n_keep, n_drop, n_unverified)

    report = {
        "_run_at_utc":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "memory_dir":        str(memory_dir),
        "dry_run":           bool(dry_run),
        "n_total_before":    n_total,
        "n_kept":            n_keep,
        "n_dropped_em_0":    n_drop,
        "n_unverified_kept": n_unverified,
        "per_benchmark":     dict(per_bench_stats),
        "rule":              "drop entries where any_match_em(entry.answer, gold_answers) == 0.0; keep em=1.0 and unverified entries",
    }

    if dry_run:
        logger.info("DRY RUN — no changes written. Report below.")
        logger.info(json.dumps(report, indent=2))
        return report

    # Backup current store before mutation
    bak_faiss = Path(str(base_path) + ".pre_em_prune.faiss.bak")
    bak_meta  = Path(str(base_path) + ".pre_em_prune.meta.bak")
    logger.info("Backing up current store to %s and %s", bak_faiss, bak_meta)
    shutil.copy2(faiss_path, bak_faiss)
    shutil.copy2(meta_path, bak_meta)

    if n_drop == 0:
        logger.info("Nothing to drop — current store already EM-clean. Writing report only.")
        report_path = memory_dir / "em_pruning_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        logger.info("Report written to %s", report_path)
        return report

    # Rebuild a fresh store from the kept entries.
    # _add_entry_internal handles index insertion + metadata bookkeeping.
    from caem.memory.store import EpisodicMemoryStore  # re-import to ensure fresh ref
    new_store = EpisodicMemoryStore(config=store.config)

    keep_ids_sorted = sorted(set(keep_ids))
    logger.info("Rebuilding store with %d kept entries ...", len(keep_ids_sorted))
    for eid in keep_ids_sorted:
        e = metadata[eid]
        # store.add returns the assigned new id; we don't preserve original ids
        new_store.add(e)
    logger.info("Rebuild complete: %d entries in new store", len(new_store._metadata))

    # Atomic-ish save: write to temp prefix, then rename
    tmp_base = memory_dir / "memory_store.tmp_em_prune"
    new_store.save(tmp_base)
    # Replace the live store
    Path(str(tmp_base) + ".faiss").replace(faiss_path)
    Path(str(tmp_base) + ".meta").replace(meta_path)
    logger.info("New pruned store written to %s.{faiss,meta}", base_path)

    # Update seed_summary.json
    seed_summary_path = memory_dir / "seed_summary.json"
    if seed_summary_path.exists():
        summary = json.loads(seed_summary_path.read_text())
        summary["total_seeded_pre_em_prune"] = summary.get("total_seeded", n_total)
        summary["total_seeded"] = n_keep
        summary["em_prune"] = {
            "applied_at_utc": report["_run_at_utc"],
            "n_dropped":      n_drop,
            "n_kept":         n_keep,
            "report":         "em_pruning_report.json",
        }
        seed_summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("Updated seed_summary.json: total_seeded %d -> %d",
                    summary["total_seeded_pre_em_prune"], n_keep)

    # Write the audit report
    report_path = memory_dir / "em_pruning_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", report_path)

    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--memory_dir", type=Path, default=Path("outputs/cold_start_memory"),
                   help="Cold-start memory directory (containing memory_store.{faiss,meta}).")
    p.add_argument("--dry_run", action="store_true",
                   help="Compute and print the report without modifying the store.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.memory_dir.exists():
        logger.error("Memory directory not found: %s", args.memory_dir)
        sys.exit(1)
    prune(args.memory_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
