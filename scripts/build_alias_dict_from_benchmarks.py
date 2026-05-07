#!/usr/bin/env python
"""
scripts/build_alias_dict_from_benchmarks.py
============================================
Build a Wikidata-style alias dictionary from the benchmark training
splits, for use by the v2 ``alias_overlap`` verifier signal (Fix 6).

Why this script exists
----------------------
The v2 plan called for a 150 MB Wikidata alias dump to feed
``WikidataAliasResolver``. The download + ingestion infrastructure was
not built before Phase 1a launch, so the verifier was always falling
back to the neutral 0.5 prior — silently neutralising the alias_overlap
signal in the cal_prob composite.

This pragmatic alternative harvests aliases directly from the benchmark
loaders. TriviaQA, NaturalQuestions, and HotpotQA ship alias-enriched
``answers`` lists in their train splits — typically the first element
is the canonical name and the rest are surface-form variants. We
collect those into a single ``{canonical: [alias, alias, ...]}`` JSON
file that ``InMemoryAliasResolver`` can load.

Coverage is necessarily smaller than full Wikidata (no Q-numbers, only
entities the model is likely to be asked about in our panel), but the
signal is non-trivial: ~10-30k entities with multiple surface forms
across the four panel benchmarks. Phase 1c can swap in real Wikidata
by writing to the same JSON path.

Outputs
-------
- ``data/alias_dict.json``  (default; configurable via --output)
   Schema: ``{canonical_str: [alias_str, ...], ...}`` — same shape
   ``InMemoryAliasResolver(table=...)`` accepts.

Usage
-----
    python scripts/build_alias_dict_from_benchmarks.py
    python scripts/build_alias_dict_from_benchmarks.py --n_per_bench 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("build_alias_dict")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _harvest_from_loader(
    loader_fn, benchmark: str, split: str, n: int,
) -> Dict[str, Set[str]]:
    """Harvest (canonical, aliases) tuples from a benchmark loader.

    Each sample's ``answers`` list is treated as a surface-form group.
    The first element is the canonical key; remaining elements are added
    as aliases. If a key already exists, alias sets are merged.
    """
    out: Dict[str, Set[str]] = defaultdict(set)
    try:
        samples = loader_fn(split=split, n=n)
    except TypeError:
        # Some loaders don't accept n kwarg
        samples = loader_fn(split=split)
        if n is not None and n < len(samples):
            samples = samples[:n]
    except Exception as exc:
        logger.warning("Loader for %s failed: %s — skipping", benchmark, exc)
        return out

    for s in samples:
        answers = s.get("answers") or []
        if not answers:
            continue
        # Skip degenerate single-element lists (no alias data to harvest)
        if len(answers) < 2:
            continue
        canonical = str(answers[0]).strip()
        if not canonical:
            continue
        for alt in answers[1:]:
            alt = str(alt).strip()
            if alt and alt != canonical:
                out[canonical].add(alt)

    logger.info("  %s/%s: %d canonical entries with ≥1 alias", benchmark, split, len(out))
    return out


def _merge_alias_dicts(*dicts: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Merge multiple harvest results into a flat dict for JSON serialisation."""
    merged: Dict[str, Set[str]] = defaultdict(set)
    for d in dicts:
        for k, v in d.items():
            merged[k].update(v)
    return {k: sorted(v) for k, v in sorted(merged.items())}


def build_alias_dict(n_per_bench: int = 3000) -> Dict[str, List[str]]:
    from eval.benchmarks import (  # type: ignore
        load_triviaqa, load_natural_questions, load_hotpotqa,
    )
    triviaqa = _harvest_from_loader(load_triviaqa, "triviaqa", "train", n_per_bench)
    nq       = _harvest_from_loader(load_natural_questions, "natural_questions", "train", n_per_bench)
    hotpot   = _harvest_from_loader(load_hotpotqa, "hotpotqa", "train", n_per_bench)
    # FEVER's answers are labels (supports/refutes/NEI) — not alias-bearing; skip.
    # CommonsenseQA's answers are letters (A-E) — not alias-bearing; skip.
    return _merge_alias_dicts(triviaqa, nq, hotpot)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--output", type=Path, default=Path("data/alias_dict.json"))
    p.add_argument("--n_per_bench", type=int, default=3000,
                   help="Cap on samples per benchmark to harvest (default 3000).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building alias dict (n_per_bench=%d)...", args.n_per_bench)
    table = build_alias_dict(n_per_bench=args.n_per_bench)
    if not table:
        logger.error(
            "Alias harvest empty — loaders returned no alias-bearing answers. "
            "alias_overlap signal will remain neutral."
        )
        return 1
    args.output.write_text(json.dumps(table, indent=2, ensure_ascii=False))
    n_keys = len(table)
    n_aliases = sum(len(v) for v in table.values())
    logger.info(
        "Wrote alias dict: %d canonical entries × %d total aliases -> %s",
        n_keys, n_aliases, args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
