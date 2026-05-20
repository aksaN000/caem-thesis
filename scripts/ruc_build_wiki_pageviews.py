#!/usr/bin/env python3
"""scripts/ruc_build_wiki_pageviews.py
=========================================
Phase 1.2 of the RUC enrichment chain — build the Wikipedia pageviews
lookup table that populates the ``log_pageviews_max`` feature.

Strategy (no Wikimedia dump downloads needed)
---------------------------------------------
1. Scan the RUC training set, extract all unique named entities via
   spaCy en_core_web_sm.
2. For each unique entity, query the Wikimedia REST API:
       https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/
           en.wikipedia/all-access/all-agents/{title}/monthly/{from}/{to}
3. Take log10 of the mean monthly pageviews over the last 12 months.
4. Cache misses (404s, redirects, ambiguous titles).
5. Write ``data/ruc/pageviews.parquet`` with columns:
       entity (str), log_pageviews (float)

The Wikimedia REST API is free, no auth, ~200 req/s rate limit per IP.
For our scale (~1,000-2,000 unique entities) total wall-clock is ~10-30 s.

Idempotent: re-runs reuse a cache file at caem/ruc/wiki_pageviews_cache.jsonl.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REST_TEMPLATE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{title}/monthly/{from_date}/{to_date}"
)
USER_AGENT = "RUC-Trainer/1.0 (CAEM thesis project; contact@example.org)"
_MIN_INTERVAL_SEC = 0.05  # 20 req/s — well under Wikimedia's 200 req/s limit


_LAST = {"t": 0.0}
_LOCK = Lock()


def _throttle() -> None:
    with _LOCK:
        elapsed = time.monotonic() - _LAST["t"]
        if elapsed < _MIN_INTERVAL_SEC:
            time.sleep(_MIN_INTERVAL_SEC - elapsed)
        _LAST["t"] = time.monotonic()


def _normalise_title(entity: str) -> str:
    """Wikipedia article-title normalisation: title-case first letter, underscores."""
    if not entity:
        return ""
    s = entity.strip()
    if not s:
        return ""
    # Title-case the first letter, keep rest as-is
    s = s[0].upper() + s[1:]
    s = s.replace(" ", "_")
    return urllib.parse.quote(s, safe="_,()")


def _query_pageviews(entity: str, months_back: int = 12,
                     max_retries: int = 3) -> Optional[float]:
    """Return mean monthly pageviews over the last ``months_back`` months,
    or None if the entity is not found or all queries failed.
    """
    title = _normalise_title(entity)
    if not title:
        return None
    # Build date range: from = today - months_back, to = today
    now = time.localtime()
    to_y, to_m = now.tm_year, now.tm_mon
    # Wikimedia REST API uses YYYYMMDD with day=01 for monthly
    from_m = to_m - months_back
    from_y = to_y
    while from_m <= 0:
        from_m += 12
        from_y -= 1
    from_date = f"{from_y:04d}{from_m:02d}01"
    to_date = f"{to_y:04d}{to_m:02d}01"
    url = REST_TEMPLATE.format(title=title, from_date=from_date, to_date=to_date)

    for attempt in range(1, max_retries + 1):
        _throttle()
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # entity not in Wikipedia
            if e.code == 429:
                time.sleep(min(2 ** attempt, 8))
                continue
            logger.debug("HTTP %s on %s", e.code, entity)
            return None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(min(2 ** attempt, 8))
            continue
        except Exception:
            return None
        items = payload.get("items") or []
        if not items:
            return None
        # Mean of monthly counts
        counts = [int(it.get("views") or 0) for it in items]
        if not counts:
            return None
        mean_views = sum(counts) / len(counts)
        if mean_views <= 0:
            return 0.0
        return float(math.log10(mean_views))
    return None


def _load_cache(path: Path) -> Dict[str, Optional[float]]:
    cache: Dict[str, Optional[float]] = {}
    if not path.exists():
        return cache
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                cache[rec["entity"]] = rec.get("log_pageviews")
            except Exception:
                continue
    logger.info("Loaded %d cached pageview lookups", len(cache))
    return cache


def _append_cache(path: Path, entity: str, value: Optional[float],
                  lock: Lock) -> None:
    rec = {"entity": entity, "log_pageviews": value,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with lock:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_parquet", type=Path,
                    default=None,
                    help=("Parquet input (column 'question'). Mutually "
                          "exclusive with --in_jsons; if neither is supplied "
                          "we fall back to caem/ruc/training_set.parquet."))
    ap.add_argument("--in_jsons", type=Path, nargs="+",
                    default=None,
                    help=("One or more list-of-dicts JSON files with a "
                          "'question' field. Used to feed v2 fresh-pool "
                          "slices (caem/ruc/fresh_pool/*_v2.json) before "
                          "the baselines + rescore have built the v2 "
                          "training parquet."))
    ap.add_argument("--cache", type=Path,
                    default=Path("caem/ruc/wiki_pageviews_cache.jsonl"))
    ap.add_argument("--out_parquet", type=Path,
                    default=Path("data/ruc/pageviews.parquet"))
    ap.add_argument("--max_workers", type=int, default=12)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.in_jsons and args.in_parquet:
        ap.error("--in_parquet and --in_jsons are mutually exclusive.")

    questions: List[str] = []
    if args.in_jsons:
        seen: Set[str] = set()
        for jp in args.in_jsons:
            if not jp.exists():
                logger.warning("missing JSON %s; skipped", jp)
                continue
            data = json.loads(jp.read_text())
            for row in data:
                q = (row.get("question") or "").strip()
                if q and q not in seen:
                    seen.add(q)
                    questions.append(q)
            logger.info("loaded %s -> running total %d unique questions",
                        jp.name, len(questions))
    else:
        in_parquet = args.in_parquet or Path("caem/ruc/training_set.parquet")
        if not in_parquet.exists():
            logger.error("Input parquet missing: %s -- pass --in_jsons or "
                         "build the training set first.", in_parquet)
            return 1
        import pandas as pd
        df = pd.read_parquet(in_parquet)
        questions = df["question"].fillna("").unique().tolist()
    logger.info("Will extract entities from %d unique questions", len(questions))

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])

    entities: Set[str] = set()
    for q in questions:
        doc = nlp(q)
        for ent in doc.ents:
            if ent.text.strip():
                entities.add(ent.text.strip())
    logger.info("Detected %d unique entities", len(entities))

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(args.cache)

    todo = [e for e in entities if e not in cache]
    logger.info("Cache hit on %d / %d entities; querying %d",
                len(entities) - len(todo), len(entities), len(todo))

    if todo:
        write_lock = Lock()

        def _process(ent):
            v = _query_pageviews(ent)
            _append_cache(args.cache, ent, v, write_lock)
            cache[ent] = v
            return ent, v

        completed = 0
        hits = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [pool.submit(_process, e) for e in todo]
            for fut in as_completed(futures):
                try:
                    _, v = fut.result()
                except Exception as exc:
                    logger.warning("worker raised: %s", exc)
                    continue
                completed += 1
                if v is not None:
                    hits += 1
                if completed % 200 == 0:
                    logger.info("progress: %d / %d  (hits=%d)",
                                completed, len(todo), hits)
        logger.info("Query pass complete: %d total, %d hits", completed, hits)

    # Build the final parquet
    import pandas as pd
    rows = [{"entity": e.lower(), "log_pageviews": float(v)}
            for e, v in cache.items() if v is not None]
    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning("No pageviews recovered. Output parquet will be empty.")
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)

    print(f"\nPageviews summary  ->  {args.out_parquet}")
    print(f"  total queried entities: {len(entities)}")
    print(f"  entities with Wikipedia article: {len(out)} "
          f"({len(out)/max(len(entities),1)*100:.1f}%)")
    if not out.empty:
        print(f"  log_pageviews stats: min={out['log_pageviews'].min():.2f}  "
              f"mean={out['log_pageviews'].mean():.2f}  "
              f"max={out['log_pageviews'].max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
