#!/usr/bin/env python3
"""scripts/ruc_train_qtype_svm.py
====================================
Phase 1.3 of the RUC enrichment chain — train a question-type classifier
that replaces the placeholder-prior qtype_* columns in the RUC training set
with proper softmax predictions from a TF-IDF + Linear SVM.

The 5 taxonomy classes (mirror the failure-mode analysis in DESIGN.md):

  factoid     — single-entity fact lookup ("Who wrote War and Peace?")
  commonsense — everyday reasoning ("Why do birds fly south?")
  multihop    — chained facts ("Did the director of Inception also direct Memento?")
  boolean     — yes/no answer expected ("Is Mars larger than Earth?")
  myth        — adversarial / myth-resistance ("Is it true vaccines cause autism?")

Steps
-----
1. Sample N questions from the v1 training set (default 1,500 — full dataset).
2. Ask Claude Sonnet 4.6 to tag each question with one of the 5 classes.
3. Cache the labels to caem/ruc/qtype_labels.jsonl (idempotent rerun).
4. Fit a TF-IDF (1-2gram, min_df=3) + linear SVM (one-vs-rest, calibrated).
5. Re-predict on every parquet row and overwrite the qtype_* columns.
6. Save the model to caem/ruc/qtype_svm.joblib.

Cost
----
- ~1,500 Sonnet calls × ~$0.0012 each = ~$2 total. Position-swap not needed
  (single-class verdict, no A/B order ambiguity).

Acceptance
----------
- Per-class precision/recall on a held-out 20% split: report all 5.
- The SVM is trained on all 5 classes simultaneously (one-vs-rest with
  Platt calibration via sklearn.svm.LinearSVC + CalibratedClassifierCV).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Pinned constants
# ---------------------------------------------------------------------------

MODEL_ID = "claude-sonnet-4-6"
PROMPT_VERSION = "qtype-v1"

QTYPES = ["factoid", "commonsense", "multihop", "boolean", "myth"]
_MIN_INTERVAL_SEC = 0.4  # ~150 req/min ceiling

QTYPE_PROMPT_TEMPLATE = """Classify the following question into exactly one of these five categories:

factoid     — single-entity fact lookup (e.g. "Who wrote War and Peace?")
commonsense — everyday reasoning that doesn't need external facts (e.g. "Why do birds fly south?")
multihop    — requires chaining several facts (e.g. "Did the director of Inception also direct Memento?")
boolean     — primarily expects a yes/no answer (e.g. "Is Mars larger than Earth?")
myth        — adversarial / tests resistance to popular misconceptions (e.g. "Is it true that vaccines cause autism?")

Question:
{question}

Reply with one word: factoid, commonsense, multihop, boolean, or myth.
Then on a new line, a single-sentence justification under 20 words."""


# ---------------------------------------------------------------------------
# Anthropic client + throttle
# ---------------------------------------------------------------------------

def _build_client(api_key: Optional[str]) -> Any:
    import anthropic
    return anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        timeout=30.0,
        max_retries=0,
    )


_LAST = {"t": 0.0}
_LOCK = Lock()


def _throttle() -> None:
    with _LOCK:
        elapsed = time.monotonic() - _LAST["t"]
        if elapsed < _MIN_INTERVAL_SEC:
            time.sleep(_MIN_INTERVAL_SEC - elapsed)
        _LAST["t"] = time.monotonic()


def _parse_qtype(text: str) -> Optional[str]:
    if not text:
        return None
    first = text.splitlines()[0].strip().lower() if text else ""
    # strip "Verdict:", "Type:", "Class:" prefixes
    if ":" in first[:12]:
        first = first.split(":", 1)[-1].strip()
    for qt in QTYPES:
        if first.startswith(qt):
            return qt
    return None


def _call_sonnet(client, question: str, max_retries: int = 4) -> Dict[str, Any]:
    prompt = QTYPE_PROMPT_TEMPLATE.format(question=question.strip() or "(empty)")
    last_err = ""
    for attempt in range(1, max_retries + 1):
        _throttle()
        try:
            resp = client.messages.create(
                model=MODEL_ID, max_tokens=64, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "")
                          for b in resp.content
                          if getattr(b, "type", "text") == "text").strip()
            qt = _parse_qtype(text)
            if qt is not None:
                lines = text.splitlines()
                reason = lines[1].strip()[:160] if len(lines) > 1 else ""
                return {"qtype": qt, "reason": reason, "raw": text}
            last_err = f"unparseable: {text[:80]!r}"
            time.sleep(min(2 ** attempt, 8))
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if "rate" in str(exc).lower() or "429" in str(exc):
                time.sleep(60.0 * min(attempt, 3))
            else:
                time.sleep(min(2 ** attempt, 16))
    logger.warning("Sonnet qtype call failed: %s", last_err)
    return {"qtype": None, "reason": last_err, "raw": ""}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("prompt_version") == PROMPT_VERSION and rec.get("model") == MODEL_ID:
                    cache[str(rec["question_id"])] = rec
            except Exception:
                continue
    logger.info("Loaded %d cached qtype labels from %s", len(cache), path)
    return cache


def _append_cache(path: Path, rec: Dict[str, Any], lock: Lock) -> None:
    with lock:
        with open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--out_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--cache", type=Path,
                    default=Path("caem/ruc/qtype_labels.jsonl"))
    ap.add_argument("--model_out", type=Path,
                    default=Path("caem/ruc/qtype_svm.joblib"))
    ap.add_argument("--n_sample", type=int, default=None,
                    help="Sample only N unique questions (default: all uniques).")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)

    import pandas as pd
    df = pd.read_parquet(args.in_parquet)

    # Get unique questions (one row per question id, ignore pairings)
    unique = df.drop_duplicates(subset=["id", "benchmark"])[
        ["id", "benchmark", "question"]
    ].reset_index(drop=True)
    if args.n_sample is not None:
        unique = unique.sample(n=min(args.n_sample, len(unique)),
                               random_state=args.seed).reset_index(drop=True)
    logger.info("Will label %d unique questions for qtype", len(unique))

    cache = _load_cache(args.cache)
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    todo = [r for _, r in unique.iterrows() if str(r["id"]) not in cache]
    logger.info("Cache hit on %d / %d; labelling %d uncached",
                len(unique) - len(todo), len(unique), len(todo))

    if todo:
        client = _build_client(api_key=None)
        write_lock = Lock()

        def _process(r):
            res = _call_sonnet(client, r["question"])
            rec = {
                "question_id": str(r["id"]),
                "benchmark": r["benchmark"],
                "model": MODEL_ID,
                "prompt_version": PROMPT_VERSION,
                "qtype": res["qtype"],
                "reason": res["reason"],
                "raw": res["raw"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _append_cache(args.cache, rec, write_lock)
            cache[str(r["id"])] = rec
            return rec

        completed = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [pool.submit(_process, r) for r in todo]
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as exc:
                    fail += 1
                    logger.warning("worker failed: %s", exc)
                    continue
                completed += 1
                if rec.get("qtype") is None:
                    fail += 1
                if completed % 100 == 0:
                    logger.info("progress: %d / %d (fail=%d)",
                                completed, len(todo), fail)

        logger.info("Done labelling: %d completed, %d failed", completed, fail)

    # Build the label table from cache
    labelled = []
    for _, r in unique.iterrows():
        rec = cache.get(str(r["id"]))
        if rec and rec.get("qtype") in QTYPES:
            labelled.append({
                "id": str(r["id"]),
                "benchmark": r["benchmark"],
                "question": r["question"],
                "qtype": rec["qtype"],
            })

    if not labelled:
        logger.error("No labelled qtype data; cannot train SVM.")
        return 1

    label_df = pd.DataFrame(labelled)
    logger.info("Labels collected: %d rows", len(label_df))
    print(f"\nLabel distribution:\n{label_df['qtype'].value_counts()}\n")

    # Train SVM
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib

    # Convert pyarrow string arrays to plain Python lists so sklearn can index them
    X_all = label_df["question"].astype(str).tolist()
    y_all = label_df["qtype"].astype(str).tolist()
    import numpy as np
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X_all, dtype=object), np.array(y_all, dtype=object),
        test_size=0.2, random_state=args.seed, stratify=y_all,
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                  min_df=3, max_features=10_000)),
        ("svm", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000),
                                       method="sigmoid", cv=5)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("\nHeld-out 20% classification report:")
    print(classification_report(y_test, y_pred, labels=QTYPES, zero_division=0))

    # Persist the model
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model_out)
    logger.info("Trained SVM -> %s", args.model_out)

    # Re-predict on all training_set.parquet rows and overwrite qtype_* columns
    proba = pipe.predict_proba(df["question"].fillna("").astype(str).tolist())
    classes = list(pipe.classes_)
    for i, qt in enumerate(QTYPES):
        if qt in classes:
            df[f"qtype_{qt}"] = proba[:, classes.index(qt)]
        else:
            df[f"qtype_{qt}"] = 0.0

    df.to_parquet(args.out_parquet, index=False)
    logger.info("Wrote qtype-enriched parquet -> %s (rows=%d)",
                args.out_parquet, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
