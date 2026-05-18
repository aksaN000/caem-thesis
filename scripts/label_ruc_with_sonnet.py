#!/usr/bin/env python3
"""scripts/label_ruc_with_sonnet.py
======================================
Retrieval Utility Classifier (RUC) — Sonnet 4.6 reference-based judge labeller.

Reads ``caem/ruc/training_set.parquet``, asks Claude Sonnet 4.6 to compare the
direct-generation answer with the RAG answer for every row, runs each query twice
with position-swapped answers to neutralise position bias, and writes back:

- ``sonnet_label``      ∈ {0, 1}   — training label; 1 if Sonnet says RAG is better
- ``sonnet_confidence`` ∈ {0.5, 1.0} — 1.0 if both swaps agree, 0.5 otherwise
- ``agreement``         ∈ {0, 1}    — derived; sonnet_label == empirical_label
- ``sonnet_pass1_verdict``, ``sonnet_pass2_verdict`` — raw "A"/"B" verdicts
- ``sonnet_pass1_reason``, ``sonnet_pass2_reason``   — Sonnet's brief reasons

All API calls cache to ``caem/ruc/sonnet_labels.jsonl`` so reruns are free.
The cache is keyed on (id, benchmark, pairing, prompt_version, model).

Modes
-----
``--mode kappa_check`` (default)
    Run 100 random rows and report Cohen's kappa between pass-1 and pass-2 verdicts
    after position-correction. ACCEPTANCE GATE: kappa >= 0.7. If kappa fails, audit
    the prompt before running the full pass.

``--mode full``
    Run the full labelling pass over every row not already cached.

``--limit N``
    Cap the number of new rows labelled (useful for smoke tests).

Cost guidance
-------------
- Per call: ~250 input tokens + ~30 output tokens
- Per row: 2 calls (position-swapped)
- 6,000 rows × 2 calls × ~$0.0012 ≈ $14.40 total

Pinned settings
---------------
Model: claude-sonnet-4-6
Prompt version: see PROMPT_VERSION below; bump when prompt changes.
SDK retries: 0 (custom retry loop with rate-limit-aware backoff).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinned constants — bump PROMPT_VERSION whenever the prompt body changes.
# Cache entries with mismatched (model, prompt_version) are recomputed.
# ---------------------------------------------------------------------------

MODEL_ID = "claude-sonnet-4-6"   # default; overridable via --model CLI
PROMPT_VERSION = "ruc-binary-judge-v3"   # v3: tolerant parser + clearer prompt

# Sonnet 4.6 default tier is generous; we still throttle conservatively.
_MIN_INTERVAL_SEC = 0.6   # ~100 req/min global cap

# Cache schema field (so deserialisers can detect stale rows)
CACHE_SCHEMA_VERSION = "1"


JUDGE_PROMPT_TEMPLATE = """You are comparing two candidate answers to a question. Pick the better answer, or call it a tie.

Decision criteria, priority order:
1. Factual accuracy against the gold reference.
2. Truthfulness on adversarial questions: prefer answers that resist popular misconceptions.
3. Honest uncertainty: prefer "I don't know" over a confidently wrong answer.

CRITICAL: the position of an answer (whether it is shown as A or B) must NOT influence your verdict. If you would judge differently with A and B swapped, the answers are equivalent — choose tie. Most clear wins should be obvious from content alone.

Question:
{question}

Gold reference answer(s):
{gold}

Answer A:
{answer_a}

Answer B:
{answer_b}

Reply with exactly one letter on the first line:
  A   if Answer A is clearly better
  B   if Answer B is clearly better
  T   if they are roughly equivalent in quality

Then on the next line, a single short justification (<=20 words). No XML tags, no extra formatting."""


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

def build_anthropic_client(api_key: Optional[str]) -> Any:
    try:
        import anthropic
    except ImportError as e:
        raise SystemExit(
            "anthropic SDK required. Install: pip install anthropic"
        ) from e
    return anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        timeout=30.0,
        max_retries=0,
    )


_LAST_CALL_T: Dict[str, float] = {"t": 0.0}
_THROTTLE_LOCK = Lock()


def _throttle() -> None:
    """Global throttle: never less than _MIN_INTERVAL_SEC between calls."""
    with _THROTTLE_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_CALL_T["t"]
        if elapsed < _MIN_INTERVAL_SEC:
            time.sleep(_MIN_INTERVAL_SEC - elapsed)
        _LAST_CALL_T["t"] = time.monotonic()


# ---------------------------------------------------------------------------
# Gold-reference formatter (per-benchmark)
# ---------------------------------------------------------------------------

def _format_gold(row: Dict[str, Any]) -> str:
    """Render the gold answer(s) for the judge prompt."""
    bench = row.get("benchmark", "")
    if bench == "truthfulqa":
        # TruthfulQA gold structure was attached during baseline rescore;
        # here we have direct_prediction and rag_prediction but the gold
        # itself comes from the upstream sample. Fall back to a generic
        # accepted-answers field, which build_ruc_training_set.py may have
        # serialised. If absent, the judge still has the question text.
        accepted = row.get("gold_answers") or row.get("correct_answers") or []
        if isinstance(accepted, str):
            accepted = [accepted]
        if accepted:
            return "\n".join(f"- {g}" for g in accepted[:5])
        return "(gold answer list not available; rely on Question text)"
    gold = row.get("gold_label") or row.get("gold_answers") or ""
    if isinstance(gold, list):
        return "\n".join(f"- {g}" for g in gold[:5])
    return str(gold)


import re

# ---------------------------------------------------------------------------
# Permissive verdict parser
# ---------------------------------------------------------------------------

# Order matters: thinking blocks and Markdown bold are stripped before we
# scan for the verdict letter. Patterns accept:
#   "A", "B", "T"
#   "<A>", "<B>", "<T>"
#   "<answer>A</answer>"
#   "**A**"
#   "Verdict: A", "Answer: B", "Line 1: A", "1) A"
# We also tolerate trailing punctuation after the letter.
_THINK_RE = re.compile(r"<thinking>.*?</thinking>", re.IGNORECASE | re.DOTALL)
_XML_VERDICT_RE = re.compile(
    r"<\s*(?:answer|verdict|choice)\s*>\s*([ABT])\s*<\s*/", re.IGNORECASE
)
_BRACKET_RE = re.compile(r"<\s*([ABT])\s*>", re.IGNORECASE)
_PREFIX_RE = re.compile(
    r"(?:verdict|answer|choice|line\s*\d+|\d+\s*[\.\):])\s*[:\-]?\s*([ABT])\b",
    re.IGNORECASE,
)
_STANDALONE_RE = re.compile(r"\*{0,2}\b([ABT])\b\*{0,2}")


def _extract_verdict(text: str) -> Tuple[Optional[str], str]:
    """Return (verdict in {'A','B','T',None}, short_reason_string).

    Strategy:
      1. Drop any <thinking>...</thinking> block.
      2. Try matchers in decreasing specificity, returning on first hit.
      3. Reason is the rest of the output minus the verdict scaffold,
         trimmed to <=160 chars.
    """
    if not text:
        return None, ""
    clean = _THINK_RE.sub(" ", text).strip()

    # Try the XML-tagged form first ("<answer>A</answer>")
    m = _XML_VERDICT_RE.search(clean)
    if m:
        return m.group(1).upper(), _reason_after(clean, m.end())

    # Then the bracketed form ("<A>")
    m = _BRACKET_RE.search(clean)
    if m:
        return m.group(1).upper(), _reason_after(clean, m.end())

    # Then prefix forms ("Verdict: A", "Line 1: A", "1) A")
    m = _PREFIX_RE.search(clean)
    if m:
        return m.group(1).upper(), _reason_after(clean, m.end())

    # Finally, the bare letter (first standalone A/B/T)
    for line in clean.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _STANDALONE_RE.match(line)
        if m:
            tail = line[m.end():].lstrip(" \t.:-,")
            return m.group(1).upper(), tail[:160]
    return None, ""


def _reason_after(text: str, end_idx: int) -> str:
    tail = text[end_idx:].strip().lstrip(":-., ").strip()
    # Take the first non-empty line under 160 chars.
    for ln in tail.splitlines():
        ln = ln.strip()
        if ln:
            return ln[:160]
    return tail[:160]


# ---------------------------------------------------------------------------
# One judge call (position-aware)
# ---------------------------------------------------------------------------

def _judge_call(
    client: Any,
    question: str,
    gold: str,
    answer_a: str,
    answer_b: str,
    *,
    model: str = MODEL_ID,
    max_retries: int = 4,
) -> Tuple[Optional[str], str, str]:
    """Run one judge call. Returns (verdict in {'A','B',None}, reason, raw_text)."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question.strip() or "(empty)",
        gold=gold or "(no gold reference available)",
        answer_a=(answer_a or "(empty)").strip()[:2000],
        answer_b=(answer_b or "(empty)").strip()[:2000],
    )
    last_err = ""
    for attempt in range(1, max_retries + 1):
        _throttle()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=128,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                getattr(b, "text", "") for b in resp.content
                if getattr(b, "type", "text") == "text"
            ).strip()
            verdict, reason = _extract_verdict(text)
            if verdict is not None:
                return verdict, reason, text
            last_err = f"unparseable verdict: {text[:120]!r}"
            time.sleep(min(2 ** attempt, 8))
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            exc_name = type(exc).__name__
            if exc_name == "RateLimitError" or "429" in str(exc) or "rate_limit" in str(exc).lower():
                wait_s = 60.0
                resp_obj = getattr(exc, "response", None)
                if resp_obj is not None:
                    hdr = getattr(resp_obj, "headers", None)
                    if hdr and "retry-after" in {k.lower() for k in hdr.keys()}:
                        try:
                            ra = next(v for k, v in hdr.items() if k.lower() == "retry-after")
                            wait_s = max(float(ra) + 1.0, 30.0)
                        except Exception:
                            pass
                wait_s = wait_s * min(attempt, 3)
                logger.info("Rate-limit hit (attempt %d); sleeping %.0fs", attempt, wait_s)
                time.sleep(wait_s)
            else:
                time.sleep(min(2 ** attempt, 16))
    logger.warning("judge call permanent failure: %s", last_err)
    return None, last_err, ""


# ---------------------------------------------------------------------------
# Combined two-pass labelling (position swap)
# ---------------------------------------------------------------------------

def _verdict_to_rag_better(verdict: Optional[str], pass_num: int) -> Optional[bool]:
    """Map a raw verdict to 'is RAG the better answer?'.

    Pass 1 ordering: direct in A, rag in B.
        A → direct wins → rag_better = False
        B → rag wins    → rag_better = True
        T → tie         → None
    Pass 2 ordering: rag in A, direct in B.
        A → rag wins    → rag_better = True
        B → direct wins → rag_better = False
        T → tie         → None
    """
    if verdict is None or verdict == "T":
        return None
    if pass_num == 1:
        return verdict == "B"
    return verdict == "A"


def label_one_row(
    client: Any,
    row: Dict[str, Any],
    model: str = MODEL_ID,
) -> Dict[str, Any]:
    """Run both position orderings and produce sonnet_label + confidence.

    Aggregation logic (v2 prompt with TIE support):
      - Both passes return TIE                  → sonnet_label = None (drop from training)
      - Exactly one pass returns TIE            → use the non-tie pass, confidence = 0.5
      - Both passes return non-tie and agree    → use that verdict, confidence = 1.0
      - Both passes return non-tie and disagree → position-bias detected; use pass 1
                                                  verdict but confidence = 0.5
      - Both passes failed (API errors)         → sonnet_label = None
    """
    gold = _format_gold(row)
    question = row.get("question") or ""
    direct = row.get("direct_prediction") or ""
    rag = row.get("rag_prediction") or ""

    v1, r1, raw1 = _judge_call(client, question, gold, direct, rag, model=model)
    v2, r2, raw2 = _judge_call(client, question, gold, rag, direct, model=model)

    rag_better_p1 = _verdict_to_rag_better(v1, pass_num=1)
    rag_better_p2 = _verdict_to_rag_better(v2, pass_num=2)

    # Classify the row.
    both_tie = (v1 == "T" and v2 == "T")
    both_failed = (v1 is None and v2 is None)

    if both_tie or both_failed:
        sonnet_label: Optional[int] = None
        sonnet_confidence: Optional[float] = None
    elif rag_better_p1 is None and rag_better_p2 is not None:
        # Pass 1 said TIE or failed; trust pass 2 but mark uncertain.
        sonnet_label = int(bool(rag_better_p2))
        sonnet_confidence = 0.5
    elif rag_better_p2 is None and rag_better_p1 is not None:
        sonnet_label = int(bool(rag_better_p1))
        sonnet_confidence = 0.5
    elif rag_better_p1 == rag_better_p2:
        sonnet_label = int(bool(rag_better_p1))
        sonnet_confidence = 1.0
    else:
        # Both passes returned non-tie verdicts that disagree → position bias.
        sonnet_label = int(bool(rag_better_p1))
        sonnet_confidence = 0.5

    return {
        "id": row.get("id"),
        "benchmark": row.get("benchmark"),
        "pairing": row.get("pairing"),
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "sonnet_label": sonnet_label,
        "sonnet_confidence": sonnet_confidence,
        "sonnet_pass1_verdict": v1,
        "sonnet_pass2_verdict": v2,
        "sonnet_pass1_reason": r1,
        "sonnet_pass2_reason": r2,
        "rag_better_pass1": rag_better_p1,
        "rag_better_pass2": rag_better_p2,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_key(row: Dict[str, Any], model: str = MODEL_ID) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("id")),
        str(row.get("benchmark")),
        str(row.get("pairing")),
        model,
        PROMPT_VERSION,
    )


def load_cache(path: Path) -> Dict[Tuple[str, str, str, str, str], Dict[str, Any]]:
    cache: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (
                str(rec.get("id")),
                str(rec.get("benchmark")),
                str(rec.get("pairing")),
                rec.get("model", ""),
                rec.get("prompt_version", ""),
            )
            cache[key] = rec
    logger.info("Loaded %d cached labels from %s", len(cache), path)
    return cache


def append_cache(path: Path, rec: Dict[str, Any], lock: Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Kappa check
# ---------------------------------------------------------------------------

def cohens_kappa(a: List[int], b: List[int]) -> float:
    """Cohen's kappa for two binary raters."""
    if not a or len(a) != len(b):
        return float("nan")
    n = len(a)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    p_e = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - p_e) < 1e-9:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def run_kappa_check(
    client: Any,
    rows: List[Dict[str, Any]],
    cache_path: Path,
    cache: Dict,
    n_check: int = 100,
    max_workers: int = 8,
    model: str = MODEL_ID,
) -> float:
    """Run n_check random rows, return Cohen's kappa of the two passes."""
    sampled = random.sample(rows, min(n_check, len(rows)))
    write_lock = Lock()
    results: List[Dict[str, Any]] = []

    def _process(r):
        rec = label_one_row(client, r, model=model)
        append_cache(cache_path, rec, write_lock)
        cache[_cache_key(r, model=model)] = rec
        return rec

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process, r) for r in sampled]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as exc:
                logger.warning("kappa worker raised: %s", exc)
                continue
            results.append(rec)
            if i % 10 == 0:
                logger.info("kappa-check: %d / %d done", i, len(sampled))

    # Compute kappa on rows where both passes returned a verdict.
    p1_labels: List[int] = []
    p2_labels: List[int] = []
    for r in results:
        b1 = r.get("rag_better_pass1")
        b2 = r.get("rag_better_pass2")
        if b1 is None or b2 is None:
            continue
        p1_labels.append(int(bool(b1)))
        p2_labels.append(int(bool(b2)))

    if len(p1_labels) < 10:
        logger.error("kappa-check: only %d / %d rows produced both verdicts -- "
                     "cannot compute reliable kappa", len(p1_labels), len(sampled))
        return float("nan")

    kappa = cohens_kappa(p1_labels, p2_labels)
    agreement = sum(1 for a, b in zip(p1_labels, p2_labels) if a == b) / len(p1_labels)
    logger.info(
        "kappa-check on n=%d (%d both-verdict): kappa=%.3f, raw agreement=%.3f",
        len(sampled), len(p1_labels), kappa, agreement,
    )
    return kappa


# ---------------------------------------------------------------------------
# Full labelling pass
# ---------------------------------------------------------------------------

def run_full_pass(
    client: Any,
    rows: List[Dict[str, Any]],
    cache_path: Path,
    cache: Dict,
    limit: Optional[int],
    max_workers: int = 8,
    model: str = MODEL_ID,
) -> None:
    todo = [r for r in rows if _cache_key(r, model=model) not in cache]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        logger.info("All rows already cached; nothing to label.")
        return
    logger.info("Labelling %d uncached rows (cache hit on %d / %d) ...",
                len(todo), len(rows) - len(todo), len(rows))

    write_lock = Lock()
    completed = 0
    fail = 0

    def _process(r):
        rec = label_one_row(client, r, model=model)
        append_cache(cache_path, rec, write_lock)
        cache[_cache_key(r, model=model)] = rec
        return rec

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process, r) for r in todo]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as exc:
                fail += 1
                logger.warning("worker raised: %s", exc)
                continue
            completed += 1
            if rec.get("sonnet_label") is None:
                fail += 1
            if completed % 200 == 0:
                logger.info("progress: %d / %d (fail=%d)", completed, len(todo), fail)

    logger.info("Full pass complete: %d labelled, %d failures.", completed, fail)


# ---------------------------------------------------------------------------
# Parquet read/write
# ---------------------------------------------------------------------------

def merge_cache_into_parquet(
    in_parquet: Path, out_parquet: Path, cache: Dict, model: str = MODEL_ID
) -> None:
    import pandas as pd
    df = pd.read_parquet(in_parquet)

    def _lookup(row):
        key = (
            str(row["id"]),
            str(row["benchmark"]),
            str(row["pairing"]),
            model,
            PROMPT_VERSION,
        )
        return cache.get(key)

    sonnet_label: List[Optional[int]] = []
    sonnet_conf: List[Optional[float]] = []
    p1_verdict: List[Optional[str]] = []
    p2_verdict: List[Optional[str]] = []
    p1_reason: List[Optional[str]] = []
    p2_reason: List[Optional[str]] = []
    for _, row in df.iterrows():
        rec = _lookup(row)
        if rec is None:
            sonnet_label.append(None)
            sonnet_conf.append(None)
            p1_verdict.append(None)
            p2_verdict.append(None)
            p1_reason.append(None)
            p2_reason.append(None)
        else:
            sonnet_label.append(rec.get("sonnet_label"))
            sonnet_conf.append(rec.get("sonnet_confidence"))
            p1_verdict.append(rec.get("sonnet_pass1_verdict"))
            p2_verdict.append(rec.get("sonnet_pass2_verdict"))
            p1_reason.append(rec.get("sonnet_pass1_reason"))
            p2_reason.append(rec.get("sonnet_pass2_reason"))

    df["sonnet_label"] = sonnet_label
    df["sonnet_confidence"] = sonnet_conf
    df["sonnet_pass1_verdict"] = p1_verdict
    df["sonnet_pass2_verdict"] = p2_verdict
    df["sonnet_pass1_reason"] = p1_reason
    df["sonnet_pass2_reason"] = p2_reason

    # Hybrid fallback training label:
    #   - Sonnet strong opinion (confidence=1.0)   → Sonnet label, weight=1.0
    #   - Sonnet weak opinion (confidence=0.5)     → Sonnet label, weight=0.5
    #   - Sonnet both-tie (label=None)             → empirical label, weight=empirical_weight*0.5
    #
    # Rationale: Sonnet's TIE verdict means "I cannot judge the quality
    # difference." On those rows the empirical EM+CHM bi-criteria rule is
    # the next-best signal — downweighted because it's a weaker signal
    # than Sonnet's strong opinion would have been.
    def _hybrid_label(s, c, e):
        try:
            if s is not None and (isinstance(s, (int, float)) and not math.isnan(float(s))):
                return int(float(s))
        except (TypeError, ValueError):
            pass
        # Sonnet abstained or failed; fall back to empirical
        return None if e is None else int(float(e))

    def _hybrid_weight(s, c, e_w):
        try:
            if s is not None and (isinstance(s, (int, float)) and not math.isnan(float(s))):
                return float(c) if c is not None else 0.5
        except (TypeError, ValueError):
            pass
        # Sonnet abstained; use half empirical weight (weaker signal than strong Sonnet)
        if e_w is None:
            return 0.1
        return float(e_w) * 0.5

    df["training_label"] = [
        _hybrid_label(s, c, e)
        for s, c, e in zip(df["sonnet_label"], df["sonnet_confidence"], df["empirical_label"])
    ]
    df["training_weight"] = [
        _hybrid_weight(s, c, ew)
        for s, c, ew in zip(df["sonnet_label"], df["sonnet_confidence"], df["empirical_weight"])
    ]
    def _label_source(s, c):
        # Treat NaN and None the same: "Sonnet didn't return a usable label"
        s_missing = (s is None) or (isinstance(s, float) and math.isnan(s))
        if s_missing:
            return "empirical_fallback"
        try:
            cf = float(c) if c is not None else 0.0
        except (TypeError, ValueError):
            cf = 0.0
        if cf >= 0.99:
            return "sonnet_strong"
        return "sonnet_weak"

    df["label_source"] = [
        _label_source(s, c)
        for s, c in zip(df["sonnet_label"], df["sonnet_confidence"])
    ]

    # agreement = (sonnet_label == empirical_label), guarded for nulls/NaN
    def _agree(s, e):
        if s is None or e is None:
            return None
        try:
            sf = float(s)
            ef = float(e)
        except (TypeError, ValueError):
            return None
        if math.isnan(sf) or math.isnan(ef):
            return None
        return int(int(sf) == int(ef))
    df["agreement"] = [
        _agree(s, e) for s, e in zip(df["sonnet_label"], df["empirical_label"])
    ]

    df.to_parquet(out_parquet, index=False)
    logger.info("Wrote labelled parquet -> %s (rows=%d, labelled=%d)",
                out_parquet, len(df), df["sonnet_label"].notna().sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--out_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"),
                    help="Write back into the same file by default.")
    ap.add_argument("--cache", type=Path,
                    default=Path("caem/ruc/sonnet_labels.jsonl"))
    ap.add_argument("--mode", choices=["kappa_check", "full"], default="kappa_check")
    ap.add_argument("--n_kappa", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None,
                    help="In full mode, cap the number of new rows labelled.")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default=MODEL_ID,
                    help=("Claude model id. Default 'claude-sonnet-4-6'. "
                          "For v2 use 'claude-haiku-4-5' (96.4 pct TQA "
                          "agreement at ~1/4 the cost). Cache is keyed on "
                          "(id, benchmark, pairing, model, prompt_version), "
                          "so switching models triggers a fresh labelling "
                          "pass without invalidating prior cached rows."))
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)

    if not args.in_parquet.exists():
        logger.error("Input parquet not found: %s -- run build_ruc_training_set.py first.",
                     args.in_parquet)
        return 1

    import pandas as pd
    df = pd.read_parquet(args.in_parquet)
    rows = df.to_dict(orient="records")
    logger.info("Loaded %d rows from %s", len(rows), args.in_parquet)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache)

    client = build_anthropic_client(api_key=None)

    logger.info("Using model = %s (prompt_version=%s)", args.model, PROMPT_VERSION)

    if args.mode == "kappa_check":
        kappa = run_kappa_check(client, rows, args.cache, cache,
                                n_check=args.n_kappa, max_workers=args.max_workers,
                                model=args.model)
        merge_cache_into_parquet(args.in_parquet, args.out_parquet, cache,
                                 model=args.model)
        print(f"\nKappa-check result: kappa = {kappa:.3f}")
        if kappa < 0.7:
            print("FAIL: kappa < 0.7. Audit the prompt before running the full pass.")
            return 2
        print("PASS: kappa >= 0.7. Safe to run --mode full.")
        return 0

    run_full_pass(client, rows, args.cache, cache,
                  limit=args.limit, max_workers=args.max_workers,
                  model=args.model)
    merge_cache_into_parquet(args.in_parquet, args.out_parquet, cache,
                             model=args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
