#!/usr/bin/env python
"""
scripts/generate_self_rag_synthetic_data.py
============================================
Synthetic-data generation for Self-RAG fine-tuning (B10 full-FT path).

Pipeline
--------
For each ``(question, gold_answer)`` pair drawn from the SIL training-pool
side of FEVER / TriviaQA / CommonsenseQA (the same pools CAEM's SIL trained
over — filtered through ``outputs/full_run/dataset_splits.json#train_ids``
so eval / calibration / purity / test folds stay isolated):

  1. Retrieve top-k passages from the CAEM Tier-3 FAISS index
     (``data/passage_index``) using the shared SBERT ``QueryEncoder``.
  2. Call the Anthropic API to annotate the example with four reflection-
     token types per Asai et al. ICLR 2024 §3.2.
  3. Insert the reflection tokens at appropriate positions in the answer
     text and write a JSONL record consumed by
     ``scripts/train_self_rag.py``.

Stage 1 of the B10 pipeline. Stage 2 is ``scripts/train_self_rag.py``;
stage 3 is the trained-model decoder hooked into
``eval.baselines.SelfRAGBaseline``.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install anthropic     # SDK is lazy-imported; install before running
    python -m scripts.generate_self_rag_synthetic_data \\
        --benchmarks fever triviaqa commonsense_qa \\
        --n_per_bench 5000 \\
        --output outputs/self_rag/synthetic_train.jsonl \\
        --max_concurrent 10 \\
        --model claude-haiku-4-5

Resume safely: if ``--output`` already exists, examples whose ``example_id``
is already present in the file are skipped on re-launch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Reflection-token vocabulary (Asai et al. ICLR 2024 §3.2).
REFLECTION_TOKENS: List[str] = [
    "[Retrieve]", "[No-Retrieve]",
    "[Relevant]", "[Irrelevant]",
    "[Supported]", "[Partial]", "[NoSupport]",
    "[Useful:1]", "[Useful:2]", "[Useful:3]", "[Useful:4]", "[Useful:5]",
]


ANNOTATION_PROMPT_TEMPLATE = """You are annotating training data for Self-RAG, a retrieval-augmented language model.

Given a question, retrieved passages, and a gold answer, produce a training example with reflection tokens inserted per the Self-RAG protocol.

The four reflection-token types:
- [Retrieve] / [No-Retrieve]: should the model retrieve for this question?
- [Relevant] / [Irrelevant]: is each retrieved passage relevant?
- [Supported] / [Partial] / [NoSupport]: is the answer supported by the passages?
- [Useful:1] to [Useful:5]: how useful is the answer (1=worst, 5=best)?

Annotation rules (Asai et al. 2024 §3.2):
1. Open-domain factual questions almost always need retrieval -> use [Retrieve]; trivial commonsense or arithmetic questions can use [No-Retrieve].
2. If a passage clearly contains evidence for the answer -> [Relevant], else [Irrelevant]. Emit one decision per passage, in input order.
3. If the answer's claims are all backed by relevant passages -> [Supported]; if some claims are not -> [Partial]; if none are -> [NoSupport].
4. Useful score: 5 if answer is correct and fluent, 1 if confused or incorrect. Use the gold answer as the reference.

Output format: a single JSON object, no surrounding commentary, with this exact structure:
{{
  "retrieve_decision": "[Retrieve]" or "[No-Retrieve]",
  "passage_decisions": ["[Relevant]", "[Irrelevant]", ...] (one per retrieved passage),
  "support_decision": "[Supported]" or "[Partial]" or "[NoSupport]",
  "useful_score": "[Useful:1]" to "[Useful:5]",
  "annotated_answer": "Reasoning: ... <token> ... <token> Answer: <answer> <token>"
}}

Input:
Question: {question}
Retrieved passages:
{passages}
Gold answer: {gold_answer}

Output JSON only, no commentary."""


# --------------------------------------------------------------------------- #
# Data loading                                                                  #
# --------------------------------------------------------------------------- #

def load_qa_pairs(
    benchmark: str,
    n: int,
    train_ids: Optional[set],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load up to ``n`` ``(question, gold_answer)`` pairs from a benchmark's
    SIL training-pool side.

    ``train_ids`` is the set of canonical sample ids belonging to
    ``BenchmarkPools.sil_train_chunks`` (read from
    ``outputs/full_run/dataset_splits.json#<benchmark>.train_ids``). When
    provided, only samples with matching ``id`` are kept, guaranteeing zero
    contamination with CAEM's calibration / eval / purity / test folds.

    When ``train_ids`` is ``None`` the full benchmark is used (e.g., for
    transfer-only benchmarks that have no train_ids slot).
    """
    from eval.benchmarks import load_benchmark

    samples = load_benchmark(benchmark, n=None, seed=seed)
    if train_ids is not None:
        samples = [s for s in samples if str(s.get("id")) in train_ids]
        logger.info(
            "Filtered %s to %d samples in the SIL training pool.",
            benchmark, len(samples),
        )
    if n is not None and n < len(samples):
        import random
        rng = random.Random(seed)
        samples = rng.sample(samples, n)

    out: List[Dict[str, Any]] = []
    for s in samples:
        answers = s.get("answers") or []
        gold = answers[0] if answers else ""
        if not gold:
            continue
        out.append({
            "example_id": f"{benchmark}::{s.get('id')}",
            "benchmark": benchmark,
            "question": s["question"],
            "gold_answer": gold,
            "all_gold_answers": list(answers),
        })
    return out


def load_train_ids(splits_path: Path) -> Dict[str, set]:
    """Read ``dataset_splits.json`` and return ``{benchmark: set(train_ids)}``.

    Returns an empty dict if the file is missing (caller falls back to the
    full benchmark distribution and logs a warning).
    """
    if not splits_path.is_file():
        logger.warning(
            "dataset_splits.json not found at %s; no train-pool filtering "
            "will be applied. Eval/cal/test contamination is then on you.",
            splits_path,
        )
        return {}
    raw = json.loads(splits_path.read_text())
    out: Dict[str, set] = {}
    for bench, spec in raw.items():
        ids = spec.get("train_ids") if isinstance(spec, dict) else None
        if ids is None:
            continue
        out[bench] = {str(x) for x in ids}
        logger.info("Loaded %d train_ids for %s from %s",
                    len(out[bench]), bench, splits_path)
    return out


# --------------------------------------------------------------------------- #
# Retrieval                                                                     #
# --------------------------------------------------------------------------- #

def build_retriever(
    passage_index_path: Path,
    encoder_device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load ``PassageStore`` + ``QueryEncoder``. Mirrors CAEM Tier 3."""
    from caem.memory.encoder import QueryEncoder
    from caem.retrieval.rag import PassageStore

    logger.info("Loading passage index from %s ...", passage_index_path)
    store = PassageStore.load(str(passage_index_path))
    logger.info("Passage store loaded: %d passages.", len(store.passages))

    encoder = QueryEncoder(device=encoder_device)
    return store, encoder


def retrieve_passages(
    question: str,
    *,
    store: Any,
    encoder: Any,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Encode + FAISS search for top-k passages. Mirrors
    ``TierThreeRAG._retrieve`` so the annotation distribution matches the
    distribution CAEM sees at inference time.
    """
    import numpy as np
    try:
        emb = encoder.encode(question).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return store.search(emb, k=k)
    except Exception as exc:
        logger.warning("retrieve_passages failed for %r: %s", question[:60], exc)
        return []


# --------------------------------------------------------------------------- #
# Anthropic annotation                                                          #
# --------------------------------------------------------------------------- #

_VALID_RETRIEVE = {"[Retrieve]", "[No-Retrieve]"}
_VALID_PASSAGE = {"[Relevant]", "[Irrelevant]"}
_VALID_SUPPORT = {"[Supported]", "[Partial]", "[NoSupport]"}
_VALID_USEFUL = {f"[Useful:{i}]" for i in range(1, 6)}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced ``{...}`` block from ``text`` and parse it.

    Claude almost always returns clean JSON, but we tolerate stray prose
    by scanning for the first ``{`` and matching braces.
    """
    text = text.strip()
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _validate_annotation(obj: Dict[str, Any], n_passages: int) -> bool:
    """Return True if ``obj`` matches the expected reflection-token schema."""
    if not isinstance(obj, dict):
        return False
    if obj.get("retrieve_decision") not in _VALID_RETRIEVE:
        return False
    pds = obj.get("passage_decisions")
    if not isinstance(pds, list) or len(pds) != n_passages:
        return False
    if any(p not in _VALID_PASSAGE for p in pds):
        return False
    if obj.get("support_decision") not in _VALID_SUPPORT:
        return False
    if obj.get("useful_score") not in _VALID_USEFUL:
        return False
    if not isinstance(obj.get("annotated_answer"), str):
        return False
    return True


def annotate_one_example(
    client: Any,
    question: str,
    passages: List[str],
    gold_answer: str,
    *,
    model: str = "claude-haiku-4-5",
    max_retries: int = 3,
    request_timeout: float = 60.0,
) -> Optional[Dict[str, Any]]:
    """Call the Anthropic API once and return the parsed annotation dict.

    Retries up to ``max_retries`` on transient errors (rate-limit, timeout,
    parse failure). Returns ``None`` on permanent failure.
    """
    passages_str = "\n".join(
        f"[{i + 1}] {p}" for i, p in enumerate(passages)
    ) or "(no passages retrieved)"
    prompt = ANNOTATION_PROMPT_TEMPLATE.format(
        question=question,
        passages=passages_str,
        gold_answer=gold_answer,
    )

    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0.0,
                timeout=request_timeout,
                messages=[{"role": "user", "content": prompt}],
            )
            # Concatenate text blocks (anthropic.types.TextBlock).
            text = "".join(
                getattr(b, "text", "") for b in resp.content
                if getattr(b, "type", "text") == "text"
            )
            obj = _extract_json_object(text)
            if obj is None:
                last_err = f"json-parse-fail (attempt {attempt})"
                continue
            if not _validate_annotation(obj, n_passages=len(passages)):
                last_err = f"schema-fail (attempt {attempt})"
                continue
            return obj
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            backoff = min(2 ** attempt, 16)
            logger.debug(
                "annotate retry %d/%d after %s; sleeping %ds",
                attempt, max_retries, last_err, backoff,
            )
            time.sleep(backoff)

    logger.warning("annotate_one_example permanent failure: %s", last_err)
    return None


def build_anthropic_client(api_key: Optional[str]) -> Any:
    """Lazy-import the Anthropic SDK and construct a client.

    Lazy so the rest of the script (loaders, retrieval test) can run on
    machines without the SDK installed.
    """
    try:
        import anthropic
    except ImportError as e:
        raise SystemExit(
            "anthropic SDK is required for synthetic-data generation. "
            "Install with `pip install anthropic` and set ANTHROPIC_API_KEY."
        ) from e
    return anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))


# --------------------------------------------------------------------------- #
# Output                                                                        #
# --------------------------------------------------------------------------- #

def load_existing_ids(output_path: Path) -> set:
    """Return the set of ``example_id`` values already written to
    ``output_path``, for resume support.
    """
    if not output_path.is_file():
        return set()
    ids: set = set()
    with output_path.open("r") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = rec.get("example_id")
            if eid:
                ids.add(eid)
    if ids:
        logger.info(
            "Resume: %d already-annotated examples in %s will be skipped.",
            len(ids), output_path,
        )
    return ids


def _annotate_and_pack(
    item: Dict[str, Any],
    *,
    store: Any,
    encoder: Any,
    client: Any,
    model: str,
    top_k: int,
) -> Optional[Dict[str, Any]]:
    """Run retrieval + annotation for one ``item`` and return the JSONL record."""
    passages_scored = retrieve_passages(
        item["question"], store=store, encoder=encoder, k=top_k,
    )
    passages = [p for p, _ in passages_scored]
    ann = annotate_one_example(
        client,
        question=item["question"],
        passages=passages,
        gold_answer=item["gold_answer"],
        model=model,
    )
    if ann is None:
        return None
    return {
        "example_id": item["example_id"],
        "benchmark": item["benchmark"],
        "question": item["question"],
        "gold_answer": item["gold_answer"],
        "all_gold_answers": item["all_gold_answers"],
        "passages": passages,
        "passage_scores": [float(s) for _, s in passages_scored],
        **ann,
    }


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--benchmarks", nargs="+",
                   default=["fever", "triviaqa", "commonsense_qa"])
    p.add_argument("--n_per_bench", type=int, default=5000)
    p.add_argument("--output", type=Path,
                   default=Path("outputs/self_rag/synthetic_train.jsonl"))
    p.add_argument("--passage_index", type=Path,
                   default=Path("data/passage_index"))
    p.add_argument("--dataset_splits", type=Path,
                   default=Path("outputs/full_run/dataset_splits.json"))
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of passages retrieved per question.")
    p.add_argument("--max_concurrent", type=int, default=10,
                   help="Parallel API calls; tune to Anthropic rate limits.")
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder_device", default=None,
                   help="Device for QueryEncoder (cpu, cuda, cuda:0). "
                        "Defaults to QueryEncoder's auto-select.")
    p.add_argument("--dry_run", action="store_true",
                   help="Load + retrieve only; skip Anthropic calls. "
                        "Useful for verifying the data pipeline.")
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(level=ns.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    ns.output.parent.mkdir(parents=True, exist_ok=True)

    # ---------- Stage 1: load benchmarks (train-pool filtered) ----------
    train_ids_by_bench = load_train_ids(ns.dataset_splits)
    todo: List[Dict[str, Any]] = []
    for bench in ns.benchmarks:
        train_ids = train_ids_by_bench.get(bench)
        if train_ids is None:
            logger.warning(
                "No train_ids slot for %s in %s; using full benchmark "
                "(no contamination filter).", bench, ns.dataset_splits,
            )
        items = load_qa_pairs(bench, ns.n_per_bench, train_ids, seed=ns.seed)
        logger.info("Queued %d items for benchmark %s", len(items), bench)
        todo.extend(items)

    # ---------- Stage 2: skip already-annotated (resume support) ----------
    existing = load_existing_ids(ns.output)
    todo = [t for t in todo if t["example_id"] not in existing]
    logger.info("Items to annotate this run: %d", len(todo))
    if not todo:
        logger.info("Nothing to do; %s already has all requested examples.",
                    ns.output)
        return 0

    # ---------- Stage 3: retriever ----------
    store, encoder = build_retriever(
        ns.passage_index, encoder_device=ns.encoder_device,
    )

    # ---------- Stage 4: client ----------
    if ns.dry_run:
        logger.warning(
            "--dry_run: skipping Anthropic calls; will write empty annotations "
            "with retrieved passages only. Useful for retrieval smoke tests.",
        )
        client: Any = None
    else:
        client = build_anthropic_client(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # ---------- Stage 5: parallel annotation with periodic flush ----------
    write_lock = Lock()
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    with ns.output.open("a") as fh:
        def _emit(rec: Optional[Dict[str, Any]]) -> None:
            nonlocal n_ok, n_fail
            with write_lock:
                if rec is None:
                    n_fail += 1
                    return
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                n_ok += 1
                if (n_ok + n_fail) % 50 == 0:
                    rate = (n_ok + n_fail) / max(time.time() - t0, 1e-6)
                    logger.info(
                        "Progress: ok=%d fail=%d (%.2f/s, eta=%.0f min)",
                        n_ok, n_fail, rate,
                        (len(todo) - n_ok - n_fail) / max(rate, 1e-6) / 60,
                    )

        if ns.dry_run:
            for item in todo:
                passages_scored = retrieve_passages(
                    item["question"], store=store, encoder=encoder, k=ns.top_k,
                )
                _emit({
                    "example_id": item["example_id"],
                    "benchmark": item["benchmark"],
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "all_gold_answers": item["all_gold_answers"],
                    "passages": [p for p, _ in passages_scored],
                    "passage_scores": [float(s) for _, s in passages_scored],
                    "retrieve_decision": None,
                    "passage_decisions": None,
                    "support_decision": None,
                    "useful_score": None,
                    "annotated_answer": None,
                    "dry_run": True,
                })
        else:
            with ThreadPoolExecutor(max_workers=ns.max_concurrent) as pool:
                futures = [
                    pool.submit(
                        _annotate_and_pack,
                        item,
                        store=store, encoder=encoder, client=client,
                        model=ns.model, top_k=ns.top_k,
                    )
                    for item in todo
                ]
                for fut in as_completed(futures):
                    try:
                        _emit(fut.result())
                    except Exception as exc:
                        logger.error("worker raised: %s", exc)
                        _emit(None)

    logger.info(
        "Done. ok=%d fail=%d total=%d wall=%.1f min output=%s",
        n_ok, n_fail, len(todo), (time.time() - t0) / 60, ns.output,
    )
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
