#!/usr/bin/env python3
"""scripts/build_ruc_training_set.py
========================================
Retrieval Utility Classifier (RUC) — training-set assembly.

Walks the rescored baseline JSONs (`outputs/baselines/<baseline>/<bench>_cycle0_with_chm.json`),
pairs each direct-generation baseline with each RAG baseline on the stable ``id`` field,
computes the 14 RUC features and the BGE-small-en-v1.5 question embedding per question,
then writes a single parquet file at ``caem/ruc/training_set.parquet`` containing
the per-pair training rows.

Pairings produced
-----------------
Canonical (primary, used for the headline LOBO-CV evaluation):
    zero_shot  vs  rag

Augmentation (variance reduction, dedup-by-(id, benchmark) at training time):
    cot           vs  cot_rag
    fiveshot_cot  vs  flare         (when flare completes all 5 benches)
    vanilla_ft    vs  rag

Per-row contents
----------------
- question, id, benchmark, pairing
- direct_baseline, rag_baseline (baseline names for traceability)
- direct_em, rag_em                (per-sample, capability_em or em_llm_judged)
- direct_chm, rag_chm              (per-sample CHM via eval.metrics.per_sample_chm)
- direct_prediction, rag_prediction
- top_passage                      (first entry of rag_sample.top_passages)
- empirical_label                  (EM-primary, CHM tiebreaker on EM ties)
- empirical_weight                 (max(|em_delta + 0.5·chm_delta|, 0.1))
- 14 engineered features (see FEATURES list)
- bge_embedding                    (list[384], frozen BGE-small-en-v1.5)

Label rule (bi-criteria, EM primary)
------------------------------------
    if rag_em > direct_em:        empirical_label = 1
    elif rag_em < direct_em:      empirical_label = 0
    else:                          empirical_label = 1 if rag_chm < direct_chm else 0

Weight (composite utility magnitude)
------------------------------------
    em_delta  = rag_em - direct_em
    chm_delta = direct_chm - rag_chm
    utility   = em_delta + 0.5 * chm_delta
    weight    = max(|utility|, 0.1)

Usage
-----
    python -m scripts.build_ruc_training_set \\
        --baselines_dir outputs/baselines \\
        --out caem/ruc/training_set.parquet

The script is idempotent. Pairings whose baseline files are not yet on disk
(e.g., flare half-rescored) are skipped with a warning. The canonical pairing
``zero_shot vs rag`` is required; if either side is missing the script exits 1.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CAEM imports — per_sample_chm must mirror the document-level CHM definition
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import per_sample_chm  # noqa: E402

# ---------------------------------------------------------------------------
# Pairing registry
# ---------------------------------------------------------------------------

BENCHES = ["fever", "triviaqa", "commonsense_qa", "strategyqa", "truthfulqa"]
# v2 panel adds two benches to compensate for the TruthfulQA / CSQA shortage.
# Use --benches at the CLI to override; the module-level constant stays at the
# v1 default so old reruns reproduce.
BENCHES_V2 = BENCHES + ["haluevalqa", "openbookqa"]

PAIRINGS: List[Tuple[str, str, str]] = [
    # (pairing_name, direct_baseline, rag_baseline)
    #
    # 2026-05-17: aug_ft (vanilla_ft vs rag) DROPPED. vanilla_ft is fine-
    # tuned for N SIL cycles, where N is the cycle CAEM stops at. Because
    # baselines now run BEFORE the (RUC-integrated) CAEM trajectory and
    # because the new trajectory's cycle count is undetermined (let the
    # retention guard decide), vanilla_ft output is not available when we
    # need it for augmentation. Three pairings replaces four. See
    # caem/ruc/ruc_next_session_plan.md § 2.3.
    ("canonical",    "zero_shot",    "rag"),
    ("aug_cot",      "cot",          "cot_rag"),
    ("aug_fiveshot", "fiveshot_cot", "flare"),
]
CANONICAL_PAIRING = "canonical"

# ---------------------------------------------------------------------------
# Engineered feature extractors
# ---------------------------------------------------------------------------

_INTERROGATIVES = ("who", "what", "when", "where", "why", "how", "yesno")
_YN_PREFIXES = re.compile(r"^\s*(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b", re.IGNORECASE)
_NEGATION_RX = re.compile(r"\b(not|never|no|none|nobody|nothing|neither|nor|n['’]?t)\b", re.IGNORECASE)
_TEMPORAL_RX = re.compile(r"\b(recently|currently|now|today|yesterday|tomorrow|last\s+(week|month|year)|nowadays|present|modern)\b|\b(19|20)\d{2}\b", re.IGNORECASE)
_NUMERIC_RX = re.compile(r"\b\d+(?:[.,]\d+)?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b", re.IGNORECASE)
_MYTH_RX = re.compile(
    r"is\s+it\s+(true|safe|real|legal)|do\s+(people|most|some|many)\s+(believe|think|say)|"
    r"common\s+(misconception|myth|belief)|what\s+happens\s+if\s+you|"
    r"can\s+(you|i)\s+(get|catch|die\s+from)|"
    r"is\s+it\s+possible\s+to\s+",
    re.IGNORECASE,
)


def _interrogative_one_hot(question: str) -> Dict[str, float]:
    q = question.strip().lower()
    out = {f"interrog_{k}": 0.0 for k in _INTERROGATIVES}
    for w in ("who", "what", "when", "where", "why", "how"):
        if q.startswith(w):
            out[f"interrog_{w}"] = 1.0
            return out
    if _YN_PREFIXES.search(q):
        out["interrog_yesno"] = 1.0
    return out


def _question_text_features(question: str) -> Dict[str, float]:
    """Cheap text features — no NER, no embeddings yet."""
    q = question or ""
    feats: Dict[str, float] = {
        "q_token_len": float(len(q.split())),
        "negation_present": 1.0 if _NEGATION_RX.search(q) else 0.0,
        "temporal_cue": 1.0 if _TEMPORAL_RX.search(q) else 0.0,
        "numerical_cue": 1.0 if _NUMERIC_RX.search(q) else 0.0,
        "myth_regex_hit": 1.0 if _MYTH_RX.search(q) else 0.0,
    }
    feats.update(_interrogative_one_hot(q))
    return feats


# ---------------------------------------------------------------------------
# NER + popularity (heavier features — lazy-init spaCy + pageview table)
# ---------------------------------------------------------------------------

_NLP_SINGLETON = None
_PAGEVIEW_TABLE: Optional[Dict[str, float]] = None


def _load_nlp():
    global _NLP_SINGLETON
    if _NLP_SINGLETON is None:
        import spacy
        try:
            _NLP_SINGLETON = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        except OSError as exc:
            logger.error(
                "spaCy model 'en_core_web_sm' not installed. Install with: "
                "python -m spacy download en_core_web_sm"
            )
            raise SystemExit(1) from exc
    return _NLP_SINGLETON


def _load_pageviews(path: Optional[Path]) -> Dict[str, float]:
    """Load offline Wikipedia pageview table. If missing, return empty dict
    and log a one-line warning. Pageview features then fall back to 0.0,
    which the LightGBM model handles as a signal."""
    global _PAGEVIEW_TABLE
    if _PAGEVIEW_TABLE is not None:
        return _PAGEVIEW_TABLE
    if path is None or not path.exists():
        logger.warning(
            "Pageview table not found at %s. Falling back to entity-count "
            "only (log_pageviews_max = 0.0 for every row). To enable, build "
            "the table from a Wikipedia pageview dump and place it at the path.",
            path,
        )
        _PAGEVIEW_TABLE = {}
        return _PAGEVIEW_TABLE
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        _PAGEVIEW_TABLE = dict(zip(df["entity"].str.lower(), df["log_pageviews"]))
        logger.info("Loaded %d entities from %s", len(_PAGEVIEW_TABLE), path)
    except Exception as exc:
        logger.warning("Failed to load pageview table (%s); using empty fallback", exc)
        _PAGEVIEW_TABLE = {}
    return _PAGEVIEW_TABLE


def _entity_features(question: str, pageview_path: Optional[Path]) -> Tuple[List[str], Dict[str, float]]:
    """Return (detected_entities, {'entity_count', 'log_pageviews_max'})."""
    nlp = _load_nlp()
    doc = nlp(question or "")
    entities = [ent.text for ent in doc.ents]
    pv = _load_pageviews(pageview_path)
    log_pv_max = 0.0
    for ent in entities:
        v = pv.get(ent.lower())
        if v is not None and v > log_pv_max:
            log_pv_max = float(v)
    return entities, {
        "entity_count": float(len(entities)),
        "log_pageviews_max": float(log_pv_max),
    }


# ---------------------------------------------------------------------------
# Question-type SVM placeholder
# ---------------------------------------------------------------------------
#
# The 5-way question-type SVM (factoid / commonsense / multihop / boolean /
# myth) is trained downstream by scripts/train_qtype_svm.py from
# Sonnet-labelled question-type tags. At this stage we emit a placeholder
# that records the source benchmark's hard label so train_ruc.py can fit
# the SVM and overwrite these columns with proper softmax probabilities.

BENCH_TO_QTYPE_PRIOR = {
    "fever":          {"factoid": 0.85, "commonsense": 0.05, "multihop": 0.05, "boolean": 0.05, "myth": 0.0},
    "triviaqa":       {"factoid": 0.90, "commonsense": 0.05, "multihop": 0.05, "boolean": 0.0, "myth": 0.0},
    "commonsense_qa": {"factoid": 0.10, "commonsense": 0.85, "multihop": 0.0, "boolean": 0.05, "myth": 0.0},
    "strategyqa":     {"factoid": 0.10, "commonsense": 0.10, "multihop": 0.40, "boolean": 0.40, "myth": 0.0},
    "truthfulqa":     {"factoid": 0.15, "commonsense": 0.15, "multihop": 0.0, "boolean": 0.0, "myth": 0.70},
}


def _qtype_placeholder(bench: str) -> Dict[str, float]:
    """Hard-prior placeholder: encodes the source-benchmark's modal type.

    The qtype SVM later overwrites these with softmax probabilities trained
    on Sonnet-labelled tags. These priors prevent the training script from
    crashing on missing columns and give a sensible LOBO-CV floor.
    """
    prior = BENCH_TO_QTYPE_PRIOR.get(bench, {})
    return {f"qtype_{k}": float(prior.get(k, 0.0))
            for k in ("factoid", "commonsense", "multihop", "boolean", "myth")}


# ---------------------------------------------------------------------------
# BGE-small-en-v1.5 frozen encoder
# ---------------------------------------------------------------------------

_BGE_SINGLETON = None


def _load_bge():
    global _BGE_SINGLETON
    if _BGE_SINGLETON is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading BGE-small-en-v1.5 (~33M params, frozen) ...")
        _BGE_SINGLETON = SentenceTransformer("BAAI/bge-small-en-v1.5")
        _BGE_SINGLETON.eval()
    return _BGE_SINGLETON


def _bge_encode(questions: List[str], batch_size: int = 64) -> np.ndarray:
    model = _load_bge()
    return np.asarray(
        model.encode(questions, batch_size=batch_size, show_progress_bar=True,
                     normalize_embeddings=True),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Passage-side features (retrieval-quality)
# ---------------------------------------------------------------------------

def _top1_passage_features(question: str,
                           question_emb: np.ndarray,
                           top_passage: Optional[str],
                           question_entities: List[str]) -> Dict[str, float]:
    """Compute top1_passage_sim + top1_passage_entity_overlap.

    Operates on the *already-retrieved* top passage stored alongside the
    rag baseline sample. We do not re-query FAISS here because the rag
    baseline ran the retrieval at generation time and the passage is on
    disk. This keeps the training-set build fully offline.

    KNOWN LIMITATION (2026-05-17): the existing rag baseline JSONs were
    generated before the pipeline persisted the top-K passages, so
    ``top_passages`` is an empty list for all rows. Both features
    therefore emit 0.0 in this build. Re-running retrieval offline using
    ``data/passage_index/passages.faiss`` is a ~30-min job that needs
    ~64 GB free RAM; it should be done once the rescore pipeline frees
    its index. Reserved as a v1.1 upgrade lift if the primary acceptance
    gate misses on the v1 build.
    """
    if not top_passage:
        return {"top1_passage_sim": 0.0, "top1_passage_entity_overlap": 0.0}

    bge = _load_bge()
    passage_emb = bge.encode([top_passage], normalize_embeddings=True)[0]
    sim = float(np.dot(question_emb, passage_emb))

    p_lower = top_passage.lower()
    overlap = sum(1 for e in question_entities if e.lower() in p_lower)

    return {
        "top1_passage_sim": sim,
        "top1_passage_entity_overlap": float(overlap),
    }


# ---------------------------------------------------------------------------
# EM helper (TruthfulQA dispatches to em_llm_judged when present)
# ---------------------------------------------------------------------------

def _sample_em(sample: Dict[str, Any], bench: str) -> Optional[float]:
    if bench == "truthfulqa" and sample.get("em_llm_judged") is not None:
        return float(sample["em_llm_judged"])
    em = sample.get("capability_em")
    if em is None:
        em = sample.get("em")
    return float(em) if em is not None else None


# ---------------------------------------------------------------------------
# Pairing + row assembly
# ---------------------------------------------------------------------------

def _load_baseline(baselines_dir: Path, baseline: str, bench: str) -> Optional[Dict[str, Any]]:
    p = baselines_dir / baseline / f"{bench}_cycle0_with_chm.json"
    if not p.exists():
        return None
    return json.load(open(p))


def _build_pair_rows(
    baselines_dir: Path,
    pairing_name: str,
    direct_baseline: str,
    rag_baseline: str,
    pageview_path: Optional[Path],
    benches: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return one row per paired (direct, rag) sample for this pairing."""
    if benches is None:
        benches = BENCHES
    rows: List[Dict[str, Any]] = []
    for bench in benches:
        d_doc = _load_baseline(baselines_dir, direct_baseline, bench)
        r_doc = _load_baseline(baselines_dir, rag_baseline, bench)
        if d_doc is None or r_doc is None:
            logger.warning(
                "pairing=%s bench=%s skipped (direct_exists=%s rag_exists=%s)",
                pairing_name, bench, d_doc is not None, r_doc is not None,
            )
            continue
        d_by_id = {s["id"]: s for s in d_doc["samples"]}
        r_by_id = {s["id"]: s for s in r_doc["samples"]}
        common = sorted(set(d_by_id) & set(r_by_id))
        if not common:
            logger.warning("pairing=%s bench=%s no overlapping ids", pairing_name, bench)
            continue

        # Encode all questions for this bench in one BGE batch.
        questions = [d_by_id[sid].get("question") or r_by_id[sid].get("question") or ""
                     for sid in common]
        bge_matrix = _bge_encode(questions)

        for sid, q_emb in zip(common, bge_matrix):
            d = d_by_id[sid]
            r = r_by_id[sid]
            question = d.get("question") or r.get("question") or ""

            d_em = _sample_em(d, bench)
            r_em = _sample_em(r, bench)
            if d_em is None or r_em is None:
                continue
            d_chm = per_sample_chm(d)
            r_chm = per_sample_chm(r)

            # Bi-criteria label, EM primary, CHM tiebreaker
            if r_em > d_em:
                label = 1
            elif r_em < d_em:
                label = 0
            else:
                label = 1 if r_chm < d_chm else 0

            em_delta = r_em - d_em
            chm_delta = d_chm - r_chm
            utility = em_delta + 0.5 * chm_delta
            weight = max(abs(utility), 0.1)

            # Engineered features
            text_feats = _question_text_features(question)
            entities, entity_feats = _entity_features(question, pageview_path)
            qtype_feats = _qtype_placeholder(bench)
            top_passage = None
            tp = r.get("top_passages")
            if isinstance(tp, list) and tp:
                top_passage = tp[0] if isinstance(tp[0], str) else (
                    tp[0].get("text") if isinstance(tp[0], dict) else None
                )
            passage_feats = _top1_passage_features(question, q_emb, top_passage, entities)

            row = {
                "id": str(sid),
                "benchmark": bench,
                "pairing": pairing_name,
                "direct_baseline": direct_baseline,
                "rag_baseline": rag_baseline,
                "question": question,
                "direct_em": float(d_em),
                "rag_em": float(r_em),
                "direct_chm": float(d_chm),
                "rag_chm": float(r_chm),
                "direct_prediction": d.get("prediction") or "",
                "rag_prediction": r.get("prediction") or "",
                "top_passage": top_passage or "",
                "empirical_label": int(label),
                "empirical_weight": float(weight),
                "em_delta": float(em_delta),
                "chm_delta": float(chm_delta),
                "utility": float(utility),
                # Engineered features
                **text_feats,
                **entity_feats,
                **qtype_feats,
                **passage_feats,
                # Dense embedding
                "bge_embedding": q_emb.tolist(),
            }
            rows.append(row)

        logger.info("pairing=%s bench=%s paired %d / %d (direct=%d, rag=%d)",
                    pairing_name, bench, len(common), len(d_by_id), len(d_by_id), len(r_by_id))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines_dir", type=Path,
                    default=Path("outputs/baselines"))
    ap.add_argument("--pageview_table", type=Path,
                    default=Path("data/ruc/pageviews.parquet"),
                    help="Optional offline Wikipedia pageview table.")
    ap.add_argument("--out", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--canonical_only", action="store_true",
                    help="Build only the zero_shot vs rag canonical pairing.")
    ap.add_argument("--benches", nargs="+", default=None,
                    help=("Override the benchmark list. Default = v1 panel "
                          "(fever, triviaqa, commonsense_qa, strategyqa, "
                          "truthfulqa). For v2 pass --benches fever "
                          "triviaqa commonsense_qa strategyqa truthfulqa "
                          "haluevalqa openbookqa."))
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    pairings = PAIRINGS[:1] if args.canonical_only else PAIRINGS

    for name, direct, rag in pairings:
        rows = _build_pair_rows(args.baselines_dir, name, direct, rag,
                                args.pageview_table, benches=args.benches)
        if name == CANONICAL_PAIRING and not rows:
            logger.error("Canonical pairing (%s vs %s) produced 0 rows -- "
                         "cannot build training set without it. Aborting.",
                         direct, rag)
            return 1
        logger.info("pairing=%s produced %d rows", name, len(rows))
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No rows produced from any pairing. Aborting.")
        return 1

    import pandas as pd
    df = pd.DataFrame(all_rows)

    # Dedup on (id, benchmark, pairing): each pairing contributes its own row.
    # Earlier versions deduped on (id, benchmark) only, which deleted all
    # augmentation rows that shared an id with the canonical pairing — that
    # defeated the augmentation purpose entirely. Multiple labels per question
    # across pairings is *the point* of augmentation: the classifier learns the
    # prompt-style-invariant retrieval-utility signal.
    #
    # LOBO-CV correctness: scripts/train_ruc.py splits by `benchmark` only, so
    # all augmentation rows for a held-out benchmark go to the test fold
    # together. No leakage from this dedup change.
    n_before = len(df)
    df = df.drop_duplicates(subset=["id", "benchmark", "pairing"], keep="first")
    n_after = len(df)
    logger.info("Dedup on (id, benchmark, pairing): %d → %d rows", n_before, n_after)

    df.to_parquet(args.out, index=False)

    # Coverage table
    print(f"\n{'='*68}")
    print(f"RUC training set summary  →  {args.out}")
    print(f"{'='*68}")
    print(f"{'pairing':<15} {'benchmark':<16} {'n':>6} {'pos_rate':>10} {'mean_weight':>12}")
    print("-" * 68)
    for (pairing, bench), g in df.groupby(["pairing", "benchmark"]):
        print(f"{pairing:<15} {bench:<16} {len(g):>6} "
              f"{g['empirical_label'].mean():>10.3f} {g['empirical_weight'].mean():>12.3f}")
    print("-" * 68)
    print(f"{'TOTAL':<15} {'':<16} {len(df):>6} "
          f"{df['empirical_label'].mean():>10.3f} {df['empirical_weight'].mean():>12.3f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
