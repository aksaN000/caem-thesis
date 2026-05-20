#!/usr/bin/env python3
"""scripts/ruc_extract_v2_features.py
====================================
v2 RUC Config B feature extraction.

Reads the fresh-pool question JSONs (``caem/ruc/fresh_pool/*.json``) and emits a
single parquet of v2-only Config B features keyed by ``(id, benchmark)``. The
output joins with the v2 training set (built by ``scripts/build_ruc_training_set``)
via ``scripts/ruc_merge_features``.

v2 Config B feature inventory (~34 dims total, this script emits 13 of them):

    top5_sim_max          FAISS top-5 cosine sim, max
    top5_sim_mean         FAISS top-5 cosine sim, mean
    top5_sim_std          FAISS top-5 cosine sim, std
    rerank_top1           Cross-encoder rerank score, top passage
    rerank_top5_mean      Cross-encoder rerank score, mean over top-5
    rerank_top5_std       Cross-encoder rerank score, std over top-5
    nli_pair_mean         NLI pairwise entailment over (5 choose 2)=10 pairs, mean
    nli_pair_min          NLI pairwise entailment over the 10 pairs, min
    nli_pair_std          NLI pairwise entailment over the 10 pairs, std
    entity_in_wikidata    1.0 if any spaCy entity has a Wikidata alias entry

Architectural notes
-------------------
- Reuses CAEM's verifier components (no parallel infrastructure):
    QueryEncoder (mpnet-base, 768-dim) for FAISS query encoding
    PassageStore (21M-passage IVF) loaded from data/passage_index
    CrossEncoder(config.cross_encoder_model) -> BAAI/bge-reranker-v2-m3
    load_verifier_judge(config, device) -> AdaptiveNLIJudge (MiniCheck <= 408
        MC-tokens, FrozenQwenJudge longer with Platt-cal) or bare MiniCheck.
    Optional InMemoryAliasResolver from config.alias_dict_path for Wikidata.
- These components mirror exactly what scripts/rescore_baselines_through_verifier
  builds inside ``_build_verifier`` (lines 199-211), so the v2 features are
  computed under the same retrieval / reranking / NLI instruments the trajectory
  verifier uses. No new dependency, no parallel model loading path.
- Per-question (not per-pairing): output is keyed by ``(id, benchmark)`` only.
  ``ruc_merge_features.py`` broadcasts the v2 features over training rows of all
  pairings on the same question.

Usage
-----
    python -m scripts.ruc_extract_v2_features \\
        --fresh_pool_glob "caem/ruc/fresh_pool/*.json" \\
        --passage_index data/passage_index \\
        --out_parquet caem/ruc/v2_features.parquet \\
        --device cuda

Smoke test (first 30 questions per bench, ~5 min on a 5090):

    python -m scripts.ruc_extract_v2_features --limit 30 --out_parquet /tmp/v2_smoke.parquet

Estimated wall-time on full 12 004 questions, 5090, bs=32:
    FAISS top-5 retrieval:       ~12 min  (CPU IVF-PQ, 50ms/q)
    Cross-encoder rerank:        ~25 min  (12k * 5 pairs, pooled per-question)
    NLI pairwise (MiniCheck):    ~45 min  (12k * 10 pairs)
    Wikidata alias lookup:       ~ 2 min  (in-memory dict)
    Total:                       ~85 min  + 5 min warmup.

Idempotency: writes the full parquet at the end. Re-running clobbers. Use
``--limit`` for smoke tests; production runs always emit the full panel.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ruc_extract_v2_features")


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #

def _bench_from_path(path: Path) -> str:
    """``caem/ruc/fresh_pool/fever_v2.json`` -> ``"fever"``."""
    stem = path.stem  # fever_v2
    if stem.endswith("_v2"):
        stem = stem[:-3]
    return stem


def _load_fresh_pool(glob_pattern: str) -> List[Dict[str, Any]]:
    """Load every fresh-pool JSON into a flat list of (id, benchmark, question)."""
    rows: List[Dict[str, Any]] = []
    for fp in sorted(glob.glob(glob_pattern)):
        bench = _bench_from_path(Path(fp))
        with open(fp) as f:
            items = json.load(f)
        for it in items:
            qid = it.get("id") or it.get("qid") or it.get("question_id")
            q = it.get("question") or it.get("query") or it.get("claim")
            if not q or qid is None:
                continue
            rows.append({"id": str(qid), "benchmark": bench, "question": q})
        logger.info("loaded %d questions from %s (bench=%s)", len(items), fp, bench)
    logger.info("fresh-pool total: %d questions across %d files",
                len(rows), len(set(r["benchmark"] for r in rows)))
    return rows


# --------------------------------------------------------------------------- #
# Component construction                                                       #
# --------------------------------------------------------------------------- #

def _passage_text(p: Any) -> str:
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        return p.get("text") or p.get("passage") or ""
    if isinstance(p, (tuple, list)) and p:
        first = p[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("text") or first.get("passage") or ""
    return ""


def _build_components(passage_index: str, device: str, pik_probe_path: Optional[Path] = None):
    """Construct the verifier-internal components + P(IK) probe used by v2
    feature extraction.

    Mirrors the verifier-side bundle built by
    ``scripts/rescore_baselines_through_verifier:_build_verifier`` (lines
    199-211), plus the Qwen base model + frozen P(IK) probe from
    ``scripts/ruc_train_pik_probe.py:_extract_hidden_states``.

    Returns (config, encoder, passage_store, cross_encoder, judge,
    alias_resolver, nlp, qwen_model, qwen_tok, pik_probe).
    """
    import torch
    from caem.config import CAEMConfig
    from caem.memory.encoder import QueryEncoder
    from caem.retrieval.rag import PassageStore
    from caem.verification import load_verifier_judge

    config = CAEMConfig()

    logger.info("loading QueryEncoder (mpnet-base) on %s", device)
    encoder = QueryEncoder(device=device)

    logger.info("loading PassageStore from %s", passage_index)
    passage_store = PassageStore.load(str(passage_index))
    logger.info("passage store loaded: %d passages", len(passage_store.passages))

    # CrossEncoder for rerank — mirrors _build_verifier exactly
    cross_encoder = None
    if config.cross_encoder_model:
        from sentence_transformers import CrossEncoder
        logger.info("loading CrossEncoder %s on %s",
                    config.cross_encoder_model, device)
        cross_encoder = CrossEncoder(
            config.cross_encoder_model, device=device,
            automodel_args=(
                {"torch_dtype": torch.bfloat16} if device != "cpu" else {}
            ),
        )

    # AdaptiveNLIJudge (or bare MiniCheck) — mirrors _build_verifier
    logger.info("loading verifier judge (AdaptiveNLIJudge or bare MiniCheck)")
    judge, nli_model, nli_tokenizer = load_verifier_judge(
        config, device, allow_fallback=True,
    )
    if judge is None:
        raise RuntimeError(
            "load_verifier_judge returned no judge (only the legacy RoBERTa "
            "path). v2 feature extraction expects the MiniCheck / Adaptive "
            "judge family for batch_entail_prob; cannot proceed."
        )

    # Optional Wikidata alias resolver
    alias_resolver = None
    alias_path = getattr(config, "alias_dict_path", None)
    if alias_path:
        ap = Path(alias_path)
        if ap.is_file():
            try:
                from caem.verification.alias_overlap import InMemoryAliasResolver
                table = json.loads(ap.read_text(encoding="utf-8"))
                alias_resolver = InMemoryAliasResolver(table=table)
                logger.info("alias_resolver loaded: %d canonical entries from %s",
                            len(table), ap)
            except Exception as exc:
                logger.warning(
                    "alias_dict load failed (%s) — entity_in_wikidata stays 0.0",
                    exc,
                )
        else:
            logger.info("alias_dict_path=%s not present — entity_in_wikidata stays 0.0",
                        ap)

    # spaCy for question entity extraction
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("en_core_web_sm not installed — entity_in_wikidata + top1_passage_entity_overlap stay 0.0")
        nlp = None

    # Qwen base model for P(IK) hidden states + frozen LR probe
    qwen_model = None
    qwen_tok = None
    pik_probe = None
    if pik_probe_path is not None and pik_probe_path.is_file():
        from transformers import AutoTokenizer, AutoModel
        import joblib
        logger.info("loading Qwen base model + tokenizer for P(IK) hidden states on %s", device)
        qwen_tok = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
        if qwen_tok.pad_token is None:
            qwen_tok.pad_token = qwen_tok.eos_token
        qwen_model = AutoModel.from_pretrained(
            config.base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        logger.info("Qwen loaded; loading P(IK) probe from %s", pik_probe_path)
        pik_probe = joblib.load(pik_probe_path)
        logger.info("P(IK) probe loaded (%s)", type(pik_probe).__name__)
    else:
        logger.info("no --pik_probe_path supplied; p_ik will stay at 0.5 neutral prior")

    return (config, encoder, passage_store, cross_encoder, judge,
            alias_resolver, nlp, qwen_model, qwen_tok, pik_probe)


# --------------------------------------------------------------------------- #
# Per-question feature extraction                                              #
# --------------------------------------------------------------------------- #

V2_FEATURE_NAMES: Tuple[str, ...] = (
    # Family C — retrieval top-1
    "top1_passage_sim",
    "top1_passage_entity_overlap",
    # Family C' — retrieval top-5 (v2 new)
    "top5_sim_max",
    "top5_sim_mean",
    "top5_sim_std",
    # Family C'' — cross-encoder rerank (v2 new)
    "rerank_top1",
    "rerank_top5_mean",
    "rerank_top5_std",
    # Family C''' — NLI pairwise cross-agreement on top-5 (v2 new)
    "nli_pair_mean",
    "nli_pair_min",
    "nli_pair_std",
    # Family B — model confidence
    "p_ik",
    # Family F — Wikidata entity-binary (v2 new)
    "entity_in_wikidata",
)


def _neutral_v2_features() -> Dict[str, float]:
    """Fallback when retrieval / rerank / NLI fails. Uses 0.5 for NLI
    (entailment prior) and 0.0 for missing similarity / rerank / Wikidata.
    """
    return {
        "top1_passage_sim": 0.0,
        "top1_passage_entity_overlap": 0.0,
        "top5_sim_max": 0.0,
        "top5_sim_mean": 0.0,
        "top5_sim_std": 0.0,
        "rerank_top1": 0.0,
        "rerank_top5_mean": 0.0,
        "rerank_top5_std": 0.0,
        "nli_pair_mean": 0.5,
        "nli_pair_min": 0.5,
        "nli_pair_std": 0.0,
        "p_ik": 0.5,
        "entity_in_wikidata": 0.0,
    }


def _retrieve_top_k(question: str, encoder, store, k: int) -> Tuple[List[str], List[float]]:
    try:
        emb = encoder.encode(question).astype(np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        results = store.search(emb, k=k)
    except Exception as exc:
        logger.warning("retrieve failed for q=%r: %s", question[:80], exc)
        return [], []
    passages: List[str] = []
    sims: List[float] = []
    for r in results:
        t = _passage_text(r)
        if not t:
            continue
        passages.append(t)
        # PassageStore returns (passage, score) tuples for compatibility with
        # the verifier closure; defensive extraction in case shape differs.
        if isinstance(r, (tuple, list)) and len(r) > 1:
            try:
                sims.append(float(r[1]))
            except Exception:
                sims.append(0.0)
        else:
            sims.append(0.0)
    return passages, sims


def _rerank_scores(question: str, passages: List[str], reranker) -> List[float]:
    if reranker is None or not passages:
        return []
    pairs = [(question, p) for p in passages]
    try:
        scores = reranker.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
    except Exception as exc:
        logger.warning("rerank failed for q=%r: %s", question[:80], exc)
        return []


def _pairwise_nli_scores(passages: List[str], judge) -> List[float]:
    """All ordered pairs (i, j) with i<j over top-K passages -> entail prob."""
    if judge is None or len(passages) < 2:
        return []
    premises, hypotheses = [], []
    for i in range(len(passages)):
        for j in range(i + 1, len(passages)):
            premises.append(passages[i])
            hypotheses.append(passages[j])
    try:
        arr = judge.batch_entail_prob(premises, hypotheses)
        return [float(x) for x in arr]
    except Exception as exc:
        logger.warning("pairwise NLI failed (%s) for %d pairs",
                       exc, len(premises))
        return []


def _entity_in_wikidata(question: str, nlp, alias_resolver) -> float:
    """Return 1.0 if any spaCy-extracted entity in the question resolves
    against the alias table. ``InMemoryAliasResolver.resolve(e)`` returns the
    (possibly empty) set of canonical / alias surface forms reachable from
    ``e``; a non-empty set means the entity is known to the table.
    """
    if alias_resolver is None or nlp is None:
        return 0.0
    try:
        doc = nlp(question or "")
        ents = [ent.text for ent in doc.ents]
        for e in ents:
            try:
                if alias_resolver.resolve(e):
                    return 1.0
            except Exception:
                continue
    except Exception as exc:
        logger.warning("spaCy entity extraction failed for q=%r: %s",
                       question[:80], exc)
    return 0.0


def _top1_passage_entity_overlap(question: str, top1_passage: str, nlp) -> float:
    """Fraction of spaCy-extracted entities from the question that appear
    (lower-case substring match) in the top-1 passage text. Same feature
    semantics as ``scripts/ruc_offline_passage_retrieval.py``.
    """
    if nlp is None or not question or not top1_passage:
        return 0.0
    try:
        doc = nlp(question)
        ents = [ent.text.lower() for ent in doc.ents]
        if not ents:
            return 0.0
        p_lower = top1_passage.lower()
        overlap = sum(1 for e in ents if e in p_lower)
        return float(overlap) / float(len(ents))
    except Exception as exc:
        logger.warning("top1_passage_entity_overlap failed for q=%r: %s",
                       question[:80], exc)
        return 0.0


def _compute_p_ik(question: str, qwen_model, qwen_tok, pik_probe, device: str) -> float:
    """Compute P(IK) for one question via Qwen last-token hidden state +
    frozen LR probe. Mirrors ``scripts/ruc_train_pik_probe.py:_extract_hidden_states``
    for the single-sample path. Returns 0.5 (neutral prior) if any component
    is None.
    """
    if qwen_model is None or qwen_tok is None or pik_probe is None:
        return 0.5
    try:
        import torch
        import numpy as np
        enc = qwen_tok(question, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = qwen_model(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)
        # Last non-padding token's hidden state (single sample so seq_len-1).
        seq_len = int(attn.sum(dim=1).item()) - 1
        hidden = out.last_hidden_state[0, seq_len].float().cpu().numpy().reshape(1, -1)
        # pik_probe is a CalibratedClassifierCV — use predict_proba
        prob = pik_probe.predict_proba(hidden)[0, 1]
        return float(prob)
    except Exception as exc:
        logger.warning("p_ik computation failed for q=%r: %s",
                       question[:80], exc)
        return 0.5


def extract_features(
    question: str,
    encoder, store, reranker, judge, alias_resolver, nlp,
    qwen_model=None, qwen_tok=None, pik_probe=None, device: str = "cuda",
    top_k: int = 5,
) -> Dict[str, float]:
    passages, sims = _retrieve_top_k(question, encoder, store, top_k)
    if not passages:
        return _neutral_v2_features()

    sims_arr = np.asarray(sims, dtype=np.float32) if sims else np.zeros(0)
    top1_passage_sim = float(sims_arr[0]) if sims_arr.size else 0.0
    top5_sim_max = float(sims_arr.max()) if sims_arr.size else 0.0
    top5_sim_mean = float(sims_arr.mean()) if sims_arr.size else 0.0
    top5_sim_std = float(sims_arr.std()) if sims_arr.size else 0.0

    top1_passage_entity_overlap = _top1_passage_entity_overlap(
        question, passages[0] if passages else "", nlp,
    )

    rerank = _rerank_scores(question, passages, reranker)
    if rerank:
        rerank_top1 = float(rerank[0])
        rerank_top5_mean = float(np.mean(rerank))
        rerank_top5_std = float(np.std(rerank))
    else:
        rerank_top1 = rerank_top5_mean = rerank_top5_std = 0.0

    nli_pairs = _pairwise_nli_scores(passages, judge)
    if nli_pairs:
        nli_pair_mean = float(np.mean(nli_pairs))
        nli_pair_min = float(np.min(nli_pairs))
        nli_pair_std = float(np.std(nli_pairs))
    else:
        nli_pair_mean = nli_pair_min = 0.5
        nli_pair_std = 0.0

    p_ik = _compute_p_ik(question, qwen_model, qwen_tok, pik_probe, device)
    wikidata_hit = _entity_in_wikidata(question, nlp, alias_resolver)

    return {
        "top1_passage_sim": top1_passage_sim,
        "top1_passage_entity_overlap": top1_passage_entity_overlap,
        "top5_sim_max": top5_sim_max,
        "top5_sim_mean": top5_sim_mean,
        "top5_sim_std": top5_sim_std,
        "rerank_top1": rerank_top1,
        "rerank_top5_mean": rerank_top5_mean,
        "rerank_top5_std": rerank_top5_std,
        "nli_pair_mean": nli_pair_mean,
        "nli_pair_min": nli_pair_min,
        "nli_pair_std": nli_pair_std,
        "p_ik": p_ik,
        "entity_in_wikidata": wikidata_hit,
    }


# --------------------------------------------------------------------------- #
# Main driver                                                                  #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--fresh_pool_glob", type=str,
                    default="caem/ruc/fresh_pool/*.json")
    ap.add_argument("--passage_index", type=str, default="data/passage_index")
    ap.add_argument("--out_parquet", type=Path,
                    default=Path("caem/ruc/v2/features.parquet"))
    ap.add_argument("--pik_probe_path", type=Path,
                    default=Path("caem/ruc/v2/pik_probe.joblib"),
                    help="Frozen P(IK) probe (joblib). Set to a non-existent path to "
                         "skip p_ik computation (feature stays at 0.5 neutral prior).")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rows (smoke test).")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    rows = _load_fresh_pool(args.fresh_pool_glob)
    if args.limit is not None:
        rows = rows[: args.limit]
        logger.info("smoke mode: truncated to %d rows", len(rows))

    # Build all components once; reuse across all questions.
    (
        _config, encoder, passage_store, cross_encoder, judge,
        alias_resolver, nlp, qwen_model, qwen_tok, pik_probe,
    ) = _build_components(args.passage_index, args.device,
                          pik_probe_path=args.pik_probe_path)

    out_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    last_log = t0
    for i, r in enumerate(rows):
        feats = extract_features(
            r["question"],
            encoder, passage_store, cross_encoder, judge,
            alias_resolver, nlp,
            qwen_model=qwen_model, qwen_tok=qwen_tok, pik_probe=pik_probe,
            device=args.device,
            top_k=args.top_k,
        )
        feats["id"] = r["id"]
        feats["benchmark"] = r["benchmark"]
        feats["question"] = r["question"][:500]  # truncated for parquet hygiene
        out_rows.append(feats)

        now = time.perf_counter()
        if now - last_log > 30.0 or (i + 1) == len(rows):
            elapsed = now - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            remaining = (len(rows) - (i + 1)) / max(rate, 1e-6)
            logger.info(
                "[%d/%d] elapsed=%.1fs rate=%.1fq/s ETA=%.1fmin",
                i + 1, len(rows), elapsed, rate, remaining / 60.0,
            )
            last_log = now

    # Write parquet
    import pandas as pd
    df = pd.DataFrame(out_rows)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    logger.info("wrote %d rows x %d cols -> %s",
                len(df), len(df.columns), args.out_parquet)

    # Print a quick sanity summary
    if len(df) >= 10:
        logger.info("v2 feature summary (first 5 rows):")
        for col in V2_FEATURE_NAMES:
            vals = df[col].astype(float)
            logger.info(
                "  %-22s  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                col, vals.mean(), vals.std(), vals.min(), vals.max(),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
