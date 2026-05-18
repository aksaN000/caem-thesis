#!/usr/bin/env python3
"""scripts/ruc_merge_features.py
====================================
After Phase 1 enrichment jobs finish, merge their outputs back into the main
RUC training parquet, overwriting the stubbed-zero feature columns:

  - top1_passage_sim, top1_passage_entity_overlap
        from caem/ruc/passage_features.parquet
        (produced by scripts/ruc_offline_passage_retrieval.py)

  - log_pageviews_max
        recomputed using data/ruc/pageviews.parquet
        (produced by scripts/ruc_build_wiki_pageviews.py)

  - qtype_factoid / qtype_commonsense / qtype_multihop / qtype_boolean / qtype_myth
        already overwritten in-place by scripts/ruc_train_qtype_svm.py;
        this script re-verifies they look right.

Usage
-----
    python -m scripts.ruc_merge_features \\
        --in_parquet caem/ruc/training_set.parquet \\
        --out_parquet caem/ruc/training_set.parquet  (overwrites)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _merge_passage_features(df, passage_features_path: Path):
    """Overwrite top1_passage_sim + top1_passage_entity_overlap in df."""
    if not passage_features_path.exists():
        logger.warning("passage features missing at %s; skipping",
                       passage_features_path)
        return df, 0
    import pandas as pd
    pf = pd.read_parquet(passage_features_path)
    pf = pf[["id", "benchmark", "top1_passage_sim",
             "top1_passage_entity_overlap"]].copy()
    pf["id"] = pf["id"].astype(str)
    df["id"] = df["id"].astype(str)

    df = df.drop(columns=[c for c in ["top1_passage_sim",
                                       "top1_passage_entity_overlap"]
                          if c in df.columns])
    merged = df.merge(pf, on=["id", "benchmark"], how="left")
    n_filled = merged["top1_passage_sim"].notna().sum()
    # NaN-fill any rows we didn't have features for
    merged["top1_passage_sim"] = merged["top1_passage_sim"].fillna(0.0)
    merged["top1_passage_entity_overlap"] = merged["top1_passage_entity_overlap"].fillna(0.0)
    return merged, n_filled


def _recompute_pageviews(df, pageviews_path: Path):
    """Recompute log_pageviews_max per row using the offline lookup table."""
    if not pageviews_path.exists():
        logger.warning("pageviews table missing at %s; skipping",
                       pageviews_path)
        return df, 0
    import pandas as pd
    pv_df = pd.read_parquet(pageviews_path)
    pv: Dict[str, float] = dict(zip(pv_df["entity"].str.lower(),
                                    pv_df["log_pageviews"]))

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])

    new_vals = []
    n_filled = 0
    for q in df["question"].fillna("").tolist():
        doc = nlp(q)
        mx = 0.0
        for ent in doc.ents:
            v = pv.get(ent.text.lower())
            if v is not None and v > mx:
                mx = float(v)
        if mx > 0.0:
            n_filled += 1
        new_vals.append(mx)
    df["log_pageviews_max"] = new_vals
    return df, n_filled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--out_parquet", type=Path,
                    default=Path("caem/ruc/training_set.parquet"))
    ap.add_argument("--passage_features", type=Path,
                    default=Path("caem/ruc/passage_features.parquet"))
    ap.add_argument("--pageviews_table", type=Path,
                    default=Path("data/ruc/pageviews.parquet"))
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    import pandas as pd
    df = pd.read_parquet(args.in_parquet)
    logger.info("Loaded %d rows from %s", len(df), args.in_parquet)

    # 1. Merge passage features
    df, n_pf = _merge_passage_features(df, args.passage_features)
    logger.info("Merged passage features: %d rows received real values", n_pf)

    # 2. Recompute pageviews
    df, n_pv = _recompute_pageviews(df, args.pageviews_table)
    logger.info("Recomputed log_pageviews_max: %d rows got non-zero", n_pv)

    # 3. qtype is already in the parquet from the SVM script
    qtype_cols = [c for c in df.columns if c.startswith("qtype_")]
    logger.info("qtype columns present: %s", qtype_cols)

    df.to_parquet(args.out_parquet, index=False)

    # Final summary
    print(f"\n{'='*68}")
    print(f"Enriched RUC training set  ->  {args.out_parquet}")
    print(f"{'='*68}")
    print(f"  rows:                                {len(df)}")
    print(f"  top1_passage_sim non-zero:           {(df['top1_passage_sim']!=0.0).sum()} / {len(df)}")
    print(f"  top1_passage_entity_overlap non-zero: {(df['top1_passage_entity_overlap']!=0.0).sum()} / {len(df)}")
    print(f"  log_pageviews_max non-zero:           {(df['log_pageviews_max']!=0.0).sum()} / {len(df)}")
    print(f"  qtype_factoid mean:                   {df.get('qtype_factoid', pd.Series([0])).mean():.3f}")
    print(f"  qtype_myth mean:                      {df.get('qtype_myth', pd.Series([0])).mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
