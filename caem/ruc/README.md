# CAEM Retrieval Utility Classifier (RUC)

The RUC is a frozen, pre-routing classifier that sits inside `caem.routing.AdaptiveRouter` at the two Tier-3 dispatch points. It decides per query whether to commit to the expensive Tier-3 RAG path or fall back to Tier-2 parametric chain-of-thought. The architectural goal: route the queries where RAG genuinely helps to Tier 3, keep the queries where RAG injects hallucination at Tier 2.

This README documents the current deployment (v1 shortcut, 24 features, LR + StandardScaler) and the planned v2 (leakage-free, fresh-pool trained). Everything you need to load and call the RUC at inference is at `caem/ruc/v1_shortcut/`.

---

## Current state — v1 shortcut

- **Model**: Logistic Regression + StandardScaler
- **Features**: 24 dense pre-routing features (see `feature_spec.json`)
- **Label**: per-sample binary, derived from EM only (case A → 1, case B → 0, ties dropped)
- **Regularisation**: C = 0.005 (tuned to close the train/CV AUROC gap diagnosed at C = 1.0)
- **Decision threshold**: τ = 0.40
- **Validation**: 5-fold stratified random CV on 519 clean A+B rows from the canonical pairing
- **Pooled CV AUROC**: 0.647
- **Per-bench CV AUROC**: TruthfulQA 0.65, StrategyQA 0.66, CSQA 0.59, TriviaQA 0.58, FEVER 0.44
- **Deployment utility at τ = 0.40 on the 1500-row pool**: +33 vs always-T3 baseline of −83 (a +116 swing)
- **Leakage status**: trained on canonical-pool samples that are content-shared with CAEM's eval pool — **suitable for methodology illustration, not for headline Ch6 numbers** (see v2 plan in PLAN.md)

### Repo layout

```
caem/ruc/
├── README.md                              # this file (overview)
├── DESIGN.md                              # architecture, label decomposition, features, math
├── PLAN.md                                # v2 build plan + status
├── training_set_v1_canonical_shortcut.parquet   # source-of-truth v1 training data (1500 rows)
├── pik_probe_v1canon.joblib               # frozen P(IK) probe
├── qwen_hidden_states_v1canon.npy         # cached Qwen hidden states (reusable at v2)
├── passage_features.parquet               # cached top-1 FAISS features (reusable at v2)
├── wiki_pageviews_cache.jsonl             # cached Wikipedia pageviews
├── sonnet_labels.jsonl                    # v1 Sonnet labels (kept for reference, unused by deployed LR)
├── fresh_pool/                            # 9004 content-hash-disjoint samples for v2
│   ├── fever_v2.json
│   ├── triviaqa_v2.json
│   ├── commonsense_qa_v2.json
│   ├── strategyqa_v2.json
│   ├── truthfulqa_v2.json
│   ├── haluevalqa_v2.json
│   └── openbookqa_v2.json
├── results/                               # CV result dumps from training scripts
├── v1_shortcut/                           # DEPLOYED v1 RUC artefacts
│   ├── feature_spec.json                  # spec with hyperparameters + utility receipt
│   └── models/
│       ├── scaler.joblib                  # StandardScaler
│       └── lr_final.joblib                # LogisticRegression(C=0.005)
└── v1_archive/                            # retired intermediate experiments
    └── intermediate/                      # legacy LightGBM, abstention experiment, etc.
```

### How to load and call the deployed RUC

```python
from pathlib import Path
from caem.routing.ruc import LRRetrievalUtilityClassifier

ruc = LRRetrievalUtilityClassifier.from_spec(Path("caem/ruc/v1_shortcut/feature_spec.json"))

decision = ruc.predict(
    question="What happens if you crack your knuckles?",
    top1_passage_text="...",         # pipeline supplies via FAISS top-1 lookup
    top1_passage_sim=0.42,
    top1_passage_entity_overlap=0.0,
    p_ik=0.5,
    source_benchmark="truthfulqa",   # optional, for logging
)
# decision.decision in {"RAG", "DIRECT"}
# decision.p_rag in [0, 1]
# decision.threshold = 0.40
```

The classifier is wired into `caem.routing.AdaptiveRouter` and fires at the two Tier-3 entry points (safety veto and score-formula fall-through). See `caem/routing/router.py:_consult_ruc` for the integration.

---

## Why v2 is needed (leakage caveat)

The v1 1500 training rows are drawn from `caem.benchmark_splits.build_all_benchmark_pools(rng_seed=42)[bench].eval` — the same eval pool CAEM uses to score its trajectory. Training the RUC on those samples and then deploying it inside CAEM evaluated on the same pool would be label leakage. v1 is therefore a methodology demonstration only.

**v2 is trained on a content-hash disjoint fresh pool** (`caem/ruc/fresh_pool/`) — 9 004 unique questions across 7 benchmarks, with zero overlap against any CAEM pool by construction. See PLAN.md for the v2 build plan and current status.
