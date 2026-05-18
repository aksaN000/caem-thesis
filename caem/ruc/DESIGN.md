# RUC design — architecture, label, features

This document covers the *what* and *why* of the deployed RUC. For the *current state* and roadmap, see README.md and PLAN.md.

---

## 1. Goal

For each query that misses Tier 1 (episodic memory exact match), CAEM's router has historically had two T3-forcing branches:

1. **Safety veto**: `u_pre < safety_threshold` (the pre-routing confidence is too low to trust direct generation).
2. **Score-formula fall-through**: combined memory score < 0.90 and similarity ≤ 0.75 (no usable memory match).

Both branches default to Tier 3 (RAG). Empirically (from the CAEM trajectory on the 5-bench panel), **RAG hurts on 3 of 5 benches** — TruthfulQA loses 19.7 EM points to RAG, CSQA 13.0, StrategyQA 4.0. Always-T3 trades a help on TriviaQA against three losses on the others.

The RUC's job is to **intercept those two T3-default branches** and re-route the queries where RAG is predicted to hurt (or be neutral) to Tier 2 instead, while keeping the queries where RAG genuinely helps on the T3 path.

The label is per-sample, not per-benchmark. The RUC does not need to know which benchmark a query came from; it only needs to predict, from features available pre-routing, whether T3 will improve over T2 on this specific query.

---

## 2. Label decomposition — A, B, C, D

Each query has both a direct (T2-style) answer and a RAG (T3-style) answer when we build training data. EM scoring of each yields one of four cases:

| Case | direct_em | rag_em | Meaning | Training label | Used in training? |
|---|:---:|:---:|---|:---:|:---:|
| **A** | 0 | 1 | RAG fixed a direct error | **1** ("send to T3") | ✅ yes |
| **B** | 1 | 0 | RAG broke a correct direct | **0** ("keep at T2") | ✅ yes |
| C | 1 | 1 | both right (tied EM) | tiebreak = noise | ❌ DROPPED |
| D | 0 | 0 | both wrong (tied EM) | tiebreak = noise | ❌ DROPPED |

The historical v0 design used a CHM-tiebreak (composite hallucination metric) to label C and D rows. Empirically that injected noise — 65 % of v0 labels were CHM-decided, pulling pooled CV AUROC down to ~0.55. Dropping C and D from training lifted AUROC to 0.65 on the same feature set.

**Path B is the locked label strategy**: train only on A and B; let the classifier learn from clean EM-decided signal.

At deployment, the classifier scores all queries (including those that would have been ties). The threshold τ converts the score to a binary decision. Tie rows simply don't contribute to training loss; they get a probability based on whatever the classifier learned from A and B, and the threshold decides their routing.

---

## 3. Features (24 dims, all pre-routing-legal)

The full feature list, in five families. Every feature is computable BEFORE Tier 2 generates an answer — that's the architectural constraint that makes the RUC a pre-routing gate rather than a post-T2 reranker.

### Family A — surface form (14 dims)

| Feature | Source |
|---|---|
| `q_token_len` | `len(question.split())` |
| `negation_present` | regex `\b(not\|never\|no\|none\|nobody\|...)` |
| `temporal_cue` | regex `recently\|currently\|now\|(19\|20)\d{2}\|...` |
| `numerical_cue` | regex `\d+\|one\|two\|...\|million` |
| `myth_regex_hit` | regex `is it true\|do people believe\|common misconception\|...` |
| `interrog_who/what/when/where/why/how/yesno` | first-word match + yes-no auxiliary prefix detector → 7 one-hot dims |
| `entity_count` | `len(spacy_doc.ents)` from `en_core_web_sm` |
| `log_pageviews_max` | max of log10(monthly pageviews) over question entities, via Wikimedia REST API |

### Family B — model confidence (1 dim)

| Feature | Source |
|---|---|
| `p_ik` | Logistic-regression probe (Kadavath 2022 style) on Qwen-2.5-3B-Instruct last-layer hidden state, trained to predict `direct_em`. 5-fold OOF AUROC 0.6585. Artefact at `caem/ruc/pik_probe_v1canon.joblib`. |

### Family C — retrieval signals (2 dims, pre-routing-legal because the FAISS top-1 lookup is cheap and reused by T3 if chosen)

| Feature | Source |
|---|---|
| `top1_passage_sim` | cosine similarity between question's `QueryEncoder` (mpnet-base, 768-dim) embedding and the top-1 passage from FAISS over the 21M-passage Wikipedia index |
| `top1_passage_entity_overlap` | fraction of question entities (spaCy NER) appearing in the top-1 passage text |

### Family D — linguistic patterns (3 dims, target the "RAG hurts" direction)

| Feature | Source |
|---|---|
| `is_adversarial_phrasing` | regex matches `is it (true\|safe\|...)\|do most people\|can you actually\|what happens if you\|...` — TruthfulQA myth archetype |
| `is_opinion_seeking` | regex `should\|would\|prefer\|opinion\|...` — CSQA / StrategyQA pattern |
| `is_open_ended` | regex `why\|how to\|describe\|explain\|...` — commonsense reasoning markers |

### Family E — answer-type match (4 dims, cheap interaction features)

| Feature | Source |
|---|---|
| `q_expects_year` | regex `what year\|when (did\|was\|were\|is)` on question |
| `passage_has_year` | regex `(19\|20)\d{2}` on top-1 passage |
| `year_match` | `q_expects_year × passage_has_year` |
| `q_expects_number`, `passage_has_number`, `number_match` | parallel structure for "how many / how much" |

### Top feature coefficients (after StandardScaler; v1 deployment model)

| Sign | Feature | Coef | Interpretation |
|:---:|---|---:|---|
| − | `interrog_yesno` | −0.54 | yes/no questions: RAG rarely helps |
| − | `interrog_how` | −0.50 | "how" questions (commonsense): RAG rarely helps |
| − | `myth_regex_hit` | −0.33 | myth pattern: RAG injects misleading content |
| − | `is_adversarial_phrasing` | −0.28 | TruthfulQA archetype: RAG hurts |
| **+** | **`top1_passage_sim`** | **+0.27** | strong retrieval candidate: RAG helps |
| − | `entity_count` | −0.26 | many entities (multi-hop): RAG less reliable |
| **+** | `numerical_cue` | +0.26 | factoid-with-numbers: RAG helps |
| − | `negation_present` | −0.22 | negations (often adversarial): RAG hurts |

---

## 4. Model

Logistic Regression with L2 regularisation + StandardScaler.

- 24 input features → 1 output (`p_rag`)
- Regularisation `C = 0.005` for v1 (heavy; chosen via 5-fold-CV gap diagnostic on n = 519)
- Decision threshold `τ = 0.40` for v1 (chosen via post-hoc utility sweep on the v1 probability distribution)
- Both `C` and `τ` are scale-specific and will be retuned for v2 (see PLAN.md § v2 retune contract)

Why LR over LightGBM:
- At v1 scale (519 rows), LightGBM with C-equivalent default settings showed train/CV AUROC gap of 0.07 (mild overfit), while LR with `C = 0.005` closes the gap to 0.013.
- Linear heads on hand-engineered + cheap retrieval features beat tree ensembles on small datasets when the features are mostly monotonic in the target.
- LR matches the architectural choice in the published RAG-routing literature (Adaptive-RAG, SR-RAG, Self-Routing-RAG).
- LR coefficients are interpretable; LightGBM SHAP is heavier and harder to ship in a runtime predict path.

---

## 5. Where the RUC fires in the router

```
ROUTE(query):
  encode → memory_search → get (similarity_top1, u_stored_top1) and u_pre

  ┌── MECHANISM 1: SAFETY VETO (evaluated first)
  │   if u_pre < safety_threshold:
  │       → consult RUC
  │           if p_rag ≥ τ:    → Tier 3
  │           else:             → Tier 2
  │       (default Tier 3 when RUC not wired)
  │
  └── MECHANISM 2: ROUTING SCORE
        score = 0.70 · similarity + 0.30 · u_stored
        if score ≥ 0.90:        → Tier 1   (true memory exact match — RUC SKIPPED)
        elif similarity > 0.75: → Tier 2   (memory near-hit       — RUC SKIPPED)
        else:                   → consult RUC (fall-through)
            if p_rag ≥ τ:    → Tier 3
            else:             → Tier 2
            (default Tier 3 when RUC not wired)
```

The RUC has exactly two fire points — both branches that previously hardcoded Tier 3.

### Cost at the RUC decision point (per query)

| Step | Cost |
|---|---|
| 14 surface features (regex + NER + interrogatives) | ~1 ms CPU |
| FAISS top-1 search on 21M-passage IVF index | ~30-50 ms CPU (page-cached) |
| 2 retrieval features (sim, entity overlap) | included in the search step |
| P(IK) Qwen forward pass on question hidden state | ~10 ms GPU |
| 3 linguistic + 4 answer-type regex features | ~1 ms CPU |
| StandardScaler.transform + LR.predict_proba | < 1 ms CPU |
| **Total per RUC consultation** | **~50–70 ms** |

The top-1 FAISS search is reused by Tier 3 if T3 is the verdict (no double retrieval cost). The only "wasted" work is on queries the RUC sends to T2: we did the cheap top-1 search but didn't use it. That's the cost of avoiding a much more expensive Tier 3 call where RAG would have hurt.

---

## 6. Validation methodology

5-fold stratified random CV, on clean A+B rows only, balanced by case label.

- Outer fold sizes: ~104 rows per fold from 519 total
- Per-bench fold sizes vary by class balance — TruthfulQA has 26 rows per fold, FEVER 25, TriviaQA 20, CSQA 19, StrategyQA 14 (small)
- AUROC reported is pooled OOF
- Per-bench AUROC computed by partitioning the OOF predictions by benchmark
- Deployment utility metric: `A_caught − B_caught` at threshold τ on the full 1500-row pool (after re-fitting the model on all 519 clean rows). Tie rows are evaluated for "did we send them to T3 or not", but they don't contribute to utility because by definition they tie.

For v2 the protocol upgrades to **nested 5-fold CV** so both the regularisation `C` and the decision threshold `τ` can be tuned without leaking validation data into hyperparameter selection. See PLAN.md.
