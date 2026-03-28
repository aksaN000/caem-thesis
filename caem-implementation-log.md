# CAEM Implementation Log

> **Auto-updated every session.** Each entry records: what was built, design decisions made (with alternatives considered and pros/cons), and the rationale. This file is the authoritative record of implementation choices — read it before modifying any module.
>
> Format: newest session at the top within each module section.

---

## How to read this log

- **[LIT]** = value fixed from literature — cite the paper, don't change arbitrarily.
- **[DES]** = design choice — principled default, change only with ablation evidence.
- **[CAL]** = empirically calibrated — initial value given; actual value comes from Cycle 1 calibration set.
- **Decision boxes** record the alternative that was considered and why it was rejected. These are the answers to "why didn't you just…?" committee questions.

---

## Session 11 — 2026-03-29

**Scope:** Episodic memory module (Sessions 1–10 completed design; Session 11 begins implementation.)

**Files created this session:**
```
caem/__init__.py
caem/config.py
caem/memory/__init__.py
caem/memory/entry.py
caem/memory/encoder.py
caem/memory/store.py
tests/__init__.py
tests/test_episodic_memory.py
```

**Test result:** 37/37 passing (no GPU, no SBERT model — all synthetic embeddings).

---

### Module: `caem/config.py`

**Purpose:** Single source of truth for all hyperparameters. Every numeric constant in the system must come from here — no magic numbers in module code.

**Design choice:** All hyperparameters annotated with their category ([LIT]/[DES]/[CAL]) inline. This makes Chapter 4 writing mechanical — the annotation tells you exactly how to write about each value.

**Key values set:**

| Hyperparameter | Value | Category | Source / rationale |
|---|---|---|---|
| `embedding_dim` | 384 | [LIT] | all-mpnet-base-v2 output dim. NOT 768. |
| `max_memory_size` | 20,000 | [DES] | ≈20 MB metadata + 30 MB FAISS. GPU budget allows it. |
| `pruning_trigger` | 0.95 | [DES] | Prune when 95% full to avoid hard-stop failures. |
| `pruning_amount` | 0.20 | [DES] | Remove bottom 20% by Value score. |
| `novelty_threshold` | 0.95 | [DES] | Skip storage if nearest sim > 0.95 (already known). |
| `tier1_combined_threshold` | 0.90 | [DES] | High-confidence recall gate. |
| `tier2_similarity_threshold` | 0.75 | [DES] | Medium-similarity knowledge adaptation zone. |
| `safety_u_pre_min` | 0.60 | [DES] | OR-condition: force Tier 3 below this. |
| `u_hat_weight_*` | 0.25 each | [CAL] | **Initial** equal weights. Post-calibration projected ~0.20/0.20/0.20/0.40. Do NOT start at the projected values. |
| `u_hat_accept_threshold` | 0.60 | [DES] | Asymmetric cost — prefer false negatives over storing bad answers. |
| `l2_lambda` | 0.01 | [LIT] | Adapted from Kirkpatrick et al. 2017 (EWC); simplified to L2 here. |
| `mc_dropout_k` | 5 | [LIT] | Gal & Ghahramani 2016. |
| `se_samples_k` | 10, T=1.0 | [LIT] | Farquhar et al. 2024. |
| `sc_chains_m` | 3 | [LIT] | Wang et al. 2022 (diminishing returns past 10 — 3 is sufficient for the pre-gate). |

---

### Module: `caem/memory/entry.py`

**Purpose:** Data schemas — `EpisodicEntry`, `PreRoutingConfidence`, `PostGenerationConfidence`, `StoredConfidence`, `RoutingDecision`.

**Key design decision — immutable/mutable split:**

> **Decision:** Content fields (`question`, `reasoning_chain`, `answer`, `embedding`, `storage_cycle`, `timestamp`) are immutable after storage. Quality fields (`u_stored`, `nli_score`, `sc_score`, `se_score`, `retrieval_count`, `success_rate`, `retroverified`) are mutable.
>
> **Why:** The fine-tuning dataset (Stage 8) is derived directly from content fields. Mutating them mid-cycle would mean training on a moving target — the audit trail would be broken and training instability would result. See: writing-suggestions.md C4-15.
>
> **Alternative considered:** Fully mutable entries. Rejected because: mid-cycle content updates would corrupt the verified training set and make results non-reproducible.

**Key design decision — three distinct confidence types:**

> **Decision:** Defined as three separate dataclasses (`PreRoutingConfidence`, `PostGenerationConfidence`, `StoredConfidence`) rather than a single `Confidence` dataclass.
>
> **Why:** Prevents the most common implementation error in confidence-aware systems — accidentally using û (Stage 4a) where û_stored (Stage 7) is required, or vice versa. Tier 3 answers never produce a û value at all. Separate types make this impossible to confuse at the type level. See: writing-suggestions.md C4-03, C4-08.
>
> **Alternative considered:** Single `Confidence` dataclass with optional fields. Rejected because: optional fields silently allow the wrong confidence value to propagate — errors would surface late in training, not at the call site.

**Key design decision — `is_safe()` on `PreRoutingConfidence`:**

> **Decision:** The OR-condition (u_pre < 0.60 → force Tier 3) is implemented as a method on `PreRoutingConfidence`, not as a formula combined with the routing score.
>
> **Why:** The OR-condition and the routing score formula are two SEPARATE mechanisms. Combining them into one formula would allow a high û_stored to arithmetically compensate for a dangerously low u_pre. A high-quality memory match to a query the model doesn't understand would still route to Tier 1. The separation makes both conditions mandatory, not tradeable. See: writing-suggestions.md C4-10b.

---

### Module: `caem/memory/encoder.py`

**Purpose:** Wraps Sentence-BERT to produce 384-dim L2-normalised embeddings.

**Key design decision — lazy loading:**

> **Decision:** The SBERT model is loaded on first `encode()` call, not at `__init__` time.
>
> **Why:** Avoids GPU memory allocation when the encoder is instantiated in test/config contexts that don't actually run inference. Allows unit tests to run without downloading the 420 MB model.

**Key design decision — post-encode re-normalisation check:**

> **Decision:** After `sentence_transformers.encode(normalize_embeddings=True)`, we do a secondary norm check and re-normalise if any vector deviates from 1.0 by > 1e-5.
>
> **Why:** Library version quirks (particularly with older sentence-transformers builds) have produced non-unit vectors despite the flag. FAISS inner product only equals cosine similarity if vectors are exactly unit-norm — a silent deviation would corrupt all similarity scores. Belt-and-suspenders check costs ~1 µs per batch.

**Key design decision — dimension validation at load time:**

> **Decision:** After loading the model, we immediately encode a probe string and verify the output dim is 384.
>
> **Why:** The 384-vs-768 confusion (mpnet vs bert-base) is the single most common setup error in SBERT projects. Catching it at load time with a clear error message saves hours of debugging. The FAISS index would silently accept wrong-dim vectors and return garbage similarity scores.

---

### Module: `caem/memory/store.py`

**Purpose:** FAISS-backed episodic memory with add, search, prune, retroverify, and save/load.

#### ⚠️ Critical design decision — FAISS backend choice

> **Thesis spec:** `IndexIVFPQ` (IVF-PQ for memory efficiency at 20k scale).
>
> **Implemented:** `IndexIDMap(IndexFlatIP)` — exact inner-product search with ID mapping.
>
> **Why the deviation:**
>
> | Criterion | IVF-PQ | FlatIP + IDMap (chosen) |
> |---|---|---|
> | Memory (20k × 384) | ~4 MB (8× compression) | ~30 MB |
> | Search accuracy | Approximate (AUROC loss ~2–3%) | Exact |
> | Training required | Yes — needs ≥ nlist (100) vectors | No |
> | `remove_ids()` support | Requires IDMap2 wrapper; complex | Native with IDMap |
> | Cold-start behaviour | Fails until nlist entries exist | Works from entry 1 |
> | GPU VRAM impact | Saves 26 MB | 30 MB (< 0.2% of 15 GB budget) |
>
> **Conclusion:** The 26 MB memory saving is negligible against the 15–19 GB training budget. The correctness benefits (exact search, no training requirement, clean remove_ids) are significant. For a correctness-critical research system, exact search is the right default.
>
> **How to switch to IVF-PQ if needed:** Replace `faiss.IndexFlatIP(dim)` with `faiss.IndexIVFPQ(faiss.IndexFlatL2(dim), dim, 100, 8, 8)` wrapped in `IndexIDMap2`. Add a training step before the first `add()` call once ≥ 100 entries exist. This would be the right choice if memory grew past ~200k entries.
>
> **Thesis writing note:** Present FlatIP in the implementation chapter as "exact cosine search via inner product on L2-normalised vectors" — IVF-PQ was the plan-stage proposal for a larger deployment; the implemented scale doesn't require approximation.

**Key design decision — retroverify only updates upward:**

> **Decision:** If retroactive re-verification produces a new u_stored lower than the stored value (but still above the prune threshold), the stored value is kept. The episode is marked `retroverified=True` to record that it was checked.
>
> **Why:** An episode that passed verification at storage time is not invalidated by a marginal score decrease in a later cycle. Downgrading would penalise good episodes from earlier cycles that the updated model happens to be slightly less confident about. Only a drop below the prune threshold (0.50) warrants removal. This is a conservative design — it errs toward retaining knowledge.
>
> **Alternative considered:** Always update to the new score (bidirectional). Rejected because: it would progressively downgrade correct early-cycle episodes as the model's confidence distribution shifts, reducing Tier 1 hit rate without improving accuracy.

**Key design decision — Value score for pruning:**

```
Value = 0.7 · Importance + 0.3 · Recency
Importance = 0.4·φ + 0.3·(r/r_max) + 0.2·success_rate + 0.1·u_stored
Recency = exp(−0.01 · age_in_seconds)
```

> **Why these weights [DES]:** φ (cycle recency proxy) gets the highest importance weight because later-cycle episodes were generated by a better model and verified by a stronger verifier — they are structurally more reliable. Retrieval frequency (r/r_max) captures usefulness. success_rate captures quality-in-use. u_stored gets the lowest importance weight because it was already used as the storage filter (threshold 0.75) — surviving entries are already high-quality on this dimension.
>
> **Alternative considered:** Pure u_stored ranking. Rejected because: it ignores usefulness (retrieval frequency) and recency, biasing pruning toward older verified-but-rarely-used episodes that may still contain unique knowledge.

---

## Pipeline Implementation Roadmap

**Status legend:** ✅ Done | 🔨 Next | ⬜ Pending

| Module | Stage | Dataset needed? | Status |
|---|---|---|---|
| `caem/memory/` | — | No | ✅ Session 11 |
| `caem/confidence/pre_routing.py` | Stage 3 | No (model only) | 🔨 Session 12 |
| `caem/routing/router.py` | Stage 3→dispatch | No | ⬜ |
| `caem/confidence/post_generation.py` | Stage 4a | No (model only) | ⬜ |
| `caem/verification/verifier.py` | Stage 5 | ⚠️ NLI layer needs ground truth | ⬜ |
| `caem/retrieval/rag.py` | Stage 6 (Tier 3) | Yes — Wikipedia passages | ⬜ |
| `caem/training/self_improvement.py` | Stage 8 | ✅ Yes — all 4 benchmarks | ⬜ |
| Dataset loaders | — | Yes — HotpotQA/StrategyQA/FEVER/TruthfulQA | ⬜ |
| Calibration harness | Post-Cycle 1 | Yes — 500-sample calibration set | ⬜ |
| Evaluation harness | Post-cycle | Yes — test splits | ⬜ |

---

## Dataset Requirements

> See "When do we need datasets?" section for the full breakdown.

| Dataset | Source | When first needed | Split sizes needed |
|---|---|---|---|
| HotpotQA | HuggingFace `hotpot_qa` | Stage 5 (NLI verifier) | train: 5k/cycle × 3; val: 1k (calibration + purity); test: eval |
| StrategyQA | HuggingFace `boolq` / official | Stage 5 | Same structure |
| FEVER | HuggingFace `fever` | Stage 5 | Same structure + 3-class labels |
| TruthfulQA | HuggingFace `truthful_qa` | Stage 5 + VE2 misconception list | Smaller dataset — full use |
| Wikipedia passages | HuggingFace `wiki_dpr` or similar | Stage 6 (Tier 3 RAG) | Dense retrieval index |

---

## Open implementation questions

| # | Question | Relevant when |
|---|---|---|
| IQ-01 | How to handle Tier 3 RAG retrieval? DPR vs BM25 vs simple TF-IDF? | Stage 6 |
| IQ-02 | Temperature scaling calibration: L-BFGS on ECE — use `netcal` library or implement from scratch? | Post-Cycle 1 |
| IQ-03 | StrategyQA: use the official dataset (discontinued) or HuggingFace `boolq` substitute? | Dataset loading |
| IQ-04 | VE2 misconception list: scrape from TruthfulQA repo or hardcode the 38 categories? | Verifier |
| IQ-05 | Multi-GPU: is the 15–19 GB VRAM budget for a single A100, or distributed? | Training loop |
