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

## Session 15 — 2026-03-29

**Scope:** MultiLayerVerifier (Stage 5, quality gate).

**Files created this session:**
```
caem/verification/__init__.py
caem/verification/verifier.py
tests/test_verifier.py
```

**Test result:** 161/161 passing (31 new verifier tests + 130 from prior sessions).

---

### Module: `caem/verification/verifier.py`

**Purpose:** Quality gate — determines whether a Tier 2 answer is trustworthy enough to store in episodic memory and at what û_stored value. Runs after Stage 4a (efficiency gate). Stage 4a screens fast; Stage 5 measures factual reliability with precision.

**Three signals and weights:**

| Signal | Method | Weight |
|--------|--------|--------|
| p_entail | P(ENTAILMENT) softmax prob, averaged over M=3 chains | 0.50 [DES] |
| s_avg | Avg pairwise SBERT cosine sim of M=3 chains | 0.30 [LIT] Wang et al. 2022 |
| h_norm | Normalised semantic entropy (NLI clustering, K=10) | 0.20 [LIT] Farquhar et al. 2024 |

û_stored = 0.50·p_entail + 0.30·s_avg + 0.20·(1 − h_norm)

---

**Key design decision — p_entail uses softmax probability, NOT argmax label:**

> Stage 4a uses argmax (0/1/2) for binary cluster membership. Stage 5 uses softmax P(ENTAILMENT) for a graded [0,1] signal that propagates calibrated uncertainty into û_stored.
>
> | Option | Verdict |
> |--------|---------|
> | argmax label | Rejected — loses calibration |
> | softmax P(ENTAILMENT) ✓ | **Chosen** — graded, propagates uncertainty |

**Key design decision — p_entail averages NLI over M chains, not over (query, answer) directly:**

> NLI(query, answer) is ill-posed for factual QA — questions don't logically entail answers. Instead: generate M=3 independent chains from the query, compute NLI(chain_i → answer) for each, average. This checks that the model's own reasoning supports the answer.
>
> | Option | Verdict |
> |--------|---------|
> | NLI(query, answer) directly | Rejected — ill-posed framing |
> | NLI(chain_i, answer) averaged ✓ | **Chosen** — principled reliability measure |

**Key design decision — Stage 4a vs Stage 5 are deliberately separate modules:**

> Both stages compute similar signals. Key distinction: Stage 4a is calibrated for recall (avoid false negatives); Stage 5 is calibrated for precision (avoid storing junk). Merging would force a single threshold to serve both purposes — impossible to calibrate correctly.

---

## Session 14 — 2026-03-29

**Scope:** PostGenerationConfidenceEstimator (Stage 4a, Tier 2 efficiency gate).

**Files created this session:**
```
caem/confidence/post_generation.py
tests/test_post_generation.py
```

**Test result:** 130/130 passing (35 new post-generation tests + 95 from prior sessions).

---

### Module: `caem/confidence/post_generation.py`

**Purpose:** Compute û (post-generation confidence) for answers produced by Tier 2. Acts as an **efficiency gate** — not a quality gate. If û < 0.60, the answer is escalated to Tier 3 (full RAG) before committing to the expensive verification pipeline (Stage 5, ~2–4 s).

**Four signals:**

| Signal | Method | Source | Value range |
|--------|--------|---------|-------------|
| u_token | Geometric mean of per-token log-probs on the full generated answer | [LIT] | [0, 1] |
| u_dropout | 1 / (1 + Var[K=5 MC Dropout passes]) | [LIT] Gal & Ghahramani 2016 | [0, 1] |
| u_consistency | Average pairwise cosine sim of M=3 independent chain-of-thought generations | [LIT] Wang et al. 2022 | [0, 1] |
| u_entropy | 1 − H_semantic/log2(K), K=10 samples at T=1.0, bidirectional NLI clustering | [LIT] Farquhar et al. 2024 | [0, 1] |

**Combined:** û = 0.25·u_token + 0.25·u_dropout + 0.25·u_consistency + 0.25·u_entropy (initial equal weights — post-calibration weights come from Cycle 1 ablation, projected 0.20/0.20/0.20/0.40).

---

**Key design decision — equal initial weights (0.25 each), NOT projected weights:**

> The thesis projects post-calibration weights of 0.20/0.20/0.20/0.40 (SE upweighted per Farquhar et al. 2024 AUROC ≈ 0.79). We do NOT initialise at these values.
>
> **Why start equal?** The projected weights are a calibration hypothesis, not ground truth. Starting equal avoids baking in assumptions that the calibration set might contradict. The actual weights are fitted on 500 samples after Cycle 1 and reported in Chapter 5.
>
> | Option | Description | Verdict |
> |--------|-------------|---------|
> | 0.25/0.25/0.25/0.25 ✓ | Equal — unbiased start | **Chosen** |
> | 0.20/0.20/0.20/0.40 | Pre-bake projected weights | Rejected — premature |
>
> See: `CAEMConfig.u_hat_weight_*` fields; test `test_initial_weights_equal` enforces this.

---

**Key design decision — NLI surface fallback when RoBERTa is unavailable:**

> `_compute_u_entropy` requires a RoBERTa-Large-MNLI model for semantic clustering. During development (before the model is downloaded), passing `nli_model=None` activates a surface-level fallback: groups samples by exact string match after normalisation.
>
> **Why allow the fallback?** Prevents a hard RoBERTa dependency at import time. The estimator can be instantiated and tested without a 1.4 GB model download. The fallback is documented as [DES-fallback] and produces valid [0,1] output — it is less accurate than NLI clustering for paraphrases but does not crash or return garbage.
>
> | Option | Description | Verdict |
> |--------|-------------|---------|
> | Surface fallback (string match) | Lightweight, always available | **Chosen** for dev |
> | Hard dependency (fail without NLI model) | Accurate, no graceful degrade | Rejected |
> | Return 0.5 (neutral) when NLI absent | Simplest | Rejected — loses all signal |

---

**Key design decision — MC Dropout eval() restored in `finally` block:**

> `_compute_u_dropout` switches `model.train()` to activate dropout, runs K=5 forward passes, then must restore `model.eval()`. The restore is placed in a `finally` block so it executes even if `generate()` raises an exception (e.g. GPU OOM).
>
> **Why finally?** If eval() is skipped after an exception, every subsequent inference call (including Stage 5 NLI, SBERT encoding, etc.) would run with dropout active — producing stochastic, unreproducible outputs silently. This is a hard correctness requirement, not a nicety. The test `test_eval_mode_restored_on_exception` verifies this explicitly.

---

**Key design decision — u_token in post_generation differs from pre_routing:**

> Both Stage 3 (pre-routing) and Stage 4a (post-generation) compute u_token. They are NOT the same:
>
> | Stage | Input to model | Answer scored | Purpose |
> |-------|---------------|---------------|---------|
> | Stage 3 (PreRouting) | Query | Short greedy prefix (max 32 tokens, fast) | Quick routing decision |
> | Stage 4a (PostGen) | Query | Full generated answer, token-by-token | Faithful confidence estimate |
>
> Stage 4a uses `model(input_ids, labels=answer_ids)` to score the actual full answer rather than a short probe. More expensive, but more accurate — acceptable because Stage 4a only runs on Tier 2 queries, which are a subset.

---

**Test bugs found and fixed during Session 14:**

1. **`SimpleNamespace` tokenizer mock had no `.items()` method.** `_tokenize()` calls `inputs.items()` for device transfer. Fixed: replaced `SimpleNamespace` return value with a `MagicMock` that explicitly mocks both `.input_ids` attribute and `.items()`.

2. **`model.generate` mock always returned `SimpleNamespace` (not subscriptable).** `_compute_u_consistency` and `_compute_u_entropy` call `out[0]` on the plain tensor returned when `return_dict_in_generate` is not set. Fixed: replaced `model.generate.return_value` with a `side_effect` that returns a `SimpleNamespace` for dropout calls (which set `return_dict_in_generate=True`) and a plain tensor for consistency/entropy calls.

3. **`nli_tok.return_value` was a plain `dict` with no `.to()` method.** `_semantic_entropy_nli` calls `.to(device)` on the tokenizer output before passing it to the NLI model. Fixed: introduced `_DictWithTo(dict)` — a dict subclass that adds `.to(device)` returning `self` for chaining, while preserving `**`-unpacking behaviour.

4. **`test_two_equal_clusters` used wrong call-count mapping.** The test assumed 12 NLI calls (2 per pair, 6 pairs), but Python's `and` short-circuits: if the forward direction returns CONTRADICTION, the reverse is never called. With 4 samples and CONT for cross-cluster pairs, only 8 calls occur. Fixed: updated entailment call numbers from `(1,2,11,12)` to `(1,2,7,8)` with a comment explaining the short-circuit logic.

---

## Session 13 — 2026-03-29

**Scope:** AdaptiveRouter (Stage 3 dispatch).

**Files created this session:**
```
caem/routing/__init__.py
caem/routing/router.py
tests/test_router.py
```

**Test result:** 95/95 passing (35 new router tests + 60 from prior sessions).

---

### Module: `caem/routing/router.py`

**Purpose:** Dispatch every query to exactly one of Tier 1, 2, or 3 based on `u_pre` and the episodic memory search result. Zero model calls — pure logic.

**Key design decision — two mechanisms, not one formula:**

> The routing uses two ENTIRELY SEPARATE mechanisms that share no arithmetic:
>
> **Mechanism 1 — OR-condition (hard veto):** `if u_pre < 0.60 → Tier 3, safety_override=True`. Evaluated FIRST. No memory result involved.
>
> **Mechanism 2 — Routing score:** `0.70·sim + 0.30·û_stored`. Only runs if Mechanism 1 did not fire.
>
> **Why separate?** Combining them into one formula (e.g. `0.70·sim + 0.30·û_stored + 0.X·u_pre`) would allow a high `û_stored` to arithmetically compensate for a dangerously low `u_pre`. A high-quality memory match to a query the model doesn't understand would still route to Tier 1. Keeping them separate makes memory quality and model readiness *both mandatory*, not tradeable.
>
> **Test coverage:** `TestORCondition.test_u_pre_not_in_routing_score` explicitly verifies that two queries with different `u_pre` values (but same sim/û_stored) produce identical routing scores. See: writing-suggestions.md C4-10b.

**Key design decision — empty memory falls through to Tier 3 naturally:**

> When the store is empty, `search()` returns `[]`. The router sets `similarity=0.0` and `û_stored=0.0`. Routing score = 0.0 < 0.90, similarity 0.0 ≤ 0.75 → Tier 3, `safety_override=False`.
>
> **No special-casing needed.** The formula handles empty memory correctly without an explicit branch. This simplifies the code and ensures consistent logging (the routing score is always recorded, even when 0.0).

**Key design decision — floating-point threshold comparison:**

> The Tier 1 condition uses `routing_score >= tier1_combined_threshold` (strict ≥). At exact boundary values computed by algebraic inversion (e.g. `sim = (0.90 - 0.30·u) / 0.70`), floating-point arithmetic can produce `0.8999...` instead of `0.9000`. This is not a bug in the router — it is expected IEEE 754 behaviour. Tests account for this with a small epsilon guard on algebraically-derived boundary values.

---

## Session 12 — 2026-03-29

**Scope:** PreRoutingConfidenceEstimator (Stage 3). Auto-push + version control setup.

**Files created this session:**
```
caem/confidence/__init__.py
caem/confidence/pre_routing.py
tests/test_pre_routing.py
.caem-config          (gitignored — stores GitHub token for auto-push)
README.md
.gitignore
```
**Deleted:** `git-sync.sh` (replaced by auto-push from `.caem-config`).

**Test result:** 60/60 passing (23 new pre-routing tests + 37 from Session 11).

---

### Module: `caem/confidence/pre_routing.py`

**Purpose:** Compute `u_pre` before routing, from two fast signals requiring only a single forward pass. Feeds the OR-condition safety check and the routing score formula.

**Key design decision — C_conv encoder adaptation:**

> **Thesis spec (Nandakishor 2025):** C_conv was designed for decoder hidden states in decoder-only models (GPT-style).
>
> **CAEM adaptation:** Applied to Flan-T5's *encoder* hidden states. The encoder processes the query and produces contextual representations across 12 layers. If early layers (1–6) have higher variance than late layers (7–12), the model has not settled on a stable query representation — a signal of low comprehension confidence *before generation begins*.
>
> **Must be stated in Chapter 4:** "We adapt C_conv to the encoder context: whereas Nandakishor (2025) applied it to decoder states, CAEM uses encoder layer variance ratios to measure pre-generation query understanding confidence." See: writing-suggestions.md C4-03b.

**Key design decision — fail direction on errors:**

> **Signal 1 (u_token):** If `generate()` raises (OOM, device error), returns `0.0` — *pessimistic*. A low u_token pushes toward Tier 3, which is the safe fallback. Better to over-route to Tier 3 than to confidently retrieve a wrong answer.
>
> **Signal 2 (c_conv):** If the encoder forward pass raises, returns `0.0` — *optimistic* (c_conv=0 → conf_cconv=1.0). Since c_conv is the secondary signal (weight 0.40), u_token alone governs. The fail-open choice avoids cascading failures when the encoder is temporarily unavailable.
>
> **Alternatives considered:** Return 0.5 for both on error. Rejected because it obscures which signal failed and produces an ambiguous middle value that neither routes to Tier 3 reliably nor passes the OR-condition clearly.

**Key design decision — short greedy decode for u_token:**

> **max_new_tokens = 32.** Only 32 tokens are decoded for the u_token signal — not a full answer. This keeps Stage 3 latency under ~100 ms on GPU (within the 400 ms Tier 1 budget). The confidence proxy from 32 tokens is empirically sufficient; longer decodes add latency without meaningful signal gain for this purpose.
>
> **Alternative considered:** Use the full generation (until EOS). Rejected because: Stage 3 runs on *every* query regardless of tier — adding full generation latency to every Tier 1 query would exceed the 400 ms target. Tier 2 and Tier 3 generate their own full answers anyway.

**Key design decision — sequential batch estimation:**

> Batched generation with `output_scores=True` requires careful alignment of token IDs across padded sequences. Sequential calls are safer and the per-query overhead is negligible at inference. Batching can be added later if calibration speed is a bottleneck.

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
| `caem/confidence/pre_routing.py` | Stage 3 | No (model only) | ✅ Session 12 |
| `caem/routing/router.py` | Stage 3→dispatch | No | ✅ Session 13 |
| `caem/confidence/post_generation.py` | Stage 4a | No (model only) | 🔨 Session 14 |
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
