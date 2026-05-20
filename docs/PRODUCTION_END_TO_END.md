# CAEM in production — end-to-end design and data flow

> **Purpose.** This document explains how CAEM operates in a real production
> deployment, end-to-end. It is the conceptual companion to
> `PRODUCTION_RUNBOOK.md` (which is command-by-command). Read this first to
> understand WHAT happens and WHY; read the runbook second to see HOW to run
> each step. The implementation gap list at the end identifies the wrapper
> scripts that are designed but not yet coded.
>
> **Status.** Architecture and core mechanisms are implemented and validated
> by Phase 1d/1e research. The production-mode wrapper scripts and serving
> infrastructure are documented designs that an operator would build during
> a real deployment (estimated 1-2 weeks of engineering).

---

## Contents

1. [Big picture](#1-big-picture)
2. [Roles — operator vs end user](#2-roles--operator-vs-end-user)
3. [Two-layer architecture](#3-two-layer-architecture)
4. [Live layer — what happens for every query](#4-live-layer--what-happens-for-every-query)
5. [Cycle-boundary layer — the periodic refresh](#5-cycle-boundary-layer--the-periodic-refresh)
6. [The serving log and the two samples](#6-the-serving-log-and-the-two-samples)
7. [Cal data vs SIL pool — the key distinction](#7-cal-data-vs-sil-pool--the-key-distinction)
8. [Tagging strategies](#8-tagging-strategies)
9. [Labeling pipeline options](#9-labeling-pipeline-options)
10. [Per-domain dispatch at inference](#10-per-domain-dispatch-at-inference)
11. [Cycle-refresh recipe](#11-cycle-refresh-recipe)
12. [Failure modes and recovery](#12-failure-modes-and-recovery)
13. [Drift detection and monitoring](#13-drift-detection-and-monitoring)
14. [Production cycle trigger policies](#14-production-cycle-trigger-policies)
15. [What's implemented vs what's not](#15-whats-implemented-vs-whats-not)
16. [Implementation roadmap](#16-implementation-roadmap)
17. [Frequently confused points](#17-frequently-confused-points)

---

## 1. Big picture

CAEM is **the same architecture in research and production**. The only difference is the source of data:

- **Research**: pre-defined benchmarks (FEVER, TriviaQA, etc.) supply queries + gold labels. Cycles are triggered by question budget.
- **Production**: live user traffic supplies queries. Labels come from the operator's labeling pipeline at cycle boundaries. Cycles are triggered by accumulation or calendar.

Everything else — three-tier router, verifier composite, storage gate, memory store, SIL fine-tune, retroactive re-verification, retention guard, RUC, hot-reload — is identical.

Production deployment is **two layers running on different time scales**:

| Layer | Time scale | Triggered by | Visible to user |
|---|---|---|---|
| Live serving | sub-second | every user query | yes — the answer |
| Cycle refresh | ~monthly | accumulation or calendar | no — maintenance window |

The live layer hits a *frozen snapshot* of CAEM's knobs (composite, thresholds, model weights). The cycle layer periodically updates that snapshot using labeled data sampled from recent live traffic, then atomically swaps the new snapshot in.

---

## 2. Roles — operator vs end user

This distinction is critical and easy to lose.

### End user

The person who sends a question and expects an answer.

- **Provides**: the query text.
- **Does NOT provide**: domain tags, benchmark identity, correctness labels, calibration data.
- **Sees**: only the answer (sub-second).
- **Does not see**: cycle boundaries, labeling, retraining, calibration refits.

End users in a hospital deployment are doctors and nurses. In a legal-QA product they are lawyers. In a public chatbot they are anonymous web visitors. None of them tag queries or label answers.

### Operator

The team running CAEM. In a research context this is "us" (the developers + researchers). In a real deployment this is the SRE/MLops team at the company that licenses or builds on CAEM.

- **Provides**:
  - Infrastructure (serving server, storage, GPU for cycle boundary).
  - Domain tagging strategy at intake (route-based, classifier-based, or none — see §8).
  - Labeling pipeline at cycle boundaries (LLM oracle, validated feedback proxy, or human review — see §9).
  - Cycle trigger policy (accumulation count or calendar period).
  - Monitoring dashboards and drift alerts.
- **Does NOT do per-query work**. The operator's effort is concentrated at cycle boundaries (monthly) and during incident response.
- **Sees**: serving logs, audit JSONs, drift dashboards, cycle outcomes.

The "operator must tag every query" rule in `PRODUCTION_RUNBOOK.md` §11.2 means **the operator's automated intake system tags the query** before it reaches CAEM. End users are unaware of the tagging.

---

## 3. Two-layer architecture

```
═══════════════════════════════════════════════════════════════════
LIVE LAYER  — sub-second, runs millions of times per day
═══════════════════════════════════════════════════════════════════

  User query  ──►  Tag attached  ──►  CAEM.answer()  ──►  Reply
                       │                    │
                       │                    ├──►  serving_log
                       │                    │     (append one line)
                       │                    │
                       │                    └──►  memory_store
                       │                          (append one entry
                       │                           if STORE decision)
                       │
                       (Tag chosen by operator's intake system)

  (CAEM uses the CURRENT calibration JSONs + CURRENT LoRA weights;
   does not pause for cycle refresh; does not block user request.)


═══════════════════════════════════════════════════════════════════
CYCLE-BOUNDARY LAYER — runs once a month, behind the scenes
═══════════════════════════════════════════════════════════════════

  Cycle trigger fires (accumulation count or calendar)
              │
              ├──►  SAMPLE A  ──►  Operator labels  ──►  refit knobs
              │     (3000 strat-                          (T, composite,
              │      ified queries)                        τ_store,
              │                                            τ_defer)
              │
              ├──►  SAMPLE B  ──►  SIL fine-tune  ──►  new LoRA
              │     (STORE-class                         weights
              │      entries from
              │      memory_store)
              │
              └──►  memory_store  ──►  retroverify  ──►  pruned memory
                    (existing)         (re-score under
                                        new composite)

  Atomic swap:  new composite + new LoRA + pruned memory go live
                on the very next user query. No restart needed.
```

The live layer reads from `outputs/production/{composite_calibration.json, conformal_gate.json, calibrated_config.json, lora_adapter/}` at inference time. The cycle layer writes the new versions, then promotes them atomically.

---

## 4. Live layer — what happens for every query

This is the per-query inference path. It must be fast (sub-second) and must NOT pause for cycle work.

### 4.1 Step-by-step

```
1. User sends query through operator's frontend
   (web form, API call, mobile app, etc.)

2. Operator's intake system attaches a tag:
   - Route-based: /medical/ask → tag = medical_qa
   - Classifier-based: domain classifier outputs tag
   - Untagged: tag = None (acceptable fallback)

3. Backend invokes:
     pipeline.answer(query, source_benchmark=tag)

4. Inside CAEMPipeline.answer():
   a. Stage 1: encode query with SBERT
   b. Stage 2: search memory_store for k=1 nearest neighbor
   c. Stage 3a: pre-routing confidence estimator (uses T_b per tag)
   d. Stage 3b: router decides T1 / T2 / T3
      - If RUC wired: consults RUC for T2 vs T3 decision
      - If untagged: uses pooled safety_u_pre_min fallback
   e. Stage 4: tier dispatch
      - T1: serve cached answer from memory
      - T2: parametric generation (Qwen + LoRA, no retrieval)
      - T3: RAG generation (retrieve + cross-encoder rerank + generate)
   f. Stage 5: verifier scores 10 signals (u_token, u_dropout, ...)
   g. Stage 5b: composite combines signals via per-tag head
      (or pooled fallback if untagged)
   h. Stage 6: storage decision
      STORE / DEFERRED / ABSTAIN / DISCARD
      (uses τ_store=0.60 and τ_defer=0.45, both global)

5. Three things happen in parallel:
   - Answer text returned to user
   - One line appended to serving_log:
       {timestamp, query_id, query, tag, signals, decision, ...}
   - If decision = STORE: entry added to memory_store immediately
     (memory grows continuously, not just at cycle boundary)

6. End user sees the answer. Total latency: ~500ms - 2s.
```

### 4.2 What the live layer reads

```
outputs/production/
├── composite_calibration.json    ← reloaded into verifier at startup
├── conformal_gate.json           ← reloaded into verifier at startup
├── calibrated_config.json        ← T_b, safety_u_pre_min_b per tag
├── tau_ruc.json                  ← RUC threshold (global or per-tag)
├── lora_adapter/                 ← current LoRA weights
└── memory_store/
    ├── memory_store.faiss
    └── memory_store.meta
```

These files are read at server startup. Hot-reload mechanism updates them in-process when a cycle boundary completes.

### 4.3 What the live layer writes

```
outputs/production/
├── serving_log/
│   └── YYYY-MM-DD.jsonl          ← one line per query, all signals
└── memory_store/
    ├── memory_store.faiss        ← grows with each STORE decision
    └── memory_store.meta
```

The serving log is the data source for the cycle boundary. Without it, there is no cal data and no SIL pool. The log discipline is therefore essential.

### 4.4 Serving log schema

Each line in `serving_log/*.jsonl` should be one JSON object with at minimum:

```jsonc
{
  "ts": "2026-05-19T14:32:11Z",
  "query_id": "uuid-xxx",
  "query": "What is the typical dose of amoxicillin?",
  "source_benchmark": "medical_qa",   // tag attached at intake; may be null
  "tier": 3,                          // routing decision
  "decision": "STORE",                // STORE/DEFERRED/ABSTAIN/DISCARD
  "u_stored": 0.78,
  "u_pre": 0.65,
  "signals": {
    "u_token": 0.12, "u_dropout": 0.08, "u_internal": 0.15,
    "s_avg": 0.84, "h_norm": 0.32, "p_entail": 0.91,
    "p_ground_max": 0.88, "p_ground_mean": 0.74, "p_ground_atomic": 0.82,
    "q_a_relevance": 0.79
  },
  "answer": "500mg three times daily for 7-10 days",
  "latency_ms": 1230,
  "model_version": "lora_cycle_7",
  "calibration_version": "composite_cycle_7"
}
```

The `signals` block is essential — cycle-boundary refit needs the raw signal vectors. The `decision` is essential — cycle-boundary sampling stratifies by it. The `model_version` and `calibration_version` are essential for audit (so you know which cycle's snapshot served this query).

---

## 5. Cycle-boundary layer — the periodic refresh

This is what runs once a month (or per the operator's cycle policy). It happens in a maintenance window and does not affect live serving — the cycle work runs on separate compute, and only the atomic-swap at the end touches the live serving paths.

### 5.1 The seven steps in plain terms

```
1. SAMPLE CAL DATA
   Take a stratified sample of ~3000 recent queries from the
   serving log (~750 per decision class).

2. LABEL THE SAMPLE
   Operator's labeling pipeline assigns EM = 0 or 1 to each
   sample's served answer. Three options for the pipeline
   (see §9): LLM oracle, validated feedback proxy, or expert
   review. End users do not participate here.

3. RE-RUN SIL FINE-TUNE
   Train a new LoRA adapter on the SIL pool (= all STORE-class
   entries accumulated in memory_store since the last cycle).
   Anchor to previous weights via L2. Probe retention. Commit
   or roll back.

4. RESCORE THE CAL FOLD
   Re-score the ~3000 labeled samples under the new LoRA so
   the calibration fit operates on post-SIL signals (the
   serving-log signals were under the previous cycle's model
   and are stale).

5. REFIT THE KNOBS
   On the rescored + labeled cal fold:
     - Refit T scalar (and T_b per tag).
     - Refit per-tag isotonic curves.
     - Refit per-tag composite LR (with shrinkage to pooled).
     - Refit τ_store, τ_defer (EMA against previous τ).
     - Optionally refit τ_RUC global or per-tag.

6. RETROVERIFY MEMORY
   Re-score every memory_store entry under the new composite +
   new model. Prune entries below τ_retro = 0.50. Back up the
   pre-prune memory snapshot first.

7. PROMOTE AND AUDIT
   Atomic-swap new composite/gate/T JSONs to canonical paths.
   Hot-reload the live verifier (millisecond operation).
   Write the cycle's audit JSON (rollback anchor).
```

### 5.2 Why this exact order

The order matters and was learned from research-mode debugging:

```
SIL fine-tune  →  hot-swap LoRA  →  rescore cal fold under new model
                                              ↓
                                       refit knobs on POST-SIL signals
                                              ↓
                                  hot-reload verifier with new knobs
                                              ↓
                                   retroverify memory under new
                                   model + new knobs
                                              ↓
                                       promote to canonical
```

Common wrong order to avoid:
- **Refit knobs BEFORE SIL** → knobs fit to old signal distribution, immediately stale after fine-tune.
- **Retroverify BEFORE refit** → memory pruned by old knobs, then memory shrinks under new knobs again. Two prune passes.
- **Promote BEFORE retroverify** → live verifier sees new knobs but stale memory, briefly produces inconsistent decisions.

The order above is the only correct one.

### 5.3 Wall-time budget

Approximate per-cycle wall-time on a 5090 GPU:

| Step | Wall-time | Compute |
|---|---|---|
| 1. Sample cal data | 1 min | CPU, single Python script |
| 2. Label sample (oracle) | 30 - 60 min | API calls to GPT-class |
| 3. SIL fine-tune | 4 - 6 h | GPU, LoRA training |
| 4. Rescore cal fold | 30 min | GPU, inference |
| 5. Refit knobs | 5 min | CPU |
| 6. Retroverify memory | 1 - 4 h | GPU, depends on memory size |
| 7. Promote + audit | <1 min | atomic file ops |
| **Total** | **~6 - 10 h** | maintenance window |

For a hospital running quarterly, a weekend window covers it.

---

## 6. The serving log and the two samples

The serving log accumulates millions of entries between cycles. At the cycle boundary, **two disjoint samples** are drawn from it for different purposes.

```
serving_log/  (≈ 5 000 000 lines accumulated this month)
       │
       ├──── SAMPLE A: ~3000 stratified by decision class
       │     ┌──────────────────────────────────────────┐
       │     │ ~750 STORE, ~750 DEFERRED,               │
       │     │ ~750 ABSTAIN, ~750 DISCARD               │
       │     │ (stratified to ensure enough mass per    │
       │     │  class for stable isotonic fit)          │
       │     └──────────────────┬───────────────────────┘
       │                        ▼
       │                  CAL DATA (labeled by operator)
       │                  used to refit composite + T + τ
       │
       └──── SAMPLE B: all STORE-class entries
             (already accumulated in memory_store
              via live STORE decisions during the month)
             ┌──────────────────────────────────────────┐
             │ ≈ 50 000 entries (depends on month's     │
             │   traffic + storage rate)                │
             │ each entry is (query, answer, signals,   │
             │   tag, u_stored)                         │
             └──────────────────┬───────────────────────┘
                                ▼
                          SIL TRAINING POOL
                          used to fine-tune LoRA
```

### Both samples come from the same source (the live stream) but are used at different points in the cycle boundary for different mechanisms.

Distinguishing them is crucial:

| | Cal data | SIL pool |
|---|---|---|
| Size | ~3000 / cycle | ~50 000 / cycle |
| Sample method | Stratified random | Continuous accumulation via STORE |
| Labels | Operator-supplied EM | Verifier-supplied STORE decision (no human EM) |
| Used for | Refit calibration knobs | Fine-tune LoRA |
| Costs | Labeling cost (~$0.001-$1/sample depending on pipeline) | GPU training cost only |
| Frequency | Once per cycle | Continuously growing during the cycle |

---

## 7. Cal data vs SIL pool — the key distinction

This is the question that confused us in v1 design — separating these two cleanly is the unlock.

### Cal data is small, expensive, precise.

- Small: ~3000 samples.
- Expensive: each sample needs an EM label, which costs labeling pipeline time/money.
- Precise: labels are direct (correct or incorrect against gold), so they drive the calibration fit accurately.

We can't make this bigger — labeling 50 000 samples per cycle is impractical for most operators.

### SIL pool is big, cheap, noisy.

- Big: ~50 000 samples per cycle.
- Cheap: no human labeling. The verifier's STORE decision (u_stored ≥ τ_store) acts as the proxy label.
- Noisy: ~15-25% of STORE entries are wrong (the verifier's calibration gap), which adds label noise to the SIL training.

We can't make this much smaller — LoRA fine-tuning benefits from more samples.

### They serve different mechanisms

| Mechanism | Why it needs cal data not SIL pool |
|---|---|
| Refit T scalar | Needs ground-truth EM to compute ECE before/after |
| Refit isotonic curves | Needs ground-truth EM to fit P(EM=1 \| signal) curves |
| Refit τ_store | Needs ground-truth EM to set the precision target at α |
| Refit per-tag composite LR | Needs ground-truth EM to fit the logistic head |

| Mechanism | Why it needs SIL pool not cal data |
|---|---|
| Fine-tune LoRA | Needs ~50k samples for stable gradient signal |
| The model has already self-verified these samples | STORE decision is the self-distillation signal |
| Labeling 50k would defeat the SIL idea (which is the model teaching itself) |

So cal data answers "how good is the verifier?" while SIL pool answers "how can the model improve at the kinds of things the verifier already accepts?"

---

## 8. Tagging strategies

The operator must decide HOW to tag queries at intake. Three valid strategies:

### Strategy A — Route-based tagging

Operator runs domain-segmented endpoints. The endpoint itself determines the tag.

```
POST /medical/ask     → tag = "medical_qa"
POST /legal/ask       → tag = "legal_qa"
POST /scheduling/ask  → tag = "admin_qa"
```

The hospital app shows separate buttons for "ask about a patient" vs "ask about scheduling," and each button hits a different endpoint. The backend hard-codes the tag based on the endpoint.

**Pros:** Zero ML, perfect tag quality within scope, easy to maintain.
**Cons:** Requires the product to be naturally segmented. Doesn't work for a general-purpose chatbot.
**When to use:** Specialized deployments (medical, legal, customer support per department).

### Strategy B — Auxiliary classifier

Operator runs a small zero-shot domain classifier at intake. The classifier reads the raw query and outputs a tag.

```python
def tag_at_intake(query: str) -> Optional[str]:
    # A lightweight LLM or trained classifier
    tag = domain_classifier.predict(query)
    if tag.confidence < 0.7:
        return None  # fall back to pooled
    return tag.label
```

**Pros:** Works for general-purpose deployments. Tags emerge dynamically from query content.
**Cons:** Maintaining a domain classifier is a separate ML effort. Classifier errors propagate to per-bench dispatch errors.
**When to use:** When the product is general-purpose but the operator wants per-domain calibration benefits.

### Strategy C — No tagging (pooled fallback)

Operator passes `source_benchmark=None` for every query. CAEM uses the pooled composite + global T + global safety thresholds.

**Pros:** Zero engineering. CAEM still works.
**Cons:** Sacrifices per-bench specialization. Phase 1d / 1e research-mode advantages don't fully transfer.
**When to use:** Initial deployments where the operator wants to validate CAEM end-to-end before investing in tagging infrastructure. Or general-purpose chatbots with no natural domain boundaries.

### Decision tree

```
Does the product have natural domain segmentation
(e.g., separate UI sections for different topics)?
├── YES → Strategy A (route-based, cheapest, best)
└── NO →
    Is per-domain calibration worth a classifier maintenance burden?
    ├── YES → Strategy B (auxiliary classifier)
    └── NO  → Strategy C (untagged, pooled fallback)
```

---

## 9. Labeling pipeline options

The operator must decide HOW to label the cal-fold at each cycle boundary. Three valid options, in preference order:

### Option 1 — Automated LLM oracle

A stronger LLM (e.g., GPT-class or Claude Opus class) scores each sample's served answer against a fresh retrieval or against an external knowledge base.

```python
def label_sample(query: str, served_answer: str) -> int:
    # Call a stronger LLM with a prompt that asks
    # "Is `served_answer` a correct answer to `query`?"
    response = oracle_llm.judge(query, served_answer)
    return 1 if response.is_correct else 0
```

**Cost:** ~$0.001 - $0.01 per sample on current API pricing. ~$3 - $30 per cycle of 3000 samples.
**Pros:** Cheapest. Fastest. Can be fully automated.
**Cons:** Inherits the oracle LLM's biases. Best for factual QA, less reliable for nuanced subjective tasks.
**When to use:** Most general-purpose deployments. Default choice for first deployments.

### Option 2 — User-feedback proxy

Operator collects implicit or explicit feedback signals from end users (thumbs up/down, dwell time, click-through, immediate re-asking). Validates that the signal correlates with actual EM, then uses it as a proxy.

```python
def label_sample(query: str, served_answer: str, feedback_signal: float) -> int:
    # feedback_signal is the validated proxy (e.g., thumbs-up rate
    # over a 24h window from that user, or aggregate engagement)
    return 1 if feedback_signal > threshold else 0
```

**Cost:** Free at labeling time, but requires upfront validation work (~$10k - $50k for the validation study) and per-query feedback infrastructure.
**Pros:** Scales with traffic. Reflects real-world utility, not just factual correctness.
**Cons:** Indirect — feedback is noisy and culturally biased. Validation must be redone if feedback channels change.
**When to use:** High-volume consumer products with feedback UI already in place.

### Option 3 — Human expert review

Operator hires domain experts (e.g., nurses, lawyers, programmers) to manually label samples.

**Cost:** $10 - $100 per sample depending on expertise. $30k - $300k per cycle of 3000 samples.
**Pros:** Highest label quality. Defensible for high-stakes domains.
**Cons:** Most expensive. Slowest. Doesn't scale to weekly cycles.
**When to use:** Medical / legal / safety-critical deployments where the labeling-cost is justified by the deployment risk.

### Hybrid: tiered labeling

Most production deployments will combine these:

```
Sample 3000 queries at cycle boundary
   ├── 2700 → LLM oracle (cheap, fast)
   ├── 200 → User-feedback proxy (validate the oracle)
   └── 100 → Human expert review (audit + spot-check)
```

This gives statistical coverage on the oracle's quality while keeping costs bounded.

---

## 10. Per-domain dispatch at inference

When CAEM serves a query, multiple knobs dispatch by the tag. Here is what each does and how it falls back.

### 10.1 The dispatch path

```python
def serve(query: str, tag: Optional[str]):
    # Stage 3a: pre-routing confidence
    T_b = config.temperature_scalar_per_benchmark.get(tag, config.temperature_scalar)
    u_pre = pre_estimator(query, T=T_b)

    # Stage 3b: router
    safety_min = config.safety_u_pre_min_per_benchmark.get(tag, config.safety_u_pre_min)
    if u_pre < safety_min:
        # ... safety fall-through path
    if RUC is wired:
        tau_RUC = tau_ruc_per_bench.get(tag, tau_ruc_global)
        # ... RUC consultation with tau_RUC

    # Stage 5b: verifier composite
    composite_head = composite_calibration["per_benchmark"].get(tag,
                                composite_calibration["global"])
    p_calibrated = composite_head.transform(signals)

    # Storage decision (uses GLOBAL τ_store, τ_defer)
    if p_calibrated >= τ_store: STORE
    elif p_calibrated >= τ_defer: DEFERRED
    else: ABSTAIN or DISCARD
```

### 10.2 What is per-tag vs global

| Knob | Per-tag? | Fallback when tag is unknown |
|---|---|---|
| T scalar (T_b) | Yes | Pooled T |
| safety_u_pre_min | Yes | Global safety_u_pre_min |
| RUC threshold (τ_RUC) | Optional (Phase 1f) | Global τ_RUC = 0.375 |
| Composite head | Yes (per_benchmark block) | global block |
| τ_store, τ_defer | No — always global | n/a |
| τ_retro | No — always global | n/a |

The store / defer / retro thresholds are deliberately global. They operate on the calibrated probability `p_calibrated`, which the per-tag composite has already normalized. Making them per-tag would introduce double per-tag dependency and over-fit.

### 10.3 Graceful degradation

If the operator sends `tag=None`:

- T scalar falls back to pooled T (typically ~1.0).
- safety_u_pre_min falls back to global default (typically ~0.30).
- Composite falls back to global head.
- τ_RUC falls back to global 0.375.

CAEM continues to work. Performance is similar to vanilla CAEM (pre-RUC, pre-per-bench-fix) — i.e., research-mode pooled. The per-domain advantages are lost but the architecture is intact.

---

## 11. Cycle-refresh recipe

This is the full sequence the operator runs at each cycle boundary. Each step references the script the operator would invoke. Wrapper scripts marked **(MISSING)** are designed but not yet implemented; see §15.

### Step 1 — Sample cal data

```bash
CYCLE=N
mkdir -p outputs/production/cycle_${CYCLE}/calibration

python scripts/build_cycle_calibration_batch.py \    # MISSING
    --serving_log_dir outputs/production/serving_log/ \
    --since_last_cycle outputs/production/audit/refresh_$((CYCLE-1)).json \
    --n_per_class 750 \
    --output outputs/production/cycle_${CYCLE}/calibration/sampled_queries.jsonl
```

The script:
1. Reads serving_log entries since the last cycle.
2. Stratifies by decision class (STORE, DEFERRED, ABSTAIN, DISCARD).
3. Draws ~750 per class with replacement-free sampling.
4. Writes the sample with query, tag, signals, decision.

### Step 2 — Operator labels the batch

The labeling pipeline produces:

```jsonl
{"query_id": "uuid-abc", "served_answer": "...", "em": 1, "labeler": "claude_opus"}
{"query_id": "uuid-def", "served_answer": "...", "em": 0, "labeler": "claude_opus"}
```

This is glued onto the sampled queries:

```bash
python scripts/join_labels_to_batch.py \              # MISSING
    --queries outputs/production/cycle_${CYCLE}/calibration/sampled_queries.jsonl \
    --labels  outputs/production/cycle_${CYCLE}/calibration/labels.jsonl \
    --output  outputs/production/cycle_${CYCLE}/calibration/labeled.jsonl
```

### Step 3 — SIL fine-tune

```bash
python scripts/production_sil_finetune.py \           # MISSING (wrapper)
    --cycle ${CYCLE} \
    --memory_store outputs/production/memory_store/ \
    --general_data outputs/production/general_data/ \
    --output_checkpoint outputs/production/cycle_${CYCLE}/model.pt
```

Internally this calls `SelfImprovementLoop.run_cycle` (implemented) with the production memory store as the SIL pool source. Same retention probe, same L2 anchor, same rollback semantics as research.

If the retention probe fails (ratio < 0.93), the wrapper logs an abort and skips steps 4-9. The cycle is recorded as aborted; the next cycle retries.

### Step 4 — Hot-swap new model

```bash
python scripts/swap_production_model.py \             # MISSING (wrapper)
    --new_checkpoint outputs/production/cycle_${CYCLE}/model.pt
```

Internally, makes an HTTP call to the live server's admin endpoint that does `pipeline.swap_lora(new_path)` — replaces the LoRA adapter in-process without server restart. ~milliseconds.

### Step 5 — Rescore cal fold under new model

```bash
python scripts/score_calibration_batch.py \           # MISSING (wrapper)
    --queries outputs/production/cycle_${CYCLE}/calibration/labeled.jsonl \
    --output_dir outputs/production/cycle_${CYCLE}/calibration/scored/
```

Re-runs each labeled sample through the updated CAEM pipeline, capturing fresh signal vectors under the new LoRA. The signals from the serving log are now stale (they were under the previous cycle's model).

### Step 6 — Refit T scalar

```bash
python scripts/run_calibration.py \                   # IMPLEMENTED
    --calib_dir outputs/production/cycle_${CYCLE}/calibration/scored/ \
    --temperature_only \
    --previous_T_path outputs/production/calibrated_config.json \
    --output outputs/production/cycle_${CYCLE}/calibrated_config.json
```

Outputs T_b per tag (and pooled T). Reports `temperature_before`, `temperature_after`, `ece_before`, `ece_after`.

### Step 7 — Refit isotonic + composite + τ

```bash
python scripts/fit_composite_calibration.py \         # IMPLEMENTED
    --calib_jsons outputs/production/cycle_${CYCLE}/calibration/scored/*.json \
    --output_json outputs/production/cycle_${CYCLE}/composite_calibration.json \
    --shrinkage_alpha 0.6 \
    --cherian_boost \
    --boost_C 0.01 \
    --previous_composite outputs/production/composite_calibration.json \
    --ema_alpha 0.7
```

Fits per-tag isotonic + LR + shrinkage. EMA against previous composite for stability (research mode does not EMA; production should).

If the runbook needs explicit conformal τ refit, an additional step calls `recalibrate_conformal_at_cycle.py` (MISSING, but functionality is split across existing scripts plus an EMA helper).

### Step 8 — Hot-reload verifier

```bash
python scripts/swap_production_calibration.py \       # MISSING (wrapper)
    --composite outputs/production/cycle_${CYCLE}/composite_calibration.json \
    --gate      outputs/production/cycle_${CYCLE}/conformal_gate.json \
    --config    outputs/production/cycle_${CYCLE}/calibrated_config.json
```

Makes admin-endpoint calls that do `pipeline.verifier.reload_calibration(...)` and updates the in-process T. Live verifier picks up new calibration on the next query.

### Step 9 — Retroverify memory

```bash
python scripts/production_retroverify.py \            # MISSING (wrapper)
    --memory_store outputs/production/memory_store/ \
    --threshold 0.50 \
    --backup_first
```

Internally calls `retroactive_reverification` (implemented). Re-scores every memory entry; prunes below 0.50. Backup goes to `outputs/production/cycle_${CYCLE-1}/memory_store/`.

### Step 10 — Promote to canonical paths

```bash
cp outputs/production/cycle_${CYCLE}/composite_calibration.json \
   outputs/production/composite_calibration.json
cp outputs/production/cycle_${CYCLE}/conformal_gate.json \
   outputs/production/conformal_gate.json
cp outputs/production/cycle_${CYCLE}/calibrated_config.json \
   outputs/production/calibrated_config.json
```

Ensures a server restart would read the new calibration at boot. Without this step, an in-process hot-reload succeeds but a restart silently reverts.

### Step 11 — Write audit JSON

```bash
python scripts/write_cycle_audit.py \                 # MISSING (wrapper)
    --cycle ${CYCLE} \
    --output outputs/production/audit/refresh_${CYCLE}.json
```

Captures every "before vs after" metric: T_before/after, ECE_before/after, τ_store before/after, n_pruned by retroverify, retention probe ratio, etc. This is the rollback anchor for cycle ${CYCLE+1}.

---

## 12. Failure modes and recovery

### 12.1 SIL retention guard fires

**Symptom:** step 3 reports `aborted=True` because retention probe ratio < 0.93. The new LoRA was rejected.

**Recovery:**
- Skip steps 4-10 (calibration refit is meaningless without new model).
- Record the abort in audit JSON.
- The next cycle retries with whatever new memory has accumulated.
- Live serving continues with the PREVIOUS cycle's LoRA and calibration.

**Why this works:** memory accumulates strictly even when weights do not. The asymmetric rollback property is built into the architecture.

### 12.2 Memory store corruption

**Symptom:** FAISS index reports inconsistent metadata, or search returns garbage.

**Recovery:**
```bash
PREV=$((CYCLE - 1))
cp outputs/production/cycle_${PREV}/memory_store/memory_store.faiss \
   outputs/production/memory_store/memory_store.faiss
cp outputs/production/cycle_${PREV}/memory_store/memory_store.meta  \
   outputs/production/memory_store/memory_store.meta
# Restart the production server.
```

Lose: all STORE decisions made since the previous cycle's retroverify checkpoint. Keep: everything else.

### 12.3 Cal fold too small for stable fit

**Symptom:** step 7 reports `n_store < 20` on the labeled batch. Conformal τ fit is unstable.

**Recovery:** re-sample with higher STORE-class stratification weight, or extend the sampling window to include more days of serving log.

### 12.4 Composite drifts catastrophically

**Symptom:** new τ_store > 0.10 away from previous τ_store. Suggests distribution shift.

**Recovery:**
- Don't auto-promote. Manually review the new composite.
- Check whether a product launch, an upstream model change, or a serving infrastructure change caused the shift.
- Either accept the shift (if explainable) or roll back to the previous composite and investigate.

### 12.5 Retention abort and the asymmetric rollback property

This is the most-important durability property in the architecture.

**When the retention guard fires** (any probe drops below 0.93 of the cycle-0 baseline), the SIL fine-tune for that cycle is rejected. The LoRA reverts to the previous cycle's weights. But **only the LoRA-weight channel pauses**. Every other learning channel keeps running.

| Channel | Status on retention abort |
|---|---|
| LoRA weight update | **STOPS** (rolled back to θ_prev) |
| Memory store growth | Continues (live STORE decisions still add entries) |
| Composite calibration refit | Continues (step 7 of cycle refresh) |
| T scalar refit | Continues |
| τ_store, τ_defer EMA update | Continues |
| Retroactive re-verification | Continues (prunes stale memory under the new composite even though LoRA reverted) |
| Deferred-buffer TTL promotion | Continues |
| Tier 1 hit rate | Keeps growing (memory growing → more cache hits) |
| RUC routing | Unchanged (RUC is frozen across cycles) |

The asymmetric rollback property: **memory accumulates strictly even when weights do not**. One cycle's abort costs that cycle's fine-tune compute and nothing else.

#### What happens if retention chronically aborts

Single-cycle aborts are routine and self-healing. The concern is chronic aborts:

| Consecutive aborts | What CAEM looks like |
|---|---|
| 1-2 | Normal — LoRA paused while memory + calibration catch up |
| 3-5 | Concerning — investigate via `coverage_diagnostic.json` |
| 6+ | Degenerate — CAEM has become "frozen model + growing memory cache." Operator must intervene. |

In the degenerate case, CAEM still answers queries — Tier 1 retrieval works, Tier 3 RAG works — but the parametric T2/T3 generation has stopped improving.

#### Diagnostic triggers (already implemented)

The `coverage_diagnostic.json` file written at each cycle close drives `caem.diagnostic.coverage.evaluate_halt_triggers`:

| Trigger | What it indicates | Operator action |
|---|---|---|
| `relax_alpha`: a bench has zero admissions for 2 cycles | Storage gate too strict for that bench | Lower `alpha_store` per-bench; resume |
| `halt`: adapter SV-collapse score > 0.95 | LoRA has rank-collapsed; gradient signal exhausted | Stop the trajectory; reset LoRA + restart from research-final |

#### Phase 1d Cycle 4 — the empirical precedent

Phase 1d's Cycle 4 retention guard fired (TriviaQA-test 0.395 → 0.345, ratio 0.8734 < 0.93). What happened:

- LoRA reverted to C3 weights.
- Memory store at C4 close was preserved (deferred-buffer promotions + retroverify pruning still ran).
- Composite refit, T refit, τ EMA all completed.
- Cycle 5 attempted SIL again on the post-C4 memory; it ran but trajectory was stop-locked at C5 per the runbook directive.

The architecture handled the abort gracefully. The thesis numbers (Ch5 main results) include the C4 abort as part of the trajectory — the abort is data, not a failure.

### 12.6 Calibration knob roll-back

If a cycle promotes a bad calibration and downstream metrics suffer:

```bash
PREV=$((CURRENT - 1))
cp outputs/production/cycle_${PREV}/composite_calibration.json \
   outputs/production/composite_calibration.json
cp outputs/production/cycle_${PREV}/conformal_gate.json \
   outputs/production/conformal_gate.json
cp outputs/production/cycle_${PREV}/calibrated_config.json \
   outputs/production/calibrated_config.json
# Hot-reload via admin endpoint.
```

The audit JSON from cycle ${PREV} is the canonical "last known good" reference.

---

## 13. Drift detection and monitoring

The operator should compute these metrics daily from the serving log:

| Metric | Healthy range | Alert threshold |
|---|---|---|
| Storage rate (STORE / total) | ±20% drift from last cycle | ±30% over 7 days |
| Defer rate | ±20% drift | sudden 2× change overnight |
| Abstain rate | ±20% drift | ±50% over 7 days |
| Mean u_stored on STORE | ±0.05 drift | ±0.10 |
| Tier 1 hit rate | grows slowly over time | sudden drop |
| Tier 3 fallback rate | shrinks as memory grows | sudden growth |
| p95 latency | ±20ms drift from baseline | >100ms increase |

If any alert fires, run §11 immediately (drift-triggered cycle boundary) regardless of the calendar/accumulation trigger.

The dashboard should also plot:
- Calibration drift (τ_store before/after each cycle).
- Memory size trajectory.
- Per-tag storage rate over time.
- LoRA SVD norms (rank-collapse early warning).

---

## 14. Production cycle trigger policies

The operator picks one of two strategies for when cycle boundaries fire:

### Strategy 1 — Accumulation-based

```jsonc
{
  "trigger": "accumulation",
  "n_store_target": 10000,    // fire when 10k STORE decisions accumulate
  "min_days": 7,              // never sooner than weekly
  "max_days": 90              // never later than quarterly
}
```

Cycle fires when STORE accumulation hits target, bounded by min/max days. Better for variable-traffic deployments (low-traffic weeks don't waste cycle compute on small batches; high-traffic weeks don't let the calibration drift).

### Strategy 2 — Calendar-based

```jsonc
{
  "trigger": "calendar",
  "period": "weekly",         // or "monthly", "quarterly"
  "day_of_week": "sunday",    // run during low-traffic window
  "hour_utc": 2               // 2 AM UTC
}
```

Cycle fires on a fixed schedule. Simpler to reason about. Standard for hospitals and regulated industries where audit cycles are calendar-driven.

### Hybrid

Most deployments use both: calendar-default with accumulation-override.

```jsonc
{
  "trigger": "hybrid",
  "calendar_period": "monthly",
  "accumulation_override": {
    "n_store_target": 20000,
    "min_days_since_last": 14
  }
}
```

Monthly cadence by default, but if STORE accumulation hits 20k more than 14 days early, fire then instead.

---

## 15. What's implemented vs what's not

### Implemented (research-grade, reusable in production)

| Component | Path | Notes |
|---|---|---|
| `CAEMPipeline.answer` | `caem/pipeline.py` | Full per-query inference path |
| `SelfImprovementLoop.run_cycle` | `caem/sil.py` | LoRA fine-tune with L2 anchor + retention probe |
| `retroactive_reverification` | `scripts/run_experiment.py` | Re-score memory under new composite |
| `fit_composite_calibration.py` | `scripts/` | Per-tag composite + shrinkage + Cherian boost |
| `run_calibration.py` | `scripts/` | T scalar refit (per-tag T_b) |
| Verifier `reload_calibration` | `caem/verification/verifier.py:622` | Hot-swap composite + gate JSONs |
| Per-tag composite dispatch | `verifier._composite` | Falls back to pooled if tag missing |
| Per-tag T_b dispatch | `pre_estimator.estimate` | Falls back to pooled if tag missing |
| Per-tag safety_u_pre_min | `router.route` | Falls back to global if tag missing |
| RUC v2 (LR + 36 features) | `caem/routing/ruc.py` | Global τ; per-tag is Phase 1f |
| Demo server | `scripts/caem_demo_server.py` | Basic FastAPI wrapper, not production-grade |

### Missing (designed in this doc and the runbook, not coded)

| Component | What it does | Estimated effort |
|---|---|---|
| `scripts/build_cycle_calibration_batch.py` | Stratified sampling from serving log | 0.5 days |
| `scripts/join_labels_to_batch.py` | Glue labels onto sampled queries | 0.25 days |
| `scripts/production_sil_finetune.py` | Thin wrapper around `SIL.run_cycle` | 0.5 days |
| `scripts/swap_production_model.py` | Admin-endpoint call to hot-swap LoRA | 0.5 days |
| `scripts/score_calibration_batch.py` | Re-run pipeline on labeled batch | 0.5 days |
| `scripts/swap_production_calibration.py` | Admin-endpoint call to hot-reload calibration | 0.25 days |
| `scripts/production_retroverify.py` | Wrapper around `retroactive_reverification` | 0.5 days |
| `scripts/recalibrate_conformal_at_cycle.py` | EMA-smoothed τ refit with previous-cycle anchor | 1 day |
| `scripts/write_cycle_audit.py` | Capture before/after metrics into audit JSON | 0.25 days |
| Cycle-trigger watcher daemon | Background process tailing serving log | 1 day |
| Serving log writer in demo server | Structured log line per query | 0.5 days |
| Production server with admin endpoints | Replace demo server with one supporting hot-swap | 2 days |
| Labeling pipeline integration | Connect to oracle LLM / feedback / review queue | 1-3 days depending on choice |
| Drift detection dashboard | Compute and display §13 metrics | 1-2 days |
| Domain classifier (Strategy B) | If using auxiliary tagging | 2-5 days for a simple classifier |

**Total estimated effort: 10 - 15 person-days** for a basic production deployment, plus the operator-specific labeling pipeline integration.

---

## 16. Implementation roadmap

### Phase A — Wire up the cycle refresh (1 week, 1 engineer)

Goal: get §11 running end-to-end with a manual cycle trigger.

1. Build `build_cycle_calibration_batch.py` and `join_labels_to_batch.py`.
2. Build the SIL fine-tune wrapper (`production_sil_finetune.py`).
3. Build the rescore + hot-swap wrappers.
4. Build the retroverify wrapper.
5. Build the audit writer.
6. Run a manual end-to-end cycle on a small synthetic serving log to verify the pipeline.

Output: cycle refresh works manually. No labeling pipeline yet.

### Phase B — Connect the labeling pipeline (3-5 days, 1 engineer)

Goal: replace manual labeling with the operator's chosen pipeline.

1. Pick the labeling strategy (LLM oracle is the recommended start).
2. Implement the API integration.
3. Validate label quality on a held-out gold set.
4. Wire labels into the cycle refresh script.

Output: cycle refresh runs end-to-end without manual intervention.

### Phase C — Stand up the production server (2-3 days, 1 engineer)

Goal: replace the demo server with a production-grade server.

1. Fork `caem_demo_server.py` into a production server.
2. Add structured serving-log writer.
3. Add admin endpoints for hot-swap LoRA, hot-reload calibration.
4. Add health checks, readiness probes, metrics endpoints.
5. Containerize and deploy behind a load balancer.

Output: production server serving live traffic, logging serving log entries.

### Phase D — Cycle trigger and monitoring (2-3 days, 1 engineer)

Goal: automate cycle scheduling and drift detection.

1. Build the cycle-trigger watcher daemon (tail serving log, decide when to fire).
2. Build the drift detection dashboard.
3. Set up alerts on drift thresholds.

Output: fully autonomous production CAEM. Operator only intervenes on alerts.

### Phase E — Optional enhancements (variable)

- Per-tag τ_RUC refit (Phase 1f from the research thesis).
- Adaptive τ_RUC across cycles (Phase 2 / v3).
- Multi-region deployment with shared memory store.
- A/B testing infrastructure for cycle-boundary calibration validation.

---

## 17. Frequently confused points

### "Operator means the user, right?"

No. The operator is the team running CAEM (us). End users only send queries; they don't tag, label, or interact with cycle boundaries. The runbook's instruction "operator must tag every query" means **the operator's intake infrastructure tags it automatically**, not that the user types a tag.

### "Do users provide EM labels?"

No. Labels come from the operator's labeling pipeline (LLM oracle, validated feedback proxy, or expert review). End users at most provide implicit signals (thumbs up/down) that the operator validates as a proxy.

### "Is the SIL pool the same as the cal data?"

No. They're disjoint samples from the same serving log:
- **Cal data**: ~3000 stratified samples per cycle, operator-labeled, used to refit calibration knobs.
- **SIL pool**: ~50000 STORE-class entries accumulated in memory_store during the cycle, used to fine-tune LoRA.

The cal data is small + expensive + precise. The SIL pool is big + cheap + noisy. Each is essential for its own mechanism.

### "Does CAEM use a fixed cal-fold per benchmark in production?"

In research, yes (500 fixed samples per training bench). In production, no — the cal-fold is dynamically sampled from recent serving log at each cycle boundary. An alternative is to use a fixed operator-curated eval set per domain, similar to research mode; both are supported by the same downstream refit logic.

### "Memory grows only at cycle boundaries, right?"

No. Memory grows continuously during the cycle. Every STORE decision adds an entry to memory_store immediately. The cycle boundary only **prunes** memory via retroverify, never blocks growth.

### "Per-bench composite breaks production because we don't know the bench?"

No. The operator's intake system tags queries (Strategy A, B, or C from §8). The per-bench composite then dispatches by tag. Untagged traffic falls back to the pooled composite — performance degrades to vanilla CAEM, but the architecture still works.

### "Is the production architecture different from research?"

No. The architecture is identical. The differences are:
- Data source: live traffic vs benchmark pool.
- Cycle trigger: accumulation/calendar vs question budget.
- Cal-fold source: dynamic sample vs fixed split.
- Labels: operator pipeline vs benchmark gold.

Every research-mode mechanism applies the same way in production.

### "Why do we need to rescore the cal fold under the new model? We already have signals in the serving log."

Because the signals in the serving log were under the previous cycle's model. After SIL fine-tunes a new LoRA, the model's signal distribution shifts (that's the whole point of SIL). Refitting calibration on stale signals would lock CAEM to the old model's biases. Step 5 of §11 rescore the labeled batch under the new LoRA so the calibration fit operates on fresh post-SIL signal vectors.

### "Could the operator skip cycle boundaries entirely and just use the research-final calibration forever?"

Technically yes, but the calibration would drift as the live distribution shifts. Within 3-6 months on most deployments, expect storage rate to drift by 20-30%, abstention rate to spike on out-of-distribution queries, and Tier 1 hit rate to plateau. Cycle boundaries are the calibration-discipline that keeps CAEM accurate over time.

### "Is the labeling cost prohibitive?"

For LLM oracle (Option 1 in §9): ~$3-$30 per cycle. Negligible at any deployment scale.
For expert review (Option 3): $30k-$300k per cycle. Only justified for medical/legal deployments.
For validated user-feedback proxy (Option 2): one-time validation cost (~$10k-$50k), then free per-cycle. Best for high-volume consumer products.

A general-purpose product can run on LLM oracle labeling for $300-$3600/year of cycle costs. Small relative to LoRA fine-tune compute.

---

## Cross-references

- `PRODUCTION_RUNBOOK.md` — command-by-command operator manual for §11 of this doc.
- `docs/PHASE_PROGRESSION.md` (if exists) — research-mode trajectory progression.
- `branch_C_log.md` — design diary entries for the production architecture decisions.
- `caem/sil.py` — `SelfImprovementLoop.run_cycle` is the heart of step 3.
- `caem/verification/verifier.py:622` — `reload_calibration` is the hot-reload entry.
- `scripts/fit_composite_calibration.py` — per-tag composite refit.
- `scripts/run_calibration.py` — T scalar refit.

---

## Document maintenance

This document should be updated when:

- A wrapper script in §15 ships (move it from "missing" to "implemented").
- A new dispatch knob is added (e.g., per-tag τ_RUC lands in Phase 1f).
- An operator deploys CAEM in production and reports practical learnings.
- The labeling pipeline options change (e.g., a new oracle model becomes preferred).

The runbook and this design doc are paired. Keep both in sync.
