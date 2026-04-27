# CAEM Production Deployment Runbook

**Last updated:** 2026-04-27
**Audience:** the operator deploying CAEM after research finishes (Phase 1a + Phase 1b complete). Assumes you already have a trained model checkpoint, a populated episodic memory, a locked CalProbComposite + ConformalStorageGate, and a labeled calibration fold from research. This document explains every step of running CAEM in production and how to operate the cycle boundary in a real deployment.

**Goal of this document:** an operator should be able to follow this end-to-end, in order, without rethinking any decisions. Every "do X" line below has a "why" line and a "when" line beneath it.

---

## 1. Why production is different from research

CAEM is **cyclic in both research and production**. Every cycle ends with the same three operations: a self-improvement fine-tune (SIL), a recalibration of T + isotonic + conformal τ on a fresh labeled calibration fold, and a retroactive re-verification pass. The architecture, the verifier, the storage gate, the memory store — all the same.

The differences between research and production are about **scheduling and label sourcing**, not about which operations run:

| Aspect | Research mode (Phase 1a) | Production mode |
|---|---|---|
| **Cycle trigger** | Fixed: 3000 fresh queries × 3 training benchmarks per cycle, 10 cycles total | Accumulation-driven: a cycle ends when N STORE-class admissions have accumulated, OR a calendar trigger fires (e.g., every Sunday, or every quarter) |
| **Cycle length** | ~7–8 hours of GPU time on RTX 5090 | Could be days, weeks, or months — depends on traffic volume and cycle policy |
| **Label sourcing** | Pre-curated calibration fold (1500 labeled samples drawn from training-benchmark dev splits) | Offline labeling pass on a stratified sample of the cycle's accumulated STORE/DEFER/ABSTAIN/DISCARD entries |
| **Cycle-boundary timing** | Inline with the runner — eval pass blocks until cycle finishes | Asynchronous — cycle-boundary operations run in a maintenance window, not on the user-facing serving path |
| **Retention probe** | MMLU on the 200-sample held-out probe at every cycle boundary | Same MMLU probe, OR a production-stable held-out distribution (e.g., a labeled sentinel batch) |
| **Per-cycle SIL fine-tune** | YES, every cycle | YES, every cycle — production cycles ARE smaller research cycles |

**The single most important takeaway:** production is the same architecture, just operating on user-traffic time rather than benchmark-pool time. Every operation that runs at a research cycle boundary also runs at a production cycle boundary. The recalibration discipline that this thesis registers as essential at every cycle boundary in research is essential at every cycle boundary in production. Frozen-calibration-between-cycles is not a thing in either mode.

---

## 2. The three calibration surfaces in plain English

Three calibration surfaces re-fit at every cycle boundary in both research and production.

### Surface 1 — Temperature scalar T

**What it is:** a single number multiplied into the composite's logit space. When `T = 1.0` outputs are unchanged. When `T < 1.0` outputs become sharper; when `T > 1.0` outputs become softer.

**What it controls:** the gap between predicted probability and observed accuracy (Expected Calibration Error). If the model says "I'm 80% confident" and gets 80% right, T is correctly fitted.

**How it's fitted:** by minimising ECE on a labeled calibration fold. **Needs gold labels.**

**Where it lives:** `config.temperature_scalar` (a float in CAEMConfig).

**In production:** re-fits at every cycle boundary against the cycle's labeled calibration fold (built per §5.2). Same script as research: `scripts/run_calibration.py --temperature_only`.

### Surface 2 — Per-signal isotonic curves + conformal τ thresholds

**What it is:** ten one-dimensional monotone curves (one per verifier signal) mapping raw signal values to calibrated probabilities of being correct. Plus τ_store and τ_defer thresholds at fixed precision targets (95% for τ_store, 60% for τ_defer).

**What it controls:** the relationship between raw verifier signals and "correct against gold," AND the precision contract on the storage class.

**How it's fitted:**
- 10 isotonic curves: sklearn `IsotonicRegression` per signal on the labeled calibration fold.
- τ values: walk fold sorted descending by composite, pick the largest threshold at which empirical precision still clears the target. EMA-smooth against the previous cycle's τ at α=0.7.

**Both fits need gold labels.**

**Where it lives:** two JSON files at `outputs/production/composite_calibration.json` and `outputs/production/conformal_gate.json`.

**In production:** re-fits at every cycle boundary. Same script as research: `scripts/recalibrate_conformal_at_cycle.py`.

### Surface 3 — Label-free quantile-only τ EMA refit

**What it is:** the same τ_store and τ_defer, but re-fitted at running quantiles of u_stored values (e.g., the 5% percentile of u_stored values).

**What it controls:** keeps τ tracking the SHAPE of the score distribution. Does NOT track precision (precision is a labeled concept).

**How it's fitted:** reads u_stored values from recent traffic; picks τ at the registered quantile; EMA-smooths against the previous τ.

**Does NOT need labels.**

**Role in production:** the SAME role as in research — a sanity-check mirror that runs alongside the labeled refit. If the labeled τ and the quantile τ ever diverge sharply at a cycle boundary, that's a signal something is wrong with the labeling pipeline.

---

## 3. One-time production setup

Do this once when first deploying.

### 3.1 — Take the research-final calibration as the production baseline

```bash
RESEARCH_LAST_CYCLE=$(ls -d outputs/full_run/cycle_* | sed 's#.*cycle_##' | sort -n | tail -1)
mkdir -p outputs/production
cp outputs/full_run/cycle_${RESEARCH_LAST_CYCLE}/composite_calibration.json outputs/production/
cp outputs/full_run/cycle_${RESEARCH_LAST_CYCLE}/conformal_gate.json outputs/production/
cp outputs/full_run/cycle_${RESEARCH_LAST_CYCLE}/calibrated_config_cycle${RESEARCH_LAST_CYCLE}.json outputs/production/calibrated_config.json
```

**Why:** the LAST research cycle's calibration is closer to the deployed model than Cycle 0. Snapshot it as `outputs/production/` so the verifier reads from a stable path.

**When:** once, immediately after research finishes and before turning production traffic on.

### 3.2 — Lock the production-mode config flags

Edit `caem/config.py` (or override at server startup):

```python
config.composite_calibration_path = "outputs/production/composite_calibration.json"
config.conformal_gate_path        = "outputs/production/conformal_gate.json"
# Cycle-boundary recalibration stays ON (same as research):
config.skip_per_cycle_temperature  = False    # T re-fits at every production cycle boundary
config.adaptive_thresholds_per_cycle = True   # isotonic + conformal τ re-fit at every production cycle boundary
config.adaptive_thresholds_ema_alpha = 0.7    # same EMA factor as research
```

**Why each line:**

- **`composite_calibration_path` / `conformal_gate_path`** — point the verifier at the production JSON paths so a server restart loads production-canonical calibration, not research-default Cycle 0.
- **`skip_per_cycle_temperature = False`** — keep T re-fitting at every production cycle boundary. Production cycles SIL-fine-tune just like research cycles, so the calibration drift defense is needed.
- **`adaptive_thresholds_per_cycle = True`** — keep the isotonic + conformal τ re-fit at every production cycle boundary. Same reason.
- **`adaptive_thresholds_ema_alpha = 0.7`** — same EMA factor as research; the smoothing prevents single-cycle batch noise from shoving τ to a degenerate value.

**When:** once, after step 3.1, before serving any production traffic.

### 3.3 — Persist the production memory store

```bash
mkdir -p outputs/production/memory_store
cp outputs/full_run/memory_store_cycle_${RESEARCH_LAST_CYCLE}.faiss outputs/production/memory_store/memory_store.faiss
cp outputs/full_run/memory_store_cycle_${RESEARCH_LAST_CYCLE}.meta  outputs/production/memory_store/memory_store.meta
```

**Why:** the production memory store carries forward the research trajectory's verified episodes. Tier 1 retrieves from here. Subsequent production cycles add to it; periodic retroverify removes stale entries.

**When:** once, alongside 3.1.

### 3.4 — Define the production cycle-trigger policy

Pick ONE strategy for when a production cycle boundary fires:

| Strategy | Trigger condition | Best for |
|---|---|---|
| **Accumulation-based** | N STORE-class admissions accumulated since last cycle (suggested N = 1000–3000) | High-traffic deployments where cycles complete in days |
| **Calendar-based** | Fixed schedule (e.g., every Sunday at 02:00, every month, every quarter) | Low/predictable traffic, audit-friendly |
| **Hybrid** | Earliest of (N STORE admissions, K days elapsed) | Most production deployments |
| **Drift-triggered** | §6.2 drift signal fires before either of the above | Strongly recommended as a SAFETY override on top of one of the above |

Record your choice + the threshold values:

```bash
cat > outputs/production/cycle_policy.json <<EOF
{
  "strategy": "hybrid",
  "n_store_admissions_threshold": 2000,
  "max_days_between_cycles": 60,
  "drift_override_enabled": true
}
EOF
```

**Why:** without this, the operator has to remember when to fire a cycle. The policy file is the trigger spec.

**When:** once, before turning production traffic on.

### 3.5 — Initialise the production audit log

```bash
mkdir -p outputs/production/audit
echo "deployment_started_at=$(date -u --iso-8601=seconds)" > outputs/production/audit/deployment.log
echo "research_last_cycle=$RESEARCH_LAST_CYCLE" >> outputs/production/audit/deployment.log
git -C /workspace/caem rev-parse HEAD >> outputs/production/audit/deployment.log
```

**Why:** every production cycle below will append to this audit trail. If a cycle ever produces unexpected results, this is the rollback anchor.

**When:** once, alongside 3.1.

---

## 4. Per-query operating mode (what runs every time a user asks a question)

Every user query goes through the standard 8-stage pipeline:

```
User query
  ↓
Stage 1: Pre-routing confidence (u_pre, no labels needed)
  ↓
Stage 2: Encode + retrieve nearest memory entry
  ↓
Stage 3: Three-tier router (Tier 1 memory / Tier 2 zero-shot / Tier 3 RAG)
  ↓
Stage 4: Generate the answer (only Tier 2 + Tier 3)
  ↓
Stage 5: Verifier scores 10 signals (reads current T + composite + gate)
  ↓
Stage 6: Calibrated composite + conformal gate → STORE / DEFER / ABSTAIN / DISCARD
  ↓
Stage 7: Return the served answer (or refusal if ABSTAIN)
```

**Important production behavior at each stage:**

- **Stage 5–6: the verifier reads the CURRENT cycle's calibration** — whichever JSONs are at `outputs/production/composite_calibration.json` and `conformal_gate.json` at this moment. After every production cycle boundary completes, those JSONs are overwritten with the new cycle's calibration; the live verifier picks them up via `reload_calibration` (no server restart needed).
- **Stage 6: the four-outcome decision tree fires.** STORE entries → memory FAISS index + write-ahead audit log. DEFER → deferred buffer. ABSTAIN → returned to user. DISCARD → dropped.
- **Cycle-boundary operations (SIL fine-tune, recalibration, retroverify) run ASYNCHRONOUSLY** in a maintenance window — NOT on the per-query serving path. The serving path stays cheap (sub-second).

**Logging discipline for cycle-boundary feed:**

For every query, log to `outputs/production/serving_log/YYYY-MM-DD.jsonl`:

```jsonl
{"timestamp": "...", "query_id": "...", "decision": "STORE", "u_stored": 0.71, "u_token": 0.84, "u_dropout": 0.12, ..., "served_answer": "...", "tier": 1}
```

**Why:** the cycle-boundary recalibration needs the per-query signal vectors and (after labeling) EM=0/1 labels. The serving log captures the signals at query time; labels get joined in offline.

**When:** every query. Use a structured logger with daily file rotation. Always include `query_id` so labels can be joined back later.

**Cycle-counter watcher:** a background process should monitor `outputs/production/serving_log/` and trigger §5 when the cycle policy from 3.4 fires.

---

## 5. Production cycle-boundary recipe (the per-cycle refresh)

Run this **at every production cycle boundary** — whenever the policy in 3.4 fires.

### 5.1 — Build the cycle's labeled calibration fold

```bash
CYCLE=N    # the new production cycle number, e.g., production cycle 7
mkdir -p outputs/production/cycle_${CYCLE}/calibration

# Sample ~1500-3000 entries from the cycle's accumulated serving log,
# stratified by decision class so each class has enough mass for the fit.
python scripts/build_cycle_calibration_batch.py \
    --serving_log_dir outputs/production/serving_log/ \
    --since_last_cycle outputs/production/audit/refresh_$((CYCLE-1)).json \
    --n_per_class 750 \
    --output outputs/production/cycle_${CYCLE}/calibration/sampled_queries.jsonl
```

**Why:** the cycle-boundary recalibration fits T, isotonic curves, and τ. Each fit needs labeled examples. ~3000 labeled samples (~750 per decision class) is enough for stable isotonic fits and stable τ at the registered precision target.

**When:** the moment the cycle policy fires.

### 5.2 — Send the batch for human (or oracle) labeling

Three sourcing options, in preference order:

1. **Automated post-hoc oracle:** a stronger LLM judge (e.g., GPT-class) scores EM=0/1 on the served answer against a fresh retrieval; cheapest and fastest.
2. **User-feedback signals:** thumbs up/down or downstream click signals, IF validated as a proxy for EM.
3. **Human expert review:** for high-stakes deployments where (1) and (2) aren't reliable.

Output: a `labeled.jsonl` where every record has:
```jsonl
{"query_id": "...", "served_answer": "...", "em": 1, "labeler": "expert_review", "labeled_at": "..."}
```

**Why:** the calibration surfaces fit against EM=0/1 binary labels. No labels means no fit.

**When:** in the first ~25% of the cycle window. For a quarterly cadence, weeks 1–2.

### 5.3 — Score the labeled batch through the live verifier under the post-SIL model

Production's cycle boundary follows the corrected research order: **SIL fine-tune first, then score the calibration fold under the post-SIL model, then recalibrate, then reload, then retroverify.**

```bash
# (A) SIL fine-tune — produces the new model checkpoint.
python -m scripts.production_sil_finetune \
    --cycle ${CYCLE} \
    --memory_store outputs/production/memory_store/ \
    --general_data outputs/production/general_data/ \
    --output_checkpoint outputs/production/cycle_${CYCLE}/model.pt
# Why: same SIL.run_cycle as research; trains on STORE-class entries above
# τ_train, applies the L2 anchor toward the previous cycle's parameters,
# probes the retention guard, commits weights only if retention ≥ 0.93.
# When: at the start of the cycle-boundary maintenance window.
# Note: the production_sil_finetune wrapper would be a thin reuse of
# SelfImprovementLoop.run_cycle (with verify_fn=None) — straightforward
# to add when production launches.

# (B) Hot-swap the new checkpoint into the live verifier.
python -m scripts.swap_production_model --new_checkpoint outputs/production/cycle_${CYCLE}/model.pt
# Why: subsequent steps need to score the calibration fold under the NEW
# model, not the previous cycle's model.
# When: immediately after the SIL fine-tune commits weights.

# (C) Score the labeled calibration batch under the now-updated verifier.
python scripts/score_calibration_batch.py \
    --queries outputs/production/cycle_${CYCLE}/calibration/labeled.jsonl \
    --output_dir outputs/production/cycle_${CYCLE}/calibration/scored/
# Why: the calibration fits need fresh signal vectors under the post-SIL
# model. The serving log captured signals before the fine-tune; those are
# stale now.
# When: immediately after (B).
```

**What "good" looks like at this stage:**
- SIL fine-tune commits weights (retention probe ≥ 0.93). If it rolls back, the cycle aborts; skip steps 5.4–5.7 and re-run on the next cycle's calibration fold.
- The scoring pass produces ~3000 JSON rows with all 10 signals + EM labels.

### 5.4 — Re-fit the temperature scalar T

```bash
python scripts/run_calibration.py \
    --calib_dir outputs/production/cycle_${CYCLE}/calibration/scored/ \
    --temperature_only \
    --previous_T_path outputs/production/calibrated_config.json \
    --output outputs/production/cycle_${CYCLE}/calibrated_config.json
```

**Why:** T re-fit absorbs the post-SIL ECE drift. Output JSON includes `temperature_before`, `temperature_after`, `ece_before`, `ece_after` so the audit logs whether the refit improved calibration.

**When:** immediately after 5.3.

**What "good" looks like:** `ece_after ≤ ece_before` AND `temperature_after` within ±20% of `temperature_before`. A 5× change suggests a model degradation upstream — investigate before deploying.

### 5.5 — Re-fit the per-signal isotonic curves and conformal τ

```bash
python scripts/recalibrate_conformal_at_cycle.py \
    --calib_jsons outputs/production/cycle_${CYCLE}/calibration/scored/*.json \
    --previous_composite outputs/production/composite_calibration.json \
    --previous_gate      outputs/production/conformal_gate.json \
    --output_composite   outputs/production/cycle_${CYCLE}/composite_calibration.json \
    --output_gate        outputs/production/cycle_${CYCLE}/conformal_gate.json \
    --ema_alpha 0.7 \
    --alpha_store 0.05 \
    --alpha_defer 0.40 \
    --cherian_boost
```

**Why:** the 10 isotonic curves re-fit fresh on the post-SIL labeled batch. The two τ values fit at the precision targets and EMA-smooth against the previous cycle's τ at α=0.7.

**When:** immediately after 5.4.

**What "good" looks like:**
- New τ_store within ±0.10 of previous τ_store (large drift = real distribution shift; review).
- Empirical precision on the labeled batch at the new τ_store ≥ 95%.
- `n_store` on the batch ≥ 20 (smaller = unstable conformal fit; re-sample with higher STORE-class stratification weight).

### 5.6 — Reload the live verifier from the new calibration JSONs

```python
# In the production server's admin REPL:
new_composite = f"outputs/production/cycle_{CYCLE}/composite_calibration.json"
new_gate      = f"outputs/production/cycle_{CYCLE}/conformal_gate.json"
new_T_config  = f"outputs/production/cycle_{CYCLE}/calibrated_config.json"

# (a) Reload verifier composite + conformal gate.
pipeline.verifier.reload_calibration(new_composite, new_gate)

# (b) Update T from the new calibrated_config.json.
import json
with open(new_T_config) as f:
    pipeline.config.temperature_scalar = json.load(f)["temperature_after"]
```

**Why each step:**
- **(a)** swaps the verifier's cal-prob composite + conformal gate from the previous cycle's JSONs to this cycle's. Cost ~milliseconds. Underlying judges (MiniCheck, BGE reranker, SBERT) are NOT touched.
- **(b)** T is read from `pipeline.config.temperature_scalar` at every inference call, so updating it in-process applies on the very next query.

**When:** immediately after 5.5.

### 5.7 — Run retroactive re-verification under the recalibrated verifier

```bash
python -m scripts.production_retroverify \
    --memory_store outputs/production/memory_store/ \
    --threshold 0.50 \
    --backup_first
```

**Why:** retroverify re-scores every memory entry under the now-recalibrated verifier. Entries below `τ_retro = 0.50` get pruned; entries above stay (and have their composite refreshed). This is the production mirror of research-mode retroverify, and it follows the SAME corrected ordering: recalibrate first, then retroverify, so the prune verdict is on freshly-fitted curves rather than stale ones. Note: `scripts/production_retroverify.py` is a thin wrapper around `retroactive_reverification` from `run_experiment.py`; the wiring is straightforward when production launches.

**When:** immediately after 5.6.

**What "good" looks like:**
- Retroverify prunes <20% of memory in a typical cycle. >40% is a red flag — investigate whether the post-SIL model has drifted or the calibration fit is unstable.
- `--backup_first` archives the previous memory snapshot to `outputs/production/cycle_${CYCLE-1}/memory_store/`, so a roll-back to the pre-retroverify state is one file copy away.

### 5.8 — Promote the new calibration to canonical production paths

```bash
cp outputs/production/cycle_${CYCLE}/composite_calibration.json outputs/production/composite_calibration.json
cp outputs/production/cycle_${CYCLE}/conformal_gate.json        outputs/production/conformal_gate.json
cp outputs/production/cycle_${CYCLE}/calibrated_config.json     outputs/production/calibrated_config.json
```

**Why:** the canonical paths are what a server restart reads at boot. Without this copy, a restart would silently fall back to the previous cycle's calibration even though the in-process `reload_calibration` succeeded.

**When:** after 5.7 confirms retroverify produced a healthy memory size.

### 5.9 — Audit the cycle

```bash
cat > outputs/production/audit/refresh_${CYCLE}.json <<EOF
{
  "cycle": ${CYCLE},
  "completed_at": "$(date -u --iso-8601=seconds)",
  "git_commit": "$(git -C /workspace/caem rev-parse HEAD)",
  "n_labeled_samples": <FILL_IN>,
  "sil_fine_tune": {
    "n_episodes_used": <FROM 5.3.A LOG>,
    "retention_ratio": <FROM 5.3.A LOG>,
    "aborted": <FROM 5.3.A LOG>,
    "final_train_loss": <FROM 5.3.A LOG>
  },
  "T_before": <FROM 5.4>,
  "T_after": <FROM 5.4>,
  "ece_before": <FROM 5.4>,
  "ece_after": <FROM 5.4>,
  "tau_store_before": <FROM 5.5>,
  "tau_store_after": <FROM 5.5>,
  "tau_defer_before": <FROM 5.5>,
  "tau_defer_after": <FROM 5.5>,
  "store_precision_on_calib": <FROM 5.5>,
  "retroverify": {
    "memory_size_before": <FROM 5.7>,
    "memory_size_after": <FROM 5.7>,
    "n_pruned": <FROM 5.7>
  }
}
EOF
```

**Why:** the audit JSON is the rollback anchor for the next cycle. If next cycle's metrics look bad, the previous audit JSON identifies the last-known-good state.

**When:** immediately after 5.8.

---

## 6. Monitoring + drift detection

### 6.1 — Daily metrics dashboard

Compute these from the serving log every day:

| Metric | Healthy range | Alert if |
|---|---|---|
| Storage rate | drift from previous-cycle ±20% | drift > ±30% over 7 days |
| Defer rate | drift from previous-cycle ±20% | sudden 2× change overnight |
| Abstain rate | drift from previous-cycle ±20% | drift > ±50% over 7 days |
| Mean u_stored on STORE | drift from previous-cycle ±0.05 | drift > ±0.10 |
| Tier 1 hit rate | grows slowly over time | sudden drop |
| Tier 3 fallback rate | shrinks as memory grows | sudden growth |

### 6.2 — Drift-triggered cycle boundaries

If ANY of the following fires, run §5 immediately, regardless of where the accumulation/calendar trigger sits:

- **Storage rate drifted >30% from last cycle's value** for 7 consecutive days.
- **Mean u_stored on STORE drifted >0.10 from last cycle's value** for 7 consecutive days.
- **User-traffic distribution shift** (manual flag) — e.g., a product launch pulls traffic into a new domain.
- **Model checkpoint changed outside the SIL loop** — e.g., a manual hot-fix patch shipped a new generator.

### 6.3 — On-demand retroactive re-verification

Retroverify happens at every cycle boundary as part of §5.7. You generally do NOT run extra retroverifies between cycle boundaries — it's expensive (O(memory_size × verify_cost)) and the cycle boundary already runs it. The exception is when memory FAISS is suspected of corruption, in which case run §7.2 recovery.

---

## 7. Failure modes and recovery

### 7.1 — Cycle aborts due to retention failure

**Symptoms:** SIL fine-tune at 5.3.A reports `aborted=True` because the retention ratio fell below 0.93. Weights rolled back to the previous cycle's checkpoint.

**Recovery:** the architecture's own design is the recovery — memory survives the rollback, weights revert. Skip steps 5.4–5.8; the cycle is recorded as aborted in the audit log; the next cycle re-attempts with whatever new memory has accumulated since.

**Why this works:** the asymmetric-rollback property guarantees memory accumulates strictly across cycles even when weights don't. An aborted cycle costs the cycle's fine-tune compute but doesn't lose verified episodes.

### 7.2 — Memory FAISS corruption

```bash
PREV_CYCLE=$((CYCLE - 1))
cp outputs/production/cycle_${PREV_CYCLE}/memory_store/memory_store.faiss outputs/production/memory_store/memory_store.faiss
cp outputs/production/cycle_${PREV_CYCLE}/memory_store/memory_store.meta  outputs/production/memory_store/memory_store.meta
# Restart the production server.
```

**Why:** every cycle's `--backup_first` retroverify run snapshotted the previous memory state. Roll back is a file replacement.

### 7.3 — Cycle produces obviously bad calibration

**Symptoms:** post-cycle storage rate spikes 5×, OR mean u_stored on STORE drops >0.20, OR empirical precision on the calibration batch <80%.

**Recovery:**

```bash
PREV_CYCLE=$((CYCLE - 1))
cp outputs/production/cycle_${PREV_CYCLE}/composite_calibration.json outputs/production/composite_calibration.json
cp outputs/production/cycle_${PREV_CYCLE}/conformal_gate.json        outputs/production/conformal_gate.json
cp outputs/production/cycle_${PREV_CYCLE}/calibrated_config.json     outputs/production/calibrated_config.json
# In the live server REPL:
pipeline.verifier.reload_calibration(
    "outputs/production/composite_calibration.json",
    "outputs/production/conformal_gate.json",
)
```

**Why this works:** every cycle preserves its calibration JSONs at `outputs/production/cycle_${N}/`. Roll back is a file copy + a single `reload_calibration` call. The model checkpoint also rolls back: `cp outputs/production/cycle_${PREV_CYCLE}/model.pt outputs/production/current_model.pt`.

### 7.4 — Hot-swap reload fails

**Symptoms:** `reload_calibration` raises an exception. Verifier in indeterminate state.

**Recovery:**
1. Verify JSON files: `python -c "import json; json.load(open('...'))"`.
2. If JSON is fine, **restart the server.** New JSONs at `outputs/production/` get loaded by `_init_cal_prob_composite` and `_init_conformal_gate` on init.

---

## 8. Does manual cycle-boundary refit downgrade performance?

**Short answer: no, refit cannot downgrade performance — it RESTORES it.** The cycle boundary in production is the same architectural pattern as research-mode cycle boundaries. The thesis-registered theorems on convergence and pool purity all assume the cycle-boundary recalibration fires; they remain in force when production runs the same cycle structure.

### 8.1 — Why refit cannot downgrade performance

Each refit fits a calibration surface against fresh labeled data:

- **T re-fit minimises ECE.** Post-refit ECE ≤ pre-refit ECE on the labeled batch (the fit is the minimum). If the previous T was already optimal on this batch, the new T equals it.
- **Isotonic curves are uniquely determined by the data.** sklearn `IsotonicRegression` is deterministic — same labels = same curves. No noisy initialisation, no local minimum.
- **Conformal τ refit picks τ at the registered precision target.** EMA smoothing against the previous cycle's τ (α=0.7) prevents single-batch noise from dragging τ to a degenerate value.

Each refit either improves the fit-quality metric on the labeled batch or holds it constant. **There's no scenario where a refit produces worse calibration than what came before, on the same labeled batch.**

### 8.2 — But what about between cycle boundaries?

Between cycle boundaries (within a cycle), the calibration surfaces ARE frozen at the previous cycle's values, because that's when the cycle boundary last refreshed them. Performance can drift in three ways within a cycle:

**Drift mode 1: Live signal distribution shifts.** User traffic distribution changes mid-cycle; raw signals shift; frozen isotonic curves map raw → calibrated probability via curves fitted on the previous cycle's distribution. Calibrated probabilities become noisier.

**Drift mode 2: Storage rate drifts.** As memory grows mid-cycle, the score distribution shifts. The label-free quantile EMA τ partially compensates by tracking distribution shape, but precision tracking requires the labeled refit at the next cycle boundary.

**Drift mode 3: Precision contract slips.** The 95% precision floor for STORE is bound by the labeled refit. Without a labeled refit, realized precision in production can drift. After a long cycle (say 60 days at low traffic) realized precision could be 92% or 96% depending on how user-traffic distribution moved. The cycle boundary re-binds the contract.

**Bottom line for your question:** the cycle boundary IS the architectural defense. It's not a separate "production-only" operation — it's the same operation that runs at every research cycle boundary. The drift you might worry about is the time WITHIN a cycle where calibration is fitted on the previous cycle's data, not the moment of refit.

### 8.3 — How long can production cycles be before drift hurts?

This depends on traffic volume and traffic stability. Empirical guidance:

| Cycle length | Expected ECE inflation at end of cycle | Expected precision contract drift | Decision |
|---|---|---|---|
| 1–7 days | <1% | <1pp | normal operation |
| 7–30 days | 1–3% | 1–2pp | normal operation, monitor §6.1 |
| 30–60 days | 2–5% | 2–4pp | reasonable, especially with stable traffic |
| 60–90 days | 4–8% | 3–6pp | upper bound for a typical production cycle |
| 90+ days | 5–15% | 5–10pp | precision contract no longer reliable; cycle is overdue |

These are heuristics. If the §6.1 dashboard shows numbers shifting fast, fire the cycle boundary sooner via the §6.2 drift trigger. If they're stable, the accumulation/calendar trigger is fine.

### 8.4 — What's NOT a downgrade vector

- **Frequent cycle boundaries** are fine. More frequent refit costs more labeling effort, but produces tighter calibration, not worse.
- **Smaller calibration batch size** is mostly fine, with limits: <500 samples produces unstable isotonic curves; <100 STORE-class entries produces unstable τ. Stay above ~1500 total.
- **Cycle-boundary after a major model retraining** is REQUIRED, not optional. The signal distribution shifts dramatically when the generator changes outside the SIL loop. Treat post-retraining as a §3 redeployment from scratch.
- **Stratified sampling for the labeled batch** is correct, not a corner-cut. Random sampling under-represents the small STORE class and over-represents DISCARD; stratification gives each class enough mass for stable per-class fit.

---

## 9. Future automation

The following are NOT required for production operation but reduce manual toil:

1. **Automated cycle-trigger watcher:** background process reads the §6.1 dashboard + §3.4 policy and fires §5 when conditions met.
2. **Online labeling pipeline:** integrate user feedback signals (👍/👎) AND automated post-hoc oracles so labels arrive without manual review delay.
3. **Cycle CI:** scripted §5.1–5.9 as a single job, with a human sign-off step before §5.6 (the actual hot-swap).
4. **Drift dashboard:** materialise §6.1 metrics into a Grafana / equivalent dashboard with the §6.2 alerts wired.

---

## 10. Appendix — every config flag explained

| Flag | Default in research | Default in production | What it does |
|---|---|---|---|
| `skip_per_cycle_temperature` | `False` | `False` | Per-cycle T re-fit ON in both modes — cycles always re-fit T against the labeled fold built at cycle boundary. |
| `adaptive_thresholds_per_cycle` | `True` | `True` | Per-cycle isotonic + conformal τ re-fit ON in both modes. Same reason. |
| `adaptive_thresholds_ema_alpha` | `0.7` | `0.7` | EMA smoothing factor for τ refits. Same value research and production. |
| `composite_calibration_path` | `outputs/cycle_0/composite_calibration.json` | `outputs/production/composite_calibration.json` | Path to the cal-prob composite JSON. Production's path is stable across cycles; the file gets overwritten in step 5.8. |
| `conformal_gate_path` | `outputs/cycle_0/conformal_gate.json` | `outputs/production/conformal_gate.json` | Path to the conformal gate JSON. Same convention. |
| `temperature_scalar` | re-fit per cycle | re-fit per cycle | T value applied at every inference. Read from `calibrated_config.json` after each cycle. |
| `retroverify_prune_threshold` | `0.50` | `0.50` | Threshold below which retroverify prunes a stored entry. Same value research and production. |
| `forgetting_tolerance` | `0.93` | `0.93` | The retention floor for SIL commit. Same in production because production also fine-tunes. |

---

**End of runbook.** Print this, follow §3 once at deploy time, follow §5 at every production cycle boundary (whatever cadence your §3.4 policy fires), and you have a self-documenting trail that survives operator turnover.
