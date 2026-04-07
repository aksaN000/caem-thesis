# CAEM — Master Execution & Writing Plan
**Updated: 2026-04-07 | Session 29 — Sequential Flow from Current State to Submission**

---

## Quick Status

| Step | Status | Notes |
|---|---|---|
| Smoke Test (Cycle 0→3) | ✅ DONE | Full pipeline verified on RTX 3060 |
| Cold-start seeding (Gap 3) | ✅ DONE | 447 verified episodes stored |
| Writing prep files | ✅ DONE | `writing-suggestions.md`, `hyperparameter-reference.md`, `caem-implementation-log.md` all updated |
| Visual elements inventory | ✅ DONE | Full figures/algorithms/equations/theorems spec added to `writing-suggestions.md` |
| Ch1 TikZ fix (IMPL-01a) | ✅ DONE | Tier 1 → Stage 5 arrow removed (2026-04-07) |
| C2-04/C2-05 text fix | ✅ DONE | chapter_2.tex "384-dim" → "768-dimensional" (both §2.1.4 and §2.2.4) |
| Mini-run (n=500 per bm) | ⏳ IN PROGRESS | All 4 benchmarks active, restarted clean |
| Full experiment (n=5000) | ❌ PLANNED | Pending lab PC (4090/5090) access |
| Chapter writing | ❌ PLANNED | Ch3+Ch4 can start now; Ch5 blocked on data |

---

## PHASE 0 — Pre-Run Text Fixes ✅ COMPLETED (2026-04-07)

~~C2-04 and C2-05 fixed in `chapter_2.tex` — both "384 dimensional" / "three hundred eighty four dimensional" changed to "768-dimensional".~~

---

## PHASE 1 — Mini-Run Validation (RTX 3060, n=500)

The mini-run is already in progress at `outputs/mini_experiment`. Do NOT restart it again.

### Step 1.1 — Let it complete

Wait for the run to finish. The run command (for reference):
```powershell
python scripts/run_experiment.py `
  --n_questions 100 `
  --benchmarks hotpotqa truthfulqa fever strategyqa `
  --output_dir outputs/mini_experiment `
  --passage_index data/passage_index `
  --cold_start_memory outputs/cold_start_memory/memory_store
```

### Step 1.2 — Validate mini-run outputs

Check each of these files exists and is non-empty:

| File | What to check |
|---|---|
| `outputs/mini_experiment/experiment_summary.csv` | Rows for all 4 benchmarks × 4 cycles |
| `outputs/mini_experiment/eval/hotpotqa_cycle3.json` | per_question array present |
| `outputs/mini_experiment/eval/truthfulqa_cycle3.json` | per_question array present |
| `outputs/mini_experiment/eval/fever_cycle3.json` | per_question array present |
| `outputs/mini_experiment/eval/strategyqa_cycle3.json` | per_question array present |
| `outputs/mini_experiment/memory_store/` | Episodes present for all 4 benchmarks |

### Step 1.3 — Check for accuracy improvement

Open `experiment_summary.csv` and confirm:
- At least one benchmark shows accuracy improvement from Cycle 0 → Cycle 3
- No benchmark shows MMLU retention below 93% (if MMLU eval is included)
- No Python exceptions in the run log

**If mini-run passes:** proceed to Phase 2.
**If mini-run fails:** check `caem-implementation-log.md` EXP-01 through EXP-13 — all known bugs are listed with their fixes.

---

## PHASE 2 — Device Confirmation and Scaling

Before launching the full 5000-question experiment, the exact device must be known.

### Step 2.1 — Confirm experiment device

Answer: will the full run execute on:
- (A) RTX 3060 (local, ~14h — feasible but slow)
- (B) Kaggle (T4/P100, free tier — ~6-8h, needs dataset upload)
- (C) University GPU cluster (A100/H100 — fastest, needs access)
- (D) Rented instance (Vast.ai, Lambda Labs — pay-per-hour)

**This answer determines which scaling changes apply.**

### Step 2.2 — Apply LAB_PC_SCALING_GUIDE changes (if device ≠ RTX 3060)

Read `LAB_PC_SCALING_GUIDE.md` before making any code changes.

| Change | Current (RTX 3060) | After scaling | Config key |
|---|---|---|---|
| Batch size | `batch_size=4` | `batch_size=16` | `caem/config.py` |
| theta_prev storage | CPU (`.detach().cpu()`) | GPU (in-place) | `caem/trainer.py` `_l2_penalty()` |
| Passage index size | 500K passages | 5M passages (optional) | `max_passages` flag |
| Target episodes | 200 | 1,000 | `target_episodes` flag |

See `LAB_PC_SCALING_GUIDE.md` §7 for the exact 3-step theta_prev GPU migration code.

**Note:** If the full experiment runs on a different device than the smoke test, update:
- §3.x hardware constraints in `chapter_3.tex` (actual hardware, not RTX 3060)
- §5.7 efficiency numbers (use `experiment_summary.csv` mean_latency_ms — not smoke-test estimates)

### Step 2.3 — Transfer data to new device (if applicable)

```bash
# Transfer these two directories to the new device:
data/passage_index/          # 1.5 GB — prebuilt FAISS index
outputs/cold_start_memory/   # ~small — 447 seeded episodes
```

---

## PHASE 3 — Full Experiment Execution (n=5000)

### Step 3.1 — Launch inside tmux (Linux/Mac) or equivalent

```bash
tmux new -s caem_full
python scripts/run_experiment.py \
  --n_questions 5000 \
  --benchmarks hotpotqa truthfulqa fever strategyqa \
  --output_dir outputs/full_experiment \
  --passage_index data/passage_index \
  --cold_start_memory outputs/cold_start_memory/memory_store
```

On Windows with PowerShell:
```powershell
python scripts/run_experiment.py `
  --n_questions 5000 `
  --benchmarks hotpotqa truthfulqa fever strategyqa `
  --output_dir outputs/full_experiment `
  --passage_index data/passage_index `
  --cold_start_memory outputs/cold_start_memory/memory_store
```

Expected duration: ~11–14h on RTX 4090, longer on T4/P100.

### Step 3.2 — Monitor for failures

Check every 2–3 hours:
- Log output: no CUDA OOM, no UTF-8 decode errors, no KeyError crashes
- `outputs/full_experiment/experiment_summary.csv` — rows accumulating

If OOM: the fixes for this are in `caem-implementation-log.md` (EXP series). Batch size was already reduced to handle RTX 3060. If running on a larger GPU, batch_size=16 should be fine.

---

## PHASE 4 — Post-Experiment Analysis Scripts

Run these IN ORDER after the full experiment completes. Each produces a specific output file that Chapter 5 sections depend on.

### Step 4.1 — Purity Validation

```bash
python scripts/run_purity_validation.py \
  --output_dir outputs/full_experiment \
  --output_file outputs/purity_validation/theory_validation.json
```

**Produces:** `outputs/purity_validation/theory_validation.json`
**Used by:** C5-03 (§5.4), TH-03, TH-04, C4-17 (replace illustrative values with actual)

### Step 4.2 — Ablation Study

```bash
python scripts/run_ablation.py \
  --output_dir outputs/full_experiment \
  --ablation_output outputs/ablation_results/ablation_summary.json
```

**Produces:** `outputs/ablation_results/ablation_summary.json`
**Used by:** C5-09 (§5.5), C5-07 MMLU eval, C5-06 A2 cost comparison

### Step 4.3 — Calibration Audit

```bash
python scripts/run_calibration.py \
  --output_dir outputs/full_experiment \
  --calibration_output calibration/calibrated_config.json
```

**Produces:** `calibration/calibrated_config.json`
**Used by:** C5-11 (§5.1), GEN-03, C4-07 (actual calibrated weights)

### Step 4.4 — Statistical Significance (McNemar's)

No separate script needed — use `eval/metrics.py` functions directly:

```python
from eval.metrics import mcnemar_test, bootstrap_ci
import json

# Load per-question EM arrays
with open("outputs/full_experiment/eval/hotpotqa_cycle3.json") as f:
    caem_results = json.load(f)["per_question"]

# Load baseline results for the same benchmark
# ... run for each bm × baseline pair
chi2, p_val = mcnemar_test(caem_em, baseline_em)
ci_low, ci_high = bootstrap_ci(caem_em)
```

**Used by:** C5-05, C5-12, GEN-12 (§5.2 statistical significance claims)

---

## PHASE 5 — Data Extraction Checklist

After scripts run, confirm these files exist before writing Chapter 5.

| Output file | Chapter 5 section | Writing entries |
|---|---|---|
| `outputs/full_experiment/experiment_summary.csv` | §5.2, §5.3, §5.7, §5.9 | C5-02, C5-06, C5-10 |
| `outputs/full_experiment/eval/{bm}_cycle{n}.json` | §5.2, §5.4 | C5-05, C5-12, GEN-12 |
| `outputs/purity_validation/theory_validation.json` | §5.4 | C5-03, TH-03, TH-04, C4-17 |
| `calibration/calibrated_config.json` | §5.1 | C5-11, GEN-03, C4-07 |
| `outputs/ablation_results/ablation_summary.json` | §5.5, §5.7, §5.8 | C5-09, C5-07, C5-06 |
| `outputs/retroverify_cycle{n}.json` | §5.8 | C5-07 forgetting scores |

---

## PHASE 6 — Chapter Writing (Sequential Order)

Write chapters in this order. Ch3 and Ch4 do NOT need experiment data and can be written immediately (even before Phase 1 completes). Ch5 is fully blocked on Phase 4–5.

---

### Step 6.1 — Write Chapter 3 (NOW — no data needed)

**File:** `pre thesis 1 report/chapters/chapter_3.tex`
**Status:** STUB (9 lines) — must write from scratch

**Content to cover:**

**§3.1 Functional Requirements** — derive from 8 pipeline stages:
1. Stage 1: Accept natural language queries
2. Stage 2: Compute pre-routing confidence (u_pre) before generation
3. Stage 3: Three-tier routing based on u_pre + û_stored
4. Stage 4: Generate answer with optional RAG context (Tier 2/3)
5. Stage 4a: Post-generation confidence gate (û ≥ 0.60 threshold)
6. Stage 5: Multi-signal verification (NLI + SC + SE)
7. Stage 6: Episodic memory storage with û_stored
8. Stage 8: Cyclic fine-tuning on verified reasoning chains

**§3.2 Non-Functional Requirements:**

| NFR | Target | Source |
|---|---|---|
| Tier 1 latency | < 400ms | Design (retrieval only) |
| Tier 2 latency | < 2.5s | Design |
| Tier 3 latency | < 6s | Design |
| Verification accuracy α | ≥ 85% | Purity theorem requires this for P > p |
| GPU budget | ≤ 50 GPU hours | RTX 3060 constraint |
| MMLU retention | ≥ 93% per cycle | Forgetting abort threshold |

**§3.3 Hardware Constraints as Design Drivers** — use C3-02 guidance:
- VRAM ≤ 12 GB → batch_size=4, theta_prev CPU offload, 500K passage index
- GPU budget ≤ 50h → n_questions=5000, n_cycles=3
- Memory capacity K=1,000 episodes → cold-start seeding strategy
- Frame constraints as a strength: "meaningful hallucination reduction achievable within real-world resource limits"

**§3.4 Societal Impact and Ethical Constraints:**
- Positive: reduces misinformation from LLM-generated content
- Risk: if verification pipeline miscalibrated, could overconfidently store wrong answers (FP rate discussion)
- Scope limitation: benchmarked on QA tasks — not medical/legal advice

---

### Step 6.2 — Write Chapter 4 (NOW — no data needed)

**File:** `pre thesis 1 report/chapters/chapter_5.tex`
**Status:** NOT WRITTEN — write from scratch

**Before writing any section of Chapter 4:** Read the Visual & Formal Elements section of `writing-suggestions.md` to know exactly which figures, algorithms, equations, and theorems belong in each section.

Write sections in this order, applying each writing-suggestions.md entry as you go:

| Section | Key prose entries | Visual elements to include |
|---|---|---|
| §4.1 Architecture Overview | C4-01, C4-03 (three-value table), GEN-01, GEN-11 | **FIG-C4-01** (main architecture) — extend from `chapter_1.tex` line 78 |
| §4.2 Three-Tier Routing | C4-04, C4-10b, C4-20, IMPL-01, BM-01–BM-06 | **FIG-C4-02** (routing flowchart), **ALG-C4-02** (routing algorithm), **EQN-C4-02** (routing score) |
| §4.3 Pre-Routing Confidence | C4-03b, C4-10 | **FIG-C4-03** (confidence architecture) — left panel only, **EQN-C4-01** |
| §4.4 Post-Generation Confidence (û) | C4-05, C4-06, C4-07, C4-09, C4-21 | **FIG-C4-03** right panel, **EQN-C4-03** (û formula), **TAB** (blind-spot 4×4), **TAB** (three-value summary) |
| §4.5 Verification Pipeline | C4-08, C4-22, IMPL-02 | **FIG-C4-04** (verification flowchart), **ALG-C4-03** (verification algorithm), **EQN-C4-04** (û_stored) |
| §4.6 Episodic Memory | C4-15, C4-03 | **FIG-C4-05** (memory schema), **ALG-C4-05** (retroactive re-verification) |
| §4.7 Self-Improvement Loop | C4-16, C4-19, GEN-11 | **FIG-C4-06** (self-improvement cycle), **ALG-C4-04** (Stage 8 algorithm), **EQN-C4-05** (L2 loss) |
| §4.8 Calibration | IMPL-03 | **EQN-C4-06** (temperature scaling), **EQN-C4-07** (ECE) |
| §4.9 Theoretical Analysis | C4-17, C4-18, C4-23, TH-01, TH-02, PUB-02 | **THM-C4-01 + proof**, **THM-C4-02**, **THM-C4-03**, **EQN-C4-08** (purity formula) |

**Algorithm source rule for Chapter 4:**
- FIG-C4-01: base = `chapter_1.tex` line 78 (extend, do NOT redraw)
- ALG-C4-02: base = unified plan line 2627 (formatting only, logic from `caem/router.py`)
- ALG-C4-03: base = unified plan lines 1832 + 2321 + 2441 (SE and SC blocks), combined for Stage 5
- ALG-C4-04: base = unified plan line 3979 (fix θ_base → θ_prev, fix training target to reasoning_chain)
- ALG-C4-05: base = unified plan line 3100 (fix field names)
- THM-C4-01 proof: base = unified plan line 3508 (replace proofbox with standard \begin{proof})
- THM-C4-02/03: base = unified plan lines 4029 + 4185 (theorem environments already correct format)

**Critical writing rules for Chapter 4:**
- Use hyperparameter-reference.md — state Category 1/2/3 provenance for every value
- Field names: `u_token`, `u_dropout`, `u_consistency`, `u_entropy`, `u_hat` (NOT u_sc, h_entropy_norm)
- StoredConfidence fields: `p_entail`, `s_avg`, `h_norm`, `u_stored`
- L2 regularisation (λ=0.01 [DES]) NOT EWC — but name EWC as the reference method
- Convergence claim: NEVER "infinite cycles → zero hallucination". Always: "bounded, monotone improvement"
- Theory section: symbolic only — no experimental numbers. One illustrative example clearly labeled.
- Device: §4.7 L2 penalty section — note CPU offload for RTX 3060; reference GEN-13 scaling rule

---

### Step 6.3 — Write Chapter 5 (AFTER Phase 4–5 complete)

**File:** `pre thesis 1 report/chapters/chapter_6.tex`
**Status:** NOT WRITTEN — blocked on experiment data

Write sections in this order, pulling numbers from specific output files:

| Section | Source file | Key entries |
|---|---|---|
| §5.1 Calibration Results | `calibration/calibrated_config.json` | C5-01, C5-11, GEN-03 | **TAB-C5-07** (calibration results table) |
| §5.2 Main Results | `eval/{bm}_cycle{n}.json` | C5-04, C5-05, C5-12, GEN-12 | **TAB-C5-01** (main results), **TAB-C5-06** (significance), **FIG-C5-01** (accuracy chart) |
| §5.3 Mechanism Evidence | `experiment_summary.csv` | C5-02, IMPL-04 | **TAB-C5-02** (mechanism table), **FIG-C5-02** (tier distribution chart) |
| §5.4 Theory Validation | `purity_validation/theory_validation.json` | C5-03, TH-03, TH-04, C4-17 | **TAB-C5-03** (purity theorem), **TAB-C5-04** (convergence Δ) |
| §5.5 Ablations | `ablation_results/ablation_summary.json` | C5-09 | **TAB-C5-05** (ablation table) |
| §5.6 Baselines | `ablation_results/ablation_summary.json` | C5-09 (baseline sub-section) | (sub-section of TAB-C5-01 or separate table) |
| §5.7 Computational Efficiency | `experiment_summary.csv` mean_latency_ms | C5-06, GEN-13 | **TAB-C5-08** (efficiency table), **FIG-C5-02** tier fraction growth |
| §5.8 Stability-Plasticity | `ablation_summary.json` + `retroverify_cycle{n}.json` | C5-07 | **FIG-C5-03** (MMLU retention chart) |
| §5.9 Memory + Error Analysis | `experiment_summary.csv` + qualitative | C5-10, C5-08 | (qualitative — no figure required) |

**Critical writing rules for Chapter 5:**
- All numbers from output files — never copy from unified plan projections (GEN-02, GEN-09)
- Every baseline comparison requires McNemar's test (p-value) — C5-05, C5-12
- Theory validation tables: actual p, α values from theory_validation.json (TH-03)
- If MMLU retention < 93%: flag as catastrophic forgetting, do not paper over it (C5-07)
- If convergence Δ doesn't decrease: report it honestly, add explanation (TH-04)
- Hallucination rate uses the operational definition: EM=0 AND û ≥ 0.5 (IMPL-02)
- Narrative thread: every result answers one of three questions (C5-04)

---

### Step 6.4 — Write Chapter 6 (AFTER Chapter 5 first draft)

**File:** `pre thesis 1 report/chapters/chapter_9.tex`
**Status:** NOT WRITTEN

§6.1 — Summary of Findings: C6-01, C6-02 (three-theory recap)
§6.2 — Limitations: IMPL-05 (SBERT routing), GEN-04 (NLI ground-truth dependency), GEN-07 (retrieval feedback scope), GEN-08 (inference mode scope)
§6.3 — Future Work: IMPL-06 (distillation, capacity scaling, reference-free NLI), GEN-08
§6.4 — Theoretical Contribution: distinguish CAEM from heuristic self-training; three theories as the formal basis

---

### Step 6.5 — Revise Chapters 1 and 2 (AFTER Chapter 5 complete)

Check after experiments whether:
- **Ch1:** Stated hallucination reduction target matches actual result — if materially different, update the claim (GEN-10)
- **Ch2:** C2-04 and C2-05 (384-dim errors) already fixed in Step 0.1 — verify they are corrected
- **Ch2:** C2-03 threshold distinction note (u_SC > 0.85 general vs > 0.90 VE1) — can be added now

---

## PHASE 7 — Cross-Cutting Checks Before Submission

Run these checks on all chapters as a final pass:

| Check | Applies to | Entry |
|---|---|---|
| No "post-hoc detection" language for CAEM as a whole | All chapters | GEN-01 |
| No projected numbers presented as confirmed results | Ch1, Ch4 | GEN-02 |
| No "384-dim" anywhere | Ch2 (already fixed) | C2-04, C2-05 |
| Hyperparameter provenance stated for every value | Ch4 | C4-10 |
| û field names correct (u_consistency not u_sc) | Ch4, Ch5 | C4-21 |
| Training target stated as reasoning_chain | Ch4 §4.7 | C4-22 |
| Convergence claim scoped correctly | Ch4 §4.9, Ch6 | C4-23 |
| Tier 1 → Stage 5 arrow absent from Ch4 diagram | Ch4 §4.2 | IMPL-01 |
| Calibration timing stated as "after Cycle 0" | Ch4 §4.8, Ch5 §5.1 | IMPL-03 |
| StrategyQA train split justification present | Ch5 §5.3 | IMPL-04 |
| SBERT routing limitation disclosed | Ch6 §6.2 | IMPL-05 |
| Purity theorem includes formal proof block | Ch4 §4.9 | PUB-02 |
| Device upgrade note in §3.x if device ≠ RTX 3060 | Ch3 | GEN-13 |

---

## Writing Entry Data Dependency Summary

### Group A — Write NOW (no experiment data required)

All of Chapter 3, and these Chapter 4 / Chapter 6 entries:
C3-01, C3-02, C4-01, C4-03, C4-03b, C4-04, C4-05, C4-06, C4-07 (present projected weights), C4-08, C4-09, C4-10, C4-10b, C4-15, C4-16, C4-17 (symbolic + illustrative only), C4-18, C4-19, C4-20, C4-21, C4-22, C4-23, IMPL-01 (Ch4 diagram), IMPL-02, IMPL-03, IMPL-04, IMPL-05, IMPL-06, BM-01–06, C2-03, GEN-01, GEN-04, GEN-05, GEN-06, GEN-07, GEN-08, GEN-11, PUB-02, TH-01, TH-02, C6-01, C6-02, C6-related limitation/future work entries

### Group B — Need `calibration/calibrated_config.json`

C5-11 (ECE before/after, T value, actual û weights), C4-07 (actual calibrated weights to fill into Ch5), GEN-03 (confirm projected vs actual weights)

### Group C — Need `experiment_summary.csv`

C5-02 (mechanism table — Tier fractions, MMLU retention, mean û_stored), C5-06 (latency by tier), C5-10 (memory utilisation)

### Group D — Need `eval/{bm}_cycle{n}.json`

C5-05, C5-12, GEN-12 (McNemar's test per bm × baseline pair)

### Group E — Need `purity_validation/theory_validation.json`

C5-03 (three theory validation tables), TH-03, TH-04, C4-17 (replace illustrative values with measured)

### Group F — Need `retroverify_cycle{n}.json`

C5-07 (forgetting scores per cycle, abort threshold check)

### Group G — Need `ablation_results/ablation_summary.json`

C5-09 (ablation table), C5-07 (MMLU retention from ablation), C5-06 (A2 cost comparison)

### Group H — Qualitative (compile from failure cases after reviewing Cycle 3 outputs)

C5-08 (error analysis: 5–10 failure examples per failure mode)

### Group I — Post-experiment Ch1/Ch2 revisit

GEN-10 (check Ch1 stated target vs actual result; update if materially different)

---

## Bug Log Summary (All Resolved — EXP-01 through EXP-13)

*Full details in `caem-implementation-log.md` Sessions 23–29.*

| ID | Problem | Fix |
|---|---|---|
| EXP-01 through EXP-09 | Pipeline stability, FAISS, UTF-8, OOM | See impl-log Sessions 23–26 |
| EXP-10 | Embedding dim 384 (wrong) → 768 (correct) | Fixed in code; C2-04/C2-05 text fix pending |
| EXP-11 | TruthfulQA EM always 0 | ROUGE-L with 0.15 threshold |
| EXP-12 | AB4 (no reverification) missing | `--disable_reverification` added |
| EXP-13 | Token limit 128 cuts CoT | cot_max_new_tokens=256, CoT induction, extract_cot_answer |
| Session 29 | L2 λ=0.4 (wrong) in early notes → λ=0.01 (correct) [DES] | Fixed in hyperparameter-reference.md + writing-suggestions.md |

---

*End of plan. This document is the single authoritative guide for all remaining work.*
