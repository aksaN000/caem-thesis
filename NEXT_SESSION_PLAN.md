# CAEM — Master Execution & Writing Plan
**Updated: 2026-04-11 | Session 34 — Vast.ai RTX 5090 Deployment Ready**

---

## Quick Status

| Step | Status | Notes |
|---|---|---|
| Smoke Test (Cycle 0→3) | ✅ DONE | Full pipeline verified on RTX 3060 |
| Cold-start seeding (Gap 3) | ✅ DONE | Fresh reseeding completed before mini-run (Fix A format) |
| Writing prep files | ✅ DONE | `writing-suggestions.md`, `hyperparameter-reference.md`, `caem-implementation-log.md` all updated |
| Visual elements inventory | ✅ DONE | Full figures/algorithms/equations/theorems spec added to `writing-suggestions.md` |
| Ch1 TikZ fix (IMPL-01a) | ✅ DONE | Tier 1 → Stage 5 arrow removed (2026-04-07) |
| C2-04/C2-05 text fix | ✅ DONE | chapter_2.tex "384-dim" → "768-dimensional" (both §2.1.4 and §2.2.4) |
| Mini-run (n=500 per bm) | ✅ DONE | Completed 2026-04-08. FEVER +10.9%, StrategyQA +9.4%, TruthfulQA net +4.0%, HotpotQA 0% (capability ceiling — see EXP-19). 871 episodes stored. |
| Ablation script audit + fix | ✅ DONE | Session 33 — both scripts audited, 13 bugs fixed. |
| **Train/Transfer Arch Split** | ✅ DONE | Session 34 — FEVER data leakage eliminated. SIL pool uses `train` split; Eval pool uses `dev`/`validation`. `store_to_memory=False` enforced at harness and calibration. |
| **Benchmark Suite Expansion** | ✅ DONE | Session 34 — TriviaQA, NQ-Open, ARC-Challenge loaders added. ARC label extraction added. |
| **10-Cycle Config** | ✅ DONE | Session 34 — `num_cycles=10`, `questions_per_cycle=5000` set in `caem/config.py`. |
| **Codebase Consistency** | ✅ DONE | Session 34 — 368/368 tests pass. PassageStore dim enforced. Calibration memory write patched. |
| **Smoke Test (New Arch)** | ✅ DONE | Session 34 — New orchestrator smoke-tested with all 6 benchmarks. Passes end-to-end. |
| Full experiment (n=5000 × 10 cycles) | ❌ PENDING | **Ready to launch on Vast.ai RTX 5090** — no code changes required |
| Chapter 3 writing | ❌ PLANNED | No data needed — can start now |
| Chapter 5 writing | ❌ PLANNED | Blocked on full experiment data |

---

## Session 34 Changes Summary

The following changes were made this session to finalise the cloud deployment pipeline:

### Architecture Changes
- **`caem/pipeline.py`**: Added `store_to_memory` parameter to `answer()`. Stage 7 (memory storage) is now conditionally bypassed when `store_to_memory=False`.
- **`eval/harness.py`**: `run()` and `run_all()` accept `store_to_memory` and propagate it to `pipeline.answer()`. Always defaults to `False` during eval calls.
- **`scripts/run_experiment.py`**: Full Train/Transfer separation:
  - `load_sil_training_pool()` — loads FEVER/TriviaQA/NQ from `train` split (`store_to_memory=True`)
  - `load_eval_transfer_pool()` — loads FEVER/TriviaQA/NQ/TruthfulQA/StrategyQA/ARC from `dev`/`validation` split (`store_to_memory=False`)
  - Cycle 0 eval: `store_to_memory=False`
  - Step 2.5 (memory generation): `store_to_memory=True` from SIL pool only
  - Step 3 (evaluation): `store_to_memory=False`
- **`scripts/run_calibration.py`**: `collect_calibration_data()` now uses `store_to_memory=False` — calibration fitting no longer inadvertently populates episodic memory.

### New Benchmark Loaders
- **`eval/benchmarks.py`**: Added `load_triviaqa()`, `load_natural_questions()`, `load_arc_challenge()`, and updated `load_benchmark()` router and `make_synthetic_samples()` for all new benchmarks.
- **`eval/metrics.py`**: Added `extract_arc_label()` for multiple-choice label parsing (A/B/C/D).
- **`eval/harness.py`**: `_score()` now handles `arc_challenge` benchmark.

### Configuration
- **`caem/config.py`**: `num_cycles=10`, `questions_per_cycle=5000`.
- **`caem/retrieval/rag.py`**: `PassageStore` now enforces embedding dim match via `CAEMConfig.embedding_dim` — consistent with `EpisodicMemoryStore`.

### Test Fixes
- **`tests/test_episodic_memory.py`**: Mock bad-dimension vectors changed from 768→384 to correctly test the dim-mismatch detection.
- **`tests/test_rag.py`**: `test_wrong_dim_raises` uses 384-dim (wrong), matching the 768-dim config expectation. `test_rag_max_new_tokens_default` updated to 256 (actual config value).

---

## PHASE 0 — Pre-Run Text Fixes ✅ COMPLETED (2026-04-07)

~~C2-04 and C2-05 fixed in `chapter_2.tex` — both "384 dimensional" / "three hundred eighty four dimensional" changed to "768-dimensional".~~

---

## PHASE 1 — Mini-Run Validation ✅ COMPLETED (2026-04-08)

Mini-run (n=500) passed. Results:
- FEVER: +10.9% (Cycle 0 → Cycle 3)
- StrategyQA: +9.4%
- TruthfulQA: +4.0% (net)
- HotpotQA: EM=0.006 all cycles (capability ceiling — model cannot multi-hop, not a CAEM failure)
- 871 episodes stored
- Convergence confirmed on FEVER (Δ = +9.1%, +1.8%, 0.0%)

---

## PHASE 2 — Vast.ai RTX 5090 Deployment ← IMMEDIATE NEXT STEP

The codebase is **fully ready**. No code modifications required on the cloud instance.

### Step 2.1 — Connect and set up Vast.ai instance

Provision an RTX 5090 instance (129GB RAM, 32GB VRAM) on Vast.ai. Use the PyTorch Docker template.

```bash
# On the Vast.ai instance:
git clone <your-repo-url>
cd caem
pip install -r requirements.txt
```

### Step 2.2 — Transfer pre-built data assets

These directories must be present before launching:

```bash
# Transfer from local machine to Vast.ai:
data/passage_index/           # ~1.5 GB — pre-built FAISS Wikipedia passage index
outputs/cold_start_memory/    # small — 447 seeded episodes (cold-start memory)
```

If cold-start memory is not transferred, the experiment still runs — it just starts from zero episodic memory.

### Step 2.3 — Launch the full 10-cycle experiment

```bash
# Inside tmux on Vast.ai:
tmux new -s caem_full

python -m scripts.run_experiment \
  --output_dir outputs \
  --n_questions 5000 \
  --passage_index data/passage_index \
  --cold_start_memory outputs/cold_start_memory/memory_store
```

**Default benchmarks** (no `--benchmarks` flag needed):
- **Train Pool (SIL)**: FEVER, TriviaQA, NQ (`train` split, `store_to_memory=True`)
- **Eval Pool (Transfer)**: FEVER, TriviaQA, NQ, TruthfulQA, StrategyQA, ARC-Challenge (`dev`/`val`, `store_to_memory=False`)

Expected duration: **~14–16 hours** on RTX 5090 for 10 cycles × 5,000 queries.

### Step 2.4 — Monitor for failures

Check every 2–3 hours:
- Log output: no CUDA OOM, no dataset download errors, no KeyError crashes
- `outputs/experiment_summary.csv` — rows accumulating per cycle per benchmark
- `outputs/retroverify_cycle{n}.json` — retroverification counts per cycle

**If OOM**: reduce batch size via `--batch_size` flag (or edit `caem/config.py`). RTX 5090 (32GB VRAM) should handle `batch_size=16` comfortably.

**If dataset download hangs**: pre-download via `python -c "from datasets import load_dataset; load_dataset('lucadiliello/fever', split='train')"` etc.

**If interrupted mid-run**: resume from any completed cycle using:
```bash
python -m scripts.run_experiment \
  --output_dir outputs \
  --n_questions 5000 \
  --resume_from_cycle <N>
```

---

## PHASE 3 — Post-Experiment Analysis Scripts

Run IN ORDER after the full experiment completes.

### Step 3.1 — Purity Validation

```bash
python -m scripts.run_purity_validation \
  --checkpoints_dir outputs \
  --output_dir outputs/purity_validation \
  --num_cycles 10
```

**Produces:** `outputs/purity_validation/theory_validation.json`
**Used by:** C5-03 (§5.4), TH-03, TH-04, C4-17 (replace illustrative values with actual)

### Step 3.2 — Ablation Study

```bash
python -m scripts.run_ablation \
  --caem_results outputs/all_cycle_results.json \
  --cycle3_checkpoint outputs/cycle_10 \
  --cycle0_checkpoint outputs/cycle_0 \
  --output_dir outputs/ablation_results \
  --n_questions 500
```

**Produces:** `outputs/ablation_results/ablation_summary.json`
**Used by:** C5-09 (§5.5), C5-07 MMLU eval, C5-06 A2 cost comparison

### Step 3.3 — Statistical Significance (McNemar's)

```python
from eval.metrics import mcnemar_test, bootstrap_ci
import json

with open("outputs/eval/fever_cycle10.json") as f:
    caem_results = json.load(f)["samples"]

caem_em = [s["em"] for s in caem_results]
# Load baseline EM array
chi2, p_val = mcnemar_test(caem_em, baseline_em)
ci_low, ci_high = bootstrap_ci(caem_em)
```

**Used by:** C5-05, C5-12, GEN-12 (§5.2 statistical significance claims)

### Step 3.4 — Add Strong External Baselines to Table 5.1 (zero compute)

| Model | HotpotQA EM | StrategyQA EM | FEVER | TruthfulQA | Source |
|---|---|---|---|---|---|
| GPT-3.5 (vanilla) | 22.1 | 65.2 | — | — | Liu et al., ACL Findings 2024 |
| GPT-3.5 (standard RAG) | 32.2 | 64.7 | — | — | Liu et al., ACL Findings 2024 |
| Self-RAG 13B | 25.4 | 67.2 | — | — | Liu et al., ACL Findings 2024 |

---

## PHASE 4 — Data Extraction Checklist

After scripts run, confirm these files exist before writing Chapter 5.

| Output file | Chapter 5 section | Writing entries |
|---|---|---|
| `outputs/experiment_summary.csv` | §5.2, §5.3, §5.7, §5.9 | C5-02, C5-06, C5-10 |
| `outputs/eval/{bm}_cycle{n}.json` | §5.2, §5.4 | C5-05, C5-12, GEN-12 |
| `outputs/purity_validation/theory_validation.json` | §5.4 | C5-03, TH-03, TH-04, C4-17 |
| `outputs/calibration/calibrated_config.json` | §5.1 | C5-11, GEN-03, C4-07 |
| `outputs/ablation_results/ablation_summary.json` | §5.5, §5.7, §5.8 | C5-09, C5-07, C5-06 |
| `outputs/retroverify_cycle{n}.json` | §5.8 | C5-07 forgetting scores |

---

## PHASE 5 — Chapter Writing (Sequential Order)

### Step 5.1 — Write Chapter 3 (NOW — no data needed)

**File:** `pre thesis 1 report/chapters/chapter_3.tex`
**Status:** STUB — write from scratch

**§3.1 Functional Requirements** — 8 pipeline stages
**§3.2 Non-Functional Requirements** — NFR table (latency, verification α, MMLU retention, GPU budget)
**§3.3 Hardware Constraints as Design Drivers** — VRAM budget, batch size, theta_prev CPU offload
**§3.4 Societal Impact and Ethical Constraints**

### Step 5.2 — Write Chapter 4 (NOW — no data needed)

**File:** `pre thesis 1 report/chapters/chapter_4.tex`

Write sections in order: §4.1 Architecture → §4.2 Routing → §4.3 Pre-Routing Confidence → §4.4 Post-Generation Confidence → §4.5 Verification → §4.6 Episodic Memory → §4.7 Self-Improvement Loop → §4.8 Calibration → §4.9 Theoretical Analysis

**Critical writing rules:**
- Use `hyperparameter-reference.md` — state Category 1/2/3 provenance for every value
- Field names: `u_token`, `u_dropout`, `u_consistency`, `u_entropy`, `u_hat`
- StoredConfidence fields: `p_entail`, `s_avg`, `h_norm`, `u_stored`
- L2 regularisation (λ=0.01 [DES]) NOT EWC
- Convergence claim: NEVER "infinite cycles → zero hallucination" — always "bounded, monotone improvement"
- New benchmarks section (§4.10 or §4.1 extension): document Train/Transfer split design, why FEVER train ≠ FEVER dev for OOD evaluation

### Step 5.3 — Write Chapter 5 (AFTER Phase 3–4 complete)

All numbers from output files. Never copy from unified plan projections.

| `\section{}` | Source file | Key entries |
|---|---|---|
| §5.1 Experimental Setup | design knowledge | BM-01, C5-01, C5-11 |
| §5.2 Main Results | `eval/{bm}_cycle{n}.json` | C5-04, TAB-C5-01 |
| §5.3 Mechanism Analysis | `experiment_summary.csv` | C5-02, TAB-C5-02 |
| §5.4 Ablation Study | `ablation_results/ablation_summary.json` | C5-09, TAB-C5-05 |
| §5.5 Theory Validation | `purity_validation/theory_validation.json` | C5-03, TH-03, TH-04 |
| §5.6 Latency and Error Analysis | `experiment_summary.csv` mean_latency_ms | C5-06, C5-08 |

---

## PHASE 6 — Cross-Cutting Checks Before Submission

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
| Train/Transfer split documented — FEVER train ≠ FEVER dev | Ch5 §5.1 | Session 34 |
| store_to_memory=False enforced during all eval passes | Addressed in code | Session 34 |
| Purity theorem includes formal proof block | Ch4 §4.9 | PUB-02 |

---

## Bug & Fix Log (All Resolved)

| ID | Problem | Fix |
|---|---|---|
| EXP-01–09 | Pipeline stability, FAISS, UTF-8, OOM | See impl-log Sessions 23–26 |
| EXP-10 | Embedding dim 384 (wrong) → 768 | Fixed in code |
| EXP-11 | TruthfulQA EM always 0 | ROUGE-L with 0.15 threshold |
| EXP-12 | AB4 (no reverification) missing | `--disable_reverification` added |
| EXP-13 | Token limit 128 cuts CoT | cot_max_new_tokens=256, extract_cot_answer |
| EXP-14 | Forgetting check used absolute accuracy → always aborted | Relative retention ratio (post/pre ≥ 0.93) |
| EXP-15 | NaN training loss | float32 L2, bf16/AMP, clip_grad_norm=1.0 |
| EXP-16 | reasoning_chain stored bare labels | Fix A: task-aware CoT prompts for FEVER + StrategyQA |
| EXP-19 | Mini-run results | PASS — see Step 1 above |
| EXP-20 | Ablation + purity script bugs (13 bugs) | Fixed in Session 33 |
| **EXP-21** | **FEVER data leakage (eval data stored to memory)** | **Fixed Session 34: Train/Transfer split, `store_to_memory=False` at harness + calibration** |
| **EXP-22** | **Calibration silently storing 500 train-split questions to memory** | **Fixed Session 34: `run_calibration.py` line 273 — added `store_to_memory=False`** |
| **EXP-23** | **PassageStore accepted any embedding dimension silently** | **Fixed Session 34: PassageStore now enforces `CAEMConfig.embedding_dim=768`** |
| **EXP-24** | **Test suite: 4 tests hardcoded old dim/config values** | **Fixed Session 34: test_episodic_memory.py + test_rag.py updated** |

---

*End of plan. This document is the single authoritative guide for all remaining work. Last updated: Session 34, 2026-04-11.*
