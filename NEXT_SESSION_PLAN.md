# CAEM — Next Session Plan
**Updated: 2026-04-01 | Session 24 — Config finalised, hardware strategy locked**

---

## Current Status

| Step | Status | Notes |
|---|---|---|
| Gap 1 — base model check | ✅ DONE | `outputs/base_model_check.json` |
| Gap 2 — passage index (500K) | ⏳ ALMOST DONE | `data/passage_index/` — verify before continuing |
| Gap 3 — cold-start seeding | ❌ NOT RUN | Run on RTX 3060 after Gap 2 confirmed |
| Full experiment (Cycle 0→3) | ❌ NOT RUN | Needs 4090 or better — see hardware plan |
| Purity validation | ❌ NOT RUN | — |
| Ablations | ❌ NOT RUN | — |
| Calibration | ❌ NOT RUN | — |

---

## ⚠️ First — verify Gap 2 completed correctly

```powershell
# On Windows PowerShell:
dir "C:\Users\aksan\OneDrive\Documents\for cowork caem\data\passage_index\"
```

Expected: `passages.faiss` = **1.0–1.5 GB**, `passages.pkl` = **50–100 MB**
If `passages.faiss` is still ~3 MB (nearly empty), the checkpoint saved but the final flush didn't complete — re-run:

```bash
python scripts/build_passage_index.py --max_passages 500000 --output_dir data/passage_index
```

---

## What n_questions=5000 actually means

`n_questions` is **per benchmark**. With 4 benchmarks:

| Allocation | Per benchmark | Total (4 bm) |
|---|---|---|
| Purity validation set | 500 | 2,000 |
| Calibration set | 500 | 2,000 |
| **Eval (active pipeline)** | **4,000** | **16,000** |

Each eval question goes through the 8-stage CAEM pipeline **per cycle**. With 3 cycles:
**48,000 inference passes + 3 fine-tuning epochs.** This cannot run in any reasonable time on RTX 3060 or Kaggle T4.

---

## Hardware Strategy — Tiered Plan

### Tier 1: Do right now (free, RTX 3060)

```bash
cd "C:\Users\aksan\OneDrive\Documents\for cowork caem"

# Gap 3 — cold-start seeding (~1–2 hours, 200 episodes × 3 benchmarks)
python scripts/seed_cold_start.py \
  --target_episodes 200 \
  --output_dir outputs/cold_start_memory
```

This is all that makes sense on RTX 3060. Do not attempt the full experiment locally.

### Tier 2: Use 2K BDT budget — Vast.ai RTX 4090 (NOW if you want results sooner)

RTX 4090 = ~3.5× faster than RTX 3060. Full experiment runs in **~11–14 hours** in one sitting.

**Cost breakdown (Vast.ai RTX 4090 at ~$0.30–0.40/h = ~33–44 BDT/h):**

| Phase | Time on 4090 | Cost (BDT) |
|---|---|---|
| Full experiment (Cycle 0→3, n=5000) | ~11–14h | ~360–615 |
| Purity validation | ~30 min | ~20–30 |
| Ablations (9 configs, n=500 per config) | ~3–5h | ~100–220 |
| Calibration | ~45 min | ~25–35 |
| **Total** | **~16–20h** | **~500–860 BDT** |

Well within your 2K BDT budget. You'd finish the **entire experiment phase** in one Vast.ai session.

**How to use Vast.ai:**
1. Go to vast.ai → Create Account
2. Search: RTX 4090, 24GB VRAM, `PyTorch` image
3. Pick cheapest offer (filter by price, confirm ~$0.30–0.40/hr)
4. SSH in, clone your repo from GitHub, copy `data/passage_index/` and `outputs/cold_start_memory/` via `scp` or upload to HuggingFace Hub first
5. Run the experiment — download outputs when done

### Tier 3: University GPU next month (FREE — best option)

| GPU | Speed vs 3060 | Full experiment time | Cost |
|---|---|---|---|
| RTX 4090 (24GB) | ~3.5× | ~11–14h | Free |
| RTX 5090 (32GB) | ~5–6× | ~7–10h | Free |

**Recommendation: Wait for uni GPU.** You're 1 month away from free, fast compute. Use that for the definitive thesis run. The 2K BDT cloud budget is insurance — use it only if the timeline is tight.

---

## Thesis-final config (locked)

```python
# caem/config.py — these values are set, do not change for thesis run
n_questions        = 5000    # per benchmark (§5.3 standard; 4000 eval + 500 purity + 500 calib)
max_passages       = 500_000 # Wikipedia passages — defensible for 780M param model
target_episodes    = 200     # cold-start per benchmark (TruthfulQA skipped — no train split)
sc_chains_m        = 3       # [LIT] Wang et al. 2022 minimum viable consensus
se_samples_k       = 10      # [LIT] Farquhar et al. 2024 standard
mc_dropout_k       = 5       # [DES]
batch_size         = 4       # RTX 3060/T4 safe; set to 16 on 4090/5090
cot_max_new_tokens = 256     # [EXP-13 fix — do not lower]
n_cycles           = 3       # thesis plan §5.2
embedding_dim      = 768     # [EXP-10 fix — all-mpnet-base-v2 is 768-dim (not all-MiniLM which is 384)]
```

**When on 4090/5090 — change only these two lines in `caem/config.py`:**
```python
batch_size = 16     # safe on 24GB+ VRAM, speeds up fine-tuning ~4×
```
Everything else stays identical.

---

## Full run sequence (copy-paste for 4090/5090)

> [!WARNING]
> Because this script runs for 11–14 hours, your SSH connection dropping will kill the script. **ALWAYS run this inside `tmux`:**
> `tmux new -s caem`

```bash
cd /path/to/caem-project

# Confirm the index path is correct — adjust if needed
ls data/passage_index/

# ── Full experiment — Cycle 0 baseline + Cycles 1–3 (~11–14h on 4090) ─────
python scripts/run_experiment.py \
  --n_questions 5000 \
  --benchmarks hotpotqa truthfulqa fever strategyqa \
  --output_dir outputs/full_experiment \
  --passage_index data/passage_index \
  --cold_start_memory outputs/cold_start_memory

# ── CRASH RECOVERY (If the script dies midway) ─────────────────────────────
# Do not run this normally! Only if it OOMs or the server reboots during Cycle X.
# Just swap 'X' for the cycle it crashed on:
python scripts/run_experiment.py \
  --n_questions 5000 \
  --resume_from_cycle X \
  --output_dir outputs/full_experiment \
  --passage_index data/passage_index
  
# ── Purity validation — validates Theorem 1/2/3 (~30 min) ─────────────────
python scripts/run_purity_validation.py \
  --experiment_dir outputs/full_experiment \
  --output_dir outputs/purity_validation

# ── Ablations — 9 configs × 500 questions (~3–5h on 4090) ────────────────
python scripts/run_ablation.py \
  --n_questions 500 \
  --cycle3_ckpt outputs/full_experiment \
  --output_dir outputs/ablations

# ── Calibration (run if routing confidence seems miscalibrated) ───────────
python scripts/run_calibration.py \
  --experiment_dir outputs/full_experiment \
  --output_dir outputs/calibration
```

**After run — bring these to the analysis session:**
- `outputs/full_experiment/experiment_summary.csv`
- `outputs/full_experiment/all_cycle_results.json`
- `outputs/purity_validation/theory_validation.json`
- `outputs/ablations/ablation_summary.json`

---

## Full cost estimate — 2K BDT budget allocation

| Scenario | Total cost | When |
|---|---|---|
| Wait for uni GPU (free) | **0 BDT** | ~1 month |
| Vast.ai RTX 4090 now | **~500–860 BDT** | Immediately |
| Vast.ai + buffer for re-runs | **~1,200–1,500 BDT** | Leaves ~500 BDT margin |
| Kaggle T4 (free, slower, multi-session) | **0 BDT** | Multi-week, risky checkpointing |

**Verdict:** Either wait for uni GPU (best) or spend ~800–1000 BDT on Vast.ai to finish in one sitting now. Do NOT attempt n=5000 on RTX 3060 or T4 — it will take 32–50 hours, fragmented across too many sessions.

---

## Lab PC config — AFTER thesis submission (bonus run)

See `LAB_PC_SCALING_GUIDE.md`. Key upgrades: 5M passages, 1000 cold-start episodes, sc_chains=10, batch_size=32, theta_prev natively on GPU (PCIe bottleneck removed → ~4× faster fine-tuning). Run only if you want to show maximum-capacity numbers in a journal extension.

---

## Open issues (non-blocking, note in thesis)

1. **TruthfulQA no training split** → cold-start skipped — note in Chapter 5 §5.3
2. **ReST comparison** — write in Chapter 4 (writing-suggestion C4-16)
3. **passage_index path** is `data/passage_index/`, NOT `outputs/passage_index/`

---

## All bugs fixed (EXP-01 through EXP-13)

| ID | File | Problem | Fix |
|---|---|---|---|
| EXP-01 | `eval/benchmarks.py` | StrategyQA test split (490) → 0 eval samples | train split (~2,290) |
| EXP-02 | Notebook Cell 11 | Hardcoded 500+500 → TruthfulQA 0 eval | split_calibration_sets() |
| EXP-03 | `scripts/run_experiment.py` | QueryEncoder(config=config) wrong kwarg | model_name=config.sbert_model |
| EXP-04 | `scripts/seed_cold_start.py` | Wrong sub-component constructors | mirrors run_experiment.py |
| EXP-05 | `scripts/seed_cold_start.py` | result.get("stored") on dataclass | result.stored |
| EXP-06 | `scripts/seed_cold_start.py` | memory_store.size() calling a property | .size (no parentheses) |
| EXP-07 | `eval/benchmarks.py` | fever/v1.0 HF legacy script removed | lucadiliello/fever mirror |
| EXP-08 | `eval/benchmarks.py` | wics/strategy-qa legacy script removed | direct JSON fetch from GitHub |
| EXP-09 | `scripts/build_passage_index.py` | wikipedia/20220301.en legacy script | wikimedia/wikipedia 20231101.en |
| EXP-10 | `caem/config.py`, `caem/retrieval/rag.py` | embedding_dim=384 (mpnet is 768) | 768 + dynamic shape detection |
| EXP-11 | `eval/metrics.py`, `eval/harness.py` | TruthfulQA EM always 0 | rouge_l with 0.15 threshold |
| EXP-12 | `scripts/run_ablation.py` | AB4 (no reverification) missing | --disable_reverification added |
| EXP-13 | Multiple files | Token limit 128 cuts CoT; no CoT prompting | cot_max_new_tokens=256, CoT induction, extract_cot_answer |
