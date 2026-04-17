# CAEM - Next Session Plan (Vast.ai Runbook)
**Updated: 2026-04-17 | Session 44 — Phase 1 (Self-Funded) Execution Plan**

This file is your **step-by-step runbook** for executing Phase 1 of the
CAEM experiments on a Vast.ai-rented GPU. Every step is ordered. Do
not skip steps. Where a command is shown, run it exactly as written
(substituting your own `PORT`, `IP`, and path arguments where marked
`<LIKE_THIS>`).

For background on the two-phase budget protocol, see `VAST_AI_DEPLOYMENT_GUIDE.md`
Part 10. For methodological rationale, see `writing-suggestions.md`
Session 44 Addendum and Chapter 3 §3.5 (Ablation Framework Plan).

---

## Where you are right now

- All code is committed. Scripts in scope for this runbook:
  `scripts/run_experiment.py`, `scripts/run_baseline.py` (B1–B5),
  `scripts/run_simple_ft.py` (B6, B7), `scripts/run_cyclic_ablation.py`,
  `scripts/aggregate_ablation.py`, `scripts/run_purity_validation.py`,
  `scripts/build_passage_index.py`, `scripts/seed_cold_start.py`.
- Ablation framework = **16 named variants** in
  `caem/ablation/variants.py`. Legacy AB1–AB7 labels are retired.
- External baseline panel = **B1–B7** (inference B1–B5 via
  `run_baseline.py`, training B6–B7 via `run_simple_ft.py`). B8
  Self-RAG is citation-only. STaR is an optional ceiling reference.
- Budget: ~USD 200 self-funded envelope for Phase 1 (single seed = 42).
  Phase 2 (+USD 700, seeds 123 and 456) is re-running the same
  confirmatory commands — no code changes.
- Target GPU for Phase 1: **RTX 4090** on Vast.ai (~$0.40/hr).
  Upgrade to A100 SXM 80 GB only if wall-clock matters more than cost.

---

## First run → last run (execution order, at a glance)

Run every step below in this exact order. Do not skip, do not re-order.
Rough wall-clock totals assume a single RTX 4090; see Part 10 of the
deployment guide for cost breakdowns. Phase 1 budget absorbs all of
this in one rental cycle.

| # | Step | Script | Wall-clock | Notes |
|---|---|---|---|---|
| 1 | Pre-flight (local PC) | — | 5 min | Push commits, make HF token, top up Vast credit |
| 2 | Rent + connect RTX 4090 | — | 10 min | Vast website |
| 3 | Environment setup | `pip install ...` | 15 min | Then cache HF models |
| 4 | Passage index | `scripts/build_passage_index.py` | 2–3 h | Build on the instance |
| 5 | Smoke test | `scripts/run_cyclic_ablation.py --smoke_test` | 20 min | Gate before main run |
| 6 | Cold-start seeding | `scripts/seed_cold_start.py` | 15 min | Produces `memory_store.*` |
| 7 | **Main 10-cycle CAEM run** | `scripts/run_experiment.py` | 16–18 h | The headline result |
| 8 | **B1 zero-shot** | `scripts/run_baseline.py --baseline zero_shot` | ~15 min | Floor |
| 9 | **B2 CoT** | `scripts/run_baseline.py --baseline cot` | ~20 min | |
| 10 | **B3 DPR-RAG** | `scripts/run_baseline.py --baseline rag` | ~45 min | Needs passage index |
| 11 | **B4 CoT + RAG** | `scripts/run_baseline.py --baseline cot_rag` | ~50 min | |
| 12 | **B5 FLARE** | `scripts/run_baseline.py --baseline flare` | ~90 min | Active retrieval |
| 13 | **B6 Vanilla FT (10 cycles)** | `scripts/run_simple_ft.py --baseline_name vanilla_ft` | ~6 h | No anchor, no guard |
| 14 | **B7 EWC-only FT (10 cycles)** | `scripts/run_simple_ft.py --baseline_name ewc_only_ft --use_l2_anchor --use_mmlu_guard` | ~6 h | Anchor + guard only |
| 15 | Screening sweep (16 variants × 3 cycles) | `scripts/run_cyclic_ablation.py --screening_mode` | 14–20 h | Ranks variants |
| 16 | Aggregate screening + pick top-N | `scripts/aggregate_ablation.py --screening_mode` | 5 min | Record top-N in impl log |
| 17 | Confirmatory sweep (top-N + `full`, 10 cycles) | `scripts/run_cyclic_ablation.py` | 50–60 h | Ch5 ablation table |
| 18 | Purity theorem validation | `scripts/run_purity_validation.py` | 30 min | |
| 19 | Aggregate all Phase 1 outputs | `scripts/aggregate_ablation.py` | 5 min | |
| 19B | **(Optional)** STaR ceiling run | `scripts/run_simple_ft.py --use_rationalisation` | ~7–8 h | Only if Phase 1 is under USD 200 envelope |
| 20 | Download + stop instance | `scp -r` then Stop on Vast website | 20 min | |

Total Phase 1 on RTX 4090: ~115–130 h wall-clock (~USD 50–55 rental
cost at $0.40/hr). Baseline sub-total (steps 8–14): ~16 h. Optional
STaR (step 19B): adds ~7–8 h (~$3) only if budget permits. The Vast
instance can be stopped and resumed between groups as long as no
step is interrupted mid-cycle.

---

## Pre-flight (do this on your local PC, before renting)

1. **Push any local commits to GitHub.** The remote Vast instance will
   `git clone` from GitHub — not from your laptop.
   ```bash
   git status
   git push origin main
   ```

2. **Make sure your passage index either (a) is on Google Drive / a
   CDN so the remote instance can `gdown` it, or (b) is built fresh on
   the instance with `build_passage_index.py --max_passages 21000000`.**
   Uploading 15–17 GB from your laptop over `scp` on a residential
   connection is slow; prefer option (b).

3. **Create a GitHub personal access token** for HTTPS `git clone` from
   the remote instance. (Settings → Developer settings → Personal
   access tokens → Generate new token → `repo` scope.)

4. **Verify your Vast.ai account has ≥ USD 75 credit.** This covers
   setup + main run + screening sweep. You can top up mid-Phase-1
   without stopping the instance.

---

## Step 1 — Rent the RTX 4090 instance

On [https://vast.ai](https://vast.ai):

1. Top-left → **Edit Image & Config** → choose a PyTorch template
   (`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` or similar).
2. In the config popup → **Allocated Disk Space slider = 100 GB**
   (default 16 GB will crash on passage index).
3. **Save**.
4. Left filter panel → **Verified**, **1× GPU**, GPU type = **RTX 4090**.
5. Sort by reliability ≥ 99% and lowest price.
6. Click **RENT**.

Wait until instance shows **Running**. Click **Connect** → copy the
`ssh -p <PORT> root@<IP>` command. You will reuse `<PORT>` and `<IP>`
throughout.

---

## Step 2 — Connect and set up the environment

Run these on the **remote instance terminal** (Remote-SSH or plain SSH):

```bash
# Clone the repo (replace TOKEN and REPO)
git clone https://<TOKEN>@github.com/<USER>/<REPO>.git caem
cd ~/caem

# Install dependencies (faiss-CPU, not GPU)
pip install torch transformers datasets sentence-transformers \
    faiss-cpu numpy scipy scikit-learn

# Verify GPU + auto-detected hardware profile
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "from scripts.hardware import print_hardware_summary; print_hardware_summary()"
```

Expected on RTX 4090:
```
GPU: NVIDIA GeForce RTX 4090
VRAM: 23.6 GB
Precision: fp16
Batch size: 16
```

Pre-cache HuggingFace models (prevents mid-run download stalls):

```bash
export HF_HOME=/root/caem/hf_cache
python - <<'PY'
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained('google/flan-t5-large')
T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
AutoTokenizer.from_pretrained('roberta-large-mnli')
AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print('All models cached.')
PY
```

---

## Step 3 — Build (or upload) the passage index

**Option A — build on the instance (recommended):**

```bash
tmux new-session -s build
python scripts/build_passage_index.py \
    --max_passages 21000000 \
    --output_dir data/passage_index
```
Detach with `Ctrl+B, D`. Reattach with `tmux attach -t build`. Expect
~2–3 h on RTX 4090. When done, you should see `passages.faiss` and
`passages.pkl` in `data/passage_index/`.

**Option B — download from Google Drive:**

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/<FOLDER_ID>" -O data/
```

---

## Step 4 — Smoke test (1 cycle, n=50 per benchmark)

Before burning ~USD 10 on the main run, verify the full pipeline runs
end-to-end. This takes ~20 min.

```bash
tmux new-session -s smoke
python scripts/run_cyclic_ablation.py \
    --variant full \
    --seed 42 \
    --smoke_test \
    --passage_index data/passage_index \
    --output_dir outputs/smoke
```

Pass criterion: `outputs/smoke/full/seed_42/ces_axes_per_cycle.json`
exists with one cycle record and `ces > 0`. If the smoke test fails,
STOP and debug before proceeding — wall-clock savings compound across
all later steps.

---

## Step 5 — Cold-start seeding

```bash
python -m scripts.seed_cold_start \
    --target_episodes 200 \
    --benchmarks fever triviaqa natural_questions \
    --output_dir outputs/cold_start_memory
```

Expected artifacts: `memory_store.faiss`, `memory_store.meta`,
`seed_summary.json` in `outputs/cold_start_memory/`.

---

## Step 6 — Main 10-cycle run (Phase 1, seed 42)

This is the headline result — ~16–18 h on RTX 4090, ~$6.50–9.50.

```bash
tmux new-session -s main
python -m scripts.run_experiment \
    --output_dir outputs/full_run \
    --num_cycles 10 \
    --n_questions 5000 \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --passage_index data/passage_index \
    --cold_start_memory outputs/cold_start_memory/memory_store \
    2>&1 | tee outputs/full_run/run.log

# NOTE: run_experiment.py currently uses seed=42 internally (hardcoded in
# self_improvement.run_cycle default). For Phase 2 multi-seed main runs,
# add a --seed CLI flag to run_experiment.py first — see "Must-do before
# Phase 2" in the status check section.
```

Detach (`Ctrl+B, D`). Watch progress from a second terminal with
`tail -f outputs/full_run/run.log`. When complete, verify:
- `outputs/full_run/experiment_summary.csv` exists with 11 rows (cycles 0–10).
- `outputs/full_run/retroverify_cycle*.json` files exist.
- `mmlu_retention_pct` column in the summary CSV is ≥ 93%.

**If the run crashes:** check `ls outputs/full_run/ | grep ^cycle_`
for the last completed cycle, then resume with
`--resume_from_cycle <N>`.

---

## Step 6B — External baseline runs (B1–B7)

Run after the main 10-cycle CAEM run in Step 6 and **before** the
screening sweep in Step 7. These populate the Chapter 5 headline
comparison table. Baselines are independent of each other, so if a
later one crashes you can resume from exactly that baseline without
re-running the earlier ones. Outputs all land under
`outputs/baselines/<name>/` in the same schema as the main run so
`eval/reporting.py` consumes them unchanged.

**Pre-flight: FLARE smoke test (mandatory before B5 — verifies the
decoder-slicing fix).** The B5 FLARE implementation at
`eval/baselines.py` was patched after an audit found that the
look-ahead was slicing `out.sequences[0, input_ids.shape[1]:]` which
yields an empty tensor for T5 encoder–decoder (correct offset is 1).
Run this first on 5 FEVER samples and confirm the output JSON has
non-empty `answer` strings and at least one sample with
`"escalated": true`. If answers are empty or escalated is always
false, the fix did not land — do not run the full B5 panel until
this passes.

```bash
python -m scripts.run_baseline \
    --baseline flare \
    --benchmarks fever \
    --n_questions 5 \
    --passage_index data/passage_index \
    --flare_theta 0.4 \
    --flare_look_ahead 64 \
    --output_dir outputs/baselines_smoke \
    2>&1 | tee outputs/baselines_smoke/B5_flare_smoke.log

python - <<'PY'
import json, glob
for f in sorted(glob.glob("outputs/baselines_smoke/flare/*.json")):
    d = json.load(open(f))
    samples = d.get("samples", [])
    empty = sum(1 for s in samples if not s.get("answer", "").strip())
    escalated = sum(1 for s in samples if s.get("escalated"))
    print(f, "n=", len(samples), "empty_answer=", empty, "escalated=", escalated)
PY
```

Expected: `empty_answer=0` and `escalated >= 1`. Anything else means
the FLARE fix regressed; inspect `eval/baselines.py` `_look_ahead`
method.

**First run → last run sequence:**

```bash
# === INFERENCE BASELINES (single script: scripts/run_baseline.py) ===

# (8) B1 Zero-shot Flan-T5-Large — the floor. ~15 min.
tmux new-session -s b1
python -m scripts.run_baseline \
    --baseline zero_shot \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B1_zero_shot.log

# (9) B2 Chain-of-Thought. ~20 min.
python -m scripts.run_baseline \
    --baseline cot \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B2_cot.log

# (10) B3 DPR-RAG (k=5, 384-tok context). ~45 min.
python -m scripts.run_baseline \
    --baseline rag \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --passage_index data/passage_index \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B3_rag.log

# (11) B4 CoT + DPR-RAG. ~50 min.
python -m scripts.run_baseline \
    --baseline cot_rag \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --passage_index data/passage_index \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B4_cot_rag.log

# (12) B5 FLARE (theta=0.4, look-ahead=64). ~90 min.
python -m scripts.run_baseline \
    --baseline flare \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --passage_index data/passage_index \
    --flare_theta 0.4 \
    --flare_look_ahead 64 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B5_flare.log

# === TRAINING BASELINES (single script: scripts/run_simple_ft.py) ===
# Each is 10 cycles — same cycle count as the main CAEM run.

# (13) B6 Vanilla FT — no L2 anchor, no MMLU guard. ~6 h.
tmux new-session -s b6
python -m scripts.run_simple_ft \
    --baseline_name vanilla_ft \
    --num_cycles 10 \
    --passage_index data/passage_index \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --output_dir outputs/baselines/vanilla_ft \
    2>&1 | tee outputs/baselines/B6_vanilla_ft.log

# (14) B7 EWC-only FT — L2 anchor + MMLU guard on, everything else off. ~6 h.
tmux new-session -s b7
python -m scripts.run_simple_ft \
    --baseline_name ewc_only_ft \
    --use_l2_anchor \
    --use_mmlu_guard \
    --num_cycles 10 \
    --passage_index data/passage_index \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --output_dir outputs/baselines/ewc_only_ft \
    2>&1 | tee outputs/baselines/B7_ewc_only_ft.log
```

**Total wall-clock for Step 6B:** ~16 h sequential on RTX 4090
(~$6.50 rental). Inference baselines B1–B5 take ~3.5 h combined;
training baselines B6–B7 take ~12 h combined.

**Sanity check after each baseline:**

```bash
ls outputs/baselines/*/
python - <<'PY'
import json, glob
for f in sorted(glob.glob("outputs/baselines/*/*.json")):
    try:
        d = json.load(open(f))
        agg = d.get("aggregate", {})
        print(f, "EM=", agg.get("em"), "F1=", agg.get("f1"), "N=", agg.get("n"))
    except Exception as e:
        print(f, "FAILED:", e)
PY
```

Expected: B1 zero-shot EM near published Flan-T5-Large numbers (FEVER
~55–60%, TriviaQA ~40–45%). Dramatic deviation means a loader or
prompt bug; debug before proceeding to Step 7.

---

## Step 7 — Phase 1 screening sweep (16 variants × 3 cycles × n=1500)

Screening ranks the 14 training-time variants + 2 inference-time
variants so that confirmatory budget goes to the high-impact ones
only. **Screening rows never enter the Chapter 5 ablation table —
they are a budget-allocation instrument.**

Create a wrapper script:

```bash
cat > run_screening.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
VARIANTS=(
    full no_self_improvement no_tier1 no_retrieval no_verification
    no_nli no_selfcons no_entropy no_routing no_cold_start
    no_abort_guard no_l2_anchor no_retroverify no_memory_prune
    no_mcdropout lower_u_threshold
)
for v in "${VARIANTS[@]}"; do
    echo "=== SCREENING $v ==="
    python scripts/run_cyclic_ablation.py \
        --variant "$v" \
        --seed 42 \
        --screening_mode \
        --passage_index data/passage_index \
        --cold_start_memory outputs/cold_start_memory/memory_store \
        --output_dir outputs/ablation \
        2>&1 | tee outputs/ablation/"$v"_screening.log
done
BASH
chmod +x run_screening.sh

tmux new-session -s screening
./run_screening.sh
```

Expected wall-clock: ~14–20 h total on RTX 4090 (batched sequentially
— about 45–75 min per variant × 16 variants).

When done, aggregate and inspect the screening ranking:

```bash
python scripts/aggregate_ablation.py \
    --output_dir outputs/ablation \
    --screening_mode
cat outputs/ablation/ablation_table.csv
```

The variants are sorted by CES (lowest first = most damaged).
Identify the **top N variants by $|\Delta\text{CES}|$ vs. `full`**
(typically N = 5–7). These are your confirmatory candidates. Record
the selected variants in `caem-implementation-log.md` before
proceeding — this is the pre-registration step for Chapter 5.

---

## Step 8 — Phase 1 confirmatory sweep (top-N + reference, 10 cycles × n=5000)

Only run the variants selected in Step 7 plus the `full` reference.
Example (replace the variant list with your screening-selected
candidates):

```bash
cat > run_confirmatory.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
# EDIT THIS LIST from your screening-mode top-N selection
VARIANTS=(
    full
    no_self_improvement
    no_verification
    no_retroverify
    no_l2_anchor
    no_cold_start
)
for v in "${VARIANTS[@]}"; do
    echo "=== CONFIRMATORY $v ==="
    python scripts/run_cyclic_ablation.py \
        --variant "$v" \
        --seed 42 \
        --passage_index data/passage_index \
        --cold_start_memory outputs/cold_start_memory/memory_store \
        --output_dir outputs/ablation \
        2>&1 | tee outputs/ablation/"$v"_confirmatory.log
done
BASH
chmod +x run_confirmatory.sh

tmux new-session -s confirmatory
./run_confirmatory.sh
```

Expected wall-clock: ~50–60 h total on RTX 4090 (~8–10 h per variant
× 6 variants). This is the longest step — you may want to stop and
resume the instance across multiple days; just remember to **stop** it
in the Vast website between sessions (Vast charges idle time).

---

## Step 9 — Purity theorem validation

```bash
python scripts/run_purity_validation.py \
    --output_dir outputs/purity_validation
```

Expected: `outputs/purity_validation/theory_validation.json` with
observed vs. predicted purity across all 6 benchmarks. This populates
Chapter 5 §5.3.

---

## Step 10 — Aggregate Phase 1 outputs

```bash
python scripts/aggregate_ablation.py \
    --output_dir outputs/ablation
```

Phase 1 aggregator outputs (single-seed point estimates):
- `outputs/ablation/ablation_table.csv` — headline ablation table.
- `outputs/ablation/ablation_per_cycle.csv` — per-cycle CES trace.
- `outputs/ablation/ablation_aggregate_manifest.json` — run provenance.

---

## Step 10B — Optional STaR ceiling run (only if Phase 1 is under envelope)

> **Gate:** run this step **only** if everything above (Steps 1–10)
> has completed and your Vast.ai spend is still comfortably inside the
> USD 200 Phase 1 envelope, with at least USD 5 headroom. STaR is a
> pure iterative-refinement scheme (Zelikman et al. 2022, NeurIPS
> \cite{NEURIPS2022_639a9a17}) cited in Ch2 §2.2 and Ch5 §5.1 as an
> aspirational ceiling; having a numerical row for it lets you claim
> "\caem{}'s gap to STaR isolates what the verifier + memory add on
> top of pure iterative refinement." If the budget is tight, skip this
> step — STaR remains a citation-only ceiling and the thesis story is
> still defensible.

STaR reuses `scripts/run_simple_ft.py` via three extra flags:
`--use_rationalisation` turns on the per-cycle forward-filter +
back-rationalise pass; the remaining flags match B6 Vanilla FT
(no $L_2$ anchor, no MMLU guard — the point of STaR is the
rationalisation delta over B6, not the retention protection).

```bash
# ~7–8 h on RTX 4090 (10 cycles × rationalisation generation + FT).
# ~30 min of the cycle time is rationalisation generation;
# the rest is the standard 3-epoch fine-tune.
tmux new-session -s star
python -m scripts.run_simple_ft \
    --baseline_name star \
    --use_rationalisation \
    --num_cycles 10 \
    --passage_index data/passage_index \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --output_dir outputs/baselines/star \
    2>&1 | tee outputs/baselines/STaR_ceiling.log
```

Sanity check after completion: inspect
`outputs/baselines/star/training_log.jsonl` and confirm every cycle
records `"use_rationalisation": true`. The per-cycle EM should be
monotonically non-decreasing on the in-domain benchmarks (FEVER,
TriviaQA, NQ); if it is not, the rationalisation pass is producing
low-quality rationales and the fine-tune is degrading the model.

If STaR beats B6 Vanilla FT and B7 EWC-only FT on the Chapter 5
headline table, \caem{}'s gap to STaR becomes the cleanest
ceiling-adjusted measurement of the verifier + memory contribution.
If STaR underperforms B7 (likely, because Flan-T5-Large's
rationalisation quality on factual QA is weaker than on the
commonsense benchmarks STaR was originally validated on), record
that finding in the impl log and cite the benchmark-mismatch
caveat in Ch2 §2.2.

---

## Step 11 — Download everything and stop the instance

From your **local PC terminal** (not the remote):

```bash
mkdir -p "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast"
scp -r -P <PORT> root@<IP>:~/caem/outputs/ \
    "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast\"
```

Then in the Vast website: **Stop** the instance. Idle instances bill
at full rate.

---

## After Phase 1 (writeup)

1. Populate Chapter 5 §5.5 ablation table from
   `outputs_from_vast/outputs/ablation/ablation_table.csv`.
2. Add the single-seed limitation paragraph (wording in
   `writing-suggestions.md` S44-03) under §5.5.
3. Populate Chapter 5 §5.1 headline tables from
   `outputs_from_vast/outputs/full_run/experiment_summary.csv`.
4. Populate Chapter 5 §5.1 external-baseline comparison row from
   `outputs_from_vast/outputs/baselines/*/` (one row per B1–B7).
5. Populate Chapter 5 §5.3 purity tables from
   `outputs_from_vast/outputs/purity_validation/theory_validation.json`.

---

## Phase 2 — post-funding upgrade (plan only; do not execute yet)

Once funding is secured, Phase 2 re-runs the confirmatory sweep on
two additional seeds (123, 456) across **all 14 cyclic variants +
reference**. Expected compute: ~300–360 h on RTX 4090
(~$130–180), or ~180–220 h on A100 SXM 80 GB (~$400–620). Budget
envelope: ~USD 700.

Phase 2 execution is the same `scripts/run_cyclic_ablation.py`
invocation pattern as Step 8, just with `--seed 123` and `--seed 456`
in place of `--seed 42`, and the full 14-variant cyclic list instead
of the Phase 1 top-N. The aggregator auto-switches from point estimate
to mean±std reporting once the second seed's outputs exist in
`outputs/ablation/<variant>/seed_<N>/`. Chapter 5 §5.5 tables are
re-generated by re-running `aggregate_ablation.py` with no CLI
changes.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| CUDA OOM during fine-tuning | Batch size too high for detected GPU | Override `batch_size` in `caem/config.py`; re-run |
| `FileNotFoundError` on passage index | Didn't build or upload | Return to Step 3 |
| Screening rows missing for some variants | One variant crashed mid-loop | `--resume_from_cycle 0` on that variant, re-aggregate |
| `ces = 0` in aggregator output | Axis collapse (CAL or VER = 0) — expected for extreme variants like `no_verification` | Inspect per-axis values in `ces_axes_per_cycle.json`; not a bug |
| `mmlu_retention_pct < 93%` | Forgetting abort should have fired | Check `retroverify_cycle*.json` for abort flag; if not set, investigate |

---

## Success criteria for Phase 1

- Main 10-cycle run completed; `experiment_summary.csv` with 11 rows, retention ≥ 93%.
- Screening aggregate CSV ranks all 16 variants; top-N recorded in impl log.
- Confirmatory sweep completed for top-N + reference; aggregate CSV populated.
- Purity validation JSON produced.
- All artifacts downloaded; instance stopped.

When all five are green, Phase 1 is done and Chapter 5 can be drafted
from point-estimate tables while Phase 2 funding is being secured.
