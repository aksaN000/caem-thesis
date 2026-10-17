# CAEM on Vast.ai — Complete Guide (April 2026)

> **Purpose:** Everything you need to rent a cloud GPU on Vast.ai

## Read This First: Local PC vs Vast.ai Remote (Most Common Confusion)

treat it as **renting another computer on the internet**.

- **Your PC (local machine):** where your normal VS Code is installed.
- **Vast instance (remote machine):** Linux server with the GPU; this is where training actually runs.
- **Vast website:** control panel to rent/stop instances and optionally open web terminal/Jupyter.

### Which VS Code do you use?

Use **your PC's VS Code** with the **Remote - SSH** extension.

After you connect to the Vast instance, your local VS Code window is now controlling the remote server:

- File explorer shows remote files (for example, `/root/caem`).
- Integrated terminal runs commands on the remote Linux machine.
- Python scripts execute on the remote GPU, not on your PC.

You do **not** install desktop VS Code on the Vast server.

### Which terminal do you use, and when?

Use this simple rule:

- **Local terminal (on your PC):** for `ssh` and `scp` commands that connect to or copy files to/from Vast.
- **Remote terminal (inside Remote-SSH VS Code, or Vast web terminal):** for `git clone`, `pip install`, `python scripts/...`, `tmux`, and all experiment runs.

### Quick mapping table

| Task | Run on Local PC terminal? | Run on Remote Vast terminal? |
|---|---|---|
| Open SSH session (`ssh -p ... root@...`) | Yes | No |
| Upload files to server (`scp ... root@...:~/caem/...`) | Yes | No |
| Clone repo on server (`git clone ...`) | No | Yes |
| Install dependencies (`pip install ...`) | No | Yes |
| Run experiment (`python scripts/run_experiment.py ...`) | No | Yes |
| Download results (`scp root@...:~/caem/outputs ...`) | Yes | No |

### Recommended beginner workflow

1. Rent instance in Vast website.
2. From your PC, connect with Remote-SSH in VS Code.
3. Open integrated terminal in that Remote-SSH window.
4. Run setup + experiment there (remote terminal).
5. From your PC terminal, use `scp` to download outputs.
6. Stop instance in Vast website to stop billing.

---

## Is CAEM Compatible With Vast.ai?

**Yes, fully.** `scripts/hardware.py` auto-detects GPU tier and sets precision + batch size automatically. You do not need to change any code to run on vast.ai — just upload and run.

---

## Full Machine Requirements

### GPU

| GPU | VRAM | batch_size (auto) | Precision (auto) | Full-run time (n=5000, 10 cycles) | Price/hr (approx) |
|---|---|---|---|---|---|
| **RTX 5090** | 32 GB | **32** | bf16 | ~10–12 h | ~$0.80–1.30 |
| A100 SXM 80 GB | 80 GB | 32 | bf16 | ~10–12 h | ~$1.80–2.80 |
| A100 PCIe 40 GB | 40 GB | 32 | bf16 | ~12–14 h | ~$1.00–1.50 |
| RTX 4090 | 24 GB | 16 | fp16 | ~16–18 h | ~$0.35–0.55 |
| T4 | 16 GB | 8 | fp16 | ~26–30 h | ~$0.35–0.50 |

> **Note on fp16 vs bf16:** On RTX 5090 and A100, the code automatically uses **bf16** (bfloat16), not fp16. This is intentional and better — bf16 has the same dynamic range as fp32, preventing L2 penalty overflow when summing 780M squared weight differences. On RTX 4090 (Ada Lovelace, no bf16 tensor cores), fp16 is used. Don't override — let `hardware.py` decide automatically.

### Single GPU Setup

CAEM's training loop runs on a **single GPU** — there is no DataParallel or distributed training in the codebase. Rent a single-GPU machine. Run the main experiment first (~10–12 h on RTX 5090), then run the ablation study sequentially after (~2 h).

### System RAM (CPU)
- **Minimum:** 32 GB RAM
- **Recommended:** 32–48 GB RAM (vast.ai machines typically come with 48–128 GB — any of these is fine)
- **Why:** theta_prev (the L2 anchor snapshot) is held in CPU RAM as fp32: 780M params × 4 bytes = **~3.1 GB**. The 21M-passage FAISS index is IVF-PQ compressed — only **~1.5 GB in RAM** (PQ codes + centroids). Plus HuggingFace dataset buffers (~2–4 GB), Python runtime. Total ~10–12 GB active. Going above 48 GB gives no further benefit — the bottleneck is always GPU VRAM, not CPU RAM.

### Storage (Disk)
- **Minimum:** 60 GB
- **Recommended:** 100 GB

| Component | Size |
|---|---|
| HuggingFace model cache (Flan-T5-Large + SBERT + RoBERTa-MNLI) | ~4–5 GB |
| HuggingFace dataset cache (FEVER/TriviaQA/NQ/TruthfulQA/StrategyQA/ARC/MMLU) | ~3–4 GB |
| Passage index (21M passages — Wikipedia corpus, IVF-PQ FAISS + passage texts) | ~15–17 GB |
| Cold-start memory (FAISS .faiss + .meta) | ~50 MB |
| Outputs (10 cycle checkpoints + eval JSONs + CSVs) | ~8–12 GB |
| Repo code | ~100 MB |
| **Total minimum** | **~35 GB** |
| **With headroom (recommended)** | **100 GB** |

> **Vast.ai default disk = 16 GB. You MUST change this to 100 GB before renting.** The passage index alone is ~15–17 GB — 16 GB default will crash immediately on index build or upload.

### FAISS: CPU or GPU? And which index type?

**Keep faiss-cpu.** The code uses standard CPU-side FAISS (`import faiss`, IVF-PQ index). There is no GPU FAISS code anywhere in the codebase — no `StandardGpuResources`, no `index_cpu_to_gpu`. Installing `faiss-gpu` would bring no benefit and frequently has CUDA version conflicts. Always install `faiss-cpu`.

**Keep IVF-PQ for both indices.** The memory store (1M capacity) auto-bootstraps with IndexFlatIP for early cycles, then auto-promotes to IVF-PQ once 20K+ episodes exist. This is the correct design. Do NOT force `memory_index_type = "flat_ip"` permanently — at 1M entries, a flat index requires a linear scan over 1M × 768 float values per query. With 10 cycles × 5000 queries = 50K total queries, that's several extra hours of CPU work and is slower than IVF-PQ. The extra RAM available on vast.ai machines (48–128 GB) does not make flat search faster. IVF-PQ stays in RAM at ~1.5 GB either way.

### theta_prev Placement (L2 Penalty Anchor)

theta_prev is the frozen weight snapshot used in the L2 regularisation loss. After the code updates in this session:

- **RTX 4090 / A100 (VRAM ≥ 24 GB):** theta_prev automatically moved to GPU once before the training loop. This eliminates ~3 GB of PCIe transfers per training batch → **3–4× faster fine-tuning per cycle**. Zero accuracy impact.
- **RTX 3060 / T4 (VRAM < 24 GB):** theta_prev stays on CPU. Correct and safe.

This is now applied automatically — you don't need to change anything.

---

## Part 1: Account Setup and Billing

1. Go to [https://vast.ai](https://vast.ai) and create an account.
2. In the top-right menu, go to **Billing**.
3. Click **Add Credit** → choose **Stripe** (Visa/Mastercard, USD international).
4. Load **$50–75** for the initial Phase 1 main-run + screening pass. The full Phase 1 envelope (main run + screening + confirmatory sweep + aggregation) fits in ~$200 on an RTX 4090 — see Part 10 for the full two-phase cost table.

---

## Part 2: Choosing and Configuring Your Instance

### Step 1 — Set the template (do this BEFORE selecting a machine)

1. On the **Search** page, find **"Edit Image & Config"** (top-left).
2. Select the **PyTorch** template (e.g., `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel`).
3. **CRITICAL — Disk Space:** Scroll to the bottom of the config pop-up. Find the **Allocated Disk Space** slider. Default is 16 GB — drag it to **100 GB**. The 21M-passage index alone is ~15–17 GB; 50 GB is not enough once you include model cache and outputs.
4. Click **Save**.

### Step 2 — Select the machine

1. In the left filter panel: check **Verified**, **1x GPU**, select your GPU type.
2. Sort by reliability (aim for ≥ 99%) and price.
3. Click **RENT**.

---

## Part 3: Connecting

Once the instance shows "Running" in the Instances tab:

### Option A -- VS Code Remote-SSH (recommended for beginners)

This is usually the easiest because you keep using the same VS Code UI on your PC.

1. Install the **Remote - SSH** extension in your local VS Code.
2. Click **CONNECT** in Vast Instances tab and copy the SSH command (example: `ssh -p 24501 root@123.45.67.89`).
3. In local VS Code: `Ctrl+Shift+P` -> `Remote-SSH: Add New SSH Host` -> paste command.
4. `Ctrl+Shift+P` -> `Remote-SSH: Connect to Host` -> select it.

How to confirm you are really remote:

- Bottom-left corner shows `SSH: <host>`.
- Integrated terminal shows Linux prompt (for example `root@...:~#`).
- `pwd` points to remote paths like `/root/caem` (not `C:\...`).

### Option B -- Plain SSH terminal

Use this if you do not want Remote-SSH UI. Run this in your **local PC terminal**:

```bash
ssh -p 24501 root@123.45.67.89
```

After this, the shell you see is remote.

### Option C -- Vast web terminal / Jupyter

Click the **Jupyter** button in Vast Instances page. This opens browser-based tools hosted on the remote machine.

- Good as backup when SSH client has issues.
- Same remote environment as SSH.
- Less comfortable than local VS Code for larger projects.

### Final clarity on your question

- You use **your PC's VS Code app**.
- But after Remote-SSH connection, that VS Code window edits and runs code on the **remote Vast machine**.
- Commands for training should run in the **remote terminal** (Remote-SSH terminal or Vast web terminal), not in a normal local terminal.

---

## Part 4: Uploading Code and Data

### Code -- always use git (run on REMOTE terminal)
```bash
# On the vast.ai machine:
git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git caem
cd caem
```

### Data -- upload with scp from your LOCAL machine
```bash
# From your local Windows terminal (use your SSH port and IP):
scp -r -P 24501 "C:\path\to\data\passage_index" root@123.45.67.89:~/caem/data/
scp -r -P 24501 "C:\path\to\outputs\cold_start_memory" root@123.45.67.89:~/caem/outputs/
```

### Alternative — build passage index on the machine
If your passage index isn't ready locally, build it on vast.ai. The full 21M-passage Wikipedia corpus:
```bash
# Full thesis-scale run (21M passages — ~1.5–2 h encode time on RTX 5090)
python scripts/build_passage_index.py --max_passages 21000000 --output_dir data/passage_index
```
This streams Wikipedia passages from HuggingFace — no file upload needed, but takes time to encode all 21M passages with SBERT. Ensure you have 100 GB disk before starting.

> **Do not use `--max_passages 500000`** — that was an old dev-scale default. The thesis uses the full 21M Wikipedia corpus.

### Alternative — Google Drive
```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/FOLDER_ID" -O data/
```

---

## Part 5: Environment Setup

From this point onward, run commands in a **REMOTE terminal** (Remote-SSH integrated terminal, plain SSH terminal, or Vast web terminal).

```bash
cd ~/caem

# Install all dependencies (faiss-cpu, NOT faiss-gpu)
pip install torch transformers datasets sentence-transformers \
    faiss-cpu numpy scipy scikit-learn

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

# Verify hardware profile (checks batch size + precision auto-detection)
python -c "from scripts.hardware import print_hardware_summary; print_hardware_summary()"
```

Expected output on A100 80GB:
```
  GPU:        NVIDIA A100-SXM4-80GB
  VRAM:       79.2 GB
  Precision:  bf16
  Batch size: 32 (recommended)
```

---

## Part 6: Running the Experiment

### Always start tmux first
If your SSH drops (internet hiccup, laptop sleep), tmux keeps the run alive.
```bash
tmux new-session -s caem
```

### Full 10-cycle run
```bash
python scripts/run_experiment.py \
    --output_dir outputs/full_run \
    --num_cycles 10 \
    --n_questions 5000 \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --passage_index data/passage_index \
    --cold_start_memory outputs/cold_start_memory/memory_store \
    2>&1 | tee outputs/run.log
```

**To detach from tmux** (leaves it running): `Ctrl+B`, then `D`  
**To reattach after reconnecting**: `tmux attach -t caem`  
**To watch logs live** from a second terminal: `tail -f outputs/full_run/run.log`

### Checkpoint recovery if the run crashes
```bash
# Check which cycles completed
ls outputs/full_run/ | grep cycle
ls outputs/full_run/eval/ | grep .json

# Resume from cycle N
python scripts/run_experiment.py \
    --resume_from_cycle 4 \
    --output_dir outputs/full_run \
    --passage_index data/passage_index \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge
```

---

## Part 6.5: External Baseline Runs (B1–B7)

> **Run order:** Baselines run **after** the main CAEM 10-cycle experiment (Part 6) and **before** the ablation study (Part 7, Step 3). They populate the Chapter 5 headline comparison table and are independent of each other, so they can be run in parallel if multiple Vast.ai instances are available, or sequentially on a single instance.

The baseline panel has eight entries (B1–B8) plus one optional ceiling (STaR). B8 Self-RAG is citation-only and not run numerically (see `chapter_5.tex` §5.1). The remaining seven are all launched from the same machine as the main run.

### Prerequisite — DPR passage index

B3, B4, and B5 all require the Wikipedia DPR passage index. If the main CAEM run already built it, reuse that path (`data/passage_index` or whatever was passed to `--passage_index` in Part 6). Otherwise build it once:

```bash
python -m scripts.build_passage_index \
    --output_dir data/passage_index
```

### Step 1 — Inference-only baselines (B1–B5), single script

`scripts/run_baseline.py` handles all five inference baselines via a `--baseline` flag. Outputs land under `outputs/baselines/<name>/<benchmark>_cycle0.json`, schema-compatible with the main run's eval JSONs so `eval/reporting.py` consumes them unchanged.

```bash
# --- B1 Zero-shot Flan-T5-Large (floor) — ~15 min on RTX 4090, n=500 ---
python -m scripts.run_baseline \
    --baseline zero_shot \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B1_zero_shot.log

# --- B2 Chain-of-Thought — ~20 min, n=500 ---
python -m scripts.run_baseline \
    --baseline cot \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B2_cot.log

# --- B3 DPR-RAG — ~45 min, n=500 (retrieval dominates wall-clock) ---
python -m scripts.run_baseline \
    --baseline rag \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --passage_index data/passage_index \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B3_rag.log

# --- B4 CoT + DPR-RAG — ~50 min, n=500 ---
python -m scripts.run_baseline \
    --baseline cot_rag \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --passage_index data/passage_index \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B4_cot_rag.log

# --- B5 FLARE — ~90 min, n=500 (active retrieval is slow) ---
python -m scripts.run_baseline \
    --baseline flare \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    --passage_index data/passage_index \
    --flare_theta 0.4 \
    --flare_look_ahead 64 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B5_flare.log
```

**Total wall-clock for B1–B5 on a single RTX 4090:** approximately 3.5–4 hours.

### Step 2 — Training baselines (B6, B7), shared script

`scripts/run_simple_ft.py` handles both B6 Vanilla FT and B7 EWC-only FT through flags; the only difference is whether `--use_l2_anchor` and `--use_mmlu_guard` are set. Both use ten cycles matching the main CAEM run.

```bash
# --- B6 Vanilla FT — ~6 h on RTX 4090, 10 cycles ---
# Neither the L2 anchor nor the MMLU guard is applied. Every selected
# episode is treated as a clean label. This baseline is expected to
# drift on MMLU; that drift is the measurement.
python -m scripts.run_simple_ft \
    --baseline_name vanilla_ft \
    --num_cycles 10 \
    --output_dir outputs/baselines/vanilla_ft \
    --passage_index data/passage_index \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    2>&1 | tee outputs/baselines/B6_vanilla_ft.log

# --- B7 EWC-only FT — ~6 h on RTX 4090, 10 cycles ---
# L2 anchor (lambda = 0.01) and MMLU guard (rho_min = 0.93) on, but
# verifier, nine-signal composite, memory, and tier routing all off.
python -m scripts.run_simple_ft \
    --baseline_name ewc_only_ft \
    --use_l2_anchor \
    --use_mmlu_guard \
    --num_cycles 10 \
    --output_dir outputs/baselines/ewc_only_ft \
    --passage_index data/passage_index \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 500 \
    2>&1 | tee outputs/baselines/B7_ewc_only_ft.log
```

**Total wall-clock for B6+B7 on a single RTX 4090:** approximately 12 hours (sequential). Can be run on two instances in parallel if budget permits.

### Step 2B — Optional STaR ceiling run (only if Phase 1 is under envelope)

STaR (Zelikman et al. 2022) is cited in Chapter 2 §2.2 and Chapter 5 §5.1 as an aspirational ceiling for pure iterative-refinement without verification. A numerical row for STaR is worth the extra ~$3 of rental **only if** Steps B1–B7 plus the full CAEM run came in under the USD 200 Phase 1 envelope. If the budget is already stretched, skip this step and leave STaR as a citation-only ceiling.

```bash
# --- Optional: STaR ceiling — ~7–8 h on RTX 4090, 10 cycles ---
# Reuses scripts/run_simple_ft.py with --use_rationalisation:
#   per-cycle forward-filter + back-rationalise pass replaces the
#   training targets with (rationale + answer) strings before the
#   3-epoch fine-tune. No L2 anchor, no MMLU guard — the point of
#   STaR is the rationalisation delta over B6, not retention.
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

Sanity check: `outputs/baselines/star/training_log.jsonl` should show `"use_rationalisation": true` on every cycle row, and per-cycle in-domain EM should be monotonically non-decreasing. If EM regresses, the rationalisation pass is producing low-quality rationales — record the finding in `caem-implementation-log.md` and cite the known "STaR-on-factual-QA" mismatch caveat in the Ch2 §2.2 positioning paragraph.

### Step 3 — Sanity check

After each baseline finishes, confirm the per-benchmark JSONs exist and the aggregate summary line is sensible:

```bash
ls outputs/baselines/*/
for f in outputs/baselines/*/*.json; do
  python -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], 'EM=', d.get('aggregate',{}).get('em'), 'F1=', d.get('aggregate',{}).get('f1'))" "$f"
done
```

Expected floor: B1 zero-shot should land near published Flan-T5-Large numbers (FEVER ~55–60% EM, TriviaQA ~40–45%). Anything dramatically below that indicates a loader or prompt bug; investigate before moving on.

---

## Part 7: Post-Experiment Scripts

> **Note on automation:** `run_experiment.py` (Part 6) is fully automated — one command runs Cycle 0 through Cycle 10 without human intervention. Part 6.5 (external baselines B1–B7) is also scripted but must be launched per baseline. Everything in Part 7 below must be run **manually after** the main experiment and the baselines complete.

### Step 1 — Fill published baselines (REQUIRED before ablation)

`run_ablation.py` reads from `published_baselines.template.json`. All values in that file are `0.0` placeholders — they are **static literature citations, not live API calls**. There is no GPT API, no OpenAI key needed. Fill the file manually before running ablation:

```bash
# Edit published_baselines.template.json in the repo root
# Set these values (from Liu et al. 2024, RA-ISF paper — StrategyQA only):
#
#   "gpt35_vanilla":   { "strategyqa": 65.2 }
#   "gpt35_rag":       { "strategyqa": 64.7 }
#   "selfrag_13b":     { "strategyqa": 67.2 }
#
# All other benchmark fields (fever, triviaqa, etc.) stay 0.0 — those
# baselines were not reported in the cited paper for those benchmarks.
nano published_baselines.template.json
```

### Step 2 — Theory validation (~30 min)
```bash
python scripts/run_purity_validation.py \
    --output_dir outputs/purity_validation
```

### Step 3 — Ablation study (two-phase protocol — see Part 10 for cost breakdown)

The ablation study uses the **named 16-variant registry** defined in
`caem/ablation/variants.py` and the cyclic driver
`scripts/run_cyclic_ablation.py`. The legacy `run_ablation.py` +
`run_pub0405_variants.py` scripts have been retired — they combined
inference-time and training-time variants under one label, which
conflates the mechanism's effect with the counterfactual confound of
running an ablated mechanism on weights shaped by that mechanism in
prior cycles.

**Phase 1 screening pass** (3 cycles, n=1500, seed=42, entire
16-variant registry):

```bash
# Loop over all 16 variants — each rerun is independent
for v in full no_self_improvement no_tier1 no_retrieval no_verification \
         no_nli no_selfcons no_entropy no_routing no_cold_start \
         no_abort_guard no_l2_anchor no_retroverify no_memory_prune \
         no_mcdropout lower_u_threshold; do
  python scripts/run_cyclic_ablation.py \
      --variant "$v" \
      --seed 42 \
      --screening_mode \
      --passage_index data/passage_index \
      --cold_start_memory outputs/cold_start_memory/memory_store \
      --output_dir outputs/ablation \
      2>&1 | tee outputs/ablation/"$v"_screening.log
done
```

Screening is a methodological instrument to rank variants — its rows
do NOT enter the Chapter 5 ablation table.

**Phase 1 confirmatory pass** (10 cycles, n=5000, seed=42, reference
run + top-N variants by $\Delta$CES from screening):

```bash
# Example — replace the variant list with screening-selected top candidates
for v in full no_self_improvement no_verification no_retroverify no_l2_anchor no_cold_start; do
  python scripts/run_cyclic_ablation.py \
      --variant "$v" \
      --seed 42 \
      --passage_index data/passage_index \
      --cold_start_memory outputs/cold_start_memory/memory_store \
      --output_dir outputs/ablation \
      2>&1 | tee outputs/ablation/"$v"_confirmatory.log
done
```

**Phase 2 multi-seed upgrade** (post-funding — same confirmatory
invocations with `--seed 123` then `--seed 456`; no other changes
required):

```bash
for seed in 123 456; do
  for v in full no_self_improvement no_tier1 no_retrieval no_verification \
           no_nli no_selfcons no_entropy no_routing no_cold_start \
           no_abort_guard no_l2_anchor no_retroverify no_memory_prune \
           no_mcdropout lower_u_threshold; do
    python scripts/run_cyclic_ablation.py \
        --variant "$v" --seed "$seed" \
        --passage_index data/passage_index \
        --cold_start_memory outputs/cold_start_memory/memory_store \
        --output_dir outputs/ablation
  done
done
```

**Aggregate across all discovered seeds** (auto-switches
point-estimate → mean$\pm$std once ≥2 seeds exist):

```bash
python scripts/aggregate_ablation.py \
    --output_dir outputs/ablation
```

Output files (written to `outputs/ablation/`):
- `ablation_table.csv` — per-variant CES axes at the final cycle.
- `ablation_per_cycle.csv` — per-variant CES axes traced across every cycle.
- `ablation_aggregate_manifest.json` — variants × seeds manifest, for reproducibility.

---

## Part 8: Downloading Results

Key output files:

| File | Chapter |
|---|---|
| `outputs/full_run/experiment_summary.csv` | Ch5 Table 5.2 (includes `mmlu_retention_pct` column) |
| `outputs/full_run/all_cycle_results.json` | Full per-cycle per-benchmark results |
| `outputs/full_run/retroverify_cycle*.json` | Per-cycle forgetting scores + MMLU retention |
| `outputs/ablation/ablation_table.csv` | Ch5 §5.5 Ablation table (Phase 1 point estimates / Phase 2 mean±std) |
| `outputs/ablation/ablation_per_cycle.csv` | Ch5 per-cycle $\Delta$CES trace figures |
| `outputs/purity_validation/theory_validation.json` | Ch5 §5.3 Theory tables |
| `outputs/full_run/calibration/calibrated_config.json` | Ch5 §5.1 Calibration results |

Download everything at once:
```bash
# From your LOCAL machine terminal:
scp -r -P 24501 root@123.45.67.89:~/caem/outputs/ "C:\Users\YourName\Desktop\caem_results\"
```
Or in VS Code Remote SSH: right-click the `outputs/` folder → Download.

**Stop the instance immediately after downloading — idle instances still bill at full rate.**

---

## Part 9: Claude Code on Vast.ai — Your Subscription Question

**Short answer: You do NOT need a separate Anthropic API key if you have a Claude.ai subscription that includes Claude Code (Pro Max or equivalent annual plan).**

Your subscription credentials work on any machine — including a remote Linux server on vast.ai. Here is how to authenticate:

### Step 1 — Install Claude Code on the vast.ai machine
```bash
# Install Node.js (vast.ai PyTorch images typically don't include it)
apt-get update && apt-get install -y nodejs npm

# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

### Step 2 — Authenticate using your existing subscription

On a headless server you can't do a browser pop-up directly, but Claude Code supports a URL-based login flow:

```bash
claude auth login
```

This will print a URL like:
```
Open this URL in your browser to authenticate:
https://claude.ai/auth/claude-code?code=XXXXXX
```

Open that URL on your **local machine's browser** (the same browser where you're logged into your Claude account). Complete the login there. The credentials are then stored on the vast.ai machine and you're authenticated — tied to your subscription, no API key needed.

### Step 3 — Use Claude Code

```bash
cd ~/caem
claude   # launches Claude Code in the project
```

Your usage is billed against your subscription quota, exactly like using Claude Code locally. Vast.ai only charges for GPU/compute hours — Claude Code usage is separate and covered by your plan.

> **One practical note:** vast.ai instances are temporary and get destroyed after you stop them. Your Claude Code credentials are stored in `~/.claude/` on the instance. If you rent a new instance, you'll need to re-authenticate once. This takes about 30 seconds.

---

## Part 10: Cost and Time Estimates

The thesis uses a **two-phase ablation protocol** (see Chapter~5 §5.5
``Ablation Methodology''). Phase~1 is self-funded and runs a screening
pass + confirmatory pass on a single seed (42). Phase~2 is a
post-funding upgrade that re-runs the confirmatory pass on two
additional seeds (123, 456) so the main ablation table can report
mean$\pm$std.

### GPU tier reference (Vast.ai market rates, April 2026)

| GPU | VRAM | batch_size | Precision | Hourly price (approx) |
|---|---|---|---|---|
| RTX 4090 | 24 GB | 16 | fp16 | ~$0.35–0.55 |
| RTX 5090 | 32 GB | 32 | bf16 | ~$0.80–1.30 |
| A100 PCIe 40 GB | 40 GB | 32 | bf16 | ~$1.00–1.50 |
| A100 SXM 80 GB | 80 GB | 32 | bf16 | ~$1.80–2.80 (Phase 2 recommended) |
| H100 80 GB | 80 GB | 32 | bf16 | ~$2.20–2.80 (Phase 2 fast track) |

### Phase 1 — self-funded (~USD 200 total, single seed)

Screening + confirmatory on seed 42 only. This is the tier you can run
immediately; results enter Chapter 5 as point estimates with a
single-seed limitation disclosure.

| Step | Command entry point | Time | GPU | Cost |
|---|---|---|---|---|
| Environment setup + HF cache pre-pull | `pip install …` + `huggingface_hub.snapshot_download` | ~20 min | 4090 | ~$0.15 |
| Passage index build (21M, one-time) | `scripts/build_passage_index.py --max_passages 21000000` | ~2–3 h | 4090 | ~$1.00–1.60 |
| Cold-start seeding | `scripts/run_cold_start.py` | ~20 min | 4090 | ~$0.15 |
| **Main 10-cycle run** (n=5000, seed=42) | `scripts/run_experiment.py` | ~16–18 h | 4090 | ~$6.50–9.50 |
| **Screening sweep** — 16 variants × 3 cycles × n=1500 × seed=42 | `scripts/run_cyclic_ablation.py --screening_mode` for each variant | ~14–20 h total (batched sequentially) | 4090 | ~$6–11 |
| **Confirmatory sweep** — top ~5 variants × 10 cycles × n=5000 × seed=42 | `scripts/run_cyclic_ablation.py` (no `--screening_mode`) | ~50–60 h total | 4090 | ~$20–33 |
| Purity validation + ablation aggregation | `run_purity_validation.py` + `aggregate_ablation.py` | ~45 min | 4090 | ~$0.30 |
| **Phase 1 total (RTX 4090, single seed)** | | **~85–105 h compute** | | **~$35–55** |
| Phase 1 total on **A100 SXM 80 GB** (faster wall-clock, higher hourly rate) | | ~55–65 h compute | | **~$100–160** |
| Phase 1 conservative budget (buffer for retries, idle minutes between steps, and rounding) | | | | **~$200** |

**Phase 1 recommendation:** rent an **RTX 4090** instance. It has
enough VRAM for Flan-T5-Large fine-tuning, is the cheapest reliable
vast.ai tier, and keeps the Phase 1 bill comfortably below USD 200
with buffer for retries. Upgrade to A100 SXM 80 GB only if you need
wall-clock speed and can absorb the ~2–3× cost multiplier.

### Phase 2 — post-funding upgrade (+USD 700, seeds 123 and 456)

Phase 2 re-runs the confirmatory sweep on two additional seeds so the
ablation table reports mean$\pm$std rather than point estimates. Drop
the screening pass entirely (Phase 1 already ranked the variants) and
run all **14 cyclic variants + reference** across both new seeds.

| Step | Command entry point | Per-seed time | GPU | Per-seed cost |
|---|---|---|---|---|
| Confirmatory sweep — 15 variants × 10 cycles × n=5000 × seed=123 | `scripts/run_cyclic_ablation.py --seed 123` per variant | ~150–180 h | 4090 | ~$65–90 |
| Same, seed=456 | `scripts/run_cyclic_ablation.py --seed 456` per variant | ~150–180 h | 4090 | ~$65–90 |
| Aggregation (auto-detects multi-seed layout) | `scripts/aggregate_ablation.py` | ~5 min | 4090 | — |
| **Phase 2 total on RTX 4090** | | ~300–360 h | | **~$130–180** |
| **Phase 2 total on A100 SXM 80 GB** (recommended once funded) | | ~180–220 h | | **~$400–620** |
| Phase 2 conservative budget (A100 tier + retry headroom + Phase 2 also re-running main run across 3 seeds if desired) | | | | **~$700** |
| **Phase 1 + Phase 2 combined ceiling** | | | | **~$900–950** |

> **BDT note:** USD 930 is roughly 1,10,000 BDT at April-2026 rates.
> Phase 1 alone (~USD 200) is approximately 23,000–24,000 BDT, which
> is the intended self-funded envelope.

### Code-level upgrade path

Moving from Phase 1 to Phase 2 requires **no code changes**. The output
layout (`outputs/ablation/<variant>/seed_<N>/`) and the aggregator
(`scripts/aggregate_ablation.py`) are already multi-seed aware: when a
single seed exists the aggregator prints point estimates; when two or
more seeds exist it automatically switches to mean$\pm$std with
independent-seed std propagation on $\Delta$CES. The only operational
change is adding `--seed 123` and `--seed 456` to your ablation
invocations.

---

## Part 11: Common Issues and Fixes

**"CUDA out of memory" during fine-tuning**
You're likely on a GPU that hardware.py assigned too-large a batch size for. Check what it detected:
```bash
python -c "from scripts.hardware import print_hardware_summary; print_hardware_summary()"
```
If batch_size looks wrong, override in `caem/config.py` directly:
```python
batch_size: int = 4   # force smaller batch
```

**SSH disconnects mid-run**
This is why you're using tmux. Reconnect: `ssh -p PORT root@IP`, then `tmux attach -t caem`.

**HuggingFace downloads fail or are slow**
Pre-cache models before the run:
```bash
export HF_HOME=/root/caem/hf_cache
python -c "
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained('google/flan-t5-large')
T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
AutoTokenizer.from_pretrained('roberta-large-mnli')
AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print('All models cached.')
"
```

**faiss import error**
```bash
pip install faiss-cpu   # always cpu, not gpu
```

**Run crashed — don't know which cycle**
```bash
ls outputs/full_run/ | grep "^cycle_"     # completed cycle checkpoints
ls outputs/full_run/eval/ | tail -20      # last saved eval files
```
Use the highest complete cycle number in `--resume_from_cycle`.

---

## Quick Reference Card

```bash
# 1. Connect (LOCAL terminal)
ssh -p PORT root@IP

# 2. Get code (REMOTE terminal)
git clone https://TOKEN@github.com/USER/REPO.git caem && cd caem

# 3. Install (REMOTE terminal)
pip install torch transformers datasets sentence-transformers faiss-cpu numpy scipy scikit-learn

# 4. Upload data (LOCAL terminal)
scp -r -P PORT data/passage_index root@IP:~/caem/data/
scp -r -P PORT outputs/cold_start_memory root@IP:~/caem/outputs/

# 5. Verify hardware (REMOTE terminal)
python -c "from scripts.hardware import print_hardware_summary; print_hardware_summary()"

# 6. Start experiment inside tmux (REMOTE terminal)
tmux new-session -s caem
python scripts/run_experiment.py \
  --output_dir outputs/full_run --num_cycles 10 --n_questions 5000 \
  --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
  --passage_index data/passage_index \
  --cold_start_memory outputs/cold_start_memory/memory_store \
  2>&1 | tee outputs/run.log

# 7. Detach (leave running)
Ctrl+B, D

# 8. Reattach after reconnect
tmux attach -t caem

# 9. Download results (LOCAL terminal)
scp -r -P PORT root@IP:~/caem/outputs/ C:\Users\YourName\Desktop\caem_results\

# 10. STOP THE INSTANCE (Vast website)
```
