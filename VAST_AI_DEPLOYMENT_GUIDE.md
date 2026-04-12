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
4. Load **$25–35**. Full run on RTX 4090: ~$8–10 total. On A100 SXM 80GB: ~$20–25 total.

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

## Part 7: Post-Experiment Scripts

> **Note on automation:** `run_experiment.py` (Part 6) is fully automated — one command runs Cycle 0 through Cycle 10 without human intervention. Everything in Part 7 below must be run **manually after** the main experiment completes.

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

### Step 3 — Ablation study (~2–3 hours, needs separate GPU time)
```bash
python scripts/run_ablation.py \
    --caem_results outputs/full_run/all_cycle_results.json \
    --cycle3_checkpoint outputs/full_run/cycle_3 \
    --output_dir outputs/ablation_results \
    --published_baselines_json published_baselines.template.json
```

> **AB3/AB4/PUB-04/PUB-05 variants** each require separate training runs (they cannot be derived from the main 10-cycle checkpoints). If you want the full ablation table, run `run_pub0405_variants.py` first — see NEXT_SESSION_PLAN.md Phase F for the exact commands.

---

## Part 8: Downloading Results

Key output files:

| File | Chapter |
|---|---|
| `outputs/full_run/experiment_summary.csv` | Ch5 Table 5.2 (includes `mmlu_retention_pct` column) |
| `outputs/full_run/all_cycle_results.json` | Full per-cycle per-benchmark results |
| `outputs/full_run/retroverify_cycle*.json` | Per-cycle forgetting scores + MMLU retention |
| `outputs/ablation_results/ablation_summary.json` | Ch5 Table 5.5 |
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

### RTX 5090 (~$1.00/hr) — your hardware, recommended

| Phase | Time | Cost |
|---|---|---|
| Environment setup | ~10 min | ~$0.15 |
| Passage index build (21M passages, upload or build) | ~1.5–2 h | ~$1.50–2.00 |
| Cold-start seeding (if not pre-seeded) | ~20 min | ~$0.30 |
| Full 10-cycle experiment (n=5000, batch=32, bf16) | ~10–12 h | ~$10–12 |
| Ablation study (sequential after main run) | ~1.5–2 h | ~$1.50–2.00 |
| Purity validation | ~20 min | ~$0.35 |
| **Total (1× RTX 5090)** | **~14–17 h** | **~$14–17** |

### RTX 4090 (~$0.40/hr) — cheapest option if 5090 unavailable

| Phase | Time | Cost |
|---|---|---|
| Environment setup | ~10 min | ~$0.07 |
| Passage index build (21M passages) | ~2–3 h | ~$0.80–1.20 |
| Full 10-cycle experiment (batch=16, fp16) | ~16–18 h | ~$6.50–7.50 |
| Ablation + validation | ~3 h | ~$1.20 |
| **Total** | **~22–25 h** | **~$9–10** |

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
