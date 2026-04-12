# CAEM on Vast.ai — Complete Guide (April 2026)

> **Purpose:** Everything you need to rent a cloud GPU on Vast.ai, upload your code and data, run the full 10-cycle CAEM experiment, and download results. Covers from account creation to final CSV. Replaces the previous stale guide.

---

## Is CAEM Compatible With Vast.ai?

**Yes, fully.** `scripts/hardware.py` already auto-detects A100, RTX 4090, T4, and V100. No code changes are needed to run on vast.ai.

### GPU options and what they give you

| GPU | VRAM | Full-run time (n=5000) | Recommended? | Price (approx) |
|---|---|---|---|---|
| **A100 SXM 80 GB** | 80 GB | ~12–14 h | ✅ Best choice | ~$1.50–2.50/hr |
| **A100 PCIe 40 GB** | 40 GB | ~14–16 h | ✅ Good | ~$1.00–1.50/hr |
| **RTX 4090** | 24 GB | ~16–18 h | ✅ Cheapest reliable | ~$0.35–0.55/hr |
| **RTX 3090** | 24 GB | ~18–20 h | ⚠️ Slower, fine | ~$0.25–0.40/hr |
| **T4** | 16 GB | ~24–28 h | ⚠️ Slow but works | ~$0.30–0.50/hr |

**Recommendation for thesis:** RTX 4090 at ~$0.35/hr gives the best price-to-speed ratio. A100 SXM if you want it done overnight and can afford it.

### Disk space you need

| Component | Size |
|---|---|
| HuggingFace model cache (Flan-T5-Large + SBERT + RoBERTa-MNLI) | ~4–5 GB |
| HuggingFace dataset cache (FEVER/TriviaQA/NQ/TruthfulQA/StrategyQA/ARC/MMLU) | ~3–4 GB |
| Passage index (500K passages, thesis standard) | ~1.5 GB |
| Cold-start memory | ~50 MB |
| Outputs (10 cycle checkpoints + eval JSONs + CSVs) | ~8–12 GB |
| Your repo code | ~100 MB |
| **Total minimum** | **~20 GB** |
| **Recommended (with headroom)** | **50 GB** |

---

## Part 1: Account Setup and Billing

1. Go to [https://vast.ai](https://vast.ai) and create an account with your email.
2. In the top-right menu, go to **Billing**.
3. Click **Add Credit** → choose **Stripe** (Visa/Mastercard that supports USD international payments).
4. Load **$20–30**. A full 10-cycle run on an RTX 4090 costs roughly **$7–9** total (18–22 hours × $0.35–0.45/hr). The rest is buffer for setup time and re-runs.

---

## Part 2: Choosing and Configuring Your Instance

### Step 1: Set up the template (do this BEFORE selecting a machine)

1. On the **Search** page, find the **"Edit Image & Config"** box (top-left).
2. From the Recommended list, select the **PyTorch** template (e.g., `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` or similar recent version). This gives you CUDA + Python + pip pre-installed.
3. **CRITICAL — Disk Space:** Scroll to the bottom of the config pop-up. Find the **Allocated Disk Space** slider. The default is **16 GB — this is too small**. Drag it to **50 GB**.
4. Click **Save**.

### Step 2: Select the machine

1. In the left filter panel, check:
   - **Verified** (filters out unstable hosts)
   - **1x GPU**
   - Select your GPU type (RTX 4090 recommended)
2. Sort by **Price** or **Reliability**.
3. Look for: reliability ≥ 99%, DLP (download/upload speed) ≥ 500 Mbps, price within budget.
4. Click **RENT**.

### Step 3: Wait for the instance to start

In the **Instances** tab, your machine will show "Loading" for 1–3 minutes, then switch to "Running". Once "Running", you can connect.

---

## Part 3: Connecting to Your Instance

Vast.ai gives you a standard Linux server with an SSH key. You have three ways to connect:

### Option A: VS Code Remote SSH (recommended)

1. In VS Code, install the **Remote - SSH** extension (search in Extensions panel).
2. In your Instances tab, click **CONNECT** next to your running instance.
3. Copy the SSH command shown (e.g., `ssh -p 24501 root@123.45.67.89 -L 8080:localhost:8080`).
4. In VS Code, press `Ctrl+Shift+P` → `Remote-SSH: Add New SSH Host` → paste the command → Enter.
5. Press `Ctrl+Shift+P` → `Remote-SSH: Connect to Host` → select the host you just added.
6. A new VS Code window opens connected to the server. You get a full IDE, integrated terminal, and drag-and-drop file upload.

### Option B: SSH from terminal (simple, works anywhere)

Copy the SSH command from the Instances tab and run it in your local terminal:
```bash
ssh -p 24501 root@123.45.67.89
```

### Option C: Jupyter Notebook (web browser, no install needed)

Click the **Jupyter** button in the Instances tab. A web-based file browser + notebook opens directly in your browser. You can upload files here and open a terminal from `New → Terminal`.

---

## Part 4: Uploading Your Code and Data

Once connected, you need to get three things onto the machine:
1. Your CAEM repository code
2. The passage index (if using RAG)
3. The cold-start memory (if pre-seeded)

### Method A: Git clone (for code — always do this)

```bash
# On the vast.ai machine:
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git caem
cd caem
```

If your repo is private, use a Personal Access Token:
```bash
git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git caem
```

If you don't have a GitHub repo yet, the fastest option is to push your code before running. From your local machine:
```bash
cd "path/to/your/caem/folder"
git init
git add .
git commit -m "CAEM full codebase for thesis run"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Method B: SCP upload (for data files — passage index, cold-start memory)

From your **local** machine terminal, use the SSH details from the Instances tab:
```bash
# Upload the passage index (1.5 GB — takes 3–5 min on good connection)
scp -r -P 24501 "path/to/data/passage_index" root@123.45.67.89:~/caem/data/

# Upload cold-start memory (50 MB — fast)
scp -r -P 24501 "path/to/outputs/cold_start_memory" root@123.45.67.89:~/caem/outputs/
```

### Method C: Google Drive via gdown (alternative if your files are in Drive)

```bash
pip install gdown
# For a folder:
gdown --folder "https://drive.google.com/drive/folders/YOUR_FOLDER_ID" -O data/
```

### Method D: Let the experiment build the passage index on the machine (easiest)

If you don't have the passage index ready on your local machine, **just build it on vast.ai**:
```bash
# ~30 min on RTX 4090 for 500K passages (thesis standard)
python scripts/build_passage_index.py \
    --max_passages 500000 \
    --output_dir data/passage_index
```

This streams Wikipedia directly from HuggingFace — no file upload needed.

---

## Part 5: Environment Setup

Once connected and code is on the machine:

```bash
cd caem   # or wherever your repo is

# Install all dependencies
pip install torch transformers datasets sentence-transformers \
    faiss-gpu numpy scipy scikit-learn

# Verify GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
```

Expected output example: `NVIDIA GeForce RTX 4090` | `VRAM: 23.7 GB`

**Note on faiss:** If `faiss-gpu` fails to install (sometimes has CUDA version conflicts), try:
```bash
pip install faiss-cpu   # slower retrieval but works on any machine
```

---

## Part 6: Running the Full Experiment

### Use tmux FIRST — this is critical

Vast.ai instances disconnect if your SSH session drops (internet hiccup, laptop sleep, etc.). Without tmux, a disconnection would kill the 14-hour run mid-way. Always use tmux:

```bash
# Create a named session
tmux new-session -s caem

# Now start the experiment inside tmux
python scripts/run_experiment.py \
    --output_dir outputs/full_run \
    --num_cycles 10 \
    --n_questions 5000 \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --passage_index data/passage_index \
    --cold_start_memory outputs/cold_start_memory/memory_store
```

**To detach from tmux** (leave it running in background): press `Ctrl+B`, then `D`.
**To reattach** after reconnecting via SSH: `tmux attach -t caem`
**To see running sessions**: `tmux ls`

### Monitoring progress

The experiment logs cycle-by-cycle to stdout. Inside tmux you can scroll back through the log. Alternatively, pipe to a file:

```bash
python scripts/run_experiment.py [args] 2>&1 | tee outputs/run.log
```

Then from another terminal (or VS Code), watch the log live:
```bash
tail -f outputs/full_run/run.log
```

### Checkpoint recovery (if the run crashes)

If the run stops at cycle N, resume without re-running previous cycles:
```bash
python scripts/run_experiment.py \
    --resume_from_cycle 4 \
    --output_dir outputs/full_run \
    [other args same as before]
```

---

## Part 7: Running Post-Experiment Scripts

After the main run finishes (or after each cycle group):

```bash
# Ablation study (~2–3 hours)
python scripts/run_ablation.py \
    --caem_results outputs/full_run/all_cycle_results.json \
    --cycle3_checkpoint outputs/full_run/cycle_3 \
    --output_dir outputs/ablation_results

# Purity validation (Chapter 5 Theory tables)
python scripts/run_purity_validation.py \
    --output_dir outputs/purity_validation

# Calibration (runs automatically in run_experiment.py, but can rerun standalone)
python scripts/run_calibration.py \
    --output_dir outputs/calibration
```

---

## Part 8: Downloading Results

Your key output files are:

| File | What it's for |
|---|---|
| `outputs/full_run/experiment_summary.csv` | Chapter 5 main results table |
| `outputs/full_run/all_cycle_results.json` | Full per-cycle per-benchmark results |
| `outputs/ablation_results/ablation_summary.json` | Ablation table |
| `outputs/purity_validation/theory_validation.json` | Theory 1/2/3 validation |
| `outputs/full_run/retroverify_cycle*.json` | Per-cycle forgetting scores + MMLU retention |

Download to your local machine with `scp`:
```bash
# Download all outputs at once (~2–5 GB)
scp -r -P 24501 root@123.45.67.89:~/caem/outputs/ "C:\Users\YourName\Desktop\caem_results\"
```

Or, in VS Code Remote SSH, just **right-click → Download** on the outputs folder in the file explorer.

**Stop the instance as soon as you have your files.** Idle instances still charge at the full hourly rate.

---

## Part 9: Can You Use Claude Code on Vast.ai?

**Yes.** Claude Code is an npm package that runs in any Linux terminal. Here's how to install it:

```bash
# Install Node.js if not present (vast.ai PyTorch images often don't have it)
apt-get update && apt-get install -y nodejs npm

# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Launch Claude Code in your project directory
cd ~/caem
claude
```

**Important caveats:**
- Claude Code usage is billed by Anthropic (API tokens) — not by vast.ai. These are two separate costs.
- You need your Anthropic API key. Get it from [console.anthropic.com](https://console.anthropic.com).
- Set the key in your tmux session before using it. For persistence across reconnects: `echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc && source ~/.bashrc`
- Claude Code on a remote Linux server is identical to using it locally — same commands, same file access.

**Practical use during the experiment:** Claude Code is most useful on vast.ai for debugging run failures, editing scripts mid-run, or inspecting output files. For the actual writing and planning work, it's better to do that locally (faster, no hourly billing pressure).

---

## Part 10: Cost and Time Estimates

For the full thesis experiment:

| Phase | Time | GPU cost (RTX 4090 @ $0.40/hr) |
|---|---|---|
| Environment setup + passage index build | ~45 min | ~$0.30 |
| Cold-start seeding (if not pre-seeded) | ~30 min | ~$0.20 |
| Full 10-cycle experiment (n=5000) | ~16–18 h | ~$6.50–7.50 |
| Ablation study | ~2–3 h | ~$1.00 |
| Purity validation | ~30 min | ~$0.20 |
| **Total** | **~20–23 h** | **~$8–10** |

**With an A100 40GB** (~$1.20/hr): same tasks take ~12–14h total → ~$15–18. Faster but more expensive.

---

## Part 11: Common Issues and Fixes

### "CUDA out of memory" during fine-tuning
The hardware.py auto-detects VRAM and sets batch size. If you still get OOM:
```bash
export CAEM_DEVICE=cuda
python scripts/run_experiment.py --n_questions 5000 [args]
```
If still failing, add `--smoke_test` to verify the pipeline works, then increase gradually.

### SSH connection times out mid-run
This is why you used tmux. Reconnect: `ssh -p PORT root@IP`, then `tmux attach -t caem`.

### HuggingFace download is slow or fails
Set the cache directory to your disk allocation:
```bash
export HF_HOME=/root/caem/hf_cache
export TRANSFORMERS_CACHE=/root/caem/hf_cache
```
Or download models before the run:
```bash
python -c "from transformers import AutoTokenizer, T5ForConditionalGeneration; AutoTokenizer.from_pretrained('google/flan-t5-large'); T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')"
```

### "No module named faiss"
```bash
pip install faiss-gpu   # try GPU version first
# If it fails:
pip install faiss-cpu
```

### Run crashed and you don't know which cycle it stopped at
```bash
ls outputs/full_run/ | grep cycle       # see which cycle checkpoints exist
ls outputs/full_run/eval/ | grep .json  # see which benchmark evals completed
```
Use the highest complete cycle number in `--resume_from_cycle`.

---

## Quick Reference Card

```bash
# 1. Connect
ssh -p PORT root@IP

# 2. Get code
git clone https://TOKEN@github.com/USER/REPO.git caem && cd caem

# 3. Upload data (from local machine)
scp -r -P PORT data/passage_index root@IP:~/caem/data/
scp -r -P PORT outputs/cold_start_memory root@IP:~/caem/outputs/

# 4. Install
pip install torch transformers datasets sentence-transformers faiss-gpu numpy scipy scikit-learn

# 5. Start experiment (inside tmux!)
tmux new-session -s caem
python scripts/run_experiment.py --output_dir outputs/full_run --num_cycles 10 --n_questions 5000 --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge --passage_index data/passage_index --cold_start_memory outputs/cold_start_memory/memory_store 2>&1 | tee outputs/run.log

# 6. Detach (leave running)
Ctrl+B, D

# 7. Reattach after reconnect
tmux attach -t caem

# 8. Download results (from local machine)
scp -r -P PORT root@IP:~/caem/outputs/ ./caem_results/

# 9. STOP THE INSTANCE to stop billing
```
