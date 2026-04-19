# CAEM — Next Session Plan (Vast.ai Runbook)

**Updated: 2026-04-19 | Phase 1a (Self-Funded) Line-by-Line Execution Plan**

This document is a **step-by-step runbook**. Read each numbered action,
execute it, verify the expected output, then move to the next. Do not
skip actions. Do not re-order. If an expected output is missing or
wrong, stop and diagnose — skipping a failing action compounds cost
downstream.

## Two-phase execution model

Phase 1 is split in two funding tranches:

**Phase 1a (self-funded, \$215 topup on a \$25 current balance, total
budget \$240):** Steps 4 → 15 + 19 → 20. Produces the headline result
package the supervisor needs to see: main 10-cycle CAEM run **at
Ch5-declared n=5000 SIL pool size** + 7 external baselines +
purity-theorem validation. Steps 16–18 (ablation sweep) are
**deferred**. Runtime estimate ~358 GPU-h = ~14 days wall-clock at
weighted 5.5 s/sample (Tier 1/2/3 mix growing across cycles). Cost
~\$230 with ~\$10 slack. The n=5000 value is what Ch5
§AblationMethodology line 751 declares for the confirmatory pass
including the reference run; reducing it would deviate from the
pre-registration.

**Phase 1 Full (supervisor-funded, ~$640 additional):**
Steps 16–18 only — 17-variant screening sweep + top-N confirmatory
runs. Runs on the same rental image after Phase 1a headline results
secure the funding. Populates the Chapter 5 ablation results table.
Runtime estimate ~1000 GPU-h.

**Terminology mapping to Chapter 5.** Chapter 5
§AblationMethodology uses "Phase 1" (single-seed, approximately
\$200) and "Phase 2" (multi-seed post-funding, approximately
\$700) as the \emph{statistical-power} partition of the ablation
program. The runbook's Phase 1a and Phase 1 Full are a
\emph{budget-funding} partition of the same work at single seed;
together they constitute Ch5's Phase 1. Ch5's Phase 2 (multi-seed
runs for variance estimation) is post-defense work not covered in
this runbook. The two terminologies are orthogonal and both are
retained: Ch5 uses funding-agnostic "Phase 1/2" because variance
estimation is the thesis-level concern; the runbook uses
"Phase 1a / Phase 1 Full" because funding acquisition is the
operational gate.

Defense sequence: Phase 1a complete → supervisor funding → Phase 1 Full
complete → defense prep → viva.

## Why the two-phase split

Ch5 §AblationMethodology line 747 originally projected "approximately
USD 200" for the Phase 1 screening sweep, assuming the pre-MiniCheck
verifier at ~4 s/sample. The MiniCheck swap adopted on 2026-04-19
(see VAST\_SESSION\_LOG.md Literature-review block Topic 1) doubles the
per-sample wall-clock to ~8 s, which doubles the screening sweep cost
to ~$400–$640 depending on the chosen n\_screen. The swap was
non-negotiable (the generic NLI backend falsifies the Chapter 4
purity theorem's α > 1/2 premise per HaluEval 2025), so the ablation
sweep gets deferred to the supervisor-funded tranche rather than the
MiniCheck swap being reverted. The headline story (Steps 7, 9–15, 19)
is unchanged and fits comfortably in Phase 1a.

---

## Active rental — Session 1 (2026-04-18)

Running session-specific state lives in `VAST_SESSION_LOG.md`; this block
captures the hardware profile so anyone reading this plan knows what
numbers the current Phase 1 results were produced on.

- **Instance ID:** 35180107 (datacenter 81036, machine 30024)
- **GPU:** NVIDIA GeForce RTX 5090, 31.8 GB VRAM, 108.1 TFLOPS, 1454.2 GB/s mem bandwidth, max CUDA 13.1
- **CPU:** AMD EPYC 9654 96-core (48 cores allocated), 96.7 GB RAM per Vast UI
- **Disk:** KIOXIA KCD8XRUG7T68 NVMe ~16 GB/s, 150.1 GB allocated
- **Net:** 250 ports, 4648.7 / 6801.6 Mbps
- **Motherboard:** GENOA2D24G-2L, PCIE 5.0 x16
- **Hourly cost:** ~$0.638/hr
- **DLPerf:** 203.1 (318.5 DLP/$/hr)

**Hardware-profile note:** The RTX 5090 auto-detects into CAEM's
`scripts/hardware.py` profile as bf16 / batch_size=32 / grad_accum=1 /
TF32-on, which is one tier faster than the RTX 4090 profile the rest of
this runbook was calibrated against. Wall-clock estimates in the
step-by-step table below remain useful but tend to be slightly
pessimistic on a 5090 for compute-bound steps.

Legend used throughout:

- **[ACTION]** — a thing you type or click.
- **[VERIFY]** — how to confirm the action worked before moving on.
- **[IF IT FAILS]** — most common failure mode and how to recover.
- **[WHY]** — one-sentence rationale so you know why the step exists.

For budget and Phase 2 context see `VAST_AI_DEPLOYMENT_GUIDE.md`
Part 10. For methodology see `writing-suggestions.md` Session 44
Addendum and Chapter 3 §3.5.

---

## Overview — where you are right now

- Code is committed and pushed to GitHub.
- Scripts in scope: `scripts/run_experiment.py`, `scripts/run_baseline.py`
  (B1–B5), `scripts/run_simple_ft.py` (B6, B7), `scripts/run_cyclic_ablation.py`,
  `scripts/aggregate_ablation.py`, `scripts/run_purity_validation.py`,
  `scripts/build_passage_index.py`, `scripts/seed_cold_start.py`.
- Ablation framework: **16 named variants** in `caem/ablation/variants.py`
  (14 mechanism + 2 inference-time). Legacy AB1–AB7 labels are retired.
- External baseline panel: **B1–B7** (inference B1–B5 via `run_baseline.py`,
  training B6–B7 via `run_simple_ft.py`). B8 Self-RAG is citation-only.
  STaR is an optional ceiling reference (Step 10B).
- Budget: ~USD 200 self-funded envelope for Phase 1, single seed = 42.
  Phase 2 (+USD 700, seeds 123 and 456) re-runs the same confirmatory
  commands with no code changes.
- Target GPU: **RTX 4090** on Vast.ai (~$0.40/hr). RTX 5090 is
  acceptable at ~$0.59/hr when a 4090 is not available. Upgrade to
  A100 SXM 80 GB only if wall-clock matters more than cost.
- Per-cycle recalibration (conservative default: `T` refit only,
  weights frozen) is **automatic** inside `run_experiment.py`. No
  extra CLI step; see the note in Step 6.

---

## At-a-glance execution order (updated 2026-04-19)

**Phase 1a** (this rental, self-funded **\$215 topup on \$25 balance,
total \$240**): ~358 GPU-h ≈ 14 days wall-clock at weighted
5.5 s/sample under MiniCheck (Tier 1/2/3 mix), ~\$230 cost with ~\$10
slack. **Phase 1 Full** (post-supervisor-funding): Steps 16–18 only,
~\$640 additional.

Status column legend: ✅ done · 🔄 in progress · ⏳ pending ·
🟡 deferred to Phase 1 Full.

| # | Step | Phase | Status | Wall-clock | Cost (5090 @ \$0.64/h) |
|---|------|-------|:---:|-----------:|----------------:|
| 1 | Pre-flight on local PC | 1a | ✅ | 5 min | \$0 |
| 2 | Rent + connect RTX 5090 | 1a | ✅ | 10 min | ~\$0.10 |
| 3 | Remote environment setup + HF model cache | 1a | ✅ | 15 min | ~\$0.15 |
| 3B | Pytest unit-test gate (95 + 9 MiniCheck tests) | 1a | ✅ | 2 min | ~\$0.02 |
| 4 | Build passage index (IndexFlatIP, 21M) | 1a | ✅ | ~8 h | ~\$5 |
| 4.5 | Archive FAISS index to HuggingFace Hub | 1a | ✅ | 10 min | ~\$0.10 |
| 5 | Smoke test (1 cycle, n=50, MiniCheck) | 1a | ✅ | 36 min | ~\$0.40 |
| 6 | Cold-start memory seeding (150 ep/bench @ τ=0.35) | 1a | 🔄 | ~2 h | ~\$1.3 |
| 7.0 | Cycle-0 eval + threshold calibration | 1a | ⏳ | ~1 h | ~\$0.6 |
| 5.5 | Verifier calibration diagnostic (MiniCheck vs RoBERTa) | 1a | ⏳ | 15 min | ~\$0.15 |
| 7 | **Main 10-cycle CAEM run** at n_questions=5000 (Ch5 declared) | 1a | ⏳ | ~335 h | **~\$215** |
| 8 | FLARE pre-flight smoke (5 samples) | 1a | ⏳ | 5 min | ~\$0.05 |
| 9 | B1 Zero-shot baseline | 1a | ⏳ | 20 min | ~\$0.20 |
| 10 | B2 Chain-of-Thought baseline | 1a | ⏳ | 50 min | ~\$0.55 |
| 11 | B3 DPR-RAG baseline | 1a | ⏳ | 1.7 h | ~\$1.1 |
| 12 | B4 CoT + DPR-RAG baseline | 1a | ⏳ | 2.1 h | ~\$1.3 |
| 13 | B5 FLARE baseline | 1a | ⏳ | 2.5 h | ~\$1.6 |
| 14 | B6 Vanilla FT (10 cycles) | 1a | ⏳ | ~5.8 h | ~\$3.7 |
| 15 | B7 EWC-only FT (10 cycles) | 1a | ⏳ | ~5.8 h | ~\$3.7 |
| 15.5 | McNemar + bootstrap CI + Holm sig-tests | 1a | ⏳ | 15 min | ~\$0.15 |
| 19 | Purity theorem validation | 1a | ⏳ | 30 min | ~\$0.30 |
| 19.2 | Cycle-2 retention diagnostic (advisory flag) | 1a | ⏳ | 10 min | ~\$0.10 |
| 19.5 | Nine-signal correlation matrix | 1a | ⏳ | 5 min | CPU only |
| 20 | Aggregate outputs + download + stop instance | 1a | ⏳ | 25 min | ~\$0.25 |
| **Phase 1a subtotal** | | | | **~358 h** | **~\$230** |
| 16 | Screening sweep (17 variants × 3 cycles × n_screen=1500) | Full | 🟡 | ~480 h | ~\$306 |
| 17 | Aggregate screening + pick top-N | Full | 🟡 | 5 min | ~\$0.05 |
| 18 | Confirmatory sweep (top-N + `full`, 10 cycles × n=5000) | Full | 🟡 | ~520 h | ~\$332 |
| **Phase 1 Full subtotal** | | | | **~1,000 h** | **~\$638** |
| 20B | **(Optional)** STaR ceiling run | 20B | 🟡 | 7–8 h | ~\$5 |

Progress today (2026-04-19 session): Steps 1, 2, 3, 3B, 4, 4.5, 5 ✅;
Step 6 🔄 (FEVER done, TriviaQA in progress, NQ pending).
Everything else ⏳ pending \$240 topup.

---

## Step 1 — Pre-flight (local PC, before renting)

**[WHY]** Avoid paying Vast rental for work you can do on your own
hardware.

1.1 **[ACTION]** Open a terminal at your repo root:

```bash
cd "C:\Users\aksan\Documents\for cowork caem"
```

1.2 **[ACTION]** Confirm working tree is clean and pushed:

```bash
git status
git log --oneline -5
git push origin main
```

**[VERIFY]** `git status` reports "nothing to commit, working tree clean"
and the last commit shows up on GitHub web UI.

1.3 **[ACTION]** Create a GitHub **personal access token** scoped to
`repo` (used by the remote instance for `git clone`):

- Browser: GitHub → Settings → Developer settings → Personal access
  tokens (classic) → Generate new token.
- Scope: tick `repo`.
- Save the token string locally — you will paste it into the Vast
  instance later. **It is shown only once.**

1.4 **[ACTION]** Verify Vast.ai account balance.

- Browser: <https://vast.ai> → Billing.
- Ensure credit ≥ **USD 75**. Top up if below.

1.5 **[ACTION]** Decide passage-index strategy and set the variable:

```bash
# Option A (recommended): build on the remote instance, no upload.
export PASSAGE_INDEX_STRATEGY=build
# Option B: download from Google Drive, requires pre-uploaded folder.
# export PASSAGE_INDEX_STRATEGY=gdrive
```

Uploading a 15–17 GB index from a residential connection is slow and
wasteful of rental time. Stick with Option A unless you already have
the index on Drive.

**[CHECKPOINT 1]** You are now ready to rent the instance.

---

## Step 2 — Rent + connect the RTX 4090 instance

**[WHY]** Every remote action below requires a running Vast instance.

2.1 **[ACTION]** Browser: <https://vast.ai> → top-left **Edit Image &
Config**.

- Choose template `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` (or
  the closest PyTorch 2.2 CUDA 12.1 image available).
- Allocated Disk Space slider = **100 GB** (the 16 GB default will
  crash the passage-index build).
- Click **Save**.

2.2 **[ACTION]** Filters on the left panel:

- **Verified** ticked.
- **1× GPU**, GPU type = **RTX 4090**.
- Reliability ≥ 99%.
- Sort ascending by price ($/hr).

2.3 **[ACTION]** Click **RENT** on the cheapest reliable instance.

2.4 **[VERIFY]** Wait until the instance row shows **Running** (not
"Scheduling", not "Loading"). Click **Connect** → the popup gives
you `ssh -p <PORT> root@<IP>`.

2.5 **[ACTION]** Copy that SSH command into a local shell and connect:

```bash
ssh -p <PORT> root@<IP>
```

The first connection may ask about host authenticity — answer `yes`.

**[IF IT FAILS]** "Connection refused" usually means the instance is
still booting; wait 30–60 s and retry. If the instance stays stuck in
"Scheduling" for > 5 min, destroy it and rent a different host.

**[CHECKPOINT 2]** You are at a root prompt on the remote RTX 4090.

---

## Step 3 — Remote environment setup + HF model cache

**[WHY]** Install dependencies once; pre-cache HuggingFace weights so
the long runs don't stall on mid-cycle downloads.

All commands below run **on the remote instance** unless otherwise
noted.

3.1 **[ACTION]** Clone your repo (replace placeholders):

```bash
export GH_TOKEN=<paste-token-from-step-1.3>
export GH_USER=<your-github-username>
export GH_REPO=<your-repo-name>
git clone https://${GH_TOKEN}@github.com/${GH_USER}/${GH_REPO}.git caem
cd ~/caem
```

**[VERIFY]** `ls` shows `caem/`, `scripts/`, `tests/`,
`pre thesis 1 report/`, etc.

3.2 **[ACTION]** Install Python deps:

```bash
# Default install (CPU FAISS): slow for Step 4 rebuilds -- k-means
# clustering on 21M x 768 embeddings takes ~80 min on CPU.
pip install torch transformers datasets "sentence-transformers<4" \
    faiss-gpu-cu12 numpy scipy scikit-learn

# If faiss-gpu-cu12 install fails (CUDA version mismatch on older Vast
# images), fall back to CPU. Every Step 4 rebuild then pays ~75 min
# extra for k-means on CPU.
# pip install torch transformers datasets "sentence-transformers<4" \
#     faiss-cpu numpy scipy scikit-learn
```

**[VERIFY]** No red errors; final line shows "Successfully installed".

**[WHY `faiss-gpu-cu12` over `faiss-cpu`]** On 2026-04-18 Session 1,
Step 4 k-means clustering with `faiss-cpu` on 2M x 768 training vectors
for 65536 IVF centroids took **~80 minutes** of wall-clock (memory-
bandwidth-bound on EPYC 9654, ~6 cores of useful parallelism before
hitting the RAM-channel ceiling). The GPU sat at 0% utilization the
entire time because `faiss-cpu` has no GPU path. `faiss-gpu-cu12` runs
the same IVF k-means on the 5090 in ~2-4 min -- a ~20x speedup on the
single slowest sub-step of the whole pipeline. Also cuts per-query
retrieval latency from ~10 ms to ~1 ms across Step 7 Tier 3 + Steps
9-13 RAG baselines (~20 min saved across those).

CUDA version pin: `faiss-gpu-cu12` matches CUDA 12.x on the
`pytorch/pytorch:2.2.0-cuda12.1` Vast image. For `pytorch:2.0.x-cuda11.7`
images (older), use `faiss-gpu-cu11` instead. If neither variant
installs cleanly, the `faiss-cpu` fallback is correctness-equivalent,
just slower. **Check which version installed after the pip call:**

```bash
python -c "import faiss; print('has gpu:', hasattr(faiss, 'StandardGpuResources'))"
```

If this prints `has gpu: True`, you got the GPU variant and Step 4
will be ~20x faster. If `False`, you're on CPU and Step 4 will take
~80 min in k-means alone -- budget for it.

**[WHY the `sentence-transformers<4` pin]** On 2026-04-18 Session 1
(`VAST_SESSION_LOG.md` Incident #1), sentence-transformers 5.4.1 failed
to import because it pulls in `torchcodec` transitively, which needs
`libnppicc.so.13` (CUDA NPP), missing from the `pytorch/pytorch:2.2.0`
Vast image. Pinning to `<4` (equivalent to 3.x) avoids the torchcodec
import chain entirely. CAEM only uses the
`SentenceTransformer(name, device=...).encode()` surface, which is
stable across 3.x–5.x.

3.2B **[ACTION]** Cap CPU threading BEFORE any FAISS call. Prevents the
thread-oversubscription collapse observed on 2026-04-18 Session 1
(`VAST_SESSION_LOG.md` entry at 22:50 UTC): on a 384-core EPYC Vast box
with `OMP_NUM_THREADS` unset, faiss-cpu spawned **379 threads** for IVF
k-means, yielding only **~5.6 effective cores** (1.5% utilization) due
to lock contention and context-switch overhead. Explicit thread caps
are essential even when using faiss-gpu, because scipy/numpy k-means
post-processing and LBFGS temperature scaling in Step 7 also use
OpenMP.

```bash
# Sweet spot for IVF k-means + BLAS-heavy inference workloads
# Don't exceed 32 on typical EPYC Vast rentals
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export FAISS_NUM_THREADS=16
```

For persistence across reconnects:

```bash
cat >> ~/.bashrc <<'EOF'
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export FAISS_NUM_THREADS=16
EOF
```

**[VERIFY]** After `source ~/.bashrc` (or reconnect), confirm the caps
took effect:

```bash
env | grep -E "OMP|MKL|OPENBLAS|FAISS" | sort
```

Should print all four env vars set to 16. If missing, re-source.

3.3 **[ACTION]** Confirm GPU is visible to PyTorch and the hardware
profile auto-detects:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "from scripts.hardware import print_hardware_summary; print_hardware_summary()"
```

**[VERIFY]** Expected output:

```
NVIDIA GeForce RTX 4090
GPU: NVIDIA GeForce RTX 4090
VRAM: 23.6 GB
Precision: fp16
Batch size: 16
```

If the VRAM number is very different (e.g. 10 GB), you rented the
wrong SKU. Stop the instance and re-rent.

3.4 **[ACTION]** Pre-cache HuggingFace models:

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

**[VERIFY]** Script prints `All models cached.` with no stack traces.

**[CHECKPOINT 3]** Environment ready.

---

## Step 3B — Pytest unit-test gate (mandatory before Step 4)

**[WHY]** Static AST validation and cross-file grep on local PC (Session
72) caught two real corruption bugs (truncated `run_experiment.py` tail,
UTF-8 BOM in the test file) and confirmed the `forgetting_score` →
`mmlu_retention_ratio` field rename is consistent across four files. But
the local PC cannot install torch, so runtime-gated code paths were not
exercised. A ~2-minute pytest pass on Vast closes that gap *before* the
2–3 h passage-index build and before any paid-GPU pipeline runs. If it
fails, you have lost <$0.05 instead of the $1 for the passage index or
the $0.14 for the smoke test.

**3B.1 [ACTION]** Install pytest (not in the main requirements):

```bash
pip install pytest --quiet
```

**3B.2 [ACTION]** Run the self-improvement test module:

```bash
cd ~/caem
python -m pytest tests/test_self_improvement.py -x --tb=short 2>&1 | tee tests_self_improvement.log
```

**[VERIFY]** All tests pass. The final line should read something like
`N passed in X.XXs`. Any failure — especially `AttributeError:
'CycleResult' object has no attribute 'forgetting_score'` — means a
call-site was missed in the rename and must be fixed before proceeding.

**[CHECKPOINT 3B]** Module-level invariants hold on the real
torch/transformers stack.

---

## Step 4 — Build the passage index

**[SHORTCUT if restoring on a new instance]** The 21M-passage
IndexFlatIP from the 2026-04-19 build is archived on HuggingFace Hub
(private dataset `aksaN000/caem-passage-index-21m`, ~72 GB). To skip
the 6-hour rebuild:

```bash
hf auth login   # paste read-scope token
mkdir -p data/passage_index
hf download aksaN000/caem-passage-index-21m --repo-type dataset \
    --local-dir data/passage_index
# verify
ls -lh data/passage_index/passages.faiss data/passage_index/passages.pkl
# expected: passages.faiss ~65 GB, passages.pkl ~7.5 GB
```

Expected download time: ~2-5 min on a typical Vast 4-5 Gbps uplink.
If download fails, fall back to the full rebuild below.

---



**[WHY]** Tier-3 RAG and the B3/B4/B5 baselines need a preprocessed
Wikipedia-passage FAISS index.

4.1 **[ACTION]** (Option A — build on the instance, recommended) Start
a tmux session and launch the build:

```bash
tmux new-session -s build
python scripts/build_passage_index.py \
    --max_passages 21000000 \
    --index_type flat_ip \
    --enable_checkpoint \
    --output_dir data/passage_index
```

**[WHY `--index_type flat_ip`]** The canonical 21M-passage RAG
configuration (DPR, Contriever, FiD) uses IndexFlatIP — not IVF-PQ.
Session 1 2026-04-18/19 spent ~15 hours stuck on IVF-PQ k-means due to
the nested OpenMP × BLAS thread-explosion bug in pip `faiss-cpu`
wheels (FAISS issue #3700 + #2477; `threadpoolctl.threadpool_info()`
confirms three duplicated OpenMP/BLAS runtimes that make
`OMP_NUM_THREADS` caps partially ineffective). FlatIP sidesteps this
entirely by skipping k-means. Build time: ~2-3 minutes vs ~15+ hours
for IVF-PQ. Full incident report in `VAST_SESSION_LOG.md` at
19:35 BDT on 2026-04-19. Trade-off: index size 61 GB (vs ~1.5 GB
for IVF-PQ) and query latency ~50-100ms (vs ~5ms for IVF-PQ).
Recall is 100% (ground truth). Memory footprint at runtime ~65 GB.

**[WHY `--enable_checkpoint`]** Writes `_checkpoint.pkl` every
`CHECKPOINT_EVERY=100_000` passages (~once every ~1.5 min on 5090).
Worst-case loss from an OOM / network disconnect / instance eviction
in the middle of the ~6 h encode drops from "start over" (~$4 + 6 h)
to "one buffer" (~100K passages, ~1.5 min, ~$0.02). Overhead is
~500 MB of pickled embedding buffers on disk, reclaimed when the
build finishes and `passages.faiss` + `passages.pkl` are written.

Session 1 (2026-04-18) did NOT pass this flag. When the FAISS
thread-oversubscription incident (`VAST_SESSION_LOG.md` 22:50 UTC)
surfaced, recovery was impossible because nothing was persisted --
the only options were "let the slow run finish" or "lose 6 h of
encoding." Always pass `--enable_checkpoint` on any run longer than
~30 min to guarantee a recovery path exists.

**[WHY `--train_sample_size 2000000`]** FAISS IVF clustering on
the default `nlist=65536` recommends 39x nlist = 2,555,904 training
vectors. 2M is the pragmatic minimum (30.5x nlist, 78% of ideal,
~1-3 pp recall loss vs. the ideal density, symmetric across CAEM
and all baselines so doesn't bias sig-test). Session 1 Incident #2
(`VAST_SESSION_LOG.md`) downgraded recall@10 by 5-15 pp with the
previous 500K default; patched default is now 2M, keep the explicit
flag for audit clarity.

4.2 **[ACTION]** Detach from tmux with `Ctrl+B, D`. Reattach at any
time with `tmux attach -t build`. If the build gets interrupted,
**resume** with:

```bash
tmux new-session -s build
python scripts/build_passage_index.py \
    --max_passages 21000000 \
    --train_sample_size 2000000 \
    --resume \
    --output_dir data/passage_index
```

(`--resume` implies `--enable_checkpoint` so future interruptions
continue to be recoverable.)

4.3 **[VERIFY]** Wait ~2–3 h. When done:

```bash
ls -lh data/passage_index/
```

Expected files: `passages.faiss`, `passages.pkl` (the `.faiss` is
several GB).

4.4 **[ACTION]** (Option B — Google Drive download; use only if you
have a pre-built index) Replace Step 4.1 with:

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/<FOLDER_ID>" \
    -O data/
```

**[IF IT FAILS]** "Disk full" → the disk slider wasn't set to 100 GB.
Destroy the instance, re-rent with Step 2 and 100 GB allocation.

**[CHECKPOINT 4]** Passage index exists on disk.

---

## Step 5 — Smoke test (1 cycle, n=50 per benchmark)

**[WHY]** Catch pipeline-integration bugs **before** paying for a
16–18 h main run.

**[VERIFIER BACKEND]** From 2026-04-19 onward the default verifier
backend is MiniCheck-Flan-T5-Large (`lytang/MiniCheck-Flan-T5-Large`),
not `roberta-large-mnli`. The swap is motivated by HaluEval 2025 /
Semantic Illusion 2025 (100% FPR at 95% recall for generic NLI on LM
hallucinations; see `VAST_SESSION_LOG.md` Literature-review block
Topic 1). To confirm:

```bash
python -c "from caem.config import CAEMConfig; \
  print('backend:', CAEMConfig().verifier_backend)"  # must print: minicheck
```

First Step 5 run on a fresh instance will download the MiniCheck
checkpoint (~1.5 GB bf16) into the HF cache. That download counts
toward smoke wall-clock; budget an extra ~2 min for it.

To fall back to the legacy RoBERTa-NLI backend (ablation only), add
`--verifier_backend roberta_nli` to any `run_experiment` /
`run_ablation` / `run_purity_validation` invocation, or set
`CAEMConfig.verifier_backend = "roberta_nli"`.

5.1 **[ACTION]** Run the smoke test in its own tmux session (~20 min):

```bash
tmux new-session -s smoke
python scripts/run_cyclic_ablation.py \
    --variant full \
    --seed 42 \
    --smoke_test \
    --passage_index data/passage_index \
    --output_dir outputs/smoke
```

5.2 **[VERIFY]** When the run completes:

```bash
ls outputs/smoke/full/seed_42/
cat outputs/smoke/full/seed_42/ces_axes_per_cycle.json
```

Expected: `ces_axes_per_cycle.json` exists with one cycle record and
a positive `ces` value (> 0).

**[IF IT FAILS]** If `ces = 0` or the JSON is missing, STOP. Re-read
the error tail from the tmux session (`tmux attach -t smoke`, scroll
back) and fix before proceeding. Wall-clock savings compound across
every later step.

**[CHECKPOINT 5]** Pipeline is end-to-end healthy.

---

## Step 5.5 — Verifier-backend calibration diagnostic (run after Step 7.0)

**[WHY]** The Ch4 purity theorem requires empirical α > ½ on the
LM-generated claim distribution. This step measures AUROC / ECE /
balanced-accuracy for both MiniCheck and legacy RoBERTa-MNLI on a
stratified 500-pair labelled set derived from the Cycle-0 eval
outputs, and ships the result into the Chapter 5 appendix as the
empirical justification for defaulting to MiniCheck.

**[ORDERING]** This step runs **after Step 7.0 Cycle-0 eval**, not
here in the sequential position of the runbook. The pair builder
needs the Cycle-0 eval JSONs (per-sample question / prediction / gold
/ em) to derive labelled (document, claim, label) triples.

5.5.1 **[ACTION]** Build the calibration pair set from Cycle-0 eval
outputs (one-time, ~3 min):

```bash
PYTHONPATH=. python scripts/build_calibration_pairs.py \
    --eval_jsons "outputs/cycle_0/eval/*_cycle0.json" \
    --passage_index data/passage_index \
    --n_pairs 500 \
    --balance 0.5 \
    --output_jsonl data/calibration/minicheck_pairs_500.jsonl
```

5.5.2 **[ACTION]** Run the head-to-head diagnostic (~8 min on 5090
for both backends):

```bash
PYTHONPATH=. python scripts/calibration_minicheck_vs_roberta.py \
    --pairs_jsonl data/calibration/minicheck_pairs_500.jsonl \
    --output_json outputs/calibration/minicheck_vs_roberta.json \
    --device cuda \
    --backends minicheck roberta_nli
```

5.5.3 **[VERIFY]** Expected outcomes:

- `outputs/calibration/minicheck_vs_roberta.json` exists with both
  `minicheck` and `roberta_nli` blocks.
- Literature expectation: MiniCheck AUROC +0.10 to +0.25 above
  RoBERTa-MNLI on this distribution.
- MiniCheck ECE < 0.15 (well calibrated); RoBERTa ECE likely
  0.25–0.45 (over-confident on LM outputs per HaluEval 2025).

5.5.4 **[GATE — three scenarios, all documented in Ch5 and
VAST\_SESSION\_LOG.md]**:

- **A. MiniCheck wins cleanly** (AUROC delta ≥ +0.05 AND ECE delta
  ≤ −0.10): proceed with MiniCheck default. No thesis edits needed.
- **B. Marginal win** (AUROC delta +0.02 to +0.05): proceed with
  MiniCheck default; soften Ch5 "10-25 AUROC advantage" to
  "measured advantage of +X on the CAEM-specific diagnostic".
- **C. Null or loss** (AUROC delta ≤ 0): revert default to
  `roberta_nli` via `CAEMConfig.verifier_backend`; re-frame Ch2 +
  Ch3 risk 6 + Ch4 empirical-scope remark + Ch5 §ExpSetup
  verifier-backend paragraph. Step 7 wall-clock halves (RoBERTa is
  2× faster than MiniCheck); Phase 1a budget drops from ~\$255 to
  ~\$130.

**[CHECKPOINT 5.5]** Verifier backend choice is empirically
defended; Ch4 Remark 4.3 (empirical scope of α) is auditable.

---

## Step 6 — Cold-start memory seeding

**[WHY]** Populate an initial FAISS episodic memory with ~200 verified
episodes so the first cycle of the main run has retrieval targets.

6.1 **[ACTION]** Run the seeding script:

```bash
python -m scripts.seed_cold_start \
    --target_episodes 200 \
    --benchmarks fever triviaqa natural_questions \
    --output_dir outputs/cold_start_memory
```

6.2 **[VERIFY]**

```bash
ls outputs/cold_start_memory/
cat outputs/cold_start_memory/seed_summary.json
```

Expected files: `memory_store.faiss`, `memory_store.meta`,
`seed_summary.json`. The JSON summary should report
`episodes_stored >= 180` (some skip-storage is normal due to the
novelty filter).

**[CHECKPOINT 6]** Cold-start memory ready.

---

## Step 7.0 — Cycle-0 baseline + threshold calibration (pre-Step-7)

**[WHY]** Before the main 10-cycle run commits, fit the decision-tree
thresholds (τ\_store, τ\_defer, τ\_train) from a Cycle-0 u\_stored
distribution at quantile targets P70 / P40 / P90. Avoids the failure
mode where RoBERTa-era defaults (0.65 / 0.45 / 0.75) produce 0% STORE
under MiniCheck. See Chapter 4 §Threshold calibration.

7.0.1 **[ACTION]** Run a small Cycle-0 eval (500 samples/benchmark)
to collect u\_stored values:

```bash
tmux new-session -s cycle0
python -m scripts.run_experiment \
    --output_dir outputs/cycle_0 \
    --num_cycles 0 \
    --n_questions 500 \
    --n_eval_questions 500 \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --passage_index data/passage_index \
    --cold_start_memory outputs/cold_start_memory/memory_store \
    2>&1 | tee outputs/cycle_0/run.log
```

7.0.2 **[ACTION]** Fit thresholds from the **n\_cal=500 calibration
fold** u\_stored values (Chapter 4 \S Threshold calibration; Chapter
5 \S Implementation Details requires this fold to be disjoint from
the n\_eval fold that Chapter 5 reports metrics on). The Cycle-0
calibration pass writes the per-sample records to
`outputs/cycle_0/calibration_fold_samples.json`:

```bash
PYTHONPATH=. python scripts/calibrate_thresholds.py \
    --calib_jsons outputs/cycle_0/calibration_fold_samples.json \
    --verifier_backend minicheck \
    --output_json outputs/cycle_0/calibrated_thresholds.json
```

The script's safety guard refuses any path containing `/eval/` or
ending `_eval.json`, so accidentally pointing it at the evaluation
fold fails loudly rather than silently peeking at the test set.

7.0.3 **[VERIFY]**

```bash
cat outputs/cycle_0/calibrated_thresholds.json
```

Expected: `thresholds.store` ≈ 0.35-0.50 under MiniCheck,
`thresholds.defer` ≈ 0.20-0.35, `thresholds.train` > store by ≥ 0.10.
The ordering assertion `train > store > defer` must hold; if it
fails, inspect the u\_stored distribution and raise target quantile
separation.

**[BENCHMARK-SIZE CAVEAT]** Chapter 5 declares n\_eval=500,
n\_cal=500, n\_purity=500, n\_MMLU=200 as disjoint splits. Three
benchmarks have strict limits that `split_calibration_sets` handles
by proportional scaling:

| Benchmark | Available split sizes | Fold allocation |
|---|---|---|
| FEVER | train ≈ 145k, dev ≈ 20k | full 500 for each fold |
| TriviaQA | train ≈ 87k, dev ≈ 11k | full 500 for each fold |
| Natural Questions | train ≈ 87k, dev ≈ 3k (dev subset) | may be n\_eval=≤3k / bench |
| TruthfulQA | single split 817 | n\_eval split ≤ 500; n\_cal and n\_purity not applicable (transfer-only, eval-only) |
| StrategyQA | train 2k, dev 229 | n\_eval ≤ 229; transfer-only |
| ARC-Challenge | train 1.1k, dev 299, test 1.2k | n\_eval ≤ 299; transfer-only |

Transfer-only benchmarks (TruthfulQA/StrategyQA/ARC) use the eval
split entirely for `n_eval`; they contribute ZERO samples to the
calibration or purity splits because there are no training splits to
draw from. This is fine because the calibration/purity splits already
cover the three in-training benchmarks (FEVER/TriviaQA/NQ) where
SIL memory population happens; the transfer benchmarks are just
held-out evaluation.

---

## Step 7 — Main 10-cycle CAEM run (Phase 1a headline)

**[WHY]** This is the headline Chapter 5 result, run at
Ch5 §AblationMethodology's declared n=5000 SIL pool size per
benchmark. Budget on MiniCheck-backend 5090: ~400 GPU-h, ~\$255.

7.1 **[ACTION]** First read the fitted thresholds from Step 7.0.2
into shell variables, then launch:

```bash
# Fitted thresholds + backend from Step 7.0 output
TAU_STORE=$(jq -r '.thresholds.store' outputs/cycle_0/calibrated_thresholds.json)
TAU_DEFER=$(jq -r '.thresholds.defer' outputs/cycle_0/calibrated_thresholds.json)
TAU_TRAIN=$(jq -r '.thresholds.train' outputs/cycle_0/calibrated_thresholds.json)
BACKEND=$(jq -r '.verifier_backend' outputs/cycle_0/calibrated_thresholds.json)
echo "Fitted: backend=$BACKEND  store=$TAU_STORE  defer=$TAU_DEFER  train=$TAU_TRAIN"

tmux new-session -s main
python -m scripts.run_experiment \
    --output_dir outputs/full_run \
    --num_cycles 10 \
    --n_questions 5000 \
    --n_eval_questions 500 \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --passage_index data/passage_index \
    --cold_start_memory outputs/cold_start_memory/memory_store \
    --verifier_backend "$BACKEND" \
    --store_threshold "$TAU_STORE" \
    --defer_threshold "$TAU_DEFER" \
    --train_threshold "$TAU_TRAIN" \
    2>&1 | tee outputs/full_run/run.log
```

The four override flags propagate the Cycle-0 fit into Cycles 1-10;
the ordering assertion inside `run_experiment.py` aborts cleanly if
`train > store > defer` is violated. Without these flags the run
would use RoBERTa-era CAEMConfig defaults (0.65 / 0.45 / 0.75) which
give near-zero STORE rate under MiniCheck, starving the
self-improvement loop.

7.2 **[ACTION]** Detach with `Ctrl+B, D`. Tail progress from a second
SSH session:

```bash
tail -f outputs/full_run/run.log
```

7.3 **[VERIFY — per cycle]** After each cycle, a new directory
`outputs/full_run/cycle_<N>/` appears, plus
`calibrated_config_cycle<N>.json` (see note below).

7.4 **[VERIFY — final]** When the run completes:

```bash
ls outputs/full_run/
cat outputs/full_run/experiment_summary.csv
```

Expected:
- `experiment_summary.csv` has **11 rows** (cycles 0 through 10).
- `retroverify_cycle*.json` files exist for cycles 1..10.
- The `mmlu_retention_pct` column is **≥ 93%** for every cycle.

7.5 **[NOTE — per-cycle recalibration (automatic)]** Each cycle ends
with an implicit calibration step before the next cycle starts: `T`
is re-fit on the disjoint calibration slice via LBFGS
(`scripts.run_calibration.calibrate_pipeline_temperature_only`), and
`outputs/full_run/calibrated_config_cycle<N>.json` is written. The
`u_stored` composite weights are fixed by design and are NOT re-fit.
Opt-out: pass `--skip_calibration` at the CLI (debug only; **do not**
do this for the headline run).

7.6 **[IF IT CRASHES]** Identify the last completed cycle and resume:

```bash
ls outputs/full_run/ | grep ^cycle_   # last N shown
python -m scripts.run_experiment --resume_from_cycle <N+1> ...(same args as 7.1)
```

7.7 **[NOTE — resume checkpoint completeness]** Each cycle N writes
four resume-critical artifacts: `memory_store_cycle_N.{faiss,meta}`
(episodic memory), `deferred_buffer_cycle_N.pkl` (deferred-decision
holds), `calibration/calibrated_config_cycle{N}.json` (per-cycle T),
and a model checkpoint under `model_checkpoint_cycle_N/`. The
`--resume_from_cycle N+1` path reloads all four so the next cycle
starts at the exact state the previous one ended at. Re-verify that
all four exist before destroying any instance intended to be resumed
later (see **Step 7S.2**).

**[CHECKPOINT 7]** Headline CAEM result produced.

---

## Step 7S — (OPTIONAL, skip by default) Split-session / 2-cycle validation run

**[DEFAULT: SKIP.]** Go straight from Step 5 smoke-test pass to the
Step 7 main run. Step 3B pytest + Step 5 smoke test are sufficient
validation before the 16–18 h, ~$7 headline run on a 4090.

**[USE 7S ONLY IF one of the following holds]:**

1. *Budget hedge on an expensive GPU.* Renting an RTX 5090 at
   ~$0.59/hr or an A100 SXM at ~$1.20/hr, where the full run is
   >$12 and spending $4–5 up front on a 2-cycle n=5000 validation
   catches silent numerical failures (degenerate fine-tuned weights,
   flat calibration, retention-floor violation) that the n=50 smoke
   test cannot see.
2. *Session-split insurance.* Your rental window is capped below
   ~10 h (e.g., spot-price interruption risk, hard Vast budget
   ceiling), so you need to run Cycles 0–2 in session one, ship the
   checkpoints home, then resume from Cycle 2 in session two. The
   `scripts/run_experiment.py` resume path (Session-81 fix) makes
   this seamless: memory store, deferred buffer, fine-tuned
   weights, and calibration `T` are all persisted per cycle and
   reloaded on `--resume_from_cycle`.

If neither condition holds, skip 7S entirely and follow Step 7 as-is.

**7S.1 [ACTION]** Launch a partial run with `--num_cycles 2` (session
one):

```bash
tmux new-session -s main
python -m scripts.run_experiment \
    --output_dir outputs/full_run \
    --num_cycles 2 \
    --n_questions 5000 \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --passage_index data/passage_index \
    --cold_start_memory outputs/cold_start_memory/memory_store \
    2>&1 | tee outputs/full_run/run.log
```

**7S.2 [VERIFY — before destroying the instance]** After Cycle 2
completes, confirm every resume-critical file exists:

```bash
ls -la outputs/full_run/memory_store_cycle_2.faiss \
       outputs/full_run/memory_store_cycle_2.meta \
       outputs/full_run/deferred_buffer_cycle_2.pkl \
       outputs/full_run/retroverify_cycle1.json \
       outputs/full_run/retroverify_cycle2.json \
       outputs/full_run/calibration/calibrated_config_cycle2.json
ls outputs/full_run/ | grep model_checkpoint_cycle_2 || echo "MISSING MODEL CKPT"
```

If any file is missing or zero-sized, **do not destroy** — inspect
`tail -200 outputs/full_run/run.log` for the cycle-2 write errors and
fix first. Destroying with an incomplete checkpoint set means rerunning
cycles 1–2 from scratch in session two.

**7S.3 [ACTION]** Download all resume-critical artifacts to local:

```bash
# From your local PC — Windows PowerShell / WSL / git-bash all work
mkdir -p "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast\partial_2cycle"
scp -r -P <PORT> root@<IP>:~/caem/outputs/full_run \
    "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast\partial_2cycle\"
scp -r -P <PORT> root@<IP>:~/caem/outputs/cold_start_memory \
    "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast\partial_2cycle\"
scp -r -P <PORT> root@<IP>:~/caem/data/passage_index \
    "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast\partial_2cycle\"
```

The passage index is the large one (~15–17 GB). Budget ~30–60 min
depending on your connection. If your home link is slow, consider
`gdrive` or `rclone` uploading to Google Drive from the instance
instead of direct scp.

**7S.4 [ACTION]** Destroy the instance — do **not** Stop:

Browser: Vast.ai instance card → **Destroy**.

**[WHY Destroy over Stop]** Stop keeps the disk (~$0.10–0.20/hr
storage rate still billed) and interruptible instances can be evicted
at any time, potentially losing work. Destroy ends all billing and
wipes the disk; since we already downloaded everything in 7S.3,
Destroy is strictly cheaper and carries no data-loss risk.

**7S.5 [ACTION — session two, on a new instance]** Rent a new
instance (repeat Steps 2 and 3: rent → SSH → clone repo → install
deps → HF-cache). Re-upload the preserved artifacts and launch the
resumed run:

```bash
# Step 1: re-upload (from local PC)
scp -r -P <NEW_PORT> "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast\partial_2cycle\full_run" \
    root@<NEW_IP>:~/caem/outputs/
scp -r -P <NEW_PORT> "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast\partial_2cycle\cold_start_memory" \
    root@<NEW_IP>:~/caem/outputs/
scp -r -P <NEW_PORT> "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast\partial_2cycle\passage_index" \
    root@<NEW_IP>:~/caem/data/

# Step 2: resume (on the new instance)
tmux new-session -s main
python -m scripts.run_experiment \
    --output_dir outputs/full_run \
    --num_cycles 10 \
    --resume_from_cycle 3 \
    --n_questions 5000 \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --passage_index data/passage_index \
    --cold_start_memory outputs/cold_start_memory/memory_store \
    2>&1 | tee -a outputs/full_run/run.log
```

**7S.6 [VERIFY — resume log lines]** The first ~30 seconds of the
resumed log must contain all four of:

```
Restored memory store from Cycle 2: <N> episodes
Restored fine-tuned model weights from Cycle 2.
Restored deferred buffer from Cycle 2: <M> entries.
Restored calibration T = <X.XXXX> from calibrated_config_cycle2.json
```

If any line is missing, the corresponding checkpoint did not upload
correctly — abort the run (`Ctrl+C`, then `tmux kill-session -t main`),
re-upload the missing artifact, and relaunch 7S.5 Step 2.

**[IF `Restored deferred buffer` warns "starting with empty buffer"]**
The `deferred_buffer_cycle_2.pkl` snapshot is absent or corrupt.
Acceptable: the buffer is small signal, not catastrophic to lose.
Proceed, but note it in the run log so the Chapter 5 analysis
acknowledges the gap.

**[IF `Restored calibration T` warns "keeping default T=1.0"]** The
`calibrated_config_cycle2.json` is absent. The next cycle-boundary
recalibration will overwrite T to the correct value, so at most one
cycle (cycle 3) runs with a slightly stale T. Proceed.

**7S.7 [NOTE]** The resumed run skips Cycle 0 (baseline eval + initial
calibration + cold-start seeding) because those are one-time. Total
Phase-1 cost for the split run is (session-one cost) + (session-two
cost for cycles 3–10, ~12–14 h on 4090 ≈ $5) + (one-time re-upload
overhead, ~$0.30 idle on 5090 or $0.20 on 4090). Typically adds only
$0.50–1.00 vs the single-session run.

**[CHECKPOINT 7S]** Split-strategy run produces identical
`experiment_summary.csv` to the single-session Step 7 run.

---

## Step 8 — FLARE pre-flight smoke test (mandatory before B5)

**[WHY]** `eval/baselines.py` had a decoder-slicing bug on T5
encoder–decoder models. Confirm the fix still lands before burning
~90 min on the full B5 run.

8.1 **[ACTION]** Run on 5 FEVER samples:

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
```

8.2 **[VERIFY]**

```bash
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

Expected: `empty_answer=0` **and** `escalated >= 1`.

**[IF IT FAILS]** If `empty_answer > 0` the decoder-slice regression
is back — inspect `eval/baselines.py` method `_look_ahead` and
confirm the offset is `+1` for T5. If `escalated == 0` across all 5,
`--flare_theta` may be too low for your cache; leave it at 0.4 and
re-check.

**[CHECKPOINT 8]** FLARE smoke passed; safe to run B5.

---

## Step 9 — B1 Zero-shot baseline

**[WHY]** The floor of the Chapter 5 external-baseline table. ~15 min.

9.1 **[ACTION]**

```bash
tmux new-session -s b1
python -m scripts.run_baseline \
    --baseline zero_shot \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 5000 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B1_zero_shot.log
```

> **n=5000 rationale (2026-04-18 session):** all B1–B7 baselines run at n=5000
> to match CAEM's `load_eval_transfer_pool` n, so the CAEM-vs-baseline McNemar
> test in `scripts/baseline_sig_tests.py` has ~500–5000 paired-sample power per
> benchmark (limited only by pool size for small benchmarks). Phase 1A decision.

9.2 **[VERIFY]**

```bash
python - <<'PY'
import json, glob
for f in sorted(glob.glob("outputs/baselines/zero_shot/*.json")):
    d = json.load(open(f)); agg = d.get("aggregate", {})
    print(f, "EM=", agg.get("em"), "F1=", agg.get("f1"))
PY
```

Expected EM on FEVER ≈ 55–60%, TriviaQA ≈ 40–45%. Large deviation =
prompt or loader bug.

---

## Step 10 — B2 Chain-of-Thought baseline

**[WHY]** Isolates the CoT contribution. ~20 min.

10.1 **[ACTION]**

```bash
python -m scripts.run_baseline \
    --baseline cot \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 5000 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B2_cot.log
```

10.2 **[VERIFY]** Re-run the sanity script from 9.2 targeted at
`outputs/baselines/cot/`.

---

## Step 11 — B3 DPR-RAG baseline

**[WHY]** Isolates external retrieval. ~45 min.

11.1 **[ACTION]**

```bash
python -m scripts.run_baseline \
    --baseline rag \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 5000 \
    --passage_index data/passage_index \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B3_rag.log
```

11.2 **[VERIFY]** Sanity script on `outputs/baselines/rag/`.

---

## Step 12 — B4 CoT + DPR-RAG baseline

**[WHY]** Combines CoT and retrieval. ~50 min.

12.1 **[ACTION]**

```bash
python -m scripts.run_baseline \
    --baseline cot_rag \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 5000 \
    --passage_index data/passage_index \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B4_cot_rag.log
```

12.2 **[VERIFY]** Sanity script on `outputs/baselines/cot_rag/`.

---

## Step 13 — B5 FLARE baseline

**[WHY]** Active retrieval reference. ~90 min. Runs only after Step 8
smoke test passed.

13.1 **[ACTION]**

```bash
python -m scripts.run_baseline \
    --baseline flare \
    --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_questions 5000 \
    --passage_index data/passage_index \
    --flare_theta 0.4 \
    --flare_look_ahead 64 \
    --output_dir outputs/baselines \
    2>&1 | tee outputs/baselines/B5_flare.log
```

13.2 **[VERIFY]** Sanity script on `outputs/baselines/flare/` and
spot-check that at least a few samples have `"escalated": true`.

---

## Step 14 — B6 Vanilla FT baseline (10 cycles)

**[WHY]** Fine-tuning baseline with **no** $L_2$ anchor and **no**
MMLU retention guard — isolates what the anchor + guard protect
against. ~6 h.

14.1 **[ACTION]**

```bash
tmux new-session -s b6
python -m scripts.run_simple_ft \
    --baseline_name vanilla_ft \
    --num_cycles 10 \
    --eval_benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_eval_per_bench 5000 \
    --n_train_per_bench 4000 \
    --output_dir outputs/baselines/vanilla_ft \
    2>&1 | tee outputs/baselines/B6_vanilla_ft.log
```

14.2 **[VERIFY]**

```bash
ls outputs/baselines/vanilla_ft/
cat outputs/baselines/vanilla_ft/training_log.jsonl | tail -n 20
```

Expected: 10 cycle records; MMLU often drops below 93% because the
guard is off — that is the whole point of this baseline.

---

## Step 15 — B7 EWC-only FT baseline (10 cycles)

**[WHY]** Anchor + guard on, everything else off. ~6 h.

15.1 **[ACTION]**

```bash
tmux new-session -s b7
python -m scripts.run_simple_ft \
    --baseline_name ewc_only_ft \
    --use_l2_anchor \
    --use_mmlu_guard \
    --num_cycles 10 \
    --eval_benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
    --n_eval_per_bench 5000 \
    --n_train_per_bench 4000 \
    --output_dir outputs/baselines/ewc_only_ft \
    2>&1 | tee outputs/baselines/B7_ewc_only_ft.log
```

15.2 **[VERIFY]** `ls outputs/baselines/ewc_only_ft/training_log.jsonl`
exists with 10 cycle records. MMLU retention should stay ≥ 93% in
every cycle; if the guard fires, the rollback is logged.

**[CHECKPOINT 9–15]** External baseline panel complete.

---

## ===== PHASE 1a ENDS HERE — STEPS 16–18 DEFERRED =====

**[BUDGET GATE]** Steps 16–18 below are the 17-variant ablation
sweep. They are **deferred** to Phase 1 Full (supervisor-funded
tranche) per the two-phase model declared at the top of this
runbook.

**When to return here:**
1. Phase 1a complete (Step 20 done, results downloaded locally).
2. Supervisor approves Phase 1 Full funding (~\$640 GPU-credit).
3. Resume a Vast rental (same image, pull FAISS from HF per Step 4
   shortcut).
4. Execute Steps 16–18 below, then re-run Step 20 with the
   augmented output set.

**If Phase 1a satisfies your defense requirement:** jump directly to
Step 19 (purity theorem validation) and Step 20 (aggregate + stop).

---

## Step 16 — Phase 1 Full screening sweep (17 variants × 3 cycles × n=1500)

**[WHY]** Rank the 17 ablation variants so confirmatory budget goes
only to the high-impact ones. **Screening rows never enter the
Chapter 5 ablation table** — this is a budget-allocation instrument.
~14–20 h total under RoBERTa per-sample time; ~480 h under MiniCheck
8 s/sample. Budget for this step under MiniCheck: ~\$306.

Variant set (17, matching `caem/ablation/variants.py`):
reference `full`; verification family `no_verifier`, `no_grounding`,
`no_contradiction_veto`, `no_store_gate`, `aggressive_store`,
**`roberta_nli_backend`** (added 2026-04-19 for the MiniCheck-backend
swap ablation); calibration family `no_internal_calibration`,
`no_semantic_entropy`, `equal_signal_weights`; safety-gate
`no_early_exit`; self-improvement `no_self_improvement`,
`no_retroverify`; routing `no_tier1`; retrieval `no_tier3_rag`,
`no_novelty_filter`, `no_recency_decay`.

16.1 **[ACTION]** Write the wrapper script:

```bash
cat > run_screening.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
VARIANTS=(
    full no_verifier no_grounding no_contradiction_veto no_store_gate
    aggressive_store roberta_nli_backend no_internal_calibration
    no_semantic_entropy equal_signal_weights no_early_exit
    no_self_improvement no_retroverify no_tier1 no_tier3_rag
    no_novelty_filter no_recency_decay
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
```

16.2 **[ACTION]** Launch in tmux:

```bash
tmux new-session -s screening
./run_screening.sh
```

16.3 **[VERIFY]** Every 45–75 min, a new variant directory appears
under `outputs/ablation/`:

```bash
ls outputs/ablation/
```

After all 16 variants complete (~14–20 h), verify each has a
`screening` subfolder with a non-empty `ces_axes_per_cycle.json`.

**[IF IT FAILS ON ONE VARIANT]** The wrapper will abort because of
`set -euo pipefail`. Edit `run_screening.sh` to restart from the
failing variant, then re-run. You do **not** need to re-run
completed variants.

---

## Step 17 — Aggregate screening + pick top-N

**[WHY]** Rank variants by `|ΔCES|` vs. `full` to select Step 18
candidates. ~5 min.

17.1 **[ACTION]**

```bash
python scripts/aggregate_ablation.py \
    --output_dir outputs/ablation \
    --screening_mode
cat outputs/ablation/ablation_table.csv
```

17.2 **[VERIFY]** Output shows 16 rows, sorted by CES ascending. The
reference row `full` should be at or near the top (highest CES).

17.3 **[ACTION]** Pick the top-N by `|ΔCES|` (N = 5–7 typically).
Record the selected variants in `caem-implementation-log.md` — this
is the **pre-registration step** for the Chapter 5 ablation table.

**[CHECKPOINT 17]** Confirmatory candidate list frozen.

---

## Step 18 — Phase 1 confirmatory sweep (top-N + `full`, 10 cycles × n=5000)

**[WHY]** Produce the Chapter 5 ablation table rows. ~50–60 h total
for N = 5–6 variants. Longest step in Phase 1.

18.1 **[ACTION]** Write the wrapper, editing the variant list to match
your Step 17 selection:

```bash
cat > run_confirmatory.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
# EDIT THIS LIST from your Step 17 selection
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
```

18.2 **[ACTION]** Launch in tmux:

```bash
tmux new-session -s confirmatory
./run_confirmatory.sh
```

18.3 **[ACTION]** Because this runs 50–60 h, expect to **stop and
resume** the Vast instance across multiple days. Vast charges idle
time, so use the **Stop** button on the instance card between
sessions, not just disconnect. Resume with **Start**. Tmux will
survive instance stop/start as long as the instance is not
destroyed.

18.4 **[VERIFY]** After each variant finishes (~8–10 h):

```bash
ls outputs/ablation/<variant>/seed_42/
cat outputs/ablation/<variant>/seed_42/experiment_summary.csv | head
```

Expected: 11-row summary CSV, MMLU retention ≥ 93% for `full`
(retention floor may be violated intentionally by some variants).

**[CHECKPOINT 18]** Ablation sweep complete.

---

## Step 19 — Purity theorem validation

**[WHY]** Produces Chapter 5 §5.3 purity tables. ~30 min.

19.1 **[ACTION]**

```bash
python scripts/run_purity_validation.py \
    --output_dir outputs/purity_validation
```

19.2 **[VERIFY]**

```bash
ls outputs/purity_validation/
cat outputs/purity_validation/theory_validation.json
```

Expected: `theory_validation.json` with observed vs. predicted purity
across all 6 benchmarks.

---

## Step 19.2 — Cycle-2 retention diagnostic (created pre-run in Step 7)

**[WHY]** Before Cycle 1 starts, freeze a 500-sample Cycle-0 slice.
After Cycle 2 completes, evaluate and check the three-percent absolute
EM drop tripwire. See Ch4 §Cycle-2 retention diagnostic. ~10 min total.

19.2.1 **[ACTION — after Step 7.0 eval, before Cycle 1]**

```bash
PYTHONPATH=. python scripts/cycle2_retention_diagnostic.py --make_slice \
    --cycle0_eval_dir outputs/cycle_0/eval \
    --output_slice data/retention/cycle0_slice_500.jsonl
```

19.2.2 **[ACTION — after Cycle 2 completes, as part of Step 19]**

```bash
PYTHONPATH=. python scripts/cycle2_retention_diagnostic.py --evaluate \
    --slice data/retention/cycle0_slice_500.jsonl \
    --cycle2_checkpoint outputs/full_run/cycle_2/model \
    --output_report outputs/full_run/cycle_2/retention_diagnostic.json
```

19.2.3 **[GATE]** If the report's `advisory` field is
`STRUCTURED_FALLBACK_TO_OLORA`, the remaining cycles 3-9 should be
re-run under O-LoRA rank-16 adapters merged every 2 cycles. See Ch4
"Cycle-2 retention diagnostic and structured fallback" paragraph.

---

## Step 19.5 — Nine-signal correlation matrix

**[WHY]** Empirically grounds the Ch5 §ExpectedResults
"Nine-signal correlation structure and effective-count analysis"
pre-registration. ~5 min, CPU-only.

19.5.1 **[ACTION]**

```bash
PYTHONPATH=. python scripts/signal_correlation_matrix.py \
    --eval_jsons "outputs/full_run/cycle_0/eval/*_cycle0.json" \
    --output_dir outputs/signal_correlation
```

19.5.2 **[VERIFY]**

```bash
cat outputs/signal_correlation/redundancy_report.json
```

Expected: `effective_signal_count_after_prune` in range 5-8 (after
|ρ|>0.9 pruning), three pre-registered clusters (token-prob
mean/min; SC / SE; NLI / SC if shared samples) probably collapse.

---

## Step 20 — Aggregate Phase 1 outputs + download + stop

**[WHY]** Consolidate artifacts, pull them locally, and halt billing.

20.1 **[ACTION — on the remote instance]** Final aggregate:

```bash
python scripts/aggregate_ablation.py --output_dir outputs/ablation
```

**[VERIFY]** The following files now exist and are non-empty:
- `outputs/ablation/ablation_table.csv` (headline ablation)
- `outputs/ablation/ablation_per_cycle.csv` (per-cycle CES traces)
- `outputs/ablation/ablation_aggregate_manifest.json` (provenance)

20.2 **[ACTION — on your local PC]** Pull everything down:

```bash
mkdir -p "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast"
scp -r -P <PORT> root@<IP>:~/caem/outputs/ \
    "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast\"
```

20.3 **[VERIFY — locally]**

```bash
ls "C:\Users\aksan\Documents\for cowork caem\outputs_from_vast\outputs\"
```

Expected sub-dirs: `full_run/`, `baselines/`, `ablation/`,
`purity_validation/`, `cold_start_memory/`.

20.4 **[ACTION]** Browser: Vast.ai instance card → **Stop**
(**not** "Destroy" unless Phase 2 is not planned). Idle instances
continue to bill.

**[CHECKPOINT 20]** Phase 1 complete; instance stopped.

---

## Step 20B — Optional STaR ceiling run (gated on budget)

**[WHY]** A numerical STaR row lets Chapter 5 claim that "CAEM's gap
to STaR isolates what the verifier + memory add on top of pure
iterative refinement." Skip if Phase 1 is already near the USD 200
envelope.

20B.1 **[GATE]** Only run if:
- Everything in Steps 1–20 is complete.
- Vast spend so far is ≤ USD 197 (≥ USD 3 headroom).

20B.2 **[ACTION]**

```bash
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

20B.3 **[VERIFY]**

```bash
grep '"use_rationalisation"' outputs/baselines/star/training_log.jsonl | head
```

Expected: every cycle record has `"use_rationalisation": true`.
Per-cycle EM on in-domain benchmarks (FEVER, TriviaQA, NQ) should be
monotonically non-decreasing; if it is not, rationalisation quality
is too weak and the fine-tune is degrading the model. Record that
finding in the impl log (benchmark-mismatch caveat in Ch2 §2.2).

20B.4 **[ACTION]** Re-run Step 20.1 (aggregate) and 20.2 (download)
to include the STaR row.

---

## After Phase 1 — writeup (local, no Vast required)

All steps below run on your local PC against
`outputs_from_vast/outputs/...`.

W.1 Populate Chapter 5 §5.5 ablation table from
`outputs_from_vast/outputs/ablation/ablation_table.csv`.

W.2 Add the single-seed limitation paragraph (wording in
`writing-suggestions.md` S44-03) under §5.5.

W.3 Populate Chapter 5 §5.1 headline tables from
`outputs_from_vast/outputs/full_run/experiment_summary.csv`.

W.4 Populate Chapter 5 §5.1 external-baseline comparison rows from
`outputs_from_vast/outputs/baselines/*/` (one row per B1–B7, plus
STaR if Step 20B ran).

W.5 Populate Chapter 5 §5.3 purity tables from
`outputs_from_vast/outputs/purity_validation/theory_validation.json`.

W.6 `pdflatex → biber → pdflatex → pdflatex` on `main.tex`; spot-check
that the ablation table references resolve (no `??` placeholders).

---

## Phase 2 — post-funding upgrade (plan only; do not execute yet)

Once ~USD 700 funding is secured, Phase 2 re-runs the confirmatory
sweep on two additional seeds (123, 456) across **all 14 cyclic
variants + reference** (14 mechanism; the 2 inference-time variants
do not touch training state so are not re-run). Expected compute:
~300–370 h on RTX 4090 (~USD 120–180), or ~180–230 h on A100 SXM
80 GB (~USD 400–620).

Phase 2 execution is the same `scripts/run_cyclic_ablation.py`
invocation pattern as Step 18, just with `--seed 123` and
`--seed 456` in place of `--seed 42`, and the full 14-variant cyclic
list instead of the Phase 1 top-N. The aggregator auto-switches from
point estimate to mean±std reporting once the second seed's outputs
exist under `outputs/ablation/<variant>/seed_<N>/`. Chapter 5 §5.5
tables regenerate by re-running `aggregate_ablation.py` with no CLI
changes.

---

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| CUDA OOM during fine-tuning | Batch size too high | Override `batch_size` in `caem/config.py`; re-run |
| `FileNotFoundError: passage_index` | Didn't build (Step 4) or not uploaded | Return to Step 4 |
| Screening rows missing for some variants | One variant crashed mid-loop | Edit `run_screening.sh` to restart from failing variant, re-aggregate |
| `ces = 0` in aggregator output | Axis collapse (CAL or VER = 0) — expected for extreme variants like `no_verification` | Inspect per-axis values in `ces_axes_per_cycle.json`; not a bug |
| `mmlu_retention_pct < 93%` on `full` | Forgetting abort should have fired | Check `retroverify_cycle*.json` for abort flag; if not set, investigate |
| `calibrated_config_cycle<N>.json` missing | `--skip_calibration` was passed | Re-run without `--skip_calibration` |
| FLARE smoke returns `empty_answer > 0` | Decoder-slice regression | Inspect `eval/baselines.py::_look_ahead`, confirm T5 offset = 1 |
| Instance stuck "Scheduling" > 5 min | Vast host issue | Destroy; rent a different host |

---

## Phase 1 success criteria — the five greens

Phase 1 is **done** when all five of these are true:

- [ ] Step 7 main run: `experiment_summary.csv` has 11 rows; every
  cycle's `mmlu_retention_pct ≥ 93%`.
- [ ] Steps 9–15 baselines: `outputs/baselines/<name>/` exists for
  each of B1, B2, B3, B4, B5, B6, B7; sanity-script numbers are
  within expected ranges.
- [ ] Step 17 screening: `ablation_table.csv` lists all 16 variants;
  top-N selection recorded in `caem-implementation-log.md`.
- [ ] Step 18 confirmatory: `ablation_table.csv` lists `full` + top-N
  in the 10-cycle mode; per-cycle CSVs populated.
- [ ] Step 19 purity: `theory_validation.json` populated across all
  6 benchmarks.

When all five are green, Chapter 5 can be drafted from point-estimate
tables while Phase 2 funding is being secured.
