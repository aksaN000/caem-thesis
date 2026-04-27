# CAEM Phase-1a — Vast Machine Resume Guide

**Written:** 2026-04-27 17:03 UTC, just before destroying current Vast 5090 instance.
**Author handoff to:** future-Aksan, when renting the next Vast machine.

---

## 0. State at moment of destruction

- **Repository:** `https://github.com/aksaN000/caem-thesis`
- **Branch:** `feat/qwen-3b-goal1`
- **Latest commit:** `f249682` (thesis: ten-signal → nine-signal sweep across all chapters)
- **Cycle progress:** Cycle 0 complete (locked artefacts on disk + on gdrive). Cycle 1 was mid Step 2.1 cal-fold scoring (FEVER ~128/500 samples) when the runner was halted at 17:03 UTC. The cycle-1 SIL fine-tune (`model.pt`) had completed at 16:45:53 UTC and was offloaded to gdrive — but the cycle-1 boundary work (Steps 2.2–2.5 + cycle-1 stream + cycle-1 eval-fold) never finished.
- **Resume strategy:** clean redo of cycle 1 from scratch (move incomplete `cycle_1/` aside, restart). The cycle-1 SIL fine-tune is fast (~10 min); losing it costs nothing material.

---

## 1. What is preserved off-machine

These survive the Vast destruction and are required for resume:

### 1.1 GitHub (`feat/qwen-3b-goal1` branch)
- All code patches (h_norm retirement, u_pre cache fast-path, OOM mitigations)
- All thesis sources
- `run_phase1a.sh` runbook
- `RESUME_GUIDE.md` (this file)

### 1.2 Hugging Face (`aksaN000/caem-passage-index-21m`, private)
- `data/passage_index/` — 64 GB FAISS DPR index (used by Tier-3 retrieval)
- `pre_main_snapshot/` — cycle-0 artefacts immediately before Step 7 main launch (created via `step_hf_upload_pre_main`)

### 1.3 Google Drive (`gdrive:caem-phase1a/full_run/`)
- `cycle_0/model.pt` — pristine post-Step-7.0 model
- `cycle_1/model.pt` — post-cycle-1 SIL fine-tune (the one offloaded at 16:45:53 today). Optional to restore; if you skip cycle 1 and start fresh, you don't need it.

### 1.4 Local repo files (committed to git, no separate backup needed)
- `outputs/cycle_0/composite_calibration.json` — locked 9-signal cal-prob composite (h_norm weight = −5×10⁻⁴, treated as retired)
- `outputs/cycle_0/conformal_gate.json` — τ_store=0.6676, τ_defer=0.5207, α_store=0.05, α_defer=0.40
- `outputs/cycle_0/calibrated_thresholds.json` — legacy quantile thresholds (used by `run_phase1a.sh` to set CLI flags)
- `outputs/cycle_0/eval_rescored/*.json` — cycle-0 baseline evals (7 benchmarks)
- `outputs/cold_start_memory/memory_store.faiss + .meta` — 137-entry seed memory
- `outputs/cycle_0/calibration/calibration_fold_samples.json` — 1500-sample cal-fold (3 benchmarks × 500)

> ⚠️ **VERIFY before destroying:** every file in §1.4 must be committed to git. If any is in `.gitignore` or oversize, copy to gdrive before destroying.

### 1.5 NOT preserved
- `outputs/full_run/cycle_1/` (incomplete cycle 1 — discard, will redo)
- `outputs/full_run/eval/` (only cycle_0 entries — those are also in `outputs/cycle_0/eval_rescored/`)
- `outputs/_halted_runs/` (history; not needed for resume)
- Pip caches, conda envs, model weight caches (Vast `~/.cache`)

---

## 2. Pre-destruction checklist (do these BEFORE you click destroy on Vast)

```bash
# 1. Halt runner cleanly (already done at 17:03 UTC — verify no zombie)
tmux ls         # should show no plan_a, or:
tmux kill-server

# 2. Verify cycle_1/model.pt is on gdrive (already confirmed at 16:45:53)
rclone ls gdrive:caem-phase1a/full_run/cycle_1/

# 3. Push any uncommitted changes
cd /workspace/caem
git status
git add -A
git commit -m "pre-destruction snapshot"   # if needed
git push

# 4. Confirm HF snapshot is current (passage index + pre-main snapshot)
huggingface-cli login
huggingface-cli download aksaN000/caem-passage-index-21m --local-dir /tmp/_verify_hf
ls /tmp/_verify_hf/data/passage_index/   # should contain *.faiss + *.json + *.tsv

# 5. (Optional) tar + upload cycle_0/* to gdrive as a hard backup
tar czf /tmp/cycle_0_backup.tar.gz outputs/cycle_0/
rclone copy /tmp/cycle_0_backup.tar.gz gdrive:caem-phase1a/backups/

# 6. Note your tmux history if anything important
tmux capture-pane -t plan_a -p > /tmp/plan_a_final_pane.txt
rclone copy /tmp/plan_a_final_pane.txt gdrive:caem-phase1a/logs/

# 7. Destroy the Vast instance from the dashboard.
```

---

## 3. Resume on new Vast machine

### 3.1 Spin up the machine
- **GPU:** RTX 5090 (32 GiB VRAM) — same as before. The OOM-mitigation patches (force_eager, bf16 anchor on CPU, batch_size=4, expandable_segments) were tuned to this envelope.
- **Disk:** ≥200 GB. Passage index alone is 64 GB; model weights, gdrive cache, outputs add ~100 GB.
- **Image:** any with CUDA 12.1+ and Python 3.10+. The repo's `run_phase1a.sh` activates `/venv/main` if present.

### 3.2 Clone repo and check out the branch
```bash
cd /workspace
git clone https://github.com/aksaN000/caem-thesis.git caem
cd caem
git checkout feat/qwen-3b-goal1
git log -1 --oneline    # confirm you see f249682 or later
```

### 3.3 Set up Python environment
```bash
# If the Vast image has no venv:
python3 -m venv /venv/main
source /venv/main/bin/activate

# Install dependencies (the repo's requirements live in scripts/setup or
# pyproject — adjust to whichever is current):
pip install -r requirements.txt   # if exists
# OR per-package:
pip install torch transformers sentence-transformers faiss-gpu \
            scikit-learn scipy numpy pandas accelerate bitsandbytes \
            flan-t5 peft datasets huggingface-hub rclone
```

### 3.4 Authenticate to gdrive + HF
```bash
# rclone for gdrive (interactive — paste token from another browser tab)
rclone config
# Test: rclone ls gdrive:caem-phase1a/

# HuggingFace
huggingface-cli login   # paste write-token from huggingface.co/settings/tokens
```

### 3.5 Download data dependencies
```bash
# A) Passage index (64 GB, takes ~30 min on a good link)
huggingface-cli download aksaN000/caem-passage-index-21m \
    --local-dir data/passage_index/ \
    --include "data/passage_index/*"
# OR if HF stored it flat:
huggingface-cli download aksaN000/caem-passage-index-21m --local-dir /tmp/_hf
mkdir -p data/passage_index
mv /tmp/_hf/data/passage_index/* data/passage_index/

# B) Cold-start memory (this is committed to git already, verify presence)
ls outputs/cold_start_memory/memory_store.faiss   # ~30 MB
ls outputs/cold_start_memory/memory_store.meta

# C) Cycle 0 artefacts (committed to git, verify presence)
ls outputs/cycle_0/composite_calibration.json
ls outputs/cycle_0/conformal_gate.json
ls outputs/cycle_0/calibrated_thresholds.json
ls outputs/cycle_0/calibration/calibration_fold_samples.json
ls outputs/cycle_0/eval_rescored/*.json | wc -l   # should be 7

# D) Cycle 0 model.pt from gdrive (NOT in git — too large)
mkdir -p outputs/full_run/cycle_0
rclone copy gdrive:caem-phase1a/full_run/cycle_0/model.pt outputs/full_run/cycle_0/
ls -lh outputs/full_run/cycle_0/model.pt   # ~6.17 GiB

# E) (OPTIONAL) Cycle 1 model.pt from gdrive — only if you want to resume
#    mid-cycle-1. We recommend SKIPPING this and redoing cycle 1 fresh
#    (~10 min SIL re-train). Cleaner state.
# rclone copy gdrive:caem-phase1a/full_run/cycle_1/model.pt outputs/full_run/cycle_1/
```

### 3.6 Verify environment
```bash
# Quick sanity check:
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from caem.config import CAEMConfig; c=CAEMConfig(); print('disable_h_norm:', c.disable_h_norm)"   # should print True
python -c "from caem.verification.verifier import UnifiedVerifier; print('verifier OK')"
nvidia-smi --query-gpu=memory.total,name --format=csv,noheader   # should show 32GB 5090

# Run the unit test suite (skip slow integration tests)
pytest tests/ -x -q --ignore=tests/test_integration.py 2>&1 | tail -20
```

### 3.7 Decide resume strategy

**Strategy A (recommended):** Clean redo of cycle 1.
- `outputs/full_run/cycle_*` should contain ONLY `cycle_0/`. If a stale `cycle_1/` is present, move it aside:
  ```bash
  mkdir -p outputs/_halted_runs
  mv outputs/full_run/cycle_1 outputs/_halted_runs/cycle_1.preexisting_$(date -u +%Y%m%d_%H%M)
  ```
- Resume detection in `run_phase1a.sh` will see only `cycle_0/` and start cycle 1 from scratch (SIL.run_cycle(1) ~10 min, then full Step 2.1–2.5, then cycle-1 stream).

**Strategy B:** Resume mid-cycle-1 (skip the SIL re-train).
- Restore `cycle_1/model.pt` from gdrive (§3.5 step E).
- The resume guard in `run_experiment.py:1349` will fail because `eval/fever_cycle1.json` is missing. You'd need to also pass `--skip_calibration` or hand-build the eval JSONs. **Not recommended** — the 10 min SIL re-train is cheaper than this surgery.

### 3.8 Relaunch the run
```bash
cd /workspace/caem
chmod +x run_phase1a.sh

# Verify the resume detection picks up cycle_0 correctly:
bash -c 'last=$(ls -d outputs/full_run/cycle_* 2>/dev/null | sed "s#.*cycle_##" | sort -n | tail -1); echo "Will resume from cycle $((last + 1))"'
# Expected output: "Will resume from cycle 1"

# Launch in tmux:
tmux new-session -d -s plan_a -x 220 -y 60 './run_phase1a.sh 2>&1 | tee -a outputs/phase1a_runner.log'
tmux ls
sleep 5
tail -20 outputs/phase1a_runner.log
```

### 3.9 First-hour validation (catch problems early)
```bash
# After ~5 min, check the runner is past env setup:
grep "Step 7" outputs/phase1a_runner.log | tail -3
grep "detected completed cycle" outputs/phase1a_runner.log | tail -3

# After ~15 min, cycle-1 SIL should be done:
grep "Cycle 1: gdrive offload OK" outputs/full_run/run.log
grep "MMLU retention" outputs/full_run/run.log | tail -3

# After ~30 min, Step 2.1 cal-fold should be in progress:
grep "Step 2.1" outputs/full_run/run.log
grep "verify_batch N=" outputs/full_run/run.log | tail -3

# Confirm h_norm retirement is active (these should NOT log K-pool overhead):
grep "pool(m+se)=" outputs/full_run/run.log | tail -5
# Old (K=10) pool(m+se) was ~14000-30000 ms.
# New (K=0) pool(m+se) should be <5000 ms (just M=3 chains).

# Confirm u_pre cache fast-path will fire on Step 2.2 (after Step 2.1 finishes):
python -c "import json; d=json.load(open('outputs/full_run/cycle_1/calibration/fever_cycle1.json')); print('u_pre present:', 'u_pre' in d['samples'][0]); print('sample[0].u_pre =', d['samples'][0].get('u_pre'))"
# Expected: u_pre present: True, sample[0].u_pre = <some float in [0,1]>
```

---

## 4. Expected timeline post-resume

| Phase | Wall-clock | Cumulative |
|------|-----------:|-----------:|
| Vast spin-up + dependency install | ~45 min | 0:45 |
| Passage index download (64 GB) | ~30 min | 1:15 |
| Sanity tests + first launch | ~10 min | 1:25 |
| Cycle 1 SIL fine-tune | ~10 min | 1:35 |
| Cycle 1 Step 2.1 cal-fold (with K=0) | ~2.5 h | 4:05 |
| Cycle 1 Step 2.2–2.5 recalib + retroverify | ~10 min | 4:15 |
| Cycle 1 stream chunk (9000, with K=0) | ~9 h | 13:15 |
| Cycle 1 eval-fold (3500) | ~2 h | 15:15 |
| **Cycle 1 complete, cycle 2 begins** | | ~15 h |
| Cycles 2–10 (each ~13 h with optimizations) | ~117 h | ~132 h |
| **Step 7 main complete** | | **~5.5 days** |

vs the pre-optimization estimate of 7–8 days. Net savings from the h_norm retirement + u_pre cache: **~50 hours** across the trajectory.

---

## 5. Post-Step-7 follow-up steps (already in `run_phase1a.sh`)

After Step 7 main completes, the script automatically runs:
- Step 8: FLARE smoke test
- Step 9–14: External baselines (B1–B7) on the matched protocol
- Step 15: Statistical significance (paired McNemar + BCa bootstrap, Holm correction)
- Step 16–18: Excluded from Phase 1a (Phase 1 Full tranche, supervisor-funded)
- Step 19: Purity validation
- Step 20.1: Aggregate results CSV
- Step 21: Ch5 table generation (`tab_headline.csv`, etc.)

These are sequential and idempotent — safe to interrupt and resume.

---

## 6. Common gotchas

1. **"FileNotFoundError: cycle_N.json missing"** on resume — the runner expects `eval/{bench}_cycleN.json` for every previously-completed cycle. If you have a `cycle_N/` dir without the eval fold finished, move it aside per §3.7 strategy A.

2. **Resume detection picks up `cycle_*.halted_*` directories** — `run_phase1a.sh:618` parses the suffix as integer. Always move halted dirs to `outputs/_halted_runs/` (outside the `cycle_*` glob) to avoid this.

3. **`disable_h_norm=False` in old runs** — if you resume from a checkpoint produced before commit `c985bc8`, the running config might pick up disable_h_norm=False from the dataclass default. Verify with `python -c "from caem.config import CAEMConfig; print(CAEMConfig().disable_h_norm)"` before launching. Default is True post-commit.

4. **Vast credit runs out mid-run** — gdrive offload happens at the END of each cycle's SIL fine-tune. If credit runs out mid-cycle, you lose work back to the previous cycle's offload. Recovery: rent a new instance, restore from gdrive, force resume from the last completed cycle. Per CAEM_PHASE1_AUTONOMY: user authorized autonomous execution until done or credit exhausted.

5. **Conformal gate file paths** — `cycle_0/conformal_gate.json` is the canonical baseline. `cycle_N/conformal_gate.json` (N≥1) is written by Step 2.3 conformal refit; the runner's `run_per_cycle_conformal_refit` also copies the new gate to `cycle_0/conformal_gate.json` so subsequent cycle's pipeline-init reads the latest. This is by design.

6. **HF download fails with "rate limited"** — pass `--max-workers 1` to `huggingface-cli download` and retry. Or use `git lfs` clone of the HF repo.

7. **rclone gdrive 403 quota** — gdrive has a daily quota per token. If you hit it during checkpoint offload, the cycle continues (offload is best-effort, not blocking) but you lose redundancy. Renew the rclone token or use a different gdrive account.

---

## 7. Where things live (cheat sheet)

| Thing | Location |
|------|----------|
| Latest code | `feat/qwen-3b-goal1` on GitHub |
| Passage index | HF `aksaN000/caem-passage-index-21m` |
| Cold-start memory | git: `outputs/cold_start_memory/` |
| Cycle 0 calibration | git: `outputs/cycle_0/composite_calibration.json` + `conformal_gate.json` + `calibrated_thresholds.json` |
| Cycle 0 model.pt | gdrive: `caem-phase1a/full_run/cycle_0/model.pt` |
| Cycle 1+ model.pt | gdrive: `caem-phase1a/full_run/cycle_<N>/model.pt` (after each SIL fine-tune offload) |
| Run logs | local `outputs/full_run/run.log`, `outputs/phase1a_runner.log` |
| Thesis sources | git: `thesis_report/` |
| Production runbook | git: `docs/PRODUCTION_RUNBOOK.md` |

---

**Final word:** start the resume from §3.1, follow each section in order, and check the §3.9 first-hour validations. Anything red there means stop and diagnose; never barrel through with a partially-broken setup. The 5-day step-7-main run is sensitive to silent failures (the h_norm retirement story is itself an example — silent zero-weight signal cost ~40% of cal-fold time before audit caught it).
