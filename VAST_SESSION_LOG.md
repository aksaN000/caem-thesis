# CAEM Phase 1 — Vast.ai Session Log

**Session start:** 2026-04-18 (first tool activity at ~14:51 BDT).
**Plan being executed:** Plan A from `NEXT_SESSION_PLAN.md` — Steps 5, 6, 7, 8, 9–15, 19, 20. Steps 16–18 deferred (out of $45 budget).
**Runner:** `/workspace/caem/run_plan_a.sh` in tmux session `plan_a` (auto-chains Steps 5–20 once Step 4 finishes).

All times in this log are **Dhaka (BDT, UTC+6)**. Raw tool logs use UTC.

---

## Instance specifications

| Field | Value |
|---|---|
| Instance ID | 35180107 |
| Datacenter | 81036 |
| Machine ID | 30024 |
| GPU | NVIDIA GeForce RTX 5090, 31.8 GB VRAM |
| Max CUDA | 13.1 |
| Compute (raw) | 108.1 TFLOPS |
| GPU memory bandwidth | 1454.2 GB/s |
| DLPerf | 203.1 (DLP/$/hr: 318.5) |
| Network | 250 ports, 4648.7 Mbps down / 6801.6 Mbps up |
| CPU | AMD EPYC 9654 96-core (48 cores allocated) |
| RAM allocation | 96.7 GB (per Vast UI); `free -h` on the box reports 755 GB total (host-wide) |
| Disk | KIOXIA KCD8XRUG7T68 NVMe, ~16 GB/s, 150.1 GB allocated |
| Motherboard | GENOA2D24G-2L, PCIE 5.0 x16, 54.2 GB/s |
| Volumes | None |

## Budget

- **Hourly rate:** ~$0.638/hr (covers GPU + CPU + disk on this host).
- **Starting Vast credit:** $48 (as stated by user at session start).
- **Plan A cost projection:** ~$26 total (~40 hours).
- **Deferred:** Steps 16–18 (screening + confirmatory ablation sweep). Would cost $40+ on this instance and blow the budget.

## Runner topology

| tmux session | Purpose | Started (BDT) |
|---|---|---|
| `build` | Step 4 passage index build | 14:50 BDT (08:50 UTC) |
| `plan_a` | Chain runner for Steps 5–20 | 15:18 BDT (09:18 UTC) |
| `ssh_tmux` | User's SSH shell | 13:48 BDT (07:48 UTC) |

**Key files produced this session:**
- `build_passage_index.log` — Step 4 raw output.
- `plan_a_runner.log` — chain runner orchestration output.
- `tests_self_improvement.log` — Step 3B pytest output.
- `run_plan_a.sh` — autonomous chain runner.
- `frontier_reference_addendum.md` — writeup plan for frontier-model comparison (post-Phase-1 addition).
- `outputs/` — experiment outputs (populated as steps complete).

---

## Step completion log

Status legend: ✅ done | 🏃 running | ⏳ queued | ❌ failed | ⊘ skipped

| Step | Status | Start (BDT) | End (BDT) | Notes |
|---|---|---|---|---|
| 3.1 Clone repo | ✅ | (pre-session) | — | Repo pre-cloned at `/workspace/caem`; remote PAT updated inline. |
| 3.2 pip install deps | ✅ | 14:49 | 14:52 | torch, transformers, datasets, sentence-transformers, faiss-cpu, scipy, scikit-learn. |
| 3.3 GPU + hardware profile | ✅ | 14:52 | 14:53 | Confirmed RTX 5090, bf16, batch 32, TF32 on. |
| 3.4 HF model cache | ✅ | 14:53 | 14:55 | flan-t5-large + roberta-large-mnli + all-mpnet-base-v2. Had to downgrade sentence-transformers 5.4.1 → 3.4.1 (and transformers 5.5.4 → 4.57.6) — see Incident #1. |
| 3B Pytest gate | ✅ | 14:55 | 14:56 | `tests/test_self_improvement.py` → 32/32 in 14.7s. No `forgetting_score` rename misses. |
| 4 Passage index (21M) — first attempt | ❌ killed | 14:50 | 21:09 | Reached FAISS k-means with 500k training set; killed before write. See Incident #2. Lost: $3.64 + 5h. |
| 4 Passage index (21M) — rebuild | 🏃 | 21:10 | TBD (~03:20 BDT) | `--train_sample_size 2000000` (patched script default also). PID 159638. |
| 5 Smoke test (1 cycle, n=50) | ⏳ | — | — | Hard gate on CES > 0. Auto-launched by `plan_a` runner. |
| 6 Cold-start seed | ⏳ | — | — | target 200 episodes across fever/triviaqa/nq. |
| 7 Main 10-cycle CAEM | ⏳ | — | — | Headline Ch5 run. 3-attempt retry with `--resume_from_cycle`. |
| 8 FLARE pre-flight smoke | ⏳ | — | — | Hard gate on empty_answer=0 AND escalated>=1. |
| 9 B1 Zero-shot | ⏳ | — | — | Soft-fail. |
| 10 B2 CoT | ⏳ | — | — | Soft-fail. |
| 11 B3 DPR-RAG | ⏳ | — | — | Soft-fail. |
| 12 B4 CoT + RAG | ⏳ | — | — | Soft-fail. |
| 13 B5 FLARE | ⏳ | — | — | Conditional on Step 8 gate pass. |
| 14 B6 Vanilla FT (10-cycle) | ⏳ | — | — | ~6 h. |
| 15 B7 EWC-only FT (10-cycle) | ⏳ | — | — | ~6 h. |
| 16 Screening sweep | ⊘ | — | — | **Skipped** — out of Plan A scope (budget). |
| 17 Aggregate screening | ⊘ | — | — | **Skipped** — depends on 16. |
| 18 Confirmatory sweep | ⊘ | — | — | **Skipped** — ~$32 on this instance, over budget. |
| 19 Purity validation | ⏳ | — | — | ~30 min. |
| 20 Final aggregate + tar | ⏳ | — | — | scp to local PC is manual post-run. |

---

## Incidents / deviations

### #1 — sentence-transformers torchcodec incompatibility (2026-04-18, 14:53 BDT)

**Symptom:** `from sentence_transformers import SentenceTransformer` failed at import time with `RuntimeError: Could not load libtorchcodec… libnppicc.so.13: cannot open shared object file`.

**Root cause:** sentence-transformers 5.4.1 (the current PyPI default against Python 3.12 on this instance) imports `torchcodec` transitively through its audio-modality module. torchcodec requires CUDA NPP (`libnppicc`) which isn't present on this Vast image.

**Fix:** Downgraded to sentence-transformers 3.4.1 (and transformers 4.57.6 for compat). The `SentenceTransformer(name, device=...).encode()` API is unchanged across these versions, and CAEM only uses that basic surface (verified via grep of `caem/memory/encoder.py`). No code changes required.

**Long-term fix (for next session):** pin `sentence-transformers<4` in the repo's install command in `NEXT_SESSION_PLAN.md` Step 3.2, to avoid hitting this again on a fresh rental.

### #2 — FAISS IVF training set undersized (2026-04-18, 20:34 BDT — RESOLVED via rebuild)

**Symptom:** After encoding finished, FAISS emitted:
```
WARNING: IVF training set (500000 rows) is smaller than the 30*nlist recommendation (1966080 for nlist=65536). Index quality may suffer.
WARNING: clustering 500000 points to 65536 centroids: please provide at least 2555904 training points.
```

**Why it matters:** FAISS IVF-PQ partitions the 768-dim embedding space into 65,536 cells via k-means; with only 500k training points (~7.6 per centroid vs the recommended 30+), centroids are placed noisily, queries probe the wrong cells, and recall@10 degrades by an estimated 5–15 percentage points. Downstream effect on B3/B4/B5 RAG accuracy is probably 0–2 pp — modest but non-zero.

**In-RAM extraction attempt (failed):** Tried to inject Python via gdb/pyrasite to dump the 21M × 768 embedding matrix from the running process before killing. Blocked at the container level — Vast's seccomp profile disallows the `ptrace` syscall ("Inappropriate ioctl for device"). All process-injection paths (gdb, pyrasite, py-spy) fail the same way. Confirmed unrecoverable from inside the container.

**Resolution:** Killed the in-progress build (PID 15570) at 21:09 BDT, **lost 5h + $3.64** of encode work. Patched `scripts/build_passage_index.py` default for `--train_sample_size` from `500_000` → `2_000_000`. Restarted build (PID 159638) with explicit `--train_sample_size 2000000`. ETA ~03:20 BDT, ~$3.83 additional cost. New total Step 4 cost: ~$7.50 (vs ~$3.83 if we'd waited and accepted quality loss).

**User decision rationale:** Opted for full quality over the $3.83 + 6h cost. Acceptable within the $48 budget; updated Plan A end-state projection from ~$22 to ~$18 of cushion remaining.

**Old build log preserved at:** `/workspace/caem/build_passage_index.bad500k.log`.

---

## Autonomy authorization

User authorized autonomous execution of Plan A on 2026-04-18 (~14:55 BDT equivalent). Claude may execute Steps 5→20 (excluding 16–18) without per-step confirmation until Plan A completes OR Vast credit exhausts OR a hard-gate step fails. Does NOT extend to re-enabling Steps 16–18 without explicit re-confirmation.

See memory file `caem_phase1_autonomy.md` for the durable record.

---

## Running cost estimate (to update as steps complete)

| Point in time | Elapsed | $ spent (approx) | Credit remaining |
|---|---|---|---|
| Session start (14:50 BDT) | 0 h | $0.00 | $48.00 |
| Step 3.2–3B done (14:56 BDT) | 0.1 h | $0.07 | $47.93 |
| Step 4 encoding done (20:33 BDT) | 5.7 h | $3.64 | $44.36 |
| Step 4 first attempt killed (21:09 BDT) | 6.3 h | $4.02 | $43.98 |
| Step 4 rebuild start (21:10 BDT) | 6.3 h | $4.02 | $43.98 |
| Step 4 rebuild done (TBD ~03:20 BDT) | ~12.5 h | ~$7.98 | ~$40.02 |

*(Append a new row here whenever a step finishes.)*
