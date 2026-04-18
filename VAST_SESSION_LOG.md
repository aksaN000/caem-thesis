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

---

## Status checkpoints

### 22:05 BDT (16:05 UTC) — Step 4 rebuild in progress, plan_a idle

- **Step 4 encoding:** 34 / ~210 buffers done (100k each → 3.4M / 21M passages).
  Current buffer #35 at 11% (batch 22/196). Throughput ~1,090 passages/s.
- **Revised encoding ETA:** ~02:25 BDT on 2026-04-19 (encoding only). FAISS
  IVF-PQ training on 2M samples + index write adds ~20–30 min → **full
  Step 4 ETA ~02:50–03:20 BDT**, consistent with the Incident #2 rebuild
  projection.
- **`plan_a` runner state:** PID 29562 alive, polling `passages.faiss` +
  `passages.pkl` every 120 s. Last log entry is the "Waiting for Step 4
  outputs..." banner from 15:18 BDT (09:18 UTC). Nothing else printed —
  the runner has not yet entered Step 5.
- **Burn rate check:** ~8.9 h × $0.638/hr ≈ $5.68 actual spend to this
  moment (vs $5.65 projected at this hour in the cost table — tracking).
- **Memory:** `caem_phase1_autonomy.md` ETA line updated to reflect the
  revised 20:50 UTC completion target.
- **Next observable event:** `/workspace/caem/data/passage_index/passages.faiss`
  and `passages.pkl` materialise, which unblocks Step 5 (smoke-test hard
  gate on CES > 0) automatically.

### 23:20 BDT (17:20 UTC) — Phase 1A scope freeze, code patches, runner relaunch

User decision: **Phase 1A clean**. Skip Steps 16–18 (ablations) and Phase 2
(multi-seed). Ship the 10-cycle CAEM run with pre-registered Holm-corrected
CAEM-vs-baseline significance table, defer ablations + multi-seed to future
work. Code patches landed before Step 4 finished:

- `eval/metrics.py:extract_arc_label` — regex rewritten letter-first, digit
  fallback (prevents CoT numbered-rationale misfire).
- `eval/reporting.py:TABLE_FILES` — within-CAEM table renamed
  `tab_sig_test.csv → tab_cycle_progression.csv` so the slot `tab_sig_test.csv`
  is reserved for the CAEM-vs-baseline table that Ch5 §5.3 references as
  `\ref{tab:sig-test}`. `scripts/make_tables.py` registry updated in lockstep.
- New script `scripts/baseline_sig_tests.py` — pairs CAEM cycle 10 (post-
  training, strongest-claim state) against every baseline (B1–B7) by
  sample `id`, Holm-within-family with `m = len(baselines_present)`,
  emits `tab_sig_test.csv` with dagger/star/baseline_wins columns.
  Synthetic 4-scenario smoke test passes. (Earlier spec had cycle 3 for
  B1–B5; reverted to cycle 10 for all because inference-only baselines
  have no training budget to match, so there is nothing to gain from
  comparing against an early CAEM checkpoint — cycle 10 is the
  strongest, least-arbitrary CAEM state. Within-CAEM trajectory remains
  visible in `tab_cycle_progression.csv`.)
- `run_plan_a.sh` — all Steps 9–15 bumped from `--n_questions 500` to
  `--n_questions 5000` (matching CAEM's eval pool); new Step 15.5 runs
  `baseline_sig_tests.py`; Step 20 tar includes `tab_sig_test.csv`.
- `NEXT_SESSION_PLAN.md` Steps 9–15 bumped to n=5000 with rationale footnote.
- `chapters/chapter_5.tex` §5.3: added metric-scoring-caveats paragraph
  (TruthfulQA ROUGE-L proxy non-comparable to GPT-judge; MMLU generation-mode
  ~3–8 pp below log-likelihood leaderboard; retention ratios are the licensed
  claim). Hard-coded "m = 8" replaced with "m = number of baselines in the
  benchmark family"; Phase 1A stated as m = 7.
- `plan_a` tmux session: killed PID 29562, relaunched as **PID 218848** so
  the updated script is what bash reads line-by-line from disk. `build`
  tmux session untouched.

### 23:40 BDT (17:40 UTC) — Paired-sample overlap footnote

With n=5000 applied to both CAEM eval and all baselines, paired-sample
overlap per benchmark is now 100% (benchmarks smaller than 5000 return the
full pool from the loader; benchmarks larger than 5000 are sampled via
`random.Random(42).sample(pool, 5000)`, which is deterministic and matches
across CAEM + baseline paths). Resulting n_paired per benchmark for the
McNemar test in `tab_sig_test.csv`:

| Benchmark | Pool size | n_paired |
|---|---:|---:|
| FEVER paper_dev | ~19,998 | 5,000 |
| TriviaQA validation | ~17,944 | 5,000 |
| NQ validation (nq_open) | 3,610 | 3,610 |
| TruthfulQA validation | **817** | **817** |
| StrategyQA test | ~2,290 | 2,290 |
| ARC-Challenge test | ~1,172 | 1,172 |

**TruthfulQA caveat (to surface in Ch5 §5.3 before submission):** TruthfulQA
is the only benchmark whose n_paired (817) is limited by the underlying
pool, not by our n_questions setting. Its McNemar has adequate power at
|ΔEM| ≥ 0.04 (α=0.05, β=0.8) but is borderline at the pre-registered
practical-significance floor |ΔEM| ≥ 0.02. The other five benchmarks clear
ΔEM ≥ 0.02 comfortably. A one-sentence footnote on this should be added
to Ch5 §5.3 alongside the TruthfulQA ROUGE-proxy disclosure — the two
caveats both concern TruthfulQA-specific comparability.

### 00:15 BDT (18:15 UTC, 2026-04-19) — B6/B7 argparse + matched training budget

Audit of `scripts/run_simple_ft.py` caught that the B6/B7 invocations in
`run_plan_a.sh` were passing three unrecognised flags that argparse would
reject at parse time:

- `--benchmarks $BENCH` → the script expects `--eval_benchmarks`.
- `--passage_index data/passage_index` → no such flag (B6/B7 don't retrieve).
- `--n_questions 5000` → the script has `--n_eval_per_bench` and
  `--n_train_per_bench` as separate knobs.

This was latent from before the session; never surfaced because B6/B7 had
not yet been reached. Two paths out of the bug:

- *Option 1 (eval-match only):* `--n_eval_per_bench 5000` with
  `--n_train_per_bench` left at default 2000. B6/B7 get 6K training pairs/cycle
  vs CAEM's 12K → 2× training-data asymmetry favouring CAEM.
- *Option 2 (matched training budget):* `--n_eval_per_bench 5000
  --n_train_per_bench 4000` → B6/B7 get 4000 × 3 ID benchmarks = **12K
  pairs/cycle, matching CAEM's SIL training pool** after
  `split_calibration_sets` carves off calib (500) + purity (500) from the
  5000 SIL pool per benchmark.

**Decision:** Option 2. Ch5 §5.3 pre-registers "matched training budget";
Option 1 would render the star markers against B6/B7 uninterpretable (any
CAEM win could be attributed to 2× training data rather than to the
verifier/memory/L2-anchor/retroverify stack). The `--n_eval_per_bench 5000`
setting also preserves the 5000-paired-sample overlap for the sig-test.

**Conservatism note:** CAEM's *effective* training data per cycle is lower
than 12K because the verifier only fine-tunes on STORE-decision episodes
(empirical storage rate ~40-60%). Handing B6/B7 the full 4K raw pool per
benchmark thus *biases the comparison against CAEM*. That direction is
reviewer-friendly: if CAEM still beats B6/B7 under a training-pair
advantage for B6/B7, the mechanism claim is stronger.

**Cost impact:** +$7-8 over Option 1 (doubling train pairs ~doubles FT
wall time per cycle, ~6h→~12h per baseline). Absorbed inside the Phase 1
Full credit top-up the user is adding for Steps 16-18 ablations later.

**Also applied (belt-and-suspenders):** `scripts/baseline_sig_tests.py`
now reads `outputs/full_run/dataset_splits.json` (written by
`run_experiment.py` line 900 as the authoritative eval-split record) and
restricts id-intersection to the `eval_ids` list per benchmark. Fallback
to raw id-intersection with a warning when the splits file is missing
(e.g. pre-CAEM-run smoke paths). Synthetic 4-scenario smoke test still
passes after the edit.

**Runner relaunch:** PID 246789 became zombie on kill; new runner under
**PID 246850**. FD 255 now points at the live
`/workspace/caem/run_plan_a.sh` (not a deleted inode), so the patched
Steps 14/15 will be read from disk when the runner reaches them.

**Updates to `NEXT_SESSION_PLAN.md`** mirror the same flag/budget changes
so future sessions don't regress.

### 00:25 BDT (18:19 UTC, 2026-04-19) — eval_id-filter patch across baseline scripts

Both baseline scripts now accept `--caem_splits_path` and filter their
per-benchmark sample slice to the exact `eval_ids` that CAEM wrote in
`outputs/full_run/dataset_splits.json`. Mirrors the
`load_and_filter` pattern in `run_purity_validation.py:1011-1023`. Removes
the implicit dependence on `load_benchmark(bench, n, split)` being
deterministic across re-invocations.

- `scripts/run_baseline.py` (B1–B5): new CLI flag, pre-loop loader reads
  `eval_ids` per benchmark, per-benchmark slice is filtered after
  `load_benchmark` returns. Empty filter → reload unfiltered (prevents
  zero-sample crash on smoke paths).
- `scripts/run_simple_ft.py` (B6/B7): same flag, filter applied inside the
  per-cycle eval loop (the TRAINING pool is deliberately NOT filtered —
  training draws from the full train split independently).
- `run_plan_a.sh` defines `CAEM_SPLITS=/workspace/caem/outputs/full_run/dataset_splits.json`
  once near the top and passes `--caem_splits_path "$CAEM_SPLITS"` to all
  seven B1–B7 invocations. The file is written by `run_experiment.py:900`
  during Step 7, so it exists by the time Step 9 fires.
- Fallback behaviour: if the splits file is missing at baseline start-time
  (e.g. Step 7 crashed), the baselines log a warning and proceed with
  un-filtered `load_benchmark` slices. Sig-test pairing would then fall
  back to raw id-intersection between CAEM + baseline JSONs, with reduced
  but non-zero n_paired.

**Runner relaunch:** old PID 246850 held a deleted inode after the
in-place edits; killed, restarted as **PID 249849**. FD 255 → live
`run_plan_a.sh` confirmed. AST-parse check passes on both baseline
scripts; `--help` confirms the new flag is registered on both.

### 00:30 BDT (18:25 UTC, 2026-04-19) — Status snapshot and B6/B7 pool-size audit

**Current state**

- Step 4 encoding: 116 / ~210 buffers (55%), buffer #117 at batch 146/196;
  throughput ~4 it/s on 5090. Encoding ETA ~21:00 UTC / ~03:00 BDT.
  Full Step 4 (FAISS IVF-PQ train-on-2M + write): ~03:30 BDT.
- Runner: tmux `plan_a`, PID 249849, FD 255 → live `run_plan_a.sh`, polling
  every 120 s. No progress past waiting loop.
- Output dirs exist as empty stubs: `ablation/`, `baselines/`,
  `baselines_smoke/`, `full_run/`, `purity_validation/`, `smoke/`.
- Burn: ~11.2 h × $0.638/hr ≈ $7.15 spent. Remaining ~$40.85 (pre-topup).

**B6/B7 evaluation pool-size behaviour (confirmed during audit)**

`run_simple_ft.py --eval_benchmarks` defaults to all 6 benchmarks; Plan A
passes all 6 explicitly. Each cycle evaluates every benchmark at
`--n_eval_per_bench 5000`, then filters to CAEM's `eval_ids` from
`dataset_splits.json`. Loader behaviour under the cap:

| Benchmark | Pool split | Pool size | Effective n at eval |
|---|---|---:|---:|
| FEVER | paper_dev | ~19,998 | 5,000 (random-sampled) |
| TriviaQA | validation | ~17,944 | 5,000 (random-sampled) |
| Natural Questions | validation | 3,610 | 3,610 (full pool) |
| TruthfulQA | validation | 817 | 817 (full pool) |
| StrategyQA | test | ~2,290 | 2,290 (full pool) |
| ARC-Challenge | test | ~1,172 | 1,172 (full pool) |

`eval/benchmarks.py` loaders all use the idiom
`if n is not None and n < len(samples): rng.sample(...)`. When `n >= pool`,
the sample() call is skipped and the full pool is returned -- no error,
no truncation. So the "5000 requested" setting is a ceiling, not a
contract. The eval_ids filter applied afterwards restricts to CAEM's
exact slice, which for small benchmarks IS the full pool (CAEM also
returned the full pool). Net: paired overlap stays 100% on every
benchmark.

**Separate finding (requires user decision — NOT patched yet)**

`run_simple_ft.py::_load_train_pool` at line 177-180 passes `split="train"`
only for FEVER; for TriviaQA and Natural Questions it calls
`load_benchmark(bench, n=n)` with no split, which defaults to
`split="validation"` in the respective loaders. The *eval* loop uses the
same `validation` splits for TriviaQA/NQ. This means B6/B7 potentially
**train on and evaluate on the same validation split** for those two
benchmarks (the specific overlap depends on how `random.Random(42).sample`
with n_train=4000 vs n_eval_per_bench=5000 intersects on the ~17,944 /
~3,610 pools). For NQ specifically, both pools would exceed the 3,610
full pool in different ways and overlap is guaranteed.

CAEM's `seed_cold_start.py` correctly uses `split="train"` for all three
ID benchmarks (it passes explicit splits everywhere). CAEM's main-run
SIL pool separately carves disjoint train/calib/purity/eval slices via
`split_calibration_sets`, so CAEM itself has no leakage. The leakage is
B6/B7-specific.

Open question for Aksan: fix B6/B7's `_load_train_pool` to pass
`split="train"` explicitly for TriviaQA and NQ (one-line patch each,
same idiom as FEVER), or leave as-is and document the asymmetry as a
known limitation of the B6/B7 training-baseline spec. Leaving as-is is
defensible only if B6/B7 are framed as "validation-split fine-tuning"
rather than "train-split fine-tuning" — which is NOT what Ch5 §5.3
pre-registers.

### 00:37 BDT (18:37 UTC, 2026-04-19) — Sig-test pairing cycle corrected

Caught in review: `baseline_sig_tests.py` was pairing CAEM cycle 3 vs
B1–B5 (rationale had been "matched self-improvement maturity"). That
rationale was wrong — inference-only baselines have no training budget
to match, so the correct comparison is CAEM at its strongest state.
**Patched `CAEM_CYCLE_INFERENCE = 3 → 10`** so all baselines pair
against CAEM cycle 10. Matched-training-budget argument for B6/B7
still holds (both sides at cycle 10). Smoke test re-run and passes.
Docstring updated. Chapter 5 §5.3 has no thesis-text footprint on the
cycle choice (grep confirmed) — it was purely an ops-level decision
that lived in the script constants.

### 00:45 BDT (18:45 UTC, 2026-04-19) — B6/B7 train-split leakage fix

Closes the "open question for Aksan" flagged earlier. Three changes
landed as a single pass, in this order:

**1. Pre-flight loader smoke test (GPU idle, Step 4 still encoding).**
Standalone Python exercised `load_fever(split="train")`,
`load_triviaqa(split="train")`, `load_natural_questions(split="train")`
with `n=10`. All three loaders return the schema the training loop
consumes (`question`, `answers[0]`, `id`), HuggingFace pool sizes
confirmed:

- FEVER train: 145,405 samples
- TriviaQA train: 138,384 samples
- Natural Questions train: 87,925 samples

All pools >> `n_train_per_bench=4000` so the 4K random-sample always
draws from genuine train-split items. Catches the loader *before* B6/B7
fires at ~hour 15 of the overnight run, not after 15 h of sunk compute.

**2. `scripts/run_simple_ft.py::_load_train_pool` — one-line fix.**
Replaced the FEVER-only special case with a single
`load_benchmark(bench, n=n, split="train")` that fires for every
training benchmark. Return signature changed to
`(pool, train_ids_by_benchmark)` so the caller can enforce the
disjointness invariant without a second pool read.

**3. Mandatory cycle-0 train/eval disjointness guard (`RuntimeError`).**
New block in `main()` between `_load_train_pool` and the cycle loop.
Intersects `train_ids_by_benchmark` against
`outputs/full_run/dataset_splits.json::eval_ids` per benchmark and
**raises `RuntimeError`** on any non-empty overlap. Uses `RuntimeError`
(not `assert`) so the check survives `python -O`. Each benchmark logs
`train/eval disjoint verified: <bm> N_train=X N_eval=Y overlap=0` at
info level so the run log records the invariant holding.

When `--caem_splits_path` is empty or the file is missing, the check
degrades to a warning (prints loudly; does NOT block). This is only
acceptable for smoke runs before CAEM has written its splits file;
Phase 1A real runs always pass `--caem_splits_path`, so the guard fires
every cycle.

**Why this matters for Ch5 §5.3.** Ch5 pre-registers B6/B7 as "10-cycle
fine-tuning on the ID training pool." Before the fix, TriviaQA and NQ
pulled silently from their `validation` splits — the same splits the
per-cycle eval reads from. For NQ (pool size 3,610 < n_train=4000 and
< n_eval=5000), both pools returned the full 3,610 → **every training
sample was also an eval sample** (100% leakage). For TriviaQA the
overlap was partial (~25-40%). The bug biased *against* CAEM (B6/B7 EM
inflated by memorisation → CAEM's relative gain under-estimated), so
it could not have produced false-positive CAEM wins, but it would have
robbed legitimately-earned star markers on NQ and TriviaQA. The fix
aligns code with Ch5 §5.3 pre-registration, no thesis text changes.

**Runner impact: zero.** The change is script-level only;
`run_plan_a.sh` already passes `--caem_splits_path "$CAEM_SPLITS"` to
B6 and B7, so the guard will fire on first B6 invocation (~hour 15 of
the overnight run). Current runner (PID 249849) does not need to be
relaunched because bash has not yet read past the Step 4 waiting loop;
by the time execution reaches Step 14, bash will re-read lines from
the live inode that now contains the patched script.

Wait — correction: the runner re-reads `run_plan_a.sh` line-by-line but
`run_plan_a.sh` itself wasn't changed; only `scripts/run_simple_ft.py`
was. That script is re-invoked fresh by `python -m scripts.run_simple_ft`
at Step 14, reading the live file at invocation time, so the fix and
the guard will take effect without any runner relaunch.

AST-parse check: `run_simple_ft.py` parses OK post-edit. Variable scope
verified: `eval_ids_by_bm` is populated at line ~586-599 (from
`--caem_splits_path`), `_load_train_pool` returns at line 624, the guard
block runs at lines 627-663 before the cycle loop begins at line 665.

### 01:00 BDT (19:00 UTC, 2026-04-19) — Pre-flight audit caught three more blockers

While Step 4 was still encoding, did a final-pass pre-flight audit of the
downstream pipeline. Three issues surfaced that would have killed Plan A
at Step 7 (CAEM main run), Step 14/15 (B6/B7 disjointness guard), or
Step 19 (purity validation). All three patched before Step 4 completion.

**Blocker #1 — HuggingFace `lucadiliello/fever` renamed `paper_dev` to `dev`.**
`load_fever(split="paper_dev")` now raises `ValueError: Unknown split
"paper_dev". Should be one of ['train', 'test', 'dev']`. This would have
aborted Step 7 at Cycle 0 eval (first FEVER load) and every B1–B5 FEVER
eval at Steps 9–13. Verified empirically: `load_fever(split="dev")` loads
19,998 samples with identical schema (`claim`, `label`, `evidence`, `key`)
— the split was renamed, content unchanged.

Files patched (all active code paths; historical docstrings left intact):
- `eval/benchmarks.py:154` — default arg `"paper_dev"` → `"dev"`, plus four
  docstring references.
- `eval/__init__.py:32-33` — quick-start example.
- `eval/harness.py:136` — docstring example.
- `scripts/run_baseline.py:109-110` — default arg + help text.
- `scripts/run_baseline.py:335` — comment.
- `scripts/run_experiment.py:289` — `load_eval_transfer_pool` split_map.
- `scripts/run_simple_ft.py:774,790` — per-cycle eval loop (both call sites).
- `scripts/run_ablation.py:151` — split_map.

Post-fix verification: `load_fever(n=3)` returns 3 samples with default
split="dev". Historical log entry `caem-implementation-log.md` EXP-22
(Session 35) saying "Always use paper_dev" is now stale — not edited, but
the note here supersedes it.

**Blocker #2 — disjointness guard false-positive on NQ by id-string collision.**
NQ uses numeric `question_id` fields that collide in VALUE across train
and validation splits even though the underlying questions are distinct.
At realistic pool sizes (train n=4000, validation full 3610), 134 id
strings appear in both sets. The previous id-intersection guard would
have raised `RuntimeError` and aborted all B6/B7 cycles.

Patched: guard now uses sha256 prefix of stripped question TEXT rather
than id. Immune to cross-split id-schema collisions; catches genuine
leakage (same question in both pools). Added `_q_hash(text)` helper;
`_load_train_pool` now returns `train_qhashes_by_benchmark` (set of
hashes, not list of ids); guard block loads eval pool per benchmark
using `_EVAL_SPLIT_MAP = {"fever": "dev", "triviaqa": "validation",
"natural_questions": "validation", "strategyqa": "test",
"arc_challenge": "test"}`, filters to CAEM's `eval_ids`, intersects
hashes. Empirical verification on all three ID benchmarks:

| Benchmark | n_train_q_hashes | n_eval_q_hashes | overlap |
|---|---:|---:|---:|
| FEVER train vs dev | 3,980 | 4,977 | 0 ✓ |
| TriviaQA train vs val | 3,946 | 4,382 | 0 ✓ |
| NQ train vs val | 4,000 | 3,610 | 0 ✓ |

(Count deltas from raw n_train=4000 are near-duplicate questions within a
split — benign, expected for FEVER claims and TriviaQA aliases.)

**Blocker #3 — Step 19 purity validator pointed at the wrong directory.**
`run_purity_validation.py --checkpoints_dir` defaults to `"outputs"` but
CAEM main run writes to `outputs/full_run/`. The runner was calling the
validator with only `--output_dir outputs/purity_validation`, no
`--checkpoints_dir`, so the validator would have looked for
`outputs/memory_store_cycle_N.faiss/.meta` and
`outputs/dataset_splits.json` — neither of which exist.

Patched `run_plan_a.sh` Step 19 invocation to pass:

    --checkpoints_dir outputs/full_run
    --passage_index data/passage_index

`--passage_index` added defensively — the validator needs it for the
ID benchmark eval during purity measurement. Without it, would fall back
to a default path that may not match our Step 4 output.

**Runner relaunch**: `run_plan_a.sh` was edited for Step 19 fix, so the
bash FD went stale on the deleted inode. Killed session, relaunched as
**PID 272262**. FD 255 → live `run_plan_a.sh`. No progress lost (still in
Step 4 wait loop).

**Audit pass status**: clean. Step 5 smoke (`run_cyclic_ablation.py`) has
no paper_dev / split= references. Step 6 seed output path (`outputs/
cold_start_memory/memory_store.{faiss,meta}`) matches Step 7's
`--cold_start_memory outputs/cold_start_memory/memory_store` argument.
Step 20 `aggregate_ablation.py` exits 1 on empty ablation dir as before;
runner's `|| say "Step 20 aggregator errored (expected if no ablation
variants)"` soft-fails correctly. Baseline scripts' `--caem_splits_path`
with missing file path warns and proceeds (fallback behaviour).

**Step 4 current**: 136 / ~210 buffers, ~64%. Encoding ETA ~02:30 BDT;
full Step 4 including FAISS train+write ~03:00 BDT.

### 01:15 BDT (19:15 UTC, 2026-04-19) — Budget reconciliation and topup plan

User flagged current credit = $42.15 (vs $48.00 start → $5.85 spent on
Step 4 encoding + Incident #2 rebuild so far). Re-estimated the remaining
Plan A cost on RTX 5090 under the current config (baselines at n=5000,
B6/B7 at n_train=4000 + n_eval=5000):

| Remaining stage | 5090 cost estimate |
|---|---:|
| Finish Step 4 encoding + FAISS train-on-2M + write | ~$2.15 |
| Steps 5–6 (smoke + seed) | ~$0.37 |
| Step 7 CAEM 10-cycle headline | ~$14.00 |
| Step 8 FLARE smoke | ~$0.05 |
| Steps 9–13 B1–B5 at n=5000 (10x runbook n) | ~$15–20 |
| Steps 14–15 B6/B7 at n_eval=5000 + n_train=4000 | ~$14–20 |
| Step 15.5 sig-test | ~$0.01 |
| Step 19 purity + Step 20 tar | ~$0.43 |
| **Total Plan A remaining** | **~$46–57** |

Shortfall vs $42.15: $4–15 in the conservative scenario; just barely
fits in the optimistic scenario. Risk: if Vast credit exhausts
mid-baseline, the instance is interrupted. CAEM main-run has per-cycle
checkpoints (resumable); baseline runs have NO per-cycle resume — a
B6 run that dies 80% through loses everything and re-running costs
the full ~$10.

**Recommended topup #1 (protect Plan A): $20 in the next 24 h.** Well
before Step 14 (B6) fires (~hour 10 after Step 4 completes, so roughly
Sunday evening BDT). Brings effective credit to ~$62, comfortably
covering conservative Plan A estimate with ~$5–15 margin.

**Phase 1 Full (Steps 16–18) cost estimate on 5090:**

| Step | 5090 estimate |
|---|---:|
| Step 16 screening (16 × 3cyc × n=1500) | ~$6–10 |
| Step 17 aggregate | ~$0.05 |
| Step 18 confirmatory (top-5 + full × 10cyc × n=5000) | ~$24–29 |
| Subtotal | ~$30–39 |
| +20% runtime buffer | +$6–8 |
| +storage idle ($0.15/hr × ~3 days between sessions) | +$10 |
| **Phase 1 Full topup target** | **~$45–55** |

**Recommended topup #2 (before resuming for Phase 1 Full): $45.**

**Total Vast spend projection for Phase 1A + Phase 1 Full:**
~$48 starting + $20 + $45 = **~$113** across the full research run.

**Runner behaviour at Step 20 completion:** runner prints
"PLAN A COMPLETE" banner and the bash script exits normally. Steps
16–18 are NOT coded into `run_plan_a.sh`. The tmux `plan_a` session
stays alive but idle. User workflow from there:

1. `scp` the `plan_a_outputs.tar.gz` from the instance to local.
2. **Stop** (NOT Destroy) the Vast instance — preserves
   `/workspace/caem/data/passage_index/`, `outputs/full_run/`,
   `outputs/cold_start_memory/`, `outputs/baselines/`, and
   `hf_cache/` at ~$0.15/hr storage rate.
3. Top up $45 for Phase 1 Full.
4. Start the stopped instance (disk returns intact).
5. Paste the Step 16/17/18 commands from `NEXT_SESSION_PLAN.md`.
   Step 16 has a `run_screening.sh` wrapper spelled out inline; Step 17
   is a single `python scripts/aggregate_ablation.py` call; Step 18
   similar to 16 with an edited variant list.

No autonomous runner written for Phase 1 Full yet. Can be added on
request as `run_phase1_full.sh` (~40 LOC, chains Steps 16→17→18 with
the same tmux pattern). Risk-free addition since it's a new file that
doesn't touch anything running.

### 01:45 BDT (19:45 UTC, 2026-04-19) — Final consolidated phase plan (user-approved)

Revised scope decision: **Phase 1 Full (Steps 16–18) is now inside the
thesis-defense envelope**, not deferred. Phase 2 (multi-seed validation
at seeds 123, 456) moves to post-defense / pre-journal-submission only.

### Pre-defense execution sequence

**Phase 1A (autonomous runner, in progress):**
- Step 4 passage index build (in progress, ETA ~03:00 BDT 2026-04-19)
- Step 5 smoke test (hard gate on CES > 0)
- Step 6 cold-start memory seed
- Step 7 main 10-cycle CAEM headline run (writes `dataset_splits.json`)
- Step 8 FLARE pre-flight smoke (hard gate)
- Steps 9–13 B1–B5 at n=5000 with `--caem_splits_path` filter
- Steps 14–15 B6/B7 at n_train=4000, n_eval=5000 with q-hash guard
- Step 15.5 CAEM-vs-baseline sig-test → `tab_sig_test.csv`
- Step 19 purity theorem validation
- Step 20 tar + "PLAN A COMPLETE" banner + script exit

**Interlude:**
- scp `plan_a_outputs.tar.gz` to local
- **Stop (NOT destroy) the Vast instance**
- Top up credit for Phase 1 Full
- Start the instance (disk intact)

**Phase 1 Full (manual, runbook-driven):**
- Step 16 screening sweep (16 variants × 3 cycles × n=1500)
- Step 17 aggregate + pick top-N (5–6 variants), freeze in
  `caem-implementation-log.md`
- Step 18 confirmatory sweep (top-N + full × 10 cycles × n=5000)

**Interlude:**
- scp Phase 1 Full outputs
- Stop instance

**→ THESIS DEFENSE** with Phase 1A + Phase 1 Full results.

### Post-defense (optional, pre-journal/conference submission)

**Phase 2 (manual, runbook-driven):** Re-runs Step-18-equivalent at
seeds 123 and 456. Two paths:

- *Narrow Phase 2*: top-N × 2 seeds = ~14 runs, ~$60–80 on 5090.
  Defense-able asymmetric reporting (top-N has mean±std, other variants
  single-seed with caveat).
- *Full Phase 2*: all 14 cyclic variants × 2 seeds = 30 runs, ~$150–180
  on 5090. Journal-cleaner, every variant reported with mean±std.

Pick at submission time based on reviewer feedback and budget state.

### Topup schedule

| When | Amount | Rationale | Credit after |
|---|---:|---|---:|
| Session start | $48 | Initial Vast credit | $48 |
| Current (19:45 UTC) | $42.15 remaining | Step 4 encoding in progress | $42.15 |
| **Within ~24 h (conditional)** | **+$20** | **Safety margin for Plan A; user will monitor actual burn vs remaining credit as baselines fire (Steps 9–15). Skip if margin looks comfortable in real time.** | ~$60 |
| After Plan A finishes, before resume | **+$45** | Phase 1 Full (Steps 16–18) | ~$45 headroom |
| Post-defense, pre-submission | +$60–80 narrow / +$150–180 full | Phase 2 | varies |

**Pre-defense total Vast spend (if $20 safety topup is exercised):**
~$113 ($48 start + $20 safety + $45 Phase 1 Full).

**Pre-defense total if $20 safety is skipped (user watches burn and Plan A finishes comfortably under $42.15):** ~$93 ($48 start + $45 Phase 1 Full).

### Safety-topup decision rule (user-chosen)

The $20 topup is **contingent**. User will watch the live burn rate via
`pgrep`/`nvidia-smi` plus the Vast billing dashboard. Trigger conditions
for actually paying the $20:

- Step 7 (CAEM main run) takes noticeably longer than the 14–16 h 5090
  projection AND the runner is still minutes from entering Step 14 (B6);
  OR
- Credit at Step 13 completion is below ~$16 (need ~$15 cushion to
  finish B6 + B7).

If neither trigger fires, skip the topup. Plan A completes on the
$42.15 that's already on the card.

### Between-session hard rules (must follow)

1. **Stop, never Destroy** — Destroy wipes `data/passage_index/` ($3.50
   + 5.5 h to rebuild), `outputs/full_run/`, `outputs/cold_start_memory/`,
   and `hf_cache/`.
2. Storage idle rate ~$0.15/hr (~$3.60/day). Tolerable for a few weeks,
   expensive for months. If Phase 1 Full is >2 months out, download
   critical artifacts locally and then Destroy — pay ~$3.50 to rebuild
   the index later, save $100+ in idle storage.
3. First operation on any resumed session:
   `ls -lh /workspace/caem/data/passage_index/passages.faiss`
   Non-empty confirms the preserved disk mounted correctly.

### 03:45 BDT (21:45 UTC, 2026-04-19) — StrategyQA loader blocker + fix (caught during pre-flight audit of Step 7 dataset availability)

FAISS IVF-PQ training emitted its "2000000 points / 2555904 recommended"
warning mid-Step-4. Warning is informational -- FAISS proceeds with whatever
training data it has, at a minor recall cost vs the 39x-nlist ideal. At
30.5x nlist we're at 78% of recommended density -- estimated 1-3 pp
recall@10 loss vs 5-15 pp in Incident #2's 500k case. Symmetric across
CAEM and all baselines (same index), so doesn't bias the sig-test
comparison. No action taken; build proceeds.

While FAISS trained, did a dataset-availability pre-flight probe for every
split Step 7's eval path will touch. Four of five OK; **StrategyQA test
split FAILS** with the deprecated `wics/strategy-qa` HF dataset script:

```
Dataset scripts are no longer supported, but found strategy-qa.py
RuntimeError: StrategyQA split 'test' unavailable or unlabeled, and
  allow_train_fallback=False.
```

This would have aborted Step 7 at Cycle 0's first StrategyQA eval pass
(~30 min after Step 4 finishes). Fix landed before runner could trigger it:

**Patch**: `eval/benchmarks.py::load_strategyqa` now tries
`ChilleD/StrategyQA` FIRST (a currently-maintained HF mirror with
labeled `train` (1,603) and `test` (687) splits, same underlying
StrategyQA data). Falls through to the legacy `wics/strategy-qa` path
for backward compatibility, then to the official `train.json` URL
fallback (still behind `allow_train_fallback=True`).

Schema maps cleanly: ChilleD's `qid` → CAEM's `id`, `answer` (bool) →
CAEM's `answers` (["yes"]/["no"]), `question` passes through with the
"Answer yes or no. Question: ..." prefix applied in `_normalise_rows`.
Verified: 687 unique ids on test split, balanced yes/no distribution
(roughly 52/48).

**Affected downstream paths (all fixed by the one patch):**
- `run_experiment.py::load_eval_transfer_pool` -- Step 7 CAEM main
  run eval on StrategyQA.
- `run_baseline.py` -- Steps 9-13 B1-B5 StrategyQA eval.
- `run_simple_ft.py` -- Steps 14-15 B6/B7 per-cycle StrategyQA eval.
- `run_purity_validation.py` -- Step 19 purity measurement on
  StrategyQA (OOD probe).

All six benchmarks on Plan A's panel are now verified loadable:

| Benchmark | HF source | split used | n |
|---|---|---|---:|
| FEVER | lucadiliello/fever | dev | 19,998 |
| TriviaQA | trivia_qa (rc.nocontext) | validation | 17,944 |
| Natural Questions | nq_open | validation | 3,610 |
| TruthfulQA | truthful_qa (generation) | validation | 817 |
| StrategyQA | ChilleD/StrategyQA | test | **687** (patched) |
| ARC-Challenge | ai2_arc (ARC-Challenge) | test | 1,172 |
| MMLU (retention probe) | cais/mmlu (all) | test | 14,042 |

No runner relaunch needed -- `run_plan_a.sh` is unchanged; the Python
script that holds the patch gets re-read fresh on every
`python -m scripts.run_experiment` invocation.

### Step 4 FAISS training status

Started clustering at 21:11 UTC. CPU-bound on EPYC 9654 (faiss-cpu, no
GPU path). ~6.5 cores active (memory-bandwidth ceiling on k-means).
Expected completion: ~21:50-22:05 UTC / ~03:50-04:05 BDT. Then Step 5
smoke test fires automatically within 120 s of `passages.faiss` +
`passages.pkl` materializing on disk.

### Download-vs-preserve playbook (applies at end of each phase)

**Always download (thesis inputs):**

- `plan_a_outputs.tar.gz` after Step 20 (~400-600 MB compressed) --
  this is produced autonomously by Plan A Step 20. Contains full_run,
  baselines, smoke, cold_start_memory, purity_validation,
  tab_sig_test.csv, and all logs.
- After Phase 1 Full (manual Steps 16-18), manually produce a second
  tar:
  ```bash
  tar czf /workspace/caem/phase1_full_outputs.tar.gz \
      outputs/ablation \
      outputs/ablation/ablation_table.csv \
      outputs/ablation/ablation_per_cycle.csv \
      outputs/ablation/ablation_aggregate_manifest.json
  ```
  scp down (~500 MB).
- Belt-and-suspenders separate scps of a few flat files:
  - `outputs/full_run/experiment_summary.csv`
  - `outputs/tab_sig_test.csv`
  - `outputs/full_run/tab_headline.csv`
  - `outputs/full_run/dataset_splits.json`
  - `outputs/purity_validation/theory_validation.json`
  - `plan_a_runner.log` + `build_passage_index.log`

**Preserve on Vast (do NOT download) while instance is Stopped:**

| Artifact | Size | Why keep | Re-acquire cost if lost |
|---|---:|---|---|
| `data/passage_index/passages.faiss`+`.pkl` | ~17 GB | resume retrieval | $3.50 + 6 h rebuild |
| `hf_cache/` | ~15 GB | avoid re-download | 15-30 min per resume |
| `outputs/full_run/memory_store_cycle_*` | ~2 GB | Step 7 resume | re-run Step 7 ($14, 14-18 h) |
| `outputs/full_run/model_checkpoint_cycle_*` | ~100 GB total | Step 7/18 resume | ditto |
| `outputs/full_run/deferred_buffer_cycle_*.pkl` | ~1 GB | Step 7 resume | lose buffer state (minor) |

**Stop vs Destroy decision rule (by expected pause duration):**

| Pause length | Strategy | Cost math |
|---|---|---|
| < 2 weeks | **Stop** instance, keep disk | $0.15/hr idle * 336 h = ~$50 |
| 2 weeks - 2 months | **Stop** is still cheaper than download + re-upload round trip | ~$72-216 idle |
| > 2 months | **Destroy**, but first download `passages.faiss`+`.pkl` as insurance (~17 GB, 2-4 h on residential) | $0 idle; pay $3.50 + 6 h to rebuild OR re-upload backup |

For Aksan's actual defense timeline (Phase 1A now -> Phase 1 Full
within 1-2 weeks -> defense): Stop is correct. For post-defense
journal prep (months-long pause before Phase 2): Destroy-with-
index-backup is correct.

**Sanity-check command for local "do I have everything" after download:**

```bash
ls -lh outputs_from_vast/outputs/full_run/experiment_summary.csv \
       outputs_from_vast/outputs/tab_sig_test.csv \
       outputs_from_vast/outputs/full_run/tab_headline.csv \
       outputs_from_vast/outputs/purity_validation/theory_validation.json \
       outputs_from_vast/outputs/ablation/ablation_table.csv   # Phase 1 Full only
```

All files non-empty = Ch5 has all inputs it needs.

### 04:35 BDT (22:35 UTC, 2026-04-19) — faiss-cpu vs faiss-gpu: retrospective + plan for Phase 1 Full rebuilds

**Incident**: Step 4's FAISS IVF-PQ clustering phase ran ~80 minutes on
CPU (faiss-cpu, 6.3 cores of useful parallelism before hitting EPYC
memory-bandwidth ceiling). GPU sat at 0% utilization the whole time
because faiss-cpu has no GPU path. Same clustering on a 5090 with
faiss-gpu would have taken 2-4 minutes (~20x speedup).

**Why this happened**: the NEXT_SESSION_PLAN Step 3.2 runbook specified
`pip install faiss-cpu` as the canonical install, chosen at runbook-
write time for "works on any Vast pytorch image" robustness over "optimal
on each GPU class." Not a correctness issue -- index quality is
identical -- but a wall-clock choice that cost 75+ minutes on this run.

**Claude self-correction**: Earlier in this session when the 2M training
warning fired, the assistant described the faiss-gpu speedup as "2-4 min
vs 15 min" -- a significant under-estimate. At 21M passages + 2M training
vectors the CPU path actually takes ~80 min, making the ratio 20x rather
than the quoted 5x. That under-estimate led to characterizing the switch
as "a minor optimization" when it was actually a major one. User flagged
this ("idk why you suggested earlier to use cpu"); accepted the
correction and pushed through the fix below rather than trying to
reinstall mid-run.

**Why we did NOT switch mid-run**: killing the current build would have
discarded 7 h of encoding + 80 min of clustering (~$5.10 spent) and
forced a ~6 h re-encode. Net cost to save the last 10-15 min of this
build: ~14 h round-trip. Not worth it. Letting the current CPU build
finish and patching for NEXT rebuild is the correct trade.

**Patches landed (for Phase 1 Full / Phase 2 / any future rebuild):**

1. `NEXT_SESSION_PLAN.md` Step 3.2 now recommends `faiss-gpu-cu12` as
   the default install, with `faiss-cpu` as a commented fallback. Added
   a runtime check (`hasattr(faiss, "StandardGpuResources")`) so users
   can tell immediately whether they got the GPU variant.

2. `caem/retrieval/rag.py::PassageStore._build_index` — the IVF-PQ
   train+add path now auto-detects faiss-gpu at runtime and routes
   clustering through the GPU via `index_cpu_to_gpu(...)`, then swaps
   back to CPU via `index_gpu_to_cpu(...)` for serialization (FAISS's
   write_index only supports CPU indexes). Falls back to CPU train+add
   silently if faiss-gpu is missing or GPU path raises. Zero behavioural
   change for CPU-only installs -- just adds a fast path for GPU ones.

**Expected impact of the patches on future rebuilds:**

| Phase | With faiss-gpu | With faiss-cpu (current) |
|---|---:|---:|
| Step 4 k-means | 2-4 min | ~80 min |
| Step 7 Tier 3 retrieval queries (~50K calls) | ~50 s total | ~8 min total |
| Steps 11-13 B3/B4/B5 RAG queries (~90K calls) | ~90 s total | ~15 min total |
| **Savings per Plan A rebuild** | — | **~95 min saved** |

**What users should do on next rental:**

```bash
# After Step 3.2 install:
python -c "import faiss; print('gpu:', hasattr(faiss, 'StandardGpuResources'))"
# Prints "gpu: True"  -> good, Step 4 will be ~20x faster
# Prints "gpu: False" -> on CPU variant, budget ~80 min for k-means
```

**No action required on the current run.** Step 4 will complete on
faiss-cpu as-is. The patches take effect on subsequent index rebuilds.

### 04:50 BDT (22:50 UTC, 2026-04-19) — Thread-oversubscription finding (local-agent flag, data-verified, no intervention)

Local agent flagged a possible "1-core effective" collapse on the
running FAISS clustering and proposed killing + restarting. Gathered
hard evidence before acting:

**Findings:**

1. **379 threads confirmed.** `/proc/159638/task/ | wc -l` shows 379
   threads. 1 in R state, 378 in S state (sleeping on locks/futexes).
2. **Machine has 384 cores available** (Vast UI claimed 48 allocated
   but `nproc` and the scheduler see all 384). OMP default spawned
   one thread per hw thread.
3. **Environment is uncapped:** `/proc/159638/environ` shows no
   `OMP_NUM_THREADS`, no `MKL_NUM_THREADS`, no `FAISS_NUM_THREADS`.
4. **CPU utilization is NOT 1-core.** 5 consecutive `ps` samples over
   5 seconds: 562, 562, 562, 562, 562 (very stable). That's **5.6
   effective cores**, not the 1-core the local agent inferred from a
   single top snapshot showing 0% per-thread. The per-thread %CPU is
   instantaneous during a 1-second window; threads rotate work and no
   individual thread is long-resident in R.
5. **Nothing persisted to disk.** `--enable_checkpoint` was not passed
   on the original run (Session 1 used default off). `find` for .npy /
   .faiss / embedding checkpoints returned empty. The 21M x 768
   matrix (~64 GB as float32) lives only in PID 159638's RAM (146 GB
   resident).
6. **Process disk I/O in 99 min of FAISS work: 864 KB total write.**
   read_bytes=0. Confirms the slowness is NOT disk-bound -- it's pure
   CPU + memory bandwidth on in-memory data.

**Decision: let current run finish.**

| Path | Wall-clock | Cost | Risk |
|---|---|---|---|
| Kill + restart with OMP=32 | 6 h re-encode + 30-45 min tuned FAISS = ~6.75 h | ~$4.30 | loses 99 min of FAISS work; re-encode dominates |
| Let current run finish | ~75-115 min remaining | ~$1.00-1.50 | process healthy, R state, no errors |

Let-finish wins by ~5 h and ~$3. Even the worst-case (current run
taking another 5 h) is still cheaper than kill+restart ($3.20 vs
$4.30). No intervention.

**Local agent's "1-core effective / 70-hour ETA" calculation was wrong
by ~5.6x** because it was anchored on the single-snapshot 1-core read
rather than multi-sample aggregate. Rebutted inline with the 562% x 5
sample evidence.

**Correct finding from local agent (kept for future):** thread
oversubscription on uncapped EPYC boxes is a real performance trap,
even if the magnitude was overstated here. Added env-var caps to
`NEXT_SESSION_PLAN.md` Step 3.2B as a runbook requirement.

### Future-proofing patches landed

**`NEXT_SESSION_PLAN.md` Step 3.2B (new sub-step):**

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export FAISS_NUM_THREADS=16
```

Sweet spot for IVF k-means + BLAS-heavy inference is 16-32 threads.
Above that, mutex contention dominates. Capping at 16 is conservative;
users on smaller Vast rentals can keep the same number without
over-subscribing (it's a cap, not a floor). All four env vars set
because different FAISS/BLAS builds respect different ones.

**Combined expected savings on next rebuild** (faiss-gpu + OMP=16):

| Sub-phase | faiss-cpu + 379 threads (this run) | faiss-gpu-cu12 + OMP=16 (next run) |
|---|---:|---:|
| Encoding 21M passages (SBERT) | ~6 h | ~6 h (GPU-bound already) |
| FAISS IVF k-means | ~1h 40min+ | ~2-4 min |
| PQ training + add 21M | ~10-15 min | ~3-5 min |
| Write passages.faiss + .pkl | ~2-3 min | ~2-3 min |
| **Total Step 4** | **~8+ h** | **~6 h 10 min** |

So ~2 hours saved per Step 4 rebuild, dominated by the k-means
fix. If ever faiss-gpu ships a faster encode path too, this drops
further.
