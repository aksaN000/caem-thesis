# CAEM — Next Session Plan
**Updated: 2026-04-07 | Session 27 — Implementation Complete, Seeding In-Progress**

---

## Current Status

| Step | Status | Notes |
|---|---|---|
| Gap 1 — Base model check | ✅ DONE | `outputs/base_model_check.json` confirmed. |
| Gap 2 — Passage index (500K) | ✅ DONE | `data/passage_index/` verified (1.5 GB). |
| Smoke Test (Cycle 0→3) | ✅ DONE | Full pipeline verified on local RTX 3060. |
| Codebase Hardening | ✅ DONE | UTF-8 stability, OOM fixes, RAG logic fixed. |
| **Gap 3 — Cold-start seeding** | ✅ **DONE** | 447 verified episodes stored. |
| **Mini Full Run (n=500)** | ⏳ **PROGRESS** | Validation before Lab PC run. |
| **Full Experiment (n=5000)** | ❌ **PLANNED** | Pending Lab PC (4090/5090) access. |

---

## Phase 1: Mini-Experiment (RTX 3060)

The goal is to validate the *entire* 4-cycle loop with real data at a small scale (n=100) to ensure the 14-hour run won't fail.

1.  **Finish Real Seeding**: Complete the `seed_cold_start.py` run with 150 verified episodes per benchmark.
2.  **Auto-launch Mini Run**:
    ```powershell
    # Mini-run (n=100) covering all mechanisms
    python scripts/run_experiment.py `
      --n_questions 100 `
      --benchmarks hotpotqa truthfulqa fever strategyqa `
      --output_dir outputs/mini_experiment `
      --passage_index data/passage_index `
      --cold_start_memory outputs/cold_start_memory/memory_store
    ```
3.  **Verify Results**: Check `outputs/mini_experiment/experiment_summary.csv` for the expected accuracy jump.

---

## Phase 2: Full Scale Execution (Lab PC, RTX 4090/5090)

Once the mini-run is verified, the orchestrator is ready for the definitive 14-hour run.

### Hardware Transition Steps:
1.  **Clone / Pull**: `git pull origin main` on the lab device.
2.  **Apply Scaling**: Update `caem/config.py` with `batch_size=16` and apply the `theta_prev` GPU optimization (see `LAB_PC_SCALING_GUIDE.md`).
3.  **Synchronize Data**: Transfer `data/passage_index/` and `outputs/cold_start_memory/` to the lab PC.
4.  **Run inside `tmux`**:
    ```bash
    tmux new -s caem
    # Cycle 0 baseline + Cycles 1–3 (~11–14h on 4090)
    python scripts/run_experiment.py \
      --n_questions 5000 \
      --benchmarks hotpotqa truthfulqa fever strategyqa \
      --output_dir outputs/full_experiment \
      --passage_index data/passage_index \
      --cold_start_memory outputs/cold_start_memory/memory_store
    ```

---

## Post-Experiment Analysis

After the 14-hour run, execute the following in order:

1.  **Purity Validation**: `python scripts/run_purity_validation.py` (Theorem 1/2/3 verification).
2.  **Ablations**: `python scripts/run_ablation.py` (Verify core mechanisms).
3.  **Calibration Audit**: `python scripts/run_calibration.py` (Verify weight distributions).

---

## Thesis Config Recap (Locked)

| Parameter | Value | Reference |
|---|---|---|
| `n_questions` | 5000 | §5.3 standard |
| `sc_chains_m` | 3 | [LIT] Wang et al. 2022 |
| `se_samples_k` | 10 | [LIT] Farquhar et al. 2024 |
| `n_cycles` | 3 | Thesis Plan §5.2 |
| `max_passages`| 500,000 | Baseline (Lab variant uses 5M) |

---

## All Bugs Resolved (EXP-01 through EXP-13)
*See `caem-implementation-log.md` Session 27 for the final hardening details.*
| ID | File | Problem | Fix |
|---|---|---|---|
| EXP-11 | `eval/metrics.py`, `eval/harness.py` | TruthfulQA EM always 0 | rouge_l with 0.15 threshold |
| EXP-12 | `scripts/run_ablation.py` | AB4 (no reverification) missing | --disable_reverification added |
| EXP-13 | Multiple files | Token limit 128 cuts CoT; no CoT prompting | cot_max_new_tokens=256, CoT induction, extract_cot_answer |
