# CAEM - Next Session Plan
**Updated: 2026-04-11 | Session 35 planning refresh**

---

## Current Verified State

| Item | Status | Notes |
|---|---|---|
| Environment check | DONE | Python and required packages load successfully |
| Quick diagnostics | DONE | `136 passed, 3 warnings` (pre-Session 35) |
| Plan caveat fix 1 | DONE | StrategyQA loader is strict by default (`test` split, no silent train fallback) |
| Plan caveat fix 2 | DONE | Cold-start default benchmarks now match SIL training group |
| **Codebase audit (Session 35)** | **DONE** | 12 issues fixed across 8 files — see impl-log Session 35 |
| **Ablation methodology (Session 36)** | **DONE** | `run_ablation.py` updated: all ablation conditions now evaluated on all 6 benchmarks (was 4). Separates in-domain delta from out-of-domain delta — see impl-log Session 36 |
| FEVER label mapping | FIXED | `_FEVER_LABEL_MAP` corrected: `1→not enough info`, `2→refutes` (EXP-21) |
| FEVER eval split | FIXED | All code now uses `paper_dev` — `"dev"` split name does not exist (EXP-22) |
| TriviaQA/NQ scoring | FIXED | `any_match_em` used for all aliases — not just `gold_answers[0]` (EXP-23) |
| TriviaQA/NQ synthetic samples | FIXED | `make_synthetic_samples` now handles both benchmarks (EXP-24) |
| config.py dead field | FIXED | `benchmark: str = "hotpotqa"` removed from `CAEMConfig` (EXP-25) |
| Stale docstrings | FIXED | All 6 benchmarks documented across `eval/` modules (EXP-26) |
| Test coverage | FIXED | 4 new tests for triviaqa/NQ scoring paths in `test_eval.py` (EXP-27) |
| seed_cold_start HotpotQA branch | FIXED | Returns `[]` with warning instead of downloading (EXP-28) |
| chapter_4.tex | PASS 4+5 DONE | Task-aware prompts, DPR corpus paragraph, em-dash sweep complete |
| Runtime outputs | RESET | `outputs/` intentionally cleared |
| Passage index | READY | `data/passage_index/passages.faiss` and `data/passage_index/passages.pkl` present |

---

## Session 35 Goals

1. Rebuild fresh runtime artifacts from a clean state.
2. Run the full CAEM experiment: 10 cycles, 5000 questions, plan-aligned dataset roles.
3. Run post-experiment analysis scripts (purity + ablation).
4. Prepare final writing inputs for Chapter 5.

---

## Execution Order (Do In This Exact Sequence)

### Phase A - Preflight checks

```bash
python scripts/check_env.py
python -m pytest -q tests/test_pipeline.py tests/test_eval.py
```

### Phase B - Cold-start seeding (plan-aligned)

Target: FEVER + TriviaQA + Natural Questions only.

```bash
python -m scripts.seed_cold_start \
  --target_episodes 200 \
  --benchmarks fever triviaqa natural_questions \
  --output_dir outputs/cold_start_memory
```

Expected artifacts:
- `outputs/cold_start_memory/memory_store.faiss`
- `outputs/cold_start_memory/memory_store.meta`
- `outputs/cold_start_memory/seed_summary.json`

### Phase C - Full 10-cycle CAEM run

```bash
python -m scripts.run_experiment \
  --output_dir outputs \
  --num_cycles 10 \
  --n_questions 5000 \
  --benchmarks fever triviaqa natural_questions truthfulqa strategyqa arc_challenge \
  --passage_index data/passage_index \
  --cold_start_memory outputs/cold_start_memory/memory_store
```

Dataset-role policy during this run:
- SIL training pool: FEVER, TriviaQA, Natural Questions (`train` split)
- Evaluation pool: FEVER (**`paper_dev`**), TriviaQA/NQ (`validation`), TruthfulQA (`validation`), StrategyQA (`test`), ARC-Challenge (`test`)

Important note:
- StrategyQA is strict by default now. If `test` is unavailable/unlabeled, the run should stop instead of silently using `train`.

Expected core artifacts:
- `outputs/all_cycle_results.json`
- `outputs/experiment_summary.csv`
- `outputs/dataset_splits.json`
- `outputs/cycle_1 ... outputs/cycle_10/`
- `outputs/memory_store_cycle_0 ... outputs/memory_store_cycle_10(.faiss/.meta)`
- `outputs/eval/*_cycle{n}.json`

### Phase D - Theory validation

```bash
python -m scripts.run_purity_validation \
  --checkpoints_dir outputs \
  --output_dir outputs/purity_validation \
  --num_cycles 10
```

Expected artifact:
- `outputs/purity_validation/theory_validation.json`

### Phase E - Ablation and baseline analysis

> **BEFORE running this phase:** Fill `published_baselines.template.json` manually.
> All values are `0.0` placeholders — there is NO GPT API, no OpenAI key, no live model calls.
> These are static citations from Liu et al. 2024 (RA-ISF). Only StrategyQA numbers apply:
>
> ```json
> "gpt35_vanilla":  { "strategyqa": 65.2 }
> "gpt35_rag":      { "strategyqa": 64.7 }
> "selfrag_13b":    { "strategyqa": 67.2 }
> ```
> All other benchmark fields stay 0.0 (not reported in the cited paper).

```bash
python -m scripts.run_ablation \
  --caem_results outputs/all_cycle_results.json \
  --cycle3_checkpoint outputs/cycle_10 \
  --cycle0_checkpoint outputs/cycle_0 \
  --output_dir outputs/ablation_results \
  --n_questions 500 \
  --published_baselines_json published_baselines.template.json
```

Expected artifact:
- `outputs/ablation_results/ablation_summary.json`

### Phase F - Optional extended baselines (PUB-04+05 / PUB-06)

```bash
python -m scripts.run_pub0405_variants \
  --base_checkpoint outputs/cycle_10 \
  --memory_store outputs/memory_store_cycle_10 \
  --output_root outputs/pub0405
```

Then:

```bash
python -m scripts.run_ablation \
  --caem_results outputs/all_cycle_results.json \
  --cycle3_checkpoint outputs/cycle_10 \
  --cycle0_checkpoint outputs/cycle_0 \
  --output_dir outputs/ablation_results \
  --n_questions 500 \
  --published_baselines_json published_baselines.template.json \
  --run_pub0405 \
  --pub0405_full_ft_ewc_checkpoint outputs/pub0405/full_ft_ewc \
  --pub0405_lora_l2_checkpoint outputs/pub0405/lora_l2 \
  --pub0405_lora_only_checkpoint outputs/pub0405/lora_only
```

---

## Writing Hand-off After Data Is Ready

Use these files for Chapter 5:
- Main trends and mechanism table: `outputs/experiment_summary.csv`
- Per-cycle benchmark details: `outputs/eval/*_cycle{n}.json`
- Theory validation: `outputs/purity_validation/theory_validation.json`
- Ablation table and MMLU retention: `outputs/ablation_results/ablation_summary.json`
- Calibration values: `outputs/calibration/calibrated_config.json` (if calibration is enabled)

---

## Session Completion Checklist

- [ ] Seeding finished with >= 600 total episodes.
- [ ] Full run completed through cycle 10.
- [ ] `all_cycle_results.json` generated.
- [ ] Purity validation completed.
- [ ] Ablation summary generated.
- [ ] Chapter 5 data extraction checklist prepared.

---

This file is now the authoritative plan for the next execution session.
