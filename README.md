# CAEM - Confidence-Aware Episodic Memory with Self-Improvement

CSE400 Final Year Thesis | BRAC University
Student: Aksan Gony Alif

## Current Status

Implementation is complete for the full 8-stage architecture, unit/integration/smoke validation has been done, and the project is in experiment execution mode.

Current run sequence:

1. Implementation complete
2. Unit tests complete
3. Integration tests complete
4. Smoke runs complete
5. Cold-start seeding in progress (running 150 episodes)
6. Next: mini full pipeline run
7. Then: full lab-device run with scaling from `LAB_PC_SCALING_GUIDE.md`

## What CAEM Is

CAEM is a hallucination-reduction architecture built on Flan-T5-Large, with:

- episodic memory for verified QA episodes
- confidence-aware adaptive routing
- multi-signal verification (NLI + self-consistency + semantic entropy)
- iterative self-improvement (fine-tuning over verified memory)
- retroactive re-verification and pruning between cycles

This is a system-level architectural approach, not only a post-hoc detection add-on.

## Core Architecture

Pipeline flow (single query):

1. Encode query embedding
2. Estimate pre-routing confidence `u_pre`
3. Search episodic memory
4. Route to Tier 1, Tier 2, or Tier 3
5. Verify answer quality (Tier 2 and Tier 3)
6. Store only verified novel episodes
7. Repeat across cycles with fine-tuning and retroverification

Tier behavior:

- Tier 1: direct memory recall fast path, no per-query Stage 5 verifier call
- Tier 2: guided generation, then confidence gate and verifier
- Tier 3: RAG generation from passage index, then verifier

Important implementation note:

- Tier 1 intentionally skips per-query verification for latency; quality refresh is handled by retroactive re-verification after cycles.

## Repository Map

```text
caem/
  config.py
  pipeline.py
  confidence/
    pre_routing.py
    post_generation.py
  memory/
    entry.py
    encoder.py
    store.py
  routing/
    router.py
  retrieval/
    rag.py
  verification/
    verifier.py
  training/
    self_improvement.py

eval/
  benchmarks.py
  metrics.py
  harness.py

scripts/
  build_passage_index.py
  check_base_model.py
  seed_cold_start.py
  run_experiment.py
  run_purity_validation.py
  run_ablation.py
  run_calibration.py
  hardware.py

tests/
  test_episodic_memory.py
  test_pre_routing.py
  test_post_generation.py
  test_router.py
  test_verifier.py
  test_rag.py
  test_pipeline.py
  test_self_improvement.py
  test_eval.py
```

## Key Technical Facts

- Embedding model is `sentence-transformers/all-mpnet-base-v2`
- Embedding dimension is 768 (not 384)
- Episodic memory uses FAISS `IndexIDMap(IndexFlatIP)`
- Initial post-generation weights start equal at `0.25, 0.25, 0.25, 0.25`
- Projected `0.20, 0.20, 0.20, 0.40` is a post-calibration target estimate only

## Benchmarks and Metrics

| Benchmark | Metric used in code | Notes |
|---|---|---|
| HotpotQA | EM + token F1 | multi-hop QA |
| TruthfulQA | ROUGE-L (with EM proxy in harness) | offline judge proxy |
| FEVER | label accuracy | supports/refutes/not enough info |
| StrategyQA | EM on yes/no | boolean reasoning |

## Environment Setup

Python setup in this repo is configured for local virtual environment usage (`.venv`) and Pyright targeting Python 3.10.

Install base dependencies:

```bash
pip install torch transformers datasets sentence-transformers faiss-cpu pytest scipy scikit-learn tqdm
```

Optional (faster FAISS on CUDA systems):

```bash
pip install faiss-gpu
```

### VS Code Fresh-Session Bootstrap

Workspace settings are pinned in `.vscode/settings.json` so new sessions use:

- the project interpreter: `.venv/Scripts/python.exe`
- Pylance as diagnostics source
- Pyrefly type errors forced off (for IDEs where the Pyrefly extension is installed)

Run this at the start of each fresh session:

1. Run task: `CAEM: Session Bootstrap`
2. If old squiggles remain: run `Python: Restart Language Server`
3. Then run `Developer: Reload Window`

Task definitions are in `.vscode/tasks.json`:

- `CAEM: Check Env` -- verifies interpreter and key imports
- `CAEM: Quick Diagnostics` -- quick pytest sanity check
- `CAEM: Session Bootstrap` -- runs both tasks in sequence

## Execution Workflow

Use this order for complete runs.

### 1) Build Passage Index

```bash
python scripts/build_passage_index.py --output_dir data/passage_index --max_passages 500000
```

For smoke test:

```bash
python scripts/build_passage_index.py --smoke_test --output_dir data/passage_index_smoke
```

### 2) Base Model Check (Gap 1)

```bash
python scripts/check_base_model.py --output outputs/base_model_check.json
```

### 3) Cold-Start Seeding (Gap 3)

Your current operation (150 episodes):

```bash
python scripts/seed_cold_start.py --target_episodes 150 --output_dir outputs/cold_start_memory
```

This writes:

- `outputs/cold_start_memory/memory_store.faiss`
- `outputs/cold_start_memory/memory_store.meta`
- `outputs/cold_start_memory/seed_summary.json`

### 4) Mini Full Pipeline Run

```bash
python scripts/run_experiment.py \
  --n_questions 500 \
  --benchmarks hotpotqa truthfulqa fever strategyqa \
  --output_dir outputs/mini_experiment \
  --passage_index data/passage_index \
  --cold_start_memory outputs/cold_start_memory/memory_store
```

Note: always pass `--passage_index` explicitly unless your index is at the script default path.

### 5) Resume a Crashed Run

```bash
python scripts/run_experiment.py \
  --resume_from_cycle 2 \
  --output_dir outputs/mini_experiment \
  --passage_index data/passage_index
```

### 6) Post-Run Analysis Scripts

```bash
python scripts/run_purity_validation.py --output_dir outputs/purity_validation
python scripts/run_ablation.py --caem_results outputs/mini_experiment/all_cycle_results.json --cycle3_checkpoint outputs/mini_experiment/cycle_3 --output_dir outputs/ablation
```

## Lab-Scale Run Plan

After the mini run succeeds:

1. Apply the scaling policy from `LAB_PC_SCALING_GUIDE.md`
2. Use lab GPU for the full run
3. Keep thesis-standard and scaled variants clearly separated in outputs

For hardware-specific behavior and precision flags, see `scripts/hardware.py`.

## Testing

Run full tests:

```bash
python -m pytest tests -v
```

Focused runs:

```bash
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_eval.py -v
```

## Main Outputs You Will Use

From `run_experiment.py`:

- `all_cycle_results.json`
- `experiment_summary.csv`
- `dataset_splits.json`
- `memory_store_cycle_0.faiss/.meta` ... `memory_store_cycle_3.faiss/.meta`
- `cycle_1/`, `cycle_2/`, `cycle_3/` checkpoints

From auxiliary scripts:

- `outputs/purity_validation/theory_validation.json`
- `outputs/ablation/ablation_summary.json`
- `outputs/calibration/calibrated_config.json`

## Documentation Pointers

- `caem-implementation-log.md`: chronological implementation decisions and fixes
- `hyperparameter-reference.md`: literature/design/calibrated hyperparameter taxonomy
- `LAB_PC_SCALING_GUIDE.md`: scaling policy for lab hardware
- `NEXT_SESSION_PLAN.md`: execution planning notes
