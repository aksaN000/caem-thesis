---
name: B1-B7 baseline implementation audit + fixes (2026-05-15)
description: Audit of scripts/run_baseline.py, scripts/run_simple_ft.py, scripts/rescore_baselines_through_verifier.py against thesis spec. 6 issues found; all fixed in-place with safe defaults + new launch_baselines.sh wrapper. Time estimate ~24h GPU for full panel.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Audit performed 2026-05-15 07:30 BDT** while C4 Step 5 was in flight (4/5 benches done). All fixes are in scripts/ — the live runner uses `caem/` + `scripts/run_experiment.py`, so no in-flight risk.

## Implementation verdict

| Baseline | Class | Spec match | Notes |
|---|---|---|---|
| B1 zero-shot | `ZeroShotBaseline` (eval/baselines.py:357) | ✓ | Bare Qwen, ChatML single-turn |
| B2 CoT | `CoTBaseline` (eval/baselines.py:377) | ✓ | Kojima 2022 "Let's think step by step" |
| B3 RAG | `RAGBaseline` (eval/baselines.py:494) | ✓ | Delegates to `TierThreeRAG`, identical to CAEM Tier 3 |
| B4 CoT+RAG | `CoTRAGBaseline` (eval/baselines.py:581) | ✓ | Subclass of B3 with CoT trigger |
| B5 5-shot CoT | `FiveShotCoTBaseline` (eval/baselines.py:405) | ⚠️ | Demos use heuristic "The answer is X" reasoning (documented limitation; fine for thesis baseline) |
| B6 vanilla FT | `scripts/run_simple_ft.py --use_l2_anchor=False` | ✓ | Full FT, no retention machinery |
| B7 EWC FT | `scripts/run_simple_ft.py --use_l2_anchor=True --use_mmlu_guard=True` | ⚠️ → ✓ | MMLU-only guard (not multi-modal); Ch5 prose updated to match code |

## 6 issues found, all fixed in-place

| # | Issue | Fix | File |
|---|---|---|---|
| 1 | Ch5 §sec:comp-ewc-ft claimed B7 uses "multi-modal retention probe and rollback guard" but code uses MMLU-only | Updated prose: "MMLU-only retention probe... multi-modal probe extension that CAEM uses... deliberate scope choice... contrast accordingly isolates the verification-and-memory machinery plus the multi-modal extension as a single contribution" | `thesis_report/chapters/chapter_5.tex:351` |
| 2 | `run_simple_ft.py --num_cycles` default was 10; CAEM stops at C5 → matched-scale broken | Changed default to 5 (matches C5-stop decision); help text references PRODUCTION_NEXT_SESSION_PLAN.md Phase 1.5 | `scripts/run_simple_ft.py:102` |
| 3 | `run_simple_ft.py --n_eval_per_bench` default was 500; CAEM uses 300/bench via CLI override | Changed default to 300; help text notes the DEFAULT_EVAL_SIZE in benchmark_splits.py is 500 but unused in production trajectory | `scripts/run_simple_ft.py:140` |
| 4 | `run_baseline.py --n_questions` default was 500; CAEM uses 300/bench | Changed default to 300; same help-text rationale | `scripts/run_baseline.py:115` |
| 5 | `run_baseline.py --eval_batch_size` default was 1; CAEM step_7_main uses 32 | Changed default to 32 (~2× speedup; FLARE still falls back to serial due to iterative decoding) | `scripts/run_baseline.py:121` |
| 6 | `rescore_baselines_through_verifier.py --composite_calibration` default pointed to retired `outputs/production/composite_calibration.json` path | Changed default to `outputs/full_run/cycle_3/composite_calibration.json` (matched-protocol pin); help text explicitly retires the production path and documents that the composite is refit per cycle | `scripts/rescore_baselines_through_verifier.py:453` |
| 7 (cosmetic) | `run_simple_ft.py` docstring referenced "Natural Questions" and "six-benchmark panel (3 ID + 3 OOD)"; both stale post-v2.1 | Rewrote docstring to reference v2.1 panel (FEVER, TriviaQA, CSQA training; TruthfulQA, StrategyQA transfer); added B7 retention-guard-scope explanatory paragraph | `scripts/run_simple_ft.py:18-30` |

## New file: launch_baselines.sh (one-shot panel launcher)

`scripts/launch_baselines.sh` — bakes in every correct override so the panel runs at matched-protocol without per-baseline flag-fiddling. Usage:

```
bash scripts/launch_baselines.sh all                # B1-B7 + rescore + stats hint
bash scripts/launch_baselines.sh zero_shot          # B1 only
bash scripts/launch_baselines.sh vanilla_ft         # B6 only
bash scripts/launch_baselines.sh ewc_only_ft        # B7 only
bash scripts/launch_baselines.sh rescore            # post-hoc rescore of all 7
```

Pre-flight: aborts cleanly if `outputs/full_run/cycle_3/composite_calibration.json` doesn't exist (C3 not yet closed) or if `data/passage_index` missing.

## Time estimates

Per-baseline wall-clock on single 5090 with `eval_batch_size=32`:

| Baseline | Generation | Rescoring | Total per-baseline |
|---|---|---|---|
| B1 zero-shot | ~10 min | ~62 min | ~72 min |
| B2 CoT | ~15 min | ~62 min | ~77 min |
| B3 RAG | ~65 min | ~62 min | ~127 min |
| B4 CoT+RAG | ~75 min | ~62 min | ~137 min |
| B5 5-shot CoT | ~20 min | ~62 min | ~82 min |
| B6 vanilla FT | ~2 h | ~62 min | ~3 h |
| B7 EWC FT | ~3 h | ~62 min | ~4 h |
| **Total Phase A + B** | ~7.5 h | ~7.3 h | **~15 h GPU** |
| With overhead + crash recovery margin | | | **~24 h** |

Phase C (statistical analysis + auto-tables, CPU only): ~30 min.

## Caveats applied at fix time

- All defaults changed to v2.1-correct values, NOT preserved-for-backward-compat. Repository is the user's own code, no external dependents, so changing defaults is safer than leaving CLI overrides as the only correct path.
- The retired `outputs/production/composite_calibration.json` path is documented as RETIRED in the rescorer help text; if it still exists on disk, it predates Phase-1d patches and would produce comparison artifacts.
- B5's heuristic demo reasoning ("The answer is X") is documented in the class docstring as a known limitation; flagged in §disc-threats already.
- Phase D (running the actual baseline panel) waits for C5 close + the 5 panel decisions in `outputs/research/topvenue_panel_2026-05-14.md` to be locked by user.

## How to apply

When C5 closes:
1. Verify pre-flights: `ls outputs/full_run/memory_store_cycle_5.faiss && ls outputs/full_run/cycle_3/composite_calibration.json`
2. Kill in-flight runner: `tmux kill-session -t plan_a`
3. Launch panel: `bash scripts/launch_baselines.sh all` (or per-baseline for debugging)
4. After completion, run stats: `python -m scripts.baseline_sig_tests --baseline_dir outputs/baselines`
5. Auto-tables: `python -m scripts.make_tables`
