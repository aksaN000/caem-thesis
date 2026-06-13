# CAEM Phase 2 — Vast Execution Log (B6 + B7 baseline fix)

**Run date:** 2026-06-12 → 2026-06-13 (BDT/UTC mixed in logs)
**Goal:** Fix the two broken external baselines — **B6 (vanilla fine-tune @ cycle 5)** and **B7 (FLARE)** — by regenerating B7 with the format-anchor fix, rescoring both through the locked CAEM C3 verifier, and rebuilding the cross-system CHM table.
**Branch:** `branch/ruc-classifier`. **Composite calibration pin:** `outputs/full_run/cycle_3/composite_calibration.json` (cycle-3, untouched).

---

## 1. Final results (what changed)

### B7 FLARE — regenerated EM/F1 (`outputs/baselines/flare/{bench}_cycle0.json`)
| bench | EM | F1 | Answer-rate |
|---|---|---|---|
| fever | 0.4367 | 0.4367 | 300/300 |
| triviaqa | 0.4167 | 0.4973 | 300/300 |
| commonsense_qa | 0.7233 | 0.7416 | 300/300 |
| truthfulqa | 0.7667 (raw) | 0.3325 | 300/300 |
| strategyqa | 0.6000 | 0.6000 | 300/300 |

- **TruthfulQA canonical = `em_llm_judged` 0.45** (Haiku LLM judge, 300/300). The raw string-match EM (0.7667) over-counts; use the judged value per `feedback_truthfulqa_llm_judge_em`.
- Pre-fix (broken) for contrast: TriviaQA EM was 0.000, FEVER ~0.157 with ~0% format compliance.

### Cross-system CHM (`outputs/baselines/chm_comparison.json`)
| bench | B6 `vanilla_ft@c5` | B7 `flare` |
|---|---|---|
| fever | 0.1308 | 0.2108 |
| triviaqa | 0.1758 | 0.2442 |
| commonsense_qa | 0.0479 | 0.1275 |
| truthfulqa | 0.1904 | 0.0950 |
| strategyqa | 0.0912 | 0.1854 |

All CHMs valid/non-NaN; signals (`p_ground_*`, `p_entail`, `u_stored`) verified real and varied (retrieval working).

---

## 2. What was done, in order

1. **Bootstrap** — torch 2.4.0+cu121 stack + caches. FAISS index (`passages.faiss` 64.5 GB / 21M, `passages.pkl` 7.5 GB) pulled from the public HF dataset `aksaN000/caem-passage-index-21m` (Drive quota-blocked the 64 GB file; HF was used). Benchmark datasets auto-download from public HF.
2. **B7 FLARE regen** (`outputs/baselines/flare/{bench}_cycle0.json`) — required **3 fixes**:
   - `e7ca96b` — 4-part matched-protocol fix (FLAREBaseline inherits `SYSTEM_PROMPT`/`FORCED_PREFIX`; `_look_ahead` uses system+user ChatML; committed buffer seeds `Reasoning:`; loop terminates on `"Answer:"`).
   - `da0cc3f` — **forced-conclusion** fix: on retrieval-poor open benches the loop exhausted `max_sentences=8` without ever emitting `Answer:` (abstain-by-exhaustion, EM dragged to ~0.14); now forces one final short generation appending `Answer:`. Lifted TriviaQA 0.143 → 0.417.
   - **Eager execution** (`TORCHDYNAMO_DISABLE=1`) — `torch.compile` thrashed on FLARE's variable-length generation (15-min hang); eager fixed it. (run_baseline's own disable code didn't take effect.)
3. **B6@C5 verifier rescore** → `outputs/baselines/vanilla_ft/eval/{bench}_cycle5_with_chm.json`.
4. **Phase 2b: B7 FLARE verifier rescore** → `outputs/baselines/flare/{bench}_cycle0_with_chm.json` + final `chm_comparison.json` (both rows).
5. **TQA Haiku rescore** → `em_llm_judged` on `flare/truthfulqa_cycle0.json`.

---

## 3. Gotchas fixed on Vast (so a rerun is clean)

- **`vast_launch.sh` `cmd_rescore_b6` is STALE**: passes `--cycle 5`, but the script wants `--training_cycles 5`. Invoke the script directly (see commands below).
- **PEFT version mismatch**: the cycle-3 adapter was saved with **peft 0.19.1**; the pinned env has **0.13.2**, which rejects newer config fields (`alora_invocation_tokens`, etc., all disabled). A compat copy was made at `outputs/full_run/cycle_3/adapter_compat/` (24 core LoRA fields kept; identical plain LoRA r=32/α=64). **Pass `--checkpoint outputs/full_run/cycle_3/adapter_compat`** to all rescores.
- **anthropic 0.39.0 + httpx 0.28** → `proxies` TypeError. Pin **`httpx<0.28`** (0.27.2 installed).
- **HF cache duplication**: a direct rescore without `HF_HOME` re-downloaded models to `/workspace/.hf_home`; the redundant `/workspace/caem/hf_cache` was deleted to reclaim 12.5 GB. Models now live in `/workspace/.hf_home` — set `HF_HOME=/workspace/.hf_home` for reruns.

### Exact rerun commands (direct, bypassing the buggy launcher)
```bash
cd /workspace/caem
ENV="TORCHDYNAMO_DISABLE=1 HF_HOME=/workspace/.hf_home HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ENABLE_HF_TRANSFER=1"
# B6@C5 rescore:
$ENV python -m scripts.rescore_baselines_through_verifier --baselines vanilla_ft --training_cycles 5 \
   --composite_calibration outputs/full_run/cycle_3/composite_calibration.json \
   --checkpoint outputs/full_run/cycle_3/adapter_compat --batch_size 32
# Phase 2b (flare) + re-aggregate vanilla_ft into chm_comparison.json:
$ENV python -m scripts.rescore_baselines_through_verifier --baselines vanilla_ft flare --training_cycles 5 \
   --composite_calibration outputs/full_run/cycle_3/composite_calibration.json \
   --checkpoint outputs/full_run/cycle_3/adapter_compat --batch_size 32
# TQA Haiku (needs ANTHROPIC_API_KEY in env):
python -m scripts.rescore_truthfulqa --files outputs/baselines/flare/truthfulqa_cycle0.json
```

---

## 4. What was pushed (this commit)
- `outputs/baselines/flare/*_cycle0.json` (5) — B7 EM (truthfulqa incl. `em_llm_judged`)
- `outputs/baselines/flare/*_cycle0_with_chm.json` (5) — B7 CHM
- `outputs/baselines/vanilla_ft/eval/*_cycle5_with_chm.json` (5) — B6 CHM
- `outputs/baselines/chm_comparison.json` — cross-system table
- `outputs/baselines_vast_rerun_logs/` — run logs
- `scripts/PHASE2_VAST_EXECUTION_LOG.md` — this file

Force-added (`git add -f`) because `outputs/` is gitignored; `.gitignore` was NOT modified. NOT pushed: the 64 GB index, models, the staged calibration/adapter, and the B6 `*_cycle5.json` inputs (already on local).

---

## 5. Phase 3 (your local next steps)
1. Pull this commit (or extract the tarball).
2. Regenerate `tab_baseline_pooled` + `tab_sig_test` from the new sidecars + `chm_comparison.json`.
3. Update Ch5 prose with the fixed B6/B7 EM + CHM numbers (TQA = `em_llm_judged` 0.45).
4. No B7 footnote needed — B7 got a real post-fix verifier CHM (Phase 2b ran).
5. Recompile thesis + poster, submit.

---

## 6. Phase 3 integration — DONE (local, 2026-06-13)

Corrected B6/B7 folded into the canonical `outputs/baselines_phase1e/` set and all tables + prose regenerated. Raw pre-overwrite inputs snapshotted to `/tmp/phase2_integration_snapshot_20260613/`.

**Data:** copied corrected B6 `vanilla_ft/eval/{bench}_cycle5_with_chm.json` over the buggy `cycle5.json` (CSQA 0.0 → 0.78 via the "Reasoning:C" re-extract; TQA via Haiku `em_llm_judged`); copied corrected B7 `flare/{bench}_cycle0{,_with_chm}.json`; patched `chm_comparison.json` `vanilla_ft@c5` + `flare` rows; added `caem_cycle5`/`caem_cycle2` CAEM per-bench CHM keys (the sig script's `caem_cycle10` was NaN post-rerun).

**Tables regenerated:** `tab_baseline_pooled.tex`, `tab_sig_test.tex` (C5), `tab_sig_test_c2.tex` (C2), `tab_ces_per_system.tex` (B-flare ACC 0.291→0.525).

**Final integrated numbers:** B6 EM 0.490 / CHM 0.127; B7 EM 0.525 / CHM 0.173; CAEM C5 0.517 / 0.109, C2 0.544 / 0.107. Sig counts: C5 = 9 EM-wins/3 + 21 CHM-wins/3; C2 = 14 EM-wins/3 + 20 CHM-wins/4. Every defeat confined to TruthfulQA vs RAG-free baselines (abstention/retrieval-amplification trade-off). Joint pooled wins: C2 8/8, C5 7/8 (B7 edges C5 on pooled EM by 0.008; CAEM −37% CHM over it).

**Prose reframed to parity + CHM-headline** (per author decision): abstract, Ch5 §comp-baselines + §summary-headline + C2 sig caption, Ch6 conclusion headline; poster two framing blocks + pooled table + wins-against footnote. EM-dominance claims softened to "leads at best cycle, parity with strongest at trajectory end"; CHM dominance retained.

**Enabling fix:** guarded `eval/__init__.py`'s torch-dependent eager imports so `eval.metrics` (McNemar/bootstrap) runs on a torch-less host; no-op where torch is installed. Tables generated with `~/miniconda3/bin/python` (numpy/scipy present; system python3 is bare).
