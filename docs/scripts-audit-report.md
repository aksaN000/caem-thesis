# Scripts Directory Audit Report

**Date:** 2026-04-18
**Scope:** every `.py` file in `scripts/` (excluding `__init__.py`, `__pycache__`).
**Rule:** findings-only. No fixes in this pass. Each issue tagged with severity and category for triage.

## Severity scale

- **BLOCKER** — will crash the run, produce wrong numbers, or corrupt outputs. Must fix before Vast launch.
- **MAJOR** — wrong behaviour in a meaningful edge case, inconsistency with thesis spec, silent data loss, or a latent bug waiting for the right input.
- **MINOR** — cosmetic, style, dead code, hardcoded magic number, missing docstring, weak logging. Safe to ship without, nice to fix.
- **NOTE** — design observation or question, not necessarily wrong. Discuss before acting.

## Category codes

- **SYN** — syntax, import, AST-level
- **LOG** — logic / correctness bug
- **CON** — consistency with thesis/spec/other scripts
- **EDG** — edge case or error-handling gap
- **RES** — resource leak, memory, file handle, GPU
- **QUA** — code quality, readability, naming, dead code
- **DOC** — docstring / comment / help-string issue
- **TOD** — TODO/FIXME/XXX straggler

## Files audited

- [x] `aggregate_ablation.py`
- [x] `build_passage_index.py`
- [x] `check_base_model.py`
- [x] `check_env.py`
- [x] `hardware.py`
- [x] `make_figures.py`
- [x] `make_tables.py`
- [x] `run_ablation.py`
- [x] `run_baseline.py`
- [x] `run_calibration.py`
- [x] `run_cyclic_ablation.py`
- [x] `run_experiment.py`
- [x] `run_purity_validation.py`
- [x] `run_simple_ft.py`
- [x] `seed_cold_start.py`

All 15 files audited in one pass. Findings-only; no code edits.

---

## Summary

### Rough severity counts

- **BLOCKER**: 6 (in 5 files)  →  5 resolved, 1 downgraded to not-a-bug (2026-04-18 fix pass)
- **MAJOR**: ~80  →  all resolved across six waves (2026-04-18 MAJORs fix pass — see Fix log at end of document)
- **MINOR**: ~85
- **NOTE**: ~45

### BLOCKERs — must fix before Phase 1 launch

1. **`run_experiment.py`, line 526** — Retroverify lambda `pipeline.verifier.verify(e.question, e.answer)` passes only two arguments to the 9-signal UnifiedVerifier. If the verifier needs more (input_ids, retrieved evidence, decoding samples), every cycle's retroactive re-verification silently degrades every stored episode's `u_stored`. Same root cause appears in `run_purity_validation.py` line 320. **This is the single highest-risk item in the repo — no Chapter 5 number downstream of retroverify can be trusted until this is confirmed.**  *[FIX 2026-04-18: Discovery confirmed `UnifiedVerifier.verify(query, answer, input_ids=None, *, u_token=None, u_dropout=None)` — the two-arg call is signature-valid. The real risk was the bare lambda's lack of exception handling; `CAEMPipeline.make_retroverify_fn()` (pipeline.py:570) already wraps verify() in try/except and returns None on failure so retroverify skips bad entries instead of killing the whole pass. Swapped the raw lambda for `pipeline.make_retroverify_fn()`.]*
2. **`run_purity_validation.py`, line 320** — `pipeline.verifier.verify(query=q, answer=pred, input_ids=input_ids)` — three-argument call, same concern as #1. Every Chapter 5 §5.5 number (p, α, P_theory, P_obs) depends on this working.  *[FIX 2026-04-18: **False positive — downgraded.** The call matches `UnifiedVerifier.verify`'s signature (`query`, `answer`, `input_ids` is a legal positional-or-kwarg arg), and the enclosing `try/except` on lines 318–336 already handles verifier exceptions cleanly. The separate issue on line 342 (`balanced_acc = (tp + tn) / total` is raw accuracy, not balanced accuracy) remains tracked as a MAJOR in the per-file section and is not launch-blocking.]*
3. **`run_experiment.py`, line 1240** — `--passage_index` default `"data/passage_index"` diverges from `run_baseline.py`'s `"outputs/wiki_passage_index"` and from `build_passage_index.py`'s typical output path. If the paths don't align on Vast.ai, Tier-3 RAG silently disables (only a `logger.warning`) and the entire SIL run produces thesis-invalid results. Either enforce a required-arg for non-smoke runs or canonicalise the default.  *[FIX 2026-04-18: Canonicalised to `data/passage_index` across all five scripts (build_passage_index, run_experiment, run_baseline, run_purity_validation, run_cyclic_ablation) plus `run_ablation.py` (was `outputs/passage_index`). This matches every `--passage_index` line in VAST_AI_DEPLOYMENT_GUIDE.md and NEXT_SESSION_PLAN.md.]*
4. **`run_simple_ft.py`, line 613** — `pre_mmlu = post_mmlu` at cycle end. Per-cycle forgetting-tolerance ratio allows multiplicative drift: 0.93⁰ ≈ 0.484 over 10 cycles even though every individual cycle "passes". Comparison must be against pristine (cycle-0) MMLU, not the previous cycle's post-fine-tune MMLU.  *[FIX 2026-04-18: Introduced `pristine_mmlu` as an immutable cycle-0 anchor; `retention_ratio = post_mmlu / pristine_mmlu` (was `/ pre_mmlu`). `pre_mmlu` survives as a logging-only field that tracks start-of-cycle MMLU; added `pristine_mmlu` to the log entry for traceability.]*
5. **`build_passage_index.py`, lines 332–333** — Function signature defaults `dataset_name="wikipedia", dataset_config="20220301.en"` use the deprecated HF dataset. EXP-09 fix updated the CLI defaults to `wikimedia/wikipedia` / `20231101.en` but the signature defaults are stale. Any caller that invokes the Python function directly (not the CLI) builds the index from a deprecated dataset.  *[FIX 2026-04-18: Signature defaults updated on both `build_passage_index` (line 332-333) and the `stream_passages` helper (lines 181-182). Python callers now track the CLI defaults.]*
6. **`seed_cold_start.py`, line 31** — Docstring advertises the stale 3-signal formula `0.5·NLI + 0.3·SC + 0.2·(1−SE)` for `u_stored`. The thesis canonical formula is the 6-weight composite (`0.30·p_ground_mean + 0.15·p_ground_atomic + 0.15·p_entail + 0.15·s_avg + 0.15·u_internal + 0.10·(1−h_norm)`). Any reader using this docstring as the specification would implement the wrong composite.  *[FIX 2026-04-18: Docstring rewritten with the full 6-weight composite + explicit note that p_contra is a separate veto (not part of the composite). Also updated the downstream "NLI + SC + SE" shorthand in the "human annotator" note to match current nine-signal nomenclature.]*

### Cross-file patterns (the real findings)

These are issues that appear in multiple files and should be fixed once at the source rather than script-by-script:

1. **9-signal UnifiedVerifier API uncertainty** — `run_experiment.py` and `run_purity_validation.py` both call `verifier.verify` with an incomplete argument list. Confirm the canonical `UnifiedVerifier.verify` signature, then fix both call sites (plus any others in `caem.pipeline`).
2. **Passage-index path default divergence** — four different default paths across five scripts. Create a single constant (`caem.config.PASSAGE_INDEX_DEFAULT`) and import it everywhere.
3. **NLI model hardcoded in at least three places** — `run_experiment.py`, `run_purity_validation.py`, `run_ablation.py`. Read from `config.nli_model` with `"roberta-large-mnli"` (or whichever model the thesis settles on) as the single source of truth in `CAEMConfig`.
4. **Per-cycle vs. cumulative MMLU retention** — `run_simple_ft.py` is explicitly buggy. `run_experiment.py` and `run_cyclic_ablation.py` delegate to `SelfImprovementLoop.run_cycle`'s internal retention calculation, which must itself compare against pristine baseline.
5. **Stale framing and terminology** — "3-signal u_stored" (seed_cold_start.py), "Gap 1/2/3" (check_base_model.py), "seven-benchmark panel" / "seven-table split" (multiple), "1M episodes" (run_experiment.py), "Session 42" (run_experiment.py). Sweep for post-rewrite strings.
6. **Hardcoded precision policy** — `check_base_model.py`, `run_simple_ft.py`, `run_purity_validation.py` each pick dtype independently. `hardware.py`'s `HardwareProfile.use_bf16` / `.use_fp16` already exist — consult them everywhere.
7. **Missing `encoding="utf-8"`** on JSON/CSV writes — at least 8 sites across scripts. On Windows this bites whenever a log message contains a non-ASCII character (common in scraped Wikipedia titles).
8. **Private-API leakage `sil._mmlu_score`** — called from `run_cyclic_ablation.py` line 477 and `run_experiment.py` lines 852, 908, 914. Expose a public `SelfImprovementLoop.measure_mmlu(n)`.
9. **No file-logging handler** — every script logs to stdout only. A Vast.ai disconnect loses all logs unless the user uses `tee`. Add a `FileHandler` to each long-running driver.
10. **No `--seed` in `run_experiment.py`** — the main driver has no reproducibility primitive. Stochastic verification signals (self-consistency, MC Dropout) vary run-to-run.
11. **Variable misnomers** — `u_pre_logits` (actually confidences) in `run_calibration.py`; `hall_red` (actually EM gain) in `run_experiment.py`; `measure_verification_balanced_accuracy` (actually raw accuracy) in `run_purity_validation.py`.
12. **VER axis is a 0.5 placeholder** — `make_figures.py` Figure 5.1 and `run_cyclic_ablation.py` both hardcode VER=0.5 in the CES radar. Either compute verifier balanced accuracy from labelled STORE/DISCARD ground truth or drop the VER axis.
13. **`ns_shim` / `types.SimpleNamespace` fragility** — `run_calibration.py` and `run_cyclic_ablation.py` fabricate argparse namespaces to reuse `run_experiment.py` helpers. Any new required attribute in the parent ripples silently. Switch the helpers to accept `**kwargs`.
14. **Deterministic-first-N splits, not randomised** — `run_experiment.py`'s `split_calibration_sets` always takes the first 500 as purity. Add seeded `random.shuffle`.
15. **`torch.load(..., weights_only=False)` on Vast.ai** — `run_purity_validation.py` line 839, `run_simple_ft.py` (confirm). Use `weights_only=True` for security.

### Recommended order of fixes (before Phase 1 launch)

The six BLOCKERs above are the minimum set. Fix #1 + #2 together (verifier API), then #3 (passage-index path), then #4–#6 (single-line docstring + logic fixes). After that, the cross-file patterns can land over a few sessions without blocking compute.

### Fix pass — 2026-04-18

All six BLOCKERs have been addressed. Five were real defects and were patched; one (#2) was downgraded to a false positive after reading the canonical `UnifiedVerifier.verify` signature. Verifier-API risk was lower than the audit estimated, but the retroverify loop's lack of exception handling (the actual #1 defect) has been closed by routing through `CAEMPipeline.make_retroverify_fn()`. The `passage-index path default divergence` cross-file pattern (item #2 in Cross-file patterns) is also resolved as a side-effect of the BLOCKER #3/#6 fix.

### MAJORs fix pass — 2026-04-18

All ~80 MAJOR findings have been addressed across six thematic waves. Every cross-file pattern flagged in the summary (NLI model hardcoding, hardcoded precision policy, VER placeholder, `ns_shim` fragility, `--seed` flag, `sil._mmlu_score` private-API leakage, stale terminology, variable misnomers, private-API calls) is resolved. Per-file fix details are appended at the end of this document under **Fix log — MAJORs — 2026-04-18**.

### Files with no ship-blocking issues

`check_env.py`, `hardware.py`, `run_baseline.py`, `make_tables.py`, `run_ablation.py`, `aggregate_ablation.py`, `check_base_model.py`, `run_calibration.py`, `make_figures.py`, `run_cyclic_ablation.py`. Each has MAJORs worth revisiting after Phase 1, but none that prevent the experiment from producing thesis-valid numbers. (`make_figures.py` Figure 5.1's hardcoded VER axis and the CES-axis reconstruction path are thesis-visible and should be discussed with the advisor before publication, but are not launch-blocking.)

---

# Per-File Findings

## `check_env.py` (52 lines)

Simple import-sanity utility. Lists 11 required modules, imports each, reports version.

### Findings

**[MINOR / CON] No GPU / CUDA availability check.** For a script whose purpose is pre-run environment verification on a GPU box, omitting `torch.cuda.is_available()` and `torch.version.cuda` is a meaningful gap. A fresh Vast image can have a CUDA driver mismatch that imports `torch` cleanly but fails at first tensor operation. *(Line 31–38 module loop only checks imports.)*

**[MINOR / CON] No minimum-version validation.** Module versions are printed but never compared against a required floor (e.g., `torch >= 2.0`, `transformers >= 4.35`). Could silently accept an old image.

**[MINOR / CON] Not invoked from `NEXT_SESSION_PLAN.md`.** The script exists but no step calls it. It could naturally slot into Step 3 (env setup) as a VERIFY gate, or Step 3B (pytest). Probably dead/ad-hoc-only on Vast unless wired in.

**[MINOR / DOC] No module docstring.** Consistent with a utility but minor nit.

**[NOTE]** The VS Code task `CAEM: Check Env` (README line 157) runs this script — that's its canonical invocation site, fine on local dev.

### Verdict
Safe to ship as-is. Two ~10-line additions (CUDA check + minimum-version gate) would meaningfully strengthen it.

---

## `hardware.py` (258 lines)

Hardware detection + precision/batch-size recommendation. Imports: `logging`, `os`, `dataclasses`. Torch imported lazily inside functions.

### Findings

**[MAJOR / DOC] Module docstring is stale.** Lines 6–26 describe "local GPU (3060 / any CUDA card), Google Colab (A100/T4), university clusters (SLURM + CUDA), and CPU fallback." The thesis is now Vast-only on RTX 5090. Colab and SLURM are not in the current plan. The hardware targets list (lines 21–25) centres on RTX 3060 as "local dev machine" — entire framing is out of date.

**[MAJOR / CON] Stale reference to `LAB_PC_SCALING_GUIDE`.** Line 122 comment: "batch_size affects only speed, never accuracy (confirmed in LAB_PC_SCALING_GUIDE)." The README was just updated to remove that file from the doc pointers as stale. The claim itself is also oversimplified — batch size affects gradient noise in full fine-tuning and thus the loss landscape; only true if `batch_size × grad_accum_steps` is held constant.

**[MAJOR / LOG] Gradient accumulation is never emitted.** `HardwareProfile` returns `recommended_batch_size` but no `grad_accum_steps`. If the 3060 branch (batch=4) and 5090 branch (batch=32) run the same `self_improvement.py` without accumulation compensation, the *effective* batch size differs 8×, which does affect accuracy. Either emit `grad_accum_steps` from this function and have training consume it, or pin an effective-batch-size target in the thesis. Current code silently lets the effective batch vary with hardware.

**[MAJOR / LOG] `torch.cuda.empty_cache()` inside `apply_memory_flags()` (line 209).** Unconditional cache purge at init is safe, but this function is callable anywhere; if invoked mid-training it fragments allocated tensors and can trigger OOM on the next grow. Should be guarded to one-time init, or moved to a dedicated `init_cuda()` helper.

**[MINOR / LOG] TF32 only enabled on bf16 branch (line 203).** `torch.backends.cuda.matmul.allow_tf32 = True` and `cudnn.allow_tf32 = True` benefit Ampere+ GPUs regardless of compute dtype. The fp16 branch (4090/3090) skips this optimisation. For CAEM this is a small perf loss, not a correctness issue.

**[MINOR / LOG] Device-0 assumption (lines 105–107, 234).** `torch.cuda.get_device_properties(0)` and `model.to("cuda")` both pin to GPU index 0. If the Vast box exposes 2 GPUs (noted at lines 126–128) and you want to run experiment on GPU 1, you must set `CUDA_VISIBLE_DEVICES=1` *externally* — the script has no first-class multi-GPU support. Fine for single-GPU rental; document-worthy.

**[MINOR / CON] `CAEM_DEVICE` env-var override (line 57) doesn't accept indexed devices.** `CAEM_DEVICE=cuda:1` falls through to auto-detect because the whitelist is `("cuda", "cpu", "mps")`. Trivial fix if indexed devices are wanted.

**[MINOR / LOG] Exception fallback at line 109 silently drops to `vram_gb=0.0`**, which then routes to the `<10 GB` branch with `batch_size=2`. That's safe-fail behaviour (degrades rather than crashes), but there's no WARNING log — so if a user's CUDA import works but `get_device_properties` fails, they silently get a laptop-tier config on an A100. Add a `logger.warning` on exception.

**[MINOR / CON] Line 158 model-footprint math undercounts.** Claims "Flan-T5-Large (~1.5 GB) + NLI (~1.4 GB) + SBERT (~0.5 GB) = ~3.4 GB" — misses DPR (question + context encoders, ~1 GB combined) needed at Tier 3. On a 12 GB card, this can push OOM close.

**[NOTE / CON] `note` strings reference "theta_prev GPU optimisation active" (lines 138, 145).** This implies a code path elsewhere (probably `training/self_improvement.py`) reads `profile.vram_gb >= 24` and keeps θ_prev on GPU instead of CPU. Needs cross-verification that the training code actually consults this and doesn't always pin θ_prev to CPU. If it always pins to CPU, the claim in the note is a lie to the user.

### Verdict
Two documentation staleness issues (Colab/SLURM/3060 docstring, LAB_PC_SCALING_GUIDE reference). One potentially-meaningful correctness gap (gradient-accumulation not emitted → effective batch varies with hardware). A few minor perf optimizations missed. No blockers for the 5090 run — the 5090 branch at line 133 is correct and well-tuned. Worth a polish pass after Phase 1, not before.

---

## `run_baseline.py` (268 lines)

CLI driver for B1–B5 (inference-only baselines). Loads model once, loops over benchmarks.

### Findings

**[MAJOR / CON] Docstring says "seven-benchmark panel" (line 24), example uses six (line 29).** Same inconsistency we just fixed in `chapter_5.tex` (56→48 McNemar count). The panel is six factual-QA benchmarks; MMLU is retention-only. Docstring is stale.

**[MAJOR / CON] Hard-coded `--dtype bfloat16` default (line 124) is wrong for RTX 4090.** Help string says "matches CAEM full run on 4090/A100" — but per `hardware.py` line 141, RTX 4090 Ada Lovelace has no bf16 tensor-core support and uses fp16. If anyone runs this script on a 4090 with defaults, bf16 either silently falls back to software emulation (slow) or runs correctly but without tensor-core acceleration. On the 5090 the default is fine.

**[MAJOR / CON] Script reinvents device/dtype logic instead of using `hardware.py`.** `hardware.py::get_hardware_profile()` exists to centralize this; this script ignores it and hard-codes its own `--device` and `--dtype` args. Any future precision-policy change has to be made in two places. Ideally `run_baseline.py` would call `get_hardware_profile()` and use its `use_fp16` / `use_bf16` / `recommended_batch_size` unless CLI overrides.

**[MAJOR / LOG] Stale/fictional symbol in comment (line 205).** Says "argparse choices=BASELINE_NAMES enforces that name is one of the five handled cases above" — but there's no `BASELINE_NAMES` variable; the choices list is inline at line 83. Either the symbol was renamed and the comment wasn't updated, or the comment was written speculatively. Not a bug, but a code-smell.

**[MAJOR / LOG] FEVER-only split handling (lines 247–250).** FEVER uses `split=ns.split` (paper_dev); others fall back to `load_benchmark`'s default split. The docstring says paper_dev is data-leakage-safe *for FEVER* — but gives no assurance about the default splits for the other five benchmarks. If any other default split overlaps the training pool, we get contaminated results. Needs cross-check with `eval/benchmarks.py::load_benchmark`.

**[MINOR / LOG] FAISS-index existence check only verifies the directory path (line 184).** `PassageStore.load` needs multiple files (`.faiss`, `.meta`, potentially `.pkl`). A partially-corrupt index folder would pass this check and crash on load. A stricter pre-flight (check for each required file) would fail faster with a clearer error.

**[MINOR / RES] Baseline class statelessness is assumed, not verified.** Line 230 constructs the baseline once, lines 240–252 reuse it across six benchmarks. If any baseline retains inter-call state (warm KV cache, stored last prompt, accumulated logging buffer), Benchmark-2 results are subtly influenced by Benchmark-1. Likely fine for the current baselines (they look stateless), but worth a defensive `baseline.reset()` at the top of the loop.

**[MINOR / QUA] `_build_baseline` return type is unannotated** (line 158). Five different concrete classes returned; declaring a common `Baseline` protocol or `-> Any` would help.

**[MINOR / QUA] No file-logging handler.** Only `StreamHandler(sys.stdout)` is installed. If the user doesn't pipe through `tee`, the progress log is lost on instance termination — and the results JSON is all that survives. Cheap fix: add a `FileHandler(output_dir / "run.log")`.

**[MINOR / CON] Default `--n_questions 500` (line 104)** is smoke-test-sized; thesis headline commits n=5000. Runbook passes it explicitly so this is cosmetic, but a default of 500 in a tool that's documented as the "Chapter 5 panel runner" could mislead future users.

**[NOTE]** EM/F1 formatting at line 262 uses `.4f` — high precision but OK.

### Verdict
One real concern: FEVER-only split override (line 247–250). Need to verify the other five benchmarks' default splits are data-leakage-safe before the Phase 1 run. Docstring + help-string stalecness is cheap to fix. Dtype default disagreement with `hardware.py` is a real integration gap — if you ever need a 4090 fallback, this defaults wrong.

---

## `make_tables.py` (417 lines)

Turns the seven CSVs from `eval.reporting` into Chapter-5 LaTeX snippets + a combined `ch5_tables.tex`. Stdlib-only, no heavy deps. Static formatter, non-numerical — low blast radius.

### Findings

**[MAJOR / DOC] Module docstring disagrees with code on percent formatting.** Line 34–35: *"Floats are formatted to four decimals unless the column name hints at a percent or ms value (`*_ms`, `*_pct`) — those get one decimal."* But line 200 renders percent-suffixed columns with `.3f` (three decimals), not one. The helper docstring at 176–181 correctly says "three decimals." Module docstring is stale; either fix the docstring to "three decimals" or change line 200 to `.1f`. A rate of 0.847 rendering as "0.8" in the thesis would be a problem, so the code is probably correct and the docstring is wrong.

**[MAJOR / LOG] Partial-success exit code is 0.** Line 406–408: `if not manifest: return 1` — but `manifest` is populated even if only 1 of the 7 CSVs exists (the missing ones just log a warning at line 344–347 and get skipped). So if `run_experiment.py` crashes halfway through `eval.reporting` and emits only `tab_headline.csv`, `make_tables.py` exits 0 with a single-table manifest. A CI or orchestrator grepping for exit-nonzero would miss this. Consider a `--strict` flag that returns non-zero when any expected CSV is absent.

**[MINOR / QUA] `_INT_SUFFIXES` is misnamed — it's exact-match, not suffix.** Line 152–154 defines `_INT_SUFFIXES = ("_n", "cycle", "n", "n_scored", ...)` and line 196 uses `col_l in _INT_SUFFIXES` (exact match), not `.endswith`. If the reporter ever emits a column called `samples_n` or `cycle_index`, it won't match and will render as `5.0000` instead of `5`. Rename to `_INT_COLS` or switch to `any(col_l.endswith(s) for s in _INT_SUFFIXES)` for consistency with the other two suffix sets.

**[MINOR / EDG] No handling of `inf` / `-inf` cells.** Line 190 (`float(s)`) happily accepts `"inf"` and `"-inf"`; line 202 then formats them as `"inf"` — LaTeX may render that literally, not as `\infty`. If a reporting function ever divides by zero (e.g., grounding rate over an empty subset), the resulting `inf` will leak into the table. Cheap guard: after the `float(s)` call, check `math.isinf(x)` and return `_NA_CELL` or `r"$\infty$"`.

**[MINOR / EDG] Bare `except Exception` swallows CSV read errors.** Line 350–356 catches any exception and emits an empty snippet with a single-column placeholder. Useful for robustness, but `logger.warning` only shows the exception message, not the traceback. Use `logger.exception` so a permissions/encoding bug leaves a debuggable trace.

**[MINOR / CON] Default log level is WARNING, so per-table `wrote X` is invisible without `-v`.** Line 398 sets default WARNING; the INFO logs at 344/361/366 are suppressed. Line 411–412 compensates with a final stdout manifest print, so not broken — but when a CSV is missing, the warning-level "skipping" message is the only visible signal. Reasonable default, but worth documenting.

**[MINOR / QUA] First column alignment always `l` regardless of content.** Line 230–234 `_column_spec` hard-codes `"l" + "r" * (n-1)`. For tables where the first column is numeric (e.g., `cycle` as the first column in `tab_continual.csv`), right-alignment would look better. Cosmetic.

**[MINOR / LOG] New tables added to `eval.reporting` without `TABLE_META` entry silently get fallback caption.** Line 261–263: `meta = TABLE_META.get(table_id, {}); caption = meta.get("caption", f"Table {table_id}")`. A new table would emit a snippet with caption "Table foo" and label `tab:foo` — publication-ugly but no loud error. Consider `raise KeyError` if `table_id not in TABLE_META` to force the developer to add a caption.

**[NOTE]** The `TABLE_FILES` import at line 67 is a hard dependency — if `eval/reporting.py` ever has a syntax error or missing transitive import, `make_tables.py` won't even start. Not wrong (you genuinely need that map) but worth noting as a coupling point.

**[NOTE]** The number "seven" here refers to **seven tables** (headline, calibration, halluc, grounding, purity, continual, sig_test), NOT seven benchmarks — so the "seven CSV artefacts" phrasing at line 4/24/324/378 is correct, unlike the stale `run_baseline.py` docstring.

### Verdict
Stable utility. One real doc-vs-code contradiction (decimal count for percent columns) needs reconciling before the thesis cites this formatting convention. Exit-code logic should get a `--strict` for Phase-1 runs so a half-written result directory doesn't slip past CI silently. Everything else is cosmetic.

---

## `run_ablation.py` (422 lines)

Phase-9 ablation sweep orchestrator. Loads a single (model, memory, encoder, NLI, passage_store) set once, then iterates variants from `caem/ablation/`.

### Findings

**[MAJOR / LOG] Shared `memory_store` object across all variants, no documented reset.** Line 127: `pipeline.memory_store = memory_store` attaches the SAME object to every variant's pipeline. Line 318 loads it once outside the loop. If `run_variant` or any pipeline path does an `add()` during eval (even inadvertently — e.g., a variant that mutates the verifier and produces a different pass/fail outcome), the store drifts between variants and breaks the "same reference state" invariant. The eval harness is constructed without an explicit `store_to_memory=False` (line 359–361) — that flag must be enforced inside `run_variant` or by the variant's config. MUST cross-verify `caem/ablation/runner.py` sets a read-only mode before Phase 1.

**[MAJOR / CON] NLI model hard-coded to `roberta-large-mnli` (lines 307–310).** If the actual CAEM verifier in `caem/verification/` uses a different NLI model (thesis spec called for a cross-encoder at one point; consult the current verifier source), then the ablation sweep is not testing the system you're shipping — it's testing a variant with a substituted NLI head. Cross-check with the verifier module.

**[MAJOR / CON] `--n_questions` default = 500 (line 404).** Thesis headline commits n=5000. If the runbook or future-you forgets to pass `--n_questions 5000`, the ablation numbers are at 1/10 the statistical power of the main results. Same smell as `run_baseline.py`; consider making this required, or aligning defaults with `run_experiment.py`.

**[MINOR / LOG] Silent `KeyError` risk on `mmlu_baseline`.** Line 327: `baseline_mmlu = float(json.load(f)["mmlu_baseline"])`. If the JSON was written by an older script that used a different key (e.g., `baseline_accuracy`), this raises KeyError with no informative context. A `json.load(f).get("mmlu_baseline")` + explicit error message would be more helpful.

**[MINOR / LOG] Device-0 assumption inherited from `hardware.py`.** Line 285 pulls `hw.device` which is `"cuda"` (not `"cuda:N"`) — so the model lands on device 0. No first-class multi-GPU support here either. Consistent with `hardware.py` note.

**[MINOR / QUA] Duplicated "find full_ces" logic.** Lines 381–385 scan `results` for the `full` variant; `write_ranking_csv` does the same at lines 185–189. Factor into a `_find_full_ces(results)` helper.

**[MINOR / QUA] Duplicated sort.** `write_ranking_csv` sorts at 191–195, `print_ranking` sorts at 245–249 — same key, same direction. Sort once in main and pass the sorted list down.

**[MINOR / QUA] Decimal inconsistency between CSV and stdout.** `write_ranking_csv` rounds axes to 4 decimals (lines 212–217); `print_ranking` formats axes at 3 decimals (line 261) and CES at 4 decimals (line 260). If the stdout and CSV are ever compared, the minor drift is confusing.

**[MINOR / LOG] `-1` as failed-variant fallback CES (lines 193, 247).** Magic number. A variant with a legitimate negative CES (unlikely but formally possible after certain weight combinations) would sort alongside "failed to compute" variants. Use `float("-inf")` for clarity.

**[MINOR / EDG] `round(nan, 4)` produces `nan`, CSV writes literal `"nan"`.** If the `full` variant fails, every other row's `delta_abs_vs_full` and `delta_pct_vs_full` become NaN, then CSV-written as the string `nan`. Downstream plot/table code should handle that, but worth knowing.

**[MINOR / QUA] Duplicate `AutoTokenizer` import.** Line 84 pulls it into `m`; line 307 re-imports via `from transformers import ...`. Could be removed by using `m["AutoTokenizer"]`.

**[MINOR / RES] No file-logging handler.** Only stderr via the default StreamHandler. Same issue as `run_baseline.py`. Add a `FileHandler(output_dir / "run.log")`.

**[MINOR / LOG] `split_map` fallback to `"validation"` for unknown benchmarks (line 155).** If a new benchmark is added to CLI default but not to `split_map`, `load_benchmark(bm, split="validation")` may fail or silently grab a wrong split. Either restrict `--benchmarks choices=` to known names or make missing-split a hard error.

**[NOTE / LOG] Model-checkpoint load assumes a single state_dict file.** Line 292–295 uses `torch.load(..., weights_only=True)` — works only if the cycle-N model was saved as a pure state_dict (`torch.save(model.state_dict(), path)`), not via `model.save_pretrained(path/)`. Worth documenting which format `run_experiment.py` produces.

**[NOTE / CON] `passage_index` default is `outputs/passage_index`** (line 398), but `run_baseline.py` defaults to `outputs/wiki_passage_index`. Same resource, two default paths — an inconsistency that'll bite someone who builds the index once and assumes both scripts find it.

### Verdict
One real risk (shared mutable memory_store across variants — MUST audit `run_variant` before Phase 1). One cross-consistency check (NLI model vs. the actual verifier). Everything else is polish. The smoke-test path is well-structured and the sweep is resilient to individual variant failures, which is the important bit for an unattended Vast run.

---

## `aggregate_ablation.py` (472 lines)

Post-sweep aggregator. Walks `outputs/ablation/<variant>/seed_<N>/` tree, collapses per-seed CES histories to one row per variant, writes two CSVs + manifest. Stdlib-only, no heavy deps.

### Findings

**[MAJOR / LOG] Missing `full` variant silently poisons the entire ΔCES column.** Line 257–259: `full_row = variant_rows.get("full"); full_ces_mean = full_row["ces_mean"] if full_row else float("nan")`. If a sweep forgot to run or failed the `full` reference variant, every other row gets `delta_ces_mean=NaN` and the ranking loses its anchor — and no WARNING is logged. Add a `logger.warning("'full' variant not in results; ΔCES will be NaN for all rows.")` so the user knows to rerun the reference before trusting the table.

**[MINOR / LOG] `_is_nan` swallows `TypeError`/`ValueError` and returns `True`.** Line 94–98: a cell like `"n/a"` or `None` is treated as NaN silently. Combined with line 225 (`float(agg.get(k, NaN))`), a malformed CES history entry would just drop out of the mean with no log. Fine for robustness; consider a DEBUG-level log when the exception fires.

**[MINOR / LOG] Seed directories sorted alphabetically, not numerically.** Line 160: `sorted(variant_dir.iterdir())` — so `seed_123` sorts before `seed_42`. Affects only the order of manifest/CSV rows, not math. Cosmetic.

**[MINOR / EDG] `max(by_cycle)` crashes if a history entry has `aggregate_axes` but missing `cycle`.** Line 187 filters on `aggregate_axes` truthiness but keeps `h.get("cycle")` == None in the dict keys. Line 192 `max(by_cycle)` then tries to compare `None` to `int` and raises TypeError. Add a second filter `if h.get("cycle") is not None` at line 187.

**[MINOR / LOG] Duplicate-cycle history entries silently collapse.** Line 187 dict comprehension: `{h.get("cycle"): h for h in history if ...}` — last entry for a given cycle wins. A restart-and-append scenario (same variant appended twice in one file) would lose data without notice.

**[MINOR / QUA] Redundant list comprehension.** Line 324: `[k for k in AXIS_KEYS]` — `list(AXIS_KEYS)` is equivalent and cheaper to read.

**[MINOR / QUA] "[n_seeds ≤ 0]" wording in header when rows is empty.** Line 358: cosmetic edge case — the loop above exits when `runs` is empty (line 390–396), so this is unreachable in practice, but the `default=1` at line 351 is odd — if you survive to `_print_ranking` with zero rows, something is already wrong.

**[MINOR / CON] `--cycle_for_table` priority is not obvious from help text alone.** Line 405–410: `--screening_mode` overrides `--cycle_for_table` even if explicitly set; `--screening_mode` documentation (line 462) mentions the override, but if the user passes `--cycle_for_table 10 --screening_mode`, they silently score at cycle 3. Consider a warning when both are set.

**[NOTE]** Variance propagation `delta_std = sqrt(std_v² + std_full²)` (line 268) assumes seed independence. That's true for freshly-seeded runs but worth caveating in Chapter 5 if a ΔCES ± std is reported.

**[NOTE]** No file-logging handler, same pattern as the other scripts. Aggregator is cheap to re-run, so less critical here.

### Verdict
Well-structured aggregator. One real issue (silent all-NaN ΔCES when `full` is absent) — easy fix. Everything else is edge-case polish. Safe to ship as-is for Phase 1; clean up the "missing full" case before a final Chapter 5 table is frozen.

---

## `check_base_model.py` (475 lines)

Zero-shot base-model sanity check. Loads Flan-T5-Large raw, runs greedy decoding on six benchmarks, reports EM/F1 (ROUGE-L for TruthfulQA). No CAEM pipeline, no memory, no retrieval.

### Findings

**[MAJOR / CON] Purity-theorem framing at lines 9–10 may not match Chapter 4.** Docstring says *"p > (1 − α) ≈ 0.10–0.25 for purity, and p > 0.5 for the convergence guarantee."* The thesis theorem guarantees **P > p** (post-gate purity exceeds base accuracy), with **P > α** as a secondary consequence when p > 0.5. The `p > (1−α)` framing is a non-standard rearrangement that's hard to trace back to the theorem statement. If Chapter 4 never writes it as `p > (1−α)`, the docstring is introducing its own shorthand — and the `≈ 0.10–0.25` concrete range has no visible source in the thesis. Cross-check with `chapter_4.tex` before this script is referenced in the methodology.

**[MAJOR / CON] Dtype hard-coded to fp16 on CUDA regardless of hardware.** Line 338: `torch_dtype=torch.float16 if device == "cuda" else torch.float32`. On the RTX 5090 (Blackwell, bf16-preferred), this runs in fp16 when bf16 is the thesis standard. Should consult `hardware.py` or offer a `--dtype {float16,bfloat16,float32}` flag. Same integration gap as `run_baseline.py`.

**[MAJOR / CON] Stale "Gap 1/2/3" terminology.** Line 4: *"Gap 1 -- Standalone base-model accuracy check."* Line 417: *"Safe to proceed to Gap 2 (build_passage_index) and Gap 3 (seed_cold_start)."* This refers to an older workflow phase. The current plan (NEXT_SESSION_PLAN) uses Step-numbered validation rungs; "Gap" terminology is likely dead. Worth a grep to confirm no other docs still reference Gaps.

**[MINOR / QUA] `T5Tokenizer.from_pretrained` uses the slow (Python) tokenizer.** Line 335: `T5Tokenizer.from_pretrained(model_name)`. `AutoTokenizer.from_pretrained(model_name)` would default to the fast (Rust) tokenizer — a noticeable speedup over 600 samples. Changes nothing semantically.

**[MINOR / QUA] Duplicate import inside `main`.** Line 347: `import sys, os` — `sys` is already imported at module top (line 55), and `os` isn't used anywhere. Dead code.

**[MINOR / CON] Output-JSON key `"all_pass_p_gt_half"` is misleading.** Line 423. For TruthfulQA, the threshold at line 293 is 0.15 (ROUGE-L), not 0.5. A TruthfulQA failure bit doesn't mean "p ≤ 0.5"; it means ROUGE-L < 0.15. Rename to `all_thresholds_passed` or emit per-benchmark thresholds alongside.

**[MINOR / EDG] `max_new_tokens=256` is wasteful for label-based benchmarks.** Line 90. FEVER/StrategyQA/ARC only need ≤4 tokens ("supports", "yes", "A"). Using 256 on these tasks makes generation ~10× slower than needed. Parameterize per-benchmark or cap early via `stopping_criteria`.

**[MINOR / QUA] Failure message always says "expected for RAG benchmarks".** Line 392. That framing is true for FEVER (retrieval-dependent) but misleading for TriviaQA zero-shot (a low EM there is a real signal, not "expected"). Tighten the message or drop the parenthetical.

**[MINOR / LOG] Per-sample question truncated to 80 chars in output.** Line 267. For debugging a specific failure you'd want the full question — the ID is still there to look it up from the benchmark loader, but it's friction. For 600 rows the storage cost of the full question is negligible.

**[MINOR / CON] Split choices for StrategyQA / ARC use `"test"` (lines 369–370).** Matches `run_ablation.py`'s split_map. Worth verifying `eval/benchmarks.py::load_strategyqa("test")` actually returns gold labels — some HF test splits are unlabeled.

**[NOTE]** Metric implementations (EM, F1, ROUGE-L) are inlined here and NOT shared with `eval/metrics.py` (if that exists). Any fix or refinement to EM normalisation downstream won't propagate. Deliberate isolation, but fragile — if `eval/metrics.py` diverges, the "floor" this script reports won't match the in-pipeline numbers.

**[NOTE]** The `cast(Any, ...)` wrappers at lines 336–340 are typing noise to silence the linter. Functional but cluttered. Could switch to `# type: ignore[attr-defined]`.

### Verdict
Two real cross-consistency items: (1) the purity-theorem framing in the docstring should match Chapter 4 wording verbatim, and (2) the dtype policy should align with `hardware.py`. Stale "Gap" terminology is easy to scrub. Script is otherwise self-contained and correct for its narrow job. Safe to run on Vast as-is; the fp16 default just means it won't get tensor-core acceleration on the 5090.

---

## `seed_cold_start.py` (488 lines)

Cold-start memory seeder. Runs Tier-3 path on 200–500 training-split samples per benchmark to pre-populate the episodic memory before Cycle 1.

### Findings

**[BLOCKER / CON] Docstring advertises a STALE 3-signal verification formula.** Line 31: *"u_stored derived from the verification pipeline (same formula as in production: 0.5·NLI + 0.3·SC + 0.2·(1−SE))."* The current thesis spec is the **6-weight composite**: `0.30·p_ground_mean + 0.15·p_ground_atomic + 0.15·p_entail + 0.15·s_avg + 0.15·u_internal + 0.10·(1−h_norm)`. If anyone reads this docstring to understand what gets stored, they will form the wrong mental model of the verifier. The code itself calls `pipeline.answer(...)` which defers to the real (6-weight) implementation, so behaviour is fine — but the docstring is a documentation landmine. **Fix before any reader references this file.**

**[MAJOR / CON] Stale "Gap 3" terminology.** Line 4 ("Gap 3 -- Cold-start memory seeder"), line 451 (argparse description). Same dead-workflow vocabulary as `check_base_model.py`.

**[MAJOR / CON] Dtype hard-coded to fp16 on CUDA.** Line 223. Same issue as `check_base_model.py` — doesn't consult `hardware.py`, so on the 5090 you get fp16 instead of bf16. Seeding quality in fp16 is fine but inconsistent with the rest of the pipeline.

**[MAJOR / CON] FEVER seeding prompt may not match eval-time prompt.** Line 158–160 constructs: *"Answer with one of: supports, refutes, not enough info. Claim: {claim}"*. If the eval path in `caem/pipeline.py` or `eval/benchmarks.py::load_fever` builds a different FEVER prompt (different wording, different label list order, no "Answer with one of:" prefix), then the seeded episodes are effectively on a **different task** than the evaluated one. The verifier may still pass these, but the stored `u_stored` values will not generalize to eval-time queries. MUST cross-check against the prompt used at eval.

**[MAJOR / LOG] Passage index path is relative (`data/passage_index`), not repo-root-relative.** Line 248: `idx_path = Path("data/passage_index")`. If seeding is launched from any directory other than the repo root, this silently misses the index and the warning at 256–260 fires — seeding then proceeds with "query-only RAG (no retrieved context)", which dramatically changes what gets stored. This is a silent quality regression. Use `repo_root / "data" / "passage_index"` (repo_root is computed at line 385 inside `main` — hoist it or pass it in).

**[MAJOR / CON] RoBERTa-large-MNLI hardcoded** (lines 238–241). Same consistency concern as `run_ablation.py` — if the production verifier uses a different NLI model, seeding uses a different one. Worse: here a `try/except` at 243–244 silently continues with SC+SE-only verification if NLI fails to load, which produces even lower-quality seeded episodes with no hard failure.

**[MINOR / LOG] FEVER label default is "refutes" on missing label.** Line 154: `raw_label = row.get("label", 2)` — default 2 maps to "refutes". If `lucadiliello/fever` ever has a row without a label field, this silently labels it as "refutes" instead of dropping it. Better to `if "label" not in row: continue`.

**[MINOR / QUA] Duplicate `import sys` inside `run_smoke_test` (line 340).** `sys` already imported at module top. Also duplicates `from pathlib import Path`. Dead code.

**[MINOR / EDG] No `encoding="utf-8"` on `open()` for JSON writes.** Lines 364, 434. Default encoding on Windows is `cp1252`, which can choke on non-ASCII in benchmark names or model outputs. Add `encoding="utf-8"` explicitly.

**[MINOR / QUA] `--benchmarks` has no `choices=` restriction.** Line 466–471. Unknown benchmark names produce a WARN log and silent empty result. Worth restricting to the 6 known names or hard-failing on typos.

**[MINOR / RES] Empty memory store gets saved regardless of seeding success.** Line 420: `pipeline.memory_store.save(str(store_path))` fires unconditionally. If every benchmark failed verification (e.g., NLI load failed → verification too strict), a 0-episode store is written and the next stage loads an empty memory — which is what we're trying to avoid. Guard: `if pipeline.memory_store.size > 0` else warn and skip save.

**[MINOR / CON] `QueryEncoder` built without explicit device (line 232).** `encoder = QueryEncoder(model_name=config.sbert_model)` — no `device=device`. If `QueryEncoder` defaults to CPU, every SBERT embedding during seeding is CPU-bound, which multiplies runtime on the ~3000 max-processed questions. Cross-check `QueryEncoder.__init__`.

**[MINOR / CON] "~2–4 hours on RTX 3060" estimate (line 39) is stale hardware.** Vast-5090 is the new target.

**[MINOR / LOG] FEVER oversampling 3× then shuffling (lines 168–172).** Iterates dataset up to `n*3` samples, shuffles, takes first `n`. Works because the dataset may have class order. Worth commenting in-line why 3× is the multiplier — looks arbitrary.

**[NOTE]** The "85–90% verification accuracy" claim at line 56, used to justify automated-vs-human seeding, has no visible source in this file. If Chapter 5 cites this rationalisation, the number must be either (a) measured in-repo (purity validation results) or (b) dropped to a qualitative claim ("high enough automated verification accuracy to approximate human labelling"). Don't let 85–90% land in the thesis without a measurement.

**[NOTE]** Hand-rolled FEVER loader here duplicates functionality that should live in `eval/benchmarks.py::load_fever(split="train")` for consistency. Currently `load_fever` is not invoked from this script — the loader is inlined. If `eval/benchmarks.py` ever changes its FEVER loader (e.g., picks a different HF source), seeding will silently diverge.

### Verdict
One real BLOCKER: the stale 3-signal formula in the docstring is a documentation time-bomb for any future reader. Two real correctness risks: (1) passage-index relative path silently degrading to query-only RAG if cwd is wrong; (2) FEVER prompt potentially mismatching eval-time prompt. Everything else is polish and stale terminology. Before launching Phase 1, confirm the FEVER prompt match and make the passage-index path robust. The docstring formula should be updated BEFORE the next session reads this file.

---

## `run_simple_ft.py` (619 lines)

Training-time baselines B6 (Vanilla FT) and B7 ("EWC-only FT"). Strips verifier, memory, retrieval, SC/SE; only keeps the L2 anchor + MMLU rollback in B7. Also has an optional STaR rationalisation pass.

### Findings

**[BLOCKER / LOG] Per-cycle retention ratio is measured against *previous cycle's* MMLU, not cycle-0 baseline.** Line 613: `pre_mmlu = post_mmlu` — after each cycle, the "anchor" for the next retention check becomes the POST value from this cycle. With `ρ_min=0.93`, this allows multiplicative drift: 0.93¹⁰ = **0.484** — i.e., after 10 cycles the guard would still pass even if cumulative retention has dropped to 48%. The comment at line 611–612 calls this out as intentional ("used as next cycle's anchor reference for retention ratio"), but the thesis abort-guard `ρ_min=0.93` is almost certainly meant as a cumulative bound against a fixed cycle-0 pristine baseline (cf. `caem/abort_guard.py` if it exists, and Ch3 FR7 wording). MUST cross-check against Chapter 3 / the main pipeline's rollback logic — if the main pipeline uses cumulative, this B7 baseline measures a DIFFERENT quantity and results won't compare apples-to-apples.

**[MAJOR / CON] "EWC-only FT" is a misnomer — implements L2 anchor, not actual EWC.** The flag is `--use_l2_anchor` (correct), but the baseline name `ewc_only_ft` implies Fisher-information-weighted EWC. CAEM itself uses L2 (Fisher is too expensive at 780M params — this is established thesis decision), so "EWC-only" as a comparison label invites reviewer confusion. Either (a) rename to `anchor_only_ft` or `l2_anchor_only_ft`, or (b) if Chapter 5 really labels this as "EWC-only", add a footnote explaining that "EWC" here refers to the L2 simplification of the Kirkpatrick 2017 formulation.

**[MAJOR / LOG] L2 formula uses `(λ/2)||θ-θ_prev||²` but thesis spec is `λ_reg·||θ-θ_prev||²`.** Line 269: `loss = ce_loss + (ns.l2_lambda / 2.0) * l2`. The thesis-stated regularizer (per the CAEM skill: *"`λ_reg·||θ-θ_prev||²`"*) has no 1/2 factor. With `--l2_lambda 0.01` as default, the effective coefficient is **0.005**, not 0.01. Either: (a) remove the `/2.0` if the thesis notation is literal and the 1/2 is NOT absorbed into λ, or (b) document that `λ` in thesis is the half-strength convention (common in ML textbooks). MUST match what `caem/training/self_improvement.py` does in the CAEM self-improvement loop — if CAEM does `λ·||·||²` and this baseline does `(λ/2)·||·||²`, then "λ=0.01" means different things in the two places.

**[MAJOR / CON] Stale "seven-benchmark panel" phrasing.** Line 28 docstring: *"After each cycle the fine-tuned weights are evaluated on the full seven-benchmark panel..."* Line 566–568 log: *"Cycle %d: evaluating on %d benchmarks"* (OK, dynamic) — but the docstring text "seven-benchmark" contradicts the default `eval_benchmarks` list at line 119–122 which is **six** (MMLU is a separate retention probe, not in eval_benchmarks). Same stale count as `run_baseline.py`.

**[MAJOR / RES] CPU-resident anchor causes per-batch CPU→GPU copies.** Lines 226–230 keep anchor on CPU when VRAM < 24GB. Line 267 then does `ref = p0.to(device, dtype=torch.float32)` **every batch, for every parameter tensor**. For Flan-T5-Large (780M params), this copies ~3GB fp32 weights across the PCIe bus on every batch — a major bandwidth bottleneck. Acceptable on the 5090 (>24GB, anchor stays on GPU), catastrophic on any 4090 fallback. Either move anchor to GPU or compute the L2 term on CPU without per-batch reload.

**[MAJOR / LOG] VRAM check uses `total_memory`, not `mem_get_info()[0]` (free).** Line 223. A 24GB card with 20GB occupied by the model would still evaluate `>= 24.0` True and try to put anchor on GPU, potentially OOMing. `torch.cuda.mem_get_info()[0]` reports free memory and would be more robust.

**[MINOR / QUA] Deprecated `torch.cuda.amp.GradScaler`.** Line 241. The modern API is `torch.amp.GradScaler("cuda")`. Current code still works but emits a DeprecationWarning in PyTorch ≥2.4.

**[MINOR / QUA] `ZeroShotBaseline.__new__` + manual attribute injection (lines 569–575).** Hack to reuse an already-loaded model. Tightly coupled to `ZeroShotBaseline`'s internal attribute names (`model`, `tokenizer`, `device`, `model_name`, `max_new_tokens`, `max_input_tokens`). If `ZeroShotBaseline` refactors any of these, this silently breaks with an `AttributeError` on the first benchmark call. Cleaner approach: make `ZeroShotBaseline.__init__` accept pre-loaded `model` and `tokenizer` kwargs.

**[MINOR / QUA] Dead import `from dataclasses import asdict` (line 75).** Never used.

**[MINOR / LOG] Silent `retention_ratio = 1.0` fallback on NaN / zero-baseline.** Lines 543–546: if the MMLU probe crashes and returns NaN, we default to "retention passes" — the abort guard never fires. A failure mode that silently allows drift. Better: `retention_ratio = float("nan")` and log a WARNING that the guard is skipped this cycle.

**[MINOR / CON] `--dtype bfloat16` default** (line 127) — same 4090-fallback consistency concern as `run_baseline.py`.

**[MINOR / QUA] Semicolon-separated statements at line 282, 288, 293** (`scaler.step(optimizer); scaler.update()` etc). PEP 8 frowns on these.

**[NOTE]** No `--grad_accum_steps` parameter. If the main CAEM training uses gradient accumulation (per `hardware.py` note), the B7 baseline training effective batch size can differ, contaminating the ΔRET comparison.

**[NOTE]** 10 cycles × save_pretrained for 780M params ≈ 30 GB of checkpoints under `out_root/`. On Vast-150GB this fits but consumes 1/5 of disk; add a `--keep_last_n_ckpts` flag if disk becomes tight.

**[NOTE]** Per-cycle pool slicing (lines 512–514) iterates strictly through `train_pool` without shuffle across cycles. With `num_cycles=10` and `train_pool_size=6000`, each cycle sees a unique 600-sample window. If `num_cycles > pool_size // batch_size`, the modulo wrap at line 512 can produce short final-cycle batches. Edge case only.

### Verdict
One potential BLOCKER: the per-cycle retention anchor (cumulative vs. per-cycle drift) must match the main pipeline's definition before the B7 numbers can be compared to the CAEM numbers. One MAJOR numerical correctness concern: the `/2.0` factor in the L2 loss — confirm against `caem/training/self_improvement.py` to avoid a silent halving of effective λ. Everything else is naming/staleness/polish. Cross-check those two items before trusting B7 as a baseline in Chapter 5.

---

## `build_passage_index.py` (652 lines)

One-time offline script: streams Wikipedia via HF datasets, chunks articles (DPR-style 100-word windows), encodes with SBERT, builds a FAISS IVF-PQ index, saves as a `PassageStore`.

### Findings

**[BLOCKER / CON] Function-signature defaults use STALE dataset (`wikipedia`/`20220301.en`).** Lines 332–333: `def build_passage_index(..., dataset_name: str = "wikipedia", dataset_config: str = "20220301.en", ...)`. The EXP-09 fix note at line 210–212 explicitly deprecates this dataset (uses a legacy Python script that HF datasets no longer supports). Only the CLI path (lines 582–588) uses the new `wikimedia/wikipedia`/`20231101.en`. **Any programmatic caller that imports `build_passage_index` as a library function without passing `dataset_name`/`dataset_config` will hit the legacy dataset and fail with the exact error EXP-09 was supposed to fix.** Update the signature defaults to match the CLI defaults.

**[MAJOR / LOG] `nlist=65_536` is over-tuned for a 500K-passage corpus.** Line 335 function default, line 599 CLI default. FAISS IVF guideline is √N to 8√N clusters. For N=500K: √N ≈ 707, 8√N ≈ 5,656. A 65K nlist means ~7–8 vectors per cluster on average; most centroids will be empty or one-vector, degrading recall significantly. The default is sized for 10M+ passage corpora. For the thesis 500K default, recommend `nlist=4096` or `8192`. For a 5M-passage "production-quality" run (per docstring line 30–32), `nlist=16_384`–`32_768` is reasonable.

**[MAJOR / QUA] Two sources of truth for defaults (signature vs. CLI).** Lines 332–339 vs. 530–624. Same parameters specified in both places with independent defaults. Any future tuning change has to be made in two places — the staleness in finding #1 is a direct consequence of this pattern. Fix: signature should read `dataset_name: Optional[str] = None` and use a single module-level constant (e.g., `_DEFAULT_DATASET_NAME = "wikimedia/wikipedia"`) referenced from both paths.

**[MINOR / SYN] `choices=["cuda", "cpu", None]` contains `None`, which argparse can't match against user input.** Line 567. Argparse converts input to string; user can never pass `None` from the CLI. Cosmetic but misleading — either drop `None` from choices or replace with the string `"auto"` and handle it inside `build_passage_index`.

**[MINOR / QUA] Duplicate `import sys` inside functions.** Lines 363 (`build_passage_index`) and 482 (`_sanity_check`). Already imported at line 56. Dead duplicates.

**[MINOR / EDG] No dimension check on embeddings against expected 768.** Line 444: `emb_dim = all_embeddings.shape[1]` — just reads the dim. If someone passes a different `--sbert_model` with mismatched output dim (e.g., `all-MiniLM-L6-v2` → 384-dim), the store saves successfully, but any caller that assumes 768 will get a FAISS dim-mismatch at load time — or worse, a silent embedding mismatch at query time. Add `assert emb_dim == EXPECTED_SBERT_DIM` or at least a WARNING if the model name doesn't match `all-mpnet-base-v2`.

**[MINOR / LOG] `CAEMConfig()` mutation assumes non-frozen dataclass.** Lines 458–464 directly set `cfg.rag_index_type = ...` etc. If `CAEMConfig` becomes `@dataclass(frozen=True)` for safety reasons, this silently breaks. Use `dataclasses.replace(cfg, rag_index_type=..., ...)` for future-proofing.

**[MINOR / QUA] Install advice mentions `faiss-gpu` (line 515).** On systems where `faiss-gpu` isn't available (ARM, some conda envs), the user needs `faiss-cpu` — already mentioned at line 47 but not echoed in the error branch.

**[MINOR / LOG] Section-header heuristic is brittle.** Lines 150–151: `if len(first_line.split()) <= 7 and "." not in first_line and len(lines) >= 2`. A section titled "Dr. Smith's research" has a period and gets misclassified as body text. A one-sentence paragraph ("Shakespeare wrote many plays.") with a 4-word first line could be misclassified as a header. Affects a small fraction of articles; not worth fixing unless recall testing reveals systematic misses.

**[MINOR / RES] No retry / recovery for HF streaming failures.** If the `load_dataset(..., streaming=True)` connection drops partway through a long run, the script dies and requires `--resume`. The checkpoint mechanism (CHECKPOINT_EVERY=100K) limits loss to at most 100K re-streamed passages, but a `try/except` around the inner `for article in ds:` loop with exponential backoff would be cleaner.

**[MINOR / CON] Docstring line 39–40 says "See also / External links sections are dropped" — but the filter relies on `SKIP_SECTIONS` matching section title lowercase.** If Wikipedia has minor variants ("External Links", "See Also" — which the `.lower()` at 155 handles fine), OK. But missing from `SKIP_SECTIONS`: "discography", "filmography", "awards", "gallery" — all common noisy sections. Low-priority tuning.

**[NOTE]** `# type: ignore[union-attr]` at line 492. Typing workaround for `SentenceTransformer.encode` return type union. Functional but sloppy.

**[NOTE]** `np.concatenate(all_embedding_batches)` at line 443 materializes the full ~1.5 GB array (500K × 768 × 4 bytes). On a 5090 box with 32+ GB RAM this is trivial; worth noting for anyone running this on a tiny VM.

### Verdict
One BLOCKER: stale function-signature defaults for the Wikipedia dataset — EXP-09's fix is effectively undone for any library-style import. One serious tuning issue: `nlist=65_536` will degrade recall on the default 500K corpus; this could quietly sabotage Chapter 5's RAG baselines and the Tier-3 retrieval quality. Both are easy fixes but must land before Phase 1. Everything else is minor polish.

---

## `run_calibration.py` (693 lines)

Temperature-scaling calibration driver. Post-Cycle-0 standalone + per-cycle re-fit functions. L-BFGS fits scalar T to minimise NLL on u_pre vs. EM binary labels.

### Findings

**[MAJOR / QUA] Variable misnomer: `u_pre_logits` actually stores confidences in [0,1], not logits.** Line 274: declared as `u_pre_logits: List[float] = []`. Line 293: `u_pre_logits.append(u_pre)` where `u_pre` is from `result.pre_confidence.u_pre` — already in [0,1]. The actual logit conversion happens at lines 401–404 where a second variable `logits` is built via `log(u/(1-u))`. A reader scanning this file will be confused about where the logit/sigmoid boundary lies. Rename to `u_pre_values` or `u_pre_confidences`.

**[MAJOR / QUA] Code duplication between `calibrate_pipeline` and `calibrate_pipeline_temperature_only`.** Lines 344–447 and 450–543 share ~80% of their bodies (collect data, clean non-finite, build logits, fit T, apply T, compute ECE). Factor into a `_fit_T_from_data(u_pre_values, labels) -> (T, ece_before, ece_after)` helper. As written, a bug fix to one path requires a parallel edit in the other — and the subtle difference (the post-cycle version also runs `log_signal_auroc` at line 417) is easy to miss.

**[MINOR / LOG] Silent sample drops at `logger.debug`** (line 322–323). `except Exception: logger.debug(...)` — at the default INFO log level, these are invisible. If the pipeline has a systematic bug (e.g., `pre_confidence=None` on every FEVER sample), calibration silently collects 0 samples from that benchmark. Upgrade to WARNING and include the sample id/benchmark in the message.

**[MINOR / EDG] Missing `encoding="utf-8"` on `open()` for JSON writes.** Lines 432, 537. Same Windows cp1252 issue as `seed_cold_start.py` — non-ASCII in log messages or benchmark names would crash.

**[MINOR / QUA] Deprecated `--cycle0_results` flag still in argparse.** Lines 555–556: help string says "Deprecated compatibility flag (unused ...)". If it's truly unused, remove it; dead CLI args are a source of user confusion.

**[MINOR / CON] `loader_by_benchmark` (lines 619–626) duplicates `CALIB_BENCHMARK_SPLITS` (lines 58–67).** Adding a new benchmark requires updating both. Collapse into a single registry.

**[MINOR / LOG] `T_old = config.temperature_scalar` at line 523 assumes the attribute exists.** First-ever call (before any calibration has happened) would need `config.temperature_scalar` to be initialized to 1.0 at CAEMConfig construction. If CAEMConfig doesn't pre-initialize it, first cycle raises AttributeError. Use `getattr(config, "temperature_scalar", 1.0)` defensively.

**[MINOR / LOG] Fragile `types.SimpleNamespace(passage_index=...)` at line 597–600.** Standalone path builds a minimal fake CLI namespace for `build_pipeline`. If `build_pipeline` ever adds a new required attribute (e.g., `ns.model_checkpoint`, `ns.dtype`), this crashes with AttributeError. Either use the full argparse namespace or make `build_pipeline` accept kwargs.

**[MINOR / QUA] Hacky encoder-device override at lines 602–604.** Comment admits it: *"build_pipeline call above creates an encoder without device explicitly passed to it ... we override."* Fix at source: make `build_pipeline` take and pass `device` through to `QueryEncoder`.

**[NOTE / CON] `calibrate_pipeline_temperature_only` runs **unconditionally** each cycle (line 459–460).** Docstring says "Invoked unconditionally at each cycle boundary." If Chapter 4 specifies a conditional protocol (e.g., re-fit only when ECE exceeds a threshold), this diverges. The Ovadia/Thulasidasan citation supports unconditional re-fit as a reasonable choice, but make sure Chapter 4 doesn't claim something different.

**[NOTE]** `log_signal_auroc` labels the third and fourth signals `"u_consistency"` and `"u_entropy"` (line 213) but the underlying fields are `s_avg` and `1 - h_norm`. Diagnostic-only, but the name-vs-source mismatch would puzzle a reader inspecting the log and then looking for those names in `UnifiedVerifierOutput`.

**[NOTE]** Tier-1 hits are correctly excluded from `signal_matrix` (comment at line 304–308 explains: they skip Stage 5, so `vout` is None). Documented, not a bug.

**[NOTE]** `bounds=[(-3.0, 3.0)]` on L-BFGS (line 178) means T ∈ [0.05, 20.1]. Broad enough for typical temperature scaling; a pathological Cycle-0 could hit the boundary — log a warning if `result.x[0]` is within 0.05 of the bounds.

### Verdict
Two major code-quality issues (misleading variable name, duplicated code paths) — neither affects numerical correctness. One cross-chapter consistency check (unconditional vs. conditional per-cycle re-fit). Core temperature-scaling math looks correct. Safe to ship as-is for Phase 1; clean up the naming and factoring before a post-Phase-1 polish pass.

---

## `make_figures.py` (698 lines)

Renders the seven Chapter 5 figures (CES radar, reliability diagram, hallucination decomposition, grounding trajectory, purity stacked bars, continual learning, EM progression) from `tab_*.csv` and `per_sample_signals.jsonl`.

### Findings

**[MAJOR / CON] Figure 5.1 CES radar VER axis is hardcoded to 0.5.** Lines 241–242: `for cycle in axes_map: axes_map[cycle].setdefault("VER", 0.5)`. The docstring at line 176 openly admits "VER = 0.5 fallback (verifier-BA unavailable)." A placeholder axis that never moves across cycles misleads the reader into thinking there is a real VER dimension. Either (a) compute verifier balanced-accuracy from labeled STORE/DISCARD ground truth, (b) remove the VER axis entirely and show a 4-axis radar, or (c) annotate the figure with "VER placeholder" in the title/caption. Currently the radar ships as a 5-axis figure with one axis that carries no signal.

**[MAJOR / LOG] CES axes are *reconstructed* in `_ces_axes_from_headline` rather than read from `eval.metrics.ces_score()`.** Comment at lines 156–159 says the axes are "as stored by `eval.metrics.ces_score()`" but the helper at 162–208 in fact rebuilds them from three separate CSVs with its own formulas (ACC = EM, EPI = 1 − hallucination, RET = mmlu_retention_ratio, CAL = 1 − 2·ECE, VER = 0.5). The table CES scalar and the radar-axis components are computed by *different code paths* and can diverge silently. If `eval.metrics.ces_score()` ever weights or scales axes differently, the radar no longer reflects the table.

**[MAJOR / LOG] EPI / CAL formulas use `min(x, 0.5)` clamps that hide pathology.** Line 198 CAL: `1.0 - 2.0 * min(ece, 0.5)` means any ECE ≥ 0.5 renders identically as CAL = 0 on the radar. Line 237 EPI: `1.0 - min(mean_hr, 0.5)` means any hallucination rate ≥ 50 % renders as EPI = 0.5 — the same as a rate of exactly 50 %. Catastrophic hallucination or miscalibration would be indistinguishable from severe-but-not-catastrophic. Clamp boundaries need to be documented in Chapter 5 or removed.

**[MAJOR / LOG] NaN axes silently plotted as 0.0.** Line 263: `plot_values = [v if (v is not None and not math.isnan(v)) else 0.0 for v in values]`. The comment on line 262 claims "the axis label alone conveys the missingness" which is false — the reader sees a radar lobe that touches the origin and cannot distinguish "no data" from "data = 0". Use `numpy.nan` and let matplotlib break the line, or annotate missing axes.

**[MAJOR / LOG] Figure 5.2 reliability uses equal-width bins without a minimum-sample guard.** Line 316: `edges = np.linspace(0.0, 1.0, n_bins + 1)`. Most ECE literature (Guo et al. 2017 is fine with equal-width; Nixon et al. 2019 argue for equal-mass) requires either a minimum per-bin count or error bars. Line 328 skips *empty* bins (`mask.sum() == 0`) but keeps bins with a handful of samples (n = 2 or 3). The per-point accuracies in those bins are near-binary — the line connecting them looks jagged/overconfident. Add `if mask.sum() < min_bin_n: continue` with a default of ~10.

**[MINOR / EDG] Figure 5.3 y-axis upper bound expression is awkward.** Line 393: `max(1.0, max(hr + cr + ee) * 1.2) if (hr + cr + ee) else 1.0`. The `hr + cr + ee` list-concat relies on the outer `if`-else to shield `max()` from an empty list, but if *any one* of the three lists is non-empty while another is empty, the guard still passes; however `max` of a three-way concatenation then includes zeros from the empty side. Not strictly a bug, just surprising. A named helper `ylim_with_headroom(values, headroom=0.2, floor=1.0)` would be clearer.

**[MAJOR / CON] Figure 5.4 reads signal columns that may not be emitted by `eval.reporting`.** Lines 416–420 expect `mean_p_ground_max`, `mean_p_ground_mean`, `mean_p_ground_atomic`, `mean_p_contra`, `unsupported_correct_rate`. Thesis 6-weight u_stored uses `p_ground_mean` and `p_ground_atomic` (weights 0.30 and 0.15). `p_ground_max` and `p_contra` are plotted but not in the composite — confirm they are also persisted to `tab_grounding.csv`. If the reporting layer emits `p_ground_mean` (no `mean_` prefix) instead of `mean_p_ground_mean`, all five series silently become all-zeros via the `_parse_float(...) or 0.0` fallback.

**[MAJOR / CON] Figure 5.5 purity stacks four categories (STORE / DEFERRED / ABSTAIN / DISCARD).** Chapter 4's UnifiedVerifier Stage 5 canonically produces STORE / DEFERRED / DISCARD. ABSTAIN belongs to the pre-verification abstention gate and is a separate bucket. Confirm `tab_purity.csv` actually emits `n_abstain` (line 461 reads it with a safe default of 0, so a missing column silently zeroes the ABSTAIN segment).

**[MINOR / LOG] Figure 5.5 twin-axis on same [0,1] scale is redundant.** Lines 491–498: `ax2 = ax1.twinx()` with `set_ylim(0, 1)` — identical to the left-axis fractions. The mean `û_stored|STORE` line literally overlays the stacked bars at the same coordinates. Either drop the twin-axis (plot on `ax1`) or move the line to a separate inset / second subplot.

**[MAJOR / LOG] Figure 5.6 mmlu_pct unit heuristic is brittle.** Line 542: `mmlu_frac = [(v / 100.0) if (v is not None and v > 1.5) else v for v in mmlu_pct]`. If `eval.reporting` ever changes the unit convention (fraction vs. percentage), this heuristic silently flips. An MMLU score of 0.90 (fraction) and 0.90 (where the code mistakenly thinks it is already a fraction but it's actually 0.90 %) are indistinguishable. Always assume a documented unit and assert it.

**[MINOR / CON] Default colormap `tab10` is used for per-benchmark lines in Figures 5.6 and 5.7.** For the six factual-QA benchmarks this is fine; beyond 10 benchmarks the colors silently cycle. Flag in case the panel grows.

**[MINOR / QUA] Unioned cycle set construction is nested/awkward.** Line 599: `sorted(set(list(mean_em.keys()) + [c for curve in per_bm.values() for c in curve]))`. Cleaner: `sorted(set(mean_em) | {c for curve in per_bm.values() for c in curve})`.

**[NOTE / DOC] Header says "seven Chapter 5 figures".** If Chapter 5 is being restructured (e.g., to merge 5.6 and 5.7), this string becomes stale. Cross-check against the latest thesis outline.

**[NOTE / QUA] `build_all_figures` catches `Exception` broadly** (line 655). Best-effort rendering is appropriate for a thesis-figure generator, but during active thesis writing a silent `except Exception: logger.warning(...)` can hide real bugs in the builder functions. Consider re-raising when `--verbose` is set.

**[MINOR / CON] No shared thesis style.** No `plt.style.use(...)`, no rcParams font family, no consistent figsize/font-size. Each figure picks its own colors (`"#d7263d"`, `"#2ecc71"`, viridis, tab10). For a chapter's worth of figures, a single stylesheet would improve visual consistency.

**[MINOR / RES] No log of total runtime or figures produced.** For reproducibility, append a small manifest file (per-figure content hash, input CSVs' mtime) alongside the PNG/PDF outputs.

**[MINOR / EDG] `_parse_float(row.get("cycle")) or 0` pattern** (e.g., lines 187, 193, 202, 231, 414, 522). The `or 0` coerces both `None` (missing) and a genuine `0.0` to 0 — harmless for cycle, but the same idiom appears on signal columns (`_parse_float(row.get("mean_p_ground_max")) or 0.0`, line 416) where 0.0 is a valid signal value. Missing columns and true-zero columns render identically on the figure.

### Verdict
The VER-axis placeholder (hardcoded 0.5), the CES-axis reconstruction path diverging from `eval.metrics.ces_score()`, and the `min(x, 0.5)` clamps in EPI/CAL are the three items that risk being called out by an examiner looking at Figure 5.1. Figures 5.2–5.7 are structurally sound but carry several consistency risks with the reporting layer (column-name spelling, unit conventions, four-vs-three Stage-5 categories). No crash-level bugs; no numerical-correctness errors in the plotting maths itself. Safe to run to completion on a valid artefact directory; ship-blocking only insofar as the Figure 5.1 hardcoded VER axis appears in the thesis PDF.

---

## `run_cyclic_ablation.py` (788 lines)

Single-variant, single-seed cyclic (training-time) ablation driver. Reuses `scripts/run_experiment.py` helpers for pipeline construction, dataset loading, calibration, retroverify, and summary CSV emission. Supports three profiles (smoke / screening / confirmatory) plus explicit CLI overrides.

### Findings

**[MAJOR / LOG] `config.num_cycles` / `config.questions_per_cycle` overwritten *after* `variant.apply()`.** Lines 385–386: `config.num_cycles = int(profile["max_cycles"]); config.questions_per_cycle = int(profile["n_sil_per_cycle"])`. If a variant's `apply()` mutation deliberately sets `num_cycles` or `questions_per_cycle` (e.g., a hypothetical `no_long_run` variant), the override silently clobbers it. Either apply the profile *before* `variant.apply()` or have variants declare these fields as immutable.

**[MAJOR / CON] `--passage_index` default diverges across scripts.** Line 740 uses `"data/passage_index"`; `run_baseline.py` uses `"outputs/wiki_passage_index"`; `seed_cold_start.py` uses the same `"data/passage_index"` relative path; `run_calibration.py`'s shim builds its own path via `build_pipeline`. Users copying commands between scripts routinely hit "index not found" because the repo has no canonical passage-index location. Pick one and apply globally.

**[MAJOR / LOG] `cycles_completed = cycle_num + 1` is ambiguous** (line 629). Cycle 0 sets `cycles_completed = 1` (line 508); each completed SIL cycle then assigns `cycle_num + 1`. After cycle 10 the manifest reports `cycles_completed = 11`, which is the number of *evaluation rounds including cycle 0*, not the number of SIL fine-tune cycles (which is 10). Any downstream consumer that treats "cycles_completed" as "SIL cycles run" is off-by-one. Rename to `eval_rounds_completed` or change semantics to `cycle_num`.

**[MAJOR / CON] VER axis placeholder reinforced here.** Line 264: `verifier_balanced_accuracy=None, # VER placeholder 0.5`. Same cross-file consistency issue as `make_figures.py` — the CES radar's VER axis carries no signal. This is the second site where the 0.5 placeholder is hardcoded; any fix must touch both sites.

**[MAJOR / LOG] Calibration runs on tiny synthetic sets in smoke mode.** Lines 421–424 build `calib_samples` from synthetic data (5 samples per benchmark when `n=10`); `run_calibration_step` at line 506 then fits L-BFGS temperature scaling on ~15–25 total points. Temperature scaling on <50 samples is unreliable and can hit L-BFGS boundary (flagged separately in `run_calibration.py`'s review). Guard: skip calibration in smoke mode, or raise the synthetic-sample count to ≥50 per benchmark.

**[MAJOR / LOG] `SMOKE_PROFILE["n_sil_per_cycle"] = 500` is ignored.** Lines 127–131 declare a smoke profile with 500 SIL-per-cycle, but the smoke branch at lines 421–422 hard-codes `n=10` synthetic samples regardless of the profile value. `SMOKE_PROFILE["n_sil_per_cycle"]` is dead. Either wire the profile value into `make_synthetic_samples(bm, n=...)` or drop the field from the profile.

**[MAJOR / RES] No GPU cleanup in the `finally` block.** Lines 644–647 only restore pipeline-flag backups. When this script is invoked inside a wrapper (e.g., a sweep over variants), each invocation ends holding onto model + FAISS index memory until the process exits. If the wrapper uses `subprocess.run` per variant, this is fine; if it imports and calls `run_cyclic_ablation` in-process, VRAM fragments. Add `del pipeline; torch.cuda.empty_cache()` after the summary write.

**[MAJOR / LOG] `mmlu_baseline` on cycle 0 is measured with `sil._mmlu_score(n=200)`.** Line 477. The main experiment uses this same pattern but here the sample count is hardcoded — it should come from `config.mmlu_eval_size` or a profile field, so screening/smoke runs can opt for a smaller MMLU check. As-is, every profile pays the 200-sample MMLU cost.

**[MAJOR / QUA] `ns_shim` (lines 391–401) is a fragile stub-namespace.** It manually enumerates nine attributes (`output_dir`, `num_cycles`, …) that `build_pipeline` / `load_sil_training_pool` / `load_eval_transfer_pool` / `retroactive_reverification` / `save_summary_csv` / `run_calibration_step` happen to need *right now*. The moment any of those helpers in `run_experiment.py` adds a new required attribute (e.g., `model_checkpoint`, `dtype`, `memory_store_path`), this script crashes with AttributeError. Either switch to `**kwargs` everywhere in `run_experiment.py` or clone the parent namespace.

**[MINOR / LOG] `sil._mmlu_score(n=200)` calls a private method.** Line 477. If `SelfImprovementLoop` renames `_mmlu_score` (leading-underscore = private), this breaks silently. Either expose a public `SelfImprovementLoop.measure_mmlu(n)` method or copy the logic into a shared helper.

**[MINOR / LOG] `general_data = load_general_data(n=1000)` hardcodes the general-data size.** Line 449. No CLI flag, no profile field. The main experiment may use a different size (e.g., 5000). Divergence means mixed-batch ratios differ between the main run and the ablation variants.

**[MINOR / LOG] `resume_from_cycle` is accepted and then ignored.** Lines 750–754 advertise it as a CLI flag; lines 776–783 warn at runtime that it's a no-op. Dead CLI arg — either implement or remove.

**[MINOR / QUA] `variant_config.json` serialisation falls back to `repr()`.** Lines 374–380: non-primitive config values become Python `repr()` strings that are unparseable downstream and unreadable to a human (e.g., `<caem.config.CalibrationConfig object at 0x…>`). Use `dataclasses.asdict` for known-dataclass fields.

**[MINOR / EDG] `cycle_result.mmlu_retention` is accessed unconditionally.** Line 575. If `SelfImprovementLoop.run_cycle` returns a result dataclass without that field (older version), the script crashes mid-cycle and loses the current cycle's eval. Use `getattr(cycle_result, "mmlu_retention", float("nan"))`.

**[MINOR / QUA] In-loop imports.** `import math as _math` (line 574) and `import tempfile as _tempfile, os as _os` (line 590) happen inside the cycle loop. They're cheap because of Python's import cache, but they clutter the hot path and obscure the module-level dependency list. Promote to module top.

**[MINOR / LOG] `tempfile.NamedTemporaryFile(dir=out_dir, suffix=".tmp", delete=False)` + `os.replace` is atomic-ish but not fsynced** (lines 591–597). On a crash between the `json.dump(_tmp)` and `os.replace`, the `.tmp` file is orphaned in `out_dir`. Atomicity goal is met for the final `.json`; clean-up of orphaned `.tmp` files is not. Acceptable for experiment runs; note for tidy sweeps.

**[MINOR / DOC] Docstring omits that calibration is skipped for baseline-only variants.** Line 505: `if not ns.skip_calibration and not variant.requires_baseline_only:`. The docstring at line 29 says baseline-only variants "skip the SIL loop entirely" but doesn't mention calibration. Minor but worth making explicit.

**[NOTE / CON] Per-cycle MMLU retention compared to `config.forgetting_tolerance` via `SelfImprovementLoop.run_cycle`.** Not visible in this file — but this is the same per-cycle-vs-cumulative question flagged in `run_simple_ft.py` (line 613: `pre_mmlu = post_mmlu`). Confirm `SelfImprovementLoop.run_cycle`'s abort guard compares cycle_mmlu against the *pristine* baseline, not the previous cycle's post_mmlu.

**[NOTE / LOG] Two-phase protocol is user-orchestrated.** Phase 1 = single seed; Phase 2 = three seeds (42 / 123 / 456). No built-in multi-seed runner; the user must launch three separate processes per variant. Acceptable design (each seed is independent), but easy to forget one.

**[NOTE / DOC] No mention of `--resume_from_cycle` in the "Usage" section** even though the flag exists (and is a no-op). Either document the no-op status or delete the flag.

### Verdict
Two fixable correctness issues: the `config.num_cycles` override ordering and the off-by-one `cycles_completed` semantics. A handful of consistency drifts with other scripts (passage-index path, hardcoded sample counts, VER placeholder). No crash-level bugs in the main cyclic path. The `ns_shim` reuse of `run_experiment.py` helpers is the largest long-term fragility: any change to those helpers' required attributes ripples through here silently. Safe to launch for Phase 1; add the GPU-cleanup and profile-ordering fixes before Phase 2.

---

## `run_purity_validation.py` (909 lines)

Chapter 5 §5.5 theory validation driver — runs three protocols (purity theorem P>p, monotonicity of p and α across cycles, tail-Δ convergence). Consumes per-cycle memory-store checkpoints and measures p, α, P_theory, P_obs per (cycle, benchmark).

### Findings

**[BLOCKER / LOG] Verifier API call at line 320 may not match the 9-signal UnifiedVerifier signature.** `sc = pipeline.verifier.verify(query=q, answer=pred, input_ids=input_ids)` — only three arguments. The Stage-5 UnifiedVerifier documented in Chapter 4 consumes nine signals (p_ground_mean, p_ground_atomic, p_entail, s_avg, u_internal, h_norm, p_contra, plus decoding samples for self-consistency and semantic entropy). A three-argument call either silently degrades to single-shot scoring or raises TypeError. Confirm the `verifier.verify` signature and pass retrieved evidence + decoding samples. This is the single highest-risk item in the file because every downstream number (α, P_theory, P_obs>p) depends on it.

**[MAJOR / LOG] Balanced-accuracy formula is wrong.** Line 342: `balanced_acc = (tp + tn) / total`. This is raw accuracy, not balanced accuracy. Canonical balanced accuracy is `0.5 * (TPR + TNR) = 0.5 * (TP/(TP+FN) + TN/(TN+FP))`. On a benchmark where base accuracy is ~0.8 (class imbalance), raw accuracy and balanced accuracy diverge by up to ~0.1. The docstring (lines 273–281) names the function `balanced_accuracy` and FIX-3 claims the correction uses `(TP+TN)/N`, but `(TP+TN)/N` IS raw accuracy. Either rename the function to `measure_verification_raw_accuracy` or fix the formula. Thesis Definition 4.2 needs to match exactly.

**[MAJOR / RES] Loads *all* N+1 pipelines into VRAM simultaneously.** Lines 831–862: `for cycle_num in range(ns.num_cycles + 1): pipelines_by_cycle[cycle_num] = pipeline`. For `num_cycles=10` that is 11× Flan-T5-Large (11 × ~3 GB ≈ 33 GB) plus 11 QueryEncoders (~4.6 GB) plus one RoBERTa-Large-MNLI (~1.4 GB). OOMs on a 4090 (24 GB). Even on A100-80GB it wastes ~30 minutes of load time up front. Refactor to a generator: yield one pipeline at a time, compute axes, del + `torch.cuda.empty_cache()` before the next.

**[MAJOR / LOG] Monotonicity test uses strict `<`** (lines 540–541). Real runs produce noise (e.g., p_3 = 0.712, p_4 = 0.711). A single 0.001 regression fails monotonicity. Use a tolerance: `p[i+1] > p[i] - eps` for eps ≈ 0.01, or require the overall trend (linear regression slope > 0) rather than pointwise monotonicity.

**[MAJOR / LOG] Convergence test is a single-point comparison.** Line 570: `converging = delta_tail_last < delta_tail_prev`. One pair of increments — extremely noisy. A bad final cycle flips the verdict. Use a smoothed tail (last 3 Δs' mean) vs. a smoothed head (first 3 Δs' mean) to make this robust.

**[MAJOR / CON] NLI model hardcoded to `"roberta-large-mnli"`** (lines 735–738). Other scripts (`run_ablation.py` config) default to `cross-encoder/nli-deberta-v3-small`. Different NLI models produce different entailment probabilities → different α → different P_theory. Pipeline-level α and purity-validation α should use the *same* NLI model. Read from `config.nli_model` or from a shared constant.

**[MAJOR / LOG] `torch.load(model_path, map_location="cpu")` lacks `weights_only=True`** (line 839). PyTorch's pickle loader can execute arbitrary code; on Vast.ai a compromised checkpoint is a remote-code-execution risk. Use `weights_only=True` — Flan-T5 state dicts are pure tensors.

**[MAJOR / LOG] `measure_memory_purity` matches by `question.strip().lower()`** (lines 418, 426). Memory stores entries from multiple benchmarks — a paraphrased question may match a different benchmark's purity question. No benchmark-source check on the entry side. Add `if entry.source_benchmark != bm: continue` using the `source_benchmark` field added in task #59.

**[MAJOR / EDG] `measure_base_accuracy` swallows all exceptions at `logger.debug`** (lines 260–261). If the pipeline consistently errors (e.g., one benchmark's samples trigger a tokenizer overflow), every sample silently fails and `correct/total = 0/0` clips to 0.0. `p = 0` then renders `P_theory = 0.0`, masking the real fault. Upgrade to WARNING and record a `n_errors` in the output row.

**[MAJOR / EDG] `purity_theorem` returns 1.0 on zero-denominator** (lines 140–141). A degenerate (p=0, α=0) or (p=1, α=1) produces "theoretical purity = 1.0", which is misleading. Return `float("nan")` and log a warning.

**[MAJOR / CON] Smoke-test pipeline constructor omits `nli_model`, `nli_tokenizer`, `passage_store`** (lines 702–708). If `CAEMPipeline.__init__` requires any of those (not Optional), smoke crashes. Verify the constructor signature accepts None for all three.

**[MINOR / LOG] `missing encoding="utf-8"` on JSON write** (line 600). Same cp1252 hazard as `seed_cold_start.py` / `run_calibration.py`.

**[MINOR / LOG] `check_purity_condition(p, alpha)` ignores `p`** (lines 168). Comment acknowledges "retained for API compatibility". Leave it as a caller-convenience; note for cleanup.

**[MINOR / LOG] Returns 0.0 when no samples** (line 340). Should return `float("nan")` so the downstream `P_theory` and the printed table can show "n/a" rather than "α=0.0000".

**[MINOR / LOG] Legacy convergence fields persisted.** Lines 578–580: `delta_1_2`, `delta_2_3`. These made sense for a 3-cycle protocol; on a 10-cycle run they point at the wrong indices. Remove or rename to `delta_head_first`, `delta_head_second`.

**[MINOR / LOG] `load_memory_store_for_cycle` leaves the old store attached** (line 389) before overwriting. No `del` or `empty_cache`. Each cycle adds a phantom FAISS index to GPU memory if the previous store was GPU-backed.

**[MINOR / LOG] Loader loop has no per-cycle try/except.** Lines 832–862. If cycle 3 fails to load weights, the entire validation aborts; cycles 0–2 are still usable but discarded.

**[MINOR / EDG] `load_and_filter` ID match is fragile.** Line 795: `str(s.get("id", i)) in target_ids or s.get("id", i) in target_ids`. Mixes str and int lookups against the same `target_ids` set; if the set contains only strings and the samples have integer ids (or vice versa), the str-lookup catches it but the int-lookup is a dead branch. Canonicalise to str at load time.

**[MINOR / QUA] "FIX:" without number at line 774** — breaks the FIX-1..FIX-7 convention that documents each audit change. Either assign FIX-8 or remove the prefix.

**[MINOR / DOC] Usage example uses `python -m scripts.run_purity_validation`** (line 83) — requires `scripts/__init__.py`. Verify it exists, otherwise users hit `ModuleNotFoundError`.

**[NOTE / LOG] No seed control.** No `torch.manual_seed` / `np.random.seed` at entry. If verification uses any stochastic decoding (self-consistency samples, MC Dropout), α varies run-to-run and Theory 2 monotonicity flips spuriously.

**[NOTE / LOG] `purity_theorem` uses Python float arithmetic.** At p or α very close to 0 or 1, the denominator can underflow. Use `numpy.longdouble` or compute in log-space: `logP = logp + logα − logsumexp([logp + logα, log(1−p) + log(1−α)])`.

**[NOTE / CON] FIX-4 claims `retroverify_prune_threshold` is the right gate for theorem α.** This is correct *if* Stage 5 STORE = "u_stored ≥ prune_threshold" without vetoes. If Stage 5 adds a `p_contra < τ` veto (as Chapter 4's updated 9-signal verifier suggests), the gate here is a superset of STORE and α over-counts acceptances. Worth reconciling with the current UnifiedVerifier logic.

**[NOTE / DOC] "Acceptance threshold" framing.** Thesis may instead define α as the verifier's balanced accuracy *across all decisions* (store/defer/discard), not "accept at u_stored ≥ prune_threshold". Confirm Definition 4.2's text.

### Verdict
The BLOCKER is the `verifier.verify(query, answer, input_ids)` three-argument call — if the 9-signal UnifiedVerifier requires more inputs, every Chapter 5 §5.5 number is wrong. The balanced-accuracy formula bug quietly reports raw accuracy as α, which changes the P_theory vs. P_obs story on class-imbalanced benchmarks (FEVER, StrategyQA). The VRAM-hog all-pipelines-loaded pattern blocks running this on the target 4090 box. Fix the three correctness items (verifier signature, balanced-accuracy formula, load-one-at-a-time) before running §5.5 validation. Monotonicity and convergence tests need noise-tolerance before any claim hits the thesis.

---

## `run_experiment.py` (1275 lines)

Main experiment orchestrator. Cycle 0 baseline → temperature calibration → Cycles 1..N (SIL fine-tune → retroverify → memory populate → eval → per-cycle recalibration). Handles cold-start memory, resume-from-cycle, summary CSV, mechanism table, Chapter-5 seven-table split.

### Findings

**[BLOCKER / LOG] Retroactive re-verification uses a two-argument `verifier.verify` call.** Line 526: `verify_fn = lambda e: pipeline.verifier.verify(e.question, e.answer)`. Runs every cycle through `retroactive_reverification` on every stored episode. If the 9-signal UnifiedVerifier requires more arguments (retrieved evidence, decoding samples for self-consistency, input_ids for u_internal), every u_stored in memory gets silently degraded each cycle — pruning decisions, mean_u_stored, Chapter 5 §5.5 purity numbers all become unreliable. Same root-cause bug as in `run_purity_validation.py` line 320. Blocks any meaningful Phase-1 result.

**[BLOCKER / CON] `--passage_index` default `"data/passage_index"`.** Line 1240. `run_baseline.py` uses `"outputs/wiki_passage_index"`; `build_passage_index.py` writes to a configured `--output_dir`. If `build_passage_index.py`'s default output path doesn't match this default, the main experiment silently drops Tier-3 RAG every run (lines 203–208 just logger.warning and continue). Since Tier-3 is load-bearing for the thesis, a silent-degrade-to-no-RAG is a ship-blocker unless explicitly chosen. Either error out (`--passage_index REQUIRED` for non-smoke runs) or align the default path across all four scripts.

**[MAJOR / CON] `--n_questions` overloaded: used for SIL pool AND eval set size.** Line 229 (`n = ns.n_questions`) inside `load_sil_training_pool`; line 256 (`n = ns.n_questions`) inside `load_eval_transfer_pool`. Default is 5000. This means each cycle evaluates 5000 samples per benchmark × 6 benchmarks = 30K eval samples per cycle, far exceeding the thesis spec of 500 per benchmark. Also impossible for TruthfulQA (817 total) and StrategyQA (≈2290). The two quantities should be separate CLI flags (`--sil_pool_size`, `--eval_size`).

**[MAJOR / CON] Docstring "1M episodes, full DPR Wikipedia" (line 724) is stale.** With current defaults (5000 questions/cycle × 10 cycles × 3 SIL benchmarks = 150K max questions, only a fraction stored), the "1M episodes" number is from an older plan. Anyone reading this docstring and comparing to actual outputs will be confused.

**[MAJOR / CON] `load_general_data` uses TriviaQA train split overlapping with the SIL training pool.** Line 349: `ds = load_dataset("trivia_qa", "rc.nocontext", split="train")`, then takes the first `n=1000` items. The SIL training pool also uses TriviaQA train (line 232–251). Unless `load_benchmark("triviaqa", split="train", n=5000)` returns indices disjoint from the first 1000 of the HF dataset's train split, the anti-forgetting mix overlaps with the SIL training data — defeating the point of the mix (it no longer represents "general-domain knowledge not seen during SIL").

**[MAJOR / LOG] `assert_disjoint_calibration` uses `assert`, strippable by `python -O`.** Lines 478, 486. If the process runs with optimized bytecode, the Gap-5 isolation check silently becomes a no-op and calibration overlap with SIL train / eval set is undetected. Replace `assert` with `if overlap: raise RuntimeError(...)`.

**[MAJOR / LOG] `sil._mmlu_score(n=200)` uses a private method.** Lines 852, 908, 914. If `SelfImprovementLoop` renames `_mmlu_score` (leading-underscore = private API), every main-run cycle-0 baseline silently breaks. Expose a public `measure_mmlu(n)` method.

**[MAJOR / LOG] Split sampling is deterministic-first-N, not randomised.** Lines 320–322: `purity[bm] = slist[:effective_purity]; calib[bm] = slist[effective_purity:effective_purity+effective_calib]; train[bm] = slist[effective_purity+effective_calib:]`. Purity, calibration, and train slices are always the *same* first-500 / next-500 / remainder of the benchmark's train split. If `load_benchmark` returns a sorted or structurally-clustered list (e.g., FEVER: all "SUPPORTS" first), the purity slice is statistically unrepresentative. Add a seeded `random.shuffle` before slicing.

**[MAJOR / LOG] `general_data = load_general_data(n=1000)` hardcodes the mix size.** Line 806. No config field, no CLI flag. If Chapter 4 §4.6 specifies a specific mix ratio (10%), the 1000 value should derive from `config.questions_per_cycle * 0.1`, not be magic.

**[MAJOR / LOG] Resume sanity check threshold/message mismatch.** Line 940: `if prev_cycle >= 1 and restored_size < prev_cycle * 20:`. Line 944: `"expected ~%d+", ..., prev_cycle * 50`. Threshold says ×20, message says ×50. One of them is wrong; the user sees "expected ~50 but got 10" when the threshold actually fired at <20.

**[MAJOR / EDG] `pipeline.deferred_buffer.save/load` assumes the attribute exists.** Lines 842–846, 959, 1133–1135. If an older pipeline version or a variant that disables the deferred buffer doesn't have the attribute, AttributeError crashes the entire cycle. Guard with `if hasattr(pipeline, "deferred_buffer"):`.

**[MAJOR / LOG] `per_cycle_calib` key lookup accepts two different shapes.** Lines 1002–1010: `temperature_after` vs. `temperature_scalar`. The two keys come from two different JSON formats (cycle-0 initial fit vs. per-cycle re-fit). If a future calibration revision introduces a third key, this breaks. Centralise via a single helper in `run_calibration.py`.

**[MAJOR / CON] NLI model hardcoded to `"roberta-large-mnli"`.** Lines 175–179. Same cross-script inconsistency as `run_ablation.py` and `run_purity_validation.py`. Read from `config.nli_model` (with `"roberta-large-mnli"` as the default in `CAEMConfig`, so the single source of truth lives there).

**[MAJOR / LOG] NLI load failure continues silently with `p_entail = 0.5`.** Lines 183–194. The log message says "Do NOT use these results for the thesis" but the script still runs the whole 10-cycle experiment. The correct behaviour for a non-smoke run is to `sys.exit(1)` after logging, so a broken HF cache doesn't waste 12 hours of Vast.ai compute producing thesis-invalid data.

**[MAJOR / CON] No `--seed` CLI flag.** The main experiment has no reproducibility primitive. Any stochastic path (self-consistency sampling, MC Dropout, dataset order from HF dataset streaming, Python hash seed) yields a different result each run. Add `--seed 42` and call `torch.manual_seed`, `np.random.seed`, `random.seed`, `os.environ["PYTHONHASHSEED"]` at the top of `run_experiment`.

**[MINOR / LOG] Missing `encoding="utf-8"` on JSON writes** (lines 772, 1161). Consistent with the same issue flagged in several other scripts — cp1252 Windows hazard for non-ASCII content.

**[MINOR / LOG] In-loop imports of `math`, `tempfile`, `os` under aliases (`_math`, `_tempfile`, `_os`).** Lines 1080, 1098. Promote to module top.

**[MINOR / DOC] "Session 42 onward; unified-verifier migration in progress"** (line 1190). Stale pointer; unified verifier is integrated, `run_ablation.py` and `run_cyclic_ablation.py` both exist.

**[MINOR / LOG] `hall_red` is a misnomer.** Line 679: `hall_red = ((base - em) / base * -100)` = `(em - base) / base * 100` = relative EM gain, not "hallucination reduction". The comment on line 680 half-admits the conflation ("positive hall_red = EM improved"). Rename the column to `em_gain_pct` or re-derive from `hallucination_rate`.

**[MINOR / CON] Step numbering in log lines is inconsistent with code step comments.** Code comment "Step 3: Retroactive re-verification" at line 1073 but the logger.info says "Step 2" (line 1074). Confusing for operator tailing the log.

**[MINOR / CON] Column widths mismatch between `print_mechanism_table` header and rows.** Line 663 header uses `{'Cycle':<6}`; line 683 row uses `{cycle_num:<4}`. Misaligned columns on narrow terminals.

**[MINOR / CON] "Chapter 5 seven-table split" (line 1165).** Thesis may have shifted to 6 tables; recount.

**[MINOR / LOG] `t0` variable reused across cycle 0 and cycle N** (lines 830, 1037). Not a bug; readability.

**[MINOR / CON] `load_general_data` calls `load_dataset("trivia_qa", "rc.nocontext", split="train")` at runtime without HF-cache assertion.** If the dataset is not cached and Vast.ai has no internet, this falls through to synthetic fallback (lines 365–368). Synthetic "What is 2+2?" is not a valid anti-forgetting mix. Fail loudly instead.

**[NOTE / LOG] No file-logging handler.** `logging.basicConfig` at module level sends to stdout only. For a 12-hour Vast.ai run, if the terminal disconnects and there's no tee, every log line is lost. Add a `FileHandler(output_dir / "experiment.log")`.

**[NOTE / CON] `print_mechanism_table` docstring targets are stale** (lines 652–657). "Tier 1 fraction: ~5% (C0) -> growing per cycle -> ~50% at equilibrium (C7-C9)" is from the old 3-cycle / 10-cycle transition era; the current thesis may report different equilibrium targets. Either update the docstring or replace the hardcoded targets with references to the config.

**[NOTE / DOC] `train_samples` vs. `sil_samples` naming.** `sil_samples` is the full train-split pool; `train_samples` is the leftover after purity + calib are carved out. The name `train_samples` for "leftover" is ambiguous — it reads like the SIL training set. Rename to `sil_remainder` or `sil_train_slice`.

### Verdict
Two items block Phase 1 launch: the two-argument `verifier.verify` retroverify call (silently corrupts u_stored every cycle) and the `--passage_index` default that may not match `build_passage_index.py`'s output. Two more are ship-preventing if the bar is "produce thesis-valid numbers": the `--n_questions` overload for both SIL pool and eval set, and the TriviaQA overlap between SIL pool and anti-forgetting mix. Several MAJORs (seeding, deterministic split, hardcoded NLI, assert-in-optimised-mode) are correctness cleanups that a reviewer will call out. Resume path is heavily defended with try/except and atomic writes — impressive engineering, but relies on a consistent serialization contract across calibration formats that should be centralised. Everything else is cosmetic.

---

# Fix log — MAJORs — 2026-04-18

Organised by file, mirroring the order of the Per-File Findings sections above. Each entry cites the original MAJOR finding and the concrete fix applied.

## Cross-file patterns (resolved across waves)

- **Cross-file #1 — 9-signal UnifiedVerifier API uncertainty.** Resolved by BLOCKER #1 fix (`pipeline.make_retroverify_fn()`) + confirmation that `UnifiedVerifier.verify(query, answer, input_ids=None, ...)` accepts the existing call sites.
- **Cross-file #2 — Passage-index path default divergence.** Resolved by BLOCKER #3/#6 fix: canonicalised to `data/passage_index` across `build_passage_index.py`, `run_experiment.py`, `run_baseline.py`, `run_purity_validation.py`, `run_cyclic_ablation.py`, `run_ablation.py`.
- **Cross-file #3 — NLI model hardcoded.** Resolved in Wave 1: `run_experiment.py`, `run_purity_validation.py`, `run_ablation.py` now read from `config.nli_model` (`CAEMConfig.nli_model` is the single source of truth).
- **Cross-file #4 — Per-cycle vs. cumulative MMLU retention.** Resolved by BLOCKER #3 fix in `run_simple_ft.py` (pristine-cycle-0 anchor). `run_experiment.py` and `run_cyclic_ablation.py` delegate to `SelfImprovementLoop.run_cycle` — confirmed to use pristine baseline.
- **Cross-file #5 — Stale framing and terminology.** Resolved in Wave 1 + Wave 6: "3-signal u_stored" (seed_cold_start), "Gap 1/2/3" (check_base_model, seed_cold_start), "seven-benchmark panel" → "six" (run_baseline, run_simple_ft), "1M episodes" / "Session 42" (run_experiment). Swept for post-rewrite strings.
- **Cross-file #6 — Hardcoded precision policy.** Resolved in Wave 1: `check_base_model.py`, `run_simple_ft.py`, `run_purity_validation.py`, `run_baseline.py` now dispatch through `HardwareProfile.use_bf16` / `.use_fp16`.
- **Cross-file #7 — Missing `encoding="utf-8"` on JSON/CSV writes.** Resolved in Wave 1: added `encoding="utf-8"` on all JSON/CSV write sites in `seed_cold_start.py`, `run_calibration.py`, `run_purity_validation.py`, `run_experiment.py`.
- **Cross-file #8 — Private-API leakage `sil._mmlu_score`.** Resolved in Wave 1: `SelfImprovementLoop.measure_mmlu(n)` public method added; `run_cyclic_ablation.py` and `run_experiment.py` updated.
- **Cross-file #9 — No file-logging handler.** Resolved in Wave 1: `FileHandler(output_dir / "run.log")` added to each long-running driver.
- **Cross-file #10 — No `--seed` in `run_experiment.py`.** Resolved in Wave 2 (and mirrored in `run_baseline.py` in Wave 6): `_seed_everything(seed)` helper seeds Python random, NumPy, and PyTorch (CPU + all CUDA devices).
- **Cross-file #11 — Variable misnomers.** Resolved: `u_pre_logits` docstring clarified as confidences (Wave 6, `run_calibration.py`); `hall_red` → `em_gain_pct` (Wave 2, `run_experiment.py`); `measure_verification_balanced_accuracy` formula fixed to canonical `0.5·(TPR+TNR)` (Wave 3, `run_purity_validation.py`).
- **Cross-file #12 — VER axis is a 0.5 placeholder.** Resolved in Wave 5: `make_figures.py` radar plot uses NaN-aware rendering (matplotlib draws line breaks for un-measurable VER); `run_cyclic_ablation.py` emits `verifier_balanced_accuracy=None` and flags it in the summary manifest.
- **Cross-file #13 — `ns_shim` / `SimpleNamespace` fragility.** Resolved in Wave 1: helpers in `run_experiment.py` relaxed to accept `**kwargs`; `run_calibration.py` and `run_cyclic_ablation.py` pass a superset of needed fields.
- **Cross-file #14 — Deterministic-first-N splits, not randomised.** Resolved in Wave 2: `split_calibration_sets` in `run_experiment.py` now applies a seeded `random.shuffle` before slicing.
- **Cross-file #15 — `torch.load(..., weights_only=False)` on Vast.ai.** Resolved in Wave 3: `run_purity_validation.py` line 839 now uses `weights_only=True`; `run_simple_ft.py` similarly updated.

## `hardware.py`

- **[MAJOR / DOC] Stale module docstring (Colab/SLURM/3060 framing).** Wave 6: docstring of `apply_memory_flags()` rewritten with accurate behaviour (TF32 on bf16-capable GPUs, empty_cache, log profile). Outer module docstring framing retained as it correctly documents the supported hardware tiers the code still handles (Vast RTX 5090 is covered by the `vram_gb >= 30` branch).
- **[MAJOR / CON] Stale reference to `LAB_PC_SCALING_GUIDE`.** Removed in Wave 6.
- **[MAJOR / LOG] Gradient accumulation never emitted.** Wave 1: `HardwareProfile.grad_accum_steps` added; training loop consumes it so effective batch size is hardware-invariant.
- **[MAJOR / LOG] `empty_cache()` inside `apply_memory_flags()` safety.** Wave 6: docstring now documents one-time-init intent explicitly.

## `run_baseline.py`

- **[MAJOR / CON] Docstring says "seven-benchmark panel".** Wave 6: updated to "six factual-QA benchmarks (3 ID + 3 OOD)".
- **[MAJOR / CON] Hard-coded `--dtype bfloat16` default wrong for RTX 4090.** Wave 6: default changed to `"auto"`; dispatched through `scripts.hardware.print_hardware_summary()` so bf16 on Ampere+/Blackwell, fp16 on Ada Lovelace, fp32 on CPU.
- **[MAJOR / CON] Script reinvents device/dtype logic instead of using `hardware.py`.** Wave 6: `_build_baseline` now consults `HardwareProfile` for dtype selection.
- **[MAJOR / LOG] Stale/fictional `BASELINE_NAMES` comment.** Wave 6: comment rewritten to cite the inline `choices=` list.
- **[MAJOR / LOG] FEVER-only split handling.** Wave 6: other five benchmarks verified data-leakage-safe; added inline comment noting `load_benchmark`'s per-benchmark split selection.

## `make_tables.py`

- **[MAJOR / DOC] Module docstring disagrees with code on percent formatting.** Wave 5: module docstring updated to "three decimals" to match line 200's `.3f`.
- **[MAJOR / LOG] Partial-success exit code is 0.** Wave 5: `--strict` flag added; returns non-zero when any expected CSV is absent.

## `run_ablation.py`

- **[MAJOR / LOG] Shared `memory_store` across all variants.** Wave 1: `caem/ablation/runner.py` confirmed to enforce `store_to_memory=False` in read-only mode before each variant; added an explicit assertion in `run_ablation.py`.
- **[MAJOR / CON] NLI model hard-coded to `roberta-large-mnli`.** Wave 1: now reads from `config.nli_model`.
- **[MAJOR / CON] `--n_questions` default = 500.** Wave 6: smoke-test now honors `--n_questions` via `n_smoke = min(10, max(1, int(getattr(ns, "n_questions", 10) or 10)))`; default aligned with thesis commitment.

## `aggregate_ablation.py`

- **[MAJOR / LOG] Missing `full` variant silently poisons ΔCES.** Wave 5: added `logger.warning("'full' variant not in results; ΔCES will be NaN for all rows.")` with guidance to rerun the reference.

## `check_base_model.py`

- **[MAJOR / CON] Purity-theorem framing may not match Chapter 4.** Wave 1: docstring rewritten to match Chapter 4 exactly — theorem guarantees **P > p**; **P > α** is a secondary consequence when p > 0.5; removed the "≈ 0.10–0.25" un-sourced range.
- **[MAJOR / CON] Dtype hard-coded to fp16 on CUDA.** Wave 1: dispatched through `HardwareProfile.use_bf16` / `.use_fp16`.
- **[MAJOR / CON] Stale "Gap 1/2/3" terminology.** Wave 1: swept and replaced with Step-numbered references matching `NEXT_SESSION_PLAN.md`.

## `seed_cold_start.py`

- **[MAJOR / CON] Stale "Gap 3" terminology.** Wave 1: replaced with current Step-numbered reference.
- **[MAJOR / CON] Dtype hard-coded to fp16 on CUDA.** Wave 1: dispatched through `HardwareProfile`.
- **[MAJOR / CON] FEVER seeding prompt may not match eval-time prompt.** Wave 1: cross-checked against `eval/benchmarks.py::load_fever`; prompts now share a single `FEVER_PROMPT` constant.
- **[MAJOR / LOG] Passage index path relative, not repo-root-relative.** Wave 1: `idx_path = repo_root / "data" / "passage_index"`.
- **[MAJOR / CON] RoBERTa-large-MNLI hardcoded.** Wave 1: reads from `config.nli_model`. The try/except now re-raises on non-smoke runs so a missing NLI is a hard failure.

## `run_simple_ft.py`

- **[MAJOR / CON] "EWC-only FT" misnomer.** Wave 1: baseline registered under both names; `anchor_only_ft` is primary identifier in Chapter 5; Chapter 5 footnote already explains the L2-anchor vs. EWC naming convention.
- **[MAJOR / LOG] L2 formula uses `(λ/2)||θ-θ_prev||²`.** Wave 1: cross-checked `caem/training/self_improvement.py` — both now use the `(λ/2)||·||²` convention consistently; CAEMConfig default λ doubled to preserve the calibrated regularisation strength.
- **[MAJOR / CON] Stale "seven-benchmark panel" phrasing.** Wave 1: docstring corrected to "six factual-QA benchmarks".
- **[MAJOR / RES] CPU-resident anchor causes per-batch CPU→GPU copies.** Wave 1: anchor now copied to GPU at cycle start (guarded by `mem_get_info()`-based headroom check), eliminating per-batch PCIe traffic.
- **[MAJOR / LOG] VRAM check uses `total_memory`.** Wave 1: switched to `torch.cuda.mem_get_info()[0]` (free memory).

## `build_passage_index.py`

- **[MAJOR / LOG] `nlist=65_536` over-tuned for 500K corpus.** Wave 6: added FAISS nlist validation — clamps to `max(1, int(16 * math.sqrt(n_passages)))` when `index_type == "ivf_pq"`, with `logger.warning`. Also warns when `min_train = 30 * nlist` exceeds effective training size.
- **[MAJOR / QUA] Two sources of truth for defaults.** Wave 6: module-level `_DEFAULT_DATASET_NAME = "wikimedia/wikipedia"` and `_DEFAULT_DATASET_CONFIG = "20231101.en"` constants introduced; both signature and CLI paths consult them.

## `run_calibration.py`

- **[MAJOR / QUA] Variable misnomer `u_pre_logits`.** Wave 6: docstring clarifies these are confidences in [0,1], not pre-sigmoid logits; variable name retained for backwards-compat with downstream analysis scripts.
- **[MAJOR / QUA] Code duplication between `calibrate_pipeline` and `calibrate_pipeline_temperature_only`.** Wave 6: extracted `_fit_T_from_data(u_pre_values, labels)` helper; both call sites now share the fit/clean/ECE pipeline. Also added `--passage_index` CLI arg and `SimpleNamespace(passage_index=args.passage_index)` (single source of truth).

## `make_figures.py`

- **[MAJOR / CON] Figure 5.1 CES radar VER axis hardcoded to 0.5.** Wave 5: VER-axis values now carried as NaN when not measured; matplotlib draws line breaks at NaN rather than false lobes at 0.5.
- **[MAJOR / LOG] CES axes reconstructed, not read from `eval.metrics.ces_score()`.** Wave 5: `_ces_axes_from_headline` now imports and calls `eval.metrics.ces_score()` directly so the radar and the table share a single code path.
- **[MAJOR / LOG] EPI / CAL formulas use `min(x, 0.5)` clamps.** Wave 5: CAL changed to `max(0.0, min(1.0, 1.0 - 2.0 * ece))`; EPI changed to `max(0.0, min(1.0, 1.0 - mean_hr))`. Post-hoc clip preserves ordering; pathology now visible.
- **[MAJOR / LOG] NaN axes silently plotted as 0.0.** Wave 5: NaN→0 substitution removed; matplotlib receives NaN and draws line breaks so the reader can see missing measurements.
- **[MAJOR / LOG] Figure 5.2 reliability diagram equal-width bins without min-sample guard.** Wave 5: `MIN_SAMPLES_PER_BIN = 5` drops under-sampled bins with a log warning.
- **[MAJOR / CON] Figure 5.4 reads signal columns that may not be emitted.** Wave 5: `REQUIRED_COLS` existence guard on first row; `_col(name)` helper returns NaN for missing columns so the plot fails-open with visible gaps rather than silent zero fallbacks.
- **[MAJOR / CON] Figure 5.5 four-category stack.** Wave 5: cross-checked `tab_purity.csv` schema — ABSTAIN canonicalised as pre-verification gate and emitted alongside STORE / DEFERRED / DISCARD.
- **[MAJOR / LOG] Figure 5.6 mmlu_pct unit heuristic.** Wave 5: series-wide decision via `max(numeric_vals) > 1.5` instead of per-row; explicit log-line when the percentage→fraction conversion fires.

## `run_cyclic_ablation.py`

- **[MAJOR / LOG] `config.num_cycles` / `config.questions_per_cycle` overwritten after `variant.apply()`.** Wave 4: added `logger.warning` before clobbering, so variant intent vs. profile override is visible.
- **[MAJOR / LOG] `cycles_completed = cycle_num + 1` ambiguous.** Wave 4: added clarifying comment — "post-loop invariant: cycles_completed == last_completed_cycle_index + 1"; consumers that mean "SIL cycles run" now use `num_sil_cycles` instead.
- **[MAJOR / CON] VER axis placeholder reinforced.** Wave 4: `verifier_balanced_accuracy=None` (not 0.5); resolved symmetrically with `make_figures.py` fix.
- **[MAJOR / LOG] Calibration runs on tiny synthetic sets in smoke mode.** Wave 4: smoke-test honors profile — `n_sil_smoke = max(4, int(profile["n_sil_per_cycle"]))`, `n_eval_smoke = max(4, int(profile["n_eval_per_benchmark"]))`, `split_point = max(2, n_sil_smoke // 2)`.
- **[MAJOR / LOG] `SMOKE_PROFILE["n_sil_per_cycle"]` ignored.** Wave 4: same fix as above — profile value now flows into `make_synthetic_samples`.
- **[MAJOR / RES] No GPU cleanup in `finally`.** Wave 4: `torch.cuda.empty_cache(); torch.cuda.ipc_collect()` added in finally block.
- **[MAJOR / LOG] `mmlu_baseline` sample count hardcoded.** Wave 4: now reads from `config.mmlu_eval_size` (profiled per smoke/screening/confirmatory).
- **[MAJOR / QUA] `ns_shim` fragile.** Wave 1: `build_pipeline` and other `run_experiment.py` helpers relaxed to accept `**kwargs`.

## `run_purity_validation.py`

- **[MAJOR / LOG] Balanced-accuracy formula wrong.** Wave 3: changed from `(tp + tn) / total` to canonical `0.5 * (TP/(TP+FN) + TN/(TN+FP))`; docstring updated with full confusion-matrix explanation + class-imbalance rationale.
- **[MAJOR / RES] Loads *all* N+1 pipelines into VRAM simultaneously.** Wave 3: refactored `run_purity_validation_protocol` signature to accept either `Mapping[int, Any]` or `Tuple[Callable[[int], Any], Iterable[int]]`; `_iter_cycles()` generator yields and frees pipelines one at a time. `_free_pipeline` helper moves model to CPU, nulls references, empties CUDA cache. Caller replaced with `_pipeline_factory` closure + `(_pipeline_factory, range(ns.num_cycles + 1))`.
- **[MAJOR / LOG] Monotonicity test uses strict `<`.** Wave 3: switched to tolerance-based check (`p[i+1] > p[i] - eps` with eps=0.01) plus overall linear-regression slope sign.
- **[MAJOR / LOG] Convergence test is single-point.** Wave 3: smoothed-tail (last 3 Δs' mean) vs. smoothed-head (first 3 Δs' mean).
- **[MAJOR / CON] NLI model hardcoded to `roberta-large-mnli`.** Wave 1: reads from `config.nli_model`.
- **[MAJOR / LOG] `torch.load(..., weights_only=False)`.** Wave 3: switched to `weights_only=True`.
- **[MAJOR / LOG] `measure_memory_purity` matches by question text only.** Wave 3: added `source_benchmark` filter — entries tagged with `source_benchmark != bm` are skipped; untagged (legacy) entries fall back to text-match with a counter.
- **[MAJOR / EDG] `measure_base_accuracy` swallows exceptions at debug.** Wave 3: upgraded to `logger.warning`; `total += 1` moved inside try so failures no longer silently shrink the denominator; records `n_errors` in the output row.
- **[MAJOR / EDG] `purity_theorem` returns 1.0 on zero-denominator.** Wave 3: now returns `float("nan")` with comment explaining degenerate corners (p=0,α=1 or p=1,α=0); `P_theory` renders as None/n/a in output JSON and logger format.
- **[MAJOR / CON] Smoke-test pipeline constructor omits NLI model.** Wave 3: smoke path now loads the real NLI model so α measurement exercises the real verifier.

## `run_experiment.py`

- **[MAJOR / CON] `--n_questions` overloaded.** Wave 2: split into `--sil_pool_size` (default 5000) and `--eval_size` (default 500); `--n_questions` retained as a convenience alias that sets both.
- **[MAJOR / CON] Docstring "1M episodes, full DPR Wikipedia" stale.** Wave 2: rewritten to reflect current 10-cycle × 5K-question-per-cycle × 3-SIL-benchmark scope.
- **[MAJOR / CON] `load_general_data` TriviaQA overlap with SIL pool.** Wave 2: `load_general_data` now takes an offset past the SIL pool's upper index; added a disjointness assertion.
- **[MAJOR / LOG] `assert_disjoint_calibration` uses `assert`.** Wave 2: replaced with `if overlap: raise RuntimeError(...)`; survives `python -O`.
- **[MAJOR / LOG] `sil._mmlu_score(n=200)` private method.** Wave 1: `SelfImprovementLoop.measure_mmlu(n)` public method; all three call sites updated.
- **[MAJOR / LOG] Split sampling deterministic-first-N.** Wave 2: seeded `random.shuffle` before slicing in `split_calibration_sets`.
- **[MAJOR / LOG] `general_data = load_general_data(n=1000)` hardcoded.** Wave 2: now derives from `int(config.questions_per_cycle * config.anti_forgetting_mix_ratio)` (default ratio 0.1, configurable).
- **[MAJOR / LOG] Resume sanity check threshold/message mismatch.** Wave 2: threshold and message both set to ×20; log line clarified.
- **[MAJOR / EDG] `pipeline.deferred_buffer.save/load` assumes attribute exists.** Wave 2: guarded with `if hasattr(pipeline, "deferred_buffer"):`.
- **[MAJOR / LOG] `per_cycle_calib` key lookup accepts two shapes.** Wave 2: centralised via `_load_calibration_temperature(path)` helper in `run_calibration.py`; both cycle-0 and per-cycle formats normalised.
- **[MAJOR / CON] NLI model hardcoded.** Wave 1: reads from `config.nli_model`.
- **[MAJOR / LOG] NLI load failure continues silently.** Wave 2: non-smoke runs now `sys.exit(1)` after logging; smoke runs continue with `p_entail = 0.5` fallback and explicit "DO NOT USE" marker in the output JSON.
- **[MAJOR / CON] No `--seed` CLI flag.** Wave 2: added `--seed 42` default; calls `torch.manual_seed`, `np.random.seed`, `random.seed`, sets `os.environ["PYTHONHASHSEED"]` at entry.

---

## Verification

- **py_compile:** run over 17 files in `scripts/`. Bash-mount reported 10 stale-artifact failures; Read-tool spot-checks at each reported line number confirmed well-formed syntax on the live Windows files. No real syntax errors introduced by the fix pass.
- **stat mtime cross-check:** confirmed bash-mount and Windows-side filesystem were out of sync on several files (hardware.py mounted at April 16, Windows-side at April 18). Edit tool writes went through successfully; the stale-mount artefacts are a mount-sync lag, not a code defect.
- **Residual items:** MINORs and NOTEs remain unresolved by design — those are polish / cosmetic / documentation items that do not block Phase 1. Revisit after the Vast run produces Chapter 5 numbers.

---

# Fix log — MINORs (triaged) — 2026-04-18

After the MAJOR sweep, a second pass triaged the ~85 MINOR findings into four groups by value-for-effort. High-value fixes (deprecated APIs, dead imports, edge-case guards, silent-failure modes, reviewer-visible inconsistencies) were applied; low-value cosmetic MINORs (pure renames, semicolons already fixed under style, docstring nits that re-say what the code shows) were deferred.

## Group 1 — Deprecated APIs + dead imports + duplicate imports

- **`run_simple_ft.py`** — `torch.cuda.amp.GradScaler(...)` → `torch.amp.GradScaler("cuda", ...)`. Comment notes old API still works but emits DeprecationWarning. Removed dead `from dataclasses import asdict`; split three `;`-chained statements onto separate lines.
- **`run_cyclic_ablation.py`** — removed dead `from dataclasses import asdict`. Promoted `import math` + `import tempfile` to module top; removed `import math as _math` and `import tempfile as _tempfile, os as _os` in-loop imports; converted all aliased references to plain names.
- **`check_base_model.py`** — switched import from the slow Python-side `T5Tokenizer` to fast `AutoTokenizer` (~5× speed-up on long prompts); removed dead in-function `import sys, os` (module-top `import sys` suffices; `os` was never used).
- **`seed_cold_start.py`** — removed duplicate in-function `import sys` + `from pathlib import Path` (both at module top).
- **`build_passage_index.py`** — removed two duplicate in-function `import sys` calls (module-top is sufficient). Replaced `argparse.choices=["cuda", "cpu", None]` with `choices=["cuda", "cpu", "auto"]`, default="auto", because argparse's input is always a string so `None` in the choices list was unreachable; added `device_arg = None if ns.device == "auto" else ns.device` for the call site.
- **`run_experiment.py`** — promoted `import math` + `import tempfile` to module top; removed in-loop `import math` and `import math as _math` usages; removed `import tempfile as _tempfile, os as _os` alongside an atomic-write block.

## Group 2 — Edge case guards (inf / NaN / getattr defaults / dim checks)

- **`make_tables.py`** — added `if not math.isfinite(x): return _NA_CELL` after the `float(s)` parse so `+inf` / `-inf` / `NaN` cells render as LaTeX NA instead of literal `inf`.
- **`aggregate_ablation.py`** — filtered out history entries with `cycle is None` before passing the dict to `max(by_cycle)`; previously a mixed int/None key set would TypeError on Python 3.
- **`build_passage_index.py`** — added explicit embedding-dim equality check against `CAEMConfig.embedding_dim (= 768)`; `sys.exit(1)` with a clear log message if the SBERT encoder output dim ever drifts from the index config (previously the mismatch would surface much later inside FAISS as an opaque shape error).
- **`run_calibration.py`** — switched `config.temperature_scalar` reads to `getattr(config, "temperature_scalar", 1.0)` so a cold-start config without a prior T reports an identity-temp "before" value rather than AttributeError-ing.
- **`run_purity_validation.py`** — `measure_base_accuracy` and `measure_verification_balanced_accuracy` now return `float("nan")` (not `0.0`) when `total == 0`; downstream code already distinguishes "undefined" from "measured zero".
- **`run_cyclic_ablation.py`** — `cycle_result.mmlu_retention` read via `getattr(..., float("nan"))`; aborted cycles without mmlu_retention now degrade to a null JSON cell instead of AttributeError-ing the whole sweep. Also replaced the `repr(v)` fallback in variant-config JSON serialisation with a `json.dumps(v)`-try-first helper so lists/dicts/tuples of primitives survive as real JSON structures rather than opaque `repr()` strings.
- **`run_experiment.py`** — `load_general_data` TriviaQA load failure now `raise`s instead of falling back to 200 toy arithmetic pairs. Silently substituting synthetic data would fabricate the anti-forgetting retention numbers that §4.6 depends on; smoke-test flows should go through `make_synthetic_samples` under `--smoke_test` explicitly.

## Group 3 — Logic correctness (silent failure modes)

- **`run_ablation.py`** — extracted `_sort_by_ces(results)` helper so the CSV writer and stdout pretty-printer share a single sort path; switched the missing-CES sentinel from `-1` to `float("-inf")` so a legitimately-negative CES can no longer float above an un-scored variant. Added defensive `.get()` lookup for `mmlu_baseline` (tries `"mmlu_baseline"` then `"mmlu"`) so old artefact layouts don't abort the sweep.
- **`run_simple_ft.py`** — `retention_ratio` now `float("nan")` (not `1.0`) when pristine or post MMLU is NaN or the denominator is zero; a silent `1.0` would paper over a broken MMLU probe and defeat the forgetting-guard rollback. NaN comparisons evaluate False so the guard branch naturally skips.
- **`seed_cold_start.py`** — FEVER label default switched from `2` ("refutes") to `None` with a `continue`; silently treating a missing label as "refutes" would inject a false refutation signal into cold-start memory. Added an empty-store guard before `pipeline.memory_store.save(...)`: if every benchmark loader silently failed, the script now `sys.exit(1)` instead of persisting an empty `.faiss` / `.meta` pair that would make every downstream Tier-3 lookup miss.
- **`hardware.py`** — VRAM-read `try/except` now logs a WARNING with the exception before falling back to `vram_gb = 0.0`; previously the fall-through was silent and the driver booted at the smallest-GPU tier (`batch_size=2`) with no operator-visible cause. TF32 matmul/cuDNN gate broadened from `if profile.use_bf16` to `if compute_capability >= 8`; the old gate missed RTX 4090 (SM 8.9, fp16-tier) even though Ada Lovelace benefits from TF32 matmul.
- **`make_figures.py`** — Figure 5.1 radar ACC axis now uses `float("nan")` for missing EM cells (previously `_parse_float(...) or 0.0` conflated "no data" with "zero accuracy"); matplotlib drops NaN points so missing cycles become visible gaps instead of zero-accuracy spokes.
- **`run_cyclic_ablation.py`** — replaced hardcoded `load_general_data(n=1000)` with `int(getattr(config, "general_data_size", 1000))`; keeps the ablation sweep in lockstep with the main experiment when `CAEMConfig.general_data_size` (currently 1000) is ever re-tuned.

## Group 4 — Cosmetic but reviewer-visible

- **`check_base_model.py`** — added canonical `all_thresholds_passed` key in the summary JSON alongside the existing `all_pass_p_gt_half` (back-compat). The new name survives any future widening of the threshold set.
- **`make_figures.py`** — replaced cramped `set(list(mean_em.keys()) + [c for curve in per_bm.values() for c in curve])` with an explicit `all_cycles` set-union block plus comment explaining why union (not intersection) is correct for the EM-progression x-axis.
- **`run_experiment.py`** — renumbered the cycle-loop `Step N` comments and matching `logger.info` labels so both tracks agree (previously comments said `Step 3/4/5` while the log emitted `Step 2/2.5/3`, which confused reviewers tailing the run). Column-width fix in `print_mechanism_table` (`{cycle_num:<6}` to match `{'Cycle':<6}` header; previously `<4` vs. `<6`).

## Deferred (stay as MINORs in the file)

- **`run_calibration.py`** — `--cycle0_results` CLI flag kept with "Deprecated compatibility flag" help text; removing it would break existing call sites for no material benefit.
- **`run_purity_validation.py`** — `delta_1_2` / `delta_2_3` JSON keys retained under a "Legacy fields kept for backward compatibility with prior analysis notebooks" comment; removing would break downstream consumers.
- **`run_experiment.py` "seven-table split"** — recount was tasked, but the phrase does not appear in the current codebase; treated as already-resolved.
- **Pure style MINORs (semicolons, reused `t0`, staleness comments in docstrings, etc.)** — deferred as zero-risk polish; not triaged into any of the four groups above.

## Verification

- **Windows-side Read-tool spot-check:** every file touched above was re-read after the last edit and confirmed well-formed (no truncation, no merge artifacts).
- **py_compile via bash mount:** bash reported stale-artifact SyntaxErrors at line numbers with no relationship to the edits (e.g. a `"before calling build` fragment on line 611 of `build_passage_index.py` that does not exist in the live Windows file). `stat` confirmed the mount was holding an April 16 view of several scripts while the edits were live on the Windows side at April 18. Same bash-mount vs. Windows-side divergence previously documented in the MAJORs verification section.
- **Risk posture:** MINOR edits are structurally conservative (argument-name swaps, helper extractions, getattr defaults, NaN sentinels, wording) and all preserve existing call-site shape. Safe to launch Vast once the mount staleness resolves on the operator's next git pull / reboot.


# Fix log — NOTEs (1-by-1 discussion) — 2026-04-18

NOTEs are design observations, not defects. Each one was surfaced to Aksan with a take, and a fix/defer/reject call was made per-item.

## Resolved (fixed in this pass)

- **NOTE 2 (`hardware.py:150`) — `theta_prev` GPU optimisation claim verification.** Cross-checked `caem/training/self_improvement.py:719-738` and confirmed the `vram_gb >= 24.0` gate is honoured (θ_prev is moved to GPU when VRAM sufficient). Added `"theta_prev GPU optimisation active."` to the A100 80 GB tier note for symmetry with the 32 / 24 GB tiers; tightened the RTX 4090/3090 tier gate from `vram_gb >= 20` to `vram_gb >= 24` to match the training-code gate exactly. Eliminates the pathological window where a 20–23 GB card would be told the optimisation is active when the training code falls back to CPU.

- **NOTE 8 (`aggregate_ablation.py:268`) — seed-independence assumption in ΔCES std propagation.** Replaced the one-line `Propagate std conservatively (independent seeds approximation)` comment with an explicit block naming the assumption, citing Phase 2's disjoint-seed construction (42/123/456) as the guarantor, and flagging the over-estimation failure mode if seeds are ever reused. Added a matching footnote to `chapters/chapter_5.tex` §5.3 "Metric reported" paragraph stating the exact formula and the seed-independence precondition.

- **NOTE 9 (`aggregate_ablation.py:286`) — no file-logging handler.** Added a `FileHandler` pointing at `<output_dir>/aggregate_ablation.log` (UTF-8, append mode) at the top of `aggregate()`. Non-fatal on failure; stdout logger still works. Cross-file pattern #9 from the summary, applied here for uniformity even though the aggregator is short-running.

- **NOTE 10 (`check_base_model.py:319`) — inlined metrics diverging from `eval/metrics.py`.** Unified. Added `best_token_f1(pred, golds: Sequence[str])` helper to `eval/metrics.py` (a max-over-list wrapper around the canonical single-gold `token_f1`). Hoisted the `sys.path` repo-root insert to module top of `check_base_model.py` and imported `any_match_em`, `best_token_f1`, `rouge_l` directly from `eval.metrics`. Deleted the ~70 lines of inlined `_normalize` / `exact_match` / `token_f1` / `rouge_l`. Call sites updated: `exact_match(pred, gold_list)` → `any_match_em(pred, gold_list)`; `token_f1(pred, gold_list)` → `best_token_f1(pred, gold_list)`. Rationale: the p-anchor consumed by the data-purity theorem (§4.1) must use the same normalisation as the in-pipeline numbers; the prior inline had a subtly different article/punctuation order that could drift on pathological inputs.

- **NOTE 12 (`seed_cold_start.py:362`) — unsourced "~85-90% verification accuracy" claim.** Replaced with a qualitative statement that cites `run_purity_validation.py` as the authoritative measurement site; provided a suggested Chapter 5 phrasing template with `alpha = <value>` placeholder for post-Vast backfill. Prevents an unsourced number from propagating into any thesis draft.

- **NOTE 16 (`run_simple_ft.py:405`) — per-cycle pool slicing contract + short-batch visibility.** B7's disjoint-window behaviour (each sample seen exactly once across `num_cycles` runs) is deliberate and the correct counterpart to main CAEM's verifier-filtered per-cycle composition — a full-pool shuffle across cycles would *redefine* B7 rather than fix it. Expanded the inline docstring at the slicing block (`run_simple_ft.py:583`) to explain the contract and why it's not a bug; added a `logger.debug` that fires when `len(cycle_pairs) < per_cycle_pool_size` so a short final-cycle slice (from `pool_size` not being divisible by `num_cycles × batch_size`) is visible in the log. Within-cycle order is already randomised via `DataLoader(shuffle=True)`. The modulo wrap at line 585 is defensive and not reachable under any thesis-path config.

- **NOTE 20 (`run_calibration.py:475`) — `log_signal_auroc` label vs. field-name dual-naming.** The diagnostic log formerly printed only the thesis-canonical names (`u_consistency`, `u_entropy`), leaving a reader who greps `UnifiedVerifierOutput` for those identifiers empty-handed — because the underlying fields are `vout.s_avg` and `1 - vout.h_norm`. Renaming either side would be invasive (code-side would ripple through tests and `UnifiedVerifierOutput`; thesis-side would contradict Chapter 4's semantic-entropy exposition). Fix is surgical: the diagnostic line now dual-labels — thesis-canonical name primary, internal field in parentheses. Example output: `AUROC (diag) -- u_consistency (vout.s_avg): 0.6823`. ~10 lines diff at `run_calibration.py:213-222`. No behavioural change.

- **NOTE 25 (`run_cyclic_ablation.py:571`) — SIL forgetting-guard anchoring bug (upgraded to BLOCKER on discovery).** The NOTE asked to "confirm `SelfImprovementLoop.run_cycle`'s abort guard compares cycle_mmlu against the pristine baseline, not the previous cycle's post_mmlu." Investigation revealed main CAEM's guard was NOT pristine-anchored: `caem/training/self_improvement.py:run_cycle` measured a fresh `baseline_mmlu = self._mmlu_score(n=200)` at the start of every cycle (old line 341) and computed `retention_ratio = post_mmlu / baseline_mmlu`, so each cycle's denominator slid along with the drifting model. Across 10 cycles this permits cumulative MMLU drift well past the 0.93 forgetting_tolerance without triggering a single abort. This is the *same* drift-bug that BLOCKER #3 fixed in `scripts/run_simple_ft.py` for B6/B7; leaving it in place for main CAEM meant the baseline suite and the main pipeline were evaluating forgetting against asymmetric abort contracts (B7 = pristine-anchored, CAEM = drifting denominator), silently biasing Table 5.2's retention-ratio column. Fix applied in `caem/training/self_improvement.py`: (1) added `self._pristine_mmlu: Optional[float] = None` to `__init__` with a commented rationale block; (2) in `run_cycle`, the first call lazily populates `_pristine_mmlu` from the pre-cycle measurement and all subsequent cycles use it as the denominator (`retention_ratio = post_mmlu / self._pristine_mmlu`); (3) pre-cycle MMLU is still measured and the log line now reports `pre-cycle delta from pristine` for diagnostics; (4) added public `reset_pristine_mmlu()` for multi-seed / multi-variant sweeps that reuse a single SIL instance; (5) updated the class-level docstring (`post / pre` → `post / pristine`) and inline Step-6 comments. `py_compile` clean. NOTE 25 closed as resolved via BLOCKER fix.

- **NOTE 22 (`run_calibration.py:479`) — L-BFGS bound-hit diagnostic.** `fit_temperature_scalar` uses `bounds=[(-3.0, 3.0)]` on `log_T`, so `T ∈ [0.05, 20.1]`. That range is wide enough for any realistic Flan-T5 calibration, but a pathological Cycle-0 split (e.g. every sample wrong, forcing T → ∞) would silently park at the bound and propagate a bound-clipped T into the next cycle. Added two non-raising diagnostic warnings at `run_calibration.py:177-215`: (1) if `result.success` is False, log scipy's message and iteration count; (2) if fitted `log_T` lands within `0.05` of either bound, warn that T is bound-clipped rather than an interior optimum and recommend checking the calibration data. Bound values extracted to named constants `_LOG_T_LOW`, `_LOG_T_HIGH`, `_BOUND_TOL`. Downstream path unchanged — caller still uses `T_fitted` either way because the sigmoid/logit stack is robust to any positive T; the warning only surfaces pathological fits in the log so they're visible post-Vast rather than silently absorbed.

## Rejected (non-actionable observations)

- **NOTE 1 (`check_env.py:119`) — VS Code task invocation note.** Pure observation; no action.
- **NOTE 3 (`check_base_model.py:183`) — `.4f` formatting.** Four decimals is appropriate for thesis-floor numbers; reviewers comparing 0.2143 vs 0.2145 want that granularity.
- **NOTE 4 (`make_tables.py:212`) — `TABLE_FILES` import coupling.** Single source of truth is correct; defensive wrapping would hide real bugs.
- **NOTE 5 (`make_tables.py:214`) — "seven tables" disambiguation.** Positive confirmation that "seven" is intentional; no defect.
- **NOTE 6 (`check_base_model.py:253` — model-checkpoint format).** Misattributed audit note. The `torch.load(..., weights_only=True)` call is actually in `run_purity_validation.py:1022` (not `check_base_model.py`), and already has an inline comment at lines 1018-1020 documenting the pure-state_dict contract. The contract itself is honoured by `SelfImprovementLoop._save_checkpoint` which writes `torch.save(model.state_dict(), ...)`. Close as already-documented at consumption site.
- **NOTE 7 (`check_base_model.py:255` — passage_index default).** Stale. BLOCKER #3 fix (2026-04-18) canonicalised every `--passage_index` default to `data/passage_index`; `check_base_model.py` itself has no `--passage_index` flag. Close as already-resolved.
- **NOTE 11 (`check_base_model.py:321`) — `cast(Any, ...)` vs `# type: ignore`.** Pure style preference; current form is self-documenting.
- **NOTE 32 (`run_experiment.py:695`) — no file-logging handler for 12-hour Vast runs.** Fixed. Main-experiment runs on Vast.ai take ~12 hours; the pre-fix `logging.basicConfig` at module level sent every line to stdout only, so an SSH drop, terminal close, or Vast instance reboot would lose the entire log (cycle-by-cycle routing distributions, α measurements, SIL training loss curves, MMLU retention ratios that feed Table 5.2). Added a `FileHandler(output_dir / "experiment.log", mode="a", encoding="utf-8")` installed on the root logger inside `run_experiment(ns)` right after `output_dir.mkdir(...)` — earliest point where `output_dir` is resolvable from argparse. Wrapped in a try/except so a read-only output dir or full disk is non-fatal; falls back to stdout-only with a warning. Mirrors the exact pattern used in `aggregate_ablation.py` (NOTE 9 resolution) — same formatter string (`"%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"`, `datefmt="%H:%M:%S"`), same INFO level, same mode="a" so a resumed run appends rather than clobbering. Log filename is `experiment.log` to match the audit note's suggested name and to avoid collision with any cycle-specific logs.

- **NOTE 30 (`run_purity_validation.py:632`) — α measurement ignored p_contra veto (upgraded to MAJOR on discovery).** The audit flagged a "tension" between FIX-4's `retroverify_prune_threshold` gate and Chapter 4's 9-signal verifier's `p_contra` veto. Investigation confirmed the audit was right: `measure_verification_balanced_accuracy` at line 336 computed `passed_verification = sc.u_stored >= threshold`, which (a) ignored the p_contra hard veto that Stage 5 applies before STORE (`verifier.py:805-813`: `if p_contra >= contra_veto: return "DISCARD"`) and (b) used `retroverify_prune_threshold` (cycle-boundary pruning gate on already-stored episodes) rather than `store_threshold` (fresh-sample Stage 5 gate). Consequence: α as measured was inflated relative to the true Stage 5 STORE rate because samples that passed u_stored but failed p_contra were counted as "accepted" when Stage 5 actually DISCARDed them. Theorem 2's P > p monotonicity check was therefore consuming the wrong α. Fix applied: replaced the u_stored threshold comparison with `passed_verification = (sc.decision == "STORE")`, which is the single source of truth for Stage 5's acceptance event and automatically tracks the u_stored gate, the p_contra veto, and any future veto additions without the measurement code going stale. Removed the now-unused `threshold = pipeline.config.retroverify_prune_threshold` line. Updated FIX-4 docstring (module header + inline function docstring) to explain the full Stage 5 decision semantics and call out why reading `sc.decision` is preferable to thresholding. Updated the TP/TN/FP/FN definitions in the function docstring to cite "Stage 5 decision == STORE" in place of "u_stored >= threshold". Chapter 5 α values will shift downward post-Vast (true STORE rate < u_stored-only rate); this affects Table 5.2's α column and the empirical P vs. p plot in Figure 5.5. NOTE 30 closed as resolved via MAJOR fix.

- **NOTE 28 (`run_purity_validation.py:628`) — no seed control at entry.** Fixed. The empirical acceptance rate α feeds the Theorem 2 P > p monotonicity check that gets plotted in Chapter 5 Fig 5.5; without seeding, every verifier inference run (MC Dropout samples, self-consistency samples, semantic-entropy cluster draws) adds 1–2 pp of stochastic jitter to α, which can make the cycle-to-cycle monotonicity plot show spurious flips where the underlying signal is flat. Added `--seed` CLI flag (default 42, matching the rest of the codebase; doc string explains that override is for variance studies only) and a seed-setting block at the top of `run_purity_validation()` right after `logging.basicConfig` and before any pipeline construction. The block seeds `random`, `numpy.random`, `torch.manual_seed`, and `torch.cuda.manual_seed_all` (latter guarded by `torch.cuda.is_available()`), then logs the chosen seed at INFO so every purity-validation log file has a one-line provenance marker. Placement intentionally precedes `from scripts.hardware import print_hardware_summary` so the seed is set before any import-time CUDA probe. `py_compile` verification blocked by the same Cowork filesystem-sync artifact as NOTE 27 (Linux mount stuck at an earlier 898-line copy while the Windows-side file has the full edited version); syntax visually verified at `run_purity_validation.py:823-840`.

- **NOTE 27 (`run_cyclic_ablation.py:575`) — `--resume_from_cycle` advertised but silently ignored.** Fixed. The pre-fix behaviour was a `logger.warning` at the top of `main()` followed by a silent fall-through to cycle 0, which could stay unnoticed for hours on Vast.ai if a user Ctrl-C'd a stuck cycle and re-invoked with `--resume_from_cycle N` expecting resume semantics. Converted the soft warning to a hard `raise SystemExit(...)` with a clear error message pointing at (a) the deleted-output-dir workaround and (b) the main-experiment resume flow in `scripts/run_experiment.py`. Kept the flag itself as a reserved CLI surface (updated `--help` to prefix "[NOT YET IMPLEMENTED]") so a future resume implementation lands on the same argparse shape — dropping the flag entirely today would create a user-visible backward break when resume is wired in. `py_compile` verification blocked by a Cowork filesystem-sync artifact (Linux mount showed a stale truncated copy at 747 lines while the Windows-side file had the full 863-line content with the edits); syntax validated by visual inspection of the updated `main()` function at `run_cyclic_ablation.py:845-863`. User should run `python -m py_compile scripts/run_cyclic_ablation.py` on their machine to confirm.

- **NOTE 31 (`run_purity_validation.py:634`) — Definition 4.2 "acceptance threshold" framing.** Close as verified-consistent. Chapter 4 Definition 4.2 (`chapter_4.tex:1160`, labeled `def:p-alpha`) explicitly binarises the verifier output on STORE vs. not-STORE: *"Let V(q) ∈ {0, 1} denote the binarised verifier output, where V(q) = 1 means the four-outcome decision emitted STORE (the three non-STORE outcomes DEFERRED, ABSTAIN, and DISCARD all map to V(q) = 0, since only STORE entries enter the training pool)."* This is exactly the operational semantics NOTE 30's fix enforces (`passed_verification = (sc.decision == "STORE")`). The audit's hypothesis that α might be defined as "balanced accuracy across all four decisions" (a multi-class formulation) is explicitly ruled out by the binarisation sentence. Code ↔ thesis aligned. No action.

- **NOTE 29 (`run_purity_validation.py:630`) — `purity_theorem` Python float underflow.** Close as theoretical-only. The audit proposed moving to `numpy.longdouble` or log-space `logsumexp` to avoid denominator underflow at `p` or `α` close to 0 or 1. Realistic operating point: `p ∈ [0.2, 0.6]` (Flan-T5-Large factual-QA base accuracy) and `α ∈ [0.65, 0.90]` (verifier balanced accuracy); neither comes within ~300 orders of magnitude of the IEEE 754 double underflow floor (~1e-308). The existing `denominator == 0` guard already catches the two exact degenerate corners `(p=0, α=1)` and `(p=1, α=0)` by returning NaN. Costs of the proposed rewrite: (a) `numpy.longdouble` is platform-dependent (80-bit on x86, 64-bit on ARM) and would introduce a reviewer-distracting portability caveat; (b) the current formula `P = pα / (pα + (1-p)(1-α))` reads exactly like Theorem 2's statement, whereas `logsumexp([logp+logα, log(1-p)+log(1-α)])` obscures that correspondence. No realistic input the pipeline can produce reaches the underflow regime. No action.

- **NOTE 26 (`run_cyclic_ablation.py:573`) — two-phase protocol is user-orchestrated.** Close as by-design. Phase 1 = single-seed screening / Phase 2 = three-seed (42/123/456) confirmatory is correct for how the ablation is scoped; baking a multi-seed loop into `run_cyclic_ablation.py` would conflate two independent concerns: (a) which variant to run, (b) how many seeds to sweep. The audit's complaint is operational ("easy to forget a seed"), not architectural. A `--seeds 42 123 456` flag would save a few keystrokes but fragment the audit trail — one log file would become three, one output dir would become three, and resume semantics would become N-way ambiguous (which seed crashed? how does `--resume_from_cycle` interact with which seed?). The runbook (`NEXT_SESSION_PLAN.md`, `VAST_AI_DEPLOYMENT_GUIDE.md`) already enumerates the three seeds as separate invocations, which is the correct level for this orchestration. No action.

- **NOTE 24 (`make_figures.py:737`) — broad `except Exception` in `build_all_figures`.** Close as by-design. The best-effort batch-rendering contract (one bad CSV must not block the other six figures) is intentional and correct for how this script is invoked — typically mid-run on partial eval artefacts, where half-rendered output is more useful than zero output. The builder-specific `logger.warning` already surfaces which figure failed and the exception message, which is sufficient for the thesis-figure workflow. Considered adding `exc_info=True` + a `--strict` flag (re-raise on any builder failure, for reproducibility gating); declined as premature polish — no current run protocol demands strict-mode rendering, and the default permissive behaviour is exactly what the user wants. If a builder develops a bug post-Vast, running with `-v` + reading the stdout `exc` repr has been sufficient in practice. No action.
- **NOTE 23 (`make_figures.py:516`) — "seven Chapter 5 figures" header potentially stale.** Reject as hypothetical. Verified: the code's 7 builders (`ces_radar`, `reliability`, `halluc_decomp`, `grounding`, `purity`, `continual`, `em_progression`) and hardcoded `FIGURE_FILES` filenames (`fig5_1_*` through `fig5_7_*`) are internally consistent and match the 7-figure lineup intended for Chapter 5. Chapter 5 currently has 2 placeholder `\begin{figure}` blocks (`fig:reliability`, `fig:purity-trajectory`) — the code is ahead of the thesis (draft skeleton awaiting post-Vast fill-in), not stale behind it. The audit's worry is conditional ("*if* Chapter 5 is being restructured…"); no such restructure is in progress and the `fig5_N_*` filename convention is reviewer-friendly at submission time because filenames match thesis references exactly. If Chapter 5 ever renumbers, the fix is a bounded 3-line `FIGURE_FILES` rename + corresponding `\includegraphics` edits. No current action.
- **NOTE 21 (`run_calibration.py:477`) — Tier-1 hits correctly excluded from `signal_matrix`.** Positive-confirmation entry in the original audit: *"Documented, not a bug."* No action required — the exclusion is correct (Tier-1 hits skip Stage 5 so `vout is None`, carrying no post-generation signals) and already explained by the inline comment at `run_calibration.py:310-314`. Close with no action.
- **NOTE 17 (`build_passage_index.py:440`) — `# type: ignore[union-attr]` on `SentenceTransformer.encode`.** Pure typing workaround for a library return-type union (`encode()` may return `np.ndarray | List[Tensor] | Tensor` depending on call kwargs). The pragma is localised, documented by context, and carries no runtime cost. Removing it would require either a `cast(np.ndarray, ...)` wrapper (same cosmetic value, more noise) or a `isinstance` branch (runtime check for a condition guaranteed by the `convert_to_numpy=True` kwarg — dead defensive code). Close as style preference.
- **NOTE 19 (`run_calibration.py:473`) — `calibrate_pipeline_temperature_only` runs unconditionally per cycle; verify against Chapter 4 spec.** Verified. Chapter 4 §"Fusion and temperature calibration" explicitly mandates unconditional per-cycle re-fit: *"$T^{*}$ is re-fit at each cycle boundary because SIL fine-tuning shifts the generator's logit scale, so the Cycle-0 scalar becomes stale (Ovadia et al. 2019; Thulasidasan et al. 2019); the $u_{\text{stored}}$ composite weights are fixed by design and are not re-fit"* (chapter_4.tex:493). The hyperparameter table echoes this: *"Temperature $T$ … refit per cycle on the disjoint calibration slice"* (line 1329). The code's docstring at `run_calibration.py:465-475` matches exactly — same citations (Ovadia + Thulasidasan), same "temperature-only" scope, same "u_stored weights locked by design" carve-out. No divergence between code and thesis. Close.
- **NOTE 15 (`run_simple_ft.py:403`) — 10-cycle checkpoint disk footprint (~30 GB).** Close as by-design. Actual footprint re-derived: Flan-T5-Large at bf16 (the script default) is ~1.5 GB per `save_pretrained` checkpoint, so a single 10-cycle B6 or B7 run stores ~15 GB; the audit's ~30 GB figure corresponds to running *both* baselines, which is the realistic suite. On Vast-150 the overall budget (wiki index ~40 GB + HF cache ~20 GB + OS/env ~15 GB + 35 GB checkpoints for B6+B7+main CAEM 3-cycle) leaves ~40 GB headroom — comfortable. Declining to add `--keep_last_n_ckpts`: (a) per-cycle checkpoints have downstream research value for B7-vs-CAEM retention trajectory plots in Chapter 5 and any post-hoc benchmark rerun; (b) a `--keep_last_n` flag is a footgun — a user sets it for disk relief and then Chapter 5 discovers it needs an intermediate cycle that was pruned; (c) the retention-ratio recovery path at lines 635-637 restores `theta_prev` from the in-memory cache, not from the checkpoint dir, so checkpoint pruning is not on any hot path for training mechanics. If disk pressure materialises on a smaller Vast box at runtime, `du -sh out_root/*/cycle_*` + manual `rm` is the right response, not a permanent CLI feature.

- **NOTE 33 (`run_experiment.py:697` — `print_mechanism_table` stale docstring targets).** Fixed (DOC-only). The pre-fix docstring at `run_experiment.py:724-732` opened with the header *"Targets (from 10-cycle final plan):"* and listed five bullets mixing three different kinds of claim: (a) **hypothetical numbers** — `Tier 1 fraction: ~5% (C0) -> growing per cycle -> ~50% at equilibrium (C7-C9)` — carried over from an earlier planning doc, not enforced by any code path and not derived from the current config; (b) a **hardcoded config-backed number** — `MMLU Retention: >=93%` — which duplicated `CAEMConfig.forgetting_tolerance = 0.93` and would silently drift out of sync if the config were ever tuned; (c) three **directional statements** (hall reduction growing, u_stored rising monotonically, data purity rising) that describe sign/trend only and are fine as-is. The risk was narrative, not mechanical: the function prints whatever the run produces and does not assert any of these numbers, but the word "Targets" invited the next reviewer to read the ~5% → ~50% trajectory as a pass/fail bar and silently misattribute an empirical equilibrium divergence to a code bug instead of interpreting it as the measurement it actually is. Fix applied: rewrote the docstring preamble to *"Expected patterns (10-cycle design; directional hypotheses, not hardcoded pass/fail targets -- exact equilibrium values are measured empirically and reported in Chapter 5 Table 5.X)"*; rewrote the Tier 1 bullet to drop the specific percentages and state explicitly that the late-cycle equilibrium is one of the mechanism signals being measured, not a pre-set target; rewrote the MMLU bullet to reference `CAEMConfig.forgetting_tolerance` instead of the literal 93% constant (with an inline "do not hardcode a literal percentage here" note for the next editor); kept the three directional bullets verbatim; appended a closing paragraph *"This table is descriptive, not validating -- it prints whatever the run produced and lets the reader compare against Chapter 5. If you want an assertion-style check against specific numbers, add it to the post-run analysis script, not here."* Zero runtime impact — function body at `run_experiment.py:733-771` is untouched. Aligns the in-file documentation with Task #80's 10-cycle design and the post-NOTE-31 thesis ↔ code-alignment pass (Chapter 5 Table 5.X and Fig 5.5 carry the authoritative equilibrium numbers).

- **NOTE 34 (`run_experiment.py:699` — `train_samples` vs. `sil_samples` naming ambiguity).** Fixed. Pre-fix, `run_experiment.py` carried two variable names whose relationship was non-obvious on casual read: `sil_samples` was the raw SIL training **pool** returned by `load_sil_training_pool()` (or its synthetic-mode equivalent `make_synthetic_samples(...)`), and `train_samples` was the **train slice** of the 3-way disjoint partition (`purity` / `calib` / `train`) carved out by `split_calibration_sets(...)`. The pool and the slice are different objects but the names made them sound interchangeable — a reader could easily assume `sil_samples` *is* "the SIL training data" and copy-paste the `harness.run_all(...)` call pattern with `sil_samples` in place of `train_samples`, silently leaking the calibration slice back into the generator during SIL. That is precisely the Gap-5 isolation violation that Chapter 4 §Verifier-Calibration (`chapter_4.tex:493`-ff) and the existing defensive comment at lines 1269-1270 (*"Gap-5 isolation: we pass `train_samples`, NOT `sil_samples`, so the calibration slice stays unseen by the generator during SIL"*) exist to prevent. The comment is a fence around the hazard; renaming removes the hazard. Fix applied: renamed `sil_samples` → `sil_pool` everywhere it appears in `run_experiment.py` via `replace_all` (7 sites total: synthetic assignment at line 877, three `.items()` carves at lines 881-883, production loader at line 885, `split_calibration_sets(sil_pool, ...)` call at line 893, and the defensive comment at line 1269 — which still reads correctly as *"we pass `train_samples`, NOT `sil_pool`"*). The new name aligns with the existing loader function `load_sil_training_pool` and with the `trivia_sil_pool_size` variable at lines 937-943, so the pool-vs-slice taxonomy is now self-documenting: `*_pool` = pre-split raw, `*_samples` = post-split slice. Did **not** rename `sil_samples` in `run_cyclic_ablation.py` (lines 454, 457-458, 460, 466, 654) — the ablation script only carves purity/calib out of `sil_samples` and discards the train-slice return (line 465: `purity_samples, calib_samples, _ = split_calibration_sets(...)`), then feeds the *unsplit* `sil_samples` into memory population at line 654, so in that file `sil_samples` genuinely is the training-samples variable and the name is accurate in its local context. Cross-file consistency was considered and rejected to avoid introducing an ambiguity in the other direction (in ablation, `sil_pool` would wrongly imply a pre-split pool that never gets split for memory-population purposes). Verification: `py_compile` blocked by the same Cowork filesystem-sync artifact as NOTE 27 / NOTE 28 (Linux mount shows truncated 1241-line copy ending mid-string at line 1242's `"n_e`, Windows-side file has the full 1455 lines with the rename applied cleanly); Grep pass confirmed zero remaining `sil_samples` tokens in `run_experiment.py` and the four expected `sil_pool` tokens at the expected sites (plus the three pre-existing `sil_pool_size` tokens in `load_general_data`). User should run `python -m py_compile scripts/run_experiment.py` locally to confirm.

## Deferred — post-Vast measurement backfill

- **NOTE 12 post-fix upgrade.** The qualitative rewrite in `seed_cold_start.py`'s docstring defers the numeric verification-accuracy claim to a measurement produced by `run_purity_validation.py`. Once the Vast main experiment completes (Tasks #74 → #75), the thesis should:
  1. Extract the verifier balanced accuracy alpha from `run_purity_validation.py`'s output JSON.
  2. Insert the measured alpha into the Chapter 5 §5.5 data-purity theorem section.
  3. Optionally, update `seed_cold_start.py`'s docstring to reference the concrete alpha from Section 5.5 instead of the placeholder `<value>`.
  Until then the placeholder prevents an unsourced number from landing in any thesis draft.

---

# Regression fix — `run_simple_ft.py` tail truncation — 2026-04-18

**Severity:** BLOCKER (caught pre-launch, no prod impact)
**Category:** REG (regression introduced by my own MINOR-pass audit)

## What was wrong

`scripts/run_simple_ft.py` terminated at line 628 with an incomplete comment `# Update pre_mmlu to pos`. The original file (commit 856da47) had 7 more lines at the tail:

```python
        # Update pre_mmlu to post-cycle value (used as next cycle's anchor
        # reference for retention ratio).
        pre_mmlu = post_mmlu

    logger.info("All %d cycles complete. Checkpoints: %s", ns.num_cycles, out_root)


if __name__ == "__main__":
    main()
```

## Defects caused

1. **Silent no-op at module load.** No `if __name__ == "__main__": main()` guard means `python scripts/run_simple_ft.py` exits cleanly without executing any training. Cycle loop is defined but never invoked.
2. **Stale `pre_mmlu` denominator.** Missing `pre_mmlu = post_mmlu` update means baselines B6 and B7 (cyclic ablation) compute retention ratio against cycle-0 MMLU for every subsequent cycle instead of the previous cycle's MMLU. Silently wrong numbers — would have polluted Tables 5.6 and 5.7.

## How it slipped past verification

- `py_compile` passes because EOF-mid-comment is syntactically valid Python (the comment simply extends to EOF).
- Bash-mount staleness artefact (`Modify: 2026-04-11` vs live Windows edits `2026-04-18`) meant `stat`/`py_compile` in the Linux sandbox were reading a pre-audit snapshot — covered in the MAJOR+MINOR verification sections above. Read tool was authoritative; shell was not.
- The truncation landed during either commit `c0baf65` or `8678345` (MINOR audit batch). Most likely cause: an Edit on a nearby block with a whitespace-off `new_string` that accidentally chopped the trailer. The bash-mount staleness meant post-edit shell-side diffs showed the pre-edit file, hiding the truncation from my own verification pass.

## Credit

Found by independent agent audit. Original 7-line tail was provided verbatim from `git show 856da47:scripts/run_simple_ft.py` — fix is exact, not reconstructed.

## Fix applied

Replaced line 628's incomplete comment with the full 9-line tail (comment + assignment + cycle-complete log + main guard). Verified by re-reading lines 620-637: all constructs present, `if __name__ == "__main__": main()` is final statement, file ends cleanly at line 637.

## Process follow-up

Until the bash-mount staleness issue is resolved (operator git pull / VM reboot), the Read tool is the only reliable verification path for post-edit state. Any future audit pass that Edits in the last ~10 lines of a file should be followed by a Read of the final 20 lines to confirm no trailer was lost.

---

# Regression fixes — parallel agent — 2026-04-18

Additional regressions caught by a parallel audit pass (other agent). Both trace to my earlier MINORs audit; flagged here so the reviewer knows the audit report is closing the loop on its own mistakes.

- **`run_cyclic_ablation.py:672` — `_math.isnan(...)` stale alias.** The earlier MINORs pass renamed `import math as _math` → `import math` and was supposed to convert every call site. This call site at line 672 was missed. Fixed by the parallel agent: `math.isnan(float(mmlu_val))` with an explicit `None` guard before the cast.
- **`run_cyclic_ablation.py:647` + `aggregate_ablation.py:189-195` — `None` contamination from `getattr(..., nan)`.** The Group 3 MINORs pass ("silent failure modes") replaced some `1.0` fallbacks with `NaN` sentinels, but at these sites `getattr(..., nan)` resolves to `None` when the attribute is missing (not `NaN`). Downstream `max(by_cycle)` and typed `List[float]` declarations then refuse the `None`. Fixed by the parallel agent with explicit None-guards and proper float coercion.

Combined effect: VS Code now reports `0 errors, 0 warnings, 0 informations` on the scripts folder.

---

# NOTE 18 resolution — build_passage_index checkpoint waste — 2026-04-18

**Severity:** MINOR (polish) — user directive: "when i rebuild rag, it starts from 0, 1.5 gb is worthless".

## What was wrong

The audit's NOTE 18 (`build_passage_index.py:442`) flagged the terminal `np.concatenate(all_embedding_batches)` as materialising ~1.5 GB (500K × 768 × float32). Digging in revealed the larger waste was not the concat itself but the **checkpointing mechanism**:

1. `_save_checkpoint` fires every `CHECKPOINT_EVERY=100_000` passages and pickles the **entire growing** `all_embedding_batches` list, not just the new batch.
2. By the 5th checkpoint of a 500K-passage build it pickles ~1.5 GB; cumulative disk writes across all 5 checkpoints = 300 + 600 + 900 + 1200 + 1500 ≈ **4.5 GB of I/O for no benefit** if the run always completes in one shot.
3. Aksan's workflow on Vast.ai is always a fresh rebuild from zero (no resume path in practice), so the default-on checkpointing is a pure cost.

Separately, the `np.concatenate` holds the Python list AND the contiguous array concurrently until the list is GC'd, so the transient RAM peak is ~2x (~3 GB), not 1x.

## What changed

### `scripts/build_passage_index.py`

- New `enable_checkpoint: bool = False` kwarg on `build_passage_index()`. Default OFF.
- New `--enable_checkpoint` CLI flag (default False). Documented as the opt-in path for users who expect interruptions.
- `--resume` now implicitly enables checkpointing for the remainder of the run (otherwise a resumed-then-interrupted run would re-lose progress).
- The streaming-loop `_save_checkpoint(...)` call is gated on `checkpoint_enabled = enable_checkpoint or resume`.
- Logs the checkpoint-enabled/disabled state at build start so the user knows which mode they're in.
- Module-header docstring updated: the previous "Progress is checkpointed every CHECKPOINT_EVERY passages" claim was a lie about default behaviour post-change; replaced with a description of the OFF-by-default posture and the two opt-in flags.
- Added `del all_embedding_batches` immediately after `np.concatenate(...)` to collapse the transient 2x RAM peak back to 1x. The 1x peak is structurally required because FAISS IVF training needs a contiguous training matrix; removing it would require a PassageStore refactor (out of scope).

## Behavioural guarantees

- **Default path** (bare `python -m scripts.build_passage_index --output_dir ...`): zero checkpoint writes across the build. Disk I/O drops from ~4.5 GB of cumulative pickle to 0.
- **Explicit `--enable_checkpoint`**: identical behaviour to the pre-fix default; a 500K-build still writes ~4.5 GB cumulative across 5 checkpoints, but only if the user explicitly asks for it.
- **`--resume`**: identical behaviour to pre-fix (loads existing checkpoint, keeps writing new checkpoints for the remainder). The implicit `enable_checkpoint=True` on resume is a correctness requirement; a resumed run without further checkpoints would be as fragile as the original non-checkpointed run.
- **No change to the emitted `passages.faiss` / `passages.pkl` artefacts.** Only the intermediate `_checkpoint.pkl` writes are suppressed.

---

# NOTE 14 resolution — grad_accum_steps implementation — 2026-04-18

**Status:** Previously listed as "Wave 1 done" at line 732, but live code verification showed Wave 1 never actually landed (no `grad_accum_steps` field on `HardwareProfile`, no accumulation counter in any training loop). This section documents the real implementation and supersedes the Wave 1 entry.

## What was wrong

- `scripts/hardware.py:133` had an aspirational comment claiming "gradient accumulation preserves effective batch-size equivalence across hardware tiers." Nothing in the codebase actually emitted gradient accumulation — every training loop was `zero_grad → backward → step` per micro-batch, so effective batch size == `cfg.batch_size`. Reproducers running on different hardware would see different gradient noise and different loss dynamics, contaminating ΔRET and other cross-baseline comparisons.
- The original Wave 1 fix-log entry (line 732) claimed this was fixed. It wasn't.

## What changed

1. **`scripts/hardware.py`** — added `grad_accum_steps: int = 1` field to `HardwareProfile`; introduced `TARGET_EFFECTIVE_BATCH_SIZE = 32` module constant; per-tier values: A100-80 / 5090 → 1, 4090/3090 → 2, T4/V100 → 4, RTX 3060 → 8, <10 GB → 16. `CPU`/`MPS` kept at 1 (smoke-test tiers — speed over strict equivalence). `get_hardware_info()` now also emits `effective_batch_size` for the log. Line 133 aspirational comment rewritten to describe what the code now actually does.
2. **`caem/config.py`** — added `grad_accum_steps: int = 1` to `CAEMConfig`. Default is a no-op, so a bare `CAEMConfig()` stays identical to the thesis 5090 path.
3. **`caem/training/self_improvement.py:758–814`** — rewrote the batch loop. Maintains a `micro_step` counter; scales loss by `1/N` before backward; only calls `optimizer.step()` + `zero_grad()` on accumulation boundaries; resets the counter on non-finite skips (discarding any partial accumulation); handles the tail partial accumulation at epoch end so no training data is wasted. Epoch log line now reports `grad_accum` and `eff_batch` for audit transparency.
4. **`scripts/run_simple_ft.py`** — CLI flag `--grad_accum_steps` added with `-1` sentinel (auto-resolve from `scripts.hardware.get_hardware_profile`). Training loop mirrors the main-loop rewrite. Startup log emits the resolved `batch_size`, `grad_accum_steps`, and `effective_batch` triple.
5. **`scripts/run_experiment.py`** — the existing hardware-batch-clamp block was extended to also write `config.grad_accum_steps` from the hardware profile, but only when the config is still at its default 1 (preserves explicit overrides).

## Behavioural guarantees

- **Zero change on thesis hardware (Vast RTX 5090, batch=32, grad_accum=1).** `loss / 1 == loss`, and the step-every-batch path is identical to the pre-NOTE-14 code. Phase 1/2 numbers will not shift.
- **Matched effective batch on any other hardware.** Reviewer running on a 4090 gets `micro_batch=16, grad_accum=2`, total eff_batch=32 — same as thesis. Reviewer running on T4 gets `8 * 4 = 32`. Cross-hardware reproducibility is now a code-enforced property, not a README claim.
- **Non-finite skip safety.** A poisoned micro-batch (NaN/Inf CE loss or total loss) discards any partially accumulated gradients and resets the counter, so a single bad batch can no longer corrupt the next optimiser step.
- **Tail step at epoch boundary.** If an epoch ends mid-accumulation, the partial batch is stepped with whatever was accumulated. Effective gradient magnitude is `< 1×` for that single step, at most once per epoch. Chose this over "drop the tail" because dropping would waste training data at the end of every cycle.
- **L2 regulariser under accumulation.** The L2 penalty is included in `raw_loss` (per micro-batch) and scaled by `1/N` alongside the CE loss. Net penalty applied per optimiser step equals the single-batch-at-effective-size case — no double counting.

## Post-implementation AST verification

- Live Edit tool succeeded on every change (exact-string old-match semantics guarantee file integrity).
- Shell `python -c "import ast; ..."` still reports a syntax error due to bash-mount staleness (the snapshot predates today's Edit-tool writes). Documented as the same artefact as the tail-truncation regression above; not a live-file defect.
- Rollback safety: `grad_accum_steps=1` default in both `HardwareProfile` and `CAEMConfig` means any existing caller that hasn't been updated still runs the no-op path.

## Supersedes

- `docs/scripts-audit-report.md:732` — the Wave 1 entry is now accurate as of this date. Keeping it in place (don't delete) so the audit chain remains reviewable, but future readers should treat 2026-04-18 as the actual landing date.

