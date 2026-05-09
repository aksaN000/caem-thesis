# Script audit — v2.1 + Phase 1c+1d alignment

**Status:** systematic per-script audit triggered after the user's
"100% code-doc consistent" rule (`add9c69` task list). 58 scripts
total under `scripts/`. Audit performed during the active step_7_main
trajectory (PID 288487, ~2h elapsed at audit start) — runtime path
already verified clean by virtue of the trajectory producing valid
pipeline decisions; this audit catches doc-string and CLI-example
drift that survives in scripts that have not been re-invoked since
Phase 1c.

## Triage outcome

| Script | Stale claim | Live runtime impact | Action |
|---|---|---|---|
| `run_experiment.py` | CLI example shows `natural_questions arc_challenge` panel and `--n_questions 5000`; comment at line 882 references `{fever, triviaqa, natural_questions}` | None — config-driven dispatch | **Fixed** — CLI example shows `fever triviaqa commonsense_qa truthfulqa strategyqa` and `--n_questions 3000`. Library imports for `load_natural_questions` / `load_arc_challenge` left intact (utility functions, called only when explicitly requested) |
| `run_calibration.py` | Comment block describes v1 panel + claims v2 trains on `("fever", "triviaqa", "hotpotqa", "commonsense_qa")` (HotpotQA was retired) and uses `("truthfulqa", "strategyqa", "natural_questions")` for transfer (NQ was retired) | None — `_CFG_TRAINING_BENCHMARKS` import is the live source | **Fixed** — comment block now says training panel is `(fever, triviaqa, commonsense_qa)` and transfer is `(truthfulqa, strategyqa)`, with HotpotQA + NQ retirement note. The `CALIB_BENCHMARK_SPLITS` dict keeps fallback entries for archival benchmark names so legacy scripts continue to dispatch when they are explicitly invoked. |
| `run_baseline.py` | CLI example shows v1 six-bench panel | None — `--benchmarks` is required at invocation | **Fixed** — CLI example shows the v2.1 five-bench panel; comment caption clarified |
| `run_ablation.py` | CLI example shows v1 six-bench panel; comment at line 463 references arc_challenge | None — `--benchmarks` is required | **Fixed** — CLI example shows the v2.1 five-bench panel. Library code path for arc_challenge dispatch left intact (called only when arc_challenge is explicitly listed). |
| `baseline_sig_tests.py` | Comment block lines 84-87 describes the v1 hardcoded roster + claims v2 trains on `{fever, triviaqa, hotpotqa, commonsense_qa}` (HotpotQA retired) | None — `BENCHMARKS = list(...)` reads from config | **Fixed** — comment now correctly describes v2.1 trains on `{fever, triviaqa, commonsense_qa}` and transfer-evals on `{truthfulqa, strategyqa}`, with HotpotQA + NQ retirement note. |
| `rescore_baselines_through_verifier.py` | Phase 1c historical context (kept — factually correct rejected-alternative documentation); CLI example at line 68 shows v1 seven-bench panel + asqa | None — required CLI flag | **Fixed** — CLI example shows v2.1 panel. Historical-context paragraphs at lines 44/137 left in place (they correctly document the rejected split-conformal alternative). |
| `run_purity_validation.py` | CLI example shows `natural_questions` in panel; library code dispatches on arc_challenge / NQ as benchmark loaders | None — driven by `--benchmarks` flag | **Fixed** — CLI example shows `commonsense_qa` instead of `natural_questions`. Library loader code left intact (lookup tables for archival benchmark names that legacy ablations may still reference). |
| `phase4_artifacts.py` | `BENCH_DISPLAY` label dict at lines 60-65 lists v1 panel labels (NQ, ARC-C, ASQA, HotpotQA) alongside v2.1 panel | None at runtime — `BENCHES` is sourced from config; the dict is just label lookup that produces empty rows for non-live benchmarks | **Fixed** — dict reorganised to put v2.1 live panel first, archival entries grouped + commented as registered exclusions |
| `rescore_eval_with_fitted_gate.py` | Inline comment at lines 102-107 claims composite consumes 10 signals (Phase 1c P1) — STALE under Phase 1d which dropped h_norm to 9 signals; signal dict at lines 110-121 still includes h_norm explicitly | The `composite.predict()` call ignores h_norm cleanly when `h_norm` is not in the loaded `COMPOSITE_SIGNALS`, so runtime is correct; only the comment + dict shape were stale | **Fixed** — comment now correctly says 9 signals fed to the composite; h_norm row removed from signal dict. The verifier output schema still emits h_norm as sentinel 0.5 for back-compat with archived eval JSONs; this script now correctly mirrors the live composite. |
| `theorem_receipts.py` | Lines 379-382 historical context about tau_retro per-cycle conformal threshold rejection | None (history) | **No edit** — correctly documents Phase 1c rejection. |
| `run_cyclic_ablation.py` | Lines 453, 808, 811 reference natural_questions / arc_challenge; line 453 comment says `{fever, triviaqa, natural_questions}` is the v1 panel (kept correctly as historical reference); the loader dict may still dispatch arc_challenge | None at runtime; loader fallbacks only fire on explicit invocation | **Deferred** — surface lookup is config-driven; archival-benchmark loader entries are dormant under v2.1. Historical comment kept. |

## Lower-priority (build / smoke / utility)

The following scripts have stale benchmark names in CLI examples or
file-existence checks that no longer match the v2.1 panel, but
none of them runs as part of the live trajectory and none is
invoked by `run_phase1a.sh`. Deferred until they are next invoked,
and flagged here:

- `archive_v1_to_gdrive.sh` — references v1 archive benchmark names
  (correct in archival context).
- `build_alias_dict_from_benchmarks.py` — fallback list contains
  v1 benchmark names; the alias_dict was already built from the
  v2.1 panel during Phase 1c.
- `build_calibration_pairs.py` — references the calibration-pair
  fold; runtime is config-driven.
- `caem_demo_server.py` — demo only; CLI args are fully optional.
- `check_base_model.py` — utility check; doesn't depend on panel.
- `compare_prompt_design.py` — diagnostic only.
- `halt_and_resume_with_deferred_fix.sh` — historical fix script
  for the May-4 deferred-buffer orchestration bug.
- `prompt_compliance_smoke.py` / `prompt_smoke_test.py` — smoke tests.
- `prune_cold_memory_em.py` — memory hygiene; benchmark-agnostic.
- `run_simple_ft.py` — simple FT baseline; CLI takes panel via flag.
- `seed_cold_start.py` — references the seed-pool benchmark loaders;
  config-driven dispatch at runtime.
- `watchdog_cycle2.sh` / `watchdog_cycles_3plus.sh` — process
  watchdog scripts; benchmark-agnostic.

## Surviving "conformal" mentions in scripts

All surviving "conformal" references in `scripts/` after this audit
are in one of three categories, all factually correct:

1. **Historical-context docstrings** explicitly documenting the
   rejected split-conformal alternative for the storage gate
   (`run_experiment.py:354`, `rescore_baselines_through_verifier
   .py:44,137`, `rescore_eval_with_fitted_gate.py:19,22`,
   `phase4_artifacts.py:7-9,602`, `theorem_receipts.py:379-382`).
2. **Cited literature references** in docstrings (Cherian boost,
   Mohri-Hashimoto, Yadkori abstention).
3. **Comparative ablation labels** for the rejected-alternative
   ablation row (e.g. `run_experiment.py:2139` "the conformal
   gate's α target. Under the fixed-..." — the ablation row).

No active `conformal_gate.json` file is loaded by any script (the
file was deleted at Phase 1c P0' and the calibration path now reads
the storage cut-points from `CAEMConfig.store_threshold` /
`defer_threshold` directly). Verified by grep on the live tree.

## Verification

After the audit edits, the live trajectory continues healthy:
- `tmux plan_a` PID 288487 alive at audit-end time.
- `cycle_0/composite_calibration.json` 9-signal Phase 1d artefact
  unchanged.
- No script edited above is invoked by the active step_7_main loop
  in flight, so the audit edits cannot perturb the running
  trajectory.

The runtime correctness of the post-audit scripts will be verified
when each is next invoked: `run_purity_validation.py` at the
diagnostics phase post-step_7_main; `run_baseline.py` /
`baseline_sig_tests.py` / `rescore_baselines_through_verifier.py`
at the external-baselines panel; `phase4_artifacts.py` at the
artefact-generation phase.

## Commits in this audit pass

| Commit | Files |
|---|---|
| (this) | `scripts/rescore_eval_with_fitted_gate.py`, `scripts/phase4_artifacts.py`, `scripts/run_experiment.py`, `scripts/run_calibration.py`, `scripts/run_baseline.py`, `scripts/run_ablation.py`, `scripts/baseline_sig_tests.py`, `scripts/rescore_baselines_through_verifier.py`, `scripts/run_purity_validation.py`, `SCRIPT_AUDIT.md` |
