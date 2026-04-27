# CAEM — Next Session Plan

**Updated 2026-04-27 09:15 UTC | Session 9 — citation audit + step_7_main monitoring + Phase 4 prep**

---

## State at session boundary (2026-04-27 09:15 UTC)

**`step_7_main` 10-cycle main run is LIVE in tmux `plan_a`.** Session 8 today landed seven architectural patches that finally got the cycle boundary working. Cycle 1 is in mid-flight:

```
07:50  Cycle 1 SIL fine-tune begins (137 verified episodes from cold-seed)
07:53→07:54  Epoch 1/2/3 → loss 1.2771 / 0.8380 / 0.7219
07:55  Post-fine-tune MMLU = 0.6250 → ρ = 1.0000 PASS (floor 0.93)
08:00  gdrive offload OK → cycle_1/model.pt; retroverify=0/0 (correctly skipped)
08:00  Step 2.1 calibration-fold scoring under post-SIL model BEGINS
09:04  FEVER calibration fold complete (cycle_1/calibration/fever_cycle1.json on disk)
09:04  TriviaQA calibration fold begins
NOW    TriviaQA fold ~25% in, NQ fold pending; Step 2.2-2.5 pending
```

**Cron `199ba368` fires every 30 min (:13 and :43)** to keep autonomous status checks running.

---

## What today's session (Session 8) accomplished

Seven coordinated patches that took the runner from "OOM at every cycle attempt" to "trajectory in flight under the architecturally-correct cycle-boundary order":

### Memory mitigation (5 patches)
1. **bf16 anchor on CPU + PCIe stream** (`caem/training/self_improvement.py`) — replaced fp32-on-GPU theta_prev with CPU-resident bf16 streamed per-parameter via PCIe at L₂-step. Saves ~12 GB GPU.
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (`run_phase1a.sh`) — defragments the allocator; cut the unused-but-reserved overhead from 646 MB → 77 MB.
3. **`torch.compiler.set_stance("force_eager")` for SIL backward** — bypasses inductor's compiled SDPA backward that materialises full-sequence attention gradient buffers. Saves ~3 GB.
4. **`batch_size=4, grad_accum_steps=4`** (`caem/config.py`) — quarters per-step memory while keeping effective batch at 16. Cuts the 4.37 GiB single-allocation OOM.
5. **L₂ penalty one-shot device+dtype cast** (`caem/training/self_improvement.py:_l2_penalty`) — `p0.to(device, dtype=p.dtype, non_blocking=True)` replaces the two-step transfer that double-allocated. Saves transient peak.

### Architectural fixes (2 patches)
6. **Early-exit `is_query_time` gate** (`caem/verification/verifier.py:verify`) — confabulation early-exit now scoped to query-time inference; retroverify passes False to avoid spurious-prune of EM-correct stored entries on memorised chains.
7. **Cycle-boundary recalibration WIRED** (`scripts/run_experiment.py` Steps 2.1-2.5 + `caem/verification/verifier.py:reload_calibration`):
   ```
   Step 1   SIL fine-tune (verify_fn=None, no internal retroverify)
   Step 2.1 score calibration fold under post-SIL model → cycle_{N}/calibration/
   Step 2.2 re-fit T (ECE-min)
   Step 2.3 re-fit isotonic + conformal τ (label-dependent, EMA α=0.7)
            → cycle_{N}/composite_calibration.json + conformal_gate.json
   Step 2.4 pipeline.verifier.reload_calibration(...) — verifier rebound
   Step 2.5 retroactive re-verify under recalibrated verifier
   ```

### Documentation
- `README.md` — full rewrite reflecting current state
- `docs/PRODUCTION_RUNBOOK.md` — end-to-end production deployment recipe
- `thesis_report/appendix/appendix_g.tex` — architecture-positioning Q&A:
  - **Part I:** 6 entries on positioning vs plain fine-tuning (5 differentiators + when-not-to-deploy)
  - **Part II:** 6 entries on design-justification (same fold across cycles, production-fold differs, recalibrate-before-retroverify, is_query_time gate, τ_train > τ_store, three-layer dedup)
- `thesis_report/core/abstract.tex` — written, 522 words, 5 strategic citations, training/transfer-panel split made explicit
- `thesis_report/chapters/chapter_4.tex` §4.2.7.3 — rewritten under the corrected cycle-boundary order; production-parity paragraph fixed
- `thesis_report/chapters/chapter_5.tex` §5.6 — boundary-of-architectural-value paragraph appended

---

## What's next — Session 9 priorities, in order

### P1 — Citation audit (NEW, manual verification task)

**Goal:** walk every `\cite{...}` reference in the thesis tree and verify against `bibliography/references.bib` AND against the cited paper's actual content. The body-prose claims about prior work must be empirically true. The citation audit covers three layers:

| Layer | What gets verified | How |
|---|---|---|
| Key existence | every `\cite{KEY}` resolves to an entry in `references.bib` | `grep -oE '\\cite[pt]?\{[^}]+\}' chapters/*.tex` → join against `references.bib` keys |
| Metadata correctness | the bib entry's author, year, venue, title match the cited paper | per-entry against arxiv / semantic-scholar / venue page |
| Claim-attribution correctness | the prose's claim about the paper is what the paper actually says | per-citation against the paper's actual content (abstract + relevant section) |

**Layer 3 is the load-bearing audit.** A claim like "Mohri-Hashimoto demonstrate a precision lift from 78% to 93% on Natural Questions" must trace to the exact numbers in their paper. A claim like "Farquhar et al. report AUROC 0.790 on fabrication detection" must trace to their specific reported figure. Misattribution is the failure mode this audit prevents.

**Scope:** all `\cite` entries in `thesis_report/chapters/{chapter_1..6}.tex` + `thesis_report/appendix/{appendix_a..g}.tex` + `thesis_report/core/abstract.tex`. Approximately 84 cited entries per the README.

**Audit method (proposed):** I'll generate an audit checklist file at `thesis_report/audit_phase_a_citations.md` with one row per citation, listing:
- Citation key
- File + line where cited
- The prose claim that surrounds the citation (excerpt)
- The bib entry's title + venue + year
- A "claim verified against paper" column to fill manually

User then walks through the checklist and ticks off each row, flagging any misattribution. Misattributions get fixed (either tighten the prose or swap the citation).

### P2 — Continue monitoring step_7_main

The 30-min cron handles this autonomously. Critical milestones to watch:
- ~10:50 UTC — Step 2.1 (calibration-fold scoring) finishes. The `outputs/full_run/cycle_1/calibration/{fever,triviaqa,natural_questions}_cycle1.json` files complete.
- ~10:51 UTC — Step 2.2-2.4 fire. Log line: `Verifier calibration reloaded: composite=outputs/full_run/cycle_1/composite_calibration.json | gate=...`
- ~10:51-11:25 UTC — Step 2.5 retroverify on 260 cold-seed entries.
- **~11:25 UTC — `Retroverify complete: N updated, M removed` log line.** This is the empirical-fix-worked confirmation:
  - **M ≪ 139** (e.g., 10–30) → architectural fix worked. Recalibrated isotonic curves correctly absorb the post-SIL signal-distribution drift. ✅
  - M ≈ 139 → recalibration didn't change retroverify behavior; would need investigation.
  - M > 139 → fresh recalibration is more aggressive than Cycle-0; would require config tweak.
- ~11:25 UTC onward — Cycle 1 Step 4 (ingest 9000 fresh queries from cycle stream chunk) + Step 5 (eval pass over 3500 queries × 7 benchmarks).
- ~14:00 UTC — Cycle 1 done. Cycle 2 SIL fine-tune begins.

**Total trajectory ETA at current cadence:** ~10 cycles × 4-6h each ≈ **2-3 days wall clock** (vs initial 7-8 day estimate; cycle pace is faster than projected because steps 2.1-2.5 run in parallel with the eval pass on different data).

### P3 — Phase 4 readiness checklist (gated on step_7_main + #112 + #113)

Once trajectory lands:

- [ ] **Task #111 — Halt + rewire B6/B7/ablation dynamic cycle count.** B6 vanilla FT and B7 EWC-only FT in `scripts/run_baseline.py` currently hardcode `--num_cycles 10`. If CAEM equilibrium-gate early-stops at, say, Cycle 6, B6/B7 must run 6 cycles too for matched-scale comparison. Patch: read realised cycle count from `outputs/full_run/equilibrium_gate.json` at runner-launch.
- [ ] **Task #112 — External baselines + significance tests.** B1-B7 on test fold under matched protocol; paired McNemar + BCa bootstrap + Holm correction; outputs go to `outputs/significance/per_benchmark.json` + `tab_sig_test.tex`.
- [ ] **Task #113 — Diagnostics.** `scripts/run_purity_validation.py`, `cycle2_retention_diagnostic.py`, `signal_correlation_matrix.py`, `aggregate_ablation.py`. Outputs feed `tab_purity.tex`, `tab_continual.tex`, `tab_calibration_trajectory.tex`.
- [ ] **Task #114 / Phase 4 — Ch5/Ch6 artefact generation.** 12 stub `figures/auto/*.tex` files get overwritten by real numbers. Trigger via `scripts/make_tables.py` + `scripts/make_figures.py` + `scripts/aggregate_calibration_trajectory.py` + `scripts/render_production_samples.py` + `scripts/phase4_artifacts.py`. Cross-check every `\Cref{tab:*}` / `\Cref{fig:*}` in chapters against the now-populated `figures/auto/`.

### P4 — Final thesis polish (post-Phase 4)

- [ ] Compile thesis_report/main.tex once; fix any unresolved \Cref / \cite errors
- [ ] Recompile after Phase-4 auto-tex regen
- [ ] Re-read Chapter 5 / 6 prose end-to-end after real numbers replace stubs (some prose framing may need tightening)
- [ ] Bibliography ordering check (Cref-style)

---

## Open architectural questions (registered, not blocking the trajectory)

1. **Within-cycle calibration drift in production with long cycles.** §5.6 §"Threats to Validity" registers this; production runbook §8 quantifies it (4-8% ECE inflation at 60-90 days, 5-15% at 120+ days). Mitigation: drift-triggered cycle boundaries that fire before the calendar cadence.

2. **Rotating-fold protocol for trajectories beyond 10 cycles.** Same calibration fold every cycle is the right protocol at the 10-cycle horizon (variance isolation dominates); at 50+ cycles a rotating-fold partition over a 5000-sample reservoir would reduce same-fold over-fitting risk. Registered as Phase 1b future-work.

3. **Long-hypothesis Qwen judge under task-specific fine-tune.** Currently ablated because it fails the registered Pearson floor of 0.70 against MiniCheck on the v5 overlap fold. Refitting on a synthetic claim-support corpus may re-license inclusion. Phase 1b.

4. **LoRA SIL primary path.** Full-parameter SIL is feasible at 3B-parameter scale on consumer-grade GPU; at 7B+ a parameter-efficient adapter alternative is the only practical path. Phase 1b.

---

## Quick-reference

| Document | Purpose |
|---|---|
| `run_phase1a.sh` | canonical 30+-step Phase 1a runbook (idempotent, resumable) |
| `docs/PRODUCTION_RUNBOOK.md` | production deployment + per-cycle labelled refresh recipe |
| `README.md` | repo-level overview + locked-configuration reference |
| `thesis_report/main.tex` | Ch 1-6 + Appendix A-G manuscript |
| `thesis_report/appendix/appendix_g.tex` | architecture-positioning + design-justification Q&A |
| `branch_C_log.md` | dated implementation diary |
| `caem-implementation-log.md` | longer-form decision rationale |

| Key path | What it is |
|---|---|
| `outputs/cycle_0/composite_calibration.json` | locked Cycle 0 isotonic + boost |
| `outputs/cycle_0/conformal_gate.json` | locked Cycle 0 τ_store=0.6676, τ_defer=0.5207 |
| `outputs/full_run/cycle_{N}/composite_calibration.json` | per-cycle refreshed isotonic + boost |
| `outputs/full_run/cycle_{N}/conformal_gate.json` | per-cycle refreshed τ |
| `outputs/full_run/cycle_{N}/calibration/{bm}_cycle{N}.json` | per-cycle calibration fold scored under post-SIL |
| `outputs/full_run/eval/{bm}_cycle{N}.json` | per-cycle eval (500 q × 7 benchmarks) |
| `outputs/full_run/memory_store_cycle_{N}.{faiss,meta}` | per-cycle memory snapshot |

---

## Resume protocol if compaction happens mid-session

If the conversation gets compacted, re-orient by:
1. Read this file
2. Read `branch_C_log.md` last 10 dated entries
3. Read `caem/training/self_improvement.py:_finetune` (lines 795-870) for the cycle-1 OOM patches
4. Read `scripts/run_experiment.py:1521-1580` for the cycle-boundary order
5. Check `tmux ls` and `tail -20 outputs/full_run/run.log` for live runner state
6. The 30-min cron `199ba368` continues firing autonomous status checks regardless of conversation state

---

**End of plan.** Next: start P1 citation audit. Generate the audit checklist file and walk it together.
