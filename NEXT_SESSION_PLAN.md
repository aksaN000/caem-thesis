# CAEM — Next Session Plan

**Updated 2026-04-28 15:10 UTC | Session 10 — autonomous Step 7 main + Phase 4 receipts after return**

---

## State at handoff

**Runner:** alive in `tmux plan_a`, relaunched at 14:56:25 UTC under fully-fixed pipeline. Cycle 1 SIL just completed (loss 1.276 → 0.721, MMLU retention 1.0080). Currently in checkpoint save → Step 2.1 cal-fold scoring.

**Bugs found and fixed today (2026-04-28):**

| Commit | Fix |
|--------|-----|
| `517e09e` | Per-cycle conformal refit inherits `α_store` from previous cycle's gate JSON instead of config default (was silently drifting 0.05 → 0.20). |
| `6adc695` | Per-cycle refit passes `--cherian_boost --boost_C 0.01` to match locked Step 7.0.1 baseline (was producing identity-mode composite, intercept=0). |
| `b4d610d` | `CAEMConfig.conformal_alpha_store` default pinned to 0.05 (belt-and-suspenders fallback). |

**Recovery (out-of-tree, outputs/ is gitignored):**
- `outputs/cycle_0/composite_calibration.json` restored from `outputs/.snapshot_staging/archive/pre_step7_main_2026-04-26/` (boost_intercept=1.099, q_a_relevance=0.500, em_rate=0.243).
- `outputs/cycle_0/conformal_gate.json` restored to α=0.05, τ_store=0.6676 from `outputs/cycle_0/sweep/variant_a050_C0.010.json`.
- Two contaminated cycle_1 directories preserved at `outputs/_halted_runs/cycle_1.contaminated_alpha020_*` and `outputs/_halted_runs/cycle_1.composite_corrupted_*`.

**Lock chain confirmed:** α=0.05 + Cherian boost + C=0.01 propagate cycle 0 → cycle 10 via three redundant paths (prev_gate inheritance, canonical-path copy at cycle close, config dataclass default). No drift path remains.

---

## Autonomous monitor while user is away

`/loop` wakeup at 60-min cadence. Per-cycle progress logged; cycle-1 boundary refit is the empirical landmark — should show `boost=fit` AND `alpha_store: 0.05` in the new `cycle_1/conformal_gate.json`. Auto-restart on crash per `RESUME_GUIDE.md`. **No halt actions planned.**

ETA Step 7 main complete: ~May 5. ETA Phase 4 (auto-runs after Step 7 per `run_phase1a.sh`): ~May 6–7.

---

## P1 — On return, verify Step 7 main completed

```bash
cd /workspace/caem
ls outputs/full_run/run_complete.json     # exists if Step 7 finished
cat outputs/full_run/experiment_summary.csv | head -15   # 11 rows = 10 cycles + cycle 0 baseline
grep -c "decision=STORE" outputs/full_run/run.log
```

If `run_complete.json` exists: Step 7 main is done; Phase 4 (Steps 8–21) may already be auto-running per `run_phase1a.sh`. If runner is still in Step 7: check tmux log for the cycle index, let it complete naturally.

---

## P2 — Run Phase 4 theorem receipts MANUALLY

**IMPORTANT:** `scripts/theorem_receipts.py` was wired into `run_phase1a.sh` on 2026-04-28 as `step_19_6_theorem_receipts`, but the **currently-running bash runner has the OLD `ALL_STEPS` array cached in memory** and will NOT pick up the new step. The wiring helps future runs only. **For this run, the receipts must be invoked manually after Phase 4 completes.**

After Step 7 main + Phase 4 (B1-B7, purity, correlations, aggregate, tables) finishes:

```bash
cd /workspace/caem && source /venv/main/bin/activate
python -m scripts.theorem_receipts \
    --output_dir outputs/full_run/theorem_receipts \
    --full_run_dir outputs/full_run \
    --summary_csv outputs/full_run/experiment_summary.csv
```

Outputs (six JSONs):
- `receipt_envelope_fit.json` — geometric envelope fit (Theorem 5)
- `receipt_eps_arch.json` — joint architectural FNR aggregation (Theorem 7)
- `receipt_gap_decay.json` — exponential decay fit (Cor convergence-rate)
- `receipt_corpus_floor.json` — top-k retrieval miss subset proxy (Cor corpus-floor)
- `receipt_self_correction.json` — survival distribution (Cor self-correction)
- `receipt_tau_retro_sensitivity.json` — τ_retro at 0.50 vs empirically-optimal per cycle

---

## P3 — Document the bug-fix story in Ch5 §threats

One paragraph (~150 words) added to `thesis_report/chapters/chapter_5.tex` near the threats-to-validity subsection:

> *"During cycle 1 of Step 7 main on 2026-04-28, the per-cycle conformal refit was discovered to drift from the locked Cycle-0 calibration via two independent code paths. First, `run_experiment.py:run_per_cycle_conformal_refit` read `α_store` from the dataclass default (0.20) instead of inheriting from the previous cycle's gate JSON (0.05). Second, the same function did not pass `--cherian_boost --boost_C 0.01` to `recalibrate_conformal_at_cycle.py`, producing an identity-weighted composite (boost intercept zero, no Cherian L2 fit). Both bugs were fixed in commits 517e09e, 6adc695, b4d610d. Cycle 1 was re-run from scratch under the corrected pipeline; cycles 2–10 inherit α=0.05 + Cherian boost via three converging fallback paths (prev-gate JSON, canonical copy, config default). The Phase 4 receipts in §sec:check-* track the empirical realisation of the locked Cycle-0 contract across the corrected trajectory."*

---

## P4 — Refine τ_retro = 0.50 prose in Ch4

The cal-fold sensitivity reading shows π_retro at τ=0.50 is ~0.44, not 0.50. Drop the "more-likely-correct-than-wrong" framing.

**Edit `thesis_report/chapters/chapter_4.tex` §retroverify:**

```diff
- The prune threshold is registered at τ_retro = 0.50, below the deferral
- threshold so that an entry that is borderline-deferred-quality but no longer
- storage-quality is still kept for the deferred-reconsider pass
+ The prune threshold is registered at τ_retro = 0.50: a recall-favouring
+ floor that maintains the architectural invariant
+ τ_retro < τ_defer < τ_store < τ_train. Empirical π_retro on the actual
+ stored pool per cycle is reported in §sec:check-purity; the cal-fold
+ sensitivity analysis (receipt_tau_retro_sensitivity.json) shows the
+ threshold's behaviour on the broader population for context.
```

---

## P5 — Phase 4 baselines + Ch5 tables (auto-runs after Step 7 main)

`run_phase1a.sh` continues automatically after Step 7 main:

- Step 8: FLARE smoke
- Step 9–14: Baselines B1–B7 (zero-shot, RAG, Self-Consistency, calibrate-then-abstain, FLARE, EWC fine-tune, EWC + retention guard)
- Step 15: Paired McNemar + BCa bootstrap with Holm correction
- Step 19: Purity validation
- Step 20.1: Aggregate results CSV
- Step 21: Ch5 table generation (`tab_headline.csv`, `tab_baselines.csv`, `tab_ablations.csv`, `tab_chm_decomp.csv`)

Verify they ran:

```bash
ls outputs/baselines/B*/                       # expect B1-B7 dirs
ls outputs/full_run/sig_tests/                 # mcnemar_holm.json
ls outputs/full_run/aggregate_results.json
ls outputs/full_run/tab_*.csv                  # 4 Ch5 tables
```

---

## P6 — Citation audit

Walk every `\cite{}` against `references.bib` AND against the cited paper's actual content. Generate `thesis_report/audit_phase_a_citations.md` with columns:

| File | Line | `\cite{key}` | bib_present | claim_in_prose | paper_supports_claim | notes |

Long manual task (~several hours). Critical for thesis integrity.

---

## P7 — Final thesis polish

After everything above:
- Splice Phase 4 tables into Ch5 (replace `\input{...}` placeholders)
- Refresh auto-figures: `python -m scripts.make_figures`
- Compile: `cd thesis_report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- Fix unresolved `\ref{}` and undefined cites
- Final structural review per `feedback_thesis_coherence` memory (terminology, registry counts, fold semantics, cross-references)

---

## Open issues (deferred, no urgency)

1. Task #42 — Document 9-subtype CHM coverage methodology in thesis (low priority)
2. Task #98 — LoRA SIL future work (Phase 1b, deferred)
3. Task #111 — B6/B7 + ablation dynamic cycle count rewire (auto-handled by `run_phase1a.sh` if Step 7 early-stops)

---

## Locked configuration snapshot (2026-04-28 15:10 UTC)

| Value | Where | Current |
|------|-------|--------:|
| `α_store` | `cycle_0/conformal_gate.json` | **0.05** (restored from sweep) |
| `α_defer` | `cycle_0/conformal_gate.json` | **0.40** |
| `τ_store` | `cycle_0/conformal_gate.json` | **0.6676** |
| `τ_defer` | `cycle_0/conformal_gate.json` | **0.5207** |
| `τ_retro` | `config.py:664` | **0.50** |
| `τ_train` | `config.py:627` | **0.75** |
| `ρ_min` retention | `config.py:636` | **0.93** |
| `α_ema` | `config.py:274` | **0.7** |
| Boost intercept | `cycle_0/composite_calibration.json` | **+1.099** (restored) |
| Boost C | hard-coded in `run_experiment.py:cmd` | **0.01** |
| `disable_h_norm` | `config.py` | **True** |
| `top_passages` logging | `eval/harness.py` | active from cycle 1+ |
| Safety floor `u_pre^min` | `config.py:127` | **0.60** |
| Tier-1 combined | `config.py:123` | **0.90** |
| Tier-2 similarity | `config.py:124` | **0.75** |
| Routing λ | `config.py:121` | **0.70** |
| M chains, T_a | `config.py:140,143` | **3, 0.7** |
| K MC-Dropout | `verifier.py:24` | **5** |

All values match thesis claims. Three independent guards lock the calibration chain across cycles 1–10.

---

## Resume command on return

```bash
cd /workspace/caem
tmux ls                                              # check plan_a alive
tail outputs/full_run/run.log | head -50             # most recent activity
ls outputs/full_run/run_complete.json 2>&1            # main run complete?
cat outputs/full_run/experiment_summary.csv 2>&1 | head -15
```

If everything is green: proceed to P2 (theorem receipts) and onward.
