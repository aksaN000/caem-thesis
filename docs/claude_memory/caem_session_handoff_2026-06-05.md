---
name: session-handoff-2026-06-05-ch4-theorem-restructure-running-ch5-rewrite-plan-locked
description: "Active state for the post-compaction Claude. Theorem-review workflow done, Ch4 restructure workflow IN FLIGHT, Ch5 stripped-rewrite Mode A queued. All analytical artifacts saved to disk."
metadata: 
  node_type: memory
  type: project
  originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---

**Logged 2026-06-05. Context was at 98%, user requested handoff.**

## Where we are RIGHT NOW

User is on local WSL (RTX 3060 12GB, 7.4 GB usable RAM), Miniconda at `~/miniconda3/` with numpy+scipy+torch-CPU installed. Vast instance destroyed. All Phase 1e data + Phase 1d eval data + analysis artefacts are local at `/workspace/caem/`.

**Active workflow**: theorem restructure (run ID `wf_ddcf45f3-f37`, script at `/home/aksan/.claude/projects/-workspace-caem-thesis-report/ae1f7815-b50e-4024-a362-c8bbbdf36803/workflows/scripts/caem-theorem-restructure-wf_ddcf45f3-f37.js`). Status when context-saved: launched, not yet completed. 4 phases: Draft restructured §11 → Discover edits in 5 dependent files → Apply (with backup) → Verify (compile + coherence). On completion, check task output file (path provided in the workflow-launch tool result earlier in the transcript) for results.

## Critical decisions locked this session

### 1. Selective theorem restructuring (chapter 4 §11)

After workflow-driven review (25 agents, 1.4M tokens, in `outputs/analysis/theorem_review.json` + `theorem_review_synthesis.md`):

**KEEP** as theorems/corollaries (empirically verified at C5):
- T1 `thm:purity` (PASS, 8-11pp margin)
- T3 `thm:monotone` (PARTIAL, honest sign disclosure)
- C1 `cor:tier1-floor` (CHM_1=0 at all 7 T1 firings)
- C4 `cor:self-correction` (prune rate 13→4% across cycles)
- C5 `cor:stochastic-equilibrium` (clean PASS — π_commit=4/5, memory grows on C4 abort)

**REFRAME** with explicit "effective limit C5 < theoretical asymptote" language:
- T5 `thm:convergence` (geometric envelope → bounded-band stationarity, ≥15-cycle test deferred to Ch6)
- C2 `cor:convergence-rate` (projected only, three named non-identifiability reasons)

**DEMOTE** from theorem/corollary to "Mathematical scaffolding" remarks in new §11.6:
- T2 `thm:bayes-purity` (Bayes-rule identity)
- T4 `thm:monotone-purification` (derivative of T2)
- T6 `thm:bayes-convergence` (textbook monotone-map convergence)
- T7 `thm:asymptotic-elim` (law of total probability)
- C3 `cor:corpus-floor` (information-theoretic minimum, receipt unobservable)

**Label policy**: ALL existing `\label{thm:*}` and `\label{cor:*}` preserved so cross-references don't break. For demoted items, only the environment changes (`\begin{theorem}` → `\begin{remark}`). The `\Cref` autoref then renders "Remark X.Y" instead of "Theorem X" — surrounding prose in dependent files patched by the discover-phase agents.

### 2. C5 IS the empirical asymptote (key framing shift)

The original review treated "asymptotic" as "t → ∞", which six cycles can't witness. The clarified framing: **C5 IS where the trajectory empirically terminated via the equilibrium-saturation gate**, so theorems are evaluated AT C5. The gap between empirical C5 limit and theoretical t→∞ asymptote is honestly explained by implementation regularisers (LoRA L2 anchor, multi-modal retention guard, parametric capacity ceiling). This framing is folded into the §11 intro paragraph.

### 3. Chapter 5 rewrite mode: Mode A (interactive, section-by-section)

NOT a full delete-and-rewrite (chapter structure stays). Strategy per section:
1. Inventory current numerical claims + tables cited
2. Verify each number against Phase 1e data
3. Build ONE clean table per claim (no NaN, no stale benchmarks, no placeholder dashes)
4. Rewrite prose around verified table
5. Adversarial review for stale claims

**Suggested start**: §5.1 Experimental setup (lightest table dependency).

**Caveat**: §5.6 Theoretical predictions Checked must wait for theorem-restructure workflow to land first.

## Phase 1e empirical headlines (for thesis writing)

### Pooled (5-bench average)
- B1 zero_shot: EM 0.507, CHM 0.120
- CAEM C3: EM 0.532, CHM 0.115
- **CAEM C5: EM 0.517, CHM 0.109** (CHM minimum across cycles)
- vs best baseline: **+1.0pp EM, -9.4% CHM**

### Three-headline decomposition
- B1→C0 (architecture only, no SIL): +3.4pp EM, **+12.7% CHM (worse)** — architecture without SIL hurts CHM
- C0→C3 (SIL only): -0.9pp EM, **-15.3% CHM** — SIL does the work
- B1→C3 (full): **+2.5pp EM, -4.5% CHM**
- B1→C5 (final): **+1.1pp EM, -9.4% CHM**

### Phase 1d → Phase 1e (RUC bundled at C3)
Pooled: **+5.80pp EM, -5.2% CHM**. T3-share drops 21-59pp across all benches. Biggest gains on RAG-hurt benches: CSQA +14pp EM (T3-share -37pp), TruthfulQA +10pp EM (T3-share -59pp). Asymmetric-rescue receipt CONFIRMED.

### Significance tests (CAEM C5 vs 8 baselines × 5 benches = 40 pairs)
Holm + |ΔEM|≥2pp: **17 sig wins for CAEM, 3 sig losses (all on TruthfulQA — abstention effect)**.

### Per-tier (production trajectory)
- T1 firing rate: 0.13% (2/1500 at C5)
- T2 EM > T3 EM throughout (SIL-as-distillation receipt)

### T1 routing probe (targeted diagnostic, n=100)
- Exact-match queries: 84% T1 hit, 38% paraphrase
- **97.6% cache-hit EM accuracy when T1 fires**
- Production rarity (0.13%) is by-design: eval is content-hash disjoint from training

## Table quality issues (must address during Ch5 rewrite)

- **Placeholder stubs**: `tab_calibration.tex` ("Table pending main run")
- **Dash-filled columns**: `tab_continual.tex` (bwt/fwt all `--`)
- **Stale-benchmark references**: `atomic_scope.tex`, `cohen_d.tex`, `cross_benchmark_summary.tex`, `prompt_compliance.tex` still reference HotpotQA / Natural Questions / ARC-Challenge / ASQA (all retired in v2.1 panel cleanup)
- **Buggy aggregator**: `scripts/aggregate_stochastic_equilibrium.py` commit-attribution loop at lines 107-125 reports π_commit=1.000 (4/4) when true value is 4/5=0.80 (C5's commit attribution gets stolen by C4 abort)

**Three known data-gap blockers for Ch5 receipts**:
1. `outputs/purity_validation/` does not exist → blocks T2/T4 residual receipts (need offline reconstruction from `outputs/full_run/cycle_N/calibration/<bench>_cycleN.json` per-sample flags vs cal-fold gold)
2. `top_passages = [None, None, None]` in all 6 cycles (rectangular-records filter in `build_calibration_pairs.py:44-51`) → blocks C3 corpus-floor receipt
3. `per_sample_signals.jsonl` is 0 bytes → blocks signal-fingerprint table

## Local environment state

- Miniconda: `~/miniconda3/`
- Critical PATH line: `export PATH="$HOME/miniconda3/bin:$PATH"`
- numpy 2.4.6, scipy 1.17.1, torch 2.12.0+cpu installed
- Python 3.13.13 via conda
- WSL 7.4 GB usable RAM (Windows takes the rest from 16 GB host)
- 3060 12GB VRAM available but unused for analytics

## File paths to know

| Purpose | Path |
|---|---|
| Main analysis report | `/workspace/caem/outputs/analysis/phase1e_analysis_report.md` |
| Phase 1d eval data (downloaded) | `/workspace/caem/outputs/phase1d_eval/` |
| Phase 1d trajectory data | `/workspace/caem/outputs/phase1d_full_run/` |
| Theorem review verdicts | `/workspace/caem/outputs/analysis/theorem_review.json` |
| Theorem review synthesis (27 KB) | `/workspace/caem/outputs/analysis/theorem_review_synthesis.md` |
| RUC bundled contrast | `/workspace/caem/outputs/analysis/phase1d_vs_phase1e_ruc.json` |
| Verifier stage costs | `/workspace/caem/outputs/analysis/verifier_stage_costs.json` |
| Cycle wall-time | `/workspace/caem/outputs/analysis/cycle_walltime_trajectory.csv` |
| Auto tables | `/workspace/caem/thesis_report/figures/auto/tab_*.tex` |
| Ch4 source | `/workspace/caem/thesis_report/chapters/chapter_4.tex` |
| Ch5 source | `/workspace/caem/thesis_report/chapters/chapter_5.tex` |
| Theorem section (Ch4 §11) | chapter_4.tex lines 440-643 (will change after restructure) |

## Next steps after compaction

1. **Verify Ch4 restructure workflow completed cleanly** — workflow run ID `wf_ddcf45f3-f37`. Look for `outputs/analysis/ch4_restructure_backup/` directory (backup created if applied). If workflow incomplete, resume via `Workflow({scriptPath: ".../caem-theorem-restructure-wf_ddcf45f3-f37.js", resumeFromRunId: "wf_ddcf45f3-f37"})`.
2. **If §11 restructure landed**: check that all 12 labels still present, verify with a quick grep of dependent files.
3. **Begin Ch5 §5.1** (Experimental setup) — first section, lightest table load.
4. **Mode A interactive** — user reviews each section, no batch workflows for Ch5 sections.
5. **Save §5.6** for last (depends on Ch4 §11 landing).

## Cross-references to other memory entries

- `caem_theorem_scope_limit.md` — pre-registered the pool-vs-model scope distinction (now operationalised)
- `caem_t1_routing_activation_finding.md` — supports C1 corollary receipt
- `caem_per_tier_em_selection_effect.md` — explains the T2>T3 EM gap
- `caem_phase1d_cycle1_adapter_lost.md` — Phase 1d archive contamination (record only)
- `caem_cycle4_retention_abort.md` — C4 abort + asymmetric rollback (supports C5 corollary)
- `caem_truthfulqa_rescore_findings.md` — Net Honesty framing for TruthfulQA (use in Ch5 §5.4)

## User preferences locked in this session

- BDT (UTC+6) time zone in clock references
- No EM dashes in body prose
- No code/file/variable identifiers in body prose (use concept words + clickable refs)
- Direct per-subsection rewrite workflow (Mode A interactive)
- Adversarial verification preferred (workflow Phase 2 pattern)
- "C5 is the empirical limit" framing — the trajectory reaches an effective asymptote at C5 via equilibrium-saturation gate, and theorems are evaluated against this empirical limit, not the textbook t→∞ limit
