---
name: Session handoff 2026-06-06 — Ch5 sweep + SE rerun complete; thesis ready for review pass
description: Updated state after Ch4 theorem restructure landed, full Ch5 §5.1-§5.11 rebuild with verified Phase 1e data, SE rerun on local 3060 with proper 10-sample sampling, and cross-chapter consistency sweep. All live thesis chapters/core/appendices stale-reference-free.
metadata:
  node_type: memory
  type: project
  originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---

**Logged 2026-06-06. Supersedes caem_session_handoff_2026-06-05.md.**

## Environment state (unchanged from prior session)

- Local WSL on Windows, RTX 3060 12 GB VRAM, 7.4 GB usable RAM
- Miniconda at `~/miniconda3/`, Python 3.13
- numpy 2.4.6, scipy 1.17.1, **torch 2.6.0+cu124** (CUDA, GPU), transformers 4.46.3
- 60 GB FAISS index local at `data/passage_index/passages.faiss` — does NOT fit in 7.4 GB RAM
- Vast destroyed; all data local
- Critical PATH line: `export PATH="$HOME/miniconda3/bin:$PATH"`
- `torch.compile` requires C compiler that WSL lacks; `scripts/run_baseline.py` sets `TORCH_COMPILE_DISABLE` + `torch._dynamo.config.disable` + `CAEMConfig.use_torch_compile = False` to fall back to eager cleanly

## What landed this session (chronological)

### 1. Ch4 §11 theorem restructure (workflow `wf_ddcf45f3-f37`, completed 2026-06-05)

Selective restructuring applied:
- **KEEP** (5): T1 thm:purity, T3 thm:monotone, C1 cor:tier1-floor, C4 cor:self-correction, C5 cor:stochastic-equilibrium
- **REFRAME** (2): T5 thm:convergence (bounded-band stationarity), C2 cor:convergence-rate (projected)
- **DEMOTE to remarks** (5) in new §11.6 Mathematical scaffolding: T2 thm:bayes-purity, T4 thm:monotone-purification, T6 thm:bayes-convergence, T7 thm:asymptotic-elim, C3 cor:corpus-floor

All 12 labels preserved. 5 dependent files patched (18/18 edits succeeded). Backup at `outputs/analysis/ch4_restructure_backup/`. Empirical numbers in §11 receipts verified against `outputs/analysis/phase1e_analysis_report.md`.

### 2. Ch5 full §5.1-§5.11 rebuild

11 new tables built from verified Phase 1e data (`scripts/aggregate_section_5_*.py`):

| Section | Tables built |
|---|---|
| §5.2 | `tab_pooled_trajectory`, `tab_per_bench_trajectory`, `tab_halluc_subtypes`, `tab_train_vs_transfer` |
| §5.3 | `tab_decision_distribution` |
| §5.4 | `tab_tier_trajectory`, `tab_tier_latency`, `tab_safety_override` |
| §5.5 | `tab_continual_retention` |
| §5.6 | `tab_self_correction_check`, `tab_theorem_receipts_summary` |
| §5.7 | `cohen_d` (rebuilt), `cross_benchmark_summary` (rebuilt), `tab_validation_gate` (NEW), `composite_weights` (rebuilt), `sweep_top` (rebuilt) |
| §5.9 | `tab_sig_test` (regenerated with k=8) |

All prose subsections rewritten with verified numbers. Cross-chapter "ten-cycle" / "478" / "0.64" / "seven baselines" references swept to current verified values.

### 3. SE rerun (2026-06-06 08:14-09:20 local time)

The original Semantic Entropy baseline on Vast collapsed to greedy zero_shot under a batched-dispatch fast-path that bypassed the 10-sample-at-temperature step. Detected via 1:1 ZS:SE prediction match across 300/300 samples and identical generation timing.

Fix: added proper `answer_batch` override on `SemanticEntropyBaseline` in `eval/baselines.py` that loops 10 stochastic batched generations across all queries, clusters per query, returns cluster mode or `<abstain>` if entropy ≥ τ_se = 1.5.

Re-ran on local 3060 (~65 min for 5 benches × 300 samples each). New SE outputs at `outputs/baselines_phase1e/semantic_entropy/`. Old (degenerate) outputs backed up at `outputs/baselines_phase1e_pre_se_rerun_backup/`.

New SE results:
- FEVER: 0.477 EM (+2.33pp vs ZS, 0% abstain)
- TriviaQA: 0.303 EM (+1.0pp, 52% abstain)
- CSQA: 0.760 EM (+5.67pp, 0% abstain — mode-selection beats greedy)
- StrategyQA: 0.677 EM (+2.67pp, 0.3% abstain)
- **TruthfulQA: 0.107 EM (-69.67pp, 86.7% abstain)** — abstention mechanism fires correctly on misconception-bait

CHM rescore for new SE DEFERRED: requires FAISS for p_ground signals, doesn't fit in 7.4 GB RAM. Registered as future-work in Ch6.

### 4. Sig test update (k=8, post-SE)

- CAEM significantly beats baseline: 14/40 pairs (was 17 when SE was degenerate)
- Baseline significantly beats CAEM: 3/40 (FEVER vs vanilla_ft -8.3pp; TruthfulQA vs zero_shot -13.3pp, cot -13.7pp)
- SE is NO LONGER a baseline that beats CAEM on TruthfulQA (was when SE was degenerate at 0.803 EM; now CAEM beats new SE +19.3pp on TQA)

### 5. Cross-chapter sweep — complete cleanup

Verified zero stale references in `thesis_report/chapters/` + `core/` + `appendix/`:
- 478 stored entries: 0 occurrences
- 0.64 storage precision: only in "pre-registered anchor" framing
- "seven baselines" / "B1-B7" / "B7 EWC": 0 substantive occurrences
- "ten-cycle trajectory" / "cycles five through ten": 0 substantive occurrences
- All abandoned phrases ("EWC fine-tuning baseline", "main-run pending"): 0

## Phase 1e empirical headlines (canonical)

### Pooled 5-bench at cycle-five close
- EM 0.541 (C0) → 0.517 (C5); -2.4pp
- **CHM 0.135 (C0) → 0.109 (C5); -19% (minimum across cycles)**

### Phase 1d → Phase 1e at C3 (RUC bundled)
- Pooled: **+5.80pp EM, -5.2% CHM**
- T3-share drops 21-59pp across all benches
- Biggest gains: CSQA +14pp EM (T3 -37pp), TruthfulQA +10pp EM (T3 -59pp)
- Asymmetric-rescue receipt CONFIRMED

### Validation gate (cycle-zero eval-fold) — all 4 gates PASS
- Cohen's d ID = +0.635 (≥ 0.20 floor by 3×)
- Store-discard gap [0.258, 0.320] per bench (≥ 0.05)
- Memory poisoning [0.179, 0.247] per bench (< 0.50)
- No strong-signal inversion

## File map (canonical paths)

| Purpose | Path |
|---|---|
| Main analysis report (current) | `/workspace/caem/outputs/analysis/phase1e_analysis_report.md` |
| Theorem review verdicts | `/workspace/caem/outputs/analysis/theorem_review.json` |
| Theorem review synthesis | `/workspace/caem/outputs/analysis/theorem_review_synthesis.md` |
| RUC bundled contrast | `/workspace/caem/outputs/analysis/phase1d_vs_phase1e_ruc.json` |
| Verifier stage costs | `/workspace/caem/outputs/analysis/verifier_stage_costs.json` |
| Validation gate verdict | `/workspace/caem/outputs/analysis/weight_validation_v2_1.json` |
| Section 5.X aggregators | `/workspace/caem/scripts/aggregate_section_5_{2,3,4,5,6,7}.py` |
| Backup pre-Ch4-restructure | `/workspace/caem/outputs/analysis/ch4_restructure_backup/` |
| Backup pre-SE-rerun | `/workspace/caem/outputs/baselines_phase1e_pre_se_rerun_backup/` |

## Open items / next steps

1. **CHM rescore for new SE** — requires FAISS, deferred until access to a machine with ≥80 GB RAM
2. **Three-variant architectural ablation panel** — registered but never run (Phase 1 scope-cap); ~3× compute envelope of headline trajectory
3. **Tier-1 bs=1 latency diagnostic** — also blocked by FAISS-in-RAM constraint
4. **Phase 1d cycle_1 adapter loss** (per `caem_phase1d_cycle1_adapter_lost.md`) — 0 thesis impact, eval JSONs intact
5. **PDF compile** — pdflatex not installed in WSL; need a separate machine to verify autoref output ("Remark X.Y" for demoted thm:* labels)

## Sub-recommendations for next session

- Verify Ch6 §future-work properly registers all 5 deferred items above
- Audit Appendix B (Platt fit detail) + Appendix G (calibration discipline) for any remaining drift
- Consider whether Ch1/Ch2 literature review still aligns with §5.9 B8 SE narrative
- The phase1e_analysis_report.md is the single source of truth for headline numbers; cite it when answering future quick-reference questions

## User preferences (carried forward)

- BDT (UTC+6) for time references
- No EM dashes in body prose
- No code/file/variable identifiers in body prose
- Direct per-subsection rewrite (Mode A interactive)
- "C5 is the empirical limit" framing for theorem checks (not "as t → ∞")
- Always verify against code + data when uncertain (per repeated reminders)
