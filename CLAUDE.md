# CAEM — Claude Project Anchor

This file is the entry point for any Claude session working on this project. Read it first.

## What this project is

**CAEM (Confidence-Aware Episodic Memory with Self-Improvement)** — Aksan Gony Alif's CSE400 thesis at BRAC University. A three-tier hallucination-reduction architecture: episodic memory (Tier 1), zero-shot generation (Tier 2), retrieval-augmented generation (Tier 3), with a calibrated-probability verifier composite gating a per-cycle self-improvement loop. Backbone: Qwen-2.5-3B-Instruct, LoRA r=32 α=64.

## Project state (anchor date: 2026-05-15)

- **Trajectory:** Cycle 4 in Step 5 eval; Cycle 5 attempts next; **stop locked at C5 close** regardless of outcome.
- **Cycle 4 retention guard fired** (first in-flight rollback): TriviaQA-test probe ratio 0.8354 < 0.93 floor, weights rolled back to C3 θ_prev, memory preserved.
- **Empirical state at C3:** Pooled CHM dropped 22% (0.156 → 0.121); per-bench CHM dropped 25-29% on 4 of 5 benches (CSQA flat). T2 share grew 1.5% → 18.1% on training panel. T2 EM (0.620) > base T3 EM (0.476) by 14 pp — clean SIL-as-distillation receipt.
- **Skip-retroverify-on-abort patch shipped** (commit 2347d60, 2026-05-14). Runner restarted with patched code at 11:20 BDT.
- **Thesis:** All 6 chapters written + audited + drift-fixed (Ch1-Ch6, abstract, appendices). Empirical-receipt integration + baselines + ablations + diagnostics + visual polish remain.

## Where to look for things

| What | Where |
|---|---|
| **Full session memory + project context** | `/root/.claude/projects/-workspace/memory/MEMORY.md` (index) + linked memory files |
| **Source-of-truth for any factual claim** | Live `caem/` code (NOT `branch_C.md` or `branch_C_log.md` which can drift) |
| **Dated decision log** | `branch_C_log.md` (most recent entries at top) |
| **Post-trajectory plan** | `PRODUCTION_NEXT_SESSION_PLAN.md` Phase 1.5 + Phase 1.6 |
| **Thesis chapters** | `thesis_report/chapters/chapter_{1..6}.tex` |
| **Empirical results** | `outputs/full_run/` (cycle_N dirs + eval/ + cycle_0/) |
| **Cycle-0 validation gate** | `outputs/cycle_0/weight_validation.json` (Cohen's d ID +0.467, transfer +0.531) |
| **Production deployment design** | `docs/PRODUCTION_RUNBOOK.md` (read end-to-end before any deployment-mode question) |
| **Ablation registry** | `caem/ablation/variants.py` (3 active: no_retroverify, no_self_improvement, no_forgetting_guard) |

## Critical user preferences (override default behavior)

- **Time zone:** Report clock times in BDT (UTC+6), not UTC.
- **Git commits:** Never include the Claude `Co-Authored-By` trailer.
- **Status checks during autonomous runs:** End with action or ScheduleWakeup, NEVER end with a question. User runs autonomous loops because they're unavailable.
- **Explanation style:** Plain-English, step-by-step, concrete numbers. Length-as-needed for clarity, not as short as possible.
- **Thesis writing style:** Top-tier paper voice. Body prose bans EM dashes, the section symbol, and all code/script/variable/file/folder names. Use concept words and clickable cross-references.
- **Thesis edits must increase coherence:** Every edit must preserve terminology, cross-references, registry counts, fold semantics, and forward/backward references across all chapters/bib/runbook/code. Drift is regression.
- **After any compaction:** READ ALL CAEM CODE END-TO-END before answering substantive questions. On-demand reads have failed too often (Tier 2 ↔ RAG, cal-fold ↔ eval-fold, α ↔ p_c confusions).
- **Verify exact sample count + dedup before any GPU run.** Vast credit is real money. Predict wall-time before launch.
- **All decisions must produce on-disk evidence** under `outputs/` with a dated `branch_C_log.md` entry. Archive failed states BEFORE overwriting.

## Conversation resume protocol (if Claude account switched or session lost)

The file system on this Vast machine persists across Claude sessions. A new Claude session can recover full project context by:

1. Reading this CLAUDE.md
2. Reading `/root/.claude/projects/-workspace/memory/MEMORY.md` and following the per-topic links
3. Reading the top of `branch_C_log.md` (most recent entries)
4. Reading `PRODUCTION_NEXT_SESSION_PLAN.md` Phase 1.5 + Phase 1.6 for current state
5. `git log -20 --oneline` for recent commits
6. `tmux ls` + `tmux capture-pane -t plan_a -p | tail -50` to see runner state if alive

**The memory files at `/root/.claude/projects/-workspace/memory/` are file-based and persist across Claude account switches as long as the new account has read access to the same directory.** They are NOT tied to a specific Claude account session.
