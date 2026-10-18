---
name: CAEM Phase 1 autonomous execution authorization
description: User authorized Claude to autonomously run the CAEM Plan A experiment runbook (Steps 5-20 minus 16-18) on Vast.ai without per-step permission until completion or credit exhaustion.
type: project
originSessionId: 30992e2c-c245-439e-947d-20a565da17c6
---
On 2026-04-18, user authorized autonomous execution of CAEM Phase 1 "Plan A" on a Vast.ai RTX 5090 instance (~$0.638/hr, starting budget $48). Scope:

- **Plan A = Steps 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20** from `NEXT_SESSION_PLAN.md`.
- **Explicitly skipped: Steps 16, 17, 18** (screening + confirmatory ablation sweeps) — out of budget on $45 budget; deferred to Phase 2.
- User may be asleep / unavailable — proceed without waiting for per-step permission.
- Stop condition: Plan A complete OR Vast credit depleted OR a hard-gate step (Step 5 smoke test, Step 8 FLARE smoke) fails.

**Why:** User explicitly said "if im not responding to your requestes then maybe im asleep, you need to keep running until plan A finishes or the credit finishes" and "keep runnning plan a until it finshes or credit finishes even if im not available to give permission to run next stages." Authorization is scoped to Plan A specifically — does NOT extend to expanding scope (e.g., running Steps 16-18) without re-confirming.

**How to apply:** Chain Steps 5→20 in a tmux-managed shell runner. Hard-gate Step 5 (smoke CES>0) and Step 8 (FLARE empty_answer=0). Soft-fail on baselines 9–15. Single retry with `--resume_from_cycle` if Step 7 crashes mid-cycle. Do NOT expand scope to 16–18 without explicit permission even if budget remains.

Repo: `/workspace/caem`. Step 4 passage-index build running in tmux session `build`, log at `/workspace/caem/build_passage_index.log`. Revised ETA (as of 16:05 UTC, after observing ~1,090 passages/s on 21M-passage target): Step 4 completion ~20:50 UTC on 2026-04-18 (not 15:50 UTC — earlier estimate was off by ~5h). Plan A runner in tmux `plan_a` (PID 29562) polls every 120s for `data/passage_index/passages.faiss` + `passages.pkl` and auto-chains Steps 5→20.
