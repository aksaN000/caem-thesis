---
name: Thesis rewrite — code is the source of truth; branch_C_log.md is supplementary only
description: During thesis rewrites, the live codebase is authoritative. branch_C_log.md / branch_C.md are continuously-updated design diaries that can drift from the actual implementation; never cite them as the sole source for a factual claim.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**The rule:** in thesis rewrite passes (Ch4 through Ch6), the live code in `caem/` is the book. `branch_C_log.md` and `branch_C.md` are design diaries — they accumulate decisions, reversals, and sometimes stale entries — and they can easily mislead if taken as authoritative.

**Why:** user feedback 2026-04-24 after Ch4 §12 rewrite, shaping the Ch5 rewrite approach: "codes are our book, branch c is for help as code is live and active but branch C log is a log that gets updated everytime and so it can easily mislead you." The log entries have been observed to drift from the live implementation in past passes (e.g., composite weights, τ_prune values, architectural-constant framings that later reversed).

**How to apply:**
1. For any factual claim about current behaviour (thresholds, weights, code paths, signal definitions, cycle logic), verify from the code first (`caem/config.py`, `caem/verification/verifier.py`, `caem/memory/store.py`, `caem/training/self_improvement.py`, `caem/eval/*`, `scripts/make_*.py`, etc.).
2. Use `branch_C_log.md` and `branch_C.md` for design \emph{rationale} ("why was it done this way?"), motivations, and pilot-measurement anchors — not for "what the code does today."
3. If the log and the code disagree, the code wins. Do not transcribe the log's version without flagging the drift.
4. Historical numbers (pilot measurements on Flan-T5-Large, Phase-1a Cycle-0 seeding data, etc.) may be cited from the log, but always framed as dated pilot anchors, never as current main-run spec.
5. For Chapter 5 specifically: every numerical claim must come from either (a) the live code paths that will generate the number at runtime, or (b) a CSV/JSONL artefact that already exists in `outputs/full_run/`. Never copy a number from `branch_C_log.md` into Ch5 prose.

**Supersedes:** the "Pull Branch C context from `branch_C_log.md` (required)" clause in `feedback_thesis_rewrite_direct_v2.md` step 2c. The log is now supplementary, not required.
