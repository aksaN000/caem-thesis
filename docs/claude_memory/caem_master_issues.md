---
name: caem_master_issues
description: "2026-10-17: single consolidated list of every issue/limitation (A system, B design, C baselines, D scoring, E text, F repo, G inherent) in ref_check/MASTER_ISSUES.md; author plan = fix all, then re-run experiment if needed, rewrite text once."
metadata:
  type: project
---
`defense/thesis_report/ref_check/MASTER_ISSUES.md` is THE checklist (supersedes scattered limitation lists; tracker ids cross-referenced). Author (2026-10-17): "im done listing limitations again and again… we will solve them all, and then we will reran the entire thesis if needed." Don't re-list limitations piecemeal; update this file instead. A-items need a new main run (~193 h on the rented 32 GB GPU; a local RTX 3060 would be several times slower); C-items are local baseline re-runs; D-items are re-scoring.
**How to apply:** add new findings to MASTER_ISSUES.md; when planning, group fixes by type (RUN/BASE/SCORE/TEXT/REPO). Related: [[caem_repo_cleanup_tracker]], [[caem_ch5_decisions]].

**2026-10-17 rewrite:** MASTER_ISSUES.md is now PLAN (phases 0–8) + issue list with status + open decisions (repeat share ~20%, A3 keep/drop, S2 threshold, budget $60–100). Verifier time: 69% passage search (CPU flat index, per question) → retrieval cache; workers only in non-memory-writing steps; stream stays sequential bs 32; repeated questions (B3) now IN; batch-1 live latency (B6) IN. WSL config written (12 GB, swap on F:); after the author's restart, verify `free -g` first.

**2026-10-17 ACTIVE PLAN = minimum defensible thesis, NO main re-run** (author: 'avoid the main runs and only do those that cant be ignored'). Must-do: matched-prompt baseline re-runs (B1,B2,B5,B8,B6 local; FLARE short rental ≥64GB RAM), B3/B4 empties, verifier pass on baselines, TQA judge on final answer, TriviaQA re-check, probe bug, accurate descriptions. Author wanted token score/early exit/atomic DROPPED from the thesis; Claude advised against (they had composite weights 0.126/0.334/0.184 and early exit fired 74× on eval) and proposed 3 short accurate sentences instead — ACCEPTED by author 2026-10-17. Optional agreed: local S1+S2 trial (subset, C0+1–2 cycles) as future-work evidence. Full re-run plan kept as appendix.
