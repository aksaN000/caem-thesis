---
name: When user starts thesis writing, surface ALL relevant memory entries first
description: Before any thesis-writing work begins, list every memory entry that informs the section being written. User explicitly requested this so empirical findings + framing decisions don't get lost.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
When the user signals they're starting thesis-writing work (any of: "let's start writing", "back to thesis", "Ch5/Ch6", "write the report", "integrate empirical findings", "draft this section"), surface ALL relevant memory entries BEFORE drafting any prose.

**Why:** Many threads of empirical findings, framing decisions, and methodological commitments are now in memory (TruthfulQA rescore, CSQA routing issue, three-headline decomposition, baseline-format-floor patches, abort-skip behavior, etc.). Some entries SUPERSEDE earlier ones (the TruthfulQA entry was updated 2026-05-16 from "EM only" framing to "hallucination-aware" framing). If I draft without checking memory first, I will:
- Reuse stale claims from training-data priors instead of the user's actual locked decisions
- Miss the user's previous "this is the framing we agreed on" commitments
- Repeat mistakes that were already corrected (e.g., looking at EM in isolation for a hallucination-reduction system)

**How to apply:** When the user opens a thesis-writing session, my first response should:
1. List relevant memory entries by topic (group by chapter/section, with one-line description each)
2. Note which are SUPERSEDED by later entries (e.g., "v1 rewrite methodology — SUPERSEDED 2026-04-24")
3. Note PENDING items (e.g., "B6/B7 verifier rescore pending; CSQA re-routing experiment pending")
4. Ask the user to confirm which framings should govern this writing session
5. THEN start drafting

**What NOT to do:** Do not start drafting Ch5 §5.3 prose with "CAEM significantly outperforms all baselines" or other heuristic-sounding claims without first checking what the memory says about the actual decomposition (B1→C0 architecture contribution + C0→C3 SIL contribution, per-bench wins/losses, TruthfulQA caveat).

**Triggers:** User says any of: "start writing", "back to thesis", "write Ch5/Ch6", "draft this section", "integrate findings", "report writing", or opens any `thesis_report/chapters/*.tex` for editing.
