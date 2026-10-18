---
name: feedback_chapter_review_workflow
description: Final thesis pass workflow — Claude's chapter pass, then Aksan reads the whole chapter and discusses, then final adjustment, then the chapter is finished; full ch1–6 consistency check at the end.
metadata:
  type: feedback
---

Aksan (2026-10-16): after Claude's work on a chapter he reads it fully himself and discusses it with Claude as he goes; they make a final adjustment and only then is the chapter finished. After all chapters are cleared, a full chapters 1–6 consistency check follows. Several chapters may be done in one conversation when context allows (ch1 and ch2 shared one).

**Why:** he wants to own every chapter's final text and catch issues Claude's pass misses; the consistency check catches cross-chapter drift that per-chapter passes cannot.

**How to apply:** don't treat a chapter as done (or ask him to merge) until his read-through discussion is closed; keep its PR open for follow-up commits. Stack the next chapter's branch on the previous review branch (PR base = that branch) and retarget to `branch/ruc-classifier` after the earlier PR merges. Keep a list of cross-chapter items for the final 1–6 check. Approved in the same exchange: ch5 may mark H2 "unsupported as registered". Related: [[caem_drafting_session_start]], [[feedback_thesis_coherence]].

**Update 2026-10-16 (merge authority):** Aksan now authorises Claude to merge the chapter PRs itself via the GitHub API ("you can do merging step yourself. fewer chances to make errors"), after his read-through is closed and he says the chapter is done. Merge in order, always into `branch/ruc-classifier`; check the PR base before merging (PR #6 was once merged into `review/ch1` by mistake and needed PR #7). Prefer branching each chapter fresh from `branch/ruc-classifier` once the previous PR is merged.

**Chapter 5 mode (Aksan, 2026-10-16):** ch5 is also an analysis chapter; work in a HYBRID loop — discuss decisions with Aksan section by section while applying the uncontroversial fixes in between, then his final read and merge. All inherited rules (SESSION_START, memory, ch1–ch4 additions) apply. PR #9 (ch4) merged by Claude at his request.
