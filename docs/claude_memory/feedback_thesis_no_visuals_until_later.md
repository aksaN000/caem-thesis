---
name: Thesis rewrite — prose only; no figures / tables / diagrams until user says otherwise
description: During the per-subsection Ch4 (and by extension Ch1-6) rewrites, touch ONLY paragraph bodies. Do not edit figures (TikZ / includegraphics), tables (tabular / tcolorbox), or diagrams (algorithm2e / algorithmic bodies beyond what a paragraph naturally references). User wants to update visuals together in a separate pass later.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**The rule:** in the thesis rewrite workflow v2 (direct full rewrite per subsection), edit only paragraph prose. Leave every figure, table, diagram, tcolorbox example, and algorithm body completely alone — even if the visual contains stale content (old composite weights, stale model names, removed veto branches, missing signals).

**Why:** user feedback 2026-04-24 after §2 System Overview rewrite, where I modified the 5 TikZ node contents (S1, T2, T3, S5, S6) and the notation table (Table 4.1). User: "do not touch any figures, tables, diagrams now, we will update those together later, we are just doing paragraphs now."

**How to apply:**
1. When rewriting a section's paragraphs, identify every figure/table/algorithm embedded in that section.
2. The `\label{}`, `\caption{}`, and `\ref{...}` targets stay identical (preserve cross-references) — those are not visual content.
3. The BODY of the visual (TikZ nodes, tabular rows, algorithmic steps inside an algorithm environment, tcolorbox content) is OFF-LIMITS.
4. If paragraph prose is accurate but visual is stale, write a clean paragraph that doesn't lean on the stale visual. Note the drift in the Phase-3 coherence task for later joint-with-user update of visuals.
5. Exception: purely text-only LaTeX elements inside a section (equations, `\paragraph{}` heads, `\emph{}` / `\textsc{}` prose markup) remain in scope — only the boxed visual content is out.

**What I already mis-touched on §2 and need to revert (2026-04-24):**
- Figure 4.1 TikZ: S1 node, T2 node, T3 node, S5 node, S6 node — revert to original box text.
- Table 4.1 (notation): 3 row edits — revert to original rows including the $\tau_{\text{veto}}$ row.

**What stays (still fair game during prose rewrites):**
- Paragraph text before / between / after visuals.
- `\paragraph{...}` headings.
- Equations with `\label{...}` (inline math, not table contents).
- Stage-by-stage narrative paragraphs.
- tcolorbox Worked Example bodies — wait, the user said "figures, tables, diagrams" which includes the Worked Example tcolorbox. I also mis-touched the Worked Example body (Query B composite calculation). That also needs to be reverted.

**Clarified scope for exceptions:** the Worked Example tcolorbox counts as a figure-equivalent visual (it's a boxed diagram in the chapter flow). Revert its body changes too.
