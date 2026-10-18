---
name: caem-book-complete
description: "The CAEM teaching book is fully drafted — 16 chapters + 2 appendices, 151pp, compiles clean"
metadata: 
  node_type: memory
  type: project
  originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---

The CAEM Book (`/workspace/caem/caem book/`, `main.tex`) is fully drafted as of 2026-06-14: 16 chapters across five parts plus Appendix A (Registered-Constant Registry) and Appendix B (Glossary), 151 pages, builds clean with TinyTeX latexmk (0 errors, 0 undefined refs).

**Structure:** Ch1 problem, Ch2 big picture, Ch3 prerequisites, Ch4–12 the eight stages (u_pre / memory / router / RUC / tiers / verifier / composite+gate / output / cycle-loop), Ch13 theory, Ch14 benchmarks-baselines-metrics, Ch15 realised trajectory C0–C5, Ch16 ablations, App A constants, App B glossary.

**Build:** `cd "/workspace/caem/caem book" && PATH=/home/aksan/.TinyTeX/bin/x86_64-linux:$PATH latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Run in background (foreground pdflatex intermittently triggers the harness `H.replace` glitch). After a build, check `main.log` for the `Fatal error`/`Undefined control sequence` strings and that `main.pdf` exists — NOT just the exit code (latexmk can race a stale log and report 0 with no PDF; that bit me once on the `\udropout` typo, the macro is `\udrop`).

**Every figure declares its status** (real-data caption / `(Schematic.)` / `(Illustrative.)`). Ch14–16 figures and tables are all real deployed numbers verified against artefacts. Conventions held throughout: no em dashes, "we"/impersonal voice, A4 textwidth ≈15.4cm (wrap wide tables in `p{}` columns, not `lll`). Per [[feedback_caem_book_code_listings]] the per-stage chapters keep caempy code listings; per [[feedback_thesis_writing_style]] body prose stays concept-word.

Finishing pass (2026-06-14): full coherence audit (18-agent) + fixes across 9 chapters (e.g. Ch4 dropped a self-contradictory sub-floor u_pre 0.376; Ch7 RUC caption: largest positive coef is the negation cue +0.70 not rerank-spread; Ch5 q_a_relevance is one of the nine, h_norm retired; Ch13 purity band 0.72-0.75). Added the main report's full-page eight-stage pipeline diagram (`chapters/fig_full_pipeline.tex`, `fig:full-pipeline`) in Ch2, ported from `thesis_report/figures/system_architecture_figure.tex`. Visual polish: shorter non-uppercase running heads (the default uppercase "CHAPTER N. TITLE" overflowed ~30pt on long titles). Committed on branch `book/caem-teaching-book` (commit a70008a; repo `/workspace/caem`) and PUSHED to origin (github.com/aksaN000/caem-thesis) as its own branch on 2026-06-13: 24 files = .tex sources + main.pdf + a `.gitignore` for build intermediates. The book lives on its OWN branch, deliberately NOT merged into `branch/ruc-classifier` (the final CAEM branch). The thesis coherence-audit fixes (see below) were committed separately to `branch/ruc-classifier` (commit 378a32e) and pushed. Book bib shares the thesis `references.bib` via relative path, so the book must stay inside the `caem/` repo tree to build.

Key honesty calls baked in: the formal ablation sweep (no_retroverify/no_self_improvement/no_forgetting_guard) was NEVER run (`outputs/ablation/` empty) — Ch16 is built from the contrasts that did run (RUC on/off bundled, SIL decomposition, C4 natural experiment) and marks the sweep deferred. SIL CHM reduction C0→C3 is ~15% (0.135→0.115), NOT the stale v1 22%. See [[caem_three_headline_decomposition]], [[caem_per_tier_em_selection_effect]], [[caem_truthfulqa_rescore_findings]].
