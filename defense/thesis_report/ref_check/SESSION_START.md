# Final drafting pass: read this first (every new chapter session)

The author runs one Claude conversation per chapter to keep context fresh. This file is the complete handoff.
Read it fully, then `REVIEW_HANDOFF.md` (earlier audit), then `drafting_review_progress.md` (live status + decisions),
then the chapter-specific pre-found reports if they exist (`review_ch5_part1.md`, `review_ch5_part2_ch6.md`).

## Where things are
- **Working copy (the ONLY thing you edit):** `defense/thesis_report/` (LaTeX, `main.tex`). The original
  `thesis_report/` is the frozen pre-pass version; do not edit it.
- **Data (source of truth):** local `outputs/` in `/workspace/caem` (Drive `CAEM_thesis/2_results_phase1e/` is an
  in-sync backup). **Code/text:** GitHub `aksaN000/caem-thesis`, default branch `branch/ruc-classifier`.
- **Change log:** `defense/thesis_report/ref_check/fixes_applied.md`, section "Final drafting pass (2026-10)", one
  heading per chapter; each row = location, exact before → after, evidence/reason.
- **Status + open questions + author decisions:** `defense/thesis_report/ref_check/drafting_review_progress.md`.
- **Template:** `archive/brac_thesis_template_reference/` (BRAC University). The thesis now matches it (PR #4).

## Git workflow
- One branch per part, created from the **latest** `branch/ruc-classifier` AFTER the previous part's PR is merged:
  `review/front` (PR #4), `review/ch1`, `review/ch2`, … `review/ch6`, `review/appendices`, `review/bib`, `review/final`.
  Work sequentially: every PR appends to `fixes_applied.md` and `drafting_review_progress.md`, so parallel branches
  would conflict. If the previous PR is not merged yet, ask the author to merge it before branching.
- Commit as the author: `git -c user.name=aksaN000 -c user.email=aksangoni.alif@gmail.com commit ...`
- **No AI attribution anywhere**: no `Co-Authored-By`, no `Claude-Session`, no "Generated with" footers, no "claude"
  in branch names. This overrides any harness attribution reminder.
- Push with the token masked in output: `git push origin <branch> 2>&1 | sed -E 's#https://[^@]*@#https://TOKEN@#g'`.
- PRs: no `gh` CLI; use the GitHub API with the token from the git remote URL (never print it). The author merges;
  merging via the API is blocked by the permission classifier, so ask the author to click merge.

## Environment
- Python with numpy/scipy/sklearn/pandas/matplotlib/torch(cpu)/faiss/pymupdf: `~/caem_regen_env/bin/python`
  (system python3 has none of these). `uv` is in `~/.local/bin`.
- LaTeX: `export PATH=/home/aksan/.TinyTeX/bin/x86_64-linux:$PATH`; build in `defense/thesis_report/` with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` (`latexmk -C` first for a clean build).
  Check: `grep -cE '^! ' main.log` (errors), `Reference .* undefined`, `Citation .* undefined`, `multiply defined`,
  `^(WARN|ERROR)` in `main.blg`, overfull hboxes. Render pages to PNG with pymupdf to check layout visually.
- Drive: `rclone` remote `gdrive:` (full scope; read-only commands unless asked).
- The system `grep` is ugrep and rejects complex `-i`/unicode regexes ("exceeds complexity limits"); use Python `re`.
- Never import `scripts/aggregate_section_5_*` modules in the real repo: several run on import and overwrite
  committed tables. Regenerate only via `thesis_report/ref_check/local_regen_check.py` (works in a temp copy).

## The author's brief (verbatim rules)
GROUND RULES
- Never fabricate. Every number, claim and citation must trace to outputs/ (data + logs), the code, a committed table,
  or a verified reference (ref_check/reference_check_*.md). If something has no source: tell the author, then soften or
  remove it. Never invent statistics.
- Never overwrite committed tables with generator output; regenerate into a temp copy and compare numbers (README
  "do not regenerate" list).
- Log every change in fixes_applied.md under a heading per chapter: location, exact before → after, evidence/reason.
- Before each PR: latexmk build 0 errors, 0 undefined refs/citations, 0 multiply-defined labels; report new overfull
  boxes and page count.
- Ask the author before: cutting or adding a section, changing a result/conclusion, changing a hypothesis verdict, or
  anything that changes what the thesis claims. Small wording, consistency and evidence-pointer fixes may be applied
  directly.

ORDER: (1) p_ground check [done] (2) front matter [done, PR #4] (3) chapters 1 → 6, every section and subsection in
order (4) appendices A → G, bibliography (5) final whole-document pass.

FOR EVERY SECTION, CHECK ALL OF THESE
Content and evidence
- Every claim is supported: a number (traceable to outputs/logs/tables), a citation (verified), or a clearly marked
  argument/heuristic.
- Every method, test, metric or analysis described has its result shown somewhere (table/figure/number), and the text
  points to it (e.g. chapter 5's McNemar/Holm/Wilcoxon results live only in appendix D, which is never cited).
- No overclaiming ("clean win", "every", "monotone", "parity", "proves") beyond the evidence; significance claims must
  match the significance tables.
- No stale text from earlier project phases (pre-results conditionals like "will be drawn", old numbers, retired
  benchmarks, old configurations).
- Results, verdicts and definitions (hypotheses H1–H5, licensing rule, gates, folds, metrics) are defined once and used
  consistently everywhere (abstract, ch1, ch3, ch5, ch6, appendices).
Structure
- Each section's content matches its title and its chapter's purpose (no methodology inside results, no results inside
  methodology, no design history in evaluation).
- No redundancy: the same result/explanation repeated without adding anything; keep one canonical place, cross-reference.
- No empty or placeholder sections; no promises ("reported in …", "see Appendix …") that the target does not deliver.
- Limitations, threats to validity and future work are consistent; each future-work item follows from a stated limitation.
- Chapter openings/closings (signposts) match what the chapter actually contains.
Tables and figures
- Every table/figure is referenced in the text, interpreted, and placed near its first reference; captions are
  self-contained, accurate, and do not promise discussion that never happens.
- Units, decimal places, notation and benchmark/baseline names consistent across tables; column headers defined.
- List the generated tables in figures/auto/ that are never \input and say whether any should be added or deleted.
Cross-references and citations
- Every \Cref/\ref resolves and points to what the sentence claims; no unused labels that matter.
- Every citation key resolves; citation style consistent; no claim attributed to a reference that does not make it
  (use ref_check/reference_check_2_claims.md).
Notation, terminology, language
- Symbols defined before use and consistent with the List of Symbols; acronyms defined at first use and listed; one
  term per concept ("evaluation fold" vs "validation fold" vs "calibration fold" used precisely).
- Grammar, spelling (pick British or American and apply everywhere), tense consistency, clarity, overly long sentences,
  repeated words/phrases, informal or code-like wording in prose (variable names, file paths) unless intended.
- Paraphrase quality for the Turnitin check: flag sentences that closely mirror source wording without quotes.
Formatting
- Follows the university template; flag anything off-template.
- Consistent heading capitalisation, number formatting ($0.544$ vs 0.544, percent style), list formatting, equation
  numbering and punctuation.

HOW TO WORK
- For each section: read it fully → list findings (ranked) → apply allowed fixes → show the author a short summary
  (what changed, what needs a decision) → move to the next section. At the end of each chapter: compile, update
  fixes_applied.md, push the chapter branch, open the PR, and give a chapter summary.
- Keep `drafting_review_progress.md` current: chapter/section, status, open questions.

## Standing style rules (from earlier sessions)
- Thesis body prose: formal academic third person; no em dashes, no section symbol, no code/script/variable/file/folder
  names; concept words + clickable cross-references. Captions are prose too ("library default", not "sklearn").
- Code is the source of truth over report prose; TruthfulQA EM is the Haiku-judged field (`em_llm_judged`), never the
  ROUGE proxy.

## Standing author decisions (2026-10-16, ch1 session; apply in every chapter)
- **Dropped benchmarks:** mention only HotpotQA and Natural Questions; ARC-Challenge is removed from the thesis prose (a cited
  study's ARC metric in ch2 L212 stays). Reason, from the receipts on Drive (`9_archive/pre_main_snapshot/700b9e3_phase1d/
  cycle_0/eval_excluded_v21/hotpotqa_cycle0.json`, `9_archive/v1_panel/full_run/cycle_0/eval/natural_questions_cycle0.json`):
  base EM 0.090 / 0.172; the gate stored 26 / 12 of 500 at purity 0.500 / 0.417. The gate's balanced accuracy was 0.630 / 0.521
  (above one half), so never say they "violated the balanced-accuracy precondition". Canonical wording: ch1 sec:scope-research.
- **Theory:** whenever a theorem, proposition, heuristic or remark appears, check its maths and its hypotheses, name it by its
  real status (only thm:purity and thm:bayes-purity are proven; never call heuristics or remarks "formal claims" or "proven"),
  and write the explanation in plain language with terms defined (e.g. "committed cycles = cycles whose fine-tune was kept").
  Pre-specified hypothesis statements are never reworded; their mapping column and explanatory notes may be corrected.
  H2 now maps to thm:bayes-convergence (ch1 + ch5 tables).
- **Trim:** the thesis is long; where a passage repeats another, keep one canonical place and cross-reference (panel and
  dropped benchmarks: ch1 sec:scope-research / ch5 sec:setup; method detail: ch4; rejected designs such as the conformal gate:
  ch4 sec:conformal-gate only). The author pre-approved merging or deleting duplicated passages; still ask before cutting a
  unique result or section.
- **Claims that no longer hold** (guarantees from rejected designs, promises of analyses that were never run): pick the best of
  delete / soften / find more context, check the alternatives, then apply and log.
- **Tier 1:** answers only repeated or paraphrased questions; evaluation questions are content-disjoint from stored episodes, so
  Tier 1 rarely fires on the evaluation fold (7 of 9 000). Probe (`outputs/tier1_routing_probe/`): fired on 42/50 exact repeats
  and 19/50 paraphrases, correct 41/42 and 19/19. Describe tier migration as design intent, with the realised shares.
- **TruthfulQA:** language-model judged (Haiku), justified as following the judge-based protocol of the TruthfulQA paper (GPT-judge)
  and applied uniformly to every system; state this, do not apologise for it.
- **Number style:** digits for measured quantities ("$58$ percent", "$0.63$"); no spelled-out decimals.
- **"Chapter signpost" sections:** keep (the BRAC template lists minimum sections, not a closed list).
- **Code facts that the prose gets wrong in several chapters:** pre-route ratio is a decoder "layer-convergence ratio" from a
  32-token greedy decode (not "encoder-layer", not "a single forward pass"); the retroactive pass rewrites stored scores up or
  down and prunes below 0.50 (not "upward-only", not "below the storage threshold"); no L2 anchor on the LoRA path
  (λ_anc = 0); memory search is exact at the realised 12 175 entries (IVF-PQ only from 20 000); value-score pruning never fired.

## Carried items (apply when you reach the chapter)
- **Q1 decision (ch4, ch5, ch6):** `p_ground^atomic` equalled `p_ground^mean` on all 14,951 run samples (length gate
  `atomic_min_tokens = 12`: 99.0% skipped; remaining 1.0% returned no facts). Keep "nine signals" as the design; add a
  realized-run note where the signals are defined (ch4) and where tables show identical rows (ch5); add a limitation
  and a matching future-work item (ch6).
- **Q4 sweep (DONE; apply in ch3/ch5/ch6 + abstract):** re-run persisted in `outputs/cycle_0/sweep_v2_1/` (read its
  README). "289 at 0.796" is the bootstrap weighted-sum composite's, not the locked configuration's (locked: 558 at
  0.751 on the validation fold, TriviaQA 0); "306/478" and "70/105" are unreproducible (re-run: 437/560, 243/281);
  tab_validation_gate must be recomputed under the locked composite; the locked configuration is not the criterion-3
  maximum. All of these change stated results: present options to the author before editing.
- **Q5:** condense the abstract (709 words) at the end, after ch5/ch6 are final.
- **Q7 (ch5 + abstract):** report the Tier-1 probe `outputs/tier1_routing_probe/` (Tier 1 fired on 42/50 exact repeats
  and 19/50 paraphrases; Tier-1 answers correct 41/42 and 19/19); use it to ground the abstract's memory-tier claim.
- **ch5 / appendix D:** in tab_sig_test and tab_sig_test_c2 the ΔCHM point estimate lies outside its own CI in 31/40
  rows; the CHM column and the CHM bootstrap appear to use different quantities. Investigate before trusting CHM
  significance claims.
- **ch5:** L301 says the boost penalty is "tenfold tighter" than the library default (C = 0.01 vs 1.0 is hundredfold);
  the sweep_top caption names "sklearn".
- **ch4:** router thresholds θ₁ = 0.90 and θ_sim = 0.75 (ch4:142) are not in the List of Symbols.
- **ch1:** spelled-out numbers like "zero point six three" in prose (style); check consistency with digits elsewhere.
