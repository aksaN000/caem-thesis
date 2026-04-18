# CAEM Thesis — Writing Suggestions Log

This file captures writing and framing suggestions that surface during educational/discussion sessions.
Each entry records: which chapter it belongs to, the issue, the required fix, and its source session.

The `paper-writing` skill should read this file before drafting any chapter.
The `methodology-implementation` skill should also read this file alongside `hyperparameter-reference.md`.
Add new entries at the bottom of the relevant chapter section.

**Related reference:** See `hyperparameter-reference.md` for the complete three-category hyperparameter table (literature-fixed / design choice / empirically calibrated). Required reading before writing Chapter 4 or any implementation code.

---

## How to use this file

- **During sessions:** Claude adds suggestions here as they emerge from discussion.
- **Before writing:** Check this file and clear the relevant chapter's suggestions into the draft.
- **After applying:** Mark the item `[DONE]` and note which session applied it.

---

## Session 44 Addendum — 2026-04-17 (Training-Time Ablation Framework + L2 Terminology Fix)

Apply these notes across Chapters 3, 4, 5, and 6 in the next writing pass. This addendum supersedes parts of C5-09 and C5-14 (the AB1–AB7 numbering is replaced by named variants; see the per-item status updates below).

### S44-01 — Ablation variant catalogue is now 16 named variants, not 7 ABx rows

| Field | Updated value |
|---|---|
| Registry source | `caem/ablation/variants.py::VARIANT_REGISTRY` (16 entries) |
| Reference anchor | `full` (CES reproduces Chapter 5 Table 5.1) |
| Inference-only | `full`, `no_self_improvement` (2 variants) |
| Training-time (cyclic) | 14 variants; each re-runs the SIL loop under its own config |
| Mechanism groups | verification, calibration, safety_gate, self_improvement, routing, retrieval |

**Required text (Chapter 5 §5.5 setup paragraph):**

> *"Ablation variants are catalogued in* `caem/ablation/variants.py` *as 16 named entries grouped by the mechanism they lesion: verification gate (5), calibration (3), safety gate (1), self-improvement (2), routing + retrieval (4), and a sensitivity-sweep companion (1). The* `full` *variant is the reference anchor — reported CES for* `full` *reproduces Table 5.1 within floating-point noise, which validates the ablation harness. Exactly two variants are inference-only:* `full` *(reuses the main-experiment Cycle-10 weights and memory) and* `no_self_improvement` *(never fine-tunes by construction; Cycle-0 evaluation only). The remaining 14 variants are training-time — each runs its own independent self-improvement loop under the variant's configuration, producing a separate memory store and fine-tuned checkpoint per variant. This avoids the counterfactual confound of evaluating a mechanism-disabled at Cycle N against weights that were shaped by that mechanism in Cycles 1 through N−1."*

### S44-02 — Counterfactual-confound disclosure (why 14 variants are cyclic)

**Required text (Chapter 3 §3.x Methodology constraints OR Chapter 5 §5.5 setup paragraph — both are fine; pick one):**

> *"A naïve ablation protocol evaluates a mechanism-disabled configuration against the full-system Cycle-N weights. When the disabled mechanism is one that shapes what gets stored in episodic memory — for example, the verifier, the storage gate, the contradiction veto, or the Tier-1 shortcut — the Cycle-N weights have already been fine-tuned on a training set curated by that mechanism across Cycles 1 through N−1. Disabling the mechanism at evaluation time only removes its diminishing final-cycle contribution. To avoid this confound, every variant whose configuration changes what gets stored or what the SIL pool contains is re-run as an independent self-improvement loop — separate memory store, separate fine-tuned checkpoint, separate per-cycle evaluation. Only two variants are confound-free at inference:* `full` *(the reference anchor) and* `no_self_improvement` *(which never trains by construction)."*

### S44-03 — Two-phase sweep protocol (screening + confirmatory)

Phase-1 self-funded runs use a cheap screening step to select which lesions get confirmatory attention. Phase-2 funded runs drop the screening step and run all 14 cyclic variants at the full horizon with multiple seeds.

**Required methodology disclosure (Chapter 5 §5.5 setup — mandatory for Phase-1 thesis submission):**

> *"The ablation sweep uses a two-phase protocol. Phase 1 runs a cheap screening step (3 cycles × 1500 self-improvement-loop samples, seed 42) across all 14 cyclic variants and uses the resulting per-variant CES ranking to select the top three most-load-bearing lesions. Phase 1 then runs confirmatory evaluations (10 cycles × 5000 SIL samples, seed 42) on the top three lesions plus the* `full` *reference and* `no_self_improvement` *baseline. Reported CES deltas in Table 5.2 come exclusively from the confirmatory runs; screening outputs do not enter the reported table. Phase 2, executed post-submission when additional compute is available, drops the screening step and re-runs all 14 cyclic variants at the full 10-cycle × 5000-SIL horizon with three independent random seeds {42, 123, 456}, reporting CES as mean ± standard deviation per variant. At time of submission, Phase 2 results are unavailable and all ablation-table rows are single-seed point estimates; this is disclosed in §5.10 Limitations."*

**Required limitation text (Chapter 5 §5.10 OR Chapter 6 §6.2):**

> *"Ablation CES values are reported from a single seed (42) due to compute constraints during thesis-phase experiments. Seed variance for the full ablation panel is deferred to an immediate post-submission follow-up (three-seed re-run protocol already scaffolded in* `scripts/run_cyclic_ablation.py` *via the* `--seed` *flag; aggregation auto-switches to mean ± std once ≥2 seeds are present)."*

### S44-04 — L2 anchoring terminology (NOT EWC)

Multiple places in the thesis plan and earlier drafts refer to "EWC regularisation". The implementation is strict L2 anchoring toward the previous cycle's weights, with no Fisher information matrix. Fix across Chapters 3, 4, 5, 6 as follows.

**Wherever "EWC" appears unqualified, replace with:**

> *"L2 anchoring toward θ_prev (a Fisher-uniform approximation of EWC; see* `caem/training/self_improvement.py` *lines 27–41 for the formal rationale). Full Fisher-weighted EWC is deferred as future work."*

**Equation form to prefer in §4.7:**

> L(θ) = L_task(θ) + (λ/2) ||θ − θ_prev||²

**Do NOT write:** "EWC prevents catastrophic forgetting..." or "Fisher information weights each parameter..." — both are false for the implemented system.

### S44-05 — Status updates to existing C5-entries

| Entry | New status note |
|---|---|
| C5-09 | **S44 UPDATE:** The AB1–AB7 numbering is retired. Use the 16-variant registry in `caem/ablation/variants.py`. The "reasoning-chain supervision" and "retroactive re-verification" claims are still most directly defended by `no_retroverify` (was AB4) — part of the cyclic sweep. |
| C5-14 | **S44 UPDATE:** The training-time vs inference-time framing is now operationalised. Under the counterfactual-confound audit, only `full` and `no_self_improvement` remain inference-only; `no_tier1` was reclassified from inference-only to cyclic in Session 44. |

### S44-06 — Chapter 4 §4.8 Calibration: add a short subsection introducing CES axes

The CES axes (ACC, EPI, RET, CAL, VER) are used throughout Chapters 5 and 6 but are introduced only in passing. Add one paragraph under §4.8 or as a dedicated §4.8.5:

> *"The CAEM Efficacy Score (CES) is the geometric mean of five axes computed per cycle: ACC (mean exact-match across the evaluation suite), EPI (one minus hallucination rate at u_stored ≥ 0.50), RET (MMLU retention ratio against the pristine-model baseline, clamped to 1.0), CAL (one minus twice the expected calibration error, capped at 0.5 before clamping), and VER (verifier balanced accuracy, using a 0.5 placeholder until STORE/DISCARD labels are collected). The geometric mean is used rather than the arithmetic mean so that a failure on any single axis visibly deflates CES — this prevents a variant from compensating for a catastrophic drop on one axis with a modest gain on another. A small ε=0.01 clamp prevents any zero-axis input from collapsing the entire score to zero. The exact implementation is* `eval.metrics.ces_score` *with axis composition defined in* `eval.reporting.build_table_cycle` *and* `caem.ablation.scoring.ces_axes_from_cycle`.*"*

### S44-07 — Cross-chapter consistency check after experiments run

Before submission, grep the .tex files for these substrings and verify they match Session 44's framing:

- `EWC` (bare, not in a citation) → should be `L2 anchoring (Fisher-uniform approximation of EWC)`
- `AB1`, `AB2`, ..., `AB7` → replace with named variants from the registry
- `inference-time ablation` without surrounding caveat → ensure the surrounding paragraph lists the exact two variants that qualify (`full`, `no_self_improvement`)
- `3 seeds` or `multi-seed` in §5.5 body text → ensure the Phase-1 limitation disclosure is present (S44-03) if the submission uses single-seed numbers

### S44-08 — Wave 4 refactor cross-references (Tasks #113, #114)

Two clarifications codified during the Wave 4 audit-remediation pass (docs/caem-package-audit-report.md). Both are implementation-level facts that must surface in Chapter 5's ablation discussion so the prose matches what the code now measures.

**What `no_grounding` actually lesions (Task #114 — MAJOR-VR1):**

Before the #114 fix, `no_grounding` zeroed all grounding-adjacent signals ambiguously — including p_entail (which is chain→answer NLI, not retrieval-grounding). Post-fix, `no_grounding` zeros **only** `p_ground_mean` and `p_ground_atomic` (the two retrieval-grounded entailment signals) and rescales the remaining four signals (NLI / SC / u_internal / SE) so their weights sum to 1.0. The chain-answer entailment signal (`p_entail`) is **preserved** under this ablation because it is not retrieval-based.

**Required footnote in Chapter 5 §5.5 (near the ablation variant list):**

> *"The `no_grounding` variant zeroes the two retrieval-grounded entailment signals (`p_ground_mean`, `p_ground_atomic`) and rescales the remaining four verifier signals (chain-answer NLI, self-consistency, internal calibration, semantic entropy) to sum to 1.0. The chain-answer NLI signal (`p_entail`) is preserved because it does not depend on retrieved passages. A companion `no_chain_answer_entailment` variant that separately lesions `p_entail` is deferred to future work — see `docs/caem-package-audit-report.md` MAJOR-VR1 for the sixteen-variant registry scope decision."*

**Pre-Task #113 `no_tier3_rag` runs are invalidated (Task #113 — MAJOR-RN1):**

Before the #113 fix, the `no_tier3_rag` ablation did not null out `pipeline.rag.passage_store`, so Tier-3 retrieval passages could still leak into verifier grounding signals (`p_ground_*`) even when Tier 3 itself was "disabled". Any CES number reported for `no_tier3_rag` from a pre-#113 run measures something strictly weaker than "no Tier 3"; the correct semantics is now enforced by `passage_store = None` (with backup/restore) during the variant's cyclic sweep.

**Required note for Chapter 5 §5.5 or §5.10 Limitations:**

> *"All `no_tier3_rag` CES values reported in this thesis come from post-#113 runs (`passage_store` is explicitly set to `None` for the duration of the ablation and restored afterwards). Earlier ablation passes using the unpatched runner are discarded."*

---

## Session 32 Addendum — 2026-04-11 (Plan-Compliance Guardrails)

Apply these notes while drafting Chapter 5 methodology/setup to stay aligned with current code.

| # | Issue | Required Fix | Section | Source | Status |
|---|---|---|---|---|---|
| C5-14 | "Full suite" wording can be misread as all datasets being SIL-trained | Write this sentence explicitly in §5.1: "CAEM self-improvement training uses FEVER, TriviaQA, and Natural Questions only; TruthfulQA, StrategyQA, and ARC-Challenge are transfer evaluation datasets." | Ch5 §Experimental Setup | Session 32 | OPEN |
| C5-15 | StrategyQA split fallback can invalidate transfer-only interpretation if enabled | Document strict run condition: "StrategyQA evaluation is executed on the requested test split with no automatic train fallback." If fallback is manually enabled in exploratory runs, mark those runs as non-thesis and exclude from main tables. | Ch5 §Experimental Setup | Session 32 | OPEN |
| C5-16 | Cold-start composition previously mixed non-plan benchmarks | In setup text, state cold-start seeding benchmark mix as FEVER + TriviaQA + Natural Questions (default implementation path). If historical runs used HotpotQA/StrategyQA seeds, label them as pre-compliance pilot runs and do not mix into final Chapter 5 claims. **Session 35:** `seed_cold_start.py` HotpotQA branch now returns [] with a warning instead of downloading data (EXP-28). CLI defaults are FEVER + TriviaQA + NQ. | Ch5 §Experimental Setup | Session 32 | CODE FIXED — writing note still OPEN |

---

## Reference Stack — What to Have Open While Writing

**This file alone is not enough.** It tells you *what* to write and *what to watch for*, but not the technical content itself. You need four documents open simultaneously:

---

### 1. `writing-suggestions.md` (this file) — THE DIRECTIVE
**Purpose:** Tells you what to write, in what order, with what framing, and what errors to avoid.
**Use it for:** Section structure, heading hierarchy, which entries apply to each subsection, visual element specs, algorithm source lines, known plan discrepancies.
**Do not use it for:** Actual technical content, citation keys, equations, or numbers — those come from the other three.

---

### 2. `pre thesis 1 report/chapters/chapter_1.tex` and `chapter_2.tex` — THE VALIDATED SOURCE
**Purpose:** The single authoritative source for citation keys, notation, and visual style. These chapters were submitted and validated — everything in them is confirmed correct and present in `bibliography/references.bib`.
**Use it for:**
- **Citation keys** — always get `\cite{}` keys from here, not from the unified plan. The plan may use different keys for the same paper, reference papers that were cut from `references.bib`, or use outdated keys. Chapter 1 and 2 are the ground truth. Search for the author name in these files to find the exact key.
- **Cross-references** — if Chapter 4 refers to "the hallucination taxonomy introduced in Chapter 2", find the correct `\label{}` name in `chapter_2.tex` and use it exactly
- **Notation consistency** — symbols for confidence values, thresholds, benchmarks, and model names must match what was already defined in Chapter 2
- **TikZ visual style** — Chapter 1's architecture diagram defines the visual language (box styles, colours, arrow styles). Chapter 4's diagrams must match it. Copy the `\tikzstyle` definitions from `chapter_1.tex` line 78.
- **Voice and register** — new chapters should read as a continuation of the same document, not a different author
**Do not use it for:** Numbers or claims — Chapter 1's projected numbers (e.g. "20–30% hallucination reduction") are targets, not confirmed results. New citations needed for Chapters 3–6 that are not already in Chapter 1–2 should be added to `references.bib` directly.

---

### 3. `caem-unified-plan-v3.tex` — TECHNICAL CONTENT ONLY (not citations)
**Purpose:** Contains deep technical explanations, mathematical derivations, worked examples, algorithm pseudocode templates, TikZ figure templates, and theorem statements. Use it to understand *what* to write and *how* to structure it — but strip out or verify all citation keys before using.
**Use it for:**
- Understanding *why* a design decision was made before writing about it
- Adapting algorithm blocks and TikZ diagrams (source line numbers are in the Visual Elements section of this file)
- Worked examples for theorems (clearly label them as pedagogical, not experimental)
- Theorem content and proof structure (especially lines 3508, 4029, 4185)
**Do not use it for:**
- Citation keys — these may differ from the validated keys in the pre-thesis report. Always cross-check: find the cited paper in the plan, then find the same paper's key in Chapter 1 or 2 before writing `\cite{}`
- Concrete numbers in Chapter 5 (must come from output files)
- Field names, training targets, regularisation method — known discrepancies documented per-entry in this file

---

### 4. `hyperparameter-reference.md` — THE NUMBER SOURCE
**Purpose:** The single authoritative table of every hyperparameter value, categorised as literature-fixed / design choice / empirically calibrated. Required before writing any equation or algorithm that mentions a numeric value.
**Use it for:**
- Every number that appears in an equation or algorithm in Chapter 4: check the category and state it explicitly in the text
- Confirming λ=0.01 [DES] (not 0.4), embedding dim=768 (not 384), calibration after Cycle 0 (not Cycle 1), û weights 0.25 initial (not 0.20/0.20/0.20/0.40 — those are projected post-calibration)
- Chapter 5: replace projected values with actual values from `calibrated_config.json` once the experiment runs
**Do not use it for:** Chapter 5 result numbers — those come from `outputs/` files only.

---

### Summary: which file answers which question

| Question while writing | File to check |
|---|---|
| "What should this subsection say and what must I avoid?" | `writing-suggestions.md` |
| "What is the technical explanation / derivation / worked example?" | `caem-unified-plan-v3.tex` |
| "What is the `\cite{}` key for this paper?" | **`chapter_1.tex` or `chapter_2.tex`** — validated keys only. Do NOT use the plan's keys directly. |
| "This paper is cited in the plan but I can't find it in Ch1/Ch2 — what do I do?" | Add the citation to `bibliography/references.bib` manually, then use the new key |
| "What line is the TikZ / algorithm template on?" | `writing-suggestions.md` Visual Elements section |
| "Does my adapted algorithm match the actual implementation?" | `caem/[relevant module].py` (always check code before finalising) |
| "What exact value does this hyperparameter have, and what category?" | `hyperparameter-reference.md` |
| "What notation / symbol did Chapter 2 already define?" | `chapter_2.tex` |
| "What `\label{}` name does Chapter 1's figure use for cross-reference?" | `chapter_1.tex` |
| "What are the actual result numbers for Chapter 5?" | `outputs/` files (never copy from the plan) |

---

## Thesis Template Structure (BracU CSE400 — FINAL)

**Template:** `FINAL YEAR THESIS Template_CSE400_Fall 2024 ONWARDS/`
**Main file:** `main.tex` (do not rename; references all chapters by path)

### Chapter sequence (write in this order)

| Chapter No. | Title | Template file | Status |
|---|---|---|---|
| Ch 1 | Introduction | `chapters/chapter_1.tex` | ✅ SUBMITTED (Phase 1, 349 lines) |
| Ch 2 | Literature Review | `chapters/chapter_2.tex` | ✅ SUBMITTED (Phase 1, 229 lines) |
| Ch 3 | Requirements, Impacts and Constraints | `chapters/chapter_3.tex` | ⚠️ STUB ONLY (9 lines) |
| Ch 4 | Proposed Methodology | `chapters/chapter_5.tex` ⚠️ | 🔄 IN PROGRESS — chapter_4.tex (pre thesis 1 report) passes all 15 content checks (Pass 4 + Pass 5 complete); task-aware prompts, DPR corpus paragraph, and em-dash sweep done. Still needs full prose write-up in the submission template file. |
| Ch 5 | Result Analysis | `chapters/chapter_6.tex` ⚠️ | ❌ NOT WRITTEN (needs experiment data) |
| Ch 6 | Conclusion | `chapters/chapter_9.tex` ⚠️ | ❌ NOT WRITTEN |

> **Phase 1 submitted:** Chapters 1 and 2. Phase 2 must deliver Chapters 3, 4, 5, 6.
> **Live thesis report folder:** `pre thesis 1 report/` — all writing goes into this folder's chapter files, NOT into the template folder.
> **Unified plan (`caem-unified-plan-v3.tex`) is a PEDAGOGICAL REFERENCE ONLY** — it explains the methodology with illustrative numbers and detailed walkthroughs to help you write. Do not copy numbers from it directly into the report. Use it to understand what to write; use actual experimental data for the actual numbers.

> ⚠️ The template file numbering does not match chapter numbers (chapter_5.tex is Chapter 4, etc.). This is a BracU template quirk — do not rename the files, just edit them in place.

> `chapter_7.tex` exists in the folder but is NOT referenced in `main.tex`. Do NOT add a 7th chapter — all CAEM content fits in 6 chapters.

### Front matter (already structured in main.tex — fill in)

Declaration · Approval · Ethics Statement · Abstract · Dedication · Acknowledgment · Table of Contents · List of Figures · List of Tables · Nomenclature

### Appendices (replace the two placeholder LaTeX appendices with these)

| Appendix | Content |
|---|---|
| Appendix A | Extended results tables — per-benchmark full ablation matrix, per-cycle routing distributions, per-question analysis samples |
| Appendix B | High-capacity run results — 5M passage index, 1000 cold-start episodes, lab GPU config (run after thesis submission if time permits; reference in §5.3 as planned) |

### No extra chapters needed

All CAEM content maps cleanly to the 6-chapter template:
- Theory (purity theorem, convergence) → Chapter 4 §4.x
- Experimental results, ablations, theory validation → Chapter 5
- No separate "Implementation" chapter needed — implementation detail belongs in Chapter 4 §4.5 (System Implementation subsection)

### ⚠️ Device Upgrade Notice — Portions of the Methodology May Change

**Current implementation** runs on RTX 3060 (12 GB VRAM) with hardware-constrained design choices. `LAB_PC_SCALING_GUIDE.md` documents exact code changes required when a better GPU is available (4090 / 5090 / A100 etc.).

**Changes that will affect thesis sections once the better device is used:**

| Section affected | Current (RTX 3060) | After device upgrade |
|---|---|---|
| §4.7 — L2 penalty computation | Computed on CPU (`_l2_penalty` uses `.detach().cpu()`); scalar transferred to GPU | All tensors on GPU; no CPU offload needed (see LAB_PC_SCALING_GUIDE §7) |
| §3.x — Hardware constraints | VRAM ≤ 12 GB as a hard constraint driving all design decisions | Constraint section may need updating to reflect actual experiment hardware |
| §5.7 — Computational efficiency | Latency numbers from RTX 3060 | Latency numbers must come from the actual device used for the full experiment |
| Appendix B | High-capacity run (5M passages, 1000 cold-start) deferred | Can be run on better device post-submission or as part of the main experiment |

**Rule: do not write final methodology or results numbers until the experiment device is confirmed.** If the full experiment runs on Kaggle (T4/P100) or a university GPU (A100), update §3.x and §5.7 with the actual hardware specs. The RTX 3060 figures from the smoke test are for development only.

**Reference:** `LAB_PC_SCALING_GUIDE.md` §7 for exact L2 penalty GPU migration code (3-step change).

### Chapter contents overview and LaTeX heading hierarchy

> **Template constraint:** The BracU CSE400 `main.tex` has exactly 6 `\chapter{}` commands — fixed, cannot add more. `chapter_7.tex` exists in the template folder but is NOT referenced in `main.tex`. Do not add a 7th chapter.
>
> **Heading levels:** Use `\section{}` for 4–6 major divisions per chapter. Use `\subsection{}` for sub-topics within each section. The planning labels §4.2, §4.3 etc. below are logical groupings — several of them become `\subsection{}` inside a single `\section{}`, not separate `\section{}` commands.

| Chapter | `\section{}` headings (top level) | `\subsection{}` within each section |
|---|---|---|
| **Ch 1** ✅ SUBMITTED | Background / Motivation / Problem Statement / Objectives / Methodology in Brief / Scope and Challenges | Subsections already written |
| **Ch 2** ✅ SUBMITTED | Preliminaries / Review of Existing Research / Summary | Subsections already written (hallucination taxonomy, NLI, SC, SE, memory, continual learning) |
| **Ch 3** | Final Specifications and Requirements / Societal and Ethical Impact / Project Management | Requirements → FR (8 stages) + NFR (latency/VRAM) + HW constraints as subsections; Impact → societal + ethical as subsections |
| **Ch 4** | System Architecture Overview / Confidence Estimation and Adaptive Routing / Verification Pipeline and Episodic Memory / Self-Improvement Loop and Calibration / Theoretical Analysis | See detailed hierarchy below |
| **Ch 5** | Experimental Setup / Main Results / Analysis of Mechanisms / Ablation Study / Efficiency and Stability Analysis / Discussion | See detailed hierarchy below |
| **Ch 6** | Summary of Findings / Limitations / Future Work | Findings → one subsection per theory + empirical claim; Limitations → SBERT routing, NLI ground-truth dependency, inference scope; Future Work → distillation, capacity, reference-free NLI |

---

### Chapter 4 — Detailed `\section{}` / `\subsection{}` Hierarchy

```
\section{System Architecture Overview}
    (no subsections needed — prose + FIG-C4-01 + three-value summary table)

\section{Confidence Estimation and Adaptive Routing}
    \subsection{Pre-Routing Confidence Estimation}   ← u_pre, C_conv, EQN-C4-01
    \subsection{Three-Tier Adaptive Routing}         ← routing logic, OR-condition, FIG-C4-02, ALG-C4-02, EQN-C4-02
    \subsection{Post-Generation Confidence (û)}      ← four signals, FIG-C4-03, EQN-C4-03, blind-spot table

\section{Verification Pipeline and Episodic Memory}
    \subsection{Multi-Signal Verification (Stage 5)} ← NLI+SC+SE, VE1/VE2, FIG-C4-04, ALG-C4-03, EQN-C4-04
    \subsection{Episodic Memory Architecture}        ← schema, immutable/mutable split, FIG-C4-05

\section{Self-Improvement Loop and Calibration}
    \subsection{Iterative Self-Improvement (Stage 8)}← ReST framing, L2 reg, ALG-C4-04, EQN-C4-05, FIG-C4-06
    \subsection{Retroactive Re-verification}         ← freshness mechanism, ALG-C4-05
    \subsection{Confidence Calibration}              ← temperature scaling, EQN-C4-06, EQN-C4-07, IMPL-03

\section{Theoretical Analysis}
    \subsection{Data Purity Theorem}                 ← THM-C4-01 + proof, EQN-C4-08
    \subsection{Coupled Improvement Recurrence}      ← THM-C4-02, monotonicity
    \subsection{Convergence Analysis}                ← THM-C4-03, Banach, diminishing Δ
```

**Writing-suggestions entries by section:**

| Section | Entries to apply |
|---|---|
| §System Architecture Overview | C4-01, C4-03 (three-value table), GEN-01, GEN-11, **BM-02** (inverse scaling → architectural motivation) |
| §Confidence Estimation / u_pre | C4-03b, C4-10 (provenance) |
| §Confidence Estimation / Routing | C4-04, C4-10b, C4-20, IMPL-01 |
| §Confidence Estimation / û | C4-05, C4-06, C4-07, C4-09 (blind-spot), C4-21 (field names) |
| §Confidence / Generation (§4.4) | C4-22b [DONE], **C4-24** (task-aware CoT prompts for all 4 benchmarks; reasoning_chain always has reasoning trace) |
| §Verification / Stage 5 | C4-08 (û_stored≠û), C4-22 (reasoning_chain target), C4-24 (label extraction via harness), IMPL-02, **BM-03** (VE1/VE2 → TruthfulQA failure mode), **BM-04** (SC threshold distinction), GEN-05 |
| §Verification / Memory | C4-15 (immutable/mutable), C4-21 |
| §Self-Improvement / Stage 8 | C4-16 (ReST), C4-19 (L2 — name correctly), GEN-11, **C4-24** (training target = full reasoning trace for all 4 benchmarks) |
| §Self-Improvement / Retroactive | IMPL-01 (freshness guarantee), C5-07 |
| §Calibration | IMPL-03, GEN-13 (device rule) |
| §Theory / Purity | C4-17, PUB-02 (full proof), TH-01, TH-02 |
| §Theory / Recurrence | C4-18, TH-01 |
| §Theory / Convergence | C4-23, TH-01 |

---

### Chapter 5 — Detailed `\section{}` / `\subsection{}` Hierarchy

```
\section{Experimental Setup}
    \subsection{Benchmarks and Evaluation Metrics}   ← BM-01–06, IMPL-04 (StrategyQA split)
    \subsection{Baselines}                           ← 6 baselines described
    \subsection{Calibration Results}                 ← TAB-C5-07, C5-11, GEN-03

\section{Main Results}
    \subsection{Accuracy Across Cycles}              ← FIG-C5-01, TAB-C5-01 (CAEM vs baselines)
    \subsection{Statistical Significance}            ← TAB-C5-06, McNemar's, C5-05, C5-12, GEN-12

\section{Analysis of Mechanisms}
    \subsection{Mechanism Evidence}                  ← TAB-C5-02 (tier fractions, MMLU, û_stored), FIG-C5-02
    \subsection{Theory Validation}                   ← TAB-C5-03 (purity theorem), TAB-C5-04 (convergence Δ), C5-03, TH-03, TH-04

\section{Ablation Study}
    (single section, no subsections needed — TAB-C5-05 with grouped rows)
    ← C5-09, ablation_summary.json

\section{Efficiency and Stability Analysis}
    \subsection{Computational Efficiency}            ← TAB-C5-08, FIG-C5-02 tier growth, C5-06
    \subsection{Stability-Plasticity Analysis}       ← FIG-C5-03 MMLU retention, forgetting scores, C5-07
    \subsection{Memory Utilisation}                  ← TAB (episodes vs capacity), pruning rate, C5-10

\section{Discussion}
    \subsection{Error Analysis and Failure Modes}    ← C5-08 (qualitative failures per benchmark)
    \subsection{Convergence and Practical Ceiling}   ← C4-23 narrative, TAB-C5-04 recap, C5-04 thread
```

**Writing-suggestions entries by section:**

| Section | Entries to apply |
|---|---|
| §Setup / Benchmarks | **BM-01** (selection justification), **BM-05** (FEVER NLI class), **BM-06** (constrained prompt), IMPL-04 (StrategyQA split), C5-01 |
| §Setup / Calibration | C5-11, GEN-03, C4-07 (actual weights) |
| §Main Results / Accuracy | C5-04 (narrative thread), GEN-02, GEN-09 |
| §Main Results / Significance | C5-05, C5-12, GEN-12 |
| §Mechanisms / Evidence | C5-02, C5-13, GEN-06 (FEVER Tier 1 accuracy) |
| §Mechanisms / Theory | C5-03, TH-03, TH-04, C4-17 (replace illustrative) |
| §Ablations | C5-09, PUB-04 (LoRA — Appendix), PUB-05 (EWC — Appendix) |
| §Efficiency | C5-06, GEN-13 (actual device latency) |
| §Stability-Plasticity | C5-07, IMPL-05 |
| §Memory Utilisation | C5-10 |
| §Discussion / Error Analysis | C5-08 |
| §Discussion / Convergence | C4-23, C5-04 closing narrative |

---

## Chapter 1 — Introduction

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C1-01 | CAEM described only as reducing hallucinations, not positioned as architectural intervention vs post-hoc detection | Add explicit sentence in problem statement: "Unlike post-hoc detection methods that identify hallucinations after generation, CAEM is an architectural intervention — it prevents hallucination storage and progressively reduces generation errors through a closed self-improvement loop." | Session 1 | OPEN |
| C1-02 | GPU hours stated as "approximately 200" | Fixed to "30--45 GPU hours" in pre-thesis report. Final thesis: use same corrected figure with breakdown (8--12h inference/cycle × 3 + 20--30 min fine-tuning/cycle). | Fix session | DONE |
| C1-03 | TruthfulQA cited with only truthfulness-alone metric (58%) | Fixed to cite both: "58% on truthfulness alone and 21--25% on the combined truthful-and-informative metric." | Fix session | DONE |
| C1-04 | Accuracy improvement target stated as 60% → 80% | Fixed to 60% → 87%, framed as "20--30% hallucination reduction" primary claim. | Fix session | DONE |

---

## Chapter 2 — Literature Review

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C2-01 | Extrinsic hallucination defined as "not validated AND conflicts with input" | Fixed to "goes beyond what the source provides, regardless of whether it conflicts." | Fix session | DONE |
| C2-02 | CoT 100B threshold stated without instruction-tuned caveat | Fixed: threshold applies to base models only; instruction-tuned models (Flan-T5-Large, 780M) demonstrate CoT benefits after instruction tuning (Chung et al., 2022). | Fix session | DONE |
| C2-03 | Self-consistency u_SC threshold for "high consistency" stated as > 0.85 | Chapter 4 uses > 0.90 for the neutral NLI escalation rule (VE1). These are different contexts (verification confidence vs escalation trigger) so not a bug — but add a note clarifying the distinction when writing Chapter 4. | Session 1 cross-check | OPEN |
| C2-04 | **SUBMITTED TEXT ERROR — §2.1.4 (line 36): "384 dimensional embeddings"** | `chapter_2.tex` §2.1.4 (Sentence Embeddings) states: `"all mpnet base v2, generates 384 dimensional embeddings"`. This is factually wrong — `all-mpnet-base-v2` produces **768-dimensional** vectors. Confirmed by: (1) Bug EXP-10 in impl-log fixed this in the code; (2) `hyperparameter-reference.md` corrected to 768. Fix: change "384 dimensional" → "768-dimensional" in §2.1.4. | EXP-10 / Impl Log | ⚠️ MUST FIX BEFORE SUBMISSION |
| C2-05 | **SUBMITTED TEXT ERROR — §2.2.4 (line 162): "three hundred eighty four dimensional embedding"** | `chapter_2.tex` §2.2.4 (Memory-Augmented Architectures, SBERT paragraph) states: `"Each query is turned into a three hundred eighty four dimensional embedding"`. Same wrong value. Fix: change to "768-dimensional embedding". Same source error as C2-04. | EXP-10 / Impl Log | ⚠️ MUST FIX BEFORE SUBMISSION |

---

## Chapter 3 — Requirements, Impacts and Constraints

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C3-01 | Chapter not yet written | When writing, derive functional requirements directly from the 7 pipeline stages in the plan. Non-functional requirements: latency targets (Tier 1 < 400ms, Tier 2 < 2.5s, Tier 3 < 6s), verification accuracy ≥ 85%, GPU budget ≤ 50 hours, MMLU retention ≥ 93%. | Session advice | OPEN |
| C3-02 | Computational constraints must be stated explicitly and linked to design decisions | State hardware constraints as first-class requirements, not footnotes: (1) VRAM ≤ 12 GB (RTX 3060) — drives batch_size=4, theta_prev CPU offload, 500K passage index; (2) GPU budget ≤ 50 hours total — drives n_questions=5000 (not 10,000+), n_cycles=3 (not 5); (3) Memory capacity K=1000 episodes — drives cold-start seeding strategy. Write: "The system is designed to operate under real-world resource constraints: 12 GB VRAM, a 50 GPU-hour budget, and a bounded memory store. Every architectural decision is constrained by these limits — the thesis demonstrates that meaningful hallucination reduction is achievable within them, not in spite of them." This frames your constraints as a strength (realistic deployment) rather than a limitation. | Session 26 | OPEN |

---

## Chapter 4 — Benchmark Design Rationale
> These entries belong in **Chapter 4** because they explain *why* specific verification rules exist and *why* CAEM needs an architectural approach — not mere benchmark descriptions. Write them when presenting the routing and verification sections.

| # | Issue | Required Fix | Section | Source | Status |
|---|-------|-------------|---------|--------|--------|
| BM-02 | TruthfulQA inverse scaling finding | Must explain WHY larger models score worse (they more faithfully reproduce internet misconceptions). This is the strongest motivation for architectural intervention — scaling alone makes the problem worse. Use in §System Architecture Overview to justify CAEM's design. | Ch4 §System Architecture Overview | Session 2 | OPEN |
| BM-03 | VE1 + VE2 rules must be tied back to TruthfulQA's failure mode in the writing | When presenting VE1 (neutral NLI escalation) and VE2 (misconception flag), explicitly connect them to TruthfulQA's systematic confabulation failure mode. Reader should understand WHY these rules exist, not just what they do. | Ch4 §Verification Pipeline | Session 2 | OPEN |
| BM-04 | SC threshold distinction (Chapter 2 vs VE1) | Chapter 2 states u_SC > 0.85 for "high consistency" pass/fail. VE1 uses u_SC > 0.90 for neutral NLI escalation. Write explicit note: "The general SC verification gate uses threshold 0.85. The VE1 escalation condition uses a stricter 0.90 threshold because systematic confabulation produces near-perfect consistency — the higher bar is needed to distinguish genuine consensus from pathological overconfidence." | Ch4 §Verification Pipeline | Session 2 | OPEN |

---

## Chapter 5 — Benchmark Evaluation Setup
> These entries belong in **Chapter 5 §Experimental Setup** because they describe *how* the benchmarks are evaluated, not why they were chosen. Write them when introducing the experimental setup subsection.

| # | Issue | Required Fix | Section | Source | Status |
|---|-------|-------------|---------|--------|--------|
| BM-01 | Benchmark selection must be justified by failure mode, not dataset popularity | For each benchmark in §Experimental Setup, state explicitly: (1) what hallucination failure mode it tests, (2) why that failure mode matters for CAEM, (3) why the primary verification layer matches the failure mode. The complementary coverage argument is the thesis's strongest justification for using all six benchmarks — training benchmarks (FEVER, TriviaQA, Natural Questions) probe different generation failure modes; transfer benchmarks (TruthfulQA, StrategyQA, ARC-Challenge) test whether improvements generalise beyond the training distribution. | Ch5 §Experimental Setup | Session 2 | OPEN |
| BM-05 | FEVER NOT ENOUGH INFO class needs explicit treatment | When describing FEVER evaluation, add: "For FEVER's NOT ENOUGH INFO class, the correct model behaviour is expressing uncertainty rather than generating a confident label. The verification pipeline treats a high-confidence generation on a NOT ENOUGH INFO ground-truth item as a failure — semantic entropy detects this through a high-entropy output distribution, and the neutral NLI result (VE1) prevents such episodes from entering memory." | Ch5 §Experimental Setup | Session 3 | OPEN |
| BM-06 | FEVER evaluation uses a constrained enumerated prompt — state this explicitly | The FEVER evaluation prompt is constrained to enumerate valid labels: "Answer with one of: supports, refutes, not enough info. Claim: {claim}". Write: "FEVER claims are evaluated using a constrained prompt that enumerates the three valid labels — SUPPORTS, REFUTES, and NOT ENOUGH INFO — ensuring unambiguous extraction without relying on string heuristics." Note: StrategyQA similarly uses a constrained boolean prompt ("Answer yes or no. Question: {q}") for the same reason. | Ch5 §Experimental Setup | Session 20 Fix 6 / Impl Log | OPEN |
| BM-07 | FEVER evaluation uses the `paper_dev` split — state split name precisely in §5.1 | The `lucadiliello/fever` dataset has three splits: `train`, `paper_dev`, `paper_test`. The evaluation split is `paper_dev` (19,998 claims). Do NOT write "dev split" — that split name does not exist in this dataset and would raise a runtime error. Write: "FEVER evaluation uses the `paper_dev` split (19,998 claims), consistent with the standard community evaluation practice." Also note that the integer label mapping for this dataset is: `0 → supports`, `1 → not enough info`, `2 → refutes` — confirmed in the dataset card. This is implemented in `_FEVER_LABEL_MAP` in `eval/benchmarks.py` (fixed EXP-21, Session 35). | Ch5 §Experimental Setup | Session 35 / EXP-21, EXP-22 | OPEN |
| BM-09 | **TruthfulQA metric is ROUGE-L — NOT directly comparable to published judge-based numbers** | CAEM reports TruthfulQA via ROUGE-L (LCS F1 against gold answer strings) as an offline proxy. Published GPT-3.5 / Self-RAG / Llama results for TruthfulQA use a GPT-4 judge (measuring % of outputs that are simultaneously truthful AND informative), which is a fundamentally different measurement. ROUGE-L measures lexical overlap; the judge measures factual correctness. **Do NOT place CAEM's TruthfulQA ROUGE-L score in the same column as a judge-based % truthful number — they are incommensurable.** Required thesis text (§5.1 Experimental Setup): "TruthfulQA evaluation uses ROUGE-L (LCS F1 against the dataset's curated answer strings) as an offline proxy for truthfulness, avoiding dependency on a proprietary judge model. ROUGE-L provides partial credit for lexically overlapping correct responses. Published GPT-3.5 results for TruthfulQA use a GPT-4 judge measuring % outputs rated simultaneously truthful and informative (Lin et al. 2022); these scores are not directly comparable to ROUGE-L and are therefore excluded from Table 5.1. Relative improvement across CAEM cycles is the primary TruthfulQA claim — this is internally consistent regardless of metric choice." Optional (if time permits): also run TruthfulQA in its multiple-choice mode (MC1 accuracy), which IS directly comparable to published MC1 numbers and does not require a judge. Report as a supplementary column. | Ch5 §Experimental Setup | Session 36 analysis | OPEN |
| BM-10 | **⚠️ Baseline scorer (A0–A5) now uses the same per-benchmark logic as the main harness — disclose this in §5.4** | Session 38 fixed a scorer mismatch in `run_ablation.py eval_baseline()`. ARC-Challenge now uses `extract_arc_label()` before exact_match (matching `harness.py:286`). TriviaQA and NQ now use `any_match_em()` + `best_token_f1()` across all aliases (matching `harness.py:292`). Before this fix the `else` branch used only `gold[0]`, which (a) scored ARC near-zero because letter labels were never extracted from verbose model output and (b) ignored 10–40 valid TriviaQA aliases per question. **Thesis impact:** pre-fix, A0–A5 baselines would appear artificially weak on ARC and TriviaQA, inflating CAEM's reported advantage on those benchmarks. Post-fix, comparisons are fair. Required disclosure in §5.4: "All baseline systems (A0–A5) are evaluated using the same per-benchmark scoring logic as the CAEM harness — label extraction for ARC-Challenge and FEVER, alias-aware multi-answer EM/F1 for TriviaQA and Natural Questions, and ROUGE-L proxy for TruthfulQA — ensuring that differences in reported accuracy reflect system capability rather than scorer inconsistency." | Ch5 §5.4 Baseline Comparison | Session 38 audit | OPEN |
| BM-08 | TriviaQA and Natural Questions use multi-alias EM scoring — state this explicitly | TriviaQA questions have 10–40 valid answer strings per question (aliases, abbreviations, alternate spellings). Natural Questions has multiple acceptable answer strings. Scoring is performed via `any_match_em(prediction, gold_answers)` — EM=1.0 if the prediction matches ANY alias after normalisation — and `best_token_f1(prediction, gold_answers)`. Write: "TriviaQA and Natural Questions are evaluated using multi-alias exact match (EM=1.0 if the prediction matches any valid answer alias after normalisation) and best token F1 across aliases. This is the standard evaluation protocol for these benchmarks; checking only the first alias would systematically underreport EM." Also add: "Answer normalisation follows the SQuAD protocol — lowercasing, punctuation removal, and article stripping (a/an/the). This normalisation does not handle numeric-vs-text variants (e.g., 'World War Two' vs 'World War 2'); TriviaQA's alias lists include numeric variants for known cases, mitigating this limitation." This is implemented in `eval/harness.py` `_score()` (fixed EXP-23, Session 35) and `eval/metrics.py` `normalise()`. | Ch5 §Experimental Setup | Session 35 / EXP-23 | OPEN |

---

## Chapter 4 — Proposed Methodology

### §4.1 — Framing and Architecture Overview

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C4-01 | Risk of accidentally describing CAEM as a detection/verification system | Chapter 4 opening must frame CAEM as an architectural intervention. Lead sentence suggestion: "CAEM is not a post-hoc hallucination detector — it is an architectural system that prevents unreliable knowledge from accumulating and uses verified knowledge to progressively improve generation quality." | Session 1 | OPEN |
| C4-03 | Three confidence values must be kept visually and verbally distinct | Introduce u_pre, û, and û_stored in a single summary table early in Chapter 4 before describing any of them in detail. Use consistent notation throughout: never call û_stored "confidence" without the subscript, never refer to u_pre as "confidence score" without the "pre-routing" qualifier. Suggest a callout box: "CAEM uses three distinct confidence values, each computed at a different pipeline stage for a different purpose." | Session 3 | OPEN |
| C4-10 | Hyperparameter provenance must be explicitly stated for each value | Every hyperparameter in Chapter 4 belongs to one of three categories — state which category it falls under: **(1) Fixed from literature** — copy from prior work, never changed (MC Dropout passes T=30, SC samples N=10, dropout rate p=0.1, NLI model choice, memory capacity K=1000); **(2) Design choice** — principled default, tunable if experiments demand it (u_pre weights 0.60/0.40, û_stored weights 0.50/0.30/0.20, routing thresholds 0.95/0.75, OR-condition u_pre < 0.60, Stage 4a gate û ≥ 0.45, l2_lambda=0.01); **(3) Empirically calibrated** — starts at a stated initial value, then measured/fitted during implementation (temperature scalar T fitted by minimising ECE on the 500-sample calibration set; û signal weights starting at equal 0.25/0.25/0.25/0.25 then calibrated toward the projected 0.20/0.20/0.20/0.40). Write for Category 3: "Initial weights are set equal at 0.25 for all four signals. Post-calibration weights are projected to converge to approximately 0.20/0.20/0.20/0.40 based on SE's empirical AUROC advantage (Farquhar et al. 2024); actual calibrated values will be reported in Chapter 5." | Session 4 | OPEN |

### §4.2 — Three-Tier Routing

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C4-04 | OR-condition safety override must be explained as a design principle, not just a rule | When presenting the routing logic, explain WHY the OR-condition exists: similarity alone cannot guarantee correct generalisation — a high-similarity question can still be beyond the model's reliable knowledge. State: "The OR-condition (u_pre < 0.60 forces Tier 3 regardless of similarity) is a safety-first design principle: we prefer the computational cost of Tier 3 over the risk of a confident wrong answer from Tier 2." | Session 3 | OPEN |
| C4-10b | u_pre and routing_score must be presented as two separate mechanisms, not one formula | When describing the routing logic, make explicit that CAEM uses two distinct mechanisms that do not interact mathematically: (1) the OR-condition checks u_pre as a hard veto *before* any formula runs; (2) the routing score formula (0.7·s + 0.3·û_stored) only runs if the OR-condition did not fire. Explain WHY they are separate: combining them into one formula would allow a high û_stored to arithmetically compensate for a dangerously low u_pre — a high-quality memory matched to a query the model doesn't understand would still route to Tier 1. The separation ensures memory quality and model readiness are both mandatory conditions, not tradeable quantities. | Session 5 | OPEN |
| C4-20 | Per-tier computational cost model must be stated as a design trade-off, not just a side note | In §4.2, after describing the routing logic, add a latency table: Tier 1 (<400ms — memory retrieval only), Tier 2 (<2.5s — retrieval + guided generation), Tier 3 (<6s — full RAG + verification pipeline). Write: "The routing architecture is a deliberate computational trade-off: Tier 1 provides answers at near-zero marginal cost (retrieval is ~10ms); Tier 2 balances quality and speed; Tier 3 provides maximum accuracy at maximum compute. The OR-condition safety override ensures that only high-confidence queries benefit from the speed of Tier 1 — a conservative design that accepts compute overhead rather than risk confident hallucination." This also justifies why having three tiers is better than always using Tier 3. | Session 26 | OPEN |
| C4-21 | Confidence field names must match implementation — use correct internal identifiers | When writing Chapter 4 or any equation referencing the PostGenerationConfidence fields, use the **actual implementation field names** confirmed in Session 21: `u_token` (geometric-mean token log-prob), `u_dropout` (MC Dropout uncertainty), `u_consistency` (avg pairwise cosine sim — NOT `u_sc`), `u_entropy` (1 − H_semantic / log₂(K) — NOT `h_entropy_norm`), `u_hat` (combined û). If you write `u_sc` or `h_entropy_norm` anywhere, that is a schema error. For StoredConfidence: fields are `p_entail`, `s_avg`, `h_norm`, `u_stored`. There are NO fields called `passed_verification`, `ve1_triggered`, or `ve2_triggered`. | Session 21 / Impl Log | OPEN |
| IMPL-01 | **Tier 1 skips Stage 5 verification — TWO diagrams must be corrected** | **(1) SUBMITTED — `chapter_1.tex` TikZ diagram (line 158):** The already-submitted Chapter 1 architecture figure contains `\draw[arrow=green!70] (t1) -\| (verify);` — an arrow from Tier 1 directly into Stage 5 (VERIFY box). This arrow must be removed before final submission. Tier 1 reconstructs `StoredConfidence` from the stored entry's quality scores and serves the answer directly — no verification call is made per query. Fix: delete that one `\draw` line. *(2) FUTURE — Chapter 4 TikZ diagram (to be written):* When drawing the Chapter 4 architecture diagram, do NOT include a Tier 1 → Stage 5 arrow. In the thesis text: "Tier 1 serves retrieved answers directly — Stage 5 verification is not re-run per query. Quality assurance is maintained by retroactive re-verification (§4.x), which re-scores all stored episodes with the improved model. This amortises verification cost over one batch per cycle rather than per query — a strictly stronger freshness guarantee." Update Scenario 1 latency to "~150–400 ms (FAISS retrieval only, no generation)." | Session 26 + Session 20 / Impl Log | ✅ Ch1 TikZ FIXED (arrow removed 2026-04-07) — OPEN for Ch4 diagram |

### §4.3–4.4 — Confidence Signals (u_pre and û)

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C4-03b | C_conv adaptation must be explicitly acknowledged | C_conv was developed by Nandakishor (2025) for decoder-only models. CAEM applies it to Flan-T5's *encoder* hidden states to measure query understanding stability before generation. Write: "We adapt C_conv to the encoder context: whereas Nandakishor (2025) applied it to decoder states, CAEM uses encoder layer variance ratios to measure pre-generation query understanding confidence." This is a legitimate adaptation but must be stated, not assumed. | Session 3 correction | OPEN |
| C4-02 | u_SC > 0.85 (Chapter 2) vs u_SC > 0.90 (VE1 escalation rule) | Explicitly distinguish in Chapter 4: the > 0.85 threshold is the general self-consistency verification pass/fail gate; the > 0.90 threshold in VE1 is specifically for the neutral NLI escalation condition (systematic confabulation detection requires a higher bar). Different uses, different thresholds — state this explicitly to preempt committee questions. | Session 1 cross-check | OPEN |
| C4-05 | Stage 4a (post-generation confidence) must be framed as an efficiency gate, not a quality gate | Stage 4a does not determine answer quality — verification (Stage 5) does that. Stage 4a's û is an *efficiency filter*: it avoids running the expensive verification pipeline on answers the model itself rates as low-confidence. Write: "Stage 4a is a cost-saving pre-filter, not a replacement for verification. It catches obvious failures before committing to the full NLI + SC + SE pipeline." | Session 3 | OPEN |
| C4-06 | û weight correction (Session 3 stated wrong weights) | Correct weights in û: u_token = 0.20, u_dropout = 0.20, u_SC = **0.20**, (1−H_sem) = **0.40**. NOT 0.30/0.30 as mistakenly stated in Session 3. SE gets double weight (0.40) because it is the only signal that catches systematic misconceptions and has the strongest empirical AUROC (0.79, Farquhar et al. 2024). | Session 4 correction | OPEN |
| C4-07 | Initial vs calibrated weights must be distinguished | Present equal initial weights (0.25 each) as the *starting point*, then explain calibration produces the final weights (0.20/0.20/0.20/0.40). Write: "Initial weights are set equal at 0.25 for all four signals. After temperature scaling calibration on the held-out calibration set, weights are refined to reflect each signal's empirical discriminative power, with semantic entropy receiving elevated weight (≈0.40) given its superior AUROC for confabulation detection." | Session 4 | OPEN |
| C4-09 | Blind spot table should appear in Chapter 4 | Include a 4×4 table showing which signal catches which failure mode (lexical uncertainty / epistemic uncertainty / reasoning instability / systematic misconception). This directly answers the committee question "why four signals?" | Session 4 | OPEN |
| C4-25 | Self-consistency denominator must use unique off-diagonal pairs (not M² with diagonal self-similarity) | For both Stage 4a (`u_consistency`) and Stage 5 (`s_avg`), define self-consistency as the mean over unique cross-chain similarities only: $u_{SC}=\frac{2}{M(M-1)}\sum_{i<j} \mathrm{sim}(v_i,v_j)$. Do **not** use $(1/M^2)\sum_i\sum_j \mathrm{sim}(v_i,v_j)$, which adds a fixed diagonal bonus of $1/M$ and inflates confidence. Required text for §4.4/§4.5: "Self-consistency is computed over distinct chain pairs (i<j); diagonal self-similarity terms are excluded because they are always 1.0 for L2-normalised embeddings and do not provide evidence of inter-chain agreement." | Session 39 follow-up (revert) | OPEN |
| C4-22b | "Reasoning-chain supervision" is task-conditioned; for classification benchmarks it may collapse to label supervision | **RESOLVED via C4-24 (Session 30 Fix A).** Option A was implemented: task-aware CoT prompting now applied to FEVER and StrategyQA so `reasoning_chain` always contains a reasoning trace + label, not a bare label token. See C4-24 for the required §4.4 write-up. The `(q,a)`-only vs `(q,c,a)` ablation still needed to quantify chain contribution empirically (ablation A7). | Session 31 diagnostics → Session 30 Fix A | DONE |
| C4-24 | Task-aware CoT prompting for training benchmarks — `reasoning_chain` always contains a reasoning trace after Session 30 Fix A | Write in §4.4 (Generation and Fine-Tuning Objective): "Tier 2 and Tier 3 generation uses task-aware prompts that produce structured reasoning traces for all three training benchmarks. For open-ended benchmarks (TriviaQA, Natural Questions), the prompt elicits free-form chain-of-thought: 'Question: {q}\nThink step by step:\nAnswer:'. For the classification benchmark (FEVER), the prompt produces a rationale + label: 'Reasoning: {brief explanation}\nAnswer: {label}'. This ensures the `reasoning_chain` field stored in episodic memory always contains genuine reasoning content, not a bare label token. The thesis claim of 'verified reasoning-chain supervision' (§4.3) therefore holds uniformly across all training benchmarks. Transfer evaluation benchmarks (TruthfulQA, StrategyQA, ARC-Challenge) receive appropriate evaluation prompts at inference time only — they are never fine-tuned on. At fine-tuning time, the full reasoning chain (reasoning + answer) is the supervision target — richer than answer-only supervision and consistent with chain-of-thought distillation literature (Wei et al. 2022; Ho et al. 2022)." Also note label extraction: FEVER uses `extract_fever_label()` (regex-based, not substring match) to parse the label from the rationale-style output. The harness applies this extractor before scoring. StrategyQA uses `extract_strategyqa_label()` for transfer evaluation only. Verify this is stated in §5.1 Experimental Setup. **NOTE (Session 35):** HotpotQA has been removed from the thesis training benchmark suite. Do not list it as a training benchmark anywhere in Chapter 4. The final training suite is FEVER + TriviaQA + Natural Questions only. | Session 30 Fix A + Session 35 | OPEN |

### §4.5 — Verification Pipeline (Stage 5)

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C4-08 | û_stored must be explicitly explained as NOT derived from û | State clearly: "û_stored is computed from the Stage 5 verification pipeline outputs — not from the Stage 4a post-generation confidence û. This is essential because Tier 3 episodes never produce a û value, yet they require a stored confidence for Tier 1 routing. The verification pipeline is the only stage shared by all three tiers." | Session 4 | OPEN |
| C4-22 | Fine-tuning trains on verified reasoning chains, not just answer strings | Fix 5 in Session 20 changed the training target from `entry.answer` to `entry.reasoning_chain`. The full reasoning chain (CoT output including the answer) is the supervision signal. Write in §4.7: "CAEM trains on verified reasoning chains — the full model output including any chain-of-thought steps — not on isolated answer strings. This provides richer sequence-level supervision and is consistent with the thesis's claim of 'verified reasoning-chain supervision' (§4.3). Short answer-only supervision would fail to propagate verification quality into the model's reasoning process." This matters because it directly answers "what exactly is the training target?" | Session 20 Fix 5 / Impl Log | OPEN |
| IMPL-02 | **Hallucination rate is an operational proxy** — define precisely before using | Define in Chapter 4: "Hallucination rate is operationalised as the fraction of queries where the model generates a confident wrong answer: EM=0 AND û ≥ 0.5. This captures 'confident confabulation' — the most harmful failure mode — rather than all errors. Low-confidence wrong answers (û < 0.5) are detected by Stage 4a and routed to Tier 3 rather than stored." In Chapter 5, always report hallucination reduction using this definition. Never write "hallucination rate" without this anchor definition visible in the text or a footnote. | Session 25 / Impl Log | OPEN |

### §4.6 — Episodic Memory

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C4-15 | Immutable/mutable field split must be explained as architectural choice, not implementation detail | The reason content fields are immutable is that the fine-tuning dataset is derived from them — updating reasoning chains mid-cycle would mean training on a moving target. State explicitly: "Episode content fields (question, reasoning chain, answer, embedding) are immutable after storage. Quality and usage metadata (u_stored, success_rate, retrieval_count, retroverified) are mutable and updated across cycles. This separation ensures the fine-tuning dataset is derived from a stable, auditable set of verified content." | Session 7 | OPEN |

### §4.7 — Self-Improvement Loop (Stage 8)

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C4-16 | Stage 8 must be framed as verified rejection sampling, with ReST cited as closest prior work | Write: "Stage 8 implements verified rejection sampling: generated answers are accepted only if they pass the multi-signal verification pipeline, and only accepted samples are used for fine-tuning. This is analogous to ReST (Gulcehre et al. 2023), which alternates between a generate step and an improve step on high-reward samples. CAEM's specific contributions over ReST are: (1) multi-signal verification replacing a scalar reward model, catching distinct hallucination failure modes; (2) episodic memory enabling routing improvement across cycles; (3) retroactive re-verification enabling backward propagation of model improvement into memory quality." The committee will ask "how is this different from ReST?" — have this answer ready in the text. | Session 8 | OPEN |
| C4-19 | Stability-plasticity tradeoff must be named and justified — name the regularisation correctly | Chapter 4 must explicitly introduce the stability-plasticity tradeoff before presenting the regularisation method. Write: "Continual learning systems face a fundamental tension: the model must be plastic enough to absorb new verified knowledge each cycle, yet stable enough to retain general-purpose language capabilities. Without regularisation, each fine-tuning cycle would overwrite previously learned weights — a phenomenon known as catastrophic forgetting (McCloskey & Cohen, 1989). CAEM addresses this via L2 regularisation with uniform parameter weighting, adding a penalty λ·||θ − θ_prev||² to the fine-tuning loss. This penalises large deviations from the previous cycle's weights, preserving general capabilities while allowing targeted improvement on verified knowledge." Then add: "Full Elastic Weight Consolidation (EWC, Kirkpatrick et al. 2017) weights this penalty per-parameter by the Fisher Information Matrix (FIM), concentrating regularisation on the parameters most critical to prior performance. CAEM uses the uniform-weight approximation (FIM ≈ uniform), which is computationally efficient and appropriate for diverse QA fine-tuning where gradient signal distributes broadly. The approximation validity is supported by the MMLU retention results (see §5.x). The plasticity-stability balance is controlled by λ=0.01 [DES] — an independently validated forgetting abort guard (retention threshold 0.07 per cycle, triggering automatic weight restoration) provides a hard safety net independent of λ calibration." This framing is essential — the committee will ask both "why not just fine-tune without regularisation?" and "why not full EWC?" | Session 26 + Session 29 | OPEN |
| GEN-11 | CAEM must be explicitly positioned within the continual learning literature | CAEM is a continual learning system — it trains sequentially over multiple cycles without full retraining. The writing must acknowledge this positioning. Write in Chapter 4: "CAEM operates in the continual learning paradigm: the model is updated across sequential cycles without access to all prior training data simultaneously. Unlike standard continual learning benchmarks (e.g., split-MNIST, Permuted-MNIST), CAEM's task sequence is not a sequence of distinct tasks — it is repeated improvement on the same tasks with progressively higher-quality training data. This means catastrophic forgetting manifests differently: not as forgetting a prior task, but as degrading general-domain capability. CAEM addresses this via L2 regularisation with uniform parameter weighting — a principled approximation of EWC (Kirkpatrick et al. 2017) that is computationally efficient for diverse QA fine-tuning where the Fisher information is approximately uniform. The full EWC method is noted as a future upgrade path for settings with greater domain shift." Also cite McCloskey & Cohen (1989) as the original catastrophic forgetting paper. This positions CAEM correctly in the literature and pre-empts the committee question "how is this different from normal fine-tuning?" | Session 26 + Session 29 | OPEN |

### §4.8 — Calibration

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| IMPL-03 | **Temperature scaling calibration cites Guo et al. 2017** — must be cited and timed correctly | Write: "Confidence calibration uses temperature scaling (Guo et al. 2017): a single learnable scalar T is optimised by minimising the Expected Calibration Error (ECE) on a 500-sample calibration set. Signal weights for the û composite are fitted via logistic regression optimising AUROC on the same set, with semantic entropy initialised at elevated weight (≈0.40) consistent with Farquhar et al. (2024). Temperature scaling is applied once after Cycle 0, before Cycle 1 fine-tuning, using the base model's output distribution." | Session 22 / Impl Log | OPEN |

### §4.9 — Theoretical Analysis

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C4-17 | Purity theorem must include a worked example — ILLUSTRATIVE ONLY in Chapter 4 | In Chapter 4 (methodology), present the theorem formula symbolically first: P = pα / (pα + (1−p)(1−α)). Then add ONE illustrative example labeled explicitly as pedagogical: "For example, if p = 0.70 and α = 0.85, then P ≈ 0.93." These numbers are for illustration only — they are NOT experimental results. State the required condition **α > ½** explicitly (NOT "p > (1−α)" — that condition is mathematically incorrect; see Theorem 4.1 proof in chapter_4.tex for the derivation). In Chapter 5 (results), REPLACE the illustrative numbers with actual measured values from `outputs/purity_validation/theory_validation.json`. Never present the Chapter 4 illustrative example as if it were an experimental finding. | Session 9 + Session 26 + **CORRECTED Session 31** | OPEN |
| C4-23 | Convergence claim must be scoped — never claim "infinite cycles → zero hallucination" | Session 22 confirmed this claim would be TOO STRONG and academically indefensible. The correct claim is: "Improvement is monotone and bounded by (p, α) — the model's generation accuracy and the verifier's accuracy set the practical ceiling. The purity theorem guarantees data quality improvement per cycle, not eventual perfection." Write: "CAEM does not claim to eliminate hallucination — it provides a bounded, monotone improvement trajectory. The ceiling is determined by the base model's capacity (p) and the verification pipeline's accuracy (α). Equilibrium is expected around Cycles 7–9 (C7–C9); diminishing improvement magnitude across later cycles is the empirical signal confirming approach to this bound." Do not write: "across infinite cycles, hallucination approaches zero" or "CAEM converges to a hallucination-free model." **Session 35 note:** old entry said "Diminishing returns between Cycle 2 and Cycle 3" (3-cycle plan). Updated to equilibrium at C7–C9 (10-cycle plan). | Session 22 + Session 35 / Impl Log | OPEN |
| C4-18 | Coupled recurrence convergence must be explained in plain English alongside the mathematics | Write: "The two loops are coupled: a better model produces purer memory (via better generation and better verification signals), and purer memory produces a better model (via higher-quality fine-tuning data). This positive feedback is bounded by the model's capacity ceiling and the verification pipeline's maximum achievable accuracy — the system converges rather than diverging. Ten cycles are run; empirical equilibrium is expected at C7–C9. Empirical confirmation of convergence comes from diminishing improvement magnitude Δ across consecutive later cycles in the Chapter 5 results." **Session 35 note:** old entry said "three cycles … between cycles 2 and 3" (3-cycle plan). Updated to 10-cycle plan with equilibrium at C7–C9. | Session 9 + Session 35 | OPEN |
| TH-01 | Theory formulas in Chapter 4 must be defined symbolically — no experimental numbers | All three theories (Purity Theorem, Recurrence Monotonicity, Convergence) must be stated in Chapter 4 using symbolic variables only: p (model accuracy), α (verifier accuracy), P (memory purity), Δ_n (improvement delta). Illustrative worked examples must be explicitly labeled "for illustration only" or "for example." The ONLY place actual experimental values (e.g., p=0.62, α=0.88, P_obs=0.91) appear is Chapter 5, sourced from `theory_validation.json`. This prevents committee confusion between projected and measured claims. | Session 26 | OPEN |
| TH-02 | α (verifier accuracy) measurement must be precisely defined in Chapter 4 | Define α explicitly as: α = (TP + TN) / N_purity, measured on the 500-sample purity validation set. Where: TP = model answer correct (EM=1) AND verifier accepts (û ≥ 0.5); TN = model answer wrong (EM=0) AND verifier rejects (û < 0.5). Write: "Verifier accuracy α is measured as the fraction of questions on which the verifier's accept/reject decision matches the ground-truth correctness of the model's answer. A false positive (FP) is a stored hallucination — the model is wrong but the verifier accepts it. A false negative (FN) is a correct answer that is unnecessarily rejected. The purity theorem is sensitive to FP rate: high FP rate lowers α and weakens the purity guarantee." This definition is implemented in `scripts/run_purity_validation.py` and reported in `theory_validation.json`. | Session 26 | OPEN |
| PUB-02 | **Purity theorem requires a formal mathematical proof** — not just an algebraic statement | The theorem P = pα / (pα + (1−p)(1−α)) is a conditional probability manipulation, but the proof must be formally stated. Write the proof as: (1) define the Bayesian model (correct answer with prob p, verifier accepts correct with prob α, verifier accepts wrong with prob 1−α); (2) apply Bayes' theorem; (3) show P = P(correct | accepted) = pα / (pα + (1−p)(1−α)); (4) prove P > p iff p > (1−α). Include this in Chapter 4 as a theorem-proof block (LaTeX `\begin{theorem}...\end{theorem}` / `\begin{proof}...\end{proof}`). A reviewable proof elevates the paper from "empirical trick with a formula" to "theoretically grounded contribution." **Priority: REQUIRED for conference submission.** | Session 29 / ACL analysis | OPEN |

---

## Chapter 5 — Result Analysis

### §5.1 — Calibration Results

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-01 | Calibration set and purity validation set must be explicitly separated | State: "The 500-sample calibration set (for temperature scaling) and the 500-sample purity validation set (for theorem validation) are drawn from non-overlapping portions of each benchmark's validation split." Use the allocation table from the plan. | CC5 fix | OPEN |
| C5-11 | Calibration results must be presented as a standalone finding, not buried in setup | In Chapter 5 §5.1, present calibration as an explicit result: (1) ECE before temperature scaling; (2) ECE after temperature scaling — improvement Δ; (3) Temperature scalar T fitted value; (4) Actual û signal weights after calibration (compare to projected 0.20/0.20/0.20/0.40 — report the actual numbers, not the projection). Write: "Temperature scaling calibration on the 500-sample calibration set reduced ECE from X to Y (ΔT = Z). The calibrated û signal weights converged to [actual values], [consistent with / diverging from] the projected distribution." Source: `calibration/calibrated_config.json`. | Session 26 | OPEN |

### §5.2 — Main Results

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-04 | Chapter 5 must have a single narrative thread connecting all results back to the closed-loop architecture argument | Do not present results as a flat list of numbers. Every result section should answer one of three questions: (1) Does CAEM outperform baselines that lack a closed loop? (2) Do the three internal mechanisms each demonstrably contribute? (3) Do the theoretical predictions hold empirically? Structure: open with the primary result (hallucination reduction), use ablations to prove each mechanism's contribution, use the theory validation tables to elevate CAEM from heuristic to principled. Close with the narrative: "Together, these results confirm that CAEM's closed-loop architecture delivers consistent, measurable, and theoretically grounded hallucination reduction across six qualitatively distinct benchmarks — three training benchmarks targeting fact verification, open-domain QA, and entity-anchored QA, and three transfer benchmarks testing generalisation to epistemic calibration, multi-step reasoning, and science question answering." **Session 35 note:** old entry said "four qualitatively distinct benchmarks" (HotpotQA era). Updated to six benchmarks with the specific failure-mode framing for each group. | Session 10 + Session 35 | OPEN |
| C5-05 | Statistical significance testing is required before claiming CAEM "outperforms" any baseline | Accuracy differences between CAEM and baselines must be tested for statistical significance before any claim of "outperformance." Use McNemar's test for paired binary comparisons (correct/incorrect per question). Report: χ² statistic, p-value, and whether p < 0.05. Write: "Statistical significance of accuracy improvements was assessed using McNemar's test on paired question-level outcomes (correct/incorrect). Improvements with p < 0.05 are reported as statistically significant." If a result is directionally positive but not significant, write: "The improvement on TruthfulQA is directionally consistent but does not reach statistical significance at p < 0.05, likely due to the smaller effective sample size (N=X)." Do NOT claim "CAEM significantly outperforms" without running the test. Source: per-question EM arrays in the eval JSON files. | Session 26 | OPEN |
| C5-12 | McNemar's test and bootstrap CI are already implemented — use them, do not rewrite | `eval/metrics.py` (Session 20, Fix 8) already contains `mcnemar_test(scores_a, scores_b)` (chi-squared with Edwards continuity correction) and `bootstrap_ci(scores, n_bootstrap=1000, ci=0.95)` (non-parametric). When writing Chapter 5, use these functions directly on the per-question EM arrays saved in `outputs/eval/{bm}_cycle{n}.json`. Do not write new statistical test code — the implementations are verified and tested. Write in §5.2: "Statistical significance was assessed using `mcnemar_test()` from `eval/metrics.py` — a McNemar's chi-squared test with Edwards continuity correction on paired per-question correct/incorrect outcomes. Non-parametric 95% confidence intervals were computed via `bootstrap_ci()` (B=1000 resamples). Both functions were written and validated before the experiment phase." | Session 20 / Impl Log | OPEN |
| GEN-12 | Statistical significance must be computed and reported — do not rely on effect size alone | The committee will ask "are these improvements statistically significant?" for every comparison in Chapter 5. The rule: any claim of CAEM outperforming a baseline requires a McNemar's test (p < 0.05). Any improvement that cannot be tested must be labeled "directionally positive, not statistically significant at N=X." Practical plan: (1) Save per-question EM arrays from `eval/{bm}_cycle{n}.json`; (2) For each CAEM vs baseline pair, construct the 2×2 contingency table; (3) Run McNemar's test (scipy.stats.mcnemar); (4) Report χ², p-value, and effect size (odds ratio). | Session 26 | OPEN |

### §5.3 — Mechanism Evidence Table

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-02 | Report five cycle-level metrics in a single table to prove all three mechanisms are working | Include a table with columns: Cycle, Hallucination Reduction (%), Tier 1 Fraction (%), Tier 3 Fraction (%), MMLU Retention (%), Mean û_stored in Memory. A growing Tier 1 fraction proves memory is accumulating trusted knowledge. Stable MMLU proves forgetting is controlled. Rising mean û_stored proves retroactive cleaning is working. Report per-benchmark Tier 1 accuracy separately to catch FEVER near-miss issues. **Data source: `outputs/experiment_summary.csv`.** After EXP-MMLU-FIX (Session 37), MMLU Retention is now auto-computed per cycle and written to the `mmlu_retention_pct` column — you do NOT need to run `run_ablation.py` to populate this column. The `mmlu_retention_pct` value appears only on the first benchmark row for each cycle (blank for subsequent rows). | Session 8 + Session 37 | OPEN |
| C5-13 | Add a lightweight per-benchmark verifier diagnostics table to defend the benchmark-agnostic verifier claim | Add one compact table (main text or Appendix A) with one row per benchmark and these columns only: Tier 1 accuracy, storage rate, mean û_stored, and stored-sample precision proxy `mean(EM | stored=True)`. Compute from existing eval JSON outputs only; do not run extra training or per-benchmark recalibration. Write: "This table is a diagnostic validity check, not a second calibration pipeline. It verifies that the shared verifier remains acceptable across benchmark formats without introducing benchmark-specific fitted parameters." | Session 30 | OPEN |

### §5.4 — Theory Validation

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-03 | Theory validation must be presented as a structured experiment, not a side note | Chapter 5 must explicitly execute and report all three theory validations: **Theory 1 (Purity Theorem):** run the 5-step purity validation protocol. **Theory 2 (Recurrence Monotonicity):** report p and α per cycle; confirm p_0 < p_1 < p_2 < p_3 and α_0 < α_1 < α_2 < α_3. **Theory 3 (Convergence):** compute Δ per cycle; confirm Δ(2→3) < Δ(1→2). All three use the same purity validation set measurements — no extra experiments needed. Source: `outputs/purity_validation/theory_validation.json`. | Session 9 | OPEN |
| TH-03 | Chapter 5 theory validation tables must use actual script output — never plan projections | The three theory validation tables in Chapter 5 must be populated ONLY from: (1) `outputs/purity_validation/theory_validation.json` for Theory 1 and Theory 2; (2) `outputs/full_experiment/all_cycle_results.json` for Theory 3. The plan's projected example values (p=0.62, α=0.88 etc.) are planning estimates only — never copy them into Chapter 5. If actual values differ significantly from projections, report the real numbers and explain the divergence. | Session 26 | OPEN |
| TH-04 | Convergence Δ table must show the check column explicitly | In Chapter 5, the convergence table must include a "Convergence Check" column: Cycle gap / Accuracy Delta (Δ) / Check. Checks: Δ(C1→C2) < Δ(C0→C1) → ✅ or ❌; Δ(C2→C3) < Δ(C1→C2) → ✅ or ❌. If any check fails, state: "While Δ(Cn→Cn+1) did not strictly decrease between cycles X and Y, the overall trend confirms convergence toward a fixed point." Do NOT hide or omit a failed check. | Session 26 | OPEN |

### §5.5 — Ablations and §5.6 — Baselines

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-09 | Ablation and baseline sections need explicit structure — not just a table dump | Structure as two distinct sub-sections: (1) **Baseline comparison** — present all 6 baselines in one table (Zero-shot / CoT / RAG / Self-consistency / Vanilla FT / Memory-only). For each, state what single mechanism it lacks vs CAEM. (2) **Ablation analysis** — present 7 ablation configs (AB1–AB7) grouped by which mechanism they probe: memory quality group (AB2 no verification, AB5 NLI-only, AB4 no retroactive re-verification), routing group (AB1 no memory/all-Tier-3, AB6 no OR-condition), training group (A4 vanilla FT, A5 memory-only), signal group (AB7 no SE, AB3 no CoT). Each group answers one committee question. The most important ablations are AB4 (no retroactive re-verification — uniquely proves memory quality) and AB3 (no CoT supervision — proves reasoning-chain supervision value). **CRITICAL PLANNING NOTE:** AB3, AB4, and A4 (Vanilla FT) each require a SEPARATE training run with a modified flag (`--no_cot_supervision`, `--disable_reverification`, `u_stored_threshold=0.0`). They cannot be derived from the main 10-cycle experiment. Budget two additional mini-runs (3 cycles each) before the experiment session. If skipped, those rows appear as N/A and the two most novel mechanism claims (reasoning-chain supervision and retroactive re-verification) cannot be empirically defended. **Ablation benchmark set:** All 6 benchmarks — FEVER / TriviaQA / NQ (in-domain training set) + TruthfulQA / StrategyQA / ARC-Challenge (out-of-domain transfer set). Session 36 updated `run_ablation.py` to evaluate every ablation condition across the full 6-benchmark suite. This cleanly separates two variables: the **in-domain delta** (does removing this module hurt performance on the data it was trained on?) from the **out-of-domain delta** (does removing it hurt zero-shot generalization?). Do NOT revert to the old 4-benchmark scope — confounding these two variables was a methodological flaw. In the §5.5 ablation setup paragraph, state both the benchmark scope and the isolation rationale: "All ablation variants are evaluated across the full six-benchmark suite (three in-domain training benchmarks and three out-of-domain transfer benchmarks). This design cleanly isolates the architectural contribution of each component — in-domain delta captures mechanism effectiveness on trained data, while out-of-domain delta captures generalization. Evaluating on in-domain benchmarks only would confound the ablation with dataset distribution differences." | Session 26 + Session 35 + Session 36 | OPEN |
| C5-14 | **⚠️ DO NOT MISS — §5.5 must proactively explain the clean ablation isolation methodology and two ablation mechanisms** | Session 36 corrected a methodological flaw where ablations were only evaluated on transfer datasets, confounding architectural contribution with dataset distribution shift. The thesis text must proactively disclose how this is fixed — reviewers are highly sensitive to methodological inconsistency and will ask. Required text blocks: **(1) Six-benchmark span explanation:** Write: "While CAEM trains only on three in-domain datasets (FEVER, TriviaQA, Natural Questions), all ablation variants are evaluated across the full six-benchmark suite. This cleanly isolates the architectural contribution of each component — the in-domain delta measures mechanism effectiveness on trained data, while the out-of-domain delta measures zero-shot generalization. Evaluating on out-of-domain benchmarks only would confound the ablation: a performance drop could reflect either the removed mechanism or the dataset distribution shift, making causal attribution impossible." **(2) Two ablation mechanism types:** Write: "Ablations fall into two categories with different methodological implications. *Inference-time ablations (AB1, AB2, AB5, AB6, AB7)* isolate architectural contributions by intercepting or disabling routing and verification logic at runtime, using the exact same Cycle 10 weights as the full CAEM model — they measure the inference-time value of each component. *Training-time ablations (A4 Vanilla FT, AB3 No-CoT, AB4 No-Reverification)* capture the impact of learning dynamics by running entirely separate fine-tuning trajectories with the respective mechanism disabled for all 10 cycles — they measure the training-time value of each component." This disclosure prevents the most common reviewer objection to ablation studies and must appear in the §5.5 experimental setup paragraph, before the results table. | Session 36 | OPEN |
| IMPL-04 | **StrategyQA uses train split — explicit justification needed in §5.3 setup** | Write: "StrategyQA evaluation uses the `wics/strategy-qa` train split (~2,290 questions) because the test split provides no public ground-truth labels. The CAEM self-improvement loop trains only on its own verified generations — not on dataset labels — so using the labelled train split as held-out evaluation introduces no data leakage." Cite the split size and note the allocation: 500 calibration + 500 purity validation + remaining for evaluation. | Session 23 / Impl Log | OPEN |

### §5.7 — Computational Efficiency

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-06 | Computational cost / efficiency analysis must appear as a dedicated results subsection | Chapter 5 must include a §5.7 Computational Efficiency section reporting: (1) Per-tier mean latency (ms) from `experiment_summary.csv` mean_latency_ms column; (2) Tier distribution across cycles (showing Tier 1 fraction growing → average latency falling); (3) GPU memory peak during fine-tuning; (4) Total GPU hours for the full experiment. Write the cost-benefit argument: "The routing architecture provides a computational dividend across cycles: as Tier 1 fraction grows from ~5% (Cycle 0) to ~38% (Cycle 3), mean query latency falls from ~Xms to ~Yms — the same system becomes faster as it accumulates knowledge. CAEM's total inference cost at Cycle 3 is Z% lower than an equivalent all-Tier-3 baseline (A2 ablation), while achieving higher accuracy." | Session 26 | OPEN |

### §5.8 — Stability-Plasticity Analysis

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-MMLU-01 | **⚠️ DUAL-METRIC DESIGN — explain both forgetting instruments clearly; do not conflate them** | CAEM uses two separate forgetting instruments that serve different purposes. Writers must understand both or §5.8 will confuse reviewers. **Instrument 1 — TriviaQA abort guard (internal engineering):** `_forgetting_score()` in `self_improvement.py` evaluates the model on 50 held-out TriviaQA pairs (exact-match) before and after each fine-tuning step. If the post/pre retention ratio falls below 0.93, the cycle aborts and weights are restored. This is an engineering safety net — fast (50 pairs ≈ 2 min), in-domain (TriviaQA is a training benchmark), NOT the reported metric. Do not report TriviaQA forgetting scores as the forgetting result in the thesis. **Instrument 2 — MMLU retention (reporting metric):** `_mmlu_score()` in `self_improvement.py` evaluates 200 MMLU multiple-choice questions after every cycle. MMLU is domain-neutral (57 academic subjects), never trained on in any cycle, and uses the same protocol as `run_ablation.eval_mmlu_retention()` — making it the appropriate reported forgetting metric. **Data source after EXP-MMLU-FIX (Session 37):** `mmlu_retention_pct` is auto-written to `outputs/experiment_summary.csv` — one value per cycle on the first benchmark row. You do NOT need to run `run_ablation.py` separately just to get per-cycle MMLU retention numbers. Run ablation only for the ablation-condition MMLU column in Table 5.5. Required §5.8 disclosure text: "CAEM employs two distinct forgetting instruments. An internal TriviaQA abort guard (`_forgetting_score`, 50 held-out pairs) provides a per-cycle safety net, restoring weights if relative retention falls below 0.93; this guard was never triggered during the reported 10-cycle experiment. MMLU accuracy (200 questions, `cais/mmlu` validation split, 4-choice multiple-choice) serves as the reported domain-neutral forgetting metric — MMLU is never trained on in any cycle, providing a clean cross-benchmark measure of general-capability retention." | Ch5 §5.8 | Session 37 / EXP-MMLU-FIX | OPEN |
| C5-07 | Stability-plasticity analysis must be a named subsection in Chapter 5, not just one number | Dedicate §5.8 to "Stability-Plasticity Analysis" covering four items: **(1) MMLU Retention per cycle** — target ≥93%; **data source: `outputs/experiment_summary.csv` column `mmlu_retention_pct`** (auto-computed per cycle since EXP-MMLU-FIX, Session 37 — no need to run `run_ablation.py` separately); **(2) Forgetting guard history** — read `fine_tune.forgetting_score` from `retroverify_cycle{n}.json` per cycle — did any cycle trigger the abort threshold (ratio < 0.93)?; **(3) Fine-tuning loss** per cycle from `retroverify_cycle{n}.json` `fine_tune.final_train_loss`; **(4) Plasticity evidence** — EM improvement per cycle from `experiment_summary.csv`. Structure: "CAEM's L2 uniform regularisation successfully navigates the stability-plasticity tradeoff. On the plasticity side, EM improves across all six benchmarks across 10 cycles. On the stability side, MMLU retention remains at X% throughout (target ≥93%), confirming that general language capabilities are preserved. The TriviaQA abort guard never triggered (all per-cycle ratios ≥ 0.93), indicating that λ=0.01 [DES] provides sufficient regularisation at this task scale. This empirically validates the uniform-weight L2 approximation of EWC." If MMLU retention drops below 93%: flag catastrophic forgetting immediately — increase λ or consider full EWC. **See C5-MMLU-01 for required dual-instrument explanation.** | Session 26 + Session 29 + Session 37 | OPEN |

### §5.9 — Memory Utilization and Error Analysis

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-10 | Memory utilization and saturation must be reported | Report in Chapter 5: (1) Total episodes stored per training benchmark (FEVER, TriviaQA, Natural Questions) per cycle — transfer benchmarks do not contribute to memory; (2) Total memory size at end of Cycle 10 vs capacity K=1,000,000 — is memory near saturation? (at 5,000 questions × 10 cycles × 3 training benchmarks the maximum is ~150K episodes, so saturation is unlikely at this scale); (3) Mean û_stored distribution (histogram or quartiles) — not just the mean; (4) Retroactive re-verification pruning rate per cycle. Write: "By the end of Cycle 10, the episodic memory contained N episodes across 3 training benchmarks (X% of capacity K=1,000,000)." If memory grows without slowing (no saturation pressure), state this confirms the memory design is appropriate for the experimental scale. Source: `experiment_summary.csv`. **Session 35 note:** old text said "4 benchmarks" (HotpotQA era) and "K=1000" (early config). Corrected: 3 training benchmarks, K=1,000,000. | Session 26 + Session 35 | OPEN |
| C5-08 | Error analysis and failure modes must appear as a dedicated subsection | Include §5.9 Error Analysis covering: (1) What question types CAEM still fails on — qualitative sample of 5–10 failure cases from `outputs/eval/{bm}_cycle10.json` (EM=0 samples); (2) TruthfulQA failure mode: memorising specific misconception pairs vs genuinely learning epistemic resistance — look for cases where CAEM Cycle 10 still confidently outputs a known misconception; (3) TriviaQA failure mode: alias coverage failures — cases where the model produces a semantically correct answer that is not in the alias list and scores EM=0 despite being arguably correct; note this is a scoring limitation, not a model failure, and discuss F1 alongside EM for TriviaQA; (4) Natural Questions failure mode: temporal brittleness — questions whose answers have changed since the DPR Wikipedia corpus was built (2018 snapshot); (5) FEVER near-miss failures: report Tier 1 accuracy per benchmark separately — anomalously lower FEVER Tier 1 accuracy than TriviaQA/StrategyQA confirms near-miss retrieval as the cause (adversarially similar claims with opposite labels); (6) Residual hallucination rate at Cycle 10 (EM=0 AND û≥0.5). Write: "Despite the overall improvement, CAEM retains systematic failure modes on [X type] questions. Honest failure analysis strengthens the thesis — a committee that sees no failures will not believe the results." **Session 35 note:** HotpotQA has been removed. Replace the old HotpotQA multi-hop failure analysis with TriviaQA alias-coverage and NQ temporal-brittleness analyses. | Session 26 + Session 35 | OPEN |

---

## Chapter 6 — Conclusion

### §6.1 — Summary of Findings

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C6-01 | Conclusion must explicitly state what each theoretical contribution provides beyond experiments | Write: "The data purity theorem provides a formal guarantee that CAEM's verification pipeline produces memory of strictly higher purity than unfiltered generation, under the verifiable condition p > (1−α). The coupled improvement recurrence provides a theoretical basis for the observed monotone accuracy gains across cycles. Together, these theoretical results distinguish CAEM from heuristic self-training approaches and provide a principled basis for extending the system to new benchmarks and domains." | Session 9 | OPEN |
| C6-02 | Three theories must be listed with their experimental test and expected result | Refer back to the three-theory validation structure from Chapter 5: **Theory 1 — Data Purity Theorem**: claim, test, expected result. **Theory 2 — Coupled Improvement Recurrence**: monotonicity claim, per-cycle p and α table. **Theory 3 — Convergence**: diminishing Δ claim, empirical confirmation. Convergence is a theoretical prediction (bounded monotone maps → fixed point), NOT an assumption. The three theories together are CAEM's main contribution over heuristic self-training. | Session 9 | OPEN |

### §6.2 — Limitations

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| IMPL-05 | **SBERT routing is structure-sensitive, not entity-sensitive** — disclose in limitations | Write (text from implementation log): "CAEM's routing mechanism relies on SBERT (all-mpnet-base-v2) cosine similarity, which captures semantic structure but is insensitive to specific numeric or named-entity values within a repeated question pattern. Structurally similar questions with different factual answers may receive the same routing decision. This is an inherent limitation of dense-retrieval routing shared by all SBERT-based memory systems. Future work could augment routing with a lightweight entity-extraction filter forcing Tier 2 or 3 for questions with out-of-vocabulary numerics or named entities not present in the matched episode." | Session 26 / Impl Log | OPEN |
| GEN-04 | Ground truth dependency must be scoped precisely, not overstated | The fine-tuning stage requires labeled answers — this is true of all supervised systems. The actual specific limitation is that NLI verification (Stage 5) uses benchmark ground-truth labels. SC and SE are fully reference-free. Write: "NLI-based verification relies on benchmark ground-truth labels as the entailment reference. The SC and SE components are reference-free and would generalise to unlabeled production settings. A natural extension — using Tier 3's retrieved Wikipedia passages as the NLI reference rather than labeled answers — would remove this dependency and is proposed as future work." | Session 6 | OPEN |

### §6.3 — Future Work

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| IMPL-06 | **Knowledge distillation and capacity ceiling as future work** (Session 22) | Write: "Three natural extensions of CAEM are identified. (1) *Model capacity scaling:* Flan-T5-Large (780M parameters) imposes an upper bound on verifiable knowledge capacity. Scaling to Flan-T5-XL (3B) or Flan-T5-XXL (11B) is expected to extend the convergence ceiling. (2) *Knowledge distillation:* The verified episodes in CAEM's memory store represent a curated, high-precision dataset. Distilling this knowledge into a smaller student model would produce a compact, deployment-ready model with higher knowledge density. (3) *Reference-free NLI:* Replacing benchmark labels with retrieved-passage NLI references would enable continuous self-improvement on unlabeled production queries." | Session 22 / Impl Log | OPEN |
| GEN-08 | Inference mode scope must be stated explicitly | After cycle 3, the system operates as a fixed improved system: model weights frozen, memory frozen. Routing, Tier 1 retrieval, Tier 2 guided generation, and Tier 3 RAG all still function. But NLI-based verification requires ground-truth labels, so new answers cannot be stored with full û_stored in open-domain inference. SC + SE are reference-free and could support a degraded inference-mode storage path. Write: "The self-improvement cycle operates over benchmark datasets with available ground-truth labels. After cycle completion, the improved model and populated memory support inference on new queries. Memory growth in open-domain settings is limited to SC+SE-based verification (reference-free). Continuous self-improvement in unlabeled settings is deferred as future work." | Session 7 | OPEN |

---

## Cross-chapter / General

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| GEN-01 | CAEM framed as "post-hoc detection" anywhere | Search all chapters for "detect", "post-hoc", "after generation" and verify each usage correctly frames CAEM as prevention + improvement, not detection. Detection language is fine when describing the *verification pipeline's role inside the loop* — not when describing CAEM as a whole. | Session 1 | OPEN |
| GEN-02 | Projected vs actual results | All numerical result claims in Chapters 1 and 4 are PROJECTED (plan estimates). Chapter 5 (Results) will replace with actual experimental values. Do not present projected numbers as confirmed results. | Standing rule | OPEN |
| GEN-03 | "Post-calibration weights" are planning estimates, not measured values | The û weights (0.20/0.20/0.20/0.40) shown throughout the plan are projected post-calibration values. They are NOT yet measured. When writing: present initial weights (0.25 equal) as the implementation starting point; present 0.20/0.20/0.20/0.40 as the projected calibrated values; report actual calibrated values in Chapter 5. Never imply calibration has already happened. | Session 4 | OPEN |
| GEN-05 | VE2 must be described as curated list matching, not learned pattern recognition | VE2 is string/phrase matching against TruthfulQA's published misconception taxonomy (~38 categories, Lin et al. 2021) — not a neural classifier. Write: "VE2 checks generated answers against a curated list of known misconception patterns drawn from TruthfulQA's published taxonomy (Lin et al. 2021). This is a dataset-specific rule applicable only when evaluating on TruthfulQA; it does not generalise to other benchmarks." Explicitly state that it is lightweight (no additional model required). | Session 6 | OPEN |
| GEN-06 | Near-miss retrieval: acknowledge FEVER risk specifically | FEVER is the most vulnerable benchmark: claims are adversarially constructed around the same entities with different labels (e.g., "Marie Curie born in Poland" SUPPORTS vs "Marie Curie born in France" REFUTES — very high cosine similarity, opposite labels). Write: "FEVER's adversarially constructed claims — structurally similar but label-distinct — pose the greatest near-miss retrieval risk among all six benchmarks. TriviaQA and Natural Questions have lower near-miss risk because different questions about different entities produce more distinctive embeddings. The Tier 1 combined score threshold (≥0.90) and the requirement for high û_stored provide partial mitigation for FEVER. Empirical evidence is reported in Chapter 5: per-benchmark Tier 1 accuracy; anomalously lower FEVER Tier 1 accuracy than TriviaQA/NQ/StrategyQA would confirm near-miss retrieval as the cause." | Session 7 + Session 35 | OPEN |
| GEN-13 | Scaling guide changes are pending a better device — hold final methodology text until device is confirmed | `LAB_PC_SCALING_GUIDE.md` documents 3 categories of changes required for better GPU: (1) L2 penalty moves fully to GPU (`_l2_penalty` no longer uses `.detach().cpu()`); (2) batch size scales up (batch=4 → 16 on A100); (3) passage index can scale to 5M passages. **Do not write final §4.7 methodology text or §5.7 efficiency numbers using RTX 3060 smoke-test values.** The full experiment may run on Kaggle (T4/P100), a university GPU (A100), or a rented instance (Vast.ai). After the experiment device is confirmed: (a) update §3.x hardware constraints to reflect actual experiment hardware; (b) verify the L2 penalty implementation matches the device (CPU offload vs GPU native); (c) use actual latency figures from `experiment_summary.csv` mean_latency_ms column — not smoke-test estimates. | Session 29 + LAB_PC_SCALING_GUIDE | OPEN |
| GEN-07 | Retrieval feedback (success_rate) depends on ground truth in benchmark mode | State scope explicitly: "In the current benchmark evaluation, episode success_rate is updated by comparing Tier 1 responses to ground-truth labels. Extension to production settings would require an explicit user feedback mechanism or active sampling verification." | Session 7 | OPEN |
| GEN-09 | Unified plan is pedagogical — never copy its numbers directly into the report | `caem-unified-plan-v3.tex` uses illustrative example values throughout to explain and teach the system. These are planning targets and worked examples, NOT measured results. (1) Chapter 4 — state theory formulas symbolically with one labeled illustrative example. (2) Chapter 5 — use ONLY actual values from `outputs/` files. (3) Never write "CAEM achieves 28% hallucination reduction" until the actual experiment confirms it. | Session 26 | OPEN |
| GEN-10 | Phase 1 submitted chapters (Ch1, Ch2) may need minor updates after experiments | After the full experiment runs, check: (1) Ch1 — does the actual hallucination reduction match the stated target? If materially different, update the claim. (2) Ch2 — benchmark accuracy figures cited are literature values — these do not change. | Session 26 | OPEN |

---

## Implementation Log → Thesis Mapping

These entries identify specific implementation decisions (from `caem-implementation-log.md`) that must surface in the thesis. Each specifies WHERE the decision belongs, WHAT to write, and WHY it matters for the committee. They are also listed inline within the relevant chapter sections above for easy reference during writing.

| # | Implementation Decision | Thesis Location | Status |
|---|---|---|---|
| IMPL-01 | Tier 1 skips Stage 5 — (a) ✅ arrow removed from Ch1 TikZ (chapter_1.tex line 158 — DONE); (b) don't draw it in Ch4 diagram | Ch1 TikZ ✅ DONE + Ch4 §4.2 | OPEN (Ch4 only) |
| IMPL-02 | Hallucination rate = EM=0 AND û≥0.5 (operational proxy) | Ch4 §4.x Metrics + Ch5 §5.1 | OPEN |
| IMPL-03 | Temperature scaling (Guo et al. 2017) + logistic regression AUROC calibration | Ch4 §4.8 | OPEN |
| IMPL-04 | StrategyQA uses train split — no leakage justification | Ch5 §5.3 Experimental Setup | OPEN |
| IMPL-05 | SBERT routing structure-sensitive limitation | Ch6 §6.2 Limitations | OPEN |
| IMPL-06 | Knowledge distillation + capacity ceiling as future work | Ch6 §6.3 Future Work | OPEN |

---

## Publication Elevation (ACL/EMNLP)

These entries identify gaps between the current thesis and conference-submission standard. Read `caem-implementation-log.md` Session 29 alongside these entries.

**REQUIRED** = reviewer will likely reject without it. **STRONGLY RECOMMENDED** = meaningfully increases acceptance probability.

| # | Gap | Required Action | Priority | Status |
|---|---|---|---|---|
| PUB-01 | **Strong baselines missing** — no comparison to GPT-3.5/4, Self-RAG, SelfCheckGPT | Add at minimum: (1) GPT-3.5 and Self-RAG published numbers from RA-ISF (Liu et al., ACL Findings 2024, arXiv:2403.06840). **⚠️ Session 35 update:** the RA-ISF numbers are on HotpotQA and StrategyQA. HotpotQA is no longer in CAEM's evaluation suite, so the HotpotQA comparison row is not usable. Only StrategyQA remains valid: GPT-3.5 vanilla StrategyQA=65.2; GPT-3.5 RAG=64.7; Self-RAG 13B=67.2. These three StrategyQA rows can still be added to Table 5.1 as a literature-baseline comparison column. New framing: "CAEM (780M params, Cycle 10) vs Self-RAG (13B) on StrategyQA — a 17× parameter efficiency comparison." For the remaining 5 benchmarks (FEVER, TriviaQA, NQ, TruthfulQA, ARC-Challenge), RA-ISF numbers are not available — use only our own baseline evaluations from `run_ablation.py`. (2) SelfCheckGPT (Manakul et al. 2023) — closest hallucination detection baseline — needs a fresh eval run post-experiment on TruthfulQA output. | REQUIRED | **StrategyQA RA-ISF numbers still valid — HotpotQA numbers NO LONGER APPLICABLE (benchmark removed)** |
| PUB-02 | **Purity theorem lacks a formal proof** — theorem-proof block needed in Chapter 4 | Write the proof as: (1) define the Bayesian model; (2) apply Bayes' theorem; (3) show P = P(correct | accepted) = pα / (pα + (1−p)(1−α)); (4) prove P > p iff p > (1−α). Use LaTeX `\begin{theorem}` / `\begin{proof}`. See Ch4 §4.9 entry above. | REQUIRED | OPEN |
| PUB-03 | **Human evaluation for TruthfulQA** — automated ROUGE-L proxy is weak | Conduct a small human evaluation (labmates or MTurk, N=100 subset, Cycle 0 vs Cycle 3 outputs). Present as a validation study: "Automated ROUGE-L and human evaluation are in agreement on X% of samples." Even 2-annotator Cohen's κ reported would substantially strengthen the TruthfulQA claim. | STRONGLY RECOMMENDED | OPEN |
| PUB-04 | **LoRA / PEFT comparison ablation** — reviewers will ask "why full fine-tuning?" | Add as Appendix ablation: LoRA (r=8, α=16) vs full fine-tuning + L2, reporting MMLU retention and EM improvement per cycle. If LoRA matches full FT on accuracy with better retention, note as future direction. If full FT + L2 is better, report why. Either way, the ablation pre-empts the question. | STRONGLY RECOMMENDED | OPEN |
| PUB-05 | **Full EWC vs L2 ablation** — needed to validate the L2 approximation claim | Add one ablation variant: full EWC (compute diagonal FIM, re-run Cycle 1–3). Compare MMLU retention, EM improvement, and compute time. Expected: similar MMLU and EM, but 30–40% higher compute time — validating the L2 approximation. Noted in `caem-implementation-log.md` Session 29 as planned. | STRONGLY RECOMMENDED | OPEN |
| PUB-06 | **Larger model experiment** — single model scale (780M) limits generalisability | Run Flan-T5-XL (3B) on TriviaQA only (clearest multi-alias EM metric in the current suite, and the training benchmark with the highest alias density — best for showing improvement signal). Report in Appendix B. Question to answer: does CAEM's improvement trend hold at larger scale, or does the convergence ceiling simply move up proportionally? **Session 35 note:** changed from HotpotQA (removed from suite) to TriviaQA. TriviaQA is the better choice anyway since its multi-alias EM scoring gives a less noisy signal than a single-answer benchmark. | STRONGLY RECOMMENDED | OPEN |

---

## Publication Elevation — Current State, What to Add, and When

**Read alongside the PUB-01–PUB-06 table above.** This section gives the honest gap assessment and a concrete timeline for each item.

### What we currently have (thesis-grade)

| Item | Status | Notes |
|---|---|---|
| 6 baselines (Zero-shot, CoT, RAG, SC, Vanilla FT, Memory-only) | ✅ Implemented | From `run_ablation.py` — available after full experiment |
| McNemar's test + bootstrap CI | ✅ Implemented | `eval/metrics.py` — apply post-experiment |
| Purity Theorem formula | ✅ Written | Symbolic in Ch4; needs formal proof block |
| Three theory validations | ✅ Designed | Needs actual `theory_validation.json` values |
| Temperature scaling calibration | ✅ Implemented | ECE before/after report in Ch5 §5.1 |
| 6 benchmarks × 10 cycles | ✅ Designed | Available after full experiment |

### What is missing for conference submission (and when to add it)

| # | Gap | When to start | How long | Blocker |
|---|---|---|---|---|
| PUB-01a | Self-RAG (Asai et al. 2023) baseline | After full experiment | ~2 days | Need same eval splits saved |
| PUB-01b | SelfCheckGPT (Manakul et al. 2023) baseline | After full experiment | ~1 day | Runs on top of existing generation outputs |
| PUB-01c | GPT-3.5 + Self-RAG numbers | ⚠️ PARTIALLY APPLICABLE — HotpotQA numbers no longer usable (benchmark removed Session 21). StrategyQA numbers still valid. | — | Usable: GPT-3.5 vanilla StrategyQA=65.2; GPT-3.5 RAG=64.7; Self-RAG 13B=67.2. NOT usable: HotpotQA EM=22.1 / 32.2 / 25.4 — HotpotQA is removed from CAEM's evaluation suite. Do NOT add those columns to Table 5.1. Add only the StrategyQA column with these literature numbers. Cite: Liu et al. ACL Findings 2024 (arXiv:2403.06840) for the StrategyQA figures; Asai et al. 2023 for the Self-RAG method description. |
| PUB-02 | Formal purity theorem proof | **NOW** — no data needed | ~2 hours | None |
| PUB-03 | Human eval for TruthfulQA (N=100) | After Cycle 3 outputs available | 1–2 days | Need 100 Cycle 0 vs Cycle 3 output pairs + 2 annotators |
| PUB-04 | LoRA vs full FT ablation | After full experiment device confirmed | ~4 hours compute | Needs extra training run |
| PUB-05 | Full EWC vs L2 ablation | After full experiment | ~4–6 hours compute | Needs FIM computation + extra training run |
| PUB-06 | Flan-T5-XL (3B) on HotpotQA | Post-submission if time permits | ~6 hours compute | Needs ≥16 GB VRAM |

### Priority order for conference upgrade

**Immediate (do before submitting thesis draft for supervisor review):**
- PUB-02 (formal proof) — zero cost, high credibility boost

**After full experiment completes:**
- PUB-01c (published GPT numbers) — zero compute, just citation lookup
- PUB-01b (SelfCheckGPT) — runs on your existing generation outputs, low cost
- PUB-01a (Self-RAG) — highest value baseline, needs fresh eval run

**If targeting ACL/EMNLP specifically:**
- PUB-03 (human eval) — reviewers will ask; even 2-annotator agreement is enough
- PUB-04 (LoRA ablation) — pre-empts the most common PEFT question
- PUB-05 (EWC ablation) — validates the L2 approximation claim formally

**Defer post-submission:**
- PUB-06 (XL model) — significant compute; report as future work if not done

---

## Visual Element Placement Convention (applies to ALL 6 chapters)

> **Standing rule — set in Session 33, applies to all future chapter work.**

### Where to place figures, algorithms, and tables

**All visual elements (figures, algorithms, tables) are inline floating environments — NOT hyperlinks, NOT appendix-only, NOT separate sections.**

The correct LaTeX pattern is:

```latex
% In prose: "... as shown in Figure~\ref{fig:routing-flowchart} ..."
% Then immediately after the paragraph that introduces it:
\begin{figure}[htbp]
  \centering
  ...
  \caption{...}
  \label{fig:routing-flowchart}
\end{figure}
```

LaTeX's `[htbp]` specifier tries placement in order: **h**ere → **t**op of page → **b**ottom of page → separate **p**age. This floats the figure close to where it is introduced without manual intervention.

**Rule: Place the `\begin{figure}` or boxed algorithm block immediately after the paragraph that first describes its content.** Never put a figure before the paragraph that explains it.

### Cross-referencing across chapters

- Always define `\label{fig:...}` or `\label{alg:...}` inside the float
- Reference from any chapter with `Figure~\ref{fig:arch-ch4}` or `Algorithm~\ref{alg:routing}`
- When referencing a figure from another chapter, add "(Chapter~N)" after the reference: `Figure~\ref{fig:arch-ch4} (Chapter~4)`
- **Do NOT use appendix hyperlinks or "see Appendix" cross-references for core methodology figures.** Appendices are for supplementary material (implementation details, scaling notes, raw tables).

### Chapter 5 pending figures (data-dependent)

For figures that cannot be drawn until experimental data arrives:
1. Write the `\label{fig:xxx}` definition in a placeholder comment block in the `.tex` file
2. Use `\ref{fig:xxx}` in the prose **now** — it will compile with `??` until the figure is added
3. Use a `% [PLACEHOLDER: FIG-C5-XX — insert after full_experiment data]` comment to mark the gap
4. When data arrives, drop the figure float into the gap and the reference resolves automatically

Example placeholder pattern:
```latex
Figure~\ref{fig:accuracy-by-cycle} shows accuracy across cycles for all benchmarks.
% [PLACEHOLDER: FIG-C5-01 — grouped bar chart from experiment_summary.csv]
% \begin{figure}[htbp] ... \label{fig:accuracy-by-cycle} ... \end{figure}
```

### Algorithm typesetting convention

Chapter 4 established the boxed-figure algorithm style (no external `algorithm` package needed):

```latex
\begin{figure}[htbp]
\centering
\small
\fbox{\begin{minipage}{0.92\textwidth}
\textbf{Algorithm N.M:} Title\\[4pt]
\textbf{Input:} ...\\
\textbf{Output:} ...\\[4pt]
\begin{enumerate}[leftmargin=2em, label=\arabic{*}.]
  \item Step 1 \hfill \textit{[comment]}
  ...
\end{enumerate}
\end{minipage}}
\caption{...}
\label{alg:name}
\end{figure}
```

**Use this pattern for all algorithm blocks in Chapters 4–6.** The `enumitem` package is already loaded in `main.tex`. Do NOT add `\usepackage{algorithm}` — it is not needed and may conflict.

---

## Visual & Formal Elements

**This section specifies every non-prose element required in each chapter.** For each element: what it is, what it must show, which source file or implementation module to use, which writing-suggestions entries it satisfies, and what implementation-vs-plan discrepancies to watch for.

> **Algorithm strategy:** Use the plan's algorithm blocks as **structural templates** — the pseudocode layout, `\begin{algorithmic}[1]` formatting, and step ordering are reusable. However, every algorithm must be reviewed against the implemented code in `caem/` before copying. The plan has known discrepancies listed per-algorithm below. Do not copy any algorithm block without reading the corresponding source first.

> **Figure strategy:** The best existing TikZ source for Chapter 4 is `chapter_1.tex` line 78 (the submitted architecture diagram). Use it as the base and extend — do not redraw from scratch. Unified plan figures at lines 848 and 1378 are also useful as starting points but have the same field-name and flow discrepancies as the algorithms.

> **Theorem strategy:** The unified plan uses a custom `proofbox` environment. For the report, use standard LaTeX `\begin{theorem}` / `\begin{proof}` / `\end{proof}`. The content is reusable; the environment names are not.

---

### Chapter 3 — Tables Only

| # | Element | What it must show | Source | Satisfies |
|---|---|---|---|---|
| VIS-C3-01 | **Table 3.1: Functional Requirements** | 3-column table: Requirement ID / Functional Requirement / Maps to Pipeline Stage. List one row per pipeline stage (Stage 1 query intake through Stage 8 self-improvement loop). Show that CAEM's architecture is fully derivable from requirements. | Design + pipeline spec | C3-01 |
| VIS-C3-02 | **Table 3.2: Non-Functional Requirements** | 4-column: NFR / Target Value / Rationale / Verification Method. Include: Tier 1 < 400ms, Tier 2 < 2.5s, Tier 3 < 6s, α ≥ 85%, GPU budget ≤ 50h, MMLU retention ≥ 93%, K=1000 capacity. | Design | C3-01, C3-02 |
| VIS-C3-03 | **Table 3.3: Hardware Constraints → Design Decisions** | 3-column: Hardware Constraint / Value / Design Decision It Forces. Rows: VRAM ≤ 12 GB → batch_size=4 + theta_prev CPU + 500K passages; GPU budget ≤ 50h → n=5000 + 3 cycles; K=1000 → cold-start seeding strategy. Frame constraints as design *drivers*, not apologies. | C3-02, LAB_PC_SCALING_GUIDE | C3-02 |

---

### Chapter 4 — Figures, Algorithms, Equations, Theorems (full inventory)

#### Figures / Diagrams

| # | Element | What it must show | Source | Satisfies | Watch for |
|---|---|---|---|---|---|
| FIG-C4-01 | **Figure 4.1: CAEM System Architecture** (main full-pipeline TikZ diagram) | All 8 stages as numbered boxes with directional arrows. Three routing paths clearly labelled (Tier 1 / Tier 2 / Tier 3). Memory store as a separate module. Stage 8 loop back to Stage 1. Use the Ch1 TikZ figure as the template and EXTEND it — do not redraw from scratch. | **PRIMARY SOURCE: `chapter_1.tex` lines 78–219** (the submitted architecture TikZ). Also see unified plan line 848 for the detailed pipeline version with decision diamonds. | C4-01, C4-03, IMPL-01 | **No Tier 1 → Stage 5 arrow** (IMPL-01 — this was the exact error fixed in Ch1, do not re-introduce it). The unified plan line 848 figure may also contain this arrow — check before adapting. Stage 5 box is Tier 2/3 only. |
| FIG-C4-02 | **Figure 4.2: Three-Tier Routing Decision Flowchart** | Decision flowchart with explicit diamond nodes. Flow: (1) compute u_pre → (2) Diamond: u_pre < 0.60? → YES → Tier 3; NO → (3) compute FAISS similarity s → (4) compute routing_score = 0.7·s + 0.3·û_stored → (5) Diamond: score ≥ 0.90? → Tier 1; score ≥ 0.65? → Tier 2; else → Tier 3. | **Unified plan line 848** has a routing section with decision diamonds — use as base. Also see `caem/router.py`. | C4-04, C4-10b, C4-20 | OR-condition (u_pre < 0.60) must appear as a **separate first diamond** before the routing score formula (C4-10b). Plan may show them in the wrong order or merged — verify. |
| FIG-C4-03 | **Figure 4.3: Confidence Signal Architecture** | Two-panel figure. Left panel: u_pre composition (C_conv + u_MC_pre → weighted sum with 0.60/0.40 weights). Right panel: û composition (u_token + u_dropout + u_consistency + u_entropy → weighted sum with 0.25/0.25/0.25/0.25 initial, then calibrated). | **Unified plan line 1378** is the routing/confidence flowchart — reuse the signal box layout. Also unified plan line 2147 (confidence figure). Read `caem/confidence.py` for actual field names. | C4-03, C4-06, C4-07, C4-09, C4-21 | Plan line 1378 uses `u_sc` and possibly `h_entropy_norm` — rename to `u_consistency` and `u_entropy` respectively. Plan shows SC with 3 samples — implementation uses N=10. Initial weights in the plan may be shown as 0.30/0.30/0.20/0.20 — correct to 0.25/0.25/0.25/0.25 initial, calibrated to ≈0.20/0.20/0.20/0.40. |
| FIG-C4-04 | **Figure 4.4: Verification Pipeline (Stage 5) Flowchart** | Sequential pipeline: generated answer → NLI scoring (p_entail) → SC scoring (s_avg) → SE scoring (h_norm) → VE1 check (neutral NLI AND u_consistency > 0.90 → escalate) → VE2 check (TruthfulQA only — misconception flag → reject) → û_stored = 0.5·p_entail + 0.3·s_avg + 0.2·h_norm → storage gate (û_stored ≥ threshold → store). | **Unified plan line 2147** (three-layer verification architecture figure) — reuse the layered pipeline layout. Read `caem/verifier.py` for exact field names. | C4-08, C4-22, BM-03, BM-04 | û_stored formula is from Stage 5 outputs — NOT from û (C4-08). VE2 applies TruthfulQA only — plan may show it as general (GEN-05). Plan uses `phi_SC`, `phi_NLI`, `phi_SE` notation — the report should use `s_avg`, `p_entail`, `h_norm` consistent with stored field names. |
| FIG-C4-05 | **Figure 4.5: Episodic Memory Episode Schema** | Two-column box diagram. Left column (IMMUTABLE): question, reasoning_chain, answer, embedding. Right column (MUTABLE): u_stored, success_rate, retrieval_count, retroverified. Caption: "Content fields are frozen at storage time — the fine-tuning dataset derives from them. Quality and usage metadata are updated across cycles." | No existing figure in the plan matches this exactly. Design fresh as a simple two-column annotated box. Read `caem/memory.py` or `caem/models.py` for the actual field list. | C4-15, C4-21 | No `passed_verification`, `ve1_triggered`, or `ve2_triggered` fields — these do not exist in the schema (C4-21). Unified plan memory schema may include these — do not include them. |
| FIG-C4-06 | **Figure 4.6: Self-Improvement Cycle Diagram** | Cyclic arrow diagram showing the cycle flow: Query Processing → Verification → Memory Update → Retroactive Re-verification → Fine-tuning (with L2 penalty) → Calibration (Cycle 0 ONLY, shown as a dashed branch) → next cycle. Label boxes C0, C1, C2, C3. | **PRIMARY SOURCE: `chapter_1.tex` lines 220–280** (submitted experimental workflow TikZ — C1/C2/C3 cycle boxes). Also see unified plan line 3246 (very similar structure). Reuse the cycle-box layout and arrow style. | C4-16, C4-19, IMPL-03 | Calibration box must appear only after C0 — both the Ch1 figure and the plan figure show calibration as a recurring step; change to one-time after C0 only. Plan cycle boxes show illustrative numbers (e.g. "Accuracy: 60% → 68%") — remove all concrete numbers from Chapter 4 version; those belong in Chapter 5. |

---

#### Algorithms

> **Format:** Use `\begin{algorithm}` / `\end{algorithm}` with `\begin{algorithmic}[1]` for line numbers. Import `algorithmicx` and `algpseudocode` packages. Number algorithms sequentially: Algorithm 4.1, 4.2, etc.

> **Source rule:** For each algorithm, READ the corresponding `caem/` Python file before writing the pseudocode. The plan may differ — the code is authoritative.

| # | Element | What it must show | Source file | Satisfies | Known plan discrepancies |
|---|---|---|---|---|---|
| ALG-C4-01 | **Algorithm 4.1: CAEM Query Processing (Master Pipeline)** | Top-level loop covering Stages 1–7. Inputs: query q. Shows: u_pre computation → routing decision (OR-condition first, then score) → branch to Tier 1/2/3 logic → Stage 4a gate → Stage 5 (for Tier 2/3 only) → memory storage. Output: answer + storage decision. | **No single plan algorithm covers the full pipeline.** Read `caem/pipeline.py` and use the Ch1 TikZ stage numbering as the structural skeleton. The plan's Algorithm at line 3100 (storage) and line 2627 (verification) can be referenced for those sub-steps. | C4-01, C4-04, C4-10b, IMPL-01 | Tier 1 path must explicitly skip Stage 5 — add a comment to the pseudocode. OR-condition check must precede routing score computation as a distinct step. |
| ALG-C4-02 | **Algorithm 4.2: Three-Tier Routing** | Inputs: u_pre, memory store M, query embedding e_q. Step 1: if u_pre < 0.60 → return Tier 3. Step 2: s ← FAISS_search(M, e_q). Step 3: û_stored ← M[nearest].u_stored. Step 4: score ← 0.7·s + 0.3·û_stored. Step 5: if score ≥ 0.90 → Tier 1; elif score ≥ 0.65 → Tier 2; else → Tier 3. | **Unified plan line 2627** (Dataset-Adaptive Multi-Layer Verification) has a `\begin{algorithmic}[1]` block with routing-style logic — use its LaTeX `\If / \ElsIf / \State` formatting as a template. Read `caem/router.py` for the actual threshold values and structure. | C4-04, C4-10b, C4-20 | OR-condition must be Step 1 (before FAISS call) — never merged into the score formula. The plan algorithm at 2627 shows per-dataset routing logic — CAEM's routing is confidence-based (not dataset-based at this stage), so do not copy that structure directly. |
| ALG-C4-03 | **Algorithm 4.3: Multi-Signal Verification Pipeline (Stage 5)** | Inputs: query q, generated answer a, context c, ground-truth label y (benchmark mode). Outputs: (accept: bool, û_stored: float). Steps: (1) p_entail ← NLI_score(q, a, y); (2) SC samples ← [generate(q) for _ in range(10)]; s_avg ← mean_cosine_sim(SC_samples); (3) SE samples ← [generate(q, temp=T_se) for _ in range(10)]; h_norm ← semantic_entropy(SE_samples); (4) VE1 check: if p_entail is NEUTRAL AND u_consistency > 0.90 → reject; (5) VE2 check (TruthfulQA only): if misconception_match(a) → reject; (6) û_stored ← 0.5·p_entail + 0.3·s_avg + 0.2·h_norm; (7) accept ← û_stored ≥ storage_threshold. | **PRIMARY SOURCE: Unified plan line 1832** (Semantic Entropy Computation) and **line 2321** (Self-Consistency Verification) — use their step formatting. Also **line 2441** (SE Verification Layer 3) — very detailed SE clustering pseudocode, reusable verbatim after field-name fixes. Read `caem/verifier.py` for the combined verification logic. | C4-08, BM-03, BM-04, BM-05, GEN-05 | Plan line 2321 uses `u_SC > 0.85` as the threshold — VE1 uses `> 0.90` (the stricter threshold for neutral NLI escalation, C4-02/BM-04). Plan uses `phi_SC`, `phi_SE` notation — replace with `s_avg`, `h_norm`. Plan shows N=3 chains for SC — implementation uses N=10. VE2 is TruthfulQA only (plan may show as general). |
| ALG-C4-04 | **Algorithm 4.4: Self-Improvement Loop (Stage 8)** | Inputs: memory M, model θ, previous weights θ_prev, λ=0.01. For each cycle c ∈ {0,1,2,3}: (1) D_train ← {(q, reasoning_chain) : episode ∈ M, episode.u_stored ≥ threshold}; (2) for each batch in D_train: L ← CE(θ, q, reasoning_chain) + (λ/2)·Σ(θᵢ − θ_prev,ᵢ)²; update θ; (3) check forgetting score; if score < 0.07 → abort, restore θ_prev; (4) θ_prev ← θ.copy(). Note: θ_prev computed on CPU in RTX 3060 config. | **Unified plan line 3979** (Catastrophic Forgetting Mitigation Protocol) has the L2 regularisation and forgetting check structure — use as base. Also plan **line 3910** (Mixed Training Data Construction). Read `caem/trainer.py` `_l2_penalty()` and `train_cycle()` for the actual implementation. | C4-16, C4-19, C4-22, GEN-11 | **CRITICAL discrepancies in plan line 3979:** (1) Plan uses `θ_base` (original pre-trained weights) — implementation uses `θ_prev` (previous cycle's weights). These are completely different. (2) Plan includes 10% general corpus mixing (`D_general`) — verify whether this is in the implementation. (3) Training target: plan has `log P_θ(c, a | q)` — correct, this is reasoning_chain (C4-22). (4) Regularisation is L2 uniform — write explicitly as `||θ − θ_prev||²` not EWC. |
| ALG-C4-05 | **Algorithm 4.5: Retroactive Re-verification** | Inputs: memory M, current model θ. At end of each cycle (before next cycle's fine-tuning): For each episode e ∈ M: (1) a_new ← generate(θ, e.question); (2) new_û_stored ← Stage5_verify(e.question, a_new, e.answer); (3) if new_û_stored < eviction_threshold → remove e from M; else → e.u_stored ← new_û_stored; e.retroverified ← True. Output: pruned, freshness-updated memory M. Note: this amortises verification cost over one batch per cycle — NOT per-query. | **Unified plan line 3100** (Strategic Storage with Novelty Checking) covers the storage-side logic. The retroactive re-verification pass (re-running the whole memory) is described in the plan's text but may not have a dedicated algorithm block. Read `caem/retroverifier.py` for the actual implementation. The plan's storage algorithm at line 3100 can provide the LaTeX formatting skeleton. | IMPL-01 (freshness), C5-07, C4-15 | Plan line 3100 uses notation `e.phi`, `e.r`, `e.s`, `e.u` — update to match actual schema fields (`u_stored`, `success_rate`, `retrieval_count`, `retroverified`). Plan uses `C_max = 20000` — correct, keep it. Caption must state: "This is what gives Tier 1 its freshness guarantee — retroactive re-verification amortises the verification cost across cycles, not per-query." |

---

#### Equations

> **Format:** Use `\begin{equation}` with `\label{eq:name}` for each. Cross-reference by label throughout the text. Number sequentially: (4.1), (4.2), etc.

| # | Element | Formula | Section | Satisfies |
|---|---|---|---|---|
| EQN-C4-01 | **Eq 4.1: Pre-Routing Confidence** | u_pre = 0.60·C_conv + 0.40·u_MC_pre | §4.3 | C4-03b, C4-10 |
| EQN-C4-02 | **Eq 4.2: Routing Score** | routing_score = 0.7·s(q, e) + 0.3·û_stored(e) | §4.2 | C4-10b, C4-20 |
| EQN-C4-03 | **Eq 4.3: Post-Generation Composite Confidence (û)** | û = w₁·u_token + w₂·u_dropout + w₃·u_consistency + w₄·u_entropy where w₁=w₂=w₃=w₄=0.25 initially, calibrated to ≈0.20/0.20/0.20/0.40 | §4.4 | C4-06, C4-07, C4-21 |
| EQN-C4-04 | **Eq 4.4: Stored Confidence (û_stored)** | û_stored = 0.5·p_entail + 0.3·s_avg + 0.2·h_norm | §4.5 | C4-08 |
| EQN-C4-05 | **Eq 4.5: Fine-tuning Loss with L2 Penalty** | L(θ) = L_CE(θ) + (λ/2)·‖θ − θ_prev‖² where λ=0.01 [DES] | §4.7 | C4-19, GEN-11 |
| EQN-C4-06 | **Eq 4.6: Temperature Scaling** | p_calibrated(y \| x) = softmax(z(x)/T) where T is fitted by minimising ECE on the 500-sample calibration set | §4.8 | IMPL-03 |
| EQN-C4-07 | **Eq 4.7: Expected Calibration Error (ECE)** | ECE = Σ_{b} (|B_b|/N) · |acc(B_b) − conf(B_b)| over M bins (M=10) | §4.8 | IMPL-03, C5-11 |
| EQN-C4-08 | **Eq 4.8: Data Purity Theorem** | P = pα / (pα + (1−p)(1−α)) | §4.9 Theorem 4.1 | C4-17, PUB-02 |
| EQN-C4-09 | **Eq 4.9: Cosine Similarity (routing)** | sim(q, e) = (e_q · e_episode) / (‖e_q‖ · ‖e_episode‖) | §4.2 (reference to Ch2 definition) | C4-02 |

---

#### Theorems and Proofs

> **Format:** Use `\begin{theorem}[Name]\label{thm:name}`, `\end{theorem}` and `\begin{proof}`, `\end{proof}`. All three theorems must appear in §4.9. Chapter 5 reports the empirical confirmation of each — cross-reference by label.

| # | Element | Statement | Proof required? | Satisfies |
|---|---|---|---|---|
| THM-C4-01 | **Theorem 4.1: Data Purity Theorem** | Let p = P(model correct), α = P(verifier accepts \| correct). Then P(episode correct \| accepted) = pα / (pα + (1−p)(1−α)). Moreover, P > p if and only if p > (1−α). | **PRIMARY SOURCE: Unified plan line 3508** (`\begin{theorem}[Data Purity Formula]`) — the content and formula are directly reusable. HOWEVER: plan uses a custom `proofbox` environment (not standard LaTeX). For the report, replace with `\begin{proof}...\end{proof}`. The proof logic in the `proofbox` (TP = N·p·α, FP = N·(1−p)·(1−α), P = TP/(TP+FP)) is mathematically correct and reusable verbatim. | C4-17, PUB-02, TH-01, TH-02 |
| THM-C4-02 | **Theorem 4.2: Coupled Improvement Recurrence (Monotonicity)** | Let p_n and α_n denote model accuracy and verifier accuracy at cycle n. If p_n+1 > p_n and α_n+1 ≥ α_n, then P_n+1 > P_n — memory purity is strictly increasing per cycle, bounded above by 1. | **Unified plan line 4029** (`\begin{theorem}[Practical Equilibrium]`) and **line 4185** (`\begin{theorem}[Coupled System Convergence]`) — both use standard `\begin{theorem}` environments, reusable directly. The coupled recurrence text and the Banach fixed-point mention are at line 4185. Proof sketch (not full proof) is acceptable for thesis. | C4-18, TH-01, C6-02 |
| THM-C4-03 | **Theorem 4.3: Convergence (Diminishing Returns)** | The improvement magnitude Δ_n = p_n+1 − p_n decreases monotonically for n ≥ 1, converging to 0 as the system approaches the fixed point. Convergence guaranteed because the coupled (p, α) system is bounded — capacity ceiling and maximum verifier accuracy set a practical ceiling. Empirical signal: diminishing Δ between consecutive cycles. | **Unified plan line 4185** (`\begin{theorem}[Coupled System Convergence]`) — cites Banach fixed-point theorem and states conditions formally. Use this text and adapt it. The `\begin{theorem}` environment is already correct format. | C4-23, TH-01, C6-02 |

---

### Chapter 5 — Figures and Tables (data-dependent)

> All figures and tables in Chapter 5 use actual values from `outputs/` files. No projected or illustrative values. See Phase 5 of `NEXT_SESSION_PLAN.md` for the output file → section mapping.

#### Figures

| # | Element | What it must show | Source | Satisfies | Notes |
|---|---|---|---|---|---|
| FIG-C5-01 | **Figure 5.1: Accuracy by Cycle (all benchmarks)** | Grouped bar chart or line chart. X-axis: Cycles 0–3. Y-axis: Accuracy (EM or ROUGE-L). One line/bar group per benchmark. Shows monotone improvement trend. | `outputs/full_experiment/experiment_summary.csv` | C5-04, C5-03 (Theory 2 visual) | If any benchmark shows non-monotone cycle, annotate and explain in caption. |
| FIG-C5-02 | **Figure 5.2: Tier Routing Distribution Across Cycles** | Stacked bar chart. X-axis: Cycles 0–3. Y-axis: Fraction of queries. Three stacks: Tier 1 (growing), Tier 2 (stable), Tier 3 (shrinking). Key narrative: growing Tier 1 fraction proves memory accumulation is working. | `outputs/full_experiment/experiment_summary.csv` (tier_1_frac, tier_2_frac, tier_3_frac columns) | C5-02, C5-06 | Report per-benchmark FEVER Tier 1 accuracy separately (GEN-06 near-miss risk). |
| FIG-C5-03 | **Figure 5.3: MMLU Retention Across Cycles** | Line chart. X-axis: Cycles 0–10. Y-axis: MMLU accuracy (%). Horizontal dashed line at 93% (retention target). Expected: stable flat line near or above 93%. **Primary data source (per-cycle): `outputs/experiment_summary.csv` column `mmlu_retention_pct`** — auto-computed since EXP-MMLU-FIX Session 37. Secondary source (ablation conditions): `outputs/ablation_results/ablation_summary.json`. Note: Cycle 0 will show NaN/blank (no fine-tuning step) — start the line chart from Cycle 1. **Session 35 note:** old entry said "Cycles 0–3" — updated to 10-cycle plan. | C5-07, C5-MMLU-01 | If any cycle drops below 93%: this is catastrophic forgetting — do NOT hide it, flag immediately. |

#### Tables

| # | Element | What it must show | Source | Satisfies |
|---|---|---|---|---|
| TAB-C5-01 | **Table 5.1: Main Results — CAEM vs All Baselines** | Rows: all methods (Zero-shot, CoT, RAG, SC, Vanilla FT, Memory-only, CAEM Cycle 0, CAEM Cycle 10). Columns: all 6 benchmarks (FEVER / TriviaQA / NQ / TruthfulQA / StrategyQA / ARC-Challenge / Average). **Session 38 update:** `run_ablation.py` `BENCHMARK_ORDER` now includes TriviaQA and NQ — baselines ARE evaluated on all 6 benchmarks (old note saying they used only 4 benchmarks is stale). **HOWEVER**, for the thesis TABLE presentation, consider marking TriviaQA/NQ baseline columns with a ‡ footnote: "‡ CAEM has a training advantage on TriviaQA and Natural Questions (these are in-domain training benchmarks); baseline scores are provided for completeness but the comparison is not equivalent to the transfer benchmarks." Alternatively, split the table clearly into In-Domain and Out-of-Domain column groups (matching TAB-C5-05 style) so readers see the distinction at a glance. Include † for statistically significant improvements (McNemar's p < 0.05). Add a sub-row or footnote for GPT-3.5 / Self-RAG literature numbers on StrategyQA only (PUB-01c). **Session 38 scorer fix (BM-10):** all baseline scores are now computed with alias-aware EM/F1 for TriviaQA/NQ and label extraction for ARC — results are now directly comparable to CAEM harness scores. | `outputs/eval/*.json` + `ablation_results/ablation_summary.json` | C5-04, C5-05, C5-12, GEN-12, BM-10 |
| TAB-C5-02 | **Table 5.2: Mechanism Evidence Table** | Rows: Cycles 0, 1, 3, 5, 7, 9, 10 (show all 10 but can compact to key milestones to save space). Columns: Hallucination Rate (%), Tier 1 Fraction (%), Tier 3 Fraction (%), MMLU Retention (%), Mean û_stored. Shows all three mechanisms active simultaneously across 10 cycles. Key trend to highlight: Tier 1 fraction grows toward equilibrium at C7–C9 while MMLU retention stays ≥93%. **Session 35 note:** old entry said "Cycles 0–3" (3-cycle plan). Updated to 10-cycle plan. | `outputs/experiment_summary.csv` | C5-02 |
| TAB-C5-03 | **Table 5.3: Theory Validation — Purity Theorem** | Rows: Cycles 0, 3, 7, 10 (representative cycles across the 10-cycle run — show early, mid, equilibrium, final). Columns: p (model accuracy), α (verifier accuracy), P_theoretical = pα/(pα+(1−p)(1−α)), P_observed (measured from memory). Show P_obs ≈ P_theoretical at each cycle to confirm theorem holds throughout. **Session 35 note:** old entry said "Cycles 0–3". Updated to show representative cycles from the 10-cycle run. | `outputs/purity_validation/theory_validation.json` | C5-03, TH-03 |
| TAB-C5-04 | **Table 5.4: Convergence — Δ per Cycle** | Rows: all consecutive cycle gaps C0→C1 through C9→C10 (10 rows). Columns: Δ_FEVER / Δ_TriviaQA / Δ_NQ / Δ_TruthfulQA / Δ_StrategyQA / Δ_ARC / Δ_avg / Convergence Check (✅ or ❌). If Δ(n+1) < Δ(n) for Δ_avg → ✅. Expected pattern: large Δ in C0→C3, small Δ in C4→C6, near-zero Δ in C7→C10 (equilibrium). If any Δ is negative (accuracy regressed), flag it explicitly — do not hide it. **Session 35 note:** old entry said "Rows: C0→C1, C1→C2, C2→C3" (3-cycle plan) and "Columns: Δ_HotpotQA/Δ_TruthfulQA/Δ_FEVER/Δ_StrategyQA". Updated to 10-cycle gaps and all 6 current benchmarks. | `outputs/eval/*.json` | TH-04, C4-23, C6-02 |
| TAB-C5-05 | **Table 5.5: Ablation Results** | 7 ablation variants (AB1–AB7) grouped by mechanism, evaluated on **all 6 benchmarks** — FEVER / TriviaQA / NQ (in-domain) + TruthfulQA / StrategyQA / ARC-Challenge (out-of-domain). Session 36 updated `run_ablation.py` to span the full 6-benchmark suite; the table structure must therefore show two column groups: *In-Domain* (FEVER / TriviaQA / NQ) and *Out-of-Domain* (TruthfulQA / StrategyQA / ARC) + MMLU Retention. This lets readers see both the architectural contribution and the generalization effect of each removed component separately. Groups: **(1) Memory quality:** AB2 (no verification — store at u_stored=0.5), AB5 (NLI-only verification), AB4 (no retroactive re-verification — needs separate checkpoint); **(2) Routing:** AB1 (no memory — EmptyMemoryStore, all-Tier-3), AB6 (no OR-condition — safety veto disabled); **(3) Training/SIL:** A4 Vanilla FT (unverified checkpoint), A5 Memory-only (Cycle 0 weights); **(4) Signal:** AB7 (no semantic entropy — NLI+SC only), AB3 (no CoT supervision — needs separate checkpoint). Mark AB3 and AB4 with ‡ footnote: "‡ Requires separate training run; omitted if checkpoint unavailable." Report EM per benchmark + MMLU retention per row. **Session 35 note:** old naming (A1/A-NLI/A-verify/A2/A3) replaced with current code names (AB1–AB7). Count corrected from 9 to 7 AB variants. **Session 36 note:** benchmark scope expanded from 4 to 6 to enable clean variable isolation. | `outputs/ablation_results/ablation_summary.json` | C5-09 |
| TAB-C5-06 | **Table 5.6: Statistical Significance** | For each CAEM Cycle 3 vs baseline pair: χ² / p-value / significant (Y/N). Use McNemar's test from `eval/metrics.py`. | Per-question EM arrays from `eval/{bm}_cycle{n}.json` | C5-05, C5-12, GEN-12 |
| TAB-C5-07 | **Table 5.7: Calibration Results** | Rows: pre-calibrati