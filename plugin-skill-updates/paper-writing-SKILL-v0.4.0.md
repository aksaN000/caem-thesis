---
name: paper-writing
description: >
  Use this skill when Aksan wants to draft, write, edit, or improve any written
  section of his CAEM thesis or paper. Triggers include: "write the abstract",
  "draft the introduction", "write the related work", "write the methodology
  section", "write the experiments section", "draft the conclusion", "edit this
  paragraph", "improve the writing", "make this clearer", "is this well-argued",
  "does this flow well", "too verbose", "rewrite this section", "write the
  limitations section", "academic tone", "check the argument", "help me write",
  "draft this", or any request to produce or improve thesis/paper text.
metadata:
  version: "0.4.0"
  author: "Aksan Gony Alif"
---

# Paper Writing

You are Aksan's thesis writing collaborator. You know the full CAEM system deeply and can write accurate, technically precise, and academically rigorous prose about it. See `references/thesis-structure.md` for the chapter outline and key claims.

## Template constraint — 6 chapters only

The BracU CSE400 template is fixed at exactly 6 chapters. Do NOT add a 7th chapter. Discussion & Limitations folds into Chapter 6 (Conclusion).

| Chapter | Title | Template file | Status |
|---|---|---|---|
| Ch1 | Introduction | chapter_1.tex | ✅ Submitted |
| Ch2 | Literature Review | chapter_2.tex | ✅ Submitted |
| Ch3 | Requirements Analysis | chapter_3.tex | ⚠️ Stub |
| Ch4 | Methodology | chapter_5.tex | ❌ Not written |
| Ch5 | Results & Analysis | chapter_6.tex | ❌ Not written |
| Ch6 | Conclusion | chapter_9.tex | ❌ Not written |

> ⚠️ Template file numbering does not match chapter numbers. Do NOT rename files.

---

## Before drafting any section

Read these files first — they contain corrections and framing decisions from prior sessions. Writing without them risks repeating known errors.

### Reference Stack (what to have open and why)

| # | File | Purpose | Use for | Do NOT use for |
|---|---|---|---|---|
| 1 | `writing-suggestions.md` | Live corrections, framing rules, visual/formal elements inventory | All writing — filter by chapter prefix | — |
| 2 | `pre thesis 1 report/chapters/chapter_1.tex`, `chapter_2.tex` | **VALIDATED citation source** — every `\cite{}` key here is confirmed in references.bib | Finding `\cite{}` keys for ALL new chapters | Anything other than citation keys |
| 3 | `caem-unified-plan-v3.tex` | Algorithm pseudocode structure, TikZ diagram layouts, theorem statements | Visual/formal element templates (adapt, verify against implementation) | Citation keys — plan keys may differ from validated keys |
| 4 | `hyperparameter-reference.md` | Three-category hyperparameter taxonomy | Any section mentioning numbers, weights, thresholds, or calibration | — |

**Quick lookup rules:**
- Citation keys → pre-thesis report chapters ONLY (`chapter_1.tex` / `chapter_2.tex`)
- Algorithm logic → `caem/` Python source (authoritative); plan provides formatting template only
- Hyperparameter categories → `hyperparameter-reference.md`
- Visual element source lines + discrepancy warnings → `writing-suggestions.md` Visual & Formal Elements section

**Critical on citations:** When writing any section with `\cite{}` commands, get the key from the pre-thesis report chapters ONLY. The unified plan was written across many sessions and citation keys may be stale or differ from what is in `references.bib`. The submitted chapters (Ch1, Ch2) went through a validation process that ensures every key exists in the bib file.

1. **`writing-suggestions.md`** — Use the Glob tool to find it: `**/for cowork caem/writing-suggestions.md`. Filter for the chapter you are currently writing (C1-*, C2-*, C3-*, C4-*, C5-*, C6-*, GEN-*, BM-*). Every OPEN entry in that chapter's section must be addressed in the draft. Mark entries DONE after applying them. Also read the **Visual & Formal Elements** section — it lists every figure, algorithm, equation, and theorem with exact source line numbers in the unified plan and CRITICAL discrepancy warnings for each algorithm.

2. **`hyperparameter-reference.md`** — Use the Glob tool to find it: `**/for cowork caem/hyperparameter-reference.md`. Read the three-category table before writing any section that mentions hyperparameters, confidence weights, or calibration. Category 3 values (û signal weights, temperature T) must be described as projected/calibrated, not as fixed design values.

---

## Writing philosophy

- **Precision over flair:** Every claim must be supportable. Avoid hedging claims that should be strong, and avoid strong claims that aren't supported yet.
- **Active voice by default:** Prefer "We evaluate CAEM on four benchmarks" over "CAEM was evaluated..."
- **Structure before prose:** For any section, confirm the argument structure before writing. A well-structured section writes itself.
- **Say the hard thing early:** Lead with the claim, then support it. Don't bury the main point.
- **Projected vs. actual:** All numerical values in the plan are PROJECTED. Do not write them as confirmed results. Use "we project", "we expect", or "as reported in Chapter 5" depending on which chapter you are writing.

---

## LaTeX heading hierarchy

**Template rule:** Each chapter uses `\section{}` (4–6 per chapter) containing `\subsection{}`. Do not add more than 6 `\section{}` per chapter. All structural labels (§4.1–4.9 etc.) are planning labels — they map to `\subsection{}`, not all to `\section{}`.

**Chapter 4 hierarchy (5 sections):**
```latex
\chapter{Methodology}  % file: chapter_5.tex
  \section{System Architecture Overview}
    \subsection{Pipeline Overview}
    \subsection{Design Rationale}          % BM-02 benchmark justification here
  \section{Confidence Estimation and Routing}
    \subsection{Pre-Routing Confidence (u\_pre)}
    \subsection{Three-Tier Adaptive Routing}
    \subsection{Post-Generation Confidence (\hat{u})}
  \section{Verification Pipeline}
    \subsection{Multi-Signal Verification}
    \subsection{Dataset-Adaptive Layers}   % BM-03/04 here
    \subsection{Episode Storage and \hat{u}\_stored}
  \section{Self-Improvement Loop}
    \subsection{Fine-Tuning Objective}
    \subsection{Retroactive Re-Verification}
    \subsection{Retrieval Feedback}
  \section{Theoretical Framework}
    \subsection{Data Purity Theorem}
    \subsection{Coupled Improvement Recurrence}
    \subsection{Convergence Analysis}
```

**Chapter 5 hierarchy (6 sections):**
```latex
\chapter{Results and Analysis}  % file: chapter_6.tex
  \section{Experimental Setup}          % BM-01/05/06 here
    \subsection{Benchmarks and Metrics}
    \subsection{Baselines}
    \subsection{Implementation Details}
  \section{Main Results}
    \subsection{Primary Accuracy Table}
    \subsection{Hallucination Reduction}
  \section{Mechanism Analysis}
    \subsection{Five-Mechanism Evidence Table}
    \subsection{Routing Distribution}
  \section{Ablation Study}
    \subsection{Per-Mechanism Ablations}
    \subsection{Signal Weight Ablation}
  \section{Theory Validation}
    \subsection{Purity Theorem Validation}
    \subsection{Monotonicity and Convergence}
  \section{Latency and Error Analysis}
    \subsection{Per-Tier Latency}
    \subsection{Failure Mode Analysis}
```

---

## Per-section guidance

### Abstract
Write last (or as a working draft). Must cover: problem (hallucination), approach (CAEM system summary), method (8-stage pipeline, 3-tier routing, self-improvement), key result (X% hallucination reduction across 4 benchmarks), and significance (single-GPU, no external API dependency).

### Introduction
Structure: (1) Problem motivation — why hallucination matters, (2) Gap — why RAG/fine-tuning alone isn't enough, (3) CAEM overview — what it is and why it's novel, (4) Contributions — bulleted, specific, measurable, (5) Paper outline.

Frame CAEM as an architectural intervention, not a post-hoc detection system. See C4-01 in writing-suggestions.md. Lead sentence suggestion: "CAEM is not a post-hoc hallucination detector — it is an architectural system that prevents unreliable knowledge from accumulating and uses verified knowledge to progressively improve generation quality."

### Related Work (Chapter 2)
Organize by theme, not chronology. Key themes:
- Hallucination in LLMs (Ji et al. 2023, Maynez et al. 2020)
- Retrieval-Augmented Generation (Lewis 2020, REALM, RETRO)
- Episodic memory in neural networks
- Uncertainty quantification (MC Dropout — Gal 2016; self-consistency — Wang 2023; semantic entropy — Farquhar 2024; C_conv — Nandakishor 2025)
- Self-improving systems (ReST — Gulcehre 2023; STaR — Zelikman 2022)
- Verification and NLI
- Catastrophic forgetting (EWC — Kirkpatrick 2017, cited as motivation for L2 choice; CAEM does NOT implement EWC itself — Fisher matrix too expensive at 780M params)
- Chain-of-thought faithfulness (Korbak et al. 2025)

Each theme: end with a crisp sentence positioning CAEM relative to that body of work. See the **Chapter 4 — Benchmark Design Rationale** section in writing-suggestions.md (BM-02/03/04) for benchmark justification framing — these entries belong in Chapter 4 §Architecture Overview and §Verification Pipeline, not in the Literature Review.

### Requirements Analysis (Chapter 3)
Currently a stub (9 lines). Must define: functional requirements (episodic memory, routing, verification, self-improvement), non-functional requirements (single GPU, inference latency targets), and evaluation criteria. Keep concise — this is a requirements chapter, not methodology. Benchmark selection rationale does NOT go here; it goes in Chapter 4 §System Architecture Overview (BM-02) and §Verification Pipeline (BM-03/04).

### Methodology (Chapter 4)
Follow the 8-stage pipeline execution order — not conceptual clusters. For each stage: what it does, why it's designed that way, formal definition. Key writing constraints:

- u_pre and routing_score are TWO SEPARATE mechanisms. The OR-condition (u_pre < 0.60) is a hard veto evaluated BEFORE the routing formula. They must not be merged into one expression. See C4-10b in writing-suggestions.md.
- û signal weights: state initial weights as 0.25 equal; post-calibration values are projected (0.20/0.20/0.20/0.40); actual calibrated values reported in Chapter 5. See C4-07.
- û_stored is NOT derived from û. It comes from the verification pipeline. State this explicitly. See C4-08.
- VE2 (TruthfulQA misconception check) is curated list matching against ~38 categories from Lin et al. 2021 — NOT a learned classifier. State explicitly. See GEN-05.
- NLI verification dependency on ground truth: scope to NLI component only. SC and SE are reference-free. See GEN-04.
- Training target for fine-tuning is `reasoning_chain` (the full chain-of-thought output), not just `answer`. See C4-22.
- Convergence claim: CAEM's diminishing gains across cycles are an empirical observation consistent with theoretical prediction — do NOT claim "infinite cycles → zero hallucination" or global convergence. Practical equilibrium, not zero-error limit. See C4-23.
- Check writing-suggestions.md **Visual & Formal Elements** section for all figures (FIG-C4-*), algorithms (ALG-C4-*), equations (EQN-C4-*), and theorems (THM-C4-*) that belong in this chapter — including exact source line numbers in the unified plan and CRITICAL discrepancy warnings for each algorithm (field names, training targets, θ_base vs θ_prev, etc.).

Key equations (all corrected — use these exactly):
```
u_pre = 0.6·u_token + 0.4·(1/(1+C_conv))
C_conv = Var(h_{1:L/2}) / (Var(h_{L/2:L}) + ε),  ε = 1e-8
û = 0.20·u_token + 0.20·u_dropout + 0.20·u_consistency + 0.40·(1-u_entropy)  [projected post-calibration]
routing_score = 0.7·s + 0.3·û_stored  [only computed if OR-condition did not fire]
û_stored = 0.50·P(ENTAIL) + 0.30·s_avg + 0.20·(1-H_norm)
P = pα / (pα + (1-p)(1-α)),  condition: p > (1-α),  guarantee: P > p
L = L_CE(q, c, reasoning_chain) + λ_reg·||θ - θ_prev||²  [L2, NOT EWC]
```

### Experimental Setup (Chapter 5, §5.1)
Structure within §5.1 Experimental Setup:
1. Baseline pilot validation — run FIRST; validates p > 0.5 on all benchmarks; contingency plan if fails
2. Data splits — state calibration set (500 samples for temperature scaling) and purity validation set (500 samples for theorem validation) are non-overlapping; use the allocation table from the plan
3. Cold start seeding — **447 total verified episodes seeded before Cycle 1** (not 200–500 per benchmark)
4. Baselines — all 6 (Zero-shot, CoT, RAG, Self-consistency, Vanilla FT, Memory-only)
5. Evaluation protocol — circular evaluation decoupling; benchmark ground-truth labels used for final evaluation only

Key statement to include: "The 500-sample calibration set (for temperature scaling) and the 500-sample purity validation set (for theorem validation) are drawn from non-overlapping portions of the official benchmark validation splits. This separation prevents circular validation." See C5-01.

Benchmark selection rationale (why these 4 benchmarks): See **Chapter 5 — Benchmark Evaluation Setup** section in writing-suggestions.md (BM-01/05/06). BM-01 covers why this combination covers complementary reasoning types. BM-05 and BM-06 cover metric justification and expected benchmark-specific patterns. These belong in §5.1, not in Chapter 4.

### Results & Analysis (Chapter 5, §5.2–5.6)
**This chapter has a single narrative thread**: every section answers one of — (1) Does CAEM beat baselines? (2) Does each mechanism contribute? (3) Do theoretical predictions hold? See C5-04.

Structure:
1. **Main Results (§5.2)** — primary accuracy table, 4 benchmarks × 6 baselines × 3 cycles; hallucination reduction figure
2. **Mechanism Analysis (§5.3)** — Five-mechanism evidence table (sec. mechanism-evidence-table in plan) — the most important diagnostic table; proves all 3 mechanisms working simultaneously. Columns: Cycle | Halluc. Reduction | Tier 1 Fraction | Tier 3 Fraction | MMLU Retention | Mean û_stored. Also report Tier 1 accuracy per benchmark separately (lower FEVER = near-miss issue). See C5-02.
3. **Ablation Study (§5.4)** — run all required ablations including A1 (no retroactive re-verification), A2 (all Tier 3), A3 (no OR-condition). A1 proves memory quality; A2 proves routing efficiency; A3 proves safety-first design.
4. **Theory Validation (§5.5)** — three structured experiments, NOT side notes. See C5-03 and C6-02:
   - Theory 1 (Purity Theorem): five-step protocol; table of p/α/P_theory/P_obs per cycle; state condition p > (1−α). Theorem guarantees P > p (NOT P > α).
   - Theory 2 (Monotonicity): verify p_0 < p_1 < p_2 < p_3 and α_0 < α_1 < α_2 < α_3
   - Theory 3 (Convergence): verify Δ_3 < Δ_2 ≤ Δ_1; state explicitly that convergence is a theoretical prediction validated empirically, not an assumption
5. **Latency and Error Analysis (§5.6)** — Tier 1 < 400ms; Tier 2 < 2.5s; Tier 3 < 6s; report mean and 95th percentile; qualitative failure examples; categorize by failure type

Check writing-suggestions.md Visual & Formal Elements section for all figures (FIG-C5-*) and tables (TAB-C5-*) that belong in Chapter 5.

**Closing sentence for Chapter 5:** "Together, these results confirm that CAEM's closed-loop architecture delivers consistent, measurable, and theoretically grounded hallucination reduction across four qualitatively distinct benchmarks."

### Conclusion (Chapter 6)
Structure: (1) Restate problem, (2) CAEM's approach in one paragraph, (3) Key empirical findings, (4) Theoretical contributions — state explicitly what each theorem provides (see C6-01), (5) Limitations honestly stated (see below), (6) Future directions.

Key statement for theoretical contributions: "The data purity theorem provides a formal guarantee that CAEM's verification pipeline produces memory of strictly higher purity than unfiltered generation, under the verifiable condition p > (1−α). The coupled improvement recurrence provides a theoretical basis for the observed monotone accuracy gains across cycles. Together, these theoretical results distinguish CAEM from heuristic self-training approaches." See C6-01.

Known limitations to address honestly (be genuinely self-critical — this demonstrates intellectual maturity):
- Data purity theorem condition p > (1-α) may not hold for near-random benchmarks
- FAISS capacity capped at 20,000
- L2 regularization is uniform, not Fisher-weighted (may under-protect important parameters vs. EWC)
- C_conv adaptation to Flan-T5 encoder not validated against original decoder-only setting
- Outcome-based verification (Korbak et al. 2025 concern — reasoning chain quality not directly verified)
- Single GPU / 780M model — scalability to larger models is future work
- NLI verification requires ground-truth labels (SC and SE do not — future: Wikipedia passages as NLI reference)
- Inference mode memory growth limited without ground truth
- Scaling guide (batch size, passage index, θ_prev GPU placement) will be applied on lab device — lab-run results may require small methodology updates

---

## Common errors to avoid

- Writing "EWC" anywhere — CAEM uses L2 regularization regularizing toward θ_prev, NOT EWC Fisher-weighted regularization
- Writing P > α — the theorem guarantees P > p (base generation accuracy), not P > α
- Writing û weights as fixed (0.20/0.20/0.20/0.40) — they are projected post-calibration values; initial weights are 0.25 equal
- Describing CAEM as "detecting" hallucinations — it prevents and reduces them architecturally
- Conflating u_pre and routing_score — they are separate mechanisms evaluated at different stages (OR-condition first, then routing formula)
- Saying the whole system requires ground truth — only NLI verification does; SC and SE are reference-free
- Describing VE2 as "learned" — it is curated list matching against ~38 TruthfulQA misconception categories
- Using `u_SC` in equations or text — the implementation field is `u_consistency`
- Using `h_entropy_norm` or `H_sem` — the implementation field is `u_entropy`
- Writing training target as `answer` alone — it is `reasoning_chain` (the full chain-of-thought output)
- Claiming convergence to zero hallucination — CAEM shows diminishing gains empirically; theoretical convergence is to a practical equilibrium, not to zero error
- Writing cold-start as "200–500 episodes per benchmark" — actual seeded total is **447 episodes across all benchmarks**
- Using citation keys from the unified plan — always get `\cite{}` keys from pre-thesis report chapters (`chapter_1.tex` / `chapter_2.tex`)
- Referencing a 7th or 8th chapter — the template has exactly 6 chapters; Discussion & Limitations folds into Chapter 6
- Writing algorithms or equations without checking Visual & Formal Elements section discrepancy warnings first
