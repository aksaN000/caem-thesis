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

## Thesis Template Structure (BracU CSE400 — FINAL)

**Template:** `FINAL YEAR THESIS Template_CSE400_Fall 2024 ONWARDS/`
**Main file:** `main.tex` (do not rename; references all chapters by path)

### Chapter sequence (write in this order)

| Chapter No. | Title | Template file | Status |
|---|---|---|---|
| Ch 1 | Introduction | `chapters/chapter_1.tex` | ✅ SUBMITTED (Phase 1, 349 lines) |
| Ch 2 | Literature Review | `chapters/chapter_2.tex` | ✅ SUBMITTED (Phase 1, 229 lines) |
| Ch 3 | Requirements, Impacts and Constraints | `chapters/chapter_3.tex` | ⚠️ STUB ONLY (9 lines) |
| Ch 4 | Proposed Methodology | `chapters/chapter_5.tex` ⚠️ | ❌ NOT WRITTEN |
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

### Chapter contents overview

| Chapter | Primary purpose | Key sections |
|---|---|---|
| Ch 1 | Motivate the problem; state CAEM's contribution | Problem statement, research objectives, contributions, thesis organisation |
| Ch 2 | Survey hallucination, memory, verification, self-training prior work | Hallucination taxonomy, episodic memory, NLI/SC/SE, continual learning, ReST, RAG |
| Ch 3 | State functional + non-functional requirements; constraints as design drivers | Functional requirements (8 pipeline stages), NFRs (latency/VRAM/GPU budget), societal impact |
| Ch 4 | Present the full CAEM architecture with theory | Routing (§4.2), confidence signals (§4.3–4.4), verification (§4.5), episodic memory (§4.6), self-improvement loop (§4.7), calibration (§4.8), theory (§4.9) |
| Ch 5 | Report and interpret all experimental results | Calibration (§5.1), main results (§5.2), mechanism evidence table (§5.3), theory validation (§5.4), ablations (§5.5), baselines (§5.6), efficiency (§5.7), stability-plasticity (§5.8), error analysis (§5.9) |
| Ch 6 | State conclusions, limitations, future work | Findings (§6.1), limitations (§6.2), future work (§6.3), theoretical contribution (§6.4) |

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

## Chapter 3 / Chapter 4 — Benchmark Rationale

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| BM-01 | Benchmark selection must be justified by failure mode, not dataset popularity | For each benchmark, state explicitly: (1) what hallucination failure mode it tests, (2) why that failure mode matters for CAEM, (3) why the primary verification layer matches the failure mode. The complementary coverage argument is the thesis's strongest justification for using all four. | Session 2 | OPEN |
| BM-02 | TruthfulQA inverse scaling finding | Must explain WHY larger models score worse (they more faithfully reproduce internet misconceptions). This is the strongest motivation for architectural intervention — scaling alone makes the problem worse. | Session 2 | OPEN |
| BM-03 | VE1 + VE2 rules must be tied back to TruthfulQA's failure mode in the writing | When presenting VE1 (neutral NLI escalation) and VE2 (misconception flag), explicitly connect them to TruthfulQA's systematic confabulation failure mode. Reader should understand WHY these rules exist, not just what they do. | Session 2 | OPEN |
| BM-04 | SC threshold distinction (Chapter 2 vs VE1) | Chapter 2 states u_SC > 0.85 for "high consistency" pass/fail. VE1 uses u_SC > 0.90 for neutral NLI escalation. Write explicit note in Chapter 4: "The general SC verification gate uses threshold 0.85. The VE1 escalation condition uses a stricter 0.90 threshold because systematic confabulation produces near-perfect consistency — the higher bar is needed to distinguish genuine consensus from pathological overconfidence." | Session 2 | OPEN |
| BM-06 | FEVER evaluation uses a constrained enumerated prompt — state this explicitly | The FEVER evaluation prompt is constrained to enumerate valid labels: "Answer with one of: supports, refutes, not enough info. Claim: {claim}". This differs from the open-ended formulation in some prior work ("Is the following claim true, false, or uncertain?"). The constrained prompt prevents label extraction failures from free-form output, consistent with instruction-tuning evaluation practice (Wei et al. 2022, FLAN). Write: "FEVER claims are evaluated using a constrained prompt that enumerates the three valid labels — SUPPORTS, REFUTES, and NOT ENOUGH INFO — ensuring unambiguous extraction without relying on string heuristics." Note: StrategyQA similarly uses a constrained boolean prompt ("Answer yes or no. Question: {q}") for the same reason. | Session 20 Fix 6 / Impl Log | OPEN |
| BM-05 | FEVER NOT ENOUGH INFO class needs explicit treatment | When describing FEVER evaluation, add: "For FEVER's NOT ENOUGH INFO class, the correct model behaviour is expressing uncertainty rather than generating a confident label. The verification pipeline treats a high-confidence generation on a NOT ENOUGH INFO ground-truth item as a failure — semantic entropy detects this through a high-entropy output distribution, and the neutral NLI result (VE1) prevents such episodes from entering memory." | Session 3 | OPEN |

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
| IMPL-01 | **Tier 1 skips Stage 5 verification — TWO diagrams must be corrected** | **(1) SUBMITTED — `chapter_1.tex` TikZ diagram (line 158):** The already-submitted Chapter 1 architecture figure contains `\draw[arrow=green!70] (t1) -\| (verify);` — an arrow from Tier 1 directly into Stage 5 (VERIFY box). This arrow must be removed before final submission. Tier 1 reconstructs `StoredConfidence` from the stored entry's quality scores and serves the answer directly — no verification call is made per query. Fix: delete that one `\draw` line. *(2) FUTURE — Chapter 4 TikZ diagram (to be written):* When drawing the Chapter 4 architecture diagram, do NOT include a Tier 1 → Stage 5 arrow. In the thesis text: "Tier 1 serves retrieved answers directly — Stage 5 verification is not re-run per query. Quality assurance is maintained by retroactive re-verification (§4.x), which re-scores all stored episodes with the improved model. This amortises verification cost over one batch per cycle rather than per query — a strictly stronger freshness guarantee." Update Scenario 1 latency to "~150–400 ms (FAISS retrieval only, no generation)." | Session 26 + Session 20 / Impl Log | ⚠️ MUST FIX Ch1 BEFORE SUBMISSION + OPEN for Ch4 |

### §4.3–4.4 — Confidence Signals (u_pre and û)

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C4-03b | C_conv adaptation must be explicitly acknowledged | C_conv was developed by Nandakishor (2025) for decoder-only models. CAEM applies it to Flan-T5's *encoder* hidden states to measure query understanding stability before generation. Write: "We adapt C_conv to the encoder context: whereas Nandakishor (2025) applied it to decoder states, CAEM uses encoder layer variance ratios to measure pre-generation query understanding confidence." This is a legitimate adaptation but must be stated, not assumed. | Session 3 correction | OPEN |
| C4-02 | u_SC > 0.85 (Chapter 2) vs u_SC > 0.90 (VE1 escalation rule) | Explicitly distinguish in Chapter 4: the > 0.85 threshold is the general self-consistency verification pass/fail gate; the > 0.90 threshold in VE1 is specifically for the neutral NLI escalation condition (systematic confabulation detection requires a higher bar). Different uses, different thresholds — state this explicitly to preempt committee questions. | Session 1 cross-check | OPEN |
| C4-05 | Stage 4a (post-generation confidence) must be framed as an efficiency gate, not a quality gate | Stage 4a does not determine answer quality — verification (Stage 5) does that. Stage 4a's û is an *efficiency filter*: it avoids running the expensive verification pipeline on answers the model itself rates as low-confidence. Write: "Stage 4a is a cost-saving pre-filter, not a replacement for verification. It catches obvious failures before committing to the full NLI + SC + SE pipeline." | Session 3 | OPEN |
| C4-06 | û weight correction (Session 3 stated wrong weights) | Correct weights in û: u_token = 0.20, u_dropout = 0.20, u_SC = **0.20**, (1−H_sem) = **0.40**. NOT 0.30/0.30 as mistakenly stated in Session 3. SE gets double weight (0.40) because it is the only signal that catches systematic misconceptions and has the strongest empirical AUROC (0.79, Farquhar et al. 2024). | Session 4 correction | OPEN |
| C4-07 | Initial vs calibrated weights must be distinguished | Present equal initial weights (0.25 each) as the *starting point*, then explain calibration produces the final weights (0.20/0.20/0.20/0.40). Write: "Initial weights are set equal at 0.25 for all four signals. After temperature scaling calibration on the held-out calibration set, weights are refined to reflect each signal's empirical discriminative power, with semantic entropy receiving elevated weight (≈0.40) given its superior AUROC for confabulation detection." | Session 4 | OPEN |
| C4-09 | Blind spot table should appear in Chapter 4 | Include a 4×4 table showing which signal catches which failure mode (lexical uncertainty / epistemic uncertainty / reasoning instability / systematic misconception). This directly answers the committee question "why four signals?" | Session 4 | OPEN |

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
| C4-17 | Purity theorem must include a worked example — ILLUSTRATIVE ONLY in Chapter 4 | In Chapter 4 (methodology), present the theorem formula symbolically first: P = pα / (pα + (1−p)(1−α)). Then add ONE illustrative example labeled explicitly as pedagogical: "For example, if p = 0.70 and α = 0.85, then P ≈ 0.93." These numbers are for illustration only — they are NOT experimental results. State the required condition p > (1−α) explicitly. In Chapter 5 (results), REPLACE the illustrative numbers with actual measured values from `outputs/purity_validation/theory_validation.json`. Never present the Chapter 4 illustrative example as if it were an experimental finding. | Session 9 + Session 26 | OPEN |
| C4-23 | Convergence claim must be scoped — never claim "infinite cycles → zero hallucination" | Session 22 confirmed this claim would be TOO STRONG and academically indefensible. The correct claim is: "Improvement is monotone and bounded by (p, α) — the model's generation accuracy and the verifier's accuracy set the practical ceiling. The purity theorem guarantees data quality improvement per cycle, not eventual perfection." Write: "CAEM does not claim to eliminate hallucination — it provides a bounded, monotone improvement trajectory. The ceiling is determined by the base model's capacity (p) and the verification pipeline's accuracy (α). Diminishing returns between Cycle 2 and Cycle 3 are the empirical signal confirming approach to this bound." Do not write: "across infinite cycles, hallucination approaches zero" or "CAEM converges to a hallucination-free model." | Session 22 / Impl Log | OPEN |
| C4-18 | Coupled recurrence convergence must be explained in plain English alongside the mathematics | Write: "The two loops are coupled: a better model produces purer memory (via better generation and better verification signals), and purer memory produces a better model (via higher-quality fine-tuning data). This positive feedback is bounded by the model's capacity ceiling and the verification pipeline's maximum achievable accuracy — the system converges rather than diverging. Three cycles serve as a practical approximation of convergence; empirical confirmation comes from diminishing improvement magnitude between cycles 2 and 3 in the Chapter 5 results." | Session 9 | OPEN |
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
| C5-04 | Chapter 5 must have a single narrative thread connecting all results back to the closed-loop architecture argument | Do not present results as a flat list of numbers. Every result section should answer one of three questions: (1) Does CAEM outperform baselines that lack a closed loop? (2) Do the three internal mechanisms each demonstrably contribute? (3) Do the theoretical predictions hold empirically? Structure: open with the primary result (hallucination reduction), use ablations to prove each mechanism's contribution, use the theory validation tables to elevate CAEM from heuristic to principled. Close with the narrative: "Together, these results confirm that CAEM's closed-loop architecture delivers consistent, measurable, and theoretically grounded hallucination reduction across four qualitatively distinct benchmarks." | Session 10 | OPEN |
| C5-05 | Statistical significance testing is required before claiming CAEM "outperforms" any baseline | Accuracy differences between CAEM and baselines must be tested for statistical significance before any claim of "outperformance." Use McNemar's test for paired binary comparisons (correct/incorrect per question). Report: χ² statistic, p-value, and whether p < 0.05. Write: "Statistical significance of accuracy improvements was assessed using McNemar's test on paired question-level outcomes (correct/incorrect). Improvements with p < 0.05 are reported as statistically significant." If a result is directionally positive but not significant, write: "The improvement on TruthfulQA is directionally consistent but does not reach statistical significance at p < 0.05, likely due to the smaller effective sample size (N=X)." Do NOT claim "CAEM significantly outperforms" without running the test. Source: per-question EM arrays in the eval JSON files. | Session 26 | OPEN |
| C5-12 | McNemar's test and bootstrap CI are already implemented — use them, do not rewrite | `eval/metrics.py` (Session 20, Fix 8) already contains `mcnemar_test(scores_a, scores_b)` (chi-squared with Edwards continuity correction) and `bootstrap_ci(scores, n_bootstrap=1000, ci=0.95)` (non-parametric). When writing Chapter 5, use these functions directly on the per-question EM arrays saved in `outputs/eval/{bm}_cycle{n}.json`. Do not write new statistical test code — the implementations are verified and tested. Write in §5.2: "Statistical significance was assessed using `mcnemar_test()` from `eval/metrics.py` — a McNemar's chi-squared test with Edwards continuity correction on paired per-question correct/incorrect outcomes. Non-parametric 95% confidence intervals were computed via `bootstrap_ci()` (B=1000 resamples). Both functions were written and validated before the experiment phase." | Session 20 / Impl Log | OPEN |
| GEN-12 | Statistical significance must be computed and reported — do not rely on effect size alone | The committee will ask "are these improvements statistically significant?" for every comparison in Chapter 5. The rule: any claim of CAEM outperforming a baseline requires a McNemar's test (p < 0.05). Any improvement that cannot be tested must be labeled "directionally positive, not statistically significant at N=X." Practical plan: (1) Save per-question EM arrays from `eval/{bm}_cycle{n}.json`; (2) For each CAEM vs baseline pair, construct the 2×2 contingency table; (3) Run McNemar's test (scipy.stats.mcnemar); (4) Report χ², p-value, and effect size (odds ratio). | Session 26 | OPEN |

### §5.3 — Mechanism Evidence Table

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-02 | Report five cycle-level metrics in a single table to prove all three mechanisms are working | Include a table with columns: Cycle, Hallucination Reduction (%), Tier 1 Fraction (%), Tier 3 Fraction (%), MMLU Retention (%), Mean û_stored in Memory. A growing Tier 1 fraction proves memory is accumulating trusted knowledge. Stable MMLU proves forgetting is controlled. Rising mean û_stored proves retroactive cleaning is working. Report per-benchmark Tier 1 accuracy separately to catch FEVER near-miss issues. Source: `experiment_summary.csv`. | Session 8 | OPEN |

### §5.4 — Theory Validation

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-03 | Theory validation must be presented as a structured experiment, not a side note | Chapter 5 must explicitly execute and report all three theory validations: **Theory 1 (Purity Theorem):** run the 5-step purity validation protocol. **Theory 2 (Recurrence Monotonicity):** report p and α per cycle; confirm p_0 < p_1 < p_2 < p_3 and α_0 < α_1 < α_2 < α_3. **Theory 3 (Convergence):** compute Δ per cycle; confirm Δ(2→3) < Δ(1→2). All three use the same purity validation set measurements — no extra experiments needed. Source: `outputs/purity_validation/theory_validation.json`. | Session 9 | OPEN |
| TH-03 | Chapter 5 theory validation tables must use actual script output — never plan projections | The three theory validation tables in Chapter 5 must be populated ONLY from: (1) `outputs/purity_validation/theory_validation.json` for Theory 1 and Theory 2; (2) `outputs/full_experiment/all_cycle_results.json` for Theory 3. The plan's projected example values (p=0.62, α=0.88 etc.) are planning estimates only — never copy them into Chapter 5. If actual values differ significantly from projections, report the real numbers and explain the divergence. | Session 26 | OPEN |
| TH-04 | Convergence Δ table must show the check column explicitly | In Chapter 5, the convergence table must include a "Convergence Check" column: Cycle gap / Accuracy Delta (Δ) / Check. Checks: Δ(C1→C2) < Δ(C0→C1) → ✅ or ❌; Δ(C2→C3) < Δ(C1→C2) → ✅ or ❌. If any check fails, state: "While Δ(Cn→Cn+1) did not strictly decrease between cycles X and Y, the overall trend confirms convergence toward a fixed point." Do NOT hide or omit a failed check. | Session 26 | OPEN |

### §5.5 — Ablations and §5.6 — Baselines

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-09 | Ablation and baseline sections need explicit structure — not just a table dump | Structure as two distinct sub-sections: (1) **Baseline comparison** — present all 6 baselines in one table (Zero-shot / CoT / RAG / Self-consistency / Vanilla FT / Memory-only). For each, state what single mechanism it lacks vs CAEM. (2) **Ablation analysis** — present 9 ablation configs grouped by which mechanism they probe: memory quality group (no verification, single-layer NLI, no retroactive re-verification A1), routing group (all-Tier-3 A2, no OR-condition A3), training group (vanilla FT, no FT / memory-only), confidence group (no SE, (q,a)-only vs (q,c,a)). Each group answers one committee question. The most important ablation to highlight is A1 (no retroactive re-verification) — it uniquely proves the memory quality mechanism. | Session 26 | OPEN |
| IMPL-04 | **StrategyQA uses train split — explicit justification needed in §5.3 setup** | Write: "StrategyQA evaluation uses the `wics/strategy-qa` train split (~2,290 questions) because the test split provides no public ground-truth labels. The CAEM self-improvement loop trains only on its own verified generations — not on dataset labels — so using the labelled train split as held-out evaluation introduces no data leakage." Cite the split size and note the allocation: 500 calibration + 500 purity validation + remaining for evaluation. | Session 23 / Impl Log | OPEN |

### §5.7 — Computational Efficiency

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-06 | Computational cost / efficiency analysis must appear as a dedicated results subsection | Chapter 5 must include a §5.7 Computational Efficiency section reporting: (1) Per-tier mean latency (ms) from `experiment_summary.csv` mean_latency_ms column; (2) Tier distribution across cycles (showing Tier 1 fraction growing → average latency falling); (3) GPU memory peak during fine-tuning; (4) Total GPU hours for the full experiment. Write the cost-benefit argument: "The routing architecture provides a computational dividend across cycles: as Tier 1 fraction grows from ~5% (Cycle 0) to ~38% (Cycle 3), mean query latency falls from ~Xms to ~Yms — the same system becomes faster as it accumulates knowledge. CAEM's total inference cost at Cycle 3 is Z% lower than an equivalent all-Tier-3 baseline (A2 ablation), while achieving higher accuracy." | Session 26 | OPEN |

### §5.8 — Stability-Plasticity Analysis

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-07 | Stability-plasticity analysis must be a named subsection in Chapter 5, not just one number | Dedicate §5.8 to "Stability-Plasticity Analysis" covering: (1) MMLU Retention per cycle — target ≥93%; report actual value from `run_ablation.py` MMLU eval; (2) Forgetting score per cycle from `retroverify_cycle{n}.json` — did any cycle trigger the abort threshold (score < 0.07)?; (3) Fine-tuning loss per cycle; (4) Plasticity evidence — EM improvement per cycle. Structure: "CAEM's L2 uniform regularisation successfully navigates the stability-plasticity tradeoff. On the plasticity side, EM improves across all four benchmarks by Cycle 3. On the stability side, MMLU retention remains at X% throughout (target ≥93%), confirming that general language capabilities are preserved. The forgetting score never triggered the abort threshold (0.07), indicating that the λ=0.01 [DES] L2 penalty provides sufficient regularisation at this task scale. This empirically validates the assumption that uniform-weight L2 approximates full EWC when the fine-tuning task is diverse QA." If MMLU retention drops below 93%: immediately flag as catastrophic forgetting — increase λ or consider full EWC. | Session 26 + Session 29 | OPEN |

### §5.9 — Memory Utilization and Error Analysis

| # | Issue | Required Fix | Source | Status |
|---|-------|-------------|--------|--------|
| C5-10 | Memory utilization and saturation must be reported | Report in Chapter 5: (1) Total episodes stored per benchmark per cycle; (2) Total memory size at end of Cycle 3 vs capacity K=1000 — is memory near saturation?; (3) Mean û_stored distribution (histogram or quartiles) — not just the mean; (4) Retroactive re-verification pruning rate per cycle. Write: "By the end of Cycle 3, the episodic memory contained N episodes across 4 benchmarks (X% of capacity K=1000)." If memory is near saturation (>90% of K=1000), note it as a scalability consideration. Source: `experiment_summary.csv`. | Session 26 | OPEN |
| C5-08 | Error analysis and failure modes must appear as a dedicated subsection | Include §5.9 Error Analysis covering: (1) What question types CAEM still fails on — qualitative sample of 5–10 failure cases; (2) TruthfulQA failure mode: memorising specific misconception pairs vs genuinely learning epistemic resistance; (3) HotpotQA failure mode: which multi-hop patterns fail (entity bridging, temporal reasoning, comparative); (4) FEVER near-miss failures: report Tier 1 accuracy per benchmark separately — lower FEVER Tier 1 accuracy than HotpotQA/StrategyQA confirms near-miss retrieval; (5) Residual hallucination rate at Cycle 3. Write: "Despite the overall improvement, CAEM retains systematic failure modes on [X type] questions. Honest failure analysis strengthens the thesis — a committee that sees no failures will not believe the results." | Session 26 | OPEN |

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
| GEN-06 | Near-miss retrieval: acknowledge FEVER risk specifically | FEVER is the vulnerable benchmark: claims are adversarially constructed around the same entities with different labels (e.g., "Marie Curie born in Poland" SUPPORTS vs "Marie Curie born in France" REFUTES — very high cosine similarity, opposite labels). Write: "FEVER's adversarially constructed claims — structurally similar but label-distinct — pose the greatest near-miss retrieval risk among the four benchmarks. The Tier 1 combined score threshold (≥0.90) and the requirement for high û_stored provide partial mitigation. Empirical evidence is reported in Chapter 5: per-benchmark Tier 1 accuracy; anomalously lower FEVER Tier 1 accuracy would indicate near-miss retrieval as the likely cause." | Session 7 | OPEN |
| GEN-13 | Scaling guide changes are pending a better device — hold final methodology text until device is confirmed | `LAB_PC_SCALING_GUIDE.md` documents 3 categories of changes required for better GPU: (1) L2 penalty moves fully to GPU (`_l2_penalty` no longer uses `.detach().cpu()`); (2) batch size scales up (batch=4 → 16 on A100); (3) passage index can scale to 5M passages. **Do not write final §4.7 methodology text or §5.7 efficiency numbers using RTX 3060 smoke-test values.** The full experiment may run on Kaggle (T4/P100), a university GPU (A100), or a rented instance (Vast.ai). After the experiment device is confirmed: (a) update §3.x hardware constraints to reflect actual experiment hardware; (b) verify the L2 penalty implementation matches the device (CPU offload vs GPU native); (c) use actual latency figures from `experiment_summary.csv` mean_latency_ms column — not smoke-test estimates. | Session 29 + LAB_PC_SCALING_GUIDE | OPEN |
| GEN-07 | Retrieval feedback (success_rate) depends on ground truth in benchmark mode | State scope explicitly: "In the current benchmark evaluation, episode success_rate is updated by comparing Tier 1 responses to ground-truth labels. Extension to production settings would require an explicit user feedback mechanism or active sampling verification." | Session 7 | OPEN |
| GEN-09 | Unified plan is pedagogical — never copy its numbers directly into the report | `caem-unified-plan-v3.tex` uses illustrative example values throughout to explain and teach the system. These are planning targets and worked examples, NOT measured results. (1) Chapter 4 — state theory formulas symbolically with one labeled illustrative example. (2) Chapter 5 — use ONLY actual values from `outputs/` files. (3) Never write "CAEM achieves 28% hallucination reduction" until the actual experiment confirms it. | Session 26 | OPEN |
| GEN-10 | Phase 1 submitted chapters (Ch1, Ch2) may need minor updates after experiments | After the full experiment runs, check: (1) Ch1 — does the actual hallucination reduction match the stated target? If materially different, update the claim. (2) Ch2 — benchmark accuracy figures cited are literature values — these do not change. | Session 26 | OPEN |

---

## Implementation Log → Thesis Mapping

These entries identify specific implementation decisions (from `caem-implementation-log.md`) that must surface in the thesis. Each specifies WHERE the decision belongs, WHAT to write, and WHY it matters for the committee. They are also listed inline within the relevant chapter sections above for easy reference during writing.

| # | Implementation Decision | Thesis Location | Status |
|---|---|---|---|
| IMPL-01 | Tier 1 skips Stage 5 — (a) remove arrow from submitted Ch1 TikZ (chapter_1.tex line 158); (b) don't draw it in Ch4 diagram | Ch1 TikZ ⚠️ + Ch4 §4.2 | ⚠️ MUST FIX |
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
| PUB-01 | **Strong baselines missing** — no comparison to GPT-3.5/4, Self-RAG, FLARE, SelfCheckGPT, RARR | Add at minimum: (1) GPT-3.5/4 zero-shot on all 4 benchmarks (API call, no GPU required — use published benchmark numbers if re-running is not feasible); (2) Self-RAG (Asai et al. 2023) — closest architectural prior; (3) SelfCheckGPT (Manakul et al. 2023) — closest hallucination detection baseline. For a conference paper, re-run Self-RAG and SelfCheckGPT on the same benchmark splits. | REQUIRED | OPEN |
| PUB-02 | **Purity theorem lacks a formal proof** — theorem-proof block needed in Chapter 4 | Write the proof as: (1) define the Bayesian model; (2) apply Bayes' theorem; (3) show P = P(correct | accepted) = pα / (pα + (1−p)(1−α)); (4) prove P > p iff p > (1−α). Use LaTeX `\begin{theorem}` / `\begin{proof}`. See Ch4 §4.9 entry above. | REQUIRED | OPEN |
| PUB-03 | **Human evaluation for TruthfulQA** — automated ROUGE-L proxy is weak | Conduct a small human evaluation (labmates or MTurk, N=100 subset, Cycle 0 vs Cycle 3 outputs). Present as a validation study: "Automated ROUGE-L and human evaluation are in agreement on X% of samples." Even 2-annotator Cohen's κ reported would substantially strengthen the TruthfulQA claim. | STRONGLY RECOMMENDED | OPEN |
| PUB-04 | **LoRA / PEFT comparison ablation** — reviewers will ask "why full fine-tuning?" | Add as Appendix ablation: LoRA (r=8, α=16) vs full fine-tuning + L2, reporting MMLU retention and EM improvement per cycle. If LoRA matches full FT on accuracy with better retention, note as future direction. If full FT + L2 is better, report why. Either way, the ablation pre-empts the question. | STRONGLY RECOMMENDED | OPEN |
| PUB-05 | **Full EWC vs L2 ablation** — needed to validate the L2 approximation claim | Add one ablation variant: full EWC (compute diagonal FIM, re-run Cycle 1–3). Compare MMLU retention, EM improvement, and compute time. Expected: similar MMLU and EM, but 30–40% higher compute time — validating the L2 approximation. Noted in `caem-implementation-log.md` Session 29 as planned. | STRONGLY RECOMMENDED | OPEN |
| PUB-06 | **Larger model experiment** — single model scale (780M) limits generalisability | Run Flan-T5-XL (3B) on HotpotQA only (clearest EM metric). Report in Appendix B. Question to answer: does CAEM's improvement trend hold at larger scale, or does the convergence ceiling simply move up proportionally? | STRONGLY RECOMMENDED | OPEN |
