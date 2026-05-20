# Paper Notes: CAEM Thesis to Conference Paper

Living document. Annotates every element of the thesis with a tag describing
how it transfers to a future conference paper (NeurIPS / ICML / ICLR /
EMNLP / ACL / COLM). Updated as chapters are written. Chapters 1, 2, 5, 6
to be filled in later.

---

## Classification legend

| Tag | Meaning |
| --- | --- |
| **READY** | Transfers to the paper with minor editing. Core scientific content. |
| **COMPRESS** | Idea transfers; current prose does not. Compress to 1 to 3 sentences or a single equation. |
| **APPENDIX** | Belongs in supplementary / appendix at venues that allow it. Not in the 8-page body. |
| **THESIS-ONLY** | Pedagogical scaffolding, chapter intros, summaries, risk enumerations. Omit from paper. |
| **TBD** | Decision deferred until Chapter 5 numbers land (affects paper framing). |

## Guideline for diagrams, tables, algorithms

Applied consistently to every new section to avoid accidental decoration.

**Figures.** Add a dedicated figure only when the figure communicates a
claim that the text alone cannot. Figure 4.1 (pipeline diagram) meets this
bar because the topology of the eight stages is itself the contribution.
A Stage-internal figure for Stage 1 would be redundant with the S1 box in
Figure 4.1 and is therefore omitted. Rule of thumb: if the figure is
summarising what one or two equations already say, delete the figure.

**Tables.** Use a table when the content is genuinely tabular (multiple
rows with shared columns), not to lay out a list with borders. Table 3.1
(NFRs), Table 3.2 (CES axes), Table 3.3 (ablation registry), and Table 4.1
(notation) all meet this bar. A 3-row tier-threshold table in §4.5 will
meet it too. Inline lists in prose do not.

**Algorithms.** Include a pseudocode box only when the stage has edge
cases or control flow the equations cannot carry: empty greedy decodes,
clamping, conditional execution, or when reproducibility depends on step
ordering. Algorithm 4.1 (Stage 1) meets this bar because of the
empty-prefix safety path and the clip-then-temperature ordering. Stage 2
(nearest-neighbour retrieval) and Stage 4 (tier execution) will not need
algorithm boxes because the mechanism collapses to one line of pseudocode
that the prose already contains.

**Rule of thumb for paper extraction.** Every figure, table, and algorithm
in the thesis is annotated below with a separate tag, because a tight
conference paper will keep perhaps three figures and two tables out of
everything the thesis contains.

## Candidate paper angles (choose after Chapter 5 results)

The angle determines which thesis material counts as core versus context.
Pick one based on which axis of Chapter 5 is strongest.

1. **Architecture angle.** "Verified episodic memory as a hallucination-
   reduction mechanism under confidence-aware routing." Strongest if EPI
   and hallucination-reduction numbers are the headline. Fits EMNLP / ACL.
2. **Dynamics angle.** "Multi-cycle self-improvement under a forgetting
   bound compounds verified memory without capability erosion." Strongest
   if cycle-over-cycle CES gain is the headline, especially if RET stays
   flat while EPI improves. Fits NeurIPS / ICLR.
3. **Practicality angle.** "A single-GPU 780M pipeline competitive with
   much larger instruction-tuned baselines on factual QA." Strongest if
   comparison to Llama-3-8B-Instruct or similar is favourable at the EM or
   CES level. Fits COLM / EMNLP systems track.

## Target paper skeleton

Standard 8 to 10 page structure assumed. Each thesis element below is
annotated with which section of this skeleton it maps to.

| # | Paper section | Budget |
| --- | --- | --- |
| 1 | Abstract | 200 words |
| 2 | Introduction | 1 page |
| 3 | Related Work | 0.75 page |
| 4 | Method (architecture + equations) | 2.5 pages |
| 5 | Experimental Setup | 0.75 page |
| 6 | Results | 2 pages |
| 7 | Discussion | 0.5 page |
| 8 | Conclusion | 0.25 page |
| (appendix) | References + Appendix | unlimited (venue permitting) |

---

# Chapter 3: Requirements, Impacts, Constraints

Chapter 3 is a requirements-engineering artefact. Its framing (functional
requirements, non-functional requirements, risk analysis) is standard for a
thesis but foreign to ML conference papers. The *content* inside the FRs is
valuable; the *scaffolding* is not. Almost every FR becomes one or two
sentences inside the paper's Method section.

## §3.1 Introduction
**Tag:** THESIS-ONLY. Chapter-level intros do not survive extraction. The
paper's single Introduction is written fresh.

## §3.2 Functional Requirements

### §3.2.1 FR1, Pre-Routing Confidence Estimation
**Tag:** COMPRESS to Method §4.1. The *mechanism* (single-encoder pass,
two-component fusion of `u_token` and `C_conv`, offline temperature
calibration) is paper-ready. The *FR-style framing* is not. Target: one
paragraph with the defining equation.

### §3.2.2 FR2, Adaptive Three-Tier Routing
**Tag:** COMPRESS to Method §4.2. Routing rule
`S(q) = α·sim + (1-α)·u_stored`, three thresholds, safety override. The
tier dispatch table can be a 3-row inline table in the paper.

### §3.2.3 FR3, Multi-Signal Post-Hoc Verification
**Tag:** COMPRESS to Method §4.3. Nine-signal verifier and six-weight
composite (Eq 3.5) are the scientific heart of the method and transfer
directly. The confabulation gate deserves one sentence with its threshold.
The atomic-decomposition mechanism is publishable on its own and should be
highlighted rather than buried.

### §3.2.4 FR4, Four-Outcome Storage Decision
**Tag:** READY to Method §4.3. The four-outcome decision tree (Store,
Deferred, Abstain, Discard) is distinctive enough to justify a small
labelled diagram. Most pipelines binary-decide; the four-outcome structure
is a genuine contribution.

### §3.2.5 FR5, Episodic Memory Management
**Tag:** COMPRESS to Method §4.4. FAISS plus `u_stored` alongside
embeddings, novelty filter at cosine 0.95, pruning by value score. Target:
2 to 3 sentences. Implementation detail of IVF-PQ index parameters to
APPENDIX.

### §3.2.6 FR6, Retroactive Re-Verification
**Tag:** READY to Method §4.5 (own paragraph). The upward-only update rule
is non-obvious and worth surfacing explicitly in the paper; it is also the
mechanism that enables the dynamics angle (angle 2 above).

### §3.2.7 FR7, Constrained Self-Improvement
**Tag:** READY to Method §4.6. EWC / L2 anchor regularisation, 90/10
episode mix, MMLU retention ratio as rollback trigger. This is the whole
"compounds across cycles without forgetting" argument in one subsection.

### §3.2.8 FR8, Ablation and Evaluation Instrumentation
**Tag:** THESIS-ONLY. An FR stating "the system must be ablatable" is an
engineering requirement, not a scientific contribution.

## §3.3 Non-Functional Requirements (Table 3.1)

**Tag:** COMPRESS to Experimental Setup §5. The forgetting bound ρ ≥ 0.93
is *the* non-functional requirement the paper must state, because it is
the abort-guard threshold and it comes up in every result. One sentence.
Calibration stability (ECE ≤ 0.05) is worth one sentence in the Method
§4.1 paragraph on temperature scaling. The rest (single-GPU feasibility,
latency envelope, reproducibility) belong in Experimental Setup or the
APPENDIX.

Table 3.1 itself: THESIS-ONLY as a table; the two cells above go inline.

## §3.4 System Constraints

### Hardware paragraph
**Tag:** COMPRESS to Experimental Setup §5. One sentence: "All experiments
run on a single RTX 4090 (24 GB)." The memory accounting breakdown stays
in the thesis.

### Compute budget paragraph
**Tag:** THESIS-ONLY. Phase-1 and Phase-2 structure is thesis-specific
project management. The paper reports final numbers; readers do not care
that Phase 1 was a single-seed screening sweep.

### Data paragraph
**Tag:** READY to Experimental Setup §5. Exact benchmark list (NQ,
TriviaQA, FEVER, HaluEval, TruthfulQA, MMLU, Wikipedia DPR passages)
transfers verbatim; it is the paper's dataset paragraph.

### Scope paragraph
**Tag:** COMPRESS to Introduction. Half a sentence in the first paragraph
of the paper's Introduction: "We restrict attention to factual QA where
grounding against an external corpus is well-defined."

## §3.5 Evaluation Methodology

### §3.5.1 Composite Evaluation Score
**Tag:** READY to Experimental Setup §5. The CES definition (Eq 3.1) and
geometric-mean justification are both publishable. Half-page at most in
the paper; the geometric-versus-arithmetic justification is a single
sentence.

**Table 3.2 (CES axes):** READY. Exact table lifts to the paper's Metrics
paragraph. If VER is still a placeholder at paper time, replace the row
with EPI-weighted hallucination rate or drop to four axes.

### §3.5.2 Diagnostic Metrics
**Tag:** READY to Experimental Setup §5. Confident-error rate, AUROC of
`u_stored`, reliability diagrams. One sentence each; reliability diagrams
should appear as a figure in Results if they are visually compelling.

## §3.6 Ablation Study Design

### Prose paragraph
**Tag:** COMPRESS to Results §6. Two sentences: "We ablate sixteen
variants covering every major mechanism. Fourteen are cyclic (require full
trajectory re-run); two are non-cyclic." Justification of the cyclic /
non-cyclic distinction to APPENDIX.

### Table 3.3 (ablation registry)
**Tag:** READY to Results §6 as the ablation table. This is paper-ready
as-is, possibly with the cyclic column dropped (paper readers do not need
to know about compute bookkeeping). The 16-variant coverage is a genuine
strength and should be foregrounded.

## §3.7 Risk Analysis and Mitigation

**Tag:** THESIS-ONLY for the "risks" framing. COMPRESS for the mechanisms.

The paper does not enumerate "risks"; it motivates mechanisms directly.
Harvest the content as follows. Verification drift motivates retroactive
re-verification in §4.5. Catastrophic forgetting motivates the rollback
guard in §4.6. Near-duplicate accumulation becomes one sentence in §4.4 on
the novelty filter. Confident confabulation motivates the confabulation
gate in §4.3. Benchmark contamination becomes one line in Experimental
Setup justifying the geometric-mean aggregation.

Each becomes a single motivating sentence; do not reproduce the
"First / Second / Third / Fourth / Fifth" structure.

## §3.8 Ethical Considerations and Broader Impact

**Tag:** APPENDIX (or Ethics Statement).

ACL / NAACL / EMNLP require an Ethics Statement; NeurIPS / ICLR require a
Broader Impacts section. Lift 2 to 3 sentences: grounding-corpus bias
inherit, memory-is-durable caveat, refusal-to-deploy clause. The full
paragraph-per-point structure is too much for a paper.

## §3.9 Chapter Summary
**Tag:** THESIS-ONLY.

---

# Chapter 4: Methodology

Written so far: §4.1 Introduction, §4.2 System Overview (pipeline figure,
notation table, stage-by-stage narrative, Example 4.1), §4.3 Pre-Routing
Confidence Estimation (two opening paragraphs, three labelled paragraphs,
five equations, Algorithm 4.1), §4.4 Episodic Memory (seven labelled
paragraphs, one schema table, eight equations), §4.5 Three-Tier Router
(opening paragraph, five labelled paragraphs, two equations, one
tier-threshold table, Algorithm 4.2, closing paragraph), §4.6 Unified
Verifier (opening paragraph, eight labelled paragraphs, one signal-family
table, twelve equations, Algorithm 4.3), §4.7 Four-Outcome Storage
Decision (opening paragraph, recoverability-argument paragraph, five
labelled paragraphs, three equations, one decision-tree table, Algorithm
4.4, cycle-level diagnostic closing paragraph), §4.8 Retroactive
Re-Verification (opening paragraph, five labelled paragraphs, one
piecewise cascade equation, Algorithm 4.5, dynamics-hook closing
paragraph), §4.9 Deferred-Entry Reconsideration (opening paragraph,
five labelled paragraphs, one piecewise cascade equation, Algorithm
4.6, diagnostic closing paragraph), §4.10 Self-Improvement Loop
(opening paragraph, seven labelled paragraphs, four equations,
Algorithm 4.7, closing disaggregation paragraph), §4.11 Implementation
(opening paragraph, prose, two hyperparameter/component tables),
§4.12 Chapter Summary (two recap paragraphs plus forward-pointer to
Chapters 5 and 6). Chapter 4 is now complete.

## §4.1 Introduction
**Tag:** THESIS-ONLY. Same rule as §3.1: chapter-level intros do not
survive.

## §4.2 System Overview

### Opening paragraph
**Tag:** COMPRESS to Method §4.0 (method-section opener). The sentence
"Figure X shows the per-query path through CAEM on a single page. Every
symbol is defined in Table Y" is the right opener for the paper's Method
section too, with adjusted references.

### Figure 4.1, full CAEM pipeline diagram
**Tag:** READY to Method §4. *The* architecture figure for the paper.
Already single-page, `\resizebox`-fitted, non-overlapping. Minor
adaptation: the conference version may need the self-improvement loop as
a separate inset (space permitting) since two figures is common for a
full-method paper.

### Table 4.1, notation
**Tag:** APPENDIX. Full notation tables are thesis convention. Papers
inline notation on first use. Keep the table but move it to a
supplementary symbols appendix.

### §4.2.1 Stage-by-stage narrative (8 paragraphs)
**Tag:** READY in compressed form, to Method §4.1 through §4.6 (one
paragraph per mechanism, not per stage). This is literally the paper's
Method content, already written at the right altitude, minus the FR cross-
reference framing. The compression target is about 60% of current length:
drop the bridging sentences that repeat what Figure 4.1 already shows.
Order in the paper:

1. Method §4.0 opener (Figure 4.1 plus one-paragraph overview).
2. §4.1 Pre-route confidence (Stage 1 paragraph plus `u_pre` equation).
3. §4.2 Memory plus routing (Stages 2 and 3 merged; they share the same
   decision surface).
4. §4.3 Verification and storage decision (Stages 5 and 6 merged; the
   four-outcome structure is the highlight, Stage 7 is a one-liner).
5. §4.4 Tier 1 short-circuit plus Tier 3 grounding (Stage 4 paragraph
   split).
6. §4.5 Retroactive re-verification (Stage 8 first half).
7. §4.6 Self-improvement plus rollback (Stage 8 second half).

Tier 2 is zero-shot and needs no dedicated subsection; fold into the
routing paragraph.

### §4.2.2 Worked example (Example 4.1)
**Tag:** THESIS-ONLY. Worked-example boxes are pedagogical and do not
appear in conference papers (see genre discussion in session notes). Drop
entirely from the paper. The two queries (Query A Tier-1 cheap path,
Query B Tier-2 full verify) may be referenced as qualitative examples in
Discussion if space permits, as a single inline sentence each rather than
as a panel.

## §4.3 Pre-Routing Confidence Estimation

### Opening two paragraphs (motivation plus signal-complementarity)
**Tag:** COMPRESS to Method §4.1 (one paragraph). The motivation (encoder-
only, decoder-free, cost paid per-query regardless of tier) survives. The
signal-complementarity argument (why `u_token` and `C_conv` disagree on
familiar-phrasing queries the model cannot actually answer) is worth half
a sentence in the paper; it is the non-obvious part of the design.

### Paragraph: Token-probability aggregate plus Eq 4.1 `u_token`
**Tag:** COMPRESS to Method §4.1. The equation transfers; the discussion
of the 32-token cap and the geometric-versus-arithmetic-mean choice moves
to APPENDIX. One sentence in the paper body: "Token confidence is the
geometric mean of per-token probabilities over a greedy prefix of at most
32 tokens."

### Paragraph: Encoder convergence ratio plus Eq 4.2 `C_conv`
**Tag:** READY to Method §4.1. This is the distinctive signal in Stage 1
(`u_token` on its own is standard; the encoder variance ratio is what
makes the signal set novel). Keep the equation; keep the interpretation
sentence ("a well-converged representation is one in which the late
layers vary much less than the early layers"). Drop the empirical
justification of the midpoint split to APPENDIX.

### Paragraph: Fusion plus Eqs 4.3, 4.4, 4.5
**Tag:** READY to Method §4.1. Eq 4.3 (the fused raw estimate) is paper-
essential. Eqs 4.4 and 4.5 (temperature scaling sigmoid form and L-BFGS
objective) COMPRESS to one sentence plus a single-line equation: "A
scalar temperature T is fit by L-BFGS on binary NLL on a held-out 500-
query calibration split, and applied as `u_pre = σ(logit(ũ)/T)`." The
explicit `arg min` statement (Eq 4.5) to APPENDIX.

### Algorithm 4.1, Stage 1 pseudocode
**Tag:** APPENDIX. Top venues rarely print stage-level pseudocode when
the equations cover the mechanism; the algorithm box is better suited to
the supplementary material. Exception: keep in the body if the paper is
targeted at a systems or reproducibility venue (COLM, NeurIPS datasets
and benchmarks track) where algorithmic clarity is expected.

### Closing paragraph (flow to Stage 3)
**Tag:** THESIS-ONLY. In the paper, Stage 1 and Stage 3 are described
contiguously, so no bridging sentence is needed.

### Citation of Guo et al. 2017 (temperature scaling)
**Tag:** READY to Method §4.1. The attribution is one of the load-bearing
citations of the whole calibration story; keep it.

## §4.4 Episodic Memory

### Opening motivation paragraph
**Tag:** COMPRESS to Method §4.2 opener. The memory's role as the
accumulating artefact (Tier 1 path widens across cycles) is the one
non-obvious architectural claim worth surfacing in the paper. Two
sentences maximum.

### Paragraph: Data schema plus Table 4.2 (entry schema)
**Tag:** Paragraph COMPRESS; Table 4.2 APPENDIX. In the paper body,
one sentence lists the immutable content (question, chain, answer,
embedding, cycle, timestamp) and one sentence lists the mutable record
(nine signals plus composite plus usage). The full schema table belongs
in the implementation appendix.

### Equation 4.6 (u_stored six-weight composite, reproduced from Ch3 Eq 3.5)
**Tag:** READY to Method §4.3 (not §4.2). The composite belongs with
verification rather than with memory in a compressed paper; it is cited
once here for cross-reference. In the paper it appears exactly once, in
the verification subsection.

### Paragraph: Embedding model plus Eq 4.7 (cosine via inner product)
**Tag:** COMPRESS to one sentence in Method §4.2: "Queries are encoded
with Sentence-BERT all-mpnet-base-v2 into 768-dim L2-normalised vectors,
so inner-product equals cosine similarity." Eq 4.7 to APPENDIX.

### Paragraph: Index structure plus IVF-PQ promotion
**Tag:** APPENDIX. This is pure implementation detail. One sentence in
the paper body: "Memory is backed by FAISS with a flat index up to 20K
vectors, promoted to IVF-PQ for the 1M-entry target." The full parameter
block (n_list=4096, m=64, nbits=8, nprobe=32) to APPENDIX. The
justification that approximate nearest-neighbour does not change routing
decisions is a good appendix paragraph but not body material.

### Paragraph: Novelty filter plus Eq 4.8
**Tag:** READY (compressed) to Method §4.2. The novelty filter and its
0.95 threshold earn a sentence because it is a direct mechanism against
training-pool bias. One sentence plus the inline threshold.

### Paragraph: Retrieval feedback update plus Eqs 4.9, 4.10
**Tag:** APPENDIX. The EMA feedback rule is between-cycle bookkeeping;
papers that do not emphasise continuous-learning dynamics treat this as
an implementation detail. Exception: if the paper angle is "dynamics"
(see Candidate paper angles), promote one sentence to the body.

### Paragraph: Value score plus Eqs 4.11 through 4.13
**Tag:** APPENDIX. Pruning policy is engineering, not contribution.
One sentence in the paper: "At 95% capacity the bottom 20% of entries
by a Value score combining recency, retrieval frequency, and success
rate are pruned." Expose the full Value decomposition only if Chapter 5
ablates it.

### Paragraph: Persistence
**Tag:** THESIS-ONLY. Save/load mechanics do not belong in a paper.

### Closing summary paragraph
**Tag:** THESIS-ONLY. Paragraph-level summaries get absorbed into the
section opener or deleted in paper extraction.

## §4.5 Three-Tier Router

### Opening paragraph (cost-control framing)
**Tag:** COMPRESS to one sentence in Method §4.2. "The router trades
generation cost against answer risk by dispatching to one of three
tiers."

### Paragraph: Two-mechanism design (safety override versus combined
score)
**Tag:** READY to Method §4.2. This is one of the most transferable
design arguments in the whole method chapter: the separation of
`u_pre` veto from the `sim + u_stored` score is non-obvious and is
exactly the kind of architectural commitment that earns a dedicated
paragraph in a paper. Target in the paper: one paragraph, roughly
three sentences (two-mechanism claim + arithmetic justification +
one-line restatement).

### Paragraph: Safety override plus Eq 4.14 (OR-condition)
**Tag:** READY to Method §4.2. Keep the floor value (0.60) inline;
keep the short justification about cost asymmetry. The three numbered
observations compress to one sentence in the paper: "The floor is set
at 0.60 because Tier 3 is the only path that introduces external
evidence, making forced grounding the operationally meaningful response
to low model readiness."

### Paragraph: Combined routing score plus Eq 4.15
**Tag:** READY to Method §4.2. Equation transfers verbatim; the
weighting justification (similarity = relevance, `u_stored` = quality
backstop, 7:3 weighting encodes priority) compresses to one sentence
plus the equation.

### Paragraph: Tier dispatch rules plus Table 4.3 (tier thresholds)
**Tag:** Paragraph COMPRESS; Table 4.3 READY. In the paper, the
threshold table is the compact way to state the three tier rules; lose
the paragraph and let the table carry the content. Three-row table
with conditions and actions.

### Paragraph: Behaviour on an empty memory
**Tag:** APPENDIX. Cold-start handling is an implementation robustness
story; papers usually handle this with a parenthetical rather than a
dedicated paragraph. One sentence at most: "Empty memory substitutes
zero for both similarity and `u_stored`, routing the query to Tier 3
by default."

### Algorithm 4.2, Stage 3 pseudocode
**Tag:** READY to Method §4.2. Unlike Algorithm 4.1 (Stage 1) which
can be tucked into an appendix because the equations carry the logic,
Algorithm 4.2 encodes the strict evaluation order (override before
score) that is the algorithmic form of the two-mechanism separation.
This is exactly the kind of control flow the paper_notes.md guideline
flags as warranting a pseudocode box. Include in body.

### Closing paragraph (logging, routing-distribution metrics)
**Tag:** COMPRESS to one sentence in Experimental Setup §5. The
distinction between safety-override-forced Tier 3 versus
default-rule Tier 3 is a reporting choice that belongs in the metrics
paragraph of the paper, not in the method section.

## §4.6 Unified Verifier

This is the anchor section of the verification story and the single
densest source of paper-ready material in the method chapter. The
three-family signal decomposition is the core architectural argument.

### Opening paragraph (three-family organising claim)
**Tag:** READY to Method §4.3. The argument that any single signal is
exploitable, so the composite is built from three families that fail
in different ways, is the paper-essential motivation. Keep roughly in
current form; compress the "section roadmap" sentence to one clause.

### Table 4.2 (nine-signal summary)
**Tag:** READY to Method §4.3. This table is the single most useful
artefact in the entire chapter for a conference reader: nine signals
grouped by family with cost and source columns. Lift verbatim, possibly
with the [L]/[D] source column merged into a parenthetical.

### Paragraph: Internal calibration (u_token-verify, u_dropout,
u_internal) plus Eqs 4.21, 4.22, 4.23
**Tag:** Eq 4.21 (u_token-verify) APPENDIX (redundant with Eq 4.1 from
§4.3); Eq 4.22 (u_dropout plurality formula) READY; Eq 4.23 (u_internal
fused scalar) READY. The MC-dropout Gal 2016 citation is paper-essential
and is the one load-bearing uncertainty-quantification citation for the
internal family. The fused-scalar equation transfers verbatim.

### Paragraph: Sample-set signals (s_avg, p_entail, h_norm) plus Eqs
4.24, 4.25, 4.26, 4.27
**Tag:** READY to Method §4.3. This is the signal cluster with the
strongest literature grounding (Wang 2023 for s_avg, Farquhar 2024 for
h_norm) and the most distinctive CAEM-specific combination choice
(p_entail as the chain-to-answer entailment filter on top of
self-consistency). All four equations transfer; only Eq 4.26 (the
bidirectional-cluster rule) can compress to one sentence in the paper.

### Paragraph: External grounding (p_ground_max, p_ground_mean,
p_contra) plus Eqs 4.28, 4.29, 4.30
**Tag:** READY to Method §4.3. The max-versus-mean argument (max
robust to retrieval failure, mean robust to single false-match) is a
publishable design argument. Keep. The retrieval parameters
(20-to-3 rerank) compress to a parenthetical.

### Paragraph: Atomic-fact decomposition plus Eqs 4.31, 4.32, 4.33
**Tag:** READY to Method §4.3 (own short paragraph). This is arguably
the most distinctive signal in the verifier: the weakest-link rule
over decomposed claims handles a specific failure mode the max/mean
aggregates cannot. Worth preserving in the body of the paper even at
tight page budget. Mention the fallback-to-p_ground_mean rule
(robustness to decomposition failure).

### Paragraph: NLI ensemble aggregation (min-entail / max-contra)
**Tag:** COMPRESS to Method §4.3. The conservative aggregation rule
earns half a sentence: "NLI outputs are aggregated by min across
ensemble bundles for entailment and by max for contradiction,
consistently biasing against storage under ensemble disagreement." The
single-model fallback rule is APPENDIX.

### Paragraph: Composite score plus Eq 4.34 (u_stored verify form)
**Tag:** READY to Method §4.3. This is the one equation the paper
reader must see. The weight-rationale paragraph (why 0.30 on
p_ground_mean, 0.15 on the four middle signals, 0.10 on 1 - h_norm,
why p_contra is NOT in the composite) is paper-essential but can
compress to two sentences. The reference to the equal-weights ablation
variant is a good forward hook to Results §6.

### Paragraph: Confabulation early-exit gate plus Eq 4.35
**Tag:** READY to Method §4.3. This is the one explicit safety rule
in the verifier and is worth its own short paragraph in the paper. The
gate is non-standard (most pipelines only threshold at the composite
level) and addresses a specific, named failure mode. Keep.

### Algorithm 4.3, verifier pseudocode
**Tag:** APPENDIX (by default) or READY if the paper targets a
systems-oriented venue. The algorithm captures step ordering (early-
exit evaluated before composite, atomic decomposition on the composite
path) which is non-trivial, but most conference verifiers at this
altitude present the same mechanism via the equation sequence alone.

### Closing paragraph (Tier 1 bypass; downstream consumers)
**Tag:** THESIS-ONLY. Bridging paragraphs do not transfer.

## §4.7 Four-Outcome Storage Decision

The four-outcome structure (Store, Deferred, Abstain, Discard) is the
most distinctive decision-logic contribution of the verification story,
because most comparable pipelines collapse Stage 6 to a binary threshold
on a composite. Paper treatment: one compact subsection with the cascade
equation, the decision-tree table, and the tau_train training-eligibility
gap foregrounded.

### Opening paragraph (Stage 6 + Stage 7 as single policy surface)
**Tag:** COMPRESS to Method §4.3 opener. One sentence: "Stage 6 emits
one of four outcomes (Store, Deferred, Abstain, Discard) from the
composite, the contradiction probability, and the max grounding; Stage 7
mechanically executes the implied memory and user-facing actions."

### Paragraph: Recoverability argument for four outcomes over binary
**Tag:** READY to Method §4.3. The argument that collapsing the middle
cases either pollutes Tier 1 with marginal answers or discards
retroverification-recoverable signal is the paper-essential motivation
for why the decision tree is not binary. One tight paragraph in the
paper, roughly three sentences.

### Paragraph: Contradiction veto plus Eq 4.36
**Tag:** READY to Method §4.3. The asymmetry argument (contradiction
cannot be arithmetically outweighed by a high composite, therefore
p_contra is vetoed rather than averaged) is a design commitment worth
preserving in the paper. The 0.30 threshold is an inline parenthetical;
the ensemble-max-versus-min rationale (why 0.30 is a weak threshold on
single-model contradiction but the correct threshold on ensemble-max
contradiction) compresses to one clause.

### Equation 4.36 (contradiction veto)
**Tag:** READY. Transfers verbatim; the veto form is compact and the
paper reader needs to see the threshold value inline.

### Paragraph: Store and Deferred thresholds plus Eq 4.37
**Tag:** READY to Method §4.3. The cascade form (one inequality covers
both bands) is the cleanest way to state the rule in a paper. The
interpretation of the middle band as "plausibly correct but has not
cleared the commit bar" is paper-essential because it directly motivates
the retroactive re-verification section. The tuning-knob discussion
(tightening the 0.45-to-0.65 gap collapses Deferred into Store) can
compress to half a sentence or move to APPENDIX.

### Equation 4.37 (Store / Deferred cascade)
**Tag:** READY. The two-branch cascade form is the compact way to state
the two thresholds in one equation.

### Paragraph: Abstain rule plus Eq 4.38
**Tag:** READY to Method §4.3. The Abstain-versus-Discard distinction is
genuine content, not bookkeeping: the paper reader must understand that
CAEM emits explicit refusals in the low-composite low-grounding regime
and suppresses outputs in the low-composite high-grounding regime. Keep
the rule and the user-facing consequence; the threshold-value
justification compresses to a parenthetical.

### Equation 4.38 (Abstain rule)
**Tag:** READY. Transfers verbatim; the conjunction form (composite low
AND grounding low) is the operationally meaningful statement.

### Paragraph: Discard as the default (two scenarios)
**Tag:** COMPRESS to one sentence in Method §4.3. The distinction
between contradiction-triggered Discard and no-grounding-triggered
Discard is useful for Chapter 5's error analysis (generation failure
versus retrieval failure) but papers at this altitude state it only if
the results section later appeals to it. Keep as a single sentence in
the paper: "Remaining cases fall through to Discard, covering both
contradiction-triggered and generation-without-recoverable-evidence
failures."

### Table 4.4 (four-outcome decision tree)
**Tag:** READY to Method §4.3 as *the* decision-logic artefact for the
paper. Three-column table (Outcome / Condition / Stage 7 action) is the
compact way to state the cascade; reader can skim it and get the whole
policy surface in one glance. Lift verbatim, possibly with a slightly
trimmed Condition column (e.g. drop the inline tau values from the
condition text since the caption or a preceding sentence can state
them).

### Paragraph: Stage 7 consequences and training eligibility
**Tag:** READY to Method §4.3. The tau_train = 0.75 / tau_store = 0.65
gap is one of the most paper-worthy architectural commitments in the
whole chapter: it encodes the claim that Tier 1 retrieval quality and
training-pool quality are different bars and that conflating them would
either starve training or over-serve Tier 1 from marginal memory. This
is the kind of crisp design argument conference papers reward. Keep the
gap explicit; the Stage 7 action list (what Store/Deferred/Abstain/
Discard each do mechanically) compresses to one sentence because
Table 4.4 already carries that content.

### Algorithm 4.4 (Stage 6 decision tree + Stage 7 action)
**Tag:** READY to Method §4.3 (by default) or APPENDIX if page budget
is tight. The algorithm encodes the strict evaluation order (veto
before composite, Abstain before default Discard, training-eligibility
after memory write) which is the algorithmic form of the recoverability
argument and cannot be reconstructed from the equations and the table
alone. Prefer keeping in body; the equations fix the thresholds but the
cascade structure lives in the pseudocode.

### Closing paragraph (cycle-level diagnostic distribution)
**Tag:** THESIS-ONLY for the paragraph; the (cycle x outcome)
distribution figure it forward-refers to is READY to Results §6. The
paragraph itself is a bridge to Chapter 5 and does not transfer, but the
diagnostic claim it sets up (healthy trajectories shift Deferred->Store
across cycles; pathological trajectories shift Store->Discard) is one of
the single strongest forward hooks in the whole method chapter and
should be the motivating caption for the outcome-distribution figure in
the paper's Results section.

## §4.8 Retroactive Re-Verification

This is the mechanism that enables the dynamics angle (Candidate paper
angle 2). The upward-only update rule is the distinctive design
commitment; it is the reason the memory compounds quality across cycles
rather than drifting under noisy re-verification. Paper treatment: one
short dedicated paragraph with the cascade equation and one forward
reference to the cycle-over-cycle (n_up, n_rm) diagnostic in Results.

### Opening paragraph (Stage 8 role; cycle-boundary versus per-query)
**Tag:** COMPRESS to one sentence in Method §4.5. "A cycle-boundary
sweep recomputes the composite for every stored episode under the
fine-tuned model; this is distinct from the continuous retrieval-
feedback update and is the only channel that reconciles stored quality
estimates with the updated model."

### Paragraph: Motivation (stored composites are snapshots, not invariants)
**Tag:** COMPRESS to one sentence in Method §4.5. "Three of the nine
signals read the model's internal state directly and drift with every
fine-tune, so stored composites are model-relative snapshots that the
retroactive pass must refresh." The signal-by-signal breakdown
(u_internal drifts with weights; s_avg, h_norm drift with decoding
distribution; NLI signals depend on regenerated answers) is APPENDIX
material; papers need only the headline claim.

### Paragraph: Three-way branch (prune / promote / keep) plus Eq 4.39
**Tag:** READY to Method §4.5. The cascade equation is the single
compact statement of the retroverify mechanism and should transfer
verbatim. The scalar-plus-full-record update distinction (promotion
overwrites nine signals plus decision plus early-exit; kept entries
only flip the retroverified flag) compresses to one clause in the
paper.

### Equation 4.39 (three-way branch cascade)
**Tag:** READY. The piecewise form makes the prune / promote / keep
distinction visible at a glance and the tau_retro = 0.50 threshold is
inline. Transfer verbatim.

### Paragraph: Upward-only update rationale
**Tag:** READY to Method §4.5. This is the paper-essential content of
the whole section. The argument that stochastic-decoding noise in a
symmetric re-verification pass could demote stable knowledge on a coin
flip, and that upward-only promotion plus categorical prune at 0.50 is
the correct asymmetry, is a design commitment that no reviewer will
have seen elsewhere in the literature. Keep the rationale tight:
three sentences. The tau_retro > tau_defer ordering (0.50 > 0.45) is a
single clause and worth preserving because it encodes the "stricter
on re-verification than on first commit" claim.

### Paragraph: Nine-signal refresh on promotion
**Tag:** COMPRESS to one sentence in Method §4.5. "On promotion, the
full nine-signal record plus p_contra and the decision string are
overwritten, so downstream calibration and decision-distribution
diagnostics always see the current verifier's view of promoted
episodes." The asymmetry (only promotion refreshes signals; kept
entries retain their storage-time record) is APPENDIX material, worth
at most a parenthetical in the paper.

### Paragraph: Retroverify versus retrieval feedback (disjoint channels)
**Tag:** READY to Method §4.5 in compressed form. The "usage quality
versus model-relative quality" distinction is genuinely non-obvious and
answers a reviewer question that would otherwise come up unprompted
("why two update mechanisms for the same field?"). Compress to one
tight sentence: "Retrieval feedback tracks usage quality via a
per-query EMA; retroverify recomputes model-relative quality via a
per-cycle verifier re-score, and the two can disagree (the current-
model composite takes precedence)."

### Paragraph: Abort handling (retroverify skipped on rollback)
**Tag:** APPENDIX. The skip-on-abort rule is operational compute
economy plus a reporting-convenience argument (aborted cycles write
n_up = n_rm = 0 to distinguish them from genuinely inert cycles). One
sentence suffices in a paper, and only if Chapter 5 actually shows an
aborted cycle. Otherwise APPENDIX.

### Algorithm 4.5 (Stage 8 retroverify loop)
**Tag:** APPENDIX (by default) or READY if the paper targets a
systems-oriented venue. The three-way branch is compact enough that
the equation plus one sentence about the evaluation order (prune-below
checked before promote-above so a below-threshold entry cannot be
promoted first and then pruned on the same pass) carries the algorithm
in paper prose. The pseudocode's snapshot-of-keys detail
(dict-shrinking-during-iteration) is pure implementation robustness.

### Closing paragraph (dynamics hook; (n_up, n_rm) trajectory)
**Tag:** READY to Results §6 as the motivating caption for the
retroverify-dynamics figure. The paragraph itself is a bridge and is
THESIS-ONLY, but the healthy-versus-pathological trajectory contrast
(n_up grows, n_rm stays small = compounding knowledge; n_rm grows,
especially concentrated on one earlier cycle = drift) is the single
most paper-relevant claim for the dynamics angle. The forward
reference to the Candidate paper angles list in this file is
internal-only and will not appear in the thesis or the paper.

## §4.9 Deferred-Entry Reconsideration

This section was added after the compile-clean revision of §4.8 to close a
thesis-code gap: the Stage 6 decision tree promises that DEFERRED entries
get a second chance to enter memory, but the original implementation
dropped them silently. The new section specifies the bounded FIFO buffer
with TTL and the cycle-boundary reconsideration pass that promotes,
drops, or requeues each held entry. The pass is parallel in spirit to
retroverify (both are cycle-boundary memory-update passes, both are
abort-guarded, both expose per-cycle counter diagnostics) but operates
on uncommitted held entries rather than committed memory, and uses a
three-way rule that mirrors retroverify's (prune, promote, keep) with
different semantics. Paper treatment: one short paragraph in Method
§4.5 after the retroverify paragraph, with a single-line statement of
the three-way rule and a forward pointer to the (n_pr, n_dr, n_kp)
diagnostic in Results.

### Opening paragraph (why the buffer exists; gap in retroverify)
**Tag:** COMPRESS to one sentence in Method §4.5. "A separate cycle-
boundary pass sweeps a bounded buffer of entries deferred at Stage 6
and promotes those that the updated verifier would now mark Store,
drops those that exceed the TTL, and requeues the rest." The framing
that retroverify cannot pick up deferred entries because it iterates
committed memory only is THESIS-ONLY motivation; the paper version
states the mechanism directly without the gap-diagnosis framing.

### Paragraph: Why the buffer exists (deferred band value argument)
**Tag:** APPENDIX. The four-band composite axis collapsing to three
outcomes if deferral is behaviourally a discard is an internal design-
commitment argument; the paper version of this claim is implicit in
the band definition of §4.6 and does not need to be re-argued here.

### Paragraph: Bounded FIFO with TTL (size bound and age bound)
**Tag:** APPENDIX. Buffer capacity N_buf = 10,000 and TTL T = 2 are
implementation-detail constants; the paper states them in the
hyperparameter table and does not need the O(N_buf) argument or the
self-cleaning-toward-relevance argument. Both are systems rationale
that does not survive compression.

### Paragraph: Three-way rule (promote / drop / keep) plus Eq 4.40
**Tag:** READY to Method §4.5. The three-way cascade is the single
compact statement of the reconsideration mechanism and transfers
verbatim. The asymmetry argument (retroverify defaults to prune-if-
below, reconsideration defaults to keep-if-not-promoted) is one
parenthetical clause in the paper.

### Equation 4.40 (promote / drop / keep three-way branch)
**Tag:** READY. The piecewise form makes the three-way distinction
visible at a glance and the TTL T = 2 threshold is inline. Transfer
verbatim.

### Paragraph: Novelty filter on promotion
**Tag:** COMPRESS to one clause in Method §4.5. "Promoted entries pass
through the Stage 7 novelty filter, so the memory's near-duplicate
guarantee is invariant across the two entry points (direct commit and
delayed reconsideration)." One sentence suffices. The invariance
framing is the paper-essential content; the fresh-verifier-record
detail is APPENDIX.

### Paragraph: Scheduling and abort handling (Step 8b; skip on rollback)
**Tag:** APPENDIX. The ordering (8b strictly after 8a so that the
novelty check sees post-retroverify memory) and the abort-skip
semantics are operational; the paper version states only that the
pass runs at cycle boundaries and is skipped on aborted cycles.

### Algorithm 4.6 (Stage 8b reconsideration pass)
**Tag:** APPENDIX (by default) or READY if the paper targets a
systems-oriented venue. The three-way cascade in the equation plus
one sentence about the evaluation order (promote-first, then TTL-
check, then requeue) carries the algorithm in paper prose. The
drain-and-refill pattern (iterate over a stable snapshot rather than
the live deque) is implementation robustness.

### Closing paragraph (diagnostic hook; (n_pr, n_dr, n_kp) trajectory)
**Tag:** READY to Results §6 as a secondary caption alongside the
retroverify-dynamics figure. The healthy-versus-pathological trajectory
contrast (n_pr grows modestly, n_dr near TTL-turnover = deferred band
is yielding its lifecycle; n_dr dominates = composite axis's four-band
design is not working as intended) is the paper-relevant claim. The
paragraph itself is a bridge and is THESIS-ONLY.

## §4.10 Self-Improvement Loop

This is the section that carries the "self-improvement" half of the
thesis title. The core argument is that fine-tuning under a strict
quality filter (tau_train = 0.75), a conservative regulariser (L2
anchor as an EWC approximation), and a tripped-wire capability guard
(MMLU retention ratio) allows weight-level knowledge accumulation
without catastrophic forgetting. Paper treatment: one subsection in
Method §4.6 with the four equations, one forward hook to the ablation
that isolates the memory's effect from fine-tuning's effect.

### Opening paragraph (Stage 8 closes the loop; safety envelope framing)
**Tag:** READY to Method §4.6 (opener). The framing that three
deliberately-conservative choices (strict filter, uniform L2, OOD
guard) together form a safety envelope inside which compounding is
allowed to operate is the single-sentence motivation for the whole
self-improvement mechanism and the cleanest way to open the paper
subsection. Keep in one paragraph.

### Paragraph: Training-pool curation plus Eq 4.40 (tau_train filter)
**Tag:** READY to Method §4.6. The tau_train = 0.75 > tau_store = 0.65
gap is one of the most paper-worthy architectural commitments in the
method chapter (distinct quality bars for Tier 1 retrieval versus
training data), and was already flagged as such in the §4.7 tagging.
This section re-states it explicitly, which is redundant in the thesis
but exactly the content that should survive compression to the paper.
The reasoning_chain-as-target rationale ("teaches how to reason, not
what to answer") is a genuine contribution over standard QA fine-
tuning and deserves one sentence in the paper.

### Equation 4.40 (training-pool set-builder)
**Tag:** READY. Compact filter rule; transfers verbatim.

### Paragraph: Episode-general mix plus Eq 4.41 (mix-size rule)
**Tag:** COMPRESS to one sentence in Method §4.6. "Each training batch
is 90% verified-episode reasoning chains and 10% general QA pairs to
buffer against topic drift." The ratio-computed-relative-to-episode-
count detail (so episodes stay at 90% of the mix regardless of pool
size) is APPENDIX material; papers state the final ratio and move on.

### Equation 4.41 (mix-size formula)
**Tag:** APPENDIX. The formula detail (ceiling function, min with
availability) is an implementation choice; the paper states the 90/10
ratio directly.

### Paragraph: L2 anchor plus Eq 4.42 (fine-tuning loss)
**Tag:** READY to Method §4.6. The L2-as-EWC-approximation argument
is a named design decision that the reviewer community will
recognise as deliberate (uniform Fisher assumption, 2x cost saving),
and the Kirkpatrick 2017 citation is load-bearing. The ablation-hook
sentence (full EWC as a Chapter 5 ablation) is a good forward
reference that also lets the paper cite the ablation table without
spending paper words justifying the approximation. Keep in one tight
paragraph.

### Equation 4.42 (fine-tuning loss with L2 anchor)
**Tag:** READY. The single equation that defines the fine-tuning
objective of the whole method. Transfer verbatim; lambda = 0.01 stays
inline.

### Paragraph: OOD forgetting guard plus Eq 4.43 (retention ratio)
**Tag:** READY to Method §4.6. The choice of MMLU as the abort trigger
(not in-distribution QA, continual-learning literature, dual-purpose
with CES RET axis) is the distinctive design decision for this
section. The continual-learning argument (catastrophic forgetting
manifests as loss of unrelated capability rather than erosion of the
target task) is genuinely non-obvious and addresses a reviewer question
that would otherwise come up unprompted ("why not TriviaQA as the
guard?"). The dual-purpose argument (one MMLU pass serves as both
rollback trigger and reported retention metric) is a compute-economy
detail worth one sentence.

### Equation 4.43 (retention ratio and abort condition)
**Tag:** READY. Compact statement of the guard; transfer verbatim.
rho_min = 0.93 is inline.

### Paragraph: Rollback and abort semantics
**Tag:** COMPRESS to one sentence in Method §4.6. "Aborted cycles
restore theta_prev and skip retroverify; memory is not rolled back."
The asymmetry argument (memory accumulates across aborts but weights
do not) compresses to a parenthetical. APPENDIX material for the full
rationale if space is tight.

### Paragraph: Checkpointing and multi-cycle trajectories
**Tag:** COMPRESS / APPENDIX. The independent-checkpoint-per-cycle
rule is engineering detail (APPENDIX). The four per-cycle quantities
reported in Chapter 5 (training-pool size, training loss, retention
ratio, retroverify counters) COMPRESS to the paper's Experimental
Setup §5 as the "reported per cycle" list, one sentence.

### Algorithm 4.6 (Stage 8 self-improvement cycle)
**Tag:** READY to Method §4.6 (by default) or APPENDIX if page budget
is very tight. The cycle order (snapshot theta_prev -> MMLU pre-probe
-> fine-tune -> MMLU post-probe -> abort-or-commit -> retroverify)
is strict and encodes the abort-and-rollback logic that cannot be
reconstructed from the equations alone. Include in body for all non-
8-page venues; fall back to APPENDIX only if cutting.

### Closing paragraph (ablation disaggregation)
**Tag:** READY to Method §4.6 in compressed form. The disaggregation
argument (the trajectory joint-configures memory-driven improvement
and fine-tuning-driven improvement, and ablations isolate each) is a
paper-essential framing for Chapter 5's headline figure. Two sentences
in the paper: "Cycle-over-cycle dynamics reflect two coupled signals
(the growing training pool and the retroactive re-verification);
ablations disabling each channel in turn isolate their individual
contributions."

## §4.11 Implementation

This is the reference section that pins the software and hardware
substrate. Most of it is APPENDIX for a conference paper (no venue
wants a page of library versions in the main body), but the
hyperparameter table is potentially APPENDIX-but-critical because any
reproducibility-oriented venue will want it.

### Opening paragraph (three principles)
**Tag:** COMPRESS to one sentence in Paper §4 Implementation Details or
Appendix A. The three principles (centralise config, separate concerns,
pin seeds) are scaffolding for the reader; the paper communicates them
implicitly by simply being reproducible.

### Software stack paragraph
**Tag:** APPENDIX. The library list (PyTorch, HF Transformers,
sentence-transformers, faiss-gpu with CPU fallback, numpy, scipy,
scikit-learn) is pure reference material. One sentence in Paper
Implementation Details suffices: "Implemented in Python 3.10 with
PyTorch, HuggingFace Transformers, sentence-transformers, faiss-gpu,
and scikit-learn."

### Pretrained components paragraph + Table 4.5
**Tag:** COMPRESS paragraph, READY table (or Paper footnote).
The table is genuinely paper-useful: readers want the three
checkpoints, parameter counts, and which one is trainable at a glance.
Compress to a single Paper Methods sentence naming the three
checkpoints (Flan-T5-Large, all-mpnet-base-v2, RoBERTa-Large-MNLI) with
their citations; keep the table itself in Paper Appendix A.

### Hardware envelope paragraph
**Tag:** APPENDIX. RTX 4090 / 24 GB VRAM / 64 GB RAM / bf16 / theta_prev
on GPU is reproducibility-facing but not paper-material. The key paper
detail (fine-tuning fits on a single 24 GB GPU because L_2 anchor state
is on-device) is already implicit in the methodology.

### Module layout paragraph
**Tag:** THESIS-ONLY. The seven-submodule breakdown is a thesis
orientation device, not paper content. Conference readers do not need
a map of the repository.

### Configuration-discipline paragraph ([LIT] / [DES] / [CAL] categories)
**Tag:** COMPRESS to Paper Implementation Details. The three-category
provenance tagging is a potentially interesting methods contribution
(explicit separation of which constants are literature-fixed versus
design choices versus empirically calibrated) and can be landed in one
sentence: "Hyperparameters are partitioned into literature-derived,
design-choice, and empirically calibrated categories (Appendix A)."
The paragraph's citations to Gal 2016, Wang 2023, Farquhar 2024, Guo
2017 are already established elsewhere in the paper.

### Table 4.6 (consolidated hyperparameter reference)
**Tag:** APPENDIX-but-critical. This is the single place the paper
audit trail converges: every numeric constant with its value, category,
and role. For reproducibility-oriented venues this must appear
verbatim in Paper Appendix A. For strict-8-page venues without an
appendix, COMPRESS to a smaller table of only the top-tier constants
(routing thresholds, verifier thresholds, u_stored weights, L_2
lambda, rho_min) and push the rest to a supplementary materials file.

### Reproducibility paragraph (seeds, config-JSON, benchmark manifests,
CUDA caveat)
**Tag:** COMPRESS to one paper sentence: "Runs are reproducible via a
single seed, serialised configuration, and pinned benchmark manifests;
residual non-determinism is attributable to low-level CUDA
floating-point ordering and is smaller than the cycle-level effects
reported in §5." The CUDA caveat is paper-honest and worth keeping.

## §4.12 Chapter Summary
**Tag:** THESIS-ONLY. Two recap paragraphs plus a forward-pointer
paragraph to Chapters 5 and 6. Does not introduce new mechanism; it
re-indexes the chapter against Chapter 3's FR list. Paper version is
covered by the method-section closer and the transition into
Experimental Setup, so this section is omitted from the paper draft.

---

# Chapters 1, 2, 5, 6: placeholders

## Abstract (revised 2026-04-17 to match current mechanism)
`core/abstract.tex` rewritten end-to-end. Replaced the old "three
innovations / multi-layer verification / 85-90\% accuracy across all
three methods / reduce by 20-30\% / HotpotQA+StrategyQA+FEVER+TruthfulQA"
framing with the current mechanism: four-outcome storage decision;
nine-signal post-hoc verifier in three families (internal, sample-set,
external grounding); combined-score three-tier router with independent
safety override on pre-routing confidence; cycle-boundary retroverify +
deferred-reconsider + $L_2$-anchored fine-tune + MMLU retention guard
with weight rollback but memory preservation; seven-benchmark factual-QA
panel (FEVER, TriviaQA, NQ, HaluEval, TruthfulQA, StrategyQA, ARC-C)
with MMLU as OOD retention control. Confident-error-rate framing
applied per Option~A: directional claim only, specific number deferred
to results chapter. Keywords left unchanged.

## Chapter 1, Introduction (revised 2026-04-17 to match current mechanism)
Rewritten end-to-end: §1.1 closes on the three current architectural
pillars (verified memory with four-outcome decision; confidence-aware
three-tier routing with independent safety override; constrained self-
improvement with L2 anchor plus retention guard); §1.2 rationale reflects
the 7-benchmark factual-QA panel plus MMLU retention control; §1.3
problem statement references current mitigation mechanisms; §1.4
objectives (all seven) rewritten to current spec including pre-routing
$u_{\text{pre}}$, combined score $\mathcal{S}$, nine-signal verifier,
retroactive re-verification with deferred-entry reconsideration, L2
anchor + retention guard + weight rollback (memory preserved); §1.5
methodology now describes all 8 stages explicitly; §1.6 benchmark list
matches the 7+1 panel; fig:architecture and fig:experimental-workflow
labels and captions updated to 8-stage pipeline, four-outcome decision,
and cycle-boundary retroverify + defer-reconsider. The "20-30%
hallucination reduction" quantitative claim was softened to a directional
claim (Option A) with the specific value deferred to Chapter 5, measured
as confident-error rate on the primary panel versus zero-shot Flan-T5-
Large. Will map mostly to Paper Introduction plus Paper Abstract. Extract
motivation, problem statement, contributions list, paper roadmap. Thesis-
chapter framing and reading-order paragraphs: THESIS-ONLY. Figure 1.1
(architecture) and the architecture caption in §1.5: READY (modulo a
minor TikZ-wiring imperfection where the "SAFETY override" box is drawn
downstream of Tier~2 generation rather than pre-routing; the label and
caption state the correct semantics — full rewire deferred).

## Chapter 2, Literature Review (revised 2026-04-17 to match current mechanism)
§2.1.6 closing updated to reflect the $L_2$ anchor as EWC approximation
plus the 90/10 batch mix plus the hard MMLU retention guard with floor
$\rho_{\min} = 0.93$ and memory-preserving weight rollback. §2.2.2
(verification methods) closing rewritten to describe how the four
methods map onto CAEM's nine-signal composite $u_{\text{stored}}$ in
three families; specific accuracy number (85-90%) dropped per Option A
and deferred to Ch 5. §2.2.3 (confidence estimation) closing rewritten
to describe the two-stage architecture (cheap pre-routing $u_{\text{pre}}$
fusing token probability with encoder convergence, temperature-scaled,
on every query; nine-signal post-hoc verifier $u_{\text{stored}}$ only
after generation). §2.2.4 (memory) closing rewritten to name the three
CAEM extensions of RAG (combined routing score with safety override,
four-outcome storage decision with deferred buffer, value-score pruning).
§2.2.5 (self-improvement) rewritten to describe $L_2$ anchor, stricter
training threshold $\tau_{\text{train}} > \tau_{\text{store}}$, and the
asymmetric rollback rule (weights can roll back; memory never does).
Specific "60% to 87%" progression number dropped per Option A. §2.2.6
(benchmarks) fully replaced in the prior editing pass: HotpotQA
removed; FEVER, TriviaQA, NQ, HaluEval, TruthfulQA, StrategyQA, ARC-C
added as primary panel; MMLU added as the out-of-distribution retention
control rather than a target benchmark. §2.3 (summary) fully rewritten
into six topical paragraphs mirroring §§2.2.1--2.2.6 and the current
mechanism names. Compressed roughly 8:1 into Paper §3 Related Work.
Thesis background-on-transformers and background-on-RAG sections:
THESIS-ONLY. Direct positioning against Farquhar 2024, Kirkpatrick
2017, Kadavath 2022, and RAG baselines: READY.

## Chapter 5, Results and Analysis (to be written)
Will be the *densest* source of paper-ready material. Every results
table, every ablation column, every cycle-over-cycle figure is a
candidate. Tag on arrival. Decision rule: any result that supports the
chosen paper angle (see Candidate paper angles) is READY; collateral
results that would dilute the narrative are APPENDIX.

## Chapter 6, Conclusion (to be written)
Mostly THESIS-ONLY. Future work becomes one sentence in paper Conclusion.
Contribution summary: discard (the paper abstract does this job).

---

# Running to-do

- [ ] Revisit the angle choice after Chapter 5 first-pass results land.
- [ ] When §4.4 through §4.12 are drafted, add per-section rows above and
  tag each figure, table, algorithm separately.
- [ ] Chapter 5 ablation table: flag which lesions are mechanism-
  validation (belong in paper) versus negative-result bookkeeping
  (APPENDIX).
- [ ] Decide VER axis fate before paper draft (still placeholder 0.5 at
  thesis stage): either compute real labels or drop the axis.
- [ ] Check whether Example 4.1 box contains any numerical claims that
  should migrate as prose to Discussion (the Tier-1 cheap-path
  illustration of "stored answer returned without generation" is a
  concrete argument worth making in prose).
- [x] Chapters 1 and 2 revised 2026-04-17: per-section rewrite against
  current mechanism (see placeholder blocks above); full paper-ready
  tag pass still pending.
- [ ] Chapter 5 benchmark panel discrepancy: Ch 5 §5.1.1 currently lists
  6 benchmarks (FEVER, TriviaQA, NQ, TruthfulQA, StrategyQA, ARC-C);
  Ch 1 and Ch 2 now commit to a 7-benchmark panel plus MMLU retention
  control (HaluEval added). Ch 5 benchmark list should be expanded to
  include HaluEval before the experiment phase.
- [ ] Ch 1 figure 1.1 has a minor TikZ wiring imperfection (the
  "SAFETY override" box is drawn downstream of Tier~2 generation, not
  before routing). Label text and caption state the correct semantics;
  full rewire deferred to a later pass.
- [x] **Citation debts accumulated during Chapter 4 drafting** (flagged
  by Aksan on 2026-04-17; resolved 2026-04-17 during the Ch 2 rewrite):
  - MMLU bib entry (`hendrycks2021measuring`, Hendrycks et al. 2021,
    "Measuring Massive Multitask Language Understanding") now present
    in references.bib; Ch 5 and Ch 2 §2.2.6 both cite it correctly.
  - `brier1950verification`, `lopezpaz2017gradient`, and
    `mcnemar1947note` (previously flagged with "to be added" footnotes
    in Ch 5) now present in references.bib; the three footnotes
    reminding us to add them have been removed from Ch 5.
  - New Ch 2 §2.2.6 benchmark citations also added to references.bib:
    `joshi-etal-2017-triviaqa`, `kwiatkowski-etal-2019-natural`,
    `li-etal-2023-halueval`, `clark2018think`.
  - Ch 5 §5.1.1 citation typos fixed: TriviaQA was cited as
    `JMLR:v25:23-0870` (Flan-T5) and Natural Questions was cited as
    `10.1162/tacl_a_00370` (StrategyQA); both corrected to their
    proper keys.
  - `geva-etal-2021-aristotle` was previously flagged as needing an
    entry; this was actually the same paper as the existing
    `10.1162/tacl_a_00370` entry. Ch 5 §5.1.1 updated to use the
    existing key; no new bib entry added.
- [ ] Residual §4.10 hyperparameter and design-choice references
  (AdamW, gradient clipping norm, mixed-precision training): verify
  citations are appropriate once Ch 5 results land.
- [ ] Audit §4.9 (deferred-entry reconsideration) for any non-trivial
  claims that should carry citations once related work on deferred-
  evaluation queues is surveyed.
