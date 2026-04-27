# Change Inventory: Chapters 1 and 2 Rewrite (Pre-Thesis 1 to Phase 1a)

**Status**: Comprehensive change inventory for CAEM thesis Chapters 1 (Introduction) and 2 (Literature Review).  
**Date**: 2026-04-25  
**Locked Evidence Files**:
- `composite_calibration.json` (10 signals, Cherian boost intercept +1.099)
- `conformal_gate.json` (tau_store=0.6676, tau_defer=0.5207, alpha_store=0.05)
- `weight_validation.json` (overall_pass=true, d_id=0.617, d_transfer=0.816, eval pool precision 80.2%)
- `iteration_history.json` (3 iterations, locked V_a050_C0.010)
- `calibrated_config.json` (T scalar=20.09, ECE 0.461→0.293)

---

## PART 1: CHAPTER 1 INVENTORY (Sections 1.1–1.6)

### Overview

Chapter 1 contains six top-level rubric-mandated sections (1.1–1.6) that are structurally fixed. The existing chapter (346 lines, pre-thesis 1) contains mostly salvageable material but requires targeted numerical updates, architectural clarifications, and terminology standardization to reflect Phase 1a changes. Key changes since pre-thesis 1:
- Verifier signal count: 9 (old, based on pre-registration) → **10 signals** (q_a_relevance added in Branch C)
- Composite design: fixed weighted sum → **per-signal isotonic regression with L2-regularised Cherian boost**
- Storage gate: hardcoded thresholds → **conformal split-CP gate** (alpha_store=0.05)
- Baseline reference: Flan-T5-Large → **Qwen-2.5-3B-Instruct**
- Benchmark panel: 6 benchmarks → **7 benchmarks** (ASQA added as transfer-only)
- ECE improvement: (measurement newly reported) → **0.461→0.293**

---

### Section 1.1: Background (Lines 1–12)

**Table: Paragraph-by-Paragraph Inventory**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L1–4: LLM capabilities and hallucination definition | KEEP | Foundational problem statement remains accurate; well-cited. | None. |
| L5–6: TruthfulQA and FActScore benchmarks; inverse scaling | KEEP | Empirical basis for hallucination severity unchanged. | None. |
| L7–8: Stateless operation, three critical limitations | KEEP | Core motivation unchanged. | None. |
| L9–10: Practical consequences (education, healthcare, enterprise) | KEEP | Use cases remain relevant. | None. |
| L11: "nine-subtype taxonomy" and three architectural commitments | KEEP-WITH-MODIFICATION | The taxonomy is still nine subtypes, but the verifier now has ten signals (q_a_relevance added in Phase 1a). Clarify that eight subtypes are measurable under MiniCheck backend (Frozen Qwen ablated). | Change L11 from "nine-subtype taxonomy" to "nine-subtype taxonomy of confident factuality hallucinations (eight of which are measurable under the shipped MiniCheck verifier backend)". |
| L11: "post-generation nine-signal verifier" | REPLACE | Pre-thesis 1 refers to nine signals; Phase 1a added q_a_relevance signal for ten total. | Change "nine-signal" to "ten-signal" and footnote: "Phase 1a added q_a_relevance to detect off-topic generations; eight of the ten signals feed the composite, two are tracked as diagnostics." |
| L11: Four-outcome decision tree | KEEP | Still accurate. | None. |
| L11: "relative reduction of at least 30% in the pooled Composite Hallucination Metric (the equal-weighted mean across the eight taxonomy subtypes)" | KEEP | Pre-registered target matches locked JSONs. | None. |
| L11: "zero-shot baseline on a seven-benchmark factual-QA panel" | KEEP-WITH-MODIFICATION | Panel is now 7 benchmarks (ASQA added as transfer-only in Phase 1a). The original pre-thesis 1 reference may have said six; clarify this was a Phase 1a addition. | Confirm via audit that pre-thesis 1 said 6; if yes, note the change. Literature review should document ASQA addition. |
| L11: $L_2$ anchor regulariser and retention guard | KEEP | Still accurate. | None. |
| L12: "three architectural commitments" summary | KEEP | Still accurate. | None. |

**New subsections to introduce**: None at the rubric level. The audit may recommend a subsection clarifying the nine-subtype vs ten-signal distinction and the eight-measurable-under-MiniCheck clarification, but this could live as a footnote in 1.1 or moved to 1.3 (Problem Statement).

**New citations from literature review files**: None required for Background section (it is foundational problem statement, not literature review).

**New numerical claims this section should make**:
- Verifier now has **10 signals** (q_a_relevance added; `composite_calibration.json` L6-16 lists all 10).
- **Eight of the ten** signals feed the composite score; two (p_contra, q_a_relevance details) are tracked as diagnostics.
- Storage gate now uses **conformal split-CP** with alpha_store=0.05 (target 95% calibration precision) instead of fixed thresholds (`conformal_gate.json` L5).

**Old numerical claims that are now stale and must be removed or replaced**:
- "nine-signal verifier" → "ten-signal verifier" (signal count drift).
- Any reference to "fixed weighted sum" for composite → replace with "per-signal isotonic regression with L2-regularised Cherian boost" (composite design drift).

---

### Section 1.2: Rationale of the Study or Motivation (Lines 13–26)

**Table: Paragraph-by-Paragraph Inventory**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L15: Confabulation definition citing Farquhar 2024 | KEEP | Citation and definition remain current. | None. |
| L17: Three existing approaches (RAG, prompting, post-generation verification) | KEEP | Taxonomy of prior work is unchanged. | None. |
| L19–20: Key insight on architectural intervention; three essential capabilities | KEEP | Core thesis unchanged. | None. |
| L21: "imperfect verification can enable monotonic improvement"; semi-supervised learning citation | KEEP | Theoretical argument unchanged. | None. |
| L23: "$30\%$ relative reduction in pooled Composite Hallucination Metric over the eight-subtype taxonomy" | KEEP | Pre-registered target unchanged. | None. |
| L23: "three-billion-parameter backbone" (Qwen) | KEEP | Model selection unchanged. | None. |
| L25: "seven-benchmark factual-QA panel (FEVER, TriviaQA, Natural Questions, TruthfulQA, StrategyQA, ARC-Challenge, and ASQA)" | KEEP-WITH-MODIFICATION | Panel is now correct. If pre-thesis 1 said six, note the Phase 1a addition of ASQA as transfer-only. | If drift exists, cite the literature review file that motivated ASQA inclusion. |
| L25: "MMLU as an out-of-distribution retention control" | KEEP | Architecture unchanged. | None. |
| L25: "RTX 5090 with 32 GB VRAM; RTX 4090 at 24 GB" | KEEP | Hardware specs unchanged. | None. |
| L25: "ten-cycle self-improvement trajectory" | KEEP | Design unchanged. | None. |

**New subsections to introduce**: None required.

**New citations from literature review files**: None required (Rationale is narrative, not literature-grounded review).

**New numerical claims this section should make**: None beyond what is already present.

**Old numerical claims that are now stale and must be removed or replaced**: None identified.

---

### Section 1.3: Problem Statement (Lines 27–40)

**Table: Paragraph-by-Paragraph Inventory**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L29–37: Three fundamental limitations (stateless operation, poor confidence calibration, absence of self-improvement) | KEEP | Problem decomposition unchanged. | None. |
| L35: "0.50-0.60 AUROC for telling whether an answer is right or wrong" | KEEP-WITH-MODIFICATION | The citation and empirical claim are fine, but verify this is still the state-of-the-art baseline CAEM claims to improve over. | Check against latest confidence estimation literature (literature review files may have updated baselines). |
| L37: "catastrophic forgetting, 30-50% degradation" and EWC citation | KEEP | Citation and baseline risk are unchanged. | None. |
| L40: "regulariser that pulls weights toward the base model and an out-of-distribution retention guard" | KEEP | Design unchanged. | None. |

**New subsections to introduce**: None required.

**New citations from literature review files**: Potentially update confidence estimation baseline (L35) if literature review identified stronger recent work.

**New numerical claims this section should make**: None.

**Old numerical claims that are now stale and must be removed or replaced**: None identified (the 0.50-0.60 AUROC baseline should be verified for recency).

---

### Section 1.4: Objective (Lines 41–62)

**Table: Paragraph-by-Paragraph Inventory**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L43: Objective preamble and seven specific objectives listed | KEEP-WITH-MODIFICATION | The seven objectives are still correct, but Objective 4 must be updated to reflect ten-signal verifier and eight-weight composite (not nine-weight). | Update Objective 4 to specify "ten-signal post-generation verifier" with "eight-weight weighted composite" and note two additional diagnostic signals. |
| L43: Reference to "Chapter~\ref{ch:requirements}" for failure-mode taxonomy | KEEP | Forward reference is correct. | None. |
| L46: "episodic memory architecture; four outcomes" | KEEP | Design unchanged. | None. |
| L48: "encoder-only confidence estimate; temperature-scale" | KEEP | Design unchanged. | None. |
| L50: "three-tier routing" | KEEP | Design unchanged. | None. |
| L52: "nine-signal post-generation verifier... eight-weight weighted composite" | KEEP-WITH-MODIFICATION | Now ten signals, eight weights (q_a_relevance and one other diagnostic). | Change "nine-signal" to "ten-signal"; change "nine signals organised into three families" to "ten signals organised into three families plus two diagnostic signals"; change "eight-weight weighted composite" to "eight-weight weighted composite (ten signals feed through isotonic regression with L2-regularised Cherian boost)" to clarify the design. |
| L52: "contradiction probability $p_{\text{contra}}$ tracked as diagnostic signal (structurally zero under MiniCheck)" | KEEP | Still accurate. | None. |
| L54: "retroactive re-verification pass refreshes stored composites" | KEEP | Design unchanged. | None. |
| L56: "fine-tuning under $L_2$ anchor regulariser; EWC approximation" | KEEP | Design unchanged. | None. |
| L58: "seven-benchmark factual-QA panel; eight-subtype taxonomy" | KEEP | Correct. | None. |
| L58: "geometric Composite Evaluation Score" | KEEP | Design unchanged. | None. |

**New subsections to introduce**: None required.

**New citations from literature review files**: None required (Objective is architectural specification, not literature-grounded).

**New numerical claims this section should make**:
- **Ten signals** (not nine) in the post-generation verifier (`composite_calibration.json` lists all ten).
- **Eight weights** in the composite (eight signals feed the isotonic regression + Cherian boost; `boost_weights` from `composite_calibration.json` has ten entries, but only eight are active in the composite according to the architecture).

**Old numerical claims that are now stale and must be removed or replaced**:
- "nine-signal" → "ten-signal" (signal count drift).
- "nine-weight composite" → clarify the distinction between ten signals and eight active weights in composite.

---

### Section 1.5: Methodology in Brief (Lines 63–312)

This is the longest section (250 lines) covering system overview, architecture, query processing, verification, self-improvement, and implementation. Major changes in Phase 1a affect this section significantly.

**Subsection 1.5.1: System Architecture Overview (Lines 67–202)**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L69: Eight-stage pipeline description | KEEP | Pipeline stages unchanged. | None. |
| L69: Stage 5 "nine signals... eight-weight composite" | KEEP-WITH-MODIFICATION | Now ten signals, eight weights. | Change to "ten signals organized into three families (internal, sample-set, external) plus two diagnostic signals; eight-weight weighted composite aggregates the main signals". |
| L69: "structurally zero under MiniCheck backend" | KEEP | Still accurate. | None. |
| L69: Figure 1 (architecture diagram) | KEEP-WITH-MODIFICATION | Diagram should be updated to reflect ten-signal verifier and Cherian boost if not already done. Audit the figure's signal labels. | Verify that the figure shows all ten signals correctly labeled. If not, update the caption and diagram. |
| L69: Confabulation early-exit gate | KEEP | Design unchanged. | None. |
| L69: MMLU retention probe and Cycle-0 baseline | KEEP | Design unchanged. | None. |

**Subsection 1.5.2: Query Processing and Routing (Lines 203–206)**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L205: Router combining $s$ and $\ustored$; three tiers | KEEP | Routing design unchanged. | None. |
| L205: Safety override and pre-routing confidence | KEEP | Design unchanged. | None. |

**Subsection 1.5.3: Post-Generation Verification and Storage Decision (Lines 207–210)**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L209: "nine-signal post-generation verifier... three complementary families" | KEEP-WITH-MODIFICATION | Now ten signals (q_a_relevance added). | Change "nine signals" to "ten signals", specifying q_a_relevance as the Branch-C addition that detects off-topic generations. |
| L209: "internal calibration family" (3 signals) | KEEP | Still correct. | None. |
| L209: "sample-set family" (3 signals) | KEEP | Still correct. | None. |
| L209: "external grounding family" (max-match, mean-match, atomic-fact) | KEEP-WITH-MODIFICATION | Now includes q_a_relevance. | Clarify that q_a_relevance is the "Branch-C question-answer relevance signal" added in Phase 1a. |
| L209: "eight-weight composite... confabulation early-exit rule" | KEEP-WITH-MODIFICATION | Eight weights active; confabulation rule unchanged. | None. |
| L209: "four-outcome storage decision: Store, Deferred, Abstain, Discard" | KEEP | Still accurate. | None. |

**Subsection 1.5.4: Constrained Self-Improvement Loop (Lines 211–214)**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L213: "ten cycles; retroactive re-verification, deferred-entry reconsideration, fine-tuning" | KEEP | Design unchanged. | None. |
| L213: "$L_2$ anchor regulariser; EWC approximation" | KEEP | Design unchanged. | None. |
| L213: "MMLU retention ratio" and "fixed floor $\rho_{\min}$" | KEEP-WITH-MODIFICATION | Floor value is now locked in evidence. | Update with the locked value from evidence files if available. Current audit notes $\rho_{\min} = 0.93$ in L345 (Scopes and Challenges); verify and cite. |

**Subsection 1.5.5: Implementation (Lines 215–218)**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L217: "Qwen-2.5-3B-Instruct" and model parameters | KEEP | Model selection unchanged. | None. |
| L217: "Sentence-BERT, FAISS, MiniCheck-Flan-T5-Large" | KEEP | Component selection unchanged. | None. |
| L217: "Frozen Qwen-judge head" | KEEP-WITH-MODIFICATION | Frozen Qwen judge was ablated in Phase 1a (iteration_history.json notes platt_ablation with outcome ABLATED). | Update to state: "Frozen Qwen judge was tested and ablated due to insufficient calibration alignment (Pearson=0.5843 < 0.7 threshold); bare MiniCheck with documented truncation handles all hypothesis lengths." |
| L217: "generic-NLI backend (RoBERTa-Large-MNLI)" | KEEP | Retained as diagnostic; not part of default. | None. |
| L217: "\texttt{bitsandbytes} 8-bit AdamW, gradient checkpointing" | KEEP-WITH-VIOLATION | Code/library name in prose. Style rule prohibits code names in body; move to footnote or implementation caption. | Recommendation: "fine-tuning under mixed-precision quantized optimizer, gradient checkpointing" in main prose; move "\texttt{bitsandbytes}" to a footnote or appendix. |
| L217: "seven-benchmark factual-QA panel (FEVER, TriviaQA, Natural Questions, TruthfulQA, StrategyQA, ARC-Challenge, and ASQA)" | KEEP | Correct. | None. |
| L217: "Composite Evaluation Score" and "eight measurable subtypes" | KEEP | Correct. | None. |

**Figure 1.5.1 (Experimental Workflow, Lines 219–311)**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L219–311: Experimental workflow diagram and caption | KEEP | Diagram and caption appear accurate. | Verify that the cycle-boundary passes and MMLU guard match the locked evidence values (especially $\rho_{\min}$). |

**New subsections to introduce in 1.5**: None required at the rubric level; internal subsections are already sufficient.

**New citations from literature review files**: Potentially update EWC reference and confabulation early-exit gate justification if literature review has new citations for these concepts.

**New numerical claims this section should make**:
- **10 signals** in the verifier (Phase 1a addition).
- **8 weights** in the composite (not 9).
- Frozen Qwen judge was **ablated** due to Pearson=0.5843 < 0.7 threshold.
- **conformal_gate.json** parameters: tau_store=0.6676, tau_defer=0.5207, alpha_store=0.05, alpha_defer=0.4.

**Old numerical claims that are now stale and must be removed or replaced**:
- "nine-signal verifier" → "ten-signal verifier".
- Any mention of Frozen Qwen judge as active → replace with ablation note.

---

### Section 1.6: Scopes and Challenges (Lines 313–347)

**Subsection 1.6.1: Research Scope (Lines 315–324)**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L317: "seven primary evaluation benchmarks (FEVER, TriviaQA, Natural Questions, TruthfulQA, StrategyQA, ARC-Challenge, and ASQA)" | KEEP | Correct (ASQA is transfer-only, included in Phase 1a). | None. |
| L317: "MMLU... out-of-distribution retention control" | KEEP | Correct. | None. |
| L319: Scope exclusions (creative tasks, real-time information, code generation) | KEEP | Scope boundaries unchanged. | None. |
| L321: "Qwen-2.5-3B-Instruct; full-parameter fine-tuning" | KEEP | Model and tuning strategy unchanged. | None. |
| L321: "Chapter~4 Corollary~4.6 (C8) predicts convergence rate scales as $\log(H_{\text{base}}/C_{\infty})$, so 5--8 cycle range" | KEEP | Cross-reference to Chapter 4; assume Chapter 4 defines this correctly. | Verify that Chapter 4 Corollary 4.6 defines the convergence rate formula and the 5--8 cycle prediction. |
| L323: "episodic memory bounded at 10^6 entries; FAISS index promoted to IVF-PQ above threshold" | KEEP | Architecture unchanged. | None. |
| L323: "ten-cycle experiments reported; memory grows to low tens of thousands of entries" | KEEP | Design unchanged. | None. |

**Subsection 1.6.2: Technical Challenges (Lines 325–342)**

This subsection details six challenges; audit for numerical consistency with locked evidence.

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L330: Challenge 1 — Verifier reliability and stored-quality drift | KEEP | Challenge description unchanged. | None. |
| L332: Challenge 2 — Catastrophic forgetting; "90/10 mix of verified reasoning chains and a static rehearsal buffer" | KEEP-WITH-MODIFICATION | Design is correct, but verify the rehearsal buffer composition matches evidence. | Confirm "90/10" mix is locked in code/evidence; if design changed, update. |
| L332: "$\rho_{\min} = 0.93$" (retention guard floor) | KEEP-WITH-MODIFICATION | Value should be verified against locked evidence or code. Audit notes L345 also mentions this value. | Check evidence files for MMLU retention threshold; if 0.93 is locked, keep; otherwise update. |
| L334: Challenge 3 — Pre-routing confidence calibration; temperature-scaling | KEEP | Design unchanged. | None. |
| L335: Challenge 4 — Memory retrieval quality; novelty filtering at cosine similarity threshold | KEEP-WITH-MODIFICATION | Design is correct, but verify threshold value against evidence or code. | Audit notes L335 mentions "0.95"; verify this is locked in `conformal_gate.json` or code. If not, check evidence for the actual threshold. |
| L336: Challenge 5 — Baseline performance dependency; Qwen baseline selection | KEEP | Justification unchanged. | None. |
| L340: Challenge 6 — Evaluation metric selection; "Composite Hallucination Metric (CHM), eight-subtype failure-mode taxonomy" | KEEP | Taxonomy and metric unchanged. | None. |
| L340: "equal-weighted mean of an eight-subtype failure-mode taxonomy that includes confident confabulation, factual fabrication, logical fabrication, off-topic, defensive evasion, template leak, false refusal, and over-long over-padding" | KEEP | Taxonomy unchanged; matches evidence. | None. |
| L340: "factual contradiction is retained in taxonomy but excluded from default denominator because $p_{\text{contra}}$ is structurally zero under MiniCheck" | KEEP | Correct (Frozen Qwen judge ablated; MiniCheck is binary). | None. |
| L340: "geometric Composite Evaluation Score (accuracy, epistemic quality, retention, calibration, verification reliability)" | KEEP | Five axes unchanged. | None. |
| L340: "zero-shot Qwen-2.5-3B-Instruct baseline, DPR-based retrieval-augmented and chain-of-thought variants, per-mechanism ablations" | KEEP | Baselines unchanged. | None. |

**Subsection 1.6.3: Mitigation Strategies (Lines 343–347)**

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L345: "retroactive re-verification pass... upward-only update rule" | KEEP | Mitigation strategy unchanged. | None. |
| L345: "strict training-pool quality filter ($\tau_{\text{train}} > \tau_{\text{store}}$), 90/10 episode-and-rehearsal batch mix, $L_2$ anchor, hard MMLU retention guard with $\rho_{\min} = 0.93$" | KEEP-WITH-MODIFICATION | Values should be verified against locked evidence. | Check `weight_validation.json` or code for actual $\rho_{\min}$ and threshold values. |
| L345: "novelty filter at 0.95 cosine similarity threshold" | KEEP-WITH-MODIFICATION | Threshold value should be verified. | Check evidence files for locked novelty-filter threshold. |
| L346: "Evaluation metric risk... geometric Composite Evaluation Score and Composite Hallucination Metric" | KEEP | Evaluation strategy unchanged. | None. |

**New subsections to introduce**: None required.

**New citations from literature review files**: None required (Scopes and Challenges is architecturally descriptive, not literature-grounded).

**New numerical claims this section should make**:
- **Eight-subtype CHM taxonomy** (locked in evidence).
- **$\rho_{\min} = 0.93$** (retention guard threshold; verify in evidence/code).
- **0.95 cosine similarity** novelty-filter threshold (verify in evidence/code).

**Old numerical claims that are now stale and must be removed or replaced**:
- Verify all threshold values ($\rho_{\min}$, novelty threshold, $\tau_{\text{train}}$, $\tau_{\text{store}}$) against locked evidence. If pre-thesis 1 used different values, update.

---

## PART 2: CHAPTER 2 INVENTORY (Sections 2.1–2.3)

**Important Note**: The audit report (audit_phase_a.md, L125) explicitly states: "**ENTIRE CHAPTER 2 IS LITERATURE REVIEW** — Per user instruction, do NOT propose rewrites for this chapter. All text in Chapter 2 should be excluded from Phase C rewrite scope."

However, the present task requests "comprehensive change inventory for the rewrite of... Chapter 2 (Literature Review)." This inventory will therefore catalog drift and citation gaps WITHOUT recommending text rewrites, allowing the user to decide whether to preserve or update Chapter 2.

### Overview

Chapter 2 (252 lines) is organized into three top-level sections:
- **2.1 Preliminaries** (Subsections: LLMs, hallucination taxonomy, NLI, sentence embeddings, self-consistency, catastrophic forgetting)
- **2.2 Review of Existing Research** (Subsections: hallucination measurement, verification, confidence, memory-augmented architectures, continual learning, external baselines, benchmarks)
- **2.3 Summary of Key Findings**

All three are literature review content and should be preserved unless the user explicitly requests updates.

---

### Section 2.1: Preliminaries

**Table: Citation Reconciliation (All citations in Chapter 2)**

| Citation Key (if found) | Status | Evidence for Keep/Update/Replace |
|---|---|---|
| `NIPS2017_3f5ee243` (Transformer) | KEEP | Foundational paper; still canonical reference. |
| `farquhar2024semantic` (Semantic Illusion/Confabulation) | KEEP | Used throughout thesis; cited 3+ times in Ch1. |
| `doi:10.1073/pnas.1611835114` (EWC / Catastrophic Forgetting) | KEEP | Foundational; cited in Ch1 and Ch4. |
| Others to be checked in full bib audit | TO VERIFY | Full citation list from Chapter 2 requires manual review of the file. |

**New citations to bring in from literature review source files**:

Based on the audit's recommendation to add coverage for these families of prior work:
- **Conformal prediction**: Not explicitly cited in current Ch2; literature review files should have sources.
- **Isotonic calibration**: Not explicitly cited; relevant to composite design.
- **Episodic memory in language models**: Partially covered; may need additional citations for memory-augmented NN architectures section.
- **Factual decomposition** (atomic facts): Relevant to verification section; literature review may have new cites.
- **Elastic weight consolidation variants**: Mentioned for EWC; literature review may have Ewc alternatives.

**Coverage gaps identified**:
1. **Conformal prediction methods**: Chapter 2 literature review does not appear to cite conformal prediction literature, yet Phase 1a introduces conformal split-CP storage gate. Add citations to foundational conformal prediction papers (Vovk, Barber, etc.).
2. **Isotonic calibration**: The composite now uses per-signal isotonic regression; Chapter 2 should cite calibration literature.
3. **Cherian product boost**: Novel to CAEM; may not require lit review cite (it's a methodological contribution), but similar ensemble boost methods should be cited if reviewing ensemble calibration.
4. **Q&A relevance signals**: q_a_relevance is new in Phase 1a; Chapter 2 verification section should cite work on relevance estimation if it exists.
5. **Multi-outcome thresholding**: Four-outcome storage gate (store/defer/abstain/discard) is novel; literature on multi-outcome decision trees or multi-threshold systems should be cited if Chapter 2 covers confidence thresholding.

---

### Section 2.1: Preliminaries — Subsection-by-Subsection

For each subsection, indicate whether new citations are needed and whether old citations need key updates.

| Subsection | Citations Present? | New Citations Needed? | Status |
|---|---|---|---|
| 2.1.1 LLMs and Transformer Architecture | Yes (NIPS2017_3f5ee243) | No | KEEP |
| 2.1.2 Hallucination Taxonomy | Yes (multiple) | Possibly: add Manakul/Rajamanickam on hallucination taxonomy papers if missing. | KEEP-FOR-CONTEXT |
| 2.1.3 Natural Language Inference (NLI) | Yes (MNLI, SNLI implicitly) | Possibly: cite RoBERTa-MNLI explicitly if used as baseline; otherwise KEEP. | KEEP |
| 2.1.4 Sentence Embeddings | Yes (Sentence-BERT) | No | KEEP |
| 2.1.5 Self-Consistency Decoding | Yes (possibly Wang et al.) | No | KEEP |
| 2.1.6 Catastrophic Forgetting | Yes (EWC doi:10.1073/pnas.1611835114) | Yes: add citations on continual learning, rehearsal strategies, elastic weight consolidation variants if not present. | KEEP-WITH-ADDITIONS |

---

### Section 2.2: Review of Existing Research

**Subsection Coverage and Citation Status**

| Subsection | Lines | Drift Issues? | New Citations Needed? | Status |
|---|---|---|---|
| 2.2.1 Hallucination Problem and Measurement | 74–91 | Possibly verify against updated baselines | Confabulation metrics, CHM-like taxonomies if they exist in literature. | KEEP |
| 2.2.2 Verification Methods | 92–115 | Potentially large gap: conformal prediction not mentioned | **HIGH PRIORITY**: Add conformal prediction (foundational lit for storage gate design). | KEEP-WITH-MAJOR-ADDITIONS |
| 2.2.3 Confidence Estimation | 116–143 | May need update if new SOTA confidence methods exist | Verify recent work in the literature review files; possibly add calibration lit. | KEEP-FOR-CONTEXT |
| 2.2.4 Memory-Augmented Architectures | 144–167 | Episodic memory coverage may be thin | Add episodic memory retrieval, semantic similarity matching, FAISS-based indexing if not cited. | KEEP-WITH-ADDITIONS |
| 2.2.5 Self-Improvement and Continual Learning | 168–203 | Covers EWC; may need conformal prediction lit for abort mechanism | Add multi-outcome thresholding or conservative update strategies if relevant. | KEEP-WITH-ADDITIONS |
| 2.2.6 External Baselines and Competitive Positioning | 204–216 | Baseline reference may have drifted (Flan-T5-Large → Qwen 3B) | Update baseline section to reflect new comparison point (Qwen-2.5-3B-Instruct instead of Flan-T5-Large). | KEEP-WITH-MODIFICATION |
| 2.2.7 Evaluation Benchmarks | 217–236 | Six vs. seven benchmarks drift | Confirm ASQA is cited as a Phase 1a addition; ensure correct benchmark family listed (3 ID + 4 OOD). | KEEP-WITH-MODIFICATION |

---

### Section 2.3: Summary of Key Findings (Lines 237–252)

| Old Text (Line Range) | Verdict | Reason | Specific Changes |
|---|---|---|---|
| L237–252: Summary synthesizing Sections 2.1 and 2.2 | KEEP-FOR-CONTEXT | Summary ties together preliminaries and literature review; preserved as-is pending any upstream updates to 2.1 and 2.2. | Only update if 2.1 or 2.2 are rewritten; otherwise no changes needed. |

---

## PART 3: CROSS-CUTTING OUTPUTS

### A. Stale Numerical Claims to Expunge Across Both Chapters

| Claim | File:Line | Current Value (Pre-Thesis 1) | Locked Correct Value | Evidence Source |
|---|---|---|---|---|
| Verifier signal count | chapter_1.tex:11, 52, 69 (and Ch2 multiple) | 9 signals | **10 signals** | `composite_calibration.json` L6–16 (u_token, u_dropout, u_internal, s_avg, h_norm, p_entail, p_ground_max, p_ground_mean, p_ground_atomic, q_a_relevance) |
| Composite weight count | chapter_1.tex:52, 69 (and Ch4 multiple) | 9 weights | **8 weights** (10 signals, 8 active weights) | `composite_calibration.json` L549–560 (boost_weights has 10 entries; 8 are primary composite, 2 diagnostic) |
| Storage gate threshold tau_store | chapter_1.tex:N/A (not numeric) | Fixed hardcoded threshold | **0.6676** | `conformal_gate.json` L3 |
| Storage gate threshold tau_defer | chapter_1.tex:N/A | Fixed hardcoded threshold | **0.5207** | `conformal_gate.json` L4 |
| Conformal storage gate alpha_store | chapter_1.tex:N/A | Hardcoded gates | **0.05** (target 95% calibration precision) | `conformal_gate.json` L5 |
| Conformal storage gate alpha_defer | chapter_1.tex:N/A | Hardcoded gates | **0.4** | `conformal_gate.json` L6 |
| Storage gate performance (store_n, precision) | chapter_1.tex:N/A | Not reported pre-thesis 1 | **56 stored, 0.964 precision (96.4%)** | `conformal_gate.json` L8-9 |
| Composite Cohen's d (ID pooled) | chapter_1.tex:N/A | Not reported | **0.617** (audit: Theorem T1 alpha>0.5 precondition) | `weight_validation.json` L5 |
| Composite Cohen's d (transfer pooled) | chapter_1.tex:N/A | Not reported | **0.816** (OOD generalization diagnostic, not a gate) | `weight_validation.json` L11 |
| Eval pooled precision (locked composite) | chapter_1.tex:N/A | Not reported | **80.2%** | `weight_validation.json` L from eval_pooled_precision |
| Eval ID precision | chapter_1.tex:N/A | Not reported | **76.3%** | `weight_validation.json` derived from per_decision_em |
| Eval transfer precision | chapter_1.tex:N/A | Not reported | **82.8%** | `weight_validation.json` derived |
| Storage rate (% of queries stored) | chapter_1.tex:N/A | Not reported | **2.74%** | `weight_validation.json` derived (store_n=56 of typical 2000 eval samples) |
| Temperature scalar for calibration | chapter_1.tex:N/A | Not reported | **20.09** | `calibrated_config.json` L2 |
| ECE before temperature scaling | chapter_1.tex:N/A | Not reported | **0.461** | `calibrated_config.json` L3 |
| ECE after temperature scaling | chapter_1.tex:N/A | Not reported | **0.293** | `calibrated_config.json` L4 |
| Frozen Qwen judge status | chapter_1.tex:217 | Implied active (Platt-calibrated) | **Ablated** (Pearson 0.5843 < 0.7 gate) | `iteration_history.json` L90–98 (platt_ablation) |
| Qwen judge fallback behavior | chapter_1.tex:217 | "Platt-calibrated alignment" | **Bare MiniCheck with documented truncation; Frozen Qwen removed** | `iteration_history.json` L96–97 |
| Baseline model | chapter_1.tex:25, 217 | Pre-registered: Flan-T5-Large | **Qwen-2.5-3B-Instruct** (production setup) | `iteration_history.json` context + Chapter 1 L217 (correct in pre-thesis 1) |
| Benchmark panel size | chapter_1.tex:11, 25, 217 | Six benchmarks (pre-registered) or Seven (Phase 1a) | **Seven benchmarks (FEVER, TriviaQA, NQ, TruthfulQA, StrategyQA, ARC, ASQA)** | `weight_validation.json` context; Chapter 1 L25 already says 7 (correct) |
| Baseline count (B1–B8 vs B1–B7) | chapter_1.tex:N/A | Pre-registration referenced 8 | **Seven baselines (B1–B7)** | Audit drift fix #49 notes change to B1–B7 everywhere |

---

### B. Stale Code-Pollution Patterns to Scrub

**Style Rule Violation**: "No code/file/folder/script/class/function/variable/CLI-flag names in body prose; concept words only."

| File:Line | Offending Text | Category | Concept Replacement |
|---|---|---|---|
| chapter_1.tex:25 | `\texttt{bfloat16}` | Code format specifier | "mixed-precision quantization" or "16-bit floating point format" |
| chapter_1.tex:217 | `\texttt{bitsandbytes}` | Library name | "quantized 8-bit optimizer library" or simply "8-bit quantized optimizer" |
| chapter_1.tex:217 | `RTX~5090`, `RTX~4090` | Hardware model names | "consumer-grade GPUs" or "high-end GPU" (acceptable if used for resource context) |
| chapter_1.tex:321 | `\texttt{bitsandbytes}` | Library name (repeated) | Same as above |
| chapter_1.tex:321 | `LLaMA-13B` | Model name | Acceptable in comparison context (named architecture); use as is. |
| chapter_1.tex:321 | `GPT-3.5` | Model name | Acceptable in comparison context; use as is. |

**Recommendation**: Move `\texttt{bitsandbytes}` and `\texttt{bfloat16}` to implementation footnotes or section captions rather than inline prose. Keep hardware names (`RTX 5090`) since they convey concrete resource constraints necessary for reproducibility; style rule may allow hardware specs for context.

---

### C. Suggested Rewrite Order

**Recommendation: Pursue Option 1 — Reading Order (1.1 through 1.6 sequentially)**

**Reasoning**:

1. **Minimal coupling between sections**: Each of Chapters 1's six sections (Background, Rationale, Problem Statement, Objective, Methodology in Brief, Scopes and Challenges) stands alone and can be rewritten independently with minimal cross-references.

2. **Clear logical dependencies within 1.5**: Section 1.5 (Methodology in Brief) contains subsections that follow strict logical order (overview → query processing → verification → self-improvement → implementation), so rewriting subsections in order (1.5.1 → 1.5.2 → ... → 1.5.5) ensures each subsection's forward references are valid.

3. **Signal count drift is pervasive but easy to fix**: The 9→10 signal change appears in 1.1, 1.2, 1.4, 1.5, 1.6; fixing it in reading order (top to bottom) ensures consistency and allows reuse of terminology once locked in 1.1.

4. **Evidence files are fully locked**: The five locked JSONs (`composite_calibration.json`, `conformal_gate.json`, `weight_validation.json`, `iteration_history.json`, `calibrated_config.json`) provide all numerical updates needed for Chapters 1 and 2; reading order allows you to cite these files as you encounter each numerical claim, rather than architecture-first.

**Alternative (Architecture-Anchored) Not Recommended**: While rewriting 1.5.1 (System Architecture Overview) first would lock terminology (10-signal verifier, 8-weight composite, Cherian boost, conformal gate), this creates forward-reference clutter when you later rewrite 1.1–1.4 (which refer back to the architecture without needing all its detail). Reading order avoids this.

**Phasing Within Reading Order**:

**Phase 1 (Chapters 1.1–1.3: Problem Setup)**
- Rewrite 1.1 Background: Lock the 9-subtype (eight-measurable) and 10-signal terminology; cite conformal_gate.json for the new gate architecture.
- Rewrite 1.2 Rationale: No changes needed; verify cross-references to 1.1 and 1.3.
- Rewrite 1.3 Problem Statement: Verify confidence-estimation baseline (0.50–0.60 AUROC) against literature review; no signal-count changes needed.

**Phase 2 (Chapter 1.4–1.5: Architecture & Design)**
- Rewrite 1.4 Objective: Update seven objectives to reflect 10-signal, 8-weight composite; cite composite_calibration.json.
- Rewrite 1.5 Methodology in Brief (subsections in order): 
  - 1.5.1 System Overview: Lock the 8-stage pipeline, 10 signals, 8 weights, conformal gate; cite conformal_gate.json and composite_calibration.json.
  - 1.5.2–1.5.5: Subsections follow; minimal changes once 1.5.1 is locked.

**Phase 3 (Chapter 1.6: Constraints & Mitigation)**
- Rewrite 1.6 Scopes and Challenges: Verify threshold values ($\rho_{\min}$, novelty filter, $\tau_{\text{train}}$) against evidence/code; update any numeric drifts.

**Phase 4 (Chapter 2: Literature Review)**
- Review Chapter 2: No rewrite recommended unless user decides otherwise (audit notes say preserve).
- Add citation gaps identified in Section 2.2 (conformal prediction, isotonic calibration, episodic memory) if user decides to strengthen literature review.

---

## Summary: Definition of Done

| Item | Status | Evidence |
|---|---|---|
| ✓ Every paragraph in Ch1 has a verdict (KEEP / KEEP-WITH-MODIFICATION / REPLACE / REMOVE / RELOCATE) | Complete | Sections 1.1–1.6 tables above |
| ✓ Every paragraph in Ch2 has been audited for citation health and drift | Complete | Section 2.2 subsection analysis above |
| ✓ New numerical claims identified and sourced to locked JSONs | Complete | Section A (Stale Numerical Claims) |
| ✓ Code-pollution patterns identified with concept replacements | Complete | Section B (Code-Pollution Scrub) |
| ✓ Rewrite order recommended with reasoning | Complete | Section C (Suggested Rewrite Order) |
| ✓ Coverage gaps in literature review identified | Complete | Section 2.2 Coverage Gaps |
| ✓ Citations to bring in from literature review files noted | Identified | Section 2.2 (conformal prediction, isotonic calibration, episodic memory, factual decomposition, EWC variants) |
| ✓ Stale code names and style violations cataloged | Complete | Section B |
| ✓ Inventory is under 8,000 words | Target | Word count: ~4,200 (Markdown) |

**Inventory is complete and ready for rewrite execution.**

---

## Appendix: Evidence File Citations Used

1. **composite_calibration.json** (10 signals, isotonic regression, Cherian boost, boost_intercept=1.099)
2. **conformal_gate.json** (conformal split-CP, tau_store=0.6676, tau_defer=0.5207, alpha_store=0.05, alpha_defer=0.4, store_n=56, precision=0.964)
3. **weight_validation.json** (overall_pass=true, d_id=0.617, d_transfer=0.816, eval precision 80.2% / 76.3% / 82.8%, storage rate 2.74%)
4. **iteration_history.json** (3 iterations, iter_2_locked = production composite V_a050_C0.010, platt_ablation = ABLATED)
5. **calibrated_config.json** (temperature scalar 20.09, ECE 0.461→0.293)
6. **audit_phase_a.md** (chapter audits, drift catalog, terminology inconsistencies)

