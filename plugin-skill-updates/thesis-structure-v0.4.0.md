# Thesis Structure & Key Claims

## Thesis Title (working)
**CAEM: Confidence-Aware Episodic Memory with Self-Improvement — A Verified Rejection Sampling System for Progressive Hallucination Reduction**

## Author
Aksan Gony Alif — BRAC University, Department of Computer Science
Supervisor: Prof. Md. Golam Rabiul Alam

> **CRITICAL:** All performance numbers, improvement percentages, accuracy figures, and latency values in this document are **PROJECTED EXPECTATIONS** from the thesis plan. No experiments have been run yet. Never present these as actual results in the thesis or in discussions.

---

## Planned Chapter Structure (6 chapters — BracU template fixed)

> ⚠️ The BracU CSE400 template has exactly 6 chapters. Do NOT add a 7th. Template file numbering does NOT match chapter numbers (do not rename files).

| Chapter | Title | Template file | Status |
|---|---|---|---|
| Ch1 | Introduction | chapter_1.tex | ✅ Submitted |
| Ch2 | Literature Review | chapter_2.tex | ✅ Submitted |
| Ch3 | Requirements Analysis | chapter_3.tex | ⚠️ Stub (9 lines) |
| Ch4 | Methodology | chapter_5.tex | ❌ Not written |
| Ch5 | Results & Analysis | chapter_6.tex | ❌ Not written |
| Ch6 | Conclusion | chapter_9.tex | ❌ Not written |

### Chapter 1: Introduction
- Problem: LLMs hallucinate — factual errors, confabulations, imitative falsehoods
  - **Extrinsic hallucination:** Model adds information not supported by source/context (regardless of whether it contradicts — broader category)
  - **Intrinsic hallucination:** Model directly contradicts source/context
  - Use these definitions precisely throughout; do not conflate them
- TruthfulQA baseline: GPT-3 achieved **58% on truthfulness-alone** metric but only **21–25% on truthful+informative combined** metric (Lin et al. 2021). These are DIFFERENT metrics — do not conflate.
- Gap: RAG, CoT, and standalone fine-tuning are each insufficient in isolation
- Proposal: CAEM — full architectural system combining memory, confidence, routing, verification, self-improvement
- Contribution list (to be finalized with actual results):
  1. Two-stage multi-signal confidence system: `u_pre` (2 signals; C_conv adapted from Nandakishor 2025) and `û` (4 signals with SE upweighted to 0.40)
  2. Three-tier adaptive routing with OR-condition safety override and combined similarity+confidence score (λ=0.7, θ=0.90)
  3. Dataset-adaptive multi-layer verification (NLI via RoBERTa-Large-MNLI + SC + SE) with neutral NLI escalation rule
  4. Iterative self-improvement via L2-regularized fine-tuning on verified data, with retroactive re-verification and retrieval feedback loop
  5. Data purity theorem: `P = pα/(pα + (1-p)(1-α))` with explicit condition `p > (1-α)`
  6. Empirical evaluation: X% hallucination reduction [PROJECTED: 20–30%] on 4 benchmarks, single GPU

### Chapter 2: Literature Review
Build narrative toward CAEM's gap — present in this thematic order:

1. **Hallucination in LLMs** — types (extrinsic/intrinsic), causes, benchmarks (Ji et al. 2023, Maynez et al. 2020)
2. **Retrieval-Augmented Generation** — Lewis 2020 (RAG), REALM (Guu 2020), RETRO (Borgeaud 2022)
3. **Episodic and external memory** for LLMs
4. **Uncertainty quantification** — token prob, MC Dropout (Gal 2016), self-consistency (Wang 2023), semantic entropy (Kuhn 2023; Farquhar 2024), C_conv/confidence-aware routing (Nandakishor 2025)
5. **Self-improving systems** — ReST (Gulcehre 2023), STaR, related bootstrapping approaches
6. **Verification and NLI** — NLI models for factual verification; limitations
7. **Catastrophic forgetting** — EWC (Kirkpatrick 2017) cited as motivation for the L2 regularisation choice; CAEM uses simpler L2 (not EWC itself — Fisher matrix too expensive at 780M params)
8. **Chain of thought faithfulness** — Korbak et al. 2025, Turpin et al. 2023; why CAEM uses outcome-based not process-based verification

**Gap statement:** No prior work combines confidence-gated episodic memory, two-stage multi-signal confidence (including C_conv adapted for encoder-decoder), dataset-adaptive multi-layer verification with neutral escalation, L2-regularized self-improvement with retroactive re-verification and retrieval feedback loop, AND a theoretical data purity guarantee — in a single inference-time system on a single consumer GPU.

### Chapter 3: Requirements Analysis
Functional requirements: episodic memory, confidence estimation, adaptive routing, multi-signal verification, self-improvement loop. Non-functional requirements: single GPU constraint, inference latency targets (Tier 1 < 400ms, Tier 2 < 2.5s, Tier 3 < 6s), single-device deployment. Evaluation criteria: hallucination reduction on 4 benchmarks, theoretical guarantee validation.

### Chapter 4: Methodology (PIPELINE EXECUTION ORDER — 8 stages)

Present sections in the order a query flows through the system:

**§4.1 System Architecture Overview**
- Pipeline overview: 8-stage flow, 3-tier routing structure
- Design rationale: why memory + confidence + routing + verification + self-improvement together
- Benchmark selection rationale (BM-02): why these 4 benchmarks were chosen at design time

**§4.2 Confidence Estimation and Routing**
- Query encoding: `sentence-transformers/all-mpnet-base-v2`, **768-dim output** (NOT 384)
- Pre-routing confidence `u_pre` from 2 signals: u_token (geometric mean token log-probs) + C_conv (adapted from Nandakishor 2025; variance ratio of early vs. late encoder hidden states)
- Combined: `u_pre = 0.6·u_token + 0.4·(1/(1+C_conv))`
- OR-condition checked FIRST: `u_pre < 0.60` → Tier 3 regardless of memory score
- Combined routing score (only if OR-condition did NOT fire): `0.7s + 0.3û_stored`; thresholds: score > 0.90 → Tier 1; s > 0.75 → Tier 2
- u_pre and routing_score are TWO SEPARATE mechanisms — do not conflate
- Post-generation confidence gate `û` (Tier 2 only): 4 signals; initial weights 0.25 equal; projected post-calibration: `û = 0.20·u_token + 0.20·u_dropout + 0.20·u_consistency + 0.40·(1-u_entropy)` (field names are `u_consistency` and `u_entropy`, not `u_SC` or `H_sem`)

**§4.3 Verification Pipeline**
- Tier-dependent generation: Tier 1 = direct retrieval (NO per-query verification); Tier 2 = fine-tuned model; Tier 3 = RAG with Wikipedia
- Multi-signal verification: NLI (RoBERTa-Large-MNLI) + SC (N=10 samples) + SE (semantic entropy)
- Dataset-adaptive layers: which VE layers active per benchmark (BM-03/04)
- VE1 (neutral NLI escalation): if NLI=NEUTRAL, escalate to Tier 3
- VE2 (TruthfulQA only): curated list matching ~38 misconception categories from Lin et al. 2021 — NOT a learned classifier
- û_stored derivation: `û_stored = 0.50·P(ENTAIL) + 0.30·s_avg + 0.20·(1-H_norm)` — NOT derived from û
- NLI requires ground truth labels; SC and SE are reference-free

**§4.4 Self-Improvement Loop**
- Fine-tuning target: `(q, c, reasoning_chain)` triples (NOT just `answer`)
- L2 regularization loss: `L = L_CE(q, c, reasoning_chain) + λ_reg·||θ - θ_prev||²` (λ_reg = 0.01, NOT EWC Fisher-weighted)
- Regularizes toward θ_prev (previous cycle weights), NOT θ_base (original pretrained weights)
- 90/10 batch mixing; 3 cycles total
- Retroactive re-verification after each cycle; retrieval feedback loop for dynamic û_stored updates

**§4.5 Theoretical Framework**
- Data purity theorem: `P = pα/(pα + (1-p)(1-α))`; condition: `p > (1-α)`; guarantee: `P > p` (NOT P > α)
- Improvement formula: `ΔA ≈ 0.4·(P - p_current)` (0.4 from Natarajan et al. 2013 — validate empirically)
- Coupled two-cycle recurrence: formalizes model improvement + memory calibration loops
- Convergence: diminishing gains across cycles empirically; practical equilibrium, NOT zero-hallucination limit — do NOT claim global convergence to zero

### Chapter 5: Results & Analysis

**Structure (6 sections):**

**§5.1 Experimental Setup**
1. Baseline pilot validation (RUN FIRST — validates `p > 0.5` on all 4 benchmarks)
2. Data splits (non-overlapping): calibration split (500 samples — for temperature scaling), purity validation split (500 samples — for theorem validation, SEPARATE from calibration), test split
3. Cold start seeding — **447 total verified episodes seeded before Cycle 1** (not 200–500 per benchmark)
4. All 6 baselines: Zero-shot, CoT, RAG, Self-consistency, Vanilla FT, Memory-only
5. Circular evaluation decoupling — final evaluation uses ONLY benchmark ground-truth labels

Key sentence: "The 500-sample calibration set (for temperature scaling) and the 500-sample purity validation set (for theorem validation) are drawn from non-overlapping portions of the official benchmark validation splits. This separation prevents circular validation."

Benchmark selection rationale for why these 4 (BM-01): HotpotQA for multi-hop reasoning, TruthfulQA for misconception reduction, FEVER for fact-checking, StrategyQA for boolean chain-of-thought. Together they cover factual precision, reasoning depth, and format diversity.

**§5.2 Main Results**
- Primary accuracy table: 4 benchmarks × 6 baselines × 3 cycles
- Hallucination reduction figure

**§5.3 Mechanism Analysis**
- Five-mechanism evidence table (primary diagnostic): Cycle | Halluc. Reduction | Tier 1 Fraction | Tier 3 Fraction | MMLU Retention | Mean û_stored
- Routing distribution per cycle; OR-condition override rate; per-benchmark Tier 1 accuracy (lower FEVER = near-miss issue)

**§5.4 Ablation Study**
Required ablations:
- A1 (no retroactive re-verification) — proves memory quality across cycles
- A2 (all Tier 3, no routing) — proves routing efficiency (accuracy similar, latency ~4×)
- A3 (no OR-condition) — proves safety-first design (Tier 1 accuracy drops, esp. FEVER)
- Remove episodic memory; Remove self-improvement loop; Remove verification entirely; Single-layer NLI only; Remove semantic entropy; (q,a)-only vs. (q,c,reasoning_chain) fine-tuning; All 4 signals vs. subsets

**§5.5 Theory Validation**
Three structured experiments (NOT side notes):
- Theory 1 (Purity Theorem): five-step protocol; table of p/α/P_theory/P_obs per cycle; state condition p > (1−α). Guarantee is P > p, NOT P > α.
- Theory 2 (Monotonicity): verify p_0 < p_1 < p_2 < p_3 and α_0 < α_1 < α_2 < α_3
- Theory 3 (Convergence): verify Δ_3 < Δ_2 ≤ Δ_1; state that convergence is a theoretical prediction validated empirically

**§5.6 Latency and Error Analysis**
- Per-tier latency: Tier 1 < 400ms; Tier 2 < 2.5s; Tier 3 < 6s; report mean and 95th percentile
- Qualitative failure examples; failure mode categories

**Closing sentence:** "Together, these results confirm that CAEM's closed-loop architecture delivers consistent, measurable, and theoretically grounded hallucination reduction across four qualitatively distinct benchmarks."

### Chapter 6: Conclusion (includes Discussion & Limitations)

**Structure:**
1. Restate problem
2. CAEM's approach in one paragraph
3. Key empirical findings
4. Theoretical contributions (Data purity theorem + coupled recurrence + convergence theory — see C6-01)
5. Limitations (genuinely self-critical)
6. Future directions

**Key sentence for theoretical contributions:** "The data purity theorem provides a formal guarantee that CAEM's verification pipeline produces memory of strictly higher purity than unfiltered generation, under the verifiable condition p > (1−α). The coupled improvement recurrence provides a theoretical basis for the observed monotone accuracy gains across cycles. Together, these theoretical results distinguish CAEM from heuristic self-training approaches."

**Known limitations:**
- Data purity theorem condition p > (1-α) may not hold for near-random benchmarks
- FAISS capacity capped at 20,000
- L2 regularization is uniform, not Fisher-weighted
- C_conv adaptation to Flan-T5 encoder not validated against original decoder-only setting
- Outcome-based verification — reasoning chain quality not directly verified (Korbak et al. 2025)
- Single GPU / 780M model — scalability is future work
- NLI verification requires ground-truth labels — SC + SE are reference-free
- Inference mode memory growth limited without ground truth

---

## Key Claims to Validate Empirically

| Claim | Measurement | Target [PROJECTED] |
|-------|-------------|-------------------|
| Hallucination reduction | EM/F1/Accuracy vs. zero-shot | 20–30% |
| Self-improvement is monotonic | Metric per cycle | Cycle 0 < 1 < 2 < 3 |
| Purity P exceeds BASE accuracy p | Measure P on purity eval set | P_obs > p (NOT P > α) |
| OR-condition prevents routing failures | Tier 1 accuracy with vs. without | Measurable drop without it |
| Verification accuracy 85–90% | Human evaluation agreement | 85–90% |
| Retroactive re-verification improves purity | Mean û_stored across cycles | Rising without A1 ablation |
| SE weight 0.40 > equal weights | AUROC ablation | Non-trivial delta |
| Convergence diminishes | Δ_3 < Δ_2 | Confirmed empirically (practical equilibrium, not zero) |

---

## Equations (LaTeX — corrected notation)

```latex
% Pre-routing confidence (2 signals)
u_{\text{pre}} = 0.6 \cdot u_{\text{token}} + 0.4 \cdot \frac{1}{1 + C_{\text{conv}}}

% C_conv internal convergence (Nandakishor 2025, adapted for encoder)
C_{\text{conv}} = \frac{\text{Var}(h_{1:L/2})}{\text{Var}(h_{L/2:L}) + \varepsilon}, \quad \varepsilon = 10^{-8}

% Token confidence (geometric mean)
u_{\text{token}} = \exp\!\left(\frac{1}{n}\sum_{i=1}^n \log p_i\right)

% Post-generation confidence (4 signals, Tier 2 only; initial weights 0.25 equal)
% FIELD NAMES: u_consistency (not u_SC), u_entropy (not H_sem or h_entropy_norm)
\hat{u} = 0.20 \cdot u_{\text{token}} + 0.20 \cdot u_{\text{dropout}} + 0.20 \cdot u_{\text{consistency}} + 0.40 \cdot (1 - u_{\text{entropy}})

% Semantic entropy (normalized)
u_{\text{entropy}} = \frac{H_{\text{sem}}}{\log_2 K} = \frac{-\sum_c p_c \log_2 p_c}{\log_2 K}

% OR-condition safety override (checked FIRST, before routing score)
\text{Tier 3 if: } u_{\text{pre}} < 0.60

% Tier 1 combined routing score (only runs if OR-condition did not fire)
\text{score} = \lambda s + (1-\lambda)\hat{u}_{\text{stored}}, \quad \lambda = 0.7, \quad \theta_1 = 0.90

% Stored confidence (full — HotpotQA)
\hat{u}_{\text{stored}} = 0.50 \cdot P(\text{ENTAIL}) + 0.30 \cdot s_{\text{avg}} + 0.20 \cdot (1 - \hat{H})

% Fine-tuning loss (L2 regularization — NOT EWC; training target is reasoning_chain, not answer)
\mathcal{L} = \mathcal{L}_{\text{CE}}(q, c, \text{reasoning\_chain}) + \lambda_{\text{reg}} \cdot \|\theta - \theta_{\text{prev}}\|^2, \quad \lambda_{\text{reg}} = 0.01

% Data purity theorem
P = \frac{p\alpha}{p\alpha + (1-p)(1-\alpha)}, \quad \text{requires } p > (1-\alpha), \quad \text{guarantees } P > p

% Improvement formula (Natarajan et al. 2013 coefficient)
\Delta A \approx 0.4 \times (P - p_{\text{current}})

% Retrieval feedback update (accepted)
\hat{u}_{\text{stored}}^{(k+1)} = \hat{u}_{\text{stored}}^{(k)} + \eta (1 - \hat{u}_{\text{stored}}^{(k)}), \quad \eta = 0.01

% Retrieval feedback update (overridden)
\hat{u}_{\text{stored}}^{(k+1)} = \hat{u}_{\text{stored}}^{(k)} - \eta \hat{u}_{\text{stored}}^{(k)}
```

---

## Contribution Novelty Checklist

Before submission, verify each contribution is genuinely novel:

- [ ] Two-stage confidence architecture (u_pre + û) — separated pre/post-generation
- [ ] C_conv adapted for encoder-decoder (Flan-T5 encoder hidden states)
- [ ] OR-condition safety override — likely novel; check RAG routing papers
- [ ] Dataset-adaptive verification layer selection
- [ ] Neutral NLI escalation rule (VE1) — likely novel in this context
- [ ] VE2 misconception taxonomy matching — curated list, not learned; dataset-specific
- [ ] Retroactive re-verification per cycle
- [ ] Retrieval feedback loop for dynamic û_stored
- [ ] Data purity theorem — verify not a known result in semi-supervised learning
- [ ] 0.4 improvement coefficient — from Natarajan (2013); validate empirically
- [ ] Coupled two-cycle recurrence formalization
- [ ] Outcome-based verification positioning vs. Korbak (2025)

---

## Writing Quality Notes

1. Frame CAEM as architectural intervention, not post-hoc detection — everywhere
2. Use extrinsic/intrinsic hallucination definitions consistently (Ji et al. 2023)
3. State routing hyperparameters as calibration set-tuned values
4. C_conv must be labeled "adapted from Nandakishor (2025) for encoder-decoder"
5. L2 regularization: write `λ_reg·||θ-θ_prev||²` — do NOT call it EWC; regularizes toward θ_prev (previous cycle), not θ_base (original pretrained)
6. Data purity theorem: state condition p > (1-α) AND discuss failure case; guarantee is P > p (NOT P > α)
7. The 0.4 coefficient in ΔA: acknowledge as borrowed from Natarajan (2013); validate empirically
8. û signal weights: initial 0.25 equal; projected post-calibration 0.20/0.20/0.20/0.40; actual values reported in Chapter 5
9. Limitations section must be genuinely self-critical
10. Outcome-based verification: add paragraph acknowledging Korbak et al. (2025)
11. Ground truth dependency: scope to NLI component only — SC and SE are reference-free
12. VE2: describe as curated list matching (not learned), dataset-specific to TruthfulQA only
13. Training target is `reasoning_chain`, not just `answer` — the fine-tuning uses full chain-of-thought output
14. Convergence: empirical diminishing gains consistent with theory — never claim infinite cycles → zero hallucination
15. Embedding dimension is **768** (all-mpnet-base-v2) — NOT 384
16. Cold-start: 447 total verified episodes seeded (not 200–500 per benchmark)
17. Citation keys: use ONLY from `chapter_1.tex` / `chapter_2.tex` (validated); do not use unified plan keys
