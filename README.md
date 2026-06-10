# CAEM — Confidence-Aware Episodic Memory with Self-Improvement

**CSE400 Final Year Thesis · BRAC University**
**Author:** Aksan Gony Alif

A three-tier hallucination-reduction architecture for open-domain question answering, built on Qwen-2.5-3B-Instruct. CAEM combines a verified episodic memory, confidence-aware routing with a retrieval-utility classifier, a nine-signal calibrated-probability verifier ensemble with a fixed-threshold storage gate, and a constrained per-cycle self-improvement loop under a multi-modal retention guard with asymmetric rollback.

All numbers in this README are anchored against the thesis report; the source location is cited in parentheses (chapter/appendix/table) so a reader can audit each claim against the canonical text in `thesis_report/main.pdf`.

| Resource | Link |
|---|---|
| Thesis report (325 pages, PDF) | [`thesis_report/main.pdf`](thesis_report/main.pdf) |
| Realised empirical artefacts (Google Drive) | <https://drive.google.com/drive/folders/1xErhsRDYe_qO0ZgUBn0WXK81eHc5jZCa> |
| Wikipedia passage FAISS index + pre-main snapshots (Hugging Face, private) | `aksaN000/caem-passage-index-21m` |
| Production runbook | [`docs/PRODUCTION_RUNBOOK.md`](docs/PRODUCTION_RUNBOOK.md) |
| Reproducibility recipe | [`thesis_report/appendix/appendix_e.tex`](thesis_report/appendix/appendix_e.tex) |
| Code-and-data availability | [`thesis_report/appendix/appendix_a.tex`](thesis_report/appendix/appendix_a.tex) §A.1 |
| Demo quickstart | [`docs/DEMO_QUICKSTART.md`](docs/DEMO_QUICKSTART.md) |

---

## TL;DR

Large language models hallucinate at deployment-relevant rates and the problem persists with scale. CAEM closes the gap with three architectural commitments: a **verified episodic memory** with a four-outcome decision (store, defer, abstain, discard) under a fixed-threshold rule on a calibrated probability composite; **confidence-aware routing** that combines memory similarity with stored quality, with an independent safety override and a **retrieval-utility classifier** (Stage 3b) that decides per query whether retrieval is expected to help; and a **constrained self-improvement loop** that fine-tunes a bounded low-rank adapter on a strict training pool under a multi-modal retention guard with asymmetric rollback.

On the five-benchmark evaluation panel (FEVER, TriviaQA, CommonsenseQA, TruthfulQA, StrategyQA) against the matched Qwen-2.5-3B-Instruct zero-shot baseline:

- **Headline contrast** at the within-trajectory best cycle (CAEM C2 vs B1 zero-shot reference): **ΔEM = +7.4%** relative, **ΔCHM = −10.8%** relative (`tab:pooled-trajectory`, Ch5 §5.2).
- **Retrieval-utility-classifier ablation** at cycle three: **+14.0 percentage-point lift** on CommonsenseQA exact match and **+10.0 percentage-point lift** on TruthfulQA exact match by re-routing misconception-bait queries away from the retrieval-augmented tier, with the retrieval tier share collapsing by **−33.8 percentage points** pooled (`tab:ruc-ablation`, Ch5 §5.7).
- **Pool purity** remains in the band **[0.732, 0.746]** on the pooled training-panel calibration fold across every committed cycle, clearing the registered floor of 0.64 by 9–11 percentage points (`tab:pi-store-trajectory`, Ch5 §5.5.1).
- **First in-flight retention rollback** at cycle four (TriviaQA-test probe ratio 0.886 < ρ_min = 0.93) with the memory store growing across the abort boundary under the asymmetric-rollback contract; cycle-five recovers to a worst-probe ratio of 0.958 (Ch5 §5.2; `cor:stochastic-equilibrium`).

The full theorem-by-theorem receipt panel (three theorems plus three corollaries) and the pre-registered five-hypothesis verdicts are reported in Chapter 5 §5.5.

---

## The problem

Modern instruction-tuned language models generate confident-sounding answers at deployment-relevant rates that exceed their actual accuracy by ten or more percentage points (`pmlr-v70-guo17a`, `kadavath2022language`). Single-signal correctness predictors achieve AUROC scores between 0.50 and 0.60, marginally above chance (`xiong2023can`). The TruthfulQA benchmark documents an inverse-scaling pattern under which larger models are *more* prone to echo human falsehoods (`lin-etal-2022-truthfulqa`). Clinical-question studies report hallucination rates of 69–77% on frontier models (`alkaissi2023artificial`). Hallucination is not a residual that scale removes; it is a structural property of stateless generation under uncertainty, as the recent survey literature argues (`ji2023survey`, `huang-etal-2023-large`).

CAEM treats this as a deployment-safety problem rather than a benchmark-accuracy problem. The architecture targets three structural shortcomings of the current inference paradigm: statelessness (no representation of past verified answers persists across sessions); overconfidence (single-signal correctness predictors are near-random); and immutability of trained capability (naive fine-tuning incurs catastrophic forgetting). The remainder of this document walks the architecture, the realised six-cycle trajectory, the pre-registered hypothesis adjudication, and the threats-to-validity disclosure.

The full motivation, the literature positioning across six areas (hallucination measurement, verification, confidence estimation, memory-augmented architectures, continual learning, evaluation benchmarks), and the architectural-opportunity argument are in Chapter 1 of the thesis.

---

## The architecture

CAEM organises every query into an **eight-stage per-query pipeline** wrapped in a **per-cycle update loop**.

```
Stage 1  Pre-routing confidence       u_pre  = w_tok · u_token + w_conv · 1/(1+c_conv), per-bench temperature scaled
Stage 2  Encode + episodic lookup     SBERT 768-d cosine over the memory store; returns (similarity, u_stored[e*])
Stage 3  Three-tier router            S(q) = 0.70 · sim + 0.30 · u_stored ; safety override: if u_pre < phi_pg force fallback
Stage 3b Retrieval-utility classifier intercepts safety-override + score-formula fall-through; 36-feature LR head
Stage 4  Execute tier                 Tier 1 = stored answer; Tier 2 = zero-shot generation; Tier 3 = retrieval-augmented
Stage 5  Verify                       Nine-signal verifier ensemble; per-benchmark calibrated probability composite
Stage 6  Decide                       Four-outcome decision tree with confabulation early-exit (precedes composite)
Stage 7  Output                       Calibrated probability + four-class decision tag rendered to user
Stage 8  Cycle-boundary update        Retroverify + deferred reconsider + consolidate + SIL + retention guard
```

### Stage 3b: the retrieval-utility classifier

At the heart of the post-baseline architectural intervention is the retrieval-utility classifier of Chapter 4 §4.5. It is a **36-feature L2-regularised logistic regression head** fitted under nested five-fold cross-validation on a content-hash-disjoint fresh pool of 3208 disagreement rows (1846 retrieval-helpful, 1362 retrieval-hurt). The classifier intercepts the dispatches that the three-tier router would have sent to the retrieval-augmented fallback and decides per query whether retrieval is expected to help, routing retrieval-hurt queries to the zero-shot tier instead. The deployed threshold is τ_RUC = 0.375. Pooled out-of-fold AUROC is 0.850 (std 0.008); per-benchmark AUROC ranges from 0.686 (OpenBookQA) to 0.936 (TriviaQA) with FEVER 0.718, CommonsenseQA 0.728, TruthfulQA 0.732, StrategyQA 0.852. The full feature ontology, the training-distribution composition, the calibration receipt, and the threshold-sweep diagnostic are in Chapter 4 §4.5.

### The nine-signal verifier ensemble

The verifier consumes nine signals organised in four families: **internal calibration** (`u_token`, `u_dropout`, `u_internal`); **sample-set agreement** (`s_avg`, `p_entail`); **external grounding** (`p_ground_max`, `p_ground_mean`, length-gated `p_ground_atomic`); and **question-answer relevance** (`q_a_rel`). The ensemble is **locked across cycles** by design: MiniCheck-Flan-T5-Large is the deployed entailment backend (the registered long-hypothesis Qwen judge was ablated under the pre-registered Pearson correlation floor failing at the v5 calibration overlap fold), the signal-extraction layer is frozen, and only the cycle-boundary composite refit (per-signal isotonic curves, per-benchmark shrinkage child fits, boost-layer weights and intercept) is cycle-indexed. **Two registered signals are retired** from the deployed composite: `h_norm` (normalised semantic entropy, registered then retired at the cycle-zero audit; the boost coefficient under the locked configuration is approximately −5e-4) and `p_contra` (structurally zero under the MiniCheck-only deployment).

### The calibrated probability composite + fixed-threshold storage gate

The verifier's nine signals enter a **per-benchmark calibrated probability composite**: per-signal isotonic regression maps each raw signal to a calibrated probability, per-benchmark child curves are shrunk toward the pooled fit through a James-Stein-style prior at α = 0.6, and an L2-regularised logistic boost layer (Cherian-style, C = 0.01) aggregates the per-signal calibrated log-odds into `u_stored ∈ [0, 1]`. The storage gate is a **fixed-threshold rule** on `u_stored`: store if ≥ 0.60, defer if ≥ 0.45, abstain if below 0.45 and `p_ground_max` < 0.20, discard otherwise. The thresholds are registered before the main run and held constant across cycles; only the composite refits. The fixed-threshold formulation replaced an earlier split-conformal gate that failed the marginal-coverage premise on the panel; the rejection evidence is recorded in Chapter 4 §4.7 and Chapter 5 §5.5.1.

### The constrained self-improvement loop

At every committed cycle boundary the runner executes five passes: **retroverify** every stored episode under the locked verifier ensemble against the updated generator's outputs and the cycle-boundary-refit composite, pruning entries whose re-scored composite falls below τ_retro = 0.50; **reconsider** every entry in the deferred buffer with a three-way branch (promote, drop, keep) under TTL = 2 cycles; **consolidate** near-duplicate-meaning clusters under a cosine-similarity floor of 0.92, an answer-equality guard, and a `u_stored`-spread guard at 0.15; **fine-tune** under LoRA (r = 32, α = 64, dropout = 0.05, all seven linear projections, lr = 2e-4, three epochs, effective batch 16) on the strict training pool that admits only entries clearing τ_train = 0.70 (strictly above τ_store = 0.60 so borderline entries do not contaminate the gradient); and **probe** the registered multi-modal retention panel (MMLU general-knowledge multiple-choice, TriviaQA-test held-out open-text factoid, CommonsenseQA-test held-out commonsense multiple-choice). The retention guard aborts the cycle and rolls the adapter back to the previous committed checkpoint when the worst per-probe ratio falls below ρ_min = 0.93, **while the memory store is preserved across the rollback** under the asymmetric-rollback contract.

The full per-stage specification, the algorithmic pseudocode, the cycle-boundary sequencing, and the theoretical scaffolding are in Chapter 4 of the thesis.

---

## Headline result

The realised programme closed at the **six-cycle horizon** (cycles C0 through C5) under the equilibrium-saturation gate, with cycle four aborted by the retention guard and weights rolled back to the cycle-three checkpoint while the memory store grew across the boundary under the asymmetric-rollback contract. The headline contrast frames CAEM against the matched-protocol zero-shot baseline (`B1`) over the same Qwen-2.5-3B-Instruct backbone.

### Per-cycle pooled trajectory (canonical reading from `tab:pooled-trajectory`, Ch5 §5.2)

| Configuration | Cycle | Pooled EM | Pooled CHM | MMLU ratio | TQA-test ratio | CSQA-test ratio | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| **B1 zero-shot (reference)** | — | **0.507** | **0.120** | — | — | — | reference floor |
| CAEM | C0 | 0.541 | 0.135 | — | — | — | baseline |
| CAEM | C1 | 0.543 | 0.112 | 1.041 | 1.013 | **1.006** | COMMIT |
| **CAEM** | **C2 ⭐** | **0.544** | **0.107** | 0.975 | **0.949** | 0.982 | **COMMIT (headline cycle)** |
| CAEM | C3 | 0.532 | 0.115 | 0.992 | 0.987 | **0.976** | COMMIT |
| CAEM | C4 | 0.532 | 0.115 | 1.018 | **0.886** | 0.982 | **ABORT** (worst probe < ρ_min) |
| CAEM | C5 | 0.517 | 0.109 | 0.999 | 1.013 | **0.958** | COMMIT (post-rollback recovery) |

Headline contrast vs B1 zero-shot at the trajectory's within-best cycle (C2): **ΔEM = +0.037 (+7.4% relative)**, **ΔCHM = −0.013 (−10.8% relative)**.

The trajectory carries three findings. First, **the composite hallucination metric reduces monotonically into a deep minimum at cycle two (0.107) and recovers a near-minimum at cycle five (0.109)**, a −19% relative reduction from the cycle-zero baseline (Ch5 §5.2). Second, **cycle four exhibits the first in-flight retention rollback** on the realised trajectory: the TriviaQA-test probe ratio falls to 0.886, crossing below the registered floor of 0.93, the cycle aborts, and the adapter is restored to the cycle-three checkpoint while the memory store grows across the boundary (Ch5 §5.2; the architectural contract is `cor:stochastic-equilibrium` sub-claim iii). Third, **cycle five recovers to a worst-probe ratio of 0.958**, well clear of the floor, and the realised commit fraction closes at π_commit = 4/5 = 0.80, strictly in the open interval (0, 1) that the corollary's sub-claim (i) predicts.

### Eight-baseline panel (canonical reading from `tab:baseline-pooled`, Ch5 §5.4)

| # | System | EM | CHM | ΔEM vs CAEM C2 (best) | ΔCHM vs CAEM C2 (best) |
|---|---|---:|---:|---:|---:|
|   | **CAEM C2 (best)** | **0.544** | **0.107** | — | — |
|   | CAEM C5 (final) | 0.517 | 0.109 | −0.027 | +0.002 |
| B1 | zero-shot | 0.507 | 0.120 | +0.037 (+7.4%) | −0.013 (−11.0%) |
| B2 | chain-of-thought | 0.500 | 0.123 | +0.044 (+8.8%) | −0.016 (−12.7%) |
| B3 | DPR-RAG | ... | ... | ... | ... |
| B4 | CoT + DPR-RAG | ... | ... | ... | ... |
| B5 | five-shot CoT | ... | ... | ... | ... |
| B6 | vanilla fine-tune | ... | ... | ... | ... |
| B7 | FLARE | ... | ... | ... | ... |
| B8 | semantic entropy | ... | ... | ... | ... |

The full per-system numbers, the paired McNemar verdicts, the Holm-corrected family-wise significance panel, and the bootstrap 95% confidence intervals on every ΔEM are reported in Chapter 5 §5.4 and `tab:sig-test`.

### Retrieval-utility classifier on/off ablation (canonical reading from `tab:ruc-ablation`, Ch5 §5.7)

Head-to-head ablation at the cycle-three close under the matched-protocol contract. Both arms share the same five-benchmark panel, the same Qwen-2.5-3B-Instruct backbone, and the same composite calibration pinned at the cycle-three anchor. The classifier-off arm routes safety-override and score-formula fall-throughs unconditionally to the retrieval-augmented tier; the classifier-on arm routes through the retrieval-utility classifier with τ_RUC = 0.375.

| Benchmark | EM off | EM on | ΔEM (pp) | CHM off | CHM on | ΔCHM (% rel) | T3 share off | T3 share on |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FEVER | 0.503 | 0.503 | 0.0 | 0.117 | 0.120 | +2.5% | 92.3% | 70.7% |
| TriviaQA | 0.437 | 0.463 | +2.7 | 0.123 | 0.121 | −2.0% | 95.7% | 68.0% |
| **CommonsenseQA** | 0.603 | **0.743** | **+14.0** | 0.131 | **0.086** | **−34.4%** | 57.7% | 20.3% |
| StrategyQA (transfer) | 0.617 | 0.640 | +2.3 | 0.119 | 0.128 | +8.1% | 100.0% | 76.7% |
| **TruthfulQA (transfer)** | 0.210 | **0.310** | **+10.0** | 0.116 | 0.119 | +2.9% | 99.0% | 40.0% |
| **Pooled (all five)** | **0.474** | **0.532** | **+5.8** | **0.121** | **0.115** | **−5.2%** | **88.9%** | **55.1%** |

The asymmetric-rescue pattern is read off the bold cells: the largest exact-match gains land on CommonsenseQA and TruthfulQA — the two benchmarks where retrieval was actively hurting in the classifier-off arm. FEVER exact match is flat, the empirical receipt that the classifier correctly preserves the retrieval-helpful regime on the textbook fact-verification benchmark. The pooled retrieval-augmented tier share collapses by 33.8 percentage points, with the largest individual collapse on TruthfulQA at 59 percentage points — the empirical receipt that the classifier engages the misconception-amplification mechanism it was registered against.

### Per-tier EM decomposition (canonical reading from `tab:per-tier-em`, Ch5 §5.4)

The realised T2 EM at C2 (51.6 FEVER, 51.0 TQA, 72.4 CSQA, 35.3 TruthfulQA, 66.4 StrategyQA) compared against the C0 base T3 EM (43.5 FEVER, 46.4 TQA, 52.0 CSQA, 29.1 TruthfulQA, 59.9 StrategyQA) on the same five-benchmark panel supplies the **SIL-as-distillation receipt**: the zero-shot tier under the SIL-fine-tuned adapter outperforms the retrieval-augmented tier under the cycle-zero pristine generator on every benchmark by margins of 5–20 percentage points. The Tier 1 (memory-direct) tier fires on a vanishing fraction of held-out queries by design under the content-hash-disjoint evaluation contract (7 firings across 9000 queries in the realised horizon, all returning the correct stored answer), so the Tier 1 trajectory is a capability bound rather than a trajectory prediction; the framing is `cor:tier1-floor` in Chapter 4 §4.10 and the receipt is in Chapter 5 §5.4.

---

## Cycle-zero validation gate verdict (canonical reading from `tab:validation-gate`, Ch5 §5.1)

A pre-registered conjunction of four checks gates the calibration phase against the main run:

| Check | Registered floor | Realised reading | Verdict |
|---|---:|---:|---|
| Pooled-ID composite Cohen's d on the held-out training-panel evaluation fold | ≥ 0.20 | **+0.635** | PASS (3× margin) |
| Per-benchmark store-vs-discard mean-composite gap | ≥ 0.05 each | FEVER 0.282; TQA 0.320; CSQA 0.258 | PASS (every gap clears 5×) |
| Per-benchmark memory-poisoning rate (share of stored entries whose stored answer is wrong) | ≤ 0.50 | FEVER 0.206; TQA 0.179; CSQA 0.247 | PASS (every rate clears 2×) |
| No strong-direction signal inversion against the registered direction | none observed | no inversion observed | PASS |

The out-of-distribution transfer diagnostic reads pooled-transfer composite Cohen's d = +0.173 on the transfer panel; this is a **diagnostic** rather than a gate (the registered scope of the gate audits the cal-fold-versus-eval-fold relationship on the training panel). The four-check verdict is PASS; the locked configuration enters the main run.

The first calibration iteration failed the gate (a signal-dump-bug that wrote only a subset of the nine per-signal values per sample produced a degenerate composite); the bug was identified by the validation-gate diagnostic, fixed in a worktree patch, and the calibration re-run on the corrected pipeline. The iteration is reported transparently in Chapter 5 §5.1 because it is empirical evidence that the validation-gate discipline catches what it was registered to catch.

---

## Pre-registered hypothesis verdicts (canonical reading from `tab:hypothesis-verdicts`, Ch5 §5.5.2)

The thesis registers five falsifiable hypotheses ahead of the empirical chapters; the realised cycle-five verdicts are:

| ID | Hypothesis | Verdict | Deciding statistic |
|---|---|---|---|
| H1 | Verifier balanced accuracy clears one-half precondition every cycle (`thm:purity` precondition) | **supported** | Pooled-ID composite Cohen's d on the cycle-zero eval fold reads +0.635 (≈ 0.673 balanced accuracy), clearing the 0.20 floor 3×; cal-fold pooled π_store at τ_store = 0.60 stays in the band [0.732, 0.746] across every committed cycle |
| H2 | Pool purity non-decreasing under the precondition (`thm:purity`) | **supported** (band reading) | Cal-fold pooled π_store traces 0.746 → 0.740 → 0.736 → 0.746 → 0.732 across the committed-cycle subsequence (C0, C1, C2, C3, C5); realised band width 0.014, comparable to per-cycle binomial sampling noise |
| H3 | Late-cycle pool-purity increments enter a bounded stationary band on the committed-cycle subsequence (`thm:convergence`) | **supported** within horizon | The cal-fold trajectory occupies the band [0.732, 0.746] of width 0.014 across C0–C5, comparable to per-cycle binomial sampling noise. The sharper geometric-rate test is registered as deferred future work pending a horizon of at least fifteen committed cycles |
| H4 | Composite hallucination at the realised horizon approaches a parameter-bounded asymptote (`thm:asymptotic-elim`) | *scope-limited* | Pooled C5 CHM reads 0.109; Tier 1 CHM reads 0 on 7/7 realised firings against the max(1−π_store, 1−π_retro) ≈ 0.36 ceiling of `cor:tier1-floor`. The decomposition is degenerate at τ_1(C5) = 0.0013 under the content-hash-disjoint evaluation contract |
| H5 | Per-cycle Tier 1 hit rate grows monotonically as memory accumulates (operational claim) | *scope-limited* | The evaluation fold is held content-hash disjoint from the training stream by construction, so the memory-direct tier fires on 7 of 9000 queries at within-tier EM = 1.0 on the realised horizon. Per-cycle pooled latency stays in the band [13.6, 14.7] seconds; per-query cost reduction is not measurable within the realised horizon |

The full hypothesis adjudication, including the per-row deciding-statistic prose and the cross-reference back to the theorem-receipt section, is `tab:hypothesis-verdicts` in Chapter 5 §5.5.2.

---

## Theorem-by-theorem closeout (canonical reading from Ch4 §4.11 + Ch5 §5.5.1)

Three theorems and three corollaries carry the formal load of the architecture; each is paired with an empirical receipt at the realised horizon:

| Statement | What it claims | Math proof status | Empirical receipt | Verdict |
|---|---|---|---|---|
| **`thm:purity`** (Pool purity invariant) | Pool purity at cycle t+1 is bounded below by min(π_store, π_retro) on the pooled training-panel cal-fold axis | Complete | Cal-fold pooled π_store ∈ [0.732, 0.746] across C0–C5, clearing the 0.64 floor by 9–11 pp | **supported** within scope (pooled-axis) |
| **`thm:monotone`** (Monotonicity under generator improvement; verifier locked) | Sign of ∂σ/∂d equals sign of (p+·φ+ − p−·φ−) at the locked threshold under within-class Gaussian assumption | Sketch (one differentiation argument) | 1 of 3 cal-fold transitions sign-aligns at large amplitude; 1 sub-noise; 1 confounded by composite refit | **consistent within horizon**, sharp test deferred (fixed-composite ablation registered as Phase 1f future work) |
| **`thm:convergence`** (Bounded-band stationarity on the committed subsequence) | The gap G_t = |π_t − π_∞| contracts geometrically in expectation on the committed-cycle subsequence | Sketch + Robbins-Monro/Polyak-Juditsky lineage; full proof registered as future work | Pooled π_t band width 0.014 over six cycles, comparable to binomial noise | **supported** within horizon; geometric-rate test deferred to ≥15-committed-cycle horizon |
| **`cor:tier1-floor`** (Tier 1 capability bound) | limsup CHM_1(t) ≤ max(1−π_store, 1−π_retro) | Complete (chains `thm:purity` + `thm:convergence`) | CHM_1 = 0 on 7/7 observed firings; trajectory unobservable under content-hash-disjoint eval contract | **supported** (capability-bound form) |
| **`cor:self-correction`** (Self-correction under locked-verifier + advancing-generator) | P(Y=1 \| survives k passes) → 1 as k → ∞; expected prune time for hallucinated entry is finite | Complete (2-paragraph sketch: likelihood-ratio + geometric-tail) | Per-cycle prune rate declines monotonically 13% → 9% → 7% → 4% across the four committed retroverify passes | **directional** within horizon; k → ∞ test deferred |
| **`cor:stochastic-equilibrium`** (Stochastic equilibrium under retention rollback) | (i) commit fraction ∈ (0, 1]; (ii) calendar-time envelope = (1 − π_commit·r)^t; (iii) memory grows monotonically across aborts; (iv) trajectory enters a near-floor band rather than diverging | Sketch | Realised commit pattern (1,1,1,0,1) → π_commit = 0.80 ∈ (0,1); memory grew across C4 abort; worst-probe ratios within 0.07 of floor at C2–C5 | (i), (iii) **supported**; (ii) inherited from thm:convergence; (iv) **heuristic** within horizon, 15-cycle probe registered as future work |

The full proof sketches, the empirical receipts with their tabulated readings, and the threats-to-validity disclosure on the sketch-only proofs are in Chapter 4 §4.11 and Chapter 5 §5.5.1.

---

## Locked configuration (canonical reading from Appendix A §A.3)

Every constant that governs runtime behaviour lives in `caem/config.py` (and `caem/ruc/v2/feature_spec.json` for the classifier). The registry below pulls the headline values from Appendix A.

| Component | Field | Value | Role |
|---|---|---:|---|
| Storage gate | `tau_store` | **0.60** | calibrated-probability cut for STORE |
| Storage gate | `tau_defer` | **0.45** | calibrated-probability cut for DEFER |
| Storage gate | `tau_retro` | 0.50 | retroactive prune floor |
| Storage gate | abstain `p_ground_max` ceiling | 0.20 | splits ABSTAIN from DISCARD when `u_stored < tau_defer` |
| SIL admission | `tau_train` | **0.70** | strict training-pool floor (above `tau_store`) |
| Retrieval-utility classifier | `tau_RUC` | **0.375** | classifier-side cut on `p_RAG` |
| Retention guard | `rho_min` | **0.93** | worst-probe-ratio floor against pristine baseline |
| Retention guard | `retention_probes` | MMLU, TriviaQA-test, CommonsenseQA-test | multi-modal probe panel |
| Retention guard | `retention_probe_n` | 200 | samples per probe |
| Router | `routing_lambda` | 0.70 | combined-score weight on similarity vs `u_stored` |
| Router | `tier1_combined_threshold` | 0.90 | Tier 1 dispatch floor |
| Router | `tier2_similarity_threshold` | 0.75 | Tier 2 dispatch floor |
| Router | `safety_u_pre_min` | 0.38 | safety-override floor on `u_pre`; re-tuned at cycle one from the registered 0.60 |
| Composite | shrinkage prior `alpha` | 0.6 | per-benchmark shrinkage toward pooled fit |
| Composite | boost `C` | 0.01 | L2 regulariser on Cherian boost layer |
| Composite | active signals | 9 | `u_token, u_dropout, u_internal, s_avg, p_entail, p_ground_max, p_ground_mean, p_ground_atomic, q_a_rel` |
| Composite | retired signals | 2 | `h_norm` (registered then retired at C0; boost coefficient ≈ −5e-4); `p_contra` (structurally zero under MiniCheck) |
| Confabulation early-exit | trigger | `u_internal ≥ 0.70 ∧ p_ground_max ≤ 0.20` | precedence over composite at query time; disabled at retroverify time |
| Memory consolidation | `tau_cons` | 0.92 | cosine-similarity floor for near-duplicate-meaning clusters |
| Deferred buffer | TTL | 2 cycles | promote/drop/keep three-way branch |
| Generator | backbone | Qwen-2.5-3B-Instruct | three billion parameters, frozen across cycle |
| Generator | LoRA | r = 32, α = 64, dropout = 0.05 | seven linear projections, lr = 2e-4, three epochs, effective batch 16 |
| Trajectory | registered cap | 10 cycles | upper bound on the SIL loop |
| Trajectory | realised closure | **6 cycles (C0 through C5)** | equilibrium-saturation gate fired at the cycle-five close after the post-rollback subsequence stabilised |

---

## Repository layout

```
caem/                         core package (pipeline, memory, routing, verification, training)
  config.py                   central CAEMConfig dataclass; every threshold and structural toggle
  pipeline.py                 serial per-query driver (production and demo paths)
  pipeline_batch.py           batched per-cycle driver (experiment runner)
  memory/                     storage gate, deferred buffer, SBERT encoder
  routing/                    three-tier router and the retrieval-utility classifier head
  verification/               nine-signal verifier ensemble; adaptive NLI dispatch; cal-prob composite
  training/                   SIL trainer; retention probe; pool reweighting; loop filter
  retrieval/                  FAISS-backed passage index reader for Tier 3 grounded generation
scripts/                      cycle runner, calibration scripts, baseline drivers, aggregators
eval/                         shared per-benchmark harness (EM, em_llm_judged, CHM, CES)
tests/                        pytest suite (storage gate, deferred buffer, retention, SIL, verifier)
docs/                         production runbook, demo quickstart, panel-demo script
thesis_report/                LaTeX thesis source (Ch1-Ch6 + 7 appendices + auto-generated tables)
  chapters/                   chapter_1 through chapter_6
  appendix/                   appendix_a through appendix_g
  figures/auto/               auto-generated empirical tables and figures
  main.pdf                    compiled thesis (325 pages)
archive/                      historical design diaries and earlier-phase runners
data/                         passage index, calibration pairs, alias dictionary (gitignored)
hf_cache/                     Hugging Face cache (gitignored)
outputs/                      per-cycle artefacts and logs (gitignored; mirrored on Google Drive)
```

A detailed module map with the role of each file is in Appendix A §A.2 (`thesis_report/appendix/appendix_a.tex`).

---

## Reproducibility

Every reported number is regenerable from the realised seed plus the published artefact bundle. The full step-by-step recipe is Appendix E; the headline sequence is:

1. **Clone this repository** and pin to commit `c5b8066` (pre-Step-7 anchor) or `700b9e3` (retrieval-utility-classifier-era anchor), depending on the trajectory phase to reproduce.
2. **Pull the realised artefacts** from the Google Drive folder at <https://drive.google.com/drive/folders/1xErhsRDYe_qO0ZgUBn0WXK81eHc5jZCa> into `outputs/`. The realised six-cycle trajectory sits under `v2_1_phase1e_main/`; the per-commit pre-main snapshot bundles sit under `pre_main_snapshot/`.
3. **Pull the Wikipedia FAISS passage index** from the Hugging Face dataset host `aksaN000/caem-passage-index-21m` (private; access on request) into `data/`. The index is 64 GB on disk and requires approximately 90 GB of resident page cache to mmap.
4. **Run the verification checklist** in Appendix E §E.4 against the per-sample evaluation files, the composite calibration record, and the per-cycle artefact catalogue. Headline tolerances: per-benchmark EM ±0.5 pp from the cuDNN attention path's non-determinism floor; pooled EM ±0.3 pp; CHM and π_store ±1.0 pp; integer quantities (commit fraction, win-and-loss tallies) exact.
5. **The four artefacts that must be regenerated rather than pulled from snapshot** are the Qwen-2.5-3B-Instruct base weights (recovered from the Hugging Face model hub at deployment time), the per-cycle LoRA adapters at cycles one through five, the per-cycle self-improvement training artefacts, and the post-cycle-zero composite refits.

### Hardware envelope

| Phase | Wall-clock | Cost |
|---|---|---|
| Cycle-zero calibration phase | tens of GPU-hours | dominated by the verifier-signal sweep over the calibration fold |
| Realised six-cycle main run | several GPU-days | dominated by per-cycle fine-tune plus cycle-boundary retroactive re-verification |
| External baseline panel | scales with sample count and decode-pass count | CoT and retrieval-augmented variants are the bulk |
| Retrieval-utility classifier on/off ablation | single contrast at C3 close | matched-protocol panel |
| **Realised total** | **~294 GPU-hours over ~40 calendar days** | **~163 USD on a Vast.ai RTX 5090 at ~0.70 USD/on-demand GPU-hour (with spot-pricing windows lower)** |

The verifier-side scoring path (nine-signal extraction across every served answer, the per-cycle composite refit, the cycle-boundary retroactive re-verification pass, and the matched-protocol Haiku-judged TruthfulQA rescore across the baseline panel) is the **dominant wall-clock consumer over the run, exceeding the per-cycle low-rank-adapter fine-tune itself**, because every entry on every cycle and every baseline-arm answer passes through the full verifier ensemble while the adapter touches only the strict training-pool subset.

A downgraded sixteen-gigabyte-GPU reproduction envelope with five parameter changes (SIL batch 4 → 2, gradient accumulation 4 → 16, bfloat16 → float16 with dynamic loss scaling, generation cap 512 → 192, activation checkpointing enabled) reproduces the trajectory at approximately 2.2–2.6× the wall-clock penalty. The full envelope, the framework version pins, and the downgraded reproduction profile are in Appendix E §E.5.

---

## Threats to validity (canonical reading from Ch5 §5.6 and Ch6 §6.2)

The thesis discloses every threat to validity transparently rather than smoothing it into the narrative. The headline disclosures are:

1. **Calibration-to-evaluation transfer cost**: a split-conformal alternative dropped 15–50 percentage points below its registered coverage target on the training-panel benchmarks at cycle zero; the fixed-threshold replacement is robust to the shift but the per-cycle drift of the evaluation-fold precision anchor at the locked cut-point is empirically measurable rather than theoretically characterised. Discussed in Ch5 §5.6 (`sec:adj-cal-eval-gap`) and Ch6 §6.2.
2. **Sketch-only proofs on three of six load-bearing statements** (`thm:monotone`, `thm:convergence`, `cor:stochastic-equilibrium`): the rigorous supermartingale construction for `thm:convergence` is registered as future work; the fixed-composite ablation that would convert `thm:monotone` from a consistency check into a sharp test is registered as a Phase 1f item in Ch6 §6.2; the 15-committed-cycle stochastic-equilibrium probe that would adjudicate `cor:stochastic-equilibrium` sub-claim (iv) is registered alongside.
3. **Long-hypothesis judge ablation**: the registered long-hypothesis Qwen judge failed the pre-registered Pearson correlation floor at the v5 calibration overlap fold and is ablated in the deployed configuration; the architecture's coverage on long-form benchmarks is narrowed and the disagreement reading is reported in Ch5 §5.6 as a future-work caveat.
4. **Single-seed scope**: the random-state surface is capped at one recorded seed; the cuDNN attention path is non-bitwise-deterministic, so determinism is enforced at the metric level rather than the bit level. The ±0.5 pp per-benchmark EM stochastic floor in the reproducibility recipe absorbs this.
5. **Six-cycle horizon**: the realised horizon is shorter than the at-least-fifteen-committed-cycle precondition under which the geometric-rate test of `thm:convergence` would be statistically separable from binomial sampling noise; the bounded-band reading is the within-horizon receipt and the sharper test is deferred to future work.
6. **TruthfulQA strict-EM stricture**: the strict exact-match column penalises CAEM's calibrated abstention as a failure on misconception-bait probes where the matched-protocol baseline B1 (which is not licensed to abstain under a zero-shot prompt) gets credit for a confidently-stated wrong answer; the architectural fix is the retrieval-utility classifier on/off ablation reported in §5.7, which lifts TruthfulQA exact match by +10.0 pp by re-routing misconception-bait queries away from the retrieval-augmented tier.
7. **Single-backbone scope**: every empirical reading is on Qwen-2.5-3B-Instruct. The architectural-claim transfer to larger backbones is registered as a future-work item in Ch6 §6.2.

---

## Future work (canonical reading from Ch6 §6.2)

The thesis registers six future-work items with concrete experimental commitments:

1. **Fixed-composite ablation** for `thm:monotone` (freeze the composite refit across two adjacent cycles; vary only the generator).
2. **15-committed-cycle stochastic-equilibrium probe** for `cor:stochastic-equilibrium` sub-claim (iv) with a Wald-Wolfowitz runs test + Bernoulli rate fit + rolling-mean stability check.
3. **Rigorous supermartingale proof** for `thm:convergence` adapted from Robbins-Monro and Polyak-Juditsky in a dedicated proofs appendix.
4. **Paraphrase-eval probe** for `cor:tier1-floor` that lifts the 0.13% Tier 1 firing rate enough to make the CHM_1(t) trajectory observable.
5. **Larger-backbone scaling study** (7B and 13B in the same open-weight family) to test architectural transfer.
6. **Long-hypothesis judge re-licence** under a task-specific fine-tune on a synthetic claim-support corpus.

---

## Citation

If you reference CAEM or build on this work, please cite:

```bibtex
@misc{alif2026caem,
  author = {Aksan Gony Alif},
  title  = {Confidence-Aware Episodic Memory with Self-Improvement: An Architectural Approach to Hallucination Reduction in Open-Domain Question Answering},
  year   = {2026},
  note   = {CSE400 Undergraduate Thesis, BRAC University},
  url    = {https://github.com/aksaN000/caem-thesis}
}
```

---

## Licence

Released under the [MIT licence](LICENSE).
