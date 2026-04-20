# Branch C — CAEM Phase 2 Upgrade Plan

**Status**: planning document, not yet executed
**Author**: integrated from 2026-04-20/21 planning discussions + `literature_review_2` (commit `48dd806`)
**Timeline**: 5-month horizon to CSE400 defense (~2026-09)
**Target end-state**: `feat/phase-2-all` → `main` merge, complete thesis rewrite, defended

---

## Table of contents

1. [Why branch C exists](#why-branch-c-exists)
2. [Research-contribution spine](#research-contribution-spine)
3. [Design decisions (what we're changing and why)](#design-decisions)
4. [Evaluation protocol (new requirements from literature review)](#evaluation-protocol)
5. [Theoretical framing updates](#theoretical-framing-updates)
6. [Branch structure and merge discipline](#branch-structure)
7. [Calibration discipline](#calibration-discipline)
8. [Chapter edit roadmap](#chapter-edit-roadmap)
9. [Ablation registry expansion](#ablation-registry-expansion)
10. [Baseline panel expansion](#baseline-panel-expansion)
11. [Timeline and cost](#timeline-and-cost)
12. [Risk register](#risk-register)
13. [Definition of done](#definition-of-done)

---

## Why branch C exists

Phase 1a (currently running on `main`, Flan-T5-Large) is the architecture proof-of-concept.
It will finish ~2026-05-07 and produce defensible-but-modest headline numbers on a
2022-era base model. It is NOT the thesis-as-defended version.

**Branch C** is the integration branch (`feat/phase-2-all`) where four upgrades merge
into a publication-grade CAEM:

- **Goal 1** — Qwen2.5-3B-Instruct base generator (replaces Flan-T5-Large)
- **Goal 2** — Question↔answer relevance signal (one new signal, NOT verifier ensemble)
- **Goal 3** — Retrieval upgrade (DEMOTED — ablation-only, not main-run)
- **Goal 4** — Memory hygiene (loop filter, consolidation, invalidation)

Each goal is developed on its own feature branch, then integrated on `feat/phase-2-all`.
Phase 1a's Flan-T5 data becomes the `flan_t5_large_backbone` ablation row — preserved,
not thrown away.

**Key constraint — Phase 1a data is preserved intact:**

- `outputs/full_run/` (10-cycle trajectory on Flan-T5-Large) feeds Ch5 Table 5.Z as an
  ablation row demonstrating CAEM works across base-model generations
- `outputs/purity_validation/theory_validation.json` is the Flan-T5 α baseline
- `outputs/cycle_0/calibrated_thresholds.json` is preserved for per-backbone τ comparison

---

## Research-contribution spine

Per `literature_review_2`, CAEM's defensible novelty rests on **three axes** none of
which are jointly covered by prior work (RF-1 through RF-5):

1. **External multi-signal verifier** — not self-prediction (distinguishes from V-STaR,
   rStar-Math, Huang 2025 sharpening)
2. **Generative LLM setting** — not binary classification (distinguishes from Das 2025
   ICML's classification purity theorem, which is the closest structural analogue)
3. **Explicit finite-round α > ½ ⇒ P > p inequality** — not convergence-only bounds
   (distinguishes from Huang 2025, Fu 2025, Lang et al. 2024 weak-to-strong)

**Plus a fourth, architectural-level novelty** flagged in the review as CAEM's quietest zone:

4. **Insertion-time scalar quality stored with memory entries, routed across reliability
   tiers** — no surveyed paper does this. Adaptive-RAG (RF-3) shares the 3-tier surface
   but routes on query-time complexity; A-MEM shares insertion-time augmentation but uses
   textual metadata; SeaKR computes confidence generator-side at query time.

**Thesis spine sentence (draft for Ch1 and Ch6):**

> CAEM's novelty is (a) a per-entry scalar confidence `u_stored` written at memory-insertion
> time and used to route subsequent queries across three reliability tiers, combined with
> (b) a finite-round α > ½ ⇒ P > p inequality that guarantees stored-memory accuracy
> exceeds base-generator accuracy under a multi-signal external verifier. Together these
> extend the classification-setting Das et al. 2025 purity result to open-ended generation
> and produce a tiered-memory routing architecture absent from Adaptive-RAG, A-MEM, and
> SeaKR.

**What we are NOT claiming** (to avoid over-sell):

- We do not claim the first or best verifier ensemble — Valentin 2024 already builds a
  4-signal verifier; our differentiation is the *loop around it*, not the verifier itself
- We do not claim a new self-improvement bound beyond the purity inequality — the
  convergence story leans on Fu 2025 and Song 2024
- We do not claim a new retrieval method — retriever is held constant on DPR in the main
  run; retrieval-upgrade ablation is a sensitivity row, not a contribution
- We do not claim a better episodic memory writer than A-MEM — we claim a new use
  (tier routing) for the same storage

---

## Design decisions

### Goal 1 — Qwen2.5-3B-Instruct base generator

**Choice**: Qwen2.5-3B-Instruct (Apache 2.0, decoder-only, 3B parameters, ~99% label
compliance, <2% loop rate, ChatML format)

**Why not Qwen2.5-7B**: headroom paradox — 7B's Cycle-0 EM on FEVER is ~85%, leaving
only ~7pp for SIL to demonstrate. 3B's Cycle-0 is ~70%, leaving ~18pp. Clean SIL
narrative needs headroom.

**Why not Qwen2.5-1.5B**: "small model" examiner objection risk; 3B is the sweet spot.

**Why not Flan-T5-XXL (11B)**: still T5 architecture, still encoder-decoder, still 2022
training era, still occasional loops. Also slower per-token than Qwen-3B despite being
larger param count (encoder-decoder dual-pass penalty). Qwen-3B wins on every axis.

**Training strategy**: **LoRA rank-16 on attention + MLP**, merged every 2 cycles.
Full FT of 3B won't fit 32 GB VRAM (needs ~48 GB); LoRA fits comfortably. This aligns
with Ch4's existing O-LoRA structured-fallback paragraph — we promote LoRA from
"fallback" to "primary" training strategy.

**Ch4 update**: §Self-improvement paragraph — LoRA becomes the primary SIL training
strategy (not fallback); MMLU retention guard and L2 anchor semantics unchanged.

### Goal 2 — Question↔answer relevance signal (ONE new signal, NO ensemble)

**Decision locked 2026-04-21**: keep MiniCheck as the sole entailment judge; add one
new signal for question↔answer relevance. No Gemma, no AlignScore, no LLM judge.

**Composite (7 signals, weights sum to 1.0):**

| Signal | Weight | Source |
|---|---|---|
| `p_ground_mean` | 0.28 | grounding, best-passage-avg (unchanged) |
| `p_ground_atomic` | 0.14 | grounding, per-atomic-fact (unchanged) |
| `p_entail` | 0.16 | MiniCheck, unchanged — NOT an ensemble |
| **`q_a_relevance`** | **0.14** | **NEW: BGE cross-encoder question↔answer cosine** |
| `s_avg` | 0.14 | self-consistency pairwise SBERT (unchanged) |
| `u_internal` | 0.10 | token-prob + MC-dropout (unchanged) |
| `h_norm` | 0.04 | 1 − normalized semantic entropy (reduced) |

**What the new signal fixes**: sample ② failure mode observed at Cycle 0 on Phase 1a —
hallucinated off-topic answer with high `p_entail` (passage matched hallucination) and
high `p_ground_max`. Neither entailment nor grounding catches question-relevance gap.
`q_a_relevance` closes this by scoring whether the answer semantically addresses the
question, independently of which passage was retrieved.

**Why this is CAEM's contribution (not just "another signal")**: entailment judges
check claim↔passage; grounding checks passage retrieval; no standard signal checks
question↔answer relevance in verification pipelines. This is a pipeline-level
contribution: identifying the orthogonal-to-entailment axis of verification failure.

**Implementation**: BGE-reranker-v2-m3 or `cross-encoder/ms-marco-MiniLM-L6-v2` cross-
encoder scoring `(question, display_answer)` pairs, ~0.2 GB VRAM, ~300 ms per query.

**Ablation to register**: `no_q_a_relevance` variant (zero weight on q_a_relevance,
redistribute the 0.14 across remaining 6 signals) so Ch5 can quantify the signal's
marginal contribution.

### Goal 3 — Retrieval upgrade (DEMOTED to ablation only)

**Per examiner objection vulnerability**: if CAEM's main run used BGE+BM25+reranker
and baselines used DPR, "how much of the gain is CAEM vs just better recall?" is
devastating.

**Main-run retriever**: **DPR** (unchanged from Phase 1a). Same index, same baselines
(B3 DPR-RAG, B4 CoT+DPR-RAG, B5 FLARE+DPR), same comparison.

**Ablation row**: `bge_hybrid_retriever` variant runs the full CAEM 10-cycle pipeline
with BGE+BM25+reranker+FLARE replacing DPR. Reports both CAEM-with-DPR and
CAEM-with-BGE numbers so the retrieval contribution is isolated from the architecture
contribution.

**Disk implication**: BGE index is 0.4 GB model + ~64 GB FAISS (same IVF-PQ sizing if
we use `bge-base` at 768 dim, matches DPR). Need to hold both indices during the
ablation build. **Upsize Vast disk 150 GB → 250 GB before Goal 3 ablation starts.**

**Index sharing**: the BGE index gets archived to `aksaN000/caem-passage-index-21m`
under a new `bge_base/` subfolder for reproducibility.

### Goal 4 — Memory hygiene

Five fixes, all model-agnostic:

1. **Loop filter in SIL pool curation** — `_collect_episodes` in
   `caem/training/self_improvement.py` gains `_is_repetitive_loop` check (distinct-4
   n-gram + compression-ratio two-signal). Loops fall back to `entry.answer` field
   rather than chain (matches existing empty/too-short fallback pattern).

2. **Loop filter in retroverify** — same check applied at cycle-boundary retroverify;
   loops get pruned from memory (not just filtered from SIL).

3. **Memory consolidation at cycle boundary** — SBERT-cluster stored episodes at
   similarity 0.88, keep max-u representative per cluster, merge retrieval metadata.
   Reduces memory size by ~20-30% over 10 cycles without information loss.

4. **Retroverify downgrade** — current retroverify only raises `u_stored`; extend to
   allow lowering, with prune trigger if new u_stored falls below retroverify-prune
   threshold.

5. **Hit-counter re-verification** — each entry tracks Tier-1 serve count; after
   N=10 hits, forced into next retroverify queue regardless of age. Popular wrong
   answers get caught quickly.

**Expected Phase 1a-side gain** (measured on current Flan-T5 data):
- Loop contamination in SIL pool drops from 30% → <2%
- Memory size at Cycle 10 drops by ~20-30%
- No theorem impact (all changes are hygiene, not decision-logic)

**Implementation order**: Goal 4 is the lowest-risk branch; merge first into
`feat/phase-2-all` after Goal 1 base.

---

## Evaluation protocol

### Mandatory new metrics (from literature review)

**M1 — Self-knowledge correlation (Moskvoretskii 2025)** [CRITICAL, not optional]

Measure Spearman correlation between `u_stored` and an external reliability signal
on held-out samples. Without this, CAEM faces the same critique leveled at DRAGIN,
SeaKR, and Adaptive-RAG: "your confidence signal correlates poorly with actual
self-knowledge."

**Protocol**:
- Sample N=200 STOREd episodes per benchmark from the Cycle-10 memory
- Label each via external judge (human-gold for in-training benchmarks; existing
  gold labels where available)
- Compute Spearman `ρ(u_stored, is_factually_correct)`
- Target: ρ > 0.3 per benchmark; pooled ρ > 0.4
- Report as Ch5 Table 5.X alongside the purity table

**M2 — Prediction-rejection curves (LM-Polygraph, Vashurin 2025 TACL)**

LM-Polygraph is the standard UQ harness with prediction-rejection curves over 14 UQ
methods. CAEM should report:
- Abstention rate vs retained-accuracy curve as `u_stored` threshold varies
- AUACC (Area Under Accuracy-Confidence Curve) — the standardized metric from the
  "Know Your Limits" abstention survey (Wen 2025 TACL)
- Report alongside EM/F1 in Ch5 Table 5.Y

**M3 — Per-benchmark α breakdown** (from Addition 2 of our planning discussion)

Report α both pooled AND per-benchmark (6 registry benches + MMLU footnote):
- If per-benchmark α varies by >5pp from pooled, discuss heterogeneity in Ch5 text
- Sets up per-benchmark stratified thresholds as Phase 2 future work if needed

**M4 — One-sided hypothesis test α > 0.5 with 95% CI**

Target sentence: *"We measure α = 0.XX ± 0.0Y (95% CI), significantly above the
α > 0.5 threshold required by Theorem 4.1 (p < 0.001)."* Bootstrap CI via 1000
resamples on the purity-validation fold.

### Continued existing metrics

- EM / F1 (per-benchmark)
- CES (CAEM Efficacy Score, geometric mean of ACC·EPI·RET·CAL·VER)
- MMLU retention ratio
- ECE (Expected Calibration Error)
- Balanced accuracy per tier
- McNemar + bootstrap CI + Holm sig-tests (Ch5 §Statistical significance)

### Evaluation output artifacts

```
outputs/phase_2_all/
├── experiment_summary.csv              # 11 rows x CES axes
├── per_benchmark_alpha.csv             # M3 output
├── self_knowledge_correlation.csv      # M1 output
├── prediction_rejection_curves.csv     # M2 output
├── bootstrap_ci_alpha.json             # M4 output
├── memory_size_trajectory.csv          # consolidation Goal 4 evidence
├── loop_filter_counts.csv              # Goal 4 hygiene metrics per cycle
├── purity_validation/
│   ├── theory_validation.json
│   ├── per_sample/purity_raw_<bench>_cycle<N>.json  # FIX-8 dumps
│   └── alpha_vs_tau_by_cycle.json      # FIX-8 replay
└── figures/
    ├── alpha_vs_tau_cal_vs_converged.pdf
    ├── alpha_trajectory.pdf
    ├── prediction_rejection_curve.pdf
    ├── u_stored_correlation_scatter.pdf   # M1 visual
    └── memory_consolidation_trajectory.pdf
```

---

## Theoretical framing updates

### Ch4 becomes α-parametric (from Addition 1 of planning discussion)

- Table 4.1 bounds expressed symbolically as functions of α over [0.5, 1.0]
- Purity row: `P = pα / (pα + (1−p)(1−α))` with α as free parameter
- Add α-sensitivity figure (`fig:alpha-sensitivity`): plot P vs α at p ∈ {0.4, 0.5,
  0.6, 0.7} on the same axes
- Shorten Remark 4.3: *"Empirical α is reported in Table 5.x per verifier backend and
  per benchmark; see fig:alpha-sensitivity for interpretation."*
- Any existing Ch4 derivations that hardcode α (e.g., α = 0.80) get generalized to
  symbolic α with a worked example in a footnote

**Effect**: Ch4's theoretical exposition becomes verifier-agnostic. Any future
verifier swap updates only the Ch5 empirical α — never the Ch4 theorem presentation.

### Ch4 gains a "specialist-verifier justification" subsection (~300 words)

Title: **"On specialist verification rather than LLM-as-judge"**

Three paragraphs:

1. **Specialist-vs-generalist verifier**: MiniCheck is a 770M T5-based model fine-tuned
   specifically on LM-generated-claim faithfulness data (Tang et al. 2024). Unlike a
   general-purpose LLM judge, it does not autoregressively generate judgments; it
   outputs a calibrated scalar probability. Cite JudgeBench (Tan et al. 2025 ICLR) —
   *"GPT-4o judges barely beat random on hard factual/logical items"* — as empirical
   justification for the specialist choice.

2. **α > ½, not α = 1**: the purity theorem (Ch4 §Theoretical Analysis) requires only
   that the verifier's balanced accuracy exceed 0.5. A specialist fact-checker with
   70-80% accuracy — well-documented for MiniCheck-class models (Tang 2024) — clears
   this bar by a wide margin. CAEM's claim is not "the verifier is perfect" but "the
   verifier is measurably more accurate than the generator on its task."

3. **Disjoint signal families**: rather than decorrelating via multiple entailment
   judges, CAEM's 10-signal composite achieves multi-perspective checking through
   orthogonal signal types: grounding (retrieval-based), entailment (MiniCheck),
   relevance (question↔answer cross-encoder), self-consistency (pairwise cosine),
   internal uncertainty (token probabilities + MC-dropout), and semantic entropy.
   Each family answers a different question about the generated claim, providing
   independent evidence. Cite Condorcet (Shteingart 2020) for the foundational
   independent-voter aggregation result; CAEM's signal-family decorrelation is the
   architectural realization.

**Placement**: Ch4 §Verifier composite, between the composite definition and the
threshold-calibration paragraph.

### New foundational citations in Ch4

- **Fu et al. 2025** (RF-4) — upstream safety anchor: verifier-filtered self-training
  provably prevents model collapse
- **Song et al. 2024** ("Mind the Gap") — generation-verification gap as the
  continuous analogue of α > ½
- **Das et al. 2025 ICML** (RF-1) — acknowledge as the closest published analogue in
  binary classification; cite CAEM's three-axis differentiation (external verifier,
  generative setting, finite-round inequality)
- **Huang et al. 2025 ICLR** (RF-2) — sharpening mechanism as a parallel self-
  improvement framework
- **Shteingart 2020** — Condorcet's jury theorem for the independent-voter foundation

### Ch2 expansion — 7-topic structure

Current Ch2 related work should be expanded to cover the 7 topics from the review,
with RF-1 through RF-5 explicitly differentiated:

- Topic 1 (UQ for LLMs): Farquhar 2024, Nikitin 2024, Kossen 2024, SNNE 2025, LM-Polygraph
- Topic 2 (Hallucination verification): MiniCheck (Tang 2024), SAFE (Wei 2024),
  JudgeBench (Tan 2025), Valentin 2024 — explicit differentiation required
- Topic 3 (Episodic memory): HippoRAG, A-MEM, EM-LLM, Larimar — CAEM's u_stored
  positioned as the "contextual binding" slot
- Topic 4 (Self-improvement): V-STaR, rStar-Math, Huang 2025 sharpening,
  Shafayat 2025 (collapse warning), Kazdan 2025 (accumulate vs replace)
- Topic 5 (Adaptive retrieval): Adaptive-RAG (differentiate), SeaKR, DRAGIN, RAGate,
  Moskvoretskii 2025 (evaluation protocol)
- Topic 6 (Continual learning): Song 2025 (L2 regularization, RF-5 differentiation),
  STABLE, GeRe (validates L2 feature anchoring), SuRe
- Topic 7 (Verifier theory): Das 2025 (RF-1), Huang 2025 (RF-2), Song 2024, Fu 2025,
  Lang 2024 (weak-to-strong), CoCoA, Condorcet

---

## Branch structure

```
main (Phase 1a completed, Flan-T5-Large — preserved as ablation source)
  │
  ├── feat/qwen-3b-goal1         (Goal 1: base model swap + port)
  │     └── no calibration commit, only smoke validation
  │
  ├── feat/memory-hygiene        (Goal 4: loop filter + consolidation + invalidation)
  │     └── no calibration commit, only smoke validation
  │
  ├── feat/q-a-relevance         (Goal 2: new signal, no ensemble)
  │     └── no calibration commit, only smoke validation
  │
  ├── feat/retrieval-ablation    (Goal 3: BGE index build for ablation-only)
  │     └── no merge to feat/phase-2-all; runs as separate ablation row
  │
  └── feat/phase-2-all           (INTEGRATION BRANCH — branch C)
        ├── merges Goal 1, 4, 2 in that order
        ├── ONE calibration pass at integration (n_cal=1500 + bootstrap CI + held-out)
        ├── full 10-cycle main run on this branch
        └── when complete, merge → main (thesis-defended version)
```

### Merge order rationale

1. **Goal 1 first** — foundation; every other change depends on the model
2. **Goal 4 second** — lowest risk (pure infrastructure), no theorem impact
3. **Goal 2 third** — adds the new signal, requires composite weight reshuffling
4. **NO calibration between merges** — only smoke validation per branch; single
   calibration at `feat/phase-2-all` integration to avoid the 4-pass-overfitting risk

### Per-branch smoke checklist

Each feature branch must pass before merge to `feat/phase-2-all`:

- [ ] `pytest tests/ -v` — all existing tests pass (fix any test-side regressions)
- [ ] Smoke run (n=50 × 1 cycle × 2 benchmarks) completes without crashes
- [ ] Output eval JSONs contain all required fields
- [ ] No new NaN/Inf in u_stored composite or sub-signals
- [ ] FLARE smoke (n=5 × FEVER) passes — ensures decoder-only generation doesn't
  break the look-ahead active-retrieval baseline

### Integration checklist on `feat/phase-2-all`

Before launching the 22-day main run:

- [ ] All three feature branches merged cleanly (no conflicts)
- [ ] Integrated smoke (n=50 × 1 cycle × 6 benchmarks) passes
- [ ] `pytest tests/ -v` still green
- [ ] Per-sample scalar dumps (FIX-8) wiring intact after merges
- [ ] Calibration script runs on expanded n_cal=1500 fold with bootstrap CI
- [ ] Held-out n=200 validation: fitted thresholds produce target STORE rate (~30%)
  to within ±3pp
- [ ] M1-M4 evaluation harnesses wired and smoke-tested on Cycle-0 data
- [ ] Vast disk upsized to 250 GB (for BGE retrieval ablation later)
- [ ] HF repo `aksaN000/caem-passage-index-21m` has `pre_phase_2_all_snapshot/` subfolder
  uploaded as credit-burnout recovery point

---

## Calibration discipline

**Rule**: one calibration pass, at integration time, no per-branch refits.

### Protocol

1. **Single fit** on `feat/phase-2-all` after all three merges
2. **Expanded fold**: n_cal = 1500 (up from 500) drawn from the disjoint training split
3. **Bootstrap CI**: 1000 resamples; report mean ± 95% CI on fitted thresholds
4. **Per-benchmark stratification reported** alongside pooled fit (even if decision
   logic uses pooled threshold) — documents heterogeneity for Ch5 Appendix
5. **Held-out validation**: n=200 from the same training distribution, never seen during
   fit; confirm fitted thresholds produce target admission rates (STORE ~30%, DEFER
   ~30%) within ±3pp; abort if not
6. **FIX-8 α(τ_store) sensitivity**: after Step 19 purity validation, run the α-vs-τ
   replay per cycle to demonstrate robustness across the [P60, P80] band

### Output

```json
{
  "verifier_backend": "minicheck + q_a_relevance",
  "target_quantiles": {"store": 0.70, "defer": 0.40, "train": 0.90},
  "n_cal": 1500,
  "thresholds": {
    "store": {"value": 0.XX, "ci_95": [0.YY, 0.ZZ]},
    "defer": {"value": 0.XX, "ci_95": [0.YY, 0.ZZ]},
    "train": {"value": 0.XX, "ci_95": [0.YY, 0.ZZ]}
  },
  "per_benchmark_thresholds": {
    "fever": {"store": 0.XX, ...},
    "triviaqa": {...},
    ...
  },
  "held_out_validation": {
    "n": 200,
    "target_store_rate": 0.30,
    "measured_store_rate": 0.XX,
    "within_tolerance": true
  },
  "source_jsons": [...]
}
```

---

## Chapter edit roadmap

### Ch1 — Introduction
- Update the thesis spine sentence (see §Research-contribution spine)
- Three-axis novelty statement in §1.2 contributions

### Ch2 — Related Work
- Expand to 7-topic structure
- Explicit RF-1 through RF-5 differentiation subsections
- Add A-MEM (Xu 2025), Adaptive-RAG (Jeong 2024), Valentin 2024 as primary comparison
  points with specific differentiation paragraphs

### Ch3 — Problem formulation and architecture
- §Base generator: Qwen2.5-3B-Instruct paragraph rewrite
- §Memory architecture: add "insertion-time scalar quality" framing (the Topic 3/5
  novelty) — cite A-MEM as comparison
- §Tier routing: add Adaptive-RAG comparison paragraph — per-entry stored confidence
  vs query-time complexity

### Ch4 — Theoretical analysis and pipeline
- **A1 applied**: α-parametric Table 4.1, symbolic purity formula, α-sensitivity figure
- **A3 applied**: new §Specialist-verifier justification subsection (~300 words)
- §Pre-routing: C_conv decoder-only formulation (matches Nandakishor 2025 original,
  no T5 adaptation needed)
- §Self-improvement: LoRA primary training strategy; cite Song 2025 (RF-5) for L2
  validation; cite Fu 2025 (RF-4) and Kazdan 2025 for non-collapse regime
- §Verifier composite: 7-signal breakdown with `q_a_relevance` paragraph explaining
  the orthogonal-to-entailment novelty
- §Threshold calibration: n_cal=1500 + bootstrap CI + held-out validation paragraph
- Cite Condorcet (Shteingart 2020) foundational independent-voter aggregation

### Ch5 — Empirical validation
- **A2 applied**: per-benchmark α table alongside pooled
- **Empirical requirement applied**: one-sided hypothesis test α > 0.5 with CI,
  target sentence format
- **M1 added**: self-knowledge correlation table (Spearman per benchmark)
- **M2 added**: prediction-rejection curves + AUACC (align with LM-Polygraph format)
- §Verifier backend: rewrite to reflect specialist-verifier + q_a_relevance design
- §Ablation results: add rows per §Ablation registry expansion below
- §Baselines: add A-MEM and Adaptive-RAG as primary baselines alongside existing B1-B7

### Ch6 — Conclusion and future work
- Three-axis novelty restated
- Per-benchmark α heterogeneity discussion (if >5pp)
- Future work:
  - Verifier ensemble (if Step 19 α is marginal) as Phase 3
  - Retrieval upgrade (demoted ablation row results inform future work)
  - Kernel Language Entropy SE primitive upgrade
  - Long-form claim decomposition (SAFE / LongFact integration)
  - Extension to Qwen-7B for scale analysis

---

## Ablation registry expansion

Current registry: 17 variants. Branch C adds:

| # | Variant | Type | Question it answers |
|---|---|---|---|
| 18 | `flan_t5_large_backbone` | inference-time | Does the architecture generalize across base-model generations? |
| 19 | `bge_hybrid_retriever` | inference-time | How much of the gain is architectural vs retrieval-quality? |
| 20 | `no_q_a_relevance` | mechanism | Marginal contribution of the new relevance signal |
| 21 | `valentin_4signal_verifier` | mechanism | Comparison against Valentin 2024's closest competitor composite |
| 22 | `kernel_language_entropy_vs_vanilla_se` | mechanism | Primitive upgrade (Nikitin 2024) sensitivity |
| 23 | `amem_memory_metadata` | architecture | A-MEM-style textual metadata vs CAEM's scalar u_stored |
| 24 | `adaptive_rag_query_time_routing` | architecture | Query-time complexity routing vs stored-confidence routing |

Variants 18-22 are mechanism/inference-time flips (cheap — single runs at n=5000).
Variants 23-24 are architecture-level and require implementing the competitor's
routing/storage logic (more expensive; consider as Phase 3 if timeline tight).

**Total ablation registry**: 17 + 7 = **24 variants**. Register in
`caem/ablation/variants.py`.

---

## Baseline panel expansion

Current baselines B1-B7 (Phase 1a): Zero-shot, CoT, DPR-RAG, CoT+DPR-RAG, FLARE,
Vanilla FT, EWC-only FT.

Branch C adds two primary competitor baselines (from literature review findings):

- **B8 — A-MEM** (Xu et al. 2025 NeurIPS) — insertion-time memory augmentation with
  textual metadata. Implementation: port their public repo code; integrate with
  CAEM's eval harness; run at n=5000 × 6 benchmarks.

- **B9 — Adaptive-RAG** (Jeong et al. 2024 NAACL) — 3-tier query-time complexity
  routing (no retrieval / single-step / multi-step IRCoT). Implementation: their
  public code + T5-Large classifier; integrate with CAEM's eval harness.

Both run on Qwen-3B generator under DPR retrieval to match CAEM main-run conditions.

**Rationale**: without these baselines, CAEM's Topic 3/5 novelty claims are not
empirically defended against the closest competitors.

**Cost**: ~4 hours compute + ~1-2 days implementation per baseline.

---

## Timeline and cost

### Assumes Phase 1a finishes on Flan-T5 before branch C work begins

| Phase | Work | Wall-clock | Compute cost |
|---|---|---|---|
| Finish Phase 1a (current) | Let it run to completion, Flan-T5 data preserved | 16 days | $247 |
| **Branch C port work (no main run)** | | | |
| `feat/qwen-3b-goal1` port + smoke | Code + tests + smoke | 5 days | $5 |
| `feat/memory-hygiene` port + smoke | Code + tests + smoke | 3 days | $2 |
| `feat/q-a-relevance` port + smoke | Code + tests + smoke | 4 days | $3 |
| Ch4/5 α-parametric rewrites + A1 figure | Thesis writing | 2 days (parallel) | $0 |
| Ch2 expansion + 7-topic related work | Thesis writing | 3 days (parallel) | $0 |
| **Integration on `feat/phase-2-all`** | | | |
| Merge + conflict resolution | Git work | 1 day | $1 |
| Integrated smoke | N=50 × 6 benches × 1 cycle | 1 day | $3 |
| Calibration (n=1500 + bootstrap CI + held-out) | One-time fit | 1 day | $5 |
| A-MEM + Adaptive-RAG baseline implementation | Code + smoke | 4 days | $5 |
| Vast disk upsize 150→250 GB (before Goal 3 ablation) | Config + restart | 30 min | — |
| **Branch C main run** | 10-cycle × Qwen-3B + q_a_relevance + memory hygiene | **22 days** | **~$380** |
| Baselines re-run on Qwen-3B (B1-B9) | DPR baselines on new base model | 3 days | $30 |
| BGE retrieval ablation (Variant 19, full 10-cycle run) | Ablation-only | 22 days | **~$380** |
| Other ablations (Variants 18, 20-24, screening mode) | Shorter runs | 5 days | ~$40 |
| Diagnostics + aggregate + calibration audits | FIX-8, M1-M4 reports | 2 days | $5 |
| Ch 2-6 rewrites + tables + figures | Thesis integration | 3 weeks (local) | $0 |
| **Total active compute** | | **~85 days / 12 weeks** | **~$1,100** |
| Viva prep + buffer | With 5-month (22 week) horizon | **10 weeks buffer** | |

### Budget summary

- Phase 1a finish: $247 (already committed)
- Branch C engineering + ablations: ~$850
- Total: **~$1,100** over the 5-month horizon

Within the supervisor-funded Phase 2 budget envelope per your memory notes (~$700),
with ~$400 overshoot — needs supervisor top-up. Alternatively, skip Variants 23-24
(architecture-level competitor baselines) to save ~$50 and implementation time;
defend as Phase 3 future work instead.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Qwen-3B α < 0.65 on some benchmark | Medium | High (theorem margin thin) | Plan Gemma-2-2B ensemble as post-hoc Phase 3 hotfix if it happens |
| MiniCheck fails on compositional claims | Medium | Medium | Monitor per-benchmark α; add AlignScore as fallback if StrategyQA/ARC α < 0.6 |
| Self-knowledge correlation ρ < 0.3 | Medium | High (Moskvoretskii critique lands) | Ch5 acknowledges and frames as Phase 3 direction; thesis still defends tier routing architecturally |
| BGE ablation shows retrieval explains >80% of CAEM-vs-baseline gap | Low | High (contribution undermined) | Frame CAEM as architecturally orthogonal; retrieval upgrade becomes complementary rather than substitutive |
| Vast 5090 unavailable during 22-day main run | Medium | High (restart overhead) | HF snapshot recovery already wired; session-split tolerant |
| LoRA convergence worse than full FT | Low | Medium | Ch4 cites O-LoRA evidence; monitor MMLU retention per cycle |
| Timeline slippage | Medium | Medium | 10-week viva buffer absorbs up to 6 weeks slip |
| Supervisor objection to base-model swap | Low | High | Brief supervisor before `feat/phase-2-all` integration; Phase 1a Flan-T5 result provides fallback defense |

---

## Definition of done

Branch C is complete when all of the following hold:

- [ ] `feat/phase-2-all` merged to `main`
- [ ] All 24 ablation variants have results (reference row + 17 mechanism + 2 architectural + 4 sensitivity; some may be "screening-only")
- [ ] All 9 baselines (B1-B9) run on Qwen-3B backbone
- [ ] Ch4 is verifier-agnostic (α-parametric bounds, symbolic theorems)
- [ ] Ch5 reports pooled α, per-benchmark α, self-knowledge correlation, prediction-rejection curves, and one-sided hypothesis test
- [ ] Ch2 covers 7 topics with RF-1/2/3/4/5 differentiation
- [ ] Three-axis novelty statement present in Ch1, Ch3, Ch6
- [ ] HF snapshot of complete artifacts at `aksaN000/caem-passage-index-21m/phase_2_all_snapshot/`
- [ ] All supporting scripts in `scripts/` pass tests
- [ ] Defense slide deck drafted and reviewed by supervisor
- [ ] Buffer remaining for viva prep

---

## Appendix A — Commit message conventions for this branch

Follow existing repo style (no conventional-commit prefix, descriptive one-line):

- `Goal 1 — port pipeline to Qwen2.5-3B-Instruct (AutoModelForCausalLM + ChatML)`
- `Goal 1 — C_conv decoder-only formulation; drop encoder-specific path`
- `Goal 4 — SIL pool loop filter + retroverify loop pruning`
- `Goal 4 — memory consolidation at cycle boundary (SBERT cluster@0.88)`
- `Goal 2 — add q_a_relevance signal to u_stored composite (7-signal)`
- `Ch 4 — α-parametric Table 4.1 + α-sensitivity figure (Addition 1)`
- `Ch 4 — specialist-verifier justification subsection (Addition 3)`
- `Ch 5 — per-benchmark α reporting + hypothesis test + self-knowledge correlation`

NO `Co-Authored-By` trailer (per existing memory convention).

---

## Appendix B — File-level change inventory (to be filled in during port)

Placeholder to be updated as each branch is ported:

```
Goal 1 files (Qwen-3B port):
  caem/pipeline.py                    (model loading, generation)
  caem/config.py                      (base_model_name, max_new_tokens)
  caem/confidence/pre_routing.py      (C_conv decoder-only)
  caem/verification/verifier.py       (remove decoder_input_ids, prefill)
  caem/retrieval/rag.py               (ChatML prompts)
  caem/training/self_improvement.py   (LoRA training, AutoModelForCausalLM)
  caem/prompts.py                     (NEW: ChatML builders per benchmark)
  scripts/run_experiment.py
  scripts/seed_cold_start.py
  scripts/run_baseline.py
  eval/baselines.py                   (ChatML wrap for all baselines)
  tests/test_pipeline*.py
  tests/test_verifier.py
  tests/test_pre_routing.py

Goal 4 files (memory hygiene):
  caem/training/self_improvement.py   (_is_repetitive_loop + filter)
  caem/memory/store.py                (consolidation at cycle boundary)
  caem/memory/entry.py                (hit-counter field)
  caem/retroverify.py                 (downgrade + hit-counter forced requeue)

Goal 2 files (q_a_relevance):
  caem/verification/verifier.py       (BGE cross-encoder signal)
  caem/config.py                      (add u_stored_weight_q_a_relevance)
  caem/ablation/variants.py           (no_q_a_relevance variant)
  caem/ablation/runner.py             (variant dispatch)

Thesis files:
  pre thesis 1 report/chapters/chapter_1.tex  (three-axis novelty)
  pre thesis 1 report/chapters/chapter_2.tex  (7-topic expansion)
  pre thesis 1 report/chapters/chapter_3.tex  (Qwen-3B, insertion-time scalar)
  pre thesis 1 report/chapters/chapter_4.tex  (α-parametric, A3 subsection)
  pre thesis 1 report/chapters/chapter_5.tex  (A2 per-benchmark, M1-M4)
  pre thesis 1 report/chapters/chapter_6.tex  (three-axis restate, heterogeneity)
  pre thesis 1 report/bibliography/           (add ~25 new citations)
```

---

## Appendix C — Confirmations still needed

1. **Benchmark list for A2 per-benchmark α** — 6 registry benchmarks + MMLU footnote,
   or include HotpotQA (thesis-scope expansion)?
2. **A3 subsection placement** — Ch4 §Verifier composite recommended; confirm
3. **Thesis files to grep for α-specific derivations** — Ch4 only, or also `caem-unified-plan-v3.tex` etc?
4. **Supervisor briefing on base-model swap** — drafted or assumed handled?
5. **A-MEM and Adaptive-RAG baseline implementations** — port from their public repos (~1-2 days each) or skip and cite as future work?

---

*This document is the single source of truth for branch C planning. Update it as
decisions refine. Do not begin integration on `feat/phase-2-all` until Appendix C
is fully resolved.*
