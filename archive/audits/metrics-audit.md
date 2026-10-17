# CAEM metrics audit (Phase 4l)

Audit date: 2026-04-17. Scope: `eval/metrics.py`, `eval/harness.py`,
`scripts/run_experiment.py` (CSV writer), `scripts/run_calibration.py`
(ECE + temperature scaling), `scripts/run_purity_validation.py`
(Theory 1 protocol), `caem/training/self_improvement.py` (MMLU retention).

The goal is the mirror of the verifier discussion: inventory every metric,
grade it for (a) *appropriateness* to what the paper claims, (b)
*comparability* to published literature, and (c) *validity* when the metric
is novel. Concrete fix recommendations follow at the end.

---

## 1. Metric inventory

### 1a. Standard benchmark accuracy metrics (comparable to literature)

| # | Metric | Where | Benchmark | Literature anchor | Appropriateness |
|---|---|---|---|---|---|
| 1 | `normalise` + `exact_match` | `metrics.py:104` | SQuAD-convention fallback | SQuAD v1.1 normalisation (Rajpurkar et al. 2016) | OK — matches standard article/punct stripping and lowercasing |
| 2 | `token_f1` | `metrics.py:139` | SQuAD-convention fallback | SQuAD v1.1 F1 (Rajpurkar 2016) | OK — exact SQuAD official algorithm |
| 3 | `any_match_em` + `best_token_f1` | `metrics.py:119,210` | TriviaQA, Natural Questions | TriviaQA (Joshi 2017); NQ (Kwiatkowski 2019) | OK — takes max over aliases, which is the standard TQA protocol |
| 4 | `rouge_l` | `metrics.py:175` | TruthfulQA (EM proxy) | Lin & Hovy 2003 ROUGE; used by e.g. ELI5 (Fan 2019) | **Conditional.** See §2a below — TruthfulQA's canonical metric is GPT-judge, not ROUGE |
| 5 | `fever_accuracy` + `extract_fever_label` | `metrics.py:229,246` | FEVER | FEVER (Thorne 2018) 3-class label accuracy | OK — label matches standard protocol |
| 6 | `extract_strategyqa_label` + EM | `metrics.py:267` | StrategyQA | Geva et al. 2021 | OK — binary EM is the StrategyQA official metric |
| 7 | `extract_arc_label` + EM | `metrics.py:283` | ARC-Challenge | Clark et al. 2018 | OK (caveat below) — extracts A-D letter |
| 8 | `_mmlu_score` (generation) | `self_improvement.py:706` | MMLU | Hendrycks 2020 | **Conditional.** See §2b — literature uses log-likelihood scoring; we use generation. Fine for *retention ratios*, not for absolute leaderboard comparison |

### 1b. CAEM-specific (novel) metrics

| # | Metric | Where | Semantic | Literature anchor | Validity |
|---|---|---|---|---|---|
| 9 | `hallucination_rate` | `metrics.py:302` | fraction of samples with EM=0 AND û_stored ≥ 0.50 | closest: "confident error rate" (Guo 2017 for calibration); SelfCheckGPT (Manakul 2023); FActScore (Min 2023) | **Conditional.** The *name* oversells what it measures. It is a *confident-error rate* under a fixed threshold, not ground-truth hallucination. See §2c |
| 10 | `routing_distribution` (tier1/2/3 fractions) | `metrics.py:346` | fraction of queries routed to each tier | novel to CAEM (mixture-of-experts routing has a loose analogue in Switch Transformer; Mechanistic-IR reports retrieval coverage) | OK — well-defined, interpretable, directly tied to the architecture story |
| 11 | `storage_rate` | `metrics.py:413` | fraction of queries whose STORE decision fires | novel to CAEM | **Conditional.** Ambiguous at eval time — see §2d |
| 12 | `mean_u_stored` | `metrics.py:414` | mean composite confidence of stored answers | no direct anchor; closest: mean model confidence in Guo 2017 ECE analyses | OK — appropriate as a "memory quality proxy" when reported alongside hallucination_rate |
| 13 | `mean_latency_ms` | `metrics.py:415` | per-sample wall-clock time | infrastructure metric; standard in RAG papers (e.g. Lewis 2020 Appendix C) | OK |
| 14 | `mmlu_retention_ratio_pct` | `run_experiment.py:505` | cycle-N MMLU accuracy / Cycle-0 MMLU accuracy × 100 | continual-learning: average accuracy (ACC) and backward-transfer (BWT) from Lopez-Paz & Ranzato 2017 (GEM), Kirkpatrick 2017 (EWC), Biesialska 2020 survey | OK — maps cleanly onto BWT when computed across cycles, and is the metric ours claim in Chapter 5 Table 5.2 |

### 1c. Training-guard metric

| # | Metric | Where | Semantic | Literature anchor | Validity |
|---|---|---|---|---|---|
| 15 | `_mmlu_score(n=200)` as abort-guard probe | `self_improvement.py:888` (probe) + `run_cycle` pre/post invocation | post/pre MMLU 4-choice retention ratio $\rho = A_{\text{post}} / A_{\text{pre}}$; drives cycle-level abort + weight rollback when $\rho < \tau_{\text{forget}} = 0.93$. Same MMLU pass feeds the RET axis of the CES (row 14), so one per-cycle evaluation serves both the rollback trigger and the reported retention metric. | Kirkpatrick et al. 2017 (EWC) for the cycle-level rollback; Lopez-Paz & Ranzato 2017 (GEM, BWT) for the retention-ratio framing | OK — OOD probe is the principled choice for catastrophic-forgetting detection, because the failure mode (erosion of unrelated capability) is systematically invisible to an in-distribution probe. |
| 15b | `_forgetting_score` (TriviaQA, 50 pairs) — **DEPRECATED** | `self_improvement.py:837` | in-domain generation exact-match on held-out TQA pairs | GEM forgetting measure (Lopez-Paz 2017) is the conceptual cousin | **Deprecated as of the MMLU-probe rewrite.** No longer invoked by `run_cycle`; kept in the module for backward-compatibility / diagnostic scripts only. Rationale for deprecation: an in-distribution probe is systematically insensitive to catastrophic forgetting, because the model has just been fine-tuned on TriviaQA-like pairs. Replaced by row 15. The legacy `CycleResult.forgetting_score` field was also renamed to `CycleResult.mmlu_retention_ratio` to remove the naming hangover; all downstream scripts (`run_experiment.py`, `run_cyclic_ablation.py`) and tests have been updated. |

### 1d. Theorem validation metrics (Purity)

| # | Metric | Where | Semantic | Validity |
|---|---|---|---|---|
| 16 | `purity_theorem(p, α)` | `run_purity_validation.py:119` | P_theory = αp / (αp + (1−α)(1−p)) | Correct Bayesian identity assuming conditional independence of correctness and verification errors. Matches our §4.9 derivation |
| 17 | `check_purity_condition(p, α)` | `run_purity_validation.py:143` | asserts α > 0.5 | OK; corresponds to the sufficient condition P > p |
| 18 | `measure_base_accuracy` | `run_purity_validation.py:221` | p — greedy generation EM on purity split | OK |
| 19 | `measure_verification_balanced_accuracy` | `run_purity_validation.py:266` | α — balanced accuracy over {verification-correct-on-correct, verification-rejects-wrong} | OK — balanced accuracy is the right choice given class imbalance (most answers are correct; naive accuracy would be inflated) |
| 20 | `measure_memory_purity` | `run_purity_validation.py:398` | P_obs — empirical fraction of correct episodes among those actually stored | OK — directly measures what the theorem predicts |

### 1e. Calibration metrics

| # | Metric | Where | Semantic | Literature anchor | Validity |
|---|---|---|---|---|---|
| 21 | `expected_calibration_error` (ECE, 10 equal-width bins) | `run_calibration.py:67` | ∑_b (|B_b|/n)·|acc(B_b) − conf(B_b)| | Guo 2017 "On Calibration of Modern Neural Networks" | OK — 10-bin equal-width is the canonical choice |
| 22 | Temperature scaling (scalar T on logits) | `run_calibration.py:119` | Platt/Guo temperature scaling | Guo 2017 | OK — minimises NLL, reported as ECE-before / ECE-after |

### 1f. Statistical testing utilities

| # | Metric | Where | Semantic | Literature anchor | Validity |
|---|---|---|---|---|---|
| 23 | `bootstrap_ci` (n=1000, 95% default) | `metrics.py:424` | non-parametric bootstrap CI for a mean | Efron & Tibshirani 1993 | OK — standard; 1000 resamples is the minimum defensible |
| 24 | `mcnemar_test` with Edwards continuity | `metrics.py:473` | paired binary significance (χ² df=1) | Edwards 1948; Dror et al. 2018 "Deep Dominance" (NLP-specific recommendation) | OK — exactly the right test for paired EM scores |

---

## 2. Issues flagged during audit

### 2a. TruthfulQA is scored with ROUGE-L, not GPT-judge

TruthfulQA's official metric is *a fine-tuned judge model* (GPT-judge, a
GPT-3 variant fine-tuned on human truthful/informative labels — Lin et al.
2021, "TruthfulQA: Measuring How Models Mimic Human Falsehoods"). Later
work (e.g. Lin 2022; Askell 2021) sometimes uses BLEURT or exact-match
against a reference answer list.

Our `rouge_l(prediction, golds)` with an EM threshold of 0.15 is a
*pragmatic offline proxy*. It is defensible because:

1. It is deterministic and reproducible (GPT-judge is not publicly
   reproducible without API access).
2. It gives *partial credit* for free-form answers (unlike strict EM).
3. It does not fabricate a judge that we could not afford to run at our
   experiment scale.

The risk is that our absolute TruthfulQA numbers are **not directly
comparable** to GPT-judge scores in the TruthfulQA leaderboard. This is
publication-critical: Chapter 5 must be explicit that TruthfulQA here is
reported under a ROUGE-L proxy, and any claim like "CAEM matches published
SOTA" would be invalid against the leaderboard. Cycle-over-cycle deltas
*are* valid under the proxy — that is what Chapter 5 reports.

### 2b. MMLU is scored by generation, not log-likelihood

Already documented in the `_mmlu_score` docstring after the Phase 4j
polish. The leaderboard protocol (Hendrycks 2020; lm-eval-harness, OpenAI
eval suite) is *log-likelihood scoring*: for each choice k, compute
P(choice_k | prompt) and take argmax. This avoids any parsing ambiguity
and is strictly more sensitive (log-likelihood uses the whole choice
string, not just the leading letter).

We score by generation because CAEM's runtime is generative and we want
the retention metric to reflect the same code path. This choice is
defensible for *retention ratios* (our Chapter 5 claim) but makes absolute
MMLU numbers ~3–8 points lower than leaderboard numbers. Must be stated
in Chapter 5 §5.3 methodology.

### 2c. `hallucination_rate` is a confident-error rate, not ground-truth hallucination

Operational definition: `EM = 0 AND û_stored ≥ 0.50`. Pitfalls:

- **Threshold choice.** 0.50 is not grounded in any calibration analysis.
  If our store threshold (from `config.py`) is, say, 0.55, then we are
  measuring something slightly inside the rejected region too.
- **Name.** "Hallucination" has a specific meaning in the 2024+ LLM
  literature (Farquhar 2024: *confabulation*; Ji 2023 survey: closed-book
  fabrication; Min 2023 FActScore: atomic-fact grounding). Our metric
  is *one specific operationalisation* — it is not ground-truth.
- **Overlap with EM.** If hallucination_rate and EM are reported side by
  side, naive readers will read hallucination_rate as (1 − EM). It is
  actually (1 − EM) × I[û_stored ≥ 0.5], i.e. a *subset* of errors. This
  should be made explicit.

Recommendation: in the *paper text*, call this metric **confident-error
rate** with a footnote "used as a hallucination proxy in the CAEM paper
and colloquially referred to as `hallucination_rate` in the codebase".
Leave the CSV column name alone to preserve downstream compatibility.

### 2d. `storage_rate` is ambiguous in eval mode

`EvalHarness.run(..., store_to_memory=False)` is used for the dev/transfer
evaluation; the SIL training pool is passed `store_to_memory=True`. The
`stored` flag returned by `pipeline.answer()` tracks **whether the verifier
decided STORE**, not whether a row was actually appended to the FAISS
index. So in eval mode, `storage_rate` is still informative (it is the
rate at which the verifier *would have stored* this answer). This is
what we want — but the column name `storage_rate` suggests persistence.

Recommendation: rename to `would_store_rate` internally, or add a comment
in `save_summary_csv` clarifying the semantics. For the paper,
"verifier STORE rate" is the precise phrasing.

### 2e. ARC-Challenge regex can misfire on numbered rationales

`re.search(r"\b([A-Da-d1-4])\b", cot_final)` matches the first isolated
letter or digit in 1–4 range. For CoT outputs like:

> "1. Photosynthesis is a process. 2. The correct answer is B."

…the regex picks `1` before it ever reaches `B`. The gold label is the
letter, so this silently counts as wrong.

Recommendation: tighten the regex to prefer letters first, then fall
back to digits. Minor fix (about 2 lines). Current code already uses
`extract_cot_answer` which splits on "Answer:", so this misfires only
when the model forgets the "Answer:" prefix — still possible under CoT.

### 2f. Bootstrap CI and McNemar are defined but never called

`bootstrap_ci` and `mcnemar_test` exist in `eval/metrics.py` but neither
`run_experiment.py` nor the CSV writer ever invokes them. For top-venue
submissions, *every* primary table should carry 95% CIs, and cycle-over-
cycle improvements should be McNemar-significant.

Recommendation: extend `save_summary_csv` to emit `em_ci_lo`,
`em_ci_hi`, `f1_ci_lo`, `f1_ci_hi` columns, and `mcnemar_p_vs_cycle0`.
This is a ~30-line addition to the aggregation pass. Out of scope for
Phase 4l but I have queued it under a new Phase 4m.

### 2g. ECE is only computed during calibration, never at evaluation

The user asks "is û_stored calibrated?". We compute ECE *once* during
calibration (500-sample calib set, pre-training) and never again. Top
venues will want per-cycle ECE on the *evaluation* set to show the
verifier stays calibrated as the model drifts.

Recommendation: run `expected_calibration_error` over each cycle's
`(u_stored, em)` pairs and add an `ece` column to the experiment CSV.
~10 lines in the aggregate step.

### 2h. No per-tier accuracy breakdown

A central claim is "memory beats generation". To demonstrate this we
need `em_tier1`, `em_tier2`, `em_tier3` — i.e., accuracy conditional on
the router's decision. Currently we only log the aggregate `em` and the
tier distribution, so the reader cannot see whether memory hits are
actually high-quality.

Recommendation: expose `em_by_tier` in `aggregate()`. Trivial change
(~8 lines).

### 2i. No confidence–correctness AUROC

Best-in-class calibration papers (Malinin 2018; Jiang 2021; Kadavath
2022 "Language Models Know What They Know") report AUROC of confidence
vs. correctness. It is the gold signal for "û_stored has information
about correctness". We should compute and report this per cycle.

Recommendation: add `auroc_u_stored_vs_em` using `sklearn.metrics.
roc_auc_score` (sklearn is already a transitive dep via HuggingFace).

---

## 3. Metrics flow diagram

Below is the end-to-end flow — analogous to the verifier discussion in
Chapter 4. "→" means "feeds into", bracketed numbers are the inventory
row from §1.

```
                ┌──────────────────────────────────────────┐
                │       CAEMPipeline.answer(query)         │
                │  produces: answer, tier ∈ {1,2,3},       │
                │  stored_confidence.u_stored, stored,     │
                │  latency_ms, escalated                   │
                └──────────────┬───────────────────────────┘
                               │
                               ▼
                ┌──────────────────────────────────────────┐
                │   EvalHarness._run_one(sample)           │
                │   extract_cot_answer → per-benchmark     │
                │   scoring branch                         │
                └──────────────┬───────────────────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
     [fever: 5]         [triviaqa/NQ:       [truthfulqa: 4]
     label extract +     3]  any_match_em    rouge_l + EM@0.15
     accuracy            + best_token_f1
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                 per-sample   EM, F1, tier, stored,
                 results ───▶ u_stored, latency
                               │
                               ▼
               ┌────────────────────────────────────────────┐
               │   eval/metrics.py::aggregate(...)          │
               │   produces:                                │
               │     em  ──────────[comparable to litR]     │
               │     f1  ──────────[comparable]             │
               │     hallucination_rate [9: novel, flagged] │
               │     routing_distribution [10: novel]       │
               │     storage_rate [11: novel, eval-ambig.]  │
               │     mean_u_stored [12: novel]              │
               │     mean_latency_ms [13: infra]            │
               └──────────────┬─────────────────────────────┘
                              │
                              ▼
               ┌────────────────────────────────────────────┐
               │  run_experiment.py:save_summary_csv(...)   │
               │  + mmlu_retention_pct [8]                  │
               │  + mmlu_retention_ratio_pct [14]           │
               │  → experiment_summary.csv                  │
               └──────────────┬─────────────────────────────┘
                              │
                ┌─────────────┴───────────────┐
                ▼                             ▼
      [Chapter 5 Table 5.1]          [Chapter 5 Table 5.2]
      per-benchmark results           retention / forgetting
      (em, f1, hallucination,         (absolute MMLU + ratio)
      tier fractions, ...)
```

A parallel flow runs **once per experiment** for the theorem validation:

```
    SIL training-pool 500 samples (purity_samples)
                  │
                  ▼
    measure_base_accuracy(p)            [18]
    measure_verification_balanced_acc(α)[19]
    measure_memory_purity(P_obs)        [20]
                  │
                  ▼
    purity_theorem(p, α)  →  P_theory   [16]
    check_purity_condition(α)           [17]
                  │
                  ▼
    [Chapter 5 §5.5 Theory 1 validation]
```

And calibration runs **once before Cycle 1**:

```
    calibration samples (500)
                  │
                  ▼
    collect (u_logits, em) pairs
                  │
                  ▼
    expected_calibration_error(pre)     [21]   → ece_before
    fit T via NLL (Platt/temp. scaling) [22]
    expected_calibration_error(post)    [21]   → ece_after
                  │
                  ▼
    calibration.json → loaded by pipeline at Cycle ≥ 1
```

---

## 4. Recommended changes (ordered by priority)

**P0 — must fix before thesis submission (scientific validity)**

1. *Document* the TruthfulQA ROUGE-L proxy choice in Chapter 5 §5.3
   methodology, explicitly stating non-comparability to the GPT-judge
   leaderboard.
2. *Document* the MMLU generation-vs-log-likelihood scoring choice
   (docstring now in place; §5.3 text still pending).
3. *Rename* `hallucination_rate` → "confident-error rate" in the paper
   text; keep the CSV column name for backward compatibility.

**P1 — strongly recommended for top-venue submission**

4. Call `bootstrap_ci` in `save_summary_csv` and emit `em_ci_lo`,
   `em_ci_hi`, `f1_ci_lo`, `f1_ci_hi` per (cycle, benchmark).
5. Call `mcnemar_test` between each cycle and Cycle 0; emit
   `mcnemar_p_vs_cycle0` column.
6. Compute per-cycle ECE on `(u_stored, em)` pairs; emit `ece_u_stored`
   column.
7. Compute per-tier EM breakdown; emit `em_tier1`, `em_tier2`,
   `em_tier3` columns.
8. Compute `auroc_u_stored_vs_em` per cycle (single scalar).

**P2 — polish (reviewer-readiness)**

9. Tighten `extract_arc_label` regex to prefer letters over digits.
10. Rename the conceptual metric `storage_rate` → "verifier STORE rate"
    in Chapter 5 to remove the persistence-implication ambiguity.

---

## 5. Verdict

The core accuracy metrics are correct and mostly standard. Two of the
headline claims (TruthfulQA and MMLU) use defensible offline proxies
that are **not** directly comparable to the respective leaderboards;
this is fine provided Chapter 5 states it plainly.

The novel CAEM metrics (`routing_distribution`, `storage_rate`,
`mean_u_stored`, `mmlu_retention_ratio_pct`) are well-defined and tied
to specific architectural claims. `hallucination_rate` is the one that
needs the most careful framing because the name is stronger than what
the metric measures.

The biggest gap is that statistical rigor (CIs, McNemar, per-cycle ECE,
per-tier EM, AUROC) is implemented in utilities but never wired into
the experiment summary. For a top venue, this must be fixed — and the
fix is small (~80 lines across `aggregate()` and `save_summary_csv`).
These are the Phase 4m additions.
