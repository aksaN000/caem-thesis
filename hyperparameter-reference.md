# CAEM Hyperparameter Reference

> **For skills:** `methodology-implementation` should read this before writing any implementation code.
> `paper-writing` should read this before writing Chapter 4 (Methodology) or Chapter 5 (Results).
>
> Created from Session 4 discussion — the distinction between categories is important for both implementation correctness and thesis writing accuracy.

---

## The Three Categories

Every hyperparameter in CAEM belongs to exactly one of three categories. Knowing which category a value falls under determines how to write about it and how to implement it.

---

### Category 1 — Fixed from literature
Copied from prior work with a citation. Do not change these during implementation unless a strong experimental reason emerges. Always cite the source when presenting them.

| Hyperparameter | Value | Source |
|---|---|---|
| MC Dropout forward passes | K = 5 | Gal & Ghahramani 2016 |
| Dropout rate at inference | p = 0.1 | Gal & Ghahramani 2016 |
| SC sample count | N = 10 | Wang et al. 2022 (diminishing returns past 10) |
| SE sample count | K = 10, T = 1.0 | Farquhar et al. 2024 |
| SE clustering method | Agglomerative + cosine | Farquhar et al. 2024 |
| NLI model | RoBERTa-Large-MNLI | Best open NLI at this scale |
| Embedding model | all-mpnet-base-v2, **768-dim** | Sentence-BERT (mpnet is 768-d; 384 was a pre-fix error — see impl-log EXP-10) |

**Writing instruction:** State these as fixed design decisions, each followed by the citation. No calibration language.

---

### Category 2 — Design choices
Principled starting values derived from design judgment. May be adjusted if ablation experiments show a clear problem, but changing them requires explicit justification in the thesis.

| Hyperparameter | Value | Basis |
|---|---|---|
| u_pre weights | 0.60 · u_token + 0.40 · (1/(1+C_conv)) | Token prob is the primary reliability signal; C_conv is secondary |
| û_stored weights | 0.50 · P(ENTAIL) + 0.30 · s_avg + 0.20 · (1−H_SE) | NLI entailment is the strongest post-hoc verification signal |
| Routing score λ | 0.70 · s + 0.30 · û_stored | Similarity dominates; confidence corrects at the margin |
| Tier 1 routing threshold | routing_score > 0.90 | High-confidence recall gate |
| Tier 2 similarity threshold | s > 0.75 | Medium-similarity knowledge adaptation zone |
| OR-condition threshold | u_pre < 0.60 → force Tier 3 | Safety-first: Tier 3 cost preferred over confident wrong answer |
| Stage 4a accept threshold | û ≥ 0.60 → accept; else escalate | Asymmetric cost → set low to avoid false negatives |
| Memory capacity | 1,000 (initial plan) / 20,000 (full) | GPU/disk budget |
| L2 regularization λ | **0.01** | Design choice [DES] — NOT from Kirkpatrick et al. 2017. Full EWC uses FIM weighting (Session 29 impl-log). λ=0.4 from early planning notes is WRONG — do not use it. |

**Writing instruction:** State these as "design decisions informed by [principle]". When presenting the OR-condition, explain the design reasoning: we explicitly choose the computational cost of Tier 3 over the risk of storing an incorrect confident answer.

---

### Category 3 — Empirically calibrated
These have stated initial values only. The actual values come from running calibration after Cycle 1 implementation. **They are not yet measured.** Chapter 5 will report the actual calibrated values.

| Hyperparameter | Initial value | Calibration method | When calibrated |
|---|---|---|---|
| Temperature scalar T | Not pre-set; fit from data | Minimise ECE on 500-sample calibration set using L-BFGS | **After Cycle 0, before Cycle 1** (calibrates base model output distribution) |
| û signal weights | 0.25 / 0.25 / 0.25 / 0.25 (equal) | Logistic regression or grid search on calibration set to maximise AUROC | **After Cycle 0, before Cycle 1** |

#### The "post-calibration weights" in the plan (0.20 / 0.20 / 0.20 / 0.40)

These values appear throughout the thesis plan as if calibration has already happened. They have not — we are still in the planning stage. They are **projected post-calibration estimates**: what the literature strongly suggests the calibration will converge toward, based on:

- Farquhar et al. (2024): SE achieves AUROC ≈ 0.79 for confabulation detection — substantially higher than token probability alone
- The logical consequence: if SE is more discriminative, calibration should assign it more weight
- Therefore SE is projected to receive ≈ double weight (0.40) and the other three ≈ 0.20 each

**This is a planning assumption, not a measured value.** The actual calibrated weights will come from fitting on real data and will be reported in Chapter 5. If SE does not dominate as expected, the weights will differ — and that discrepancy is itself an interesting finding to discuss.

#### Implementation instruction
```python
# Initialise with equal weights — DO NOT start at the projected values
signal_weights = {
    'u_token':    0.25,
    'u_dropout':  0.25,
    'u_sc':       0.25,
    'u_entropy':  0.25,
}

# After Cycle 1: run calibration routine on 500-sample calibration set
# Store calibrated weights in config for Cycles 2 and 3
# Report actual calibrated weights in Chapter 5
```

#### Writing instruction for Chapter 4
> "Initial signal weights are set equal at 0.25 for all four signals. After Cycle 0, temperature scaling calibration is performed on the held-out calibration set (500 samples), fitting the temperature scalar T and û signal weights via logistic regression on AUROC. Based on Farquhar et al.'s (2024) finding that semantic entropy achieves AUROC ≈ 0.79 for confabulation detection, we project the calibrated weights will converge toward approximately 0.20/0.20/0.20/0.40, with semantic entropy receiving elevated weight. Calibration is performed once only — it uses the base model's output distribution and applies for all subsequent cycles. Actual calibrated values are reported in Chapter 5."

---

## Summary Table

| Category | Set when? | Change how? | Write as... |
|---|---|---|---|
| 1 — Literature | Before implementation | Only with new citation and experimental reason | "Fixed at X (Author, Year)" |
| 2 — Design choice | Before implementation | With ablation evidence | "Set to X based on [design principle]" |
| 3 — Calibrated | After Cycle 0 | N/A — measured from data | "Initial value X; projected post-calibration ≈ Y; actual measured value Z (Chapter 5)" |
