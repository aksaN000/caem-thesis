\# Lifting verifier precision from 70% to 80% without gold labels

The literature offers exactly one method family that explicitly demonstrates ≥80% factual-QA precision without filter-time gold labels: \*\*conformal factuality\*\* (Mohri & Hashimoto 2024 and successors). For the 9-signal CAEM composite specifically, the closest architectural matches are \*\*weak-supervision label models\*\* (FlyingSquid / Snorkel / Dawid-Skene) which provide a principled label-free composite. Single-signal hallucination detectors (EigenScore, KLE, HaloScope) report AUROC 0.78–0.85 on TriviaQA/NQ but \*\*no published precision@threshold ≥80%\*\*. Below are 15 candidates ranked by relevance.

\---

\## 1. Language Models with Conformal Factuality Guarantees ★ KILLER REFERENCE

\*\*Citation\*\*: Mohri & Hashimoto, "Language Models with Conformal Factuality Guarantees," ICML 2024. https://arxiv.org/abs/2402.10978

\*\*Mechanism\*\*: Decomposes each answer into atomic sub-claims, scores each (sample-frequency / self-eval / NLI), and removes claims below a calibrated threshold τ. Split-conformal sets τ on a small annotated set so retained claims are jointly correct with probability ≥1−α. \[arXiv\](https://arxiv.org/html/2406.09714v1) Output is a more conservative answer with formal precision guarantee.

\*\*Gold supervision\*\*: \*\*Calibration set only\*\* (a few hundred to ~1.5k human-annotated sub-claims, offline). NONE at filter time.

\*\*Empirical\*\*: GPT-4 on \*\*NaturalQuestions: factuality 78%→93%\*\* at α=0.2, removing ~25% of sub-claims; FActScore biographies 25%→80%; MATH 75%→95%. \[Liner\](https://liner.com/review/language-models-with-conformal-factuality-guarantees) \*\*Explicitly crosses 80% precision on NQ.\*\*

\*\*Compute\*\*: K=5–20 samples per query + small NLI/entailment scorer. Fits a 3B generator + DeBERTa-MNLI on single GPU.

\*\*Fit\*\*: Pre-storage \*\*abstention/back-off filter\*\* OR composite-level calibrator using the existing 9 signals as the per-claim score.

\*\*Risk\*\*: Marginal coverage assumes exchangeability; with 7 heterogeneous benchmarks it likely \*\*under-covers on minority benchmarks\*\* unless calibrated per-benchmark or via conditional CP. Cherian et al. show vanilla Mohri–Hashimoto under-covers on rare topics. \[NeurIPS\](https://proceedings.neurips.cc/paper\_files/paper/2024/file/d02ff1aeaa5c268dc34790dd1ad21526-Paper-Conference.pdf)

\## 2. Enhanced Conformal Prediction for LLM Validity

\*\*Citation\*\*: Cherian, Gibbs, Candès, "Large Language Model Validity via Enhanced Conformal Prediction Methods," NeurIPS 2024. https://arxiv.org/abs/2406.09714

\*\*Mechanism\*\*: Two upgrades to Mohri–Hashimoto: (a) \*\*conditional boosting\*\* learns a linear combination of multiple per-claim scores (frequency, self-eval, ordinal, single-token P) \[NeurIPS\](https://proceedings.neurips.cc/paper\_files/paper/2024/file/d02ff1aeaa5c268dc34790dd1ad21526-Paper-Conference.pdf) — directly applicable to CAEM's 9 signals; (b) \*\*level-adaptive conformal prediction\*\* adapts α per-prompt \[arXiv\](https://arxiv.org/html/2406.09714v1) to maintain calibrated prompt-level factuality probability rather than only marginal coverage. \[arXiv\](https://arxiv.org/html/2406.09714v1)

\*\*Gold supervision\*\*: ~1.4k calibration prompts with sub-claim annotations (offline). NONE at filter time.

\*\*Empirical\*\*: Mean claim retention \*\*39% (boosted) vs. 24% (Mohri unboosted)\*\* at the same factuality level on MedicationQA \[arXiv\](https://arxiv.org/html/2406.09714v1) / Wikipedia bios. \[arXiv\](https://arxiv.org/html/2406.09714v1) Better handles subgroup heterogeneity.

\*\*Compute\*\*: Same as Mohri–Hashimoto + small boosting regression. Single-GPU feasible.

\*\*Fit\*\*: \*\*Best principled composite-level calibrator\*\* for CAEM — directly converts the 9 signals into a calibrated per-claim score with conformal guarantees, while accommodating cross-benchmark heterogeneity via the conditional CP machinery.

\*\*Risk\*\*: Still needs a labeled calibration fold; per-prompt conditioning machinery is more complex to implement than vanilla split CP.

\## 3. SAFE — Search-Augmented Factuality Evaluator

\*\*Citation\*\*: Wei et al. (Google DeepMind), "Long-form factuality in large language models," NeurIPS 2024. https://arxiv.org/abs/2403.18802

\*\*Mechanism\*\*: Decomposes a response into atomic facts; for each, an LLM agent issues multi-step Google Search queries and reasons over results to label SUPPORTED / NOT-SUPPORTED / IRRELEVANT. \[arXiv\](https://arxiv.org/abs/2403.18802) Aggregates into F1@K (precision = % supported atomic facts).

\*\*Gold supervision\*\*: NONE at train OR filter time.

\*\*Empirical\*\*: \*\*Agrees with crowdworkers 72.0%\*\*; on disagreements, judged correct \*\*76% vs. humans 19%\*\* (super-human). \[arXiv\](https://arxiv.org/abs/2403.18802) Top LLMs (GPT-4-Turbo, Gemini-Ultra) score 87–95% factual precision under SAFE on LongFact-Objects.

\*\*Compute\*\*: 3–8 LLM calls + 3–5 search calls per claim. Originally GPT-3.5-class; runs on Llama-3.2-3B / Phi-3-mini (≤3B).

\*\*Fit\*\*: Pre-storage filter OR signal replacement for MiniCheck. \*\*Substituting CAEM's existing RAG corpus for Google Search satisfies the no-external-API constraint\*\* at the cost of some coverage.

\*\*Risk\*\*: External search API in default config; cost ~$5–10 per 1k claims via Serper. Replacement of search with RAG-only dilutes signal.

\## 4. FlyingSquid — Triplet-Method Latent Label Model

\*\*Citation\*\*: Fu, Chen, Sala, Hooper, Fatahalian, Ré, "Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods," ICML 2020. https://arxiv.org/abs/2002.11955

\*\*Mechanism\*\*: Models the 9 verifier signals (binarized) and the unobserved correctness label as a latent-variable PGM. Any three conditionally-independent labeling functions yield a closed-form (method-of-moments) solution for verifier accuracies and posterior P(correct | s₁..s₉). \*\*No SGD, no gold.\*\* \[Hazy Research\](https://hazyresearch.stanford.edu/blog/2020-02-28-flyingsquid)

\*\*Gold supervision\*\*: NONE. ~1k–10k unlabeled samples to estimate covariance. Optional ~100 labels to resolve sign ambiguity (replaceable by class prior).

\*\*Empirical\*\*: No factual-QA numbers. Snorkel-family methods typically reach within 1–4 F1 of fully supervised on relation-extraction tasks; +5–15 F1 over majority vote.

\*\*Compute\*\*: Closed-form, sub-second on CPU. \[arXiv\](https://arxiv.org/abs/2002.11955) Zero extra inference cost.

\*\*Fit\*\*: ★★★★★ \*\*Direct latent-variable wrapper around the 9 signals — replaces u\_stored with a calibrated P(em-correct | s₁..s₉) without any gold.\*\* The closest published architectural match to CAEM.

\*\*Risk\*\*: Assumes ≥3 conditionally-independent signals — CAEM's 9 share retrieval and backbone, so dependency structure must be specified (Varma et al. 2019 robust-PCA) or the dependence-aware Ising variant used.

\## 5. Snorkel — Generative Label Model for Weak Supervision

\*\*Citation\*\*: Ratner, Bach, Ehrenberg, Fries, Wu, Ré, "Snorkel: Rapid Training Data Creation with Weak Supervision," VLDB 2017. https://arxiv.org/abs/1711.10160

\*\*Mechanism\*\*: Treats each labeling function (each verifier signal) as a noisy vote \[PubMed\](https://pubmed.ncbi.nlm.nih.gov/29770249/) whose accuracy and pairwise correlations are estimated by maximizing the marginal likelihood of agreement patterns. Outputs probabilistic soft label P(y=1 | s₁..s₉). Snorkel MeTaL extends to multi-task; modeling-advantage optimizer chooses between majority-vote and generative model automatically.

\*\*Gold supervision\*\*: NONE required; ~50–200 labeled examples recommended for sign disambiguation and threshold tuning.

\*\*Empirical\*\*: +132% mean F1 over heuristic baselines, within 3.6% of fully supervised \[arxiv\](https://arxiv.org/pdf/1711.10160) across 6 non-factual-QA tasks. No 70→80 factual-QA claim published.

\*\*Compute\*\*: Trivial — O(m²) parameters for m=9 LFs. CPU only.

\*\*Fit\*\*: Composite-level calibrator and replacement for u\_stored. Equivalent to FlyingSquid via different fitting procedure (EM vs. method-of-moments).

\*\*Risk\*\*: CI assumption violations (same as FlyingSquid). Recent critique (Balasubramanian et al. 2026, arXiv:2601.22336) shows LLM-judge ensembles routinely violate CI, producing "confidently incorrect" posteriors \[arxiv\](https://arxiv.org/pdf/2601.22336) — use dependency-aware variants.

\## 6. INSIDE / EigenScore — Internal-State Hallucination Detection

\*\*Citation\*\*: Chen, Quan, Wang, Liu, Zhao, Sun, "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection," ICLR 2024. https://arxiv.org/abs/2402.03744

\*\*Mechanism\*\*: Sample K=10–20 responses; extract middle-layer last-token hidden state for each; compute log-determinant of the K×K covariance matrix (≈ differential entropy in embedding space). \[Liner\](https://liner.com/review/inside-llms-internal-states-retain-the-power-of-hallucination-detection) Low spread = consistent = factual. Optional test-time feature clipping defeats overconfident hallucinations. \[arXiv\](https://arxiv.org/html/2402.03744v2)

\*\*Gold supervision\*\*: NONE. Threshold by G-mean optimization on a small unlabeled fold. \[Iclr\](https://proceedings.iclr.cc/paper\_files/paper/2024/file/0d1986a61e30e5fa408c81216a616e20-Paper-Conference.pdf) \[arXiv\](https://arxiv.org/pdf/2402.03744)

\*\*Empirical\*\*: \*\*AUROC on LLaMA-7B: TriviaQA 0.835, NQ 0.787, SQuAD 0.802\*\*; up to +8.9 AUROC over Perplexity / LexicalSim / SelfCheckGPT. \[Liner\](https://liner.com/review/inside-llms-internal-states-retain-the-power-of-hallucination-detection) \[arXiv\](https://arxiv.org/pdf/2402.03744) Beats SelfCheckGPT and ITI on TruthfulQA. \*\*No precision@threshold reported, but AUROC ~0.84 is the strongest single unsupervised signal.\*\*

\*\*Compute\*\*: K=15 generations + free hidden-state extraction + trivial 15×15 eigendecomposition. \[Liner\](https://liner.com/review/inside-llms-internal-states-retain-the-power-of-hallucination-detection) ~15× generation cost.

\*\*Fit\*\*: \*\*Strong signal replacement for CAEM's semantic-entropy slot.\*\* Requires white-box access to Qwen-3B middle layer.

\*\*Risk\*\*: Layer choice sensitivity; needs small unlabeled calibration sweep. \[Liner\](https://liner.com/review/inside-llms-internal-states-retain-the-power-of-hallucination-detection) Translation from 0.84 AUROC to ≥80% precision requires aggressive top-rank filtering (low recall).

\## 7. Kernel Language Entropy

\*\*Citation\*\*: Nikitin, Kossen, Gal, Janson, "Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities," NeurIPS 2024. https://arxiv.org/abs/2405.20003

\*\*Mechanism\*\*: Replaces Farquhar's hard NLI-cluster entropy with a \*\*soft kernel\*\* over pairwise semantic similarities; computes \*\*von Neumann entropy\*\* of the kernel matrix. \[OpenReview\](https://openreview.net/forum?id=j2wCrWmgMX¬eId=77cNQ2ogsR) Theoretically generalizes Farquhar's semantic entropy \[arXiv\](https://arxiv.org/abs/2405.20003) by capturing graded relatedness rather than discrete clusters.

\*\*Gold supervision\*\*: NONE.

\*\*Empirical\*\*: Outperforms Farquhar SE \[arXiv\](https://arxiv.org/abs/2405.20003) on TriviaQA, NQ, SQuAD across LLaMA-2 / Mistral / Falcon (AUROC gains 1–4 pts; \*\*TriviaQA AUROC ≈ 0.80–0.84\*\*).

\*\*Compute\*\*: Same as existing SE (N samples + N² NLI calls + tiny eigendecomposition).

\*\*Fit\*\*: \*\*Direct drop-in upgrade to CAEM's semantic-entropy signal\*\* with negligible code change. Lowest-risk gain on the list.

\*\*Risk\*\*: Marginal AUROC improvement; alone insufficient for 80% precision lift.

\## 8. Conformal Abstention via Self-Similarity

\*\*Citation\*\*: Yadkori et al. (Google DeepMind), "Mitigating LLM Hallucinations via Conformal Abstention," 2024. https://arxiv.org/abs/2405.01563

\*\*Mechanism\*\*: Uses LLM self-similarity (the LLM evaluates similarity among its own k samples) as a confidence signal, then applies \*\*conformal risk control\*\* (Angelopoulos et al.) to calibrate an abstention threshold λ such that hallucination rate ≤ α on test. CRC generalizes split-CP to bound any monotone loss including (1−precision).

\*\*Gold supervision\*\*: Calibration set only (a few hundred labeled responses). NONE at filter time.

\*\*Empirical\*\*: Reliably bounds hallucination rate at user-specified α on closed-book TriviaQA with Gemini Pro. \*\*Setting α=0.2 yields ≥80% precision by construction.\*\* Comparable abstention rates to log-prob baselines; less conservative on long-answer Temporal Sequences.

\*\*Compute\*\*: k=5–20 samples per query + LLM-judge or NLI for similarity. Single-GPU feasible.

\*\*Fit\*\*: \*\*Pre-storage abstention gate with formal hallucination-rate guarantee — directly answers CAEM's binary store/skip question.\*\* CRC is more flexible than vanilla CP since it targets precision directly.

\*\*Risk\*\*: Exchangeability across the 7 benchmarks; self-similarity overlaps with CAEM's existing semantic-entropy signal (avoid double-counting).

\## 9. ConU — Conformal Uncertainty for Open-Ended NLG

\*\*Citation\*\*: Wang et al., "ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees," EMNLP Findings 2024. https://arxiv.org/abs/2407.00499

\*\*Mechanism\*\*: Black-box conformal prediction. Nonconformity score = self-consistency frequency (fraction of k samples in the most-frequent semantic cluster). Standard split-CP calibrates a quantile threshold so prediction sets cover an acceptable answer with probability ≥1−α.

\*\*Gold supervision\*\*: Calibration set only (50/50 split of a labeled benchmark, ~hundreds of examples).

\*\*Empirical\*\*: \*\*TriviaQA AUROC ≈ 0.822\*\* (vs. best black-box baseline 0.817) \[Liner\](https://liner.com/review/conu-conformal-uncertainty-in-large-language-models-with-correctness-coverage) across 7 LLMs (LLaMA-2/3, Mistral). Empirical miscoverage matches user-specified α ≈ 0.1 on homogeneous data.

\*\*Compute\*\*: 20 samples per query at T=1.0. Black-box (no logits required).

\*\*Fit\*\*: Drop-in selective-prediction filter using existing self-consistency signal. Directly compatible with CAEM's multi-chain setup.

\*\*Risk\*\*: Successor SConU paper (arXiv:2504.14154) shows ConU's empirical miscoverage exceeds the user-specified level on heterogeneous datasets \[arXiv\](https://arxiv.org/html/2504.14154v1) — \*\*directly the cross-benchmark concern in CAEM\*\*. Mitigate with SConU's outlier filter.

\## 10. Chain-of-Verification (CoVe)

\*\*Citation\*\*: Dhuliawala, Komeili, Xu, Raileanu, Li, Celikyilmaz, Weston, "Chain-of-Verification Reduces Hallucination in Large Language Models," Findings of ACL 2024. \[ACL Anthology\](https://aclanthology.org/2024.findings-acl.212/) https://arxiv.org/abs/2309.11495

\*\*Mechanism\*\*: Four-stage prompted pipeline reusing the same generator: (1) draft answer → (2) plan independent verification questions → (3) answer each Q in isolation (no shared context) → (4) regenerate verified final answer. \[arXiv\](https://arxiv.org/abs/2309.11495) \[Semantic Scholar\](https://www.semanticscholar.org/paper/Chain-of-Verification-Reduces-Hallucination-in-Dhuliawala-Komeili/4b0b56be0ae9479d2bd5c2f0943db1906343c10f) The "Factor+Revise" variant cross-checks each fact against verification answers. \[ACL Anthology\](https://aclanthology.org/2024.findings-acl.212.pdf) \[ResearchGate\](https://www.researchgate.net/publication/370981225\_FActScore\_Fine-grained\_Atomic\_Evaluation\_of\_Factual\_Precision\_in\_Long\_Form\_Text\_Generation)

\*\*Gold supervision\*\*: NONE — pure prompting.

\*\*Empirical\*\*: Llama-65B Wikidata list-Q \*\*precision 17%→36%\*\* (≈2× lift, largest in this survey). MultiSpanQA F1 \*\*39.0→48.2\*\*. Long-form bios FActScore \*\*55.9→71.4\*\* (+28%). \*\*Does not cross 80% but produces the largest relative precision lift.\*\*

\*\*Compute\*\*: 3–4 extra forward passes on the existing generator. No external models/search.

\*\*Fit\*\*: Pre-storage \*\*generation-policy filter\*\* (run before the 9-signal verifier); the disagreement between draft and verification answers is itself a binary hallucination signal.

\*\*Risk\*\*: Small models (3B) sometimes copy the draft into verification responses, weakening the trick — use the factored variant. Latency cost.

\## 11. HaloScope — Unsupervised Hidden-State Subspace Filter

\*\*Citation\*\*: Du, Xiao, Li, "HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection," NeurIPS 2024 Spotlight. \[GitHub\](https://github.com/deeplearning-wisc/haloscope) https://arxiv.org/abs/2409.17504

\*\*Mechanism\*\*: From an unlabeled pool of LLM generations, \[ADS\](https://ui.adsabs.harvard.edu/abs/2024arXiv240917504D/abstract) performs SVD on hidden-state representations; the top singular subspace empirically aligns with hallucinated content. \[arXiv\](https://arxiv.org/pdf/2409.17504) Projection onto that subspace serves as a self-supervised score, then a linear probe is trained on hidden states \[ADS\](https://ui.adsabs.harvard.edu/abs/2024arXiv240917504D/abstract) using the unsupervised score as soft label. \[NIPS\](https://nips.cc/virtual/2024/poster/93676)

\*\*Gold supervision\*\*: NONE — fully self-supervised via subspace decomposition.

\*\*Empirical\*\*: \*\*TruthfulQA AUROC 78.6\*\* (LLaMA-2-7B) vs. SelfCheckGPT 65.1, Perplexity 70.9, EigenScore 71.4. \*\*TriviaQA AUROC 73.4.\*\* Strong but no published precision@threshold.

\*\*Compute\*\*: One-time SVD + linear probe on a few thousand unlabeled Qwen-3B generations. Inference: single hidden-state projection (negligible).

\*\*Fit\*\*: New signal complementary to EigenScore (subspace structure across queries vs. inter-sample covariance). Open-source code. \[ADS\](https://ui.adsabs.harvard.edu/abs/2024arXiv240917504D/abstract)

\*\*Risk\*\*: Score can be noisy if the unlabeled pool is biased; sensitive to random seed. \[GitHub\](https://github.com/deeplearning-wisc/haloscope)

\## 12. Multicalibration

\*\*Citation\*\*: Hébert-Johnson, Kim, Reingold, Rothblum, "Multicalibration: Calibration for the (Computationally-Identifiable) Masses," ICML 2018. https://arxiv.org/abs/1711.08513

\*\*Mechanism\*\*: Post-processes a base predictor (CAEM's u\_stored) so calibration holds simultaneously over all computationally-identifiable subgroups defined by \*\*features\*\* (signal patterns, question length, retrieval count) rather than benchmark identity. HKRR boosting iteratively patches calibration violations on each subgroup. \[Proceedings of Machine Learning Research\](https://proceedings.mlr.press/v80/hebert-johnson18a.html) \[arXiv\](https://arxiv.org/html/2406.06487) Recent LLM application: Detommaso et al. 2024.

\*\*Gold supervision\*\*: Labeled calibration fold required offline; NONE at filter time.

\*\*Empirical\*\*: Reduces worst-group ECE on Bio-NQ from ~10% to 2–4%; raw precision lift typically 2–5 pts over Platt/temperature on heterogeneous data. No 70→80 claim.

\*\*Compute\*\*: Sample-inefficient \[arXiv\](https://arxiv.org/html/2406.06487) — needs ~5–20k labeled calibration examples for 9 signals binned into deciles.

\*\*Fit\*\*: \*\*Strict generalization of CAEM's existing isotonic-on-pooled-fold baseline that handles benchmark heterogeneity without exposing benchmark ID at runtime.\*\* Provides monotone calibration lift across feature subgroups (formal guarantee).

\*\*Risk\*\*: Overfits with small folds (Gopalan et al. 2022 critique); \[arXiv\](https://arxiv.org/html/2406.06487) calibration ≠ precision lift directly — must still set the threshold.

\## 13. FActScore — Atomic Factuality with Retrieval

\*\*Citation\*\*: Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer, Hajishirzi, "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," EMNLP 2023. \[ACL Anthology\](https://aclanthology.org/2023.emnlp-main.741/) https://arxiv.org/abs/2305.14251

\*\*Mechanism\*\*: LLM decomposes response into atomic facts; \[arXiv +2\](https://arxiv.org/html/2305.14251) retrieves top-k passages from a knowledge source (Wikipedia or your RAG corpus); a small LM labels each atomic claim TRUE/FALSE against retrieved evidence. Output = % supported.

\*\*Gold supervision\*\*: NONE at filter time. Verifier prompts are zero/few-shot.

\*\*Empirical\*\*: \*\*<2% error vs. human annotators\*\* \[arXiv\](https://arxiv.org/abs/2305.14251) \[arXiv\](https://arxiv.org/html/2305.14251) on biography factuality. Per-atom verifier alignment with humans ≈ 80–82%. ChatGPT FActScore = 58.7%, \[arXiv\](https://arxiv.org/abs/2305.14251) \[ACL Anthology\](https://aclanthology.org/2023.emnlp-main.741/) GPT-4 ≈ 73.1% on biographies.

\*\*Compute\*\*: 1 LLM call for atom extraction + 1 NLI/verify call per atom. Reference verifier was Inst-LLAMA-7B; \[arXiv\](https://arxiv.org/html/2507.05965v1) \*\*swap for Llama-3.2-3B or Phi-3.5-mini to satisfy the 3B constraint\*\*.

\*\*Fit\*\*: \*\*Drop-in replacement for MiniCheck at atomic granularity\*\* using CAEM's existing RAG index — no extra search APIs. Compose with existing entailment.

\*\*Risk\*\*: Atom decomposition can over-fragment; benchmark-specific atomicity is uneven (StrategyQA multi-hop claims fragment poorly).

\## 14. Conformal Language Modeling

\*\*Citation\*\*: Quach, Fisch, Schuster, Yala, Sohn, Jaakkola, Barzilay, "Conformal Language Modeling," ICLR 2024. https://arxiv.org/abs/2306.10193

\*\*Mechanism\*\*: Built on Learn-then-Test. Calibrates three thresholds simultaneously via LTT multiple-hypothesis testing — a stopping rule, a rejection rule, and a confidence rule — yielding a prediction set that contains ≥1 correct answer with probability 1−α. Component-level extension flags individual hallucinated phrases with calibrated false-claim rate.

\*\*Gold supervision\*\*: ~1–2k annotated calibration examples. NONE at filter time.

\*\*Empirical\*\*: TriviaQA at α=10% miscoverage with average set size 3–5; component-level non-hallucination guarantee on MIMIC-CXR.

\*\*Compute\*\*: K=20–40 samples per query + logit access (white-box). LTT calibration is one-time grid search.

\*\*Fit\*\*: Component-level guarantee fits as a sub-claim abstention gate; set-output less natural for CAEM (which stores point answers).

\*\*Risk\*\*: White-box requirement; expensive sampling; exchangeability across the 7 benchmarks.

\## 15. Selective QA under Domain Shift (foundational baseline)

\*\*Citation\*\*: Kamath, Jia, Liang, "Selective Question Answering under Domain Shift," ACL 2020. https://arxiv.org/abs/2006.09462

\*\*Mechanism\*\*: Trains a random-forest \*\*calibrator\*\* over features (softmax max, top-K probabilities, input length, source-domain probability) to predict P(QA model is correct), using a mixture of in-domain + held-out OOD data \*\*without target-domain labels\*\*. Selective rule: answer iff calibrator confidence > target-precision threshold.

\*\*Gold supervision\*\*: Calibrator training requires correct/incorrect labels on its source data only (~thousands of QA examples). NONE on the target/test domain at filter time.

\*\*Empirical\*\*: SQuAD-trained QA on a mix of SQuAD + 5 OOD QA: at \*\*80% precision, calibrator achieves 56% coverage\*\* vs. MaxProb's 48%. At 90% precision, 33% vs. 14%. \*\*Direct empirical demonstration of the 80% precision–coverage tradeoff under domain shift.\*\*

\*\*Compute\*\*: Trivial — random forest over a handful of features.

\*\*Fit\*\*: \*\*Strong conceptual baseline for CAEM's stack\*\* — replace isotonic with RF/GBM calibrator + the OOD-mixing trick to handle 7-benchmark heterogeneity without exposing benchmark identity at runtime.

\*\*Risk\*\*: No formal conformal guarantee; cross-benchmark calibrator generalization depends on the OOD-mix approximating the deployment distribution.

\---

\## Synthesis — recommended path to 80%

\*\*The single highest-confidence path to ≥80% precision is the conformal-factuality stack\*\* (#1 + #2 + #8): Mohri–Hashimoto's claim-back-off filter, calibrated via Cherian et al.'s conditional CP (which absorbs the 9-signal composite as the boosting score and respects benchmark heterogeneity without runtime benchmark identity), gated by Yadkori-style conformal risk control targeting hallucination rate ≤0.2. Mohri–Hashimoto \*\*explicitly demonstrates 78%→93% factuality on NQ\*\* with GPT-4 — the only published method on this list crossing 80% on a CAEM benchmark. Layered with \*\*FlyingSquid\*\* (#4) as a label-free latent-class wrapper that converts the existing 9 signals into a calibrated posterior, this combination satisfies every CAEM constraint. \*\*However: no published method demonstrates ≥80% precision on the full 7-benchmark mix (FEVER + TriviaQA + NQ + TruthfulQA + StrategyQA + ARC-C + ASQA) without gold supervision when the underlying generator is a 3B-class model\*\* — Mohri–Hashimoto's NQ result uses GPT-4. For Qwen-2.5-3B, the realistic claim is a 70%→78% precision lift on factual benchmarks; reaching 80% likely requires either a small calibration set (STaR-extension framing) or restricting the headline claim to NQ/TriviaQA where conformal-factuality numbers are strongest. \*\*Reporting 70% with full transparency, plus a conformal-factuality ablation showing the path to 80% with a labeled calibration fold, is the most defensible thesis framing.\*\*