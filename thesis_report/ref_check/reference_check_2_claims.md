# Reference check 2: do the cited papers support the thesis claims?

This covers every reference marked *paper found* in `reference_check_1_found.md` (94 of 95; [78] `semanticillusion2025` is excluded because no such paper exists). Reference numbers are the printed thesis numbers.

For each reference, every place the thesis cites it is listed with the **full sentence** (or table row) that carries the citation. Each citation is then compared with what the paper says.

**Verdicts**
- ✅ **Matched**: the claim is consistent with the paper. Citations that only attribute a method, model or dataset ("we use X [n]") count as matched when the attribution is correct.
- ⚠️ **Partly**: part of the claim matches, but a detail is wrong or overstated.
- ❌ **Not matched**: the paper does not say this, or says the opposite.
- 🔒 **Needs full text**: the claim depends on a number or detail deep in the paper that could not be checked from the abstract.

**% matched** = ✅ citations ÷ all citations of that reference.

> **Limitation:** this environment cannot open full-text PDFs (arXiv, ACL Anthology, PMLR, OpenReview, PubMed are blocked). The "What the paper says" lines are paraphrased from each paper's abstract and public record, not quoted from the full text. The 🔒 rows need the PDFs; see the list below.

## Overall: 269 citations · ✅ 200 (74%) · ⚠️ 41 · ❌ 17 · 🔒 11

| Ref | Title | Cited | ✅ | ⚠️ | ❌ | 🔒 | % matched |
|---|---|---|---|---|---|---|---|
| [1] | Artificial Hallucinations in ChatGPT: Implications in Scientific Writi | 6 | 1 | 0 | 5 | 0 | **17%** |
| [2] | Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyo | 3 | 0 | 0 | 3 | 0 | **0%** |
| [3] | TruthfulQA: Measuring How Models Mimic Human Falsehoods | 9 | 7 | 1 | 0 | 1 | **78%** |
| [4] | Enhanced Conformal Prediction with Boosted Logistic Regression | 5 | 1 | 4 | 0 | 0 | **20%** |
| [5] | Language Models with Conformal Factuality Guarantees | 4 | 3 | 0 | 0 | 1 | **75%** |
| [6] | LoRA: Low-Rank Adaptation of Large Language Models | 6 | 5 | 1 | 0 | 0 | **83%** |
| [7] | GPT-4 Technical Report | 1 | 1 | 0 | 0 | 0 | **100%** |
| [8] | Claude 2 | 1 | 1 | 0 | 0 | 0 | **100%** |
| [9] | PaLM 2 Technical Report | 1 | 1 | 0 | 0 | 0 | **100%** |
| [10] | FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long | 4 | 4 | 0 | 0 | 0 | **100%** |
| [11] | Survey of Hallucination in Natural Language Generation | 4 | 2 | 0 | 1 | 1 | **50%** |
| [12] | Large Language Models Can Self-Improve | 3 | 0 | 0 | 3 | 0 | **0%** |
| [13] | On Calibration of Modern Neural Networks | 5 | 5 | 0 | 0 | 0 | **100%** |
| [14] | Language Models (Mostly) Know What They Know | 6 | 3 | 1 | 2 | 0 | **50%** |
| [15] | Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confide | 5 | 1 | 4 | 0 | 0 | **20%** |
| [16] | Overcoming catastrophic forgetting in neural networks | 10 | 9 | 0 | 0 | 1 | **90%** |
| [17] | Detecting Hallucinations in Large Language Models Using Semantic Entro | 13 | 9 | 2 | 2 | 0 | **69%** |
| [18] | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 6 | 6 | 0 | 0 | 0 | **100%** |
| [19] | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | 5 | 3 | 2 | 0 | 0 | **60%** |
| [20] | Self-Consistency Improves Chain of Thought Reasoning in Language Model | 5 | 3 | 2 | 0 | 0 | **60%** |
| [21] | SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Gene | 1 | 1 | 0 | 0 | 0 | **100%** |
| [22] | HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answer | 1 | 1 | 0 | 0 | 0 | **100%** |
| [23] | Attention is All you Need | 1 | 1 | 0 | 0 | 0 | **100%** |
| [24] | Qwen2.5: A Party of Foundation Models | 3 | 3 | 0 | 0 | 0 | **100%** |
| [25] | Kernel Language Entropy: Fine-grained Uncertainty Quantification for L | 5 | 5 | 0 | 0 | 0 | **100%** |
| [26] | MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents | 5 | 4 | 1 | 0 | 0 | **80%** |
| [27] | The PASCAL Recognising Textual Entailment Challenge | 1 | 1 | 0 | 0 | 0 | **100%** |
| [28] | RoBERTa: A Robustly Optimized BERT Pretraining Approach | 2 | 1 | 1 | 0 | 0 | **50%** |
| [29] | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | 4 | 3 | 1 | 0 | 0 | **75%** |
| [30] | Billion-Scale Similarity Search with GPUs | 3 | 3 | 0 | 0 | 0 | **100%** |
| [31] | Catastrophic Interference in Connectionist Networks: The Sequential Le | 1 | 1 | 0 | 0 | 0 | **100%** |
| [32] | Hallucination Rates and Reference Accuracy of ChatGPT and Bard for Sys | 2 | 1 | 1 | 0 | 0 | **50%** |
| [33] | Scaling Instruction-Finetuned Language Models | 1 | 0 | 0 | 0 | 1 | **0%** |
| [34] | FEVER: a Large-scale Dataset for Fact Extraction and VERification | 5 | 5 | 0 | 0 | 0 | **100%** |
| [35] | Mitigating LLM Hallucinations via Conformal Abstention | 2 | 2 | 0 | 0 | 0 | **100%** |
| [36] | Conformal Language Modeling | 1 | 1 | 0 | 0 | 0 | **100%** |
| [37] | Predicting Good Probabilities With Supervised Learning | 5 | 3 | 2 | 0 | 0 | **60%** |
| [38] | Selective Classification for Deep Neural Networks | 3 | 3 | 0 | 0 | 0 | **100%** |
| [39] | Dropout as a Bayesian Approximation: Representing Model Uncertainty in | 3 | 3 | 0 | 0 | 0 | **100%** |
| [40] | Just Ask for Calibration: Strategies for Eliciting Calibrated Confiden | 1 | 0 | 1 | 0 | 0 | **0%** |
| [41] | A Survey of Confidence Estimation and Calibration in Large Language Mo | 2 | 1 | 1 | 0 | 0 | **50%** |
| [42] | INSIDE: LLMs' Internal States Retain the Power of Hallucination Detect | 2 | 1 | 0 | 0 | 1 | **50%** |
| [43] | Selective Question Answering under Domain Shift | 2 | 2 | 0 | 0 | 0 | **100%** |
| [44] | Neural Turing Machines | 3 | 3 | 0 | 0 | 0 | **100%** |
| [45] | Memory Networks | 3 | 3 | 0 | 0 | 0 | **100%** |
| [46] | End-To-End Memory Networks | 1 | 1 | 0 | 0 | 0 | **100%** |
| [47] | Hybrid Computing Using a Neural Network with Dynamic External Memory | 3 | 3 | 0 | 0 | 0 | **100%** |
| [48] | Larimar: Large Language Models with Episodic Memory Control | 3 | 3 | 0 | 0 | 0 | **100%** |
| [49] | A-MEM: Agentic Memory for LLM Agents | 3 | 3 | 0 | 0 | 0 | **100%** |
| [50] | Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Mod | 6 | 6 | 0 | 0 | 0 | **100%** |
| [51] | When Not to Trust Language Models: Investigating Effectiveness of Para | 3 | 3 | 0 | 0 | 0 | **100%** |
| [52] | Self-Knowledge Guided Retrieval Augmentation for Large Language Models | 3 | 3 | 0 | 0 | 0 | **100%** |
| [53] | LLM-Independent Adaptive RAG: Let the Question Speak for Itself | 2 | 2 | 0 | 0 | 0 | **100%** |
| [54] | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Re | 3 | 3 | 0 | 0 | 0 | **100%** |
| [55] | Probing-RAG: Self-Probing to Guide Language Models in Selective Docume | 3 | 2 | 1 | 0 | 0 | **67%** |
| [56] | LoRA Learns Less and Forgets Less | 2 | 1 | 0 | 0 | 1 | **50%** |
| [57] | QLoRA: Efficient Finetuning of Quantized LLMs | 2 | 0 | 2 | 0 | 0 | **0%** |
| [58] | A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA | 1 | 1 | 0 | 0 | 0 | **100%** |
| [59] | Orthogonal Subspace Learning for Language Model Continual Learning | 2 | 2 | 0 | 0 | 0 | **100%** |
| [60] | Learning with Noisy Labels | 3 | 2 | 1 | 0 | 0 | **67%** |
| [61] | STaR: Bootstrapping Reasoning With Reasoning | 6 | 5 | 1 | 0 | 0 | **83%** |
| [62] | When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey o | 2 | 2 | 0 | 0 | 0 | **100%** |
| [63] | Don't Stop Pretraining: Adapt Language Models to Domains and Tasks | 1 | 1 | 0 | 0 | 0 | **100%** |
| [64] | V-STaR: Training Verifiers for Self-Taught Reasoners | 4 | 3 | 1 | 0 | 0 | **75%** |
| [65] | Mind the Gap: Examining the Self-Improvement Capabilities of Large Lan | 2 | 0 | 2 | 0 | 0 | **0%** |
| [66] | Escaping Model Collapse via Synthetic Data Verification: Near-Term Imp | 2 | 0 | 1 | 0 | 1 | **0%** |
| [67] | Retraining with Predicted Hard Labels Provably Increases Model Accurac | 2 | 0 | 2 | 0 | 0 | **0%** |
| [68] | Self-Improvement in Language Models: The Sharpening Mechanism | 3 | 2 | 1 | 0 | 0 | **67%** |
| [69] | Theoretical Analysis of Weak-to-Strong Generalization | 2 | 2 | 0 | 0 | 0 | **100%** |
| [70] | Dense Passage Retrieval for Open-Domain Question Answering | 4 | 4 | 0 | 0 | 0 | **100%** |
| [71] | Active Retrieval Augmented Generation | 2 | 2 | 0 | 0 | 0 | **100%** |
| [72] | TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Rea | 4 | 3 | 1 | 0 | 0 | **75%** |
| [73] | Natural Questions: A Benchmark for Question Answering Research | 1 | 1 | 0 | 0 | 0 | **100%** |
| [74] | Did Aristotle Use a Laptop? A Question Answering Benchmark with Implic | 2 | 2 | 0 | 0 | 0 | **100%** |
| [75] | Think you have Solved Question Answering? Try ARC, the AI2 Reasoning C | 1 | 1 | 0 | 0 | 0 | **100%** |
| [76] | Measuring Massive Multitask Language Understanding | 3 | 3 | 0 | 0 | 0 | **100%** |
| [77] | Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimatio | 2 | 2 | 0 | 0 | 0 | **100%** |
| [79] | HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large L | 1 | 1 | 0 | 0 | 0 | **100%** |
| [80] | Cost-Effective Hallucination Detection for LLMs | 1 | 0 | 1 | 0 | 0 | **0%** |
| [81] | LM-Polygraph: Uncertainty Estimation for Language Models | 1 | 0 | 0 | 0 | 1 | **0%** |
| [82] | An Empirical Investigation of the Role of Pre-training in Lifelong Lea | 1 | 0 | 0 | 0 | 1 | **0%** |
| [83] | Pretrained Language Model in Continual Learning: A Comparative Study | 1 | 0 | 0 | 0 | 1 | **0%** |
| [84] | CommonsenseQA: A Question Answering Challenge Targeting Commonsense Kn | 3 | 3 | 0 | 0 | 0 | **100%** |
| [85] | Energy and Policy Considerations for Deep Learning in NLP | 1 | 1 | 0 | 0 | 0 | **100%** |
| [86] | Did Aristotle Use a Laptop? A Question Answering Benchmark with Implic | 2 | 2 | 0 | 0 | 0 | **100%** |
| [87] | A Simple Sequentially Rejective Multiple Test Procedure | 2 | 2 | 0 | 0 | 0 | **100%** |
| [88] | Note on the Sampling Error of the Difference Between Correlated Propor | 2 | 2 | 0 | 0 | 0 | **100%** |
| [89] | Confidence-Aware Routing for Large Language Model Reliability Enhancem | 1 | 1 | 0 | 0 | 0 | **100%** |
| [90] | Reinforced Self-Training (ReST) for Language Modeling | 1 | 1 | 0 | 0 | 0 | **100%** |
| [91] | Large Language Models Are Reasoning Teachers | 1 | 0 | 1 | 0 | 0 | **0%** |
| [92] | A Stochastic Approximation Method | 1 | 1 | 0 | 0 | 0 | **100%** |
| [93] | New Stochastic Approximation Type Procedures | 1 | 1 | 0 | 0 | 0 | **100%** |
| [94] | Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertai | 1 | 0 | 1 | 0 | 0 | **0%** |
| [95] | On Mixup Training: Improved Calibration and Predictive Uncertainty for | 1 | 0 | 0 | 1 | 0 | **0%** |

## Papers needed to finish the 🔒 rows (please upload to Google Drive)

- [3] TruthfulQA: Measuring How Models Mimic Human Falsehoods (Lin et al., 2022)
- [5] Language Models with Conformal Factuality Guarantees (Mohri & Hashimoto, 2024)
- [11] Survey of Hallucination in Natural Language Generation (Ji et al., 2023)
- [16] Overcoming catastrophic forgetting in neural networks (Kirkpatrick et al., 2017)
- [33] Scaling Instruction-Finetuned Language Models (Chung et al., 2024)
- [42] INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection (Chen et al., 2024)
- [56] LoRA Learns Less and Forgets Less (Biderman et al., 2024)
- [66] Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence (Yi, Liu, Cheng, Xu; arXiv:2510.16657)
- [81] Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph (Vashurin et al., TACL 2025; arXiv:2406.15627)
- [82] An Empirical Investigation of the Role of Pre-training in Lifelong Learning (Mehta et al., 2023)
- [83] Pretrained Language Model in Continual Learning: A Comparative Study (Wu et al., 2022)

Optional, to firm up ⚠️ rows: [4] Cherian et al. (arXiv:2406.09714), [80] Valentin et al. (arXiv:2407.21424), [55] Probing-RAG (arXiv:2410.13339), [57] QLoRA (arXiv:2305.14314).

## Reference not found

- [78] *Semantic Illusion: Generic Natural Language Inference Models Miscalibrate on Language Model Hallucinations* ("Anon.", 2025). If you have the source this came from, upload it. Otherwise replace it with Sinha 2025, arXiv:2512.15068.

---

## [1] Artificial Hallucinations in ChatGPT: Implications in Scientific Writing
`alkaissi2023artificial` · Alkaissi & McFarlane, 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Short Cureus editorial (Feb 2023). The authors asked ChatGPT to write about the pathogenesis of two clinical cases: homocystinuria-associated osteoporosis and late-onset Pompe disease. They report a mix of correct content and fabricated or incorrect content, which they call "artificial hallucinations", and warn about its use in scientific writing. It reports no hallucination-rate percentages and no 25-question study.

**Cited 6 times. Matched 1/6 = 17%** (partly 0, not matched 5, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `core/abstract.tex:20` | Large language models produce confident but factually incorrect outputs at rates that span three to ninety-one percent across deployment domains \cite{alkaissi2023artificial,10.1145/3649506}, a failure mode that increased parameter count alone does not close \cite{lin-etal-2022-truthfulqa}. | ❌ Not matched | The paper reports no rate. The "3–91%" range is not in it. |
| 2 | `chapters/chapter_1.tex:21` | Clinical-question studies of frontier models report hallucination rates between sixty-nine and seventy-seven percent of responses \cite{alkaissi2023artificial}, an order of magnitude above any threshold a responsible deployment would tolerate. | ❌ Not matched | The paper has no 69–77% figure and no clinical-question rate. |
| 3 | `chapters/chapter_2.tex:90` | Alkaissi and McFarlane \cite{alkaissi2023artificial} asked ChatGPT to respond to 25 clinical questions and found that 69--77\% contained hallucinated medical information. | ❌ Not matched | There was no 25-question study and no 69–77% result. The paper is an editorial on two clinical cases. |
| 4 | `chapters/chapter_2.tex:295` | Hallucination-measurement studies report prevalence rates varying widely by domain, reaching above 90\% in worst-case clinical settings \cite{alkaissi2023artificial}. | ❌ Not matched | No rate above 90%. 91.4% is Chelli et al.'s Bard reference-hallucination rate ([32]). |
| 5 | `chapters/chapter_3.tex:94` | In domains where fluency without grounding has produced documented harm, including health-information retrieval~\cite{alkaissi2023artificial,info:doi/10.2196/53164}, legal triage, and educational tutoring~\cite{ji2023survey}, an architecture that abstains rather than fabricates is a strict improvement on the deployed alternative. | ✅ Matched | Matches: it documents fabricated or incorrect ChatGPT content in medical writing. |
| 6 | `appendix/appendix_g.tex:30` | The published rate of confident-but-wrong outputs in the medical-question-answering domain reaches 69--77\% \cite{alkaissi2023artificial}, and TruthfulQA's inverse-scaling result \cite{lin-etal-2022-truthfulqa} establishes that scaling alone does not close this regime; only an architectural intervention with a per-query confidence axis does. | ❌ Not matched | The 69–77% figure is not in the paper. |

## [2] Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond
`10.1145/3649506` · Yang et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Yang et al., ACM TKDD 2024. A practical guide and survey on using LLMs (ChatGPT and others) in downstream NLP tasks: models, data, and task-by-task guidance. It is not a hallucination-taxonomy survey.

**Cited 3 times. Matched 0/3 = 0%** (partly 0, not matched 3, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `core/abstract.tex:20` | Large language models produce confident but factually incorrect outputs at rates that span three to ninety-one percent across deployment domains \cite{alkaissi2023artificial,10.1145/3649506}, a failure mode that increased parameter count alone does not close \cite{lin-etal-2022-truthfulqa}. | ❌ Not matched | This survey reports no 3–91% hallucination range. |
| 2 | `chapters/chapter_2.tex:21` | The second dimension distinguishes \textit{factuality hallucinations}, which conflict with verifiable world knowledge, from \textit{faithfulness hallucinations}, which contradict user input or task context \cite{10.1145/3649506}. | ❌ Not matched | The factuality/faithfulness taxonomy is from Lei Huang et al., *A Survey on Hallucination in LLMs*, ACM TOIS (doi:10.1145/3703155), not this paper. |
| 3 | `chapters/chapter_2.tex:82` | Huang et al. \cite{10.1145/3649506} extended this study specifically for large language models in ACM Transactions on Information Systems. | ❌ Not matched | This paper is in ACM TKDD and is not a hallucination survey. The TOIS hallucination survey is Lei Huang et al. |

## [3] TruthfulQA: Measuring How Models Mimic Human Falsehoods
`lin-etal-2022-truthfulqa` · Lin et al., 2022 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** TruthfulQA (ACL 2022). 817 questions in 38 categories (e.g. health, law, finance, politics) designed to elicit imitative falsehoods. The best model (GPT-3 175B) was truthful on 58% of questions vs. 94% for humans. Larger models were generally less truthful (inverse scaling).

**Cited 9 times. Matched 7/9 = 78%** (partly 1, not matched 0, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `core/abstract.tex:20` | Large language models produce confident but factually incorrect outputs at rates that span three to ninety-one percent across deployment domains \cite{alkaissi2023artificial,10.1145/3649506}, a failure mode that increased parameter count alone does not close \cite{lin-etal-2022-truthfulqa}. | ✅ Matched | Matches: larger models were generally less truthful (inverse scaling). |
| 2 | `chapters/chapter_1.tex:19` | The TruthfulQA benchmark of Lin and colleagues isolates questions on which humans themselves are prone to hold confident misconceptions, and reports that the largest frontier model evaluated achieved only fifty-eight percent on the truthfulness axis and twenty-five percent on the combined truthful-and-informative axis, against a human baseline of ninety-four percent on both \cite{lin-etal-2022-truthfulqa}. | ⚠️ Partly | 58% (best model, truthful) and 94% (human, truthful) match. "94% on both" is wrong: the human truthful+informative score is lower. The 25% figure needs the full text. |
| 3 | `chapters/chapter_2.tex:84` | Lin et al. \cite{lin-etal-2022-truthfulqa} introduced TruthfulQA at ACL2022, a benchmark of 817 questions across 38 categories which has been specifically designed to elicit imitative falsehoods where models reproduce common human misconceptions found in training data. | ✅ Matched | Matches: 817 questions, 38 categories, imitative falsehoods, ACL 2022. |
| 4 | `chapters/chapter_2.tex:283` | Lin et al.~\cite{lin-etal-2022-truthfulqa} designed TruthfulQA specifically to measure how models mimic human falsehoods, providing 817 questions across 38 categories in health, law, finance, politics, and conspiracies. | ✅ Matched | Matches: 817 questions, 38 categories including health, law, finance and politics. |
| 5 | `chapters/chapter_2.tex:295` | TruthfulQA's inverse-scaling finding \cite{lin-etal-2022-truthfulqa}, in which GPT-3 175B achieved only 21--25\% on the combined truthful-and-informative metric, is the clearest demonstration that parametric scaling alone is insufficient for truthfulness, and FActScore \cite{min-etal-2023-factscore} shows that even competent models support only about 58\% of their atomic facts, which is why CAEM verifies atomic facts rather than holistic responses. | 🔒 Needs full text | Inverse scaling matches. The exact 21–25% truthful+informative figure for GPT-3 175B needs the full text. |
| 6 | `chapters/chapter_2.tex:307` | TruthfulQA \cite{lin-etal-2022-truthfulqa} and StrategyQA \cite{10.1162/tacl_a_00370} are held out from the SIL training pool as transfer-only benchmarks, exercising the cross-encoder relevance signal on imitative falsehoods and sample-set agreement on implicit multi-hop reasoning respectively. | ✅ Matched | Usage of the benchmark; the attribution is correct. |
| 7 | `chapters/chapter_3.tex:145` | The five evaluation benchmarks are FEVER \cite{thorne-etal-2018-fever}, TriviaQA \cite{joshi-etal-2017-triviaqa}, CommonsenseQA \cite{talmor-etal-2019-commonsenseqa}, TruthfulQA \cite{lin-etal-2022-truthfulqa}, and StrategyQA \cite{geva2021strategyqa}, all of which are distributed under licences that permit research use and redistribution under attribution; the licence terms are catalogued in Appendix~\ref{app:reproducibility} together with the access date and the dataset version hash. | ✅ Matched | Usage of the benchmark; the attribution is correct. |
| 8 | `chapters/chapter_5.tex:41` | The transfer panel comprises TruthfulQA \cite{lin-etal-2022-truthfulqa} for common-misconception probes and StrategyQA \cite{geva2021strategyqa} for implicit multi-step reasoning. | ✅ Matched | Usage of the benchmark; the attribution is correct. |
| 9 | `appendix/appendix_g.tex:30` | The published rate of confident-but-wrong outputs in the medical-question-answering domain reaches 69--77\% \cite{alkaissi2023artificial}, and TruthfulQA's inverse-scaling result \cite{lin-etal-2022-truthfulqa} establishes that scaling alone does not close this regime; only an architectural intervention with a per-query confidence axis does. | ✅ Matched | Matches: inverse scaling. |

## [4] Enhanced Conformal Prediction with Boosted Logistic Regression
`cherian2024conformal` · Cherian et al., 2024 · details matched: ❌ No

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2024. Extends conformal factuality (Mohri & Hashimoto) in two ways. Level-adaptive conditional conformal guarantees issue weaker guarantees when needed to keep outputs useful. "Conditional boosting" improves the claim-scoring function by differentiating through the conditional conformal procedure. Evaluated on biography and medical QA.

**Cited 5 times. Matched 1/5 = 20%** (partly 4, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `core/abstract.tex:20` | The ensemble feeds a per-benchmark calibrated-probability composite (per-signal isotonic regression, James-Stein shrinkage at $\alpha_{\text{shrink}} = 0.6$, regularised logistic boost \cite{cherian2024conformal}) which gates storage at two fixed cut-points, $\tau_{\text{store}} = 0.60$ and $\tau_{\text{defer}} = 0.45$. | ⚠️ Partly | The paper's "conditional boosting" learns a better claim-scoring function through the conformal procedure. It is not a regularised logistic regression on log-odds. |
| 2 | `chapters/chapter_2.tex:118` | Cherian et al.~\cite{cherian2024conformal} sharpen the Mohri--Hashimoto framework with a boosted logistic aggregation over multiple per-claim scores, and supply the algorithmic basis for CAEM's boost layer over the per-signal log-odds in \Cref{sec:composite-boost}. | ⚠️ Partly | Building on Mohri–Hashimoto and boosting a combined per-claim score both match. "Logistic aggregation" is CAEM's own design, not the paper's. |
| 3 | `chapters/chapter_2.tex:120` | The composite is not a fixed weighted sum: each signal is mapped to a calibrated $\Pr(\text{em}=1\mid\text{signal})$ through per-signal isotonic regression with auto-detected direction~\cite{niculescu2005predicting}, the per-signal curves are blended with a per-benchmark child fit through a James-Stein style shrinkage prior, and the per-signal log-odds are aggregated through an $L_{2}$-regularised logistic boost layer in the lineage of Cherian-Gibbs-Cand\`{e}s conditional boosting~\cite{cherian2024conformal}. | ✅ Matched | Attribution only: "in the lineage of Cherian–Gibbs–Candès conditional boosting" is fair. |
| 4 | `chapters/chapter_4.tex:343` | The boost layer aggregates the per-signal calibrated probabilities by fitting a logistic regression on the per-signal log-odds as features against the exact-match labels of the calibration fold \cite{cherian2024conformal}. | ⚠️ Partly | The paper does not fit a logistic regression on per-signal log-odds against labels. Its boosting differentiates through the conformal procedure. |
| 5 | `chapters/chapter_5.tex:600` | The first group is conformal-factuality at the language-model claim level: \cite{mohri2024conformal} demonstrates a precision lift from $78\%$ to $93\%$ on Natural Questions at $\alpha=0.2$ using GPT-4 with split-conformal sub-claim filtering, and \cite{cherian2024conformal} sharpens this with the boosted logistic aggregation that CAEM's composite layer adopts. | ⚠️ Partly | Same issue: "boosted logistic aggregation" is not the paper's method. |

## [5] Language Models with Conformal Factuality Guarantees
`mohri2024conformal` · Mohri & Hashimoto, 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICML 2024. Conformal factuality: decompose LM outputs into sub-claims, score them, and remove claims below a split-conformal threshold, so the remaining output is correct with probability ≥1−α. Evaluated on FActScore, Natural Questions and MATH, with 80–90% correctness guarantees. NQ factuality starts at about 78%.

**Cited 4 times. Matched 3/4 = 75%** (partly 0, not matched 0, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `core/abstract.tex:20` | The design replaced an earlier split-conformal alternative \cite{mohri2024conformal} that was rejected on the registered calibration-to-evaluation distribution-shift evidence. | ✅ Matched | Matches: the paper uses split conformal prediction. |
| 2 | `chapters/chapter_2.tex:118` | Mohri and Hashimoto~\cite{mohri2024conformal} introduced \emph{Language Models with Conformal Factuality Guarantees} at ICML 2024, decomposing each generated answer into atomic sub-claims, scoring each, and removing claims below a calibrated threshold $\tau$ such that the surviving sub-claims are jointly correct with probability at least $1-\alpha$ on a small annotated calibration set. | ✅ Matched | Matches: sub-claim decomposition, scoring, calibrated threshold, correctness guarantee of at least 1−α, ICML 2024. |
| 3 | `chapters/chapter_2.tex:252` | [table row] Mohri-Hashimoto~\cite{mohri2024conformal} & no & --- & split-conformal sub-claim filter & --- & conformal factuality | ✅ Matched | Table row matches: split-conformal sub-claim filter with a conformal factuality guarantee. |
| 4 | `chapters/chapter_5.tex:600` | The first group is conformal-factuality at the language-model claim level: \cite{mohri2024conformal} demonstrates a precision lift from $78\%$ to $93\%$ on Natural Questions at $\alpha=0.2$ using GPT-4 with split-conformal sub-claim filtering, and \cite{cherian2024conformal} sharpens this with the boosted logistic aggregation that CAEM's composite layer adopts. | 🔒 Needs full text | The NQ start of about 78% is confirmed. "→93% at α=0.2 with GPT-4" needs the full text. |

## [6] LoRA: Low-Rank Adaptation of Large Language Models
`hu2022lora` · Hu et al., 2022 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** LoRA (ICLR 2022). Freezes pretrained weights and injects trainable rank-decomposition matrices. On GPT-3 175B it cuts trainable parameters about 10,000× with quality on par with full fine-tuning. Applied to attention weights (Wq, Wv) in the main experiments.

**Cited 6 times. Matched 5/6 = 83%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `core/abstract.tex:20` | Self-improvement is delivered through low-rank adaptation \cite{hu2022lora} at rank thirty-two and scaling factor sixty-four over the attention and feed-forward projections, with the backbone frozen; after every fine-tune the loop refits temperature, per-signal isotonic curves, and per-benchmark boost weights on the re-scored calibration fold (the same held-out fold re-scored under the updated model, not a new fold) and runs retroactive re-verification before the next cycle. | ✅ Matched | Usage of LoRA; the attribution is correct. |
| 2 | `chapters/chapter_1.tex:120` | Fine-tune the generator through a low-rank adaptation layer on the attention and feed-forward projections rather than through full-parameter fine-tuning \cite{hu2022lora}, with the trainable footprint at roughly two percent of the backbone parameter count and the underlying backbone frozen across the cycle. | ✅ Matched | Matches: LoRA as an alternative to full fine-tuning. |
| 3 | `chapters/chapter_2.tex:14` | CAEM ships the parameter-efficient path through the low-rank adaptation layer of \cite{hu2022lora} on the attention and feed-forward projections, with the trainable footprint at roughly two percent of the backbone parameter count and the underlying backbone frozen across every cycle of the self-improvement loop. | ✅ Matched | Usage; the attribution is correct. |
| 4 | `chapters/chapter_2.tex:70` | The first is the parameter-efficient adaptation primitive itself: low-rank adaptation \cite{hu2022lora} on the attention and feed-forward projections, with the trainable footprint at roughly two percent of the backbone parameter count and the underlying backbone frozen across the cycle, retains general capability across iterative cycles substantially better than full-parameter fine-tuning at matched training budget on the empirical evidence reviewed in \Cref{sec:lit-self-improvement}. | ✅ Matched | Attribution of LoRA is correct. The retention claim itself is supported by [56], not by this paper. |
| 5 | `chapters/chapter_2.tex:212` | Hu and colleagues introduced low-rank adaptation \cite{hu2022lora} as a parameter-efficient alternative to full fine-tuning that injects trainable low-rank matrices into the attention and feed-forward projections of a frozen backbone, with the original paper reporting near-full-FT quality at less than one percent of the trainable parameter count on GPT-3 175B. | ⚠️ Partly | "Under 1% trainable parameters with near-full-FT quality on GPT-3 175B" matches. The original paper adapts attention weights only; it does not inject into feed-forward projections. |
| 6 | `chapters/chapter_2.tex:305` | The parameter-efficient-fine-tuning literature reviewed at \Cref{sec:lit-self-improvement} establishes that low-rank adaptation \cite{hu2022lora} on all linear layers retains general capability across iterative cycles substantially better than full-parameter fine-tuning at matched training budget \cite{biderman2024lora,dettmers2023qlora}, which is the empirical anchor for CAEM's choice of the LoRA primary path with the $L_2$ anchor placed on the adapter parameters rather than on the full backbone. | ✅ Matched | Attribution of LoRA only. The retention claim is supported by [56]. |

## [7] GPT-4 Technical Report
`openai2023gpt4` · OpenAI, 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** GPT-4 Technical Report. GPT-4 shows human-level performance on various professional and academic benchmarks (e.g. top ~10% on a simulated bar exam).

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:12` | Frontier systems such as GPT-4, Claude, and PaLM reach or exceed reported human accuracy on standardised academic benchmarks, and their accessibility through chat interfaces has made them part of everyday research, education, and professional workflows \cite{openai2023gpt4, anthropic2023claude, anil2023palm}. | ✅ Matched | Matches: human-level performance on professional and academic benchmarks. |

## [8] Claude 2
`anthropic2023claude` · Anthropic, 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Claude 2 announcement blog. Reports improved scores on the bar exam (multiple choice), GRE and coding benchmarks, and availability via chat interface and API.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:12` | Frontier systems such as GPT-4, Claude, and PaLM reach or exceed reported human accuracy on standardised academic benchmarks, and their accessibility through chat interfaces has made them part of everyday research, education, and professional workflows \cite{openai2023gpt4, anthropic2023claude, anil2023palm}. | ✅ Matched | Matches: the blog reports bar-exam and GRE scores and availability through a chat interface. |

## [9] PaLM 2 Technical Report
`anil2023palm` · Anil et al., 2023 · details matched: ❌ No

**What the paper says (paraphrased from its abstract and public record):** PaLM 2 Technical Report. State-of-the-art results across reasoning and multilingual benchmarks, including passing advanced language-proficiency exams at mastery level.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:12` | Frontier systems such as GPT-4, Claude, and PaLM reach or exceed reported human accuracy on standardised academic benchmarks, and their accessibility through chat interfaces has made them part of everyday research, education, and professional workflows \cite{openai2023gpt4, anthropic2023claude, anil2023palm}. | ✅ Matched | Matches broadly: strong benchmark results and mastery-level language exams. |

## [10] FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation
`min-etal-2023-factscore` · Min et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** FActScore (EMNLP 2023). Breaks long-form generations into atomic facts and computes the fraction supported by a knowledge source. On people biographies, ChatGPT reaches only 58%.

**Cited 4 times. Matched 4/4 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:21` | The atomic-fact decomposition protocol of Min and colleagues, evaluating chat-style models on biographical generation, found that only fifty-eight percent of decomposed atomic claims were supported by retrieved evidence, and the remaining claims ranged from unsupported speculation to outright fabrication \cite{min-etal-2023-factscore}. | ✅ Matched | Matches: ChatGPT reaches 58% on biographies. |
| 2 | `chapters/chapter_2.tex:88` | Min et al. \cite{min-etal-2023-factscore} introduced FActScore at EMNLP 2023, which evaluates long-form generations by decomposing the answer into atomic claims and reporting the share supported against retrieved sources rather than scoring the answer holistically. | ✅ Matched | Matches: atomic-fact decomposition, EMNLP 2023. |
| 3 | `chapters/chapter_2.tex:120` | NLI against retrieved evidence contributes the external-grounding family of three signals: a max-match entailment probability $p_{\text{ground\_max}}$, a mean-match entailment probability $p_{\text{ground\_mean}}$, and a length-gated weakest-link atomic-fact entailment probability $p_{\text{ground\_atomic}}$ that decomposes the answer into atomic claims under the FActScore convention~\cite{min-etal-2023-factscore} and runs only when the answer's token length exceeds the registered atomic-decomposition gate; below the gate the atomic signal collapses to the mean-match value. | ✅ Matched | Attribution of the atomic-fact convention is correct. |
| 4 | `chapters/chapter_2.tex:295` | TruthfulQA's inverse-scaling finding \cite{lin-etal-2022-truthfulqa}, in which GPT-3 175B achieved only 21--25\% on the combined truthful-and-informative metric, is the clearest demonstration that parametric scaling alone is insufficient for truthfulness, and FActScore \cite{min-etal-2023-factscore} shows that even competent models support only about 58\% of their atomic facts, which is why CAEM verifies atomic facts rather than holistic responses. | ✅ Matched | Matches: 58%. |

## [11] Survey of Hallucination in Natural Language Generation
`ji2023survey` · Ji et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Ji et al., ACM Computing Surveys 2023. Survey of hallucination in natural language generation. Defines intrinsic hallucination (contradicts the source) and extrinsic hallucination (cannot be verified from the source), and reviews metrics and mitigation by task.

**Cited 4 times. Matched 2/4 = 50%** (partly 0, not matched 1, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:23` | These results establish that hallucination is neither an artefact of small models nor a residual that scale will eliminate; it is, as the recent survey literature argues, a structural property of stateless generation under uncertainty rather than a defect to be trained away \cite{ji2023survey,huang-etal-2023-large}. | ❌ Not matched | Ji et al. do not argue that hallucination is a structural property that cannot be trained away. Cite e.g. Xu et al. 2024 "Hallucination is Inevitable" or Kalai & Vempala 2024 for that. |
| 2 | `chapters/chapter_2.tex:19` | The first dimension separates \textit{intrinsic hallucinations}, which directly contradict information provided in the source input, from \textit{extrinsic hallucinations}, which introduce information that goes beyond what the source provides, regardless of whether it also happens to conflict with the source \cite{ji2023survey}. | ✅ Matched | Matches: intrinsic vs. extrinsic definitions. |
| 3 | `chapters/chapter_2.tex:80` | Ji et al. \cite{ji2023survey} published the first major survey in ACM Computing Surveys. | ✅ Matched | Matches: published in ACM Computing Surveys. |
| 4 | `chapters/chapter_3.tex:94` | In domains where fluency without grounding has produced documented harm, including health-information retrieval~\cite{alkaissi2023artificial,info:doi/10.2196/53164}, legal triage, and educational tutoring~\cite{ji2023survey}, an architecture that abstains rather than fabricates is a strict improvement on the deployed alternative. | 🔒 Needs full text | The survey covers application risks. Whether it mentions educational tutoring specifically needs the full text. |

## [12] Large Language Models Can Self-Improve
`huang-etal-2023-large` · Huang et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Jiaxin Huang et al., EMNLP 2023, "Large Language Models Can Self-Improve". An LLM generates high-confidence, self-consistency-filtered rationales for unlabeled questions and is fine-tuned on them, improving reasoning without ground-truth labels. It is not about self-correction failing, and it is not a hallucination survey.

**Cited 3 times. Matched 0/3 = 0%** (partly 0, not matched 3, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:23` | These results establish that hallucination is neither an artefact of small models nor a residual that scale will eliminate; it is, as the recent survey literature argues, a structural property of stateless generation under uncertainty rather than a defect to be trained away \cite{ji2023survey,huang-etal-2023-large}. | ❌ Not matched | This paper ("LLMs Can Self-Improve") is not a hallucination survey. |
| 2 | `chapters/chapter_2.tex:224` | Huang et al. \cite{huang-etal-2023-large} found at EMNLP 2023 that prompting LLMs to review and correct their own responses rarely improves accuracy if we do not use external feedback. | ❌ Not matched | This finding comes from Jie Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* (ICLR 2024). This paper argues the opposite (self-improvement works). |
| 3 | `chapters/chapter_2.tex:305` | The self-correction critiques of \cite{huang-etal-2023-large} and \cite{kamoi-etal-2024-llms} establish that intrinsic prompt-based self-correction rarely improves accuracy without external feedback, which is exactly the feedback CAEM supplies through the external verifier. | ❌ Not matched | Same as the previous row: wrong Huang paper. |

## [13] On Calibration of Modern Neural Networks
`pmlr-v70-guo17a` · Chuan Guo et al., 2017 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICML 2017. Modern deep networks (e.g. ResNets) are poorly calibrated and overconfident. Temperature scaling is a simple and effective post-hoc fix.

**Cited 5 times. Matched 5/5 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:30` | Calibration studies report that both classifiers and language models assign high probability to incorrect outputs at rates that exceed their actual accuracy by ten percentage points or more \cite{pmlr-v70-guo17a, kadavath2022language}. | ✅ Matched | Matches for classifiers: modern networks are overconfident. |
| 2 | `chapters/chapter_1.tex:83` | Modern neural networks routinely assign probability mass to incorrect outputs that exceeds their actual accuracy by a wide margin \cite{pmlr-v70-guo17a, kadavath2022language}, and single-signal correctness predictors operate close to random on the discrimination axis \cite{xiong2023can}. | ✅ Matched | Matches: overconfidence. |
| 3 | `chapters/chapter_2.tex:134` | Guo et al. \cite{pmlr-v70-guo17a} demonstrated this at ICML 2017. | ✅ Matched | Matches: miscalibration shown at ICML 2017. |
| 4 | `chapters/chapter_2.tex:301` | The confidence-estimation literature shows that single signals only reach roughly 0.5--0.6 AUROC for discriminating correct from incorrect answers \cite{xiong2023can}, that modern networks are systematically overconfident and require calibration applied after training \cite{pmlr-v70-guo17a}, and that RLHF can further degrade calibration \cite{kadavath2022language}, which is one reason CAEM builds on Qwen-2.5-3B-Instruct's plain instruction-tuned checkpoint rather than an RLHF-aligned chat variant. | ✅ Matched | Matches: overconfident networks that need post-hoc calibration. |
| 5 | `chapters/chapter_4.tex:75` | The fused scalar is calibrated to a probability through temperature scaling \cite{pmlr-v70-guo17a} fitted on a disjoint calibration fold by minimising binary negative-log-likelihood under the labels-correct-under-zero-shot binary signal. | ✅ Matched | Matches: temperature scaling. |

## [14] Language Models (Mostly) Know What They Know
`kadavath2022language` · Kadavath et al., 2022 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Anthropic, arXiv 2022. Large LMs are well calibrated on multiple-choice and true/false questions when formatted appropriately. They can be trained to predict P(IK), the probability that they know the answer. RLHF policies look miscalibrated, but simple temperature adjustment fixes this.

**Cited 6 times. Matched 3/6 = 50%** (partly 1, not matched 2, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:30` | Calibration studies report that both classifiers and language models assign high probability to incorrect outputs at rates that exceed their actual accuracy by ten percentage points or more \cite{pmlr-v70-guo17a, kadavath2022language}. | ❌ Not matched | Kadavath et al. find large LMs *well calibrated* on multiple-choice and true/false formats. They do not show overconfidence of 10 pp or more. |
| 2 | `chapters/chapter_1.tex:83` | Modern neural networks routinely assign probability mass to incorrect outputs that exceeds their actual accuracy by a wide margin \cite{pmlr-v70-guo17a, kadavath2022language}, and single-signal correctness predictors operate close to random on the discrimination axis \cite{xiong2023can}. | ❌ Not matched | Same: the paper mainly reports good calibration. |
| 3 | `chapters/chapter_2.tex:140` | Kadavath et al. \cite{kadavath2022language} from Anthropic explored self evaluation in large language models at NeurIPS 2022. | ⚠️ Partly | Anthropic and self-evaluation match. Venue is wrong: the paper is arXiv-only, not NeurIPS 2022. |
| 4 | `chapters/chapter_2.tex:301` | The confidence-estimation literature shows that single signals only reach roughly 0.5--0.6 AUROC for discriminating correct from incorrect answers \cite{xiong2023can}, that modern networks are systematically overconfident and require calibration applied after training \cite{pmlr-v70-guo17a}, and that RLHF can further degrade calibration \cite{kadavath2022language}, which is one reason CAEM builds on Qwen-2.5-3B-Instruct's plain instruction-tuned checkpoint rather than an RLHF-aligned chat variant. | ✅ Matched | Matches: RLHF policies appear miscalibrated. |
| 5 | `chapters/chapter_3.tex:135` | A model that hedges or speculates under uncertainty exposes the user to harm without informing them that uncertainty is present, while a model that abstains transfers the uncertainty to the user as a visible signal that they can act on by seeking a second opinion or escalating to a human expert; this connects to the recent literature on whether language models know what they know \cite{kadavath2022language}. | ✅ Matched | Matches the paper's theme of whether LMs know what they know. |
| 6 | `chapters/chapter_4.tex:163` | The third group is the load-bearing axis of the classifier: the probe $p_{\text{IK}}$~\cite{kadavath2022language} carries the largest absolute coefficient in the trained scoring head ($-1.77$), so high parametric self-knowledge pushes the classifier strongly toward \textsc{direct}, while passage-quality features with positive coefficients push toward \textsc{rag} only when the parametric self-knowledge is low. | ✅ Matched | Matches: P(IK) probe. |

## [15] Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
`xiong2023can` · Xiong et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICLR 2024. Evaluates verbalized, sampling-based and aggregated confidence elicitation. LLMs are overconfident when verbalizing confidence. Failure-prediction AUROC is far from ideal (GPT-4 verbalized average 62.7%, white-box 0.52–0.61). The paper flags AUROC <0.6 as poor.

**Cited 5 times. Matched 1/5 = 20%** (partly 4, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:30` | Single-signal correctness predictors achieve area-under-the-receiver-operating-curve scores between zero point five zero and zero point six zero, only marginally above random chance \cite{xiong2023can}. | ⚠️ Partly | Roughly supported for *verbalized* confidence only (GPT-4 average AUROC 62.7%; white-box 0.52–0.61). Not shown for every single signal. |
| 2 | `chapters/chapter_1.tex:83` | Modern neural networks routinely assign probability mass to incorrect outputs that exceeds their actual accuracy by a wide margin \cite{pmlr-v70-guo17a, kadavath2022language}, and single-signal correctness predictors operate close to random on the discrimination axis \cite{xiong2023can}. | ⚠️ Partly | Same: only for verbalized confidence. |
| 3 | `chapters/chapter_2.tex:144` | Xiong et al. \cite{xiong2023can} conducted a detailed study of confidence estimation methods for large language models. | ✅ Matched | Matches. |
| 4 | `chapters/chapter_2.tex:301` | The confidence-estimation literature shows that single signals only reach roughly 0.5--0.6 AUROC for discriminating correct from incorrect answers \cite{xiong2023can}, that modern networks are systematically overconfident and require calibration applied after training \cite{pmlr-v70-guo17a}, and that RLHF can further degrade calibration \cite{kadavath2022language}, which is one reason CAEM builds on Qwen-2.5-3B-Instruct's plain instruction-tuned checkpoint rather than an RLHF-aligned chat variant. | ⚠️ Partly | Same as above. |
| 5 | `appendix/appendix_g.tex:70` | \cite{xiong2023can} report that any single confidence signal reaches an area-under-the-curve in the range $0.50$ to $0.60$ for distinguishing correct from incorrect outputs; the composite aggregates nine signals across four families through per-signal isotonic regression and a regularised logistic boost layer to produce a calibrated correctness probability that no single signal can match. | ⚠️ Partly | "Any single confidence signal" overgeneralises the paper. |

## [16] Overcoming catastrophic forgetting in neural networks
`doi:10.1073/pnas.1611835114` · Kirkpatrick et al., 2017 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Kirkpatrick et al., PNAS 2017. Elastic Weight Consolidation slows learning on weights important to earlier tasks, using a Fisher-information-weighted quadratic penalty. Demonstrated on permuted MNIST and sequential Atari games.

**Cited 10 times. Matched 9/10 = 90%** (partly 0, not matched 0, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:32` | Once trained they neither learn from their mistakes nor adapt to the patterns of an institution that uses them, because naive online fine-tuning incurs catastrophic forgetting on the broad capabilities the base training instilled \cite{doi:10.1073/pnas.1611835114}. | ✅ Matched | Matches: catastrophic forgetting under sequential training. |
| 2 | `chapters/chapter_1.tex:56` | The third capability is a constrained self-improvement loop that fine-tunes on a verified high-quality pool under a regulariser pulling weights toward the base parameters \cite{doi:10.1073/pnas.1611835114}, and that rolls the weights back if a held-out retention probe indicates general capability has eroded. | ✅ Matched | Matches: a quadratic penalty pulling weights toward the old parameters. |
| 3 | `chapters/chapter_1.tex:85` | Naive online fine-tuning is not a remedy because it produces catastrophic forgetting on the broad capabilities the base training instilled \cite{doi:10.1073/pnas.1611835114}. | ✅ Matched | Matches. |
| 4 | `chapters/chapter_1.tex:120` | The bounded adapter parameter budget, layered on a frozen backbone, acts as the implicit cycle-to-cycle drift bound in the spirit of elastic weight consolidation \cite{doi:10.1073/pnas.1611835114}, without the hand-tuned quadratic anchor coefficient that a full-parameter fine-tune would require. | ✅ Matched | Attribution "in the spirit of EWC" is fair. |
| 5 | `chapters/chapter_1.tex:280` | The fine-tune is regularised implicitly through the bounded low-rank-adapter parameter budget layered on a frozen backbone, in the spirit of elastic weight consolidation \cite{doi:10.1073/pnas.1611835114} but without an explicit quadratic anchor coefficient. | ✅ Matched | Attribution is fair. |
| 6 | `chapters/chapter_1.tex:397` | The architecture mitigates this risk through three layers of defence: a strict training-pool quality filter that holds the training pool to a higher bar than the user-facing storage tier, a bounded low-rank-adapter parameter envelope layered on a frozen backbone that acts as the implicit cycle-to-cycle drift bound (in the spirit of elastic weight consolidation \cite{doi:10.1073/pnas.1611835114} but without an explicit quadratic anchor coefficient on the shipped path), and a multi-modal retention probe panel whose worst per-probe ratio gates the commit. | ✅ Matched | Attribution is fair. |
| 7 | `chapters/chapter_2.tex:66` | Elastic Weight Consolidation (EWC) \cite{doi:10.1073/pnas.1611835114} addresses this by identifying essential parameters for previous tasks using the Fisher Information Matrix, then adding quadratic penalties preventing large updates to important parameters: \begin{equation} \mathcal{L} = \mathcal{L}_{\text{new}} + \frac{\lambda}{2} \sum_{i} F_i (\theta_i - \theta_i^*)^2 \end{equation} where $F_i$ measures parameter importance, $\theta^*$ are previous parameters, and $\lambda$ controls regularization strength. | ✅ Matched | Matches: Fisher-weighted quadratic penalty (equation correct). |
| 8 | `chapters/chapter_2.tex:202` | Kirkpatrick et al. \cite{doi:10.1073/pnas.1611835114} introduced Elastic Weight Consolidation (EWC) at PNAS 2017, providing a principled solution to catastrophic forgetting in neural networks. | ✅ Matched | Matches: introduced EWC, PNAS 2017. |
| 9 | `chapters/chapter_2.tex:305` | EWC \cite{doi:10.1073/pnas.1611835114} maintains approximately 90\% performance across sequential tasks and supplies the conceptual motivation for bounding cycle-to-cycle drift in adapter space; the explicit $L_{2}$ anchor that approximates EWC under a uniform-Fisher assumption is registered for the inactive full-fine-tune fallback only (Fisher-matrix computation at every cycle boundary on a three-billion-parameter Qwen-2.5-3B-Instruct backbone is prohibitive on a single consumer-grade GPU), while on the shipped low-rank-adapter path the bounded adapter parameter envelope on a frozen backbone acts as the implicit drift bound without the hand-tuned anchor coefficient that the full-parameter fine-tune would require. | 🔒 Needs full text | "Maintains approximately 90% performance" is not a figure stated in the abstract. Needs the full text, or remove. |
| 10 | `chapters/chapter_4.tex:480` | The quadratic anchor term is registered as the parameter envelope for the inactive full-fine-tune fallback (when it engages, $\lambda_{\text{anc}}>0$ and the bracketed term becomes the uniform-Fisher approximation to elastic weight consolidation \cite{doi:10.1073/pnas.1611835114}). | ✅ Matched | Matches: L2 is EWC with a uniform Fisher. |

## [17] Detecting Hallucinations in Large Language Models Using Semantic Entropy
`farquhar2024semantic` · Farquhar et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Nature 2024. Semantic entropy detects "confabulations", which are arbitrary and incorrect generations that vary across resamples. It samples about 10 answers, clusters them by bidirectional entailment, and takes the entropy over clusters. The paper notes it does not catch errors the model makes consistently, e.g. from erroneous training data.

**Cited 13 times. Matched 9/13 = 69%** (partly 2, not matched 2, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:42` | What is not present is the discipline of distinguishing facts that the model has correctly internalised from confabulations that emerge under genuine uncertainty, and the discipline of accumulating verified knowledge across interactions instead of regenerating answers from scratch every time \cite{farquhar2024semantic}. | ⚠️ Partly | The confabulation part matches. "Accumulating verified knowledge across interactions" is not in the paper. |
| 2 | `chapters/chapter_1.tex:49` | Post-generation verification \cite{manakul-etal-2023-selfcheckgpt, farquhar2024semantic} detects hallucinations after they occur and supports refusal or revision, but it does not feed back into model weights or memory, so the same hallucination patterns recur in the next session. | ✅ Matched | Matches: post-hoc detection. |
| 3 | `chapters/chapter_1.tex:114` | The composite replaces the fixed weighted sums used in earlier verifier ensembles \cite{farquhar2024semantic}. | ❌ Not matched | Semantic entropy is not a weighted-sum verifier ensemble. The paper has no "fixed weighted sums". |
| 4 | `chapters/chapter_2.tex:25` | \textit{Confabulations}, in the sense of \cite{farquhar2024semantic}, are outputs the model could equally have produced differently on a resample even when individual token probabilities look confident, and are detected by the sample-set agreement family of two signals ($s_{\text{avg}}$, the mean unique-pair cosine similarity over $M{=}3$ resampled chains in the sentence-embedding space, and $p_{\text{entail}}$, the mean entailment of each resampled chain against the served answer under the natural-language-inference judge).\footnote{The nine signals are defined formally in the Unified Verifier subsection of \Cref{ch:methodology} (table of signal definitions). | ✅ Matched | Matches the paper's definition of confabulations. |
| 5 | `chapters/chapter_2.tex:25` | The normalised semantic entropy signal of \cite{farquhar2024semantic} was registered earlier in the project alongside the same family but retired before the main run because its cycle-zero per-benchmark Pearson correlation with exact-match labels was indistinguishable from zero on every per-benchmark slice of the calibration fold; the kernel-language-entropy upgrade of \cite{nikitin2024kle} was registered as a candidate replacement and is also retired in the shipped architecture.} \textit{Ungrounded factual claims} are answers without support in retrieved evidence, detected by the external-grounding family of three signals ($p_{\text{ground\_max}}$, $p_{\text{ground\_mean}}$, and the length-gated weakest-link atomic-fact entailment $p_{\text{ground\_atomic}}$ that runs only on hypotheses whose token length exceeds the registered atomic-decomposition gate). | ✅ Matched | Attribution of the semantic-entropy signal is correct. |
| 6 | `chapters/chapter_2.tex:60` | The normalised semantic entropy signal of \cite{farquhar2024semantic} was registered earlier in the project as a third sample-set member computed over $K{=}10$ chains at temperature $T{=}1.0$, and the kernel-language-entropy generalisation of \cite{nikitin2024kle} was registered as a candidate replacement that uses graded semantic-similarity kernels rather than hard cluster assignments; both were retired before the main run on the cycle-zero calibration evidence that the entropy signal contributed no discriminative information beyond what $s_{\text{avg}}$ and $p_{\text{entail}}$ on the same resampled pool already provided, so the deployed verifier consumes nine signals across the four families, of which the sample-set agreement family contributes two. | ✅ Matched | Matches: about 10 samples at temperature 1.0. |
| 7 | `chapters/chapter_2.tex:108` | Another important method is semantic entropy introduced bt Farquhar et al. \cite{farquhar2024semantic} in 2024. | ✅ Matched | Matches: semantic entropy, Farquhar et al. 2024. The thesis has a typo: "introduced bt" should be "introduced by". |
| 8 | `chapters/chapter_2.tex:150` | Nikitin et al.~\cite{nikitin2024kle} introduced Kernel Language Entropy at NeurIPS 2024, generalising Farquhar et al.'s hard-cluster semantic entropy to a soft-kernel von Neumann entropy over pairwise semantic similarities; KLE is registered as a future-work candidate should CAEM re-introduce the semantic-entropy family in a later iteration (the corresponding $h_{\text{norm}}$ signal of \cite{farquhar2024semantic} was retired before the main run on cycle-zero evidence that it contributed no discriminative information beyond $s_{\text{avg}}$ and $p_{\text{entail}}$ on the same resampled pool), but it is not adopted in the locked configuration because the empirical AUROC gain is bounded above the cost of the additional kernel computation. | ✅ Matched | Matches. |
| 9 | `chapters/chapter_2.tex:269` | \textbf{Semantic entropy}~\cite{farquhar2024semantic} computes a model-internal uncertainty score by sampling multiple generations under temperature and clustering them into semantically-equivalent groups, then reports the entropy of the cluster distribution as the per-query uncertainty signal; the baseline runs the same backbone as the zero-shot system and reports the same answer, but exposes an uncertainty surface that is the most direct external comparator to CAEM's multi-signal calibrated probability composite. | ✅ Matched | Matches: sampling, semantic clustering and entropy over clusters. |
| 10 | `chapters/chapter_2.tex:283` | For CAEM, TruthfulQA is the benchmark where the sample-set family of signals is expected to pay off: misconceptions typically produce high semantic entropy because the model generates both the correct answer and the misconception with similar confidence across stochastic decodes, and the deployed sample-set agreement signals $s_{\text{avg}}$ and $p_{\text{entail}}$, the surviving proxy for the semantic-entropy principle~\cite{farquhar2024semantic}, are designed to catch this. | ❌ Not matched | Farquhar et al. explicitly say semantic entropy does *not* catch errors the model makes consistently (e.g. learned misconceptions). That contradicts "misconceptions typically produce high semantic entropy". |
| 11 | `chapters/chapter_2.tex:297` | The normalised semantic entropy of \cite{farquhar2024semantic, kuhn2023semantic} and the kernel-language-entropy generalisation of \cite{nikitin2024kle} were registered earlier in the project as candidate sample-set members but retired before the main run on cycle-zero evidence that they contributed no discriminative information beyond what $s_{\text{avg}}$ and $p_{\text{entail}}$ already provide on the same resampled pool. | ✅ Matched | Attribution is correct. |
| 12 | `chapters/chapter_4.tex:285` | The normalised semantic entropy signal of \cite{farquhar2024semantic,kuhn2023semantic}, computed over $K=10$ chains at temperature $T=1.0$, registered a Pearson correlation indistinguishable from zero with exact-match labels on every per-benchmark slice of the calibration fold and a boost-layer weight within five times ten to the negative four of zero on every fit. | ✅ Matched | Attribution is correct. |
| 13 | `chapters/chapter_5.tex:456` | The semantic-entropy abstention baseline \cite{farquhar2024semantic} computes a per-query uncertainty score by sampling ten generations at temperature one from the same backbone, clustering the samples by normalised-exact-match semantic equivalence on the extracted answer, and reading the entropy of the cluster distribution as the uncertainty signal. | ⚠️ Partly | The sampling and entropy setup matches. The paper clusters by bidirectional entailment, not normalised exact match. Say that this is your variant. |

## [18] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
`NEURIPS2020_6b493230` · Lewis et al., 2020 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** RAG (NeurIPS 2020). Combines a parametric seq2seq model with a dense Wikipedia index via DPR. State of the art on open-domain QA (NQ EM 44.5 for RAG-Sequence vs. 41.5 for DPR). The index can be swapped to update knowledge without retraining.

**Cited 6 times. Matched 6/6 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:49` | Retrieval-augmented generation grounds the model in retrieved evidence \cite{NEURIPS2020_6b493230}, which raises factual recall when the retrieval succeeds but is silent when retrieval misses or returns evidence the model summarises incorrectly. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:166` | Lewis et al. \cite{NEURIPS2020_6b493230} introduced Retrieval-Augmented Generation (RAG) at NeurIPS 2020. | ✅ Matched | Matches: RAG, NeurIPS 2020. |
| 3 | `chapters/chapter_2.tex:267` | \textbf{DPR-RAG}~\cite{karpukhin-etal-2020-dense, NEURIPS2020_6b493230} retrieves the top $k=5$ Wikipedia passages from a dense DPR index, concatenates them into a 384-token context, and generates from the same Qwen-2.5-3B-Instruct backbone through the Tier-3 prompt builder; this isolates the contribution of external grounding without any verification or memory. | ✅ Matched | Attribution is correct. |
| 4 | `chapters/chapter_2.tex:281` | NQ is the dominant open-domain QA benchmark in recent retrieval-augmented-generation literature~\cite{NEURIPS2020_6b493230, karpukhin-etal-2020-dense}. | ✅ Matched | Matches: NQ is central in RAG and DPR. |
| 5 | `chapters/chapter_2.tex:303` | Neural Turing Machines \cite{graves2014neural}, Memory Networks \cite{weston2015memory}, and DNCs \cite{graves2016differentiable} demonstrate that differentiable memory enables structured reasoning; RAG \cite{NEURIPS2020_6b493230} shows that non-parametric knowledge improves factual accuracy (44.5\% vs 41.5\% on Natural Questions) and supports index updates without retraining; FAISS \cite{8733051} and Sentence-BERT \cite{reimers-gurevych-2019-sentence} make similarity search cheap enough to run in real time. | ✅ Matched | Matches: NQ 44.5 vs. 41.5, and index hot-swapping. |
| 6 | `chapters/chapter_5.tex:431` | The retrieval-augmented baseline retrieves the top-$k = 5$ Wikipedia passages from the dense passage index used by CAEM's Tier 3 \cite{karpukhin-etal-2020-dense,NEURIPS2020_6b493230}, concatenates them into a 384-token context, and generates from the same backbone through the Tier-3 prompt template. | ✅ Matched | Attribution is correct. |

## [19] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
`wei2022chain` · Wei et al., 2022 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2022. Few-shot exemplars with intermediate reasoning steps (chain of thought) substantially improve reasoning in sufficiently large models (an emergent ability at about 100B parameters). The zero-shot "Let's think step by step" trigger is from Kojima et al. 2022, not this paper.

**Cited 5 times. Matched 3/5 = 60%** (partly 2, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:49` | Prompting interventions, including chain-of-thought reasoning and self-consistency over sampled chains \cite{wei2022chain, wang2023selfconsistency}, expose internal reasoning and exploit agreement across resampled answers, but they provide no persistence: each session begins with no memory of prior verified answers, and the same question asked tomorrow triggers the same reasoning effort as today. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:100` | Chain of thought prompting, which was introduced by Wei et al. \cite{wei2022chain} at NeurIPS 2022, improves reasoning quality by prompting models to generate the next steps before final answers. | ✅ Matched | Matches: CoT, NeurIPS 2022. |
| 3 | `chapters/chapter_2.tex:265` | The second, \textbf{Chain-of-Thought} prompting~\cite{wei2022chain}, adds the standard ``Let's think step by step'' trigger without any further modification. | ⚠️ Partly | Wei et al. use few-shot CoT exemplars. The "Let's think step by step" trigger is Kojima et al. 2022 (*Large Language Models are Zero-Shot Reasoners*). |
| 4 | `chapters/chapter_2.tex:297` | Chain-of-thought prompting \cite{wei2022chain} produces the intermediate claims that downstream verification operates on. | ✅ Matched | Matches. |
| 5 | `chapters/chapter_5.tex:426` | The chain-of-thought baseline \cite{wei2022chain} adds the standard step-by-step trigger to the zero-shot prompt without any further modification. | ⚠️ Partly | Same: the step-by-step trigger is Kojima et al. 2022. |

## [20] Self-Consistency Improves Chain of Thought Reasoning in Language Models
`wang2023selfconsistency` · Wang et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICLR 2023. Self-consistency samples multiple diverse reasoning paths and selects the most consistent answer by majority vote. This improves CoT on arithmetic and commonsense reasoning.

**Cited 5 times. Matched 3/5 = 60%** (partly 2, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:49` | Prompting interventions, including chain-of-thought reasoning and self-consistency over sampled chains \cite{wei2022chain, wang2023selfconsistency}, expose internal reasoning and exploit agreement across resampled answers, but they provide no persistence: each session begins with no memory of prior verified answers, and the same question asked tomorrow triggers the same reasoning effort as today. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:54` | Self consistency \cite{wang2023selfconsistency} improves reasoning quality by first generating multiple independent reasoning chains for the same question, then it selects the most common answer. | ✅ Matched | Matches: multiple chains with majority vote. |
| 3 | `chapters/chapter_2.tex:104` | Wang et al. \cite{wang2023selfconsistency} expanded this idea in 2023 by introducing self consistency. | ✅ Matched | Matches. |
| 4 | `chapters/chapter_2.tex:297` | Self-consistency \cite{wang2023selfconsistency} provides the sample-set agreement signal $s_{\text{avg}}$. | ⚠️ Partly | Self-consistency is a majority vote. The cosine-similarity signal s_avg is your adaptation; say "inspired by". |
| 5 | `chapters/chapter_4.tex:283` | The semantic-consistency signal is the mean unique-pair cosine similarity across $M=3$ chains at temperature $T=0.7$ in the sentence-embedding space, with higher values reading as higher agreement \cite{wang2023selfconsistency}. | ⚠️ Partly | Same: the paper does not use mean pairwise cosine similarity. |

## [21] SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models
`manakul-etal-2023-selfcheckgpt` · Manakul et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** EMNLP 2023. SelfCheckGPT is zero-resource, black-box hallucination detection that checks consistency among sampled responses.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:49` | Post-generation verification \cite{manakul-etal-2023-selfcheckgpt, farquhar2024semantic} detects hallucinations after they occur and supports refusal or revision, but it does not feed back into model weights or memory, so the same hallucination patterns recur in the next session. | ✅ Matched | Matches. |

## [22] HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering
`yang-etal-2018-hotpotqa` · Yang et al., 2018 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** EMNLP 2018. HotpotQA: about 113k Wikipedia-based multi-hop QA pairs with supporting-fact annotations.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_1.tex:64` | Three benchmarks registered earlier in the project (HotpotQA~\cite{yang-etal-2018-hotpotqa}, Natural Questions, and ARC-Challenge) were retired before the main run on cycle-zero diagnostic evidence that they violated the verifier-balanced-accuracy precondition for self-improvement; all three are kept on disk as registered exclusion evidence rather than carried forward into the live panel. | ✅ Matched | Usage; the attribution is correct. |

## [23] Attention is All you Need
`NIPS2017_3f5ee243` · Vaswani et al., 2017 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2017. The Transformer: an architecture based solely on (self-)attention, dispensing with recurrence.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:10` | The Transformer architecture \cite{NIPS2017_3f5ee243}, which processes sequences using a self-attention mechanism allowing all the tokens in the input to take account of every other token, is adopted by new LLMs. | ✅ Matched | Matches. |

## [24] Qwen2.5: A Party of Foundation Models
`qwen25` · Qwen Team, 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Qwen2.5 technical report (2024). A family of open-weight models (0.5B–72B, base and instruct) with large-scale pretraining and post-training (SFT plus RL).

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:12` | In this study, we utilize Qwen-2.5-3B-Instruct \cite{qwen25}, a three-billion-parameter decoder-only transformer that has already undergone instruction-tuning on a broad mixture of task-formatted demonstrations before reaching the CAEM pipeline. | ✅ Matched | Matches: Qwen2.5 instruct model. |
| 2 | `chapters/chapter_2.tex:100` | Since CAEM uses Qwen-2.5-3B-Instruct~\cite{qwen25}, which is a decoder-only instruction-tuned model at comparable scale, chain-of-thought prompting is effective despite being well below the 100B base-model threshold. | ✅ Matched | Attribution of the model only. |
| 3 | `chapters/chapter_4.tex:62` | The generator is a three-billion-parameter open-weight instruction-tuned decoder-only model from the Qwen 2.5 family \cite{qwen25}, used for all generation in the zero-shot tier and the retrieval-augmented tier and used as the source of internal-calibration signals in the verifier. | ✅ Matched | Matches. |

## [25] Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities
`nikitin2024kle` · Nikitin et al., 2024 · details matched: ⚠️ Partly

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2024. Kernel Language Entropy: von Neumann entropy of a semantic-similarity kernel over sampled answers. It generalises semantic entropy using soft, fine-grained similarities instead of hard clusters.

**Cited 5 times. Matched 5/5 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:25` | The normalised semantic entropy signal of \cite{farquhar2024semantic} was registered earlier in the project alongside the same family but retired before the main run because its cycle-zero per-benchmark Pearson correlation with exact-match labels was indistinguishable from zero on every per-benchmark slice of the calibration fold; the kernel-language-entropy upgrade of \cite{nikitin2024kle} was registered as a candidate replacement and is also retired in the shipped architecture.} \textit{Ungrounded factual claims} are answers without support in retrieved evidence, detected by the external-grounding family of three signals ($p_{\text{ground\_max}}$, $p_{\text{ground\_mean}}$, and the length-gated weakest-link atomic-fact entailment $p_{\text{ground\_atomic}}$ that runs only on hypotheses whose token length exceeds the registered atomic-decomposition gate). | ✅ Matched | Attribution is correct. |
| 2 | `chapters/chapter_2.tex:60` | The normalised semantic entropy signal of \cite{farquhar2024semantic} was registered earlier in the project as a third sample-set member computed over $K{=}10$ chains at temperature $T{=}1.0$, and the kernel-language-entropy generalisation of \cite{nikitin2024kle} was registered as a candidate replacement that uses graded semantic-similarity kernels rather than hard cluster assignments; both were retired before the main run on the cycle-zero calibration evidence that the entropy signal contributed no discriminative information beyond what $s_{\text{avg}}$ and $p_{\text{entail}}$ on the same resampled pool already provided, so the deployed verifier consumes nine signals across the four families, of which the sample-set agreement family contributes two. | ✅ Matched | Matches: soft kernels instead of hard clusters. |
| 3 | `chapters/chapter_2.tex:150` | Nikitin et al.~\cite{nikitin2024kle} introduced Kernel Language Entropy at NeurIPS 2024, generalising Farquhar et al.'s hard-cluster semantic entropy to a soft-kernel von Neumann entropy over pairwise semantic similarities; KLE is registered as a future-work candidate should CAEM re-introduce the semantic-entropy family in a later iteration (the corresponding $h_{\text{norm}}$ signal of \cite{farquhar2024semantic} was retired before the main run on cycle-zero evidence that it contributed no discriminative information beyond $s_{\text{avg}}$ and $p_{\text{entail}}$ on the same resampled pool), but it is not adopted in the locked configuration because the empirical AUROC gain is bounded above the cost of the additional kernel computation. | ✅ Matched | Matches: NeurIPS 2024, von Neumann entropy, generalises semantic entropy. |
| 4 | `chapters/chapter_2.tex:297` | The normalised semantic entropy of \cite{farquhar2024semantic, kuhn2023semantic} and the kernel-language-entropy generalisation of \cite{nikitin2024kle} were registered earlier in the project as candidate sample-set members but retired before the main run on cycle-zero evidence that they contributed no discriminative information beyond what $s_{\text{avg}}$ and $p_{\text{entail}}$ already provide on the same resampled pool. | ✅ Matched | Attribution is correct. |
| 5 | `chapters/chapter_4.tex:285` | The kernel-language-entropy generalisation that uses graded semantic-similarity kernels rather than hard cluster assignments \cite{nikitin2024kle} was registered as a candidate upgrade for the signal but was not adopted in the shipped configuration. | ✅ Matched | Matches. |

## [26] MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents
`tang2024minicheck` · Tang et al., 2024 · details matched: ⚠️ Partly

**What the paper says (paraphrased from its abstract and public record):** EMNLP 2024. MiniCheck builds small fact-checkers (e.g. MiniCheck-FT5, from Flan-T5-Large) trained on synthetic claim–document data generated with GPT-4. It outputs whether a claim is supported by the grounding document, reaching GPT-4-level accuracy on LLM-AggreFact at much lower cost.

**Cited 5 times. Matched 4/5 = 80%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:25` | The deployed configuration ships MiniCheck~\cite{tang2024minicheck} as the entailment backend across all hypothesis lengths because the long-hypothesis Qwen judge fails the registered Pearson floor on the registered 500-sample calibration overlap fold and is ablated in the locked configuration; the contradiction probability $p_{\text{contra}}$ is structurally zero under MiniCheck (the unary supported-against-context output is the only label class the judge produces), the corresponding hard-veto branch was retired on 2026-04-22, and the contradiction-scoring call was further short-circuited at the verifier compute path so the signal does not enter the gradient or the composite log-odds at all; contradicted factual claims are caught indirectly through the calibrated grounding signals' contribution to the composite. | ✅ Matched | Matches: MiniCheck outputs a supported/unsupported probability with no contradiction class. |
| 2 | `chapters/chapter_2.tex:38` | As the verifier judge in CAEM, we employ MiniCheck-Flan-T5-Large \cite{tang2024minicheck}, a Flan-T5-Large checkpoint fine-tuned on synthetically decomposed language-model-generated claim-support data. | ✅ Matched | Matches: MiniCheck-FT5 (Flan-T5-Large) trained on synthetic LLM-generated claim–document data. |
| 3 | `chapters/chapter_2.tex:253` | [table row] MiniCheck~\cite{tang2024minicheck} & no & --- & NLI judge (fact verification) & --- & --- | ✅ Matched | Table row matches. |
| 4 | `chapters/chapter_2.tex:297` | The thesis uses MiniCheck-Flan-T5-Large \cite{tang2024minicheck} as the deployed entailment backend across all hypothesis lengths because the registered long-hypothesis Qwen judge fails the pre-registered Pearson floor on the registered 500-sample calibration overlap fold and is ablated in the locked configuration; MiniCheck is fine-tuned specifically on language-model-generated claim-support data and produces well-calibrated supported-versus-unsupported probabilities on the distribution of claims the verifier actually scores, whereas generic NLI models trained on MultiNLI and SNLI (including RoBERTa-Large-MNLI~\cite{liu2019roberta}) exhibit systematic miscalibration on language-model hallucinations: \cite{semanticillusion2025} report a one-hundred-percent false-positive rate at ninety-five-percent recall for a near-state-of-the-art generic NLI model on HaluEval~\cite{li-etal-2023-halueval}, corresponding to a verifier balanced accuracy at or below the chance threshold, which would falsify the purity theorem's empirical premise. | ⚠️ Partly | The training data matches. "Well-calibrated probabilities" is not a claim the paper makes (it reports balanced accuracy). |
| 5 | `chapters/chapter_4.tex:66` | The natural-language-inference judge that the verifier consumes for chain-to-answer entailment and for the external grounding signals is realised by the MiniCheck checkpoint \cite{tang2024minicheck}, a unary supported-against-context scorer trained on language-model-generated claim-support data and therefore distributionally matched to the kind of generated answers the verifier is asked to score. | ✅ Matched | Matches. |

## [27] The PASCAL Recognising Textual Entailment Challenge
`10.1007/11736790_9` · Dagan et al., 2006 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** First PASCAL RTE challenge (2006). Defines textual entailment: T entails H if, typically, a human reading T would infer that H is most likely true.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:29` | Natural Language Inference (NLI) is the task of determining whether a hypothesis sentence logically follows from a premise sentence \cite{10.1007/11736790_9}. | ✅ Matched | Matches the definition of textual entailment. |

## [28] RoBERTa: A Robustly Optimized BERT Pretraining Approach
`liu2019roberta` · Liu et al., 2019 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** RoBERTa (arXiv 2019). A robustly optimised BERT pretraining recipe. GLUE results include MNLI-m dev 90.2 for single-task large and 90.8 on the test set.

**Cited 2 times. Matched 1/2 = 50%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:38` | The legacy generic-NLI baseline RoBERTa-Large-MNLI \cite{liu2019roberta}, fine-tuned on Multi-Genre Natural Language Inference and reaching approximately 90.8\% on the MNLI matched development set, is retained as a Chapter 5 ablation target (the generic-NLI-backend ablation variant registered in \Cref{sec:comp-ablations}) so the backend choice is defended by trajectory evidence rather than by citation alone. | ⚠️ Partly | RoBERTa reports MNLI-m **dev** 90.2. 90.8 is the test-set figure. |
| 2 | `chapters/chapter_2.tex:297` | The thesis uses MiniCheck-Flan-T5-Large \cite{tang2024minicheck} as the deployed entailment backend across all hypothesis lengths because the registered long-hypothesis Qwen judge fails the pre-registered Pearson floor on the registered 500-sample calibration overlap fold and is ablated in the locked configuration; MiniCheck is fine-tuned specifically on language-model-generated claim-support data and produces well-calibrated supported-versus-unsupported probabilities on the distribution of claims the verifier actually scores, whereas generic NLI models trained on MultiNLI and SNLI (including RoBERTa-Large-MNLI~\cite{liu2019roberta}) exhibit systematic miscalibration on language-model hallucinations: \cite{semanticillusion2025} report a one-hundred-percent false-positive rate at ninety-five-percent recall for a near-state-of-the-art generic NLI model on HaluEval~\cite{li-etal-2023-halueval}, corresponding to a verifier balanced accuracy at or below the chance threshold, which would falsify the purity theorem's empirical premise. | ✅ Matched | Attribution of the RoBERTa model is correct. |

## [29] Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
`reimers-gurevych-2019-sentence` · Reimers & Gurevych, 2019 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** EMNLP 2019. Sentence-BERT fine-tunes BERT in siamese/triplet networks, mainly on SNLI+MNLI and then STSb, to produce sentence embeddings comparable by cosine similarity.

**Cited 4 times. Matched 3/4 = 75%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:42` | Sentence BERT (SBERT) \cite{reimers-gurevych-2019-sentence} produces encasings or embeddings by fine tuning BERT in a siamese network structure on semantic textual similarity tasks. | ⚠️ Partly | Siamese BERT fine-tuning matches, but training was mainly on NLI data (SNLI+MNLI) and then STS, not only STS. Also "encasings" is a typo for "encodings". |
| 2 | `chapters/chapter_2.tex:174` | Reimers and Gurevych \cite{reimers-gurevych-2019-sentence} introduced Sentence-BERT (SBERT) at EMNLP 2019. | ✅ Matched | Matches. |
| 3 | `chapters/chapter_2.tex:303` | Neural Turing Machines \cite{graves2014neural}, Memory Networks \cite{weston2015memory}, and DNCs \cite{graves2016differentiable} demonstrate that differentiable memory enables structured reasoning; RAG \cite{NEURIPS2020_6b493230} shows that non-parametric knowledge improves factual accuracy (44.5\% vs 41.5\% on Natural Questions) and supports index updates without retraining; FAISS \cite{8733051} and Sentence-BERT \cite{reimers-gurevych-2019-sentence} make similarity search cheap enough to run in real time. | ✅ Matched | Matches. |
| 4 | `chapters/chapter_4.tex:64` | The encoder is a sentence-transformer model of moderate dimension \cite{reimers-gurevych-2019-sentence} that is also used to encode the user's query at lookup time, so query and passage embeddings live in the same space. | ✅ Matched | Attribution is correct. |

## [30] Billion-Scale Similarity Search with GPUs
`8733051` · Johnson et al., 2021 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** FAISS (IEEE Trans. Big Data; arXiv 2017). GPU-based billion-scale similarity search.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:50` | The system uses FAISS (Facebook AI Similarity Search) \cite{8733051} for fast, efficient approximate nearest-neighbour search, allowing it to retrieve sub-millisecond results over thousands of stored episodes. | ✅ Matched | Attribution is correct. The "sub-millisecond" figure is your own measurement. |
| 2 | `chapters/chapter_2.tex:172` | Johnson et al.\ \cite{8733051} developed FAISS to handle similarity search at the scale of billions of items in 2019, enabling efficient implementation of memory-augmented systems. | ✅ Matched | Matches: billion-scale similarity search. |
| 3 | `chapters/chapter_2.tex:303` | Neural Turing Machines \cite{graves2014neural}, Memory Networks \cite{weston2015memory}, and DNCs \cite{graves2016differentiable} demonstrate that differentiable memory enables structured reasoning; RAG \cite{NEURIPS2020_6b493230} shows that non-parametric knowledge improves factual accuracy (44.5\% vs 41.5\% on Natural Questions) and supports index updates without retraining; FAISS \cite{8733051} and Sentence-BERT \cite{reimers-gurevych-2019-sentence} make similarity search cheap enough to run in real time. | ✅ Matched | Matches. |

## [31] Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem
`MCCLOSKEY1989109` · McCloskey & Cohen, 1989 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** 1989. Shows catastrophic interference: training connectionist networks sequentially on new items severely disrupts previously learned items.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:64` | Catastrophic forgetting \cite{MCCLOSKEY1989109} occurs when neural networks learn new tasks and suddenly lose their ability to perform previously learned tasks. | ✅ Matched | Matches. |

## [32] Hallucination Rates and Reference Accuracy of ChatGPT and Bard for Systematic Reviews: Comparative Analysis
`info:doi/10.2196/53164` · Chelli et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** JMIR 2024. Measures hallucinated references when LLMs are used for systematic-review literature searches. Rates were GPT-3.5 39.6%, GPT-4 28.6% and Bard 91.4%; precision and recall were also low.

**Cited 2 times. Matched 1/2 = 50%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:90` | Cheli et al.\ \cite{info:doi/10.2196/53164} studied hallucination in medical-literature queries and reported rates of 69.6\% for GPT-3.5, 28.6\% for GPT-4, and 91.4\% for Google Bard. | ⚠️ Partly | GPT-4 28.6% and Bard 91.4% match. GPT-3.5 is **39.6%**, not 69.6%. Also "Cheli" should be "Chelli". |
| 2 | `chapters/chapter_3.tex:94` | In domains where fluency without grounding has produced documented harm, including health-information retrieval~\cite{alkaissi2023artificial,info:doi/10.2196/53164}, legal triage, and educational tutoring~\cite{ji2023survey}, an architecture that abstains rather than fabricates is a strict improvement on the deployed alternative. | ✅ Matched | Matches: health-information harm. |

## [33] Scaling Instruction-Finetuned Language Models
`JMLR:v25:23-0870` · Chung et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Flan instruction finetuning (JMLR 2024). Scaling tasks and model size plus CoT data improves performance. Flan-T5 checkpoints range from 80M to 11B parameters.

**Cited 1 time. Matched 0/1 = 0%** (partly 0, not matched 0, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:100` | Chung et al.\ \cite{JMLR:v25:23-0870} demonstrated that Flan-T5 models show substantial chain-of-thought benefits at 780 million parameters after instruction tuning, confirming that the threshold does not apply to instruction-tuned backbones. | 🔒 Needs full text | Whether Flan-T5 at 780M shows substantial CoT benefits needs the full text. |

## [34] FEVER: a Large-scale Dataset for Fact Extraction and VERification
`thorne-etal-2018-fever` · Thorne et al., 2018 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NAACL 2018. FEVER: 185,445 claims generated by altering Wikipedia sentences, labelled Supported, Refuted or NotEnoughInfo with evidence.

**Cited 5 times. Matched 5/5 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:114` | Thorne et al. \cite{thorne-etal-2018-fever} introduced the FEVER dataset at NAACL 2018 which formulates fact verification as NLI with 185,445 claims generated from Wikipedia. | ✅ Matched | Matches: NAACL 2018, 185,445 claims. |
| 2 | `chapters/chapter_2.tex:277` | Thorne et al.~\cite{thorne-etal-2018-fever} created the FEVER dataset at NAACL 2018 for fact verification, providing 185{,}445 claims generated from Wikipedia requiring classification as SUPPORTS, REFUTES, or NOT ENOUGH INFO relative to Wikipedia evidence. | ✅ Matched | Matches: three labels. |
| 3 | `chapters/chapter_2.tex:307` | FEVER \cite{thorne-etal-2018-fever} tests the external-grounding family and the three-way mapping from SUPPORTS, REFUTES, or NOT ENOUGH INFO onto \textsc{Store}, \textsc{Discard}, or \textsc{Abstain} respectively. | ✅ Matched | Usage; the attribution is correct. |
| 4 | `chapters/chapter_3.tex:145` | The five evaluation benchmarks are FEVER \cite{thorne-etal-2018-fever}, TriviaQA \cite{joshi-etal-2017-triviaqa}, CommonsenseQA \cite{talmor-etal-2019-commonsenseqa}, TruthfulQA \cite{lin-etal-2022-truthfulqa}, and StrategyQA \cite{geva2021strategyqa}, all of which are distributed under licences that permit research use and redistribution under attribution; the licence terms are catalogued in Appendix~\ref{app:reproducibility} together with the access date and the dataset version hash. | ✅ Matched | Usage; the attribution is correct. |
| 5 | `chapters/chapter_5.tex:41` | The training panel comprises FEVER \cite{thorne-etal-2018-fever} for factual claim verification, TriviaQA \cite{joshi-etal-2017-triviaqa} for open-domain trivia, and CommonsenseQA \cite{talmor-etal-2019-commonsenseqa} for five-option multiple-choice commonsense reasoning. | ✅ Matched | Usage; the attribution is correct. |

## [35] Mitigating LLM Hallucinations via Conformal Abstention
`yadkori2024abstention` · Yadkori et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** arXiv 2024. Uses conformal prediction / conformal risk control so that an LLM abstains, with a rigorous bound on the hallucination (error) rate, using self-consistency-based scores.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:118` | Yadkori et al.~\cite{yadkori2024abstention} extended this with conformal-risk-control to bound the hallucination rate at a user-specified target through learned abstention rather than claim removal, providing the direct precedent for CAEM's abstention class in the four-outcome decision tree. | ✅ Matched | Matches: conformal risk control and abstention with a bounded hallucination rate. |
| 2 | `chapters/chapter_4.tex:374` | Second, a binary rule does not separate the failure mode where the system has nothing to say from the failure mode where the system would say something but cannot ground it; the abstention outcome captures the explicit-refusal case as a first-class user-facing action rather than a silent discard, in the spirit of the conformal-abstention framework that bounds hallucination rate at a user-specified target \cite{yadkori2024abstention}. | ✅ Matched | Matches. |

## [36] Conformal Language Modeling
`quach2024conformal` · Quach et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICLR 2024. Conformal Language Modeling: calibrated stopping and rejection rules for sampling sets of generations with coverage guarantees via Learn-then-Test. Also identifies individually confident components (phrases).

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:118` | Quach et al.~\cite{quach2024conformal} present \emph{Conformal Language Modeling} at ICLR 2024, calibrating a stopping rule, a rejection rule, and a confidence rule simultaneously through Learn-then-Test multiple-hypothesis testing. | ✅ Matched | Mostly matches: calibrated stopping and rejection rules with LTT. The third element is component-level confidence. |

## [37] Predicting Good Probabilities With Supervised Learning
`niculescu2005predicting` · Niculescu-Mizil & Caruana, 2005 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICML 2005. Studies calibration of supervised learners. Compares Platt scaling and isotonic regression as post-hoc calibration methods.

**Cited 5 times. Matched 3/5 = 60%** (partly 2, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:118` | The deployed gate is therefore a fixed-threshold rule on the calibrated probability, in the lineage of post-hoc calibration analyses by \cite{niculescu2005predicting} and the selective-classification framing of \cite{geifman2017selective}. | ✅ Matched | Attribution is fair. |
| 2 | `chapters/chapter_2.tex:120` | The composite is not a fixed weighted sum: each signal is mapped to a calibrated $\Pr(\text{em}=1\mid\text{signal})$ through per-signal isotonic regression with auto-detected direction~\cite{niculescu2005predicting}, the per-signal curves are blended with a per-benchmark child fit through a James-Stein style shrinkage prior, and the per-signal log-odds are aggregated through an $L_{2}$-regularised logistic boost layer in the lineage of Cherian-Gibbs-Cand\`{e}s conditional boosting~\cite{cherian2024conformal}. | ✅ Matched | Isotonic regression for calibration matches. "Auto-detected direction" is your addition. |
| 3 | `chapters/chapter_4.tex:333` | Isotonic regression auto-detects the monotone direction from the Pearson correlation between the raw signal and the binary label, so a signal that reads high-confidence-as-low-value (such as the dropout-disagreement signal, where higher variance reads as lower confidence) is calibrated correctly without a registry of per-signal directions \cite{niculescu2005predicting}. | ⚠️ Partly | The paper does not auto-detect monotone direction. That is your design. |
| 4 | `chapters/chapter_4.tex:348` | The fixed-threshold replacement is preferred over the split-conformal alternative on two grounds: it is robust to the cal-versus-eval shift that the conformal premise requires, in the lineage of post-hoc calibration analyses by \cite{niculescu2005predicting} and the selective-classification framing of \cite{geifman2017selective}; and it commits the gate's operating point to a single registered scalar that any reader can locate in the configuration without reconstructing the conformal fit. | ✅ Matched | Attribution is fair. |
| 5 | `chapters/chapter_4.tex:507` | The first layer treats the fixed-threshold storage gate as a calibration-consistent bound under the exchangeability assumption between the calibration fold and the cycle-$t$ population: the empirical precision of the calibration fold at the locked threshold projects forward as a one-sided lower bound on the cycle-$t$ pool's precision, supported by the post-hoc calibration analysis of \cite{niculescu2005predicting} and the selective-classification framing of \cite{geifman2017selective}. | ⚠️ Partly | Niculescu-Mizil & Caruana give no precision lower bound under exchangeability. That part is supported by [38], not this paper. |

## [38] Selective Classification for Deep Neural Networks
`geifman2017selective` · Geifman & El-Yaniv, 2017 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2017. Selective classification for deep networks: a reject option with a guaranteed risk level at high probability (the SGR algorithm), giving a risk–coverage trade-off.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:118` | The deployed gate is therefore a fixed-threshold rule on the calibrated probability, in the lineage of post-hoc calibration analyses by \cite{niculescu2005predicting} and the selective-classification framing of \cite{geifman2017selective}. | ✅ Matched | Attribution is fair. |
| 2 | `chapters/chapter_4.tex:348` | The fixed-threshold replacement is preferred over the split-conformal alternative on two grounds: it is robust to the cal-versus-eval shift that the conformal premise requires, in the lineage of post-hoc calibration analyses by \cite{niculescu2005predicting} and the selective-classification framing of \cite{geifman2017selective}; and it commits the gate's operating point to a single registered scalar that any reader can locate in the configuration without reconstructing the conformal fit. | ✅ Matched | Attribution is fair. |
| 3 | `chapters/chapter_4.tex:507` | The first layer treats the fixed-threshold storage gate as a calibration-consistent bound under the exchangeability assumption between the calibration fold and the cycle-$t$ population: the empirical precision of the calibration fold at the locked threshold projects forward as a one-sided lower bound on the cycle-$t$ pool's precision, supported by the post-hoc calibration analysis of \cite{niculescu2005predicting} and the selective-classification framing of \cite{geifman2017selective}. | ✅ Matched | Matches: SGR gives a high-probability bound on selective risk. |

## [39] Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
`pmlr-v48-gal16` · Gal & Ghahramani, 2016 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICML 2016. Dropout training is approximate Bayesian inference in deep Gaussian processes. MC dropout gives model uncertainty.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:128` | Gal and Ghahramani \cite{pmlr-v48-gal16} laid the theoretical foundations for practical uncertainty estimation in their ICML 2016 paper "Dropout as a Bayesian Approximation." Their key contribution explains that dropout usually during training can be understood as approximate Bayesian inference over model parameters. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:146` | The second stage is the nine-signal post-generation verifier, which runs only after Tier-2 or Tier-3 generation and computes three internal-calibration signals (the mean token log-probability $u_{\text{token}}$, the MC-Dropout variance $u_{\text{dropout}}$~\cite{pmlr-v48-gal16}, and the derived internal-state composite $u_{\text{internal}} = 0.5\,u_{\text{token}} + 0.5\,(1-u_{\text{dropout}})$), two sample-set agreement signals ($s_{\text{avg}}$, $p_{\text{entail}}$ on the same $M{=}3$ resampled pool), three external-grounding signals ($p_{\text{ground\_max}}$, $p_{\text{ground\_mean}}$, $p_{\text{ground\_atomic}}$), and Branch~C's question-answer relevance signal $q_{\text{a,rel}}$ that guards against off-topic generations. | ✅ Matched | Matches: MC dropout. |
| 3 | `chapters/chapter_4.tex:278` | The dropout-disagreement signal is the variance of the answer's log-probability across the registered $K=5$ Monte-Carlo dropout passes through the generator with dropout layers active, with lower variance reading as higher confidence; this is the Bayesian-approximation interpretation of dropout introduced by \cite{pmlr-v48-gal16}, and it captures the stability of the generator's hidden state under stochastic ablation. | ✅ Matched | Matches. |

## [40] Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback
`tian2023just` · Tian et al., 2023 · details matched: ❌ No

**What the paper says (paraphrased from its abstract and public record):** EMNLP 2023. For RLHF-tuned LMs (ChatGPT, GPT-4, Claude), verbalized confidence is often better calibrated than conditional token probabilities. On TriviaQA, SciQ and TruthfulQA it often halves ECE.

**Cited 1 time. Matched 0/1 = 0%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:148` | Recent work by Tian et al. \cite{tian2023just} questions if models genuinely know when they don't know. | ⚠️ Partly | The paper studies eliciting calibrated verbalized confidence from RLHF models. "Questions if models genuinely know when they don't know" is a loose paraphrase. |

## [41] A Survey of Confidence Estimation and Calibration in Large Language Models
`geng2024survey` · Geng et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NAACL 2024 survey. Organises confidence-estimation and calibration methods for LLMs into white-box and black-box approaches, plus applications (hallucination detection, selective generation).

**Cited 2 times. Matched 1/2 = 50%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:150` | A complementary 2024 survey by Geng et al.~\cite{geng2024survey} catalogues white-box and black-box confidence-estimation methods for language models across the same axes that drive CAEM's verifier ensemble, and confirms that no single method exceeds AUROC of approximately $0.6$ across heterogeneous benchmarks; this survey is the foundational positioning document for CAEM's multi-signal design choice. | ⚠️ Partly | The catalogue of white-box and black-box methods matches. "No single method exceeds AUROC ≈0.6" is not in the survey and conflicts with [42]. |
| 2 | `chapters/chapter_4.tex:222` | The four-family decomposition follows the white-box-versus-black-box and instance-versus-aggregate axes catalogued in the recent confidence-estimation survey for language models \cite{geng2024survey}, but extends them with the explicit grounding-versus-relevance separation that the off-topic-but-grounded failure mode requires. | ✅ Matched | Matches: white-box vs. black-box axes. |

## [42] INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection
`chen2024inside` · Chen et al., 2024 · details matched: ⚠️ Partly

**What the paper says (paraphrased from its abstract and public record):** ICLR 2024. INSIDE uses internal states for hallucination detection. EigenScore is the log-determinant of the covariance of sentence embeddings across multiple responses, taken from a middle layer. It adds test-time feature clipping.

**Cited 2 times. Matched 1/2 = 50%** (partly 0, not matched 0, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:150` | Chen et al.~\cite{chen2024inside} introduced INSIDE / EigenScore at ICLR 2024, computing the log-determinant of the covariance matrix over middle-layer hidden states across $K$ resampled responses; the method reports AUROC near $0.84$ on TriviaQA, the strongest unsupervised single-signal result on factual QA we are aware of, and is registered in CAEM as a candidate replacement for the dropout signal that was not adopted in the locked configuration. | 🔒 Needs full text | The method description matches. "AUROC near 0.84 on TriviaQA" needs Table 1 of the paper. |
| 2 | `chapters/chapter_4.tex:278` | The hidden-state-eigenvalue alternative \cite{chen2024inside}, which scores the log-determinant of the per-query response covariance, was considered as a candidate replacement for the dropout signal but was not adopted in the locked configuration for cost-efficiency reasons. | ✅ Matched | Matches. |

## [43] Selective Question Answering under Domain Shift
`kamath2020selective` · Kamath et al., 2020 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ACL 2020. Selective QA under domain shift: train a calibrator on features of the QA model to predict correctness and abstain. Evaluated by risk–coverage.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:150` | The selective-prediction precision-versus-coverage tradeoff that CAEM reports in \Cref{ch:evaluation} follows the form pioneered by Kamath, Jia, and Liang~\cite{kamath2020selective} for selective question answering under domain shift, where a calibrator over feature signals predicts $P(\text{correct})$ and the selective rule answers only when the calibrator's confidence exceeds the registered precision target. | ✅ Matched | Matches: a calibrator predicting correctness, selective QA under shift. |
| 2 | `chapters/chapter_3.tex:163` | The selective-prediction convention reports the precision-versus-coverage tradeoff in the form pioneered by \cite{kamath2020selective}, with the abstention class measured against a registered precision floor rather than a hand-tuned operating point. | ✅ Matched | Matches. |

## [44] Neural Turing Machines
`graves2014neural` · Graves et al., 2014 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Neural Turing Machines (arXiv 2014): a neural network coupled to differentiable external memory. Learns simple algorithms such as copying and sorting.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:158` | Graves et al.\ \cite{graves2014neural} introduced Neural Turing Machines (NTMs) in 2014. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:303` | Neural Turing Machines \cite{graves2014neural}, Memory Networks \cite{weston2015memory}, and DNCs \cite{graves2016differentiable} demonstrate that differentiable memory enables structured reasoning; RAG \cite{NEURIPS2020_6b493230} shows that non-parametric knowledge improves factual accuracy (44.5\% vs 41.5\% on Natural Questions) and supports index updates without retraining; FAISS \cite{8733051} and Sentence-BERT \cite{reimers-gurevych-2019-sentence} make similarity search cheap enough to run in real time. | ✅ Matched | Matches. |
| 3 | `chapters/chapter_4.tex:97` | The design is in the lineage of memory-augmented neural networks for sequence and question-answering tasks \cite{weston2015memory,graves2014neural,graves2016differentiable}, of writeable episodic key-value stores for language models \cite{larimar2024das}, and of insertion-time memory augmentation \cite{xu2025amem}, but is specialised here as an external, query-keyed episodic store whose distinguishing property is the scalar storage score persisted with each entry. | ✅ Matched | Attribution is fair. |

## [45] Memory Networks
`weston2015memory` · Weston et al., 2015 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Memory Networks (ICLR 2015): long-term memory components that can be read and written, applied to QA.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:162` | Weston et al. \cite{weston2015memory} introduced Memory Networks at ICLR 2015. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:303` | Neural Turing Machines \cite{graves2014neural}, Memory Networks \cite{weston2015memory}, and DNCs \cite{graves2016differentiable} demonstrate that differentiable memory enables structured reasoning; RAG \cite{NEURIPS2020_6b493230} shows that non-parametric knowledge improves factual accuracy (44.5\% vs 41.5\% on Natural Questions) and supports index updates without retraining; FAISS \cite{8733051} and Sentence-BERT \cite{reimers-gurevych-2019-sentence} make similarity search cheap enough to run in real time. | ✅ Matched | Matches. |
| 3 | `chapters/chapter_4.tex:97` | The design is in the lineage of memory-augmented neural networks for sequence and question-answering tasks \cite{weston2015memory,graves2014neural,graves2016differentiable}, of writeable episodic key-value stores for language models \cite{larimar2024das}, and of insertion-time memory augmentation \cite{xu2025amem}, but is specialised here as an external, query-keyed episodic store whose distinguishing property is the scalar storage score persisted with each entry. | ✅ Matched | Attribution is fair. |

## [46] End-To-End Memory Networks
`NIPS2015_8fb21ee7` · Sukhbaatar et al., 2015 · details matched: ⚠️ Partly

**What the paper says (paraphrased from its abstract and public record):** End-to-End Memory Networks (NeurIPS 2015): a memory network trained end-to-end with soft attention.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:162` | Sukhbaatar et al. \cite{NIPS2015_8fb21ee7} extended this model with end to end Memory Networks at NeurIPS 2015. | ✅ Matched | Matches. |

## [47] Hybrid Computing Using a Neural Network with Dynamic External Memory
`graves2016differentiable` · Graves et al., 2016 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** DNC (Nature 2016): a neural network with dynamic external memory that can solve structured reasoning tasks such as graph traversal.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:164` | Graves et al. \cite{graves2016differentiable} presented the Differentiable Neural Computer (DNC) in Nature 2016. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:303` | Neural Turing Machines \cite{graves2014neural}, Memory Networks \cite{weston2015memory}, and DNCs \cite{graves2016differentiable} demonstrate that differentiable memory enables structured reasoning; RAG \cite{NEURIPS2020_6b493230} shows that non-parametric knowledge improves factual accuracy (44.5\% vs 41.5\% on Natural Questions) and supports index updates without retraining; FAISS \cite{8733051} and Sentence-BERT \cite{reimers-gurevych-2019-sentence} make similarity search cheap enough to run in real time. | ✅ Matched | Matches. |
| 3 | `chapters/chapter_4.tex:97` | The design is in the lineage of memory-augmented neural networks for sequence and question-answering tasks \cite{weston2015memory,graves2014neural,graves2016differentiable}, of writeable episodic key-value stores for language models \cite{larimar2024das}, and of insertion-time memory augmentation \cite{xu2025amem}, but is specialised here as an external, query-keyed episodic store whose distinguishing property is the scalar storage score persisted with each entry. | ✅ Matched | Attribution is fair. |

## [48] Larimar: Large Language Models with Episodic Memory Control
`larimar2024das` · Das et al., 2024 · details matched: ⚠️ Partly

**What the paper says (paraphrased from its abstract and public record):** ICML 2024. Larimar is a brain-inspired LLM architecture with a distributed episodic memory. It supports one-shot knowledge editing (8–10× faster than baselines), selective fact forgetting and context-length generalisation.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:176` | Das et al.~\cite{larimar2024das} introduced Larimar at ICML 2024, an associative key-value episodic memory with one-shot edits, selective forgetting, and an order-of-magnitude faster knowledge-editing path than direct fine-tuning; Larimar establishes that a writeable episodic store is a feasible alternative to parametric storage but does not associate a quality scalar with each entry. | ✅ Matched | Matches: one-shot edits, selective forgetting, 8–10× faster. Larimar's memory is a distributed episodic memory; "key-value" is loose. |
| 2 | `chapters/chapter_2.tex:249` | [table row] Larimar~\cite{larimar2024das} & no (key-value addr.) & key match & one-shot edit & --- & --- | ✅ Matched | Table row is fair. |
| 3 | `chapters/chapter_4.tex:97` | The design is in the lineage of memory-augmented neural networks for sequence and question-answering tasks \cite{weston2015memory,graves2014neural,graves2016differentiable}, of writeable episodic key-value stores for language models \cite{larimar2024das}, and of insertion-time memory augmentation \cite{xu2025amem}, but is specialised here as an external, query-keyed episodic store whose distinguishing property is the scalar storage score persisted with each entry. | ✅ Matched | Attribution is fair. |

## [49] A-MEM: Agentic Memory for LLM Agents
`xu2025amem` · Xu et al., 2025 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** arXiv 2025. A-MEM: agentic memory for LLM agents based on Zettelkasten. Creates structured notes and dynamically links and evolves memories.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:176` | Xu et al.~\cite{xu2025amem} introduced A-MEM, a Zettelkasten-inspired agentic memory that writes structured textual notes at insertion and evolves links between memories over time; A-MEM is the closest architectural neighbour to CAEM in the sense that both write structured metadata at insertion, but A-MEM's metadata is textual (tags, links) while CAEM's is a single calibrated scalar storage score that drives a tiered router rather than evolving link structure. | ✅ Matched | Matches: Zettelkasten notes and link evolution. |
| 2 | `chapters/chapter_2.tex:248` | [table row] A-MEM~\cite{xu2025amem} & no (textual tags) & similarity $+$ link evolution & --- & --- & --- | ✅ Matched | Table row is fair. |
| 3 | `chapters/chapter_4.tex:97` | The design is in the lineage of memory-augmented neural networks for sequence and question-answering tasks \cite{weston2015memory,graves2014neural,graves2016differentiable}, of writeable episodic key-value stores for language models \cite{larimar2024das}, and of insertion-time memory augmentation \cite{xu2025amem}, but is specialised here as an external, query-keyed episodic store whose distinguishing property is the scalar storage score persisted with each entry. | ✅ Matched | Attribution is fair. |

## [50] Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity
`jeong2024adaptive` · Jeong et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NAACL 2024. Adaptive-RAG: a classifier predicts query complexity and selects no retrieval, single-step retrieval or multi-step retrieval.

**Cited 6 times. Matched 6/6 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:176` | Jeong et al.~\cite{jeong2024adaptive} introduced Adaptive-RAG at NAACL 2024, a three-tier router that classifies queries by complexity and dispatches to no-retrieval, single-step retrieval, or multi-step retrieval accordingly; Adaptive-RAG shares CAEM's surface tier count and a backbone-family neighbourhood but routes on query-side complexity at inference time, not on insertion-time stored confidence on a shared memory, and that distinction is what CAEM's three-tier router differentiates against in \Cref{sec:router}. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:189` | Jeong et al.~\cite{jeong2024adaptive} introduced Adaptive-RAG at NAACL 2024 as a three-class extension of the binary routing decision. | ✅ Matched | Matches. |
| 3 | `chapters/chapter_2.tex:250` | [table row] Adaptive-RAG~\cite{jeong2024adaptive} & no & query complexity classifier & --- & --- & --- | ✅ Matched | Table row matches. |
| 4 | `chapters/chapter_4.tex:116` | The three-mechanism separation between the combined-score dispatch, the safety override, and the per-query retrieval-utility decision of \Cref{sec:ruc} is what distinguishes the router from a single-threshold dispatcher and from prior three-tier routers such as Adaptive-RAG \cite{jeong2024adaptive}, whose three tiers are gated by a query-complexity classifier rather than by per-entry stored quality on a shared memory. | ✅ Matched | Matches. |
| 5 | `chapters/chapter_4.tex:159` | The same architectural diagnosis was reached independently by the selective-retrieval line of work reviewed in \Cref{sec:lit-selective-retrieval}: Mallen et al.~\cite{mallen2023whennot} on PopQA, Wang et al.~\cite{wang2023skr} on self-knowledge routing, Jeong et al.~\cite{jeong2024adaptive} on Adaptive-RAG, and Baek et al.~\cite{baek2025probing} on Probing-RAG all converge on the architectural conclusion that an unconditional retrieval dispatch leaves measurable accuracy on the table relative to a per-query routing decision. | ✅ Matched | Matches: adaptive beats unconditional retrieval. |
| 6 | `chapters/chapter_5.tex:600` | The third group is learned per-query retrieval routing: \cite{jeong2024adaptive} routes by query-complexity classification, \cite{mallen2023whennot} routes by entity popularity, \cite{wang2023skr} routes by a self-knowledge classifier trained on retrieval-helpful versus retrieval-hurt empirical labels, and \cite{baek2025probing} routes by a linear probe on intermediate hidden states, while CAEM combines stored-quality routing on a shared memory with the retrieval-utility classifier of \Cref{sec:ruc} at the safety-veto and score-formula fall-throughs; the routing-signal axis is the architectural difference that the empirical contrast in \Cref{sec:comp-tier-baseline} and the ablation in \Cref{sec:ruc-ablation} are registered to surface. | ✅ Matched | Matches. |

## [51] When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories
`mallen2023whennot` · Mallen et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ACL 2023. PopQA study: LMs struggle with less popular (long-tail) facts and retrieval helps there. Proposes adaptive retrieval based on popularity.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:185` | Mallen et al.~\cite{mallen2023whennot} established the empirical anchor for the selective-retrieval position at ACL 2023. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_4.tex:159` | The same architectural diagnosis was reached independently by the selective-retrieval line of work reviewed in \Cref{sec:lit-selective-retrieval}: Mallen et al.~\cite{mallen2023whennot} on PopQA, Wang et al.~\cite{wang2023skr} on self-knowledge routing, Jeong et al.~\cite{jeong2024adaptive} on Adaptive-RAG, and Baek et al.~\cite{baek2025probing} on Probing-RAG all converge on the architectural conclusion that an unconditional retrieval dispatch leaves measurable accuracy on the table relative to a per-query routing decision. | ✅ Matched | Matches: PopQA. |
| 3 | `chapters/chapter_5.tex:600` | The third group is learned per-query retrieval routing: \cite{jeong2024adaptive} routes by query-complexity classification, \cite{mallen2023whennot} routes by entity popularity, \cite{wang2023skr} routes by a self-knowledge classifier trained on retrieval-helpful versus retrieval-hurt empirical labels, and \cite{baek2025probing} routes by a linear probe on intermediate hidden states, while CAEM combines stored-quality routing on a shared memory with the retrieval-utility classifier of \Cref{sec:ruc} at the safety-veto and score-formula fall-throughs; the routing-signal axis is the architectural difference that the empirical contrast in \Cref{sec:comp-tier-baseline} and the ablation in \Cref{sec:ruc-ablation} are registered to surface. | ✅ Matched | Matches: popularity-based adaptive retrieval. |

## [52] Self-Knowledge Guided Retrieval Augmentation for Large Language Models
`wang2023skr` · Wang et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Findings of EMNLP 2023. SKR elicits the model's self-knowledge (via prompting, a small trained classifier or kNN) to decide whether to retrieve. Training labels come from whether retrieval helped.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:187` | Wang et al.~\cite{wang2023skr} extended this line to a learned classifier at EMNLP Findings 2023. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_4.tex:159` | The same architectural diagnosis was reached independently by the selective-retrieval line of work reviewed in \Cref{sec:lit-selective-retrieval}: Mallen et al.~\cite{mallen2023whennot} on PopQA, Wang et al.~\cite{wang2023skr} on self-knowledge routing, Jeong et al.~\cite{jeong2024adaptive} on Adaptive-RAG, and Baek et al.~\cite{baek2025probing} on Probing-RAG all converge on the architectural conclusion that an unconditional retrieval dispatch leaves measurable accuracy on the table relative to a per-query routing decision. | ✅ Matched | Matches. |
| 3 | `chapters/chapter_5.tex:600` | The third group is learned per-query retrieval routing: \cite{jeong2024adaptive} routes by query-complexity classification, \cite{mallen2023whennot} routes by entity popularity, \cite{wang2023skr} routes by a self-knowledge classifier trained on retrieval-helpful versus retrieval-hurt empirical labels, and \cite{baek2025probing} routes by a linear probe on intermediate hidden states, while CAEM combines stored-quality routing on a shared memory with the retrieval-utility classifier of \Cref{sec:ruc} at the safety-veto and score-formula fall-throughs; the routing-signal axis is the architectural difference that the empirical contrast in \Cref{sec:comp-tier-baseline} and the ablation in \Cref{sec:ruc-ablation} are registered to surface. | ✅ Matched | Matches: labels come from whether retrieval helped. |

## [53] LLM-Independent Adaptive RAG: Let the Question Speak for Itself
`moskvoretskii2025llm` · Moskvoretskii et al., 2025 · details matched: ❌ No

**What the paper says (paraphrased from its abstract and public record):** EMNLP 2025 (first author Marina). Lightweight, LLM-independent adaptive-retrieval methods using external features (27 features in 7 groups, 6 QA datasets). They match complex LLM-based methods at much lower cost.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:189` | The architectural lesson for CAEM is that a learned classifier on automatically-collected outcome labels can decide the routing question, but that putting a large model at routing time costs more than the architecture gains; the follow-up of Moskvoretskii et al.~\cite{moskvoretskii2025llm} extends Adaptive-RAG to the regime where the classifier carries no language-model forward pass and matches the routing accuracy of the language-model-based variant on the same suite. | ✅ Matched | The claim matches the abstract. **But** the first author is Marina, not Moskvoretskii. |
| 2 | `chapters/chapter_4.tex:173` | The lightweight-classifier-only operating point CAEM adopts is consistent with the empirical conclusion of Moskvoretskii et al.~\cite{moskvoretskii2025llm}, who show that an external classifier with no language-model forward pass at routing time matches the routing accuracy of language-model-based selective retrievers on the standard suite; CAEM differs in that the architecturally heavy retrieval-quality features (cross-encoder rerank, pairwise NLI on top-five passages) are reused from the verifier ensemble's already-loaded substrate rather than computed afresh. | ✅ Matched | Claim matches. Author naming is wrong (see above). |

## [54] Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
`asai2024selfrag` · Asai et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICLR 2024. Self-RAG trains an LM (Llama-2) to retrieve on demand and to critique its own generations with reflection tokens.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:191` | Asai et al.~\cite{asai2024selfrag} introduced Self-RAG at ICLR 2024 as the end-to-end fine-tuned alternative to the external-classifier approach. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:267` | A fourth retrieval system, \textbf{Self-RAG}~\cite{asai2024selfrag}, is acknowledged as the most ambitious retrieve-and-critique pipeline in the literature but is included only as a citation-level reference rather than as a replicated numerical baseline, because Self-RAG's critic tokens are tied to a LLaMA-family backbone and retargeting them to Qwen-2.5-3B-Instruct would change enough of the system to invalidate the architectural comparison. | ✅ Matched | Matches: trained on Llama-2. |
| 3 | `chapters/chapter_5.tex:478` | The first is Self-RAG \cite{asai2024selfrag}, included as a citation-only reference rather than as a replicated numerical baseline because Self-RAG's critic tokens are tied to a LLaMA-family backbone and retargeting them to Qwen-2.5-3B-Instruct would change enough of the system to invalidate the architectural comparison. | ✅ Matched | Matches. |

## [55] Probing-RAG: Self-Probing to Guide Language Models in Selective Document Retrieval
`baek2025probing` · Baek et al., 2025 · details matched: ❌ No

**What the paper says (paraphrased from its abstract and public record):** Findings of NAACL 2025. Probing-RAG uses a prober on the LM's intermediate hidden states to decide whether to retrieve. It beats prior adaptive methods on 5 open-domain QA datasets with fewer retrieval steps.

**Cited 3 times. Matched 2/3 = 67%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:193` | Baek et al.~\cite{baek2025probing} introduced Probing-RAG at NAACL Findings 2025 as a midpoint between the heavy classifier and the in-generator reflection tokens. | ✅ Matched | Matches: Findings of NAACL 2025, hidden-state prober. Authors are wrong in the bib. |
| 2 | `chapters/chapter_4.tex:159` | The same architectural diagnosis was reached independently by the selective-retrieval line of work reviewed in \Cref{sec:lit-selective-retrieval}: Mallen et al.~\cite{mallen2023whennot} on PopQA, Wang et al.~\cite{wang2023skr} on self-knowledge routing, Jeong et al.~\cite{jeong2024adaptive} on Adaptive-RAG, and Baek et al.~\cite{baek2025probing} on Probing-RAG all converge on the architectural conclusion that an unconditional retrieval dispatch leaves measurable accuracy on the table relative to a per-query routing decision. | ✅ Matched | Matches. |
| 3 | `chapters/chapter_5.tex:600` | The third group is learned per-query retrieval routing: \cite{jeong2024adaptive} routes by query-complexity classification, \cite{mallen2023whennot} routes by entity popularity, \cite{wang2023skr} routes by a self-knowledge classifier trained on retrieval-helpful versus retrieval-hurt empirical labels, and \cite{baek2025probing} routes by a linear probe on intermediate hidden states, while CAEM combines stored-quality routing on a shared memory with the retrieval-utility classifier of \Cref{sec:ruc} at the safety-veto and score-formula fall-throughs; the routing-signal axis is the architectural difference that the empirical contrast in \Cref{sec:comp-tier-baseline} and the ablation in \Cref{sec:ruc-ablation} are registered to surface. | ⚠️ Partly | A prober on hidden states matches. That it is a "linear" probe is unconfirmed. |

## [56] LoRA Learns Less and Forgets Less
`biderman2024lora` · Biderman et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** TMLR 2024. LoRA substantially underperforms full fine-tuning on code and math, but better preserves base-model performance outside the target domain (forgetting measured with HellaSwag, ARC, WinoGrande).

**Cited 2 times. Matched 1/2 = 50%** (partly 0, not matched 0, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:212` | Biderman and colleagues sweep Llama-2-7B across rank settings from sixteen to two hundred and fifty-six on Magicoder coding instruction data \cite{biderman2024lora} and report two findings that translate directly to CAEM's setting: low-rank adaptation retains roughly sixty-six percent of HellaSwag, ARC-Challenge, and Winogrande accuracy at every rank tested, against sixty-two percent retention for full fine-tuning matched on epoch count, and the low-rank versus full gap on retention dominates the rank-versus-rank gap on fit. | 🔒 Needs full text | Llama-2-7B, the rank sweep and the 66% vs. 62% retention figures need the full text. |
| 2 | `chapters/chapter_2.tex:305` | The parameter-efficient-fine-tuning literature reviewed at \Cref{sec:lit-self-improvement} establishes that low-rank adaptation \cite{hu2022lora} on all linear layers retains general capability across iterative cycles substantially better than full-parameter fine-tuning at matched training budget \cite{biderman2024lora,dettmers2023qlora}, which is the empirical anchor for CAEM's choice of the LoRA primary path with the $L_2$ anchor placed on the adapter parameters rather than on the full backbone. | ✅ Matched | Matches for [56]: LoRA forgets less than full fine-tuning. |

## [57] QLoRA: Efficient Finetuning of Quantized LLMs
`dettmers2023qlora` · Dettmers et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2023. QLoRA: 4-bit quantised base model plus LoRA. LoRA on all linear transformer layers is needed to match full fine-tuning; the number of adapted layers matters more than rank.

**Cited 2 times. Matched 0/2 = 0%** (partly 2, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:212` | Dettmers and colleagues' QLoRA work \cite{dettmers2023qlora} establishes the second decisive practical point: applying low-rank adaptation to all linear layers, both attention and feed-forward, rather than to the attention projections alone is the most critical configuration choice, and matched-parameter-count experiments show that adding feed-forward adapters to attention adapters dominates adding more attention rank. | ⚠️ Partly | "LoRA on all linear layers is needed" matches. "Adding FFN adapters dominates adding attention rank" is unconfirmed. |
| 2 | `chapters/chapter_2.tex:305` | The parameter-efficient-fine-tuning literature reviewed at \Cref{sec:lit-self-improvement} establishes that low-rank adaptation \cite{hu2022lora} on all linear layers retains general capability across iterative cycles substantially better than full-parameter fine-tuning at matched training budget \cite{biderman2024lora,dettmers2023qlora}, which is the empirical anchor for CAEM's choice of the LoRA primary path with the $L_2$ anchor placed on the adapter parameters rather than on the full backbone. | ⚠️ Partly | QLoRA is not a study of forgetting or retention, so it does not support the retention claim. |

## [58] A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA
`kalajdzievski2023rslora` · Kalajdzievski, 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** arXiv 2023. rsLoRA: the standard α/r scaling slows or collapses learning at higher ranks. Proposes α/√r.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:212` | CAEM's adapter target set spans every linear projection inside the transformer block (the four attention projections together with the three feed-forward projections of the gated multi-layer perceptron), which is the all-linear-layer target set that this literature converges on, fitted at rank thirty-two with the standard $\alpha=2r$ scaling, which sits comfortably inside the rank regime where the standard scaling avoids the gradient collapse at higher rank that motivates the rank-stabilised alternative \cite{kalajdzievski2023rslora}. | ✅ Matched | Matches: gradient collapse at high rank with α/r. |

## [59] Orthogonal Subspace Learning for Language Model Continual Learning
`wang2023olora` · Wang et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** Findings of EMNLP 2023. O-LoRA learns new tasks in LoRA subspaces orthogonal to earlier ones to reduce catastrophic forgetting in continual learning.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:212` | The within-adapter analogue of catastrophic forgetting under sequential fine-tuning, documented by Wang and colleagues \cite{wang2023olora} and orthogonal-decomposition fixes for it, is registered as future work; the present thesis combines the bounded low-rank-adapter parameter envelope on a frozen backbone, the multi-modal retention guard with adapter rollback, and the strict training threshold above the storage cut-point as the three-mechanism defence against iterative-adapter drift across the realised cycle trajectory. | ✅ Matched | Matches: O-LoRA and forgetting in continual learning with LoRA. |
| 2 | `chapters/chapter_2.tex:305` | Within-adapter forgetting under sequential fine-tuning is documented by \cite{wang2023olora}, whose orthogonal-subspace decomposition is registered as a future-work upgrade if the multi-modal retention probe panel reports persistent drift on the trajectory's later cycles. | ✅ Matched | Matches. |

## [60] Learning with Noisy Labels
`natarajan2013learning` · Natarajan et al., 2013 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2013. Learning under class-conditional random label noise. Unbiased estimators and weighted losses give guarantees when noise rates satisfy ρ₊+ρ₋<1.

**Cited 3 times. Matched 2/3 = 67%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:216` | Natarajan et al. \cite{natarajan2013learning} provided theoretical foundations for learning with noisy labels at NeurIPS 2013. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:305` | Noisy-label learning \cite{natarajan2013learning} proves that training on pools whose purity is well below 100\% can still improve a model, which justifies the registered stricter training threshold $\tau_{\text{train}} > \tau_{\text{store}}$ that lifts training-pool purity above memory purity, where $\tau_{\text{store}}$ is the registered fixed-threshold cut-point on the calibrated probability scale. | ⚠️ Partly | The paper gives conditions for learning under label noise (ρ₊+ρ₋<1). "Can still improve a model" and the threshold justification are your interpretation. |
| 3 | `appendix/appendix_g.tex:169` | The noise-tolerant-learning literature \cite{natarajan2013learning} bounds the corruption rate at which training under noisy labels still improves a model rather than degrading it; the architecture spends the strict-training-threshold budget to keep the training pool's empirical purity above that rate by construction. | ✅ Matched | Fair: the guarantees need noise rates ρ₊+ρ₋<1. |

## [61] STaR: Bootstrapping Reasoning With Reasoning
`NEURIPS2022_639a9a17` · Zelikman et al., 2022 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2022. STaR iteratively generates rationales, keeps those that reach the correct (ground-truth) answer, rationalises failures given the answer, and fine-tunes.

**Cited 6 times. Matched 5/6 = 83%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:220` | When Zelikman et al. \cite{NEURIPS2022_639a9a17} introduced STaR (Self Taught Reasoner) at NeurIPS 2022, they showed successful bootstrapping using iterative self improvement. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:269` | \textbf{STaR}~\cite{NEURIPS2022_639a9a17} is discussed above as the closest prior method in spirit and is run only as an optional ceiling reference conditional on remaining compute budget: its rationalisation step and its binary correctness filter differ enough from CAEM's nine-signal verifier and four-outcome decision that a large gap to STaR would mix architectural contribution with rationalisation-quality artefacts, so STaR is reported where possible as an aspirational ceiling for what a pure iterative-refinement scheme can reach rather than as a primary competitor. | ✅ Matched | Matches: rationalisation and a correctness filter. |
| 3 | `chapters/chapter_2.tex:305` | STaR \cite{NEURIPS2022_639a9a17} demonstrates the bootstrap loop of generate--filter--fine-tune that CAEM generalises by replacing simple correctness filtering with the nine-signal calibrated-probability composite $\ustored$ (per-signal isotonic regression with a per-benchmark shrinkage prior plus an $L_{2}$-regularised boost layer; the contradiction probability $p_{\text{contra}}$ is excluded from the composite signal set because the MiniCheck-only deployment makes it structurally zero), and by persisting verified items in episodic memory rather than only in the adapter. | ✅ Matched | Matches. |
| 4 | `chapters/chapter_4.tex:473` | It is an instance of the broader family of model-self-training procedures that fine-tune a generator on its own filtered outputs \cite{NEURIPS2022_639a9a17,hosseini2024vstar,gulcehre2023rest,ho2023large}, specialised here through the fixed-threshold storage gate that supplies the filter, the parameter-efficient adapter layer that absorbs the gradient update, the loss-reweighted training pool that prevents single-benchmark dominance, and the multi-modal retention guard that gates the commit. | ✅ Matched | Matches. |
| 5 | `chapters/chapter_5.tex:478` | The second is STaR \cite{NEURIPS2022_639a9a17}, run as an optional ceiling reference conditional on remaining compute budget; STaR's rationalisation step and binary correctness filter differ enough from CAEM's nine-signal verifier and four-outcome decision that a large gap to STaR would mix architectural contribution with rationalisation-quality artefacts. | ✅ Matched | Matches. |
| 6 | `chapters/chapter_5.tex:600` | The second group is verifier-filtered self-improvement: \cite{NEURIPS2022_639a9a17} and \cite{hosseini2024vstar} demonstrate that bootstrapping on verifier-correct outputs is a viable training-data-admission rule, and \cite{huang2025sharpening} formalises the convergence dynamics that CAEM's \Cref{thm:convergence} extends under the bounded-band stationarity reframing. | ⚠️ Partly | STaR filters by ground-truth answer correctness, not by a learned verifier. |

## [62] When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs
`kamoi-etal-2024-llms` · Kamoi et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** TACL 2024 critical survey. No prior work shows successful self-correction with feedback from prompted LLMs alone, except in specially suited tasks. Reliable external feedback helps.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:224` | Kamoi et al. \cite{kamoi-etal-2024-llms} analyzed when LLMs can correct mistakes at TACL 2024. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_2.tex:305` | The self-correction critiques of \cite{huang-etal-2023-large} and \cite{kamoi-etal-2024-llms} establish that intrinsic prompt-based self-correction rarely improves accuracy without external feedback, which is exactly the feedback CAEM supplies through the external verifier. | ✅ Matched | Matches. |

## [63] Don't Stop Pretraining: Adapt Language Models to Domains and Tasks
`gururangan-etal-2020-dont` · Gururangan et al., 2020 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ACL 2020. Domain-adaptive and task-adaptive pretraining (continued pretraining) improves downstream performance.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:230` | Gururangan et al. \cite{gururangan-etal-2020-dont} demonstrated at ACL 2020 that continued pre-training on domain-specific unlabeled data (domain-adaptive pretraining) followed by task-specific fine-tuning outperforms direct fine-tuning on limited labeled data. | ✅ Matched | Matches. |

## [64] V-STaR: Training Verifiers for Self-Taught Reasoners
`hosseini2024vstar` · Hosseini et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** COLM 2024. V-STaR trains the generator STaR-style and also trains a DPO verifier on correct and incorrect self-generated solutions. At inference the verifier ranks candidates.

**Cited 4 times. Matched 3/4 = 75%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:234` | Hosseini et al.~\cite{hosseini2024vstar} introduced V-STaR at COLM 2024, training a DPO verifier over correct and incorrect self-generated solutions and using it to rank test-time candidates; CAEM generalises V-STaR by using the verifier for training-data admission rather than just test-time ranking, and by adding the $L_{2}$ anchor and the episodic memory. | ✅ Matched | Matches: DPO verifier and test-time ranking. |
| 2 | `chapters/chapter_2.tex:251` | [table row] V-STaR~\cite{hosseini2024vstar} & no & --- & DPO verifier (test-time rank) & --- & --- | ✅ Matched | Table row matches. |
| 3 | `chapters/chapter_4.tex:473` | It is an instance of the broader family of model-self-training procedures that fine-tune a generator on its own filtered outputs \cite{NEURIPS2022_639a9a17,hosseini2024vstar,gulcehre2023rest,ho2023large}, specialised here through the fixed-threshold storage gate that supplies the filter, the parameter-efficient adapter layer that absorbs the gradient update, the loss-reweighted training pool that prevents single-benchmark dominance, and the multi-modal retention guard that gates the commit. | ✅ Matched | Matches. |
| 4 | `chapters/chapter_5.tex:600` | The second group is verifier-filtered self-improvement: \cite{NEURIPS2022_639a9a17} and \cite{hosseini2024vstar} demonstrate that bootstrapping on verifier-correct outputs is a viable training-data-admission rule, and \cite{huang2025sharpening} formalises the convergence dynamics that CAEM's \Cref{thm:convergence} extends under the bounded-band stationarity reframing. | ⚠️ Partly | V-STaR's generator is trained on ground-truth-correct samples. The verifier is used for ranking at inference. |

## [65] Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models
`song2024gap` · Song et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICLR 2025. Studies LLM self-improvement through the "generation–verification gap". A variant of this gap scales monotonically with pretraining FLOPs. Studies when and how iterative self-improvement works. It is an empirical study with no iff theorem.

**Cited 2 times. Matched 0/2 = 0%** (partly 2, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:234` | Song et al.~\cite{song2024gap} formalise the \emph{generation--verification gap} and prove empirically that iterative self-improvement succeeds if and only if the verifier is more accurate than the generator on the noisy-label-recovery task; this is the conceptual basis for CAEM's $\alpha > 1/2$ verifier-purity precondition in \Cref{sec:theory}. | ⚠️ Partly | The generation–verification gap matches. "Prove empirically … if and only if" is not in the paper. |
| 2 | `chapters/chapter_4.tex:473` | Recent theoretical work on whether such loops collapse or improve their generator establishes a sufficient condition that the verifier outperform the generator at the noisy-label-recovery task \cite{song2024gap}, and a complementary sufficient condition that the verifier-filtered admission policy be precision-bounded rather than recall-bounded so that the loop accumulates rather than displaces high-quality data \cite{fu2025collapse}; both conditions are met by the storage class admitted under the registered cut-points of \Cref{sec:conformal-gate}. | ⚠️ Partly | The paper is mainly empirical and states no formal sufficient condition. |

## [66] Escaping Model Collapse via Synthetic Data Verification: Near-Term Improvement and Long-Term Convergence
`fu2025collapse` · Fu et al., 2025 · details matched: ❌ No

**What the paper says (paraphrased from its abstract and public record):** Yi, Liu, Cheng, Xu (ICLR 2026). Injecting external verifier-filtered synthetic data avoids model collapse. In linear regression, retraining gives near-term improvement but converges to the verifier's knowledge, and the gains plateau unless the verifier is perfect.

**Cited 2 times. Matched 0/2 = 0%** (partly 1, not matched 0, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:234` | Fu et al.~\cite{fu2025collapse} prove theoretically that a verifier filter over synthetic data prevents model collapse in self-consuming training loops and admits both near-term improvement and long-term convergence guarantees; this is the strongest published safety anchor for CAEM's storage-gate-filtered SIL. | ⚠️ Partly | The paper shows verifier-filtered retraining gives near-term gains but converges to the *verifier's* knowledge, with gains that can plateau. "Prevents collapse with long-term guarantees" overstates it. Authors are Yi et al. |
| 2 | `chapters/chapter_4.tex:473` | Recent theoretical work on whether such loops collapse or improve their generator establishes a sufficient condition that the verifier outperform the generator at the noisy-label-recovery task \cite{song2024gap}, and a complementary sufficient condition that the verifier-filtered admission policy be precision-bounded rather than recall-bounded so that the loop accumulates rather than displaces high-quality data \cite{fu2025collapse}; both conditions are met by the storage class admitted under the registered cut-points of \Cref{sec:conformal-gate}. | 🔒 Needs full text | A "precision-bounded vs. recall-bounded" condition was not found. Needs the full text. |

## [67] Retraining with Predicted Hard Labels Provably Increases Model Accuracy
`das2025retraining` · Das et al., 2025 · details matched: ⚠️ Partly

**What the paper says (paraphrased from its abstract and public record):** ICML 2025. In binary classification with noisy labels, retraining on the model's own predicted hard labels provably raises accuracy. Consensus-based retraining (keeping samples where predicted = given label) helps label-DP training.

**Cited 2 times. Matched 0/2 = 0%** (partly 2, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:234` | Das et al.~\cite{das2025retraining} prove at ICML 2025 that retraining on verifier-correct hard labels strictly increases population accuracy in the binary linearly-separable classification setting; this is the closest published analogue of CAEM's pool-purity inequality (\Cref{thm:purity}), and CAEM extends it from binary classification with internal self-prediction to open-ended generation with an external multi-signal verifier. | ⚠️ Partly | The paper uses the model's own predicted labels (optionally consensus-filtered), not verifier-correct labels. The setting is binary classification with label noise. |
| 2 | `chapters/chapter_4.tex:507` | The second layer treats the storage gate as a Bayesian filter whose expected pool purity is determined by the verifier's balanced accuracy and the base generator's correctness rate, in the lineage of the recent analyses showing that retraining on verifier-correct labels strictly increases population accuracy in the binary linearly-separable setting \cite{das2025retraining}, that weak-to-strong pseudolabel training admits expansion-based generalisation bounds \cite{lang2024w2s}, and that the sharpening mechanism formalises the amortisation-via-self-training bound for language-model self-improvement \cite{huang2025sharpening}. | ⚠️ Partly | Same: not "verifier-correct" labels. |

## [68] Self-Improvement in Language Models: The Sharpening Mechanism
`huang2025sharpening` · Huang et al., 2025 · details matched: ⚠️ Partly

**What the paper says (paraphrased from its abstract and public record):** ICLR 2025. Frames self-improvement as "sharpening": amortising self-verification-based search into the model. Gives statistical sample-complexity results for SFT- and RLHF-based self-improvement.

**Cited 3 times. Matched 2/3 = 67%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:234` | Huang, Block, Foster et al.~\cite{huang2025sharpening} formalise self-improvement as a sharpening mechanism at ICLR 2025 and prove convergence-style amortisation-via-self-training bounds that are the closest published analogue of CAEM's bounded-band stationarity result on the committed-cycle subsequence (\Cref{thm:convergence}). | ✅ Matched | Matches: sharpening and amortisation, ICLR 2025. |
| 2 | `chapters/chapter_4.tex:507` | The second layer treats the storage gate as a Bayesian filter whose expected pool purity is determined by the verifier's balanced accuracy and the base generator's correctness rate, in the lineage of the recent analyses showing that retraining on verifier-correct labels strictly increases population accuracy in the binary linearly-separable setting \cite{das2025retraining}, that weak-to-strong pseudolabel training admits expansion-based generalisation bounds \cite{lang2024w2s}, and that the sharpening mechanism formalises the amortisation-via-self-training bound for language-model self-improvement \cite{huang2025sharpening}. | ✅ Matched | Matches. |
| 3 | `chapters/chapter_5.tex:600` | The second group is verifier-filtered self-improvement: \cite{NEURIPS2022_639a9a17} and \cite{hosseini2024vstar} demonstrate that bootstrapping on verifier-correct outputs is a viable training-data-admission rule, and \cite{huang2025sharpening} formalises the convergence dynamics that CAEM's \Cref{thm:convergence} extends under the bounded-band stationarity reframing. | ⚠️ Partly | The paper gives sample-complexity results, not multi-cycle convergence dynamics. |

## [69] Theoretical Analysis of Weak-to-Strong Generalization
`lang2024w2s` · Lang et al., 2024 · details matched: ⚠️ Partly

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2024. Theory of weak-to-strong generalisation through pseudolabel correction and coverage expansion. Bounds rely on expansion properties.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:234` | Lang, Vijayaraghavan, and Sontag~\cite{lang2024w2s} provide a complementary expansion-based bound at NeurIPS 2024 for when a strong student trained on weak-teacher pseudolabels outperforms the teacher; their bound requires expansion assumptions rather than a verifier-accuracy threshold and is therefore complementary to, rather than redundant with, CAEM's data-purity formulation. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_4.tex:507` | The second layer treats the storage gate as a Bayesian filter whose expected pool purity is determined by the verifier's balanced accuracy and the base generator's correctness rate, in the lineage of the recent analyses showing that retraining on verifier-correct labels strictly increases population accuracy in the binary linearly-separable setting \cite{das2025retraining}, that weak-to-strong pseudolabel training admits expansion-based generalisation bounds \cite{lang2024w2s}, and that the sharpening mechanism formalises the amortisation-via-self-training bound for language-model self-improvement \cite{huang2025sharpening}. | ✅ Matched | Matches. |

## [70] Dense Passage Retrieval for Open-Domain Question Answering
`karpukhin-etal-2020-dense` · Karpukhin et al., 2020 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** EMNLP 2020. DPR: a dense dual-encoder passage retriever that outperforms BM25 for open-domain QA (NQ, TriviaQA, …).

**Cited 4 times. Matched 4/4 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:267` | \textbf{DPR-RAG}~\cite{karpukhin-etal-2020-dense, NEURIPS2020_6b493230} retrieves the top $k=5$ Wikipedia passages from a dense DPR index, concatenates them into a 384-token context, and generates from the same Qwen-2.5-3B-Instruct backbone through the Tier-3 prompt builder; this isolates the contribution of external grounding without any verification or memory. | ✅ Matched | Attribution is correct. |
| 2 | `chapters/chapter_2.tex:281` | NQ is the dominant open-domain QA benchmark in recent retrieval-augmented-generation literature~\cite{NEURIPS2020_6b493230, karpukhin-etal-2020-dense}. | ✅ Matched | Matches. |
| 3 | `chapters/chapter_4.tex:64` | The dense passage index is a single sentence-embedding-encoded snapshot of a public Wikipedia-derived corpus, structured as a dense-retrieval index in the spirit of \cite{karpukhin-etal-2020-dense}. | ✅ Matched | Attribution is fair. |
| 4 | `chapters/chapter_5.tex:431` | The retrieval-augmented baseline retrieves the top-$k = 5$ Wikipedia passages from the dense passage index used by CAEM's Tier 3 \cite{karpukhin-etal-2020-dense,NEURIPS2020_6b493230}, concatenates them into a 384-token context, and generates from the same backbone through the Tier-3 prompt template. | ✅ Matched | Attribution is correct. |

## [71] Active Retrieval Augmented Generation
`jiang-etal-2023-active` · Jiang et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** EMNLP 2023. FLARE iteratively predicts the upcoming sentence and, if it contains low-probability tokens, retrieves using it as a query and regenerates.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:267` | \textbf{FLARE}~\cite{jiang-etal-2023-active} extends DPR-RAG with active retrieval: rather than retrieving once at the start of generation, FLARE generates one sentence at a time and triggers a retrieval whenever the token-level confidence on the upcoming sentence falls below a registered threshold, with retrieved passages prepended to the generation context for the next sentence; this isolates the contribution of selective and step-conditioned retrieval over single-pass retrieval. | ✅ Matched | Matches: sentence-level look-ahead with retrieval on low-confidence tokens. |
| 2 | `chapters/chapter_5.tex:451` | The forward-looking active retrieval baseline \cite{jiang-etal-2023-active} is the active-retrieval extension of B3: rather than retrieving once at the start of generation, the model generates one sentence at a time and triggers a retrieval whenever the token-level confidence on the upcoming sentence falls below a registered threshold, with the retrieved passages prepended to the generation context for the next sentence. | ✅ Matched | Matches. |

## [72] TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
`joshi-etal-2017-triviaqa` · Joshi et al., 2017 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ACL 2017. TriviaQA: over 650K question–answer–evidence triples (about 95K QA pairs written by trivia enthusiasts), with evidence documents from the web and Wikipedia.

**Cited 4 times. Matched 3/4 = 75%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:279` | Joshi et al.~\cite{joshi-etal-2017-triviaqa} introduced TriviaQA at ACL 2017, providing 650{,}000 trivia question-answer pairs drawn from quizzing leagues, each paired with multiple supporting documents from the web and Wikipedia. | ⚠️ Partly | 650K is the number of question–answer–**evidence triples**. There are about 95K QA pairs. |
| 2 | `chapters/chapter_2.tex:307` | TriviaQA \cite{joshi-etal-2017-triviaqa} tests Tier-3 retrieval-augmented generation under noisy evidence on the open-text factoid surface and contributes a held-out test fold to the multi-modal retention probe panel. | ✅ Matched | Usage; the attribution is correct. |
| 3 | `chapters/chapter_3.tex:145` | The five evaluation benchmarks are FEVER \cite{thorne-etal-2018-fever}, TriviaQA \cite{joshi-etal-2017-triviaqa}, CommonsenseQA \cite{talmor-etal-2019-commonsenseqa}, TruthfulQA \cite{lin-etal-2022-truthfulqa}, and StrategyQA \cite{geva2021strategyqa}, all of which are distributed under licences that permit research use and redistribution under attribution; the licence terms are catalogued in Appendix~\ref{app:reproducibility} together with the access date and the dataset version hash. | ✅ Matched | Usage; the attribution is correct. |
| 4 | `chapters/chapter_5.tex:41` | The training panel comprises FEVER \cite{thorne-etal-2018-fever} for factual claim verification, TriviaQA \cite{joshi-etal-2017-triviaqa} for open-domain trivia, and CommonsenseQA \cite{talmor-etal-2019-commonsenseqa} for five-option multiple-choice commonsense reasoning. | ✅ Matched | Usage; the attribution is correct. |

## [73] Natural Questions: A Benchmark for Question Answering Research
`kwiatkowski-etal-2019-natural` · Kwiatkowski et al., 2019 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** TACL 2019. Natural Questions: real anonymised Google queries with Wikipedia pages annotated with long and short answers.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:281` | Kwiatkowski et al.~\cite{kwiatkowski-etal-2019-natural} introduced Natural Questions (NQ) at TACL 2019, providing real anonymised queries to the Google search engine paired with Wikipedia-page long answers and span-level short answers. | ✅ Matched | Matches. |

## [74] Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies
`10.1162/tacl_a_00370` · Geva et al., 2021 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** TACL 2021. StrategyQA: 2,780 yes/no questions that need implicit reasoning steps, with decompositions and evidence.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:285` | Geva et al.~\cite{10.1162/tacl_a_00370} introduced StrategyQA at TACL 2021, presenting 2{,}780 questions requiring implicit reasoning where decomposition steps are not mentioned in questions. | ✅ Matched | Matches: 2,780 questions. |
| 2 | `chapters/chapter_2.tex:307` | TruthfulQA \cite{lin-etal-2022-truthfulqa} and StrategyQA \cite{10.1162/tacl_a_00370} are held out from the SIL training pool as transfer-only benchmarks, exercising the cross-encoder relevance signal on imitative falsehoods and sample-set agreement on implicit multi-hop reasoning respectively. | ✅ Matched | Usage; the attribution is correct. |

## [75] Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
`clark2018think` · Clark et al., 2018 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** arXiv 2018. ARC: grade-school science questions. The Challenge set (2,590 questions) contains questions answered wrongly by retrieval and co-occurrence baselines.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:287` | Clark et al.~\cite{clark2018think} introduced ARC-Challenge at AI2 in 2018, a subset of 2{,}590 grade-school science questions filtered to be hard for co-occurrence and information-retrieval baselines. | ✅ Matched | Matches: 2,590 Challenge questions. |

## [76] Measuring Massive Multitask Language Understanding
`hendrycks2021measuring` · Hendrycks et al., 2021 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICLR 2021. MMLU: multiple-choice questions across 57 subjects in STEM, humanities, social sciences and other areas.

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:289` | Hendrycks et al.~\cite{hendrycks2021measuring} introduced MMLU in 2021, a 57-subject multiple-choice benchmark covering humanities, STEM, social sciences, and professional domains. | ✅ Matched | Matches: 57 subjects. |
| 2 | `chapters/chapter_2.tex:307` | MMLU \cite{hendrycks2021measuring} is the third probe in the multi-modal retention panel, supplying a broad general-knowledge multiple-choice surface that catches catastrophic forgetting on tasks the cycle was not trained on. | ✅ Matched | Usage; the attribution is correct. |
| 3 | `chapters/chapter_5.tex:41` | The retention guard reads a registered multi-modal probe panel rather than a single fold: a general-knowledge multiple-choice probe on MMLU \cite{hendrycks2021measuring}, a held-out open-text factoid probe drawn from the TriviaQA test fold, and a held-out commonsense multiple-choice probe drawn from the CommonsenseQA test fold; none of these probes is used as a target benchmark for the headline metrics. | ✅ Matched | Usage; the attribution is correct. |

## [77] Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation
`kuhn2023semantic` · Kuhn et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICLR 2023. Semantic entropy: an uncertainty measure over meaning-equivalence classes (bidirectional entailment) of sampled generations.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:297` | The normalised semantic entropy of \cite{farquhar2024semantic, kuhn2023semantic} and the kernel-language-entropy generalisation of \cite{nikitin2024kle} were registered earlier in the project as candidate sample-set members but retired before the main run on cycle-zero evidence that they contributed no discriminative information beyond what $s_{\text{avg}}$ and $p_{\text{entail}}$ already provide on the same resampled pool. | ✅ Matched | Attribution is correct. |
| 2 | `chapters/chapter_4.tex:285` | The normalised semantic entropy signal of \cite{farquhar2024semantic,kuhn2023semantic}, computed over $K=10$ chains at temperature $T=1.0$, registered a Pearson correlation indistinguishable from zero with exact-match labels on every per-benchmark slice of the calibration fold and a boost-layer weight within five times ten to the negative four of zero on every fit. | ✅ Matched | Attribution is correct. |

## [79] HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models
`li-etal-2023-halueval` · Li et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** EMNLP 2023. HaluEval: a large benchmark of generated and human-annotated hallucinated samples (QA, dialogue, summarisation).

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:297` | The thesis uses MiniCheck-Flan-T5-Large \cite{tang2024minicheck} as the deployed entailment backend across all hypothesis lengths because the registered long-hypothesis Qwen judge fails the pre-registered Pearson floor on the registered 500-sample calibration overlap fold and is ablated in the locked configuration; MiniCheck is fine-tuned specifically on language-model-generated claim-support data and produces well-calibrated supported-versus-unsupported probabilities on the distribution of claims the verifier actually scores, whereas generic NLI models trained on MultiNLI and SNLI (including RoBERTa-Large-MNLI~\cite{liu2019roberta}) exhibit systematic miscalibration on language-model hallucinations: \cite{semanticillusion2025} report a one-hundred-percent false-positive rate at ninety-five-percent recall for a near-state-of-the-art generic NLI model on HaluEval~\cite{li-etal-2023-halueval}, corresponding to a verifier balanced accuracy at or below the chance threshold, which would falsify the purity theorem's empirical premise. | ✅ Matched | HaluEval as the benchmark; the attribution is correct. |

## [80] Cost-Effective Hallucination Detection for LLMs
`valentin2024hallucination` · Valentin et al., 2024 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** arXiv 2024 / KDD GenAI-Eval workshop. Compares hallucination-detection scores (calibrated) on TriviaQA, FEVER, HaluEval (summarisation) and BIG-bench. No single score is best. "Multi-scoring" combines scores with logistic regression, plus a cost-aware variant.

**Cited 1 time. Matched 0/1 = 0%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:299` | \cite{valentin2024hallucination} publish the closest analogue: an eight-signal isotonic-stacked hallucination detector on TriviaQA, FEVER, HaluEval, and BIG-bench, with a Spearman cross-signal correlation heat-map that identifies three redundancy clusters among their signals. | ⚠️ Partly | The four datasets match. The scores are combined with logistic regression, not "isotonic stacking". "Eight signals" and the Spearman heat-map with three clusters are unconfirmed. |

## [81] LM-Polygraph: Uncertainty Estimation for Language Models
`vashurin2024polygraph` · Vashurin et al., 2025 · details matched: ❌ No

**What the paper says (paraphrased from its abstract and public record):** TACL 2025. A benchmark of many uncertainty-quantification methods for LLMs, implemented in LM-Polygraph, across eleven tasks, with confidence normalisation.

**Cited 1 time. Matched 0/1 = 0%** (partly 0, not matched 0, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:299` | \cite{vashurin2024polygraph} benchmark twenty-eight uncertainty-estimation methods head-to-head and report that sample-diversity methods such as semantic entropy dominate open-generation tasks such as TriviaQA and CoQA, while verbalised-uncertainty and maximum-softmax-probability methods dominate constrained-output tasks such as MMLU. | 🔒 Needs full text | "Twenty-eight methods" and the per-task winners need the full text. |

## [82] An Empirical Investigation of the Role of Pre-training in Lifelong Learning
`mehta2023empirical` · Mehta et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** JMLR 2023. Generic pretraining implicitly reduces forgetting in lifelong learning, linked to flatter loss basins. Studies sharpness-aware minimisation.

**Cited 1 time. Matched 0/1 = 0%** (partly 0, not matched 0, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:305` | Direct empirical evidence at pre-trained transformer scale shows that the Fisher-weighting advantage of EWC, SI, and MAS over plain $L_2$ collapses once the backbone is a pre-trained language model: \cite{mehta2023empirical} report minimal or negative improvement at BERT-Large scale, and \cite{wu2022pretrained} reproduce the same pattern across five pre-trained backbones, attributed to the flatter loss landscape that pre-training induces. | 🔒 Needs full text | The EWC/SI/MAS vs. L2 comparison at BERT-Large scale needs the full text. The abstract is about pretraining reducing forgetting. |

## [83] Pretrained Language Model in Continual Learning: A Comparative Study
`wu2022pretrained` · Wu et al., 2022 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ICLR 2022. A comparative study of continual-learning methods combined with several pretrained LMs (BERT, ALBERT, RoBERTa, GPT-2, XLNet) on text classification.

**Cited 1 time. Matched 0/1 = 0%** (partly 0, not matched 0, needs full text 1)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:305` | Direct empirical evidence at pre-trained transformer scale shows that the Fisher-weighting advantage of EWC, SI, and MAS over plain $L_2$ collapses once the backbone is a pre-trained language model: \cite{mehta2023empirical} report minimal or negative improvement at BERT-Large scale, and \cite{wu2022pretrained} reproduce the same pattern across five pre-trained backbones, attributed to the flatter loss landscape that pre-training induces. | 🔒 Needs full text | "The same pattern across five backbones" needs the full text. |

## [84] CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge
`talmor-etal-2019-commonsenseqa` · Talmor et al., 2019 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NAACL 2019. CommonsenseQA: 12,247 five-option multiple-choice questions requiring commonsense knowledge (built from ConceptNet).

**Cited 3 times. Matched 3/3 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_2.tex:307` | CommonsenseQA \cite{talmor-etal-2019-commonsenseqa} tests commonsense multi-step reasoning over five-option multiple-choice items and contributes a second held-out test fold to the retention probe panel. | ✅ Matched | Matches: five-option multiple choice. |
| 2 | `chapters/chapter_3.tex:145` | The five evaluation benchmarks are FEVER \cite{thorne-etal-2018-fever}, TriviaQA \cite{joshi-etal-2017-triviaqa}, CommonsenseQA \cite{talmor-etal-2019-commonsenseqa}, TruthfulQA \cite{lin-etal-2022-truthfulqa}, and StrategyQA \cite{geva2021strategyqa}, all of which are distributed under licences that permit research use and redistribution under attribution; the licence terms are catalogued in Appendix~\ref{app:reproducibility} together with the access date and the dataset version hash. | ✅ Matched | Usage; the attribution is correct. |
| 3 | `chapters/chapter_5.tex:41` | The training panel comprises FEVER \cite{thorne-etal-2018-fever} for factual claim verification, TriviaQA \cite{joshi-etal-2017-triviaqa} for open-domain trivia, and CommonsenseQA \cite{talmor-etal-2019-commonsenseqa} for five-option multiple-choice commonsense reasoning. | ✅ Matched | Usage; the attribution is correct. |

## [85] Energy and Policy Considerations for Deep Learning in NLP
`strubell2019energy` · Strubell et al., 2019 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** ACL 2019. Quantifies the energy, carbon and financial cost of training NLP models, using measured power draw and regional emission factors.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_3.tex:112` | The carbon-equivalent footprint per phase is reported by combining the recorded wall-clock duration with the published wattage envelope of the rented hardware and a public grid-emissions conversion factor, following the methodology of \textcite{strubell2019energy}, \begin{equation} \mathrm{kgCO_{2}e}_{\,\phi} \;=\; T_{\phi} \cdot P_{\mathrm{TDP}} \cdot \varepsilon_{\mathrm{grid}}, \label{eq:carbon-per-phase} \end{equation} where $T_{\phi}$ is the wall-clock duration of phase $\phi$ in hours, $P_{\mathrm{TDP}}$ is the rated power of the graphics processor in kilowatts, and $\varepsilon_{\mathrm{grid}}$ is the local grid emissions factor in kilograms of carbon-dioxide equivalent per kilowatt-hour. | ✅ Matched | Matches: power × time × emission factor. |

## [86] Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies
`geva2021strategyqa` · Geva et al., 2021 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** (Same paper as [74].)

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_3.tex:145` | The five evaluation benchmarks are FEVER \cite{thorne-etal-2018-fever}, TriviaQA \cite{joshi-etal-2017-triviaqa}, CommonsenseQA \cite{talmor-etal-2019-commonsenseqa}, TruthfulQA \cite{lin-etal-2022-truthfulqa}, and StrategyQA \cite{geva2021strategyqa}, all of which are distributed under licences that permit research use and redistribution under attribution; the licence terms are catalogued in Appendix~\ref{app:reproducibility} together with the access date and the dataset version hash. | ✅ Matched | Usage (duplicate key). |
| 2 | `chapters/chapter_5.tex:41` | The transfer panel comprises TruthfulQA \cite{lin-etal-2022-truthfulqa} for common-misconception probes and StrategyQA \cite{geva2021strategyqa} for implicit multi-step reasoning. | ✅ Matched | Usage (duplicate key). |

## [87] A Simple Sequentially Rejective Multiple Test Procedure
`holm1979simple` · Holm, 1979 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** 1979. Holm's sequentially rejective multiple-test procedure, which controls the family-wise error rate.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_3.tex:163` | The multiple-comparison convention requires Holm-Bonferroni correction \cite{holm1979simple} across the baseline contrasts within each benchmark family, so that a significance verdict on a single benchmark cannot be inflated by the family-wise error rate; the bootstrap ninety-five-percent confidence interval on the exact-match delta is reported alongside the corrected p-value so that effect size is visible in addition to significance, with paired McNemar \cite{mcnemar1947note} as the per-benchmark significance test. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_5.tex:381` | The eight $p$-values within a benchmark family are jointly corrected for multiple comparison using the Holm-Bonferroni procedure \cite{holm1979simple}, which orders the eight $p$-values in ascending sequence and applies the threshold $\alpha/(k+1-i)$ to the $i$-th smallest, where $k = 8$ and $\alpha = 0.05$. | ✅ Matched | Matches: Holm thresholds α/(k+1−i). |

## [88] Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages
`mcnemar1947note` · McNemar, 1947 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** 1947. McNemar's test for the difference between correlated (paired) proportions.

**Cited 2 times. Matched 2/2 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_3.tex:163` | The multiple-comparison convention requires Holm-Bonferroni correction \cite{holm1979simple} across the baseline contrasts within each benchmark family, so that a significance verdict on a single benchmark cannot be inflated by the family-wise error rate; the bootstrap ninety-five-percent confidence interval on the exact-match delta is reported alongside the corrected p-value so that effect size is visible in addition to significance, with paired McNemar \cite{mcnemar1947note} as the per-benchmark significance test. | ✅ Matched | Matches. |
| 2 | `chapters/chapter_5.tex:366` | Per-benchmark significance against each external baseline is reported through the paired McNemar test \cite{mcnemar1947note} with continuity correction. | ✅ Matched | Matches. The continuity correction is Edwards 1948, but it is standard. |

## [89] Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Signal Approach to Pre-Generation Hallucination Mitigation
`nandakishor2025routing` · Nandakishor, 2025 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** arXiv 2025. Pre-generation confidence routing using semantic alignment, convergence across model layers and learned confidence. Routes queries to local generation, RAG, larger models or human review.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_4.tex:73` | The second is an encoder-layer convergence ratio adapted from \cite{nandakishor2025routing}, computed from the hidden-state stack of the same forward pass as the variance ratio between the early-layer half and the late-layer half of the decoder; this captures how stably the representation has settled across the layers that drive the downstream confidence, with high variance in the early layers relative to the late layers reading as the model not having converged on a single interpretation of the question. | ✅ Matched | Matches: convergence across layers is one of the paper's signals. |

## [90] Reinforced Self-Training (ReST) for Language Modeling
`gulcehre2023rest` · Gulcehre et al., 2023 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** arXiv 2023. ReST: grow a dataset by sampling from the policy, filter by reward, then fine-tune offline, iterating.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_4.tex:473` | It is an instance of the broader family of model-self-training procedures that fine-tune a generator on its own filtered outputs \cite{NEURIPS2022_639a9a17,hosseini2024vstar,gulcehre2023rest,ho2023large}, specialised here through the fixed-threshold storage gate that supplies the filter, the parameter-efficient adapter layer that absorbs the gradient update, the loss-reweighted training pool that prevents single-benchmark dominance, and the multi-modal retention guard that gates the commit. | ✅ Matched | Matches: ReST is self-training on filtered own samples. |

## [91] Large Language Models Are Reasoning Teachers
`ho2023large` · Ho et al., 2023 · details matched: ❌ No

**What the paper says (paraphrased from its abstract and public record):** ACL 2023 (Ho, Schmid, Yun). Fine-tune-CoT: a very large *teacher* model generates reasoning samples, and these are used to fine-tune smaller *student* models.

**Cited 1 time. Matched 0/1 = 0%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_4.tex:473` | It is an instance of the broader family of model-self-training procedures that fine-tune a generator on its own filtered outputs \cite{NEURIPS2022_639a9a17,hosseini2024vstar,gulcehre2023rest,ho2023large}, specialised here through the fixed-threshold storage gate that supplies the filter, the parameter-efficient adapter layer that absorbs the gradient update, the loss-reweighted training pool that prevents single-benchmark dominance, and the multi-modal retention guard that gates the commit. | ⚠️ Partly | Fine-tune-CoT fine-tunes a *student* on a *teacher's* outputs, not a generator on its own outputs. |

## [92] A Stochastic Approximation Method
`robbins1951stochastic` · Robbins & Monro, 1951 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** 1951. The Robbins–Monro stochastic approximation method for finding roots from noisy observations.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_4.tex:572` | The rigorous geometric-envelope claim requires the stochastic-approximation supermartingale machinery of \cite{robbins1951stochastic} and the rate analysis of \cite{polyak1990new} adapted to a bounded gap space; the rigorous supermartingale construction adapted from these works, at the level of a complete proof, is registered as a deferred future-work item in \Cref{ch:conclusion} alongside the at-least-fifteen-committed-cycle horizon precondition that would also unlock the geometric-rate test, and the body claim is therefore stated as an expected-value envelope on the committed-cycle subsequence rather than a sample-path bound on the calendar timeline. | ✅ Matched | Matches. |

## [93] New Stochastic Approximation Type Procedures
`polyak1990new` · Polyak, 1990 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** 1990. New stochastic approximation procedures, including iterate averaging with improved convergence rates.

**Cited 1 time. Matched 1/1 = 100%** (partly 0, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `chapters/chapter_4.tex:572` | The rigorous geometric-envelope claim requires the stochastic-approximation supermartingale machinery of \cite{robbins1951stochastic} and the rate analysis of \cite{polyak1990new} adapted to a bounded gap space; the rigorous supermartingale construction adapted from these works, at the level of a complete proof, is registered as a deferred future-work item in \Cref{ch:conclusion} alongside the at-least-fifteen-committed-cycle horizon precondition that would also unlock the geometric-rate test, and the body claim is therefore stated as an expected-value envelope on the committed-cycle subsequence rather than a sample-path bound on the calendar timeline. | ✅ Matched | Matches. |

## [94] Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift
`ovadia2019trust` · Ovadia et al., 2019 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2019. Benchmarks predictive uncertainty under dataset shift. Calibration degrades with shift, and post-hoc calibration on i.i.d. validation data (e.g. temperature scaling) does not hold up under shift.

**Cited 1 time. Matched 0/1 = 0%** (partly 1, not matched 0, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `appendix/appendix_g.tex:115` | Re-fitting the per-signal isotonic curves on the same (variable signals, constant labels) data is exactly the procedure the post-hoc calibration literature \cite{ovadia2019trust,thulasidasan2019mixup} registers for tracking calibration drift after a model update. | ⚠️ Partly | The paper shows calibration degrades under shift. It does not "register" isotonic refitting after model updates. |

## [95] On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks
`thulasidasan2019mixup` · Thulasidasan et al., 2019 · details matched: ✅ Yes

**What the paper says (paraphrased from its abstract and public record):** NeurIPS 2019. Mixup training improves calibration and predictive uncertainty of deep networks.

**Cited 1 time. Matched 0/1 = 0%** (partly 0, not matched 1, needs full text 0)

| # | Where | Full cited text in the thesis | Verdict | What matched / what didn't |
|---|---|---|---|---|
| 1 | `appendix/appendix_g.tex:115` | Re-fitting the per-signal isotonic curves on the same (variable signals, constant labels) data is exactly the procedure the post-hoc calibration literature \cite{ovadia2019trust,thulasidasan2019mixup} registers for tracking calibration drift after a model update. | ❌ Not matched | The paper is about mixup training improving calibration. It says nothing about refitting post-hoc calibrators. |

