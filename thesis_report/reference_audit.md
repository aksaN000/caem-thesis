# Reference audit — `bibliography/references.bib`

Audit date: 2026-10-14. Scope: all 95 entries, all of which are cited in the thesis (no missing or unused keys).

## How each entry was checked
1. **Does the paper exist, with the right metadata?** Each entry's title, authors, year, venue and pages were compared against web-search results from arXiv, the ACL Anthology, PMLR, NeurIPS, PubMed, ICLR and publisher pages.
2. **Does the paper support the claim?** Every sentence in `chapters/`, `appendix/` and `core/` that contains `\cite` (198 sentences) was compared against the paper's abstract and summaries. Full PDFs were **not** read: direct access to arxiv.org, Crossref, DBLP and Semantic Scholar is blocked in this environment. Entries marked ✅ have correct metadata and a claim that is consistent with the abstract. Specific numbers quoted from deep inside a paper are only confirmed where noted.

Legend: ✅ verified · ⚠️ fixable error (metadata or wording) · ❌ wrong or nonexistent reference, or claim not supported · 🔒 needs the full PDF to confirm a specific number

## Fix first (❌)
1. `semanticillusion2025`: the bib entry describes a paper that does not exist. Replace it with Sinha, arXiv:2512.15068, and reword the claim.
2. `alkaissi2023artificial`: the "69–77% of 25 clinical questions" figure is not in this paper. It appears in the abstract, ch1, ch2 and appendix G. I could not find any source for this number.
3. `huang-etal-2023-large` and `10.1145/3649506` are cited for findings from *other* papers (Jie Huang et al. ICLR 2024, and Lei Huang et al. ACM TOIS). Add those two entries.
4. `cherian2024conformal`: the title is wrong, and "boosted logistic aggregation" overstates the method.
5. Wrong author lists: `anil2023palm`, `ho2023large`, `moskvoretskii2025llm`, `baek2025probing`, `fu2025collapse` (the text's "Fu et al." should be "Yi et al."), `vashurin2024polygraph` (wrong title).
6. Chelli et al.: the GPT-3.5 rate is 39.6%, not 69.6%.

## All entries
Totals: ✅ 56 · ⚠️ 23 · ❌ 11 · 🔒 5 (of 95)

| Key | Title | Status | Issue |
|---|---|---|---|
| `ovadia2019trust` | Can You Trust Your Model's Uncertainty? {E}valuating Predictive Uncertainty Under Dataset Shift | ✅ |  |
| `thulasidasan2019mixup` | On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks | ✅ |  |
| `openai2023gpt4` | {GPT-4} Technical Report | ✅ |  |
| `anthropic2023claude` | Claude 2 | ✅ |  |
| `anil2023palm` | {PaLM} 2 Technical Report | ❌ | Wrong authors. After Anil, the names (Borgeaud, Alayrac, Soricut, Schalkwyk, Hauth, Millican…) are the **Gemini** report's. PaLM 2 is Anil, Rohan and Dai, Andrew M. and Firat, Orhan and … (≈128 authors). Use `Anil, Rohan and Dai, Andrew M. and Firat, Orhan and others`. |
| `lin-etal-2022-truthfulqa` | {T}ruthful{QA}: Measuring How Models Mimic Human Falsehoods | ⚠️ | Metadata OK. Ch1 says the human baseline is 94% "on both" axes. Humans were 94% truthful but lower on truthful+informative. Check the 25% vs. 21% figure against the paper. |
| `min-etal-2023-factscore` | {FA}ct{S}core: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation | ✅ |  |
| `alkaissi2023artificial` | Artificial Hallucinations in {ChatGPT}: Implications in Scientific Writing | ❌ | Real paper, wrong claim. It is a short Cureus editorial in which the authors had ChatGPT write about two clinical cases (homocysteine/osteoporosis and late-onset Pompe disease). It does **not** ask 25 clinical questions or report a 69–77% hallucination rate. That figure is used in the abstract, ch1, ch2 (×2) and appendix G. The ch2 "above 90% in worst-case clinical settings" is Chelli et al.'s Bard figure (91.4%), not Alkaissi's. |
| `info:doi/10.2196/53164` | Hallucination Rates and Reference Accuracy of {ChatGPT} and Bard for Systematic Reviews: Comparative Analysis | ⚠️ | Metadata OK. The text (ch2) says GPT-3.5 = **69.6%**, but the paper reports **39.6%** (55/139); GPT-4 28.6% and Bard 91.4% are correct. These are rates of hallucinated *references* in a systematic-review search task. The author's name is misspelled "Cheli" in the text. |
| `kadavath2022language` | Language Models (Mostly) Know What They Know | ⚠️ | Metadata OK. The text says "at NeurIPS 2022", but it is arXiv-only. In ch1 it is cited for models being overconfident by ≥10 pp, yet Kadavath et al. mainly find large models *well calibrated* on multiple choice. Keep it for P(IK) and the RLHF-miscalibration point only. |
| `NEURIPS2020_6b493230` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | ✅ |  |
| `wang2023selfconsistency` | Self-Consistency Improves Chain of Thought Reasoning in Language Models | ✅ |  |
| `farquhar2024semantic` | Detecting Hallucinations in Large Language Models Using Semantic Entropy | ⚠️ | Metadata OK. Ch1 cites it for "fixed weighted sums used in earlier verifier ensembles"; semantic entropy is not a weighted-sum ensemble. Mis-cite in that one sentence. |
| `manakul-etal-2023-selfcheckgpt` | {S}elf{C}heck{GPT}: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models | ✅ |  |
| `natarajan2013learning` | Learning with Noisy Labels | ✅ |  |
| `NIPS2017_3f5ee243` | Attention is All you Need | ✅ |  |
| `JMLR:v25:23-0870` | Scaling Instruction-Finetuned Language Models | 🔒 | Metadata OK. The "CoT benefits at 780M" claim needs the PDF. |
| `ji2023survey` | Survey of Hallucination in Natural Language Generation | ✅ |  |
| `10.1145/3649506` | Harnessing the Power of {LLMs} in Practice: A Survey on {ChatGPT} and Beyond | ❌ | Metadata OK (Yang et al., ACM **TKDD** 18(6), 2024). Wrong paper for its uses: ch2 credits it with the factuality vs. faithfulness taxonomy and says it is in "ACM Transactions on Information Systems". That is Lei Huang et al., *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions*, ACM TOIS, doi:10.1145/3703155. The abstract's "three to ninety-one percent" is also not from this paper. |
| `10.1007/11736790_9` | The PASCAL Recognising Textual Entailment Challenge | ✅ |  |
| `liu2019roberta` | {RoBERTa}: A Robustly Optimized {BERT} Pretraining Approach | ⚠️ | Metadata OK. The text says 90.8% on MNLI-m **dev**; the RoBERTa paper's dev figure is 90.2 (90.8 is the test-set ensemble). |
| `reimers-gurevych-2019-sentence` | Sentence-{BERT}: Sentence Embeddings using {S}iamese {BERT}-Networks | ✅ |  |
| `8733051` | Billion-Scale Similarity Search with GPUs | ✅ |  |
| `pmlr-v48-gal16` | Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning | ✅ |  |
| `MCCLOSKEY1989109` | Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem | ✅ |  |
| `doi:10.1073/pnas.1611835114` | Overcoming catastrophic forgetting in neural networks | ⚠️ | Metadata OK. "EWC maintains approximately 90% performance across sequential tasks" is not a figure the paper states. Remove it or cite a specific experiment. |
| `graves2014neural` | Neural Turing Machines | ✅ |  |
| `weston2015memory` | Memory Networks | ✅ |  |
| `NIPS2015_8fb21ee7` | End-To-End Memory Networks | ✅ |  |
| `graves2016differentiable` | Hybrid Computing Using a Neural Network with Dynamic External Memory | ✅ |  |
| `pmlr-v70-guo17a` | On Calibration of Modern Neural Networks | ✅ |  |
| `xiong2023can` | Can {LLMs} Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in {LLMs} | ⚠️ | Published at **ICLR 2024**. The claim "any single confidence signal AUROC 0.50–0.60" is roughly supported for *verbalized* confidence (GPT-4 avg 62.7%; white-box 0.52–0.61) but not for every signal. Narrow the wording. |
| `tian2023just` | Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback | ⚠️ | Incomplete or wrong authors. The real authors are Tian, Mitchell, **Zhou, Sharma, Rafailov**, Yao, Finn, Manning. Published at **EMNLP 2023**, pp. 5433–5442; cite that, not arXiv. |
| `NEURIPS2022_639a9a17` | STaR: Bootstrapping Reasoning With Reasoning | ✅ |  |
| `huang-etal-2023-large` | Large Language Models Can Self-Improve | ❌ | Metadata OK, wrong paper for two of its uses. *LLMs Can Self-Improve* (Jiaxin Huang) is cited in ch2 for "self-correction rarely improves accuracy without external feedback". That finding is the opposite paper: Jie Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR 2024 (arXiv:2310.01798). Ch1 also cites it as "survey literature" on hallucination. Add the ICLR 2024 entry and re-point those cites. |
| `kamoi-etal-2024-llms` | When Can {LLM}s Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of {LLM}s | ✅ |  |
| `gururangan-etal-2020-dont` | Don{'}t Stop Pretraining: Adapt Language Models to Domains and Tasks | ✅ |  |
| `yang-etal-2018-hotpotqa` | {H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering | ✅ |  |
| `10.1162/tacl_a_00370` | Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies | ⚠️ | **Duplicate** of `geva2021strategyqa`, so StrategyQA is printed twice in the References. Keep one key and replace the other cites. |
| `thorne-etal-2018-fever` | {FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification | ✅ |  |
| `gulcehre2023rest` | Reinforced Self-Training ({ReST}) for Language Modeling | ⚠️ | Real. `@article` with an arXiv "journal" should be `@misc`. |
| `ho2023large` | Large Language Models Are Reasoning Teachers | ❌ | Wrong authors. The real authors are **Ho, Namgyu and Schmid, Laura and Yun, Se-Young** (the bib lists Salimans, Gritsenko, Chan, Sohl-Dickstein, Ermon). Pages are 14852–14882 (bib: 14873). |
| `nandakishor2025routing` | Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Signal Approach to Pre-Generation Hallucination Mitigation | ⚠️ | Real (single-author arXiv:2510.01237, Sept 2025), but not peer-reviewed. Use `@misc`. |
| `joshi-etal-2017-triviaqa` | {T}rivia{QA}: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension | ✅ |  |
| `kwiatkowski-etal-2019-natural` | Natural Questions: A Benchmark for Question Answering Research | ✅ |  |
| `li-etal-2023-halueval` | {H}alu{E}val: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models | ✅ |  |
| `clark2018think` | Think you have Solved Question Answering? Try {ARC | ✅ |  |
| `hendrycks2021measuring` | Measuring Massive Multitask Language Understanding | ✅ |  |
| `mcnemar1947note` | Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages | ✅ |  |
| `holm1979simple` | A Simple Sequentially Rejective Multiple Test Procedure | ✅ |  |
| `karpukhin-etal-2020-dense` | Dense Passage Retrieval for Open-Domain Question Answering | ✅ |  |
| `jiang-etal-2023-active` | Active Retrieval Augmented Generation | ✅ |  |
| `asai2024selfrag` | Self-{RAG}: Learning to Retrieve, Generate, and Critique through Self-Reflection | ✅ |  |
| `wei2022chain` | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | ✅ |  |
| `tang2024minicheck` | {MiniCheck}: Efficient Fact-Checking of {LLM}s on Grounding Documents | ⚠️ | Wrong venue: it is **EMNLP 2024** (main), pp. 8818–8847, not ACL 2024. |
| `semanticillusion2025` | Semantic Illusion: Generic Natural Language Inference Models Miscalibrate on Language Model Hallucinations | ❌ | No such paper. Placeholder entry ("Anon.", "Working paper"). The closest real paper is Sinha, *The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems*, arXiv:2512.15068 (2025). Its 100% FPR on HaluEval is reported for **embedding/cross-encoder** detectors, and it gives NLI models AUC 0.81, so the ch2 sentence about a "generic NLI model" needs rewording too. |
| `wu2022pretrained` | Pretrained Language Model in Continual Learning: A Comparative Study | 🔒 | Metadata OK. The "same pattern across five backbones" claim needs the PDF. |
| `mehta2023empirical` | An Empirical Investigation of the Role of Pre-training in Lifelong Learning | 🔒 | Metadata OK (JMLR 24, 2023). The EWC/SI/MAS-vs-L2 claim needs the PDF. |
| `biderman2024lora` | {LoRA} Learns Less and Forgets Less | 🔒 | Metadata OK (TMLR is a journal, so use `@article`). The 66% vs. 62% retention numbers need the PDF. |
| `wang2023olora` | Orthogonal Subspace Learning for Language Model Continual Learning | ✅ |  |
| `valentin2024hallucination` | Cost-Effective Hallucination Detection for {LLM}s | ⚠️ | Real; also a KDD 2024 GenAI-Evaluation workshop paper. `@inproceedings` with an arXiv booktitle should be `@misc`. Datasets (TriviaQA, FEVER, HaluEval, BIG-bench) are correct, but the scores are combined with **logistic regression**. "Eight-signal isotonic-stacked" and the "Spearman heat-map with three clusters" are not confirmed. |
| `vashurin2024polygraph` | {LM-Polygraph}: Uncertainty Estimation for Language Models | ❌ | Wrong title. The 2025 TACL paper is *Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph*, TACL 13:220–248, doi:10.1162/tacl_a_00737 (the bib's title is the 2023 EMNLP-demo paper by Fadeeva et al.). Use `@article`. The "twenty-eight methods" and per-task-winner claims are unconfirmed. |
| `kuhn2023semantic` | Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation | ✅ |  |
| `qwen25` | Qwen2.5: A Party of Foundation Models | ✅ |  |
| `geva2021strategyqa` | Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies | ⚠️ | Duplicate of `10.1162/tacl_a_00370` (see above). |
| `strubell2019energy` | Energy and Policy Considerations for Deep Learning in {NLP} | ✅ |  |
| `niculescu2005predicting` | Predicting Good Probabilities With Supervised Learning | ✅ |  |
| `cherian2024conformal` | Enhanced Conformal Prediction with Boosted Logistic Regression | ❌ | **Wrong title.** The real title is *Large Language Model Validity via Enhanced Conformal Prediction Methods* (Cherian, Gibbs, Candès, NeurIPS 2024). Claim: the paper's "conditional boosting" optimises the conformity score by differentiating through the conditional-conformal procedure. It is not a logistic regression over per-signal log-odds, so "boosted logistic aggregation that CAEM adopts" overstates it. Reword to "inspired by". |
| `mohri2024conformal` | Language Models with Conformal Factuality Guarantees | 🔒 | Metadata OK. The 78% NQ starting point is confirmed; the "→93% at α=0.2 with GPT-4" figure needs the PDF. |
| `yadkori2024abstention` | Mitigating {LLM} Hallucinations via Conformal Abstention | ⚠️ | Real. `@inproceedings` with an arXiv booktitle should be `@misc`. |
| `quach2024conformal` | Conformal Language Modeling | ✅ |  |
| `hosseini2024vstar` | {V-STaR}: Training Verifiers for Self-Taught Reasoners | ✅ |  |
| `fu2025collapse` | Escaping Model Collapse via Synthetic Data Verification: Near-Term Improvement and Long-Term Convergence | ❌ | Wrong authors. The real authors are **Yi, Bingji and Liu, Qiyuan and Cheng, Yuwei and Xu, Haifeng**, so the text's "Fu et al." should be "Yi et al.". The title ends "Near-term **Improvements** and Long-term Convergence". Accepted at ICLR 2026. Claim: the paper finds verifier-filtered retraining gives near-term gains but converges to the *verifier's* knowledge, and the gains plateau unless the verifier is perfect. "Prevents collapse with long-term guarantees" and "precision-bounded rather than recall-bounded" overstate it. |
| `song2024gap` | Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models | ⚠️ | Metadata OK; now published at **ICLR 2025**. Claim: the text says they "prove empirically that self-improvement succeeds **if and only if**" the verifier beats the generator. The paper is an empirical study of the generation–verification gap and how it scales, with no iff result. Reword. |
| `das2025retraining` | Retraining with Predicted Hard Labels Provably Increases Model Accuracy | ⚠️ | Missing authors: add Mao, Jieming; Sanghavi, Sujay; Zhong, Peilin (8 total). PMLR 267:12509–12538. Claim: the paper retrains on the model's *own predicted* labels (optionally keeping only those that agree with the given label) under label noise, not on "verifier-correct" labels. Soften the wording. |
| `huang2025sharpening` | Self-Improvement in Language Models: The Sharpening Mechanism | ⚠️ | "Foster, Dean" → **Foster, Dylan J.** |
| `lang2024w2s` | Theoretical Analysis of Weak-to-Strong Generalization | ⚠️ | Author order is Lang, **Sontag, Vijayaraghavan**; the text also lists the wrong order. |
| `jeong2024adaptive` | Adaptive-{RAG}: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity | ✅ |  |
| `chen2024inside` | {INSIDE}: {LLMs}' Internal States Retain the Power of Hallucination Detection | ⚠️ | "Wu, Yufei" → **Wu, Yue**. The "AUROC near 0.84 on TriviaQA" figure is not confirmed (needs the PDF). |
| `nikitin2024kle` | Kernel Language Entropy: Fine-grained Uncertainty Quantification for {LLMs} from Semantic Similarities | ⚠️ | "Janson, Pekka" → **Marttinen, Pekka**. NeurIPS 2024, vol. 37, pp. 8901–8929. |
| `kamath2020selective` | Selective Question Answering under Domain Shift | ✅ |  |
| `geng2024survey` | A Survey of Confidence Estimation and Calibration in Large Language Models | ⚠️ | Metadata OK (pp. 6577–6595). The claim that it "confirms no single method exceeds AUROC ≈0.6" was not found in the survey and conflicts with your own INSIDE ≈0.84 sentence. Remove or source it. |
| `larimar2024das` | Larimar: Large Language Models with Episodic Memory Control | ⚠️ | Missing author Navratil, Jiri. It is ICML 2024 (PMLR 235:10109–10126), so use `@inproceedings`. |
| `xu2025amem` | {A-MEM}: Agentic Memory for {LLM} Agents | ✅ | Real. Check whether you want to cite the later peer-reviewed version. |
| `robbins1951stochastic` | A Stochastic Approximation Method | ✅ |  |
| `polyak1990new` | New Stochastic Approximation Type Procedures | ✅ |  |
| `hu2022lora` | {LoRA}: Low-Rank Adaptation of Large Language Models | ✅ |  |
| `geifman2017selective` | Selective Classification for Deep Neural Networks | ✅ |  |
| `dettmers2023qlora` | {QLoRA}: Efficient Finetuning of Quantized {LLMs} | ✅ |  |
| `kalajdzievski2023rslora` | A Rank Stabilization Scaling Factor for Fine-Tuning with {LoRA} | ✅ |  |
| `talmor-etal-2019-commonsenseqa` | {CommonsenseQA}: A Question Answering Challenge Targeting Commonsense Knowledge | ✅ |  |
| `mallen2023whennot` | When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories | ✅ |  |
| `wang2023skr` | Self-Knowledge Guided Retrieval Augmentation for Large Language Models | ✅ |  |
| `baek2025probing` | Probing-{RAG}: Self-Probing to Guide Language Models in Selective Document Retrieval | ❌ | Wrong authors. The real authors are **Baek, Ingeol and Chang, Hwan and Kim, ByeongJeong and Lee, Jimin and Lee, Hwanhee** (Chung-Ang Univ.), not Jinheon Baek et al. Findings of NAACL 2025, pp. 3287–3304. The "linear probe" and "enforced 50/50 class ratio" details are not confirmed. |
| `moskvoretskii2025llm` | {LLM}-Independent Adaptive {RAG}: Let the Question Speak for Itself | ❌ | Wrong authors. The real authors are Marina, Ivanov, Pletenev, Salnikov, Galimzianova, Krayko, Konovalov, Panchenko, **Moskvoretskii (last)**. Published at **EMNLP 2025** (pp. 8697–8709). The text says "Moskvoretskii et al."; it should say "Marina et al.". |
