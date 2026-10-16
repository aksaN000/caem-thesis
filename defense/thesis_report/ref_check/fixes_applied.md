# Citation and reference fixes applied (tiers A and B)

Applied 2026-10-15 on branch `audit/thesis-consistency`. The evidence for every change is in `reference_check_2_claims.md` (Parts 1–3).

**How this was checked.** LaTeX is not installed in this environment, so the thesis was not compiled. Instead:
- `bibliography/references.bib` was parsed with pybtex: 99 entries, no duplicate keys, braces balanced.
- Every `\cite` key in the abstract, chapters and appendices (96 distinct keys) resolves to a bib entry.
- Every edited `.tex` file keeps balanced braces and `$` delimiters.
- No non-ASCII characters or unescaped `%` were introduced.

**Please compile once** (`latexmk -pdf main.tex` with biber) to confirm the build and the new reference numbering.

## Tier A: bibliography (`bibliography/references.bib`)

| Entry | Change |
|---|---|
| `semanticillusion2025` | **Removed** (fake placeholder). Replaced by `sinha2025semantic`: D. Sinha, *The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems*, arXiv:2512.15068. |
| `geva2021strategyqa` | **Removed** (duplicate of `10.1162/tacl_a_00370`); its two citations now use `10.1162/tacl_a_00370`. |
| `cherian2024conformal` | Title corrected to *Large Language Model Validity via Enhanced Conformal Prediction Methods*. |
| `anil2023palm` | Author list corrected (Anil, Dai, Firat, Johnson, Lepikhin, Passos, Shakeri, Taropa, Bailey, Chen, and others). |
| `ho2023large` | Authors corrected to Ho, Schmid, Yun; pages 14852–14882. |
| `moskvoretskii2025llm` | Authors corrected (Marina, Ivanov, Pletenev, Salnikov, Galimzianova, Krayko, Konovalov, Panchenko, Moskvoretskii); now EMNLP 2025, pp. 8697–8709. |
| `baek2025probing` | Authors corrected to Ingeol Baek, Chang, Kim, Lee, Lee; pages 3287–3304. |
| `fu2025collapse` | Authors corrected to Yi, Liu, Cheng, Xu; title corrected; ICLR 2026. The key is unchanged; the text now says "Yi et al.". |
| `vashurin2024polygraph` | Title corrected to *Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph*; full author list; TACL 13:220–248, doi:10.1162/tacl_a_00737. |
| `tian2023just` | Full author list (8 authors); EMNLP 2023, pp. 5433–5442. |
| `tang2024minicheck` | Venue corrected to EMNLP 2024, pp. 8818–8847. |
| `huang2025sharpening` | "Foster, Dean" → Foster, Dylan J. |
| `nikitin2024kle` | "Janson, Pekka" → Marttinen, Pekka. |
| `chen2024inside` | "Wu, Yufei" → Wu, Yue. |
| `das2025retraining` | Added Mao, Sanghavi, Zhong (8 authors); PMLR 267:12509–12538. |
| `larimar2024das` | Added Navrátil; first name Sarathkrishna; `@inproceedings`, PMLR 235. |
| `lang2024w2s` | Author order corrected to Lang, Sontag, Vijayaraghavan (also in the chapter 2 text). |
| `NIPS2015_8fb21ee7` | "szlam, arthur" → Szlam, Arthur. |
| `xiong2023can`, `song2024gap` | Now cite the published ICLR 2024 / ICLR 2025 versions. |
| `valentin2024hallucination`, `yadkori2024abstention`, `gulcehre2023rest`, `nandakishor2025routing` | Entry type corrected to `@misc` (arXiv preprints). |
| `biderman2024lora` | Entry type corrected to `@article` (TMLR is a journal). |
| **New** `huang2025survey` | Lei Huang et al., *A Survey on Hallucination in LLMs*, ACM TOIS 43(2), 2025, doi:10.1145/3703155. Now cited for the factuality/faithfulness taxonomy (PDF read). |
| **New** `huang2024cannot` | Jie Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR 2024. Now cited for the self-correction claim (PDF read). |
| **New** `kojima2022large` | Kojima et al., *Large Language Models are Zero-Shot Reasoners*, NeurIPS 2022. Now cited for "Let's think step by step" (PDF read). |
| **New** `robbins1971convergence` | Robbins & Siegmund 1971, almost-supermartingale convergence theorem. Added from the standard bibliographic record; **PDF not checked**. |
| **New** `naeini2015obtaining` | Naeini, Cooper & Hauskrecht, AAAI 2015 (origin of ECE). The details match the entry in Guo et al.'s reference list; **PDF not checked**. |

**No longer cited.** Every use of these three entries was a wrong attribution, so they no longer appear in the printed References. The entries remain in the .bib file and can be deleted.
- `10.1145/3649506` (Yang et al., TKDD survey)
- `huang-etal-2023-large` (Jiaxin Huang, *LLMs Can Self-Improve*)
- `thulasidasan2019mixup`

**Numbering.** Printed reference numbers will shift. The references are in first-citation order, and entries were added, removed or merged.

## Tier B: wording fixes in the thesis text

Each row is an exact replacement. The *Before* text is what the thesis said; the *After* text is now in the file.

| # | File | Before | After |
|---|---|---|---|
| 1 | `core/abstract.tex` | produce confident but factually incorrect outputs at rates that span three to ninety-one percent across deployment domains \cite{alkaissi2023artificial,10.1145/3649506}, a failure mode | produce confident but factually incorrect outputs across deployment domains: asked for supporting literature, chatbots return references that do not exist \cite{alkaissi2023artificial}, for between $28.6$ and $91.4$ percent of the references they generate in systematic-review searches \cite{info:doi/10.2196/53164}, a failure mode |
| 2 | `core/abstract.tex` | regularised logistic boost \cite{cherian2024conformal}) | regularised logistic boost over the per-signal log-odds \cite{valentin2024hallucination}) |
| 3 | `chapters/chapter_1.tex` | achieved only fifty-eight percent on the truthfulness axis and twenty-five percent on the combined truthful-and-informative axis, against a human baseline of ninety-four percent on both \cite{lin-etal-2022-truthfulqa} | achieved only fifty-eight percent on the truthfulness axis and twenty-one percent on the combined truthful-and-informative axis, against a human baseline of ninety-four and eighty-seven percent respectively \cite{lin-etal-2022-truthfulqa} |
| 4 | `chapters/chapter_1.tex` | Clinical-question studies of frontier models report hallucination rates between sixty-nine and seventy-seven percent of responses \cite{alkaissi2023artificial}, an order of magnitude above any threshold a responsible deployment would tolerate. | Asked to support a clinical explanation with references, ChatGPT returned five citations of which none existed \cite{alkaissi2023artificial}, and a study of systematic-review searches found that $39.6$, $28.6$ and $91.4$ percent of the references produced by GPT-3.5, GPT-4 and Bard respectively were hallucinated \cite{info:doi/10.2196/53164}, far above any threshold a responsible deployment would tolerate. |
| 5 | `chapters/chapter_1.tex` | These results establish that hallucination is neither an artefact of small models nor a residual that scale will eliminate; it is, as the recent survey literature argues, a structural property of stateless generation under uncertainty rather than a defect to be trained away \cite{ji2023survey,huang-etal-2023-large}. Any system that aims to reduce it must change the structure rather than merely train harder. | These results indicate that hallucination is neither an artefact of small models nor a residual that scale alone will eliminate; the survey literature describes it as an artefact of neural generation that persists in large, instruction-tuned and preference-tuned models \cite{ji2023survey,huang2025survey}. This motivates changing the structure of the system around the model rather than relying on further training alone. |
| 6 | `chapters/chapter_1.tex` | Calibration studies report that both classifiers and language models assign high probability to incorrect outputs at rates that exceed their actual accuracy by ten percentage points or more \cite{pmlr-v70-guo17a, kadavath2022language}. Single-signal correctness predictors achieve area-under-the-receiver-operating-curve scores between zero point five zero and zero point six zero, only marginally above random chance \cite{xiong2023can}. A user therefore cannot reliably tell which answers are likely to be wrong from confidence alone. | Calibration studies report that modern neural classifiers assign confidence that exceeds their actual accuracy, with expected calibration errors above ten percentage points on standard image benchmarks \cite{pmlr-v70-guo17a}, and that language models fine-tuned with reinforcement learning from human feedback appear badly miscalibrated until their outputs are rescaled \cite{kadavath2022language}. The confidence a model states in words discriminates correct from incorrect answers only weakly: for GPT-4 it averages an area under the receiver operating curve of about zero point six three, close to the zero point five of random guessing, and no single elicitation method is consistently best \cite{xiong2023can}. A user therefore cannot reliably tell which answers are likely to be wrong from the model's stated confidence alone. |
| 7 | `chapters/chapter_1.tex` | What is not present is the discipline of distinguishing facts that the model has correctly internalised from confabulations that emerge under genuine uncertainty, and the discipline of accumulating verified knowledge across interactions instead of regenerating answers from scratch every time \cite{farquhar2024semantic}. | What is not present is the discipline of distinguishing facts that the model has correctly internalised from confabulations that emerge under genuine uncertainty \cite{farquhar2024semantic}, and the discipline of accumulating verified knowledge across interactions instead of regenerating answers from scratch every time. |
| 8 | `chapters/chapter_1.tex` | students who learn from confidently incorrect tutoring acquire misconceptions that persist after the source has been corrected. | students who learn from confidently incorrect tutoring risk acquiring misconceptions. |
| 9 | `chapters/chapter_1.tex` | In healthcare information services, hallucinated drug interactions or treatment dosages produce decisions | In healthcare information services, a hallucinated drug interaction or treatment dosage would produce decisions |
| 10 | `chapters/chapter_1.tex` | registered to run for up to ten cycles in line with the equilibrium horizons reported in continual-learning studies, with | registered to run for up to ten cycles, with |
| 11 | `chapters/chapter_1.tex` | the realised trajectory closes at the cycle-five close under that gate. | the realised trajectory closes at the cycle-five close under that gate, consistent with reports that iterative self-improvement saturates within a few rounds \cite{song2024gap}. |
| 12 | `chapters/chapter_1.tex` | exceeds their actual accuracy by a wide margin \cite{pmlr-v70-guo17a, kadavath2022language}, and single-signal correctness predictors operate close to random on the discrimination axis \cite{xiong2023can}. | exceeds their actual accuracy by a wide margin \cite{pmlr-v70-guo17a}, and the confidence a model verbalises operates close to random on the discrimination axis \cite{xiong2023can}. |
| 13 | `chapters/chapter_1.tex` | The composite replaces the fixed weighted sums used in earlier verifier ensembles \cite{farquhar2024semantic}. | The learned aggregation follows the multi-scoring approach of \cite{valentin2024hallucination}, which combines the logits of several hallucination scores through logistic regression, in place of a fixed weighted sum. |
| 14 | `chapters/chapter_2.tex` | which introduce information that goes beyond what the source provides, regardless of whether it also happens to conflict with the source \cite{ji2023survey}. | which introduce information that can be neither supported nor contradicted by the source \cite{ji2023survey}. |
| 15 | `chapters/chapter_2.tex` | which contradict user input or task context \cite{10.1145/3649506}. If a model claims "photosynthesis produces oxygen from carbon dioxide and water" and demonstrates factual accuracy. But on the other hand it it claims "photosynthesis occurs in mitochondria" then this demonstrates factual hallucination. | which contradict user input or task context \cite{huang2025survey}. For example, a model that states ``photosynthesis produces oxygen from carbon dioxide and water'' is factually accurate, whereas one that states ``photosynthesis occurs in mitochondria'' produces a factuality hallucination. |
| 16 | `chapters/chapter_2.tex` | Natural Language Inference (NLI) is the task of determining whether a hypothesis sentence logically follows from a premise sentence \cite{10.1007/11736790_9}. Given premise $P$ and hypothesis $H$, an NLI model classifies the relationship as: | Natural Language Inference (NLI) is the task of determining whether a hypothesis sentence logically follows from a premise sentence \cite{10.1007/11736790_9}. The PASCAL challenge framed this as a binary decision (entailed or not entailed); later NLI corpora, including the Multi-Genre NLI corpus on which RoBERTa-Large-MNLI is fine-tuned, use three classes. Given premise $P$ and hypothesis $H$, a three-class NLI model classifies the relationship as: |
| 17 | `chapters/chapter_2.tex` | reaching approximately 90.8\% on the MNLI matched development set | reaching approximately 90.2\% on the MNLI matched development set |
| 18 | `chapters/chapter_2.tex` | then $P$ and $H$ are considered semantically equivalent. | then $P$ and $H$ are considered semantically equivalent \cite{kuhn2023semantic}. |
| 19 | `chapters/chapter_2.tex` | produces encasings or embeddings by fine tuning BERT in a siamese network structure on semantic textual similarity tasks. The model we employ, all mpnet base v2, generates 768-dimensional embeddings. | produces sentence embeddings by fine-tuning BERT in a siamese network structure, first on natural language inference data (SNLI and MultiNLI) and optionally on semantic textual similarity data. The model we employ, all-mpnet-base-v2 from the same Sentence-Transformers family (built on an MPNet backbone rather than the original BERT), generates 768-dimensional embeddings. |
| 20 | `chapters/chapter_2.tex` | Fine-tuning LLMs on domain-specific data can reduce accuracy on general benchmarks by 30--50\% without mitigation strategies. | In large language models, fine-tuning on a new domain likewise erodes performance outside that domain, and the erosion grows with training duration \cite{biderman2024lora}. |
| 21 | `chapters/chapter_2.tex` | A cheaper first-order approximation replaces the full Fisher matrix with an isotropic $L_2$ anchor, $\frac{\lambda}{2}\lVert \boldsymbol{\theta} - \boldsymbol{\theta}_{\text{prev}} \rVert^2$, applied to the adapter parameters of the previous cycle rather than to the full backbone. | A cheaper alternative replaces the full Fisher matrix with an isotropic $L_2$ anchor, $\frac{\lambda}{2}\lVert \boldsymbol{\theta} - \boldsymbol{\theta}_{\text{prev}} \rVert^2$, applied to the adapter parameters of the previous cycle rather than to the full backbone; this is the uniform-Fisher special case, which Kirkpatrick et al. show to be weaker than EWC because it protects all weights equally and leaves little capacity for the new task \cite{doi:10.1073/pnas.1611835114}. |
| 22 | `chapters/chapter_2.tex` | retains general capability across iterative cycles substantially better than full-parameter fine-tuning at matched training budget on the empirical evidence reviewed in | forgets less of the base model's general capability than full-parameter fine-tuning on the empirical evidence reviewed in |
| 23 | `chapters/chapter_2.tex` | Early studies on hallucination shows summarization and data to text generation. Ji et al. \cite{ji2023survey} published the first major survey in ACM Computing Surveys. It establishes a study differentiating intrinsic hallucinations that contradicts source material from extrinsic hallucinations. Even though the claims were unverified. Their work shows that hallucinations occurs at several stages. It includes data collection | Early studies of hallucination focused on summarization and data-to-text generation. Ji et al. \cite{ji2023survey} published the first major survey in ACM Computing Surveys. It distinguishes intrinsic hallucinations, which contradict the source material, from extrinsic hallucinations, which can be neither supported nor contradicted by it. Their work shows that hallucinations arise at several stages, including data collection |
| 24 | `chapters/chapter_2.tex` | Huang et al. \cite{10.1145/3649506} extended this study specifically for large language models in ACM Transactions on Information Systems. It is extended by adding the differencs between factuality and faithfulness. Their analysis of causes across the model lifecycle suggests various groundbreaking research. It suggests that effective mitigation should have intervention at several stages instead of relying solely on detection applied after the model has already produced an answer. This directly supports CAEM's multi component strategy which combines memory, verification, and iterative learning. | Huang et al. \cite{huang2025survey} extended this study specifically for large language models in ACM Transactions on Information Systems, distinguishing factuality hallucinations from faithfulness hallucinations. Their analysis traces the causes of hallucination across the model lifecycle to data, training and inference, and ties each family of mitigation methods to the cause it addresses. This suggests that effective mitigation should intervene at several stages instead of relying solely on detection applied after the model has already produced an answer, which supports CAEM's multi-component strategy combining memory, verification, and iterative learning. |
| 25 | `chapters/chapter_2.tex` | Quantitative measurement studies of hallucination prevalence reveals concerning | Quantitative measurement studies of hallucination prevalence reveal concerning |
| 26 | `chapters/chapter_2.tex` | Their result demonstrates an inverse scaling effect: GPT-3 175B achieved only 21-25\% on the truthful and informative metric (or 58\% on truthfulness alone) compared to 94\% human performance. Larger models often performed worse than smaller ones. This proves that only scaling is not sufficient for truthfulness and thus it needs strong motivation for architectural innovation. | Their result demonstrates an inverse scaling effect: the best GPT-3 175B configuration achieved only 21\% on the truthful-and-informative metric (and 58\% on truthfulness alone), against 87\% and 94\% respectively for the human baseline. Larger models often performed worse than smaller ones. This suggests that scaling alone is not sufficient for truthfulness, which motivates architectural innovation. |
| 27 | `chapters/chapter_2.tex` | Alkaissi and McFarlane \cite{alkaissi2023artificial} asked ChatGPT to respond to 25 clinical questions and found that 69--77\% contained hallucinated medical information. Cheli et al.\ \cite{info:doi/10.2196/53164} studied hallucination in medical-literature queries and reported rates of 69.6\% for GPT-3.5, 28.6\% for GPT-4, and 91.4\% for Google Bard. | Alkaissi and McFarlane \cite{alkaissi2023artificial} asked ChatGPT to explain a clinical finding and to supply supporting references; none of the five paper titles it returned existed. Chelli et al.\ \cite{info:doi/10.2196/53164} checked the references that chatbots generated for systematic reviews and reported hallucination rates of 39.6\% for GPT-3.5, 28.6\% for GPT-4, and 91.4\% for Google Bard. |
| 28 | `chapters/chapter_2.tex` | improves reasoning quality by prompting models to generate the next steps before final answers. Showing these intermediate steps leads to stronger reasoning and provides better starting points for verification. When tested on the PaLM 540B model, chain of thought greatly improved performance on math reasoning tasks, increasing accuracy on GSM8K from about eighteen percent to over fifty eight percent. The authors found that this technique showed an emergent ability threshold around 100 billion parameters for \textit{base models} trained on raw text, below which chain-of-thought provided minimal benefit. However, this threshold does not apply to instruction-tuned models, which learn to follow step-by-step reasoning patterns during the instruction-tuning phase itself. Chung et al.\ \cite{JMLR:v25:23-0870} demonstrated that Flan-T5 models show substantial chain-of-thought benefits at 780 million parameters after instruction tuning, confirming that the threshold does not apply to instruction-tuned backbones. Since CAEM uses Qwen-2.5-3B-Instruct~\cite{qwen25}, which is a decoder-only instruction-tuned model at comparable scale, chain-of-thought prompting is effective despite being well below the 100B base-model threshold. | improves reasoning quality by prompting models with worked examples whose intermediate steps precede the final answer. Showing these intermediate steps leads to stronger reasoning and provides better starting points for verification. When tested on the PaLM 540B model, chain of thought greatly improved performance on math reasoning tasks, increasing accuracy on GSM8K from about eighteen percent to about fifty-seven percent. The authors found that the technique is an emergent ability of model scale: below roughly 100 billion parameters chain-of-thought prompting did not help and could lower performance, a pattern that held for the instruction-tuned InstructGPT models in their GPT-3 series as well. Kojima et al.~\cite{kojima2022large} later showed that the single trigger ``Let's think step by step'' elicits chain-of-thought reasoning without worked examples. Instruction tuning does not remove the small-scale effect: Chung et al.\ \cite{JMLR:v25:23-0870} report that their instruction-tuned Flan-T5-Large (780 million parameters) still scores lower with chain-of-thought prompting than with direct answering on MMLU (40.5 versus 45.1 percent) and on BIG-Bench Hard (31.5 versus 37.5 percent). Whether chain-of-thought prompting helps CAEM's three-billion-parameter instruction-tuned backbone, Qwen-2.5-3B-Instruct~\cite{qwen25}, is therefore an empirical question, which the chain-of-thought baseline in Chapter~5 answers. |
| 29 | `chapters/chapter_2.tex` | The key idea is that difficult and complex problems usually has multiple valid reasoning paths which leads to the same correct answer. | The key idea is that difficult and complex problems usually have multiple valid reasoning paths that lead to the same correct answer. |
| 30 | `chapters/chapter_2.tex` | Quantitative improvements with PaLM 540B demonstrate substantial gains: +17.9\% on GSM8K, +11.0\% on SVAMP, +12.2\% on AQuA, and +6.4\% on StrategyQA. | Reported gains over chain-of-thought prompting reach +17.9 percentage points on GSM8K, +11.0 on SVAMP, +12.2 on AQuA, and +6.4 on StrategyQA; these are the largest gains across the four models tested, and for PaLM 540B the corresponding gains are +17.9, +7.6, +12.5 and +6.3. |
| 31 | `chapters/chapter_2.tex` | introduced the FEVER dataset at NAACL 2018 which formulates fact verification as NLI with 185,445 claims generated from Wikipedia. | introduced the FEVER dataset at NAACL 2018, a fact-verification task closely related to NLI, with 185,445 claims generated from Wikipedia; unlike NLI, the evidence for each claim must first be retrieved from Wikipedia rather than being given. |
| 32 | `chapters/chapter_2.tex` | Cherian et al.~\cite{cherian2024conformal} sharpen the Mohri--Hashimoto framework with a boosted logistic aggregation over multiple per-claim scores, and supply the algorithmic basis for CAEM's boost layer over the per-signal log-odds in \Cref{sec:composite-boost}. | Cherian et al.~\cite{cherian2024conformal} sharpen the Mohri--Hashimoto framework with conditional guarantees and a boosted score, learning a linear combination of several per-claim scores by differentiating through the conformal procedure; the closest precedent for CAEM's boost layer over the per-signal log-odds in \Cref{sec:composite-boost} is the logistic-regression multi-scoring of Valentin et al.~\cite{valentin2024hallucination}. |
| 33 | `chapters/chapter_2.tex` | They showed modern deep networks predict confidence scores substantially which exceeds actual accuracy rates. For example, predictions at 80\% confidence may achieve only 60\% accuracy and create a  dangerous miscalibration for decision making. They introduced Expected Calibration Error (ECE) as the standard metric, dividing | They showed that modern deep networks predict confidence scores that substantially exceed their actual accuracy; a ResNet-110 on CIFAR-100, for example, has an expected calibration error of 16.5 percentage points before calibration. They popularised the Expected Calibration Error (ECE) of Naeini et al.~\cite{naeini2015obtaining} as the standard metric, dividing |
| 34 | `chapters/chapter_2.tex` | explored self evaluation in large language models at NeurIPS 2022. | explored self-evaluation in large language models in a 2022 preprint. |
| 35 | `chapters/chapter_2.tex` | And, P(IK) or "I Know" prompts models to predict confidence before generating answers. | Second, P(IK), the probability that ``I know'' the answer, is predicted by a model trained for the purpose before any answer is proposed. |
| 36 | `chapters/chapter_2.tex` | However, reinforcement learning from human feedback can make calibration worse, since it often rewards answers that sound helpful or confident rather than ones that are strictly accurate. | However, policies fine-tuned with reinforcement learning from human feedback appear badly miscalibrated, because reinforcement learning collapses predictions toward the behaviours that receive the most reward, although a single temperature adjustment largely restores calibration. |
| 37 | `chapters/chapter_2.tex` | Their systematic evaluation across multiple benchmarks reveals striking findings. That is, even the strongest single methods only reached about 0.50 to 0.60 AUROC for telling correct answers from incorrect ones, which is just slightly better than random guessing. | Their systematic evaluation across multiple benchmarks reveals striking findings. Verbalised confidence is overconfident and discriminates poorly: even for GPT-4 its average AUROC for telling correct answers from incorrect ones is about 0.63, close to the 0.5 of random guessing. Sampling-consistency signals do much better on some tasks (up to 0.93 AUROC on GSM8K), but no single method consistently outperforms the others, and all of them struggle on tasks that require professional knowledge. |
| 38 | `chapters/chapter_2.tex` | CAEM's multi-signal approach is influenced by this limited discriminative ability of any individual signal. | CAEM's multi-signal approach is influenced by this finding that no individual signal is reliable across tasks. |
| 39 | `chapters/chapter_2.tex` | Recent work by Tian et al. \cite{tian2023just} questions if models genuinely know when they don't know. It also questions if the models only match patterns for uncertain expressions from training data.Their study found that verbalized confidence often reflects patterns in the training data rather than the model's actual knowledge or uncertainty. This reinforces CAEM's strategy of combining implicit signals like token probability, dropout, and self consistency with explicit methods, instead of relying only on verbalized confidence, which can be misleading or superficial. | Recent work by Tian et al. \cite{tian2023just} asks how calibrated confidence can best be elicited from models fine-tuned with human feedback. Their study found that the raw token probabilities of such models are consistently overconfident, and that confidence verbalised as output tokens is typically better calibrated, often reducing the expected calibration error by a relative 50\%. Because the best confidence source differs between model families and tasks, this reinforces CAEM's strategy of combining implicit signals like token probability, dropout, and self-consistency with explicit ones, and of recalibrating every signal rather than trusting any raw score. |
| 40 | `chapters/chapter_2.tex` | and confirms that no single method exceeds AUROC of approximately $0.6$ across heterogeneous benchmarks; this survey is the foundational positioning document for CAEM's multi-signal design choice. | and is the positioning document for CAEM's multi-signal design choice. |
| 41 | `chapters/chapter_2.tex` | the method reports AUROC near $0.84$ on TriviaQA, the strongest unsupervised single-signal result on factual QA we are aware of, and is registered | the method reports AUROC of $0.80$ to $0.83$ on TriviaQA and clear gains over other unsupervised signals on CoQA, SQuAD and Natural Questions (on TriviaQA, perplexity scores slightly higher), and is registered |
| 42 | `chapters/chapter_2.tex` | The components areL: turning the input into features, updating the memory, turning the memory output into features, and finally producing a response. Furthermore, Memory Networks performed strongly on bAbI question. With answering tasks that required reasoning over multiple facts. Sukhbaatar et al. \cite{NIPS2015_8fb21ee7} extended this  model with end to end Memory Networks at NeurIPS 2015. He removed the need for supervision on retrieval. He did this model by continuous attention mechanisms. This described how CAEM builds its own style of step by step reasoning using stored knowledge. | The components are: turning the input into features, updating the memory, turning the memory output into features, and finally producing a response. Furthermore, Memory Networks performed strongly on bAbI question-answering tasks that required reasoning over multiple facts. Sukhbaatar et al. \cite{NIPS2015_8fb21ee7} extended this model with end-to-end Memory Networks at NeurIPS 2015. They removed the need for supervision on which memories to retrieve by using a continuous attention mechanism over the memory. This line of work motivates CAEM's use of stored knowledge as an explicit, inspectable memory. |
| 43 | `chapters/chapter_2.tex` | As a result it keep track of the order in which things were written. DNCs achieved 3.8\% average error on bAbI tasks compared to what it achieved for memory networks which is 6.4\%. It clearly shows that calculated memory management improves performance. | It also keeps track of the order in which things were written. DNCs achieved a 3.8\% mean error on the bAbI tasks, compared with 7.5\% for the best previous jointly trained model, which suggests that learned memory management improves performance. |
| 44 | `chapters/chapter_2.tex` | The adaptive retrieval rule reaches forty-six and a half percent exact match on PopQA against GPT-3 davinci-003, approximately ten absolute percentage points above the non-adaptive baseline that retrieves on every query. | With GPT-3 davinci-003, the adaptive retrieval rule reaches forty-six and a half percent accuracy on PopQA, 5.3 percentage points above the best non-adaptive method. |
| 45 | `chapters/chapter_2.tex` | A BERT-based binary classifier is then trained on the labelled set, and at inference the classifier dispatches | A classifier is then built on the labelled set (the paper compares direct prompting, in-context learning, a BERT-base classifier and a k-nearest-neighbour rule, of which the last performs best), and at inference the classifier dispatches |
| 46 | `chapters/chapter_2.tex` | on a five-dataset open-domain question-answering suite (SQuAD, Natural Questions, TriviaQA, HotpotQA, 2WikiMultiHopQA, MuSiQue) beats single-strategy baselines and the unconditional retrieval-augmented baseline, but the T5-large routing cost is the system's main weakness. | on a six-dataset open-domain question-answering suite (SQuAD, Natural Questions, TriviaQA, HotpotQA, 2WikiMultiHopQA, MuSiQue) beats single-strategy baselines and the unconditional retrieval-augmented baseline; for CAEM's purposes the T5-large routing cost is the design's main weakness, while the authors themselves single out the quality of the automatically assigned complexity labels. |
| 47 | `chapters/chapter_2.tex` | removing retrieval from the system drops PopQA performance by approximately forty percent relative, | removing retrieval at inference time drops PopQA accuracy by roughly forty-five percent relative in the authors' ablation, |
| 48 | `chapters/chapter_2.tex` | The probe predicts whether retrieval-augmented generation will improve the answer over zero-shot generation; the reported accuracy across five open-domain question-answering datasets beats FLARE, Self-RAG, and Adaptive-RAG on the same suite with fewer retrieval steps. | (wording refined again in row 97) The probe is trained to predict whether the model would answer correctly without retrieval, and retrieval is triggered when it predicts not; the reported accuracy across five open-domain question-answering datasets beats single-step retrieval, FLARE, DRAGIN, and Adaptive-RAG on the same suite while retrieving roughly half as often. |
| 49 | `chapters/chapter_2.tex` | that injects trainable low-rank matrices into the attention and feed-forward projections of a frozen backbone, with the original paper reporting near-full-FT quality at less than one percent of the trainable parameter count on GPT-3 175B. | that injects trainable low-rank matrices into the weight matrices of a frozen backbone; the original paper adapts only the attention projections and reports quality on par with full fine-tuning at less than one percent of the trainable parameter count on GPT-3 175B, and later work extends the adapters to the feed-forward projections. |
| 50 | `chapters/chapter_2.tex` | Biderman and colleagues sweep Llama-2-7B across rank settings from sixteen to two hundred and fifty-six on Magicoder coding instruction data \cite{biderman2024lora} and report two findings that translate directly to CAEM's setting: low-rank adaptation retains roughly sixty-six percent of HellaSwag, ARC-Challenge, and Winogrande accuracy at every rank tested, against sixty-two percent retention for full fine-tuning matched on epoch count, and the low-rank versus full gap on retention dominates the rank-versus-rank gap on fit. Higher rank trades retention for fit, but every rank setting they tested retained more general capability than full fine-tuning at the same training-pool size. | Biderman and colleagues compare low-rank adaptation with full fine-tuning for Llama-2-7B on code and mathematics, sweeping ranks from sixteen to two hundred and fifty-six \cite{biderman2024lora}, and report two findings relevant to CAEM's setting: low-rank adaptation learns less of the target domain than full fine-tuning, but it forgets less of the base model's general capability, measured as average HellaSwag, ARC-Challenge, and WinoGrande accuracy. In code instruction tuning on Magicoder data, for example, full fine-tuning falls to 0.414 against 0.509 for low-rank adaptation at rank sixty-four. Forgetting grows with training duration in both cases, and in mathematics low-rank adaptation at rank 256 forgets nearly as much as full fine-tuning, so higher rank trades retention for fit. |
| 51 | `chapters/chapter_2.tex` | establishes the second decisive practical point: applying low-rank adaptation to all linear layers, both attention and feed-forward, rather than to the attention projections alone is the most critical configuration choice, and matched-parameter-count experiments show that adding feed-forward adapters to attention adapters dominates adding more attention rank. | establishes the second decisive practical point: applying low-rank adaptation to all linear transformer-block layers, both attention and feed-forward, rather than to the attention projections alone is the most critical configuration choice for matching full fine-tuning, whereas the rank itself has little effect once all layers are adapted. |
| 52 | `chapters/chapter_2.tex` | fitted at rank thirty-two with the standard $\alpha=2r$ scaling, which sits comfortably inside the rank regime where the standard scaling avoids the gradient collapse at higher rank that motivates the rank-stabilised alternative \cite{kalajdzievski2023rslora}. | fitted at rank thirty-two with $\alpha=2r$; with this choice the effective scale $\alpha/r$ stays at two, so the rank-driven gradient collapse that motivates the rank-stabilised alternative for fixed-$\alpha$ adapters does not arise, and rank thirty-two is far below the large ranks at which the theory of \cite{kalajdzievski2023rslora} predicts instability for a constant scale. |
| 53 | `chapters/chapter_2.tex` | Their method achieves more than 88\% accuracy with 40\% label corruption which substantially outperforms standard training. | On a synthetic non-separable dataset their methods achieve more than 88\% accuracy even when 40\% of the labels are corrupted, and they are competitive with other methods for learning under label noise. |
| 54 | `chapters/chapter_2.tex` | When the model produces a correct answer without valid reasoning, it generates a rationale conditioned on the correct answer. This data augmentation stops the model from learning problems that it already knows how to solve. | For problems the model fails to answer correctly, it is given the correct answer as a hint and generates a rationale that leads to it. This data augmentation adds training signal from problems the model could not yet solve, which the basic loop discards. |
| 55 | `chapters/chapter_2.tex` | Huang et al. \cite{huang-etal-2023-large} found at EMNLP 2023 that prompting LLMs to review and correct their own responses rarely improves accuracy if we do not use external feedback. | Huang et al. \cite{huang2024cannot} found at ICLR 2024 that prompting LLMs to review and correct their own reasoning does not improve accuracy without external feedback, and at times degrades it. |
| 56 | `chapters/chapter_2.tex` | formalise the \emph{generation--verification gap} and prove empirically that iterative self-improvement succeeds if and only if the verifier is more accurate than the generator on the noisy-label-recovery task; this is the conceptual basis for CAEM's $\alpha > 1/2$ verifier-purity precondition in \Cref{sec:theory}. | formalise the \emph{generation--verification gap}, the improvement obtained by reweighting a model's generations with its own verification, and show empirically that self-improvement requires a positive gap, that small models fail to achieve one, and that iterative self-improvement saturates within two or three rounds; this is the conceptual motivation for CAEM's $\alpha > 1/2$ verifier-purity precondition in \Cref{sec:theory}. |
| 57 | `chapters/chapter_2.tex` | Fu et al.~\cite{fu2025collapse} prove theoretically that a verifier filter over synthetic data prevents model collapse in self-consuming training loops and admits both near-term improvement and long-term convergence guarantees; this is the strongest published safety anchor for CAEM's storage-gate-filtered SIL. | Yi et al.~\cite{fu2025collapse} prove theoretically, in linear regression, that a verifier filter over synthetic data prevents model collapse in self-consuming training loops and yields near-term improvement, while in the long run the model converges to the verifier's knowledge centre, so that gains plateau or reverse unless the verifier is unbiased; this is the strongest published safety anchor for CAEM's storage-gate-filtered SIL, and also its main caution. |
| 58 | `chapters/chapter_2.tex` | prove at ICML 2025 that retraining on verifier-correct hard labels strictly increases population accuracy in the binary linearly-separable classification setting; this is the closest published analogue of CAEM's pool-purity inequality (\Cref{thm:purity}), and CAEM extends it from binary classification with internal self-prediction to open-ended generation with an external multi-signal verifier. | prove at ICML 2025 that retraining a model on its own predicted hard labels increases population accuracy in a binary linearly-separable classification setting with randomly flipped labels, and show empirically that retraining only on samples where the prediction agrees with the given label helps further; this is the closest published analogue of CAEM's pool-purity inequality (\Cref{thm:purity}), and CAEM extends the idea from binary classification with self-predicted labels to open-ended generation with an external multi-signal verifier. |
| 59 | `chapters/chapter_2.tex` | The second, \textbf{Chain-of-Thought} prompting~\cite{wei2022chain}, adds the standard ``Let's think step by step'' trigger without any further modification. | The second, \textbf{Chain-of-Thought} prompting~\cite{wei2022chain}, adds the standard zero-shot ``Let's think step by step'' trigger~\cite{kojima2022large} without any further modification. |
| 60 | `chapters/chapter_2.tex` | Claims are specifically designed to test factual knowledge, such as ``Barack Obama was the 44th President of the United States'' (SUPPORTS) or ``The Beatles formed in Manchester'' (REFUTES, they formed in Liverpool). | Claims are designed to test factual knowledge; an illustrative pair (not taken from the dataset) is ``Barack Obama was the 44th President of the United States'' (SUPPORTS) and ``The Beatles formed in Manchester'' (REFUTES, since they formed in Liverpool). |
| 61 | `chapters/chapter_2.tex` | providing 650{,}000 trivia question-answer pairs drawn from quizzing leagues, each paired with multiple supporting documents from the web and Wikipedia. | providing about 95{,}000 trivia question-answer pairs authored by trivia enthusiasts, which with on average six supporting documents each from the web and Wikipedia give over 650{,}000 question-answer-evidence triples. |
| 62 | `chapters/chapter_2.tex` | Human performance reaches 94\% while GPT-3 175B achieves only 21--25\% on the truthful-and-informative metric. | Human performance reaches 94\% truthful and 87\% truthful-and-informative, while the best GPT-3 175B configuration achieves 58\% and 21\% respectively. |
| 63 | `chapters/chapter_2.tex` | For example, ``Could a llama birth twice during the War in Vietnam?'' requires | For example, the dataset question ``Could a llama birth twice during the War in Vietnam?'' requires |
| 64 | `chapters/chapter_2.tex` | Hallucination-measurement studies report prevalence rates varying widely by domain, reaching above 90\% in worst-case clinical settings \cite{alkaissi2023artificial}. | Hallucination-measurement studies report prevalence rates varying widely by domain, with up to 91.4\% of chatbot-generated references hallucinated in systematic-review searches \cite{info:doi/10.2196/53164} and fabricated references in clinical writing \cite{alkaissi2023artificial}. |
| 65 | `chapters/chapter_2.tex` | in which GPT-3 175B achieved only 21--25\% on the combined truthful-and-informative metric, | in which the best GPT-3 175B configuration achieved only 21\% on the combined truthful-and-informative metric, |
| 66 | `chapters/chapter_2.tex` | Self-consistency \cite{wang2023selfconsistency} provides the sample-set agreement signal $s_{\text{avg}}$. | Self-consistency \cite{wang2023selfconsistency} shows that agreement across sampled reasoning chains is an uncertainty signal; CAEM's sample-set agreement signal $s_{\text{avg}}$ adapts it to embedding space as the mean pairwise cosine similarity of the sampled chains. |
| 67 | `chapters/chapter_2.tex` | MiniCheck is fine-tuned specifically on language-model-generated claim-support data and produces well-calibrated supported-versus-unsupported probabilities on the distribution of claims the verifier actually scores, whereas | MiniCheck is fine-tuned specifically on language-model-generated claim-support data and outperforms prior specialised fact-checkers on claims of the kind the verifier actually scores (its raw scores are bimodal rather than calibrated, which CAEM's per-signal isotonic step corrects), whereas |
| 68 | `chapters/chapter_2.tex` | \cite{semanticillusion2025} report a one-hundred-percent | \cite{sinha2025semantic} report a one-hundred-percent |
| 69 | `chapters/chapter_2.tex` | \cite{valentin2024hallucination} publish the closest analogue: an eight-signal isotonic-stacked hallucination detector on TriviaQA, FEVER, HaluEval, and BIG-bench, with a Spearman cross-signal correlation heat-map that identifies three redundancy clusters among their signals. \cite{vashurin2024polygraph} benchmark twenty-eight uncertainty-estimation methods | \cite{valentin2024hallucination} publish the closest analogue: a multi-scoring hallucination detector that calibrates about ten scores with multicalibration and combines their logits through logistic regression, evaluated on TriviaQA, FEVER, HaluEval, and BIG-bench, with a Spearman cross-score correlation heat-map that reveals clusters of related scores. \cite{vashurin2024polygraph} benchmark around thirty uncertainty-estimation methods |
| 70 | `chapters/chapter_2.tex` | is a direct extension of the Valentin 2024 four-benchmark-by-eight-signal matrix to the retrieval and MC-Dropout signal families that their registry did not cover. | is a direct extension of the Valentin 2024 four-benchmark cross-score matrix to the retrieval and MC-Dropout signal families that their score set did not cover. |
| 71 | `chapters/chapter_2.tex` | The confidence-estimation literature shows that single signals only reach roughly 0.5--0.6 AUROC for discriminating correct from incorrect answers \cite{xiong2023can}, | The confidence-estimation literature shows that verbalised confidence discriminates correct from incorrect answers only weakly (an AUROC of about 0.6) and that no single confidence signal is consistently best \cite{xiong2023can}, |
| 72 | `chapters/chapter_2.tex` | EWC \cite{doi:10.1073/pnas.1611835114} maintains approximately 90\% performance across sequential tasks and supplies the conceptual motivation for bounding cycle-to-cycle drift in adapter space; the explicit $L_{2}$ anchor that approximates EWC under a uniform-Fisher assumption is registered | EWC \cite{doi:10.1073/pnas.1611835114} maintains performance on earlier tasks while learning new ones and supplies the conceptual motivation for bounding cycle-to-cycle drift in adapter space; the explicit $L_{2}$ anchor, the uniform-Fisher special case that the EWC paper itself shows to be weaker than EWC, is registered |
| 73 | `chapters/chapter_2.tex` | establishes that low-rank adaptation \cite{hu2022lora} on all linear layers retains general capability across iterative cycles substantially better than full-parameter fine-tuning at matched training budget \cite{biderman2024lora,dettmers2023qlora}, which is the empirical anchor | establishes that low-rank adaptation \cite{hu2022lora} forgets less of the base model's general capability than full-parameter fine-tuning \cite{biderman2024lora}, and that adapters on all linear layers are needed to match full fine-tuning quality \cite{dettmers2023qlora}, which is the empirical anchor |
| 74 | `chapters/chapter_2.tex` | Direct empirical evidence at pre-trained transformer scale shows that the Fisher-weighting advantage of EWC, SI, and MAS over plain $L_2$ collapses once the backbone is a pre-trained language model: \cite{mehta2023empirical} report minimal or negative improvement at BERT-Large scale, and \cite{wu2022pretrained} reproduce the same pattern across five pre-trained backbones, attributed to the flatter loss landscape that pre-training induces. | Empirical studies at pre-trained transformer scale suggest that weight-regularisation methods add little once the backbone is pre-trained: \cite{mehta2023empirical} find that pre-trained weights themselves ease forgetting by leading to wider minima, and \cite{wu2022pretrained} find across five pre-trained language models that EWC performs worst among the continual-learning methods they compare, while experience replay is the most robust. |
| 75 | `chapters/chapter_2.tex` | Noisy-label learning \cite{natarajan2013learning} proves that training on pools whose purity is well below 100\% can still improve a model, which justifies the registered stricter training threshold | Noisy-label learning \cite{natarajan2013learning} shows that a classifier can still be learned from substantially corrupted labels, provided the class-conditional noise rates sum to less than one and the loss is corrected for them; CAEM does not correct its loss, and instead lowers the noise rate of the training pool directly through the registered stricter training threshold |
| 76 | `chapters/chapter_2.tex` | The self-correction critiques of \cite{huang-etal-2023-large} and \cite{kamoi-etal-2024-llms} | The self-correction critiques of \cite{huang2024cannot} and \cite{kamoi-etal-2024-llms} |
| 77 | `appendix/appendix_g.tex` | The published rate of confident-but-wrong outputs in the medical-question-answering domain reaches 69--77\% \cite{alkaissi2023artificial}, and TruthfulQA's | Chatbots have been documented fabricating every reference offered for a clinical explanation \cite{alkaissi2023artificial}, up to 91.4\% of chatbot-generated references in systematic-review searches are hallucinated \cite{info:doi/10.2196/53164}, and TruthfulQA's |
| 78 | `appendix/appendix_g.tex` | \cite{xiong2023can} report that any single confidence signal reaches an area-under-the-curve in the range $0.50$ to $0.60$ for distinguishing correct from incorrect outputs; the composite aggregates nine signals across four families through per-signal isotonic regression and a regularised logistic boost layer to produce a calibrated correctness probability that no single signal can match. | \cite{xiong2023can} report that verbalised confidence distinguishes correct from incorrect outputs only weakly (an area under the curve of about $0.6$ for GPT-4) and that no single confidence-elicitation method is consistently best across tasks; the composite therefore aggregates nine signals across four families through per-signal isotonic regression and a regularised logistic boost layer to produce a calibrated correctness probability that does not depend on any one signal. |
| 79 | `appendix/appendix_g.tex` | Re-fitting the per-signal isotonic curves on the same (variable signals, constant labels) data is exactly the procedure the post-hoc calibration literature \cite{ovadia2019trust,thulasidasan2019mixup} registers for tracking calibration drift after a model update. | Re-fitting the per-signal isotonic curves on the same (variable signals, constant labels) data keeps the calibrator matched to the current model, which is the regime in which the post-hoc calibration literature finds such calibration reliable; calibration fitted before a model or data shift degrades after it \cite{ovadia2019trust}. |
| 80 | `chapters/chapter_3.tex` | including health-information retrieval~\cite{alkaissi2023artificial,info:doi/10.2196/53164}, legal triage, and educational tutoring~\cite{ji2023survey}, an architecture | such as health-information retrieval~\cite{alkaissi2023artificial,info:doi/10.2196/53164}, and in other high-stakes settings such as legal triage and educational tutoring, an architecture |
| 81 | `chapters/chapter_4.tex` | with higher values reading as higher agreement \cite{wang2023selfconsistency}. | with higher values reading as higher agreement; it is an embedding-space adaptation of the self-consistency agreement signal \cite{wang2023selfconsistency}. |
| 82 | `chapters/chapter_4.tex` | Isotonic regression auto-detects the monotone direction from the Pearson correlation between the raw signal and the binary label, so a signal that reads high-confidence-as-low-value (such as the dropout-disagreement signal, where higher variance reads as lower confidence) is calibrated correctly without a registry of per-signal directions \cite{niculescu2005predicting}. | Isotonic regression fits a monotonically increasing map from score to probability \cite{niculescu2005predicting}; CAEM's implementation auto-detects the monotone direction from the Pearson correlation between the raw signal and the binary label, so a signal that reads high-confidence-as-low-value (such as the dropout-disagreement signal, where higher variance reads as lower confidence) is calibrated correctly without a registry of per-signal directions. |
| 83 | `chapters/chapter_4.tex` | by fitting a logistic regression on the per-signal log-odds as features against the exact-match labels of the calibration fold \cite{cherian2024conformal}. | by fitting a logistic regression on the per-signal log-odds as features against the exact-match labels of the calibration fold, following the multi-scoring approach of \cite{valentin2024hallucination}. |
| 84 | `chapters/chapter_4.tex` | \cite{NEURIPS2022_639a9a17,hosseini2024vstar,gulcehre2023rest,ho2023large}, specialised here | \cite{NEURIPS2022_639a9a17,hosseini2024vstar,gulcehre2023rest}, and related to distilling a teacher's filtered reasoning into a student \cite{ho2023large}, specialised here |
| 85 | `chapters/chapter_4.tex` | Recent theoretical work on whether such loops collapse or improve their generator establishes a sufficient condition that the verifier outperform the generator at the noisy-label-recovery task \cite{song2024gap}, and a complementary sufficient condition that the verifier-filtered admission policy be precision-bounded rather than recall-bounded so that the loop accumulates rather than displaces high-quality data \cite{fu2025collapse}; both conditions are met by the storage class admitted under the registered cut-points of \Cref{sec:conformal-gate}. | Recent work on whether such loops collapse or improve their generator finds empirically that self-improvement requires a positive generation--verification gap and saturates within a few rounds \cite{song2024gap}, and proves in linear regression that filtering synthetic data through an external verifier prevents collapse but draws the model toward the verifier's own knowledge, so that gains plateau unless the verifier is unbiased \cite{fu2025collapse}; CAEM's external multi-signal verifier and the registered cut-points of \Cref{sec:conformal-gate} are designed to keep the admitted storage class above the purity precondition of \Cref{thm:purity}. |
| 86 | `chapters/chapter_4.tex` | supported by the post-hoc calibration analysis of \cite{niculescu2005predicting} and the selective-classification framing of \cite{geifman2017selective}. | supported by the selective-classification risk bound of \cite{geifman2017selective}, which holds when calibration and test data are drawn from the same distribution, with the per-signal post-hoc calibration on a held-out fold following \cite{niculescu2005predicting}. |
| 87 | `chapters/chapter_4.tex` | showing that retraining on verifier-correct labels strictly increases population accuracy in the binary linearly-separable setting \cite{das2025retraining} | showing that retraining on a model's own predicted hard labels increases population accuracy in the binary linearly-separable setting \cite{das2025retraining} |
| 88 | `chapters/chapter_4.tex` | requires the stochastic-approximation supermartingale machinery of \cite{robbins1951stochastic} and the rate analysis of \cite{polyak1990new} adapted to a bounded gap space; | requires the stochastic-approximation framework of \cite{robbins1951stochastic}, the almost-supermartingale convergence theorem of \cite{robbins1971convergence} (used for verifier-filtered retraining by \cite{fu2025collapse}), and the rate analysis of \cite{polyak1990new}, adapted to a bounded gap space; |
| 89 | `chapters/chapter_5.tex` | the McNemar statistic is the maximum-likelihood test for the matched-pairs binary case. | the McNemar statistic is the standard test for matched-pairs binary outcomes. |
| 90 | `chapters/chapter_5.tex` | The chain-of-thought baseline \cite{wei2022chain} adds the standard step-by-step trigger to the zero-shot prompt without any further modification. | The chain-of-thought baseline \cite{wei2022chain} adds the zero-shot step-by-step trigger of \cite{kojima2022large} to the zero-shot prompt without any further modification. |
| 91 | `chapters/chapter_5.tex` | clustering the samples by normalised-exact-match semantic equivalence on the extracted answer, and reading | clustering the samples by normalised-exact-match equivalence on the extracted answer (a simplification of the original method's bidirectional-entailment clustering, appropriate for short answers), and reading |
| 92 | `chapters/chapter_5.tex` | demonstrates a precision lift from $78\%$ to $93\%$ on Natural Questions at $\alpha=0.2$ using GPT-4 with split-conformal sub-claim filtering, and \cite{cherian2024conformal} sharpens this with the boosted logistic aggregation that CAEM's composite layer adopts. | demonstrates a correctness lift from $78\%$ to $93\%$ on Natural Questions using GPT-4 with split-conformal sub-claim filtering, and \cite{cherian2024conformal} sharpens this with conditionally valid guarantees and a boosted combination of claim scores; CAEM's composite layer instead follows the logistic multi-scoring of \cite{valentin2024hallucination}. |
| 93 | `chapters/chapter_5.tex` | \cite{NEURIPS2022_639a9a17} and \cite{hosseini2024vstar} demonstrate that bootstrapping on verifier-correct outputs is a viable training-data-admission rule, and \cite{huang2025sharpening} formalises the convergence dynamics that CAEM's \Cref{thm:convergence} extends under the bounded-band stationarity reframing. | \cite{NEURIPS2022_639a9a17} and \cite{hosseini2024vstar} demonstrate that bootstrapping on correctness-filtered self-generated outputs is a viable training-data-admission rule, with V-STaR additionally training a verifier on the same pool for inference-time ranking, and \cite{huang2025sharpening} formalises self-improvement as sharpening with statistical sample-complexity guarantees, which CAEM's \Cref{thm:convergence} complements with a multi-cycle bounded-band stationarity result. |
| 94 | `chapters/chapter_2.tex` | EWC supplies the conceptual motivation for bounding cycle-to-cycle drift in adapter space; the explicit $L_{2}$ anchor that approximates EWC under a uniform-Fisher assumption is registered for the inactive full-fine-tune fallback only, | EWC supplies the conceptual motivation for bounding cycle-to-cycle drift in adapter space; the explicit $L_{2}$ anchor, the uniform-Fisher special case that the EWC paper shows to be weaker than EWC itself, is registered for the inactive full-fine-tune fallback only, |
| 95 | `chapters/chapter_2.tex` | which the parameter-efficient-fine-tuning literature shows retains general capability across iterative cycles substantially better than full fine-tuning at matched training budget. | which the parameter-efficient-fine-tuning literature shows forgets less of the base model's general capability than full fine-tuning. |
| 96 | `chapters/chapter_2.tex` | Noisy-label learning provides theoretical justification for training on imperfectly verified data and motivates | Noisy-label learning shows that learning from imperfectly labelled data remains possible when the label noise is bounded, and motivates |
| 97 | `chapters/chapter_2.tex` | The probe is trained to predict whether the model would answer correctly without retrieval, and retrieval is triggered when it predicts not; | The probe is trained to predict, from the hidden states, whether the model's current answer is correct and hence whether additional retrieval is needed; |
| 98 | `chapters/chapter_4.tex` | (used for verifier-filtered retraining by \cite{fu2025collapse}) | (supermartingale arguments of this kind underpin the convergence analysis of verifier-filtered retraining in \cite{fu2025collapse}) |
| 99 | `chapters/chapter_2.tex` | Lang, Vijayaraghavan, and Sontag~\cite{lang2024w2s} | Lang, Sontag, and Vijayaraghavan~\cite{lang2024w2s} |
| 100 | `chapters/chapter_3.tex, chapters/chapter_5.tex` | \cite{…geva2021strategyqa…} | \cite{…10.1162/tacl_a_00370…} (duplicate key merged) |

## Tier C: status

The table below lists the design-level items found by the check. Items C#1–C#4, C#6 and the McNemar/RTE citations were **applied on 2026-10-15** with the recommended options (see the next section for the exact text). The 50-sample isotonic floor (`chapter_4.tex:333`) and the new design findings 9–19 remain open.

| Where | Issue | Proposed direction |
|---|---|---|
| `chapter_2.tex:283` (TruthfulQA paragraph) | Expects semantic entropy / sample agreement to catch misconceptions; Farquhar et al. exclude "consistently wrong … common misconceptions". | Rewrite as an explanation of the observed TruthfulQA result (C#2). |
| `chapter_5.tex:456` (B8 expectation) | Same issue for the B8 semantic-entropy baseline. | Same as above (C#2). |
| `chapter_2.tex:301` | Says CAEM uses Qwen2.5-3B-Instruct's "plain instruction-tuned checkpoint rather than an RLHF-aligned chat variant"; the Qwen2.5 report says the Instruct models use DPO and GRPO. | Re-justify via post-hoc recalibration, citing Kadavath §3.3 and Tian (C#3). |
| `chapter_5.tex:426` | Expects a positive CoT contrast at 3B; Wei, Chung and the thesis's own B2 < B1 say otherwise. (The chapter 2 literature review was already corrected under tier B.) | Rewrite as the observed result (C#4). |
| `chapter_4.tex:348` | The fixed threshold is called "robust to the cal-versus-eval shift"; the cited papers assume no shift. | Use the thesis's own evidence and cite Ovadia/Kamath as the limitation (C#6). |
| `chapter_4.tex:333` | A 50-sample floor for isotonic fits, versus Niculescu-Mizil's ~200–1000. | Add one sentence on why the shrinkage prior is needed (new item 11). |
| `chapter_5.tex:366` | The −1 continuity correction is attributed to McNemar 1947 (usually Edwards 1948). | Verify once the McNemar PDF is available. |
| Theory (`sec:theory`) | Re-anchor α > ½ on Natarajan's ρ₊ + ρ₋ < 1 as a learnability condition; weaken the retroactive-purity corollary. | C#1. |
| New design findings 9–19 | Saturation within 2–3 rounds and format-learning confound (Song); over-optimisation to the verifier (ReST, Yi); STaR binary-task caveat; and others. | See "New findings" in `reference_check_2_claims.md`. |

## Tier C: changes applied (2026-10-15)

| # | File | Before | After |
|---|---|---|---|
| C1 | `chapters/chapter_2.tex` | For CAEM, TruthfulQA is the benchmark where the sample-set family of signals is expected to pay off: misconceptions typically produce high semantic entropy because the model generates both the correct answer and the misconception with similar confidence across stochastic decodes, and the deployed sample-set agreement signals $s_{\text{avg}}$ and $p_{\text{entail}}$, the surviving proxy for the semantic-entropy principle~\cite{farquhar2024semantic}, are designed to catch this. | For CAEM, TruthfulQA was registered as the benchmark where the sample-set family of signals was expected to pay off, on the reasoning that a misconception would make the model alternate between the correct answer and the misconception across stochastic decodes, which the sample-set agreement signals $s_{\text{avg}}$ and $p_{\text{entail}}$ would detect. The literature predicts otherwise: semantic entropy targets arbitrary, sample-to-sample errors and explicitly does not cover models that are consistently wrong because of common misconceptions~\cite{farquhar2024semantic}, since such a model gives the same wrong answer on every decode and agreement-based signals then read it as confident. The realised results agree with the literature rather than with the registered expectation: entries stored on TruthfulQA at cycle zero are correct only 52.2\% of the time, against 75 to 82\% on the training panel (\Cref{tab:cross-bench-summary}), and the semantic-entropy baseline is among the highest-scoring baselines on TruthfulQA exact match rather than being penalised for heavy abstention (Chapter~5). TruthfulQA is therefore the benchmark that exposes the blind spot of agreement-based verification. |
| C2 | `chapters/chapter_5.tex` | is expected to produce a small positive contrast against the zero-shot baseline. | is expected to produce a small positive contrast against the zero-shot baseline. The TruthfulQA part of this expectation was not borne out, as the literature predicts: semantic entropy does not detect consistently held misconceptions \cite{farquhar2024semantic}, and B8 ranks among the high-exact-match retrieval-free baselines on TruthfulQA rather than being penalised for heavy abstention. |
| C3 | `chapters/chapter_2.tex` | and that RLHF can further degrade calibration \cite{kadavath2022language}, which is one reason CAEM builds on Qwen-2.5-3B-Instruct's plain instruction-tuned checkpoint rather than an RLHF-aligned chat variant. | and that models tuned with reinforcement learning from human feedback appear badly miscalibrated until their outputs are rescaled \cite{kadavath2022language}, with raw token probabilities that are consistently overconfident \cite{tian2023just}. CAEM's backbone, Qwen-2.5-3B-Instruct, is itself preference-tuned (its post-training uses DPO and GRPO \cite{qwen25}), which is why CAEM never consumes a raw confidence: the pre-routing score is temperature-scaled and every verifier signal is isotonically recalibrated on a held-out fold. |
| C4 | `chapters/chapter_5.tex` | and a smaller contrast on the single-step benchmarks (FEVER, TriviaQA). | and a smaller contrast on the single-step benchmarks (FEVER, TriviaQA). The literature predicts the opposite at this scale: below roughly 100 billion parameters chain-of-thought prompting does not help and can hurt \cite{wei2022chain}, and even an instruction-tuned 780-million-parameter model scores lower with it than with direct answering \cite{JMLR:v25:23-0870}. The realised pooled exact match (B2 $0.500$ against B1 $0.507$, \Cref{tab:baseline-pooled}) agrees with the literature rather than with the registered expectation. |
| C5 | `chapters/chapter_4.tex` | it is robust to the cal-versus-eval shift that the conformal premise requires, in the lineage of post-hoc calibration analyses by \cite{niculescu2005predicting} and the selective-classification framing of \cite{geifman2017selective}; | it makes no marginal-coverage claim that the cal-versus-eval shift can silently violate, because its precision is audited directly on every evaluation fold, in the lineage of thresholding a calibrated probability \cite{niculescu2005predicting} and of selective classification \cite{geifman2017selective}; both assume that calibration and test data share a distribution, so under shift the realised precision at the threshold can still fall \cite{ovadia2019trust,kamath2020selective}, and the transfer-panel TruthfulQA reading in Chapter~5 is such a case; |
| C6 | `chapters/chapter_5.tex` | The fixed-threshold rule is robust to the calibration-versus-evaluation shift that the conformal premise required because it does not depend on the marginal-coverage statement; | The fixed-threshold rule does not depend on the marginal-coverage statement that the calibration-versus-evaluation shift broke, so the shift cannot silently invalidate a guarantee, although it can still lower the realised precision, as the transfer-panel TruthfulQA store precision of $0.522$ in \Cref{tab:cross-bench-summary} shows; |
| C7 | `chapters/chapter_5.tex` | does not depend on coverage transfer and is robust to the shift. | does not depend on coverage transfer, so the shift cannot silently void a guarantee, although it can still lower the realised precision. |
| C8 | `chapters/chapter_4.tex` | with $P_c > p_c$ whenever $\alpha_c > 1/2$. | with $P_c > p_c$ whenever $\alpha_c > 1/2$. Without the symmetric reduction, $P_c > p_c$ holds exactly when $\mathrm{TPR}_c > \mathrm{FPR}_c$, which is equivalent to a balanced accuracy above one half and, reading the gate's decisions as noisy labels with flip rates $\rho_{+} = 1-\mathrm{TPR}_c$ and $\rho_{-} = \mathrm{FPR}_c$, to the class-conditional noise condition $\rho_{+} + \rho_{-} < 1$ under which learning from noisy labels remains possible \cite{natarajan2013learning}. Natarajan et al.'s guarantees additionally require a loss corrected for the noise rates, which CAEM does not apply, so the condition is used here as the precondition for purification rather than as a transferred learning guarantee. |
| C9 | `chapters/chapter_4.tex` | converges to $1$ as $k \to \infty$, and any hallucinated episode is pruned within finitely many cycles in expectation. | converges to $1$ as $k \to \infty$, and any hallucinated episode is pruned within finitely many cycles in expectation, provided the verifier's errors on an entry are independent across passes; errors that recur at every pass are not removed by repetition (see the scope note after the proof). |
| C10 | `chapters/chapter_4.tex` | the rate constant $\delta$ is conservative rather than tight. \end{proof} | the rate constant $\delta$ is conservative rather than tight. \end{proof}  \paragraph{Scope of the independence premise:} The convergence to one requires the verifier's errors on an entry to be independent across passes. Errors that are systematic, such as the same wrong answer accepted at every pass because the model and the verifier share a consistently held misconception \cite{farquhar2024semantic}, are not removed by repetition: in the fully correlated limit every pass after the first returns the same verdict and pool purity stays at the single-pass value of \Cref{thm:bayes-purity}. Repeated passes therefore remove only the independent part of the verifier's error, and purity rises toward a ceiling below one that is set by the verifier's systematic errors, consistent with the finding that verifier-filtered retraining converges to the verifier's knowledge rather than to the truth \cite{fu2025collapse}. The transfer-panel TruthfulQA reading (stored precision $0.522$ at cycle zero, \Cref{tab:cross-bench-summary}) is the empirical instance of such systematic errors. |
| C11 | `chapters/chapter_4.tex` | the implementation regularisers above pin the effective pool-purity ceiling strictly below one, | the implementation regularisers above, and verifier errors that are systematic rather than independent across cycles (see the scope note on \Cref{cor:self-correction}), pin the effective pool-purity ceiling strictly below one, |
| C12 | `chapters/chapter_5.tex` | \cite{mcnemar1947note} with continuity correction. | \cite{mcnemar1947note} with the continuity correction of \cite{edwards1948note}. |
| C13 | `chapters/chapter_2.tex` | from a premise sentence \cite{10.1007/11736790_9}. The PASCAL challenge framed | from a premise sentence \cite{barhaim2006second}. The PASCAL Recognising Textual Entailment challenges framed |

New bib entries for tier C: `edwards1948note` (Edwards 1948, Psychometrika 13:185–187; standard record, **PDF not checked**) and `barhaim2006second` (RTE-2, PDF read). `10.1007/11736790_9` (RTE-1) is no longer cited; if you upload RTE-1 and prefer it, swap the key back in `chapter_2.tex`.

## Follow-up (2026-10-15): cleanup, limitations, theory restructure

**Citations removed.**
- McNemar 1947 is no longer cited. The McNemar test with continuity correction is now cited to Edwards (1948) in chapters 3 and 5.
- RTE-1 is no longer cited; RTE-2 is cited instead.
- The five uncited entries were deleted from the .bib: `mcnemar1947note`, `10.1007/11736790_9`, `10.1145/3649506`, `huang-etal-2023-large`, `thulasidasan2019mixup`.

**Limitations.** A new paragraph, "Limitations identified against the literature", was added to `chapter_5.tex` (`sec:disc-limitations`). It covers six items:
1. saturation and the answer-format confound (Song);
2. optimising toward the verifier (ReST, Yi), including the point that the CHM is read through the same verifier;
3. systematic verifier errors (Farquhar; TruthfulQA 0.522; Sinha on cosine similarity);
4. calibration-fold size for isotonic regression (Niculescu-Mizil);
5. low-chance-floor tasks such as StrategyQA and FEVER (STaR, Ho);
6. adapter capacity and drift (Biderman).

**Theory restructure (`chapter_4.tex`, `sec:theory`).**
- **Proven results:**
  - `thm:purity` stays a Theorem.
  - `thm:bayes-purity` moves from the scaffolding remarks into a new subsection, "When filtering helps: the purification condition". It becomes a Proposition with a complete proof: purity exceeds the base rate if and only if TPR > FPR, which is the same as balanced accuracy > ½. The Natarajan interpretation sits in its own paragraph.
- **New subsection "Design heuristics for the cycle-to-cycle trajectory"** (`sec:theory-heuristics`). It states the idealising assumptions (Gaussian score classes, stationarity, independent verifier errors) and why the realised system departs from them.
  - `thm:monotone`, `thm:convergence`, `cor:tier1-floor`, `cor:self-correction` and `cor:stochastic-equilibrium` now use a new **Heuristic** environment, with "Argument sketch" instead of "Proof sketch". The environment is defined in `main.tex` and shares the theorem counter; cleveref prints "Heuristic".
- **Mathematical scaffolding:** the four remaining remarks are unchanged and sit under the heuristics subsection.
- **False sentence removed:** chapter 4 said full proofs were "relocated to Appendix E". No proofs exist there.
- **All labels are kept**, so every cross-reference still resolves. The wording around them was updated in:
  - the abstract;
  - chapter 1: the hypothesis table now links H1 to `thm:bayes-purity`;
  - chapter 2: the related-work table and the analogue sentences;
  - chapter 3: the scope summary;
  - chapter 5: the theory-checks section, the hypothesis table and the precondition references;
  - chapter 6: the contributions paragraph and the future-work subsections;
  - appendices A and F;
  - `figures/auto/tab_theorem_receipts_summary.tex`: types, T2 row and caption.
- **Script:** `scripts/aggregate_section_5_6.py` has its type labels updated. Its other row text was already out of sync with the hand-edited `.tex` table before this change (for example, it still contains a `cor:convergence-rate` row). Do not regenerate that table from the script without checking.

## Follow-up: Naeini 2015 checked against the full text
- PDF read in full, 7 pages. Title, authors (Naeini, Cooper, Hauskrecht), venue (Proceedings of the Twenty-Ninth AAAI Conference) and pages 2901–2907 all match the .bib entry.
- §"Calibration Measures" (p. 2903) defines ECE: predictions are sorted into K bins and ECE = Σ P(i)·|o_i − e_i|. MCE is defined in the same section. This supports the chapter 2 sentence (binning by confidence, then the confidence–accuracy gap per bin). Verdict: ✅
- Edwards 1948 (Psychometrika 13(3):185–187, doi 10.1007/BF02289261): metadata confirmed via indexes. No free PDF; the title itself supports the continuity-correction claim.
- Robbins & Siegmund 1971 (Optimizing Methods in Statistics, ed. Rustagi, Academic Press): the theorem's content is confirmed via secondary sources (arXiv 2509.26442; Anbar 1976, Ann. Statist.). The page range has not been confirmed from the original volume.

## Follow-up: "pre-registered" → "pre-specified" (option a)
- All 48 occurrences across the chapters, appendices, abstract and `figures/auto/sweep_top.tex` changed: "pre-registered"/"Pre-registered" → "pre-specified"/"Pre-specified", and "pre-registration convention" → "pre-specification convention". The matching strings in `scripts/aggregate_section_5_7.py` were also updated, so regenerating the table keeps the new wording.
- `chapter_3.tex` (protocol standards paragraph): added "Throughout the thesis, a quantity described as pre-specified or registered was fixed in the project's configuration and runbook before the main evaluation run; the term does not refer to an entry in an external pre-registration registry."
- Same paragraph: "three … protocol standards" → "four", because the paragraph lists four (matched-protocol, pre-specification, selective-prediction, multiple-comparison).
- The ~300 uses of plain "registered" (e.g. "registered floor") are unchanged; the definition sentence covers them.

## Numbers audit, Part 1: abstract and chapters 1, 3, 4, 6 (2026-10-15)

Source: FINAL_FIX_LIST.md. Each row gives the item, location, exact before/after text and evidence.
Additional change: ch3:233 storage rate tightened from 50–53% to 52–53%, because the 49.9% low end is the C5 rate, outside cycles 1–3.

# Fix log (front matter, ch1-4, ch6)

## #15 — thesis_report/core/abstract.tex:22
- BEFORE: `Cycle zero locks a configuration with evaluation-fold storage precision $0.796$ over $289$ stored entries, drawn`
- AFTER: `Cycle zero locks a configuration with storage precision $0.796$ over $289$ stored entries on the cycle-zero validation fold, drawn`
- Evidence: numbers_mismatch_origins A19/O2; aggregate_section_5_7.py:387 (n=500/bench, 1500 ID panel)

## #27 — thesis_report/core/abstract.tex:22
- BEFORE: `with bias-corrected-and-accelerated bootstrap intervals`
- AFTER: `with percentile bootstrap intervals`
- Evidence: scripts/baseline_sig_tests_v2.py:142,161-167 (percentile 2.5/97.5 of 1000 replicates); no replicate count stated in abstract

## #15 — thesis_report/chapters/chapter_3.tex:58
- BEFORE: `over $289$ stored entries on the held-out training-panel evaluation fold under the locked composite`
- AFTER: `over $289$ stored entries on the held-out cycle-zero training-panel validation fold ($500$ samples per benchmark, $1500$ in total) under the locked composite`
- Evidence: A19/O2

## #15 — thesis_report/chapters/chapter_3.tex:73
- BEFORE: `NFR2 & Storage precision anchor at $\tau_{\text{store}} = 0.60$ & Evaluation-fold $\pi_{\text{store}}$ registered`
- AFTER: `NFR2 & Storage precision anchor at $\tau_{\text{store}} = 0.60$ & Held-out $\pi_{\text{store}}$ registered`
- Evidence: A19/O2 (keeps target cell consistent with receipt cell)

## #15 — thesis_report/chapters/chapter_3.tex:73
- BEFORE: `$\pi_{\text{store}} = 0.796$ over $289$ stored entries on the cycle-zero training-panel evaluation fold \\`
- AFTER: `$\pi_{\text{store}} = 0.796$ over $289$ stored entries on the cycle-zero training-panel validation fold \\`
- Evidence: A19/O2

## #15/#31 — thesis_report/chapters/chapter_3.tex:233
- BEFORE: `to the storage class on the cycle-zero evaluation fold (the realised cycle-zero reading is $289$ stored entries over the $1500$-sample training-panel evaluation fold, a storage rate near $19\%$), and the realised per-cycle stream-chunk storage rate sits between $36\%$ and $43\%$ over cycles one to three of the trajectory;`
- AFTER: `to the storage class on the cycle-zero validation fold (the realised cycle-zero reading is $289$ stored entries over the $1500$-sample training-panel validation fold, $500$ samples per benchmark, a storage rate near $19\%$), and the realised per-cycle stream-chunk storage rate sits at about $50\%$ to $53\%$ of the $4700$-sample stream chunk over cycles one to three of the trajectory;`
- Evidence: A19/O2; #31: log_resolution D7 (pi_store_trajectory.csv:14,22: 2443, 2477 /4700 = 52.0%, 52.7%); tab_storage_pool_quality panel b gate-pass 2456/2494/2490 = 52.3-53.1%; 289/1500=19.3%

## #15 — thesis_report/chapters/chapter_6.tex:32
- BEFORE: `whose evaluation-fold precision anchor sits at $0.796$ over $289$ stored entries on the cycle-zero training-panel evaluation fold,`
- AFTER: `whose held-out precision anchor sits at $0.796$ over $289$ stored entries on the cycle-zero training-panel validation fold ($500$ samples per benchmark),`
- Evidence: A19/O2

## #11 — thesis_report/chapters/chapter_6.tex:75
- BEFORE: `closes both pre-specified acceptance gates within the realised horizon: a pooled exact-match lift of $+5.8$ percentage points with zero per-benchmark exact-match regression closes the first gate, and a TruthfulQA exact-match recovery from $0.210$ to $0.310$ above the B1 zero-shot floor closes the second.`
- AFTER: `closes the first of its two pre-specified acceptance gates within the realised horizon: a pooled exact-match lift of $+5.8$ percentage points with zero per-benchmark exact-match regression closes the first gate. The second gate does not close as stated: TruthfulQA exact match recovers from $0.210$ to $0.310$, toward the B1 zero-shot floor of $0.433$, but stays below it.`
- Evidence: tab_per_bench_trajectory.tex TruthfulQA (EM) B1 column 0.433; tab_sig_test.csv:33 zero_shot truthfulqa 0.4333; origins A2/O5

## #35 — thesis_report/chapters/chapter_1.tex:58
- BEFORE: `At cycle zero almost every query falls to the retrieval-augmented tier because neither mechanism has had time to accumulate;`
- AFTER: `At cycle zero about half of the queries ($51.3\%$) fall to the retrieval-augmented tier and none is served from memory, because neither mechanism has had time to accumulate;`
- Evidence: tab_tier_trajectory.tex panel (a) C0: Tier1 0.0%, Tier2 48.7%, Tier3 51.3%

## #48 — thesis_report/chapters/chapter_4.tex:163
- BEFORE: `an entity count, a myth-regex hit indicator`
- AFTER: `an entity count, the logarithm of the largest Wikipedia page-view count among the detected entities, a myth-regex hit indicator`
- Evidence: caem/ruc/v2/feature_spec.json features (log_pageviews_max); scripts/build_ruc_training_set.py:203-215

## #48 — thesis_report/chapters/chapter_4.tex:163
- BEFORE: `three answer-type matches between the question's expected answer type and the passage's surface (year-match, number-match, and a numeric-presence flag),`
- AFTER: `six answer-type features between the question's expected answer type and the passage's surface (whether the question expects a year, whether the passage contains one, and whether the two match, with the same three indicators for numbers),`
- Evidence: feature_spec.json: q_expects_year, passage_has_year, year_match, q_expects_number, passage_has_number, number_match; count now 17+9+10=36

## #49 — thesis_report/chapters/chapter_4.tex:343
- BEFORE: `$C\in\{0.01,\,0.10,\,1.00\}$`
- AFTER: `$C\in\{0.010,\,0.025,\,0.050,\,0.100,\,1.000\}$`
- Evidence: figures/auto/sweep_top.tex:12; origins A22; archive/branch_C_log_historical.md:284

## #19 — thesis_report/chapters/chapter_4.tex:355
- BEFORE: `which compared two candidates on the labelled cycle-zero calibration fold: a stricter cut`
- AFTER: `which compared two candidates on the labelled cycle-zero calibration fold of the calibration sweep run before the main trajectory: a stricter cut`
- Evidence: G 04fe2e7 tab_pi_store_trajectory (478 @ 0.640); numbers_check_ch4_ch5 #23 and :79

## #19 — thesis_report/chapters/chapter_4.tex:355
- BEFORE: `for an empirical precision of approximately sixty-four percent. The two operating points`
- AFTER: `for an empirical precision of approximately sixty-four percent (the main run's own cycle-zero calibration fold admits $575$ entries at $0.746$, \Cref{tab:pi-store-trajectory}). The two operating points`
- Evidence: tab_pi_store_trajectory.tex C0 cal-fold n=575, pi=0.746

## #19 — thesis_report/chapters/chapter_4.tex:355
- BEFORE: `the lower cut admits roughly four-and-a-half times as many correct memories`
- AFTER: `the lower cut admits roughly $4.4$ times as many correct memories`
- Evidence: A: 306/70 = 4.37 (4.5 is 478/105 total admissions)

## #50 — thesis_report/chapters/chapter_4.tex:532
- BEFORE: `clearing the registered floor by eight to eleven percentage points`
- AFTER: `clearing the registered floor by nine to eleven percentage points`
- Evidence: tab_pi_store_trajectory cal-fold [0.732,0.746] - 0.64 = 9.2-10.6 pp

## #53 — thesis_report/chapters/chapter_4.tex:603
- BEFORE: `tightly bounded band of width below $0.03$ across the cycles`
- AFTER: `tightly bounded band of width $0.014$ across the cycles`
- Evidence: tab_pi_store_trajectory cal-fold 0.746-0.732 = 0.014

## #51 — thesis_report/chapters/chapter_4.tex:625
- BEFORE: `$[0.246, 0.277]$`
- AFTER: `$[0.247, 0.276]$`
- Evidence: tab_storage_pool_quality panel b pi 0.753..0.724 -> 1-pi = 0.247..0.276; 606/2456=0.2467, 688/2490=0.2763

## Not changed / skipped
- #11 ch6 "~:102": chapter_6.tex has 96 lines; the only "above the B1 floor" / gate-closing site is :75 (fixed). ch6:27 says the classifier "lifts TruthfulQA exact match by ten percentage points" (0.210 -> 0.310), which is true, so it was left as is. ch5:473 belongs to another agent.
- "+2.0%" C5-vs-B1 EM: does not appear in abstract or ch6 (only ch5:491, owned elsewhere). tab_baseline_pooled gives +2.1%.
- #27: the abstract states no replicate count, so none was added.
- #19: only the ch4:355 part (ch5:301/:595 belong to the ch5 agent). The ch4:532 "pi_store ~ 0.64" anchor wording was not in scope and was left.
- #31: spec range "about 50-53%" applied. For C1-C3 specifically, admissions are 2443/2477 (/4700 = 52.0%, 52.7%; C3 not in csv) and gate passes are 2456/2494/2490 (52.3-53.1%), so "about 52-53%" would be tighter. The 49.9% lower end of the spec is the C5 rate (2343/4700), outside cycles one to three.
- ch2: no assigned items.

## Numbers audit, Part 1: chapter 5 (2026-10-15)

# Fix log: chapter_5.tex

Line numbers are pre-edit (positions in the file before each replacement applied in order; approx).

## #21 (line ~120)
- BEFORE: even though cycle two carries the within-trajectory exact-match and composite-hallucination optima reported in \Cref{tab:per-bench-trajectory} and \Cref{tab:baseline-pooled},
- AFTER: even though cycle two carries the within-trajectory optima of the five-benchmark pooled exact match and composite hallucination metric reported in \Cref{tab:pooled-trajectory} and \Cref{tab:baseline-pooled} (the training-panel pooled exact match of \Cref{tab:per-bench-trajectory} peaks at cycle one, at $0.584$),
- Evidence: T tab_pooled_trajectory C2 EM 0.544 max / CHM 0.107 min; tab_per_bench_trajectory training pooled EM peaks C1 0.584

## #40 (line ~127)
- BEFORE: FEVER and TriviaQA exact match readings sit within a narrow band of two percentage points across the six cycles,
- AFTER: FEVER and TriviaQA exact match readings sit within a band of six percentage points across the six cycles, with a net cycle-zero to cycle-five change of about one percentage point on each,
- Evidence: T tab_per_bench_trajectory FEVER 0.477-0.533, TriviaQA 0.453-0.493, net -1.0 pp each; StrategyQA 0.640 at C3-C5; tab_halluc_subtypes confab 0.087/0.223/0.239/0.268/0.270/0.269

## #40 (line ~127)
- BEFORE: while StrategyQA stabilises at $0.640$ from cycle one onward.
- AFTER: while StrategyQA stabilises at $0.640$ from cycle three onward.
- Evidence: T tab_per_bench_trajectory FEVER 0.477-0.533, TriviaQA 0.453-0.493, net -1.0 pp each; StrategyQA 0.640 at C3-C5; tab_halluc_subtypes confab 0.087/0.223/0.239/0.268/0.270/0.269

## #40 (line ~127)
- BEFORE: Confident confabulation rises monotonically from $0.087$ at cycle zero to $0.269$ at cycle five,
- AFTER: Confident confabulation rises from $0.087$ at cycle zero to approximately $0.27$ by cycle three and plateaus there ($0.269$ at cycle five),
- Evidence: T tab_per_bench_trajectory FEVER 0.477-0.533, TriviaQA 0.453-0.493, net -1.0 pp each; StrategyQA 0.640 at C3-C5; tab_halluc_subtypes confab 0.087/0.223/0.239/0.268/0.270/0.269

## #39 (line ~138)
- BEFORE: ($+17.7$ percentage points at the best cycle, $+16.0$ at the trajectory-end cycle)
- AFTER: ($+20.0$ percentage points at the best cycle, $+16.0$ at the trajectory-end cycle)
- Evidence: T tab_per_bench_trajectory TriviaQA B1->C2 +0.200

## #41 (line ~143)
- BEFORE: The gate's separation between store and the next-highest outcome widens monotonically across cycles, from a $14$ percentage-point gap at cycle one to a $24$-to-$28$ percentage-point gap from cycle two onward, the empirical signal that the cycle-boundary refit is sharpening the gate's discrimination rather than degrading it under distribution drift.
- AFTER: The gate's separation between store and the next-highest outcome widens from a $14$ percentage-point gap at cycle one to a $24$-to-$30$ percentage-point gap from cycle two onward ($30$, $28$, and $24$ points at cycles two, three, and five, so the widening is not monotone), the empirical signal that the cycle-boundary refit sharpened the gate's discrimination after cycle one rather than degrading it under distribution drift.
- Evidence: T tab_decision_distribution store EM minus next: 14/30/28/24 pp

## #42 (line ~147)
- BEFORE: The within-bench ordering $\pi_{\text{store}} > \mathrm{prec}(\text{defer}) > \mathrm{prec}(\text{discard})$ holds at every cell of the matrix, the structural receipt that the fixed-threshold gate sorts the candidate pool by correctness on each benchmark family rather than only at the panel level.
- AFTER: The storage class is more precise than the defer class in every row, and the full within-bench ordering $\pi_{\text{store}} > \mathrm{prec}(\text{defer}) > \mathrm{prec}(\text{discard})$ holds in five of the ten rows of the matrix (TriviaQA and TruthfulQA at both cycles, StrategyQA at cycle five); in the other five (FEVER and CommonsenseQA at both cycles, StrategyQA at cycle two) the discard class, which holds between one and one hundred and three entries in those rows, is more precise than the defer class. The ordering is therefore a partial receipt that the fixed-threshold gate sorts the candidate pool by correctness on each benchmark family rather than only at the panel level.
- Evidence: T tab_decision_matrix_trajectory defer vs discard precision per row (e.g. C2 FEVER 0.451<0.563, C5 FEVER 0.523<0.833); store>defer in all 10 rows

## #17 (line ~163)
- BEFORE: seven of the nine thousand evaluation queries across the six cycles, a rate of $0.13$ percent.
- AFTER: seven of the nine thousand evaluation queries across the six cycles, a rate of approximately $0.08$ percent.
- Evidence: A 7/9000=0.078%

## #43 (line ~167)
- BEFORE: and the ordering is preserved across the trajectory rather than reversed under the SIL distillation step.
- AFTER: and on those four benchmarks the ordering is preserved at cycles two and five rather than reversed under the SIL distillation step, with two exceptions: FEVER at cycle two ($51.6$ against $54.5$) and StrategyQA at cycle five ($63.9$ against $64.0$).
- Evidence: T tab_per_tier_em C2 FEVER T2 51.6<T3 54.5; C5 StrategyQA 63.9<64.0

## #18 (line ~216)
- BEFORE: with the $\tau_{\text{train}}$ filter removing between $35\%$ and $49\%$ of the wrong-stored entries
- AFTER: with the $\tau_{\text{train}}$ filter removing between $56\%$ and $73\%$ of the wrong-stored stream-chunk entries
- Evidence: T tab_storage_pool_quality panel b: removed wrong=(em-wrong - n_pass*(1-pi_SIL))/em-wrong = 60.7/55.7/60.7/57.8/73.1%

## #47 (line ~246)
- BEFORE: a twenty-fold increase across the four committed cycles.
- AFTER: about a twenty-four-fold increase across the four committed cycles.
- Evidence: T tab_self_correction_check 431->10,214 = 23.7x

## #13 (line ~250)
- BEFORE: Across the four committed retroverify passes the pooled $1\,094$ pruned entries are $38.4\%$ em-wrong
- AFTER: Across the four committed retroverify passes the runtime log records $1\,061$ prunes; the offline reconstruction of \Cref{tab:em-conditional-breakdown} identifies $1\,094$ pruned entries from memory-snapshot differences, and these are $38.4\%$ em-wrong
- Evidence: L experiment.log:5710,19133,33683,60457 (56/271/352/382=1061); :16404,30357,55957 (131/220/299=650); T tab_em_conditional_breakdown is offline reconstruction (1094/620)

## #13 (line ~250)
- BEFORE: The cohort of $620$ entries promoted from the deferred buffer is $55.6\%$ em-correct:
- AFTER: Of the $650$ promotions from the deferred buffer that the runtime log records, the reconstruction identifies $620$, and this reconstructed cohort is $55.6\%$ em-correct:
- Evidence: L experiment.log:5710,19133,33683,60457 (56/271/352/382=1061); :16404,30357,55957 (131/220/299=650); T tab_em_conditional_breakdown is offline reconstruction (1094/620)

## #13 (line ~254)
- BEFORE: relative to the $1\,094$ retroverify prunes and $620$ deferred-buffer promotions
- AFTER: relative to the $1\,061$ retroverify prunes and $650$ deferred-buffer promotions
- Evidence: L experiment.log:5710,19133,33683,60457 (56/271/352/382=1061); :16404,30357,55957 (131/220/299=650); T tab_em_conditional_breakdown is offline reconstruction (1094/620)

## #15 (line ~294)
- BEFORE: admits $289$ entries on the held-out cycle-zero evaluation fold at $\pi_{\text{store}} = 0.796$ across the three training benchmarks,
- AFTER: admits $289$ entries on the held-out cycle-zero validation fold ($500$ samples per benchmark, $1\,500$ on the training panel) at $\pi_{\text{store}} = 0.796$ across the three training benchmarks,
- Evidence: L run_phase1a_ruc.log:71 eval=500/bench; full_run/experiment.log:47 eval=300; T cross_benchmark_summary n=500

## #15 (line ~317)
- BEFORE: On the held-out evaluation fold the storage class admits $63$, $145$, and $81$ entries
- AFTER: On the held-out cycle-zero validation fold the storage class admits $63$, $145$, and $81$ entries
- Evidence: L run_phase1a_ruc.log:71 eval=500/bench; full_run/experiment.log:47 eval=300; T cross_benchmark_summary n=500

## #15 (line ~317)
- BEFORE: over the $289$ stored entries on the $1500$-sample ID evaluation fold.
- AFTER: over the $289$ stored entries on the $1500$-sample ID cycle-zero validation fold ($500$ per benchmark).
- Evidence: L run_phase1a_ruc.log:71 eval=500/bench; full_run/experiment.log:47 eval=300; T cross_benchmark_summary n=500

## #15 (line ~534)
- BEFORE: (eval-fold $\pi_{\text{store}} = 0.796$ on $n=289$)
- AFTER: (cycle-zero validation-fold $\pi_{\text{store}} = 0.796$ on $n=289$)
- Evidence: L run_phase1a_ruc.log:71 eval=500/bench; full_run/experiment.log:47 eval=300; T cross_benchmark_summary n=500

## #15 (line ~575)
- BEFORE: The evaluation-fold storage precision shelf documented in \Cref{sec:adj-cal-eval-gap}
- AFTER: The cycle-zero validation-fold storage precision shelf documented in \Cref{sec:adj-cal-eval-gap}
- Evidence: L run_phase1a_ruc.log:71 eval=500/bench; full_run/experiment.log:47 eval=300; T cross_benchmark_summary n=500

## #15 (line ~585)
- BEFORE: ($289$ stored entries on the pooled training-panel evaluation fold)
- AFTER: ($289$ stored entries on the pooled training-panel cycle-zero validation fold, $500$ samples per benchmark and $1\,500$ in all)
- Evidence: L run_phase1a_ruc.log:71 eval=500/bench; full_run/experiment.log:47 eval=300; T cross_benchmark_summary n=500

## #34 (line ~301)
- BEFORE: The configuration is preferred over the alternative $V_{\tau=0.65}$ in \Cref{tab:sweep-top} for two reasons.
- AFTER: The configuration is preferred over the alternative $V_{\tau=0.65}$ from the registered sweep for two reasons.
- Evidence: G 449dacb: sweep_top.tex now locked-parameter table, no V_0.65 row

## #19 (line ~301)
- BEFORE: First, the lower cut admits roughly four-and-a-half times as many correct memories at the same per-store precision shelf at cycle close,
- AFTER: First, the lower cut admits roughly $4.4$ times as many correct memories at the same per-store precision shelf at cycle close ($306$ against $70$ on the cycle-zero calibration fold of the calibration sweep run before the main trajectory),
- Evidence: ch4:355 counts 306/70 (=4.37), 105/478; G 04fe2e7 tab_pi_store_trajectory C0 n=478 pi=0.640; wording matched to edited ch4:355

## #45 (line ~301)
- BEFORE: with a tenfold tighter L2 penalty than the library default
- AFTER: with a hundredfold tighter L2 penalty than the library default
- Evidence: A 1.0/0.01=100

## #46 (line ~308)
- BEFORE: (an absolute movement of $0.218$, more than fivefold the cycle-zero magnitude)
- AFTER: (an absolute movement of $0.218$, more than threefold the cycle-zero magnitude)
- Evidence: T tab_calibration_trajectory p_entail 0.067->0.285: delta 0.218=3.3x, ratio 4.3x

## #27 (line ~376)
- BEFORE: The confidence interval is the bias-corrected and accelerated (BCa) interval at the $0.025$ and $0.975$ quantiles of the bootstrap distribution, with the standard $B = 10\,000$ replicates registered in the runbook.
- AFTER: The confidence interval is the percentile interval at the $0.025$ and $0.975$ quantiles of the bootstrap distribution, with $B = 1\,000$ replicates. The interval on the composite-hallucination delta is approximate: it resamples only the baseline's per-subtype rates, with CAEM's composite hallucination value held fixed, so it can exclude the point difference.
- Evidence: C eval/metrics.py:850 bootstrap_ci percentile, n_bootstrap=1000; scripts/baseline_sig_tests_v2.py:237; C scripts/baseline_sig_tests_v2.py _approx_chm_ci: binomial resample of baseline subtype rates, caem_chm held fixed

## #20 (line ~469)
- BEFORE: on the smallest test fold (the three hundred-sample StrategyQA panel that is also
- AFTER: on its three-hundred-sample evaluation fold (every benchmark's evaluation fold has three hundred samples; StrategyQA's $387$-sample test fold is the smallest test fold), the StrategyQA panel that is also
- Evidence: tab:benchmark-pools: eval 300 all, StrategyQA test 387; L experiment.log:66

## #11 (line ~473)
- BEFORE: The second gate predicts a recovery of TruthfulQA exact match toward the B1 zero-shot floor of $0.27$; the realised reading is $0.310$ in the classifier-on arm against $0.210$ in the classifier-off arm, lifting above the B1 floor and closing the second gate as well.
- AFTER: The second gate predicts a recovery of TruthfulQA exact match toward the B1 zero-shot reference of $0.433$; the realised reading is $0.310$ in the classifier-on arm against $0.210$ in the classifier-off arm, which recovers toward the B1 reference but stays below it, so the second gate as stated does not close.
- Evidence: T tab_per_bench_trajectory B1 TQA 0.433; tab_ruc_ablation 0.210/0.310; L tab_sig_test.csv:33

## #7 (line ~491)
- BEFORE: reads $+7.4\%$ exact match and $-11.0\%$ composite hallucination metric at cycle two and $+2.0\%$ exact match
- AFTER: reads $+7.4\%$ exact match and $-11.0\%$ composite hallucination metric at cycle two and $+2.1\%$ exact match
- Evidence: T tab_baseline_pooled B1 C5 +2.1%; tab_sig_test(_c2): 5 vanilla_ft CHM cells n/a -> 35 CHM contrasts

## #7 (line ~491)
- BEFORE: on nine of forty per-benchmark contrasts on the exact-match axis and on twenty-one of forty contrasts on the composite-hallucination axis at the cycle-five anchor, and on fourteen and twenty of forty contrasts respectively at the best-cycle anchor of \Cref{tab:sig-test-c2},
- AFTER: on nine of forty per-benchmark contrasts on the exact-match axis and on twenty-one of thirty-five contrasts on the composite-hallucination axis at the cycle-five anchor (the five vanilla fine-tuning composite-hallucination cells are not available), and on fourteen of forty and twenty of thirty-five contrasts respectively at the best-cycle anchor of \Cref{tab:sig-test-c2},
- Evidence: T tab_baseline_pooled B1 C5 +2.1%; tab_sig_test(_c2): 5 vanilla_ft CHM cells n/a -> 35 CHM contrasts

## #34 (line ~595)
- BEFORE: is read off the sweep panel in \Cref{tab:sweep-top}. The two top-tier cut-point variants in the table differ on
- AFTER: is read off the registered cycle-zero sweep whose locked configuration is reported in \Cref{tab:sweep-top}. The two top-tier cut-point variants of the sweep differ on
- Evidence: G 449dacb: sweep_top.tex now locked-parameter table, no V_0.65 row

## #19 (line ~595)
- BEFORE: the lower cut admits roughly four-and-a-half times as many correct memories at the same per-store precision shelf, which is the rate-volume frontier
- AFTER: the lower cut admits roughly $4.4$ times as many correct memories at the same per-store precision shelf ($306$ against $70$ on the cycle-zero calibration fold of the calibration sweep run before the main trajectory), which is the rate-volume frontier
- Evidence: ch4:355 counts 306/70 (=4.37), 105/478; G 04fe2e7 tab_pi_store_trajectory C0 n=478 pi=0.640; wording matched to edited ch4:355

## #29 (line ~609)
- BEFORE: reaches the registered upper bound of the optimisation envelope from cycle one onward, so single-parameter temperature scaling on this surface is functionally inert across the trajectory;
- AFTER: reaches the registered upper bound of the optimisation envelope at the cycle-zero fit ($T = 20.086$), so single-parameter temperature scaling on this surface is functionally inert at the cycle-zero operating point (the cycle-boundary refits return $1.089$, $0.836$, $0.740$, and $0.869$ at cycles one, two, three, and five, \Cref{tab:calibration-trajectory});
- Evidence: T tab_calibration_trajectory pooled T 20.086/1.089/0.836/0.740/0.869

## #29 (line ~609)
- BEFORE: and the safety override even at the saturated temperature scalar.
- AFTER: and the safety override even at the saturated cycle-zero temperature scalar.
- Evidence: T tab_calibration_trajectory pooled T 20.086/1.089/0.836/0.740/0.869

## #30 (line ~609)
- BEFORE: The deferred-entry reconsideration pass specified in \Cref{sec:deferred-reconsider} is reported in two phases across the realised trajectory. Ongoing analysis at the close of cycle four established that the cycle-boundary reconsideration pass had not been engaging the deferral buffer through cycles one through four under the trajectory's initial orchestration; the buffer accumulated correctly under the four-outcome decision tree but was not revisited under the post-fine-tune verifier at cycle boundaries. The orchestration was corrected at the cycle-four-close transition and the trajectory was resumed at cycle five with the reconsideration pass engaged at every subsequent cycle boundary under the registered time-to-live of two cycles. The cycle-five boundary therefore exhibits a one-time backlog reconciliation as the entries deferred during cycles one through four are rescored together under the cycle-five verifier. The equilibrium-saturation gate fires after the cycle-five close, so the realised trajectory does not extend into the post-reconciliation steady-state window of the deferred-pool design; the within-horizon receipt for the deferred-pool half of the two-track design is therefore the cycle-five reconciliation reading alone, with the steady-state two-track survivability behaviour specified in \Cref{sec:cycle-boundary} registered as a future-work measurement at a longer horizon. The correction to engage the reconsideration pass mid-trajectory is registered as a methodology refinement applied during ongoing analysis rather than a post-hoc reinterpretation.
- AFTER: The deferred-entry reconsideration pass specified in \Cref{sec:deferred-reconsider} engages at every successful cycle boundary of the realised trajectory under the registered time-to-live of two reconsider invocations: it promotes $131$ entries at cycle two, $220$ at cycle three, and $299$ at cycle five (\Cref{tab:survival-distribution}), while the cycle-one pass finds an empty buffer and the cycle-four pass is skipped under the asymmetric-rollback contract. The equilibrium-saturation gate fires after the cycle-five close, so the within-horizon receipt for the deferred-pool half of the two-track design is these three reconsideration passes, with the steady-state two-track survivability behaviour specified in \Cref{sec:cycle-boundary} registered as a future-work measurement at a longer horizon.
- Evidence: T tab_survival_distribution promoted 0/131/220/-/299; L as #13

## #62
Applied within the #27 replacement (ch5:376): sentence "The interval on the composite-hallucination delta is approximate: ..." Evidence: C _approx_chm_ci.

## Skipped / partially skipped
- #44 (ch5:216 u_stored bucket monotonicity) SKIPPED. Spec proposes "eval fold: not monotone at C0, C3 and C5", but tab_u_stored_distribution panel (a) is also non-monotone at C2 (0.729 -> 0.716), so eval fold is monotone only at C1 and C4 (non-monotone at C0, C2, C3, C5). Stream chunks monotone at every cycle (confirmed). Current text ("every cycle except cycle two") remains false. Suggested true wording: "on the stream chunks the em-rate increases with the bucket index at every cycle; on the evaluation fold it does so only at cycles one and four".
- #46 second half (ch5:308 FEVER/TriviaQA temperature "an order of magnitude above CommonsenseQA at every cycle") SKIPPED. Spec says 4.8-10.5x; table gives FEVER 10.50/8.19/6.45/4.84x and TriviaQA 10.52/8.28/6.61/10.74x (C5 TriviaQA 11.040/1.028 = 10.74), i.e. 4.8-10.7x. Spec value disagrees.
- #34 at ch5:294: the sentence there ("locked configuration and its parameter rationale reported in tab:sweep-top") is already accurate; no change. Fixed at ch5:301 and ch5:595.
- #12 not applied (instructed); ch5:181 untouched.

## Related sites noticed, not edited (outside assigned item scope)
- ch5:491 end: "the supplementary retrieval-utility classifier ablation ... closes the TruthfulQA exact-match regression ... lifting TruthfulQA exact match from 0.210 to 0.310" -- same issue as #11 (0.310 < B1 0.433), not in the assigned ch5:473 part.
- ch5:208 "pre-specified cycle-zero anchor at pi_store ~ 0.64" -- same pre-main-run calibration sweep (#19 anchor), not relabelled.
- ch5:341 / H1 row (ch5:506) "training-pooled evaluation fold" / "held-out cycle-zero evaluation fold" Cohen's d +0.635 may be the same cycle-zero validation fold as #15; not in spec's site list.

## Caption changes needed in figures/auto (not mine)
- tab_decision_matrix_trajectory caption: "holds at every cell" -> five of ten rows (#42).
- tab_storage_pool_quality caption: "removes between 35% and 49% of those wrong entries" -> 56%-73% (#18).
- tab_u_stored_distribution caption: eval-fold exceptions (#44; C0, C2, C3, C5 non-monotone).
- tab_per_tier_em caption: "the gap is preserved across the trajectory" -> exceptions C2 FEVER, C5 StrategyQA (#43).
- tab_sig_test / tab_sig_test_c2 captions (in ch5 \caption, already ch5-owned? No: captions at ch5:404/414 are in chapter_5.tex) -- #62 sentence placed in sig-bootstrap prose instead.
- sweep_top.tex "tenfold" -> already changed to "hundredfold" by another agent (#45).

## Numbers audit, Part 1: appendices C, D, E (2026-10-15)

# Fix log: appendix C, D, E (FINAL_FIX_LIST Part 1)

Scope: thesis_report/appendix/appendix_c.tex, appendix_d.tex, appendix_e.tex only. No git state changes.
Full unified diff: /tmp/claude-0/-home-user-caem-thesis/96eedb08-eb50-59ca-a1c7-fdab15fb78c5/scratchpad/appCDE.diff
All edits were made by this agent: appendix C and D before the rate-limit interruption, appendix E after it. Nothing was redone; `git diff` confirmed the earlier edits.
Line numbers are post-edit.

## #1 Significance tallies
Recount of figures/auto/tab_sig_test.tex (C5) and tab_sig_test_c2.tex (C2). A win is `*` with CAEM better; a defeat is `(B)`.
- C5 EM: 9 wins (TriviaQA: cot, fiveshot, SE, vanilla_ft, zero_shot; CSQA: cot_rag, rag; TQA: fiveshot; SQA: fiveshot) / 3 defeats (TQA: cot, flare, zero_shot).
- C5 CHM: 21 wins (FEVER 6, TriviaQA 5, CSQA 3, TQA 3, SQA 4) / 3 defeats (TQA: cot=B2, semantic_entropy=B8, zero_shot=B1). FLARE TQA CHM is +0.009 and not significant.
- C2 EM: 14 wins (TriviaQA 8, CSQA 2, TQA 3, SQA 1) / 3 defeats (TQA: cot, flare, zero_shot).
- C2 CHM: 20 wins (FEVER 7, TriviaQA 6, CSQA 3, TQA 1, SQA 3) / 4 defeats (TQA: B1, B2, B7, B8).
- These agree with abstract.tex:22 (14/3, 9/3) and chapter_5.tex:394 and :414.

| file:line | BEFORE | AFTER |
|---|---|---|
| appendix_d:155 | "...are thirteen exact-match wins and three defeats and twenty-three composite-hallucination wins and four defeats; at the within-trajectory best cycle the receipts are seventeen exact-match wins and two defeats and twenty-six composite-hallucination wins and four defeats. The four composite-hallucination defeats at both anchors are confined to TruthfulQA against the high-exact-match retrieval-free baselines (B1 zero-shot, B2 chain-of-thought, B7 FLARE, B8 semantic-entropy)," | "...are nine exact-match wins and three defeats and twenty-one composite-hallucination wins and three defeats; at the within-trajectory best cycle the receipts are fourteen exact-match wins and three defeats and twenty composite-hallucination wins and four defeats. The composite-hallucination defeats at both anchors are confined to TruthfulQA against the high-exact-match retrieval-free baselines (B1 zero-shot, B2 chain-of-thought, and B8 semantic-entropy at both anchors, plus B7 FLARE at the best cycle)," |
| appendix_d:98 | "TruthfulQA carries the four composite-hallucination defeats discussed in" | "TruthfulQA carries the three composite-hallucination defeats at the cycle-five anchor discussed in" |
| appendix_e:235 | "Expected: thirteen exact-match wins and three defeats at C5; twenty-three composite-hallucination wins and four defeats at C5; seventeen exact-match wins and two defeats at C2; twenty-six composite-hallucination wins and four defeats at C2." | "Expected: nine exact-match wins and three defeats at C5; twenty-one composite-hallucination wins and three defeats at C5; fourteen exact-match wins and three defeats at C2; twenty composite-hallucination wins and four defeats at C2." |

## #22 Appendix C "CES" column relabelled (values unchanged)
Evidence: log_resolution.md D6 (equilibrium_fit.json:4-6 history = 0.5573, 0.5673, 0.5473...; theorem_review.json:368 "pooled eval-fold EM").
| file:line | BEFORE | AFTER |
|---|---|---|
| appendix_c:102 (header) | `\textbf{MMLU $\boldsymbol{\rho}$} & \textbf{CES} \\` | `\textbf{MMLU acc.} & \textbf{Pooled EM} \\` (also covers #23) |
| appendix_c:94 | "together with the worst-probe retention ratio and the composite evaluation score on each cycle." | "together with the worst-probe retention ratio, the post-training MMLU probe accuracy, and the pooled evaluation-fold exact match on each cycle (the series the equilibrium fit reads)." |
| appendix_c:111 caption (appended) | "...of the registered horizon are skipped.}" | "...of the registered horizon are skipped. The MMLU column is the post-training MMLU probe accuracy against the pristine anchor of $0.610$; the pooled-EM column is the pooled evaluation-fold exact match, the series stored as the history array of the cycle-five equilibrium-fit record.}" |
| appendix_c:115 | "over the five-cycle composite-evaluation-score history;" | "over the five-cycle pooled evaluation-fold exact-match history (the pooled-EM column of \Cref{tab:app-per-cycle-equilibrium}, which the record stores as its composite-evaluation-score history);" |

## #23 MMLU column: off by one cycle; shown as accuracies
Chose the accuracy option and relabelled the column "MMLU acc.". Evidence: experiment.log:3649 (C1 0.635), :16105 (C2 ratio 0.9754 × 0.61 = 0.595), :29739 (C3 0.9918 × 0.61 = 0.605), :44266 (C4 0.6212), :55961 (C5 0.999 → 0.609). The experiment_summary MMLU % (C0..C5 = 63.5, 63.5, 59.5, 60.5, 62.12, 60.91) is in numbers_mismatch_origins O15. The pristine 0.610 comes from experiment.log:1998.
| file:line | BEFORE | AFTER |
|---|---|---|
| appendix_c:104 C1 | 0.635 | 0.635 (unchanged) |
| appendix_c:105 C2 | `$0.949$ & $0.635$` | `$0.949$ & $0.595$` |
| appendix_c:106 C3 | `$0.976$ & $0.595$` | `$0.976$ & $0.605$` |
| appendix_c:107 C4 | `$\mathbf{0.886}$ & $0.605$` | `$\mathbf{0.886}$ & $0.621$` |
| appendix_c:108 C5 | 0.609 | 0.609 (unchanged) |

## #55
Evidence: the appendix's own table (C2–C5 T = 0.836/0.740/0.869) and verdict rows (C1–C3 CONTINUE).
| file:line | BEFORE | AFTER |
|---|---|---|
| appendix_c:16 | "stabilises in the band $[0.74, 1.09]$ from cycle two onward." | "stabilises in the band $[0.74, 0.87]$ from cycle two onward." |
| appendix_c:94 | "records four CONTINUE verdicts at cycles one through three," | "records three CONTINUE verdicts at cycles one through three," |

## #56 Consolidation column added to the memory table (5 → 6 data columns, 7 in total)
Evidence: figures/auto/tab_consolidation_trajectory.tex episodes removed: C1 0, C2 3, C3 11, C4 ---, C5 44. C0 is set to 0 because pool 431 = seed 431.
Reconciliation (admissions − prunes + promotions − consolidation):
- C1: 2443 − 56 = 2387, which matches.
- C2: 2477 − 271 + 131 − 3 = 2334 against 2333. The residual of 1 matches the consolidation pass seeing 2,948 against 2,818 + 131 = 2,949 (experiment.log:16405).
- C3: 2469 − 352 + 220 − 11 = 2326, which matches.
- C4: 2484, which matches.
- C5: 2343 − 382 + 299 − 44 = 2216 against 2214. The residual of 2 matches the pass seeing 10,258 against 9,961 + 299 = 10,260 (experiment.log:55958).
| file:line | BEFORE | AFTER |
|---|---|---|
| appendix_c:74 | `\begin{tabular}{@{}lrrrrr@{}}` | `\begin{tabular}{@{}lrrrrrr@{}}` |
| appendix_c:76–77 header | ... `\textbf{Deferred-buffer} & \textbf{Net} \\` / `\textbf{promotions} & \textbf{change} \\` | ... `\textbf{Deferred-buffer} & \textbf{Consolidation} & \textbf{Net} \\` / `\textbf{promotions} & \textbf{removals} & \textbf{change} \\` |
| appendix_c:79–84 rows | C0 `$0$ & $0$ & $+431$`; C1 `$56$ & $0$ & $+2\,387$`; C2 `$131$ & $+2\,333$`; C3 `$220$ & $+2\,326$`; C4 `--- & --- & $+2\,484$`; C5 `$2\,339$ & $382$ & $299$ & $+2\,214$` | inserted column values 0 / 0 / 3 / 11 / --- / 44 (C5 admissions → 2,343, see #57) |
| appendix_c:68 prose | "together with the four mechanisms that move the pool size across the cycle boundary: stream-chunk admissions ..., retroactive re-verification prunes ..., deferred-buffer promotions from the bounded reconsider queue, and the cycle's net change in pool size." | "together with the four mechanisms that move the pool size across the cycle boundary (stream-chunk admissions ..., retroactive re-verification prunes ..., deferred-buffer promotions from the bounded reconsider queue, and consolidation removals of near-duplicate episodes) and the cycle's net change in pool size." |
| appendix_c:68 prose | "The cycle-four row records zero retroverify prunes and zero deferred-buffer promotions under the asymmetric-rollback contract that skips both passes on the aborted cycle;" | "The cycle-four row records no retroverify prunes, deferred-buffer promotions, or consolidation removals under the asymmetric-rollback contract that skips these passes on the aborted cycle;" |
| appendix_c:87 caption | "the retroverify and deferred-reconsider passes are skipped ..." | "the retroverify, deferred-reconsider, and consolidation passes are skipped ..." |
| appendix_c:87 caption (inserted before "Sources:") | — | "...The consolidation-removal column is the episodes-removed count of \Cref{tab:consolidation-trajectory}. Admissions minus prunes plus promotions minus consolidation removals reproduces the net change at every cycle except a residual of one entry at cycle two and two entries at cycle five; the residual is promoted entries dropped as duplicates of entries already in the store (the cycle-two consolidation pass sees $2\,948$ entries against $2\,818 + 131 = 2\,949$)." |

## #57 Novelty-filter footnote; C5 admissions changed to 2,343
Evidence: log_resolution.md D7 (12,175 − 9,832 = 2,343; experiment.log:60457 store 9,832 after retroverify); caem/memory/store.py:301 `is_novel`.
| file:line | BEFORE | AFTER |
|---|---|---|
| appendix_c:84 | `C5 & $12\,175$ & $2\,339$ & ...` | `C5 & $12\,175$ & $2\,343$ & ...` |
| appendix_c:87 caption | "stays in the band $[2\,339, 2\,484]$" | "stays in the band $[2\,343, 2\,484]$" |
| appendix_c:87 caption (inserted) | — | "The admissions count the entries actually added to the store; they sit slightly below the gate-pass counts $n_{\text{stored}}$ of \Cref{tab:storage-pool-quality} because the memory store's novelty filter rejects near-duplicates of existing entries between the gate and the insert. The cycle-five admission count is the cycle-close pool minus the post-re-verification pool ($12\,175 - 9\,832$)." |
Note: C3 admissions 2,469 have no log source (D7 says NOT FOUND). They are left unchanged because they reconcile exactly in the new column arithmetic.

## #58
| appendix_d:39 | "($\chi^{2} = 0.010$, $p_{\text{raw}} = 0.92$, $n_{\text{paired}} = 300$); the two systems agree on every TruthfulQA correctness pattern at the per-sample level." | "($\chi^{2} = 0.010$, $p_{\text{raw}} = 0.92$, $n_{\text{paired}} = 300$)." |
Evidence: tab_sig_test semantic_entropy truthfulqa Δ = −0.007.

## #59 (appendix D part only; appendix_f is not mine)
| appendix_d:96 | "is multiplied by $m - k$ (equivalently the textbook $m + 1 - i$ ...)" | "is multiplied by $m - k + 1$ (equivalently the textbook $m + 1 - i$ ...)" |
Evidence: the Holm table rank 1 uses ×8 (2.4e-5 → 1.9e-4).

## #24
| appendix_e:166 | "The held-out evaluation pass consumes approximately seventy-five to ninety-five minutes across the five-benchmark panel." | "The run log does not time the held-out evaluation pass on its own; the measured window from the end of the fine-tune to the end of the retroactive re-verification pass, which holds the calibration-fold scoring, the held-out evaluation pass, and the re-verification pass, is five hours forty-five minutes at cycle one, nine hours forty-four minutes at cycle two, thirteen hours fifty-four minutes at cycle three, and twenty-one hours fifty minutes at cycle five." |
Evidence: log_resolution.md U12 (experiment.log:3659→5710, 16408→19133, 30361→33683, 55961→60458).
FLAG (not changed; not in the approved list): the same paragraph still says "The cycle close consumes approximately five to seven hours end-to-end", and appendix_e:93 says "Per-cycle wall-time is approximately five to seven hours". The measured windows (up to 21 h 50 m at C5) contradict both, and so do the "twenty to thirty minutes" retroverify and "fifteen to twenty minutes" recalibration figures. They need an owner decision; Part 3 calls them step runtimes.

## #25
| appendix_e:21 | "because the source benchmark's development split admits only six hundred and eighty-seven labelled samples," | "because the source split the loader reads (the ChilleD StrategyQA test split) admits only six hundred and eighty-seven labelled samples," |
Evidence: experiment.log:66 "StrategyQA: 687 samples loaded from ChilleD/StrategyQA:test." appendix_e:17 has no "dev" wording for StrategyQA, so it is unchanged.

## #33
| appendix_e:168 | "the gradient accumulation step count rises from four to sixteen (preserving the effective batch size at thirty-two)," | "the gradient accumulation step count rises from four to eight (preserving the effective batch size at sixteen)," |
Evidence: caem/config.py:727–741 (batch 4 × accum 4 = 16; TARGET 16 since 2026-04-27); appendix_a:151–152; 2 × 8 = 16.
Note: scripts/hardware.py:46 still has TARGET_EFFECTIVE_BATCH_SIZE = 32. That is a code inconsistency outside my scope.

## #60
| appendix_e:168 | "so the full trajectory on a downgraded GPU consumes approximately twelve to sixteen GPU-days of wall-clock time." | "so the full trajectory on a downgraded GPU takes correspondingly longer than on the production envelope, by the same factor." |
Reason: 12–16 cannot be derived. From the thesis's own numbers, 5–7 h/cycle × 6 cycles × 2.2–2.6 ≈ 2.8–4.6 days, and 960 GPU-h × 2.2–2.6 ≈ 88–104 days. The 5–7 h/cycle base is itself contradicted by the measured windows (#24). No defensible figure exists, so the sentence now asserts no number.

## #61
| appendix_e:211 | "Expected: $12\,170$ entries." | "Expected: $12\,175$ entries." |
Evidence: appendix C memory table; O24.

## Part 3 attribution
| appendix_e:164 | "...six hundred and seventy-two United States dollars (forty days at twenty-four hours per day at the on-demand rate)." | "...six hundred and seventy-two United States dollars (from the Vast.ai billing records; forty days at twenty-four hours per day at the on-demand rate)." |
Added once. No numbers changed. The 8–10 GPU-h baseline figure (:97) is left as is.

## Skipped / not applied
- #2–#6: these need p-values (instructed not to apply). The stale rows are listed below.
- #59 appendix_f part: not my file.

## Stale appendix D content awaiting p-values (#2–#6) — untouched
- **#2** appendix_d:52, B6 TriviaQA contingency row: `0.190 & +0.263 & 56.860 & <1e-12`. Current table: EM_base 0.227, Δ +0.227, significant `*`. χ² and p need recomputation.
- **#3** appendix_d:53, B7 FLARE StrategyQA row: `0.250 & +0.390 & 76.891 & <1e-12`. Current: 0.600, +0.040, not significant.
- **#2/#3 prose** appendix_d:39: "The B6 vanilla-fine-tune cell on TriviaQA records the largest single contingency gap in the panel ($\chi^{2} = 56.9$ at $n = 300$). The B7 FLARE cell on StrategyQA records a similarly large gap ..."
- **#2/#3 caption** appendix_d:59: "The B6 TriviaQA cell is the architecture's largest single-contrast lift in the panel; the B7 StrategyQA cell is the next-largest."
- **#4** appendix_d:84, B6 CSQA bootstrap row: `+0.717 & [+0.663, +0.767] & supported (largest panel lift)`. Current: −0.063 [−0.120, −0.010]. The row needs replacing or relabelling, and so do the prose at appendix_d:72 ("the architecture's largest single-contrast bootstrap interval in the panel") and the caption at appendix_d:87 ("The B6 CommonsenseQA contrast is the architecture's largest ...").
- **#5** appendix_d:108–115, TruthfulQA Holm table, all 8 rows. FLARE (row 8, raw 0.923, "no") is now a significant defeat (Δ −0.150 (B)); the order and Holm p need recomputing. Caption at appendix_d:118 has "the two CAEM defeats on TruthfulQA exact match (B1 ..., B2 ...)", which should be three (B1, B2, B7). It also has "B5 five-shot CoT is the lone CAEM win" and "The four CAEM defeats on the composite-hallucination-metric axis", which should be three at C5.
- **#6** appendix_d:130–137, FEVER Holm table, all 8 rows. B7 FLARE (rank 1, "yes") and B6 (rank 2, "yes") are both not significant now (+0.040; −0.067); no FEVER EM contrast is significant at C5. Caption at appendix_d:140: "Two of eight baselines clear the corrected threshold: B7 FLARE ... and B6 vanilla-fine-tune (CAEM loss by approximately eight percentage points ...)".
- appendix_d:50 B1 FEVER row (χ² 0.232, p 0.630) and :55 B8 TQA row (χ² 0.010, p 0.920): the EM values match the current table, but χ² and p have no source (U5/U10). Verify when the p-values are regenerated.

## Checks
- `git diff --stat`: appendix_c 38 lines, appendix_d 8, appendix_e 12 (29+/29−).
- Column counts: every tabular row in appendix_c matches its spec (5, 8, 7, 7, 6). appendix_d and appendix_e tables are unchanged.

## Numbers audit, Part 1: appendices A, B, F, G, generated tables, README (2026-10-15)

# fixlog_rest.md: README, figures/auto, appendices A/B/F/G (2026-10-15)

Exact BEFORE (`-`) / AFTER (`+`) text for every edit is in the per-file diffs under "Diffs" below (`git diff -U0`,
line numbers in the hunk headers). The index maps each item to its file:line and evidence.
Every changed figures/auto file has the comment line "% Hand-corrected 2026-10-15 (see ref_check/fixes_applied.md)"
under its "% Auto-generated" header.

## Index of applied items

| # | File:line (after edit) | Change | Evidence / verification |
|---|---|---|---|
| 1 | README.md:172 | 13/40, 23/40 (C5); 17/40, 26/40 (C2) -> 9/40, 21/35 (C5); 14/40, 20/35 (C2). Defeats restated: 3 EM at both anchors; CHM 3 (C5), 4 (C2); all on TruthfulQA vs zero-shot/CoT/FLARE/semantic entropy. "RAG-free" dropped because FLARE is among the defeats. CHM denominator 40 -> 35 because the B6 CHM cells are n/a (same rationale as #7). | Recounted from figures/auto/tab_sig_test.tex and tab_sig_test_c2.tex (awk over the Sig column): C5 EM 9 wins/3 defeats, CHM 21/3 (cot, semantic_entropy, zero_shot); C2 EM 14/3, CHM 20/4 (+flare) |
| 8 | README.md:53 | "beats every baseline ... at both reference cycles" -> C2 beats 8/8; C5 beats 7/8; FLARE leads C5 EM by 0.008 | tab_baseline_pooled.tex B7 row (ΔEM vs C5 = −0.008) and footer "C5 beats 7/8" |
| 8 | README.md:121 | −10.8% -> −11.0% | tab_baseline_pooled B1 ΔCHM vs C2 (−11.0%) |
| 8 | README.md:161–168 | B1–B8 rows replaced with tab_baseline_pooled values (B1 +0.011/+2.1%; B2 +3.5%; B3 +13.0%; B4 0.464 ...; B5 +27.6%; B6 0.490/0.127 ...; B7 0.525/0.173 ...; B8 +7.8%, +0.013/+2.5%). README system labels kept. | tab_baseline_pooled.tex rows B1–B8, cell for cell |
| 8 | README.md:170 | "C5 beats 8/8" -> "7/8 (FLARE leads CAEM C5 on pooled EM by 0.008)" | same |
| 10 | README.md:208–212 | per-tier C5 EMs replaced with tab_per_tier_em C5 rows (as fractions); TruthfulQA pooled 0.493 -> 0.300. Shares unchanged; they match the n's exactly (130/170, 1/111/188, 247/53, 1/193/106, 72/228 of 300). | tab_per_tier_em.tex C5 rows |
| P3 | README.md:410 | "(Vast.ai billing records)" appended to the Realised-total cost cell | Part 3 instruction |
| 9 | figures/auto/tab_tier_trajectory.tex:22–28 + caption | panel (b) EMs rebuilt from tab_per_tier_em C5 (TQA 0.321/0.255, pooled 0.300; FEVER 0.500/0.459, etc.). Shares already matched the per-tier n exactly, so the panel was kept. Caption: "Panel b copies the C5 rows of tab:per-tier-em; TruthfulQA EM uses the Haiku-judged correctness rule." | Cross-check: panel (a) C5 (T2 0.537, T3 0.497, pooled 0.517) agrees with the new panel (b) (T2 404/753, T3 370/745, total 776/1500). The old panel (b) did not. |
| 14 | figures/auto/tab_em_conditional_breakdown.tex caption | added: totals are an offline reconstruction from memory-store set differences; the run log records 1,061 prunes (56/271/352/382) and 650 promotions (131/220/299). Table numbers unchanged. | log_resolution.md D2, D3 (experiment.log:5710, 19133, 33683, 60457; 16404, 30357, 55957) |
| 16 | figures/auto/cross_benchmark_summary.tex caption | "calibration-fold sample count" -> "sample count on the cycle-zero validation fold ($n = 500$ per benchmark; not the calibration fold)" | numbers_mismatch_origins A19; log_resolution D1 (run_phase1a_ruc.log:71 eval=500) |
| 18 | figures/auto/tab_storage_pool_quality.tex caption | "removes between 35% and 49% of those wrong entries" -> "56% and 73%" | Recomputed from panel (b): wrong admitted vs n_pass×(1−π_SIL): C1 60.7%, C2 55.7%, C3 60.7%, C4 57.8%, C5 73.1% |
| 36 | figures/auto/tab_per_tier_em_capability.tex:18 | ">50% of CSQA queries to T3" -> "20%" | own row: commonsense_qa share 0%/80%/20%, T3 n=61/300 |
| 37 | figures/auto/tab_decision_matrix_trajectory.tex:3 (comment) | "at C3 (joint best in-distribution) and C5" -> "at C2 (within-trajectory best) and C5" | table body holds C2 and C5 panels |
| 42 | figures/auto/tab_decision_matrix_trajectory.tex caption | "holds at every cell" -> holds in 5 of 10 rows (C2 TriviaQA, C2 TruthfulQA, C5 TriviaQA, C5 TruthfulQA, C5 StrategyQA). Store > defer holds in every row; the exceptions are C2 FEVER, C2 CSQA, C2 StrategyQA, C5 FEVER and C5 CSQA, where discard precision > defer precision (C2 CSQA and C2 StrategyQA discard = 1 entry). | Recomputed from table counts, e.g. C2 FEVER defer 0.451 < discard 0.563; C5 FEVER 0.523 < 0.833; C5 CSQA 0.500 < 0.647 |
| 38 | figures/auto/tab_ruc_lobo.tex | deleted (rm) | grep before deletion: no \input of tab_ruc_lobo and no \ref/\Cref of tab:ruc-lobo anywhere in thesis_report/*.tex. Only scripts/retrieval_utility_classifier.py (the generator) names it; that script would recreate the file if run. |
| 45 | figures/auto/sweep_top.tex:13 | "tenfold tighter" -> "hundredfold tighter" | 1.0 / 0.010 = 100 |
| 52 | figures/auto/tab_theorem_receipts_summary.tex T5 row | "band of width ≤ 0.031" -> "Pooled cal-fold π_t confined to a band of width 0.014 (0.746 − 0.732)" | tab_pi_store_trajectory cal-fold column 0.746/0.740/0.736/0.746/0.732 |
| 52 | figures/auto/tab_theorem_receipts_summary.tex T3 row | "Sign alignment holds on three of four cal-fold transitions; honest reverse-sign disclosure at C3→C5" -> three observable transitions (C1→C2, C2→C3, C3→C5; C4 skipped): C1→C2 sign-aligns, C2→C3 sub-noise in Cohen's d, C3→C5 small reverse sign | Taken from chapter_4.tex:582 and chapter_5.tex:227 (the prose counts three transitions, one aligned). PARTLY VERIFIED: tab_pi_store_trajectory gives cal-fold n 631→1001→899→868, but it does not carry per-cycle Cohen's d or the storage rate, so the sign alignment itself was not recomputed. The row now restates the prose without a count of "aligned" transitions. |
| 54 | figures/auto/tab_pooled_trajectory.tex:28 | −10.8% -> −11.0% | spec: computed from unrounded values; matches tab_baseline_pooled B1 ΔCHM vs C2 (−11.0%) |
| 28 | appendix/appendix_a.tex:150, :325 | :150 rationale now says the flag applies only to the full fine-tuning fallback and LoRA uses 32-bit AdamW (value `true` kept, since that is the config). :325 "eight-bit AdamW optimiser" -> "32-bit AdamW optimiser (the eight-bit AdamW flag applies only to the full fine-tuning fallback path)" | caem/training/self_improvement.py:1036–1045 (`if getattr(self, "_lora_active", False): want_8bit = False`) |
| 32 | appendix/appendix_a.tex:275, :318 | ≈3,000 seeded entries -> 431 (the ≈3,000 described as the seeding plan's candidate target) | tab_memory_store_trajectory.tex C0 = 431; experiment.log:5710 "431 total"; archive plan in numbers_mismatch_origins O28 |
| P3 | appendix/appendix_a.tex:323 | added "; Vast.ai billing records" to the parenthetical after the $672 sentence | Part 3 instruction |
| 34 | appendix/appendix_b.tex:16 | "The headline ten-variant table appears in tab:sweep-top" -> "The locked configuration selected from the sweep ... appears in tab:sweep-top"; "the auto-generated comparison table is reproduced in the body chapter rather than duplicated here" -> "no per-variant comparison table is reproduced in the thesis" | sweep_top.tex is now the locked-parameter table (G 449dacb) |
| 59 | appendix/appendix_f.tex:113 | "$m - k$" -> "$m - k + 1$" | standard Holm; spec notes the table uses ×8 at rank 1 with m = 8 |
| 26 | appendix/appendix_g.tex:60 | ≈3,000 batch now marked as the production deployment batch (≈750 per decision class), compared with 1,500 per cycle in the research run | docs/PRODUCTION_RUNBOOK.md:233 "~3000 labeled samples (~750 per decision class)" |

## Skipped or not applicable
- #18 body-text part (ch5:216): not my file. Caption part applied.
- #42 prose part (ch5:147): not my file. Caption part applied.
- #45 ch5:301 part, and #52's companion #53 (ch4:603): not my files.
- Part 3 in README: one attribution added, on the Realised-total row. The other rows of the hardware table give no numeric GPU-hours or USD figures, and the "2.2–2.6×" penalty at README:414 is not a billing figure, so it was not tagged.

## Noted, not changed (outside the spec)
- README.md:214 says the C2 T2 EM beats the C0 T3 EM "on every benchmark by margins of 5–20 pp". tab_per_tier_em gives TriviaQA 51.0 vs 46.4 = 4.6 pp, so the lower bound is slightly off.
- cross_benchmark_summary caption says the transfer panel "does not enter the storage gate at cycle zero", yet the table lists store counts for TruthfulQA and StrategyQA (67, 21).
- scripts/retrieval_utility_classifier.py, aggregate_section_5_4/5_6/5_7.py and the per-tier/decision-matrix generators would overwrite these hand corrections if re-run (Part 4).

## Diffs (exact BEFORE/AFTER)
```diff
diff --git a/README.md b/README.md
index 178ffca..0472f30 100755
--- a/README.md
+++ b/README.md
@@ -53 +53 @@ Against the matched Qwen-2.5-3B-Instruct zero-shot baseline `B1` over the same b
-- **External eight-baseline panel:** CAEM beats every baseline on both the exact-match and the composite-hallucination axes in the pooled comparison at both reference cycles (`tab:baseline-pooled`).
+- **External eight-baseline panel:** in the pooled comparison, CAEM C2 beats all eight baselines jointly on the exact-match and composite-hallucination axes, and CAEM C5 beats seven of eight; FLARE leads CAEM C5 on pooled exact match by 0.008 (`tab:baseline-pooled`).
@@ -121 +121 @@ The realised programme closed at the **six-cycle horizon** (cycles C0 through C5
-Headline contrast vs B1 zero-shot at the trajectory's within-best cycle (C2): **ΔEM = +0.037 (+7.4% relative)**, **ΔCHM = −0.013 (−10.8% relative)**.
+Headline contrast vs B1 zero-shot at the trajectory's within-best cycle (C2): **ΔEM = +0.037 (+7.4% relative)**, **ΔCHM = −0.013 (−11.0% relative)**.
@@ -161,8 +161,8 @@ All baselines evaluated under the matched-protocol contract on the five-benchmar
-| B1 | zero-shot | 0.507 | 0.120 | +0.037 (+7.4%) | −0.013 (−11.0%) | +0.010 (+2.0%) | −0.011 (−9.3%) |
-| B2 | chain-of-thought | 0.500 | 0.123 | +0.044 (+8.8%) | −0.016 (−12.7%) | +0.017 (+3.4%) | −0.014 (−11.0%) |
-| B3 | DPR-RAG | 0.458 | 0.148 | +0.086 (+18.8%) | −0.041 (−27.7%) | +0.059 (+12.9%) | −0.039 (−26.4%) |
-| B4 | CoT + DPR-RAG | 0.461 | 0.142 | +0.083 (+17.9%) | −0.035 (−24.6%) | +0.056 (+12.1%) | −0.033 (−23.2%) |
-| B5 | five-shot CoT | 0.405 | 0.136 | +0.139 (+34.2%) | −0.029 (−21.4%) | +0.112 (+27.5%) | −0.027 (−19.9%) |
-| B6 | vanilla fine-tune (C5) | 0.325 | 0.144 | +0.219 (+67.2%) | −0.037 (−25.5%) | +0.192 (+58.9%) | −0.035 (−24.1%) |
-| B7 | FLARE | 0.292 | 0.153 | +0.252 (+86.3%) | −0.046 (−29.9%) | +0.225 (+77.1%) | −0.044 (−28.6%) |
-| B8 | semantic entropy | 0.505 | 0.120 | +0.039 (+7.7%) | −0.013 (−10.5%) | +0.012 (+2.4%) | −0.011 (−8.9%) |
+| B1 | zero-shot | 0.507 | 0.120 | +0.037 (+7.4%) | −0.013 (−11.0%) | +0.011 (+2.1%) | −0.011 (−9.3%) |
+| B2 | chain-of-thought | 0.500 | 0.123 | +0.044 (+8.8%) | −0.016 (−12.7%) | +0.017 (+3.5%) | −0.014 (−11.0%) |
+| B3 | DPR-RAG | 0.458 | 0.148 | +0.086 (+18.8%) | −0.041 (−27.7%) | +0.059 (+13.0%) | −0.039 (−26.4%) |
+| B4 | CoT + DPR-RAG | 0.464 | 0.142 | +0.080 (+17.2%) | −0.035 (−24.6%) | +0.053 (+11.5%) | −0.033 (−23.2%) |
+| B5 | five-shot CoT | 0.405 | 0.136 | +0.139 (+34.2%) | −0.029 (−21.4%) | +0.112 (+27.6%) | −0.027 (−19.9%) |
+| B6 | vanilla fine-tune (C5) | 0.490 | 0.127 | +0.054 (+11.0%) | −0.020 (−15.9%) | +0.027 (+5.6%) | −0.018 (−14.3%) |
+| B7 | FLARE | 0.525 | 0.173 | +0.019 (+3.5%) | −0.066 (−38.0%) | −0.008 (−1.5%) | −0.064 (−36.8%) |
+| B8 | semantic entropy | 0.505 | 0.120 | +0.039 (+7.8%) | −0.013 (−10.5%) | +0.013 (+2.5%) | −0.011 (−8.9%) |
@@ -170 +170 @@ All baselines evaluated under the matched-protocol contract on the five-benchmar
-**Wins-against count** (CAEM beats baseline on ΔEM and ΔCHM jointly): **CAEM C2 beats 8/8 baselines; CAEM C5 beats 8/8 baselines**.
+**Wins-against count** (CAEM beats baseline on ΔEM and ΔCHM jointly): **CAEM C2 beats 8/8 baselines; CAEM C5 beats 7/8 baselines** (FLARE leads CAEM C5 on pooled EM by 0.008).
@@ -172 +172 @@ All baselines evaluated under the matched-protocol contract on the five-benchmar
-The paired McNemar exact-match receipts and paired Wilcoxon composite-hallucination receipts of `tab:sig-test` clear the joint significance-and-effect-size licensing rule of Chapter 5 §5.4 on **13 of 40 per-benchmark contrasts on the exact-match axis** and **23 of 40 contrasts on the composite-hallucination axis** at the cycle-five anchor, and on **17 of 40 and 26 of 40 contrasts respectively** at the best-cycle anchor of `tab:sig-test-c2`, with three significant exact-match defeats and four significant composite-hallucination defeats on the per-benchmark axis confined to TruthfulQA against the high-EM RAG-free baselines (the surface where the architecture's abstention behaviour interacts with the Haiku-judged correctness rule and the false-refusal subtype of the composite hallucination metric).
+The paired McNemar exact-match receipts and paired Wilcoxon composite-hallucination receipts of `tab:sig-test` clear the joint significance-and-effect-size licensing rule of Chapter 5 §5.4 on **9 of 40 per-benchmark contrasts on the exact-match axis** and **21 of 35 contrasts on the composite-hallucination axis** (the five vanilla-fine-tune CHM cells are n/a) at the cycle-five anchor, and on **14 of 40 and 20 of 35 contrasts respectively** at the best-cycle anchor of `tab:sig-test-c2`. The significant defeats number three on the exact-match axis at both anchors, and three (C5) and four (C2) on the composite-hallucination axis; all are confined to TruthfulQA against the zero-shot, chain-of-thought, FLARE and semantic-entropy baselines (the surface where the architecture's abstention behaviour interacts with the Haiku-judged correctness rule and the false-refusal subtype of the composite hallucination metric).
@@ -208,5 +208,5 @@ The cycle-zero baseline routes 51.3% of queries to the retrieval-augmented tier
-| FEVER | 0.0% / n/a | 43.3% / 0.554 | 56.7% / 0.418 | 0.477 |
-| TriviaQA | 0.3% / 1.000 | 37.0% / 0.514 | 62.7% / 0.420 | 0.453 |
-| CommonsenseQA | 0.0% / n/a | 82.3% / 0.733 | 17.7% / 0.642 | 0.717 |
-| TruthfulQA | 0.3% / 1.000 | 64.3% / 0.591 | 35.3% / 0.311 | 0.493 |
-| StrategyQA | 0.0% / n/a | 24.0% / 0.736 | 76.0% / 0.610 | 0.640 |
+| FEVER | 0.0% / n/a | 43.3% / 0.500 | 56.7% / 0.459 | 0.477 |
+| TriviaQA | 0.3% / 1.000 | 37.0% / 0.423 | 62.7% / 0.468 | 0.453 |
+| CommonsenseQA | 0.0% / n/a | 82.3% / 0.745 | 17.7% / 0.585 | 0.717 |
+| TruthfulQA | 0.3% / 1.000 | 64.3% / 0.321 | 35.3% / 0.255 | 0.300 |
+| StrategyQA | 0.0% / n/a | 24.0% / 0.639 | 76.0% / 0.640 | 0.640 |
@@ -410 +410 @@ Every reported number is regenerable from the realised seed plus the published a
-| **Realised total** | **~960 GPU-hours over ~40 calendar days (40 × 24)** | **~672 USD on a Vast.ai RTX 5090 at ~0.70 USD/on-demand GPU-hour** |
+| **Realised total** | **~960 GPU-hours over ~40 calendar days (40 × 24)** | **~672 USD on a Vast.ai RTX 5090 at ~0.70 USD/on-demand GPU-hour** (Vast.ai billing records) |
diff --git a/thesis_report/appendix/appendix_a.tex b/thesis_report/appendix/appendix_a.tex
index dea9ee5..a0f7262 100644
--- a/thesis_report/appendix/appendix_a.tex
+++ b/thesis_report/appendix/appendix_a.tex
@@ -150 +150 @@ Every constant that governs runtime behaviour lives in the \verb|CAEMConfig| dat
-\texttt{use\_8bit\_adamw}             & \texttt{true}  & design       & Fits the bounded GPU envelope at batch $4$ with gradient checkpointing. \\
+\texttt{use\_8bit\_adamw}             & \texttt{true}  & design       & Applies only to the full fine-tuning fallback; the LoRA path overrides it and uses 32-bit AdamW. \\
@@ -275 +275 @@ The end-to-end execution sequence partitions into a pre-run setup block, a six-s
-\item \textbf{Cold-start seeding} (\verb|step_6_reseed| and \verb|step_6_5_em_prune|). Seeds the episodic memory with approximately three thousand verified entries from the training benchmarks' dev splits under a cold-start storage threshold of $0.50$. Gate: deduplication ratio sane; the EM-prune step removes fewer than ten percent of seeded entries.
+\item \textbf{Cold-start seeding} (\verb|step_6_reseed| and \verb|step_6_5_em_prune|). Seeds the episodic memory with verified entries from the training benchmarks' dev splits under a cold-start storage threshold of $0.50$; the realised cold-start store holds $431$ entries (the seeding plan targeted approximately three thousand candidates before deduplication and hygiene). Gate: deduplication ratio sane; the EM-prune step removes fewer than ten percent of seeded entries.
@@ -318 +318 @@ The runner picks up at the highest-cycle artefact set on disk.
-The snapshot contains the cold-start memory store (approximately three thousand seeded episodes), the cycle-zero memory and deferred-buffer snapshots, the cycle-zero calibration fold (per-sample nine-signal vectors and gold labels), the cycle-zero composite calibration JSON (per-signal isotonic curves, per-benchmark shrinkage children, boost-layer weights), the calibrated configuration JSON (per-benchmark temperature scalars and safety floors), the cycle-zero validation-gate verdict, the cycle-zero evaluation fold (pristine model), the post-calibration rescored evaluation fold, the dataset-splits manifest, the MiniCheck-versus-RoBERTa head-to-head calibration record, the long-hypothesis judge Platt-ablation diagnostic, the cycle-zero retention slice baseline, the phase-four artefact bundle, and the run logs. Items that must be regenerated rather than pulled from the snapshot are the Qwen-2.5-3B-Instruct base weights (recovered from the Hugging Face model hub at deployment time), the per-cycle LoRA adapters at cycles one through five, the per-cycle SIL training artefacts, and the post-cycle-zero composite refits.
+The snapshot contains the cold-start memory store ($431$ seeded episodes), the cycle-zero memory and deferred-buffer snapshots, the cycle-zero calibration fold (per-sample nine-signal vectors and gold labels), the cycle-zero composite calibration JSON (per-signal isotonic curves, per-benchmark shrinkage children, boost-layer weights), the calibrated configuration JSON (per-benchmark temperature scalars and safety floors), the cycle-zero validation-gate verdict, the cycle-zero evaluation fold (pristine model), the post-calibration rescored evaluation fold, the dataset-splits manifest, the MiniCheck-versus-RoBERTa head-to-head calibration record, the long-hypothesis judge Platt-ablation diagnostic, the cycle-zero retention slice baseline, the phase-four artefact bundle, and the run logs. Items that must be regenerated rather than pulled from the snapshot are the Qwen-2.5-3B-Instruct base weights (recovered from the Hugging Face model hub at deployment time), the per-cycle LoRA adapters at cycles one through five, the per-cycle SIL training artefacts, and the post-cycle-zero composite refits.
@@ -323 +323 @@ The snapshot contains the cold-start memory store (approximately three thousand
-\paragraph{Production hardware:} The trajectory ran on a Vast.ai-rented RTX 5090 with thirty-two gigabytes of video memory (approximately thirty-one gigabytes usable after framework overhead), one hundred and fifty gigabytes of disk, and approximately ninety gigabytes of host random-access memory. The rental rate sat at approximately seventy cents per on-demand GPU-hour. The realised six-cycle trajectory plus cycle-zero calibration plus post-run analysis spans approximately forty calendar days of elapsed time as cycles, ablations, and rescore passes run sequentially on the single-card envelope, for approximately nine hundred and sixty GPU-hours at a total billed cost of approximately six hundred and seventy-two United States dollars (forty days at twenty-four hours per day at the on-demand rate). The verifier-side scoring path (nine-signal extraction, composite refit, retroactive re-verification, and the matched-protocol Haiku-judged TruthfulQA rescore across the baseline panel) is the dominant wall-clock consumer over the run, exceeding the per-cycle low-rank-adapter fine-tune itself, because every entry on every cycle and every baseline-arm answer passes through the full verifier ensemble while the adapter touches only the strict training-pool subset.
+\paragraph{Production hardware:} The trajectory ran on a Vast.ai-rented RTX 5090 with thirty-two gigabytes of video memory (approximately thirty-one gigabytes usable after framework overhead), one hundred and fifty gigabytes of disk, and approximately ninety gigabytes of host random-access memory. The rental rate sat at approximately seventy cents per on-demand GPU-hour. The realised six-cycle trajectory plus cycle-zero calibration plus post-run analysis spans approximately forty calendar days of elapsed time as cycles, ablations, and rescore passes run sequentially on the single-card envelope, for approximately nine hundred and sixty GPU-hours at a total billed cost of approximately six hundred and seventy-two United States dollars (forty days at twenty-four hours per day at the on-demand rate; Vast.ai billing records). The verifier-side scoring path (nine-signal extraction, composite refit, retroactive re-verification, and the matched-protocol Haiku-judged TruthfulQA rescore across the baseline panel) is the dominant wall-clock consumer over the run, exceeding the per-cycle low-rank-adapter fine-tune itself, because every entry on every cycle and every baseline-arm answer passes through the full verifier ensemble while the adapter touches only the strict training-pool subset.
@@ -325 +325 @@ The snapshot contains the cold-start memory store (approximately three thousand
-\paragraph{Backbone model:} The base generator is Qwen2.5-3B-Instruct (\verb|Qwen/Qwen2.5-3B-Instruct| on the Hugging Face model hub), a three-billion-parameter decoder-only transformer with instruction tuning. The model is held in bfloat16 precision throughout the trajectory. The LoRA adapter sits on top with rank thirty-two, alpha sixty-four, dropout zero point zero five, all seven linear projections targeted (query, key, value, output, gate, up, down), at learning rate two times ten to the minus four under an eight-bit AdamW optimiser.
+\paragraph{Backbone model:} The base generator is Qwen2.5-3B-Instruct (\verb|Qwen/Qwen2.5-3B-Instruct| on the Hugging Face model hub), a three-billion-parameter decoder-only transformer with instruction tuning. The model is held in bfloat16 precision throughout the trajectory. The LoRA adapter sits on top with rank thirty-two, alpha sixty-four, dropout zero point zero five, all seven linear projections targeted (query, key, value, output, gate, up, down), at learning rate two times ten to the minus four under a 32-bit AdamW optimiser (the eight-bit AdamW flag applies only to the full fine-tuning fallback path).
diff --git a/thesis_report/appendix/appendix_b.tex b/thesis_report/appendix/appendix_b.tex
index 4899bd1..1122cdf 100644
--- a/thesis_report/appendix/appendix_b.tex
+++ b/thesis_report/appendix/appendix_b.tex
@@ -16 +16 @@
-The headline ten-variant table appears in \Cref{tab:sweep-top} of \Cref{ch:evaluation}. The full sweep over the pre-specified $\tau_{\text{store}} \times C \times \alpha_{\text{shrink}}$ grid ($4 \times 5 \times 5 = 100$ variants) is recorded in the cycle-zero sweep archive with one record per variant; the auto-generated comparison table is reproduced in the body chapter rather than duplicated here. Per-benchmark sweep diagnostics for any variant in the grid are recoverable by reading the corresponding JSON entry.
+The locked configuration selected from the sweep, with the rationale for each parameter, appears in \Cref{tab:sweep-top} of \Cref{ch:evaluation}. The full sweep over the pre-specified $\tau_{\text{store}} \times C \times \alpha_{\text{shrink}}$ grid ($4 \times 5 \times 5 = 100$ variants) is recorded in the cycle-zero sweep archive with one record per variant; no per-variant comparison table is reproduced in the thesis. Per-benchmark sweep diagnostics for any variant in the grid are recoverable by reading the corresponding JSON entry.
diff --git a/thesis_report/appendix/appendix_f.tex b/thesis_report/appendix/appendix_f.tex
index da566f9..8d8f0a8 100644
--- a/thesis_report/appendix/appendix_f.tex
+++ b/thesis_report/appendix/appendix_f.tex
@@ -113 +113 @@ The glossary collects every concept word that recurs in body prose and whose mea
-\item[Holm-Bonferroni.] The step-down multiple-comparison correction applied within each benchmark family of size $m = 8$ on each metric axis. Sorts the raw $p$-values ascending and multiplies the $k$-th smallest by $m - k$ under a running-maximum monotonicity constraint, clipping at one. See \Cref{sec:sig-holm}.
+\item[Holm-Bonferroni.] The step-down multiple-comparison correction applied within each benchmark family of size $m = 8$ on each metric axis. Sorts the raw $p$-values ascending and multiplies the $k$-th smallest by $m - k + 1$ under a running-maximum monotonicity constraint, clipping at one. See \Cref{sec:sig-holm}.
diff --git a/thesis_report/appendix/appendix_g.tex b/thesis_report/appendix/appendix_g.tex
index 782b5da..fcd064f 100644
--- a/thesis_report/appendix/appendix_g.tex
+++ b/thesis_report/appendix/appendix_g.tex
@@ -60 +60 @@ Plain fine-tuning consumes exact-match-labelled samples as the gradient signal.
-The mechanism that produces the order-of-magnitude difference is the verifier ensemble. The verifier runs label-free on every query and admits to memory only the high-confidence subset of traffic; the cycle boundary's labelled batch is a stratified sample of approximately three thousand entries drawn from the memory-admission distribution rather than from the unlabelled-traffic distribution. The architecture spends labels validating the verifier rather than training the generator. Plain fine-tuning at the same three-thousand-sample budget would produce a much weaker fine-tune because three thousand labels carrying the gradient signal directly under-trains a three-billion-parameter model. The architecture's labelling budget is therefore not the same instrument as the plain-fine-tuning labelling budget; it is a verifier-precision-validation instrument that is informationally efficient by a factor of ten to a hundred.
+The mechanism that produces the order-of-magnitude difference is the verifier ensemble. The verifier runs label-free on every query and admits to memory only the high-confidence subset of traffic; in a production deployment, the cycle boundary's labelled batch is a stratified sample of approximately three thousand entries (about seven hundred and fifty per decision class) drawn from the memory-admission distribution rather than from the unlabelled-traffic distribution, against the fifteen hundred labelled samples per cycle used in the research run. The architecture spends labels validating the verifier rather than training the generator. Plain fine-tuning at the same three-thousand-sample budget would produce a much weaker fine-tune because three thousand labels carrying the gradient signal directly under-trains a three-billion-parameter model. The architecture's labelling budget is therefore not the same instrument as the plain-fine-tuning labelling budget; it is a verifier-precision-validation instrument that is informationally efficient by a factor of ten to a hundred.
diff --git a/thesis_report/figures/auto/cross_benchmark_summary.tex b/thesis_report/figures/auto/cross_benchmark_summary.tex
index 9bb2ced..43c5c2d 100644
--- a/thesis_report/figures/auto/cross_benchmark_summary.tex
+++ b/thesis_report/figures/auto/cross_benchmark_summary.tex
@@ -2,0 +3 @@
+% Hand-corrected 2026-10-15 (see ref_check/fixes_applied.md)
@@ -16 +17 @@ StrategyQA (T) & Trans & 500 & 0.606 & 21 & 4.2\% & 0.905 \\
-\caption[Cycle-zero per-benchmark headline diagnostics under the locked configuration.]{Cycle-zero per-benchmark headline diagnostics under the locked configuration. Each row reports the calibration-fold sample count, the exact-match rate, the storage-class count and rate at the locked cut-point $\tau_{\text{store}} = 0.60$, and the realised precision of the stored entries. The training panel (FEVER, TriviaQA, CommonsenseQA) supplies the calibration fold and the self-improvement training pool; the transfer panel (TruthfulQA, StrategyQA) is held out and does not enter the storage gate at cycle zero. Three benchmarks registered earlier in the project (HotpotQA, Natural Questions, and ARC-Challenge) were retired before the main run on the registered verifier-balanced-accuracy precondition evidence and are absent from the panel.}
+\caption[Cycle-zero per-benchmark headline diagnostics under the locked configuration.]{Cycle-zero per-benchmark headline diagnostics under the locked configuration. Each row reports the sample count on the cycle-zero validation fold ($n = 500$ per benchmark; not the calibration fold), the exact-match rate, the storage-class count and rate at the locked cut-point $\tau_{\text{store}} = 0.60$, and the realised precision of the stored entries. The training panel (FEVER, TriviaQA, CommonsenseQA) supplies the calibration fold and the self-improvement training pool; the transfer panel (TruthfulQA, StrategyQA) is held out and does not enter the storage gate at cycle zero. Three benchmarks registered earlier in the project (HotpotQA, Natural Questions, and ARC-Challenge) were retired before the main run on the registered verifier-balanced-accuracy precondition evidence and are absent from the panel.}
diff --git a/thesis_report/figures/auto/sweep_top.tex b/thesis_report/figures/auto/sweep_top.tex
index 6f058ef..e6cadda 100644
--- a/thesis_report/figures/auto/sweep_top.tex
+++ b/thesis_report/figures/auto/sweep_top.tex
@@ -2,0 +3 @@
+% Hand-corrected 2026-10-15 (see ref_check/fixes_applied.md)
@@ -12 +13 @@ Abstention floor $\phi_{\text{pg}}$ & $0.20$ & Early-exit rule \Cref{eq:early-ex
-Boost regularisation $C$ & $0.010$ & Strongest-regularised value in grid $\{0.010, 0.025, 0.050, 0.100, 1.000\}$; tenfold tighter than the sklearn default of $C = 1.0$ \\
+Boost regularisation $C$ & $0.010$ & Strongest-regularised value in grid $\{0.010, 0.025, 0.050, 0.100, 1.000\}$; hundredfold tighter than the sklearn default of $C = 1.0$ \\
diff --git a/thesis_report/figures/auto/tab_decision_matrix_trajectory.tex b/thesis_report/figures/auto/tab_decision_matrix_trajectory.tex
index 315ec56..1cdf190 100644
--- a/thesis_report/figures/auto/tab_decision_matrix_trajectory.tex
+++ b/thesis_report/figures/auto/tab_decision_matrix_trajectory.tex
@@ -3 +3,2 @@
-% at C3 (joint best in-distribution) and C5 (trajectory-end).
+% at C2 (within-trajectory best) and C5 (trajectory-end).
+% Hand-corrected 2026-10-15 (see ref_check/fixes_applied.md)
@@ -28 +29 @@ C5 & StrategyQA & $45$ & $10$ & $0.818$ & $100$ & $53$ & $0$ & $1$ \\
-\caption[Per-benchmark four-outcome decomposition at the best (C2) and final (C5) cycles.]{Per-benchmark four-outcome decomposition at the within-trajectory best cycle (C2) and the trajectory-end cycle (C5). \emph{Store} $\checkmark$ and \emph{Store} $\times$ are the correct and incorrect counts in the storage class; \emph{Store $\pi$} is the realised per-bench storage precision $\pi_{\text{store}}$ at the locked cut-point $\tau_{\text{store}} = 0.60$. The training-panel benchmarks (FEVER, TriviaQA, CommonsenseQA) admit large storage classes at $\pi_{\text{store}}$ in the band $[0.66, 0.78]$ across the two anchor cycles; the transfer-panel TruthfulQA storage class is small (storage gate does not fire on transfer-panel queries by design, so the few storage cells reflect the calibration-fold rescore rather than live storage) and StrategyQA records the highest per-bench precision at $0.818$ on the trajectory-end cycle. The within-bench ordering $\pi_{\text{store}} > \mathrm{prec}(\text{defer}) > \mathrm{prec}(\text{discard})$ holds at every cell, the structural receipt that the fixed-threshold gate sorts the candidate pool by correctness on each benchmark family.}
+\caption[Per-benchmark four-outcome decomposition at the best (C2) and final (C5) cycles.]{Per-benchmark four-outcome decomposition at the within-trajectory best cycle (C2) and the trajectory-end cycle (C5). \emph{Store} $\checkmark$ and \emph{Store} $\times$ are the correct and incorrect counts in the storage class; \emph{Store $\pi$} is the realised per-bench storage precision $\pi_{\text{store}}$ at the locked cut-point $\tau_{\text{store}} = 0.60$. The training-panel benchmarks (FEVER, TriviaQA, CommonsenseQA) admit large storage classes at $\pi_{\text{store}}$ in the band $[0.66, 0.78]$ across the two anchor cycles; the transfer-panel TruthfulQA storage class is small (storage gate does not fire on transfer-panel queries by design, so the few storage cells reflect the calibration-fold rescore rather than live storage) and StrategyQA records the highest per-bench precision at $0.818$ on the trajectory-end cycle. The within-bench ordering $\pi_{\text{store}} > \mathrm{prec}(\text{defer}) > \mathrm{prec}(\text{discard})$ holds in five of the ten rows (C2 TriviaQA, C2 TruthfulQA, C5 TriviaQA, C5 TruthfulQA, C5 StrategyQA). The first inequality, $\pi_{\text{store}} > \mathrm{prec}(\text{defer})$, holds in every row; the exceptions are C2 FEVER, C2 CommonsenseQA, C2 StrategyQA, C5 FEVER and C5 CommonsenseQA, where the discard class is more precise than the defer class (in C2 CommonsenseQA and C2 StrategyQA the discard class holds a single entry).}
diff --git a/thesis_report/figures/auto/tab_em_conditional_breakdown.tex b/thesis_report/figures/auto/tab_em_conditional_breakdown.tex
index 73b75c2..cc20bab 100644
--- a/thesis_report/figures/auto/tab_em_conditional_breakdown.tex
+++ b/thesis_report/figures/auto/tab_em_conditional_breakdown.tex
@@ -5,0 +6 @@
+% Hand-corrected 2026-10-15 (see ref_check/fixes_applied.md)
@@ -26 +27 @@ C5 & $422$ & $184$ & $238$ & $43.6\%$ & $293$ & $161$ & $132$ & $54.9\%$ \\
-\caption[EM-conditional decomposition of retroverify prunes and deferred-buffer promotions]{Em-conditional decomposition of retroverify-pruned entries and deferred-buffer-promoted entries across the realised trajectory. \emph{Wrong-caught rate} is the share of pruned entries whose ground-truth em label was zero, the principled measure of whether retroverify selects against hallucinations rather than rolling random; the pooled rate of $38.4\%$ is approximately $1.75\times$ the pool-base wrong rate of $\sim 22\%$, the empirical receipt that retroverify is enriched against wrong entries rather than uniform. The cycle-one anomaly ($0\%$ wrong-caught on a small $56$-entry prune cohort) reflects signal-noise on cold-start TriviaQA and CSQA entries that were actually correct; this is the noisy regime where the retroverify pass has the least lift over the gate's initial admission. \emph{Correct-recovered rate} is the share of promoted entries whose ground-truth em label was one; the pooled $55.6\%$ sits below the $\sim 78\%$ pool-base correct rate, confirming that the deferred-buffer cohort is intrinsically harder than the entries that cleared the storage cut-point on the first attempt, but is well above chance and recovers $345$ correct entries that the initial gate had rejected as too uncertain. The two mechanisms are complementary: retroverify shaves a modest hallucination-enriched tail off the storage pool, while the deferred buffer reclaims correct entries the initial gate over-rejected. Em labels are not persisted per entry by the runtime; the reconstruction used offline lookup against the benchmark gold (one-hundred-percent match coverage on the realised $12\,175$ stored entries) and an entry-identifier set-difference between consecutive memory-store metadata snapshots for prune identification (a small consolidation-removed cohort of $\le 5\%$ runs in the same step and is indistinguishable at the snapshot level).}
+\caption[EM-conditional decomposition of retroverify prunes and deferred-buffer promotions]{Em-conditional decomposition of retroverify-pruned entries and deferred-buffer-promoted entries across the realised trajectory. \emph{Wrong-caught rate} is the share of pruned entries whose ground-truth em label was zero, the principled measure of whether retroverify selects against hallucinations rather than rolling random; the pooled rate of $38.4\%$ is approximately $1.75\times$ the pool-base wrong rate of $\sim 22\%$, the empirical receipt that retroverify is enriched against wrong entries rather than uniform. The cycle-one anomaly ($0\%$ wrong-caught on a small $56$-entry prune cohort) reflects signal-noise on cold-start TriviaQA and CSQA entries that were actually correct; this is the noisy regime where the retroverify pass has the least lift over the gate's initial admission. \emph{Correct-recovered rate} is the share of promoted entries whose ground-truth em label was one; the pooled $55.6\%$ sits below the $\sim 78\%$ pool-base correct rate, confirming that the deferred-buffer cohort is intrinsically harder than the entries that cleared the storage cut-point on the first attempt, but is well above chance and recovers $345$ correct entries that the initial gate had rejected as too uncertain. The two mechanisms are complementary: retroverify shaves a modest hallucination-enriched tail off the storage pool, while the deferred buffer reclaims correct entries the initial gate over-rejected. Em labels are not persisted per entry by the runtime; the reconstruction used offline lookup against the benchmark gold (one-hundred-percent match coverage on the realised $12\,175$ stored entries) and an entry-identifier set-difference between consecutive memory-store metadata snapshots for prune identification (a small consolidation-removed cohort of $\le 5\%$ runs in the same step and is indistinguishable at the snapshot level). The Pruned and Promoted totals are therefore an offline reconstruction from memory-store set differences, not the run's own counts: the run log records $1\,061$ retroactive prunes ($56/271/352/382$ at C1/C2/C3/C5) and $650$ deferred-buffer promotions ($131/220/299$ at C2/C3/C5).}
diff --git a/thesis_report/figures/auto/tab_per_tier_em_capability.tex b/thesis_report/figures/auto/tab_per_tier_em_capability.tex
index 5aa1749..a7b4c44 100644
--- a/thesis_report/figures/auto/tab_per_tier_em_capability.tex
+++ b/thesis_report/figures/auto/tab_per_tier_em_capability.tex
@@ -3,0 +4 @@
+% Hand-corrected 2026-10-15 (see ref_check/fixes_applied.md)
@@ -17 +18 @@ EM values in percentage points; (n) is per-tier sample count.
-CommonsenseQA row shows the selection effect: T2 EM $>$ T3 EM but the router still sends $>$50\% of CSQA queries to T3 (see \Cref{sec:disc-threats}).
+CommonsenseQA row shows the selection effect: T2 EM $>$ T3 EM but the router still sends 20\% of CSQA queries to T3 (see \Cref{sec:disc-threats}).
diff --git a/thesis_report/figures/auto/tab_pooled_trajectory.tex b/thesis_report/figures/auto/tab_pooled_trajectory.tex
index 716e7f3..59aeff6 100644
--- a/thesis_report/figures/auto/tab_pooled_trajectory.tex
+++ b/thesis_report/figures/auto/tab_pooled_trajectory.tex
@@ -5,0 +6 @@
+% Hand-corrected 2026-10-15 (see ref_check/fixes_applied.md)
@@ -27 +28 @@ CAEM & C5 & 300 & 0.517 & 0.109 & 0.999 & 1.013 & \underline{0.958} & \textsc{co
-\multicolumn{9}{@{}l}{\textbf{Headline contrast (CAEM C2 vs.\ B1 zero-shot)}: $\Delta$EM $=+0.037$ ($+7.4\%$ relative); $\Delta$CHM $=-0.013$ ($-10.8\%$ relative).} \\
+\multicolumn{9}{@{}l}{\textbf{Headline contrast (CAEM C2 vs.\ B1 zero-shot)}: $\Delta$EM $=+0.037$ ($+7.4\%$ relative); $\Delta$CHM $=-0.013$ ($-11.0\%$ relative).} \\
diff --git a/thesis_report/figures/auto/tab_storage_pool_quality.tex b/thesis_report/figures/auto/tab_storage_pool_quality.tex
index 3e56a45..5824002 100644
--- a/thesis_report/figures/auto/tab_storage_pool_quality.tex
+++ b/thesis_report/figures/auto/tab_storage_pool_quality.tex
@@ -5,0 +6 @@
+% Hand-corrected 2026-10-15 (see ref_check/fixes_applied.md)
@@ -32 +33 @@ C5 & $2\,380$ & $1\,724$ & $656$ & $0.724$ & $895$ & $37.6\%$ & $0.803$ & $5\,00
-\caption[Storage-pool quality and SIL training-pool filter chain per cycle]{Storage-pool quality and SIL training-pool filter chain per cycle, reported on both the held-out evaluation fold (panel a) and the SIL stream chunks (panel b). $n_{\text{stored}}$ is the count of samples that cleared the storage cut-point $\tau_{\text{store}} = 0.60$; the em-correct and em-wrong columns split these by ground-truth correctness; $\pi_{\text{store}}$ is the realised storage precision. The training-pool admission threshold $\tau_{\text{train}} = 0.70$ selects a higher-confidence subset of the stored pool for SIL fine-tuning; the $n_{\text{pass}\,\tau_{\text{train}}}$ and pass-rate columns report the size and share of this subset. The $\pi_{\text{SIL}}$ column is the empirical precision of the SIL pool itself, the em-correct fraction within the $\tau_{\text{train}}$-passing subset. The stream-chunk panel is the deployment-relevant pool because storage decisions on stream-chunk samples populate the memory store (the eval fold is content-hash disjoint by construction and does not enter the pool); on the stream chunks $\pi_{\text{SIL}}$ stays in the band $[0.798, 0.818]$ across the five SIL cycles, a $+6$ to $+8$ percentage-point lift above the stream-chunk storage precision band of $[0.724, 0.753]$, the empirical receipt that the SIL pool has higher precision than the storage pool and that the $\tau_{\text{train}} = 0.70$ filter selects a strictly cleaner training substrate than the storage gate alone. The eval-fold lift is smaller ($+2$ to $+6$ percentage points) because the eval fold is held-disjoint from the training distribution and is a content-shifted reading rather than a deployment-realised one. The complementary read on memory poisoning is the em-wrong column: across the stream-chunk trajectory the storage gate admits between $606$ and $688$ wrong entries per cycle (the high-$u_{\text{stored}}$-but-wrong cohort) against $1\,802$ to $1\,873$ correct entries, and the $\tau_{\text{train}}$ filter removes between $35\%$ and $49\%$ of those wrong entries on the way into the SIL pool, the architectural mechanism by which the two-stage gate decouples the storage-purity contract from the training-purity contract. The cycle-five drop in pass rate on both folds reflects the post-C4 composite refit re-tightening the upper tail; the $\pi_{\text{SIL}}$ column shows the precision contract holds across the re-tightening. The $n_{\text{used SIL}}$ column reports the actual count of episodes that entered the per-cycle SIL training batch from the cumulative memory store, ranging from $107$ at the first SIL fine-tune to $5\,006$ at the last successful cycle; this exceeds the per-cycle pass count because the SIL pool draws from the full memory store accumulated across all prior cycles, not only from the current cycle's stream chunk.}
+\caption[Storage-pool quality and SIL training-pool filter chain per cycle]{Storage-pool quality and SIL training-pool filter chain per cycle, reported on both the held-out evaluation fold (panel a) and the SIL stream chunks (panel b). $n_{\text{stored}}$ is the count of samples that cleared the storage cut-point $\tau_{\text{store}} = 0.60$; the em-correct and em-wrong columns split these by ground-truth correctness; $\pi_{\text{store}}$ is the realised storage precision. The training-pool admission threshold $\tau_{\text{train}} = 0.70$ selects a higher-confidence subset of the stored pool for SIL fine-tuning; the $n_{\text{pass}\,\tau_{\text{train}}}$ and pass-rate columns report the size and share of this subset. The $\pi_{\text{SIL}}$ column is the empirical precision of the SIL pool itself, the em-correct fraction within the $\tau_{\text{train}}$-passing subset. The stream-chunk panel is the deployment-relevant pool because storage decisions on stream-chunk samples populate the memory store (the eval fold is content-hash disjoint by construction and does not enter the pool); on the stream chunks $\pi_{\text{SIL}}$ stays in the band $[0.798, 0.818]$ across the five SIL cycles, a $+6$ to $+8$ percentage-point lift above the stream-chunk storage precision band of $[0.724, 0.753]$, the empirical receipt that the SIL pool has higher precision than the storage pool and that the $\tau_{\text{train}} = 0.70$ filter selects a strictly cleaner training substrate than the storage gate alone. The eval-fold lift is smaller ($+2$ to $+6$ percentage points) because the eval fold is held-disjoint from the training distribution and is a content-shifted reading rather than a deployment-realised one. The complementary read on memory poisoning is the em-wrong column: across the stream-chunk trajectory the storage gate admits between $606$ and $688$ wrong entries per cycle (the high-$u_{\text{stored}}$-but-wrong cohort) against $1\,802$ to $1\,873$ correct entries, and the $\tau_{\text{train}}$ filter removes between $56\%$ and $73\%$ of those wrong entries on the way into the SIL pool, the architectural mechanism by which the two-stage gate decouples the storage-purity contract from the training-purity contract. The cycle-five drop in pass rate on both folds reflects the post-C4 composite refit re-tightening the upper tail; the $\pi_{\text{SIL}}$ column shows the precision contract holds across the re-tightening. The $n_{\text{used SIL}}$ column reports the actual count of episodes that entered the per-cycle SIL training batch from the cumulative memory store, ranging from $107$ at the first SIL fine-tune to $5\,006$ at the last successful cycle; this exceeds the per-cycle pass count because the SIL pool draws from the full memory store accumulated across all prior cycles, not only from the current cycle's stream chunk.}
diff --git a/thesis_report/figures/auto/tab_theorem_receipts_summary.tex b/thesis_report/figures/auto/tab_theorem_receipts_summary.tex
index 12b2b17..accdc46 100644
--- a/thesis_report/figures/auto/tab_theorem_receipts_summary.tex
+++ b/thesis_report/figures/auto/tab_theorem_receipts_summary.tex
@@ -2,0 +3 @@
+% Hand-corrected 2026-10-15 (see ref_check/fixes_applied.md)
@@ -13 +14 @@ T2 & \texttt{thm:\allowbreak bayes-\allowbreak purity} & Proposition & precondit
-T3 & \texttt{thm:\allowbreak monotone} & Heuristic & consistent within horizon, sharp test deferred & Sign alignment holds on three of four cal-fold transitions; honest reverse-sign disclosure at $C3 \to C5$. & \Cref{tab:pi-store-trajectory} \\
+T3 & \texttt{thm:\allowbreak monotone} & Heuristic & consistent within horizon, sharp test deferred & Three cal-fold transitions are observable ($C1 \to C2$, $C2 \to C3$, $C3 \to C5$; C4 skipped): $C1 \to C2$ sign-aligns at large amplitude, $C2 \to C3$ is sub-noise in Cohen's $d$, and $C3 \to C5$ shows a small reverse-sign storage-rate movement. & \Cref{tab:pi-store-trajectory} \\
@@ -15 +16 @@ T4 & \texttt{thm:\allowbreak monotone-\allowbreak purification} & Remark & ident
-T5 & \texttt{thm:\allowbreak convergence} & Heuristic & BOUNDED-BAND & Pooled $\pi_t$ confined to a band of width $\le 0.031$ across the six-cycle horizon; the rate-scaling form $r = \mathcal{O}\!\big(\sigma \cdot \Phi(d/\sqrt{2})\big)$ is a Gaussian-model approximation asserted in the argument sketch rather than measured at the present horizon, and the measured geometric-rate test is deferred to a $\ge 15$-committed-cycle horizon. & \Cref{tab:pi-store-trajectory} \\
+T5 & \texttt{thm:\allowbreak convergence} & Heuristic & BOUNDED-BAND & Pooled cal-fold $\pi_t$ confined to a band of width $0.014$ ($0.746 - 0.732$) across the six-cycle horizon; the rate-scaling form $r = \mathcal{O}\!\big(\sigma \cdot \Phi(d/\sqrt{2})\big)$ is a Gaussian-model approximation asserted in the argument sketch rather than measured at the present horizon, and the measured geometric-rate test is deferred to a $\ge 15$-committed-cycle horizon. & \Cref{tab:pi-store-trajectory} \\
diff --git a/thesis_report/figures/auto/tab_tier_trajectory.tex b/thesis_report/figures/auto/tab_tier_trajectory.tex
index 77d8dcc..27fbccf 100644
--- a/thesis_report/figures/auto/tab_tier_trajectory.tex
+++ b/thesis_report/figures/auto/tab_tier_trajectory.tex
@@ -3,0 +4 @@
+% Hand-corrected 2026-10-15 (see ref_check/fixes_applied.md)
@@ -21,5 +22,5 @@ C5 & 0.1\% & 1.000 & 50.2\% & 0.537 & 49.7\% & 0.497 & 0.517 \\
-FEVER & 0.0\% & -- & 43.3\% & 0.554 & 56.7\% & 0.418 & 0.477 \\
-TriviaQA & 0.3\% & 1.000 & 37.0\% & 0.514 & 62.7\% & 0.420 & 0.453 \\
-CommonsenseQA & 0.0\% & -- & 82.3\% & 0.733 & 17.7\% & 0.642 & 0.717 \\
-TruthfulQA & 0.3\% & 1.000 & 64.3\% & 0.591 & 35.3\% & 0.311 & 0.493 \\
-StrategyQA & 0.0\% & -- & 24.0\% & 0.736 & 76.0\% & 0.610 & 0.640 \\
+FEVER & 0.0\% & -- & 43.3\% & 0.500 & 56.7\% & 0.459 & 0.477 \\
+TriviaQA & 0.3\% & 1.000 & 37.0\% & 0.423 & 62.7\% & 0.468 & 0.453 \\
+CommonsenseQA & 0.0\% & -- & 82.3\% & 0.745 & 17.7\% & 0.585 & 0.717 \\
+TruthfulQA & 0.3\% & 1.000 & 64.3\% & 0.321 & 35.3\% & 0.255 & 0.300 \\
+StrategyQA & 0.0\% & -- & 24.0\% & 0.639 & 76.0\% & 0.640 & 0.640 \\
@@ -28 +29 @@ StrategyQA & 0.0\% & -- & 24.0\% & 0.736 & 76.0\% & 0.610 & 0.640 \\
-\caption[Per-cycle pooled tier share and per-tier exact match across the five-benchmark evaluation panel (panel a), with a per-benchmark snapshot at the cycle-five close (panel b).]{Per-cycle pooled tier share and per-tier exact match across the five-benchmark evaluation panel (panel a), with a per-benchmark snapshot at the cycle-five close (panel b). Tier 1 (memory-direct) fires on a small share of queries by construction of the content-hash-disjoint evaluation fold; every realised Tier 1 firing returns the correct stored answer (the empirical receipt for \Cref{cor:tier1-floor}). The Tier-2-versus-Tier-3 contrast at cold start and the per-benchmark tier-mix heterogeneity surfaced by panel b are discussed in the surrounding prose.}
+\caption[Per-cycle pooled tier share and per-tier exact match across the five-benchmark evaluation panel (panel a), with a per-benchmark snapshot at the cycle-five close (panel b).]{Per-cycle pooled tier share and per-tier exact match across the five-benchmark evaluation panel (panel a), with a per-benchmark snapshot at the cycle-five close (panel b). Tier 1 (memory-direct) fires on a small share of queries by construction of the content-hash-disjoint evaluation fold; every realised Tier 1 firing returns the correct stored answer (the empirical receipt for \Cref{cor:tier1-floor}). Panel b copies the C5 rows of \Cref{tab:per-tier-em}; TruthfulQA EM uses the Haiku-judged correctness rule. The Tier-2-versus-Tier-3 contrast at cold start and the per-benchmark tier-mix heterogeneity surfaced by panel b are discussed in the surrounding prose.}
```
(tab_ruc_lobo.tex: whole file deleted; recoverable with `git show HEAD:thesis_report/figures/auto/tab_ruc_lobo.tex`.)

## Numbers audit, Part 1: final corrections after review (2026-10-15)

| Item | Location | Before | After | Evidence |
|---|---|---|---|---|
| #33 (corrects the editor's change) | appendix_e:168 | editor had changed accumulation to "four to eight (preserving the effective batch size at sixteen)", a setting no source records | "four to sixteen (an effective batch size of thirty-two, twice the production value of sixteen)" | README:414 and the original appendix E text record accumulation 4 → 16; production is 4 × 4 = 16 (caem/config.py grad_accum_steps = 4) |
| #44 | ch5:216 | "in every cycle except cycle two the em-rate weakly increases with the bucket index" | "on the stream chunks the em-rate increases with the bucket index at every cycle, while on the evaluation fold it does so only at cycles one and four" | tab_u_stored_distribution panels (a) and (b) |
| #44 caption | tab_u_stored_distribution | "preserves the same ordering at every cycle except the cycle-two and cycle-three rows…" | "increases monotonically with the bucket index only at cycles one and four; at cycles zero, two, three and five at least one adjacent pair of buckets reverses, mostly where the upper buckets hold few entries." | same |
| #46 second half | ch5:308 | "sit an order of magnitude above the CommonsenseQA scalar at every cycle" | "sit $4.8$ to $10.7$ times above the CommonsenseQA scalar across the committed cycles" | tab_calibration_trajectory panel (b): FEVER/CSQA 10.50, 8.19, 6.45, 4.84; TriviaQA/CSQA 10.52, 8.28, 6.61, 10.74 |
| #11 (second site) | end of ch5:491 | "closes the TruthfulQA exact-match regression … lifting TruthfulQA exact match from 0.210 to 0.310 while…" | "narrows … from 0.210 to 0.310, still below the B1 zero-shot reference of 0.433, while…" | tab_per_bench_trajectory B1 TruthfulQA 0.433 |
| #43 caption | tab_per_tier_em | "the gap is preserved across the trajectory" | names the exceptions: FEVER at C2 (51.6 vs 54.5) and StrategyQA at C5 (63.9 vs 64.0) | tab_per_tier_em rows |

Still open (need the local data from step 2): #12 (ch5:181 safety override) and #2–#6 (appendix D contingency rows and Holm tables, plus their prose at D:39, D:59, D:72, D:87, D:118, D:140).
Also noted for step 3: appendix E:93/:166 still say a cycle takes "five to seven hours", which conflicts with the measured windows; README:214 says "5–20 pp" where TriviaQA is 4.6 pp; the cross_benchmark_summary caption says the transfer panel does not enter the storage gate, but the table lists store counts for it; scripts/hardware.py still targets an effective batch of 32.

## Numbers audit, Part 2 lookups applied (2026-10-15)

Source: `caem_part2_lookups.txt` (local agent, HEAD da05919).

| Item | Result | Thesis change |
|---|---|---|
| U2 RUC fit | Confirmed: caem/ruc/v2/cv_results.json and feature_spec.json give pooled OOF AUROC 0.8501, n_rows 3208, coef −1.765, the per-benchmark AUROCs 0.686/0.718/0.728/0.732/0.852/0.936; pik_probe_metrics.json gives n 11,980 and AUROC 0.7707 | none |
| U4 Qwen-judge ρ | Confirmed: outputs/calibration/qwen_judge_platt.ablation.json `_pearson_rho_logit_observed: 0.5843`, `_min_rho_threshold: 0.70` | none |
| U7 ECE-before | Confirmed: experiment.log:3601 (C0 0.276084) and :5592 (C1 0.050760) | none |
| U5 appendix D | B1 FEVER χ² 0.2323, p 0.6299 and B8 TruthfulQA χ² 0.0102, p 0.9195 confirmed (tab_sig_test_c5_post_haiku.csv). ROUGE 0.803 appears in outputs/analysis/phase1e_analysis_report.md | none |
| U3 conformal 0.9643 / 56 | No run artefact retained; recorded in presentation/caem_supervisor.tex (τ_store 0.6676, n 56, π 0.9643) | ch5:317: "…over $56$ stored entries;" → "…over $56$ stored entries at $\tau_{\text{store}} = 0.6676$ (recorded in the project's interim supervisor report; the run artefacts of this rejected formulation were not retained);" |
| U8 off-arm abort cycle | phase1d logs not present locally; notes say C4, the thesis said C5 | ch5:473: "was aborted at cycle five under the retention guard" → "was aborted under the retention guard" |
| U9 ≈53% C1 prune | No source anywhere | appendix_g:147: "approximately fifty-three percent of the cold-seed memory" → "a large share of the cold-seed memory" |
| U6 70/105, 306/478 | Not in Phase 1e data; matches the pre-Phase-1e calibration table in git 04fe2e7 (478 at 0.640). The text now labels it as the earlier sweep | none (already relabelled in Part 1) |
| U1 safety override | Using the pooled 0.38 floor: one eval sample, StrategyQA C5, u_pre 0.3764. **But** the router uses per-benchmark floors (caem/routing/router.py:159), and CommonsenseQA's floor was refit to 0.491 (C3) and 0.5687 (C5), so CommonsenseQA queries may also have fired | **held**; ch5:181 unchanged until the per-benchmark count is checked |
| #2–#6 appendix D rows | The two CSVs are dated 9 June, before the 13 June B6/B7 fix; their B6/B7 rows are pre-fix | **held**; needs a post-fix rerun of the significance script |

## Safety override recount and appendix C column note (2026-10-15)

Source: `caem_task_safety_sig.txt` (local agent).
- The router fires when u_pre < the benchmark's floor (caem/routing/router.py:181), using the per-benchmark floor where one exists (config.get_safety_u_pre_min_for).
- The floor schedule comes from experiment.log lines 3622, 5591, 18384, 32332 and 57893: CommonsenseQA 0.38 → 0.491 (C3, C4) → 0.5687 (C5); every other benchmark 0.38.
- Counting eval samples below their floor: CommonsenseQA 12 / 12 / 38 at C3 / C4 / C5 and StrategyQA 1 at C5, so 63 of 9,000.

| Location | Before | After |
|---|---|---|
| ch5:181 | "the override does not fire on any of the nine thousand evaluation queries … never activates …" | fires on 63 of 9,000 (0.700%): 62 CommonsenseQA after its floor refit (12, 12, 38 at C3–C5) and one StrategyQA query at C5 (u_pre 0.376). Under the original 0.38 floor it would have fired once. Wording changed to "activates rarely". |
| tab_safety_override, panel (a) | CommonsenseQA 0 fires (0.000%); subtotal 1 (0.011%) | CommonsenseQA 62 (3.444%); subtotal 63 (0.700%) |
| tab_safety_override, panel (b) and combined | panel (b) label had no floor note; combined 4 (0.012%) | panel (b) is marked "pooled floor 0.38 only", because the per-cycle stream-chunk files are on Drive only and were not recounted; combined ≥66 (≥0.203%) |
| tab_safety_override caption | "proxy u_pre < φ_safe = 0.38 … would have fired on four queries (0.012%)" | describes the per-benchmark floors and the 63/9,000 count; panel (b)'s CommonsenseQA count from C3 is a lower bound |
| appendix_c per-cycle equilibrium table | column "Pooled EM" | "Pooled EM*", with a note: five-benchmark pooled EM with string-match TruthfulQA (exactly 0.5573/0.5673/0.5473/0.548/0.556 from experiment_summary.csv), the series the runner's equilibrium fit stores in its composite-evaluation-score field. It is neither the CES of eq:ces (tab:ces-caem-trajectory: 0.678–0.710) nor the judged pooled EM of tab:pooled-trajectory. |

Open: recount stream-chunk CommonsenseQA firings at C3–C5 with the per-benchmark floor (needs the Drive-only `cycle_N/eval_stream_chunk/commonsense_qa_cycle*_streamchunk.json`).

## Appendix D: B6/B7 rows and Holm tables rebuilt from post-fix statistics (2026-10-15)

Source: sig_c5_postfix.csv, produced today by scripts/baseline_sig_tests_v2.py on local outputs; it agrees with tab_sig_test.tex. χ² values are recovered from the raw p via p = erfc(√(χ²/2)). The unchanged rows check out: B1 FEVER 0.2323 and B8 TruthfulQA 0.0102. Discordant counts are uniquely determined: B6 TriviaQA b=85, c=17; B7 StrategyQA b=85, c=73.
Also: the driver path at D:14 was "scripts/baseline/baseline_sig_tests.py", a path that does not exist. It now names scripts/baseline_sig_tests_v2.py, which reuses the McNemar and bootstrap routines of baseline_sig_tests.py.

# fixlog_appD.md — appendix/appendix_d.tex post-B6/B7-fix corrections (2026-10-15)

Source: sig_c5_postfix.csv (Drive 1iuyGEu0t83AYP0xgmJ7ZrzQEutX3p1b7), cross-checked against figures/auto/tab_sig_test.tex (agrees).
Every table/prose item in appendix D is captioned "cycle-five anchor" -> all values from the C5 CSV. The C2 CSV
(1h-i5HR6MYLjJpj7qpsEerTMovxAzDfP4) was read; nothing in appendix D draws on it except the final paragraph's tallies (14/3, 20/4), which are unchanged and match.
Baseline map: B1 zero_shot, B2 cot, B3 rag, B4 cot_rag, B5 fiveshot_cot, B6 vanilla_ft, B7 flare, B8 semantic_entropy.

chi^2 = inverse of p = erfc(sqrt(chi2/2)) (scipy missing; bisection with math.erfc).
Verification on unchanged rows: B1 FEVER p 0.6298545 -> chi2 0.2323 (table 0.232, OK); B8 TQA p 0.9195384 -> 0.0102 (table 0.010, OK).
Discordant counts: b-c = n*Delta_EM is exact, b+c = (|b-c|-1)^2/chi2 comes out integer, so b and c are determined:
B1 FEVER b+c=155 (b=81,c=74); B6 TriviaQA b+c=102.000001 (b=85,c=17); B7 StrategyQA b+c=158.000 (b=85,c=73); B8 TQA b+c=98 (b=48,c=50).
The table has no b/c columns, so none were added (the existing caption already says they are not persisted).

Full unified diff: /tmp/claude-0/-home-user-caem-thesis/96eedb08-eb50-59ca-a1c7-fdab15fb78c5/scratchpad/appD.diff
Pre-edit copy: /tmp/claude-0/-home-user-caem-thesis/96eedb08-eb50-59ca-a1c7-fdab15fb78c5/scratchpad/appendix_d.before.tex

## Changes (before -> after)

1. Line 39 prose (#2/#3).
 - BEFORE: "The B6 vanilla-fine-tune cell on TriviaQA records the largest single contingency gap in the panel ($\chi^{2} = 56.9$ at $n = 300$). The B7 FLARE cell on StrategyQA records a similarly large gap because FLARE's active-retrieval profile is poorly suited to StrategyQA's implicit multi-hop boolean surface. The B8 semantic-entropy cell on TruthfulQA records the tightest contrast in the panel under ..."
 - AFTER: "The B6 vanilla-fine-tune cell on TriviaQA records the second-largest single contingency gap in the panel ($\chi^{2} = 44.0$ at $n = 300$), behind only the B5 five-shot CoT cell on the same benchmark, whose raw $p$-value underflows to zero in double precision. The B7 FLARE cell on StrategyQA records a small lift that does not separate from zero ($\Delta_{\text{EM}} = +0.040$, $\chi^{2} = 0.766$, $p_{\text{raw}} = 0.38$). The B8 semantic-entropy cell on TruthfulQA records the tightest contrast in the TruthfulQA family under ..."
 - Rows: vanilla_ft,triviaqa (p 3.267e-11 -> chi2 44.0098); fiveshot_cot,triviaqa (p 0.0, diff +0.317, largest); flare,strategyqa (p 0.38151 -> chi2 0.7658).
 - "tightest in the panel" -> "tightest in the TruthfulQA family": semantic_entropy,fever has larger p (0.93493) and Delta 0.000, so the panel-wide claim was false; B8 TQA p 0.91954 is the largest p in the TQA family.
 - FLARE/StrategyQA causal explanation removed (no longer a large gap).
2. Line 52 B6 TriviaQA row (#2): `$0.190$ & $+0.263$ & $56.860$ & $< 10^{-12}$` -> `$0.227$ & $+0.227$ & $44.010$ & $3.3 \times 10^{-11}$`. Row vanilla_ft,triviaqa: em_base 0.22667, em_diff 0.22667, p_em_raw 3.267353e-11.
3. Line 53 B7 StrategyQA row (#3): `$0.250$ & $+0.390$ & $76.891$ & $< 10^{-12}$` -> `$0.600$ & $+0.040$ & $0.766$ & $0.382$`. Row flare,strategyqa: em_base 0.6, em_diff 0.04, p_em_raw 0.3815124.
4. Line 59 caption (#2/#3): "The B6 TriviaQA cell is the architecture's largest single-contrast lift in the panel; the B7 StrategyQA cell is the next-largest." -> "The B6 TriviaQA cell is the architecture's second-largest single-contrast lift in the panel, behind B5 five-shot CoT on TriviaQA; the B7 StrategyQA cell is a small lift that is not significant." (C5 em_diff ranking: fiveshot_cot triviaqa 0.317 > vanilla_ft triviaqa 0.227 > cot triviaqa 0.183.)
   Source sentence: "Source: the post-Haiku-rescore significance-test record." -> "Source: the post-Haiku-rescore significance-test record, with the B6 and B7 rows recomputed after the baseline fix (\Cref{tab:sig-test}); $\chi^{2}$ is the continuity-corrected one-degree-of-freedom statistic implied by the reported raw $p$-value."
5. Line 84 bootstrap row (#4): replaced (not relabelled) so that the table's third example and prose line 72 ("largest single-contrast bootstrap interval in the panel") stay true.
 - BEFORE: `B6 vanilla-FT $\cdot$ CommonsenseQA & $+0.717$ & $[+0.663,\, +0.767]$ & supported (largest panel lift)`
 - AFTER: `B5 five-shot CoT $\cdot$ TriviaQA & $+0.317$ & $[+0.253,\, +0.373]$ & supported (largest panel lift)`
 - Row fiveshot_cot,triviaqa: em_diff 0.31667, ci 0.25333/0.37333, sig_em *, holm 0.0. (Post-fix vanilla_ft,commonsense_qa is -0.063 [-0.120,-0.010], Holm 0.226, not significant.)
 - Prose line 72 unchanged (still true). Line 87 caption: "The B6 CommonsenseQA contrast is ..." -> "The B5 five-shot CoT TriviaQA contrast is ...".
 - Other bootstrap rows verified unchanged: zero_shot,fever +0.023 [-0.060,+0.110]; rag,truthfulqa +0.057 [-0.003,+0.117].
6. Line 98: "and FEVER is the cleanest single-bench win regime in the panel." -> "and FEVER is the one family in which no exact-match contrast clears the corrected threshold." (C5 sig_em on FEVER: all '-'; every other benchmark has at least one '*' or '(B)'.)
7. Both Holm tables: added the multiplier column m-k+1 (tabular {@{}llrrl@{}} -> {@{}llrrrl@{}}, header gains `$\boldsymbol{m-k+1}$`).
8. TruthfulQA Holm table (#5), rows rebuilt (raw p x multiplier, running max, clip 1; all match holm_p_em):
   1 B5 2.43e-5 x8 = 1.94e-4 (CSV 1.9444e-4) yes
   2 B7 1.07e-4 x7 = 7.49e-4 (CSV 7.4949e-4) yes
   3 B2 2.77e-4 x6 = 1.66e-3 (CSV 1.6591e-3) yes
   4 B1 6.25e-4 x5 = 3.13e-3 (CSV 3.1250e-3) yes
   5 B3 0.079 x4 = 0.316 (CSV 0.31620) no
   6 B4 0.083 x3 = 0.248 -> running max 0.316 (CSV 0.31620) no
   7 B6 0.512 x2 -> 1.000 (CSV 1.0) no
   8 B8 0.920 x1 -> 1.000 (CSV 1.0) no
   BEFORE order was B5,B2,B1,B3,B4,B6(0.347),B8,B7(0.923); B1 Holm was 3.75e-3, B3/B4 Holm 0.395.
9. TQA caption (#5): "Three of eight baselines clear the corrected threshold; the two CAEM defeats on TruthfulQA exact match (B1 zero-shot and B2 chain-of-thought, the high-exact-match retrieval-free baselines whose confidently-stated answers exploit the false-refusal subtype of the composite-hallucination metric) sit at ranks two and three." -> "Four of eight baselines clear the corrected threshold; the three CAEM defeats on TruthfulQA exact match (B7 FLARE, B2 chain-of-thought and B1 zero-shot, the three highest-exact-match baselines on the benchmark) sit at ranks two to four." (em_base flare 0.450, cot 0.437, zero_shot 0.433 are the three highest; sig_em (B) for exactly these three.) The "retrieval-free"/false-refusal clause was dropped because FLARE is retrieval-based.
   "The four CAEM defeats on the composite-hallucination-metric axis discussed" -> "The three CAEM defeats on the composite-hallucination-metric axis (B1 zero-shot, B2 chain-of-thought and B8 semantic-entropy) discussed" (sig_chm (B) at C5: cot, semantic_entropy, zero_shot). "B5 ... lone CAEM win" and "B8 raw p ~0.92, Holm ~1.0" verified, unchanged.
10. FEVER Holm table (#6), rows rebuilt:
   1 B6 0.029 x8 = 0.234 (CSV 0.23439) no
   2 B2 0.210 x7 -> 1 ; 3 B3 0.425 x6 -> 1 ; 4 B7 0.437 x5 -> 1 ; 5 B4 0.441 x4 -> 1 ; 6 B1 0.630 x3 -> 1 ; 7 B5 0.798 x2 -> 1 ; 8 B8 0.935 x1 -> 1 (running max). All CSV holm 1.0, all no.
   BEFORE: rank1 B7 <1e-6 yes; rank2 B6 6.93e-3 / 0.0485 yes; then B2..B8 as above.
11. FEVER caption (#6): "Two of eight baselines clear the corrected threshold: B7 FLARE (CAEM win by a large exact-match margin) and B6 vanilla-fine-tune (CAEM loss by approximately eight percentage points; the matched-scale fine-tune contrast is the registered defeat). The remaining six baselines sit at adjusted $p$-values of one, the empirical receipt that the cycle-five trained architecture and the prompt-only baselines do not separate on FEVER under the matched-protocol contract." -> "None of the eight baselines clears the corrected threshold. The smallest raw $p$-value belongs to B6 vanilla-fine-tune (CAEM loss of approximately seven percentage points, $\Delta_{\text{EM}} = -0.067$), which does not survive the step-down correction (Holm $p = 0.234$); B7 FLARE ($\Delta_{\text{EM}} = +0.040$) is not separable either. The remaining seven baselines sit at adjusted $p$-values of one, the empirical receipt that the cycle-five trained architecture does not separate from any baseline on FEVER exact match under the matched-protocol contract." (vanilla_ft,fever diff -0.0667, holm 0.2344; flare,fever diff +0.040, p 0.4367.)

## Unchanged, verified
- B1 FEVER row (0.477/0.453/+0.023/0.232/0.630) and B8 TQA row (0.300/0.307/-0.007/0.010/0.920): chi2 and p now sourced from the CSV (closes U5/U10 note).
- Final paragraph tallies (C5 9/3, 21/3; C2 14/3, 20/4) and the CHM defeat list (B1, B2, B8 both anchors, +B7 at best cycle) match both CSVs.

## Not sourced / not touched
- B5 five-shot CoT TriviaQA chi^2 is not quoted anywhere (p_em_raw = 0.0 underflow); prose says only that its p underflows, no chi^2 invented.
- The "0.210 -> 0.310" RUC-ablation TruthfulQA figure (line 155) is outside the significance CSVs; not checked.
- The output path cited at line 144 (outputs/full_run/tab_sig_test.csv) and the script name at line 14 (baseline_sig_tests.py vs the v2 script used today) were not changed.

## Checks
- Column counts: table 1 = 7 cols (5 rows), bootstrap = 4 cols (4 rows), both Holm tables = 6 cols (18 rows incl. headers). Braces balanced, $ count even.
- \Cref{tab:sig-test} label exists (chapters/chapter_5.tex:405).

## Leftovers (2026-10-15)

| Location | Before | After | Evidence |
|---|---|---|---|
| appendix_e:93 | "Per-cycle wall-time is approximately five to seven hours on the production GPU." | Wall time grows with the cumulative memory, from roughly six hours at C1 to more than twenty-two hours at C5. | Measured windows in experiment.log: 5 h 45 m (C1) to 21 h 50 m (C5) |
| appendix_e:166 | "The cycle close consumes approximately five to seven hours end-to-end." | A full cycle takes at least the measured windows plus the fine-tune and stream-chunk steps, from roughly six to more than twenty-two hours. | same |
| README:214 | "by margins of 5–20 percentage points" | "by margins of 4.6–20.4 percentage points" (TriviaQA 4.6, TruthfulQA 6.2, StrategyQA 6.5, FEVER 8.1, CommonsenseQA 20.4) | tab_per_tier_em: C2 T2 vs C0 T3 |
| cross_benchmark_summary caption | "the transfer panel … is held out and does not enter the storage gate at cycle zero", while the table lists transfer-panel store counts | "is held out from calibration and from the self-improvement pool, and its storage columns report the gate's decision on held-out queries as a diagnostic only" | the table's own rows |

| cohen_d, tab_validation_gate captions (+ scripts/aggregate_section_5_7.py) | "precondition of \\Cref{thm:purity}" | "precondition of \\Cref{thm:bayes-purity}" | ch5:341, :506 tie the d ≥ 0.20 gate to thm:bayes-purity after the theory restructure |

README consistency pass logged separately in ref_check/fixes_readme.md.

## Follow-ups after merge (2026-10-15)

| Location | Before | After | Reason |
|---|---|---|---|
| tab_decision_distribution | panel (b) "Per-benchmark snapshot at the cycle-three anchor (calibration fold, n=500 each)": FEVER/TriviaQA/CommonsenseQA shares and EMs; caption referred to panel b | panel (b) removed; caption trimmed | local review found no source for panel (b) in the data, logs or code; no prose cites its values |
| tab_decision_distribution (correction, 2026-10-15) | panel (b) removed | panel (b) restored with identical values; caption now limits the store > defer > discard > abstain ordering to the pooled panel and states the per-benchmark exceptions (FEVER discard, 21 queries; CommonsenseQA discard and abstain, 8 each); the false "discussed in the surrounding prose" sentence stays removed | the removal was wrong: all 36 panel (b) values recompute exactly from outputs/full_run/cycle_3/calibration/{fever,triviaqa,commonsense_qa}_cycle3.json with the panel (a) computation (the earlier search missed these per-benchmark files) |
| ch5:181 | "the present reading is therefore registered as an as-shipped null receipt" | "the present reading, sixty-three firings concentrated on CommonsenseQA after its floor refit, is therefore registered as a low-rate receipt" | contradicted the corrected 63/9,000 count |
| README, "Checking the tables" | — | "do not regenerate by overwriting" table: survival distribution, decision distribution, tier trajectory, validation gate, theorem receipts, hand-built tables | aggregate_survival_distribution disagrees with the run log; several tables are hand-finished |

---

# Final drafting pass (2026-10), working copy `defense/thesis_report/`

## Front matter (branch `review/front`)

| Location | Before | After | Evidence / reason |
|---|---|---|---|
| core/abstract.tex, para 2 | "StrategyQA shows a clean transfer-panel win of $-17.5$ percent composite hallucination at trajectory end." | "On the transfer panel, StrategyQA's composite hallucination falls by $17.5$ percent at trajectory end, a reduction that does not reach statistical significance." | tab_per_bench_trajectory StrategyQA CHM 0.120 → 0.099 (−17.5%); tab_sig_test zero_shot·strategyqa Sig "- / -", ΔCHM CI [−0.017, +0.002] includes 0; StrategyQA EM at C5 is −1.5% vs B1 |
| core/abstract.tex, para 2 | "and reaches exact-match parity with the strongest fine-tuned and active-retrieval baselines at the trajectory-end cycle;" | "and at the trajectory-end cycle leads the fine-tuned baseline on pooled exact match ($0.517$ against $0.490$) while trailing the active-retrieval baseline by $0.008$;" | tab_baseline_pooled: CAEM C5 0.517, B6 0.490 (+0.027), B7 FLARE 0.525 (−0.008); REVIEW_HANDOFF A |
| core/abstract.tex, para 3 | "and the registered ablations are reported in the empirical chapter and appendices." | "and the retrieval-utility classifier ablation are reported in the empirical chapter and appendices." | ch5 L463: the RUC on/off contrast "is the one architectural ablation reported within the realised cycle horizon"; outputs/ablation/ holds only aggregate_ablation.log, no variant results |
| core/approval.tex | "Large Language Models ”" | "Large Language Models”" | stray space before closing quote |
| core/lists/list_of_symbols.tex | $\phi_{\text{pg}}$ "Pre-routing safety floor" | "Abstention floor on top-passage entailment ($0.20$)" | every use in the text is the abstention floor 0.20 (ch4 early-exit rule, ch5 L301, sweep_top, tab captions) |
| core/lists/list_of_symbols.tex | (missing) | new row $\phi_{\text{safe}}$ "Pre-routing safety floor on $\upre$ ($0.38$; per-benchmark refit)" | used in 9 places (ch2:301, ch4:142/144, ch5:308/595, app A/C/F, tab_calibration_trajectory); config `safety_u_pre_min = 0.38` |
| core/lists/list_of_symbols.tex | $B$ "Bootstrap replicates ($10\,000$)" | "($1\,000$)" | scripts/baseline_sig_tests_v2.py:142,237 `n_bootstrap=1000`; ch5:376 and appendix D:68,87 already say 1,000 |
| core/lists/list_of_symbols.tex | "$\mathrm{EM},\mathrm{F}_1,\mathrm{ROUGE\text{-}L}$" | "$\mathrm{EM},\mathrm{F}_1$" | ROUGE-L never used in the text (0 occurrences) |
| core/lists/list_of_acronyms.tex | ROUGE-L "Longest-common-subsequence overlap" | ROUGE "Recall-Oriented Understudy for Gisting Evaluation" | "ROUGE" used 4× undefined (appendix D:26 ×3, ch4:165); "ROUGE-L" used 0× |
| core/lists/list_of_acronyms.tex | BCa, SOTA, SDPA | TPR, FPR, TTL | BCa/SOTA/SDPA used 0× (intervals are percentile, not BCa); TPR 20×, FPR 16×, TTL 8× used but unlisted |

Build after these edits (latexmk): 324 pages, 0 errors, 0 undefined refs/citations, 0 multiply-defined labels, overfull hboxes 74 (baseline 75).

### Front matter, part 2: BRAC template formatting and dates (author decisions Q2, Q3, Q6)

Template reference: `archive/brac_thesis_template_reference/main.tex`.

| Location | Before | After | Evidence / reason |
|---|---|---|---|
| main.tex class | `\documentclass[12pt,a4paper,oneside]{book}` | `\documentclass[12pt,a4paper,oneside,openany]{report}` | template uses `report` with `oneside,openany` |
| main.tex geometry | top/bottom 1in, left 1.25in, right 1in (text width 434.9 pt) | `width=150mm,top=25mm,bottom=25mm` | template geometry |
| main.tex spacing | `\onehalfspacing` | single spacing | template sets no spacing (default single) |
| main.tex paragraphs | default indent | `\setlength{\parindent}{0em}` | template |
| main.tex page style | book running headers | `\pagestyle{plain}`, `\let\cleardoublepage=\clearpage` | template |
| main.tex hyperref | `urlcolor=blue`, `hidelinks` | all link colours black, `\urlstyle{same}` | template `\hypersetup` |
| main.tex biblatex | `style=numeric-comp,sorting=none,maxbibnames=99` | `style=ieee,sorting=ynt` | template biblatex line (installed `biblatex-ieee` via tlmgr) |
| main.tex front matter | `\frontmatter` … `\mainmatter` | title page `\thispagestyle{empty}`; `\pagenumbering{roman}` after it; `\pagenumbering{arabic}` before chapter 1; TOC entries for Declaration, Approval, Abstract, Acknowledgment, Table of Contents, List of Figures/Tables/Algorithms; `\contentsname` = "Table of Contents" | template front-matter block (empty optional Ethics/Dedication pages get no TOC entry) |
| core/abstract.tex | `\chapter*{Abstract}` + own `\addcontentsline` | `\section*{Abstract}` … `\pagebreak` (TOC entry now in main.tex) | template abstract.tex |
| core/titlepage.tex | "February 2026." | "Fall 2026." | author: final defense is in Fall 2026 (exact month to confirm) |
| core/approval.tex | "… degree of B.Sc. in Computer Science in Spring 2026." | "… in Fall 2026." | author: final defense Fall 2026 (start semester "Fall, 2025" kept) |
| core/acknowledgement.tex | empty | draft thanking supervisor, co-supervisor, department, families | template expects the page; DRAFT for the authors to personalise |

Build (latexmk, clean): 279 pages (was 324: single spacing), 0 errors, 0 undefined refs/citations, 0 multiply-defined labels,
0 biber warnings, overfull hboxes 90 (>10 pt: 40; narrower 150 mm text block). Visual check: references now IEEE-formatted and
year-sorted; with `\parindent 0em` and no `\parskip` (as in the template) consecutive paragraphs are not visually separated
(open question Q8).
| main.tex paragraphs (Q8) | `\parindent 0em`, no paragraph spacing (consecutive paragraphs visually merged) | `\usepackage[skip=0.5\baselineskip plus 2pt]{parskip}` | the template sets only `\parindent 0em` and says nothing about spacing; author delegated the choice. Build: 281 pages, 0 errors/undefined/multiply-defined; overfull 90 (unchanged) |

## Chapter 1 (branch `review/ch1`)

Line numbers refer to `chapters/chapter_1.tex` before this pass (base `3923766`).

| Location | Before | After | Evidence / reason |
|---|---|---|---|
| L14 | "incremental decoding tricks" | "incremental decoding adjustments" | informal wording |
| L19 | "fifty-eight percent … twenty-one percent … ninety-four and eighty-seven percent" | "$58$ percent … $21$ percent … $94$ and $87$ percent" | number style: digits + "percent", as in the abstract (carried item); values verified (reference_check_2_claims row 205) |
| L21 | "only fifty-eight percent of decomposed atomic claims were supported by retrieved evidence, and the remaining claims ranged from unsupported speculation to outright fabrication" | "only $58$ percent of the atomic facts in ChatGPT's biographies were supported by the reference knowledge source" | FActScore reports the supported share for ChatGPT; "speculation to fabrication" is not in the paper |
| L30 | "above ten percentage points"; "about zero point six three, close to the zero point five" | "above $10$ percentage points"; "about $0.63$, close to the $0.5$" | number style (carried item); Xiong et al. AUROC 62.7% |
| L51 | "each yield diminishing returns when used in isolation" | "each remain bounded by the structural limitation of their family when used in isolation" | uncited empirical claim → argument that follows from the preceding paragraph |
| L56 (2nd capability) | one 90-word sentence | split in two, same content | readability |
| L56 (3rd capability) | "fine-tunes on a verified high-quality pool under a regulariser pulling weights toward the base parameters \cite{EWC}" | "fine-tunes a small low-rank adapter on a frozen backbone … so that the bounded adapter plays the role that the parameter anchor plays in elastic weight consolidation \cite{EWC}" | code: `caem/config.py:670` "L2 anchor: REMOVED on the LoRA path"; `use_lora_training = True`; run log "LoRA wrap OK … trainable=59.87M / 3145.81M (1.9031%)". Consistent with ch1 L120/L280/L397 |
| L58 | "A practical consequence of the joint architecture is that the three serving tiers are not static cost categories but a self-organising compute ladder" | "… joint architecture, named the confidence-aware episodic memory with self-improvement (CAEM), is that the three serving tiers are intended to act … as a compute ladder that reorganises itself" | "CAEM" was used in L58 before being defined anywhere in the chapter; design framing |
| L58 | "The next cycle is therefore able to answer a measurable share of the previously retrieval-dependent queries from the zero-shot tier alone" | "The design expectation is that the next cycle answers part of …" | no measurement of this share exists |
| L58 | "At cycle zero about half of the queries ($51.3\%$) fall to the retrieval-augmented tier … cycle after cycle, the zero-shot tier absorbs … the retrieval-augmented tier retains only the genuinely novel queries … The architecture iterates this redistribution until … bounded stationary band …; the six-cycle horizon … is a finite, verifiable demonstration of the same pattern" | design described as design; realised: "the retrieval-augmented tier served $51.3$ percent … at cycle zero and between $44.7$ and $55.1$ percent at later cycles, and the memory-direct tier served at most $0.1$ percent, because the evaluation questions are content-disjoint from the stored episodes by construction (\Cref{tab:tier-trajectory})" | tab_tier_trajectory pooled shares T3 51.3/50.2/44.7/55.1/54.3/49.7%, T1 0.0–0.1%; ch5 L163 ("second redistribution empirically muted and the first structurally invisible"; Tier 1 fires on 7 of 9,000) |
| L60 | "hallucination is reduced cycle after cycle …, per-query cost falls as queries migrate …, and the user is never given a confident answer that the system itself could not certify" | "The architecture therefore targets three deployment-time properties at once: hallucination that falls across cycles …, per-query cost that falls as queries migrate …, and answers that always carry the system's own calibrated estimate of their reliability" | pooled CHM 0.135/0.112/0.107/0.115/0.115/0.109 (tab_pooled_trajectory) is not monotone; per-query cost reduction not measurable (ch5 L510); DISCARD answers are served with a "conflicting" tag (`caem/production/confidence.py:54`), so "never" overclaims |
| L62 | "the purity of the stored pool can exceed the verification accuracy itself, because base-rate dynamics favour the gate whenever the verifier is more likely to admit a correct answer … together with the conditions under which it implies monotonic improvement across cycles, appears in \Cref{ch:methodology} … the empirical chapters report whether that condition is satisfied" | "By Bayes' rule, the stored pool is purer than the generator's raw output whenever the storage gate is more likely to admit a correct answer than an incorrect one … The formal statement is \Cref{thm:bayes-purity}; how the property is expected to carry across cycles is stated … as a design heuristic rather than a proven result … \Cref{ch:evaluation} reports the evidence on whether that condition holds" | ch4 Proposition (thm:bayes-purity): TPR > FPR ⇔ P_c > p_c (raw accuracy), not > verifier accuracy; ch4 states monotonicity/convergence as heuristics (sec:theory-heuristics) |
| L92 | "and that satisfies the four properties registered in \Cref{sec:formal-problem}" | "and that is designed to satisfy, and evaluated against, the four properties …" | the cost-reduction and precision-under-shift properties are not shown (ch5 L510; conformal rejected for shift) |
| L100 | "The architecture, named the confidence-aware episodic memory with self-improvement, addresses" | "The architecture, CAEM, addresses" | acronym now defined at first use (L58) |
| L100 | "guarded by a multi-modal retention probe on held-out distributions" | "… multi-modal retention probe, a panel of held-out probes that spans three task formats (general-knowledge multiple choice, open-text factoid, and commonsense multiple choice)" | "multi-modal" normally means input modalities; the term is fixed across ch2 (frozen)–ch5, so it is glossed at first use instead of renamed |
| L102 | "anchored against a single registered scalar that any reader can locate in the configuration" | "anchored to a single scalar fixed before the main run" | code-like wording |
| L112 (Obj 2), L168 (Stage 1), L268, L399, L411 | "single short forward pass … short-greedy-decode token-probability aggregate with an encoder-layer convergence ratio" (5 sites) | "short greedy decode over the query … token-probability aggregate (of the first thirty-two decoded tokens) with a layer-convergence ratio (read from the hidden states over the query)" | `caem/confidence/pre_routing.py`: greedy decode `max_new_tokens=32` + decoder-stack hidden-state variance ratio; backbone is decoder-only ("Flan-T5 / encoder-decoder support has been removed") |
| L118 (Obj 5) | "rescores stored episodes under the updated model and prunes any entry whose composite has fallen below the storage threshold" | "… overwrites each stored composite with its new value, raised or lowered, and prunes any entry whose composite has fallen below a prune threshold set beneath the storage threshold" | `caem/config.py:859` `retroverify_prune_threshold = 0.50` (storage 0.60); `:876` `retroverify_allow_downgrade = True`; `caem/memory/store.py:661`; ch4 thm:purity (ii) "τ_retro < τ_store … raised or lowered" |
| L128 | "pair with the load-bearing formal claims registered in \Cref{ch:methodology}" | "pair with the formal statements of \Cref{ch:methodology}, which comprise two proven results, design heuristics, and a decomposition remark" | ch4: thm:purity theorem, thm:bayes-purity proposition, thm:convergence heuristic, thm:asymptotic-elim remark |
| L137 (H4) | "the asymptotic-elimination identity recorded in \Cref{thm:asymptotic-elim}" | "the full-pipeline hallucination decomposition recorded in …" | the ch4 remark's title; hypothesis content unchanged |
| L147 (tab:hyp-registration) | `p{0.6cm}p{6.6cm}p{4.5cm}p{2.6cm}` | `p{0.6cm}` + ragged-right `p{6.0cm}p{4.3cm}p{2.6cm}` | 16 pt overfull under the template's 150 mm text width; justified narrow columns spread badly |
| L275 | "… breaks the marginal-coverage premise of the conformal procedure, with the realised eval-fold precision falling fifteen to fifty percentage points below the conformal target on training benchmarks." | "… breaks the marginal-coverage premise of the conformal procedure (\Cref{sec:conformal-gate})." | "15–50 pp" traces only to archive/branch_C_log_historical.md:4788; ch5 L317 says the artefacts of the rejected formulation were not retained. Number left to ch4/ch5 (carried) |
| L280 | "rescores every stored episode …, promotes entries whose composite has crossed the storage threshold, and prunes entries whose composite has fallen below it" | "… updates each stored composite to its new value, and prunes entries whose composite has fallen below the prune threshold" | as L118; promotion is the deferred-buffer pass, not the retroactive pass |
| L280 | "the cycle probes a held-out distribution and computes a retention ratio. When the retention ratio falls" | "the cycle probes the three held-out retention surfaces and computes the worst per-probe retention ratio. When that ratio falls" | three-probe worst-ratio rule (L383, L397; ch5 sec:retention) |
| fig:experimental-workflow caption | "Self-improvement trajectory across early, mid, and late phases of the realised six-cycle horizon." | "Intended self-improvement trajectory across early, mid, and late phases (design view; the realised per-cycle readings are in \Cref{ch:evaluation})." | figure shows "Tier 1 share rises / Highest Tier 1 share"; realised Tier 1 ≤ 0.1% (tab_tier_trajectory). Figure body unchanged (see decision D6) |
| L378 | `\section{Scopes and Challenges}` (no label) | + `\label{sec:scope}` | only unlabelled section in the chapter |
| L383 | "Each benchmark admits an objective scoring rule, namely exact match or label accuracy, depending on the task family." | "…; TruthfulQA's free-text answers are judged against the benchmark's reference answers by a language-model judge." | TruthfulQA EM is the Haiku-judged field (standing rule) |
| L387 | "Second, its baseline accuracy on the five benchmarks … is high enough that the verifier's expected false-positive rate on stored episodes can be bounded below the level at which a self-improvement loop turns counterproductive." | "Second, as an instruction-tuned model it has competitive baseline accuracy on factual question answering for its size, which the self-improvement argument requires (see the baseline-performance challenge below)." | no such bound is computed anywhere; B1 TriviaQA 0.293, TruthfulQA 0.433 |
| L387 | "predicted by the convergence analysis in \Cref{ch:methodology}, while remaining feasible within a twelve-month compute envelope." | "expected by the convergence heuristic …; the realised trajectory closed at cycle five under the equilibrium-saturation gate." | convergence is a heuristic in ch4; experiment.log:70711 "equilibrium gate \| cycle=5 \| … stop=True" |
| L387 | "grow the memory to the low tens of thousands of entries, well within the capacity envelope and sufficient to exercise the novelty filter and the value-score pruning rule without saturating them." | "grow the memory to $12\,175$ entries at cycle five (\Cref{tab:memory-store-trajectory}), below the size at which the index switches to the quantised structure and far below the capacity at which value-score pruning triggers, so the realised run uses exact similarity search and never invokes pruning." | tab_memory_store_trajectory C5 = 12,175; `caem/config.py` `faiss_train_min_points = 20_000` (IVF-PQ promotion), `max_memory_size = 1_000_000`, `pruning_trigger = 0.95`; novelty skips are logged at debug level only, so their count is not recoverable |
| L395 | "applies an upward-only update rule that promotes entries whose composite has crossed the storage threshold and prunes those that have fallen below it. … an order of magnitude smaller than fine-tuning itself but not negligible at scale." | "…, raising or lowering it to the new value, and prunes entries whose composite has fallen below the prune threshold. … which grows with the size of the memory." | as L118; experiment.log: retroactive pass 431 entries ≈ 38 min (C1) vs 10,214 entries ≈ 16 h (C5); no fine-tune comparison supports "order of magnitude" |
| L397 | "The empirical contribution of each layer is verified through the architectural ablations reported in \Cref{ch:evaluation}, of which the retrieval-utility-classifier on/off contrast … is the principal landed comparison." | "The three layers are not ablated individually within the realised horizon; the retention guard's effect is visible in the cycle-four rollback …, and the only architectural ablation reported there is the retrieval-utility-classifier on/off contrast." | ch5 L463 (RUC contrast is the one architectural ablation); outputs/ablation holds no variant results; tab_pooled_trajectory C4 ABORT |
| L401 | "The episodic memory uses approximate nearest-neighbour search, which raises …" | "At deployment scale the episodic memory uses approximate nearest-neighbour search …; at the scale of the realised run the search is exact." | IVF-PQ promotion at 20,000 vectors > 12,175 |
| L401 | "…held constant across cycles, with sensitivity analyses reported in \Cref{ch:evaluation}." | "…held constant across cycles." | no novelty-threshold sensitivity analysis exists in ch5 or the appendices |
| L403 | "and the in-distribution and out-of-distribution training-pool gate." | "and the in-distribution training-pool gate that excludes transfer-benchmark episodes from fine-tuning." | consistent with L411 |
| L405 | "with identical samples, prompts, and scoring rules" | "with the same evaluation samples, backbone, answer format, and scoring rules" | baselines use their own prompting methods (CoT, self-consistency, FLARE, …) |
| L411 | "retroactive re-verification pass with its upward-only update rule"; "triggers only when capacity is exceeded"; repeated CES definition; "The empirical receipts for each mitigation are reported in \Cref{ch:evaluation}." | "… pass and its prune threshold"; "triggers only near capacity"; CES definition removed (stated in L405); "\Cref{ch:evaluation} reports the evidence available for each mitigation on the realised trajectory." | as L395; redundancy; not every mitigation has a receipt (novelty threshold) |
| L413 | "Although the problems are significant, they can be managed within the resources and the time frame. The aim is to improve the system clearly and measurably …" | removed | stale proposal-stage text |
| L418 (signpost) | "will adjudicate"; covers chapter 2 only | "adjudicate"; adds one sentence each for chapters 3–6 and the appendices | signpost must match the thesis that follows |

Build (latexmk, clean): 281 pages, 0 errors, 0 undefined refs/citations, 0 multiply-defined labels, 0 biber warnings, overfull
hboxes 89 (base 90; chapter 1 now has two small ones: 2.0 pt in the H1 list item, 3.9 pt in the challenges list).

### Chapter 1, part 2: author decisions D1–D6 (plus D1 sites in other files)

| Location | Before | After | Evidence / reason |
|---|---|---|---|
| ch1 rationale (scope paragraph), ch1 sec:scope-research | "Three benchmarks registered earlier … (HotpotQA, Natural Questions, and ARC-Challenge) were retired … on cycle-zero diagnostic evidence that they violated the verifier-balanced-accuracy precondition …; all three are kept on disk …" | in sec:scope-research: "Two further benchmarks, HotpotQA and Natural Questions, were evaluated at cycle zero and dropped …: the base generator answered only $9$ and $17$ percent of their questions correctly, respectively, and the storage gate kept just $26$ and $12$ of $500$ answers, of which half or fewer were correct, too few and too noisy to train on. Their cycle-zero results are preserved in the project archive, and the Natural Questions training split is reused only to train the retrieval-utility classifier." | D1 (author: drop ARC entirely). Receipts: Drive `CAEM_thesis/9_archive/pre_main_snapshot/700b9e3_phase1d/cycle_0/eval_excluded_v21/hotpotqa_cycle0.json` (n 500, EM 0.090, 26 stored, 13 correct, purity 0.500, gate TPR 0.289 / FPR 0.029, balanced accuracy 0.630) and `9_archive/v1_panel/full_run/cycle_0/eval/natural_questions_cycle0.json` (n 500, EM 0.172, 12 stored, 5 correct, purity 0.417, balanced accuracy 0.521). Balanced accuracy was above one half on both, so the old reason was false; base accuracy was the problem |
| ch2 L275, L281, L287, L291 (frozen chapter; author-directed) | retirement on the balanced-accuracy precondition; ARC-Challenge paragraph; repeated retirement sentence | NQ/HotpotQA dropped for low base accuracy (pointer to sec:scope-research); ARC paragraph and repeated sentence deleted | D1. ch2 L212 keeps "ARC-Challenge" because it reports a cited study's metric, not a CAEM benchmark |
| ch3 L145, ch5 L41, appendix E L15 | same retirement claim, three benchmarks, "kept on disk as registered exclusion evidence", "verifier-balanced-accuracy audit" | HotpotQA and NQ only, dropped for low base accuracy ($0.09$, $0.17$), "preserved in the project archive", "cycle-zero diagnostics" | D1 |
| ch4 L555 | "…no purer than the generator's raw output, which is the ground on which HotpotQA, Natural Questions, and ARC-Challenge were retired before the main run." | "…no purer than the generator's raw output." | D1: the gate's balanced accuracy was 0.63 and 0.52 on the dropped benchmarks, so this was not the ground |
| list_of_acronyms | NQ "(NQ-Open; registered then retired)"; ARC row; HotpotQA "(registered then retired)" | NQ "(dropped at cycle zero; RUC training only)"; ARC removed; HotpotQA "(dropped at cycle zero)" | D1 |
| ch1 tab:hyp-registration, ch5 tab:hypothesis-verdicts (H2 mapping only) | H2 → \Cref{thm:purity} | H2 → \Cref{thm:bayes-convergence} | D2. thm:purity is a lower bound (π_{t+1} ≥ min(π_store, π_retro)); it cannot predict "non-decreasing". The statement that predicts monotone purity under α > 1/2 is the Bayes-purification recurrence (ch4 remark): f(P) = Pα/(Pα+(1−P)(1−α)) > P ⇔ α > 1/2, checked. Hypothesis wording unchanged (pre-specified). ch5 verdict left for the ch5 session |
| ch1 hypotheses H1–H5 explanatory notes | jargon-heavy notes ("empirical precondition of the purification condition on which the pool-purity analysis rests", "convergence prediction in its reframed form", "committed-cycle subsequence", "asymptote is bounded by the verifier balanced accuracy and the base accuracy") | plain notes: what each tests and how it is decided; "committed cycles" defined; H4 floor described as in the remark (memory-tier hit rate and generator correctness ceiling) | D2 / author request for readable theory. H4's old note did not match thm:asymptotic-elim, whose floor is (1−τ₁)(1−x), not a function of verifier balanced accuracy. Bold hypothesis statements unchanged |
| ch1 L128 | "two proven results, design heuristics, and a decomposition remark" | "two proven results, design heuristics, and remarks on idealised models"; adds a one-line reading guide | H2 now maps to a remark too |
| ch1 L72 (problem statement) | "whose user-facing answers carry a precision guarantee that holds under modest distribution shift between calibration data and live data" | "whose user-facing answers carry a calibrated probability of being correct that is refitted on labelled data at every cycle, so that its precision can be checked against held-out data" | D3. No guarantee under shift exists: the conformal gate was dropped because shift broke it, and thm:purity assumes exchangeability. The realised design refits the composite per cycle (code: per-cycle CalProbComposite refit) |
| ch1 sec:three-shortcomings + sec:coupling-exclusions | two subsections restating sec:stateless-gap (≈470 words) | one subsection "Shortcomings addressed and exclusions": one paragraph pointing back to sec:stateless-gap + the unchanged exclusions paragraph | D4 (labels unreferenced elsewhere) |
| ch1 objective section, 2nd paragraph | calibration-supervised regime, fixed gate, 1500-sample fold, production fold | moved into sec:calibration-discipline | D4: methodology inside Objective |
| ch1 Obj 2 | RUC fitting details (nested CV, 36 features, threshold) | "(\Cref{sec:ruc}); its contribution is measured by the on/off ablation in \Cref{ch:evaluation}" | D4: detail lives in ch4 |
| ch1 Obj 4 | "selected against the cycle-zero threshold sweep so that the realised per-store empirical precision on the calibration fold is anchored to a known number"; conformal-rejection history | "all fixed before the main run …; the storage cut-point is chosen on the cycle-zero calibration fold and sets how sure the system must be before an answer enters memory; the deferral cut-point, set lower, …" | D6: no number in ch1, so it cannot conflict with the Q4 sweep outcome; rejected-design history removed from ch1 (kept in ch4 sec:conformal-gate) |
| ch1 Obj 7 | full panel listing (third copy) | "the five-benchmark panel and the three-probe retention panel described in \Cref{sec:scope-research}" | D4 |
| ch1 sec:arch-overview stage 3, stage 5–6 | tiers unnamed; stage 5 verifier details duplicated in sec:per-query-pipeline | tier bands named (moved from the deleted paragraph); stage 5–6 shortened to a pointer | D4 |
| ch1 sec:per-query-pipeline | first paragraph repeated stages 1–4 (≈230 words); title "Per-query pipeline" | paragraph deleted; title "Verification and the storage decision"; early-exit rule stated once here | D4 |
| ch1 sec:calibration-discipline | sweep "known number"; conformal history (EMA refit etc.) | fixed cut-points chosen before the main run, conformal pointer to sec:conformal-gate; adds the moved calibration-regime and 1 500-sample fold text | D4, D6 |
| ch1 sec:scope-mitigations | subsection restating each challenge bullet's mitigation | deleted; one closing sentence after the bullet list | D4 (label unreferenced) |
| ch1 rationale scope paragraph | panel, retired benchmarks, horizon, hardware (duplicated sec:scope-research) | two sentences + pointer to sec:scope; horizon (with \cite{song2024gap}) and sixteen-bit arithmetic moved into sec:scope-research | D4 |
| ch1 L58 | (no repeat-question evidence) | "+ The memory-direct tier is built for repeated and paraphrased questions: on a separate probe of $100$ questions built from episodes stored by cycle five, it fired on $42$ of $50$ exact repeats and $19$ of $50$ paraphrases, and its answers were correct in $41$ of the $42$ and all $19$ cases." | D5 / Q7; `outputs/tier1_routing_probe/em_analysis.json` (t1 exact 42 fired, 41 correct; paraphrase 19, 19) |
| fig:experimental-workflow bullets + caption | "Tier 1 share rises"; "Highest Tier 1 share"; "Retroverify promotes/prunes" | "Tier 1 serves repeated and paraphrased questions"; "Tier 1 share highest when questions recur"; "Retroverify rescores/prunes"; caption adds that Tier 1 rarely fires on the distinct evaluation questions | D5 (author: Tier 1 hits only on duplicates/paraphrases; evaluation inputs are distinct) |
| ch1 sec:scope-research | "TruthfulQA's free-text answers are judged … by a language-model judge." | "+ following the judge-based evaluation introduced with the benchmark \cite{lin-etal-2022-truthfulqa}, and the same judge scores every system." | author question: LLM judging follows the TruthfulQA paper's own GPT-judge protocol; appendix D states the uniform Haiku-judged rule across systems |
| ch1 challenge "pre-routing confidence" | "two cheap signals from a single short forward pass, namely a short-greedy-decode token-probability aggregate" | "two cheap signals, namely the token-probability aggregate of a short greedy decode" | leftover from part 1 |

Build (latexmk, clean): 278 pages (was 281), 0 errors, 0 undefined refs/citations, 0 multiply-defined labels, 0 biber warnings,
overfull 89. Chapter 1: 9 215 → 8 377 words.

### Chapter 1, part 3: author read-through (2026-10-16)

- sec:stateless-gap retitled "The stateless gap" → "Three reinforcing shortcomings" (the section lists three distinct shortcomings;
  the phrase matches the objective; label kept). The First/Second/Third paragraphs now open with bold lead-ins:
  **No memory.** / **No reliable confidence.** / **No learning after deployment.**
- sec:prior-pipelines: the single paragraph split into three with bold lead-ins (**Retrieval-augmented generation.**,
  **Prompting interventions.**, **Post-generation verification.**); "verification ... supports/does" → plural to match
  "verifiers such as SelfCheckGPT and semantic entropy".
- Citation style (author rule): studies named where the sentence relies on their finding or number: Guo et al., Kadavath et al.,
  Xiong et al., Kirkpatrick et al., Lewis et al., Wei et al., Wang et al., Manakul et al., Farquhar et al., Alkaissi and
  McFarlane, Chelli et al.; "Lin and colleagues" / "Min and colleagues" → "et al." with the citation next to the name;
  bracket-as-noun fixed (Valentin et al.; Song et al.). Bare citations kept on general background sentences.

## Chapter 2 (branch `review/ch2`, stacked on `review/ch1`)

Line numbers refer to `chapters/chapter_2.tex` at `6ec1993`. Chapter 2: 16 152 → 11 740 words; thesis 278 → 268 pages.

| Location | Before (gist) | After (gist) | Evidence / reason |
|---|---|---|---|
| L10, L12 | broken grammar ("quite big text collections", "is adopted by new LLMs"); "we utilize"; Qwen "largest backbone that fits"; "verifier outputs sharp enough … from the first cycle" | plain third-person description; backbone chosen because the loop fits one GPU in bf16 and instruction tuning gives the required answer format | style rule (third person, British); "largest" and "sharp enough" were unsupported; Qwen2.5 report: post-training uses DPO + GRPO |
| L25 (taxonomy) | 400-word paragraph with backend history (Qwen judge Pearson floor, p_contra retired "on 2026-04-22", "short-circuited at the verifier compute path"), long footnote on semantic entropy | four subclasses → signal families in plain words; exclusions kept; footnote points to ch4 | design history and code-level detail belong in ch4 (author: trim duplicates); dates/code wording banned in prose |
| L36 | "if we assume … and hypothesis is", straight quotes | premise/hypothesis example with LaTeX quotes | grammar; typography |
| L38 (NLI) | contradiction-veto history; RoBERTa "retained as a Chapter 5 ablation target (sec:comp-ablations) so the backend choice is defended by trajectory evidence"; bidirectional entailment as a CAEM purpose | MiniCheck described; long-hypothesis judge dropped (pointer to ch4); RoBERTa named as the conventional alternative with the Sinha et al. transfer result (moved from the old summary); bidirectional entailment tied to semantic entropy, used only as baseline B8 | ch5 L463: the RUC contrast is the only architectural ablation, so the RoBERTa promise was undeliverable; CAEM's verifier does not cluster answers |
| L42, L48, L50 | "map the length of variable text sequences"; "-1" in text; "approximately 0.98"; FAISS "approximate … sub-millisecond" | fixed wording; $-1$; number removed; "search is exact at the memory sizes reached; approximate indexes only at much larger sizes" | 0.98 unsourced; memory 12 175 < 20 000 IVF-PQ promotion (`caem/config.py`) |
| L54–60 (self-consistency) | grammar; "s_avg > 0.85 indicates robust reasoning"; "older literature writes it u_SC"; semantic entropy retired "because its Pearson correlation with EM was indistinguishable from zero" (and elsewhere "no discriminative information") | grammar fixed; threshold and u_SC remark removed; semantic entropy dropped because its learned weight in the cycle-zero composite fit was ≈ −0.0005 and its ten extra samples were the verifier's largest cost (≈ 4 s/answer); KLE not adopted for the same cost reason; new label `sec:lit-self-consistency` | `caem/config.py:195–200` (2026-04-27: boost weight −5e-4 on the n = 1 500 ten-signal fit; K = 10 samples ≈ 4 s/sample, ≈ 40% of the calibration step). The "Pearson = 0" reading was measured after the signal had already been disabled and replaced by a constant 0.5 (archive/branch_C_log_historical.md:5028–5032) |
| L70 (EWC prelim) | λ = 0 story with "zero-anchored full-backbone reference"; "three-mechanism combination" listing the adapter twice | uniform-penalty variant explained; CAEM uses neither penalty; three mechanisms = training-pool filter, LoRA on frozen backbone, retention guard | matches ch1 L397; code `caem/config.py:670` |
| L86, L88 | "has major result"; FActScore numbers in words; "CAEM adopts the FActScore convention at the storage gate" | grammar; digits; atomic signal described with realised note: coincided with the passage-mean signal on every evaluated answer | Q1 decision (p_ground^atomic = p_ground^mean on 14 951/14 951 samples) |
| L92 | "Google's FACTS benchmark (2025) … 6.4% … Vectara Hallucination Leaderboard … 3% … 27.2%", broken grammar | removed | no citation in the bibliography or in the reference audit; unverifiable |
| L94 | "rates ranging from 3% to over 90%"; "motivating CAEM's commitment to fact-level NLI-based verification" | rates "from 28.6% to 91.4% of generated references in a single study"; "evidence-based grounding signals" | 3% came from the removed Vectara figure; atomic step did not fire in the realised run |
| L100 | GSM8K "about eighteen … about fifty-seven percent"; "Chapter~5" | 17.9% → 56.9%; \Cref{ch:evaluation} | Wei et al. Table: 17.9 → 56.9 (audit F77) |
| L102 | CoT "gives explicit reasoning chains that enables step by step verification through NLI which also checks whether each reasoning step is supported" | CoT is the answer format and source of resampled chains; the verifier scores only the final answer, so an invalid step that reaches a correct conclusion goes undetected | contradicted ch2's own exclusion list ("only the final answer is verified"); memory: verifier blind to step-level errors |
| L106 | grammar ("It chooses reasoning chain agreement", "differin") | rewritten | grammar |
| L108–112 | semantic entropy "serves as the third verification layer for CAEM … will generate 10 diverse samples" (Canberra example); "motivating its inclusion in CAEM"; "0.698 verbalized probability" | semantic entropy reviewed as literature; averages over 30 task–model pairs: 0.790 vs naive entropy 0.691, P(True) 0.698, embedding regression 0.687; dropped as a signal, kept as baseline B8 | stale (signal removed 2026-05-09); numbers verified against the paper (PMC11186750): 0.698 is P(True), not verbalised probability |
| L114 | FEVER labels "would map directly to CAEM's storage decisions" | mapping removed | storage depends on whether the generator's answer is correct and supported, not on the gold label |
| L116 | "has achieved"; "invalid reference" | grammar; "invalid inference" | grammar |
| L118 | Mohri–Hashimoto "the closest published method that explicitly crosses an eighty-percent factuality target"; CAEM rejection with "fifteen to fifty percentage points" | "raises FActScore factuality from about 30% to 80% while retaining most sub-claims"; rejection with pointer to sec:conformal-gate, no number | Mohri et al. abstract ("30% → 80% on FactScore"); 15–50 pp unsourced (carried item from ch1) |
| L120 | 400-word methodology paragraph ("Branch~C's", "removed on 2026-04-22", boost "in the lineage of Cherian-Gibbs-Candès") | 150-word synthesis: families, isotonic + logistic combination "in the manner of" Valentin et al.; full spec in ch4 | duplicate of ch4; project jargon; ch1 Obj 3 and ch2 L118 name Valentin as the precedent |
| L122 | "complementary strand of architectural literature" on prompt-design fixes (no citation), "deltas reported in Chapter 5" | removed | CAEM design content, not literature; the tables live in appendix B |
| L128–130 | grammar ("dropout usually during training", "It describes as performing", "group training") | rewritten | grammar |
| L138 | "CAEM uses temperature scaling … once on the pre-routing u_pre and once on the post-generation composite u_stored" | temperature scaling on u_pre only; u_stored calibrated by per-signal isotonic regression + logistic combination | `caem/confidence/pre_routing.py:162` (temperature), `caem/verification/cal_prob_composite.py` (isotonic + LR) |
| L140, L142, L144 | "First is P(True) asks"; grammar; "provided comprehensive approaches into white box" | rewritten | grammar |
| L146 | "encoder-side convergence ratio over the final attention layers"; repeated full verifier description; "Branch~C's" | two-stage design in brief; "layer-convergence ratio over the query's hidden states" | decoder-only backbone (ch1 carried item) |
| L150 | "alternatives we considered"; KLE "empirical AUROC gain is bounded above the cost"; semantic-entropy reason repeated | "considered for CAEM"; KLE not adopted because it needs the same extra samples | first person; garbled; one canonical reason |
| L152 | "pre-RLHF models or instruction-tuned checkpoints such as Qwen-2.5-3B-Instruct are expected to give more reliable confidence signals"; "fitted … with no live label information at deployment time" | four findings; the backbone is itself preference-tuned, so no raw confidence is used unrecalibrated; both calibrations refitted on labelled data every cycle | contradicted the old summary (Qwen post-training uses DPO + GRPO); ch1 L72 / production runbook use labelled folds each cycle |
| L156, L158, L160 | grammar ("It  helps", "NTMs ventured into the idea", "The importance of CAEM is that it shows", "without any delay") | rewritten | grammar; "without any delay" overclaim |
| L172 | "low tens of thousands of stored episodes"; "keeping search time below one tenth of a second is important" | "$12\,175$ at the end of the realised trajectory" | tab_memory_store_trajectory; latency claim unsourced |
| L176 | Adaptive-RAG described in full (also at L189) | one sentence pointing to sec:lit-selective-retrieval | redundancy |
| L178 | safety override "forces Tier 3 whenever u_pre falls below the floor"; "high-similarity matches … still route to Tier~3" | override sends the query to the retrieval-augmented default, after which the retrieval-utility classifier chooses retrieval or zero-shot; a low-confidence close match is "not served from memory" | `caem/routing/router.py`: safety branch calls the RUC with default tier 3 |
| L185 | PopQA "forty-six and a half percent"; "does not require a heavy language-model forward pass at routing time, the design constraint that sec:ruc adopts" | 46.5%; popularity is one of the classifier's inputs | the RUC's p_IK feature reads a generator forward pass (ch4 L163), so the old claim was false |
| L189 | "Moskvoretskii et al."; "only language-model dependency being an offline self-knowledge probe that is fit once at cycle zero and consumed as a fixed feature" | "Marina et al."; classifier is a logistic head whose features reuse pipeline quantities, including a self-knowledge probe from one forward pass of the base generator | reference_check_1 #53 (first author Marina); ch4 L163 |
| L193 | "consumes the last-token hidden state under the disable-adapter context" | "reads the base generator's last-token hidden state, without the adapter" | code-like wording |
| L195 | "empirical ceiling … AUC 0.80–0.86 in distribution and 0.65–0.75 out of distribution … SeaKR … No published system convincingly exceeds 0.90"; "0.85 on a 3 208-row held-out fold" | "pooled out-of-fold AUROC 0.850 under nested cross-validation on 3 208 labelled rows (sec:ruc); not directly comparable with published systems" | ceiling figures had no source (SeaKR not cited); ch4 L171 |
| L202–208 | second copy of the EWC equation and explanation | one sentence pointing to the Preliminaries | redundancy |
| L210 | "EWC maintained approximately 90% of single-task performance when sequentially learning 10 games"; λ = 0 story again; "sufficient to meet the retention target on the empirical evidence reported in Chapter 5" | Atari result in the paper's terms; pointer to Preliminaries | Kirkpatrick et al.: EWC learns many games without catastrophic forgetting but below ten separate agents; no 90% figure. Retention guard fired at C4 and CommonsenseQA/TruthfulQA show forgetting, so "sufficient" overclaimed |
| L212 | "permits CAEM to fit ten cycles … the agent-research audit accompanying this thesis registers full fine-tuning as the rejected alternative" | "lets the self-improvement loop fit on a single consumer-grade graphics processor; full fine-tuning would not fit" | the trajectory ran six cycles; the "audit" reference is unexplained |
| L214 | 330-word paragraph on SIL pool reweighting, 90/10 mix history, retention-guard mechanics | removed | duplicate of ch4 (sec:sil, sec:retention) |
| L218 | Natarajan "proves that training on pools whose purity is well below 100% can still improve the model … justifies training on the resulting pool at all" | guarantees assume a noise-corrected loss CAEM does not use (pointer to thm:bayes-purity); CAEM lowers the noise rate with τ_train > τ_store | matches ch4's interpretation paragraph and the old summary's own correction |
| L222 | "+35.9% improvement over few-shot baseline" | 35.9 percentage points (72.5% vs 36.6%) | Zelikman et al. Table/abstract |
| L224, L226, L228 | grammar; external verification "through an NLI cross-encoder, multi-chain self-consistency, and semantic-entropy analysis"; RAG "provides exactly the same external feedback" | rewritten without semantic entropy; RAG "supplies feedback of this kind" | stale signal; overclaim |
| L230 | "unlabeled/labeled/pretraining" | British spelling | style |
| L234 | V-STaR extended "by adding the $L_{2}$ anchor and the episodic memory" | "by adding the episodic memory and the retention guard" | λ_anc = 0 on the shipped path |
| L236 | synthesis repeating the λ = 0 story | short synthesis | redundancy |
| tab:positioning caption | "the joint calibration-consistent precision floor and convergence guarantees … defensible novelty axis" | "the calibration-consistent precision floor with its purification condition … distinguishing combination" | convergence is a heuristic (ch4) |
| L263 | "Chapter~5 instantiates one representative from each family" (three families) | representatives of three families plus a confidence-calibration method | B1–B8 include several per family and B8 |
| L265 | B5 demos "from the TriviaQA train split" | "from a benchmark training split" | `scripts/run_baseline.py:296–310`: demos come from the first benchmark of the launch list (FEVER when listed first); exact source carried to ch5 |
| L269 | B6 "trains … on the verified episode pool … without the bounded low-rank-adapter envelope … every selected episode is treated as a clean label"; STaR "run only as an optional ceiling conditional on remaining compute budget" | B6 fine-tunes all weights on raw question–answer pairs from the training benchmarks, no verifier, no retention guard, no memory; STaR is not run | `scripts/run_simple_ft.py` docstring ("Raw (question, answer) pairs … No verifier filter"); `scripts/launch_baselines.sh` (`n_train_per_bench 700`, 5 cycles); STaR never run |
| L271 | STaR "reported separately as an optional ceiling when compute permits"; "which Chapter 5's retrieval-utility-classifier on/off ablation isolates" | neither Self-RAG nor STaR is run; the combination is tested by the baseline comparison | stale; the RUC ablation does not isolate the four-family gap |
| L277 (FEVER) | SUPPORTS → Store, REFUTES → Discard, NOT ENOUGH INFO → Abstain | removed | conceptual error (see L114) |
| L283 | "(Chapter~5)" | \Cref{ch:evaluation} | cross-reference style |
| L289 | FEVER "mapping from NOT ENOUGH INFO to the Abstain outcome"; TruthfulQA "exercises the semantic-entropy proxy"; "success across the panel demonstrates that CAEM's improvements generalise" | NOT ENOUGH INFO as recognising missing evidence; TruthfulQA tests agreement-based and relevance signals; the panel "shows whether CAEM's effects carry across reasoning types" | stale signal; overclaim (TruthfulQA and CommonsenseQA show forgetting, tab_continual_retention) |
| sec Summary of Key Findings | ≈ 2 000 words re-telling the chapter and the methodology, incl. "LoRA primary path with the $L_2$ anchor placed on the adapter parameters", the wrong FEVER mapping, "CHM … aggregated into a geometric Composite Evaluation Score", "nine-by-five Spearman matrix (scheduled in Chapter 5 as a Phase 1 Full deliverable)", "legacy backend retained in Chapter 5's verifier-calibration diagnostic", "Branch~C's", "2026-04-22" | ≈ 520 words: five findings, each tied to the CAEM design decision it motivates | template section kept; stale and false statements removed (λ_anc = 0; CHM and CES are separate metrics in ch5 L68; no Spearman matrix or RoBERTa diagnostic exists) |
| L6, L76; ch1 signpost | "six areas" | the eight areas actually reviewed | section intros must match their sections |
| L97 | verification intro: four approaches "that together supply the signals aggregated inside CAEM's composite verifier" (incl. semantic entropy) | five approaches "from which CAEM's verifier draws its signals or against which its design is positioned" | semantic entropy no longer supplies a signal |
| ch4 L285, appendix G L70 | semantic entropy retired on "a Pearson correlation indistinguishable from zero" (ch4) / "per-signal Cohen's d audit found it dominated by chain-to-answer entailment" (app G) | learned weight ≈ −0.0005 in the ten-signal cycle-zero fit; ten extra samples the largest verifier cost | same evidence as L60; one reason everywhere |
| ch2 throughout | American spellings (characterization, summarizing, optimized, rationalization, …) | British | thesis spelling convention |

Build (latexmk, clean): 268 pages, 0 errors, 0 undefined refs/citations, 0 multiply-defined labels, 0 biber warnings, overfull 89.

### Chapter 2, part 2: author read-through (2026-10-16)

- Citation style (author rule; see SESSION_START "Standing author decisions"). Bracket-as-noun fixed: confabulations "in the
  sense of Farquhar et al.", Sinha ("report" → "reports", single author), semantic entropy of Farquhar et al., kernel language
  entropy of Nikitin et al., LoRA layer of Hu et al., Niculescu-Mizil and Caruana, Geifman and El-Yaniv, Valentin et al.,
  Kalajdzievski. Studies named where the sentence relies on their finding: Kuhn et al. (bidirectional entailment), McCloskey
  and Cohen, Biderman et al. (erosion grows with training), Chelli et al. (28.6–91.4%), Farquhar et al. (misconceptions not
  covered). Citation moved next to the name for Kirkpatrick et al.; "Hu/Biderman/Dettmers/Wang and colleagues" → "et al.";
  "sixteen to two hundred and fifty-six" → "$16$ to $256$". LoRA acronym introduced at first use (Preliminaries).
- Kept bare: general background (Transformer, NLI definition, survey taxonomies, FAISS), named systems (Self-consistency,
  MiniCheck, DPR-RAG, FLARE, Adaptive-RAG), and the Summary of Key Findings recap (studies already named in the Review).

## Chapter 3 (branch `review/ch3`, from `branch/ruc-classifier` @ ecb2957)

Words 6,437 → 4,861. Every claim checked against code, run logs, outputs and appendices.

**Factual corrections**
- NFR2 / cost-benefit / ethics: removed "π_store = 0.796 over 289 stored entries … under the locked composite", "near 19%",
  and "τ 0.65 admits a factor of four fewer at the same precision". Sweep re-run (`outputs/cycle_0/sweep_v2_1/README.md`):
  289 @ 0.796 is the legacy weighted-sum bootstrap composite; the locked configuration stores 558 @ 0.751 (validation fold);
  on the calibration fold τ 0.60 → 560 @ 0.780 vs τ 0.65 → 281 @ 0.865 (half as many, higher precision). Ch3 is now
  number-free on the sweep ("chosen on the cycle-zero calibration fold before the main run", matching ch1 D6); the Q4
  numbers stay an open author decision for ch5.
- Retroactive pass: "prunes below the storage threshold" → re-scores up or down and removes entries below 0.50
  (`retroverify_prune_threshold = 0.50`, `retroverify_allow_downgrade = True`).
- Software standards rewritten: there is no pinned requirements file (appendix E says so), no structural pattern matching
  in the code, no 88-character configuration, and only two Git tags (not a tag per cycle). Now: PEP 8 broadly, versions in
  app:repro-versions, single seed with non-bitwise-deterministic attention kernels, one branch per phase, per-cycle archives.
- Undelivered appendix promises removed: per-phase carbon table, licence catalogue with access dates and version hashes,
  milestone planned/actual dates, rented-credit envelope (none exists in appendix E). Downgraded 16 GB envelope kept
  (app:repro-hardware exists).
- "Public dataset host" snapshot → the Hugging Face repository is private (on request); Drive folder is the shared archive.
- Environmental footprint: added main-run wall-clock from `outputs/full_run/experiment.log` ("Cycle N done in … min":
  361.0, 1742.9, 2125.1, 2590.0, 1457.1, 3278.9 → 6.0/29.0/35.4/43.2/24.3/54.6 h, ≈193 h) and an upper bound on GPU energy
  at the RTX 5090's rated 575 W (≈111 kWh); no carbon figure because the host grid factor is unknown.
- Cost-benefit: "ΔC_store ≤ 0" was false (every stored entry is re-verified each later cycle: ≈38 min at C1 for 431 entries,
  ≈16 h at C5 for 10,214). Equation reduced to the benefit side; benefit tied to repeated questions (Tier 1 probe 42/50,
  19/50; 7 of 9,000 on the evaluation fold).
- Deployment projection: "headline economic claim" → design intent; realised tier shares (tab_tier_trajectory pooled):
  Tier 1 ≤ 0.1%, Tier 2 and Tier 3 each 45–55% across cycles, so neither moved as designed.
- Cycle-budget rationale: the ten-cycle cap no longer "is the point the convergence analysis predicts"; thm:convergence named as
  a heuristic; Song et al. cited for saturation.
- "Rehearsal buffer" removed (no such component); "at the chapter-one level … quantified in sec:economic" removed
  (wrong chapter; not quantified); unsupported "shortens the time to convergence" removed.
- Milestones: the first calibration iteration failed on the signal-dump bug (appendix B), not on the judge correlation floor
  (that was the separate judge ablation); main-run completion date 27 May 2026 (`run_complete.json`).
- Societal claims softened: "strict improvement" → "preferable"; "removes the most dangerous failure mode" → "reduces";
  "honest abstention rather than confident harm" → mechanisms reduce but do not remove the risk.
- Risk register: stale "Pending main run" entries resolved — calibration drift Partly (cal-fold precision 0.732–0.746;
  cal-eval gap in ch5); new Forgetting row (C4 abort, TriviaQA-test 0.886); credit exhaustion Yes once (commit 88dcc3a,
  2026-05-17, earlier run, resumed); node reclamation and storage exhaustion Not observed; early-stop row neutral pending
  author decision (see progress file). Caption no longer "as of the cycle-zero validation gate".

**Structure and style**
- Functional and non-functional paragraphs no longer repeat their tables item by item; bold lead-ins for the NFRs, the
  dual-use mechanisms and the evaluation-protocol standards; itemised lists for the design commitments, reproducibility
  artefacts, phases, deliverables and risks; risk paragraphs merged with the register (mitigation text no longer
  duplicated three times).
- Citations: Kamath et al. (bracket-as-noun fixed), Alkaissi and McFarlane, Chelli et al., Kadavath et al., Song et al.,
  Edwards. Header comment "Status: SKELETON" removed.

### Chapter 3, part 2: author decisions (2026-10-16)

- Lead-in punctuation: every bold opening now ends in a colon instead of a full stop (ch3: 36; ch1: 6, the three
  shortcomings and the three prior-pipeline families). Author: the full-stop form looks machine-written.
- Early stop disclosed (author approved): sec:env-cycles now states that the run stopped after cycle five as planned after the
  cycle-four abort, and that the automatic check fired at the same point on two mis-wired inputs (evaluation-fold storage rate,
  always 0, instead of the stream admission rate ≈ half; raw MMLU accuracy ≈0.61 against the 0.93 ratio floor instead of the
  ratio, 0.94–1.00), so its firing is not evidence of saturation. Risk register row: Yes, with that reading.
- Tier 1: cost-benefit and deployment sections now say why it never fired: each cycle drew a fresh stream chunk sharing no
  question with earlier chunks, and the evaluation fold is disjoint from both (disjoint per-cycle chunks built in
  `scripts/run_experiment.py`, "Build disjoint 6-way pool splits").

### Chapter 3, part 3: author read-through (2026-10-16)

- RUC ablation is not a separate run: tab_ruc_ablation compares Phase 1d (classifier off; Drive `3_ruc_ablation_phase1d/`,
  `run_complete.json`: committed cycles 1–3, cycles 4–5 aborted) with the main run at C3. Compute phases are now three; the
  ablation "needed no run of its own"; econ-budget explains why C3 (last cycle both trajectories committed).
- Earlier runs added: three earlier trajectories (first benchmark panel = Drive `9_archive/v1_panel/`; run stopped May 2026 =
  `9_archive/phase1c_failed_2026-05-09/`; Phase 1d). Phase 1d wall clock from its run log (Drive
  `3_ruc_ablation_phase1d/full_run/experiment.log`, "Cycle N done in": 312.5, 1506.3, 1874.1, 2053.9, 1149.7, 1150.3 min →
  ≈134 h). Logs of the first two not archived; "several times the compute of the main run" per author's account (3–4 earlier
  runs of similar cost). Author to confirm the count.
- Digits: "thirty-two-gigabyte" → "$32$-gigabyte", "sixteen-gigabyte" → "$16$-gigabyte" (matches "$64$-gigabyte").
- Strict training threshold stated: 0.70 against 0.60 (`min_u_stored_for_training = 0.70`).
- Abstention checked against `caem/verification/verifier.py` L64–70: ABSTAIN = u_stored < 0.45 and p_ground_max < 0.20;
  DISCARD = u_stored < 0.45 with p_ground_max ≥ 0.20 (answer shown with a "conflicting" warning); STORE and DEFERRED shown
  with tags (`caem/production/confidence.py`). Ch1/ch2 wording already matches; ch3 marginalised-user sentence tightened.
- Live demonstration removed (not planned; 15-minute defence); deliverables now five, lead-ins with colons.
- Early-stop: code fixed (`scripts/run_experiment.py`: storage signal now reads the Step 4 stream-pass admission rate,
  sample-weighted; retention signal receives the ratio current/pristine). Replay
  (`scripts/replay_early_stop_gate.py` → `outputs/analysis/early_stop_gate_replay.json`): realised wiring fires
  storage_saturation + mmlu_floor (stop); corrected wiring fires none (no stop) at C5. Ch3 sec:env-cycles, risk item and
  register row now say the wiring was corrected after the run and what the replay shows.
- Storage exhaustion: realised (author's account): the rented node's disk filled; data copied node to node with the
  provider's support; run resumed from its last checkpoint. Register row Yes.

### Chapter 3, part 4: closing checks (2026-10-16)

- Earlier-run count confirmed by the author as three.
- Ablation sentence notes that the main run continued to cycle five and its other results cover C0–C5.
- Early-stop fix re-checked: gate docstring asks for the retention ratio against the guard's 0.93 floor; the eval-pass
  `storage_rate` in `outputs/full_run/eval/<bench>_cycle<N>.json` is 0.0 for every benchmark and cycle; the stream-pass files
  `<bench>_cycle<N>_streamchunk.json` record actual memory insertions (`stored_flag` is True only after the novelty filter and
  `memory_store.add`): 0.520, 0.527, 0.526, 0.529, 0.499 for C1–C5, matching appendix C. Replay script now reads these
  admission rates; result unchanged (corrected wiring fires no signal at C5). `tests/test_equilibrium.py`: 23 passed.
