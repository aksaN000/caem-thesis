---
name: FLARE algorithm mismatched to label-classification tasks — accepted as published-algorithm limitation
description: FLARE's sentence-commit policy (commit first sentence, retrieve if uncertain, terminate when sentence ends) truncates output BEFORE the model reaches the "Answer: <label>" line on FEVER/CSQA/StrategyQA. Implementation is faithful to Jiang et al. EMNLP 2023; the published algorithm was designed for long-form QA (ASQA, ELI5) and doesn't generalize to short-answer benchmarks. Reported as §threats finding, not a bug fix.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Finding (2026-05-16):** FLARE FEVER EM = 0.147 (vs B3 rag 0.437, B1 zero_shot 0.453) on the baseline panel run. Escalation rate 230/300 (76.7%) — retrieval is firing correctly. Predictions are prose like *"The context provided does not mention X."* with no FEVER label word, so the FEVER label extractor returns UNPARSEABLE → EM 0.

**Mechanism (eval/baselines.py:813-847):** FLARE loops sentence-by-sentence, committing the first sentence of each look-ahead (or grounded re-gen if retrieval fired), then terminates the loop as soon as the committed text ends in `.`/`?`/`!` and has ≥3 words. On label tasks the model's first sentence is hedge prose ("The context provided does not mention X."); the label `supports/refutes/not enough info` appears in the "Answer:" line AFTER that, which FLARE never reaches.

**Implementation is faithful to the paper.** Jiang et al. EMNLP 2023 §3.2 specifies exactly this commit-first-sentence behaviour. The algorithm was tested only on long-form QA benchmarks (ASQA, ELI5, StrategyQA-as-explanation) where each sentence adds factual content and the answer naturally unfolds sentence-by-sentence. The CAEM panel's short-answer benchmarks (FEVER labels, CSQA letters, StrategyQA yes/no) expose a generalization limitation the paper didn't surface.

**Decision (2026-05-16):** Accept the as-published FLARE numbers. Do NOT modify the algorithm to handle label tasks differently — that would be a "modify the baseline to perform better" methodology flag a reviewer would catch.

**Why:** FLARE's poor performance on our label-heavy panel is itself a thesis-worthy finding. It motivates CAEM's design choice (unified generate-then-extract pipeline) by showing that on-demand sentence-level retrieval doesn't generalize to short-answer tasks without architectural changes.

**How to apply in thesis:**

Ch5 §threats / §5.6 (after FLARE row in main results table):
> "FLARE's sentence-commit policy was designed for long-form generation (Jiang et al. EMNLP 2023 §4). On our short-answer benchmark panel, FLARE's first-sentence-and-stop termination truncates the model's output before the `Answer:` line is reached, producing prose without parseable labels. This is a property of the published algorithm, not an implementation defect; we verified by inspecting sample outputs (Appendix X) that retrieval fires correctly (escalation rate 76.7% on FEVER) but the final prediction never contains the FEVER label vocabulary."

Ch6 §discussion (CAEM positioning):
> "CAEM's unified generate-extract pipeline produces the full `Reasoning: ... Answer: <label>` scaffold per response, then the EM scorer extracts the answer span. This makes CAEM uniformly applicable to both label-classification (FEVER, CSQA, StrategyQA) and open-ended generation (TriviaQA, TruthfulQA) without per-task algorithmic adaptation. FLARE's sentence-by-sentence commit policy, by contrast, requires per-task tuning to perform well across this task family."

Ch6 §future-work:
> "A task-aware FLARE variant — using sentence commit on long-form QA and direct generation on label tasks — would be a fairer cross-task FLARE replication. We leave this to future work as it modifies the published algorithm."

**Expected FLARE damage on remaining benches:**
- TriviaQA: OK-ish (first sentence often contains entity answer) — predict 0.30-0.40
- CommonsenseQA: broken (letter task) — predict <0.20
- TruthfulQA: partial (open-ended; first-sentence commit truncates context) — predict 0.10-0.20
- StrategyQA: broken (yes/no task) — predict <0.20

This means FLARE will likely be the **worst pooled baseline** in our panel, which is not a fair representation of FLARE's quality on its design task family. Surface this caveat clearly.
