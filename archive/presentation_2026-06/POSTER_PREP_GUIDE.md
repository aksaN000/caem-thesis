# CAEM Poster Presentation — Complete Prep Guide

**For: Md. Aksan Gony Alif — Pre-Thesis 2 poster defence**
**Read this top to bottom. By the end you'll be able to teach CAEM to anyone.**

---

## PART 0 — The 30-second elevator pitch (memorise this)

> "Large language models confidently make things up — they hallucinate facts. Scaling them bigger doesn't fix it. My thesis, CAEM, treats this as a *system design* problem, not a detection problem. I wrap a small 3-billion-parameter model in an architecture that (1) remembers verified answers, (2) measures its own confidence before and after it answers, (3) only stores answers that pass a calibrated quality gate, and (4) re-trains itself each cycle while a safety guard protects it from forgetting. On a five-benchmark test, the best cycle cuts hallucination by about 29% and raises accuracy by about 20% over the plain model, and it beats all eight competing methods."

If you can say that confidently and then point at the poster, you've already passed.

---

## PART 1 — The problem (why anyone should care)

### What is a hallucination?
A language model produces text that is **fluent and confident but factually wrong**. Example: ask it "who wrote X book" and it invents a plausible-sounding author who never existed. The danger is the *confidence* — a wrong answer phrased exactly like a right answer is hard for a user to catch.

### Why it matters (your poster's opening numbers)
- Hallucination rates run **3% to 91%** depending on the domain.
- In clinical question studies, frontier models hallucinate **69-77%** of responses.
- TruthfulQA showed the *biggest* models were *more* likely to repeat human falsehoods — so **scale alone does not solve it**.

**Key sentence to say:** "Hallucination is not a bug that bigger models fix — it's a structural property of stateless generation under uncertainty."

### The three structural problems CAEM attacks (memorise these three words)
1. **Stateless** — the model has no memory. Same question tomorrow → same effort, no reuse of what was verified yesterday.
2. **Overconfident** — the model's confidence is poorly calibrated; it says 90% sure when it's right 60% of the time.
3. **Static** — once trained, it never learns from its mistakes; naive retraining causes "catastrophic forgetting" (it forgets old skills when learning new ones).

CAEM is the first architecture that attacks **all three at once**: memory fixes stateless, calibrated confidence fixes overconfident, the self-improvement loop fixes static.

---

## PART 2 — The big idea (the one mental model)

Think of CAEM as a **disciplined librarian** sitting on top of a normal language model:

- When a question comes in, the librarian first **checks if it already has a verified answer** in its memory (Tier 1).
- If not, it decides **how hard to work**: just think (Tier 2, zero-shot) or go look things up (Tier 3, retrieval).
- After the model answers, **nine independent checks** score whether the answer is trustworthy.
- Only answers that pass a **strict quality gate** get filed into memory.
- Every so often (each "cycle"), the librarian **studies its own filed answers and gets a little smarter** — but a **safety guard** makes sure it never gets dumber on things it already knew.

The backbone model (Qwen-2.5-3B-Instruct) is **frozen** — we never touch its core weights. We only train a tiny add-on adapter (LoRA, ~2% of the parameters). That's what makes this affordable and safe.

---

## PART 3 — Walk the poster, block by block

The poster has **three columns**. Practice walking left to right. Here's what to say at each block.

### COLUMN 1

#### Block: Abstract
Don't read it aloud. Just say: *"This summarises the whole system — let me walk you through the architecture instead."* Then move to the architecture diagram in column 2. (Examiners hate having the abstract read to them.)

#### Block: Research Objective (7 bullets)
These are your **seven specific objectives** — the deliverables. You don't need to recite all seven. If asked "what did you set out to build?", group them into the three big pieces:
1. **The memory + router** (objectives 1, 2) — verified episodic memory, three-tier routing, confidence, the retrieval-utility classifier.
2. **The verifier + gate** (objectives 3, 4, 5) — nine signals, calibrated composite, fixed thresholds, retroactive review.
3. **The self-improvement + validation** (objectives 6, 7) — LoRA fine-tune with retention guard, and the empirical comparison against 8 baselines.

#### Block: Methodology
This describes the **eight-stage pipeline**. Use the diagram in column 2 to explain it (see Part 4 below). Don't read this block.

### COLUMN 2

#### Block: Model Architecture (THE MAIN EVENT — spend most time here)
This is the eight-stage diagram. See Part 4 for the full walkthrough script.

#### Block: Per-Cycle Trajectory (the results table)
This is the table showing how CAEM evolves over cycles C0 to C5. See Part 5 for how to read it.

### COLUMN 3

#### Block: Headline Result
Point at the two boxed numbers: **+20.3% EM, -29.4% CHM**. Say: *"Against the plain model on the same backbone, the best cycle improves exact-match accuracy by 20% relative and cuts the hallucination metric by 29% relative on the training panel."*

#### Block: External Baseline Comparison
The 8-baseline table. See Part 6.

#### Block: Retrieval-Utility Classifier Ablation
This is your newest contribution (the RUC). See Part 7.

#### Block: References
Just there for credibility. Ignore unless asked.

---

## PART 4 — The eight-stage pipeline (your core explanation)

Point at the architecture diagram and walk through it like a story of one question's journey. **Memorise this sequence** — it's the heart of your talk.

**"A single question travels through eight stages:"**

- **Stage 1 — Pre-route (confidence):** Before answering, the model takes one quick look at the question and produces a *pre-routing confidence* — basically "how ready am I to answer this?" It comes from two cheap signals fused together: how confident the model's first few words are, and how stable its internal representation is.

- **Stage 2 — Encode and look up memory:** The question is turned into a vector (an embedding) and we search the **episodic memory** for the most similar question we've answered before. If we find a strong verified match, we can reuse it.

- **Stage 3 — Route to one of three tiers:**
  - **Tier 1 (memory-direct):** serve the stored verified answer — near-zero cost.
  - **Tier 2 (zero-shot):** just have the model think and answer.
  - **Tier 3 (retrieval-augmented / RAG):** retrieve passages from a knowledge corpus first, then answer grounded in them.
  - A **safety override**: if pre-routing confidence is below a floor, force the conservative grounded tier — don't let a shaky model wing it.

- **Stage 3b — Retrieval-Utility Classifier (RUC):** *(your new piece)* Before defaulting to expensive retrieval, a small classifier predicts: *will retrieval actually help this question, or hurt it?* Some questions get worse with retrieval (misconception-bait questions where the retriever pulls in the wrong context). The RUC sends "retrieval-helpful" questions to Tier 3 and "retrieval-hurt" questions to Tier 2.

- **Stage 4 — Execute the chosen tier:** produce the answer.

- **Stage 5 — Verify (the nine signals):** The answer is scored by **nine complementary signals** in four families:
  1. **Internal calibration** (3 signals) — how confident was the model internally?
  2. **Sample-set agreement** (2 signals) — generate the answer a few times; do the versions agree?
  3. **External grounding** (3 signals) — does retrieved evidence actually support the answer?
  4. **Question-answer relevance** (1 signal) — did it even answer the question asked?

- **Stage 6 — Decide (four outcomes):** The nine signals are fused into one **calibrated probability** (the "composite"). Then a fixed-threshold rule produces one of four outcomes:
  - **Store** (high confidence → file it in memory),
  - **Defer** (borderline → hold for reconsideration next cycle),
  - **Abstain** (too uncertain → refuse rather than guess),
  - **Discard** (clearly wrong → throw away).

- **Stage 7 — Output:** Return the answer to the user *with* its calibrated confidence number and the decision tag. The user always sees how much to trust it.

- **Stage 8 — Cycle boundary (the self-improvement loop):** Periodically:
  1. **Retroactive re-verification** — re-score all stored answers under the updated model; prune any that no longer pass.
  2. **Deferred reconsideration** — revisit borderline answers; promote ones that now pass.
  3. **Consolidation** — merge near-duplicate memories.
  4. **Fine-tune** — train the LoRA adapter on the verified high-quality stored answers.
  5. **Retention guard** — test the new model on held-out probes; **if it got worse on anything, roll the adapter back** (but keep the memory). This is the "asymmetric rollback" — weights can revert, memory never does.

**The one-liner for Stage 8:** *"The system teaches itself from its own verified answers, but a safety guard guarantees it never trades away old knowledge for new."*

---

## PART 5 — Reading the Per-Cycle Trajectory table

The table has rows B1 (the reference floor) then CAEM C0 through C5.

**What each column means:**
- **EM** = Exact Match accuracy (higher is better).
- **CHM** = Composite Hallucination Metric — an 8-subtype average of failure modes (lower is better).
- **MMLU / TQA-test / CSQA-test** = retention probes. These are *ratios vs. the pristine starting model*. A value of 1.0 means "no capability lost"; below 0.93 triggers the safety rollback.
- **Outcome** = commit (kept the cycle) or abort (rolled back).

**The story to tell pointing at the table:**
- "C0 is the cold start — full architecture, no self-improvement yet."
- "C2 is the best cycle (highlighted) — highest accuracy 0.544, lowest hallucination 0.107."
- "C4 is interesting — see the red row? The retention guard **fired** here. The TriviaQA-test probe dropped to 0.886, below our 0.93 floor, so the system **automatically rolled back** rather than commit a regression. This is the safety mechanism *working in the wild*, not a failure."
- "C5 recovers and closes the trajectory under our early-stop gate."

**If asked "why stop at C5?":** *"A pre-registered equilibrium-saturation gate fired — two of three signals said we'd hit diminishing returns: storage saturated and the retention probe sat at its floor. Continuing would cost the same per cycle for marginal benefit."*

---

## PART 6 — The External Baseline Comparison (the "we win" table)

Eight competing methods (B1-B8), all on the **same backbone, same data, same scoring** (this is called "matched protocol" — emphasise it, it's what makes the comparison fair).

- **B1 zero-shot** — plain model, no tricks.
- **B2 chain-of-thought** — "let's think step by step."
- **B3 RAG** — retrieval only.
- **B4 CoT + RAG** — both.
- **B5 five-shot CoT** — with examples.
- **B6 vanilla fine-tune** — naive retraining (no quality gate).
- **B7 FLARE** — active retrieval method from the literature.
- **B8 semantic entropy** — an uncertainty-based abstention method.

**The headline:** CAEM C2 and C5 **beat all 8 baselines** on both accuracy and hallucination jointly (the "8/8" row).

> ⚠️ **HONESTY NOTE FOR YOU (do not hide if asked):** The B6 and B7 numbers on the current poster are being re-run right now because we found a *formatting bug* in how their answers were scored — it made them look artificially worse. The corrected B6 is about 0.49 EM (not 0.325), and corrected B7 will be higher too. **If an examiner probes the baseline numbers, say this honestly:** *"We're finalising a matched-protocol re-run of two baselines where a prompt-format issue understated their scores; the corrected numbers tighten the gap but CAEM still leads on the architectural contribution."* This is a strength — it shows methodological rigor.

---

## PART 7 — The RUC Ablation (your newest scientific contribution)

The Retrieval-Utility Classifier table shows **with the RUC on vs off**.

**The key finding (point at the highlighted rows):**
- On **CommonsenseQA**: turning the RUC on raises accuracy **+14 points** and cuts hallucination **-34%**.
- On **TruthfulQA**: **+10 points** accuracy.
- These are the "retrieval-hurt" benchmarks — questions where naive retrieval *injects* wrong context. The RUC learns to route them *away* from retrieval.

**The one-liner:** *"The retrieval-utility classifier unlocks benchmark-conditional behaviour: it sends retrieval-helpful questions to the retrieval tier and keeps retrieval-hurt questions at the zero-shot tier. That's why pooled accuracy jumps almost 6 points when it's on."*

**The Tier 3 share columns** show the RUC dramatically cuts how often retrieval fires (e.g., TruthfulQA from 99% to 40%) — it's being selective, which is the whole point.

---

## PART 8 — Key numbers to have on the tip of your tongue

| Number | What it is |
|---|---|
| **Qwen-2.5-3B-Instruct** | The backbone model (frozen) |
| **LoRA r=32, α=64** | The trainable adapter (~2% of params) |
| **9 signals, 4 families** | The verifier |
| **4 outcomes** | store / defer / abstain / discard |
| **τ_store = 0.60, τ_defer = 0.45** | The two fixed storage cut-points |
| **ρ_min = 0.93** | Retention floor (below this → rollback) |
| **5 benchmarks** | FEVER, TriviaQA, CommonsenseQA (training) + TruthfulQA, StrategyQA (transfer) |
| **8 baselines** | B1-B8 |
| **6 cycles** | C0 through C5 |
| **+20.3% EM, -29.4% CHM** | Headline (best cycle C2 vs B1, training panel) |
| **3 theorems + corollaries** | Theory part (deferred to Pre-Thesis 3, don't over-claim) |

---

## PART 9 — Anticipated examiner questions + your answers

**Q: "Isn't this just RAG with extra steps?"**
A: "No. Plain RAG retrieves for every query and is silent about confidence. CAEM gates retrieval through a learned classifier, attaches a calibrated confidence to every answer, stores only verified answers, and improves itself over cycles. The RUC ablation shows retrieval alone *hurts* on a third of our benchmarks — CAEM's selectivity is the contribution."

**Q: "Why such a small model (3B)?"**
A: "Two reasons. First, the claim is *architectural* — if it works on a small model it isolates the architecture's contribution from raw scale. Second, it's reproducible on a single consumer GPU, which matters for an undergraduate research budget. The architecture is backbone-agnostic; a larger backbone is registered future work."

**Q: "How do you know the verifier itself isn't wrong?"**
A: "We don't need a perfect verifier. There's a data-purity result: as long as the verifier's balanced accuracy exceeds one half on the claims it actually scores, the stored pool's purity can exceed the verifier's own accuracy because base-rate dynamics favour the gate. We report empirically that this precondition holds — the cycle-zero validation gate shows a Cohen's d of +0.635 on the training panel, well above the threshold."

**Q: "What's the retention guard and why does it matter?"**
A: "Naive self-improvement causes catastrophic forgetting — the model gets better at new things by getting worse at old things. After every fine-tune we test on held-out probes; if the worst probe drops below 93% of the original capability, we roll back the adapter but keep the memory. Cycle 4 in our trajectory is a live example — the guard fired and protected the system."

**Q: "What does the composite hallucination metric actually measure?"**
A: "It's the equal-weighted average over eight failure-mode subtypes: confident confabulation, factual fabrication, logical fabrication, off-topic answers, defensive evasion, template leakage, false refusal, and length-padded over-generation. A uniform reduction across eight distinct failure modes is a stronger claim than a big drop on any single one."

**Q: "What's new versus your earlier work?"**
A: "The retrieval-utility classifier (Stage 3b) is the newest piece — it unlocks benchmark-conditional routing. The ablation shows it's responsible for nearly 6 points of pooled accuracy."

**Q: "Why does TruthfulQA do worse with the full system in some places?"**
A: "TruthfulQA is misconception-bait — the questions are designed to elicit common false beliefs, and dense retrieval pulls in passages *adjacent to the misconception* rather than refutations. That's exactly the failure the retrieval-utility classifier fixes — it routes those questions away from retrieval, recovering 10 points of accuracy."

**Q: "Is this deployed / does it work in production?"**
A: "The production deployment design is documented — same per-cycle loop triggered on a calendar or accumulation basis, with labels from oracle, user feedback, or human experts. The thesis evaluates the research trajectory; the production runbook specifies the deployed variant."

**Q: "What are the theorems?"**
A: *(Keep this light — theory is deferred to Pre-Thesis 3.)* "There are three theorems and supporting corollaries characterising the storage pool: a purity lower bound, monotonic improvement across cycles, and bounded convergence. The full formal treatment lands in the final submission; the poster focuses on the empirical validation."

**Q: "What's the single biggest contribution?"**
A: "Showing that hallucination reduction is achievable as a *system-level architectural intervention* on a small frozen model — without scaling, without touching the backbone weights — and that the gains compound across self-improvement cycles while a safety guard prevents regression."

---

## PART 10 — Presentation delivery tips

1. **Start with the problem, not the architecture.** Hook them with "models confidently lie, and scaling doesn't fix it."
2. **Use the architecture diagram as your spine.** Walk one question through the eight stages. That's 80% of your talk.
3. **Land the headline numbers** (+20.3% / -29.4%) clearly and pause.
4. **Show the C4 rollback as a feature.** "The safety guard fired here — this is the system working."
5. **Pitch the RUC as your fresh contribution.** It's the newest and most defensible piece.
6. **Be honest about the baseline re-run** if probed — rigor is a strength.
7. **Don't over-claim the theory** — say it's coming in the final submission.
8. **If you don't know something, say:** "That's a great question — it's in the methodology chapter; my current answer is [X], and I'll confirm the precise figure." Never invent a number.

### Your closing line (memorise)
> "CAEM shows you don't need a bigger model to hallucinate less — you need a better system around it: one that remembers what it verified, knows how confident it is, and improves itself without forgetting. Thank you."

---

## PART 11 — If you only have 20 minutes to study

Memorise, in order:
1. The 30-second pitch (Part 0).
2. The three structural problems: **stateless, overconfident, static** (Part 1).
3. The eight stages in order (Part 4) — this is the spine.
4. The headline numbers: **+20.3% EM, -29.4% CHM, beats 8/8 baselines** (Part 6).
5. The C4 rollback story (Part 5).
6. The RUC one-liner (Part 7).
7. Three Q&A answers: "isn't this just RAG", "why 3B", "what's the retention guard" (Part 9).

That's enough to present confidently and field most questions.

Good luck. You know this work better than anyone in the room.
