# Frontier-Model Reference Addendum (Writeup Plan)

**Purpose:** Add one or more "frontier reference" rows to Chapter 5 tables so the thesis directly addresses the scaling critique ("why not just use a bigger model?") without inviting apples-to-oranges comparisons.

**Out of scope:** Phase 1 Vast budget. This is an API-only addendum, run during writeup from a local machine (no GPU needed).

---

## Why add this

1. Every reader silently asks "what does GPT-4 / Claude / Gemini get on these benchmarks?" Answering it pre-empts the question.
2. Without a frontier row, Chapter 6 has a silent hole: "CAEM beats same-scale baselines" — which concedes that scale might just solve the problem. The current Phase 1 panel (B1–B7 all on Flan-T5-Large 780M) cannot answer the scaling critique.
3. It enables one concrete, citeable claim: **"CAEM recovers X% of the frontier gap at 1/N × parameters, without RLHF."**

## What *not* to do

- Do NOT add the frontier model as "B8" or "B9" in the main baseline panel. That implies controlled comparison, which is false (different training data, RLHF, tool access, tokenizer, inference stack).
- Do NOT compare CAEM zero-shot to GPT-4 with CoT + browse. Match the inference path: zero-shot vs zero-shot, or match-to-match.
- Do NOT claim CAEM "beats" or "loses to" the frontier in raw EM/F1. Frame as a recovery-fraction, not a head-to-head.

## What to do

Treat the frontier row as a **scale-ceiling reference**, not a baseline. Separate table row (or a clearly demarcated section) with explicit caveats.

### Model choices

Pick one primary + optionally one mid-tier reference. All API-accessible:

| Tier | Candidate | Reason | Approx. cost for 6×500 questions |
|---|---|---|---|
| Frontier | GPT-5 / Claude 4.6 / Gemini 3 | State-of-the-art scale ceiling | $5–15 |
| Mid-tier | GPT-5-mini / Claude Haiku 4.5 / Gemini 3 Flash | Shows the distillation/efficiency frontier | $1–3 |
| Small open (optional) | Phi-4 (14B) | Same "smart small model" philosophy as CAEM; strong baseline | free if self-hosted |

Single primary reference (one frontier + one mid-tier) is enough. Don't run three — signal-to-noise drops with more rows and the critique is addressed after one.

### Implementation sketch

1. Write `scripts/run_frontier_reference.py` mirroring `run_baseline.py` but calling the API instead of Flan-T5.
2. Use the exact same benchmark loaders, n_questions, prompts, and metrics as B1 Zero-shot. This keeps the comparison maximally controlled.
3. Log EM, F1, token F1, and (important) **prompt + full response** per-sample for manual inspection and contamination analysis.
4. Output format: `outputs/frontier_reference/<model>/results.json` — matches the baselines layout so `aggregate_ablation.py`-style tooling can pick it up uniformly.

### Benchmark contamination disclosure

GPT-4+ has almost certainly seen FEVER, TriviaQA, Natural Questions, TruthfulQA, StrategyQA, and ARC-Challenge during training. **Disclose this.** One paragraph in Chapter 5:

> *Frontier reference models have likely encountered all six evaluation benchmarks during pretraining, inflating their zero-shot numbers via leakage rather than ability. We therefore interpret the frontier row as a loose upper bound on what "scale plus alignment" achieves, not as a fair zero-shot measurement. For CAEM, Flan-T5-Large's pretraining cutoff predates FEVER-2018-test-split and ARC-Challenge-test-split refreshes, so leakage exposure is lower but not zero.*

(Verify Flan-T5's actual training cutoff before committing that last sentence — it's a fact-check item.)

## Suggested Chapter 5 wording (drop-in)

> **Scale-ceiling reference.** Table N+1 reports one frontier model (`<model_name>`, ~`<params>` parameters, `<training_tokens>` training tokens, RLHF-aligned) and one mid-tier model (`<mid_name>`) under identical benchmark conditions to B1 Zero-shot. This row is a **reference, not a baseline**: the comparison is uncontrolled along training data, alignment pipeline, and inference path. Its purpose is to position CAEM's 780M results within the broader LM landscape and to quantify the fraction of the frontier premium that CAEM's architectural approach recovers at 1/N × scale.
>
> We compute the recovery fraction as:
>
> $$ R_{recov} = \frac{\text{CAEM} - \text{B1 Zero-shot}}{\text{Frontier} - \text{B1 Zero-shot}} $$
>
> averaged over the six factual-QA benchmarks. Per-benchmark values appear in Table N+2.

## Suggested Chapter 6 claim (drop-in)

> *On factual QA across six benchmarks, CAEM recovers `<X>`% of the gap between Flan-T5-Large zero-shot and `<frontier_model>` zero-shot on confident-error rate, at 1/`<N>` × parameter count and without RLHF. This supports the claim that architectural interventions — verified episodic memory, pre-routing safety override, and constrained self-improvement with bounded forgetting — capture a meaningful fraction of the reliability improvements conventionally attributed to scale, in a regime where scale is not available (consumer GPU, university budget).*

## Cost and timeline

- **Compute:** Zero GPU. ~3000 API calls total (6 benchmarks × 500 questions × one model), runnable on a laptop over lunch.
- **Money:** $5–20 per model at 2026 frontier pricing. Within typical writeup-phase discretionary spending.
- **Time:** ~1 day of work — API script + run + numbers-in-tables + prose paragraphs.

## Order of operations (after Phase 1 Vast run completes)

1. Ensure Phase 1 `outputs/baselines/zero_shot/` has final B1 numbers (our anchor point).
2. Write the frontier-reference script.
3. Run against the same 500 questions × 6 benchmarks as B1.
4. Compute recovery-fraction table.
5. Insert Chapter 5 / Chapter 6 paragraphs drafted above.
6. Add contamination disclosure.

## Checklist before committing to this

- [ ] API budget available (~$20)
- [ ] API key obtained (OpenAI / Anthropic / Google — pick one)
- [ ] B1 Zero-shot numbers final (needed as anchor)
- [ ] Pick the one frontier + one mid-tier model
- [ ] Verify Flan-T5-Large training cutoff date for contamination paragraph
- [ ] Confirm thesis committee is comfortable with the framing (some prefer strict same-scale-only panels)

---

**Origin:** Discussion with Claude on 2026-04-18 about whether CAEM's thesis addresses the scaling critique. Conclusion: adding one reference row (framed as ceiling, not baseline) is the cheapest and most honest way to close this gap.
