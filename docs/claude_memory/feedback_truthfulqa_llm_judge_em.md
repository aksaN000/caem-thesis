---
name: TruthfulQA EM must use em_llm_judged (Haiku), never the default em field
description: For TruthfulQA only, always use the em_llm_judged field (Haiku rescore) when computing EM, EM deltas, or any EM-derived metric. The default em field uses ROUGE-L > 0.15 which inflates EM artificially on open-ended QA. This rule has been violated multiple times; never miss again.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Rule.** When reporting any TruthfulQA EM number — pooled or per-bench — read the `em_llm_judged` field from the eval JSON, not the default `em` (which is ROUGE-L > 0.15 inflated).

**Why.** TruthfulQA is open-ended Q&A about adversarial misconceptions. ROUGE-L matches surface tokens, so an answer that contains the right keywords but flips the meaning ("yes lightning strikes twice in the same place" vs "no, it can") can score em=1 under ROUGE. The Haiku-judged EM evaluates semantic correctness against gold and is the canonical TruthfulQA metric. Documented in `caem_truthfulqa_rescore_findings.md` and `caem_baseline_format_floor_issue.md`. Aksan flagged this 2026-05-19 after I used the default em twice in a row when computing B1 → C5 contrasts.

**How to apply.**
- TruthfulQA EM lookups: `sample.get("em_llm_judged")` (fall back to `em` only if `em_llm_judged` is missing).
- Other benches (FEVER, TriviaQA, CommonsenseQA, StrategyQA, etc.): regular `em` field is correct.
- CHM is unaffected by this rule — CHM is computed from verifier signals + decision, not from EM directly. But subtype rates like `confident_confabulation_rate` use the sample's `em` field internally; for TruthfulQA the per-sample CHM should also use `em_llm_judged`. (Note: requires overriding the `em` key in samples before passing to `composite_hallucination_metric`.)
- Net Honesty, abstention rate, and gen→final gap on TruthfulQA all require the Haiku-judged EM.
- When computing pooled metrics across benches, swap in `em_llm_judged` for TruthfulQA before averaging.

**Verification checklist before reporting any TruthfulQA result:**
1. Did I pull `em_llm_judged` (not `em`) from the JSON?
2. Did I check that `em_llm_judged` is non-null for the samples I'm averaging?
3. If pooling across benches, did I use `em_llm_judged` for TruthfulQA and `em` for the others?

**Failure history (2026-05-19):**
- Phase 1d B1 → C5 audit (this conversation): used default `em` for TruthfulQA, produced inflated TruthfulQA EM = 0.387 at C5 and 0.803 at B1. Real numbers under Haiku judge are different (likely much lower B1, and a different C5 number). Recompute required.
- Earlier in same conversation: same mistake on first per-bench audit. Aksan corrected with "never miss this again."

**Adjacent memory:** `caem_truthfulqa_rescore_findings.md` (the original Haiku rescore findings), `caem_baseline_format_floor_issue.md` (B1/B2/B5 strict-EM zero on open-ended QA).

**Thesis framing (Aksan, 2026-10-16):** LLM judging is not a weakness to apologise for: it follows the judge-based protocol the TruthfulQA paper itself introduced (GPT-judge, Lin et al. 2022), and the same Haiku judge scores every system, so all comparisons stay matched. Absolute numbers are not directly comparable with papers that used the original GPT-judge; say so once, in ch5.
