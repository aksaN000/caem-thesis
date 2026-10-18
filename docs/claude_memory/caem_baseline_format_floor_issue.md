---
name: B1/B2/B5 strict-EM is artificially zero on open-ended QA — matched-protocol fix needed
description: Prose-output inference baselines fail EM because they emit prose without "Answer:" markers; raw Qwen capability is ~60% on TriviaQA, not 0%. Headline B1→C3 framing conflates format scaffolding with CAEM machinery until fixed.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Empirical finding (2026-05-16, post-baseline-panel-launch):** On 300 held-out TriviaQA eval samples, B1 zero_shot reports strict EM = 0.0% but the gold answer appears as a normalized substring of the prediction in 59.0% of samples. B2 cot reports strict EM = 2.7% with the same relaxed-match rate of 62.3%. The model knows the answer; the prose lacks the "Answer:" / "the answer is" markers the scorer in `eval/metrics.py` requires.

**Mechanism:**
- `extract_answer()` in `eval/metrics.py:80-127` looks for "Answer:" or "the answer is" markers. Without either, the full prose string is normalized and strict-compared to gold — almost always fails on open-ended QA.
- B1's `_build_prompt` is bare `_wrap_chatml_user(query)` — no system prompt, no forced prefix. Model emits "The actor who won...was Geoffrey Rush." Gold span "Geoffrey Rush" is buried in prose.
- B2's CoT trigger ("Let's think step by step.") makes Qwen enumerate reasoning steps and burn 256 `max_new_tokens` before reaching the answer line. Pattern: "1. **Identify the Song**... 2. **Research the Song**..." then EOS.
- B3/B4 use `caem.prompts.build_tier3_prompt` with the same FORCED_PREFIX as CAEM Tier 3 — fine. B6/B7 are fine-tuned to produce the format — fine. B5 fiveshot_cot expected to inherit the same issue.

**Why this matters:** The locked headline is B1 → C3 (architecture+SIL). With B1 TriviaQA = 0.0%, the "+60 pp" reads as CAEM machinery contribution but is mostly format scaffolding. Violates the matched-protocol invariant in `feedback_uniform_verifier_input.md`.

**Why:** Without the same answer-format scaffolding CAEM Tier 2 gets, B1/B2/B5 measure "prose generation tendency" not "raw QA capability". The published thesis comparison must pair like with like.

**How to apply:** Before claiming the B1 → C3 headline, patch `eval/baselines.py:ZeroShotBaseline`, `CoTBaseline`, `FiveShotCoTBaseline` to apply a minimal system prompt mandating the "Answer:" line plus a forced "Reasoning:" prefix for the CoT variants (mirror `caem.prompts.SYSTEM_PROMPT`). Re-run those three baselines (~40 min) after the in-flight B6-B9 panel finishes. Don't interrupt the current run — finish the bulk first, queue the matched-protocol re-run at the end. The rescore phase reads whatever's in `outputs/baselines/<name>/*_cycle0.json`, so re-running just B1/B2/B5 is sufficient.

**Alternative philosophy (don't recommend):** Keep B1 as a "raw unscaffolded floor" and report both pre-rescore EM and a separate relaxed-EM as a Ch5 §threats hedge. This preserves the "absolute floor" framing but the threats section then has to spend a paragraph explaining why the headline is inflated.

**Status (2026-05-16):** Diagnosed, not yet patched. User to confirm philosophy before re-run.
