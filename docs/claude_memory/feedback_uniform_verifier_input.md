---
name: Verifier must see uniform reasoning+answer surface across all benchmarks
description: every signal in UnifiedVerifier (p_entail, p_ground_*, alias_overlap, q_a_relevance, atomic) must read the same canonical claim shape regardless of benchmark — never raw "Answer: C" or bare entity "Paris"
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
For the CAEM verifier, every signal that reads the model's answer must
operate on a uniform "Reasoning + canonical answer claim" surface,
regardless of which benchmark produced the sample. The verifier must
NEVER receive raw outputs that differ in shape across benchmarks
(e.g. "Answer: C" from CommonsenseQA, "Answer: B" from ARC, "Answer:
Paris" from TriviaQA, "Answer: supports" from FEVER).

**Why:** User repeated this twice. First time during the Fix 10
scoping discussion ("our verifier shoould see same time of reasoning
everythime"); second time on 2026-05-07 when reviewing my partial Fix
10 work. Per-signal patches (multichoice_scorer for p_ground,
EntityExpansionScorer for bare-entity p_ground) are insufficient
because p_entail, p_ground_atomic, q_a_relevance, and alias_overlap
still see the bare letter / bare entity and degrade to noise floor.
Inconsistent verifier inputs corrupt the per-bench composite isotonic
calibration — each benchmark's signals end up on a different scale.

**How to apply:** When adding any new verifier signal or modifying
verify() / verify_batch():

1. Compute one `canonical_answer` at the top of verify() via a single
   canonicalisation function that reads the query task family and
   the raw answer.
   - MCQ letter ("C", "B"): expand to the matching option text from
     the Choices block
   - Bare entity ("Paris", "1969"): wrap as
     "The answer to the question is: <entity>."
   - FEVER label / yes-no / already declarative: leave unchanged
2. Thread `canonical_answer` into every signal that takes the
   answer string: _score_p_entail, _score_atomic,
   _compute_q_a_relevance, _compute_alias_overlap.
3. Existing per-scorer patches (multichoice_scorer,
   EntityExpansionScorer) should consume the canonical_answer
   instead of re-deriving the canonicalisation themselves.
4. The raw display answer is still used for EM scoring at the eval
   harness — that path should NOT be canonicalised.

This is a hard architectural invariant for v2, not an optimisation.
