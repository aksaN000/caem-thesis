---
name: feedback_theory_rigor_plain
description: "Whenever thesis theory appears (theorem/proposition/heuristic/remark/hypothesis mapping), Claude must verify the maths and present it in plain, jargon-free language; Aksan is not strong in theory and relies on Claude here."
metadata:
  node_type: memory
  type: feedback
  originSessionId: 9bbed419-78c6-48fb-ad84-30c1f48abc6e
  modified: 2026-10-16T02:27:17.150Z
---

Aksan (2026-10-16, ch1 drafting session): "as it's a theoretical thing, I am weak here, you have to carefully deal with it; whenever a theory comes, check it thoroughly for validity, correctness, presentation of the proofs or definitions; it shouldn't look like jargon, it should be easy to read and understand." He then delegated the H2 mapping decision outright ("if you think change is needed, do it").

**Why:** theory errors are high-risk viva questions and he cannot catch them himself; earlier drafts mislabelled heuristics as proven and mapped H2 (monotone purity) to thm:purity, which is only a lower bound.

**How to apply:**
- Re-derive every formal statement you touch (algebra, conditions, direction of inequalities); check that each hypothesis maps to a statement that actually predicts it.
- Name status honestly: in CAEM only thm:purity and thm:bayes-purity are proven; the rest are heuristics or remarks on idealised models.
- Explain in plain words, define terms inline ("committed cycles = cycles whose fine-tune was kept").
- Never reword pre-specified hypothesis statements; fix mappings and explanatory notes instead; verdict changes still go to Aksan.
- Related: [[feedback_explain_easier_detailed]], [[caem_theorem_scope_limit]], [[caem_drafting_session_start]].
