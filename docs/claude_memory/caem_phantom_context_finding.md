---
name: caem_phantom_context_finding
description: "Found 2026-10-16 (ch4 session): after the first SIL fine-tune, ~93–99% of Tier 2 (no-retrieval) answers cite 'the context' they never had (3% at C0) — SIL trained on RAG-written reasoning chains. Report in ch5/ch6."
metadata:
  node_type: memory
  type: project
  originSessionId: 9bbed419-78c6-48fb-ad84-30c1f48abc6e
  modified: 2026-10-16T09:04:21.753Z
---

Evidence (`outputs/full_run/eval/<bench>_cycle<N>.json`, Tier 2 samples whose prediction mentions "context" in its first 200 characters, all five benchmarks): C0 25/731 (3%), C1 693/747, C2 808/828, C3 651/671, C4 663/684, C5 746/753. Tier 2 prompt has no passages (`caem/prompts.py::build_tier2_prompt`, `caem/pipeline.py::_tier2`). Example: FEVER id 16149 (C4 stream chunk, Tier 2, STORE, EM 1): "Reasoning: The context provides detailed historical accounts ...".

**Why:** SIL training targets are stored reasoning chains (`self_improvement.py` "chain preview"), most written under the retrieval prompt, so the adapter learned the "The context ..." phrasing. It is a faithfulness-style artefact (references a non-existent source); answers can still be correct.

**How to apply:** not yet in the thesis. Raise with Aksan in the ch5 session (result/limitation; possibly ties to the template-leakage CHM subtype) and ch6 future work (strip context references from training chains, or train on Tier-2-style rewrites). Ch4 only notes that training targets are the stored chains. Related: [[caem_sil_pool_precision_finding]], [[caem_verifier_reasoning_blindspot]].

**Author's view (2026-10-16):** "we should've trimmed the 'the context provides' part somehow as a preprocessing step". Present in ch5/ch6 as a limitation with that fix as future work: strip or rewrite context references in SIL training chains (or train Tier-2-style targets) before fine-tuning.
