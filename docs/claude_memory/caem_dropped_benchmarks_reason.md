---
name: caem_dropped_benchmarks_reason
description: "Thesis mentions only HotpotQA + Natural Questions as dropped benchmarks (ARC removed entirely); real reason is low base accuracy, NOT verifier balanced accuracy ≤ 0.5 (receipts show BA 0.63 / 0.52)."
metadata:
  node_type: memory
  type: project
  originSessionId: 9bbed419-78c6-48fb-ad84-30c1f48abc6e
  modified: 2026-10-16T02:27:29.059Z
---

Decision 2026-10-16 (Aksan, ch1 session): name only HotpotQA and Natural Questions as benchmarks dropped before the main run; remove ARC-Challenge from the thesis prose entirely (ch2's citation of a study's ARC metric stays).

Receipts (Drive): `CAEM_thesis/9_archive/pre_main_snapshot/700b9e3_phase1d/cycle_0/eval_excluded_v21/hotpotqa_cycle0.json` and `CAEM_thesis/9_archive/v1_panel/full_run/cycle_0/eval/natural_questions_cycle0.json` (n = 500 each):
- HotpotQA: base EM 0.090; gate stored 26, 13 correct (purity 0.500); TPR 0.289, FPR 0.029, balanced accuracy 0.630.
- NQ: base EM 0.172; stored 12, 5 correct (purity 0.417); balanced accuracy 0.521.

**Why it matters:** every chapter used to say they "violated the verifier-balanced-accuracy precondition"; the data show BA > 0.5, so that was false. Aksan remembered "below 0.5", which is the base accuracy. Canonical wording lives in ch1 sec:scope-research ("too few and too noisy to train on"); NQ's training split is reused only for RUC training.

Open item: appendix B tables (prompt_compliance, atomic_scope, tab_prompt_design_ablation) still carry ARC-C rows from an earlier panel — appendix session decides. Related: [[caem_hotpotqa_vs_nq_history]], [[caem_drafting_session_start]].
