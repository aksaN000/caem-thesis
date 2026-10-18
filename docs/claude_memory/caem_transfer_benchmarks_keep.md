---
name: caem_transfer_benchmarks_keep
description: "Decision 2026-10-16: keep TruthfulQA + StrategyQA transfer benchmarks despite negative TruthfulQA; ch5/ch6/abstract must frame headline as training panel and split TQA loss into C0 (architecture) vs SIL parts."
metadata:
  node_type: memory
  type: project
  originSessionId: 9bbed419-78c6-48fb-ad84-30c1f48abc6e
  modified: 2026-10-16T07:22:37.672Z
---

Aksan asked (2026-10-16) whether to drop the transfer benchmarks because they look negative; he agreed to keep them.

**Numbers (tab_per_bench_trajectory, TQA EM Haiku-judged):** TruthfulQA EM 0.433 (B1) → 0.380 (C0, before any training) → 0.300 (C5): −5.3 pts at C0 (retrieval/architecture, cannot be forgetting) + −8.0 pts during SIL. StrategyQA EM 0.650 → 0.610 → 0.640 (flat), CHM 0.120 → 0.099 (better).

**Why keep:** pre-registered in ch1 objective/scope (dropping post hoc = selective reporting); headline is already the training-panel pooled number; TruthfulQA is the second RUC-ablation gate in ch6 (EM 0.210 → 0.310).

**How to apply (ch5, ch6, abstract sessions):** state plainly the headline is the training panel; split the TQA loss as above; the SIL-phase mechanism is "consistent with" drift, not shown.

Related: [[caem_truthfulqa_rescore_findings]], [[caem_ruc_thesis_framing]], [[feedback_chapter_review_workflow]].
