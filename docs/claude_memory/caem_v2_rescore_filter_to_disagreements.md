---
name: v2 RUC rescore should filter to A+B disagreements before invoking verifier; headline reporting rescores use eval pool only, composite-swap between arms
description: Two RUC-build cost optimizations Aksan locked in 2026-05-18. (1) For v2 fresh-pool RUC-training rescore, only the ~3 500-4 500 case-A and case-B disagreement samples (where direct_em ≠ rag_em) need verifier signals; case-C and case-D ties are dropped by Path B labelling anyway, so rescoring them wastes 65 % of GPU-h. (2) For with-RUC vs without-RUC reporting comparisons (Phase 1d vs Phase 1e headlines), baseline rescores use the CAEM eval pool (~1 500 × 9 = 13 500 samples max), not the 24 008-sample fresh pool. Per-arm headlines differ only in the composite applied to the same baseline signals — adapter held constant at Phase 1d C3, just swap composite_calibration.json.
type: feedback
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Aksan locked two rescore-cost optimizations on 2026-05-18** during the v2 build.

### (1) Filter to A+B disagreements BEFORE rescoring the v2 fresh pool

**Rule.** The v2 fresh-pool baseline rescore should only compute verifier signals for samples where `direct_em != rag_em` (cases A and B). Cases C (both right) and D (both wrong) are dropped from training by Path B labelling, so computing CHM / 9-signal composite on them is wasted GPU work.

**Why.** Path B's locked labelling rule (`caem/ruc/README.md` §2.1) trains the RUC only on rows where the two paths disagree at EM. C and D ties are noise on the disagreement-boundary calibration. Rescoring them produces signals nobody consumes.

**How to apply.** Before invoking `scripts/rescore_baselines_through_verifier`:

1. Walk both v2 baseline JSONs (`outputs/baselines_v2/zero_shot/<bench>_cycle0.json`, `outputs/baselines_v2/rag/<bench>_cycle0.json`).
2. Pair by `id` field; for each pair compute `(direct_em, rag_em)` using the dispatched EM (Haiku-judged for TruthfulQA, native EM for the others).
3. Keep only pairs where `direct_em != rag_em` (the A+B set, ~3 500–4 500 rows out of 12 004 per Aksan's yield estimate).
4. Write filtered baseline JSONs to a staging dir; invoke rescore against those filtered files.

**Magnitude.** Saves ~65 % of rescore wall-time. Full rescore on 24 008 sample-rescores ≈ 25 GPU-h; filtered rescore on ~8 000 sample-rescores ≈ 8–10 GPU-h. Cost saving ~$7.

**Reverse direction warning.** The C+D samples *can* still be useful for the Step 5 acceptance receipt (does the LR predict T2-default for C-like queries and T3-default for D-like queries?). That receipt only needs the trained LR's predictions on C+D samples, NOT their verifier signals — so the filter is safe.

### (2) Headline-reporting rescores use the eval pool, not the fresh pool

**Rule.** The v2 fresh-pool rescore is for **RUC training only, one-time**. The with-RUC vs without-RUC headline comparison in Ch5/Ch6 uses the CAEM eval pool (5-bench panel, 300 × 5 = 1 500 samples), with baselines rescored under each arm's reporting composite.

**Why.** The eval pool is the canonical CAEM-trajectory measurement panel. Baselines on the eval pool already exist (`outputs/baselines/`, the 9-baseline Ch5 panel) and were already rescored at Phase 1d cycle-3 composite (per memory `caem_baseline_audit_2026-05-15.md`). The fresh pool is RUC-training-disjoint by construction; using its samples for Ch5 headlines would be unfair (different distribution than the trajectory's eval pool).

**How to apply.** For the Phase 1e headline:

- Same baselines as Phase 1d (B1, B3, …). They were generated once on the eval pool, don't regenerate.
- **Same verifier signals** (computed under Phase 1d C3 adapter; verifier instrument held constant across arms for matched-protocol comparison).
- **Different composite weights**: apply Phase 1e's endpoint composite_calibration.json instead of Phase 1d's C3 composite to the existing 9 signals → recompute `u_stored` + `decision` per row.
- This is a **CPU operation**, ~10 seconds per file, ~$0 — not a full GPU rescore.

**Methodological cleanliness.** Fixing the verifier instrument (adapter + signals) and varying only the composite weight rule lets the headline contrast isolate "composite under Phase 1d's distribution" vs "composite under Phase 1e's distribution" — a calibration ablation, not a confounded model-vs-composite mash.

**When this rule kicks in.** Whenever a session asks "rescore baselines for the with-RUC headline" or "do we need to rerun baselines for Phase 1e" — the answer is "no full rescore, just composite-weight swap on the existing eval-pool baseline signals."
