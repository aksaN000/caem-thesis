---
name: v1 cycle 0-4 trajectory audit — the empirical record that drove v2 redesign
description: Per-cycle pooled EM, memory composition, per-benchmark EM, three-mechanism failure diagnosis from branch_C_log.md 2026-05-06 entry
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**Pooled trajectory cycles 0-4 (n=3500 eval/cycle, 7-benchmark panel):**

| Cycle | Pooled EM | Pooled CHM | SIL pool composition |
|---:|---:|---:|---|
| 0 | 0.3834 | 0.1706 | 58% FEVER / 28% TQA / 14% NQ (cold-start) |
| 1 | 0.3889 | 0.1656 | 89% FEVER |
| 2 | 0.3657 | 0.1585 | 96% FEVER |
| 3 | 0.3331 | 0.1583 | 98% FEVER |
| 4 | 0.3157 | 0.1526 | **100% FEVER (873/874)** |

Pooled EM dropped −17.7% across c0→c4. CHM dropped −10.5%.

**Per-bench EM c0 → c4:** TriviaQA 0.374→0.098 (−73.8%), NQ 0.172→0.066 (−61.6%), TruthfulQA 0.334→0.250 (−25.1%), FEVER 0.456→0.474 (flat, peaked 0.490 c2). StrategyQA/ARC/ASQA flat.

**Three failure mechanisms (diagnosed 2026-05-06):**
1. Verifier-discrimination decay on FEVER: prec@k=20 collapsed 0.95→1.00→0.95→0.80 across c1-c4; per-bench gate degenerate at c4 (needs α≥0.20).
2. Precondition starvation on TQA/NQ: global α=0.05 unreachable → 0% storage on those benches → SIL pool 100% FEVER → capacity competition during full-FT bulldozes TQA/NQ representations.
3. NQ structural: verifier balanced accuracy <0.5 on every cycle (0.45/0.25/0.25/0.35), no threshold relaxation recovers; bare-entity NLI fundamentally non-discriminative.

**Per-bench conformal ablation receipts:** FEVER fits c1-c3 only, TriviaQA degenerate every cycle (smallest viable α=0.40), NQ degenerate at every α∈[0.05, 0.50].

**MMLU retention guard false-positive:** reported 1.04× pristine across c1-c5 but model specialized on bounded-label MCQ while forgetting open-text generation. v2 Fix 4 (multi-modal retention probe) added TriviaQA-test + HotpotQA-test to close this hole.

**Mid-trajectory parameter change:** safety_u_pre_min relaxed 0.60→0.38 at cycle 1 close. FEVER lifted +0.046 c1→c3, but TQA −0.184 / NQ −0.066. Trajectory continued under 0.38 through c4 as a registered ablation.

**Cycle-4 close failure mode discovery → v2 architectural lock 2026-05-06.** Cycle 5 SIL on broken architecture is methodologically discarded (preserved at outputs/full_run/aborted_cycle5_sil_2026-05-04/ as audit only).

**How to apply:** When evaluating v2 trajectory health, compare against these v1 receipts:
- v1 c4 SIL pool was 100% FEVER → v2 c4 must be <70% FEVER for Fix 1+3 to be working
- v1 c4 TQA EM was 0.098 → v2 c4 TriviaQA must be >0.30 for Fix 8 LoRA + Fix 3 pool reweighting to be working
- v1 c4 NQ admissions zero → v2 dropped NQ from training; HotpotQA admissions must be non-zero across all cycles
- v1 c4 FEVER prec@k=20 was 0.80 → v2 FEVER prec@k=20 must hold ≥0.85 across cycles (proves no verifier-decay even with adapter)

**Source artefacts:** outputs/full_run/eval/*_cycle{0,1,2,3,4}.json, outputs/full_run/cycle_{2,3,4}/conditional_conformal_ablation.json, branch_C_log.md 2026-05-06 entry "v2 architecture lock: discovered FEVER-monoculture failure mode + 13-fix response."
