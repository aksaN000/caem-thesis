---
name: v1 EM-drop disambiguation — Story A and Story B both real, Story B dominant (2026-05-09)
description: v1 cycle 0→4 EM regression decomposed into catastrophic forgetting (3 factoid benches, ~10-28pp gen_em drop) AND over-conservative gate (30-50pp gen_em vs final_em gap, universal). v1 published EM = gen_em; final_em was always near zero.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
The v1 cycle 0→4 "EM regression" 0.3834 → 0.3157 reported in the audit memory is the **gen_em** track (raw model prediction correctness, independent of gate decision). The actual committed EM (final_em = STORE-correct fraction over the population) was 2.20% at cycle 0 and 0.06% at cycle 4. The gate stored 96 items at cycle 0, **2 items at cycle 4** out of 3500 eval samples.

**Story A (catastrophic forgetting)** is real but localised:
- TriviaQA gen_em: 37.4% → 9.8% (−27.6 pp)
- NaturalQuestions: 17.2% → 6.6% (−10.6 pp)
- TruthfulQA: 33.4% → 25.0% (−8.4 pp)
- All three are open-domain factoid QA. FEVER / arc_challenge / strategyqa / asqa stayed within ±2 pp.
- The pattern matches the SIL-pool composition (58%→100% FEVER) — fine-tuning almost exclusively on FEVER claim-verification overwrote unrelated factoid knowledge.

**Story B (over-conservative gate)** is real and dominant:
- gen_em − final_em gap is 30-50 pp at every cycle on every non-trivial benchmark.
- At cycle 0 the model knows the right answer 38% pooled but commits 2.2%.
- By cycle 4 the model knows 32% but commits 0.06%.
- The "hallucination reduction" v1 reported is partly the gate refusing to engage (n_STORE = 2 at 100% precision is "0 false stores out of 0 stores").

**Why this matters for thesis + Phase 1c**:
- The gen-final gap is the higher-leverage redesign target than catastrophic forgetting. Shrinking it by half = +15 pp committed EM; eliminating Story A entirely = +7 pp pooled gen_em.
- The Phase 1c signal redesign (semantic entropy, fresh-retrieval atomic NLI, paraphrase invariance, cross-model agreement, synthetic-negative discriminator) is aimed at exactly this gap.
- v2 Fix #3 (loss-reweighted SIL pool) and v2 Fix #4 (multi-modal retention probe at 0.93 halt) already partially address Story A.
- Honest thesis framing is two-sided: traded factoid knowledge for hallucination resistance AND traded commit volume for precision. Story B is the bigger structural problem.

**Source artefact**: `outputs/research/v1_em_drop_diagnostic.md` with per-benchmark tables across cycles 0/1/3/4 from `gdrive:caem-phase1a/archive_v1/full_run/cycle_*/eval_transfer/`.
