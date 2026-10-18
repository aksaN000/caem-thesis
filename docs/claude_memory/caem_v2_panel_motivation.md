---
name: v2 panel + Fix 3 motivation — v1 failed via EM regression + FEVER-dominated storage
description: The actual reason for the v2 panel + SIL pool reweighting redesign — v1 EM declined cycle-over-cycle and storage skewed to easy benchmarks
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**v1 failure observed by user (load-bearing for v2 design):**
1. EM was *reducing* across cycles — negative SIL trajectory; the model got worse, not better.
2. Storage became FEVER-dominated — easy bench filled the memory, drowning out signal from harder benches; SIL pool re-trained the model on its FEVER strengths and forgot/regressed on TriviaQA + NQ.

**Why HotpotQA over NQ in v2 panel:**
NQ in v1 had base EM 0.156, 78 em=1 / 500, STORE precision 0.333. So NQ was both rare-positive AND noisy-positive, contributing little SIL signal AND occasionally injecting wrong "stored truth" that the model trained on. HotpotQA is structurally harder but RAG-grounded: multi-hop with retrieved Wikipedia context produces more verifiable em=1 episodes than NQ's open-domain factoids did. Plus CSQA replaces the easy-bench-dominance problem by adding multiple-choice (binary verifier signal).

**v2 mechanisms aimed at each v1 failure axis:**

*FEVER-dominated storage:*
- Fix 1 (per-bench conformal gate): caps STORE rate independently per bench at α_store=0.05; FEVER cannot dominate volumetrically.
- Fix 3 (pool_reweighting): temperature-mixed softmax T=2.0 (10× count gap → 3× weight gap), upsample_cap=3.0, doremi_floor=50 samples/bench. Docstring literally cites "duplicate-100× pathology destroyed cycle-1 storage diversity in v1."
- Fix 13 (drop general-domain mix): SIL pool is panel-only; no irrelevant tokens from c4/wiki crowding the gradient.

*EM regression cycle-over-cycle:*
- Fix 8 (LoRA SIL primary): adapter-only training preserves base weights → EM regression has a hard floor; worst case is a discardable adapter, not a corrupted base.
- Fix 4 (multi-modal retention probe): every cycle measures retention against a fixed pristine baseline; halt trigger fires if any bench's retention drops below threshold.
- Fix 5 (coverage feedback + halt triggers): wired in run_experiment.py to abort the trajectory if EM collapses on a training bench.
- Loop filter (Goal 4 item 1): repetitive-loop chains fall back to short verified answer instead of training on the loop, preventing the CoT-collapse-into-loop pathology that worsens generation quality.
- Retroverify allow_downgrade (Goal 4 item 4): stale-confident entries can be demoted on re-scoring instead of silently lingering across cycles.

**What HotpotQA must NOT do at step_7_0_3 to prove v2's redesign worked:**
- Sustain HotpotQA STORE volume <10% of FEVER's STORE volume → FEVER-dominance returning despite Fix 1+3.
- HotpotQA per-cycle EM trending DOWN cycle 1→3 → SIL is over-fitting to FEVER again despite Fix 3 reweighting.
- Multi-modal retention probe triggers halt on TriviaQA or HotpotQA → confirms regression.

**How to apply:** When evaluating cycle-0/cycle-1 results, pull TWO metrics: per-bench STORE count ratio (FEVER:other should stay <2:1 after pool_reweighting), and per-bench EM trajectory across cycles (must be flat or up; any down-trend is the v1 failure recurring). The v2 design is bet specifically on Fix 1 + Fix 3 + Fix 8 collectively preventing this; if they don't, the redesign failed and the panel choice is secondary.
