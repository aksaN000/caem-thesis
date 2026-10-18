---
name: HotpotQA was swapped for NQ in v1 due to harder cold-start; v2 brought HotpotQA back
description: Empirical history — HotpotQA cold-start yield is materially lower than NQ; user removed it from v1 training panel, v2 reintroduced it
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
In v1 (main branch / feat/qwen-3b-goal1), the user replaced HotpotQA with NaturalQuestions in the training panel because HotpotQA was empirically harder to seed at cold-start. v2 brought HotpotQA back into the training panel and pushed NQ to transfer.

**Why:** v2 design intent — test cross-domain transfer on NQ, train on a multi-hop benchmark (HotpotQA) for capability coverage. CSQA is the new common-sense slot.

**Empirical confirmation (2026-05-07 Phase 1a cold-start):**
- FEVER yield: 29.8% (200/672)
- TriviaQA yield: 39.3% (201/512)
- HotpotQA yield: 15.4% at processed=792 — roughly half of FEVER, ~40% of TriviaQA.
- Train chunk capped at 1000 → if yield holds, HotpotQA seeds at ~154 instead of the 200 target.

**How to apply:** A short HotpotQA cold-start (150–170 stored) is NOT fatal — the cycle-0 cal-fold draws from a separate held-out split, so per-bench composite fitting still runs on ~200 labeled samples per bench. Impact is slightly thinner Tier-1/2 retrieval and slightly fewer SIL pool rows for HotpotQA in early cycles. Do NOT swap HotpotQA → NQ mid-trajectory pre-cycle-0: the v2 panel is wired across config, splits, sanitizer, prompts, transfer-storage guard, per-bench composite, per-bench conformal, per-bench T_b. Swap blast radius is bigger than the under-seed cost. If the next trajectory iteration revisits the panel, weigh swapping back to NQ as a design choice up front, not as a mid-run patch.

**v1 NQ failure profile (the bar HotpotQA must clear at step_7_0_3 decision gate):** base EM 0.156, STORE n=9/500, STORE precision 0.333, empty rate 44.4%, "I do not know" 38.8%, u_dropout ≥0.8 in 97.8%, conformal @70% mathematically capped (only 78 em=1 / 500). NQ's 3B-base ceiling + over-abstention + sparse em=1 stack-failure is what made the v1 swap necessary. v2 has per-bench composite + per-bench conformal + per-bench T_b + open-QA prompt fix + empty-answer compliance fix, all designed against this audit, so HotpotQA may NOT fail the same way even though it's structurally harder than NQ. Decision gate: step_7_0_3 (after cycle-0 cal-fold + composite fit + conformal fit). If HotpotQA STORE precision <0.55 or conformal can't reach α=0.05 → option 1: surgically swap HotpotQA→NQ at the gate (3 GPU-h restart), option 2: widen HotpotQA-only α to 0.40 + document weakness, option 3: continue if it clears.
