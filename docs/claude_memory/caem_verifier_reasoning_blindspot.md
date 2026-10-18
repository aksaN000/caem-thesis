---
name: CAEM verifier blind to wrong-step-correct-conclusion reasoning (deferred to Ch5/Ch6 + Phase 1c)
description: Architectural limitation logged 2026-05-07; all composite signals score the canonical answer claim, never per-step claims in the reasoning chain. CORRECTED 2026-06-13 - composite uses exactly NINE signals (not 11; alias_overlap + entity_head_consistency + h_norm + p_contra are computed but EXCLUDED post-Phase-1d). Land in Ch5 §sec:disc-threats + Ch6 §future-work during the rewrite session, NOT a code fix in Phase 1a.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
**CORRECTION (2026-06-13, verified against live code):** the deployed composite uses exactly NINE signals per `caem/verification/cal_prob_composite.py` `COMPOSITE_SIGNALS` and the cycle-3 `composite_calibration.json`: u_token, u_dropout, u_internal, s_avg, p_entail, p_ground_max, p_ground_mean, p_ground_atomic, q_a_relevance. The earlier "11/12" count predated Phase 1d (2026-05-09), which removed alias_overlap + entity_head_consistency from the composite; h_norm + p_contra were also retired. All four are still COMPUTED and recorded by UnifiedVerifier (for diagnostics/back-compat) but contribute ZERO to u_stored. The "nine-signal verifier" name is therefore exact. (This stale "11" misled the CAEM Book Ch9 draft on 2026-06-13; fixed.)

The Unified Verifier scores all nine composite signals against the canonical answer claim, never against intermediate steps in the chain-of-thought. `p_entail` is the only signal that touches the chain and only as a single premise: it asks "does the chain entail the answer?", not "is each intermediate claim individually true?". A chain with a false middle step that still lands on the correct answer registers as a STORE-class episode at full confidence, and its full reasoning chain enters the SIL training pool.

**Why:** consistent with FActScore (Min EMNLP 2023, §3) — atomic decomposition is on the output, not the reasoning trace. Inherited identically from v1 (Session 42). None of the 13 v2 fixes target it. Decision logged in branch_C_log.md 2026-05-07 to NOT add a Fix 14: empirical case not built (cycle-0 audit found other dominant failure modes), no published step-wise verification technique with clean gains under tight compute, and architectural lock at 13 fixes prevents mid-flight expansion.

**How to apply:**
1. **During the Phase 2 rewrite session** (chapter rewrite, scheduled post-stat-quiz): add the fourth threat-to-validity bullet to Ch5 §sec:disc-threats; add chain-step verification to Ch6 §future-work alongside Phase 1c (7B QLoRA, ReFinED, SE Probes); add a one-sentence note to Ch4 §sec:verifier that the verification frame is answer-surface-anchored, citing FActScore. Anchor citations for future work: Ling et al. 2024 ("Deductive Verification of Chain-of-Thought"), Lyu et al. 2023 ("Faithful Chain-of-Thought").
2. **During Phase 1a trajectory monitoring:** watch cycle 1-10 for HotpotQA per-cycle EM dropping while p_entail stays high (chains deceiving the verifier on multi-hop), SIL train-loss / eval-loss divergence (overfitting to admitted bad reasoning), and qualitative samples in cycle_<N>/eval where stored episodes pair correct answers with obviously broken chains. These build the empirical receipt that justifies the Phase 1c extension.
3. **Do NOT propose a code fix mid-trajectory.** The 13-fix architectural lock is committed; re-fitting cycle-0 calibration and re-validating Step 7.0.3 to add a 14th signal carries higher risk than registering the limitation cleanly.

**Where the prose belongs:**
- Ch5 §sec:disc-threats: fourth threat alongside FEVER monoculture, verifier task-conditioning asymmetry, iterative-LoRA SV-collapse.
- Ch6 §future-work: chain-step verification line item joins Phase 1c.
- Ch4 §sec:verifier: one-sentence FActScore-aligned scope note.
