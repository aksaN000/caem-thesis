---
name: CAEM Phase 2 comprehensive patch (Branch C 2026-04-25)
description: Eight architectural patches replacing the broken-composite Cycle-0 verifier with cal-prob + conformal gate + LoRA SIL; Phase 3 launched 2026-04-25 12:29 UTC.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
CAEM's Phase 1 Cycle-0 audit (n=3500, 7 benchmarks, 2026-04-25) revealed
the post-2026-04-24 verifier composite was structurally inadequate:
atomic 100% fallback on every benchmark including ASQA, q_a_relevance
sign-flips on FEVER (-0.58) vs open-QA (+0.50), pooled @80% precision
unreachable. Phase 2 designed and applied 8 patches; Phase 3 re-runs
Cycle-0 to validate them empirically.

## The 8 Phase 2 patches (all committed, awaiting Phase 3 empirical validation)

  2.1  CalProbComposite — per-signal isotonic + log-odds sum, replaces
       fixed weighted sum. Auto-handles q_a_relevance sign flip via
       Pearson-direction detection. Module: caem/verification/cal_prob_composite.py.

  2.2  Kernel Language Entropy (KLE) — module written
       (caem/verification/kle.py), generalizes Farquhar SE; replaces
       s_avg + h_norm with one principled signal. Wiring deferred until
       fresh M-chain data arrives in Phase 3.

  2.3  FActScore-style atomic decomp at storage gate — length-gated
       (cfg.atomic_min_tokens = 12). Short answers fallback to
       p_ground_mean; long-form answers (ASQA scope) get real per-fact
       NLI. Aligns with Min et al. EMNLP 2023 §3.

  2.4  Conformal split-CP storage gate — module
       caem/verification/conformal_gate.py. Fits τ_store at α=0.20
       (target 80% precision) and τ_defer at α=0.40 (60%) on the
       calibration fold. Replaces hardcoded 0.65/0.45 thresholds.

  2.5  Cherian boosting — folded into CalProbComposite as optional
       fit_boost flag. Logistic regression on per-signal calibrated
       probabilities; corrects residual cross-signal correlations.

  2.6  Empty-Answer compliance fix — strengthened SYSTEM_PROMPT in
       caem/prompts.py: "Answer:" line is MANDATORY, reasoning bounded
       3-4 sentences. Targets the 43-53% empty rate observed across
       benchmarks.

  2.7  Open-QA over-abstention fix — caem/prompts.py _task_spec for
       open-QA: refusal allowed only on Tier-3 with empty context, not
       as Tier-2 default fallback. Targets 38-44% IDK rate on
       TriviaQA/NQ/ASQA.

  2.9  LoRA SIL — DESIGN INTENT, IMPLEMENTATION DEFERRED to Phase 1b.
       Audit 2026-04-25 13:00 UTC confirmed the LoRA training-loop
       wiring (peft.LoraConfig, get_peft_model, adapter-only optimiser
       and L2 anchor, adapter checkpoint save/load) is NOT implemented
       in caem/training/self_improvement.py. The use_lora_training flag
       reverted to False; setting it True is a silent no-op. Phase 1a
       thesis runs full FT (the original design); LoRA swap is Phase 1b
       future work, motivated by ~100 SIL samples/cycle/benchmark
       sparse-pool regime where LoRA dominates full FT (Wang et al.
       2023; Biderman et al. 2024).

  2.8  CoVe — DROPPED (4× generation cost outweighs marginal precision
       lift over the other 7 patches).

## Architectural choices made and their justification

The 80% pooled precision target IS achievable under the comprehensive
patch + proper training-only calibration discipline (3.3% storage rate
ID, 1.4% transfer per Phase 1.5b/c offline tests). Per-benchmark
precision varies — ARC over-delivers (multi-choice base EM 73.6%), ASQA
under-delivers (base EM 1.4% caps the math).

LoRA replaces full FT because Phase 1 data shows ~100 verified episodes
per benchmark per cycle — well inside LoRA's home turf. Crossover to
full FT would require >5000 samples per cycle; production deployment
would also stay in LoRA regime at typical traffic volumes.

The conformal split-CP storage gate replaces the dual-threshold proposal
I had earlier; the existing four-outcome decision tree (STORE/DEFERRED/
ABSTAIN/DISCARD) already encodes that semantics, so no separate
architectural mechanism needed.

## Phase 3 launch state (2026-04-25 12:29 UTC)

- tmux session `plan_a`, runner pid 1871522
- Branch feat/qwen-3b-goal1 @ a6a65d8
- All cycle_0 artifacts cleared; pre-Phase-2 state archived to
  outputs/archive/pre_phase2_2026-04-25_T2/cycle_0/
- Watcher PID 1829502 still polling outputs/cycle_0/eval/

## Phase 4 deferred (post-Phase 3)

Empirical-evidence artifacts for Ch3/Ch4/Ch5/branch_C.md depend on Phase
3 data; Phase 4 is the artifact-generation + thesis-prose-update phase.

## How to apply / why

Why pooled isotonic over per-benchmark fit: production-parity
commitment (no benchmark label at deployment time). Per-benchmark fit
would break this commitment.

Why split-CP at α=0.20 / α=0.40: matches Mohri-Hashimoto 2024 +
Yadkori et al. 2024 conformal-abstention design. Calibration
(em-labels) used offline only; deployment uses fitted thresholds, no
labels.

Why no API: every cited mechanism (Mohri-Hashimoto, Cherian, Yadkori,
KLE, FActScore, LoRA) has a published local-only configuration using
Qwen-3B + MiniCheck + the local FAISS index.
