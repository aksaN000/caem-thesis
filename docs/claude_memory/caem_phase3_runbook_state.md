---
name: CAEM Phase 3 runbook state (post-Phase-2 re-run, launched 2026-04-25 12:29 UTC)
description: Where Phase 3 is, what to expect, what milestones to watch for, and how to resume context if compaction happens.
type: project
originSessionId: ae1f7815-b50e-4024-a362-c8bbbdf36803
---
## Phase 3 = re-run Cycle-0 with the comprehensive Phase 2 patch + run Step 7 main with LoRA SIL

Launched: 2026-04-25 12:29 UTC in tmux session `plan_a` (runner pid 1871522).

## What's running

```
runner branch:    feat/qwen-3b-goal1 @ a6a65d8
generator:        Qwen-2.5-3B-Instruct, frozen (LoRA SIL primary)
verifier:         MiniCheck-Flan-T5-Large + cal-prob composite + conformal split-CP gate
reranker:         BGE-reranker-v2-m3
embeddings:       all-mpnet-base-v2
RAG corpus:       21M-passage Wikipedia FAISS index (CPU-mapped)
hardware:         RTX 5090 32GB on Vast.ai (~$1/h; can swap to 4090 24GB ~$0.50/h after Phase 3)
```

## Expected timeline

```
T+0:00     12:29 UTC  Phase 3 launched
T+2-3h     ~15:00 UTC FEVER eval JSON     ← first decision point: empty rate?
T+4-6h     ~17:00 UTC TriviaQA eval JSON  ← open-QA prompt fix verification
T+6-9h     ~20:00 UTC NQ eval JSON        ← confirms TriviaQA pattern
T+8-12h    ~23:00 UTC TruthfulQA eval JSON
T+11-14h   ~02:00 UTC StrategyQA eval JSON
T+13-16h   ~04:00 UTC ARC eval JSON
T+15-18h   ~06:00 UTC ASQA eval JSON      ← long-form, atomic fires here
T+18h      ~06:30 UTC Step 7.0.2 calibration (composite_calibration.json + conformal_gate.json)
T+18h      ~07:00 UTC Step 7.0.3 validation gate (Cohen's d ≥ 0.20 floor)
T+18h+     ~07:30 UTC Step 7 main launches (LoRA SIL, 10 cycles)
T+25h      ~13:00 UTC Step 7 main complete
```

## Decision points (where to halt and discuss)

  - **After FEVER JSON** — if empty-Answer rate is still 40%+ or Cohen's d
    on the composite collapsed, halt before burning 12 more hours. Watch
    for: empty rate <10%, composite Cohen's d > 0.30.
  - **After Step 7.0.3 validation gate** — Cohen's d on every signal
    across the panel; pass/fail on whether the patch delivered. Locks
    thesis claims for Phase 4.
  - **After Step 7 main first 2-3 cycles** — does LoRA SIL produce
    measurable cycle-over-cycle CHM reduction? Self-improvement claim
    gets empirically tested here.

## Live status checks

```bash
tail -f /workspace/caem/outputs/cycle_0/run.log                              # raw pipeline
tail -f /workspace/caem/outputs/cycle_0/run.log | grep --line-buffered -E "harness|EvalHarness done"
tail -f /tmp/atomic_watch.log                                                # auto-prints fallback% + Cohen's d per JSON
tail -5 /workspace/caem/outputs/heartbeat.log                                # health
ls /workspace/caem/outputs/cycle_0/eval/                                     # which benchmarks done
```

## If a conversation compaction happens

When the context gets compressed and a new conversation resumes, the agent should:
  1. Read this memory file + caem_phase2_comprehensive_patch.md + caem_phase1_audit_findings.md
  2. Check tmux ls + tail outputs/cycle_0/run.log to see how far Phase 3 got
  3. Read outputs/cycle_0/eval/*.json for the actual numbers if any have landed
  4. Resume from the appropriate decision point above

The full conversation transcript for this session is at:
  /root/.claude/projects/-workspace/<session-id>.jsonl
  (find latest via: ls -t /root/.claude/projects/-workspace/*.jsonl | head -1)

## Key files to reference (in /workspace/caem/)

```
caem/verification/cal_prob_composite.py    Phase 2.1 module (per-signal isotonic + Cherian boost)
caem/verification/conformal_gate.py        Phase 2.4 module (split-CP threshold fitter)
caem/verification/kle.py                   Phase 2.2 module (Kernel Language Entropy)
caem/verification/verifier.py              Patched _composite() + _decide() dispatch
caem/training/self_improvement.py          LoRA-primary SIL docstring
caem/prompts.py                            Empty-Answer fix + over-abstention fix
caem/config.py                             composite_mode, conformal_*, atomic_min_tokens, use_lora_training
scripts/fit_composite_calibration.py       Step 7.0.2(A)
scripts/fit_conformal_gate.py              Step 7.0.2(B)
run_phase1a.sh                             Updated step_7_0_calibrate (3-stage fit)
outputs/cycle_0/                           Active run output dir
outputs/archive/pre_phase2_2026-04-25_T2/  Pre-Phase-2 archive snapshot
outputs/archive/pre_a2v3_broken_composite_2026-04-25/  Broken-composite snapshot (Phase 1 audit)
```

## Phase 4 (post-Phase 3) deferred work

Empirical-evidence artifacts for thesis: Cohen's d per benchmark table
(Ch5), composite-weight justification table (Ch4), τ_store
precision-cliff curves (Ch5), atomic-decomp scope figure (Ch4),
prompt-compliance recovery table (Ch5). Plus Ch3 §FR4/§FR7 prose
edits and branch_C.md design log entry.
