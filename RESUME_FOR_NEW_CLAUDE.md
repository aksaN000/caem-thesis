# Resume Bootstrap for a Fresh Claude Session

**You are a new Claude session.** The user (Aksan) just started talking to you. They're not going to explain context. Read this file end-to-end first, then announce "Bootstrapped from RESUME_FOR_NEW_CLAUDE.md — anchor date 2026-05-15, project state Cycle 4 Step 5 eval. Ready." Then wait for their actual request.

If they ask anything substantive, the rule is: **the live code under `caem/` is the source of truth for any factual claim.** Memory files and design docs can drift. Read code first, then answer.

---

## 0. The one-paragraph orientation

You are inside `/workspace/caem/` on a Vast.ai 5090 GPU box. The user is **Aksan Gony Alif**, 4th-year CSE undergrad at BRAC University (Bangladesh), defending his CSE400 thesis on **CAEM (Confidence-Aware Episodic Memory with Self-Improvement)**. CAEM is a 3-tier hallucination-reduction architecture: memory (Tier 1) → zero-shot (Tier 2) → RAG (Tier 3), with a 9-signal calibrated-probability verifier gating per-cycle self-improvement. Backbone is Qwen-2.5-3B-Instruct + LoRA r=32 α=64. He has been autonomously running a 10-cycle SIL trajectory; Cycle 4 just aborted on the retention guard, and the locked plan is to stop after Cycle 5, then pivot to baselines + ablations + diagnostics + thesis finish. All 6 thesis chapters are already written and audited.

Treat him as a peer researcher who knows the architecture deeply. He's running on tight credit (~$70-85 of Vast budget remaining) and a thesis deadline.

---

## 1. Read these files NOW, in this order

```bash
# 1. Project anchor (this file + the one referenced)
cat /workspace/caem/CLAUDE.md

# 2. Memory index + per-topic memory (file-based, persists across accounts)
cat /root/.claude/projects/-workspace/memory/MEMORY.md
# Then read individual memory files as the index references them when relevant

# 3. Dated decision log — most recent at top, last 200 lines is the working memory
head -200 /workspace/caem/branch_C_log.md

# 4. Post-trajectory roadmap (Phase 1.5 + 1.6 = what to do next)
grep -A 200 "Phase 1.5" /workspace/caem/PRODUCTION_NEXT_SESSION_PLAN.md | head -250

# 5. Recent commits (last 20)
cd /workspace/caem && git log -20 --oneline

# 6. Live runner status
tmux ls
tmux capture-pane -t plan_a -p | tail -30
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# 7. Spot-check current cycle data
ls -lt /workspace/caem/outputs/full_run/eval/*.json | head -10
```

If `/root/.claude/projects/-workspace/memory/` is empty or inaccessible (because the new account uses a different memory path), skip step 2. The other steps give you ~90% of context anyway.

---

## 2. Project state snapshot (anchor: 2026-05-15)

### Architecture (locked v2.1)

- **Backbone:** Qwen-2.5-3B-Instruct, no quantization
- **SIL primitive:** LoRA r=32, α=64, all-linear targets {q,k,v,o,gate,up,down}, LR=2e-4, 3 epochs/cycle
- **Storage gate:** FIXED threshold on calibrated probability (NOT conformal — retired Phase 1c)
  - `store_threshold = 0.60`, `defer_threshold = 0.45`
  - Per-benchmark composite with shrinkage prior α=0.6 toward pooled fit
- **Verifier:** 9 signals (NOT 11, NOT 10 — `h_norm`, `alias_overlap`, `entity_head_consistency` retired at Phase 1c)
  - NLI judge: bare MiniCheck-Flan-T5-Large (frozen Qwen long-hypothesis judge ABLATED at Phase 2; Pearson ρ=0.5843 < 0.70 floor)
- **Retention:** multi-modal probe panel (MMLU + TriviaQA-test + CommonsenseQA-test); rollback floor 0.93 on worst-probe ratio
- **L2 anchor (λ_anc):** ZERO on shipped LoRA path; only active on inactive Full-FT fallback (which is registered Ch6 future work)
- **Optimizer:** torch.optim.AdamW (32-bit) on LoRA path; 8-bit variant inactive
- **Trajectory:** 10 cycles planned, stop-locked at C5 (see Cycle 4 abort below)

### Panel (5 benchmarks)

- **Training panel:** FEVER, TriviaQA, CommonsenseQA (3 — these provide SIL stream chunks)
- **Transfer panel:** TruthfulQA, StrategyQA (2 — held out, never trained on)
- **Retired:** HotpotQA, Natural Questions, ARC-Challenge (cycle-0 evidence of precondition violation)
- **Stream chunk size:** FEVER+TQA at 2000/cycle, CSQA at 700/cycle (CSQA train is only 9.7k)

### Cycle-by-cycle empirical state

| Cycle | Status | Pooled CHM | Pooled EM | T2 share (3-bench) | Notes |
|---|---|---|---|---|---|
| C0 | closed | 0.156 | 0.480 | 1.5% | baseline |
| C1 | closed | 0.122 | 0.515 | 4.8% | big CHM drop (-22% rel.) |
| C2 | closed | 0.120 | 0.520 | 7.8% | diminishing returns |
| C3 | closed | 0.121 | 0.482 | 18.1% | **parametric high-water mark** |
| C4 | Step 5 eval, ~3/5 benches done | ~0.124 | ~0.482 | 20.9% | **SIL aborted on retention; weights=C3** |
| C5 | will start after C4 | predicted abort | predicted ~C3 | TBD | trajectory stops here regardless |

### Cycle 4 retention abort (2026-05-14)

The retention guard fired for the first time in flight. TriviaQA-test probe dropped from pristine 0.395 to 0.345 (then 0.330 on retry), giving ratio 0.8734 (then 0.8354), both well below the 0.93 floor. **Weights rolled back to cycle-3 θ_prev; memory store preserved.** This is the **asymmetric-rollback property** working as designed — direct empirical receipt for H4 + load-bearing differentiator #4. Cause: C4 SIL training pool was 2881 episodes vs C3's 1542 (2× gradient signal in one shot exceeded the implicit LoRA envelope).

### Skip-retroverify-on-abort patch (commit 2347d60)

Step 2.5 (retroverify) is now SKIPPED when `cycle_result.aborted=True`. Saves ~6h GPU per aborted cycle. Patch verified firing on C4 retry. Runner restarted with patched code at 11:20 BDT 2026-05-14.

### Headline empirical findings so far

1. **Pooled CHM dropped 22%** C0 → C3 (0.156 → 0.121). Per-benchmark: 25-29% reductions on 4 of 5 benches (CSQA flat — already low at C0).
2. **Both transfer benchmarks (TruthfulQA, StrategyQA) show same CHM reductions as training panel** → strong transfer evidence.
3. **T2 EM (0.620 at C3) > base T3 EM at C0 (0.476) by +14 pp** → SIL-as-distillation working: fine-tuned model answering zero-shot beats base model with RAG. Every query that migrates T3 → T2 is BOTH cheaper AND more correct.
4. **False-refusal rate: 0.296 → 0.069 (−77%)** — biggest single-subtype win. Gate is learning to trust the fine-tuned model.
5. **Confident-confabulation rate: 0.110 → 0.242 (+120%)** — ONLY regression. Register as Ch5 §sec:disc-threats item (known SIL self-distillation side effect).
6. **Cycle-0 validation gate:** Cohen's d ID +0.467, transfer +0.531 — both clear the 0.20 floor by 2.3-2.7×.

### What's NOT yet evidenced (the post-C5 work)

- **H2 (CHM reduction vs baselines under matched protocol):** Not started. Need B1-B7 panel.
- **H5 (cycle-boundary mechanisms contribute):** Not started. Need 3 ablations (no_retroverify, no_self_improvement, no_forgetting_guard).
- **Statistical significance panel:** depends on baselines + ablations.

---

## 3. User preferences (override default behavior)

These are the MOST important rules. Violating them annoys the user.

| Rule | Why |
|---|---|
| **Report clock times in BDT (UTC+6), not UTC** | User is in Bangladesh |
| **NEVER include the Claude `Co-Authored-By` trailer in git commits** | Explicit user preference |
| **End scheduled-loop ticks with action or ScheduleWakeup, NEVER a question** | User runs autonomous loops because they're unavailable |
| **Plain-English explanations with concrete numbers, step-by-step, as long as needed** | User wants clarity over brevity |
| **Thesis prose bans: EM dashes, `§` symbol, all code/file/variable/folder names** | Top-tier paper style |
| **Thesis edits must increase coherence** | Preserve terminology, cross-refs, registry counts, fold semantics across all chapters/bib/runbook/code |
| **After any compaction, READ ALL CAEM CODE END-TO-END before substantive answers** | On-demand reads have failed too often (Tier 2↔RAG, cal-fold↔eval-fold, α↔p_c confusions) |
| **Verify exact sample count + dedup before any GPU run** | Vast credit is real money; predict wall-time before launch |
| **All decisions must produce on-disk evidence** | Every parameter choice/gate result/iteration must leave a JSON/TeX/PNG/log artifact under `outputs/` with dated `branch_C_log.md` entry |
| **Code is the source of truth, NOT branch_C_log.md or branch_C.md** | Design diaries can drift; live `caem/` code is authoritative |

---

## 4. File map

| Path | What it is |
|---|---|
| `/workspace/caem/CLAUDE.md` | Project anchor (shorter than this file) |
| `/workspace/caem/branch_C_log.md` | Dated decision log (reverse-chrono; read top first) |
| `/workspace/caem/PRODUCTION_NEXT_SESSION_PLAN.md` | Roadmap with Phase 1.5 (C5-stop decision) + Phase 1.6 (post-C5 finish line) |
| `/workspace/caem/thesis_report/chapters/chapter_{1..6}.tex` | All 6 chapters, already audited + drift-fixed |
| `/workspace/caem/thesis_report/appendix/appendix_*.tex` | Appendices |
| `/workspace/caem/thesis_report/figures/auto/*.tex` | Auto-generated tables (populated by `scripts/make_tables.py`) |
| `/workspace/caem/caem/` | Live code (the source of truth) |
| `/workspace/caem/caem/pipeline.py` | Main inference pipeline |
| `/workspace/caem/caem/router.py` | 3-tier dispatch |
| `/workspace/caem/caem/verification/verifier.py` | 9-signal verifier |
| `/workspace/caem/caem/verification/cal_prob_composite.py` | Calibrated probability composite + shrinkage prior |
| `/workspace/caem/caem/training/self_improvement.py` | SIL loop + retention guard |
| `/workspace/caem/caem/ablation/variants.py` | 3 registered ablations (no_retroverify, no_self_improvement, no_forgetting_guard) |
| `/workspace/caem/scripts/run_experiment.py` | Cycle orchestration (the file the runner executes) |
| `/workspace/caem/run_phase1a.sh` | Wrapper that launches `run_experiment.py` with auto-resume detection |
| `/workspace/caem/outputs/full_run/` | All trajectory artifacts (cycle_N dirs + eval/ + cycle_0/) |
| `/workspace/caem/outputs/cycle_0/weight_validation.json` | Cycle-0 validation gate verdict (Cohen's d, store-discard gaps, poisoning rates) |
| `/workspace/caem/docs/PRODUCTION_RUNBOOK.md` | Deployment design (read end-to-end before any deployment-mode question) |
| `/workspace/caem/eval/metrics.py` | Metric definitions (CHM, CES, subtypes) |
| `/workspace/caem/presentation/caem_supervisor.tex` | Slide deck for thesis defense |
| `/workspace/caem/outputs/research/topvenue_panel_2026-05-14.md` | 5 pending baseline-panel decisions (need user sign-off before B1-B7 launches) |

---

## 5. Where the trajectory is RIGHT NOW

```bash
# Live status checks (run these to orient)
date && tmux ls
ps -ef | grep run_experiment | grep -v grep
tail -5 /workspace/caem/outputs/full_run/experiment.log
ls -lt /workspace/caem/outputs/full_run/eval/*.json | head -8
```

**As of anchor date (2026-05-15 05:30 BDT):**
- Runner PID 507749 alive, `tmux plan_a`
- C4 Step 5 held-out eval: 3 of 5 benchmarks done (FEVER, TriviaQA, CommonsenseQA); TruthfulQA + StrategyQA in flight
- C4 expected close: ~06:30 BDT May 15
- C5 starts immediately after, will likely abort on TQA-test, skip Step 2.5 (patch), Step 4 (~14h) + Step 5 (~4h)
- C5 expected close: ~01:00-07:00 BDT May 16

By the time you read this, the timeline may have advanced. Run the live checks first.

---

## 6. What to do next (after C5 closes)

This is the post-C5 finish-line roadmap. Detail in `PRODUCTION_NEXT_SESSION_PLAN.md` Phase 1.6.

| Step | What | Duration | Cost |
|---|---|---|---|
| P-1 | Kill runner, verify C5 close, gdrive offload | 5 min | $0 |
| P-2 | Baselines B1-B7 (need 5 panel decisions locked first) | 2-3 days | $30-40 |
| P-3 | Ablation panel (3 lesions) | 1-2 days | $15 |
| P-4 | Diagnostics (purity, signal-corr, Tier-1 amortisation, Cohen's d) | 1 day | $5 |
| P-5 | Statistical analysis (McNemar + bootstrap BCa + Holm) | hours | $0 |
| P-6 | Auto-table generation (`scripts/make_tables.py`) | hours | $0 |
| P-7 | Ch5/Ch6 integration writing | 2 days | $0 |
| P-8 | Visual polish (TikZ figures, plots) | 1-2 days | $0 |
| P-9 | Final cross-bench coherence pass | 1 day | $0 |
| P-10 | PDF build + defense slides update + practice | 1 day | $0 |
| **Total** | | **~12-13 days** | **~$50-60** |

Realistic thesis-defensible PDF: **~May 28-30**.

---

## 7. The 5 pending baseline-panel decisions (user must lock before B1-B7)

From `outputs/research/topvenue_panel_2026-05-14.md`:

1. Drop the research-agent's "MiniCheck as 10th composite signal" recommendation (CONFIRMED REDUNDANT — CAEM already uses MiniCheck as primary NLI judge)
2. Substitute AlignScore as companion evaluator? (yes/no)
3. Lock panel of 3 vs panel of 7 baselines? (decide)
4. Defer LongFact + VeriScore to rebuttal phase? (decide)
5. Skip GPT-4o-mini TruthfulQA judge? (decide)

Don't launch B1-B7 until these are locked.

---

## 8. Common pitfalls that have bitten earlier sessions

- **Tier 2 ≠ RAG.** Tier 2 is zero-shot parametric (no retrieval). Tier 3 is RAG. Easy to mix up because both involve generation.
- **Calibration fold ≠ Evaluation fold.** They're disjoint partitions of the per-benchmark sample budget. Sample IDs are content-hash distinct.
- **α (pool purity precondition) ≠ p_c (per-cycle accuracy).** Theorem 4.x distinguishes them sharply.
- **Verifier has 9 signals, NOT 10 or 11.** `h_norm`, `alias_overlap`, `entity_head_consistency` were retired at Phase 1c.
- **L2 anchor is ZERO on shipped LoRA path.** Only active on inactive Full-FT fallback. Don't say "CAEM uses EWC-style L2 anchor" — false.
- **Storage gate is FIXED THRESHOLD, not conformal.** Conformal was retired at Phase 1c P0' after cal-vs-eval distribution shift broke marginal coverage by 15-50 pp.
- **CES is alive, not deprecated.** Some earlier audit suggested removing it; user pushed back. Lives at `eval/metrics.py` (geometric mean over 5 axes: ACC, EPI=1-CHM, RET, CAL, VER).
- **9 hallucination subtypes total, 8 in default CHM.** factual_contradiction is pinned at 0 under MiniCheck's binary judge; excluded from CHM denominator. Documented, not a bug.
- **TriviaQA-test is a retention PROBE, not a training fold.** Even though we train on TriviaQA-train, the TriviaQA-test fold is intentionally held-out as the retention probe.
- **The retention rollback is GOOD news, not a bug.** C4 abort = guard working as designed. Direct empirical receipt for H4 + asymmetric-rollback differentiator.

---

## 9. Closing — once you've read this

Once you've read this file end-to-end, run the live status checks (section 5) to see the current trajectory state, then respond to the user's actual request. Don't summarize this file back at them unless asked — they wrote it.

If you're missing context on something specific, the canonical fallback is **read the code under `caem/`** — it's the source of truth.

Welcome to the project. Have fun.
