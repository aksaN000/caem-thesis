# Branch C — Continuous Log

Reverse-chronological log of branch-C work. Each entry is dated (BDT) and tagged.
Tags:

- `[DECISION]` — a design/scope/process decision locked or revised
- `[IMPL]` — code change landed
- `[PERF]` — performance measurement or optimization
- `[BUG]` — bug found and fixed (or diagnosed)
- `[BLOCKER]` — waiting on external signal
- `[GATE]` — a pre-registered gate passed or failed
- `[NOTE]` — context, observation, or reference

Keep entries short (ideally one sentence + one line). Link commits by short SHA.
Detail belongs in the commit message; the log is for quick rewind.

---

## 2026-04-22 (BDT — date rolls based on activity)

### 2026-04-21 03:00 BDT  `[DECISION]`  Goal 5 optimal decisions locked

- GPU util target: **70-80% sustained**, 90%+ bursts; not 85%+ target (hurts Tier 1 latency)
- `torch.compile` on inference forward passes (generator + verifier); NOT on training loop (LoRA needs determinism)
- Regression gate: ≥10% improvement on targeted metric AND no regression >2% on any other tracked metric
- Goal 5 branch ordering: after Goals 1/4/2, before `feat/phase-2-all` integration
- Profiling: three-mode via `CAEM_PROFILE` env var (0/1/2 for prod/dev/diagnostic)

### 2026-04-21 02:45 BDT  `[DECISION]`  Nine review issues from user applied to `branch_C.md`

- Signal-count framing unified to "7-family composite covering 10 underlying signals
  across seven decorrelated axes" — canonical phrase linked from Ch4/Ch5/A3
- Moskvoretskii self-knowledge correlation promoted from metric to **pre-integration
  epistemic gate** (ρ>0.5 proceed, 0.3<ρ≤0.5 proceed with honesty, ρ≤0.3 STOP)
- Variants 23 and 24 (A-MEM / Adaptive-RAG as ablations) removed; they stay only as
  baselines B8/B9. Registry: 24 → 22 variants
- Phase 1a ablation row backbone+composite confound resolved via revalidation pass
  (re-score existing generated triples through 7-family composite; 1 day, ~$2)
- Ch2 7-topic expansion: 3 → 6 days realistic
- Variant 21 Valentin 4-signal: +3 days engineering booked at integration
- α-sensitivity figure added to outputs artifact list (Addition 1)
- Loop-filter thresholds declared as config params
  (`loop_distinct4_threshold=0.25`, `loop_compression_threshold=0.35`,
  `loop_min_tokens=20`), not hardcoded; defaults cited to Phase 1a Cycle-0 analysis
- Appendix C five confirmations resolved inline (MMLU as full row not footnote; A3
  → Ch4; Ch4-only grep; supervisor briefing blocker before Goal 1; A-MEM + Adaptive-RAG
  ported from public repos with 2-day budget each)

### 2026-04-21 02:00 BDT  `[DECISION]`  Verifier-ensemble dropped; MiniCheck + q_a_relevance

- Adding Gemma/AlignScore dilutes CAEM's specific contribution (ensemble is generic ML
  practice since 1990s; not novel)
- Adding `q_a_relevance` as ONE new signal (BGE cross-encoder on question↔answer)
  is a specific, identifiable CAEM contribution (closes sample-② failure mode)
- Composite: 7 families, 10 underlying signals (unchanged framing from pre-Branch-C)
- If Step 19 on current Phase 1a shows α < 0.65 on any benchmark, ensemble becomes
  a Phase-3 post-hoc additive patch (not a day-1 commitment)

### 2026-04-20 23:00 BDT  `[DECISION]`  Goal 3 (retrieval upgrade) demoted to ablation-only

- Examiner objection risk: if main-run uses BGE+hybrid and baselines use DPR, "how
  much of the gain is CAEM vs just better recall?" is unanswerable
- Main-run retriever stays DPR (matches Phase 1a baseline data, fair comparison)
- BGE+BM25+reranker+FLARE runs as Variant 19 ablation only
- Baselines B3/B4/B5 stay on DPR

### 2026-04-20 22:30 BDT  `[DECISION]`  Base generator: Qwen2.5-3B-Instruct (not 7B)

- 7B's ~85% FEVER Cycle-0 leaves only ~7pp SIL headroom → headline reads as noise-level
- 3B's ~70% Cycle-0 leaves ~18pp → compelling SIL demonstration
- Both have ~99% label compliance and <2% loop rates (modern instruction tuning)
- 3B at ~16 GB bf16 VRAM leaves comfortable room for MiniCheck + BGE retriever +
  embedder on 5090 32GB
- LoRA rank-16 SIL training mandatory (full FT needs ~48 GB VRAM, won't fit)
- Phase 1a Flan-T5 data preserved as Variant 18 `flan_t5_large_backbone` ablation row

### 2026-04-20 21:15 BDT  `[NOTE]`  Literature review 2 pulled from GitHub (commit `48dd806`)

- Confirms simpler design choice (no ensemble needed — Valentin 2024 is closest competitor)
- Adds mandatory Moskvoretskii self-knowledge correlation eval (critical risk flag)
- Adds A-MEM (Xu 2025) + Adaptive-RAG (Jeong 2024) as primary baselines to defend
  CAEM's Topic 3/5 novelty claims
- Adds LM-Polygraph (Vashurin 2025) as standard UQ harness for prediction-rejection curves
- Adds foundational citations (Fu 2025, Song 2024, Das 2025, Huang 2025, Condorcet)
- Three-axis novelty spine crystallizes: (i) external multi-signal verifier, (ii)
  generative setting, (iii) finite-round α > ½ ⇒ P > p — distinguishes from RF-1 Das 2025

### 2026-04-20 20:00 BDT  `[NOTE]`  FIX-8 applied (α-parametric calibration-sensitivity analysis)

- `scripts/run_purity_validation.py` now dumps per-sample scalars per (bench, cycle)
- New `scripts/calibration_alpha_curve.py` replays Stage-5 decision tree at a grid
  of candidate τ_store values; produces α(τ) curves + 2 PDF figures
- Sync'd TPR/TNR zero-denominator convention to match main purity script (0.0 default)
- All cross-refs verified across Ch4 + Ch5 edits

### 2026-04-20 19:00 BDT  `[IMPL]`  `branch_C.md` pushed (commit `d330e3f`)

- 724-line planning document integrating 2026-04-20/21 planning + lit-review-2 findings
- Documents research-contribution spine, design decisions, evaluation protocol,
  theoretical framing, branch structure, calibration discipline, chapter edit roadmap,
  ablation registry, baseline panel, timeline, risk register, definition of done
- Initial version lacked signal-count reconciliation, epistemic gate, loop-filter
  config params — all fixed in 2026-04-21 02:45 entry above

### 2026-04-20 18:00 BDT  `[NOTE]`  Phase 1a run on Flan-T5-Large still in Step 7.0 Cycle-0 eval

- 5 of 6 benchmarks completed (FEVER/TriviaQA/NQ/TruthfulQA/StrategyQA)
- ARC-Challenge remaining (~299 samples, ~30-45 min)
- Cumulative decisions (Step 6 + Step 7.0): STORE 865 / DEFERRED 828 / ABSTAIN 745 / DISCARD 1549
- Step 6 final: 600 episodes stored (200/bench × 3) at τ_store=0.45 override
- Observed loop rate: ~30% of STOREd samples are severe repetition loops
  (distinct-4 < 0.25), concentrated in binary-label benches (FEVER 52%, StrategyQA 56%)
- Observed sample-② failure: hallucinated content with p_entail=0.91, p_ground_max=0.94
  (verifier fooled because passage matched the hallucination, not the question)
- Both observations motivate Branch C's Goals 2 and 4

---

## How to update this log

- Append new entries at the TOP of the most recent date section
- Each entry: timestamp (BDT) + tag + one-sentence title + optional 2-5 bullets
- Use commit short SHA when referencing committed work
- Use file paths + line numbers when referencing code touch points
- Keep it honest: log blockers and regressions too, not just wins
- Commit this file to `main` alongside other branch-C work; it's the thesis's
  research diary for the Phase 2 upgrade window
