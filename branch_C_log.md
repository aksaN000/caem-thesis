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

### 2026-04-22 14:45 BDT  `[DECISION]`  Dual-backbone doctrine — Qwen primary, Flan-T5 preserved legacy

- User question: "keep both or remove T5?" — answered: **keep both, dispatched at single points with shared logic**
- Qwen-3B = primary path (thesis-defended, tested, target of new features)
- Flan-T5 = preserved legacy path (Variant 18 ablation, halted Phase 1a data reuse, regression safety, rollback option)
- Total dispatch overhead ~5-10 lines per file × ~8 files ≈ 100-200 lines across the codebase
- In exchange: Variant 18 enabled, $13 Phase 1a reusable, 560 existing tests stay green, Qwen mid-run issues have immediate fallback
- Discipline: new features Qwen-first; T5 retrofitted only if Variant 18 needs; only blocking T5 bugs get fixed
- Documented in branch_C.md §Dual-backbone doctrine

### 2026-04-22 14:30 BDT  `[IMPL]`  Goal 1 Phase C committed on `feat/qwen-3b-goal1` (`7680885`)

- `caem/confidence/pre_routing.py::_compute_c_conv` dispatches on `model.config.is_encoder_decoder`:
  - Encoder-decoder (Flan-T5): `self.model.encoder(...)` → encoder hidden_states (CAEM-specific adaptation)
  - Decoder-only (Qwen/Gemma/Llama): `self.model(...)` direct forward on query tokens → full decoder-stack hidden_states (canonical Nandakishor 2025 formulation)
- Variance-ratio math identical across both branches; shared-hidden-states unit test proves dispatch is math-agnostic to 1e-6
- `_is_encoder_decoder` helper simplified (removed `_orig_mod` unwrap — unnecessary; also broke MagicMock dispatch)
- Module docstring updated to frame decoder-only as the canonical path, Flan-T5 as the adaptation
- `tests/test_pre_routing.py::TestCConvDecoderOnlyDispatch` (5 mock tests) + `test_c_conv_live_qwen_3b` (live CUDA smoke)
- Live finding documented: raw-text queries to Qwen-3B produce near-zero u_token (~1e-9) because instruction-tuned Qwen expects ChatML; predictions scatter across 151k vocab. Not a bug — Phase D fix (format pre-routing queries with ChatML). Test asserts finiteness + boundedness only, not signal quality.
- **Test results: 6/6 new PASS; 582/582 pass total; zero regression**

### 2026-04-22 13:00 BDT  `[IMPL]`  Goal 1 Phase B committed on `feat/qwen-3b-goal1` (`9697a7b`)

- `caem/prompts.py` (new, 400 lines): centralizes Tier 2 (no-RAG) + Tier 3 (RAG) prompt construction with `prompt_style` dispatch
- `chatml_scaffold`: system + example-user + example-assistant + real-user + generation-prompt marker; Reasoning: prefix applied as prefill by caller
- `flan_t5_scaffold`: bit-identical to legacy `pipeline.py::_build_tier2_prompt` + `rag.py::_build_prompt`; preserved for Variant-18 flan_t5_large_backbone ablation reproducibility
- Shared internals: `detect_query_task`, `_task_spec` (Tier-2/3 wording switch via `with_passages`), `_few_shot_parts`
- `tests/test_prompts.py` (new, 16 tests): 4 task-detection + 4 Flan-T5 Tier-2 per-task (incl. bit-identity) + 2 Flan-T5 Tier-3 (incl. bit-identity) + 6 ChatML structure/dispatch
- pipeline.py / rag.py NOT yet migrated to call prompts.py — that wiring is the next commit, keeping this one independently reviewable
- **Test results: 16/16 new PASS; 576/576 pass total (560 existing + 16 new); zero regression**
- Eyeball check: Qwen tokenizer renders correct ChatML (system + Obama-example-turn-pair + real-claim-turn + `<|im_start|>assistant\n`); appending "Reasoning:" and tokenizing → last decoded pair = `'assistant\nReasoning:'` (correct prefill)

### 2026-04-22 12:15 BDT  `[IMPL]`  Goal 1 Phase A committed on `feat/qwen-3b-goal1` (`2fa08bb`)

- `caem/config.py`: added `base_model_name="Qwen/Qwen2.5-3B-Instruct"`, `prompt_style="chatml_scaffold"`, Goal 5 perf flags (`use_flash_attention_2`, `use_torch_compile`), LoRA config (r=16, alpha=32, dropout=0.05), per-architecture LoRA target-module tuples
- `caem/model_loader.py` (new, 180 lines): `load_base_generator()` architecture-dispatching loader (AutoModelForCausalLM vs AutoModelForSeq2SeqLM via `HfConfig.is_encoder_decoder`), Flash Attention 2 opt-in w/ graceful fallback, torch.compile opt-in (skipped for encoder-decoder), `tokenizer.pad_token = eos_token` for decoder-only, import-time `RAYON_NUM_THREADS` hook (prevents Rust-tokenizer panic on high-core-count hosts), `dtype=` kwarg w/ `torch_dtype=` legacy fallback
- `tests/test_model_loader.py` (new, 95 lines): 4 tests covering rayon-thread hook, Qwen-3B decoder-only live load + forward-pass + greedy ("capital of France is" → " Paris"), Flan-T5-Large encoder-decoder legacy-path load + generate, CAEMConfig Branch-C defaults
- Live smoke on RTX 5090: Qwen-3B loads in 3.0s / 6.29 GB VRAM / 3.08B params ✓; Flan-T5-Large loads in 1.7s / 1.62 GB VRAM / 783M params ✓
- **Test results: 4/4 new tests PASS; 560/560 existing tests PASS; zero regression**

### 2026-04-22 11:00 BDT  `[DECISION]`  Initial Phase 1a redefined — Branch C IS the new Phase 1a

- User decision: stop Flan-T5 Phase 1a; Branch C IS Phase 1a; Phase 1 Full = Phase 1a + Steps 16-18
- Turned out runner had already halted itself at Step 7.0.2 (path bug, see NOTE below)
- Killed `phase1a` and `watcher` tmux sessions (no-op; already exited)
- No additional compute burned; Flan-T5 ~$13 already spent bought complete Cycle-0 data

### 2026-04-22 10:50 BDT  `[IMPL]`  Flan-T5 halted state archived to HF

- Uploaded 8.63 MB to `aksaN000/caem-passage-index-21m/phase_1a_flan_t5_halted/`
- Contents: Step 6 seed (600 episodes), Step 7.0 Cycle-0 eval (6 benches × 500), calibration fold,
  memory/deferred snapshots, MMLU baseline, run logs, MANIFEST.json
- Usable as reference for Variant 18 `flan_t5_large_backbone` ablation row (after Branch C
  main run completes and the revalidation pass re-scores these triples through the 7-family composite)

### 2026-04-22 10:30 BDT  `[NOTE]`  Phase 1a interim interpretation — 6 axes computed

See `branch_C.md` §Why branch C exists and 2026-04-22 entry for full findings. Headlines:

- **Cycle-0 EM dramatically below published Flan-T5 ZS**: FEVER 21.0% (vs 55-60%), TriviaQA 0.0% (vs 40-45%), NQ 0.0% (vs 25-35%), TruthfulQA 13.8% (vs ~20%), StrategyQA 34.6% (vs 55-65%), ARC 26.2% (vs 35-45%). CAEM-scaffolded prompt + Flan-T5 is a broken substrate for the thesis headline. **Reinforces Qwen-3B decision.**
- **u_stored distribution pooled**: P40=0.421, P70=0.561, P90=0.686 — would be the fitted thresholds. Per-benchmark P70 spread [0.512, 0.652] = 14pp; moderate heterogeneity worth per-benchmark reporting (Addition 2).
- **Decision mix** at default τ=0.65: STORE rate 5.8-22.2% per bench (design target 30%; confirms calibration is necessary).
- **ρ(u_stored, EM) ≈ 0**: pooled -0.011; STORE-only +0.106. Below gate threshold 0.3. BUT EM is heavily confounded (loops match labels randomly, 13% label extraction failures, paraphrase answers fail EM). Does NOT refute u_stored — refutes "u_stored predicts EM-match." Real gate needs faithfulness labels on Qwen-3B data.
- **Loop contamination in STOREs**: FEVER 52.6%, StrategyQA 56.1%, NQ 22.4%, ARC 11.7%, TriviaQA 10.3%, TruthfulQA 9.1%. Pooled 30%. 85/241 STOREs would enter SIL pool. **Goal 4 loop filter empirically justified.**
- **Non-loop STORE EM**: NQ 0%, TriviaQA 0%, FEVER 8.1%, StrategyQA 16.7%, ARC 22.4%, TruthfulQA 12.5%. Confirms paraphrase-failure-on-open-ended pattern — answers are semantically close but not string-matching gold.

### 2026-04-22 10:00 BDT  `[BUG]`  Runner halted at Step 7.0.2 — path bug

- `run_phase1a.sh:step_7_0_calibrate` passes `--calib_jsons outputs/cycle_0/calibration_fold_samples.json`
- `run_experiment.py` actually writes the file at `outputs/cycle_0/calibration/calibration_fold_samples.json`
- `calibrate_thresholds.py` logged `No files match` and exited; main runner exited too
- Fix: updated path to `outputs/cycle_0/calibration/calibration_fold_samples.json` (commit pending)
- Net effect on current work: runner stopped at 2026-04-21 03:58 BDT; no additional compute wasted after Step 7.0 complete
- Lesson for Branch C runner: add pre-flight path-existence checks before invoking each script

### 2026-04-22 09:00 BDT  `[DECISION]`  Prompt style port: ChatML envelope + scaffolded-CoT semantics preserved

- Flan-T5: flat text with forced decoder_input_ids for "Reasoning:" prefix
- Qwen-3B: ChatML with role tokens (user/assistant/system) + prefill for forced prefix
- Preserved: scaffolded CoT structure (Reasoning → Answer), per-benchmark label instruction, few-shot pattern
- Changed: ChatML envelope (required by Qwen), system prompt added (decoder-only benefit), few-shot as separate turn pairs (Qwen-native instruction-tuning format)
- Thesis framing: "CAEM's scaffolded-CoT design is backbone-agnostic; the ChatML port preserves the design semantically while adapting delivery to decoder-only conventions."

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
