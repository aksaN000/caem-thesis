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

### 2026-04-22 21:00 BDT  `[IMPL]`  In-repo consolidation — registry count, label wrapper, Ch5 transition note

Closes the three in-repo follow-ups flagged in the Goal-5-phase-1 log.

**1. `branch_C.md` ablation registry count updated**

- `§Ablation registry expansion` now reflects the live state: 17
  pre-Branch-C + 2 Branch-C-landed (`no_q_a_relevance`,
  `no_memory_consolidation`) = **19 today**; 4 more planned for the
  main-run integration (`flan_t5_large_backbone`, `bge_hybrid_retriever`,
  `valentin_4signal_verifier`, `kernel_language_entropy_vs_vanilla_se`);
  **23 target at ship-ready**. Table marks landed vs planned explicitly.
- `§Definition of done` count updated from stale "24 ablation variants"
  to "23 ablation variants (reference row + 19 mechanism + 2
  architectural + 1 sensitivity)".
- Source of truth is `caem/ablation/variants.py::VARIANT_REGISTRY`; the
  doc cites `len(VARIANT_REGISTRY)` as the assertion target so thesis
  tables can't silently drift.

**2. `scripts/label_faithfulness.py` — MiniCheck-labelled faithfulness JSON**

- Wraps `caem.verification.load_verifier_judge(backend="minicheck")` to
  produce the `{question: P(supported in [0, 1])}` JSON that
  `scripts/epistemic_gate.py` consumes via `--labels_path`.
- Scoring semantics: `max_p P(passage entails answer)` across retrieved
  passages -- matches the verifier's `p_ground_max` axis so the
  faithfulness label correlates against `u_stored` on a consistent
  semantic scale.
- Two-layer split (same pattern as `perf_baseline.py`): pure-Python
  aggregation (`score_record`, `build_labels`, `_extract_passages`) is
  CPU-testable; the live MiniCheck load is Vast-GPU-gated and runs
  manually when the Qwen-3B Cycle-0 data is ready.
- 16 new tests cover: `_extract_passages` across both on-disk shapes,
  `score_record` max-across-passages / missing-field / clamping / empty-
  answer, `build_labels` key-field / unscoreable-skip / duplicate-key /
  missing-key / custom-key, JSONL loader valid + malformed-skip.

**3. Thesis Ch5 Branch-C transition note**

- Added a scoped `\section*{Note on reported data and Branch C}` at the
  top of `chapters/chapter_5.tex` that explicitly frames the reported
  numerical content as Phase-1a Flan-T5-Large data (6-signal composite,
  17 ablation variants, DPR retrieval, 780M encoder-decoder backbone).
- Forward-references Branch C's migration to Qwen-2.5-3B-Instruct and
  the seven-family / ten-signal composite, lists the four Branch-C
  memory-hygiene mechanisms (loop filter, retro-prune, downgrade,
  consolidation @ 0.92, hit-counter queue), and cites the 19-landed
  /23-target ablation registry.
- Explicit: the Phase-1a revalidation pass is how the backbone +
  composite confound gets resolved; the epistemic gate is a
  pre-integration precondition, not a post-hoc diagnostic.
- Crucially, the Ch5 body is left untouched -- rewriting every
  "Flan-T5-Large" / "nine-signal" / "seventeen variants" reference
  without replacing the underlying numbers would introduce drift of
  exactly the kind the coherence rule prohibits. The transition note
  reframes the chapter's scope so a viva reader has one anchor for the
  forward reference to Branch C; the body's numerical content remains
  internally consistent with the 6-signal / 17-variant / Flan-T5 data
  it reports. Full body replacement happens in the next chapter
  revision, after the Vast Qwen-3B run produces replacement data.

**Regression gate**: **713 passed, 2 skipped** (up from 697; 16 net-new
from `label_faithfulness`).

**Branch C in-repo status**: all CPU-testable work is landed. The
remaining items are Vast-GPU-gated: Qwen-3B Cycle-0 smoke, MiniCheck
label production for the epistemic gate, torch.compile regression
measurement, and the main-run 10-cycle experiment.

### 2026-04-22 20:15 BDT  `[IMPL]`  Goal 5 phase 1 — CPU-testable infrastructure landed

Lands the three Goal-5 pieces that don't need a live GPU to verify,
clearing the way for the Vast perf-tuning sweep to consume them.

**1. `CAEM_PROFILE` env-var scaffolding (`caem/_profile.py`)**

- Tri-mode switch parsed at import time:
  - 0 (default, production) — `section()` is a null context manager;
    zero overhead; profiler libraries never imported.
  - 1 (dev) — NVTX `range_push` / `range_pop` for Nsight Systems
    (~1% overhead).
  - 2 (diagnostic) — adds `torch.profiler.record_function` (~10%
    overhead; never on main experiments).
- Graceful fallback on non-CUDA hosts — a profile-enabled harness must
  never crash the workload being observed. All failure paths degrade
  to the null body.
- Mode 0 import graph never touches torch (short-lived helper scripts
  pay nothing).
- 14 tests cover mode parsing, nesting, exception-propagation, and the
  non-CUDA fallback path.

**2. Adaptive FAISS nprobe for PassageStore**

- `PassageStore.search(query, k, nprobe=None)` temporarily overrides
  the IVF index's nprobe for one call and restores the prior value on
  return. No-op on non-IVF indexes (FlatIP fallback + debug builds).
- Two new `[DES]` config fields:
  - `rag_faiss_nprobe_tier3: Optional[int] = 64` — full-RAG high recall.
  - `rag_faiss_nprobe_tier2: Optional[int] = 16` — memory-hit
    confirmations, cheap probe.
  - Either None falls back to the global `rag_faiss_nprobe`
    (pre-Goal-5 behaviour).
- `TierThreeRAG._retrieve` now threads `cfg.rag_faiss_nprobe_tier3` into
  `search()`. Tier-2 confirmation callers can pass `rag_faiss_nprobe_tier2`
  when that path gets wired during the Vast sweep.
- 3 new tests in `TestPassageStoreAdaptiveNprobe`: override-applied-and-
  restored on a stub IVF index, None-leaves-index-untouched, and
  flat-index-override-is-noop for FlatIP fallbacks.

**3. Perf harness skeleton (`scripts/perf_baseline.py` + tests)**

- Clean two-layer split:
  - *Statistics layer*: `TimingRecorder`, `summarise`, `_percentile`,
    `PerfRow`, CSV writer. Pure Python, CPU-testable, no CUDA dep.
  - *GPU sampler*: `GPUSampler` polls nvidia-smi on a background
    thread; `poll_fn` is injectable so tests mock it, and on a
    CPU-only CI host the sampler yields zero samples + NaN summary
    rather than crashing.
- Writes to `outputs/perf_log.csv` with a header + one row per
  batch-size measurement. `--smoke` runs a deterministic synthetic
  workload so the CSV writer and sampler lifecycle can be verified
  without loading a real model.
- 19 new tests: percentile edge cases, `summarise` NaN-on-empty,
  single-observation handling, `TimingRecorder` exception-propagation,
  GPU sampler with mocked polls, CSV header-skip-on-append, metadata
  JSON round-trip, end-to-end smoke write.

**Regression gate**: **697 passed, 2 skipped** (up from 661; 36 net-new).

**What still belongs to Goal 5 but needs the Vast run**:

- Live-model `scripts/perf_baseline.py` arm (model loading + tier dispatch
  + GPU-util measurement under real workloads).
- `torch.compile` on inference forward passes — wiring is in place via
  `cfg.use_torch_compile`, but the regression gate (≥10% improvement vs
  ≤2% any-regression) must be measured on hardware.
- Remaining bs=1 call sites audit.
- KV-cache prefix sharing for K semantic-entropy samples.
- `scripts/aggregate_perf.py` (reads `outputs/perf_log.csv` to compute
  per-optimization deltas) -- planned after the first Vast run produces
  baseline rows.

### 2026-04-22 19:30 BDT  `[IMPL]`  Goal 4 item 5 — hit-counter forced re-verification

Closes the last Goal-4 slot. The popular-but-wrong failure mode compounds
retrieval harm: a stored entry served to many Tier-1 queries costs more per
hallucination than a long-tail entry. Hit-counter queueing gives the cycle-
boundary retroverify scheduler a priority lane to catch high-pressure
entries early, independent of their age.

- `caem/config.py`: new `[DES]` field
  `hit_counter_force_retroverify: int = 10`. Set to 0 to disable the
  priority queue.
- `caem/memory/entry.py`: new `EpisodicEntry.hit_counter: int = 0`.
  Distinct from `retrieval_count` (cumulative across all tiers, never
  reset) -- this field tracks Tier-1 serves *since the last retroverify*
  and is reset on each re-score.
- `caem/memory/store.py`:
  - `update_retrieval_stats` — single source-of-truth serve point for
    Tier-1 hits (called from `pipeline._update_tier1_stats`). Increments
    `hit_counter` by 1 alongside `retrieval_count`.
  - `retroverify` — resets `hit_counter` to 0 on every re-scored entry
    (both upgrade and no-op-tie paths). Pruned entries lose the counter
    with themselves (no special handling needed).
  - `consolidate` — sums `hit_counter` across cluster members onto the
    representative (closes the `TODO(goal-4-item-5)` from item 3). Sum is
    the right semantic: the consolidated rep inherits ALL Tier-1 serve
    pressure, so it reaches the force queue as aggressively as the most-
    served member would have.
  - New method `force_retroverify_queue(hit_threshold=None)` returns
    `List[int]` of entry_ids crossed the threshold, sorted by descending
    pressure with deterministic entry_id tiebreak. Caller can iterate
    to run a between-cycle forced-retroverify fast path, or pass the
    list as prioritised input to the next cycle-boundary sweep.

- `tests/test_episodic_memory.py::TestHitCounter` (9 new tests):
  default-zero, increments-on-serve (3 calls → counter=3), reset-on-
  upgrade-retroverify, reset-on-tie-path, consolidation-sums-across-
  cluster, force-queue-returns-crossed-entries (sorted by pressure),
  threshold-override, config-driven default, prune-path-disposes-counter-
  with-entry.
- Regression gate: **661 passed, 2 skipped** (up from 652; 9 net-new).

**Goal 4 is now fully closed** (items 1-5 all landed on `feat/memory-hygiene`):

| Item | Mechanism | Status |
|---|---|---|
| 1 | SIL-pool loop filter | ✅ |
| 2 | Retroverify loop prune | ✅ |
| 3 | Memory consolidation @ SBERT 0.92 (two review rounds) | ✅ |
| 4 | Retroverify downgrade | ✅ |
| 5 | Hit-counter forced re-verification | ✅ |

Next merge step: integrate `feat/memory-hygiene` into `feat/phase-2-all`
after the epistemic gate clears on Qwen-3B Cycle-0 data.

### 2026-04-22 19:00 BDT  `[IMPL]`  Goal 4 item 3 — memory consolidation @ SBERT 0.92 (two design-review rounds)

Lands the cycle-boundary consolidation pass with TWO rounds of pre-merge
review incorporated. Initial design proposed in branch_C.md flagged four
failure modes; round-two review added six more refinements. Final
implementation bakes in every one.

**Behaviour**

`EpisodicMemoryStore.consolidate(cycle_num=None, audit_log_path=None,
qa_embed_fn=None)`:

1. **Cycle-0 protection**: `cycle_num < 1` is a no-op (cold-start memory
   stays diverse; first pass runs at Cycle 1 → Cycle 2). `None` runs
   unconditionally for back-compat with single-call scripts.
2. **Pre-split by pool**: entries partition into TRAINING / TRANSFER /
   UNTAGGED pools before clustering. Union-find is restricted to
   within-pool pairs; cross-pool neighbours are rejected with
   `SKIP_CROSS_POOL` audit records. Guarantees SIL training-pool purity
   by construction rather than by downstream gate.
3. **Candidate clustering** via FAISS top-k neighbour search on query-
   side SBERT embeddings, threshold `cfg.consolidation_similarity_threshold`
   (default 0.92, raised from 0.88 after round-one review).
4. **Safety guards** per candidate cluster:
   - **Answer consistency**: reuses `eval.metrics.normalise` (the SAME
     normaliser that drives EM scoring) — prevents consolidation equality
     from drifting apart from EM equality across refactors. Reason code
     `SKIP_ANSWER_MISMATCH`.
   - **u_stored spread** > `cfg.consolidation_max_u_spread` (0.15) →
     `SKIP_U_SPREAD_EXCEEDED`.
5. **Merge formulas** (explicit; round-one review flagged the under-
   specification):
   - `u_stored`: representative's max (NEVER averaged)
   - `retrieval_count`: sum across cluster
   - `success_rate`: retrieval-count-weighted average
   - `source_benchmark`: representative's tag preserved
   - `merged_source_benchmarks`: tuple union of cluster members' tags
     minus the rep's own (within-pool only after pre-split)
   - `retroverified`: OR across members (+ TODO for a future timestamp
     field to take max-cycle instead)
   - TODO(goal-4-item-5) for `hit_counter` sum when Goal 4 item 5 lands
6. **Deterministic tiebreaker** `(u_stored, -storage_cycle, -entry_id)`
   — highest u wins, oldest cycle breaks ties (most-observed-across-
   cycles member), lowest eid is the final fallback. Guards against
   FAISS-ordering-dependent non-determinism that would break checkpoint
   reproducibility.
7. **Audit log JSONL** when `audit_log_path` is supplied: every cluster
   decision (merge / skip_answer_mismatch / skip_u_spread /
   skip_cross_pool) appends one record with machine-readable enum codes.
   Ch1 review can grep the `outcome` / `reason` field directly.
8. **qa_embed_fn hook** reserved for the joint (query, answer) embedding
   upgrade (branch_C_log Cycle-1 audit follow-up), typed as
   `Optional[Callable]` with `None` default so no callsite churn is
   needed now.

**Cycle ordering**: SIL `run_cycle` runs **retroverify → deferred
reconsideration → consolidation** in that order. Retroverify operates at
the per-entry level first (downgrade + prune); deferred reconsideration
promotes newly-confident buffered entries; consolidation then clusters
the settled surviving set. Documented explicitly in the `run_cycle`
docstring.

**Config additions** (all `[DES]` with docstrings in `caem/config.py`):

- `enable_consolidation: bool = True` — master toggle.
- `consolidation_similarity_threshold: float = 0.92` — tightened from
  0.88 proposal after review.
- `consolidation_search_k: int = 20` — FAISS top-k depth.
- `consolidation_max_u_spread: float = 0.15` — no-merge guardrail.

**Schema additions**

- `EpisodicEntry.merged_source_benchmarks: Tuple[str, ...] = ()` — tuple
  union of merged-in benchmark tags. Populated by consolidation;
  consumed by the SIL training-pool gate (defense-in-depth alongside the
  pre-split).

**SIL gate update**

- `_collect_episodes` now computes the effective benchmark set as
  `{source_benchmark} ∪ merged_source_benchmarks` and excludes the entry
  if the set intersects `TRANSFER_BENCHMARKS`. Pre-split makes this case
  unreachable in production consolidation; the gate remains as defense-
  in-depth.

**Ablation**

- `no_memory_consolidation` variant registered (#18 in the registry),
  mutation flips `cfg.enable_consolidation` to False. Chapter 5 can
  report the marginal-contribution delta.

**Tests** (`tests/test_episodic_memory.py`)

Three new test classes (21 new tests) plus a shared `_near_dup_pair`
helper with deterministic known pairwise cosine similarity:

- `TestConsolidate` (8 tests): empty-store no-op, singletons untouched,
  near-dup pair collapses, retrieval metadata merged, config gate
  disables, breakdown attribute populated, threshold override strict,
  tiebreaker determinism.
- `TestConsolidateSafetyGuards` (7 tests): answer-mismatch skip,
  normalised-equality semantics (punctuation/case), u_spread guard skip
  at default, u_spread guard configurable, merged_source_benchmarks
  populated, SIL-gate defense-in-depth for manually-mixed entries,
  audit log uses enum skip reasons.
- `TestConsolidatePreSplitAndCycleGuard` (5 tests): Cycle-0 protected,
  Cycle-1 runs, None-runs-unconditionally, cross-pool pair never merges,
  within-pool pair merges + no OOD tag leaks onto rep.
- `TestConsolidateTiebreaker` (2 tests): u-then-cycle-then-eid ordering,
  final eid fallback.

**Regression gate**: **652 passed, 2 skipped** (up from 629; 23 net-new).

**Honesty note for Ch5**: the "20-30% memory size reduction over 10
cycles" was a planning ceiling. Empirical reduction depends on
benchmark-mix duplication rate; under 30% STORE rate + 25% duplication
the actual number is closer to 7-8%. Chapter 5 should report the
measured number from Branch-C data rather than the planning claim.
Mechanism's load-bearing claims stay (a) FAISS retrieval latency,
(b) Tier-1 top-k noise reduction, (c) retrieval-history concentration
on the representative.

Items 5 (hit-counter forced re-verification) remains open on
`feat/memory-hygiene`. Items 1, 2, 3, 4 now closed.

### 2026-04-22 18:15 BDT  `[IMPL]`  Goal 4 items 2 + 4 — retroverify loop-prune + downgrade

- `caem/memory/store.py::retroverify`:
  - **Item 2 (loop-prune)**: before re-scoring, if `is_repetitive_loop`
    trips on the stored `reasoning_chain`, the entry is removed outright
    — no verifier forward pass spent. Gated via
    `cfg.retroverify_prune_loops` (default True). Loop-pruned and
    threshold-pruned counts are logged separately; the 2-tuple return
    shape `(n_updated, n_removed)` is preserved so existing SIL and
    diagnostics call sites stay intact.
  - **Item 4 (downgrade)**: new u_stored below the stored value (but
    above the prune threshold) now downgrades the stored value and
    overwrites the nine-signal block with the fresh verifier view
    (previously the raise-only path left stale confident entries in
    memory for cycles). Gated via `cfg.retroverify_allow_downgrade`
    (default True).
  - New transient diagnostic attribute `self._last_retroverify_breakdown`
    = `{"loop_pruned": int, "threshold_pruned": int}` so per-cycle CSV
    loggers can record the split without parsing the log line.
  - Fixed a potential circular import (`caem.memory.store` →
    `caem.training.loop_filter` → `caem.config`, but
    `caem.training.__init__` eagerly imports SelfImprovementLoop which
    imports caem.memory.store) by deferring the `is_repetitive_loop`
    import to inside the method.
- `caem/config.py`: new `[DES]` fields `retroverify_prune_loops` and
  `retroverify_allow_downgrade`, both default True. Both can be flipped
  off to recover the pre-Goal-4 behaviour for ablations / reproductions.
- `tests/test_episodic_memory.py`:
  - `test_retroverify_does_not_downgrade` renamed + rewritten as
    `test_retroverify_downgrades_by_default`.
  - New `test_retroverify_respects_allow_downgrade_false` guards the
    pre-Goal-4 back-compat path.
  - New `test_retroverify_loop_prunes_by_default` — loop-contaminated
    entries are removed and the verifier is NOT called on them.
  - New `test_retroverify_respects_prune_loops_false` — back-compat path.
  - New `test_retroverify_loop_prune_counts_separately_from_threshold` —
    breakdown-diagnostic attribute carries the split.
- Regression gate: **629 passed, 2 skipped** (up from 625; 4 net-new).

Items 3 (consolidation @ SBERT-cluster 0.88) and 5 (hit-counter forced
re-verification) remain open on the `feat/memory-hygiene` slot.

### 2026-04-22 17:45 BDT  `[IMPL]`  Epistemic gate — `scripts/epistemic_gate.py`

Pre-integration Moskvoretskii ρ gate wired. Branch_C.md §Evaluation protocol
makes this a hard precondition for the `feat/phase-2-all` integration merge:
if pooled Spearman ρ(u_stored, is_faithful) ≤ 0.3 on Qwen-3B Cycle-0 data,
the Branch-C main-run compute is blocked until the correlation defect is
diagnosed. Implementing now so the gate run itself is just a data-in call
once Qwen-3B Cycle-0 finishes on Vast.

- `scripts/epistemic_gate.py` (new, ~375 lines):
  - Pure-Python Spearman (avg-rank tiebreaker, no SciPy dependency) + 1000-
    resample bootstrap 95% CI.
  - Three-way decision tree (PROCEED / PROCEED_WITH_HONESTY / STOP) keyed off
    the pooled ρ. Exit code propagates the STOP outcome so CI can block on
    the gate directly.
  - Loads `per_sample_signals.jsonl` (already produced by the main run) +
    a labels JSON mapping question → faithfulness label in [0, 1]. Graded
    labels from MiniCheck's unary P(supported) are accepted alongside
    binary labels.
  - Emits `epistemic_gate_report.json` (machine-readable) + `.md`
    (human-readable for the Ch5 appendix).
- `tests/test_epistemic_gate.py` (new, 32 tests):
  - Rank-data ties, perfect/inverted/uncorrelated correlations, bootstrap
    CI ordering + seed determinism, decision-threshold boundaries, record/
    label pairing with missing keys, multi-benchmark pooling (including a
    regression guard against "optimise away rank ties" that would overstate
    ρ by ignoring the statistically-correct Spearman convention).
  - `tmp_path`-based round-trip for the JSON + Markdown writers.
- Smoke-run: end-to-end via `main(argv=[...])` produces `PROCEED` on a
  perfectly-correlated synthetic dataset, exit code 0.
- Regression gate: **625 passed, 2 skipped** (up from 593; 32 new tests).

The companion faithfulness-labelling step (MiniCheck re-score on a held-out
subset) is NOT in this script by design — keeping the statistics + decision
logic free of a live-GPU dependency. A dedicated `scripts/label_faithfulness.py`
wrapper over MiniCheck (or multi-judge cross-check) will feed the labels
JSON and can live behind a `--judge_backend` flag when the Vast run is
scheduled.

### 2026-04-22 17:00 BDT  `[IMPL]`  Ablation variant `no_q_a_relevance` + composite-rescale bug fix

Closes Goal 2 by registering the Ch5 contribution-ablation variant. Also
catches and fixes a weights-sum-to-1.14 bug introduced when Goal 2's
q_a_relevance weight was added without updating the existing rescale
mutations — they knew only about the six legacy weights and would leave
q_a_relevance at 0.14 after zeroing another signal.

- `caem/ablation/variants.py`:
  - New `_U_STORED_WEIGHT_FIELDS` tuple + `_rescale_u_stored_weights_excluding`
    helper. Every `_mut_no_*` rescale now routes through this single path, so
    adding an eighth signal in future phases only requires updating the tuple
    (no per-mutation edits).
  - `_mut_no_grounding`, `_mut_no_internal_calibration`, `_mut_no_semantic_entropy`
    rewritten to use the helper; all three now correctly include q_a_relevance
    in the rescale and the kept weights sum to exactly 1.0 (regression fix).
  - `_mut_equal_signal_weights` now flattens to 1/7 (was 1/6); sources field
    list from the shared tuple.
  - New `_mut_no_q_a_relevance`: zero the 0.14 q_a_relevance weight, pro-rata
    rescale the six legacy weights. Registered under `mechanism_tag="verification"`,
    `needs_cyclic_rerun=True`.
- `tests/test_ablation.py`:
  - `test_weights_sum_to_one` parametrisation extended with `no_q_a_relevance`;
    all four zero-and-rescale variants now verified.
  - `test_no_grounding_zeros_pground_only` gains a companion assertion that
    `q_a_relevance` stays non-zero (it's a question-answer relevance signal,
    not external grounding).
  - New `test_no_q_a_relevance_zeros_qa_weight`,
    `test_no_q_a_relevance_preserves_branch_c_ratios`,
    `test_equal_signal_weights_uses_seven_families`.
- Regression gate: **593 passed, 2 skipped** (up from 588; 5 new tests).

Ablation registry count: 17 variants (was 16 pre-Goal-2 + 1 new). Ch5
ablation-table caption and registry-count cells in branch_C.md will need
an explicit "seventeen named ablation variants" pass when the chapter
edits land.

### 2026-04-22 16:30 BDT  `[IMPL]`  Goal 2 — q_a_relevance signal (seven-family composite)

Closes the sample-② failure mode from Phase-1a Cycle 0: hallucinated off-topic
answers scored high on `p_entail` + `p_ground_max` because the retrieved
passage happened to match the hallucination rather than the question. None of
the existing signals check question↔answer relevance — q_a_relevance adds
that orthogonal axis.

- `caem/verification/verifier.py`:
  - `UnifiedVerifier.__init__` accepts a `qa_relevance_scorer` injection
    (any duck-typed object with `.predict(List[Tuple[str, str]]) -> np.ndarray`
    — sentence-transformers CrossEncoder, FlagReranker, or a thin wrapper).
  - `_compute_q_a_relevance(query, answer)` clips to [0, 1]; returns 0.5 when
    the scorer is absent, raises, emits NaN, or sees an empty q/a. Matches
    p_entail's neutral-absence-fallback convention.
  - `_composite` gains a `q_a_relevance` keyword with default 0.5 so legacy
    callers run unchanged; weight sourced via `getattr` for pre-Goal-2 configs.
  - `verify()` threads q_a_relevance into both the early-exit-returned and
    happy-path UnifiedVerifierOutput instances; debug log line extended.
- `caem/memory/entry.py`: `EpisodicEntry.q_a_relevance: float = 0.5` —
  mutable quality metadata alongside p_contra.
- `caem/memory/store.py::retroverify`: updates `q_a_relevance` when u_stored
  rises (same branch as the other signals).
- `caem/pipeline.py`: new-episode store path copies `vout.q_a_relevance`
  into the EpisodicEntry.
- `caem/config.py`: composite rebalanced to seven weights summing to 1.0
  (pg_mean 0.28, pg_atom 0.14, nli 0.16, **q_a_relevance 0.14**, sc 0.14,
  uinternal 0.10, se 0.04). The Session-42 six-weight baseline was
  (0.30, 0.15, 0.15, —, 0.15, 0.15, 0.10); the q_a_relevance mass comes
  out of pg_mean, pg_atom, sc, uinternal, and se pro-rata per the Goal-2
  plan in branch_C.md.
- `tests/test_verifier.py`: composite tests now assert the seven-weight
  sum + thread `q_a_relevance` through uniform/zero/monotone coverage;
  `_blank_verifier` sets `qa_relevance_scorer=None` on the isolation shim.
- `tests/test_self_improvement.py::make_entry`: scalar-only path sets
  `q_a_relevance=v`; vout-copy path propagates `vout.q_a_relevance`. The
  u_stored=v invariant holds under the seven-weight composite.
- `tests/test_qa_relevance.py` (new): 18 tests covering scorer absence,
  clipping, exception/NaN fallback, empty-input neutrality, monotonicity,
  self-consistency invariant, EpisodicEntry and UnifiedVerifierOutput schemas.
- Regression gate: **588 passed, 2 skipped** (570 + 18 new).

The `no_q_a_relevance` ablation variant (zero weight, redistribute 0.14
across the six other signals) and the actual BGE / MiniLM cross-encoder
wiring on the runtime paths remain on the `feat/q-a-relevance` slot for
post-integration.

### 2026-04-22 15:30 BDT  `[IMPL]`  Goal 4 item 1 — loop filter in SIL _collect_episodes

Phase-1a Cycle-0 data showed **~30% of STOREd samples were severe repetition
loops** (FEVER 52%, StrategyQA 56%; 82/241 stored samples had distinct-4 ≤ 0.13).
Unfiltered, these chains poison the SIL training target with
"yes yes yes ..." supervision. This entry closes the loop-filter slot.

- `caem/training/loop_filter.py` — new module. `is_repetitive_loop(text, cfg)`
  combines two independent signals (OR-logic, matches contradiction-veto
  convention):
  1. Distinct-4 n-gram ratio < `cfg.loop_distinct4_threshold` (default 0.25)
  2. zlib compression ratio < `cfg.loop_compression_threshold` (default 0.35)
  Chains shorter than `cfg.loop_min_tokens` (default 20) are exempt — short
  verified answers legitimately score low on distinct-n.
- `caem/config.py` — three new `[DES]` fields (`loop_distinct4_threshold`,
  `loop_compression_threshold`, `loop_min_tokens`) with docstrings referencing
  the phase-1a empirical basis.
- `caem/training/self_improvement.py::_collect_episodes` — adds a fourth
  fallback branch: loops fall back to `entry.answer` (same as empty / too-
  short chains). New diagnostic counter `n_loop_filtered` surfaces in
  `_log_chain_diagnostics`.
- `tests/test_loop_filter.py` — 15 new tests covering primitive signals,
  integration with SIL, short-text exemption, and config-driven thresholds.
- Regression gate: **570 passed, 2 skipped** (555 + 15 new).

Other Goal 4 items (retroverify loop prune, consolidation @ 0.88, retroverify
downgrade, hit-counter) remain future work on `feat/memory-hygiene`.

### 2026-04-22 15:00 BDT  `[IMPL]`  Cleanup — rename `_pooled_sample_t5*` → `_pooled_sample*`

Deferred cleanup from Phase D4. The methods are now backbone-agnostic
(decoder-only since D4); the `_t5` suffix was stale. Rename via `sed -i`
across `caem/verification/verifier.py`, `tests/test_verifier.py`, and
`CH_AUDIT_TRACKING.md`. Docstrings refreshed to drop T5-era wording.
51 verifier tests pass post-rename.

### 2026-04-22 14:30 BDT  `[IMPL]`  Phase D6 + D7 — baselines + scripts migrated to Qwen ChatML

Phase D6 (`eval/baselines.py`):

- Replaced `T5ForConditionalGeneration` load with `load_base_generator(cfg.base_model_name)`;
  defaults now flow from `CAEMConfig.base_model_name` (Qwen-3B).
- New `BaselineBase._wrap_chatml_user` helper + `_run_generation` slices
  `output_ids[0, input_len:]` (was `output_ids[0]` for T5's decoder-only-returning generate).
- B1 `ZeroShotBaseline` / B2 `CoTBaseline`: single-turn ChatML user messages
  (no few-shot, no system prompt — clean ablation against CAEM's scaffolded CoT).
- B3 `RAGBaseline`: delegates to `TierThreeRAG.generate()` (already decoder-only
  after Phase D1).
- B4 `CoTRAGBaseline`: uses `build_tier3_prompt` + prepends CoT trigger to the
  forced prefill (was injecting before the `\nAnswer:` cue on the flat T5 prompt,
  which no longer exists in the ChatML layout).
- B5 `FLAREBaseline._look_ahead` rewritten for decoder-only: ChatML wrap + use
  `committed` as assistant prefill; slice `out.sequences[0, input_len:]` (was
  `out.sequences[0, 1:]`, which was T5-only). New `_grounded_generate` helper
  builds the grounded regeneration prompt via `build_tier3_prompt`. `retrieve()`
  now consumes `TierThreeRAG`'s public `(text, score)` tuples (was dict form).
- `tests/test_flare_smoke.py` rewritten for Qwen-3B; still env-gated behind
  `RUN_FLARE_SMOKE=1`, still excluded from the default suite.

Phase D7 (8 scripts):

- `scripts/run_experiment.py`, `run_ablation.py`, `run_baseline.py`,
  `run_simple_ft.py`, `run_purity_validation.py`, `seed_cold_start.py`,
  `cycle2_retention_diagnostic.py`, `check_base_model.py` — all T5 hardcodes
  (`T5ForConditionalGeneration` + `google/flan-t5-large` string literals)
  replaced with `load_base_generator(CAEMConfig().base_model_name)`.
- CLI `--model_name` / `--model` defaults changed from `"google/flan-t5-large"`
  to `None` → resolved at call site from `CAEMConfig.base_model_name`.
- `run_simple_ft.py::_mmlu_accuracy` + `_rationalise_pool` get ChatML wrap
  + decoder-only slice (left-padding for batched generation, `pad_token_id`
  passed to `generate`).
- `cycle2_retention_diagnostic.py::_evaluate` loads base weights via
  `load_base_generator` and overlays the cycle-2 state_dict; ChatML wrap +
  slice for its per-row `model.generate` call.
- `check_base_model.py::generate_answer` gets ChatML wrap + slice.
- `run_purity_validation.py` per-cycle pipeline factory now calls
  `load_base_generator` once per cycle (replacing the T5 weights-reuse
  pattern) so cycle-N checkpoints load on top of the correct architecture.
- Regression gate: **555 passed, 2 skipped**; all 8 scripts parse cleanly
  under `ast.parse`. No runtime exercise on Vast yet (guarded by cost).

### 2026-04-22 13:30 BDT  `[IMPL]`  Phase D5 — SIL migrated to Qwen causal LM + Full FT + 8-bit AdamW

- Rewrote `caem/training/self_improvement.py` for decoder-only teacher-forcing:
  prompt = `build_tier2_prompt(question) + "Reasoning:"`; target = reasoning_chain with
  the forced-prefix stripped + EOS; labels mask all prompt positions with -100
- `_build_optimizer`: bitsandbytes `AdamW8bit` when `cfg.use_8bit_adamw=True` and
  CUDA is present; graceful fallback to `torch.optim.AdamW` on ImportError / CPU /
  init failure (keeps CPU-only CI & unit tests on the fallback path)
- L2 anchor unchanged: `L = L_task + (λ/2)||θ − θ_base||²` with λ=0.01. Snapshot
  stays fp32 on CPU (or GPU when VRAM ≥ 24 GB) per existing optimisation
- Gradient checkpointing enabled inside `_finetune` (`use_cache` saved/restored);
  silently no-ops on mock models
- `_mmlu_score` rewritten for causal LM: ChatML-wraps the MMLU prompt via
  `_wrap_chatml_user`, slices `out[0, input_len:]` before decode so the letter-
  prefix match is evaluated on the continuation (not the echoed prompt)
- `_forgetting_score` given the same slice treatment (still deprecated — MMLU
  remains the abort guard)
- Module docstring documents the Full FT → LoRA → memory-only cascade and cites
  Song 2025 + GeRe for L2 anchoring validation
- `tests/test_model_loader.py::test_config_defaults_branch_c` updated to assert
  `use_8bit_adamw is True` and `use_lora_training is False` (doctrine flip)
- `tests/test_self_improvement.py::make_mock_tokenizer` stubs `apply_chat_template`
  so the decoder-only training path exercises the real ChatML branch
- Regression gate: **555 passed, 2 skipped** on the full suite

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
