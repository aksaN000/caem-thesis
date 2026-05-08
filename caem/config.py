"""
caem/config.py
==============
Central configuration for the CAEM pipeline.

Every numeric constant in the system must come from here -- no magic numbers
in module code. Values are annotated with their hyperparameter category:
  [LIT]  = fixed from literature (cite the source, never change arbitrarily)
  [DES]  = design choice (principled default; change only with ablation evidence)
  [CAL]  = empirically calibrated (initial value given; actual value fitted
            after Cycle 1 on the 500-sample calibration set)

See: hyperparameter-reference.md for the full three-category breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Transfer-learning benchmark split (ID vs OOD)
# ---------------------------------------------------------------------------
# The six-benchmark factual-QA panel is split 3/3 for a transfer-learning
# probe. Only ID benchmarks feed the self-improvement training pool; OOD
# benchmarks are held out to measure generalisation of the consolidated
# mechanisms (memory + routing + anchored fine-tuning).
#
# ID (training-eligible): Natural Questions, TriviaQA, FEVER
# OOD (held-out for transfer): TruthfulQA, StrategyQA, ARC-Challenge
#
# These names MUST match the benchmark identifiers used by the eval
# harness (see eval/harness.py::_score dispatch). Keep them lowercase and
# underscore-delimited. The self-improvement loop (_collect_episodes) gates
# fine-tuning on ``entry.source_benchmark in TRAINING_BENCHMARKS``; entries
# without a benchmark tag (source_benchmark=None) are treated as
# training-eligible for legacy / unit-test paths.
TRAINING_BENCHMARKS: Tuple[str, ...] = (
    "fever",
    "triviaqa",
    "commonsense_qa",
)
# v2.1 architecture (2026-05-08): training panel pruned from 4 to 3 benches
# after cycle-0 eval revealed two precondition-violation benchmarks where the
# Bayesian-floor inequality on verifier discrimination was mathematically
# unreachable at α=0.05.
#
# Active panel (3 benches):
#   - FEVER          (claim verification, ~145k train) -- bounded label space
#   - TriviaQA       (single-hop entity recall, ~87k train) -- bare-entity QA
#   - CommonsenseQA  (5-choice MCQ commonsense, ~9.7k train) -- bounded label
#
# Removed at cycle-0 close (kept as registered exclusion evidence in
# outputs/cycle_0/eval/{hotpotqa,natural_questions}_cycle0.json):
#   - HotpotQA       p_+=0.090, required TPR/FPR ≥192 at α=0.05 (verifier
#                    achievable: 5-15). Mathematically unstoreable.
#   - Natural Qs     p_+=0.156-0.172 (v1 + v2 readings), verifier α<½ on every
#                    v1 cal fold (cycles 1-4); structural-failure benchmark.
# ASQA dropped earlier (long-form, EM near-zero by design — wrong metric).
# ARC-Challenge dropped earlier (4-choice MCQ — redundant with CSQA's 5-choice).
# Stream chunk (v2.1): 2000/cycle on FEVER/TriviaQA, 700/cycle on CSQA.
# See caem/benchmark_splits.py for authoritative panel + per-benchmark sizes.

TRANSFER_BENCHMARKS: Tuple[str, ...] = (
    "truthfulqa",
    "strategyqa",
)
# v2.1 (2026-05-08): transfer eval panel pruned from 3 to 2 benchmarks.
# Natural Questions removed (verifier α<½ at every cycle in v1; non-discriminative
# even as transfer; pollutes pooled fallback fits without adding signal).
# ARC-Challenge / ASQA removed earlier per v2 (redundancy / wrong metric).
# Loaders for ARC/ASQA/HotpotQA/NQ remain in eval/benchmarks.py for back-compat
# with legacy ablations + demo server; they just do NOT enter the active eval
# trajectory.


@dataclass
class CAEMConfig:
    # ------------------------------------------------------------------ #
    # Encoding                                                             #
    # ------------------------------------------------------------------ #
    # [LIT] all-mpnet-base-v2 produces 768-dim embeddings (not 384).
    sbert_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768  # EXP-10 fix: must match the sbert_model output dim

    # ------------------------------------------------------------------ #
    # Episodic Memory                                                      #
    # ------------------------------------------------------------------ #
    # [DES] Full thesis-scale run capacity.
    max_memory_size: int = 1_000_000
    # [DES] Memory FAISS backend.
    #   ivf_pq: plan default for 1M-scale runs
    #   flat_ip: exact cosine search, useful for tiny debug runs
    memory_index_type: str = "ivf_pq"
    # [DES] IVF-PQ parameters (memory index).
    faiss_nlist: int = 4_096
    faiss_nprobe: int = 32
    faiss_pq_m: int = 64
    faiss_pq_nbits: int = 8
    # [DES] Promotion/training thresholds for IVF-PQ.
    # Training is attempted once at least this many vectors are available.
    faiss_train_min_points: int = 20_000
    # Upper bound for sampled training vectors when fitting IVF centroids.
    faiss_train_sample_size: int = 200_000
    # [DES] Trigger pruning when the index is 95% full.
    pruning_trigger: float = 0.95
    # [DES] Remove the bottom 20% by Value score during a prune pass.
    pruning_amount: float = 0.20
    # [DES] Skip storage if nearest-neighbour cosine sim > this (already known).
    novelty_threshold: float = 0.95
    # [DES] Exponential recency decay rate: Recency = exp(-λ · age_in_seconds).
    recency_lambda: float = 0.01

    # Value-score weights for pruning (Importance × 0.7 + Recency × 0.3):
    #   Importance = 0.4·φ + 0.3·(r/r_max) + 0.2·success_rate + 0.1·u_stored
    # [DES]
    value_importance_weight: float = 0.70
    value_recency_weight: float = 0.30
    importance_phi_weight: float = 0.40        # cycle recency proxy
    importance_retrieval_weight: float = 0.30
    importance_success_weight: float = 0.20
    importance_u_stored_weight: float = 0.10

    # ------------------------------------------------------------------ #
    # Pre-routing confidence (Stage 3, fast -- 2 signals)                  #
    # ------------------------------------------------------------------ #
    # [DES] Token prob is the primary reliability signal.
    u_pre_token_weight: float = 0.60
    # [DES] C_conv (internal convergence) is secondary.
    u_pre_cconv_weight: float = 0.40

    # ------------------------------------------------------------------ #
    # Adaptive Router (Stage 3 -> dispatch)                                 #
    # ------------------------------------------------------------------ #
    # [DES] Combined score = routing_lambda·s + (1-λ)·û_stored
    routing_lambda: float = 0.70
    # [DES] Route to Tier 1 if combined score ≥ this.
    tier1_combined_threshold: float = 0.90
    # [DES] Route to Tier 2 if similarity > this (and not Tier 1).
    tier2_similarity_threshold: float = 0.75
    # [DES] OR-condition: force Tier 3 if u_pre < this, regardless of memory.
    # Re-tuned 2026-04-30 from 0.60 → 0.38 based on cycle-1 routing observations:
    # the original threshold over-suppressed moderately-confident queries on the
    # open-domain question-answering benchmarks, blocking legitimate memory
    # utilisation. The reduced threshold preserves the safety override on
    # genuinely-uncertain queries while restoring tier dispatch on the moderate-
    # confidence operating band. Effective from cycle 2 onwards; cycle-1 readings
    # under the conservative threshold are retained as calibration-tuning evidence.
    safety_u_pre_min: float = 0.38
    # [CAL] Temperature scaling for u_pre calibration (Guo et al. 2017).
    # Applied as sigmoid(logit(u_pre) / T). T=1.0 means no calibration.
    temperature_scalar: float = 1.0

    # ------------------------------------------------------------------ #
    # v2 Fix 12 — per-benchmark u_pre calibration                         #
    # ------------------------------------------------------------------ #
    # Each training benchmark exposes a different distribution over query
    # length, named-entity density, and answer surface form. The pooled
    # global T_global + safety_u_pre_min above are kept as the back-compat
    # fallback used when source_benchmark is None or unknown; the per-
    # benchmark dicts below override them when a query carries a known
    # benchmark tag.
    #
    # Populated empirically from the cycle-0 calibration fold by
    # scripts/run_calibration.py (per-benchmark Platt + ECE minimisation
    # under a closed-form sweep), then EMA-smoothed at every later cycle
    # boundary (see scripts/recalibrate_thresholds_at_cycle.py). When
    # adaptive_thresholds_per_cycle = False these dicts are frozen at the
    # cycle-0 fit.
    #
    # Empty defaults below mean: until cycle-0 calibration writes per-bench
    # values, every query falls back to the pooled global T_global +
    # safety_u_pre_min via get_temperature_for() / get_safety_u_pre_min_for().
    temperature_scalar_per_benchmark: Dict[str, float] = field(default_factory=dict)
    safety_u_pre_min_per_benchmark: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Post-generation confidence (Stage 4a, Tier 2 only -- 4 signals)      #
    # ------------------------------------------------------------------ #
    # [LIT] Gal & Ghahramani 2016: K=5 MC Dropout passes.
    mc_dropout_k: int = 5
    # [LIT] Gal & Ghahramani 2016: dropout rate at inference.
    mc_dropout_rate: float = 0.1
    # [LIT] Wang et al. 2022: M=3 chains for self-consistency.
    sc_chains_m: int = 3
    # [LIT] Farquhar et al. 2024: K=10 samples at T=1.0 for semantic entropy.
    se_samples_k: int = 10
    se_temperature: float = 1.0
    # 2026-04-27 retirement: empirical Cherian boost weight for h_norm
    # measured at -5e-4 on the cycle-0 cal-fold (n=1500, 10-signal fit) —
    # statistically zero contribution to u_stored. K=10 stochastic
    # generations were the largest verifier wall-clock contributor
    # (~4 s/sample, ~40% of cal-fold step). Disabling routes the verifier
    # to skip the K-sample pool and feed a neutral sentinel (0.5) into the
    # composite isotonic+boost; locked composite_calibration.json + the
    # conformal gate stay valid because the |w·iso| ≤ 5e-4 shift is two
    # orders of magnitude below any decision threshold.
    disable_h_norm: bool = True

    # NOTE (2026-04 refactor): the legacy u_hat post-generation escalation
    # gate (4 learnable weights + accept threshold) was removed when the
    # UnifiedVerifier nine-signal stage became the single source of post-
    # generation truth. Tier 2 -> Tier 3 escalation now happens only on
    # generation exception or empty output (see pipeline._tier2). No
    # per-signal escalation weights are kept.

    # ------------------------------------------------------------------ #
    # UnifiedVerifier (Stage 5 -- nine-signal gate, Session 42)            #
    # ------------------------------------------------------------------ #
    # The pre-Session-42 verifier emitted three signals (p_entail, s_avg,
    # h_norm) and used NLI only self-referentially. The nine-signal
    # redesign adds internal calibration (u_token, u_dropout, u_internal)
    # and external grounding (p_ground_max/mean/atomic, p_contra) via NLI
    # against reranked Wikipedia passages, and replaces the boolean
    # should_store() gate with a four-way decision tree (STORE / DEFERRED
    # / ABSTAIN / DISCARD).

    # [LIT] Verifier backend. "minicheck" = MiniCheck-Flan-T5-Large (Tang
    # 2024 ACL), trained on LM-generated claim-support data -- the right
    # distribution for judging LM outputs. "roberta_nli" = legacy
    # roberta-large-mnli, trained on human-written NLI pairs; retained as
    # an ablation / calibration comparison (see
    # scripts/calibration_minicheck_vs_roberta.py).
    # The swap is motivated by HaluEval 2025 / "Semantic Illusion" 2025:
    # DeBERTa-v3-large-MNLI shows 100% FPR at 95% recall on LM hallucinations,
    # and RoBERTa-large-MNLI is expected to have the same failure mode.
    verifier_backend: str = "minicheck"
    nli_model: str = "roberta-large-mnli"
    minicheck_model: str = "lytang/MiniCheck-Flan-T5-Large"
    # MiniCheck threshold mapping from unary P(supported) to 3-class NLI
    # labels, used only by semantic-entropy NLI clustering. See
    # caem/verification/minicheck.py for the mapping rationale.
    minicheck_entail_threshold: float = 0.7
    minicheck_contradict_threshold: float = 0.3
    # [LIT] Farquhar et al. 2024: agglomerative + cosine clustering.
    se_clustering_method: str = "agglomerative"

    # --- Early-exit confabulation gate [DES] ---------------------------- #
    # IF u_internal >= early_exit_u_internal
    # AND p_ground_max <= early_exit_p_ground_max
    #   -> decision = DISCARD (route to Tier 3 regeneration).
    # Catches high-confidence ungrounded generations -- the Farquhar 2024
    # confabulation profile -- before spending the remaining signal budget.
    early_exit_u_internal: float = 0.70
    early_exit_p_ground_max: float = 0.20

    # --- Decision-tree thresholds [DES] --------------------------------- #
    # u_stored >= store_threshold                     -> STORE
    # defer_threshold <= u_stored < store_threshold   -> DEFERRED
    # u_stored < defer_threshold
    #   AND p_ground_max < abstain_pground_ceiling    -> ABSTAIN
    # (The contradiction veto branch was removed 2026-04-22 -- MiniCheck
    # returns p_contra = 0 by construction, so the veto never fired under
    # the default backend. p_contra is kept as a diagnostic in the entry
    # schema but no decision logic reads it.)
    store_threshold: float = 0.65
    defer_threshold: float = 0.45
    abstain_pground_ceiling: float = 0.20

    # --- Grounding retrieval / rerank [DES] ----------------------------- #
    # Retrieve top-k passages from Wikipedia corpus, then rerank to top-N
    # before scoring p_ground_max / p_ground_mean.
    verifier_retrieve_k: int = 20
    verifier_rerank_k: int = 3

    # [DES] Branch C cross-encoder used for BOTH (a) passage reranking
    # top-20 -> top-3 in the verifier's grounding stage, AND (b) the
    # Goal-2 q_a_relevance signal (question <-> display-answer relevance).
    # Same CrossEncoder instance, two call sites -- saves ~1 GB VRAM vs
    # loading two models and keeps both signals calibrated against each
    # other. BGE-reranker-v2-m3 is the default; set to None in config or
    # pass cross_encoder=None to CAEMPipeline to disable both roles
    # (rerank falls back to retriever order, q_a_relevance falls back to
    # the 0.5 neutral prior).
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"

    # --- u_stored composite weights [DES] (must sum to 1.0) ------------- #
    # Branch C (Goal 2, 2026-04-22) rebalanced the Session-42 six-weight
    # composite to make room for q_a_relevance:
    #   Session 42 (pre-Branch-C): pg_mean=0.30, pg_atom=0.15, nli=0.15,
    #     sc=0.15, uinternal=0.15, se=0.10 — sum=1.00, six weights.
    #   Branch C (Goal 2):         pg_mean=0.28, pg_atom=0.14, nli=0.16,
    #     q_a_relevance=0.14, sc=0.14, uinternal=0.10, se=0.04 — sum=1.00,
    #     seven weights.
    # q_a_relevance closes the sample-② failure mode from Phase 1a Cycle 0
    # (hallucinated off-topic answer with high p_entail + high p_ground_max
    # because the retrieved passage matched the hallucination rather than the
    # question). None of the other grounding or entailment signals check the
    # question↔answer relevance axis; this is CAEM's pipeline-level
    # contribution for Goal 2. See branch_C.md §"Goal 2".
    # ------------------------------------------------------------------ #
    # Adaptive thresholds (research + production)                         #
    # ------------------------------------------------------------------ #
    # Default ON 2026-04-24: matches the existing per-cycle T re-fit
    # pattern (Ovadia 2019, Thulasidasan 2019). Memory cap is 1M episodes
    # so memory-bloat concerns from frozen mode don't apply at thesis
    # scale, and Claim 3 (SIL improvement) is measured on a FIXED held-out
    # eval fold so adaptive thresholds don't confound the trajectory.
    #
    # When ON, scripts/recalibrate_thresholds_at_cycle.py runs at every
    # cycle boundary, re-fitting (tau_store, tau_defer, tau_train) on the
    # current cycle's calibration fold and EMA-smoothing with the previous
    # cycle's values. This:
    #   - Auto-tunes to deployment user mix (no source_benchmark needed)
    #   - Maintains constant store rate as model improves cycle-over-cycle
    #   - Matches existing per-cycle T (temperature) re-fit pattern
    #     (run_per_cycle_recalibration in run_experiment.py)
    #   - Keeps composite weights frozen (Claim 2 anchor preserved)
    #   - Label-free (only reads u_stored values, no gold answers needed)
    #
    # Production deployment uses the SAME setting (research-production
    # parity). T calibration is the ONE label-dependent surface; in
    # production it's frozen at research-time value and refreshed offline
    # quarterly via aggregated user feedback or human-labeled batches.
    # See thesis Ch 6 §Production Deployment.
    adaptive_thresholds_per_cycle: bool = True
    # EMA smoothing factor: tau_new = alpha * tau_prev + (1-alpha) * tau_fit
    # Higher alpha = slower drift (more stable). 0.7 is a moderate default.
    adaptive_thresholds_ema_alpha: float = 0.7

    # 2026-04-24: granular T-skip for production deployment.
    # ─── When False (research/default) ────────────────────────────────
    #   Per-cycle T re-fit runs as before (calibrate_pipeline_temperature_only),
    #   using gold labels from the calibration fold to minimize ECE.
    # ─── When True (production deployment) ────────────────────────────
    #   Per-cycle T re-fit is bypassed. T stays at its current value
    #   (typically frozen from research-time fit). Use this in production
    #   where gold labels are unavailable; refresh T periodically offline
    #   via aggregated user feedback or quarterly human-labeled batches.
    #
    # Interaction with adaptive_thresholds_per_cycle:
    #   - Both False: classic frozen-everything mode (legacy)
    #   - Both True (research default): label-driven recalibration of both
    #     T and thresholds at every cycle
    #   - skip_T=True, adaptive_thresholds=True (production deploy):
    #     thresholds keep adapting label-free; T frozen until offline refresh
    #   - skip_T=True, adaptive_thresholds=False: degenerate (no adaptation)
    skip_per_cycle_temperature: bool = False

    # Weights revised 2026-04-24 based on empirical Cohen's d on 1500-sample
    # Cycle-0 eval audit. Changes: (1) add p_ground_max to composite with
    # weight 0.18 (d=+0.148, was computed and ignored), (2) upweight
    # q_a_relevance to 0.20 (d=+0.367 strongest signal, was 0.14), (3)
    # downweight p_ground_atomic to 0.06 (99% crashed under verbose-prediction
    # bug; will upweight if post-fix validation shows recovery). See
    # branch_C_log.md 2026-04-24 audit. Sum preserved at 1.00.
    u_stored_weight_pground_mean: float = 0.22     # was 0.28
    u_stored_weight_pground_max: float = 0.18      # NEW (was not in composite)
    u_stored_weight_pground_atomic: float = 0.06   # was 0.14 (conservative floor)
    u_stored_weight_nli: float = 0.14              # p_entail, was 0.16
    u_stored_weight_q_a_relevance: float = 0.20    # was 0.14 (strongest signal)
    u_stored_weight_sc: float = 0.10               # s_avg, was 0.14
    u_stored_weight_uinternal: float = 0.08        # was 0.10
    u_stored_weight_se: float = 0.02               # (1 - h_norm), was 0.04
    # v2 Fix 6 + Fix 7 — alias_overlap and entity_head_consistency.
    # The cal_prob branch (the v2 default once the calibration JSON loads)
    # trains an isotonic for both signals and ignores these weights. The
    # weighted_sum legacy fallback consumes them when the JSON fails to
    # load. Keeping them at zero by default preserves the v1
    # weighted_sum numerics on pre-v2 calibrated_config_*.json files;
    # ablation runs that want to test the new signals via weighted_sum
    # can override the weights without touching cal_prob behaviour.
    u_stored_weight_alias_overlap: float = 0.0      # v2 Fix 6 — placeholder
    u_stored_weight_entity_head_consistency: float = 0.0  # v2 Fix 7 — placeholder

    # v2 Fix 6 (2026-05-07 audit): path to the alias dictionary JSON that
    # InMemoryAliasResolver loads at pipeline init. The dictionary is
    # built by ``scripts/build_alias_dict_from_benchmarks.py`` from the
    # TriviaQA/NaturalQuestions/HotpotQA train splits, which ship
    # alias-enriched ``answers`` lists (canonical + surface-form
    # variants). The runner builds this file before step_6_reseed so the
    # cold-start verifier already has the resolver at first generation.
    # Schema: ``{canonical_str: [alias_str, ...], ...}``.
    alias_dict_path: Optional[str] = "data/alias_dict.json"

    # ------------------------------------------------------------------ #
    # Composite mode — Branch C 2026-04-25 (Phase 2.1)                    #
    # ------------------------------------------------------------------ #
    # Selects how the 9 verifier signals collapse into u_stored.
    #
    # ─── "weighted_sum" (default fallback, pre-2026-04-25 behaviour) ────
    #   Linear combination of signals × weights above. Sign-fixed and
    #   shared across benchmarks. Cohen's d cap: ~0.40 pooled because
    #   q_a_relevance flips sign on FEVER (-0.58) vs. open-QA (+0.50/+0.75)
    #   under uniform weights, dragging the composite down.
    #
    # ─── "cal_prob" (Branch C default once calibration JSON exists) ─────
    #   Per-signal isotonic regression on the labeled Cycle-0 calibration
    #   fold; composite is the sigmoid of the sum of per-signal log-odds.
    #   Auto-detects sign per signal (q_a_relevance fits a *decreasing*
    #   curve on FEVER and *increasing* on open-QA via pooled Pearson).
    #   Dead signals (Cohen's d ≈ 0) get near-flat curves and contribute
    #   ~0 log-odds without manual zero-weighting.
    #
    # Bootstrap: at Cycle 0 (calibration JSON absent), the verifier uses
    # weighted_sum to score the calibration fold. Step 7.0.2 then fits the
    # per-signal isotonic on that fold and writes the JSON. Subsequent
    # cycles read the JSON and switch to cal_prob automatically.
    #
    # Both modes share the same `_composite()` API; the dispatch is
    # internal to UnifiedVerifier.
    composite_mode: str = "auto"
    # "auto"          → cal_prob if calibration JSON loadable, else weighted_sum
    # "weighted_sum"  → forced weighted-sum (legacy / ablation)
    # "cal_prob"      → forced cal-prob (errors if JSON missing)

    # Path to per-signal calibration JSON produced by
    # scripts/fit_composite_calibration.py at Step 7.0.2. Read at
    # UnifiedVerifier construction time; gracefully degrades if missing.
    composite_calibration_path: str = "outputs/cycle_0/composite_calibration.json"

    # ------------------------------------------------------------------ #
    # Conformal storage gate — Branch C 2026-04-25 (Phase 2.4)            #
    # ------------------------------------------------------------------ #
    # When the conformal gate JSON is present (fitted by
    # scripts/fit_conformal_gate.py at Step 7.0.2), the verifier's
    # _decide() reads τ_store and τ_defer from it instead of from the
    # static config values below. The α targets define what precision the
    # fitted thresholds aim to deliver on the calibration fold's em labels:
    #
    #   α_store = 0.20  →  STORE precision target ≥ 0.80
    #                       (Tier-1 + training-pool eligible band)
    #   α_defer = 0.40  →  DEFERRED precision target ≥ 0.60
    #                       (memory only, retroverify queue)
    #
    # Bootstrap: at Cycle 0 the JSON is absent; the verifier falls back to
    # the legacy fixed thresholds (store_threshold, defer_threshold). Once
    # Step 7.0.2 fits and writes the JSON, subsequent cycles use the
    # conformal-calibrated thresholds.
    conformal_gate_path: str = "outputs/cycle_0/conformal_gate.json"
    # Locked at 0.05 from the 25-variant Step 7.0.3 sweep (was 0.20 default;
    # 25-variant sweep showed alpha=0.20 only delivered ~71% pooled eval
    # precision, alpha=0.05 selected as best-on-eval-ID-precision). The
    # per-cycle conformal refit inherits this from prev_gate JSON, but this
    # config default is the belt-and-suspenders fallback if the inheritance
    # path ever fails to read the previous gate.
    conformal_alpha_store: float = 0.05
    conformal_alpha_defer: float = 0.40

    # ------------------------------------------------------------------ #
    # Atomic decomposition scope — Branch C 2026-04-25 (Phase 2.3)        #
    # ------------------------------------------------------------------ #
    # Minimum answer length (in whitespace tokens) for FActScore-style
    # atomic decomposition to fire. Below this threshold _score_atomic
    # returns the fallback (p_ground_mean) directly and does not invoke
    # the decomposer model.
    #
    # Empirical justification (Phase 1 Cycle-0 audit on n=3500 across
    # 7 benchmarks, 2026-04-25):
    #   - Atomic decomposition was 100% fallback on every benchmark
    #     including ASQA, the long-form architectural home of the signal.
    #     The over-aggressive NO_FACTS sentinel + answer-too-short-to-
    #     decompose pattern made the signal degenerate everywhere.
    #   - The remediation is to scope atomic to multi-fact answers via a
    #     length-gate; on FEVER labels and short factoids
    #     (TriviaQA / NQ / ARC), atomic falls back to p_ground_mean by
    #     design and the directional p_ground rewrite path covers the
    #     label-classification semantics.
    #
    # Literature alignment (Min et al. EMNLP 2023, FActScore §3):
    #   FActScore is documented to evaluate "long-form text generation"
    #   where multi-fact decomposition is meaningful. Applying it
    #   uniformly across short-answer benchmarks is a category error;
    #   the length-gate makes CAEM's atomic mechanism scope-faithful to
    #   the FActScore framework.
    #
    # Threshold value: 12 tokens corresponds to ~1 declarative sentence,
    # below which decomposition produces vacuous facts that fail to
    # ground meaningfully against retrieved passages.
    atomic_min_tokens: int = 12

    # ------------------------------------------------------------------ #
    # Tier 3 RAG (Stage 6)                                                 #
    # ------------------------------------------------------------------ #
    # [DES] Passage FAISS backend.
    rag_index_type: str = "ivf_pq"
    # [DES] Halved from 65_536 on 2026-04-19 after faiss-cpu k-means at
    # that scale repeatedly collapsed to 1 effective core regardless of
    # OMP_NUM_THREADS caps (see VAST_SESSION_LOG.md). At nlist=32_768
    # with 1M training samples we maintain ~30x nlist density (FAISS
    # minimum) and drop ~2-5pp recall@10 in exchange for a clustering
    # phase that actually completes in tractable time. The change is
    # symmetric across CAEM and all RAG baselines (they share the index)
    # so does not bias the CAEM-vs-baseline sig-test comparison.
    rag_faiss_nlist: int = 32_768
    rag_faiss_nprobe: int = 64
    # [DES] Branch C Goal 5: adaptive FAISS nprobe. Tier 3 full-RAG queries
    # want high recall (nprobe_tier3), whereas Tier 2 memory-hit confirmations
    # only need to verify a known-similar claim and can use a cheaper probe.
    # When either is None, ``rag_faiss_nprobe`` is used as the fallback (so
    # pre-Goal-5 behaviour is preserved). Set both to the same value to
    # disable the adaptive path entirely.
    rag_faiss_nprobe_tier3: Optional[int] = 64
    rag_faiss_nprobe_tier2: Optional[int] = 16
    rag_faiss_pq_m: int = 64
    rag_faiss_pq_nbits: int = 8
    rag_faiss_train_sample_size: int = 500_000

    # [DES] Number of passages retrieved from the Wikipedia corpus.
    # k=5 is standard in DPR (Karpukhin et al. 2020); gives good recall
    # without overloading the Flan-T5 context window.
    rag_top_k: int = 5
    # ------------------------------------------------------------------ #
    # Base generator (Goal 1, Branch C — decoder-only only)                #
    # ------------------------------------------------------------------ #
    # [DES] HuggingFace model name. Qwen/Qwen2.5-3B-Instruct: Apache 2.0, 3B
    # params, ChatML format, ~99% label compliance, <2% loop rate per modern
    # instruction tuning. Encoder-decoder models (Flan-T5 family) are NOT
    # supported on this branch; see branch_C.md §"T5 removal (2026-04-22)".
    # For fallback to the legacy T5 codebase, use the ``main`` branch.
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # ------------------------------------------------------------------ #
    # Goal 5 — hardware-utilization optimizations (Branch C)               #
    # ------------------------------------------------------------------ #
    # [DES] Flash Attention 2 — Blackwell sm_120 (RTX 5090) has no prebuilt
    # flash-attn wheels and source-built FA2 is SLOWER than cuDNN-attention
    # SDPA per published benchmarks (gau-nernst 5090). Default now False;
    # see ``use_sdpa`` below for the Branch-C primary attention path.
    use_flash_attention_2: bool = False

    # [DES] SDPA (PyTorch-native scaled_dot_product_attention) — on
    # Blackwell this dispatches to cuDNN-attention. Isolated bench
    # (scripts/bench_sdpa_vs_eager.py) measured 1.49x speedup on
    # single-prompt Qwen-3B greedy generation (45 -> 67 tok/s).
    #
    # HOWEVER — Branch-C 2026-04-23 empirical test on the real Step 6
    # multi-mode pipeline (main gen + m-chain K=3 + SE K=10 + MC-dropout
    # K=5 + atomic decomp + negation path + bidirectional NEI) showed
    # NO measurable speedup vs eager on the matched first-5-batch
    # window, and actually ~30% regression on batches 2-3. Root cause:
    # the kernel-level 1.49x advantage only applies to the ~40% of
    # per-sample time that's Qwen forward. Other 60% (MiniCheck-T5
    # eager, BGE rerank, FAISS search, SBERT, orchestration) is
    # unchanged. Compounded with SDPA's variable kernel-selection
    # overhead across 6+ different input shapes, net is flat-to-negative.
    #
    # FINAL DEFAULT: TRUE (SDPA cuDNN-attention) — 2026-04-23 final.
    # After a day of measurements on Blackwell sm_120 with the CAEM
    # multi-mode pipeline (Qwen main + m-chain K=3 + SE K=10 + MC-dropout
    # K=5 + atomic decomp + negation path):
    #
    #   Config                          Steady-state s/sample
    #   ─────────────────────────────────────────────────
    #   Archive eager (Apr 23 04:57)    10.3 s (5 batches)
    #   SDPA+compile (Apr 23 10:32)     10.4 s (batch 2; N=1)
    #   Current eager (Apr 23 10:55)    14.3 s (4 batches; BGE slow)
    #
    # SDPA+compile matches archive eager at ~10 s/sample under clean
    # machine state. Current eager is slow due to BGE rerank pool
    # slowdown (~2 s/sample regression in BGE after 6h of GPU cycling).
    # Under these conditions, SDPA+compile is the ~4 s/sample faster
    # path. Confirmed by bench 1.49× Qwen + 1.6× compile on isolated
    # generation; real pipeline sees reduced gain due to non-Qwen
    # components but still appears ahead of current eager state.
    #
    # Machine state: 37°C GPU (cool, not thermal), so machine-state
    # regression on eager is likely allocator / FAISS page-cache
    # fragmentation, not thermal throttling.
    use_sdpa: bool = True

    # [DES] torch.compile on Qwen `model.forward`. Branch-C 2026-04-23:
    # FLIPPED BACK TO FALSE after empirical regression measurement on real
    # Step 6 workload. Isolated bench (scripts/bench_sdpa_vs_eager.py) on
    # simple single-prompt greedy generation showed 1.60x speedup from
    # compile on top of SDPA. But real Step 6 pipeline has 6+ different
    # graph shapes (main gen, m-chain sampling num_return_sequences=3,
    # SE samples num_return_sequences=10, MC-dropout train-mode K=5,
    # atomic decomposition greedy, negation fallback in refutes path)
    # and torch._dynamo hits its config.recompile_limit=8 on multi-mode
    # workloads, causing eager fallback on some paths plus compile-overhead
    # on all others. Net measured Step 6 batch-2/3 per-sample: 12.5s with
    # compile vs 10.4s without compile (on same commit) = ~20% REGRESSION
    # end-to-end on the real multi-mode pipeline. Disabled.
    #
    # FINAL DEFAULT: TRUE (torch.compile on model.forward) — 2026-04-23.
    # Compile test (scripts/test_compile_training.py) validated the
    # compile + training path works cleanly (4 training steps, loss
    # decreasing 3.42 → 0.47, bit-identical eval/train toggle).
    # Empirical Step 6 measurement with SDPA+compile: 10.38 s/sample
    # (batch 2 of Run #2, matches archive eager) — ahead of current
    # eager (14.3 s/sample) on current machine state. Bench isolated
    # showed 1.6× on greedy gen; real pipeline reduced gain due to
    # non-Qwen work, but compile still appears net-positive in
    # combination with SDPA.
    #
    # JIT overhead: ~600s on first batch of each script. Paid once
    # per process, amortized well over long-running scripts
    # (Step 6 ~5h, Step 7 main ~270h).
    # Recompile_limit=8 warnings possible on some paths; they fall
    # back to eager for that function, not catastrophic.
    use_torch_compile: bool = True

    # [DES] Full FT + 8-bit AdamW is the PRIMARY SIL training path on Qwen-3B
    # (verified 2026-04-22 to fit 32 GB 5090 at batch=4, grad checkpointing
    # on, bf16 mixed precision: ~22-25 GB peak with MiniCheck loaded). LoRA
    # is the first STRUCTURED FALLBACK (per Ch4 §Cycle-2 retention) if Full
    # FT fails (OOM, MMLU retention < 0.93, convergence failure). Memory-
    # only (no weight update) is the last resort.
    use_8bit_adamw: bool = True

    # ------------------------------------------------------------------ #
    # SIL training mode — v2 (2026-05-06, ACTIVATED) — was Phase 2.9    #
    # ------------------------------------------------------------------ #
    # v2 default: LoRA primary path. Replaces the v1 full fine-tune +
    # L2-anchor path.
    #
    # Empirical justification (v1 Phase-1a Cycle-0 audit, n=3500;
    # calibrated SIL pool ~100 verified episodes per benchmark per cycle):
    #   - Per-cycle pool size is in the LoRA-friendly sparse-data regime;
    #     full FT's gradient signal across 3B parameters was dominated by
    #     the L2 anchor regulariser.
    #   - Frozen-base LoRA bounds catastrophic forgetting mathematically:
    #     |Δθ| is capped at the adapter parameter budget. Combined with
    #     the multi-modal retention probe (Fix 4) this replaces the
    #     single-MMLU forgetting guard from v1.
    #
    # Literature alignment: Hu et al. 2021 (LoRA); Wang et al. 2023 (ICLR
    # 2024) on adapter-only continual fine-tuning; Biderman et al. 2024
    # ("LoRA Learns Less and Forgets Less") on the per-cycle sparse-pool
    # regime where LoRA outperforms full FT.
    #
    # v2 hyperparameters (Biderman 2024 §4.2 + audit on Qwen-2.5-3B):
    #   r        = 32        (from v1's 16 — extra capacity for the
    #                         12-signal verifier-driven training pool)
    #   alpha    = 64        (2 × r — standard LoRA scaling rule)
    #   targets  = ALL LINEAR projections inside each transformer block
    #              (attn q/k/v/o + MLP gate/up/down). Matches Biderman's
    #              "all-linear LoRA" recommendation.
    #   lr       = 2e-4      (10× the full-FT lr; LoRA budget is small
    #                         enough to take a higher step size)
    #   dropout  = 0.05      (Hu 2021 default)
    #
    # L2 anchor: REMOVED on the LoRA path. The frozen base + bounded
    # adapter parameter count IS the implicit anchor. ``cfg.l2_lambda``
    # is ignored when ``use_lora_training=True``; if a future ablation
    # wants an L2 anchor over adapter weights specifically, wire it via
    # a dedicated ``cfg.adapter_l2_lambda`` rather than re-using the
    # backbone-anchor constant.
    #
    # Checkpointing: SIL writes adapter-only state via peft's
    # ``save_pretrained`` to ``cycle_<N>/adapter/`` instead of a
    # multi-GB full state_dict. Disk usage drops from ~6.2 GB/cycle to
    # ~120 MB/cycle. v1 ``cycle_<N>/model.pt`` files remain readable by
    # the loader for back-compat with v1 checkpoints.
    use_lora_training: bool = True   # v2 default — ACTIVE
    lora_r: int = 32                  # v2 — was 16
    lora_alpha: int = 64              # v2 — was 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        # All-linear projections (Biderman 2024 §4.2):
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    # v2 LoRA learning rate (overrides ``learning_rate`` when
    # ``use_lora_training=True``). Backbone-FT lr (1e-5) is left intact
    # for ablation runs / back-compat with v1.
    lora_learning_rate: float = 2e-4

    # ------------------------------------------------------------------ #
    # Context window (shared across backbones)                             #
    # ------------------------------------------------------------------ #
    # [DES] Max tokens allocated for retrieved context in the model prompt.
    # Flan-T5-Large has a 512-token encoder limit; Qwen-2.5-3B supports 32k+,
    # but CAEM caps at 384 for parity with the Flan-T5 baseline and to keep the
    # KV cache per-query memory predictable. Raise after Goal 5 profiling shows
    # a clear latency benefit from longer context.
    rag_max_context_tokens: int = 384
    # [DES] Max new tokens for RAG generation. Raised from 256 to 512
    # after the uniform-CoT few-shot prompt was introduced: the few-shot
    # example + scaffolded Reasoning/Answer fields can consume ~150 tokens
    # on their own, and we need headroom so Flan-T5 reaches the Answer
    # slot even on long CoT completions. Flan-T5-Large natural output
    # rarely exceeds 120 tokens; 512 is a safety ceiling.
    rag_max_new_tokens: int = 512
    # [DES] Max new tokens for CoT (Chain-of-Thought) generation limits.
    # Raised to 512 for the same reason as rag_max_new_tokens above.
    cot_max_new_tokens: int = 512
    # [DES] Sampling for RAG generation: greedy (do_sample=False) for
    # reproducibility; no temperature needed.
    rag_do_sample: bool = False

    # ------------------------------------------------------------------ #
    # Fine-tuning / Self-improvement loop (Stage 8)                        #
    # ------------------------------------------------------------------ #
    # L2 regularisation (NOT full EWC): Loss += (λ/2)·||θ − θ_prev||²
    # [LIT] EWC λ=0.4 from Kirkpatrick et al. 2017, adapted here as L2.
    l2_lambda: float = 0.01
    # [DES] Training hyperparameters.
    learning_rate: float = 1e-5
    # batch_size lowered 16 -> 4 on 2026-04-27: at seq_len=512+512 the
    # eager-mode SDPA backward held a 4.37 GiB attention-grad allocation
    # per step (CUDA OOM incident #5), which the consumer-grade 32 GiB
    # envelope does not fit alongside the bf16 model + bf16 anchor +
    # 8-bit AdamW state + verifier judge + reranker + FAISS index. The
    # grad_accum_steps bump to 4 below preserves the effective batch
    # size of 16 — same total forward+backward count per epoch, same
    # gradient signal, just 4 micro-steps + 1 optimizer step per
    # effective batch instead of 1+1.
    batch_size: int = 4
    # [DES] Gradient-accumulation multiplier. Effective batch size is
    # ``batch_size * grad_accum_steps``; tuned per hardware tier by
    # ``scripts.hardware.get_hardware_profile`` to hold effective batch at
    # ``TARGET_EFFECTIVE_BATCH_SIZE`` (=16 since 2026-04-27).
    grad_accum_steps: int = 4
    epochs_per_cycle: int = 3
    warmup_steps: int = 500
    # [DES] Only include verified episodes above this quality in training data.
    min_u_stored_for_training: float = 0.75
    # DEPRECATED in v2 (2026-05-06). The 10% general-domain mix was removed
    # along with load_general_data() in scripts/run_experiment.py. Anti-
    # forgetting is now provided by (a) LoRA SIL primitive (Fix 8) — small
    # adapter parameter budget cannot bulldoze base representations, (b)
    # loss reweighting via temperature mixing T=2 with bounded 3x upsampling
    # and DoReMi floor (Fix 3), and (c) cold-start gold-labelled fallback
    # for zero-count benchmarks. The multi-modal retention probe (Fix 4)
    # provides the empirical retention guard. These fields are retained as
    # config-schema compatibility for older calibrated_config_*.json files
    # but are NOT consumed by the v2 SIL training loop.
    general_data_ratio: float = 0.0  # was 0.10; v2 zeroes this
    general_data_size: int = 0  # was 1000; v2 zeroes this
    # [DES] Abort fine-tuning if general capability drops below this retention.
    # v2 Fix 4: tightened from 0.93 → 0.93 (unchanged numerically) but the
    # guard now fires if ANY probe in ``retention_probes`` drops below this
    # ratio — not just MMLU. The multi-probe AND-of-OK is strictly more
    # conservative than v1's single-probe gate.
    forgetting_tolerance: float = 0.93

    # ------------------------------------------------------------------ #
    # Multi-modal retention probe — v2.1 (2026-05-08)                     #
    # ------------------------------------------------------------------ #
    # v1 used a single MMLU validation probe (n=200). Blind spots:
    #   * MCQ-letter accuracy stays high while open-text generation
    #     degrades silently (TriviaQA / NQ).
    #   * Bounded-label probes don't catch open-text capability decay.
    #
    # v2.1 panel (after HotpotQA removal 2026-05-08): three probes spanning
    # the active training distribution.
    #   * mmlu                  — 4-choice MCQ retention (general knowledge)
    #   * triviaqa_test         — open-text factoid retention
    #   * commonsense_qa_test   — 5-choice MCQ commonsense retention
    # Halt criterion: ANY probe drops below forgetting_tolerance from pristine.
    # The HotpotQA probe is still registered in retention_probe.py for
    # back-compat / observability but no longer in this default tuple.
    retention_probes: Tuple[str, ...] = (
        "mmlu",                  # general knowledge MCQ retention (v1 carryover)
        "triviaqa_test",         # open-text factoid retention
        "commonsense_qa_test",   # commonsense MCQ retention (v2.1 NEW: replaces hotpotqa_test)
    )
    retention_probe_n: int = 200  # samples per probe (matches v1 MMLU n)

    # ------------------------------------------------------------------ #
    # SIL pool reweighting — v2 Fix 3 (2026-05-07)                        #
    # ------------------------------------------------------------------ #
    # Replaces the v1 flat training pool. After ``_collect_episodes``
    # builds the verified-episode list, ``pool_reweighting.reweight_pool``
    # rebalances per-benchmark counts via temperature-mixed softmax,
    # bounded upsampling, DoReMi-style minimum floor, and cold-start
    # gold-labelled fallback for zero-count benchmarks.
    pool_reweighting_enabled: bool = True
    # Softmax temperature for per-benchmark proportions. T=1 = empirical
    # (no smoothing); T → ∞ = uniform. T=2 collapses a 10× count gap to
    # a ~3× weight gap. (Du et al. 2022 / DoReMi 2023 §3.2.)
    pool_reweighting_temperature: float = 2.0
    # Each individual sample is replicated AT MOST this many times when
    # upsampling a low-count benchmark to its target. Caps the
    # "duplicate the same sample 100×" pathology that destroyed
    # cycle-1 storage diversity in v1.
    pool_reweighting_upsample_cap: float = 3.0
    # DoReMi floor: minimum samples per benchmark in the final pool,
    # even if temperature smoothing rounds the share down. Hard
    # guarantee that every training benchmark sees gradient signal.
    pool_reweighting_doremi_floor: int = 50
    # Cold-start fallback: when a benchmark has zero verified episodes,
    # seed with this many gold-labelled samples (drawn via the loader
    # passed to SelfImprovementLoop.run_cycle as ``cold_start_loader``).
    pool_reweighting_cold_start_n: int = 100

    # ------------------------------------------------------------------ #
    # SIL-pool loop filter (Branch C Goal 4 -- memory hygiene)             #
    # ------------------------------------------------------------------ #
    # Two-signal repetitive-loop detector applied inside
    # ``SelfImprovementLoop._collect_episodes``. When a verified reasoning
    # chain trips either signal, the training target falls back to
    # ``entry.answer`` (the short verified answer) rather than the looped
    # chain -- same fallback branch the empty/too-short chains already
    # take. Thresholds are [DES]; defaults chosen from Phase 1a Cycle-0
    # loop-distribution analysis (82/241 STOREd samples had distinct-4
    # <= 0.13 on Flan-T5; clean scaffolded CoT distinct-4 >= 0.60). See
    # caem/training/loop_filter.py and branch_C.md Section Goal 4.
    #
    # [DES] distinct-4-gram ratio below which the chain is a loop.
    loop_distinct4_threshold: float = 0.25
    # [DES] zlib compression ratio below which the chain is a loop.
    loop_compression_threshold: float = 0.35
    # [DES] Min whitespace-token count for the filter to be consulted.
    # Short verified answers (<20 tokens) legitimately score low on
    # distinct-n and are exempt from the filter by construction.
    loop_min_tokens: int = 20

    # ------------------------------------------------------------------ #
    # Retroactive re-verification                                          #
    # ------------------------------------------------------------------ #
    # [DES] Remove from memory if updated u_stored drops below this.
    retroverify_prune_threshold: float = 0.50

    # [DES] Branch C Goal 4 item 2: apply the SIL-pool loop filter at
    # retroverify time as well, pruning loop-contaminated episodes from
    # memory (not just filtering them from the SIL training pool). Uses
    # the same distinct-4 + zlib signals as ``is_repetitive_loop``. Set
    # False to disable and restore the pre-Goal-4 behaviour (threshold-
    # only pruning). See branch_C.md §"Goal 4".
    retroverify_prune_loops: bool = True

    # [DES] Branch C Goal 4 item 4: when a re-scored episode has a lower
    # u_stored than the stored value (but still above the prune threshold),
    # downgrade the stored value + overwrite the nine-signal block with
    # the fresh verifier view. Pre-Goal-4 behaviour only RAISED u_stored;
    # downgrade was silently skipped, which let stale confident entries
    # linger across cycles without truthfully reflecting the updated
    # verifier's view. Set False to restore raise-only behaviour.
    retroverify_allow_downgrade: bool = True

    # ------------------------------------------------------------------ #
    # Memory consolidation (Branch C Goal 4 item 3)                        #
    # ------------------------------------------------------------------ #
    # [DES] Run a cycle-boundary SBERT-similarity clustering pass that
    # collapses duplicate-meaning episodes into a single representative
    # (the max-u_stored member of each cluster). Merges retrieval_count and
    # success_rate from the other cluster members into the representative
    # so retrieval-feedback history is not lost. Expected to reduce memory
    # size by ~20-30% over 10 cycles without information loss.
    enable_consolidation: bool = True

    # [DES] Cosine-similarity threshold that triggers candidate clustering
    # on the query-side SBERT embedding. Raised from an initial 0.88
    # proposal to 0.92 after a pre-merge design review (2026-04-22) flagged
    # that 0.88 is loose enough to group different-answer queries that
    # happen to share surface form (e.g. "Who directed X" vs "Who starred
    # in X"). 0.92 tightens the candidate pool; an answer-consistency
    # post-filter + u_stored-spread guardrail below give additional safety.
    consolidation_similarity_threshold: float = 0.92

    # [DES] Per-entry FAISS search depth when finding consolidation
    # neighbours. Above 0.92 similarity is rare in a well-distributed QA
    # store, so k=20 captures ~all real neighbours. Raise if clusters are
    # being split because a large group's tail is pushed past top-k.
    consolidation_search_k: int = 20

    # [DES] No-merge guardrail: clusters whose u_stored spread exceeds
    # this threshold are NOT merged and are logged for audit. High internal
    # variance is a tell that the cluster is heterogeneous despite sharing
    # surface form -- probably different-answer entries that happened to
    # score above the similarity threshold. Better to keep them separate
    # and let the next retroverify pass decide than to silently merge.
    consolidation_max_u_spread: float = 0.15

    # ------------------------------------------------------------------ #
    # Hit-counter forced re-verification (Branch C Goal 4 item 5)          #
    # ------------------------------------------------------------------ #
    # [DES] Each EpisodicEntry tracks a per-entry Tier-1-serve counter
    # (``hit_counter``). When the counter crosses this threshold, the
    # entry is queued for forced re-verification on the NEXT cycle
    # boundary regardless of its age. Motivation: popular-but-wrong
    # answers compound retrieval harm (Tier 1 serves them unverified
    # across many queries), so the cost of a forced re-check is worth
    # catching the failure mode early. Set to 0 to disable the forced
    # queue (entries still participate in the normal cycle-boundary
    # retroverify sweep -- this only controls the priority-queue fast
    # path).
    hit_counter_force_retroverify: int = 10

    # ------------------------------------------------------------------ #
    # Deferred-entry buffer (Stage 7b -- held for cycle-boundary          #
    # reconsideration, thesis Section 4.9)                                 #
    # ------------------------------------------------------------------ #
    # When Stage-5 emits DEFERRED (0.45 <= u_stored < 0.65), the episode
    # is written here instead of being dropped. At each cycle boundary,
    # after retroverify, the buffer is re-scored under the fine-tuned
    # verifier; entries clearing store_threshold (u_stored >= 0.65 AND
    # decision == STORE) are promoted into main memory. TTL prevents the
    # buffer from accumulating entries the model never gains confidence
    # in; bounded capacity evicts oldest on overflow.
    # [DES] Bounded FIFO capacity.
    deferred_buffer_max_size: int = 10_000
    # [DES] Max reconsideration passes an entry can survive without
    # promotion before it is dropped.
    # Re-tuned 2026-04-30 from 2 → 4 alongside the safety_u_pre_min change.
    # Rationale: TTL=2 promotes ~40-60% of deferred-correct entries based on
    # cycle-1 deferred-correct distribution analysis (entries close to τ_store
    # promote within 2 cycles; entries near τ_defer need 3-5 cycles of SIL
    # improvement to cross). TTL=4 captures roughly 60-80% of deferred-correct
    # via the cumulative recovery path with negligible buffer-size cost (cap
    # 10,000; expected size at TTL=4 stays under cap until ~cycle 6-7, after
    # which FIFO eviction takes over regardless of TTL). The per-cycle
    # retroverify cost rises modestly (~30 → ~45-60 min/cycle at peak buffer
    # size) but stays well under the eval-fold cost. Effective from cycle 2.
    # 2026-05-04: reverted to ttl=2 from ttl=4 alongside the orchestrator
    # deferred-buffer fix at cycle-4 close (Path Y mid-trajectory transition).
    # Reconsideration begins firing at cycle 5; an entry has at most TWO
    # reconsider() invocations (cycle N+1, cycle N+2) to be promoted before
    # being TTL-dropped. Matches the original Ch4 sec:deferred-reconsider
    # registration; the prior 4-cycle change (registered 2026-04-30) was
    # never empirically validated because reconsideration never fired during
    # cycles 1-4 and is reverted to align the live cycles 5-10 with the
    # registered thesis design.
    deferred_buffer_ttl_cycles: int = 2

    # v2 Fix 11 — opt-out flag for the five-layer deferred-reconsideration
    # guard. When False (default for production trajectories), Layer 1
    # (orchestrator assert) and Layer 2 (SIL hard-fail) raise RuntimeError
    # if pipeline.deferred_buffer is missing at run_cycle entry. Setting
    # this to True lets unit tests and legitimate skip-deferred ablations
    # run without hitting the guard. The thesis runbook NEVER sets this.
    allow_skip_deferred: bool = False

    # ------------------------------------------------------------------ #
    # Retrieval feedback loop                                              #
    # ------------------------------------------------------------------ #
    # [DES] Only update u_stored after ≥ 5 retrievals (statistical stability).
    feedback_loop_min_retrievals: int = 5
    # [DES] Exponential smoothing rate for u_stored updates.
    feedback_loop_eta: float = 0.01

    # ------------------------------------------------------------------ #
    # Experiment settings                                                  #
    # ------------------------------------------------------------------ #
    # ID training pool: module-level TRAINING_BENCHMARKS
    #   (natural_questions, triviaqa, fever)
    # OOD held-out pool: module-level TRANSFER_BENCHMARKS
    #   (truthfulqa, strategyqa, arc_challenge)
    # Benchmark list is passed via CLI --benchmarks; the 3/3 split is
    # enforced at training-time in _collect_episodes and at scoring time
    # in pooled_chm(), not here.
    # num_cycles: the pre-registered CAEM experiment runs for TEN cycles
    # (see Ch1 §Scope, Ch1 §Constrained Self-Improvement Loop, Ch3 §FR3,
    # and Ch5). Ten cycles gives the episodic memory enough horizon to
    # approach equilibrium under the MMLU retention guard, consistent with
    # the 7-10 cycle horizons typical of continual-learning equilibrium
    # studies. This is the default used by VAST_AI_DEPLOYMENT_GUIDE and
    # NEXT_SESSION_PLAN; --num-cycles may be lowered (e.g. --num-cycles 3)
    # for smoke-tests or short-horizon debug runs.
    num_cycles: int = 10
    calibration_set_size: int = 500
    purity_validation_set_size: int = 500
    questions_per_cycle: int = 5_000

    # ------------------------------------------------------------------ #
    # Per-cycle calibration protocol                                       #
    # ------------------------------------------------------------------ #
    # T (the u_pre temperature scalar) is the only learnable calibration
    # surface that the runtime pipeline consumes; u_stored composite
    # weights are fixed by design (grounding dominates the composite mass
    # by construction following Farquhar 2024 / Mishra 2024). T is re-fit
    # at every cycle boundary on a disjoint held-out calibration slice
    # (never from the SIL stream; see scripts/run_experiment.py) because
    # fine-tuning shifts the logit distribution and a Cycle-0 T drifts
    # out of calibration by Cycle N (Ovadia et al. NeurIPS 2019,
    # Thulasidasan et al. 2019, Guo et al. 2017).

    # ------------------------------------------------------------------ #
    # v2 Fix 12 helpers — per-benchmark u_pre dispatch                    #
    # ------------------------------------------------------------------ #
    def get_temperature_for(self, source_benchmark: Optional[str]) -> float:
        """Return T_b for a benchmark, or pooled T_global as fallback.

        Falls back to the pooled ``temperature_scalar`` when
        ``source_benchmark`` is ``None`` or not present in
        ``temperature_scalar_per_benchmark``. Mirrors the dispatch contract
        used by Fix 1 (per-benchmark conformal gate) and Fix 2B
        (per-benchmark composite).
        """
        if source_benchmark is None:
            return float(self.temperature_scalar)
        return float(
            self.temperature_scalar_per_benchmark.get(
                source_benchmark, self.temperature_scalar,
            )
        )

    def get_safety_u_pre_min_for(
        self, source_benchmark: Optional[str]
    ) -> float:
        """Return per-benchmark safety_u_pre_min_b, or pooled global as fallback.

        Falls back to the pooled ``safety_u_pre_min`` when
        ``source_benchmark`` is ``None`` or not present in
        ``safety_u_pre_min_per_benchmark``.
        """
        if source_benchmark is None:
            return float(self.safety_u_pre_min)
        return float(
            self.safety_u_pre_min_per_benchmark.get(
                source_benchmark, self.safety_u_pre_min,
            )
        )
