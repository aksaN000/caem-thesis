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
from typing import Optional, Tuple


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
    "natural_questions",
)
# Branch C 2026-04-22 evening decision: ASQA demoted to TRANSFER-ONLY.
# Training panel capped at 3 benchmarks with large train splits (FEVER ~145k,
# TriviaQA ~87k, NQ ~87k) — enough to sustain 10-cycle stream mode at
# 5000 samples/cycle without reuse. ASQA's 4353 train is structurally
# insufficient for stream mode. ASQA remains in the eval panel
# (500 eval/cycle from dev) as Path B (Qwen-judge) long-form evidence.
# See caem/benchmark_splits.py for authoritative panel definition.

TRANSFER_BENCHMARKS: Tuple[str, ...] = (
    "truthfulqa",
    "strategyqa",
    "arc_challenge",
    "asqa",  # Branch C 2026-04-22 evening: transfer-only for Path B (Qwen-judge)
             # long-form evidence. Eval trajectory only; never trained on.
             # Must stay in sync with caem/benchmark_splits.py TRANSFER_BENCHMARKS.
)


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
    safety_u_pre_min: float = 0.60
    # [CAL] Temperature scaling for u_pre calibration (Guo et al. 2017).
    # Applied as sigmoid(logit(u_pre) / T). T=1.0 means no calibration.
    temperature_scalar: float = 1.0

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
    u_stored_weight_pground_mean: float = 0.28
    u_stored_weight_pground_atomic: float = 0.14
    u_stored_weight_nli: float = 0.16          # p_entail (chain -> answer)
    u_stored_weight_q_a_relevance: float = 0.14   # [DES] NEW (Branch C Goal 2)
    u_stored_weight_sc: float = 0.14           # s_avg (pairwise SBERT cosine)
    u_stored_weight_uinternal: float = 0.10    # 0.5·u_token + 0.5·(1 - u_dropout)
    u_stored_weight_se: float = 0.04           # applied as (1 - h_norm)

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
    # Blackwell this dispatches to cuDNN-attention for 4-6x Qwen-3B
    # throughput (~40 → ~200 tok/s) with zero composite-architecture
    # impact. Requires PyTorch 2.9+ and cuDNN 9.15+ (both present on
    # our Vast image). Set False to revert to eager attention — only
    # useful for an ablation comparing attention-kernel throughput.
    # Branch-C landing: 2026-04-23, research_attn_alternatives.md.
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
    # To re-enable for a single-path workload (e.g., B6/B7 pure inference
    # baseline, which only runs main generate + no MC-dropout/atomic),
    # explicitly pass use_torch_compile=True to load_base_generator in
    # that script — Python default remains False for the multi-mode
    # Step 6/7/main path.
    use_torch_compile: bool = False

    # [DES] Full FT + 8-bit AdamW is the PRIMARY SIL training path on Qwen-3B
    # (verified 2026-04-22 to fit 32 GB 5090 at batch=4, grad checkpointing
    # on, bf16 mixed precision: ~22-25 GB peak with MiniCheck loaded). LoRA
    # is the first STRUCTURED FALLBACK (per Ch4 §Cycle-2 retention) if Full
    # FT fails (OOM, MMLU retention < 0.93, convergence failure). Memory-
    # only (no weight update) is the last resort.
    use_8bit_adamw: bool = True

    # [DES] LoRA fallback config. Only consulted when use_lora_training=True
    # (either set directly, or auto-set by the SIL harness after a full-FT
    # failure mode is detected per the Ch4 §Cycle-2 retention cascade).
    use_lora_training: bool = False  # PRIMARY path is full FT; set True only as fallback
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

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
    batch_size: int = 16
    # [DES] Gradient-accumulation multiplier. Effective batch size is
    # ``batch_size * grad_accum_steps``; tuned per hardware tier by
    # ``scripts.hardware.get_hardware_profile`` to hold effective batch at
    # ``TARGET_EFFECTIVE_BATCH_SIZE`` (=32). Default 1 (no-op) so a bare
    # CAEMConfig() stays identical to the thesis 5090 path.
    grad_accum_steps: int = 1
    epochs_per_cycle: int = 3
    warmup_steps: int = 500
    # [DES] Only include verified episodes above this quality in training data.
    min_u_stored_for_training: float = 0.75
    # [DES] Mix 10% general-domain data to prevent catastrophic forgetting.
    general_data_ratio: float = 0.10
    # [DES] Number of TriviaQA-train general-domain QA pairs to load for the
    # 10% anti-forgetting mix. Sized so there is never a shortage when the
    # cycle's training batch grows. Scripts offset past the SIL pool so the
    # mix is disjoint from the TriviaQA SIL training episodes.
    general_data_size: int = 1000
    # [DES] Abort fine-tuning if general capability drops below this retention.
    forgetting_tolerance: float = 0.93

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
    deferred_buffer_ttl_cycles: int = 2

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
    # in pooled_ce(), not here.
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
