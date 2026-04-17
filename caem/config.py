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

    # [CAL] Initial weights are EQUAL (0.25 each).
    # Projected post-calibration: ~0.20/0.20/0.20/0.40 (SE upweighted due to
    # AUROC ≈ 0.79, Farquhar et al. 2024). Actual values come from calibration
    # set after Cycle 1 and are reported in Chapter 5.
    u_hat_weight_token: float = 0.25       # initial; calibrated -> ~0.20
    u_hat_weight_dropout: float = 0.25     # initial; calibrated -> ~0.20
    u_hat_weight_sc: float = 0.25          # initial; calibrated -> ~0.20
    u_hat_weight_entropy: float = 0.25     # initial; calibrated -> ~0.40

    # [DES] Asymmetric cost: accept Tier 2 answer if û ≥ this; else escalate.
    u_hat_accept_threshold: float = 0.60

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

    # [LIT] Best open NLI model at this parameter scale.
    nli_model: str = "roberta-large-mnli"
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
    # any p_contra >= contradiction_veto_threshold    -> DISCARD
    store_threshold: float = 0.65
    defer_threshold: float = 0.45
    abstain_pground_ceiling: float = 0.20
    contradiction_veto_threshold: float = 0.30

    # --- Grounding retrieval / rerank [DES] ----------------------------- #
    # Retrieve top-k passages from Wikipedia corpus, then rerank to top-N
    # before scoring p_ground_max / p_ground_mean.
    verifier_retrieve_k: int = 20
    verifier_rerank_k: int = 3

    # --- u_stored composite weights [DES] (must sum to 1.0) ------------- #
    # Session 42 rebalanced from the legacy three-signal mix
    # (nli=0.50, sc=0.30, se=0.20). External grounding (pground_mean +
    # pground_atomic = 0.45) now dominates the prior, reflecting the
    # Chapter 1 promise that NLI verifies correctness against retrieved
    # evidence rather than self-consistency alone.
    u_stored_weight_pground_mean: float = 0.30
    u_stored_weight_pground_atomic: float = 0.15
    u_stored_weight_nli: float = 0.15          # p_entail (chain -> answer)
    u_stored_weight_sc: float = 0.15           # s_avg (pairwise SBERT cosine)
    u_stored_weight_uinternal: float = 0.15    # 0.5·u_token + 0.5·(1 - u_dropout)
    u_stored_weight_se: float = 0.10           # applied as (1 - h_norm)

    # ------------------------------------------------------------------ #
    # Tier 3 RAG (Stage 6)                                                 #
    # ------------------------------------------------------------------ #
    # [DES] Passage FAISS backend.
    rag_index_type: str = "ivf_pq"
    rag_faiss_nlist: int = 65_536
    rag_faiss_nprobe: int = 64
    rag_faiss_pq_m: int = 64
    rag_faiss_pq_nbits: int = 8
    rag_faiss_train_sample_size: int = 500_000

    # [DES] Number of passages retrieved from the Wikipedia corpus.
    # k=5 is standard in DPR (Karpukhin et al. 2020); gives good recall
    # without overloading the Flan-T5 context window.
    rag_top_k: int = 5
    # [DES] Max tokens allocated for retrieved context in the model prompt.
    # Flan-T5-Large has a 512-token encoder limit; 384 leaves room for the
    # question and instruction prefix.
    rag_max_context_tokens: int = 384
    # [DES] Max new tokens for RAG generation (longer than Tier 2 because
    # the model now has supporting context to draw from).
    rag_max_new_tokens: int = 256
    # [DES] Max new tokens for CoT (Chain-of-Thought) generation limits
    cot_max_new_tokens: int = 256
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
    epochs_per_cycle: int = 3
    warmup_steps: int = 500
    # [DES] Only include verified episodes above this quality in training data.
    min_u_stored_for_training: float = 0.75
    # [DES] Mix 10% general-domain data to prevent catastrophic forgetting.
    general_data_ratio: float = 0.10
    # [DES] Abort fine-tuning if general capability drops below this retention.
    forgetting_tolerance: float = 0.93

    # ------------------------------------------------------------------ #
    # Retroactive re-verification                                          #
    # ------------------------------------------------------------------ #
    # [DES] Remove from memory if updated u_stored drops below this.
    retroverify_prune_threshold: float = 0.50

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
    # Training benchmarks: fever, triviaqa, natural_questions
    # Transfer eval benchmarks: truthfulqa, strategyqa, arc_challenge
    # (benchmark list is passed via CLI --benchmarks; no single-benchmark field)
    num_cycles: int = 10
    calibration_set_size: int = 500
    purity_validation_set_size: int = 500
    questions_per_cycle: int = 5_000
