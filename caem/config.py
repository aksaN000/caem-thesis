"""
caem/config.py
==============
Central configuration for the CAEM pipeline.

Every numeric constant in the system must come from here — no magic numbers
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
    # [DES] 20k entries ≈ 20 MB metadata + ~30 MB FAISS index.
    max_memory_size: int = 20_000
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
    # Pre-routing confidence (Stage 3, fast — 2 signals)                  #
    # ------------------------------------------------------------------ #
    # [DES] Token prob is the primary reliability signal.
    u_pre_token_weight: float = 0.60
    # [DES] C_conv (internal convergence) is secondary.
    u_pre_cconv_weight: float = 0.40

    # ------------------------------------------------------------------ #
    # Adaptive Router (Stage 3 → dispatch)                                 #
    # ------------------------------------------------------------------ #
    # [DES] Combined score = routing_lambda·s + (1-λ)·û_stored
    routing_lambda: float = 0.70
    # [DES] Route to Tier 1 if combined score ≥ this.
    tier1_combined_threshold: float = 0.90
    # [DES] Route to Tier 2 if similarity > this (and not Tier 1).
    tier2_similarity_threshold: float = 0.75
    # [DES] OR-condition: force Tier 3 if u_pre < this, regardless of memory.
    safety_u_pre_min: float = 0.60

    # ------------------------------------------------------------------ #
    # Post-generation confidence (Stage 4a, Tier 2 only — 4 signals)      #
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
    u_hat_weight_token: float = 0.25       # initial; calibrated → ~0.20
    u_hat_weight_dropout: float = 0.25     # initial; calibrated → ~0.20
    u_hat_weight_sc: float = 0.25          # initial; calibrated → ~0.20
    u_hat_weight_entropy: float = 0.25     # initial; calibrated → ~0.40

    # [DES] Asymmetric cost: accept Tier 2 answer if û ≥ this; else escalate.
    u_hat_accept_threshold: float = 0.60

    # ------------------------------------------------------------------ #
    # Multi-Layer Verifier (Stage 5)                                       #
    # ------------------------------------------------------------------ #
    # [LIT] Best open NLI model at this parameter scale.
    nli_model: str = "roberta-large-mnli"
    # [DES] NLI entailment threshold for hard-pass.
    nli_entailment_threshold: float = 0.90
    # [DES] Self-consistency gate: s_avg > this → accept.
    sc_accept_threshold: float = 0.85
    # [LIT] Farquhar et al. 2024: agglomerative + cosine clustering.
    se_clustering_method: str = "agglomerative"
    # [DES] Semantic entropy gate: H < this (bits) → accept.
    se_entropy_threshold: float = 1.5
    # [DES] VE1 escalation: NEUTRAL + SC > 0.90 → escalate (systematic confab).
    se_escalation_sc_min: float = 0.90

    # û_stored weights [DES]: NLI entailment is strongest post-hoc signal.
    u_stored_weight_nli: float = 0.50
    u_stored_weight_sc: float = 0.30
    u_stored_weight_se: float = 0.20   # applied as (1 - h_norm)

    # ------------------------------------------------------------------ #
    # Tier 3 RAG (Stage 6)                                                 #
    # ------------------------------------------------------------------ #
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
    # Retrieval feedback loop                                              #
    # ------------------------------------------------------------------ #
    # [DES] Only update u_stored after ≥ 5 retrievals (statistical stability).
    feedback_loop_min_retrievals: int = 5
    # [DES] Exponential smoothing rate for u_stored updates.
    feedback_loop_eta: float = 0.01

    # ------------------------------------------------------------------ #
    # Experiment settings                                                  #
    # ------------------------------------------------------------------ #
    benchmark: str = "hotpotqa"
    num_cycles: int = 3
    calibration_set_size: int = 500
    purity_validation_set_size: int = 500
    questions_per_cycle: int = 5_000
