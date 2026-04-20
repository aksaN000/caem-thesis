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
from typing import Tuple


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
    "natural_questions",
    "triviaqa",
    "fever",
)

TRANSFER_BENCHMARKS: Tuple[str, ...] = (
    "truthfulqa",
    "strategyqa",
    "arc_challenge",
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
    rag_faiss_pq_m: int = 64
    rag_faiss_pq_nbits: int = 8
    rag_faiss_train_sample_size: int = 500_000

    # [DES] Number of passages retrieved from the Wikipedia corpus.
    # k=5 is standard in DPR (Karpukhin et al. 2020); gives good recall
    # without overloading the Flan-T5 context window.
    rag_top_k: int = 5
    # ------------------------------------------------------------------ #
    # Base generator (Goal 1, Branch C — 2026-04-21 decision)              #
    # ------------------------------------------------------------------ #
    # [DES] HuggingFace model name. Branch-C default is Qwen/Qwen2.5-3B-Instruct
    # (decoder-only, 3B params, Apache 2.0, ChatML format, ~99% label compliance
    # and <2% loop rate per modern instruction tuning). Legacy Flan-T5-Large path
    # retained for Variant 18 `flan_t5_large_backbone` ablation; architecture is
    # auto-detected via ``HfConfig.is_encoder_decoder`` so both paths coexist.
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # [DES] Prompt template style. One of:
    #   "chatml_scaffold"  — ChatML envelope + scaffolded Reasoning/Answer (Qwen,
    #                         Gemma, Llama-3.x); forced prefix via prefill.
    #   "flan_t5_scaffold" — Flat-text few-shot + forced ``decoder_input_ids``
    #                         "Reasoning:" prefix (Flan-T5 encoder-decoder).
    # Must match ``base_model_name``'s architecture family; a mismatch raises at
    # pipeline construction time.
    prompt_style: str = "chatml_scaffold"

    # ------------------------------------------------------------------ #
    # Goal 5 — hardware-utilization optimizations (Branch C)               #
    # ------------------------------------------------------------------ #
    # [DES] Enable Flash Attention 2 if the `flash_attn` library is installed on
    # the rental. Gracefully falls back to eager attention (with log warning) on
    # unavailable. Typically yields 10-15% generator throughput on 5090.
    use_flash_attention_2: bool = True

    # [DES] torch.compile on the generator forward pass. Nondeterminism risk
    # (~1e-4 logit drift) is below u_stored composite's grounding-signal noise
    # floor per Branch-C analysis. Start False during port+smoke; enable after
    # regression gate validation on perf_log.csv.
    use_torch_compile: bool = False

    # [DES] LoRA SIL training (Qwen-3B won't fit full FT in 32 GB VRAM even with
    # bf16 mixed precision and 8-bit AdamW; full FT needs ~48 GB). Promotes the
    # Ch4 §Cycle-2 retention O-LoRA structured-fallback from fallback to primary
    # training strategy.
    use_lora_training: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # [DES] Target modules by architecture. Decoder-only models use the full
    # attention+MLP family (Qwen/Llama convention). Encoder-decoder narrows to
    # q/v projections (standard PEFT default for Flan-T5). Selected at train time
    # based on ``model.config.is_encoder_decoder``.
    lora_target_modules_decoder_only: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    lora_target_modules_encoder_decoder: Tuple[str, ...] = ("q", "v")

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
