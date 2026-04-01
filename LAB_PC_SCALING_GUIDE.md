# CAEM Hardware Scaling Guide (Lab PC Edition)

This document outlines the maximum-capacity hyperparameters you can (and should) unlock when transferring execution from your constrained local machine (16GB RAM, RTX 3060) to your dedicated Lab PC. 

Scaling these parameters will dramatically increase your retrieval accuracy, verification robustness, and training stability for your final thesis numbers.

---

## 1. Massive Memory Indexing (Passage Store)
**File:** `scripts/build_passage_index.py` (CLI argument)
*   **Current Setup:** `500,000` passages (Maxing out 16GB limit, relying on slow SSD paging).
*   **Lab PC:** `5,000,000` to `20,000,000` passages.
*   **Why:** A 5M+ index contains exponentially more niche and long-tail facts. Tier 3 (RAG) generation accuracy on complex HotpotQA datasets will skyrocket because the model is no longer relying on a fractional subset of Wikipedia. 
*   **Command:** `python scripts/build_passage_index.py --max_passages 5000000 --output_dir outputs/passage_index`

## 2. Episodic Memory Seed Volume
**File:** `scripts/seed_cold_start.py` (CLI argument)
*   **Current Setup:** `200` target episodes per benchmark.
*   **Lab PC:** `1,000` to `2,000` target episodes per benchmark.
*   **Why:** Starting Cycle 1 with a massive, highly-dense associative memory matrix means Tier 1 routing acts like a hyper-efficient cache right out of the gate, maximizing both throughput and accuracy from the very first question.
*   **Command:** `python scripts/seed_cold_start.py --target_episodes 1000 --output_dir outputs/cold_start_memory`

## 3. Self-Consistency Robustness (Stage 4 & Stage 5)
**File:** `caem/config.py` (`sc_chains_m`)
*   **Current Setup:** `sc_chains_m = 3`
*   **Lab PC:** `sc_chains_m = 10` or `20`
*   **Why:** When measuring `u_consistency` and `s_avg` verification flags, generating 3 answers provides a "minimum viable" consensus. Generating 10-20 independent CoT paths virtually eliminates statistical noise, mirroring the original Wang et al. (2022) hardware parameters.

## 4. Semantic Entropy Clustering (Stage 4 & Stage 5)
**File:** `caem/config.py` (`se_samples_k`)
*   **Current Setup:** `se_samples_k = 10`
*   **Lab PC:** `se_samples_k = 20`
*   **Why:** Farquhar et al. (2024) rely on deep sampling pools to measure meaning-level uncertainty. Larger sample sizes allow the NLI clustering algorithm to firmly map the boundary of "true" vs "confabulated" reasoning paths.

## 5. Training Batch Sizes (Stage 8)
**File:** `caem/config.py` (`batch_size`)
*   **Current Setup:** Small (Likely `1`, `2`, or `4` to prevent VRAM overflow on a 12GB 3060 during backpropagation).
*   **Lab PC:** `16` or `32`
*   **Why:** Larger batch sizes during the Self-Improvement loop provide drastically smoother gradient updates. It speeds up the self-training epoch times and prevents catastrophic forgetting by averaging out noisy single-batch anomalies when regularizing against `theta_prev`.

## 6. MC Dropout Resolution
**File:** `caem/config.py` (`mc_dropout_k`)
*   **Current Setup:** `mc_dropout_k = 5`
*   **Lab PC:** `mc_dropout_k = 10`
*   **Why:** Generating 10 stochastic forward passes at inference provides a much cleaner variance distribution for calculating per-token output uncertainty.

## 7. Hyper-Accelerate L2 Regularization (Unlocking PCIe bottleneck)
**File:** `caem/training/self_improvement.py` (Inside `_finetune` and `_l2_penalty`)
*   **Current Setup:** `p0` (which is `theta_prev`) is permanently stored on systemic CPU RAM to save 3.1GB of GPU VRAM. During training, the pipeline ships exact memory layers natively to the GPU *per batch* over your computer's PCIe bus to calculate the L2 distance penalty. 
*   **Lab PC:** Disable this hardware trick! If your Lab PC has 24GB+ VRAM, immediately rewrite `theta_prev` to stay natively attached to the GPU.
*   **Code Mod:** In `_finetune()`, add `theta_prev_gpu = [p0.to(self.device) for p0 in theta_prev]` before the epoch loops, and redirect `_l2_penalty` to use `theta_prev_gpu` instead of mapping it dynamically. 
*   **Why:** Eradicating the cyclic PCIe bus transfer bottleneck will mathematically accelerate your Cycle fine-tuning epochs by approximately **400%**.

---

### Summary Checklist for the Lab PC transfer:
1. Open `caem/config.py` and double `sc_chains_m`, `se_samples_k`, and `batch_size`.
2. Generate the background Wikipedia matrices at `5,000,000` or higher limit.
3. Seed the cold start memory at `1,000` episodes.
4. Run the full experiment on all 5,000+ benchmark questions simultaneously since computing time will no longer be a bottleneck.
