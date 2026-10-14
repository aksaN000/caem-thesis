# SIL Poison-Drift: Diagnosis, Fix Plan, and Reproduction Config

**Date:** 2026-06-16 (Dhaka time)
**Status:** Diagnosis complete (static, no-GPU). Causal ablation pending GPU machine.
**Scope:** Why pooled EM regresses after Cycle 2, whether it is "memory poisoning," and exactly what to run to confirm + fix it.

> One-line takeaway: the post-C2 EM decline on the heavily-stored training benches is **SIL fine-tuning drift driven mainly by the ~25% of stored episodes that are wrong but passed the gate (verifier false positives), not by Tier-1 retrieval poisoning**. The fix does **not** require a perfect verifier — the poison is calibrated and low-confidence-concentrated, so confidence-weighted / noise-robust SIL can strip most of its gradient.

---

## 1. The finding — EM peaks at C2, then drifts down

Pooled EM (equal-weight over the 5 benchmarks), computed directly from per-sample eval JSONs:

| cycle | RUC-OFF (Phase 1d, `phase1d_eval/`) | RUC-ON (Phase 1e, `full_run/eval/`) |
|---|---|---|
| C0 | 48.0 | 54.8 |
| C1 | 51.5 | 55.7 |
| C2 | 52.0 (peak) | **56.7 (peak)** |
| C3 | 50.9 | 54.7 |
| C4 | 50.9 | 54.8 |
| C5 | 50.9 | 55.6 |

- RUC-on is **+4 to +7 pp** higher EM at every cycle (the RUC headline win).
- Both phases peak at **C2** then decline; the post-C2 dip is essentially the same magnitude with or without RUC (pooled ≈ −1.1 pp). **RUC does not remove the post-C2 drift.**
- C3 = C4 in Phase 1d because the **C4 retention guard fired** (TriviaQA-test ratio 0.835 < 0.93 floor) and rolled the adapter back to C3.

### Decomposition: it is capability loss, not routing/selection
Oaxaca split of the C2→C5 change into "router moved queries to a worse tier" (selection) vs "the tiers themselves got worse" (capability):

| | ΔEM | selection | capability |
|---|---|---|---|
| Total / 5 benches | −2.64 | **−0.37** | **−2.38** |

~90% of the drop is **within-tier capability loss** — the model getting worse, concentrated on the SIL-trained parametric tier (T2). FEVER T2 60.4→50.0, TriviaQA T2 51.0→42.1.

### Tier-1 is essentially unused at eval → retrieval poisoning is ruled out
Tier-1 (memory-direct serving) share at eval is ≤ 0.003 (1 query in 300) across all cycles, both phases. No wrong stored answer is being retrieved-and-served. So "poisoning" is **not** the retrieval kind; it is the **training kind** (the stored pool is the SIL training set).

---

## 2. The mechanism — two kinds of SIL drift

Each cycle, SIL LoRA-fine-tunes the model on the stored pool (~74% correct, ~26% wrong). Two distinct drift sources:

- **Type A — "clean" drift (training on the ~74% verified-correct episodes).** The model already produces these with high confidence, so on the trained benches they are largely **gradient-inert**. Their harm (forgetting, distribution-narrowing) lands mostly on capabilities *under-represented* in the pool, and is smaller/more diffuse.
- **Type B — "poison" drift (training on the ~26% wrong episodes the verifier missed; this is `ε_train`).** These carry **outsized gradient** because the (wrong) target diverges from the model's natural output. They directly teach the model to emit specific wrong answers — and to be confident about them.

### Poison fingerprint (confident-when-wrong rises as EM falls)
If the decline were *forgetting* (Type A), wrong answers would get *less* confident. If it is *poison* (Type B), wrong answers get *more* confident. Fraction of **wrong** answers that are confident (`u_stored ≥ 0.5`), RUC-on:

| bench | C0 | C2 (peak) | C5 |
|---|---|---|---|
| FEVER | 0.169 | 0.571 | **0.783** |
| TriviaQA | 0.000 | 0.388 | **0.470** |

Across the post-peak window (C2→C5) where EM is falling, confidence-when-wrong keeps **rising** — the poison signature, not the forgetting signature.

---

## 3. Static diagnosis (run 2026-06-16, no GPU)

### (a) ε_train — poison rate in the SIL-eligible pool (`u_stored ≥ 0.70`), eval-proxy
| cycle | n eligible | purity | ε_train | poison n |
|---|---|---|---|---|
| C0 | 65 | 0.815 | 0.185 | 12 |
| C1 | 296 | 0.794 | 0.206 | 61 |
| C2 | 332 | 0.741 | 0.259 | 86 |
| C3 | 333 | 0.742 | 0.258 | 86 |
| C5 | 223 | 0.753 | 0.247 | 55 |

SIL pool runs ~21–26% poison. (Training-stream storage purity from `pi_store_trajectory.csv` is flatter, ~0.72–0.75, so "the pool got dirtier exactly at C2" is suggestive, not proven.)

### (b) WHERE the poison sits — the actionable result
| `u_stored` band | n | purity | % of all poison |
|---|---|---|---|
| **[0.70, 0.80)** | 1266 | 0.741 | **88.2%** |
| [0.80, 0.90) | 228 | 0.816 | 11.3% |
| [0.90, 1.0] | 17 | 0.882 | 0.5% |

**88% of poison is in the lowest confidence band, and purity rises monotonically with confidence.** The verifier is calibrated in the right direction; the poison is *not* hiding at high confidence. → Confidence-weighting / higher τ / noise-robust loss will strip most of the poison gradient **without a perfect verifier**.

### (c) Per-bench poison rate vs EM drop (C2→C5)
| bench | SIL poison rate | ΔEM |
|---|---|---|
| **FEVER** | **0.328** (highest) | **−5.7** (worst) |
| TriviaQA | 0.213 | −4.0 |
| CommonsenseQA | 0.236 | 0.0 |
| TruthfulQA | 0.250 | **+5.3** (RUC rescue) |
| StrategyQA | 0.151 (lowest) | −1.3 |

Dirtiest pool (FEVER) drifts most; cleanest (StrategyQA) least. TruthfulQA is the exception: high poison but it *improved*, because RUC routed its RAG-hurt queries away — overriding the poison effect.

### Memory purity summary
Storage-pool purity `π_store` is **stable at ~73–75%** across cycles (not collapsing). It is constant ~26% impurity, but the pool grows 431 → 12,175 episodes, so the absolute poison count climbs ~112 → ~3,165, and the impurity is benchmark-skewed (TriviaQA lowest, 0.664 at C5).

---

## 4. The fix — robustness over a perfect verifier

A perfect verifier is impossible *and* self-defeating (if you had one you would not need SIL). The right target is **calibration, not perfection**, plus bounded damage. Concrete levers, none requiring perfect identification of poison:

1. **Confidence-weighted SIL loss** — `Loss_i = w(u_stored,i) · CE_i`. Poison clusters at low confidence (§3b), so it contributes less gradient on average. ~10-line change in `self_improvement.py`.
2. **Noise-robust loss** (generalized/symmetric cross-entropy, label smoothing) — caps the gradient of high-loss examples, i.e. exactly the poison.
3. **Agreement / small-loss filtering (co-teaching)** — train only where the model's self-consistent output agrees with the stored target; label-free dynamic filter.
4. **FEVER systematic blind spot** — FEVER has the highest poison and is the known prompt-boilerplate-flip case; some of its poison may be *high*-confidence (weighting won't catch it). Needs a **signal fix** (strip the FEVER prefix before SBERT encoding) + per-step claim checking for the reasoning blind spot.

**What CAEM already does (so it never relied on a perfect verifier):** two operating points (store τ=0.60, SIL τ=0.70), retroactive re-verification each cycle, retention-guard rollback (fired at C4), and the RUC reducing poison inflow. Missing: the gradient-level defenses (1)–(3).

---

## 5. Ablation plan (to run on the GPU machine)

Goal: causally separate Type-A clean drift from Type-B poison, and measure the EM ceiling a clean pool would buy.

### Arms
- **Arm 0 — baseline:** the existing Phase 1e trajectory (real ~74% pool). Already done; reuse.
- **Arm 1 — oracle-clean pool:** filter the SIL pool to actually-correct (EM=1) episodes before fine-tuning. If post-C2 drift vanishes → poison-driven. If it persists → Type-A forgetting is bigger than thought. **This is the ceiling and the primary diagnostic.**
- **Arm 2 — confidence-weighted SIL:** weight CE by calibrated `u_stored` (no labels needed). The deployable fix; compare to Arm 0/1.
- **Arm 3 — FEVER prefix-strip:** re-encode FEVER without the boilerplate prefix; checks whether its poison is the high-confidence systematic kind.

### Cheaper "replay-from-fixed-pool" variant (recommended first)
Dump each cycle's stored pool from Phase 1e once, then **re-run only the SIL fine-tune + eval per cycle** under each arm — skipping the expensive fresh-stream processing, RAG retrieval, and retroverify. This isolates the *immediate* per-cycle poison effect (the diagnostic question) at a fraction of the cost. It does not capture the downstream "cleaner model → cleaner next pool" compounding, but that is a follow-up, not the first test.

### What to measure (per arm, per cycle)
Pooled + per-bench EM; the confident-when-wrong fingerprint (§2); per-tier EM; CHM. Success = Arm 1/2 flatten or reverse the post-C2 decline relative to Arm 0.

---

## 6. EXACT configuration to reproduce (from Phase 1e audit)

**Match Phase 1e exactly; change only the SIL pool filter (Arm 1) or the loss weighting (Arm 2).** All values verified against `caem/config.py` + `caem/training/self_improvement.py` + the run log.

### Hardware / environment
| item | value |
|---|---|
| GPU | 1× NVIDIA RTX 5090, 31.4 GB VRAM (A100 40 GB equivalent; **≥32 GB VRAM** target) |
| Host RAM | **Full-RAG run: provision ~96 GB** (index 64.5 GB + anonymous ~25 GB ≈ ~90 GB hot working set, +headroom; see §7.1). **No-index replay variant: ~30 GB suffices** (index never loaded). 86 GB is OOM-safe either way but marginal for keeping the flat index fully hot. Add 16–32 GB swap as insurance. |
| Disk | ~80 GB (index ~72 GB + model weights ~6–7 GB; 150 GB NVMe provisioned in prod) |
| Precision | bf16, TF32 matmuls; attn = SDPA cuDNN-attention (generator); MiniCheck T5 = eager |
| Stack | Python 3.12, PyTorch 2.10+cu130 (sm_120 Blackwell), transformers 4.57.6, faiss-cpu 1.8+, sentence-transformers 4.1.0, peft 0.19+, bitsandbytes 0.43+, scikit-learn, scipy, anthropic SDK |
| 16 GB-VRAM fallback | possible with: SIL micro-batch 4→2, grad-accum 4→16, bf16→fp16, gen cap 512→192, activation checkpointing → ~2.2–2.6× wall-clock |

### Model + SIL training
| param | value | source |
|---|---|---|
| Backbone | Qwen/Qwen2.5-3B-Instruct (frozen base, bf16, torch.compile) | config.py:547 |
| LoRA | r=32, α=64, dropout=0.05, all-linear (q,k,v,o,gate,up,down) | config.py:683–690 |
| SIL gate (τ_train) | `min_u_stored_for_training = 0.70` (> store τ=0.60) | config.py:755 |
| Batch | micro=4 × grad_accum=4 → **effective 16** (eval inference batch = 32) | config.py:736,741 |
| Epochs / cycle | 3 | config.py:742 |
| LR | LoRA 2e-4, warmup 500; (full-FT fallback 1e-5) | config.py:694,743 |
| Optimizer | plain AdamW on LoRA path (8-bit AdamW disabled for LoRA), grad clip max_norm=1.0, grad checkpointing ON, force_eager during fine-tune | self_improvement.py:1045–1072,1253 |
| Anti-forgetting anchor | **L2 anchor DISABLED on the active LoRA path** (frozen base is the implicit anchor); l2_lambda=0.01 only on full-FT fallback | self_improvement.py:1225–1229 |
| Max seq len | input 512 / target 512; RAG context cap 384; rag_top_k=5 | self_improvement.py:215–216; config.py:704 |

### Gates / loop control
| param | value |
|---|---|
| store / defer thresholds | τ_store=0.60, τ_defer=0.45, abstain_pground_ceiling=0.20 |
| retention floor | 0.93; probes mmlu, triviaqa_test, commonsense_qa_test (n=200 each); rollback to θ_prev on breach |
| RUC | τ_RUC=0.375, 36 features (`caem/ruc/v2/feature_spec.json`) |
| pool reweighting (v2 Fix 3) | temp=2.0, upsample_cap=3.0, doremi_floor=50, max_share=0.40, cold_start_n=100 |
| cold-start store τ | 0.50 (CLI override at seed step only) |
| seed | 42 (single seed; deterministic pools) |

### Data / sample counts
| set | size |
|---|---|
| eval | 300 / bench × 5 benches (fever, triviaqa, commonsense_qa, truthfulqa, strategyqa) |
| calibration fold | 500 / bench |
| purity fold | 500 / bench |
| training stream / cycle | FEVER 2000, TriviaQA 2000, CommonsenseQA 700 (`questions_per_cycle=5000`); only the 3 ID benches train; TruthfulQA + StrategyQA are eval-only transfer |
| cycles | configured 10, **completed 6** (C0 cold-start + C1–C5, locked-stop at C5) |

### Retrieval / verifier stack (footprint)
| component | model | size / note |
|---|---|---|
| RAG index | **FAISS FLAT fp32**, 21,000,000 passages (Wikipedia 20231101.en, ~100-word chunks) | passages.faiss 64.5 GB + passages.pkl 7.5 GB ≈ **72 GB**; mmap (CPU, not VRAM). Lazily paged, but flat index brute-force-scans all 21M vectors per query → whole 64.5 GB becomes resident (reclaimable page cache) in normal RAG use. Local `data/passage_index/`; mirror HF `aksaN000/caem-passage-index-21m` |
| Embedder | sentence-transformers/all-mpnet-base-v2 | 768-dim, ~0.42 GB; shared by memory + retriever + verifier s_avg |
| Reranker | BAAI/bge-reranker-v2-m3 | ~2.2 GB; reranks top-20→top-3 AND drives q_a_relevance |
| Verifier (primary) | lytang/MiniCheck-Flan-T5-Large | ~1.5 GB bf16; NLI judge for hypotheses ≤408 MC-tokens |
| Verifier (long) | FrozenQwenJudge | shares base Qwen weights via `disable_adapter()` (0 extra VRAM); ablated out of deployed composite (MiniCheck-only ships) |
| TruthfulQA judge | Anthropic claude-haiku-4-5 (API) | for `em_llm_judged` only; not GPU-resident |

---

## 7. Phase 1e resource usage (actual)

GPU: 1× RTX 5090 (31.4 GB). No CUDA OOM. Peak VRAM not logged. Min VRAM to run full pipeline ≈ 32 GB.

Per-cycle wall-clock (full cycle = SIL train + 5 evals + calibration + gdrive offload + retroverify):

| cycle | wall-time | note |
|---|---|---|
| C0 | 361.0 min (6.0 h) | baseline eval, no SIL |
| C1 | 1742.9 min (29.1 h) | |
| C2 | 2125.1 min (35.4 h) | |
| C3 | 2590.0 min (43.2 h) | |
| C4 | 1457.1 min (24.3 h) | retention guard fired; retroverify skipped (~6 h saved) |
| C5 | 3278.9 min (54.6 h) | longest |
| **total** | **~11,555 min ≈ 192.6 h ≈ 8.0 days** single-5090 wall-clock | eval-only sum ≈ 35.3 h |

The bulk of per-cycle time is **pipeline inference** (RAG over 21M index + verifier over the fresh stream of 5000 + retroverify over the growing pool), **not** the LoRA step. → The cheaper replay-from-fixed-pool ablation (§5) avoids most of this.

**Cost implication for the ablation:** a full new trajectory arm ≈ ~8 days single-5090. The replay-from-fixed-pool variant is dramatically cheaper (≈ SIL train + eval per cycle only). Recommend the replay variant first; budget/predict wall-time before launch.

### 7.1 Host RAM — what the 140 GB on the Vast dashboard means

During the Phase 2a baseline rerun (flare + b6 on a Vast RTX 4090), the Vast webpage RAM meter showed **~140 GB**. That is **not** a hard requirement and does **not** indicate OOM risk. Reconciliation:

- The webpage meter reads the container **cgroup memory usage, which counts reclaimable page cache**, not just process memory. The flat fp32 index gets fully paged in during brute-force scans (64.5 GB), plus `passages.pkl` file cache (7.5 GB) + its Python object (~9 GB) + model/dataset file caches + Python/torch anonymous (~15–20 GB) ≈ ~140 GB cgroup total — but ~110 GB of that is evictable cache.
- **OOM-relevant (anonymous, non-reclaimable) demand ≈ 25–30 GB** (pkl object + Python/torch/CUDA buffers; HF datasets are mmap'd Arrow, also reclaimable). OOM-kill only fires on anonymous memory, so there is a wide safety margin at 86–96 GB.
- **The run logs do NOT contain host-RAM data.** `scripts/hardware.py` only probes `torch.cuda.get_device_properties(0).total_memory` (VRAM); there is zero `psutil`/`/proc/meminfo`/RSS instrumentation anywhere in `caem/` or `scripts/`. The 140 GB figure exists only on the Vast dashboard, never in a log. Every OOM in the logs is **GPU VRAM** (the batch-32 RAG blowup fixed at batch-8 via `recover_oom_baselines.sh`), never host RAM.

**Sizing verdict:**
- **96 GB → clean for both variants.** Full-RAG: fits the hot index + anonymous with ~6 GB headroom. Replay: hugely over-provisioned.
- **86 GB → OOM-safe for both; marginal for full-RAG speed** (a few GB under the ~90 GB "index fully hot" line → mild page-cache eviction + re-read thrash on the flat scan, not a crash). Fully fine for the no-index replay variant.
- **Action before next run:** add psutil host-RAM logging (peak process RSS + system `available`, once per cycle) to `scripts/hardware.py` so the ablation measures the real anonymous-vs-cache split instead of relying on a dashboard glance.

---

## 8. Caveats / open questions
- §3 is correlational + eval-proxy; it cannot prove causation or give a clean poison-vs-clean-drift % split. Only Arm 1 (oracle-clean re-train) settles it.
- TruthfulQA EM in the direct trajectory tables uses the raw `em` (ROUGE proxy); the thesis-clean number is `em_llm_judged` (Haiku). The on/off *direction* holds; absolute TruthfulQA needs the Haiku field.
- The "confident-when-wrong" rise partly reflects verifier calibration drift (composite refit each cycle), not only generator poisoning. The EM fall is unambiguously the generator.
- FEVER poison may include a high-confidence systematic component (boilerplate flips) that confidence-weighting won't catch → needs the prefix-strip signal fix.

## 9. Key evidence files
- Trajectory EM: `outputs/phase1d_eval/*_cycle{0..5}.json` (RUC-off), `outputs/full_run/eval/*_cycle{0..5}.json` (RUC-on)
- Purity: `outputs/full_run/pi_store_trajectory.csv`, `outputs/full_run/tab_purity.csv`
- Pool size: `outputs/tab_memory_store_trajectory.csv`
- Per-tier EM: `outputs/tab_per_tier_em_capability.csv`
- Run log + config: `outputs/full_run/experiment.log`, `outputs/analysis/cycle_walltime_trajectory.csv`, `caem/config.py`, `caem/training/self_improvement.py`
- RUC: `caem/ruc/v2/feature_spec.json`
- Index: `data/passage_index/` (mirror: HF `aksaN000/caem-passage-index-21m`)
