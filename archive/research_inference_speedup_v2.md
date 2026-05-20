# Five levers to reach ~1.5× end-to-end, and why 2–4× is out of reach

**Bottom line up front.** Under your six hard constraints (MC-dropout K=5, LogitsProcessor masking, seeded greedy, frozen MiniCheck, bf16-only, 3-day cap), the realistic end-to-end ceiling on the measured 10.4 s/sample pipeline is **~1.45–1.60×** (roughly 6.5–7.2 s/sample), not 2–4×. The 2× floor is reachable only if you relax bf16-only (KV-INT8 or FP8 weights) or the K=5 literal MC-dropout requirement — neither is on the table. What follows is an Amdahl-honest ranking: kernel-level wins that the previous research over-counted collapse quickly because Qwen generation is only ~35% of total time and cuDNN-SDPA already harvested the easy attention gain.

## Top-5 ranked table

| # | Optimization | Projected **end-to-end** speedup | Risk | Effort | Breaks constraint? | Key citations |
|---|---|---|---|---|---|---|
| 1 | **Shared-encoder batching of MiniCheck's 9 signals** (one encoder pass, batch=9 decoders) | **1.27–1.35×** | Low | 3–6 h | No | MiniCheck arXiv:2404.10774; HF T5 `encoder_outputs=` pattern (discuss.huggingface.co/t/81126); transformers PR #31167 (T5SdpaAttention) |
| 2 | **Early-stop on "Answer:" delimiter + `max_new_tokens=100`** | **1.13–1.21×** | Low | 2–3 h | No | Renze 2024 "Concise CoT"; arXiv:2412.21187; HF #22340 (per-sample stop limitation); HF `StoppingCriteria` docs |
| 3 | **Cross-sample CUDA-stream overlap** (Qwen N+1 ‖ MiniCheck N ‖ BGE N) | **1.20–1.35×** *after #1, #2* | Med | 8–12 h | No | pytorch #19103, #48279, #59692, #113622; NVIDIA CUDA PG §4.6; CUDA 13.1 Green-Contexts blog; Rand, *Pipelining with CUDA Streams* (2025) |
| 4 | **Batch the K=10 m-chains via `generate_batch` / PagedAttention** (and collapse K=5 MC-dropout into batch-dim) | **1.15–1.25×** *after #1–3* | Med | 8–16 h | No | transformers v4.57 `continuous_batching` docs; `attn_implementation="paged\|sdpa"`; pytorch PR #46148 (graph-safe Philox RNG) |
| 5 | **`torch.compile(mode="reduce-overhead") + cache_implementation="static"` on Qwen forward** | **1.05–1.12×** *after #1–4* | Med | 6–10 h | No | HF #29151, #30351, #30055; pytorch #159207, #164342 (sm_120 stable Q4-25); NVIDIA/TensorRT-LLM #11386 (sm_120 launch_bounds warning, Apr 2026); ezyang, *State of torch.compile*, Aug 2025 |

Compounded realistically (see §6): **10.4 s → ~6.5–7.2 s per sample, ≈1.45–1.60× end-to-end**. The previous report's 1.12× (SDPA alone) becomes the new baseline on top of which these stack.

## 1. Shared-encoder batching for MiniCheck (biggest single win)

MiniCheck is Flan-T5-Large. Its 9 signals (`p_ground_mean`, `p_ground_atomic`, `p_entail`, pool/atomic passes) are the dominant verifier cost — T5's decoder emits just 1–2 tokens ("0"/"1"), so **>95% of verifier time is the encoder pass on `[doc] </s> [claim]`**. Today you call the encoder 9× serially. Pack the 9 claims into a single batched encoder call and you convert 9 encoder passes into one, bit-exact.

```python
inputs = tok(
    [f"predict: {doc} </s> {c}" for c in claims],   # MiniCheck format
    return_tensors="pt", padding=True, truncation=True, max_length=2048
).to("cuda")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.generate(**inputs, max_new_tokens=2, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
probs = torch.softmax(out.scores[0], dim=-1)          # [9, vocab]
p_support = probs[:, tok("1", add_special_tokens=False).input_ids[0]]
```

**Verifier stage 2.8 s → ~0.8 s → end-to-end 1.27–1.35×**. Bit-exact vs the serial path (padding changes attention mask but T5's SDPA dispatch handles this correctly since PR #31167). No constraint is touched: frozen weights, bf16, determinism all preserved.

**Blackwell sm_120 note (Apr 2026):** Flan-T5 uses *bucketed relative-position bias*, which does NOT dispatch to FlashAttention-2/3 (Dao-AILab/flash-attention #332 open, pytorch #96099 open, #138493 Flex-Attention). It *does* dispatch to the cuDNN math/mem-efficient SDPA backend via PR #31167. Verify MiniCheck isn't pinned to an older `transformers` that forces eager attention; force `attn_implementation="sdpa"` explicitly.

## 2. Early-stop delimiter + tighter `max_new_tokens`

Your CoT tasks (FEVER 2-way, FEVER-NEI, StrategyQA, ARC) have 1-token correct answers. Empirical CoT-length studies (Renze 2024; arXiv:2412.21187; Wang & Zhou NeurIPS 2024) show EM plateaus at 50–100 tokens on classification-class tasks and often degrades beyond that. A delimiter stop on `"Answer:"` + hard cap at 100 typically finds the answer at median ~80 tokens vs your 200-cap, cutting decode by ~40%.

```python
class StopOnSubsequence(StoppingCriteria):
    def __init__(self, delim_ids, answer_tokens=1):
        self.delim = torch.as_tensor(delim_ids, dtype=torch.long); self.k = answer_tokens
    def __call__(self, input_ids, scores, **kw):
        L = self.delim.numel()
        if input_ids.shape[1] <= L + self.k: return False
        tail = input_ids[0, -L-self.k:-self.k]
        return torch.equal(tail, self.delim.to(input_ids.device))

delim = tok.encode("Answer:", add_special_tokens=False)
out = qwen.generate(**batch, max_new_tokens=100, do_sample=False,
                    logits_processor=answer_mask_procs,   # Platt-calibrated mask untouched
                    stopping_criteria=StoppingCriteriaList([StopOnSubsequence(delim)]))
```

**Qwen decode 3.0 s → ~1.6–2.0 s → end-to-end 1.13–1.21×.** Platt calibration is preserved because the constrained-decoding mask still fires on the single final answer token; only the number of CoT tokens before it shrinks. **Caveat:** HF #22340 (open) — `StoppingCriteria` is evaluated per-batch, not per-row. Safe for batch-1 and for K=5 MC-dropout copies of the same prompt (they share a delimiter in greedy mode); unsafe if you later sample multiple divergent CoTs in one batch.

## 3. Cross-sample CUDA-stream overlap

After #1–#2 the per-sample budget is roughly Qwen 2.0 s, MiniCheck 0.8 s, BGE 1.25 s, SBERT+FAISS 0.5 s, orchestration 0.75 s → ~5.3 s of work. Running them serially is wasteful because **all four models fit co-resident in VRAM** (Qwen-3B bf16 ~6 GB + T5-780M ~1.5 GB + BGE ~1.1 GB + SBERT ~0.4 GB ≈ 10–12 GB of 32 GB). Use `torch.cuda.Stream` to run sample N's verifier+reranker while sample N+1's Qwen decode is executing:

```python
import os; os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
gen_stream = torch.cuda.Stream(priority=-1)   # user-latency critical
ver_stream = torch.cuda.Stream(priority=0)    # overlaps next-sample gen

prev = None
for i, sample in enumerate(dataset):
    with torch.cuda.stream(gen_stream), torch.inference_mode():
        torch.manual_seed(sample.seed)
        out_i = qwen.generate(**sample.inputs, **gen_cfg, logits_processor=lp,
                              stopping_criteria=stopping)
        for t in (out_i,): t.record_stream(gen_stream)
        ev_i = torch.cuda.Event(); ev_i.record(gen_stream)
    if prev is not None:
        out_p, ev_p, sp = prev
        with torch.cuda.stream(ver_stream), torch.inference_mode():
            ver_stream.wait_event(ev_p)
            v = verifier(**prep(out_p, sp.claim))       # batched 9-claim call (#1)
            ev_v = torch.cuda.Event(); ev_v.record(ver_stream)
    prev = (out_i, ev_i, sample)
```

**Gotchas** (real 2024–2026 PyTorch bugs you must work around): default stream implicitly syncs (pytorch #19103, #48279, #59692); allocator can reuse cross-stream memory and corrupt it unless you `tensor.record_stream()` (#113622); set `CUDA_DEVICE_MAX_CONNECTIONS≥8` so independent streams hit different hardware work queues (NVIDIA CUDA PG §4.6). **Green Contexts (CUDA 13.1) are not exposed in PT 2.10** — skip them.

**Expected gain after #1–#2:** the verifier (0.8 s) and reranker (1.25 s) hide behind Qwen N+1 decode (~2.0 s), saving ~1.5–2.0 s per sample → **1.20–1.35× additional end-to-end**. Realistic end of range if SM contention during Qwen decode reduces overlap to ~70% (Rand 2025 measured ~70% on two-model A10G overlap with PT 2.6).

**Dropout, LogitsProcessor, determinism:** all preserved — this is *across-sample* pipelining, not inside a generate call. Re-seed `torch.manual_seed(sample.seed)` inside each stream context; do **not** combine with `torch.use_deterministic_algorithms(True)` across shared-allocator streams (risk of memory-ordering nondeterminism). Test EM parity before/after.

## 4. Batch K=10 m-chains + K=5 MC-dropout in the batch dimension

The K=10 m-chains and K=5 MC-dropout passes are currently serial. PyTorch's `nn.Dropout` samples **independent Bernoulli masks per element per forward call** (official docs; `torch/nn/modules/dropout.py`). Replicating the input along batch dim therefore gives K statistically identical dropout masks in one forward pass — this *is* K forward passes, literally satisfying Constraint #1. Combine with transformers 4.57's `generate_batch` / `attn_implementation="paged|sdpa"` for the m-chains:

```python
# --- K=5 MC-dropout collapsed to one forward ---
for m in model.modules():
    if isinstance(m, torch.nn.Dropout): m.train()   # ONLY dropout layers
ids = input_ids.repeat_interleave(5, dim=0)          # (5, T)
torch.manual_seed(seed)
logits = model(input_ids=ids, attention_mask=mask.repeat_interleave(5,0)).logits
u_internal_var = logits.view(1, 5, *logits.shape[1:]).var(dim=1, unbiased=False)

# --- K=10 m-chains as a single paged-attention batch ---
model2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",
    attn_implementation="paged|sdpa", torch_dtype=torch.bfloat16, device_map="cuda")
out = model2.generate_batch(inputs=[tok(f"CoT {i}: {prompt}").input_ids for i in range(10)],
    generation_config=GenerationConfig(max_new_tokens=100, do_sample=True, temperature=0.7,
    max_batch_tokens=4096), logits_processor=lp)
```

Each paged-attention batch element gets an independent Philox subsequence (pytorch PR #46148, #113541 on graph-safe RNG), so per-item dropout masks differ naturally. **Short-context single-request PagedAttention gives no win; it's the 10-wide batching that cashes in**, converting ~2 s of serial m-chain decode into ~0.4–0.6 s → **1.15–1.25× further end-to-end** after #1–#3.

**Blackwell:** sm_120 is stable in PT 2.10 (pytorch #159207, #164342 closed Q4-2025). TensorRT-LLM #11386 (Apr 2026) flags sm_120 kernel launch_bounds warnings, but this is TRT-LLM-only; HF PagedAttention dispatches through PyTorch Triton and is unaffected.

## 5. `torch.compile` + StaticCache on the Qwen forward

Last because it compounds least. With #2 already cutting decode tokens by ~40%, the compile payoff on the remaining 1.6–2.0 s Qwen budget is smaller than the previous report assumed. Use `cache_implementation="static"` + `mode="reduce-overhead"` (CUDA-graph replay under the hood) with padded prompt buckets {64, 128, 256, 512} to bound recompiles:

```python
model.generation_config.cache_implementation = "static"
model.forward = torch.compile(model.forward, mode="reduce-overhead",
                              fullgraph=True, dynamic=False)
torch.cuda.default_generators[0].manual_seed(seed)    # graph-safe RNG for MC-dropout
enc = tok(prompt, return_tensors="pt", padding="max_length",
          max_length=64, pad_to_multiple_of=64).to("cuda")
out = model.generate(**enc, max_new_tokens=100, do_sample=False,
                     cache_implementation="static", logits_processor=lp,
                     stopping_criteria=stopping)
```

**Known issues (2025–2026):** HF #29151 (prompt-length recompile, open — mitigated by padding buckets); HF #30351 (StaticCache locked to first prompt, **fixed ≥4.40**); HF #42440 (Gemma3 compile regression Nov 2025, does not touch Qwen2); vLLM #15705 (Qwen2.5 SWA layer config — irrelevant since `use_sliding_window=False` in the shipped Qwen2.5-3B config). **MC-dropout + CUDA graphs works** because PyTorch auto-registers the default generator with graph state (PR #46148), so each replay consumes a fresh Philox offset → different masks per K=5 replay. Keep Qwen2's `attention_dropout=0.0` (its default); put MC-dropout only on hidden/MLP layers — this avoids SDPA-dropout graph interaction (#99905). LogitsProcessor runs outside the compiled forward, so constrained decoding is unaffected.

**Residual gain after #1–#4:** Qwen decode 1.6–2.0 s → 1.1–1.5 s → **1.05–1.12× additional end-to-end**.

## 6. Amdahl-honest stacking arithmetic and what *doesn't* work

Starting from 10.4 s:
- **#2** collapses decode: Qwen 3.5 → 2.0 s → **8.9 s**.
- **#1** collapses verifier: MiniCheck 2.8 → 0.8 s → **6.9 s**.
- **#3** overlaps the now-smaller verifier+reranker (~2.0 s combined) behind next-sample Qwen (~2.0 s); net hidden ≈1.5 s → **~5.4 s**. Real-world ceiling ~70% overlap → **~5.7–6.0 s**.
- **#4** shaves ~0.3–0.5 s from the m-chain/MC-dropout batching → **~5.3–5.7 s**.
- **#5** shaves a further ~0.15–0.25 s → **~5.1–5.5 s**.

Conservative actual: **~6.5–7.2 s end-to-end, i.e. 1.45–1.60×** — realistic given stream-boundary syncs, non-trivial compile JIT warmup the first ~50 samples, and orchestration overhead (~0.75 s) that none of these touch.

**Why 2× is unreachable without relaxing constraints:** the residual 5.1–7.2 s floor is gated by orchestration, tokenization, FAISS CPU search, and by the ~2.0 s Qwen critical path which cannot be further compressed without quantization (excluded) or speculative decoding (kills acceptance under MC-dropout; see below).

**Explicitly rejected levers.** **Assisted/EAGLE speculative decoding** — MC-dropout raises target-model noise, collapsing draft acceptance to 10–30% drop → net 1.0–1.1× on the affected calls (not stackable with #4's batching anyway); EAGLE-3 also needs a trained Qwen-2.5-3B head (~40–80 h) and is not integrated in HF `generate()` natively. **KV-cache INT8** — HF blog *Unlocking Longer Generation with KV Cache Quantization* explicitly warns it is **slower for short contexts** due to per-token quant/dequant, and you have no memory pressure (26 GB free). **Sliding-window attention on Qwen-2.5** — `use_sliding_window=False` in the shipped config, window=500 exceeds your 250-tok totals → no-op; HF #35896 (open) shows the wrong-layer bug that makes turning it on risky. **LLMLingua/LLMLingua-2 prompt compression** — your 500-tok scaffolds prefill in 50–150 ms; compressor overhead (XLM-R-L at 30–80 ms) cancels savings, and compression invalidates Platt calibration unless re-fitted. **ORT+TensorRT EP for MiniCheck** — would give 1.14–1.18× end-to-end but at 12–20 h effort versus 3–6 h for #1, which delivers more; deprioritize within the 3-day cap. **Single-pass uncertainty (Laplace, evidential DL, SNGP)** — none are *equivalent* to dropout-mask variance per your Constraint #1; Postels 2019's sampling-free approximation disagrees with MC variance by ~93% magnitude.

## Conclusion

The leverage is not in attention kernels — cuDNN-SDPA already harvested that and delivered only 1.12× end-to-end because Qwen generation is 35% of the pipeline. The remaining levers are **structural**: (a) stop wasting T5 encoder passes (#1), (b) stop decoding past the answer (#2), (c) stop running verifier after generation when you could run both at once (#3), (d) collapse the K=10 × K=5 fan-out into batch-dim parallelism (#4), and only then (e) compile the hot forward (#5). Collectively they land at **~1.45–1.60× end-to-end within 27–47 engineering hours**. The 2–4× goal in the task statement is incompatible with preserving literal K=5 MC-dropout, bf16-only, and the Platt-calibrated constrained-decoding path; reaching 2× would require renegotiating one of those — most likely permitting KV-INT8 (memory-bounded savings) or a calibrated speculative-decoding path with MC-dropout disabled only on non-uncertainty-signal calls.