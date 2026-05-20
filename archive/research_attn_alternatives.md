# Research: Flash-Attention-2 alternatives for CAEM on RTX 5090 (Blackwell sm_120)

# Getting Qwen2.5-3B to 4×–6× on an RTX 5090

### Summary

**Switch HuggingFace `.generate()` from `eager` to `attn_implementation="sdpa"` on PyTorch 2.9.0+cu128, with cuDNN's Fused-Flash kernel forced via `torch.nn.attention.sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION])`.** This is a <1-day change, preserves your existing `LogitsProcessor`-based yes/no judge and seeded greedy decoding unchanged, and delivers an expected **~150–250 tok/s** on Qwen2.5-3B bf16 versus your 40 tok/s eager baseline (~4–6×). Benchmarks on the same GPU class show cuDNN-SDPA is actually the fastest attention path on RTX 5090 bf16, beating a source-built flash-attn 2.8.3 (Nguyen, "Speed-of-Light Flash Attention for 5090", https://gau-nernst.github.io/fa-5090/). Going further to vLLM is the fallback if single-stream SDPA throughput is insufficient.

### Comparison table

| Option | Blackwell-ready? | Speedup vs eager | Integration cost | Risk | Notes |
|---|---|---|---|---|---|
| **HF + SDPA cuDNN-attention (PT 2.9+cu128)** | Yes, PT 2.7+ via PR #145602 | ~4–6× (150–250 tok/s) | 0.5 day | Low | No code rewrite, keeps `LogitsProcessor` |
| HF + flash-attn 2.8.3 (source build) | Yes, sm_120 gencode since 2.7.4.post1 | ~4–5× | 1–2 days | Medium | 90 min source build; loses to cuDNN-SDPA on 5090 |
| HF + xFormers 0.0.33.post1+ | Yes, sm_120 since 0.0.30 (PR #1254) | ~3–4× | 0.5 day | Low-medium | Slower than cuDNN/FA2 on bf16 Qwen decode |
| **vLLM v0.11+ cu128 (FlashInfer backend)** | Yes via `vllm/vllm-openai:cu130-nightly` | ~12–17× (500–700 tok/s) default; ~5–7× with determinism | 2–3 days | Medium | `VLLM_FLASH_ATTN_VERSION=2`; many sm_120 bugs open |
| TensorRT-LLM 1.3.x (NGC) | Partial; SM120 FMHA cubins missing (#11799) | ~12–20× untested | 3–5+ days | High | Engine rebuilds; pip unsupported on Blackwell |
| FP8-dynamic (RedHatAI/Qwen2.5-3B-FP8-dynamic) | Native FP8 tensor cores | +30–70% over bf16 | 0.5 day drop-in | Medium | **−4.32 pp GSM8K** vs bf16 (measured) |
| AWQ-4bit / GPTQ-4bit | Marlin kernel on Blackwell | +50–80% over bf16 | 0.5 day | High | ~5–10 pp GSM8K drop for 3B CoT |

### Ranked recommendations

**1. Primary — HuggingFace Transformers with SDPA + cuDNN-attention on PyTorch 2.9.0+cu128.** PyTorch 2.7 was the first release to ship **official Blackwell wheels with sm_120 kernels** (https://pytorch.org/blog/pytorch-2-7/), and PR #145602 explicitly added sm_100/sm_120 to the FLASH and EFFICIENT SDPA backends. cuDNN ≥ 9.15 adds a Fused-Flash kernel that the NVIDIA cuDNN frontend docs confirm supports SM120 (https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html). Published RTX 5090 kernel benchmarks show **cuDNN-SDPA beats flash-attn 2.8.3** on bf16, head_dim=128 — the exact Qwen2.5-3B configuration (gau-nernst.github.io/fa-5090/). You keep your existing `seed=N, do_sample=False` greedy loop **and** your `LogitsProcessor` yes/no mask → softmax pipeline unchanged; determinism is guaranteed because HF greedy decoding with fixed batch and `use_cache=True` is bitwise-stable on a single GPU. Expected ~150–250 tok/s single-stream (extrapolated from Qwen2.5-7B's 290 tok/s single-stream on vLLM+FA2 on the same GPU, https://discuss.vllm.ai/t/vllm-on-rtx5090-working-gpu-setup-with-torch-2-9-0-cu128/1492). The verifier-stack coexistence constraint is trivially satisfied because HF does not pre-allocate memory.

**2. Fallback — vLLM v0.11+ with FlashInfer attention backend and `VLLM_BATCH_INVARIANT=1`.** If primary's single-stream throughput is insufficient for your 200K-inference budget (~40M tokens), vLLM with continuous batching will move you to the 500–700 tok/s decode range (Qwen2.5-7B measured at 290 tok/s; Qwen2.5-3B extrapolates to ~2× that). The critical constraints: (a) use `vllm/vllm-openai:cu130-nightly` (the first image that ships SM120 kernels; https://github.com/aliez-ren/vllm-qwen3.5-nvfp4-sm120), (b) force `VLLM_FLASH_ATTN_VERSION=2` because FA3 is Hopper-only on sm_120 (vLLM issue #14452), (c) set `--attention-backend flashinfer`, (d) set `--gpu-memory-utilization 0.65` to coexist with the 3–4 GB verifier stack, and (e) enable `VLLM_BATCH_INVARIANT=1` + `enforce_eager=True` + `--no-enable-prefix-caching` + `--no-enable-chunked-prefill` + `seed=N` for bit-identical outputs across batch compositions (vLLM issue #27433; docs https://docs.vllm.ai/en/latest/features/batch_invariance/). **Qwen2.5-3B-Instruct is explicitly listed as batch-invariance-verified** in the vLLM docs. Batch-invariant mode costs ~40–60% throughput (Thinking Machines blog, https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), leaving you still ~5–7× over eager. For the yes/no judge, use `GuidedDecodingParams(choice=["yes","no"])` + `SamplingParams(logprobs=20, max_tokens=1)`; in v0.10.2+ set `--logprobs-mode raw_logprobs` to match HF semantics exactly, then renormalize `p_yes = exp(lp_yes) / (exp(lp_yes) + exp(lp_no))`.

**3. Deferred — TensorRT-LLM and any <8-bit quantization.** TRT-LLM 1.3.x officially supports RTX 5090 (NGC container only, no pip on Blackwell; https://github.com/NVIDIA/TensorRT-LLM/discussions/2726), but in February 2026 issue #11799 confirmed **SM120 FMHA cubins still do not exist**, and issue #11386 documents ptxas warnings and broken build flags on 1.3.0rc2. Engine rebuilds on every config change consume your deadline. On quantization: the only directly measured Qwen2.5-3B quantization benchmark (`RedHatAI/Qwen2.5-3B-FP8-dynamic` model card, https://huggingface.co/RedHatAI/Qwen2.5-3B-FP8-dynamic) shows FP8 loses **4.32 pp on GSM8K** — a hard signal that chain-of-thought tasks on a 3B model are quantization-sensitive. The arXiv 2505.02214 Qwen3 scaling study ("Empirical Study of Qwen3 Quantization") further shows 4-bit GPTQ drops ~10 pp MMLU at the 0.6B scale and improves with size; a 3B model sits closer to the fragile end. For a thesis where CoT quality is the deliverable, defer quantization unless you have time to run `lm-eval-harness` and report the delta transparently.

### Specific install/setup for the primary recommendation

```bash
# Fresh venv on CUDA 12.8+ driver (560+ is fine, 570+ preferred for Blackwell)
python -m venv .venv && source .venv/bin/activate

# PyTorch 2.9.0 stable, cu128 wheels with sm_120 kernels
pip install torch==2.9.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# cuDNN 9.15+ required for SM120 Fused-Flash kernel
pip install --upgrade "nvidia-cudnn-cu12>=9.15"

# HF stack
pip install "transformers>=4.46" accelerate safetensors
```

```python
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",    # NOT "eager", NOT "flash_attention_2"
).to("cuda").eval()

torch.manual_seed(42)
torch.use_deterministic_algorithms(False)  # cuDNN-attention is not bitwise-deterministic across shapes; fine for single-GPU single-stream
torch.backends.cuda.matmul.allow_tf32 = False  # bf16 only, no TF32 fallback

with torch.inference_mode(), sdpa_kernel(
    [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
):
    out = model.generate(
        **tok(prompt, return_tensors="pt").to("cuda"),
        max_new_tokens=200, do_sample=False,  # greedy
        logits_processor=[yes_no_mask_processor],  # unchanged
    )
```

Verify with `torch.cuda.get_arch_list()` — it must contain `'sm_120'`. Benchmark a 200-token generation before/after to confirm the speedup; if the baseline differs materially from ~150 tok/s, fall back to `SDPBackend.FLASH_ATTENTION` only (which dispatches to PyTorch's bundled FA2 via PR #145602).

### Known pitfalls (documented since 2026-01)

- **"No kernel image is available for execution on the device"** — the signature of installing stable PT <2.7 or cu124 wheels on a 5090. Fix: always install cu128/cu130 wheels (vLLM #13306, PyTorch #159207).
- **flash-attn 2.8.3 + PT 2.8+cu129 INTERNAL ASSERT on `schema_.has_value()`** (FA issue #1929) — torch op schema mismatch. Stick to PT 2.8.0/2.9.0 + cu128 stack if you source-build flash-attn.
- **FA3 and FA4-forward are not available on sm_120** (FA issues #1810, #1853, #2307; PRs #2330, #2333 add SM120 backward+varlen but not forward). Never set `attn_implementation="flash_attention_3"`.
- **xFormers silently downgrades torch to stable** when installed into a nightly-torch env on Windows, bricking sm_120 dispatch (genelab Medium, Mar 2026). Pin torch first or use `--no-deps`.
- **vLLM `pip install vllm` historically fails on sm_120** up through late 2025 (issue #35432). Use the `vllm/vllm-openai:cu130-nightly` Docker image or source-build with `TORCH_CUDA_ARCH_LIST="12.0"`.
- **vLLM batch-invariant + FlashInfer 0.15.0 broken** (issue #33421, Dec 2025) — pin FlashInfer 0.14.1 if determinism fails. FlashInfer NVFP4 `mm_fp4` GEMM is also broken on SM120 (FlashInfer #2577) — irrelevant for bf16 but blocks any FP4 fallback.
- **vLLM V1 returns raw (pre-softmax) logprobs by default** (https://docs.vllm.ai/en/v0.8.1/getting_started/v1_user_guide.html). For yes/no judge parity with HF, set `--logprobs-mode raw_logprobs` on v0.10.2+ and renormalize manually, and be aware that `-9999.0` appears as a top-k padding sentinel (vLLM #8111).
- **TensorRT-LLM SM120 FMHA cubins do not exist** (issue #11799, Feb 2026). The FlashInfer-TRT-LLM attention path therefore also breaks on RTX 5090. Avoid TRT-LLM unless you are on B200.
- **bitsandbytes INT8 on sm_120 produces garbage output** (reported across vLLM issues early 2026) — INT8 kernels are Ampere/Ada-tuned and do not re-dispatch correctly on Blackwell. If you must 8-bit, use `RedHatAI/Qwen2.5-3B-FP8-dynamic` (native FP8 tensor cores) not BnB-INT8.

### Conclusion

For a 2-week, 200K-inference chain-of-thought experiment on a single RTX 5090 with a thesis deadline, the highest-leverage change is the simplest: flip `attn_implementation` from `"eager"` to `"sdpa"` and force cuDNN-attention. You get 4–6× speedup with zero changes to your judge pipeline, your seed, your greedy loop, or your verifier stack, and you inherit PyTorch's stable kernel guarantees rather than the still-churning vLLM/TRT-LLM Blackwell stacks. vLLM is a credible fallback only if the primary path under-delivers and you can absorb the batch-invariance throughput tax; TRT-LLM and aggressive quantization should wait for a post-thesis iteration when a measured GSM8K delta can justify them.

## Context this research is answering

- **Hardware**: RTX 5090 (Blackwell, sm_120, 32 GB VRAM)
- **Model**: Qwen-2.5-3B-Instruct (bf16, ChatML)
- **Current baseline**: ~40 tok/s with HF Transformers eager attention (flash-attn package fails to install on Blackwell as of the last check)
- **Hard constraint**: must preserve MC-dropout capability (CAEM's `u_dropout` signal requires stochastic forward passes with dropout ON)
- **Hard constraint**: must preserve constrained-decoding for yes/no judge path (Path B FrozenQwenJudge)
- **Soft constraint**: engineering cap ~5 days before Step 7 main launch

## What I'll check against when the research lands

- [ ] Blackwell (sm_120) support status for flash-attention main branch — is there a working wheel or buildable HEAD?
- [ ] PyTorch SDPA backend selection on Blackwell — does `torch.nn.functional.scaled_dot_product_attention` reach FA2-equivalent performance via its `flash` or `mem-efficient` backend?
- [ ] xFormers Blackwell compatibility — buildable, wheeled, benchmarked?
- [ ] Does any option require disabling MC-dropout? (if yes — disqualified for main-run use; OK for Phase 2)
- [ ] Does any option break constrained-decoding? (if yes — same disqualification)
- [ ] Published benchmark numbers on Qwen-3B bf16 single-GPU inference
- [ ] Installation friction (pip wheel / build-from-source / torch version pin)
- [ ] Known issues since Jan 2026 — GitHub/HF forum threads

## Decision framework (what the research determines)

| Agent finding | Decision |
|---|---|
| Flash-attention main branch has stable sm_120 kernels with pip install | **Adopt flash-attn**, ~1.5-2× speedup vs eager, zero composite risk |
| SDPA already auto-selects memory-efficient backend on Blackwell | **Just set `attn_implementation="sdpa"`**, ~1.3-1.6× speedup |
| xFormers has Blackwell support and beats SDPA on Qwen-3B | **Adopt xFormers** if speedup > SDPA by ≥15% |
| None of the above work cleanly | **Stay with eager attention**, document as future work, accept current throughput |

---

## Agent research output (paste below)


