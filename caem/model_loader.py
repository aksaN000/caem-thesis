"""
caem/model_loader.py
======================
Architecture-dispatching base-generator loader (Goal 1, Branch C).

Dispatches ``AutoModelForCausalLM`` (Qwen-3B, Gemma, Llama, Phi, Mistral) vs
``AutoModelForSeq2SeqLM`` (Flan-T5 family) based on the HuggingFace config's
``is_encoder_decoder`` flag. Handles Flash Attention 2 (Goal 5), bf16 dtype,
torch.compile (Goal 5, opt-in), device placement, and the decoder-only
``pad_token = eos_token`` convention required for batched generation.

The returned ``is_encoder_decoder`` flag is the single source of truth that
callers use to dispatch:

- prompt-builder style (``prompts.py::build_tier_prompt`` picks chatml vs flan-t5)
- forced-prefix mechanism (``decoder_input_ids`` for encoder-decoder, prefill text
  for decoder-only)
- LoRA target modules (``lora_target_modules_decoder_only`` vs
  ``lora_target_modules_encoder_decoder`` in ``CAEMConfig``)

Existing scripts currently hardcode ``T5ForConditionalGeneration.from_pretrained``;
they migrate to ``load_base_generator`` in a follow-up commit. Both paths remain
valid — the architecture dispatch is where the divergence happens, not at the
import.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


def _ensure_rayon_threads_set() -> None:
    """Mirror OMP thread cap into ``RAYON_NUM_THREADS`` BEFORE tokenizer import.

    HuggingFace fast tokenizers use Rust's rayon threadpool. On high-core-count
    hosts (EPYC 96-core Vast rentals) with OMP capped at 16, rayon tries to
    spawn ~N_cpus threads and hits the pthread rlimit, panicking with
    ``ThreadPoolBuildError: Resource temporarily unavailable``. Setting
    ``RAYON_NUM_THREADS`` to match OMP prevents this.
    """
    if "RAYON_NUM_THREADS" in os.environ:
        return
    # Fall back to OMP cap, then conservative default 16
    target = (
        os.environ.get("OMP_NUM_THREADS")
        or os.environ.get("MKL_NUM_THREADS")
        or "16"
    )
    os.environ["RAYON_NUM_THREADS"] = target
    logger.info(
        "Set RAYON_NUM_THREADS=%s (mirrors OMP cap) to prevent tokenizer panic "
        "on high-core-count hosts.",
        target,
    )


# Apply at import time so callers that import AutoTokenizer directly also benefit.
_ensure_rayon_threads_set()


def load_base_generator(
    model_name: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_flash_attention_2: bool = True,
    use_torch_compile: bool = False,
) -> Tuple[Any, Any, bool]:
    """Load the base generator using the correct HF class for its architecture.

    Parameters
    ----------
    model_name
        HuggingFace repo ID (e.g., ``"Qwen/Qwen2.5-3B-Instruct"`` or
        ``"google/flan-t5-large"``). Architecture is detected from the
        downloaded config's ``is_encoder_decoder`` flag.
    device
        Target device. Defaults to ``"cuda"``; callers on CPU (tests) should
        pass ``"cpu"``.
    dtype
        Weight dtype. Defaults to bf16 (native on RTX 5090/40xx/A100; works
        on newer consumer cards). For older GPUs pass ``torch.float16``.
    use_flash_attention_2
        If True, attempts to load with ``attn_implementation="flash_attention_2"``
        when the ``flash_attn`` package is installed AND the architecture is
        decoder-only. Gracefully falls back to eager attention with a WARNING
        log line if either condition fails. Encoder-decoder models get eager
        attention regardless (Flan-T5's flash-attention story in transformers
        is still incomplete as of 4.x).
    use_torch_compile
        If True, applies ``torch.compile(mode="reduce-overhead")`` to the
        returned model when the architecture is decoder-only. Nondeterminism
        risk (~1e-4 logit drift) is below u_stored composite noise floor per
        Branch-C perf analysis; safe for inference forward passes. Leave
        False during initial port and smokes; enable after regression-gate
        validation.

    Returns
    -------
    model
        ``PreTrainedModel`` loaded on ``device`` with ``dtype``, in eval mode.
    tokenizer
        Matching ``PreTrainedTokenizer``. For decoder-only models,
        ``pad_token`` is set to ``eos_token`` if missing (needed for batched
        generation to produce correct attention masks).
    is_encoder_decoder
        True for Flan-T5 family; False for decoder-only families. Callers
        dispatch on this flag for prompt style, forced-prefix mechanism, and
        LoRA target modules.
    """
    hf_config = AutoConfig.from_pretrained(model_name)
    is_enc_dec = bool(getattr(hf_config, "is_encoder_decoder", False))

    # transformers 4.x is deprecating `torch_dtype` in favor of `dtype`; we try
    # `dtype=` first and fall back if the installed version is older.
    model_kwargs: dict = {"dtype": dtype}
    flash_attn_requested_but_unavailable = False
    if use_flash_attention_2 and not is_enc_dec:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled on decoder-only model.")
        except ImportError:
            flash_attn_requested_but_unavailable = True

    def _load_with_kwargs(klass, kwargs):
        try:
            return klass.from_pretrained(model_name, **kwargs)
        except TypeError as exc:
            # transformers < 4.45 uses `torch_dtype`; retry with the old name.
            if "dtype" in kwargs and "torch_dtype" not in kwargs:
                logger.info(
                    "Falling back to legacy torch_dtype kwarg (transformers<4.45): %s",
                    exc,
                )
                legacy = {**kwargs, "torch_dtype": kwargs.pop("dtype")}
                return klass.from_pretrained(model_name, **legacy)
            raise

    if is_enc_dec:
        model = _load_with_kwargs(AutoModelForSeq2SeqLM, model_kwargs)
    else:
        model = _load_with_kwargs(AutoModelForCausalLM, model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Decoder-only models commonly ship without a default pad_token; set to eos
    # to unblock batched generation. Encoder-decoder models (T5 family) already
    # have pad_token_id from the original training.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            "Tokenizer lacked pad_token; aliased pad_token = eos_token "
            "for batched-generation correctness."
        )

    model.to(device)
    model.eval()

    if use_torch_compile and not is_enc_dec:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile applied to generator (reduce-overhead mode).")
        except Exception as exc:  # defensive: compile backend failures vary
            logger.warning(
                "torch.compile failed (%s); continuing without compilation.",
                exc,
            )
    elif use_torch_compile and is_enc_dec:
        logger.info(
            "use_torch_compile=True requested but model is encoder-decoder; "
            "skipping compile (Flan-T5 compile path is fragile in transformers 4.x)."
        )

    if flash_attn_requested_but_unavailable:
        logger.warning(
            "use_flash_attention_2=True but `flash_attn` is not installed; "
            "loaded with eager attention. Install with: "
            "`pip install flash-attn --no-build-isolation` on a CUDA-enabled host."
        )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded base generator: name=%s  architecture=%s  params=%s  "
        "device=%s  dtype=%s  flash_attn_2=%s  compiled=%s",
        model_name,
        "encoder-decoder" if is_enc_dec else "decoder-only",
        f"{n_params:,}",
        device,
        dtype,
        "yes" if "attn_implementation" in model_kwargs else "no",
        "yes" if (use_torch_compile and not is_enc_dec) else "no",
    )
    return model, tokenizer, is_enc_dec


__all__ = ["load_base_generator"]
