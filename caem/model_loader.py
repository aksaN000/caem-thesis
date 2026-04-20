"""
caem/model_loader.py
======================
Base-generator loader for CAEM Branch C — decoder-only only.

Loads decoder-only instruction-tuned models (Qwen-2.5, Gemma-2, Llama-3.2,
Phi-3, Mistral) via ``AutoModelForCausalLM``. Handles Flash Attention 2
(Goal 5, opt-in), bf16 dtype, torch.compile (Goal 5, opt-in), device
placement, and the ``pad_token = eos_token`` convention required for batched
generation on decoder-only models.

**Branch C decision (2026-04-22)**: Flan-T5 / encoder-decoder backbones are
NOT supported on this branch. The halted Phase-1a Flan-T5 pilot data remains
archived on HF (aksaN000/caem-passage-index-21m/phase_1a_flan_t5_halted) as
a historical artefact but is not referenced in the thesis. For fallback to
the T5 codebase, check out the ``main`` branch — branch-C is Qwen-only by
design. See branch_C.md §"T5 removal (2026-04-22)" for the rationale.

Passing an encoder-decoder model name (e.g., ``google/flan-t5-large``) raises
``ValueError`` at load time; we fail fast rather than silently producing
wrong results (decoder-only's full-stack forward semantics don't match T5's
encoder-decoder structure).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


_ensure_rayon_threads_set()


def load_base_generator(
    model_name: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_flash_attention_2: bool = True,
    use_torch_compile: bool = False,
) -> Tuple[Any, Any]:
    """Load a decoder-only base generator.

    Parameters
    ----------
    model_name
        HuggingFace repo ID (e.g., ``"Qwen/Qwen2.5-3B-Instruct"``). MUST be a
        decoder-only model — encoder-decoder models (Flan-T5) raise
        ``ValueError``.
    device
        Target device. Defaults to ``"cuda"``; tests on CPU-only hosts should
        pass ``"cpu"``.
    dtype
        Weight dtype. Defaults to bf16 (native on RTX 5090 / 40xx / A100). For
        older GPUs pass ``torch.float16``.
    use_flash_attention_2
        If True, attempts ``attn_implementation="flash_attention_2"`` when
        the ``flash_attn`` package is installed. Gracefully falls back to
        eager attention with a WARNING log if unavailable.
    use_torch_compile
        If True, applies ``torch.compile(mode="reduce-overhead")``.
        Nondeterminism risk (~1e-4 logit drift) is below u_stored composite
        noise floor per Branch-C perf analysis; safe for inference forward
        passes. Leave False during initial port and smokes; enable after
        regression-gate validation.

    Returns
    -------
    model
        ``PreTrainedModel`` loaded on ``device`` with ``dtype``, in eval mode.
    tokenizer
        Matching ``PreTrainedTokenizer`` with ``pad_token = eos_token`` if the
        tokenizer lacked a ``pad_token`` (needed for batched generation).

    Raises
    ------
    ValueError
        If ``model_name`` resolves to an encoder-decoder architecture.
    """
    hf_config = AutoConfig.from_pretrained(model_name)
    if getattr(hf_config, "is_encoder_decoder", False):
        raise ValueError(
            f"load_base_generator: {model_name!r} is an encoder-decoder model "
            "(is_encoder_decoder=True). Branch C supports decoder-only only; "
            "encoder-decoder backbones (Flan-T5 family) are not supported. "
            "Use the `main` branch for the legacy T5 codebase. "
            "See branch_C.md §'T5 removal (2026-04-22)' for rationale."
        )

    model_kwargs: dict = {"dtype": dtype}
    flash_attn_requested_but_unavailable = False
    if use_flash_attention_2:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled.")
        except ImportError:
            flash_attn_requested_but_unavailable = True

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError as exc:
        # transformers < 4.45 still uses `torch_dtype`; retry with legacy name
        if "dtype" in model_kwargs:
            logger.info(
                "Falling back to legacy torch_dtype kwarg (transformers<4.45): %s",
                exc,
            )
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            raise

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Decoder-only models commonly ship without a default pad_token. Set to eos
    # so batched generation produces correct attention masks.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            "Tokenizer lacked pad_token; aliased pad_token = eos_token for "
            "batched-generation correctness."
        )

    model.to(device)
    model.eval()

    if use_torch_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile applied (reduce-overhead mode).")
        except Exception as exc:  # compile backend failures are driver/version-dependent
            logger.warning(
                "torch.compile failed (%s); continuing without compilation.",
                exc,
            )

    if flash_attn_requested_but_unavailable:
        logger.warning(
            "use_flash_attention_2=True but `flash_attn` is not installed; "
            "loaded with eager attention. Install with: "
            "`pip install flash-attn --no-build-isolation` on a CUDA-enabled host."
        )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded base generator: name=%s  params=%s  device=%s  dtype=%s  "
        "flash_attn_2=%s  compiled=%s",
        model_name,
        f"{n_params:,}",
        device,
        dtype,
        "yes" if "attn_implementation" in model_kwargs else "no",
        "yes" if use_torch_compile else "no",
    )
    return model, tokenizer


__all__ = ["load_base_generator"]
