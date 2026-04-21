"""Tests for caem.model_loader (Goal 1, Branch C).

Branch C is decoder-only only — encoder-decoder backbones (Flan-T5) are
not supported. ``load_base_generator`` should load Qwen successfully and
raise ``ValueError`` on any encoder-decoder model name.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("CAEM_SKIP_MODEL_LOAD_TESTS") == "1",
    reason="Live model loads skipped via CAEM_SKIP_MODEL_LOAD_TESTS=1 "
           "(set in CI when HF downloads are not available).",
)


def test_rayon_threads_autoset():
    """Importing caem.model_loader must set RAYON_NUM_THREADS to prevent
    Rust-tokenizer panics on high-core-count hosts."""
    os.environ.pop("RAYON_NUM_THREADS", None)
    import importlib, caem.model_loader as ml
    importlib.reload(ml)
    assert "RAYON_NUM_THREADS" in os.environ, (
        "caem.model_loader import must set RAYON_NUM_THREADS"
    )
    val = os.environ["RAYON_NUM_THREADS"]
    assert val.isdigit() and int(val) >= 1, f"unexpected value: {val!r}"


def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return "cuda"


@pytest.mark.slow
def test_load_qwen_3b():
    """Qwen-2.5-3B-Instruct loads and produces coherent output."""
    from caem.model_loader import load_base_generator

    device = _device_or_skip()
    model, tok = load_base_generator(
        "Qwen/Qwen2.5-3B-Instruct",
        device=device,
        dtype=torch.bfloat16,
        use_flash_attention_2=False,  # CI host may lack flash_attn
        use_torch_compile=False,
    )
    try:
        assert tok.pad_token is not None, "pad_token must be set for batched gen"
        assert sum(p.numel() for p in model.parameters()) > 2_000_000_000, \
            "Qwen-3B should have >2B params"

        ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape[-1] >= 151_000, \
            f"unexpected vocab dim: {out.logits.shape}"

        next_id = out.logits[0, -1].argmax().item()
        text = tok.decode([next_id]).strip().lower()
        assert "paris" in text or "pa" in text, (
            f"expected Paris-like next token on 'capital of France is', got {text!r}"
        )
    finally:
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_encoder_decoder_models_rejected():
    """Loading an encoder-decoder model (Flan-T5) raises ValueError.

    Branch C is decoder-only only; attempting to load a T5 model should
    fail fast with a clear error pointing to the ``main`` branch for
    legacy T5 support.
    """
    from caem.model_loader import load_base_generator

    try:
        # Verify we can reach the HF config API without downloading weights
        from transformers import AutoConfig
        AutoConfig.from_pretrained("google/flan-t5-small")
    except Exception:
        pytest.skip("cannot reach HuggingFace to fetch T5 config")

    with pytest.raises(ValueError, match="encoder-decoder"):
        load_base_generator(
            "google/flan-t5-small",
            device="cpu",
            dtype=torch.float32,
            use_flash_attention_2=False,
            use_torch_compile=False,
        )


def test_config_defaults_branch_c():
    """CAEMConfig defaults for Branch C: Qwen-3B + Full FT with 8-bit AdamW
    primary, LoRA as structured fallback.

    The training-path doctrine flipped on 2026-04-22: Full FT + 8-bit AdamW
    fits 32 GB on a 5090 (batch=4, grad checkpointing, bf16), so LoRA moved
    from primary to first fallback (per Ch4 Cycle-2 retention cascade). See
    branch_C.md "T5 removal" and "Full FT + 8-bit AdamW as primary" sections.
    """
    from caem.config import CAEMConfig

    c = CAEMConfig()
    assert c.base_model_name == "Qwen/Qwen2.5-3B-Instruct"
    # Full FT + 8-bit AdamW is the PRIMARY path; LoRA is only consulted when
    # use_lora_training is flipped after a Full-FT failure mode.
    assert c.use_8bit_adamw is True
    assert c.use_lora_training is False
    # LoRA fallback config must still be valid so the fallback cascade can
    # activate it without code changes.
    assert c.lora_r == 16
    assert "q_proj" in c.lora_target_modules
    assert "gate_proj" in c.lora_target_modules
    # Dual-backbone fields were removed in the T5-removal refactor
    assert not hasattr(c, "prompt_style"), (
        "prompt_style was removed in the Branch-C T5-removal refactor"
    )
    assert not hasattr(c, "lora_target_modules_encoder_decoder"), (
        "lora_target_modules_encoder_decoder was removed; "
        "use lora_target_modules directly"
    )
    assert not hasattr(c, "lora_target_modules_decoder_only"), (
        "lora_target_modules_decoder_only was renamed to lora_target_modules"
    )
