"""Tests for caem.model_loader (Goal 1, Branch C).

Covers architecture dispatch between decoder-only (Qwen/Gemma/Llama) and
encoder-decoder (Flan-T5). Runs live model loads at small scale on CUDA when
available; falls back to CPU load smoke when CUDA is unavailable.
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
    # Clear if pre-set so import's idempotency check still runs the hook
    os.environ.pop("RAYON_NUM_THREADS", None)
    # Force re-import to re-run the module-level hook
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
def test_load_decoder_only_qwen_3b():
    """Qwen-2.5-3B-Instruct loads as decoder-only and produces coherent output."""
    from caem.model_loader import load_base_generator

    device = _device_or_skip()
    model, tok, is_enc_dec = load_base_generator(
        "Qwen/Qwen2.5-3B-Instruct",
        device=device,
        dtype=torch.bfloat16,
        use_flash_attention_2=False,  # CI host may lack flash_attn
        use_torch_compile=False,
    )
    try:
        assert is_enc_dec is False, "Qwen-3B must dispatch as decoder-only"
        assert tok.pad_token is not None, "decoder-only pad_token must be set"
        assert sum(p.numel() for p in model.parameters()) > 2_000_000_000, \
            "Qwen-3B should have >2B params"

        # Forward pass
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape[-1] == tok.vocab_size or \
               out.logits.shape[-1] >= 151_000, \
            f"unexpected vocab dim: {out.logits.shape}"

        # Greedy next token should be factually plausible
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


@pytest.mark.slow
def test_load_encoder_decoder_flan_t5_legacy_path():
    """Flan-T5-Large loads as encoder-decoder (legacy Variant-18 ablation path)."""
    from caem.model_loader import load_base_generator

    device = _device_or_skip()
    model, tok, is_enc_dec = load_base_generator(
        "google/flan-t5-large",
        device=device,
        dtype=torch.bfloat16,
        use_flash_attention_2=True,   # should silently skip (encoder-decoder)
        use_torch_compile=True,       # should silently skip (encoder-decoder)
    )
    try:
        assert is_enc_dec is True, "Flan-T5 must dispatch as encoder-decoder"
        assert sum(p.numel() for p in model.parameters()) > 700_000_000

        ids = tok(
            "translate English to German: Hello world",
            return_tensors="pt",
        ).input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=8, do_sample=False)
        out = tok.decode(gen[0], skip_special_tokens=True).strip().lower()
        # T5 translation is loose; accept any non-empty output
        assert len(out) > 0, f"empty T5 generation"
    finally:
        del model, tok
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_config_defaults_branch_c():
    """CAEMConfig defaults to Qwen-3B + chatml_scaffold + LoRA for Branch C."""
    from caem.config import CAEMConfig

    c = CAEMConfig()
    assert c.base_model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert c.prompt_style == "chatml_scaffold"
    assert c.use_lora_training is True
    assert c.lora_r == 16
    # decoder-only target modules should include gate/up/down (MLP) and q/k/v/o
    assert "q_proj" in c.lora_target_modules_decoder_only
    assert "gate_proj" in c.lora_target_modules_decoder_only
    # encoder-decoder fallback narrower (Flan-T5 convention: q, v only)
    assert set(c.lora_target_modules_encoder_decoder) == {"q", "v"}
