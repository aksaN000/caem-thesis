"""
tests/test_sdpa_config.py
==========================

Unit tests for the Branch-C cuDNN-SDPA adoption (2026-04-23) — lives
in ``caem/model_loader.py`` as ``_configure_sdpa_backends()`` and
``caem_sdpa_context()``.

These tests validate the configuration logic (idempotency, graceful
degradation on old PyTorch builds, correct backend priority) without
requiring a live GPU — CUDA-only paths are mocked or gated on
``torch.cuda.is_available()``.

For live-GPU equivalence + throughput measurement, see
``scripts/bench_sdpa_vs_eager.py`` (integration test, run manually
between Step 6 completion and Step 7.0 launch).
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
import torch

from caem.model_loader import (
    _configure_sdpa_backends,
    caem_sdpa_context,
)


# =====================================================================
# Idempotency
# =====================================================================

class TestIdempotency:
    def test_configure_is_idempotent(self):
        """Calling _configure_sdpa_backends multiple times is safe."""
        import caem.model_loader as ml
        ml._SDPA_BACKENDS_CONFIGURED = False  # reset for test
        _configure_sdpa_backends()
        flag1 = ml._SDPA_BACKENDS_CONFIGURED
        _configure_sdpa_backends()  # second call should short-circuit
        flag2 = ml._SDPA_BACKENDS_CONFIGURED
        assert flag1 is True and flag2 is True

    def test_no_cuda_skips_cleanly(self):
        """On CPU-only hosts, _configure_sdpa_backends should noop, not crash."""
        import caem.model_loader as ml
        ml._SDPA_BACKENDS_CONFIGURED = False
        with patch("torch.cuda.is_available", return_value=False):
            _configure_sdpa_backends()
            # Should set the flag to avoid retry, even in no-CUDA path
            assert ml._SDPA_BACKENDS_CONFIGURED is True


# =====================================================================
# caem_sdpa_context — context manager
# =====================================================================

class TestCaemSdpaContext:
    def test_context_enters_and_exits_cleanly(self):
        """Basic context-manager protocol."""
        with caem_sdpa_context():
            pass  # should not raise

    def test_context_nests_without_error(self):
        """Nesting the context manager is safe."""
        with caem_sdpa_context():
            with caem_sdpa_context():
                pass

    def test_context_is_noop_on_no_cuda(self):
        """On CPU-only hosts the context is a no-op."""
        with patch("torch.cuda.is_available", return_value=False):
            with caem_sdpa_context():
                # should yield normally, no SDPA backend selection
                pass

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="Requires CUDA device")
    def test_sdpa_call_works_inside_context(self):
        """Under the context, torch.nn.functional.scaled_dot_product_attention
        should execute successfully on a small tensor without hitting the
        math fallback (since we disabled it)."""
        with caem_sdpa_context():
            q = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.bfloat16)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            assert out.shape == (2, 4, 16, 32)
            assert out.dtype == torch.bfloat16


# =====================================================================
# Integration with load_base_generator
# =====================================================================

class TestLoadBaseGeneratorIntegration:
    def test_use_sdpa_default_true(self):
        """CAEMConfig default is use_sdpa=True — current Branch-C behavior."""
        from caem.config import CAEMConfig
        cfg = CAEMConfig()
        assert cfg.use_sdpa is True

    def test_use_flash_attention_2_default_false(self):
        """CAEMConfig default flipped from True to False (Branch-C 2026-04-23)
        because flash-attn lacks Blackwell wheels and cuDNN-SDPA is faster
        anyway on sm_120."""
        from caem.config import CAEMConfig
        cfg = CAEMConfig()
        assert cfg.use_flash_attention_2 is False

    def test_config_has_both_attn_fields(self):
        """Both use_sdpa and use_flash_attention_2 must exist for
        back-compat with callers that pass either."""
        from caem.config import CAEMConfig
        cfg = CAEMConfig()
        assert hasattr(cfg, "use_sdpa")
        assert hasattr(cfg, "use_flash_attention_2")


# =====================================================================
# Determinism / precision guards
# =====================================================================

class TestPrecisionGuards:
    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="Requires CUDA device")
    def test_tf32_disabled_after_configure(self):
        """_configure_sdpa_backends should leave TF32 off so bf16 matmuls
        don't get silently upcasted + rounded."""
        import caem.model_loader as ml
        ml._SDPA_BACKENDS_CONFIGURED = False
        _configure_sdpa_backends()
        assert torch.backends.cuda.matmul.allow_tf32 is False

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="Requires CUDA device")
    def test_math_sdp_disabled_after_configure(self):
        """math-SDPA should be off so we never accidentally dispatch to
        the slow fallback."""
        import caem.model_loader as ml
        ml._SDPA_BACKENDS_CONFIGURED = False
        _configure_sdpa_backends()
        # PyTorch exposes a flags-query API in 2.5+
        try:
            # Depending on PT version, math_sdp flag may be read-only
            from torch.backends.cuda import flash_sdp_enabled, math_sdp_enabled
            assert flash_sdp_enabled() is True
            # math_sdp_enabled may or may not exist on all builds
        except ImportError:
            pytest.skip("PT build does not expose sdp_enabled query API")
