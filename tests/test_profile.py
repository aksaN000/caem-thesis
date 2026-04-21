"""
tests/test_profile.py
=====================
Unit tests for ``caem._profile`` -- the CAEM_PROFILE env var scaffolding.

Coverage:
  - Mode parsing: valid values, empty, garbage, out-of-range
  - section() under mode 0 is a no-op that yields cleanly
  - section() under mode 1 still yields on non-CUDA hosts (graceful
    fallback; the profiler-enabled harness must never crash the
    workload)
  - Nested section() calls work under both modes
  - profile_mode() reflects the imported mode
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from caem import _profile as profile_mod
from caem._profile import _parse_profile_mode, profile_mode, section


# -----------------------------------------------------------------------------
# Mode parsing
# -----------------------------------------------------------------------------

class TestParseProfileMode:
    def test_none_returns_zero(self):
        assert _parse_profile_mode(None) == 0

    def test_empty_returns_zero(self):
        assert _parse_profile_mode("") == 0
        assert _parse_profile_mode("   ") == 0

    def test_valid_modes(self):
        assert _parse_profile_mode("0") == 0
        assert _parse_profile_mode("1") == 1
        assert _parse_profile_mode("2") == 2

    def test_whitespace_stripped(self):
        assert _parse_profile_mode("  1  ") == 1

    def test_garbage_falls_back_to_zero(self):
        assert _parse_profile_mode("abc") == 0
        assert _parse_profile_mode("dev") == 0

    def test_out_of_range_falls_back_to_zero(self):
        assert _parse_profile_mode("3") == 0
        assert _parse_profile_mode("-1") == 0
        assert _parse_profile_mode("99") == 0


# -----------------------------------------------------------------------------
# section() under production mode (0)
# -----------------------------------------------------------------------------

class TestSectionMode0:
    def test_yields_noop(self):
        """section() under mode 0 yields with no side effects; body executes."""
        # Module-level _PROFILE_MODE is set at import time; test by
        # monkeypatching it to 0 for this test regardless of host env.
        with patch.object(profile_mod, "_PROFILE_MODE", 0):
            triggered = []
            with section("work-unit"):
                triggered.append(1)
            assert triggered == [1]

    def test_nested_yields_both_bodies(self):
        with patch.object(profile_mod, "_PROFILE_MODE", 0):
            seen = []
            with section("outer"):
                seen.append("outer")
                with section("inner"):
                    seen.append("inner")
                seen.append("after-inner")
            assert seen == ["outer", "inner", "after-inner"]

    def test_exception_propagates(self):
        with patch.object(profile_mod, "_PROFILE_MODE", 0):
            with pytest.raises(ValueError, match="boom"):
                with section("will-raise"):
                    raise ValueError("boom")


# -----------------------------------------------------------------------------
# section() under mode 1 (NVTX) on a non-CUDA host
# -----------------------------------------------------------------------------

class TestSectionMode1Fallback:
    def test_yields_even_without_cuda(self):
        """On a CPU-only test host, NVTX push fails gracefully and the
        workload still runs. Critical: a profiler-enabled harness must
        never crash the process it's observing.
        """
        with patch.object(profile_mod, "_PROFILE_MODE", 1):
            seen = []
            with section("section-a"):
                seen.append("ran")
            assert seen == ["ran"]

    def test_nested_yields_cleanly_without_cuda(self):
        with patch.object(profile_mod, "_PROFILE_MODE", 1):
            seen = []
            with section("outer"):
                with section("inner"):
                    seen.append("inner-ran")
            assert seen == ["inner-ran"]


# -----------------------------------------------------------------------------
# section() under mode 2 (torch.profiler) on a non-CUDA host
# -----------------------------------------------------------------------------

class TestSectionMode2Fallback:
    def test_yields_even_when_profiler_unavailable(self):
        with patch.object(profile_mod, "_PROFILE_MODE", 2):
            seen = []
            with section("tier3.rag"):
                seen.append("body")
            assert seen == ["body"]


# -----------------------------------------------------------------------------
# profile_mode() + is_profiling() accessors
# -----------------------------------------------------------------------------

class TestAccessors:
    def test_profile_mode_returns_parsed_value(self):
        with patch.object(profile_mod, "_PROFILE_MODE", 1):
            assert profile_mode() == 1
        with patch.object(profile_mod, "_PROFILE_MODE", 0):
            assert profile_mode() == 0

    def test_is_profiling_true_for_nonzero(self):
        with patch.object(profile_mod, "_PROFILE_MODE", 0):
            assert profile_mod.is_profiling() is False
        with patch.object(profile_mod, "_PROFILE_MODE", 1):
            assert profile_mod.is_profiling() is True
        with patch.object(profile_mod, "_PROFILE_MODE", 2):
            assert profile_mod.is_profiling() is True
