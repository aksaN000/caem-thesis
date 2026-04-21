"""
caem/_profile.py
=================
Profiling scaffolding for Branch C Goal 5 (hardware utilization).

Tri-mode switch driven by the ``CAEM_PROFILE`` environment variable:

    CAEM_PROFILE=0  (default, production)
        section() is a null context manager; zero overhead; the import
        graph never touches NVTX or torch.profiler. Byte-identical to
        pre-Goal-5 behaviour.

    CAEM_PROFILE=1  (dev, smoke tests)
        section() emits NVTX range markers
        (torch.cuda.nvtx.range_push / range_pop). Nsight Systems picks
        these up with ~1% overhead. Safe for integration smokes.

    CAEM_PROFILE=2  (diagnostic, hotspot hunting)
        section() additionally records torch.profiler.record_function
        events. ~10% overhead; do NOT run this during main experiments.

Usage
-----
    from caem._profile import section

    with section("tier2.generate"):
        output = model.generate(...)

The call site is mode-agnostic -- the context manager does the right
thing based on the env var at import time. Callers never branch on the
mode themselves.

Mode is read once at import time. Changing ``CAEM_PROFILE`` after import
is a no-op; restart the Python process to switch modes.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


_VALID_MODES = (0, 1, 2)


def _parse_profile_mode(raw: Optional[str]) -> int:
    """Parse ``CAEM_PROFILE`` into a bounded int. Unknown values -> 0."""
    if raw is None:
        return 0
    raw = raw.strip()
    if not raw:
        return 0
    try:
        mode = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "CAEM_PROFILE=%r not an integer; defaulting to 0 (production).",
            raw,
        )
        return 0
    if mode not in _VALID_MODES:
        logger.warning(
            "CAEM_PROFILE=%d outside {0, 1, 2}; defaulting to 0.", mode,
        )
        return 0
    return mode


_PROFILE_MODE: int = _parse_profile_mode(os.environ.get("CAEM_PROFILE"))


def profile_mode() -> int:
    """Return the active profile mode as parsed at import time."""
    return _PROFILE_MODE


def is_profiling() -> bool:
    """True when CAEM_PROFILE >= 1."""
    return _PROFILE_MODE >= 1


@contextlib.contextmanager
def section(name: str) -> Iterator[None]:
    """Context manager marking a named section for profilers.

    Mode 0: no-op (zero runtime cost; the ``with`` block still executes).
    Mode 1: NVTX range push/pop so Nsight Systems traces see ``name``.
    Mode 2: NVTX + torch.profiler.record_function so a torch.profiler
            trace also sees the scope.

    Nested ``section`` calls stack correctly under mode 1/2 (NVTX ranges
    are LIFO) and fall to plain nesting under mode 0.

    The ``name`` is forwarded verbatim to the profiler; keep it short
    and hierarchical (``"tier2.generate"``, ``"verifier.m_chain"``,
    ``"retrieval.faiss_search"``) so flame-graphs group meaningfully.
    """
    if _PROFILE_MODE == 0:
        yield
        return

    # Mode 1 + 2: NVTX range markers. Import torch lazily so a
    # profile-disabled run does not pay the torch-import cost at all
    # (matters when CAEM_PROFILE=0 is used to bypass the profiling
    # machinery entirely in very-short-lived helper scripts).
    try:
        import torch
        torch.cuda.nvtx.range_push(name)
        nvtx_pushed = True
    except Exception as exc:
        # Non-CUDA host or torch missing -> fall back to mode-0 semantics
        # rather than crashing the workload we're trying to observe.
        logger.debug("CAEM_PROFILE %d: NVTX push failed (%s).", _PROFILE_MODE, exc)
        nvtx_pushed = False

    if _PROFILE_MODE >= 2:
        try:
            import torch
            with torch.profiler.record_function(name):
                try:
                    yield
                finally:
                    if nvtx_pushed:
                        torch.cuda.nvtx.range_pop()
            return
        except Exception as exc:
            logger.debug(
                "CAEM_PROFILE 2: torch.profiler.record_function failed (%s); "
                "falling back to NVTX-only.", exc,
            )

    try:
        yield
    finally:
        if nvtx_pushed:
            try:
                import torch
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass


__all__ = [
    "profile_mode",
    "is_profiling",
    "section",
]
