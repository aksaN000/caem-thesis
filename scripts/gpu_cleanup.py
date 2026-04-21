"""
scripts/gpu_cleanup.py
======================
Manual CUDA-process reaper. Thin CLI wrapper around
``caem.model_loader._reap_stale_cuda_processes``.

Use when a previous CAEM run crashed or was killed mid-load and its
Python child is still holding GPU memory -- a classic Vast
session-recovery symptom. ``load_base_generator`` runs this check
automatically at the top of every pipeline construction (warn always,
reap under ``CAEM_FORCE_GPU_CLEANUP=1``), but sometimes you want to
clean up between runs without launching a pipeline.

Typical use
-----------
    # Before a Vast run, dry-run to see what's holding GPU memory:
    PYTHONPATH=. python scripts/gpu_cleanup.py --dry_run

    # Actually reap (sends SIGKILL to every CUDA-holding PID other
    # than this script's own PID):
    PYTHONPATH=. python scripts/gpu_cleanup.py

    # Only reap processes holding at least 1 GiB (avoids jupyter
    # kernels that have the CUDA runtime loaded but no real tensors):
    PYTHONPATH=. python scripts/gpu_cleanup.py --min_memory_mib 1024

Safety
------
The current process's PID is always excluded from reaping. This script
never kills its own caller. Non-root invocations cannot kill processes
owned by other users and will log a permission warning.

Always safe on shared CUDA hosts when run in --dry_run mode.
"""

from __future__ import annotations

import argparse
import logging
import sys

from caem.model_loader import (
    _list_cuda_processes,
    _reap_stale_cuda_processes,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reap stale CUDA-holding processes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="List what WOULD be killed but do not send signals.",
    )
    p.add_argument(
        "--min_memory_mib", type=int, default=100,
        help="Only reap processes holding at least this much GPU memory.",
    )
    return p


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ns = _build_parser().parse_args(argv)

    all_procs = _list_cuda_processes()
    if not all_procs:
        logger.info("No CUDA-holding processes on this host. GPU is clean.")
        return 0

    logger.info(
        "Detected %d CUDA-holding process(es) before reap:", len(all_procs),
    )
    for pid, mib in all_procs:
        logger.info("  PID %d -- %d MiB", pid, mib)

    reaped = _reap_stale_cuda_processes(
        min_memory_mib=ns.min_memory_mib, dry_run=ns.dry_run,
    )
    if not reaped:
        logger.info(
            "No PIDs exceeded min_memory_mib=%d. Nothing to reap.",
            ns.min_memory_mib,
        )
        return 0

    verb = "would reap" if ns.dry_run else "reaped"
    total = sum(m for _, m in reaped)
    logger.info(
        "%s %d process(es) (%d MiB total).", verb.capitalize(),
        len(reaped), total,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
