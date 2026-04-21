"""
tests/test_gpu_cleanup.py
=========================
Unit tests for the CUDA-process-cleanup helpers in ``caem.model_loader``.

The actual ``nvidia-smi`` invocation + kill(9) call are mocked; we test
the parse, filter, dry-run, and self-exclusion logic. Running these on
a host without CUDA is fine.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from caem import model_loader as ml


# -----------------------------------------------------------------------------
# _list_cuda_processes
# -----------------------------------------------------------------------------

class TestListCudaProcesses:
    def test_parses_csv_output(self):
        fake = b"1234, 512\n5678, 2048\n9012, 16384\n"
        with patch.object(
            ml.subprocess, "check_output", return_value=fake,
        ):
            result = ml._list_cuda_processes()
        assert result == [(1234, 512), (5678, 2048), (9012, 16384)]

    def test_empty_output(self):
        with patch.object(
            ml.subprocess, "check_output", return_value=b"",
        ):
            assert ml._list_cuda_processes() == []

    def test_malformed_lines_skipped(self):
        fake = b"1234, 512\ngarbage line\n5678, not-a-number\n9012, 16384\n"
        with patch.object(
            ml.subprocess, "check_output", return_value=fake,
        ):
            result = ml._list_cuda_processes()
        # Only the two valid rows survive.
        assert result == [(1234, 512), (9012, 16384)]

    def test_subprocess_failure_returns_empty(self):
        """On a non-CUDA host nvidia-smi is missing; must degrade silently
        rather than crash the import graph."""
        with patch.object(
            ml.subprocess, "check_output",
            side_effect=FileNotFoundError("nvidia-smi"),
        ):
            assert ml._list_cuda_processes() == []

    def test_nvidia_smi_timeout_returns_empty(self):
        """Stuck nvidia-smi -> empty, don't hang the caller."""
        from subprocess import TimeoutExpired
        with patch.object(
            ml.subprocess, "check_output",
            side_effect=TimeoutExpired(cmd="nvidia-smi", timeout=5),
        ):
            assert ml._list_cuda_processes() == []


# -----------------------------------------------------------------------------
# _reap_stale_cuda_processes
# -----------------------------------------------------------------------------

class TestReapStaleCudaProcesses:
    def test_self_pid_always_excluded(self):
        """The reaper must NEVER kill its own caller, even if nvidia-smi
        reports the current PID as a CUDA-holder (which is normal when
        the caller has loaded a torch model).
        """
        my_pid = os.getpid()
        fake_procs = [(my_pid, 5000), (9999, 2048)]
        killed: list = []
        with patch.object(ml, "_list_cuda_processes", return_value=fake_procs), \
             patch.object(ml.os, "kill",
                          side_effect=lambda p, s: killed.append((p, s))):
            reaped = ml._reap_stale_cuda_processes()
        # Only the other PID was reaped.
        assert reaped == [(9999, 2048)]
        assert killed == [(9999, 9)]

    def test_dry_run_does_not_kill(self):
        fake_procs = [(1111, 500), (2222, 1500)]
        killed: list = []
        with patch.object(ml, "_list_cuda_processes", return_value=fake_procs), \
             patch.object(ml.os, "kill",
                          side_effect=lambda p, s: killed.append((p, s))):
            reaped = ml._reap_stale_cuda_processes(dry_run=True)
        # Reported the intended targets, but no signals sent.
        assert reaped == [(1111, 500), (2222, 1500)]
        assert killed == []

    def test_min_memory_threshold(self):
        """Processes below the memory floor are ignored."""
        fake_procs = [(1111, 50), (2222, 500), (3333, 50_000)]
        killed: list = []
        with patch.object(ml, "_list_cuda_processes", return_value=fake_procs), \
             patch.object(ml.os, "kill",
                          side_effect=lambda p, s: killed.append((p, s))):
            reaped = ml._reap_stale_cuda_processes(min_memory_mib=1000)
        # Only PID 3333 cleared the 1000 MiB floor.
        assert reaped == [(3333, 50_000)]
        assert killed == [(3333, 9)]

    def test_process_lookup_error_counts_as_reaped(self):
        """nvidia-smi can lag the kernel; if the PID is already gone by
        the time os.kill fires, count it as reaped (net effect: GPU freed).
        """
        fake_procs = [(7777, 1024)]
        with patch.object(ml, "_list_cuda_processes", return_value=fake_procs), \
             patch.object(ml.os, "kill", side_effect=ProcessLookupError):
            reaped = ml._reap_stale_cuda_processes()
        assert reaped == [(7777, 1024)]

    def test_permission_error_does_not_raise(self):
        """Non-root invocations cannot kill other users' processes.
        The function should log + skip, not crash.
        """
        fake_procs = [(4444, 2000)]
        with patch.object(ml, "_list_cuda_processes", return_value=fake_procs), \
             patch.object(ml.os, "kill", side_effect=PermissionError):
            reaped = ml._reap_stale_cuda_processes()
        # PermissionError does NOT count as reaped (memory still held).
        assert reaped == []

    def test_empty_process_list_is_noop(self):
        with patch.object(ml, "_list_cuda_processes", return_value=[]):
            assert ml._reap_stale_cuda_processes() == []


# -----------------------------------------------------------------------------
# _warn_or_reap_gpu_state integration (the auto-called pre-load hook)
# -----------------------------------------------------------------------------

class TestWarnOrReapGPUState:
    def test_clean_gpu_is_noop(self):
        """No CUDA-holders -> no warning, no reap."""
        with patch.object(ml, "_list_cuda_processes", return_value=[]), \
             patch.object(ml.os, "kill") as kill_mock:
            ml._warn_or_reap_gpu_state()
        kill_mock.assert_not_called()

    def test_only_self_is_noop(self):
        """If only the current PID is listed, nothing to do."""
        my_pid = os.getpid()
        with patch.object(
            ml, "_list_cuda_processes", return_value=[(my_pid, 5000)],
        ), patch.object(ml.os, "kill") as kill_mock:
            ml._warn_or_reap_gpu_state()
        kill_mock.assert_not_called()

    def test_other_pid_without_flag_only_warns(self):
        """Default behaviour: WARN but do NOT reap."""
        fake_procs = [(9999, 10_000)]
        killed: list = []
        env = dict(os.environ)
        env.pop("CAEM_FORCE_GPU_CLEANUP", None)
        with patch.dict(os.environ, env, clear=True), \
             patch.object(ml, "_list_cuda_processes", return_value=fake_procs), \
             patch.object(ml.os, "kill",
                          side_effect=lambda p, s: killed.append((p, s))):
            ml._warn_or_reap_gpu_state()
        # No kills despite the stale PID.
        assert killed == []

    def test_other_pid_with_flag_triggers_reap(self):
        """With CAEM_FORCE_GPU_CLEANUP=1 the stale PID gets reaped."""
        fake_procs = [(9999, 10_000)]
        killed: list = []
        with patch.dict(os.environ, {"CAEM_FORCE_GPU_CLEANUP": "1"}), \
             patch.object(ml, "_list_cuda_processes", return_value=fake_procs), \
             patch.object(ml.os, "kill",
                          side_effect=lambda p, s: killed.append((p, s))):
            ml._warn_or_reap_gpu_state()
        assert killed == [(9999, 9)]
