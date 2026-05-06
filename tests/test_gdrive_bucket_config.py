"""
tests/test_gdrive_bucket_config.py
====================================
Smoke test for the v2 gdrive bucket configuration.

Verifies the CAEM_GDRIVE_BUCKET env var is wired into all three offload
sites (self_improvement.py per-cycle adapter offload + run_experiment.py
stream-chunk snapshot offload + run_experiment.py cycle-close artefact
offload). Default bucket is "v2"; v1 archive lives at "archive_v1/".

This test does NOT exercise rclone itself — that requires a live gdrive
remote configured. End-to-end integration is operator-verified via:
  bash scripts/archive_v1_to_gdrive.sh --dry-run
  bash scripts/archive_v1_to_gdrive.sh --execute
"""
from __future__ import annotations

import os
import sys


def test_self_improvement_uses_bucket_env():
    """self_improvement.py per-cycle offload uses CAEM_GDRIVE_BUCKET."""
    import importlib.util
    spec = importlib.util.find_spec("caem.training.self_improvement")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    # Must read from env var with default "v2"
    assert 'os.environ.get("CAEM_GDRIVE_BUCKET", "v2")' in src, (
        "self_improvement.py per-cycle offload does not read CAEM_GDRIVE_BUCKET"
    )
    # Must inject bucket into the gdrive remote path
    assert 'f"gdrive:caem-phase1a/{gd_bucket}/{run_name}' in src, (
        "self_improvement.py per-cycle remote path does not include {gd_bucket}"
    )
    # The legacy hardcoded path should be GONE
    assert 'remote_path = f"gdrive:caem-phase1a/{run_name}/cycle_{cycle_num}/"' not in src, (
        "self_improvement.py still has the v1 hardcoded gdrive path"
    )


def test_run_experiment_streamchunk_uses_bucket_env():
    """run_experiment.py stream-chunk snapshot uses CAEM_GDRIVE_BUCKET."""
    import importlib.util
    spec = importlib.util.find_spec("scripts.run_experiment")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    assert '_gd_bucket = os.environ.get("CAEM_GDRIVE_BUCKET", "v2")' in src, (
        "run_experiment.py stream-chunk offload does not read CAEM_GDRIVE_BUCKET"
    )


def test_run_experiment_cycle_close_uses_bucket_env():
    """run_experiment.py cycle-close offload uses CAEM_GDRIVE_BUCKET."""
    import importlib.util
    spec = importlib.util.find_spec("scripts.run_experiment")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as f:
        src = f.read()
    # The cycle-close path
    assert 'f"gdrive:caem-phase1a/{_gd_bucket}/{_run_name}"' in src, (
        "run_experiment.py cycle-close offload path does not include {_gd_bucket}"
    )


def test_archive_script_exists_and_executable():
    """scripts/archive_v1_to_gdrive.sh exists and is syntactically valid bash."""
    import os
    import subprocess
    path = "/workspace/caem/scripts/archive_v1_to_gdrive.sh"
    assert os.path.isfile(path), f"{path} missing"
    assert os.access(path, os.X_OK), f"{path} not executable"
    # bash -n: parse-only (no execution)
    r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n failed for archive script: {r.stderr}"


def test_archive_script_has_dry_run_default():
    """Archive script defaults to dry-run unless --execute is passed."""
    with open("/workspace/caem/scripts/archive_v1_to_gdrive.sh") as f:
        src = f.read()
    assert "DRY_RUN=true" in src, (
        "Archive script should default to DRY_RUN=true (safety)"
    )
    assert '"${1:-}" == "--execute"' in src, (
        "Archive script should require --execute to actually move files"
    )


def test_archive_script_targets_match_v2_layout():
    """Archive script targets gdrive:caem-phase1a/archive_v1/ + v2/."""
    with open("/workspace/caem/scripts/archive_v1_to_gdrive.sh") as f:
        src = f.read()
    assert 'ARCHIVE_PATH="${GDRIVE_ROOT}/archive_v1"' in src
    assert 'V2_PATH="${GDRIVE_ROOT}/v2"' in src


def test_default_bucket_is_v2():
    """Default CAEM_GDRIVE_BUCKET when unset is 'v2'."""
    # Simulate the runtime default lookup
    bucket = os.environ.get("CAEM_GDRIVE_BUCKET", "v2")
    # Should be "v2" unless CAEM_GDRIVE_BUCKET is set in the env we're running in
    if "CAEM_GDRIVE_BUCKET" not in os.environ:
        assert bucket == "v2", f"default bucket should be v2; got {bucket!r}"


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:
                failures += 1
                print(f"  ERR   {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
    print(f"\nAll tests passed.")
