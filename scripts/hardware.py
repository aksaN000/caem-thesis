"""
scripts/hardware.py
====================
Hardware detection and configuration for CAEM experiments.

Supports local GPU (3060 / any CUDA card), Google Colab (A100/T4),
university clusters (SLURM + CUDA), and CPU fallback.

All CAEM scripts import get_device() from here -- never hardcode "cuda" or "cpu".

Usage
-----
    from scripts.hardware import get_device, get_hardware_info, apply_memory_flags

    device = get_device()                  # "cuda" | "mps" | "cpu"
    info = get_hardware_info()             # dict with VRAM, GPU name, etc.
    apply_memory_flags()                   # enables memory-saving flags if < 12 GB VRAM

Hardware targets
----------------
  RTX 3060 (12 GB VRAM)    -- local dev machine; fp16 recommended, batch_size=8
  A100 (40/80 GB)          -- Colab Pro+; fp16 or bf16, batch_size=16
  T4 (16 GB)               -- Colab free; fp16, batch_size=8
  V100 (16/32 GB)          -- university clusters; fp16, batch_size=8-16
  CPU fallback              -- smoke tests only; batch_size=2
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Describes available hardware and recommended settings for CAEM."""
    device: str                      # "cuda", "mps", or "cpu"
    gpu_name: Optional[str]          # e.g. "NVIDIA GeForce RTX 3060"
    vram_gb: float                   # 0.0 if CPU
    use_fp16: bool                   # half-precision inference
    use_bf16: bool                   # bfloat16 (A100 only)
    recommended_batch_size: int      # for SelfImprovementLoop
    note: str = ""


def get_device() -> str:
    """Return the best available device string.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    Respects CAEM_DEVICE env var if set (e.g. CAEM_DEVICE=cpu for debug).
    """
    env_device = os.environ.get("CAEM_DEVICE", "").strip().lower()
    if env_device in ("cuda", "cpu", "mps"):
        logger.info("Device override from CAEM_DEVICE=%s", env_device)
        return env_device

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_hardware_profile() -> HardwareProfile:
    """Detect hardware and return recommended settings for CAEM.

    This is the primary entry point for experiment scripts that need
    to configure batch size, precision, and memory flags.
    """
    device = get_device()

    if device == "cpu":
        return HardwareProfile(
            device="cpu",
            gpu_name=None,
            vram_gb=0.0,
            use_fp16=False,
            use_bf16=False,
            recommended_batch_size=2,
            note="CPU only -- smoke tests only; full experiments will be very slow.",
        )

    if device == "mps":
        return HardwareProfile(
            device="mps",
            gpu_name="Apple Silicon MPS",
            vram_gb=0.0,     # shared memory -- hard to report
            use_fp16=False,  # fp16 unstable on MPS as of PyTorch 2.x
            use_bf16=False,
            recommended_batch_size=4,
            note="Apple Silicon MPS. fp16 disabled (stability).",
        )

    # CUDA path
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        # Report in GB
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
    except Exception:
        gpu_name = "Unknown CUDA GPU"
        vram_gb = 0.0

    # Choose precision and batch size based on VRAM.
    # Tier breakdown (2026 cloud GPU landscape):
    #   A100 SXM/PCIe 80 GB  (vram_gb >= 70)  → bf16, batch=32
    #   RTX 5090 / A100 40GB (vram_gb >= 30)  → bf16, batch=32
    #   RTX 4090 / 3090 24GB (vram_gb >= 20)  → fp16, batch=16
    #   T4 / V100 16 GB      (vram_gb >= 14)  → fp16, batch=8
    #   RTX 3060 / 3070 12GB (vram_gb >= 10)  → fp16, batch=4
    #   < 10 GB              (laptop)          → fp16, batch=2
    #
    # batch_size affects only speed, never accuracy (confirmed in LAB_PC_SCALING_GUIDE).
    # bf16: native on A100 (Ampere) and RTX 5090 (Blackwell). Larger dynamic range
    # than fp16 — avoids overflow when L2 penalty sums 780M squared diffs. RTX 4090
    # uses fp16 (no bf16 tensor core support on Ada Lovelace).
    # Multi-GPU: CAEM fine-tunes on a single GPU. On a 2x5090 machine, only one GPU
    # is used by run_experiment.py; use CUDA_VISIBLE_DEVICES to pin it and run the
    # ablation study on the second GPU in parallel.
    if vram_gb >= 70:          # A100 SXM 80 GB / PCIe 80 GB (79–80 GB reported)
        use_fp16, use_bf16 = False, True
        batch_size = 32
        note = "A100 80 GB. bf16, batch_size=32. TF32 enabled for matmuls."
    elif vram_gb >= 30:        # RTX 5090 (32 GB), A100 40 GB (39–40 GB reported)
        use_fp16, use_bf16 = False, True
        batch_size = 32
        note = (
            "RTX 5090 / A100 40 GB (30–70 GB range). bf16, batch_size=32. "
            "TF32 enabled for matmuls. theta_prev GPU optimisation active."
        )
    elif vram_gb >= 20:        # RTX 4090 (24 GB), RTX 3090 (24 GB)
        use_fp16, use_bf16 = True, False
        batch_size = 16
        note = (
            "RTX 4090/3090-class (24 GB). fp16, batch_size=16. "
            "theta_prev GPU optimisation active (VRAM >= 24 GB)."
        )
    elif vram_gb >= 14:        # T4 (16 GB), V100 (16 GB), RTX 3080 Ti (12 GB edge)
        use_fp16, use_bf16 = True, False
        batch_size = 8
        note = "16 GB GPU (T4/V100-class). fp16, batch_size=8."
    elif vram_gb >= 10:        # RTX 3060 (12 GB), RTX 3070 (8 GB edge)
        use_fp16, use_bf16 = True, False
        batch_size = 4
        note = (
            f"~12 GB VRAM (RTX 3060-class). "
            "fp16 enabled, batch_size=4. "
            "Flan-T5-Large (~1.5 GB) + NLI (~1.4 GB) + SBERT (~0.5 GB) "
            "= ~3.4 GB model footprint + ~4-6 GB activations -- fits with headroom."
        )
    else:                      # < 10 GB (laptop GPUs, GTX 1080, etc.)
        use_fp16, use_bf16 = True, False
        batch_size = 2
        note = "< 10 GB VRAM. fp16, batch_size=2. Consider --no_nli to save ~1.4 GB."

    return HardwareProfile(
        device=device,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        recommended_batch_size=batch_size,
        note=note,
    )


def get_hardware_info() -> dict:
    """Return hardware info as a plain dict (for JSON serialization in logs)."""
    profile = get_hardware_profile()
    return {
        "device": profile.device,
        "gpu_name": profile.gpu_name,
        "vram_gb": round(profile.vram_gb, 1),
        "use_fp16": profile.use_fp16,
        "use_bf16": profile.use_bf16,
        "recommended_batch_size": profile.recommended_batch_size,
        "note": profile.note,
    }


def apply_memory_flags(profile: Optional[HardwareProfile] = None) -> None:
    """Apply CUDA memory-saving flags based on hardware profile.

    Safe to call unconditionally -- no-ops on CPU/MPS.
    """
    if profile is None:
        profile = get_hardware_profile()
    if profile.device != "cuda":
        return

    try:
        import torch
        # Enable TF32 for A100 (speeds up matmuls with minimal accuracy loss)
        if profile.use_bf16:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for A100-class GPU.")

        # Empty cache to start clean
        torch.cuda.empty_cache()
        logger.info(
            "GPU: %s | VRAM: %.1f GB | fp16=%s | bf16=%s | batch=%d",
            profile.gpu_name,
            profile.vram_gb,
            profile.use_fp16,
            profile.use_bf16,
            profile.recommended_batch_size,
        )
        logger.info("Hardware note: %s", profile.note)
    except Exception as exc:
        logger.warning("apply_memory_flags: %s", exc)


def move_model_to_device(model, profile: Optional[HardwareProfile] = None):
    """Move a model to the correct device with the right precision.

    Applies fp16 or bf16 if the hardware profile recommends it.
    Returns the model on the target device.
    """
    if profile is None:
        profile = get_hardware_profile()

    try:
        import torch
        model = model.to(profile.device)
        if profile.use_bf16:
            model = model.to(torch.bfloat16)
        elif profile.use_fp16:
            model = model.to(torch.float16)
    except Exception as exc:
        logger.warning("move_model_to_device failed (%s); using default precision.", exc)
    return model


def print_hardware_summary() -> HardwareProfile:
    """Print a formatted hardware summary and return the profile."""
    profile = get_hardware_profile()
    print("\n" + "-" * 55)
    print("  CAEM Hardware Profile")
    print("-" * 55)
    print(f"  Device:     {profile.device.upper()}")
    if profile.gpu_name:
        print(f"  GPU:        {profile.gpu_name}")
        print(f"  VRAM:       {profile.vram_gb:.1f} GB")
    print(f"  Precision:  {'bf16' if profile.use_bf16 else 'fp16' if profile.use_fp16 else 'fp32'}")
    print(f"  Batch size: {profile.recommended_batch_size} (recommended)")
    print(f"  Note:       {profile.note}")
    print("-" * 55 + "\n")
    return profile
