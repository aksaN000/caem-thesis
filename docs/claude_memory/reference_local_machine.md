---
name: reference_local_machine
description: "Author's local machine (2026-10-17): RTX 3060 12 GB, 15.2 GB RAM (WSL 12 GB + 16 GB swap on G:), WSL disk on a HDD (D:), SSD = C:/G:; project env ~/caem_env; HF model cache on SSD; live exact search impossible locally."
metadata:
  type: reference
---
- GPU RTX 3060 12 GB (driver CUDA 13.0); Windows display uses ~1.4 GB → ~10.5 GB usable. All four CAEM models = 9.86 GB; local batches ~4.
- RAM 15.2 GB; `C:\Users\aksan\.wslconfig` memory=12GB, swap=16GB at G:\WSL\swap.vhdx (old F:\WSL\swap.vhdx can be deleted after a restart).
- WSL ext4 disk is D:\WSL\Ubuntu on the Toshiba HDD (90–120 MB/s). SSD (TEAM) holds C: (36 GB free) and G: (~65 GB free after models). /mnt/g reads ~245 MB/s.
- Env: `~/caem_env` (uv, py3.12, torch 2.10.0+cu130, transformers 4.57.6, sentence-transformers 4.1.0, faiss 1.15.1). Scratch env `~/caem_reranker_env` (older versions; can be deleted).
- `HF_HUB_CACHE=/mnt/g/caem_hf/hub` in ~/.bashrc (Qwen cold load 27 s vs 535 s from HDD).
- Passage index data/passage_index (61 GB flat + 7 GB pkl) on HDD: batched pass ~10 min; live single exact search NOT feasible locally (index > RAM).
- Use `python -u` + log file for long jobs; system python lacks numpy/pytest.

**2026-10-17 later:** author: local PC too slow, ALL compute moves to Vast.ai. Removed: ~/caem_env, ~/caem_reranker_env, ~/.cache/uv, G:\caem_hf (models), HF_HUB_CACHE line in ~/.bashrc, outputs/rerun_v2/retrieval. Kept: .wslconfig (12 GB RAM, swap file G:\WSL\swap.vhdx grows only when used), phase-0 GPU fit log.
