#!/usr/bin/env bash
# =============================================================================
# scripts/vast_launch.sh
# =============================================================================
# Self-contained bootstrap + execution of the Phase 2 CAEM baseline rerun on a
# fresh Vast.ai RTX 5090 instance.
#
# Phase 2 plan (full plan in chat history):
#   2.4  Regenerate B7 FLARE outputs (post format-anchor fix) ~90 min
#   2.5  Rescore B6@C5 vanilla_ft through full verifier         ~3.5 h
#   2.6  Rescore B7 FLARE through full verifier                 ~3.5 h
#   2.7  Regenerate cross-system chm_comparison.json            ~10 min
#
# Usage on Vast:
#   $ git clone https://github.com/aksaN000/caem-thesis.git /workspace/caem
#   $ cd /workspace/caem
#   $ git checkout branch/ruc-classifier
#   $ bash scripts/vast_launch.sh bootstrap
#   $ bash scripts/vast_launch.sh run
#
# Or all-in-one:
#   $ bash scripts/vast_launch.sh all
# =============================================================================

set -euo pipefail

CMD="${1:-all}"

WORKSPACE=/workspace/caem
LOG_DIR=${WORKSPACE}/outputs/baselines_vast_rerun_logs
mkdir -p "${LOG_DIR}"

# Composite calibration pin: the locked CAEM C3 composite calibration that
# corresponds to the canonical numbers in the thesis. Do NOT change unless
# the C3 calibration JSON changes upstream.
COMPOSITE_CALIBRATION_PATH=${WORKSPACE}/outputs/full_run/cycle_3/composite_calibration.json

# CUDA + memory hardening (per audit risk-mitigation list).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAEM_FORCE_GPU_CLEANUP=1
export HF_HOME=${WORKSPACE}/hf_cache
export TRANSFORMERS_OFFLINE=0   # need HF downloads on first run


# -------------------------------------------------------------------------- #
# Subcommands                                                                 #
# -------------------------------------------------------------------------- #

cmd_bootstrap() {
    echo "[$(date '+%F %T')] === BOOTSTRAP ==="

    # Pin to CUDA 12.1 + torch 2.4 stack (matches main runs on Vast 5090).
    pip install --upgrade pip
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
    pip install \
        transformers==4.46.3 \
        accelerate==1.0.1 \
        peft==0.13.2 \
        sentence-transformers==3.3.1 \
        faiss-cpu==1.9.0.post1 \
        datasets==3.1.0 \
        huggingface_hub==0.26.2 \
        anthropic==0.39.0 \
        protobuf==4.25.5 \
        tiktoken \
        numpy==1.26.4 \
        scipy==1.13.1

    # Verify torch + cuda match.
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu_ok', torch.cuda.is_available())"

    # FAISS passage index (~64 GB). Resume-safe download from the HF repo.
    if [ ! -f "${WORKSPACE}/data/passage_index/passages.faiss" ]; then
        echo "[$(date '+%F %T')] FAISS index missing; downloading from aksaN000/caem-passage-index-21m ..."
        mkdir -p "${WORKSPACE}/data/passage_index"
        python <<'PYEOF'
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="aksaN000/caem-passage-index-21m",
    repo_type="dataset",
    local_dir="/workspace/caem/data/passage_index",
    local_dir_use_symlinks=False,
    resume_download=True,
)
PYEOF
    else
        echo "[$(date '+%F %T')] FAISS index already present at data/passage_index/"
    fi

    # Pre-fetch Qwen base + MiniCheck so the rescore doesn't pay download cost
    # in the middle of a multi-hour run.
    python <<'PYEOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch.bfloat16)
print("Qwen-2.5-3B-Instruct cached")
PYEOF
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('lytang/MiniCheck-Flan-T5-Large'); print('MiniCheck tokenizer cached')"

    echo "[$(date '+%F %T')] === BOOTSTRAP DONE ==="
}


cmd_upload_b6_eval() {
    # Phase 2 step 2.3: copy the B6@C5 eval JSONs from Phase 1.3 into the
    # canonical eval/ directory the rescore script reads from.
    echo "[$(date '+%F %T')] === Stage B6@C5 eval JSONs (Phase 1.3 outputs) ==="
    SRC=${WORKSPACE}/outputs/baselines_phase1e/vanilla_ft/eval_fixed
    DST=${WORKSPACE}/outputs/baselines/vanilla_ft/eval
    if [ ! -d "${SRC}" ]; then
        echo "ERROR: ${SRC} not found. Did Phase 1.3 finish locally and were the JSONs scp'd?"
        exit 2
    fi
    mkdir -p "${DST}"
    cp -v "${SRC}/"*_cycle5.json "${DST}/"
}


cmd_flare_regen() {
    # Phase 2 step 2.4: B7 FLARE re-generation with the format-anchor fix.
    echo "[$(date '+%F %T')] === B7 FLARE regen (Phase 2.4) ==="
    cd "${WORKSPACE}"
    bash scripts/launch_baselines.sh flare 2>&1 | tee "${LOG_DIR}/flare_regen.log"
}


cmd_rescore_b6() {
    # Phase 2 step 2.5: rescore B6@C5 through the full CAEM verifier.
    echo "[$(date '+%F %T')] === Verifier rescore B6@C5 (Phase 2.5) ==="
    cd "${WORKSPACE}"
    python -m scripts.rescore_baselines_through_verifier \
        --baselines vanilla_ft \
        --cycle 5 \
        --composite_calibration "${COMPOSITE_CALIBRATION_PATH}" \
        --batch_size 32 \
        2>&1 | tee "${LOG_DIR}/rescore_b6_c5.log"
}


cmd_rescore_flare() {
    # Phase 2 step 2.6: rescore B7 FLARE through the full CAEM verifier.
    echo "[$(date '+%F %T')] === Verifier rescore B7 FLARE (Phase 2.6) ==="
    cd "${WORKSPACE}"
    python -m scripts.rescore_baselines_through_verifier \
        --baselines flare \
        --composite_calibration "${COMPOSITE_CALIBRATION_PATH}" \
        --batch_size 32 \
        2>&1 | tee "${LOG_DIR}/rescore_flare.log"
}


cmd_rebuild_chm_comparison() {
    # Phase 2 step 2.7: cross-system chm_comparison.json regen (CPU op).
    echo "[$(date '+%F %T')] === Rebuild chm_comparison.json (Phase 2.7) ==="
    cd "${WORKSPACE}"
    python -m scripts.rescore_baselines_through_verifier \
        --composite_calibration "${COMPOSITE_CALIBRATION_PATH}" \
        2>&1 | tee "${LOG_DIR}/chm_comparison_rebuild.log"
}


cmd_pull_results() {
    # Phase 2 step 2.8: tar up everything we want back on local for Phase 3.
    echo "[$(date '+%F %T')] === Tarball results for local pull ==="
    cd "${WORKSPACE}"
    OUT_TAR=/tmp/rescore_results_$(date '+%Y%m%d_%H%M%S').tgz
    tar czvf "${OUT_TAR}" \
        outputs/baselines/chm_comparison.json \
        outputs/baselines/vanilla_ft/eval/*_with_chm.json \
        outputs/baselines/flare/*_cycle0.json \
        outputs/baselines/flare/*_with_chm.json \
        "${LOG_DIR}/" \
        2>/dev/null || true
    ls -lh "${OUT_TAR}"
    echo "  scp this tarball back to local with: scp vast:${OUT_TAR} /tmp/"
}


cmd_run() {
    cmd_upload_b6_eval
    cmd_flare_regen
    cmd_rescore_b6
    cmd_rescore_flare
    cmd_rebuild_chm_comparison
    cmd_pull_results
}


cmd_all() {
    cmd_bootstrap
    cmd_run
}


# -------------------------------------------------------------------------- #
# Dispatch                                                                    #
# -------------------------------------------------------------------------- #

case "${CMD}" in
    bootstrap)              cmd_bootstrap ;;
    upload_b6_eval)         cmd_upload_b6_eval ;;
    flare_regen)            cmd_flare_regen ;;
    rescore_b6)             cmd_rescore_b6 ;;
    rescore_flare)          cmd_rescore_flare ;;
    rebuild_chm_comparison) cmd_rebuild_chm_comparison ;;
    pull_results)           cmd_pull_results ;;
    run)                    cmd_run ;;
    all)                    cmd_all ;;
    *)
        echo "Usage: $0 {bootstrap|upload_b6_eval|flare_regen|rescore_b6|rescore_flare|rebuild_chm_comparison|pull_results|run|all}"
        exit 1
        ;;
esac
