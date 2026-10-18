#!/usr/bin/env bash
# Final-version rental runbook (2026-10-17, MASTER_ISSUES M-A). Run from the repository root on the rented GPU:
#   bash scripts/rental/final_version.sh <stage>
# Stages, in order (each is resumable; finished outputs are skipped):
#   preflight  GPU / RAM / disk / credentials / uploaded files
#   env        Python 3.12 environment (.venv) with the pinned versions
#   data       passage index from Hugging Face, models, B6 weights
#   search     exact passage search for the 1,500 evaluation questions + check against run 1 (gate)
#   baselines  B1, B2, B5, B8 on all five benchmarks; B3, B4 on TruthfulQA (matched prompts)
#   flare      FLARE on all five benchmarks (later searches live, exact)
#   b6         B6 evaluation from its saved cycle-5 weights (matched prompt)
#   gate       evaluation questions identical to CAEM's, no empty answers
#   verify     verifier pass on every re-run baseline (C3 adapter + C3 composite, as in run 1)
#   pack       tar the outputs for download
#   all        everything above in order
# Uploaded beforehand from the local machine (scripts/rental/make_bundle.sh): the repository, outputs/full_run/eval,
# outputs/full_run/cycle_3/{adapter,composite_calibration.json}, outputs/full_run/dataset_splits.json.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Off: with it on, every model load force-kills any other process holding GPU memory (caem/model_loader.py), which
# killed parallel stages on 2026-10-18. Stale processes are cleared by hand instead.
export CAEM_FORCE_GPU_CLEANUP=0
# Pooled token-score/dropout forwards, as in CAEM's main run (outputs/full_run/run.log: pool(u_tok_drop) non-zero in
# all 8,204 batches). Run 1's baseline verifier pass left this unset and used the per-sample path.
export CAEM_BATCH_U_TOK_DROP=1
OUT=outputs/rerun_v2/baselines
LOG=outputs/rerun_v2/logs
INDEX=data/passage_index
B6_CKPT=outputs/baselines_phase1e/vanilla_ft/cycle_5
mkdir -p "$OUT" "$LOG"
ts() { TZ=Asia/Dhaka date '+%Y-%m-%d %H:%M BDT'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG/runbook.log"; }
die() { log "STOP: $*"; exit 1; }
py() { .venv/bin/python -u "$@"; }

preflight() {
    log "== preflight"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG/runbook.log"
    local ram_gb disk_gb
    ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    disk_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
    log "RAM ${ram_gb} GB, free disk ${disk_gb} GB"
    [ "$ram_gb" -ge 88 ] || die "need >= 96 GB RAM (61 GB index + 7 GB passages + models)"
    # The 68 GB index may live on a RAM disk (data/passage_index -> /dev/shm/passage_index, set up before the
    # data stage on machines with a small disk); then the disk only needs the models, env, weights and outputs.
    if [ -L "$INDEX" ] && [[ "$(readlink -f "$INDEX")" == /dev/shm/* ]]; then
        local shm_gb; shm_gb=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
        log "index on RAM disk $(readlink -f "$INDEX"), /dev/shm free ${shm_gb} GB"
        [ "$shm_gb" -ge 75 ] || [ -f "$INDEX/passages.pkl" ] || die "need >= 75 GB free in /dev/shm for the index"
        [ "$disk_gb" -ge 45 ] || die "need >= 45 GB free disk (index on RAM disk)"
    else
        [ "$disk_gb" -ge 120 ] || die "need >= 120 GB free disk"
    fi
    [ -n "${HF_TOKEN:-}" ] || die "export HF_TOKEN (private index repo)"
    for f in outputs/full_run/eval/fever_cycle0.json outputs/full_run/cycle_3/composite_calibration.json \
             outputs/full_run/cycle_3/adapter/adapter_config.json outputs/full_run/dataset_splits.json; do
        [ -f "$f" ] || die "missing uploaded file $f"
    done
    { cat COMMIT 2>/dev/null || git log --oneline -1; } | tee -a "$LOG/runbook.log"
}

env_setup() {
    log "== env"
    [ -x .venv/bin/python ] || uv venv -q .venv --python 3.12
    VIRTUAL_ENV=.venv uv pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
    VIRTUAL_ENV=.venv uv pip install -q -r requirements.txt
    py -c "import torch, transformers, faiss, sentence_transformers as st; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, faiss.__version__, st.__version__)" | tee -a "$LOG/runbook.log"
}

data() {
    log "== data"
    if [ ! -f "$INDEX/passages.faiss" ] || [ ! -f "$INDEX/passages.pkl" ]; then
        mkdir -p "$INDEX"
        .venv/bin/hf download aksaN000/caem-passage-index-21m passages.faiss passages.pkl \
            --repo-type dataset --local-dir "$INDEX"
    fi
    ls -l "$INDEX" | tee -a "$LOG/runbook.log"
    py - <<'EOF'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen2.5-3B-Instruct", "lytang/MiniCheck-Flan-T5-Large", "BAAI/bge-reranker-v2-m3",
          "sentence-transformers/all-mpnet-base-v2"):
    print(m, snapshot_download(m))
EOF
    if [ ! -f "$B6_CKPT/model-00002-of-00002.safetensors" ]; then
        if rclone listremotes 2>/dev/null | grep -q '^gdrive:'; then
            rclone copy gdrive:CAEM_thesis/2_results_phase1e/phase1e_baselines/vanilla_ft/cycle_5 "$B6_CKPT" -P
        else
            die "B6 weights missing: upload $B6_CKPT (5.8 GB) or configure rclone 'gdrive:'"
        fi
    fi
}

search() {
    log "== search"
    local R=outputs/rerun_v2/retrieval
    [ -f "$R/eval_fold_ids.npy" ] || py scripts/precompute_passages.py search --name eval_fold --k 100 --engine faiss 2>&1 | tee -a "$LOG/search.log"
    [ -f "$R/eval_fold_passages.json" ] || py scripts/precompute_passages.py texts --name eval_fold 2>&1 | tee -a "$LOG/search.log"
    [ -f "$R/eval_fold_check.json" ] || py scripts/precompute_passages.py check --name eval_fold 2>&1 | tee -a "$LOG/search.log"
    py - <<'EOF' || die "search check failed: inspect outputs/rerun_v2/retrieval/eval_fold_check.json"
import json, sys
r = json.load(open("outputs/rerun_v2/retrieval/eval_fold_check.json"))
live_ok = r["live_vs_cache_identical_top20"][0] >= 0.96 * r["live_vs_cache_identical_top20"][1]  # batch vs single-query float ties
g = r["run1_signal_agreement"]["p_ground_max"]["within_0.01"]
print({"live_vs_cache": r["live_vs_cache_identical_top20"], "p_ground_max_within_0.01": g})
sys.exit(0 if live_ok and g >= 0.90 else 1)
EOF
}

run_bl() {  # name, benchmarks...
    local name="$1"; shift
    local missing=0
    for b in "$@"; do [ -f "$OUT/$name/${b}_cycle0.json" ] || missing=1; done
    [ "$missing" = 1 ] || { log "$name: done, skipped"; return 0; }
    log "-- $name on $*"
    py -m scripts.run_baseline --baseline "$name" --output_dir "$OUT" --benchmarks "$@" \
        --n_questions 300 --eval_batch_size 32 --eval_prefetch --passage_index "$INDEX" --seed 42 \
        2>&1 | tee -a "$LOG/$name.gen.log"
}

ALL5=(fever triviaqa commonsense_qa truthfulqa strategyqa)
baselines() {
    log "== baselines"
    run_bl zero_shot "${ALL5[@]}"
    run_bl cot "${ALL5[@]}"
    run_bl fiveshot_cot "${ALL5[@]}"
    run_bl rag truthfulqa
    run_bl cot_rag truthfulqa
    run_bl semantic_entropy "${ALL5[@]}"
}

flare() { log "== flare"; run_bl flare "${ALL5[@]}"; }

b6() {
    log "== b6"
    if ls "$OUT/vanilla_ft/eval/"*_cycle5.json >/dev/null 2>&1 && [ "$(ls "$OUT/vanilla_ft/eval/"*_cycle5.json | wc -l)" -ge 5 ]; then
        log "b6: done, skipped"; return 0
    fi
    CAEM_B6_OUTPUT_DIR="$OUT/vanilla_ft/eval" py scripts/eval_b6_c5_only.py 2>&1 | tee -a "$LOG/b6.eval.log"
}

gate() { log "== gate"; py scripts/rental/check_eval_questions.py "$OUT" 2>&1 | tee -a "$LOG/gate.log"; }

verify() {
    log "== verify"
    py -m scripts.rescore_baselines_through_verifier \
        --composite_calibration outputs/full_run/cycle_3/composite_calibration.json \
        --checkpoint outputs/full_run/cycle_3/adapter \
        --passage_index "$INDEX" \
        --baselines zero_shot cot fiveshot_cot rag cot_rag flare semantic_entropy vanilla_ft \
        --training_baselines vanilla_ft --training_cycles 5 \
        --benchmarks "${ALL5[@]}" \
        --baselines_dir "$OUT" --output_dir "$OUT" \
        2>&1 | tee -a "$LOG/verify.log"
}

pack() {
    log "== pack"
    tar czf outputs/rerun_v2_results.tgz outputs/rerun_v2/baselines outputs/rerun_v2/retrieval/*.json \
        outputs/rerun_v2/retrieval/*.npy outputs/rerun_v2/logs
    ls -lh outputs/rerun_v2_results.tgz | tee -a "$LOG/runbook.log"
}

case "${1:-}" in
    preflight) preflight ;;
    env) env_setup ;;
    data) data ;;
    search) search ;;
    baselines) baselines ;;
    flare) flare ;;
    b6) b6 ;;
    gate) gate ;;
    verify) verify ;;
    pack) pack ;;
    all) preflight; env_setup; data; search; baselines; flare; b6; gate; verify; pack; log "ALL STAGES DONE" ;;
    *) sed -n 2,17p "$0"; exit 2 ;;
esac
