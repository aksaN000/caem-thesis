#!/usr/bin/env bash
# scripts/recover_oom_baselines.sh
# =================================
# Auto-discovers OOM-damaged baseline eval JSONs and recovers them.
#
# Background (2026-05-16):
#   During the B1-B9 baseline panel, RAG-based baselines suffered batch-32
#   CUDA-OOM on certain benches. The error message: "Tried to allocate
#   2.04 GiB. GPU 0 has a total capacity of 31.36 GiB of which 1.01 GiB
#   is free." Each failure left ~32 samples (one batch) with empty
#   predictions per affected file, which the EM scorer marks as 0 and
#   the LLM judge marks as false — artificially depressing scores.
#
#   Confirmed OOMs at script-creation time:
#     - B3 rag    TruthfulQA: 32 empty samples (04:57 UTC)
#     - B4 cot_rag TruthfulQA: 32 empty samples (06:28 UTC)
#
#   Additional OOMs may land while B6/B7/B8/B9 are still running. This
#   script DISCOVERS damaged files at run time by scanning every
#   outputs/baselines/<name>/<bench>_cycle0.json for empty-prediction
#   counts above a threshold.
#
# Recovery strategy per damaged (baseline, bench) pair:
#   1. Re-run inference at --eval_batch_size 8 (1/4 of original bs=32).
#      At bs=8 the per-batch allocation is ~0.5 GiB vs 2 GiB — well
#      within the ~1 GiB free envelope.
#   2. run_baseline.py overwrites the existing JSON with the new
#      full-300-sample run. Greedy decoding (do_sample=False) makes
#      previously-successful samples produce essentially identical
#      predictions; previously-OOM'd samples now produce real outputs.
#   3. If the affected bench is TruthfulQA, re-run the LLM judge on
#      the recovered JSON (the overwrite cleared em_llm_judged → the
#      script picks up all 300 samples fresh).
#
# Detection threshold:
#   - empty_predictions >= 5 → flagged as OOM victim
#   - "Empty" means literal zero-length prediction text. Legitimate
#     short answers (FEVER labels, CSQA letters, StrategyQA yes/no)
#     all produce non-empty strings, so this signal is clean.
#
# Safety:
#   - Refuses to fire while the original baseline panel is still running
#     (would corrupt files mid-write or compete for GPU).
#   - Refuses to fire if ANTHROPIC_API_KEY isn't set (the rescore step
#     needs it for TruthfulQA recovery).
#   - Original logs preserved at <name>.gen.log; recovery output goes
#     to <name>.recovery.log.
#
# Run after the B1-B9 baseline panel finishes (~22:00 BDT) and before
# the verifier rescore starts.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONPATH="$PWD"

if [[ -f /venv/main/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi

# Same CUDA env as launch_baselines.sh (defense in depth).
export CAEM_FORCE_GPU_CLEANUP="${CAEM_FORCE_GPU_CLEANUP:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Tunables
EMPTY_THRESHOLD="${EMPTY_THRESHOLD:-5}"      # samples with empty prediction to flag
RECOVERY_BATCH_SIZE="${RECOVERY_BATCH_SIZE:-8}"  # bs=8 vs original bs=32

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }
band() { log "==================== $* ===================="; }

# --- Safety: refuse to run if baselines are still in flight ---
if pgrep -f "scripts.run_baseline" >/dev/null; then
    echo "FATAL: scripts.run_baseline is currently running."
    echo "       Wait for the baseline panel to finish before launching recovery."
    pgrep -af "scripts.run_baseline" | sed 's/^/         /'
    exit 1
fi
if pgrep -f "launch_baselines.sh" >/dev/null; then
    echo "FATAL: launch_baselines.sh is currently running. Wait for the panel."
    exit 1
fi

# Note: rescore_truthfulqa.py running in background is fine — we'll re-run it.

# --- Discover damaged files ---
band "Scanning outputs/baselines/ for OOM-damaged JSONs (empty>=$EMPTY_THRESHOLD)"

# Emit lines: "<baseline> <bench> <empty_count> <total>"
DAMAGED=$(python3 - <<PY
import json, glob, os, re

THRESHOLD = $EMPTY_THRESHOLD

# Two failure signatures we detect:
#   1. Truly empty prediction       — pure RAG OOM (returns "")
#   2. Stub == forced prefix only    — CoT-RAG OOM (returns "Reasoning:" or "Reasoning")
#      Generally any prediction shorter than 20 chars where display_answer is
#      identical to prediction (i.e., no answer span was extracted) is an OOM
#      stub. Legitimate short answers (FEVER labels, CSQA letters, StrategyQA
#      yes/no) all parse cleanly so display_answer differs from prediction.
STUB_PREFIXES = ("reasoning:", "reasoning", "answer:")

def is_oom_stub(s):
    pred = (s.get('prediction') or '').strip()
    if not pred:
        return True
    # Stub: very short AND looks like a bare forced-prefix
    if len(pred) <= 20 and pred.lower().strip(":").strip() in {p.strip(":").strip() for p in STUB_PREFIXES}:
        return True
    return False

out = []
for path in sorted(glob.glob('outputs/baselines/*/*_cycle0.json')):
    parts = path.split('/')
    baseline = parts[2]
    fname = parts[3]
    m = re.match(r'(.+)_cycle0\.json$', fname)
    if not m:
        continue
    bench = m.group(1)
    try:
        d = json.load(open(path))
    except Exception:
        continue
    samples = d.get('samples') or []
    n = len(samples)
    if n == 0:
        continue
    n_stub = sum(1 for s in samples if is_oom_stub(s))
    if n_stub >= THRESHOLD:
        out.append(f"{baseline} {bench} {n_stub} {n}")
print('\n'.join(out))
PY
)

if [[ -z "$DAMAGED" ]]; then
    band "No OOM-damaged baseline JSONs found — nothing to recover"
    exit 0
fi

log "Damaged files (baseline | bench | n_empty / n_total):"
echo "$DAMAGED" | while IFS=' ' read -r bl bn ne nt; do
    log "  ${bl} ${bn}: ${ne} / ${nt} empty"
done

# --- Re-run inference for each damaged (baseline, bench) at smaller batch ---
echo "$DAMAGED" | while IFS=' ' read -r bl bn ne nt; do
    band "Recovering B-${bl} ${bn} at --eval_batch_size ${RECOVERY_BATCH_SIZE} (was bs=32)"
    python -m scripts.run_baseline \
        --baseline "$bl" \
        --benchmarks "$bn" \
        --eval_batch_size "$RECOVERY_BATCH_SIZE" \
        --n_questions "$nt" \
        --output_dir outputs/baselines \
        --passage_index data/passage_index \
        --seed 42 \
        2>&1 | tee -a "outputs/baselines/${bl}.recovery.log"
done

# --- LLM-judge rescore only on recovered TruthfulQA files ---
# Other benches don't use LLM-judge (only TruthfulQA was rescored).
TRUTHFULQA_FILES=$(echo "$DAMAGED" | awk '$2 == "truthfulqa" {print "outputs/baselines/" $1 "/truthfulqa_cycle0.json"}')

if [[ -n "$TRUTHFULQA_FILES" ]]; then
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        log "WARNING: ANTHROPIC_API_KEY not set — skipping LLM-judge rescore on recovered TruthfulQA files."
        log "         Set the key and run: python -m scripts.rescore_truthfulqa --files $TRUTHFULQA_FILES"
    else
        band "Re-judging recovered TruthfulQA files with LLM judge"
        # shellcheck disable=SC2086
        python -m scripts.rescore_truthfulqa \
            --files $TRUTHFULQA_FILES \
            --max_concurrent 5 \
            2>&1 | tee -a outputs/baselines/recovery_rescore.log
    fi
fi

# --- Verification ---
band "Post-recovery verification"
echo "$DAMAGED" | while IFS=' ' read -r bl bn ne nt; do
    python3 - <<PY
import json
f = "outputs/baselines/${bl}/${bn}_cycle0.json"
d = json.load(open(f))
samples = d['samples']
n = len(samples)
n_empty = sum(1 for s in samples if not (s.get('prediction') or '').strip())
em = sum(s.get('em',0) for s in samples)/max(n,1)
judged = [s.get('em_llm_judged') for s in samples if s.get('em_llm_judged') is not None]
em_j = sum(judged)/len(judged) if judged else None
status = "OK" if n_empty == 0 else f"STILL DAMAGED (n_empty={n_empty})"
print(f"  ${bl}/${bn}: status={status}  EM={em:.4f}  judged_EM={em_j}")
PY
done

band "OOM recovery complete at $(ts)"
log "Recovered files are now consistent at bs=${RECOVERY_BATCH_SIZE} throughout."
log "Next: re-sync to gdrive (next runner pass auto-fires), then verifier rescore."
