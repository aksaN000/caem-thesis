# CAEM Phase 2 Vast Runbook — Agent Instructions

**You are an autonomous agent running on a Vast.ai RTX 4090 instance. This file is your full set of instructions for completing CAEM's Phase 2 baseline rerun.**

**Read this file end-to-end before doing anything.** Then execute Phase 2a fully, evaluate the result + remaining budget at the decision point, and either continue into Phase 2b or stop and report.

Time zone reference for all log messages: **BDT (UTC+6)**.

---

## 0. Context — what you are doing and why

The CAEM thesis (`https://github.com/aksaN000/caem-thesis`, branch `branch/ruc-classifier`) has an 8-baseline panel (B1–B8) that compares the CAEM architecture against external baselines on five QA benchmarks. Two of the eight baselines have broken results due to a format-anchor bug in their evaluation code:

- **B6 (vanilla fine-tune at cycle 5)** — eval shim's hard-coded `max_new_tokens=256` truncated the `Answer:` line on short-form benchmarks. CommonsenseQA EM = 0.000 in the broken results.
- **B7 (FLARE)** — `FLAREBaseline` class never inherits `SYSTEM_PROMPT` + `FORCED_PREFIX`, and the iterative loop strips the `Reasoning:` prefix before the `Answer:` line lands. TruthfulQA EM = 0.000 in the broken results.

**The matched-protocol fix has been applied at the code level** (commits `9b60b1c`, `e6ecdae`, `d0f681f`, `86a0e54`). The B6@C5 EM has been re-evaluated locally on the user's machine — those new eval JSONs need to be **uploaded to this Vast instance before you start** (see Section 1 prerequisites).

Your job here is to:

1. **Bootstrap** the Vast environment (deps + 64 GB FAISS index from HF + Qwen + MiniCheck caches)
2. **Re-generate B7 FLARE outputs** with the format-anchor fix applied
3. **Verifier-rescore B6@C5** through the locked CAEM verifier (to refresh CHM signals)
4. **Verifier-rescore B7 FLARE** if budget allows (Phase 2b — conditional)
5. **Rebuild the cross-system `chm_comparison.json`** so downstream aggregators consume the new signals
6. **Tarball results + report** so the user can pull them back to local

---

## 1. Pre-flight prerequisites

**Before invoking you, the user should have done these on their local machine:**

- ☑ Locally evaluated B6@C5 with the matched-protocol fix → `outputs/baselines_phase1e/vanilla_ft/eval_fixed/{bench}_cycle5.json` (5 files)
- ☑ scp'd those 5 JSONs to this Vast instance at `/workspace/caem/outputs/baselines/vanilla_ft/eval/`
- ☑ Exported `ANTHROPIC_API_KEY` in the shell that will launch you (for the TQA Haiku rescore step)
- ☑ Configured the GitHub repo SSH access (if pushing) OR the agent will use the user's PAT via `https://aksaN000:<PAT>@github.com/...` already in `.git/config` after clone

**Verify these prerequisites before doing any work.** If any are missing, **stop and ask the user** — do not improvise.

```bash
# Verification check — run this FIRST
ls /workspace/caem 2>/dev/null && echo "Repo present" || echo "REPO MISSING — clone first"
ls /workspace/caem/outputs/baselines/vanilla_ft/eval/*_cycle5.json 2>/dev/null | wc -l
[ -n "$ANTHROPIC_API_KEY" ] && echo "API key set (length=${#ANTHROPIC_API_KEY})" || echo "API KEY MISSING — needed for B7 TQA Haiku rescore"
```

Expected output:
- Repo present
- 5 cycle5 JSONs (or 0 if they need to be uploaded — wait for upload)
- API key set, length ~108

---

## 2. Budget envelope and decision rules

The user has **$5.43 in Vast credit** at start. Your hourly burn rate is **$0.429/h** (compute + disk) plus internet usage (~$0.22 one-time for FAISS + model downloads).

### Phase 2a budget — Partial Path B (must-have)

| Step | Wall time | Cost |
|------|-----------|------|
| Bootstrap (FAISS pull + deps + model caches) | ~30 min | ~$0.21 + $0.22 internet |
| B7 FLARE re-generate | ~2.5 h | ~$1.07 |
| B6@C5 verifier rescore | ~5.0 h | ~$2.15 |
| chm_comparison.json rebuild | ~10 min | ~$0.07 |
| Pull results, stop instance | ~5 min | ~$0.04 |
| **Phase 2a total** | **~8.2 h** | **~$3.76** |

**Phase 2a must complete.** If you hit a blocker, retry once. If still failing, save partial state to `/tmp/phase2a_partial_state.txt` and report to the user via the final summary.

### Phase 2b budget — B7 verifier rescore (conditional)

After Phase 2a finishes, check remaining Vast credit. **Run Phase 2b ONLY if remaining credit ≥ $2.20**. The B7 verifier rescore is ~5 hours × $0.429/h = ~$2.15. If you have less than $2.20, stop after Phase 2a with the partial-rescore footnote (Phase 2.5 handles this).

```bash
# Check remaining credit via the Vast CLI (if installed) or just compute it:
# REMAINING = STARTING_CREDIT - sum of (hourly_rate × hours_elapsed_so_far)
# If REMAINING < 2.20, skip Phase 2b. Otherwise, proceed.
```

If Phase 2b is skipped, this is acceptable — the user has pre-agreed to footnote the B7 CHM as "directional only, pre-fix predictions" in Ch5. Your job is just to be honest about why you stopped.

---

## 3. Phase 2a — execute these steps in order

All these are wrappers around `bash scripts/vast_launch.sh`. The script is on the repo and self-contained. Each step writes outputs incrementally so a pre-emption mid-step only loses the in-progress benchmark.

### Step 1 — clone repo + bootstrap

```bash
cd /workspace
git clone https://github.com/aksaN000/caem-thesis.git caem 2>/dev/null || true
cd caem
git checkout branch/ruc-classifier
git pull origin branch/ruc-classifier
bash scripts/vast_launch.sh bootstrap 2>&1 | tee /tmp/bootstrap.log
```

**Verify:** at the end of bootstrap, run:

```bash
python -c "
import torch, faiss
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())
idx = faiss.read_index('/workspace/caem/data/passage_index/passages.faiss')
print('FAISS index ntotal', idx.ntotal)
"
ls -lh /workspace/caem/data/passage_index/passages.faiss
```

Expected: torch 2.4.x, CUDA 12.1, gpu True, FAISS index ntotal ~21000000, passages.faiss ~64 GB.

If FAISS is not present or `ntotal == 0`, **stop and report** — the rest of Phase 2 cannot run.

### Step 2 — verify B6@C5 eval JSONs are present

```bash
ls /workspace/caem/outputs/baselines/vanilla_ft/eval/*_cycle5.json | wc -l
```

Expected: **5** (one per benchmark). If less than 5, **stop and ask the user to scp them**. The verifier rescore in Step 4 cannot proceed without these.

### Step 3 — B7 FLARE re-generate

```bash
bash scripts/vast_launch.sh flare_regen 2>&1 | tee /tmp/flare_regen.log
```

This runs `scripts/launch_baselines.sh flare` which invokes the patched `FLAREBaseline` class. Expected wall time: ~2.5 h. Expected output: 5 new JSONs at `outputs/baselines/flare/{bench}_cycle0.json`.

**Verify:**

```bash
ls /workspace/caem/outputs/baselines/flare/*_cycle0.json | wc -l   # should be 5
python -c "
import json
for b in ['fever','triviaqa','commonsense_qa','truthfulqa','strategyqa']:
    d = json.load(open(f'/workspace/caem/outputs/baselines/flare/{b}_cycle0.json'))
    em = d['meta']['em']
    # Sanity check: post-fix B7 should have EM > 0.2 on at least 3 benches
    print(f'{b}: EM={em:.4f}, n={d[\"meta\"][\"n\"]}')
"
```

Expected: each bench EM > 0.0, n=300. The TriviaQA EM should now be > 0.0 (was 0.0 in the broken pre-fix run).

### Step 4 — B6@C5 verifier rescore

```bash
bash scripts/vast_launch.sh rescore_b6 2>&1 | tee /tmp/rescore_b6.log
```

This invokes `scripts/rescore_baselines_through_verifier.py` scoped to `--baselines vanilla_ft --cycle 5` with the locked CAEM C3 composite calibration. Expected wall time: ~5 h. Expected output: 5 new sidecars at `outputs/baselines/vanilla_ft/eval/{bench}_cycle5_with_chm.json`.

**Verify:**

```bash
ls /workspace/caem/outputs/baselines/vanilla_ft/eval/*_cycle5_with_chm.json | wc -l   # should be 5
python -c "
import json
for b in ['fever','triviaqa','commonsense_qa','truthfulqa','strategyqa']:
    d = json.load(open(f'/workspace/caem/outputs/baselines/vanilla_ft/eval/{b}_cycle5_with_chm.json'))
    chm = d['meta'].get('chm', float('nan'))
    print(f'{b}: CHM={chm:.4f}')
"
```

Expected: each bench CHM in [0.05, 0.25]. None should be NaN.

### Step 5 — rebuild chm_comparison.json

```bash
bash scripts/vast_launch.sh rebuild_chm_comparison 2>&1 | tee /tmp/rebuild_chm.log
```

This re-aggregates the per-baseline `_with_chm.json` sidecars (including the new B6 ones and, if Phase 2b ran, the new B7 ones) into `outputs/baselines/chm_comparison.json`.

**Verify:**

```bash
python -c "
import json
d = json.load(open('/workspace/caem/outputs/baselines/chm_comparison.json'))
for k in ['zero_shot', 'vanilla_ft@c5', 'flare']:
    if k in d:
        chms = [v.get('chm', float('nan')) for v in d[k].values() if isinstance(v, dict)]
        print(f'{k}: per-bench CHMs = {[f\"{c:.3f}\" for c in chms]}')
"
```

Expected: `vanilla_ft@c5` should have updated (non-NaN) CHM values across all 5 benches.

---

## 4. Decision point (Phase 2.5) — should you do Phase 2b?

After Phase 2a completes, **stop and evaluate**:

1. Check Vast remaining credit. Run:
   ```bash
   # No clean Vast CLI for this — compute manually:
   # remaining = starting_credit - (current_time - start_time) * $0.429/h - $0.22 internet
   # Estimate it from /tmp/bootstrap.log timestamps and current time.
   date
   stat /tmp/bootstrap.log | head -3   # use 'Modify' or 'Birth' field for start time
   ```

2. Compute spend so far. Compare to $2.20 threshold.

3. **If remaining credit ≥ $2.20**: proceed to Phase 2b (Section 5).

4. **If remaining credit < $2.20**: skip Phase 2b. Document the decision in the final report and proceed to Section 6 (pull results + stop).

---

## 5. Phase 2b — B7 FLARE verifier rescore (conditional)

```bash
bash scripts/vast_launch.sh rescore_flare 2>&1 | tee /tmp/rescore_flare.log
```

Expected wall time: ~5 h. Expected output: 5 new sidecars at `outputs/baselines/flare/{bench}_cycle0_with_chm.json`.

After this completes, re-run Step 5 (rebuild chm_comparison) to integrate the new FLARE CHM values:

```bash
bash scripts/vast_launch.sh rebuild_chm_comparison 2>&1 | tee /tmp/rebuild_chm_phase2b.log
```

Verify FLARE CHM values are now populated:

```bash
python -c "
import json
d = json.load(open('/workspace/caem/outputs/baselines/chm_comparison.json'))
if 'flare' in d:
    chms = [v.get('chm') for v in d['flare'].values() if isinstance(v, dict)]
    print('FLARE per-bench CHMs:', [f'{c:.3f}' if c is not None else 'NaN' for c in chms])
"
```

---

## 6. Final step — TruthfulQA Haiku rescore for new B7 outputs

The new B7 FLARE TruthfulQA predictions need the canonical Haiku judgment (per memory entry `feedback_truthfulqa_llm_judge_em.md`). This is API-only, no GPU, ~10 min wall time, ~$0.20 in Haiku credits.

```bash
[ -z "$ANTHROPIC_API_KEY" ] && echo "ERROR: ANTHROPIC_API_KEY not set" && exit 1

ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" python -m scripts.rescore_truthfulqa \
    --files /workspace/caem/outputs/baselines/flare/truthfulqa_cycle0.json \
    2>&1 | tee /tmp/flare_tqa_haiku.log
```

Verify the `em_llm_judged` field landed on all 300 samples:

```bash
python -c "
import json
d = json.load(open('/workspace/caem/outputs/baselines/flare/truthfulqa_cycle0.json'))
samples = d['samples']
judged = [s.get('em_llm_judged') for s in samples if s.get('em_llm_judged') is not None]
print(f'Judged: {len(judged)}/{len(samples)}')
print(f'em_llm_judged mean: {sum(judged)/len(judged):.4f}')
"
```

Expected: 300/300 judged, em_llm_judged in [0.1, 0.5] range.

**Do this step regardless of whether Phase 2b ran.** It's cheap and the user wants the canonical TQA number for B7 either way.

---

## 7. Final step — tarball results, push to repo, stop instance

### 7a — Pull results into a tarball

```bash
cd /workspace/caem
bash scripts/vast_launch.sh pull_results 2>&1 | tee /tmp/pull_results.log
ls -lh /tmp/rescore_results_*.tgz
```

Verify the tarball includes:
- `outputs/baselines/chm_comparison.json`
- `outputs/baselines/vanilla_ft/eval/*_with_chm.json` (5 files)
- `outputs/baselines/flare/*_cycle0.json` (5 files, the regenerated FLARE outputs)
- `outputs/baselines/flare/*_with_chm.json` (5 files, **only if Phase 2b ran**)
- `outputs/baselines_vast_rerun_logs/` (all `/tmp/*.log` symlinks or copies)

### 7b — Print the user's `scp` command to pull the tarball

```bash
TARBALL=$(ls -t /tmp/rescore_results_*.tgz | head -1)
HOSTNAME=$(hostname)
echo ""
echo "================================================================"
echo "USER: scp this tarball back to your local machine:"
echo "  scp -P <vast_ssh_port> root@<vast_ssh_host>:${TARBALL} /tmp/"
echo "Then on local:"
echo "  cd /workspace/caem && tar xzf /tmp/$(basename ${TARBALL})"
echo "================================================================"
echo ""
ls -lh ${TARBALL}
```

### 7c — Stop the Vast instance to stop billing

**Do not destroy the instance yet** — if the user wants to re-run anything, the FAISS index + caches are still on disk. Just stop it.

The user must click "Stop" on the Vast dashboard. You cannot stop the instance from inside it (well, you can `shutdown now`, but that triggers a destroy on most Vast hosts).

Print this message to the user and exit:

```
Phase 2 done. Vast instance is idle.

USER ACTION REQUIRED:
  1. scp the tarball above to your local machine
  2. Stop the Vast instance on the dashboard (not Destroy — keep state in case
     a rerun is needed)
  3. Once you confirm the tarball arrived on local, you can Destroy the instance

If Destroy is clicked, the 64 GB FAISS index will be re-downloaded on any
future Phase 2 rerun (~15 min + $0.18 internet).
```

---

## 8. Error handling

| Failure mode | Symptom | Action |
|--------------|---------|--------|
| FAISS download stalls | Bootstrap takes >60 min | Kill it, retry `bash scripts/vast_launch.sh bootstrap` (uses `resume_download=True`) |
| MiniCheck OOM at bs=32 | CUDA out of memory in rescore log | Edit `scripts/vast_launch.sh` to set `--batch_size 16`, retry only the failed step |
| Anthropic rate limit (429) during Haiku rescore | Script retries automatically | Wait — script throttles to 45 req/min and resumes |
| CUDA version mismatch | torch.cuda.is_available() == False after bootstrap | Verify `pip list \| grep torch` shows `2.4.0+cu121`, not `2.5.x+cu124`; reinstall if needed |
| Vast pre-emption | Instance reboots mid-step | Re-rent same instance (or new one), re-run vast_launch.sh from the step that was in flight; partial outputs on disk survive in `outputs/baselines/` |
| Disk fills up | "No space left on device" | `du -sh /workspace/caem/* \| sort -h`; the 174 GB disk should not fill. If outputs grow >50 GB, something is wrong — check for runaway log files |
| Composite calibration mis-pin | rescore log shows wrong `composite_calibration.json` path | Verify `scripts/vast_launch.sh` hardcodes `outputs/full_run/cycle_3/composite_calibration.json`, not a different cycle |

For any error not in this table, **stop and produce a final report describing what failed**. Do not improvise destructive actions like `git restore` or deleting partial outputs without authorization.

---

## 9. Final summary template

When everything completes, produce a structured summary at `/tmp/PHASE2_SUMMARY.txt` with this format:

```
========================================
CAEM PHASE 2 SUMMARY — <date> BDT
========================================

Phase 2a (Partial Path B): COMPLETE
  - B7 FLARE regen:        5/5 benches, EM range [X.XXX, X.XXX]
  - B6@C5 verifier rescore: 5/5 benches, CHM range [X.XXX, X.XXX]
  - chm_comparison.json rebuilt with B6+B7

Phase 2b (B7 verifier rescore): COMPLETE | SKIPPED
  - reason: <budget exceeded | completed in ~X.X h>
  - if SKIPPED: B7 CHM column in the final tex will reflect pre-fix predictions
    with a footnote in Ch5 §5.1

Phase 2 Haiku rescore (B7 TQA): COMPLETE
  - em_llm_judged: X.XXX over 300 samples

Vast spend total: $X.XX of $5.43 credit
Remaining credit: $X.XX
Wall time:        X h XX min

Tarball location: /tmp/rescore_results_<timestamp>.tgz (X MB)

USER NEXT STEPS:
  1. scp the tarball to local
  2. tar xzf <tarball> -C /workspace/caem/
  3. Stop the Vast instance on dashboard
  4. Run Phase 3 finalisation locally (regenerate tab_baseline_pooled,
     tab_sig_test, update Ch5 prose, recompile, push, submit)

KNOWN CAVEATS / OPEN ISSUES:
  <if any — e.g. Phase 2b skipped, one benchmark had retries, etc.>
========================================
```

Save this file, print it to stdout, and exit cleanly.

---

## 10. References (canonical project state — do not duplicate or override)

- Repo: `https://github.com/aksaN000/caem-thesis`, branch `branch/ruc-classifier`
- Latest commit at runbook authoring: `86a0e54`
- Locked CAEM composite calibration: `outputs/full_run/cycle_3/composite_calibration.json` — do not point at any other cycle
- HF dataset for FAISS index: `aksaN000/caem-passage-index-21m` (private; access via user's HF token)
- HF model for verifier judge: `lytang/MiniCheck-Flan-T5-Large`
- HF backbone: `Qwen/Qwen2.5-3B-Instruct`
- Project-level CLAUDE.md anchor file: `/workspace/caem/CLAUDE.md` — read for additional context (time zone is BDT, commits must not include the Claude Co-Authored-By trailer)

---

## 11. What you must NOT do

- **Do not commit the user's Anthropic API key** to git. It's in `~/.bashrc` and the env var; do not hard-code it into any file in the repo.
- **Do not run `git restore` or `git checkout -- .`** to "fix" anything. If something is wrong, surface it. Destructive git operations require user authorization.
- **Do not modify `outputs/full_run/cycle_3/composite_calibration.json`** — it's the locked calibration that anchors every CHM number in the thesis.
- **Do not push results to GitHub from this Vast instance** — `outputs/` is gitignored. Just produce the tarball; the user handles the local integration + push.
- **Do not run the C-anchor (`--caem_cycle 10`) rescore branch** — the locked configuration is cycle 3 composite, cycle-5 weights. Cycle 10 doesn't exist in the realised trajectory.
- **Do not add the Claude `Co-Authored-By` trailer** to any commit message. (User preference, enforced project-wide.)

---

End of runbook. Execute Phase 2a now and produce the summary at the end.
