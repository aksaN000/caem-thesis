# CAEM Production Deployment — Next-Session Plan

**Created:** 2026-05-04
**Goal:** Ship one unified production-grade CAEM deployment that you, your friends, and the defense panel can use locally on the lab PC, remotely via tunnel, or post-defense on a consumer laptop via Ollama.
**Source of authority:** `docs/PRODUCTION_RUNBOOK.md` (deployment design), `branch_C_log.md` 2026-05-04 entry (corrections + framing).

Mark each step `[x]` once complete. After every batch, commit + push.

---

## Step status legend
- `[ ]` = not started
- `[~]` = in progress
- `[x]` = done

---

## Current state — HALTED 2026-05-04 23:13 UTC pending Vast recharge

The Phase 1a trajectory was halted cleanly at the cycle-5 SIL-and-reconsideration phase (mid-cycle) on 2026-05-04 to stop GPU spend until the next Vast credit recharge. The halt is by user choice, not a failure.

### What completed before the halt

- Cycles 0 through 4: full trajectory artefacts on disk and on gdrive (`gdrive:caem-phase1a/full_run/cycle_{0,1,2,3,4}/`). Per-cycle `model.pt`, `composite_calibration.json`, `conformal_gate.json`, `calibrated_thresholds.json`, `meta.pkl`, plus `memory_store_cycle_N.{faiss,meta}`, `deferred_buffer_cycle_N.pkl`, and `retroverify_cycleN.json` are all backed up. (Cycle 2's calibration JSONs are split across `cycle_2/`, `cycle_2_artefacts/`, and `cycle_2_calibration/` for legacy reasons but are complete.)
- Path Y orchestrator patch was applied and tested live: cycle 5 SIL fine-tune completed cleanly (MMLU retention 1.04× pristine, well above the 0.93 threshold), and the deferred-reconsideration sweep fired for the first time in the trajectory at 23:11:43 with `Deferred reconsideration: sweeping 6422 entries (promote_threshold=0.451, ttl=2 cycles)` — empirical confirmation that the orchestrator gap is closed and TTL=2 reconsideration is now active.
- `cycle_5/model.pt` and `cycle_5/meta.pkl` exist locally (~6.2 GB) but are mid-cycle artefacts: cycle 5's eval pass, retroverify pass, and per-cycle persistence files (`memory_store_cycle_5.*`, `deferred_buffer_cycle_5.pkl`, `retroverify_cycle5.json`) were not written before the halt. These will be regenerated on resume.

### What was running at the moment of halt

The reconsideration sweep was in flight (6,422 entries against the post-cycle-5 verifier); the per-cycle eval pass had not started. The cycle-5 close artefacts were not written, so on resume the runner restarts cycle 5 from scratch (the `--resume_from_cycle 5` path restores from cycle 4 and re-runs cycle 5's SIL + reconsideration + eval).

### Resume protocol (next session, after Vast recharge)

1. Spin up a Vast 5090 instance and pull the repo + restore artefacts:
   ```bash
   git clone https://github.com/aksaN000/caem-thesis.git
   cd caem-thesis
   git checkout feat/qwen-3b-goal1
   # restore outputs/ from gdrive
   rclone copy gdrive:caem-phase1a/full_run/ outputs/full_run/ --transfers 4 --checkers 8
   ```
2. Verify cycle-4 artefacts on disk (the resume base): `cycle_4/{model.pt,composite_calibration.json,conformal_gate.json,calibrated_thresholds.json,meta.pkl}`, `memory_store_cycle_4.{faiss,meta}`, `deferred_buffer_cycle_4.pkl`, `retroverify_cycle4.json`, `mmlu_baseline.json`, `dataset_splits.json`.
3. Confirm the orchestrator patch is in place at `scripts/run_experiment.py:1591` (the `deferred_buffer=pipeline.deferred_buffer, reconsider_fn=pipeline.make_reconsider_deferred_fn()` kwargs).
4. Confirm `caem/config.py:770` has `deferred_buffer_ttl_cycles: int = 2`.
5. Launch resumed runner in tmux:
   ```bash
   tmux new-session -d -s plan_a -x 220 -y 60 \
       'source /venv/main/bin/activate && cd /workspace/caem && \
        ./run_phase1a.sh --resume_from_cycle 5 2>&1 | tee -a outputs/phase1a_runner.log'
   ```
6. Verify within 60s: `grep "RESUMING EXPERIMENT FROM CYCLE 5" outputs/full_run/run.log` returns the marker line.
7. Verify within ~30 min (during cycle 5 SIL): a `Deferred reconsideration: sweeping N entries` line appears in run.log. Without it the orchestrator patch did not engage and the run should be halted again.

### Estimated resume cost

Cycles 5 through 10 = six cycles. Each cycle is roughly 4-6 hours wall-clock under the current batched-cal envelope (SIL fine-tune ~30 min, reconsideration sweep ~5 min, eval pass on seven benchmarks ~3-4 hours, retroverify ~30 min). Budget ~30 GPU-hours for the trajectory completion alone, ~$15-20 of Vast credit. Add another ~$15-20 for B.5b baseline rescoring (Phase B.5b in this plan) and ~$10-15 for the B1-B7 baseline runs themselves (Phase B preceding the rescore). Total recharge target for the path through to the cycle-10 close + comparison panel is around ~$40-55 of Vast credit.

---

## Phase A — Pre-defense polish (parallel to cycles 5-10, ~7 hours total)

All Phase A steps are CPU-only with zero GPU contention.

### A.1 Cleanup + verification (today, ~30 min)

- [x] **A.1.1** Mark `scripts/caem_chat.py` as legacy (one-line header note pointing at `caem_demo_server.py`). Do NOT delete; keep as terminal-only fallback. **(2026-05-04 — header note added)**
- [x] **A.1.2** Smoke-test `scripts/caem_demo_server.py` imports on local CPU (no model load): `python -c "import scripts.caem_demo_server"` should succeed. **(2026-05-04 — clean import)**
- [x] **A.1.3** Smoke-test `scripts/caem_chat.py` imports on local CPU. **(2026-05-04 — clean import)**
- [x] **A.1.4** Decide demo memory snapshot for pre-defense rehearsal: cold-start (260 entries) OR cycle-3 (rich enough to demo Tier 1) OR wait for cycle-10 (final). Log decision in this file. **(2026-05-04 — DECISION: cycle-3 memory snapshot for pre-defense rehearsal. Rationale: cycle-3 is post-3-SIL-cycles trained memory with Tier 1 hits visible on cal fold (1+8+9 across c1-c3 = 18 hits) and stream-chunk Tier 1 = 3.07% on FEVER c3, large enough to demonstrate memory routing live. Available locally at `outputs/full_run/memory_store_cycle_3.{faiss,meta}` (~3 MB FAISS) and on gdrive at `caem-phase1a/full_run/cycle_3/`. Switch to cycle-10 in Phase B.4 once it closes ~May 15.)**

### A.2 Demo assets (this week, ~3 hours)

- [x] **A.2.1** Write `docs/PANEL_DEMO_SCRIPT.md` with 5-7 questions exercising: verified answer, deferred answer, refusal (ABSTAIN), refusal (DISCARD), memory hit if available, safety override on out-of-domain query. **(2026-05-04 — 7 questions written, all anchored in actual cycle-3 cal-fold readings; ~15 min total demo + Q&A budget)**
- [x] **A.2.2** Write `docs/DEMO_QUICKSTART.md` — one-page operator cheat sheet for day-of demo (single launch command, troubleshooting, port forwarding, fallback to legacy CLI). **(2026-05-04 — 10-section quickstart with launch commands for cycle-3 + cycle-10, endpoints, troubleshooting matrix, terminal-fallback path)**
- [x] **A.2.3** Write `scripts/expose_demo_remote.sh` — Cloudflare tunnel one-liner exposing `localhost:8000` to a public URL for hybrid/remote defense. **(2026-05-04 — script written, chmod +x, sanity-checks demo server is listening before exposing, prints public URL via cloudflared ephemeral tunnel)**

### A.3 UI polish (this week, ~2.5 hours)

All edits to `scripts/caem_demo_server.py`'s inlined `_INDEX_HTML`.

- [x] **A.3.1** Add expandable "Evidence" section showing top-3 reranked passages on click (~30 lines HTML + JS). Demonstrates grounding to panel. **(2026-05-04 — `<details>` panel showing top-3 reranked passages with text + passage id; surfaces from `vout.top_passages` via `/query` endpoint)**
- [x] **A.3.2** Add "Memory match" sidebar showing nearest-neighbor stored episode + cosine similarity (~50 lines HTML + JS + JSON field plumbed from pipeline result). Demonstrates Tier 1 routing. **(2026-05-04 — `<details>` panel showing matched entry id, cosine sim %, stored question, stored answer, storage cycle, source benchmark, stored u_stored; uses _MEMORY_STORE.search_with_ids in /query)**
- [x] **A.3.3** Add tier + latency badges to response card header (currently hidden in expandable explain block; surface to top). **(2026-05-04 — green/blue/purple tier badges + latency badge in response card header alongside tag and confidence%)**

### A.4 Dry-run rehearsal (this week, ~1.5 hours) — OPERATOR-EXECUTED

⚠ A.4 must be run on the lab PC (or Vast instance with GPU) by the operator. The conversation-side prep (A.1 to A.3) is complete; the dry-run is the empirical verification step.

- [ ] **A.4.1** Run `caem_demo_server.py` locally on the lab PC with the chosen Phase A.1.4 memory snapshot. **Command:** see `docs/DEMO_QUICKSTART.md` §1.
- [ ] **A.4.2** Walk through `PANEL_DEMO_SCRIPT.md` end-to-end. Time each query. Capture screenshots. Save screenshots to `outputs/production/audit/dryrun_screenshots_YYYYMMDD/`.
- [ ] **A.4.3** Test the tunnel script: launch `bash scripts/expose_demo_remote.sh`, share the public URL with one friend, verify they can submit a query and see the rendered card.
- [ ] **A.4.4** Note any UX rough edges + fix in A.3 if needed. Append fixes as new commits referencing A.3.x in the message.

---

## Phase B — Cycle 10 close + production swap (~30 minutes once cycle 10 lands, ~May 15)

### B.1 Halt the live run cleanly

- [ ] **B.1.1** Wait for `Cycle 10 done in X min.` log marker in `outputs/full_run/run.log`.
- [ ] **B.1.2** Verify cycle-10 artifacts exist: `outputs/full_run/cycle_10/{conformal_gate,composite_calibration,calibrated_config_cycle10,model.pt}.json/pt`, `outputs/full_run/memory_store_cycle_10.{faiss,meta}`, `outputs/full_run/deferred_buffer_cycle_10.pkl`, `outputs/full_run/retroverify_cycle10.json`.
- [ ] **B.1.3** `tmux send-keys -t plan_a C-c` — interrupt the runner cleanly, wait 30s.
- [ ] **B.1.4** Confirm runner exited cleanly (no stack trace, exit code 0 in `outputs/phase1a_runner.log`).
- [ ] **B.1.5** `tmux send-keys -t watchdog_3plus C-c` — stop the watchdog (it auto-exits on cycle-10 close marker, but kill explicitly to be safe).

### B.2 Run the swap script (per `docs/PRODUCTION_RUNBOOK.md` §3)

- [ ] **B.2.1** `mkdir -p outputs/production/memory_store outputs/production/audit`.
- [ ] **B.2.2** Copy cycle-10 calibration artifacts into `outputs/production/`:
  - `cp outputs/full_run/cycle_10/composite_calibration.json outputs/production/`
  - `cp outputs/full_run/cycle_10/conformal_gate.json outputs/production/`
  - `cp outputs/full_run/cycle_10/calibrated_config_cycle10.json outputs/production/calibrated_config.json`
- [ ] **B.2.3** Copy cycle-10 memory store + deferred buffer:
  - `cp outputs/full_run/memory_store_cycle_10.faiss outputs/production/memory_store/memory_store.faiss`
  - `cp outputs/full_run/memory_store_cycle_10.meta outputs/production/memory_store/memory_store.meta`
  - `cp outputs/full_run/deferred_buffer_cycle_10.pkl outputs/production/deferred_buffer.pkl`
- [ ] **B.2.4** Copy cycle-10 model checkpoint to production location (large file, ~6 GB):
  - `cp outputs/full_run/cycle_10/model.pt outputs/production/cycle_0/model.pt` (production cycle counter starts at 0)
  - **The demo server loads this via `--checkpoint outputs/production/cycle_0/model.pt` (added 2026-05-04). Without this flag the server runs base HuggingFace Qwen, which does NOT reflect post-SIL CAEM behaviour.**
- [ ] **B.2.5** Edit `caem/config.py` to point composite + gate paths at production:
  - `composite_calibration_path = "outputs/production/composite_calibration.json"`
  - `conformal_gate_path        = "outputs/production/conformal_gate.json"`
- [ ] **B.2.6** Write `outputs/production/cycle_policy.json` with chosen trigger policy (default: `"strategy": "frozen", "rationale": "defense-mode, no further cycles"` for the defense; switch to `hybrid` post-defense if going to actual production).
- [ ] **B.2.7** Initialize `outputs/production/audit/deployment.log` with timestamp + research_last_cycle=10.

### B.3 Smoke-test production swap

- [ ] **B.3.1** Launch demo server pointed at production memory: `python scripts/caem_demo_server.py --memory outputs/production/memory_store/memory_store --passage_index data/passage_index --port 8000`.
- [ ] **B.3.2** Open `http://localhost:8000` in a browser. Verify UI renders.
- [ ] **B.3.3** Submit one query. Verify response card shows tag + confidence + tier + latency.
- [ ] **B.3.4** `GET /health` should return `{"ok": true, "memory_size": <cycle-10-count>, ...}`.
- [ ] **B.3.5** `GET /stats` should return the cycle-10 memory size.

### B.4 Final dry-run with cycle-10 memory

- [ ] **B.4.1** Run full `PANEL_DEMO_SCRIPT.md` walkthrough against the cycle-10-backed server.
- [ ] **B.4.2** Time each query, log to `outputs/production/audit/demo_dryrun_cycle10.log`.
- [ ] **B.4.3** Compare tier hit rates and tag distribution to expected cycle-10 readings.
- [ ] **B.4.4** Capture screenshots for backup video.

### B.5 Backup the production state

- [ ] **B.5.1** `rclone copy outputs/production gdrive:caem-phase1a/production/ --transfers 4 --checkers 8` — full backup to gdrive.
- [ ] **B.5.2** Tar + copy to USB stick or external drive: `tar czf caem_production_cycle10_$(date +%Y%m%d).tar.gz outputs/production/`.
- [ ] **B.5.3** Record a screen-capture video of one successful demo walkthrough as failsafe (~5 min). Store alongside the tarball.

### B.5b Baseline-output verifier rescoring for CHM comparison (post-cycle-10, ~25-35h, ~$15-20 GPU)

**Why:** B1-B7 baseline outputs record only bare prediction + EM/F1; they do not carry CAEM's 9 verifier signals or the 4-outcome decision class. Without these, CHM (the 8-subtype confident-error metric registered in Ch5 §sec:setup-metrics) cannot be computed on baseline outputs and the cross-comparison "CAEM has lower hallucination rate than baselines" claim is unsupported. The rescoring pass measures everyone's outputs through the same locked CAEM verifier (cycle-10 calibration) — same metric, same instrument, different generators — which is the methodologically standard approach for cross-system hallucination comparison (FActScore, MiniCheck-style benchmarks).

- [ ] **B.5b.1** Write `scripts/rescore_baselines_through_verifier.py` (~80 lines): for each baseline in B1-B7, for each benchmark eval JSON row, run `caem.verifier.verify(question, prediction)` to populate the 9 signals; apply the 4-outcome decision tree at the locked cycle-10 thresholds; recompute CHM 8-subtypes per row; aggregate per baseline per benchmark.
- [ ] **B.5b.2** Use the locked cycle-10 verifier configuration uniformly across all baselines: `--composite_calibration outputs/production/composite_calibration.json --conformal_gate outputs/production/conformal_gate.json --checkpoint outputs/production/cycle_0/model.pt` (post-Phase-B production swap). Same instrument applied to every system.
- [ ] **B.5b.3** Run on a Vast GPU instance after the B1-B7 baselines themselves complete: ~3500 verifier calls per baseline × 7 baselines ≈ 25-35 GPU-hours total. Estimate ~$15-20 of credit.
- [ ] **B.5b.4** Output: `outputs/baselines/{baseline}/{benchmark}_cycle0_with_chm.json` per (baseline, benchmark). Contains the original prediction + 9 verifier signals + decision class + CHM 8-subtype flags + EM/F1.
- [ ] **B.5b.5** Aggregate: `outputs/baselines/chm_comparison.json` with per-baseline CHM rate vs CAEM cycle-10 CHM rate, broken down by benchmark and by subtype. This is the canonical "CAEM vs baselines on hallucination rate" table for Ch5 §sec:comp-baselines.
- [ ] **B.5b.6** Update Ch5 §sec:summary-hypotheses H1 verdict cells against the baseline-CHM comparison.

**Methodological preservation:** the rescoring contract uses the **locked cycle-10 CAEM verifier** as the measurement instrument; CAEM's per-cycle internal verifier improvement is a separate property. The cross-comparison measures generators (CAEM vs zero-shot vs CoT vs RAG vs CoT+RAG vs 5-shot vs vanilla-FT vs EWC-FT) under one fixed verification instrument. This isolates the generator-side contribution from the verifier-side improvement.

**Phase 1b future work registered separately:** instrument-side ablation — measure baseline CHM under each cycle's verifier (not just cycle-10) to characterise how the verifier itself contributes to the CAEM advantage. NOT part of Phase 1a defense scope.

### B.6 Counterfactual reconstruction — REMOVED 2026-05-04 (replaced by Path Y live data)

**Originally registered** as a post-cycle-10 ~$3-5 / 6-10h GPU job to replay `DeferredBuffer.reconsider()` against frozen cycle-1..10 artefacts, recovering the per-entry promotion outcome that the live trajectory missed because of the orchestrator gap at `scripts/run_experiment.py:1591`.

**Replaced 2026-05-04** by Path Y: the orchestrator gap was patched (`deferred_buffer=pipeline.deferred_buffer, reconsider_fn=pipeline.make_reconsider_deferred_fn()` added to the `sil.run_cycle()` call), the `deferred_buffer_ttl_cycles` config was reverted from 4 back to the originally-registered 2, the cycle-4-close halt-restart procedure was executed cleanly, and the runner resumed from cycle 5 with reconsideration ACTIVE at every subsequent cycle boundary. See `scripts/halt_and_resume_with_deferred_fix.sh` and `branch_C_log.md` 2026-05-04 entries (Path Y + TTL=2 revert).

**What this means for the receipt:**
- The deferred-pool empirical receipt is now provided by **live cycle-5-through-10 trajectory data** rather than by counterfactual reconstruction.
- Cycle 5 is the documented one-time backlog reconciliation: the ~6,400 entries deferred across cycles 1-4 (which all sat at age=0 because the dormant pass never advanced their ages) are rescored together under the post-cycle-5 verifier, with promotion or TTL-drop outcomes logged in `outputs/full_run/run.log` and persisted in `deferred_buffer_cycle_5.pkl`.
- Cycles 6-10 are the steady-state two-track survivability window: each cycle boundary's reconsideration sweep emits per-entry promote / age / TTL-drop log lines, and the per-cycle stored-pool growth attributable to the deferred path is read directly off the trajectory.
- The cycle-5-through-10 window is the empirical receipt for the deferred-pool half of the two-track design; cycles 1-4 contribute to the storage-pool receipt only. Ch5 §sec:disc-limitations was updated 2026-05-04 to register this two-phase reading and document the cycle-4-close transition as the methodology-refinement point.

**Methodological note:** Path Y delivers a stronger receipt than the counterfactual reconstruction would have, because the live trajectory captures the path-dependent downstream effect of promoted entries on subsequent SIL fine-tunes (the counterfactual could only recover per-entry outcome in isolation). Zero additional GPU cost for the empirical receipt.

---

## Phase C — Defense day (D-day, ~15 minutes prep)

### C.1 Pre-defense launch

- [ ] **C.1.1** Boot the lab PC. Open terminal in `/workspace/caem`.
- [ ] **C.1.2** Run quickstart launch command from `docs/DEMO_QUICKSTART.md`.
- [ ] **C.1.3** Wait for "Pipeline ready. Memory: N episodes." log line (~60-120s on lab GPU).
- [ ] **C.1.4** Verify `http://localhost:8000` loads in lab PC browser.

### C.2 If defense is hybrid / remote / projector-driven

- [ ] **C.2.1** Run `scripts/expose_demo_remote.sh` to start the Cloudflare tunnel.
- [ ] **C.2.2** Note the public URL printed by the tunnel.
- [ ] **C.2.3** Test from a phone or second device that the public URL renders.

### C.3 During defense

- [ ] **C.3.1** Walk through `PANEL_DEMO_SCRIPT.md` with the panel (5-10 min).
- [ ] **C.3.2** Invite panel to type their own queries (any duration).
- [ ] **C.3.3** If anything fails: play the backup video from B.5.3.

---

## Phase D — Post-defense polish, optional Phase 1c (~15-20 hours, post-defense)

### D.1 Ollama backend (consumer-laptop deployment)

- [ ] **D.1.1** Add `caem/inference/__init__.py` + `caem/inference/ollama_backend.py` (~150-200 lines): adapter implementing the same `model.generate(...)` interface CAEM expects, proxying via Ollama's HTTP API.
- [ ] **D.1.2** Implement `u_token` extraction from Ollama's logprob output.
- [ ] **D.1.3** Implement `u_dropout` substitute via temperature-resampling variance (K=3 generations at temp=0.5, compute variance).
- [ ] **D.1.4** Add `--backend ollama` flag to `caem_demo_server.py`. Default stays `huggingface`.
- [ ] **D.1.5** Test the Ollama-backed pipeline end-to-end with `PANEL_DEMO_SCRIPT.md` on a consumer laptop. Document VRAM/RAM footprint.
- [ ] **D.1.6** Document the Ollama path in `docs/PRODUCTION_RUNBOOK.md` as a "consumer-hardware deployment option" appendix.

### D.2 Live-cycle production mode (optional, post-defense)

- [ ] **D.2.1** Implement `scripts/production_sil_finetune.py` per RUNBOOK §5.3(A).
- [ ] **D.2.2** Implement `scripts/swap_production_model.py` per RUNBOOK §5.3(B).
- [ ] **D.2.3** Implement `scripts/build_cycle_calibration_batch.py` per RUNBOOK §5.1.
- [ ] **D.2.4** Implement the cycle-policy watcher per RUNBOOK §3.4.
- [ ] **D.2.5** Set up a serving log path with daily rotation per RUNBOOK §4.

---

## Decisions log

Append every operator decision here with date.

- 2026-05-04 — Plan created. Single product = `caem_demo_server.py`. `caem_chat.py` marked legacy. Three access modes: local browser, public URL via tunnel, Ollama (Phase 1c). Default backend: HuggingFace bf16 Qwen-3B for defense.
- 2026-05-04 — A.1.4 decision: pre-defense rehearsal demo will run on cycle-3 memory (`outputs/full_run/memory_store_cycle_3.{faiss,meta}`, ~3 MB, 18 cal-fold Tier 1 hits + 3% stream-chunk Tier 1 on FEVER). Re-stage with cycle-10 memory after cycle-10 close per Phase B.4.
- (next decision goes here)

---

## Notes

- All `outputs/production/` paths are gitignored. The swap creates them locally + gdrive; never committed to repo.
- `caem/config.py` edits in B.2.5 are committable changes that lock the production paths into the codebase. Commit after the swap.
- Phase 1c (Ollama) is fully optional for defense. Defense passes with Phase A + B alone.
