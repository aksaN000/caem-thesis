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
