# CAEM Demo Quickstart — One-Page Operator Cheat Sheet

**For:** day-of demo, friends trying it out, or pre-defense rehearsal. Print or pin this open in a second monitor.

---

## 1. Launch the demo (lab PC, ~60-120 sec to ready)

### Pre-defense rehearsal (cycle-3 memory)

```bash
cd /workspace/caem
python scripts/caem_demo_server.py \
    --memory outputs/full_run/memory_store_cycle_3 \
    --passage_index data/passage_index \
    --model Qwen/Qwen2.5-3B-Instruct \
    --device cuda \
    --port 8000
```

### Defense day (cycle-10 production-swapped memory)

```bash
cd /workspace/caem
python scripts/caem_demo_server.py \
    --memory outputs/production/memory_store/memory_store \
    --passage_index data/passage_index \
    --model Qwen/Qwen2.5-3B-Instruct \
    --device cuda \
    --port 8000
```

### Wait for the ready line

```
Pipeline ready. Memory: NNN episodes.
```

If you don't see it within ~3 minutes, see "Troubleshooting" below.

---

## 2. Open in browser

| Where | URL |
|---|---|
| Lab PC | `http://localhost:8000` |
| Same network (LAN) | `http://<lab-pc-ip>:8000` |
| Public (after step 4) | The Cloudflare URL printed by `expose_demo_remote.sh` |

---

## 3. Endpoints (for curl tests or API clients)

```bash
# Health check
curl http://localhost:8000/health

# Stats
curl http://localhost:8000/stats

# Submit a query
curl -X POST http://localhost:8000/query \
    -H 'Content-Type: application/json' \
    -d '{"question": "Game of Thrones is a television series. Answer with one of: supports, refutes, not enough info."}'
```

---

## 4. Optional — expose publicly via Cloudflare tunnel (~5 sec setup)

```bash
bash scripts/expose_demo_remote.sh
```

Prints a public URL like `https://<random>.trycloudflare.com`. Anyone with the URL can submit queries. **No login or auth** — only share with trusted parties; tunnel is ephemeral and dies when you stop the script.

---

## 5. Defense flow (refer to `PANEL_DEMO_SCRIPT.md` for the full script)

1. Walk through Q1-Q7 from the demo script (~6-8 min)
2. Open the floor for panel questions (~5-10 min)
3. If anything fails, play backup video (Phase B.5.3 of `PRODUCTION_NEXT_SESSION_PLAN.md`)

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Server hangs at "Loading Qwen generator" for >3 min | First-time HuggingFace download | wait — first launch downloads ~6 GB of weights; subsequent launches use cache |
| `RuntimeError: CUDA out of memory` | Another GPU process running, or batch size too big | `nvidia-smi` to check; kill stray processes; relaunch |
| `FileNotFoundError: outputs/full_run/memory_store_cycle_3.faiss` | Wrong cycle picked or memory not on this machine | `ls outputs/full_run/memory_store_cycle_*.faiss` to see what's available; pick the highest cycle number |
| `FileNotFoundError: data/passage_index/...` | Wikipedia FAISS index missing | restore from gdrive: `rclone copy gdrive:caem-passage-index-21m/passage_index data/passage_index` (~64 GB, ~30 min) |
| Browser shows "This site can't be reached" | Server not bound to 0.0.0.0, or firewall blocking 8000 | restart with explicit `--host 0.0.0.0`; check firewall |
| `/query` returns 503 "Pipeline not loaded" | Server still starting | wait for the "Pipeline ready" line in terminal |
| `/query` returns 500 with traceback | Model crash mid-generation | check terminal for the traceback; retry the query (often transient) |
| Cloudflare tunnel URL doesn't work | Tunnel was killed or rotated | re-run `bash scripts/expose_demo_remote.sh` to get a new URL |
| Latency > 30 sec on a query | First query warms up RAG; or query hit Tier 3 with cold passage cache | normal on first query; subsequent queries are faster |
| Confidence percentages look wrong | Wrong calibration JSON loaded | confirm `caem/config.py` `composite_calibration_path` and `conformal_gate_path` point at the production paths |

---

## 7. Stop the demo

```bash
# In the terminal running the server: Ctrl+C
# If tunnel was running: Ctrl+C in its terminal too
```

---

## 8. Failsafe — terminal-only fallback (no browser available)

If the browser path fails completely (network down, projector broken), the legacy CLI still works:

```bash
python scripts/caem_chat.py \
    --memory outputs/full_run/memory_store_cycle_3 \
    --passage_index data/passage_index
```

Type questions, see ANSI-colored tags in the terminal. Slower than the web UI for panel demos but always works.

---

## 9. Backup demo video

`outputs/production/audit/demo_backup.mp4` (created in Phase B.5.3 of `PRODUCTION_NEXT_SESSION_PLAN.md`). Play this if everything else fails.

---

## 10. Where to find more detail

| Topic | File |
|---|---|
| Full deployment plan | `PRODUCTION_NEXT_SESSION_PLAN.md` |
| Operator runbook (production cycles, label sourcing, drift detection) | `docs/PRODUCTION_RUNBOOK.md` |
| 7-question demo script | `docs/PANEL_DEMO_SCRIPT.md` |
| Architecture chapter | `thesis_report/chapters/chapter_4.tex` |
| Empirical results chapter | `thesis_report/chapters/chapter_5.tex` |
| Theorems chapter | `thesis_report/chapters/chapter_4.tex` §11 |
