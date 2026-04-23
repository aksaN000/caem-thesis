#!/usr/bin/env python
"""
scripts/caem_demo_server.py
============================
FastAPI server — exposes the CAEM pipeline with the production
confidence envelope over HTTP + a minimal single-page web UI.

Defense-demo-grade: examiner types a question in the browser, sees
CAEM's response with the verified/provisional/conflicting/insufficient
tag, continuous confidence score, and the underlying 7-signal breakdown
on request.

Endpoints
---------
  GET  /                    Single-page HTML UI
  POST /query               Run a query, return production_response JSON
  GET  /health              Model + memory status
  GET  /stats               Memory size, cycle count, config summary

Usage
-----
    # After Phase 1a (trained cycle_10):
    uvicorn scripts.caem_demo_server:app --host 0.0.0.0 --port 8000 \\
        --factory  # construct via make_app()

    # Or directly:
    python scripts/caem_demo_server.py --memory outputs/full_run/cycle_10/memory_store_cycle_10

    # Cold-start test (Step 7.0 gap):
    python scripts/caem_demo_server.py --memory outputs/cold_start_memory/memory_store

Then open http://<host>:8000/ and ask questions. Or curl:
    curl -X POST http://localhost:8000/query \\
        -H 'Content-Type: application/json' \\
        -d '{"question": "Who wrote Hamlet?"}'
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global pipeline handle (set by make_app / CLI entry)
_PIPELINE = None
_MEMORY_STORE = None
_DEFAULT_CADENCE: Optional[str] = None


# --------------------------------------------------------------------- #
# Pydantic request/response models (module-level so FastAPI can resolve) #
# --------------------------------------------------------------------- #
try:
    from pydantic import BaseModel

    class QueryIn(BaseModel):
        question: str
        deferred_cadence: Optional[str] = None
        deferred_eta: Optional[str] = None

    class QueryOut(BaseModel):
        question: str
        production_response: Optional[Dict[str, Any]]
        tier: int
        latency_ms: float
        stored: bool
        explanation: Optional[Dict[str, Any]] = None
except ImportError:  # pragma: no cover
    QueryIn = None  # type: ignore[assignment]
    QueryOut = None  # type: ignore[assignment]


# --------------------------------------------------------------------- #
# Pipeline construction                                                  #
# --------------------------------------------------------------------- #
def _build_pipeline(
    memory_path: Path,
    passage_index: Path,
    model: str,
    device: str,
):
    import torch

    from caem.config import CAEMConfig
    from caem.memory.encoder import QueryEncoder
    from caem.memory.store import EpisodicMemoryStore
    from caem.model_loader import load_base_generator
    from caem.pipeline import CAEMPipeline
    from caem.retrieval.rag import PassageStore
    from caem.verification import load_verifier_judge

    logger.info("Loading config + models...")
    config = CAEMConfig()

    t0 = time.perf_counter()
    gen_model, tokenizer = load_base_generator(
        model, use_sdpa=True, use_torch_compile=True,
    )
    logger.info("Qwen loaded in %.1fs", time.perf_counter() - t0)

    encoder = QueryEncoder(device=device)
    passage_store = PassageStore.load(str(passage_index))
    judge, nli_model, nli_tokenizer = load_verifier_judge(
        config, device, allow_fallback=True,
    )

    cross_encoder = None
    if config.cross_encoder_model:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(
            config.cross_encoder_model, device=device,
            automodel_args={"torch_dtype": torch.bfloat16} if device != "cpu" else {},
        )

    memory_store = EpisodicMemoryStore(config=config)
    memory_store.load(str(memory_path))

    pipeline = CAEMPipeline(
        model=gen_model, tokenizer=tokenizer, encoder=encoder,
        judge=judge, nli_model=nli_model, nli_tokenizer=nli_tokenizer,
        passage_store=passage_store, memory_store=memory_store,
        cross_encoder=cross_encoder, config=config,
    )
    logger.info("Pipeline ready. Memory: %d episodes.", len(memory_store))
    return pipeline, memory_store


# --------------------------------------------------------------------- #
# FastAPI app                                                            #
# --------------------------------------------------------------------- #
def make_app():
    """Build the FastAPI app. Models must be pre-loaded via _set_pipeline()."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="CAEM Demo Server", version="1.0")

    @app.get("/")
    def root() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": _PIPELINE is not None,
            "memory_size": len(_MEMORY_STORE) if _MEMORY_STORE else 0,
            "default_cadence": _DEFAULT_CADENCE,
        }

    @app.get("/stats")
    def stats() -> Dict[str, Any]:
        if _PIPELINE is None:
            raise HTTPException(503, "Pipeline not loaded")
        return {
            "memory_size": len(_MEMORY_STORE),
            "current_cycle": getattr(_PIPELINE, "current_cycle", 0),
            "default_cadence": _DEFAULT_CADENCE,
        }

    @app.post("/query", response_model=QueryOut)
    def query(q: QueryIn) -> QueryOut:
        from caem.production.confidence import render_dict

        if _PIPELINE is None:
            raise HTTPException(503, "Pipeline not loaded")

        t0 = time.perf_counter()
        try:
            result = _PIPELINE.answer(q.question)
        except Exception as exc:
            raise HTTPException(500, f"Pipeline error: {exc}")
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = getattr(result, "display_answer", None) or result.answer or ""
        vout = getattr(result, "verifier_output", None)

        pr: Optional[Dict[str, Any]]
        explanation: Optional[Dict[str, Any]] = None
        if vout is None:
            # Tier 1 / no verifier — synthesize a verified envelope
            pr = {
                "answer": answer,
                "confidence": 1.0,
                "tag": "verified",
                "label": "high",
                "icon": "check",
                "show_answer": True,
                "caveat": None,
                "decision": "STORE",
            }
            explanation = {"source": "memory_hit", "tier": result.tier}
        else:
            cad = q.deferred_cadence or _DEFAULT_CADENCE
            pr = render_dict(
                answer=answer,
                u_stored=float(vout.u_stored or 0.0),
                decision=vout.decision or "DISCARD",
                deferred_eta=q.deferred_eta,
                deferred_cadence=cad,
            )
            explanation = {
                "source": "verification",
                "tier": result.tier,
                "u_stored": float(vout.u_stored or 0.0),
                "p_ground_max": float(getattr(vout, "p_ground_max", 0) or 0),
                "p_ground_mean": float(getattr(vout, "p_ground_mean", 0) or 0),
                "p_ground_atomic": float(getattr(vout, "p_ground_atomic", 0) or 0),
                "p_entail": float(getattr(vout, "p_entail", 0) or 0),
                "u_internal": float(getattr(vout, "u_internal", 0) or 0),
                "u_dropout": float(getattr(vout, "u_dropout", 0) or 0),
                "s_avg": float(getattr(vout, "s_avg", 0) or 0),
                "decision": vout.decision,
            }

        return QueryOut(
            question=q.question,
            production_response=pr,
            tier=result.tier,
            latency_ms=latency_ms,
            stored=getattr(result, "stored", False),
            explanation=explanation,
        )

    return app


def _set_pipeline(pipeline, memory_store, default_cadence: Optional[str]) -> None:
    global _PIPELINE, _MEMORY_STORE, _DEFAULT_CADENCE
    _PIPELINE = pipeline
    _MEMORY_STORE = memory_store
    _DEFAULT_CADENCE = default_cadence


# --------------------------------------------------------------------- #
# Minimal single-page UI (inlined — no separate static/ dir needed)      #
# --------------------------------------------------------------------- #
_INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>CAEM Demo</title>
<style>
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       max-width:780px;margin:40px auto;padding:0 20px;color:#222;background:#fafafa;}
  h1{font-weight:600;margin-bottom:4px;}
  .sub{color:#888;margin-bottom:24px;font-size:0.9em;}
  .q{width:100%;padding:12px;font-size:1.05em;border:1px solid #ccc;border-radius:6px;}
  button{margin-top:8px;padding:8px 18px;font-size:1em;border-radius:6px;
         border:1px solid #4a6fd0;background:#4a6fd0;color:#fff;cursor:pointer;}
  button:disabled{background:#aaa;border-color:#aaa;cursor:not-allowed;}
  .resp{margin-top:24px;padding:16px;border-radius:8px;border:1px solid;}
  .verified{background:#e8f5e9;border-color:#2e7d32;}
  .provisional{background:#fff8e1;border-color:#f9a825;}
  .conflicting{background:#ffebee;border-color:#c62828;}
  .insufficient{background:#eceff1;border-color:#546e7a;}
  .tag{font-weight:600;text-transform:uppercase;letter-spacing:0.05em;font-size:0.85em;}
  .pct{float:right;font-variant-numeric:tabular-nums;}
  .answer{margin-top:10px;font-size:1.05em;line-height:1.5;}
  .caveat{margin-top:10px;color:#555;font-style:italic;font-size:0.95em;}
  .explain{margin-top:14px;padding-top:10px;border-top:1px dashed #bbb;
           font-size:0.85em;color:#666;font-family:monospace;}
  .explain span{margin-right:12px;}
  .hint{color:#888;margin-top:10px;font-size:0.85em;}
  #loader{display:none;margin-top:20px;color:#888;}
</style>
</head><body>
<h1>CAEM</h1>
<div class="sub">Confidence-Aware Episodic Memory — live demo</div>
<textarea id="q" class="q" rows="3" placeholder="Ask anything..."></textarea><br>
<button id="ask">Ask</button>
<span class="hint">Ctrl+Enter to submit</span>
<div id="loader">Thinking...</div>
<div id="out"></div>
<script>
const qbox = document.getElementById('q');
const btn = document.getElementById('ask');
const out = document.getElementById('out');
const loader = document.getElementById('loader');
const ICONS = {check:'✓', tilde:'~', warning:'⚠', cross:'✗'};
async function ask() {
  const question = qbox.value.trim();
  if (!question) return;
  btn.disabled = true; loader.style.display = 'block'; out.innerHTML = '';
  try {
    const r = await fetch('/query', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question})
    });
    if (!r.ok) { out.innerHTML = '<div class="resp conflicting">Error '+r.status+'</div>'; return; }
    const d = await r.json();
    const pr = d.production_response;
    if (!pr) { out.innerHTML = '<div class="resp insufficient">No response produced.</div>'; return; }
    const pct = Math.round(100 * pr.confidence);
    const icon = ICONS[pr.icon] || '·';
    const shown = pr.show_answer ? pr.answer : "I don't have enough evidence to answer this reliably.";
    let html = `<div class="resp ${pr.tag}">`;
    html += `<div><span class="tag">${icon} ${pr.tag} · ${pr.label.replace('_',' ')}</span><span class="pct">${pct}%</span></div>`;
    html += `<div class="answer">${escapeHtml(shown)}</div>`;
    if (pr.caveat) html += `<div class="caveat">${escapeHtml(pr.caveat)}</div>`;
    if (d.explanation) {
      html += '<div class="explain">';
      html += `<span>tier=${d.tier}</span><span>latency=${d.latency_ms.toFixed(0)}ms</span>`;
      for (const [k,v] of Object.entries(d.explanation)) {
        if (typeof v === 'number') html += `<span>${k}=${v.toFixed(3)}</span>`;
      }
      html += '</div>';
    }
    html += '</div>';
    out.innerHTML = html;
  } catch(e) {
    out.innerHTML = '<div class="resp conflicting">Network error: ' + e + '</div>';
  } finally {
    btn.disabled = false; loader.style.display = 'none';
  }
}
function escapeHtml(s){return (s||'').replace(/[<>&]/g, c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));}
btn.onclick = ask;
qbox.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask();
});
qbox.focus();
</script>
</body></html>
"""


# --------------------------------------------------------------------- #
# CLI entry                                                              #
# --------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--memory", type=Path, required=True,
                   help="Memory store path (without .faiss/.meta suffix).")
    p.add_argument("--passage_index", type=Path,
                   default=Path("data/passage_index"))
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cadence", type=str, default=None,
                   help="Default deferred cadence (nightly/hourly/continuous/shortly).")
    ns = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    pipeline, memory_store = _build_pipeline(
        memory_path=ns.memory,
        passage_index=ns.passage_index,
        model=ns.model,
        device=ns.device,
    )
    _set_pipeline(pipeline, memory_store, ns.cadence)

    import uvicorn
    app = make_app()
    uvicorn.run(app, host=ns.host, port=ns.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
