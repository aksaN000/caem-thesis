#!/usr/bin/env python
"""
scripts/caem_chat.py
====================
Interactive CLI chat — runs the full CAEM pipeline and renders each
response through the Ship 1 production envelope.

This is the ollama-equivalent local REPL. Point it at a trained memory
store (cold_start_memory or outputs/full_run/cycle_N) and it serves the
same pipeline used by the thesis experiments, but with user-facing
confidence rendering.

Usage
-----
    # Cold-start memory (pre-Step-7-main):
    python scripts/caem_chat.py \\
        --memory outputs/cold_start_memory/memory_store \\
        --passage_index data/passage_index

    # Post-Step-7-main (trained cycle_10 memory):
    python scripts/caem_chat.py \\
        --memory outputs/full_run/cycle_10/memory_store_cycle_10 \\
        --passage_index data/passage_index

Controls
--------
    <type your question> Enter      query the pipeline
    /quit | /exit | Ctrl-D          exit
    /stats                          memory + model stats
    /cadence nightly|hourly|...     set deferred-caveat cadence
    /help                           show this

The CLI loads Qwen-3B + MiniCheck + BGE + SBERT (all bf16 after
commit 8926117). Needs ~10-12 GB VRAM. Cold-start takes ~60-120s.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ANSI colors for tag rendering (falls back to plain on non-TTY)
_USE_COLOR = sys.stdout.isatty()
_COLORS = {
    "verified":     "\033[1;32m",  # bold green
    "provisional":  "\033[1;33m",  # bold yellow
    "conflicting":  "\033[1;31m",  # bold red
    "insufficient": "\033[1;90m",  # bold grey
    "reset":        "\033[0m",
}
_ICONS = {"check": "✓", "tilde": "~", "warning": "⚠", "cross": "✗"}


def _c(text: str, color: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--memory", type=Path, required=True,
                   help="Path to memory_store (without .faiss/.meta suffix).")
    p.add_argument("--passage_index", type=Path,
                   default=Path("data/passage_index"),
                   help="FAISS passage index directory.")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                   help="Generator model name.")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda or cpu.")
    p.add_argument("--cadence", type=str, default=None,
                   help="Deferred-caveat cadence hint "
                        "(nightly/hourly/continuous/shortly).")
    return p.parse_args()


def _build_pipeline(ns: argparse.Namespace):
    """Construct the full CAEM pipeline. Mirrors scripts/run_experiment.py's
    build_pipeline() but skipped thresholds/cold-start defaults since we
    just need a live-serving config."""
    import torch

    from caem.config import CAEMConfig
    from caem.memory.encoder import QueryEncoder
    from caem.memory.store import EpisodicMemoryStore
    from caem.model_loader import load_base_generator
    from caem.pipeline import CAEMPipeline
    from caem.retrieval.rag import PassageStore
    from caem.verification import load_verifier_judge

    print(_c("→ Loading config...", "provisional"))
    config = CAEMConfig()

    print(_c("→ Loading Qwen generator (bf16)...", "provisional"))
    t0 = time.perf_counter()
    model, tokenizer = load_base_generator(
        ns.model, use_sdpa=True, use_torch_compile=True,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    print(_c("→ Loading SBERT encoder (bf16)...", "provisional"))
    encoder = QueryEncoder(device=ns.device)

    print(_c("→ Loading passage index...", "provisional"))
    passage_store = PassageStore.load(str(ns.passage_index))

    print(_c("→ Loading verifier judge (MiniCheck)...", "provisional"))
    judge, nli_model, nli_tokenizer = load_verifier_judge(
        config, ns.device, allow_fallback=True,
    )

    cross_encoder = None
    if config.cross_encoder_model:
        print(_c("→ Loading BGE cross-encoder (bf16)...", "provisional"))
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(
            config.cross_encoder_model,
            device=ns.device,
            automodel_args={"torch_dtype": torch.bfloat16}
            if ns.device != "cpu" else {},
        )

    print(_c("→ Loading episodic memory...", "provisional"))
    memory_store = EpisodicMemoryStore(config=config)
    memory_store.load(str(ns.memory))
    print(f"  memory: {len(memory_store)} episodes")

    print(_c("→ Constructing pipeline...", "provisional"))
    pipeline = CAEMPipeline(
        model=model, tokenizer=tokenizer, encoder=encoder,
        judge=judge, nli_model=nli_model, nli_tokenizer=nli_tokenizer,
        passage_store=passage_store, memory_store=memory_store,
        cross_encoder=cross_encoder, config=config,
    )
    print(_c("✓ Ready.\n", "verified"))
    return pipeline, memory_store


def _render_response(result, *, cadence: Optional[str]) -> str:
    """Take a PipelineResult, render via the production envelope."""
    from caem.production.confidence import (
        INSUFFICIENT_EVIDENCE_MESSAGE,
        render,
    )

    answer = getattr(result, "display_answer", None) or result.answer or ""
    vout = getattr(result, "verifier_output", None)
    if vout is None:
        # Tier 1 memory hit — no verifier ran, answer is from a stored episode
        return (
            f"{_c('[' + _ICONS['check'] + ' verified · memory hit · tier ' + str(result.tier) + ']', 'verified')}\n"
            f"{answer}\n"
        )

    u_stored = float(vout.u_stored) if vout.u_stored is not None else 0.0
    decision = vout.decision or "DISCARD"
    resp = render(
        answer, u_stored, decision,
        deferred_cadence=cadence,
    )
    if resp is None:
        return f"[unrecognized decision {decision}] {answer}\n"

    icon = _ICONS.get(resp.icon, "·")
    pct = int(round(100 * resp.confidence))
    header = f"[{icon} {resp.tag} · {pct}% · {resp.label.replace('_', ' ')}]"
    shown = resp.answer if resp.show_answer else INSUFFICIENT_EVIDENCE_MESSAGE
    out = [f"{_c(header, resp.tag)}", shown]
    if resp.caveat:
        out.append(_c(f"  └─ {resp.caveat}", resp.tag))
    return "\n".join(out) + "\n"


def _show_help() -> None:
    print("""
Commands:
  <question>              Ask CAEM anything
  /stats                  Show memory + model stats
  /cadence <name>         Set deferred cadence (nightly/hourly/continuous/shortly/off)
  /help                   Show this help
  /quit | /exit | Ctrl-D  Exit
""")


def main() -> int:
    ns = _parse_args()
    logging.basicConfig(
        level=logging.WARNING,  # quiet pipeline chatter in chat mode
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pipeline, memory_store = _build_pipeline(ns)
    cadence = ns.cadence

    print(f"CAEM ready. Memory: {len(memory_store)} episodes. Type /help for commands.\n")
    while True:
        try:
            line = input(_c("> ", "verified")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            _show_help()
            continue
        if line == "/stats":
            print(f"Memory episodes: {len(memory_store)}")
            print(f"Cadence:         {cadence or '(default / generic)'}")
            continue
        if line.startswith("/cadence"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                new_cad = parts[1].strip()
                cadence = None if new_cad.lower() == "off" else new_cad
                print(f"Cadence set to: {cadence or '(off)'}")
            else:
                print("Usage: /cadence <nightly|hourly|continuous|shortly|off>")
            continue

        try:
            t0 = time.perf_counter()
            result = pipeline.answer(line)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(_c(f"[error] {type(exc).__name__}: {exc}", "conflicting"))
            continue

        print(_render_response(result, cadence=cadence))
        print(_c(f"  (tier {result.tier}, {elapsed:.2f}s)", "insufficient"))
        print()

    print("Bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
