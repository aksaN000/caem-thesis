#!/usr/bin/env python
"""
scripts/rescore_baselines_through_verifier.py
==============================================
Rescore B1-B7 baseline outputs through the locked CAEM verifier (cycle-10
calibration) so the composite hallucination metric (CHM) and its 8-subtype
decomposition are computable on every baseline at matched instrument.

Why this exists
---------------
``eval/baselines.py`` writes ``verifier_output=None`` and ``u_stored=None``
on every ``PipelineResult`` because the baselines do not own a verifier.
The CHM defined in ``eval/metrics.py:hallucination_subtypes`` requires the
nine verifier signals (u_token, u_dropout, u_internal, s_avg, h_norm,
p_entail, p_ground_max, p_ground_mean, p_ground_atomic, p_contra) plus
the 4-outcome decision class to compute confident_confabulation,
factual_fabrication, logical_fabrication, off_topic, defensive_evasion,
template_leak, false_refusal, and over_long subtype rates. Without these,
the cross-system hallucination contrast in Ch5 §sec:comp-baselines is
unsupported on the registered axis.

This script applies the locked cycle-10 ``UnifiedVerifier`` to every
baseline (question, prediction) pair, populates the 9 signals + decision
on each row, and recomputes the 8 subtype rates plus the CHM headline.
The same instrument is applied to every baseline and to CAEM, isolating
the generator-side contribution from the verifier-side contribution. This
is the methodologically standard approach for cross-system hallucination
comparison in the FActScore / MiniCheck-style benchmark literature.

Inputs
------
Per (baseline, benchmark): outputs/baselines/<baseline>/<benchmark>_cycle0.json
  Schema (per row): id, question, prediction, em, f1, gold_answers, tier,
  latency_ms, escalated. Verifier signals + u_stored + decision are None
  on baseline outputs by construction.

Locked cycle-10 calibration:
  --composite_calibration  outputs/production/composite_calibration.json
  --checkpoint             outputs/production/cycle_0/model.pt
                           (the SIL-fine-tuned generator at cycle-10 is
                            staged at the production-cycle-0 path per the
                            Phase B swap convention)

Phase 1c (2026-05-09): the conformal storage gate was replaced by a fixed
threshold on u_stored from CAEMConfig (store_threshold, defer_threshold).
No --conformal_gate argument is needed; the verifier reads the thresholds
from the config defaults.

Outputs
-------
outputs/baselines/<baseline>/<benchmark>_cycle0_with_chm.json
  Per-row: every field from the input + the 9 verifier signals +
  early_exit_triggered + decision + abstained + new u_stored.
  Aggregate: chm + 8 subtype rates + union rate + n.

outputs/baselines/chm_comparison.json
  Cross-baseline summary keyed by (baseline, benchmark) with chm,
  union_rate, per_subtype_rates, n. Plus a CAEM cycle-10 row read off
  outputs/full_run/eval/<benchmark>_cycle10.json so the registered
  "CAEM vs baselines on hallucination rate" contrast is a single table.

Usage
-----
GPU run (post-cycle-10 close, after baselines complete):

    python -m scripts.rescore_baselines_through_verifier \\
        --baselines zero_shot cot rag cot_rag fiveshot_cot vanilla_ft ewc_ft \\
        --benchmarks fever triviaqa commonsense_qa truthfulqa strategyqa \\
        --baselines_dir outputs/baselines \\
        --output_dir outputs/baselines \\
        --composite_calibration outputs/production/composite_calibration.json \\
        --checkpoint outputs/production/cycle_0/model.pt \\
        --device cuda \\
        --caem_eval_dir outputs/full_run/eval \\
        --caem_cycle 10

Dry-run (CPU, no GPU; validates the CHM-aggregation path on existing CAEM
eval JSONs, which already have all 9 signals and decisions populated):

    python -m scripts.rescore_baselines_through_verifier \\
        --dry_run \\
        --dry_run_eval outputs/full_run/eval/fever_cycle3.json

The dry-run path bypasses the verifier construction (no GPU required) and
exercises only the per-row CHM computation + aggregation logic. It is the
CPU-host smoke test before the full GPU rescoring run.

Estimated GPU cost
------------------
~3500 (verifier calls per baseline) × 7 (baselines) × ~2 s/sample on a
5090 with batched verify_batch ≈ 14 GPU-hours. Allowing for ASQA's longer
hypotheses (which dispatch to the longer-pass under MiniCheck), budget
~25-35 GPU-hours and ~$15-20 of Vast credit total.

Reference
---------
- Ch5 §sec:setup-metrics — CHM definition + 8-axis taxonomy
- Ch5 §sec:comp-baselines — registered cross-system contrast
- ``eval/metrics.py:composite_hallucination_metric`` — canonical CHM
- ``caem/verification/verifier.py:UnifiedVerifier.verify_batch`` — verifier
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rescore_baselines")


# ----------------------------------------------------------------------- #
# Verifier construction (GPU path)                                         #
# ----------------------------------------------------------------------- #

def _build_verifier(
    composite_calibration: Path,
    checkpoint: Optional[Path],
    passage_index: Path,
    device: str,
) -> Tuple[Any, Any]:
    """Construct a UnifiedVerifier with cycle-10 locked calibration.

    Returns (verifier, encoder) where encoder is the QueryEncoder used to
    tokenize queries. Mirrors the construction in
    ``scripts/caem_demo_server.py:_build_pipeline`` so the rescored
    instrument matches the deployed verifier exactly.

    Phase 1c (2026-05-09): the conformal storage gate was removed; the
    verifier reads tau_store, tau_defer, abstain_pground_ceiling from
    CAEMConfig directly. Only the composite calibration JSON path is
    overridden here.
    """
    import torch

    from caem.config import CAEMConfig
    from caem.memory.encoder import QueryEncoder
    from caem.model_loader import load_base_generator
    from caem.retrieval.rag import PassageStore
    from caem.verification import UnifiedVerifier, load_verifier_judge

    config = CAEMConfig()
    config.composite_calibration_path = str(composite_calibration)
    logger.info("Composite calibration: %s", composite_calibration)

    t0 = time.perf_counter()
    gen_model, tokenizer = load_base_generator(
        config.base_model_name, use_sdpa=True, use_torch_compile=True,
    )
    logger.info("Base generator loaded in %.1fs", time.perf_counter() - t0)

    # 2026-05-16: support both legacy full-state-dict checkpoints AND v2.1
    # LoRA adapter directories. CAEM v2.1 stores per-cycle weights as PEFT
    # LoRA adapters under outputs/full_run/cycle_<N>/adapter/ rather than
    # the v1 "production/cycle_0/model.pt" single-file checkpoint.
    if checkpoint is not None and str(checkpoint) not in {"", "none", "None"}:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"--checkpoint {ckpt_path} does not exist. Pass an empty "
                f"--checkpoint '' to use base Qwen weights (with the matched-"
                f"protocol warning), or point at a real adapter directory / "
                f"state-dict checkpoint."
            )
        t1 = time.perf_counter()
        if ckpt_path.is_dir() and (ckpt_path / "adapter_config.json").is_file():
            # PEFT LoRA adapter directory — load via peft.PeftModel
            from peft import PeftModel
            gen_model = PeftModel.from_pretrained(gen_model, str(ckpt_path))
            logger.info("Loaded LoRA adapter from %s in %.1fs",
                        ckpt_path, time.perf_counter() - t1)
        else:
            # Legacy full-state-dict checkpoint (.pt file)
            state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            gen_model.load_state_dict(state)
            logger.info("Loaded SIL checkpoint %s in %.1fs",
                        ckpt_path, time.perf_counter() - t1)
    else:
        logger.warning(
            "No --checkpoint provided. Verifier will use base Qwen weights, "
            "NOT the cycle-N SIL-fine-tuned adapter. Composite signals from "
            "the locked cycle-3 composite_calibration.json still apply, but "
            "generator-dependent signals (u_token, u_dropout, u_internal) "
            "will be computed against base Qwen logits rather than C3."
        )

    encoder = QueryEncoder(device=device)

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

    passage_store = PassageStore.load(str(passage_index))

    def _retrieve(q: str, k: int) -> List[str]:
        return passage_store.search(q, k)

    verifier = UnifiedVerifier(
        model=gen_model,
        tokenizer=tokenizer,
        sbert_encoder=encoder,
        judge=judge,
        nli_model=nli_model,
        nli_tokenizer=nli_tokenizer,
        reranker=cross_encoder,
        passage_retriever=_retrieve,
        qa_relevance_scorer=cross_encoder,
        config=config,
        device=device,
    )
    return verifier, encoder


# ----------------------------------------------------------------------- #
# Row-level rescoring                                                      #
# ----------------------------------------------------------------------- #

_VERIFIER_FIELDS: Tuple[str, ...] = (
    "u_token", "u_dropout", "u_internal",
    "s_avg", "h_norm", "p_entail",
    "p_ground_max", "p_ground_mean", "p_ground_atomic", "p_contra",
    "u_stored", "decision", "early_exit_triggered", "abstained",
    "q_a_relevance",
    # v2 Fix 6 + Fix 7: alias_overlap and entity_head_consistency.
    # UnifiedVerifierOutput defaults both to 0.5 when missing, so
    # baseline rows produced before these fields existed still
    # serialise cleanly with the neutral prior.
    "alias_overlap",
    "entity_head_consistency",
)


def _vout_to_dict(vout: Any) -> Dict[str, Any]:
    """Project a UnifiedVerifierOutput onto the per-row signal dict."""
    return {
        "u_token":              float(getattr(vout, "u_token", 0.5)),
        "u_dropout":            float(getattr(vout, "u_dropout", 0.5)),
        "u_internal":           float(getattr(vout, "u_internal", 0.5)),
        "s_avg":                float(getattr(vout, "s_avg", 0.5)),
        "h_norm":               float(getattr(vout, "h_norm", 0.5)),
        "p_entail":             float(getattr(vout, "p_entail", 0.5)),
        "p_ground_max":         float(getattr(vout, "p_ground_max", 0.5)),
        "p_ground_mean":        float(getattr(vout, "p_ground_mean", 0.5)),
        "p_ground_atomic":      float(getattr(vout, "p_ground_atomic", 0.5)),
        "p_contra":             float(getattr(vout, "p_contra", 0.0)),
        "u_stored":             float(getattr(vout, "u_stored", 0.0)),
        "decision":             str(getattr(vout, "decision", "DISCARD")),
        "early_exit_triggered": bool(getattr(vout, "early_exit_triggered", False)),
        "abstained":            bool(getattr(vout, "abstained", False)),
        "q_a_relevance":        float(getattr(vout, "q_a_relevance", 0.5)),
    }


def _rescore_rows(verifier: Any, rows: Sequence[Dict[str, Any]],
                  batch_size: int = 16) -> List[Dict[str, Any]]:
    """Run verifier.verify_batch over (question, prediction) pairs in
    chunks and annotate each row with the 9 signals + decision. The
    verifier owns its own retrieval + rerank under the locked
    configuration. Rows missing question or prediction are passed
    through with DISCARD decision and None-valued signals.
    """
    triples: List[Tuple[int, str, str]] = []
    for i, r in enumerate(rows):
        q = r.get("question") or ""
        p = r.get("prediction") or r.get("display_answer") or r.get("answer") or ""
        if q and p:
            triples.append((i, q, p))

    vout_by_idx: Dict[int, Any] = {}
    for start in range(0, len(triples), batch_size):
        chunk = triples[start:start + batch_size]
        pairs = [(q, p) for (_, q, p) in chunk]
        t0 = time.perf_counter()
        vouts = verifier.verify_batch(pairs)
        dt = time.perf_counter() - t0
        logger.info("verify_batch [%d-%d / %d] in %.1fs (%.0f ms/sample)",
                    start, start + len(chunk), len(triples), dt,
                    dt / max(len(chunk), 1) * 1000.0)
        for (idx, _, _), vout in zip(chunk, vouts):
            vout_by_idx[idx] = vout

    final: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        new_r = dict(r)
        if i in vout_by_idx:
            new_r.update(_vout_to_dict(vout_by_idx[i]))
        else:
            new_r.update({
                "u_token": None, "u_dropout": None, "u_internal": None,
                "s_avg": None, "h_norm": None, "p_entail": None,
                "p_ground_max": None, "p_ground_mean": None,
                "p_ground_atomic": None, "p_contra": None,
                "u_stored": None, "q_a_relevance": None,
                "decision": "DISCARD",
                "early_exit_triggered": False,
                "abstained": False,
            })
        final.append(new_r)
    return final


# ----------------------------------------------------------------------- #
# CHM aggregation (CPU-only, used by both GPU and dry-run paths)          #
# ----------------------------------------------------------------------- #

def _aggregate_chm(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute CHM + 8 subtype rates + union rate over a row list. Uses
    the canonical ``eval.metrics.composite_hallucination_metric``.

    Loads ``eval/metrics.py`` directly via importlib to avoid pulling in
    ``eval/__init__.py``, which transitively imports ``eval/baselines.py``
    and therefore ``torch``. The dry-run path runs on CPU hosts where
    torch may not be installed.
    """
    import importlib.util
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "eval_metrics_isolated", repo_root / "eval" / "metrics.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    composite_hallucination_metric = mod.composite_hallucination_metric

    chm = composite_hallucination_metric(rows)
    return {
        "n":                    len(rows),
        "chm":                  chm["chm"],
        "union_rate":           chm["union_rate"],
        "n_measured_subtypes":  chm["n_measured_subtypes"],
        "per_subtype_rates":    chm["per_subtype_rates"],
        "weights":              chm["weights"],
    }


# ----------------------------------------------------------------------- #
# Per-(baseline, benchmark) rescore + write                                #
# ----------------------------------------------------------------------- #

def _resolve_src_paths(
    baselines_dir: Path,
    baseline: str,
    bench: str,
    *,
    training_baselines: set,
    training_cycles: List[int],
) -> List[Tuple[int, Path]]:
    """Return [(cycle, src_path), ...] for a given (baseline, bench).

    Inference baselines: a single (0, <baseline>/<bench>_cycle0.json) pair.
    Training baselines: one pair per cycle in ``training_cycles``, e.g.
    [(3, <baseline>/eval/<bench>_cycle3.json), (5, ...)].

    Missing files are silently dropped from the returned list — the caller
    logs the resulting empty list at info level.
    """
    out: List[Tuple[int, Path]] = []
    if baseline in training_baselines:
        for cyc in training_cycles:
            p = baselines_dir / baseline / "eval" / f"{bench}_cycle{cyc}.json"
            if p.exists():
                out.append((cyc, p))
        return out

    # Inference layout — single cycle 0.
    p = baselines_dir / baseline / f"{bench}_cycle0.json"
    if p.exists():
        out.append((0, p))
    return out


def _rescore_one(
    verifier: Any,
    src_path: Path,
    dst_path: Path,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Read one baseline JSON, rescore through the verifier, write sidecar
    JSON with the 9 signals + decision + CHM aggregate, return summary.
    """
    with open(src_path) as f:
        doc = json.load(f)
    rows = doc.get("samples") or doc.get("results") or []
    if not rows:
        logger.warning("No samples in %s", src_path)
        return {"n": 0, "chm": float("nan"), "per_subtype_rates": {}}

    rescored = _rescore_rows(verifier, rows, batch_size=batch_size)
    summary = _aggregate_chm(rescored)

    out_doc = dict(doc)
    out_doc["samples"] = rescored
    out_doc["_rescored"] = {
        "instrument":  "cycle-10 locked CAEM verifier",
        "script":      "scripts/rescore_baselines_through_verifier.py",
        "chm":         summary["chm"],
        "union_rate":  summary["union_rate"],
        "n":           summary["n"],
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(out_doc, f)
    logger.info("Wrote rescored %s (n=%d, chm=%.4f)",
                dst_path, summary["n"], summary["chm"])
    return summary


# ----------------------------------------------------------------------- #
# Dry-run smoke-test                                                       #
# ----------------------------------------------------------------------- #

def _dry_run(eval_json: Path) -> int:
    """CPU smoke-test: load an existing CAEM eval JSON (which already
    has all 9 signals + decision populated by the live harness) and run
    only the CHM aggregation path. Validates the metric pipeline before
    the GPU rescore run consumes credit on Vast.
    """
    if not eval_json.exists():
        logger.error("Dry-run input %s does not exist", eval_json)
        return 1
    with open(eval_json) as f:
        doc = json.load(f)
    rows = doc.get("samples") or doc.get("results") or []
    if not rows:
        logger.error("No samples in %s", eval_json)
        return 1

    n_with_signals = sum(
        1 for r in rows
        if r.get("u_stored") is not None and r.get("decision") is not None
    )
    logger.info("Dry-run: %s — %d rows, %d with signals",
                eval_json.name, len(rows), n_with_signals)

    summary = _aggregate_chm(rows)
    logger.info("Aggregate CHM (n=%d):", summary["n"])
    logger.info("  chm        = %.4f", summary["chm"])
    logger.info("  union_rate = %.4f", summary["union_rate"])
    logger.info("  per-subtype:")
    for k, v in sorted(summary["per_subtype_rates"].items()):
        logger.info("    %-30s = %.4f", k, v)
    return 0


# ----------------------------------------------------------------------- #
# CLI                                                                      #
# ----------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry_run", action="store_true",
                   help="CPU-only smoke test: aggregate CHM on an existing "
                        "CAEM eval JSON. Skips verifier construction.")
    p.add_argument("--dry_run_eval", type=Path,
                   default=Path("outputs/full_run/eval/fever_cycle3.json"),
                   help="Eval JSON to aggregate under --dry_run.")
    p.add_argument("--baselines", nargs="+",
                   default=["zero_shot", "cot", "rag", "cot_rag",
                            "fiveshot_cot", "vanilla_ft", "ewc_ft"],
                   help="Baseline names matching subdirectories of "
                        "<baselines_dir>.")
    # 2026-05-16: training baselines (B6 vanilla_ft, B7 ewc_only_ft) write
    # under <baseline>/eval/<bench>_cycle{N}.json with N>=1 per cycle, not
    # the inference layout <baseline>/<bench>_cycle0.json. The path
    # resolver tries inference layout first, then falls back to training
    # layout. --training_cycles selects which cycles get rescored for
    # training baselines (default 3, 5: matched-protocol headline +
    # retention-guard-ablation receipt).
    p.add_argument("--training_cycles", nargs="+", type=int,
                   default=[3, 5],
                   help="Cycles to rescore for training baselines "
                        "(B6/B7). C3 = matched-protocol headline "
                        "(CAEM's last successful SIL cycle), "
                        "C5 = retention-guard-ablation receipt.")
    p.add_argument("--training_baselines", nargs="+",
                   default=["vanilla_ft", "ewc_only_ft", "ewc_ft"],
                   help="Baseline names that use the training-baseline "
                        "directory layout (predictions under <baseline>/"
                        "eval/<bench>_cycle{N}.json).")
    # v2 Fix 9b: argparse default tracks caem.config so the rescoring
    # CSV columns line up with the live v2 panel
    # (was hardcoded to v1 7-bench list including arc_challenge + asqa).
    from caem.config import (
        TRAINING_BENCHMARKS as _CFG_TRAINING_BENCHMARKS,
        TRANSFER_BENCHMARKS as _CFG_TRANSFER_BENCHMARKS,
    )
    p.add_argument("--benchmarks", nargs="+",
                   default=list(_CFG_TRAINING_BENCHMARKS) + list(_CFG_TRANSFER_BENCHMARKS))
    p.add_argument("--baselines_dir", type=Path,
                   default=Path("outputs/baselines"),
                   help="Root directory holding <baseline>/<bench>_cycle0.json.")
    p.add_argument("--output_dir", type=Path,
                   default=Path("outputs/baselines"),
                   help="Sidecar JSONs land at "
                        "<output_dir>/<baseline>/<bench>_cycle{N}_with_chm.json "
                        "for inference baselines (N=0) and "
                        "<output_dir>/<baseline>/eval/<bench>_cycle{N}_with_chm.json "
                        "for training baselines (N in --training_cycles).")
    p.add_argument("--composite_calibration", type=Path,
                   default=Path("outputs/full_run/cycle_3/composite_calibration.json"),
                   help=("Composite-version pin. The composite is refit at "
                         "every cycle boundary; matched-protocol cross-system "
                         "evaluation requires the baselines score under the "
                         "SAME composite that CAEM C3 was scored against. "
                         "Default 2026-05-15: cycle-3 composite. Override "
                         "explicitly to compare against a different cycle's "
                         "composite (e.g. for cycle-by-cycle methodology "
                         "diagnostics). The pre-Phase-1d "
                         "outputs/production/composite_calibration.json path "
                         "is RETIRED — that composite predates the Phase-1d "
                         "patches and would produce comparison artifacts."))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("outputs/full_run/cycle_3/adapter"),
                   help="SIL fine-tuned model checkpoint (cycle-10 weights "
                        "staged at production-cycle-0 path per Phase B swap).")
    p.add_argument("--passage_index", type=Path,
                   default=Path("data/passage_index"),
                   help="Pre-built dense passage index for verifier retrieval.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--caem_eval_dir", type=Path,
                   default=Path("outputs/full_run/eval"),
                   help="Directory holding CAEM cycle-N eval JSONs for the "
                        "side-by-side row in the comparison table.")
    p.add_argument("--caem_cycle", type=int, default=10,
                   help="Cycle index of CAEM eval JSONs to include as the "
                        "CAEM row in the comparison table.")
    return p.parse_args()


def main() -> int:
    ns = _parse_args()

    if ns.dry_run:
        return _dry_run(ns.dry_run_eval)

    # GPU path: rescore every (baseline, benchmark) pair.
    if not ns.composite_calibration.exists():
        logger.error("Required calibration artefact not found: %s", ns.composite_calibration)
        return 1
    if not ns.passage_index.exists() and not ns.passage_index.with_suffix(".faiss").exists():
        logger.error("Passage index not found at %s", ns.passage_index)
        return 1

    verifier, _ = _build_verifier(
        composite_calibration=ns.composite_calibration,
        checkpoint=ns.checkpoint,
        passage_index=ns.passage_index,
        device=ns.device,
    )

    # comparison key for training baselines is "<name>@c<cycle>" so the
    # downstream comparison table can distinguish B6@C3 (matched-protocol
    # headline) from B6@C5 (retention-guard-ablation receipt).
    comparison: Dict[str, Dict[str, Any]] = {}
    training_baselines = set(ns.training_baselines)

    for baseline in ns.baselines:
        for bench in ns.benchmarks:
            pairs = _resolve_src_paths(
                ns.baselines_dir, baseline, bench,
                training_baselines=training_baselines,
                training_cycles=ns.training_cycles,
            )
            if not pairs:
                logger.info("Skip %s/%s: no source JSONs found (inference "
                            "layout <baseline>/<bench>_cycle0.json or "
                            "training layout <baseline>/eval/<bench>_cycle"
                            "{%s}.json)",
                            baseline, bench,
                            ",".join(str(c) for c in ns.training_cycles))
                continue
            for cyc, src in pairs:
                # Preserve the source layout in the output path so
                # downstream readers can find the rescored sidecar next
                # to the original prediction JSON.
                if baseline in training_baselines:
                    dst = (ns.output_dir / baseline / "eval"
                           / f"{bench}_cycle{cyc}_with_chm.json")
                    key = f"{baseline}@c{cyc}"
                else:
                    dst = (ns.output_dir / baseline
                           / f"{bench}_cycle{cyc}_with_chm.json")
                    key = baseline
                summary = _rescore_one(verifier, src, dst,
                                       batch_size=ns.batch_size)
                comparison.setdefault(key, {})[bench] = summary

    # Add the CAEM cycle-N row by aggregating the live eval JSONs that
    # already carry signals + decision (no rescoring needed).
    if ns.caem_eval_dir.exists():
        caem_row: Dict[str, Any] = {}
        for bench in ns.benchmarks:
            caem_path = ns.caem_eval_dir / f"{bench}_cycle{ns.caem_cycle}.json"
            if not caem_path.exists():
                logger.warning("CAEM cycle-%d eval missing for %s",
                               ns.caem_cycle, bench)
                continue
            with open(caem_path) as f:
                doc = json.load(f)
            rows = doc.get("samples") or doc.get("results") or []
            caem_row[bench] = _aggregate_chm(rows)
        comparison[f"caem_cycle{ns.caem_cycle}"] = caem_row

    out_path = ns.output_dir / "chm_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Wrote %s", out_path)

    # Print the comparison table.
    print()
    print("=" * 78)
    print("CHM cross-baseline comparison (locked cycle-10 verifier)")
    print("=" * 78)
    header = f"{'baseline':<22}" + "".join(f"{b[:10]:>11}" for b in ns.benchmarks)
    print(header)
    print("-" * 78)
    for system in list(comparison.keys()):
        row = f"{system:<22}"
        for bench in ns.benchmarks:
            v = comparison[system].get(bench, {})
            row += f"{(v.get('chm') if v.get('chm') is not None else float('nan')):>11.4f}"
        print(row)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
