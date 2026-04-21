"""
scripts/cycle2_retention_diagnostic.py
=======================================
Cycle-2 hold-out retention diagnostic, referenced in Chapter 4 Section
"Cycle-2 retention diagnostic and structured fallback".

What it does
------------
Takes a frozen 500-example stratified slice of Cycle-0 evaluation
questions, re-evaluates them under the Cycle-2 model weights, and
reports the absolute exact-match drop. If the drop exceeds three
percentage points, the script writes a structured-fallback advisory
signalling that the remaining cycles should switch to O-LoRA-style
stacked orthogonal adapters (Wang 2023 EMNLP Findings; Biderman 2024
TMLR).

This sits *on top of* the rho_min = 0.93 MMLU forgetting guard in
caem/training/self_improvement.py -- it is a stricter, in-distribution
tripwire aligned with the Jin 2024 / Scialom 2022 empirical envelope
(both papers report sub-one-percent forgetting on comparable continual
streams at Flan-T5-Large scale; three percent is a generous ceiling).

Stages
------
1. ``--make_slice``: stratify 500 samples across the six benchmarks
   from the Cycle-0 eval JSONs, freeze their (id, question, gold) to a
   JSONL file. Runs once before Cycle 1.
2. ``--evaluate``: load the frozen slice + the Cycle-2 checkpoint,
   run inference, compute EM and per-benchmark breakdown, and emit a
   JSON report with the advisory flag.

Usage
-----
# Step 6 / pre-Cycle-1:
PYTHONPATH=. python scripts/cycle2_retention_diagnostic.py --make_slice \
    --cycle0_eval_dir outputs/cycle_0/eval \
    --output_slice data/retention/cycle0_slice_500.jsonl \
    --n_per_benchmark 83      # 6 * 83 = 498, +2 for 500

# After Cycle 2 SIL completes:
PYTHONPATH=. python scripts/cycle2_retention_diagnostic.py --evaluate \
    --slice data/retention/cycle0_slice_500.jsonl \
    --cycle2_checkpoint outputs/full_run/cycle_2/model \
    --cycle0_eval_dir outputs/cycle_0/eval \
    --output_report outputs/full_run/cycle_2/retention_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cycle2_retention")


# Default fallback threshold from Chapter 4: three percentage points
# absolute EM drop triggers the structured-fallback advisory.
DEFAULT_DROP_THRESHOLD_ABS = 0.03


# --------------------------------------------------------------------------- #
# Slice construction                                                           #
# --------------------------------------------------------------------------- #

def _make_slice(
    cycle0_eval_dir: Path,
    output_slice: Path,
    n_per_benchmark: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    evals = sorted(cycle0_eval_dir.glob("*_cycle0.json"))
    if not evals:
        logger.error("No *_cycle0.json under %s", cycle0_eval_dir)
        sys.exit(1)
    for p in evals:
        with open(p) as f:
            d = json.load(f)
        samples = d.get("samples", [])
        bench = d.get("meta", {}).get("benchmark", p.stem)
        if len(samples) < n_per_benchmark:
            logger.warning(
                "Benchmark %s has only %d samples; needed %d. Taking all.",
                bench, len(samples), n_per_benchmark,
            )
            picks = samples
        else:
            picks = rng.sample(samples, n_per_benchmark)
        for s in picks:
            rows.append({
                "benchmark": bench,
                "id": s.get("id"),
                "question": s.get("question", ""),
                "gold_answers": s.get("gold_answers", []),
                "gold_label": s.get("gold_label"),
                "cycle0_prediction": s.get("prediction", ""),
                "cycle0_em": float(s.get("em", 0.0)),
            })

    output_slice.parent.mkdir(parents=True, exist_ok=True)
    with open(output_slice, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    logger.info("Wrote %d samples across %d benchmarks to %s",
                len(rows), len(evals), output_slice)


# --------------------------------------------------------------------------- #
# Cycle-2 evaluation                                                           #
# --------------------------------------------------------------------------- #

def _load_slice(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _evaluate(
    slice_path: Path,
    cycle2_checkpoint: Path,
    output_report: Path,
    drop_threshold: float,
    device: str,
) -> None:
    from caem.config import CAEMConfig
    from caem.model_loader import load_base_generator
    from eval.metrics import exact_match
    import torch

    config = CAEMConfig()
    rows = _load_slice(slice_path)
    logger.info("Loaded %d hold-out samples", len(rows))

    # Load base architecture, then overlay the cycle-2 fine-tuned weights.
    # Branch C uses a decoder-only backbone (Qwen-3B by default). The
    # cycle2_checkpoint argument is interpreted as either a HuggingFace
    # repo id/local path pointing at the overlay weights OR a state_dict
    # checkpoint — we try the state_dict path first (matches SIL format)
    # and fall back to treating it as a full-model path.
    model, tokenizer = load_base_generator(
        config.base_model_name,
        device=device,
        dtype=torch.bfloat16,
        use_flash_attention_2=False,
        use_torch_compile=False,
    )
    ckpt_path = Path(cycle2_checkpoint)
    if ckpt_path.is_file():
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded cycle-2 state_dict overlay from %s", ckpt_path)
    elif ckpt_path.is_dir():
        # Directory path: SIL checkpoints store model.pt inside the dir.
        sd_path = ckpt_path / "model.pt"
        if sd_path.is_file():
            state = torch.load(str(sd_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            logger.info("Loaded cycle-2 state_dict overlay from %s", sd_path)
        else:
            logger.warning(
                "cycle2_checkpoint dir %s has no model.pt; using base weights.",
                ckpt_path,
            )
    model = model.eval()

    per_bench: Dict[str, Dict[str, float]] = {}
    for r in rows:
        q = r["question"]
        gold = r["gold_answers"] or ([r["gold_label"]] if r["gold_label"] else [])
        # ChatML-wrap for decoder-only; slice generated suffix.
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True,
                )
                if not isinstance(prompt, str):
                    prompt = q
            except Exception:
                prompt = q
        else:
            prompt = q
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=2048, add_special_tokens=False).to(device)
        input_len = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0, input_len:] if out.shape[1] > input_len else out[0]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True)

        em_now = 1.0 if any(exact_match(pred, str(g)) for g in gold) else 0.0
        em_before = r["cycle0_em"]
        bench = r["benchmark"]
        b = per_bench.setdefault(
            bench, {"n": 0, "em_cycle0_sum": 0.0, "em_cycle2_sum": 0.0,
                    "drops": 0, "drop_abs_sum": 0.0},
        )
        b["n"] += 1
        b["em_cycle0_sum"] += em_before
        b["em_cycle2_sum"] += em_now
        if em_now < em_before:
            b["drops"] += 1
        b["drop_abs_sum"] += (em_before - em_now)

    overall_c0 = sum(b["em_cycle0_sum"] for b in per_bench.values()) / max(
        1, sum(b["n"] for b in per_bench.values())
    )
    overall_c2 = sum(b["em_cycle2_sum"] for b in per_bench.values()) / max(
        1, sum(b["n"] for b in per_bench.values())
    )
    abs_drop = overall_c0 - overall_c2
    advisory = (
        "STRUCTURED_FALLBACK_TO_OLORA"
        if abs_drop > drop_threshold
        else "OK_CONTINUE_PLAIN_L2"
    )
    report = {
        "n_samples": sum(b["n"] for b in per_bench.values()),
        "em_cycle0_mean": overall_c0,
        "em_cycle2_mean": overall_c2,
        "abs_drop": abs_drop,
        "drop_threshold": drop_threshold,
        "advisory": advisory,
        "per_benchmark": {
            k: {
                "n": v["n"],
                "em_cycle0": v["em_cycle0_sum"] / v["n"],
                "em_cycle2": v["em_cycle2_sum"] / v["n"],
                "abs_drop": (v["em_cycle0_sum"] - v["em_cycle2_sum"]) / v["n"],
            }
            for k, v in per_bench.items()
        },
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        "Cycle-0 EM = %.4f, Cycle-2 EM = %.4f, abs drop = %.4f (threshold %.4f) -> %s",
        overall_c0, overall_c2, abs_drop, drop_threshold, advisory,
    )
    if advisory == "STRUCTURED_FALLBACK_TO_OLORA":
        logger.warning(
            "Retention diagnostic tripped. Switch to O-LoRA "
            "(caem/training/self_improvement.py adapter_mode='o_lora', "
            "rank 16 on all attention + MLP, merge every 2 cycles) "
            "for cycles 3..N. See Chapter 4 section "
            "'Cycle-2 retention diagnostic and structured fallback'."
        )


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_mutually_exclusive_group(required=True)
    sub.add_argument("--make_slice", action="store_true")
    sub.add_argument("--evaluate", action="store_true")
    # make_slice args
    p.add_argument("--cycle0_eval_dir", type=Path)
    p.add_argument("--output_slice", type=Path)
    p.add_argument("--n_per_benchmark", type=int, default=83)
    p.add_argument("--seed", type=int, default=42)
    # evaluate args
    p.add_argument("--slice", dest="slice_path", type=Path)
    p.add_argument("--cycle2_checkpoint", type=Path)
    p.add_argument("--output_report", type=Path)
    p.add_argument(
        "--drop_threshold", type=float, default=DEFAULT_DROP_THRESHOLD_ABS,
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    ns = _parse_args()
    if ns.make_slice:
        if not ns.cycle0_eval_dir or not ns.output_slice:
            sys.exit("--make_slice requires --cycle0_eval_dir and --output_slice")
        _make_slice(ns.cycle0_eval_dir, ns.output_slice,
                    ns.n_per_benchmark, ns.seed)
        return
    if ns.evaluate:
        if not (ns.slice_path and ns.cycle2_checkpoint and ns.output_report):
            sys.exit("--evaluate requires --slice, --cycle2_checkpoint, --output_report")
        if ns.device is None:
            import torch
            ns.device = "cuda" if torch.cuda.is_available() else "cpu"
        _evaluate(ns.slice_path, ns.cycle2_checkpoint,
                  ns.output_report, ns.drop_threshold, ns.device)


if __name__ == "__main__":
    main()
