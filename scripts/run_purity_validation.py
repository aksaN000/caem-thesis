"""
scripts/run_purity_validation.py
==================================
CAEM Theory Validation — Three Protocols (Chapter 5, Section 5.5)

Validates all three theoretical claims in the thesis:

  Theory 1 — Data Purity Theorem
    Claim:   P = pα / (pα + (1-p)(1-α))   AND   P > p
    Condition: p > (1-α)  must hold for the theorem to guarantee P > p
    CRITICAL: theorem guarantees P > p (base accuracy), NOT P > α.
    Protocol: measure p and α per cycle on the purity validation set;
              compute P_theory; compare with P_obs (actual memory accuracy).

  Theory 2 — Coupled Improvement Recurrence (Monotonicity)
    Claim:   p_0 < p_1 < p_2 < p_3  AND  α_0 < α_1 < α_2 < α_3
    Protocol: report p and α per cycle; confirm strict monotonicity.

  Theory 3 — Convergence
    Claim:   Δ(2→3) < Δ(1→2)   where Δ(k→k+1) = p_{k+1} - p_k
    Protocol: compute Δ per cycle pair; confirm diminishing improvement.

All three validations use the purity validation set (500 samples, separate
from both the calibration set and the eval set).

Thesis reference
----------------
  §4.8  Theoretical analysis (purity theorem + convergence)
  §5.5  Theory validation (three protocols, one table each)
  §6.1  Conclusion: theoretical grounding distinguishes CAEM from heuristics
  benchmarks-and-baselines.md — purity theorem protocol

Purity theorem note
-------------------
The theorem guarantees P > p — purity in memory EXCEEDS BASE generation
accuracy. It does NOT claim P > α (verification accuracy). Confusion between
these two is a committee-facing risk (writing-suggestions.md C5-03).

Usage
-----
  python -m scripts.run_purity_validation \\
      --cycle_results outputs/all_cycle_results.json \\
      --purity_samples outputs/dataset_splits.json \\
      --output_dir outputs/purity_validation

  # Smoke test:
  python -m scripts.run_purity_validation --smoke_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Theory 1: Data Purity Theorem
# ─────────────────────────────────────────────────────────────────────────────

def purity_theorem(p: float, alpha: float) -> float:
    """Compute theoretical memory purity P from base accuracy p and verification α.

    P = pα / (pα + (1-p)(1-α))

    Valid when p > (1 - α).  If the condition fails, the theorem provides
    no guarantee and is reported as violated.

    Parameters
    ----------
    p     : float — base generation accuracy (fraction correct before verification)
    alpha : float — verification precision (fraction of stored answers that are correct)

    Returns
    -------
    float — theoretical purity P_theory in [0, 1]
    """
    numerator = p * alpha
    denominator = p * alpha + (1 - p) * (1 - alpha)
    if denominator == 0:
        return 1.0
    return numerator / denominator


def check_purity_condition(p: float, alpha: float) -> bool:
    """Return True if the purity theorem's condition p > (1 - α) holds."""
    return p > (1 - alpha)


def measure_base_accuracy(pipeline, purity_samples: List[dict], bm: str) -> float:
    """Measure p — the fraction of questions the model answers correctly BEFORE
    verification. This is the raw generation accuracy on the purity validation set.

    Parameters
    ----------
    pipeline       : CAEMPipeline
    purity_samples : list of BenchmarkSample
    bm             : str — benchmark name

    Returns
    -------
    float — p (base accuracy in [0, 1])
    """
    from eval.metrics import (
        any_match_em, exact_match, extract_fever_label, fever_accuracy
    )
    import torch

    correct = 0
    total = 0

    for sample in purity_samples:
        q = sample["question"]
        gold = sample.get("answers", [])
        gold_label = sample.get("gold_label")

        try:
            # Generate directly (bypass memory routing — we want raw model accuracy)
            inputs = pipeline.tokenizer(
                q, return_tensors="pt", truncation=True, max_length=512
            ).to(pipeline.device)
            with torch.no_grad():
                out = pipeline.model.generate(**inputs, max_new_tokens=64, do_sample=False)
            pred = pipeline.tokenizer.decode(out[0], skip_special_tokens=True)

            if bm == "fever":
                pred_label = extract_fever_label(pred)
                ref = gold_label or (gold[0] if gold else "not enough info")
                em = fever_accuracy(pred_label, ref)
            elif bm == "truthfulqa":
                em = any_match_em(pred, gold)
            else:
                em = exact_match(pred, gold[0] if gold else "")

            correct += em
        except Exception as exc:
            logger.debug("measure_base_accuracy: skipped sample (%s)", exc)

        total += 1

    return correct / total if total > 0 else 0.0


def measure_verification_precision(pipeline, purity_samples: List[dict], bm: str) -> float:
    """Measure α — verification precision on the purity validation set.

    α = (correctly verified answers that were actually correct) /
        (all answers that passed verification)

    This tells us: of the answers the verifier ACCEPTED, what fraction
    were actually correct? A perfect verifier would have α = 1.0.

    Parameters
    ----------
    pipeline       : CAEMPipeline
    purity_samples : list of BenchmarkSample
    bm             : str — benchmark name

    Returns
    -------
    float — α (verification precision in [0, 1])
    """
    from eval.metrics import (
        any_match_em, exact_match, extract_fever_label, fever_accuracy
    )

    accepted_and_correct = 0
    accepted_total = 0

    for sample in purity_samples:
        q = sample["question"]
        gold = sample.get("answers", [])
        gold_label = sample.get("gold_label")

        try:
            result = pipeline.answer(q)
            if result.stored_confidence is None:
                continue

            # StoredConfidence has no passed_verification field.
            # An answer "passed" verification iff u_stored >= prune threshold.
            threshold = pipeline.config.retroverify_prune_threshold
            if result.stored_confidence.u_stored < threshold:
                continue

            # Answer passed verification — was it actually correct?
            pred = result.answer
            if bm == "fever":
                pred_label = extract_fever_label(pred)
                ref = gold_label or (gold[0] if gold else "not enough info")
                em = fever_accuracy(pred_label, ref)
            elif bm == "truthfulqa":
                em = any_match_em(pred, gold)
            else:
                em = exact_match(pred, gold[0] if gold else "")

            accepted_and_correct += em
            accepted_total += 1
        except Exception as exc:
            logger.debug("measure_verification_precision: skipped (%s)", exc)

    if accepted_total == 0:
        logger.warning("No accepted answers found — α cannot be measured.")
        return 0.0
    return accepted_and_correct / accepted_total


def measure_memory_purity(memory_store, purity_samples: List[dict], bm: str, pipeline) -> float:
    """Measure P_obs — the observed purity of episodes in the memory store.

    P_obs = (correct answers in memory) / (total answers in memory)
    Measured by checking stored answers against the gold labels from the
    purity validation set.

    Note: only stored answers whose questions are in the purity set are
    checked. This is an approximation — the full memory also contains
    answers from other sources.

    Returns
    -------
    float — P_obs (observed memory purity in [0, 1])
    """
    from eval.metrics import (
        any_match_em, exact_match, extract_fever_label, fever_accuracy
    )

    # Build gold answer lookup from purity samples
    gold_lookup: Dict[str, dict] = {}
    for s in purity_samples:
        gold_lookup[s["question"].strip().lower()] = s

    # Use the public all_entries() method, not the private _metadata dict
    store_entries = memory_store.all_entries()
    correct_in_memory = 0
    checked = 0

    for entry in store_entries:
        key = entry.question.strip().lower()
        if key not in gold_lookup:
            continue

        sample = gold_lookup[key]
        gold = sample.get("answers", [])
        gold_label = sample.get("gold_label")
        pred = entry.answer

        if bm == "fever":
            pred_label = extract_fever_label(pred)
            ref = gold_label or (gold[0] if gold else "not enough info")
            em = fever_accuracy(pred_label, ref)
        elif bm == "truthfulqa":
            em = any_match_em(pred, gold)
        else:
            em = exact_match(pred, gold[0] if gold else "")

        correct_in_memory += em
        checked += 1

    if checked == 0:
        return float("nan")
    return correct_in_memory / checked


# ─────────────────────────────────────────────────────────────────────────────
# Full validation protocol
# ─────────────────────────────────────────────────────────────────────────────

def run_purity_validation_protocol(
    pipelines_by_cycle: Dict[int, object],
    purity_samples: Dict[str, list],
    output_dir: Path,
) -> Dict:
    """Run the 5-step purity validation protocol for all cycles.

    Parameters
    ----------
    pipelines_by_cycle : dict[cycle_num → CAEMPipeline]
                         The pipeline at each cycle state (post fine-tuning).
                         At minimum cycle 0 and cycle 3 are required.
    purity_samples     : dict[bm → list of BenchmarkSample] — 500-sample set
    output_dir         : Path

    Returns
    -------
    dict — all three theory validation tables
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theory1_rows: List[Dict] = []
    theory2_p_values: Dict[str, List[float]] = {}
    theory2_alpha_values: Dict[str, List[float]] = {}

    for cycle_num, pipeline in sorted(pipelines_by_cycle.items()):
        for bm, bm_samples in purity_samples.items():
            logger.info("Purity validation: cycle=%d  bm=%s …", cycle_num, bm)

            # Step 1: Measure p (base accuracy)
            p = measure_base_accuracy(pipeline, bm_samples, bm)

            # Step 2: Measure α (verification precision)
            alpha = measure_verification_precision(pipeline, bm_samples, bm)

            # Step 3: Compute P_theory
            P_theory = purity_theorem(p, alpha)

            # Step 4: Measure P_obs (observed memory purity)
            P_obs = measure_memory_purity(pipeline.memory_store, bm_samples, bm, pipeline)

            # Step 5: Check condition p > (1 - α)
            condition_holds = check_purity_condition(p, alpha)

            row = {
                "cycle": cycle_num,
                "benchmark": bm,
                "p": round(p, 4),
                "alpha": round(alpha, 4),
                "P_theory": round(P_theory, 4),
                "P_obs": round(P_obs, 4) if not math.isnan(P_obs) else None,
                "P_obs_minus_p": round(P_obs - p, 4) if not math.isnan(P_obs) else None,
                "condition_p_gt_1_minus_alpha": condition_holds,
                "theorem_confirmed": (
                    condition_holds and
                    not math.isnan(P_obs) and
                    P_obs > p
                ),
            }
            theory1_rows.append(row)

            logger.info(
                "  Theory 1: p=%.4f  α=%.4f  P_theory=%.4f  P_obs=%.4f  "
                "condition=%s  P_obs>p=%s",
                p, alpha, P_theory,
                P_obs if not math.isnan(P_obs) else float("nan"),
                "✓" if condition_holds else "✗",
                "✓" if (not math.isnan(P_obs) and P_obs > p) else "✗",
            )

            # Accumulate Theory 2 values
            if bm not in theory2_p_values:
                theory2_p_values[bm] = []
                theory2_alpha_values[bm] = []
            theory2_p_values[bm].append(p)
            theory2_alpha_values[bm].append(alpha)

    # ── Theory 2: Monotonicity ─────────────────────────────────────────── #
    theory2_rows: List[Dict] = []
    for bm in purity_samples:
        p_vals = theory2_p_values.get(bm, [])
        a_vals = theory2_alpha_values.get(bm, [])
        if len(p_vals) < 2:
            continue
        p_mono = all(p_vals[i] < p_vals[i + 1] for i in range(len(p_vals) - 1))
        a_mono = all(a_vals[i] < a_vals[i + 1] for i in range(len(a_vals) - 1))
        theory2_rows.append({
            "benchmark": bm,
            "p_values": p_vals,
            "alpha_values": a_vals,
            "p_monotone": p_mono,
            "alpha_monotone": a_mono,
            "theory_2_confirmed": p_mono and a_mono,
        })
        logger.info(
            "  Theory 2 (%s): p monotone=%s  α monotone=%s",
            bm,
            "✓" if p_mono else "✗",
            "✓" if a_mono else "✗",
        )

    # ── Theory 3: Convergence ──────────────────────────────────────────── #
    theory3_rows: List[Dict] = []
    for bm in purity_samples:
        p_vals = theory2_p_values.get(bm, [])
        if len(p_vals) < 4:
            logger.warning("Theory 3 (%s): need 4 cycles, got %d.", bm, len(p_vals))
            continue
        deltas = [p_vals[i + 1] - p_vals[i] for i in range(len(p_vals) - 1)]
        # Δ(2→3) < Δ(1→2) for convergence
        converging = deltas[-1] < deltas[-2] if len(deltas) >= 2 else False
        theory3_rows.append({
            "benchmark": bm,
            "deltas": [round(d, 4) for d in deltas],
            "delta_1_2": round(deltas[1], 4) if len(deltas) > 1 else None,
            "delta_2_3": round(deltas[2], 4) if len(deltas) > 2 else None,
            "theory_3_confirmed": converging,
        })
        logger.info(
            "  Theory 3 (%s): Δ(1→2)=%.4f  Δ(2→3)=%.4f  converging=%s",
            bm,
            deltas[1] if len(deltas) > 1 else float("nan"),
            deltas[2] if len(deltas) > 2 else float("nan"),
            "✓" if converging else "✗",
        )

    # ── Save all results ────────────────────────────────────────────────── #
    results = {
        "theory_1_purity_theorem": theory1_rows,
        "theory_2_monotonicity": theory2_rows,
        "theory_3_convergence": theory3_rows,
    }
    out_path = output_dir / "theory_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Theory validation results saved → %s", out_path)

    # ── Print tables ─────────────────────────────────────────────────────── #
    _print_theory_tables(results)

    return results


def _print_theory_tables(results: Dict) -> None:
    """Print all three theory validation tables to stdout (Chapter 5)."""
    print("\n" + "═" * 80)
    print("THEORY VALIDATION TABLES  (Chapter 5, Section 5.5)")
    print("═" * 80)

    # ── Table 1: Purity Theorem ───────────────────────────────────────────── #
    print("\nTable T1 — Data Purity Theorem: P = pα / (pα + (1-p)(1-α))")
    print("CRITICAL: theorem guarantees P > p (base accuracy), NOT P > α (verification)")
    print("─" * 80)
    print(f"  {'Cycle':<5} {'BM':<12} {'p':>7} {'α':>7} {'P_theory':>10} "
          f"{'P_obs':>8} {'P_obs>p':>8} {'Cond':>6} {'✓':>4}")
    print("─" * 80)
    for row in results["theory_1_purity_theorem"]:
        pobs_str = f"{row['P_obs']:.4f}" if row.get("P_obs") is not None else "  n/a "
        confirmed = "✓" if row.get("theorem_confirmed") else "✗"
        cond = "✓" if row.get("condition_p_gt_1_minus_alpha") else "✗"
        pobs_gt_p = "✓" if (row.get("P_obs_minus_p") or 0) > 0 else "✗"
        print(
            f"  {row['cycle']:<5} {row['benchmark']:<12} "
            f"{row['p']:>7.4f} {row['alpha']:>7.4f} "
            f"{row['P_theory']:>10.4f} {pobs_str:>8} "
            f"{pobs_gt_p:>8} {cond:>6} {confirmed:>4}"
        )
    print("─" * 80)
    print("  Cond = p > (1-α) must hold. ✓ = theorem confirmed.\n")

    # ── Table 2: Monotonicity ─────────────────────────────────────────────── #
    print("Table T2 — Coupled Improvement Recurrence (Monotonicity)")
    print("─" * 80)
    for row in results["theory_2_monotonicity"]:
        p_str = " → ".join(f"{v:.4f}" for v in row["p_values"])
        a_str = " → ".join(f"{v:.4f}" for v in row["alpha_values"])
        p_ok = "✓" if row["p_monotone"] else "✗"
        a_ok = "✓" if row["alpha_monotone"] else "✗"
        print(f"  {row['benchmark']:<12}  p: {p_str}  [{p_ok}]")
        print(f"  {'':<12}  α: {a_str}  [{a_ok}]")
    print("─" * 80 + "\n")

    # ── Table 3: Convergence ─────────────────────────────────────────────── #
    print("Table T3 — Convergence: Δ(2→3) < Δ(1→2)")
    print("─" * 80)
    for row in results["theory_3_convergence"]:
        d_str = " → ".join(f"{d:+.4f}" for d in row["deltas"])
        ok = "✓" if row["theory_3_confirmed"] else "✗"
        d12 = row.get("delta_1_2") or float("nan")
        d23 = row.get("delta_2_3") or float("nan")
        print(
            f"  {row['benchmark']:<12}  Δ: {d_str}  "
            f"Δ(1→2)={d12:+.4f}  Δ(2→3)={d23:+.4f}  [{ok}]"
        )
    print("═" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_purity_validation(ns: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    from scripts.hardware import print_hardware_summary
    profile = print_hardware_summary()

    output_dir = Path(ns.output_dir)

    # ── Load pipeline and purity samples ─────────────────────────────────── #
    if ns.smoke_test:
        from eval.benchmarks import make_synthetic_samples
        purity_samples = {
            bm: make_synthetic_samples(bm, n=10)
            for bm in ["hotpotqa", "truthfulqa", "fever", "strategyqa"]
        }
        # Build a single dummy pipeline for smoke test
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        from caem.config import CAEMConfig
        from caem.memory.encoder import QueryEncoder
        from caem.pipeline import CAEMPipeline

        config = CAEMConfig()
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to(profile.device)
        encoder = QueryEncoder(config=config)
        pipeline = CAEMPipeline(model=model, tokenizer=tokenizer, encoder=encoder,
                                config=config, device=profile.device)
        pipelines_by_cycle = {0: pipeline}
    else:
        # Load all cycle pipelines
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        from caem.config import CAEMConfig
        from caem.memory.encoder import QueryEncoder
        from caem.pipeline import CAEMPipeline
        from eval.benchmarks import load_hotpotqa, load_truthfulqa, load_fever, load_strategyqa

        config = CAEMConfig()
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

        # Load purity validation samples
        # (first 500 per benchmark — from dataset_splits.json or reload)
        purity_samples = {
            "hotpotqa":   load_hotpotqa(n=500)[:500],
            "truthfulqa": load_truthfulqa(n=500)[:500],
            "fever":      load_fever(n=500)[:500],
            "strategyqa": load_strategyqa(n=500)[:500],
        }

        # Build a pipeline for each cycle checkpoint
        pipelines_by_cycle = {}
        for cycle_num in range(ns.num_cycles + 1):
            ckpt_dir = Path(ns.checkpoints_dir) / f"cycle_{cycle_num}"
            model_path = ckpt_dir / "model.pt"

            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
            if model_path.exists():
                logger.info("Loading cycle %d weights from %s …", cycle_num, model_path)
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            else:
                logger.warning("Cycle %d checkpoint not found at %s — using base weights.", cycle_num, model_path)

            if profile.use_fp16:
                model = model.half()
            elif profile.use_bf16:
                model = model.to(torch.bfloat16)
            model = model.to(profile.device).eval()

            encoder = QueryEncoder(config=config)
            pipeline = CAEMPipeline(
                model=model, tokenizer=tokenizer, encoder=encoder,
                config=config, device=profile.device, current_cycle=cycle_num,
            )
            pipelines_by_cycle[cycle_num] = pipeline
            logger.info("Cycle %d pipeline ready.", cycle_num)

    # ── Run validation ────────────────────────────────────────────────────── #
    run_purity_validation_protocol(pipelines_by_cycle, purity_samples, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAEM Theory Validation (Three Protocols)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoints_dir", default="outputs",
                   help="Root directory containing cycle_{n}/ subdirectories.")
    p.add_argument("--output_dir", default="outputs/purity_validation",
                   help="Where to save validation results.")
    p.add_argument("--num_cycles", type=int, default=3,
                   help="Number of self-improvement cycles (0–N).")
    p.add_argument("--cycle_results", default="outputs/all_cycle_results.json",
                   help="Path to all_cycle_results.json (for cross-reference).")
    p.add_argument("--smoke_test", action="store_true",
                   help="Tiny synthetic run (no GPU or datasets needed).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_purity_validation(args)
