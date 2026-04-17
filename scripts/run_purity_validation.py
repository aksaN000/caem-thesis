"""
scripts/run_purity_validation.py
==================================
CAEM Theory Validation -- Three Protocols (Chapter 5, Section 5.5)

Validates all three theoretical claims in the thesis:

  Theory 1 -- Data Purity Theorem
    Claim:   P = pα / (pα + (1-p)(1-α))   AND   P > p
    Condition: α > 0.5  must hold for the theorem to guarantee P > p
    CRITICAL: theorem guarantees P > p (base accuracy), NOT P > α.
    Protocol: measure p and α per cycle on the purity validation set;
              compute P_theory; compare with P_obs (actual memory accuracy).

  Theory 2 -- Coupled Improvement Recurrence (Monotonicity)
        Claim:   p_0 < p_1 < ... < p_N  AND  α_0 < α_1 < ... < α_N
    Protocol: report p and α per cycle; confirm strict monotonicity.

  Theory 3 -- Convergence
        Claim:   late-cycle Δ shrinks, where Δ(k->k+1) = p_{k+1} - p_k
        Protocol: compare the last two Δ values; confirm diminishing improvement.

All three validations use the purity validation set (500 samples, separate
from both the calibration set and the eval set).

Fixes applied (audit 2025-04)
------------------------------
  FIX-1  check_purity_condition(): the correct sufficient condition for
         P > p is α > 0.5, NOT p > (1-α). Theorem 1 (§4.9) states:
         P > p iff α > ½. The old condition was a stricter and incorrect
         restatement. This was a critical logic error in the validation.

  FIX-2  measure_base_accuracy(): max_new_tokens raised 64→256 (was
         truncating answers for multi-sentence benchmarks). TruthfulQA
         now uses ROUGE-L > 0.15 (matches eval/harness.py L267-269);
         StrategyQA now uses extract_strategyqa_label() (matches
         eval/harness.py L272-275).

  FIX-3  measure_verification_precision() renamed to
         measure_verification_balanced_accuracy(). Old implementation
         measured precision = TP/(TP+FP). Thesis Definition 4.2 requires
         balanced accuracy α = (TP+TN)/N. The fix counts both correctly
         accepted correct answers AND correctly rejected wrong answers.

  FIX-4  Acceptance threshold for theorem α uses
      retroverify_prune_threshold (the storage acceptance gate), not
      u_hat_accept_threshold (Tier-2 post-generation gate). Theorem 1 is
      about purity of accepted memory episodes, so it must use the
      storage gate.

  FIX-5  Memory store loading now matches run_experiment.py persistence
      format: outputs/memory_store_cycle_{n}.faiss + .meta. This prevents
      P_obs from being NaN due to searching for a non-existent
      memory_store.pkl format.

  FIX-6  α measurement no longer calls pipeline.answer() (which mutates
      memory via Stage 7). It now runs generation + verifier directly,
      keeping the loaded cycle memory unchanged during measurement.

  FIX-7  Purity scoring now uses extract_cot_answer() before EM scoring,
      matching eval/harness.py behavior for rationale-style outputs.

Thesis reference
----------------
  §4.9  Theoretical analysis (purity theorem + convergence)
  §5.5  Theory validation (three protocols, one table each)
  §6.1  Conclusion: theoretical grounding distinguishes CAEM from heuristics
  benchmarks-and-baselines.md -- purity theorem protocol

Purity theorem note
-------------------
The theorem guarantees P > p -- purity in memory EXCEEDS BASE generation
accuracy. It does NOT claim P > α (verification accuracy). Confusion between
these two is a committee-facing risk (writing-suggestions.md C5-03).
The sufficient condition is α > 0.5 (Theorem 1, §4.9). When α ≤ 0.5 the
verifier is no better than chance and the theorem provides no guarantee.

Usage
-----
  python -m scripts.run_purity_validation \\
      --checkpoints_dir outputs \\
      --output_dir outputs/purity_validation \\
            --num_cycles 10 \\
            --benchmarks fever triviaqa natural_questions

  # Smoke test:
  python -m scripts.run_purity_validation --smoke_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

logger = logging.getLogger(__name__)


PURITY_BENCHMARK_DEFAULTS = ["fever", "triviaqa", "natural_questions"]
PURITY_BENCHMARK_SPLITS = {
    # Purity/calibration sets are carved from the SIL training pool.
    "fever": "train",
    "triviaqa": "train",
    "natural_questions": "train",
    # Transfer-only benchmarks are optional here and usually do not have purity_ids.
    "truthfulqa": "validation",
    "strategyqa": "test",
    "arc_challenge": "test",
}


# -----------------------------------------------------------------------------
# Theory 1: Data Purity Theorem
# -----------------------------------------------------------------------------

def purity_theorem(p: float, alpha: float) -> float:
    """Compute theoretical memory purity P from base accuracy p and verification α.

    P = pα / (pα + (1-p)(1-α))

    Valid when α > 0.5.  If the condition fails, the theorem provides
    no guarantee (P may be <= p) and is reported as violated.

    Parameters
    ----------
    p     : float -- base generation accuracy (fraction correct before verification)
    alpha : float -- verification balanced accuracy (TP+TN)/N

    Returns
    -------
    float -- theoretical purity P_theory in [0, 1]
    """
    numerator = p * alpha
    denominator = p * alpha + (1 - p) * (1 - alpha)
    if denominator == 0:
        return 1.0
    return numerator / denominator


def check_purity_condition(p: float, alpha: float) -> bool:
    """Return True if the purity theorem's sufficient condition α > 0.5 holds.

    FIX-1: Thesis Theorem 1 (§4.9) proves P > p if and only if α > ½.
    The old implementation returned p > (1 - alpha), which is equivalent to
    alpha > (1 - p). That is a strictly stronger condition than α > 0.5 and
    is INCORRECT -- it would report the theorem as violated even when α > 0.5
    (e.g., p=0.6, α=0.7: correct condition holds since 0.7 > 0.5, but old
    code returned 0.6 > 0.3 = True by coincidence only because p was also > 0.5).
    The clearest counterexample: p=0.3, α=0.8 -- correct: α=0.8 > 0.5 ✓,
    old code: 0.3 > (1-0.8)=0.2 ✓ coincidentally, but for p=0.1, α=0.8:
    old code: 0.1 > 0.2 = False ✗ WRONG (theorem still holds since α=0.8 > 0.5).

    Parameters
    ----------
    p     : float -- base generation accuracy (unused in the correct condition,
                     retained for API compatibility)
    alpha : float -- verification balanced accuracy

    Returns
    -------
    bool -- True iff α > 0.5 (theorem guarantee holds)
    """
    return alpha > 0.5


def _score_em(prediction: str, gold: List[str], gold_label: Optional[str], bm: str) -> float:
    """Return EM score aligned with eval/harness.py benchmark-specific scoring."""
    from eval.metrics import (
        exact_match,
        extract_arc_label,
        extract_cot_answer,
        extract_fever_label,
        extract_strategyqa_label,
        fever_accuracy,
        rouge_l,
    )

    pred = extract_cot_answer(prediction)
    if bm == "fever":
        pred_label = extract_fever_label(pred)
        ref = gold_label or (gold[0] if gold else "not enough info")
        return fever_accuracy(pred_label, ref)
    if bm == "truthfulqa":
        return float(rouge_l(pred, gold) > 0.15)
    if bm == "strategyqa":
        pred_label = extract_strategyqa_label(pred)
        ref_label = extract_strategyqa_label(gold[0] if gold else "no")
        return float(pred_label == ref_label) if pred_label and ref_label else 0.0
    if bm == "arc_challenge":
        pred_label = extract_arc_label(pred)
        ref = gold[0] if gold else ""
        return exact_match(pred_label, ref)
    return exact_match(pred, gold[0] if gold else "")


def _generate_greedy_answer(pipeline, query: str, max_new_tokens: int = 256) -> Tuple[str, Any]:
    """Generate one deterministic answer and return (decoded_answer, input_ids)."""
    import torch

    inputs = pipeline.tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(pipeline.device)

    with torch.no_grad():
        out = pipeline.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    pred = pipeline.tokenizer.decode(out[0], skip_special_tokens=True)
    return pred, inputs["input_ids"]


def measure_base_accuracy(pipeline, purity_samples: List[dict], bm: str) -> float:
    """Measure p -- the fraction of questions the model answers correctly BEFORE
    verification. This is the raw generation accuracy on the purity validation set.

    FIX-2: max_new_tokens raised from 64 to 256 to match harness.py (prevents
    answer truncation on multi-sentence benchmarks). TruthfulQA scoring changed
    to ROUGE-L > 0.15 (was any_match_em exact string match). StrategyQA scoring
    changed to use extract_strategyqa_label() (was raw exact_match).

    Parameters
    ----------
    pipeline       : CAEMPipeline
    purity_samples : list of BenchmarkSample
    bm             : str -- benchmark name

    Returns
    -------
    float -- p (base accuracy in [0, 1])
    """
    correct = 0
    total = 0

    for sample in purity_samples:
        q = sample["question"]
        gold = sample.get("answers", [])
        gold_label = sample.get("gold_label")

        try:
            # Generate directly (bypass memory routing -- we want raw model accuracy).
            pred, _ = _generate_greedy_answer(
                pipeline,
                q,
                max_new_tokens=256,  # FIX-2: was 64 (too short for multi-sentence answers)
            )
            em = _score_em(pred, gold, gold_label, bm)

            correct += em
        except Exception as exc:
            logger.debug("measure_base_accuracy: skipped sample (%s)", exc)

        total += 1

    return correct / total if total > 0 else 0.0


def measure_verification_balanced_accuracy(
    pipeline, purity_samples: List[dict], bm: str
) -> float:
    """Measure α -- verification BALANCED ACCURACY on the purity validation set.

    α = (TP + TN) / N

    where:
      TP = answer was correct   AND passed verification (u_stored >= threshold)
      TN = answer was wrong     AND failed verification (u_stored <  threshold)
      N  = total samples evaluated

    FIX-3: The old implementation measured precision = TP/(TP+FP), i.e., only
    counting accepted answers. Thesis Definition 4.2 defines α as balanced
    accuracy over ALL samples, including correctly rejected wrong answers.
    Balanced accuracy more faithfully represents verifier quality: a verifier
    that rejects everything has precision=undefined but balanced accuracy=0.5.

    FIX-4: The acceptance threshold for theorem α is retroverify_prune_threshold,
    because Theorem 1 concerns memory purity of accepted/stored episodes.
    u_hat_accept_threshold is the Tier-2 post-generation gate and is not the
    storage acceptance criterion.

    FIX-6: Uses generation + verifier directly instead of pipeline.answer() so
    measurement does not mutate memory (Stage 7 storage side effects).

    Parameters
    ----------
    pipeline       : CAEMPipeline
    purity_samples : list of BenchmarkSample
    bm             : str -- benchmark name

    Returns
    -------
    float -- α (verification balanced accuracy in [0, 1])
    """
    # FIX-4: Use the storage acceptance threshold for theorem α.
    threshold = pipeline.config.retroverify_prune_threshold

    tp = 0  # correct answer, passed verification
    tn = 0  # wrong answer,   failed verification
    fp = 0  # wrong answer,   passed verification  (false acceptance)
    fn = 0  # correct answer, failed verification  (false rejection)
    total = 0

    for sample in purity_samples:
        q = sample["question"]
        gold = sample.get("answers", [])
        gold_label = sample.get("gold_label")

        try:
            pred, input_ids = _generate_greedy_answer(pipeline, q, max_new_tokens=256)
            sc = pipeline.verifier.verify(query=q, answer=pred, input_ids=input_ids)
            passed_verification = sc.u_stored >= threshold
            is_correct = bool(_score_em(pred, gold, gold_label, bm))

            # Accumulate confusion matrix
            if is_correct and passed_verification:
                tp += 1
            elif (not is_correct) and (not passed_verification):
                tn += 1
            elif (not is_correct) and passed_verification:
                fp += 1
            else:  # is_correct and not passed_verification
                fn += 1

            total += 1
        except Exception as exc:
            logger.debug("measure_verification_balanced_accuracy: skipped (%s)", exc)

    if total == 0:
        logger.warning("No samples evaluated for α -- balanced accuracy cannot be measured.")
        return 0.0

    balanced_acc = (tp + tn) / total
    logger.debug(
        "Verifier confusion matrix (%s): TP=%d TN=%d FP=%d FN=%d "
        "N=%d  α_balanced=%.4f  precision=%.4f",
        bm, tp, tn, fp, fn, total,
        balanced_acc,
        tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
    )
    return balanced_acc


def load_memory_store_for_cycle(pipeline, cycle_num: int, checkpoints_dir: str):
    """Load the memory store for a cycle.

    FIX-5: When a pipeline is reconstructed from model.pt, its memory_store
    is freshly initialised (empty). This causes P_obs = NaN for every cycle
    since measure_memory_purity() finds no matching episodes.

        This function loads the same persistence format produced by
        run_experiment.py:
            {checkpoints_dir}/memory_store_cycle_{n}.faiss
            {checkpoints_dir}/memory_store_cycle_{n}.meta

    Parameters
    ----------
    pipeline       : CAEMPipeline -- pipeline to update in place
    cycle_num      : int
    checkpoints_dir: str -- root dir containing cycle_{n}/ subdirs

    Returns
    -------
    bool -- True if a memory store was loaded, False otherwise
    """
    from caem.memory.store import EpisodicMemoryStore

    store_base = Path(checkpoints_dir) / f"memory_store_cycle_{cycle_num}"
    meta_path = Path(str(store_base) + ".meta")

    if not meta_path.exists():
        logger.warning(
            "FIX-5: memory store checkpoint not found at %s(.faiss/.meta) -- "
            "P_obs may be NaN for cycle %d.",
            store_base, cycle_num,
        )
        return False

    try:
        pipeline.memory_store = EpisodicMemoryStore.load(str(store_base))
        logger.info(
            "FIX-5: Loaded memory store for cycle %d (%d episodes) from %s.",
            cycle_num, pipeline.memory_store.size, store_base,
        )
        return True
    except Exception as exc:
        logger.warning("FIX-5: Failed to load memory store from %s: %s", store_base, exc)
        return False


def measure_memory_purity(memory_store, purity_samples: List[dict], bm: str) -> float:
    """Measure P_obs -- the observed purity of episodes in the memory store.

    P_obs = (correct answers in memory) / (total answers in memory)
    Measured by checking stored answers against the gold labels from the
    purity validation set.

    Note: only stored answers whose questions are in the purity set are
    checked. This is an approximation -- the full memory also contains
    answers from other sources.

    Returns
    -------
    float -- P_obs (observed memory purity in [0, 1]), NaN if no overlap found
    """
    # Build gold answer lookup from purity samples
    gold_lookup: Dict[str, dict] = {}
    for s in purity_samples:
        gold_lookup[s["question"].strip().lower()] = s

    # Use the public all_entries() method
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
        em = _score_em(entry.answer, gold, gold_label, bm)

        correct_in_memory += em
        checked += 1

    if checked == 0:
        return float("nan")
    return correct_in_memory / checked


# -----------------------------------------------------------------------------
# Full validation protocol
# -----------------------------------------------------------------------------

def run_purity_validation_protocol(
    pipelines_by_cycle: Mapping[int, Any],
    purity_samples: Dict[str, list],
    output_dir: Path,
    checkpoints_dir: Optional[str] = None,
) -> Dict:
    """Run the 5-step purity validation protocol for all cycles.

    Parameters
    ----------
    pipelines_by_cycle : dict[cycle_num -> CAEMPipeline]
                         The pipeline at each cycle state (post fine-tuning).
                         At least 4 cycle states are required for Theory 3.
    purity_samples     : dict[bm -> list of BenchmarkSample] -- 500-sample set
    output_dir         : Path
    checkpoints_dir    : str or None -- root dir for cycle checkpoint subdirs,
                         used by FIX-5 to load memory stores

    Returns
    -------
    dict -- all three theory validation tables
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    theory1_rows: List[Dict] = []
    theory2_p_values: Dict[str, List[float]] = {}
    theory2_alpha_values: Dict[str, List[float]] = {}

    for cycle_num, pipeline in sorted(pipelines_by_cycle.items()):
        # FIX-5: Load memory store for this cycle before measuring P_obs
        if checkpoints_dir is not None:
            load_memory_store_for_cycle(pipeline, cycle_num, checkpoints_dir)

        for bm, bm_samples in purity_samples.items():
            logger.info("Purity validation: cycle=%d  bm=%s ...", cycle_num, bm)

            # Step 1: Measure p (base accuracy, bypassing memory routing)
            p = measure_base_accuracy(pipeline, bm_samples, bm)

            # Step 2: Measure α (verification balanced accuracy)   [FIX-3, FIX-4]
            alpha = measure_verification_balanced_accuracy(pipeline, bm_samples, bm)

            # Step 3: Compute P_theory
            P_theory = purity_theorem(p, alpha)

            # Step 4: Measure P_obs (observed memory purity)
            P_obs = measure_memory_purity(pipeline.memory_store, bm_samples, bm)

            # Step 5: Check condition α > 0.5  [FIX-1]
            condition_holds = check_purity_condition(p, alpha)

            row = {
                "cycle": cycle_num,
                "benchmark": bm,
                "p": round(p, 4),
                "alpha": round(alpha, 4),
                "P_theory": round(P_theory, 4),
                "P_obs": round(P_obs, 4) if not math.isnan(P_obs) else None,
                "P_obs_minus_p": round(P_obs - p, 4) if not math.isnan(P_obs) else None,
                # FIX-1: renamed from condition_p_gt_1_minus_alpha
                "condition_alpha_gt_0_5": condition_holds,
                "theorem_confirmed": (
                    condition_holds and
                    not math.isnan(P_obs) and
                    P_obs > p
                ),
            }
            theory1_rows.append(row)

            logger.info(
                "  Theory 1: p=%.4f  α=%.4f  P_theory=%.4f  P_obs=%.4f  "
                "cond(α>0.5)=%s  P_obs>p=%s",
                p, alpha, P_theory,
                P_obs if not math.isnan(P_obs) else float("nan"),
                "OK" if condition_holds else "✗ (α≤0.5, theorem no-guarantee)",
                "OK" if (not math.isnan(P_obs) and P_obs > p) else "✗",
            )

            # Accumulate Theory 2 values
            if bm not in theory2_p_values:
                theory2_p_values[bm] = []
                theory2_alpha_values[bm] = []
            theory2_p_values[bm].append(p)
            theory2_alpha_values[bm].append(alpha)

    # -- Theory 2: Monotonicity ------------------------------------------- #
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
            "OK" if p_mono else "✗",
            "OK" if a_mono else "✗",
        )

    # -- Theory 3: Convergence -------------------------------------------- #
    theory3_rows: List[Dict] = []
    for bm in purity_samples:
        p_vals = theory2_p_values.get(bm, [])
        if len(p_vals) < 4:
            logger.warning("Theory 3 (%s): need at least 4 cycle points, got %d.", bm, len(p_vals))
            continue
        deltas = [p_vals[i + 1] - p_vals[i] for i in range(len(p_vals) - 1)]
        # Tail comparison confirms diminishing returns in late cycles.
        tail_prev_idx = len(p_vals) - 3
        tail_last_idx = len(p_vals) - 2
        delta_tail_prev = deltas[-2]
        delta_tail_last = deltas[-1]
        converging = delta_tail_last < delta_tail_prev
        theory3_rows.append({
            "benchmark": bm,
            "deltas": [round(d, 4) for d in deltas],
            "tail_prev_pair": f"{tail_prev_idx}->{tail_prev_idx + 1}",
            "tail_last_pair": f"{tail_last_idx}->{tail_last_idx + 1}",
            "delta_tail_prev": round(delta_tail_prev, 4),
            "delta_tail_last": round(delta_tail_last, 4),
            # Legacy fields kept for backward compatibility with prior analysis notebooks.
            "delta_1_2": round(deltas[1], 4) if len(deltas) > 1 else None,
            "delta_2_3": round(deltas[2], 4) if len(deltas) > 2 else None,
            "theory_3_confirmed": converging,
        })
        logger.info(
            "  Theory 3 (%s): Δ(%s)=%.4f  Δ(%s)=%.4f  converging=%s",
            bm,
            f"{tail_prev_idx}->{tail_prev_idx + 1}",
            delta_tail_prev,
            f"{tail_last_idx}->{tail_last_idx + 1}",
            delta_tail_last,
            "OK" if converging else "✗",
        )

    # -- Save all results -------------------------------------------------- #
    results = {
        "theory_1_purity_theorem": theory1_rows,
        "theory_2_monotonicity": theory2_rows,
        "theory_3_convergence": theory3_rows,
    }
    out_path = output_dir / "theory_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Theory validation results saved -> %s", out_path)

    # -- Print tables ------------------------------------------------------- #
    _print_theory_tables(results)

    return results


def _print_theory_tables(results: Dict) -> None:
    """Print all three theory validation tables to stdout (Chapter 5)."""
    print("\n" + "=" * 80)
    print("THEORY VALIDATION TABLES  (Chapter 5, Section 5.5)")
    print("=" * 80)

    # -- Table 1: Purity Theorem --------------------------------------------- #
    print("\nTable T1 -- Data Purity Theorem: P = pα / (pα + (1-p)(1-α))")
    print("CRITICAL: theorem guarantees P > p (base accuracy), NOT P > α (verification)")
    print("Sufficient condition: α > 0.5  [FIX-1: was p > (1-α), now corrected]")
    print("-" * 80)
    print(f"  {'Cycle':<5} {'BM':<12} {'p':>7} {'α':>7} {'P_theory':>10} "
          f"{'P_obs':>8} {'P>p':>5} {'α>0.5':>7} {'OK':>4}")
    print("-" * 80)
    for row in results["theory_1_purity_theorem"]:
        pobs_str = f"{row['P_obs']:.4f}" if row.get("P_obs") is not None else "  n/a "
        confirmed = "OK" if row.get("theorem_confirmed") else "✗"
        # FIX-1: column renamed from condition_p_gt_1_minus_alpha
        cond = "OK" if row.get("condition_alpha_gt_0_5") else "✗"
        pobs_gt_p = "OK" if (row.get("P_obs_minus_p") or 0) > 0 else "✗"
        print(
            f"  {row['cycle']:<5} {row['benchmark']:<12} "
            f"{row['p']:>7.4f} {row['alpha']:>7.4f} "
            f"{row['P_theory']:>10.4f} {pobs_str:>8} "
            f"{pobs_gt_p:>5} {cond:>7} {confirmed:>4}"
        )
    print("-" * 80)
    print("  α>0.5 = sufficient condition for P>p. OK = both condition holds AND P_obs>p.\n")

    # -- Table 2: Monotonicity ----------------------------------------------- #
    print("Table T2 -- Coupled Improvement Recurrence (Monotonicity)")
    print("-" * 80)
    for row in results["theory_2_monotonicity"]:
        p_str = " -> ".join(f"{v:.4f}" for v in row["p_values"])
        a_str = " -> ".join(f"{v:.4f}" for v in row["alpha_values"])
        p_ok = "OK" if row["p_monotone"] else "✗"
        a_ok = "OK" if row["alpha_monotone"] else "✗"
        print(f"  {row['benchmark']:<12}  p: {p_str}  [{p_ok}]")
        print(f"  {'':<12}  α: {a_str}  [{a_ok}]")
    print("-" * 80 + "\n")

    # -- Table 3: Convergence ----------------------------------------------- #
    print("Table T3 -- Convergence: tail Δ decreases")
    print("-" * 80)
    for row in results["theory_3_convergence"]:
        d_str = " -> ".join(f"{d:+.4f}" for d in row["deltas"])
        ok = "OK" if row["theory_3_confirmed"] else "✗"
        pair_prev = row.get("tail_prev_pair") or "n/a"
        pair_last = row.get("tail_last_pair") or "n/a"
        d_prev = row.get("delta_tail_prev")
        d_last = row.get("delta_tail_last")
        print(
            f"  {row['benchmark']:<12}  Δ: {d_str}  "
            f"Δ({pair_prev})={float(d_prev):+.4f}  Δ({pair_last})={float(d_last):+.4f}  [{ok}]"
        )
    print("=" * 80 + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_purity_validation(ns: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    from scripts.hardware import print_hardware_summary
    profile = print_hardware_summary()

    output_dir = Path(ns.output_dir)

    # -- Load pipeline and purity samples ----------------------------------- #
    if ns.smoke_test:
        from eval.benchmarks import make_synthetic_samples
        purity_samples = {
            bm: make_synthetic_samples(bm, n=10)
            for bm in PURITY_BENCHMARK_DEFAULTS
        }
        # Build a single dummy pipeline for smoke test
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        from caem.config import CAEMConfig
        from caem.memory.encoder import QueryEncoder
        from caem.pipeline import CAEMPipeline

        config = CAEMConfig()
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        model = cast(Any, model).to(torch.device(profile.device))
        encoder = QueryEncoder(model_name=config.sbert_model, device=profile.device)
        pipeline = CAEMPipeline(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            config=config,
            device=profile.device,
        )
        pipelines_by_cycle = {0: pipeline}
        checkpoints_dir = None
    else:
        # Load all cycle pipelines
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        from caem.config import CAEMConfig
        from caem.memory.encoder import QueryEncoder
        from caem.pipeline import CAEMPipeline
        from caem.retrieval.rag import PassageStore
        from eval.benchmarks import (
            load_arc_challenge,
            load_fever,
            load_natural_questions,
            load_strategyqa,
            load_triviaqa,
            load_truthfulqa,
        )

        config = CAEMConfig()
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

        nli_model, nli_tokenizer = None, None
        try:
            from transformers import AutoModelForSequenceClassification
            logger.info("Loading RoBERTa-Large-MNLI for purity verification ...")
            nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
            nli_model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-large-mnli"
            ).to(profile.device)
            nli_model.eval()
        except Exception as exc:
            logger.warning("NLI model load failed (%s); continuing without NLI.", exc)

        passage_store = None
        passage_index_path = Path(ns.passage_index)
        if passage_index_path.exists():
            try:
                passage_store = PassageStore.load(str(passage_index_path))
            except Exception as exc:
                logger.warning("Passage store load failed (%s); continuing without RAG.", exc)
        else:
            logger.warning("Passage index not found at %s; continuing without RAG.", passage_index_path)

        checkpoints_dir = ns.checkpoints_dir
        requested_benchmarks = [bm.strip().lower() for bm in ns.benchmarks if bm.strip()]
        if not requested_benchmarks:
            requested_benchmarks = list(PURITY_BENCHMARK_DEFAULTS)

        loader_by_benchmark = {
            "fever": lambda: load_fever(split=PURITY_BENCHMARK_SPLITS["fever"]),
            "triviaqa": lambda: load_triviaqa(split=PURITY_BENCHMARK_SPLITS["triviaqa"]),
            "natural_questions": lambda: load_natural_questions(split=PURITY_BENCHMARK_SPLITS["natural_questions"]),
            "truthfulqa": lambda: load_truthfulqa(),
            "strategyqa": lambda: load_strategyqa(split=PURITY_BENCHMARK_SPLITS["strategyqa"]),
            "arc_challenge": lambda: load_arc_challenge(split=PURITY_BENCHMARK_SPLITS["arc_challenge"]),
        }

        unknown_benchmarks = [bm for bm in requested_benchmarks if bm not in loader_by_benchmark]
        if unknown_benchmarks:
            raise ValueError(
                f"Unknown benchmarks for purity validation: {unknown_benchmarks}. "
                f"Supported: {sorted(loader_by_benchmark.keys())}"
            )

        # FIX: Load dataset splits to get exact purity_ids for the specific experiment run
        splits_path = Path(checkpoints_dir) / "dataset_splits.json"
        purity_ids_by_bm = {}
        if splits_path.exists():
            try:
                with open(splits_path, "r", encoding="utf-8") as f:
                    splits = json.load(f)
                for bm, splits_dict in splits.items():
                    purity_ids_by_bm[bm] = set(splits_dict.get("purity_ids", []))
                logger.info("Loaded exact purity_ids from %s", splits_path)
            except Exception as exc:
                logger.warning("Failed to load dataset_splits.json: %s", exc)
        else:
            logger.warning("dataset_splits.json not found at %s. Falling back to top 500.", splits_path)

        def load_and_filter(bm_name: str):
            # Load from the benchmark's configured split for purity validation.
            samples = loader_by_benchmark[bm_name]()
            if bm_name in purity_ids_by_bm and purity_ids_by_bm[bm_name]:
                target_ids = purity_ids_by_bm[bm_name]
                # Match either the string ID or the fallback index
                filtered = [s for i, s in enumerate(samples) if str(s.get("id", i)) in target_ids or s.get("id", i) in target_ids]
                if filtered:
                    logger.info("Filtered %s to %d precise purity_ids", bm_name, len(filtered))
                    return filtered
            # Fallback to top 500 if no splits file or no matching IDs
            logger.info("Falling back to top 500 items for %s", bm_name)
            return samples[:500]

        # Prefer benchmarks with explicit purity_ids from this experiment run.
        benchmarks_with_purity_ids = [
            bm for bm in requested_benchmarks
            if bm in purity_ids_by_bm and purity_ids_by_bm[bm]
        ]

        if benchmarks_with_purity_ids:
            selected_benchmarks = benchmarks_with_purity_ids
            skipped_no_ids = [bm for bm in requested_benchmarks if bm not in set(selected_benchmarks)]
            if skipped_no_ids:
                logger.info(
                    "Skipping benchmarks without purity_ids in dataset_splits.json: %s",
                    skipped_no_ids,
                )
        else:
            selected_benchmarks = requested_benchmarks
            logger.warning(
                "No purity_ids found for requested benchmarks; using top-500 fallback slices for: %s",
                selected_benchmarks,
            )

        # Load purity validation samples based on the exact IDs held out during the experiment
        purity_samples = {
            bm: load_and_filter(bm)
            for bm in selected_benchmarks
        }

        # Build a pipeline for each cycle checkpoint
        pipelines_by_cycle = {}
        for cycle_num in range(ns.num_cycles + 1):
            ckpt_dir = Path(checkpoints_dir) / f"cycle_{cycle_num}"
            model_path = ckpt_dir / "model.pt"

            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
            if model_path.exists():
                logger.info("Loading cycle %d weights from %s ...", cycle_num, model_path)
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            else:
                logger.warning(
                    "Cycle %d checkpoint not found at %s -- using base weights.",
                    cycle_num, model_path,
                )

            if profile.use_fp16:
                model = model.half()
            elif profile.use_bf16:
                model = model.bfloat16()
            model = cast(Any, model).to(torch.device(profile.device)).eval()

            encoder = QueryEncoder(model_name=config.sbert_model, device=profile.device)
            pipeline = CAEMPipeline(
                model=model, tokenizer=tokenizer, encoder=encoder,
                nli_model=nli_model, nli_tokenizer=nli_tokenizer,
                passage_store=passage_store,
                config=config, device=profile.device, current_cycle=cycle_num,
            )
            # FIX-5: memory store is loaded inside run_purity_validation_protocol
            # just before measuring P_obs for each cycle
            pipelines_by_cycle[cycle_num] = pipeline
            logger.info("Cycle %d pipeline ready.", cycle_num)

    # -- Run validation ------------------------------------------------------ #
    run_purity_validation_protocol(
        pipelines_by_cycle,
        purity_samples,
        output_dir,
        checkpoints_dir=checkpoints_dir,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAEM Theory Validation (Three Protocols)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoints_dir", default="outputs",
                   help="Root directory containing cycle_{n}/ subdirectories. "
                        "Memory checkpoints are expected as memory_store_cycle_{n}.faiss/.meta under this root.")
    p.add_argument("--output_dir", default="outputs/purity_validation",
                   help="Where to save validation results.")
    p.add_argument("--num_cycles", type=int, default=10,
                   help="Number of self-improvement cycles (0-N).")
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(PURITY_BENCHMARK_DEFAULTS),
        help=(
            "Benchmarks to validate. Defaults to SIL purity benchmarks "
            "used by run_experiment (fever, triviaqa, natural_questions)."
        ),
    )
    p.add_argument("--passage_index", default="data/passage_index",
                   help="Path to PassageStore directory (used for Tier-3 RAG during validation).")
    p.add_argument("--cycle_results", default="outputs/all_cycle_results.json",
                   help="Path to all_cycle_results.json (for cross-reference).")
    p.add_argument("--smoke_test", action="store_true",
                   help="Tiny synthetic run (no GPU or datasets needed).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_purity_validation(args)
