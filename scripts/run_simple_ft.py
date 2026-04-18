"""
scripts/run_simple_ft.py
========================
Training-time baselines for the Chapter 5 panel.

Two baselines share this runner, toggled by flags:

  B6 Vanilla FT     --use_l2_anchor=False  --use_mmlu_guard=False
  B7 EWC-only FT    --use_l2_anchor=True   --use_mmlu_guard=True

Both strip CAEM's verifier, episodic memory, self-consistency, and
semantic-entropy gates; neither uses retrieval augmentation at train
or eval time. The only mechanism preserved in B7 is the continual-
learning machinery (θ-anchor + MMLU retention rollback), which
isolates the contribution of confidence-aware storage from the
contribution of the anchor itself.

Training data
-------------
Raw (question, answer) pairs are loaded straight from the benchmark
training splits (FEVER, TriviaQA, Natural Questions). No verifier
filter, no memory curation, no 90/10 general-data mix (unless
--general_data_frac > 0).

Evaluation
----------
After each cycle the fine-tuned weights are evaluated on the full
six-benchmark panel (3 ID + 3 OOD; see ``TRAINING_BENCHMARKS`` /
``TRANSFER_BENCHMARKS`` in ``caem/config.py``) via
``eval.harness.EvalHarness`` driven by a
``ZeroShotBaseline`` that wraps the current model, so the baseline
output JSONs are schema-compatible with CAEM runs and with the
inference-baseline JSONs produced by ``scripts/run_baseline.py``.

Output
------
outputs/baselines/<name>/
    cycle_<n>/                -- checkpoint directory (HF save_pretrained).
    eval/<bench>_cycle<n>.json -- per-cycle, per-benchmark results.
    training_log.jsonl        -- one JSON line per cycle with train loss,
                                 MMLU pre/post, rollback flag.

Usage (vast.ai / A100)
----------------------
# B6: Vanilla FT
python -m scripts.run_simple_ft \\
    --baseline_name vanilla_ft \\
    --num_cycles 10 \\
    --n_train_per_bench 2000 \\
    --output_dir outputs/baselines

# B7: EWC-only FT
python -m scripts.run_simple_ft \\
    --baseline_name ewc_only_ft \\
    --use_l2_anchor \\
    --use_mmlu_guard \\
    --num_cycles 10 \\
    --n_train_per_bench 2000 \\
    --output_dir outputs/baselines

Thesis reference
----------------
  §5.1 Baselines -- B6 Vanilla FT, B7 EWC-only FT.
  §4.10 Training ablation design -- why B7 keeps the anchor but not the
        verifier or memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_simple_ft")


# -----------------------------------------------------------------------------
# CLI                                                                           #
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline_name", required=True,
                   help="Subdirectory name under --output_dir "
                        "(e.g. 'vanilla_ft' or 'ewc_only_ft').")
    p.add_argument("--output_dir", default="outputs/baselines")
    p.add_argument("--num_cycles", type=int, default=10)
    p.add_argument("--epochs_per_cycle", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=-1,
                   help="Gradient-accumulation multiplier. -1 (default) auto-"
                        "selects from scripts.hardware so that batch_size * "
                        "grad_accum_steps == TARGET_EFFECTIVE_BATCH_SIZE (=32) "
                        "on every tier. Pass a positive integer to override.")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--l2_lambda", type=float, default=0.01,
                   help="Anchor strength when --use_l2_anchor is set.")
    p.add_argument("--use_l2_anchor", action="store_true",
                   help="B7 EWC-only FT: add (λ/2)||θ − θ_prev||² to the loss.")
    p.add_argument("--use_mmlu_guard", action="store_true",
                   help="B7 EWC-only FT: rollback if MMLU retention drops "
                        "below --forgetting_tolerance after a cycle.")
    p.add_argument("--forgetting_tolerance", type=float, default=0.93)
    p.add_argument("--mmlu_n", type=int, default=200)
    p.add_argument("--train_benchmarks", nargs="+",
                   default=["fever", "triviaqa", "natural_questions"],
                   help="Benchmarks whose train splits supply fine-tuning data.")
    p.add_argument("--eval_benchmarks", nargs="+",
                   default=[
                       "fever", "triviaqa", "natural_questions",
                       "truthfulqa", "strategyqa", "arc_challenge",
                   ])
    p.add_argument("--n_train_per_bench", type=int, default=2000)
    p.add_argument("--n_eval_per_bench", type=int, default=500)
    p.add_argument("--model_name", default="google/flan-t5-large")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_cycle", type=int, default=0,
                   help="Skip earlier cycles; useful for recovery from crash.")
    # -- Optional STaR-style rationalisation (Zelikman et al. 2022) --
    p.add_argument("--use_rationalisation", action="store_true",
                   help="STaR ceiling baseline: at each cycle, generate "
                        "CoT rationales on the training slice, keep "
                        "forward-correct (q, rationale, answer) triples "
                        "and back-rationalise forward-wrong ones, then "
                        "fine-tune on the rationale-augmented targets.")
    p.add_argument("--rationalise_max_new_tokens", type=int, default=128,
                   help="Max new tokens when generating rationales.")
    p.add_argument("--rationalise_batch_size", type=int, default=8,
                   help="Batch size for the rationalisation generation pass.")
    p.add_argument("--caem_splits_path", default="",
                   help=(
                       "Optional path to the CAEM main-run's "
                       "dataset_splits.json (e.g. "
                       "outputs/full_run/dataset_splits.json). When supplied, "
                       "the per-cycle EVAL pool is filtered to the exact "
                       "eval_ids CAEM used, guaranteeing the sig-test paired "
                       "overlap is the full eval slice. The TRAINING pool is "
                       "NOT filtered -- training data draws from the full "
                       "train split independently. Empty string disables "
                       "the filter (smoke runs)."
                   ))
    return p.parse_args()


# -----------------------------------------------------------------------------
# Training-data assembly                                                        #
# -----------------------------------------------------------------------------

def _q_hash(text: str) -> str:
    """Stable 16-hex-char sha256 prefix of a trimmed question string.

    Used by the train/eval disjointness guard to compare pools by question
    TEXT rather than by sample id. Comparing by id is fragile: some HF
    datasets (notably nq_open) use numeric `question_id` values that happen
    to collide across splits even though the underlying questions are
    distinct -- which would cause the guard to raise a false-positive
    RuntimeError. Hashing the question string is immune to this.
    """
    import hashlib
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def _load_train_pool(
    ns: argparse.Namespace,
) -> Tuple[List, Dict[str, set]]:
    """Load raw (question, answer) pairs from each training benchmark.

    Always pulls from ``split="train"`` -- the previous FEVER-only special
    case silently pulled TriviaQA and NQ from their ``validation`` splits,
    which are the same splits the per-cycle eval loop measures on. The
    train/eval disjointness invariant asserted by the caller assumes all
    training benchmarks are loaded from their dedicated train splits.

    Returns
    -------
    (pool, train_qhashes_by_benchmark)
        pool: shuffled flat list of QAPair instances.
        train_qhashes_by_benchmark: {benchmark: set(question_hash)} used by
            the caller to assert train/eval disjointness by question TEXT
            (not by id) against the eval pool loaded at cycle-0 startup.
    """
    from caem.training.self_improvement import QAPair
    from eval.benchmarks import load_benchmark

    pool: List[QAPair] = []
    train_qhashes_by_bm: Dict[str, set] = {}
    for bench in ns.train_benchmarks:
        n = ns.n_train_per_bench
        logger.info("Loading %d train-split samples from %s ...", n, bench)
        samples = load_benchmark(bench, n=n, split="train")
        train_qhashes_by_bm[bench] = {_q_hash(s["question"])
                                      for s in samples
                                      if s.get("question")}
        for s in samples:
            answer = s["answers"][0] if s.get("answers") else ""
            if s["question"] and answer:
                pool.append(QAPair(question=s["question"], answer=answer))
    random.Random(ns.seed).shuffle(pool)
    logger.info("Total training pool: %d pairs across %d benchmarks.",
                len(pool), len(ns.train_benchmarks))
    return pool, train_qhashes_by_bm


# -----------------------------------------------------------------------------
# One fine-tuning cycle                                                         #
# -----------------------------------------------------------------------------

def _finetune_one_cycle(
    model, tokenizer, train_pairs, ns: argparse.Namespace, device: str,
) -> Tuple[float, int, Optional[list]]:
    """Run ns.epochs_per_cycle epochs of fine-tuning.

    Returns (final_loss, epochs_completed, theta_prev_snapshot_or_None).
    theta_prev_snapshot is captured once before the first epoch so the
    MMLU guard can roll back if necessary.
    """
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from caem.training.self_improvement import QADataset

    # Snapshot θ_prev ONLY if we need rollback or the L2 anchor.
    theta_prev = None
    if ns.use_l2_anchor or ns.use_mmlu_guard:
        theta_prev = [p.detach().clone().cpu().float()
                      for p in model.parameters()]

    dataset = QADataset(train_pairs, tokenizer)

    def _collate(batch):
        pad_id = tokenizer.pad_token_id or 0
        max_in  = max(b["input_ids"].shape[0] for b in batch)
        max_tgt = max(b["labels"].shape[0]    for b in batch)
        input_ids = torch.full((len(batch), max_in),  pad_id, dtype=torch.long)
        attn_mask = torch.zeros(len(batch), max_in,          dtype=torch.long)
        labels    = torch.full((len(batch), max_tgt), -100,  dtype=torch.long)
        for i, b in enumerate(batch):
            il, tl = b["input_ids"].shape[0], b["labels"].shape[0]
            input_ids[i, :il] = b["input_ids"]
            attn_mask[i, :il] = b["attention_mask"]
            labels[i, :tl]    = b["labels"]
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

    loader = DataLoader(dataset, batch_size=ns.batch_size, shuffle=True,
                        collate_fn=_collate)
    optimizer = AdamW(model.parameters(), lr=ns.learning_rate)

    # L2 anchor snapshot placement: keep on GPU when VRAM ≥ 24 GB.
    anchor_tensors = None
    if ns.use_l2_anchor and theta_prev is not None:
        if str(device).startswith("cuda"):
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            except Exception:
                vram_gb = 0.0
            if vram_gb >= 24.0:
                anchor_tensors = [p0.to(device, dtype=torch.float32)
                                  for p0 in theta_prev]
            else:
                anchor_tensors = theta_prev
        else:
            anchor_tensors = theta_prev

    model.train()
    model.to(device)

    param_dtype = next(model.parameters()).dtype
    use_cuda = str(device).startswith("cuda")
    amp_enabled = use_cuda and param_dtype in (torch.float16, torch.bfloat16)
    amp_dtype = torch.float16 if param_dtype == torch.float16 else torch.bfloat16
    # Modern API (torch>=2.4): torch.amp.GradScaler("cuda", ...). The old
    # torch.cuda.amp.GradScaler(...) still works but emits a DeprecationWarning.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )

    epochs_done, final_loss = 0, 0.0
    # Gradient-accumulation multiplier (1 = no-op). Optimiser step is taken
    # every `grad_accum_steps` micro-batches so effective batch size equals
    # `ns.batch_size * ns.grad_accum_steps`, which is pinned to the thesis
    # target (=32) by the main() auto-resolution. Loss is scaled by 1/N so
    # the summed gradient matches a single step at the effective batch size.
    grad_accum_steps = max(int(getattr(ns, "grad_accum_steps", 1) or 1), 1)
    for epoch in range(ns.epochs_per_cycle):
        epoch_loss, n_batches = 0.0, 0
        micro_step = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)
            if int((labels != -100).sum().item()) == 0:
                continue

            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=amp_enabled):
                outputs = model(input_ids=input_ids,
                                attention_mask=attn_mask, labels=labels)
                ce_loss = outputs.loss
            if not torch.isfinite(ce_loss):
                # Discard any partial accumulation to avoid poisoning the step.
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0
                continue

            if anchor_tensors is not None:
                l2 = torch.tensor(0.0, device=device)
                for p, p0 in zip(model.parameters(), anchor_tensors):
                    ref = p0.to(device, dtype=torch.float32)
                    l2 = l2 + ((p.float() - ref) ** 2).sum()
                raw_loss = ce_loss + (ns.l2_lambda / 2.0) * l2
            else:
                raw_loss = ce_loss

            if not torch.isfinite(raw_loss):
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0
                continue

            loss = raw_loss / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=ns.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=ns.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

            epoch_loss += raw_loss.item()
            n_batches += 1

        # Tail: epoch ended mid-accumulation — step on the partial batch
        # so no training data is wasted. At most one partial step per epoch.
        if micro_step > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=ns.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=ns.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0

        avg = epoch_loss / max(n_batches, 1)
        logger.info("  Epoch %d/%d -- loss: %.4f (grad_accum=%d, eff_batch=%d)",
                    epoch + 1, ns.epochs_per_cycle, avg,
                    grad_accum_steps, ns.batch_size * grad_accum_steps)
        epochs_done += 1
        final_loss = avg

    model.eval()
    return final_loss, epochs_done, theta_prev


# -----------------------------------------------------------------------------
# MMLU retention                                                                #
# -----------------------------------------------------------------------------

def _mmlu_accuracy(model, tokenizer, device: str, n: int, seed: int) -> float:
    """Evaluate 4-choice MMLU accuracy on an n-sample shuffled subset.

    Matches the CAEM MMLU probe: shuffled seed 42 across 57 subjects,
    greedy decode, letter extraction (A/B/C/D).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets package unavailable; returning NaN MMLU.")
        return float("nan")
    try:
        import torch
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as exc:
        logger.warning("MMLU load failed (%s); returning NaN.", exc)
        return float("nan")

    letters = ["A", "B", "C", "D"]
    correct = 0
    model.eval()
    with torch.no_grad():
        for ex in ds:
            ex_d: dict = ex  # type: ignore[assignment]
            prompt = (
                f"Question: {ex_d['question']}\n"
                + "\n".join(f"{L}. {c}" for L, c in zip(letters, ex_d["choices"]))
                + "\nAnswer:"
            )
            enc = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=512).to(device)
            out = model.generate(enc["input_ids"], max_new_tokens=8,
                                 do_sample=False)
            pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            # First A/B/C/D character wins.
            pred_letter = next(
                (ch for ch in pred if ch in letters),
                "",
            )
            gold_letter = letters[int(ex_d["answer"])]
            if pred_letter == gold_letter:
                correct += 1
    return correct / max(len(ds), 1)


# -----------------------------------------------------------------------------
# STaR rationalisation (Zelikman et al. 2022)                                   #
# -----------------------------------------------------------------------------

# Prompt prefixes used for forward generation and back-rationalisation.
_COT_PREFIX = "Let's think step by step."
_RATIONALISE_TEMPLATE = (
    "Question: {q}\n"
    "The correct answer is: {a}\n"
    "Explain step by step why this answer is correct, then restate the answer."
)


def _answer_matches(pred: str, gold: str) -> bool:
    """Token-insensitive match used to filter forward-correct rationales."""
    def _norm(x: str) -> str:
        return "".join(c.lower() for c in x if c.isalnum())
    if not pred or not gold:
        return False
    p, g = _norm(pred), _norm(gold)
    return p == g or (len(g) >= 3 and g in p)


def _rationalise_pool(
    model,
    tokenizer,
    pairs: list,
    device: str,
    max_new_tokens: int,
    batch_size: int,
) -> list:
    """Apply STaR's forward-filter + back-rationalise pass to a QAPair list.

    For each (q, gold) pair:
      1. Forward pass: prompt ``CoT + Question`` and greedy-decode a
         (rationale, answer) string. If the decoded answer matches gold,
         keep the full rationale-augmented target.
      2. Back-rationalise: for mismatched pairs, prompt the
         :data:`_RATIONALISE_TEMPLATE` that reveals the gold answer and
         asks for reasoning conditioned on it; keep the rationalised
         target.

    Returns a new list of :class:`QAPair` with the ``answer`` field
    replaced by ``"{rationale}\\nAnswer: {gold}"`` so downstream
    supervised fine-tuning regresses onto the rationale-augmented
    string without any other code change.
    """
    from caem.training.self_improvement import QAPair
    import torch

    model.eval()
    out_pairs: list = []
    kept_forward = kept_rationalised = 0

    # --- Pass 1: forward generation with CoT prefix ---
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        prompts = [f"{_COT_PREFIX}\nQuestion: {p.question}" for p in batch]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        for pair, dec in zip(batch, decoded):
            if _answer_matches(dec, pair.answer):
                # Keep the model's own rationale-augmented target.
                out_pairs.append(QAPair(
                    question=pair.question,
                    answer=f"{dec.strip()}\nAnswer: {pair.answer}",
                ))
                kept_forward += 1
            else:
                out_pairs.append(None)  # placeholder -- back-rationalise below

    # --- Pass 2: back-rationalise forward-wrong samples ---
    for i, entry in enumerate(out_pairs):
        if entry is not None:
            continue
        # Skipping batching here keeps the code simple; back-rationalisation
        # is typically a small fraction of the pool after CoT filtering.
        pair = pairs[i]
        prompt = _RATIONALISE_TEMPLATE.format(q=pair.question, a=pair.answer)
        enc = tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                enc["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        rationale = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        out_pairs[i] = QAPair(
            question=pair.question,
            answer=f"{rationale}\nAnswer: {pair.answer}",
        )
        kept_rationalised += 1

    logger.info(
        "STaR rationalisation: %d forward-correct + %d back-rationalised = %d / %d kept.",
        kept_forward, kept_rationalised, len(out_pairs), len(pairs),
    )
    return [p for p in out_pairs if p is not None]


# -----------------------------------------------------------------------------
# Main                                                                          #
# -----------------------------------------------------------------------------

def main() -> None:
    ns = _parse_args()
    random.seed(ns.seed)

    # Resolve --grad_accum_steps sentinel from the hardware profile so that
    # `batch_size * grad_accum_steps` matches the thesis effective batch
    # target on whatever hardware the script is invoked on. Explicit ints
    # override the auto value. 1 is a no-op on the thesis 5090 path.
    if ns.grad_accum_steps is None or ns.grad_accum_steps < 1:
        try:
            from scripts.hardware import get_hardware_profile
            _hw_profile = get_hardware_profile()
            ns.grad_accum_steps = int(_hw_profile.grad_accum_steps)
        except Exception as _exc:  # pragma: no cover -- defensive
            logger.warning(
                "Hardware-profile lookup failed (%s); using grad_accum_steps=1.",
                _exc,
            )
            ns.grad_accum_steps = 1
    logger.info(
        "Gradient accumulation: batch_size=%d, grad_accum_steps=%d "
        "(effective batch=%d).",
        ns.batch_size, ns.grad_accum_steps,
        ns.batch_size * ns.grad_accum_steps,
    )

    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from eval.baselines import ZeroShotBaseline
    from eval.benchmarks import load_benchmark
    from eval.harness import EvalHarness

    dtype_map = {"float32": torch.float32,
                 "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[ns.dtype]

    out_root = Path(ns.output_dir) / ns.baseline_name
    eval_root = out_root / "eval"
    out_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "training_log.jsonl"

    # Optional CAEM-split filter for the EVAL pool only (NOT training).
    # Mirrors run_purity_validation.py::load_and_filter so B6/B7 evaluate on
    # the exact eval_ids CAEM used, guaranteeing 1:1 sig-test pairing.
    eval_ids_by_bm: Dict[str, set] = {}
    if ns.caem_splits_path:
        import json as _json
        splits_path = Path(ns.caem_splits_path)
        if splits_path.is_file():
            try:
                with splits_path.open("r", encoding="utf-8") as f:
                    splits = _json.load(f)
                for bm, meta in splits.items():
                    if isinstance(meta, dict):
                        eval_ids_by_bm[bm] = {str(i) for i
                                              in meta.get("eval_ids", [])}
                logger.info("Loaded eval_ids from %s for %d benchmarks.",
                            splits_path, len(eval_ids_by_bm))
            except Exception as exc:
                logger.warning("Failed to parse %s (%s). Proceeding without "
                               "eval_id filter.", splits_path, exc)
        else:
            logger.warning("--caem_splits_path %s not found. Proceeding "
                           "without eval_id filter.", splits_path)

    logger.info("Loading %s ...", ns.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ns.model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        ns.model_name, torch_dtype=dtype
    ).to(ns.device)

    # Pre-cycle MMLU baseline (needed for retention ratio even if guard off,
    # because the RET axis of CES consumes it).
    pristine_mmlu = _mmlu_accuracy(
        model, tokenizer, ns.device, n=ns.mmlu_n, seed=ns.seed,
    )
    logger.info("Pre-cycle MMLU: %.4f", pristine_mmlu)

    # Keep a copy of pristine MMLU for retention ratio computation.
    # pre_mmlu will be updated per-cycle but pristine_mmlu stays fixed.
    pre_mmlu = pristine_mmlu

    train_pool, train_qhashes_by_bm = _load_train_pool(ns)
    per_cycle_pool_size = max(len(train_pool) // ns.num_cycles, ns.batch_size)

    # -- Mandatory train/eval disjointness invariant ---------------------- #
    # Guards against a loader-default change silently re-introducing the
    # train/eval leak that the split="train" fix just closed. Compares
    # the training-pool question TEXT (sha256 prefix) against the question
    # text of the eval slice CAEM used. Comparing by question text (not by
    # id) is immune to the fact that some HF datasets (notably nq_open) use
    # numeric question_id values that collide across train/val splits even
    # though the underlying questions are distinct.
    #
    # Raises RuntimeError (NOT assert -- survives `python -O`). Skipped
    # only when --caem_splits_path is empty or absent; the skip path logs
    # loudly so it's visible in the run log.
    _EVAL_SPLIT_MAP = {
        "fever": "dev",
        "triviaqa": "validation",
        "natural_questions": "validation",
        "strategyqa": "test",
        "arc_challenge": "test",
        # truthfulqa: no split kwarg needed; load_benchmark dispatch returns
        # the validation pool by default and truthfulqa isn't in the ID
        # training set anyway, so this benchmark won't hit the guard.
    }
    if eval_ids_by_bm:
        for bm, train_hashes in train_qhashes_by_bm.items():
            e_ids = eval_ids_by_bm.get(bm)
            if not e_ids:
                logger.info(
                    "train/eval disjointness check: SKIPPED for %s "
                    "(benchmark not in dataset_splits.json eval_ids).", bm)
                continue
            eval_split = _EVAL_SPLIT_MAP.get(bm)
            if eval_split:
                eval_samples = load_benchmark(bm, n=ns.n_eval_per_bench,
                                              split=eval_split)
            else:
                eval_samples = load_benchmark(bm, n=ns.n_eval_per_bench)
            # Restrict to CAEM's exact eval ids where possible; if the
            # filter leaves the set empty, fall back to the unfiltered pool
            # so the invariant still fires (but log the degradation).
            filtered = [s for i, s in enumerate(eval_samples)
                        if str(s.get("id", i)) in e_ids]
            if not filtered:
                logger.warning(
                    "train/eval disjointness check: %s filtered to 0 "
                    "samples via eval_ids; falling back to unfiltered "
                    "eval pool for the check.", bm)
                filtered = eval_samples
            eval_hashes = {_q_hash(s["question"])
                           for s in filtered
                           if s.get("question")}
            overlap = train_hashes & eval_hashes
            if overlap:
                raise RuntimeError(
                    f"train/eval LEAKAGE detected for {bm}: "
                    f"{len(overlap)} question-text hashes appear in both "
                    f"the training pool (split='train', "
                    f"n_train_qhashes={len(train_hashes)}) and the eval "
                    f"pool (n_eval_qhashes={len(eval_hashes)}). This "
                    f"indicates a loader regression re-introducing the "
                    f"bug closed by the split='train' fix. Refusing to "
                    f"proceed to cycle 1."
                )
            logger.info(
                "train/eval disjoint verified: %s N_train_q=%d N_eval_q=%d "
                "overlap=0.", bm, len(train_hashes), len(eval_hashes))
    else:
        logger.warning(
            "train/eval disjointness check: SKIPPED entirely -- no "
            "--caem_splits_path provided or dataset_splits.json missing. "
            "This is acceptable ONLY for smoke runs where the CAEM main "
            "run has not written its splits file yet. Phase 1A real runs "
            "MUST pass --caem_splits_path so this invariant fires.")

    for cycle in range(1, ns.num_cycles + 1):
        if cycle <= ns.resume_from_cycle:
            logger.info("Cycle %d: skipped (--resume_from_cycle)", cycle)
            continue

        logger.info("=" * 72)
        logger.info("CYCLE %d / %d -- %s", cycle, ns.num_cycles, ns.baseline_name)
        logger.info("=" * 72)

        # Per-cycle slice of the pool (without replacement across cycles if
        # possible, else wrap). This is B6/B7's deliberate contract: partition
        # the shuffled-once pool into num_cycles disjoint windows so each
        # sample is seen exactly once across the full run. Main CAEM instead
        # resamples general_data per cycle and mixes in verifier-selected
        # episodes; B7 is the "no verifier, no memory" baseline, so the fixed
        # partitioning is the correct counterpart to CAEM's verifier-driven
        # cycle composition. The DataLoader inside _finetune_one_cycle
        # shuffles within the window, so within-cycle order is randomised.
        # The modulo wrap is a defensive guard for pathological configs
        # (num_cycles x per_cycle > pool_size); thesis configs never trigger
        # it. The min(..., len(train_pool)) clamp may yield a short final
        # cycle if pool size is not a multiple of per_cycle_pool_size --
        # log a debug line so the truncation is visible.
        start = ((cycle - 1) * per_cycle_pool_size) % max(len(train_pool), 1)
        end = min(start + per_cycle_pool_size, len(train_pool))
        cycle_pairs = train_pool[start:end]
        logger.info("Cycle %d: training on %d pairs.", cycle, len(cycle_pairs))
        if len(cycle_pairs) < per_cycle_pool_size:
            logger.debug(
                "Cycle %d: short batch (%d < per_cycle_pool_size=%d) -- "
                "pool_size=%d is not divisible by num_cycles=%d.",
                cycle, len(cycle_pairs), per_cycle_pool_size,
                len(train_pool), ns.num_cycles,
            )

        # Optional STaR rationalisation pass: regenerate training targets
        # as (rationale + answer) strings using the current weights before
        # the fine-tune step. Runs on every cycle so the rationales are
        # always generated by the most recent checkpoint, matching
        # Zelikman et al. (2022) §3.
        if ns.use_rationalisation:
            logger.info("Cycle %d: applying STaR rationalisation pass ...", cycle)
            t_rat = time.perf_counter()
            cycle_pairs = _rationalise_pool(
                model, tokenizer, cycle_pairs, ns.device,
                max_new_tokens=ns.rationalise_max_new_tokens,
                batch_size=ns.rationalise_batch_size,
            )
            logger.info("Cycle %d: rationalisation took %.1fs, %d pairs remain.",
                        cycle, time.perf_counter() - t_rat, len(cycle_pairs))

        t0 = time.perf_counter()
        final_loss, epochs_done, theta_prev = _finetune_one_cycle(
            model, tokenizer, cycle_pairs, ns, ns.device,
        )
        train_seconds = time.perf_counter() - t0

        # MMLU retention probe.
        post_mmlu = _mmlu_accuracy(
            model, tokenizer, ns.device, n=ns.mmlu_n, seed=ns.seed,
        )
        # If either probe is NaN or the pristine baseline is ~0, report NaN
        # (not 1.0). Silently substituting 1.0 would paper over a genuinely
        # broken MMLU probe by claiming perfect retention, defeating the
        # forgetting-guard rollback below. The guard branch already skips
        # rollback when `retention_ratio < forgetting_tolerance` is False
        # (NaN comparisons are False), so NaN propagation is safe.
        if math.isnan(pristine_mmlu) or math.isnan(post_mmlu) or pristine_mmlu <= 1e-6:
            retention_ratio = float("nan")
        else:
            retention_ratio = post_mmlu / pristine_mmlu

        aborted = False
        if ns.use_mmlu_guard and retention_ratio < ns.forgetting_tolerance \
                and theta_prev is not None:
            logger.warning(
                "Cycle %d: retention %.4f < %.4f -- ROLLBACK.",
                cycle, retention_ratio, ns.forgetting_tolerance,
            )
            with torch.no_grad():
                for p, p0 in zip(model.parameters(), theta_prev):
                    p.data.copy_(p0.to(p.device, dtype=p.dtype))
            aborted = True

        # Checkpoint.
        ckpt_dir = out_root / f"cycle_{cycle}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

        # Evaluate the cycle weights on the six-benchmark panel (3 ID + 3 OOD).
        logger.info("Cycle %d: evaluating on %d benchmarks ...",
                    cycle, len(ns.eval_benchmarks))
        baseline = ZeroShotBaseline.__new__(ZeroShotBaseline)
        baseline.model_name = ns.model_name
        baseline.device = ns.device
        baseline.max_new_tokens = 256
        baseline.max_input_tokens = 512
        baseline.tokenizer = tokenizer
        baseline.model = model
        harness = EvalHarness(
            pipeline=baseline,
            output_dir=str(eval_root),
            log_every=100,
            fail_on_error=False,
        )
        for bench in ns.eval_benchmarks:
            if bench == "fever":
                samples = load_benchmark(bench, n=ns.n_eval_per_bench,
                                         split="dev")
            else:
                samples = load_benchmark(bench, n=ns.n_eval_per_bench)
            allowed = eval_ids_by_bm.get(bench)
            if allowed:
                before = len(samples)
                samples = [s for i, s in enumerate(samples)
                           if str(s.get("id", i)) in allowed]
                logger.info("eval_id filter: %s kept %d / %d samples (cycle %d).",
                            bench, len(samples), before, cycle)
                if not samples:
                    logger.warning("eval_id filter left 0 samples for %s; "
                                   "reloading unfiltered slice to avoid "
                                   "empty-eval crash.", bench)
                    if bench == "fever":
                        samples = load_benchmark(bench, n=ns.n_eval_per_bench,
                                                 split="dev")
                    else:
                        samples = load_benchmark(bench, n=ns.n_eval_per_bench)
            harness.run(benchmark=bench, samples=samples, cycle=cycle,
                        store_to_memory=False)

        # Log the cycle.
        log_entry = {
            "baseline": ns.baseline_name,
            "cycle": cycle,
            "n_train_pairs": len(cycle_pairs),
            "epochs": epochs_done,
            "final_train_loss": final_loss,
            "train_seconds": train_seconds,
            "pristine_mmlu": pristine_mmlu,
            "pre_mmlu": pre_mmlu,
            "post_mmlu": post_mmlu,
            "retention_ratio": retention_ratio,
            "aborted": aborted,
            "use_l2_anchor": ns.use_l2_anchor,
            "use_mmlu_guard": ns.use_mmlu_guard,
            "use_rationalisation": ns.use_rationalisation,
            "checkpoint": str(ckpt_dir),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Update pre_mmlu to post-cycle value (used as next cycle's anchor
        # reference for retention ratio).
        pre_mmlu = post_mmlu

    logger.info("All %d cycles complete. Checkpoints: %s", ns.num_cycles, out_root)


if __name__ == "__main__":
    main()
