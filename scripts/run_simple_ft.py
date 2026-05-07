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
import os
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
    p.add_argument("--batch_size", type=int, default=16,
                   help="Training DataLoader batch_size (not eval). See --eval_batch_size for eval.")
    p.add_argument("--eval_batch_size", type=int, default=1,
                   help=("Goal 5 Level B batch size for the post-cycle eval. "
                         "bs=1 keeps the serial path; bs=8-32 uses the baseline's "
                         "native answer_batch for ~1.5-2x eval speedup."))
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
    # v2 Fix 9b: read defaults from caem.config so the v1 ↔ v2 panel
    # change does not silently leave the B6/B7 baselines training on the
    # old roster.
    from caem.config import (
        TRAINING_BENCHMARKS as _CFG_TRAINING_BENCHMARKS,
        TRANSFER_BENCHMARKS as _CFG_TRANSFER_BENCHMARKS,
    )
    p.add_argument("--train_benchmarks", nargs="+",
                   default=list(_CFG_TRAINING_BENCHMARKS),
                   help="Benchmarks whose train splits supply fine-tuning data.")
    p.add_argument("--eval_benchmarks", nargs="+",
                   default=list(_CFG_TRAINING_BENCHMARKS) + list(_CFG_TRANSFER_BENCHMARKS))
    p.add_argument("--n_train_per_bench", type=int, default=2000)
    p.add_argument("--n_eval_per_bench", type=int, default=500)
    p.add_argument(
        "--model_name",
        default=None,
        help=(
            "HF model ID. Defaults to CAEMConfig.base_model_name "
            "(Qwen/Qwen2.5-3B-Instruct on Branch C)."
        ),
    )
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
    # --caem_splits_path removed 2026-04-22 evening. Pool alignment is now
    # deterministic via caem.benchmark_splits.build_all_benchmark_pools
    # (rng_seed=42 matches run_experiment.py). B6/B7 eval + train pools are
    # byte-identical to CAEM's cycle-N pools by construction.
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

    Branch C 2026-04-22 evening: pool now drawn from the SAME deterministic
    content-hash split as CAEM's run_experiment.py stream chunks, via
    caem.benchmark_splits.build_all_benchmark_pools. This guarantees B6/B7
    see byte-identical per-cycle chunks to CAEM's Step 4 -- critical for
    matched-scale comparison in the Ch5 sig-tests. The legacy
    `load_benchmark(bench, n=n, split="train")` path is replaced because it
    drew an independent random sample; the pool builder's content-hash split
    enforces disjointness with CAEM's calib/purity/test/eval pools.

    Returns
    -------
    (pool, train_qhashes_by_benchmark, cycle_chunks_by_bm)
        pool: shuffled flat list of QAPair instances (concatenated chunks).
        train_qhashes_by_benchmark: {benchmark: set(question_hash)} for
            disjointness assertions.
        cycle_chunks_by_bm: {benchmark: [chunk_1, chunk_2, ...]} where each
            chunk is a list of QAPair. Used by the cycle loop to draw the
            exact same samples as CAEM cycle N.
    """
    from caem.training.self_improvement import QAPair
    from caem.benchmark_splits import (
        build_all_benchmark_pools, TRAINING_BENCHMARKS,
    )

    # Only training-eligible benchmarks contribute SIL pool samples.
    # Transfer benchmarks (truthfulqa, strategyqa, arc_challenge, asqa) have
    # empty sil_train_chunks and are skipped by the pool builder.
    train_capable = [bm for bm in ns.train_benchmarks if bm in TRAINING_BENCHMARKS]
    if not train_capable:
        raise RuntimeError(
            f"No train-capable benchmarks in --train_benchmarks={ns.train_benchmarks}. "
            f"Must include at least one of {TRAINING_BENCHMARKS}."
        )

    # 2026-05-07 audit fix: drop the n_train_per_bench-derived chunk_size
    # override so build_benchmark_pools consults PER_BENCHMARK_TRAIN_CHUNK_SIZE
    # per benchmark. With chunk_size=n_train_per_bench/n_cycles=3000 explicit,
    # CSQA failed with InsufficientBenchmarkDataError because 1000+500+500+
    # 10×3000 = 32000 needed vs 9741 available.
    #
    # The matched-scale invariant (B6/B7 train on the same chunks as CAEM
    # step_7_main) is now PRESERVED automatically because both go through
    # the same build_benchmark_pools per-bench dispatch (FEVER/TQA/HotpotQA
    # at 1000-chunk × 10 cycles = 10000; CSQA at 700-chunk × 10 = 7000).
    # The --n_train_per_bench CLI flag is retained for back-compat with
    # older runners but its value is now informational only when the bench
    # has a per-bench override registered.
    logger.info(
        "Building benchmark pools for B6/B7 training: %s (n_cycles=%d, "
        "per-bench chunk_size from PER_BENCHMARK_TRAIN_CHUNK_SIZE)",
        train_capable, ns.num_cycles,
    )
    benchmark_pools = build_all_benchmark_pools(
        benchmarks=train_capable,
        n_cycles=int(ns.num_cycles),
        eval_size=500,  # not used here; eval loop uses benchmark_pools[bm].eval
        rng_seed=int(ns.seed),
    )

    pool: List[QAPair] = []
    train_qhashes_by_bm: Dict[str, set] = {}
    cycle_chunks_by_bm: Dict[str, List[List[QAPair]]] = {}
    for bench in train_capable:
        pools = benchmark_pools[bench]
        per_cycle_pairs: List[List[QAPair]] = []
        bench_pairs: List[QAPair] = []
        for chunk in pools.sil_train_chunks:
            cycle_pairs_this = []
            for s in chunk:
                answer = s["answers"][0] if s.get("answers") else ""
                if s.get("question") and answer:
                    qp = QAPair(question=s["question"], answer=answer)
                    cycle_pairs_this.append(qp)
                    bench_pairs.append(qp)
            per_cycle_pairs.append(cycle_pairs_this)
        cycle_chunks_by_bm[bench] = per_cycle_pairs
        pool.extend(bench_pairs)
        train_qhashes_by_bm[bench] = {
            _q_hash(s["question"]) for chunk in pools.sil_train_chunks
            for s in chunk if s.get("question")
        }
        logger.info(
            "  %s: %d chunks × ~%d pairs = %d total (matches CAEM cycle chunks exactly).",
            bench, len(per_cycle_pairs),
            len(per_cycle_pairs[0]) if per_cycle_pairs else 0,
            len(bench_pairs),
        )

    random.Random(ns.seed).shuffle(pool)
    logger.info("Total training pool: %d pairs across %d benchmarks.",
                len(pool), len(train_capable))
    return pool, train_qhashes_by_bm, cycle_chunks_by_bm


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
            user_content = (
                f"Question: {ex_d['question']}\n"
                + "\n".join(f"{L}. {c}" for L, c in zip(letters, ex_d["choices"]))
                + "\nAnswer with just the letter (A, B, C, or D)."
            )
            # ChatML-wrap for decoder-only instruction-tuned backbones
            # (Qwen/Gemma/Llama). Falls back to raw prompt when the
            # tokenizer has no chat template.
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_content}],
                        tokenize=False, add_generation_prompt=True,
                    )
                    if not isinstance(prompt, str):
                        prompt = user_content
                except Exception:
                    prompt = user_content
            else:
                prompt = user_content
            enc = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=1024,
                            add_special_tokens=False).to(device)
            input_ids = enc["input_ids"]
            input_len = int(input_ids.shape[1])
            out = model.generate(input_ids, max_new_tokens=8,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            # Decoder-only slice: generate() echoes the prompt in out[0].
            gen_ids = out[0, input_len:] if out.shape[1] > input_len else out[0]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred_letter = next(
                (ch for ch in pred.upper() if ch in letters),
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

    def _chatml_user(user_content: str) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False, add_generation_prompt=True,
                )
                if isinstance(rendered, str):
                    return rendered
            except Exception:
                pass
        return user_content

    # --- Pass 1: forward generation with CoT prefix (ChatML-wrapped) ---
    # Left-padding is mandatory for decoder-only batched generation so the
    # generated continuations align regardless of input length.
    original_side = getattr(tokenizer, "padding_side", None)
    try:
        tokenizer.padding_side = "left"
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            prompts = [
                _chatml_user(f"{_COT_PREFIX}\nQuestion: {p.question}")
                for p in batch
            ]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=1024,
                add_special_tokens=False,
            ).to(device)
            input_len = int(enc["input_ids"].shape[1])
            with torch.no_grad():
                out_ids = model.generate(
                    enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # Slice the generated continuation off each row.
            gen_ids = (
                out_ids[:, input_len:]
                if out_ids.shape[1] > input_len
                else out_ids
            )
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for pair, dec in zip(batch, decoded):
                if _answer_matches(dec, pair.answer):
                    out_pairs.append(QAPair(
                        question=pair.question,
                        answer=f"{dec.strip()}\nAnswer: {pair.answer}",
                    ))
                    kept_forward += 1
                else:
                    out_pairs.append(None)  # back-rationalise below

        # --- Pass 2: back-rationalise forward-wrong samples ---
        for i, entry in enumerate(out_pairs):
            if entry is not None:
                continue
            pair = pairs[i]
            prompt = _chatml_user(
                _RATIONALISE_TEMPLATE.format(q=pair.question, a=pair.answer)
            )
            enc = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=1024,
                            add_special_tokens=False).to(device)
            input_len = int(enc["input_ids"].shape[1])
            with torch.no_grad():
                out_ids = model.generate(
                    enc["input_ids"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = (
                out_ids[0, input_len:]
                if out_ids.shape[1] > input_len
                else out_ids[0]
            )
            rationale = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            out_pairs[i] = QAPair(
                question=pair.question,
                answer=f"{rationale}\nAnswer: {pair.answer}",
            )
            kept_rationalised += 1
    finally:
        if original_side is not None:
            tokenizer.padding_side = original_side

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

    from caem.config import CAEMConfig
    from caem.model_loader import load_base_generator
    from eval.baselines import ZeroShotBaseline
    from eval.benchmarks import load_benchmark
    from eval.harness import EvalHarness

    dtype_map = {"float32": torch.float32,
                 "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[ns.dtype]

    # Resolve --model_name default from CAEMConfig so simple-FT baselines
    # track the Branch-C backbone unless the caller overrides.
    resolved_model_name = ns.model_name or CAEMConfig().base_model_name

    out_root = Path(ns.output_dir) / ns.baseline_name
    eval_root = out_root / "eval"
    out_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "training_log.jsonl"

    # Branch C 2026-04-22 evening: eval_ids_by_bm superseded by
    # benchmark_pools[bm].eval (deterministic via rng_seed=42 in
    # caem.benchmark_splits.build_all_benchmark_pools). B6/B7 draw eval
    # samples directly from the pool builder in the cycle loop below, so
    # the legacy dataset_splits.json path is no longer needed. Kept
    # eval_ids_by_bm empty for back-compat with downstream code that
    # `gets` from it (all `.get()` calls return None, correctly skipping
    # legacy filter logic).
    eval_ids_by_bm: Dict[str, set] = {}

    logger.info("Loading %s ...", resolved_model_name)
    model, tokenizer = load_base_generator(
        resolved_model_name,
        device=ns.device,
        dtype=dtype,
        use_flash_attention_2=False,  # script stays driver-agnostic
        use_torch_compile=False,
    )

    # Pre-cycle MMLU baseline (needed for retention ratio even if guard off,
    # because the RET axis of CES consumes it).
    pristine_mmlu = _mmlu_accuracy(
        model, tokenizer, ns.device, n=ns.mmlu_n, seed=ns.seed,
    )
    logger.info("Pre-cycle MMLU: %.4f", pristine_mmlu)

    # Keep a copy of pristine MMLU for retention ratio computation.
    # pre_mmlu will be updated per-cycle but pristine_mmlu stays fixed.
    pre_mmlu = pristine_mmlu

    train_pool, train_qhashes_by_bm, cycle_chunks_by_bm = _load_train_pool(ns)
    per_cycle_pool_size = max(len(train_pool) // ns.num_cycles, ns.batch_size)

    # -- Mandatory train/eval disjointness invariant ---------------------- #
    # Branch C 2026-04-22 evening: train/eval disjointness is now ENFORCED BY
    # CONSTRUCTION via caem.benchmark_splits.build_all_benchmark_pools +
    # assert_no_leakage + assert_cross_benchmark_disjoint. The legacy runtime
    # check (question-text sha256 overlap between train_qhashes and eval_ids)
    # is redundant because content-hash disjointness is a pool-builder
    # invariant that raises PoolLeakageError at build time if violated.
    # Retained `train_qhashes_by_bm` as a local artefact for forward
    # compatibility with any debug tooling; the disjointness assertion
    # block itself is deleted.

    for cycle in range(1, ns.num_cycles + 1):
        if cycle <= ns.resume_from_cycle:
            logger.info("Cycle %d: skipped (--resume_from_cycle)", cycle)
            continue

        logger.info("=" * 72)
        logger.info("CYCLE %d / %d -- %s", cycle, ns.num_cycles, ns.baseline_name)
        logger.info("=" * 72)

        # Branch C 2026-04-22 evening: draw this cycle's training pairs from
        # cycle_chunks_by_bm, which maps 1:1 to CAEM's run_experiment.py
        # sil_train_chunks[cycle - 1]. B6/B7 now train on byte-identical
        # per-cycle samples as CAEM, giving apples-to-apples matched-scale
        # comparison for Ch5 sig-tests. Cross-benchmark pairs are concatenated
        # then shuffled (random.Random(seed + cycle)) for within-cycle order.
        if cycle_chunks_by_bm:
            cycle_pairs = []
            for bm, chunks in cycle_chunks_by_bm.items():
                if cycle - 1 < len(chunks):
                    cycle_pairs.extend(chunks[cycle - 1])
            random.Random(ns.seed + cycle).shuffle(cycle_pairs)
            logger.info(
                "Cycle %d: training on %d pairs (stream-mode chunks from "
                "benchmark_splits, matches CAEM cycle %d).",
                cycle, len(cycle_pairs), cycle,
            )
        else:
            # Legacy fallback: partition the flat pool by cycle index
            start = ((cycle - 1) * per_cycle_pool_size) % max(len(train_pool), 1)
            end = min(start + per_cycle_pool_size, len(train_pool))
            cycle_pairs = train_pool[start:end]
            logger.info("Cycle %d: training on %d pairs (legacy partition).",
                        cycle, len(cycle_pairs))
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

        # Phase 1a offload (2026-04-22): Google Drive every-cycle upload then
        # local rolling-N retention.
        #
        # Tier 2 — Google Drive upload (env-gated CAEM_GDRIVE_OFFLOAD=1):
        # rclone copy the whole cycle_dir (save_pretrained produces multiple
        # files) to gdrive:caem-phase1a/<ft_variant>/cycle_<n>/. Runs BEFORE
        # local retention so the current cycle is safely on Drive before we
        # delete any local data.
        if os.environ.get("CAEM_GDRIVE_OFFLOAD", "0") == "1":
            import subprocess
            ft_variant = out_root.name
            remote_path = f"gdrive:caem-phase1a/{ft_variant}/cycle_{cycle}/"
            try:
                result = subprocess.run(
                    ["rclone", "copy", str(ckpt_dir), remote_path,
                     "--transfers", "4", "--checkers", "8"],
                    capture_output=True, text=True, timeout=1800,
                )
                if result.returncode == 0:
                    logger.info("Cycle %d: gdrive offload OK -> %s", cycle, remote_path)
                else:
                    logger.warning(
                        "Cycle %d: gdrive offload failed (rc=%d): %s -- local "
                        "rolling-N still active; training continues.",
                        cycle, result.returncode, result.stderr[:500],
                    )
            except Exception as exc:
                logger.warning(
                    "Cycle %d: gdrive offload errored (%s) -- local rolling-N "
                    "still active; training continues.", cycle, exc,
                )

        # Tier 1 — local rolling-N retention (always on). Qwen-2.5-3B full
        # checkpoint ~6 GB; keeping 10 would use 60 GB per baseline run.
        # Delete cycle_{n-2}/ weights (but keep cycle_0 and the final cycle)
        # so peak local disk stays at ~18-24 GB. save_pretrained writes
        # model.safetensors + config.json + tokenizer files; we delete just
        # the weight shards, leaving metadata for downstream inspect.
        if cycle >= 2:
            old_cycle = cycle - 2
            if old_cycle > 0 and old_cycle != ns.num_cycles:
                old_dir = out_root / f"cycle_{old_cycle}"
                for weight_file in list(old_dir.glob("*.safetensors")) + list(old_dir.glob("*.bin")):
                    try:
                        size_mb = weight_file.stat().st_size / (1024 * 1024)
                        weight_file.unlink()
                        logger.info(
                            "Cycle %d: reclaimed %.0f MB from cycle_%d/%s "
                            "(rolling-N retention).",
                            cycle, size_mb, old_cycle, weight_file.name,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Cycle %d: could not delete cycle_%d/%s: %s",
                            cycle, old_cycle, weight_file.name, exc,
                        )

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
            batch_size=getattr(ns, "eval_batch_size", 1),
            use_prefetch=False,
        )
        # Branch C 2026-04-22 evening: eval samples drawn from the shared
        # benchmark_pools[bm].eval (same 500/bench as CAEM's cycle-N eval),
        # ensuring matched-pool comparison for Ch5 sig-tests. Build pools
        # once lazily on first cycle; they're deterministic given seed=42.
        if not hasattr(main, "_eval_pools_cache") or not main._eval_pools_cache:  # type: ignore[attr-defined]
            from caem.benchmark_splits import build_all_benchmark_pools as _bap, ALL_BENCHMARKS as _ALL
            panel = [b for b in _ALL if b in ns.eval_benchmarks]
            main._eval_pools_cache = _bap(  # type: ignore[attr-defined]
                benchmarks=panel,
                n_cycles=int(ns.num_cycles),
                train_chunk_size=max(1, int(ns.n_train_per_bench) // int(ns.num_cycles)),
                eval_size=int(ns.n_eval_per_bench),
                rng_seed=int(ns.seed),
            )
        _eval_pools = main._eval_pools_cache  # type: ignore[attr-defined]
        for bench in ns.eval_benchmarks:
            if bench in _eval_pools:
                samples = list(_eval_pools[bench].eval)
                logger.info(
                    "Cycle %d eval on %s: %d samples from benchmark_pools (shared with CAEM).",
                    cycle, bench, len(samples),
                )
            else:
                logger.warning(
                    "%s not in benchmark_pools; falling back to legacy load_benchmark.",
                    bench,
                )
                if bench == "fever":
                    samples = load_benchmark(bench, n=ns.n_eval_per_bench, split="dev")
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
