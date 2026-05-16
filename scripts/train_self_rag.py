#!/usr/bin/env python
"""
scripts/train_self_rag.py
==========================
Self-RAG full-parameter fine-tune on Qwen-2.5-3B-Instruct (B10 full-FT path).

Stage 2 of the Self-RAG pipeline:
  1. ``scripts/generate_self_rag_synthetic_data.py`` -> annotated JSONL
  2. THIS SCRIPT -> fine-tuned Qwen with reflection-token vocabulary
  3. ``eval/baselines.py:SelfRAGBaseline`` -> custom decoder uses the
     trained model's emitted reflection tokens to drive retrieval and
     abstention.

Hardware envelope
-----------------
Full FT on Qwen-3B fits on RTX 5090 (32 GiB) when:
  - ``use_8bit_adamw=True`` (default)
  - gradient checkpointing on
  - bf16 weights + bf16 gradients (no fp32 master)
  - effective batch size 4-8 via gradient accumulation (default 1x8=8)
Memory budget at the above settings: ~24 GiB for weights+grads+optim,
leaves ~8 GiB for activations at sequence length 512.

Expected runtime: 25-35 GPU-h for 3 epochs on 10-30K examples at seq=512
on the 5090. This is the same envelope CAEM's v1 full-FT cycles ran in.

Usage
-----
    python -m scripts.train_self_rag \\
        --train_data outputs/self_rag/synthetic_train.jsonl \\
        --base_model Qwen/Qwen2.5-3B-Instruct \\
        --output_dir outputs/self_rag/model \\
        --num_epochs 3 \\
        --batch_size 1 \\
        --grad_accum_steps 8 \\
        --learning_rate 5e-6 \\
        --use_8bit_adamw \\
        --gradient_checkpointing
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


REFLECTION_TOKENS: List[str] = [
    "[Retrieve]", "[No-Retrieve]",
    "[Relevant]", "[Irrelevant]",
    "[Supported]", "[Partial]", "[NoSupport]",
    "[Useful:1]", "[Useful:2]", "[Useful:3]", "[Useful:4]", "[Useful:5]",
]


# --------------------------------------------------------------------------- #
# Tokenizer + model setup                                                       #
# --------------------------------------------------------------------------- #

def extend_tokenizer_with_reflection_tokens(tokenizer) -> int:
    """Add reflection-token strings to the tokenizer's vocab.

    Returns the number of *newly added* tokens (0 if the tokenizer already
    knows all of them — idempotent under reruns).
    """
    n_added = tokenizer.add_tokens(REFLECTION_TOKENS, special_tokens=True)
    logger.info(
        "Tokenizer extended: %d reflection tokens added (total vocab = %d).",
        n_added, len(tokenizer),
    )
    return int(n_added)


def init_new_embedding_rows(model, n_added: int) -> None:
    """Initialise the freshly-added embedding rows to the mean of existing
    rows. Mean-init is a well-known heuristic for added vocabulary
    (Hewitt 2021); produces lower initial loss than the default random init
    on a few-shot vocabulary like the reflection tokens.
    """
    if n_added <= 0:
        return
    import torch
    embed = model.get_input_embeddings()
    with torch.no_grad():
        weight = embed.weight.data
        existing = weight[:-n_added]
        mean = existing.mean(dim=0, keepdim=True)
        weight[-n_added:] = mean
    # Tie the LM head too if applicable.
    out_embed = model.get_output_embeddings()
    if out_embed is not None and out_embed.weight.shape[0] == weight.shape[0]:
        with torch.no_grad():
            out_w = out_embed.weight.data
            out_w[-n_added:] = out_w[:-n_added].mean(dim=0, keepdim=True)
    logger.info("Initialised %d new embedding rows with mean-init.", n_added)


# --------------------------------------------------------------------------- #
# Data formatting                                                               #
# --------------------------------------------------------------------------- #

def build_chat_example(
    record: Dict[str, Any],
    *,
    tokenizer: Any,
    max_seq_len: int,
) -> Optional[Dict[str, List[int]]]:
    """Format one annotated record into Qwen ChatML and return tokenised IDs.

    Loss is applied to the assistant continuation only (user prompt tokens
    have ``label = -100``), matching standard instruction-tuning practice.
    """
    question = record.get("question", "")
    passages: List[str] = record.get("passages", []) or []
    retrieve = record.get("retrieve_decision") or ""
    passage_decisions: List[str] = record.get("passage_decisions") or []
    support = record.get("support_decision") or ""
    useful = record.get("useful_score") or ""
    annotated = record.get("annotated_answer") or ""
    if not question or not annotated:
        return None

    # User message: question + retrieved passages (if any). Passage block
    # mirrors the Tier-3 prompt builder so train/eval distributions agree.
    user_parts: List[str] = [f"Question: {question}"]
    if passages:
        user_parts.append("Retrieved passages:")
        for i, p in enumerate(passages):
            user_parts.append(f"[{i + 1}] {p}")
    user_msg = "\n".join(user_parts)

    # Assistant message: reflection tokens woven through the answer.
    asst_chunks: List[str] = [retrieve]
    for i, pd in enumerate(passage_decisions):
        asst_chunks.append(f"Passage {i + 1}: {pd}")
    asst_chunks.append(annotated)
    asst_chunks.append(support)
    asst_chunks.append(useful)
    asst_msg = "\n".join(c for c in asst_chunks if c)

    # Tokenise prompt and full sequence separately to compute the boundary.
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": asst_msg},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
    prompt_len = min(len(prompt_ids), len(full_ids))

    labels = list(full_ids)
    for i in range(prompt_len):
        labels[i] = -100  # mask the user/system prefix

    if all(t == -100 for t in labels):
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def load_synthetic_training_data(
    path: Path,
    *,
    tokenizer: Any,
    max_seq_len: int,
) -> List[Dict[str, List[int]]]:
    """Load JSONL written by ``generate_self_rag_synthetic_data.py`` and
    tokenise each example into ``{input_ids, attention_mask, labels}``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Training data not found: {path}")

    out: List[Dict[str, List[int]]] = []
    n_dropped = 0
    with path.open("r") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_dropped += 1
                continue
            if rec.get("dry_run"):
                n_dropped += 1
                continue
            ex = build_chat_example(
                rec, tokenizer=tokenizer, max_seq_len=max_seq_len,
            )
            if ex is None:
                n_dropped += 1
                continue
            out.append(ex)
    logger.info(
        "Loaded %d training examples from %s (%d dropped).",
        len(out), path, n_dropped,
    )
    return out


def collate(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, Any]:
    """Left-or-right pad a list of variable-length training examples.

    Right-pads with ``pad_id`` for ``input_ids`` and ``-100`` for labels so
    masked positions remain masked. Decoder-only causal LM training reads
    labels as shifted-by-one internally.
    """
    import torch
    max_len = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
        mask[i, :n] = torch.tensor(b["attention_mask"], dtype=torch.long)
        labels[i, :n] = torch.tensor(b["labels"], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# --------------------------------------------------------------------------- #
# Optimiser                                                                     #
# --------------------------------------------------------------------------- #

def build_optimizer(model, *, lr: float, use_8bit_adamw: bool, on_cuda: bool):
    """Construct the optimiser. Prefers ``bitsandbytes.optim.AdamW8bit`` for
    ~50% optimiser-state VRAM savings on CUDA; falls back to
    ``torch.optim.AdamW`` otherwise. Same path CAEM SIL uses.
    """
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    if use_8bit_adamw and on_cuda:
        try:
            import bitsandbytes as bnb
            logger.info("Optimiser: bitsandbytes.AdamW8bit (lr=%.2e).", lr)
            return bnb.optim.AdamW8bit(params, lr=lr)
        except ImportError:
            logger.warning(
                "bitsandbytes not installed; falling back to torch.optim.AdamW. "
                "Install `pip install bitsandbytes` to recover Full-FT headroom.",
            )
        except Exception as exc:
            logger.warning(
                "AdamW8bit init failed (%s); using torch.optim.AdamW.", exc,
            )
    logger.info("Optimiser: torch.optim.AdamW (lr=%.2e).", lr)
    return torch.optim.AdamW(params, lr=lr)


# --------------------------------------------------------------------------- #
# Training loop                                                                 #
# --------------------------------------------------------------------------- #

def train(args) -> int:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available; full FT on Qwen-3B requires a GPU. "
            "Refusing to run this on CPU (would take weeks).",
        )
        return 2

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_added = extend_tokenizer_with_reflection_tokens(tokenizer)

    # ---------- Data ----------
    dataset = load_synthetic_training_data(
        args.train_data, tokenizer=tokenizer, max_seq_len=args.max_seq_len,
    )
    if not dataset:
        logger.error("No training examples loaded; aborting.")
        return 3

    def _collate(batch):
        return collate(batch, pad_id=tokenizer.pad_token_id)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=True,
    )

    # ---------- Model ----------
    logger.info("Loading base model %s in bf16 ...", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )
    if n_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        init_new_embedding_rows(model, n_added=n_added)

    if args.gradient_checkpointing:
        # Disable model cache when checkpointing; transformers raises a
        # warning otherwise and silently falls back to the cached path.
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    model.to(device)
    model.train()

    # ---------- Optimiser ----------
    optimizer = build_optimizer(
        model,
        lr=args.learning_rate,
        use_8bit_adamw=args.use_8bit_adamw,
        on_cuda=True,
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.num_epochs
    logger.info(
        "Training plan: %d examples, %d micro-batches/epoch, %d update "
        "steps/epoch, %d total steps over %d epochs.",
        len(dataset), len(loader), steps_per_epoch, total_steps,
        args.num_epochs,
    )

    # ---------- Loop ----------
    global_step = 0
    t0 = time.time()
    for epoch in range(args.num_epochs):
        logger.info("=== Epoch %d/%d ===", epoch + 1, args.num_epochs)
        running_loss = 0.0
        n_micro = 0
        optimizer.zero_grad(set_to_none=True)
        for micro_idx, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum_steps
            loss.backward()
            running_loss += float(loss.detach()) * args.grad_accum_steps
            n_micro += 1

            if (micro_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    avg = running_loss / max(n_micro, 1)
                    elapsed = time.time() - t0
                    eta = elapsed / max(global_step, 1) * (total_steps - global_step) / 60
                    logger.info(
                        "step=%d/%d epoch=%d loss=%.4f wall=%.1fm eta=%.1fm",
                        global_step, total_steps, epoch + 1,
                        avg, elapsed / 60, eta,
                    )
                    running_loss = 0.0
                    n_micro = 0

        # Flush a partial accumulation at epoch boundary.
        if n_micro > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # ---------- Save ----------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model + tokenizer to %s ...", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    meta = {
        "base_model": args.base_model,
        "reflection_tokens": REFLECTION_TOKENS,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "n_train_examples": len(dataset),
        "wall_time_seconds": time.time() - t0,
    }
    (args.output_dir / "self_rag_train_meta.json").write_text(
        json.dumps(meta, indent=2),
    )
    logger.info("Done. wall=%.1f min", (time.time() - t0) / 60)
    return 0


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train_data", type=Path,
                   default=Path("outputs/self_rag/synthetic_train.jsonl"))
    p.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir", type=Path,
                   default=Path("outputs/self_rag/model"))
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--use_8bit_adamw", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--log_level", default="INFO")
    ns = p.parse_args()

    logging.basicConfig(
        level=ns.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    return train(ns)


if __name__ == "__main__":
    sys.exit(main())
