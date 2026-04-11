"""
scripts/run_pub0405_variants.py
================================
Build PUB-04+05 variant checkpoints from existing CAEM artifacts.

This script produces the checkpoint layouts expected by scripts/run_ablation.py:
  - outputs/pub0405/full_ft_ewc/model.pt
  - outputs/pub0405/lora_l2/adapter_config.json (+ adapter weights)
  - outputs/pub0405/lora_only/adapter_config.json (+ adapter weights)

Design goal
-----------
Keep this runner aligned with the existing CAEM codebase:
  - Uses the same memory artifact format (EpisodicMemoryStore.load)
  - Uses the same training pair schema (QAPair + QADataset)
  - Uses run_experiment.load_general_data() for anti-forgetting mix
  - Uses CAEMConfig defaults unless overridden by CLI flags

Important note on EWC
---------------------
This implementation uses a tensor-wise Fisher approximation (one scalar per
parameter tensor, estimated from gradient squared means) to keep memory
requirements practical. It is still a principled EWC-style penalty:

  loss = CE + (lambda_ewc / 2) * sum_i F_i * ||theta_i - theta_i_prev||^2

where i indexes parameter tensors and F_i is the estimated Fisher scalar.

Usage
-----
python -m scripts.run_pub0405_variants \
    --base_checkpoint outputs/cycle_10 \
    --memory_store outputs/memory_store_cycle_10 \
  --output_root outputs/pub0405

Then run ablations with generated checkpoints:
python -m scripts.run_ablation \
  --run_pub0405 \
  --pub0405_full_ft_ewc_checkpoint outputs/pub0405/full_ft_ewc \
  --pub0405_lora_l2_checkpoint outputs/pub0405/lora_l2 \
  --pub0405_lora_only_checkpoint outputs/pub0405/lora_only
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration

from caem.config import CAEMConfig
from caem.memory.store import EpisodicMemoryStore
from caem.training.self_improvement import QADataset, QAPair
from scripts.hardware import apply_memory_flags, print_hardware_summary

logger = logging.getLogger("run_pub0405_variants")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dynamic_collate(tokenizer):
    """Dynamic padding collator copied from self_improvement training path."""

    def _collate(batch):
        pad_id = tokenizer.pad_token_id or 0

        max_in = max(b["input_ids"].shape[0] for b in batch)
        max_tgt = max(b["labels"].shape[0] for b in batch)

        input_ids = torch.full((len(batch), max_in), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_in, dtype=torch.long)
        labels = torch.full((len(batch), max_tgt), -100, dtype=torch.long)

        for i, b in enumerate(batch):
            in_len = b["input_ids"].shape[0]
            tgt_len = b["labels"].shape[0]
            input_ids[i, :in_len] = b["input_ids"]
            attention_mask[i, :in_len] = b["attention_mask"]
            labels[i, :tgt_len] = b["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _collate


def _load_base_model(
    model_name: str,
    checkpoint_dir: str,
    device: str,
    use_fp16: bool,
    use_bf16: bool,
):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    ckpt_path = Path(checkpoint_dir) / "model.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        logger.info("Loaded base checkpoint from %s", ckpt_path)
    else:
        logger.warning("Checkpoint not found at %s. Using pretrained %s.", ckpt_path, model_name)

    if use_fp16:
        model = model.half()
    if use_bf16:
        model = model.bfloat16()

    model = cast(Any, model).to(torch.device(device))
    cast(Any, model).train()
    return model


def _collect_episode_pairs(memory_store: EpisodicMemoryStore, min_u: float) -> List[QAPair]:
    """Collect high-confidence memory episodes as training pairs."""
    pairs: List[QAPair] = []
    for entry in memory_store.all_entries():
        if entry.u_stored < min_u:
            continue
        chain = (entry.reasoning_chain or "").strip()
        if len(chain) < 10:
            chain = (entry.answer or "").strip()
        if not chain:
            continue
        q = (entry.question or "").strip()
        if not q:
            continue
        pairs.append(QAPair(question=q, answer=chain))
    return pairs


def _mix_with_general_data(
    episode_pairs: List[QAPair],
    general_data: List[QAPair],
    ratio: float,
    seed: int,
) -> Tuple[List[QAPair], int]:
    """Match CAEM's 90/10 style mixing rule."""
    rng = random.Random(seed)
    n_episodes = len(episode_pairs)
    n_general_target = round(n_episodes * ratio / max(1.0 - ratio, 1e-9))
    n_general_used = min(n_general_target, len(general_data))

    general_sample = rng.sample(general_data, n_general_used) if n_general_used > 0 else []
    mixed = episode_pairs + general_sample
    rng.shuffle(mixed)
    return mixed, n_general_used


def _build_train_loader(
    train_pairs: List[QAPair],
    tokenizer,
    batch_size: int,
    seed: int,
) -> DataLoader:
    dataset = QADataset(train_pairs, tokenizer)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_dynamic_collate(tokenizer),
        generator=generator,
    )


def _estimate_tensorwise_fisher(
    model,
    loader: DataLoader,
    device: str,
    max_steps: int,
) -> Dict[str, float]:
    """Estimate one Fisher scalar per parameter tensor."""
    fisher: Dict[str, float] = {
        name: 0.0 for name, p in model.named_parameters() if p.requires_grad
    }

    model.train()
    steps = 0
    for batch in loader:
        if max_steps > 0 and steps >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if int((labels != -100).sum().item()) == 0:
            continue

        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss):
            continue
        loss.backward()

        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            fisher[name] += float((p.grad.detach().float().pow(2).mean()).item())

        steps += 1

    if steps == 0:
        logger.warning("Fisher estimation used 0 steps. Falling back to uniform tensor weights.")
        for k in fisher:
            fisher[k] = 1.0
        return fisher

    for k in fisher:
        fisher[k] /= steps
    logger.info("Estimated tensor-wise Fisher on %d steps.", steps)
    return fisher


def _train_full_ft_ewc(
    model,
    loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    lambda_ewc: float,
    fisher_steps: int,
) -> Dict[str, float]:
    """Train full model with tensor-wise EWC penalty."""
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    prev_params_cpu: Dict[str, torch.Tensor] = {
        name: p.detach().cpu().float().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    fisher = _estimate_tensorwise_fisher(model, loader, device=device, max_steps=fisher_steps)

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if int((labels != -100).sum().item()) == 0:
                continue

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ce_loss = outputs.loss
            if not torch.isfinite(ce_loss):
                continue

            ewc_penalty = torch.tensor(0.0, device=device)
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                fi = fisher.get(name, 1.0)
                ref = prev_params_cpu[name].to(device=device, dtype=torch.float32)
                diff_sq = (p.float() - ref).pow(2).sum()
                ewc_penalty = ewc_penalty + (float(fi) * diff_sq)

            loss = ce_loss + (lambda_ewc / 2.0) * ewc_penalty
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)
        logger.info("full_ft_ewc epoch %d/%d loss=%.4f", epoch + 1, epochs, final_loss)

    return {
        "final_loss": final_loss,
        "epochs": float(epochs),
    }


def _train_lora_variant(
    model,
    loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    l2_lambda: float,
    with_l2: bool,
) -> Dict[str, float]:
    """Train LoRA adapter params with optional L2 penalty."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found.")

    optimizer = AdamW(trainable, lr=learning_rate)

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if int((labels != -100).sum().item()) == 0:
                continue

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ce_loss = outputs.loss
            if not torch.isfinite(ce_loss):
                continue

            loss = ce_loss
            if with_l2 and l2_lambda > 0.0:
                l2_penalty = torch.tensor(0.0, device=device)
                for p in trainable:
                    l2_penalty = l2_penalty + p.float().pow(2).sum()
                loss = ce_loss + (l2_lambda / 2.0) * l2_penalty

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        final_loss = epoch_loss / max(n_batches, 1)
        tag = "lora_l2" if with_l2 else "lora_only"
        logger.info("%s epoch %d/%d loss=%.4f", tag, epoch + 1, epochs, final_loss)

    return {
        "final_loss": final_loss,
        "epochs": float(epochs),
    }


def _save_meta(path: Path, payload: Dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@dataclass
class VariantOutcome:
    name: str
    path: str
    status: str
    detail: str


def run(ns: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    _set_seed(ns.seed)

    profile = print_hardware_summary()
    apply_memory_flags(profile)
    device = profile.device

    cfg = CAEMConfig()
    batch_size = ns.batch_size if ns.batch_size is not None else min(cfg.batch_size, profile.recommended_batch_size)
    learning_rate = ns.learning_rate if ns.learning_rate is not None else cfg.learning_rate
    epochs = ns.epochs if ns.epochs is not None else cfg.epochs_per_cycle
    l2_lambda = ns.l2_lambda if ns.l2_lambda is not None else cfg.l2_lambda

    output_root = Path(ns.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    memory_store = EpisodicMemoryStore.load(ns.memory_store)
    logger.info("Loaded memory store with %d episodes from %s", memory_store.size, ns.memory_store)

    episode_pairs = _collect_episode_pairs(memory_store, min_u=cfg.min_u_stored_for_training)
    logger.info("Collected %d high-confidence episode pairs.", len(episode_pairs))

    from scripts.run_experiment import load_general_data

    general_data = load_general_data(n=ns.general_data_n)
    logger.info("Loaded %d general-data pairs.", len(general_data))

    train_pairs, n_general_used = _mix_with_general_data(
        episode_pairs=episode_pairs,
        general_data=general_data,
        ratio=cfg.general_data_ratio,
        seed=ns.seed,
    )

    if ns.max_train_pairs > 0 and len(train_pairs) > ns.max_train_pairs:
        rng = random.Random(ns.seed)
        train_pairs = rng.sample(train_pairs, ns.max_train_pairs)

    if not train_pairs:
        raise RuntimeError("No training pairs available. Check memory store path and thresholds.")

    logger.info(
        "Training pool: %d pairs (%d episode + %d general used before cap).",
        len(train_pairs),
        len(episode_pairs),
        n_general_used,
    )

    tokenizer = AutoTokenizer.from_pretrained(ns.model_name)
    train_loader = _build_train_loader(
        train_pairs=train_pairs,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seed=ns.seed,
    )

    outcomes: List[VariantOutcome] = []

    # ------------------------------------------------------------------
    # full_ft_ewc
    # ------------------------------------------------------------------
    if not ns.skip_full_ft_ewc:
        variant_dir = output_root / "full_ft_ewc"
        if ns.dry_run:
            outcomes.append(VariantOutcome("full_ft_ewc", str(variant_dir), "skipped", "dry_run"))
        else:
            model = _load_base_model(
                model_name=ns.model_name,
                checkpoint_dir=ns.base_checkpoint,
                device=device,
                use_fp16=profile.use_fp16,
                use_bf16=profile.use_bf16,
            )
            stats = _train_full_ft_ewc(
                model=model,
                loader=train_loader,
                device=device,
                epochs=epochs,
                learning_rate=learning_rate,
                lambda_ewc=ns.ewc_lambda,
                fisher_steps=ns.fisher_steps,
            )
            variant_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), variant_dir / "model.pt")
            _save_meta(
                variant_dir,
                {
                    "variant": "full_ft_ewc",
                    "base_checkpoint": ns.base_checkpoint,
                    "memory_store": ns.memory_store,
                    "n_train_pairs": len(train_pairs),
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "ewc_lambda": ns.ewc_lambda,
                    "fisher_steps": ns.fisher_steps,
                    **stats,
                },
            )
            outcomes.append(VariantOutcome("full_ft_ewc", str(variant_dir), "ok", "model.pt"))
    else:
        outcomes.append(VariantOutcome("full_ft_ewc", str(output_root / "full_ft_ewc"), "skipped", "flag"))

    # ------------------------------------------------------------------
    # LoRA variants
    # ------------------------------------------------------------------
    run_lora_variants = not ns.skip_lora_l2 or not ns.skip_lora_only
    peft_available = True
    peft_err = ""
    peft_mod: Optional[Any] = None
    if run_lora_variants:
        try:
            peft_mod = importlib.import_module("peft")
            _ = getattr(peft_mod, "LoraConfig")
            _ = getattr(peft_mod, "TaskType")
            _ = getattr(peft_mod, "get_peft_model")
        except Exception as exc:
            peft_available = False
            peft_err = str(exc)
            logger.warning("peft not available; LoRA variants will be skipped (%s)", exc)

    def _run_lora(name: str, with_l2: bool, skip_flag: bool) -> None:
        if skip_flag:
            outcomes.append(VariantOutcome(name, str(output_root / name), "skipped", "flag"))
            return
        if not peft_available:
            outcomes.append(VariantOutcome(name, str(output_root / name), "skipped", f"peft_missing: {peft_err}"))
            return
        if ns.dry_run:
            outcomes.append(VariantOutcome(name, str(output_root / name), "skipped", "dry_run"))
            return

        if peft_mod is None:
            outcomes.append(VariantOutcome(name, str(output_root / name), "skipped", "peft_not_loaded"))
            return

        LoraConfig = getattr(peft_mod, "LoraConfig")
        TaskType = getattr(peft_mod, "TaskType")
        get_peft_model = getattr(peft_mod, "get_peft_model")

        model = _load_base_model(
            model_name=ns.model_name,
            checkpoint_dir=ns.base_checkpoint,
            device=device,
            use_fp16=profile.use_fp16,
            use_bf16=profile.use_bf16,
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=ns.lora_r,
            lora_alpha=ns.lora_alpha,
            lora_dropout=ns.lora_dropout,
            target_modules=["q", "v"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.train()

        stats = _train_lora_variant(
            model=model,
            loader=train_loader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            l2_lambda=l2_lambda,
            with_l2=with_l2,
        )

        variant_dir = output_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(variant_dir))
        _save_meta(
            variant_dir,
            {
                "variant": name,
                "base_checkpoint": ns.base_checkpoint,
                "memory_store": ns.memory_store,
                "n_train_pairs": len(train_pairs),
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "l2_lambda": l2_lambda,
                "with_l2": with_l2,
                "lora_r": ns.lora_r,
                "lora_alpha": ns.lora_alpha,
                "lora_dropout": ns.lora_dropout,
                **stats,
            },
        )
        outcomes.append(VariantOutcome(name, str(variant_dir), "ok", "adapter"))

    _run_lora(name="lora_l2", with_l2=True, skip_flag=ns.skip_lora_l2)
    _run_lora(name="lora_only", with_l2=False, skip_flag=ns.skip_lora_only)

    manifest = {
        "output_root": str(output_root),
        "base_checkpoint": ns.base_checkpoint,
        "memory_store": ns.memory_store,
        "model_name": ns.model_name,
        "n_episode_pairs": len(episode_pairs),
        "n_train_pairs": len(train_pairs),
        "dry_run": ns.dry_run,
        "outcomes": [o.__dict__ for o in outcomes],
        "notes": {
            "full_ft_l2_reference": "Use outputs/cycle_10 (existing CAEM full FT + L2 baseline)",
            "run_ablation_flags": {
                "pub0405_full_ft_ewc_checkpoint": str(output_root / "full_ft_ewc"),
                "pub0405_lora_l2_checkpoint": str(output_root / "lora_l2"),
                "pub0405_lora_only_checkpoint": str(output_root / "lora_only"),
            },
        },
    }

    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("PUB-04+05 variant generation complete. Manifest: %s", output_root / "manifest.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate PUB-04+05 checkpoints compatible with run_ablation.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_checkpoint", default="outputs/cycle_10",
                   help="Directory containing base model.pt (full FT + L2 reference).")
    p.add_argument("--memory_store", default="outputs/memory_store_cycle_10",
                   help="Base path for memory store artifacts (.faiss/.meta omitted).")
    p.add_argument("--output_root", default="outputs/pub0405",
                   help="Directory where PUB-04+05 variant checkpoints will be written.")
    p.add_argument("--model_name", default="google/flan-t5-large",
                   help="Backbone model used for all PUB-04+05 variants.")
    p.add_argument("--general_data_n", type=int, default=1000,
                   help="Number of general QA pairs to load for anti-forgetting mix.")
    p.add_argument("--max_train_pairs", type=int, default=50_000,
                   help="Cap total training pairs (0 or negative means no cap).")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override training batch size. Default uses CAEM/hardware recommendation.")
    p.add_argument("--learning_rate", type=float, default=None,
                   help="Override learning rate. Default from CAEMConfig.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs. Default from CAEMConfig.epochs_per_cycle.")
    p.add_argument("--l2_lambda", type=float, default=None,
                   help="L2 lambda for LoRA+L2 variant. Default from CAEMConfig.")
    p.add_argument("--ewc_lambda", type=float, default=0.01,
                   help="Tensor-wise EWC penalty coefficient for full_ft_ewc.")
    p.add_argument("--fisher_steps", type=int, default=200,
                   help="Max mini-batches used for Fisher estimation.")
    p.add_argument("--lora_r", type=int, default=8,
                   help="LoRA rank r.")
    p.add_argument("--lora_alpha", type=int, default=16,
                   help="LoRA alpha.")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")

    p.add_argument("--skip_full_ft_ewc", action="store_true",
                   help="Skip generating full_ft_ewc variant.")
    p.add_argument("--skip_lora_l2", action="store_true",
                   help="Skip generating lora_l2 variant.")
    p.add_argument("--skip_lora_only", action="store_true",
                   help="Skip generating lora_only variant.")
    p.add_argument("--dry_run", action="store_true",
                   help="Plan outputs only (no training).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args)
