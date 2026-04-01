"""
caem/training/self_improvement.py
===================================
SelfImprovementLoop — Stage 8 of the CAEM pipeline.

Runs at the end of each cycle. Fine-tunes Flan-T5 on verified episodes
from the episodic memory store, using L2 regularisation against the
previous cycle's weights to prevent catastrophic forgetting.

What a cycle looks like
-----------------------
1. Collect training data: filter EpisodicMemoryStore for entries where
   û_stored ≥ min_u_stored_for_training (default 0.75). These are the
   high-confidence episodes verified by Stage 5.
2. Mix with general-domain data: add 10% general QA pairs to prevent
   the model from drifting too far from its pre-trained capabilities.
3. Fine-tune with L2 regularisation:
       Loss = CE(generated, target) + (λ/2) · ||θ − θ_prev||²
   θ_prev are the weights frozen at the START of this cycle (previous
   checkpoint). The L2 term penalises large deviations from the prior.
4. Forgetting check: after fine-tuning, measure accuracy on a held-out
   general-domain set. If retention < forgetting_tolerance (0.93),
   abort and restore θ_prev. This is the catastrophic forgetting guard.
5. Save checkpoint: model weights + cycle metadata to outputs/cycle_{n}/.

L2 vs full EWC
--------------
Full EWC (Kirkpatrick et al. 2017) weights the L2 penalty per-parameter
by the Fisher information matrix (FIM). The FIM requires a forward pass
over the entire dataset to compute, adding significant overhead (~same
cost as one training epoch).

CAEM uses L2 with uniform weighting across all parameters. This is a
principled approximation: when the Fisher information is approximately
uniform across parameters (reasonable for a model fine-tuned on diverse
QA), L2 and EWC produce similar results. The computational saving is
significant — no FIM computation, no extra memory for Fisher diagonal.

This is documented as [DES] in the config (l2_lambda = 0.01). The thesis
reports this choice and compares with full EWC in the ablation study.

Checkpoint design
-----------------
Each cycle saves independently to outputs/cycle_{n}/ so:
  - Cycles can be compared post-hoc (cycle_1/model.pt vs cycle_2/model.pt)
  - A failed cycle does not overwrite a successful one
  - Restarting from any checkpoint is possible without retraining from scratch

The SelfImprovementLoop does NOT hold a reference to the memory store
between cycles — it receives one at call time so the caller controls
which store state is used (important for ablations).
"""

from __future__ import annotations

import copy
import logging
import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from caem.config import CAEMConfig
from caem.memory.store import EpisodicMemoryStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QAPair:
    """A single (question, answer) training example."""
    question: str
    answer:   str


@dataclass
class CycleResult:
    """Summary of a completed self-improvement cycle."""
    cycle_num:          int
    n_episodes_used:    int
    n_general_used:     int
    epochs_completed:   int
    final_train_loss:   float
    forgetting_score:   float   # retention on general set (0–1, higher = less forgetting)
    aborted:            bool    # True if forgetting check failed and weights were restored
    checkpoint_path:    str


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset for fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

class QADataset(Dataset):
    """Tokenised (question → answer) dataset for seq2seq fine-tuning."""

    def __init__(
        self,
        pairs: List[QAPair],
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 512,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_input  = max_input_length
        self.max_target = max_target_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        enc = self.tokenizer(
            pair.question,
            max_length=self.max_input,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            dec = self.tokenizer(
                pair.answer,
                max_length=self.max_target,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        labels = dec["input_ids"].squeeze(0)
        # Replace padding token id with -100 so cross-entropy ignores it
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SelfImprovementLoop
# ─────────────────────────────────────────────────────────────────────────────

class SelfImprovementLoop:
    """Fine-tune Flan-T5 on verified episodes at the end of each CAEM cycle.

    Parameters
    ----------
    model : transformers.T5ForConditionalGeneration
        The shared Flan-T5-Large model (same instance used by all tiers).
        Weights are mutated in-place; caller should save a copy beforehand
        if they want to preserve the pre-cycle state externally.
    tokenizer : transformers.AutoTokenizer
        Matching tokenizer.
    config : CAEMConfig
    device : str or None
    output_dir : str
        Root directory for cycle checkpoints.
        Cycle n saves to: {output_dir}/cycle_{n}/

    Usage
    -----
    >>> loop = SelfImprovementLoop(model, tokenizer, config,
    ...                            output_dir="outputs")
    >>> result = loop.run_cycle(
    ...     cycle_num=1,
    ...     memory_store=store,
    ...     general_data=[QAPair("What is 2+2?", "4"), ...]
    ... )
    >>> result.aborted   # False → weights updated; True → restored
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[CAEMConfig] = None,
        device: Optional[str] = None,
        output_dir: str = "outputs",
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.config    = config or CAEMConfig()
        self.output_dir = Path(output_dir)

        if device is None:
            device = str(next(model.parameters()).device)
        self.device = device

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_cycle(
        self,
        cycle_num: int,
        memory_store: EpisodicMemoryStore,
        general_data: List[QAPair],
        seed: int = 42,
    ) -> CycleResult:
        """Run one self-improvement cycle.

        Parameters
        ----------
        cycle_num : int
            Cycle index (1-based). Used for checkpoint naming and logging.
        memory_store : EpisodicMemoryStore
            The current state of the episodic memory. Episodes with
            û_stored ≥ min_u_stored_for_training are used as training data.
        general_data : list of QAPair
            General-domain QA pairs for forgetting prevention. Both the
            10% training mix AND the held-out forgetting check draw from
            this list (different random splits).
        seed : int
            Random seed for reproducibility. Stored in the checkpoint.

        Returns
        -------
        CycleResult
            Summary including whether the cycle was aborted.
        """
        cfg = self.config
        random.seed(seed)
        torch.manual_seed(seed)

        logger.info("=== Self-Improvement Cycle %d ===", cycle_num)

        # Step 1: collect episode training data
        episode_pairs = self._collect_episodes(memory_store)
        logger.info("Cycle %d: %d verified episodes collected.", cycle_num, len(episode_pairs))

        if not episode_pairs:
            logger.warning("Cycle %d: no episodes meet quality threshold — skipping.", cycle_num)
            return CycleResult(
                cycle_num=cycle_num, n_episodes_used=0, n_general_used=0,
                epochs_completed=0, final_train_loss=0.0, forgetting_score=1.0,
                aborted=False,
                checkpoint_path=str(self.output_dir / f"cycle_{cycle_num}"),
            )

        # Step 2: split general_data → training mix + held-out forgetting check
        random.shuffle(general_data)
        n_general_total = len(general_data)
        split = max(1, int(n_general_total * 0.5))
        general_train = general_data[:split]
        general_eval  = general_data[split:] or general_data   # fallback if tiny

        # Step 3: build mixed training set (90% episodes + 10% general)
        train_pairs, n_general_used = self._mix(episode_pairs, general_train)
        logger.info(
            "Cycle %d: training on %d pairs (%d episodes + %d general).",
            cycle_num, len(train_pairs), len(episode_pairs), n_general_used,
        )

        # Step 4: snapshot θ_prev BEFORE fine-tuning (used for L2 reg + forgetting restore)
        theta_prev = self._snapshot_weights()

        # Step 5: fine-tune with L2 regularisation
        epochs_done, final_loss = self._finetune(train_pairs, theta_prev)

        # Step 6: forgetting check
        forgetting_score = self._forgetting_score(general_eval)
        logger.info(
            "Cycle %d: forgetting score = %.4f (tolerance = %.4f)",
            cycle_num, forgetting_score, cfg.forgetting_tolerance,
        )

        aborted = False
        if forgetting_score < cfg.forgetting_tolerance:
            logger.warning(
                "Cycle %d: forgetting check FAILED (%.4f < %.4f) — restoring θ_prev.",
                cycle_num, forgetting_score, cfg.forgetting_tolerance,
            )
            self._restore_weights(theta_prev)
            aborted = True

        # Step 7: save checkpoint
        ckpt_path = self._save_checkpoint(cycle_num, seed, epochs_done, final_loss,
                                           forgetting_score, aborted, theta_prev if aborted else None)

        return CycleResult(
            cycle_num=cycle_num,
            n_episodes_used=len(episode_pairs),
            n_general_used=n_general_used,
            epochs_completed=epochs_done,
            final_train_loss=final_loss,
            forgetting_score=forgetting_score,
            aborted=aborted,
            checkpoint_path=ckpt_path,
        )

    def load_checkpoint(self, cycle_num: int) -> dict:
        """Load cycle metadata from a saved checkpoint directory."""
        ckpt_dir = self.output_dir / f"cycle_{cycle_num}"
        meta_path = ckpt_dir / "meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        # Restore model weights
        weights_path = ckpt_dir / "model.pt"
        if weights_path.exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            logger.info("Restored model weights from %s", weights_path)
        return meta

    # ------------------------------------------------------------------ #
    # Training data construction                                           #
    # ------------------------------------------------------------------ #

    def _collect_episodes(self, memory_store: EpisodicMemoryStore) -> List[QAPair]:
        """Return QAPairs from memory entries that meet the quality threshold.

        Training target is `reasoning_chain`, NOT `answer`.

        Rationale (thesis §4.3 — Chain-of-Thought Supervision):
        EpisodicEntry stores a `reasoning_chain` field — the full chain-of-
        thought trace associated with the verified answer. Fine-tuning on the
        reasoning chain teaches the model *how* to reason toward the correct
        answer, not just to memorise answer strings. This matches the thesis
        claim of "verified reasoning-chain supervision" and is the key
        distinction from standard QA fine-tuning.

        Note: in the current pipeline, `reasoning_chain` is set equal to the
        model's full generation output (which may or may not include explicit
        CoT steps depending on prompting). When a dedicated CoT prompting
        strategy is added, `reasoning_chain` and `answer` will diverge, and
        this training target will automatically benefit from richer supervision
        without any code change here.
        """
        threshold = self.config.min_u_stored_for_training
        pairs = []
        for entry in memory_store.all_entries():
            if entry.u_stored >= threshold:
                # Use reasoning_chain as the seq2seq target — not answer.
                # This aligns training with the chain-of-thought supervision
                # described in the thesis methodology.
                pairs.append(QAPair(question=entry.question, answer=entry.reasoning_chain))
        return pairs

    def _mix(
        self,
        episode_pairs: List[QAPair],
        general_data: List[QAPair],
    ) -> Tuple[List[QAPair], int]:
        """Mix episode data with general_data at config.general_data_ratio.

        The general data ratio is computed relative to the episode count:
            n_general = round(n_episodes * ratio / (1 - ratio))
        This keeps episodes at (1 - ratio) of the total regardless of
        how many general pairs are available.

        Returns (mixed_list_shuffled, n_general_actually_used).
        """
        ratio = self.config.general_data_ratio
        n_episodes = len(episode_pairs)
        n_general_target = round(n_episodes * ratio / max(1.0 - ratio, 1e-9))
        n_general_used = min(n_general_target, len(general_data))

        general_sample = random.sample(general_data, n_general_used)
        combined = episode_pairs + general_sample
        random.shuffle(combined)
        return combined, n_general_used

    # ------------------------------------------------------------------ #
    # Fine-tuning                                                          #
    # ------------------------------------------------------------------ #

    def _finetune(
        self,
        train_pairs: List[QAPair],
        theta_prev: List[torch.Tensor],
    ) -> Tuple[int, float]:
        """Fine-tune the model on train_pairs with L2 regularisation.

        Loss = CrossEntropy(output, labels) + (λ/2) · ||θ − θ_prev||²

        Returns (epochs_completed, final_avg_loss).
        """
        cfg = self.config
        dataset = QADataset(train_pairs, self.tokenizer)
        loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=cfg.learning_rate)

        self.model.train()
        self.model.to(self.device)

        epochs_done = 0
        final_loss  = 0.0

        for epoch in range(cfg.epochs_per_cycle):
            epoch_loss = 0.0
            n_batches  = 0

            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                ce_loss = outputs.loss

                # L2 regularisation: penalise deviation from θ_prev
                l2_loss = self._l2_penalty(theta_prev)
                loss = ce_loss + (cfg.l2_lambda / 2.0) * l2_loss

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for training stability [DES]
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info("  Epoch %d/%d — loss: %.4f", epoch + 1, cfg.epochs_per_cycle, avg_loss)
            epochs_done += 1
            final_loss   = avg_loss

        self.model.eval()
        return epochs_done, final_loss

    def _l2_penalty(self, theta_prev: List[torch.Tensor]) -> torch.Tensor:
        """Compute ||θ − θ_prev||² summed over all parameters."""
        penalty = torch.tensor(0.0, device=self.device)
        for p, p0 in zip(self.model.parameters(), theta_prev):
            penalty = penalty + ((p - p0.to(self.device)) ** 2).sum()
        return penalty

    # ------------------------------------------------------------------ #
    # Forgetting check                                                     #
    # ------------------------------------------------------------------ #

    def _forgetting_score(self, general_eval: List[QAPair]) -> float:
        """Estimate general-capability retention on held-out general pairs.

        For each pair, run greedy generation and do an exact-match check
        against the reference answer (lowercased, stripped). Retention =
        fraction of pairs where the model still produces the correct answer.

        Exact match is a conservative lower bound — the real evaluation
        harness uses F1 and EM against benchmark datasets. Here it is
        used only as a fast forgetting guard, not a quality metric.

        Returns float in [0, 1].
        """
        if not general_eval:
            return 1.0   # no eval data → assume no forgetting

        correct = 0
        self.model.eval()

        with torch.no_grad():
            for pair in general_eval:
                enc = self.tokenizer(
                    pair.question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                out = self.model.generate(
                    enc["input_ids"],
                    max_new_tokens=256,
                    do_sample=False,
                )
                pred = self.tokenizer.decode(out[0], skip_special_tokens=True).strip().lower()
                ref  = pair.answer.strip().lower()
                if pred == ref:
                    correct += 1

        return correct / len(general_eval)

    # ------------------------------------------------------------------ #
    # Weight management                                                    #
    # ------------------------------------------------------------------ #

    def _snapshot_weights(self) -> List[torch.Tensor]:
        """Return a deepcopy of all model parameter tensors (CPU)."""
        return [p.detach().cpu().clone() for p in self.model.parameters()]

    def _restore_weights(self, theta_prev: List[torch.Tensor]) -> None:
        """Restore model parameters to θ_prev in-place."""
        with torch.no_grad():
            for p, p0 in zip(self.model.parameters(), theta_prev):
                p.copy_(p0.to(self.device))
        logger.info("Model weights restored to pre-cycle state.")

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(
        self,
        cycle_num: int,
        seed: int,
        epochs_done: int,
        final_loss: float,
        forgetting_score: float,
        aborted: bool,
        theta_prev: Optional[List[torch.Tensor]],
    ) -> str:
        """Save model weights + metadata to outputs/cycle_{n}/."""
        ckpt_dir = self.output_dir / f"cycle_{cycle_num}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights.
        # INVARIANT: at this point self.model always holds the correct state:
        #   - not aborted → fine-tuned weights (training just completed)
        #   - aborted     → _restore_weights(theta_prev) was already called
        #                   before this method, so state_dict() == theta_prev
        # Saving self.model.state_dict() is therefore always correct.
        # Do NOT compute a separate dict from theta_prev here — that would
        # save tensors without proper parameter names, breaking load_state_dict.
        torch.save(self.model.state_dict(), str(ckpt_dir / "model.pt"))

        # Save metadata
        meta = {
            "cycle_num":       cycle_num,
            "seed":            seed,
            "epochs_done":     epochs_done,
            "final_loss":      final_loss,
            "forgetting_score": forgetting_score,
            "aborted":         aborted,
            "config":          self.config,
        }
        with open(ckpt_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info("Cycle %d checkpoint saved to %s.", cycle_num, ckpt_dir)
        return str(ckpt_dir)
