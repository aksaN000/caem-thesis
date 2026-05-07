"""
caem/training/self_improvement.py
===================================
SelfImprovementLoop -- Stage 8 of the CAEM pipeline (Branch C, decoder-only).

Runs at the end of each cycle. Fine-tunes the Qwen-2.5-3B-Instruct base
generator on verified episodes from the episodic memory store, using L2
regularisation against the previous cycle's weights to prevent catastrophic
forgetting.

What a cycle looks like
-----------------------
1. Collect training data: filter EpisodicMemoryStore for entries where
   û_stored >= min_u_stored_for_training (default 0.75). These are the
   high-confidence episodes verified by Stage 5.
2. Mix with general-domain data: add 10% general QA pairs to prevent
   the model from drifting too far from its pre-trained capabilities.
3. Fine-tune with L2 regularisation:
       Loss = CE(generated, target) + (lambda/2) * ||theta - theta_prev||^2
   theta_prev are the weights frozen at the START of this cycle (previous
   checkpoint). The L2 term penalises large deviations from the prior.
4. Forgetting check: after fine-tuning, run a fixed MMLU split (n=200)
   and compare post-cycle MMLU against the PRISTINE cycle-0 MMLU anchor.
   If (post / pristine) < forgetting_tolerance (0.93), abort and restore
   theta_prev.
5. Save checkpoint: model weights + cycle metadata to outputs/cycle_{n}/.

Branch C training-path cascade
------------------------------
PRIMARY (Phase 1a — what runs in this thesis): Full fine-tune + 8-bit
  AdamW (bitsandbytes) + L2 anchor. Verified to fit 32 GB on an RTX
  5090 at batch=4, gradient checkpointing enabled, bf16 mixed precision
  (~22-25 GB peak with MiniCheck co-resident). See config.use_8bit_adamw.

DEFERRED (Phase 2.9 design intent, implementation pending Phase 1b):
  LoRA rank-16 on Qwen attention + MLP modules with frozen base + L2
  anchor on adapter weights. Empirically motivated by the Phase 1
  Cycle-0 audit (~100 SIL samples/benchmark/cycle is in LoRA's
  sparse-pool home turf; full FT is regularisation-dominated at this
  pool size). Wang et al. 2023 + Biderman et al. 2024 cited as
  literature support. Audit conducted 2026-04-25 13:00 UTC confirmed
  the LoRA training-loop wiring (peft.LoraConfig, get_peft_model,
  adapter-only optimiser, adapter-only L2 anchor, adapter-only
  checkpoint save/load) is NOT implemented in this file. Setting
  ``cfg.use_lora_training=True`` is currently a silent no-op — the
  full FT path runs regardless. Phase 1a thesis runs full FT; LoRA
  swap is documented as Phase 1b future work.

FAILURE FALLBACK (no weight update): memory-only cycle. The SIL
  run_cycle still consolidates via the episodic memory store
  (retroverify, deferred-entry reconsideration) but does not fine-tune
  the base generator. Reached when full FT itself fails (OOM, MMLU
  retention < 0.93, convergence failure).

L2 vs full EWC
--------------
Full EWC (Kirkpatrick et al. 2017) weights the L2 penalty per-parameter by
the Fisher information matrix (FIM). CAEM uses uniform-weighting L2; when
the FIM is approximately uniform across parameters (reasonable for an
instruction-tuned QA model), L2 and EWC give similar results. Modern
continual-learning work corroborates L2 anchoring as a competitive simple
baseline -- see Song 2025 and GeRe on anchored continual fine-tuning.

Decoder-only training format
----------------------------
The prompt/target format mirrors the inference-time contract used in
``caem.pipeline._tier2``:

  - The question string is wrapped in ChatML via ``build_tier2_prompt``
    (few-shot scaffolded, task-aware) and the ``"Reasoning:"`` forced-prefix
    is appended as prefill.
  - The training target is the episode's ``reasoning_chain``, with any
    leading ``"Reasoning:"`` token stripped so it does not duplicate the
    forced prefix.
  - input_ids = concat(prompt_ids, target_ids, eos). labels mask all
    prompt positions with -100 so cross-entropy only penalises errors on
    the reasoning chain + answer span.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW as TorchAdamW
from torch.utils.data import DataLoader, Dataset

from caem.config import CAEMConfig, TRAINING_BENCHMARKS
from caem.memory.store import EpisodicMemoryStore
from caem.prompts import FORCED_PREFIX, build_tier2_prompt
from caem.training.loop_filter import is_repetitive_loop

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

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
    mmlu_retention_ratio: float
    aborted:            bool
    checkpoint_path:    str
    mmlu_retention:     float = 0.0
    n_retroverified:    int   = 0
    n_retropruned:      int   = 0
    n_deferred_promoted:   int = 0
    n_deferred_ttl_dropped: int = 0
    n_deferred_kept:       int = 0
    # v2 Fix 11 layer 3 fields — let the orchestrator's post-cycle assertion
    # verify that the deferred reconsideration sweep fired when expected.
    deferred_buffer_size_at_entry: int = 0
    reconsider_fired: bool = False
    # Branch C Goal 4 item 3 (memory consolidation @ SBERT 0.88).
    # Counts the clusters that actually had >= 2 members and the total
    # number of episodes the consolidation pass removed. Zero when the
    # cycle is aborted or when cfg.enable_consolidation is False.
    n_consolidated_clusters: int = 0
    n_consolidated_removed:  int = 0


# -----------------------------------------------------------------------------
# Training dataset (causal LM, ChatML + prompt-position label-masking)
# -----------------------------------------------------------------------------

def _strip_forced_prefix(text: str) -> str:
    # Strip a leading "Reasoning:" (with optional trailing whitespace) so the
    # training target continues from the prefill rather than duplicating it.
    s = text.lstrip()
    if s.lower().startswith(FORCED_PREFIX.lower()):
        return s[len(FORCED_PREFIX):].lstrip()
    return text


class QADataset(Dataset):
    """Causal-LM training dataset for decoder-only backbones.

    Each item is a single sequence built from:
        prompt_ids = tokenize(ChatML(question) + "Reasoning:")
        target_ids = tokenize(reasoning_chain_minus_prefix) + [eos]
        input_ids  = concat(prompt_ids, target_ids)
        labels     = [-100]*len(prompt_ids) + target_ids   # mask prompt

    so cross-entropy only fires on the reasoning-chain tokens. Keeping the
    prompt tokens in input_ids (rather than dropping them) preserves the
    attention pattern the model sees at inference time.
    """

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

        # Build ChatML prompt with forced prefix appended (matches inference).
        prompt_text, forced_prefix = build_tier2_prompt(pair.question, self.tokenizer)
        if not isinstance(prompt_text, str):
            # Fallback for test stubs whose apply_chat_template returns a
            # non-string sentinel; treat the question itself as the prompt.
            prompt_text = str(pair.question)
        full_prompt = f"{prompt_text}{forced_prefix}"
        target_text = _strip_forced_prefix(pair.answer or "")

        # Tokenize prompt and target separately so we know exactly where the
        # prompt ends (for label masking). add_special_tokens=False on the
        # target avoids re-emitting BOS inside the continuation.
        prompt_enc = self.tokenizer(
            full_prompt,
            max_length=self.max_input,
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target,
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_enc["input_ids"].squeeze(0).long()
        target_ids = target_enc["input_ids"].squeeze(0).long()

        # Append EOS so the model learns to terminate.
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None and (target_ids.numel() == 0 or int(target_ids[-1].item()) != int(eos_id)):
            target_ids = torch.cat(
                [target_ids, torch.tensor([int(eos_id)], dtype=target_ids.dtype)]
            )

        input_ids = torch.cat([prompt_ids, target_ids], dim=0)
        attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        labels[: prompt_ids.shape[0]] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# -----------------------------------------------------------------------------
# SelfImprovementLoop
# -----------------------------------------------------------------------------

class SelfImprovementLoop:
    """Fine-tune the decoder-only base generator on verified episodes.

    The shared ``model`` instance is mutated in-place. The caller should
    snapshot externally if they need to preserve pre-cycle state beyond
    the per-cycle ``theta_prev`` L2 anchor.
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
        self._last_chain_diagnostics: Optional[Dict[str, object]] = None

        # Pristine MMLU anchor for the forgetting guard -- lazily measured on
        # the first run_cycle() and reused so retention_ratio always compares
        # against the unmodified cycle-0 baseline. See reset_pristine_mmlu().
        self._pristine_mmlu: Optional[float] = None

        # v2 Fix 8 — wrap the model with LoRA adapters when configured.
        # Keep the resolved flag on self so other methods (optimizer
        # builder, L2 penalty, checkpoint, snapshot) all branch on the
        # same value. ``_lora_active`` is False when peft is missing or
        # the model is a mock without any wrappable linear modules; the
        # full-FT path remains available as the back-compat fallback.
        self._lora_active: bool = self._apply_lora_if_enabled()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def reset_pristine_mmlu(self) -> None:
        """Clear the cached pristine MMLU anchor so the next ``run_cycle``
        re-measures it. Use between independent SIL runs (e.g. variants in
        ``run_cyclic_ablation.py`` that share a constructed
        SelfImprovementLoop instance, or multi-seed sweeps where each seed
        must re-establish its own cycle-0 baseline)."""
        self._pristine_mmlu = None

    # ------------------------------------------------------------------ #
    # v2 Fix 8 — LoRA wrapping                                            #
    # ------------------------------------------------------------------ #

    def _apply_lora_if_enabled(self) -> bool:
        """Wrap ``self.model`` with a peft LoRA adapter when configured.

        Returns ``True`` when the wrap succeeded (LoRA path active) and
        ``False`` otherwise (full-FT path remains active).

        Failure modes that fall back to full FT (with a warning):
          * ``cfg.use_lora_training=False`` (back-compat ablation)
          * peft is not installed
          * the model is a mock without wrappable Linear modules
            (most unit-test paths)
          * the model is already a PeftModel (idempotency)
        """
        cfg = self.config
        if not bool(getattr(cfg, "use_lora_training", False)):
            return False

        # Already wrapped? Skip.
        try:
            from peft import PeftModel  # type: ignore
            if isinstance(self.model, PeftModel):
                logger.info(
                    "LoRA wrap: model is already a PeftModel — skipping "
                    "re-wrap (idempotency).",
                )
                return True
        except ImportError:
            pass

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            logger.warning(
                "use_lora_training=True but peft is not installed; "
                "falling back to full FT. Install with `pip install peft` "
                "to enable the LoRA SIL primary path."
            )
            return False

        # Detect whether the model has any wrappable Linear modules with
        # the configured target names. Mock models in unit tests do not,
        # and forcing get_peft_model on them raises a confusing error.
        try:
            import torch.nn as _nn
            target_names = tuple(getattr(cfg, "lora_target_modules", ()) or ())
            wrappable = any(
                isinstance(m, _nn.Linear) and any(t in name for t in target_names)
                for name, m in self.model.named_modules()
            )
        except Exception:
            wrappable = True  # be permissive on unusual model objects
        if not wrappable:
            logger.info(
                "LoRA wrap: no Linear modules matching target names "
                "%s found on this model (likely a mock). Skipping wrap.",
                target_names,
            )
            return False

        try:
            lora_cfg = LoraConfig(
                r=int(cfg.lora_r),
                lora_alpha=int(cfg.lora_alpha),
                lora_dropout=float(cfg.lora_dropout),
                bias="none",
                target_modules=list(cfg.lora_target_modules),
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_cfg)
        except Exception as exc:
            logger.warning(
                "LoRA wrap failed (%s); falling back to full FT.", exc,
            )
            return False

        # Log adapter footprint for the per-cycle SVD diagnostic
        # (Fix 5 will read these counts).
        try:
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.model.parameters())
            ratio = n_trainable / max(n_total, 1)
            logger.info(
                "LoRA wrap OK | r=%d alpha=%d dropout=%.2f targets=%s | "
                "trainable=%.2fM / total=%.2fM (%.4f%%)",
                cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout,
                cfg.lora_target_modules,
                n_trainable / 1e6, n_total / 1e6, 100.0 * ratio,
            )
        except Exception:
            logger.info("LoRA wrap OK (param-count probe failed; nonfatal).")
        return True

    def run_cycle(
        self,
        cycle_num: int,
        memory_store: EpisodicMemoryStore,
        seed: int = 42,
        verify_fn: Optional[Callable[[Any], Any]] = None,
        deferred_buffer: Optional[Any] = None,
        reconsider_fn: Optional[Callable[[Any], Any]] = None,
    ) -> CycleResult:
        """Run one self-improvement cycle. See module docstring for the flow."""
        cfg = self.config
        random.seed(seed)
        torch.manual_seed(seed)

        logger.info("=== Self-Improvement Cycle %d ===", cycle_num)

        # v2 Fix 11 LAYER 2 — Hard-fail early if deferred_buffer is missing.
        # This must fire BEFORE _collect_episodes (which can early-return) so
        # the orchestrator-wiring bug is caught regardless of training-pool
        # contents. Pre-empts the May-4 v1 gap where the buffer-less code
        # path silently skipped reconsideration for cycles 1-4.
        if deferred_buffer is None:
            if not getattr(cfg, "allow_skip_deferred", False):
                raise RuntimeError(
                    "v2 Fix 11 layer 2: deferred_buffer kwarg is required at "
                    "cycle %d run_cycle entry. Pass pipeline.deferred_buffer "
                    "and pipeline.make_reconsider_deferred_fn() to "
                    "sil.run_cycle(). Set cfg.allow_skip_deferred=True to "
                    "opt out (NOT recommended for production runs). This "
                    "guard prevents the May-4 orchestrator gap from "
                    "recurring." % cycle_num
                )
            logger.warning(
                "Cycle %d: deferred_buffer not provided AND allow_skip_deferred=True. "
                "Skipping deferred reconsideration. NOT recommended for production.",
                cycle_num,
            )

        episode_pairs = self._collect_episodes(memory_store)
        logger.info("Cycle %d: %d verified episodes collected.", cycle_num, len(episode_pairs))
        self._log_chain_diagnostics(cycle_num)

        if not episode_pairs:
            logger.warning("Cycle %d: no episodes meet quality threshold -- skipping.", cycle_num)
            return CycleResult(
                cycle_num=cycle_num, n_episodes_used=0, n_general_used=0,
                epochs_completed=0, final_train_loss=0.0, mmlu_retention_ratio=1.0,
                aborted=False,
                checkpoint_path=str(self.output_dir / f"cycle_{cycle_num}"),
                mmlu_retention=float("nan"),
            )

        # v2 architecture (2026-05-06): general-domain mix REMOVED. The
        # SIL training pool is now built entirely from verified episodes
        # passing the per-benchmark conformal gate, with loss reweighting
        # (Fix 3) providing benchmark balance. Anti-forgetting is provided
        # by LoRA's small parameter budget + multi-modal retention probe.
        train_pairs = list(episode_pairs)
        random.shuffle(train_pairs)
        n_general_used = 0
        logger.info(
            "Cycle %d: training on %d pairs (%d episodes; general-mix REMOVED in v2).",
            cycle_num, len(train_pairs), len(episode_pairs),
        )

        theta_prev = self._snapshot_weights()

        pre_cycle_mmlu = self._mmlu_score(n=200)
        if self._pristine_mmlu is None:
            self._pristine_mmlu = pre_cycle_mmlu
            logger.info(
                "Cycle %d: anchoring pristine MMLU baseline = %.4f "
                "(forgetting guard will compare all future cycles against this value).",
                cycle_num,
                pre_cycle_mmlu if not math.isnan(pre_cycle_mmlu) else float("nan"),
            )
        else:
            logger.info(
                "Cycle %d: pristine MMLU anchor = %.4f | pre-cycle MMLU = %.4f "
                "(pre-cycle delta from pristine = %+.4f)",
                cycle_num,
                self._pristine_mmlu,
                pre_cycle_mmlu if not math.isnan(pre_cycle_mmlu) else float("nan"),
                (pre_cycle_mmlu - self._pristine_mmlu)
                    if (not math.isnan(pre_cycle_mmlu)
                        and not math.isnan(self._pristine_mmlu))
                    else float("nan"),
            )

        if math.isnan(self._pristine_mmlu):
            logger.warning(
                "Cycle %d: pristine MMLU is NaN -- forgetting guard DISABLED; "
                "cycle will complete without abort check.", cycle_num,
            )

        epochs_done, final_loss = self._finetune(train_pairs, theta_prev)

        post_mmlu = self._mmlu_score(n=200)
        pristine = self._pristine_mmlu
        if math.isnan(post_mmlu) \
                or pristine is None \
                or math.isnan(pristine) \
                or pristine <= 1e-6:
            retention_ratio = 1.0
            guard_active = False
        else:
            retention_ratio = post_mmlu / pristine
            guard_active = True

        if guard_active:
            logger.info(
                "Cycle %d: post-training MMLU = %.4f | retention ratio vs. pristine "
                "= %.4f (threshold = %.4f)",
                cycle_num, post_mmlu, retention_ratio, cfg.forgetting_tolerance,
            )
        else:
            logger.info(
                "Cycle %d: post-training MMLU = %s (forgetting guard inactive)",
                cycle_num,
                ("%.4f" % post_mmlu) if not math.isnan(post_mmlu) else "N/A",
            )

        mmlu_retention_ratio = retention_ratio
        mmlu_retention = post_mmlu

        aborted = False
        if guard_active and retention_ratio < cfg.forgetting_tolerance:
            logger.warning(
                "Cycle %d: MMLU forgetting check FAILED (retention %.4f < %.4f) "
                "-- restoring theta_prev.",
                cycle_num, retention_ratio, cfg.forgetting_tolerance,
            )
            self._restore_weights(theta_prev)
            aborted = True

        ckpt_path = self._save_checkpoint(
            cycle_num, seed, epochs_done, final_loss,
            mmlu_retention_ratio, aborted, theta_prev if aborted else None,
        )

        n_retroverified = 0
        n_retropruned = 0
        if verify_fn is not None and not aborted:
            # Goal 4 item 5: report the hit-counter pressure BEFORE retroverify
            # resets the counters. The forced queue is the set of entries that
            # crossed cfg.hit_counter_force_retroverify (default 10) since
            # their last re-score. Retroverify re-scores every entry so the
            # forced queue is automatically covered; this log line documents
            # the population size so between-cycle fast-path scripts can
            # calibrate against it. See also scripts/force_retroverify.py for
            # the between-cycle entry point that operates on just this queue.
            try:
                force_queue = memory_store.force_retroverify_queue()
            except Exception as exc:
                logger.debug(
                    "Cycle %d: force_retroverify_queue probe failed (%s); "
                    "continuing.", cycle_num, exc,
                )
                force_queue = []
            if force_queue:
                logger.info(
                    "Cycle %d: hit-counter forced-queue size = %d "
                    "(top-pressure entries that would have been force-re-scored "
                    "mid-cycle if the between-cycle fast path were enabled; "
                    "these will be re-scored by the full retroverify pass below).",
                    cycle_num, len(force_queue),
                )
            logger.info(
                "Cycle %d: running retroactive re-verification on %d episodes ...",
                cycle_num, memory_store.size,
            )
            try:
                n_retroverified, n_retropruned = memory_store.retroverify(verify_fn)
                logger.info(
                    "Cycle %d: retroverify complete -- %d updated, %d pruned (store size %d).",
                    cycle_num, n_retroverified, n_retropruned, memory_store.size,
                )
            except Exception as exc:
                logger.warning(
                    "Cycle %d: retroverify loop raised %s -- continuing without retroactive update.",
                    cycle_num, exc,
                )
        elif verify_fn is None:
            logger.debug(
                "Cycle %d: no verify_fn supplied -- skipping retroactive re-verification.",
                cycle_num,
            )

        n_deferred_promoted = 0
        n_deferred_ttl_dropped = 0
        n_deferred_kept = 0
        deferred_buffer_size_at_entry = (
            deferred_buffer.size if deferred_buffer is not None else 0
        )
        reconsider_fired = False

        # Layer 2 guard fired at the top of run_cycle if deferred_buffer was None.
        # Reaching this point with deferred_buffer is None implies allow_skip_deferred
        # was True (legitimate opt-out) — proceed without reconsideration.
        if deferred_buffer is not None and not aborted:
            effective_fn = reconsider_fn or verify_fn
            if effective_fn is None:
                logger.debug(
                    "Cycle %d: deferred_buffer supplied but no reconsider_fn "
                    "or verify_fn -- skipping reconsideration.", cycle_num,
                )
            else:
                logger.info(
                    "Cycle %d: running deferred-entry reconsideration on "
                    "%d buffered episodes ...", cycle_num, deferred_buffer.size,
                )
                try:
                    n_deferred_promoted, n_deferred_ttl_dropped, n_deferred_kept = \
                        deferred_buffer.reconsider(
                            verify_fn=effective_fn,
                            memory_store=memory_store,
                        )
                    reconsider_fired = True
                    logger.info(
                        "Cycle %d: deferred reconsideration complete -- "
                        "%d promoted, %d TTL-dropped, %d kept (buffer size %d, "
                        "store size %d).",
                        cycle_num, n_deferred_promoted, n_deferred_ttl_dropped,
                        n_deferred_kept, deferred_buffer.size, memory_store.size,
                    )
                except Exception as exc:
                    logger.warning(
                        "Cycle %d: deferred reconsideration raised %s -- "
                        "continuing without reconsideration pass.",
                        cycle_num, exc,
                    )
        elif deferred_buffer is None:
            logger.debug(
                "Cycle %d: no deferred_buffer supplied -- skipping deferred "
                "reconsideration.", cycle_num,
            )

        # Step 8c: memory consolidation (Branch C Goal 4 item 3).
        # Ordering: retroverify (Step 8) -> deferred reconsideration
        # (Step 8b) -> consolidation (this step). Rationale: retroverify
        # downgrades / prunes entries at the individual level first;
        # deferred reconsideration promotes newly-confident buffered
        # entries; THEN consolidation operates on the clean settled set
        # so cluster representative selection sees the freshest u_stored
        # values and avoids wasted work on entries that would have been
        # pruned anyway. Cycle-0 is protected by the cycle_num guard in
        # consolidate() (cold-start memory should not be collapsed on the
        # first pass). Skipped on aborted cycles (nothing has changed
        # since the previous cycle's consolidation).
        n_consolidated_clusters = 0
        n_consolidated_removed = 0
        if not aborted and getattr(cfg, "enable_consolidation", True):
            try:
                n_consolidated_clusters, n_consolidated_removed = \
                    memory_store.consolidate(cycle_num=cycle_num)
                logger.info(
                    "Cycle %d: consolidation complete -- %d clusters merged, "
                    "%d episodes removed (store size %d).",
                    cycle_num, n_consolidated_clusters,
                    n_consolidated_removed, memory_store.size,
                )
            except Exception as exc:
                logger.warning(
                    "Cycle %d: consolidation raised %s -- continuing without "
                    "consolidation pass.", cycle_num, exc,
                )

        return CycleResult(
            cycle_num=cycle_num,
            n_episodes_used=len(episode_pairs),
            n_general_used=n_general_used,
            epochs_completed=epochs_done,
            final_train_loss=final_loss,
            mmlu_retention_ratio=mmlu_retention_ratio,
            aborted=aborted,
            checkpoint_path=ckpt_path,
            mmlu_retention=mmlu_retention,
            n_retroverified=n_retroverified,
            n_retropruned=n_retropruned,
            n_deferred_promoted=n_deferred_promoted,
            n_deferred_ttl_dropped=n_deferred_ttl_dropped,
            n_deferred_kept=n_deferred_kept,
            deferred_buffer_size_at_entry=deferred_buffer_size_at_entry,
            reconsider_fired=reconsider_fired,
            n_consolidated_clusters=n_consolidated_clusters,
            n_consolidated_removed=n_consolidated_removed,
        )

    def load_checkpoint(self, cycle_num: int) -> dict:
        """Load cycle metadata from a saved checkpoint directory."""
        ckpt_dir = self.output_dir / f"cycle_{cycle_num}"
        meta_path = ckpt_dir / "meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt_dir}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
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

        Training target is ``reasoning_chain``, NOT ``answer`` (thesis Section
        4.3 -- Chain-of-Thought Supervision). Only ID benchmarks flow into the
        training pool (see config.TRAINING_BENCHMARKS); OOD tags are skipped
        and untagged entries pass through for legacy / unit-test paths.
        """
        threshold = self.config.min_u_stored_for_training
        pairs = []
        n_empty = 0
        n_too_short = 0
        n_ood_skipped = 0
        n_loop_filtered = 0
        chain_lengths = []
        preview_texts = []

        for entry in memory_store.all_entries():
            if entry.u_stored < threshold:
                continue

            # Branch C Goal 4 item 3: multi-source gate. The entry's
            # effective benchmark set is
            #     {source_benchmark} ∪ merged_source_benchmarks
            # (both optional). If any member is in TRANSFER_BENCHMARKS,
            # the entry is OOD-contaminated and excluded -- otherwise a
            # post-consolidation entry from FEVER ∪ TruthfulQA would
            # leak held-out TruthfulQA content into the training pool.
            # Untagged entries pass through (legacy/unit-test default).
            effective_bms = set()
            if entry.source_benchmark is not None:
                effective_bms.add(entry.source_benchmark)
            for extra in getattr(entry, "merged_source_benchmarks", ()):
                effective_bms.add(extra)
            if effective_bms and not effective_bms.issubset(TRAINING_BENCHMARKS):
                n_ood_skipped += 1
                continue

            chain = (entry.reasoning_chain or "").strip()
            chain_len = len(chain)
            chain_lengths.append(chain_len)

            if chain_len == 0:
                n_empty += 1
                chain = (entry.answer or "").strip()
            elif chain_len < 10:
                n_too_short += 1
                chain = (entry.answer or "").strip()
            elif is_repetitive_loop(chain, self.config):
                # Goal 4: loop-contaminated chains fall back to entry.answer,
                # matching the empty / too-short fallback branch. Phase 1a
                # Cycle-0 showed ~30% of STOREd samples on Flan-T5 were loops
                # (FEVER 52%, StrategyQA 56%); left unfiltered, the SIL
                # training target is "yes yes yes ..."-style garbage.
                n_loop_filtered += 1
                chain = (entry.answer or "").strip()

            pairs.append(QAPair(question=entry.question, answer=chain))
            preview_texts.append(chain.replace("\n", " ").strip()[:120])

        previews = random.sample(preview_texts, min(3, len(preview_texts))) if preview_texts else []
        self._last_chain_diagnostics = {
            "count": len(pairs),
            "min_len": min(chain_lengths) if chain_lengths else 0,
            "avg_len": (sum(chain_lengths) / len(chain_lengths)) if chain_lengths else 0.0,
            "max_len": max(chain_lengths) if chain_lengths else 0,
            "n_empty": n_empty,
            "n_too_short": n_too_short,
            "n_loop_filtered": n_loop_filtered,
            "n_ood_skipped": n_ood_skipped,
            "previews": previews,
        }

        if n_ood_skipped > 0:
            logger.info(
                "Training-pool gate skipped %d OOD episodes (transfer benchmarks); "
                "kept %d ID episodes.",
                n_ood_skipped, len(pairs),
            )

        return pairs

    def _log_chain_diagnostics(self, cycle_num: int) -> None:
        """Log chain quality diagnostics for the latest _collect_episodes call."""
        diag = self._last_chain_diagnostics or {}
        count_raw = diag.get("count", 0)
        count = count_raw if isinstance(count_raw, int) else 0

        if count == 0:
            logger.info("Cycle %d: chain diagnostics -- no collected episodes.", cycle_num)
            return

        min_len_raw = diag.get("min_len", 0)
        avg_len_raw = diag.get("avg_len", 0.0)
        max_len_raw = diag.get("max_len", 0)
        n_empty_raw = diag.get("n_empty", 0)
        n_too_short_raw = diag.get("n_too_short", 0)
        n_loop_raw = diag.get("n_loop_filtered", 0)

        min_len = min_len_raw if isinstance(min_len_raw, int) else 0
        avg_len = float(avg_len_raw) if isinstance(avg_len_raw, (int, float)) else 0.0
        max_len = max_len_raw if isinstance(max_len_raw, int) else 0
        n_empty = n_empty_raw if isinstance(n_empty_raw, int) else 0
        n_too_short = n_too_short_raw if isinstance(n_too_short_raw, int) else 0
        n_loop = n_loop_raw if isinstance(n_loop_raw, int) else 0
        n_fallback = n_empty + n_too_short + n_loop

        logger.info(
            "Cycle %d: chain diagnostics -- length min/avg/max = %d / %.1f / %d",
            cycle_num,
            min_len,
            avg_len,
            max_len,
        )
        logger.info(
            "Cycle %d: chain diagnostics -- fallback to entry.answer: %d "
            "(empty=%d, too_short=%d, loop_filtered=%d)",
            cycle_num,
            n_fallback,
            n_empty,
            n_too_short,
            n_loop,
        )

        previews = diag.get("previews", [])
        if isinstance(previews, list):
            for i, preview in enumerate(previews, start=1):
                logger.info("Cycle %d: chain preview %d/3: %s", cycle_num, i, str(preview))

    # _mix() removed in v2 architecture (2026-05-06). The general-domain
    # anti-forgetting mix is replaced by Fix 3 (loss reweighting) + Fix 8
    # (LoRA's parameter budget) + Fix 4 (multi-modal retention probe).
    # See PRODUCTION_NEXT_SESSION_PLAN v2 Phase 0 Fix 13 for the rationale.

    # ------------------------------------------------------------------ #
    # Fine-tuning                                                          #
    # ------------------------------------------------------------------ #

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Construct the SIL optimiser -- 8-bit AdamW with torch AdamW fallback.

        Branch C primary path: ``bitsandbytes.optim.AdamW8bit`` cuts optimiser
        state VRAM by ~50% relative to fp32 AdamW, which is what buys us Full
        FT headroom on 32 GB for a 3B model at batch=4 with grad checkpointing
        (see module docstring). Graceful fallback to ``torch.optim.AdamW`` on
        ImportError / non-CUDA device keeps CPU-only CI + smoke tests working.

        v2 Fix 8 — LoRA path: when ``self._lora_active`` is True, build the
        optimiser over the small adapter parameter set (everything with
        ``requires_grad=True`` after the peft wrap) and use
        ``cfg.lora_learning_rate`` (10× the backbone-FT lr — adapters are
        small enough to take a higher step size).
        """
        cfg = self.config
        if getattr(self, "_lora_active", False):
            lr = float(getattr(cfg, "lora_learning_rate", 2e-4))
            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                logger.warning(
                    "LoRA path: no trainable parameters found after wrap; "
                    "falling back to full parameter set.",
                )
                params = list(self.model.parameters())
            logger.info(
                "Optimiser parameter set: %d adapter tensors (LoRA path; "
                "lr=%.2e).", len(params), lr,
            )
        else:
            lr = cfg.learning_rate
            params = list(self.model.parameters())

        want_8bit = bool(getattr(cfg, "use_8bit_adamw", False))
        on_cuda = str(self.device).startswith("cuda")

        if want_8bit and on_cuda:
            try:
                import bitsandbytes as bnb  # type: ignore
                optim = bnb.optim.AdamW8bit(params, lr=lr)
                logger.info("Fine-tuning optimiser: bitsandbytes AdamW8bit (primary path).")
                return optim
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed; falling back to torch.optim.AdamW. "
                    "Install with `pip install bitsandbytes` on a CUDA host to "
                    "recover Full-FT VRAM headroom."
                )
            except Exception as exc:
                logger.warning(
                    "bitsandbytes AdamW8bit init failed (%s); falling back to torch.optim.AdamW.",
                    exc,
                )
        elif want_8bit and not on_cuda:
            logger.info(
                "cfg.use_8bit_adamw=True but device=%s is not CUDA; using torch.optim.AdamW.",
                self.device,
            )

        optim = TorchAdamW(params, lr=lr)
        logger.info("Fine-tuning optimiser: torch.optim.AdamW.")
        return optim

    def _finetune(
        self,
        train_pairs: List[QAPair],
        theta_prev: List[torch.Tensor],
    ) -> Tuple[int, float]:
        """Fine-tune the model on train_pairs with L2 regularisation.

        Loss = CrossEntropy(output, labels) + (lambda/2) * ||theta - theta_prev||^2

        Returns (epochs_completed, final_avg_loss).
        """
        cfg = self.config
        dataset = QADataset(train_pairs, self.tokenizer)
        pad_id = self.tokenizer.pad_token_id or 0

        def _collate(batch):
            # Dynamic right-pad to the longest sequence in the batch. pad_id
            # fills input_ids; attention_mask is 0 on pad; labels get -100
            # on pad so cross-entropy ignores the padding region.
            max_len = max(b["input_ids"].shape[0] for b in batch)
            input_ids      = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros(len(batch), max_len,           dtype=torch.long)
            labels         = torch.full((len(batch), max_len), -100,    dtype=torch.long)

            for i, b in enumerate(batch):
                L = b["input_ids"].shape[0]
                input_ids[i, :L]      = b["input_ids"]
                attention_mask[i, :L] = b["attention_mask"]
                labels[i, :L]         = b["labels"]

            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate)

        optimizer = self._build_optimizer()

        # Gradient checkpointing saves ~40% activation memory on 3B models
        # at the cost of ~20% extra compute per step -- the right trade on a
        # 32 GB 5090 where activation memory dominates. Silent no-op on mock
        # models that lack the method.
        prev_use_cache = None
        try:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                    prev_use_cache = self.model.config.use_cache
                    self.model.config.use_cache = False
                logger.info("Gradient checkpointing enabled for SIL fine-tune.")
        except Exception as exc:
            logger.debug("Gradient checkpointing unavailable (%s); continuing.", exc)

        # SIL fine-tune forces eager execution to recover the ~80-150 MiB
        # of activation memory that torch.compile's SDPA backward kernel
        # holds (CUDA OOM 2026-04-27 incidents #3 and #4). The model loader
        # at caem/model_loader.py monkey-patches model.forward via
        # ``model.forward = torch.compile(model.forward, ...)``, so the
        # OptimizedModule._orig_mod swap does not apply; instead we use
        # torch.compiler.set_stance("force_eager") which bypasses any
        # compiled callable (decorator, monkey-patch, or context-manager)
        # process-wide for the duration of the fine-tune. The default
        # stance is restored at the end so eval-time generation keeps its
        # compiled per-query latency target. Compile is also speed-negative
        # at SIL scale (~500 steps/cycle): the JIT cost plus recompile-
        # limit churn outweighs the ~5-10% per-step speedup.
        try:
            import torch.compiler as _torch_compiler
            _torch_compiler.set_stance("force_eager")
            logger.info("Fine-tune: torch.compiler stance set to force_eager (memory headroom).")
        except Exception as exc:
            logger.warning("Fine-tune: torch.compiler.set_stance unavailable (%s); compile remains active.", exc)
        self.model.train()
        self.model.to(self.device)

        # theta_prev placement: kept on CPU for the L2-anchor compute. On the
        # consumer-grade 32 GiB envelope, every prior attempt to GPU-resident
        # the anchor (fp32 incident #1, bf16 incidents #2-#5) ended in OOM
        # because the live model + 8-bit AdamW state + grad buffers + auxiliary
        # verifier models leave no headroom for an additional 6 GB anchor.
        # Streaming each parameter's anchor over PCIe at L2-step time costs
        # ~50 ms per parameter * ~130 params per step ~= 6.5 s/step extra
        # wall-clock, which is the necessary trade-off to get the trajectory
        # to run at all on this hardware. Keep theta_prev on CPU; _l2_penalty
        # below will move each parameter's anchor onto GPU just-in-time.
        theta_prev_for_penalty = theta_prev
        logger.info("theta_prev kept on CPU (memory-safe path); L2 anchor streams per-parameter via PCIe.")

        param_dtype = next(self.model.parameters()).dtype
        use_cuda = str(self.device).startswith("cuda")
        if use_cuda and param_dtype == torch.float16 and torch.cuda.is_bf16_supported():
            logger.info("Fine-tuning: switching fp16 -> bf16 for numerical stability.")
            self.model.to(dtype=torch.bfloat16)
            param_dtype = torch.bfloat16
        amp_enabled = use_cuda and param_dtype in (torch.float16, torch.bfloat16)
        amp_dtype = torch.float16 if param_dtype == torch.float16 else torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
        if amp_enabled:
            logger.info("Fine-tuning with AMP (%s).", str(param_dtype))

        epochs_done = 0
        final_loss  = 0.0
        grad_accum_steps = max(int(getattr(cfg, "grad_accum_steps", 1)), 1)

        for epoch in range(cfg.epochs_per_cycle):
            epoch_loss = 0.0
            n_batches  = 0
            micro_step = 0
            optimizer.zero_grad(set_to_none=True)

            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                if int((labels != -100).sum().item()) == 0:
                    continue

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    ce_loss = outputs.loss

                if not torch.isfinite(ce_loss):
                    logger.warning(
                        "Non-finite CE loss encountered; discarding %d "
                        "accumulated micro-batch(es) and skipping.",
                        micro_step,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    micro_step = 0
                    continue

                # v2 Fix 8 — LoRA path skips the backbone-anchor L2 term.
                # The frozen base + bounded adapter parameter budget IS
                # the implicit anchor (Biderman 2024); a backbone L2
                # would mix in a zero-anchored regulariser over weights
                # that aren't being updated, which is meaningless and
                # wastes wall-clock on the per-step PCIe stream.
                if getattr(self, "_lora_active", False):
                    raw_loss = ce_loss
                else:
                    l2_loss = self._l2_penalty(theta_prev_for_penalty)
                    raw_loss = ce_loss + (cfg.l2_lambda / 2.0) * l2_loss

                if not torch.isfinite(raw_loss):
                    logger.warning(
                        "Non-finite total loss encountered; discarding %d "
                        "accumulated micro-batch(es) and skipping.",
                        micro_step,
                    )
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
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    micro_step = 0

                epoch_loss += raw_loss.item()
                n_batches  += 1

            if micro_step > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                "  Epoch %d/%d -- loss: %.4f (grad_accum=%d, eff_batch=%d)",
                epoch + 1, cfg.epochs_per_cycle, avg_loss,
                grad_accum_steps, cfg.batch_size * grad_accum_steps,
            )
            epochs_done += 1
            final_loss   = avg_loss

        # Restore the default torch.compiler stance so verifier / RAG
        # inference re-enters the compiled fast path for eval-time generation.
        try:
            import torch.compiler as _torch_compiler
            _torch_compiler.set_stance("default")
            logger.info("Fine-tune: torch.compiler stance restored to default (compile re-enabled for inference).")
        except Exception as exc:
            logger.warning("Fine-tune: torch.compiler.set_stance restore failed (%s).", exc)
        self.model.eval()
        # Disable grad checkpointing so inference paths (verifier, RAG) are
        # not slowed by recomputation after SIL completes.
        try:
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
            if prev_use_cache is not None and hasattr(self.model, "config"):
                self.model.config.use_cache = prev_use_cache
        except Exception as exc:
            logger.debug("Gradient checkpointing teardown noop (%s).", exc)
        return epochs_done, final_loss

    def _l2_penalty(self, theta_prev: List[torch.Tensor]) -> torch.Tensor:
        """Compute ||theta - theta_prev||^2 summed over all parameters.

        theta_prev is bf16 when GPU-resident (matches the live-model dtype)
        and fp32 when CPU-resident (the safety fallback). Per-parameter
        squared sums are accumulated in fp32 so the running total has full
        precision; only the per-element diff is in the storage dtype, which
        matches the live model and avoids the upcast that previously doubled
        peak VRAM during the L2 step (CUDA OOM 2026-04-27 incident).
        """
        penalty = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for p, p0 in zip(self.model.parameters(), theta_prev):
            # One-shot device + dtype conversion: when p0 is fp32 on CPU
            # (the memory-safe default for the consumer-grade 32 GiB envelope)
            # we cast to bf16 during the PCIe transfer rather than after, so
            # the transient on-GPU buffer is half the size (bf16 not fp32).
            # When p0 is already on the device with matching dtype, the call
            # is a no-op and returns p0 itself.
            if p0.device == self.device:
                ref = p0 if p0.dtype == p.dtype else p0.to(p.dtype)
            else:
                ref = p0.to(self.device, dtype=p.dtype, non_blocking=True)
            diff = p - ref
            penalty = penalty + (diff * diff).sum().to(torch.float32)
            del ref, diff  # hint GC: release transients before next iteration
        return penalty

    # ------------------------------------------------------------------ #
    # Forgetting check                                                     #
    # ------------------------------------------------------------------ #
    #
    # The deprecated ``_forgetting_score`` diagnostic (in-distribution EM
    # probe) was removed 2026-04-22 along with the contradiction-veto dead
    # code. The abort guard is driven entirely by ``_mmlu_score`` --- an
    # out-of-distribution 4-choice benchmark that is never in the training
    # pool and is the scientifically correct probe for catastrophic
    # forgetting. In-distribution probes are systematically insensitive to
    # the failure mode the guard exists to catch.

    def measure_mmlu(self, n: int = 200) -> float:
        """Public entry point for MMLU scoring -- see ``_mmlu_score``."""
        return self._mmlu_score(n=n)

    def _mmlu_score(self, n: int = 200) -> float:
        """Measure MMLU 4-choice accuracy as the neutral forgetting proxy.

        ChatML-wraps the prompt (matches Qwen's instruction-tuning distribution)
        and slices the generated suffix off ``model.generate`` output before
        decoding. Serves as both the abort-guard trigger (via post/pristine
        retention ratio) and the CES RET reporting axis (Ch5 Table 5.2).

        Returns NaN if the MMLU dataset is unavailable (no internet / no HF
        cache); NaN is propagated cleanly to CSV so the experiment does not
        abort.
        """
        try:
            from datasets import load_dataset
            ds = load_dataset("cais/mmlu", "all", split="validation")
            ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        except Exception as exc:
            logger.warning(
                "MMLU dataset load failed (%s); mmlu_retention set to NaN.", exc
            )
            return float("nan")

        choices_labels = ["A", "B", "C", "D"]
        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for item in ds:
                item_d: dict = item  # type: ignore[assignment]
                question = str(item_d.get("question", ""))
                choices  = item_d.get("choices", [])
                answer_idx = item_d.get("answer", -1)
                if not question or not choices or answer_idx < 0:
                    continue
                if not (0 <= answer_idx < 4):
                    continue

                choice_str = "\n".join(
                    f"{choices_labels[i]}. {c}"
                    for i, c in enumerate(choices)
                    if i < len(choices_labels)
                )
                user_content = (
                    f"Question: {question}\n"
                    f"Choices:\n{choice_str}\n"
                    f"Answer with just the letter (A, B, C, or D)."
                )
                prompt_text = self._wrap_chatml_user(user_content)

                enc = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    add_special_tokens=False,
                ).to(self.device)
                input_ids = enc["input_ids"]
                input_len = int(input_ids.shape[1])
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=8,
                    do_sample=False,
                )
                if hasattr(out, "shape") and out.shape[1] > input_len:
                    gen_tokens = out[0, input_len:]
                else:
                    gen_tokens = out[0]
                pred = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().upper()
                expected = (
                    choices_labels[answer_idx]
                    if answer_idx < len(choices_labels)
                    else ""
                )
                if pred.startswith(expected):
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info("MMLU retention: %.4f (%d/%d correct)", accuracy, correct, total)
        return accuracy

    def _wrap_chatml_user(self, user_content: str) -> str:
        """Wrap a user string in ChatML with ``add_generation_prompt=True``.

        Falls back to the raw user content when ``apply_chat_template`` is
        unavailable or returns a non-string (mock tokenizers in unit tests).
        """
        tok = self.tokenizer
        messages = [{"role": "user", "content": user_content}]
        if hasattr(tok, "apply_chat_template"):
            try:
                rendered = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                if isinstance(rendered, str):
                    return rendered
            except Exception:
                pass
        return user_content

    # ------------------------------------------------------------------ #
    # Weight management                                                    #
    # ------------------------------------------------------------------ #

    def _snapshot_weights(self) -> List[torch.Tensor]:
        """Return a deepcopy of model parameters for the L2 anchor / restore.

        fp32 regardless of model dtype so ``_l2_penalty`` can sum squared
        differences over billions of parameters without fp16 overflow.

        v2 Fix 8 — LoRA path: snapshots ONLY the trainable adapter tensors
        (``requires_grad=True``). Skips the frozen base, which would
        otherwise eat ~6 GB of CPU RAM per cycle for a 3B-parameter
        backbone that the SIL loop never touches anyway. ``_restore_weights``
        is updated in lockstep to only restore the same trainable subset.
        """
        if getattr(self, "_lora_active", False):
            return [
                p.detach().cpu().float().clone()
                for p in self.model.parameters()
                if p.requires_grad
            ]
        return [p.detach().cpu().float().clone() for p in self.model.parameters()]

    def _restore_weights(self, theta_prev: List[torch.Tensor]) -> None:
        """Restore model parameters to theta_prev in-place.

        v2 Fix 8 — LoRA path: restores ONLY the trainable adapter tensors,
        matching the subset that ``_snapshot_weights`` captured. The
        frozen base is never touched on the LoRA path so it does not need
        restoring.
        """
        if getattr(self, "_lora_active", False):
            model_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            model_params = list(self.model.parameters())
        if len(theta_prev) != len(model_params):
            raise RuntimeError(
                f"theta_prev has {len(theta_prev)} tensors but model has "
                f"{len(model_params)} parameters. Model definition may have "
                f"changed between _snapshot_weights() and _restore_weights()."
            )
        with torch.no_grad():
            for p, p0 in zip(model_params, theta_prev):
                p.copy_(p0.to(self.device))
        logger.info(
            "Model weights restored to pre-cycle state (%s).",
            "LoRA adapter only" if getattr(self, "_lora_active", False)
            else "full backbone",
        )

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(
        self,
        cycle_num: int,
        seed: int,
        epochs_done: int,
        final_loss: float,
        mmlu_retention_ratio: float,
        aborted: bool,
        theta_prev: Optional[List[torch.Tensor]],
    ) -> str:
        """Save model weights + metadata to outputs/cycle_{n}/.

        Phase 1a disk-optimisation (2026-04-22): Qwen-2.5-3B bf16 state_dicts
        are ~6.2 GB each. At 10 cycles that's ~62 GB — larger than the free
        space on a 150 GB Vast cgroup. Two-tier retention:

          1. Local rolling-N (always on): after each save, delete ``model.pt``
             for cycle_{n-2} if that cycle isn't 0 and isn't the final.
             Keeps cycle_0 (baseline) + last 2 cycles + final cycle.
             Peak local disk ~25 GB instead of ~62 GB.

          2. HF-Hub milestone offload (env-gated CAEM_HF_OFFLOAD=1):
             upload cycles 0, N/2, and N (final) to
             aksaN000/caem-passage-index-21m:checkpoints/<run>/cycle_<n>/
             for cross-instance resume insurance. Selective (not every
             cycle) to stay within free-tier storage headroom. Failures
             are non-fatal — the local rolling-N is the guarantee; HF is
             opportunistic backup.

        ``meta.pkl`` is tiny and always kept under every cycle_{n}/ dir
        so downstream scripts can find cycle boundaries without the
        weights file.
        """
        ckpt_dir = self.output_dir / f"cycle_{cycle_num}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # v2 Fix 8 — adapter-only checkpoint on the LoRA path.
        # peft's save_pretrained writes ``adapter_config.json`` +
        # ``adapter_model.safetensors`` (~120 MB for r=32 on a 3B base)
        # to ckpt_dir/adapter/. Falls through to legacy full state_dict
        # if peft is unavailable or the save fails. The Tier-1
        # rolling-N retention below targets ``model.pt`` only — adapter
        # files are kept for every cycle (small enough that 10×120 MB
        # fits trivially).
        weights_path = ckpt_dir / "model.pt"
        adapter_dir = ckpt_dir / "adapter"
        if getattr(self, "_lora_active", False):
            try:
                # peft.PeftModel exposes save_pretrained
                if hasattr(self.model, "save_pretrained"):
                    self.model.save_pretrained(str(adapter_dir))
                    logger.info(
                        "Cycle %d: LoRA adapter saved to %s/", cycle_num, adapter_dir,
                    )
                else:
                    # Fallback: manually pickle the trainable subset.
                    adapter_dir.mkdir(parents=True, exist_ok=True)
                    state = {
                        n: p.detach().cpu()
                        for n, p in self.model.named_parameters()
                        if p.requires_grad
                    }
                    torch.save(state, adapter_dir / "adapter_model.pt")
                    logger.info(
                        "Cycle %d: LoRA adapter (manual fallback) saved to %s/",
                        cycle_num, adapter_dir,
                    )
            except Exception as exc:
                logger.error(
                    "Cycle %d: LoRA adapter save failed (%s); falling back "
                    "to full state_dict at %s.", cycle_num, exc, weights_path,
                )
                try:
                    torch.save(self.model.state_dict(), weights_path)
                except Exception as exc2:
                    logger.error(
                        "Cycle %d: full state_dict fallback also failed: %s",
                        cycle_num, exc2,
                    )
                    raise
        else:
            try:
                torch.save(self.model.state_dict(), weights_path)
            except Exception as exc:
                logger.error(
                    "Cycle %d: failed to save model weights to %s: %s",
                    cycle_num, weights_path, exc,
                )
                raise

        meta = {
            "cycle_num": cycle_num,
            "seed": seed,
            "epochs_done": epochs_done,
            "final_loss": float(final_loss),
            "mmlu_retention_ratio": float(mmlu_retention_ratio),
            "aborted": bool(aborted),
            "timestamp": time.time(),
            "theta_prev_was_restored": theta_prev is not None,
        }
        meta_path = ckpt_dir / "meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        # --- Tier 1: local rolling-N retention (always on) --------------
        # Keep cycle_0 (baseline) + last 2 + final. Delete cycle_{n-2}/model.pt
        # when cycle_num >= 2 and cycle_{n-2} is not 0 and not the final cycle.
        total_cycles = getattr(self, "_total_cycles", 10)
        if cycle_num >= 2:
            old_cycle = cycle_num - 2
            if old_cycle > 0 and old_cycle != total_cycles:
                old_path = self.output_dir / f"cycle_{old_cycle}" / "model.pt"
                if old_path.exists():
                    try:
                        size_mb = old_path.stat().st_size / (1024 * 1024)
                        old_path.unlink()
                        logger.info(
                            "Cycle %d: reclaimed %.0f MB from cycle_%d/model.pt "
                            "(rolling-N retention; meta.pkl preserved).",
                            cycle_num, size_mb, old_cycle,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Cycle %d: could not delete old cycle_%d/model.pt: %s",
                            cycle_num, old_cycle, exc,
                        )

        # --- Tier 2: Google Drive every-cycle offload (env-gated) -------
        # When CAEM_GDRIVE_OFFLOAD=1, upload EVERY cycle's adapter via rclone.
        # v2 (2026-05-06): per-cycle artefacts land under
        # gdrive:caem-phase1a/<bucket>/<run_name>/cycle_<n>/  where <bucket>
        # is read from CAEM_GDRIVE_BUCKET (defaults to "v2"). v1 cycles 0-4
        # are archived under gdrive:caem-phase1a/archive_v1/full_run/ as
        # failure-mode evidence (see branch_C_log 2026-05-06).
        if os.environ.get("CAEM_GDRIVE_OFFLOAD", "0") == "1":
            import subprocess
            run_name = self.output_dir.name
            gd_bucket = os.environ.get("CAEM_GDRIVE_BUCKET", "v2")
            remote_path = f"gdrive:caem-phase1a/{gd_bucket}/{run_name}/cycle_{cycle_num}/"
            try:
                result = subprocess.run(
                    ["rclone", "copy", str(weights_path), remote_path,
                     "--transfers", "4", "--checkers", "8"],
                    capture_output=True, text=True, timeout=1800,
                )
                if result.returncode == 0:
                    logger.info(
                        "Cycle %d: gdrive offload OK -> %smodel.pt",
                        cycle_num, remote_path,
                    )
                else:
                    logger.warning(
                        "Cycle %d: gdrive offload failed (rc=%d): %s -- local "
                        "rolling-N retention still active; training continues.",
                        cycle_num, result.returncode, result.stderr[:500],
                    )
            except Exception as exc:
                logger.warning(
                    "Cycle %d: gdrive offload errored (%s) -- local rolling-N "
                    "retention still active; training continues.",
                    cycle_num, exc,
                )

        logger.info(
            "Cycle %d: checkpoint saved to %s (aborted=%s).",
            cycle_num, ckpt_dir, aborted,
        )
        return str(ckpt_dir)
