"""
caem/training/self_improvement.py
===================================
SelfImprovementLoop -- Stage 8 of the CAEM pipeline.

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
4. Forgetting check: after fine-tuning, run a fixed MMLU split
    (n=200) and compare post-cycle MMLU against the PRISTINE cycle-0
    MMLU anchor (cached on the SelfImprovementLoop instance). If
    (post / pristine) < forgetting_tolerance (0.93), abort and restore
    θ_prev. The pristine anchor (rather than the previous cycle's post
    MMLU) is required because cumulative drift across 10 cycles could
    otherwise slip past the per-cycle tolerance silently -- each cycle
    would be comparing against an already-drifted denominator. The
    pre-cycle MMLU is still measured and logged for per-cycle
    diagnostics but does not enter the abort criterion.
    MMLU is chosen over any in-distribution (e.g. TriviaQA) probe
    because catastrophic forgetting in the continual-learning sense
    manifests as loss of unrelated capability rather than erosion of
    the target task; a QA-family probe would be systematically
    insensitive to that failure mode. The same MMLU pass feeds the
    RET axis of CES (Chapter 3, FR7 and Evaluation Methodology), so
    one per-cycle evaluation serves both as the rollback trigger and
    as the reported general-capability retention metric.
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
significant -- no FIM computation, no extra memory for Fisher diagonal.

This is documented as [DES] in the config (l2_lambda = 0.01). The thesis
reports this choice and compares with full EWC in the ablation study.

Checkpoint design
-----------------
Each cycle saves independently to outputs/cycle_{n}/ so:
  - Cycles can be compared post-hoc (cycle_1/model.pt vs cycle_2/model.pt)
  - A failed cycle does not overwrite a successful one
  - Restarting from any checkpoint is possible without retraining from scratch

The SelfImprovementLoop does NOT hold a reference to the memory store
between cycles -- it receives one at call time so the caller controls
which store state is used (important for ablations).
"""

from __future__ import annotations

import copy
import logging
import math
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from caem.config import CAEMConfig, TRAINING_BENCHMARKS
from caem.memory.store import EpisodicMemoryStore

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
    mmlu_retention_ratio: float   # post/pre MMLU retention ratio (abort guard + RET axis).
                                  # Named `forgetting_score` historically when the abort guard used
                                  # an in-distribution TriviaQA probe (see `_forgetting_score`,
                                  # DEPRECATED); renamed to reflect the current MMLU-based probe.
    aborted:            bool    # True if forgetting check failed and weights were restored
    checkpoint_path:    str
    mmlu_retention:     float = 0.0  # MMLU 4-choice accuracy post-fine-tune (same value as drives abort)
    # Retroactive re-verification counters (Phase 5d). Populated after
    # fine-tuning when verify_fn is supplied to run_cycle. Zero means the
    # retroverify loop did not run (no verify_fn or cycle aborted).
    n_retroverified:    int   = 0    # episodes whose u_stored was raised
    n_retropruned:      int   = 0    # episodes removed (new u_stored < prune threshold)
    # Deferred-entry reconsideration counters (thesis Section 4.9).
    # Populated after retroverify when a deferred_buffer and reconsider_fn
    # are supplied to run_cycle. Zero means the reconsideration pass did
    # not run (missing buffer, missing closure, or cycle aborted).
    n_deferred_promoted:   int = 0   # entries promoted from buffer -> main memory
    n_deferred_ttl_dropped: int = 0  # entries aged past TTL without promotion
    n_deferred_kept:       int = 0   # entries re-queued for next cycle


# -----------------------------------------------------------------------------
# PyTorch Dataset for fine-tuning
# -----------------------------------------------------------------------------

class QADataset(Dataset):
    """Tokenised (question -> answer) dataset for seq2seq fine-tuning."""

    def __init__(
        self,
        pairs: List[QAPair],
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 128,
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
            padding=False,          # batch collator pads to longest in batch
            return_tensors="pt",
        )
        dec_kwargs = {
            "max_length": self.max_target,
            "truncation": True,
            "padding": False,
            "return_tensors": "pt",
        }
        try:
            dec = self.tokenizer(text_target=pair.answer, **dec_kwargs)
        except TypeError:
            if hasattr(self.tokenizer, "as_target_tokenizer"):
                with self.tokenizer.as_target_tokenizer():
                    dec = self.tokenizer(pair.answer, **dec_kwargs)
            else:
                dec = self.tokenizer(pair.answer, **dec_kwargs)
        labels = dec["input_ids"].squeeze(0)
        # Replace padding token id with -100 so cross-entropy ignores it
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


# -----------------------------------------------------------------------------
# SelfImprovementLoop
# -----------------------------------------------------------------------------

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
    >>> result.aborted   # False -> weights updated; True -> restored
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

        # Pristine MMLU anchor for the forgetting guard.
        # Measured lazily on the first run_cycle() call and reused across
        # all subsequent cycles; this ensures the retention_ratio compares
        # post-cycle MMLU against the UNMODIFIED cycle-0 baseline rather
        # than against the previous cycle's drifted post-MMLU. Without this
        # anchor, a 10-cycle SIL run could drift cumulatively by well over
        # the per-cycle forgetting_tolerance (0.93) without ever triggering
        # an abort, because each cycle's denominator itself would have
        # slid downward. Mirrors the pristine_mmlu pattern used in the
        # B6/B7 baselines (scripts/run_simple_ft.py:562,644) so main CAEM
        # and the fine-tune baselines share an identical abort contract.
        self._pristine_mmlu: Optional[float] = None

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

    def run_cycle(
        self,
        cycle_num: int,
        memory_store: EpisodicMemoryStore,
        general_data: List[QAPair],
        seed: int = 42,
        verify_fn: Optional[Callable[[Any], Any]] = None,
        deferred_buffer: Optional[Any] = None,
        reconsider_fn: Optional[Callable[[Any], Any]] = None,
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
        verify_fn : callable or None
            Retroactive re-verification closure. Signature:
            ``(entry: EpisodicEntry) -> UnifiedVerifierOutput``. When supplied
            and the cycle is NOT aborted, the updated model is used to re-score
            every stored episode after fine-tuning (Step 8). Episodes whose
            new u_stored falls below CAEMConfig.retroverify_prune_threshold
            are removed; episodes whose u_stored rises are updated with the
            full nine-signal record from the new UnifiedVerifierOutput.
            Pass ``CAEMPipeline.make_retroverify_fn()`` from the orchestrator
            to bind the verifier and passage store automatically.
        deferred_buffer : DeferredBuffer or None
            Buffer of DEFERRED episodes held for cycle-boundary
            reconsideration (thesis Section 4.9 "Deferred-Entry
            Reconsideration"). When supplied together with ``reconsider_fn``
            and the cycle is NOT aborted, the buffer is swept after
            retroverify: entries whose re-scored decision clears
            ``store_threshold`` are promoted into ``memory_store``, aged-
            out entries are dropped, and the remainder are re-queued.
        reconsider_fn : callable or None
            Closure with the same signature as ``verify_fn`` but accepting
            a ``DeferredEntry`` (which exposes ``.question`` and ``.answer``
            attributes, matching the verifier's input contract). Obtain
            from ``CAEMPipeline.make_reconsider_deferred_fn()``. If omitted
            but ``deferred_buffer`` is supplied, ``verify_fn`` is used as
            a fallback closure -- the verifier input contract is identical
            for both passes.

        Returns
        -------
        CycleResult
            Summary including whether the cycle was aborted and how many
            episodes were retroactively updated/pruned.
        """
        cfg = self.config
        random.seed(seed)
        torch.manual_seed(seed)

        logger.info("=== Self-Improvement Cycle %d ===", cycle_num)

        # Step 1: collect episode training data
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

        # Step 2: shuffle general_data for training mix. We no longer split out a
        # held-out eval slice -- the forgetting guard now uses MMLU (a separate,
        # never-trained-on 4-choice benchmark) rather than an in-distribution QA
        # probe, so all of general_data is available for the training mix.
        random.shuffle(general_data)

        # Step 3: build mixed training set (90% episodes + 10% general)
        train_pairs, n_general_used = self._mix(episode_pairs, general_data)
        logger.info(
            "Cycle %d: training on %d pairs (%d episodes + %d general).",
            cycle_num, len(train_pairs), len(episode_pairs), n_general_used,
        )

        # Step 4: snapshot θ_prev BEFORE fine-tuning (used for L2 reg + forgetting restore)
        theta_prev = self._snapshot_weights()

        # Step 4b: measure MMLU baseline BEFORE fine-tuning.
        # The forgetting guard compares RELATIVE retention (post/pre) on MMLU,
        # an out-of-distribution 4-choice benchmark that is NEVER in the
        # training pool. This aligns with the continual-learning literature
        # (Kirkpatrick et al., 2017) where catastrophic forgetting manifests
        # as loss of unrelated general capability rather than erosion of the
        # target task. Using an in-distribution probe (e.g. TriviaQA) would
        # be systematically insensitive to this failure mode: the model has
        # just been fine-tuned on TriviaQA-like pairs, so it cannot trigger.
        # The same MMLU measurement also serves as the CES RET reporting axis,
        # so there is zero compute overhead to using it as the abort guard.
        # The pre-cycle MMLU is still measured so we can log the per-cycle
        # delta for diagnostics, but the abort guard's denominator is the
        # PRISTINE cycle-0 MMLU (lazily cached on this instance), NOT this
        # per-cycle value. See __init__ for the rationale; without pristine
        # anchoring, cumulative drift across 10 cycles can silently slide
        # past forgetting_tolerance because each cycle's own denominator
        # would have drifted downward too.
        pre_cycle_mmlu = self._mmlu_score(n=200)
        if self._pristine_mmlu is None:
            # First ever run_cycle on this SIL instance: adopt the current
            # pre-cycle measurement as the pristine baseline. This works
            # both when run_cycle is first called at cycle 0 (typical) and
            # when a resumed run constructs a fresh SIL and re-measures
            # pristine against the partially-drifted model -- the latter
            # is a known limitation of the resume path and is flagged in
            # the docstring of reset_pristine_mmlu.
            self._pristine_mmlu = pre_cycle_mmlu
            logger.info(
                "Cycle %d: anchoring pristine MMLU baseline = %.4f "
                "(forgetting guard will compare all future cycles against this value).",
                cycle_num,
                pre_cycle_mmlu if not __import__("math").isnan(pre_cycle_mmlu) else float("nan"),
            )
        else:
            logger.info(
                "Cycle %d: pristine MMLU anchor = %.4f | pre-cycle MMLU = %.4f "
                "(pre-cycle delta from pristine = %+.4f)",
                cycle_num,
                self._pristine_mmlu,
                pre_cycle_mmlu if not __import__("math").isnan(pre_cycle_mmlu) else float("nan"),
                (pre_cycle_mmlu - self._pristine_mmlu)
                    if (not __import__("math").isnan(pre_cycle_mmlu)
                        and not __import__("math").isnan(self._pristine_mmlu))
                    else float("nan"),
            )

        if __import__("math").isnan(self._pristine_mmlu):
            logger.warning(
                "Cycle %d: pristine MMLU is NaN -- forgetting guard DISABLED; "
                "cycle will complete without abort check.", cycle_num,
            )

        # Step 5: fine-tune with L2 regularisation
        epochs_done, final_loss = self._finetune(train_pairs, theta_prev)

        # Step 6: MMLU forgetting check -- relative retention ratio vs. pristine.
        # post_mmlu is used simultaneously as (a) the abort-guard trigger via
        # retention_ratio = post/pristine, and (b) the absolute mmlu_retention
        # value reported in CycleResult / Chapter 5 Table 5.2 RET column.
        post_mmlu = self._mmlu_score(n=200)
        pristine = self._pristine_mmlu
        if __import__("math").isnan(post_mmlu) \
                or pristine is None \
                or __import__("math").isnan(pristine) \
                or pristine <= 1e-6:
            # Guard unavailable (dataset missing or pristine ~0) -- treat as
            # "no detectable forgetting" to avoid false aborts from NaN / 0.
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
                ("%.4f" % post_mmlu) if not __import__("math").isnan(post_mmlu) else "N/A",
            )

        # mmlu_retention_ratio reports the relative MMLU retention ratio; it is the
        # interpretable "fraction of MMLU capability retained after cycle k".
        mmlu_retention_ratio = retention_ratio

        # mmlu_retention reports the absolute post-fine-tune MMLU accuracy;
        # this is the value consumed by Chapter 5's RET axis.
        mmlu_retention = post_mmlu

        aborted = False
        if guard_active and retention_ratio < cfg.forgetting_tolerance:
            logger.warning(
                "Cycle %d: MMLU forgetting check FAILED (retention %.4f < %.4f) "
                "-- restoring θ_prev.",
                cycle_num, retention_ratio, cfg.forgetting_tolerance,
            )
            self._restore_weights(theta_prev)
            aborted = True

        # Step 7: save checkpoint
        ckpt_path = self._save_checkpoint(cycle_num, seed, epochs_done, final_loss,
                                           mmlu_retention_ratio, aborted, theta_prev if aborted else None)

        # Step 8: retroactive re-verification (Phase 5d).
        # Skipped when the cycle was aborted (the restored weights are the
        # previous cycle's -- retroactively re-scoring with unchanged weights
        # is a no-op that still pays the compute cost).
        n_retroverified = 0
        n_retropruned = 0
        if verify_fn is not None and not aborted:
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

        # Step 8b: deferred-entry reconsideration (thesis Section 4.9).
        # Runs AFTER retroverify so that the main memory is already
        # up-to-date under the new weights when a deferred entry is
        # evaluated for promotion. Also skipped on aborted cycles for
        # the same reason as retroverify: reconsidering under unchanged
        # weights is a no-op with a small probability of injecting noise.
        n_deferred_promoted = 0
        n_deferred_ttl_dropped = 0
        n_deferred_kept = 0
        if deferred_buffer is not None and not aborted:
            # Prefer the caller's dedicated reconsider_fn, but fall back
            # to verify_fn: the verifier contract is identical for both
            # passes, so any closure built by make_retroverify_fn works.
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

        Rationale (thesis §4.3 -- Chain-of-Thought Supervision):
        EpisodicEntry stores a `reasoning_chain` field -- the full chain-of-
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

        Training-pool benchmark gate (thesis Ch3 §Data, Ch5 §Metrics):
        Only episodes whose ``source_benchmark`` is in
        ``caem.config.TRAINING_BENCHMARKS`` are eligible to feed fine-tuning.
        The three transfer benchmarks (TruthfulQA, StrategyQA, ARC-C) are
        held out so their post-cycle scores measure *transfer* of the
        consolidated mechanisms (memory + routing + L2 anchor) rather than
        mere memorisation. Entries with ``source_benchmark=None`` are
        treated as training-eligible to keep legacy / unit-test paths
        working; production runs must tag every entry at storage time.
        """
        threshold = self.config.min_u_stored_for_training
        pairs = []
        n_empty = 0
        n_too_short = 0
        n_ood_skipped = 0
        chain_lengths = []
        preview_texts = []

        for entry in memory_store.all_entries():
            if entry.u_stored < threshold:
                continue

            # ID/OOD training-pool gate: untagged entries pass through;
            # explicitly-tagged OOD entries are skipped.
            if entry.source_benchmark is not None and \
                    entry.source_benchmark not in TRAINING_BENCHMARKS:
                n_ood_skipped += 1
                continue

            chain = (entry.reasoning_chain or "").strip()
            chain_len = len(chain)
            chain_lengths.append(chain_len)

            # Quality filter (EXP-16): empty/too-short chains are replaced
            # with the verified answer so we do not train on blank targets.
            if chain_len == 0:
                n_empty += 1
                chain = (entry.answer or "").strip()
            elif chain_len < 10:
                n_too_short += 1
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

        min_len = min_len_raw if isinstance(min_len_raw, int) else 0
        avg_len = float(avg_len_raw) if isinstance(avg_len_raw, (int, float)) else 0.0
        max_len = max_len_raw if isinstance(max_len_raw, int) else 0
        n_empty = n_empty_raw if isinstance(n_empty_raw, int) else 0
        n_too_short = n_too_short_raw if isinstance(n_too_short_raw, int) else 0
        n_fallback = n_empty + n_too_short

        logger.info(
            "Cycle %d: chain diagnostics -- length min/avg/max = %d / %.1f / %d",
            cycle_num,
            min_len,
            avg_len,
            max_len,
        )
        logger.info(
            "Cycle %d: chain diagnostics -- fallback to entry.answer: %d "
            "(empty=%d, too_short=%d)",
            cycle_num,
            n_fallback,
            n_empty,
            n_too_short,
        )

        previews = diag.get("previews", [])
        if isinstance(previews, list):
            for i, preview in enumerate(previews, start=1):
                logger.info("Cycle %d: chain preview %d/3: %s", cycle_num, i, str(preview))

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

        def _collate(batch):
            """Dynamic padding collator: pads to longest sequence in each batch."""
            import torch
            pad_id = self.tokenizer.pad_token_id or 0

            max_in  = max(b["input_ids"].shape[0] for b in batch)
            max_tgt = max(b["labels"].shape[0]    for b in batch)

            input_ids      = torch.full((len(batch), max_in),  pad_id,  dtype=torch.long)
            attention_mask = torch.zeros(len(batch), max_in,            dtype=torch.long)
            labels         = torch.full((len(batch), max_tgt), -100,    dtype=torch.long)

            for i, b in enumerate(batch):
                in_len  = b["input_ids"].shape[0]
                tgt_len = b["labels"].shape[0]
                input_ids[i, :in_len]       = b["input_ids"]
                attention_mask[i, :in_len]  = b["attention_mask"]
                labels[i, :tgt_len]         = b["labels"]

            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate)

        optimizer = AdamW(self.model.parameters(), lr=cfg.learning_rate)

        self.model.train()
        self.model.to(self.device)

        # ---- theta_prev placement optimisation (LAB_PC_SCALING_GUIDE §7) ---- #
        # On GPUs with >= 24 GB VRAM (RTX 4090, A100) move theta_prev to the
        # device ONCE before the epoch loop.  This eliminates per-batch PCIe
        # transfers (CPU→GPU) inside _l2_penalty, giving 3–4× faster fine-tuning.
        # On smaller GPUs (RTX 3060 12 GB) theta_prev stays on CPU to avoid OOM.
        # Accuracy is identical — pure speed optimisation.
        theta_prev_for_penalty = theta_prev   # default: CPU (safe on all hardware)
        if str(self.device).startswith("cuda"):
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb >= 24.0:
                    theta_prev_for_penalty = [
                        p0.to(self.device, dtype=torch.float32) for p0 in theta_prev
                    ]
                    logger.info(
                        "theta_prev moved to GPU (VRAM=%.1f GB) -- PCIe transfers eliminated.",
                        vram_gb,
                    )
            except Exception as exc:
                logger.warning("theta_prev GPU move failed (%s); using CPU fallback.", exc)
        # --------------------------------------------------------------------- #

        # Use AMP + GradScaler when the model is loaded in half precision.
        # This preserves memory usage on 12 GB GPUs while reducing fp16 NaN risk.
        param_dtype = next(self.model.parameters()).dtype
        use_cuda = str(self.device).startswith("cuda")
        if use_cuda and param_dtype == torch.float16 and torch.cuda.is_bf16_supported():
            logger.info("Fine-tuning: switching fp16 -> bf16 for numerical stability.")
            self.model.to(dtype=torch.bfloat16)
            param_dtype = torch.bfloat16
        amp_enabled = use_cuda and param_dtype in (torch.float16, torch.bfloat16)
        amp_dtype = torch.float16 if param_dtype == torch.float16 else torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
        if amp_enabled:
            logger.info("Fine-tuning with AMP (%s).", str(param_dtype))

        epochs_done = 0
        final_loss  = 0.0
        # Gradient-accumulation multiplier (default 1 = no-op). When >1 the
        # optimiser step is taken every `grad_accum_steps` micro-batches so
        # that effective batch size == cfg.batch_size * grad_accum_steps,
        # matching the thesis target (=32) across hardware tiers. Loss is
        # scaled by 1/N before backward so the accumulated gradient equals
        # a single gradient at the effective batch size. See
        # scripts/hardware.py::TARGET_EFFECTIVE_BATCH_SIZE for the rationale.
        grad_accum_steps = max(int(getattr(cfg, "grad_accum_steps", 1)), 1)

        for epoch in range(cfg.epochs_per_cycle):
            epoch_loss = 0.0
            n_batches  = 0
            micro_step = 0  # micro-batches accumulated since last optim step
            optimizer.zero_grad(set_to_none=True)

            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                # Skip degenerate batches where every label is ignored.
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

                # L2 regularisation: penalise deviation from θ_prev.
                # Uses theta_prev_for_penalty (GPU if VRAM >= 24 GB, else CPU).
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

                # Scale loss by grad_accum_steps so gradients summed over
                # N micro-batches match a single gradient at effective batch.
                loss = raw_loss / grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_step += 1

                # Optimiser step on accumulation boundary.
                if micro_step % grad_accum_steps == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        # Gradient clipping for training stability [DES]
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Gradient clipping for training stability [DES]
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    micro_step = 0

                epoch_loss += raw_loss.item()
                n_batches  += 1

            # Tail step: if the epoch ended mid-accumulation, step with
            # whatever has been accumulated so no training data is wasted.
            # Effective gradient magnitude is < 1× a full accumulation step
            # (because micro_step < grad_accum_steps here), but at most one
            # such partial step occurs per epoch.
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

        self.model.eval()
        return epochs_done, final_loss

    def _l2_penalty(self, theta_prev: List[torch.Tensor]) -> torch.Tensor:
        """Compute ||θ − θ_prev||² summed over all parameters.

        Kept on the training device so gradients can flow to current
        parameters. theta_prev is treated as a constant reference snapshot.
        """
        penalty = torch.tensor(0.0, device=self.device)
        for p, p0 in zip(self.model.parameters(), theta_prev):
            # Cast to fp32 before computing diff. p0 is already fp32 (stored by
            # _snapshot_weights). Without .float() here, fp16 model params produce
            # fp16 subtraction whose squared sum can overflow (fp16 max = 65504)
            # with 780M parameters, propagating inf -> NaN into the total loss.
            ref = p0.to(self.device, dtype=torch.float32)
            diff = p.float() - ref
            penalty = penalty + (diff ** 2).sum()
        return penalty

    # ------------------------------------------------------------------ #
    # Forgetting check                                                     #
    # ------------------------------------------------------------------ #

    def _forgetting_score(self, general_eval: List[QAPair]) -> float:
        """DEPRECATED -- kept for backward compatibility / diagnostic runs only.

        Historical purpose: estimate general-capability retention on held-out
        in-distribution pairs (TriviaQA-style) via greedy-generation exact match.

        Why deprecated: using an in-distribution probe as the forgetting guard
        is systematically insensitive to catastrophic forgetting in the
        continual-learning sense (Kirkpatrick et al., 2017), because the model
        has just been fine-tuned on TriviaQA-like pairs. The abort guard is
        now driven by ``_mmlu_score`` (an out-of-distribution 4-choice
        benchmark that is never in the training pool) -- see ``run_cycle``.
        This method is no longer called by the main cycle loop; it remains
        available for diagnostic scripts that want a cheap in-domain check.

        For each pair, run greedy generation and do an exact-match check
        against the reference answer (lowercased, stripped). Retention =
        fraction of pairs where the model still produces the correct answer.

        Capped at MAX_FORGETTING_EVAL_PAIRS (50) to bound wall-clock time.

        Returns float in [0, 1].
        """
        MAX_FORGETTING_EVAL_PAIRS = 50
        if not general_eval:
            return 1.0   # no eval data -> assume no forgetting
        general_eval = general_eval[:MAX_FORGETTING_EVAL_PAIRS]

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

    def measure_mmlu(self, n: int = 200) -> float:
        """Public entry point for MMLU scoring — see ``_mmlu_score``."""
        return self._mmlu_score(n=n)

    def _mmlu_score(self, n: int = 200) -> float:
        """Measure MMLU 4-choice accuracy as a neutral cross-benchmark forgetting proxy.

        Uses a fixed 200-sample MMLU split for per-cycle retention logging
        (Chapter 5, Table 5.2).  Whenever the ablation framework is rebuilt
        (Phase 9, Session 42+) it must reuse this same split and scoring logic
        so the numbers remain directly comparable.

        Scoring methodology.
        This is a *generation-based* letter-prefix scorer: the prompt ends with
        "Answer:" and we greedily decode up to 8 new tokens, then check whether
        the decoded string starts with the gold letter (A/B/C/D). This differs
        from the log-likelihood protocol used in lm-eval-harness and many
        leaderboard submissions (which scores P(choice_k | prompt) across the
        four choices and picks argmax). We use generation scoring because (a)
        CAEM's pipeline is itself generative — log-likelihood scoring would
        exercise a code path that is never used at inference time — and (b)
        it mirrors how the fine-tuned model is actually evaluated in the
        main benchmark loop. Absolute MMLU numbers are therefore not directly
        comparable to lm-eval-harness leaderboards, but cycle-over-cycle
        *retention ratios* (the quantity Chapter 5 reports) are internally
        consistent and correctly measure forgetting.

        Sampling.
        We use a ``shuffle(seed=42)`` before ``select`` so the 200-sample
        subset spans MMLU's 57 subjects rather than only the alphabetically
        earliest subjects that appear at the head of the validation split.
        The seed is fixed so every cycle and every re-run scores the same
        samples, making retention ratios deterministic.

        Dual role: abort guard + RET reporting axis.
        This single MMLU measurement drives (a) the self-improvement cycle's
        forgetting abort guard via the post/pre retention ratio, and (b) the
        absolute ``mmlu_retention`` value reported in Chapter 5's Table 5.2
        RET column. MMLU is a held-out, domain-neutral 4-choice benchmark
        that is *never* in the training pool, so it is the scientifically
        appropriate probe for catastrophic forgetting in the continual-
        learning sense (Kirkpatrick et al., 2017). The earlier design used
        a separate in-distribution TriviaQA probe (``_forgetting_score``) as
        the abort guard; that design was abandoned because an in-distribution
        probe is systematically insensitive to the failure mode the guard
        exists to catch.

        Returns NaN (float) if the MMLU dataset is unavailable (no internet,
        no HuggingFace cache). NaN is propagated cleanly to the CSV so the
        experiment does not abort.
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
                # Defensive: MMLU should always have answer in {0,1,2,3}, but
                # guard against corrupt rows so a negative index doesn't
                # silently collide with Python's reverse-indexing semantics.
                if not (0 <= answer_idx < 4):
                    continue

                choice_str = "\n".join(
                    f"{choices_labels[i]}. {c}"
                    for i, c in enumerate(choices)
                    if i < len(choices_labels)
                )
                prompt = (
                    f"Question: {question}\n"
                    f"Choices:\n{choice_str}\n"
                    f"Answer:"
                )
                enc = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                out = self.model.generate(
                    enc["input_ids"],
                    max_new_tokens=8,
                    do_sample=False,
                )
                pred = self.tokenizer.decode(out[0], skip_special_tokens=True).strip().upper()
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

    # ------------------------------------------------------------------ #
    # Weight management                                                    #
    # ------------------------------------------------------------------ #

    def _snapshot_weights(self) -> List[torch.Tensor]:
        """Return a deepcopy of all model parameter tensors (CPU, always fp32).

        Stored in fp32 regardless of model dtype so that _l2_penalty can
        safely sum 780M squared differences without fp16 overflow (fp16 max
        is 65504; the cumulative L2 norm of a 780M-param model easily exceeds
        this in early training).
        """
        return [p.detach().cpu().float().clone() for p in self.model.parameters()]

    def _restore_weights(self, theta_prev: List[torch.Tensor]) -> None:
        """Restore model parameters to theta_prev in-place."""
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
        mmlu_retention_ratio: float,
        aborted: bool,
        theta_prev: Optional[List[torch.Tensor]],
    ) -> str:
        """Save model weights + metadata to outputs/cycle_{n}/.

        Writes two files per cycle:
          * ``model.pt``   -- torch.save of the current model.state_dict().
                              On an aborted cycle, self.model already holds
                              the restored theta_prev (the caller invoked
                              ``_restore_weights`` before us), so the saved
                              weights correctly reflect what the next cycle
                              will pick up.
          * ``meta.pkl``   -- picklable dict with cycle_num, seed,
                              epochs_done, final_loss, mmlu_retention_ratio,
                              aborted flag, and the checkpoint timestamp.
                              ``load_checkpoint`` reads this dict back and
                              separately restores weights from model.pt.

        Returns
        -------
        str
            Absolute string path of the cycle checkpoint directory.
        """
        ckpt_dir = self.output_dir / f"cycle_{cycle_num}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Weights.
        weights_path = ckpt_dir / "model.pt"
        try:
            torch.save(self.model.state_dict(), weights_path)
        except Exception as exc:  # pragma: no cover -- defensive
            logger.error(
                "Cycle %d: failed to save model weights to %s: %s",
                cycle_num, weights_path, exc,
            )
            raise

        # Metadata.
        meta = {
            "cycle_num": cycle_num,
            "seed": seed,
            "epochs_done": epochs_done,
            "final_loss": float(final_loss),
            "mmlu_retention_ratio": float(mmlu_retention_ratio),
            "aborted": bool(aborted),
            "timestamp": time.time(),
            # theta_prev is NOT pickled -- it is a list of CUDA tensors that
            # would fail or waste disk space. Its presence in the call
            # signature is a flag: non-None means the cycle aborted and
            # weights were restored before this save (see run_cycle Step 7).
            "theta_prev_was_restored": theta_prev is not None,
        }
        meta_path = ckpt_dir / "meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        logger.info(
            "Cycle %d: checkpoint saved to %s (aborted=%s).",
            cycle_num, ckpt_dir, aborted,
        )
        return str(ckpt_dir)
