"""
caem/memory/deferred.py
=======================
DeferredBuffer -- bounded FIFO for episodes whose Stage-5 verifier
decision was DEFERRED (0.45 <= u_stored < 0.65).

Why this exists
---------------
The Stage-5 decision tree produces four outcomes:

    STORE     u_stored >= tau_store (0.65)          -> write to main memory
    DEFERRED  tau_defer <= u_stored < tau_store     -> HOLD for later
    ABSTAIN   u_stored < tau_defer AND              -> user-facing refusal
              p_ground_max < tau_abs (0.20)
    DISCARD   otherwise                             -> drop

The thesis (Section 4.7, closing paragraph) makes the explicit dynamic
claim that "a healthy trajectory shifts mass from DEFERRED into STORE
as the base model fine-tunes on the training pool (answers the model
was borderline about in cycle n become commits in cycle n+1)." That
claim requires a buffer: without state carry-over across cycles, a
DEFERRED entry is indistinguishable from DISCARD at runtime, and the
cycle-n -> cycle-(n+1) reconsideration has nothing to operate on.

This module supplies that buffer.

Lifecycle
---------
1. push(entry, vout)              called at Stage 7 when decision == DEFERRED.
                                   Entry contents captured: question, answer,
                                   embedding (for novelty), storage_cycle, full
                                   UnifiedVerifierOutput snapshot, and age = 0.
2. reconsider(verify_fn, store)   called once per cycle boundary by
                                   SelfImprovementLoop, AFTER retroverify.
                                   For each buffered entry:
                                     - re-invoke verify_fn under the new theta;
                                     - if new vout.decision == STORE and
                                       new u_stored >= tau_store: promote into
                                       main memory via store.add(entry);
                                     - elif age + 1 >= ttl_cycles: drop;
                                     - else: increment age; keep.
3. Counters (n_promoted, n_ttl_dropped, n_kept) reported back to run_cycle
   and surfaced in CycleResult.

Design commitments
------------------
Bounded size. The buffer has a hard cap (deferred_buffer_max_size). When
pushing to a full buffer, the oldest entry is evicted (FIFO). Bounded
capacity prevents runaway memory growth under adversarial workloads that
produce many DEFERRED entries but few promotions.

TTL. An entry that survives deferred_buffer_ttl_cycles reconsideration
passes without clearing tau_store is dropped. The TTL is a soundness
safeguard: if the model has not gained confidence in an answer after
multiple cycles of fine-tuning, holding it indefinitely wastes buffer
capacity and introduces cross-cycle coupling that complicates ablations.

Novelty interaction. A promoted entry is subject to the standard
novelty filter on store.add -- if an episode covering the same question
was added in the intervening cycle by another path, the near-duplicate
is rejected cleanly. This is the correct behaviour: the buffer is a
queue of candidates, not an alternative store.

Persistence. Picklable so the outer pipeline can save/load alongside
EpisodicMemoryStore.

See: thesis Section 4.9 (Deferred-Entry Reconsideration), which
introduces and formalises this mechanism (the companion cycle-boundary
pass to Section 4.8 Retroactive Re-Verification).
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from caem.config import CAEMConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Buffered entry
# -----------------------------------------------------------------------------

@dataclass
class DeferredEntry:
    """A single held candidate awaiting cycle-boundary reconsideration.

    Captures enough state to (a) re-invoke the verifier at cycle boundary,
    (b) construct an EpisodicEntry if promoted, and (c) enforce the TTL.

    Fields are intentionally a superset of the STORE path in pipeline._maybe_store
    so that promotion is a pure reconstruction rather than a second pipeline call.
    """

    # Immutable content (set once at push time)
    question: str
    answer: str
    embedding: np.ndarray           # (768,) float32, L2-normalised
    storage_cycle_pushed: int       # cycle at which the entry was first deferred

    # Snapshot of the verifier output at push time -- used only for the
    # initial u_stored at the moment of deferral. The reconsideration pass
    # recomputes everything from scratch under the new verifier.
    initial_u_stored: float
    initial_decision: str = "DEFERRED"

    # Reconsideration state
    age: int = 0                    # number of reconsideration passes survived


# -----------------------------------------------------------------------------
# Buffer
# -----------------------------------------------------------------------------

class DeferredBuffer:
    """Bounded FIFO of DEFERRED episodes awaiting cycle-boundary reconsideration.

    Public interface
    ----------------
    push(question, answer, embedding, storage_cycle, vout)  Add a new DEFERRED entry.
    reconsider(verify_fn, memory_store, promote_threshold)  One sweep, per cycle.
    size                                                     Current entry count.
    save(path) / load(path)                                  Pickle persistence.
    clear()                                                  Empty the buffer.

    Thread safety: NOT thread-safe. Caller owns synchronisation if parallelising.
    """

    def __init__(self, config: Optional[CAEMConfig] = None) -> None:
        self.config = config or CAEMConfig()
        self._entries: Deque[DeferredEntry] = deque(
            maxlen=self.config.deferred_buffer_max_size
        )

    # ------------------------------------------------------------------ #
    # Basic state                                                          #
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def all_entries(self) -> List[DeferredEntry]:
        """Snapshot of buffer contents (safe against concurrent mutation)."""
        return list(self._entries)

    # ------------------------------------------------------------------ #
    # Push                                                                 #
    # ------------------------------------------------------------------ #

    def push(
        self,
        question: str,
        answer: str,
        embedding: np.ndarray,
        storage_cycle: int,
        vout: Any,
    ) -> None:
        """Hold a DEFERRED episode for the next cycle-boundary reconsideration.

        Parameters
        ----------
        question, answer : str
            The (query, generated answer) pair.
        embedding : np.ndarray
            768-dim L2-normalised SBERT embedding of the query.
        storage_cycle : int
            Cycle index at which the episode was first deferred.
        vout : UnifiedVerifierOutput
            Full Stage-5 verifier output captured at push time. Only
            ``u_stored`` and ``decision`` are retained explicitly; the
            reconsideration pass recomputes the signals from scratch.

        If the buffer is full (``size == max_size``), the oldest entry is
        evicted (FIFO via the bounded ``deque``). Eviction is logged at
        WARNING level because a full buffer implies DEFERRED entries are
        accumulating faster than cycles can promote them -- a diagnostic
        worth surfacing.
        """
        # Validate embedding shape defensively; the pipeline promises
        # (768,) float32 at this point but the buffer does not re-enter
        # the novelty filter, so a bad shape would blow up later at
        # promotion-time inside store.add rather than here.
        if embedding.ndim != 1 or embedding.shape[0] != 768:
            raise ValueError(
                f"DeferredBuffer.push: embedding shape must be (768,), "
                f"got {embedding.shape}."
            )
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)

        if len(self._entries) == self._entries.maxlen:
            evicted = self._entries[0]
            logger.warning(
                "DeferredBuffer full (%d); evicting oldest entry "
                "(pushed at cycle %d, age=%d, q=%r).",
                self._entries.maxlen, evicted.storage_cycle_pushed,
                evicted.age, evicted.question[:60],
            )

        self._entries.append(DeferredEntry(
            question=question,
            answer=answer,
            embedding=embedding,
            storage_cycle_pushed=storage_cycle,
            initial_u_stored=float(getattr(vout, "u_stored", 0.0)),
            initial_decision=str(getattr(vout, "decision", "DEFERRED")),
            age=0,
        ))
        logger.debug(
            "Deferred push: q=%r, u_stored=%.4f, cycle=%d, buffer=%d/%d.",
            question[:60], float(getattr(vout, "u_stored", 0.0)),
            storage_cycle, len(self._entries), self._entries.maxlen,
        )

    # ------------------------------------------------------------------ #
    # Reconsider (cycle boundary, after retroverify)                       #
    # ------------------------------------------------------------------ #

    def reconsider(
        self,
        verify_fn: Callable[[Any], Any],
        memory_store,
        promote_threshold: Optional[float] = None,
        ttl_cycles: Optional[int] = None,
    ) -> Tuple[int, int, int]:
        """Re-score every buffered entry; promote, drop, or keep.

        Called once per self-improvement cycle boundary, AFTER retroverify,
        by :meth:`SelfImprovementLoop.run_cycle`. Skipped on aborted
        cycles (the caller guards the call, identically to retroverify).

        Each buffered entry is re-verified under the fine-tuned weights:

          * ``decision == STORE`` AND ``u_stored >= promote_threshold``
            -> the entry is promoted: a fresh EpisodicEntry is
            constructed from the new verifier output and inserted into
            ``memory_store`` via ``add``. The standard novelty filter on
            ``add`` applies (a near-duplicate already in memory silently
            drops the promotion, which is correct).

          * age + 1 >= ttl_cycles -> drop.

          * else -> keep with ``age`` incremented, awaiting the next
            cycle's reconsideration pass.

        Parameters
        ----------
        verify_fn : callable
            Signature: ``(entry_like) -> UnifiedVerifierOutput``. Identical
            contract to the retroverify closure returned by
            :meth:`CAEMPipeline.make_retroverify_fn`. The ``entry_like``
            object must expose ``.question`` and ``.answer`` attributes.
            ``DeferredEntry`` satisfies that contract natively.
        memory_store : EpisodicMemoryStore
            Target store for promotions.
        promote_threshold : float or None
            New u_stored must be >= this to promote. Defaults to
            ``config.store_threshold`` (0.65) -- i.e. the same bar that
            controls STORE at initial verification, NOT the lower
            retroverify prune threshold.
        ttl_cycles : int or None
            Max reconsiderations an entry may survive without promotion.
            Defaults to ``config.deferred_buffer_ttl_cycles``.

        Returns
        -------
        (n_promoted, n_ttl_dropped, n_kept) : tuple of int
            Non-overlapping counts. ``n_promoted + n_ttl_dropped + n_kept``
            equals the buffer size at call-time minus any verifier errors
            (which are skipped and logged, leaving the entry in place).
        """
        if promote_threshold is None:
            promote_threshold = self.config.store_threshold
        if ttl_cycles is None:
            ttl_cycles = self.config.deferred_buffer_ttl_cycles

        # Deferred import to avoid a cycle at module load:
        # entry.py imports nothing from store; store imports entry; this
        # module imports neither at class-definition time.
        from caem.memory.entry import EpisodicEntry

        if not self._entries:
            logger.info("Deferred reconsideration: buffer empty, nothing to do.")
            return 0, 0, 0

        n_at_start = len(self._entries)
        logger.info(
            "Deferred reconsideration: sweeping %d entries "
            "(promote_threshold=%.3f, ttl=%d cycles).",
            n_at_start, promote_threshold, ttl_cycles,
        )

        # Snapshot and clear; we will rebuild the buffer from kept entries.
        # This is simpler and safer than in-place mutation of a deque.
        snapshot = list(self._entries)
        self._entries.clear()

        n_promoted = 0
        n_ttl_dropped = 0
        n_kept = 0

        for de in snapshot:
            try:
                vout = verify_fn(de)
            except Exception as exc:
                logger.warning(
                    "Deferred reconsider: verify_fn raised %s for q=%r "
                    "-- keeping entry for next cycle.",
                    exc, de.question[:60],
                )
                # Skip: do not re-queue with an age bump; the verifier
                # failure is our fault, not the entry's. Re-queue as-is.
                self._entries.append(de)
                n_kept += 1
                continue

            if vout is None:
                # Verifier deliberately returned None (e.g. retrieval failure);
                # treat identically to an exception.
                self._entries.append(de)
                n_kept += 1
                continue

            new_u = float(getattr(vout, "u_stored", 0.0))
            new_decision = str(getattr(vout, "decision", ""))

            # Promote: clears STORE bar under the new verifier.
            if new_decision == "STORE" and new_u >= promote_threshold:
                promoted_entry = EpisodicEntry(
                    question=de.question,
                    reasoning_chain=de.answer,
                    answer=de.answer,
                    embedding=de.embedding,
                    storage_cycle=memory_store.config.num_cycles \
                        if False else de.storage_cycle_pushed,
                    # ^ keep original storage_cycle: it is the cycle the
                    # episode originated from, which is the correct audit
                    # trail for the training pool's cycle-of-origin filter.
                    u_stored=new_u,
                    u_token=float(getattr(vout, "u_token", 0.0)),
                    u_dropout=float(getattr(vout, "u_dropout", 0.0)),
                    u_internal=float(getattr(vout, "u_internal", 0.0)),
                    s_avg=float(getattr(vout, "s_avg", 0.0)),
                    h_norm=float(getattr(vout, "h_norm", 0.0)),
                    p_entail=float(getattr(vout, "p_entail", 0.0)),
                    p_ground_max=float(getattr(vout, "p_ground_max", 0.0)),
                    p_ground_mean=float(getattr(vout, "p_ground_mean", 0.0)),
                    p_ground_atomic=float(getattr(vout, "p_ground_atomic", 0.0)),
                    p_contra=float(getattr(vout, "p_contra", 0.0)),
                    decision="STORE",
                    early_exit_triggered=bool(
                        getattr(vout, "early_exit_triggered", False)
                    ),
                )
                # Novelty filter on add: if a near-duplicate was stored in
                # the intervening cycle by another path, the promotion is
                # silently rejected. This is correct -- the buffer is a
                # queue of candidates, not a parallel memory.
                if memory_store.is_novel(de.embedding):
                    eid = memory_store.add(promoted_entry)
                    n_promoted += 1
                    logger.debug(
                        "Deferred promoted: q=%r, u %.3f -> %.3f (pushed at "
                        "cycle %d, age %d, new entry_id=%d).",
                        de.question[:60], de.initial_u_stored, new_u,
                        de.storage_cycle_pushed, de.age, eid,
                    )
                else:
                    # Near-duplicate already present; promotion is a no-op.
                    # Counted as dropped (not re-queued) because the entry
                    # has achieved its purpose, even if the write was
                    # suppressed.
                    n_promoted += 1
                    logger.debug(
                        "Deferred promoted but near-duplicate already in "
                        "memory (q=%r); entry closed out.",
                        de.question[:60],
                    )
                continue

            # Not promoted: either age-out or requeue.
            new_age = de.age + 1
            if new_age >= ttl_cycles:
                n_ttl_dropped += 1
                logger.debug(
                    "Deferred TTL-dropped: q=%r, new u %.3f, decision=%s, "
                    "aged %d >= ttl %d.",
                    de.question[:60], new_u, new_decision, new_age, ttl_cycles,
                )
            else:
                de.age = new_age
                self._entries.append(de)
                n_kept += 1

        logger.info(
            "Deferred reconsideration done: %d promoted, %d TTL-dropped, "
            "%d kept (buffer size: %d -> %d).",
            n_promoted, n_ttl_dropped, n_kept, n_at_start, len(self._entries),
        )
        return n_promoted, n_ttl_dropped, n_kept

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist buffer contents to a pickle file at ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": list(self._entries),
            "max_size": self._entries.maxlen,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("DeferredBuffer saved (%d entries) to %s.", len(self._entries), path)

    def load(self, path: str | Path) -> None:
        """Restore buffer contents from a pickle file at ``path``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DeferredBuffer save not found at {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        loaded_entries: List[DeferredEntry] = payload.get("entries", [])
        self._entries = deque(loaded_entries, maxlen=self.config.deferred_buffer_max_size)
        logger.info("DeferredBuffer loaded (%d entries) from %s.", len(self._entries), path)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def age_distribution(self) -> Dict[int, int]:
        """Return a histogram ``{age: count}`` over current buffer entries.

        Useful for Chapter 5 diagnostics: a healthy buffer has mass
        concentrated at age 0--1 with few entries at age == ttl - 1 (which
        would drop out on the next pass).
        """
        hist: Dict[int, int] = {}
        for de in self._entries:
            hist[de.age] = hist.get(de.age, 0) + 1
        return hist
