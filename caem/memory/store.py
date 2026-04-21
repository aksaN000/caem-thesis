"""
caem/memory/store.py
====================
EpisodicMemoryStore: FAISS-backed episodic memory for the CAEM pipeline.

Architecture
------------
FAISS backend: configurable via CAEMConfig.memory_index_type
    - "flat_ip": IndexIDMap(IndexFlatIP), exact cosine search.
    - "ivf_pq": bootstrap on FlatIP, then auto-promote to
        IndexIDMap2(IndexIVFPQ) once enough vectors exist for IVF training.
        This keeps early-cycle behavior stable and enables 1M-scale retrieval.

Metadata storage: Python dict {entry_id: EpisodicEntry}
  - The embedding is stored in FAISS; all other fields live in this dict.
  - entry_ids are monotonically increasing integers (never reused).

Thread safety: NOT thread-safe. Add an external lock if parallelising.

See: pipeline-technical.md §EpisodicMemoryStore
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from caem.config import CAEMConfig
from caem.memory.entry import EpisodicEntry

# NOTE: ``caem.training.loop_filter`` is imported lazily inside
# ``retroverify`` to avoid a circular import. ``caem.training.__init__``
# eagerly imports ``SelfImprovementLoop``, which imports
# ``caem.memory.store`` -- a top-level import here would close the cycle
# during ``caem`` package initialisation.

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Consolidation helpers (Branch C Goal 4 item 3)
# -----------------------------------------------------------------------------

# NOTE: ``eval.metrics.normalise`` is imported lazily inside
# ``consolidate()`` (via ``_get_answer_normaliser``) to avoid the
# ``caem.memory.store -> eval.metrics -> eval.__init__ -> eval.baselines
# -> caem.pipeline -> caem.memory.store`` circular import chain. Reusing
# the EM normaliser is a deliberate invariant: consolidation-equality
# and EM-equality must never drift apart across refactors.


def _get_answer_normaliser():
    """Lazy accessor for ``eval.metrics.normalise`` (see NOTE above)."""
    from eval.metrics import normalise
    return normalise


def _json_dumps(obj: dict) -> str:
    """Compact-but-sortable JSON serialisation for the consolidation audit
    log. ``sort_keys=True`` keeps diffs across runs stable.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


# Audit log skip-reason enum (Ch1 audit + Cycle-1 review grep on this field).
# Kept as module-level strings rather than an Enum so JSON serialisation is
# trivial and the codes remain greppable across the log corpus.
SKIP_ANSWER_MISMATCH: str = "answer_mismatch"
SKIP_U_SPREAD_EXCEEDED: str = "u_spread_exceeded"
SKIP_CROSS_POOL: str = "cross_pool_split"
OUTCOME_MERGED: str = "merged"


# Training vs transfer pool assignment for the pre-split consolidation pass.
# Pre-split avoids ever creating a mixed-pool cluster, so the SIL training-
# pool gate never has to veto a consolidated entry for OOD leak. Untagged
# entries get their own pool so legacy / unit-test paths still consolidate.
_POOL_TRAINING: str = "training"
_POOL_TRANSFER: str = "transfer"
_POOL_UNTAGGED: str = "untagged"


class EpisodicMemoryStore:
    """FAISS-backed store for verified (question, reasoning, answer) triples.

    Public interface
    ----------------
    add(entry) -> int                        Store a new episode; return its ID.
    search(embedding, k) -> list of (EpisodicEntry, float)  Top-k cosine search.
    is_novel(embedding) -> bool              True if no similar episode exists.
    update_u_stored(entry_id, new_val)      Retrieval-feedback u_stored update.
    update_retrieval_stats(id, accepted)    Increment count; update success_rate.
    retroverify(verify_fn, threshold)       Per-cycle re-check of all episodes.
    prune()                                 Remove bottom 20% by Value score.
    remove(entry_id)                        Hard-delete one episode.
    save(path) / load(path)                 Persist and reload the full store.
    """

    def __init__(self, config: CAEMConfig | None = None) -> None:
        self.config = config or CAEMConfig()
        self._index = None          # FAISS index (lazy init on first add)
        self._metadata: Dict[int, EpisodicEntry] = {}
        self._next_id: int = 0      # Monotonically increasing external ID counter
        self._dim: int = self.config.embedding_dim   # 768
        self._index_type: str = str(
            getattr(self.config, "memory_index_type", "flat_ip")
        ).lower()
        self._ivf_enabled: bool = False

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _init_index(self) -> None:
        """Initialise the FAISS index on first add (avoids import at module load)."""
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss is required for EpisodicMemoryStore. "
                "Install with: pip install faiss-gpu  (or faiss-cpu)"
            ) from e

        # Always bootstrap with FlatIP for early cycles. IVF-PQ requires a
        # non-trivial training set and is promoted later when enough vectors exist.
        flat = faiss.IndexFlatIP(self._dim)   # Exact inner-product (= cosine after norm)
        self._index = faiss.IndexIDMap(flat)  # Wrap to support remove_ids()
        self._ivf_enabled = False

        if self._index_type == "ivf_pq":
            logger.info(
                "FAISS memory backend bootstrap: IndexIDMap(IndexFlatIP), dim=%d "
                "(auto-promote to IVF-PQ after training threshold).",
                self._dim,
            )
        else:
            logger.info("FAISS IndexIDMap(IndexFlatIP) initialised. dim=%d", self._dim)

    def _maybe_promote_to_ivf_pq(self) -> None:
        """Promote the active index from FlatIP to IVF-PQ when enough vectors exist."""
        if self._index_type != "ivf_pq" or self._ivf_enabled:
            return
        if self._index is None:
            return

        try:
            import faiss
        except ImportError:
            return

        nlist = int(getattr(self.config, "faiss_nlist", 4096))
        nprobe = int(getattr(self.config, "faiss_nprobe", 32))
        pq_m = int(getattr(self.config, "faiss_pq_m", 64))
        pq_nbits = int(getattr(self.config, "faiss_pq_nbits", 8))
        train_min = int(getattr(self.config, "faiss_train_min_points", 20_000))

        if self.size < max(train_min, nlist):
            return

        ids = np.array(sorted(self._metadata.keys()), dtype=np.int64)
        if ids.size == 0:
            return

        vectors = np.vstack([self._metadata[int(eid)].embedding for eid in ids]).astype(np.float32)

        # FAISS requires training examples >= nlist for k-means quantiser training.
        if vectors.shape[0] < nlist:
            return

        sample_cap = int(getattr(self.config, "faiss_train_sample_size", 200_000))
        if sample_cap > 0 and vectors.shape[0] > sample_cap:
            rng = np.random.default_rng(seed=0)
            sample_idx = rng.choice(vectors.shape[0], size=sample_cap, replace=False)
            train_vectors = vectors[sample_idx]
        else:
            train_vectors = vectors

        try:
            quantizer = faiss.IndexFlatIP(self._dim)
            try:
                core = faiss.IndexIVFPQ(
                    quantizer,
                    self._dim,
                    nlist,
                    pq_m,
                    pq_nbits,
                    faiss.METRIC_INNER_PRODUCT,
                )
            except TypeError:
                core = faiss.IndexIVFPQ(
                    quantizer,
                    self._dim,
                    nlist,
                    pq_m,
                    pq_nbits,
                )
                if hasattr(core, "metric_type"):
                    core.metric_type = faiss.METRIC_INNER_PRODUCT

            core.nprobe = nprobe
            new_index = faiss.IndexIDMap2(core)

            core.train(np.ascontiguousarray(train_vectors))  # type: ignore[call-arg]
            new_index.add_with_ids(np.ascontiguousarray(vectors), ids)  # type: ignore[call-arg]

            self._index = new_index
            self._ivf_enabled = True
            logger.info(
                "Promoted memory index to IndexIDMap2(IndexIVFPQ): "
                "nlist=%d, nprobe=%d, m=%d, nbits=%d, vectors=%d",
                nlist,
                nprobe,
                pq_m,
                pq_nbits,
                vectors.shape[0],
            )
        except Exception as exc:
            logger.warning(
                "IVF-PQ promotion skipped (continuing with FlatIP): %s",
                exc,
            )

    def _to_faiss_matrix(self, embedding: np.ndarray) -> np.ndarray:
        """Return embedding as a float32 2-D row matrix for FAISS calls."""
        emb = embedding.astype(np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._dim}, got {emb.shape[1]}. "
                "Check that all-mpnet-base-v2 is being used (768-dim)."
            )
        return np.ascontiguousarray(emb)

    def _compute_value_score(self, entry: EpisodicEntry) -> float:
        """Compute the Value score used for pruning decisions.

        Value = 0.7 · Importance + 0.3 · Recency

        Importance = 0.4·φ + 0.3·(r/r_max) + 0.2·success_rate + 0.1·u_stored

        Where:
          φ  = cycle recency proxy = (storage_cycle + 1) / (num_cycles + 1)
              Episodes from later cycles are generally more reliable.
              Denominator is (num_cycles + 1) so that the final cycle (num_cycles)
              gives φ = 1.0 exactly without exceeding it.
          r  = retrieval_count (normalised by the maximum across all episodes)
          r_max prevents a single very-popular episode from dominating.

        Recency = exp(−λ · age_in_seconds)
          λ = CAEMConfig.recency_lambda (0.01 by default).
        """
        cfg = self.config

        # φ: proxy for how recently this episode was stored (by cycle).
        # Denominator = (num_cycles + 1) so cycle num_cycles → φ = 1.0 exactly.
        # Example: num_cycles=10, cycle 10 → (10+1)/11 = 1.0.
        phi = (entry.storage_cycle + 1) / (cfg.num_cycles + 1)

        # Normalised retrieval count.
        max_r = max(
            (e.retrieval_count for e in self._metadata.values()),
            default=1
        )
        r_norm = entry.retrieval_count / max(max_r, 1)

        importance = (
            cfg.importance_phi_weight       * phi
            + cfg.importance_retrieval_weight * r_norm
            + cfg.importance_success_weight   * entry.success_rate
            + cfg.importance_u_stored_weight  * entry.u_stored
        )

        recency = math.exp(-cfg.recency_lambda * entry.age())

        return (
            cfg.value_importance_weight * importance
            + cfg.value_recency_weight  * recency
        )

    # ------------------------------------------------------------------ #
    # Core operations                                                      #
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        """Number of episodes currently stored."""
        return len(self._metadata)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def is_novel(self, embedding: np.ndarray) -> bool:
        """Return True if no existing episode is too similar to this embedding.

        'Too similar' is defined as cosine similarity > CAEMConfig.novelty_threshold
        (default 0.95). Episodes above this threshold are considered known
        and should NOT be re-stored (they would add noise to the training set).
        """
        if self.is_empty:
            return True
        results = self.search(embedding, k=1)
        if not results:
            return True
        _, best_sim = results[0]
        return best_sim <= self.config.novelty_threshold

    def add(self, entry: EpisodicEntry) -> int:
        """Store a new episode in the FAISS index and metadata dict.

        Parameters
        ----------
        entry : EpisodicEntry
            The episode to store. The embedding must already be L2-normalised
            (QueryEncoder.encode_for_storage guarantees this).

        Returns
        -------
        int
            The external entry_id assigned to this episode.

        Raises
        ------
        RuntimeError
            If the store is full (size ≥ max_memory_size). Caller should
            check is_novel() and trigger prune() before calling add().
        """
        if self.size >= self.config.max_memory_size:
            raise RuntimeError(
                f"EpisodicMemoryStore is full ({self.size} / {self.config.max_memory_size}). "
                "Call prune() before adding new episodes."
            )

        if self._index is None:
            self._init_index()

        entry_id = self._next_id
        self._next_id += 1

        vec = self._to_faiss_matrix(entry.embedding)
        ids = np.array([entry_id], dtype=np.int64)
        self._index.add_with_ids(vec, ids)  # type: ignore[union-attr,call-arg]
        self._metadata[entry_id] = entry

        # Promote to IVF-PQ once the configured training threshold is reached.
        self._maybe_promote_to_ivf_pq()

        logger.debug(
            "Stored episode %d | cycle=%d | u_stored=%.3f | q='%s...'",
            entry_id, entry.storage_cycle, entry.u_stored,
            entry.question[:60]
        )
        return entry_id

    def search(
        self,
        embedding: np.ndarray,
        k: int = 1,
    ) -> List[Tuple[EpisodicEntry, float]]:
        """Retrieve the top-k most similar episodes by cosine similarity.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding, shape (768,), L2-normalised float32.
        k : int
            Number of results to return (1 for routing; higher for inspection).

        Returns
        -------
        list of (EpisodicEntry, float)
            Sorted by descending similarity. May be shorter than k if fewer
            episodes exist. Returns [] if the store is empty.
        """
        if self.is_empty:
            return []

        if self._index is None:
            return []   # Shouldn't happen if size > 0, but defensive.

        k_actual = min(k, self.size)
        vec = self._to_faiss_matrix(embedding)

        similarities, ids = self._index.search(vec, k_actual)  # type: ignore[call-arg]
        similarities = similarities[0]   # Unwrap batch dimension
        ids = ids[0]

        results = []
        for sim, eid in zip(similarities, ids):
            if eid == -1:
                # FAISS returns -1 for unfilled slots (shouldn't occur with IDMap).
                continue
            entry = self._metadata.get(int(eid))
            if entry is None:
                logger.warning("FAISS returned ID %d not found in metadata -- skipping.", eid)
                continue
            results.append((entry, float(sim)))

        return results

    def search_with_ids(
        self,
        embedding: np.ndarray,
        k: int = 1,
    ) -> List[Tuple[EpisodicEntry, int, float]]:
        """Like search(), but also returns the entry_id for each result.

        Returns
        -------
        list of (EpisodicEntry, entry_id, similarity)
            entry_id is the integer FAISS external ID -- the same value used
            by update_u_stored(), update_retrieval_stats(), and remove().

        Use this instead of search() whenever downstream code needs to call
        back into the store (e.g. update_retrieval_stats) -- avoids scanning
        _metadata to recover IDs.
        """
        if self.is_empty:
            return []
        if self._index is None:
            return []

        k_actual = min(k, self.size)
        vec = self._to_faiss_matrix(embedding)

        similarities, ids = self._index.search(vec, k_actual)  # type: ignore[call-arg]
        similarities = similarities[0]
        ids = ids[0]

        results = []
        for sim, eid in zip(similarities, ids):
            if eid == -1:
                continue
            eid_int = int(eid)
            entry = self._metadata.get(eid_int)
            if entry is None:
                logger.warning("FAISS returned ID %d not found in metadata -- skipping.", eid_int)
                continue
            results.append((entry, eid_int, float(sim)))

        return results

    def get(self, entry_id: int) -> Optional[EpisodicEntry]:
        """Return the episode with the given entry_id, or None if not found."""
        return self._metadata.get(entry_id)

    def all_entries(self) -> list:
        """Return all EpisodicEntry objects currently in the store.

        Used by SelfImprovementLoop to collect training data. Returns a
        snapshot list -- mutations to the store after this call are not
        reflected in the returned list.
        """
        return list(self._metadata.values())

    # ------------------------------------------------------------------ #
    # Mutable field updates                                                #
    # ------------------------------------------------------------------ #

    def update_u_stored(self, entry_id: int, new_val: float) -> None:
        """Update the stored confidence of an episode.

        Called by:
          - Retrieval feedback loop (after ≥ 5 retrievals).
          - Retroactive re-verification (once per improvement cycle).

        Parameters
        ----------
        entry_id : int
        new_val : float
            New u_stored value ∈ [0, 1].
        """
        entry = self._metadata.get(entry_id)
        if entry is None:
            logger.warning("update_u_stored: entry_id %d not found.", entry_id)
            return
        entry.u_stored = float(np.clip(new_val, 0.0, 1.0))

    def update_retrieval_stats(self, entry_id: int, was_accepted: bool) -> None:
        """Update retrieval_count and success_rate for a retrieved episode.

        Also applies the retrieval feedback loop u_stored update if the
        episode has accumulated ≥ min_retrievals retrievals.

        Parameters
        ----------
        entry_id : int
        was_accepted : bool
            True if the Tier 1/2 response was accepted; False if overridden.
        """
        entry = self._metadata.get(entry_id)
        if entry is None:
            logger.warning("update_retrieval_stats: entry_id %d not found.", entry_id)
            return

        # Increment count and update running success rate.
        n = entry.retrieval_count
        old_rate = entry.success_rate
        entry.retrieval_count = n + 1
        # Incremental mean update: new_rate = (old_rate * n + outcome) / (n + 1)
        entry.success_rate = (old_rate * n + float(was_accepted)) / (n + 1)
        # Branch C Goal 4 item 5: increment the Tier-1-serve hit counter.
        # Reset to 0 on retroverify; the "popular-needs-recheck" queue
        # reads it via ``force_retroverify_queue``. update_retrieval_stats
        # is the single source-of-truth serve-point that the pipeline
        # calls on every Tier-1 hit (see pipeline._update_tier1_stats).
        entry.hit_counter = entry.hit_counter + 1

        # Retrieval feedback u_stored update (only after min_retrievals).
        cfg = self.config
        if entry.retrieval_count >= cfg.feedback_loop_min_retrievals:
            eta = cfg.feedback_loop_eta
            current = entry.u_stored
            if was_accepted:
                new_u = current + eta * (1.0 - current)   # Nudge toward 1.0
            else:
                new_u = current - eta * current            # Nudge toward 0.0
            self.update_u_stored(entry_id, new_u)
            logger.debug(
                "Feedback update: entry %d | accepted=%s | u_stored: %.4f -> %.4f",
                entry_id, was_accepted, current, new_u
            )

    # ------------------------------------------------------------------ #
    # Pruning                                                              #
    # ------------------------------------------------------------------ #

    def _should_prune(self) -> bool:
        """True if size has exceeded the pruning trigger threshold."""
        return self.size >= self.config.max_memory_size * self.config.pruning_trigger

    def prune(self) -> int:
        """Remove the bottom pruning_amount (20%) of episodes by Value score.

        Returns the number of episodes removed.

        Design rationale:
          - Triggered when the store reaches 95% capacity.
          - Value score = weighted combination of retrieval frequency,
            success rate, cycle recency (φ), and u_stored.
          - Least valuable episodes are dropped; the FAISS index is rebuilt
            to reflect the removal. Rebuilding is O(N) but pruning is rare.
        """
        if self.size == 0:
            return 0

        n_to_remove = max(1, int(self.size * self.config.pruning_amount))
        logger.info(
            "Pruning %d / %d episodes (bottom %.0f%% by Value score).",
            n_to_remove, self.size, self.config.pruning_amount * 100
        )

        # Score all episodes.
        scored = [
            (eid, self._compute_value_score(entry))
            for eid, entry in self._metadata.items()
        ]
        # Sort ascending: lowest value first.
        scored.sort(key=lambda x: x[1])
        ids_to_remove = [eid for eid, _ in scored[:n_to_remove]]

        for eid in ids_to_remove:
            self.remove(eid)

        logger.info("Pruning complete. Store size: %d", self.size)
        return n_to_remove

    def remove(self, entry_id: int) -> bool:
        """Hard-delete an episode from both the FAISS index and metadata dict.

        Parameters
        ----------
        entry_id : int

        Returns
        -------
        bool
            True if the episode was found and removed; False otherwise.
        """
        if entry_id not in self._metadata:
            logger.warning("remove: entry_id %d not found.", entry_id)
            return False

        if self._index is not None:
            import faiss
            id_selector = faiss.IDSelectorBatch(  # type: ignore[call-arg]
                np.array([entry_id], dtype=np.int64)
            )
            self._index.remove_ids(id_selector)  # type: ignore[call-arg]

        del self._metadata[entry_id]
        logger.debug("Removed episode %d. Store size: %d", entry_id, self.size)
        return True

    # ------------------------------------------------------------------ #
    # Retroactive re-verification (called once per improvement cycle)     #
    # ------------------------------------------------------------------ #

    def retroverify(
        self,
        verify_fn,
        threshold: Optional[float] = None,
    ) -> Tuple[int, int]:
        """Re-verify all stored episodes with the updated model.

        Called at the end of each self-improvement cycle (Stage 6 / Phase 6).

        Branch C Goal 4 extensions (2026-04-22):

        * **Item 2 -- loop prune**: before re-scoring, if the stored
          ``reasoning_chain`` trips the loop filter
          (``caem.training.loop_filter.is_repetitive_loop``) and
          ``cfg.retroverify_prune_loops`` is True, the episode is removed
          outright. No verifier forward pass is spent on loop contamination.
          Counts of loop-pruned vs. threshold-pruned entries are logged
          separately; the returned tuple sums them into ``n_removed`` to
          preserve the pre-Goal-4 API shape.

        * **Item 4 -- downgrade**: previously the stored ``u_stored`` was
          only ever raised; lower re-scores above the prune threshold were
          silently ignored, leaving stale confident entries in memory. When
          ``cfg.retroverify_allow_downgrade`` is True (default), the stored
          value is replaced with the new score in either direction, and the
          full nine-signal block is overwritten so downstream calibration
          tables see the fresh verifier view of the episode.

        Parameters
        ----------
        verify_fn : callable
            Signature: ``(entry: EpisodicEntry) -> UnifiedVerifierOutput``.
            Uses the CURRENT model weights (already updated by fine-tuning).
        threshold : float or None
            Remove episodes whose new u_stored falls below this.
            Defaults to ``cfg.retroverify_prune_threshold`` (0.50).

        Returns
        -------
        (n_updated, n_removed) : tuple of int
            ``n_removed`` aggregates loop-prunes and threshold-prunes so the
            pre-Goal-4 API shape is preserved. See the "loop-prunes vs.
            threshold-prunes" info log for the breakdown.
        """
        # Lazy import: caem.training imports caem.memory at module scope,
        # so a top-level import here would close the cycle.
        from caem.training.loop_filter import is_repetitive_loop

        threshold = threshold or self.config.retroverify_prune_threshold
        prune_loops = bool(
            getattr(self.config, "retroverify_prune_loops", True)
        )
        allow_downgrade = bool(
            getattr(self.config, "retroverify_allow_downgrade", True)
        )

        n_updated = 0
        n_loop_pruned = 0
        n_threshold_pruned = 0

        entry_ids = list(self._metadata.keys())
        logger.info("Retroactive re-verification of %d episodes.", len(entry_ids))

        for eid in entry_ids:
            entry = self._metadata.get(eid)
            if entry is None:
                continue

            # Goal 4 item 2: loop-prune before re-verification. Loops are
            # ungrammatical memorisation artefacts; no amount of re-scoring
            # rehabilitates them. Skip the verifier entirely.
            if prune_loops and is_repetitive_loop(
                entry.reasoning_chain or "", self.config
            ):
                self.remove(eid)
                n_loop_pruned += 1
                logger.debug(
                    "Retroverify: loop-pruned episode %d (stored chain failed "
                    "distinct-4/compression filter).", eid,
                )
                continue

            try:
                new_scores = verify_fn(entry)
            except Exception as exc:
                logger.warning(
                    "retroverify: verify_fn raised %s for entry %d -- skipping.",
                    exc, eid,
                )
                continue

            new_u = new_scores.u_stored

            if new_u < threshold:
                self.remove(eid)
                n_threshold_pruned += 1
                logger.debug(
                    "Retroverify: removed episode %d (new u_stored=%.3f < %.3f).",
                    eid, new_u, threshold,
                )
                continue

            # Goal 4 item 4: update in either direction (raise OR lower).
            # Gated behind allow_downgrade so the pre-Goal-4 raise-only
            # behaviour is recoverable via config.
            direction_ok = (new_u > entry.u_stored) or (
                allow_downgrade and new_u != entry.u_stored
            )
            if direction_ok:
                old_u = entry.u_stored
                self.update_u_stored(eid, new_u)
                entry.u_token = new_scores.u_token
                entry.u_dropout = new_scores.u_dropout
                entry.u_internal = new_scores.u_internal
                entry.s_avg = new_scores.s_avg
                entry.h_norm = new_scores.h_norm
                entry.p_entail = new_scores.p_entail
                entry.p_ground_max = new_scores.p_ground_max
                entry.p_ground_mean = new_scores.p_ground_mean
                entry.p_ground_atomic = new_scores.p_ground_atomic
                entry.p_contra = new_scores.p_contra
                entry.q_a_relevance = new_scores.q_a_relevance
                entry.decision = new_scores.decision
                entry.early_exit_triggered = new_scores.early_exit_triggered
                entry.retroverified = True
                # Goal 4 item 5: reset the Tier-1-serve counter whenever
                # the entry is re-scored. Next cycle's "popular-needs-
                # recheck" queue then reflects post-retroverify pressure.
                entry.hit_counter = 0
                n_updated += 1
                direction = "raised" if new_u > old_u else "lowered"
                logger.debug(
                    "Retroverify: %s episode %d u_stored %.3f -> %.3f.",
                    direction, eid, old_u, new_u,
                )
            else:
                # Exact tie, or allow_downgrade=False and new_u < old_u.
                entry.retroverified = True
                entry.hit_counter = 0

        n_removed = n_loop_pruned + n_threshold_pruned
        logger.info(
            "Retroverify complete: %d updated, %d removed "
            "(loop-pruned=%d, threshold-pruned=%d). Store size: %d",
            n_updated, n_removed, n_loop_pruned, n_threshold_pruned, self.size,
        )
        # Transient diagnostic attribute for callers that want the breakdown
        # (e.g. diagnostics/run_experiment.py per-cycle CSV). The public
        # tuple return stays 2-element so no existing unpacking breaks.
        self._last_retroverify_breakdown = {
            "loop_pruned": n_loop_pruned,
            "threshold_pruned": n_threshold_pruned,
        }
        return n_updated, n_removed

    def force_retroverify_queue(
        self,
        hit_threshold: Optional[int] = None,
    ) -> List[int]:
        """Return entry_ids of episodes that have crossed the hit-counter
        threshold since their last retroverify.

        Branch C Goal 4 item 5. The caller (retroverify scheduler or a
        between-cycles fast-path script) can iterate this list to force
        re-verification of popular-but-possibly-wrong entries before the
        regular cycle-boundary sweep catches them. Popular-wrong failure
        mode: an entry Tier-1-served to many queries compounds retrieval
        harm; catching it at N=10 serves is cheap insurance.

        Parameters
        ----------
        hit_threshold : int or None
            Minimum ``hit_counter`` value. Defaults to
            ``cfg.hit_counter_force_retroverify`` (10). A value <= 0
            returns the empty list (force-queue disabled).

        Returns
        -------
        list of int
            Entry IDs with ``hit_counter >= hit_threshold``, sorted by
            descending ``hit_counter`` (highest-pressure first). Sorted
            deterministically by entry_id on ties so callers get a stable
            order across runs.
        """
        if hit_threshold is None:
            hit_threshold = int(
                getattr(self.config, "hit_counter_force_retroverify", 10)
            )
        if hit_threshold <= 0:
            return []
        eids = [
            eid for eid, entry in self._metadata.items()
            if entry.hit_counter >= hit_threshold
        ]
        # Highest hit_counter first; ties broken by lower entry_id for
        # determinism (matches the consolidation tiebreaker convention).
        eids.sort(key=lambda e: (-self._metadata[e].hit_counter, e))
        return eids

    # ------------------------------------------------------------------ #
    # Memory consolidation (Branch C Goal 4 item 3, 2026-04-22)           #
    # ------------------------------------------------------------------ #

    def consolidate(
        self,
        similarity_threshold: Optional[float] = None,
        *,
        cycle_num: Optional[int] = None,
        audit_log_path: Optional["Path | str"] = None,
        qa_embed_fn: Optional[Any] = None,  # noqa: ARG002 -- reserved hook
    ) -> Tuple[int, int]:
        """Cluster near-paraphrase episodes; keep max-u representative per cluster.

        Algorithm
        ---------
        1. **Candidate clustering**: for each stored episode, query the
           FAISS index for its top-k (``cfg.consolidation_search_k``,
           default 20) neighbours. Pairs with cosine similarity above
           ``similarity_threshold`` (default
           ``cfg.consolidation_similarity_threshold`` = 0.92) are unioned
           via union-find into candidate clusters.
        2. **Safety guards** (applied per candidate cluster before merging):

           a. *Answer consistency* -- the representative's normalised
              answer string must equal every member's normalised answer
              (lowercased, stripped, ASCII-punctuation removed). This
              catches the "Who directed X" vs "Who starred in X" failure
              mode where query-side SBERT embedding similarity is high but
              the correct answers diverge. Clusters failing this check are
              logged to the audit JSONL and NOT merged.

              A cleaner long-term fix is to cluster on a joint
              ``(query, answer)`` embedding. We leave this as a hook via
              the ``qa_embed_fn`` protocol (not yet wired) and defend
              with the surface-form check until the Cycle-1 audit is in
              hand.

           b. *u_stored spread guardrail* -- if ``max(u) - min(u) >
              cfg.consolidation_max_u_spread`` (default 0.15) within the
              cluster, skip the merge. High internal variance is a tell
              that the candidate cluster is actually heterogeneous.

        3. **Merge formulas** (applied when both guards pass):

           * ``u_stored`` -- keep the representative's value (the cluster
             maximum; NEVER averaged). Ties broken by lower entry_id for
             checkpoint-snapshot reproducibility.
           * ``retrieval_count`` -- sum across all cluster members.
           * ``success_rate`` -- retrieval-count-weighted average,
             recovered as ``total_successes / total_retrievals``.
           * ``source_benchmark`` -- retain the representative's tag.
           * ``merged_source_benchmarks`` -- tuple union of every
             cluster member's (source_benchmark ∪
             merged_source_benchmarks), minus the representative's own
             tag, deduplicated, deterministic order. The SIL
             training-pool gate then treats the effective benchmark set
             as ``{source_benchmark} ∪ merged_source_benchmarks`` and
             excludes the entry if ANY member is in TRANSFER_BENCHMARKS
             (prevents OOD leak through consolidation).
           * ``retroverified`` -- OR across members (any retroverified ->
             rep retroverified).

           Hit-counter merge semantics and a retroverify timestamp are
           planned for Goal 4 item 5; the current schema has no such
           fields so the merge is a no-op on those axes.

        4. **Audit log** (when ``audit_log_path`` is supplied): every
           cluster decision (merge or skip-for-answer-mismatch or
           skip-for-u-spread) is appended as one JSONL record. Read this
           file after Cycle 1 to catch false-positive merges before they
           compound across cycles.

        Expected gain -- honesty note
        -----------------------------
        The initial design claimed "20-30% memory size reduction over 10
        cycles". Realistic reduction depends on benchmark-mix duplication
        rate; under a 30% STORE rate with ~25% semantic duplication the
        actual reduction is closer to 7-8%. The thesis should report the
        empirically-measured number from Branch-C data, not the planning
        ceiling. The mechanism's load-bearing claims are (a) FAISS
        retrieval latency improvement, (b) Tier-1 top-k noise reduction,
        (c) retrieval-history concentration on the representative.

        Returns
        -------
        (n_clusters_merged, n_removed) : tuple of int
            n_clusters_merged counts clusters that PASSED both safety
            guards and were actually merged. Clusters skipped for
            answer-mismatch or u-spread do not contribute to either
            counter but are recorded in the audit log.
        """
        if not getattr(self.config, "enable_consolidation", True):
            logger.info("Consolidation disabled via cfg.enable_consolidation.")
            return 0, 0
        if self.size <= 1:
            return 0, 0

        # Skip Cycle 0: cold-start memory is intentionally diverse (seed
        # episodes span all benchmarks) and should not be collapsed on the
        # first pass. Consolidation begins at the Cycle-1 -> Cycle-2
        # boundary. None is treated as "caller didn't specify, run anyway"
        # so existing single-call scripts and CPU unit tests are unaffected.
        if cycle_num is not None and cycle_num < 1:
            logger.info(
                "Consolidation skipped for cycle %d (Cycle 0 cold-start is "
                "protected; first pass runs at the Cycle-1 -> Cycle-2 boundary).",
                cycle_num,
            )
            return 0, 0

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.config.consolidation_similarity_threshold
        )
        search_k = max(2, int(self.config.consolidation_search_k))
        max_u_spread = float(
            getattr(self.config, "consolidation_max_u_spread", 0.15)
        )
        normalise_answer = _get_answer_normaliser()
        logger.info(
            "Consolidation pass: %d episodes, threshold=%.3f, search_k=%d, "
            "max_u_spread=%.3f.",
            self.size, threshold, search_k, max_u_spread,
        )

        audit_records: List[dict] = []

        # ---------- Pre-split by training / transfer pool -----------------
        # Partition entries into TRAINING / TRANSFER / UNTAGGED pools and
        # consolidate WITHIN each pool only. This avoids ever creating a
        # mixed-pool cluster -- the SIL training-pool gate therefore never
        # has to veto a consolidated entry for OOD leak, and legitimate
        # training data is not silently sacrificed when a transfer-pool
        # member slips into the cluster. See branch_C.md Goal 4 item 3.
        from caem.config import TRAINING_BENCHMARKS, TRANSFER_BENCHMARKS

        def _pool_for(entry: EpisodicEntry) -> str:
            # An entry's effective benchmark set is
            # {source_benchmark} ∪ merged_source_benchmarks. If ANY member
            # is transfer -> transfer pool; else all-training -> training;
            # else (empty / unknown tags) -> untagged.
            bms = set()
            if entry.source_benchmark is not None:
                bms.add(entry.source_benchmark)
            bms.update(entry.merged_source_benchmarks)
            if not bms:
                return _POOL_UNTAGGED
            if bms & set(TRANSFER_BENCHMARKS):
                return _POOL_TRANSFER
            if bms.issubset(set(TRAINING_BENCHMARKS)):
                return _POOL_TRAINING
            # Tagged with something outside both pools (e.g., custom benchmark)
            # -- treat as untagged so the clustering still happens within a
            # single deterministic bucket.
            return _POOL_UNTAGGED

        pool_members: Dict[str, List[int]] = {
            _POOL_TRAINING: [],
            _POOL_TRANSFER: [],
            _POOL_UNTAGGED: [],
        }
        for eid, entry in self._metadata.items():
            pool_members[_pool_for(entry)].append(eid)

        # ---------- Union-find restricted to within-pool pairs ------------
        parent: Dict[int, int] = {eid: eid for eid in self._metadata}

        def find(e: int) -> int:
            root = e
            while parent[root] != root:
                root = parent[root]
            while parent[e] != root:
                parent[e], e = root, parent[e]
            return root

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # FAISS retrieval returns neighbours from the whole index; the
        # cross-pool check below vetoes any neighbour that lives in a
        # different pool. Flagged in the audit log with SKIP_CROSS_POOL so
        # Ch1 review can confirm the gate is active when expected.
        cross_pool_skips: List[Tuple[int, int, float]] = []
        entry_pool: Dict[int, str] = {
            eid: pool for pool, members in pool_members.items() for eid in members
        }
        for eid, entry in list(self._metadata.items()):
            neighbours = self.search_with_ids(
                entry.embedding, k=min(search_k, self.size),
            )
            for _neighbour_entry, nid, sim in neighbours:
                if nid == eid or sim < threshold or nid not in parent:
                    continue
                if entry_pool.get(nid) != entry_pool.get(eid):
                    cross_pool_skips.append((eid, nid, float(sim)))
                    continue
                union(eid, nid)

        # ---------- Group by root ----------------------------------------
        clusters: Dict[int, List[int]] = {}
        for eid in parent:
            clusters.setdefault(find(eid), []).append(eid)

        n_clusters_merged = 0
        n_clusters_skipped_answer = 0
        n_clusters_skipped_spread = 0
        n_clusters_skipped_cross_pool = len(cross_pool_skips)
        n_removed = 0

        # Record each cross-pool rejection individually so Ch1 can audit
        # that the pre-split gate actually ran.
        for a, b, sim in cross_pool_skips:
            audit_records.append({
                "members": sorted([a, b]),
                "outcome": SKIP_CROSS_POOL,
                "reason": SKIP_CROSS_POOL,
                "similarity": sim,
                "pools": {a: entry_pool.get(a), b: entry_pool.get(b)},
            })

        for _root, members in clusters.items():
            if len(members) < 2:
                continue

            u_vals = [self._metadata[m].u_stored for m in members]
            u_spread = max(u_vals) - min(u_vals)

            # Deterministic representative: highest u_stored; ties broken
            # by oldest storage_cycle (the member that has been observed
            # across more cycles is the most-verified), then by lowest
            # entry_id for final determinism. The max() key uses negative
            # cycle and entry_id so lower values win on ties.
            rep = max(
                members,
                key=lambda eid: (
                    self._metadata[eid].u_stored,
                    -self._metadata[eid].storage_cycle,
                    -eid,
                ),
            )
            rep_entry = self._metadata[rep]
            rep_norm_answer = normalise_answer(rep_entry.answer)

            member_answers = {
                m: normalise_answer(self._metadata[m].answer)
                for m in members
            }
            answers_consistent = all(
                a == rep_norm_answer for a in member_answers.values()
            )

            audit_record = {
                "members": sorted(members),
                "u_stored": {m: self._metadata[m].u_stored for m in members},
                "answers": {
                    m: self._metadata[m].answer for m in members
                },
                "rep": rep,
                "u_spread": u_spread,
                "pool": entry_pool.get(rep),
            }

            if not answers_consistent:
                audit_record["outcome"] = SKIP_ANSWER_MISMATCH
                audit_record["reason"] = SKIP_ANSWER_MISMATCH
                audit_records.append(audit_record)
                n_clusters_skipped_answer += 1
                logger.debug(
                    "Consolidation skip: cluster size %d has inconsistent "
                    "answers (rep=%r); not merging.",
                    len(members), rep_entry.answer,
                )
                continue

            if u_spread > max_u_spread:
                audit_record["outcome"] = SKIP_U_SPREAD_EXCEEDED
                audit_record["reason"] = SKIP_U_SPREAD_EXCEEDED
                audit_records.append(audit_record)
                n_clusters_skipped_spread += 1
                logger.debug(
                    "Consolidation skip: cluster size %d has u_stored "
                    "spread %.3f > %.3f; flagged for audit.",
                    len(members), u_spread, max_u_spread,
                )
                continue

            # Guards passed -- perform the merge.
            # ---- retrieval metadata: sum + weighted-average --------------
            total_retrievals = rep_entry.retrieval_count
            total_successes = rep_entry.retrieval_count * rep_entry.success_rate
            any_retroverified = bool(rep_entry.retroverified)

            for mid in members:
                if mid == rep:
                    continue
                mem_entry = self._metadata[mid]
                total_retrievals += mem_entry.retrieval_count
                total_successes += (
                    mem_entry.retrieval_count * mem_entry.success_rate
                )
                any_retroverified = any_retroverified or bool(mem_entry.retroverified)

            rep_entry.retrieval_count = int(total_retrievals)
            if total_retrievals > 0:
                rep_entry.success_rate = float(
                    total_successes / total_retrievals
                )

            # Branch C Goal 4 item 5: hit_counter sums across cluster members.
            # Summing is the right semantic -- the consolidated representative
            # inherits ALL Tier-1 serve pressure that accumulated across the
            # cluster, so the "popular-needs-recheck" queue picks it up as
            # aggressively as it would have picked up the most-served member.
            total_hits = sum(
                self._metadata[mid].hit_counter for mid in members
            )
            rep_entry.hit_counter = int(total_hits)

            # TODO(retroverify-timestamp): current schema has only a bool
            # ``retroverified``; OR-semantics here is defensible as long as
            # retroverify trigger stays cycle-boundary (not timestamp-gated).
            # When a ``last_retroverify_cycle`` field is added, take the
            # max across cluster members instead.
            rep_entry.retroverified = any_retroverified

            # ---- benchmark set union (within-pool; pre-split guarantees
            # no cross-pool contamination here) ---------------------------
            bm_set: List[str] = []
            seen_bm: set = set()
            rep_bm = rep_entry.source_benchmark
            rep_prev_merged = rep_entry.merged_source_benchmarks

            for mid in members:
                if mid == rep:
                    continue
                mem_entry = self._metadata[mid]
                if mem_entry.source_benchmark is not None \
                        and mem_entry.source_benchmark != rep_bm \
                        and mem_entry.source_benchmark not in seen_bm:
                    bm_set.append(mem_entry.source_benchmark)
                    seen_bm.add(mem_entry.source_benchmark)
                for extra in mem_entry.merged_source_benchmarks:
                    if extra != rep_bm and extra not in seen_bm:
                        bm_set.append(extra)
                        seen_bm.add(extra)

            merged_bm = list(rep_prev_merged) + [
                b for b in bm_set if b not in rep_prev_merged
            ]
            rep_entry.merged_source_benchmarks = tuple(merged_bm)

            # ---- Remove the non-representative members -------------------
            # NOTE: FAISS IVF-PQ indexes may not support remove_ids natively;
            # the store.remove() method relies on whatever the active index
            # implementation supports. IVF-PQ rebuild cost (~seconds on a
            # 15K-entry index) is borne at cycle-boundary consolidation
            # time; budget this in the cycle-boundary scheduler. Flat-IP
            # (the test default) supports remove natively.
            for mid in members:
                if mid == rep:
                    continue
                self.remove(mid)
                n_removed += 1

            n_clusters_merged += 1
            audit_record["outcome"] = OUTCOME_MERGED
            audit_record["reason"] = OUTCOME_MERGED
            audit_record["merged_source_benchmarks"] = \
                list(rep_entry.merged_source_benchmarks)
            audit_record["merged_retrieval_count"] = rep_entry.retrieval_count
            audit_record["merged_success_rate"] = rep_entry.success_rate
            audit_records.append(audit_record)

            logger.debug(
                "Consolidation merge: cluster size %d -> 1 representative "
                "(eid=%d u_stored=%.3f retrievals=%d success=%.3f bm+={%s}).",
                len(members), rep, rep_entry.u_stored,
                rep_entry.retrieval_count, rep_entry.success_rate,
                ",".join(rep_entry.merged_source_benchmarks),
            )

        # ---------- Audit log write --------------------------------------
        if audit_log_path is not None and audit_records:
            p = Path(audit_log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                for rec in audit_records:
                    f.write(_json_dumps(rec) + "\n")
            logger.info(
                "Consolidation: appended %d audit records to %s.",
                len(audit_records), p,
            )

        logger.info(
            "Consolidation complete: %d merged, skipped (answer=%d, "
            "u_spread=%d, cross_pool=%d), %d episodes removed. Store size: %d",
            n_clusters_merged,
            n_clusters_skipped_answer,
            n_clusters_skipped_spread,
            n_clusters_skipped_cross_pool,
            n_removed,
            self.size,
        )

        self._last_consolidation_breakdown = {
            "clusters_merged": n_clusters_merged,
            "clusters_skipped_answer_mismatch": n_clusters_skipped_answer,
            "clusters_skipped_u_spread": n_clusters_skipped_spread,
            "clusters_skipped_cross_pool": n_clusters_skipped_cross_pool,
            "removed": n_removed,
            "final_size": self.size,
        }
        return n_clusters_merged, n_removed

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist the full store (FAISS index + metadata) to disk.

        Saves two files:
          {path}.faiss   -- the FAISS index binary
          {path}.meta    -- pickled metadata dict

        Parameters
        ----------
        path : str or Path
            Base path (without extension). Extensions are added automatically.
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss is required for save/load.") from e

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss_path = str(path) + ".faiss"
        meta_path = str(path) + ".meta"

        if self._index is not None:
            faiss.write_index(self._index, faiss_path)
            logger.info("Saved FAISS index -> %s", faiss_path)
        else:
            logger.warning("save: index is None (empty store) -- no .faiss file written.")

        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "metadata": self._metadata,
                    "next_id": self._next_id,
                    "dim": self._dim,
                    "config": self.config,
                },
                f,
            )
        logger.info(
            "Saved metadata (%d episodes) -> %s", len(self._metadata), meta_path
        )

    @classmethod
    def load(cls, path: str | Path) -> "EpisodicMemoryStore":
        """Load a previously saved store from disk.

        Parameters
        ----------
        path : str or Path
            Base path (same as used in save(), without extension).

        Returns
        -------
        EpisodicMemoryStore
            A fully reconstructed store ready for use.
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss is required for save/load.") from e

        path = Path(path)
        faiss_path = str(path) + ".faiss"
        meta_path = str(path) + ".meta"

        with open(meta_path, "rb") as f:
            saved = pickle.load(f)

        store = cls(config=saved["config"])
        store._metadata = saved["metadata"]
        store._next_id = saved["next_id"]
        store._dim = saved["dim"]

        if Path(faiss_path).exists():
            store._index = faiss.read_index(faiss_path)
            try:
                ivf_index = faiss.extract_index_ivf(store._index)
                ivf_index.nprobe = int(getattr(store.config, "faiss_nprobe", ivf_index.nprobe))
                store._ivf_enabled = True
            except Exception:
                store._ivf_enabled = False
            logger.info(
                "Loaded FAISS index (%d vectors) from %s", store._index.ntotal, faiss_path
            )
        else:
            logger.warning("No .faiss file found at %s -- index will be rebuilt on next add.", faiss_path)

        logger.info("Loaded store: %d episodes.", store.size)
        return store

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """Return a dict of summary statistics for logging and experiment tracking."""
        if self.size == 0:
            return {"size": 0}

        u_stored_vals = [e.u_stored for e in self._metadata.values()]
        retrieval_counts = [e.retrieval_count for e in self._metadata.values()]
        cycle_counts: dict = {}
        for e in self._metadata.values():
            cycle_counts[e.storage_cycle] = cycle_counts.get(e.storage_cycle, 0) + 1

        return {
            "size": self.size,
            "capacity": self.config.max_memory_size,
            "fill_pct": round(100 * self.size / self.config.max_memory_size, 1),
            "mean_u_stored": round(float(np.mean(u_stored_vals)), 4),
            "min_u_stored": round(float(np.min(u_stored_vals)), 4),
            "max_u_stored": round(float(np.max(u_stored_vals)), 4),
            "total_retrievals": sum(retrieval_counts),
            "episodes_by_cycle": cycle_counts,
        }

    def __repr__(self) -> str:
        return (
            f"EpisodicMemoryStore("
            f"size={self.size}/{self.config.max_memory_size}, "
            f"dim={self._dim})"
        )
