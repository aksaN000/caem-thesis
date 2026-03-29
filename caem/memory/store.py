"""
caem/memory/store.py
====================
EpisodicMemoryStore: FAISS-backed episodic memory for the CAEM pipeline.

Architecture
------------
FAISS backend: IndexIDMap(IndexFlatIP)
  - IndexFlatIP: exact inner-product search on L2-normalised vectors
    → equivalent to cosine similarity, no approximation error.
  - IndexIDMap: maps external integer IDs → internal FAISS positions,
    enabling targeted remove_ids() calls during pruning.
  - Why not IndexIVFPQ (as the thesis spec mentions)?
    At 20k × 384 float32, the index is ~30 MB — trivially within VRAM budget.
    IVF-PQ reduces this to ~4 MB but requires training on ≥ nlist vectors
    and does not support remove_ids() without an IDMap2 wrapper.
    For correctness at this scale, FlatIP + IDMap is the right trade-off.
    If the dataset grows beyond ~200k entries, switch to IndexIDMap2 + IVFFlat.

Metadata storage: Python dict {entry_id: EpisodicEntry}
  - The embedding is stored in FAISS; all other fields live in this dict.
  - entry_ids are monotonically increasing integers (never reused).

Thread safety: NOT thread-safe. Add an external lock if parallelising.

See: pipeline-technical.md §EpisodicMemoryStore
"""

from __future__ import annotations

import logging
import math
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from caem.config import CAEMConfig
from caem.memory.entry import EpisodicEntry

logger = logging.getLogger(__name__)


class EpisodicMemoryStore:
    """FAISS-backed store for verified (question, reasoning, answer) triples.

    Public interface
    ----------------
    add(entry) → int                        Store a new episode; return its ID.
    search(embedding, k) → list of (EpisodicEntry, float)  Top-k cosine search.
    is_novel(embedding) → bool              True if no similar episode exists.
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
        self._dim: int = self.config.embedding_dim   # 384

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

        flat = faiss.IndexFlatIP(self._dim)   # Exact inner-product (= cosine after norm)
        self._index = faiss.IndexIDMap(flat)  # Wrap to support remove_ids()
        logger.info("FAISS IndexIDMap(IndexFlatIP) initialised. dim=%d", self._dim)

    def _to_faiss_matrix(self, embedding: np.ndarray) -> np.ndarray:
        """Return embedding as a float32 2-D row matrix for FAISS calls."""
        emb = embedding.astype(np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._dim}, got {emb.shape[1]}. "
                "Check that all-mpnet-base-v2 is being used (384-dim)."
            )
        return np.ascontiguousarray(emb)

    def _compute_value_score(self, entry: EpisodicEntry) -> float:
        """Compute the Value score used for pruning decisions.

        Value = 0.7 · Importance + 0.3 · Recency

        Importance = 0.4·φ + 0.3·(r/r_max) + 0.2·success_rate + 0.1·u_stored

        Where:
          φ  = cycle recency proxy = (storage_cycle + 1) / (max_cycle + 1)
              Episodes from later cycles are generally more reliable.
          r  = retrieval_count (normalised by the maximum across all episodes)
          r_max prevents a single very-popular episode from dominating.

        Recency = exp(−λ · age_in_seconds)
          λ = CAEMConfig.recency_lambda (0.01 by default).
        """
        cfg = self.config

        # φ: proxy for how recently this episode was stored (by cycle).
        # max_cycle = num_cycles - 1 (0-indexed); +1 to avoid divide-by-zero.
        max_cycle = cfg.num_cycles - 1
        phi = (entry.storage_cycle + 1) / (max_cycle + 1 + 1)

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
        self._index.add_with_ids(vec, ids)
        self._metadata[entry_id] = entry

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
            Query embedding, shape (384,), L2-normalised float32.
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

        similarities, ids = self._index.search(vec, k_actual)
        similarities = similarities[0]   # Unwrap batch dimension
        ids = ids[0]

        results = []
        for sim, eid in zip(similarities, ids):
            if eid == -1:
                # FAISS returns -1 for unfilled slots (shouldn't occur with IDMap).
                continue
            entry = self._metadata.get(int(eid))
            if entry is None:
                logger.warning("FAISS returned ID %d not found in metadata — skipping.", eid)
                continue
            results.append((entry, float(sim)))

        return results

    def get(self, entry_id: int) -> Optional[EpisodicEntry]:
        """Return the episode with the given entry_id, or None if not found."""
        return self._metadata.get(entry_id)

    def all_entries(self) -> list:
        """Return all EpisodicEntry objects currently in the store.

        Used by SelfImprovementLoop to collect training data. Returns a
        snapshot list — mutations to the store after this call are not
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
                "Feedback update: entry %d | accepted=%s | u_stored: %.4f → %.4f",
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
            id_selector = faiss.IDSelectorBatch(
                np.array([entry_id], dtype=np.int64)
            )
            self._index.remove_ids(id_selector)

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
        If the updated model produces a higher u_stored for an episode, the
        stored value is raised. If it falls below the pruning threshold, the
        episode is removed (stale / harmful knowledge).

        Parameters
        ----------
        verify_fn : callable
            Signature: (entry: EpisodicEntry) → StoredConfidence
            Uses the CURRENT model weights (already updated by fine-tuning).
        threshold : float or None
            Remove episodes whose new u_stored falls below this.
            Defaults to CAEMConfig.retroverify_prune_threshold (0.50).

        Returns
        -------
        (n_updated, n_removed) : tuple of int
        """
        threshold = threshold or self.config.retroverify_prune_threshold
        n_updated = 0
        n_removed = 0

        # Iterate over a snapshot of keys (dict may shrink during iteration).
        entry_ids = list(self._metadata.keys())
        logger.info("Retroactive re-verification of %d episodes.", len(entry_ids))

        for eid in entry_ids:
            entry = self._metadata.get(eid)
            if entry is None:
                continue   # Already removed.

            try:
                new_scores = verify_fn(entry)
            except Exception as exc:
                logger.warning(
                    "retroverify: verify_fn raised %s for entry %d — skipping.",
                    exc, eid
                )
                continue

            new_u = new_scores.u_stored

            if new_u < threshold:
                # Episode quality has degraded — remove it.
                self.remove(eid)
                n_removed += 1
                logger.debug("Retroverify: removed episode %d (new u_stored=%.3f < %.3f).", eid, new_u, threshold)
            elif new_u > entry.u_stored:
                # Quality improved — update upward only (don't downgrade).
                old_u = entry.u_stored
                self.update_u_stored(eid, new_u)
                entry.nli_score = new_scores.p_entail
                entry.sc_score = new_scores.s_avg
                entry.se_score = 1.0 - new_scores.h_norm
                entry.retroverified = True
                n_updated += 1
                logger.debug(
                    "Retroverify: updated episode %d u_stored %.3f → %.3f.",
                    eid, old_u, new_u
                )
            else:
                # New score is lower (but above threshold) — keep old u_stored.
                # Mark as retroverified so we know it was checked this cycle.
                entry.retroverified = True

        logger.info(
            "Retroverify complete: %d updated, %d removed. Store size: %d",
            n_updated, n_removed, self.size
        )
        return n_updated, n_removed

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist the full store (FAISS index + metadata) to disk.

        Saves two files:
          {path}.faiss   — the FAISS index binary
          {path}.meta    — pickled metadata dict

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
            logger.info("Saved FAISS index → %s", faiss_path)
        else:
            logger.warning("save: index is None (empty store) — no .faiss file written.")

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
            "Saved metadata (%d episodes) → %s", len(self._metadata), meta_path
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
            logger.info(
                "Loaded FAISS index (%d vectors) from %s", store._index.ntotal, faiss_path
            )
        else:
            logger.warning("No .faiss file found at %s — index will be rebuilt on next add.", faiss_path)

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
