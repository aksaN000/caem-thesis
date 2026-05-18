"""
caem/routing/router.py
======================
AdaptiveRouter -- Stage 3 dispatch logic for the CAEM pipeline.

Takes a pre-routing confidence estimate and an episodic memory search result,
and returns a RoutingDecision specifying which tier handles the query.

Three-tier dispatch
-------------------
  Tier 1 -- Direct retrieval
    The retrieved episode is highly similar AND historically reliable.
    Return the stored reasoning chain and answer directly. No generation.
    Latency target: < 400 ms.

  Tier 2 -- Guided generation
    Moderate similarity: the memory holds relevant but not directly applicable
    knowledge. Use the retrieved reasoning chain as a soft prompt to guide
    generation of a new answer.
    Latency target: < 2.5 s.

  Tier 3 -- Full RAG (Retrieval-Augmented Generation)
    No reliable memory match, or model confidence too low to trust any match.
    Generate from scratch with Wikipedia passages as context.
    Latency target: < 6 s.

Two-mechanism design (CRITICAL -- do not merge into one formula)
--------------------------------------------------------------
Mechanism 1 -- OR-condition (hard veto, evaluated FIRST):
    if u_pre < safety_u_pre_min (0.60):
        -> Tier 3, safety_override=True
    This fires before any memory lookup result is considered.

Mechanism 2 -- Routing score formula (only if OR-condition did not fire):
    routing_score = 0.70 · similarity + 0.30 · û_stored
    if routing_score >= 0.90  -> Tier 1
    elif similarity > 0.75    -> Tier 2
    else                      -> Tier 3

WHY they are separate: combining them into one formula would allow a high
û_stored to arithmetically compensate for a dangerously low u_pre. A high-
quality memory match to a query the model doesn't understand would still
route to Tier 1. Memory quality and model readiness are mandatory conditions,
not tradeable quantities. See: writing-suggestions.md C4-10b.

Empty memory
------------
If the episodic store is empty (no retrieved episode), similarity = 0.0 and
û_stored = 0.0. The routing score will be 0.0, which falls below both thresholds
-- the router naturally dispatches to Tier 3 without any special-casing.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

from caem.config import CAEMConfig
from caem.memory.entry import EpisodicEntry, PreRoutingConfidence, RoutingDecision

logger = logging.getLogger(__name__)


class AdaptiveRouter:
    """Dispatch queries to Tier 1, 2, or 3 based on confidence and memory.

    Parameters
    ----------
    config : CAEMConfig
        All thresholds come from here -- no magic numbers in this class.

    Usage
    -----
    >>> router = AdaptiveRouter(config)
    >>> decision = router.route(
    ...     pre_confidence=pc,          # PreRoutingConfidence from Stage 3
    ...     search_results=results,     # From EpisodicMemoryStore.search(k=1)
    ... )
    >>> decision.tier    # 1, 2, or 3
    >>> decision.safety_override   # True if u_pre OR-condition fired
    """

    def __init__(
        self,
        config: Optional[CAEMConfig] = None,
        ruc: Optional["LRRetrievalUtilityClassifier"] = None,
    ) -> None:
        """Initialise the router.

        Parameters
        ----------
        config : CAEMConfig or None
            Threshold container. Defaults to a fresh CAEMConfig.
        ruc : LRRetrievalUtilityClassifier or None
            Optional frozen Retrieval Utility Classifier. When provided, the
            router consults it at the two Tier-3 entry points (safety veto +
            score-formula fall-through) and re-routes a fraction of those
            queries to Tier 2 based on the RUC's verdict. Tier 1 and the
            Tier-2-via-similarity path are unaffected. When None (default),
            the router behaves exactly as before this commit.
        """
        self.config = config or CAEMConfig()
        self.ruc = ruc

    def route(
        self,
        pre_confidence: PreRoutingConfidence,
        search_results: Sequence[Union[Tuple[EpisodicEntry, float], Tuple[EpisodicEntry, int, float]]],
        source_benchmark: Optional[str] = None,
        question: Optional[str] = None,
        *,
        top1_passage_text: Optional[str] = None,
        top1_passage_sim: Optional[float] = None,
        top1_passage_entity_overlap: Optional[float] = None,
        p_ik: Optional[float] = None,
    ) -> RoutingDecision:
        """Compute a routing decision for one query.

        Parameters
        ----------
        pre_confidence : PreRoutingConfidence
            Output of PreRoutingConfidenceEstimator.estimate(query).
        search_results : list of tuples
            Output of EpisodicMemoryStore.search() as (entry, similarity),
            or search_with_ids() as (entry, entry_id, similarity).
            May be empty if the store has no episodes yet.
        source_benchmark : str or None, default None
            Benchmark tag (v2 Fix 12). When provided and present in
            ``CAEMConfig.safety_u_pre_min_per_benchmark``, the per-benchmark
            safety threshold is used for the OR-condition check; otherwise
            the pooled ``safety_u_pre_min`` is used.
        question : str or None
            Raw query text. Required for the RUC to fire; without it the
            router falls back to pre-RUC default Tier 3 at both fire
            points.
        top1_passage_text, top1_passage_sim, top1_passage_entity_overlap : optional
            Pre-computed top-1 FAISS retrieval features against CAEM's
            21M-passage index. The pipeline supplies these; the router
            does not touch FAISS itself. If any is ``None``, the RUC
            silently substitutes the training-time stub (``""`` for text,
            ``0.0`` for the floats) — which degrades the RUC's signal but
            preserves correctness.
        p_ik : float or None
            Pre-computed Kadavath-style direct-correctness probe value
            on the question's Qwen hidden state. ``None`` falls back to
            ``0.5`` (no-information prior), again preserving correctness
            at the cost of one feature.

        Returns
        -------
        RoutingDecision
            tier, similarity, u_stored_retrieved, u_pre, routing_score,
            safety_override, retrieved_entry_id, ruc_output.
        """
        cfg = self.config
        u_pre = pre_confidence.u_pre
        # v2 Fix 12: per-benchmark safety floor falls back to pooled global.
        if hasattr(cfg, "get_safety_u_pre_min_for"):
            safety_thr = cfg.get_safety_u_pre_min_for(source_benchmark)
        else:
            safety_thr = float(cfg.safety_u_pre_min)

        # -- Unpack memory search result ----------------------------------- #
        if search_results:
            first = search_results[0]
            if len(first) == 3:
                best_entry, retrieved_entry_id, similarity = first
            else:
                best_entry, similarity = first
                retrieved_entry_id = None
            u_stored_retrieved = best_entry.u_stored
        else:
            # Empty memory -- no episode retrieved.
            best_entry = None
            similarity = 0.0
            u_stored_retrieved = 0.0
            retrieved_entry_id = None

        # -- Mechanism 1 -- OR-condition (MUST be evaluated first) ---------- #
        if u_pre < safety_thr:
            routing_score = cfg.routing_lambda * similarity + (1 - cfg.routing_lambda) * u_stored_retrieved
            tier, ruc_out = self._consult_ruc(
                question=question,
                top1_passage_text=top1_passage_text,
                top1_passage_sim=top1_passage_sim,
                top1_passage_entity_overlap=top1_passage_entity_overlap,
                p_ik=p_ik,
                source_benchmark=source_benchmark,
                entry_point="safety_veto",
                default_tier=3,
            )
            decision = RoutingDecision(
                tier=tier,
                similarity=similarity,
                u_stored_retrieved=u_stored_retrieved,
                u_pre=u_pre,
                routing_score=routing_score,
                safety_override=True,
                retrieved_entry_id=retrieved_entry_id,
                ruc_output=ruc_out,
            )
            logger.debug(
                "OR-condition fired: u_pre=%.4f < %.2f (bench=%s) -> Tier %d "
                "(safety override; RUC=%s).",
                u_pre, safety_thr, source_benchmark, tier,
                "DIRECT" if (ruc_out and ruc_out.get("decision") == "DIRECT") else
                ("RAG" if ruc_out else "n/a"),
            )
            return decision

        # -- Mechanism 2 -- Routing score formula --------------------------- #
        routing_score = (
            cfg.routing_lambda * similarity
            + (1.0 - cfg.routing_lambda) * u_stored_retrieved
        )

        ruc_out: Optional[dict] = None
        if routing_score >= cfg.tier1_combined_threshold:
            tier = 1
        elif similarity > cfg.tier2_similarity_threshold:
            tier = 2
        else:
            # Fall-through case: previously forced Tier 3. Consult the RUC if
            # one is wired in; the RUC re-routes a fraction of these queries
            # to Tier 2 (parametric CoT) when retrieval is predicted to hurt.
            tier, ruc_out = self._consult_ruc(
                question=question,
                top1_passage_text=top1_passage_text,
                top1_passage_sim=top1_passage_sim,
                top1_passage_entity_overlap=top1_passage_entity_overlap,
                p_ik=p_ik,
                source_benchmark=source_benchmark,
                entry_point="fall_through",
                default_tier=3,
            )

        decision = RoutingDecision(
            tier=tier,
            similarity=similarity,
            u_stored_retrieved=u_stored_retrieved,
            u_pre=u_pre,
            routing_score=routing_score,
            safety_override=False,
            retrieved_entry_id=retrieved_entry_id,
            ruc_output=ruc_out,
        )

        logger.debug(
            "Routed to Tier %d | u_pre=%.4f | sim=%.4f | û_stored=%.4f | "
            "score=%.4f | thresholds=[%.2f, %.2f]",
            tier, u_pre, similarity, u_stored_retrieved, routing_score,
            cfg.tier1_combined_threshold, cfg.tier2_similarity_threshold,
        )
        return decision

    # ------------------------------------------------------------------ #
    # RUC consultation helper                                             #
    # ------------------------------------------------------------------ #

    def _consult_ruc(
        self,
        *,
        question: Optional[str],
        top1_passage_text: Optional[str],
        top1_passage_sim: Optional[float],
        top1_passage_entity_overlap: Optional[float],
        p_ik: Optional[float],
        source_benchmark: Optional[str],
        entry_point: str,
        default_tier: int,
    ) -> Tuple[int, Optional[dict]]:
        """Ask the RUC whether to route to RAG (T3) or DIRECT (T2).

        Returns ``(tier, ruc_output_dict)``. When no RUC is wired in, or the
        question text is unavailable, falls back to ``default_tier`` and
        returns ``(default_tier, None)`` — behaviour identical to the pre-RUC
        router.

        Retrieval features (``top1_passage_*``) and ``p_ik`` are supplied by
        the pipeline. The router does not touch FAISS or the Qwen model. When
        any of these are ``None``, the LR RUC silently substitutes the
        training-time stub — the classifier still produces a probability, but
        on a degraded feature set.
        """
        if self.ruc is None or question is None:
            return default_tier, None
        try:
            kwargs = dict(
                question=question,
                source_benchmark=source_benchmark,
            )
            if top1_passage_text is not None:
                kwargs["top1_passage_text"] = top1_passage_text
            if top1_passage_sim is not None:
                kwargs["top1_passage_sim"] = top1_passage_sim
            if top1_passage_entity_overlap is not None:
                kwargs["top1_passage_entity_overlap"] = top1_passage_entity_overlap
            if p_ik is not None:
                kwargs["p_ik"] = p_ik
            res = self.ruc.predict(**kwargs)
            ruc_out = {
                "decision": res.decision,
                "p_rag": res.p_rag,
                "threshold": res.threshold,
                "features": res.features,
                "flavour": res.flavour,
                "entry_point": entry_point,
            }
            tier = 3 if res.decision == "RAG" else 2
            return tier, ruc_out
        except Exception as exc:
            logger.warning(
                "RUC.predict raised at entry_point=%s; falling back to tier=%d. "
                "Error: %s", entry_point, default_tier, exc,
            )
            return default_tier, None

    # ------------------------------------------------------------------ #
    # Convenience: explain a routing decision in plain English            #
    # ------------------------------------------------------------------ #

    def explain(self, decision: RoutingDecision) -> str:
        """Return a human-readable explanation of a routing decision.

        Useful for logging, debugging, and thesis experiment write-ups.
        """
        cfg = self.config
        lines = []

        if decision.safety_override:
            lines.append(
                f"OR-condition fired: u_pre={decision.u_pre:.4f} < "
                f"safety threshold {cfg.safety_u_pre_min:.2f}."
            )
            lines.append(
                "Routed to Tier 3 regardless of memory similarity. "
                "Model readiness check failed -- Tier 3 cost preferred over "
                "confident-but-wrong Tier 1/2 answer."
            )
        else:
            lines.append(f"u_pre={decision.u_pre:.4f} ≥ {cfg.safety_u_pre_min:.2f} (OR-condition: safe).")
            lines.append(
                f"Routing score = {cfg.routing_lambda:.2f}·sim + "
                f"{1-cfg.routing_lambda:.2f}·û_stored = "
                f"{cfg.routing_lambda:.2f}·{decision.similarity:.4f} + "
                f"{1-cfg.routing_lambda:.2f}·{decision.u_stored_retrieved:.4f} "
                f"= {decision.routing_score:.4f}."
            )
            if decision.tier == 1:
                lines.append(
                    f"Score {decision.routing_score:.4f} ≥ Tier 1 threshold "
                    f"{cfg.tier1_combined_threshold:.2f} -> Tier 1 (direct retrieval)."
                )
            elif decision.tier == 2:
                lines.append(
                    f"Score {decision.routing_score:.4f} < {cfg.tier1_combined_threshold:.2f}, "
                    f"but similarity {decision.similarity:.4f} > Tier 2 threshold "
                    f"{cfg.tier2_similarity_threshold:.2f} -> Tier 2 (guided generation)."
                )
            else:
                lines.append(
                    f"Score {decision.routing_score:.4f} < {cfg.tier1_combined_threshold:.2f} "
                    f"and similarity {decision.similarity:.4f} ≤ {cfg.tier2_similarity_threshold:.2f} "
                    "-> Tier 3 (full RAG)."
                )

        return " ".join(lines)

    # ------------------------------------------------------------------ #
    # Batch routing (for experiment harness)                              #
    # ------------------------------------------------------------------ #

    def route_batch(
        self,
        items: Sequence[Tuple[PreRoutingConfidence, Sequence[Union[Tuple[EpisodicEntry, float], Tuple[EpisodicEntry, int, float]]]]],
    ) -> List[RoutingDecision]:
        """Route a batch of (confidence, search_results) pairs.

        Parameters
        ----------
        items : list of (PreRoutingConfidence, search_results)

        Returns
        -------
        list of RoutingDecision, one per item.
        """
        return [self.route(pc, sr) for pc, sr in items]
