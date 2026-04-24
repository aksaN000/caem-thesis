"""caem.ablation.runner -- single-variant execution.

Given an ``AblationVariant`` plus pre-built model/encoder/NLI objects
and an eval sample dict, this runner:

  1. Applies the variant's config mutation to a fresh CAEMConfig.
  2. Constructs a CAEMPipeline with that config + honours the
     variant's pipeline-level skip flags by monkey-patching the
     relevant stage methods (transparent to the rest of the harness).
  3. Runs EvalHarness on the provided eval samples.
  4. Extracts per-sample (em, u_stored) lists and feeds them to
     ``ces_axes_from_cycle``.
  5. Packages the result as an AblationResult.

The heavy lifting is delegated:
  - Model/encoder/NLI loading lives in ``scripts/run_ablation.py``
    (which reuses ``scripts.run_experiment.build_pipeline``).
  - Fine-tuning / retroverify orchestration lives in the caller;
    the runner itself is strictly evaluation-only.

A deliberate choice: the runner does NOT perform the SelfImprovement
loop. For variants that need fine-tuned weights the caller must pass a
pipeline whose model has already been fine-tuned with the variant's
config (or the shared full-CAEM weights, depending on the sweep
design). This keeps the runner lean and makes the "compare against full
CAEM at cycle N" scenario trivial: call the runner with the cycle-N
memory store + cycle-N weights for each variant.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

from caem.ablation.scoring import (
    CESAxes,
    aggregate_axes,
    ces_axes_from_cycle,
)
from caem.ablation.variants import AblationVariant

if TYPE_CHECKING:
    from caem.pipeline import CAEMPipeline

logger = logging.getLogger("caem.ablation")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """One variant's evaluation result across all benchmarks."""

    variant: str
    description: str
    mechanism_tag: str
    per_benchmark_axes: Dict[str, CESAxes]
    aggregate_axes: CESAxes
    per_benchmark_meta: Dict[str, Dict[str, Any]]   # raw harness meta pass-through
    runtime_seconds: float
    cycle: int

    def to_json_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict (CESAxes dataclasses unwrapped)."""
        return {
            "variant": self.variant,
            "description": self.description,
            "mechanism_tag": self.mechanism_tag,
            "cycle": self.cycle,
            "runtime_seconds": round(self.runtime_seconds, 2),
            "aggregate_axes": self.aggregate_axes.as_dict(),
            "per_benchmark_axes": {
                bm: ax.as_dict() for bm, ax in self.per_benchmark_axes.items()
            },
            "per_benchmark_meta": self.per_benchmark_meta,
        }


# ---------------------------------------------------------------------------
# Pipeline-level flag honouring (monkey patches stage methods)
# ---------------------------------------------------------------------------

def _apply_pipeline_flags(
    pipeline: "CAEMPipeline",
    variant: AblationVariant,
) -> List[Any]:
    """Patch the pipeline to honour the variant's skip flags.

    Returns a list of (obj, attr, original) triples so the caller can
    restore originals with ``_restore_pipeline_flags`` when the variant
    run completes. This keeps the pipeline reusable across variants in
    one process.
    """
    backups: List[Any] = []

    # skip_verifier: bypass Stage 5. The pipeline's _maybe_store uses
    # vout.decision == "STORE" to gate; replace _maybe_store with a
    # version that stores unconditionally using pre-routing confidence.
    if variant.skip_verifier:
        original_maybe_store = pipeline._maybe_store

        def _maybe_store_bypass(self, *args, **kwargs):
            # Force decision=="STORE" by substituting a trivial verifier
            # output, then defer to the original logic. This keeps the
            # audit trail (entry metadata) populated without running NLI.
            # Callers pass (query, answer, u_pre, tier, ...) -- we don't
            # rewrite those; we only short-circuit the gate.
            if "force_store" in original_maybe_store.__code__.co_varnames:
                return original_maybe_store(*args, force_store=True, **kwargs)  # type: ignore[call-arg]
            return original_maybe_store(*args, **kwargs)

        # Fallback: if the pipeline doesn't expose a force_store knob,
        # neutralise the verifier by patching .verify to always return a
        # STORE decision with u_stored taken from pre-routing.
        if "force_store" not in getattr(
            original_maybe_store, "__code__", type("x", (), {"co_varnames": ()})()
        ).co_varnames:  # type: ignore[attr-defined]
            from caem.verification.verifier import UnifiedVerifierOutput
            original_verify = pipeline.verifier.verify

            def _verify_stub(question, answer, *a, **kw):
                return UnifiedVerifierOutput(
                    u_token=0.5, u_dropout=0.5, u_internal=0.5,
                    s_avg=1.0, h_norm=0.0,
                    p_entail=1.0, p_ground_max=1.0,
                    p_ground_mean=1.0, p_ground_atomic=1.0,
                    p_contra=0.0, u_stored=1.0,
                    decision="STORE",
                    early_exit_triggered=False,
                    abstained=False,
                )

            backups.append(("verifier", "verify", original_verify))
            pipeline.verifier.verify = _verify_stub  # type: ignore[assignment]
        else:
            backups.append(("self", "_maybe_store", original_maybe_store))
            pipeline._maybe_store = _maybe_store_bypass.__get__(pipeline)  # type: ignore

    # skip_tier1: force the router to never dispatch to Tier 1. Done by
    # tier1_combined_threshold mutation in variants.py already; this flag
    # is a belt-and-braces no-op here.

    # skip_tier3_rag: disable Tier 3 passage retrieval by nulling the
    # PassageStore on ``pipeline.rag`` for the variant's lifetime. The
    # passage_store does NOT live directly on CAEMPipeline — it lives on
    # self.rag.passage_store (see caem/pipeline.py:273-280 construction
    # site). The prior code `pipeline.passage_store = None` created a
    # dangling attribute on the wrong object; Tier 3 continued to retrieve
    # and the `no_tier3_rag` ablation silently measured the same system as
    # `full`. Audit MAJOR-RN1 / Task #113.
    #
    # Note: UnifiedVerifier's grounding retriever closure captures the
    # original ``passage_store`` at pipeline init time (pipeline.py:244-
    # 250), so nulling rag.passage_store removes Tier 3 retrieval WITHOUT
    # breaking the verifier's grounding signals — which is the correct
    # semantic for `no_tier3_rag` (ablates RAG, not verification).
    if variant.skip_tier3_rag:
        backups.append(("rag", "passage_store", pipeline.rag.passage_store))
        pipeline.rag.passage_store = None  # type: ignore[assignment]
        # Tier-3 RAG sees a null passage_store -> search() raises, the
        # except-Exception guard in rag._retrieve returns [] empty
        # passages -> RAG degrades to zero-shot generation.

    return backups


def _restore_pipeline_flags(
    pipeline: "CAEMPipeline",
    backups: Sequence[Any],
) -> None:
    """Undo ``_apply_pipeline_flags``."""
    for entry in backups:
        obj_key, attr, original = entry
        target = pipeline if obj_key == "self" else getattr(pipeline, obj_key)
        setattr(target, attr, original)


# ---------------------------------------------------------------------------
# Sample extraction -- harness JSON -> (em, u_stored) flat lists
# ---------------------------------------------------------------------------

def _extract_em_and_u_from_samples(
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Any]]:
    """Pull per-sample em (0/1) and u_stored (float or None) from harness rows.

    The harness writes one row per eval query with keys matching the
    schema in ``eval.harness``. u_stored may be null on Tier-1 hits
    that don't surface a composite; that's encoded as None in the
    returned list (scoring.py handles the filtering).
    """
    em: List[float] = []
    u: List[Optional[float]] = []
    for s in samples:
        # em key may live at top level or inside a "metrics" sub-dict
        e_val = s.get("em", s.get("metrics", {}).get("em"))
        if e_val is None:
            e_val = float("nan")
        em.append(float(e_val))
        u_val = s.get("u_stored")
        if u_val is None:
            # Tier-1 passthrough: look for a tier hint; skip only when
            # the field is explicitly absent rather than assuming NaN
            u.append(None)
        else:
            try:
                u.append(float(u_val))
            except (TypeError, ValueError):
                u.append(None)
    return {"em": em, "u_stored": u}


# ---------------------------------------------------------------------------
# Main runner entry point
# ---------------------------------------------------------------------------

def run_variant(
    variant: AblationVariant,
    pipeline: "CAEMPipeline",
    eval_samples: Mapping[str, Sequence[Mapping[str, Any]]],
    harness,
    *,
    cycle: int = 0,
    baseline_mmlu: Optional[float] = None,
    cycle_mmlu: Optional[float] = None,
    verifier_balanced_accuracy: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> AblationResult:
    """Evaluate one ablation variant against a fixed eval set.

    The caller is responsible for having already applied the variant's
    config mutation to the pipeline's config (and, if fine-tuning is
    enabled for this variant, having trained weights accordingly).
    This function only:

      - Installs pipeline-flag monkey patches (e.g. skip_verifier).
      - Runs ``harness.run_all`` with store_to_memory=False (pure eval).
      - Computes per-benchmark CES axes and the cross-benchmark aggregate.
      - Restores the pipeline to its pre-variant state.

    Parameters
    ----------
    variant : AblationVariant
        Which variant is being evaluated (used for flag handling + labeling).
    pipeline : CAEMPipeline
        Built by caller with variant.apply(base_cfg) already applied.
    eval_samples : dict[str, list[dict]]
        Same shape ``EvalHarness.run_all`` consumes.
    harness : EvalHarness
        Pre-constructed against ``pipeline``.
    cycle : int
        Which cycle's weights/memory are loaded. Logged + written to output.
    baseline_mmlu, cycle_mmlu : float
        Scalars for RET. See scoring.py for semantics.
    verifier_balanced_accuracy : float
        Optional VER axis scalar; defaults to 0.5 placeholder.
    output_dir : Path
        If given, writes ``<variant>_cycle<n>.json`` to this directory.
    """
    logger.info(
        "Ablation variant '%s' (mechanism=%s): starting cycle=%d",
        variant.name, variant.mechanism_tag, cycle,
    )
    t0 = time.time()

    backups = _apply_pipeline_flags(pipeline, variant)
    try:
        # EvalHarness.run_all returns {bm: meta_dict}; per-sample rows
        # get written to <bm>_cycle{c}.json by the harness. We read them
        # back to compute CES axes.
        meta_by_bm = harness.run_all(
            eval_samples, cycle=cycle, store_to_memory=False,
        )
    finally:
        _restore_pipeline_flags(pipeline, backups)

    per_bm_axes: Dict[str, CESAxes] = {}
    per_bm_meta: Dict[str, Dict[str, Any]] = {}

    harness_output_dir = Path(getattr(harness, "output_dir", "outputs/eval"))
    for bm, meta in meta_by_bm.items():
        per_bm_meta[bm] = dict(meta)  # shallow copy so downstream edits are safe
        # Locate per-sample file written by the harness
        per_sample_path = harness_output_dir / f"{bm}_cycle{cycle}.json"
        em_list: List[float] = []
        u_list: List[Optional[float]] = []
        # samples_field carries the full per-sample dicts (12 verifier fields
        # each) — passed to ces_axes_from_cycle so the EPI axis uses the
        # CHM path (matches the headline reporting.build_table_headline).
        # Empty list triggers _epi_axis's legacy confabulation_rate fallback,
        # which is the right behaviour for baselines with no verifier output.
        samples_field: List[Mapping[str, Any]] = []
        if per_sample_path.exists():
            try:
                with open(per_sample_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                samples_field = data.get("samples", [])
                extracted = _extract_em_and_u_from_samples(samples_field)
                em_list = extracted["em"]
                u_list = extracted["u_stored"]
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not read per-sample file %s (%s); falling back "
                    "to meta-level scalars for CES.", per_sample_path, exc,
                )

        if not em_list:
            # Previously this branch synthesised a flat list by broadcasting
            # the meta scalar (em-aggregate and mean u_stored) N times,
            # which destroys per-sample variance — EPI and CAL axes then
            # collapse to constants and CES is artificially inflated. The
            # ablation table silently downgrades to the meta-only system
            # and the Ch5 ranking between variants becomes untrustworthy.
            # Audit MAJOR-RN2 / Task #115.
            #
            # Raise loudly instead so the operator sees the missing
            # per-sample artefact before the ablation table is built.
            # The message names the file so rerunning the failed harness
            # slice is mechanical.
            raise RuntimeError(
                f"Ablation runner: per-sample file not available for "
                f"variant='{variant.name}', benchmark='{bm}', "
                f"cycle={cycle} (looked for {per_sample_path}). "
                f"CES axes require per-sample em/u_stored lists — "
                f"broadcasting the meta scalar would destroy EPI/CAL "
                f"variance and inflate CES. Fix: rerun the harness step "
                f"for this variant, or check harness.output_dir wiring."
            )

        axes = ces_axes_from_cycle(
            em_flags=em_list,
            u_stored=u_list,
            cycle_mmlu=cycle_mmlu,
            baseline_mmlu=baseline_mmlu,
            verifier_balanced_accuracy=verifier_balanced_accuracy,
            requires_baseline_only=variant.requires_baseline_only,
            samples=list(samples_field),
        )
        per_bm_axes[bm] = axes

    agg = aggregate_axes(per_bm_axes.values())

    runtime = time.time() - t0
    result = AblationResult(
        variant=variant.name,
        description=variant.description,
        mechanism_tag=variant.mechanism_tag,
        per_benchmark_axes=per_bm_axes,
        aggregate_axes=agg,
        per_benchmark_meta=per_bm_meta,
        runtime_seconds=runtime,
        cycle=cycle,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{variant.name}_cycle{cycle}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.to_json_dict(), f, indent=2)
        logger.info(
            "Ablation '%s' done in %.1fs | CES=%.4f | -> %s",
            variant.name, runtime, agg.ces, out_path,
        )
    else:
        logger.info(
            "Ablation '%s' done in %.1fs | CES=%.4f",
            variant.name, runtime, agg.ces,
        )

    return result
